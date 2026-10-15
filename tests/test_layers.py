"""Unit tests for density-matrix and Quantum-Geometric Born layers."""

from __future__ import annotations

import torch

from qdlblocks import (
    ComplexLinear,
    ComplexSuperpositionAttention,
    DepolarizingNoise,
    IsometricHilbertLift,
    KrausChannel,
    POVMMeasurement,
    ProjectivePOVMHead,
    PureQuantumAttention,
    QuantumBornBlock,
    QuantumBornMoE,
    SPAMNoise,
    SparseKrausMoE,
    SubspaceProjectorBornAttention,
    UnitaryRoPE,
    is_valid_density_matrix,
    l2_normalize,
    state_to_density,
)


def test_density_matrix_layers_and_gradients() -> None:
    torch.manual_seed(10)
    x = torch.randn(4, 32, requires_grad=True)

    attn = PureQuantumAttention(
        in_features=32,
        hilbert_dim=12,
        num_heads=4,
        purification_placement="post_fusion",
    )
    kraus = KrausChannel(in_hilbert_dim=12, out_hilbert_dim=12, num_kraus=3, residual=True)
    depol = DepolarizingNoise(p=0.05)
    depol.train()
    povm = POVMMeasurement(hilbert_dim=12, num_classes=5)

    rho1 = attn(x)
    assert is_valid_density_matrix(rho1)[0]

    rho2 = kraus(rho1)
    assert is_valid_density_matrix(rho2)[0]

    rho3 = depol(rho2)
    assert is_valid_density_matrix(rho3)[0]

    logits = povm(rho3)
    assert logits.shape == (4, 5)

    loss = logits.sum()
    loss.backward()
    assert x.grad is not None
    assert attn.W_q_r.grad is not None
    assert kraus.A_r.grad is not None
    assert povm.L_r.grad is not None


def test_complex_superposition_pooling_modes() -> None:
    torch.manual_seed(11)
    tokens = torch.randn(3, 6, 16)

    for mode in ("coherent", "incoherent", "none"):
        layer = ComplexSuperpositionAttention(
            in_features=16,
            hilbert_dim=8,
            num_heads=2,
            pooling_mode=mode,  # type: ignore[arg-type]
            purification_placement="post_fusion",
        )
        rho = layer(tokens)
        if mode == "none":
            assert rho.shape == (3, 6, 8, 8)
        else:
            assert rho.shape == (3, 8, 8)
        assert is_valid_density_matrix(rho)[0]


def test_complex_linear_and_spam_noise() -> None:
    layer = ComplexLinear(in_features=10, out_features=6, bias=True)
    spam = SPAMNoise(std=0.05, clamp_range=(-1.0, 1.0))
    spam.train()
    x = torch.zeros(4, 10)
    x_noisy = spam(x)
    assert not torch.allclose(x, x_noisy)
    out = layer(x_noisy)
    assert out.shape == (4, 6)
    assert out.is_complex()


def test_born_attention_and_block_variants() -> None:
    torch.manual_seed(12)
    idx = torch.randint(0, 100, (2, 16))
    lift = IsometricHilbertLift(vocab_size=100, seed_dim=16, embed_dim=64)
    rope = UnitaryRoPE(head_dim=16, max_seq_len=32)

    psi_0 = lift(idx)
    assert psi_0.shape == (2, 16, 64)
    cos, sin = rope.get_cos_sin(16)

    # Test both real positive-cone and complex phase-invariant Born attention
    for complex_phase in (False, True):
        block = QuantumBornBlock(
            embed_dim=64,
            num_heads=4,
            block_size=32,
            hidden_dim=128,
            learnable_temperature=True,
            value_residual_mixing=True,
            input_highway_mixing=True,
            complex_phase_attention=complex_phase,
            riemannian_centering=True,
        )
        psi_out = block(psi_0, cos=cos, sin=sin, psi_0=psi_0)
        assert psi_out.shape == (2, 16, 64)
        norms = torch.norm(psi_out, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    head = ProjectivePOVMHead(vocab_size=100, embed_dim=64, learnable_temperature=True)
    logits = head(psi_out)
    assert logits.shape == (2, 16, 100)
    logits.sum().backward()
    assert head.inv_temp_param is not None and head.inv_temp_param.grad is not None


def test_causal_masking_in_born_attention() -> None:
    torch.manual_seed(13)
    attn = SubspaceProjectorBornAttention(embed_dim=32, num_heads=4, block_size=16, is_causal=True)
    attn.record_attention = True
    x = torch.randn(2, 8, 32)
    _ = attn(x)
    assert attn.last_born_weights is not None
    # Strictly upper triangular entries (j > i) must be exact 0.0
    upper = torch.triu(attn.last_born_weights, diagonal=1)
    assert torch.all(upper == 0.0)


def test_quantum_born_moe_and_sparse_kraus_moe() -> None:
    torch.manual_seed(14)
    # 1. Pure-state QuantumBornMoE & MoE-enabled QuantumBornBlock
    x = l2_normalize(torch.randn(3, 12, 64, requires_grad=True), dim=-1)
    moe = QuantumBornMoE(embed_dim=64, expert_hidden_dim=128, num_experts=6, top_k=2)
    y = moe(x)
    assert y.shape == (3, 12, 64)
    norms = torch.norm(y, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert moe.last_routing_stats["positive_cone_zero_fraction"] > 0.0
    y.sum().backward()
    assert moe.router_proj.weight.grad is not None

    block_moe = QuantumBornBlock(
        embed_dim=64, num_heads=4, block_size=16, hidden_dim=96, num_experts=4, top_k_experts=2
    )
    out_block = block_moe(x.detach())
    assert out_block.shape == (3, 12, 64)

    # 2. Density-matrix SparseKrausMoE
    psi = torch.complex(torch.randn(4, 10), torch.randn(4, 10))
    rho = state_to_density(psi)
    sparse_kraus = SparseKrausMoE(in_dim=10, out_dim=8, num_experts=6, top_k=2)
    rho_out = sparse_kraus(rho)
    assert rho_out.shape == (4, 8, 8)
    assert is_valid_density_matrix(rho_out)[0]

