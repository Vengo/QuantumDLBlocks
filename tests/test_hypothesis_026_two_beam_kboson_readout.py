"""Unit tests for Experiment Plan 026: 100% Quantum Two-Beam Mixed-State Density Readout, Non-Isometric Kraus Semantic Bath, & k-Copy Bloch Fidelity."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from qdlblocks.benchmarks import parse_cli_args_to_card_sections
from qdlblocks.layers.direct_chunked_ce import (
    compute_two_beam_kboson_logits,
    direct_chunked_two_beam_kboson_ce,
)
from qdlblocks.layers.dynamic_povm_purity import ContextualBlochPurityGate
from qdlblocks.layers.hierarchical_qubit import (
    Hierarchical16QubitEmbedder,
    Hierarchical16QubitPOVMReadout,
    initialize_hierarchical_16qubit_with_ppmi_svd,
)
from qdlblocks.layers.attention import l2_normalize
from qdlblocks.layers.overcomplete_two_beam import OvercompleteTwoBeamInterference
from qdlblocks.models.born_gpt import QuantumBornGPT
from qdlblocks.optim.muon import polar_snap_weights


def test_direct_chunked_two_beam_kboson_ce_forward_backward_parity_and_synonym_amplification() -> None:
    """Verify exact forward/backward parity of direct_chunked_two_beam_kboson_ce and 1/(1+c_v) amplification."""
    torch.manual_seed(2026)
    n_tokens = 96
    vocab_padded = 256
    valid_vocab = 240
    embed_dim = 64
    diffuse_rank = 16

    psi_sharp_raw = F.normalize(torch.randn(n_tokens, embed_dim, dtype=torch.float32), p=2, dim=-1)
    psi_diff_raw = torch.randn(n_tokens, diffuse_rank, dtype=torch.float32) * 0.35
    m_sharp_raw = F.normalize(torch.randn(vocab_padded, embed_dim, dtype=torch.float32), p=2, dim=-1)
    m_diff_raw = torch.randn(vocab_padded, diffuse_rank, dtype=torch.float32) * 0.35
    k_raw = torch.tensor(34.0, dtype=torch.float32)
    b_raw = torch.randn(vocab_padded, dtype=torch.float32) * 0.2
    beta_t = torch.linspace(24.0, 42.0, n_tokens, dtype=torch.float32)
    targets = torch.randint(0, valid_vocab, (n_tokens,), dtype=torch.long)
    targets[7] = -100  # verify ignore_index handling

    # Chunked autograd kernel
    psi_s1 = psi_sharp_raw.clone().requires_grad_(True)
    psi_d1 = psi_diff_raw.clone().requires_grad_(True)
    m_s1 = m_sharp_raw.clone().requires_grad_(True)
    m_d1 = m_diff_raw.clone().requires_grad_(True)
    beta_1 = beta_t.clone().requires_grad_(True)
    k_1 = k_raw.clone().requires_grad_(True)
    lam_1 = torch.tensor(0.18, dtype=torch.float32, requires_grad=True)
    b_1 = b_raw.clone().requires_grad_(True)

    loss_chunked = direct_chunked_two_beam_kboson_ce(
        psi_s1,
        psi_d1,
        m_s1,
        m_d1,
        beta_1,
        k_1,
        lam_1,
        targets,
        log_prior=b_1,
        chunk_size=24,
        ignore_index=-100,
        valid_vocab_size=valid_vocab,
    )
    loss_chunked.backward()

    # Materialized PyTorch autograd reference
    psi_s2 = psi_sharp_raw.clone().requires_grad_(True)
    psi_d2 = psi_diff_raw.clone().requires_grad_(True)
    m_s2 = m_sharp_raw.clone().requires_grad_(True)
    m_d2 = m_diff_raw.clone().requires_grad_(True)
    beta_2 = beta_t.clone().requires_grad_(True)
    k_2 = k_raw.clone().requires_grad_(True)
    lam_2 = torch.tensor(0.18, dtype=torch.float32, requires_grad=True)
    b_2 = b_raw.clone().requires_grad_(True)

    logits_ref = compute_two_beam_kboson_logits(
        psi_s2,
        psi_d2,
        m_s2,
        m_d2,
        beta_2,
        k_2,
        lam_2,
        log_prior=b_2,
    )
    loss_ref = F.cross_entropy(logits_ref[:, :valid_vocab], targets, ignore_index=-100)
    loss_ref.backward()

    assert abs(float(loss_chunked.item() - loss_ref.item())) < 1e-5
    assert psi_s1.grad is not None and psi_s2.grad is not None
    assert psi_d1.grad is not None and psi_d2.grad is not None
    assert m_s1.grad is not None and m_s2.grad is not None
    assert m_d1.grad is not None and m_d2.grad is not None
    assert beta_1.grad is not None and beta_2.grad is not None
    assert k_1.grad is not None and k_2.grad is not None
    assert lam_1.grad is not None and lam_2.grad is not None
    assert b_1.grad is not None and b_2.grad is not None

    assert float((psi_s1.grad - psi_s2.grad).abs().max().item()) < 1e-5
    assert float((psi_d1.grad - psi_d2.grad).abs().max().item()) < 1e-5
    assert float((m_s1.grad - m_s2.grad).abs().max().item()) < 1e-5
    assert float((m_d1.grad - m_d2.grad).abs().max().item()) < 1e-5
    assert float((beta_1.grad - beta_2.grad).abs().max().item()) < 1e-5
    assert float((k_1.grad - k_2.grad).abs().max().item()) < 1e-4
    assert float((lam_1.grad - lam_2.grad).abs().max().item()) < 1e-5
    assert float((b_1.grad - b_2.grad).abs().max().item()) < 1e-5

    # Verify the k/(1 + c_v) orthogonal synonym gradient amplification law:
    # For pure Bloch fidelity F(c_v) = (1 + c_v)/2, E(c_v) = k * ln((1 + c_v)/2),
    # dE/dc_v = k / (1 + c_v), so at c_v = 0.0 vs c_v = 0.60 the ratio is exactly 1.60x.
    c_ortho = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    c_top1 = torch.tensor(0.60, dtype=torch.float64, requires_grad=True)
    k_val = 34.0
    e_ortho = k_val * torch.log(0.5 * (1.0 + c_ortho))
    e_top1 = k_val * torch.log(0.5 * (1.0 + c_top1))
    e_ortho.backward()
    e_top1.backward()
    assert c_ortho.grad is not None and c_top1.grad is not None
    ratio = float(c_ortho.grad.item() / c_top1.grad.item())
    assert abs(ratio - 1.60) < 1e-6


def test_kraus_diffuse_bath_avoids_low_rank_normalization_trap_and_preserves_unit_sphere() -> None:
    """Verify Argon Fix 1 & 2: get_two_beam_kboson_factors does NOT L2-normalize inside R^r and keeps sharp vectors on S^{d-1}."""
    torch.manual_seed(2026)
    vocab_size = 512
    embed_dim = 512
    diffuse_rank = 16
    batch_size = 4
    seq_len = 32

    embedder = Hierarchical16QubitEmbedder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_registers=16,
        register_dim=16,
        num_higher_states=64,
        num_layers=2,
        use_ortho_decouple=True,
    )
    readout = Hierarchical16QubitPOVMReadout(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_registers=16,
        register_dim=16,
        num_higher_states=64,
        use_ortho_decouple=True,
        diffuse_rank=diffuse_rank,
    )

    base_povm_head = F.normalize(torch.randn(vocab_size, embed_dim), p=2, dim=-1)
    psi_sharp = F.normalize(torch.randn(batch_size, seq_len, embed_dim), p=2, dim=-1)
    psi_beam2 = F.normalize(torch.randn(batch_size, seq_len, embed_dim), p=2, dim=-1)

    psi_sharp_norm, psi_diff_r, m_sharp_norm, m_diff_r, k_copies, lambda_diff = (
        readout.get_two_beam_kboson_factors(
            embedder,
            psi_sharp,
            psi_beam2,
            base_povm_head=base_povm_head,
        )
    )

    # Argon Fix 2: psi_sharp_norm and m_sharp_norm must remain strictly unit-norm on S^{d-1}
    sharp_state_norms = psi_sharp_norm.norm(p=2, dim=-1)
    sharp_povm_norms = m_sharp_norm.norm(p=2, dim=-1)
    assert torch.allclose(sharp_state_norms, torch.ones_like(sharp_state_norms), atol=1e-5)
    assert torch.allclose(sharp_povm_norms, torch.ones_like(sharp_povm_norms), atol=1e-5)

    # Argon Fix 1: psi_diff_r and m_diff_r are NOT unit-normalized in R^r, avoiding the 1/sqrt(r)=0.25 floor
    c_diff = psi_diff_r.reshape(-1, diffuse_rank) @ m_diff_r.t()
    mean_abs_random_overlap = float(c_diff.abs().mean().item())
    # In R^512, 1/sqrt(512) ~ 0.044; without L2 normalization in R^16, random overlap is ~0.03-0.05 (well below 1/sqrt(16)=0.25)
    assert mean_abs_random_overlap < 0.065

    # Effective rank of diffuse channel is bounded in [3.0, 16.0]
    svals = torch.linalg.svdvals(psi_diff_r.reshape(-1, diffuse_rank).float())
    p_s = svals / svals.sum().clamp_min(1e-12)
    eff_rank = float(torch.exp(-(p_s * torch.log(p_s.clamp_min(1e-12))).sum()).item())
    assert 3.0 <= eff_rank <= float(diffuse_rank) + 1e-4
    assert 28.0 <= float(k_copies.item()) <= 48.0
    assert 0.0 < lambda_diff < 0.40


def test_ppmi_svd_initializes_kraus_diffuse_subspace_to_top_r_singular_vectors() -> None:
    """Verify Argon Fix 3: initialize_hierarchical_16qubit_with_ppmi_svd initializes kraus_u_diff and kraus_v_diff Orthonormally."""
    torch.manual_seed(77)
    vocab_size = 128
    embed_dim = 64
    diffuse_rank = 8

    model = QuantumBornGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seed_dim=32,
        num_heads=4,
        num_layers=2,
        block_size=16,
        readout_mode="asymmetric",
        use_ortho_decouple=True,
        use_hierarchical_16qubit=True,
        qubit_registers=16,
        qubit_amplitudes=2,
        num_higher_states=16,
        use_two_beam_kboson_readout=True,
        diffuse_rank=diffuse_rank,
    )
    u_vocab = F.normalize(torch.randn(vocab_size, embed_dim), p=2, dim=-1)
    initialize_hierarchical_16qubit_with_ppmi_svd(model, ppmi_embeddings=u_vocab)

    readout = model.hierarchical_povm_readout
    assert readout is not None
    assert torch.allclose(readout.kraus_u_diff, readout.kraus_v_diff, atol=1e-6)
    gram = readout.kraus_u_diff.t() @ readout.kraus_u_diff
    eye_r = torch.eye(diffuse_rank, dtype=gram.dtype)
    assert torch.allclose(gram, eye_r, atol=1e-5)


def test_overcomplete_two_beam_second_beam_extraction_and_purity_gate_cap() -> None:
    """Verify OvercompleteTwoBeamInterference(return_second_beam=True) and ContextualBlochPurityGate max_beta=45.0."""
    torch.manual_seed(101)
    layer = OvercompleteTwoBeamInterference(embed_dim=64, frame_expansion=2.0)
    psi = l2_normalize(torch.randn(2, 16, 64), dim=-1)

    out_single = layer(psi, return_second_beam=False)
    out_main, out_beam2 = layer(psi, return_second_beam=True)

    assert torch.equal(out_single, out_main)
    assert layer.last_beam2 is not None
    assert torch.allclose(out_beam2.norm(p=2, dim=-1), torch.ones(2, 16), atol=1e-5)

    gate_default = ContextualBlochPurityGate(embed_dim=64)
    gate_kboson = ContextualBlochPurityGate(embed_dim=64, use_two_beam_kboson_readout=True)
    assert gate_default.max_beta == 160.0
    assert gate_kboson.max_beta == 45.0


def test_polar_snap_weights_excludes_value_embeds_and_kraus_parameters() -> None:
    """Verify polar_snap_weights does not column-orthogonalize value_embeds, ngram_embed, or kraus_u_diff."""
    torch.manual_seed(2026)
    model = QuantumBornGPT(
        vocab_size=256,
        embed_dim=64,
        seed_dim=32,
        num_heads=4,
        num_layers=2,
        block_size=16,
        readout_mode="asymmetric",
        use_overcomplete_two_beam=True,
        use_ortho_decouple=True,
        use_hierarchical_16qubit=True,
        qubit_registers=16,
        qubit_amplitudes=2,
        num_higher_states=16,
        use_two_beam_kboson_readout=True,
        diffuse_rank=8,
        num_value_embeddings=2,
        use_ngram_hash=True,
        ngram_vocab_size=512,
    )
    assert len(model.value_embeds) == 2
    assert model.ngram_embed is not None
    assert model.hierarchical_povm_readout is not None

    # Set non-zero weights on value_embeds and ngram_embed and verify polar_snap_weights leaves them untouched
    with torch.no_grad():
        nn.init.normal_(model.value_embeds[0].weight, std=0.5)
        nn.init.normal_(model.ngram_embed.table.weight, std=0.5)
        ve_before = model.value_embeds[0].weight.clone()
        ng_before = model.ngram_embed.table.weight.clone()
        ku_before = model.hierarchical_povm_readout.kraus_u_diff.clone()

    polar_snap_weights(model, polar_method="polar_express")

    assert torch.equal(model.value_embeds[0].weight, ve_before)
    assert torch.equal(model.ngram_embed.table.weight, ng_before)
    assert torch.equal(model.hierarchical_povm_readout.kraus_u_diff, ku_before)


def test_quantum_born_gpt_two_beam_kboson_forward_backward_and_causality() -> None:
    """Verify end-to-end forward/backward, strict causality (0.0 prefix leakage), and CLI parsing for Two-Beam k-Boson Readout."""
    torch.manual_seed(2026)
    vocab_size = 128
    embed_dim = 64
    seq_len = 16

    for use_ckpt in (False, True):
        model = QuantumBornGPT(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            seed_dim=32,
            num_heads=4,
            num_layers=2,
            block_size=seq_len,
            readout_mode="asymmetric",
            use_overcomplete_two_beam=True,
            frame_expansion=2.0,
            use_purified_attn=True,
            true_layer0_v_residual=True,
            use_gated_spherical_residual=True,
            use_dynamic_povm_beta=True,
            use_povm_log_prior=True,
            use_ortho_decouple=True,
            use_hierarchical_16qubit=True,
            qubit_registers=16,
            qubit_amplitudes=2,
            num_higher_states=16,
            use_block_hierarchy_bridge=True,
            bridge_causal_memory=False,
            use_two_beam_kboson=True,
            diffuse_bath_rank=8,
            use_activation_checkpointing=use_ckpt,
        )
        assert model.use_two_beam_kboson_readout is True
        assert model.purity_gate is not None
        assert model.purity_gate.max_beta == 45.0

        x = torch.randint(0, vocab_size, (2, seq_len), dtype=torch.long)
        y = torch.randint(0, vocab_size, (2, seq_len), dtype=torch.long)

        _, loss = model(x, y)
        assert loss is not None and torch.isfinite(loss)
        loss.backward()

        assert model.hierarchical_povm_readout is not None
        assert model.hierarchical_povm_readout.kraus_u_diff.grad is not None
        assert model.hierarchical_povm_readout.kraus_v_diff.grad is not None
        assert model.hierarchical_povm_readout.kboson_copies.grad is not None
        assert model.hierarchical_povm_readout.diffuse_mix_logit.grad is not None

        # Verify strict autoregressive causality (0.0 prefix leakage)
        model.eval()
        x_pert = x.clone()
        x_pert[:, 8:] = (x_pert[:, 8:] + 5) % vocab_size
        with torch.no_grad():
            logits_a, _ = model(x)
            logits_b, _ = model(x_pert)
        prefix_leakage = float((logits_a[:, :8, :] - logits_b[:, :8, :]).abs().max().item())
        assert prefix_leakage < 1e-5

    arch, _, _ = parse_cli_args_to_card_sections(
        "--architecture quantum_born --hierarchical-16qubit --two-beam-kboson-readout --diffuse-rank 16"
    )
    assert arch["two_beam_kboson_readout"] is True
    assert arch["diffuse_rank"] == 16
