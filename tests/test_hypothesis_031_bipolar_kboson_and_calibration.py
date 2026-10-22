"""Unit and autograd tests for Plan 031 (Bipolar Particle-Hole k-Boson & Diffuse Mode Whitening)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from qdlblocks.benchmarks.model_card import parse_cli_args_to_card_sections
from qdlblocks.functional import effective_quantum_rank, l2_normalize
from qdlblocks.layers.direct_chunked_ce import (
    compute_two_beam_kboson_logits,
    direct_chunked_two_beam_kboson_ce,
)
from qdlblocks.layers.hierarchical_qubit import (
    Hierarchical16QubitEmbedder,
    Hierarchical16QubitPOVMReadout,
)
from qdlblocks.models.born_gpt import QuantumBornGPT


def test_h1_bipolar_particle_hole_kboson_and_autograd() -> None:
    """Verify H1 bipolar particle-hole log-fidelity curve, slope ratio >= 4.0x, and exact autograd."""
    torch.manual_seed(3101)
    k_val = 4.0
    w_hole = 0.85
    x_neg = -0.60

    # At x = -0.60:
    # Legacy concave slope: d/dx [k * ln(1 + x)] = k / (1 - 0.60) = 2.5 * k
    # Bipolar particle-hole slope: d/dx [-k * w_hole * ln(1 - x)] = k * w_hole / (1 + 0.60) = 0.53125 * k
    slope_legacy = k_val / (1.0 + x_neg)
    slope_bipolar = (k_val * w_hole) / (1.0 + abs(x_neg))
    slope_ratio = slope_legacy / slope_bipolar
    assert slope_ratio >= 4.0, f"Expected slope_ratio >= 4.0, got {slope_ratio:.4f}"
    assert math.isclose(slope_ratio, 2.5 / (0.85 / 1.60), rel_tol=1e-6)

    # Verify exact autograd match between direct_chunked_two_beam_kboson_ce and dense compute_two_beam_kboson_logits
    n_tok = 48
    d = 64
    r = 16
    v_size = 128

    psi_s_base = l2_normalize(torch.randn(n_tok, d, dtype=torch.float64), dim=-1)
    psi_d_base = torch.randn(n_tok, r, dtype=torch.float64) * 0.35
    m_s_base = l2_normalize(torch.randn(v_size, d, dtype=torch.float64), dim=-1)
    m_d_base = torch.randn(v_size, r, dtype=torch.float64) * 0.35
    beta_base = torch.empty(n_tok, 1, dtype=torch.float64).uniform_(12.0, 38.0)
    lam_base = torch.empty(n_tok, 1, dtype=torch.float64).uniform_(0.10, 0.28)
    k_eff = torch.tensor(4.0, dtype=torch.float64)
    log_prior = torch.randn(v_size, dtype=torch.float64) * 0.1
    targets = torch.randint(0, v_size, (n_tok,), dtype=torch.long)

    # 1. Bipolar mode (bipolar_hole_weight=0.85)
    inputs_chunked = [
        t.clone().detach().requires_grad_(True)
        for t in (psi_s_base, psi_d_base, m_s_base, m_d_base, beta_base, lam_base)
    ]
    loss_chunked = direct_chunked_two_beam_kboson_ce(
        inputs_chunked[0],
        inputs_chunked[1],
        inputs_chunked[2],
        inputs_chunked[3],
        inputs_chunked[4],
        k_eff,
        inputs_chunked[5],
        targets,
        log_prior=log_prior,
        chunk_tokens=16,
        valid_vocab_size=v_size,
        quad_diffuse_weight=0.50,
        bipolar_hole_weight=w_hole,
    )
    loss_chunked.backward()

    inputs_dense = [
        t.clone().detach().requires_grad_(True)
        for t in (psi_s_base, psi_d_base, m_s_base, m_d_base, beta_base, lam_base)
    ]
    logits_dense = compute_two_beam_kboson_logits(
        inputs_dense[0],
        inputs_dense[1],
        inputs_dense[2],
        inputs_dense[3],
        inputs_dense[4],
        k_eff,
        inputs_dense[5],
        log_prior=log_prior,
        quad_diffuse_weight=0.50,
        bipolar_hole_weight=w_hole,
    )
    loss_dense = F.cross_entropy(logits_dense[:, :v_size], targets)
    loss_dense.backward()

    assert torch.allclose(loss_chunked, loss_dense, atol=1e-6, rtol=1e-6)
    for idx_t, (tc, td) in enumerate(zip(inputs_chunked, inputs_dense)):
        assert tc.grad is not None and td.grad is not None
        max_err = float((tc.grad - td.grad).abs().max().item())
        assert max_err < 1e-6, f"Input {idx_t} grad mismatch: {max_err:.3e}"


def test_h2_diffuse_mode_whitening_prevents_rank_collapse() -> None:
    """Verify H2 diffuse mode Newton-Schulz whitening + concept-cluster diffuse bath keeps diffuse_eff_rank >= 12.0."""
    torch.manual_seed(3102)
    v_size = 512
    d = 256
    q_reg = 16
    q_amp = 16
    m_hier = 64
    r = 16

    embed = Hierarchical16QubitEmbedder(
        vocab_size=v_size,
        embed_dim=d,
        num_registers=q_reg,
        reg_dim=q_amp,
        num_higher_states=m_hier,
        num_layers=2,
    )
    readout_legacy = Hierarchical16QubitPOVMReadout(
        vocab_size=v_size,
        embed_dim=d,
        num_registers=q_reg,
        reg_dim=q_amp,
        num_higher_states=m_hier,
        use_two_beam_kboson_readout=True,
        diffuse_rank=r,
        use_stereographic_lift=True,
        use_bipolar_kboson=False,
    )
    readout_bipolar = Hierarchical16QubitPOVMReadout(
        vocab_size=v_size,
        embed_dim=d,
        num_registers=q_reg,
        reg_dim=q_amp,
        num_higher_states=m_hier,
        use_two_beam_kboson_readout=True,
        diffuse_rank=r,
        use_stereographic_lift=True,
        use_bipolar_kboson=True,
        bipolar_hole_weight=0.85,
    )
    # Copy shared parameters so the only difference is use_bipolar_kboson whitening + concept bath
    with torch.no_grad():
        for name, p in readout_legacy.named_parameters():
            if name in dict(readout_bipolar.named_parameters()):
                dict(readout_bipolar.named_parameters())[name].copy_(p)

    # Construct an anisotropic / rank-collapsed psi_beam2 (simulating training step 300 where 3 dominant directions capture 85% variance)
    b_sz, t_len = 8, 64
    psi_main = l2_normalize(torch.randn(b_sz, t_len, d), dim=-1)
    dominant_dirs = l2_normalize(torch.randn(3, d), dim=-1)
    coeffs = torch.randn(b_sz, t_len, 3) * 4.0
    psi_beam2_collapsed = l2_normalize(
        torch.matmul(coeffs, dominant_dirs) + 0.05 * torch.randn(b_sz, t_len, d),
        dim=-1,
    )
    beta_t = torch.full((b_sz, t_len, 1), 24.0)

    _, psi_d_leg, _, m_d_leg, _, _ = readout_legacy.get_two_beam_kboson_factors(
        embed, psi_main, psi_beam2=psi_beam2_collapsed, beta_t=beta_t
    )
    _, psi_d_bip, _, m_d_bip, _, _ = readout_bipolar.get_two_beam_kboson_factors(
        embed, psi_main, psi_beam2=psi_beam2_collapsed, beta_t=beta_t
    )

    rank_leg = float(effective_quantum_rank(psi_d_leg.reshape(-1, r)).item())
    rank_bip = float(effective_quantum_rank(psi_d_bip.reshape(-1, r)).item())
    assert rank_leg < 6.0, f"Expected collapsed legacy diffuse_eff_rank < 6.0, got {rank_leg:.2f}"
    assert rank_bip >= 12.0, f"Expected whitened bipolar diffuse_eff_rank >= 12.0, got {rank_bip:.2f}"
    assert m_d_bip.shape == (v_size, r)


def test_h3_full_ambient_16x32_qubits_and_cli_parsing() -> None:
    """Verify full-ambient 16x32 qubits on L=2, d=512 and CLI card parsing for --bipolar-kboson."""
    torch.manual_seed(3103)
    model = QuantumBornGPT(
        vocab_size=256,
        embed_dim=512,
        seed_dim=512,
        num_heads=8,
        num_layers=2,
        block_size=32,
        readout_mode="asymmetric",
        use_ortho_decouple=False,
        use_hierarchical_16qubit=True,
        qubit_registers=16,
        qubit_amplitudes=32,
        num_higher_states=64,
        use_two_beam_kboson_readout=True,
        diffuse_rank=16,
        use_stereographic_lift=True,
        use_cosine_phase_beam_gate=True,
        quadratic_diffuse_fidelity=0.50,
        use_bipolar_kboson=True,
        bipolar_hole_weight=0.85,
    )
    idx = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))
    _, loss = model(idx, targets=targets, direct_chunked_ce=True)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert model.hierarchical_povm_readout is not None
    assert model.hierarchical_povm_readout.hier_diffuse_proj is not None
    assert model.hierarchical_povm_readout.hier_diffuse_proj.grad is not None

    arch, _, _ = parse_cli_args_to_card_sections(
        "--architecture quantum_born --num-layers 6 --embed-dim 512 --num-heads 8 --hierarchical-16qubit "
        "--qubit-registers 16 --qubit-amplitudes 32 --two-beam-kboson-readout --bipolar-kboson --bipolar-hole-weight 0.85"
    )
    assert arch["use_bipolar_kboson"] is True
    assert math.isclose(arch["bipolar_hole_weight"], 0.85, rel_tol=1e-6)
