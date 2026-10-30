"""Unit tests for Plan 059 / Wave 44 (`MultiHeadMixedStateKroneckerGPT` & Tracks 3, 4, 6).

Tests:
1. Track 3: Carrier-orthogonal complex self-echo (`apply_carrier_orthogonal_complex_self_echo`)
   preserving `<c_ref | raw_phi^{(r,m)}> == <c_ref | op^{(r,m)}>` (`< 1e-6`), removing `+i`
   phase-rotated `H_perp` self-echo under `"ortho_complex_full"`, and scaling `H_perp` overlap by
   `(1 - lambda)` under `"ortho_complex_learned"`.
2. Track 4: Non-diluted (`1.0x`) secondary branch initialization (`H3.1` Step-0 purity `= 1.0`
   and gate ratio `= 1.0`), `R`-invariant per-factor normalization (`H3.2` across `R in {2, 3, 4}`),
   and phase-sensitive context routing (`H3.3` Step-0 uniform `1/sqrt(R)` + 4-phase differentiation
   across `{+1, -1, +i, -i}`).
3. Track 6: Multi-head superposition attention matmul count reduction (`H1.1`: `11 -> 3` `(T x T)`
   matmuls and `5` total `torch.matmul` calls), Hermitian PSD unit-trace cross-head Gram matrix and
   Stinespring reduced-state purity (`H2.1`), Schmidt spectrum parity between the `R x R` Gram
   matrix and `16 x 16` outer density matrix, `U(1)` global branch gauge invariance of mixed-state
   Born readout, and exact analytical gradients + single-pass `pow` call count on
   `MixedStateKroneckerBornChunkedNLLFn` (`H6.1`).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from qdlblocks.layers.multihead_superposition_mixed_state_attn import (
    MultiHeadSuperpositionAttnConfig,
    MultiHeadSuperpositionMixedStateAttention,
    compute_normalized_cross_head_gram,
    count_attention_matmul_calls,
)
from qdlblocks.models.calibrated_kronecker_born_gpt import (
    CalibratedKroneckerBornChunkedNLLFn,
    eval_calibrated_kronecker_born_probs,
)
from qdlblocks.models.multihead_mixed_state_kronecker_gpt import (
    DEFAULT_NON_DILUTED_SECONDARY_BRANCH_SCALE,
    LEGACY_DILUTED_SECONDARY_BRANCH_SCALE,
    MixedStateKroneckerBornChunkedNLLFn,
    MultiHeadMixedStateKroneckerConfig,
    MultiHeadMixedStateKroneckerGPT,
    apply_carrier_orthogonal_complex_self_echo,
    build_multihead_mixed_state_kronecker_optimizers,
    compute_multihead_mixed_state_kronecker_penalties,
    compute_phase_sensitive_branch_beta,
    eval_mixed_state_kronecker_born_probs,
    normalize_raw_phi_per_factor_r_invariant,
)
from qdlblocks.models.ngram_calibrated_kronecker_born_gpt import (
    Ngram900MCalibrationData,
    evaluate_modded_nanogpt_validation,
)

# ==============================================================================
# Rule 7 Named Constants for Plan 059 Unit Tests
# ==============================================================================
TEST_TOL_STRICT: float = 1e-6
TEST_TOL_MEDIUM: float = 1e-5


def test_track3_carrier_orthogonal_complex_self_echo_invariants() -> None:
    """Verify carrier preservation (`< 1e-6`), `+i` phase self-echo removal, and mode='no_self_echo' identity."""
    torch.manual_seed(5901)
    n_tok, r_num, m_num, d_f = 12, 2, 4, 16
    op_re = torch.randn(n_tok, r_num, m_num, d_f, dtype=torch.float64)
    op_im = torch.randn(n_tok, r_num, m_num, d_f, dtype=torch.float64)
    # Construct p0 whose H_perp projection is a +i phase-rotated copy of op's H_perp component, plus carrier offset
    op_perp_re = op_re - op_re.mean(dim=-1, keepdim=True)
    op_perp_im = op_im - op_im.mean(dim=-1, keepdim=True)
    p0_re = -op_perp_im + 0.75
    p0_im = op_perp_re - 0.45

    for mode in ("ortho_complex_full", "ortho_complex_learned", "carrier_ortho_complex", "no_self_echo"):
        out_re, out_im = apply_carrier_orthogonal_complex_self_echo(
            op_re, op_im, p0_re, p0_im, self_echo_mode=mode
        )
        # Carrier inner product <c_ref | raw_phi> is proportional to mean(dim=-1)
        carrier_diff = float(
            (
                (out_re.mean(dim=-1) - op_re.mean(dim=-1)).abs().max()
                + (out_im.mean(dim=-1) - op_im.mean(dim=-1)).abs().max()
            ).item()
        )
        assert carrier_diff < TEST_TOL_STRICT, f"Mode {mode} altered carrier projection: {carrier_diff}"

    # Verify mode="no_self_echo" is exact identity
    out_none_re, out_none_im = apply_carrier_orthogonal_complex_self_echo(
        op_re, op_im, p0_re, p0_im, self_echo_mode="no_self_echo"
    )
    assert torch.allclose(out_none_re, op_re, atol=TEST_TOL_STRICT)
    assert torch.allclose(out_none_im, op_im, atol=TEST_TOL_STRICT)

    # Verify ortho_complex_full completely removes the +i phase-shifted H_perp component
    out_full_re, out_full_im = apply_carrier_orthogonal_complex_self_echo(
        op_re, op_im, p0_re, p0_im, self_echo_mode="ortho_complex_full"
    )
    rem_perp_re = out_full_re - out_full_re.mean(dim=-1, keepdim=True)
    rem_perp_im = out_full_im - out_full_im.mean(dim=-1, keepdim=True)
    assert float((rem_perp_re.square() + rem_perp_im.square()).sum(dim=-1).max().item()) < TEST_TOL_STRICT

    # Verify ortho_complex_learned with u=0 (lambda=0.5) scales the H_perp component by exactly 0.5
    out_half_re, out_half_im = apply_carrier_orthogonal_complex_self_echo(
        op_re,
        op_im,
        p0_re,
        p0_im,
        self_echo_mode="ortho_complex_learned",
        echo_gate_logits=torch.zeros(r_num, m_num, dtype=torch.float64),
    )
    half_perp_re = out_half_re - out_half_re.mean(dim=-1, keepdim=True)
    half_perp_im = out_half_im - out_half_im.mean(dim=-1, keepdim=True)
    assert torch.allclose(half_perp_re, 0.5 * op_perp_re, atol=TEST_TOL_STRICT)
    assert torch.allclose(half_perp_im, 0.5 * op_perp_im, atol=TEST_TOL_STRICT)


def test_track4_h3_1_and_h3_2_non_diluted_polysemy_and_r_invariance() -> None:
    """Verify H3.1 non-diluted Step-0 branch purity = 1.0 and H3.2 per-factor R-invariance."""
    torch.manual_seed(5902)
    v_eff = 64
    cfg_nd = MultiHeadMixedStateKroneckerConfig(
        vocab_size=v_eff,
        valid_vocab_size=v_eff,
        num_meanings=4,
        num_factors=4,
        factor_dim=16,
        num_layers=1,
        secondary_branch_scale=DEFAULT_NON_DILUTED_SECONDARY_BRANCH_SCALE,
    )
    model_nd = MultiHeadMixedStateKroneckerGPT(cfg_nd)
    tokens = torch.randint(0, v_eff, (2, 12))
    tel_nd = model_nd.compute_branch_routing_telemetry(tokens)
    assert abs(float(tel_nd["step0_branch_ket_purity"]) - 1.0) < TEST_TOL_STRICT
    assert abs(float(tel_nd["secondary_to_primary_gate_ratio"]) - 1.0) < TEST_TOL_STRICT

    cfg_diluted = MultiHeadMixedStateKroneckerConfig(
        vocab_size=v_eff,
        valid_vocab_size=v_eff,
        num_meanings=4,
        num_factors=4,
        factor_dim=16,
        num_layers=1,
        secondary_branch_scale=LEGACY_DILUTED_SECONDARY_BRANCH_SCALE,
    )
    model_diluted = MultiHeadMixedStateKroneckerGPT(cfg_diluted)
    tel_dil = model_diluted.compute_branch_routing_telemetry(tokens)
    assert abs(float(tel_dil["secondary_to_primary_gate_ratio"]) - 0.22) < 1e-4

    # Check spectral cache calibration purity: non-diluted (1.0) vs diluted (0.22)
    cache = Ngram900MCalibrationData(
        unigram_counts=torch.ones(v_eff, dtype=torch.float64),
        spectral_vocab_ortho=torch.randn(v_eff, 4, 30),
        vocab_to_bi_slot=torch.zeros(v_eff, dtype=torch.int64),
        bi_slot_ortho_kets=torch.randn(v_eff + 1, 4, 30),
        tri_bucket_ortho_kets=torch.randn(cfg_nd.hashed_trigram_buckets, 4, 30),
        bi_factor_op_re_im=torch.stack([torch.eye(16).unsqueeze(0).repeat(4, 1, 1), torch.zeros(4, 16, 16)]),
        tri_factor_op_re_im=torch.stack([torch.eye(16).unsqueeze(0).repeat(4, 1, 1), torch.zeros(4, 16, 16)]),
        total_tokens_scanned=1000,
    )
    model_nd._initialize_spectral_wavefunctions_and_tables_from_cache(cache, rho_eff=0.0, seed=1337)
    model_diluted._initialize_spectral_wavefunctions_and_tables_from_cache(cache, rho_eff=0.0, seed=1337)

    # H3.2: Per-factor norm is 1.0 across R in {2, 3, 4} and invariant to adding branches
    base_re = torch.randn(2, 6, 4, 4, 16, dtype=torch.float64)
    base_im = torch.randn(2, 6, 4, 4, 16, dtype=torch.float64)
    norm_r2_re, norm_r2_im = normalize_raw_phi_per_factor_r_invariant(
        base_re[:, :, :2], base_im[:, :, :2]
    )
    norm_r4_re, norm_r4_im = normalize_raw_phi_per_factor_r_invariant(base_re, base_im)
    assert torch.allclose(norm_r2_re[:, :, 0], norm_r4_re[:, :, 0], atol=TEST_TOL_STRICT)
    assert torch.allclose(norm_r2_im[:, :, 0], norm_r4_im[:, :, 0], atol=TEST_TOL_STRICT)

    for r_val in (2, 3, 4):
        phi_re, phi_im = normalize_raw_phi_per_factor_r_invariant(
            base_re[:, :, :r_val], base_im[:, :, :r_val]
        )
        factor_sq = (phi_re.square() + phi_im.square()).sum(dim=-1)
        assert torch.allclose(factor_sq, torch.ones_like(factor_sq), atol=TEST_TOL_STRICT)


def test_track4_h3_3_phase_sensitive_router_step0_and_four_phases() -> None:
    """Verify H3.3 Step-0 uniform 1/sqrt(R) and strict differentiation of {+1, -1, +i, -i}."""
    torch.manual_seed(5903)
    bsz, seq_len, r_branches = 2, 5, 4
    psi0_re = torch.randn(bsz, seq_len, r_branches, 4, 16, dtype=torch.float64)
    psi0_im = torch.randn(bsz, seq_len, r_branches, 4, 16, dtype=torch.float64)

    zero_w_re = torch.zeros(r_branches, 4, 16, dtype=torch.float64)
    zero_w_im = torch.zeros(r_branches, 4, 16, dtype=torch.float64)
    zero_b = torch.zeros(r_branches, dtype=torch.float64)

    beta_0 = compute_phase_sensitive_branch_beta(
        psi0_re, psi0_im, psi0_re, psi0_im, zero_w_re, zero_w_im, zero_b
    )
    expected_unif = 1.0 / math.sqrt(float(r_branches))
    assert torch.allclose(beta_0, torch.full_like(beta_0, expected_unif), atol=TEST_TOL_STRICT)

    w_re = torch.randn(r_branches, 4, 16, dtype=torch.float64) * 0.4
    w_im = torch.randn(r_branches, 4, 16, dtype=torch.float64) * 0.4
    b_vec = torch.zeros(r_branches, dtype=torch.float64)

    beta_pos1 = compute_phase_sensitive_branch_beta(
        psi0_re, psi0_im, psi0_re, psi0_im, w_re, w_im, b_vec
    )
    beta_neg1 = compute_phase_sensitive_branch_beta(
        -psi0_re, -psi0_im, psi0_re, psi0_im, w_re, w_im, b_vec
    )
    beta_posi = compute_phase_sensitive_branch_beta(
        -psi0_im, psi0_re, psi0_re, psi0_im, w_re, w_im, b_vec
    )
    beta_negi = compute_phase_sensitive_branch_beta(
        psi0_im, -psi0_re, psi0_re, psi0_im, w_re, w_im, b_vec
    )

    for beta_tensor in (beta_pos1, beta_neg1, beta_posi, beta_negi):
        sq_sum = beta_tensor.square().sum(dim=-1)
        assert torch.allclose(sq_sum, torch.ones_like(sq_sum), atol=TEST_TOL_STRICT)

    assert float((beta_pos1 - beta_neg1).abs().max().item()) > 1e-3
    assert float((beta_posi - beta_negi).abs().max().item()) > 1e-3
    assert float((beta_pos1 - beta_posi).abs().max().item()) > 1e-3

    # Also test 6-arg factored (phi_re, phi_im, router_w_re, router_w_im, router_gain, router_bias) signature
    beta_6arg_pos1 = compute_phase_sensitive_branch_beta(
        psi0_re, psi0_im, w_re, w_im, torch.ones(r_branches, dtype=torch.float64), b_vec
    )
    beta_6arg_neg1 = compute_phase_sensitive_branch_beta(
        -psi0_re, -psi0_im, w_re, w_im, torch.ones(r_branches, dtype=torch.float64), b_vec
    )
    assert float((beta_6arg_pos1 - beta_6arg_neg1).abs().max().item()) > 1e-3


def test_track6_h1_1_and_h2_1_multihead_matmul_counts_and_gram_psd_purity() -> None:
    """Verify H1.1 matmul count reduction (11 -> 3 TxT / 5 total) and H2.1 PSD Gram purity."""
    torch.manual_seed(5904)
    cfg_mh = MultiHeadSuperpositionAttnConfig(
        num_meanings=2,
        num_factors=4,
        d_register=16,
        n_subheads_per_factor=2,
        attn_evolution_mode="multihead_gram",
    )
    layer_mh = MultiHeadSuperpositionMixedStateAttention(cfg_mh)
    psi_re = torch.randn(2, 24, cfg_mh.d_model)
    psi_im = torch.randn(2, 24, cfg_mh.d_model)

    total_mh, txt_mh = count_attention_matmul_calls(layer_mh, psi_re, psi_im)
    assert txt_mh == 3, f"Expected 3 (T x T) matmuls in multihead_gram, got {txt_mh}"
    assert total_mh == 5, f"Expected 5 total torch.matmul calls in multihead_gram, got {total_mh}"

    cfg_leg = MultiHeadSuperpositionAttnConfig(
        num_meanings=2,
        num_factors=4,
        d_register=16,
        n_subheads_per_factor=2,
        attn_evolution_mode="legacy_krylov_11mm",
    )
    layer_leg = MultiHeadSuperpositionMixedStateAttention(cfg_leg)
    total_leg, txt_leg = count_attention_matmul_calls(layer_leg, psi_re, psi_im)
    assert txt_leg == 11, f"Expected 11 (T x T) matmuls in legacy_krylov_11mm, got {txt_leg}"
    assert total_leg == 13

    # Check Hermitian, PSD, unit-trace properties of compute_normalized_cross_head_gram
    y_re = torch.randn(2, 12, 4, 3, 16, dtype=torch.float64)
    y_im = torch.randn(2, 12, 4, 3, 16, dtype=torch.float64)
    y_norm = torch.sqrt((y_re.square() + y_im.square()).sum(dim=-1, keepdim=True).clamp_min(1e-12))
    y_re, y_im = y_re / y_norm, y_im / y_norm
    g_hat_re, g_hat_im = compute_normalized_cross_head_gram(y_re, y_im)
    g_complex = torch.complex(g_hat_re, g_hat_im)
    # Hermitian: G == G^H
    assert torch.allclose(g_complex, g_complex.conj().transpose(-1, -2), atol=TEST_TOL_STRICT)
    # Trace == 1
    traces = torch.diagonal(g_hat_re, dim1=-2, dim2=-1).sum(dim=-1)
    assert torch.allclose(traces, torch.ones_like(traces), atol=TEST_TOL_STRICT)
    # PSD: eigenvalues >= 0
    eigvals = torch.linalg.eigvalsh(g_complex)
    assert float(eigvals.min().item()) >= -TEST_TOL_STRICT

    out_re, out_im, _, _ = layer_mh(psi_re, psi_im)
    assert out_re.shape == psi_re.shape
    assert out_im.shape == psi_im.shape
    tel_mh = layer_mh.compute_subhead_purity_telemetry()
    purity = float(tel_mh["reduced_density_purity_mean"])
    assert (1.0 / 2.0) - TEST_TOL_MEDIUM <= purity <= 1.0 + TEST_TOL_MEDIUM


def test_track6_schmidt_spectrum_parity_and_branch_non_cancellation() -> None:
    """Verify Schmidt spectrum parity (R x R Gram vs 16 x 16 density) and non-cancellation of opposite-sign branches."""
    torch.manual_seed(5905)
    n_tok, r_branches, n_factors, factor_dim, vocab_size = 6, 4, 4, 16, 32
    phi_re = torch.randn(n_tok, r_branches, n_factors, factor_dim, dtype=torch.float64)
    phi_im = torch.randn(n_tok, r_branches, n_factors, factor_dim, dtype=torch.float64)
    phi_re, phi_im = normalize_raw_phi_per_factor_r_invariant(phi_re, phi_im)

    raw_beta = torch.rand(n_tok, r_branches, dtype=torch.float64) + 0.2
    beta = raw_beta / torch.sqrt(raw_beta.square().sum(dim=-1, keepdim=True))

    # Compare 16 x 16 density matrix spectrum vs R x R Gram matrix spectrum
    phi_c = torch.complex(phi_re, phi_im)  # (N, R, M, 16)
    w_c = beta.unsqueeze(-1).unsqueeze(-1).to(torch.complex128) * phi_c  # (N, R, M, 16)
    rho_16 = torch.einsum("nrmi,nrmj->nmij", w_c, w_c.conj())
    gram_r = torch.einsum("nrmi,nsmi->nmrs", w_c.conj(), w_c)

    purity_16 = torch.diagonal(torch.matmul(rho_16, rho_16), dim1=-2, dim2=-1).real.sum(dim=-1)
    purity_r = torch.diagonal(torch.matmul(gram_r, gram_r), dim1=-2, dim2=-1).real.sum(dim=-1)
    assert torch.allclose(purity_16, purity_r, atol=TEST_TOL_STRICT)

    eigs_16 = torch.linalg.eigvalsh(rho_16)[..., -r_branches:]
    eigs_r = torch.linalg.eigvalsh(gram_r)
    assert torch.allclose(eigs_16, eigs_r, atol=TEST_TOL_STRICT)

    # Verify mixed-state density readout never suffers destructive branch cancellation when
    # two branches have identical kets up to an overall -1 sign on one factor (so M_re, M_im, K_re, K_im flip sign)
    psi_re = torch.randn(vocab_size, 2, n_factors, factor_dim, dtype=torch.float64)
    psi_im = torch.randn(vocab_size, 2, n_factors, factor_dim, dtype=torch.float64)
    psi_re, psi_im = normalize_raw_phi_per_factor_r_invariant(psi_re, psi_im)
    psi_re[:, 1] = psi_re[:, 0]
    psi_im[:, 1] = psi_im[:, 0]
    psi_factors = torch.cat([psi_re, psi_im], dim=-1)

    phi2_re = phi_re[:, :2].clone()
    phi2_im = phi_im[:, :2].clone()
    phi2_re[:, 1] = phi2_re[:, 0]
    phi2_im[:, 1] = phi2_im[:, 0]
    # Flip sign of branch 1 across all factors so x_bar and y_bar negate in branch 1
    phi2_flip_re = phi2_re.clone()
    phi2_flip_im = phi2_im.clone()
    phi2_flip_re[:, 1] = -phi2_re[:, 0]
    phi2_flip_im[:, 1] = -phi2_im[:, 0]

    beta_2 = torch.full((n_tok, 2), 1.0 / math.sqrt(2.0), dtype=torch.float64)
    alpha_2 = torch.full((vocab_size, 2), 1.0 / math.sqrt(2.0), dtype=torch.float64)
    povm_w = torch.ones((vocab_size,), dtype=torch.float64)
    hp = torch.tensor([2.0, 0.85, 0.0, 0.0], dtype=torch.float64)

    probs_same = eval_mixed_state_kronecker_born_probs(
        torch.cat([phi2_re, phi2_im], dim=-1), psi_factors, beta_2, alpha_2, povm_w, hp
    )
    probs_coh_flip = eval_calibrated_kronecker_born_probs(
        torch.cat([phi2_flip_re, phi2_flip_im], dim=-1), psi_factors, beta_2, alpha_2, povm_w, hp
    )
    probs_mix_flip = eval_mixed_state_kronecker_born_probs(
        torch.cat([phi2_flip_re, phi2_flip_im], dim=-1), psi_factors, beta_2, alpha_2, povm_w, hp
    )
    # Coherent superposition collapses to uniform when branch 1 negates branch 0 (kron_blend=0, coinc_gamma=0)
    assert float(probs_coh_flip.std(dim=-1).max().item()) < TEST_TOL_STRICT
    # Mixed-state density preserves non-degenerate distribution (sum of positive branch intensities)
    assert float(probs_mix_flip.std(dim=-1).mean().item()) > 5e-4
    assert float((probs_same - probs_coh_flip).abs().max().item()) > 1e-3

    # Exact U(1) branch phase invariance on pure 4-factor coincidence (holo_alpha=0, coinc_gamma=1):
    # rotating branch 1 by exp(i * pi / 4) per factor rotates K_{1,n,c} by exp(i * pi) = -1,
    # which leaves mixed-state |K_{r,n,c}|^2 invariant (< 1e-6) while coherent superposition cancels!
    hp_coinc = torch.tensor([2.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    c45 = math.cos(math.pi / 4.0)
    s45 = math.sin(math.pi / 4.0)
    phi2_phase_re = phi2_re.clone()
    phi2_phase_im = phi2_im.clone()
    phi2_phase_re[:, 1] = c45 * phi2_re[:, 0] - s45 * phi2_im[:, 0]
    phi2_phase_im[:, 1] = s45 * phi2_re[:, 0] + c45 * phi2_im[:, 0]
    probs_mix_orig = eval_mixed_state_kronecker_born_probs(
        torch.cat([phi2_re, phi2_im], dim=-1), psi_factors, beta_2, alpha_2, povm_w, hp_coinc
    )
    probs_mix_phase = eval_mixed_state_kronecker_born_probs(
        torch.cat([phi2_phase_re, phi2_phase_im], dim=-1), psi_factors, beta_2, alpha_2, povm_w, hp_coinc
    )
    assert torch.allclose(probs_mix_orig, probs_mix_phase, atol=TEST_TOL_STRICT)


def test_track6_h6_1_mixed_state_chunked_nll_backward_exactness_and_pow_count() -> None:
    """Verify H6.1 analytical gradients against reference autograd and single-pass `pow` call count."""
    torch.manual_seed(5906)
    n_tok, r_branches, vocab_size = 10, 4, 24
    phi_re_0 = torch.randn(n_tok, r_branches, 4, 16, dtype=torch.float64)
    phi_im_0 = torch.randn(n_tok, r_branches, 4, 16, dtype=torch.float64)
    phi_re_0, phi_im_0 = normalize_raw_phi_per_factor_r_invariant(phi_re_0, phi_im_0)
    phi_0 = torch.cat([phi_re_0, phi_im_0], dim=-1)

    psi_re_0 = torch.randn(vocab_size, r_branches, 4, 16, dtype=torch.float64)
    psi_im_0 = torch.randn(vocab_size, r_branches, 4, 16, dtype=torch.float64)
    psi_re_0, psi_im_0 = normalize_raw_phi_per_factor_r_invariant(psi_re_0, psi_im_0)
    psi_0 = torch.cat([psi_re_0, psi_im_0], dim=-1)

    raw_beta = torch.rand(n_tok, r_branches, dtype=torch.float64) + 0.2
    beta_0 = raw_beta / torch.sqrt(raw_beta.square().sum(dim=-1, keepdim=True))
    raw_alpha = torch.rand(vocab_size, r_branches, dtype=torch.float64) + 0.2
    alpha_0 = raw_alpha / torch.sqrt(raw_alpha.square().sum(dim=-1, keepdim=True))
    povm_0 = torch.rand(vocab_size, dtype=torch.float64) + 0.5
    targets = torch.randint(0, vocab_size, (n_tok,))

    MixedStateKroneckerBornChunkedNLLFn.vocab_chunk_size = 8
    for p_degree in (1.0, 2.0):
        hp = torch.tensor([p_degree, 0.85, 0.30, 0.15], dtype=torch.float64)
        phi_a = phi_0.clone().detach().requires_grad_(True)
        psi_a = psi_0.clone().detach().requires_grad_(True)
        beta_a = beta_0.clone().detach().requires_grad_(True)
        alpha_a = alpha_0.clone().detach().requires_grad_(True)
        povm_a = povm_0.clone().detach().requires_grad_(True)

        loss_chunked = MixedStateKroneckerBornChunkedNLLFn.apply(
            phi_a, psi_a, beta_a, alpha_a, povm_a, targets, hp, None
        )
        loss_chunked.backward()

        phi_b = phi_0.clone().detach().requires_grad_(True)
        psi_b = psi_0.clone().detach().requires_grad_(True)
        beta_b = beta_0.clone().detach().requires_grad_(True)
        alpha_b = alpha_0.clone().detach().requires_grad_(True)
        povm_b = povm_0.clone().detach().requires_grad_(True)

        probs_ref = eval_mixed_state_kronecker_born_probs(phi_b, psi_b, beta_b, alpha_b, povm_b, hp)
        target_p = probs_ref.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        loss_ref = -torch.log(target_p).mean()
        loss_ref.backward()

        assert torch.allclose(loss_chunked, loss_ref, atol=TEST_TOL_STRICT)
        assert torch.allclose(phi_a.grad, phi_b.grad, atol=TEST_TOL_STRICT)
        assert torch.allclose(psi_a.grad, psi_b.grad, atol=TEST_TOL_STRICT)
        assert torch.allclose(beta_a.grad, beta_b.grad, atol=TEST_TOL_STRICT)
        assert torch.allclose(alpha_a.grad, alpha_b.grad, atol=TEST_TOL_STRICT)
        assert torch.allclose(povm_a.grad, povm_b.grad, atol=TEST_TOL_STRICT)

    # Verify single-pass `torch.pow` call count per chunk in backward (1 fwd + 1 bwd = 2 total for 1 chunk)
    MixedStateKroneckerBornChunkedNLLFn.vocab_chunk_size = 64
    CalibratedKroneckerBornChunkedNLLFn.vocab_chunk_size = 64
    orig_pow = torch.pow
    ms_pow_calls = 0
    cal_pow_calls = 0

    def _count_ms_pow(self_t: torch.Tensor, exp_val: Any) -> torch.Tensor:
        nonlocal ms_pow_calls
        ms_pow_calls += 1
        return orig_pow(self_t, exp_val)

    def _count_cal_pow(self_t: torch.Tensor, exp_val: Any) -> torch.Tensor:
        nonlocal cal_pow_calls
        cal_pow_calls += 1
        return orig_pow(self_t, exp_val)

    hp2 = torch.tensor([2.0, 0.85, 0.30, 0.15], dtype=torch.float64)
    phi_p = phi_0.clone().detach().requires_grad_(True)
    psi_p = psi_0.clone().detach().requires_grad_(True)
    beta_p = beta_0.clone().detach().requires_grad_(True)
    alpha_p = alpha_0.clone().detach().requires_grad_(True)
    povm_p = povm_0.clone().detach().requires_grad_(True)

    with patch("torch.pow", side_effect=_count_ms_pow):
        l_ms = MixedStateKroneckerBornChunkedNLLFn.apply(
            phi_p, psi_p, beta_p, alpha_p, povm_p, targets, hp2, None
        )
        l_ms.backward()

    with patch("torch.pow", side_effect=_count_cal_pow):
        l_cal = CalibratedKroneckerBornChunkedNLLFn.apply(
            phi_p, psi_p, beta_p, alpha_p, povm_p, targets, hp2, None
        )
        l_cal.backward()

    assert ms_pow_calls == 2, f"Expected 2 torch.pow calls (1 fwd + 1 bwd) for 1 chunk, got {ms_pow_calls}"
    assert cal_pow_calls == 2, f"Expected 2 torch.pow calls (1 fwd + 1 bwd) for 1 chunk, got {cal_pow_calls}"


@pytest.mark.parametrize(
    "attn_mode,readout_mode",
    [
        ("multihead_gram", "mixed_density"),
        ("legacy_krylov_11mm", "mixed_density"),
        ("adhoc_poly_gamma", "coherent_superposition"),
    ],
)
def test_end_to_end_multihead_mixed_state_kronecker_gpt_and_eval(
    attn_mode: str,
    readout_mode: str,
    tmp_path: Path,
) -> None:
    """Verify full MultiHeadMixedStateKroneckerGPT forward, backward, optimizer step, and eval."""
    torch.manual_seed(5907)
    vocab_size = 64
    seq_len = 16
    cfg = MultiHeadMixedStateKroneckerConfig(
        vocab_size=vocab_size,
        valid_vocab_size=vocab_size,
        num_meanings=2,
        num_factors=4,
        factor_dim=16,
        num_layers=1,
        n_subheads_per_factor=2,
        max_seq_len=seq_len,
        vocab_chunk_size=32,
        attn_evolution_mode=attn_mode,
        readout_density_mode=readout_mode,
    )
    model = MultiHeadMixedStateKroneckerGPT(cfg)
    muon_opt, adamw_opt = build_multihead_mixed_state_kronecker_optimizers(model)

    x = torch.randint(0, vocab_size, (2, seq_len))
    y = torch.randint(0, vocab_size, (2, seq_len))
    _, nll_loss = model(x, targets=y)
    assert nll_loss is not None
    pen_loss, pen_stats = compute_multihead_mixed_state_kronecker_penalties(model)
    total_loss = nll_loss + pen_loss
    assert torch.isfinite(total_loss)
    total_loss.backward()

    muon_opt.step()
    adamw_opt.step()

    purity_tel = model.compute_subhead_purity_telemetry()
    branch_tel = model.compute_branch_routing_telemetry()
    assert "reduced_density_purity_mean" in purity_tel
    assert "step0_branch_ket_purity" in branch_tel
    assert float(pen_stats["unitary_pen"]) < 0.05

    # Verify evaluate_modded_nanogpt_validation compatibility
    shard_path = tmp_path / "fineweb_val_000000.bin"
    eval_tokens = 64
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 1
    header[2] = eval_tokens + 4
    tokens = np.random.randint(0, vocab_size, size=(eval_tokens + 4,), dtype=np.uint16)
    with shard_path.open("wb") as fh:
        fh.write(header.tobytes())
        fh.write(tokens.tobytes())

    val_metrics = evaluate_modded_nanogpt_validation(
        model=model,
        val_tokens=shard_path,
        max_eval_tokens=eval_tokens,
        batch_size=2,
        seq_len=seq_len,
    )
    assert math.isfinite(float(val_metrics["val_ce"]))
    assert int(val_metrics["eval_tokens"]) == eval_tokens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused Triton kernel parity test")
def test_cuda_fused_kronecker_mixed_state_kernels_parity() -> None:
    """Verify numerical equivalence between fused Triton/CUDA kernels and reference implementations."""
    from qdlblocks.kernels.fused_kronecker_mixed_state_kernels import (
        flash_causal_pboson_complex_attn_triton,
        fused_cross_head_gram_stinespring_step,
        fused_mixed_state_kronecker_chunk_bwd_triton,
        fused_mixed_state_kronecker_chunk_fwd_triton,
    )
    from qdlblocks.layers.multihead_superposition_mixed_state_attn import (
        apply_stinespring_subhead_to_branch_coupling,
        causal_born_p2_complex_attn,
        compute_cross_head_hermitian_gram_step,
    )
    from qdlblocks.models.multihead_mixed_state_kronecker_gpt import (
        _mixed_state_kronecker_chunk_bwd_step,
        _mixed_state_kronecker_chunk_fwd_step,
    )

    torch.manual_seed(5908)
    device = torch.device("cuda")

    # 1. Fused Triton Mixed-State Kronecker Chunk Forward & Backward across R in {2, 4}
    for r_num in (2, 4):
        n_tok, v_chunk = 128, 256
        z_cat = torch.randn(r_num * 4, 2 * n_tok, v_chunk, device=device, dtype=torch.bfloat16) * 0.25
        beta_f = torch.rand(n_tok, r_num, device=device, dtype=torch.float32) + 0.2
        beta_f = beta_f / torch.sqrt(beta_f.square().sum(dim=-1, keepdim=True))
        alpha_c = torch.rand(v_chunk, r_num, device=device, dtype=torch.float32) + 0.2
        alpha_c = alpha_c / torch.sqrt(alpha_c.square().sum(dim=-1, keepdim=True))
        povm_c = torch.rand(v_chunk, device=device, dtype=torch.float32) + 0.5
        targets_in_chunk = torch.randint(-5, v_chunk + 5, (n_tok,), device=device)
        in_chunk_f = ((targets_in_chunk >= 0) & (targets_in_chunk < v_chunk)).to(torch.float32)
        rel_idx = targets_in_chunk.clamp(0, v_chunk - 1).unsqueeze(1)
        v_idx = torch.arange(v_chunk, device=device, dtype=torch.long).unsqueeze(0)
        targets_col = targets_in_chunk.unsqueeze(1)
        scale_t = (torch.rand(n_tok, 1, device=device, dtype=torch.float32) * 0.05).detach()
        inv_z = (torch.rand(n_tok, 1, device=device, dtype=torch.float32) + 0.5).detach()
        scale_inv_z = (scale_t, inv_z)
        target_match = (v_idx, targets_col)
        coeffs = (0.85, 0.30, 0.15)

        z_ref, t_ref = _mixed_state_kronecker_chunk_fwd_step(
            z_cat, beta_f, alpha_c, povm_c, rel_idx, in_chunk_f, 2.0, coeffs
        )
        z_tri, t_tri = fused_mixed_state_kronecker_chunk_fwd_triton(
            z_cat, beta_f, alpha_c, povm_c, rel_idx, in_chunk_f, 2.0, coeffs
        )
        assert torch.allclose(z_tri, z_ref, rtol=1e-2, atol=1e-3), f"R={r_num} fwd z_sum mismatch"
        assert torch.allclose(t_tri, t_ref, rtol=1e-2, atol=1e-3), f"R={r_num} fwd t_num mismatch"

        g_ref, db_ref, da_ref, dp_ref = _mixed_state_kronecker_chunk_bwd_step(
            z_cat, beta_f, alpha_c, povm_c, scale_inv_z, target_match, 2.0, coeffs
        )
        g_tri, db_tri, da_tri, dp_tri = fused_mixed_state_kronecker_chunk_bwd_triton(
            z_cat, beta_f, alpha_c, povm_c, scale_inv_z, target_match, 2.0, coeffs
        )
        assert torch.allclose(g_tri, g_ref, rtol=2e-2, atol=2e-3), f"R={r_num} bwd g_cat mismatch"
        assert torch.allclose(db_tri, db_ref, rtol=2e-2, atol=2e-3), f"R={r_num} bwd d_beta mismatch"
        assert torch.allclose(da_tri, da_ref, rtol=2e-2, atol=2e-3), f"R={r_num} bwd d_alpha mismatch"
        assert torch.allclose(dp_tri, dp_ref, rtol=2e-2, atol=2e-3), f"R={r_num} bwd d_povm mismatch"

    # 2. Flash Causal Complex p=2 Born Attention Forward & Backward across T in {64, 256}
    for seq_t in (64, 256):
        bsz, n_heads, d_h = 2, 8, 16
        q_re_0 = torch.randn(bsz, n_heads, seq_t, d_h, device=device, dtype=torch.float32)
        q_im_0 = torch.randn(bsz, n_heads, seq_t, d_h, device=device, dtype=torch.float32)
        q_n = torch.sqrt((q_re_0.square() + q_im_0.square()).sum(dim=-1, keepdim=True).clamp_min(1e-6))
        q_re_0, q_im_0 = (q_re_0 / q_n).detach(), (q_im_0 / q_n).detach()

        k_re_0 = torch.randn(bsz, n_heads, seq_t, d_h, device=device, dtype=torch.float32)
        k_im_0 = torch.randn(bsz, n_heads, seq_t, d_h, device=device, dtype=torch.float32)
        k_n = torch.sqrt((k_re_0.square() + k_im_0.square()).sum(dim=-1, keepdim=True).clamp_min(1e-6))
        k_re_0, k_im_0 = (k_re_0 / k_n).detach(), (k_im_0 / k_n).detach()

        v_re_0 = torch.randn(bsz, n_heads, seq_t, d_h, device=device, dtype=torch.float32)
        v_im_0 = torch.randn(bsz, n_heads, seq_t, d_h, device=device, dtype=torch.float32)
        v_n = torch.sqrt((v_re_0.square() + v_im_0.square()).sum(dim=-1, keepdim=True).clamp_min(1e-6))
        v_re_0, v_im_0 = (v_re_0 / v_n).detach(), (v_im_0 / v_n).detach()

        grad_yr = torch.randn_like(v_re_0)
        grad_yi = torch.randn_like(v_im_0)

        qr_a, qi_a = q_re_0.clone().requires_grad_(True), q_im_0.clone().requires_grad_(True)
        kr_a, ki_a = k_re_0.clone().requires_grad_(True), k_im_0.clone().requires_grad_(True)
        vr_a, vi_a = v_re_0.clone().requires_grad_(True), v_im_0.clone().requires_grad_(True)
        yr_ref, yi_ref = causal_born_p2_complex_attn(qr_a, qi_a, kr_a, ki_a, vr_a, vi_a)
        ((yr_ref * grad_yr).sum() + (yi_ref * grad_yi).sum()).backward()

        qr_b, qi_b = q_re_0.clone().requires_grad_(True), q_im_0.clone().requires_grad_(True)
        kr_b, ki_b = k_re_0.clone().requires_grad_(True), k_im_0.clone().requires_grad_(True)
        vr_b, vi_b = v_re_0.clone().requires_grad_(True), v_im_0.clone().requires_grad_(True)
        yr_tri, yi_tri = flash_causal_pboson_complex_attn_triton(qr_b, qi_b, kr_b, ki_b, vr_b, vi_b)
        ((yr_tri * grad_yr).sum() + (yi_tri * grad_yi).sum()).backward()

        assert torch.allclose(yr_tri, yr_ref, rtol=1e-3, atol=1e-4), f"T={seq_t} attn fwd y_re mismatch"
        assert torch.allclose(yi_tri, yi_ref, rtol=1e-3, atol=1e-4), f"T={seq_t} attn fwd y_im mismatch"
        assert torch.allclose(qr_b.grad[:, :, 1:], qr_a.grad[:, :, 1:], rtol=2e-3, atol=1e-3), f"T={seq_t} attn dQ_re mismatch"
        assert torch.allclose(kr_b.grad[:, :, 1:], kr_a.grad[:, :, 1:], rtol=2e-3, atol=1e-3), f"T={seq_t} attn dK_re mismatch"
        assert torch.allclose(vr_b.grad, vr_a.grad, rtol=2e-3, atol=1e-3), f"T={seq_t} attn dV_re mismatch"
        assert float(torch.nn.functional.cosine_similarity(qr_b.grad.flatten(), qr_a.grad.flatten(), dim=0).item()) > 0.99999
        assert float(torch.nn.functional.cosine_similarity(kr_b.grad.flatten(), kr_a.grad.flatten(), dim=0).item()) > 0.99999

    # 3. Fused Cross-Head Gram + Stinespring Coupling vs. Eager Reference
    y_re = torch.randn(2, 16, 4, 2, 16, device=device, dtype=torch.float32)
    y_im = torch.randn(2, 16, 4, 2, 16, device=device, dtype=torch.float32)
    y_norm = torch.sqrt((y_re.square() + y_im.square()).sum(dim=-1, keepdim=True).clamp_min(1e-6))
    y_re, y_im = y_re / y_norm, y_im / y_norm
    gram_eta = torch.tensor(0.35, device=device)
    sw_re = torch.eye(2, device=device).unsqueeze(0).repeat(4, 1, 1)
    sw_im = torch.zeros(4, 2, 2, device=device)

    t_re, t_im, pur_ref = compute_cross_head_hermitian_gram_step(y_re, y_im, gram_eta)
    zr_ref, zi_ref, nu_ref = apply_stinespring_subhead_to_branch_coupling(t_re, t_im, sw_re, sw_im)
    zr_fus, zi_fus, pur_fus, nu_fus = fused_cross_head_gram_stinespring_step(
        y_re, y_im, gram_eta, sw_re, sw_im
    )
    assert torch.allclose(zr_fus, zr_ref, rtol=1e-3, atol=1e-4)
    assert torch.allclose(zi_fus, zi_ref, rtol=1e-3, atol=1e-4)
    assert torch.allclose(nu_fus, nu_ref, rtol=1e-3, atol=1e-4)
    assert torch.allclose(pur_fus, pur_ref, rtol=1e-3, atol=1e-4)

