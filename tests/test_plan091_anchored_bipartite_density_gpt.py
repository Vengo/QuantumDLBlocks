"""Unit tests for Wave 71 (`EXPERIMENT_PLAN_091.md`) `AnchoredBipartitePureDensityGPT` and Theorem 12."""

from __future__ import annotations

import torch

from qdlblocks.benchmarks.anchored_bipartite_density_gpt_lab import (
    AnchoredBipartitePureDensityGPT,
    Wave71VariantMode,
    _build_deterministic_haar_unitary_stack,
    apply_bipartite_sensory_predictive_register_split,
    apply_dual_unitary_sensory_anchored_qkv,
    build_wave71_variant_spec,
)
from qdlblocks.benchmarks.pure_density_matrix_gpt_lab import (
    build_wave69_optimizers,
    normalize_cholesky_factor_frobenius,
    pack_cholesky_factor_outer_isometric,
    unpack_hermitian_isometric,
)


def test_theorem12ab_unanchored_collision_vs_anchored_distinguishability() -> None:
    """Verify Theorem 12a (unanchored collision = 0) and Theorem 12b (anchored separation > 0 + unit trace)."""
    torch.manual_seed(42)
    bsz, num_h, seq_len, d_h, r_rank = 2, 4, 6, 16, 4

    l_star_re, l_star_im = normalize_cholesky_factor_frobenius(
        torch.randn(bsz, num_h, seq_len, d_h, r_rank),
        torch.randn(bsz, num_h, seq_len, d_h, r_rank),
    )
    l0_a_re, l0_a_im = normalize_cholesky_factor_frobenius(
        torch.randn(bsz, num_h, seq_len, d_h, r_rank),
        torch.randn(bsz, num_h, seq_len, d_h, r_rank),
    )
    l0_b_re, l0_b_im = normalize_cholesky_factor_frobenius(
        torch.randn(bsz, num_h, seq_len, d_h, r_rank),
        torch.randn(bsz, num_h, seq_len, d_h, r_rank),
    )
    u_evol_re, u_evol_im = _build_deterministic_haar_unitary_stack(num_h, d_h, 43)
    u_anc_re, u_anc_im = _build_deterministic_haar_unitary_stack(num_h, d_h, 44)
    anchor_logit = torch.zeros(num_h, dtype=torch.float32)

    # Unanchored collision (Theorem 12a)
    un_a_re, un_a_im = apply_dual_unitary_sensory_anchored_qkv((l_star_re, l_star_im), (l0_a_re, l0_a_im), (u_evol_re, u_evol_im), None, None)
    un_b_re, un_b_im = apply_dual_unitary_sensory_anchored_qkv((l_star_re, l_star_im), (l0_b_re, l0_b_im), (u_evol_re, u_evol_im), None, None)
    rho_un_a = pack_cholesky_factor_outer_isometric(un_a_re, un_a_im)
    rho_un_b = pack_cholesky_factor_outer_isometric(un_b_re, un_b_im)
    assert torch.allclose(rho_un_a, rho_un_b, atol=0.0)

    # Anchored distinguishability & CPTP unit trace (Theorem 12b)
    anc_a_re, anc_a_im = apply_dual_unitary_sensory_anchored_qkv(
        (l_star_re, l_star_im), (l0_a_re, l0_a_im), (u_evol_re, u_evol_im), (u_anc_re, u_anc_im), anchor_logit
    )
    anc_b_re, anc_b_im = apply_dual_unitary_sensory_anchored_qkv(
        (l_star_re, l_star_im), (l0_b_re, l0_b_im), (u_evol_re, u_evol_im), (u_anc_re, u_anc_im), anchor_logit
    )
    rho_anc_a = pack_cholesky_factor_outer_isometric(anc_a_re, anc_a_im)
    rho_anc_b = pack_cholesky_factor_outer_isometric(anc_b_re, anc_b_im)
    assert float((rho_anc_a - rho_anc_b).norm(dim=-1).mean().item()) > 0.15
    frob_sq = anc_a_re.square().sum(dim=(-2, -1)) + anc_a_im.square().sum(dim=(-2, -1))
    assert torch.allclose(frob_sq, torch.ones_like(frob_sq), atol=1e-5)
    rho_re, _ = unpack_hermitian_isometric(rho_anc_a, d_h)
    tr = torch.diagonal(rho_re, dim1=-2, dim2=-1).sum(dim=-1)
    assert torch.allclose(tr, torch.ones_like(tr), atol=1e-5)


def test_theorem12c_bipartite_register_and_decoupled_io_frames() -> None:
    """Verify Theorem 12c bipartite register trace preservation and decoupled input/output frames."""
    torch.manual_seed(91)
    bsz, num_h, seq_len, d_h, r_rank = 2, 4, 5, 16, 4
    l_out_re, l_out_im = normalize_cholesky_factor_frobenius(
        torch.randn(bsz, num_h, seq_len, d_h, r_rank),
        torch.randn(bsz, num_h, seq_len, d_h, r_rank),
    )
    l0_re, l0_im = normalize_cholesky_factor_frobenius(
        torch.randn(bsz, num_h, seq_len, d_h, r_rank),
        torch.randn(bsz, num_h, seq_len, d_h, r_rank),
    )
    lb_re, lb_im = apply_bipartite_sensory_predictive_register_split((l_out_re, l_out_im), (l0_re, l0_im), torch.zeros(num_h))
    frob_sq = lb_re.square().sum(dim=(-2, -1)) + lb_im.square().sum(dim=(-2, -1))
    assert torch.allclose(frob_sq, torch.ones_like(frob_sq), atol=1e-5)

    spec_a2 = build_wave71_variant_spec(Wave71VariantMode.W71_A2_ANCHORED_BIPARTITE_DECOUPLED_4L, seed=1337)
    model_a2 = AnchoredBipartitePureDensityGPT(spec_a2)
    assert len(model_a2.bridge_parameter_pairs()) == 0
    assert float((model_a2.boundary.q_in_ortho - model_a2.boundary.q_out_ortho).norm().item()) > 10.0


def test_anchored_bipartite_gpt_forward_backward_and_quotient_step() -> None:
    """Verify forward/backward pass and `PureDensityQuotientMuon` step on `AnchoredBipartitePureDensityGPT`."""
    spec = build_wave71_variant_spec(Wave71VariantMode.W71_A2_ANCHORED_BIPARTITE_DECOUPLED_4L, seed=1337)
    model = AnchoredBipartitePureDensityGPT(spec)
    trunk_opt, vocab_opt, scalar_opt = build_wave69_optimizers(model)

    idx = torch.randint(0, 1024, (2, 8))
    targets = torch.randint(0, 1024, (2, 8))
    loss, metrics = model(idx, targets=targets, return_metrics=True)
    assert torch.isfinite(loss)
    assert "val_ce" in metrics or "top1_acc" in metrics
    loss.backward()

    trunk_opt.record_diagnostics = True
    trunk_opt.step()
    vocab_opt.step_active(torch.unique(torch.cat([idx.reshape(-1), targets.reshape(-1)])), lr_scale=1.0)
    scalar_opt.step_active(active_idx=None, lr_scale=1.0)
    model.restrict_scalars_()
    gd = trunk_opt.last_gauge_diagnostics
    assert float(gd.get("su16_step_phase_frac", 0.0)) < 1e-6
    assert float(gd.get("kraus_step_gauge_frac", 0.0)) < 1e-5
