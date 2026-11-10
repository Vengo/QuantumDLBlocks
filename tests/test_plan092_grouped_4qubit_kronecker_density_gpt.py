"""Unit tests for Wave 72 (`EXPERIMENT_PLAN_092.md`) `GroupedFourQubitKroneckerDensityGPT` and Theorem 13."""

from __future__ import annotations

import torch

from qdlblocks.benchmarks.anchored_bipartite_density_gpt_lab import _build_deterministic_haar_unitary_stack
from qdlblocks.benchmarks.grouped_4qubit_kronecker_density_gpt_lab import (
    GroupedFourQubitKroneckerDensityGPT,
    Wave72VariantMode,
    build_deterministic_16qubit_binary_codebook,
    build_wave72_variant_spec,
    compute_four_qubit_partial_trace_purity,
    compute_grouped_4qubit_segre_product,
    evaluate_factorized_65536d_kronecker_born_overlap,
)
from qdlblocks.benchmarks.pure_density_matrix_gpt_lab import (
    apply_complex_unitary_to_cholesky,
    build_wave69_optimizers,
    normalize_cholesky_factor_frobenius,
)


def test_theorem13a_segre_norm_orthogonality_and_entanglement() -> None:
    """Verify Theorem 13a: 4-qubit Segre norm preservation, 65,536D orthogonality, and SU(16) intra-head entanglement."""
    torch.manual_seed(92)
    q_re = torch.randn(16, 4, 4, 2)
    q_im = torch.randn(16, 4, 4, 2)
    phi_re, phi_im = compute_grouped_4qubit_segre_product(q_re, q_im)
    norms = (phi_re.square() + phi_im.square()).sum(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # Separable Segre product state has 2-qubit partial-trace purity == 1.0
    l_re, l_im = phi_re.unsqueeze(2).unsqueeze(-1), phi_im.unsqueeze(2).unsqueeze(-1)
    pur_sep = compute_four_qubit_partial_trace_purity(l_re, l_im)
    assert torch.allclose(pur_sep, torch.ones_like(pur_sep), atol=1e-5)

    # Haar SU(16) entangles the 4 qubits (purity < 0.70)
    u16_re, u16_im = _build_deterministic_haar_unitary_stack(4, 16, 93)
    ent_re, ent_im = apply_complex_unitary_to_cholesky(l_re, l_im, u16_re, u16_im)
    pur_ent = compute_four_qubit_partial_trace_purity(ent_re, ent_im)
    assert float(pur_ent.mean().item()) < 0.70

    # Binary 16-qubit codebook has exact mutual orthogonality in C^{65,536}
    cb_re, cb_im = build_deterministic_16qubit_binary_codebook(32, perturb_std=0.0, seed=94)
    cb_phi_re, cb_phi_im = compute_grouped_4qubit_segre_product(cb_re, cb_im)
    ov_re = torch.einsum("ugi,vgi->uvg", cb_phi_re, cb_phi_re) + torch.einsum("ugi,vgi->uvg", cb_phi_im, cb_phi_im)
    ov_im = torch.einsum("ugi,vgi->uvg", cb_phi_re, cb_phi_im) - torch.einsum("ugi,vgi->uvg", cb_phi_im, cb_phi_re)
    gram_65536 = (ov_re.square() + ov_im.square()).prod(dim=-1)
    assert torch.allclose(gram_65536, torch.eye(32), atol=1e-5)


def test_theorem13bc_multibranch_ghz_and_65536d_factorization() -> None:
    """Verify Theorem 13b (mean-field collapse vs multi-branch exactness) and Theorem 13c (65,536D factorized identity)."""
    patterns = torch.zeros(16, 4, 16)
    for idx in range(16):
        for g in range(4):
            patterns[idx, g, (idx >> g) & 1] = 1.0
    patterns_im = torch.zeros_like(patterns)

    # Mean-field B=1 collapses to 1/16 on targets and 14/16 on cross-terms
    l_mf_re = torch.zeros(1, 4, 16, 2)
    l_mf_re[0, :, 0, 0] = (0.5) ** 0.5
    l_mf_re[0, :, 1, 1] = (0.5) ** 0.5
    probs_mf = evaluate_factorized_65536d_kronecker_born_overlap((l_mf_re, torch.zeros_like(l_mf_re)), (patterns, patterns_im), torch.ones(1))
    assert abs(float(probs_mf[0].item()) - 0.0625) < 1e-5
    assert abs(float(probs_mf[1:15].sum().item()) - 0.875) < 1e-5

    # Multi-branch B=2 recovers exact 0.5 / 0.5 on targets and 0.0 on cross-terms
    l_mb_re = torch.zeros(2, 4, 16, 1)
    l_mb_re[0, :, 0, 0] = 1.0
    l_mb_re[1, :, 1, 0] = 1.0
    probs_mb = evaluate_factorized_65536d_kronecker_born_overlap(
        (l_mb_re, torch.zeros_like(l_mb_re)), (patterns, patterns_im), torch.tensor([0.5, 0.5])
    )
    assert abs(float(probs_mb[0].item()) - 0.5) < 1e-5
    assert abs(float(probs_mb[15].item()) - 0.5) < 1e-5
    assert float(probs_mb[1:15].sum().item()) == 0.0


def test_grouped_4qubit_kronecker_gpt_forward_backward_and_quotient_step() -> None:
    """Verify forward/backward pass and `PureDensityQuotientMuon` step on `GroupedFourQubitKroneckerDensityGPT`."""
    spec = build_wave72_variant_spec(Wave72VariantMode.W72_Q2_GROUPED_4QUBIT_MULTIBRANCH_KRON_4L, seed=1337)
    model = GroupedFourQubitKroneckerDensityGPT(spec)
    trunk_opt, vocab_opt, scalar_opt = build_wave69_optimizers(model)

    idx = torch.randint(0, 1024, (2, 8))
    targets = torch.randint(0, 1024, (2, 8))
    loss, metrics = model(idx, targets=targets, return_metrics=True)
    assert torch.isfinite(loss)
    assert "top1_acc" in metrics
    loss.backward()

    trunk_opt.record_diagnostics = True
    trunk_opt.step()
    vocab_opt.step_active(torch.unique(torch.cat([idx.reshape(-1), targets.reshape(-1)])), lr_scale=1.0)
    scalar_opt.step_active(active_idx=None, lr_scale=1.0)
    model.restrict_scalars_()
    gd = trunk_opt.last_gauge_diagnostics
    assert float(gd.get("su16_step_phase_frac", 0.0)) < 1e-6
    assert float(gd.get("kraus_step_gauge_frac", 0.0)) < 1e-5
