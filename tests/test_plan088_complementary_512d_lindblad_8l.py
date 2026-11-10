"""Unit tests for Wave 68 (`EXPERIMENT_PLAN_088.md`): Complementary `512D` ($256 \\oplus 256$) Readout, Lindblad Gates, & Parallel Transport."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdlblocks.benchmarks.complementary_512d_lindblad_8l_lab import (  # noqa: E402
    Complementary512DLindbladGPT,
    Wave68VariantMode,
    build_wave68_optimizers,
    build_wave68_variant_spec,
)
from scripts.complementary_512d_lindblad_inspector import (  # noqa: E402
    audit_complementary_512d_parseval_and_normalization,
    audit_dual_subspace_custom_autograd,
    audit_isometric_parallel_transport,
    audit_lindblad_highway_density_invariants,
)
from scripts.verify_lean4_theorems import verify_lean4_package  # noqa: E402


def test_lean4_theorems_package_zero_sorry_zero_axiom() -> None:
    """Verify all 5 Lean 4 quantum-geometry theorem modules compile with zero `sorry`/`axiom`."""
    rep = verify_lean4_package(lean_dir=ROOT / "lean", clean_first=False)
    assert rep["overall_passed"] is True
    assert rep["static_zero_sorry_admit_axiom_passed"] is True
    assert rep["verified_theorems_count"] == 14
    assert len(rep["missing_theorems"]) == 0
    assert len(rep["disallowed_axioms"]) == 0


def test_complementary_512d_parseval_and_50257_normalization() -> None:
    """Verify `100.0%` Parseval `512D` norm preservation and exact `50,257`-way normalization."""
    rep = audit_complementary_512d_parseval_and_normalization(torch.device("cpu"))
    assert abs(rep["parseval_512d_retention_ratio"] - 1.0) < 1e-5
    assert rep["subspace_cross_ortho_max_err"] < 1e-5
    assert rep["luders_50257_prob_sum_max_err"] < 1e-4
    assert rep["dual_50257_prob_sum_max_err"] < 1e-4
    assert rep["luders_nll_vs_full_probs_err"] < 1e-4
    assert rep["dual_nll_vs_full_probs_err"] < 1e-4


def test_dual_subspace_custom_autograd_matches_reference() -> None:
    """Verify `_DualSubspace512DSoftcapCEFn` forward and backward match the PyTorch reference."""
    rep = audit_dual_subspace_custom_autograd(torch.device("cpu"))
    assert rep["fwd_loss_abs_diff"] < 1e-4
    assert rep["grad_h1_max_err"] < 1e-4
    assert rep["grad_h2_max_err"] < 1e-4
    assert rep["grad_w_tok_max_err"] < 1e-4
    assert rep["grad_e_coarse_max_err"] < 1e-4
    assert rep["grad_e_fine_max_err"] < 1e-4
    assert rep["grad_bias_max_err"] < 1e-4


def test_lindblad_highway_and_parallel_transport_invariants() -> None:
    """Verify unit trace, PSD eigenvalues, and isometric parallel transport norm conservation."""
    dev = torch.device("cpu")
    l_rep = audit_lindblad_highway_density_invariants(dev)
    assert l_rep["lindblad_trace_max_err"] < 1e-5
    assert l_rep["lindblad_hermitian_max_err"] < 1e-5
    assert l_rep["lindblad_min_eigenvalue"] >= -1e-6

    t_rep = audit_isometric_parallel_transport(dev)
    assert abs(t_rep["real_pt_norm_ratio"] - 1.0) < 1e-5
    assert t_rep["real_pt_sym_tangent_err"] < 1e-5
    assert abs(t_rep["complex_pt_norm_ratio"] - 1.0) < 1e-5
    assert t_rep["complex_pt_kraus_gauge_leak"] < 1e-6


def test_wave68_variants_forward_backward_and_optimizer_step() -> None:
    """Smoke-test 1 forward+backward+optimizer step across `W68_L1` and `W68_L3` (`L=2` override)."""
    dev = torch.device("cpu")
    idx = torch.randint(0, 50257, (2, 16), device=dev)
    targets = torch.randint(0, 50257, (2, 16), device=dev)

    for mode in (
        Wave68VariantMode.W68_L1_COMPLEMENTARY_LUDERS_512D_4L,
        Wave68VariantMode.W68_L3_DEEP_8L_CONSOLIDATED_512D,
    ):
        spec = build_wave68_variant_spec(mode, num_layers_override=2)
        model = Complementary512DLindbladGPT(spec).to(dev)
        model.initialize_model_()
        vocab_opt, trunk_opt, scalar_opt = build_wave68_optimizers(model)

        loss, metrics = model(idx, targets, return_metrics=True)
        assert torch.isfinite(loss)
        loss.backward()

        act_idx = torch.unique(torch.cat([idx.reshape(-1), targets.reshape(-1)], dim=0))
        vocab_opt.step_active(act_idx, lr_scale=1.0)
        trunk_opt.step()
        scalar_opt.step_active(active_idx=None, lr_scale=1.0)
        model.restrict_scalars_()

        g_diag = trunk_opt.last_gauge_diagnostics
        assert g_diag.get("svd_degenerate_spin_frac", 0.0) < 1e-5
        assert g_diag.get("su16_step_phase_frac", 0.0) < 1e-5
        assert g_diag.get("kraus_step_gauge_frac", 0.0) < 1e-5
        assert model.current_mub_unbiasedness_error() < 1e-5
        assert metrics["analytical_beta"] > 0.0
