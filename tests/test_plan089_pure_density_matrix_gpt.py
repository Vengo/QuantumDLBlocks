r"""Unit tests for Wave 69 (`EXPERIMENT_PLAN_089.md`): Pure Density-Matrix Architecture (`PureDensityMatrixGPT`)."""

from __future__ import annotations

import torch

from qdlblocks.benchmarks.pure_density_matrix_gpt_lab import (
    PureDensityMatrixGPT, Wave69VariantMode, build_wave69_optimizers, build_wave69_variant_spec,
)
from scripts.pure_density_matrix_inspector import (
    verify_theorem6_hilbert_schmidt_and_gauge,
    verify_theorem7_quantum_or_and_schur_closure,
    verify_theorem8_online_ema_and_stateful_history,
    verify_unconstrained_born_readout_and_model_step,
)


def _get_test_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_theorem6_hilbert_schmidt_isometric_and_gauge_invariance() -> None:
    """Verify Theorem 6: flattened outer-product Hilbert-Schmidt similarity, U(R) gauge invariance, and exp-free positivity."""
    res = verify_theorem6_hilbert_schmidt_and_gauge(_get_test_device())
    assert res["passed"], f"Theorem 6 check failed: {res}"
    assert res["isometric_overlap_rel_err"] <= 1e-5
    assert res["ur_gauge_invariance_rel_err"] <= 1e-5
    assert res["min_attention_weight"] >= 0.0
    assert res["causal_upper_tri_leak"] <= 1e-8


def test_theorem7_quantum_or_and_schur_product_closure() -> None:
    """Verify Theorem 7: Quantum OR + AND (Schur Product) preserves PSD, Hermiticity, and unit trace."""
    res = verify_theorem7_quantum_or_and_schur_closure(_get_test_device())
    assert res["passed"], f"Theorem 7 check failed: {res}"
    assert res["orthostochastic_row_sum_err"] <= 1e-5
    assert res["unit_trace_err"] <= 1e-5
    assert res["min_psd_eigenvalue"] >= -1e-5


def test_theorem8_online_ema_ngram_and_stateful_history_density() -> None:
    """Verify Theorem 8: OnlineEmaNgramDensityBank and StatefulQuantumDensityHistoryBuffer preserve PSD and unit trace."""
    res = verify_theorem8_online_ema_and_stateful_history(_get_test_device())
    assert res["passed"], f"Theorem 8 check failed: {res}"
    assert res["online_ema_trace_err"] <= 1e-5
    assert res["history_trace_err"] <= 1e-5


def test_unconstrained_born_readout_and_quotient_step() -> None:
    """Verify Path E & Rule 9: _UnconstrainedDualSubspace512DCEFn backward accuracy and 50,257-way normalization."""
    res = verify_unconstrained_born_readout_and_model_step(_get_test_device())
    assert res["passed"], f"Unconstrained Born readout check failed: {res}"
    assert res["custom_autograd_max_rel_err"] <= 1e-4
    assert res["vocab_50257_prob_sum_err"] <= 1e-4
    assert res["su16_step_phase_frac"] <= 1e-6
    assert res["kraus_step_gauge_frac"] <= 1e-6


def test_all_wave69_variants_forward_backward() -> None:
    """Verify all 4 Wave 69 variants (W69_P0..W69_P3) execute forward + backward + optimizer step cleanly."""
    dev = _get_test_device()
    x = torch.randint(0, 50257, (2, 16), device=dev, dtype=torch.long)
    y = torch.randint(0, 50257, (2, 16), device=dev, dtype=torch.long)
    for mode in (
        Wave69VariantMode.W69_P0_PURE_DENSITY_HS_ATTN_4L,
        Wave69VariantMode.W69_P1_PURE_DENSITY_UNCONSTRAINED_BORN_4L,
        Wave69VariantMode.W69_P2_PURE_DENSITY_EMA_NGRAM_HISTORY_4L,
        Wave69VariantMode.W69_P3_PURE_DENSITY_DEEP_8L_CONSOLIDATED,
    ):
        spec = build_wave69_variant_spec(mode, num_layers_override=2, seed=1337)
        model = PureDensityMatrixGPT(spec).to(dev)
        trunk_opt, vocab_opt, scalar_opt = build_wave69_optimizers(model)
        act_idx = torch.unique(torch.cat([x.reshape(-1), y.reshape(-1)], dim=0))
        loss, metrics = model(x, targets=y, return_metrics=True)
        assert torch.isfinite(loss)
        assert "analytical_beta" in metrics
        loss.backward()
        trunk_opt.step()
        vocab_opt.step_active(act_idx, lr_scale=1.0)
        scalar_opt.step_active(active_idx=None, lr_scale=1.0)
        model.restrict_scalars_()
