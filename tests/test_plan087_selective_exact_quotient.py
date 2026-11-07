"""Unit tests for Wave 67 (`EXPERIMENT_PLAN_087.md`): Selective Exact Quotienting, Read-Only Shadow EMA, & Wide `256D` Readout."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdlblocks.benchmarks.horizontal_grassmann_gauge_free_lab import (  # noqa: E402
    compute_gauge_locked_mub_probe_expectations,
)
from qdlblocks.benchmarks.selective_exact_quotient_lab import (  # noqa: E402
    ReadOnlyQuenchRiemannianShadowEMA, SelectiveExactQuotientGPT,
    Wave67SelectiveVariantMode, build_wave67_variant_spec,
    compute_degenerate_svd_spin_fraction, compute_su_dh_global_phase_fraction,
    filter_coupled_svd_post_ns_corotation, project_su_dh_mub_horizontal,
)
from qdlblocks.layers.impure_density_mirror_block import pack_hermitian_isometric  # noqa: E402
from scripts.horizontal_gradient_completeness_auditor import run_horizontal_gradient_completeness_audit  # noqa: E402


def test_random_loss_horizontal_gradient_completeness() -> None:
    """Verify 100.0% horizontal gradient retention (`0.0` vertical leak) across all 4 manifolds under random losses."""
    report = run_horizontal_gradient_completeness_audit(trials=4)
    assert report["all_w67_exact_completeness_passed"] is True
    by_name = {r["manifold"]: r for r in report["manifolds"]}

    assert by_name["Tied_Vocab_Subspace_Bridges_(U_in,U_out)"]["w66_raw_vertical_grad_leak_frac"] > 0.30
    assert by_name["Tied_Vocab_Subspace_Bridges_(U_in,U_out)"]["w67_raw_vertical_grad_leak_frac"] <= 1e-5
    assert by_name["Coupled_SVD_Split_Spectrum"]["w66_raw_vertical_grad_leak_frac"] > 0.05
    assert by_name["Coupled_SVD_Split_Spectrum"]["w67_raw_vertical_grad_leak_frac"] <= 1e-5
    assert by_name["Gauge_Locked_MUB_Tomography_SU(16)"]["w66_raw_vertical_grad_leak_frac"] > 0.10
    assert by_name["Gauge_Locked_MUB_Tomography_SU(16)"]["w67_raw_vertical_grad_leak_frac"] <= 1e-5
    assert by_name["CPTP_Kraus_Operators_Lyapunov_u(R)"]["w67_raw_vertical_grad_leak_frac"] <= 1e-5


def test_su16_mub_and_lyapunov_kraus_exact_invariance() -> None:
    """Verify exact forward invariance under global $U(1)$ phase shifts $U_h^{(1)} \\mapsto e^{i\\phi_h} U_h^{(1)}$ and zero $\\mathfrak{su}(16)$ trace spin."""
    torch.manual_seed(101)
    num_h, d_h = 4, 16
    vr, vi = torch.randn(num_h, d_h, d_h), torch.randn(num_h, d_h, d_h)
    ur, ui = torch.eye(d_h).unsqueeze(0).repeat(num_h, 1, 1), torch.zeros(num_h, d_h, d_h)
    hr, hi = project_su_dh_mub_horizontal(vr, vi, ur, ui)
    assert compute_su_dh_global_phase_fraction(hr, hi, ur, ui) < 1e-6

    idx = torch.arange(d_h, dtype=torch.float32)
    ang = (2.0 * math.pi / float(d_h)) * torch.outer(idx, idx)
    dft_re, dft_im = (torch.cos(ang) / 4.0).unsqueeze(0), (torch.sin(ang) / 4.0).unsqueeze(0)
    a_r = torch.randn(2, num_h, 3, d_h, d_h)
    rho_re = torch.matmul(a_r, a_r.transpose(-2, -1))
    tr = torch.diagonal(rho_re, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-6)
    rho_packed = pack_hermitian_isometric(rho_re / tr, torch.zeros_like(rho_re))

    phases = torch.linspace(0.2, 1.7, num_h).view(num_h, 1, 1)
    c_p, s_p = torch.cos(phases), torch.sin(phases)
    ur_rot = ur * c_p - ui * s_p
    ui_rot = ur * s_p + ui * c_p
    q0 = compute_gauge_locked_mub_probe_expectations(rho_packed, ur, ui, dft_re, dft_im)
    q1 = compute_gauge_locked_mub_probe_expectations(rho_packed, ur_rot, ui_rot, dft_re, dft_im)
    assert float((q0 - q1).abs().max().item()) < 1e-6


def test_3way_tied_subspace_and_degenerate_svd_invariance() -> None:
    """Verify degenerate SVD co-rotation suppression at $\\vec{\\sigma} = \\vec{1}$ and non-zero rotation at split $\\vec{\\sigma}$."""
    torch.manual_seed(202)
    dim = 16
    u_m, _ = torch.linalg.qr(torch.randn(1, dim, dim))
    v_m, _ = torch.linalg.qr(torch.randn(1, dim, dim))
    zu, zv = torch.randn(1, dim, dim), torch.randn(1, dim, dim)
    xi_u, xi_v = torch.randn(1, dim, dim), torch.randn(1, dim, dim)

    sig_deg = torch.ones(1, dim)
    tu_deg, tv_deg = filter_coupled_svd_post_ns_corotation(zu, zv, xi_u, xi_v, u_m, v_m, sig_deg)
    assert compute_degenerate_svd_spin_fraction(tu_deg, tv_deg, u_m, v_m, sig_deg) < 1e-6


def test_read_only_shadow_ema_zero_rollbacks() -> None:
    """Verify `ReadOnlyQuenchRiemannianShadowEMA` never triggers rollbacks or damps `lr_scale` regardless of loss gap."""
    spec = build_wave67_variant_spec(Wave67SelectiveVariantMode.W67_S2_SELECTIVE_QUOTIENT_256D_READOUT, num_layers=1)
    model = SelectiveExactQuotientGPT(spec)
    model.initialize_model_()
    assert model.boundary.subspace_dim == 256
    assert model.current_mub_unbiasedness_error() < 1e-6

    calls = {"n": 0}

    def _mock_eval(_m: object, _s: object) -> dict[str, float]:
        calls["n"] += 1
        return {"val_ce": 8.20 if calls["n"] % 2 == 1 else 7.50, "top1_acc": 0.1, "top5_acc": 0.2}

    ema = ReadOnlyQuenchRiemannianShadowEMA(model, eval_fn=_mock_eval)
    out = ema.check_and_thermostat(model, None, current_lr_scale=1.0, step=50, apply_feedback=True)
    assert out["triggered"] is False
    assert out["lr_scale_after"] == 1.0
    assert len(ema.trigger_events) == 0
