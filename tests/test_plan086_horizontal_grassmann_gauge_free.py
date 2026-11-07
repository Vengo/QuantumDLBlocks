"""Unit tests for Wave 66 (`EXPERIMENT_PLAN_086.md`): Horizontal-Grassmannian & Gauge-Free Quantum Density-Matrix Architecture."""

from __future__ import annotations

import math

import torch

from qdlblocks.benchmarks.horizontal_grassmann_gauge_free_lab import (
    HorizontalGrassmannGaugeFreeGPT,
    SpikeGuardedRiemannianShadowEMAThermostat,
    Wave66GaugeVariantMode,
    build_wave66_gauge_free_optimizers,
    build_wave66_variant_spec,
    compute_coupled_svd_joint_spin_fraction,
    compute_cp15_diag_phase_fraction,
    compute_grassmann_vertical_spin_fraction,
    compute_kraus_branch_gauge_fraction,
    evaluate_wave66_model,
    project_coupled_svd_pair_horizontal,
    project_cp15_flag_horizontal,
    project_kraus_branch_gauge_horizontal,
    project_real_grassmann_horizontal,
    step_wave66_optimizers,
)


def test_real_grassmann_horizontal_projection_eliminates_vertical_spin() -> None:
    """Verify $G_{\\text{Grassmann}} = (I_D - V V^\\top) G$ strictly annihilates $V^\\top G_{\\text{Grassmann}} = 0$."""
    torch.manual_seed(42)
    q_raw = torch.randn(4, 64, 16)
    v_stiefel, _ = torch.linalg.qr(q_raw)
    g_raw = torch.randn(4, 64, 16)

    raw_spin = compute_grassmann_vertical_spin_fraction(g_raw, v_stiefel)
    assert raw_spin > 0.01

    g_horiz = project_real_grassmann_horizontal(g_raw, v_stiefel)
    vt_gh = torch.matmul(v_stiefel.transpose(-2, -1), g_horiz)
    assert float(vt_gh.abs().max().item()) < 1e-5
    assert compute_grassmann_vertical_spin_fraction(g_horiz, v_stiefel) < 1e-6


def test_coupled_svd_pair_horizontal_projection_eliminates_joint_or_gauge() -> None:
    """Verify coupled SVD horizontal projection annihilates $\\operatorname{skew}(U^\\top \\xi_U) + \\operatorname{skew}(V^\\top \\xi_V) = 0$."""
    torch.manual_seed(43)
    u_stiefel, _ = torch.linalg.qr(torch.randn(3, 64, 16))
    v_stiefel, _ = torch.linalg.qr(torch.randn(3, 48, 16))
    gu_raw = torch.randn(3, 64, 16)
    gv_raw = torch.randn(3, 48, 16)

    raw_joint_spin = compute_coupled_svd_joint_spin_fraction(gu_raw, gv_raw, u_stiefel, v_stiefel)
    assert raw_joint_spin > 0.01

    xi_u, xi_v = project_coupled_svd_pair_horizontal(gu_raw, gv_raw, u_stiefel, v_stiefel)
    joint_spin_after = compute_coupled_svd_joint_spin_fraction(xi_u, xi_v, u_stiefel, v_stiefel)
    assert joint_spin_after < 1e-6


def test_cp15_flag_and_kraus_branch_gauge_projections() -> None:
    """Verify $\\mathbb{CP}^{15}$ Flag projection zeroes diagonal phase spin and Kraus quotient zeroes $\\mathfrak{u}(R)$ spin."""
    torch.manual_seed(44)
    z_mub = torch.complex(torch.randn(4, 16, 16), torch.randn(4, 16, 16))
    q_mub, _ = torch.linalg.qr(z_mub)
    u_re, u_im = q_mub.real.contiguous(), q_mub.imag.contiguous()
    gr_mub, gi_mub = torch.randn_like(u_re), torch.randn_like(u_im)

    assert compute_cp15_diag_phase_fraction(gr_mub, gi_mub, u_re, u_im) > 0.01
    xi_re, xi_im = project_cp15_flag_horizontal(gr_mub, gi_mub, u_re, u_im)
    assert compute_cp15_diag_phase_fraction(xi_re, xi_im, u_re, u_im) < 1e-6

    z_kr = torch.complex(torch.randn(2, 64, 16), torch.randn(2, 64, 16))
    q_kr, _ = torch.linalg.qr(z_kr)
    k_re, k_im = q_kr.real.contiguous(), q_kr.imag.contiguous()
    h_skew_re = torch.randn(2, 4, 4)
    h_skew_re = 0.5 * (h_skew_re - h_skew_re.transpose(-2, -1))
    h_skew_im = torch.randn(2, 4, 4)
    h_skew_im = 0.5 * (h_skew_im + h_skew_im.transpose(-2, -1))
    k_br_re = k_re.view(2, 4, 256)
    k_br_im = k_im.view(2, 4, 256)
    g_pure_re = (torch.matmul(h_skew_re, k_br_re) - torch.matmul(h_skew_im, k_br_im)).reshape(2, 64, 16)
    g_pure_im = (torch.matmul(h_skew_re, k_br_im) + torch.matmul(h_skew_im, k_br_re)).reshape(2, 64, 16)

    kr_horiz_re, kr_horiz_im = project_kraus_branch_gauge_horizontal(
        g_pure_re + 0.25 * torch.randn_like(k_re),
        g_pure_im + 0.25 * torch.randn_like(k_im),
        k_re,
        k_im,
    )
    assert compute_kraus_branch_gauge_fraction(kr_horiz_re, kr_horiz_im, k_re, k_im) < 1e-5


def test_wave66_variants_forward_backward_mub_lock_and_spike_thermostat() -> None:
    """Verify all 4 Wave 66 variants execute cleanly and that W66_G2/W66_G3 preserve exact MUB unbiasedness."""
    torch.manual_seed(1337)
    idx = torch.randint(0, 50257, (2, 32))
    targets = torch.randint(0, 50257, (2, 32))
    val_batches = [(idx, targets)]

    for mode in (
        Wave66GaugeVariantMode.W66_G0_STIEFEL_AMBIENT_BASELINE,
        Wave66GaugeVariantMode.W66_G1_HORIZONTAL_GRASSMANN_VOCAB_AND_BRIDGES,
        Wave66GaugeVariantMode.W66_G2_GAUGE_LOCKED_CP15_MUB_AND_KRAUS,
        Wave66GaugeVariantMode.W66_G3_FULL_GAUGE_FREE_SPIKE_ONLY_THERMOSTAT,
    ):
        spec = build_wave66_variant_spec(mode, num_layers=2, seed=1337)
        model = HorizontalGrassmannGaugeFreeGPT(spec)
        model.initialize_model_()
        optimizers = build_wave66_gauge_free_optimizers(model)
        q_trunk_opt = optimizers[0]
        thermostat = SpikeGuardedRiemannianShadowEMAThermostat(model, eval_fn=evaluate_wave66_model)

        for _ in range(2):
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss, _ = model(idx, targets)
            assert math.isfinite(float(loss.item()))
            loss.backward()
            step_wave66_optimizers(model, optimizers, tokens_in_step=int(targets.numel()))
            thermostat.update_after_step(model)

        gd = q_trunk_opt.last_gauge_diagnostics
        if spec.enable_horizontal_grassmann:
            assert gd["grassmann_step_spin_frac"] < 1e-5
            assert gd["coupled_svd_step_spin_frac"] < 1e-5
        else:
            assert gd["grassmann_step_spin_frac"] > 1e-4

        if spec.enable_gauge_locked_mub_kraus:
            assert model.current_mub_unbiasedness_error() < 1e-5
            assert gd["cp15_step_phase_frac"] < 1e-5
            assert gd["kraus_step_gauge_frac"] < 1e-4

        th_out = thermostat.check_and_thermostat(model, val_batches, current_lr_scale=1.0, step=2, apply_feedback=True)
        assert math.isfinite(th_out["hot_val_ce"])
        assert th_out["triggered"] is False
