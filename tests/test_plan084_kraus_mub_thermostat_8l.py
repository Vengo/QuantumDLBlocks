r"""Unit Tests for `EXPERIMENT_PLAN_084.md` (`Wave 64`): Analytical Parseval Temperature, Magnitude-Gated Muon, Two-Basis $U(16)$ MUB Tomography, Rank-2 CPTP Stiefel Kraus Channels, `L = 8` Highway, & Closed-Loop Shadow-EMA Thermostat."""

from __future__ import annotations

import torch

from qdlblocks.benchmarks.kraus_mub_thermostat_8l_lab import (
    DEFAULT_ANALYTICAL_BETA_FAST_TOKENS, DEFAULT_ANALYTICAL_BETA_INIT, DEFAULT_ANALYTICAL_BETA_MID,
    DEFAULT_ANALYTICAL_BETA_TARGET, DEFAULT_ANALYTICAL_BETA_WARMUP_TOKENS, DutyMatchedGroupMultipliers,
    KrausMUBThermostatGPT, KrausMUBThermostatSpec, KrausMUBVariantMode, RiemannianShadowEMAThermostat,
    build_initial_kraus_stiefel_operators, build_kraus_mub_optimizers, build_mub_two_basis_unitary_probes,
    compute_analytical_parseval_beta, compute_mub_two_basis_probe_expectations,
    conjugate_packed_hermitian_by_kraus_stiefel, initialize_kraus_mub_model_, step_kraus_mub_optimizers,
)
from qdlblocks.layers.impure_density_mirror_block import complex_l2_normalize, pack_rank1_ket_outer_isometric


def test_analytical_parseval_beta_schedule_locks_at_10() -> None:
    """Verify deterministic token-indexed Parseval ramp (`0.50 -> 4.0 -> 10.0`) with zero drift after `1.23M` tokens."""
    assert abs(compute_analytical_parseval_beta(0) - DEFAULT_ANALYTICAL_BETA_INIT) < 1e-6
    assert abs(compute_analytical_parseval_beta(DEFAULT_ANALYTICAL_BETA_FAST_TOKENS) - DEFAULT_ANALYTICAL_BETA_MID) < 1e-6
    assert abs(compute_analytical_parseval_beta(DEFAULT_ANALYTICAL_BETA_WARMUP_TOKENS) - DEFAULT_ANALYTICAL_BETA_TARGET) < 1e-6
    assert abs(compute_analytical_parseval_beta(100_000_000) - DEFAULT_ANALYTICAL_BETA_TARGET) < 1e-6


def test_mub_two_basis_unitary_probes_exact_overlap_and_trace_resolution() -> None:
    r"""Verify $C_h^{(m)} \in U(16)$, $|\langle c_{j,h}^{(1)} \mid c_{k,h}^{(2)} \rangle|^2 = 1/16$, and $\sum_{k=1}^{32} q_{t,h,k} = 2\operatorname{Tr}(\rho_{t,h})$."""
    u1_re, u1_im, u2_re, u2_im = build_mub_two_basis_unitary_probes(num_heads=16, d_h=16, seed=42)
    eye16 = torch.eye(16, dtype=torch.float32).unsqueeze(0)
    for ur, ui in [(u1_re, u1_im), (u2_re, u2_im)]:
        gr = torch.matmul(ur, ur.transpose(-2, -1)) + torch.matmul(ui, ui.transpose(-2, -1))
        gi = torch.matmul(ui, ur.transpose(-2, -1)) - torch.matmul(ur, ui.transpose(-2, -1))
        err = float(torch.sqrt((gr - eye16).square().sum(dim=(-2, -1)) + gi.square().sum(dim=(-2, -1))).max().item())
        assert err < 1e-5

    ov_re = torch.matmul(u1_re, u2_re.transpose(-2, -1)) + torch.matmul(u1_im, u2_im.transpose(-2, -1))
    ov_im = torch.matmul(u1_im, u2_re.transpose(-2, -1)) - torch.matmul(u1_re, u2_im.transpose(-2, -1))
    mod_sq = ov_re.square() + ov_im.square()
    assert float((mod_sq - (1.0 / 16.0)).abs().max().item()) < 1e-5

    psi = torch.randn(2, 16, 4, 2, 16, dtype=torch.float32)
    pr, pi = complex_l2_normalize(psi[..., 0, :], psi[..., 1, :])
    rho_packed = pack_rank1_ket_outer_isometric(pr, pi)
    q_512 = compute_mub_two_basis_probe_expectations(rho_packed, u1_re, u1_im, u2_re, u2_im)
    q_per_head_sum = q_512.view(2, 4, 16, 32).sum(dim=-1)
    assert float((q_per_head_sum - 2.0).abs().max().item()) < 1e-4


def test_rank2_cptp_stiefel_kraus_channel_preserves_trace_and_hermiticity() -> None:
    r"""Verify $\mathbf{K}_h \in \operatorname{St}_{\mathbb{C}}(32, 16)$ satisfies $\sum_{r=1}^2 K_{h,r}^\dagger K_{h,r} = I_{16}$ and $\operatorname{Tr}(\mathcal{E}_h(\rho)) = 1$."""
    k_re, k_im = build_initial_kraus_stiefel_operators(num_heads=16, d_h=16, kraus_rank=2)
    assert k_re.shape == (16, 32, 16)
    eye16 = torch.eye(16, dtype=torch.float32).unsqueeze(0)
    gram_re = torch.matmul(k_re.transpose(-2, -1), k_re) + torch.matmul(k_im.transpose(-2, -1), k_im)
    gram_im = torch.matmul(k_re.transpose(-2, -1), k_im) - torch.matmul(k_im.transpose(-2, -1), k_re)
    assert float(torch.sqrt((gram_re - eye16).square().sum() + gram_im.square().sum()).item()) < 1e-5

    psi = torch.randn(2, 16, 5, 2, 16, dtype=torch.float32)
    pr, pi = complex_l2_normalize(psi[..., 0, :], psi[..., 1, :])
    rho_packed = pack_rank1_ket_outer_isometric(pr, pi)
    _, rot_re, rot_im = conjugate_packed_hermitian_by_kraus_stiefel(rho_packed, k_re, k_im)
    tr = torch.diagonal(rot_re, dim1=-2, dim2=-1).sum(dim=-1)
    assert float((tr - 1.0).abs().max().item()) < 1e-5
    assert float((rot_re - rot_re.transpose(-2, -1)).abs().max().item()) < 1e-6
    assert float((rot_im + rot_im.transpose(-2, -1)).abs().max().item()) < 1e-6


def test_kraus_mub_8l_step_and_shadow_ema_retraction() -> None:
    """Verify `L = 8` `KrausMUBThermostatGPT` steps cleanly with multi-shape `(16, 32, 16)` + `(16, 16, 16)` complex operators."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = KrausMUBThermostatSpec(
        variant_mode=KrausMUBVariantMode.W64_A3_FULL_CONSOLIDATED_8L,
        num_layers=8, kraus_rank=2, use_mub_unitary_probes=True,
    )
    model = KrausMUBThermostatGPT(spec).to(device=device)
    initialize_kraus_mub_model_(model)
    optimizers = build_kraus_mub_optimizers(model, mults=DutyMatchedGroupMultipliers())
    thermostat = RiemannianShadowEMAThermostat(model)

    idx = torch.randint(0, 50257, (2, 16), device=device, dtype=torch.long)
    targets = torch.randint(0, 50257, (2, 16), device=device, dtype=torch.long)
    for _ in range(2):
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss = model.forward_loss(idx, targets)
        assert torch.isfinite(loss).all()
        loss.backward()
        active_ids = torch.unique(torch.cat([idx.reshape(-1), targets.reshape(-1)]))
        step_kraus_mub_optimizers(model, optimizers, active_ids=active_ids, tokens_in_step=163840, lr_mult=1.0, mom_beta=0.80)
        thermostat.update_after_step(model)

    assert abs(model.current_analytical_beta() - compute_analytical_parseval_beta(327680)) < 1e-5
    thermostat.retract_shadow_ema_manifold_(hot_model=model)
    telem = thermostat.ema_model.get_manifold_and_quantum_telemetry()
    assert telem["kraus_stiefel_err"] < 1e-5
    assert telem["probe_unitary_ortho_err"] < 1e-5
    for p_st in thermostat.ema_model.real_stiefel_parameters():
        wf = p_st.detach().float()
        if wf.size(-2) < wf.size(-1):
            wf = wf.transpose(-2, -1)
        cols = wf.size(-1)
        eye_c = torch.eye(cols, device=wf.device, dtype=torch.float32)
        ortho_err = float((torch.matmul(wf.t(), wf) - eye_c).norm().item() / (cols ** 0.5))
        assert ortho_err < 1e-5, f"Real Stiefel matrix shape {tuple(p_st.shape)} had ortho_err={ortho_err}"
