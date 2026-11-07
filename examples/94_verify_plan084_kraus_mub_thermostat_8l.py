r"""Verification Script for `EXPERIMENT_PLAN_084.md` (`Wave 64`): Analytical Parseval Temperature ($\beta_0 = 10.0$), Magnitude-Gated Muon, Two-Basis $U(16)$ MUB Tomography, Rank-2 CPTP Stiefel Kraus Channels, `L = 8` Highway, & Closed-Loop Shadow-EMA Thermostat."""

from __future__ import annotations

import json
from pathlib import Path
import time

import torch

from qdlblocks.benchmarks.kraus_mub_thermostat_8l_lab import (
    DEFAULT_ANALYTICAL_BETA_FAST_TOKENS, DEFAULT_ANALYTICAL_BETA_INIT, DEFAULT_ANALYTICAL_BETA_MID,
    DEFAULT_ANALYTICAL_BETA_TARGET, DEFAULT_ANALYTICAL_BETA_WARMUP_TOKENS, DutyMatchedGroupMultipliers,
    KrausMUBThermostatGPT, KrausMUBThermostatSpec, KrausMUBVariantMode, RiemannianShadowEMAThermostat,
    build_initial_kraus_stiefel_operators, build_kraus_mub_optimizers, build_mub_two_basis_unitary_probes,
    compute_analytical_parseval_beta, conjugate_packed_hermitian_by_kraus_stiefel, initialize_kraus_mub_model_,
    step_kraus_mub_optimizers,
)
from qdlblocks.layers.impure_density_mirror_block import complex_l2_normalize, pack_rank1_ket_outer_isometric


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1337)

    # 1. Verify H1: Analytical Parseval Temperature Schedule & Zero Trainable Readout Temperature
    b0 = compute_analytical_parseval_beta(0)
    b_fast = compute_analytical_parseval_beta(DEFAULT_ANALYTICAL_BETA_FAST_TOKENS)
    b_warm = compute_analytical_parseval_beta(DEFAULT_ANALYTICAL_BETA_WARMUP_TOKENS)
    b_post = compute_analytical_parseval_beta(DEFAULT_ANALYTICAL_BETA_WARMUP_TOKENS * 10)
    assert abs(b0 - DEFAULT_ANALYTICAL_BETA_INIT) < 1e-6
    assert abs(b_fast - DEFAULT_ANALYTICAL_BETA_MID) < 1e-6
    assert abs(b_warm - DEFAULT_ANALYTICAL_BETA_TARGET) < 1e-6
    assert abs(b_post - DEFAULT_ANALYTICAL_BETA_TARGET) < 1e-6

    # 2. Verify H3a: Two-Basis U(16) DFT Mutually Unbiased Bases (MUB) Probes
    u1_re, u1_im, u2_re, u2_im = build_mub_two_basis_unitary_probes(16, 16, seed=1337, device=device)
    eye16 = torch.eye(16, device=device, dtype=torch.float32).unsqueeze(0)
    g1_re = torch.matmul(u1_re, u1_re.transpose(-2, -1)) + torch.matmul(u1_im, u1_im.transpose(-2, -1))
    g1_im = torch.matmul(u1_im, u1_re.transpose(-2, -1)) - torch.matmul(u1_re, u1_im.transpose(-2, -1))
    u1_ortho_err = float(torch.sqrt((g1_re - eye16).square().sum(dim=(-2, -1)) + g1_im.square().sum(dim=(-2, -1))).max().item())
    ov_re = torch.matmul(u1_re, u2_re.transpose(-2, -1)) + torch.matmul(u1_im, u2_im.transpose(-2, -1))
    ov_im = torch.matmul(u1_im, u2_re.transpose(-2, -1)) - torch.matmul(u1_re, u2_im.transpose(-2, -1))
    mub_mod_sq = ov_re.square() + ov_im.square()
    mub_max_dev = float((mub_mod_sq - (1.0 / 16.0)).abs().max().item())
    assert u1_ortho_err < 1e-5, f"U(16) ortho error too large: {u1_ortho_err}"
    assert mub_max_dev < 1e-5, f"MUB overlap deviation from 1/16 too large: {mub_max_dev}"

    # 3. Verify H3b: Rank-2 CPTP Stiefel Kraus Channel Trace Preservation
    k_re, k_im = build_initial_kraus_stiefel_operators(16, 16, kraus_rank=2, device=device)
    assert k_re.shape == (16, 32, 16) and k_im.shape == (16, 32, 16)
    k_gram_re = torch.matmul(k_re.transpose(-2, -1), k_re) + torch.matmul(k_im.transpose(-2, -1), k_im)
    k_gram_im = torch.matmul(k_re.transpose(-2, -1), k_im) - torch.matmul(k_im.transpose(-2, -1), k_re)
    kraus_stiefel_err = float(torch.sqrt((k_gram_re - eye16).square().sum(dim=(-2, -1)) + k_gram_im.square().sum(dim=(-2, -1))).max().item())
    assert kraus_stiefel_err < 1e-5, f"Rank-2 Kraus Stiefel error too large: {kraus_stiefel_err}"

    psi_raw = torch.randn(2, 16, 8, 2, 16, device=device, dtype=torch.float32)
    pr, pi = complex_l2_normalize(psi_raw[..., 0, :], psi_raw[..., 1, :])
    rho_in = pack_rank1_ket_outer_isometric(pr, pi)
    _, rho_out_re, _ = conjugate_packed_hermitian_by_kraus_stiefel(rho_in, k_re, k_im)
    traces = torch.diagonal(rho_out_re, dim1=-2, dim2=-1).sum(dim=-1)
    max_trace_err = float((traces - 1.0).abs().max().item())
    assert max_trace_err < 1e-5, f"CPTP Kraus trace error too large: {max_trace_err}"

    # 4. Verify L=8 Consolidated Model + Magnitude-Gated Multi-Shape Stiefel/Kraus/U(16) Step + Thermostat Retraction
    spec_8l = KrausMUBThermostatSpec(
        variant_mode=KrausMUBVariantMode.W64_A3_FULL_CONSOLIDATED_8L,
        num_layers=8, kraus_rank=2, use_mub_unitary_probes=True,
    )
    model = KrausMUBThermostatGPT(spec_8l).to(device=device)
    init_ms, init_meta = initialize_kraus_mub_model_(model)
    assert not hasattr(model, "readout_temp") or getattr(model, "readout_temp", None) is None
    optimizers = build_kraus_mub_optimizers(model, mults=DutyMatchedGroupMultipliers())
    thermostat = RiemannianShadowEMAThermostat(model)

    idx = torch.randint(0, 50257, (2, 32), device=device, dtype=torch.long)
    targets = torch.randint(0, 50257, (2, 32), device=device, dtype=torch.long)
    for step in range(1, 4):
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss = model.forward_loss(idx, targets)
        assert torch.isfinite(loss).all()
        loss.backward()
        active_ids = torch.unique(torch.cat([idx.reshape(-1), targets.reshape(-1)]))
        step_kraus_mub_optimizers(model, optimizers, active_ids=active_ids, tokens_in_step=8192, lr_mult=1.0, mom_beta=0.80)
        thermostat.update_after_step(model)

    thermostat.retract_shadow_ema_manifold_()
    telem_hot = model.get_manifold_and_quantum_telemetry()
    telem_ema = thermostat.ema_model.get_manifold_and_quantum_telemetry()
    assert telem_hot["kraus_stiefel_err"] < 1e-2
    assert telem_hot["probe_unitary_ortho_err"] < 1e-2
    assert telem_ema["kraus_stiefel_err"] < 1e-2
    assert telem_ema["probe_unitary_ortho_err"] < 1e-2

    summary = {
        "plan": "EXPERIMENT_PLAN_084",
        "wave": "W64",
        "status": "PASS",
        "device": str(device),
        "init_ms": round(init_ms, 2),
        "init_meta": init_meta,
        "analytical_beta_schedule": {"step0": b0, "fast_163k": b_fast, "warmup_1p23m": b_warm, "locked_post": b_post},
        "mub_u16_ortho_err": round(u1_ortho_err, 7),
        "mub_cross_basis_max_dev_from_1_16": round(mub_max_dev, 7),
        "rank2_kraus_stiefel_err_step0": round(kraus_stiefel_err, 7),
        "rank2_kraus_cptp_trace_err": round(max_trace_err, 7),
        "post_3step_hot_telemetry": telem_hot,
        "post_3step_ema_telemetry": telem_ema,
        "elapsed_sec": round(time.perf_counter() - t0, 3),
    }
    out_path = Path(".cache/verify_plan084_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
