#!/usr/bin/env python3
"""Wave 71 (`EXPERIMENT_PLAN_091.md`) Sensory-Anchored Bipartite Density Inspector.

Verifies:
1. Theorem 12a (Unanchored Memoryless State-Collision Witness: `unanchored_collision_dist == 0`).
2. Theorem 12b (Dual-Unitary Sensory-Anchored Distinguishability & CPTP Unit Trace).
3. Theorem 12c (Bipartite Register Trace Invariance & Step-0 Self-Copy Attractor Elimination).
4. `AnchoredBipartitePureDensityGPT` Forward/Backward & `PureDensityQuotientMuon` Quotient Invariants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from qdlblocks.benchmarks.anchored_bipartite_density_gpt_lab import (
    AnchoredBipartitePureDensityGPT,
    Wave71VariantMode,
    _build_deterministic_haar_unitary_stack,
    apply_bipartite_sensory_predictive_register_split,
    apply_dual_unitary_sensory_anchored_qkv,
    build_wave71_variant_spec,
)
from qdlblocks.benchmarks.factored_kraus_density_gpt_lab import compute_restored_riemannian_momentum_schedule
from qdlblocks.benchmarks.pure_density_matrix_gpt_lab import (
    _collect_active_vocab_indices_w68,
    build_wave69_optimizers,
    normalize_cholesky_factor_frobenius,
    pack_cholesky_factor_outer_isometric,
    unpack_hermitian_isometric,
)

TRACE_TOL: float = 1e-5
MIN_ANCHORED_DIST: float = 0.15


def verify_theorem12ab_collision_distinguishability_and_cptp(device: torch.device) -> dict[str, Any]:
    """Verify Theorem 12a (unanchored collision) and Theorem 12b (anchored distinguishability + CPTP) (`<= 8` args)."""
    torch.manual_seed(2026091)
    bsz, num_h, seq_len, d_h, r_rank = 2, 16, 8, 16, 4

    # Construct an identical evolved state L_* shared by two tokens A and B (state collision at Layer l)
    l_star_re = torch.randn(bsz, num_h, seq_len, d_h, r_rank, device=device, dtype=torch.float32)
    l_star_im = torch.randn(bsz, num_h, seq_len, d_h, r_rank, device=device, dtype=torch.float32)
    l_star_re, l_star_im = normalize_cholesky_factor_frobenius(l_star_re, l_star_im)

    # Distinct Layer-0 sensory anchors L_0(A) != L_0(B)
    l0_a_re = torch.randn(bsz, num_h, seq_len, d_h, r_rank, device=device, dtype=torch.float32)
    l0_a_im = torch.randn(bsz, num_h, seq_len, d_h, r_rank, device=device, dtype=torch.float32)
    l0_a_re, l0_a_im = normalize_cholesky_factor_frobenius(l0_a_re, l0_a_im)

    l0_b_re = torch.randn(bsz, num_h, seq_len, d_h, r_rank, device=device, dtype=torch.float32)
    l0_b_im = torch.randn(bsz, num_h, seq_len, d_h, r_rank, device=device, dtype=torch.float32)
    l0_b_re, l0_b_im = normalize_cholesky_factor_frobenius(l0_b_re, l0_b_im)

    u_evol_re, u_evol_im = _build_deterministic_haar_unitary_stack(num_h, d_h, 2026092)
    u_anc_re, u_anc_im = _build_deterministic_haar_unitary_stack(num_h, d_h, 2026093)
    u_evol_re, u_evol_im = u_evol_re.to(device), u_evol_im.to(device)
    u_anc_re, u_anc_im = u_anc_re.to(device), u_anc_im.to(device)
    anchor_logit = torch.zeros(num_h, device=device, dtype=torch.float32)

    # 1. Theorem 12a: Unanchored channel produces identical output for (L_*, L_0(A)) and (L_*, L_0(B))
    un_a_re, un_a_im = apply_dual_unitary_sensory_anchored_qkv(
        (l_star_re, l_star_im), (l0_a_re, l0_a_im), (u_evol_re, u_evol_im), None, None
    )
    un_b_re, un_b_im = apply_dual_unitary_sensory_anchored_qkv(
        (l_star_re, l_star_im), (l0_b_re, l0_b_im), (u_evol_re, u_evol_im), None, None
    )
    rho_un_a = pack_cholesky_factor_outer_isometric(un_a_re, un_a_im)
    rho_un_b = pack_cholesky_factor_outer_isometric(un_b_re, un_b_im)
    unanchored_collision_dist = float((rho_un_a - rho_un_b).norm(dim=-1).mean().item())
    unanchored_chol_dist = float(
        torch.sqrt((un_a_re - un_b_re).square().sum(dim=(-2, -1)) + (un_a_im - un_b_im).square().sum(dim=(-2, -1))).mean().item()
    )

    # 2. Theorem 12b: Dual-Unitary Sensory-Anchored channel separates (L_*, L_0(A)) from (L_*, L_0(B))
    anc_a_re, anc_a_im = apply_dual_unitary_sensory_anchored_qkv(
        (l_star_re, l_star_im), (l0_a_re, l0_a_im), (u_evol_re, u_evol_im), (u_anc_re, u_anc_im), anchor_logit
    )
    anc_b_re, anc_b_im = apply_dual_unitary_sensory_anchored_qkv(
        (l_star_re, l_star_im), (l0_b_re, l0_b_im), (u_evol_re, u_evol_im), (u_anc_re, u_anc_im), anchor_logit
    )
    rho_anc_a = pack_cholesky_factor_outer_isometric(anc_a_re, anc_a_im)
    rho_anc_b = pack_cholesky_factor_outer_isometric(anc_b_re, anc_b_im)
    anchored_rho_dist = float((rho_anc_a - rho_anc_b).norm(dim=-1).mean().item())
    anchored_chol_dist = float(
        torch.sqrt((anc_a_re - anc_b_re).square().sum(dim=(-2, -1)) + (anc_a_im - anc_b_im).square().sum(dim=(-2, -1))).mean().item()
    )

    # Verify CPTP unit trace and PSD spectrum of rho_anc_a
    frob_norm_sq = anc_a_re.square().sum(dim=(-2, -1)) + anc_a_im.square().sum(dim=(-2, -1))
    frob_unit_err = float((frob_norm_sq - 1.0).abs().max().item())
    rho_mat_re, rho_mat_im = unpack_hermitian_isometric(rho_anc_a, d_h)
    tr_err = float((torch.diagonal(rho_mat_re, dim1=-2, dim2=-1).sum(dim=-1) - 1.0).abs().max().item())
    rho_c = torch.complex(rho_mat_re, rho_mat_im)
    min_eig = float(torch.linalg.eigvalsh(rho_c).min().item())

    passed = (
        unanchored_collision_dist == 0.0
        and unanchored_chol_dist == 0.0
        and anchored_rho_dist > MIN_ANCHORED_DIST
        and anchored_chol_dist > MIN_ANCHORED_DIST
        and frob_unit_err < TRACE_TOL
        and tr_err < TRACE_TOL
        and min_eig >= -1e-6
    )
    return {
        "passed": bool(passed),
        "unanchored_collision_dist": round(unanchored_collision_dist, 8),
        "unanchored_chol_dist": round(unanchored_chol_dist, 8),
        "anchored_rho_dist": round(anchored_rho_dist, 6),
        "anchored_chol_dist": round(anchored_chol_dist, 6),
        "frob_unit_err": round(frob_unit_err, 8),
        "trace_err": round(tr_err, 8),
        "min_eigenvalue": round(min_eig, 8),
    }


def verify_theorem12c_bipartite_and_self_copy_elimination(device: torch.device) -> dict[str, Any]:
    """Verify Theorem 12c (bipartite register trace invariance + Step-0 self-copy attractor elimination) (`<= 8` args)."""
    torch.manual_seed(2026093)
    bsz, num_h, seq_len, d_h, r_rank = 2, 16, 16, 16, 4
    l_out_re, l_out_im = normalize_cholesky_factor_frobenius(
        torch.randn(bsz, num_h, seq_len, d_h, r_rank, device=device),
        torch.randn(bsz, num_h, seq_len, d_h, r_rank, device=device),
    )
    l0_re, l0_im = normalize_cholesky_factor_frobenius(
        torch.randn(bsz, num_h, seq_len, d_h, r_rank, device=device),
        torch.randn(bsz, num_h, seq_len, d_h, r_rank, device=device),
    )
    tether_logit = torch.zeros(num_h, device=device)
    lb_re, lb_im = apply_bipartite_sensory_predictive_register_split((l_out_re, l_out_im), (l0_re, l0_im), tether_logit)
    bipartite_frob_err = float(((lb_re.square().sum(dim=(-2, -1)) + lb_im.square().sum(dim=(-2, -1))) - 1.0).abs().max().item())

    # Sensory-register overlap between lb[:, :, :, :, 0:2] and l0[:, :, :, :, 0:2]
    sens_dot = (lb_re[..., :2] * l0_re[..., :2] + lb_im[..., :2] * l0_im[..., :2]).sum(dim=(-2, -1))
    sens_norm = torch.sqrt(
        (lb_re[..., :2].square().sum(dim=(-2, -1)) + lb_im[..., :2].square().sum(dim=(-2, -1)))
        * (l0_re[..., :2].square().sum(dim=(-2, -1)) + l0_im[..., :2].square().sum(dim=(-2, -1)))
    ).clamp_min(1e-8)
    sensory_cosine_fidelity = float((sens_dot / sens_norm).mean().item())

    # Compare Step-0 self-copy alignment of W71_A0 (coupled q_in == q_out, reading mode 0) vs W71_A2 (decoupled q_in != q_out, reading mode 2:4)
    idx = torch.arange(64, 64 + 32, device=device, dtype=torch.long).view(2, 16)
    spec_a0 = build_wave71_variant_spec(Wave71VariantMode.W71_A0_UNANCHORED_W70_REF_4L, seed=1337)
    spec_a2 = build_wave71_variant_spec(Wave71VariantMode.W71_A2_ANCHORED_BIPARTITE_DECOUPLED_4L, seed=1337)
    model_a0 = AnchoredBipartitePureDensityGPT(spec_a0).to(device).eval()
    model_a2 = AnchoredBipartitePureDensityGPT(spec_a2).to(device).eval()

    with torch.no_grad():
        # Measure direct boundary alignment between Layer-0 ket and readout ket for the same token x_t
        l0_a0_re, l0_a0_im, _ = model_a0.prepare_layer0_density_state(idx)
        ket_a0 = torch.stack([l0_a0_re[..., 0], l0_a0_im[..., 0]], dim=-2).permute(0, 2, 1, 3, 4).reshape(2, 16, 512).float()
        q_purif_a0 = F.rms_norm(torch.matmul(ket_a0, model_a0.q_readout_mix.float()), (512,), eps=1e-6)
        proj_a0 = F.normalize(torch.matmul(q_purif_a0, model_a0.boundary.q_out_ortho.float()[:, :256]), dim=-1)

        l0_a2_re, l0_a2_im, _ = model_a2.prepare_layer0_density_state(idx)
        ket_a2 = torch.stack([l0_a2_re[..., 2], l0_a2_im[..., 2]], dim=-2).permute(0, 2, 1, 3, 4).reshape(2, 16, 512).float()
        q_purif_a2 = F.rms_norm(torch.matmul(ket_a2, model_a2.q_readout_mix.float()), (512,), eps=1e-6)
        proj_a2 = F.normalize(torch.matmul(q_purif_a2, model_a2.boundary.q_out_ortho.float()[:, :256]), dim=-1)

        w_self = F.normalize(model_a0.boundary.w_vocab_256[idx].float(), dim=-1)
        a0_self_cos = float((proj_a0 * w_self).sum(dim=-1).mean().item())
        a2_self_cos = float((proj_a2 * w_self).sum(dim=-1).mean().item())
        q_in_out_diff_a0 = float((model_a0.boundary.q_in_ortho - model_a0.boundary.q_out_ortho).norm().item())
        q_in_out_diff_a2 = float((model_a2.boundary.q_in_ortho - model_a2.boundary.q_out_ortho).norm().item())

    passed = (
        bipartite_frob_err < TRACE_TOL
        and sensory_cosine_fidelity > 0.45
        and a0_self_cos > 0.80
        and abs(a2_self_cos) < 0.15
        and q_in_out_diff_a0 == 0.0
        and q_in_out_diff_a2 > 10.0
    )
    return {
        "passed": bool(passed),
        "bipartite_frob_err": round(bipartite_frob_err, 8),
        "sensory_cosine_fidelity": round(sensory_cosine_fidelity, 6),
        "w71_a0_step0_self_copy_cosine": round(a0_self_cos, 6),
        "w71_a2_step0_self_copy_cosine": round(a2_self_cos, 6),
        "q_in_out_frobenius_diff_a0": round(q_in_out_diff_a0, 6),
        "q_in_out_frobenius_diff_a2": round(q_in_out_diff_a2, 6),
    }


def verify_wave71_model_step_and_quotient(device: torch.device) -> dict[str, Any]:
    """Verify forward/backward + 2-step `PureDensityQuotientMuon` invariants on `AnchoredBipartitePureDensityGPT` (`<= 8` args)."""
    spec = build_wave71_variant_spec(Wave71VariantMode.W71_A2_ANCHORED_BIPARTITE_DECOUPLED_4L, seed=1337)
    model = AnchoredBipartitePureDensityGPT(spec).to(device)
    model.train()
    trunk_opt, vocab_opt, scalar_opt = build_wave69_optimizers(model)
    total_params = sum(p.numel() for p in model.parameters())

    idx = torch.randint(0, 4096, (2, 16), device=device)
    targets = torch.randint(0, 4096, (2, 16), device=device)
    losses: list[float] = []
    for step in (1, 2):
        for opt in (trunk_opt, vocab_opt, scalar_opt):
            opt.zero_grad(set_to_none=True)
        loss, metrics = model(idx, targets=targets, return_metrics=True)
        loss.backward()
        lr_mult, mom_beta = compute_restored_riemannian_momentum_schedule(step, 100)
        trunk_opt.record_diagnostics = True
        trunk_opt.set_momentum(mom_beta)
        trunk_opt.set_lr_scale(lr_mult)
        trunk_opt.step()
        vocab_opt.set_momentum(mom_beta)
        vocab_opt.step_active(_collect_active_vocab_indices_w68(idx, targets), lr_scale=lr_mult)
        scalar_opt.set_momentum(min(0.90, mom_beta))
        scalar_opt.step_active(active_idx=None, lr_scale=lr_mult)
        model.restrict_scalars_()
        losses.append(float(loss.detach().item()))

    gd = trunk_opt.last_gauge_diagnostics
    su16_phase = float(gd.get("su16_step_phase_frac", 0.0))
    kraus_gauge = float(gd.get("kraus_step_gauge_frac", 0.0))
    passed = su16_phase < 1e-6 and kraus_gauge < 1e-5 and all(v > 0.0 for v in losses)
    return {
        "passed": bool(passed),
        "total_params": total_params,
        "step1_loss": round(losses[0], 4),
        "step2_loss": round(losses[1], 4),
        "su16_step_phase_frac": round(su16_phase, 8),
        "kraus_step_gauge_frac": round(kraus_gauge, 8),
        "analytical_beta": round(float(metrics["analytical_beta"]), 4),
    }


def main() -> None:
    """CLI entrypoint for Wave 71 sensory-anchored bipartite density verification (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Wave 71 Sensory-Anchored Bipartite Density Inspector")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json-out", type=str, default=".cache/wave71_anchored_bipartite_inspector.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    t12ab = verify_theorem12ab_collision_distinguishability_and_cptp(device)
    t12c = verify_theorem12c_bipartite_and_self_copy_elimination(device)
    model_chk = verify_wave71_model_step_and_quotient(device)

    report = {
        "plan_id": "EXPERIMENT_PLAN_091",
        "wave_id": "W71",
        "device": str(device),
        "overall_passed": bool(t12ab["passed"] and t12c["passed"] and model_chk["passed"]),
        "theorem12ab_collision_and_anchored_cptp": t12ab,
        "theorem12c_bipartite_and_self_copy_elimination": t12c,
        "wave71_model_step_and_quotient": model_chk,
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
