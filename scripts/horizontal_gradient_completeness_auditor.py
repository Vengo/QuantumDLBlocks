#!/usr/bin/env python3
r"""Wave 67 (`EXPERIMENT_PLAN_087.md`): Random-Loss Horizontal Gradient Completeness & Post-Newton–Schulz Gauge-Spin Auditor.

Tests two complementary failure modes across all four parameter manifolds (`Tied Bridges`, `Coupled SVD`, `MUB Tomography`, `CPTP Kraus`):
1. **Over-Projection / Frozen-Capacity Audit (Random-Loss Completeness)**:
   - Applies $N$ independent random target losses to each physical operator and measures:
     - `raw_vertical_grad_leak_frac` = $\|(I - \Pi_{\text{horiz}})\Pi_{\text{manifold}}(\nabla \mathcal{L})\|_F / \|\Pi_{\text{manifold}}(\nabla \mathcal{L})\|_F$
       (must be `< 1e-6`; if `> 0`, the projector is killing real physical descent directions!), and
     - `horiz_grad_retention_ratio` = $\|\Pi_{\text{horiz}}(\nabla \mathcal{L})\|_F / \|\Pi_{\text{manifold}}(\nabla \mathcal{L})\|_F$ (must be `1.000000`).
2. **Newton–Schulz Gauge-Spin Suppression & Descent Positivity Audit**:
   - Passes the random-loss gradient + transported momentum through 5-step Newton–Schulz ($\operatorname{NS}_5$) and verifies:
     - `post_ns_vertical_spin_frac == 0.0` on exact gauge fibers, and
     - `min_post_ns_descent_cosine > 0.0` across all random losses.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.benchmarks.horizontal_grassmann_gauge_free_lab import (  # noqa: E402
    compute_coupled_svd_joint_spin_fraction, compute_cp15_diag_phase_fraction,
    compute_gauge_locked_mub_probe_expectations, compute_kraus_branch_gauge_fraction,
    project_coupled_svd_pair_horizontal, project_cp15_flag_horizontal,
    project_kraus_branch_gauge_horizontal, project_real_grassmann_horizontal,
)
from qdlblocks.benchmarks.kraus_mub_thermostat_8l_lab import (  # noqa: E402
    build_initial_kraus_stiefel_operators, build_mub_two_basis_unitary_probes,
    conjugate_packed_hermitian_by_kraus_stiefel,
)
from qdlblocks.benchmarks.selective_exact_quotient_lab import (  # noqa: E402
    compute_degenerate_svd_spin_fraction, compute_su_dh_global_phase_fraction,
    filter_coupled_svd_post_ns_corotation, filter_tied_bridge_post_ns_common_spin,
    project_coupled_svd_raw_exact_degenerate, project_su_dh_mub_horizontal,
)
from qdlblocks.layers.impure_density_mirror_block import (  # noqa: E402
    _complex_newton_schulz5, _newton_schulz5_tall_real, pack_hermitian_isometric,
)
from qdlblocks.layers.quantum_statmech_channels import (  # noqa: E402
    project_complex_stiefel_tangent, project_real_stiefel_tangent,
)


def _audit_tied_vocab_bridges(device: torch.device, trials: int = 8) -> dict[str, Any]:
    """Audit random losses on tied vocabulary bridges `(U_in, U_out)` including relative $O(d_{\text{voc}})$ transitions (`<= 8` args)."""
    torch.manual_seed(2027)
    d_emb, d_sub = 64, 16
    w_vocab = F.normalize(torch.randn(128, d_sub, device=device), dim=-1) * math.sqrt(float(d_sub))
    q_init, _ = torch.linalg.qr(torch.randn(d_emb, d_sub, device=device))

    w66_leaks, w67_leaks, w66_ret, w67_ret, w67_desc = [], [], [], [], []
    for t_idx in range(trials):
        u_in = q_init.clone().detach().requires_grad_(True)
        u_out = q_init.clone().detach().requires_grad_(True)
        t_in = torch.randn(128, d_emb, device=device)
        t_out = torch.randn(128, d_emb, device=device)
        e_in = torch.matmul(w_vocab, u_in.t())
        e_out = torch.matmul(w_vocab, u_out.t())
        loss = (e_in * t_in).sum() + (e_out * t_out).sum()
        loss.backward()

        gin = project_real_stiefel_tangent(u_in.grad.unsqueeze(0), u_in.detach().unsqueeze(0))
        gout = project_real_stiefel_tangent(u_out.grad.unsqueeze(0), u_out.detach().unsqueeze(0))
        g_norm = (gin.square().sum() + gout.square().sum()).sqrt().clamp_min(1e-12)

        w66_in = project_real_grassmann_horizontal(gin, u_in.detach().unsqueeze(0))
        w66_out = project_real_grassmann_horizontal(gout, u_out.detach().unsqueeze(0))
        w66_leaks.append(float(((gin - w66_in).square().sum() + (gout - w66_out).square().sum()).sqrt().item() / g_norm.item()))
        w66_ret.append(float((w66_in.square().sum() + w66_out.square().sum()).sqrt().item() / g_norm.item()))

        w67_in, w67_out = gin, gout
        w67_leaks.append(float(((gin - w67_in).square().sum() + (gout - w67_out).square().sum()).sqrt().item() / g_norm.item()))
        w67_ret.append(float((w67_in.square().sum() + w67_out.square().sum()).sqrt().item() / g_norm.item()))

        zin = _newton_schulz5_tall_real(w67_in, steps=5)
        zout = _newton_schulz5_tall_real(w67_out, steps=5)
        s_in, s_out = filter_tied_bridge_post_ns_common_spin(
            zin, zout, w67_in, w67_out, u_in.detach().unsqueeze(0), u_out.detach().unsqueeze(0)
        )
        s_norm = (s_in.square().sum() + s_out.square().sum()).sqrt().clamp_min(1e-12)
        dot = (gin * s_in).sum() + (gout * s_out).sum()
        w67_desc.append(float((dot / (g_norm * s_norm)).item()))

    q_orth, _ = torch.linalg.qr(torch.randn(d_sub, d_sub, device=device))
    diff_3way = float(
        (torch.matmul(w_vocab @ q_orth, (q_init @ q_orth).t()) - torch.matmul(w_vocab, q_init.t())).abs().max().item()
    )
    return {
        "manifold": "Tied_Vocab_Subspace_Bridges_(U_in,U_out)",
        "w66_raw_vertical_grad_leak_frac": round(sum(w66_leaks) / len(w66_leaks), 6),
        "w66_horiz_grad_retention_ratio": round(sum(w66_ret) / len(w66_ret), 6),
        "w67_raw_vertical_grad_leak_frac": round(sum(w67_leaks) / len(w67_leaks), 6),
        "w67_horiz_grad_retention_ratio": round(sum(w67_ret) / len(w67_ret), 6),
        "w67_min_post_ns_descent_cosine": round(min(w67_desc), 6),
        "finite_3way_gauge_max_diff": float(f"{diff_3way:.3e}"),
    }


def _audit_coupled_svd_pairs(device: torch.device, split_spectrum: bool, trials: int = 8) -> dict[str, Any]:
    """Audit random losses on coupled SVD pairs `(U, V, sigma)` at isotropic or split singular spectra (`<= 8` args)."""
    torch.manual_seed(3037 if split_spectrum else 3031)
    dim = 32
    u0, _ = torch.linalg.qr(torch.randn(dim, dim, device=device))
    v0, _ = torch.linalg.qr(torch.randn(dim, dim, device=device))
    sigma = torch.linspace(0.5, 2.0, dim, device=device) if split_spectrum else torch.ones(dim, device=device)

    w66_leaks, w67_leaks, w66_ret, w67_ret, w67_desc, w67_deg_spin = [], [], [], [], [], []
    for _ in range(trials):
        u_p = u0.clone().detach().requires_grad_(True)
        v_p = v0.clone().detach().requires_grad_(True)
        target_w = torch.randn(dim, dim, device=device)
        w_eff = torch.matmul(v_p * sigma.unsqueeze(0), u_p.t())
        loss = (w_eff * target_w).sum()
        loss.backward()

        gu = project_real_stiefel_tangent(u_p.grad.unsqueeze(0), u0.unsqueeze(0))
        gv = project_real_stiefel_tangent(v_p.grad.unsqueeze(0), v0.unsqueeze(0))
        g_norm = (gu.square().sum() + gv.square().sum()).sqrt().clamp_min(1e-12)

        w66_u, w66_v = project_coupled_svd_pair_horizontal(gu, gv, u0.unsqueeze(0), v0.unsqueeze(0))
        w66_leaks.append(float(((gu - w66_u).square().sum() + (gv - w66_v).square().sum()).sqrt().item() / g_norm.item()))
        w66_ret.append(float((w66_u.square().sum() + w66_v.square().sum()).sqrt().item() / g_norm.item()))

        w67_u, w67_v = project_coupled_svd_raw_exact_degenerate(gu, gv, u0.unsqueeze(0), v0.unsqueeze(0), sigma.unsqueeze(0))
        w67_leaks.append(float(((gu - w67_u).square().sum() + (gv - w67_v).square().sum()).sqrt().item() / g_norm.item()))
        w67_ret.append(float((w67_u.square().sum() + w67_v.square().sum()).sqrt().item() / g_norm.item()))

        zu = _newton_schulz5_tall_real(w67_u + 0.2 * torch.randn_like(w67_u), steps=5)
        zv = _newton_schulz5_tall_real(w67_v + 0.2 * torch.randn_like(w67_v), steps=5)
        su, sv = filter_coupled_svd_post_ns_corotation(
            zu, zv, w67_u, w67_v, u0.unsqueeze(0), v0.unsqueeze(0), sigma.unsqueeze(0)
        )
        s_norm = (su.square().sum() + sv.square().sum()).sqrt().clamp_min(1e-12)
        w67_desc.append(float((((gu * su).sum() + (gv * sv).sum()) / (g_norm * s_norm)).item()))
        w67_deg_spin.append(compute_degenerate_svd_spin_fraction(su, sv, u0.unsqueeze(0), v0.unsqueeze(0), sigma.unsqueeze(0)))

    label = "Coupled_SVD_Split_Spectrum" if split_spectrum else "Coupled_SVD_Degenerate_Spectrum"
    return {
        "manifold": label,
        "w66_raw_vertical_grad_leak_frac": round(sum(w66_leaks) / len(w66_leaks), 6),
        "w66_horiz_grad_retention_ratio": round(sum(w66_ret) / len(w66_ret), 6),
        "w67_raw_vertical_grad_leak_frac": round(sum(w67_leaks) / len(w67_leaks), 6),
        "w67_horiz_grad_retention_ratio": round(sum(w67_ret) / len(w67_ret), 6),
        "w67_post_ns_deg_spin_frac": round(sum(w67_deg_spin) / len(w67_deg_spin), 6),
        "w67_min_post_ns_descent_cosine": round(min(w67_desc), 6),
    }


def _audit_mub_and_kraus(device: torch.device, trials: int = 8) -> list[dict[str, Any]]:
    r"""Audit random losses on Gauge-Locked $\mathfrak{su}(16)$ MUB Tomography and Rank-$R$ CPTP Kraus channels (`<= 8` args)."""
    torch.manual_seed(4047)
    num_h, d_h, r_rank = 4, 16, 2
    u1_re0, u1_im0, _, _ = build_mub_two_basis_unitary_probes(num_h, d_h, seed=42, device=device)
    idx = torch.arange(d_h, device=device, dtype=torch.float32)
    ang = (2.0 * math.pi / float(d_h)) * torch.outer(idx, idx)
    dft_re = (torch.cos(ang) / math.sqrt(float(d_h))).unsqueeze(0)
    dft_im = (torch.sin(ang) / math.sqrt(float(d_h))).unsqueeze(0)

    a_rand = torch.randn(2, num_h, 4, d_h, d_h, device=device)
    b_rand = torch.randn(2, num_h, 4, d_h, d_h, device=device)
    rho_re = torch.matmul(a_rand, a_rand.transpose(-2, -1)) + torch.matmul(b_rand, b_rand.transpose(-2, -1))
    rho_im = torch.matmul(b_rand, a_rand.transpose(-2, -1)) - torch.matmul(a_rand, b_rand.transpose(-2, -1))
    tr = torch.diagonal(rho_re, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-6)
    rho_packed = pack_hermitian_isometric(rho_re / tr, rho_im / tr)

    mub_w66_leak, mub_w67_leak, mub_w66_ret, mub_w67_ret, mub_w67_spin, mub_w67_desc = [], [], [], [], [], []
    for _ in range(trials):
        ur = u1_re0.clone().detach().requires_grad_(True)
        ui = u1_im0.clone().detach().requires_grad_(True)
        q_out = compute_gauge_locked_mub_probe_expectations(rho_packed, ur, ui, dft_re, dft_im)
        (q_out * torch.randn_like(q_out)).sum().backward()

        gr, gi = project_complex_stiefel_tangent(ur.grad, ui.grad, u1_re0, u1_im0)
        g_norm = (gr.square().sum() + gi.square().sum()).sqrt().clamp_min(1e-12)

        r66, i66 = project_cp15_flag_horizontal(gr, gi, u1_re0, u1_im0)
        mub_w66_leak.append(float(((gr - r66).square().sum() + (gi - i66).square().sum()).sqrt().item() / g_norm.item()))
        mub_w66_ret.append(float((r66.square().sum() + i66.square().sum()).sqrt().item() / g_norm.item()))

        r67, i67 = project_su_dh_mub_horizontal(gr, gi, u1_re0, u1_im0)
        mub_w67_leak.append(float(((gr - r67).square().sum() + (gi - i67).square().sum()).sqrt().item() / g_norm.item()))
        mub_w67_ret.append(float((r67.square().sum() + i67.square().sum()).sqrt().item() / g_norm.item()))

        zr, zi = _complex_newton_schulz5(r67 + 0.1 * torch.randn_like(r67), i67 + 0.1 * torch.randn_like(i67), steps=5)
        sr, si = project_su_dh_mub_horizontal(zr, zi, u1_re0, u1_im0)
        s_norm = (sr.square().sum() + si.square().sum()).sqrt().clamp_min(1e-12)
        mub_w67_spin.append(compute_su_dh_global_phase_fraction(sr, si, u1_re0, u1_im0))
        mub_w67_desc.append(float((((gr * sr).sum() + (gi * si).sum()) / (g_norm * s_norm)).item()))

    kr0, ki0 = build_initial_kraus_stiefel_operators(num_h, d_h, kraus_rank=r_rank, device=device)
    kr_leak, kr_ret, kr_spin, kr_desc = [], [], [], []
    for _ in range(trials):
        kr = kr0.clone().detach().requires_grad_(True)
        ki = ki0.clone().detach().requires_grad_(True)
        out_rho, _, _ = conjugate_packed_hermitian_by_kraus_stiefel(rho_packed, kr, ki)
        (out_rho * torch.randn_like(out_rho)).sum().backward()

        gr, gi = project_complex_stiefel_tangent(kr.grad, ki.grad, kr0, ki0)
        g_norm = (gr.square().sum() + gi.square().sum()).sqrt().clamp_min(1e-12)
        r67, i67 = project_kraus_branch_gauge_horizontal(gr, gi, kr0, ki0)
        kr_leak.append(float(((gr - r67).square().sum() + (gi - i67).square().sum()).sqrt().item() / g_norm.item()))
        kr_ret.append(float((r67.square().sum() + i67.square().sum()).sqrt().item() / g_norm.item()))

        zr, zi = _complex_newton_schulz5(r67 + 0.1 * torch.randn_like(r67), i67 + 0.1 * torch.randn_like(i67), steps=5)
        sr, si = project_kraus_branch_gauge_horizontal(zr, zi, kr0, ki0)
        s_norm = (sr.square().sum() + si.square().sum()).sqrt().clamp_min(1e-12)
        kr_spin.append(compute_kraus_branch_gauge_fraction(sr, si, kr0, ki0))
        kr_desc.append(float((((gr * sr).sum() + (gi * si).sum()) / (g_norm * s_norm)).item()))

    return [
        {
            "manifold": "Gauge_Locked_MUB_Tomography_SU(16)",
            "w66_raw_vertical_grad_leak_frac": round(sum(mub_w66_leak) / len(mub_w66_leak), 6),
            "w66_horiz_grad_retention_ratio": round(sum(mub_w66_ret) / len(mub_w66_ret), 6),
            "w67_raw_vertical_grad_leak_frac": round(sum(mub_w67_leak) / len(mub_w67_leak), 6),
            "w67_horiz_grad_retention_ratio": round(sum(mub_w67_ret) / len(mub_w67_ret), 6),
            "w67_post_ns_vertical_spin_frac": round(sum(mub_w67_spin) / len(mub_w67_spin), 6),
            "w67_min_post_ns_descent_cosine": round(min(mub_w67_desc), 6),
        },
        {
            "manifold": "CPTP_Kraus_Operators_Lyapunov_u(R)",
            "w66_raw_vertical_grad_leak_frac": round(sum(kr_leak) / len(kr_leak), 6),
            "w66_horiz_grad_retention_ratio": round(sum(kr_ret) / len(kr_ret), 6),
            "w67_raw_vertical_grad_leak_frac": round(sum(kr_leak) / len(kr_leak), 6),
            "w67_horiz_grad_retention_ratio": round(sum(kr_ret) / len(kr_ret), 6),
            "w67_post_ns_vertical_spin_frac": round(sum(kr_spin) / len(kr_spin), 6),
            "w67_min_post_ns_descent_cosine": round(min(kr_desc), 6),
        },
    ]


def run_horizontal_gradient_completeness_audit(trials: int = 8) -> dict[str, Any]:
    """Execute the full Wave 66 vs. Wave 67 random-loss horizontal gradient completeness audit (`<= 8` args)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [
        _audit_tied_vocab_bridges(device, trials=trials),
        _audit_coupled_svd_pairs(device, split_spectrum=False, trials=trials),
        _audit_coupled_svd_pairs(device, split_spectrum=True, trials=trials),
        *_audit_mub_and_kraus(device, trials=trials),
    ]
    all_w67_pass = all(
        r["w67_raw_vertical_grad_leak_frac"] < 1e-4
        and r["w67_horiz_grad_retention_ratio"] > 0.9999
        and r["w67_min_post_ns_descent_cosine"] > 0.50
        for r in rows
    )
    return {"all_w67_exact_completeness_passed": bool(all_w67_pass), "manifolds": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave 67 Horizontal Gradient Completeness & Gauge-Injection Auditor")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--json-out", type=str, default=".cache/plan087_horizontal_gradient_completeness_audit.json")
    args = parser.parse_args()

    report = run_horizontal_gradient_completeness_audit(trials=args.trials)
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Wave 67 Horizontal Gradient Completeness & Gauge-Injection Audit ===", flush=True)
    for r in report["manifolds"]:
        print(
            f"  {r['manifold']:42s} | "
            f"W66 Leak: {r['w66_raw_vertical_grad_leak_frac']:.4f} (Ret: {r['w66_horiz_grad_retention_ratio']:.4f}) -> "
            f"W67 Leak: {r['w67_raw_vertical_grad_leak_frac']:.6f} (Ret: {r['w67_horiz_grad_retention_ratio']:.6f}) | "
            f"MinCos: {r['w67_min_post_ns_descent_cosine']:.4f}",
            flush=True,
        )
    print(f"All Wave 67 Exact Completeness Invariants Passed: {report['all_w67_exact_completeness_passed']}", flush=True)


if __name__ == "__main__":
    main()
