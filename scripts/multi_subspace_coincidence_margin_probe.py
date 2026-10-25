#!/usr/bin/env python3
"""Diagnostic probe for Plan 046 (Wave 31, Lever 3): Product-of-Spheres Multi-Subspace Coincidence Born.

Audits across embedding widths d in {256, 384, 512, 768} and subspace counts M in {1, 2, 4, 8}:
1. Exact Product-of-Spheres unit L2 norm conservation (max | ||x||_2 - 1.0 | < 1e-6) and
   Parseval muP background variance invariance under beta_0(d) = 1.48 * sqrt(d).
2. Sharp detector participation ratio R_sharp retention under anisotropic branch skew
   (simulating high-gradient branches like readout_ngram and wav_256 growing in norm).
3. Positive runner-up CE tax reduction when impostor tokens align with 1 or 2 subspaces
   while the true target aligns across all M subspaces.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qdlblocks.layers.subspace_coincidence_born import (
    compute_multi_subspace_coincidence_observable,
    compute_subspace_cosines,
    subspace_l2_normalize,
)


def compute_analytical_parseval_beta(embed_dim: int, c_star: float = 1.48) -> float:
    """Return analytical Parseval muP inverse temperature beta_0(d) = c* sqrt(d)."""
    return float(c_star) * math.sqrt(float(embed_dim))


def compute_participation_ratio(matrix: torch.Tensor) -> float:
    """Compute effective rank (participation ratio) Tr(G)^2 / ||G||_F^2 of rows in matrix."""
    x = matrix.detach().to(torch.float32)
    gram = x.transpose(0, 1) @ x
    trace_val = torch.diagonal(gram).sum()
    frob_sq = (gram * gram).sum().clamp_min(1e-12)
    return float((trace_val * trace_val / frob_sq).item())


def build_anisotropic_raw_detectors(
    vocab_size: int,
    embed_dim: int,
    num_subspaces_skewed: int = 4,
    dominant_scale: float = 4.2,
    seed: int = 42,
    device: str = "cpu",
) -> torch.Tensor:
    """Simulate raw detector synthesis where one subspace branch grows in norm during training."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    raw = torch.randn((vocab_size, embed_dim), generator=gen, device=device, dtype=torch.float32)
    sub_dim = embed_dim // num_subspaces_skewed
    low_rank_u = torch.randn((vocab_size, 16), generator=gen, device=device, dtype=torch.float32)
    low_rank_v = torch.randn((16, sub_dim), generator=gen, device=device, dtype=torch.float32) / math.sqrt(16.0)
    raw[:, :sub_dim] = raw[:, :sub_dim] + dominant_scale * (low_rank_u @ low_rank_v)
    return raw


def evaluate_runnerup_impostor_ce_tax(
    m_sharp: torch.Tensor,
    num_subspaces: int,
    coincidence_gamma: float,
    subspace_var_penalty: float,
    num_queries: int = 64,
    num_impostors: int = 24,
    seed: int = 1046,
) -> dict[str, float]:
    """Measure positive runner-up CE tax when impostors match 1 of 4 subspaces vs target matching all."""
    vocab_size, embed_dim = m_sharp.shape
    device = m_sharp.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    targets = torch.randint(0, vocab_size, (num_queries,), generator=gen, device=device)
    target_vecs = m_sharp.index_select(0, targets).clone()

    noise = torch.randn((num_queries, embed_dim), generator=gen, device=device, dtype=torch.float32)
    raw_query = 0.68 * target_vecs + 0.32 * subspace_l2_normalize(noise, num_subspaces=max(num_subspaces, 4))
    psi_q = subspace_l2_normalize(raw_query, num_subspaces=num_subspaces)

    m_mod = m_sharp.clone()
    quarter_dim = embed_dim // 4
    for q_idx in range(min(num_queries, 16)):
        imp_indices = torch.randint(0, vocab_size, (num_impostors,), generator=gen, device=device)
        t_id = int(targets[q_idx].item())
        imp_indices = torch.where(imp_indices == t_id, (imp_indices + 1) % vocab_size, imp_indices)
        m_mod[imp_indices, :quarter_dim] = 2.8 * psi_q[q_idx : q_idx + 1, :quarter_dim]
    m_mod = subspace_l2_normalize(m_mod, num_subspaces=num_subspaces)

    cos_m = compute_subspace_cosines(psi_q, m_mod, num_subspaces=num_subspaces)
    obs_dict = compute_multi_subspace_coincidence_observable(
        cos_m,
        traceless_gamma=0.25,
        coincidence_gamma=coincidence_gamma if num_subspaces > 1 else 0.0,
        subspace_var_penalty=subspace_var_penalty if num_subspaces > 1 else 0.0,
    )
    obs = obs_dict["observable"]
    c_bar = obs_dict["c_bar"]
    beta0 = compute_analytical_parseval_beta(embed_dim, c_star=1.48)
    logits = beta0 * obs

    target_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    one_hot_target = torch.zeros_like(logits, dtype=torch.bool)
    one_hot_target.scatter_(1, targets.unsqueeze(1), True)

    pos_runnerup_mask = (c_bar > 0.0) & (~one_hot_target)
    shifted = logits - target_logits.unsqueeze(1)
    exp_shifted = torch.exp(shifted.clamp(max=40.0))

    z_pos_runnerup = (exp_shifted * pos_runnerup_mask.to(exp_shifted.dtype)).sum(dim=-1)
    z_all_other = (exp_shifted * (~one_hot_target).to(exp_shifted.dtype)).sum(dim=-1)

    pos_runnerup_tax = torch.log1p(z_pos_runnerup).mean().item()
    total_ce = torch.log1p(z_all_other).mean().item()
    neg_tail_tax = max(0.0, total_ce - pos_runnerup_tax)
    top1_acc = float((logits.argmax(dim=-1) == targets).to(torch.float32).mean().item())

    return {
        "total_ce_nats": float(total_ce),
        "positive_runnerup_ce_tax_nats": float(pos_runnerup_tax),
        "negative_tail_ce_tax_nats": float(neg_tail_tax),
        "top1_acc": top1_acc,
    }


def run_probe_grid(
    dims: list[int],
    subspaces: list[int],
    vocab_size: int = 50257,
    coincidence_gamma: float = 0.40,
    subspace_var_penalty: float = 0.20,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run the full (d, M) diagnostic grid and return structured metrics."""
    records: list[dict[str, Any]] = []
    by_dim_summary: dict[str, Any] = {}

    for d in dims:
        raw_det = build_anisotropic_raw_detectors(
            vocab_size=vocab_size,
            embed_dim=d,
            num_subspaces_skewed=4,
            dominant_scale=4.2,
            seed=42 + d,
            device=device,
        )
        dim_entries: dict[int, dict[str, Any]] = {}
        for m in subspaces:
            if d % m != 0:
                continue
            m_sharp = subspace_l2_normalize(raw_det, num_subspaces=m)
            row_norms = torch.linalg.vector_norm(m_sharp, dim=-1)
            max_unit_norm_err = float((row_norms - 1.0).abs().max().item())

            sub_view = m_sharp.view(vocab_size, m, d // m)
            sub_norms = torch.linalg.vector_norm(sub_view, dim=-1)
            max_subspace_norm_err = float((sub_norms - (1.0 / math.sqrt(float(m)))).abs().max().item())

            r_sharp = compute_participation_ratio(m_sharp[:8192])
            welch_rms_offdiag = 1.0 / math.sqrt(max(r_sharp, 1.0))

            gen = torch.Generator(device=device)
            gen.manual_seed(700 + d + m)
            rand_q = subspace_l2_normalize(
                torch.randn((32, d), generator=gen, device=device, dtype=torch.float32),
                num_subspaces=m,
            )
            beta0 = compute_analytical_parseval_beta(d, c_star=1.48)
            base_logits = beta0 * (rand_q @ m_sharp[:4096].transpose(0, 1))
            bg_logit_std = float(base_logits.std(dim=-1).mean().item())

            ce_metrics = evaluate_runnerup_impostor_ce_tax(
                m_sharp=m_sharp[:8192],
                num_subspaces=m,
                coincidence_gamma=coincidence_gamma,
                subspace_var_penalty=subspace_var_penalty,
                num_queries=64,
                num_impostors=24,
                seed=1046 + d,
            )

            entry = {
                "embed_dim": d,
                "num_subspaces": m,
                "subspace_dim": d // m,
                "beta0": float(beta0),
                "max_unit_norm_err": max_unit_norm_err,
                "max_subspace_norm_err": max_subspace_norm_err,
                "r_sharp_participation_ratio": r_sharp,
                "welch_rms_offdiag": welch_rms_offdiag,
                "background_logit_std": bg_logit_std,
                **ce_metrics,
            }
            records.append(entry)
            dim_entries[m] = entry

        if 1 in dim_entries and 4 in dim_entries:
            e1 = dim_entries[1]
            e4 = dim_entries[4]
            by_dim_summary[str(d)] = {
                "r_sharp_m1": e1["r_sharp_participation_ratio"],
                "r_sharp_m4": e4["r_sharp_participation_ratio"],
                "r_sharp_ratio_m4_over_m1": e4["r_sharp_participation_ratio"] / max(e1["r_sharp_participation_ratio"], 1e-6),
                "runnerup_tax_m1_nats": e1["positive_runnerup_ce_tax_nats"],
                "runnerup_tax_m4_nats": e4["positive_runnerup_ce_tax_nats"],
                "runnerup_tax_reduction_m4_vs_m1_nats": (
                    e1["positive_runnerup_ce_tax_nats"] - e4["positive_runnerup_ce_tax_nats"]
                ),
                "total_ce_m1_nats": e1["total_ce_nats"],
                "total_ce_m4_nats": e4["total_ce_nats"],
            }

    d512 = by_dim_summary.get("512", next(iter(by_dim_summary.values())))
    max_norm_err_all = max(r["max_unit_norm_err"] for r in records)
    h1_passed = (
        max_norm_err_all < 1e-5
        and d512["r_sharp_ratio_m4_over_m1"] >= 2.2
        and d512["r_sharp_m4"] >= 180.0
    )
    h2_passed = d512["runnerup_tax_reduction_m4_vs_m1_nats"] >= 0.12

    return {
        "plan_id": "EXPERIMENT_PLAN_046",
        "wave_id": "W31",
        "vocab_size": vocab_size,
        "coincidence_gamma": coincidence_gamma,
        "subspace_var_penalty": subspace_var_penalty,
        "records": records,
        "by_dim_summary": by_dim_summary,
        "H1_product_of_spheres_rank_preservation": {
            "passed": bool(h1_passed),
            "max_unit_norm_err": max_norm_err_all,
            "d512_r_sharp_m1": d512["r_sharp_m1"],
            "d512_r_sharp_m4": d512["r_sharp_m4"],
            "d512_r_sharp_ratio_m4_over_m1": d512["r_sharp_ratio_m4_over_m1"],
        },
        "H2_coincidence_runnerup_suppression": {
            "passed": bool(h2_passed),
            "d512_runnerup_tax_m1_nats": d512["runnerup_tax_m1_nats"],
            "d512_runnerup_tax_m4_nats": d512["runnerup_tax_m4_nats"],
            "d512_runnerup_tax_reduction_nats": d512["runnerup_tax_reduction_m4_vs_m1_nats"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Subspace Coincidence Margin & Participation Ratio Probe")
    parser.add_argument("--dims", type=str, default="256,384,512,768")
    parser.add_argument("--subspaces", type=str, default="1,2,4,8")
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--coincidence-gamma", type=float, default=0.40)
    parser.add_argument("--subspace-var-penalty", type=float, default=0.20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--json-out",
        type=str,
        default=str(REPO_ROOT / ".cache" / "multi_subspace_coincidence_margin_probe.json"),
    )
    args = parser.parse_args()

    dims = [int(x.strip()) for x in args.dims.split(",") if x.strip()]
    subspaces = [int(x.strip()) for x in args.subspaces.split(",") if x.strip()]
    report = run_probe_grid(
        dims=dims,
        subspaces=subspaces,
        vocab_size=args.vocab_size,
        coincidence_gamma=args.coincidence_gamma,
        subspace_var_penalty=args.subspace_var_penalty,
        device=args.device,
    )

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 98)
    print("Plan 046 (Wave 31): Product-of-Spheres Multi-Subspace Coincidence Born Margin Probe")
    print("=" * 98)
    print(f"{'d':>5} | {'M':>3} | {'d_s':>4} | {'R_sharp':>9} | {'Welch_RMS':>9} | {'RunnerUp_Tax':>12} | {'Total_CE':>9} | {'Top1':>6}")
    print("-" * 98)
    for rec in report["records"]:
        print(
            f"{rec['embed_dim']:5d} | {rec['num_subspaces']:3d} | {rec['subspace_dim']:4d} | "
            f"{rec['r_sharp_participation_ratio']:9.2f} | {rec['welch_rms_offdiag']:9.4f} | "
            f"{rec['positive_runnerup_ce_tax_nats']:12.4f} | {rec['total_ce_nats']:9.4f} | "
            f"{100.0 * rec['top1_acc']:5.1f}%"
        )
    print("-" * 98)
    print(
        f"H1 (Product-of-Spheres Rank Preservation): passed={report['H1_product_of_spheres_rank_preservation']['passed']} "
        f"(d=512 R_sharp M=1: {report['H1_product_of_spheres_rank_preservation']['d512_r_sharp_m1']:.2f} -> "
        f"M=4: {report['H1_product_of_spheres_rank_preservation']['d512_r_sharp_m4']:.2f}, "
        f"ratio={report['H1_product_of_spheres_rank_preservation']['d512_r_sharp_ratio_m4_over_m1']:.2f}x)"
    )
    print(
        f"H2 (Coincidence Runner-Up Tax Suppression): passed={report['H2_coincidence_runnerup_suppression']['passed']} "
        f"(d=512 Runner-Up Tax M=1: {report['H2_coincidence_runnerup_suppression']['d512_runnerup_tax_m1_nats']:.4f} -> "
        f"M=4: {report['H2_coincidence_runnerup_suppression']['d512_runnerup_tax_m4_nats']:.4f}, "
        f"reduction=-{report['H2_coincidence_runnerup_suppression']['d512_runnerup_tax_reduction_nats']:.4f} nats)"
    )
    print(f"Saved JSON report to {out_path}")


if __name__ == "__main__":
    main()
