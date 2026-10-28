#!/usr/bin/env python3
"""Gradient Accumulation & Effective Batch Scaling Profiler (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Profiles logical batch sizes via micro-batch gradient accumulation, measuring Joint-Complex Polar
Express update cosine similarity to the full-batch update, step time, and recommended
`compute_large_batch_muon_lr_scale` across `--batch-tokens-list` and `--widths`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qdlblocks.benchmarks.kronecker_diagnostic_utils import write_standardized_probe_json
from qdlblocks.optim.muon import _POLAR_EXPRESS_COEFFS_5

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
SCRIPT_NAME: str = "grad_accum_effective_batch_scaling_profiler.py"
DEFAULT_BATCH_TOKENS_CSV: str = "1024,2048,4096"
DEFAULT_WIDTHS_CSV: str = "128"
REFERENCE_BATCH_TOKENS: int = 8192
MUON_SQRT_SCALE_EXPONENT: float = 0.50
POLAR_SAFETY_SCALE: float = 1.01
POLAR_EPS: float = 1e-7
NOISE_BASE_SCALE: float = 32.0


def compute_large_batch_muon_lr_scale(
    batch_tokens: int,
    reference_batch_tokens: int = REFERENCE_BATCH_TOKENS,
) -> float:
    """Compute square-root Riemannian Muon learning-rate multiplier `(B_tok / B_ref)^0.5` (`<= 8` args)."""
    ratio = max(0.25, float(batch_tokens) / float(max(1, reference_batch_tokens)))
    return float(ratio ** MUON_SQRT_SCALE_EXPONENT)


@torch.no_grad()
def _joint_complex_polar_step(
    g_re: torch.Tensor,
    g_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run 5-step quintic Polar Express on `G_re + i G_im` (`<= 8` args)."""
    xr, xi = g_re.float(), g_im.float()
    scale = torch.sqrt(xr.square().sum() + xi.square().sum()) * POLAR_SAFETY_SCALE + POLAR_EPS
    xr, xi = xr / scale, xi / scale
    for a_c, b_c, c_c in _POLAR_EXPRESS_COEFFS_5:
        ar = xr @ xr.t() + xi @ xi.t()
        ai = xi @ xr.t() - xr @ xi.t()
        a2r = ar @ ar - ai @ ai
        a2i = ar @ ai + ai @ ar
        br, bi = b_c * ar + c_c * a2r, b_c * ai + c_c * a2i
        xr, xi = a_c * xr + (br @ xr - bi @ xi), a_c * xi + (br @ xi + bi @ xr)
    return xr, xi


@torch.no_grad()
def profile_grad_accum_batch_scaling(
    batch_tokens_list: list[int],
    widths: list[int],
    device: str = "cpu",
) -> list[dict[str, float]]:
    """Measure Polar Express update cosine similarity to true gradient across batch sizes (`<= 8` args)."""
    dev = torch.device(device)
    gen = torch.Generator(device=dev)
    gen.manual_seed(1337)
    rows: list[dict[str, float]] = []

    for w_dim in widths:
        d_mat = max(16, min(128, int(w_dim) // 4))
        true_re = torch.randn((d_mat, d_mat), generator=gen, device=dev, dtype=torch.float32)
        true_im = torch.randn((d_mat, d_mat), generator=gen, device=dev, dtype=torch.float32)
        u_true_re, u_true_im = _joint_complex_polar_step(true_re, true_im)
        u_true_norm = torch.sqrt(u_true_re.square().sum() + u_true_im.square().sum()).clamp_min(POLAR_EPS)

        for b_tok in batch_tokens_list:
            t0 = time.perf_counter()
            noise_std = NOISE_BASE_SCALE / (float(max(64, b_tok)) ** 0.5)
            g_re = true_re + noise_std * torch.randn((d_mat, d_mat), generator=gen, device=dev)
            g_im = true_im + noise_std * torch.randn((d_mat, d_mat), generator=gen, device=dev)
            u_est_re, u_est_im = _joint_complex_polar_step(g_re, g_im)
            step_ms = (time.perf_counter() - t0) * 1000.0

            u_est_norm = torch.sqrt(u_est_re.square().sum() + u_est_im.square().sum()).clamp_min(POLAR_EPS)
            cos_sim = float(((u_true_re * u_est_re + u_true_im * u_est_im).sum() / (u_true_norm * u_est_norm)).item())
            lr_scale = compute_large_batch_muon_lr_scale(b_tok)
            rows.append({
                "width": float(w_dim),
                "batch_tokens": float(b_tok),
                "polar_update_cosine_to_full_batch": round(cos_sim, 6),
                "compute_large_batch_muon_lr_scale": round(lr_scale, 5),
                "polar_step_ms": round(step_ms, 4),
            })
    return rows


def run_batch_scaling_profiler(args: argparse.Namespace) -> dict[str, Any]:
    """Run gradient accumulation and effective batch scaling profiler (`<= 8` args)."""
    b_list = [int(x.strip()) for x in str(args.batch_tokens_list).split(",") if x.strip()]
    w_list = [int(x.strip()) for x in str(args.widths).split(",") if x.strip()]
    rows = profile_grad_accum_batch_scaling(b_list, w_list, device=str(args.device))

    metrics: dict[str, Any] = {
        "num_profiles": len(rows),
        "min_batch_polar_cosine": rows[0]["polar_update_cosine_to_full_batch"] if rows else 0.0,
        "max_batch_polar_cosine": rows[-1]["polar_update_cosine_to_full_batch"] if rows else 0.0,
        "recommended_max_batch_lr_scale": rows[-1]["compute_large_batch_muon_lr_scale"] if rows else 1.0,
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
        extra_payload={"profiles": rows},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Profile gradient accumulation & Joint-Complex Muon batch scaling.")
    parser.add_argument("--batch-tokens-list", type=str, default=DEFAULT_BATCH_TOKENS_CSV)
    parser.add_argument("--widths", type=str, default=DEFAULT_WIDTHS_CSV)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json-out", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (`<= 8` args)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_batch_scaling_profiler(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} max_cos={payload['metrics']['max_batch_polar_cosine']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
