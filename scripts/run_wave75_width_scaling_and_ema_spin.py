"""CLI Runner for `Wave 75` (`EXPERIMENT_PLAN_095.md`):
1. Runs `verify_wave73_exact_mixed_state_equivalence()` across `d ∈ {64, 128, 192, 256}` in FP32.
2. Benchmarks throughput (`tok/s`), step latency (`ms`), and peak VRAM (`MiB`) across
   `d ∈ {64, 128, 192, 256}` (`1L` and `2L`, `W74 Coherent Ket-Avg` and `W75 Theorem 16 Exact Mixed-State`).
3. Runs FineWeb training across `d ∈ {64, 128, 192, 256}` with live `Hot Val CE`, `EMA Val CE`,
   and `Thermal Gap = Hot CE - EMA CE` tracking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qdlblocks.benchmarks.exact_mixed_state_baton_width_scaling_lab import (
    DEFAULT_SHADOW_EMA_DECAY,
    DEFAULT_WIDTH_CHOICES,
    Wave75WidthRunSpec,
    benchmark_width_scaling_throughput,
    run_wave75_width_training_with_ema,
    verify_wave73_exact_mixed_state_equivalence,
)
from qdlblocks.benchmarks.four_compartment_baton_lm_lab import REPO_ROOT


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments (`<= 8` args)."""
    parser = argparse.ArgumentParser(
        description="Run Wave 75 equivalence audit, width-scaling speed benchmark, and Hot/EMA FineWeb runs."
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=600,
        help="Training steps per width variant (default: 600 steps = 9.83M tokens at B=64, T=256).",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=150,
        help="Validation interval in steps for Hot & EMA CE evaluation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Sequence length (default: 256).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / ".cache" / "wave75_width_scaling_and_ema_summary.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute Wave 75 audit, throughput sweep, and FineWeb width-scaling training runs (`<= 8` args)."""
    args = parse_args()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)

    print("=== [1/3] Wave 73 vs Wave 74 vs Wave 75 (Theorem 16) Equivalence Audit ===", flush=True)
    equivalence_audits: dict[str, Any] = {}
    for d in DEFAULT_WIDTH_CHOICES:
        audit = verify_wave73_exact_mixed_state_equivalence(
            batch_size=2, seq_len=16, hilbert_dim=d, seed=1337 + d
        )
        equivalence_audits[f"d_{d}"] = audit
        print(
            f"[equiv d={d}] passed={audit['passed']} | "
            f"W75(Thm16) vs W73(Explicit d×d) deep_err={audit['w75_theorem16_vs_w73_explicit_deep_max_abs_err']:.3e}, "
            f"id_err={audit['w75_theorem16_vs_w73_explicit_id_max_abs_err']:.3e} | "
            f"W74(KetAvg) vs W73 diff={audit['w74_coherent_ket_avg_vs_w73_explicit_diff']:.4f}",
            flush=True,
        )

    print(
        "\n=== [2/3] Width-Scaling Throughput & Peak VRAM Benchmark (d = 64, 128, 192, 256) ===",
        flush=True,
    )
    throughput_table = benchmark_width_scaling_throughput(
        widths=DEFAULT_WIDTH_CHOICES,
        num_layers_choices=(1, 2),
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        warmup_steps=3,
        timed_steps=10,
    )

    print(
        "\n=== [3/3] FineWeb Width-Scaling Training with Live Hot & Shadow-EMA Val CE ===",
        flush=True,
    )
    training_specs = [
        Wave75WidthRunSpec(
            run_key="w75_d64_2l_ketavg_ema",
            hilbert_dim=64,
            num_layers=2,
            exact_mixed_state=False,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            total_steps=args.train_steps,
            eval_interval=args.eval_interval,
        ),
        Wave75WidthRunSpec(
            run_key="w75_d128_2l_ketavg_ema",
            hilbert_dim=128,
            num_layers=2,
            exact_mixed_state=False,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            total_steps=args.train_steps,
            eval_interval=args.eval_interval,
        ),
        Wave75WidthRunSpec(
            run_key="w75_d192_2l_ketavg_ema",
            hilbert_dim=192,
            num_layers=2,
            exact_mixed_state=False,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            total_steps=args.train_steps,
            eval_interval=args.eval_interval,
        ),
        Wave75WidthRunSpec(
            run_key="w75_d256_2l_ketavg_ema",
            hilbert_dim=256,
            num_layers=2,
            exact_mixed_state=False,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            total_steps=args.train_steps,
            eval_interval=args.eval_interval,
        ),
        Wave75WidthRunSpec(
            run_key="w75_d128_2l_exact_mixed_thm16_ema",
            hilbert_dim=128,
            num_layers=2,
            exact_mixed_state=True,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            total_steps=args.train_steps,
            eval_interval=args.eval_interval,
        ),
    ]

    training_runs: dict[str, Any] = {}
    for spec in training_specs:
        res = run_wave75_width_training_with_ema(
            spec,
            lr_peak=3.0e-3,
            lr_floor=3.0e-4,
            beta_ema=DEFAULT_SHADOW_EMA_DECAY,
        )
        training_runs[spec.run_key] = res
        summary = {
            "equivalence_audits": equivalence_audits,
            "throughput_table": throughput_table,
            "training_runs": training_runs,
        }
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n[wave75] Saved complete summary to {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
