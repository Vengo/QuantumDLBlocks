#!/usr/bin/env python3
r"""CLI Benchmark Runner for Variable-Rank $R$-Mode Uncollapsed Cholesky & Vacuum Subspace (`EXPERIMENT_PLAN_079.md`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qdlblocks.benchmarks.born_schur_density_lab import DEFAULT_FINEWEB_EVAL_INTERVAL, DEFAULT_FINEWEB_STEPS
from qdlblocks.benchmarks.variable_rank_cholesky_vacuum_lab import (
    DEFAULT_WINNING_CHOLESKY_RANK,
    run_variable_rank_cholesky_ablation_benchmark,
    run_variable_rank_cholesky_long_scaling_benchmark,
)
from qdlblocks.benchmarks.wavefunction_outer_density_lab import DEFAULT_LONG_FINEWEB_EVAL_INTERVAL, DEFAULT_LONG_FINEWEB_STEPS


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `variable_rank_cholesky_vacuum_bench.py` (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Run Variable-Rank R-Mode Cholesky & Vacuum Subspace Benchmarks (R0..R6)")
    parser.add_argument("--steps", type=int, default=DEFAULT_FINEWEB_STEPS, help="Training steps for Part 1 ablation")
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_FINEWEB_EVAL_INTERVAL, help="Eval interval for Part 1")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip Part 1 300-step R0..R5 ablation")
    parser.add_argument("--run-long-scaling", action="store_true", help="Run Part 2 1,200-step R6 scaling benchmark")
    parser.add_argument("--long-steps", type=int, default=DEFAULT_LONG_FINEWEB_STEPS, help="Steps for R6 long scaling run")
    parser.add_argument("--long-eval-interval", type=int, default=DEFAULT_LONG_FINEWEB_EVAL_INTERVAL, help="Eval interval for R6")
    parser.add_argument("--long-rank", type=int, default=DEFAULT_WINNING_CHOLESKY_RANK, help="Cholesky rank R for R6 scaling")
    parser.add_argument(
        "--json-out",
        type=str,
        default=str(REPO_ROOT / ".cache" / "variable_rank_cholesky_vacuum_results.json"),
        help="Path to write JSON benchmark report",
    )
    return parser.parse_args()


def main() -> int:
    """Execute Part 1 (`R0..R5`) and optional Part 2 (`R6`) variable-rank Cholesky benchmarks."""
    args = parse_args()
    report: dict[str, Any] = {}

    if not args.skip_ablation:
        print("=== Part 1: Variable-Rank R-Mode Uncollapsed Cholesky & Vacuum Subspace 300-Step Ablation (R0..R5) ===", flush=True)
        report["variable_rank_cholesky_ablation"] = run_variable_rank_cholesky_ablation_benchmark(
            steps=args.steps,
            eval_interval=args.eval_interval,
        )

    if args.run_long_scaling:
        print(
            f"\n=== Part 2: 1,200-Step Pure-Neural Unrestricted Rank R={args.long_rank} Vacuum Cholesky Scaling Run (R6) ===",
            flush=True,
        )
        report["variable_rank_cholesky_long_scaling"] = run_variable_rank_cholesky_long_scaling_benchmark(
            steps=args.long_steps,
            eval_interval=args.long_eval_interval,
            num_cholesky_modes=args.long_rank,
        )

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved complete Variable-Rank Vacuum Cholesky report to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
