"""CLI Runner for Orthogonal Vacuum Subspace (`H_vac`), Layer-0 Cholesky Initialization, and 4-Mode 512D Readout (`V0..V4`).

Usage:
    /home/festivity/agy/QuantumDLBlocks/.venv/bin/python scripts/vacuum_subspace_cholesky_bench.py \
        --json-out .cache/vacuum_subspace_cholesky_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qdlblocks.benchmarks.vacuum_subspace_cholesky_lab import (
    DEFAULT_FINEWEB_EVAL_INTERVAL,
    DEFAULT_FINEWEB_STEPS,
    DEFAULT_LONG_FINEWEB_EVAL_INTERVAL,
    DEFAULT_LONG_FINEWEB_STEPS,
    run_vacuum_subspace_cholesky_ablation_benchmark,
    run_vacuum_subspace_cholesky_long_scaling_benchmark,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Vacuum Subspace Cholesky laboratory (`<= 8` args)."""
    parser = argparse.ArgumentParser(
        description="Orthogonal Vacuum Subspace (H_vac) & 4-Mode 512D Cholesky Benchmark CLI"
    )
    parser.add_argument("--fineweb-steps", type=int, default=DEFAULT_FINEWEB_STEPS)
    parser.add_argument("--fineweb-eval-interval", type=int, default=DEFAULT_FINEWEB_EVAL_INTERVAL)
    parser.add_argument("--long-scaling-steps", type=int, default=DEFAULT_LONG_FINEWEB_STEPS)
    parser.add_argument("--long-scaling-eval-interval", type=int, default=DEFAULT_LONG_FINEWEB_EVAL_INTERVAL)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-v0-baseline", action="store_true")
    parser.add_argument("--run-long-scaling", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(".cache/vacuum_subspace_cholesky_results.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Execute Vacuum Subspace Cholesky ablation (`V0..V3`) and optional 1,200-step scaling (`V4`)."""
    args = parse_args()
    report: dict[str, Any] = {}
    if args.json_out.exists():
        try:
            report = json.loads(args.json_out.read_text(encoding="utf-8"))
        except Exception:
            report = {}

    if not args.skip_ablation:
        print("=== Part 1: Orthogonal Vacuum Subspace & 4-Mode 512D Cholesky 300-Step Ablation (V0..V3) ===", flush=True)
        report["vacuum_cholesky_ablation"] = run_vacuum_subspace_cholesky_ablation_benchmark(
            steps=args.fineweb_steps,
            eval_interval=args.fineweb_eval_interval,
            device=args.device,
            include_v0_baseline=not args.skip_v0_baseline,
        )
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.run_long_scaling:
        print("\n=== Part 2: 1,200-Step Pure-Neural Vacuum Cholesky Scaling Run (V4) ===", flush=True)
        report["vacuum_cholesky_long_scaling"] = run_vacuum_subspace_cholesky_long_scaling_benchmark(
            steps=args.long_scaling_steps,
            eval_interval=args.long_scaling_eval_interval,
            device=args.device,
        )
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nSaved complete Vacuum Subspace Cholesky report to {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
