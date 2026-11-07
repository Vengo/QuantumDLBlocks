"""CLI Runner for Wavefunction -> Outer-Product Input Density (`W0..W3`) & Cone-Free Native Cholesky Readout (`D5, D6`).

Usage:
    /home/festivity/agy/QuantumDLBlocks/.venv/bin/python scripts/wavefunction_outer_density_bench.py \
        --json-out .cache/wavefunction_outer_density_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qdlblocks.benchmarks.wavefunction_outer_density_lab import (
    DEFAULT_FINEWEB_EVAL_INTERVAL,
    DEFAULT_FINEWEB_STEPS,
    DEFAULT_LONG_FINEWEB_EVAL_INTERVAL,
    DEFAULT_LONG_FINEWEB_STEPS,
    DEFAULT_REVERSAL_EVAL_INTERVAL,
    DEFAULT_REVERSAL_STEPS,
    run_native_cholesky_density_fineweb_benchmark,
    run_uncollapsed_cholesky_ablation_benchmark,
    run_uncollapsed_cholesky_long_scaling_benchmark,
    run_wavefunction_outer_input_reversal_benchmark,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Wavefunction Outer-Product Density laboratory."""
    parser = argparse.ArgumentParser(
        description="Wavefunction Outer-Product Input Density & Cone-Free Cholesky Readout Benchmark CLI"
    )
    parser.add_argument("--reversal-steps", type=int, default=DEFAULT_REVERSAL_STEPS)
    parser.add_argument("--reversal-eval-interval", type=int, default=DEFAULT_REVERSAL_EVAL_INTERVAL)
    parser.add_argument("--fineweb-steps", type=int, default=DEFAULT_FINEWEB_STEPS)
    parser.add_argument("--fineweb-eval-interval", type=int, default=DEFAULT_FINEWEB_EVAL_INTERVAL)
    parser.add_argument("--long-scaling-steps", type=int, default=DEFAULT_LONG_FINEWEB_STEPS)
    parser.add_argument("--long-scaling-eval-interval", type=int, default=DEFAULT_LONG_FINEWEB_EVAL_INTERVAL)
    parser.add_argument("--skip-reversal", action="store_true")
    parser.add_argument("--skip-fineweb", action="store_true")
    parser.add_argument("--run-uncollapsed", action="store_true")
    parser.add_argument("--run-long-scaling", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(".cache/wavefunction_outer_density_results.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Execute Wavefunction Outer-Product Input and Cone-Free Native Cholesky benchmarks."""
    args = parse_args()
    report: dict[str, Any] = {}

    if not args.skip_reversal:
        print("=== Part 1: 50,304-Vocab Reversal Input Outer-Product Density Benchmark (W0..W3) ===", flush=True)
        report["input_reversal_benchmark"] = run_wavefunction_outer_input_reversal_benchmark(
            steps=args.reversal_steps,
            eval_interval=args.reversal_eval_interval,
            device=args.device,
        )

    if not args.skip_fineweb:
        print("\n=== Part 2: FineWeb L=4, d=512, T=512 Cone-Free Native Cholesky Density Benchmark (D5, D6) ===", flush=True)
        report["cone_free_fineweb_benchmark"] = run_native_cholesky_density_fineweb_benchmark(
            steps=args.fineweb_steps,
            eval_interval=args.fineweb_eval_interval,
            device=args.device,
        )

    if args.run_uncollapsed:
        print("\n=== Part 3: Uncollapsed 4-Mode Cholesky Inter-Layer Highway Benchmark (D7, D8) ===", flush=True)
        report["uncollapsed_cholesky_ablation"] = run_uncollapsed_cholesky_ablation_benchmark(
            steps=args.fineweb_steps,
            eval_interval=args.fineweb_eval_interval,
            device=args.device,
        )

    if args.run_long_scaling:
        print("\n=== Part 4: Longer FineWeb Scaling Run (D9_UNCOLLAPSED_CHOLESKY_LONG_SCALING_1200) ===", flush=True)
        report["uncollapsed_cholesky_long_scaling"] = run_uncollapsed_cholesky_long_scaling_benchmark(
            steps=args.long_scaling_steps,
            eval_interval=args.long_scaling_eval_interval,
            device=args.device,
        )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved complete Wavefunction Outer-Product Density report to {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
