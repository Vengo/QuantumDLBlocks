"""CLI Runner for Born-Schur Density-Matrix Attention & Density-Matrix Highway Benchmarks.

Usage:
    /home/festivity/agy/QuantumDLBlocks/.venv/bin/python scripts/born_schur_density_bench.py \
        --json-out .cache/born_schur_density_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qdlblocks.benchmarks.born_schur_density_lab import (
    DEFAULT_FINEWEB_EVAL_INTERVAL,
    DEFAULT_FINEWEB_STEPS,
    DEFAULT_REVERSAL_EVAL_INTERVAL,
    DEFAULT_REVERSAL_STEPS,
    run_born_schur_fineweb_benchmark,
    run_born_schur_reversal_benchmark,
    run_schur_thermodynamic_cooling_probe,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Born-Schur Density-Matrix laboratory."""
    parser = argparse.ArgumentParser(description="Born-Schur Density-Matrix Attention & Highway Benchmark CLI")
    parser.add_argument("--reversal-steps", type=int, default=DEFAULT_REVERSAL_STEPS)
    parser.add_argument("--reversal-eval-interval", type=int, default=DEFAULT_REVERSAL_EVAL_INTERVAL)
    parser.add_argument("--fineweb-steps", type=int, default=DEFAULT_FINEWEB_STEPS)
    parser.add_argument("--fineweb-eval-interval", type=int, default=DEFAULT_FINEWEB_EVAL_INTERVAL)
    parser.add_argument("--reversal-variants", type=str, default="", help="Optional comma-separated filter of reversal variant IDs")
    parser.add_argument("--skip-cooling", action="store_true")
    parser.add_argument("--skip-reversal", action="store_true")
    parser.add_argument("--skip-fineweb", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(".cache/born_schur_density_results.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Execute selected Born-Schur Density-Matrix probes and persist JSON report."""
    args = parse_args()
    report: dict[str, Any] = {}

    if not args.skip_cooling:
        print("=== Part 1: Schur Thermodynamic Cooling & Hermitian Packing Invariants ===", flush=True)
        cooling = run_schur_thermodynamic_cooling_probe(device=args.device)
        report["cooling_probe"] = cooling
        print(
            f"  Pure Schur Identity Err: {cooling['pure_schur_identity_err']:.2e} | "
            f"Packed GEMM Equivalence Err: {cooling['packed_gemm_equivalence_err']:.2e}",
            flush=True,
        )
        for mode_name, layers in cooling["cooling_vs_thermal_death"].items():
            p_start = layers[0]["mean_purity_tr_rho2"]
            p_end = layers[-1]["mean_purity_tr_rho2"]
            d_end = layers[-1]["diag_coincidence_d_tr_schur"]
            r_end = layers[-1]["effective_rank"]
            print(
                f"  [{mode_name:26s}] L1->L{len(layers)} SpectralPurity Tr(rho^2): {p_start:.4f} -> {p_end:.4f} | "
                f"DiagCoinc D*Tr(rho*rho): {d_end:.4f} | EffRank: {r_end:.2f}",
                flush=True,
            )

    if not args.skip_reversal:
        print("\n=== Part 2: 50,304-Vocab Sequence Reversal Benchmark (D0..D3) ===", flush=True)
        v_filter = {v.strip() for v in args.reversal_variants.split(",") if v.strip()} if args.reversal_variants else None
        report["reversal_benchmark"] = run_born_schur_reversal_benchmark(
            steps=args.reversal_steps,
            eval_interval=args.reversal_eval_interval,
            device=args.device,
            variant_filter=v_filter,
        )

    if not args.skip_fineweb:
        print("\n=== Part 3: FineWeb L=4, d=512, T=512 Born-Schur + Geodesic-Cone Benchmark ===", flush=True)
        report["fineweb_benchmark"] = run_born_schur_fineweb_benchmark(
            steps=args.fineweb_steps,
            eval_interval=args.fineweb_eval_interval,
            device=args.device,
        )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved complete Born-Schur Density report to {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
