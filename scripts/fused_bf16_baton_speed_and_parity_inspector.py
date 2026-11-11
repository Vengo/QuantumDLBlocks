#!/usr/bin/env python3
"""Wave 74 (`EXPERIMENT_PLAN_094.md`): Fused `bfloat16` Baton Parity & Throughput Inspector.

Verifies Theorems 15a–15c numerically (`< 1e-5` max error vs. explicit `64 × 64` density matrix
evaluation) and benchmarks training throughput (`tok/s`) and peak VRAM (`MiB`) of `Wave 74`
Fused `bfloat16` (`1L` and `2L` 2-Hop Triangle) against `Wave 73` explicit FP32.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qdlblocks.benchmarks.fused_bf16_four_compartment_baton_lab import (
    REPO_ROOT,
    benchmark_wave73_vs_wave74_throughput,
    verify_theorem15_algebraic_identities,
)

DEFAULT_JSON_OUT: Path = REPO_ROOT / ".cache" / "wave74_fused_bf16_parity_and_speed.json"


def main() -> None:
    """Run the Theorem 15a–15c parity check and throughput benchmark (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Wave 74 Fused BF16 Baton Parity & Speed Inspector")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    args = parser.parse_args()

    parity = verify_theorem15_algebraic_identities()
    throughput = benchmark_wave73_vs_wave74_throughput(
        batch_size=args.batch_size, seq_len=args.seq_len, steps=args.steps
    )
    report = {
        "plan_id": "EXPERIMENT_PLAN_094",
        "wave_id": "W74",
        "overall_passed": bool(parity["passed"]),
        "theorem15_parity": parity,
        "throughput_benchmark": throughput,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
