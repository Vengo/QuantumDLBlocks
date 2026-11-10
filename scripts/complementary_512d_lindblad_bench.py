"""Wave 68 (`EXPERIMENT_PLAN_088.md`): Complementary `512D` ($256 \\oplus 256$), Lindblad Highway, & `8L` Depth Benchmark Runner.

Executes the 4-variant Wave 68 ablation (`W68_L0` .. `W68_L3`) or a single long-horizon run on official `FineWeb` shards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qdlblocks.benchmarks.complementary_512d_lindblad_8l_lab import (
    Wave68VariantMode,
    run_single_wave68_variant,
)

DEFAULT_BENCH_JSON_OUT: str = ".cache/complementary_512d_lindblad_bench.json"


def _parse_variant_list(variants_arg: str) -> list[Wave68VariantMode]:
    """Parse comma-separated variant list or `'all'` into `Wave68VariantMode` enums."""
    if variants_arg.strip().lower() == "all":
        return list(Wave68VariantMode)
    out: list[Wave68VariantMode] = []
    for tok in variants_arg.split(","):
        key = tok.strip().upper()
        if not key:
            continue
        matched = None
        for mode in Wave68VariantMode:
            if mode.value == key or mode.value.startswith(key):
                matched = mode
                break
        if matched is None:
            raise ValueError(f"Unknown Wave 68 variant: {tok!r}")
        out.append(matched)
    return out


def main() -> None:
    """CLI entrypoint for `complementary_512d_lindblad_bench.py`."""
    parser = argparse.ArgumentParser(description="Run Wave 68 Complementary 512D + Lindblad + 8L benchmarks.")
    parser.add_argument("--variants", type=str, default="all", help="Comma-separated variants or 'all'")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=None, help="Optional layer count override")
    parser.add_argument("--json-out", type=str, default=DEFAULT_BENCH_JSON_OUT)
    args = parser.parse_args()

    variants = _parse_variant_list(args.variants)
    results: list[dict[str, Any]] = []
    for mode in variants:
        summary = run_single_wave68_variant(
            variant=mode,
            steps=args.steps,
            eval_interval=args.eval_interval,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_layers_override=args.num_layers,
        )
        results.append(summary)

    payload = {
        "plan_id": "EXPERIMENT_PLAN_088",
        "wave_id": "W68",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "results": results,
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved Wave 68 benchmark summary to {out_path}")


if __name__ == "__main__":
    main()
