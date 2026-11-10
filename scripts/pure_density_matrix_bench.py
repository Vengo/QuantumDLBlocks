#!/usr/bin/env python3
r"""Wave 69 (`EXPERIMENT_PLAN_089.md`): Benchmark Runner for `PureDensityMatrixGPT` (`W69_P0`..`W69_P3`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from qdlblocks.benchmarks.pure_density_matrix_gpt_lab import (
    Wave69VariantMode, build_wave69_variant_spec, run_wave69_variant,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import LongSeqFineWebStream

VARIANT_ALIASES: dict[str, Wave69VariantMode] = {
    "W69_P0": Wave69VariantMode.W69_P0_PURE_DENSITY_HS_ATTN_4L,
    "W69_P1": Wave69VariantMode.W69_P1_PURE_DENSITY_UNCONSTRAINED_BORN_4L,
    "W69_P2": Wave69VariantMode.W69_P2_PURE_DENSITY_EMA_NGRAM_HISTORY_4L,
    "W69_P3": Wave69VariantMode.W69_P3_PURE_DENSITY_DEEP_8L_CONSOLIDATED,
    Wave69VariantMode.W69_P0_PURE_DENSITY_HS_ATTN_4L.value: Wave69VariantMode.W69_P0_PURE_DENSITY_HS_ATTN_4L,
    Wave69VariantMode.W69_P1_PURE_DENSITY_UNCONSTRAINED_BORN_4L.value: Wave69VariantMode.W69_P1_PURE_DENSITY_UNCONSTRAINED_BORN_4L,
    Wave69VariantMode.W69_P2_PURE_DENSITY_EMA_NGRAM_HISTORY_4L.value: Wave69VariantMode.W69_P2_PURE_DENSITY_EMA_NGRAM_HISTORY_4L,
    Wave69VariantMode.W69_P3_PURE_DENSITY_DEEP_8L_CONSOLIDATED.value: Wave69VariantMode.W69_P3_PURE_DENSITY_DEEP_8L_CONSOLIDATED,
}


def resolve_wave69_variants(raw_csv: str) -> list[Wave69VariantMode]:
    """Parse comma-separated variant list into `Wave69VariantMode` enums (`<= 8` args)."""
    modes: list[Wave69VariantMode] = []
    for token in raw_csv.split(","):
        key = token.strip()
        if not key:
            continue
        if key not in VARIANT_ALIASES:
            raise ValueError(f"Unknown Wave 69 variant '{key}'. Valid keys: {sorted(VARIANT_ALIASES.keys())}")
        modes.append(VARIANT_ALIASES[key])
    return modes


def main() -> None:
    """CLI entrypoint for `scripts/pure_density_matrix_bench.py`."""
    parser = argparse.ArgumentParser(description="Wave 69 Pure Density-Matrix GPT Benchmark Runner.")
    parser.add_argument(
        "--variants",
        type=str,
        default="W69_P0,W69_P1,W69_P2,W69_P3",
        help="Comma-separated Wave 69 variants (`W69_P0,W69_P1,W69_P2,W69_P3`).",
    )
    parser.add_argument("--steps", type=int, default=200, help="Number of training steps per variant.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per step.")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length per step.")
    parser.add_argument("--eval-interval", type=int, default=25, help="Evaluation interval in steps.")
    parser.add_argument("--eval-batches", type=int, default=8, help="Number of validation batches per eval.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument(
        "--json-out",
        type=str,
        default="/home/festivity/agy/QuantumDLBlocks/.cache/plan089_pure_density_bench.json",
        help="Path to write JSON benchmark summary.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = LongSeqFineWebStream(seq_len=args.seq_len, batch_size=args.batch_size, val_tokens=65_536, device=device)
    variants = resolve_wave69_variants(args.variants)

    results: list[dict[str, Any]] = []
    for mode in variants:
        spec = build_wave69_variant_spec(mode, seed=args.seed)
        res = run_wave69_variant(
            spec=spec,
            stream=stream,
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            eval_interval=args.eval_interval,
            eval_batches=args.eval_batches,
        )
        results.append(res)

    summary = {
        "plan_id": "EXPERIMENT_PLAN_089",
        "wave_id": "W69",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "results": results,
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved Wave 69 benchmark summary to {out_path}", flush=True)


if __name__ == "__main__":
    main()
