#!/usr/bin/env python3
"""Wave 73 (`EXPERIMENT_PLAN_093.md`) FineWeb Shard-0 Benchmark CLI for `FourCompartmentBatonHighwayLM`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

from qdlblocks.benchmarks.four_compartment_baton_lm_lab import (
    ALL_WAVE73_VARIANTS,
    DEFAULT_TRAIN_SHARD,
    DEFAULT_VAL_SHARD,
    REPO_ROOT,
    Wave73TrainSpec,
    load_fineweb_tokens,
    run_wave73_variant,
)

DEFAULT_JSON_OUT: Path = REPO_ROOT / ".cache" / "wave73_four_compartment_bench_summary.json"


def parse_variant_list(raw: str) -> list[str]:
    """Parse comma-separated Wave 73 variant IDs (`<= 8` args)."""
    items = [x.strip() for x in raw.split(",") if x.strip()]
    for v in items:
        if v not in ALL_WAVE73_VARIANTS:
            raise ValueError(f"Unknown Wave 73 variant {v!r}. Valid: {ALL_WAVE73_VARIANTS}")
    return items


def main() -> int:
    """CLI entrypoint for `scripts/four_compartment_baton_bench.py` (`<= 8` args)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(ALL_WAVE73_VARIANTS),
        help="Comma-separated Wave 73 variant IDs.",
    )
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3.5e-3)
    parser.add_argument("--eval-batches", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_JSON_OUT,
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    train_tokens = load_fineweb_tokens(DEFAULT_TRAIN_SHARD, max_tokens=1_500_000)
    val_tokens = load_fineweb_tokens(DEFAULT_VAL_SHARD, max_tokens=250_000)
    variants = parse_variant_list(args.variants)

    summary: dict[str, Any] = {
        "plan_id": "EXPERIMENT_PLAN_093",
        "wave_id": "W73",
        "device": str(device),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "learning_rate": args.learning_rate if hasattr(args, "learning_rate") else args.lr,
        "runs": {},
    }

    out_path = args.json_out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for variant_id in variants:
        spec = Wave73TrainSpec(
            variant_id=variant_id,
            total_steps=args.steps,
            eval_interval=args.eval_interval,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            learning_rate=args.lr,
            eval_batches=args.eval_batches,
            seed=args.seed,
        )
        run_res = run_wave73_variant(spec, train_tokens, val_tokens, device)
        summary["runs"][variant_id] = run_res
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(
            f"[Wave 73] {variant_id}: step0_ce={run_res['step0_val_ce']:.4f} "
            f"-> final_val_ce={run_res['final_val_ce']:.4f} "
            f"(ppl={run_res['final_val_ppl']:.2f}, top1={run_res['final_top1_acc']:.4f}, "
            f"purity={run_res['final_out_purity']:.4f}, tok/s={run_res['tokens_per_sec']:.1f})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
