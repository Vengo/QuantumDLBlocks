#!/usr/bin/env python3
"""Wave 72 (`EXPERIMENT_PLAN_092.md`) Benchmark Runner for `GroupedFourQubitKroneckerDensityGPT`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from qdlblocks.benchmarks.grouped_4qubit_kronecker_density_gpt_lab import (
    Wave72VariantMode,
    build_wave72_variant_spec,
    run_wave72_variant,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import LongSeqFineWebStream


def parse_variants(raw: str) -> list[Wave72VariantMode]:
    """Parse comma-separated `Wave72VariantMode` names (`<= 8` args)."""
    by_name = {m.value: m for m in Wave72VariantMode}
    modes: list[Wave72VariantMode] = []
    for item in raw.split(","):
        key = item.strip()
        if not key:
            continue
        if key not in by_name:
            raise ValueError(f"Unknown Wave 72 variant {key!r}. Choices: {list(by_name)}")
        modes.append(by_name[key])
    return modes


def main() -> None:
    """CLI entrypoint for Wave 72 `GroupedFourQubitKroneckerDensityGPT` experiments (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Wave 72 GroupedFourQubitKroneckerDensityGPT Benchmark Runner")
    parser.add_argument(
        "--variants",
        type=str,
        default="W72_Q2_GROUPED_4QUBIT_MULTIBRANCH_KRON_4L",
        help="Comma-separated Wave72VariantMode names.",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json-out", type=str, default=".cache/plan092_wave72_results.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    stream = LongSeqFineWebStream(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        val_tokens=65_536,
        device=device,
    )
    modes = parse_variants(args.variants)
    results: dict[str, Any] = {
        "plan_id": "EXPERIMENT_PLAN_092",
        "wave_id": "W72",
        "device": str(device),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "runs": {},
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        spec = build_wave72_variant_spec(mode, seed=args.seed)
        summary = run_wave72_variant(
            spec=spec,
            stream=stream,
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            eval_interval=args.eval_interval,
        )
        results["runs"][mode.value] = summary
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
