#!/usr/bin/env python3
"""Wave 67 (`EXPERIMENT_PLAN_087.md`): Selective Exact Quotienting, Read-Only Shadow EMA, & Wide `256D` Readout Benchmark CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.benchmarks.selective_exact_quotient_lab import (  # noqa: E402
    DEFAULT_TOMO_NUM_LAYERS, run_wave67_selective_quotient_comparison_suite,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import LongSeqFineWebStream  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave 67 Selective Exact Quotienting & Wide 256D Readout Benchmark")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--val-tokens", type=int, default=65_536)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_TOMO_NUM_LAYERS)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--only-variant", type=str, default="")
    parser.add_argument("--json-out", type=str, default=".cache/plan087_selective_exact_quotient_bench.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = LongSeqFineWebStream(
        seq_len=args.seq_len, batch_size=args.batch_size, val_tokens=args.val_tokens, device=device
    )
    suite = run_wave67_selective_quotient_comparison_suite(
        stream, steps=args.steps, eval_interval=args.eval_interval, wave_id="W67",
        num_layers=args.num_layers, grad_accum_steps=args.grad_accum_steps, only_variant=args.only_variant,
    )

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(suite, indent=2), encoding="utf-8")
    print(f"\nSaved Wave 67 benchmark results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
