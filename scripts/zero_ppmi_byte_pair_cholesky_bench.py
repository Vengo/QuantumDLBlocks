#!/usr/bin/env python3
"""CLI Benchmark Runner for Strict Speedrun-Compliant Zero-PPMI Byte-Pair Cholesky GPT (`EXPERIMENT_PLAN_081.md` / `Wave 61`).

Runs:
1. 300-step (`2.46M` tokens) Zero-PPMI Byte-Pair Cholesky ablation suite (`Z0..Z4`).
2. Optional 1,200-step (`9.83M` tokens) Strict Speedrun-Compliant Zero-PPMI Byte-Pair + $R=12$ No-Click Vacuum Cholesky scaling run (`Z5`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import (
    DEFAULT_SHALLOW_BATCH_SIZE,
    DEFAULT_SHALLOW_SEQ_LEN,
    LongSeqFineWebStream,
)
from qdlblocks.benchmarks.zero_ppmi_byte_pair_cholesky_lab import (
    run_zero_ppmi_byte_pair_ablation_benchmark,
    run_zero_ppmi_byte_pair_scaling_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXPERIMENT_PLAN_081 Zero-PPMI Byte-Pair Cholesky benchmark.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_SHALLOW_BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SHALLOW_SEQ_LEN)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--run-long-scaling", action="store_true")
    parser.add_argument("--long-steps", type=int, default=1200)
    parser.add_argument("--long-eval-interval", type=int, default=150)
    parser.add_argument("--json-out", type=str, default=".cache/zero_ppmi_byte_pair_cholesky_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = LongSeqFineWebStream(device=device, batch_size=args.batch_size, seq_len=args.seq_len)

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ablation_results = run_zero_ppmi_byte_pair_ablation_benchmark(
        stream=stream, steps=args.steps, eval_interval=args.eval_interval
    )
    out_path.write_text(
        json.dumps(
            {
                "plan_id": "EXPERIMENT_PLAN_081",
                "wave_id": "W61",
                "init_tokens_scanned": 0,
                "ablation_300_step": ablation_results,
                "scaling_1200_step": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    scaling_results = []
    if args.run_long_scaling:
        scaling_results = run_zero_ppmi_byte_pair_scaling_benchmark(
            stream=stream, steps=args.long_steps, eval_interval=args.long_eval_interval
        )

    payload = {
        "plan_id": "EXPERIMENT_PLAN_081",
        "wave_id": "W61",
        "init_tokens_scanned": 0,
        "ablation_300_step": ablation_results,
        "scaling_1200_step": scaling_results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved Zero-PPMI Byte-Pair Cholesky results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
