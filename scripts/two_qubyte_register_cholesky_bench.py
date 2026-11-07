#!/usr/bin/env python3
"""Benchmark CLI for Two 8-Qubit Registers ("Qubytes"), Step-1 Multi-Mode Qubyte Projection, Step-3 Latent 512D Contraction, & Rank-R Schmidt Entanglement (`EXPERIMENT_PLAN_082.md` / `Wave 62`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import (
    DEFAULT_SHALLOW_BATCH_SIZE,
    DEFAULT_SHALLOW_SEQ_LEN,
    LongSeqFineWebStream,
)
from qdlblocks.benchmarks.two_qubyte_register_cholesky_lab import (
    run_two_qubyte_register_ablation_benchmark,
    run_two_qubyte_register_scaling_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Two 8-Qubit Registers & Step-1/Step-3 Cholesky Benchmark (Plan 082).")
    parser.add_argument("--mode", type=str, default="all", choices=["ablation", "scaling", "all"])
    parser.add_argument("--ablation-steps", type=int, default=300)
    parser.add_argument("--ablation-eval-interval", type=int, default=100)
    parser.add_argument("--scaling-steps", type=int, default=1200)
    parser.add_argument("--scaling-eval-interval", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_SHALLOW_BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SHALLOW_SEQ_LEN)
    parser.add_argument("--val-tokens", type=int, default=10_485_760)
    parser.add_argument("--json-out", type=str, default=".cache/two_qubyte_register_cholesky_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = LongSeqFineWebStream(
        device=device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        val_tokens=args.val_tokens,
    )

    payload: dict[str, Any] = {
        "plan_id": "EXPERIMENT_PLAN_082",
        "wave_id": "W62",
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "tokens_per_step": args.batch_size * args.seq_len,
        "val_tokens": args.val_tokens,
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode in ("ablation", "all"):
        payload["ablation_300_results"] = run_two_qubyte_register_ablation_benchmark(
            stream, steps=args.ablation_steps, eval_interval=args.ablation_eval_interval
        )
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.mode in ("scaling", "all"):
        payload["scaling_1200_results"] = run_two_qubyte_register_scaling_benchmark(
            stream, steps=args.scaling_steps, eval_interval=args.scaling_eval_interval
        )
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[two_qubyte_register_cholesky_bench] Saved results to {out_path}")


if __name__ == "__main__":
    main()
