#!/usr/bin/env python3
"""Wave 74 (`EXPERIMENT_PLAN_094.md`): Fused `bfloat16` 1L/2L 2-Hop Triangle `300M`-Token Runner.

Runs:
1. A fast head-to-head sprint comparing `1L` (`w74_f1_bf16_fused_2hop_1l_sprint`) vs. `2L`
   (`w74_f2_bf16_fused_2hop_2l_sprint`) 2-Hop Triangle 4-Compartment Baton in fused `bfloat16`, and
2. A full `300,000,000`-token (`300M` token) multi-shard FineWeb training run across Shards 1, 2,
   and 3 (`fineweb_train_000001..000003.bin`, validating on `fineweb_val_000000.bin`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qdlblocks.benchmarks.fused_bf16_four_compartment_baton_lab import (
    DEFAULT_TARGET_TOKENS_300M,
    FINEWEB_TRAIN_SHARDS_300M,
    FINEWEB_VAL_SHARD_PATH,
    REPO_ROOT,
    MultiShardFineWebStreamer,
    Wave74SpinRunSpec,
    run_wave74_spin_experiment,
)

DEFAULT_SUMMARY_JSON_OUT: Path = REPO_ROOT / ".cache" / "wave74_300m_spin_summary.json"


def main() -> None:
    """Execute the Wave 74 1L vs. 2L sprint and the 300M-token multi-shard spin (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Wave 74 300M-Token 4-Compartment Baton Spin")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("sprint_and_300m", "sprint_only", "300m_only"),
        default="sprint_and_300m",
    )
    parser.add_argument("--layers-300m", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--sprint-steps", type=int, default=400)
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS_300M)
    parser.add_argument("--eval-interval", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=4.0e-3)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_SUMMARY_JSON_OUT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    streamer = MultiShardFineWebStreamer(FINEWEB_TRAIN_SHARDS_300M)
    val_header = np.memmap(FINEWEB_VAL_SHARD_PATH, dtype=np.int32, mode="r", shape=(256,))
    val_n_tok = int(val_header[2])
    val_mm = np.memmap(
        FINEWEB_VAL_SHARD_PATH, dtype=np.uint16, mode="r", offset=256 * 4, shape=(val_n_tok,)
    )

    runs_out: dict[str, Any] = {}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    if args.json_out.exists():
        try:
            existing = json.loads(args.json_out.read_text(encoding="utf-8"))
            runs_out.update(existing.get("runs", {}))
        except Exception:
            pass

    if args.mode in ("sprint_and_300m", "sprint_only"):
        for num_l in (1, 2):
            run_key = f"w74_f{num_l}_bf16_fused_2hop_{num_l}l_sprint"
            spec = Wave74SpinRunSpec(
                run_key=run_key,
                num_layers=num_l,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                total_steps=args.sprint_steps,
                eval_interval=max(50, args.sprint_steps // 4),
                learning_rate=args.lr,
            )
            runs_out[run_key] = run_wave74_spin_experiment(
                spec=spec, streamer=streamer, val_mm=val_mm, device=device, eval_batches=16
            )
            args.json_out.write_text(
                json.dumps(
                    {"plan_id": "EXPERIMENT_PLAN_094", "wave_id": "W74", "runs": runs_out},
                    indent=2,
                ),
                encoding="utf-8",
            )

    if args.mode in ("sprint_and_300m", "300m_only"):
        tokens_per_step = args.batch_size * args.seq_len
        steps_300m = max(1, (args.target_tokens + tokens_per_step - 1) // tokens_per_step)
        num_l = args.layers_300m
        run_key = f"w74_bf16_fused_2hop_{num_l}l_300m_spin"
        spec_300m = Wave74SpinRunSpec(
            run_key=run_key,
            num_layers=num_l,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            total_steps=steps_300m,
            eval_interval=args.eval_interval,
            learning_rate=args.lr,
        )
        runs_out[run_key] = run_wave74_spin_experiment(
            spec=spec_300m, streamer=streamer, val_mm=val_mm, device=device, eval_batches=20
        )
        args.json_out.write_text(
            json.dumps(
                {"plan_id": "EXPERIMENT_PLAN_094", "wave_id": "W74", "runs": runs_out},
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
