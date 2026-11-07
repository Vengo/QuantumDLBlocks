r"""Step 2 CLI Probe: Large-Batch Gradient Accumulation (`8,192 -> 65,536 -> 262,144` Tokens/Step) Duty Cycle, Geodesic Speed Balance, & Lookahead Step Alignment (`EXPERIMENT_PLAN_084.md` / `Wave 64`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from qdlblocks.benchmarks.kraus_mub_thermostat_8l_lab import run_grad_accum_duty_cycle_and_curvature_probe
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import LongSeqFineWebStream


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wave 64 Step 2: Large-Batch Gradient Accumulation Probe (8K -> 65K -> 262K tok/step)")
    parser.add_argument("--accum-steps", type=str, default="1,8,32", help="Comma-separated gradient accumulation factors")
    parser.add_argument("--steps-per-setting", type=int, default=15, help="Optimizer steps per accumulation setting")
    parser.add_argument("--batch-size", type=int, default=16, help="Micro-batch size B")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length T")
    parser.add_argument("--val-tokens", type=int, default=262_144, help="Validation token slice")
    parser.add_argument("--lr-scale", type=float, default=1.50, help="Geodesic step multiplier")
    parser.add_argument("--json-out", type=str, default=".cache/plan084_grad_accum_262k_probe.json", help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accum_tuple = tuple(int(x.strip()) for x in args.accum_steps.split(",") if x.strip())

    print(
        f"=== WAVE 64 STEP 2: LARGE-BATCH GRADIENT ACCUMULATION PROBE ===\n"
        f"Device: {device} | Micro-batch: B={args.batch_size}, T={args.seq_len} ({args.batch_size * args.seq_len} tok) | "
        f"AccumSteps: {accum_tuple} | OptSteps/Setting: {args.steps_per_setting}",
        flush=True,
    )
    t0 = time.perf_counter()
    stream = LongSeqFineWebStream(
        seq_len=args.seq_len, batch_size=args.batch_size, val_tokens=args.val_tokens, device=device
    )
    report = run_grad_accum_duty_cycle_and_curvature_probe(
        stream, accum_steps_list=accum_tuple, steps_per_setting=args.steps_per_setting, lr_scale=args.lr_scale
    )
    report["total_wall_sec"] = round(time.perf_counter() - t0, 2)

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[Saved Step-2 Gradient Accumulation Probe JSON -> {out_path} ({report['total_wall_sec']:.1f}s)]", flush=True)


if __name__ == "__main__":
    main()
