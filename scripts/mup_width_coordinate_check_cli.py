#!/usr/bin/env python3
"""First-Class CLI for MuP Width Coordinate Checking (`Wave 77` / `EXPERIMENT_PLAN_097.md`).

Sweeps Hilbert widths `d in {64, 128, 256, 512}` (`128D..1024D` real) over `20` FineWeb training
steps comparing `naive_sp_adamw` (fixed-LR `AdamW`) against `mup_stiefel_muon` (`PolarExpress`
unit-spectral trunk updates + `1/D` untied `lm_head` `Adam` LR), printing formatted coordinate
tables and writing JSON diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qdlblocks.benchmarks.mup_width_coordinate_check import (
    DEFAULT_COORD_CHECK_BATCH_SIZE,
    DEFAULT_COORD_CHECK_NUM_LAYERS,
    DEFAULT_COORD_CHECK_SEQ_LEN,
    DEFAULT_COORD_CHECK_STEPS,
    MuPCoordinateCheckConfig,
    run_mup_width_coordinate_check,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments (`<= 8` args)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=str, default="64,128,256,512")
    parser.add_argument("--steps", type=int, default=DEFAULT_COORD_CHECK_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_COORD_CHECK_BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_COORD_CHECK_SEQ_LEN)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_COORD_CHECK_NUM_LAYERS)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / ".cache" / "wave77_mup_coordinate_check.json",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the MuP width coordinate check and write structured JSON report (`<= 8` args)."""
    args = parse_args()
    widths = tuple(int(x.strip()) for x in args.widths.split(",") if x.strip())
    device = torch.device(args.device)
    cfg = MuPCoordinateCheckConfig(
        widths=widths,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_layers=args.num_layers,
    )

    print("========================================================================================")
    print("WAVE 77 STEP 1: MUP WIDTH COORDINATE CHECK (NAIVE SP ADAMW vs. MUP STIEFEL-MUON)")
    print(f"Device: {device} | Complex Widths: {widths} | Steps: {cfg.steps} | Batch: {cfg.batch_size}x{cfg.seq_len}")
    print("========================================================================================")

    report = run_mup_width_coordinate_check(cfg, device=device)

    for mode_key, rows in report["records_by_mode"].items():
        sp = report["spread_summary"][mode_key]
        gate_str = "PASS" if sp["passes_mup_coordinate_gate"] else "FAIL (DRIFT)"
        print(
            f"\n[{mode_key}] LogitStd Max/Min={sp['raw_lm_logit_std_max_min_ratio']:.3f}x | "
            f"DeltaActRMS Max/Min={sp['delta_act_rms_max_min_ratio']:.3f}x | "
            f"ActRMS Max/Min={sp['layer_act_rms_max_min_ratio']:.3f}x -> [{gate_str}]"
        )
        for r in rows:
            print(
                f"  d={r['complex_dim']:3d} (D={r['real_dim']:4d}) | "
                f"head_lr={r['head_lr']:.5f} | ActRMS={r['stepN_mean_layer_rms']:.4f} | "
                f"DeltaActRMS={r['stepN_delta_act_rms_mean']:.4f} | "
                f"RawLogitStd={r['stepN_raw_lm_logit_std']:.4f} | "
                f"RelUpd={r['mean_rel_trunk_fro_update']:.4f} | "
                f"CE: {r['step0_val_ce']:.3f} -> {r['stepN_val_ce']:.3f} | "
                f"{r['tok_per_s']:8,.1f} tok/s"
            )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved coordinate check report to {args.json_out} ({report['wall_sec']:.2f}s)")


if __name__ == "__main__":
    main()
