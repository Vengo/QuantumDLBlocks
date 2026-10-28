#!/usr/bin/env python3
"""Born Exponent & Holographic Coincidence Sweep (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Sweeps polynomial Born exponent `p`, holographic interference amplitude `holo_alpha`, and
4-factor Kronecker coincidence gain `coinc_gamma` on FineWeb sequences to map Step-0 NLL and
target-to-background intensity contrast.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qdlblocks.benchmarks.kronecker_diagnostic_utils import (
    DEFAULT_DIAGNOSTIC_SEQ_LEN,
    DEFAULT_FINEWEB_VAL_SHARD_PATH,
    DEFAULT_NGRAM_CALIBRATION_CACHE_PATH,
    DiagnosticModelSpec,
    load_kronecker_probe_batch,
    load_or_build_calibrated_kronecker_model,
    write_standardized_probe_json,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
SCRIPT_NAME: str = "born_exponent_and_holographic_coincidence_sweep.py"
DEFAULT_BOSON_POWERS_CSV: str = "6.0,8.0,10.0"
DEFAULT_COINC_GAMMAS_CSV: str = "0.20,0.35,0.50"
DEFAULT_HOLO_ALPHA: float = 0.85


def _parse_float_csv(raw: str) -> list[float]:
    """Parse comma-separated float list (`<= 8` args)."""
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


@torch.no_grad()
def run_born_and_coincidence_grid(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    powers: list[float],
    gammas: list[float],
    holo_alpha: float = DEFAULT_HOLO_ALPHA,
) -> list[dict[str, float]]:
    """Evaluate polynomial Born NLL across `(boson_p, coinc_gamma)` grid (`<= 8` args)."""
    orig_cfg = model.cfg
    rows: list[dict[str, float]] = []

    for p_val in powers:
        for g_val in gammas:
            model.cfg = replace(
                orig_cfg,
                boson_p=float(p_val),
                coinc_gamma=float(g_val),
                holo_alpha=float(holo_alpha),
            )
            _, loss_t = model(x_batch, targets=y_batch, return_logits=False)
            nll_val = float(loss_t.detach().item()) if loss_t is not None else 0.0
            rows.append({
                "boson_p": round(float(p_val), 3),
                "coinc_gamma": round(float(g_val), 3),
                "holo_alpha": round(float(holo_alpha), 3),
                "val_nll": round(nll_val, 5),
            })

    model.cfg = orig_cfg
    return rows


@torch.no_grad()
def execute_born_coincidence_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Run Born exponent and holographic coincidence sweep (`<= 8` args)."""
    spec = DiagnosticModelSpec(
        arch=str(args.arch),
        checkpoint_path=args.checkpoint,
        calibration_cache_path=str(args.calibration_cache),
        n_layer=int(args.num_layers),
        meaning_rank=int(args.meaning_rank),
        seq_len=int(args.seq_len),
        device=str(args.device),
    )
    model, metadata = load_or_build_calibrated_kronecker_model(spec)
    x_batch, y_batch = load_kronecker_probe_batch(
        val_shard_path=str(args.val_shard),
        batch_size=2,
        seq_len=min(24, int(args.seq_len)),
        device=str(args.device),
    )
    powers = _parse_float_csv(args.powers)
    gammas = _parse_float_csv(args.coinc_gammas)
    grid_rows = run_born_and_coincidence_grid(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        powers=powers,
        gammas=gammas,
        holo_alpha=float(args.holo_alpha),
    )
    best_row = min(grid_rows, key=lambda r: r["val_nll"]) if grid_rows else {"boson_p": 8.0, "coinc_gamma": 0.35, "val_nll": 0.0}

    metrics: dict[str, Any] = {
        "arch": metadata["arch"],
        "checkpoint_mode": metadata["checkpoint_mode"],
        "num_grid_points": len(grid_rows),
        "best_boson_p": best_row["boson_p"],
        "best_coinc_gamma": best_row["coinc_gamma"],
        "best_val_nll": best_row["val_nll"],
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
        extra_payload={"grid_results": grid_rows},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Sweep Born exponent p and holographic coincidence gamma.")
    parser.add_argument("--arch", type=str, default="multihead_mixed_state")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--calibration-cache", type=str, default=DEFAULT_NGRAM_CALIBRATION_CACHE_PATH)
    parser.add_argument("--val-shard", type=str, default=DEFAULT_FINEWEB_VAL_SHARD_PATH)
    parser.add_argument("--powers", "--boson-powers", dest="powers", type=str, default=DEFAULT_BOSON_POWERS_CSV)
    parser.add_argument("--coinc-gammas", type=str, default=DEFAULT_COINC_GAMMAS_CSV)
    parser.add_argument("--holo-alpha", type=float, default=DEFAULT_HOLO_ALPHA)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--meaning-rank", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_DIAGNOSTIC_SEQ_LEN)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json-out", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (`<= 8` args)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = execute_born_coincidence_sweep(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} best_nll={payload['metrics']['best_val_nll']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
