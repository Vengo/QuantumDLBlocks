#!/usr/bin/env python3
"""Wave Preflight Hardware, Parameter Budget, & Throughput Gate (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Audits local GPU/CPU hardware availability, FineWeb shard & `900M`-token calibration cache presence,
parameter group partition (`StiefelMuon` complex pairs vs. `AdamW`), and 1-step forward+backward
throughput before launching a training wave.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
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
SCRIPT_NAME: str = "wave_preflight_hardware_and_throughput_gate.py"
DEFAULT_PREFLIGHT_SEQ_LEN: int = 32
DEFAULT_PREFLIGHT_BATCH_SIZE: int = 2
BYTES_PER_MIB: float = 1048576.0


def inspect_hardware_and_parameter_budget(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
) -> dict[str, Any]:
    """Count parameter groups and time a 1-step forward+backward pass (`<= 8` args)."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    complex_pair_params = sum(
        p.numel() for name, p in model.named_parameters() if name.endswith(("_re", "_im")) and p.ndim in (2, 3)
    )

    t0 = time.perf_counter()
    model.zero_grad(set_to_none=True)
    _, loss_t = model(x_batch, targets=y_batch, return_logits=False)
    loss_val = float(loss_t.detach().item()) if loss_t is not None else 0.0
    if loss_t is not None:
        loss_t.backward()
    step_sec = max(1e-6, time.perf_counter() - t0)
    tok_per_s = float(x_batch.numel()) / step_sec
    model.zero_grad(set_to_none=True)

    cuda_avail = bool(torch.cuda.is_available())
    dev_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    vram_total_mib = (
        float(torch.cuda.get_device_properties(0).total_memory) / BYTES_PER_MIB if cuda_avail else 0.0
    )

    return {
        "device_name": dev_name,
        "cuda_available": cuda_avail,
        "vram_total_mib": round(vram_total_mib, 1),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "complex_unitary_pair_params": int(complex_pair_params),
        "smoke_step_loss_nll": round(loss_val, 5),
        "smoke_step_tok_per_s": round(tok_per_s, 1),
        "preflight_pass": bool(math.isfinite(loss_val) and trainable_params > 0),
    }


import math


def run_preflight_gate(args: argparse.Namespace) -> dict[str, Any]:
    """Execute hardware, cache, and 1-step throughput preflight gate (`<= 8` args)."""
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
        batch_size=DEFAULT_PREFLIGHT_BATCH_SIZE,
        seq_len=min(DEFAULT_PREFLIGHT_SEQ_LEN, int(args.seq_len)),
        device=str(args.device),
    )
    hw_stats = inspect_hardware_and_parameter_budget(model, x_batch, y_batch)
    cal_exists = Path(args.calibration_cache).exists()
    val_exists = Path(args.val_shard).exists()

    metrics: dict[str, Any] = {
        "arch": metadata["arch"],
        "checkpoint_mode": metadata["checkpoint_mode"],
        "calibration_cache_present": bool(cal_exists),
        "validation_shard_present": bool(val_exists),
        **hw_stats,
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Run Wave 44 preflight hardware, cache, and throughput gate.")
    parser.add_argument("--arch", type=str, default="multihead_mixed_state")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--calibration-cache", type=str, default=DEFAULT_NGRAM_CALIBRATION_CACHE_PATH)
    parser.add_argument("--val-shard", type=str, default=DEFAULT_FINEWEB_VAL_SHARD_PATH)
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
    payload = run_preflight_gate(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} tok/s={payload['metrics']['smoke_step_tok_per_s']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
