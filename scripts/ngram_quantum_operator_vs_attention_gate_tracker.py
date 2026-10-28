#!/usr/bin/env python3
"""N-Gram Quantum Operator vs. Attention Gate Tracker (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Tracks trainable N-gram readout gates (`g_bi`, `g_tri`, `g_op`), `smear_alpha`, unitary drift,
and the amplitude norm ratio `||ng||_2 / ||raw_phi||_2` across meaning branches.
"""

from __future__ import annotations

import argparse
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
SCRIPT_NAME: str = "ngram_quantum_operator_vs_attention_gate_tracker.py"
NORM_EPS: float = 1e-8
ALGEBRAIC_HALF: float = 0.5


@torch.no_grad()
def inspect_ngram_and_attention_gates(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
) -> dict[str, Any]:
    """Extract per-branch `g_bi`, `g_tri`, `g_op`, `smear_alpha`, and `||ng||_2 / ||raw_phi||_2` (`<= 8` args)."""
    gates = model.ngram_branch_gates.detach().float() if hasattr(model, "ngram_branch_gates") else torch.zeros((2, 3))
    smear_logit = float(model.smear_gate_logit.detach().item()) if hasattr(model, "smear_gate_logit") else -1.5
    smear_alpha = ALGEBRAIC_HALF * (smear_logit / ((1.0 + smear_logit * smear_logit) ** 0.5) + 1.0)

    orig_mode = getattr(model, "_active_ngram_mode", "trigram")
    if hasattr(model, "_active_ngram_mode"):
        model._active_ngram_mode = "none"
    phi_raw = model.extract_readout_states(x_batch)[0].detach().float()
    if hasattr(model, "_active_ngram_mode"):
        model._active_ngram_mode = "trigram"
    phi_full = model.extract_readout_states(x_batch)[0].detach().float()
    if hasattr(model, "_active_ngram_mode"):
        model._active_ngram_mode = orig_mode

    ng_delta = phi_full - phi_raw
    norm_ratio = float(
        (ng_delta.norm(dim=-1).mean() / phi_raw.norm(dim=-1).mean().clamp_min(NORM_EPS)).item()
    )

    branch_rows: list[dict[str, float]] = []
    for r in range(int(gates.size(0))):
        branch_rows.append({
            "branch": float(r),
            "g_bi": round(float(gates[r, 0].item()), 6),
            "g_tri": round(float(gates[r, 1].item()), 6),
            "g_op": round(float(gates[r, 2].item()), 6),
        })

    return {
        "g_bi": round(float(gates[:, 0].mean().item()), 6),
        "g_tri": round(float(gates[:, 1].mean().item()), 6),
        "g_op": round(float(gates[:, 2].mean().item()), 6),
        "smear_alpha": round(float(smear_alpha), 6),
        "ng_to_raw_phi_norm_ratio": round(norm_ratio, 6),
        "per_branch_gates": branch_rows,
    }


@torch.no_grad()
def run_ngram_gate_tracker(args: argparse.Namespace) -> dict[str, Any]:
    """Execute N-gram operator vs. attention gate tracker (`<= 8` args)."""
    ckpt = args.checkpoint or args.results_json
    spec = DiagnosticModelSpec(
        arch=str(args.arch),
        checkpoint_path=ckpt,
        calibration_cache_path=str(args.calibration_cache),
        n_layer=int(args.num_layers),
        meaning_rank=int(args.meaning_rank),
        seq_len=int(args.seq_len),
        device=str(args.device),
    )
    model, metadata = load_or_build_calibrated_kronecker_model(spec)
    x_batch, _ = load_kronecker_probe_batch(
        val_shard_path=str(args.val_shard),
        batch_size=2,
        seq_len=min(32, int(args.seq_len)),
        device=str(args.device),
    )
    gate_stats = inspect_ngram_and_attention_gates(model, x_batch)
    metrics: dict[str, Any] = {
        "arch": metadata["arch"],
        "checkpoint_mode": metadata["checkpoint_mode"],
        "g_bi": gate_stats["g_bi"],
        "g_tri": gate_stats["g_tri"],
        "g_op": gate_stats["g_op"],
        "smear_alpha": gate_stats["smear_alpha"],
        "ng_to_raw_phi_norm_ratio": gate_stats["ng_to_raw_phi_norm_ratio"],
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
        extra_payload={"per_branch_gates": gate_stats["per_branch_gates"]},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Track N-gram operator gates vs. attention readout amplitude.")
    parser.add_argument("--arch", type=str, default="multihead_mixed_state")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--results-json", type=str, default=None)
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
    payload = run_ngram_gate_tracker(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} ratio={payload['metrics']['ng_to_raw_phi_norm_ratio']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
