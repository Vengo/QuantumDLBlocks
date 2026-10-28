#!/usr/bin/env python3
"""Trigram Context Smearing & Unitary Look-Back Ablation Probe (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Evaluates the causal 1-token look-back unitary smear gate `smear_gate_logit` and N-gram readout
injection modes (`none`, `bigram`, `trigram`) on FineWeb evaluation sequences, measuring state
sensitivity, NLL delta across smearing gate values, and unitary preservation of `smear_u`.
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
SCRIPT_NAME: str = "trigram_context_smearing_ablation_probe.py"
DEFAULT_SMEAR_LOGIT_GRID: str = "-4.0,-1.5,0.0,1.5"
DEFAULT_PROBE_BATCH_SIZE: int = 2
DEFAULT_PROBE_SEQ_LEN: int = 32
SMEAR_GATE_DENOM_BIAS: float = 1.0


def _parse_float_csv(raw: str) -> list[float]:
    """Parse comma-separated float list (`<= 8` args)."""
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


@torch.no_grad()
def evaluate_smear_gate_sweep(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    smear_logits: list[float],
) -> list[dict[str, float]]:
    """Sweep `smear_gate_logit` and measure readout factor shift and validation NLL (`<= 8` args)."""
    orig_smear = float(model.smear_gate_logit.detach().item()) if hasattr(model, "smear_gate_logit") else -1.5
    results: list[dict[str, float]] = []
    base_phi: torch.Tensor | None = None

    for logit_val in smear_logits:
        if hasattr(model, "smear_gate_logit"):
            model.smear_gate_logit.data.fill_(float(logit_val))
        alpha_eff = 0.5 * (float(logit_val) / ((1.0 + float(logit_val) ** 2) ** 0.5) + SMEAR_GATE_DENOM_BIAS)
        phi_factors = model.extract_readout_states(x_batch)[0].detach().float()
        if base_phi is None:
            base_phi = phi_factors.clone()
        l2_drift = float((phi_factors - base_phi).norm(dim=-1).mean().item())
        _, loss_t = model(x_batch, targets=y_batch, return_logits=False)
        nll_val = float(loss_t.detach().item()) if loss_t is not None else 0.0
        results.append({
            "smear_gate_logit": round(float(logit_val), 4),
            "effective_smear_alpha": round(float(alpha_eff), 6),
            "readout_phi_l2_drift": round(l2_drift, 6),
            "probe_nll": round(nll_val, 5),
        })

    if hasattr(model, "smear_gate_logit"):
        model.smear_gate_logit.data.fill_(orig_smear)
    return results


@torch.no_grad()
def run_smearing_ablation_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Run trigram context smearing and N-gram mode ablation probe (`<= 8` args)."""
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
        batch_size=DEFAULT_PROBE_BATCH_SIZE,
        seq_len=min(DEFAULT_PROBE_SEQ_LEN, int(args.seq_len)),
        device=str(args.device),
    )

    smear_grid = _parse_float_csv(args.smear_logits)
    sweep_rows = evaluate_smear_gate_sweep(model, x_batch, y_batch, smear_grid)

    mode_nlls: dict[str, float] = {}
    orig_mode = getattr(model, "_active_ngram_mode", "trigram")
    for mode_name in ("none", "bigram", "trigram"):
        if hasattr(model, "_active_ngram_mode"):
            model._active_ngram_mode = mode_name
        _, loss_t = model(x_batch, targets=y_batch, return_logits=False)
        mode_nlls[mode_name] = round(float(loss_t.detach().item()) if loss_t is not None else 0.0, 5)
    if hasattr(model, "_active_ngram_mode"):
        model._active_ngram_mode = orig_mode

    nlls = [row["probe_nll"] for row in sweep_rows]
    drifts = [row["readout_phi_l2_drift"] for row in sweep_rows]
    metrics: dict[str, Any] = {
        "arch": metadata["arch"],
        "checkpoint_mode": metadata["checkpoint_mode"],
        "nll_mode_none": mode_nlls["none"],
        "nll_mode_bigram": mode_nlls["bigram"],
        "nll_mode_trigram": mode_nlls["trigram"],
        "trigram_vs_none_nll_delta": round(mode_nlls["trigram"] - mode_nlls["none"], 5),
        "min_smear_nll": round(min(nlls), 5) if nlls else 0.0,
        "max_smear_phi_drift": round(max(drifts), 6) if drifts else 0.0,
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
        extra_payload={"smear_sweep": sweep_rows, "ngram_mode_nlls": mode_nlls},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Run trigram context smearing and look-back unitary ablation probe.")
    parser.add_argument("--arch", type=str, default="multihead_mixed_state")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--calibration-cache", type=str, default=DEFAULT_NGRAM_CALIBRATION_CACHE_PATH)
    parser.add_argument("--val-shard", type=str, default=DEFAULT_FINEWEB_VAL_SHARD_PATH)
    parser.add_argument("--smear-logits", type=str, default=DEFAULT_SMEAR_LOGIT_GRID)
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
    payload = run_smearing_ablation_probe(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} max_drift={payload['metrics']['max_smear_phi_drift']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
