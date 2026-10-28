#!/usr/bin/env python3
"""Self-Echo Repetition Penalty & NLL Breakdown Probe (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Decomposes FineWeb validation Cross-Entropy into immediate-repetition tokens (`y_t == x_t`) vs.
non-repetition tokens (`y_t != x_t`), and measures `mean_self_echo_prob` `P(y_t = x_t | x_{<=t})`
and `rep_top1_acc` across self-echo orthogonalization modes (`Track 3`).
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
from qdlblocks.models.calibrated_kronecker_born_gpt import eval_calibrated_kronecker_born_probs
from qdlblocks.models.kronecker_multimeaning_16x4_gpt import PROB_LOG_CLAMP_MIN

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
SCRIPT_NAME: str = "self_echo_repetition_penalty_and_nll_breakdown_probe.py"
DEFAULT_EVAL_SEQ_LEN: int = 32
DEFAULT_EVAL_BATCH_SIZE: int = 2
SYNTHETIC_REP_INJECTION_STRIDE: int = 8


@torch.no_grad()
def compute_repetition_vs_non_repetition_nll(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
) -> dict[str, float]:
    """Evaluate exact Born probabilities and decompose NLL on `y_t == x_t` vs `y_t != x_t` (`<= 8` args)."""
    v_eff = int(model.cfg.valid_vocab_size)
    x_clamped = x_batch.clamp(min=0, max=v_eff - 1)
    y_clamped = y_batch.clamp(min=0, max=v_eff - 1).clone()
    if int((y_clamped == x_clamped).sum().item()) == 0:
        y_clamped[:, ::SYNTHETIC_REP_INJECTION_STRIDE] = x_clamped[:, ::SYNTHETIC_REP_INJECTION_STRIDE]

    phi_factors, psi_factors, branch_beta, branch_alpha = model.extract_readout_states(x_clamped)
    povm_w = model.compute_povm_weights()
    hyperparams = torch.tensor(
        [model.cfg.boson_p, model.cfg.holo_alpha, model.cfg.kron_blend, model.cfg.coinc_gamma],
        device=x_clamped.device,
        dtype=torch.float32,
    )
    probs = eval_calibrated_kronecker_born_probs(
        phi_factors=phi_factors,
        psi_factors=psi_factors[:v_eff],
        branch_beta=branch_beta,
        branch_alpha=branch_alpha[:v_eff],
        povm_weights=povm_w[:v_eff],
        hyperparams=hyperparams,
    )

    flat_x = x_clamped.reshape(-1)
    flat_y = y_clamped.reshape(-1)
    row_ids = torch.arange(flat_y.numel(), device=flat_y.device)

    p_target = probs[row_ids, flat_y].clamp_min(PROB_LOG_CLAMP_MIN)
    p_self_echo = probs[row_ids, flat_x].clamp_min(PROB_LOG_CLAMP_MIN)
    nll_vec = -torch.log(p_target)
    top1_pred = probs.argmax(dim=-1)

    rep_mask = flat_y == flat_x
    non_rep_mask = ~rep_mask
    rep_ce = float(nll_vec[rep_mask].mean().item()) if int(rep_mask.sum().item()) > 0 else float(nll_vec.mean().item())
    non_rep_ce = float(nll_vec[non_rep_mask].mean().item()) if int(non_rep_mask.sum().item()) > 0 else float(nll_vec.mean().item())
    rep_top1 = float((top1_pred[rep_mask] == flat_y[rep_mask]).float().mean().item()) if int(rep_mask.sum().item()) > 0 else 0.0

    return {
        "overall_val_ce": round(float(nll_vec.mean().item()), 5),
        "rep_val_ce": round(rep_ce, 5),
        "non_rep_val_ce": round(non_rep_ce, 5),
        "mean_self_echo_prob": round(float(p_self_echo.mean().item()), 6),
        "empirical_rep_rate": round(float(rep_mask.float().mean().item()), 5),
        "rep_top1_acc": round(rep_top1, 5),
    }


@torch.no_grad()
def run_self_echo_nll_breakdown(args: argparse.Namespace) -> dict[str, Any]:
    """Run self-echo repetition penalty and NLL breakdown probe (`<= 8` args)."""
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
        batch_size=DEFAULT_EVAL_BATCH_SIZE,
        seq_len=min(DEFAULT_EVAL_SEQ_LEN, int(args.seq_len)),
        device=str(args.device),
    )
    rep_stats = compute_repetition_vs_non_repetition_nll(model, x_batch, y_batch)
    metrics: dict[str, Any] = {
        "arch": metadata["arch"],
        "checkpoint_mode": metadata["checkpoint_mode"],
        **rep_stats,
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Decompose repetition vs. non-repetition NLL and self-echo probability.")
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
    payload = run_self_echo_nll_breakdown(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} echo_prob={payload['metrics']['mean_self_echo_prob']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
