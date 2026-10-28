#!/usr/bin/env python3
"""Rare-Token Gradient Starvation Tracker (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Runs `N` forward+backward steps and outputs a 10-decile breakdown of `embed_grad_rms`,
`povm_scale_grad_rms`, and `active_token_fraction` across Zipf token-frequency deciles.
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
    DEFAULT_NUM_FREQUENCY_DECILES,
    DiagnosticModelSpec,
    compute_fineweb_10_frequency_deciles,
    load_kronecker_probe_batch,
    load_or_build_calibrated_kronecker_model,
    write_standardized_probe_json,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
SCRIPT_NAME: str = "rare_token_gradient_starvation_tracker.py"
DEFAULT_TRACKER_STEPS: int = 1
DEFAULT_TRACKER_SEQ_LEN: int = 16
GRAD_ACTIVE_THRESHOLD: float = 1e-10


def profile_per_decile_gradients(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    steps: int = DEFAULT_TRACKER_STEPS,
    num_deciles: int = DEFAULT_NUM_FREQUENCY_DECILES,
) -> dict[str, Any]:
    """Run `steps` backward passes and compute 10-decile `embed_grad_rms`, `povm_scale_grad_rms`, `active_token_fraction` (`<= 8` args)."""
    v_eff = int(model.cfg.valid_vocab_size)
    dev = model.embed_re.device
    model.zero_grad(set_to_none=True)

    for s_idx in range(max(1, int(steps))):
        x_step = (x_batch + s_idx) % v_eff
        y_step = (y_batch + s_idx) % v_eff
        _, loss_t = model(x_step, targets=y_step, return_logits=False)
        if loss_t is not None:
            loss_t.backward()

    uni_p = model.unigram_probs[:v_eff].detach().float() if hasattr(model, "unigram_probs") else torch.ones((v_eff,), device=dev) / float(v_eff)
    deciles = compute_fineweb_10_frequency_deciles(uni_p, active_vocab_size=v_eff, num_deciles=num_deciles)

    if model.embed_re.grad is not None and model.embed_im.grad is not None:
        g_emb = (model.embed_re.grad[:v_eff].detach().float().square() + model.embed_im.grad[:v_eff].detach().float().square()).mean(dim=(1, 2, 3))
    else:
        g_emb = torch.zeros((v_eff,), device=dev)

    if hasattr(model, "povm_scale_logits") and model.povm_scale_logits.grad is not None:
        g_povm = model.povm_scale_logits.grad[:v_eff].detach().float().square()
    else:
        g_povm = torch.zeros((v_eff,), device=dev)

    decile_rows: list[dict[str, float]] = []
    for d_idx in range(int(num_deciles)):
        mask = deciles == d_idx
        if int(mask.sum().item()) == 0:
            continue
        emb_rms = float(torch.sqrt(g_emb[mask].mean()).item())
        povm_rms = float(torch.sqrt(g_povm[mask].mean()).item())
        active_frac = float(((g_emb[mask] + g_povm[mask]) > GRAD_ACTIVE_THRESHOLD).float().mean().item())
        decile_rows.append({
            "decile": float(d_idx),
            "embed_grad_rms": round(emb_rms, 8),
            "povm_scale_grad_rms": round(povm_rms, 8),
            "active_token_fraction": round(active_frac, 6),
        })

    model.zero_grad(set_to_none=True)
    return {
        "embed_grad_rms": round(float(torch.sqrt(g_emb.mean()).item()), 8),
        "povm_scale_grad_rms": round(float(torch.sqrt(g_povm.mean()).item()), 8),
        "active_token_fraction": round(float(((g_emb + g_povm) > GRAD_ACTIVE_THRESHOLD).float().mean().item()), 6),
        "deciles": decile_rows,
    }


def run_gradient_starvation_tracker(args: argparse.Namespace) -> dict[str, Any]:
    """Execute rare-token gradient starvation tracker (`<= 8` args)."""
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
        seq_len=min(DEFAULT_TRACKER_SEQ_LEN, int(args.seq_len)),
        device=str(args.device),
    )
    grad_stats = profile_per_decile_gradients(model, x_batch, y_batch, steps=int(args.steps))
    metrics: dict[str, Any] = {
        "arch": metadata["arch"],
        "checkpoint_mode": metadata["checkpoint_mode"],
        "embed_grad_rms": grad_stats["embed_grad_rms"],
        "povm_scale_grad_rms": grad_stats["povm_scale_grad_rms"],
        "active_token_fraction": grad_stats["active_token_fraction"],
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
        extra_payload={"deciles": grad_stats["deciles"]},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Track per-decile rare-token gradient starvation.")
    parser.add_argument("--arch", type=str, default="multihead_mixed_state")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--steps", type=int, default=DEFAULT_TRACKER_STEPS)
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
    payload = run_gradient_starvation_tracker(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} active={payload['metrics']['active_token_fraction']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
