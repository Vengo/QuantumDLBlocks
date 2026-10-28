#!/usr/bin/env python3
"""Modded-NanoGPT 10M Validation Per-Decile CE Breakdown (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Decomposes FineWeb Shard-0 (`fineweb_val_000000.bin`) validation Cross-Entropy and Top-1 accuracy
across 10 token-frequency deciles (`Decile 0`..`Decile 9`) and repetition vs. non-repetition tokens
(`rep_val_ce`, `non_rep_val_ce`, `mean_self_echo_prob`).
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
    load_fineweb_eval_slice,
    load_or_build_calibrated_kronecker_model,
    write_standardized_probe_json,
)
from qdlblocks.models.calibrated_kronecker_born_gpt import eval_calibrated_kronecker_born_probs
from qdlblocks.models.kronecker_multimeaning_16x4_gpt import PROB_LOG_CLAMP_MIN

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
SCRIPT_NAME: str = "modded_nanogpt_10m_per_decile_ce_breakdown.py"
DEFAULT_SMOKE_EVAL_TOKENS: int = 256
MAX_SINGLE_PASS_TOKENS: int = 512


@torch.no_grad()
def evaluate_per_decile_validation_ce(
    model: torch.nn.Module,
    val_tokens_1d: torch.Tensor,
    seq_len: int = DEFAULT_DIAGNOSTIC_SEQ_LEN,
    num_deciles: int = DEFAULT_NUM_FREQUENCY_DECILES,
) -> dict[str, Any]:
    """Compute 10-decile `val_ce`, `top1_acc`, `rep_val_ce`, `non_rep_val_ce`, `mean_self_echo_prob` (`<= 8` args)."""
    v_eff = int(model.cfg.valid_vocab_size)
    dev = model.embed_re.device
    n_usable = min(int(val_tokens_1d.numel()) - 1, MAX_SINGLE_PASS_TOKENS)
    t_eff = max(4, min(int(seq_len), n_usable))
    b_eff = max(1, n_usable // t_eff)
    n_tot = b_eff * t_eff

    x_seq = val_tokens_1d[:n_tot].reshape(b_eff, t_eff).to(device=dev).clamp(0, v_eff - 1)
    y_seq = val_tokens_1d[1 : n_tot + 1].reshape(b_eff, t_eff).to(device=dev).clamp(0, v_eff - 1)

    phi_factors, psi_factors, branch_beta, branch_alpha = model.extract_readout_states(x_seq)
    povm_w = model.compute_povm_weights()
    hyperparams = torch.tensor(
        [model.cfg.boson_p, model.cfg.holo_alpha, model.cfg.kron_blend, model.cfg.coinc_gamma],
        device=dev,
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

    flat_x, flat_y = x_seq.reshape(-1), y_seq.reshape(-1)
    rows = torch.arange(n_tot, device=dev)
    p_target = probs[rows, flat_y].clamp_min(PROB_LOG_CLAMP_MIN)
    p_echo = probs[rows, flat_x].clamp_min(PROB_LOG_CLAMP_MIN)
    nll_vec = -torch.log(p_target)
    correct_vec = (probs.argmax(dim=-1) == flat_y).float()

    uni_p = model.unigram_probs[:v_eff].detach().float() if hasattr(model, "unigram_probs") else torch.ones((v_eff,), device=dev) / float(v_eff)
    deciles = compute_fineweb_10_frequency_deciles(uni_p, active_vocab_size=v_eff, num_deciles=num_deciles)
    target_deciles = deciles[flat_y]

    decile_rows: list[dict[str, float]] = []
    for d_idx in range(int(num_deciles)):
        mask = target_deciles == d_idx
        cnt = int(mask.sum().item())
        ce_d = float(nll_vec[mask].mean().item()) if cnt > 0 else float(nll_vec.mean().item())
        acc_d = float(correct_vec[mask].mean().item()) if cnt > 0 else 0.0
        decile_rows.append({
            "decile": float(d_idx),
            "token_count": float(cnt),
            "val_ce": round(ce_d, 5),
            "top1_acc": round(acc_d, 5),
        })

    rep_mask = flat_y == flat_x
    non_rep_mask = ~rep_mask
    rep_ce = float(nll_vec[rep_mask].mean().item()) if int(rep_mask.sum().item()) > 0 else float(nll_vec.mean().item())
    non_rep_ce = float(nll_vec[non_rep_mask].mean().item()) if int(non_rep_mask.sum().item()) > 0 else float(nll_vec.mean().item())

    return {
        "val_ce": round(float(nll_vec.mean().item()), 5),
        "top1_acc": round(float(correct_vec.mean().item()), 5),
        "rep_val_ce": round(rep_ce, 5),
        "non_rep_val_ce": round(non_rep_ce, 5),
        "mean_self_echo_prob": round(float(p_echo.mean().item()), 6),
        "deciles": decile_rows,
    }


@torch.no_grad()
def run_per_decile_ce_breakdown(args: argparse.Namespace) -> dict[str, Any]:
    """Run FineWeb validation per-decile CE and repetition breakdown (`<= 8` args)."""
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
    val_slice = load_fineweb_eval_slice(
        val_shard_path=str(args.val_shard),
        max_tokens=max(32, int(args.max_tokens)) + 1,
    )
    breakdown = evaluate_per_decile_validation_ce(model, val_slice, seq_len=min(32, int(args.seq_len)))

    metrics: dict[str, Any] = {
        "arch": metadata["arch"],
        "checkpoint_mode": metadata["checkpoint_mode"],
        "val_ce": breakdown["val_ce"],
        "top1_acc": breakdown["top1_acc"],
        "rep_val_ce": breakdown["rep_val_ce"],
        "non_rep_val_ce": breakdown["non_rep_val_ce"],
        "mean_self_echo_prob": breakdown["mean_self_echo_prob"],
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
        extra_payload={"deciles": breakdown["deciles"]},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Decompose FineWeb Shard-0 validation CE across 10 rarity deciles.")
    parser.add_argument("--arch", type=str, default="multihead_mixed_state")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-tokens", "--max-eval-tokens", dest="max_tokens", type=int, default=DEFAULT_SMOKE_EVAL_TOKENS)
    parser.add_argument("--calibration-cache", type=str, default=DEFAULT_NGRAM_CALIBRATION_CACHE_PATH)
    parser.add_argument("--val-shard", type=str, default=DEFAULT_FINEWEB_VAL_SHARD_PATH)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--meaning-rank", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_DIAGNOSTIC_SEQ_LEN)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json-out", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (`<= 8` args)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_per_decile_ce_breakdown(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} val_ce={payload['metrics']['val_ce']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
