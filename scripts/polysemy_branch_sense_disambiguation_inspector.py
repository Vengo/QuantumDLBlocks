#!/usr/bin/env python3
"""Polysemy Branch Sense Disambiguation Inspector (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Inspects Kronecker models on paired polysemous contexts (e.g., `"river bank"` vs. `"central bank"`),
measuring contextual vs. static branch routing KL shift `mean_beta_kl_shift` and per-branch top-5
predicted token continuations.
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
    DEFAULT_NGRAM_CALIBRATION_CACHE_PATH,
    DiagnosticModelSpec,
    load_or_build_calibrated_kronecker_model,
    write_standardized_probe_json,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
SCRIPT_NAME: str = "polysemy_branch_sense_disambiguation_inspector.py"
DEFAULT_PROBE_WORDS_CSV: str = "bank,light,run,state,wave"
TOP_K_CONTINUATIONS: int = 5
CANDIDATE_VOCAB_SUBSET: int = 1024
KL_CLAMP_EPS: float = 1e-8


@torch.no_grad()
def evaluate_paired_polysemous_contexts(
    model: torch.nn.Module,
    probe_words: list[str],
    device: str = "cpu",
) -> dict[str, Any]:
    """Compare branch routing `beta_{t,r}` across paired contexts sharing the same final token (`<= 8` args)."""
    dev = torch.device(device)
    v_eff = int(model.cfg.valid_vocab_size)
    n_pairs = max(1, len(probe_words))

    ctx_a = torch.zeros((n_pairs, 8), device=dev, dtype=torch.int64)
    ctx_b = torch.zeros((n_pairs, 8), device=dev, dtype=torch.int64)
    for i, word in enumerate(probe_words):
        target_id = (abs(hash(word)) % (v_eff - 200)) + 100
        ctx_a[i] = torch.tensor([11, 27, 43, 59, 71, 89, 103, target_id], device=dev)
        ctx_b[i] = torch.tensor([211, 347, 509, 631, 757, 883, 997, target_id], device=dev)

    phi_a, psi_factors, beta_a, _ = model.extract_readout_states(ctx_a)
    phi_b, _, beta_b, _ = model.extract_readout_states(ctx_b)

    ba_last = beta_a.view(n_pairs, 8, -1)[:, -1, :].float().clamp_min(KL_CLAMP_EPS)
    bb_last = beta_b.view(n_pairs, 8, -1)[:, -1, :].float().clamp_min(KL_CLAMP_EPS)
    ba_last = ba_last / ba_last.sum(dim=-1, keepdim=True)
    bb_last = bb_last / bb_last.sum(dim=-1, keepdim=True)

    kl_shift = (ba_last * (torch.log(ba_last) - torch.log(bb_last))).sum(dim=-1).abs()
    mean_kl_shift = float(kl_shift.mean().item())

    phi_a_last = phi_a.view(n_pairs, 8, int(phi_a.size(1)), int(phi_a.size(2)), -1)[:, -1]
    phi_b_last = phi_b.view(n_pairs, 8, int(phi_b.size(1)), int(phi_b.size(2)), -1)[:, -1]
    state_l2_diff = float((phi_a_last - phi_b_last).norm(dim=-1).mean().item())

    n_cand = min(CANDIDATE_VOCAB_SUBSET, int(psi_factors.size(0)))
    top5_per_branch: dict[str, list[int]] = {}
    for r in range(int(phi_a.size(1))):
        scores = (psi_factors[:n_cand, r].float() * phi_a_last[0, r].float().unsqueeze(0)).sum(dim=(-1, -2))
        top5_per_branch[f"branch_{r}"] = [int(x) for x in torch.topk(scores, k=TOP_K_CONTINUATIONS).indices.tolist()]

    return {
        "mean_beta_kl_shift": round(mean_kl_shift, 6),
        "paired_context_state_l2_shift": round(state_l2_diff, 6),
        "per_branch_top5_continuations": top5_per_branch,
    }


@torch.no_grad()
def run_sense_disambiguation_inspector(args: argparse.Namespace) -> dict[str, Any]:
    """Execute polysemy branch sense disambiguation inspection (`<= 8` args)."""
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
    words = [w.strip() for w in str(args.probe_words).split(",") if w.strip()]
    disambig_stats = evaluate_paired_polysemous_contexts(model, words, device=str(args.device))

    metrics: dict[str, Any] = {
        "arch": metadata["arch"],
        "checkpoint_mode": metadata["checkpoint_mode"],
        "mean_beta_kl_shift": disambig_stats["mean_beta_kl_shift"],
        "paired_context_state_l2_shift": disambig_stats["paired_context_state_l2_shift"],
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
        extra_payload={"per_branch_top5_continuations": disambig_stats["per_branch_top5_continuations"], "probe_words": words},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Inspect polysemy branch sense disambiguation on paired contexts.")
    parser.add_argument("--arch", type=str, default="multihead_mixed_state")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--probe-words", type=str, default=DEFAULT_PROBE_WORDS_CSV)
    parser.add_argument("--calibration-cache", type=str, default=DEFAULT_NGRAM_CALIBRATION_CACHE_PATH)
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
    payload = run_sense_disambiguation_inspector(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} l2_shift={payload['metrics']['paired_context_state_l2_shift']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
