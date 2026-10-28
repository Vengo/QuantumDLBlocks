#!/usr/bin/env python3
"""Multi-Branch Polysemy Entropy & Subspace Angle Probe (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Measures polysemy meaning-branch routing entropy `H(beta_t)`, temporal routing standard deviation
`std_t(beta_{t,0})`, inter-branch subspace angles, and 4-factor Kronecker cross-branch overlap
across vocabulary wavefunctions and context states.
"""

from __future__ import annotations

import argparse
import math
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
    compute_pairwise_complex_branch_overlap,
    load_kronecker_probe_batch,
    load_or_build_calibrated_kronecker_model,
    write_standardized_probe_json,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
SCRIPT_NAME: str = "multibranch_polysemy_entropy_and_subspace_angle_probe.py"
DEFAULT_MAX_VOCAB_SUBSET: int = 2048
ENTROPY_CLAMP_EPS: float = 1e-8
NATS_TO_BITS: float = 1.4426950408889634


@torch.no_grad()
def compute_routing_beta_statistics(
    branch_beta: torch.Tensor,
    branch_alpha: torch.Tensor,
) -> dict[str, Any]:
    """Compute context routing entropy, temporal std, and vocabulary branch alpha entropy (`<= 8` args)."""
    beta_f = branch_beta.detach().float().clamp_min(ENTROPY_CLAMP_EPS)
    beta_norm = beta_f / beta_f.sum(dim=-1, keepdim=True).clamp_min(ENTROPY_CLAMP_EPS)
    ctx_ent_nats = -(beta_norm * torch.log(beta_norm)).sum(dim=-1)
    ctx_ent_bits = ctx_ent_nats * NATS_TO_BITS

    alpha_f = branch_alpha.detach().float().clamp_min(ENTROPY_CLAMP_EPS)
    alpha_norm = alpha_f / alpha_f.sum(dim=-1, keepdim=True).clamp_min(ENTROPY_CLAMP_EPS)
    vocab_ent_bits = -(alpha_norm * torch.log(alpha_norm)).sum(dim=-1) * NATS_TO_BITS

    r_num = int(beta_norm.size(-1))
    max_possible_bits = math.log2(float(max(1, r_num))) if r_num > 1 else 0.0
    return {
        "num_meaning_branches": r_num,
        "mean_context_routing_entropy_bits": round(float(ctx_ent_bits.mean().item()), 6),
        "max_possible_routing_entropy_bits": round(max_possible_bits, 6),
        "context_beta0_temporal_std": round(float(beta_norm[:, 0].std().item()), 6),
        "mean_vocab_branch_entropy_bits": round(float(vocab_ent_bits.mean().item()), 6),
        "mean_branch_beta_weights": [round(float(v), 6) for v in beta_norm.mean(dim=0).tolist()],
    }


@torch.no_grad()
def run_polysemy_entropy_and_angle_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Execute multi-branch polysemy routing entropy and subspace angle audit (`<= 8` args)."""
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
    x_batch, _ = load_kronecker_probe_batch(
        val_shard_path=str(args.val_shard),
        batch_size=2,
        seq_len=min(32, int(args.seq_len)),
        device=str(args.device),
    )

    _, _, branch_beta, branch_alpha = model.extract_readout_states(x_batch)
    routing_stats = compute_routing_beta_statistics(
        branch_beta=branch_beta,
        branch_alpha=branch_alpha[: min(DEFAULT_MAX_VOCAB_SUBSET, int(branch_alpha.size(0)))],
    )
    overlap_stats = compute_pairwise_complex_branch_overlap(
        embed_re=model.embed_re,
        embed_im=model.embed_im,
        max_tokens=min(DEFAULT_MAX_VOCAB_SUBSET, int(model.cfg.valid_vocab_size)),
    )

    metrics: dict[str, Any] = {
        "arch": metadata["arch"],
        "checkpoint_mode": metadata["checkpoint_mode"],
        **routing_stats,
        "mean_kronecker_branch_overlap": overlap_stats["mean_kronecker_branch_overlap"],
        "max_kronecker_branch_overlap": overlap_stats["max_kronecker_branch_overlap"],
        "mean_factor_branch_overlap": overlap_stats["mean_factor_branch_overlap"],
        "subspace_angle_deg": overlap_stats["subspace_angle_deg"],
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
        extra_payload={"branch_overlap": overlap_stats},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Audit multi-branch polysemy routing entropy and subspace angles.")
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
    payload = run_polysemy_entropy_and_angle_probe(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} angle_deg={payload['metrics']['subspace_angle_deg']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
