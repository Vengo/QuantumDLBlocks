#!/usr/bin/env python3
"""Polysemy Router Token Sense Clustering CLI (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Audits which contexts activate each meaning branch `beta_{t,r}` on polysemous probe words
(`bank,light,run,state,wave`), computing `mean_beta_weight`, `branch_specialization_entropy_bits`,
`pairwise_branch_cosine`, and top sense-clustered token IDs per branch `r in {0, ..., R-1}`.
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
    compute_pairwise_complex_branch_overlap,
    load_kronecker_probe_batch,
    load_or_build_calibrated_kronecker_model,
    write_standardized_probe_json,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
SCRIPT_NAME: str = "polysemy_router_token_sense_clustering_cli.py"
DEFAULT_PROBE_WORDS_CSV: str = "bank,light,run,state,wave"
TOP_K_CLUSTER_TOKENS: int = 15
CANDIDATE_VOCAB_LIMIT: int = 1024
NATS_TO_BITS: float = 1.4426950408889634
ENTROPY_EPS: float = 1e-8


@torch.no_grad()
def cluster_polysemy_branch_senses(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
    probe_words: list[str],
) -> dict[str, Any]:
    """Compute per-branch beta weights, specialization entropy, and top-15 token clusters (`<= 8` args)."""
    phi_factors, psi_factors, branch_beta, _ = model.extract_readout_states(x_batch)
    beta_f = branch_beta.detach().float().clamp_min(ENTROPY_EPS)
    beta_norm = beta_f / beta_f.sum(dim=-1, keepdim=True).clamp_min(ENTROPY_EPS)
    ent_bits = float((-(beta_norm * torch.log(beta_norm)).sum(dim=-1) * NATS_TO_BITS).mean().item())
    mean_beta = [round(float(v), 6) for v in beta_norm.mean(dim=0).tolist()]

    overlap = compute_pairwise_complex_branch_overlap(model.embed_re, model.embed_im, max_tokens=CANDIDATE_VOCAB_LIMIT)
    n_cand = min(CANDIDATE_VOCAB_LIMIT, int(psi_factors.size(0)))
    r_num = int(phi_factors.size(1))

    branch_clusters: dict[str, list[int]] = {}
    for r in range(r_num):
        phi_r = phi_factors[:, r].mean(dim=0).float()
        psi_r = psi_factors[:n_cand, r].float()
        scores = (psi_r * phi_r.unsqueeze(0)).sum(dim=(-1, -2))
        top_ids = torch.topk(scores, k=min(TOP_K_CLUSTER_TOKENS, n_cand)).indices.tolist()
        branch_clusters[f"branch_{r}"] = [int(i) for i in top_ids]

    return {
        "probe_words": probe_words,
        "mean_beta_weight": mean_beta,
        "branch_specialization_entropy_bits": round(ent_bits, 6),
        "pairwise_branch_cosine": overlap["mean_kronecker_branch_overlap"],
        "top_sense_clustered_tokens": branch_clusters,
    }


@torch.no_grad()
def run_polysemy_sense_clustering(args: argparse.Namespace) -> dict[str, Any]:
    """Execute polysemy router token sense clustering CLI (`<= 8` args)."""
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
    words = [w.strip() for w in str(args.probe_words).split(",") if w.strip()]
    cluster_stats = cluster_polysemy_branch_senses(model, x_batch, words)

    metrics: dict[str, Any] = {
        "arch": metadata["arch"],
        "checkpoint_mode": metadata["checkpoint_mode"],
        "mean_beta_weight": cluster_stats["mean_beta_weight"],
        "branch_specialization_entropy_bits": cluster_stats["branch_specialization_entropy_bits"],
        "pairwise_branch_cosine": cluster_stats["pairwise_branch_cosine"],
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
        extra_payload={"top_sense_clustered_tokens": cluster_stats["top_sense_clustered_tokens"], "probe_words": words},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Cluster token senses per polysemy meaning branch.")
    parser.add_argument("--arch", type=str, default="multihead_mixed_state")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--probe-words", "--tokens", dest="probe_words", type=str, default=DEFAULT_PROBE_WORDS_CSV)
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
    payload = run_polysemy_sense_clustering(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} entropy_bits={payload['metrics']['branch_specialization_entropy_bits']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
