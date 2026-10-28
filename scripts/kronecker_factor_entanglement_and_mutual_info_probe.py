#!/usr/bin/env python3
"""Kronecker 4-Factor Entanglement & Mutual Information Probe (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Measures per-factor participation ratios `R_factor^{(m)}` (`m in {0, 1, 2, 3}`), the `4 x 4`
pairwise empirical factor correlation matrix, and quantum Schmidt entanglement entropy across
the `16 x 16 x 16 x 16` Kronecker hierarchy.
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
    compute_complex_factor_gram_spectrum,
    load_kronecker_probe_batch,
    load_or_build_calibrated_kronecker_model,
    write_standardized_probe_json,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
SCRIPT_NAME: str = "kronecker_factor_entanglement_and_mutual_info_probe.py"
DEFAULT_VOCAB_SUBSET: int = 2048
CORR_EPS: float = 1e-8


@torch.no_grad()
def compute_4x4_factor_correlation_and_entanglement(
    z_re: torch.Tensor,
    z_im: torch.Tensor,
) -> dict[str, Any]:
    """Compute `4x4` pairwise cross-factor correlation matrix and participation ratios (`<= 8` args)."""
    gram_stats = compute_complex_factor_gram_spectrum(z_re, z_im)
    zr, zi = z_re.detach().float(), z_im.detach().float()
    n_samples, m_num, _ = zr.shape
    zc = torch.complex(zr, zi)
    zc = zc / torch.sqrt((zr.square() + zi.square()).sum(dim=-1, keepdim=True).clamp_min(CORR_EPS))

    corr_matrix: list[list[float]] = []
    off_diag_vals: list[float] = []
    for m1 in range(m_num):
        row_vals: list[float] = []
        for m2 in range(m_num):
            ov = (zc[:, m1].conj() * zc[:, m2]).sum(dim=-1).abs()
            c_val = float(ov.mean().item())
            row_vals.append(round(c_val, 6))
            if m1 != m2:
                off_diag_vals.append(c_val)
        corr_matrix.append(row_vals)

    mean_off_diag = float(sum(off_diag_vals) / max(1, len(off_diag_vals))) if off_diag_vals else 0.0
    return {
        "effective_factor_rank_c16": gram_stats["mean_participation_ratio_c16"],
        "effective_factor_rank_real32": gram_stats["mean_participation_ratio_real32"],
        "per_factor_participation_ratios": gram_stats["per_factor_participation_ratios"],
        "mean_cross_factor_correlation": round(mean_off_diag, 6),
        "schmidt_entanglement_entropy_nats": gram_stats["mean_von_neumann_entropy_nats"],
        "factor_correlation_matrix_4x4": corr_matrix,
    }


@torch.no_grad()
def run_factor_entanglement_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Execute 4-factor Kronecker entanglement and mutual information probe (`<= 8` args)."""
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
    n_sub = min(DEFAULT_VOCAB_SUBSET, int(model.cfg.valid_vocab_size))
    vocab_ent = compute_4x4_factor_correlation_and_entanglement(
        model.embed_re[:n_sub, 0],
        model.embed_im[:n_sub, 0],
    )
    phi_factors = model.extract_readout_states(x_batch)[0].detach().float()
    d_f = int(model.cfg.factor_dim)
    ctx_ent = compute_4x4_factor_correlation_and_entanglement(
        phi_factors[:, 0, :, :d_f],
        phi_factors[:, 0, :, d_f:],
    )

    metrics: dict[str, Any] = {
        "arch": metadata["arch"],
        "checkpoint_mode": metadata["checkpoint_mode"],
        "vocab_effective_factor_rank_real32": vocab_ent["effective_factor_rank_real32"],
        "vocab_mean_cross_factor_correlation": vocab_ent["mean_cross_factor_correlation"],
        "ctx_effective_factor_rank_real32": ctx_ent["effective_factor_rank_real32"],
        "ctx_schmidt_entanglement_entropy_nats": ctx_ent["schmidt_entanglement_entropy_nats"],
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
        extra_payload={
            "factor_correlation_matrix_4x4": vocab_ent["factor_correlation_matrix_4x4"],
            "per_factor_participation_ratios": vocab_ent["per_factor_participation_ratios"],
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Measure 4-factor Kronecker entanglement and correlation matrix.")
    parser.add_argument("--arch", type=str, default="multihead_mixed_state")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--calibration-cache", type=str, default=DEFAULT_NGRAM_CALIBRATION_CACHE_PATH)
    parser.add_argument("--val-shard", type=str, default=DEFAULT_FINEWEB_VAL_SHARD_PATH)
    parser.add_argument("--max-tokens", type=int, default=256)
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
    payload = run_factor_entanglement_probe(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} rank={payload['metrics']['vocab_effective_factor_rank_real32']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
