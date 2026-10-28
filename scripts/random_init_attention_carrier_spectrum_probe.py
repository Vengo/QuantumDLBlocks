#!/usr/bin/env python3
"""Random-Init Attention Carrier Spectrum Probe (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Measures Step-0 random-init attention output carrier overlap `|<c_ref | psi_{t,r,m}>|^2` and
singular value spectrum across sequence positions `t in [0, T)` before and after Unitary Procrustes
readout alignment.
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
from qdlblocks.layers.packed_complex_register_density_attn import complex_l2_normalize
from qdlblocks.models.calibrated_kronecker_born_gpt import _build_canonical_carrier_ket
from qdlblocks.models.ngram_calibrated_kronecker_born_gpt import _apply_factor_unitary_bank

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
SCRIPT_NAME: str = "random_init_attention_carrier_spectrum_probe.py"
DEFAULT_PROBE_SEQ_LEN: int = 32


@torch.no_grad()
def measure_pre_and_post_procrustes_carrier_overlap(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
) -> dict[str, Any]:
    """Measure `|<c_ref | psi_{t,r,m}>|^2` before and after `out_factor_u` Procrustes rotation (`<= 8` args)."""
    d_f = int(model.cfg.factor_dim)
    dev = x_batch.device
    c_re, c_im = _build_canonical_carrier_ket(d_f, dev)

    q_re, q_im = model._collect_backbone_pre_operator_states(x_batch)[:2]
    q_norm_re, q_norm_im = complex_l2_normalize(q_re, q_im, dim=-1)

    pre_ov_re = (q_norm_re * c_re.view(1, 1, d_f) + q_norm_im * c_im.view(1, 1, d_f)).sum(dim=-1)
    pre_ov_im = (q_norm_re * c_im.view(1, 1, d_f) - q_norm_im * c_re.view(1, 1, d_f)).sum(dim=-1)
    pre_carrier_overlap_sq = float((pre_ov_re.square() + pre_ov_im.square()).mean().item())

    rot_re, rot_im = _apply_factor_unitary_bank(
        q_norm_re,
        q_norm_im,
        model.out_factor_u_re.detach().float(),
        model.out_factor_u_im.detach().float(),
    )
    post_ov_re = (rot_re * c_re.view(1, 1, d_f) + rot_im * c_im.view(1, 1, d_f)).sum(dim=-1)
    post_ov_im = (rot_re * c_im.view(1, 1, d_f) - rot_im * c_re.view(1, 1, d_f)).sum(dim=-1)
    post_carrier_overlap_sq = float((post_ov_re.square() + post_ov_im.square()).mean().item())

    m_num = int(model.cfg.num_factors)
    gram_post = compute_complex_factor_gram_spectrum(
        rot_re.reshape(-1, m_num, d_f),
        rot_im.reshape(-1, m_num, d_f),
    )
    return {
        "pre_procrustes_carrier_overlap_sq": round(pre_carrier_overlap_sq, 6),
        "post_procrustes_carrier_overlap_sq": round(post_carrier_overlap_sq, 6),
        "carrier_alignment_fidelity": float(model.compute_carrier_alignment_fidelity(q_norm_re.mean(dim=0), q_norm_im.mean(dim=0))),
        "attention_output_participation_ratio_c16": gram_post["mean_participation_ratio_c16"],
        "attention_output_von_neumann_entropy_nats": gram_post["mean_von_neumann_entropy_nats"],
    }


@torch.no_grad()
def run_attention_carrier_spectrum_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Execute random-init attention carrier spectrum probe (`<= 8` args)."""
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
        seq_len=min(DEFAULT_PROBE_SEQ_LEN, int(args.seq_len)),
        device=str(args.device),
    )
    carrier_stats = measure_pre_and_post_procrustes_carrier_overlap(model, x_batch)
    metrics: dict[str, Any] = {
        "arch": metadata["arch"],
        "checkpoint_mode": metadata["checkpoint_mode"],
        **carrier_stats,
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Analyze random-init complex attention carrier overlap and spectrum.")
    parser.add_argument("--arch", type=str, default="multihead_mixed_state")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--layers", type=str, default="2,4,6")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_DIAGNOSTIC_SEQ_LEN)
    parser.add_argument("--calibration-cache", type=str, default=DEFAULT_NGRAM_CALIBRATION_CACHE_PATH)
    parser.add_argument("--val-shard", type=str, default=DEFAULT_FINEWEB_VAL_SHARD_PATH)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--meaning-rank", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json-out", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (`<= 8` args)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_attention_carrier_spectrum_probe(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} post_ov={payload['metrics']['post_procrustes_carrier_overlap_sq']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
