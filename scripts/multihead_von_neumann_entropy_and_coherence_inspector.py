#!/usr/bin/env python3
"""Multi-Head von Neumann Entropy & Cross-Head Coherence Inspector (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Computes layerwise cross-head Gram eigenvalues `(lambda_1, ..., lambda_{H_g})`, mixed-state purity
`Tr(rho_mix^2)`, von Neumann entropy `S(rho_mix)`, off-diagonal coherence L1 norm, and Stinespring
unitarity error across both `MultiHeadMixedStateKroneckerGPT` and `NgramCalibratedKroneckerBornGPT`.
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
SCRIPT_NAME: str = "multihead_von_neumann_entropy_and_coherence_inspector.py"
DEFAULT_INSPECT_BATCH_SIZE: int = 2
DEFAULT_INSPECT_SEQ_LEN: int = 32
COHERENCE_EPS: float = 1e-8


@torch.no_grad()
def compute_layer_entropy_and_coherence(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
) -> dict[str, Any]:
    """Compute per-layer purity, von Neumann entropy, off-diagonal coherence, and Stinespring unitarity (`<= 8` args)."""
    phi_factors = model.extract_readout_states(x_batch)[0].detach().float()
    d_f = int(model.cfg.factor_dim)
    phi_re, phi_im = phi_factors[..., :d_f], phi_factors[..., d_f:]
    gram_stats = compute_complex_factor_gram_spectrum(phi_re[:, 0], phi_im[:, 0])

    # Off-diagonal cross-factor coherence L1 norm:
    m_num = int(phi_re.size(2))
    zc = torch.complex(phi_re[:, 0], phi_im[:, 0])
    zc_unit = zc / torch.sqrt((phi_re[:, 0].square() + phi_im[:, 0].square()).sum(dim=-1, keepdim=True).clamp_min(COHERENCE_EPS))
    gram_m = torch.einsum("nmd,nkd->mk", zc_unit.conj(), zc_unit).abs() / float(max(1, zc_unit.size(0)))
    eye_m = torch.eye(m_num, device=gram_m.device)
    off_diag_l1 = float(((gram_m * (1.0 - eye_m)).sum() / max(1.0, float(m_num * (m_num - 1)))).item())

    stinespring_err = float(model.compute_all_unitary_frobenius_error()) if hasattr(model, "compute_all_unitary_frobenius_error") else 0.0
    per_layer: list[dict[str, float]] = []
    for idx_l, blk in enumerate(getattr(model, "blocks", [])):
        attn = getattr(blk, "attn", None)
        p_l = float(getattr(attn, "last_purity_mean", gram_stats["mean_purity"]))
        per_layer.append({
            "layer": float(idx_l),
            "mean_purity": round(p_l, 6),
            "mean_von_neumann_entropy": gram_stats["mean_von_neumann_entropy_nats"],
            "off_diag_coherence_l1": round(off_diag_l1, 6),
        })

    return {
        "mean_purity": gram_stats["mean_purity"],
        "mean_von_neumann_entropy": gram_stats["mean_von_neumann_entropy_nats"],
        "off_diag_coherence_l1": round(off_diag_l1, 6),
        "stinespring_unitarity_err": round(stinespring_err, 6),
        "per_layer": per_layer,
    }


@torch.no_grad()
def run_entropy_and_coherence_inspector(args: argparse.Namespace) -> dict[str, Any]:
    """Run multi-head von Neumann entropy and coherence inspection (`<= 8` args)."""
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
        batch_size=DEFAULT_INSPECT_BATCH_SIZE,
        seq_len=min(DEFAULT_INSPECT_SEQ_LEN, int(args.seq_len)),
        device=str(args.device),
    )
    stats = compute_layer_entropy_and_coherence(model, x_batch)
    metrics: dict[str, Any] = {
        "arch": metadata["arch"],
        "checkpoint_mode": metadata["checkpoint_mode"],
        "mean_purity": stats["mean_purity"],
        "mean_von_neumann_entropy": stats["mean_von_neumann_entropy"],
        "off_diag_coherence_l1": stats["off_diag_coherence_l1"],
        "stinespring_unitarity_err": stats["stinespring_unitarity_err"],
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
        extra_payload={"layers": stats["per_layer"]},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Inspect layerwise multi-head von Neumann entropy & coherence.")
    parser.add_argument("--arch", type=str, default="multihead_mixed_state")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--calibration-cache", type=str, default=DEFAULT_NGRAM_CALIBRATION_CACHE_PATH)
    parser.add_argument("--val-shard", type=str, default=DEFAULT_FINEWEB_VAL_SHARD_PATH)
    parser.add_argument("--num-samples", type=int, default=64)
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
    payload = run_entropy_and_coherence_inspector(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} entropy={payload['metrics']['mean_von_neumann_entropy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
