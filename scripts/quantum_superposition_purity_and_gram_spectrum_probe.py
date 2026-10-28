#!/usr/bin/env python3
"""Quantum Superposition Purity & Cross-Head Gram Spectrum Probe (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Measures layerwise and readout-level quantum superposition purity `Tr(rho^2)`, effective mixed-state
rank `1 / Tr(rho^2)`, von Neumann entropy `S_vN(rho)`, and Hermitian Gram eigenvalue spectra on FineWeb.
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
SCRIPT_NAME: str = "quantum_superposition_purity_and_gram_spectrum_probe.py"
DEFAULT_PROBE_BATCH_SIZE: int = 2
DEFAULT_PROBE_SEQ_LEN: int = 32
PURITY_FLOOR_EPS: float = 1e-8


@torch.no_grad()
def inspect_superposition_purity_and_spectrum(args: argparse.Namespace) -> dict[str, Any]:
    """Run forward pass on FineWeb slice and compute superposition purity & Gram spectrum (`<= 8` args)."""
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
        batch_size=DEFAULT_PROBE_BATCH_SIZE,
        seq_len=min(DEFAULT_PROBE_SEQ_LEN, int(args.seq_len)),
        device=str(args.device),
    )

    phi_factors = model.extract_readout_states(x_batch)[0].detach().float()
    d_f = int(model.cfg.factor_dim)
    phi_re, phi_im = phi_factors[..., :d_f], phi_factors[..., d_f:]
    readout_gram = compute_complex_factor_gram_spectrum(phi_re[:, 0], phi_im[:, 0])

    layer_purities: list[float] = []
    for blk in getattr(model, "blocks", []):
        attn_mod = getattr(blk, "attn", None)
        if attn_mod is not None and hasattr(attn_mod, "last_purity_mean"):
            layer_purities.append(round(float(getattr(attn_mod, "last_purity_mean", readout_gram["mean_purity"])), 6))
        else:
            layer_purities.append(readout_gram["mean_purity"])

    mean_layer_purity = float(sum(layer_purities) / max(1, len(layer_purities)))
    metrics: dict[str, Any] = {
        "arch": metadata["arch"],
        "checkpoint_mode": metadata["checkpoint_mode"],
        "mean_layer_purity": round(mean_layer_purity, 6),
        "mean_layer_mixed_rank": round(1.0 / max(mean_layer_purity, PURITY_FLOOR_EPS), 4),
        "readout_participation_ratio_c16": readout_gram["mean_participation_ratio_c16"],
        "readout_von_neumann_entropy_nats": readout_gram["mean_von_neumann_entropy_nats"],
        "top_gram_eigenvalue": readout_gram["top_eigenvalue_share"],
        "layer_purities": layer_purities,
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
        extra_payload={"readout_gram_spectrum": readout_gram},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Inspect quantum superposition purity and Hermitian Gram spectrum.")
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
    payload = inspect_superposition_purity_and_spectrum(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} purity={payload['metrics']['mean_layer_purity']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
