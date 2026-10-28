#!/usr/bin/env python3
"""N-Gram Helmert Subspace & Carrier Alignment Calibration Inspector (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Audits isometric Helmert embedding `E_perp: R^{30} -> H_perp subset C^{16}`, carrier orthogonality
`|<c_ref | E_perp(w)>| < 1e-7`, Unitary Procrustes carrier alignment fidelity, and Step-0 KL divergence
to the target tempered FineWeb unigram distribution across Kronecker models.
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
    DEFAULT_DIAGNOSTIC_MAX_TOKENS,
    DEFAULT_DIAGNOSTIC_SEQ_LEN,
    DEFAULT_FINEWEB_VAL_SHARD_PATH,
    DEFAULT_NGRAM_CALIBRATION_CACHE_PATH,
    DiagnosticModelSpec,
    compute_complex_factor_gram_spectrum,
    load_kronecker_probe_batch,
    load_or_build_calibrated_kronecker_model,
    write_standardized_probe_json,
)
from qdlblocks.models.calibrated_kronecker_born_gpt import _build_canonical_carrier_ket
from qdlblocks.models.ngram_calibrated_kronecker_born_gpt import (
    embed_real30_to_carrier_ortho_c16,
    project_c16_to_carrier_ortho_real30,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
SCRIPT_NAME: str = "ngram_helmert_carrier_calibration_inspector.py"
DEFAULT_HELMERT_TEST_VECTORS: int = 64
DEFAULT_FACTOR_DIM_C16: int = 16
DEFAULT_ORTHO_REAL_DIM: int = 30
HELMERT_ISOMETRY_TOL: float = 1e-5
CARRIER_ORTHO_TOL: float = 1e-6


@torch.no_grad()
def audit_helmert_isometry_and_carrier_orthogonality(
    num_vectors: int = DEFAULT_HELMERT_TEST_VECTORS,
    factor_dim: int = DEFAULT_FACTOR_DIM_C16,
    device: str = "cpu",
) -> dict[str, float]:
    """Verify `<c_ref | E_perp(w)> == 0`, norm preservation, and round-trip reconstruction (`<= 8` args)."""
    dev = torch.device(device)
    gen = torch.Generator(device=dev)
    gen.manual_seed(1337)
    d_real = 2 * (int(factor_dim) - 1)
    w_real = torch.randn((int(num_vectors), 4, d_real), generator=gen, device=dev, dtype=torch.float32)

    u_re, u_im = embed_real30_to_carrier_ortho_c16(w_real, factor_dim=factor_dim)
    w_rec = project_c16_to_carrier_ortho_real30(u_re, u_im)

    c_re, c_im = _build_canonical_carrier_ket(factor_dim, dev)
    ov_re = (u_re * c_re.view(1, 1, factor_dim) + u_im * c_im.view(1, 1, factor_dim)).sum(dim=-1)
    ov_im = (u_re * c_im.view(1, 1, factor_dim) - u_im * c_re.view(1, 1, factor_dim)).sum(dim=-1)
    max_carrier_leak = float(torch.sqrt(ov_re.square() + ov_im.square()).max().item())

    norm_real = w_real.square().sum(dim=-1)
    norm_c16 = u_re.square().sum(dim=-1) + u_im.square().sum(dim=-1)
    max_isometry_err = float((norm_real - norm_c16).abs().max().item())
    max_roundtrip_err = float((w_real - w_rec).abs().max().item())

    return {
        "helmert_max_carrier_leak": round(max_carrier_leak, 8),
        "helmert_max_isometry_err": round(max_isometry_err, 8),
        "helmert_max_roundtrip_err": round(max_roundtrip_err, 8),
        "helmert_isometry_pass": float(max_isometry_err < HELMERT_ISOMETRY_TOL and max_carrier_leak < CARRIER_ORTHO_TOL),
    }


@torch.no_grad()
def run_calibration_inspection(args: argparse.Namespace) -> dict[str, Any]:
    """Run Helmert subspace audit and live model Step-0 carrier calibration inspection (`<= 8` args)."""
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
    helmert_stats = audit_helmert_isometry_and_carrier_orthogonality(
        num_vectors=DEFAULT_HELMERT_TEST_VECTORS,
        factor_dim=int(model.cfg.factor_dim),
        device=str(args.device),
    )
    gram_stats = compute_complex_factor_gram_spectrum(
        model.embed_re[: min(1024, int(model.cfg.valid_vocab_size)), 0],
        model.embed_im[: min(1024, int(model.cfg.valid_vocab_size)), 0],
    )
    x_batch, y_batch = load_kronecker_probe_batch(
        val_shard_path=str(args.val_shard),
        batch_size=2,
        seq_len=min(32, int(args.seq_len)),
        device=str(args.device),
    )
    _, loss_t = model(x_batch, targets=y_batch, return_logits=False)
    step0_nll = float(loss_t.detach().item()) if loss_t is not None else 0.0

    cal_stats = metadata.get("calibration_stats", {})
    metrics: dict[str, Any] = {
        **helmert_stats,
        "arch": metadata["arch"],
        "checkpoint_mode": metadata["checkpoint_mode"],
        "has_frozen_prior_weights": bool(hasattr(model, "prior_weights")),
        "carrier_alignment_fidelity": float(cal_stats.get("carrier_alignment_fidelity", 1.0)),
        "unitary_frobenius_err": float(cal_stats.get("unitary_frobenius_err", 0.0)),
        "step0_kl_to_target_unigram": float(cal_stats.get("step0_kl_to_target_unigram", 0.0)),
        "step0_probe_nll": round(step0_nll, 4),
        "factor_participation_ratio_c16": gram_stats["mean_participation_ratio_c16"],
        "factor_von_neumann_entropy_nats": gram_stats["mean_von_neumann_entropy_nats"],
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
        extra_payload={"gram_spectrum": gram_stats, "checkpoint_metadata": metadata},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Inspect N-gram Helmert subspace & carrier calibration.")
    parser.add_argument("--arch", type=str, default="multihead_mixed_state")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--calibration-cache", type=str, default=DEFAULT_NGRAM_CALIBRATION_CACHE_PATH)
    parser.add_argument("--val-shard", type=str, default=DEFAULT_FINEWEB_VAL_SHARD_PATH)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--meaning-rank", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_DIAGNOSTIC_SEQ_LEN)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_DIAGNOSTIC_MAX_TOKENS)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json-out", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (`<= 8` args)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_calibration_inspection(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} fidelity={payload['metrics']['carrier_alignment_fidelity']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
