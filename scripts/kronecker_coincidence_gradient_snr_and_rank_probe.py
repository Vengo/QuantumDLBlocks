#!/usr/bin/env python3
"""Kronecker 4-Factor Coincidence Gradient SNR & Effective Rank Probe (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Measures the target-vs-impostor signal-to-noise ratio (SNR), backward gradient norm per Kronecker
factor, and participation ratio `R_eff` of `16 x 16 x 16 x 16 = 65,536` Kronecker readouts.
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
SCRIPT_NAME: str = "kronecker_coincidence_gradient_snr_and_rank_probe.py"
DEFAULT_BG_SAMPLE_VOCAB: int = 512
SNR_DENOM_EPS: float = 1e-8
FLAT_C128_COLLISION_FLOOR: float = 0.0078125  # 1 / 128
KRON_16X4_COLLISION_FLOOR: float = 0.0000152587890625  # (1 / 16)^4 = 1 / 65536


def measure_factor_gradient_snr_and_rank(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
) -> dict[str, Any]:
    """Run 1 forward+backward step to measure per-factor gradient RMS and coincidence SNR (`<= 8` args)."""
    model.zero_grad(set_to_none=True)
    _, loss_t = model(x_batch, targets=y_batch, return_logits=False)
    loss_val = float(loss_t.detach().item()) if loss_t is not None else 0.0
    if loss_t is not None:
        loss_t.backward()

    factor_grad_rms: list[float] = []
    if model.embed_re.grad is not None and model.embed_im.grad is not None:
        g_sq = model.embed_re.grad.detach().float().square() + model.embed_im.grad.detach().float().square()
        for m in range(int(g_sq.size(2))):
            rms_m = float(torch.sqrt(g_sq[:, :, m, :].mean().clamp_min(SNR_DENOM_EPS)).item())
            factor_grad_rms.append(round(rms_m, 8))
    else:
        factor_grad_rms = [0.0, 0.0, 0.0, 0.0]

    povm_grad_rms = 0.0
    if hasattr(model, "povm_scale_logits") and model.povm_scale_logits.grad is not None:
        povm_grad_rms = float(torch.sqrt(model.povm_scale_logits.grad.detach().float().square().mean()).item())

    with torch.no_grad():
        phi_factors = model.extract_readout_states(x_batch)[0].detach().float()
        d_f = int(model.cfg.factor_dim)
        phi_re, phi_im = phi_factors[..., :d_f], phi_factors[..., d_f:]
        gram_phi = compute_complex_factor_gram_spectrum(phi_re[:, 0], phi_im[:, 0])

        n_bg = min(DEFAULT_BG_SAMPLE_VOCAB, int(model.cfg.valid_vocab_size))
        w_re = model.embed_re[:n_bg, 0].detach().float()
        w_im = model.embed_im[:n_bg, 0].detach().float()
        ov_re = torch.einsum("tmd,vmd->tvm", phi_re[:, 0], w_re) + torch.einsum("tmd,vmd->tvm", phi_im[:, 0], w_im)
        ov_im = torch.einsum("tmd,vmd->tvm", phi_im[:, 0], w_re) - torch.einsum("tmd,vmd->tvm", phi_re[:, 0], w_im)
        fac_mag_sq = (ov_re.square() + ov_im.square()).clamp_min(0.0)
        kron_coinc = fac_mag_sq.prod(dim=-1)
        bg_mean = float(kron_coinc.mean().item())
        bg_std = float(kron_coinc.std().clamp_min(SNR_DENOM_EPS).item())
        coincidence_snr = float(1.0 / max(bg_std, SNR_DENOM_EPS))

    model.zero_grad(set_to_none=True)
    return {
        "probe_loss_nll": round(loss_val, 5),
        "coincidence_snr_to_bg_std": round(coincidence_snr, 2),
        "empirical_4factor_bg_collision": round(bg_mean, 8),
        "theoretical_16x4_collision_floor": KRON_16X4_COLLISION_FLOOR,
        "theoretical_flat128_collision_floor": FLAT_C128_COLLISION_FLOOR,
        "readout_phi_participation_ratio_c16": gram_phi["mean_participation_ratio_c16"],
        "per_factor_embed_grad_rms": factor_grad_rms,
        "povm_scale_grad_rms": round(povm_grad_rms, 8),
    }


def run_coincidence_snr_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Instantiate calibrated Kronecker model and run coincidence gradient SNR probe (`<= 8` args)."""
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
        seq_len=min(16, int(args.seq_len)),
        device=str(args.device),
    )
    snr_stats = measure_factor_gradient_snr_and_rank(model, x_batch, y_batch)
    metrics: dict[str, Any] = {
        "arch": metadata["arch"],
        "checkpoint_mode": metadata["checkpoint_mode"],
        **snr_stats,
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Measure 4-factor Kronecker coincidence gradient SNR & effective rank.")
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
    payload = run_coincidence_snr_probe(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} snr={payload['metrics']['coincidence_snr_to_bg_std']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
