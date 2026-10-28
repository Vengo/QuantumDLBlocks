#!/usr/bin/env python3
"""Calibrated Kronecker Vocabulary Spectrum & POVM Drift Probe (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Audits the vocabulary factor Gram eigenvalue spectrum across all 4 Kronecker factors, global
sharp effective rank `R_emb`, carrier-tilt distribution `|<c_ref | psi_{v,r}^{(m)}>|`, and
POVM operator weight drift `||w_povm - w_povm^{(0)}||` across Zipf frequency deciles.
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
    DEFAULT_NUM_FREQUENCY_DECILES,
    DiagnosticModelSpec,
    compute_complex_factor_gram_spectrum,
    compute_fineweb_10_frequency_deciles,
    load_or_build_calibrated_kronecker_model,
    write_standardized_probe_json,
)
from qdlblocks.models.calibrated_kronecker_born_gpt import _build_canonical_carrier_ket
from qdlblocks.models.kronecker_multimeaning_16x4_gpt import DEFAULT_BRANCH_EPS

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
SCRIPT_NAME: str = "calibrated_kronecker_vocab_spectrum_and_povm_drift_probe.py"
DEFAULT_VOCAB_SPECTRUM_SAMPLES: int = 2048
ALGEBRAIC_POS_HALF: float = 0.5


@torch.no_grad()
def compute_vocab_carrier_and_povm_deciles(
    model: torch.nn.Module,
    num_deciles: int = DEFAULT_NUM_FREQUENCY_DECILES,
) -> dict[str, Any]:
    """Compute carrier alignment and POVM scale drift across `num_deciles` Zipf buckets (`<= 8` args)."""
    v_eff = int(model.cfg.valid_vocab_size)
    d_f = int(model.cfg.factor_dim)
    dev = model.embed_re.device

    uni_p = model.unigram_probs[:v_eff].detach().float() if hasattr(model, "unigram_probs") else torch.ones((v_eff,), device=dev) / float(v_eff)
    deciles = compute_fineweb_10_frequency_deciles(uni_p, active_vocab_size=v_eff, num_deciles=num_deciles)

    theta_cur = model.povm_scale_logits[:v_eff].detach().float()
    theta_init = model.initial_povm_logits[:v_eff].detach().float() if hasattr(model, "initial_povm_logits") else theta_cur
    w_cur = ALGEBRAIC_POS_HALF * (theta_cur + torch.sqrt(theta_cur.square() + 1.0)) + DEFAULT_BRANCH_EPS
    w_init = ALGEBRAIC_POS_HALF * (theta_init + torch.sqrt(theta_init.square() + 1.0)) + DEFAULT_BRANCH_EPS
    w_drift = (w_cur - w_init).abs()

    c_re, c_im = _build_canonical_carrier_ket(d_f, dev)
    er = model.embed_re[:v_eff, 0].detach().float()
    ei = model.embed_im[:v_eff, 0].detach().float()
    c_ov_re = (er * c_re.view(1, 1, d_f) + ei * c_im.view(1, 1, d_f)).sum(dim=-1).mean(dim=-1)

    decile_rows: list[dict[str, float]] = []
    for d_idx in range(int(num_deciles)):
        mask = deciles == d_idx
        if int(mask.sum().item()) == 0:
            continue
        decile_rows.append({
            "decile": float(d_idx),
            "token_count": float(mask.sum().item()),
            "mean_povm_weight": round(float(w_cur[mask].mean().item()), 6),
            "mean_povm_abs_drift": round(float(w_drift[mask].mean().item()), 6),
            "mean_carrier_tilt_overlap": round(float(c_ov_re[mask].mean().item()), 6),
        })

    return {
        "global_povm_l2_drift": round(float(torch.sqrt(w_drift.square().mean()).item()), 6),
        "top_decile_carrier_tilt": decile_rows[0]["mean_carrier_tilt_overlap"] if decile_rows else 0.0,
        "rare_decile_carrier_tilt": decile_rows[-1]["mean_carrier_tilt_overlap"] if decile_rows else 0.0,
        "decile_breakdown": decile_rows,
    }


@torch.no_grad()
def run_vocab_spectrum_and_povm_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Run vocabulary spectrum and POVM drift diagnostic (`<= 8` args)."""
    ckpt = args.checkpoint or args.results_json
    spec = DiagnosticModelSpec(
        arch=str(args.arch),
        checkpoint_path=ckpt,
        calibration_cache_path=str(args.calibration_cache),
        n_layer=int(args.num_layers),
        meaning_rank=int(args.meaning_rank),
        seq_len=int(args.seq_len),
        device=str(args.device),
    )
    model, metadata = load_or_build_calibrated_kronecker_model(spec)
    n_sub = min(DEFAULT_VOCAB_SPECTRUM_SAMPLES, int(model.cfg.valid_vocab_size))
    gram_stats = compute_complex_factor_gram_spectrum(model.embed_re[:n_sub, 0], model.embed_im[:n_sub, 0])
    povm_stats = compute_vocab_carrier_and_povm_deciles(model, num_deciles=DEFAULT_NUM_FREQUENCY_DECILES)
    sharp_rank = float(model.compute_sharp_eff_rank()) if hasattr(model, "compute_sharp_eff_rank") else gram_stats["mean_participation_ratio_real32"]

    metrics: dict[str, Any] = {
        "arch": metadata["arch"],
        "checkpoint_mode": metadata["checkpoint_mode"],
        "sharp_eff_rank": round(sharp_rank, 4),
        "mean_factor_participation_ratio_c16": gram_stats["mean_participation_ratio_c16"],
        "mean_factor_von_neumann_entropy_nats": gram_stats["mean_von_neumann_entropy_nats"],
        "global_povm_l2_drift": povm_stats["global_povm_l2_drift"],
        "top_decile_carrier_tilt": povm_stats["top_decile_carrier_tilt"],
        "rare_decile_carrier_tilt": povm_stats["rare_decile_carrier_tilt"],
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
        extra_payload={"gram_spectrum": gram_stats, "povm_deciles": povm_stats["decile_breakdown"]},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Inspect calibrated Kronecker vocabulary spectrum and POVM drift.")
    parser.add_argument("--arch", type=str, default="multihead_mixed_state")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--results-json", type=str, default=None)
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
    payload = run_vocab_spectrum_and_povm_probe(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} sharp_eff_rank={payload['metrics']['sharp_eff_rank']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
