#!/usr/bin/env python3
"""POVM Scale Drift & Rarity Decile Analyzer (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Tracks how trainable POVM operator scales `gamma_v = w_{povm,v}^{1/p}` evolve across 10 Zipf
token-frequency deciles (`Decile 0`: Top 10% most frequent active IDs to `Decile 9`: Bottom 10%
rarest active IDs), reporting `mean_povm_scale_exp`, `std_povm_scale_exp`, and `decile_mass_ratio`.
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
    compute_fineweb_10_frequency_deciles,
    load_or_build_calibrated_kronecker_model,
    write_standardized_probe_json,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
SCRIPT_NAME: str = "povm_scale_drift_and_rarity_decile_analyzer.py"
SCALE_CLAMP_MIN: float = 1e-12


@torch.no_grad()
def analyze_povm_scales_by_decile(
    model: torch.nn.Module,
    num_deciles: int = DEFAULT_NUM_FREQUENCY_DECILES,
) -> dict[str, Any]:
    """Compute 10-decile breakdown of `mean_povm_scale_exp`, `std_povm_scale_exp`, and `decile_mass_ratio` (`<= 8` args)."""
    v_eff = int(model.cfg.valid_vocab_size)
    dev = model.embed_re.device
    uni_p = model.unigram_probs[:v_eff].detach().float() if hasattr(model, "unigram_probs") else torch.ones((v_eff,), device=dev) / float(v_eff)
    deciles = compute_fineweb_10_frequency_deciles(uni_p, active_vocab_size=v_eff, num_deciles=num_deciles)

    w_povm = model.compute_povm_weights()[:v_eff].detach().float().clamp_min(SCALE_CLAMP_MIN)
    inv_p = 1.0 / max(1.0, float(model.cfg.boson_p))
    gamma_v = torch.pow(w_povm, inv_p)
    total_w = w_povm.sum().clamp_min(SCALE_CLAMP_MIN)

    decile_rows: list[dict[str, float]] = []
    for d_idx in range(int(num_deciles)):
        mask = deciles == d_idx
        if int(mask.sum().item()) == 0:
            continue
        g_sub = gamma_v[mask]
        w_sub = w_povm[mask]
        decile_rows.append({
            "decile": float(d_idx),
            "mean_povm_scale_exp": round(float(g_sub.mean().item()), 6),
            "std_povm_scale_exp": round(float(g_sub.std().item()), 6),
            "decile_mass_ratio": round(float((w_sub.sum() / total_w).item()), 6),
        })

    return {
        "mean_povm_scale_exp": round(float(gamma_v.mean().item()), 6),
        "std_povm_scale_exp": round(float(gamma_v.std().item()), 6),
        "decile_0_mass_ratio": decile_rows[0]["decile_mass_ratio"] if decile_rows else 0.0,
        "decile_9_mass_ratio": decile_rows[-1]["decile_mass_ratio"] if decile_rows else 0.0,
        "deciles": decile_rows,
    }


@torch.no_grad()
def run_povm_decile_analyzer(args: argparse.Namespace) -> dict[str, Any]:
    """Run POVM scale drift and rarity decile analysis (`<= 8` args)."""
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
    decile_stats = analyze_povm_scales_by_decile(model, num_deciles=DEFAULT_NUM_FREQUENCY_DECILES)

    metrics: dict[str, Any] = {
        "arch": metadata["arch"],
        "checkpoint_mode": metadata["checkpoint_mode"],
        "mean_povm_scale_exp": decile_stats["mean_povm_scale_exp"],
        "std_povm_scale_exp": decile_stats["std_povm_scale_exp"],
        "decile_0_mass_ratio": decile_stats["decile_0_mass_ratio"],
        "decile_9_mass_ratio": decile_stats["decile_9_mass_ratio"],
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
        extra_payload={"deciles": decile_stats["deciles"]},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Analyze POVM scale drift across 10 Zipf frequency deciles.")
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
    payload = run_povm_decile_analyzer(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} mean_scale={payload['metrics']['mean_povm_scale_exp']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
