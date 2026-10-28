#!/usr/bin/env python3
"""Multi-Lag Window (`tau in {0,1,2,3}`) & Coincidence Schedule Probe (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Profiles causal multi-lag context overlap across lags `tau in {0, 1, 2, 3}`, polynomial XOR-hashed
N-gram bucket entropy, and progress-ramped coincidence scheduling `coinc_gamma(tau_progress)`.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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
    load_kronecker_probe_batch,
    load_or_build_calibrated_kronecker_model,
    write_standardized_probe_json,
)
from qdlblocks.models.ngram_calibrated_kronecker_born_gpt import compute_trigram_xor_hash_buckets

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
SCRIPT_NAME: str = "multilag_window_and_coincidence_schedule_probe.py"
DEFAULT_TAU_PROGRESS_GRID: str = "0.0,0.25,0.50,0.75,1.0"
DEFAULT_MAX_LAG: int = 3
COINCIDENCE_START_FRAC: float = 0.60


def _parse_float_csv(raw: str) -> list[float]:
    """Parse comma-separated float list (`<= 8` args)."""
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


@torch.no_grad()
def compute_multilag_autocorrelation(
    phi_factors: torch.Tensor,
    seq_len: int,
    max_lag: int = DEFAULT_MAX_LAG,
) -> dict[str, float]:
    """Compute per-lag complex factor autocorrelation across `lag in 1..max_lag` (`<= 8` args)."""
    d_f = int(phi_factors.size(-1)) // 2
    n_tok, r_num, m_num, _ = phi_factors.shape
    t_eff = max(4, min(int(seq_len), n_tok))
    b_eff = max(1, n_tok // t_eff)
    phi_seq = phi_factors[: b_eff * t_eff].view(b_eff, t_eff, r_num, m_num, 2 * d_f).float()
    pr, pi = phi_seq[..., :d_f], phi_seq[..., d_f:]

    lag_overlaps: dict[str, float] = {}
    for lag in range(1, int(max_lag) + 1):
        if t_eff <= lag:
            lag_overlaps[f"lag_{lag}_overlap"] = 0.0
            continue
        ov_re = (pr[:, lag:] * pr[:, :-lag] + pi[:, lag:] * pi[:, :-lag]).sum(dim=-1)
        ov_im = (pr[:, lag:] * pi[:, :-lag] - pi[:, lag:] * pr[:, :-lag]).sum(dim=-1)
        mag = torch.sqrt(ov_re.square() + ov_im.square()).mean()
        lag_overlaps[f"lag_{lag}_overlap"] = round(float(mag.item()), 6)
    return lag_overlaps


@torch.no_grad()
def run_multilag_and_schedule_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Execute multi-lag window and coincidence schedule probe (`<= 8` args)."""
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
    t_len = min(32, int(args.seq_len))
    x_batch, y_batch = load_kronecker_probe_batch(
        val_shard_path=str(args.val_shard),
        batch_size=2,
        seq_len=t_len,
        device=str(args.device),
    )

    phi_factors = model.extract_readout_states(x_batch)[0].detach()
    lag_stats = compute_multilag_autocorrelation(phi_factors, seq_len=t_len, max_lag=DEFAULT_MAX_LAG)

    buckets = compute_trigram_xor_hash_buckets(x_batch[:, :-1], x_batch[:, 1:])
    unique_bucket_ratio = float(buckets.unique().numel()) / float(max(1, buckets.numel()))

    tau_grid = _parse_float_csv(args.tau_grid)
    orig_cfg = model.cfg
    base_gamma = float(orig_cfg.coinc_gamma)
    schedule_rows: list[dict[str, float]] = []
    for tau in tau_grid:
        gamma_eff = base_gamma * (COINCIDENCE_START_FRAC + (1.0 - COINCIDENCE_START_FRAC) * float(tau))
        model.cfg = replace(orig_cfg, coinc_gamma=gamma_eff)
        _, loss_t = model(x_batch, targets=y_batch, return_logits=False)
        schedule_rows.append({
            "tau_progress": round(float(tau), 3),
            "coinc_gamma_eff": round(gamma_eff, 5),
            "val_nll": round(float(loss_t.detach().item()) if loss_t is not None else 0.0, 5),
        })
    model.cfg = orig_cfg

    metrics: dict[str, Any] = {
        "arch": metadata["arch"],
        "checkpoint_mode": metadata["checkpoint_mode"],
        **lag_stats,
        "trigram_unique_bucket_ratio": round(unique_bucket_ratio, 4),
        "schedule_start_nll": schedule_rows[0]["val_nll"] if schedule_rows else 0.0,
        "schedule_end_nll": schedule_rows[-1]["val_nll"] if schedule_rows else 0.0,
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
        extra_payload={"coincidence_schedule": schedule_rows},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Profile multi-lag temporal window and coincidence schedule.")
    parser.add_argument("--arch", type=str, default="multihead_mixed_state")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--calibration-cache", type=str, default=DEFAULT_NGRAM_CALIBRATION_CACHE_PATH)
    parser.add_argument("--val-shard", type=str, default=DEFAULT_FINEWEB_VAL_SHARD_PATH)
    parser.add_argument("--tau-grid", type=str, default=DEFAULT_TAU_PROGRESS_GRID)
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
    payload = run_multilag_and_schedule_probe(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} lag1={payload['metrics']['lag_1_overlap']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
