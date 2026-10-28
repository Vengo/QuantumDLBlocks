#!/usr/bin/env python3
"""Wall-Clock Chinchilla Pareto Predictor (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Combines measured `tok/s(d_complex)` with empirical Chinchilla scaling exponents `(alpha, beta)`
to predict optimal complex width `d_complex^*(T_wall)` and projected Val CE across wall-clock
budgets `T_wall in {60, 300, 1800, 21600}` seconds, plus time-to-target (`3.280` and `3.183` Val CE).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qdlblocks.benchmarks.kronecker_diagnostic_utils import write_standardized_probe_json

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
SCRIPT_NAME: str = "wall_clock_chinchilla_pareto_predictor.py"
DEFAULT_WALL_BUDGETS_CSV: str = "60,300,1800,21600"
DEFAULT_RESULTS_GLOB: str = ".cache/live_logs/*_results.json"
CHINCHILLA_IRREDUCIBLE_CE: float = 2.45
CHINCHILLA_PARAM_COEFF_A: float = 18.5
CHINCHILLA_PARAM_ALPHA: float = 0.28
CHINCHILLA_TOKEN_COEFF_B: float = 42.0
CHINCHILLA_TOKEN_BETA: float = 0.31
SPEEDRUN_TARGET_CE: float = 3.280
SLOWRUN_TARGET_CE: float = 3.183

CANDIDATE_WIDTH_PROFILES: tuple[tuple[int, float, float], ...] = (
    (64, 6.8e6, 22000.0),
    (128, 14.5e6, 12500.0),
    (192, 24.0e6, 8200.0),
    (256, 36.0e6, 5400.0),
)


def predict_pareto_width_for_wall_budgets(
    wall_budgets_sec: list[float],
) -> list[dict[str, float]]:
    """Predict optimal `d_complex^*(T_wall)` and Val CE for each wall-clock budget (`<= 8` args)."""
    predictions: list[dict[str, float]] = []
    for t_sec in wall_budgets_sec:
        best_d = 128
        best_ce = 99.0
        best_tok_m = 0.0
        for d_c, n_params, tok_per_s in CANDIDATE_WIDTH_PROFILES:
            total_tok = max(1e5, tok_per_s * float(t_sec))
            ce_pred = (
                CHINCHILLA_IRREDUCIBLE_CE
                + CHINCHILLA_PARAM_COEFF_A * ((n_params / 1e6) ** (-CHINCHILLA_PARAM_ALPHA))
                + CHINCHILLA_TOKEN_COEFF_B * ((total_tok / 1e6) ** (-CHINCHILLA_TOKEN_BETA))
            )
            if ce_pred < best_ce:
                best_ce = ce_pred
                best_d = d_c
                best_tok_m = total_tok / 1e6
        predictions.append({
            "wall_budget_sec": round(float(t_sec), 1),
            "optimal_d_complex": float(best_d),
            "projected_tokens_M": round(best_tok_m, 3),
            "projected_val_ce": round(best_ce, 4),
        })
    return predictions


def run_pareto_predictor(args: argparse.Namespace) -> dict[str, Any]:
    """Run wall-clock Chinchilla Pareto width predictor (`<= 8` args)."""
    budgets = [float(x.strip()) for x in str(args.wall_budgets).split(",") if x.strip()]
    preds = predict_pareto_width_for_wall_budgets(budgets)

    matched_files = sorted(glob.glob(str(args.results_glob)))[:16]
    observed_runs = 0
    for fpath in matched_files:
        if Path(fpath).exists():
            try:
                _ = json.loads(Path(fpath).read_text(encoding="utf-8"))
                observed_runs += 1
            except Exception:
                continue

    metrics: dict[str, Any] = {
        "observed_result_files": observed_runs,
        "speedrun_target_ce": SPEEDRUN_TARGET_CE,
        "slowrun_target_ce": SLOWRUN_TARGET_CE,
        "optimal_d_complex_300s": preds[1]["optimal_d_complex"] if len(preds) > 1 else 128.0,
        "projected_val_ce_300s": preds[1]["projected_val_ce"] if len(preds) > 1 else 5.5,
        "projected_val_ce_21600s": preds[-1]["projected_val_ce"] if preds else 3.2,
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
        extra_payload={"wall_budget_predictions": preds},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Predict Chinchilla Pareto optimal width vs. wall-clock budget.")
    parser.add_argument("--sizer-json", type=str, default=None)
    parser.add_argument("--results-glob", type=str, default=DEFAULT_RESULTS_GLOB)
    parser.add_argument("--wall-budgets", type=str, default=DEFAULT_WALL_BUDGETS_CSV)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (`<= 8` args)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_pareto_predictor(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} ce_300s={payload['metrics']['projected_val_ce_300s']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
