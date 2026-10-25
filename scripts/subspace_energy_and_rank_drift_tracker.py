#!/usr/bin/env python3
"""Subspace Energy & Participation-Ratio Drift Tracker (`Wave 34` / `EXPERIMENT_PLAN_049.md`).

Promoted from `scripts/TOOLING_BACKLOG.md` (`Wave 33` Nice-to-Have #1 -> `Wave 34` Build-Before-Launch #1):
1.When to Use It:
   - Use this CLI script to audit per-subspace energy conservation (`||psi^{(m)}||_2^2 = 1/M = 0.25`),
     per-subspace sharp detector participation ratios (`r_sharp_coarse_256`, `r_sharp_mid_2048`,
     `r_sharp_sub_8192`, `r_sharp_fine_65536`), global `sharp_eff_rank` (`centered_r_sharp`),
     effective Parseval temperature `mean_beta`, and progress-ramped `coincidence_gamma_eff` drift
     across evaluation milestones in `.cache/hypothesis_experiments_048.json`,
     `.cache/hypothesis_experiments_049.json`, or `.cache/live_logs/*_results.json`.
2. Why It Was Built & Mathematical Provenance (`EXPERIMENT_PLAN_046.md` – `EXPERIMENT_PLAN_049.md`):
   - In `Wave 33` (`EXPERIMENT_PLAN_048.md`), non-symplectic coordinate folding dropped step-0
     `sharp_eff_rank` from `13.75` (`Wave 28`) to `9.49`, and `r_sharp_coarse_256` drifted from
     `9.76` at step 0 down to `3.42` by step 1,800 while `r_sharp_mid_2048` expanded to `28.91`.
   - Tracking per-subspace energy fractions and participation ratios across steps provides an
     automated diagnostic gate for Product-of-Spheres `(1/sqrt(M) S^{d/M - 1})^M` specialization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import json
from pathlib import Path
from typing import Any

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Documented in EXPERIMENT_PLAN_049.md (Wave 34)
# ==============================================================================
DEFAULT_JSON_IN: str = ".cache/hypothesis_experiments_048.json"
DEFAULT_JSON_OUT: str = ".cache/subspace_energy_and_rank_drift_report.json"
DEFAULT_NUM_SUBSPACES: int = 4
ENERGY_CONSERVATION_ATOL: float = 1e-3
SUBSPACE_LEVEL_NAMES: tuple[str, ...] = (
    "coarse_256",
    "mid_2048",
    "sub_8192",
    "fine_65536",
)


@dataclass(frozen=True)
class DriftTrackerConfig:
    """Configuration for `subspace_energy_and_rank_drift_tracker.py` (`<= 8` arguments)."""

    json_in: str = DEFAULT_JSON_IN
    json_out: str = DEFAULT_JSON_OUT
    num_subspaces: int = DEFAULT_NUM_SUBSPACES
    energy_atol: float = ENERGY_CONSERVATION_ATOL


def resolve_input_paths(json_in_spec: str) -> list[Path]:
    """Resolve comma-separated file paths or glob patterns into sorted existing `Path` objects."""
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw_part in json_in_spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        matched = sorted(glob.glob(part))
        candidates = [Path(m).resolve() for m in matched] if matched else [Path(part).resolve()]
        for cand in candidates:
            if cand.is_file() and cand not in seen:
                seen.add(cand)
                resolved.append(cand)
    return resolved


def extract_milestone_list(run_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize either a `milestones` dict (`step_0..step_N`) or an `evals` list into sorted step records."""
    raw_milestones = run_dict.get("milestones")
    records: list[dict[str, Any]] = []
    if isinstance(raw_milestones, dict):
        for key, val in raw_milestones.items():
            if isinstance(val, dict):
                rec = dict(val)
                if "step" not in rec and key.startswith("step_"):
                    try:
                        rec["step"] = int(key.split("_", 1)[1])
                    except ValueError:
                        rec["step"] = 0
                records.append(rec)
    elif isinstance(raw_milestones, list):
        records = [dict(v) for v in raw_milestones if isinstance(v, dict)]
    elif isinstance(run_dict.get("evals"), list):
        records = [dict(v) for v in run_dict["evals"] if isinstance(v, dict)]

    if not records and ("best_val_ce" in run_dict or "final_val_ce" in run_dict):
        fallback: dict[str, Any] = {
            "step": int(run_dict.get("steps", run_dict.get("current_step", 0))),
            "tokens_M": float(run_dict.get("tokens_M", run_dict.get("live_tokens_M", 0.0))),
            "val_ce": float(run_dict.get("final_val_ce", run_dict.get("best_val_ce", 0.0))),
            "top1_acc": float(run_dict.get("final_top1_acc", 0.0)),
            "sharp_eff_rank": float(run_dict.get("final_sharp_eff_rank", run_dict.get("sharp_eff_rank", 0.0))),
            "centered_r_sharp": float(run_dict.get("final_centered_r_sharp", 0.0)),
            "kraus_stiefel_err": float(run_dict.get("final_kraus_stiefel_err", 0.0)),
        }
        for name in SUBSPACE_LEVEL_NAMES:
            if f"energy_{name}" in run_dict:
                fallback[f"energy_{name}"] = float(run_dict[f"energy_{name}"])
            if f"r_sharp_{name}" in run_dict:
                fallback[f"r_sharp_{name}"] = float(run_dict[f"r_sharp_{name}"])
        records.append(fallback)

    records.sort(key=lambda r: int(r.get("step", 0)))
    return records


def collect_run_dicts(node: Any, source_file: str, found: list[tuple[str, dict[str, Any]]]) -> None:
    """Recursively traverse a JSON payload to locate run dictionaries containing `run_key`."""
    if isinstance(node, dict):
        if "run_key" in node and isinstance(node["run_key"], str):
            found.append((source_file, node))
        for val in node.values():
            if isinstance(val, (dict, list)):
                collect_run_dicts(val, source_file, found)
    elif isinstance(node, list):
        for item in node:
            collect_run_dicts(item, source_file, found)


def summarize_single_run(
    run_dict: dict[str, Any],
    source_file: str,
    cfg: DriftTrackerConfig,
) -> dict[str, Any]:
    """Compute per-subspace energy conservation and participation-ratio drift across a run's milestones."""
    run_key = str(run_dict.get("run_key", "unknown_run"))
    num_sub = int(run_dict.get("num_subspaces", cfg.num_subspaces))
    target_share = 1.0 / float(max(1, num_sub))
    milestones = extract_milestone_list(run_dict)

    step_rows: list[dict[str, Any]] = []
    max_energy_dev = 0.0
    has_subspace_telemetry = False

    for rec in milestones:
        step = int(rec.get("step", 0))
        tokens_m = float(rec.get("tokens_M", 0.0))
        val_ce = float(rec.get("val_ce", 0.0))
        top1_acc = float(rec.get("top1_acc", 0.0))
        sharp_rank = float(rec.get("sharp_eff_rank", 0.0))
        centered_r_sharp = float(rec.get("centered_r_sharp", sharp_rank))
        mean_beta = float(rec.get("mean_beta", rec.get("povm_beta", 0.0)))
        gamma_eff = float(rec.get("coincidence_gamma_eff", 0.0))
        var_pen_eff = float(rec.get("subspace_var_penalty_eff", 0.0))
        blend_eff = float(rec.get("subspace_balance_blend_eff", 0.0))

        energies: dict[str, float] = {}
        r_sharps: dict[str, float] = {}
        for name in SUBSPACE_LEVEL_NAMES:
            e_key = f"energy_{name}"
            r_key = f"r_sharp_{name}"
            if e_key in rec and rec[e_key] is not None:
                e_val = float(rec[e_key])
                energies[name] = round(e_val, 6)
                if num_sub > 1:
                    max_energy_dev = max(max_energy_dev, abs(e_val - target_share))
                has_subspace_telemetry = True
            if r_key in rec and rec[r_key] is not None:
                r_sharps[name] = round(float(rec[r_key]), 4)
                has_subspace_telemetry = True

        step_rows.append(
            {
                "step": step,
                "tokens_M": round(tokens_m, 4),
                "val_ce": round(val_ce, 4),
                "top1_acc": round(top1_acc, 4),
                "sharp_eff_rank": round(sharp_rank, 4),
                "centered_r_sharp": round(centered_r_sharp, 4),
                "mean_beta": round(mean_beta, 4),
                "coincidence_gamma_eff": round(gamma_eff, 6),
                "subspace_var_penalty_eff": round(var_pen_eff, 6),
                "subspace_balance_blend_eff": round(blend_eff, 6),
                "subspace_energy": energies,
                "subspace_r_sharp": r_sharps,
            }
        )

    first = step_rows[0] if step_rows else {}
    last = step_rows[-1] if step_rows else {}

    subspace_drift: dict[str, dict[str, float]] = {}
    for name in SUBSPACE_LEVEL_NAMES:
        vals = [row["subspace_r_sharp"][name] for row in step_rows if name in row.get("subspace_r_sharp", {})]
        e_vals = [row["subspace_energy"][name] for row in step_rows if name in row.get("subspace_energy", {})]
        if vals:
            subspace_drift[name] = {
                "step0_r_sharp": round(vals[0], 4),
                "final_r_sharp": round(vals[-1], 4),
                "min_r_sharp": round(min(vals), 4),
                "max_r_sharp": round(max(vals), 4),
                "drift_r_sharp": round(vals[-1] - vals[0], 4),
                "mean_energy_fraction": round(sum(e_vals) / len(e_vals), 6) if e_vals else 0.0,
            }

    step0_global_r = float(first.get("sharp_eff_rank", 0.0))
    final_global_r = float(last.get("sharp_eff_rank", 0.0))
    step0_centered_r = float(first.get("centered_r_sharp", 0.0))
    final_centered_r = float(last.get("centered_r_sharp", 0.0))
    step0_beta = float(first.get("mean_beta", 0.0))
    final_beta = float(last.get("mean_beta", 0.0))
    step0_gamma = float(first.get("coincidence_gamma_eff", 0.0))
    final_gamma = float(last.get("coincidence_gamma_eff", 0.0))
    step0_ce = float(first.get("val_ce", 0.0))
    final_ce = float(last.get("val_ce", float(run_dict.get("final_val_ce", 0.0))))
    best_ce = min([ float(r["val_ce"]) for r in step_rows ] or [final_ce])

    return {
        "run_key": run_key,
        "source_file": source_file,
        "num_subspaces": num_sub,
        "num_milestones": len(step_rows),
        "has_subspace_telemetry": has_subspace_telemetry,
        "energy_conservation_passed": bool(max_energy_dev <= cfg.energy_atol),
        "max_energy_dev_from_quarter": round(max_energy_dev, 6),
        "step0_val_ce": round(step0_ce, 4),
        "best_val_ce": round(best_ce, 4),
        "final_val_ce": round(final_ce, 4),
        "val_ce_improvement": round(step0_ce - final_ce, 4),
        "step0_sharp_eff_rank": round(step0_global_r, 4),
        "final_sharp_eff_rank": round(final_global_r, 4),
        "global_sharp_eff_rank_drift": round(final_global_r - step0_global_r, 4),
        "step0_centered_r_sharp": round(step0_centered_r, 4),
        "final_centered_r_sharp": round(final_centered_r, 4),
        "centered_r_sharp_drift": round(final_centered_r - step0_centered_r, 4),
        "step0_mean_beta": round(step0_beta, 4),
        "final_mean_beta": round(final_beta, 4),
        "mean_beta_drift": round(final_beta - step0_beta, 4),
        "step0_coincidence_gamma_eff": round(step0_gamma, 6),
        "final_coincidence_gamma_eff": round(final_gamma, 6),
        "coincidence_gamma_drift": round(final_gamma - step0_gamma, 6),
        "subspace_summary": subspace_drift,
        "milestone_trajectory": step_rows,
    }


def run_drift_tracker(cfg: DriftTrackerConfig) -> dict[str, Any]:
    """Parse all input JSON files, audit subspace energy & rank drift across runs, and save report."""
    paths = resolve_input_paths(cfg.json_in)
    if not paths:
        raise FileNotFoundError(f"No existing JSON files matched --json-in={cfg.json_in!r}")

    found_runs: list[tuple[str, dict[str, Any]]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        collect_run_dicts(payload, str(path.name), found_runs)

    dedup_runs: dict[str, dict[str, Any]] = {}
    for src_name, run_dict in found_runs:
        summary = summarize_single_run(run_dict, src_name, cfg)
        r_key = summary["run_key"]
        if r_key not in dedup_runs or summary["num_milestones"] >= dedup_runs[r_key]["num_milestones"]:
            dedup_runs[r_key] = summary

    runs_list = list(dedup_runs.values())
    all_energy_passed = all(r["energy_conservation_passed"] for r in runs_list)

    report: dict[str, Any] = {
        "script": "scripts/subspace_energy_and_rank_drift_tracker.py",
        "wave_provenance": "Wave 34 / EXPERIMENT_PLAN_049.md (Promoted from Wave 33 Nice-to-Have #1)",
        "input_files": [str(p) for p in paths],
        "num_runs_analyzed": len(runs_list),
        "all_energy_conservation_passed": bool(all_energy_passed),
        "runs": runs_list,
    }

    out_path = Path(cfg.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _format_subspace_drift_cell(sub: dict[str, dict[str, float]], name: str) -> str:
    """Format a single subspace `step0 -> final (drift)` Markdown table cell."""
    if name not in sub:
        return "n/a"
    s = sub[name]
    return f"`{s['step0_r_sharp']:.2f} -> {s['final_r_sharp']:.2f}` (`{s['drift_r_sharp']:+.2f}`)"


def format_markdown_summary(report: dict[str, Any]) -> str:
    """Render a clean Markdown summary table of global and per-subspace rank/energy drift."""
    lines: list[str] = [
        "### Subspace Energy & Participation-Ratio (`R_sharp`) Drift Summary (`Wave 34` / `Plan 049`)",
        "",
        "| Run Key | M | Steps | Val CE (`0 -> Final`) | Global `R_sharp` (`0 -> Final`) | Centered `R_sharp` (`0 -> Final`) | Coarse256 `R_s` (`0 -> Final`) | Mid2048 `R_s` (`0 -> Final`) | Sub8192 `R_s` (`0 -> Final`) | Fine65536 `R_s` (`0 -> Final`) | `beta_eff` (`0 -> Final`) | `gamma_c` (`0 -> Final`) | Max `|E_m - 0.25|` |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for run in report.get("runs", []):
        traj = run.get("milestone_trajectory", [])
        last_step = traj[-1]["step"] if traj else 0
        sub = run.get("subspace_summary", {})
        lines.append(
            f"| `{run['run_key']}` | `{run['num_subspaces']}` | `{last_step}` | "
            f"`{run['step0_val_ce']:.4f} -> {run['final_val_ce']:.4f}` | "
            f"`{run['step0_sharp_eff_rank']:.2f} -> {run['final_sharp_eff_rank']:.2f}` | "
            f"`{run['step0_centered_r_sharp']:.2f} -> {run['final_centered_r_sharp']:.2f}` | "
            f"{_format_subspace_drift_cell(sub, 'coarse_256')} | {_format_subspace_drift_cell(sub, 'mid_2048')} | "
            f"{_format_subspace_drift_cell(sub, 'sub_8192')} | {_format_subspace_drift_cell(sub, 'fine_65536')} | "
            f"`{run['step0_mean_beta']:.2f} -> {run['final_mean_beta']:.2f}` | "
            f"`{run['step0_coincidence_gamma_eff']:.3f} -> {run['final_coincidence_gamma_eff']:.3f}` | "
            f"`{run['max_energy_dev_from_quarter']:.6f}` |"
        )
    return "\n".join(lines)


def parse_args() -> DriftTrackerConfig:
    """Parse CLI arguments into `DriftTrackerConfig`."""
    parser = argparse.ArgumentParser(
        description="Audit per-subspace energy fractions and participation ratio R_sharp drift across milestones."
    )
    parser.add_argument(
        "--json-in",
        type=str,
        default=DEFAULT_JSON_IN,
        help="Comma-separated JSON file paths or glob patterns (default: .cache/hypothesis_experiments_048.json).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=DEFAULT_JSON_OUT,
        help="Output JSON report path (default: .cache/subspace_energy_and_rank_drift_report.json).",
    )
    parser.add_argument(
        "--subspaces",
        type=int,
        default=DEFAULT_NUM_SUBSPACES,
        help="Default number of Product-of-Spheres subspaces M (default: 4).",
    )
    parser.add_argument(
        "--energy-atol",
        type=float,
        default=ENERGY_CONSERVATION_ATOL,
        help="Maximum allowed deviation from 1/M subspace energy share (default: 1e-3).",
    )
    args = parser.parse_args()
    return DriftTrackerConfig(
        json_in=args.json_in,
        json_out=args.json_out,
        num_subspaces=args.subspaces,
        energy_atol=args.energy_atol,
    )


def main() -> None:
    """Entrypoint for `subspace_energy_and_rank_drift_tracker.py`."""
    cfg = parse_args()
    report = run_drift_tracker(cfg)
    print(format_markdown_summary(report))
    print(f"\nSaved subspace energy & rank drift report ({report['num_runs_analyzed']} runs) -> {cfg.json_out}")


if __name__ == "__main__":
    main()
