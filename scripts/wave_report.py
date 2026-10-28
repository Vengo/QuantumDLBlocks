#!/usr/bin/env python3
"""Wave Pareto Leaderboard & Telemetry Report Generator (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Aggregates `.cache/live_logs/*_results.json` and `.cache/model_cards/*.json` across requested
waves (`--wave W42,W43,W44`) into a unified Markdown comparison table and standardized JSON report.
"""

from __future__ import annotations

import argparse
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
SCRIPT_NAME: str = "wave_report.py"
DEFAULT_WAVES_CSV: str = "W42,W43,W44"
DEFAULT_LIVE_LOGS_DIR: str = ".cache/live_logs"
DEFAULT_MODEL_CARDS_DIR: str = ".cache/model_cards"
MAX_SCAN_FILES: int = 128


def collect_wave_leaderboard_entries(
    wave_filters: list[str],
    live_logs_dir: str = DEFAULT_LIVE_LOGS_DIR,
    model_cards_dir: str = DEFAULT_MODEL_CARDS_DIR,
) -> list[dict[str, Any]]:
    """Scan `live_logs` and `model_cards` for runs matching `wave_filters` (`<= 8` args)."""
    norm_filters = [w.strip().lower() for w in wave_filters if w.strip()]
    entries: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    candidate_dirs = [Path(model_cards_dir), Path(live_logs_dir)]
    for c_dir in candidate_dirs:
        if not c_dir.exists():
            continue
        for jpath in sorted(c_dir.glob("*.json"))[:MAX_SCAN_FILES]:
            if jpath.name == "latest_snapshot.json":
                continue
            try:
                raw = json.loads(jpath.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue
            run_key = str(raw.get("run_key", raw.get("key", jpath.stem.replace("_results", ""))))
            if run_key in seen_keys:
                continue
            wave_id = str(raw.get("wave_id", raw.get("wave", ""))).lower()
            if norm_filters and not any(wf in run_key.lower() or wf in wave_id for wf in norm_filters):
                continue
            val_ce = raw.get("val_ce", raw.get("best_val_ce", raw.get("final_val_ce")))
            if isinstance(val_ce, (int, float)):
                seen_keys.add(run_key)
                entries.append({
                    "run_key": run_key,
                    "wave_id": str(raw.get("wave_id", "W44")),
                    "val_ce": round(float(val_ce), 4),
                    "top1_acc": round(float(raw.get("top1_acc", 0.0)), 4),
                    "tok_per_s": round(float(raw.get("tok_per_s", raw.get("throughput_tok_s", 0.0))), 1),
                })

    entries.sort(key=lambda r: r["val_ce"])
    return entries


def format_markdown_leaderboard(entries: list[dict[str, Any]]) -> str:
    """Format leaderboard entries into a GitHub-Flavored Markdown table (`<= 8` args)."""
    lines = [
        "| Rank | Run Key | Wave | Val CE | Top-1 Acc | Tok/s |",
        "| :---: | :--- | :---: | :---: | :---: | :---: |",
    ]
    for idx, row in enumerate(entries[:20], start=1):
        lines.append(
            f"| {idx} | `{row['run_key']}` | `{row['wave_id']}` | `{row['val_ce']:.4f}` | "
            f"`{row['top1_acc']:.4f}` | `{row['tok_per_s']:.1f}` |"
        )
    return "\n".join(lines)


def run_wave_report(args: argparse.Namespace) -> dict[str, Any]:
    """Generate unified wave Pareto leaderboard and JSON report (`<= 8` args)."""
    wave_list = [w.strip() for w in str(args.wave).split(",") if w.strip()]
    entries = collect_wave_leaderboard_entries(
        wave_filters=wave_list,
        live_logs_dir=str(args.live_logs_dir),
        model_cards_dir=str(args.model_cards_dir),
    )
    best_ce = entries[0]["val_ce"] if entries else 0.0
    best_run = entries[0]["run_key"] if entries else "none"
    md_table = format_markdown_leaderboard(entries)

    metrics: dict[str, Any] = {
        "waves_queried": ",".join(wave_list),
        "matched_runs_count": len(entries),
        "best_run_key": best_run,
        "best_val_ce": best_ce,
        "update_summary_requested": bool(args.update_summary),
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
        extra_payload={"leaderboard": entries, "markdown_table": md_table},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Generate Markdown + JSON Pareto leaderboard across waves.")
    parser.add_argument("--wave", "--waves", dest="wave", type=str, default=DEFAULT_WAVES_CSV)
    parser.add_argument("--live-logs-dir", type=str, default=DEFAULT_LIVE_LOGS_DIR)
    parser.add_argument("--model-cards-dir", type=str, default=DEFAULT_MODEL_CARDS_DIR)
    parser.add_argument("--update-summary", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (`<= 8` args)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_wave_report(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} matched_runs={payload['metrics']['matched_runs_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
