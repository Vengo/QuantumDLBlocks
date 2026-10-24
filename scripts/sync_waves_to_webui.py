#!/usr/bin/env python3
"""Normalize and sync all Waves 18–29, Vision Ablations, and Plans 013–045 into SQLite & WebUI.

Implements Experiment Plan 044 (`Build-Before-Launch` Script):
- Calls `ingest_all_waves_and_verifications()` to repair `ModelCard v1.0` files in `.cache/model_cards/`,
  backfill missing `.log` files in `.cache/live_logs/`, and upsert all runs into `.cache/benchmarks.db`.
- Prints a Markdown summary table across `runs_catalog` (Full FineWeb), `fineweb_edu_catalog`
  (`Waves 18–29`), and `micro_runs_catalog` (`Plans 013–045` & Vision Ablations).
- Saves `.cache/webui_wave_sync_report.json` and optionally triggers `POST http://127.0.0.1:8765/api/reload`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import urllib.request
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qdlblocks.benchmarks.db import DEFAULT_DB_PATH, BenchmarkSQLiteDB
from qdlblocks.benchmarks.model_card import DEFAULT_LIVE_LOGS_DIR, DEFAULT_MODEL_CARDS_DIR
from qdlblocks.benchmarks.wave_ingestor import ingest_all_waves_and_verifications


@dataclass(frozen=True)
class SyncWavesConfig:
    """CLI configuration for `scripts/sync_waves_to_webui.py`."""

    db_path: str = str(DEFAULT_DB_PATH)
    cache_dir: str = str(REPO_ROOT / ".cache")
    live_logs_dir: str = str(DEFAULT_LIVE_LOGS_DIR)
    model_cards_dir: str = str(DEFAULT_MODEL_CARDS_DIR)
    report_json: str = str(REPO_ROOT / ".cache" / "webui_wave_sync_report.json")
    reload_url: str = "http://127.0.0.1:8765/api/reload"
    trigger_reload: bool = False


def trigger_webui_reload(reload_url: str) -> dict[str, Any]:
    """Send a `POST /api/reload` request to the running WebUI daemon, ignoring connection errors."""
    req = urllib.request.Request(
        reload_url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return {"triggered": True, "status_code": int(resp.status), "response": body[:200]}
    except Exception as exc:
        return {"triggered": False, "reason": str(exc)}


def format_catalog_summary_markdown(snapshot: dict[str, Any], sync_stats: dict[str, Any]) -> str:
    """Render a concise Markdown summary table of all three WebUI catalogs."""
    main_cat = snapshot.get("runs_catalog", [])
    edu_cat = snapshot.get("fineweb_edu_catalog", [])
    micro_cat = snapshot.get("micro_runs_catalog", [])

    def _best_row_str(rows: list[dict[str, Any]]) -> str:
        valid = [r for r in rows if r.get("best_val_ce") is not None]
        if not valid:
            return "—"
        best = min(valid, key=lambda r: float(r["best_val_ce"]))
        return f"`{best['key']}` (`{float(best['best_val_ce']):.4f}` CE)"

    pareto_edu = [r["key"] for r in edu_cat if r.get("pareto_frontier")]
    lines = [
        "### WebUI Wave Sync Summary (`scripts/sync_waves_to_webui.py`)",
        "",
        "| Catalog Tier | Total Runs | Completed | Pareto Frontier | Best Run (`Val CE`) |",
        "| :--- | :---: | :---: | :---: | :--- |",
        (
            f"| `runs_catalog` (Full FineWeb Speedrun) | `{len(main_cat)}` | "
            f"`{sum(1 for r in main_cat if r.get('completed'))}` | "
            f"`{sum(1 for r in main_cat if r.get('pareto_frontier'))}` | {_best_row_str(main_cat)} |"
        ),
        (
            f"| `fineweb_edu_catalog` (`Waves 18–29`) | `{len(edu_cat)}` | "
            f"`{sum(1 for r in edu_cat if r.get('completed'))}` | "
            f"`{len(pareto_edu)}` | {_best_row_str(edu_cat)} |"
        ),
        (
            f"| `micro_runs_catalog` (`Plans 013–045` & Vision) | `{len(micro_cat)}` | "
            f"`{sum(1 for r in micro_cat if r.get('completed'))}` | "
            f"`{sum(1 for r in micro_cat if r.get('pareto_frontier'))}` | {_best_row_str(micro_cat)} |"
        ),
        "",
        f"- **Ingested FineWeb-Edu (`W18–W29`)**: `{sync_stats.get('ingested_fineweb_edu_count', 0)}`",
        f"- **Ingested Vision Ablations (`CIFAR-100` / `1,000-Class` / `ImageNet-1K`)**: `{sync_stats.get('ingested_vision_ablation_count', 0)}`",
        f"- **FineWeb-Edu Pareto Frontier Keys**: `{', '.join(pareto_edu[:10])}`",
    ]
    return "\n".join(lines)


def run_wave_sync(cfg: SyncWavesConfig) -> dict[str, Any]:
    """Execute full Wave 18–29 + Micro-Verification ingestion and persist report JSON."""
    db = BenchmarkSQLiteDB(Path(cfg.db_path))
    try:
        sync_stats = ingest_all_waves_and_verifications(
            db=db,
            cache_dir=Path(cfg.cache_dir),
            live_logs_dir=Path(cfg.live_logs_dir),
            model_cards_dir=Path(cfg.model_cards_dir),
        )
        snapshot = db.load_snapshot_payload()
    finally:
        db.close()

    reload_res = trigger_webui_reload(cfg.reload_url) if cfg.trigger_reload else {"triggered": False}
    md_summary = format_catalog_summary_markdown(snapshot, sync_stats)
    print(md_summary)

    edu_cat = snapshot.get("fineweb_edu_catalog", [])
    report = {
        "config": asdict(cfg),
        "sync_stats": sync_stats,
        "reload_result": reload_res,
        "fineweb_edu_top10": [
            {
                "key": r["key"],
                "wave_id": r.get("wave_id"),
                "params_M": r.get("params_M"),
                "target_tokens_M": r.get("target_tokens_M"),
                "best_val_ce": r.get("best_val_ce"),
                "best_top1": r.get("best_top1"),
                "avg_tok_per_s": r.get("avg_tok_per_s"),
                "pareto_frontier": r.get("pareto_frontier"),
            }
            for r in edu_cat[:10]
        ],
    }
    out_p = Path(cfg.report_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> SyncWavesConfig:
    """Parse CLI arguments into `SyncWavesConfig`."""
    p = argparse.ArgumentParser(description="Sync Waves 18–29 and micro-verifications into SQLite & WebUI.")
    p.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH))
    p.add_argument("--cache-dir", type=str, default=str(REPO_ROOT / ".cache"))
    p.add_argument("--live-logs-dir", type=str, default=str(DEFAULT_LIVE_LOGS_DIR))
    p.add_argument("--model-cards-dir", type=str, default=str(DEFAULT_MODEL_CARDS_DIR))
    p.add_argument("--report-json", type=str, default=str(REPO_ROOT / ".cache" / "webui_wave_sync_report.json"))
    p.add_argument("--reload-url", type=str, default="http://127.0.0.1:8765/api/reload")
    p.add_argument("--trigger-reload", action="store_true", help="Send POST /api/reload to running WebUI server.")
    ns = p.parse_args(argv)
    return SyncWavesConfig(
        db_path=ns.db_path,
        cache_dir=ns.cache_dir,
        live_logs_dir=ns.live_logs_dir,
        model_cards_dir=ns.model_cards_dir,
        report_json=ns.report_json,
        reload_url=ns.reload_url,
        trigger_reload=bool(ns.trigger_reload),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    run_wave_sync(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
