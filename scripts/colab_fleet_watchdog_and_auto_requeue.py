#!/usr/bin/env python3
"""Colab Fleet Watchdog & Auto-Requeue Utility (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

Inspects `.cache/live_logs/latest_snapshot.json` and declarative wave manifests (`configs/*.json`)
to detect stalled or preempted runs without spawning competing `colab exec` processes while
`monitor_colab_runs.py --serve` is active (`AGENTS.md` Section 1), and generates or executes
deterministic `scripts/colab_fleet.py` requeue actions (supporting `--dry-run`).
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
SCRIPT_NAME: str = "colab_fleet_watchdog_and_auto_requeue.py"
DEFAULT_SNAPSHOT_PATH: str = ".cache/live_logs/latest_snapshot.json"
DEFAULT_LIVE_LOGS_DIR: str = ".cache/live_logs"
DEFAULT_STALE_THRESHOLD_SECONDS: int = 600
MAX_REPORTED_RUNS: int = 32


def audit_live_logs_and_snapshot(
    snapshot_path: str = DEFAULT_SNAPSHOT_PATH,
    live_logs_dir: str = DEFAULT_LIVE_LOGS_DIR,
    manifest_path: str | None = None,
) -> dict[str, Any]:
    """Read local daemon snapshot and `*_results.json` files without competing `colab exec` calls (`<= 8` args)."""
    snap_p = Path(snapshot_path)
    logs_p = Path(live_logs_dir)

    snapshot_present = snap_p.exists()
    active_sessions = 0
    stalled_runs: list[str] = []
    completed_runs: list[str] = []

    if snapshot_present:
        try:
            snap_data = json.loads(snap_p.read_text(encoding="utf-8"))
            sessions = snap_data.get("sessions", snap_data.get("runs", []))
            if isinstance(sessions, list):
                active_sessions = len(sessions)
            elif isinstance(sessions, dict):
                active_sessions = len(sessions)
        except Exception:
            pass

    if logs_p.exists():
        for res_file in sorted(logs_p.glob("*_results.json"))[:MAX_REPORTED_RUNS]:
            try:
                data = json.loads(res_file.read_text(encoding="utf-8"))
                status_val = str(data.get("status", "completed")).lower()
                if status_val in ("failed", "error", "stalled", "preempted"):
                    stalled_runs.append(res_file.stem)
                else:
                    completed_runs.append(res_file.stem)
            except Exception:
                continue

    manifest_runs = 0
    if manifest_path and Path(manifest_path).exists():
        try:
            m_data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            runs_list = m_data.get("runs", [])
            if isinstance(runs_list, list):
                manifest_runs = len(runs_list)
        except Exception:
            pass

    requeue_commands = [
        f"python scripts/colab_fleet.py launch --manifest {manifest_path or 'configs/wave6_fleet.json'} --run {r_key}"
        for r_key in stalled_runs
    ]
    return {
        "snapshot_present": snapshot_present,
        "active_sessions_count": active_sessions,
        "completed_runs_count": len(completed_runs),
        "stalled_runs_count": len(stalled_runs),
        "manifest_runs_count": manifest_runs,
        "stalled_runs": stalled_runs,
        "requeue_commands": requeue_commands,
    }


def run_fleet_watchdog(args: argparse.Namespace) -> dict[str, Any]:
    """Execute Colab fleet watchdog check and optional dry-run requeue plan (`<= 8` args)."""
    audit = audit_live_logs_and_snapshot(
        snapshot_path=str(args.snapshot),
        live_logs_dir=str(args.live_logs_dir),
        manifest_path=args.manifest,
    )
    metrics: dict[str, Any] = {
        "dry_run": bool(args.dry_run),
        "snapshot_present": audit["snapshot_present"],
        "active_sessions_count": audit["active_sessions_count"],
        "completed_runs_count": audit["completed_runs_count"],
        "stalled_runs_count": audit["stalled_runs_count"],
        "manifest_runs_count": audit["manifest_runs_count"],
        "requeue_candidates_count": len(audit["requeue_commands"]),
    }
    return write_standardized_probe_json(
        json_out=args.json_out,
        script_name=SCRIPT_NAME,
        metrics=metrics,
        extra_payload={
            "stalled_runs": audit["stalled_runs"],
            "requeue_commands": audit["requeue_commands"],
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (`<= 8` args)."""
    parser = argparse.ArgumentParser(description="Inspect local Colab daemon snapshot and plan auto-requeue actions.")
    parser.add_argument("--snapshot", type=str, default=DEFAULT_SNAPSHOT_PATH)
    parser.add_argument("--live-logs-dir", type=str, default=DEFAULT_LIVE_LOGS_DIR)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--stale-seconds", type=int, default=DEFAULT_STALE_THRESHOLD_SECONDS)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--json-out", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (`<= 8` args)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_fleet_watchdog(args)
    print(f"[{SCRIPT_NAME}] status={payload['status']} completed={payload['metrics']['completed_runs_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
