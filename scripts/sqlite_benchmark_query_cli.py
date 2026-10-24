#!/usr/bin/env python3
"""Fast Terminal Query CLI over `.cache/benchmarks.db` (`scripts/sqlite_benchmark_query_cli.py`).

Implements Experiment Plan 044:
- Queries `.cache/benchmarks.db` filtered by `--tier` (`all`, `main_benchmark`, `fineweb_edu_local`,
  `micro_verification`), `--wave` (e.g., `W28`), `--family`, and `--pareto-only`.
- Sorts by `best_val_ce`, `wall_min`, `avg_tok_per_s`, `live_tokens_M`, or `wave_num`.
- Prints a clean Markdown table with `key`, `wave_id`, `run_tier`, `params_M`, `live_tokens_M`,
  `compute_tflops`, `best_val_ce`, `best_top1`, `avg_tok_per_s`, and `pareto_frontier`, and optionally
  writes `--json-out`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qdlblocks.benchmarks.db import DEFAULT_DB_PATH, BenchmarkSQLiteDB


@dataclass(frozen=True)
class QueryCLIConfig:
    """Configuration for `scripts/sqlite_benchmark_query_cli.py`."""

    db_path: str = str(DEFAULT_DB_PATH)
    tier: str = "all"
    wave: str = ""
    family: str = ""
    pareto_only: bool = False
    sort: str = "best_val_ce"
    limit: int = 25
    json_out: str = ""


def query_benchmark_rows(cfg: QueryCLIConfig) -> list[dict[str, Any]]:
    """Load and filter benchmark rows from `BenchmarkSQLiteDB`."""
    db = BenchmarkSQLiteDB(Path(cfg.db_path))
    try:
        snap = db.load_snapshot_payload()
    finally:
        db.close()

    tier_low = cfg.tier.strip().lower()
    if tier_low == "main_benchmark":
        pool = list(snap.get("runs_catalog", []))
    elif tier_low == "fineweb_edu_local":
        pool = list(snap.get("fineweb_edu_catalog", []))
    elif tier_low == "micro_verification":
        pool = list(snap.get("micro_runs_catalog", []))
    else:
        pool = (
            list(snap.get("runs_catalog", []))
            + list(snap.get("fineweb_edu_catalog", []))
            + list(snap.get("micro_runs_catalog", []))
        )

    wave_up = cfg.wave.strip().upper()
    fam_low = cfg.family.strip().lower()
    filtered: list[dict[str, Any]] = []
    for r in pool:
        if wave_up and str(r.get("wave_id") or "").upper() != wave_up:
            continue
        if fam_low and str(r.get("arch_family") or "").lower() != fam_low:
            continue
        if cfg.pareto_only and not bool(r.get("pareto_frontier")):
            continue
        params_m = float(r.get("params_M") or 0.0)
        tok_m = float(r.get("live_tokens_M") or r.get("target_tokens_M") or 0.0)
        tflops = float(
            r.get("compute_tflops")
            if r.get("compute_tflops") is not None
            else round(6.0 * params_m * tok_m, 2)
        )
        filtered.append(
            {
                "key": str(r.get("key") or r.get("run_key") or ""),
                "wave_id": str(r.get("wave_id") or ""),
                "wave_num": float(r.get("wave_num") or 0.0),
                "run_tier": str(r.get("run_tier") or ""),
                "arch_family": str(r.get("arch_family") or ""),
                "params_M": round(params_m, 2),
                "live_tokens_M": round(tok_m, 2),
                "compute_tflops": round(tflops, 2),
                "best_val_ce": r.get("best_val_ce"),
                "best_top1": r.get("best_top1"),
                "avg_tok_per_s": round(float(r.get("avg_tok_per_s") or 0.0), 1),
                "wall_min": round(float(r.get("wall_min") or 0.0), 3),
                "pareto_frontier": bool(r.get("pareto_frontier", False)),
                "sharp_eff_rank": r.get("final_r_eff"),
                "kraus_stiefel_err": r.get("final_iso_err"),
            }
        )

    sort_key = cfg.sort.strip()
    if sort_key == "best_val_ce":
        filtered.sort(
            key=lambda x: (
                x["best_val_ce"] is None,
                float(x["best_val_ce"]) if x["best_val_ce"] is not None else 999.0,
            )
        )
    elif sort_key in ("avg_tok_per_s", "live_tokens_M", "compute_tflops", "wave_num", "best_top1"):
        filtered.sort(key=lambda x: -float(x.get(sort_key) or 0.0))
    elif sort_key == "wall_min":
        filtered.sort(key=lambda x: float(x.get("wall_min") or 9999.0))

    return filtered[: max(1, int(cfg.limit))]


def format_query_markdown_table(rows: list[dict[str, Any]], cfg: QueryCLIConfig) -> str:
    """Format queried rows into a GitHub-Flavored Markdown table."""
    lines = [
        f"### SQLite Benchmark Query (`tier={cfg.tier}`, `wave={cfg.wave or 'ALL'}`, `sort={cfg.sort}`, `rows={len(rows)}`)",
        "",
        "| `key` | `wave_id` | `run_tier` | `params_M` | `live_tokens_M` | `compute_tflops` | `best_val_ce` | `best_top1` | `avg_tok_per_s` | `pareto_frontier` |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        ce_str = f"{float(r['best_val_ce']):.4f}" if r.get("best_val_ce") is not None else "—"
        t1_str = f"{float(r['best_top1']):.2f}%" if r.get("best_top1") is not None else "—"
        pf_str = "★ Yes" if r.get("pareto_frontier") else "No"
        lines.append(
            f"| `{r['key']}` | `{r['wave_id']}` | `{r['run_tier']}` | "
            f"`{r['params_M']:.2f}M` | `{r['live_tokens_M']:.2f}M` | `{r['compute_tflops']:.1f}` | "
            f"**{ce_str}** | `{t1_str}` | `{r['avg_tok_per_s']:,.1f}` | {pf_str} |"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> QueryCLIConfig:
    """Parse CLI arguments into `QueryCLIConfig`."""
    p = argparse.ArgumentParser(description="Query `.cache/benchmarks.db` and render a Markdown table.")
    p.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH))
    p.add_argument("--tier", type=str, default="all", help="all | main_benchmark | fineweb_edu_local | micro_verification")
    p.add_argument("--wave", type=str, default="", help="Optional wave filter (e.g. W28, W27, W26)")
    p.add_argument("--family", type=str, default="", help="Optional arch_family filter")
    p.add_argument("--pareto-only", action="store_true")
    p.add_argument("--sort", type=str, default="best_val_ce")
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--json-out", type=str, default="")
    ns = p.parse_args(argv)
    return QueryCLIConfig(
        db_path=ns.db_path,
        tier=ns.tier,
        wave=ns.wave,
        family=ns.family,
        pareto_only=bool(ns.pareto_only),
        sort=ns.sort,
        limit=ns.limit,
        json_out=ns.json_out,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    cfg = parse_args(argv)
    rows = query_benchmark_rows(cfg)
    print(format_query_markdown_table(rows, cfg))
    if cfg.json_out:
        out_p = Path(cfg.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(
            json.dumps({"config": asdict(cfg), "count": len(rows), "rows": rows}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
