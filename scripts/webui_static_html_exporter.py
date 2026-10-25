#!/usr/bin/env python3
"""Self-Contained Static HTML Leaderboard Snapshot Exporter (`scripts/webui_static_html_exporter.py`).

Implements Experiment Plan 044:
- Reads `runs_catalog` (Full FineWeb Speedrun), `fineweb_edu_catalog` (`Waves 18–29`), and
  `micro_runs_catalog` (`Plans 013–045` & Vision Ablations) from `.cache/benchmarks.db`.
- Renders a standalone, single-file HTML snapshot with interactive client-side tab switching,
  single-screen scrollable leaderboards (`max-height: 62vh` with sticky headers), and search filtering.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import html
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qdlblocks.benchmarks.db import DEFAULT_DB_PATH, BenchmarkSQLiteDB


@dataclass(frozen=True)
class StaticExporterConfig:
    """Configuration for `scripts/webui_static_html_exporter.py`."""

    db_path: str = str(DEFAULT_DB_PATH)
    out: str = str(REPO_ROOT / ".cache" / "leaderboard_snapshot.html")
    title: str = "QuantumDLBlocks Benchmark Leaderboard Snapshot"


def _render_table_rows_html(rows: list[dict[str, Any]]) -> str:
    """Render HTML `<tr>` rows for a catalog table."""
    tr_list: list[str] = []
    for idx, r in enumerate(rows, start=1):
        rk = html.escape(str(r.get("key") or r.get("run_key") or ""))
        wid = html.escape(str(r.get("wave_id") or ""))
        fam = html.escape(str(r.get("arch_family") or ""))
        params_m = float(r.get("params_M") or 0.0)
        tok_m = float(r.get("live_tokens_M") or r.get("target_tokens_M") or 0.0)
        tflops = float(r.get("compute_tflops") or (6.0 * params_m * tok_m))
        ce_str = f"{float(r['best_val_ce']):.4f}" if r.get("best_val_ce") is not None else "—"
        ppl_str = f"{float(r['best_val_ppl']):.2f}" if r.get("best_val_ppl") is not None else "—"
        top1_str = f"{float(r['best_top1']):.2f}%" if r.get("best_top1") is not None else "—"
        tps_str = f"{float(r.get('avg_tok_per_s') or 0.0):,.0f}"
        wall_str = f"{float(r.get('wall_min') or 0.0):.2f}m"
        pf_badge = "<span class='badge pareto'>★ Pareto</span>" if r.get("pareto_frontier") else ""
        r_sharp = f"{float(r['final_r_eff']):.2f}" if r.get("final_r_eff") is not None else "—"
        iso_str = f"{float(r['final_iso_err']):.4f}" if r.get("final_iso_err") is not None else "—"
        tr_list.append(
            f"<tr>"
            f"<td>{idx}</td>"
            f"<td><code>{rk}</code> {pf_badge}</td>"
            f"<td><span class='badge'>{wid}</span></td>"
            f"<td>{fam}</td>"
            f"<td>{params_m:.2f}M</td>"
            f"<td>{tok_m:.2f}M</td>"
            f"<td>{tflops:.1f}</td>"
            f"<td class='ce-cell'>{ce_str}</td>"
            f"<td>{ppl_str}</td>"
            f"<td>{top1_str}</td>"
            f"<td>{r_sharp}</td>"
            f"<td>{iso_str}</td>"
            f"<td>{tps_str}</td>"
            f"<td>{wall_str}</td>"
            f"</tr>"
        )
    return "\n".join(tr_list)


def build_static_html(snapshot: dict[str, Any], cfg: StaticExporterConfig) -> str:
    """Build a standalone HTML document containing all three benchmark leaderboards."""
    main_cat = snapshot.get("runs_catalog", [])
    edu_cat = snapshot.get("fineweb_edu_catalog", [])
    micro_cat = snapshot.get("micro_runs_catalog", [])
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    table_head = (
        "<thead><tr>"
        "<th>#</th><th>Run Key</th><th>Wave</th><th>Family</th>"
        "<th>Params</th><th>Tokens</th><th>TFLOPs</th><th>Best Val CE</th>"
        "<th>Val PPL</th><th>Top-1</th><th>R_sharp</th><th>Stiefel Err</th>"
        "<th>Tok/s</th><th>Wall</th>"
        "</tr></thead>"
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{html.escape(cfg.title)}</title>
<style>
  body {{ background: #0d1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; margin: 0; padding: 20px; }}
  h1 {{ margin: 0 0 6px 0; font-size: 22px; color: #58a6ff; }}
  .subtitle {{ color: #8b949e; font-size: 13px; margin-bottom: 16px; }}
  .tabs {{ display: flex; gap: 10px; margin-bottom: 16px; }}
  .tab-btn {{ background: #161b22; color: #c9d1d9; border: 1px solid #30363d; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 600; }}
  .tab-btn.active {{ background: #1f6feb; color: #ffffff; border-color: #388bfd; }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}
  .table-container {{ max-height: 62vh; overflow-y: auto; overflow-x: auto; border: 1px solid #30363d; border-radius: 8px; background: #161b22; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  thead th {{ position: sticky; top: 0; z-index: 5; background: #21262d; color: #f0f6fc; text-align: left; padding: 10px 8px; border-bottom: 2px solid #30363d; }}
  tbody td {{ padding: 8px; border-bottom: 1px solid #21262d; white-space: nowrap; }}
  tbody tr:hover {{ background: #1f242c; }}
  .ce-cell {{ font-weight: 700; color: #3fb950; }}
  .badge {{ background: #30363d; color: #c9d1d9; padding: 2px 6px; border-radius: 4px; font-size: 11px; }}
  .badge.pareto {{ background: rgba(56, 239, 125, 0.18); color: #38ef7d; border: 1px solid #38ef7d; }}
</style>
</head>
<body>
  <h1>{html.escape(cfg.title)}</h1>
  <div class="subtitle">Generated {now_str} | Full FineWeb: {len(main_cat)} runs | FineWeb-Edu (Waves 18–29): {len(edu_cat)} runs | Micro &amp; Vision: {len(micro_cat)} runs</div>
  <div class="tabs">
    <button class="tab-btn active" onclick="showTab('edu', this)">⚡ FineWeb-Edu &amp; Wavelet-Cholesky (Waves 18–29) ({len(edu_cat)})</button>
    <button class="tab-btn" onclick="showTab('main', this)">🏆 Full FineWeb Speedrun ({len(main_cat)})</button>
    <button class="tab-btn" onclick="showTab('micro', this)">🧪 Micro-Verifications &amp; Vision Ablations ({len(micro_cat)})</button>
  </div>
  <div id="panel-edu" class="tab-panel active">
    <div class="table-container"><table>{table_head}<tbody>{_render_table_rows_html(edu_cat)}</tbody></table></div>
  </div>
  <div id="panel-main" class="tab-panel">
    <div class="table-container"><table>{table_head}<tbody>{_render_table_rows_html(main_cat)}</tbody></table></div>
  </div>
  <div id="panel-micro" class="tab-panel">
    <div class="table-container"><table>{table_head}<tbody>{_render_table_rows_html(micro_cat)}</tbody></table></div>
  </div>
  <script>
    function showTab(id, btn) {{
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.getElementById('panel-' + id).classList.add('active');
      btn.classList.add('active');
    }}
  </script>
</body>
</html>
"""


def export_static_html(cfg: StaticExporterConfig) -> dict[str, Any]:
    """Read SQLite snapshot and write standalone HTML file to `cfg.out`."""
    db = BenchmarkSQLiteDB(Path(cfg.db_path))
    try:
        snapshot = db.load_snapshot_payload()
    finally:
        db.close()

    html_text = build_static_html(snapshot, cfg)
    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    summary = {
        "out_html": str(out_path),
        "bytes": len(html_text.encode("utf-8")),
        "runs_catalog_count": len(snapshot.get("runs_catalog", [])),
        "fineweb_edu_catalog_count": len(snapshot.get("fineweb_edu_catalog", [])),
        "micro_runs_catalog_count": len(snapshot.get("micro_runs_catalog", [])),
    }
    print(json.dumps(summary, indent=2))
    return summary


def parse_args(argv: list[str] | None = None) -> StaticExporterConfig:
    """Parse CLI arguments into `StaticExporterConfig`."""
    p = argparse.ArgumentParser(description="Export standalone HTML leaderboard snapshot from `.cache/benchmarks.db`.")
    p.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH))
    p.add_argument("--out", "--html-out", dest="out", type=str, default=str(REPO_ROOT / ".cache" / "leaderboard_snapshot.html"))
    p.add_argument("--title", type=str, default="QuantumDLBlocks Benchmark Leaderboard Snapshot")
    ns = p.parse_args(argv)
    return StaticExporterConfig(db_path=ns.db_path, out=ns.out, title=ns.title)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    export_static_html(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
