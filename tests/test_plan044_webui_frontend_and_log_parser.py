"""Unit tests for Plan 044 WebUI Frontend & Task-Aware Log Parser (`scripts/monitor_colab_runs.py`).

Covers:
1. Change 3A: Task-aware `parse_log_metrics()` supporting both `dict` and `list` `results_json`
   (Waves 20-29) and `[EXPERIMENT_INTENT]` header parsing (`task_domain`, `hypothesis_intent`).
2. Change 2A: Single-screen scrollable `.table-container` (`max-height: 62vh; overflow-y: auto;`)
   with sticky `th` (`position: sticky; top: 0; z-index: 5; background: #1c2128;`).
3. Change 2B: Click-and-drag rectangular box zoom on `SvgTrajectoryChart` (`zoomBox`, `dragRect`,
   mouse handlers, `"🔍 Reset Zoom"`, `"🖱️ Drag to Box-Zoom"`).
4. Change 2C: Dedicated Tab 2 (`activeTab === "fineweb_edu"`) for Waves 18-29 FineWeb-Edu &
   Wavelet-Cholesky scaling runs and partitioning of `fineweb_edu_catalog` in `collect_snapshot()`.
5. Change 3B: Standardized horizontal X-axis toggle (`xAxisMode`: `"tokens"` | `"tflops"` | `"tau"` | `"time"`),
   removal of `SvgRankScalingChart` (`"Quantum State Capacity Law"`), and placement of
   `SvgSeedScalingChart` + `"Seed Quantum State Effective Rank R_eff(rho_seed)"` at the bottom of Tab 1.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.monitor_colab_runs import (  # noqa: E402
    LIVE_DIR,
    collect_snapshot,
    parse_log_metrics,
    render_html_dashboard,
)


def test_parse_log_metrics_supports_wave28_dict_results_json() -> None:
    """Verify `parse_log_metrics()` parses Wave 28 dict `*_results.json` and ignores step=0 CE."""
    w28_path = LIVE_DIR / "w28_flagfree_parseval_mup_b16_d512_900steps_results.json"
    assert w28_path.exists(), f"Expected Wave 28 results file at {w28_path}"
    raw_dict = json.loads(w28_path.read_text(encoding="utf-8"))
    assert isinstance(raw_dict, dict)

    parsed = parse_log_metrics(
        log_text="",
        peak_tflops=100.0,
        default_tps=20000.0,
        results_json=raw_dict,
        log_mtime=0.0,
        ps_alive=False,
    )

    assert parsed["completed"] is True
    assert parsed["run_state"] == "COMPLETED"
    assert parsed["best_val_ce"] is not None
    assert abs(float(parsed["best_val_ce"]) - 5.3671) < 1e-3
    assert parsed["embed_dim"] == 512
    assert parsed["num_layers"] == 6
    assert abs(float(parsed["params_M"]) - 33.4769) < 0.05
    assert float(parsed["avg_tok_per_s"]) > 8000.0
    assert float(parsed["sharp_eff_rank"]) > 20.0
    assert float(parsed["kraus_stiefel_err"]) <= 0.03

    evals = parsed["evals"]
    assert len(evals) >= 4
    last_ev = evals[-1]
    assert abs(float(last_ev["val_ce"]) - 5.3671) < 1e-3
    assert float(last_ev["compute_tflops"]) > 10.0
    assert abs(float(last_ev["tau_progress"]) - 1.0) < 1e-3
    assert float(last_ev["sharp_eff_rank"]) > 20.0
    assert float(last_ev["kraus_stiefel_err"]) <= 0.03
    assert float(last_ev["top50_acc"]) > 60.0


def test_parse_log_metrics_parses_experiment_intent_and_list_json() -> None:
    """Verify `parse_log_metrics()` extracts `[EXPERIMENT_INTENT]` fields and supports list JSON."""
    log_text = """
====================================================================================
[EXPERIMENT_INTENT] Run: w29_15gb_flagfree_parseval_mup_b32_d384_L6_1800steps (Wave W29 • EXPERIMENT_PLAN_044)
  Task Domain:       fineweb_edu_local
  Hypothesis Intent: Scale Flag-Free Parseval muP + 256-Root Dyadic GPT to d=384, L=6 over 147.5M tokens.
  Architecture:      ParsevalMuPDyadicGPT (d=384, L=6, R=4, 256-root dyadic)
  Hardware Budget:   Local 15GB CUDA Budget | B=32, T=512 (32,768 tok/step) | 1800 steps (147.5M tok)
====================================================================================
"""
    list_results = [
        {
            "step": 450,
            "epoch": 0.25,
            "tokens_M": 36.86,
            "compute_tflops": 9.5,
            "tau_progress": 0.25,
            "val_ce": 5.6200,
            "val_ppl": 275.89,
            "top1_acc": 21.4,
            "top5_acc": 41.2,
            "sharp_eff_rank": 29.5,
            "kraus_stiefel_err": 0.022,
            "elapsed_s": 1800.0,
        },
        {
            "step": 1800,
            "epoch": 1.0,
            "tokens_M": 147.46,
            "compute_tflops": 38.0,
            "tau_progress": 1.0,
            "val_ce": 4.8900,
            "val_ppl": 132.95,
            "top1_acc": 25.1,
            "top5_acc": 46.8,
            "sharp_eff_rank": 31.8,
            "kraus_stiefel_err": 0.021,
            "elapsed_s": 7200.0,
        },
    ]

    parsed = parse_log_metrics(
        log_text=log_text,
        peak_tflops=100.0,
        default_tps=20000.0,
        results_json=list_results,
        log_mtime=0.0,
        ps_alive=False,
    )

    assert parsed["task_domain"] == "fineweb_edu_local"
    assert "Scale Flag-Free Parseval muP" in parsed["hypothesis_intent"]
    assert abs(float(parsed["best_val_ce"]) - 4.8900) < 1e-4
    assert len(parsed["evals"]) == 2
    assert abs(float(parsed["evals"][1]["compute_tflops"]) - 38.0) < 1e-4
    assert abs(float(parsed["evals"][1]["tau_progress"]) - 1.0) < 1e-4


def test_collect_snapshot_populates_fineweb_edu_catalog() -> None:
    """Verify `collect_snapshot()` partitions Waves 18-29 FineWeb-Edu runs into `fineweb_edu_catalog`."""
    snap = collect_snapshot(skip_remote_poll=True)
    assert "runs_catalog" in snap
    assert "fineweb_edu_catalog" in snap
    assert "micro_runs_catalog" in snap

    edu_catalog = snap["fineweb_edu_catalog"]
    assert len(edu_catalog) >= 10
    edu_keys = {r["key"] for r in edu_catalog}
    assert "w28_flagfree_parseval_mup_b16_d512_900steps" in edu_keys
    assert "w28_flagfree_parseval_mup_b16_d256_900steps" in edu_keys

    # Ensure CIFAR-100 / 1000-class vision ablations are routed to micro_runs_catalog, not fineweb_edu_catalog
    for k in edu_keys:
        assert "cifar100" not in k.lower()
        assert "1000class" not in k.lower()


def test_render_html_dashboard_contains_sticky_table_box_zoom_tab2_and_xaxis_toggle() -> None:
    """Verify HTML dashboard includes sticky table headers, box zoom, Tab 2, X-axis toggle, and decluttered Tab 1."""
    snap = collect_snapshot(skip_remote_poll=True)
    html = render_html_dashboard(snap)

    # Change 2A: Single-screen scrollable .table-container + sticky th
    assert "max-height: 62vh;" in html
    assert "overflow-y: auto;" in html
    assert "position: sticky;" in html
    assert "top: 0;" in html
    assert "z-index: 5;" in html
    assert "background: #1c2128;" in html

    # Change 2B: Click-and-drag rectangular box zoom in SvgTrajectoryChart
    assert "zoomBox" in html
    assert "dragRect" in html
    assert "onMouseDown: handleMouseDown" in html
    assert "onMouseMove: handleMouseMove" in html
    assert "onMouseUp: handleMouseUp" in html
    assert "onDoubleClick: handleDoubleClick" in html
    assert "🔍 Reset Zoom" in html
    assert "🖱️ Drag to Box-Zoom" in html

    # Change 2C: Dedicated Tab 2 (FineWeb-Edu & Wavelet-Cholesky Waves 18-29)
    assert 'activeTab === "fineweb_edu"' in html
    assert "⚡ FineWeb-Edu & Wavelet-Cholesky (Waves 18–29" in html
    assert "w28_flagfree_parseval_mup_b16_d512_900steps" in html
    assert "5.3671 Val CE" in html
    assert "4.9228 Val CE" in html
    assert "5.8943 CE at τ=0.50" in html
    assert "FineWeb-Edu Validation CE vs" in html
    assert "Sharp Detector Effective Rank R_sharp vs" in html
    assert "Kraus & Stiefel Manifold Drift ||K_r^T K_r - I||_F / sqrt(d) vs" in html

    # Change 3B: Standardized X-Axis Toggle & Decluttered Tab 1
    assert "xAxisMode" in html
    assert "Tokens (M)" in html
    assert "Compute (TFLOPs)" in html
    assert "Schedule Progress τ" in html
    assert "Wall Time (min)" in html

    # Quantum State Capacity Law (SvgRankScalingChart) must be completely removed
    assert "SvgRankScalingChart" not in html
    assert "Quantum State Capacity Law" not in html

    # SvgSeedScalingChart and Seed Effective Rank must be placed AFTER the main leaderboard table
    table_idx = html.find('activeTab === "main"')
    assert table_idx != -1
    main_section = html[table_idx:]
    runs_table_pos = main_section.find('className: "table-container"')
    secondary_panel_pos = main_section.find(
        "📉 Secondary & Historical Diagnostics (Seed Dimension D_seed & Seed State Effective Rank)"
    )
    seed_chart_pos = main_section.find("h(SvgSeedScalingChart")
    seed_reff_pos = main_section.find("Seed Quantum State Effective Rank R_eff(rho_seed) vs")

    assert runs_table_pos != -1
    assert secondary_panel_pos > runs_table_pos
    assert seed_chart_pos > secondary_panel_pos
    assert seed_reff_pos > secondary_panel_pos


class TestPlan044WebUIFrontendAndLogParser(unittest.TestCase):
    """Unittest wrapper for Plan 044 WebUI Frontend & Task-Aware Log Parser tests."""

    def test_wave28_dict_results_json(self) -> None:
        test_parse_log_metrics_supports_wave28_dict_results_json()

    def test_experiment_intent_and_list_json(self) -> None:
        test_parse_log_metrics_parses_experiment_intent_and_list_json()

    def test_snapshot_fineweb_edu_catalog(self) -> None:
        test_collect_snapshot_populates_fineweb_edu_catalog()

    def test_html_dashboard_features(self) -> None:
        test_render_html_dashboard_contains_sticky_table_box_zoom_tab2_and_xaxis_toggle()


if __name__ == "__main__":
    unittest.main(verbosity=2)
