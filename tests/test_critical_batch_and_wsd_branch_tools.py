#!/usr/bin/env python3
"""Unit and JSON schema tests for Wave 27 promoted diagnostic scripts.

Tests:
1. `scripts/critical_batch_size_gradient_noise_profiler.py`
2. `scripts/isotoken_wsd_cooldown_branch_evaluator.py`
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.critical_batch_size_gradient_noise_profiler import (  # noqa: E402
    CriticalBatchProfilerConfig,
    parse_args as parse_crit_args,
    run_critical_batch_profiler,
)
from scripts.isotoken_wsd_cooldown_branch_evaluator import (  # noqa: E402
    WSDCooldownEvaluatorConfig,
    parse_args as parse_wsd_args,
    predict_wsd_cooldown_branch_delta,
    run_wsd_branch_evaluator,
    wsd_lr_multiplier,
)


def test_critical_batch_size_gradient_noise_profiler_cli_and_json() -> None:
    """Verify CLI parsing, McCandlish B_crit scaling, and JSON output of the critical batch profiler."""
    cfg = parse_crit_args([
        "--dims", "256,384,512",
        "--batch-sizes", "8,16,24,32,48",
        "--seq-len", "512",
        "--device", "cpu",
        "--json-out", ".cache/critical_batch_size_gradient_noise_profiler.json",
    ])
    assert cfg.dims == (256, 384, 512)
    assert cfg.batch_sizes == (8, 16, 24, 32, 48)
    assert cfg.seq_len == 512

    report = run_critical_batch_profiler(cfg)
    assert report["overall_pass"] is True
    assert all(report["gates"].values())

    sm = report["summary"]
    b_crit_5p4 = float(sm["mean_b_crit_tokens_at_ce5p40_7p37M"])
    b_crit_4p82 = float(sm["mean_b_crit_tokens_at_ce4p82_29p49M"])
    assert 12000.0 <= b_crit_5p4 <= 17500.0
    assert b_crit_4p82 > 24576.0 > b_crit_5p4

    ref_rows = {int(r["batch_size"]): r for r in report["reference_batch_table_ce5p40"]}
    assert ref_rows[16]["tokens_per_step"] == 8192
    assert ref_rows[24]["tokens_per_step"] == 12288
    assert ref_rows[48]["tokens_per_step"] == 24576
    assert ref_rows[16]["token_efficiency_eta"] > ref_rows[24]["token_efficiency_eta"] > ref_rows[48]["token_efficiency_eta"]
    assert ref_rows[48]["b_over_b_crit_ratio"] > 1.40

    out_path = ROOT / ".cache" / "critical_batch_size_gradient_noise_profiler.json"
    assert out_path.exists()
    disk_data = json.loads(out_path.read_text(encoding="utf-8"))
    assert disk_data["overall_pass"] is True
    assert "d_256" in disk_data["by_dim"] and "d_384" in disk_data["by_dim"] and "d_512" in disk_data["by_dim"]


def test_isotoken_wsd_cooldown_branch_evaluator_cli_and_json() -> None:
    """Verify WSD schedule phase deconfounding (`-0.1505` nats at 7.37M) and live CPU branch evaluation."""
    assert abs(wsd_lr_multiplier(0.25) - 1.0) < 1e-6
    assert abs(wsd_lr_multiplier(1.0) - 0.10) < 1e-6
    assert abs(predict_wsd_cooldown_branch_delta(0.25, 7.37, -0.1505) - (-0.1505)) < 1e-4
    assert predict_wsd_cooldown_branch_delta(1.0, 7.37, -0.1505) == 0.0

    cfg = parse_wsd_args([
        "--waves", "w25,w26,w27",
        "--milestones", "2.46,7.37,14.75,29.49,44.24",
        "--device", "cpu",
        "--json-out", ".cache/isotoken_wsd_cooldown_branch_evaluator.json",
    ])
    assert cfg.waves == ("w25", "w26", "w27")
    assert cfg.milestones_m == (2.46, 7.37, 14.75, 29.49, 44.24)

    report = run_wsd_branch_evaluator(cfg)
    assert report["overall_pass"] is True
    assert all(report["gates"].values())

    sm = report["summary"]
    assert abs(float(sm["calibrated_w25_7p37m_cooldown_delta_nats"]) - (-0.1505)) < 1e-3
    assert abs(float(sm["w25_3600_step900_standardized_tau1_val_ce"]) - 5.4465) < 1e-3
    assert float(sm["live_cpu_branch_cooldown_delta_nats"]) < 0.0

    out_path = ROOT / ".cache" / "isotoken_wsd_cooldown_branch_evaluator.json"
    assert out_path.exists()
    disk_data = json.loads(out_path.read_text(encoding="utf-8"))
    assert disk_data["overall_pass"] is True
    assert "w25_8gb_convex_traceless_born_3600steps" in disk_data["runs_by_key"]


def main() -> int:
    """Run all tests directly when invoked as a script."""
    test_critical_batch_size_gradient_noise_profiler_cli_and_json()
    test_isotoken_wsd_cooldown_branch_evaluator_cli_and_json()
    print("All critical batch size & WSD cooldown branch evaluator tests PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
