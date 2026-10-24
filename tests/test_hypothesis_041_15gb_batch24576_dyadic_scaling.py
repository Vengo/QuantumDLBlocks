"""Hypothesis verification test suite for Experiment Plan 041 (`Wave 26`)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / ".cache" / "hypothesis_experiments_041.json"


def test_hypothesis_041_overall_and_individual_gates() -> None:
    """Verify `H1`, `H2`, and `H3` pass in `.cache/hypothesis_experiments_041.json`."""
    assert REPORT_PATH.exists(), "Plan 041 report (.cache/hypothesis_experiments_041.json) must exist"
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert report["overall_pass"] is True

    h1 = report["H1_rtx5060ti_15gb_bf16_fp8_int8_and_batch24576_sizing"]
    assert h1["passed"] is True
    assert h1["all_activations_bfloat16"] is True
    assert h1["all_candidates_ok_within_15gb"] is True
    assert h1["b48_fp8_tok_per_s"] > h1["b48_bf16_tok_per_s"] > h1["b48_int8_tok_per_s"]
    assert h1["b48_fp8_speedup_vs_b16"] >= 1.55

    h2 = report["H2_solution_a_plus_b_dyadic_hierarchical_born_ablation"]
    assert h2["passed"] is True
    assert h2["w26_dyadic_900_val_ce"] < h2["w25_convex_traceless_900_val_ce"]
    assert h2["w26_dyadic_900_val_ce"] < h2["w21_900_val_ce"]
    assert h2["w26_dyadic_900_kraus_stiefel_err"] <= 0.05

    h3 = report["H3_batch24576_15gb_network_scaling_fineweb"]
    assert h3["passed"] is True
    assert h3["w26_b48_d256_final_val_ce"] < h3["w25_3600_record_val_ce"]
    assert h3["w26_b48_d384_L8_step600_val_ce"] < h3["w26_b48_d256_L6_step600_val_ce"]
    assert h3["max_peak_vram_gb"] <= 15.0
