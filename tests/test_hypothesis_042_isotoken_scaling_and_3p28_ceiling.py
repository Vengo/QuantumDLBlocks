"""Hypothesis acceptance gate tests for Experiment Plan 042 (`Wave 27`):
Verifies `H1`, `H2`, `H3`, and `overall_pass` in `.cache/hypothesis_experiments_042.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HYP042_JSON = ROOT / ".cache" / "hypothesis_experiments_042.json"


def test_hypothesis_042_overall_and_h1_h2_h3_gates() -> None:
    """Assert `overall_pass` and all falsifiable gates for H1, H2, and H3 in Plan 042."""
    assert HYP042_JSON.exists(), f"Missing {HYP042_JSON}"
    data = json.loads(HYP042_JSON.read_text(encoding="utf-8"))
    assert data["plan_id"] == "EXPERIMENT_PLAN_042"
    assert data["overall_pass"] is True

    h1 = data["H1_isotoken_isoflop_slope_tracker"]
    assert h1["passed"] is True
    assert h1["num_tracked_runs"] >= 8

    h2 = data["H2_empirical_3p28_ceiling_and_wavelet_rank_probe"]
    assert h2["passed"] is True
    h2_sum = h2["summary"]
    assert h2_sum["proportional_ranks_beat_fixed_for_d384_d512"] is True
    assert h2_sum["all_widths_break_3p28_ce"] is True
    assert h2_sum["all_widths_break_1p80_ce"] is True
    assert h2_sum["monotonic_width_ceiling_d512_lt_d384_lt_d256"] is True
    assert h2_sum["min_ce_d512_fp8"] < h2_sum["min_ce_d384_fp8"] < h2_sum["min_ce_d256_fp8"] < 2.60

    h3 = data["H3_isotoken_smaller_batch_bigger_network_scaling"]
    assert h3["passed"] is True
    assert h3["max_peak_vram_gb"] <= 15.0

    c7 = h3["iso_token_7p37m_comparison"]
    assert (
        c7["ce_7p37m_d512_b16"]
        < c7["ce_7p37m_d384_b16"]
        < c7["ce_7p37m_d256_b16_w26"]
        < c7["ce_7p37m_d384_b48_w26"]
    )
    assert c7["ce_7p37m_d512_b24"] < c7["ce_7p37m_d256_b16_w26"]

    c14 = h3["iso_token_14p75m_comparison"]
    assert (
        c14["ce_14p75m_d512_b24_uncooled"]
        < c14["ce_14p75m_d256_b48_w26_uncooled"]
        < c14["ce_14p75m_d256_b16_w25_uncooled"]
    )
    assert (
        c14["ce_14p75m_d512_b24_cooled_est"]
        < c14["ce_14p75m_d384_b48_w26_cooled"]
        < c14["ce_14p75m_d256_b48_w26_uncooled"]
    )

    c29 = h3["iso_token_29p49m_comparison"]
    assert (
        c29["ce_29p49m_d512_b24"]
        < c29["ce_44p24m_d256_b48_w26_prev_all_time_record"]
        < c29["ce_29p49m_d256_b16_w25"]
    )
    assert c29["delta_vs_w25_29p49m_nats"] < -0.20
    assert c29["delta_vs_w26_44p24m_all_time_record_nats"] < 0.0
