"""Unit tests for `scripts/fineweb_convex_vs_concave_slope_tracker.py` (Plans 041–042)."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.fineweb_convex_vs_concave_slope_tracker import (
    READOUT_FAMILIES,
    SlopeTrackerConfig,
    main,
    parse_args,
    run_slope_tracker,
)


def test_parse_args_and_cli_execution(tmp_path: Path) -> None:
    """Verify CLI flag parsing and end-to-end JSON generation across Waves 19–27."""
    out_json = tmp_path / "slope_tracker_test.json"
    argv = [
        "--waves",
        "w19,w20,w21,w22,w23,w24,w25,w26,w27",
        "--target-tokens",
        "100000000",
        "--json-out",
        str(out_json),
    ]
    cfg = parse_args(argv)
    assert cfg.waves == ("w19", "w20", "w21", "w22", "w23", "w24", "w25", "w26", "w27")
    assert cfg.target_tokens == 100_000_000

    rc = main(argv)
    assert rc == 0
    assert out_json.exists()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "iso_token_comparison" in payload
    assert "iso_token_table" in payload
    assert "part_a_empirical_scaling" in payload
    assert "part_b_readout_derivative_profiler" in payload
    assert "runs_by_key" in payload
    assert "readout_families" in payload
    assert "summary" in payload


def test_part_a_empirical_scaling_and_asymptotic_floors() -> None:
    """Verify Part A log-token slopes, 100M/300M projections, and concave vs. convex floors."""
    report = run_slope_tracker(SlopeTrackerConfig(json_out=""))
    part_a = report["part_a_empirical_scaling"]
    runs_by_key = part_a["runs_by_key"]

    required_keys = (
        "w25_8gb_convex_traceless_born_3600steps",
        "w25_8gb_convex_traceless_born_900steps",
        "w25_8gb_last_layer_meas_convex_traceless_900steps",
        "w23_8gb_last_layer_meas_traceless",
        "w22_8gb_radical_2pow16_1bank_fast",
        "w21_8gb_2pow16_16bank",
    )
    for k in required_keys:
        assert k in runs_by_key, f"Missing expected run key: {k}"

    w25_3600 = runs_by_key["w25_8gb_convex_traceless_born_3600steps"]
    w25_900 = runs_by_key["w25_8gb_convex_traceless_born_900steps"]
    w25_llm_900 = runs_by_key["w25_8gb_last_layer_meas_convex_traceless_900steps"]
    w23_3600 = runs_by_key["w23_8gb_last_layer_meas_traceless"]
    w22_3600 = runs_by_key["w22_8gb_radical_2pow16_1bank_fast"]
    w21_3600 = runs_by_key["w21_8gb_2pow16_16bank"]

    # Check exact early and late log-token slope definitions on w25_3600
    expected_early = (5.5970 - 6.1478) / math.log(900.0 / 300.0)
    expected_late = (5.0162 - 5.5970) / math.log(3600.0 / 900.0)
    assert abs(w25_3600["early_log_slope_300_to_900"] - expected_early) < 1e-3
    assert abs(w25_3600["late_log_slope_900_to_3600"] - expected_late) < 1e-3
    assert w25_3600["slope_acceleration_ratio"] > 0.80

    # 900-step convex traceless runs have steeper early log-token slope than concave w23/w22/w21
    assert w25_900["early_log_slope_300_to_900"] < w25_llm_900["early_log_slope_300_to_900"] < -0.63
    assert w25_3600["final_val_ce"] < w21_3600["final_val_ce"] < w22_3600["final_val_ce"] < w23_3600["final_val_ce"]

    # Asymptotic floor & 100M/300M projections: concave stalls > 3.40 nats, convex < 3.40 nats
    assert w23_3600["asymptotic_floor_ce"] > 3.40
    assert w22_3600["asymptotic_floor_ce"] > 3.40
    assert w21_3600["asymptotic_floor_ce"] > 3.40
    assert w25_3600["asymptotic_floor_ce"] < 3.40
    assert w25_3600["projected_ce_100m_log_linear"] < w23_3600["projected_ce_100m_log_linear"]
    assert w25_3600["projected_ce_100m_power_law"] < w23_3600["projected_ce_100m_power_law"]
    assert w25_3600["projected_ce_300m_power_law"] < w25_3600["projected_ce_100m_power_law"]


def test_iso_token_and_iso_flop_standardization_across_batch_sizes_and_scales() -> None:
    """Verify standardized token milestones (`ce_at_*`), FLOPs, update density, and iso-token slopes."""
    report = run_slope_tracker(SlopeTrackerConfig(json_out=""))
    iso_comp = report["iso_token_comparison"]
    assert iso_comp["milestones_M"] == [2.46, 7.37, 14.75, 29.49, 44.24]
    assert len(iso_comp["rows"]) >= 10
    assert "joint_chinchilla_scaling" in iso_comp
    assert "best_by_milestone" in iso_comp

    runs = report["runs_by_key"]
    w25_3600 = runs["w25_8gb_convex_traceless_born_3600steps"]
    w26_b16_900 = runs["w26_15gb_dyadic_convex_traceless_b16_900steps"]
    w26_b48_d256 = runs["w26_15gb_b48_d256_L6_dyadic_fp8_1800steps"]
    w26_b48_d384 = runs["w26_15gb_b48_d384_L8_dyadic_fp8_600steps"]

    # B=16 (8,192 tok/step) standardized token milestones & FLOPs
    assert w25_3600["tokens_per_step"] == 8192
    assert w25_3600["optimizer_updates_per_10m_tokens"] == 1220.7
    assert w25_3600["flops_per_token_M"] == 62.4
    assert w25_3600["total_compute_tflops"] == round((6.0 * 10.4 * 1e6 * 29.49 * 1e6) / 1e12, 2)
    assert w25_3600["ce_at_2p46m"] == 6.1478
    assert w25_3600["ce_at_7p37m"] == 5.5970
    assert w25_3600["ce_at_14p75m"] == 5.3782
    assert w25_3600["ce_at_29p49m"] == 5.0162
    assert abs(w25_3600["slope_2p46m_to_7p37m"] - w25_3600["early_log_slope_300_to_900"]) < 1e-4
    assert abs(w25_3600["slope_7p37m_to_final"] - w25_3600["late_log_slope_900_to_3600"]) < 1e-4

    # B=48 (24,576 tok/step) standardized token milestones & FLOPs
    assert w26_b48_d256["batch_size"] == 48
    assert w26_b48_d256["tokens_per_step"] == 24576
    assert w26_b48_d256["optimizer_updates_per_10m_tokens"] == 406.9
    assert w26_b48_d256["total_compute_tflops"] == round((6.0 * 10.4 * 1e6 * 44.24 * 1e6) / 1e12, 2)
    assert w26_b48_d256["ce_at_2p46m"] == 6.2001
    assert w26_b48_d256["ce_at_7p37m"] == 5.5536
    assert w26_b48_d256["ce_at_14p75m"] == 5.2203
    assert w26_b48_d256["ce_at_29p49m"] == 5.0368
    assert w26_b48_d256["ce_at_44p24m"] == 4.8199

    # Model width scaling (d=384 vs d=256 at B=48) and batch-size step density (B=16 vs B=48 at 7.37M)
    assert w26_b48_d384["embed_dim"] == 384
    assert w26_b48_d384["ce_at_14p75m"] < w26_b48_d256["ce_at_14p75m"]
    assert w26_b48_d384["slope_7p37m_to_14p75m"] < w26_b48_d256["slope_7p37m_to_14p75m"]
    assert w26_b16_900["ce_at_7p37m"] < w26_b48_d256["ce_at_7p37m"]


def test_part_b_readout_derivative_and_slope_ratio_hierarchy() -> None:
    """Verify exact analytical vs. autograd derivatives and peak-to-tail slope ratio hierarchy."""
    report = run_slope_tracker(SlopeTrackerConfig(json_out=""))
    part_b = report["part_b_readout_derivative_profiler"]
    fams = part_b["families"]

    for fam in READOUT_FAMILIES:
        assert fam in fams
        entry = fams[fam]
        assert entry["max_autograd_abs_err"] < 1e-6
        for pt in ("+0.2", "+0.4", "+0.6", "+0.8"):
            assert pt in entry["peak_points"]
        for pt in ("-0.2", "-0.1", "-0.0625"):
            assert pt in entry["tail_points"]

    r_naive = fams["naive_kboson_born"]["peak_to_tail_slope_ratio"]
    r_w23 = fams["w23_traceless_kboson"]["peak_to_tail_slope_ratio"]
    r_convex = fams["convex_traceless_born"]["peak_to_tail_slope_ratio"]
    r_dyadic = fams["convex_traceless_dyadic_born"]["peak_to_tail_slope_ratio"]

    # Concave readouts compress peak derivatives (< 1.0x), whereas convex traceless readouts amplify (> 1.0x)
    assert r_naive < r_w23 < 1.0 < r_convex < r_dyadic
    assert not fams["naive_kboson_born"]["peak_slope_monotonic_increasing"]
    assert fams["convex_traceless_born"]["peak_slope_monotonic_increasing"]
    assert fams["convex_traceless_dyadic_born"]["peak_slope_monotonic_increasing"]


if __name__ == "__main__":
    import tempfile

    test_parse_args_and_cli_execution(Path(tempfile.mkdtemp()))
    test_part_a_empirical_scaling_and_asymptotic_floors()
    test_iso_token_and_iso_flop_standardization_across_batch_sizes_and_scales()
    test_part_b_readout_derivative_and_slope_ratio_hierarchy()
    print("ALL test_fineweb_slope_tracker.py TESTS PASSED")
