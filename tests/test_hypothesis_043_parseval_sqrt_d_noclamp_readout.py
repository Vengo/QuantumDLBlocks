"""Hypothesis acceptance gate tests for Experiment Plan 043 (`Wave 28`):
Verifies `H1`, `H2`, `H3`, `H4`, and `overall_pass` in `.cache/hypothesis_experiments_043.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
HYP043_JSON = ROOT / ".cache" / "hypothesis_experiments_043.json"


def test_hypothesis_043_overall_and_h1_h2_h3_h4_gates() -> None:
    """Assert `overall_pass` and all falsifiable gates for H1, H2, H3, and H4 in Plan 043."""
    assert HYP043_JSON.exists(), f"Missing {HYP043_JSON}"
    data = json.loads(HYP043_JSON.read_text(encoding="utf-8"))
    assert data.get("plan") == "EXPERIMENT_PLAN_043"
    assert data["overall_pass"] is True

    h1 = data["H1"]
    assert h1["pass"] is True
    assert h1["bg_logit_std_cv_parseval"] < 0.015
    assert h1["bg_logit_std_cv_fixed"] > 0.30
    assert h1["max_beam_unification_err"] < 1e-6
    assert h1["max_transfer_regret_nats"] <= 0.01
    assert h1["cholesky_max_abs_logit_err"] < 1e-4

    h2 = data["H2"]
    assert h2["pass"] is True
    assert h2["final_val_ce"] <= 5.4447
    assert h2["zero_purity_gate"] is True
    assert h2["zero_kboson_raw_k"] is True
    assert h2["zero_beta_clamps"] is True

    h3 = data["H3"]
    assert h3["pass"] is True
    assert h3["strict_monotonic_improvement"] is True
    assert h3["ce_d512"] < h3["ce_d384"] < h3["ce_d256"] <= 5.4447

    h4 = data["H4"]
    assert h4["pass"] is True
    assert 0.004 <= h4["eta_base"] <= 0.015
    assert 0.020 <= h4["eta_max"] <= 0.045
    assert h4["profiler_triangular2_beats_wsd_at_tau_0p50"] is True
    assert h4["smith_clr_val_ce_at_tau_0p50"] < h4["wsd_val_ce_at_tau_0p50"]
    assert h4["smith_clr_final_val_ce"] <= h4["wsd_final_val_ce"] + 0.02


class TestHypothesis043Gates(unittest.TestCase):
    """Unittest wrapper for Plan 043 hypothesis acceptance gates."""

    def test_gates(self) -> None:
        test_hypothesis_043_overall_and_h1_h2_h3_h4_gates()


if __name__ == "__main__":
    unittest.main()
