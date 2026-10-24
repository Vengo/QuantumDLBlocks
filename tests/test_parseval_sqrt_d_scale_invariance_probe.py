#!/usr/bin/env python3
"""Unit and integration tests for `scripts/parseval_sqrt_d_scale_invariance_probe.py` (Plan 043 / Wave 28)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import parseval_sqrt_d_scale_invariance_probe as probe  # noqa: E402


class TestParsevalSqrtDScaleInvarianceProbe(unittest.TestCase):
    """Tests for H1 analytical Parseval sqrt(d) scale invariance & algebraic unification probe."""

    def test_cached_or_fresh_full_probe_report_passes_all_h1_gates(self) -> None:
        cache_json = ROOT / ".cache" / "parseval_sqrt_d_scale_invariance_probe.json"
        if cache_json.exists():
            payload = json.loads(cache_json.read_text(encoding="utf-8"))
        else:
            cfg = probe.ParsevalProbeConfig()
            payload = probe.run_probe(cfg)

        self.assertEqual(payload["plan"], "EXPERIMENT_PLAN_043")
        self.assertEqual(payload["wave"], "Wave 28")
        self.assertEqual(payload["hypothesis"], "H1")
        self.assertTrue(payload["overall_pass"])

        sm = payload["summary_metrics"]
        self.assertLess(sm["bg_logit_std_cv_parseval"], 0.015)
        self.assertGreater(sm["bg_logit_std_cv_fixed"], 0.30)
        self.assertGreater(sm["unscaled_beta_1_min_cos_ce_rho1_nats"], 9.4)
        self.assertLess(sm["algebraic_unification_max_err"], 1e-6)
        self.assertEqual(sm["params_saved_d256"], 16514)
        self.assertEqual(sm["params_saved_d384"], 24706)
        self.assertEqual(sm["params_saved_d512"], 32898)

    def test_cli_execution_and_json_schema_on_compact_vocab(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "probe_out.json"
            rc = probe.main([
                "--dims", "128,256,384,512,768,1024",
                "--vocab-size", "50257",
                "--parseval-c", "1.48",
                "--num-queries", "8",
                "--device", "cpu",
                "--json-out", str(out_file),
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(out_file.exists())

            data = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertTrue(data["overall_pass"])
            self.assertIn("part1_scaling_laws", data)
            self.assertIn("part2_algebraic_and_param_audit", data)
            self.assertLess(data["summary_metrics"]["bg_logit_std_cv_parseval"], 0.015)
            self.assertGreater(data["summary_metrics"]["bg_logit_std_cv_fixed"], 0.30)
            self.assertLess(data["summary_metrics"]["algebraic_unification_max_err"], 1e-6)


if __name__ == "__main__":
    unittest.main()
