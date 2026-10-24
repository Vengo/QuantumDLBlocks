#!/usr/bin/env python3
"""Unit and integration tests for `unit_sphere_mup_transfer_grid_search.py` and `cholesky_metric_temperature_absorber.py`."""

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

import cholesky_metric_temperature_absorber as chol_abs  # noqa: E402
import unit_sphere_mup_transfer_grid_search as mup_grid  # noqa: E402


class TestMuPTransferAndCholeskyAbsorber(unittest.TestCase):
    """Tests for unit-sphere Parseval muP zero-shot transfer and Cholesky SPD metric temperature absorption."""

    def test_mup_transfer_grid_search_cached_or_fresh_passes_all_gates(self) -> None:
        cache_json = ROOT / ".cache" / "unit_sphere_mup_transfer_grid_search.json"
        if cache_json.exists():
            payload = json.loads(cache_json.read_text(encoding="utf-8"))
        else:
            cfg = mup_grid.MuPGridSearchConfig()
            payload = mup_grid.evaluate_parameterizations(cfg)

        self.assertEqual(payload["script"], "unit_sphere_mup_transfer_grid_search")
        self.assertTrue(payload["overall_pass"])

        sm = payload["summary_metrics"]
        self.assertTrue(sm["sp_optimal_beta_shifts_upward"])
        self.assertTrue(sm["zero_shot_transfer_holds"])
        self.assertAlmostEqual(sm["mup_invariant_c_star"], 1.48, places=4)
        self.assertGreater(sm["sp_zero_shot_transfer_regret_nats"], 0.05)
        self.assertAlmostEqual(sm["mup_zero_shot_transfer_regret_nats"], 0.0, places=9)

        sp_betas = sm["sp_optimal_betas_by_dim"]
        self.assertLess(sp_betas["128"], sp_betas["256"])
        self.assertLess(sp_betas["256"], sp_betas["384"])
        self.assertLess(sp_betas["384"], sp_betas["512"])

        for d_str in ("128", "256", "384", "512"):
            self.assertAlmostEqual(sm["mup_optimal_c_by_dim"][d_str], 1.48, places=4)

    def test_mup_transfer_cli_execution_and_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "mup_grid.json"
            rc = mup_grid.main([
                "--dims", "128,256,384,512",
                "--c-grid", "1.15,1.30,1.48,1.65,1.85",
                "--steps", "35",
                "--vocab-size", "1024",
                "--num-queries", "8",
                "--device", "cpu",
                "--json-out", str(out_file),
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(out_file.exists())

            data = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertTrue(data["overall_pass"])
            self.assertTrue(data["summary_metrics"]["zero_shot_transfer_holds"])
            self.assertTrue(data["summary_metrics"]["sp_optimal_beta_shifts_upward"])

    def test_cholesky_absorber_cached_or_fresh_passes_all_gates(self) -> None:
        cache_json = ROOT / ".cache" / "cholesky_metric_temperature_absorber.json"
        if cache_json.exists():
            payload = json.loads(cache_json.read_text(encoding="utf-8"))
        else:
            cfg = chol_abs.CholeskyAbsorberConfig()
            payload = chol_abs.run_absorber_audit(cfg)

        self.assertEqual(payload["script"], "cholesky_metric_temperature_absorber")
        self.assertTrue(payload["overall_pass"])

        sm = payload["summary_metrics"]
        self.assertLess(sm["max_logit_abs_err"], 1e-5)
        self.assertLess(sm["max_grad_abs_err"], 1e-5)
        self.assertLess(sm["max_cholesky_factor_err"], 1e-6)
        self.assertLess(sm["max_spd_cond_number_rel_err"], 1e-5)
        self.assertEqual(sm["absorbed_kernel_beta"], 1.0)

        for d in (256, 384, 512):
            row = payload["by_dim"][f"d_{d}"]
            self.assertLess(row["max_logit_abs_err"], 1e-5)
            self.assertLess(row["max_grad_abs_err"], 1e-5)
            self.assertEqual(row["absorbed_kernel_beta"], 1.0)

    def test_cholesky_absorber_cli_execution_and_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "chol_absorber.json"
            rc = chol_abs.main([
                "--dims", "256,384,512",
                "--parseval-c", "1.48",
                "--device", "cpu",
                "--json-out", str(out_file),
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(out_file.exists())

            data = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertTrue(data["overall_pass"])
            self.assertLess(data["summary_metrics"]["max_logit_abs_err"], 1e-5)
            self.assertLess(data["summary_metrics"]["max_grad_abs_err"], 1e-5)


if __name__ == "__main__":
    unittest.main()
