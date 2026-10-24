"""Hypothesis & AST Compliance Tests for Experiment Plan 040 (`Wave 25`):
1. `H1`: 900-Step (`7.37M` Tokens) `FineWeb-Edu` Ablation of `Convex-Peak Traceless Born`.
2. `H2`: 3,600-Step (`29.49M` Tokens) `FineWeb-Edu` Scaling of `Convex-Peak Traceless Born`.
3. `H3`: Genuine `ILSVRC ImageNet-1K` (`1,281,167` Train / `50,000` Val / `100,000` Test, `1,000` Classes)
   Quantum Building-Block & Readout Ablation Study.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]

PLAN040_TARGET_FILES = (
    "src/qdlblocks/models/wavelet_cholesky_vision.py",
    "src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py",
    "scripts/multiclass_ce_margin_decomposer.py",
    "scripts/dyadic_tree_conditional_calibration_probe.py",
    "examples/50_verify_plan040_fineweb_3600_and_imagenet1k_ablations.py",
    "tests/test_plan040_imagenet1k_and_convex_traceless_scaling.py",
    "tests/test_hypothesis_040_fineweb_3600_and_imagenet1k_ablations.py",
)


def _load_linter_module():
    linter_path = ROOT / "scripts" / "ast_signature_and_length_linter.py"
    spec = importlib.util.spec_from_file_location("ast_signature_and_length_linter", linter_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_plan040_ast_best_practices_compliance() -> None:
    """Enforce Rule 1 (`<= 1,000` lines, zero semicolons), Rule 2 (`<= 8` args), and Rule 5 on Plan 040 files."""
    linter = _load_linter_module()
    cfg = linter.LinterConfig(
        paths=PLAN040_TARGET_FILES,
        max_lines=1000,
        hard_ceiling_lines=1200,
        max_args=8,
        max_nested_lines=40,
        strict=True,
    )
    report = linter.run_linter(cfg)
    assert report["all_passed"] is True, f"AST linter violations: {report['rows']}"


def test_hypothesis_040_results_json() -> None:
    """Verify H1 (`FineWeb` 900-step), H2 (`FineWeb` 3,600-step), and H3 (`ILSVRC ImageNet-1K`) in `.cache/hypothesis_experiments_040.json`."""
    json_path = ROOT / ".cache" / "hypothesis_experiments_040.json"
    if not json_path.exists():
        pytest.skip(f"{json_path} not yet generated (GPU runner still in progress).")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["plan_id"] == "EXPERIMENT_PLAN_040"
    hyps = payload["hypotheses"]

    # H1: 900-Step (7.37M Tokens) FineWeb-Edu Convex-Peak Traceless Born Ablation
    h1 = hyps["H1_fineweb_900step_convex_traceless_ablation"]
    assert h1["passed"] is True
    best_900_ce = min(
        h1["w25_convex_traceless_900_val_ce"],
        h1["w25_last_layer_meas_convex_traceless_900_val_ce"],
    )
    assert best_900_ce < 5.4734
    assert h1["w25_convex_traceless_900_val_ce"] < h1["w23_900_val_ce"]
    assert h1["w25_convex_traceless_900_val_ce"] < h1["w22_900_val_ce"]
    assert h1["w25_last_layer_meas_convex_traceless_900_val_ce"] < h1["w23_900_val_ce"]
    assert h1["w25_last_layer_meas_convex_traceless_900_val_ce"] < h1["w22_900_val_ce"]
    assert h1["gain_over_w21_nats"] > 0.0
    assert h1["gain_over_classical_nats"] > 0.40

    # H2: 3,600-Step (29.49M Tokens) FineWeb-Edu Convex-Peak Traceless Born Scaling
    h2 = hyps["H2_fineweb_3600step_convex_traceless_scaling"]
    assert h2["passed"] is True
    assert h2["w25_convex_traceless_3600_val_ce"] < 5.0211
    assert h2["w25_convex_traceless_3600_val_ce"] < h2["w23_3600_val_ce"]
    assert h2["w25_convex_traceless_3600_val_ce"] < h2["w22_3600_val_ce"]
    assert h2["w25_convex_traceless_3600_val_ce"] < h2["w21_3600_val_ce"]
    assert h2["w25_convex_traceless_3600_peak_vram_gb"] <= 8.0
    assert h2["gain_over_w21_nats"] > 0.0
    assert h2["gain_over_classical_nats"] > 0.45

    # H3: Genuine ILSVRC ImageNet-1K (1,281,167 Train / 50,000 Val / 100,000 Test) Ablation Study
    h3 = hyps["H3_ilsvrc_imagenet1k_quantum_building_block_ablations"]
    assert h3["passed"] is True
    assert h3["total_imagenet_train_rows"] == 1281167
    assert h3["total_imagenet_val_rows"] == 50000
    assert h3["total_imagenet_test_rows"] == 100000
    assert h3["wavelet_cholesky_contribution_ce_nats"] > 0.0
    assert h3["mach_zehnder_contribution_ce_nats"] > 0.0
    assert h3["kraus_bath_contribution_ce_nats"] > 0.0
    assert h3["convex_traceless_gain_over_naive_ce_nats"] >= 0.35
    assert h3["convex_traceless_gain_over_classical_ce_nats"] >= 0.0
