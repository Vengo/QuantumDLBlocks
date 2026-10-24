"""Hypothesis & AST Compliance Tests for Experiment Plan 039 (`Wave 24`):
Fast-Loop Multi-Class Vision Benchmarks (`CIFAR-100` $K=100$ & `1,000`-Class $K=1000$)
and `FineWeb-Edu` ($V=50,257$) Transfer Verification.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]

PLAN039_TARGET_FILES = (
    "src/qdlblocks/models/wavelet_cholesky_vision.py",
    "src/qdlblocks/models/wavelet_cholesky_2pow16_gpt.py",
    "scripts/last_layer_measurement_adjoint_bench.py",
    "scripts/traceless_observable_rank_bucket_probe.py",
    "examples/49_verify_plan039_cifar100_and_1000class_fast_loop.py",
    "tests/test_plan039_cifar100_and_1000class_wavelet_cholesky.py",
    "tests/test_hypothesis_039_cifar100_and_1000class_fast_loop.py",
)


def _load_linter_module():
    linter_path = ROOT / "scripts" / "ast_signature_and_length_linter.py"
    spec = importlib.util.spec_from_file_location("ast_signature_and_length_linter", linter_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_plan039_ast_best_practices_compliance() -> None:
    """Enforce Rule 1 (`<= 1,000` lines, zero semicolons), Rule 2 (`<= 8` args), and Rule 5 on Plan 039 files."""
    linter = _load_linter_module()
    cfg = linter.LinterConfig(
        paths=PLAN039_TARGET_FILES,
        max_lines=1000,
        hard_ceiling_lines=1200,
        max_args=8,
        max_nested_lines=40,
        strict=True,
    )
    report = linter.run_linter(cfg)
    assert report["all_passed"] is True, f"AST linter violations: {report['rows']}"


def test_hypothesis_039_results_json() -> None:
    """Verify H1 (`CIFAR-100`), H2 (`1,000`-Class), and H3 (`FineWeb-Edu` 300-Step) in `.cache/hypothesis_experiments_039.json`."""
    json_path = ROOT / ".cache" / "hypothesis_experiments_039.json"
    if not json_path.exists():
        pytest.skip(f"{json_path} not yet generated (GPU runner still in progress).")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["plan_id"] == "EXPERIMENT_PLAN_039"
    hyps = payload["hypotheses"]

    # H1: CIFAR-100 (K=100, K_coarse=20, N=128)
    h1 = hyps["H1_cifar100_k100_paradox_and_resolution"]
    assert h1["passed"] is True
    assert h1["naive_vs_classical_ce_penalty_nats"] >= 0.20
    assert h1["convex_gain_over_naive_ce_nats"] >= 0.25
    assert h1["hierarchical_gain_over_naive_ce_nats"] >= 0.25
    assert h1["convex_traceless_born_test_ce"] <= h1["classical_softmax_test_ce"]
    assert h1["dyadic_hierarchical_born_test_ce"] <= h1["classical_softmax_test_ce"]
    assert max(h1["convex_traceless_born_top1_acc"], h1["dyadic_hierarchical_born_top1_acc"]) >= h1["classical_softmax_top1_acc"]

    # H2: ImageNet-1K-Scale 1,000-Class (K=1000, K_coarse=32, N=1024)
    h2 = hyps["H2_imagenet1k_scale_k1000_paradox_and_resolution"]
    assert h2["passed"] is True
    assert h2["naive_vs_classical_ce_penalty_nats"] >= 0.40
    assert h2["convex_gain_over_naive_ce_nats"] >= 0.50
    assert h2["hierarchical_gain_over_naive_ce_nats"] >= 0.50
    assert h2["convex_traceless_born_test_ce"] <= h2["classical_softmax_test_ce"]
    assert h2["dyadic_hierarchical_born_test_ce"] <= h2["classical_softmax_test_ce"]
    assert max(h2["convex_traceless_born_top1_acc"], h2["dyadic_hierarchical_born_top1_acc"]) >= h2["classical_softmax_top1_acc"]

    # H3: FineWeb-Edu 50,257-Class Transfer Probe (300 steps)
    h3 = hyps["H3_fineweb_50257_convex_traceless_transfer"]
    assert h3["passed"] is True
    assert h3["autograd_gate_passed"] is True
    assert h3["forward_loss_abs_diff"] < 1e-5
    assert h3["max_grad_abs_diff"] < 1e-4
    assert h3["w24_step300_val_ce"] < h3["w23_step300_val_ce"]
    assert h3["w24_step300_val_ce"] < h3["w22_step300_val_ce"]
    assert h3["w24_step300_val_ce"] < h3["classical_53p61M_step300_val_ce"]
    assert h3["w24_peak_vram_gb"] <= 8.0
