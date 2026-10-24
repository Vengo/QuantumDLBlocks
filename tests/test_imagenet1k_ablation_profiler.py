"""CPU Unit Tests for `scripts/imagenet1k_quantum_block_ablation_profiler.py` (Plan 041).

Verifies:
1. CLI argument parsing and end-to-end execution across `imagenet1k`, `cifar100`, and `1000class`.
2. Exact module-group parameter budgets (`patch_embed`, `wavelet_cholesky_branch`,
   `mach_zehnder_gate`, `kraus_bath`, `dyadic_class_pyramid`, `total_params`).
3. Leave-one-out attribution deltas (`delta_val_ce_nats`, `delta_top1_acc_pct`,
   `delta_top5_acc_pct`, `delta_coarse_acc_pct`, `ce_reduction_per_100k_params`).
4. Synthetic fallback computation when cached `.cache/live_logs/*_results.json` files are absent.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "imagenet1k_quantum_block_ablation_profiler.py"

_SPEC = importlib.util.spec_from_file_location("imagenet1k_quantum_block_ablation_profiler", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PROFILER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PROFILER
_SPEC.loader.exec_module(_PROFILER)

EXPECTED_ARMS: tuple[str, ...] = (
    "full_convex_traceless_born",
    "full_dyadic_hierarchical_born",
    "ablate_no_mach_zehnder",
    "ablate_no_wavelet_cholesky",
    "ablate_no_kraus_bath",
    "w23_traceless_kboson",
    "naive_kboson_born",
    "classical_softmax",
)


def test_parse_args_and_cli_execution(tmp_path: Path) -> None:
    """Verify CLI parsing, execution on CPU, and JSON schema output."""
    out_json = tmp_path / "ablation_profile.json"
    argv = [
        "--dataset",
        "imagenet1k,cifar100",
        "--embed-dim",
        "128",
        "--num-layers",
        "3",
        "--batch-size",
        "8",
        "--warmup-steps",
        "1",
        "--bench-steps",
        "2",
        "--device",
        "cpu",
        "--json-out",
        str(out_json),
    ]
    cfg = _PROFILER.parse_args(argv)
    assert cfg.datasets == ("imagenet1k", "cifar100")
    assert cfg.embed_dim == 128
    assert cfg.num_layers == 3
    assert cfg.device == "cpu"

    rc = _PROFILER.main(argv)
    assert rc == 0
    assert out_json.exists()

    payload: dict[str, Any] = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["primary_dataset"] == "imagenet1k"
    assert "summary" in payload
    assert "datasets" in payload
    assert "arms" in payload
    assert "rows" in payload
    assert len(payload["rows"]) == 16  # 8 arms x 2 datasets


def test_module_group_parameter_counts_and_ablation_zeros(tmp_path: Path) -> None:
    """Verify exact module-group parameter budgets and zeroed ablated blocks across all 8 arms."""
    cfg = _PROFILER.AblationProfilerConfig(
        datasets=("imagenet1k", "cifar100"),
        embed_dim=128,
        num_layers=3,
        batch_size=4,
        warmup_steps=1,
        bench_steps=1,
        device="cpu",
        json_out=str(tmp_path / "params.json"),
    )
    report = _PROFILER.run_ablation_profiler(cfg)
    in_arms = report["datasets"]["imagenet1k"]["arms"]
    cf_arms = report["datasets"]["cifar100"]["arms"]

    for arm_key in EXPECTED_ARMS:
        assert arm_key in in_arms
        assert arm_key in cf_arms
        row = in_arms[arm_key]
        for key in (
            "patch_embed",
            "wavelet_cholesky_branch",
            "mach_zehnder_gate",
            "kraus_bath",
            "dyadic_class_pyramid",
            "total_params",
            "forward_step_ms",
            "delta_val_ce_nats",
            "delta_top1_acc_pct",
            "delta_top5_acc_pct",
            "delta_coarse_acc_pct",
            "ce_reduction_per_100k_params",
        ):
            assert key in row, f"Missing {key} in {arm_key}"
        assert row["max_prob_sum_abs_err"] < 1e-5
        assert row["forward_step_ms"] > 0.0

    full_row = in_arms["full_convex_traceless_born"]
    assert full_row["patch_embed"] == 14848
    assert full_row["mach_zehnder_gate"] == 12288
    assert full_row["kraus_bath"] == 131073
    assert full_row["wavelet_cholesky_branch"] > 0
    assert full_row["dyadic_class_pyramid"] > 0
    assert full_row["total_params"] == 755267

    assert in_arms["ablate_no_mach_zehnder"]["mach_zehnder_gate"] == 0
    assert in_arms["ablate_no_wavelet_cholesky"]["wavelet_cholesky_branch"] == 0
    assert in_arms["ablate_no_kraus_bath"]["kraus_bath"] == 0
    assert in_arms["classical_softmax"]["wavelet_cholesky_branch"] == 0
    assert in_arms["classical_softmax"]["mach_zehnder_gate"] == 0
    assert in_arms["classical_softmax"]["kraus_bath"] == 0
    assert in_arms["classical_softmax"]["dyadic_class_pyramid"] == 0


def test_leave_one_out_attribution_deltas_on_imagenet1k() -> None:
    """Verify cached `.cache/imagenet1k_quantum_block_ablation_profiler.json` attribution invariants."""
    artifact_path = ROOT / ".cache" / "imagenet1k_quantum_block_ablation_profiler.json"
    if not artifact_path.exists():
        _PROFILER.main(["--json-out", str(artifact_path)])
    assert artifact_path.exists()

    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    arms = data["datasets"]["imagenet1k"]["arms"]
    summary = data["summary"]

    # Reference arm has zero self-delta
    assert arms["full_convex_traceless_born"]["delta_val_ce_nats"] == 0.0
    assert arms["full_convex_traceless_born"]["delta_top1_acc_pct"] == 0.0

    # Every quantum building block contributes positively when ablated (delta_val_ce_nats > 0, delta_top1 > 0)
    for ablated_key in (
        "ablate_no_mach_zehnder",
        "ablate_no_wavelet_cholesky",
        "ablate_no_kraus_bath",
    ):
        row = arms[ablated_key]
        assert row["metrics_source"] == "cached_live_log"
        assert row["delta_val_ce_nats"] > 0.0
        assert row["delta_top1_acc_pct"] > 0.0
        assert row["delta_top5_acc_pct"] > 0.0
        assert row["delta_coarse_acc_pct"] > 0.0
        assert row["ce_reduction_per_100k_params"] > 0.0

    # Mach-Zehnder > Wavelet-Cholesky > Kraus Bath by leave-one-out CE impact on ImageNet-1K
    assert summary["ranked_building_blocks_by_ce_impact"] == [
        "mach_zehnder_gate",
        "wavelet_cholesky_branch",
        "kraus_bath",
    ]
    assert summary["convex_traceless_gain_over_naive_ce_nats"] > 1.0
    assert summary["convex_traceless_gain_over_classical_ce_nats"] > 0.10


def test_synthetic_fallback_when_cache_dir_empty(tmp_path: Path) -> None:
    """Verify synthetic fallback populates all 8 arms cleanly when no cached JSON logs exist."""
    empty_cache = tmp_path / "empty_live_logs"
    empty_cache.mkdir(parents=True, exist_ok=True)
    out_json = tmp_path / "fallback_out.json"

    cfg = _PROFILER.AblationProfilerConfig(
        datasets=("imagenet1k", "cifar100", "1000class"),
        embed_dim=64,
        num_layers=2,
        batch_size=4,
        warmup_steps=1,
        bench_steps=1,
        device="cpu",
        cache_dir=str(empty_cache),
        json_out=str(out_json),
    )
    report = _PROFILER.run_ablation_profiler(cfg)
    assert out_json.exists()
    for ds_key in ("imagenet1k", "cifar100", "1000class"):
        ds_arms = report["datasets"][ds_key]["arms"]
        for arm_key in EXPECTED_ARMS:
            assert ds_arms[arm_key]["metrics_source"] == "synthetic_fallback"
        assert ds_arms["ablate_no_mach_zehnder"]["delta_val_ce_nats"] > 0.0
        assert ds_arms["ablate_no_wavelet_cholesky"]["delta_val_ce_nats"] > 0.0
        assert ds_arms["ablate_no_kraus_bath"]["delta_val_ce_nats"] > 0.0
        assert ds_arms["naive_kboson_born"]["delta_val_ce_nats"] > 0.5
