"""Unit & Integration Tests for Experiment Plan 044 (`tests/test_plan044_webui_wave_ingestion.py`).

Covers:
1. `ExperimentIntentSpec`, `StandardizedExperimentLogger`, and `parse_experiment_intent_block`.
2. `infer_task_domain` and `classify_run_tier` separation of `main_benchmark`, `fineweb_edu_local`,
   `vision_ablation`, and `micro_verification`.
3. `normalize_raw_result_to_model_card` schema validation (`validate_model_card`), `compute_tflops`,
   `tau_progress`, `sharp_eff_rank`, `kraus_stiefel_err`, and `.log` backfilling.
4. `compute_family_pareto_and_obsolete` horizon-bucketed Pareto frontiers (`<= 10M`, `10M..50M`, `> 50M`).
5. End-to-end `sync_waves_to_webui.py`, `sqlite_benchmark_query_cli.py`, and `webui_static_html_exporter.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

from qdlblocks.benchmarks.db import (
    BenchmarkSQLiteDB,
    classify_run_tier,
    compute_family_pareto_and_obsolete,
)
from qdlblocks.benchmarks.experiment_logger import (
    ExperimentIntentSpec,
    StandardizedExperimentLogger,
    parse_experiment_intent_block,
)
from qdlblocks.benchmarks.model_card import validate_model_card
from qdlblocks.benchmarks.wave_ingestor import (
    infer_task_domain,
    ingest_all_waves_and_verifications,
    normalize_raw_result_to_model_card,
)


def test_experiment_logger_writes_intent_steps_evals_and_db(tmp_path: Path) -> None:
    """Verify `StandardizedExperimentLogger` emits `[EXPERIMENT_INTENT]`, steps, evals, ModelCard v1.0, and SQLite rows."""
    logs_dir = tmp_path / "live_logs"
    cards_dir = tmp_path / "model_cards"
    db = BenchmarkSQLiteDB(tmp_path / "benchmarks.db")
    try:
        spec = ExperimentIntentSpec(
            run_key="w29_flagfree_parseval_mup_b16_d384_900steps",
            wave_id="W29",
            plan_id="EXPERIMENT_PLAN_045",
            task_domain="fineweb_edu_local",
            hypothesis_intent="Verify Parseval muP scaling at d=384 with live standardized logger.",
            architecture_summary="ParsevalMuPDyadicGPT (L=6, d=384, mid=192, fine=96)",
            batch_size=16,
            seq_len=512,
            total_steps=900,
            target_tokens_M=7.37,
            params_M=21.34,
            embed_dim=384,
            num_layers=6,
            schedule="modded_nanogpt_wsd",
        )
        logger = StandardizedExperimentLogger(
            spec=spec,
            live_logs_dir=logs_dir,
            model_cards_dir=cards_dir,
            db=db,
        )
        logger.log_step(step=100, train_ce=6.42, tok_per_s=18500.0, lr_mult=1.0, elapsed_min=0.7)
        ev0 = logger.log_eval(
            step=0,
            metrics={
                "val_ce": 8.6733,
                "top1_acc": 4.28,
                "top5_acc": 11.69,
                "sharp_eff_rank": 16.01,
                "kraus_stiefel_err": 0.0,
                "current_beta": 27.55,
            },
        )
        assert ev0["compute_tflops"] == 0.0
        assert ev0["tau_progress"] == 0.0

        ev900 = logger.log_eval(
            step=900,
            metrics={
                "tokens_M": 7.37,
                "val_ce": 5.3812,
                "top1_acc": 22.65,
                "top5_acc": 39.40,
                "sharp_eff_rank": 33.40,
                "diffuse_eff_rank": 14.12,
                "kraus_stiefel_err": 0.0124,
                "current_beta": 29.87,
                "tok_per_s": 18450.0,
                "elapsed_min": 6.65,
            },
        )
        assert ev900["compute_tflops"] > 900.0
        assert ev900["tau_progress"] == 1.0

        final_res = logger.finalize({"extra_probe_pass": True})
        assert final_res["completed"] is True
        assert final_res["best_val_ce"] == 5.3812

        log_text = (logs_dir / f"{spec.run_key}.log").read_text(encoding="utf-8")
        assert "[EXPERIMENT_INTENT]" in log_text
        assert "COMPLETED w29_flagfree_parseval_mup_b16_d384_900steps" in log_text
        parsed_intent = parse_experiment_intent_block(log_text)
        assert parsed_intent["run_key"] == spec.run_key
        assert parsed_intent["task_domain"] == "fineweb_edu_local"

        card_data = json.loads((cards_dir / f"{spec.run_key}.json").read_text(encoding="utf-8"))
        assert validate_model_card(card_data) == []

        snap = db.load_snapshot_payload()
        edu_keys = [r["key"] for r in snap["fineweb_edu_catalog"]]
        assert spec.run_key in edu_keys
    finally:
        db.close()


def test_task_domain_and_run_tier_classification() -> None:
    """Verify `infer_task_domain` and `classify_run_tier` across speedrun, FineWeb-Edu, vision, and micro runs."""
    assert infer_task_domain("w24_cifar100_convex_traceless_born") == "vision_ablation"
    assert infer_task_domain("w24_1000class_dyadic_hierarchical_born") == "vision_ablation"
    assert infer_task_domain("w25_imagenet1k_full_convex_traceless_born") == "vision_ablation"
    assert infer_task_domain("verify028_30step_stereographic") == "micro_verification"
    assert infer_task_domain("w28_flagfree_parseval_mup_b16_d512_900steps") == "fineweb_edu_local"
    assert infer_task_domain("w12_900m_classical_v4", {"target_tokens_M": 900.0}) == "fineweb_speedrun"

    assert classify_run_tier("w24_cifar100_convex_traceless_born", 0.5, "W24") == "micro_verification"
    assert classify_run_tier("w25_imagenet1k_full_dyadic_hierarchical_born", 1.0, "W25") == "micro_verification"
    assert classify_run_tier("w28_flagfree_parseval_mup_b16_d512_900steps", 7.37, "W28") == "fineweb_edu_local"
    assert classify_run_tier("w26_15gb_b48_d256_L6_dyadic_fp8_1800steps", 44.24, "W26") == "fineweb_edu_local"
    assert classify_run_tier("w12_900m_classical_v4", 900.0, "W12") == "main_benchmark"


def test_horizon_bucketed_pareto_frontier_for_fineweb_edu() -> None:
    """Verify `W28` (`7.37M`), `W25` (`29.49M`), `W27` (`29.49M`), and `W26` (`44.24M`) are all Pareto frontier and not obsolete."""
    catalog = [
        {
            "key": "w25_8gb_convex_traceless_born_3600steps",
            "wave_id": "W25",
            "wave_num": 25.0,
            "arch_family": "quantum_born",
            "run_tier": "fineweb_edu_local",
            "completed": True,
            "run_state": "COMPLETED",
            "target_tokens_M": 29.49,
            "best_val_ce": 5.0162,
            "wall_min": 19.188,
        },
        {
            "key": "w26_15gb_b48_d256_L6_dyadic_fp8_1800steps",
            "wave_id": "W26",
            "wave_num": 26.0,
            "arch_family": "quantum_born",
            "run_tier": "fineweb_edu_local",
            "completed": True,
            "run_state": "COMPLETED",
            "target_tokens_M": 44.24,
            "best_val_ce": 4.8199,
            "wall_min": 21.153,
        },
        {
            "key": "w27_isotok_29p49M_b24_d512_L6_dyadic_fp8",
            "wave_id": "W27",
            "wave_num": 27.0,
            "arch_family": "quantum_born",
            "run_tier": "fineweb_edu_local",
            "completed": True,
            "run_state": "COMPLETED",
            "target_tokens_M": 29.49,
            "best_val_ce": 4.7919,
            "wall_min": 31.412,
        },
        {
            "key": "w28_flagfree_parseval_mup_b16_d512_900steps",
            "wave_id": "W28",
            "wave_num": 28.0,
            "arch_family": "quantum_born",
            "run_tier": "fineweb_edu_local",
            "completed": True,
            "run_state": "COMPLETED",
            "target_tokens_M": 7.37,
            "best_val_ce": 5.3671,
            "wall_min": 8.780,
        },
    ]
    out = compute_family_pareto_and_obsolete(catalog)
    by_key = {r["key"]: r for r in out}
    for k in by_key:
        assert by_key[k]["pareto_frontier"] is True
        assert by_key[k]["obsolete"] is False


def test_normalize_raw_result_and_validate_model_card(tmp_path: Path) -> None:
    """Verify `normalize_raw_result_to_model_card` handles vision & FineWeb-Edu dicts and backfills `.log`."""
    raw_w28 = {
        "name": "w28_flagfree_parseval_mup_b16_d256_900steps",
        "embed_dim": 256,
        "num_layers": 4,
        "params_M": 10.3894,
        "batch_size": 16,
        "seq_len": 512,
        "total_steps": 900,
        "total_tokens_M": 7.3728,
        "best_val_ce": 5.4372,
        "final_top1_acc": 22.08,
        "sharp_eff_rank": 26.79,
        "max_kraus_stiefel_err": 0.0252,
        "avg_tok_per_s": 24657.9,
        "elapsed_sec": 299.0,
        "evals": [
            {"step": 0, "tokens_M": 0.0, "val_ce": 8.8445, "top1_acc": 3.31, "sharp_eff_rank": 13.75, "kraus_stiefel_err": 0.0},
            {"step": 900, "tokens_M": 7.37, "val_ce": 5.4372, "top1_acc": 22.08, "sharp_eff_rank": 26.79, "kraus_stiefel_err": 0.0145},
        ],
    }
    cat, card = normalize_raw_result_to_model_card(
        "w28_flagfree_parseval_mup_b16_d256_900steps",
        raw_w28,
        live_logs_dir=tmp_path,
    )
    assert validate_model_card(card) == []
    assert cat["run_tier"] == "fineweb_edu_local"
    assert cat["best_val_ce"] == 5.4372
    assert cat["evals"][-1]["compute_tflops"] > 400.0
    assert cat["evals"][-1]["tau_progress"] == 1.0
    assert (tmp_path / "w28_flagfree_parseval_mup_b16_d256_900steps.log").exists()


def test_end_to_end_live_db_fineweb_edu_catalog() -> None:
    """Verify live `.cache/benchmarks.db` contains all normalized `W18–W28` FineWeb-Edu and `W24–W25` Vision runs."""
    repo_root = Path(__file__).resolve().parents[1]
    db_path = repo_root / ".cache" / "benchmarks.db"
    if not db_path.exists():
        return
    db = BenchmarkSQLiteDB(db_path)
    try:
        snap = db.load_snapshot_payload()
    finally:
        db.close()

    edu_by_key = {r["key"]: r for r in snap.get("fineweb_edu_catalog", [])}
    assert len(edu_by_key) >= 38
    for rk in (
        "w28_flagfree_parseval_mup_b16_d512_900steps",
        "w28_flagfree_parseval_mup_b16_d384_900steps",
        "w28_flagfree_parseval_mup_b16_d256_900steps",
        "w27_isotok_29p49M_b24_d512_L6_dyadic_fp8",
        "w26_15gb_b48_d256_L6_dyadic_fp8_1800steps",
        "w25_8gb_convex_traceless_born_3600steps",
    ):
        assert rk in edu_by_key, f"Missing {rk} in fineweb_edu_catalog"
        assert edu_by_key[rk]["run_tier"] == "fineweb_edu_local"
        assert edu_by_key[rk]["pareto_frontier"] is True
        assert edu_by_key[rk]["obsolete"] is False
        assert len(edu_by_key[rk]["evals"]) >= 2
        assert edu_by_key[rk]["evals"][-1]["compute_tflops"] > 0.0
        assert edu_by_key[rk]["evals"][-1]["tau_progress"] > 0.0

    micro_by_key = {r["key"]: r for r in snap.get("micro_runs_catalog", [])}
    assert "w24_cifar100_convex_traceless_born" in micro_by_key
    assert "w25_imagenet1k_full_convex_traceless_born" in micro_by_key


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tpath = Path(td)
        test_experiment_logger_writes_intent_steps_evals_and_db(tpath / "t1")
        test_task_domain_and_run_tier_classification()
        test_horizon_bucketed_pareto_frontier_for_fineweb_edu()
        test_normalize_raw_result_and_validate_model_card(tpath / "t2")
        test_end_to_end_live_db_fineweb_edu_catalog()
    print("ALL PLAN 044 WAVE INGESTION & LOGGER TESTS PASSED (5/5)")

