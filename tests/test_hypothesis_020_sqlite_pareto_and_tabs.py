"""Unit tests for Plan 020: SQLite Telemetry & Model Card Database, Automatic Pareto-Frontier Obsolete Lifecycle, Multi-Tab WebUI, Automatic Colab VM Shutdown, and Wave-12 Fleet."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sqlite3
import sys
import threading
import time
import urllib.request

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from qdlblocks.benchmarks.db import (
    BenchmarkSQLiteDB,
    backfill_micro_verifications,
    classify_run_tier,
    compute_family_pareto_and_obsolete,
)
from qdlblocks.benchmarks.model_card import (
    build_model_card_from_manifest_entry,
    validate_model_card,
)
from monitor_colab_runs import (
    CONFIGS_DIR,
    DEFAULT_OBSOLETE_KEYS,
    LIVE_DIR,
    MODEL_CARDS_DIR,
    collect_snapshot,
    create_benchmark_http_server,
    render_html_dashboard,
)


def _http_get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5.0) as resp:
        assert resp.status == 200
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        assert resp.status == 200
        return json.loads(resp.read().decode("utf-8"))


def test_sqlite_db_schema_and_sync(tmp_path: Path) -> None:
    """Verify BenchmarkSQLiteDB creates all 6 WAL-mode tables, syncs main & micro runs, and round-trips losslessly."""
    db_file = tmp_path / "test_benchmarks.db"
    cards_dir = tmp_path / "model_cards"
    logs_dir = tmp_path / "live_logs"
    shutil.copytree(MODEL_CARDS_DIR, cards_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.tmp", ".*"))
    shutil.copytree(LIVE_DIR, logs_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.tmp", ".*"))

    db = BenchmarkSQLiteDB(db_path=db_file)

    # 1. Verify WAL mode and all 6 required tables exist
    with sqlite3.connect(str(db_file)) as conn:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert str(journal_mode).lower() == "wal"
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    expected_tables = {"runs", "run_evals", "run_steps", "colab_sessions", "session_queue", "ui_state"}
    assert expected_tables.issubset(tables), f"Missing SQLite tables: {expected_tables - tables}"

    # 2. Collect snapshot backed by this SQLite DB and verify round-trip
    snap = collect_snapshot(
        skip_remote_poll=True,
        configs_dir=CONFIGS_DIR,
        model_cards_dir=cards_dir,
        live_logs_dir=logs_dir,
        include_historical_specs=True,
        db_path=db_file,
    )
    loaded = db.load_snapshot_payload()

    assert len(loaded["runs_catalog"]) == len(snap["runs_catalog"])
    assert len(loaded["runs_catalog"]) >= 50
    assert len(loaded["micro_runs_catalog"]) >= 20
    assert len(loaded["colab_sessions"]) == 5
    assert len(loaded["session_queue"]) >= 5

    # Verify all main_benchmark runs have run_tier == "main_benchmark" and boolean pareto_frontier
    for r in loaded["runs_catalog"]:
        assert r["run_tier"] == "main_benchmark"
        assert isinstance(r.get("pareto_frontier"), bool)

    # Verify all micro_verification runs have run_tier == "micro_verification"
    for mr in loaded["micro_runs_catalog"]:
        assert mr["run_tier"] == "micro_verification"

    # Verify Wave-12 manifest Model Cards are valid and queued in session_queue
    w12_path = CONFIGS_DIR / "wave12_fleet.json"
    assert w12_path.exists()
    w12_data = json.loads(w12_path.read_text(encoding="utf-8"))
    assert len(w12_data["runs"]) == 5
    queued_w12 = [q for q in loaded["session_queue"] if q["wave_id"] == "W12"]
    assert len(queued_w12) == 5
    for r_spec in w12_data["runs"]:
        card = build_model_card_from_manifest_entry(
            r_spec,
            wave_id="W12",
            wave_description=w12_data["description"],
            plan_id="EXPERIMENT_PLAN_020",
        )
        assert validate_model_card(card) == []


def test_pareto_frontier_and_auto_obsolete_by_family(tmp_path: Path) -> None:
    """Verify independent 2D (wall_min, best_val_ce) Pareto frontiers and wave-superseded auto-obsolete with user override."""
    synthetic_catalog = [
        # Classical family:
        # c_w11_fast_best: W11, 3.3688 CE in 10.7m -> on Pareto frontier
        {
            "key": "c_w11_fast_best",
            "wave_id": "W11",
            "wave_num": 11.0,
            "arch_family": "classical_modded",
            "run_tier": "main_benchmark",
            "run_state": "COMPLETED",
            "completed": True,
            "best_val_ce": 3.3688,
            "wall_min": 10.7,
            "target_tokens_M": 900.0,
            "obsolete": False,
            "obsolete_reason": "",
        },
        # c_w11_ultra_fast: W11, 3.4106 CE in 6.7m -> faster than 10.7m, so also on Pareto frontier!
        {
            "key": "c_w11_ultra_fast",
            "wave_id": "W11",
            "wave_num": 11.0,
            "arch_family": "classical_modded",
            "run_tier": "main_benchmark",
            "run_state": "COMPLETED",
            "completed": True,
            "best_val_ce": 3.4106,
            "wall_min": 6.7,
            "target_tokens_M": 900.0,
            "obsolete": False,
            "obsolete_reason": "",
        },
        # c_w9_dominated: W9 (< W11), 3.4285 CE in 20.0m -> strictly dominated in BOTH CE and time by c_w11_fast_best
        {
            "key": "c_w9_dominated",
            "wave_id": "W9",
            "wave_num": 9.0,
            "arch_family": "classical_modded",
            "run_tier": "main_benchmark",
            "run_state": "COMPLETED",
            "completed": True,
            "best_val_ce": 3.4285,
            "wall_min": 20.0,
            "target_tokens_M": 900.0,
            "obsolete": False,
            "obsolete_reason": "",
        },
        # Quantum family (evaluated independently of Classical!):
        # q_w11_best: W11, 3.6787 CE in 27.6m -> on Quantum Pareto frontier (even though Classical has 3.3688 in 10.7m)
        {
            "key": "q_w11_best",
            "wave_id": "W11",
            "wave_num": 11.0,
            "arch_family": "quantum_born",
            "run_tier": "main_benchmark",
            "run_state": "COMPLETED",
            "completed": True,
            "best_val_ce": 3.6787,
            "wall_min": 27.6,
            "target_tokens_M": 900.0,
            "obsolete": False,
            "obsolete_reason": "",
        },
        # q_w10_dominated: W10 (< W11), 3.6926 CE in 42.3m -> dominated by q_w11_best
        {
            "key": "q_w10_dominated",
            "wave_id": "W10",
            "wave_num": 10.0,
            "arch_family": "quantum_born",
            "run_tier": "main_benchmark",
            "run_state": "COMPLETED",
            "completed": True,
            "best_val_ce": 3.6926,
            "wall_min": 42.3,
            "target_tokens_M": 900.0,
            "obsolete": False,
            "obsolete_reason": "",
        },
        # q_w8_fast_pareto: W8, 3.7484 CE in 18.5m -> faster than q_w11_best (18.5m < 27.6m), so on Quantum Pareto frontier!
        {
            "key": "q_w8_fast_pareto",
            "wave_id": "W8",
            "wave_num": 8.0,
            "arch_family": "quantum_born",
            "run_tier": "main_benchmark",
            "run_state": "COMPLETED",
            "completed": True,
            "best_val_ce": 3.7484,
            "wall_min": 18.5,
            "target_tokens_M": 900.0,
            "obsolete": False,
            "obsolete_reason": "",
        },
    ]

    out = compute_family_pareto_and_obsolete(
        synthetic_catalog,
        explicit_overrides=set(),
        default_obsolete_reasons={},
    )
    by_k = {r["key"]: r for r in out}

    # Check Pareto frontier membership
    assert by_k["c_w11_fast_best"]["pareto_frontier"] is True
    assert by_k["c_w11_ultra_fast"]["pareto_frontier"] is True
    assert by_k["c_w9_dominated"]["pareto_frontier"] is False

    assert by_k["q_w11_best"]["pareto_frontier"] is True
    assert by_k["q_w8_fast_pareto"]["pareto_frontier"] is True
    assert by_k["q_w10_dominated"]["pareto_frontier"] is False

    # Check automatic wave-superseded obsolete marking & canonical reason strings
    assert by_k["c_w11_fast_best"]["obsolete"] is False
    assert by_k["c_w9_dominated"]["obsolete"] is True
    assert (
        by_k["c_w9_dominated"]["obsolete_reason"]
        == "Not on ClassicalModdedGPT Pareto frontier: superseded by c_w11_fast_best (W11) with lower Val CE (3.3688 vs 3.4285) in less time (10.7m vs 20.0m)"
    )

    assert by_k["q_w11_best"]["obsolete"] is False
    assert by_k["q_w8_fast_pareto"]["obsolete"] is False
    assert by_k["q_w10_dominated"]["obsolete"] is True
    assert (
        by_k["q_w10_dominated"]["obsolete_reason"]
        == "Not on QuantumBornGPT Pareto frontier: superseded by q_w11_best (W11) with lower Val CE (3.6787 vs 3.6926) in less time (27.6m vs 42.3m)"
    )

    # Check explicit_user_override precedence: if user explicitly restores q_w10_dominated, it stays obsolete=False
    by_k["q_w10_dominated"]["obsolete"] = False
    by_k["q_w10_dominated"]["obsolete_reason"] = ""
    by_k["q_w10_dominated"]["explicit_user_override"] = True
    out_override = compute_family_pareto_and_obsolete(
        list(by_k.values()),
        explicit_overrides={"q_w10_dominated"},
        default_obsolete_reasons={},
    )
    by_k_ov = {r["key"]: r for r in out_override}
    assert by_k_ov["q_w10_dominated"]["obsolete"] is False
    assert by_k_ov["q_w10_dominated"]["obsolete_reason"] == ""


def test_micro_verification_separation_and_verify019_presence(tmp_path: Path) -> None:
    """Verify verify019_60step_* runs are classified as micro_verification, appear in micro_runs_catalog, and do not pollute main_benchmark."""
    db_file = tmp_path / "micro_test.db"
    cards_dir = tmp_path / "model_cards"
    logs_dir = tmp_path / "live_logs"
    shutil.copytree(MODEL_CARDS_DIR, cards_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.tmp", ".*"))
    shutil.copytree(LIVE_DIR, logs_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.tmp", ".*"))

    assert classify_run_tier("verify019_60step_leaderboard_exact", {"target_tokens_M": 1.97}) == "micro_verification"
    assert classify_run_tier("verify019_60step_wave9_v2", {"target_tokens_M": 1.97}) == "micro_verification"
    assert classify_run_tier("w11_900m_classical_leaderboard_exact_b262k_3429st", {"target_tokens_M": 900.0}) == "main_benchmark"

    snap = collect_snapshot(
        skip_remote_poll=True,
        configs_dir=CONFIGS_DIR,
        model_cards_dir=cards_dir,
        live_logs_dir=logs_dir,
        include_historical_specs=True,
        db_path=db_file,
    )

    main_keys = {r["key"] for r in snap["runs_catalog"]}
    micro_by_key = {r["key"]: r for r in snap["micro_runs_catalog"]}

    # 1. Zero verify* runs in main_benchmark catalog
    assert all(not k.startswith("verify") for k in main_keys)
    assert "verify019_60step_leaderboard_exact" not in main_keys
    assert "verify019_60step_wave9_v2" not in main_keys

    # 2. Both verify019 runs are present in micro_runs_catalog with exact expected Val CE
    assert "verify019_60step_leaderboard_exact" in micro_by_key
    assert "verify019_60step_wave9_v2" in micro_by_key

    v_exact = micro_by_key["verify019_60step_leaderboard_exact"]
    v_w9 = micro_by_key["verify019_60step_wave9_v2"]
    assert v_exact["run_tier"] == "micro_verification"
    assert v_exact["best_val_ce"] == pytest.approx(2.9254, abs=1e-4)
    assert v_exact["obsolete"] is False
    assert len(v_exact["evals"]) >= 2

    assert v_w9["run_tier"] == "micro_verification"
    assert v_w9["best_val_ce"] == pytest.approx(3.2974, abs=1e-4)
    assert v_w9["obsolete"] is False

    # 3. Verify HTML dashboard renders both the Main Leaderboard tab, Micro-Verifications tab, and Pareto Frontier controls
    html = render_html_dashboard(snap)
    assert "Pre-Launch Micro-Verifications & Hypothesis Tests" in html
    assert "verify019_60step_leaderboard_exact" in html
    assert "PARETO FRONTIER" in html
    assert "/api/colab-stop" in html


def test_http_api_dual_ingest_heartbeat_and_queue(tmp_path: Path) -> None:
    """Verify GET /api/snapshot, GET /api/micro-verifications, POST /api/ingest, POST /api/colab-heartbeat, POST /api/queue, and POST /api/obsolete."""
    db_file = tmp_path / "server_test.db"
    cards_dir = tmp_path / "model_cards"
    logs_dir = tmp_path / "live_logs"
    shutil.copytree(MODEL_CARDS_DIR, cards_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.tmp", ".*"))
    shutil.copytree(LIVE_DIR, logs_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.tmp", ".*"))

    server, _ = create_benchmark_http_server(
        host="127.0.0.1",
        port=0,
        configs_dir=CONFIGS_DIR,
        model_cards_dir=cards_dir,
        live_logs_dir=logs_dir,
        include_historical_specs=True,
        start_bg_poller=False,
        initial_skip_remote_poll=True,
        db_path=db_file,
    )
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()

    try:
        # 1. GET /api/snapshot & GET /api/micro-verifications
        snap = _http_get_json(f"{base_url}/api/snapshot")
        assert len(snap["runs_catalog"]) >= 50
        assert len(snap["micro_runs_catalog"]) >= 20
        assert "db_stats" in snap
        assert snap["db_stats"]["pareto_classical_count"] >= 1
        assert snap["db_stats"]["pareto_quantum_count"] >= 1

        micro_resp = _http_get_json(f"{base_url}/api/micro-verifications")
        assert micro_resp["ok"] is True
        micro_keys = {r["key"] for r in micro_resp["micro_runs"]}
        assert "verify019_60step_leaderboard_exact" in micro_keys

        # 2. POST /api/ingest for a new micro_verification run
        ing_res = _http_post_json(
            f"{base_url}/api/ingest",
            {
                "run_key": "verify020_synthetic_probe",
                "run_tier": "micro_verification",
                "arch_family": "quantum_born",
                "wave_id": "VERIFY",
                "wave_num": 20.0,
                "target_tokens_M": 1.31,
                "log_append": (
                    "Params: 45,000,000\n"
                    "[Epoch 1.00 | Step 40 | 1.3M tok | T=512] Val CE: 3.1234 | Val PPL: 22.72 | Top-1: 38.50% | Top-5: 62.10% | MedRank: 8 | IsoErr: 0.0000\n"
                    "COMPLETED in 0.25m\n"
                ),
                "results_json": [
                    {"step": 0, "epoch": 0.0, "tokens_M": 0.0, "val_ce": 10.825, "val_ppl": 50262.0, "top1_acc": 0.0, "top5_acc": 0.0, "elapsed_s": 0.1},
                    {"step": 40, "epoch": 1.0, "tokens_M": 1.31, "val_ce": 3.1234, "val_ppl": 22.72, "top1_acc": 38.5, "top5_acc": 62.1, "elapsed_s": 15.0},
                ],
            },
        )
        assert ing_res["ok"] is True
        assert ing_res["run_tier"] == "micro_verification"
        assert ing_res["best_val_ce"] == pytest.approx(3.1234)

        micro_resp_2 = _http_get_json(f"{base_url}/api/micro-verifications")
        probe_entry = next(r for r in micro_resp_2["micro_runs"] if r["key"] == "verify020_synthetic_probe")
        assert probe_entry["best_val_ce"] == pytest.approx(3.1234)

        # 3. POST /api/queue & GET /api/queue
        q_post = _http_post_json(
            f"{base_url}/api/queue",
            {
                "action": "enqueue",
                "item": {
                    "session": "gpu_g4_w6_1",
                    "run_key": "w12_900m_custom_queued_job",
                    "wave_id": "W12",
                    "arch_family": "classical_modded",
                    "hardware": "NVIDIA RTX PRO 6000 Blackwell (96GB)",
                    "short_label": "[W12] Custom Queued Job",
                    "cli_args": "--backend cuda --architecture classical_modded --leaderboard-exact-classical",
                    "status": "QUEUED",
                    "priority": 50,
                },
            },
        )
        assert q_post["ok"] is True
        assert any(q["run_key"] == "w12_900m_custom_queued_job" for q in q_post["queue"])

        # 4. POST /api/colab-heartbeat transitions matching QUEUED job to RUNNING
        hb_res = _http_post_json(
            f"{base_url}/api/colab-heartbeat",
            {
                "session": "gpu_g4_w6_1",
                "device": "NVIDIA RTX PRO 6000 Blackwell",
                "gpu_util_pct": 98.0,
                "vram_used_gib": 64.2,
                "vram_total_gib": 95.6,
                "ps_alive": True,
                "ps_cmd": "python3 examples/11_fineweb_100m_multiepoch_gpu_tpu.py --out-json /content/w12_900m_custom_queued_job.json",
                "active_run_key": "w12_900m_custom_queued_job",
                "wave_id": "W12",
                "run_state": "RUNNING",
            },
        )
        assert hb_res["ok"] is True
        q_get = _http_get_json(f"{base_url}/api/queue")
        custom_q = next(q for q in q_get["queue"] if q["run_key"] == "w12_900m_custom_queued_job")
        assert custom_q["status"] == "RUNNING"
    finally:
        server.shutdown()
        server.server_close()
