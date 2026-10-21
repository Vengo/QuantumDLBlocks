"""Unit tests for Experiment Plan 017: Zero-Downtime Benchmarking Server,
Declarative Run Model Cards, and Hot-Reloading Wave Discovery."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
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

from qdlblocks.benchmarks.model_card import (
    DynamicWaveRegistry,
    RunModelCard,
    build_model_card_from_manifest_entry,
    deterministic_run_color,
    render_model_card_markdown,
    save_model_card_atomic,
    update_model_card_telemetry,
    validate_model_card,
)
import monitor_colab_runs


def test_cli_to_model_card_parser_classical_and_quantum(tmp_path: Path) -> None:
    """Verify H2: CLI flag parsing, validation, and Markdown rendering across Wave-9 runs."""
    wave9_path = REPO_ROOT / "configs" / "wave9_fleet.json"
    assert wave9_path.exists(), "configs/wave9_fleet.json must exist"
    manifest = json.loads(wave9_path.read_text())
    runs = manifest["runs"]
    assert len(runs) >= 4

    cards_by_key = {}
    for entry in runs:
        card = build_model_card_from_manifest_entry(
            entry,
            wave_id=manifest.get("wave", "W9"),
            wave_description=manifest.get("description", ""),
            plan_id="EXPERIMENT_PLAN_016",
        )
        errors = validate_model_card(card)
        assert errors == [], f"Validation errors for {entry['key']}: {errors}"

        # Round-trip through RunModelCard dataclass & atomic save
        card_obj = RunModelCard.from_dict(card)
        saved_path = save_model_card_atomic(card_obj.to_dict(), model_cards_dir=tmp_path)
        assert saved_path.exists()
        reloaded = json.loads(saved_path.read_text())
        assert validate_model_card(reloaded) == []

        # Verify Markdown rendering includes all 5 required sections
        md = render_model_card_markdown(reloaded)
        assert "## 1. Hypothesis & Milestone Summary" in md
        assert "## 2. Architecture & Geometric Primitives" in md
        assert "## 3. Optimizer, Schedule & Batch Configuration" in md
        assert "## 4. Quantum-Geometric Observables" in md
        assert "## 5. Hardware & Reproducible CLI Command" in md
        assert entry["key"] in md

        cards_by_key[entry["key"]] = reloaded

    # 1. Verify Classical PR #360 Speedrun v2 flags
    c_pr360 = cards_by_key["w9_900m_classical_speedrun_v2_pr360"]
    assert c_pr360["architecture"]["family"] == "classical_modded"
    assert c_pr360["architecture"]["num_layers"] == 12
    assert c_pr360["architecture"]["embed_dim"] == 768
    assert c_pr360["architecture"]["num_heads"] == 12
    assert c_pr360["architecture"]["head_dim"] == 64
    assert c_pr360["architecture"]["num_value_embeddings"] == 3
    assert c_pr360["architecture"]["unet_skip_mode"] == "scalar"
    assert c_pr360["architecture"]["skip_attn_layers"] == [6]
    assert c_pr360["architecture"]["zero_half_rope"] is True
    assert c_pr360["architecture"]["sliding_window"] == 384
    assert c_pr360["optimization"]["optimizer_mode"] == "muon_adamw"
    assert c_pr360["optimization"]["schedule"] == "wsd"
    assert c_pr360["optimization"]["muon_lr"] == pytest.approx(0.025)
    assert c_pr360["optimization"]["muon_momentum_warmup_steps"] == 300
    assert c_pr360["data_and_batch"]["seq_len_train"] == 1024
    assert c_pr360["data_and_batch"]["seq_len_val"] == 512
    assert c_pr360["data_and_batch"]["batch_size"] == 64
    assert c_pr360["data_and_batch"]["grad_accum"] == 4
    assert c_pr360["data_and_batch"]["tokens_per_step"] == 262144

    # 2. Verify Classical Pure AdamW flags
    c_adamw = cards_by_key["w9_900m_classical_opt_ablation_pure_adamw"]
    assert c_adamw["optimization"]["optimizer_mode"] == "pure_adamw"
    assert c_adamw["optimization"]["muon_lr"] == pytest.approx(0.003)
    assert c_adamw["data_and_batch"]["tokens_per_step"] == 262144
    assert c_adamw["data_and_batch"]["seq_len_val"] == 512

    # 3. Verify Quantum 4d-Frame Expansion flags
    q_frame = cards_by_key["w9_900m_qborn_speedrun_v2_frame4d_synergy512"]
    assert q_frame["architecture"]["family"] == "quantum_born"
    assert q_frame["architecture"]["seed_dim"] == 512
    assert q_frame["architecture"]["frame_expansion"] == pytest.approx(4.0)
    assert q_frame["architecture"]["hidden_dim"] == 4096
    assert q_frame["architecture"]["purified_born_attn"] is True
    assert q_frame["architecture"]["dynamic_povm_beta"] is True
    assert q_frame["data_and_batch"]["seq_len_train"] == 512
    assert q_frame["data_and_batch"]["seq_len_val"] == 512
    assert q_frame["data_and_batch"]["batch_size"] == 128
    assert q_frame["data_and_batch"]["grad_accum"] == 1
    assert q_frame["data_and_batch"]["tokens_per_step"] == 65536

    # 4. Verify Quantum Asymmetric 1024D L=8 flags
    q_asym = cards_by_key["w9_900m_qborn_speedrun_v2_asym1024_L8"]
    assert q_asym["architecture"]["family"] == "quantum_born"
    assert q_asym["architecture"]["num_layers"] == 8
    assert q_asym["architecture"]["readout_mode"] == "asymmetric"
    assert q_asym["architecture"]["frame_expansion"] == pytest.approx(3.0)
    assert q_asym["data_and_batch"]["seq_len_val"] == 512

    # 5. Verify telemetry merge updates milestones & quantum observables
    updated = update_model_card_telemetry(
        q_frame,
        {
            "run_state": "RUNNING",
            "completed": False,
            "current_step": 763,
            "total_steps": 3433,
            "live_tokens_M": 200.0,
            "progress_pct": 22.2,
            "avg_tok_per_s": 54000.0,
            "wall_min": 61.7,
            "best_val_ce": 3.55,
            "best_val_ppl": 34.81,
            "best_top1": 0.362,
            "best_top5": 0.612,
            "params_M": 195.8,
            "eval_checkpoints": [
                {"step": 191, "tokens_M": 50.1, "val_ce": 4.12},
                {"step": 381, "tokens_M": 99.9, "val_ce": 3.78},
                {"step": 763, "tokens_M": 200.0, "val_ce": 3.55},
            ],
            "quantum_telemetry": {
                "seed_eff_rank": 388.4,
                "entanglement_nats": 2.91,
                "iso_err": 1.2e-4,
                "povm_beta": 0.96,
                "gamma_h": 1.04,
            },
        },
    )
    assert validate_model_card(updated) == []
    assert updated["architecture"]["params_M"] == pytest.approx(195.8)
    assert updated["telemetry"]["milestones_val_ce"]["50M"] == pytest.approx(4.12)
    assert updated["telemetry"]["milestones_val_ce"]["100M"] == pytest.approx(3.78)
    assert updated["telemetry"]["milestones_val_ce"]["200M"] == pytest.approx(3.55)
    assert updated["telemetry"]["quantum_observables"]["seed_eff_rank"] == pytest.approx(388.4)


def test_dynamic_wave_registry_hot_discovery_without_restart(tmp_path: Path) -> None:
    """Verify H1: DynamicWaveRegistry discovers newly added manifests & model cards without restart."""
    configs_dir = tmp_path / "configs"
    cards_dir = tmp_path / "model_cards"
    logs_dir = tmp_path / "live_logs"
    configs_dir.mkdir(parents=True)
    cards_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    # Seed with one Wave-9 manifest
    w9_manifest = {
        "wave": "W9",
        "description": "Wave 9 Test Fleet",
        "runs": [
            {
                "key": "w9_test_run_1",
                "session": "gpu_g4_w6_1",
                "hardware": "NVIDIA RTX PRO 6000 Blackwell (96GB)",
                "label": "W9 Test Run 1",
                "cli_args": "--backend cuda --architecture classical_modded --epochs 1 --num-shards 9 --batch-size 64 --grad-accum 4 --seq-len 1024",
                "remote_log": "/content/w9_test_run_1.log",
                "remote_json": "/content/w9_test_run_1_results.json",
            }
        ],
    }
    (configs_dir / "wave9_fleet.json").write_text(json.dumps(w9_manifest, indent=2))

    registry = DynamicWaveRegistry(
        configs_dir=configs_dir,
        model_cards_dir=cards_dir,
        live_logs_dir=logs_dir,
        historical_specs=[],
    )
    specs_1 = registry.discover_all_specs()
    assert len(specs_1) == 1
    assert specs_1[0]["key"] == "w9_test_run_1"
    assert specs_1[0]["color"] == deterministic_run_color("w9_test_run_1")

    # Dynamically drop a Wave-10 manifest without recreating registry
    w10_manifest = {
        "wave": "W10",
        "description": "Wave 10 Dynamic Zero-Downtime Fleet",
        "runs": [
            {
                "key": "w10_hot_added_run",
                "session": "gpu_g4_w6_2",
                "hardware": "NVIDIA RTX PRO 6000 Blackwell (96GB)",
                "label": "W10 Hot Added Run",
                "cli_args": "--backend cuda --architecture quantum_born --epochs 1 --num-shards 9 --batch-size 128 --grad-accum 4 --seq-len 512 --frame-expansion 4.0",
                "remote_log": "/content/w10_hot_added_run.log",
                "remote_json": "/content/w10_hot_added_run_results.json",
            }
        ],
    }
    (configs_dir / "wave10_fleet.json").write_text(json.dumps(w10_manifest, indent=2))

    specs_2 = registry.discover_all_specs()
    keys_2 = [s["key"] for s in specs_2]
    assert "w9_test_run_1" in keys_2
    assert "w10_hot_added_run" in keys_2
    wave_meta = registry.get_wave_metadata(specs_2)
    assert "W10" in wave_meta
    assert "Wave 10 Dynamic Zero-Downtime Fleet" in wave_meta["W10"]["description"]


def test_zero_downtime_http_reload_and_ingest_endpoints(tmp_path: Path) -> None:
    """Verify H1 & H3: Zero-downtime HTTP reload (<100ms), push ingest (<50ms), and concurrent GET (<25ms p95)."""
    configs_dir = tmp_path / "configs"
    cards_dir = tmp_path / "model_cards"
    logs_dir = tmp_path / "live_logs"
    configs_dir.mkdir(parents=True)
    cards_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    # Copy real wave9_fleet.json into temp configs_dir
    w9_text = (REPO_ROOT / "configs" / "wave9_fleet.json").read_text()
    w9_count = len(json.loads(w9_text)["runs"])
    (configs_dir / "wave9_fleet.json").write_text(w9_text)

    server, _state = monitor_colab_runs.create_benchmark_http_server(
        host="127.0.0.1",
        port=0,
        configs_dir=configs_dir,
        model_cards_dir=cards_dir,
        live_logs_dir=logs_dir,
        include_historical_specs=False,
        initial_skip_remote_poll=True,
    )
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        # 1. Initial GET /api/status (w9_count manifest runs + 5 reference baselines)
        with urllib.request.urlopen(f"{base_url}/api/status", timeout=2.0) as resp:
            assert resp.status == 200
            status_1 = json.loads(resp.read().decode("utf-8"))
        initial_runs_count = status_1["runs_count"]
        assert initial_runs_count >= w9_count
        uptime_1 = float(status_1["server_uptime_s"])

        # 2. Hot-register a brand-new Wave-10 manifest on disk and call POST /api/reload
        time.sleep(0.02)
        w10_manifest = {
            "wave": "W10",
            "description": "Wave 10 Zero-Downtime Verification",
            "runs": [
                {
                    "key": "w10_900m_zero_downtime_probe",
                    "session": "gpu_g4_w6_1",
                    "hardware": "NVIDIA RTX PRO 6000 Blackwell (96GB)",
                    "label": "W10 900M Zero-Downtime Probe",
                    "cli_args": "--backend cuda --architecture classical_modded --epochs 1 --num-shards 9 --batch-size 64 --grad-accum 4 --seq-len 1024 --num-value-embeddings 3",
                    "remote_log": "/content/w10_900m_zero_downtime_probe.log",
                    "remote_json": "/content/w10_900m_zero_downtime_probe_results.json",
                }
            ],
        }
        (configs_dir / "wave10_fleet.json").write_text(json.dumps(w10_manifest, indent=2))

        t0 = time.perf_counter()
        req = urllib.request.Request(f"{base_url}/api/reload", data=b"{}", method="POST")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            assert resp.status == 200
            reload_res = json.loads(resp.read().decode("utf-8"))
        reload_rtt_ms = (time.perf_counter() - t0) * 1000.0
        assert reload_res["ok"] is True
        assert reload_res["runs_count"] == initial_runs_count + 1
        assert reload_res["reload_ms"] < 100.0
        assert reload_rtt_ms < 100.0
        assert float(reload_res["uptime_s"]) > uptime_1

        # Verify GET /api/model-cards/<run_key> works in both JSON and Markdown formats
        with urllib.request.urlopen(
            f"{base_url}/api/model-cards/w10_900m_zero_downtime_probe", timeout=2.0
        ) as resp:
            assert resp.status == 200
            card_res = json.loads(resp.read().decode("utf-8"))
        assert card_res["run_key"] == "w10_900m_zero_downtime_probe"
        assert card_res["wave_id"] == "W10"
        assert validate_model_card(card_res) == []

        with urllib.request.urlopen(
            f"{base_url}/api/model-cards/w10_900m_zero_downtime_probe?format=md", timeout=2.0
        ) as resp:
            assert resp.status == 200
            md_text = resp.read().decode("utf-8")
        assert "Model Card:" in md_text
        assert "## 2. Architecture & Geometric Primitives" in md_text

        # 3. Test POST /api/ingest push update (< 50 ms)
        ingest_payload = {
            "run_key": "w10_900m_zero_downtime_probe",
            "log_append": (
                "Params: 171,622,144\n"
                "Step 191/3431 (Epoch 0.06 | 50.1M tok) | Train CE: 4.1500 | Speed: 112,000 tok/s\n"
                "[Epoch 0.06 | Step 191 | 50.1M tok | T=512] Val CE: 4.0812 | Val PPL: 59.21 | Top-1: 31.20% | Top-5: 55.40% | MedRank: 12 | IsoErr: 0.0001\n"
                "Step 382/3431 (Epoch 0.11 | 100.1M tok) | Train CE: 3.6800 | Speed: 113,500 tok/s\n"
                "[Epoch 0.11 | Step 382 | 100.1M tok | T=512] Val CE: 3.6120 | Val PPL: 37.04 | Top-1: 35.80% | Top-5: 60.10% | MedRank: 10 | IsoErr: 0.0001\n"
            ),
            "results_json": [
                {"step": 191, "epoch": 0.06, "tokens_M": 50.1, "val_ce": 4.0812, "val_ppl": 59.21, "top1_acc": 31.2, "top5_acc": 55.4},
                {"step": 382, "epoch": 0.11, "tokens_M": 100.1, "val_ce": 3.6120, "val_ppl": 37.04, "top1_acc": 35.8, "top5_acc": 60.1},
            ],
        }
        t_ing0 = time.perf_counter()
        ing_req = urllib.request.Request(
            f"{base_url}/api/ingest",
            data=json.dumps(ingest_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(ing_req, timeout=2.0) as resp:
            assert resp.status == 200
            ing_res = json.loads(resp.read().decode("utf-8"))
        ing_rtt_ms = (time.perf_counter() - t_ing0) * 1000.0
        assert ing_res["ok"] is True
        assert ing_res["ingest_ms"] < 50.0
        assert ing_rtt_ms < 50.0
        assert ing_res["best_val_ce"] == pytest.approx(3.6120)

        # Verify GET /api/status immediately reflects the ingested telemetry
        with urllib.request.urlopen(f"{base_url}/api/status", timeout=2.0) as resp:
            status_2 = json.loads(resp.read().decode("utf-8"))
        assert float(status_2["server_uptime_s"]) >= float(reload_res["uptime_s"])
        probe_entry = next(
            r for r in status_2["runs_catalog"] if r["key"] == "w10_900m_zero_downtime_probe"
        )
        assert probe_entry["best_val_ce"] == pytest.approx(3.6120)
        assert probe_entry["current_step"] == 382
        assert probe_entry["model_card"]["telemetry"]["milestones_val_ce"]["50M"] == pytest.approx(4.0812)
        assert probe_entry["model_card"]["telemetry"]["milestones_val_ce"]["100M"] == pytest.approx(3.6120)

        # 4. Concurrent read latency benchmark (50 concurrent GET requests, p95 < 25ms)
        def _fetch_one(idx: int) -> tuple[int, float]:
            ep = "/api/status" if idx % 2 == 0 else "/api/model-cards"
            t_start = time.perf_counter()
            with urllib.request.urlopen(f"{base_url}{ep}", timeout=2.0) as r:
                _ = r.read()
                code = r.status
            return code, (time.perf_counter() - t_start) * 1000.0

        with ThreadPoolExecutor(max_workers=10) as pool:
            results = list(pool.map(_fetch_one, range(50)))

        codes = [c for c, _ in results]
        latencies_ms = sorted(lat for _, lat in results)
        p95_ms = latencies_ms[int(0.95 * (len(latencies_ms) - 1))]
        assert all(c == 200 for c in codes)
        assert p95_ms < 25.0, f"Expected p95 < 25ms, got {p95_ms:.2f}ms"
    finally:
        server.shutdown()
        server.server_close()


def test_session_deduplicated_remote_polling_selection(tmp_path: Path) -> None:
    """Verify that when W6, W7, W8, and W9 share a Colab session name, only the latest active wave is polled."""
    configs_dir = tmp_path / "configs"
    cards_dir = tmp_path / "model_cards"
    logs_dir = tmp_path / "live_logs"
    configs_dir.mkdir(parents=True)
    cards_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    # Create W6, W7, W8, W9 specs sharing session gpu_g4_w6_1
    historical_specs = [
        {"key": "w6_run_a", "session": "gpu_g4_w6_1", "kind": "gpu", "hw": "Blackwell", "title": "W6 Run A", "wave": "W6"},
        {"key": "w7_run_a", "session": "gpu_g4_w6_1", "kind": "gpu", "hw": "Blackwell", "title": "W7 Run A", "wave": "W7"},
        {"key": "w8_run_a", "session": "gpu_g4_w6_1", "kind": "gpu", "hw": "Blackwell", "title": "W8 Run A", "wave": "W8"},
        {"key": "w9_run_a", "session": "gpu_g4_w6_1", "kind": "gpu", "hw": "Blackwell", "title": "W9 Run A", "wave": "W9"},
    ]

    registry = DynamicWaveRegistry(
        configs_dir=configs_dir,
        model_cards_dir=cards_dir,
        live_logs_dir=logs_dir,
        historical_specs=historical_specs,
    )
    specs = registry.discover_all_specs()
    targets = registry.select_session_poll_targets(
        specs, active_sessions={"gpu_g4_w6_1"}
    )
    assert set(targets.keys()) == {"gpu_g4_w6_1"}
    assert targets["gpu_g4_w6_1"]["key"] == "w9_run_a"
    assert registry.get_remote_poll_run_keys(specs, active_sessions={"gpu_g4_w6_1"}) == {"w9_run_a"}

    # Once w9_run_a, w8_run_a, w7_run_a, w6_run_a complete in local logs, none are selected for remote polling
    for rk in ("w9_run_a", "w8_run_a", "w7_run_a", "w6_run_a"):
        (logs_dir / f"{rk}.log").write_text("Step 3431/3431\nCOMPLETED in 48.2m\n")
    targets_after = registry.select_session_poll_targets(
        specs, active_sessions={"gpu_g4_w6_1"}
    )
    assert targets_after == {}
