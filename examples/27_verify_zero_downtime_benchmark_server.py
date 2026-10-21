#!/usr/bin/env python3
"""Controlled Verification Script for Experiment Plan 017:
Zero-Downtime Permanent Benchmarking Server, Declarative Run Model Cards,
Hot-Reloading Wave Discovery, & Home-LAN / GCP Deployment Architecture.

Executes:
1. Full backfill of `.cache/model_cards/<run_key>.json` across Waves 1-9 (`FLEET_SPECS` + `configs/wave*_fleet.json` + `.cache/live_logs/*.log`).
2. Quantitative verification of Hypothesis H1 (Zero-Downtime Hot Registration < 100 ms, 0 process restarts, deterministic color & wave grouping).
3. Quantitative verification of Hypothesis H2 (>= 35 valid backfilled Model Cards with 0 schema errors, 100% CLI & telemetry round-trip accuracy, 5-section Markdown rendering).
4. Quantitative verification of Hypothesis H3 (POST /api/ingest push latency < 50 ms, 50 concurrent GET /api/status & /api/model-cards requests with 100% HTTP 200 and p95 < 25 ms).
5. Idempotent activation/hot-reload of the permanent server on `0.0.0.0:8765` via `scripts/ensure_benchmark_server.sh`.
6. Persists structured verification results to `.cache/hypothesis_experiments_017.json`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from qdlblocks.benchmarks.model_card import (
    DynamicWaveRegistry,
    build_model_card_from_manifest_entry,
    deterministic_run_color,
    render_model_card_markdown,
    update_model_card_telemetry,
    validate_model_card,
)
import monitor_colab_runs


def main() -> None:
    print("=" * 88)
    print("Experiment Plan 017 Verification: Zero-Downtime Server & Declarative Run Model Cards")
    print("=" * 88)

    model_cards_dir = REPO_ROOT / ".cache" / "model_cards"
    configs_dir = REPO_ROOT / "configs"
    live_logs_dir = REPO_ROOT / ".cache" / "live_logs"

    # Remove any leftover temporary probe cards before backfilling real fleet cards
    for stale_probe in model_cards_dir.glob("w10_*probe*.json"):
        stale_probe.unlink(missing_ok=True)

    # Step 1: Backfill all historical + Wave 6-9 Model Cards via collect_snapshot & colab_fleet register-cards
    t_bf0 = time.perf_counter()
    snap = monitor_colab_runs.collect_snapshot(
        skip_remote_poll=True,
        configs_dir=configs_dir,
        model_cards_dir=model_cards_dir,
        live_logs_dir=live_logs_dir,
        include_historical_specs=True,
    )
    reg_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "colab_fleet.py"),
        "register-cards",
        "--manifest",
        str(configs_dir / "wave9_fleet.json"),
    ]
    subprocess.run(reg_cmd, check=True, capture_output=True, text=True)
    backfill_ms = (time.perf_counter() - t_bf0) * 1000.0

    # Validate all generated model cards on disk
    card_files = sorted(model_cards_dir.glob("*.json"))
    validation_errors_by_key: dict[str, list[str]] = {}
    waves_discovered: set[str] = set()
    markdown_sections_ok_count = 0

    for cf in card_files:
        card = json.loads(cf.read_text(encoding="utf-8"))
        rkey = str(card.get("run_key", cf.stem))
        errs = validate_model_card(card)
        if errs:
            validation_errors_by_key[rkey] = errs
        waves_discovered.add(str(card.get("wave_id", "")))
        md = render_model_card_markdown(card)
        if all(
            sec in md
            for sec in (
                "## 1. Hypothesis & Milestone Summary",
                "## 2. Architecture & Geometric Primitives",
                "## 3. Optimizer, Schedule & Batch Configuration",
                "## 4. Quantum-Geometric Observables",
                "## 5. Hardware & Reproducible CLI Command",
            )
        ):
            markdown_sections_ok_count += 1

    print(
        f"[H2 Backfill] Generated {len(card_files)} Model Cards across waves {sorted(waves_discovered)} "
        f"in {backfill_ms:.1f} ms | Schema Errors: {len(validation_errors_by_key)}"
    )

    # Verify round-trip CLI flag accuracy for Classical PR #360 and Quantum 4d-Frame runs
    w9_manifest = json.loads((configs_dir / "wave9_fleet.json").read_text(encoding="utf-8"))
    w9_entries = {r["key"]: r for r in w9_manifest["runs"]}

    c_card = build_model_card_from_manifest_entry(
        w9_entries["w9_900m_classical_speedrun_v2_pr360"],
        wave_id="W9",
        wave_description=w9_manifest["description"],
        plan_id="EXPERIMENT_PLAN_016",
    )
    q_card = build_model_card_from_manifest_entry(
        {
            "key": "w9_quantum_roundtrip_check",
            "session": "gpu_g4_w6_3",
            "hardware": "NVIDIA RTX PRO 6000 Blackwell (96GB)",
            "label": "Wave-9 QuantumBornGPT Round-Trip Check",
            "cli_args": (
                "--backend cuda --architecture quantum_born --epochs 1 --num-shards 9 "
                "--schedule wsd --seed-dim 512 --overcomplete-two-beam --frame-expansion 4.0 "
                "--purified-born-attn --dynamic-povm-beta --batch-size 128 --grad-accum 4 --seq-len 512"
            ),
        },
        wave_id="W9",
        wave_description=w9_manifest["description"],
        plan_id="EXPERIMENT_PLAN_016",
    )

    classical_checks = {
        "num_layers_12": c_card["architecture"]["num_layers"] == 12,
        "embed_dim_768": c_card["architecture"]["embed_dim"] == 768,
        "num_value_embeddings_3": c_card["architecture"]["num_value_embeddings"] == 3,
        "unet_skip_scalar": c_card["architecture"]["unet_skip_mode"] == "scalar",
        "skip_attn_layers_6": c_card["architecture"]["skip_attn_layers"] == [6],
        "zero_half_rope": c_card["architecture"]["zero_half_rope"] is True,
        "sliding_window_384": c_card["architecture"]["sliding_window"] == 384,
        "grad_accum_4": c_card["data_and_batch"]["grad_accum"] == 4,
        "seq_len_train_1024": c_card["data_and_batch"]["seq_len_train"] == 1024,
        "seq_len_val_512": c_card["data_and_batch"]["seq_len_val"] == 512,
        "tokens_per_step_262144": c_card["data_and_batch"]["tokens_per_step"] == 262144,
        "optimizer_muon_adamw": c_card["optimization"]["optimizer_mode"] == "muon_adamw",
    }
    quantum_checks = {
        "frame_expansion_4": abs(float(q_card["architecture"]["frame_expansion"]) - 4.0) < 1e-6,
        "seed_dim_512": q_card["architecture"]["seed_dim"] == 512,
        "purified_born_attn": q_card["architecture"]["purified_born_attn"] is True,
        "dynamic_povm_beta": q_card["architecture"]["dynamic_povm_beta"] is True,
        "tokens_per_step_262144": q_card["data_and_batch"]["tokens_per_step"] == 262144,
        "seq_len_val_512": q_card["data_and_batch"]["seq_len_val"] == 512,
    }
    h2_pass = (
        len(card_files) >= 35
        and len(validation_errors_by_key) == 0
        and markdown_sections_ok_count == len(card_files)
        and all(classical_checks.values())
        and all(quantum_checks.values())
    )

    # Step 2: Controlled H1 & H3 Benchmark on an isolated HTTP Server instance
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        tmp_configs = tmp_dir / "configs"
        tmp_cards = tmp_dir / "model_cards"
        tmp_logs = tmp_dir / "live_logs"
        shutil.copytree(configs_dir, tmp_configs)
        shutil.copytree(model_cards_dir, tmp_cards)
        tmp_logs.mkdir(parents=True, exist_ok=True)

        server, _state = monitor_colab_runs.create_benchmark_http_server(
            host="127.0.0.1",
            port=0,
            configs_dir=tmp_configs,
            model_cards_dir=tmp_cards,
            live_logs_dir=tmp_logs,
            include_historical_specs=True,
            initial_skip_remote_poll=True,
        )
        port = server.server_address[1]
        base_url = f"http://127.0.0.1:{port}"
        srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
        srv_thread.start()

        try:
            with urllib.request.urlopen(f"{base_url}/api/status", timeout=2.0) as r:
                s0 = json.loads(r.read().decode("utf-8"))
            uptime_before = float(s0["server_uptime_s"])
            count_before = int(s0["runs_count"])

            time.sleep(0.02)
            # Write a new Wave-10 manifest without specifying color to test deterministic color & hot reload
            w10_key = "w10_900m_hot_discovery_benchmark"
            w10_manifest = {
                "wave": "W10",
                "description": "Wave 10 Zero-Downtime Hot Discovery Benchmark",
                "runs": [
                    {
                        "key": w10_key,
                        "session": "gpu_g4_w6_1",
                        "hardware": "NVIDIA RTX PRO 6000 Blackwell (96GB)",
                        "label": "W10 900M Zero-Downtime Hot Discovery Probe",
                        "cli_args": (
                            "--backend cuda --architecture classical_modded --epochs 1 --num-shards 9 "
                            "--schedule wsd --num-layers 12 --embed-dim 768 --num-heads 12 --seq-len 1024 "
                            "--batch-size 64 --grad-accum 4 --num-value-embeddings 3 --unet-skip-mode scalar"
                        ),
                        "remote_log": f"/content/{w10_key}.log",
                        "remote_json": f"/content/{w10_key}_results.json",
                    }
                ],
            }
            (tmp_configs / "wave10_fleet.json").write_text(json.dumps(w10_manifest, indent=2), encoding="utf-8")

            t_rel0 = time.perf_counter()
            req_rel = urllib.request.Request(f"{base_url}/api/reload", data=b"{}", method="POST")
            with urllib.request.urlopen(req_rel, timeout=2.0) as r:
                rel_resp = json.loads(r.read().decode("utf-8"))
            reload_rtt_ms = (time.perf_counter() - t_rel0) * 1000.0
            uptime_after_reload = float(rel_resp["server_uptime_s"])

            with urllib.request.urlopen(f"{base_url}/api/model-cards/{w10_key}", timeout=2.0) as r:
                w10_card_status = r.status
                w10_card = json.loads(r.read().decode("utf-8"))

            expected_color = deterministic_run_color(w10_key)
            h1_pass = (
                rel_resp["ok"] is True
                and int(rel_resp["runs_count"]) == count_before + 1
                and uptime_after_reload > uptime_before
                and float(rel_resp["reload_ms"]) < 100.0
                and reload_rtt_ms < 100.0
                and w10_card_status == 200
                and w10_card["color"] == expected_color
                and w10_card["wave_id"] == "W10"
            )
            print(
                f"[H1 Hot Reload] Runs: {count_before} -> {rel_resp['runs_count']} | "
                f"Server Reload: {rel_resp['reload_ms']:.2f} ms (RTT: {reload_rtt_ms:.2f} ms < 100 ms) | "
                f"Uptime: {uptime_before:.4f}s -> {uptime_after_reload:.4f}s (0 restarts)"
            )

            # H3 Part 1: Push Ingestion via POST /api/ingest (< 50 ms)
            ingest_body = {
                "run_key": w10_key,
                "log_append": (
                    "Params: 171,622,144\n"
                    "Step 191/3431 (Epoch 0.06 | 50.1M tok) | Train CE: 4.1100 | Speed: 118,000 tok/s\n"
                    "[Epoch 0.06 | Step 191 | 50.1M tok | T=512] Val CE: 4.0415 | Val PPL: 56.91 | Top-1: 31.80% | Top-5: 56.10% | MedRank: 12 | IsoErr: 0.0001\n"
                    "Step 382/3431 (Epoch 0.11 | 100.1M tok) | Train CE: 3.6200 | Speed: 119,500 tok/s\n"
                    "[Epoch 0.11 | Step 382 | 100.1M tok | T=512] Val CE: 3.5890 | Val PPL: 36.20 | Top-1: 36.10% | Top-5: 60.70% | MedRank: 10 | IsoErr: 0.0001\n"
                ),
                "results_json": [
                    {"step": 191, "epoch": 0.06, "tokens_M": 50.1, "val_ce": 4.0415, "val_ppl": 56.91},
                    {"step": 382, "epoch": 0.11, "tokens_M": 100.1, "val_ce": 3.5890, "val_ppl": 36.20},
                ],
            }
            t_ing0 = time.perf_counter()
            req_ing = urllib.request.Request(
                f"{base_url}/api/ingest",
                data=json.dumps(ingest_body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req_ing, timeout=2.0) as r:
                ing_resp = json.loads(r.read().decode("utf-8"))
            ingest_rtt_ms = (time.perf_counter() - t_ing0) * 1000.0

            with urllib.request.urlopen(f"{base_url}/api/status", timeout=2.0) as r:
                s_after_ing = json.loads(r.read().decode("utf-8"))
            w10_cat = next(item for item in s_after_ing["runs_catalog"] if item["key"] == w10_key)

            # H3 Part 2: 50 Concurrent GET /api/status and GET /api/model-cards requests (< 25 ms p95)
            def _concurrent_get(i: int) -> tuple[int, float]:
                ep = "/api/status" if i % 2 == 0 else "/api/model-cards"
                t0_req = time.perf_counter()
                with urllib.request.urlopen(f"{base_url}{ep}", timeout=2.0) as resp:
                    _ = resp.read()
                    st_code = resp.status
                return st_code, (time.perf_counter() - t0_req) * 1000.0

            with ThreadPoolExecutor(max_workers=10) as pool:
                conc_results = list(pool.map(_concurrent_get, range(50)))

            http_codes = [c for c, _ in conc_results]
            sorted_lat = sorted(lat for _, lat in conc_results)
            p50_ms = sorted_lat[len(sorted_lat) // 2]
            p95_ms = sorted_lat[int(0.95 * (len(sorted_lat) - 1))]
            max_ms = sorted_lat[-1]

            h3_pass = (
                ing_resp["ok"] is True
                and float(ing_resp["ingest_ms"]) < 50.0
                and ingest_rtt_ms < 50.0
                and abs(float(w10_cat["best_val_ce"]) - 3.5890) < 1e-4
                and int(w10_cat["current_step"]) == 382
                and all(c == 200 for c in http_codes)
                and p95_ms < 25.0
            )
            print(
                f"[H3 Ingest & Concurrency] Ingest: {ing_resp['ingest_ms']:.2f} ms (RTT: {ingest_rtt_ms:.2f} ms < 50 ms) | "
                f"50 Concurrent GETs: 100% HTTP 200, p50={p50_ms:.2f} ms, p95={p95_ms:.2f} ms (< 25 ms), max={max_ms:.2f} ms"
            )
        finally:
            server.shutdown()
            server.server_close()

    # Step 3: Ensure permanent live server on port 8765 is running & hot-reloaded
    ensure_res = subprocess.run(
        [str(REPO_ROOT / "scripts" / "ensure_benchmark_server.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"[Live Server 0.0.0.0:8765] {ensure_res.stdout.strip()}")

    results_payload = {
        "experiment_plan": "EXPERIMENT_PLAN_017",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "all_hypotheses_passed": bool(h1_pass and h2_pass and h3_pass),
        "hypotheses": {
            "H1_zero_downtime_dynamic_wave_registration": {
                "passed": h1_pass,
                "runs_count_before": count_before,
                "runs_count_after": int(rel_resp["runs_count"]),
                "server_uptime_before_s": uptime_before,
                "server_uptime_after_s": uptime_after_reload,
                "server_reload_ms": float(rel_resp["reload_ms"]),
                "client_reload_rtt_ms": round(reload_rtt_ms, 2),
                "target_max_reload_ms": 100.0,
                "deterministic_color": expected_color,
                "discovered_wave_id": w10_card["wave_id"],
                "discovered_wave_label": w10_card["wave_label"],
            },
            "H2_complete_cli_and_log_model_card_fidelity": {
                "passed": h2_pass,
                "backfilled_model_cards_count": len(card_files),
                "target_min_model_cards": 35,
                "schema_validation_errors_count": len(validation_errors_by_key),
                "markdown_5_section_valid_count": markdown_sections_ok_count,
                "waves_discovered": sorted(waves_discovered),
                "classical_pr360_roundtrip_checks": classical_checks,
                "quantum_frame4x_roundtrip_checks": quantum_checks,
            },
            "H3_push_ingestion_and_concurrent_lan_http": {
                "passed": h3_pass,
                "server_ingest_ms": float(ing_resp["ingest_ms"]),
                "client_ingest_rtt_ms": round(ingest_rtt_ms, 2),
                "target_max_ingest_ms": 50.0,
                "ingested_best_val_ce": float(w10_cat["best_val_ce"]),
                "ingested_step": int(w10_cat["current_step"]),
                "concurrent_requests": 50,
                "http_200_rate": sum(1 for c in http_codes if c == 200) / len(http_codes),
                "latency_p50_ms": round(p50_ms, 2),
                "latency_p95_ms": round(p95_ms, 2),
                "latency_max_ms": round(max_ms, 2),
                "target_max_p95_ms": 25.0,
            },
        },
    }

    out_path = REPO_ROOT / ".cache" / "hypothesis_experiments_017.json"
    out_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    print(f"[Saved] {out_path}")
    assert results_payload["all_hypotheses_passed"], f"Verification failed: {json.dumps(results_payload, indent=2)}"
    print("ALL EXPERIMENT PLAN 017 HYPOTHESES (H1, H2, H3) PASSED.")


if __name__ == "__main__":
    main()
