"""Unit tests for Plan 018: Unpinned Leaderboard Sorting, Persistent Obsolete Lifecycle, --muon-lr, & Wave-10 Manifest."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import threading
import time
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from qdlblocks.benchmarks.model_card import (
    BASELINE_RUN_KEYS,
    DynamicWaveRegistry,
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


def test_catalog_deduplication_and_unpinned_baseline_sorting(tmp_path: Path) -> None:
    """Verify catalog deduplication, numeric wave_num, arch_family, and unpinned Best Val CE sorting."""
    cards_dir = tmp_path / "model_cards"
    logs_dir = tmp_path / "live_logs"
    shutil.copytree(MODEL_CARDS_DIR, cards_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.tmp", ".*"))
    shutil.copytree(LIVE_DIR, logs_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.tmp", ".*"))
    # Remove any custom ui_prefs.json so default obsolete behavior is tested cleanly
    if (logs_dir / "ui_prefs.json").exists():
        (logs_dir / "ui_prefs.json").unlink()

    snap = collect_snapshot(
        skip_remote_poll=True,
        configs_dir=CONFIGS_DIR,
        model_cards_dir=cards_dir,
        live_logs_dir=logs_dir,
        include_historical_specs=True,
    )
    catalog = snap["runs_catalog"]
    keys = [r["key"] for r in catalog]

    # 1. Every key in runs_catalog is 100% unique (0 duplicates of baseline keys)
    assert len(keys) == len(set(keys)), f"Duplicate keys found in runs_catalog: {len(keys)} vs {len(set(keys))}"
    for b_key in BASELINE_RUN_KEYS:
        assert keys.count(b_key) == 1, f"Expected exactly 1 entry for {b_key}, got {keys.count(b_key)}"

    # 2. Every entry has numeric wave_num, arch_family, and boolean obsolete
    for r in catalog:
        assert isinstance(r.get("wave_num"), (int, float)), f"Missing numeric wave_num on {r['key']}"
        assert r.get("arch_family") in ("quantum_born", "classical_modded"), f"Invalid arch_family on {r['key']}"
        assert isinstance(r.get("obsolete"), bool), f"Missing boolean obsolete on {r['key']}"
        if r["key"] in DEFAULT_OBSOLETE_KEYS or r.get("run_state") == "OOM_CRASH":
            assert r["obsolete"] is True, f"Expected {r['key']} to be obsolete by default"

    # 3. Pure Best Val CE ascending sort places the #1 Classical Speedrun model at index 0
    #    and the #1 QuantumBornGPT model at index 0 of quantum_born, with all Baseline runs at rank >= 40
    ce_sorted = sorted(
        catalog,
        key=lambda r: (
            r.get("best_val_ce") is None,
            float(r["best_val_ce"]) if r.get("best_val_ce") is not None else 999.0,
        ),
    )
    assert ce_sorted[0]["arch_family"] == "classical_modded"
    assert float(ce_sorted[0]["best_val_ce"]) <= 3.4285

    qborn_sorted = [r for r in ce_sorted if r["arch_family"] == "quantum_born" and r.get("best_val_ce") is not None]
    assert qborn_sorted[0]["key"] in (
        "w11_900m_qborn_speedrun_v4_L8_leaderboard_hybrids",
        "w10_900m_qborn_speedrun_v3_asym1024_frame4d_L8",
        "w9_900m_qborn_speedrun_v2_asym1024_L8",
    )
    assert float(qborn_sorted[0]["best_val_ce"]) <= 3.7205

    # Verify default backend catalog sort also has the #1 non-obsolete model at index 0
    assert catalog[0]["key"] == ce_sorted[0]["key"]

    for idx, r in enumerate(ce_sorted):
        if r["wave_id"] == "Baseline":
            assert r["wave_num"] == 0.0
            assert idx >= 40, f"Baseline run {r['key']} pinned too high at index {idx}"

    # 4. Sorting by wave_num descending places W11/W10 > W9 > W8 > ... > W1 > Baseline
    wave_desc = sorted(catalog, key=lambda r: float(r["wave_num"]), reverse=True)
    wave_nums = [float(r["wave_num"]) for r in wave_desc]
    assert wave_nums == sorted(wave_nums, reverse=True)
    assert wave_desc[0]["wave_id"] in ("W10", "W11", "W12", "W13")
    assert wave_desc[0]["wave_num"] >= 10.0
    assert wave_desc[-1]["wave_id"] == "Baseline"
    assert wave_desc[-1]["wave_num"] == 0.0


def test_zero_downtime_obsolete_endpoint_and_persistence(tmp_path: Path) -> None:
    """Verify POST /api/obsolete marks and restores runs in <25ms with 0 restarts and full disk persistence."""
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
    )
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()

    try:
        s0 = _http_get_json(f"{base_url}/api/status")
        up0 = float(s0["server_uptime_s"])

        # Mark w6_900m_classical_modded_124m and gpu_5ep obsolete
        t0 = time.perf_counter()
        res_mark = _http_post_json(
            f"{base_url}/api/obsolete",
            {
                "run_keys": ["w6_900m_classical_modded_124m", "gpu_5ep"],
                "obsolete": True,
                "reason": "Superseded by Wave-9/10 PR#360 classical speedrun",
            },
        )
        mark_ms = (time.perf_counter() - t0) * 1000.0
        assert res_mark["ok"] is True
        assert res_mark["obsolete"] is True
        assert mark_ms < 50.0
        assert float(res_mark["obsolete_ms"]) < 25.0
        assert float(res_mark["server_uptime_s"]) >= up0

        # Verify disk persistence in both model_cards/*.json and ui_prefs.json
        w6_disk = json.loads((cards_dir / "w6_900m_classical_modded_124m.json").read_text(encoding="utf-8"))
        assert w6_disk["obsolete"] is True
        assert "Superseded" in w6_disk["obsolete_reason"]
        assert bool(w6_disk["obsolete_at"])

        prefs_disk = json.loads((logs_dir / "ui_prefs.json").read_text(encoding="utf-8"))
        assert prefs_disk["obsoleteMap"]["w6_900m_classical_modded_124m"] is True
        assert prefs_disk["obsoleteMap"]["gpu_5ep"] is True

        # Verify live /api/status snapshot reflects obsolete=True immediately
        s1 = _http_get_json(f"{base_url}/api/status")
        w6_entry = next(r for r in s1["runs_catalog"] if r["key"] == "w6_900m_classical_modded_124m")
        assert w6_entry["obsolete"] is True

        # Now restore w6_900m_classical_modded_124m via POST /api/obsolete
        res_restore = _http_post_json(
            f"{base_url}/api/obsolete",
            {
                "run_key": "w6_900m_classical_modded_124m",
                "obsolete": False,
            },
        )
        assert res_restore["ok"] is True
        assert res_restore["obsolete"] is False
        assert float(res_restore["server_uptime_s"]) >= float(res_mark["server_uptime_s"])

        w6_disk_restored = json.loads((cards_dir / "w6_900m_classical_modded_124m.json").read_text(encoding="utf-8"))
        assert w6_disk_restored["obsolete"] is False
        assert w6_disk_restored["obsolete_reason"] == ""

        s2 = _http_get_json(f"{base_url}/api/status")
        w6_entry_restored = next(r for r in s2["runs_catalog"] if r["key"] == "w6_900m_classical_modded_124m")
        assert w6_entry_restored["obsolete"] is False
    finally:
        server.shutdown()
        server.server_close()


def test_wave10_manifest_and_muon_lr_card_registration(tmp_path: Path) -> None:
    """Verify configs/wave10_fleet.json registers 5 valid W10 Model Cards and parses --muon-lr and sgd_nesterov."""
    cards_dir = tmp_path / "model_cards"
    logs_dir = tmp_path / "live_logs"
    cards_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    registry = DynamicWaveRegistry(
        configs_dir=CONFIGS_DIR,
        model_cards_dir=cards_dir,
        live_logs_dir=logs_dir,
        historical_specs=[],
    )
    specs = registry.discover_all_specs(persist_cards=True)
    w10_specs = [s for s in specs if s["wave_id"] == "W10"]
    assert len(w10_specs) == 5

    by_key = {s["key"]: s["model_card"] for s in w10_specs}
    expected_keys = {
        "w10_900m_classical_speedrun_v3_fastsdpa_b131k",
        "w10_900m_classical_speedrun_v3_b65k_13725steps",
        "w10_900m_qborn_speedrun_v3_asym1024_frame4d_L8",
        "w10_900m_qborn_speedrun_v3_asym1024_L12_deep",
        "w10_900m_classical_opt_ablation_sgd_nesterov",
    }
    assert set(by_key.keys()) == expected_keys

    for rkey, card in by_key.items():
        errs = validate_model_card(card)
        assert not errs, f"Validation errors for {rkey}: {errs}"
        assert card["wave_id"] == "W10"
        assert card["wave_num"] == 10
        assert card["plan_id"] == "EXPERIMENT_PLAN_018"
        assert (cards_dir / f"{rkey}.json").exists()

    c1 = by_key["w10_900m_classical_speedrun_v3_fastsdpa_b131k"]
    assert c1["optimization"]["muon_lr"] == 0.04
    assert c1["optimization"]["optimizer_mode"] == "muon_adamw"
    assert c1["data_and_batch"]["tokens_per_step"] == 131072
    assert c1["architecture"]["doc_mask_attn"] is False
    assert c1["architecture"]["sliding_window"] == 0

    c2 = by_key["w10_900m_classical_speedrun_v3_b65k_13725steps"]
    assert c2["optimization"]["muon_lr"] == 0.035
    assert c2["data_and_batch"]["tokens_per_step"] == 65536

    c3 = by_key["w10_900m_qborn_speedrun_v3_asym1024_frame4d_L8"]
    assert c3["architecture"]["family"] == "quantum_born"
    assert c3["architecture"]["frame_expansion"] == 4.0
    assert c3["architecture"]["readout_mode"] == "asymmetric"
    assert c3["architecture"]["seed_dim"] == 1024

    c4 = by_key["w10_900m_qborn_speedrun_v3_asym1024_L12_deep"]
    assert c4["architecture"]["num_layers"] == 12
    assert c4["architecture"]["skip_attn_layers"] == [6]

    c5 = by_key["w10_900m_classical_opt_ablation_sgd_nesterov"]
    assert c5["optimization"]["optimizer_mode"] == "sgd_nesterov"
    assert c5["optimization"]["muon_lr"] == 0.05


def test_html_dashboard_contains_obsolete_and_family_controls(tmp_path: Path) -> None:
    """Verify render_html_dashboard includes POST /api/obsolete, Show Obsolete, Family filters, and wave_num sort."""
    cards_dir = tmp_path / "model_cards"
    logs_dir = tmp_path / "live_logs"
    shutil.copytree(MODEL_CARDS_DIR, cards_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.tmp", ".*"))
    shutil.copytree(LIVE_DIR, logs_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.tmp", ".*"))

    snap = collect_snapshot(
        skip_remote_poll=True,
        configs_dir=CONFIGS_DIR,
        model_cards_dir=cards_dir,
        live_logs_dir=logs_dir,
        include_historical_specs=True,
    )
    html = render_html_dashboard(snap)

    assert "/api/obsolete" in html
    assert "showObsolete" in html
    assert "Show Obsolete" in html
    assert "filterFamily" in html
    assert "Quantum Born" in html
    assert "Classical Baseline" in html
    assert "a.wave_num" in html
    assert "bestClassicalRun" in html
    assert "bestQuantumRun1" in html
