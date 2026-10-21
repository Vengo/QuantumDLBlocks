#!/usr/bin/env python3
"""Declarative Colab GPU/TPU Fleet Manager for QuantumDLBlocks (`scripts/colab_fleet.py`).

Replaces ad-hoc `colab exec` base64 tarballs, manual detached `subprocess.Popen` one-liners,
and manual log/JSON harvesting with a single deterministic CLI tool.

Key Features:
- Sequential execution across sessions to prevent `~/.config/colab-cli/sessions.json` race conditions.
- Automatic retry with exponential backoff on transient Colab WebSocket/proxy disconnects.
- Safe IPython kernel release idiom (`kill -9 <kernel_pid> && sleep 2`) when launching detached runs
  so GPU VRAM and TPU `/dev/accel0` locks are cleanly freed.
- Double-launch guard (refuses to launch on a session that already has an active training process
  unless `--force` is passed).

Subcommands:
  status     Inspect GPU hardware, running PIDs, and latest log tail across a wave manifest.
  push-code  Package `src/qdlblocks`, `src/qdlblocks_jax`, and `examples/11_*.py` and sync to sessions.
  launch     Launch detached training runs from a declarative JSON wave manifest (and verify PIDs).
  harvest    Pull completed `.log` and `_results.json` files into `.cache/live_logs/` and print summary.
  kill       Terminate running training processes on specified sessions.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
COLAB_BIN = Path("/home/festivity/.local/bin/colab")
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "live_logs"
DEFAULT_MODEL_CARDS_DIR = REPO_ROOT / ".cache" / "model_cards"
DEFAULT_SERVER_RELOAD_URL = "http://127.0.0.1:8765/api/reload"


def run_colab_exec(
    session: str,
    python_code: str,
    timeout: int = 90,
    retries: int = 3,
) -> tuple[int, str, str]:
    """Execute Python code on a remote Colab session sequentially with retry backoff.

    Uses `-f <temp_file>` so payloads of any size bypass Linux MAX_ARG_STRLEN limits.
    """
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_script = DEFAULT_CACHE_DIR / f"_tmp_colab_exec_{session}.py"
    tmp_script.write_text(python_code, encoding="utf-8")
    cmd = [
        str(COLAB_BIN),
        "exec",
        "-s",
        session,
        "--timeout",
        str(float(timeout)),
        "-f",
        str(tmp_script),
    ]
    last_rc, last_out, last_err = 1, "", ""
    try:
        for attempt in range(1, retries + 1):
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout + 15,
                    check=False,
                )
                last_rc, last_out, last_err = proc.returncode, proc.stdout, proc.stderr
                if proc.returncode == 0:
                    return 0, proc.stdout, proc.stderr
            except subprocess.TimeoutExpired as exc:
                last_rc = 124
                last_out = (exc.stdout or b"").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                last_err = f"Timeout after {timeout}s on attempt {attempt}/{retries}"
            if attempt < retries:
                time.sleep(2.0 * attempt)
    finally:
        tmp_script.unlink(missing_ok=True)
    return last_rc, last_out, last_err


def build_code_bundle_b64(include_paths: list[str] | None = None) -> tuple[str, int]:
    """Create an in-memory xz-compressed tarball of `src/` and `examples/11_*.py` and return (b64_str, byte_size)."""
    if include_paths is None:
        include_paths = [
            "src/qdlblocks",
            "src/qdlblocks_jax",
            "examples/11_fineweb_100m_multiepoch_gpu_tpu.py",
        ]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz", preset=9) as tar:
        for rel in include_paths:
            abs_path = REPO_ROOT / rel
            if not abs_path.exists():
                raise FileNotFoundError(f"Required bundle path not found: {abs_path}")
            tar.add(str(abs_path), arcname=rel, filter=lambda ti: None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti)
    raw_bytes = buf.getvalue()
    return base64.b64encode(raw_bytes).decode("ascii"), len(raw_bytes)


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and validate a wave manifest JSON file."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "runs" not in data or not isinstance(data["runs"], list):
        raise ValueError(f"Manifest {manifest_path} must contain a top-level 'runs' list.")
    return data


def filter_runs(
    manifest: dict[str, Any],
    only_keys: str | None = None,
    only_sessions: str | None = None,
) -> list[dict[str, Any]]:
    runs = list(manifest["runs"])
    if only_keys:
        allowed = {k.strip() for k in only_keys.split(",") if k.strip()}
        runs = [r for r in runs if r["key"] in allowed]
    if only_sessions:
        allowed_s = {s.strip() for s in only_sessions.split(",") if s.strip()}
        runs = [r for r in runs if r["session"] in allowed_s]
    return runs


def cmd_ensure_sessions(runs: list[dict[str, Any]]) -> bool:
    """Ensure all sessions in `runs` exist in `~/.config/colab-cli/sessions.json`, creating missing ones sequentially."""
    sessions_file = Path.home() / ".config" / "colab-cli" / "sessions.json"
    existing_sessions: dict[str, Any] = {}
    if sessions_file.exists():
        try:
            existing_sessions = json.loads(sessions_file.read_text(encoding="utf-8"))
        except Exception:
            existing_sessions = {}

    all_ok = True
    for run in runs:
        session = run["session"]
        is_a100 = "a100" in session.lower() or "a100" in str(run.get("hardware", "")).lower()
        gpu_variant = run.get("gpu_variant", "A100" if is_a100 else "G4")
        tpu_variant = run.get("tpu_variant")
        high_mem = bool(run.get("high_mem", is_a100))

        if session in existing_sessions:
            rc, out, _ = run_colab_exec(session, 'print("ALIVE")', timeout=45, retries=3)
            if rc == 0 and "ALIVE" in out:
                print(f"  [SESSION OK] {session} is already active.")
                continue
            print(f"  [SESSION STALE] {session} not responding after 3 retries; recreating...")
            subprocess.run([str(COLAB_BIN), "stop", "-s", session], capture_output=True, text=True, check=False)

        if tpu_variant:
            candidates = [("tpu", str(tpu_variant))]
        else:
            fallbacks = [str(gpu_variant)] + [g for g in ("G4", "A100", "L4", "T4") if g != str(gpu_variant)]
            candidates = [("gpu", g) for g in fallbacks]

        provisioned = False
        for accel_flag, accel_val in candidates:
            cmd = [str(COLAB_BIN), "new", "-s", session, f"--{accel_flag}", accel_val]
            if high_mem:
                cmd.append("--high-mem")
            print(f"  [PROVISIONING] {' '.join(cmd)} ...")
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)
            dt = time.time() - t0
            if proc.returncode == 0:
                print(f"  [PROVISIONED] {session} with {accel_flag.upper()}={accel_val} ({dt:.1f}s)")
                provisioned = True
                break
            print(
                f"  [WARN PROVISIONING] {session} ({accel_val}, rc={proc.returncode}, {dt:.1f}s): "
                f"{proc.stdout.strip()} | {proc.stderr.strip()}",
                file=sys.stderr,
            )
        if not provisioned:
            all_ok = False
    return all_ok


def cmd_push_code(runs: list[dict[str, Any]], ensure_sessions: bool = True) -> bool:
    """Push the current local `src/` and `examples/11_*.py` bundle to all unique sessions in `runs`."""
    if ensure_sessions and not cmd_ensure_sessions(runs):
        return False
    b64_payload, num_bytes = build_code_bundle_b64()
    sessions = list(dict.fromkeys(r["session"] for r in runs))
    print(f"[push-code] Built xz code bundle ({num_bytes:,} bytes compressed) for {len(sessions)} session(s): {sessions}")

    remote_script = f"""
import base64, io, tarfile, sys
raw = base64.b64decode({b64_payload!r})
with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as tar:
    tar.extractall("/content")
sys.path.insert(0, "/content/src")
import qdlblocks
print("SYNC_OK:", qdlblocks.__file__)
"""
    all_ok = True
    for session in sessions:
        t0 = time.time()
        rc, out, err = run_colab_exec(session, remote_script, timeout=90)
        dt = time.time() - t0
        if rc == 0 and "SYNC_OK:" in out:
            print(f"  [OK] {session} ({dt:.1f}s): {out.strip().splitlines()[-1]}")
        else:
            all_ok = False
            print(f"  [ERR] {session} (rc={rc}, {dt:.1f}s): {out.strip()} | {err.strip()}", file=sys.stderr)
    return all_ok


def cmd_status(runs: list[dict[str, Any]]) -> int:
    """Check GPU hardware, running processes, and latest log progress for each run."""
    print(f"{'Session':<16} | {'Run Key':<38} | {'GPU':<24} | {'PID':<7} | {'Latest Log Line'}")
    print("-" * 135)
    for run in runs:
        session = run["session"]
        key = run["key"]
        log_path = run.get("remote_log", f"/content/{key}.log")
        probe_code = f"""
import json, os, subprocess
gpu = subprocess.getoutput("nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null").splitlines()
gpu_str = gpu[0].strip() if gpu else "No GPU / TPU"
pids = subprocess.getoutput("pgrep -f '[1]1_fineweb_100m_multiepoch_gpu_tpu.py'").splitlines()
last_line = ""
if os.path.exists({log_path!r}):
    with open({log_path!r}, "r", errors="replace") as f:
        lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
        last_line = lines[-1] if lines else "(empty log)"
else:
    last_line = "(log not found)"
print("STATUS_JSON:" + json.dumps({{"gpu": gpu_str, "pids": pids, "last_line": last_line[-90:]}}))
"""
        rc, out, err = run_colab_exec(session, probe_code, timeout=45, retries=2)
        if rc == 0 and "STATUS_JSON:" in out:
            payload_line = [ln for ln in out.splitlines() if "STATUS_JSON:" in ln][-1]
            info = json.loads(payload_line.split("STATUS_JSON:", 1)[1])
            pid_str = ",".join(info["pids"]) if info["pids"] else "-"
            print(f"{session:<16} | {key:<38} | {info['gpu'][:24]:<24} | {pid_str:<7} | {info['last_line']}")
        else:
            print(f"{session:<16} | {key:<38} | {'ERROR':<24} | {'-':<7} | rc={rc} {err.strip()[:60]}")
    return 0


def trigger_benchmark_server_reload(
    server_url: str = DEFAULT_SERVER_RELOAD_URL,
    timeout: float = 2.0,
) -> bool:
    """Send a non-blocking POST /api/reload to the permanent benchmarking server if running."""
    import urllib.request

    req = urllib.request.Request(
        server_url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status) == 200
    except Exception:
        return False


def cmd_register_cards(
    runs: list[dict[str, Any]],
    manifest: dict[str, Any] | None = None,
    cards_dir: Path = DEFAULT_MODEL_CARDS_DIR,
    server_url: str = DEFAULT_SERVER_RELOAD_URL,
) -> int:
    """Generate `.cache/model_cards/<run_key>.json` for each run and trigger zero-downtime server reload."""
    from qdlblocks.benchmarks.model_card import (
        build_model_card_from_manifest_entry,
        save_model_card_atomic,
    )

    cards_dir.mkdir(parents=True, exist_ok=True)
    wave_id = str((manifest or {}).get("wave", ""))
    wave_desc = str((manifest or {}).get("description", ""))
    plan_id = str((manifest or {}).get("plan_id", ""))

    for run in runs:
        key = str(run["key"])
        existing_path = cards_dir / f"{key}.json"
        existing_telemetry = None
        if existing_path.exists():
            try:
                old_card = json.loads(existing_path.read_text(encoding="utf-8"))
                existing_telemetry = old_card.get("telemetry")
            except Exception:
                existing_telemetry = None

        card = build_model_card_from_manifest_entry(
            run,
            wave_id=wave_id,
            wave_description=wave_desc,
            plan_id=plan_id,
        )
        if existing_telemetry:
            card["telemetry"] = existing_telemetry
        out_file = save_model_card_atomic(card, cards_dir)
        print(f"  [MODEL CARD] {key} -> {out_file}")

    reloaded = trigger_benchmark_server_reload(server_url=server_url, timeout=2.0)
    if reloaded:
        print(f"  [HOT RELOAD] Notified permanent benchmark server at {server_url} (0s downtime).")
    else:
        print(f"  [HOT RELOAD] Benchmark server at {server_url} not reachable (cards saved to disk for next scan).")
    return 0


def cmd_launch(
    runs: list[dict[str, Any]],
    push_code: bool = True,
    force: bool = False,
    verify_wait_sec: float = 8.0,
    manifest: dict[str, Any] | None = None,
) -> int:
    """Register model cards, ensure sessions exist, push code bundle (optional), launch detached runs, and verify live PIDs."""
    cmd_register_cards(runs, manifest=manifest)
    if not cmd_ensure_sessions(runs):
        print("[launch] Aborting launch because one or more sessions could not be provisioned.", file=sys.stderr)
        return 1
    if push_code:
        if not cmd_push_code(runs, ensure_sessions=False):
            print("[launch] Aborting launch because code sync failed on one or more sessions.", file=sys.stderr)
            return 1

    for run in runs:
        session = run["session"]
        key = run["key"]
        cli_args = run["cli_args"]
        remote_log = run.get("remote_log", f"/content/{key}.log")
        remote_json = run.get("remote_json", f"/content/{key}_results.json")

        full_train_cmd = (
            f"PYTHONPATH=/content/src /usr/bin/python3 -u /content/examples/11_fineweb_100m_multiepoch_gpu_tpu.py "
            f"{cli_args} --out-json {remote_json} > {remote_log} 2>&1"
        )

        launch_script = f"""
import os, subprocess, sys
existing = subprocess.getoutput("pgrep -f '[1]1_fineweb_100m_multiepoch_gpu_tpu.py'").split()
if existing and not {force!r}:
    print("ALREADY_RUNNING:" + ",".join(existing))
    sys.exit(0)
if existing and {force!r}:
    subprocess.run(["pkill", "-9", "-f", "[1]1_fineweb_100m_multiepoch_gpu_tpu.py"], check=False)

pid = os.getpid()
detached_cmd = f"sleep 1 && kill -9 {{pid}} && sleep 2 && {full_train_cmd}"
subprocess.Popen(["bash", "-c", detached_cmd], start_new_session=True)
print("LAUNCH_DISPATCHED")
"""
        print(f"[launch] Dispatching {key} on session {session} ...")
        rc, out, err = run_colab_exec(session, launch_script, timeout=45, retries=2)
        if "ALREADY_RUNNING:" in out:
            print(f"  [SKIP] {session} ({key}) already has running PID(s): {out.strip()} (use --force to restart)")
        elif rc == 0 or "LAUNCH_DISPATCHED" in out or "Websocket closed" in err or "EOF" in err:
            # Note: killing the IPython kernel PID intentionally closes the current execution channel after LAUNCH_DISPATCHED
            print(f"  [DISPATCHED] {session} -> {remote_log}")
        else:
            print(f"  [WARN] {session} returned rc={rc}: {out.strip()} | {err.strip()}")

    trigger_benchmark_server_reload(timeout=2.0)
    if verify_wait_sec > 0:
        print(f"\n[launch] Waiting {verify_wait_sec:.1f}s before verifying running PIDs across fleet...")
        time.sleep(verify_wait_sec)
        cmd_status(runs)
    return 0


def cmd_harvest(runs: list[dict[str, Any]], out_dir: Path = DEFAULT_CACHE_DIR) -> int:
    """Download `.log` and `_results.json` from each session into `out_dir` and print a summary table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    import re

    eval_re = re.compile(
        r"Val CE:\s*([0-9.]+)\s*\|\s*Val PPL:\s*([0-9.]+).*?\|\s*([0-9,]+)\s*tok/s"
    )
    summary_rows: list[dict[str, Any]] = []
    for run in runs:
        session = run["session"]
        key = run["key"]
        remote_log = run.get("remote_log", f"/content/{key}.log")
        remote_json = run.get("remote_json", f"/content/{key}_results.json")
        local_log = out_dir / f"{key}.log"
        local_json = out_dir / f"{key}_results.json"

        harvest_script = f"""
import base64, json, os
res = {{"log_b64": None, "json_b64": None}}
if os.path.exists({remote_log!r}):
    with open({remote_log!r}, "rb") as f:
        res["log_b64"] = base64.b64encode(f.read()).decode("ascii")
if os.path.exists({remote_json!r}):
    with open({remote_json!r}, "rb") as f:
        res["json_b64"] = base64.b64encode(f.read()).decode("ascii")
print("HARVEST_PAYLOAD:" + json.dumps(res))
"""
        rc, out, _ = run_colab_exec(session, harvest_script, timeout=90, retries=2)
        if rc == 0 and "HARVEST_PAYLOAD:" in out:
            payload_line = [ln for ln in out.splitlines() if "HARVEST_PAYLOAD:" in ln][-1]
            data = json.loads(payload_line.split("HARVEST_PAYLOAD:", 1)[1])
            if data.get("log_b64"):
                local_log.write_bytes(base64.b64decode(data["log_b64"]))
            if data.get("json_b64"):
                local_json.write_bytes(base64.b64decode(data["json_b64"]))

        best_ce: float | None = None
        best_ppl: float | None = None
        best_tps: float | None = None
        status_str = f"ERR(rc={rc})"

        if local_json.exists():
            res_raw = json.loads(local_json.read_text(encoding="utf-8"))
            if isinstance(res_raw, list) and res_raw:
                res_obj = min(res_raw, key=lambda r: float(r.get("val_ce", r.get("best_val_loss", 999.0))))
            elif isinstance(res_raw, dict):
                res_obj = res_raw
            else:
                res_obj = {}
            best_ce = res_obj.get("val_ce") or res_obj.get("best_val_loss") or res_obj.get("final_val_loss")
            best_ppl = res_obj.get("val_ppl") or res_obj.get("best_val_ppl") or res_obj.get("final_val_ppl")
            best_tps = res_obj.get("tok_per_s") or res_obj.get("mean_tokens_per_sec") or res_obj.get("tokens_per_sec")
            status_str = "COMPLETED"
        elif local_log.exists():
            log_text = local_log.read_text(encoding="utf-8", errors="replace")
            for ln in log_text.splitlines():
                m = eval_re.search(ln)
                if m:
                    ce_v = float(m.group(1))
                    ppl_v = float(m.group(2))
                    tps_v = float(m.group(3).replace(",", ""))
                    if best_ce is None or ce_v <= best_ce:
                        best_ce, best_ppl, best_tps = ce_v, ppl_v, tps_v
            status_str = "COMPLETED" if "COMPLETED" in log_text else "IN_PROGRESS"

        summary_rows.append(
            {
                "key": key,
                "session": session,
                "status": status_str,
                "best_val_ce": best_ce,
                "best_val_ppl": best_ppl,
                "tok_per_s": best_tps,
            }
        )

    trigger_benchmark_server_reload(timeout=2.0)
    print(f"\nHarvested {len(summary_rows)} run(s) into {out_dir}:")
    print(f"{'Run Key':<38} | {'Session':<14} | {'Status':<12} | {'Best Val CE':<12} | {'Val PPL':<9} | {'Tok/s'}")
    print("-" * 105)
    for row in summary_rows:
        ce_str = f"{row['best_val_ce']:.4f}" if isinstance(row["best_val_ce"], (int, float)) else "-"
        ppl_str = f"{row['best_val_ppl']:.2f}" if isinstance(row["best_val_ppl"], (int, float)) else "-"
        tps_str = f"{row['tok_per_s']:,.0f}" if isinstance(row["tok_per_s"], (int, float)) else "-"
        print(f"{row['key']:<38} | {row['session']:<14} | {row['status']:<12} | {ce_str:<12} | {ppl_str:<9} | {tps_str}")
    return 0


def cmd_kill(runs: list[dict[str, Any]]) -> int:
    """Kill running `11_fineweb_100m_multiepoch_gpu_tpu.py` processes on the target sessions."""
    sessions = list(dict.fromkeys(r["session"] for r in runs))
    for session in sessions:
        rc, out, _ = run_colab_exec(
            session,
            'import subprocess; subprocess.run(["pkill", "-9", "-f", "11_fineweb_100m_multiepoch_gpu_tpu.py"], check=False); print("KILLED")',
            timeout=30,
        )
        print(f"[kill] {session}: rc={rc} {out.strip()}")
    return 0


def cmd_compare(
    waves_str: str = "w6,w7,w8,w9",
    log_dir: Path = DEFAULT_CACHE_DIR,
    optimizer_filter: str | None = None,
    cards_dir: Path = DEFAULT_MODEL_CARDS_DIR,
) -> int:
    """Print an aligned 50M-token trajectory comparison table across waves from `.cache/live_logs/`."""
    import re

    prefixes = [w.strip().lower() for w in waves_str.split(",") if w.strip()]
    opt_filter_norm = optimizer_filter.strip().lower() if optimizer_filter else None
    eval_re = re.compile(
        r"\[\s*Epoch\s+[0-9.]+\s*\|\s*Step\s+([0-9]+)\s*\|\s*([0-9.]+)M\s*tok.*?Val CE:\s*([0-9.]+)\s*\|\s*Val PPL:\s*([0-9.]+).*?IsoErr:\s*([0-9.]+).*?\|\s*([0-9,]+)\s*tok/s"
    )
    milestones = [0, 50, 100, 150, 200, 300, 450, 600, 750, 900]
    rows: list[tuple[str, dict[int, float], float | None, float | None, float | None]] = []

    if log_dir.exists():
        for log_file in sorted(log_dir.glob("*.log")):
            stem = log_file.stem
            if prefixes and not any(stem.lower().startswith(p) for p in prefixes):
                continue
            log_text = log_file.read_text(encoding="utf-8", errors="replace")
            if opt_filter_norm:
                card_file = cards_dir / f"{stem}.json"
                card_opt = ""
                if card_file.exists():
                    try:
                        cdata = json.loads(card_file.read_text(encoding="utf-8"))
                        card_opt = str(cdata.get("optimization", {}).get("optimizer_mode", "")).lower()
                    except Exception:
                        card_opt = ""
                if opt_filter_norm not in card_opt and opt_filter_norm not in stem.lower() and opt_filter_norm not in log_text.lower():
                    continue
            pts: dict[int, float] = {}
            best_ce: float | None = None
            last_iso: float | None = None
            last_tps: float | None = None
            for ln in log_text.splitlines():
                m = eval_re.search(ln)
                if m:
                    tok_m = float(m.group(2))
                    ce_v = float(m.group(3))
                    iso_v = float(m.group(5))
                    tps_v = float(m.group(6).replace(",", ""))
                    nearest_m = min(milestones, key=lambda target: abs(tok_m - target))
                    if abs(tok_m - nearest_m) <= 15.0:
                        pts[nearest_m] = ce_v
                    if best_ce is None or ce_v < best_ce:
                        best_ce = ce_v
                    last_iso = iso_v
                    if tps_v > 0:
                        last_tps = tps_v
            if pts or best_ce is not None:
                rows.append((stem, pts, best_ce, last_iso, last_tps))

    hdr_ms = " | ".join(f"{m:>5}M" for m in (0, 50, 100, 200, 450, 900))
    print(f"{'Run Key':<42} | {hdr_ms} | {'Best CE':>7} | {'IsoErr':>6} | {'Tok/s':>8}")
    print("-" * 120)
    for stem, pts, best_ce, last_iso, last_tps in rows:
        ms_str = " | ".join(f"{pts[m]:6.4f}" if m in pts else f"{'-':>6}" for m in (0, 50, 100, 200, 450, 900))
        b_str = f"{best_ce:7.4f}" if best_ce is not None else f"{'-':>7}"
        i_str = f"{last_iso:6.2f}" if last_iso is not None else f"{'-':>6}"
        t_str = f"{last_tps:8,.0f}" if last_tps is not None else f"{'-':>8}"
        print(f"{stem[:42]:<42} | {ms_str} | {b_str} | {i_str} | {t_str}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Declarative Colab GPU/TPU Fleet Manager for QuantumDLBlocks.")
    parser.add_argument(
        "--compare-waves",
        type=str,
        default=None,
        help="Optional comma-separated wave prefixes (e.g. 'w6,w7,w8,w9') to print aligned trajectory table from .cache/live_logs/.",
    )
    parser.add_argument(
        "--optimizer-filter",
        type=str,
        default=None,
        help="Optional optimizer mode filter (e.g. 'muon_adamw', 'pure_adamw', 'sgd_nesterov') when comparing runs.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=False)

    for name in ("status", "push-code", "launch", "harvest", "kill", "register-cards"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True, help="Path to wave manifest JSON file.")
        p.add_argument("--only", type=str, default=None, help="Comma-separated run keys to filter.")
        p.add_argument("--sessions", type=str, default=None, help="Comma-separated session names to filter.")
        if name == "launch":
            p.add_argument("--no-push-code", action="store_true", help="Skip pushing the code bundle before launching.")
            p.add_argument("--force", action="store_true", help="Kill any existing training process on the session before launching.")
            p.add_argument("--verify-wait-sec", type=float, default=8.0, help="Seconds to wait before checking live PIDs.")
        if name == "harvest":
            p.add_argument("--out-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Local directory for harvested logs/JSON.")
        if name == "register-cards":
            p.add_argument("--cards-dir", type=Path, default=DEFAULT_MODEL_CARDS_DIR, help="Directory for Model Card JSON files.")
            p.add_argument("--server-url", type=str, default=DEFAULT_SERVER_RELOAD_URL, help="Reload endpoint of benchmarking server.")

    p_cmp = sub.add_parser("compare", help="Compare aligned trajectories across waves from .cache/live_logs/.")
    p_cmp.add_argument("--waves", "--compare-waves", dest="waves", type=str, default="w6,w7,w8,w9", help="Comma-separated wave prefixes.")
    p_cmp.add_argument("--log-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Directory containing synced .log files.")
    p_cmp.add_argument("--optimizer-filter", type=str, default=None, help="Filter runs by optimizer mode (e.g. muon_adamw, pure_adamw).")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.compare_waves:
        return cmd_compare(args.compare_waves, optimizer_filter=args.optimizer_filter)
    if args.subcommand == "compare":
        return cmd_compare(args.waves, log_dir=args.log_dir, optimizer_filter=args.optimizer_filter)
    if not args.subcommand:
        parser.print_help()
        return 1

    manifest = load_manifest(args.manifest)
    runs = filter_runs(manifest, only_keys=args.only, only_sessions=args.sessions)
    if not runs:
        print("No matching runs found in manifest after filtering.", file=sys.stderr)
        return 1

    if args.subcommand == "status":
        return cmd_status(runs)
    if args.subcommand == "push-code":
        return 0 if cmd_push_code(runs) else 1
    if args.subcommand == "register-cards":
        return cmd_register_cards(runs, manifest=manifest, cards_dir=args.cards_dir, server_url=args.server_url)
    if args.subcommand == "launch":
        return cmd_launch(
            runs,
            push_code=not args.no_push_code,
            force=args.force,
            verify_wait_sec=args.verify_wait_sec,
            manifest=manifest,
        )
    if args.subcommand == "harvest":
        return cmd_harvest(runs, out_dir=args.out_dir)
    if args.subcommand == "kill":
        return cmd_kill(runs)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

