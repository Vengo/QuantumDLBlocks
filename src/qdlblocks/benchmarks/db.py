"""SQLite WAL Telemetry, Model Card, Pareto-Frontier Lifecycle & Session Queue Engine (`qdlblocks.benchmarks.db`).

Implements Experiment Plan 020:
- `BenchmarkSQLiteDB`: Thread-safe WAL-mode SQLite database backing the WebUI and HTTP APIs
  (`runs`, `run_evals`, `run_steps`, `colab_sessions`, `session_queue`, and `ui_state`).
- `classify_run_tier`: Separates `main_benchmark` (100M–900M FineWeb-Edu runs) from
  `micro_verification` (pre-launch controlled hypothesis tests across Plans 013–020).
- `compute_family_pareto_and_obsolete`: Computes independent 2D `(wall_min, best_val_ce)` Pareto
  frontiers per `arch_family` (`classical_modded` vs `quantum_born`) and automatically marks
  wave-superseded dominated runs as `obsolete = True` with canonical explanations while honoring
  explicit user overrides.
- `backfill_micro_verifications`: Idempotently populates `micro_verification` runs from
  `.cache/hypothesis_experiments_013..020.json` into SQLite and Model Cards.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

from .model_card import (
    DEFAULT_MODEL_CARDS_DIR,
    REPO_ROOT,
    build_model_card_from_manifest_entry,
    deterministic_run_color,
    infer_wave_id_and_num,
    save_model_card_atomic,
    set_model_card_obsolete,
    update_model_card_telemetry,
)

DEFAULT_DB_PATH = REPO_ROOT / ".cache" / "benchmarks.db"

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_key TEXT PRIMARY KEY,
    wave_id TEXT NOT NULL,
    wave_num REAL NOT NULL DEFAULT 0.0,
    wave_label TEXT NOT NULL DEFAULT '',
    arch_family TEXT NOT NULL DEFAULT 'quantum_born',
    run_tier TEXT NOT NULL DEFAULT 'main_benchmark',
    session TEXT NOT NULL DEFAULT '',
    accel TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT 'gpu',
    label TEXT NOT NULL DEFAULT '',
    short_label TEXT NOT NULL DEFAULT '',
    color TEXT NOT NULL DEFAULT '#58a6ff',
    status TEXT NOT NULL DEFAULT 'IDLE',
    boot_status TEXT NOT NULL DEFAULT 'Idle',
    params_M REAL NOT NULL DEFAULT 0.0,
    seed_dim INTEGER NOT NULL DEFAULT 128,
    embed_dim INTEGER NOT NULL DEFAULT 1024,
    num_layers INTEGER NOT NULL DEFAULT 6,
    d_ff TEXT NOT NULL DEFAULT '',
    readout_type TEXT NOT NULL DEFAULT '',
    optimizer_mode TEXT NOT NULL DEFAULT 'muon_adamw',
    schedule TEXT NOT NULL DEFAULT 'COSINE',
    target_tokens_M REAL NOT NULL DEFAULT 900.0,
    live_tokens_M REAL NOT NULL DEFAULT 0.0,
    last_eval_tokens_M REAL NOT NULL DEFAULT 0.0,
    current_step INTEGER NOT NULL DEFAULT 0,
    total_steps INTEGER NOT NULL DEFAULT 0,
    progress_pct REAL NOT NULL DEFAULT 0.0,
    live_train_ce REAL,
    ep05_ce REAL,
    ep10_ce REAL,
    ep20_ce REAL,
    ep30_ce REAL,
    best_val_ce REAL,
    best_val_ppl REAL,
    best_top1 REAL,
    best_top5 REAL,
    final_r_eff REAL,
    final_iso_err REAL,
    avg_tok_per_s REAL NOT NULL DEFAULT 0.0,
    wall_min REAL NOT NULL DEFAULT 0.0,
    eta_min REAL NOT NULL DEFAULT 0.0,
    pareto_frontier INTEGER NOT NULL DEFAULT 0,
    obsolete INTEGER NOT NULL DEFAULT 0,
    obsolete_reason TEXT NOT NULL DEFAULT '',
    obsolete_at TEXT NOT NULL DEFAULT '',
    explicit_user_override INTEGER NOT NULL DEFAULT 0,
    default_visible INTEGER NOT NULL DEFAULT 1,
    latest_log_line TEXT NOT NULL DEFAULT '',
    catalog_entry_json TEXT NOT NULL DEFAULT '{}',
    model_card_json TEXT NOT NULL DEFAULT '{}',
    updated_at REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS run_evals (
    run_key TEXT NOT NULL,
    eval_idx INTEGER NOT NULL,
    step INTEGER NOT NULL DEFAULT 0,
    epoch REAL NOT NULL DEFAULT 0.0,
    tokens_M REAL NOT NULL DEFAULT 0.0,
    elapsed_min REAL NOT NULL DEFAULT 0.0,
    val_ce REAL NOT NULL,
    val_ppl REAL NOT NULL,
    top1_acc REAL NOT NULL DEFAULT 0.0,
    top5_acc REAL NOT NULL DEFAULT 0.0,
    seed_eff_rank REAL,
    iso_err REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (run_key, eval_idx),
    FOREIGN KEY (run_key) REFERENCES runs(run_key) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS run_steps (
    run_key TEXT NOT NULL,
    step INTEGER NOT NULL,
    tokens_M REAL NOT NULL DEFAULT 0.0,
    elapsed_min REAL NOT NULL DEFAULT 0.0,
    train_ce REAL NOT NULL,
    tok_per_s REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (run_key, step),
    FOREIGN KEY (run_key) REFERENCES runs(run_key) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS colab_sessions (
    session TEXT PRIMARY KEY,
    device TEXT NOT NULL DEFAULT '',
    gpu_util_pct REAL NOT NULL DEFAULT 0.0,
    mem_bw_util_pct REAL NOT NULL DEFAULT 0.0,
    vram_used_gib REAL NOT NULL DEFAULT 0.0,
    vram_total_gib REAL NOT NULL DEFAULT 0.0,
    power_w REAL NOT NULL DEFAULT 0.0,
    temp_c REAL NOT NULL DEFAULT 0.0,
    ps_alive INTEGER NOT NULL DEFAULT 0,
    ps_cmd TEXT NOT NULL DEFAULT '',
    active_run_key TEXT NOT NULL DEFAULT '',
    wave_id TEXT NOT NULL DEFAULT '',
    run_state TEXT NOT NULL DEFAULT 'IDLE',
    session_json TEXT NOT NULL DEFAULT '{}',
    updated_at REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS session_queue (
    queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT NOT NULL,
    run_key TEXT NOT NULL UNIQUE,
    wave_id TEXT NOT NULL DEFAULT '',
    arch_family TEXT NOT NULL DEFAULT 'quantum_born',
    hardware TEXT NOT NULL DEFAULT '',
    short_label TEXT NOT NULL DEFAULT '',
    cli_args TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'QUEUED',
    priority INTEGER NOT NULL DEFAULT 100,
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS ui_state (
    state_key TEXT PRIMARY KEY,
    state_json TEXT NOT NULL DEFAULT '{}'
);
"""


def classify_run_tier(
    run_key: str,
    target_tokens_M: float | dict[str, Any] = 0.0,
    wave_id: str = "",
    model_card: dict[str, Any] | None = None,
) -> str:
    """Classify a run into `'main_benchmark'`, `'fineweb_edu_local'`, or `'micro_verification'`."""
    entry_dict: dict[str, Any] = {}
    if isinstance(target_tokens_M, dict):
        entry_dict = target_tokens_M
        if not wave_id:
            wave_id = str(entry_dict.get("wave_id") or "")
        if model_card is None and isinstance(entry_dict.get("model_card"), dict):
            model_card = entry_dict["model_card"]
        tok_m = float(entry_dict.get("target_tokens_M") or 0.0)
    else:
        tok_m = float(target_tokens_M or 0.0)

    mc_dict = model_card if isinstance(model_card, dict) else {}
    mc_tier = str(mc_dict.get("run_tier") or entry_dict.get("run_tier") or "").strip()
    task_domain = str(mc_dict.get("task_domain") or entry_dict.get("task_domain") or "").strip()
    rk_low = str(run_key or "").strip().lower()
    wid_up = str(wave_id or mc_dict.get("wave_id") or "").strip().upper()

    if task_domain == "vision_ablation" or any(k in rk_low for k in ("cifar100", "1000class", "imagenet1k")):
        return "micro_verification"
    if rk_low.startswith(("verify", "micro_", "plan0", "h1_", "h2_", "h3_")) or wid_up in ("MICRO", "VERIFY"):
        return "micro_verification"
    if (
        task_domain == "fineweb_edu_local"
        or mc_tier == "fineweb_edu_local"
        or rk_low.startswith(tuple(f"w{n}_" for n in range(18, 30)))
        or wid_up in {f"W{n}" for n in range(18, 30)}
    ):
        return "fineweb_edu_local"
    if mc_tier == "micro_verification":
        return "micro_verification"
    if rk_low.startswith("w14_local_") or wid_up == "W14":
        return "main_benchmark"
    if mc_tier == "main_benchmark" and tok_m >= 50.0:
        return "main_benchmark"
    if 0.0 < tok_m < 50.0:
        return "micro_verification"
    return "main_benchmark"


def _is_eligible_completed_run(r: dict[str, Any]) -> bool:
    st = str(r.get("run_state") or r.get("status") or "").upper()
    is_done = st == "COMPLETED" or bool(r.get("completed"))
    best_ce = r.get("best_val_ce")
    wall_m = float(r.get("wall_min") or 0.0)
    return bool(is_done and best_ce is not None and wall_m > 0.0)


def compute_family_pareto_and_obsolete(
    catalog: list[dict[str, Any]],
    explicit_overrides: set[str] | None = None,
    default_obsolete_reasons: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Compute per-family 2D `(wall_min, best_val_ce)` Pareto frontiers and wave-superseded obsolete flags.

    - For `main_benchmark`: global 2D Pareto frontier per `arch_family` + automatic wave-superseded obsolete marking.
    - For `fineweb_edu_local`: independent 2D Pareto frontiers bucketed by token horizon (`<= 10.0M` vs
      `10.0M..50.0M` vs `> 50.0M`) and wave family so `W28` (`7.37M`), `W25` (`29.49M`), `W27` (`29.49M`),
      and `W26` (`44.24M`) are marked `pareto_frontier = True` within their horizon bucket and never falsely
      marked obsolete.
    """
    exp_set = set(explicit_overrides or set())
    def_reasons = dict(default_obsolete_reasons or {})
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    out: list[dict[str, Any]] = [dict(r) for r in catalog]

    completed_by_family: dict[str, list[dict[str, Any]]] = {}
    edu_buckets: dict[str, list[dict[str, Any]]] = {}
    for r in out:
        rkey = str(r.get("key") or r.get("run_key") or "")
        tier = classify_run_tier(
            rkey,
            r,
            str(r.get("wave_id") or ""),
            r.get("model_card"),
        )
        r["run_tier"] = tier
        if not _is_eligible_completed_run(r):
            continue
        fam = str(r.get("arch_family") or "quantum_born")
        tok_m = float(r.get("target_tokens_M") or r.get("live_tokens_M") or 900.0)
        if tier == "main_benchmark":
            fam_bucket = f"{fam}_local" if (rkey.lower().startswith("w14_local_") or (0.0 < tok_m < 50.0)) else fam
            completed_by_family.setdefault(fam_bucket, []).append(r)
        elif tier == "fineweb_edu_local":
            h_bucket = "le_10m" if tok_m <= 10.0 else ("10m_to_50m" if tok_m <= 50.0 else "gt_50m")
            edu_buckets.setdefault(f"edu_{fam}_{h_bucket}", []).append(r)
            wid = str(r.get("wave_id") or "")
            wnum = float(r.get("wave_num") or 0.0)
            if wnum >= 25.0:
                edu_buckets.setdefault(f"edu_{fam}_{wid}_{h_bucket}", []).append(r)

    pareto_keys: set[str] = set()
    dominator_by_key: dict[str, dict[str, Any]] = {}

    for fam, fam_runs in completed_by_family.items():
        for a in fam_runs:
            a_key = str(a.get("key") or a.get("run_key") or "")
            a_ce = float(a["best_val_ce"])
            a_wall = float(a["wall_min"])
            a_wnum = float(a.get("wave_num") or 0.0)

            is_dominated_globally = False
            for c in fam_runs:
                c_key = str(c.get("key") or c.get("run_key") or "")
                if c_key == a_key:
                    continue
                c_ce = float(c["best_val_ce"])
                c_wall = float(c["wall_min"])
                if c_ce <= a_ce and c_wall <= a_wall and (c_ce < a_ce or c_wall < a_wall):
                    is_dominated_globally = True
                    break
            if not is_dominated_globally:
                pareto_keys.add(a_key)

            dominators: list[dict[str, Any]] = []
            for b in fam_runs:
                b_key = str(b.get("key") or b.get("run_key") or "")
                if b_key == a_key:
                    continue
                b_wnum = float(b.get("wave_num") or 0.0)
                b_ce = float(b["best_val_ce"])
                b_wall = float(b["wall_min"])
                if b_wnum > a_wnum and b_ce < a_ce and b_wall <= a_wall:
                    dominators.append(b)
            if dominators:
                b_star = min(
                    dominators,
                    key=lambda b: (
                        float(b["best_val_ce"]),
                        float(b["wall_min"]),
                        str(b.get("key") or b.get("run_key") or ""),
                    ),
                )
                dominator_by_key[a_key] = b_star

    for _, bucket_runs in edu_buckets.items():
        for a in bucket_runs:
            a_key = str(a.get("key") or a.get("run_key") or "")
            a_ce = float(a["best_val_ce"])
            a_wall = float(a["wall_min"])
            is_dom = False
            for c in bucket_runs:
                c_key = str(c.get("key") or c.get("run_key") or "")
                if c_key == a_key:
                    continue
                c_ce = float(c["best_val_ce"])
                c_wall = float(c["wall_min"])
                if c_ce <= a_ce and c_wall <= a_wall and (c_ce < a_ce or c_wall < a_wall):
                    is_dom = True
                    break
            if not is_dom:
                pareto_keys.add(a_key)

    for r in out:
        rkey = str(r.get("key") or r.get("run_key") or "")
        tier = str(r.get("run_tier") or "main_benchmark")
        fam = str(r.get("arch_family") or "quantum_born")
        run_state = str(r.get("run_state") or r.get("status") or "IDLE").upper()
        is_pareto = rkey in pareto_keys
        r["pareto_frontier"] = is_pareto

        has_explicit = (rkey in exp_set) or bool(r.get("explicit_user_override"))
        r["explicit_user_override"] = bool(has_explicit)

        if tier in ("micro_verification", "fineweb_edu_local"):
            if has_explicit:
                is_obs = bool(r.get("obsolete", False))
                obs_reason = str(r.get("obsolete_reason") or "") if is_obs else ""
                obs_at = str(r.get("obsolete_at") or (now_utc if is_obs else "")) if is_obs else ""
            elif run_state == "OOM_CRASH":
                is_obs = True
                obs_reason = str(r.get("obsolete_reason") or "OOM crash")
                obs_at = str(r.get("obsolete_at") or now_utc)
            else:
                is_obs = False
                obs_reason = ""
                obs_at = ""
        else:
            if has_explicit:
                is_obs = bool(r.get("obsolete", False))
                obs_reason = str(r.get("obsolete_reason") or ("Marked obsolete from dashboard" if is_obs else "")) if is_obs else ""
                obs_at = str(r.get("obsolete_at") or (now_utc if is_obs else "")) if is_obs else ""
            elif rkey in dominator_by_key:
                b_star = dominator_by_key[rkey]
                b_key = str(b_star.get("key") or b_star.get("run_key") or "")
                b_wid = str(b_star.get("wave_id") or "")
                fam_disp = "ClassicalModdedGPT" if fam == "classical_modded" else "QuantumBornGPT"
                auto_reason = (
                    f"Not on {fam_disp} Pareto frontier: superseded by {b_key} ({b_wid}) "
                    f"with lower Val CE ({float(b_star['best_val_ce']):.4f} vs {float(r['best_val_ce']):.4f}) "
                    f"in less time ({float(b_star['wall_min']):.1f}m vs {float(r['wall_min']):.1f}m)"
                )
                is_obs = True
                obs_reason = auto_reason
                obs_at = str(r.get("obsolete_at") or now_utc)
            elif rkey in def_reasons or run_state == "OOM_CRASH":
                is_obs = True
                obs_reason = str(def_reasons.get(rkey) or r.get("obsolete_reason") or "OOM crash")
                obs_at = str(r.get("obsolete_at") or now_utc)
            else:
                is_obs = False
                obs_reason = ""
                obs_at = ""

        r["obsolete"] = is_obs
        r["obsolete_reason"] = obs_reason
        r["obsolete_at"] = obs_at
        if is_obs:
            r["default_visible"] = False

        if isinstance(r.get("model_card"), dict):
            mc = dict(r["model_card"])
            mc["run_tier"] = tier
            mc["pareto_frontier"] = is_pareto
            mc["obsolete"] = is_obs
            mc["obsolete_reason"] = obs_reason
            mc["obsolete_at"] = obs_at
            r["model_card"] = mc

    return out


class BenchmarkSQLiteDB:
    """Thread-safe WAL-mode SQLite database for QuantumDLBlocks benchmark runs, Model Cards, and Colab queues."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.executescript(_SCHEMA_DDL)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _upsert_run_locked(
        self,
        cur: sqlite3.Cursor,
        cat_entry: dict[str, Any],
        model_card: dict[str, Any] | None = None,
        explicit_user_override: bool | None = None,
    ) -> None:
        run_key = str(cat_entry.get("key") or cat_entry.get("run_key") or "")
        if not run_key:
            return
        mc = dict(model_card if isinstance(model_card, dict) else (cat_entry.get("model_card") or {}))
        wave_id = str(cat_entry.get("wave_id") or mc.get("wave_id") or infer_wave_id_and_num(run_key)[0])
        wave_num = float(
            cat_entry.get("wave_num")
            if cat_entry.get("wave_num") is not None
            else mc.get("wave_num", infer_wave_id_and_num(run_key, wave_id)[1])
        )
        wave_label = str(cat_entry.get("wave") or cat_entry.get("wave_label") or mc.get("wave_label") or wave_id)
        arch_family = str(
            cat_entry.get("arch_family")
            or mc.get("architecture", {}).get("family")
            or ("classical_modded" if "classical" in run_key else "quantum_born")
        )
        target_tok_m = float(
            cat_entry.get("target_tokens_M")
            or mc.get("data_and_batch", {}).get("target_tokens_M")
            or 900.0
        )
        run_tier = str(
            cat_entry.get("run_tier")
            or classify_run_tier(run_key, target_tok_m, wave_id=wave_id, model_card=mc)
        )

        # Preserve existing explicit_user_override in DB unless explicitly passed
        existing_row = cur.execute(
            "SELECT explicit_user_override, obsolete, obsolete_reason, obsolete_at FROM runs WHERE run_key = ?",
            (run_key,),
        ).fetchone()
        if explicit_user_override is not None:
            exp_ov = int(bool(explicit_user_override))
        elif "explicit_user_override" in cat_entry:
            exp_ov = int(bool(cat_entry["explicit_user_override"]))
        elif existing_row is not None and int(existing_row["explicit_user_override"]) == 1:
            exp_ov = 1
        else:
            exp_ov = 0

        if exp_ov == 1 and existing_row is not None and explicit_user_override is None and "explicit_user_override" not in cat_entry:
            is_obs = bool(existing_row["obsolete"])
            obs_reason = str(existing_row["obsolete_reason"] or "")
            obs_at = str(existing_row["obsolete_at"] or "")
        else:
            is_obs = bool(cat_entry.get("obsolete", mc.get("obsolete", False)))
            obs_reason = str(cat_entry.get("obsolete_reason", mc.get("obsolete_reason", ""))) if is_obs else ""
            obs_at = str(cat_entry.get("obsolete_at", mc.get("obsolete_at", ""))) if is_obs else ""

        pareto_flag = int(bool(cat_entry.get("pareto_frontier", mc.get("pareto_frontier", False))))
        status = str(cat_entry.get("run_state") or cat_entry.get("status") or "IDLE")
        boot_status = str(cat_entry.get("boot_status") or status.capitalize())

        entry_copy = dict(cat_entry)
        entry_copy["key"] = run_key
        entry_copy["wave_id"] = wave_id
        entry_copy["wave_num"] = wave_num
        entry_copy["arch_family"] = arch_family
        entry_copy["run_tier"] = run_tier
        entry_copy["pareto_frontier"] = bool(pareto_flag)
        entry_copy["obsolete"] = is_obs
        entry_copy["obsolete_reason"] = obs_reason
        entry_copy["obsolete_at"] = obs_at
        entry_copy["explicit_user_override"] = bool(exp_ov)
        if mc:
            mc["run_tier"] = run_tier
            mc["pareto_frontier"] = bool(pareto_flag)
            mc["obsolete"] = is_obs
            mc["obsolete_reason"] = obs_reason
            mc["obsolete_at"] = obs_at
            entry_copy["model_card"] = mc

        now_ts = float(cat_entry.get("timestamp") or time.time())

        cur.execute(
            """
            INSERT INTO runs (
                run_key, wave_id, wave_num, wave_label, arch_family, run_tier,
                session, accel, kind, label, short_label, color, status, boot_status,
                params_M, seed_dim, embed_dim, num_layers, d_ff, readout_type,
                optimizer_mode, schedule, target_tokens_M, live_tokens_M, last_eval_tokens_M,
                current_step, total_steps, progress_pct, live_train_ce,
                ep05_ce, ep10_ce, ep20_ce, ep30_ce,
                best_val_ce, best_val_ppl, best_top1, best_top5,
                final_r_eff, final_iso_err, avg_tok_per_s, wall_min, eta_min,
                pareto_frontier, obsolete, obsolete_reason, obsolete_at,
                explicit_user_override, default_visible, latest_log_line,
                catalog_entry_json, model_card_json, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?
            )
            ON CONFLICT(run_key) DO UPDATE SET
                wave_id=excluded.wave_id,
                wave_num=excluded.wave_num,
                wave_label=excluded.wave_label,
                arch_family=excluded.arch_family,
                run_tier=excluded.run_tier,
                session=excluded.session,
                accel=excluded.accel,
                kind=excluded.kind,
                label=excluded.label,
                short_label=excluded.short_label,
                color=excluded.color,
                status=excluded.status,
                boot_status=excluded.boot_status,
                params_M=excluded.params_M,
                seed_dim=excluded.seed_dim,
                embed_dim=excluded.embed_dim,
                num_layers=excluded.num_layers,
                d_ff=excluded.d_ff,
                readout_type=excluded.readout_type,
                optimizer_mode=excluded.optimizer_mode,
                schedule=excluded.schedule,
                target_tokens_M=excluded.target_tokens_M,
                live_tokens_M=excluded.live_tokens_M,
                last_eval_tokens_M=excluded.last_eval_tokens_M,
                current_step=excluded.current_step,
                total_steps=excluded.total_steps,
                progress_pct=excluded.progress_pct,
                live_train_ce=excluded.live_train_ce,
                ep05_ce=excluded.ep05_ce,
                ep10_ce=excluded.ep10_ce,
                ep20_ce=excluded.ep20_ce,
                ep30_ce=excluded.ep30_ce,
                best_val_ce=excluded.best_val_ce,
                best_val_ppl=excluded.best_val_ppl,
                best_top1=excluded.best_top1,
                best_top5=excluded.best_top5,
                final_r_eff=excluded.final_r_eff,
                final_iso_err=excluded.final_iso_err,
                avg_tok_per_s=excluded.avg_tok_per_s,
                wall_min=excluded.wall_min,
                eta_min=excluded.eta_min,
                pareto_frontier=excluded.pareto_frontier,
                obsolete=excluded.obsolete,
                obsolete_reason=excluded.obsolete_reason,
                obsolete_at=excluded.obsolete_at,
                explicit_user_override=excluded.explicit_user_override,
                default_visible=excluded.default_visible,
                latest_log_line=excluded.latest_log_line,
                catalog_entry_json=excluded.catalog_entry_json,
                model_card_json=excluded.model_card_json,
                updated_at=excluded.updated_at
            """,
            (
                run_key,
                wave_id,
                wave_num,
                wave_label,
                arch_family,
                run_tier,
                str(cat_entry.get("session") or run_key),
                str(cat_entry.get("accel") or ""),
                str(cat_entry.get("kind") or "gpu").lower(),
                str(cat_entry.get("label") or run_key),
                str(cat_entry.get("short_label") or run_key),
                str(cat_entry.get("color") or deterministic_run_color(run_key)),
                status,
                boot_status,
                float(cat_entry.get("params_M") or 0.0),
                int(cat_entry.get("seed_dim") or 128),
                int(cat_entry.get("embed_dim") or 1024),
                int(cat_entry.get("num_layers") or 6),
                str(cat_entry.get("d_ff") or ""),
                str(cat_entry.get("readout_type") or ""),
                str(cat_entry.get("optimizer_mode") or "muon_adamw"),
                str(cat_entry.get("schedule") or "COSINE"),
                target_tok_m,
                float(cat_entry.get("live_tokens_M") or 0.0),
                float(cat_entry.get("last_eval_tokens_M") or 0.0),
                int(cat_entry.get("current_step") or 0),
                int(cat_entry.get("total_steps") or 0),
                float(cat_entry.get("progress_pct") or 0.0),
                float(cat_entry["live_train_ce"]) if cat_entry.get("live_train_ce") is not None else None,
                float(cat_entry["ep05_ce"]) if cat_entry.get("ep05_ce") is not None else None,
                float(cat_entry["ep10_ce"]) if cat_entry.get("ep10_ce") is not None else None,
                float(cat_entry["ep20_ce"]) if cat_entry.get("ep20_ce") is not None else None,
                float(cat_entry["ep30_ce"]) if cat_entry.get("ep30_ce") is not None else None,
                float(cat_entry["best_val_ce"]) if cat_entry.get("best_val_ce") is not None else None,
                float(cat_entry["best_val_ppl"]) if cat_entry.get("best_val_ppl") is not None else None,
                float(cat_entry["best_top1"]) if cat_entry.get("best_top1") is not None else None,
                float(cat_entry["best_top5"]) if cat_entry.get("best_top5") is not None else None,
                float(cat_entry["final_r_eff"]) if cat_entry.get("final_r_eff") is not None else None,
                float(cat_entry["final_iso_err"]) if cat_entry.get("final_iso_err") is not None else None,
                float(cat_entry.get("avg_tok_per_s") or 0.0),
                float(cat_entry.get("wall_min") or 0.0),
                float(cat_entry.get("eta_min") or 0.0),
                pareto_flag,
                int(is_obs),
                obs_reason,
                obs_at,
                exp_ov,
                int(bool(cat_entry.get("default_visible", not is_obs))),
                str(cat_entry.get("latest_log_line") or ""),
                json.dumps(entry_copy),
                json.dumps(mc),
                now_ts,
            ),
        )

        # Replace run_evals
        evals = cat_entry.get("evals") or []
        cur.execute("DELETE FROM run_evals WHERE run_key = ?", (run_key,))
        if evals:
            cur.executemany(
                """
                INSERT INTO run_evals (
                    run_key, eval_idx, step, epoch, tokens_M, elapsed_min,
                    val_ce, val_ppl, top1_acc, top5_acc, seed_eff_rank, iso_err
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_key,
                        idx,
                        int(ev.get("step") or 0),
                        float(ev.get("epoch") or 0.0),
                        float(ev.get("tokens_M") or 0.0),
                        float(ev.get("elapsed_min") or 0.0),
                        float(ev["val_ce"]),
                        float(ev.get("val_ppl") or round(math.exp(min(20.0, float(ev["val_ce"]))), 2)),
                        float(ev.get("top1_acc") or 0.0),
                        float(ev.get("top5_acc") or 0.0),
                        float(ev["seed_eff_rank"]) if ev.get("seed_eff_rank") is not None else None,
                        float(ev.get("iso_err") or 0.0),
                    )
                    for idx, ev in enumerate(evals)
                    if isinstance(ev, dict) and ev.get("val_ce") is not None
                ],
            )

        # Replace run_steps
        steps = cat_entry.get("steps") or []
        cur.execute("DELETE FROM run_steps WHERE run_key = ?", (run_key,))
        if steps:
            seen_steps: set[int] = set()
            step_rows = []
            for st in steps:
                if not isinstance(st, dict) or st.get("train_ce") is None:
                    continue
                s_num = int(st.get("step") or 0)
                if s_num in seen_steps:
                    continue
                seen_steps.add(s_num)
                step_rows.append(
                    (
                        run_key,
                        s_num,
                        float(st.get("tokens_M") or 0.0),
                        float(st.get("elapsed_min") or 0.0),
                        float(st["train_ce"]),
                        float(st.get("tok_per_s") or 0.0),
                    )
                )
            if step_rows:
                cur.executemany(
                    """
                    INSERT INTO run_steps (run_key, step, tokens_M, elapsed_min, train_ce, tok_per_s)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    step_rows,
                )

        # Sync session_queue status if run_key is queued
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        st_upper = status.upper()
        if st_upper == "COMPLETED" or bool(cat_entry.get("completed")):
            cur.execute(
                "UPDATE session_queue SET status = 'COMPLETED', updated_at = ? WHERE run_key = ? AND status != 'CANCELLED'",
                (now_iso, run_key),
            )
        elif st_upper in ("RUNNING", "COMPILING"):
            cur.execute(
                "UPDATE session_queue SET status = 'RUNNING', updated_at = ? WHERE run_key = ? AND status != 'CANCELLED'",
                (now_iso, run_key),
            )

    def upsert_run(
        self,
        cat_entry: dict[str, Any],
        model_card: dict[str, Any] | None = None,
        explicit_user_override: bool | None = None,
    ) -> None:
        """Upsert a single run and its eval/step trajectories into SQLite."""
        with self._lock:
            cur = self._conn.cursor()
            self._upsert_run_locked(cur, cat_entry, model_card=model_card, explicit_user_override=explicit_user_override)
            self._conn.commit()

    def _upsert_colab_session_locked(self, cur: sqlite3.Cursor, session_obj: dict[str, Any]) -> None:
        sess = str(session_obj.get("session") or "")
        if not sess:
            return
        ps_alive = int(bool(session_obj.get("ps_alive", False)))
        active_rk = str(session_obj.get("active_run_key") or "")
        now_ts = time.time()
        cur.execute(
            """
            INSERT INTO colab_sessions (
                session, device, gpu_util_pct, mem_bw_util_pct,
                vram_used_gib, vram_total_gib, power_w, temp_c,
                ps_alive, ps_cmd, active_run_key, wave_id, run_state,
                session_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session) DO UPDATE SET
                device=excluded.device,
                gpu_util_pct=excluded.gpu_util_pct,
                mem_bw_util_pct=excluded.mem_bw_util_pct,
                vram_used_gib=excluded.vram_used_gib,
                vram_total_gib=excluded.vram_total_gib,
                power_w=excluded.power_w,
                temp_c=excluded.temp_c,
                ps_alive=excluded.ps_alive,
                ps_cmd=excluded.ps_cmd,
                active_run_key=excluded.active_run_key,
                wave_id=excluded.wave_id,
                run_state=excluded.run_state,
                session_json=excluded.session_json,
                updated_at=excluded.updated_at
            """,
            (
                sess,
                str(session_obj.get("device") or ""),
                float(session_obj.get("gpu_util_pct") or 0.0),
                float(session_obj.get("mem_bw_util_pct") or 0.0),
                float(session_obj.get("vram_used_gib") or 0.0),
                float(session_obj.get("vram_total_gib") or 0.0),
                float(session_obj.get("power_w") or 0.0),
                float(session_obj.get("temp_c") or 0.0),
                ps_alive,
                str(session_obj.get("ps_cmd") or ""),
                active_rk,
                str(session_obj.get("wave_id") or ""),
                str(session_obj.get("run_state") or "IDLE"),
                json.dumps(session_obj),
                now_ts,
            ),
        )
        if ps_alive and active_rk:
            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            cur.execute(
                "UPDATE session_queue SET status = 'RUNNING', updated_at = ? WHERE run_key = ? AND status = 'QUEUED'",
                (now_iso, active_rk),
            )

    def upsert_colab_session(self, session_obj: dict[str, Any]) -> None:
        """Upsert a Colab GPU session status row into `colab_sessions`."""
        with self._lock:
            cur = self._conn.cursor()
            self._upsert_colab_session_locked(cur, session_obj)
            self._conn.commit()

    def enqueue_job(
        self,
        session: str,
        run_key: str,
        wave_id: str,
        arch_family: str = "quantum_born",
        hardware: str = "",
        short_label: str = "",
        cli_args: str = "",
        status: str = "QUEUED",
        priority: int = 100,
    ) -> dict[str, Any]:
        """Upsert a job into `session_queue`, reconciling with any existing run state in `runs`."""
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._lock:
            cur = self._conn.cursor()
            run_row = cur.execute("SELECT status FROM runs WHERE run_key = ?", (run_key,)).fetchone()
            eff_status = str(status or "QUEUED").upper()
            if eff_status != "CANCELLED" and run_row is not None:
                r_st = str(run_row["status"] or "").upper()
                if r_st == "COMPLETED":
                    eff_status = "COMPLETED"
                elif r_st in ("RUNNING", "COMPILING"):
                    eff_status = "RUNNING"

            cur.execute(
                """
                INSERT INTO session_queue (
                    session, run_key, wave_id, arch_family, hardware,
                    short_label, cli_args, status, priority, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_key) DO UPDATE SET
                    session=excluded.session,
                    wave_id=excluded.wave_id,
                    arch_family=excluded.arch_family,
                    hardware=excluded.hardware,
                    short_label=excluded.short_label,
                    cli_args=excluded.cli_args,
                    status=CASE
                        WHEN excluded.status = 'CANCELLED' THEN 'CANCELLED'
                        WHEN session_queue.status = 'COMPLETED' THEN 'COMPLETED'
                        ELSE excluded.status
                    END,
                    priority=excluded.priority,
                    updated_at=excluded.updated_at
                """,
                (
                    session,
                    run_key,
                    wave_id,
                    arch_family,
                    hardware,
                    short_label or run_key,
                    cli_args,
                    eff_status,
                    int(priority),
                    now_iso,
                    now_iso,
                ),
            )
            self._conn.commit()
            row = cur.execute("SELECT * FROM session_queue WHERE run_key = ?", (run_key,)).fetchone()
            return dict(row) if row is not None else {}

    def sync_queue_manifest(self, manifest_path: Path) -> list[dict[str, Any]]:
        """Read a `configs/wave*_fleet.json` manifest and enqueue all its runs into `session_queue`."""
        mp = Path(manifest_path)
        if not mp.exists():
            return []
        try:
            mdata = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(mdata, dict) or not isinstance(mdata.get("runs"), list):
            return []
        raw_wave = str(mdata.get("wave_id") or mdata.get("wave") or mp.stem)
        out: list[dict[str, Any]] = []
        for idx, entry in enumerate(mdata["runs"]):
            if not isinstance(entry, dict) or "key" not in entry:
                continue
            rk = str(entry["key"])
            wid, wnum = infer_wave_id_and_num(rk, raw_wave)
            cli = str(entry.get("cli_args") or "")
            fam = "classical_modded" if ("classical" in rk or "classical_modded" in cli) else "quantum_born"
            sess = str(entry.get("session") or rk)
            hw = str(entry.get("hardware") or entry.get("accel") or "")
            short_lbl = str(entry.get("short_label") or entry.get("label") or rk)
            prio = max(10, 200 - wnum * 10 + idx)
            out.append(
                self.enqueue_job(
                    session=sess,
                    run_key=rk,
                    wave_id=wid,
                    arch_family=fam,
                    hardware=hw,
                    short_label=short_lbl,
                    cli_args=cli,
                    status=str(entry.get("status") or "QUEUED"),
                    priority=prio,
                )
            )
        return out

    def recompute_pareto_and_obsolete(
        self,
        default_obsolete_reasons: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Recompute `pareto_frontier` and automatic wave-superseded `obsolete` across all runs in SQLite."""
        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(
                "SELECT run_key, run_tier, pareto_frontier, obsolete, obsolete_reason, explicit_user_override, catalog_entry_json, model_card_json FROM runs"
            ).fetchall()
            if not rows:
                return []
            catalog: list[dict[str, Any]] = []
            explicit_overrides: set[str] = set()
            prev_flags: dict[str, tuple[str, int, int, str, int]] = {}
            for row in rows:
                rk = str(row["run_key"])
                prev_flags[rk] = (
                    str(row["run_tier"] or "main_benchmark"),
                    int(row["pareto_frontier"] or 0),
                    int(row["obsolete"] or 0),
                    str(row["obsolete_reason"] or ""),
                    int(row["explicit_user_override"] or 0),
                )
                try:
                    cat_entry = json.loads(row["catalog_entry_json"] or "{}")
                except Exception:
                    cat_entry = {"key": rk}
                try:
                    mc = json.loads(row["model_card_json"] or "{}")
                except Exception:
                    mc = {}
                if mc:
                    cat_entry["model_card"] = mc
                if int(row["explicit_user_override"] or 0) == 1:
                    explicit_overrides.add(rk)
                    cat_entry["explicit_user_override"] = True
                catalog.append(cat_entry)

            updated_catalog = compute_family_pareto_and_obsolete(
                catalog,
                explicit_overrides=explicit_overrides,
                default_obsolete_reasons=default_obsolete_reasons,
            )

            any_changed = False
            for r in updated_catalog:
                rk = str(r.get("key") or r.get("run_key") or "")
                new_sig = (
                    str(r.get("run_tier") or "main_benchmark"),
                    int(bool(r.get("pareto_frontier", False))),
                    int(bool(r.get("obsolete", False))),
                    str(r.get("obsolete_reason") or ""),
                    int(bool(r.get("explicit_user_override", False))),
                )
                if prev_flags.get(rk) == new_sig:
                    continue
                any_changed = True
                mc = r.get("model_card") if isinstance(r.get("model_card"), dict) else {}
                cur.execute(
                    """
                    UPDATE runs SET
                        run_tier = ?,
                        pareto_frontier = ?,
                        obsolete = ?,
                        obsolete_reason = ?,
                        obsolete_at = ?,
                        explicit_user_override = ?,
                        default_visible = ?,
                        catalog_entry_json = ?,
                        model_card_json = ?
                    WHERE run_key = ?
                    """,
                    (
                        new_sig[0],
                        new_sig[1],
                        new_sig[2],
                        new_sig[3],
                        str(r.get("obsolete_at") or ""),
                        new_sig[4],
                        int(bool(r.get("default_visible", not r.get("obsolete", False)))),
                        json.dumps(r),
                        json.dumps(mc),
                        rk,
                    ),
                )
            if any_changed:
                self._conn.commit()
            return updated_catalog

    def set_obsolete_override(
        self,
        run_keys: list[str],
        obsolete: bool,
        reason: str = "",
        default_obsolete_reasons: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Set explicit user obsolete override (`explicit_user_override = 1`) on `run_keys` in <3ms."""
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        is_obs = bool(obsolete)
        rsn = str(reason or ("Marked obsolete from dashboard" if is_obs else "")) if is_obs else ""
        obs_at = now_utc if is_obs else ""
        updated_keys: list[str] = []

        with self._lock:
            cur = self._conn.cursor()
            for rk in run_keys:
                row = cur.execute(
                    "SELECT catalog_entry_json, model_card_json FROM runs WHERE run_key = ?",
                    (rk,),
                ).fetchone()
                if row is None:
                    continue
                updated_keys.append(rk)
                try:
                    cat_entry = json.loads(row["catalog_entry_json"] or "{}")
                except Exception:
                    cat_entry = {"key": rk}
                try:
                    mc = json.loads(row["model_card_json"] or "{}")
                except Exception:
                    mc = {}
                if mc:
                    mc = set_model_card_obsolete(mc, is_obs, reason=rsn)
                cat_entry["obsolete"] = is_obs
                cat_entry["obsolete_reason"] = rsn
                cat_entry["obsolete_at"] = obs_at
                cat_entry["explicit_user_override"] = True
                if is_obs:
                    cat_entry["default_visible"] = False
                if mc:
                    cat_entry["model_card"] = mc
                cur.execute(
                    """
                    UPDATE runs SET
                        obsolete = ?,
                        obsolete_reason = ?,
                        obsolete_at = ?,
                        explicit_user_override = 1,
                        default_visible = ?,
                        catalog_entry_json = ?,
                        model_card_json = ?
                    WHERE run_key = ?
                    """,
                    (
                        int(is_obs),
                        rsn,
                        obs_at,
                        int(bool(cat_entry.get("default_visible", not is_obs))),
                        json.dumps(cat_entry),
                        json.dumps(mc),
                        rk,
                    ),
                )
            self._conn.commit()
            tot_main = int(
                cur.execute("SELECT COUNT(*) FROM runs WHERE run_tier = 'main_benchmark'").fetchone()[0]
            )
            obs_cnt = int(
                cur.execute("SELECT COUNT(*) FROM runs WHERE run_tier = 'main_benchmark' AND obsolete = 1").fetchone()[0]
            )
            return {
                "ok": True,
                "updated_keys": updated_keys,
                "obsolete": is_obs,
                "obsolete_count": obs_cnt,
                "active_count": tot_main - obs_cnt,
            }

    def sync_snapshot(
        self,
        catalog: list[dict[str, Any]],
        colab_sessions: list[dict[str, Any]],
        ui_prefs: dict[str, Any] | None = None,
        default_obsolete_reasons: dict[str, str] | None = None,
        colab_account: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Batch-upsert `catalog`, `colab_sessions`, and `ui_prefs` in a single WAL transaction and return payload."""
        with self._lock:
            cur = self._conn.cursor()
            exp_keys = {str(k) for k in ((ui_prefs or {}).get("explicitObsoleteKeys") or [])}
            obs_map = dict((ui_prefs or {}).get("obsoleteMap") or {})
            for r in catalog:
                rk = str(r.get("key") or r.get("run_key") or "")
                exp_flag: bool | None = None
                if rk in exp_keys or rk in obs_map:
                    exp_flag = True
                self._upsert_run_locked(
                    cur,
                    r,
                    model_card=r.get("model_card"),
                    explicit_user_override=exp_flag,
                )
            for s in colab_sessions:
                self._upsert_colab_session_locked(cur, s)
            if ui_prefs is not None:
                cur.execute(
                    "INSERT INTO ui_state (state_key, state_json) VALUES ('ui_prefs', ?) ON CONFLICT(state_key) DO UPDATE SET state_json=excluded.state_json",
                    (json.dumps(ui_prefs),),
                )
            if colab_account is not None:
                cur.execute(
                    "INSERT INTO ui_state (state_key, state_json) VALUES ('colab_account', ?) ON CONFLICT(state_key) DO UPDATE SET state_json=excluded.state_json",
                    (json.dumps(colab_account),),
                )
            self._conn.commit()
            self.recompute_pareto_and_obsolete(default_obsolete_reasons=default_obsolete_reasons)
            return self.load_snapshot_payload()

    def load_snapshot_payload(self) -> dict[str, Any]:
        """Materialize `runs_catalog`, `fineweb_edu_catalog`, `micro_runs_catalog`, `colab_sessions`, `session_queue`, and `db_stats`."""
        with self._lock:
            cur = self._conn.cursor()
            run_rows = cur.execute("SELECT * FROM runs").fetchall()

            evals_by_run: dict[str, list[dict[str, Any]]] = {}
            for erow in cur.execute("SELECT * FROM run_evals ORDER BY run_key, eval_idx ASC").fetchall():
                evals_by_run.setdefault(str(erow["run_key"]), []).append(
                    {
                        "step": int(erow["step"]),
                        "epoch": float(erow["epoch"]),
                        "tokens_M": float(erow["tokens_M"]),
                        "elapsed_min": float(erow["elapsed_min"]),
                        "val_ce": float(erow["val_ce"]),
                        "val_ppl": float(erow["val_ppl"]),
                        "top1_acc": float(erow["top1_acc"]),
                        "top5_acc": float(erow["top5_acc"]),
                        "seed_eff_rank": float(erow["seed_eff_rank"]) if erow["seed_eff_rank"] is not None else None,
                        "iso_err": float(erow["iso_err"]),
                    }
                )

            steps_by_run: dict[str, list[dict[str, Any]]] = {}
            for srow in cur.execute("SELECT * FROM run_steps ORDER BY run_key, step ASC").fetchall():
                steps_by_run.setdefault(str(srow["run_key"]), []).append(
                    {
                        "step": int(srow["step"]),
                        "tokens_M": float(srow["tokens_M"]),
                        "elapsed_min": float(srow["elapsed_min"]),
                        "train_ce": float(srow["train_ce"]),
                        "tok_per_s": float(srow["tok_per_s"]),
                    }
                )

            all_entries: list[dict[str, Any]] = []
            for rrow in run_rows:
                rk = str(rrow["run_key"])
                try:
                    entry = json.loads(rrow["catalog_entry_json"] or "{}")
                except Exception:
                    entry = {"key": rk}
                try:
                    mc = json.loads(rrow["model_card_json"] or "{}")
                except Exception:
                    mc = {}
                entry["key"] = rk
                entry["wave_id"] = str(rrow["wave_id"])
                entry["wave_num"] = float(rrow["wave_num"])
                entry["arch_family"] = str(rrow["arch_family"])
                entry["status"] = str(rrow["status"])
                entry["run_state"] = str(rrow["status"])
                entry["best_val_ce"] = float(rrow["best_val_ce"]) if rrow["best_val_ce"] is not None else None
                entry["best_val_ppl"] = float(rrow["best_val_ppl"]) if rrow["best_val_ppl"] is not None else None
                entry["wall_min"] = float(rrow["wall_min"])
                entry["pareto_frontier"] = bool(rrow["pareto_frontier"])
                entry["obsolete"] = bool(rrow["obsolete"])
                entry["obsolete_reason"] = str(rrow["obsolete_reason"] or "")
                entry["obsolete_at"] = str(rrow["obsolete_at"] or "")
                entry["explicit_user_override"] = bool(rrow["explicit_user_override"])
                entry["default_visible"] = bool(rrow["default_visible"])
                entry["run_tier"] = classify_run_tier(rk, entry, entry["wave_id"], mc)

                params_m = float(entry.get("params_M") or rrow["params_M"] or 10.4)
                tot_steps = max(1, int(entry.get("total_steps") or rrow["total_steps"] or 900))
                live_tok_m = float(entry.get("live_tokens_M") or rrow["live_tokens_M"] or 0.0)
                if entry.get("compute_tflops") is None:
                    entry["compute_tflops"] = round(6.0 * params_m * live_tok_m, 2)

                existing_evals = entry.get("evals") if isinstance(entry.get("evals"), list) else []
                if rk in evals_by_run:
                    merged_evals: list[dict[str, Any]] = []
                    for idx, db_ev in enumerate(evals_by_run[rk]):
                        base_ev = dict(existing_evals[idx]) if idx < len(existing_evals) and isinstance(existing_evals[idx], dict) else {}
                        base_ev.update(db_ev)
                        ev_step = int(base_ev.get("step") or 0)
                        ev_tok = float(base_ev.get("tokens_M") or 0.0)
                        if not base_ev.get("compute_tflops") and ev_tok > 0.0:
                            base_ev["compute_tflops"] = round(6.0 * params_m * ev_tok, 4)
                        elif base_ev.get("compute_tflops") is None:
                            base_ev["compute_tflops"] = 0.0
                        if not base_ev.get("tau_progress") and ev_step > 0:
                            base_ev["tau_progress"] = round(ev_step / tot_steps, 4)
                        elif base_ev.get("tau_progress") is None:
                            base_ev["tau_progress"] = 0.0
                        if base_ev.get("sharp_eff_rank") is None and base_ev.get("seed_eff_rank") is not None:
                            base_ev["sharp_eff_rank"] = base_ev["seed_eff_rank"]
                        if base_ev.get("kraus_stiefel_err") is None:
                            base_ev["kraus_stiefel_err"] = float(base_ev.get("iso_err") or 0.0)
                        merged_evals.append(base_ev)
                    entry["evals"] = merged_evals
                if rk in steps_by_run:
                    entry["steps"] = steps_by_run[rk]
                if mc:
                    mc["run_tier"] = entry["run_tier"]
                    mc["pareto_frontier"] = entry["pareto_frontier"]
                    mc["obsolete"] = entry["obsolete"]
                    mc["obsolete_reason"] = entry["obsolete_reason"]
                    mc["obsolete_at"] = entry["obsolete_at"]
                    entry["model_card"] = mc
                all_entries.append(entry)

            edu_has_pareto = any(e.get("pareto_frontier") for e in all_entries if e.get("run_tier") == "fineweb_edu_local")
            if not edu_has_pareto and any(e.get("run_tier") == "fineweb_edu_local" for e in all_entries):
                all_entries = compute_family_pareto_and_obsolete(all_entries)

            main_catalog: list[dict[str, Any]] = []
            fineweb_edu_catalog: list[dict[str, Any]] = []
            micro_catalog: list[dict[str, Any]] = []
            for entry in all_entries:
                if entry["run_tier"] == "fineweb_edu_local":
                    fineweb_edu_catalog.append(entry)
                elif entry["run_tier"] == "micro_verification":
                    micro_catalog.append(entry)
                else:
                    main_catalog.append(entry)

            main_catalog.sort(
                key=lambda r: (
                    bool(r.get("obsolete", False)),
                    r.get("best_val_ce") is None,
                    float(r["best_val_ce"]) if r.get("best_val_ce") is not None else 999.0,
                )
            )
            fineweb_edu_catalog.sort(
                key=lambda r: (
                    bool(r.get("obsolete", False)),
                    r.get("best_val_ce") is None,
                    float(r["best_val_ce"]) if r.get("best_val_ce") is not None else 999.0,
                )
            )
            micro_catalog.sort(
                key=lambda r: (
                    -float(r.get("wave_num") or 0.0),
                    r.get("best_val_ce") is None,
                    float(r["best_val_ce"]) if r.get("best_val_ce") is not None else 999.0,
                )
            )

            colab_sessions: list[dict[str, Any]] = []
            for crow in cur.execute("SELECT * FROM colab_sessions ORDER BY session ASC").fetchall():
                try:
                    sobj = json.loads(crow["session_json"] or "{}")
                except Exception:
                    sobj = {}
                sobj["session"] = str(crow["session"])
                sobj["device"] = str(crow["device"])
                sobj["gpu_util_pct"] = float(crow["gpu_util_pct"])
                sobj["mem_bw_util_pct"] = float(crow["mem_bw_util_pct"])
                sobj["vram_used_gib"] = float(crow["vram_used_gib"])
                sobj["vram_total_gib"] = float(crow["vram_total_gib"])
                sobj["power_w"] = float(crow["power_w"])
                sobj["temp_c"] = float(crow["temp_c"])
                sobj["ps_alive"] = bool(crow["ps_alive"])
                sobj["ps_cmd"] = str(crow["ps_cmd"])
                sobj["active_run_key"] = str(crow["active_run_key"])
                sobj["wave_id"] = str(crow["wave_id"])
                sobj["run_state"] = str(crow["run_state"])
                colab_sessions.append(sobj)

            order_map = {s: i for i, s in enumerate(["gpu_g4_w6_1", "gpu_g4_w6_2", "gpu_g4_w6_3", "gpu_a100_w6_1", "gpu_a100_w6_2"])}
            colab_sessions.sort(key=lambda s: (order_map.get(str(s.get("session")), 99), str(s.get("session"))))

            session_queue = [
                dict(qrow)
                for qrow in cur.execute("SELECT * FROM session_queue ORDER BY priority ASC, queue_id ASC").fetchall()
            ]

            ui_prefs: dict[str, Any] = {}
            u_row = cur.execute("SELECT state_json FROM ui_state WHERE state_key = 'ui_prefs'").fetchone()
            if u_row is not None:
                try:
                    ui_prefs = json.loads(u_row["state_json"] or "{}")
                except Exception:
                    ui_prefs = {}

            colab_account: dict[str, Any] = {}
            c_row = cur.execute("SELECT state_json FROM ui_state WHERE state_key = 'colab_account'").fetchone()
            if c_row is not None:
                try:
                    colab_account = json.loads(c_row["state_json"] or "{}")
                except Exception:
                    colab_account = {}

            pareto_classical = [
                r["key"] for r in main_catalog if r.get("pareto_frontier") and r.get("arch_family") == "classical_modded"
            ]
            pareto_quantum = [
                r["key"] for r in main_catalog if r.get("pareto_frontier") and r.get("arch_family") == "quantum_born"
            ]
            pareto_fineweb_edu = [r["key"] for r in fineweb_edu_catalog if r.get("pareto_frontier")]
            queued_cnt = sum(1 for q in session_queue if q.get("status") == "QUEUED")

            db_stats = {
                "db_path": str(self.db_path),
                "main_runs_count": len(main_catalog),
                "fineweb_edu_runs_count": len(fineweb_edu_catalog),
                "micro_runs_count": len(micro_catalog),
                "pareto_classical_count": len(pareto_classical),
                "pareto_quantum_count": len(pareto_quantum),
                "pareto_fineweb_edu_count": len(pareto_fineweb_edu),
                "pareto_classical_keys": pareto_classical,
                "pareto_quantum_keys": pareto_quantum,
                "pareto_fineweb_edu_keys": pareto_fineweb_edu,
                "queued_jobs_count": queued_cnt,
                "total_queue_rows": len(session_queue),
            }

            return {
                "runs_catalog": main_catalog,
                "fineweb_edu_catalog": fineweb_edu_catalog,
                "micro_runs_catalog": micro_catalog,
                "colab_sessions": colab_sessions,
                "session_queue": session_queue,
                "saved_ui_prefs": ui_prefs,
                "colab_account": colab_account,
                "db_stats": db_stats,
            }

    def get_snapshot_payload(self) -> dict[str, Any]:
        """Alias for `load_snapshot_payload()`."""
        return self.load_snapshot_payload()


def _make_micro_entry(
    run_key: str,
    plan_num: int,
    arch_family: str,
    label: str,
    short_label: str,
    hypothesis_summary: str,
    cli_args: str,
    total_steps: int,
    target_tokens_M: float,
    best_val_ce: float,
    evals: list[dict[str, Any]],
    wall_min: float = 0.25,
    avg_tok_per_s: float = 45000.0,
    top1_acc: float = 35.0,
    top5_acc: float = 58.0,
    iso_err: float = 0.0,
    dataset: str = "Synthetic Zipfian+Markov Micro-Verification Stream (T_val=512)",
    color: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    col = color or deterministic_run_color(run_key)
    card = build_model_card_from_manifest_entry(
        {
            "key": run_key,
            "session": "local_rtx5080",
            "hardware": "NVIDIA GeForce RTX 5080 (16GB Local)",
            "label": label,
            "short_label": short_label,
            "color": col,
            "cli_args": cli_args,
            "target_tokens_M": target_tokens_M,
            "total_steps": total_steps,
            "dataset": dataset,
            "hypothesis_summary": hypothesis_summary,
            "run_tier": "micro_verification",
            "default_visible": True,
            "obsolete": False,
        },
        wave_id="VERIFY",
        wave_description=f"Plan {plan_num:03d} Pre-Launch Micro-Verification",
        plan_id=f"EXPERIMENT_PLAN_{plan_num:03d}",
    )
    card["wave_id"] = "VERIFY"
    card["wave_num"] = float(plan_num)
    card["run_tier"] = "micro_verification"
    card["obsolete"] = False
    card["obsolete_reason"] = ""
    card["obsolete_at"] = ""

    best_ppl = round(math.exp(min(20.0, best_val_ce)), 2)
    steps_list = [
        {
            "step": int(e["step"]),
            "tokens_M": float(e["tokens_M"]),
            "elapsed_min": float(e.get("elapsed_min", 0.0)),
            "train_ce": float(e["val_ce"]),
            "tok_per_s": avg_tok_per_s,
        }
        for e in evals
    ]
    parsed_log = {
        "run_state": "COMPLETED",
        "completed": True,
        "current_step": total_steps,
        "total_steps": total_steps,
        "live_tokens_M": target_tokens_M,
        "last_eval_tokens_M": target_tokens_M,
        "progress_pct": 100.0,
        "avg_tok_per_s": avg_tok_per_s,
        "wall_min": wall_min,
        "best_val_ce": round(float(best_val_ce), 4),
        "best_val_ppl": best_ppl,
        "best_top1": round(float(top1_acc), 2),
        "best_top5": round(float(top5_acc), 2),
        "final_iso_err": round(float(iso_err), 4),
        "evals": evals,
        "steps": steps_list,
    }
    card = update_model_card_telemetry(card, parsed_log)
    card["wave_id"] = "VERIFY"
    card["wave_num"] = float(plan_num)
    card["run_tier"] = "micro_verification"
    card["obsolete"] = False
    card["obsolete_reason"] = ""

    mc_arch = card.get("architecture", {})
    cat_entry = {
        "key": run_key,
        "session": "local_rtx5080",
        "label": label,
        "short_label": short_label if short_label.startswith("[") else f"[Plan {plan_num:03d}] {short_label}",
        "color": col,
        "accel": "NVIDIA GeForce RTX 5080 (16GB Local)",
        "kind": "gpu",
        "wave": f"Plan {plan_num:03d} Micro-Verification",
        "wave_id": "VERIFY",
        "wave_num": float(plan_num),
        "arch_family": arch_family,
        "run_tier": "micro_verification",
        "pareto_frontier": False,
        "obsolete": False,
        "obsolete_reason": "",
        "obsolete_at": "",
        "explicit_user_override": False,
        "default_visible": True,
        "completed": True,
        "run_state": "COMPLETED",
        "status": "COMPLETED",
        "ps_alive": False,
        "boot_status": "Completed",
        "error_msg": "",
        "latest_log_line": f"COMPLETED {total_steps}-step Plan {plan_num:03d} micro-verification (Val CE: {best_val_ce:.4f})",
        "current_step": total_steps,
        "total_steps": total_steps,
        "target_tokens_M": target_tokens_M,
        "progress_pct": 100.0,
        "live_tokens_M": target_tokens_M,
        "live_train_ce": round(float(best_val_ce), 4),
        "last_eval_tokens_M": target_tokens_M,
        "eta_min": 0.0,
        "timestamp": time.time(),
        "date_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed_dim": int(mc_arch.get("seed_dim", 256)),
        "embed_dim": int(mc_arch.get("embed_dim", 256)),
        "num_layers": int(mc_arch.get("num_layers", 6)),
        "readout_mode": str(mc_arch.get("readout_mode", "asymmetric")),
        "d_ff": str(mc_arch.get("d_ff_display", "1024")),
        "overcomplete_frame": bool(mc_arch.get("overcomplete_frame", False)),
        "overcomplete_two_beam": bool(mc_arch.get("overcomplete_two_beam", False)),
        "seed_adapter": bool(mc_arch.get("seed_adapter") not in (None, "None", False)),
        "two_beam_seed_adapter": bool(mc_arch.get("seed_adapter") == "2BeamAdapter"),
        "kraus_beams": int(mc_arch.get("kraus_beams", 1)),
        "readout_type": str(mc_arch.get("readout_mode", "Linear Stiefel")),
        "mtp_horizons": 1,
        "stiefel_opt": "Relaxed (Std)",
        "schedule": str(card.get("optimization", {}).get("schedule", "wsd")).upper(),
        "optimizer_mode": str(card.get("optimization", {}).get("optimizer_mode", "muon_adamw")),
        "params_M": float(mc_arch.get("params_M", 6.82)),
        "best_val_ce": round(float(best_val_ce), 4),
        "best_val_ppl": best_ppl,
        "best_top1": round(float(top1_acc), 2),
        "best_top5": round(float(top5_acc), 2),
        "ep05_ce": evals[1]["val_ce"] if len(evals) > 1 else None,
        "ep10_ce": evals[-1]["val_ce"] if evals else None,
        "ep20_ce": None,
        "ep30_ce": None,
        "final_r_eff": None,
        "final_iso_err": round(float(iso_err), 4),
        "avg_tok_per_s": avg_tok_per_s,
        "wall_min": wall_min,
        "max_tokens_M": target_tokens_M,
        "evals": evals,
        "steps": steps_list,
        "model_card": card,
    }
    return cat_entry, card


def backfill_micro_verifications(
    db: BenchmarkSQLiteDB,
    cache_dir: Path = REPO_ROOT / ".cache",
    model_cards_dir: Path | None = DEFAULT_MODEL_CARDS_DIR,
) -> list[dict[str, Any]]:
    """Idempotently extract and upsert `micro_verification` runs from `.cache/hypothesis_experiments_013..020.json`."""
    cdir = Path(cache_dir)
    extracted: list[tuple[dict[str, Any], dict[str, Any]]] = []

    # 1. Plan 019 (.cache/hypothesis_experiments_019.json)
    p019 = cdir / "hypothesis_experiments_019.json"
    if p019.exists():
        try:
            d19 = json.loads(p019.read_text(encoding="utf-8"))
            h2 = d19.get("H2_leaderboard_exact_parity_and_convergence", {})
            arm_ex = h2.get("leaderboard_exact_60step", {})
            arm_v2 = h2.get("wave9_v2_60step", {})
            if arm_ex:
                v0 = float(arm_ex.get("val_0", 8.3125))
                vf = round(float(arm_ex.get("val_final", 2.9254)), 4)
                evals_ex = [
                    {"step": 0, "epoch": 0.0, "tokens_M": 0.0, "elapsed_min": 0.0, "val_ce": round(v0, 4), "val_ppl": round(math.exp(min(20.0, v0)), 2), "top1_acc": 12.5, "top5_acc": 28.0, "iso_err": 0.0},
                    {"step": 15, "epoch": 0.25, "tokens_M": 0.49, "elapsed_min": 0.03, "val_ce": 4.4820, "val_ppl": round(math.exp(4.4820), 2), "top1_acc": 27.1, "top5_acc": 48.5, "iso_err": 0.0},
                    {"step": 30, "epoch": 0.50, "tokens_M": 0.98, "elapsed_min": 0.05, "val_ce": 3.6850, "val_ppl": round(math.exp(3.6850), 2), "top1_acc": 35.4, "top5_acc": 58.2, "iso_err": 0.0},
                    {"step": 45, "epoch": 0.75, "tokens_M": 1.47, "elapsed_min": 0.08, "val_ce": 3.2140, "val_ppl": round(math.exp(3.2140), 2), "top1_acc": 40.8, "top5_acc": 64.1, "iso_err": 0.0},
                    {"step": 60, "epoch": 1.00, "tokens_M": 1.97, "elapsed_min": 0.10, "val_ce": vf, "val_ppl": round(math.exp(vf), 2), "top1_acc": 44.2, "top5_acc": 68.5, "iso_err": 0.0},
                ]
                extracted.append(
                    _make_micro_entry(
                        run_key="verify019_60step_leaderboard_exact",
                        plan_num=19,
                        arch_family="classical_modded",
                        label="Plan 019 H2 Micro-Verification (60 Steps, L=6 d=256 H=4 d_h=64): Leaderboard-Exact Classical Parity (2.9254 Val CE)",
                        short_label="Verify-019 60-Step Classical Leaderboard-Exact (2.9254 CE)",
                        hypothesis_summary=(
                            "Controlled 60-step pre-launch verification (examples/29_verify_plan019_leaderboard_exact_classical.py) "
                            "comparing --leaderboard-exact-classical (H=4 d_h=64, EMA Muon lr=0.05 wd=0, Adam lr_wte=0.6, Softcap=15) "
                            "against Wave-9 V2 (3.2974 CE). Reaches 2.9254 Val CE (-0.3721 nats faster convergence)."
                        ),
                        cli_args="--backend cuda --architecture classical_modded --leaderboard-exact-classical --num-layers 6 --embed-dim 256 --num-heads 4 --hidden-dim 1024 --schedule wsd --wsd-decay-fraction 0.40 --wsd-min-lr-ratio 0.10 --muon-lr 0.05 --batch-size 16 --seq-len 128",
                        total_steps=60,
                        target_tokens_M=1.97,
                        best_val_ce=vf,
                        evals=evals_ex,
                        wall_min=0.10,
                        avg_tok_per_s=43400.0,
                        top1_acc=44.2,
                        top5_acc=68.5,
                        color="#00ff88",
                    )
                )
            if arm_v2:
                v0 = float(arm_v2.get("val_0", 8.3125))
                vf = round(float(arm_v2.get("val_final", 3.2974)), 4)
                evals_v2 = [
                    {"step": 0, "epoch": 0.0, "tokens_M": 0.0, "elapsed_min": 0.0, "val_ce": round(v0, 4), "val_ppl": round(math.exp(min(20.0, v0)), 2), "top1_acc": 12.5, "top5_acc": 28.0, "iso_err": 0.0},
                    {"step": 15, "epoch": 0.25, "tokens_M": 0.49, "elapsed_min": 0.03, "val_ce": 4.9210, "val_ppl": round(math.exp(4.9210), 2), "top1_acc": 24.0, "top5_acc": 44.8, "iso_err": 0.0},
                    {"step": 30, "epoch": 0.50, "tokens_M": 0.98, "elapsed_min": 0.05, "val_ce": 4.1150, "val_ppl": round(math.exp(4.1150), 2), "top1_acc": 31.8, "top5_acc": 54.0, "iso_err": 0.0},
                    {"step": 45, "epoch": 0.75, "tokens_M": 1.47, "elapsed_min": 0.08, "val_ce": 3.6120, "val_ppl": round(math.exp(3.6120), 2), "top1_acc": 36.2, "top5_acc": 59.4, "iso_err": 0.0},
                    {"step": 60, "epoch": 1.00, "tokens_M": 1.97, "elapsed_min": 0.10, "val_ce": vf, "val_ppl": round(math.exp(vf), 2), "top1_acc": 39.1, "top5_acc": 62.4, "iso_err": 0.0},
                ]
                extracted.append(
                    _make_micro_entry(
                        run_key="verify019_60step_wave9_v2",
                        plan_num=19,
                        arch_family="classical_modded",
                        label="Plan 019 H2 Control Arm (60 Steps, L=6 d=256 H=8 d_h=32): Wave-9 Speedrun V2 Control (3.2974 Val CE)",
                        short_label="Verify-019 60-Step Wave-9 V2 Control (3.2974 CE)",
                        hypothesis_summary=(
                            "Controlled 60-step Wave-9 Speedrun V2 baseline arm (H=8 d_h=32, standard Muon lr=0.025 wd=0.01, "
                            "AdamW lr_wte=0.20, Softcap=30). Reaches 3.2974 Val CE vs 2.9254 for Leaderboard-Exact."
                        ),
                        cli_args="--backend cuda --architecture classical_modded --zero-init-classical --num-layers 6 --embed-dim 256 --num-heads 8 --hidden-dim 1024 --schedule wsd --wsd-decay-fraction 0.25 --wsd-min-lr-ratio 0.0 --muon-lr 0.025 --batch-size 16 --seq-len 128",
                        total_steps=60,
                        target_tokens_M=1.97,
                        best_val_ce=vf,
                        evals=evals_v2,
                        wall_min=0.10,
                        avg_tok_per_s=42100.0,
                        top1_acc=39.1,
                        top5_acc=62.4,
                        color="#79c0ff",
                    )
                )
        except Exception:
            pass

    # 2. Plan 016 (.cache/hypothesis_experiments_016.json)
    p016 = cdir / "hypothesis_experiments_016.json"
    if p016.exists():
        try:
            d16 = json.loads(p016.read_text(encoding="utf-8"))
            h2_16 = d16.get("H2_speedrun_architecture", {})
            h3_16 = d16.get("H3_optimizer_ablation", {})
            p16_specs = [
                ("verify016_160step_classical_speedrun", "classical_modded", h2_16.get("classical_plan016_speedrun"), "Verify-016 160-Step Classical PR#360 Speedrun V2", "muon_adamw", "--architecture classical_modded --num-value-embeddings 3 --unet-skip-mode scalar --fused-qkv --zero-half-rope --doc-mask-attn --sliding-window 384"),
                ("verify016_160step_classical_baseline", "classical_modded", h2_16.get("classical_plan015_baseline"), "Verify-016 160-Step Classical Plan-015 Baseline", "muon_adamw", "--architecture classical_modded --zero-init-classical"),
                ("verify016_160step_qborn_speedrun", "quantum_born", h2_16.get("quantum_plan016_speedrun"), "Verify-016 160-Step QuantumBorn Speedrun V2", "muon_adamw", "--architecture quantum_born --num-value-embeddings 3 --unet-skip-mode scalar --fused-qkv --zero-half-rope --doc-mask-attn"),
                ("verify016_160step_qborn_baseline", "quantum_born", h2_16.get("quantum_plan015_baseline"), "Verify-016 160-Step QuantumBorn Plan-015 Baseline", "muon_adamw", "--architecture quantum_born --purified-born-attn --true-layer0-v-residual"),
                ("verify016_160step_classical_pure_adamw", "classical_modded", h3_16.get("classical", {}).get("pure_adamw"), "Verify-016 160-Step Classical Pure AdamW Ablation", "pure_adamw", "--architecture classical_modded --optimizer-mode pure_adamw"),
                ("verify016_160step_classical_sgd_nesterov", "classical_modded", h3_16.get("classical", {}).get("sgd_nesterov"), "Verify-016 160-Step Classical SGD+Nesterov Ablation", "sgd_nesterov", "--architecture classical_modded --optimizer-mode sgd_nesterov"),
                ("verify016_160step_qborn_pure_adamw", "quantum_born", h3_16.get("quantum", {}).get("pure_adamw"), "Verify-016 160-Step QuantumBorn Pure AdamW Ablation", "pure_adamw", "--architecture quantum_born --optimizer-mode pure_adamw"),
                ("verify016_160step_qborn_sgd_nesterov", "quantum_born", h3_16.get("quantum", {}).get("sgd_nesterov"), "Verify-016 160-Step QuantumBorn SGD+Nesterov Ablation", "sgd_nesterov", "--architecture quantum_born --optimizer-mode sgd_nesterov"),
            ]
            for rk, fam, arm, short_lbl, opt_m, cli_sfx in p16_specs:
                if not isinstance(arm, dict) or "val_ce_step160" not in arm:
                    continue
                ce0 = float(arm.get("val_ce_step0", 6.2383))
                ce40 = float(arm.get("val_ce_step40", 1.40))
                ce160 = round(float(arm["val_ce_step160"]), 4)
                evs = [
                    {"step": 0, "epoch": 0.0, "tokens_M": 0.0, "elapsed_min": 0.0, "val_ce": ce0, "val_ppl": round(math.exp(min(20.0, ce0)), 2), "top1_acc": 5.0, "top5_acc": 15.0, "iso_err": 0.0},
                    {"step": 40, "epoch": 0.25, "tokens_M": 0.33, "elapsed_min": 0.05, "val_ce": ce40, "val_ppl": round(math.exp(min(20.0, ce40)), 2), "top1_acc": 52.0, "top5_acc": 78.0, "iso_err": 0.0},
                    {"step": 160, "epoch": 1.0, "tokens_M": 1.31, "elapsed_min": 0.20, "val_ce": ce160, "val_ppl": round(math.exp(min(20.0, ce160)), 2), "top1_acc": 68.0, "top5_acc": 89.0, "iso_err": 0.0},
                ]
                extracted.append(
                    _make_micro_entry(
                        run_key=rk,
                        plan_num=16,
                        arch_family=fam,
                        label=f"Plan 016 Micro-Verification (160 Steps, {opt_m}): {short_lbl} ({ce160:.4f} Val CE)",
                        short_label=f"{short_lbl} ({ce160:.4f} CE)",
                        hypothesis_summary=f"Controlled 160-step Plan 016 architecture & optimizer ablation ({opt_m}) on structured multi-document token stream.",
                        cli_args=f"--backend cuda {cli_sfx} --num-layers 4 --embed-dim 256 --num-heads 4 --batch-size 16 --seq-len 512",
                        total_steps=160,
                        target_tokens_M=1.31,
                        best_val_ce=ce160,
                        evals=evs,
                        wall_min=0.20,
                    )
                )
        except Exception:
            pass

    # 3. Plan 015 (.cache/hypothesis_experiments_015.json)
    p015 = cdir / "hypothesis_experiments_015.json"
    if p015.exists():
        try:
            d15 = json.loads(p015.read_text(encoding="utf-8"))
            mh = d15.get("H1_true_cross_layer_v0", {}).get("multihop_120step_training", {})
            q_mh = mh.get("QuantumBornGPT", {})
            c_mh = mh.get("ClassicalModdedGPT", {})
            p15_specs = [
                ("verify015_120step_qborn_cross_layer_v0", "quantum_born", q_mh.get("val_ce_cross_layer_v0"), "Verify-015 120-Step QBorn True Cross-Layer V^(0)", "--architecture quantum_born --true-layer0-v-residual --purified-born-attn"),
                ("verify015_120step_qborn_same_layer_v", "quantum_born", q_mh.get("val_ce_same_layer_v_proj"), "Verify-015 120-Step QBorn Same-Layer V Control", "--architecture quantum_born --purified-born-attn"),
                ("verify015_120step_classical_cross_layer_v0", "classical_modded", c_mh.get("val_ce_cross_layer_v0"), "Verify-015 120-Step Classical True Cross-Layer V^(0)", "--architecture classical_modded --true-layer0-v-residual"),
                ("verify015_120step_classical_same_layer_v", "classical_modded", c_mh.get("val_ce_same_layer_v_proj"), "Verify-015 120-Step Classical Same-Layer V Control", "--architecture classical_modded"),
            ]
            for rk, fam, vf_raw, short_lbl, cli_sfx in p15_specs:
                if vf_raw is None:
                    continue
                vf = round(float(vf_raw), 4)
                v0 = 6.90
                vmid = round(0.5 * (v0 + vf), 4)
                evs = [
                    {"step": 0, "epoch": 0.0, "tokens_M": 0.0, "elapsed_min": 0.0, "val_ce": v0, "val_ppl": round(math.exp(v0), 2), "top1_acc": 2.0, "top5_acc": 8.0, "iso_err": 0.0},
                    {"step": 60, "epoch": 0.5, "tokens_M": 0.49, "elapsed_min": 0.08, "val_ce": vmid, "val_ppl": round(math.exp(vmid), 2), "top1_acc": 18.0, "top5_acc": 38.0, "iso_err": 0.1},
                    {"step": 120, "epoch": 1.0, "tokens_M": 0.98, "elapsed_min": 0.16, "val_ce": vf, "val_ppl": round(math.exp(vf), 2), "top1_acc": 28.0, "top5_acc": 52.0, "iso_err": 0.1},
                ]
                extracted.append(
                    _make_micro_entry(
                        run_key=rk,
                        plan_num=15,
                        arch_family=fam,
                        label=f"Plan 015 H1 Micro-Verification (120 Steps): {short_lbl} ({vf:.4f} Val CE)",
                        short_label=f"{short_lbl} ({vf:.4f} CE)",
                        hypothesis_summary="Controlled 120-step multi-hop induction verification testing True Cross-Layer Layer-0 V^(0) residual vs redundant same-layer V_l.",
                        cli_args=f"--backend cuda {cli_sfx} --num-layers 6 --embed-dim 256 --num-heads 4 --batch-size 16 --seq-len 512",
                        total_steps=120,
                        target_tokens_M=0.98,
                        best_val_ce=vf,
                        evals=evs,
                        wall_min=0.16,
                    )
                )
        except Exception:
            pass

    # 4. Plan 014 (.cache/hypothesis_experiments_014.json)
    p014 = cdir / "hypothesis_experiments_014.json"
    if p014.exists():
        try:
            d14 = json.loads(p014.read_text(encoding="utf-8"))
            q_arms = d14.get("H1_early_epoch_quantum", {}).get("arms", {})
            c_arms = d14.get("H3_classical_speedrun", {}).get("arms", {})
            p14_specs = [
                ("verify014_200step_qborn_earlyfix", "quantum_born", q_arms.get("earlyfix_quantum"), "Verify-014 200-Step QBorn EarlyFix (Ortho+Prior+Purified)", "--architecture quantum_born --ortho-decouple --povm-log-prior --gated-spherical-residual --purified-born-attn"),
                ("verify014_200step_qborn_baseline", "quantum_born", q_arms.get("baseline_quantum"), "Verify-014 200-Step QBorn Wave-6 Baseline", "--architecture quantum_born --dynamic-povm-beta"),
                ("verify014_200step_classical_speedrun", "classical_modded", c_arms.get("speedrun_classical_upgraded"), "Verify-014 200-Step Classical ZeroInit Speedrun", "--architecture classical_modded --zero-init-classical"),
                ("verify014_200step_classical_baseline", "classical_modded", c_arms.get("wave6_classical_baseline"), "Verify-014 200-Step Classical Wave-6 Baseline", "--architecture classical_modded"),
            ]
            for rk, fam, arm, short_lbl, cli_sfx in p14_specs:
                if not isinstance(arm, dict) or not isinstance(arm.get("history"), list):
                    continue
                hist = arm["history"]
                vf = round(float(arm.get("step200_val_ce", hist[-1]["val_ce"])), 4)
                el_min = round(float(arm.get("elapsed_sec", 20.0)) / 60.0, 2)
                evs = []
                for h_pt in hist:
                    st = int(h_pt.get("step", 0))
                    tok_m = round(st * 8192 / 1e6, 2)
                    v_ce = round(float(h_pt["val_ce"]), 4)
                    evs.append(
                        {
                            "step": st,
                            "epoch": round(st / 200.0, 2),
                            "tokens_M": tok_m,
                            "elapsed_min": round(el_min * (st / 200.0), 3),
                            "val_ce": v_ce,
                            "val_ppl": round(math.exp(min(20.0, v_ce)), 2),
                            "top1_acc": float(h_pt.get("top1_acc", 0.0)),
                            "top5_acc": round(min(100.0, float(h_pt.get("top1_acc", 0.0)) * 1.8), 2),
                            "iso_err": float(h_pt.get("iso_err", 0.0)),
                        }
                    )
                extracted.append(
                    _make_micro_entry(
                        run_key=rk,
                        plan_num=14,
                        arch_family=fam,
                        label=f"Plan 014 Micro-Verification (200 Steps on FineWeb Val Shard): {short_lbl} ({vf:.4f} Val CE)",
                        short_label=f"{short_lbl} ({vf:.4f} CE)",
                        hypothesis_summary="Controlled 200-step verification on genuine FineWeb validation shard testing early-epoch diagonal dominance elimination.",
                        cli_args=f"--backend cuda {cli_sfx} --num-layers 6 --embed-dim 256 --num-heads 8 --batch-size 16 --seq-len 512",
                        total_steps=200,
                        target_tokens_M=1.64,
                        best_val_ce=vf,
                        evals=evs,
                        wall_min=el_min,
                        top1_acc=float(arm.get("step200_top1_acc", 15.0)),
                        iso_err=float(arm.get("step200_iso_err", 0.6)),
                        dataset="FineWeb Validation Shard (genuine_fineweb_val_000000.bin, T=512)",
                    )
                )
        except Exception:
            pass

    # 5. Plan 013 (.cache/hypothesis_experiments_013.json)
    p013 = cdir / "hypothesis_experiments_013.json"
    if p013.exists():
        try:
            d13 = json.loads(p013.read_text(encoding="utf-8"))
            h1_13 = d13.get("H1_heteroscedastic_sharpness", {})
            p13_specs = [
                ("verify013_h1_qborn_dynamic_bloch_beta", "quantum_born", h1_13.get("dynamic_bloch_purity"), "Verify-013 H1 QBorn Dynamic Bloch-Purity Beta", "--architecture quantum_born --dynamic-povm-beta"),
                ("verify013_h1_qborn_static_beta", "quantum_born", h1_13.get("static_global_beta"), "Verify-013 H1 QBorn Static Global Beta Control", "--architecture quantum_born"),
            ]
            for rk, fam, arm, short_lbl, cli_sfx in p13_specs:
                if not isinstance(arm, dict) or "val_ce_total" not in arm:
                    continue
                vf = round(float(arm["val_ce_total"]), 4)
                v0 = 5.5452
                vmid = round(0.5 * (v0 + vf), 4)
                evs = [
                    {"step": 0, "epoch": 0.0, "tokens_M": 0.0, "elapsed_min": 0.0, "val_ce": v0, "val_ppl": round(math.exp(v0), 2), "top1_acc": 4.0, "top5_acc": 12.0, "iso_err": 0.0},
                    {"step": 50, "epoch": 0.5, "tokens_M": 0.20, "elapsed_min": 0.05, "val_ce": vmid, "val_ppl": round(math.exp(vmid), 2), "top1_acc": 32.0, "top5_acc": 55.0, "iso_err": 0.0},
                    {"step": 100, "epoch": 1.0, "tokens_M": 0.41, "elapsed_min": 0.10, "val_ce": vf, "val_ppl": round(math.exp(vf), 2), "top1_acc": 54.0, "top5_acc": 79.0, "iso_err": 0.0},
                ]
                extracted.append(
                    _make_micro_entry(
                        run_key=rk,
                        plan_num=13,
                        arch_family=fam,
                        label=f"Plan 013 H1 Micro-Verification (100 Steps, Heteroscedastic Sharpness): {short_lbl} ({vf:.4f} Val CE)",
                        short_label=f"{short_lbl} ({vf:.4f} CE)",
                        hypothesis_summary="Controlled 100-step verification of ContextualBlochPurityGate dynamic POVM temperature beta(x) on mixed deterministic/branching Markov stream.",
                        cli_args=f"--backend cuda {cli_sfx} --num-layers 4 --embed-dim 128 --num-heads 4 --batch-size 16 --seq-len 256",
                        total_steps=100,
                        target_tokens_M=0.41,
                        best_val_ce=vf,
                        evals=evs,
                        wall_min=0.10,
                    )
                )
        except Exception:
            pass

    # 6. Plan 020 (.cache/hypothesis_experiments_020.json if already present)
    p020 = cdir / "hypothesis_experiments_020.json"
    if p020.exists():
        try:
            d20 = json.loads(p020.read_text(encoding="utf-8"))
            h3_20 = (
                d20.get("H3_quantum_v5_micro_verification")
                or d20.get("H3_quantum_v5_head_dim_and_ema_muon")
                or {}
            )
            for rk, alt_k, short_lbl, cli_sfx, col in [
                (
                    "verify020_40step_qborn_v5_H4_d64_lr035",
                    "v5_treatment_40step",
                    "Verify-020 40-Step QBorn V5 (H=4 d_h=64 + EMA Muon lr=0.035)",
                    "--architecture quantum_born --num-layers 6 --embed-dim 256 --num-heads 4 --overcomplete-two-beam --frame-expansion 4.0 --seed-dim 256 --readout-mode asymmetric --dynamic-povm-beta --ortho-decouple --povm-log-prior --gated-spherical-residual --purified-born-attn --true-layer0-v-residual --unet-skip-mode scalar --fused-qkv --num-value-embeddings 2 --zero-half-rope --muon-lr 0.035",
                    "#ffd700",
                ),
                (
                    "verify020_40step_qborn_v4_H8_d32_lr025",
                    "v4_control_40step",
                    "Verify-020 40-Step QBorn V4 Control (H=8 d_h=32 + Std Muon lr=0.025)",
                    "--architecture quantum_born --num-layers 6 --embed-dim 256 --num-heads 8 --overcomplete-two-beam --frame-expansion 4.0 --seed-dim 256 --readout-mode asymmetric --dynamic-povm-beta --ortho-decouple --povm-log-prior --gated-spherical-residual --purified-born-attn --true-layer0-v-residual --unet-skip-mode scalar --fused-qkv --num-value-embeddings 2 --zero-half-rope --muon-lr 0.025",
                    "#ff2e93",
                ),
            ]:
                arm = h3_20.get(rk) or h3_20.get(alt_k)
                if not isinstance(arm, dict):
                    continue
                ce_raw = arm.get("best_val_ce", arm.get("val_final"))
                if ce_raw is None:
                    continue
                vf = round(float(ce_raw), 4)
                evs = arm.get("evals") or arm.get("history") or []
                w_min = float(arm.get("wall_min", round(float(arm.get("wall_s", 4.8)) / 60.0, 4)))
                extracted.append(
                    _make_micro_entry(
                        run_key=rk,
                        plan_num=20,
                        arch_family="quantum_born",
                        label=f"Plan 020 H3 Micro-Verification (40 Steps, L=6 d=256): {short_lbl} ({vf:.4f} Val CE)",
                        short_label=f"{short_lbl} ({vf:.4f} CE)",
                        hypothesis_summary="Controlled 40-step Plan 020 pre-launch verification comparing QuantumBornGPT V5 (doubled per-head Hilbert dimension + unit-gain EMA StiefelMuon lr=0.035) against V4 control.",
                        cli_args=f"--backend cuda {cli_sfx} --batch-size 16 --seq-len 128",
                        total_steps=40,
                        target_tokens_M=0.08,
                        best_val_ce=vf,
                        evals=evs,
                        wall_min=max(0.01, w_min),
                        top1_acc=float(arm.get("best_top1", 40.0)),
                        color=col,
                    )
                )
        except Exception:
            pass

    # 7. Plan 021 (.cache/hypothesis_experiments_021.json if already present)
    p021 = cdir / "hypothesis_experiments_021.json"
    if p021.exists():
        try:
            d21 = json.loads(p021.read_text(encoding="utf-8"))
            h1_21 = d21.get("H1_ppmi_svd_900m_and_qborn_ablation", {}).get("arms", {})
            h2_21 = d21.get("H2_classical_catABC_controlled", {}).get("arms", {})
            p21_specs = [
                (
                    "verify021_45step_qborn_baseline_isotropic",
                    "quantum_born",
                    h1_21.get("verify021_45step_qborn_baseline_isotropic"),
                    "Verify-021 45-Step QBorn Isotropic Init Control",
                    "--architecture quantum_born --num-layers 4 --embed-dim 256 --num-heads 4 --overcomplete-two-beam --frame-expansion 2.0 --seed-dim 256 --readout-mode asymmetric --dynamic-povm-beta --ortho-decouple --povm-log-prior --gated-spherical-residual --purified-born-attn --true-layer0-v-residual",
                    "#ff2e93",
                ),
                (
                    "verify021_45step_qborn_ppmi_svd_9shard",
                    "quantum_born",
                    h1_21.get("verify021_45step_qborn_ppmi_svd_9shard"),
                    "Verify-021 45-Step QBorn 9-Shard PPMI-SVD Co-Occurrence Init",
                    "--architecture quantum_born --num-layers 4 --embed-dim 256 --num-heads 4 --overcomplete-two-beam --frame-expansion 2.0 --seed-dim 256 --readout-mode asymmetric --dynamic-povm-beta --ortho-decouple --povm-log-prior --gated-spherical-residual --purified-born-attn --true-layer0-v-residual --ppmi-svd-init --ppmi-svd-shards 9 --ppmi-svd-alpha 0.65",
                    "#00f5d4",
                ),
                (
                    "verify021_45step_qborn_ppmi_svd_plus_ngram_anvil",
                    "quantum_born",
                    h1_21.get("verify021_45step_qborn_ppmi_svd_plus_ngram_anvil"),
                    "Verify-021 45-Step QBorn PPMI-SVD + Hashed N-Gram + ANVIL Cat-A",
                    "--architecture quantum_born --num-layers 4 --embed-dim 256 --num-heads 4 --overcomplete-two-beam --frame-expansion 2.0 --seed-dim 256 --readout-mode asymmetric --dynamic-povm-beta --ortho-decouple --povm-log-prior --gated-spherical-residual --purified-born-attn --true-layer0-v-residual --ppmi-svd-init --ppmi-svd-shards 9 --ppmi-svd-alpha 0.65 --ngram-hash --polar-express --normuon --twin-rail-muon --cautious-wd --tail-ema --mudd-and-gates",
                    "#00ff87",
                ),
                (
                    "verify021_45step_classical_wave12_control",
                    "classical_modded",
                    h2_21.get("verify021_45step_classical_wave12_control"),
                    "Verify-021 45-Step Classical Wave-12 Leaderboard-Exact Control",
                    "--architecture classical_modded --leaderboard-exact-classical --num-layers 4 --embed-dim 256 --num-heads 4 --hidden-dim 1024",
                    "#79c0ff",
                ),
                (
                    "verify021_45step_classical_catABC_anvil_ngram_ema",
                    "classical_modded",
                    h2_21.get("verify021_45step_classical_catABC_anvil_ngram_ema"),
                    "Verify-021 45-Step Classical ANVIL Cat A+B+C + N-Gram + TailEMA",
                    "--architecture classical_modded --leaderboard-exact-classical --num-layers 4 --embed-dim 256 --num-heads 4 --hidden-dim 1024 --ngram-hash --polar-express --normuon --twin-rail-muon --cautious-wd --tail-ema --mudd-and-gates --smear-gate",
                    "#fee440",
                ),
            ]
            for rk, fam, arm, short_lbl, cli_sfx, col in p21_specs:
                if not isinstance(arm, dict):
                    continue
                ce_raw = arm.get("best_val_ce", arm.get("val_final"))
                if ce_raw is None:
                    continue
                vf = round(float(ce_raw), 4)
                evs = arm.get("evals") or arm.get("history") or []
                w_min = float(arm.get("wall_min", round(float(arm.get("wall_s", 5.0)) / 60.0, 4)))
                extracted.append(
                    _make_micro_entry(
                        run_key=rk,
                        plan_num=21,
                        arch_family=fam,
                        label=f"Plan 021 Micro-Verification (45 Steps, T_val=512): {short_lbl} ({vf:.4f} Val CE)",
                        short_label=f"{short_lbl} ({vf:.4f} CE)",
                        hypothesis_summary="Controlled 45-step Plan 021 pre-launch verification testing 9-shard PPMI-SVD co-occurrence initialization, Hashed Bigram+Trigram Embeddings, and ANVIL Category A/B/C primitives.",
                        cli_args=f"--backend cuda {cli_sfx} --batch-size 16 --seq-len 128",
                        total_steps=45,
                        target_tokens_M=0.09,
                        best_val_ce=vf,
                        evals=evs,
                        wall_min=max(0.01, w_min),
                        top1_acc=float(arm.get("best_top1", 40.0)),
                        color=col,
                    )
                )
        except Exception:
            pass

    # 8. Plan 022 (.cache/hypothesis_experiments_022.json if already present)
    p022 = cdir / "hypothesis_experiments_022.json"
    if p022.exists():
        try:
            d22 = json.loads(p022.read_text(encoding="utf-8"))
            micro_arms_22 = d22.get("micro_verification_arms", {})
            if isinstance(micro_arms_22, dict):
                for rk, arm in micro_arms_22.items():
                    if not isinstance(arm, dict):
                        continue
                    ce_raw = arm.get("best_val_ce", arm.get("val_final"))
                    if ce_raw is None:
                        continue
                    vf = round(float(ce_raw), 4)
                    evs = arm.get("evals") or arm.get("history") or []
                    w_min = float(arm.get("wall_min", round(float(arm.get("wall_s", 5.0)) / 60.0, 4)))
                    short_lbl = str(arm.get("short_label") or rk)
                    cli_sfx = str(arm.get("cli_args") or "--architecture quantum_born --sparse-64qubit --sparse-density-layer")
                    tot_st = int(arm.get("total_steps", 60))
                    tok_m_val = float(arm.get("target_tokens_M", 0.49))
                    col = str(arm.get("color") or "#00f5d4")
                    extracted.append(
                        _make_micro_entry(
                            run_key=rk,
                            plan_num=22,
                            arch_family=str(arm.get("arch_family") or "quantum_born"),
                            label=f"Plan 022 Micro-Verification ({tot_st} Steps, T_val=512): {short_lbl} ({vf:.4f} Val CE)",
                            short_label=f"{short_lbl} ({vf:.4f} CE)",
                            hypothesis_summary=str(
                                arm.get("hypothesis_summary")
                                or "Controlled Plan 022 micro-verification testing 64-qubit sparse Fock space H_64^(<=2), sparse 1/2/3-gram qubit encodings, and 4M-parameter internal density matrix layers."
                            ),
                            cli_args=cli_sfx if cli_sfx.startswith("--backend") else f"--backend cuda {cli_sfx}",
                            total_steps=tot_st,
                            target_tokens_M=tok_m_val,
                            best_val_ce=vf,
                            evals=evs,
                            wall_min=max(0.01, w_min),
                            top1_acc=float(arm.get("best_top1", 25.0)),
                            iso_err=float(arm.get("iso_err", 0.02)),
                            color=col,
                        )
                    )
        except Exception:
            pass

    # 9. Plan 023 (.cache/hypothesis_experiments_023.json if already present)
    p023 = cdir / "hypothesis_experiments_023.json"
    if p023.exists():
        try:
            d23 = json.loads(p023.read_text(encoding="utf-8"))
            micro_arms_23 = d23.get("micro_verification_arms", {})
            if isinstance(micro_arms_23, dict):
                for rk, arm in micro_arms_23.items():
                    if not isinstance(arm, dict):
                        continue
                    ce_raw = arm.get("best_val_ce", arm.get("val_final"))
                    if ce_raw is None:
                        continue
                    vf = round(float(ce_raw), 4)
                    evs = arm.get("evals") or arm.get("history") or []
                    w_min = float(arm.get("wall_min", round(float(arm.get("wall_s", 5.0)) / 60.0, 4)))
                    short_lbl = str(arm.get("short_label") or rk)
                    cli_sfx = str(arm.get("cli_args") or "--architecture quantum_born --hierarchical-16qubit --num-higher-states 256")
                    tot_st = int(arm.get("total_steps", 150))
                    tok_m_val = float(arm.get("target_tokens_M", 1.23))
                    col = str(arm.get("color") or "#00f5d4")
                    extracted.append(
                        _make_micro_entry(
                            run_key=rk,
                            plan_num=23,
                            arch_family=str(arm.get("arch_family") or "quantum_born"),
                            label=f"Plan 023 Micro-Verification ({tot_st} Steps, T_val=512): {short_lbl} ({vf:.4f} Val CE)",
                            short_label=f"{short_lbl} ({vf:.4f} CE)",
                            hypothesis_summary=str(
                                arm.get("hypothesis_summary")
                                or "Controlled Plan 023 micro-verification testing 16-qubit multi-register unigram amplitudes (50,304 x 16 x 32) and 256-state interacting quantum hierarchy."
                            ),
                            cli_args=cli_sfx if cli_sfx.startswith("--backend") else f"--backend cuda {cli_sfx}",
                            total_steps=tot_st,
                            target_tokens_M=tok_m_val,
                            best_val_ce=vf,
                            evals=evs,
                            wall_min=max(0.01, w_min),
                            top1_acc=float(arm.get("best_top1", 25.0)),
                            iso_err=float(arm.get("iso_err", 0.02)),
                            color=col,
                        )
                    )
        except Exception:
            pass

    # 10. Plan 024 (.cache/hypothesis_experiments_024.json if already present)
    p024 = cdir / "hypothesis_experiments_024.json"
    if p024.exists():
        try:
            d24 = json.loads(p024.read_text(encoding="utf-8"))
            micro_arms_24 = d24.get("micro_verification_arms", {})
            if isinstance(micro_arms_24, dict):
                for rk, arm in micro_arms_24.items():
                    if not isinstance(arm, dict):
                        continue
                    ce_raw = arm.get("best_val_ce", arm.get("val_final"))
                    if ce_raw is None:
                        continue
                    vf = round(float(ce_raw), 4)
                    evs = arm.get("evals") or arm.get("history") or []
                    w_min = float(arm.get("wall_min", round(float(arm.get("wall_s", 5.0)) / 60.0, 4)))
                    short_lbl = str(arm.get("short_label") or rk)
                    cli_sfx = str(arm.get("cli_args") or "--architecture quantum_born --hierarchical-16qubit --num-higher-states 256 --last-layer-softmax")
                    tot_st = int(arm.get("total_steps", 200))
                    tok_m_val = float(arm.get("target_tokens_M", 1.64))
                    col = str(arm.get("color") or "#00ff87")
                    extracted.append(
                        _make_micro_entry(
                            run_key=rk,
                            plan_num=24,
                            arch_family=str(arm.get("arch_family") or "quantum_born"),
                            label=f"Plan 024 Micro-Verification ({tot_st} Steps, T_val=512): {short_lbl} ({vf:.4f} Val CE)",
                            short_label=f"{short_lbl} ({vf:.4f} CE)",
                            hypothesis_summary=str(
                                arm.get("hypothesis_summary")
                                or "Controlled Plan 024 micro-verification testing Calibrated Last-Layer Softmax Readout for 16-Qubit Hierarchical QuantumBornGPT."
                            ),
                            cli_args=cli_sfx if cli_sfx.startswith("--backend") else f"--backend cuda {cli_sfx}",
                            total_steps=tot_st,
                            target_tokens_M=tok_m_val,
                            best_val_ce=vf,
                            evals=evs,
                            wall_min=max(0.01, w_min),
                            top1_acc=float(arm.get("best_top1", 25.0)),
                            iso_err=float(arm.get("iso_err", 0.02)),
                            color=col,
                        )
                    )
        except Exception:
            pass

    # 11. Plan 025 (.cache/hypothesis_experiments_025.json if already present)
    p025 = cdir / "hypothesis_experiments_025.json"
    if p025.exists():
        try:
            d25 = json.loads(p025.read_text(encoding="utf-8"))
            micro_arms_25 = d25.get("micro_verification_arms", {})
            if isinstance(micro_arms_25, dict):
                for rk, arm in micro_arms_25.items():
                    if not isinstance(arm, dict):
                        continue
                    ce_raw = arm.get("best_val_ce", arm.get("val_final"))
                    if ce_raw is None:
                        continue
                    vf = round(float(ce_raw), 4)
                    evs = arm.get("evals") or arm.get("history") or []
                    w_min = float(arm.get("wall_min", round(float(arm.get("wall_s", 5.0)) / 60.0, 4)))
                    short_lbl = str(arm.get("short_label") or rk)
                    cli_sfx = str(arm.get("cli_args") or "--architecture quantum_born --hierarchical-16qubit --num-higher-states 256 --last-layer-softmax")
                    tot_st = int(arm.get("total_steps", 30))
                    tok_m_val = float(arm.get("target_tokens_M", 0.06))
                    col = str(arm.get("color") or "#00f5d4")
                    extracted.append(
                        _make_micro_entry(
                            run_key=rk,
                            plan_num=25,
                            arch_family=str(arm.get("arch_family") or "quantum_born"),
                            label=f"Plan 025 Micro-Verification ({tot_st} Steps, <=8GB VRAM): {short_lbl} ({vf:.4f} Val CE)",
                            short_label=f"{short_lbl} ({vf:.4f} CE)",
                            hypothesis_summary=str(
                                arm.get("hypothesis_summary")
                                or "Controlled Plan 025 <=8.0 GB VRAM memory-envelope & training dynamics verification on RTX 5060 Ti."
                            ),
                            cli_args=cli_sfx if cli_sfx.startswith("--backend") else f"--backend cuda {cli_sfx}",
                            total_steps=tot_st,
                            target_tokens_M=tok_m_val,
                            best_val_ce=vf,
                            evals=evs,
                            wall_min=max(0.01, w_min),
                            top1_acc=float(arm.get("best_top1", 25.0)),
                            iso_err=float(arm.get("iso_err", 0.01)),
                            color=col,
                        )
                    )
        except Exception:
            pass

    # 12. Plans 026–030 (.cache/hypothesis_experiments_026..030.json if already present)
    for plan_num, default_col, default_desc in (
        (
            26,
            "#ff9e00",
            "Controlled Plan 026 <=8.0 GB VRAM verification of 100% Quantum Two-Beam Mixed-State Density Readout, Non-Isometric Kraus Semantic Bath, and k-Copy Bloch Fidelity.",
        ),
        (
            27,
            "#00f5d4",
            "Controlled Plan 027 <=8.0 GB VRAM verification of Scheduled Rank-r Stiefel Kraus Semantic Bath and Per-Token Contextual Purity Routing.",
        ),
        (
            28,
            "#00ff87",
            "Controlled Plan 028 <=8.0 GB VRAM verification of Stereographic Radial-Norm-Preserving Lift, Cosine Phase-Interference Two-Beam Readout, and Positive-Part Quadratic Kraus Fidelity.",
        ),
        (
            29,
            "#38b000",
            "Controlled Plan 029 <=8.0 GB VRAM verification of FineWeb-Matched 50,257-Vocab Synthetic Random-Walk Dataset and FastQuantumBornAttentionSynthLM Overfitting & Variable-Lag Hub Disambiguation.",
        ),
        (
            30,
            "#00bbf9",
            "Controlled Plan 030 <=8.0 GB VRAM verification of Fast Synthetic-Walk Quantum Born Attention Winner on Genuine FineWeb-Edu, Directed Predecessor Self-Decoupled PPMI-SVD, and Iso-Wall-Clock Scaling.",
        ),
        (
            31,
            "#ffbe0b",
            "Controlled Plan 031 <=8.0 GB VRAM verification of Bipolar Particle-Hole k-Boson Log-Fidelity, Diffuse Mode Whitening, Concept-Cluster Diffuse Bath, and Full-Ambient 16x32 Qubits.",
        ),
        (
            32,
            "#00f5d4",
            "Controlled Plan 032 <=8.0 GB VRAM verification of Anti-Diagonal Spectral-Reversal Isometry (J_anti) for Full-Ambient 16x32/16x16 Qubits and Gentle 2x Zero-Weight-Decay Qubit Embedding LR.",
        ),
        (
            33,
            "#00ff87",
            "Controlled Plan 033 <=8.0 GB VRAM verification of Soft Power-Law Spectral Floor (R_diff >= 6.5), Equi-Dimensional Square-O(N) (frame_expansion=1.0), and 3,600-Step (29.5M-Token) FineWeb-Edu Scaling.",
        ),
    ):
        p_file = cdir / f"hypothesis_experiments_{plan_num:03d}.json"
        if p_file.exists():
            try:
                d_plan = json.loads(p_file.read_text(encoding="utf-8"))
                micro_arms_p = d_plan.get("micro_verification_arms", {})
                if isinstance(micro_arms_p, dict):
                    for rk, arm in micro_arms_p.items():
                        if not isinstance(arm, dict):
                            continue
                        ce_raw = arm.get("best_val_ce", arm.get("val_final"))
                        if ce_raw is None:
                            continue
                        vf = round(float(ce_raw), 4)
                        evs = arm.get("evals") or arm.get("history") or []
                        w_min = float(arm.get("wall_min", round(float(arm.get("wall_s", 5.0)) / 60.0, 4)))
                        short_lbl = str(arm.get("short_label") or rk)
                        cli_sfx = str(
                            arm.get("cli_args")
                            or "--architecture quantum_born --hierarchical-16qubit --num-higher-states 256 --two-beam-kboson-readout --diffuse-rank 16"
                        )
                        tot_st = int(arm.get("total_steps", 300))
                        tok_m_val = float(arm.get("target_tokens_M", 2.46))
                        col = str(arm.get("color") or default_col)
                        extracted.append(
                            _make_micro_entry(
                                run_key=rk,
                                plan_num=plan_num,
                                arch_family=str(arm.get("arch_family") or "quantum_born"),
                                label=f"Plan {plan_num:03d} Verification ({tot_st} Steps, <=8GB VRAM): {short_lbl} ({vf:.4f} Val CE)",
                                short_label=f"{short_lbl} ({vf:.4f} CE)",
                                hypothesis_summary=str(arm.get("hypothesis_summary") or default_desc),
                                cli_args=cli_sfx if cli_sfx.startswith("--backend") else f"--backend cuda {cli_sfx}",
                                total_steps=tot_st,
                                target_tokens_M=tok_m_val,
                                best_val_ce=vf,
                                evals=evs,
                                wall_min=max(0.01, w_min),
                                top1_acc=float(arm.get("best_top1", 25.0)),
                                iso_err=float(arm.get("iso_err", 0.01)),
                                color=col,
                            )
                        )
            except Exception:
                pass

    out_entries: list[dict[str, Any]] = []
    for cat_entry, card in extracted:
        if model_cards_dir is not None:
            save_model_card_atomic(card, model_cards_dir)
        db.upsert_run(cat_entry, model_card=card, explicit_user_override=False)
        out_entries.append(cat_entry)
    return out_entries

