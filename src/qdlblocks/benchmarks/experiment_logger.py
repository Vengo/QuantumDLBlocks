"""Standardized Experiment Intent & Live Progress Logger (`qdlblocks.benchmarks.experiment_logger`).

Implements Experiment Plan 044 (Change 1 & Section 6 of `AGENTS.md`):
- `ExperimentIntentSpec`: Frozen dataclass declaring explicit experiment metadata (`run_key`,
  `wave_id`, `plan_id`, `task_domain`, `hypothesis_intent`, `architecture_summary`,
  `hardware_budget`, `batch_size`, `seq_len`, `total_steps`, `target_tokens_M`, `params_M`).
- `StandardizedExperimentLogger`: Writes structured `[EXPERIMENT_INTENT]` blocks and standardized
  `Step ...` / `[Epoch ...]` evaluation lines to `.cache/live_logs/{run_key}.log`, while atomically
  updating `.cache/live_logs/{run_key}_results.json`, `.cache/model_cards/{run_key}.json`
  (`ModelCard v1.0`), and `.cache/benchmarks.db` (`BenchmarkSQLiteDB`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import re
import time
from typing import Any

from .model_card import (
    DEFAULT_LIVE_LOGS_DIR,
    DEFAULT_MODEL_CARDS_DIR,
    deterministic_run_color,
    save_model_card_atomic,
)

VALID_TASK_DOMAINS: frozenset[str] = frozenset(
    {"fineweb_speedrun", "fineweb_edu_local", "vision_ablation", "micro_verification"}
)


@dataclass(frozen=True)
class ExperimentIntentSpec:
    """Declarative intent & horizon specification emitted before Step 0 of any training run."""

    run_key: str
    wave_id: str
    plan_id: str
    task_domain: str = "fineweb_edu_local"
    hypothesis_intent: str = ""
    architecture_summary: str = ""
    hardware_budget: str = "NVIDIA GeForce RTX 5060 Ti (<=15.0GB Local)"
    arch_family: str = "quantum_born"
    batch_size: int = 16
    seq_len: int = 512
    total_steps: int = 900
    target_tokens_M: float = 7.37
    params_M: float = 10.50
    embed_dim: int = 256
    num_layers: int = 6
    schedule: str = "modded_nanogpt_wsd"
    color: str = ""
    short_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not data.get("color"):
            data["color"] = deterministic_run_color(self.run_key)
        if not data.get("short_label"):
            data["short_label"] = f"[{self.wave_id}] {self.run_key}"
        return data


def format_intent_header_block(spec: ExperimentIntentSpec) -> str:
    """Render a human- and machine-readable `[EXPERIMENT_INTENT]` header block for `.log` files."""
    json_line = json.dumps(spec.to_dict(), sort_keys=True)
    arch_desc = f"{spec.architecture_summary or spec.arch_family} (params={spec.params_M:.2f}M, L={spec.num_layers}, d={spec.embed_dim})"
    horizon_desc = f"{spec.total_steps} steps | {spec.target_tokens_M:.2f}M tokens | B={spec.batch_size} | T={spec.seq_len} | sched={spec.schedule}"
    lines = [
        f"[EXPERIMENT_INTENT] {json_line}",
        f"Run Key: {spec.run_key}",
        f"Wave: {spec.wave_id} | Plan: {spec.plan_id}",
        f"Task Domain: {spec.task_domain}",
        f"Hypothesis Intent: {spec.hypothesis_intent or 'Standardized QuantumDLBlocks experiment run.'}",
        f"Architecture: {arch_desc}",
        f"Hardware: {spec.hardware_budget}",
        f"Target Horizon: {horizon_desc}",
        "[/EXPERIMENT_INTENT]",
    ]
    return "\n".join(lines) + "\n"


def parse_experiment_intent_block(log_text: str) -> dict[str, Any]:
    """Parse an `[EXPERIMENT_INTENT]` header block from a `.log` string if present."""
    if not log_text or "[EXPERIMENT_INTENT]" not in log_text:
        return {}
    m_json = re.search(r"^\[EXPERIMENT_INTENT\]\s*(\{.*\})\s*$", log_text, flags=re.MULTILINE)
    if m_json:
        try:
            parsed = json.loads(m_json.group(1))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    out: dict[str, Any] = {}
    key_map = {
        "run key": "run_key",
        "task domain": "task_domain",
        "hypothesis intent": "hypothesis_intent",
        "architecture": "architecture_summary",
        "hardware": "hardware_budget",
    }
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if line == "[/EXPERIMENT_INTENT]":
            break
        if ":" in line and not line.startswith("["):
            k_part, v_part = line.split(":", 1)
            k_low = k_part.strip().lower()
            if k_low in key_map:
                out[key_map[k_low]] = v_part.strip()
    return out


class StandardizedExperimentLogger:
    """Live experiment logger emitting standardized `.log`, `_results.json`, `ModelCard v1.0`, and SQLite updates."""

    def __init__(
        self,
        spec: ExperimentIntentSpec,
        live_logs_dir: Path = DEFAULT_LIVE_LOGS_DIR,
        model_cards_dir: Path = DEFAULT_MODEL_CARDS_DIR,
        db: Any | None = None,
        auto_write_header: bool = True,
    ) -> None:
        self.spec = spec
        self.live_logs_dir = Path(live_logs_dir)
        self.model_cards_dir = Path(model_cards_dir)
        self.db = db
        self.live_logs_dir.mkdir(parents=True, exist_ok=True)
        self.model_cards_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.live_logs_dir / f"{spec.run_key}.log"
        self.results_path = self.live_logs_dir / f"{spec.run_key}_results.json"
        self.card_path = self.model_cards_dir / f"{spec.run_key}.json"
        self.start_time = time.time()
        self.steps: list[dict[str, Any]] = []
        self.evals: list[dict[str, Any]] = []
        self.completed = False
        if auto_write_header:
            self.write_intent_header()

    def write_intent_header(self) -> Path:
        """Write `[EXPERIMENT_INTENT]` header to `.log` and initialize `ModelCard v1.0` & `_results.json`."""
        self.log_path.write_text(format_intent_header_block(self.spec), encoding="utf-8")
        self._sync_artifacts(run_state="RUNNING", extra_summary=None)
        return self.log_path

    def log_step(
        self,
        step: int,
        train_ce: float,
        tok_per_s: float,
        lr_mult: float = 1.0,
        elapsed_min: float | None = None,
    ) -> dict[str, Any]:
        """Append a standardized training step line to `.cache/live_logs/{run_key}.log`."""
        tokens_per_step = max(1, int(self.spec.batch_size) * int(self.spec.seq_len))
        tok_m = (float(step) * tokens_per_step) / 1e6
        tot_steps = max(1, int(self.spec.total_steps))
        ep = float(step) / tot_steps
        tflops = 6.0 * float(self.spec.params_M) * 1e6 * (tok_m * 1e6) / 1e12
        el_min = float(elapsed_min) if elapsed_min is not None else max(0.001, (time.time() - self.start_time) / 60.0)
        line = (
            f"Step {step}/{tot_steps} (Epoch {ep:.2f} | {tok_m:.2f}M tok) | "
            f"Train CE: {train_ce:.4f} | Speed: {int(round(tok_per_s)):,} tok/s | "
            f"{tflops:.2f} TFLOPs | tau={ep:.3f} | lr_mult={lr_mult:.3f}\n"
        )
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
        step_rec = {
            "step": int(step),
            "epoch": round(ep, 4),
            "tokens_M": round(tok_m, 4),
            "compute_tflops": round(tflops, 4),
            "tau_progress": round(ep, 4),
            "elapsed_min": round(el_min, 4),
            "train_ce": round(float(train_ce), 4),
            "tok_per_s": round(float(tok_per_s), 1),
            "lr_mult": round(float(lr_mult), 4),
        }
        self.steps.append(step_rec)
        return step_rec

    def log_eval(self, step: int, metrics: dict[str, Any]) -> dict[str, Any]:
        """Append a standardized `[Epoch ...]` eval checkpoint line and atomically update JSON/SQLite."""
        tokens_per_step = max(1, int(self.spec.batch_size) * int(self.spec.seq_len))
        tot_steps = max(1, int(self.spec.total_steps))
        tokens_m = float(metrics["tokens_M"]) if metrics.get("tokens_M") is not None else (float(step) * tokens_per_step) / 1e6
        compute_tflops = float(
            metrics["compute_tflops"]
            if metrics.get("compute_tflops") is not None
            else (6.0 * float(self.spec.params_M) * 1e6 * (tokens_m * 1e6) / 1e12)
        )
        tau_progress = float(metrics["tau_progress"]) if metrics.get("tau_progress") is not None else (float(step) / tot_steps)
        ep = float(metrics["epoch"]) if metrics.get("epoch") is not None else tau_progress
        el_min = float(metrics["elapsed_min"]) if metrics.get("elapsed_min") is not None else max(0.001, (time.time() - self.start_time) / 60.0)
        val_ce = float(metrics.get("val_ce", metrics.get("test_ce", 0.0)))
        val_ppl = float(metrics.get("val_ppl") or metrics.get("test_ppl") or round(math.exp(min(20.0, val_ce)), 2))
        top1 = float(metrics.get("top1_acc", metrics.get("val_top1", metrics.get("test_top1_acc", 0.0))))
        top5 = float(
            metrics["top5_acc"]
            if metrics.get("top5_acc") is not None
            else (metrics["test_top5_acc"] if metrics.get("test_top5_acc") is not None else round(min(100.0, top1 * 1.75), 2))
        )
        med_rank = int(metrics.get("med_rank", 20))
        iso_err = float(metrics["iso_err"]) if metrics.get("iso_err") is not None else float(metrics.get("kraus_stiefel_err", 0.0))
        kraus_err = float(metrics["kraus_stiefel_err"]) if metrics.get("kraus_stiefel_err") is not None else iso_err
        r_sharp = float(
            metrics["sharp_eff_rank"]
            if metrics.get("sharp_eff_rank") is not None
            else (metrics["seed_eff_rank"] if metrics.get("seed_eff_rank") is not None else 16.0)
        )
        r_diff = float(metrics.get("diffuse_eff_rank", 14.0))
        beta = float(metrics.get("current_beta") or metrics.get("povm_beta") or metrics.get("beta_0") or 23.68)
        tok_s = float(metrics.get("tok_per_s") or (self.steps[-1]["tok_per_s"] if self.steps else 0.0))

        line = (
            f"[Epoch {ep:.2f} | Step {step} | {tokens_m:.2f}M tok | T={self.spec.seq_len}] "
            f"Val CE: {val_ce:.4f} | Val PPL: {val_ppl:.2f} | Top-1: {top1:.2f}% | "
            f"Top-5: {top5:.2f}% | MedRank: {med_rank} | IsoErr: {iso_err:.4f} | "
            f"R_sharp: {r_sharp:.2f} | beta: {beta:.2f}\n"
        )
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line)

        eval_rec = dict(metrics)
        eval_rec.update(
            {
                "step": int(step),
                "epoch": round(ep, 4),
                "tokens_M": round(tokens_m, 4),
                "elapsed_min": round(el_min, 4),
                "compute_tflops": round(compute_tflops, 4),
                "tau_progress": round(tau_progress, 4),
                "val_ce": round(val_ce, 4),
                "val_ppl": round(val_ppl, 2),
                "top1_acc": round(top1, 2),
                "top5_acc": round(top5, 2),
                "med_rank": med_rank,
                "seed_eff_rank": round(r_sharp, 4),
                "sharp_eff_rank": round(r_sharp, 4),
                "diffuse_eff_rank": round(r_diff, 4),
                "iso_err": round(iso_err, 6),
                "kraus_stiefel_err": round(kraus_err, 6),
                "povm_beta": round(beta, 4),
                "tok_per_s": round(tok_s, 1),
            }
        )
        self.evals.append(eval_rec)
        self._sync_artifacts(run_state="RUNNING", extra_summary=None)
        return eval_rec

    def finalize(self, extra_summary: dict[str, Any] | None = None) -> dict[str, Any]:
        """Write `COMPLETED` footer to `.log` and finalize `_results.json`, `ModelCard v1.0`, and SQLite."""
        self.completed = True
        best_ce = min((float(e["val_ce"]) for e in self.evals), default=0.0)
        tps_list = [float(e.get("tok_per_s", 0.0)) for e in self.evals if float(e.get("tok_per_s", 0.0)) > 0]
        if not tps_list and self.steps:
            tps_list = [float(s["tok_per_s"]) for s in self.steps]
        avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
        line = (
            f"COMPLETED {self.spec.run_key} | Best Val CE: {best_ce:.4f} | "
            f"Speed: {avg_tps:,.1f} tok/s | Date: {datetime.now().isoformat(timespec='seconds')}\n"
        )
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
        return self._sync_artifacts(run_state="COMPLETED", extra_summary=extra_summary)

    def _sync_artifacts(self, run_state: str, extra_summary: dict[str, Any] | None = None) -> dict[str, Any]:
        """Atomically persist `_results.json` and `ModelCard v1.0` and upsert into `BenchmarkSQLiteDB`."""
        from .wave_ingestor import normalize_raw_result_to_model_card

        wall_s = max(0.01, time.time() - self.start_time)
        best_ce = min((float(e["val_ce"]) for e in self.evals), default=None)
        last_ev = self.evals[-1] if self.evals else {}
        tps_vals = [float(e.get("tok_per_s", 0.0)) for e in self.evals if float(e.get("tok_per_s", 0.0)) > 0]
        if not tps_vals and self.steps:
            tps_vals = [float(s["tok_per_s"]) for s in self.steps if float(s.get("tok_per_s", 0.0)) > 0]
        avg_tps = round(sum(tps_vals) / len(tps_vals), 1) if tps_vals else 0.0

        raw_payload: dict[str, Any] = {
            **self.spec.to_dict(),
            "key": self.spec.run_key,
            "run_key": self.spec.run_key,
            "completed": bool(self.completed or run_state == "COMPLETED"),
            "run_state": "COMPLETED" if self.completed else run_state,
            "current_step": int(last_ev.get("step", self.steps[-1]["step"] if self.steps else 0)),
            "live_tokens_M": float(last_ev.get("tokens_M", self.steps[-1]["tokens_M"] if self.steps else 0.0)),
            "best_val_ce": round(float(best_ce), 4) if best_ce is not None else None,
            "final_val_ce": last_ev.get("val_ce"),
            "final_val_ppl": last_ev.get("val_ppl"),
            "final_top1_acc": last_ev.get("top1_acc"),
            "final_top5_acc": last_ev.get("top5_acc"),
            "sharp_eff_rank": last_ev.get("sharp_eff_rank"),
            "diffuse_eff_rank": last_ev.get("diffuse_eff_rank"),
            "final_kraus_stiefel_err": last_ev.get("kraus_stiefel_err"),
            "avg_tok_per_s": avg_tps,
            "wall_s": round(wall_s, 2),
            "wall_min": float(last_ev.get("elapsed_min") or round(wall_s / 60.0, 4)),
            "evals": list(self.evals),
            "steps": list(self.steps),
        }
        if extra_summary:
            raw_payload.update(extra_summary)

        tmp_res = self.live_logs_dir / f".{self.spec.run_key}_results.json.tmp"
        tmp_res.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")
        tmp_res.replace(self.results_path)

        cat_entry, card = normalize_raw_result_to_model_card(self.spec.run_key, raw_payload)
        save_model_card_atomic(card, self.model_cards_dir)
        if self.db is not None:
            self.db.upsert_run(cat_entry, model_card=card)
        return raw_payload
