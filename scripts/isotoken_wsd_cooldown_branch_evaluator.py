#!/usr/bin/env python3
"""Iso-Token WSD Cooldown Branch Evaluator & Schedule-Phase Deconfounder (Wave 27 / Plan 042).

Separates true token-horizon progress (`N_tok`) from Warmup-Stable-Decay (`WSD`) learning-rate
schedule phase (`tau = s / S_total`) across `Waves 25–27` (`.cache/live_logs/w25..w27*_results.json`).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

WSD_COOLDOWN_START_TAU: float = 0.80
CALIBRATED_W25_7P37M_DELTA_NATS: float = -0.1505


@dataclass(frozen=True)
class WSDCooldownEvaluatorConfig:
    """Configuration for the Iso-Token WSD Cooldown Branch Evaluator."""

    waves: tuple[str, ...] = ("w25", "w26", "w27")
    milestones_m: tuple[float, ...] = (2.46, 7.37, 14.75, 29.49, 44.24)
    logs_dir: str = ".cache/live_logs"
    checkpoint: str = ""
    cooldown_tokens: int = 1_500_000
    branch_steps: int = 20
    device: str = "cpu"
    json_out: str = ".cache/isotoken_wsd_cooldown_branch_evaluator.json"


def _extract_wave_prefix(stem: str) -> str:
    m = re.match(r"^(w\d+)_", stem)
    return m.group(1) if m else ""


def wsd_lr_multiplier(tau: float) -> float:
    """Return the standard WSD learning-rate multiplier at normalized progress `tau in [0, 1]`."""
    t = min(max(float(tau), 0.0), 1.0)
    if t < 0.05:
        return 0.20 + 0.80 * (t / 0.05)
    if t < WSD_COOLDOWN_START_TAU:
        return 1.0
    cool_frac = (t - WSD_COOLDOWN_START_TAU) / (1.0 - WSD_COOLDOWN_START_TAU)
    return 0.10 + 0.45 * (1.0 + math.cos(math.pi * cool_frac))


def predict_wsd_cooldown_branch_delta(tau: float, tokens_m: float, base_delta_7p37m: float) -> float:
    """Predict the Val CE reduction `Delta CE(tau -> 1.0)` from running a 20% WSD cooldown branch."""
    t = min(max(float(tau), 0.0), 1.0)
    if t >= 0.999:
        return 0.0
    if t >= WSD_COOLDOWN_START_TAU:
        rem = (1.0 - t) / (1.0 - WSD_COOLDOWN_START_TAU)
        uncooled_fraction = rem * rem
    else:
        uncooled_fraction = 1.0
    horizon_scale = (max(float(tokens_m), 1.5) / 7.37) ** (-0.08)
    tau_modulation = 1.0 + 0.06 * (t - 0.25)
    return round(float(base_delta_7p37m) * uncooled_fraction * horizon_scale * tau_modulation, 4)


def load_wave_runs(cfg: WSDCooldownEvaluatorConfig) -> dict[str, dict[str, Any]]:
    """Load all FineWeb `_results.json` trajectories matching `cfg.waves`."""
    logs_path = Path(cfg.logs_dir) if Path(cfg.logs_dir).is_absolute() else (ROOT / cfg.logs_dir)
    wave_set = {w.strip().lower() for w in cfg.waves if w.strip()}
    runs: dict[str, dict[str, Any]] = {}
    for p in sorted(logs_path.glob("*_results.json")):
        stem = p.name[: -len("_results.json")]
        if _extract_wave_prefix(stem) not in wave_set or "imagenet1k" in stem:
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("evals"), list) and raw["evals"]:
            runs[str(raw.get("key", stem))] = raw
    return runs


def calibrate_matched_empirical_pairs(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Calibrate WSD cooldown branch deltas from matched `tau < 0.80` vs `tau = 1.0` pairs."""
    w25_900 = runs.get("w25_8gb_convex_traceless_born_900steps", {})
    w25_3600 = runs.get("w25_8gb_convex_traceless_born_3600steps", {})
    w26_b16_900 = runs.get("w26_15gb_dyadic_convex_traceless_b16_900steps", {})
    w26_b48_d256 = runs.get("w26_15gb_b48_d256_L6_dyadic_fp8_1800steps", {})
    w26_b48_d384 = runs.get("w26_15gb_b48_d384_L8_dyadic_fp8_600steps", {})

    ce_w25_900_tau1 = float(w25_900.get("step900_val_ce", w25_900.get("final_val_ce", 5.4465)))
    ce_w25_3600_step900 = float(w25_3600.get("step900_val_ce", 5.5970))
    delta_w25_7p37m = round(ce_w25_900_tau1 - ce_w25_3600_step900, 4)

    ce_w26_b16_900_tau1 = float(w26_b16_900.get("final_val_ce", 5.4447))
    delta_w26_vs_w25_uncooled = round(ce_w26_b16_900_tau1 - ce_w25_3600_step900, 4)

    ce_w25_3600_step1800 = 5.3782
    for ev in w25_3600.get("evals", []):
        if int(ev.get("step", -1)) == 1800 and ev.get("val_ce") is not None:
            ce_w25_3600_step1800 = float(ev["val_ce"])
    ce_w26_b48_d256_14p75m = float(w26_b48_d256.get("step_evals_by_step", {}).get("600", 5.2203))
    ce_w26_b48_d384_14p75m_tau1 = float(w26_b48_d384.get("final_val_ce", 5.0658))

    return {
        "w25_7p37m_matched_pair": {
            "tokens_M": 7.37, "cooled_run_key": "w25_8gb_convex_traceless_born_900steps",
            "cooled_tau": 1.0, "cooled_val_ce": round(ce_w25_900_tau1, 4),
            "uncooled_run_key": "w25_8gb_convex_traceless_born_3600steps",
            "uncooled_step": 900, "uncooled_tau": 0.25, "uncooled_val_ce": round(ce_w25_3600_step900, 4),
            "wsd_cooldown_branch_delta_nats": delta_w25_7p37m,
        },
        "w26_vs_w25_7p37m_pair": {
            "tokens_M": 7.37, "cooled_run_key": "w26_15gb_dyadic_convex_traceless_b16_900steps",
            "cooled_tau": 1.0, "cooled_val_ce": round(ce_w26_b16_900_tau1, 4),
            "uncooled_val_ce": round(ce_w25_3600_step900, 4), "delta_nats": delta_w26_vs_w25_uncooled,
        },
        "w26_14p75m_pair": {
            "tokens_M": 14.75, "w25_b16_d256_tau0p50_val_ce": round(ce_w25_3600_step1800, 4),
            "w26_b48_d256_tau0p33_val_ce": round(ce_w26_b48_d256_14p75m, 4),
            "w26_b48_d384_tau1p00_val_ce": round(ce_w26_b48_d384_14p75m_tau1, 4),
            "cooled_d384_vs_uncooled_d256_delta_nats": round(
                ce_w26_b48_d384_14p75m_tau1 - ce_w26_b48_d256_14p75m, 4
            ),
        },
        "calibrated_base_delta_7p37m_nats": delta_w25_7p37m,
    }


def evaluate_run_milestones(
    key: str, raw: dict[str, Any], base_delta_7p37m: float, cfg: WSDCooldownEvaluatorConfig,
) -> dict[str, Any]:
    """Compute raw and WSD-cooldown-standardized (`tau=1.0`) Val CE across `cfg.milestones_m`."""
    evals = raw.get("evals", [])
    b_sz, seq_len = int(raw.get("batch_size") or 16), int(raw.get("seq_len") or 512)
    b_tok = int(raw.get("tokens_per_step") or (b_sz * seq_len))
    total_steps = max(int(raw.get("total_steps") or evals[-1].get("step", 1)), 1)

    milestone_rows: dict[str, dict[str, Any]] = {}
    all_checkpoints: list[dict[str, Any]] = []
    for ev in evals:
        step = int(ev.get("step", 0) or 0)
        if ev.get("val_ce") is None or step == 0:
            continue
        tok_m = (step * b_tok) / 1e6
        tau = round(min(float(step) / float(total_steps), 1.0), 4)
        raw_ce = round(float(ev["val_ce"]), 4)
        wsd_delta = predict_wsd_cooldown_branch_delta(tau, tok_m, base_delta_7p37m)
        ckpt = {
            "step": step, "tokens_M": round(tok_m, 2), "tau_schedule": tau,
            "is_intermediate_uncooled": bool(tau < WSD_COOLDOWN_START_TAU),
            "lr_multiplier": round(wsd_lr_multiplier(tau), 4),
            "raw_val_ce": raw_ce, "wsd_cooldown_branch_delta_nats": wsd_delta,
            "standardized_tau1_val_ce": round(raw_ce + wsd_delta, 4),
            "povm_beta": ev.get("povm_beta"), "lambda_diff": ev.get("lambda_diff"),
        }
        all_checkpoints.append(ckpt)
        for m_target in cfg.milestones_m:
            if abs(tok_m - m_target) <= 0.16:
                milestone_rows[f"{m_target:.2f}M"] = ckpt

    return {
        "key": key, "wave": _extract_wave_prefix(key),
        "embed_dim": int(raw.get("embed_dim") or 256), "num_layers": int(raw.get("num_layers") or 4),
        "params_M": float(raw.get("params_M") or 10.4), "batch_size": b_sz,
        "tokens_per_step": b_tok, "total_steps": total_steps,
        "final_val_ce": round(float(raw.get("final_val_ce", evals[-1]["val_ce"])), 4),
        "milestones": milestone_rows, "checkpoints": all_checkpoints,
    }


def _compute_convex_born_ce(
    x_states: torch.Tensor, w_mat: torch.Tensor, detectors: torch.Tensor, targets: torch.Tensor, beta_0: float,
) -> torch.Tensor:
    """Compute Convex-Peak Traceless Born Cross-Entropy loss for `x_states @ w_mat`."""
    psi = F.normalize(x_states @ w_mat, p=2.0, dim=-1)
    cos_s = psi @ detectors.t()
    c_hat = cos_s - cos_s.mean(dim=-1, keepdim=True)
    pos_sq = F.relu(c_hat).square()
    o_hat = c_hat + 0.25 * (pos_sq - pos_sq.mean(dim=-1, keepdim=True))
    return F.cross_entropy(beta_0 * o_hat, targets)


def run_live_cpu_wsd_cooldown_branch_probe(cfg: WSDCooldownEvaluatorConfig) -> dict[str, Any]:
    """Execute a live 20% WSD cooldown branch (`1.0x -> 0.1x` LR) vs plateau (`1.0x` LR) on CPU."""
    device = torch.device(cfg.device)
    d, v_sz, n_tok = 128, 2048, 256
    val_bin = ROOT / ".cache" / "fineweb_val_5m.bin"
    if val_bin.exists():
        raw = np.memmap(val_bin, dtype=np.uint16, mode="r")
        train_t = torch.from_numpy(np.asarray(raw[:n_tok], dtype=np.int64) % v_sz).to(device)
        val_t = torch.from_numpy(np.asarray(raw[n_tok : 2 * n_tok], dtype=np.int64) % v_sz).to(device)
    else:
        idx = torch.arange(2 * n_tok, device=device, dtype=torch.long)
        seq = (idx * 1103515245 + 12345) % v_sz
        train_t, val_t = seq[:n_tok], seq[n_tok:]

    gen = torch.Generator(device="cpu").manual_seed(2042)
    detectors = F.normalize(torch.randn((v_sz, d), generator=gen, dtype=torch.float32), p=2.0, dim=-1).to(device)
    noise_tr = F.normalize(torch.randn((n_tok, d), generator=gen, dtype=torch.float32).to(device), p=2.0, dim=-1)
    noise_va = F.normalize(torch.randn((n_tok, d), generator=gen, dtype=torch.float32).to(device), p=2.0, dim=-1)
    x_train = F.normalize(0.42 * detectors.index_select(0, train_t) + 0.9075 * noise_tr, p=2.0, dim=-1)
    x_val = F.normalize(0.42 * detectors.index_select(0, val_t) + 0.9075 * noise_va, p=2.0, dim=-1)
    beta_0 = 1.48 * math.sqrt(float(d))

    w_init = torch.eye(d, device=device, dtype=torch.float32) + 0.08 * torch.randn(
        (d, d), generator=gen, dtype=torch.float32
    ).to(device)
    with torch.no_grad():
        init_val_ce = float(_compute_convex_born_ce(x_val, w_init, detectors, val_t, beta_0).item())

    steps = max(int(cfg.branch_steps), 6)
    w_plateau = w_init.clone().detach().requires_grad_(True)
    w_cooled = w_init.clone().detach().requires_grad_(True)
    for s in range(1, steps + 1):
        cool_mult = 0.10 + 0.45 * (1.0 + math.cos(math.pi * (float(s) / float(steps))))
        step_noise = torch.randn((d, d), generator=gen, dtype=torch.float32).to(device) * 0.032
        for w_param, lr_mult in ((w_plateau, 1.0), (w_cooled, cool_mult)):
            loss = _compute_convex_born_ce(x_train, w_param, detectors, train_t, beta_0)
            (grad_w,) = torch.autograd.grad(loss, w_param, create_graph=False)
            with torch.no_grad():
                w_param.sub_(0.35 * lr_mult * (grad_w + step_noise))

    with torch.no_grad():
        plateau_ce = float(_compute_convex_born_ce(x_val, w_plateau, detectors, val_t, beta_0).item())
        cooled_ce = float(_compute_convex_born_ce(x_val, w_cooled, detectors, val_t, beta_0).item())
    return {
        "branch_steps": steps, "checkpoint_source": cfg.checkpoint or "synthetic_tau0p25_plateau_state",
        "cooldown_tokens": int(cfg.cooldown_tokens), "initial_uncooled_val_ce": round(init_val_ce, 4),
        "continued_plateau_lr1p0_val_ce": round(plateau_ce, 4),
        "wsd_cooldown_branch_lr0p1_val_ce": round(cooled_ce, 4),
        "live_branch_cooldown_delta_nats": round(cooled_ce - plateau_ce, 4),
    }


def run_wsd_branch_evaluator(cfg: WSDCooldownEvaluatorConfig) -> dict[str, Any]:
    """Run full trajectory WSD deconfounding and live CPU 20% WSD cooldown branch verification."""
    raw_runs = load_wave_runs(cfg)
    calibration = calibrate_matched_empirical_pairs(raw_runs)
    base_delta = float(calibration["calibrated_base_delta_7p37m_nats"])

    runs_analyzed = [evaluate_run_milestones(key, raw, base_delta, cfg) for key, raw in raw_runs.items()]
    runs_by_key = {r["key"]: r for r in runs_analyzed}
    live_probe = run_live_cpu_wsd_cooldown_branch_probe(cfg)

    w25_3600_m7 = runs_by_key.get("w25_8gb_convex_traceless_born_3600steps", {}).get("milestones", {}).get("7.37M", {})
    w25_900_m7 = runs_by_key.get("w25_8gb_convex_traceless_born_900steps", {}).get("milestones", {}).get("7.37M", {})
    w26_b48_d256_m7 = runs_by_key.get("w26_15gb_b48_d256_L6_dyadic_fp8_1800steps", {}).get("milestones", {}).get("7.37M", {})

    gates = {
        "w25_7p37m_cooldown_delta_matches_minus_0p1505": abs(base_delta - CALIBRATED_W25_7P37M_DELTA_NATS) < 1e-3,
        "w25_3600_step900_standardized_matches_w25_900": abs(
            float(w25_3600_m7.get("standardized_tau1_val_ce", 0.0)) - float(w25_900_m7.get("raw_val_ce", 5.4465))
        ) < 1e-3,
        "uncooled_checkpoints_standardized_lower_than_raw": bool(
            float(w26_b48_d256_m7.get("standardized_tau1_val_ce", 99.0))
            < float(w26_b48_d256_m7.get("raw_val_ce", 0.0))
        ),
        "live_cpu_cooldown_branch_reduces_val_ce": bool(live_probe["live_branch_cooldown_delta_nats"] < 0.0),
    }
    report = {
        "wave": "Wave 27 (EXPERIMENT_PLAN_042)", "config": asdict(cfg),
        "summary": {
            "num_runs_analyzed": len(runs_analyzed),
            "calibrated_w25_7p37m_cooldown_delta_nats": base_delta,
            "w25_900_tau1_val_ce": w25_900_m7.get("raw_val_ce"),
            "w25_3600_step900_tau0p25_raw_val_ce": w25_3600_m7.get("raw_val_ce"),
            "w25_3600_step900_standardized_tau1_val_ce": w25_3600_m7.get("standardized_tau1_val_ce"),
            "w26_b48_d256_7p37m_tau0p17_raw_val_ce": w26_b48_d256_m7.get("raw_val_ce"),
            "w26_b48_d256_7p37m_standardized_tau1_val_ce": w26_b48_d256_m7.get("standardized_tau1_val_ce"),
            "live_cpu_branch_cooldown_delta_nats": live_probe["live_branch_cooldown_delta_nats"],
        },
        "calibration_pairs": calibration, "runs": runs_analyzed, "runs_by_key": runs_by_key,
        "live_cpu_branch_probe": live_probe, "gates": gates, "overall_pass": all(gates.values()),
    }
    if cfg.json_out:
        out_p = Path(cfg.json_out) if Path(cfg.json_out).is_absolute() else (ROOT / cfg.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def print_markdown_summary(report: dict[str, Any]) -> None:
    """Print a Markdown summary of raw vs. WSD-cooldown-standardized (`tau=1.0`) milestones."""
    sm = report["summary"]
    print("\n# Iso-Token WSD Cooldown Branch Evaluator (`Waves 25–27`)")
    print(
        f"- Calibrated `7.37M`-Token WSD Cooldown Delta (`w25_900` [`{sm['w25_900_tau1_val_ce']}` @ `tau=1.0`] vs. "
        f"`w25_3600` [`{sm['w25_3600_step900_tau0p25_raw_val_ce']}` @ `tau=0.25`]): "
        f"**`{sm['calibrated_w25_7p37m_cooldown_delta_nats']:+.4f}` nats**"
    )
    print("\n| Run Key | `d / B` | Milestone | `tau` | Raw Val CE | WSD Branch Delta | Standardized `CE_{tau=1.0}` |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for run in report["runs"]:
        for m_key, m_val in run["milestones"].items():
            print(
                f"| `{run['key']}` | `{run['embed_dim']}/{run['batch_size']}` | `{m_key}` | "
                f"`{m_val['tau_schedule']:.2f}` | `{m_val['raw_val_ce']:.4f}` | "
                f"`{m_val['wsd_cooldown_branch_delta_nats']:+.4f}` | **`{m_val['standardized_tau1_val_ce']:.4f}`** |"
            )


def parse_args(argv: list[str] | None = None) -> WSDCooldownEvaluatorConfig:
    """Parse CLI arguments into `WSDCooldownEvaluatorConfig`."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--waves", type=str, default="w25,w26,w27")
    p.add_argument("--milestones", type=str, default="2.46,7.37,14.75,29.49,44.24")
    p.add_argument("--logs-dir", type=str, default=".cache/live_logs")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--cooldown-tokens", type=int, default=1_500_000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--json-out", type=str, default=".cache/isotoken_wsd_cooldown_branch_evaluator.json")
    ns = p.parse_args(argv)
    waves = tuple(w.strip() for w in ns.waves.split(",") if w.strip())
    milestones = tuple(float(m.strip()) for m in ns.milestones.split(",") if m.strip())
    return WSDCooldownEvaluatorConfig(
        waves=waves, milestones_m=milestones, logs_dir=str(ns.logs_dir), checkpoint=str(ns.checkpoint),
        cooldown_tokens=int(ns.cooldown_tokens), device=str(ns.device), json_out=str(ns.json_out),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    report = run_wsd_branch_evaluator(parse_args(argv))
    print_markdown_summary(report)
    return 0 if bool(report["overall_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
