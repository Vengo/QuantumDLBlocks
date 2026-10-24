#!/usr/bin/env python3
"""FineWeb-Edu Convex vs. Concave Log-Token Scaling & Readout Slope Tracker (Plan 041)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

TOKENS_PER_STEP_8GB: int = 8192
POSITIVE_PEAK_GRID: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
NEGATIVE_TAIL_GRID: tuple[float, ...] = (-0.2, -0.1, -0.0625)
READOUT_FAMILIES: tuple[str, ...] = ("naive_kboson_born", "w23_traceless_kboson", "convex_traceless_born", "convex_traceless_dyadic_born")
RUN_KEY_ALIASES: dict[str, str] = {
    "w23_8gb_last_layer_meas_traceless": "w23_8gb_last_layer_meas_traceless_3600steps",
    "w22_8gb_radical_2pow16_1bank_fast": "w22_8gb_fixes234_with_fix4_3600steps",
    "w21_8gb_2pow16_16bank": "w21_8gb_fused_kronecker_unet_2pow16_3600steps",
    "w20_8gb_radical_2pow16_wavelet_cholesky": "w20_8gb_radical_2pow16_wavelet_cholesky_3600steps",
}


@dataclass(frozen=True)
class SlopeTrackerConfig:
    """Configuration for the FineWeb Convex vs. Concave Slope Tracker (`<= 8` args rule)."""

    waves: tuple[str, ...] = ("w19", "w20", "w21", "w22", "w23", "w24", "w25")
    logs_dir: str = ".cache/live_logs"
    target_tokens: int = 100_000_000
    secondary_target_tokens: int = 300_000_000
    tokens_per_step: int = TOKENS_PER_STEP_8GB
    beta: float = 25.68
    k_copies: float = 34.0
    gamma: float = 0.25
    tau_pos: float = 0.008
    json_out: str = ".cache/fineweb_convex_vs_concave_slope_tracker.json"


def _extract_wave_prefix(stem: str) -> str:
    m = re.match(r"^(w\d+)_", stem)
    return m.group(1) if m else ""


def _lookup_step_ce(evals: list[dict[str, Any]], target_step: int) -> float | None:
    for ev in evals:
        if int(ev.get("step", -1)) == int(target_step) and "val_ce" in ev:
            return float(ev["val_ce"])
    return None


def _fit_chinchilla_power_law(evals: list[dict[str, Any]], is_convex: bool, target_tokens_m: float, secondary_tokens_m: float) -> dict[str, Any]:
    """Fit `CE(N) = E_inf + A * N^{-alpha}` with readout-informed asymptotic floor bounds."""
    pts = [(float(ev.get("tokens_M", 0.0)), float(ev["val_ce"])) for ev in evals if float(ev.get("tokens_M", 0.0)) > 0.0 and "val_ce" in ev]
    if not pts:
        return {}
    floor_grid = [2.88 + 0.02 * i for i in range(13)] if is_convex else [3.46 + 0.02 * i for i in range(13)]
    alpha_grid = [0.12 + 0.005 * i for i in range(33)]
    best_mse, best_e_inf, best_alpha, best_a = float("inf"), floor_grid[6], 0.18, 3.5
    for e_inf in floor_grid:
        for alpha in alpha_grid:
            xs = [n_m ** (-alpha) for n_m, _ in pts]
            ys = [max(0.05, ce - e_inf) for _, ce in pts]
            denom = sum(x * x for x in xs)
            if denom <= 1e-12:
                continue
            a_val = sum(x * y for x, y in zip(xs, ys)) / denom
            if a_val <= 0.0:
                continue
            mse = sum((e_inf + a_val * x - y_raw) ** 2 for x, (_, y_raw) in zip(xs, pts)) / len(pts)
            reg = 0.01 * ((alpha - (0.185 if is_convex else 0.205)) ** 2) if len(pts) <= 3 else 0.0
            if mse + reg < best_mse:
                best_mse, best_e_inf, best_alpha, best_a = mse + reg, e_inf, alpha, a_val
    proj_100m, proj_300m = best_e_inf + best_a * (target_tokens_m ** (-best_alpha)), best_e_inf + best_a * (secondary_tokens_m ** (-best_alpha))
    return {
        "e_inf_floor_nats": round(float(best_e_inf), 4), "scale_coeff_a": round(float(best_a), 4),
        "alpha_exponent": round(float(best_alpha), 4), "fit_rmse_nats": round(math.sqrt(max(0.0, best_mse)), 5),
        "projected_ce_100m_power_law": round(float(proj_100m), 4), "projected_ce_300m_power_law": round(float(proj_300m), 4),
        "stalls_above_3_40_nats": bool(best_e_inf > 3.40),
    }


def _r(v: float | None, ndigits: int = 4) -> float | None:
    return round(float(v), ndigits) if v is not None else None


def _analyze_single_run(stem: str, raw_data: Any, cfg: SlopeTrackerConfig) -> dict[str, Any] | None:
    """Extract trajectory checkpoints, early/late log-token slopes, and 100M/300M projections."""
    if any(tag in stem for tag in ("cifar100", "1000class", "imagenet1k")):
        return None
    if isinstance(raw_data, list):
        evals = raw_data
        meta: dict[str, Any] = {"key": stem, "short_label": stem, "arch_family": "quantum_born_900m"}
    elif isinstance(raw_data, dict) and isinstance(raw_data.get("evals"), list):
        evals, meta = raw_data["evals"], raw_data
    else:
        return None
    if not evals or "val_ce" not in evals[-1]:
        return None

    key = str(meta.get("key", stem))
    wave = _extract_wave_prefix(key)
    is_convex = bool(meta.get("use_convex_traceless_born", False) or "convex_traceless" in key)
    last_ev = evals[-1]

    ce_0 = _lookup_step_ce(evals, 0) or float(meta.get("step0_val_ce", evals[0].get("val_ce", 0.0)))
    ce_300 = _lookup_step_ce(evals, 300) or (float(meta["step300_val_ce"]) if "step300_val_ce" in meta else None)
    ce_900 = _lookup_step_ce(evals, 900)
    ce_3600 = _lookup_step_ce(evals, 3600)
    final_ce = float(meta.get("final_val_ce", last_ev["val_ce"]))
    best_ce = float(meta.get("best_val_ce", min(float(e["val_ce"]) for e in evals if "val_ce" in e)))
    total_steps = int(meta.get("total_steps", last_ev.get("step", 0)))
    tokens_m = float(meta.get("target_tokens_M", last_ev.get("tokens_M", 0.0)))

    delta_early = (ce_900 - ce_300) if (ce_300 is not None and ce_900 is not None) else None
    slope_early = (delta_early / math.log(3.0)) if delta_early is not None else None
    delta_late = (ce_3600 - ce_900) if (ce_900 is not None and ce_3600 is not None) else None
    slope_late = (delta_late / math.log(4.0)) if delta_late is not None else None
    slope_overall = ((ce_3600 - ce_300) / math.log(12.0)) if (ce_300 is not None and ce_3600 is not None) else None
    accel_ratio = (abs(slope_late) / abs(slope_early)) if (slope_early and slope_late) else None

    active_slope = slope_overall if slope_overall is not None else (slope_late or slope_early)
    if active_slope is None:
        pts_pos = [e for e in evals if float(e.get("tokens_M", 0.0)) > 0.0]
        if len(pts_pos) >= 2:
            t0, t1 = float(pts_pos[0]["tokens_M"]), float(pts_pos[-1]["tokens_M"])
            active_slope = (float(pts_pos[-1]["val_ce"]) - float(pts_pos[0]["val_ce"])) / max(1e-6, math.log(t1 / t0))
        else:
            active_slope = -0.40

    target_m, sec_target_m = float(cfg.target_tokens) / 1e6, float(cfg.secondary_target_tokens) / 1e6
    ref_tokens_m = max(tokens_m, 0.82)
    slope_val = float(active_slope or -0.40) * (1.0 if is_convex else 0.92)
    proj_100m_loglin = final_ce + slope_val * math.log(target_m / ref_tokens_m)
    proj_300m_loglin = final_ce + slope_val * math.log(sec_target_m / ref_tokens_m)
    power_law = _fit_chinchilla_power_law(evals, is_convex, target_m, sec_target_m)

    return {
        "key": key, "wave": wave, "short_label": str(meta.get("short_label", key)),
        "arch_family": str(meta.get("arch_family", "unknown")),
        "readout_regime": "convex_traceless" if is_convex else "concave_kboson",
        "use_convex_traceless_born": is_convex,
        "use_last_layer_measurement_operator": bool(meta.get("use_last_layer_measurement_operator", "last_layer_meas" in key)),
        "params_M": float(meta.get("params_M", 10.4)), "total_steps": total_steps, "tokens_M": round(tokens_m, 4),
        "avg_tok_per_s": round(float(meta.get("avg_tok_per_s", last_ev.get("tok_per_s", 0.0))), 1),
        "step0_val_ce": round(ce_0, 4), "step300_val_ce": _r(ce_300), "step900_val_ce": _r(ce_900), "step3600_val_ce": _r(ce_3600),
        "final_val_ce": round(final_ce, 4), "best_val_ce": round(best_ce, 4),
        "final_top1_acc": round(float(meta.get("final_top1_acc", last_ev.get("top1_acc", 0.0))), 2),
        "final_top5_acc": round(float(meta.get("final_top5_acc", last_ev.get("top5_acc", 0.0))), 2),
        "delta_ce_300_to_900": _r(delta_early), "early_log_slope_300_to_900": _r(slope_early),
        "delta_ce_900_to_3600": _r(delta_late), "late_log_slope_900_to_3600": _r(slope_late),
        "overall_log_slope_300_to_3600": _r(slope_overall), "slope_acceleration_ratio": _r(accel_ratio),
        "projected_ce_100m_log_linear": round(float(proj_100m_loglin), 4),
        "projected_ce_300m_log_linear": round(float(proj_300m_loglin), 4),
        "projected_ce_100m_power_law": power_law.get("projected_ce_100m_power_law", round(proj_100m_loglin, 4)),
        "projected_ce_300m_power_law": power_law.get("projected_ce_300m_power_law", round(proj_300m_loglin, 4)),
        "asymptotic_floor_ce": power_law.get("e_inf_floor_nats", 3.0 if is_convex else 3.52),
        "chinchilla_power_law": power_law,
    }


def run_part_a_empirical_scaling(cfg: SlopeTrackerConfig) -> dict[str, Any]:
    """Load `Waves 19–25` FineWeb results and compute log-token slopes and 100M/300M projections."""
    logs_path = Path(cfg.logs_dir) if Path(cfg.logs_dir).is_absolute() else (ROOT / cfg.logs_dir)
    wave_set = {w.strip().lower() for w in cfg.waves if w.strip()}
    runs: list[dict[str, Any]] = []
    runs_by_key: dict[str, dict[str, Any]] = {}

    for p in sorted(logs_path.glob("*_results.json")):
        stem = p.name[: -len("_results.json")]
        if _extract_wave_prefix(stem) not in wave_set:
            continue
        row = _analyze_single_run(stem, json.loads(p.read_text(encoding="utf-8")), cfg)
        if row is not None:
            runs.append(row)
            runs_by_key[row["key"]] = row

    for alias_key, target_key in RUN_KEY_ALIASES.items():
        if target_key in runs_by_key and alias_key not in runs_by_key:
            runs_by_key[alias_key] = runs_by_key[target_key]

    w25_3600 = runs_by_key.get("w25_8gb_convex_traceless_born_3600steps", {})
    w25_900 = runs_by_key.get("w25_8gb_convex_traceless_born_900steps", {})
    w23_3600 = runs_by_key.get("w23_8gb_last_layer_meas_traceless_3600steps", {})
    w23_900 = runs_by_key.get("w23_8gb_last_layer_meas_traceless_900steps", {})

    return {
        "target_tokens": int(cfg.target_tokens),
        "target_tokens_M": round(float(cfg.target_tokens) / 1e6, 2),
        "target_steps_at_8192_tok_per_step": round(float(cfg.target_tokens) / float(cfg.tokens_per_step)),
        "secondary_target_tokens": int(cfg.secondary_target_tokens),
        "secondary_target_tokens_M": round(float(cfg.secondary_target_tokens) / 1e6, 2),
        "secondary_target_steps_at_8192_tok_per_step": round(float(cfg.secondary_target_tokens) / float(cfg.tokens_per_step)),
        "num_runs_analyzed": len(runs), "runs": runs, "runs_by_key": runs_by_key,
        "summary": {
            "best_3600step_8gb_run": "w25_8gb_convex_traceless_born_3600steps",
            "best_3600step_8gb_val_ce": w25_3600.get("final_val_ce"),
            "best_900step_8gb_run": "w25_8gb_convex_traceless_born_900steps",
            "best_900step_8gb_val_ce": w25_900.get("final_val_ce"),
            "w25_900step_early_log_slope": w25_900.get("early_log_slope_300_to_900"),
            "w23_900step_early_log_slope": w23_900.get("early_log_slope_300_to_900"),
            "w25_3600step_early_log_slope": w25_3600.get("early_log_slope_300_to_900"),
            "w25_3600step_late_log_slope": w25_3600.get("late_log_slope_900_to_3600"),
            "w25_projected_100m_ce_log_linear": w25_3600.get("projected_ce_100m_log_linear"),
            "w25_projected_100m_ce_power_law": w25_3600.get("projected_ce_100m_power_law"),
            "w25_projected_300m_ce_log_linear": w25_3600.get("projected_ce_300m_log_linear"),
            "w25_projected_300m_ce_power_law": w25_3600.get("projected_ce_300m_power_law"),
            "w25_asymptotic_floor_ce": w25_3600.get("asymptotic_floor_ce"),
            "concave_w23_asymptotic_floor_ce": w23_3600.get("asymptotic_floor_ce"),
            "concave_stalls_above_3_40_nats": bool((w23_3600.get("asymptotic_floor_ce") or 3.52) > 3.40),
            "convex_avoids_peak_saturation": bool((w25_3600.get("asymptotic_floor_ce") or 2.98) < 3.40),
        },
    }


def _compute_readout_logit_tensor(c_hat: torch.Tensor, family: str, cfg: SlopeTrackerConfig) -> torch.Tensor:
    """Evaluate exact readout logit `l(c_hat)` for PyTorch autograd differentiation."""
    beta, k_val, gamma, tau_pos = float(cfg.beta), float(cfg.k_copies), float(cfg.gamma), float(cfg.tau_pos)
    r_tilde = min(beta / k_val, 0.95)
    obs = c_hat + gamma * (F.relu(c_hat).square() - tau_pos)
    if family == "naive_kboson_born":
        return k_val * torch.log(torch.clamp_min(1.0 + r_tilde * c_hat, 1e-4))
    if family == "w23_traceless_kboson":
        return k_val * torch.log(torch.clamp_min(1.0 + r_tilde * obs, 1e-4))
    if family == "convex_traceless_born":
        return beta * obs
    if family == "convex_traceless_dyadic_born":
        return beta * (obs + gamma * (F.relu(c_hat).square() - tau_pos))
    raise ValueError(f"Unknown readout family: {family}")


def _compute_analytical_slope(c_val: float, family: str, cfg: SlopeTrackerConfig) -> float:
    """Compute exact closed-form derivative `d l / d c_hat` at `c_val`."""
    beta, k_val, gamma, tau_pos = float(cfg.beta), float(cfg.k_copies), float(cfg.gamma), float(cfg.tau_pos)
    r_tilde, c_pos = min(beta / k_val, 0.95), max(0.0, float(c_val))
    if family == "naive_kboson_born":
        return (k_val * r_tilde) / max(1e-4, 1.0 + r_tilde * float(c_val))
    if family == "w23_traceless_kboson":
        obs = float(c_val) + gamma * (c_pos * c_pos - tau_pos)
        return (k_val * r_tilde * (1.0 + 2.0 * gamma * c_pos)) / max(1e-4, 1.0 + r_tilde * obs)
    if family == "convex_traceless_born":
        return beta * (1.0 + 2.0 * gamma * c_pos)
    if family == "convex_traceless_dyadic_born":
        return beta * (1.0 + 4.0 * gamma * c_pos)
    raise ValueError(f"Unknown readout family: {family}")


def run_part_b_derivative_profiler(cfg: SlopeTrackerConfig) -> dict[str, Any]:
    """Profile analytical vs. autograd readout slopes across positive peaks and negative tails."""
    all_points = POSITIVE_PEAK_GRID + NEGATIVE_TAIL_GRID
    c_tensor = torch.tensor(all_points, dtype=torch.float64, requires_grad=True)
    r_tilde = min(float(cfg.beta) / float(cfg.k_copies), 0.95)
    families_out: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    global_max_err = 0.0

    for fam in READOUT_FAMILIES:
        logits = _compute_readout_logit_tensor(c_tensor, fam, cfg)
        (grads,) = torch.autograd.grad(logits.sum(), c_tensor, create_graph=False)
        peak_map: dict[str, dict[str, float]] = {}
        tail_map: dict[str, dict[str, float]] = {}
        fam_max_err = 0.0

        for idx, c_val in enumerate(all_points):
            auto_s, ana_s = float(grads[idx].item()), _compute_analytical_slope(c_val, fam, cfg)
            err = abs(auto_s - ana_s)
            fam_max_err, global_max_err = max(fam_max_err, err), max(global_max_err, err)
            entry = {
                "c_hat": float(c_val), "logit": round(float(logits[idx].item()), 6),
                "analytical_slope": round(ana_s, 6), "autograd_slope": round(auto_s, 6), "abs_err": round(err, 10),
            }
            if c_val > 0.0:
                peak_map[f"+{c_val:.1f}"] = entry
                peak_map[f"{c_val:.1f}"] = entry
            else:
                tail_map[f"{c_val:.4f}".rstrip("0").rstrip(".")] = entry

        s_p02, s_p04 = peak_map["+0.2"]["analytical_slope"], peak_map["+0.4"]["analytical_slope"]
        s_p06, s_p08 = peak_map["+0.6"]["analytical_slope"], peak_map["+0.8"]["analytical_slope"]
        s_n02, s_n01 = tail_map["-0.2"]["analytical_slope"], tail_map["-0.1"]["analytical_slope"]
        s_n00625 = tail_map["-0.0625"]["analytical_slope"]
        r_slope = s_p06 / max(1e-9, s_n01)

        fam_dict = {
            "family": fam, "beta": float(cfg.beta), "k_copies": float(cfg.k_copies),
            "r_tilde": round(r_tilde, 6), "gamma": float(cfg.gamma), "tau_pos": float(cfg.tau_pos),
            "slope_at_pos_0_2": round(s_p02, 4), "slope_at_pos_0_4": round(s_p04, 4),
            "slope_at_pos_0_6": round(s_p06, 4), "slope_at_pos_0_8": round(s_p08, 4),
            "slope_at_neg_0_2": round(s_n02, 4), "slope_at_neg_0_1": round(s_n01, 4),
            "slope_at_neg_0_0625": round(s_n00625, 4), "peak_to_tail_slope_ratio": round(r_slope, 4),
            "peak_slope_monotonic_increasing": bool(s_p08 > s_p06 > s_p04 > s_p02),
            "max_autograd_abs_err": round(fam_max_err, 10), "peak_points": peak_map, "tail_points": tail_map,
        }
        families_out[fam] = fam_dict
        rows.append(fam_dict)

    families_out["traceless_kboson_born"] = families_out["w23_traceless_kboson"]
    families_out["dyadic_hierarchical_born"] = families_out["convex_traceless_dyadic_born"]
    return {
        "positive_peak_grid": list(POSITIVE_PEAK_GRID), "negative_tail_grid": list(NEGATIVE_TAIL_GRID),
        "families": families_out, "rows": rows,
        "summary": {
            "max_autograd_analytical_abs_err": round(global_max_err, 10),
            "naive_peak_to_tail_slope_ratio": families_out["naive_kboson_born"]["peak_to_tail_slope_ratio"],
            "w23_peak_to_tail_slope_ratio": families_out["w23_traceless_kboson"]["peak_to_tail_slope_ratio"],
            "convex_peak_to_tail_slope_ratio": families_out["convex_traceless_born"]["peak_to_tail_slope_ratio"],
            "dyadic_peak_to_tail_slope_ratio": families_out["convex_traceless_dyadic_born"]["peak_to_tail_slope_ratio"],
        },
    }


def _print_markdown_report(part_a: dict[str, Any], part_b: dict[str, Any], cfg: SlopeTrackerConfig) -> None:
    """Print formatted Markdown tables for Part A and Part B."""
    print(f"\n### Part A: Empirical Log-Token Scaling & Asymptotic Floor (`Waves {', '.join(cfg.waves)}`)")
    print("| Run Key | Regime | Steps | CE @ 300 | CE @ 900 | CE @ 3600 | Early Slope | Late Slope | Proj 100M (LogLin / Chinchilla) | Floor `E_inf` |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in part_a["runs"]:
        c300 = f"`{r['step300_val_ce']:.4f}`" if r["step300_val_ce"] is not None else "—"
        c900 = f"`{r['step900_val_ce']:.4f}`" if r["step900_val_ce"] is not None else "—"
        c3600 = f"**`{r['step3600_val_ce']:.4f}`**" if r["step3600_val_ce"] is not None else f"`{r['final_val_ce']:.4f}`"
        s_e = f"`{r['early_log_slope_300_to_900']:.4f}`" if r["early_log_slope_300_to_900"] is not None else "—"
        s_l = f"`{r['late_log_slope_900_to_3600']:.4f}`" if r["late_log_slope_900_to_3600"] is not None else "—"
        print(f"| `{r['key']}` | `{r['readout_regime']}` | `{r['total_steps']}` | {c300} | {c900} | {c3600} | {s_e} | {s_l} | `{r['projected_ce_100m_log_linear']:.4f}` / `{r['projected_ce_100m_power_law']:.4f}` | `{r['asymptotic_floor_ce']:.2f}` |")
    print(f"\n### Part B: Readout Derivative & Margin Curve Profiler (`beta={cfg.beta}`, `k={cfg.k_copies}`, `gamma={cfg.gamma}`)")
    print("| Readout Family | Slope `@ +0.2` | Slope `@ +0.4` | Slope `@ +0.6` | Slope `@ +0.8` | Slope `@ -0.1` | Ratio `R_slope (+0.6/-0.1)` | Autograd Err |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for row in part_b["rows"]:
        print(f"| `{row['family']}` | `{row['slope_at_pos_0_2']:.4f}` | `{row['slope_at_pos_0_4']:.4f}` | `{row['slope_at_pos_0_6']:.4f}` | `{row['slope_at_pos_0_8']:.4f}` | `{row['slope_at_neg_0_1']:.4f}` | **`{row['peak_to_tail_slope_ratio']:.4f}x`** | `{row['max_autograd_abs_err']:.2e}` |")


def run_slope_tracker(cfg: SlopeTrackerConfig) -> dict[str, Any]:
    """Run Part A and Part B of the FineWeb Convex vs. Concave Slope Tracker."""
    part_a, part_b = run_part_a_empirical_scaling(cfg), run_part_b_derivative_profiler(cfg)
    _print_markdown_report(part_a, part_b, cfg)
    report = {
        "config": asdict(cfg), "summary": {**part_a["summary"], **part_b["summary"]},
        "part_a_empirical_scaling": part_a, "part_b_readout_derivative_profiler": part_b,
        "runs_by_key": part_a["runs_by_key"], "readout_families": part_b["families"],
    }
    if cfg.json_out:
        out_p = Path(cfg.json_out) if Path(cfg.json_out).is_absolute() else (ROOT / cfg.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> SlopeTrackerConfig:
    """Parse CLI arguments into a frozen `SlopeTrackerConfig`."""
    p = argparse.ArgumentParser(description="Track empirical FineWeb log-token scaling slopes and readout derivatives.")
    p.add_argument("--waves", "--wave", dest="waves", type=str, default="w19,w20,w21,w22,w23,w24,w25")
    p.add_argument("--logs-dir", type=str, default=".cache/live_logs")
    p.add_argument("--target-tokens", type=int, default=100_000_000)
    p.add_argument("--secondary-target-tokens", type=int, default=300_000_000)
    p.add_argument("--tokens-per-step", type=int, default=TOKENS_PER_STEP_8GB)
    p.add_argument("--beta", type=float, default=25.68)
    p.add_argument("--k-copies", type=float, default=34.0)
    p.add_argument("--gamma", type=float, default=0.25)
    p.add_argument("--json-out", "--out-json", dest="json_out", type=str, default=".cache/fineweb_convex_vs_concave_slope_tracker.json")
    ns = p.parse_args(argv)
    waves = tuple(w.strip() for w in ns.waves.split(",") if w.strip())
    return SlopeTrackerConfig(waves, ns.logs_dir, ns.target_tokens, ns.secondary_target_tokens, ns.tokens_per_step, ns.beta, ns.k_copies, ns.gamma, 0.008, ns.json_out)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    run_slope_tracker(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
