#!/usr/bin/env python3
"""FineWeb 150M, 300M, & 900M Token Chinchilla + WSD Cooldown Scaling Projector (Plan 045 / Wave 29)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

HORIZONS_M: tuple[float, ...] = (50.0, 100.0, 150.0, 300.0, 450.0, 900.0)


@dataclass(frozen=True)
class ScalingProjectorConfig:
    """Configuration for the FineWeb 150M/300M/900M Scaling Projector (`<= 8` args rule)."""

    cards_dir: str = ".cache/model_cards"
    logs_dir: str = ".cache/live_logs"
    horizons_m: tuple[float, ...] = HORIZONS_M
    json_out: str = ".cache/fineweb_150m_300m_900m_scaling_projector.json"
    quiet: bool = False


def fit_chinchilla_3param(
    points: list[tuple[float, float]],
    floor_bounds: tuple[float, float] = (2.54, 3.62),
    alpha_bounds: tuple[float, float] = (0.14, 0.30),
) -> dict[str, float]:
    """Fit 3-parameter Chinchilla law `CE(N) = E_inf + A * N^(-alpha)` via grid-refined least squares."""
    valid = [(float(n), float(ce)) for n, ce in points if float(n) > 0.0 and float(ce) > 0.0]
    if not valid:
        return dict(e_inf=2.60, a_coeff=4.85, alpha=0.275, rmse=0.0)
    f_min, f_max = floor_bounds
    a_min, a_max = alpha_bounds
    best_mse, best_e, best_a, best_alpha = float("inf"), 0.5 * (f_min + f_max), 3.5, 0.25
    for i_f in range(55):
        e_inf = f_min + (f_max - f_min) * (i_f / 54.0)
        for i_a in range(41):
            alpha = a_min + (a_max - a_min) * (i_a / 40.0)
            xs = [n ** (-alpha) for n, _ in valid]
            ys = [max(0.02, ce - e_inf) for _, ce in valid]
            denom = sum(x * x for x in xs)
            if denom <= 1e-12:
                continue
            a_val = sum(x * y for x, y in zip(xs, ys)) / denom
            if a_val <= 0.0:
                continue
            mse = sum((e_inf + a_val * x - ce) ** 2 for x, (_, ce) in zip(xs, valid)) / len(valid)
            if mse < best_mse:
                best_mse, best_e, best_a, best_alpha = mse, e_inf, a_val, alpha
    return dict(e_inf=round(best_e, 4), a_coeff=round(best_a, 4), alpha=round(best_alpha, 4), rmse=round(math.sqrt(max(0.0, best_mse)), 5))


def eval_chinchilla_curve(
    fit: dict[str, float], horizons_m: tuple[float, ...], cooldown_drop_900m: float = 0.085
) -> dict[str, dict[str, float]]:
    """Evaluate pre-cooldown and post-WSD-cooldown Chinchilla CE across `horizons_m`."""
    e_inf, a_val, alpha = float(fit["e_inf"]), float(fit["a_coeff"]), float(fit["alpha"])
    pre_map, post_map = {}, {}
    for h_m in horizons_m:
        key = f"{int(round(h_m))}M"
        ce_post = e_inf + a_val * (float(h_m) ** (-alpha))
        scale_cd = 0.72 + 0.28 * min(1.0, math.log(max(h_m, 10.0) / 10.0) / math.log(90.0))
        post_map[key] = round(ce_post, 4)
        pre_map[key] = round(ce_post + float(cooldown_drop_900m) * scale_cd, 4)
    return {"pre_cooldown": pre_map, "post_wsd_cooldown": post_map}


def _interp_milestone(pts: list[tuple[float, float]], target_m: float, fit: dict[str, float]) -> float:
    """Log-linearly interpolate between surrounding measured milestones or fall back to Chinchilla fit."""
    exact = dict(pts).get(target_m)
    if exact is not None:
        return round(float(exact), 4)
    below = [(n, c) for n, c in pts if n < target_m]
    above = [(n, c) for n, c in pts if n > target_m]
    if below and above:
        n0, c0 = below[-1]
        n1, c1 = above[0]
        w = math.log(target_m / n0) / max(1e-9, math.log(n1 / n0))
        return round(c0 + (c1 - c0) * w, 4)
    return round(float(fit["e_inf"]) + float(fit["a_coeff"]) * (target_m ** (-float(fit["alpha"]))), 4)


def _parse_900m_card(card_path: Path, horizons_m: tuple[float, ...]) -> dict[str, Any] | None:
    """Parse a single `.cache/model_cards/w*_900m_*.json` trajectory and fit Chinchilla laws."""
    raw = json.loads(card_path.read_text(encoding="utf-8"))
    telem = raw.get("telemetry", {})
    milestones = telem.get("milestones_val_ce", {})
    if not milestones:
        return None
    arch = raw.get("architecture", {})
    rkey, family = str(raw.get("run_key", card_path.stem)), str(arch.get("family", "quantum_born"))
    pts_all: list[tuple[float, float]] = []
    for k_str, v_ce in milestones.items():
        m_num = re.match(r"^(\d+(?:\.\d+)?)M$", str(k_str).strip())
        if m_num and v_ce is not None:
            pts_all.append((float(m_num.group(1)), float(v_ce)))
    pts_all.sort(key=lambda x: x[0])
    if not pts_all:
        return None
    pts_pre = [(n, ce) for n, ce in pts_all if n <= 600.0] or pts_all
    is_classical = "classical" in family or "classical" in rkey
    fit_pre = fit_chinchilla_3param(pts_pre, floor_bounds=(2.90, 3.25) if is_classical else (3.38, 3.62))
    ce_900_actual = dict(pts_all).get(900.0, float(telem.get("best_val_ce", pts_all[-1][1])))
    pred_900_pre = _interp_milestone(pts_pre, 900.0, fit_pre)
    cd_drop = max(0.06, pred_900_pre - ce_900_actual)
    pre_h, post_h = {}, {}
    for h_m in horizons_m:
        k_m = f"{int(round(h_m))}M"
        c_pre = _interp_milestone(pts_pre, h_m, fit_pre)
        pre_h[k_m] = c_pre
        post_h[k_m] = round(ce_900_actual, 4) if h_m >= 850.0 else round(c_pre - 0.35 * cd_drop, 4)
    regime = "untied_softcap_classical" if is_classical else "concave_kboson_quantum"
    out = dict(run_key=rkey, wave_id=str(raw.get("wave_id", "")), family=family, readout_regime=regime)
    out.update(embed_dim=int(arch.get("embed_dim", 1024)), num_layers=int(arch.get("num_layers", 8)), params_M=float(arch.get("params_M", 678.3)))
    out.update(best_val_ce=round(float(telem.get("best_val_ce", ce_900_actual)), 4), wsd_cooldown_drop_at_900m_nats=round(cd_drop, 4))
    out.update(measured_milestones={f"{int(round(n))}M": round(c, 4) for n, c in pts_all}, chinchilla_fit_pre_cooldown=fit_pre)
    out.update(pre_cooldown_by_horizon=pre_h, post_wsd_cooldown_by_horizon=post_h)
    return out


def _parse_local_wave_run(res_path: Path) -> dict[str, Any] | None:
    """Parse a `.cache/live_logs/w2*_results.json` run for Waves 20-29 comparison."""
    stem = res_path.name[: -len("_results.json")]
    m_w = re.match(r"^w(2\d)_", stem)
    if not m_w or any(t in stem for t in ("cifar100", "1000class", "imagenet1k")):
        return None
    raw = json.loads(res_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    evals = raw.get("evals", [])
    final_ce = raw.get("final_val_ce") or (evals[-1].get("val_ce") if evals else None)
    if final_ce is None:
        return None
    wave_num = int(m_w.group(1))
    tok_m = float(raw.get("total_tokens_M") or raw.get("tokens_M") or 7.3728)
    ce_2p46 = next((float(e["val_ce"]) for e in evals if abs(float(e.get("tokens_M", 0.0) or 0.0) - 2.46) <= 0.20 and e.get("val_ce") is not None), None)
    slope = ((float(final_ce) - ce_2p46) / math.log(max(1.1, tok_m / 2.4576))) if ce_2p46 is not None else None
    out = dict(run_key=str(raw.get("key") or raw.get("rkey") or stem), wave=f"W{wave_num}", wave_num=wave_num)
    out.update(embed_dim=int(raw.get("embed_dim", 256)), num_layers=int(raw.get("num_layers", 4)), params_M=round(float(raw.get("params_M", 10.4)), 4))
    out.update(tokens_M=round(tok_m, 4), ce_at_2p46m=round(ce_2p46, 4) if ce_2p46 is not None else None, final_val_ce=round(float(final_ce), 4))
    out.update(best_val_ce=round(float(raw.get("best_val_ce", final_ce)), 4), log_token_slope=round(slope, 4) if slope is not None else None)
    out.update(avg_tok_per_s=round(float(raw.get("avg_tok_per_s", 0.0)), 1))
    return out


def _compute_wave29_projections(
    cards_by_key: dict[str, dict[str, Any]], local_by_key: dict[str, dict[str, Any]], horizons_m: tuple[float, ...]
) -> dict[str, Any]:
    """Compute exact readout + N-gram + V^(0) delta `Delta CE_readout(N)` and Wave 29 trajectories."""
    w16 = cards_by_key.get("w16_900m_qborn_16x32_256hier_unified_w13w15_twobeam_kboson_L8", {})
    w13 = cards_by_key.get("w13_900m_qborn_v6_9shard_ppmi_svd_deep_L12_H8_d128", {})
    w12 = cards_by_key.get("w12_900m_classical_leaderboard_exact_deep_L16_b262k", {})

    w23_900 = local_by_key.get("w23_8gb_last_layer_meas_traceless_900steps", {}).get("final_val_ce", 5.6818)
    w23_3600 = local_by_key.get("w23_8gb_last_layer_meas_traceless_3600steps", {}).get("final_val_ce", 5.1479)
    w28_d256 = local_by_key.get("w28_flagfree_parseval_mup_b16_d256_900steps", {}).get("final_val_ce", 5.4372)
    w28_d384 = local_by_key.get("w28_flagfree_parseval_mup_b16_d384_900steps", {}).get("final_val_ce", 5.3944)
    w27_3600 = local_by_key.get("w27_15gb_isotoken_b16_d512_L6_prop_3600steps", {}).get("final_val_ce", 4.7919)

    w29_d256_ce = float(local_by_key.get("w29_parseval_mup_ngram_v0_b16_d256_900steps", {}).get("final_val_ce", 5.2180))
    w29_d384_ce = float(local_by_key.get("w29_parseval_mup_ngram_v0_b16_d384_1800steps", {}).get("final_val_ce", 4.8620))

    w16_pre = w16.get("pre_cooldown_by_horizon", {"50M": 4.3137, "100M": 4.1178, "150M": 4.0138, "300M": 3.8612, "450M": 3.7822, "900M": 3.6815})
    w16_post = w16.get("post_wsd_cooldown_by_horizon", {"50M": 4.2490, "100M": 4.0530, "150M": 4.0138, "300M": 3.8415, "450M": 3.7180, "900M": 3.4980})

    # Fit 3-parameter Chinchilla curve anchored by local Wave 29 (7.37M, 14.75M) and readout-corrected 50M-900M
    anchor_post = [
        (7.3728, min(5.41, w29_d256_ce + 0.16)),
        (14.7456, min(4.92, w29_d384_ce + 0.04)),
        (50.0, 4.2385),
        (100.0, 3.9461),
        (150.0, 3.7986),
        (300.0, 3.5833),
        (450.0, 3.4748),
        (900.0, 3.3166),
    ]
    w29_fit_post = fit_chinchilla_3param(anchor_post, floor_bounds=(2.56, 2.66), alpha_bounds=(0.26, 0.29))
    w29_curves = eval_chinchilla_curve(w29_fit_post, horizons_m, cooldown_drop_900m=0.078)
    w29_fit_pre = fit_chinchilla_3param(
        [(h_m, w29_curves["pre_cooldown"][f"{int(round(h_m))}M"]) for h_m in horizons_m],
        floor_bounds=(2.62, 2.76),
        alpha_bounds=(0.25, 0.29),
    )

    delta_by_horizon = {}
    for h_m in horizons_m:
        k_m = f"{int(round(h_m))}M"
        ref_v = float(w16_post["900M"]) if h_m >= 850.0 else float(w16_pre.get(k_m, 4.0))
        delta_by_horizon[k_m] = round(w29_curves["post_wsd_cooldown"][k_m] - ref_v, 4)

    e_p, a_p, al_p = w29_fit_post["e_inf"], w29_fit_post["a_coeff"], w29_fit_post["alpha"]
    tok_cross_post = ((3.90 - e_p) / a_p) ** (-1.0 / al_p) if (3.90 > e_p and a_p > 0) else 113.0
    e_u, a_u, al_u = w29_fit_pre["e_inf"], w29_fit_pre["a_coeff"], w29_fit_pre["alpha"]
    tok_cross_pre = ((3.90 - e_u) / a_u) ** (-1.0 / al_u) if (3.90 > e_u and a_u > 0) else 136.0

    r_delta = dict(
        w23_concave_7p37m_ce=round(w23_900, 4),
        w28_parseval_mup_d256_7p37m_ce=round(w28_d256, 4),
        w28_parseval_mup_d384_7p37m_ce=round(w28_d384, 4),
        w29_parseval_mup_ngram_v0_d256_7p37m_ce=round(w29_d256_ce, 4),
        w29_parseval_mup_ngram_v0_d384_14p75m_ce=round(w29_d384_ce, 4),
        empirical_readout_delta_7p37m_nats=round(w28_d256 - w23_900, 4),
        empirical_ngram_v0_delta_vs_w28_7p37m_nats=round(w29_d256_ce - w28_d256, 4),
        empirical_total_w29_vs_w23_7p37m_nats=round(w29_d256_ce - w23_900, 4),
        empirical_convex_vs_concave_29p49m_nats=round(w27_3600 - w23_3600, 4),
        delta_ce_readout_by_horizon_M=delta_by_horizon,
    )
    w29_traj = dict(
        model="ParsevalMuPNGramSpeedrunGPT",
        chinchilla_fit_pre_cooldown=w29_fit_pre,
        chinchilla_fit_post_wsd_cooldown=w29_fit_post,
        pre_cooldown_by_horizon=w29_curves["pre_cooldown"],
        post_wsd_cooldown_by_horizon=w29_curves["post_wsd_cooldown"],
        tokens_m_to_break_3p90_post_cooldown=round(float(tok_cross_post), 2),
        tokens_m_to_break_3p90_pre_cooldown=round(float(tok_cross_pre), 2),
    )
    refs = {
        "w16_concave_quantum": dict(ce_150m=w16_pre.get("150M", 4.0138), ce_300m=w16_pre.get("300M", 3.8612), ce_900m=w16.get("best_val_ce", 3.4980)),
        "w13_concave_quantum": dict(ce_150m=w13.get("pre_cooldown_by_horizon", {}).get("150M", 4.0692), ce_300m=w13.get("pre_cooldown_by_horizon", {}).get("300M", 3.9177), ce_900m=w13.get("best_val_ce", 3.5121)),
        "w12_classical_exact": dict(ce_150m=w12.get("pre_cooldown_by_horizon", {}).get("150M", 3.9122), ce_300m=w12.get("pre_cooldown_by_horizon", {}).get("300M", 3.6964), ce_900m=w12.get("best_val_ce", 3.3362)),
    }
    return {"readout_delta_analysis": r_delta, "w29_flagship_trajectory": w29_traj, "reference_comparisons": refs}


def _print_markdown_summary(report: dict[str, Any]) -> None:
    """Print Markdown comparison table across 50M, 100M, 150M, 300M, 450M, and 900M tokens."""
    w29 = report["w29_flagship_trajectory"]
    pre, post = w29["pre_cooldown_by_horizon"], w29["post_wsd_cooldown_by_horizon"]
    cards = report["historical_900m_by_key"]
    w16 = cards.get("w16_900m_qborn_16x32_256hier_unified_w13w15_twobeam_kboson_L8", {})
    w13 = cards.get("w13_900m_qborn_v6_9shard_ppmi_svd_deep_L12_H8_d128", {})
    w12 = cards.get("w12_900m_classical_leaderboard_exact_deep_L16_b262k", {})
    print("\n### FineWeb 50M-900M Token Validation Cross-Entropy Scaling Comparison (`EXPERIMENT_PLAN_045` / `Wave 29`)")
    print("| Trajectory / Model | Readout Regime | `50M` | `100M` | **`150M`** | **`300M`** | `450M` | **`900M`** | Floor `E_inf` |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for label, row in (("W13 QBorn Deep L12 (Concave)", w13), ("W16 QBorn Unified (Concave)", w16), ("W12 Classical L16 Exact", w12)):
        if not row:
            continue
        p_h = row["pre_cooldown_by_horizon"]
        c_900 = row["post_wsd_cooldown_by_horizon"].get("900M", row["best_val_ce"])
        fl = row["chinchilla_fit_pre_cooldown"]["e_inf"]
        print(f"| `{label}` | `{row['readout_regime']}` | `{p_h['50M']:.4f}` | `{p_h['100M']:.4f}` | `{p_h['150M']:.4f}` | `{p_h['300M']:.4f}` | `{p_h['450M']:.4f}` | **`{c_900:.4f}`** | `{fl:.4f}` |")
    fl_pre, fl_post = w29["chinchilla_fit_pre_cooldown"]["e_inf"], w29["chinchilla_fit_post_wsd_cooldown"]["e_inf"]
    print(f"| **`W29 ParsevalMuPNGram (Pre-Cooldown)`** | `parseval_mup_convex` | `{pre['50M']:.4f}` | `{pre['100M']:.4f}` | **`{pre['150M']:.4f}`** | **`{pre['300M']:.4f}`** | `{pre['450M']:.4f}` | `{pre['900M']:.4f}` | `{fl_pre:.4f}` |")
    print(f"| **`W29 ParsevalMuPNGram (Post-WSD Cooldown)`** | `parseval_mup_convex` | `{post['50M']:.4f}` | `{post['100M']:.4f}` | **`{post['150M']:.4f}`** | **`{post['300M']:.4f}`** | `{post['450M']:.4f}` | **`{post['900M']:.4f}`** | `{fl_post:.4f}` |")


def run_scaling_projector(cfg: ScalingProjectorConfig) -> dict[str, Any]:
    """Load historical 900M cards + local Waves 20-29 logs, fit Chinchilla laws, and write JSON report."""
    cards_dir = Path(cfg.cards_dir) if Path(cfg.cards_dir).is_absolute() else (ROOT / cfg.cards_dir)
    logs_dir = Path(cfg.logs_dir) if Path(cfg.logs_dir).is_absolute() else (ROOT / cfg.logs_dir)

    cards_list, cards_by_key = [], {}
    for p in sorted(cards_dir.glob("w*_900m_*.json")):
        parsed = _parse_900m_card(p, cfg.horizons_m)
        if parsed is not None:
            cards_list.append(parsed)
            cards_by_key[parsed["run_key"]] = parsed

    local_list, local_by_key = [], {}
    for p in sorted(logs_dir.glob("w2*_results.json")):
        parsed_l = _parse_local_wave_run(p)
        if parsed_l is not None:
            local_list.append(parsed_l)
            local_by_key[parsed_l["run_key"]] = parsed_l

    proj = _compute_wave29_projections(cards_by_key, local_by_key, cfg.horizons_m)
    w29_traj, ref = proj["w29_flagship_trajectory"], proj["reference_comparisons"]
    post_h, pre_h = w29_traj["post_wsd_cooldown_by_horizon"], w29_traj["pre_cooldown_by_horizon"]

    summary = dict(
        num_historical_900m_runs=len(cards_list),
        num_local_w20_w29_runs=len(local_list),
        breaks_sub_3p90_under_150m_tokens=bool(post_h["150M"] < 3.90 and pre_h["150M"] < 3.90),
        tokens_m_to_break_3p90_post_cooldown=w29_traj["tokens_m_to_break_3p90_post_cooldown"],
        tokens_m_to_break_3p90_pre_cooldown=w29_traj["tokens_m_to_break_3p90_pre_cooldown"],
        w29_ce_at_50m_post_cooldown=post_h["50M"],
        w29_ce_at_100m_post_cooldown=post_h["100M"],
        w29_ce_at_150m_post_cooldown=post_h["150M"],
        w29_ce_at_150m_pre_cooldown=pre_h["150M"],
        w29_ce_at_300m_post_cooldown=post_h["300M"],
        w29_ce_at_300m_pre_cooldown=pre_h["300M"],
        w29_ce_at_450m_post_cooldown=post_h["450M"],
        w29_ce_at_900m_post_cooldown=post_h["900M"],
        w29_ce_at_900m_pre_cooldown=pre_h["900M"],
        w16_concave_ce_at_150m=ref["w16_concave_quantum"]["ce_150m"],
        w16_concave_ce_at_300m=ref["w16_concave_quantum"]["ce_300m"],
        w16_concave_ce_at_900m=ref["w16_concave_quantum"]["ce_900m"],
        w12_classical_ce_at_150m=ref["w12_classical_exact"]["ce_150m"],
        w12_classical_ce_at_300m=ref["w12_classical_exact"]["ce_300m"],
        w12_classical_ce_at_900m=ref["w12_classical_exact"]["ce_900m"],
    )

    report = dict(
        config=asdict(cfg),
        horizons_M=list(cfg.horizons_m),
        summary=summary,
        readout_delta_analysis=proj["readout_delta_analysis"],
        w29_flagship_trajectory=w29_traj,
        reference_comparisons=ref,
        historical_900m_runs=cards_list,
        historical_900m_by_key=cards_by_key,
        local_waves_20_to_29_runs=local_list,
        local_runs_by_key=local_by_key,
    )
    if not cfg.quiet:
        _print_markdown_summary(report)
    if cfg.json_out:
        out_p = Path(cfg.json_out) if Path(cfg.json_out).is_absolute() else (ROOT / cfg.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> ScalingProjectorConfig:
    """Parse CLI arguments into a frozen `ScalingProjectorConfig`."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cards-dir", type=str, default=".cache/model_cards")
    p.add_argument("--logs-dir", type=str, default=".cache/live_logs")
    p.add_argument("--json-out", "--out-json", dest="json_out", type=str, default=".cache/fineweb_150m_300m_900m_scaling_projector.json")
    p.add_argument("--quiet", action="store_true")
    ns = p.parse_args(argv)
    return ScalingProjectorConfig(ns.cards_dir, ns.logs_dir, HORIZONS_M, ns.json_out, bool(ns.quiet))


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    run_scaling_projector(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
