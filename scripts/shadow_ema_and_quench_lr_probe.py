"""Wave 63 (`EXPERIMENT_PLAN_083.md`): Hot + Shadow-EMA + Branch-and-Quench LR & Subsystem Curvature Probe.

Addresses two core optimization questions on `ExplicitDensityTomographicGPT` (100% `AdamW`-free):
1. **In-Advance Subsystem Curvature, Duty-Cycle, & Bounce Estimator (`Phase 1`)**:
   - Measures why the three parameter families have fundamentally different optimal learning-rate multipliers:
     - `wte_512` `(50304, 512)`: Sparse update duty cycle ($f_{\\text{active}} \\approx 4.6\\%$ per batch) and `L = 1` linear depth,
       allowing/requiring a hotter `2.0x–3.0x` step size per active hit.
     - `trunk` (`Stiefel / Unitary`, `L = 4` layers): `100%` update duty cycle and quadratic density conjugation
       ($U_h \\rho U_h^\\dagger$) compounded across 4 layers, favoring `1.0x–2.0x`.
     - `global_scalars` (`readout_temp`, `tomo_scale`, `filter_scale`): `100%` duty cycle with global rank-1 coupling across all
       `50,304` logits simultaneously, favoring `0.5x–1.0x`.
2. **Hot vs. Shadow-EMA vs. Branch-and-Quench Probe (`Phase 2`)**:
   - Tracks both the raw iterate `val_ce_hot` ($L(\\theta_t)$) and a manifold-retracted Shadow-EMA state `val_ce_ema`
     ($L(\\bar{\\theta}_{\\text{ema}})$) during a constant-LR plateau (`480` steps), followed by a `120`-step geodesic quench to `0.1x`.
   - Extracts a quantitative **Lower-LR Re-Run Estimator** from a single probe (such as `trunk=4.0x, vocab=1.0x`) using the
     thermal gap $\\Delta L_{\\text{thermal}} = L_{\\text{hot}} - L_{\\text{ema}}$ and the quench acceleration ratio.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.benchmarks.explicit_density_tomographic_512d_lab import (  # noqa: E402
    DEFAULT_SHALLOW_NGRAM_LR,
    DEFAULT_SHALLOW_SCALAR_LR,
    DEFAULT_TOK_NGRAM_BUCKETS,
    ExplicitDensityTomographicGPT,
    ExplicitDensityTomographicSpec,
    ExplicitDensityVariantMode,
    build_explicit_density_tomographic_optimizers,
    evaluate_explicit_density_tomographic_model,
    initialize_explicit_density_tomographic_model_,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import (  # noqa: E402
    LongSeqFineWebStream,
)
from qdlblocks.layers.impure_density_mirror_block import (  # noqa: E402
    _complex_newton_schulz5,
    _newton_schulz5_tall_real,
)
from qdlblocks.optim.vocab_manifold_muon import (  # noqa: E402
    project_rows_to_rms_sphere_,
)

# ---------------------------------------------------------------------------
# Top-of-File Named Constants (Rule 7 — Synchronized with EXPERIMENT_PLAN_083.md)
# ---------------------------------------------------------------------------
DEFAULT_CURVATURE_PROBE_STEPS: int = 50
DEFAULT_PLATEAU_STEPS: int = 240
DEFAULT_QUENCH_STEPS: int = 60
DEFAULT_PROBE_EVAL_INTERVAL: int = 60
DEFAULT_SHADOW_EMA_DECAY: float = 0.975
DEFAULT_QUENCH_FINAL_FRACTION: float = 0.10
DEFAULT_STEADY_MOMENTUM_BETA: float = 0.70
DEFAULT_PROBE_SEED: int = 1337
DIVERGENCE_CE_THRESHOLD: float = 13.5


@dataclass(frozen=True)
class SubsystemLRMultipliers:
    """Per-subsystem learning-rate multipliers for decoupled manifold and scalar probing (`<= 8` fields, Rule 2)."""

    run_id: str
    trunk_mult: float = 1.0
    vocab_mult: float = 1.0
    ngram_mult: float = 1.0
    scalar_mult: float = 1.0
    plateau_steps: int = DEFAULT_PLATEAU_STEPS
    quench_steps: int = DEFAULT_QUENCH_STEPS
    ema_decay: float = DEFAULT_SHADOW_EMA_DECAY


@torch.no_grad()
def update_shadow_ema_buffers_(
    ema_model: ExplicitDensityTomographicGPT,
    hot_model: ExplicitDensityTomographicGPT,
    ema_decay: float = DEFAULT_SHADOW_EMA_DECAY,
) -> None:
    """Perform fast in-place ambient lerp from `hot_model` into `ema_model` (`<= 8` args)."""
    alpha = float(1.0 - ema_decay)
    ema_params = [p.data for p in ema_model.parameters()]
    hot_params = [p.data for p in hot_model.parameters()]
    torch._foreach_lerp_(ema_params, hot_params, alpha)
    ema_model.unigram_bias.data.copy_(hot_model.unigram_bias.data)


@torch.no_grad()
def retract_shadow_ema_manifolds_(ema_model: ExplicitDensityTomographicGPT) -> None:
    """Retract `ema_model` parameters back onto $\\mathbb{S}^{511}$, $\\operatorname{St}(n, p)$, and $U(16)$ before evaluation."""
    project_rows_to_rms_sphere_(ema_model.wte_512)
    project_rows_to_rms_sphere_(ema_model.w_det_512)
    for w in ema_model.real_stiefel_parameters():
        if w.ndim == 2:
            if w.size(0) >= w.size(1):
                w.data.copy_(_newton_schulz5_tall_real(w.data).to(dtype=w.dtype))
            else:
                w.data.copy_(_newton_schulz5_tall_real(w.data.t()).t().to(dtype=w.dtype))
    for u_re, u_im in ema_model.complex_unitary_pairs():
        r_re, r_im = _complex_newton_schulz5(u_re.data, u_im.data)
        u_re.data.copy_(r_re.to(dtype=u_re.dtype))
        u_im.data.copy_(r_im.to(dtype=u_im.dtype))
    ema_model.restrict_scalars_()


@torch.no_grad()
def _step_subsystem_scaled_optimizers(
    model: ExplicitDensityTomographicGPT,
    optimizers: tuple[Any, ...],
    idx_in: torch.Tensor,
    targets: torch.Tensor,
    mults: SubsystemLRMultipliers,
    quench_scale: float = 1.0,
) -> None:
    """Step all 4 optimizers with independent `(trunk, vocab, ngram, scalar)` multipliers and `quench_scale` (`<= 8` args)."""
    q_trunk_opt, wte_opt, head_opt, geo_opt = optimizers
    t_scale = float(mults.trunk_mult * quench_scale)
    v_scale = float(mults.vocab_mult * quench_scale)
    n_scale = float(mults.ngram_mult * quench_scale)
    s_scale = float(mults.scalar_mult * quench_scale)

    q_trunk_opt.set_momentum(DEFAULT_STEADY_MOMENTUM_BETA)
    q_trunk_opt.set_lr_scale(t_scale)
    q_trunk_opt.step()

    active_ids = torch.unique(torch.cat([idx_in.reshape(-1), targets.reshape(-1)]))
    wte_opt.set_momentum(DEFAULT_STEADY_MOMENTUM_BETA)
    if len(wte_opt.param_groups) >= 2:
        wte_opt.param_groups[0]["lr"] = 0.060 * (v_scale / max(1e-8, v_scale))
        wte_opt.param_groups[1]["lr"] = DEFAULT_SHALLOW_NGRAM_LR * (n_scale / max(1e-8, v_scale))
    wte_opt.step_active(active_ids, lr_scale=v_scale)

    head_opt.set_momentum(DEFAULT_STEADY_MOMENTUM_BETA)
    head_opt.step_full(lr_scale=v_scale, active_idx=active_ids)

    geo_opt.set_momentum(0.80)
    geo_opt.step_active(active_idx=active_ids, lr_scale=s_scale)
    model.restrict_scalars_()


def run_subsystem_curvature_and_bounce_estimator(
    stream: LongSeqFineWebStream,
    steps: int = DEFAULT_CURVATURE_PROBE_STEPS,
) -> dict[str, Any]:
    """Measure update duty cycles, geodesic step angles, and lookahead gradient bounce cosines per subsystem (`<= 8` args)."""
    device = stream.device
    spec = ExplicitDensityTomographicSpec(
        variant_mode=ExplicitDensityVariantMode.T2_DENSITY_TOMOGRAPHY_PLUS_NOCLICK_AND_VACUUM,
        enable_density_highway=True, enable_tomographic_readout=True,
        enable_noclick_and_vacuum=True, enable_two_register_bilinear=False, seed=DEFAULT_PROBE_SEED,
    )
    torch.manual_seed(DEFAULT_PROBE_SEED)
    model = ExplicitDensityTomographicGPT(spec).to(device=device)
    initialize_explicit_density_tomographic_model_(model)
    optimizers = build_explicit_density_tomographic_optimizers(model)
    mults_1x = SubsystemLRMultipliers(run_id="CURVATURE_1X")

    stiefel_params = model.real_stiefel_parameters()
    unitary_params = [p for pair in model.complex_unitary_pairs() for p in pair]
    wte_param = model.wte_512
    wdet_param = model.w_det_512
    ngram_params = [model.tok_ngram_in.weight, model.tok_ngram_out.weight, model.tok_unigram_out.weight]
    fast_scalars = [model.readout_temp, model.tomo_scale, model.filter_scale]

    duty_vocab: list[float] = []
    duty_ngram: list[float] = []
    rel_step_trunk: list[float] = []
    rel_step_unitary: list[float] = []
    rel_step_vocab_active: list[float] = []
    rel_step_ngram_active: list[float] = []
    rel_step_scalars: list[float] = []
    bounce_cos_trunk: list[float] = []
    bounce_cos_vocab: list[float] = []
    bounce_cos_scalars: list[float] = []

    prev_dw_trunk: list[torch.Tensor] | None = None
    prev_dw_vocab: torch.Tensor | None = None
    prev_dw_scalars: list[torch.Tensor] | None = None

    model.train()
    for step in range(1, steps + 1):
        idx_in, targets = stream.sample_train_batch(step)
        for opt in optimizers:
            if opt is not None:
                opt.zero_grad(set_to_none=True)
        loss = model.forward_loss(idx_in, targets)
        loss.backward()
        del loss

        with torch.no_grad():
            if step > 5 and prev_dw_trunk is not None and prev_dw_vocab is not None and prev_dw_scalars is not None:
                num_t = sum(float((dw * (-p.grad.float())).sum().item()) for dw, p in zip(prev_dw_trunk, stiefel_params) if p.grad is not None)
                den_t = math.sqrt(
                    max(1e-12, sum(float(dw.square().sum().item()) for dw in prev_dw_trunk))
                    * max(1e-12, sum(float(p.grad.float().square().sum().item()) for p in stiefel_params if p.grad is not None))
                )
                bounce_cos_trunk.append(num_t / den_t)

                if wdet_param.grad is not None:
                    g_voc = -wdet_param.grad.float()
                    num_v = float((prev_dw_vocab * g_voc).sum().item())
                    den_v = math.sqrt(max(1e-12, float(prev_dw_vocab.square().sum().item()) * float(g_voc.square().sum().item())))
                    bounce_cos_vocab.append(num_v / den_v)

                num_s = sum(float((dw * (-p.grad.float())).sum().item()) for dw, p in zip(prev_dw_scalars, fast_scalars) if p.grad is not None)
                den_s = math.sqrt(
                    max(1e-12, sum(float(dw.square().sum().item()) for dw in prev_dw_scalars))
                    * max(1e-12, sum(float(p.grad.float().square().sum().item()) for p in fast_scalars if p.grad is not None))
                )
                bounce_cos_scalars.append(num_s / den_s)

            snap_stiefel = [p.data.float().clone() for p in stiefel_params]
            snap_unitary = [p.data.float().clone() for p in unitary_params]
            snap_vocab = wdet_param.data.float().clone()
            snap_ngram = [p.data.float().clone() for p in ngram_params]
            snap_scalars = [p.data.float().clone() for p in fast_scalars]

            active_ids = torch.unique(torch.cat([idx_in.reshape(-1), targets.reshape(-1)]))
            idx_l = idx_in.to(torch.long)
            idx_p1 = torch.cat([idx_l[:, :1], idx_l[:, :-1]], dim=1) if idx_l.size(1) > 1 else idx_l
            idx_p2 = torch.cat([idx_l[:, :2], idx_l[:, :-2]], dim=1) if idx_l.size(1) > 2 else idx_p1
            h_tok = (idx_l * 1000003 + idx_p1 * 65537 + idx_p2 * 257) & (DEFAULT_TOK_NGRAM_BUCKETS - 1)
            h_uni = (idx_l * 2654435761) & (DEFAULT_TOK_NGRAM_BUCKETS - 1)
            active_ngram_ids = torch.unique(torch.cat([h_tok.reshape(-1), h_uni.reshape(-1)]))

            duty_vocab.append(float(active_ids.numel()) / float(wte_param.size(0)))
            duty_ngram.append(float(active_ngram_ids.numel()) / float(DEFAULT_TOK_NGRAM_BUCKETS))

            _step_subsystem_scaled_optimizers(model, optimizers, idx_in, targets, mults_1x, quench_scale=1.0)

            prev_dw_trunk = [p.data.float() - s for p, s in zip(stiefel_params, snap_stiefel)]
            prev_dw_vocab = wdet_param.data.float() - snap_vocab
            prev_dw_scalars = [p.data.float() - s for p, s in zip(fast_scalars, snap_scalars)]

            if step > 5:
                rel_t = sum(
                    float(dw.norm().item() / s.norm().clamp_min(1e-8).item())
                    for dw, s in zip(prev_dw_trunk, snap_stiefel)
                ) / float(len(snap_stiefel))
                rel_u = sum(
                    float((p.data.float() - s).norm().item() / s.norm().clamp_min(1e-8).item())
                    for p, s in zip(unitary_params, snap_unitary)
                ) / float(len(snap_unitary))
                rel_v_act = float(
                    prev_dw_vocab.index_select(0, active_ids).norm().item()
                    / snap_vocab.index_select(0, active_ids).norm().clamp_min(1e-8).item()
                )
                rel_n_act = sum(
                    float(
                        (p.data.float() - s).index_select(0, active_ngram_ids).norm().item()
                        / p.data.float().index_select(0, active_ngram_ids).norm().clamp_min(1e-4).item()
                    )
                    for p, s in zip(ngram_params, snap_ngram)
                ) / float(len(snap_ngram))
                rel_s = sum(
                    float((p.data.float() - s).abs().mean().item() / s.abs().mean().clamp_min(1e-8).item())
                    for p, s in zip(fast_scalars, snap_scalars)
                ) / float(len(snap_scalars))

                rel_step_trunk.append(rel_t)
                rel_step_unitary.append(rel_u)
                rel_step_vocab_active.append(rel_v_act)
                rel_step_ngram_active.append(rel_n_act)
                rel_step_scalars.append(rel_s)

    del model, optimizers
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mean_duty_v = sum(duty_vocab) / len(duty_vocab)
    mean_duty_n = sum(duty_ngram) / len(duty_ngram)
    mean_rel_t = sum(rel_step_trunk) / len(rel_step_trunk)
    mean_rel_u = sum(rel_step_unitary) / len(rel_step_unitary)
    mean_rel_v_act = sum(rel_step_vocab_active) / len(rel_step_vocab_active)
    mean_rel_n_act = sum(rel_step_ngram_active) / len(rel_step_ngram_active)
    mean_rel_s = sum(rel_step_scalars) / len(rel_step_scalars)

    # Expected cumulative displacement per step = duty_cycle * active_rel_step
    eff_rate_trunk = 1.0 * mean_rel_t
    eff_rate_unitary_compounded = 1.0 * mean_rel_u * 2.0 * math.sqrt(4.0)  # quadratic U rho U^dagger across L=4 layers
    eff_rate_vocab = mean_duty_v * mean_rel_v_act
    eff_rate_ngram = mean_duty_n * mean_rel_n_act

    summary = {
        "steps": steps,
        "vocab_active_duty_cycle_pct": round(100.0 * mean_duty_v, 2),
        "ngram_active_duty_cycle_pct": round(100.0 * mean_duty_n, 2),
        "trunk_duty_cycle_pct": 100.0,
        "scalar_duty_cycle_pct": 100.0,
        "rel_step_per_hit": {
            "trunk_stiefel_rad": round(mean_rel_t, 5),
            "trunk_unitary_rad": round(mean_rel_u, 5),
            "trunk_unitary_4layer_quadratic_rad": round(eff_rate_unitary_compounded, 5),
            "vocab_wte512_active_rad": round(mean_rel_v_act, 5),
            "ngram_active_rad": round(mean_rel_n_act, 5),
            "global_scalars_rel": round(mean_rel_s, 5),
        },
        "expected_displacement_per_step_at_1x": {
            "trunk_stiefel": round(eff_rate_trunk, 5),
            "trunk_unitary_compounded": round(eff_rate_unitary_compounded, 5),
            "vocab_wte512_expected": round(eff_rate_vocab, 5),
            "ngram_expected": round(eff_rate_ngram, 5),
            "trunk_to_vocab_speed_ratio_at_1x": round(eff_rate_trunk / max(1e-8, eff_rate_vocab), 2),
        },
        "lookahead_step_alignment_cosine": {
            "trunk_stiefel": round(sum(bounce_cos_trunk) / max(1, len(bounce_cos_trunk)), 5),
            "vocab_wte512": round(sum(bounce_cos_vocab) / max(1, len(bounce_cos_vocab)), 5),
            "global_scalars": round(sum(bounce_cos_scalars) / max(1, len(bounce_cos_scalars)), 5),
        },
    }
    print("\n=== Phase 1: In-Advance Subsystem Curvature & Duty-Cycle Diagnostic (50 Steps @ 1.0x) ===", flush=True)
    print(
        f"  Duty Cycles       : trunk=100.0% | scalars=100.0% | ngram={summary['ngram_active_duty_cycle_pct']:.2f}% | "
        f"wte_512={summary['vocab_active_duty_cycle_pct']:.2f}% (~{int(mean_duty_v * 50304)} / 50304 tokens/batch)",
        flush=True,
    )
    print(
        f"  Expected Rate @1x : trunk_stiefel={eff_rate_trunk:.5f} rad/step | "
        f"trunk_U16_compounded={eff_rate_unitary_compounded:.5f} rad/step | "
        f"wte_512_expected={eff_rate_vocab:.5f} rad/step ({summary['expected_displacement_per_step_at_1x']['trunk_to_vocab_speed_ratio_at_1x']}x slower!)",
        flush=True,
    )
    print(
        f"  Lookahead Cosine  : trunk={summary['lookahead_step_alignment_cosine']['trunk_stiefel']:+.5f} | "
        f"wte_512={summary['lookahead_step_alignment_cosine']['vocab_wte512']:+.5f} | "
        f"scalars={summary['lookahead_step_alignment_cosine']['global_scalars']:+.5f}",
        flush=True,
    )
    return summary


def estimate_lower_lr_recommendation_from_probe(probe_result: dict[str, Any]) -> dict[str, Any]:
    """Compute a quantitative lower-LR re-run diagnostic from a single Hot + Shadow-EMA + Quench probe (`<= 8` args)."""
    pre_q = probe_result["pre_quench_eval"]
    post_q = probe_result["post_quench_eval"]
    early_pt = probe_result["curve"][1]  # Step 120

    thermal_gap_pre_quench = float(pre_q["val_ce_hot"] - pre_q["val_ce_ema"])
    quench_drop_hot = float(pre_q["val_ce_hot"] - post_q["val_ce_hot"])
    quench_drop_ema = float(pre_q["val_ce_ema"] - post_q["val_ce_ema"])

    plateau_span_steps = max(1, int(pre_q["step"]) - int(early_pt["step"]))
    quench_span_steps = max(1, int(post_q["step"]) - int(pre_q["step"]))

    plateau_rate_per_100 = 100.0 * float(early_pt["val_ce_hot"] - pre_q["val_ce_hot"]) / float(plateau_span_steps)
    quench_rate_per_100 = 100.0 * quench_drop_hot / float(quench_span_steps)
    quench_acceleration_ratio = quench_rate_per_100 / max(1e-4, plateau_rate_per_100)

    # If thermal_gap_pre_quench > 0.12 nats OR quench_acceleration_ratio > 2.0, the plateau LR was above the basin sweet spot.
    # Under quadratic steady-state noise Delta L_thermal ~ c * eta^2, optimal multiplier scales as 1 / sqrt(1 + quench_acceleration_ratio)
    trunk_m = float(probe_result["trunk_mult"])
    est_optimal_trunk_mult = trunk_m / math.sqrt(max(1.0, quench_acceleration_ratio))
    should_rerun_lower_lr = (thermal_gap_pre_quench > 0.10) or (quench_acceleration_ratio > 1.8)

    return {
        "source_run_id": probe_result["run_id"],
        "tested_trunk_mult": trunk_m,
        "tested_vocab_mult": probe_result["vocab_mult"],
        "pre_quench_hot_val_ce": round(float(pre_q["val_ce_hot"]), 4),
        "pre_quench_ema_val_ce": round(float(pre_q["val_ce_ema"]), 4),
        "thermal_gap_pre_quench_nats": round(thermal_gap_pre_quench, 4),
        "post_quench_hot_val_ce": round(float(post_q["val_ce_hot"]), 4),
        "post_quench_ema_val_ce": round(float(post_q["val_ce_ema"]), 4),
        "quench_drop_hot_nats": round(quench_drop_hot, 4),
        "quench_drop_ema_nats": round(quench_drop_ema, 4),
        "plateau_drop_rate_nats_per_100steps": round(plateau_rate_per_100, 4),
        "quench_drop_rate_nats_per_100steps": round(quench_rate_per_100, 4),
        "quench_to_plateau_acceleration_ratio": round(quench_acceleration_ratio, 2),
        "should_rerun_with_lower_trunk_lr": bool(should_rerun_lower_lr),
        "estimated_optimal_trunk_mult": round(est_optimal_trunk_mult, 2),
    }


def run_single_shadow_ema_and_quench_probe(
    stream: LongSeqFineWebStream,
    mults: SubsystemLRMultipliers,
    eval_interval: int = DEFAULT_PROBE_EVAL_INTERVAL,
) -> dict[str, Any]:
    """Run one Constant-LR Plateau (`plateau_steps`) + Geodesic Quench (`quench_steps`) with live Shadow-EMA (`<= 8` args)."""
    device = stream.device
    tokens_per_step = stream.batch_size * stream.seq_len
    total_steps = mults.plateau_steps + mults.quench_steps

    spec = ExplicitDensityTomographicSpec(
        variant_mode=ExplicitDensityVariantMode.T2_DENSITY_TOMOGRAPHY_PLUS_NOCLICK_AND_VACUUM,
        enable_density_highway=True, enable_tomographic_readout=True,
        enable_noclick_and_vacuum=True, enable_two_register_bilinear=False, seed=DEFAULT_PROBE_SEED,
    )
    torch.manual_seed(DEFAULT_PROBE_SEED)
    hot_model = ExplicitDensityTomographicGPT(spec).to(device=device)
    initialize_explicit_density_tomographic_model_(hot_model)
    ema_model = copy.deepcopy(hot_model).to(device=device)
    ema_model.eval()
    optimizers = build_explicit_density_tomographic_optimizers(hot_model)

    step0_ev = evaluate_explicit_density_tomographic_model(hot_model, stream)
    curve: list[dict[str, Any]] = [{
        "step": 0, "tokens_M": 0.0, "phase": "init", "quench_scale": 1.0,
        "val_ce_hot": step0_ev["val_ce"], "val_ce_ema": step0_ev["val_ce"],
        "thermal_gap_nats": 0.0, "top1_hot_pct": round(100.0 * step0_ev["top1_acc"], 2),
        "top1_ema_pct": round(100.0 * step0_ev["top1_acc"], 2),
        "rho_rank_hot": step0_ev["mean_rho_eff_rank"], "p_nc_hot": step0_ev["mean_noclick_prob"],
    }]

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    hot_model.train()

    for step in range(1, total_steps + 1):
        idx_in, targets = stream.sample_train_batch(step)
        for opt in optimizers:
            if opt is not None:
                opt.zero_grad(set_to_none=True)

        loss = hot_model.forward_loss(idx_in, targets)
        loss_val = float(loss.detach().item())
        loss.backward()
        del loss

        if step <= mults.plateau_steps:
            phase = "plateau"
            quench_scale = 1.0
        else:
            phase = "quench"
            q_prog = float(step - mults.plateau_steps) / float(max(1, mults.quench_steps))
            # Cosine-exponential geodesic quench from 1.0 down to DEFAULT_QUENCH_FINAL_FRACTION (0.10)
            quench_scale = DEFAULT_QUENCH_FINAL_FRACTION + (1.0 - DEFAULT_QUENCH_FINAL_FRACTION) * 0.5 * (
                1.0 + math.cos(math.pi * q_prog)
            )

        _step_subsystem_scaled_optimizers(hot_model, optimizers, idx_in, targets, mults, quench_scale=quench_scale)
        update_shadow_ema_buffers_(ema_model, hot_model, ema_decay=mults.ema_decay)

        should_eval = (
            (step % eval_interval == 0)
            or (step == mults.plateau_steps)
            or (step == mults.plateau_steps + mults.quench_steps // 2)
            or (step == total_steps)
        )
        if should_eval:
            ev_hot = evaluate_explicit_density_tomographic_model(hot_model, stream)
            ema_snap = [p.data.clone() for p in ema_model.parameters()]
            retract_shadow_ema_manifolds_(ema_model)
            ev_ema = evaluate_explicit_density_tomographic_model(ema_model, stream)
            for p, s in zip(ema_model.parameters(), ema_snap):
                p.data.copy_(s)
            del ema_snap

            thermal_gap = round(float(ev_hot["val_ce"] - ev_ema["val_ce"]), 4)
            rec = {
                "step": step,
                "tokens_M": round((step * tokens_per_step) / 1e6, 3),
                "phase": phase,
                "quench_scale": round(quench_scale, 4),
                "train_ce": round(loss_val, 4),
                "val_ce_hot": ev_hot["val_ce"],
                "val_ce_ema": ev_ema["val_ce"],
                "thermal_gap_nats": thermal_gap,
                "top1_hot_pct": round(100.0 * ev_hot["top1_acc"], 2),
                "top1_ema_pct": round(100.0 * ev_ema["top1_acc"], 2),
                "readout_temp": round(float(hot_model.readout_temp.item()), 3),
                "rho_rank_hot": ev_hot["mean_rho_eff_rank"],
                "rho_rank_ema": ev_ema["mean_rho_eff_rank"],
                "p_nc_hot": ev_hot["mean_noclick_prob"],
                "p_nc_ema": ev_ema["mean_noclick_prob"],
            }
            curve.append(rec)
            print(
                f"[{mults.run_id} ({phase:7s} q={quench_scale:.2f})] Step {step:3d} ({rec['tokens_M']:4.2f}M) | "
                f"Hot CE: {rec['val_ce_hot']:.4f} | EMA CE: {rec['val_ce_ema']:.4f} (gap {thermal_gap:+.4f}) | "
                f"Top1(hot/ema): {rec['top1_hot_pct']:.2f}%/{rec['top1_ema_pct']:.2f}% | "
                f"temp={rec['readout_temp']:5.2f} | RhoRank={rec['rho_rank_hot']:.2f}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0

    pre_quench_recs = [r for r in curve if r["step"] == mults.plateau_steps]
    pre_quench_eval = pre_quench_recs[-1] if pre_quench_recs else curve[-2]
    post_quench_eval = curve[-1]

    del hot_model, ema_model, optimizers
    if device.type == "cuda":
        torch.cuda.empty_cache()

    res = {
        "run_id": mults.run_id,
        "trunk_mult": mults.trunk_mult,
        "vocab_mult": mults.vocab_mult,
        "ngram_mult": mults.ngram_mult,
        "scalar_mult": mults.scalar_mult,
        "plateau_steps": mults.plateau_steps,
        "quench_steps": mults.quench_steps,
        "elapsed_sec": round(elapsed_s, 2),
        "pre_quench_eval": pre_quench_eval,
        "post_quench_eval": post_quench_eval,
        "best_val_ce_hot": round(min(r["val_ce_hot"] for r in curve[1:]), 4),
        "best_val_ce_ema": round(min(r["val_ce_ema"] for r in curve[1:]), 4),
        "curve": curve,
    }
    res["single_probe_Diagnostic"] = estimate_lower_lr_recommendation_from_probe(res)
    return res


def main() -> None:
    """CLI entrypoint for the Hot + Shadow-EMA + Branch-and-Quench LR & Subsystem Curvature Probe."""
    parser = argparse.ArgumentParser(description="Hot + Shadow-EMA + Branch-and-Quench LR Probe")
    parser.add_argument("--plateau-steps", type=int, default=DEFAULT_PLATEAU_STEPS)
    parser.add_argument("--quench-steps", type=int, default=DEFAULT_QUENCH_STEPS)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_PROBE_EVAL_INTERVAL)
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--json-out", type=str, default=".cache/plan083_shadow_ema_and_quench_probe.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = LongSeqFineWebStream(seq_len=512, batch_size=16, val_tokens=65_536, device=device)
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "wave_id": "W63_SHADOW_QUENCH",
        "plan_id": "EXPERIMENT_PLAN_083",
        "plateau_steps": args.plateau_steps,
        "quench_steps": args.quench_steps,
    }

    if not args.skip_phase1:
        payload["phase1_subsystem_curvature_diagnostic"] = run_subsystem_curvature_and_bounce_estimator(
            stream, steps=DEFAULT_CURVATURE_PROBE_STEPS
        )
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Phase 2: Hot + Shadow-EMA + Branch-and-Quench Probes (Plateau + Quench) ===", flush=True)
    probe_configs = [
        SubsystemLRMultipliers(
            run_id="PROBE_1_TRUNK4X_VOC1X_SCL1X",
            trunk_mult=4.0, vocab_mult=1.0, ngram_mult=1.0, scalar_mult=1.0,
            plateau_steps=args.plateau_steps, quench_steps=args.quench_steps,
        ),
        SubsystemLRMultipliers(
            run_id="PROBE_2_UNIFORM_1X",
            trunk_mult=1.0, vocab_mult=1.0, ngram_mult=1.0, scalar_mult=1.0,
            plateau_steps=args.plateau_steps, quench_steps=args.quench_steps,
        ),
        SubsystemLRMultipliers(
            run_id="PROBE_3_UNIFORM_2X",
            trunk_mult=2.0, vocab_mult=2.0, ngram_mult=2.0, scalar_mult=2.0,
            plateau_steps=args.plateau_steps, quench_steps=args.quench_steps,
        ),
        SubsystemLRMultipliers(
            run_id="PROBE_4_DUTY_MATCHED_T1p5_V2p5_S0p75",
            trunk_mult=1.5, vocab_mult=2.5, ngram_mult=1.5, scalar_mult=0.75,
            plateau_steps=args.plateau_steps, quench_steps=args.quench_steps,
        ),
        SubsystemLRMultipliers(
            run_id="PROBE_5_CALM_TRUNK_T1p0_V2p5_S0p75",
            trunk_mult=1.0, vocab_mult=2.5, ngram_mult=1.25, scalar_mult=0.75,
            plateau_steps=args.plateau_steps, quench_steps=args.quench_steps,
        ),
    ]

    probes: list[dict[str, Any]] = []
    for cfg in probe_configs:
        res = run_single_shadow_ema_and_quench_probe(stream, cfg, eval_interval=args.eval_interval)
        diag = res["single_probe_Diagnostic"]
        print(
            f"  -> [Single-Probe Estimator ({cfg.run_id})] "
            f"PreQuench Thermal Gap={diag['thermal_gap_pre_quench_nats']:+.4f} nats | "
            f"Quench Drop={diag['quench_drop_hot_nats']:+.4f} nats | "
            f"Quench Accel={diag['quench_to_plateau_acceleration_ratio']:.2f}x | "
            f"Rerun Lower Trunk LR? {diag['should_rerun_with_lower_trunk_lr']} "
            f"(Est. Optimal Trunk Mult={diag['estimated_optimal_trunk_mult']:.2f}x)\n",
            flush=True,
        )
        probes.append(res)
        payload["phase2_shadow_ema_and_quench_probes"] = probes
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nSaved Shadow-EMA & Quench Probe results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
