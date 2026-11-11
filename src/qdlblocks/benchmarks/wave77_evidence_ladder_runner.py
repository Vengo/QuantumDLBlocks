"""Wave 77 5-Rung Evidence Ladder, Mechanistic Probes, & Multi-Shard Confirmation Runner (`EXPERIMENT_PLAN_097.md`).

Provenance & Mathematical Rationale
-----------------------------------
Orchestrates **Step 2** (Controlled `400`-Step 5-Rung Evidence Ladder + Carbon 3-Way State-Routing
Ablation + `50,257`-Vocab Induction/MQAR Probes) and **Step 3** (`100M`-Token Multi-Shard FineWeb
Confirmation Spin with Warmup-Stable-Decay & Official `10,485,760`-Token Shard-0 Validation) for
`Wave 77` (`EXPERIMENT_PLAN_097.md`), obeying Rules 1–8 of `docs/SCRIPT_BEST_PRACTICES.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .architecture_mechanistic_probes import (
    DualHorizonShadowEMATracker,
    evaluate_quick_ce_and_top1,
    fit_early_powerlaw_asymptote,
    generate_fuzzy_induction_batch,
    generate_mqar_batch,
)
from .architecture_xray_diagnostics import (
    FINEWEB_HEADER_INT32_COUNT,
    REPO_ROOT,
    SYNTHETIC_INDUCTION_SEQ_LENS,
    SYNTHETIC_MQAR_PAIR_COUNTS,
    ShortTrainConfig,
    SyntheticProbeConfig,
    TeacherReferenceProfile,
    compute_stratified_nat_gap_xray,
    load_fineweb_token_slice,
    resolve_fineweb_shard_path,
)
from .experiment_logger import ExperimentIntentSpec, StandardizedExperimentLogger
from .fused_bf16_four_compartment_baton_lab import (
    FINEWEB_TRAIN_SHARDS_300M,
    MultiShardFineWebStreamer,
)
from .wave77_baton_ladder_models import (
    OFFICIAL_SHARD0_VAL_TOKENS,
    CalibratedCopySinkBatonLM,
    QuantumBatonMixerLM,
    Wave77DiagnosticAdapter,
    Wave77RungKind,
    build_wave77_rung_adapter,
)

# ==============================================================================
# Top-of-File Named Constants (Synchronized with EXPERIMENT_PLAN_097.md Section 3)
# ==============================================================================
DEFAULT_CONFIRM_TARGET_TOKENS: int = 100_000_000
DEFAULT_CONFIRM_BATCH_SIZE: int = 32
DEFAULT_CONFIRM_SEQ_LEN: int = 512
DEFAULT_CONFIRM_EVAL_INTERVAL: int = 500
DEFAULT_W77_CKPT_DIR: Path = REPO_ROOT / ".cache" / "checkpoints" / "wave77"


@dataclass(frozen=True)
class MultiShardConfirmConfig:
    """Configuration for the Phase 4 multi-shard FineWeb confirmation spin (`<= 8` fields, Rule 2)."""

    target_tokens: int = DEFAULT_CONFIRM_TARGET_TOKENS
    batch_size: int = DEFAULT_CONFIRM_BATCH_SIZE
    seq_len: int = DEFAULT_CONFIRM_SEQ_LEN
    eval_interval: int = DEFAULT_CONFIRM_EVAL_INTERVAL
    official_val_tokens: int = OFFICIAL_SHARD0_VAL_TOKENS
    wave_id: str = "W77"
    plan_id: str = "EXPERIMENT_PLAN_097"
    seed: int = 1337


@torch.no_grad()
def _extract_rung_post_sprint_telemetry(
    adapter: Wave77DiagnosticAdapter,
    x_sample: Tensor,
) -> dict[str, Any]:
    """Extract Carbon `mean_p_and`/`packet_purity` or `<BOS>` row-sum telemetry (`<= 8` args)."""
    xb = x_sample[:2].to(adapter.device)
    if isinstance(adapter.model, QuantumBatonMixerLM):
        was_tr = adapter.model.training
        adapter.model.eval()
        amp_on = xb.is_cuda
        with torch.amp.autocast(
            device_type=xb.device.type, dtype=torch.bfloat16, enabled=amp_on
        ):
            _, _, copy_conf, _, _ = adapter.model.encode_trunk_features(xb)
        if was_tr:
            adapter.model.train()
        telem = adapter.model.collect_carbon_routing_telemetry()
        telem["mean_copy_confidence"] = round(float(copy_conf.float().mean().item()), 5)
        return telem

    if isinstance(adapter.model, CalibratedCopySinkBatonLM):
        out = adapter.model.forward_with_diagnostics(
            xb, targets=None, ingest_online=False, return_logits=False
        )
        probs = out["seq_attn_probs"].float()
        row_sums = probs.sum(dim=-1)
        return {
            "has_schur_gate": False,
            "mean_copy_confidence": round(float(out["mean_copy_conf"]), 5),
            "min_attn_row_sum": round(float(row_sums.amin().item()), 6),
            "max_attn_row_sum": round(float(row_sums.amax().item()), 6),
        }
    return {}


def run_wave77_controlled_sprint(
    rung: Wave77RungKind,
    train_cfg: ShortTrainConfig,
    x_val: Tensor,
    y_val: Tensor,
    device: torch.device,
) -> tuple[Wave77DiagnosticAdapter, dict[str, Any]]:
    """Run a controlled `400`-step (`3.28M`-token) FineWeb sprint on `rung` (`<= 8` args)."""
    adapter = build_wave77_rung_adapter(rung, device=device, seed=train_cfg.seed)
    p_info = adapter.parameter_breakdown()
    tokens_per_step = train_cfg.batch_size * train_cfg.seq_len
    total_tokens = train_cfg.steps * tokens_per_step

    train_tokens = load_fineweb_token_slice(
        "fineweb_train_000001.bin", num_tokens=total_tokens + train_cfg.seq_len + 8
    )

    run_key = f"{train_cfg.wave_id.lower()}_{rung.value}_{train_cfg.steps}s"
    intent = ExperimentIntentSpec(
        run_key=run_key,
        wave_id=train_cfg.wave_id,
        plan_id=train_cfg.plan_id,
        task_domain="fineweb_speedrun",
        hypothesis_intent=(
            f"Wave 77 controlled {train_cfg.steps}-step ({total_tokens / 1e6:.2f}M token) evidence-ladder "
            f"sprint for {adapter.label} measuring Hot/EMA CE, asymptote L_inf, and stratified nat-gap."
        ),
        architecture_summary=(
            f"{adapter.label} ({p_info['total_params'] / 1e6:.2f}M total, "
            f"{p_info['trunk_params'] / 1e6:.2f}M trunk)"
        ),
        hardware_budget=torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        arch_family="wave77_quantum_baton_ladder",
        batch_size=train_cfg.batch_size,
        seq_len=train_cfg.seq_len,
        total_steps=train_cfg.steps,
        target_tokens_M=round(total_tokens / 1e6, 4),
        params_M=round(float(p_info["total_params"]) / 1e6, 4),
        embed_dim=adapter.embed_dim,
        num_layers=adapter.num_layers,
        schedule="cosine_warmup_sprint",
    )
    logger = StandardizedExperimentLogger(intent)
    ema_tracker = DualHorizonShadowEMATracker(adapter.model)

    step0_ce, step0_top1 = evaluate_quick_ce_and_top1(adapter, x_val, y_val)
    checkpoints: list[dict[str, float]] = [
        {
            "step": 0.0,
            "tokens_M": 0.0,
            "hot_val_ce": round(step0_ce, 5),
            "fast_ema_val_ce": round(step0_ce, 5),
            "slow_ema_val_ce": round(step0_ce, 5),
            "thermal_gap_fast": 0.0,
            "thermal_gap_slow": 0.0,
            "hot_top1_acc": round(step0_top1, 5),
        }
    ]
    logger.log_eval(
        step=0,
        metrics={
            "tokens_M": 0.0,
            "compute_tflops": 0.0,
            "tau_progress": 0.0,
            "val_ce": step0_ce,
            "top1_acc": step0_top1 * 100.0,
        },
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    train_time_accum = 0.0

    for step in range(1, train_cfg.steps + 1):
        offset = (step - 1) * tokens_per_step
        span_slice = train_tokens[offset : offset + tokens_per_step + 1]
        xb = span_slice[:-1].view(train_cfg.batch_size, train_cfg.seq_len).to(device)
        yb = span_slice[1:].view(train_cfg.batch_size, train_cfg.seq_len).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_step0 = time.perf_counter()
        tr_loss = adapter.train_step(xb, yb, step=step, total_steps=train_cfg.steps)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time_accum += time.perf_counter() - t_step0
        ema_tracker.update(adapter.model)

        if step == 50 or step % train_cfg.eval_interval == 0 or step == train_cfg.steps:
            tok_m = (step * tokens_per_step) / 1e6
            cur_tok_s = float(step * tokens_per_step) / max(1e-6, train_time_accum)
            ema_stats = ema_tracker.evaluate_hot_fast_slow(adapter, x_val, y_val)
            ema_stats["step"] = float(step)
            ema_stats["tokens_M"] = round(tok_m, 4)
            ema_stats["train_loss"] = round(tr_loss, 5)
            checkpoints.append(ema_stats)

            approx_tflops = (
                6.0 * float(p_info["total_params"]) * float(step * tokens_per_step)
            ) / 1e12
            logger.log_step(step=step, train_ce=tr_loss, tok_per_s=cur_tok_s, lr_mult=1.0)
            logger.log_eval(
                step=step,
                metrics={
                    "tokens_M": tok_m,
                    "compute_tflops": approx_tflops,
                    "tau_progress": float(step) / float(max(1, train_cfg.steps)),
                    "val_ce": ema_stats["hot_val_ce"],
                    "top1_acc": ema_stats["hot_top1_acc"] * 100.0,
                    "tok_per_s": cur_tok_s,
                    "fast_ema_val_ce": ema_stats["fast_ema_val_ce"],
                    "slow_ema_val_ce": ema_stats["slow_ema_val_ce"],
                    "thermal_gap_slow": ema_stats["thermal_gap_slow"],
                },
            )

    total_wall_sec = time.perf_counter() - t_start
    tok_per_s = float(total_tokens) / max(1e-6, train_time_accum)

    d_pts = [c["tokens_M"] for c in checkpoints if c["tokens_M"] > 0.0]
    ce_pts = [c["slow_ema_val_ce"] for c in checkpoints if c["tokens_M"] > 0.0]
    asymptote = fit_early_powerlaw_asymptote(d_pts, ce_pts)
    final_ckpt = checkpoints[-1]
    best_ce = min(
        min(c["hot_val_ce"] for c in checkpoints),
        min(c["slow_ema_val_ce"] for c in checkpoints),
    )
    routing_telem = _extract_rung_post_sprint_telemetry(adapter, x_val)

    summary = {
        "run_key": run_key,
        "arch": rung.value,
        "label": adapter.label,
        "total_params": p_info["total_params"],
        "trunk_params": p_info["trunk_params"],
        "vocab_params": p_info["vocab_params"],
        "steps": train_cfg.steps,
        "tokens_M": round(total_tokens / 1e6, 4),
        "tok_per_s": round(tok_per_s, 1),
        "train_sec": round(train_time_accum, 2),
        "wall_sec": round(total_wall_sec, 2),
        "step0_val_ce": round(step0_ce, 5),
        "final_hot_val_ce": final_ckpt["hot_val_ce"],
        "final_fast_ema_val_ce": final_ckpt["fast_ema_val_ce"],
        "final_slow_ema_val_ce": final_ckpt["slow_ema_val_ce"],
        "best_any_val_ce": round(best_ce, 5),
        "final_thermal_gap_fast": final_ckpt["thermal_gap_fast"],
        "final_thermal_gap_slow": final_ckpt["thermal_gap_slow"],
        "asymptote_fit": asymptote,
        "routing_telemetry": routing_telem,
        "checkpoints": checkpoints,
    }
    logger.finalize(extra_summary=summary)
    return adapter, summary


def run_wave77_induction_and_mqar_probes(
    rungs: list[Wave77RungKind],
    cfg: SyntheticProbeConfig,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run identical `Wave 76` `50,257`-vocab Fuzzy Induction and `MQAR` probes on `rungs`."""
    ind_results: list[dict[str, Any]] = []
    mqar_results: list[dict[str, Any]] = []

    for rung in rungs:
        for seq_t in SYNTHETIC_INDUCTION_SEQ_LENS:
            adapter = build_wave77_rung_adapter(rung, device=device, seed=cfg.seed)
            gen = torch.Generator().manual_seed(cfg.seed + int(seq_t))
            t0 = time.perf_counter()
            for step in range(1, cfg.induction_steps + 1):
                xb, yb, _ = generate_fuzzy_induction_batch(
                    cfg.batch_size, int(seq_t), cfg.vocab_size, gen, device
                )
                adapter.train_step(xb, yb, step=step, total_steps=cfg.induction_steps)

            q_nll = 0.0
            q_top1 = 0
            q_trap = 0
            q_cnt = 0
            eff_v = max(256, int(cfg.vocab_size) - 1)
            for _ in range(cfg.eval_batches):
                xb, yb, q_mask = generate_fuzzy_induction_batch(
                    cfg.batch_size, int(seq_t), cfg.vocab_size, gen, device
                )
                false_t = (yb[q_mask] + 17) % eff_v
                with torch.no_grad():
                    log_p, _ = adapter.predict_log_probs_and_span(xb)
                    nll = -log_p.gather(dim=-1, index=yb.unsqueeze(-1)).squeeze(-1)
                    preds = log_p.argmax(dim=-1)
                    q_nll += float(nll[q_mask].sum().item())
                    q_top1 += int((preds[q_mask] == yb[q_mask]).sum().item())
                    q_trap += int((preds[q_mask] == false_t).sum().item())
                    q_cnt += int(q_mask.sum().item())
            denom = float(max(1, q_cnt))
            ind_results.append(
                {
                    "arch": rung.value,
                    "label": adapter.label,
                    "seq_len": int(seq_t),
                    "induction_query_ce": round(q_nll / denom, 5),
                    "induction_query_top1_acc": round(float(q_top1) / denom, 5),
                    "pre_bos_false_trap_rate": round(float(q_trap) / denom, 5),
                    "elapsed_sec": round(time.perf_counter() - t0, 2),
                }
            )
            del adapter
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for k_pairs in SYNTHETIC_MQAR_PAIR_COUNTS:
            adapter = build_wave77_rung_adapter(rung, device=device, seed=cfg.seed + int(k_pairs))
            gen = torch.Generator().manual_seed(cfg.seed + 100 * int(k_pairs))
            t0 = time.perf_counter()
            for step in range(1, cfg.mqar_steps + 1):
                xb, yb, _ = generate_mqar_batch(
                    cfg.batch_size, 128, int(k_pairs), cfg.vocab_size, gen, device
                )
                adapter.train_step(xb, yb, step=step, total_steps=cfg.mqar_steps)

            nll_s = 0.0
            ok_s = 0
            tot_q = 0
            for _ in range(cfg.eval_batches):
                xb, yb, q_mask = generate_mqar_batch(
                    cfg.batch_size, 128, int(k_pairs), cfg.vocab_size, gen, device
                )
                with torch.no_grad():
                    log_p, _ = adapter.predict_log_probs_and_span(xb)
                    nll = -log_p.gather(dim=-1, index=yb.unsqueeze(-1)).squeeze(-1)
                    preds = log_p.argmax(dim=-1)
                    nll_s += float(nll[q_mask].sum().item())
                    ok_s += int((preds[q_mask] == yb[q_mask]).sum().item())
                    tot_q += int(q_mask.sum().item())
            denom = float(max(1, tot_q))
            mqar_results.append(
                {
                    "arch": rung.value,
                    "label": adapter.label,
                    "num_pairs": int(k_pairs),
                    "mqar_ce": round(nll_s / denom, 5),
                    "mqar_top1_acc": round(float(ok_s) / denom, 5),
                    "elapsed_sec": round(time.perf_counter() - t0, 2),
                }
            )
            del adapter
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return ind_results, mqar_results


def run_wave77_multishard_confirmation_spin(
    rung: Wave77RungKind,
    spin_cfg: MultiShardConfirmConfig,
    x_xray: Tensor,
    y_xray: Tensor,
    decile_map: Tensor,
    ref_550m: TeacherReferenceProfile,
    device: torch.device,
) -> dict[str, Any]:
    """Execute a multi-shard `100M`-token WSD confirmation spin + Official `10.49M` Shard-0 eval."""
    adapter = build_wave77_rung_adapter(
        rung, device=device, seed=spin_cfg.seed, use_wsd_schedule=True
    )
    p_info = adapter.parameter_breakdown()
    tokens_per_step = spin_cfg.batch_size * spin_cfg.seq_len
    total_steps = max(1, (spin_cfg.target_tokens + tokens_per_step - 1) // tokens_per_step)
    total_tokens = total_steps * tokens_per_step
    tok_m_tag = int(round(total_tokens / 1e6))

    streamer = MultiShardFineWebStreamer(FINEWEB_TRAIN_SHARDS_300M)
    run_key = f"{spin_cfg.wave_id.lower()}_{rung.value}_{tok_m_tag}m_confirm_wsd"

    intent = ExperimentIntentSpec(
        run_key=run_key,
        wave_id=spin_cfg.wave_id,
        plan_id=spin_cfg.plan_id,
        task_domain="fineweb_speedrun",
        hypothesis_intent=(
            f"Wave 77 {tok_m_tag}M-token multi-shard confirmation spin on {adapter.label} testing "
            f"whether MuP + StiefelMuon Quantum Baton Mixer beats Wave 74's 300M-token 5.2158 CE record."
        ),
        architecture_summary=(
            f"{adapter.label} ({p_info['total_params'] / 1e6:.2f}M total, "
            f"{p_info['trunk_params'] / 1e6:.2f}M trunk)"
        ),
        hardware_budget=torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        arch_family="wave77_quantum_baton_confirm",
        batch_size=spin_cfg.batch_size,
        seq_len=spin_cfg.seq_len,
        total_steps=total_steps,
        target_tokens_M=round(total_tokens / 1e6, 3),
        params_M=round(float(p_info["total_params"]) / 1e6, 4),
        embed_dim=adapter.embed_dim,
        num_layers=adapter.num_layers,
        schedule="wsd_linear_quench",
    )
    logger = StandardizedExperimentLogger(intent)
    ema_tracker = DualHorizonShadowEMATracker(adapter.model, shadow_device=device)

    step0_ce, step0_top1 = evaluate_quick_ce_and_top1(adapter, x_xray, y_xray)
    checkpoints: list[dict[str, float]] = [
        {
            "step": 0.0,
            "tokens_M": 0.0,
            "hot_val_ce": round(step0_ce, 5),
            "fast_ema_val_ce": round(step0_ce, 5),
            "slow_ema_val_ce": round(step0_ce, 5),
            "thermal_gap_slow": 0.0,
            "hot_top1_acc": round(step0_ce * 0.0 + step0_top1, 5),
        }
    ]
    logger.log_eval(
        step=0,
        metrics={
            "tokens_M": 0.0,
            "compute_tflops": 0.0,
            "tau_progress": 0.0,
            "val_ce": step0_ce,
            "top1_acc": step0_top1 * 100.0,
        },
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_wall0 = time.perf_counter()
    train_sec = 0.0

    for step in range(1, total_steps + 1):
        xb, yb, _ = streamer.sample_sequential_batch(
            step - 1, spin_cfg.batch_size, spin_cfg.seq_len, device
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts0 = time.perf_counter()
        tr_loss = adapter.train_step(xb, yb, step=step, total_steps=total_steps)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_sec += time.perf_counter() - ts0
        ema_tracker.update(adapter.model)

        if step == 100 or step % spin_cfg.eval_interval == 0 or step == total_steps:
            tok_m = (step * tokens_per_step) / 1e6
            cur_tok_s = float(step * tokens_per_step) / max(1e-6, train_sec)
            ema_stats = ema_tracker.evaluate_hot_fast_slow(adapter, x_xray, y_xray)
            ema_stats["step"] = float(step)
            ema_stats["tokens_M"] = round(tok_m, 3)
            ema_stats["train_loss"] = round(tr_loss, 5)
            checkpoints.append(ema_stats)

            tflops = (6.0 * float(p_info["total_params"]) * float(step * tokens_per_step)) / 1e12
            lr_m = adapter._compute_lr_multiplier(step, total_steps)
            logger.log_step(step=step, train_ce=tr_loss, tok_per_s=cur_tok_s, lr_mult=lr_m)
            logger.log_eval(
                step=step,
                metrics={
                    "tokens_M": tok_m,
                    "compute_tflops": tflops,
                    "tau_progress": float(step) / float(total_steps),
                    "val_ce": ema_stats["hot_val_ce"],
                    "top1_acc": ema_stats["hot_top1_acc"] * 100.0,
                    "tok_per_s": cur_tok_s,
                    "fast_ema_val_ce": ema_stats["fast_ema_val_ce"],
                    "slow_ema_val_ce": ema_stats["slow_ema_val_ce"],
                    "thermal_gap_slow": ema_stats["thermal_gap_slow"],
                },
            )
            print(
                f"  [Confirm {step:5d}/{total_steps}] {tok_m:6.2f}M tok | "
                f"lr_mult={lr_m:.3f} | {cur_tok_s:8,.1f} tok/s | "
                f"Hot CE={ema_stats['hot_val_ce']:.4f} | SlowEMA CE={ema_stats['slow_ema_val_ce']:.4f} "
                f"(Δth={ema_stats['thermal_gap_slow']:+.4f})",
                flush=True,
            )

    # Select cleaner of Hot vs Slow-EMA weights before Official 10,485,760-token Shard-0 evaluation!
    final_ckpt = checkpoints[-1]
    used_ema_weights = False
    if final_ckpt["slow_ema_val_ce"] < final_ckpt["hot_val_ce"]:
        with torch.no_grad():
            for name, p in adapter.model.named_parameters():
                if name in ema_tracker.slow_shadow:
                    p.copy_(ema_tracker.slow_shadow[name].to(device=p.device, dtype=p.dtype))
        used_ema_weights = True

    xray = compute_stratified_nat_gap_xray(
        adapter, x_xray, y_xray, decile_map=decile_map, ref_profile=ref_550m
    )

    shard0_path = resolve_fineweb_shard_path("fineweb_val_000000.bin")
    official_eval: dict[str, float] = {}
    if shard0_path is not None and spin_cfg.official_val_tokens > 0:
        official_eval = evaluate_wave77_official_shard0_validation(
            adapter,
            shard0_path=shard0_path,
            val_tokens_budget=spin_cfg.official_val_tokens,
            batch_size=spin_cfg.batch_size,
            seq_len=spin_cfg.seq_len,
        )
        logger.log_eval(
            step=total_steps,
            metrics={
                "tokens_M": round(total_tokens / 1e6, 3),
                "compute_tflops": (6.0 * float(p_info["total_params"]) * float(total_tokens)) / 1e12,
                "tau_progress": 1.0,
                "val_ce": official_eval["official_shard0_val_ce"],
                "top1_acc": official_eval["official_shard0_top1_pct"],
                "tok_per_s": round(float(total_tokens) / max(1e-6, train_sec), 1),
            },
        )

    DEFAULT_W77_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_out = DEFAULT_W77_CKPT_DIR / f"{run_key}.pt"
    torch.save(
        {
            "run_key": run_key,
            "rung": rung.value,
            "tokens_trained": total_tokens,
            "used_slow_ema_weights": used_ema_weights,
            "official_shard0_eval": official_eval,
            "model_state_dict": adapter.model.state_dict(),
        },
        ckpt_out,
    )

    d_pts = [c["tokens_M"] for c in checkpoints if c["tokens_M"] > 0.0]
    ce_pts = [c["slow_ema_val_ce"] for c in checkpoints if c["tokens_M"] > 0.0]
    asym = fit_early_powerlaw_asymptote(d_pts, ce_pts)

    summary = {
        "run_key": run_key,
        "arch": rung.value,
        "label": adapter.label,
        "checkpoint_path": str(ckpt_out),
        "used_slow_ema_for_final_eval": used_ema_weights,
        "total_params": p_info["total_params"],
        "trunk_params": p_info["trunk_params"],
        "steps": total_steps,
        "tokens_M": round(total_tokens / 1e6, 3),
        "tok_per_s": round(float(total_tokens) / max(1e-6, train_sec), 1),
        "train_sec": round(train_sec, 2),
        "wall_sec": round(time.perf_counter() - t_wall0, 2),
        "final_hot_16k_ce": final_ckpt["hot_val_ce"],
        "final_slow_ema_16k_ce": final_ckpt["slow_ema_val_ce"],
        "official_shard0_10m_eval": official_eval,
        "asymptote_fit": asym,
        "stratified_xray": xray,
        "checkpoints": checkpoints,
    }
    logger.finalize(extra_summary=summary)
    return summary


@torch.no_grad()
def evaluate_wave77_official_shard0_validation(
    adapter: Wave77DiagnosticAdapter,
    shard0_path: Path,
    val_tokens_budget: int = OFFICIAL_SHARD0_VAL_TOKENS,
    batch_size: int = 32,
    seq_len: int = 512,
) -> dict[str, float]:
    """Evaluate `adapter` on the official `10,485,760` (`10 * 2^20`) sequential Shard-0 tokens (`<= 8` args)."""
    was_train = adapter.model.training
    adapter.model.eval()
    t0 = time.perf_counter()

    mm = np.memmap(shard0_path, dtype=np.uint16, mode="r", offset=FINEWEB_HEADER_INT32_COUNT * 4)
    n_avail = min(len(mm) - 1, int(val_tokens_budget))
    n_seqs = max(1, n_avail // int(seq_len))
    total_eval_tok = n_seqs * int(seq_len)
    arr = np.asarray(mm[: total_eval_tok + 1], dtype=np.int64)
    flat_cpu = torch.from_numpy(arr)

    x_all = flat_cpu[:total_eval_tok].view(n_seqs, int(seq_len))
    y_all = flat_cpu[1 : total_eval_tok + 1].view(n_seqs, int(seq_len))

    total_nll = 0.0
    total_top1 = 0
    amp_on = adapter.device.type == "cuda"

    for b0 in range(0, n_seqs, batch_size):
        b1 = min(n_seqs, b0 + batch_size)
        xb = x_all[b0:b1].to(adapter.device, non_blocking=True)
        yb = y_all[b0:b1].to(adapter.device, non_blocking=True)

        if isinstance(adapter.model, QuantumBatonMixerLM):
            with torch.amp.autocast(
                device_type=adapter.device.type, dtype=torch.bfloat16, enabled=amp_on
            ):
                x_fin, copy_lo, _, _, _ = adapter.model.encode_trunk_features(xb)
                sub_b = max(1, 4096 // int(seq_len))
                uni_s = F.softplus(adapter.model.uni_prior_logit.float()).to(x_fin.dtype)
                uni_lp = (
                    torch.log(adapter.model.online_uni_token_probs.detach().clamp_min(1.0e-9))
                    .to(x_fin.dtype)
                    .view(1, 1, -1)
                )
                c_idx = xb.unsqueeze(1).expand(xb.size(0), seq_len, seq_len)
                cap = adapter.model.cfg.logit_softcap
                inv_c = 1.0 / cap
                for s0 in range(0, xb.size(0), sub_b):
                    s1 = min(xb.size(0), s0 + sub_b)
                    lg = F.linear(x_fin[s0:s1], adapter.model.lm_head.weight) + uni_s * uni_lp
                    lg = lg.scatter_add(
                        dim=-1, index=c_idx[s0:s1], src=copy_lo[s0:s1].to(lg.dtype)
                    )
                    capped = cap * torch.tanh(lg.float() * inv_c)
                    nll = F.cross_entropy(
                        capped.reshape(-1, capped.size(-1)), yb[s0:s1].reshape(-1), reduction="sum"
                    )
                    total_nll += float(nll.item())
                    total_top1 += int((capped.argmax(dim=-1) == yb[s0:s1]).sum().item())
        else:
            log_p, _ = adapter.predict_log_probs_and_span(xb)
            nll = -log_p.gather(dim=-1, index=yb.unsqueeze(-1)).squeeze(-1).sum()
            total_nll += float(nll.item())
            total_top1 += int((log_p.argmax(dim=-1) == yb).sum().item())

    if was_train:
        adapter.model.train()
    wall_s = max(1.0e-6, time.perf_counter() - t0)
    val_ce = total_nll / float(max(1, total_eval_tok))
    return {
        "official_shard0_val_ce": round(val_ce, 5),
        "official_shard0_val_ppl": round(math.exp(min(20.0, val_ce)), 4),
        "official_shard0_top1_pct": round(100.0 * float(total_top1) / float(max(1, total_eval_tok)), 4),
        "eval_tokens": int(total_eval_tok),
        "eval_tok_per_s": round(float(total_eval_tok) / wall_s, 1),
        "eval_wall_s": round(wall_s, 2),
    }

