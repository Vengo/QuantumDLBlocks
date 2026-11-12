"""Wave 79 Verification Sprints, Induction/MQAR Probes, & `300M` Multi-Shard WSD Runner (`EXPERIMENT_PLAN_099.md`).

Provenance & Mathematical Rationale
-----------------------------------
Orchestrates the `Wave 79` (`EXPERIMENT_PLAN_099.md`) controlled `400`-step (`3.28M`-token) `V0..V3`
verification sprints, `50,257`-vocab Fuzzy Induction & `MQAR` synthetic probes, and the `300M`-token
multi-shard `FineWeb` `WSD` speedrun (with an automatic `100M`-token iso-horizon branched WSD quench
evaluation on Official `10,485,760` Shard-0 tokens), obeying Rules 1–8 of `docs/SCRIPT_BEST_PRACTICES.md`.
"""

from __future__ import annotations

import copy
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
)
from .experiment_logger import ExperimentIntentSpec, StandardizedExperimentLogger
from .fused_bf16_four_compartment_baton_lab import (
    FINEWEB_TRAIN_SHARDS_300M,
    MultiShardFineWebStreamer,
)
from .wave77_baton_ladder_models import OFFICIAL_SHARD0_VAL_TOKENS
from .wave77_evidence_ladder_runner import MultiShardConfirmConfig
from .wave79_flex_and_mtp_diagnostics import (
    FlexDocMaskBenchConfig,
    audit_wave79_copy_and_schur_diagnostics,
    benchmark_flex_attention_bos_docmask_parity_and_speed,
    inspect_mtp_horizon_weights_and_rollout_cosines,
)
from .wave79_fused_schur_and_induction_baton import (
    FusedUnifiedSchurBatonLM,
    Wave79FusedAdapter,
    Wave79VariantKind,
    build_wave79_mup_optimizers,
    build_wave79_variant_adapter,
)

# ==============================================================================
# Top-of-File Named Constants (Synchronized with EXPERIMENT_PLAN_099.md Section 3)
# ==============================================================================
DEFAULT_W79_CKPT_DIR: Path = REPO_ROOT / ".cache" / "checkpoints" / "wave79"
W79_ISO100M_QUENCH_STEPS: int = 150
W79_ISO100M_TOKEN_MILESTONE: int = 100_000_000

__all__ = [
    "DEFAULT_W79_CKPT_DIR",
    "FlexDocMaskBenchConfig",
    "W79_ISO100M_QUENCH_STEPS",
    "audit_wave79_copy_and_schur_diagnostics",
    "benchmark_flex_attention_bos_docmask_parity_and_speed",
    "evaluate_wave79_official_shard0_validation",
    "inspect_mtp_horizon_weights_and_rollout_cosines",
    "run_wave79_300m_multishard_wsd_speedrun",
    "run_wave79_controlled_sprint",
    "run_wave79_induction_and_mqar_probes",
]


def run_wave79_controlled_sprint(
    variant: Wave79VariantKind,
    train_cfg: ShortTrainConfig,
    x_val: Tensor,
    y_val: Tensor,
    device: torch.device,
) -> tuple[Wave79FusedAdapter, dict[str, Any]]:
    """Run a controlled `400`-step (`3.28M`-token) FineWeb Shard-1 sprint on `variant` (`<= 8` args)."""
    adapter = build_wave79_variant_adapter(variant, device=device, seed=train_cfg.seed)
    p_info = adapter.parameter_breakdown()
    tokens_per_step = train_cfg.batch_size * train_cfg.seq_len
    total_tokens = train_cfg.steps * tokens_per_step

    train_tokens = load_fineweb_token_slice(
        "fineweb_train_000001.bin", num_tokens=total_tokens + train_cfg.seq_len + 8
    )

    run_key = f"{train_cfg.wave_id.lower()}_{variant.value}_{train_cfg.steps}s"
    intent = ExperimentIntentSpec(
        run_key=run_key,
        wave_id=train_cfg.wave_id,
        plan_id=train_cfg.plan_id,
        task_domain="fineweb_speedrun",
        hypothesis_intent=(
            f"Wave 79 controlled {train_cfg.steps}-step ({total_tokens / 1e6:.2f}M token) verification "
            f"sprint for {adapter.label} measuring steady-state tok/s, Hot/EMA CE, and L_inf."
        ),
        architecture_summary=(
            f"{adapter.label} ({p_info['total_params'] / 1e6:.2f}M total, "
            f"{p_info['trunk_params'] / 1e6:.2f}M trunk)"
        ),
        hardware_budget=torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        arch_family="wave79_fused_schur_and_induction_baton",
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
    ema_tracker = DualHorizonShadowEMATracker(adapter.model, shadow_device=device)

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
    steady_train_sec = 0.0
    steady_tokens = 0
    compile_step1_sec = 0.0

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
        dt = time.perf_counter() - t_step0
        if step <= 2 and adapter.fused_model._compiled_trunk_fn is not None:
            compile_step1_sec += dt
        else:
            steady_train_sec += dt
            steady_tokens += tokens_per_step
        ema_tracker.update(adapter.model)

        if step == 50 or step % train_cfg.eval_interval == 0 or step == train_cfg.steps:
            tok_m = (step * tokens_per_step) / 1e6
            cur_tok_s = float(max(tokens_per_step, steady_tokens)) / max(1.0e-6, steady_train_sec)
            ema_stats = ema_tracker.evaluate_hot_fast_slow(adapter, x_val, y_val)
            ema_stats["step"] = float(step)
            ema_stats["tokens_M"] = round(tok_m, 4)
            ema_stats["train_loss"] = round(tr_loss, 5)
            checkpoints.append(ema_stats)

            approx_tflops = (
                6.0 * float(p_info["total_params"]) * float(step * tokens_per_step)
            ) / 1e12
            lr_m = adapter._compute_lr_multiplier(step, train_cfg.steps)
            logger.log_step(step=step, train_ce=tr_loss, tok_per_s=cur_tok_s, lr_mult=lr_m)
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
    tok_per_s = float(max(tokens_per_step, steady_tokens)) / max(1.0e-6, steady_train_sec)

    d_pts = [c["tokens_M"] for c in checkpoints if c["tokens_M"] > 0.0]
    ce_pts = [c["slow_ema_val_ce"] for c in checkpoints if c["tokens_M"] > 0.0]
    asymptote = fit_early_powerlaw_asymptote(d_pts, ce_pts)
    final_ckpt = checkpoints[-1]
    best_ce = min(
        min(c["hot_val_ce"] for c in checkpoints),
        min(c["slow_ema_val_ce"] for c in checkpoints),
    )
    routing_telem = adapter.fused_model.inspect_routing_telemetry_on_sample(x_val.to(device))

    summary = {
        "run_key": run_key,
        "arch": variant.value,
        "label": adapter.label,
        "total_params": p_info["total_params"],
        "trunk_params": p_info["trunk_params"],
        "vocab_params": p_info["vocab_params"],
        "steps": train_cfg.steps,
        "tokens_M": round(total_tokens / 1e6, 4),
        "tok_per_s": round(tok_per_s, 1),
        "compile_warmup_sec": round(compile_step1_sec, 2),
        "steady_train_sec": round(steady_train_sec, 2),
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


def _eval_induction_batches(
    adapter: Wave79FusedAdapter,
    cfg: SyntheticProbeConfig,
    seq_t: int,
    gen: torch.Generator,
    device: torch.device,
) -> tuple[float, float, float]:
    """Return `(query_ce, query_top1_acc, pre_bos_trap_rate)` on unseen `50k`-vocab induction."""
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
    return q_nll / denom, float(q_top1) / denom, float(q_trap) / denom


def run_wave79_induction_and_mqar_probes(
    variants: list[Wave79VariantKind],
    cfg: SyntheticProbeConfig,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run `50,257`-vocab Fuzzy Induction ($T \\in \\{64, 128, 256\\}$) & `MQAR` ($K \\in \\{4, 8\\}$) probes."""
    ind_results: list[dict[str, Any]] = []
    mqar_results: list[dict[str, Any]] = []

    for variant in variants:
        for seq_t in SYNTHETIC_INDUCTION_SEQ_LENS:
            adapter = build_wave79_variant_adapter(
                variant, device=device, seed=cfg.seed, enable_compile=False
            )
            gen0 = torch.Generator().manual_seed(cfg.seed + 777 + int(seq_t))
            _, step0_acc, _ = _eval_induction_batches(adapter, cfg, int(seq_t), gen0, device)

            gen = torch.Generator().manual_seed(cfg.seed + int(seq_t))
            t0 = time.perf_counter()
            for step in range(1, cfg.induction_steps + 1):
                xb, yb, _ = generate_fuzzy_induction_batch(
                    cfg.batch_size, int(seq_t), cfg.vocab_size, gen, device
                )
                adapter.train_step(xb, yb, step=step, total_steps=cfg.induction_steps)

            q_ce, q_acc, trap_r = _eval_induction_batches(adapter, cfg, int(seq_t), gen, device)
            ind_results.append(
                {
                    "arch": variant.value,
                    "label": adapter.label,
                    "seq_len": int(seq_t),
                    "step0_zero_shot_top1_acc": round(step0_acc, 5),
                    "induction_query_ce": round(q_ce, 5),
                    "induction_query_top1_acc": round(q_acc, 5),
                    "pre_bos_false_trap_rate": round(trap_r, 5),
                    "elapsed_sec": round(time.perf_counter() - t0, 2),
                }
            )
            del adapter
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for k_pairs in SYNTHETIC_MQAR_PAIR_COUNTS:
            adapter = build_wave79_variant_adapter(
                variant, device=device, seed=cfg.seed + int(k_pairs), enable_compile=False
            )
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
                    "arch": variant.value,
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


@torch.no_grad()
def evaluate_wave79_official_shard0_validation(
    adapter: Wave79FusedAdapter,
    shard0_path: Path,
    val_tokens_budget: int = OFFICIAL_SHARD0_VAL_TOKENS,
    batch_size: int = 32,
    seq_len: int = 512,
) -> dict[str, float]:
    """Evaluate `adapter` on the official `10,485,760` (`10 * 2^20`) sequential Shard-0 tokens (`<= 8` args)."""
    mod = adapter.fused_model
    was_train = mod.training
    mod.eval()
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
    cap = mod.cfg.logit_softcap
    inv_c = 1.0 / cap
    sub_b = max(1, 4096 // int(seq_len))
    uni_s = F.softplus(mod.uni_prior_logit.float())
    uni_lp = torch.log(mod.online_uni_token_probs.detach().clamp_min(1.0e-9)).view(1, 1, -1)

    for b0 in range(0, n_seqs, batch_size):
        b1 = min(n_seqs, b0 + batch_size)
        xb = x_all[b0:b1].to(adapter.device, non_blocking=True)
        yb = y_all[b0:b1].to(adapter.device, non_blocking=True)

        with torch.amp.autocast(
            device_type=adapter.device.type, dtype=torch.bfloat16, enabled=amp_on
        ):
            x_fin, copy_lo, _, _ = mod.encode_trunk_features(xb)
            u_bias = (uni_s * uni_lp).to(x_fin.dtype)
            c_idx = xb.unsqueeze(1).expand(xb.size(0), seq_len, seq_len)
            for s0 in range(0, xb.size(0), sub_b):
                s1 = min(xb.size(0), s0 + sub_b)
                lg = F.linear(x_fin[s0:s1], mod.lm_head.weight) + u_bias
                lg = lg.scatter_add(
                    dim=-1, index=c_idx[s0:s1], src=copy_lo[s0:s1].to(lg.dtype)
                )
                capped = cap * torch.tanh(lg.float() * inv_c)
                nll = F.cross_entropy(
                    capped.reshape(-1, capped.size(-1)), yb[s0:s1].reshape(-1), reduction="sum"
                )
                total_nll += float(nll.item())
                total_top1 += int((capped.argmax(dim=-1) == yb[s0:s1]).sum().item())

    if was_train:
        mod.train()
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


def _run_iso100m_branched_quench_eval(
    adapter: Wave79FusedAdapter,
    ema_tracker: DualHorizonShadowEMATracker,
    streamer: MultiShardFineWebStreamer,
    start_step: int,
    spin_cfg: MultiShardConfirmConfig,
    shard0_path: Path,
    ckpt_out_path: Path,
) -> dict[str, Any]:
    """Clone model at `100M` tokens, run a `150`-step linear cooldown quench, & evaluate Official Shard-0."""
    t0 = time.perf_counter()
    saved_model_state = {k: v.detach().cpu().clone() for k, v in adapter.model.state_dict().items()}
    branch_model = FusedUnifiedSchurBatonLM(adapter.fused_model.cfg).to(adapter.device)
    branch_model.load_state_dict(saved_model_state, strict=True)
    with torch.no_grad():
        for name, p in branch_model.named_parameters():
            if name in ema_tracker.slow_shadow:
                p.copy_(ema_tracker.slow_shadow[name].to(device=p.device, dtype=p.dtype))
    if adapter.device.type == "cuda" and branch_model.cfg.compile_backbone:
        branch_model.enable_compilation()

    branch_opts = build_wave79_mup_optimizers(branch_model, warmup_steps=10)
    base_lrs = [[float(pg["lr"]) for pg in opt.param_groups] for opt in branch_opts]

    q_steps = W79_ISO100M_QUENCH_STEPS
    branch_model.train()
    for q_idx in range(1, q_steps + 1):
        lr_m = max(0.05, 0.55 * (1.0 - float(q_idx - 1) / float(max(1, q_steps))))
        for opt, b_lrs in zip(branch_opts, base_lrs):
            for pg, blr in zip(opt.param_groups, b_lrs):
                pg["lr"] = blr * lr_m
            opt.zero_grad(set_to_none=True)
        xb, yb, _ = streamer.sample_sequential_batch(
            start_step + q_idx - 1, spin_cfg.batch_size, spin_cfg.seq_len, adapter.device
        )
        with torch.amp.autocast(
            device_type=adapter.device.type,
            dtype=torch.bfloat16,
            enabled=(adapter.device.type == "cuda"),
        ):
            _, loss = branch_model(xb, targets=yb, ingest_online=True, return_logits=False)
        assert loss is not None
        loss.backward()
        for opt in branch_opts:
            opt.step()

    branch_adapter = Wave79FusedAdapter(
        key=f"{adapter.key}_iso100m_quench",
        label=f"{adapter.label} [Iso-100M Branched Quench]",
        model=branch_model,
        optimizers=branch_opts,
        device=adapter.device,
        use_wsd_schedule=True,
    )
    val_budget = (
        OFFICIAL_SHARD0_VAL_TOKENS
        if adapter.device.type == "cuda"
        else min(16384, OFFICIAL_SHARD0_VAL_TOKENS)
    )
    shard0_eval = evaluate_wave79_official_shard0_validation(
        branch_adapter,
        shard0_path=shard0_path,
        val_tokens_budget=val_budget,
        batch_size=spin_cfg.batch_size,
        seq_len=spin_cfg.seq_len,
    )
    ckpt_out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variant": adapter.key,
            "milestone": "100M_branched_quench",
            "quench_steps": q_steps,
            "official_shard0_eval": shard0_eval,
            "model_state_dict": branch_model.state_dict(),
        },
        ckpt_out_path,
    )
    del branch_adapter, branch_model, branch_opts
    if adapter.device.type == "cuda":
        torch.cuda.empty_cache()
    shard0_eval["branch_quench_wall_s"] = round(time.perf_counter() - t0, 2)
    shard0_eval["checkpoint_path"] = str(ckpt_out_path)
    return shard0_eval


def run_wave79_300m_multishard_wsd_speedrun(
    variant: Wave79VariantKind,
    spin_cfg: MultiShardConfirmConfig,
    x_xray: Tensor,
    y_xray: Tensor,
    decile_map: Tensor,
    ref_550m: TeacherReferenceProfile,
    device: torch.device,
) -> dict[str, Any]:
    """Execute a multi-shard `WSD` speedrun (`300M` default + `100M` iso-quench) & Official Shard-0 eval."""
    adapter = build_wave79_variant_adapter(
        variant, device=device, seed=spin_cfg.seed, use_wsd_schedule=True
    )
    p_info = adapter.parameter_breakdown()
    tokens_per_step = spin_cfg.batch_size * spin_cfg.seq_len
    total_steps = max(1, (spin_cfg.target_tokens + tokens_per_step - 1) // tokens_per_step)
    total_tokens = total_steps * tokens_per_step
    tok_m_tag = int(round(total_tokens / 1e6))
    iso100m_step = W79_ISO100M_TOKEN_MILESTONE // max(1, tokens_per_step)

    streamer = MultiShardFineWebStreamer(FINEWEB_TRAIN_SHARDS_300M)
    shard0_path = REPO_ROOT / ".cache" / "fineweb_100M" / "fineweb_val_000000.bin"
    run_key = f"{spin_cfg.wave_id.lower()}_{variant.value}_{tok_m_tag}m_wsd_spin"

    intent = ExperimentIntentSpec(
        run_key=run_key,
        wave_id=spin_cfg.wave_id,
        plan_id=spin_cfg.plan_id,
        task_domain="fineweb_speedrun",
        hypothesis_intent=(
            f"Wave 79 {tok_m_tag}M-token multi-shard WSD speedrun on {adapter.label} testing "
            f"100M iso-horizon Official Shard-0 CE (< 4.3087) and 300M Official Shard-0 CE (< 4.00)."
        ),
        architecture_summary=(
            f"{adapter.label} ({p_info['total_params'] / 1e6:.2f}M total, "
            f"{p_info['trunk_params'] / 1e6:.2f}M trunk)"
        ),
        hardware_budget=torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        arch_family="wave79_fused_unified_wsd_speedrun",
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
    t_wall0 = time.perf_counter()
    steady_sec = 0.0
    steady_toks = 0
    iso100m_eval: dict[str, Any] | None = None

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
        dt = time.perf_counter() - ts0
        if step > 2:
            steady_sec += dt
            steady_toks += tokens_per_step
        ema_tracker.update(adapter.model)

        if step == 100 or step % spin_cfg.eval_interval == 0 or step == iso100m_step or step == total_steps:
            tok_m = (step * tokens_per_step) / 1e6
            cur_tok_s = float(max(tokens_per_step, steady_toks)) / max(1.0e-6, steady_sec)
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
                    "val_ce": min(ema_stats["hot_val_ce"], ema_stats["slow_ema_val_ce"]),
                    "top1_acc": ema_stats["hot_top1_acc"] * 100.0,
                    "tok_per_s": cur_tok_s,
                    "hot_val_ce": ema_stats["hot_val_ce"],
                    "fast_ema_val_ce": ema_stats["fast_ema_val_ce"],
                    "slow_ema_val_ce": ema_stats["slow_ema_val_ce"],
                    "thermal_gap_slow": ema_stats["thermal_gap_slow"],
                },
            )
            print(
                f"  [W79 Spin {step:5d}/{total_steps}] {tok_m:6.2f}M tok | "
                f"lr_m={lr_m:.3f} | {cur_tok_s:8,.1f} tok/s | "
                f"Hot={ema_stats['hot_val_ce']:.4f} | FastEMA={ema_stats['fast_ema_val_ce']:.4f} | "
                f"SlowEMA={ema_stats['slow_ema_val_ce']:.4f}",
                flush=True,
            )

        if step == iso100m_step and total_tokens >= 150_000_000 and shard0_path.exists():
            ckpt_100m = DEFAULT_W79_CKPT_DIR / f"{run_key}_100m_quenched.pt"
            iso100m_eval = _run_iso100m_branched_quench_eval(
                adapter, ema_tracker, streamer, step, spin_cfg, shard0_path, ckpt_100m
            )
            print(
                f"  >>> [Iso-100M Branched Quench @ step {step}] Official 10.49M Shard-0 Val CE = "
                f"{iso100m_eval['official_shard0_val_ce']:.5f} "
                f"(Top-1={iso100m_eval['official_shard0_top1_pct']:.2f}%, "
                f"{iso100m_eval['branch_quench_wall_s']:.1f}s) <<<",
                flush=True,
            )

    final_ckpt = checkpoints[-1]
    used_ema_weights = False
    if final_ckpt["slow_ema_val_ce"] < final_ckpt["hot_val_ce"]:
        with torch.no_grad():
            for name, p in adapter.model.named_parameters():
                if name in ema_tracker.slow_shadow:
                    p.copy_(ema_tracker.slow_shadow[name].to(device=p.device, dtype=p.dtype))
        used_ema_weights = True

    val_budget = (
        OFFICIAL_SHARD0_VAL_TOKENS
        if device.type == "cuda"
        else min(16384, OFFICIAL_SHARD0_VAL_TOKENS)
    )
    official_eval = evaluate_wave79_official_shard0_validation(
        adapter,
        shard0_path=shard0_path,
        val_tokens_budget=val_budget,
        batch_size=spin_cfg.batch_size,
        seq_len=spin_cfg.seq_len,
    )
    xray = compute_stratified_nat_gap_xray(
        adapter, x_xray, y_xray, decile_map=decile_map, ref_profile=ref_550m
    )
    copy_schur_diag = audit_wave79_copy_and_schur_diagnostics(adapter, x_xray, y_xray, decile_map)
    mtp_diag = inspect_mtp_horizon_weights_and_rollout_cosines(adapter.fused_model, x_xray, y_xray)
    routing_telem = adapter.fused_model.inspect_routing_telemetry_on_sample(x_xray.to(device))
    asym_300m = fit_early_powerlaw_asymptote(
        [c["tokens_M"] for c in checkpoints if c["tokens_M"] > 0.0],
        [c["slow_ema_val_ce"] for c in checkpoints if c["tokens_M"] > 0.0],
    )

    DEFAULT_W79_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_final = DEFAULT_W79_CKPT_DIR / f"{run_key}_final.pt"
    torch.save(
        {
            "variant": variant.value,
            "run_key": run_key,
            "total_steps": total_steps,
            "total_tokens": total_tokens,
            "official_shard0_eval": official_eval,
            "iso100m_branched_quench_eval": iso100m_eval,
            "model_state_dict": copy.deepcopy(adapter.model.state_dict()),
        },
        ckpt_final,
    )

    total_wall_s = time.perf_counter() - t_wall0
    tok_per_s = float(max(tokens_per_step, steady_toks)) / max(1.0e-6, steady_sec)
    logger.log_eval(
        step=total_steps,
        metrics={
            "tokens_M": total_tokens / 1e6,
            "compute_tflops": (6.0 * float(p_info["total_params"]) * float(total_tokens)) / 1e12,
            "tau_progress": 1.0,
            "val_ce": official_eval["official_shard0_val_ce"],
            "val_ppl": official_eval["official_shard0_val_ppl"],
            "top1_acc": official_eval["official_shard0_top1_pct"],
            "tok_per_s": tok_per_s,
        },
    )

    summary = {
        "run_key": run_key,
        "variant": variant.value,
        "label": adapter.label,
        "total_params": p_info["total_params"],
        "trunk_params": p_info["trunk_params"],
        "total_steps": total_steps,
        "total_tokens": total_tokens,
        "tok_per_s": round(tok_per_s, 1),
        "steady_train_sec": round(steady_sec, 2),
        "total_wall_sec": round(total_wall_s, 2),
        "used_slow_ema_for_final_eval": used_ema_weights,
        "final_hot_val_ce": final_ckpt["hot_val_ce"],
        "final_fast_ema_val_ce": final_ckpt["fast_ema_val_ce"],
        "final_slow_ema_val_ce": final_ckpt["slow_ema_val_ce"],
        "asymptote_fit": asym_300m,
        "iso100m_branched_quench_eval": iso100m_eval,
        "official_shard0_eval": official_eval,
        "routing_telemetry": routing_telem,
        "copy_and_schur_diagnostics": copy_schur_diag,
        "mtp_horizon_diagnostics": mtp_diag,
        "xray_vs_550m": xray,
        "checkpoint_path": str(ckpt_final),
        "checkpoints": checkpoints,
    }
    logger.finalize(extra_summary=summary)
    return summary
