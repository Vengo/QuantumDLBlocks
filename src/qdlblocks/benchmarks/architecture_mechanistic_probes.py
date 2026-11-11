"""Mechanistic Induction, MQAR, Readout Ceiling, Compartment Ablation, & Short-Sprint Probes (`Wave 76` / `EXPERIMENT_PLAN_096.md`).

Provenance & Mathematical Rationale
-----------------------------------
Companion module to `architecture_xray_diagnostics.py` built for `Wave 76` (`EXPERIMENT_PLAN_096.md`)
obeying Rules 1–8 of `docs/SCRIPT_BEST_PRACTICES.md` (`<= 1,000` SLOC per file, `<= 8` args):

1. **Optimization Thermodynamics & Early Asymptote Prediction**:
   - `DualHorizonShadowEMATracker`: Tracks `Hot CE`, `Fast-EMA CE` ($\\beta = 0.980$), and
     `Slow-EMA CE` ($\\beta = 0.995$) to separate high-LR thermal noise from structural capacity ceilings.
   - `fit_early_powerlaw_asymptote`: Fits $L(D) = L_\\infty + B D^{-\\alpha}$ ($B \\ge 0$) over early
     token horizons (`0.4M..3.3M` tokens) to estimate the `50M` and `300M` scaling floor $L_\\infty$.
   - `run_controlled_short_fineweb_sprint`: Standardized short-horizon FineWeb sprint runner
     emitting `StandardizedExperimentLogger` logs, `ModelCard v1.0`, and SQLite records.

2. **Mechanistic & Ceiling Probes**:
   - `run_synthetic_induction_sweep`: Tests `50,257`-vocab Fuzzy Induction `[prior_doc <BOS> ... A B ... A -> B]`
     with a competing pre-`<BOS>` trap `[A, C_false]` across $T \\in \\{64, 128, 256\\}$.
   - `run_synthetic_mqar_probe`: Tests simultaneous $K \\in \\{4, 8\\}$ key-value associative binding.
   - `run_memorization_and_readout_ceiling_probe`: Karpathy `N=1, 4` real FineWeb record memorization
     testing whether unit-sphere Born readout clips sharp distributions.
   - `run_baton_compartment_ablation_audit`: Counterfactual leave-one-out $\\Delta\\text{CE}$ across
     compartments, post-hoc $\\beta \\in \\{8..40\\}$ readout scale sweep, attention SNR / `<BOS>`
     leakage telemetry, and frozen-trunk uncollapsed ridge readout probing.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .architecture_xray_diagnostics import (
    ASYMPTOTE_ALPHA_MAX,
    ASYMPTOTE_ALPHA_MIN,
    ASYMPTOTE_ALPHA_STEPS,
    DEFAULT_EMA_FAST_DECAY,
    DEFAULT_EMA_SLOW_DECAY,
    FINEWEB_BOS_TOKEN_ID,
    LOG_PROB_FLOOR,
    READOUT_SCALE_SWEEP_VALUES,
    RIDGE_PROBE_LAMBDA,
    SYNTHETIC_INDUCTION_SEQ_LENS,
    SYNTHETIC_MQAR_PAIR_COUNTS,
    VALID_FINEWEB_VOCAB_SIZE,
    DiagnosticModelAdapter,
    KnownArchitectureKind,
    ShortTrainConfig,
    SyntheticProbeConfig,
    build_known_architecture_adapter,
    load_fineweb_token_slice,
)
from .exact_mixed_state_baton_width_scaling_lab import (
    exact_mixed_state_attention_schur_action,
    exact_token_id_mixed_state_action,
)
from .experiment_logger import ExperimentIntentSpec, StandardizedExperimentLogger
from .fused_bf16_four_compartment_baton_lab import (
    DEFAULT_BORN_READOUT_SCALE,
    DEFAULT_DOC_BATH_MIX,
    DEFAULT_IN_CONTEXT_COPY_SCALE,
    DEFAULT_LOGIT_SOFTCAP,
    DEFAULT_TOKEN_ID_COPY_SCALE,
    DEFAULT_TRACE_EPS,
    DEFAULT_UNIGRAM_PRIOR_WEIGHT,
    DEFAULT_VACUUM_EPS,
    FusedBF16FourCompartmentBatonLM,
    apply_rank_beams_to_probe,
    diagonal_conjugated_bath_schur_action,
    encode_walsh_hadamard_token_id_kets,
    factorized_rank_beams_schur_intersect,
    normalize_complex_ket,
)

# ==============================================================================
# Top-of-File Named Constants (Synchronized with EXPERIMENT_PLAN_096.md Section 3)
# ==============================================================================
PARSEVAL_MUP_C_STAR: float = 1.48
ASYMPTOTE_MIN_FLOOR_CE: float = 1.50


# ==============================================================================
# Engine 2: Dual-Horizon Thermal Gap & Power-Law Asymptote Predictor
# ==============================================================================
@torch.no_grad()
def evaluate_quick_ce_and_top1(
    adapter: DiagnosticModelAdapter,
    x_val: Tensor,
    y_val: Tensor,
    micro_batch: int = 8,
) -> tuple[float, float]:
    """Fast chunked validation CE and Top-1 fraction (`0..1`) evaluation (`<= 8` args)."""
    total_nll = 0.0
    total_ok = 0
    total_tok = 0
    n_seq = x_val.size(0)
    for start in range(0, n_seq, micro_batch):
        end = min(n_seq, start + micro_batch)
        bx = x_val[start:end].to(adapter.device)
        by = y_val[start:end].to(adapter.device)
        log_p, _ = adapter.predict_log_probs_and_span(bx)
        nll = -log_p.gather(dim=-1, index=by.unsqueeze(-1)).squeeze(-1)
        total_nll += float(nll.sum().item())
        total_ok += int((log_p.argmax(dim=-1) == by).sum().item())
        total_tok += int(by.numel())
    denom = float(max(1, total_tok))
    return total_nll / denom, float(total_ok) / denom


class DualHorizonShadowEMATracker:
    """Maintains Fast ($\\beta=0.980$) and Slow ($\\beta=0.995$) parameter EMAs to measure Thermal Gap."""

    def __init__(
        self,
        model: nn.Module,
        fast_decay: float = DEFAULT_EMA_FAST_DECAY,
        slow_decay: float = DEFAULT_EMA_SLOW_DECAY,
        shadow_device: torch.device | str = "cpu",
    ) -> None:
        self.fast_decay = float(fast_decay)
        self.slow_decay = float(slow_decay)
        self.shadow_device = torch.device(shadow_device)
        self.fast_shadow: dict[str, Tensor] = {}
        self.slow_shadow: dict[str, Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                snap = param.detach().float().to(self.shadow_device).clone()
                self.fast_shadow[name] = snap.clone()
                self.slow_shadow[name] = snap.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update both Fast and Slow shadow EMA buffers from current hot parameters (`<= 8` args)."""
        b_fast, b_slow = self.fast_decay, self.slow_decay
        for name, param in model.named_parameters():
            if name in self.fast_shadow:
                cur = param.detach().float().to(self.shadow_device)
                self.fast_shadow[name].mul_(b_fast).add_(cur, alpha=1.0 - b_fast)
                self.slow_shadow[name].mul_(b_slow).add_(cur, alpha=1.0 - b_slow)

    @torch.no_grad()
    def evaluate_hot_fast_slow(
        self,
        adapter: DiagnosticModelAdapter,
        x_val: Tensor,
        y_val: Tensor,
        micro_batch: int = 8,
    ) -> dict[str, float]:
        """Evaluate `Hot CE`, `Fast-EMA CE`, and `Slow-EMA CE`, then restore exact hot parameters (`<= 8` args)."""
        hot_ce, hot_top1 = evaluate_quick_ce_and_top1(adapter, x_val, y_val, micro_batch=micro_batch)
        backup: dict[str, Tensor] = {
            name: p.detach().clone()
            for name, p in adapter.model.named_parameters()
            if name in self.fast_shadow
        }
        for name, p in adapter.model.named_parameters():
            if name in self.fast_shadow:
                p.copy_(self.fast_shadow[name].to(device=p.device, dtype=p.dtype))
        fast_ce, _ = evaluate_quick_ce_and_top1(adapter, x_val, y_val, micro_batch=micro_batch)

        for name, p in adapter.model.named_parameters():
            if name in self.slow_shadow:
                p.copy_(self.slow_shadow[name].to(device=p.device, dtype=p.dtype))
        slow_ce, slow_top1 = evaluate_quick_ce_and_top1(adapter, x_val, y_val, micro_batch=micro_batch)

        for name, p in adapter.model.named_parameters():
            if name in backup:
                p.copy_(backup[name])

        return {
            "hot_val_ce": round(hot_ce, 5),
            "fast_ema_val_ce": round(fast_ce, 5),
            "slow_ema_val_ce": round(slow_ce, 5),
            "thermal_gap_fast": round(hot_ce - fast_ce, 5),
            "thermal_gap_slow": round(hot_ce - slow_ce, 5),
            "hot_top1_acc": round(hot_top1, 5),
            "slow_ema_top1_acc": round(slow_top1, 5),
        }


def fit_early_powerlaw_asymptote(
    tokens_millions: list[float],
    val_ces: list[float],
) -> dict[str, float]:
    """Fit $L(D) = L_\\infty + B \\cdot D^{-\\alpha}$ ($B \\ge 0, L_\\infty \\ge 1.5$) via grid OLS (`<= 8` args)."""
    pts = [(float(d), float(l)) for d, l in zip(tokens_millions, val_ces) if d > 0.0]
    if len(pts) < 3:
        last_ce = float(val_ces[-1]) if val_ces else 10.825
        return {
            "l_inf_asymptote": round(last_ce, 4),
            "scale_b": 0.0,
            "alpha_exp": 0.0,
            "r2_fit": 0.0,
            "pred_ce_50m": round(last_ce, 4),
            "pred_ce_300m": round(last_ce, 4),
        }
    d_arr = np.array([p[0] for p in pts], dtype=np.float64)
    y_arr = np.array([p[1] for p in pts], dtype=np.float64)
    y_var = float(np.sum((y_arr - y_arr.mean()) ** 2))

    best_sse = float("inf")
    best_params = (float(y_arr[-1]), 0.0, 0.25)
    alphas = np.linspace(ASYMPTOTE_ALPHA_MIN, ASYMPTOTE_ALPHA_MAX, ASYMPTOTE_ALPHA_STEPS)
    for alpha in alphas:
        x_feat = d_arr ** (-float(alpha))
        x_mean = float(x_feat.mean())
        y_mean = float(y_arr.mean())
        denom = float(np.sum((x_feat - x_mean) ** 2))
        if denom < 1e-12:
            continue
        b_val = max(0.0, float(np.sum((x_feat - x_mean) * (y_arr - y_mean))) / denom)
        l_inf = max(ASYMPTOTE_MIN_FLOOR_CE, y_mean - b_val * x_mean)
        pred = l_inf + b_val * x_feat
        sse = float(np.sum((y_arr - pred) ** 2))
        if sse < best_sse:
            best_sse = sse
            best_params = (l_inf, b_val, float(alpha))

    l_inf, b_val, alpha = best_params
    r2 = 1.0 - (best_sse / max(1e-12, y_var))
    pred_50m = l_inf + b_val * (50.0 ** (-alpha))
    pred_300m = l_inf + b_val * (300.0 ** (-alpha))
    return {
        "l_inf_asymptote": round(l_inf, 4),
        "scale_b": round(b_val, 4),
        "alpha_exp": round(alpha, 4),
        "r2_fit": round(max(0.0, r2), 5),
        "pred_ce_50m": round(pred_50m, 4),
        "pred_ce_300m": round(pred_300m, 4),
    }


def run_controlled_short_fineweb_sprint(
    kind: KnownArchitectureKind,
    train_cfg: ShortTrainConfig,
    x_val: Tensor,
    y_val: Tensor,
    device: torch.device,
) -> tuple[DiagnosticModelAdapter, dict[str, Any]]:
    """Train `kind` on sequential Shard-1 FineWeb batches with StandardizedExperimentLogger + Dual-EMA."""
    adapter = build_known_architecture_adapter(kind, device=device, seed=train_cfg.seed)
    p_info = adapter.parameter_breakdown()
    tokens_per_step = train_cfg.batch_size * train_cfg.seq_len
    total_tokens = train_cfg.steps * tokens_per_step

    train_tokens = load_fineweb_token_slice(
        "fineweb_train_000001.bin", num_tokens=total_tokens + train_cfg.seq_len + 8
    )

    run_key = f"{train_cfg.wave_id.lower()}_{kind.value}_{train_cfg.steps}s"
    intent = ExperimentIntentSpec(
        run_key=run_key,
        wave_id=train_cfg.wave_id,
        plan_id=train_cfg.plan_id,
        task_domain="fineweb_speedrun",
        hypothesis_intent=(
            f"Wave 76 controlled {train_cfg.steps}-step ({total_tokens / 1e6:.2f}M token) diagnostic sprint "
            f"for {adapter.label} measuring stratified nat-gap, dual-EMA thermal gap, and asymptote L_inf."
        ),
        architecture_summary=(
            f"{adapter.label} ({p_info['total_params'] / 1e6:.2f}M total, "
            f"{p_info['trunk_params'] / 1e6:.2f}M trunk)"
        ),
        hardware_budget=torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        arch_family="quantum_xray_diagnostic",
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

        is_ckpt = (
            step == 50
            or step % train_cfg.eval_interval == 0
            or step == train_cfg.steps
        )
        if is_ckpt:
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
            logger.log_step(
                step=step,
                train_ce=tr_loss,
                tok_per_s=cur_tok_s,
                lr_mult=1.0,
            )
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

    summary = {
        "run_key": run_key,
        "arch": kind.value,
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
        "checkpoints": checkpoints,
    }
    logger.finalize(extra_summary=summary)
    return adapter, summary


# ==============================================================================
# Engine 3: Synthetic Induction, MQAR, Readout Ceiling, & Compartment Ablations
# ==============================================================================
def generate_fuzzy_induction_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    gen: torch.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate `50k`-vocab induction sequences `[prior_doc: ... A C_false ... <BOS> ... A B ... A -> B]`.

    Places a competing false pair `[A, C_false]` *before* `<BOS>` in the previous document and the
    true pair `[A, B]` *after* `<BOS>` in the active document, querying `A` at `t = seq_len - 1`.
    """
    eff_v = max(256, int(vocab_size) - 1)
    tokens = torch.randint(0, eff_v, (batch_size, seq_len + 1), generator=gen)
    bos_pos = max(6, seq_len // 4)
    tokens[:, bos_pos] = FINEWEB_BOS_TOKEN_ID

    key_a = torch.randint(0, eff_v, (batch_size,), generator=gen)
    val_b = torch.randint(0, eff_v, (batch_size,), generator=gen)
    val_false = (val_b + 17) % eff_v

    trap_pos = max(1, bos_pos // 2 - 1)
    tokens[:, trap_pos] = key_a
    tokens[:, trap_pos + 1] = val_false

    low_src = bos_pos + 2
    high_src = max(low_src + 1, seq_len - 6)
    src_offsets = torch.randint(low_src, high_src, (batch_size,), generator=gen)
    for b_idx in range(batch_size):
        p = int(src_offsets[b_idx].item())
        tokens[b_idx, p] = key_a[b_idx]
        tokens[b_idx, p + 1] = val_b[b_idx]
    tokens[:, seq_len - 1] = key_a
    tokens[:, seq_len] = val_b

    x_b = tokens[:, :seq_len].to(device)
    y_b = tokens[:, 1 : seq_len + 1].to(device)
    q_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    q_mask[:, seq_len - 1] = True
    return x_b, y_b, q_mask


def run_synthetic_induction_sweep(
    kinds: list[KnownArchitectureKind],
    cfg: SyntheticProbeConfig,
    device: torch.device,
    seq_lens: tuple[int, ...] = SYNTHETIC_INDUCTION_SEQ_LENS,
) -> list[dict[str, Any]]:
    """Train & evaluate architectures on `50,257`-vocab Fuzzy Induction across $T \\in \\{64, 128, 256\\}$."""
    results: list[dict[str, Any]] = []
    for kind in kinds:
        for seq_t in seq_lens:
            adapter = build_known_architecture_adapter(kind, device=device, seed=cfg.seed)
            gen = torch.Generator().manual_seed(cfg.seed + int(seq_t))
            t0 = time.perf_counter()
            for step in range(1, cfg.induction_steps + 1):
                xb, yb, _ = generate_fuzzy_induction_batch(
                    cfg.batch_size, seq_t, cfg.vocab_size, gen, device
                )
                adapter.train_step(xb, yb, step=step, total_steps=cfg.induction_steps)

            q_nll_sum = 0.0
            q_top1 = 0
            q_false_trap = 0
            q_count = 0
            for _ in range(cfg.eval_batches):
                xb, yb, q_mask = generate_fuzzy_induction_batch(
                    cfg.batch_size, seq_t, cfg.vocab_size, gen, device
                )
                eff_v = max(256, int(cfg.vocab_size) - 1)
                false_targets = (yb[q_mask] + 17) % eff_v
                with torch.no_grad():
                    log_p, _ = adapter.predict_log_probs_and_span(xb)
                    nll = -log_p.gather(dim=-1, index=yb.unsqueeze(-1)).squeeze(-1)
                    preds = log_p.argmax(dim=-1)
                    q_nll_sum += float(nll[q_mask].sum().item())
                    q_top1 += int((preds[q_mask] == yb[q_mask]).sum().item())
                    q_false_trap += int((preds[q_mask] == false_targets).sum().item())
                    q_count += int(q_mask.sum().item())

            elapsed = time.perf_counter() - t0
            denom = float(max(1, q_count))
            results.append(
                {
                    "arch": kind.value,
                    "label": adapter.label,
                    "seq_len": int(seq_t),
                    "induction_query_ce": round(q_nll_sum / denom, 5),
                    "induction_query_top1_acc": round(float(q_top1) / denom, 5),
                    "pre_bos_false_trap_rate": round(float(q_false_trap) / denom, 5),
                    "elapsed_sec": round(elapsed, 2),
                }
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return results


def generate_mqar_batch(
    batch_size: int,
    seq_len: int,
    num_pairs: int,
    vocab_size: int,
    gen: torch.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate `50,257`-vocab Multi-Query Associative Recall (`MQAR`) batch with `num_pairs` bindings."""
    eff_v = max(512, int(vocab_size) - 1)
    half_v = eff_v // 2
    tokens = torch.randint(0, eff_v, (batch_size, seq_len + 1), generator=gen)
    q_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

    spacing = max(3, (seq_len // 2) // max(1, num_pairs))
    query_start = seq_len - 2 * num_pairs - 1
    for b_idx in range(batch_size):
        keys = torch.randperm(min(4096, half_v), generator=gen)[:num_pairs]
        vals = half_v + torch.randint(0, half_v, (num_pairs,), generator=gen)
        for p_i in range(num_pairs):
            pos = 1 + p_i * spacing
            tokens[b_idx, pos] = keys[p_i]
            tokens[b_idx, pos + 1] = vals[p_i]
        perm = torch.randperm(num_pairs, generator=gen)
        for q_i in range(num_pairs):
            idx_pair = int(perm[q_i].item())
            q_pos = query_start + 2 * q_i
            tokens[b_idx, q_pos] = keys[idx_pair]
            tokens[b_idx, q_pos + 1] = vals[idx_pair]
            q_mask[b_idx, q_pos] = True

    return (
        tokens[:, :seq_len].to(device),
        tokens[:, 1 : seq_len + 1].to(device),
        q_mask.to(device),
    )


def run_synthetic_mqar_probe(
    kinds: list[KnownArchitectureKind],
    cfg: SyntheticProbeConfig,
    device: torch.device,
    pair_counts: tuple[int, ...] = SYNTHETIC_MQAR_PAIR_COUNTS,
) -> list[dict[str, Any]]:
    """Evaluate simultaneous $K$-pair associative binding capacity on `T=128` across architectures."""
    seq_t = 128
    results: list[dict[str, Any]] = []
    for kind in kinds:
        for k_pairs in pair_counts:
            adapter = build_known_architecture_adapter(
                kind, device=device, seed=cfg.seed + k_pairs
            )
            gen = torch.Generator().manual_seed(cfg.seed + 100 * k_pairs)
            t0 = time.perf_counter()
            for step in range(1, cfg.mqar_steps + 1):
                xb, yb, _ = generate_mqar_batch(
                    cfg.batch_size, seq_t, k_pairs, cfg.vocab_size, gen, device
                )
                adapter.train_step(xb, yb, step=step, total_steps=cfg.mqar_steps)

            nll_sum = 0.0
            top1_ok = 0
            total_q = 0
            for _ in range(cfg.eval_batches):
                xb, yb, q_mask = generate_mqar_batch(
                    cfg.batch_size, seq_t, k_pairs, cfg.vocab_size, gen, device
                )
                with torch.no_grad():
                    log_p, _ = adapter.predict_log_probs_and_span(xb)
                    nll = -log_p.gather(dim=-1, index=yb.unsqueeze(-1)).squeeze(-1)
                    preds = log_p.argmax(dim=-1)
                    nll_sum += float(nll[q_mask].sum().item())
                    top1_ok += int((preds[q_mask] == yb[q_mask]).sum().item())
                    total_q += int(q_mask.sum().item())

            denom = float(max(1, total_q))
            results.append(
                {
                    "arch": kind.value,
                    "label": adapter.label,
                    "num_pairs": int(k_pairs),
                    "mqar_ce": round(nll_sum / denom, 5),
                    "mqar_top1_acc": round(float(top1_ok) / denom, 5),
                    "elapsed_sec": round(time.perf_counter() - t0, 2),
                }
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return results


@torch.no_grad()
def inspect_baton_compartment_and_readout_states(
    model: FusedBF16FourCompartmentBatonLM,
    idx: Tensor,
    targets: Tensor,
) -> dict[str, Any]:
    """Extract uncollapsed Baton features, attention SNR/<BOS> leakage, scale sweep, & leave-one-out CE."""
    was_train = model.training
    model.eval()
    bsz, seq_len = idx.shape
    d = model.config.hilbert_dim

    ket_re, ket_im = normalize_complex_ket(
        F.embedding(idx, model.token_ket_re).float(),
        F.embedding(idx, model.token_ket_im).float(),
    )
    steps = torch.arange(1, seq_len + 1, device=idx.device, dtype=torch.float32).view(
        1, seq_len, 1
    )
    doc_ket_re, doc_ket_im = normalize_complex_ket(
        torch.cumsum(ket_re, dim=1) / steps,
        torch.cumsum(ket_im, dim=1) / steps,
    )
    doc_diag = doc_ket_re.square() + doc_ket_im.square()
    uni_bath_re = model.online_uni_bath_re.detach().float()
    uni_bath_im = model.online_uni_bath_im.detach().float()
    trans_re = model.online_trans_ema_re.detach().float()
    trans_im = model.online_trans_ema_im.detach().float()
    uni_probs = model.online_uni_token_probs.detach().float()

    uni_global_diag = torch.diagonal(uni_bath_re).view(1, 1, d)
    uni_diag = (1.0 - DEFAULT_DOC_BATH_MIX) * uni_global_diag + DEFAULT_DOC_BATH_MIX * doc_diag
    id_ket_re, id_ket_im = encode_walsh_hadamard_token_id_kets(idx, model.walsh_parity_pairs, d)
    id_ket_re, id_ket_im = id_ket_re.float(), id_ket_im.float()

    hash_ids = model._compute_causal_ngram_hash_ids(idx)
    hash_ket_re = F.embedding(hash_ids, model.online_ngram_hash_re).detach().float()
    hash_ket_im = F.embedding(hash_ids, model.online_ngram_hash_im).detach().float()

    geom = model._get_cached_sequence_geometry(seq_len, idx.device, torch.float32)
    doc_ids = (idx == FINEWEB_BOS_TOKEN_ID).cumsum(dim=1)
    doc_same = (doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)).float()
    aux_pack = {
        "cos_pos": geom["cos_pos"].float(),
        "sin_pos": geom["sin_pos"].float(),
        "head_masks": geom["head_masks"].float(),
        "strict_past_mask": geom["strict_past_mask"].float(),
        "causal_decay": geom["causal_decay"].float(),
        "local_op": geom["local_op"].float(),
        "id_ket_re": id_ket_re,
        "id_ket_im": id_ket_im,
        "doc_same_mask": doc_same,
    }

    curr_re, curr_im = ket_re, ket_im
    prev_deep_re: Tensor | None = None
    prev_deep_im: Tensor | None = None
    layer_out: dict[str, Tensor] = {}
    for layer_idx, lyr in enumerate(model.layers):
        if hasattr(lyr, "current_doc_same_mask"):
            lyr.current_doc_same_mask = doc_same.bool() if getattr(lyr, "mask_bos_documents", False) else None
        layer_out = lyr(
            curr_re, curr_im, uni_diag, trans_re, trans_im, hash_ket_re, hash_ket_im, aux_pack
        )
        if hasattr(lyr, "current_doc_same_mask"):
            lyr.current_doc_same_mask = None
        curr_re, curr_im = layer_out["next_ket_re"].float(), layer_out["next_ket_im"].float()
        if (
            layer_idx > 0
            and prev_deep_re is not None
            and prev_deep_im is not None
            and "deep_beams_re" in layer_out
        ):
            fw_re, fw_im, _ = factorized_rank_beams_schur_intersect(
                prev_deep_re,
                prev_deep_im,
                layer_out["deep_beams_re"].float(),
                layer_out["deep_beams_im"].float(),
            )
            bl_re = 0.65 * layer_out["deep_beams_re"].float() + 0.35 * fw_re
            bl_im = 0.65 * layer_out["deep_beams_im"].float() + 0.35 * fw_im
            inv_f = torch.rsqrt(
                (bl_re.square() + bl_im.square())
                .sum(dim=(-2, -1), keepdim=True)
                .clamp_min(DEFAULT_TRACE_EPS)
            )
            layer_out["deep_beams_re"] = bl_re * inv_f
            layer_out["deep_beams_im"] = bl_im * inv_f
        if "deep_beams_re" in layer_out:
            prev_deep_re = layer_out["deep_beams_re"].float()
            prev_deep_im = layer_out["deep_beams_im"].float()

    seq_attn = layer_out["seq_attn_probs"].float()
    eff_horizon = 1.0 / seq_attn.square().sum(dim=-1).clamp_min(1e-9)
    max_attn_peak = seq_attn.amax(dim=-1)
    diff_doc_mask = 1.0 - doc_same.unsqueeze(1)
    bos_leak_mass = (seq_attn * diff_doc_mask).sum(dim=-1)
    has_prior_doc = (doc_ids > 0).unsqueeze(1).expand_as(bos_leak_mass)
    cross_bos_leak = (
        float(bos_leak_mass[has_prior_doc].mean().item())
        if bool(has_prior_doc.any().item())
        else 0.0
    )
    long_slice = slice(seq_len // 2, seq_len)

    probe_re, probe_im = normalize_complex_ket(
        torch.matmul(curr_re, model.readout_probe_re.float())
        - torch.matmul(curr_im, model.readout_probe_im.float()),
        torch.matmul(curr_re, model.readout_probe_im.float())
        + torch.matmul(curr_im, model.readout_probe_re.float()),
    )
    ngram_re, ngram_im = layer_out["ngram_beams_re"].float(), layer_out["ngram_beams_im"].float()
    if "deep_beams_re" in layer_out:
        deep_re, deep_im = layer_out["deep_beams_re"].float(), layer_out["deep_beams_im"].float()
        act_deep_re, act_deep_im = apply_rank_beams_to_probe(deep_re, deep_im, probe_re, probe_im)
        bm_src_re, bm_src_im = deep_re, deep_im
    else:
        v_re, v_im = layer_out["v_re"].float(), layer_out["v_im"].float()
        act_deep_re, act_deep_im, _ = exact_mixed_state_attention_schur_action(
            seq_attn, v_re, v_im, ngram_re, ngram_im, probe_re, probe_im
        )
        bm_src_re, bm_src_im = ngram_re, ngram_im

    act_ng_re, act_ng_im, _ = diagonal_conjugated_bath_schur_action(
        ngram_re, ngram_im, uni_bath_re, uni_bath_im, probe_re, probe_im
    )
    act_uni_re = torch.matmul(probe_re, uni_bath_re.t()) - torch.matmul(probe_im, uni_bath_im.t())
    act_uni_im = torch.matmul(probe_re, uni_bath_im.t()) + torch.matmul(probe_im, uni_bath_re.t())
    if "id_routed_re" in layer_out:
        id_rt_re, id_rt_im = layer_out["id_routed_re"].float(), layer_out["id_routed_im"].float()
        id_ov = (id_rt_re * probe_re).sum(dim=-1, keepdim=True)
        act_id_re, act_id_im = id_rt_re * id_ov, id_rt_im * id_ov
    else:
        act_id_re, act_id_im = exact_token_id_mixed_state_action(
            layer_out["id_attn_probs"].float(), id_ket_re, probe_re, probe_im
        )

    gates = layer_out["gates"].float()
    w_id, w_deep, w_ngram, w_uni = (
        gates[..., 0:1],
        gates[..., 1:2],
        gates[..., 2:3],
        gates[..., 3:4],
    )
    mix_re, mix_im = model.beam_readout_mix_re.float(), model.beam_readout_mix_im.float()
    beam_mix_re = (bm_src_re * mix_re - bm_src_im * mix_im).sum(dim=-2)
    beam_mix_im = (bm_src_re * mix_im + bm_src_im * mix_re).sum(dim=-2)

    vocab_re, vocab_im = normalize_complex_ket(
        model.token_ket_re.float(), model.token_ket_im.float()
    )
    packed_vocab_t = (
        torch.cat([vocab_re, vocab_im], dim=-1)[:VALID_FINEWEB_VOCAB_SIZE].t().contiguous()
    )
    uni_log_prior = torch.log(uni_probs[:VALID_FINEWEB_VOCAB_SIZE].clamp_min(1e-9)).view(1, 1, -1)
    in_ctx_w = layer_out["in_ctx_copy_weights"].float()
    id_attn_p = layer_out["id_attn_probs"].float()
    copy_index = idx.clamp(0, VALID_FINEWEB_VOCAB_SIZE - 1).unsqueeze(1).expand(bsz, seq_len, seq_len)

    ablation_modes = {
        "full": (1.0, 1.0, 1.0, 1.0, 1.0),
        "no_bath_log_prior": (1.0, 1.0, 1.0, 0.0, 1.0),
        "no_copy_scatter": (1.0, 1.0, 1.0, 1.0, 0.0),
        "no_c3_deep_schur": (1.0, 0.0, 1.0, 1.0, 1.0),
        "no_c2_ngram_mps": (1.0, 1.0, 0.0, 1.0, 1.0),
        "no_c4_token_id": (0.0, 1.0, 1.0, 1.0, 1.0),
    }
    ablation_ce: dict[str, float] = {}
    born_full: Tensor | None = None
    for m_name, (m_id, m_deep, m_ng, m_prior, m_copy) in ablation_modes.items():
        sv_re = (
            m_id * w_id * act_id_re
            + m_deep * w_deep * (act_deep_re + 0.35 * beam_mix_re)
            + m_ng * w_ngram * (act_ng_re + 0.25 * hash_ket_re)
            + (w_uni + DEFAULT_VACUUM_EPS) * act_uni_re
        )
        sv_im = (
            m_id * w_id * act_id_im
            + m_deep * w_deep * (act_deep_im + 0.35 * beam_mix_im)
            + m_ng * w_ngram * (act_ng_im + 0.25 * hash_ket_im)
            + (w_uni + DEFAULT_VACUUM_EPS) * act_uni_im
        )
        rd_re, rd_im = normalize_complex_ket(sv_re + 0.50 * curr_re, sv_im + 0.50 * curr_im)
        born_mat = torch.matmul(torch.cat([rd_re, rd_im], dim=-1), packed_vocab_t)
        if m_name == "full":
            born_full = born_mat
        lg = DEFAULT_BORN_READOUT_SCALE * born_mat + m_prior * (
            DEFAULT_UNIGRAM_PRIOR_WEIGHT + 0.50 * w_uni
        ) * uni_log_prior
        if m_copy > 0.0:
            cm = (
                DEFAULT_IN_CONTEXT_COPY_SCALE * in_ctx_w
                + m_id * DEFAULT_TOKEN_ID_COPY_SCALE * w_id * id_attn_p
            )
            lg = lg.scatter_add(dim=-1, index=copy_index, src=cm)
        lg = DEFAULT_LOGIT_SOFTCAP * torch.tanh(lg / DEFAULT_LOGIT_SOFTCAP)
        ablation_ce[m_name] = float(
            F.cross_entropy(lg.reshape(-1, VALID_FINEWEB_VOCAB_SIZE), targets.reshape(-1)).item()
        )

    assert born_full is not None
    scale_sweep_ce: dict[str, float] = {}
    cm_full = (
        DEFAULT_IN_CONTEXT_COPY_SCALE * in_ctx_w
        + DEFAULT_TOKEN_ID_COPY_SCALE * w_id * id_attn_p
    )
    for beta in READOUT_SCALE_SWEEP_VALUES:
        lg_b = (
            float(beta) * born_full
            + (DEFAULT_UNIGRAM_PRIOR_WEIGHT + 0.50 * w_uni) * uni_log_prior
        )
        lg_b = lg_b.scatter_add(dim=-1, index=copy_index, src=cm_full)
        lg_b = DEFAULT_LOGIT_SOFTCAP * torch.tanh(lg_b / DEFAULT_LOGIT_SOFTCAP)
        scale_sweep_ce[f"beta_{beta:.1f}"] = round(
            float(
                F.cross_entropy(
                    lg_b.reshape(-1, VALID_FINEWEB_VOCAB_SIZE), targets.reshape(-1)
                ).item()
            ),
            5,
        )

    uncollapsed_feats = torch.cat(
        [
            curr_re,
            curr_im,
            act_deep_re,
            act_deep_im,
            bm_src_re.reshape(bsz, seq_len, -1),
            bm_src_im.reshape(bsz, seq_len, -1),
            ngram_re.reshape(bsz, seq_len, -1),
            ngram_im.reshape(bsz, seq_len, -1),
            act_id_re,
        ],
        dim=-1,
    )

    if was_train:
        model.train()

    base_ce = ablation_ce["full"]
    delta_ablations = {
        k: round(v - base_ce, 5) for k, v in ablation_ce.items() if k != "full"
    }
    gate_means = gates.mean(dim=(0, 1)).tolist()
    return {
        "full_ce": round(base_ce, 5),
        "ablation_ce": {k: round(v, 5) for k, v in ablation_ce.items()},
        "leave_one_out_delta_ce": delta_ablations,
        "scale_sweep_ce": scale_sweep_ce,
        "gate_means": {
            "w_id": round(float(gate_means[0]), 5),
            "w_deep": round(float(gate_means[1]), 5),
            "w_ngram": round(float(gate_means[2]), 5),
            "w_uni": round(float(gate_means[3]), 5),
        },
        "attn_telemetry": {
            "mean_eff_horizon_all": round(float(eff_horizon.mean().item()), 3),
            "mean_eff_horizon_long_half": round(
                float(eff_horizon[:, :, long_slice].mean().item()), 3
            ),
            "mean_max_attn_peak_long_half": round(
                float(max_attn_peak[:, :, long_slice].mean().item()), 5
            ),
            "cross_bos_leakage_fraction": round(cross_bos_leak, 5),
        },
        "uncollapsed_feats": uncollapsed_feats.detach(),
        "packed_vocab_t": packed_vocab_t.detach(),
        "uni_log_prior": uni_log_prior.detach(),
        "cm_full": cm_full.detach(),
        "copy_index": copy_index.detach(),
    }


def run_baton_compartment_ablation_audit(
    model: FusedBF16FourCompartmentBatonLM,
    x_eval: Tensor,
    y_eval: Tensor,
    max_batches: int = 4,
) -> dict[str, Any]:
    """Aggregate compartment leave-one-out, scale sweep, attention SNR, & uncollapsed ridge probe on Shard 0."""
    n_seq = min(x_eval.size(0), max_batches * 8)
    dev = next(model.parameters()).device
    bx = x_eval[:n_seq].to(dev)
    by = y_eval[:n_seq].to(dev)

    info = inspect_baton_compartment_and_readout_states(model, bx, by)
    feats = info["uncollapsed_feats"]
    vocab_t = info["packed_vocab_t"]
    uni_lp = info["uni_log_prior"]
    cm_full = info["cm_full"]
    copy_idx = info["copy_index"]

    mid = max(1, n_seq // 2)
    f_tr = feats[:mid].reshape(-1, feats.size(-1))
    y_tr = by[:mid].reshape(-1).clamp(0, VALID_FINEWEB_VOCAB_SIZE - 1)
    target_kets = vocab_t.t()[y_tr]

    gram = torch.matmul(f_tr.t(), f_tr) + RIDGE_PROBE_LAMBDA * torch.eye(
        f_tr.size(1), device=f_tr.device, dtype=f_tr.dtype
    )
    rhs = torch.matmul(f_tr.t(), target_kets)
    w_ridge = torch.linalg.solve(gram, rhs)

    f_te = feats[mid:]
    y_te = by[mid:].clamp(0, VALID_FINEWEB_VOCAB_SIZE - 1)
    pred_kets = F.normalize(torch.matmul(f_te, w_ridge), dim=-1)
    mup_beta = PARSEVAL_MUP_C_STAR * math.sqrt(float(vocab_t.size(0)))
    ridge_lg = mup_beta * torch.matmul(pred_kets, vocab_t) + DEFAULT_UNIGRAM_PRIOR_WEIGHT * uni_lp
    ridge_lg = ridge_lg.scatter_add(dim=-1, index=copy_idx[mid:], src=cm_full[mid:])
    ridge_lg = DEFAULT_LOGIT_SOFTCAP * torch.tanh(ridge_lg / DEFAULT_LOGIT_SOFTCAP)
    ridge_heldout_ce = float(
        F.cross_entropy(ridge_lg.reshape(-1, VALID_FINEWEB_VOCAB_SIZE), y_te.reshape(-1)).item()
    )

    return {
        "full_ce": info["full_ce"],
        "leave_one_out_delta_ce": info["leave_one_out_delta_ce"],
        "scale_sweep_ce": info["scale_sweep_ce"],
        "best_scale_sweep_ce": min(info["scale_sweep_ce"].values()),
        "uncollapsed_ridge_mup_heldout_ce": round(ridge_heldout_ce, 5),
        "uncollapsed_feature_dim": int(feats.size(-1)),
        "gate_means": info["gate_means"],
        "attn_telemetry": info["attn_telemetry"],
    }


def run_memorization_and_readout_ceiling_probe(
    kinds: list[KnownArchitectureKind],
    cfg: SyntheticProbeConfig,
    device: torch.device,
    record_counts: tuple[int, ...] = (1, 4),
) -> list[dict[str, Any]]:
    """Run Karpathy `N=1, 4` real FineWeb sequence memorization ceiling across architectures (`<= 8` args)."""
    seq_len = 256
    max_rec = max(record_counts)
    raw = load_fineweb_token_slice("fineweb_val_000000.bin", num_tokens=max_rec * seq_len + 1)
    results: list[dict[str, Any]] = []

    for kind in kinds:
        for n_rec in record_counts:
            adapter = build_known_architecture_adapter(kind, device=device, seed=cfg.seed)
            sub = raw[: n_rec * seq_len + 1]
            xb = sub[:-1].view(n_rec, seq_len).to(device)
            yb = sub[1:].view(n_rec, seq_len).to(device)
            t0 = time.perf_counter()
            for step in range(1, cfg.overfit_steps + 1):
                adapter.train_step(xb, yb, step=step, total_steps=cfg.overfit_steps)

            with torch.no_grad():
                log_p, span = adapter.predict_log_probs_and_span(xb)
                probs = torch.exp(log_p)
                nll = -log_p.gather(dim=-1, index=yb.unsqueeze(-1)).squeeze(-1)
                preds = log_p.argmax(dim=-1)
                ent = -(probs * log_p.clamp_min(LOG_PROB_FLOOR)).sum(dim=-1)
                max_p = probs.amax(dim=-1)

            results.append(
                {
                    "arch": kind.value,
                    "label": adapter.label,
                    "num_records": int(n_rec),
                    "total_tokens": int(n_rec * seq_len),
                    "final_memorization_ce": round(float(nll.mean().item()), 5),
                    "memorization_top1_acc": round(float((preds == yb).float().mean().item()), 5),
                    "mean_max_prob": round(float(max_p.mean().item()), 5),
                    "mean_pred_entropy_nats": round(float(ent.mean().item()), 5),
                    "mean_logit_span": round(float(span.mean().item()), 4),
                    "elapsed_sec": round(time.perf_counter() - t0, 2),
                }
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return results
