#!/usr/bin/env python3
"""Wave 39 (`EXPERIMENT_PLAN_054.md`): Multiples-of-32 `d_complex` 5-Minute Tournament & 900M-Token Launch.

Evaluates:
1. `H1`: Masked non-power-of-2 Triton `fused_complex_norm_rope_triton` ($D \\in \\{16..256\\}$) and
   Scale-Invariant Krylov Normalization (`probe_mode="v_krylov_norm"`) parity (`< 1e-5` fp32 error).
2. `H2`: Throughput (`tok/s`), VRAM (`MiB`), and 5-minute token budgets across multiples of 32
   ($d_{\\text{complex}} \\in \\{32, 64, 96, 128, 160, 192\\}$) via `.cache/complex_krylov_small_d_sweep.json`.
3. `H3`: 5-minute (`300s` pure training wall-clock) tournament on `FineWeb-Edu` across multiples of 32,
   where each candidate runs at its native hardware speed (`tok/s`) with its Smith one-cycle cosine
   schedule calibrated to $\\tau = t_{\\text{train}} / 300\\text{s} \\in [0, 1]$.
4. `H4`: 900M-token (`109,863` steps at `8,192` tok/step across 9 `FineWeb` shards) production run
   of the 5-minute tournament winner, streaming live telemetry to the WebUI (`:8765`).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for p_str in (str(ROOT / "src"), str(ROOT)):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from qdlblocks.benchmarks.experiment_logger import (  # noqa: E402
    ExperimentIntentSpec,
    StandardizedExperimentLogger,
)
from qdlblocks.benchmarks.fineweb_stream import FineWebMultiShardStream  # noqa: E402
from qdlblocks.kernels.flash_complex_krylov_triton import (  # noqa: E402
    flash_complex_krylov_attention,
    fused_complex_norm_rope_triton,
)
from qdlblocks.layers.packed_complex_register_density_attn import (  # noqa: E402
    ComplexRoPE,
    complex_l2_normalize,
    krylov_purified_density_probe,
    pack_hermitian_outer_product_and_ket,
    unpack_hermitian_matrix,
)
from qdlblocks.models.gemini_packed_complex_rope_gpt import (  # noqa: E402
    GeminiPackedComplexRoPEConfig,
    GeminiPackedComplexRoPEGPT,
    build_gemini_complex_optimizers,
    initialize_gemini_complex_gpt_with_ppmi_svd,
)

_EX48_SPEC = importlib.util.spec_from_file_location(
    "plan038_runner", ROOT / "examples" / "48_verify_plan038_last_layer_measurement_and_traceless.py"
)
assert _EX48_SPEC is not None and _EX48_SPEC.loader is not None
_EX48 = importlib.util.module_from_spec(_EX48_SPEC)
sys.modules["plan038_runner"] = _EX48
_EX48_SPEC.loader.exec_module(_EX48)

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95, device=0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Documented in EXPERIMENT_PLAN_054.md (Wave 39)
# ==============================================================================
DEFAULT_WALL_CLOCK_SEC: float = 300.0
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_SEQ_LEN: int = 512
DEFAULT_NUM_LAYERS: int = 6
VALID_VOCAB_SIZE: int = 50257
INIT_PRIOR_TOKENS: int = 2_000_000
BASE_MUON_ETA_MAX: float = 0.015
TARGET_900M_TOKENS: float = 900.0
WAVE38_D128_300STEP_CE: float = 6.2951
WAVE38_MAX5060TI_300STEP_CE: float = 6.2452
WEBUI_RELOAD_URL: str = "http://127.0.0.1:8765/api/reload"


@dataclass(frozen=True)
class SmallDTournamentCandidate:
    """Configuration for a `d_complex` multiple-of-32 tournament arm (`<= 8` fields)."""

    d_complex: int
    n_registers: int
    d_register: int
    num_layers: int
    batch_size: int
    est_steps_5min: int
    label: str


def _ping_webui_reload(url: str = WEBUI_RELOAD_URL) -> None:
    """Trigger non-blocking reload on the local `:8765` WebUI daemon."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=1.0):
            pass
    except Exception:
        pass


def _resolve_candidate_spec(d_c: int, sizer_rows: dict[int, dict[str, Any]]) -> SmallDTournamentCandidate:
    """Build `SmallDTournamentCandidate` for `d_complex = d_c` using measured sizer throughput if available."""
    if d_c == 32:
        h_num, d_reg = 2, 16
    else:
        d_reg = 32
        h_num = d_c // 32
    row = sizer_rows.get(d_c, {})
    est_steps = int(row.get("est_5min_steps", max(400, int(round(300.0 * 25000.0 / 8192.0)))))
    return SmallDTournamentCandidate(
        d_complex=d_c,
        n_registers=h_num,
        d_register=d_reg,
        num_layers=DEFAULT_NUM_LAYERS,
        batch_size=DEFAULT_BATCH_SIZE,
        est_steps_5min=est_steps,
        label=f"d_complex={d_c} (H={h_num}, d_h={d_reg}, d_real={2 * d_c}, L={DEFAULT_NUM_LAYERS})",
    )


def run_h1_nonpow2_and_scale_invariant_krylov_audit(device: torch.device) -> dict[str, Any]:
    """Verify `H1` masked Triton Norm+RoPE across $D \\in \\{16..256\\}$ and scale-invariant Krylov parity."""
    torch.manual_seed(5401)
    max_norm_err = 0.0
    for d_dim in (16, 24, 32, 64, 96, 128, 160, 192, 256):
        z_re = torch.randn(2, 3, 32, d_dim, device=device, dtype=torch.float32)
        z_im = torch.randn(2, 3, 32, d_dim, device=device, dtype=torch.float32)
        rope = ComplexRoPE(d_register=d_dim).to(device)
        cos_2d, sin_2d = rope.get_cos_sin(32, device, torch.float32)
        tr_re, tr_im = fused_complex_norm_rope_triton(z_re, z_im, cos_2d, sin_2d, eps=1e-7)
        n_sq = z_re.square().sum(dim=-1, keepdim=True) + z_im.square().sum(dim=-1, keepdim=True)
        inv_n = torch.rsqrt(n_sq + 1e-7)
        nr, ni = z_re * inv_n, z_im * inv_n
        c, s = cos_2d.view(1, 1, 32, d_dim), sin_2d.view(1, 1, 32, d_dim)
        ref_re, ref_im = nr * c - ni * s, nr * s + ni * c
        err = max(float((tr_re - ref_re).abs().max().item()), float((tr_im - ref_im).abs().max().item()))
        max_norm_err = max(max_norm_err, err)

    max_krylov_err = 0.0
    min_grad_cos = 1.0
    for dh, h_num in ((16, 2), (32, 3), (32, 5)):
        seq_t = 48
        rope = ComplexRoPE(d_register=dh).to(device)
        cos_2d, sin_2d = rope.get_cos_sin(seq_t, device, torch.float32)
        q_re, q_im = fused_complex_norm_rope_triton(
            torch.randn(2, h_num, seq_t, dh, device=device),
            torch.randn(2, h_num, seq_t, dh, device=device),
            cos_2d,
            sin_2d,
        )
        k_re, k_im = fused_complex_norm_rope_triton(
            torch.randn(2, h_num, seq_t, dh, device=device),
            torch.randn(2, h_num, seq_t, dh, device=device),
            cos_2d,
            sin_2d,
        )
        v_re, v_im = fused_complex_norm_rope_triton(
            torch.randn(2, h_num, seq_t, dh, device=device),
            torch.randn(2, h_num, seq_t, dh, device=device),
            None,
            None,
        )
        qr1, qi1 = q_re.detach().clone().requires_grad_(True), q_im.detach().clone().requires_grad_(True)
        kr1, ki1 = k_re.detach().clone().requires_grad_(True), k_im.detach().clone().requires_grad_(True)
        vr1, vi1 = v_re.detach().clone().requires_grad_(True), v_im.detach().clone().requires_grad_(True)
        qr2, qi2 = q_re.detach().clone().requires_grad_(True), q_im.detach().clone().requires_grad_(True)
        kr2, ki2 = k_re.detach().clone().requires_grad_(True), k_im.detach().clone().requires_grad_(True)
        vr2, vi2 = v_re.detach().clone().requires_grad_(True), v_im.detach().clone().requires_grad_(True)

        u_re_fast, u_im_fast, _, _ = flash_complex_krylov_attention(
            qr1, qi1, kr1, ki1, vr1, vi1, probe_mode="v_krylov_norm", eps=1e-7
        )
        c_re = torch.matmul(qr2, kr2.transpose(-2, -1)) + torch.matmul(qi2, ki2.transpose(-2, -1))
        c_im = torch.matmul(qr2, ki2.transpose(-2, -1)) - torch.matmul(qi2, kr2.transpose(-2, -1))
        idx_t = torch.arange(seq_t, device=device)
        causal = (idx_t[:, None] >= idx_t[None, :]).view(1, 1, seq_t, seq_t)
        u_int = (c_re.square() + c_im.square()).square().masked_fill(~causal, 0.0)
        w = u_int / (u_int.sum(dim=-1, keepdim=True) + 1e-7)
        v_pk = pack_hermitian_outer_product_and_ket(vr2, vi2)
        m_all = torch.matmul(w, v_pk)
        d_sq = dh * dh
        r_re, r_im = unpack_hermitian_matrix(m_all[..., :d_sq], dh)
        pr_re, pr_im = complex_l2_normalize(m_all[..., d_sq : d_sq + dh], m_all[..., d_sq + dh : d_sq + 2 * dh], dim=-1)
        ur_raw, ui_raw = krylov_purified_density_probe(r_re, r_im, pr_re, pr_im, eps=1e-7)
        n_sq = ur_raw.square().sum(dim=-1, keepdim=True) + ui_raw.square().sum(dim=-1, keepdim=True)
        inv_n = torch.rsqrt(n_sq + 1e-7)
        u_re_ref, u_im_ref = ur_raw * inv_n, ui_raw * inv_n

        err = max(float((u_re_fast - u_re_ref).abs().max().item()), float((u_im_fast - u_im_ref).abs().max().item()))
        max_krylov_err = max(max_krylov_err, err)
        up_re, up_im = torch.randn_like(u_re_fast), torch.randn_like(u_im_fast)
        (u_re_fast * up_re + u_im_fast * up_im).sum().backward()
        (u_re_ref * up_re + u_im_ref * up_im).sum().backward()
        for g1, g2 in ((qr1.grad, qr2.grad), (kr1.grad, kr2.grad), (vr1.grad, vr2.grad)):
            assert g1 is not None and g2 is not None
            cos = float(F.cosine_similarity(g1.reshape(1, -1), g2.reshape(1, -1)).item())
            min_grad_cos = min(min_grad_cos, cos)

    h1_pass = (max_norm_err < 1e-6) and (max_krylov_err < 1e-5) and (min_grad_cos > 0.999)
    return {
        "h1_pass": h1_pass,
        "max_nonpow2_norm_rope_err_fp32": max_norm_err,
        "max_fast_krylov_unit_err_fp32": max_krylov_err,
        "min_fast_krylov_grad_cosine_fp32": round(min_grad_cos, 6),
    }


@torch.no_grad()
def evaluate_complex_model(
    model: GeminiPackedComplexRoPEGPT,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    """Evaluate standalone `GeminiPackedComplexRoPEGPT` on held-out `FineWeb` validation batches."""
    model.eval()
    v_valid = int(model.cfg.valid_vocab_size)
    total_loss = 0.0
    total_top1 = 0
    total_top50 = 0
    total_tokens = 0
    dev_type = val_batches[0][0].device.type

    for x_val, y_val in val_batches:
        with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=(dev_type == "cuda")):
            logits, _ = model(x_val, targets=None, return_logits=True)
        assert logits is not None
        logits_f = logits[..., :v_valid].float()
        y_clamped = y_val.clamp(min=0, max=v_valid - 1)
        loss = F.cross_entropy(logits_f.reshape(-1, v_valid), y_clamped.reshape(-1), reduction="sum")
        total_loss += float(loss.item())
        preds = logits_f.argmax(dim=-1)
        total_top1 += int((preds == y_clamped).sum().item())
        top50 = logits_f.topk(k=min(50, v_valid), dim=-1).indices
        total_top50 += int((top50 == y_clamped.unsqueeze(-1)).any(dim=-1).sum().item())
        total_tokens += int(y_val.numel())

    val_ce = total_loss / max(1, total_tokens)
    val_ppl = math.exp(min(val_ce, 20.0))
    top1_pct = round(100.0 * total_top1 / max(1, total_tokens), 2)
    top50_pct = round(100.0 * total_top50 / max(1, total_tokens), 2)
    purity_vals = [float(blk.attn.last_purity_mean) for blk in model.blocks]
    probe_norms = [float(blk.attn.last_probe_norm_mean) for blk in model.blocks]
    return {
        "val_ce": round(val_ce, 4),
        "val_ppl": round(val_ppl, 2),
        "val_top1": top1_pct,
        "top1_acc": top1_pct,
        "top50_acc": top50_pct,
        "sharp_eff_rank": round(model.compute_sharp_eff_rank(), 2),
        "mean_density_purity": round(sum(purity_vals) / max(1, len(purity_vals)), 4),
        "mean_raw_probe_norm": round(sum(probe_norms) / max(1, len(probe_norms)), 4),
    }


def _apply_progress_lr(
    opt_muon: torch.optim.Optimizer,
    opt_adam: torch.optim.Optimizer,
    tau: float,
    min_lr_ratio: float = 0.20,
    warmup_frac: float = 0.15,
) -> float:
    """Apply Smith one-cycle warmup + cosine decay envelope for normalized progress `tau in [0, 1]`."""
    progress = max(0.0, min(1.0, float(tau)))
    w_f = max(1e-6, min(0.50, float(warmup_frac)))
    if progress < w_f:
        env = 0.25 + 0.75 * (progress / w_f)
    else:
        t_dec = (progress - w_f) / max(1e-6, 1.0 - w_f)
        env = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * t_dec))
    for g in opt_muon.param_groups:
        g["lr"] = float(g.get("base_lr", BASE_MUON_ETA_MAX)) * env
    for g in opt_adam.param_groups:
        g["lr"] = float(g.get("base_lr", 0.005)) * env
    return float(opt_muon.param_groups[0]["lr"])


def run_wall_clock_tournament_arm(
    cand: SmallDTournamentCandidate,
    work_gpu: torch.Tensor,
    prior_tokens: torch.Tensor,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    ppmi_vecs: torch.Tensor,
    wall_clock_sec: float,
    device: torch.device,
) -> dict[str, Any]:
    """Train a `SmallDTournamentCandidate` for `wall_clock_sec` pure training seconds (`<= 8` args)."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(1337)
    cfg = GeminiPackedComplexRoPEConfig(
        vocab_size=50304,
        valid_vocab_size=VALID_VOCAB_SIZE,
        d_complex=cand.d_complex,
        n_registers=cand.n_registers,
        d_register=cand.d_register,
        num_layers=cand.num_layers,
        block_size=DEFAULT_SEQ_LEN,
        probe_mode="v_krylov",
        normalize_krylov_ket=True,
        checkpoint_blocks=False,
    )
    model = GeminiPackedComplexRoPEGPT(cfg).to(device)
    init_stats = initialize_gemini_complex_gpt_with_ppmi_svd(model, ppmi_vecs, prior_tokens)
    opt_muon, opt_adam = build_gemini_complex_optimizers(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = round(trainable_params / 1e6, 2)
    tokens_per_step = cand.batch_size * DEFAULT_SEQ_LEN
    sec_tag = int(round(wall_clock_sec))
    run_key = f"w39_5min_tourney_d{cand.d_complex}c_H{cand.n_registers}_dh{cand.d_register}_L{cand.num_layers}_{sec_tag}s"
    est_tokens_m = round((cand.est_steps_5min * tokens_per_step * (wall_clock_sec / 300.0)) / 1e6, 2)

    intent = ExperimentIntentSpec(
        run_key=run_key,
        wave_id="W39",
        plan_id="EXPERIMENT_PLAN_054",
        task_domain="fineweb_edu_local",
        hypothesis_intent=(
            f"Wave 39 {sec_tag}s wall-clock tournament arm at d_complex={cand.d_complex} "
            f"(H={cand.n_registers}, d_h={cand.d_register}, d_real={2 * cand.d_complex}, L={cand.num_layers})."
        ),
        architecture_summary=(
            f"GeminiPackedComplexRoPEGPT ({cand.label}, {params_m:.2f}M params, Scale-Invariant Krylov)"
        ),
        hardware_budget="1x RTX 5060 Ti 16GB (Masked Triton Norm+RoPE + Scale-Invariant Flash-Krylov)",
        batch_size=cand.batch_size,
        seq_len=DEFAULT_SEQ_LEN,
        total_steps=cand.est_steps_5min,
        target_tokens_M=est_tokens_m,
        params_M=params_m,
        embed_dim=2 * cand.d_complex,
        num_layers=cand.num_layers,
        schedule="smith_one_cycle_wallclock",
    )
    logger = StandardizedExperimentLogger(intent)
    _ping_webui_reload()

    eval_0 = evaluate_complex_model(model, val_batches)
    eval_0.update(
        {
            "step": 0,
            "tokens_M": 0.0,
            "train_ce": eval_0["val_ce"],
            "compute_tflops": 0.0,
            "tau_progress": 0.0,
            "elapsed_min": 0.0,
            "tok_per_s": 0.0,
            "peak_vram_mib": 0.0,
        }
    )
    logger.log_eval(0, eval_0)
    print(
        f"[{run_key}] Step 0 | Params={params_m:.2f}M | Val CE={eval_0['val_ce']:.4f} "
        f"| Top1={eval_0['top1_acc']:.2f}% | R_sharp={eval_0['sharp_eff_rank']:.1f}",
        flush=True,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    n_work = int(work_gpu.numel())
    train_wall_s = 0.0
    step = 0
    next_eval_frac = 0.25
    last_train_ce = eval_0["val_ce"]
    last_stiefel_err = 0.0
    peak_train_vram_mib = 0.0
    history: list[dict[str, Any]] = [dict(eval_0)]

    while train_wall_s < wall_clock_sec:
        step += 1
        t_step_0 = time.perf_counter()
        model.train()
        tau = min(1.0, train_wall_s / max(1e-6, wall_clock_sec))
        model.set_training_progress(tau)
        cur_lr = _apply_progress_lr(opt_muon, opt_adam, tau, min_lr_ratio=0.20)
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)

        offset = ((step - 1) * tokens_per_step) % max(1, n_work - tokens_per_step - 2)
        c_i64 = work_gpu[offset : offset + tokens_per_step + 1].to(torch.int64) & 0xFFFF
        x_b = c_i64[:-1].view(cand.batch_size, DEFAULT_SEQ_LEN)
        y_b = c_i64[1:].view(cand.batch_size, DEFAULT_SEQ_LEN)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, loss_t = model(x_b, targets=y_b, return_logits=False)
        assert loss_t is not None
        loss_t.backward()
        last_train_ce = float(loss_t.detach().item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_muon.step()
        opt_adam.step()
        if step % 2 == 0:
            last_stiefel_err = model.retract_complex_unitary_weights(rate=0.05)

        if step % 20 == 0 and device.type == "cuda":
            torch.cuda.synchronize(device)
            cur_mib = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
            peak_train_vram_mib = max(peak_train_vram_mib, cur_mib)

        train_wall_s += time.perf_counter() - t_step_0
        tok_seen = step * tokens_per_step
        tok_s = tok_seen / max(1e-6, train_wall_s)
        tok_m = tok_seen / 1e6
        frac_done = min(1.0, train_wall_s / max(1e-6, wall_clock_sec))

        if step == 1 or step % 50 == 0 or frac_done >= next_eval_frac or train_wall_s >= wall_clock_sec:
            logger.log_step(step, last_train_ce, tok_s, cur_lr / BASE_MUON_ETA_MAX, train_wall_s / 60.0)

        if frac_done >= next_eval_frac or train_wall_s >= wall_clock_sec:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                cur_mib = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
                peak_train_vram_mib = max(peak_train_vram_mib, cur_mib)
            ev = evaluate_complex_model(model, val_batches)
            ev.update(
                {
                    "step": step,
                    "tokens_M": round(tok_m, 2),
                    "train_ce": round(last_train_ce, 4),
                    "compute_tflops": round(6.0 * params_m * tok_m, 4),
                    "tau_progress": round(frac_done, 4),
                    "elapsed_min": round(train_wall_s / 60.0, 4),
                    "tok_per_s": round(tok_s, 1),
                    "peak_vram_mib": round(peak_train_vram_mib, 1),
                    "kraus_stiefel_err": round(last_stiefel_err, 6),
                }
            )
            history.append(dict(ev))
            logger.log_eval(step, ev)
            print(
                f"[{run_key}] t={train_wall_s:5.1f}s/{wall_clock_sec:.0f}s | Step {step:4d} ({tok_m:5.2f}M tok) "
                f"| Train CE={last_train_ce:.4f} | Val CE={ev['val_ce']:.4f} | PPL={ev['val_ppl']:.2f} "
                f"| Top1={ev['top1_acc']:.2f}% | VRAM={peak_train_vram_mib:.0f}MiB | {tok_s:,.0f} tok/s",
                flush=True,
            )
            _ping_webui_reload()
            while next_eval_frac <= frac_done:
                next_eval_frac += 0.25

    final_ev = history[-1]
    actual_tokens_m = round((step * tokens_per_step) / 1e6, 2)
    summary = {
        "run_key": run_key,
        "label": cand.label,
        "d_complex": cand.d_complex,
        "real_equiv_dim": 2 * cand.d_complex,
        "n_registers": cand.n_registers,
        "d_register": cand.d_register,
        "num_layers": cand.num_layers,
        "batch_size": cand.batch_size,
        "params_M": params_m,
        "steps": step,
        "tokens_M": actual_tokens_m,
        "target_tokens_M": actual_tokens_m,
        "step0_val_ce": eval_0["val_ce"],
        "final_train_ce": round(last_train_ce, 4),
        "final_val_ce": final_ev["val_ce"],
        "final_val_ppl": final_ev["val_ppl"],
        "final_top1_acc": final_ev["top1_acc"],
        "final_top50_acc": final_ev["top50_acc"],
        "final_sharp_eff_rank": final_ev["sharp_eff_rank"],
        "final_mean_density_purity": final_ev["mean_density_purity"],
        "final_mean_raw_probe_norm": final_ev["mean_raw_probe_norm"],
        "tok_s": round((step * tokens_per_step) / max(1e-6, train_wall_s), 1),
        "peak_vram_mib": round(peak_train_vram_mib, 1),
        "wall_clock_s": round(train_wall_s, 2),
        "init_stats": init_stats,
        "history": history,
    }
    logger.finalize(extra_summary=summary)
    _ping_webui_reload()
    del model, opt_muon, opt_adam
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def run_900m_winner_production(
    cand: SmallDTournamentCandidate,
    prior_tokens: torch.Tensor,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    ppmi_vecs: torch.Tensor,
    target_tokens_m: float,
    device: torch.device,
    out_json_path: Path,
) -> dict[str, Any]:
    """Train the winning `SmallDTournamentCandidate` for `target_tokens_m` (`900M` tokens) across 9 shards (`<= 8` args)."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(1337)
    cfg = GeminiPackedComplexRoPEConfig(
        vocab_size=50304,
        valid_vocab_size=VALID_VOCAB_SIZE,
        d_complex=cand.d_complex,
        n_registers=cand.n_registers,
        d_register=cand.d_register,
        num_layers=cand.num_layers,
        block_size=DEFAULT_SEQ_LEN,
        probe_mode="v_krylov",
        normalize_krylov_ket=True,
        checkpoint_blocks=False,
    )
    model = GeminiPackedComplexRoPEGPT(cfg).to(device)
    init_stats = initialize_gemini_complex_gpt_with_ppmi_svd(model, ppmi_vecs, prior_tokens)
    opt_muon, opt_adam = build_gemini_complex_optimizers(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = round(trainable_params / 1e6, 2)
    tokens_per_step = cand.batch_size * DEFAULT_SEQ_LEN
    total_steps = max(1, int(round((target_tokens_m * 1e6) / tokens_per_step)))
    tok_tag = int(round(target_tokens_m))
    run_key = f"w39_winner_d{cand.d_complex}c_H{cand.n_registers}_dh{cand.d_register}_L{cand.num_layers}_{tok_tag}Mtok"

    intent = ExperimentIntentSpec(
        run_key=run_key,
        wave_id="W39",
        plan_id="EXPERIMENT_PLAN_054",
        task_domain="fineweb_edu_local",
        hypothesis_intent=(
            f"Wave 39 900M-token long-horizon run of 5-minute tournament winner d_complex={cand.d_complex} "
            f"(H={cand.n_registers}, d_h={cand.d_register}, d_real={2 * cand.d_complex}, L={cand.num_layers})."
        ),
        architecture_summary=(
            f"GeminiPackedComplexRoPEGPT ({cand.label}, {params_m:.2f}M params, 900M-Token FineWeb Stream)"
        ),
        hardware_budget="1x RTX 5060 Ti 16GB (Masked Triton Norm+RoPE + Scale-Invariant Flash-Krylov)",
        batch_size=cand.batch_size,
        seq_len=DEFAULT_SEQ_LEN,
        total_steps=total_steps,
        target_tokens_M=round(target_tokens_m, 2),
        params_M=params_m,
        embed_dim=2 * cand.d_complex,
        num_layers=cand.num_layers,
        schedule="smith_one_cycle_900M",
    )
    logger = StandardizedExperimentLogger(intent)
    _ping_webui_reload()

    eval_0 = evaluate_complex_model(model, val_batches)
    eval_0.update(
        {
            "step": 0,
            "tokens_M": 0.0,
            "train_ce": eval_0["val_ce"],
            "compute_tflops": 0.0,
            "tau_progress": 0.0,
            "elapsed_min": 0.0,
            "tok_per_s": 0.0,
            "peak_vram_mib": 0.0,
        }
    )
    logger.log_eval(0, eval_0)
    print(
        f"[{run_key}] Step 0/{total_steps} (0.00M/{target_tokens_m:.0f}M tok) | Params={params_m:.2f}M "
        f"| Val CE={eval_0['val_ce']:.4f} | Top1={eval_0['top1_acc']:.2f}%",
        flush=True,
    )

    num_shards = max(1, min(9, int(math.ceil(target_tokens_m / 100.0))))
    stream = FineWebMultiShardStream(
        data_dir=ROOT / ".cache" / "fineweb_100M",
        num_shards=num_shards,
        prefetch_async=True,
    )
    stream.prefetch_shard(1)
    shard_arr, _ = stream.get_shard(1)
    shard_gpu = torch.from_numpy(np.array(shard_arr, dtype=np.uint16, copy=True).view(np.int16)).to(device)
    current_shard_idx = 1

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    last_train_ce = eval_0["val_ce"]
    last_stiefel_err = 0.0
    peak_train_vram_mib = 0.0
    history: list[dict[str, Any]] = [dict(eval_0)]
    early_evals = {250, 500, 1000}
    warmup_frac_900m = min(0.15, 600.0 / float(max(1, total_steps)))

    for step in range(1, total_steps + 1):
        # Advance to next 100M-token shard every ~12,207 steps (100M tokens)
        desired_shard = min(num_shards, 1 + ((step - 1) * tokens_per_step) // 100_000_000)
        if desired_shard != current_shard_idx:
            try:
                next_arr, wait_ms = stream.get_shard(desired_shard)
                del shard_gpu
                shard_gpu = torch.from_numpy(np.array(next_arr, dtype=np.uint16, copy=True).view(np.int16)).to(device)
                current_shard_idx = desired_shard
                print(
                    f"[{run_key}] Switched to FineWeb Shard {current_shard_idx}/{num_shards} "
                    f"(prefetch wait={wait_ms:.1f}ms)",
                    flush=True,
                )
            except Exception as exc:
                print(f"[{run_key}] Shard {desired_shard} load fallback to shard {current_shard_idx}: {exc}", flush=True)
                current_shard_idx = desired_shard

        model.train()
        tau = float(step - 1) / float(max(1, total_steps))
        model.set_training_progress(tau)
        cur_lr = _apply_progress_lr(opt_muon, opt_adam, tau, min_lr_ratio=0.12, warmup_frac=warmup_frac_900m)
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)

        n_shard = int(shard_gpu.numel())
        offset = ((step - 1) * tokens_per_step) % max(1, n_shard - tokens_per_step - 2)
        c_i64 = shard_gpu[offset : offset + tokens_per_step + 1].to(torch.int64) & 0xFFFF
        x_b = c_i64[:-1].view(cand.batch_size, DEFAULT_SEQ_LEN)
        y_b = c_i64[1:].view(cand.batch_size, DEFAULT_SEQ_LEN)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, loss_t = model(x_b, targets=y_b, return_logits=False)
        assert loss_t is not None
        loss_t.backward()
        last_train_ce = float(loss_t.detach().item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_muon.step()
        opt_adam.step()
        if step % 2 == 0 or step == total_steps:
            last_stiefel_err = model.retract_complex_unitary_weights(rate=0.05)

        elapsed = max(1e-6, time.perf_counter() - t0)
        tok_seen = step * tokens_per_step
        tok_s = tok_seen / elapsed
        tok_m = tok_seen / 1e6

        is_eval_step = (step in early_evals) or (step % 2000 == 0) or (step == total_steps)
        if step == 1 or step % 100 == 0 or is_eval_step:
            logger.log_step(step, last_train_ce, tok_s, cur_lr / BASE_MUON_ETA_MAX, elapsed / 60.0)

        if is_eval_step:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                cur_mib = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
                peak_train_vram_mib = max(peak_train_vram_mib, cur_mib)
            ev = evaluate_complex_model(model, val_batches)
            tau_eval = min(1.0, float(step) / float(max(1, total_steps)))
            ev.update(
                {
                    "step": step,
                    "tokens_M": round(tok_m, 2),
                    "train_ce": round(last_train_ce, 4),
                    "compute_tflops": round(6.0 * params_m * tok_m, 4),
                    "tau_progress": round(tau_eval, 4),
                    "elapsed_min": round(elapsed / 60.0, 4),
                    "tok_per_s": round(tok_s, 1),
                    "peak_vram_mib": round(peak_train_vram_mib, 1),
                    "kraus_stiefel_err": round(last_stiefel_err, 6),
                }
            )
            history.append(dict(ev))
            logger.log_eval(step, ev)
            eta_min = ((total_steps - step) * tokens_per_step / max(1.0, tok_s)) / 60.0
            print(
                f"[{run_key}] Step {step:6d}/{total_steps} ({tok_m:6.2f}M/{target_tokens_m:.0f}M tok) "
                f"| Train CE={last_train_ce:.4f} | Val CE={ev['val_ce']:.4f} | PPL={ev['val_ppl']:.2f} "
                f"| Top1={ev['top1_acc']:.2f}% | {tok_s:,.0f} tok/s | ETA={eta_min:.1f}m",
                flush=True,
            )
            _ping_webui_reload()

    stream.close()
    total_wall = time.perf_counter() - t0
    final_ev = history[-1]
    summary = {
        "run_key": run_key,
        "label": f"900M-Token Winner: {cand.label}",
        "d_complex": cand.d_complex,
        "real_equiv_dim": 2 * cand.d_complex,
        "n_registers": cand.n_registers,
        "d_register": cand.d_register,
        "num_layers": cand.num_layers,
        "batch_size": cand.batch_size,
        "params_M": params_m,
        "steps": total_steps,
        "tokens_M": round(target_tokens_m, 2),
        "target_tokens_M": round(target_tokens_m, 2),
        "step0_val_ce": eval_0["val_ce"],
        "final_train_ce": round(last_train_ce, 4),
        "final_val_ce": final_ev["val_ce"],
        "final_val_ppl": final_ev["val_ppl"],
        "final_top1_acc": final_ev["top1_acc"],
        "final_top50_acc": final_ev["top50_acc"],
        "final_sharp_eff_rank": final_ev["sharp_eff_rank"],
        "final_mean_density_purity": final_ev["mean_density_purity"],
        "final_mean_raw_probe_norm": final_ev["mean_raw_probe_norm"],
        "tok_s": round((total_steps * tokens_per_step) / max(1e-6, total_wall), 1),
        "peak_vram_mib": round(peak_train_vram_mib, 1),
        "wall_clock_s": round(total_wall, 2),
        "init_stats": init_stats,
        "history": history,
    }
    logger.finalize(extra_summary=summary)
    _ping_webui_reload()
    if out_json_path.exists():
        try:
            existing = json.loads(out_json_path.read_text(encoding="utf-8"))
            existing["run_900m_summary"] = summary
            out_json_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except Exception:
            pass
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave 39 Small-d_complex 5-Min Tournament & 900M-Token Runner")
    parser.add_argument("--stage", type=str, choices=("tournament", "run_900m", "both"), default="tournament")
    parser.add_argument("--wall-clock-sec", type=float, default=DEFAULT_WALL_CLOCK_SEC)
    parser.add_argument("--candidates", type=str, default="32,64,96,128")
    parser.add_argument("--d-complex", type=int, default=96)
    parser.add_argument("--target-tokens-m", type=float, default=TARGET_900M_TOKENS)
    parser.add_argument(
        "--sizer-json",
        type=str,
        default=str(ROOT / ".cache" / "complex_krylov_small_d_sweep.json"),
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / ".cache" / "hypothesis_experiments_054.json"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sizer_path = Path(args.sizer_json)
    sizer_rows: dict[int, dict[str, Any]] = {}
    if sizer_path.exists():
        sizer_data = json.loads(sizer_path.read_text(encoding="utf-8"))
        for row in sizer_data.get("all_tiers", []):
            sizer_rows[int(row["d_complex"])] = row

    train_arr, val_arr = _EX48.load_fineweb_tokens(max_train_tokens=30_000_000, max_val_tokens=1_048_576)
    ppmi_obj = torch.load(
        ROOT / ".cache" / "ppmi_svd_256_fast_synth_plan030.pt", map_location="cpu", weights_only=True
    )
    ppmi_vecs = ppmi_obj["embeddings"] if isinstance(ppmi_obj, dict) else ppmi_obj
    val_batches = _EX48.build_evaluation_batches(
        val_arr, batch_size=4, seq_len=DEFAULT_SEQ_LEN, num_batches=6, device=device
    )
    prior_tokens = torch.from_numpy(train_arr[:INIT_PRIOR_TOKENS].astype("int64"))
    out_path = Path(args.out_json)

    winner_d_complex = int(args.d_complex)
    if args.stage in ("tournament", "both"):
        print(f"[Wave 39] Running H1 non-pow2 Triton & Scale-Invariant Krylov audit on {device}...", flush=True)
        h1_report = run_h1_nonpow2_and_scale_invariant_krylov_audit(device)
        print(f"[Wave 39] H1 Audit: {json.dumps(h1_report, indent=2)}", flush=True)

        work_gpu = torch.from_numpy(train_arr[:30_000_000].view("int16")).to(device)
        cand_dims = [int(x.strip()) for x in args.candidates.split(",") if x.strip()]
        tournament_runs: list[dict[str, Any]] = []
        for d_c in cand_dims:
            cand_spec = _resolve_candidate_spec(d_c, sizer_rows)
            res = run_wall_clock_tournament_arm(
                cand_spec, work_gpu, prior_tokens, val_batches, ppmi_vecs, float(args.wall_clock_sec), device
            )
            tournament_runs.append(res)

        del work_gpu
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        best_run = min(tournament_runs, key=lambda r: float(r["final_val_ce"]))
        winner_d_complex = int(best_run["d_complex"])
        print("\n" + "=" * 96, flush=True)
        print(
            f"[Wave 39 Tournament Winner] d_complex={winner_d_complex} (H={best_run['n_registers']}, "
            f"d_h={best_run['d_register']}, {best_run['params_M']}M params) "
            f"| Val CE={best_run['final_val_ce']:.4f} | PPL={best_run['final_val_ppl']:.2f} "
            f"| Top1={best_run['final_top1_acc']:.2f}% | Steps={best_run['steps']} ({best_run['tokens_M']}M tok) "
            f"| {best_run['tok_s']:,.0f} tok/s",
            flush=True,
        )
        print("=" * 96 + "\n", flush=True)

        payload = {
            "wave_id": "W39",
            "plan_id": "EXPERIMENT_PLAN_054",
            "h1_report": h1_report,
            "wall_clock_sec": float(args.wall_clock_sec),
            "winner_d_complex": winner_d_complex,
            "winner_val_ce": best_run["final_val_ce"],
            "winner_val_ppl": best_run["final_val_ppl"],
            "winner_top1_acc": best_run["final_top1_acc"],
            "winner_tokens_M": best_run["tokens_M"],
            "winner_tok_s": best_run["tok_s"],
            "delta_vs_wave38_d128_300step": round(float(best_run["final_val_ce"]) - WAVE38_D128_300STEP_CE, 4),
            "delta_vs_wave38_max5060ti_300step": round(float(best_run["final_val_ce"]) - WAVE38_MAX5060TI_300STEP_CE, 4),
            "tournament_runs": tournament_runs,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[Wave 39] Saved tournament results to {out_path}", flush=True)

    if args.stage in ("run_900m", "both"):
        win_cand = _resolve_candidate_spec(winner_d_complex, sizer_rows)
        run_900m_winner_production(
            win_cand, prior_tokens, val_batches, ppmi_vecs, float(args.target_tokens_m), device, out_path
        )


if __name__ == "__main__":
    main()
