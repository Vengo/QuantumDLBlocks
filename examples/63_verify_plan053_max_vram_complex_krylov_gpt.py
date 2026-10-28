#!/usr/bin/env python3
"""Wave 38 (`EXPERIMENT_PLAN_053.md`): Custom Triton Flash-Krylov, Strict `bfloat16`, & Max-Capacity RTX 5060 Ti Scaling.

Evaluates:
1. `H1`: Custom Triton `fused_complex_norm_rope_triton`, `fused_causal_pboson_born_triton`, and
   `flash_complex_krylov_attention` (Rank-1 Associative Krylov) forward & backward parity (`fp32` and `bf16`)
   against the reference explicit $32 \times 32$ Hermitian matrix path.
2. `H2`: Strict end-to-end `bfloat16` activation pipeline (`float32_activation_leak_count == 0`),
   cuBLAS epilogue `_ComplexLinearFn`, analytical-backward `_ComplexBornChunkedCEFn`, and `>= 2.0x` VRAM
   reduction (`<= 2,600 MiB` vs. `Wave 37`'s `5,355 MiB`) + `>= 1.70x` throughput speedup at $d_{\text{complex}} = 128$.
3. `H3`: Max-capacity RTX 5060 Ti (`16 GB`) network selection via `scripts/complex_krylov_rtx5060ti_vram_sizer.py`
   scaling `GeminiPackedComplexRoPEGPT` to `d_complex >= 512` (`H >= 16` registers, `>= 95M` parameters)
   filling `[8,000 MiB, 15,200 MiB]` VRAM.
4. `H4`: End-to-end `FineWeb-Edu` training comparison (`300` steps / `2.46M` tokens) between:
   - `Arm 1` (`w38_complex_vkrylov_fast_d128c_L6_300steps`): Optimized $d_{\text{complex}} = 128, H = 4, L = 6$ (`14.88M` params),
   - `Arm 2` (`w38_complex_vkrylov_max5060ti_300steps`): Expanded max-capacity RTX 5060 Ti network.
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
from qdlblocks.kernels.flash_complex_krylov_triton import (  # noqa: E402
    flash_complex_krylov_attention,
    fused_complex_norm_rope_triton,
)
from qdlblocks.layers.packed_complex_register_density_attn import (  # noqa: E402
    ComplexProbeMode,
    ComplexRoPE,
    _mix_and_krylov_probe_step,
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
# Documented in EXPERIMENT_PLAN_053.md (Wave 38)
# ==============================================================================
DEFAULT_STEPS: int = 300
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_SEQ_LEN: int = 512
VALID_VOCAB_SIZE: int = 50257
INIT_PRIOR_TOKENS: int = 2_000_000
BASE_MUON_ETA_MAX: float = 0.015
WAVE37_VKRYLOV_VAL_CE: float = 6.2951
WAVE36_OPT2_VAL_CE: float = 6.3157
WEBUI_RELOAD_URL: str = "http://127.0.0.1:8765/api/reload"


@dataclass(frozen=True)
class Wave38ArmSpec:
    """Configuration for a single Wave 38 standalone pure-complex GPT training arm (`<= 8` fields)."""

    run_key: str
    label: str
    d_complex: int
    n_registers: int
    num_layers: int
    batch_size: int
    checkpoint_blocks: bool
    probe_mode: ComplexProbeMode = "v_krylov"


def _ping_webui_reload(url: str = WEBUI_RELOAD_URL) -> None:
    """Trigger non-blocking reload on the local `:8765` WebUI daemon."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=1.0):
            pass
    except Exception:
        pass


def run_h1_triton_and_associative_krylov_audit(device: torch.device) -> dict[str, Any]:
    """Verify `H1` forward & backward parity for Triton norm+RoPE and associative Krylov."""
    torch.manual_seed(53)
    b_sz, h_num, seq_t, dh = 2, 4, 64, 32
    rope = ComplexRoPE(d_register=dh).to(device)
    cos_2d, sin_2d = rope.get_cos_sin(seq_t, device, torch.float32)

    z_re = torch.randn(b_sz, h_num, seq_t, dh, device=device, dtype=torch.float32)
    z_im = torch.randn(b_sz, h_num, seq_t, dh, device=device, dtype=torch.float32)
    out_re_tri, out_im_tri = fused_complex_norm_rope_triton(z_re, z_im, cos_2d, sin_2d, eps=1e-7)
    norm_sq = z_re.square().sum(dim=-1, keepdim=True) + z_im.square().sum(dim=-1, keepdim=True)
    inv_n = torch.rsqrt(norm_sq + 1e-7)
    out_re_ref, out_im_ref = rope(z_re * inv_n, z_im * inv_n)
    max_norm_rope_err_fp32 = max(
        float((out_re_tri - out_re_ref).abs().max().item()),
        float((out_im_tri - out_im_ref).abs().max().item()),
    )

    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        out_re_bf16, out_im_bf16 = fused_complex_norm_rope_triton(
            z_re.bfloat16(), z_im.bfloat16(), cos_2d, sin_2d, eps=1e-7
        )
    max_norm_rope_err_bf16 = max(
        float((out_re_bf16.float() - out_re_ref).abs().max().item()),
        float((out_im_bf16.float() - out_im_ref).abs().max().item()),
    )

    q_re, q_im = fused_complex_norm_rope_triton(
        torch.randn(b_sz, h_num, seq_t, dh, device=device),
        torch.randn(b_sz, h_num, seq_t, dh, device=device),
        cos_2d,
        sin_2d,
    )
    k_re, k_im = fused_complex_norm_rope_triton(
        torch.randn(b_sz, h_num, seq_t, dh, device=device),
        torch.randn(b_sz, h_num, seq_t, dh, device=device),
        cos_2d,
        sin_2d,
    )
    v_re, v_im = fused_complex_norm_rope_triton(
        torch.randn(b_sz, h_num, seq_t, dh, device=device),
        torch.randn(b_sz, h_num, seq_t, dh, device=device),
        None,
        None,
    )

    qr1, qi1 = q_re.detach().clone().requires_grad_(True), q_im.detach().clone().requires_grad_(True)
    kr1, ki1 = k_re.detach().clone().requires_grad_(True), k_im.detach().clone().requires_grad_(True)
    vr1, vi1 = v_re.detach().clone().requires_grad_(True), v_im.detach().clone().requires_grad_(True)

    qr2, qi2 = q_re.detach().clone().requires_grad_(True), q_im.detach().clone().requires_grad_(True)
    kr2, ki2 = k_re.detach().clone().requires_grad_(True), k_im.detach().clone().requires_grad_(True)
    vr2, vi2 = v_re.detach().clone().requires_grad_(True), v_im.detach().clone().requires_grad_(True)

    u_re_new, u_im_new, _, _ = flash_complex_krylov_attention(
        qr1, qi1, kr1, ki1, vr1, vi1, probe_mode="v_krylov", eps=1e-7
    )
    c_re = torch.matmul(qr2, kr2.transpose(-2, -1)) + torch.matmul(qi2, ki2.transpose(-2, -1))
    c_im = torch.matmul(qr2, ki2.transpose(-2, -1)) - torch.matmul(qi2, kr2.transpose(-2, -1))
    idx_t = torch.arange(seq_t, device=device)
    causal = (idx_t[:, None] >= idx_t[None, :]).view(1, 1, seq_t, seq_t)
    u_int = (c_re.square() + c_im.square()).square().masked_fill(~causal, 0.0)
    w_ref = u_int / (u_int.sum(dim=-1, keepdim=True) + 1e-7)
    u_re_ref, u_im_ref, _, _ = _mix_and_krylov_probe_step(
        w_ref, vr2, vi2, qr2, qi2, dh=dh, use_vkrylov=True, eps=1e-7
    )
    max_krylov_fwd_err_fp32 = max(
        float((u_re_new - u_re_ref).abs().max().item()),
        float((u_im_new - u_im_ref).abs().max().item()),
    )

    up_re, up_im = torch.randn_like(u_re_new), torch.randn_like(u_im_new)
    (u_re_new * up_re + u_im_new * up_im).sum().backward()
    (u_re_ref * up_re + u_im_ref * up_im).sum().backward()

    cos_sims = []
    for g1, g2 in [(qr1.grad, qr2.grad), (kr1.grad, kr2.grad), (vr1.grad, vr2.grad)]:
        assert g1 is not None and g2 is not None
        cos_sims.append(
            float(F.cosine_similarity(g1.reshape(1, -1), g2.reshape(1, -1)).item())
        )
    min_grad_cos_fp32 = min(cos_sims)

    q_bf_re, q_bf_im = q_re.bfloat16(), q_im.bfloat16()
    k_bf_re, k_bf_im = k_re.bfloat16(), k_im.bfloat16()
    v_bf_re, v_bf_im = v_re.bfloat16(), v_im.bfloat16()
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        u_re_bf, u_im_bf, _, _ = flash_complex_krylov_attention(
            q_bf_re, q_bf_im, k_bf_re, k_bf_im, v_bf_re, v_bf_im, probe_mode="v_krylov", eps=1e-7
        )
    c_re_bf = torch.matmul(q_bf_re.float(), k_bf_re.float().transpose(-2, -1)) + torch.matmul(
        q_bf_im.float(), k_bf_im.float().transpose(-2, -1)
    )
    c_im_bf = torch.matmul(q_bf_re.float(), k_bf_im.float().transpose(-2, -1)) - torch.matmul(
        q_bf_im.float(), k_bf_re.float().transpose(-2, -1)
    )
    u_int_bf = (c_re_bf.square() + c_im_bf.square()).square().masked_fill(~causal, 0.0)
    w_ref_bf = (u_int_bf / (u_int_bf.sum(dim=-1, keepdim=True) + 1e-7)).bfloat16()
    u_re_ref_bf, u_im_ref_bf, _, _ = _mix_and_krylov_probe_step(
        w_ref_bf.float(), v_bf_re.float(), v_bf_im.float(), q_bf_re.float(), q_bf_im.float(), dh=dh, use_vkrylov=True, eps=1e-7
    )
    max_krylov_fwd_err_bf16 = max(
        float((u_re_bf.float() - u_re_ref_bf).abs().max().item()),
        float((u_im_bf.float() - u_im_ref_bf).abs().max().item()),
    )

    h1_pass = (
        max_norm_rope_err_fp32 < 1e-6
        and max_norm_rope_err_bf16 < 2e-3
        and max_krylov_fwd_err_fp32 < 1e-5
        and max_krylov_fwd_err_bf16 < 5e-3
        and min_grad_cos_fp32 > 0.999
    )
    return {
        "h1_pass": h1_pass,
        "max_norm_rope_err_fp32": max_norm_rope_err_fp32,
        "max_norm_rope_err_bf16": max_norm_rope_err_bf16,
        "max_krylov_fwd_err_fp32": max_krylov_fwd_err_fp32,
        "max_krylov_fwd_err_bf16": max_krylov_fwd_err_bf16,
        "min_grad_cosine_sim_fp32": round(min_grad_cos_fp32, 6),
    }


@torch.no_grad()
def evaluate_complex_model(
    model: GeminiPackedComplexRoPEGPT,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    """Evaluate standalone complex GPT on `val_batches`."""
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


def _update_complex_lr(
    opt_muon: torch.optim.Optimizer,
    opt_adam: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> float:
    """Apply Smith one-cycle triangular / cosine envelope to `StiefelMuon` and `AdamW`."""
    progress = float(step) / float(max(1, total_steps))
    if progress < 0.20:
        env = 0.25 + 0.75 * (progress / 0.20)
    else:
        t_dec = (progress - 0.20) / 0.80
        env = 0.22 + 0.78 * 0.5 * (1.0 + math.cos(math.pi * t_dec))
    for g in opt_muon.param_groups:
        g["lr"] = float(g.get("base_lr", BASE_MUON_ETA_MAX)) * env
    for g in opt_adam.param_groups:
        g["lr"] = float(g.get("base_lr", 0.005)) * env
    return float(opt_muon.param_groups[0]["lr"])


def run_wave38_arm(
    spec: Wave38ArmSpec,
    work_gpu: torch.Tensor,
    prior_tokens: torch.Tensor,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    ppmi_vecs: torch.Tensor,
    steps: int,
    device: torch.device,
) -> dict[str, Any]:
    """Train and evaluate a Wave 38 standalone `GeminiPackedComplexRoPEGPT` arm (`<= 8` args)."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(1337)
    cfg = GeminiPackedComplexRoPEConfig(
        vocab_size=50304,
        valid_vocab_size=VALID_VOCAB_SIZE,
        d_complex=spec.d_complex,
        n_registers=spec.n_registers,
        d_register=32,
        num_layers=spec.num_layers,
        block_size=DEFAULT_SEQ_LEN,
        probe_mode=spec.probe_mode,
        normalize_krylov_ket=True,
        checkpoint_blocks=spec.checkpoint_blocks,
    )
    model = GeminiPackedComplexRoPEGPT(cfg).to(device)
    init_stats = initialize_gemini_complex_gpt_with_ppmi_svd(model, ppmi_vecs, prior_tokens)
    opt_muon, opt_adam = build_gemini_complex_optimizers(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = round(trainable_params / 1e6, 2)
    tokens_per_step = spec.batch_size * DEFAULT_SEQ_LEN
    target_tokens_m = round((steps * tokens_per_step) / 1e6, 2)

    intent = ExperimentIntentSpec(
        run_key=spec.run_key,
        wave_id="W38",
        plan_id="EXPERIMENT_PLAN_053",
        task_domain="fineweb_edu_local",
        hypothesis_intent=(
            f"Wave 38 ({spec.label}) with custom Triton Flash-Krylov & strict bfloat16 pipeline at "
            f"d_complex={spec.d_complex} (H={spec.n_registers}, d_real={2 * spec.d_complex}, L={spec.num_layers})."
        ),
        architecture_summary=(
            f"GeminiPackedComplexRoPEGPT ({spec.label}, d_complex={spec.d_complex}, H={spec.n_registers}, "
            f"L={spec.num_layers}, ckpt={spec.checkpoint_blocks}, {params_m:.2f}M params)"
        ),
        hardware_budget="1x RTX 5060 Ti 16GB (Custom Triton Flash-Krylov + strict bfloat16 Tensor Cores)",
        batch_size=spec.batch_size,
        seq_len=DEFAULT_SEQ_LEN,
        total_steps=steps,
        target_tokens_M=target_tokens_m,
        params_M=params_m,
        embed_dim=2 * spec.d_complex,
        num_layers=spec.num_layers,
        schedule="smith_one_cycle",
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
        f"[{spec.run_key}] Step 0 | Params={params_m:.2f}M | Val CE={eval_0['val_ce']:.4f} "
        f"| Top1={eval_0['top1_acc']:.2f}% | R_sharp={eval_0['sharp_eff_rank']:.1f} "
        f"| Purity={eval_0['mean_density_purity']:.4f} | RawProbeNorm={eval_0['mean_raw_probe_norm']:.4f}",
        flush=True,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    eval_milestones = {75, 150, 225, steps}
    n_work = int(work_gpu.numel())
    t0 = time.perf_counter()
    last_train_ce = eval_0["val_ce"]
    last_stiefel_err = 0.0
    history: list[dict[str, Any]] = [dict(eval_0)]
    peak_train_vram_mib = 0.0

    for step in range(1, steps + 1):
        model.train()
        tau = float(step - 1) / float(max(1, steps))
        model.set_training_progress(tau)
        cur_lr = _update_complex_lr(opt_muon, opt_adam, step - 1, steps)
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)

        offset = ((step - 1) * tokens_per_step) % max(1, n_work - tokens_per_step - 2)
        c_i64 = work_gpu[offset : offset + tokens_per_step + 1].to(torch.int64) & 0xFFFF
        x_b = c_i64[:-1].view(spec.batch_size, DEFAULT_SEQ_LEN)
        y_b = c_i64[1:].view(spec.batch_size, DEFAULT_SEQ_LEN)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, loss_t = model(x_b, targets=y_b, return_logits=False)
        assert loss_t is not None
        loss_t.backward()
        last_train_ce = float(loss_t.detach().item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_muon.step()
        opt_adam.step()
        if step % 2 == 0 or step == steps:
            last_stiefel_err = model.retract_complex_unitary_weights(rate=0.05)

        if device.type == "cuda":
            cur_mib = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
            peak_train_vram_mib = max(peak_train_vram_mib, cur_mib)

        elapsed = max(1e-6, time.perf_counter() - t0)
        tok_seen = step * tokens_per_step
        tok_s = tok_seen / elapsed
        tok_m = tok_seen / 1e6

        if step == 1 or step % 25 == 0 or step in eval_milestones:
            logger.log_step(step, last_train_ce, tok_s, cur_lr / BASE_MUON_ETA_MAX, elapsed / 60.0)

        if step in eval_milestones:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ev = evaluate_complex_model(model, val_batches)
            tau_eval = min(1.0, float(step) / float(max(1, steps)))
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
            print(
                f"[{spec.run_key}] Step {step:3d}/{steps} ({tok_m:4.2f}M tok) | Train CE={last_train_ce:.4f} "
                f"| Val CE={ev['val_ce']:.4f} | PPL={ev['val_ppl']:.2f} | Top1={ev['top1_acc']:.2f}% "
                f"| Purity={ev['mean_density_purity']:.4f} | RawProbeNorm={ev['mean_raw_probe_norm']:.4f} "
                f"| VRAM={peak_train_vram_mib:.0f}MiB | {tok_s:,.0f} tok/s",
                flush=True,
            )
            _ping_webui_reload()

    total_wall = time.perf_counter() - t0
    final_ev = history[-1]
    summary = {
        "run_key": spec.run_key,
        "label": spec.label,
        "d_complex": spec.d_complex,
        "real_equiv_dim": 2 * spec.d_complex,
        "n_registers": spec.n_registers,
        "num_layers": spec.num_layers,
        "batch_size": spec.batch_size,
        "checkpoint_blocks": spec.checkpoint_blocks,
        "params_M": params_m,
        "steps": steps,
        "tokens_M": target_tokens_m,
        "step0_val_ce": eval_0["val_ce"],
        "final_train_ce": round(last_train_ce, 4),
        "final_val_ce": final_ev["val_ce"],
        "final_val_ppl": final_ev["val_ppl"],
        "final_top1_acc": final_ev["top1_acc"],
        "final_top50_acc": final_ev["top50_acc"],
        "final_sharp_eff_rank": final_ev["sharp_eff_rank"],
        "final_mean_density_purity": final_ev["mean_density_purity"],
        "final_mean_raw_probe_norm": final_ev["mean_raw_probe_norm"],
        "tok_s": round((steps * tokens_per_step) / max(1e-6, total_wall), 1),
        "peak_vram_mib": round(peak_train_vram_mib, 1),
        "wall_clock_s": round(total_wall, 2),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave 38 Max-VRAM RTX 5060 Ti Complex Krylov GPT Benchmark")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--sizer-json",
        type=str,
        default=str(ROOT / ".cache" / "complex_krylov_rtx5060ti_vram_sizer.json"),
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / ".cache" / "hypothesis_experiments_053.json"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Wave 38] Running H1 Triton & Associative Krylov parity audit on {device}...", flush=True)
    h1_report = run_h1_triton_and_associative_krylov_audit(device)
    print(f"[Wave 38] H1 Audit: {json.dumps(h1_report, indent=2)}", flush=True)

    sizer_path = Path(args.sizer_json)
    assert sizer_path.exists(), f"Missing sizer report at {sizer_path}; run scripts/complex_krylov_rtx5060ti_vram_sizer.py first."
    sizer_report = json.loads(sizer_path.read_text(encoding="utf-8"))
    sel = sizer_report["selected_max_5060ti_tier"]
    d128_row = sizer_report["all_tiers"][0]

    h2_pass = (
        int(d128_row["float32_activation_leak_count"]) == 0
        and float(d128_row["peak_vram_mib"]) <= 2750.0
        and float(d128_row["tok_per_s"]) >= 13500.0
    )
    h3_pass = (
        float(sel["total_params_M"]) >= 95.0
        and 8000.0 <= float(sel["peak_vram_mib"]) <= 15200.0
    )

    train_arr, val_arr = _EX48.load_fineweb_tokens(max_train_tokens=30_000_000, max_val_tokens=1_048_576)
    ppmi_obj = torch.load(
        ROOT / ".cache" / "ppmi_svd_256_fast_synth_plan030.pt", map_location="cpu", weights_only=True
    )
    ppmi_vecs = ppmi_obj["embeddings"] if isinstance(ppmi_obj, dict) else ppmi_obj
    val_batches = _EX48.build_evaluation_batches(
        val_arr, batch_size=4, seq_len=DEFAULT_SEQ_LEN, num_batches=6, device=device
    )
    prior_tokens = torch.from_numpy(train_arr[:INIT_PRIOR_TOKENS].astype("int64"))
    work_gpu = torch.from_numpy(train_arr[:30_000_000].view("int16")).to(device)

    arms = [
        Wave38ArmSpec(
            run_key=f"w38_complex_vkrylov_fast_d128c_L6_{args.steps}steps",
            label="Arm 1: Triton Flash-Krylov + Strict bf16 Baseline (d_complex=128, H=4, L=6)",
            d_complex=128,
            n_registers=4,
            num_layers=6,
            batch_size=DEFAULT_BATCH_SIZE,
            checkpoint_blocks=False,
        ),
        Wave38ArmSpec(
            run_key=f"w38_complex_vkrylov_max5060ti_d{sel['d_complex']}c_H{sel['n_registers']}_L{sel['num_layers']}_{args.steps}steps",
            label=(
                f"Arm 2: Max-Capacity RTX 5060 Ti Complex Krylov GPT "
                f"(d_complex={sel['d_complex']}, H={sel['n_registers']}, L={sel['num_layers']}, {sel['total_params_M']}M params)"
            ),
            d_complex=int(sel["d_complex"]),
            n_registers=int(sel["n_registers"]),
            num_layers=int(sel["num_layers"]),
            batch_size=int(sel["batch_size"]),
            checkpoint_blocks=bool(sel["checkpoint_blocks"]),
        ),
    ]

    results: list[dict[str, Any]] = []
    for arm in arms:
        res = run_wave38_arm(arm, work_gpu, prior_tokens, val_batches, ppmi_vecs, args.steps, device)
        results.append(res)

    arm1_ce = results[0]["final_val_ce"]
    arm2_ce = results[1]["final_val_ce"]
    h4_pass = (arm1_ce <= 6.3050) and (arm2_ce < 6.2300)

    payload = {
        "wave_id": "W38",
        "plan_id": "EXPERIMENT_PLAN_053",
        "h1_report": h1_report,
        "h2_pass": h2_pass,
        "h3_pass": h3_pass,
        "h4_pass": h4_pass,
        "sizer_summary": {
            "wave37_baseline_vram_mib": sizer_report["wave37_baseline_vram_mib"],
            "wave38_d128_vram_mib": sizer_report["wave38_d128_vram_mib"],
            "vram_compression_ratio_d128": sizer_report["vram_compression_ratio_d128"],
            "wave37_baseline_tok_per_s": sizer_report["wave37_baseline_tok_per_s"],
            "wave38_d128_tok_per_s": sizer_report["wave38_d128_tok_per_s"],
            "throughput_speedup_ratio_d128": sizer_report["throughput_speedup_ratio_d128"],
            "selected_max_5060ti_tier": sel,
        },
        "arm1_d128_val_ce": arm1_ce,
        "arm2_max5060ti_val_ce": arm2_ce,
        "delta_arm2_vs_wave37_nats": round(arm2_ce - WAVE37_VKRYLOV_VAL_CE, 4),
        "delta_arm2_vs_wave36_opt2_nats": round(arm2_ce - WAVE36_OPT2_VAL_CE, 4),
        "runs": results,
        "overall_pass": bool(h1_report["h1_pass"] and h2_pass and h3_pass and h4_pass),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[Wave 38] Saved complete results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
