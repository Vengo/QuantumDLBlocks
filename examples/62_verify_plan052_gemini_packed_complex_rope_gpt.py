#!/usr/bin/env python3
"""Wave 37 (`EXPERIMENT_PLAN_052.md`): Standalone Pure-Complex Register Density GPT Verification & Benchmark.

Evaluates the standalone pure-complex architecture (`GeminiPackedComplexRoPEGPT`) built around
`GeminiBornComplexPackedWithRoPE.py` (with Refinements 1, 3, and 4, and zero `gamma_coh`):
1. `H1`: Exact 1-GEMM Hermitian Triangle Packing/Unpacking (`< 1e-6`), `ComplexRoPE` relative-distance
   shift invariance (`< 1e-6`), and associative two-step Krylov `O(d_h^2)` vs. explicit McWeeny `O(d_h^3)`
   purification parity (`< 1e-6`).
2. `H2`: Step-0 Subspace Alignment & Shift-Equivariance Audit comparing the original `Q-Probe`
   (`rho_t^purified |q_t^rot>`) against `V-Space Krylov Power Iteration` (`rho_t^purified |v_bar_t>`).
3. `H3`: 3-arm standalone pure-complex GPT training comparison on `FineWeb-Edu` (`300` steps / `2.46M` tokens
   at `d_complex = 128`, `H = 4`, `d_h = 32` -> `256` real degrees of freedom per token, `L = 6, B = 16, T = 512`)
   on the exact same training stream and validation batches as `Wave 36`:
   - `w37_gemini_complex_qprobe_d128c_L6_300steps`: Original `Q-Probe` (`probe_mode="q_probe"`, unnormalized),
   - `w37_gemini_complex_vkrylov_raw_d128c_L6_300steps`: `V-Krylov` (`probe_mode="v_krylov"`, unnormalized),
   - `w37_gemini_complex_vkrylov_norm_d128c_L6_300steps`: `V-Krylov` (`probe_mode="v_krylov"`, unit-normalized ket).
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
from qdlblocks.layers.packed_complex_register_density_attn import (  # noqa: E402
    ComplexProbeMode,
    ComplexRoPE,
    complex_l2_normalize,
    explicit_mcweeny_density_probe,
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
# Documented in EXPERIMENT_PLAN_052.md (Wave 37)
# ==============================================================================
DEFAULT_STEPS: int = 300
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_SEQ_LEN: int = 512
DEFAULT_D_COMPLEX: int = 128
DEFAULT_N_REGISTERS: int = 4
DEFAULT_D_REGISTER: int = 32
DEFAULT_NUM_LAYERS: int = 6
VALID_VOCAB_SIZE: int = 50257
INIT_PRIOR_TOKENS: int = 2_000_000
BASE_MUON_ETA_MIN: float = 0.004
BASE_MUON_ETA_MAX: float = 0.015
WEBUI_RELOAD_URL: str = "http://127.0.0.1:8765/api/reload"


@dataclass(frozen=True)
class ComplexVariantSpec:
    """Specification for a single Wave 37 standalone complex GPT training run (`<= 8` fields)."""

    run_key: str
    label: str
    d_complex: int
    n_registers: int
    d_register: int
    num_layers: int
    probe_mode: ComplexProbeMode
    normalize_krylov_ket: bool


def _ping_webui_reload(url: str = WEBUI_RELOAD_URL) -> None:
    """Trigger non-blocking reload on the local `:8765` WebUI daemon."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=1.0):
            pass
    except Exception:
        pass


def run_h1_and_h2_algebraic_probes(device: torch.device) -> dict[str, Any]:
    """Execute H1 (packing/RoPE/associative Krylov) and H2 (Step-0 V-Krylov vs Q-Probe) audits."""
    torch.manual_seed(7)
    b_sz, h_num, seq_t, dh = 2, 8, 64, 32
    v_re, v_im = complex_l2_normalize(
        torch.randn(b_sz, h_num, seq_t, dh, device=device),
        torch.randn(b_sz, h_num, seq_t, dh, device=device),
        dim=-1,
    )
    w_born = torch.softmax(torch.randn(b_sz, h_num, seq_t, seq_t, device=device), dim=-1).tril()
    w_born = w_born / (w_born.sum(dim=-1, keepdim=True) + 1e-8)

    ref_outer_re = (
        v_re.unsqueeze(-1) * v_re.unsqueeze(-2) + v_im.unsqueeze(-1) * v_im.unsqueeze(-2)
    ).view(b_sz, h_num, seq_t, dh * dh)
    ref_outer_im = (
        v_im.unsqueeze(-1) * v_re.unsqueeze(-2) - v_re.unsqueeze(-1) * v_im.unsqueeze(-2)
    ).view(b_sz, h_num, seq_t, dh * dh)
    ref_rho_re = torch.matmul(w_born, ref_outer_re).view(b_sz, h_num, seq_t, dh, dh)
    ref_rho_im = torch.matmul(w_born, ref_outer_im).view(b_sz, h_num, seq_t, dh, dh)

    packed_and_ket = pack_hermitian_outer_product_and_ket(v_re, v_im)
    mixed_all = torch.matmul(w_born, packed_and_ket)
    rho_re, rho_im = unpack_hermitian_matrix(mixed_all[..., : dh * dh], dh=dh)
    vbar_re, vbar_im = complex_l2_normalize(
        mixed_all[..., dh * dh : dh * dh + dh], mixed_all[..., dh * dh + dh :], dim=-1
    )

    err_pack_re = float((ref_rho_re - rho_re).abs().max().item())
    err_pack_im = float((ref_rho_im - rho_im).abs().max().item())

    rope = ComplexRoPE(d_register=dh).to(device)
    q1_re, q1_im = complex_l2_normalize(
        torch.randn(1, 1, 1, dh, device=device), torch.randn(1, 1, 1, dh, device=device)
    )
    k1_re, k1_im = complex_l2_normalize(
        torch.randn(1, 1, 1, dh, device=device), torch.randn(1, 1, 1, dh, device=device)
    )
    qs_re, qs_im = rope(q1_re.expand(1, 1, 24, dh), q1_im.expand(1, 1, 24, dh))
    ks_re, ks_im = rope(k1_re.expand(1, 1, 24, dh), k1_im.expand(1, 1, 24, dh))
    c_re = torch.matmul(qs_re, ks_re.transpose(-2, -1)) + torch.matmul(qs_im, ks_im.transpose(-2, -1))
    c_im = torch.matmul(qs_re, ks_im.transpose(-2, -1)) - torch.matmul(qs_im, ks_re.transpose(-2, -1))
    err_rope_re = float((c_re[0, 0, 9, 3] - c_re[0, 0, 19, 13]).abs().item())
    err_rope_im = float((c_im[0, 0, 9, 3] - c_im[0, 0, 19, 13]).abs().item())

    mc_re, mc_im = explicit_mcweeny_density_probe(rho_re, rho_im, vbar_re, vbar_im, normalize_output=False)
    kr_re, kr_im = krylov_purified_density_probe(rho_re, rho_im, vbar_re, vbar_im, normalize_output=False)
    err_krylov_re = float((mc_re - kr_re).abs().max().item())
    err_krylov_im = float((mc_im - kr_im).abs().max().item())

    # H2: Single-dominant-token Step-0 norm & overlap
    vs_re, vs_im = v_re[:, :, :1, :], v_im[:, :, :1, :]
    q_rand_re, q_rand_im = complex_l2_normalize(
        torch.randn(b_sz, h_num, 1, dh, device=device),
        torch.randn(b_sz, h_num, 1, dh, device=device),
        dim=-1,
    )
    p1 = pack_hermitian_outer_product_and_ket(vs_re, vs_im)
    rho1_re, rho1_im = unpack_hermitian_matrix(p1[..., : dh * dh], dh=dh)
    uq_re, uq_im = krylov_purified_density_probe(rho1_re, rho1_im, q_rand_re, q_rand_im, normalize_output=False)
    uv_re, uv_im = krylov_purified_density_probe(rho1_re, rho1_im, vs_re, vs_im, normalize_output=False)
    q_probe_norm = float(torch.sqrt(uq_re.square().sum(dim=-1) + uq_im.square().sum(dim=-1)).mean().item())
    v_krylov_norm = float(torch.sqrt(uv_re.square().sum(dim=-1) + uv_im.square().sum(dim=-1)).mean().item())
    v_krylov_ov_re = float((vs_re * uv_re + vs_im * uv_im).sum(dim=-1).mean().item())

    # H2: Competing 3-token orthogonal mixture (0.60, 0.25, 0.15)
    eye_re = torch.eye(dh, device=device).unsqueeze(0).unsqueeze(0)[:, :, :3, :]
    eye_im = torch.zeros_like(eye_re)
    w_comp = torch.tensor([0.60, 0.25, 0.15], device=device, dtype=torch.float32).view(1, 1, 1, 3)
    p3 = pack_hermitian_outer_product_and_ket(eye_re, eye_im)
    m3 = torch.matmul(w_comp, p3)
    r3_re, r3_im = unpack_hermitian_matrix(m3[..., : dh * dh], dh=dh)
    vb3_re, vb3_im = complex_l2_normalize(m3[..., dh * dh : dh * dh + dh], m3[..., dh * dh + dh :], dim=-1)
    uk3_re, _ = krylov_purified_density_probe(r3_re, r3_im, vb3_re, vb3_im, normalize_output=True)
    fid_vbar = float(vb3_re[0, 0, 0, 0].square().item())
    fid_krylov = float(uk3_re[0, 0, 0, 0].square().item())

    h1_pass = max(err_pack_re, err_pack_im, err_rope_re, err_rope_im, err_krylov_re, err_krylov_im) < 1e-6
    h2_pass = (q_probe_norm < 0.25) and (abs(v_krylov_norm - 1.0) < 1e-5) and (fid_krylov > 0.96)
    return {
        "h1_pass": h1_pass,
        "h2_pass": h2_pass,
        "err_pack_re": err_pack_re,
        "err_pack_im": err_pack_im,
        "err_rope_re": err_rope_re,
        "err_rope_im": err_rope_im,
        "err_krylov_re": err_krylov_re,
        "err_krylov_im": err_krylov_im,
        "single_token_q_probe_norm": round(q_probe_norm, 4),
        "single_token_v_krylov_norm": round(v_krylov_norm, 4),
        "single_token_v_krylov_overlap_re": round(v_krylov_ov_re, 4),
        "competing_vbar_principal_fidelity": round(fid_vbar, 4),
        "competing_krylov_principal_fidelity": round(fid_krylov, 4),
    }


@torch.no_grad()
def evaluate_complex_model(
    model: GeminiPackedComplexRoPEGPT,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    """Evaluate standalone complex GPT on `val_batches` (`val_ce`, `val_ppl`, `top1_acc`, `top50_acc`, `R_sharp`)."""
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


def run_complex_variant(
    spec: ComplexVariantSpec,
    work_gpu: torch.Tensor,
    prior_tokens: torch.Tensor,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    ppmi_vecs: torch.Tensor,
    steps: int,
    device: torch.device,
) -> dict[str, Any]:
    """Train and evaluate a single standalone `GeminiPackedComplexRoPEGPT` variant on `FineWeb-Edu`."""
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
        d_register=spec.d_register,
        num_layers=spec.num_layers,
        block_size=DEFAULT_SEQ_LEN,
        probe_mode=spec.probe_mode,
        normalize_krylov_ket=spec.normalize_krylov_ket,
    )
    model = GeminiPackedComplexRoPEGPT(cfg).to(device)
    init_stats = initialize_gemini_complex_gpt_with_ppmi_svd(model, ppmi_vecs, prior_tokens)
    opt_muon, opt_adam = build_gemini_complex_optimizers(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = round(trainable_params / 1e6, 2)
    tokens_per_step = DEFAULT_BATCH_SIZE * DEFAULT_SEQ_LEN
    target_tokens_m = round((steps * tokens_per_step) / 1e6, 2)

    intent = ExperimentIntentSpec(
        run_key=spec.run_key,
        wave_id="W37",
        plan_id="EXPERIMENT_PLAN_052",
        task_domain="fineweb_edu_local",
        hypothesis_intent=(
            f"Wave 37 Standalone Pure-Complex GPT ({spec.label}, probe_mode={spec.probe_mode}, "
            f"normalize_krylov_ket={spec.normalize_krylov_ket}, zero gamma_coh) at d_complex={spec.d_complex} "
            f"(H={spec.n_registers}, d_h={spec.d_register}, L={spec.num_layers})."
        ),
        architecture_summary=(
            f"GeminiPackedComplexRoPEGPT ({spec.label}, d_complex={spec.d_complex}, H={spec.n_registers}, "
            f"d_h={spec.d_register}, L={spec.num_layers}, {params_m:.2f}M params)"
        ),
        hardware_budget="1x RTX 5060 Ti 16GB (bfloat16 Tensor-Core 1-GEMM width 1088)",
        batch_size=DEFAULT_BATCH_SIZE,
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

    # Verify exact train/eval parity before Step 1
    model.eval()
    x0, y0 = val_batches[0]
    with torch.no_grad():
        logits0, loss_train0 = model(x0[:2], targets=y0[:2], return_logits=True)
        assert logits0 is not None and loss_train0 is not None
        loss_eval0 = F.cross_entropy(
            logits0[..., : cfg.valid_vocab_size].float().reshape(-1, cfg.valid_vocab_size),
            y0[:2].clamp(0, cfg.valid_vocab_size - 1).reshape(-1),
        )
        parity_diff = abs(float(loss_train0.item()) - float(loss_eval0.item()))

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
        f"[{spec.run_key}] Step 0 | Val CE={eval_0['val_ce']:.4f} | Top1={eval_0['top1_acc']:.2f}% "
        f"| R_sharp={eval_0['sharp_eff_rank']:.1f} | Purity={eval_0['mean_density_purity']:.4f} "
        f"| RawProbeNorm={eval_0['mean_raw_probe_norm']:.4f} | ParityDiff={parity_diff:.2e}",
        flush=True,
    )

    eval_milestones = {75, 150, 225, steps}
    n_work = int(work_gpu.numel())
    t0 = time.perf_counter()
    last_train_ce = eval_0["val_ce"]
    last_stiefel_err = 0.0
    history: list[dict[str, Any]] = [dict(eval_0)]

    for step in range(1, steps + 1):
        model.train()
        tau = float(step - 1) / float(max(1, steps))
        model.set_training_progress(tau)
        cur_lr = _update_complex_lr(opt_muon, opt_adam, step - 1, steps)
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)

        offset = ((step - 1) * tokens_per_step) % max(1, n_work - tokens_per_step - 2)
        c_i64 = work_gpu[offset : offset + tokens_per_step + 1].to(torch.int64) & 0xFFFF
        x_b = c_i64[:-1].view(DEFAULT_BATCH_SIZE, DEFAULT_SEQ_LEN)
        y_b = c_i64[1:].view(DEFAULT_BATCH_SIZE, DEFAULT_SEQ_LEN)

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
            peak_mib = (
                float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
                if device.type == "cuda"
                else 0.0
            )
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
                    "peak_vram_mib": round(peak_mib, 1),
                    "kraus_stiefel_err": round(last_stiefel_err, 6),
                }
            )
            history.append(dict(ev))
            logger.log_eval(step, ev)
            print(
                f"[{spec.run_key}] Step {step:3d}/{steps} ({tok_m:4.2f}M tok) | Train CE={last_train_ce:.4f} "
                f"| Val CE={ev['val_ce']:.4f} | PPL={ev['val_ppl']:.2f} | Top1={ev['top1_acc']:.2f}% "
                f"| Purity={ev['mean_density_purity']:.4f} | RawProbeNorm={ev['mean_raw_probe_norm']:.4f} "
                f"| VRAM={peak_mib:.0f}MiB | {tok_s:,.0f} tok/s",
                flush=True,
            )
            _ping_webui_reload()

    total_wall = time.perf_counter() - t0
    peak_vram_mib = (
        round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 1) if device.type == "cuda" else 0.0
    )
    final_ev = history[-1]
    summary = {
        "run_key": spec.run_key,
        "label": spec.label,
        "probe_mode": spec.probe_mode,
        "normalize_krylov_ket": spec.normalize_krylov_ket,
        "d_complex": spec.d_complex,
        "real_equiv_dim": 2 * spec.d_complex,
        "n_registers": spec.n_registers,
        "d_register": spec.d_register,
        "num_layers": spec.num_layers,
        "params_M": params_m,
        "steps": steps,
        "tokens_M": target_tokens_m,
        "train_eval_parity_diff": parity_diff,
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
        "peak_vram_mib": peak_vram_mib,
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
    parser = argparse.ArgumentParser(description="Wave 37 Standalone Pure-Complex Register Density GPT Benchmark")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--out-json", type=str, default=str(ROOT / ".cache" / "hypothesis_experiments_052.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Wave 37] Running H1 & H2 algebraic/subspace probes on {device}...", flush=True)
    probe_report = run_h1_and_h2_algebraic_probes(device)
    print(f"[Wave 37] H1 & H2 Probe Report: {json.dumps(probe_report, indent=2)}", flush=True)

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

    variants = [
        ComplexVariantSpec(
            run_key=f"w37_gemini_complex_qprobe_d128c_L6_{args.steps}steps",
            label="Arm 1: Original Q-Probe (rho_t^purified |q_t^rot>, Unnormalized)",
            d_complex=DEFAULT_D_COMPLEX,
            n_registers=DEFAULT_N_REGISTERS,
            d_register=DEFAULT_D_REGISTER,
            num_layers=DEFAULT_NUM_LAYERS,
            probe_mode="q_probe",
            normalize_krylov_ket=False,
        ),
        ComplexVariantSpec(
            run_key=f"w37_gemini_complex_vkrylov_raw_d128c_L6_{args.steps}steps",
            label="Arm 2: V-Space Krylov Probe (Refinements 1+3+4, Unnormalized)",
            d_complex=DEFAULT_D_COMPLEX,
            n_registers=DEFAULT_N_REGISTERS,
            d_register=DEFAULT_D_REGISTER,
            num_layers=DEFAULT_NUM_LAYERS,
            probe_mode="v_krylov",
            normalize_krylov_ket=False,
        ),
        ComplexVariantSpec(
            run_key=f"w37_gemini_complex_vkrylov_norm_d128c_L6_{args.steps}steps",
            label="Arm 3: V-Space Krylov Probe (Refinements 1+3+4, Unit-Normalized Ket)",
            d_complex=DEFAULT_D_COMPLEX,
            n_registers=DEFAULT_N_REGISTERS,
            d_register=DEFAULT_D_REGISTER,
            num_layers=DEFAULT_NUM_LAYERS,
            probe_mode="v_krylov",
            normalize_krylov_ket=True,
        ),
    ]

    results: list[dict[str, Any]] = []
    for v_spec in variants:
        res = run_complex_variant(v_spec, work_gpu, prior_tokens, val_batches, ppmi_vecs, args.steps, device)
        results.append(res)

    q_ce = results[0]["final_val_ce"]
    vk_raw_ce = results[1]["final_val_ce"]
    vk_norm_ce = results[2]["final_val_ce"]
    best_vk_ce = min(vk_raw_ce, vk_norm_ce)
    h3_pass = best_vk_ce < q_ce

    payload = {
        "wave_id": "W37",
        "plan_id": "EXPERIMENT_PLAN_052",
        "probe_report": probe_report,
        "h3_pass": h3_pass,
        "q_probe_val_ce": q_ce,
        "v_krylov_raw_val_ce": vk_raw_ce,
        "v_krylov_norm_val_ce": vk_norm_ce,
        "delta_vkrylov_vs_qprobe_nats": round(best_vk_ce - q_ce, 4),
        "wave36_option2_d256_baseline_val_ce": 6.3157,
        "runs": results,
        "overall_pass": bool(probe_report["h1_pass"] and probe_report["h2_pass"] and h3_pass),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[Wave 37] Saved complete results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
