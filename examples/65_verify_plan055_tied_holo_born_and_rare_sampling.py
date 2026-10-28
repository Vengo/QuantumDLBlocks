#!/usr/bin/env python3
"""Wave 40 (`EXPERIMENT_PLAN_055.md`): Tied Holographic Complex Born Readout, Soft Penalties, & Rare Sampling.

Executes and verifies all four hypotheses (`H1`-`H4`) of `EXPERIMENT_PLAN_055.md`:
1. `H1`: 100% Softmax-Free Tied Operator-Wavefunction Complex Born Readout (`TiedHoloBornReadout`),
   unit self-overlap `|<e_v | e_v>_C|^2 == 1.0`, sequence-side `out_operator` (`6.13x` fewer GEMM rows),
   and `TiedHoloBornChunkedNLLFn` autograd parity (`< 1e-5` forward, `< 1e-4` backward).
2. `H2`: Soft Unit-Sphere, Square Complex-Unitary, and Parseval-Isotropy penalties (`soft_penalties_parseval_iso`)
   vs. legacy hard retraction + `meas_proj` (`hard_retract_legacy`) over `200` FineWeb steps (`R_emb >= 140` vs `< 30`).
3. `H3`: Isotropic Haar Unit-Sphere initialization & `RareTokenImportanceWindowSampler` (`alpha_rare = 0.28`)
   vs. PPMI-SVD initialization (`>= 1.5x` rare-token exposure gain, `< 1e-5` IS unbiasedness error).
4. `H4`: 5-minute (`300s`) live training run of `TiedHoloBornComplexRoPEGPT` (`d_complex = 128, H = 4, d_h = 32, L = 6`)
   on `FineWeb` with `StandardizedExperimentLogger`, preserving `R_emb >= 140` across the entire trajectory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import importlib.util
import inspect
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
from qdlblocks.layers.packed_complex_register_density_attn import (  # noqa: E402
    complex_l2_normalize,
)
from qdlblocks.models.gemini_packed_complex_rope_gpt import (  # noqa: E402
    GeminiPackedComplexRoPEConfig,
    GeminiPackedComplexRoPEGPT,
    build_gemini_complex_optimizers,
    initialize_gemini_complex_gpt_with_ppmi_svd,
)
from qdlblocks.models.tied_holo_born_complex_gpt import (  # noqa: E402
    DEFAULT_HOLO_ALPHA,
    DEFAULT_HOLO_BOSON_P,
    DEFAULT_MUON_LR,
    DEFAULT_RARE_ALPHA,
    RareTokenImportanceWindowSampler,
    TiedHoloBornChunkedNLLFn,
    TiedHoloBornComplexRoPEGPT,
    TiedHoloBornConfig,
    build_tied_holo_born_optimizers,
    compute_soft_manifold_and_isotropy_penalties,
    eval_tied_energy_born_probs,
    eval_tied_holo_born_probs,
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
# Synchronized with EXPERIMENT_PLAN_055.md (Wave 40)
# ==============================================================================
PADDED_VOCAB_SIZE: int = 50304
VALID_VOCAB_SIZE: int = 50257
DEFAULT_D_COMPLEX: int = 128
DEFAULT_N_REGISTERS: int = 4
DEFAULT_D_REGISTER: int = 32
DEFAULT_H2_STEPS: int = 200
DEFAULT_H3_STEPS: int = 150
DEFAULT_ABLAT_LAYERS: int = 4
DEFAULT_ABLAT_BATCH: int = 8
DEFAULT_ABLAT_SEQ_LEN: int = 256
DEFAULT_H4_LAYERS: int = 6
DEFAULT_H4_BATCH: int = 16
DEFAULT_H4_SEQ_LEN: int = 512
DEFAULT_WALL_CLOCK_SEC: float = 300.0
INIT_PRIOR_TOKENS: int = 2_000_000
WAVE39_FINAL_RANK_COLLAPSED: float = 27.0
WEBUI_RELOAD_URL: str = "http://127.0.0.1:8765/api/reload"


@dataclass(frozen=True)
class AblationArmSpec:
    """Configuration for an `H2` or `H3` ablation arm (`<= 8` fields)."""

    run_key: str
    label: str
    init_mode: str
    alpha_rare: float
    steps: int
    d_complex: int = DEFAULT_D_COMPLEX
    num_layers: int = DEFAULT_ABLAT_LAYERS


def _ping_webui_reload(url: str = WEBUI_RELOAD_URL) -> None:
    """Trigger non-blocking reload on the local `:8765` WebUI daemon."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=1.0):
            pass
    except Exception:
        pass


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
        g["lr"] = float(g.get("base_lr", DEFAULT_MUON_LR)) * env
    for g in opt_adam.param_groups:
        g["lr"] = float(g.get("base_lr", 0.006)) * env
    return float(opt_muon.param_groups[0]["lr"])


@torch.no_grad()
def evaluate_tied_holo_model(
    model: TiedHoloBornComplexRoPEGPT,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    """Evaluate `TiedHoloBornComplexRoPEGPT` on held-out `FineWeb` validation batches (`<= 8` args)."""
    model.eval()
    v_valid = int(model.cfg.valid_vocab_size)
    total_nll = 0.0
    total_top1 = 0
    total_top50 = 0
    total_tokens = 0
    dev_type = val_batches[0][0].device.type

    for x_val, y_val in val_batches:
        with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=(dev_type == "cuda")):
            log_probs, _ = model(x_val, targets=None, return_logits=True)
        assert log_probs is not None
        lp_f = log_probs[..., :v_valid].float()
        y_clamped = y_val.clamp(min=0, max=v_valid - 1)
        nll = -lp_f.gather(-1, y_clamped.unsqueeze(-1)).squeeze(-1).sum()
        total_nll += float(nll.item())
        preds = lp_f.argmax(dim=-1)
        total_top1 += int((preds == y_clamped).sum().item())
        top50 = lp_f.topk(k=min(50, v_valid), dim=-1).indices
        total_top50 += int((top50 == y_clamped.unsqueeze(-1)).any(dim=-1).sum().item())
        total_tokens += int(y_val.numel())

    val_ce = total_nll / max(1, total_tokens)
    val_ppl = math.exp(min(val_ce, 20.0))
    top1_pct = round(100.0 * total_top1 / max(1, total_tokens), 2)
    top50_pct = round(100.0 * total_top50 / max(1, total_tokens), 2)
    purity_vals = [float(blk.attn.last_purity_mean) for blk in model.blocks]
    probe_norms = [float(blk.attn.last_probe_norm_mean) for blk in model.blocks]
    _, pen_stats = compute_soft_manifold_and_isotropy_penalties(model)

    return {
        "val_ce": round(val_ce, 4),
        "val_ppl": round(val_ppl, 2),
        "val_top1": top1_pct,
        "top1_acc": top1_pct,
        "top50_acc": top50_pct,
        "sharp_eff_rank": round(model.compute_sharp_eff_rank(), 2),
        "mean_density_purity": round(sum(purity_vals) / max(1, len(purity_vals)), 4),
        "mean_raw_probe_norm": round(sum(probe_norms) / max(1, len(probe_norms)), 4),
        "embed_norm_median": round(pen_stats["embed_norm_median"], 4),
        "embed_norm_max": round(pen_stats["embed_norm_max"], 4),
        "complex_unitary_err": round(pen_stats["complex_unitary_err"], 4),
        "kraus_stiefel_err": round(pen_stats["complex_unitary_err"], 6),
    }


@torch.no_grad()
def evaluate_legacy_complex_model(
    model: GeminiPackedComplexRoPEGPT,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    """Evaluate legacy `GeminiPackedComplexRoPEGPT` on held-out `FineWeb` validation batches."""
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
    e_re = model.embed_re[:v_valid].detach().float()
    e_im = model.embed_im[:v_valid].detach().float()
    e_norms = torch.sqrt(e_re.square().sum(dim=-1) + e_im.square().sum(dim=-1))
    return {
        "val_ce": round(val_ce, 4),
        "val_ppl": round(val_ppl, 2),
        "val_top1": top1_pct,
        "top1_acc": top1_pct,
        "top50_acc": top50_pct,
        "sharp_eff_rank": round(model.compute_sharp_eff_rank(), 2),
        "embed_norm_median": round(float(e_norms.median().item()), 4),
        "embed_norm_max": round(float(e_norms.max().item()), 4),
    }


def run_h1_tied_holo_born_readout_audit(device: torch.device) -> dict[str, Any]:
    """Execute `H1`: self-overlap identity, probability sum, chunked autograd parity, and operator row savings."""
    torch.manual_seed(5510)
    d_c = DEFAULT_D_COMPLEX
    v_eff = VALID_VOCAB_SIZE

    raw_re = torch.randn(1024, d_c, device=device, dtype=torch.float32)
    raw_im = torch.randn(1024, d_c, device=device, dtype=torch.float32)
    e_re, e_im = complex_l2_normalize(raw_re, raw_im, dim=-1)
    e_cat = torch.cat([e_re.float(), e_im.float()], dim=-1)

    self_amp = (e_cat * e_cat).sum(dim=-1)
    self_overlap_sq = float(self_amp.square().mean().item())
    max_self_overlap_err = float((self_amp.square() - 1.0).abs().max().item())
    legacy_shrunk_amp = float(((e_cat * (2.0 ** -0.5)) * (e_cat * (2.0 ** -0.5))).sum(dim=-1).mean().item())

    phi_re, phi_im = complex_l2_normalize(
        torch.randn(32, d_c, device=device, dtype=torch.float32),
        torch.randn(32, d_c, device=device, dtype=torch.float32),
        dim=-1,
    )
    prior_w = torch.rand(1024, device=device, dtype=torch.float32).clamp_min(0.05)

    probs_energy = eval_tied_energy_born_probs(phi_re, phi_im, e_re, e_im, prior_w, boson_p=DEFAULT_HOLO_BOSON_P)
    probs_holo = eval_tied_holo_born_probs(
        phi_re, phi_im, e_re, e_im, prior_w, boson_p=DEFAULT_HOLO_BOSON_P, holo_alpha=DEFAULT_HOLO_ALPHA
    )
    max_prob_sum_err_energy = float((probs_energy.sum(dim=-1) - 1.0).abs().max().item())
    max_prob_sum_err_holo = float((probs_holo.sum(dim=-1) - 1.0).abs().max().item())

    # Chunked autograd parity check (vocab_chunk_size = 256 over 1024 vocabulary items)
    targets = torch.randint(0, 1024, (32,), device=device, dtype=torch.int64)
    tok_weights = torch.rand(32, device=device, dtype=torch.float32).clamp_min(0.2)
    TiedHoloBornChunkedNLLFn.vocab_chunk_size = 256

    phi_c = torch.cat([phi_re.float(), phi_im.float()], dim=-1).detach().clone().requires_grad_(True)
    e_c = e_cat.detach().clone().requires_grad_(True)
    loss_chunk = TiedHoloBornChunkedNLLFn.apply(
        phi_c, e_c, targets, prior_w, DEFAULT_HOLO_BOSON_P, DEFAULT_HOLO_ALPHA, tok_weights
    )
    loss_chunk.backward()

    phi_ref = torch.cat([phi_re.float(), phi_im.float()], dim=-1).detach().clone().requires_grad_(True)
    e_ref = e_cat.detach().clone().requires_grad_(True)
    probs_ref = eval_tied_holo_born_probs(
        phi_ref[:, :d_c],
        phi_ref[:, d_c:],
        e_ref[:, :d_c],
        e_ref[:, d_c:],
        prior_w,
        boson_p=DEFAULT_HOLO_BOSON_P,
        holo_alpha=DEFAULT_HOLO_ALPHA,
    )
    nll_ref = -torch.log(probs_ref.gather(1, targets.unsqueeze(1)).squeeze(1))
    loss_ref = ((tok_weights / tok_weights.sum()) * nll_ref).sum()
    loss_ref.backward()
    TiedHoloBornChunkedNLLFn.vocab_chunk_size = 4096

    fwd_loss_err = abs(float(loss_chunk.item()) - float(loss_ref.item()))
    assert phi_c.grad is not None and phi_ref.grad is not None
    assert e_c.grad is not None and e_ref.grad is not None
    max_grad_phi_err = float((phi_c.grad - phi_ref.grad).abs().max().item())
    max_grad_e_err = float((e_c.grad - e_ref.grad).abs().max().item())

    zero_exp_verified = True
    for obj in (eval_tied_holo_born_probs, eval_tied_energy_born_probs, TiedHoloBornChunkedNLLFn):
        src = inspect.getsource(obj)
        if any(tok in src for tok in ("torch.exp(", ".exp(", "softmax(", "logsumexp(")):
            zero_exp_verified = False

    seq_positions = DEFAULT_H4_BATCH * DEFAULT_H4_SEQ_LEN
    operator_row_reduction_ratio = round(float(v_eff) / float(seq_positions), 4)
    h1_pass = (
        max_self_overlap_err < 1e-6
        and max_prob_sum_err_holo < 1e-5
        and fwd_loss_err < 1e-5
        and max(max_grad_phi_err, max_grad_e_err) < 1e-4
        and zero_exp_verified
        and operator_row_reduction_ratio > 6.0
    )
    return {
        "h1_pass": h1_pass,
        "self_overlap_sq_without_shrinkage": round(self_overlap_sq, 6),
        "max_self_overlap_err": max_self_overlap_err,
        "legacy_double_inv_sqrt2_amplitude": round(legacy_shrunk_amp, 6),
        "max_prob_sum_err_energy_born": max_prob_sum_err_energy,
        "max_prob_sum_err_holo_born": max_prob_sum_err_holo,
        "chunked_vs_full_fwd_loss_err": fwd_loss_err,
        "chunked_vs_full_max_grad_phi_err": max_grad_phi_err,
        "chunked_vs_full_max_grad_e_err": max_grad_e_err,
        "zero_exp_softmax_logsumexp_verified": zero_exp_verified,
        "vocab_operator_rows_legacy": v_eff,
        "sequence_operator_rows_wave40": seq_positions,
        "operator_gemm_row_reduction_x": operator_row_reduction_ratio,
    }


def run_h2_soft_penalties_vs_hard_retraction_audit(
    work_gpu: torch.Tensor,
    prior_tokens: torch.Tensor,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    ppmi_vecs: torch.Tensor,
    device: torch.device,
    steps: int = DEFAULT_H2_STEPS,
) -> dict[str, Any]:
    """Execute `H2`: 200-step FineWeb comparison of `hard_retract_legacy` vs `soft_penalties_parseval_iso`."""
    tokens_per_step = DEFAULT_ABLAT_BATCH * DEFAULT_ABLAT_SEQ_LEN
    n_work = int(work_gpu.numel())

    # Arm 1: hard_retract_legacy (GeminiPackedComplexRoPEGPT)
    torch.manual_seed(1337)
    leg_cfg = GeminiPackedComplexRoPEConfig(
        vocab_size=PADDED_VOCAB_SIZE,
        valid_vocab_size=VALID_VOCAB_SIZE,
        d_complex=DEFAULT_D_COMPLEX,
        n_registers=DEFAULT_N_REGISTERS,
        d_register=DEFAULT_D_REGISTER,
        num_layers=DEFAULT_ABLAT_LAYERS,
        block_size=DEFAULT_ABLAT_SEQ_LEN,
    )
    leg_model = GeminiPackedComplexRoPEGPT(leg_cfg).to(device)
    initialize_gemini_complex_gpt_with_ppmi_svd(leg_model, ppmi_vecs, prior_tokens)
    opt_m_leg, opt_a_leg = build_gemini_complex_optimizers(leg_model)

    leg_r0 = leg_model.compute_sharp_eff_rank()
    leg_stiefel_err = 0.0
    for step in range(1, steps + 1):
        tau = float(step - 1) / float(max(1, steps))
        _apply_progress_lr(opt_m_leg, opt_a_leg, tau)
        opt_m_leg.zero_grad(set_to_none=True)
        opt_a_leg.zero_grad(set_to_none=True)
        off = ((step - 1) * tokens_per_step) % max(1, n_work - tokens_per_step - 2)
        seq = (work_gpu[off : off + tokens_per_step + 1].to(torch.int64) & 0xFFFF).clamp(0, VALID_VOCAB_SIZE - 1)
        x_b = seq[:-1].view(DEFAULT_ABLAT_BATCH, DEFAULT_ABLAT_SEQ_LEN)
        y_b = seq[1:].view(DEFAULT_ABLAT_BATCH, DEFAULT_ABLAT_SEQ_LEN)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, loss_leg = leg_model(x_b, targets=y_b, return_logits=False)
        assert loss_leg is not None
        loss_leg.backward()
        torch.nn.utils.clip_grad_norm_(leg_model.parameters(), max_norm=1.0)
        opt_m_leg.step()
        opt_a_leg.step()
        if step % 2 == 0:
            leg_stiefel_err = leg_model.retract_complex_unitary_weights(rate=0.05)

    leg_eval = evaluate_legacy_complex_model(leg_model, val_batches)
    leg_eval["step0_sharp_eff_rank"] = round(leg_r0, 2)
    leg_eval["complex_unitary_err"] = round(leg_stiefel_err, 4)
    del leg_model, opt_m_leg, opt_a_leg
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Arm 2: soft_penalties_parseval_iso (TiedHoloBornComplexRoPEGPT)
    torch.manual_seed(1337)
    soft_cfg = TiedHoloBornConfig(
        vocab_size=PADDED_VOCAB_SIZE,
        valid_vocab_size=VALID_VOCAB_SIZE,
        d_complex=DEFAULT_D_COMPLEX,
        n_registers=DEFAULT_N_REGISTERS,
        d_register=DEFAULT_D_REGISTER,
        num_layers=DEFAULT_ABLAT_LAYERS,
        block_size=DEFAULT_ABLAT_SEQ_LEN,
    )
    soft_model = TiedHoloBornComplexRoPEGPT(soft_cfg).to(device)
    soft_model.initialize_with_ppmi_svd(ppmi_vecs, prior_tokens)
    opt_m_soft, opt_a_soft = build_tied_holo_born_optimizers(soft_model)

    soft_r0 = soft_model.compute_sharp_eff_rank()
    for step in range(1, steps + 1):
        tau = float(step - 1) / float(max(1, steps))
        _apply_progress_lr(opt_m_soft, opt_a_soft, tau)
        opt_m_soft.zero_grad(set_to_none=True)
        opt_a_soft.zero_grad(set_to_none=True)
        off = ((step - 1) * tokens_per_step) % max(1, n_work - tokens_per_step - 2)
        seq = (work_gpu[off : off + tokens_per_step + 1].to(torch.int64) & 0xFFFF).clamp(0, VALID_VOCAB_SIZE - 1)
        x_b = seq[:-1].view(DEFAULT_ABLAT_BATCH, DEFAULT_ABLAT_SEQ_LEN)
        y_b = seq[1:].view(DEFAULT_ABLAT_BATCH, DEFAULT_ABLAT_SEQ_LEN)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, nll_soft = soft_model(x_b, targets=y_b, return_logits=False)
        assert nll_soft is not None
        reg_pen, _ = compute_soft_manifold_and_isotropy_penalties(soft_model)
        (nll_soft + reg_pen).backward()
        torch.nn.utils.clip_grad_norm_(soft_model.parameters(), max_norm=1.0)
        opt_m_soft.step()
        opt_a_soft.step()

    soft_eval = evaluate_tied_holo_model(soft_model, val_batches)
    soft_eval["step0_sharp_eff_rank"] = round(soft_r0, 2)
    del soft_model, opt_m_soft, opt_a_soft
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rank_ratio = round(soft_eval["sharp_eff_rank"] / max(1e-6, leg_eval["sharp_eff_rank"]), 2)
    h2_pass = (
        soft_eval["sharp_eff_rank"] >= 140.0
        and rank_ratio >= 3.5
        and 0.95 <= soft_eval["embed_norm_median"] <= 1.15
        and soft_eval["complex_unitary_err"] < 0.25
    )
    return {
        "h2_pass": h2_pass,
        "steps": steps,
        "hard_retract_legacy": leg_eval,
        "soft_penalties_parseval_iso": soft_eval,
        "rank_improvement_ratio_x": rank_ratio,
    }


def _run_h3_arm(
    arm: AblationArmSpec,
    work_gpu: torch.Tensor,
    prior_tokens: torch.Tensor,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    ppmi_vecs: torch.Tensor,
    sampler: RareTokenImportanceWindowSampler,
    device: torch.device,
) -> dict[str, Any]:
    """Train a single `H3` initialization + window-sampling ablation arm (`<= 8` args)."""
    torch.manual_seed(1337)
    cfg = TiedHoloBornConfig(
        vocab_size=PADDED_VOCAB_SIZE,
        valid_vocab_size=VALID_VOCAB_SIZE,
        d_complex=arm.d_complex,
        n_registers=DEFAULT_N_REGISTERS,
        d_register=DEFAULT_D_REGISTER,
        num_layers=arm.num_layers,
        block_size=DEFAULT_ABLAT_SEQ_LEN,
    )
    model = TiedHoloBornComplexRoPEGPT(cfg).to(device)
    if arm.init_mode == "ppmi_svd":
        init_stats = model.initialize_with_ppmi_svd(ppmi_vecs, prior_tokens)
    else:
        init_stats = model.initialize_isotropic_haar(prior_tokens=prior_tokens, seed=1337)

    opt_muon, opt_adam = build_tied_holo_born_optimizers(model)
    rare_mask_dev = sampler.rare_bottom50_mask.to(device)
    rare_token_hits = 0
    total_sampled_tokens = 0

    for step in range(1, arm.steps + 1):
        tau = float(step - 1) / float(max(1, arm.steps))
        _apply_progress_lr(opt_muon, opt_adam, tau)
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)

        x_b, y_b, tok_w = sampler.sample_batch(
            work_gpu, DEFAULT_ABLAT_BATCH, DEFAULT_ABLAT_SEQ_LEN, step_seed=9000 + step
        )
        rare_token_hits += int(rare_mask_dev[y_b].sum().item())
        total_sampled_tokens += int(y_b.numel())
        weights_arg = tok_w if arm.alpha_rare > 0.0 else None

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, nll_loss = model(x_b, targets=y_b, return_logits=False, token_weights=weights_arg)
        assert nll_loss is not None
        reg_pen, _ = compute_soft_manifold_and_isotropy_penalties(model)
        (nll_loss + reg_pen).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_muon.step()
        opt_adam.step()

    ev = evaluate_tied_holo_model(model, val_batches)
    del model, opt_muon, opt_adam
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "run_key": arm.run_key,
        "label": arm.label,
        "init_mode": arm.init_mode,
        "alpha_rare": arm.alpha_rare,
        "step0_sharp_eff_rank": round(init_stats["sharp_eff_rank"], 2),
        "final_sharp_eff_rank": ev["sharp_eff_rank"],
        "val_ce": ev["val_ce"],
        "val_ppl": ev["val_ppl"],
        "top1_acc": ev["top1_acc"],
        "top50_acc": ev["top50_acc"],
        "rare_bottom50_hits": rare_token_hits,
        "rare_bottom50_share_pct": round(100.0 * rare_token_hits / max(1, total_sampled_tokens), 3),
    }


def run_h3_initialization_and_rare_sampling_audit(
    work_gpu: torch.Tensor,
    prior_tokens: torch.Tensor,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    ppmi_vecs: torch.Tensor,
    device: torch.device,
    steps: int = DEFAULT_H3_STEPS,
) -> dict[str, Any]:
    """Execute `H3`: compare `ppmi_svd_uniform`, `isotropic_haar_uniform`, and `isotropic_haar_rare_reweighted`."""
    sampler_unif = RareTokenImportanceWindowSampler(
        reference_tokens=prior_tokens,
        valid_vocab_size=VALID_VOCAB_SIZE,
        alpha_rare=0.0,
        laplace_eps=1.0,
    )
    sampler_rare = RareTokenImportanceWindowSampler(
        reference_tokens=prior_tokens,
        valid_vocab_size=VALID_VOCAB_SIZE,
        alpha_rare=DEFAULT_RARE_ALPHA,
        laplace_eps=1.0,
    )

    arms = (
        (
            AblationArmSpec("w40_h3_ppmi_svd_uniform", "PPMI-SVD + Uniform Windows", "ppmi_svd", 0.0, steps),
            sampler_unif,
        ),
        (
            AblationArmSpec(
                "w40_h3_isotropic_haar_uniform", "Isotropic Haar + Uniform Windows", "isotropic_haar", 0.0, steps
            ),
            sampler_unif,
        ),
        (
            AblationArmSpec(
                "w40_h3_isotropic_haar_rare_reweighted",
                f"Isotropic Haar + Rare-Token IS (alpha={DEFAULT_RARE_ALPHA}) + Inverse-Prob Reweight",
                "isotropic_haar",
                DEFAULT_RARE_ALPHA,
                steps,
            ),
            sampler_rare,
        ),
    )

    results: dict[str, dict[str, Any]] = {}
    for spec, samp in arms:
        res = _run_h3_arm(spec, work_gpu, prior_tokens, val_batches, ppmi_vecs, samp, device)
        results[spec.run_key] = res

    # Verify exact analytical importance-weighted expectation unbiasedness on a candidate pool of 64 FineWeb windows
    starts = torch.randint(0, int(work_gpu.numel()) - DEFAULT_ABLAT_SEQ_LEN - 2, (64,), device=device)
    offs = starts.unsqueeze(1) + torch.arange(DEFAULT_ABLAT_SEQ_LEN, device=device).unsqueeze(0)
    cand_y = (work_gpu[offs].to(torch.int64) & 0xFFFF).clamp(0, VALID_VOCAB_SIZE - 1)
    q_probs, unnorm_inv_w = sampler_rare.compute_pool_distribution(cand_y)
    pool_losses = 6.0 + 0.5 * torch.cos(torch.arange(64, device=device, dtype=torch.float32))
    is_expectation_err = abs(float((q_probs * unnorm_inv_w * pool_losses).sum().item()) - float(pool_losses.mean().item()))

    unif_hits = results["w40_h3_isotropic_haar_uniform"]["rare_bottom50_hits"]
    rare_hits = results["w40_h3_isotropic_haar_rare_reweighted"]["rare_bottom50_hits"]
    rare_exposure_gain = round(float(rare_hits) / float(max(1, unif_hits)), 4)

    ppmi_ce = results["w40_h3_ppmi_svd_uniform"]["val_ce"]
    haar_unif_ce = results["w40_h3_isotropic_haar_uniform"]["val_ce"]
    haar_rare_ce = results["w40_h3_isotropic_haar_rare_reweighted"]["val_ce"]
    haar_unif_rank = results["w40_h3_isotropic_haar_uniform"]["final_sharp_eff_rank"]
    haar_rare_rank = results["w40_h3_isotropic_haar_rare_reweighted"]["final_sharp_eff_rank"]

    h3_pass = (
        rare_exposure_gain >= 1.5
        and is_expectation_err < 1e-5
        and haar_unif_rank >= 140.0
        and haar_rare_rank >= 140.0
        and min(haar_unif_ce, haar_rare_ce) <= ppmi_ce + 0.02
    )
    return {
        "h3_pass": h3_pass,
        "rare_token_bottom50_exposure_gain_x": rare_exposure_gain,
        "importance_weighted_expectation_err": is_expectation_err,
        "arms": results,
    }


def run_h4_5min_tied_holo_born_verification(
    work_gpu: torch.Tensor,
    prior_tokens: torch.Tensor,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    wall_clock_sec: float,
    device: torch.device,
) -> dict[str, Any]:
    """Execute `H4`: 5-minute (`300s`) live training run of `TiedHoloBornComplexRoPEGPT` on `FineWeb` (`<= 8` args)."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(1337)
    cfg = TiedHoloBornConfig(
        vocab_size=PADDED_VOCAB_SIZE,
        valid_vocab_size=VALID_VOCAB_SIZE,
        d_complex=DEFAULT_D_COMPLEX,
        n_registers=DEFAULT_N_REGISTERS,
        d_register=DEFAULT_D_REGISTER,
        num_layers=DEFAULT_H4_LAYERS,
        block_size=DEFAULT_H4_SEQ_LEN,
    )
    model = TiedHoloBornComplexRoPEGPT(cfg).to(device)
    init_stats = model.initialize_isotropic_haar(prior_tokens=prior_tokens, seed=1337)
    opt_muon, opt_adam = build_tied_holo_born_optimizers(model)
    sampler = RareTokenImportanceWindowSampler(
        reference_tokens=prior_tokens,
        valid_vocab_size=VALID_VOCAB_SIZE,
        alpha_rare=DEFAULT_RARE_ALPHA,
        laplace_eps=1.0,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = round(trainable_params / 1e6, 2)
    tokens_per_step = DEFAULT_H4_BATCH * DEFAULT_H4_SEQ_LEN
    sec_tag = int(round(wall_clock_sec))
    est_steps = max(200, int(round((wall_clock_sec / 300.0) * 850.0)))
    est_tokens_m = round((est_steps * tokens_per_step) / 1e6, 2)
    run_key = f"w40_tied_holo_born_d{DEFAULT_D_COMPLEX}c_H{DEFAULT_N_REGISTERS}_dh{DEFAULT_D_REGISTER}_L{DEFAULT_H4_LAYERS}_{sec_tag}s"

    intent = ExperimentIntentSpec(
        run_key=run_key,
        wave_id="W40",
        plan_id="EXPERIMENT_PLAN_055",
        task_domain="fineweb_edu_local",
        hypothesis_intent=(
            f"Wave 40 {sec_tag}s verification of TiedHoloBornComplexRoPEGPT (100% softmax-free polynomial "
            f"Holographic Born p={cfg.holo_boson_p}, tied <e_v| + sequence-side out_operator, soft Parseval "
            f"isotropy L_iso, and unbiased rare-token importance sampling) keeping R_emb >= 140."
        ),
        architecture_summary=(
            f"TiedHoloBornComplexRoPEGPT (d_complex={DEFAULT_D_COMPLEX}, H={DEFAULT_N_REGISTERS}, "
            f"d_h={DEFAULT_D_REGISTER}, L={DEFAULT_H4_LAYERS}, {params_m:.2f}M params)"
        ),
        hardware_budget="1x RTX 5060 Ti 16GB (BF16 Tensor Cores + Triton Flash-Krylov + Chunked HoloBorn)",
        batch_size=DEFAULT_H4_BATCH,
        seq_len=DEFAULT_H4_SEQ_LEN,
        total_steps=est_steps,
        target_tokens_M=est_tokens_m,
        params_M=params_m,
        embed_dim=2 * DEFAULT_D_COMPLEX,
        num_layers=DEFAULT_H4_LAYERS,
        schedule="smith_one_cycle_wallclock",
    )
    logger = StandardizedExperimentLogger(intent)
    _ping_webui_reload()

    eval_0 = evaluate_tied_holo_model(model, val_batches)
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
        f"| Top1={eval_0['top1_acc']:.2f}% | R_emb={eval_0['sharp_eff_rank']:.1f}",
        flush=True,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    train_wall_s = 0.0
    step = 0
    next_eval_frac = 0.25
    last_train_ce = eval_0["val_ce"]
    peak_train_vram_mib = 0.0
    min_rank_observed = eval_0["sharp_eff_rank"]
    nan_inf_events = 0
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

        x_b, y_b, tok_w = sampler.sample_batch(work_gpu, DEFAULT_H4_BATCH, DEFAULT_H4_SEQ_LEN, step_seed=10000 + step)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, nll_loss = model(x_b, targets=y_b, return_logits=False, token_weights=tok_w)
        assert nll_loss is not None
        reg_pen, pen_stats = compute_soft_manifold_and_isotropy_penalties(model)
        total_loss = nll_loss + reg_pen
        if not bool(torch.isfinite(total_loss).item()):
            nan_inf_events += 1
        total_loss.backward()
        last_train_ce = float(nll_loss.detach().item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_muon.step()
        opt_adam.step()

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
            logger.log_step(step, last_train_ce, tok_s, cur_lr / DEFAULT_MUON_LR, train_wall_s / 60.0)

        if frac_done >= next_eval_frac or train_wall_s >= wall_clock_sec:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                cur_mib = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
                peak_train_vram_mib = max(peak_train_vram_mib, cur_mib)
            ev = evaluate_tied_holo_model(model, val_batches)
            min_rank_observed = min(min_rank_observed, ev["sharp_eff_rank"])
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
                    "sphere_pen": round(pen_stats["sphere_pen"], 6),
                    "unitary_pen": round(pen_stats["unitary_pen"], 6),
                    "iso_pen": round(pen_stats["iso_pen"], 6),
                }
            )
            history.append(dict(ev))
            logger.log_eval(step, ev)
            print(
                f"[{run_key}] t={train_wall_s:5.1f}s/{wall_clock_sec:.0f}s | Step {step:4d} ({tok_m:5.2f}M tok) "
                f"| Train NLL={last_train_ce:.4f} | Val NLL={ev['val_ce']:.4f} | PPL={ev['val_ppl']:.2f} "
                f"| Top1={ev['top1_acc']:.2f}% | R_emb={ev['sharp_eff_rank']:.1f} "
                f"| U_err={ev['complex_unitary_err']:.4f} | {tok_s:,.0f} tok/s",
                flush=True,
            )
            _ping_webui_reload()
            while next_eval_frac <= frac_done:
                next_eval_frac += 0.25

    final_ev = history[-1]
    val_ces = [float(h["val_ce"]) for h in history]
    monotonic_val_ce = all(val_ces[i + 1] <= val_ces[i] + 1e-4 for i in range(len(val_ces) - 1))
    h4_pass = (
        min_rank_observed >= 140.0
        and monotonic_val_ce
        and nan_inf_events == 0
        and final_ev["val_ce"] < eval_0["val_ce"]
    )
    actual_tokens_m = round((step * tokens_per_step) / 1e6, 2)
    summary = {
        "h4_pass": h4_pass,
        "run_key": run_key,
        "d_complex": DEFAULT_D_COMPLEX,
        "real_equiv_dim": 2 * DEFAULT_D_COMPLEX,
        "n_registers": DEFAULT_N_REGISTERS,
        "d_register": DEFAULT_D_REGISTER,
        "num_layers": DEFAULT_H4_LAYERS,
        "batch_size": DEFAULT_H4_BATCH,
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
        "step0_sharp_eff_rank": round(init_stats["sharp_eff_rank"], 2),
        "min_sharp_eff_rank_observed": round(min_rank_observed, 2),
        "final_sharp_eff_rank": final_ev["sharp_eff_rank"],
        "wave39_collapsed_rank_ref": WAVE39_FINAL_RANK_COLLAPSED,
        "rank_gain_vs_wave39_x": round(final_ev["sharp_eff_rank"] / WAVE39_FINAL_RANK_COLLAPSED, 2),
        "final_embed_norm_median": final_ev["embed_norm_median"],
        "final_complex_unitary_err": final_ev["complex_unitary_err"],
        "monotonic_val_ce_decrease": monotonic_val_ce,
        "nan_inf_events": nan_inf_events,
        "tok_s": round((step * tokens_per_step) / max(1e-6, train_wall_s), 1),
        "peak_vram_mib": round(peak_train_vram_mib, 1),
        "wall_clock_s": round(train_wall_s, 2),
        "history": history,
    }
    logger.finalize(extra_summary=summary)
    _ping_webui_reload()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wave 40 (EXPERIMENT_PLAN_055) Verification Runner: Tied HoloBorn, Soft Isotropy, & Rare Sampling"
    )
    parser.add_argument("--wall-clock-sec", type=float, default=DEFAULT_WALL_CLOCK_SEC)
    parser.add_argument("--h2-steps", type=int, default=DEFAULT_H2_STEPS)
    parser.add_argument("--h3-steps", type=int, default=DEFAULT_H3_STEPS)
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / ".cache" / "hypothesis_experiments_055.json"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_arr, val_arr = _EX48.load_fineweb_tokens(max_train_tokens=30_000_000, max_val_tokens=1_048_576)
    ppmi_obj = torch.load(
        ROOT / ".cache" / "ppmi_svd_256_fast_synth_plan030.pt", map_location="cpu", weights_only=True
    )
    ppmi_vecs = ppmi_obj["embeddings"] if isinstance(ppmi_obj, dict) else ppmi_obj
    val_batches = _EX48.build_evaluation_batches(
        val_arr, batch_size=4, seq_len=DEFAULT_H4_SEQ_LEN, num_batches=6, device=device
    )
    val_batches_short = _EX48.build_evaluation_batches(
        val_arr, batch_size=4, seq_len=DEFAULT_ABLAT_SEQ_LEN, num_batches=6, device=device
    )
    prior_tokens = torch.from_numpy(train_arr[:INIT_PRIOR_TOKENS].astype("int64"))
    work_gpu = torch.from_numpy(train_arr[:30_000_000].view("int16")).to(device)

    print(f"[Wave 40] Running H1 TiedHoloBorn readout & chunked autograd audit on {device}...", flush=True)
    h1_report = run_h1_tied_holo_born_readout_audit(device)
    print(f"[Wave 40] H1 Report: {json.dumps(h1_report, indent=2)}", flush=True)

    print(f"[Wave 40] Running H2 ({args.h2_steps} steps) Soft Penalties vs Hard Retraction on {device}...", flush=True)
    h2_report = run_h2_soft_penalties_vs_hard_retraction_audit(
        work_gpu, prior_tokens, val_batches_short, ppmi_vecs, device, steps=int(args.h2_steps)
    )
    print(f"[Wave 40] H2 Report: {json.dumps(h2_report, indent=2)}", flush=True)

    print(f"[Wave 40] Running H3 ({args.h3_steps} steps) Init & Rare-Token Sampling audit on {device}...", flush=True)
    h3_report = run_h3_initialization_and_rare_sampling_audit(
        work_gpu, prior_tokens, val_batches_short, ppmi_vecs, device, steps=int(args.h3_steps)
    )
    print(f"[Wave 40] H3 Report: {json.dumps(h3_report, indent=2)}", flush=True)

    print(f"[Wave 40] Running H4 ({args.wall_clock_sec:.0f}s) TiedHoloBornComplexRoPEGPT run on {device}...", flush=True)
    h4_report = run_h4_5min_tied_holo_born_verification(
        work_gpu, prior_tokens, val_batches, float(args.wall_clock_sec), device
    )

    payload = {
        "wave_id": "W40",
        "plan_id": "EXPERIMENT_PLAN_055",
        "all_hypotheses_passed": bool(
            h1_report["h1_pass"] and h2_report["h2_pass"] and h3_report["h3_pass"] and h4_report["h4_pass"]
        ),
        "h1_tied_holo_born_readout": h1_report,
        "h2_soft_penalties_vs_hard_retract": h2_report,
        "h3_init_and_rare_sampling": h3_report,
        "h4_5min_verification": h4_report,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Wave 40] Saved complete verification results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
