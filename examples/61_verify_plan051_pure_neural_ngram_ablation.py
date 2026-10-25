#!/usr/bin/env python3
"""Wave 36 (`EXPERIMENT_PLAN_051.md`): Pure-Neural Head-to-Head N-Gram Ablation (`Option 1` vs `Option 2`).

Compares two 100% pure-neural configurations (both with `_kn_state = None`, removing the 85M-token
non-differentiable CSR Kneser-Ney lookup table and post-hoc evaluation logit blend):
- `Option 1` (`disable_kn_lookup()`): Keeps differentiable `HashedSurfaceNGramProjector.forward`
  (`unigram_table`, `bigram_table`, `trigram_table` lifted onto `S^{d-1}` via Stiefel `u_ngram`).
- `Option 2` (`disable_projector()`): Disables & freezes `HashedSurfaceNGramProjector` completely
  (`-12.65M` fewer AdamW parameters), relying strictly on `MultiRateWavefunctionConvBank` `(4,2,1,1)`
  + `SubspaceProjectorBornAttention` + `Tied2Pow16WaveletCholeskyPyramid`.

Evaluates both options head-to-head at `d = 256, L = 6` (`B = 16`) and `d = 1024, L = 6` Mode B
(`B_micro = 8, grad_accum = 2`) over `300` steps (`2.46M` tokens) on `FineWeb-Edu`.
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
from qdlblocks.models.level_separated_wavelet_conv_gpt import (  # noqa: E402
    LevelSeparatedWaveletConvConfig,
    LevelSeparatedWaveletConvGPT,
    PreAttnMultiRateWaveletConvBlock,
    build_level_separated_wavelet_conv_optimizers,
    initialize_level_separated_wavelet_conv_with_ppmi_svd,
    update_parseval_mup_ngram_learning_rates,
)

_EX48_SPEC = importlib.util.spec_from_file_location(
    "plan038_runner", ROOT / "examples" / "48_verify_plan038_last_layer_measurement_and_traceless.py"
)
assert _EX48_SPEC is not None and _EX48_SPEC.loader is not None
_EX48 = importlib.util.module_from_spec(_EX48_SPEC)
sys.modules["plan038_runner"] = _EX48
_EX48_SPEC.loader.exec_module(_EX48)

_EX53_SPEC = importlib.util.spec_from_file_location(
    "plan043_runner", ROOT / "examples" / "53_verify_plan043_parseval_sqrt_d_noclamp_readout.py"
)
assert _EX53_SPEC is not None and _EX53_SPEC.loader is not None
_EX53 = importlib.util.module_from_spec(_EX53_SPEC)
sys.modules["plan043_runner"] = _EX53
_EX53_SPEC.loader.exec_module(_EX53)

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95, device=0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Documented in EXPERIMENT_PLAN_051.md (Wave 36)
# ==============================================================================
DEFAULT_ABLATION_STEPS: int = 300
PURE_NEURAL_INIT_TOKENS: int = 3_500_000
TRAIN_EVAL_PARITY_TOL: float = 5e-3
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_SEQ_LEN: int = 512
DEFAULT_NUM_LAYERS: int = 6
BASE_MUON_ETA_MIN: float = 0.0065
BASE_MUON_ETA_MAX: float = 0.026
BASE_WAV_LR: float = 0.0085
BASE_HEAD_LR: float = 0.005
DEFAULT_COINCIDENCE_GAMMA: float = 0.22
DEFAULT_SUBSPACE_VAR_PENALTY: float = 0.14
WEBUI_RELOAD_URL: str = "http://127.0.0.1:8765/api/reload"


@dataclass(frozen=True)
class PureNeuralVariantSpec:
    """Specification for a single Wave 36 pure-neural ablation run (`<= 8` fields)."""

    run_key: str
    option_name: str
    embed_dim: int
    num_layers: int
    batch_size: int
    grad_accum_steps: int
    steps: int
    disable_diff_ngram: bool


def _ping_webui_reload(url: str = WEBUI_RELOAD_URL) -> None:
    """Trigger non-blocking reload on the local `:8765` WebUI daemon."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=1.0):
            pass
    except Exception:
        pass


@torch.no_grad()
def evaluate_pure_neural_model(
    model: LevelSeparatedWaveletConvGPT,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    """Evaluate 100% pure-neural `val_ce`, `val_ppl`, `top1_acc`, `top50_acc`, and `R_sharp^{(m)}`."""
    model.eval()
    assert getattr(model.ngram_proj, "_kn_state", None) is None
    v_valid = int(model.cfg.valid_vocab_size)
    total_loss = 0.0
    total_top1 = 0
    total_top50 = 0
    total_tokens = 0
    dev_type = val_batches[0][0].device.type

    for x_val, y_val in val_batches:
        with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=dev_type == "cuda"):
            logits, _ = model(x_val, targets=None, return_logits=True)
        assert logits is not None
        logits_valid = logits[..., :v_valid].float()
        y_flat = y_val.reshape(-1)
        flat_logits = logits_valid.reshape(-1, v_valid)
        loss = F.cross_entropy(flat_logits, y_flat, reduction="sum")
        total_loss += float(loss.item())
        preds = flat_logits.argmax(dim=-1)
        total_top1 += int((preds == y_flat).sum().item())
        top50_idx = torch.topk(flat_logits, k=min(50, v_valid), dim=-1).indices
        total_top50 += int((top50_idx == y_flat.unsqueeze(-1)).any(dim=-1).sum().item())
        total_tokens += int(y_flat.numel())

    mean_ce = total_loss / max(1, total_tokens)
    m_sharp, _, _ = model.synthesize_detectors(use_checkpoint=False)
    sharp_rank = model.compute_sharp_eff_rank(m_sharp)
    level_stats = model.inspect_level_energy_and_rank()
    c_spec = model.build_coincidence_spec()

    out: dict[str, float] = {
        "val_ce": round(mean_ce, 4),
        "val_ppl": round(math.exp(min(20.0, mean_ce)), 2),
        "top1_acc": round(total_top1 / max(1, total_tokens), 4),
        "top50_acc": round(total_top50 / max(1, total_tokens), 4),
        "val_top1": round(100.0 * total_top1 / max(1, total_tokens), 2),
        "val_top50": round(100.0 * total_top50 / max(1, total_tokens), 2),
        "sharp_eff_rank": round(sharp_rank, 2),
        "diffuse_eff_rank": round(float(model._last_diffuse_eff_rank), 2),
        "mean_beta": round(float(model.current_beta()), 3),
        "coincidence_gamma_eff": round(float(c_spec.coincidence_gamma), 4),
        "subspace_var_penalty_eff": round(float(c_spec.subspace_var_penalty), 4),
        "kraus_stiefel_err": round(float(model.pyramid.compute_kraus_stiefel_err()), 6),
    }
    for k, v in level_stats.items():
        out[k] = round(float(v), 4)
    return out


def verify_h1_train_eval_parity_without_kn(
    device: torch.device,
    ppmi_embeddings: torch.Tensor,
    init_tokens: torch.Tensor,
    prior_tokens: torch.Tensor,
    val_batch: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    """Verify H1: Disabling `_kn_state` produces exact parity between chunked train CE and eval logit CE."""
    results: dict[str, Any] = {}
    x_b, y_b = val_batch
    for opt_label, disable_diff in (("option_1_diff_ngram", False), ("option_2_no_ngram", True)):
        t_init0 = time.perf_counter()
        cfg = LevelSeparatedWaveletConvConfig(
            embed_dim=256,
            num_layers=6,
            num_heads=4,
            block_size=DEFAULT_SEQ_LEN,
            num_subspaces=4,
            coincidence_gamma=DEFAULT_COINCIDENCE_GAMMA,
            subspace_var_penalty=DEFAULT_SUBSPACE_VAR_PENALTY,
            use_coincidence_warmup=True,
        )
        torch.manual_seed(42)
        model = LevelSeparatedWaveletConvGPT(cfg).to(device)
        if disable_diff:
            model.ngram_proj.disable_projector()
        else:
            model.ngram_proj.disable_kn_lookup()
        initialize_level_separated_wavelet_conv_with_ppmi_svd(model, ppmi_embeddings, init_tokens, prior_tokens)
        init_sec = time.perf_counter() - t_init0

        model.eval()
        max_diff = 0.0
        for tau in (0.0, 0.5, 1.0):
            model.set_training_progress(tau)
            with torch.no_grad(), torch.amp.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                _, chunked_ce = model(x_b, targets=y_b, return_logits=False)
                logits, _ = model(x_b, targets=None, return_logits=True)
            assert chunked_ce is not None and logits is not None
            v_valid = int(model.cfg.valid_vocab_size)
            eval_ce = F.cross_entropy(logits[..., :v_valid].float().reshape(-1, v_valid), y_b.reshape(-1))
            max_diff = max(max_diff, abs(float(chunked_ce.item()) - float(eval_ce.item())))

        active_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results[opt_label] = {
            "active_params_M": round(active_params / 1e6, 4),
            "init_seconds": round(init_sec, 2),
            "max_train_eval_ce_diff": round(max_diff, 6),
            "kn_state_is_none": getattr(model.ngram_proj, "_kn_state", None) is None,
            "pass": bool(max_diff <= TRAIN_EVAL_PARITY_TOL and init_sec <= 15.0),
        }
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    h1_pass = bool(
        results["option_1_diff_ngram"]["pass"] and results["option_2_no_ngram"]["pass"]
    )
    print(
        f"[H1 Audit] Opt1 diff={results['option_1_diff_ngram']['max_train_eval_ce_diff']:.2e} "
        f"({results['option_1_diff_ngram']['init_seconds']:.2f}s init, "
        f"{results['option_1_diff_ngram']['active_params_M']:.2f}M params) | "
        f"Opt2 diff={results['option_2_no_ngram']['max_train_eval_ce_diff']:.2e} "
        f"({results['option_2_no_ngram']['init_seconds']:.2f}s init, "
        f"{results['option_2_no_ngram']['active_params_M']:.2f}M params) | PASS={h1_pass}",
        flush=True,
    )
    results["hypothesis_1_pass"] = h1_pass
    return results


def run_pure_neural_variant(
    spec: PureNeuralVariantSpec,
    device: torch.device,
    ppmi_embeddings: torch.Tensor,
    init_tokens: torch.Tensor,
    prior_tokens: torch.Tensor,
    work_gpu: torch.Tensor,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, Any]:
    """Run a single pure-neural (`_kn_state = None`) variant (`Option 1` or `Option 2`)."""
    torch._dynamo.reset()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    t_init0 = time.perf_counter()
    num_heads = max(4, spec.embed_dim // 64)
    cfg = LevelSeparatedWaveletConvConfig(
        embed_dim=spec.embed_dim,
        num_layers=spec.num_layers,
        num_heads=num_heads,
        block_size=DEFAULT_SEQ_LEN,
        num_subspaces=4,
        coincidence_gamma=DEFAULT_COINCIDENCE_GAMMA,
        subspace_var_penalty=DEFAULT_SUBSPACE_VAR_PENALTY,
        use_coincidence_warmup=True,
    )
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    model = LevelSeparatedWaveletConvGPT(cfg).to(device)
    if spec.disable_diff_ngram:
        model.ngram_proj.disable_projector()
    else:
        model.ngram_proj.disable_kn_lookup()

    init_stats = initialize_level_separated_wavelet_conv_with_ppmi_svd(
        model=model,
        ppmi_embeddings=ppmi_embeddings,
        init_tokens=init_tokens,
        prior_tokens=prior_tokens,
    )
    assert getattr(model.ngram_proj, "_kn_state", None) is None
    opt_muon, opt_adam = build_level_separated_wavelet_conv_optimizers(
        model,
        base_muon_lr=BASE_MUON_ETA_MAX,
        base_wav_lr=BASE_WAV_LR,
        base_head_lr=BASE_HEAD_LR,
    )
    init_sec = time.perf_counter() - t_init0

    active_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = round(active_params / 1e6, 4)
    muon_params_m = round(sum(p.numel() for g in opt_muon.param_groups for p in g["params"]) / 1e6, 4)
    ngram_params_m = round(
        sum(p.numel() for g in opt_adam.param_groups if g.get("group_name") == "surface_ngram" for p in g["params"])
        / 1e6,
        4,
    )
    batch_size = spec.batch_size
    grad_accum = max(1, spec.grad_accum_steps)
    micro_batch = max(1, batch_size // grad_accum)
    tok_per_step = batch_size * DEFAULT_SEQ_LEN
    target_tok_m = round((spec.steps * tok_per_step) / 1e6, 2)

    intent = ExperimentIntentSpec(
        run_key=spec.run_key,
        wave_id="W36",
        plan_id="EXPERIMENT_PLAN_051",
        task_domain="fineweb_edu_local",
        hypothesis_intent=(
            f"Wave 36 Pure-Neural Ablation ({spec.option_name}, _kn_state=None): "
            f"d={spec.embed_dim}, L={spec.num_layers} ({params_m:.2f}M active params, "
            f"ngram_params={ngram_params_m:.2f}M) over {spec.steps} steps ({target_tok_m}M tok)."
        ),
        architecture_summary=(
            f"LevelSeparatedWaveletConvGPT ({spec.option_name}, d={spec.embed_dim}, L={spec.num_layers}, "
            f"H={model.cfg.num_heads}, M=4, pure neural)"
        ),
        hardware_budget="1x Local RTX 5060 Ti (16 GB VRAM)",
        batch_size=batch_size,
        seq_len=DEFAULT_SEQ_LEN,
        total_steps=spec.steps,
        target_tokens_M=target_tok_m,
        params_M=params_m,
        embed_dim=spec.embed_dim,
        num_layers=spec.num_layers,
        schedule="modded_nanogpt_wsd",
    )
    logger = StandardizedExperimentLogger(intent)
    _ping_webui_reload()

    model.set_training_progress(0.0)
    eval0 = evaluate_pure_neural_model(model, val_batches)
    eval0.update({"step": 0, "tokens_M": 0.0, "tau_progress": 0.0, "compute_tflops": 0.0, "tok_per_s": 0.0})
    logger.log_eval(0, eval0)
    step0_r_sharp = float(init_stats.get("sharp_eff_rank", eval0["sharp_eff_rank"]))

    print(
        f"\n[W36 START] {spec.run_key} ({spec.option_name}) | active_params={active_params:,} "
        f"({params_m:.2f}M, muon={muon_params_m:.2f}M, ngram={ngram_params_m:.2f}M) | "
        f"init={init_sec:.1f}s | d={spec.embed_dim}, L={spec.num_layers} | "
        f"B_micro={micro_batch}x{grad_accum} | steps={spec.steps} ({target_tok_m}M tok)",
        flush=True,
    )
    print(
        f"  [{spec.run_key}] step=  0/{spec.steps} (0.00M tok) | "
        f"val_ce={eval0['val_ce']:.4f} | top1={eval0['val_top1']:.2f}% | R_sharp={eval0['sharp_eff_rank']:.1f}",
        flush=True,
    )

    eval_milestones = {75, 150, 225, spec.steps}
    if spec.steps > 300:
        for s_m in range(300, spec.steps + 1, 300):
            eval_milestones.add(s_m)

    milestones: dict[str, dict[str, Any]] = {"step_0": dict(eval0)}
    best_val_ce = float(eval0["val_ce"])
    final_eval = dict(eval0)
    n_work = int(work_gpu.numel())
    t0 = time.perf_counter()

    for step in range(1, spec.steps + 1):
        model.train()
        tok_prog = min(1.0, float(step - 1) / float(max(1, spec.steps)))
        model.set_training_progress(tok_prog)
        lr_256, mom = _EX53.compute_step_lr_and_momentum(
            "modded_nanogpt_wsd", step - 1, spec.steps, BASE_MUON_ETA_MIN, BASE_MUON_ETA_MAX
        )
        update_parseval_mup_ngram_learning_rates(
            opt_muon, opt_adam, lr_256, base_muon_lr_256=BASE_MUON_ETA_MAX
        )
        for g in opt_muon.param_groups:
            g["momentum"] = mom
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)

        offset = ((step - 1) * tok_per_step) % max(1, n_work - tok_per_step - 2)
        c_i64 = work_gpu[offset : offset + tok_per_step + 1].to(torch.int64) & 0xFFFF
        x_full = c_i64[:-1].view(batch_size, DEFAULT_SEQ_LEN)
        y_full = c_i64[1:].view(batch_size, DEFAULT_SEQ_LEN)

        step_loss_val = 0.0
        if grad_accum == 1:
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, loss = model(x_full, targets=y_full, return_logits=False)
            assert loss is not None
            step_loss_val = float(loss.item())
            loss.backward()
        else:
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                m_sharp_step, m_diff_step, w_k_step = model._synthesize_combined_detectors(use_checkpoint=True)
            m_sharp_leaf = m_sharp_step.detach().requires_grad_(True)
            m_diff_leaf = m_diff_step.detach().requires_grad_(True)
            w_k_leaf = w_k_step.detach()
            for mb_idx in range(grad_accum):
                mb_s = mb_idx * micro_batch
                mb_e = min(batch_size, mb_s + micro_batch)
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    _, mb_loss = model(
                        x_full[mb_s:mb_e],
                        targets=y_full[mb_s:mb_e],
                        return_logits=False,
                        precomputed_detectors=(m_sharp_leaf, m_diff_leaf, w_k_leaf),
                    )
                assert mb_loss is not None
                step_loss_val += float(mb_loss.item()) / float(grad_accum)
                (mb_loss / float(grad_accum)).backward()
            if m_sharp_leaf.grad is not None and m_diff_leaf.grad is not None:
                torch.autograd.backward([m_sharp_step, m_diff_step], [m_sharp_leaf.grad, m_diff_leaf.grad])
            del m_sharp_step, m_diff_step, w_k_step, m_sharp_leaf, m_diff_leaf, w_k_leaf

        for g in (*opt_muon.param_groups, *opt_adam.param_groups):
            torch.nn.utils.clip_grad_norm_(g["params"], 1.0)
        opt_muon.step()
        opt_adam.step()
        if step % 2 == 0 or step == spec.steps:
            model.retract_kraus_generators()

        elapsed = max(time.perf_counter() - t0, 1e-6)
        tok_seen = step * tok_per_step
        tok_s = tok_seen / elapsed
        tok_m = tok_seen / 1e6

        if step % 25 == 0 or step in eval_milestones:
            logger.log_step(step, step_loss_val, tok_s, lr_256 / BASE_MUON_ETA_MAX, elapsed / 60.0)

        if step in eval_milestones:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tau_eval = min(1.0, float(step) / float(max(1, spec.steps)))
            model.set_training_progress(tau_eval)
            ev = evaluate_pure_neural_model(model, val_batches)
            peak_mib = (
                float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
                if device.type == "cuda"
                else 0.0
            )
            ev.update(
                {
                    "step": step,
                    "tokens_M": round(tok_m, 2),
                    "train_ce": round(step_loss_val, 4),
                    "compute_tflops": round(6.0 * params_m * tok_m, 4),
                    "tau_progress": round(tau_eval, 4),
                    "elapsed_min": round(elapsed / 60.0, 4),
                    "tok_per_s": round(tok_s, 1),
                    "peak_vram_mib": round(peak_mib, 1),
                }
            )
            best_val_ce = min(best_val_ce, float(ev["val_ce"]))
            final_eval = dict(ev)
            logger.log_eval(step, ev)
            milestones[f"step_{step}"] = dict(ev)
            _ping_webui_reload()
            print(
                f"  [{spec.run_key}] step={step:3d}/{spec.steps} ({tok_m:4.2f}M tok, tau={tau_eval:.2f}) | "
                f"train_ce={step_loss_val:.4f} | val_ce={ev['val_ce']:.4f} (best={best_val_ce:.4f}) | "
                f"top1={ev['val_top1']:.2f}% | R_sharp={ev['sharp_eff_rank']:.1f} "
                f"(sub8k={ev['r_sharp_sub_8192']:.1f}, fine64k={ev['r_sharp_fine_65536']:.1f}) | "
                f"vram={peak_mib:.0f}MiB | {tok_s:,.0f} tok/s",
                flush=True,
            )

    peak_vram_mib = (
        round(float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0), 1)
        if device.type == "cuda"
        else 0.0
    )
    pre_attn_blocks = sum(1 for b in model.blocks if isinstance(b, PreAttnMultiRateWaveletConvBlock))
    summary = {
        "run_key": spec.run_key,
        "option_name": spec.option_name,
        "embed_dim": spec.embed_dim,
        "num_layers": spec.num_layers,
        "num_heads": int(model.cfg.num_heads),
        "batch_size": batch_size,
        "micro_batch_size": micro_batch,
        "grad_accum_steps": grad_accum,
        "steps": spec.steps,
        "tokens_M": target_tok_m,
        "active_params_M": params_m,
        "muon_params_M": muon_params_m,
        "ngram_params_M": ngram_params_m,
        "pre_attn_conv_blocks": pre_attn_blocks,
        "init_seconds": round(init_sec, 2),
        "step0_r_sharp": round(step0_r_sharp, 2),
        "step0_val_ce": round(float(eval0["val_ce"]), 4),
        "best_val_ce": round(best_val_ce, 4),
        "final_val_ce": round(float(final_eval["val_ce"]), 4),
        "final_train_ce": round(float(final_eval.get("train_ce", 0.0)), 4),
        "final_val_ppl": round(float(final_eval["val_ppl"]), 2),
        "final_top1_acc": round(float(final_eval["top1_acc"]), 4),
        "final_top50_acc": round(float(final_eval["top50_acc"]), 4),
        "final_sharp_eff_rank": round(float(final_eval["sharp_eff_rank"]), 2),
        "r_sharp_coarse_256": round(float(final_eval["r_sharp_coarse_256"]), 2),
        "r_sharp_mid_2048": round(float(final_eval["r_sharp_mid_2048"]), 2),
        "r_sharp_sub_8192": round(float(final_eval["r_sharp_sub_8192"]), 2),
        "r_sharp_fine_65536": round(float(final_eval["r_sharp_fine_65536"]), 2),
        "peak_vram_mib": peak_vram_mib,
        "tok_per_s": round(float(final_eval.get("tok_per_s", 0.0)), 1),
        "milestones": milestones,
    }
    logger.finalize(extra_summary=summary)
    _ping_webui_reload()
    del model, opt_muon, opt_adam
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_ABLATION_STEPS)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_arr, val_arr = _EX48.load_fineweb_tokens(max_train_tokens=30_000_000, max_val_tokens=1_048_576)
    ppmi_obj = torch.load(
        ROOT / ".cache" / "ppmi_svd_256_fast_synth_plan030.pt", map_location="cpu", weights_only=True
    )
    ppmi_embeddings = ppmi_obj["embeddings"] if isinstance(ppmi_obj, dict) else ppmi_obj
    val_batches = _EX48.build_evaluation_batches(
        val_arr, batch_size=4, seq_len=DEFAULT_SEQ_LEN, num_batches=6, device=device
    )
    init_tokens = torch.from_numpy(train_arr[:PURE_NEURAL_INIT_TOKENS].astype("int32"))
    prior_tokens = torch.from_numpy(train_arr[:2_000_000].astype("int64"))
    work_gpu = torch.from_numpy(train_arr[:30_000_000].view("int16")).to(device)

    h1_report = verify_h1_train_eval_parity_without_kn(
        device, ppmi_embeddings, init_tokens, prior_tokens, val_batches[0]
    )

    s = int(args.steps)
    specs = [
        PureNeuralVariantSpec(
            run_key=f"w36_opt1_diff_ngram_d256_L6_{s}steps",
            option_name="Option 1 (Diff N-Gram Only, No _kn_state)",
            embed_dim=256,
            num_layers=DEFAULT_NUM_LAYERS,
            batch_size=DEFAULT_BATCH_SIZE,
            grad_accum_steps=1,
            steps=s,
            disable_diff_ngram=False,
        ),
        PureNeuralVariantSpec(
            run_key=f"w36_opt2_no_ngram_d256_L6_{s}steps",
            option_name="Option 2 (No N-Gram At All, Pure Wavelet+Conv)",
            embed_dim=256,
            num_layers=DEFAULT_NUM_LAYERS,
            batch_size=DEFAULT_BATCH_SIZE,
            grad_accum_steps=1,
            steps=s,
            disable_diff_ngram=True,
        ),
        PureNeuralVariantSpec(
            run_key=f"w36_opt1_diff_ngram_d1024_L6_mode_b_{s}steps",
            option_name="Option 1 (Diff N-Gram Only, No _kn_state, d=1024 Mode B)",
            embed_dim=1024,
            num_layers=DEFAULT_NUM_LAYERS,
            batch_size=DEFAULT_BATCH_SIZE,
            grad_accum_steps=2,
            steps=s,
            disable_diff_ngram=False,
        ),
        PureNeuralVariantSpec(
            run_key=f"w36_opt2_no_ngram_d1024_L6_mode_b_{s}steps",
            option_name="Option 2 (No N-Gram At All, Pure Wavelet+Conv, d=1024 Mode B)",
            embed_dim=1024,
            num_layers=DEFAULT_NUM_LAYERS,
            batch_size=DEFAULT_BATCH_SIZE,
            grad_accum_steps=2,
            steps=s,
            disable_diff_ngram=True,
        ),
    ]

    runs: dict[str, dict[str, Any]] = {}
    out_path = ROOT / ".cache" / "hypothesis_experiments_051.json"
    for sp in specs:
        res = run_pure_neural_variant(
            sp, device, ppmi_embeddings, init_tokens, prior_tokens, work_gpu, val_batches
        )
        runs[sp.run_key] = res
        partial = {
            "plan_id": "EXPERIMENT_PLAN_051",
            "wave_id": "W36",
            "h1_train_eval_parity_without_kn": h1_report,
            "h2_pure_neural_head_to_head": runs,
        }
        out_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")

    r_opt1_256 = runs[specs[0].run_key]
    r_opt2_256 = runs[specs[1].run_key]
    r_opt1_1024 = runs[specs[2].run_key]
    r_opt2_1024 = runs[specs[3].run_key]

    comparison = {
        "d256_opt2_minus_opt1_val_ce": round(r_opt2_256["best_val_ce"] - r_opt1_256["best_val_ce"], 4),
        "d1024_opt2_minus_opt1_val_ce": round(r_opt2_1024["best_val_ce"] - r_opt1_1024["best_val_ce"], 4),
        "opt1_d1024_minus_d256_val_ce": round(r_opt1_1024["best_val_ce"] - r_opt1_256["best_val_ce"], 4),
        "opt2_d1024_minus_d256_val_ce": round(r_opt2_1024["best_val_ce"] - r_opt2_256["best_val_ce"], 4),
    }
    full_report = {
        "plan_id": "EXPERIMENT_PLAN_051",
        "wave_id": "W36",
        "h1_train_eval_parity_without_kn": h1_report,
        "h2_pure_neural_head_to_head": runs,
        "comparison_summary": comparison,
    }
    out_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"\n[W36 COMPLETE] Saved report to {out_path}", flush=True)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
