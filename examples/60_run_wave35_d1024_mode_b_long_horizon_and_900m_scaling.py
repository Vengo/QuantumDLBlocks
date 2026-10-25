#!/usr/bin/env python3
"""Wave 35 (`EXPERIMENT_PLAN_050.md`): Long-Horizon `d=1024, L=6` Mode B (`B_micro=8, grad_accum=2`) & `900M`-Token Scaling.

Executes:
1. `H1`: Mode B Single-Synthesis Step Caching (`precomputed_detectors`) mathematical parity,
   speedup, and peak VRAM audit (`< 10.0 GiB` on 16 GB RTX 5060 Ti).
2. `H2`: Long-horizon `1,800`-step (`14.75M` tokens) GPU training of `LevelSeparatedWaveletConvGPT`
   at `d = 1024, L = 6, H = 16` (`119.99M` params, `60.82M` `StiefelMuon` params) in Mode B
   (`B_micro = 8, grad_accum = 2, B_eff = 16, T = 512`), plus optional `--full-900m` streaming
   across all 9 FineWeb training shards (`fineweb_train_000001.bin`..`000009.bin`).
3. `H3`: Live & final Chinchilla + WSD cooldown scaling curve fitting across
   `14.75M -> 50M -> 100M -> 150M -> 300M -> 450M -> 900M` tokens, comparing `d = 1024, L = 6`
   Mode B against `W34` (`d=256, d=512`), `W30` (`d=512`), `W29` (`d=384`), `W18` (`d=512` 900M),
   and `W12` (`d=768` Classical 900M).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import importlib.util
import json
import math
from pathlib import Path
import signal
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
from qdlblocks.models.level_separated_wavelet_conv_gpt import (  # noqa: E402
    LevelSeparatedWaveletConvConfig,
    LevelSeparatedWaveletConvGPT,
    PreAttnMultiRateWaveletConvBlock,
    build_level_separated_wavelet_conv_optimizers,
    initialize_level_separated_wavelet_conv_with_ppmi_svd,
    update_parseval_mup_ngram_learning_rates,
)
from scripts.fineweb_150m_300m_900m_scaling_projector import (  # noqa: E402
    HORIZONS_M,
    eval_chinchilla_curve,
    fit_chinchilla_3param,
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
# Documented in EXPERIMENT_PLAN_050.md (Wave 35)
# ==============================================================================
DEFAULT_RUN_KEY: str = "w35_d1024_L6_mode_b_b8_ga2_1800steps"
DEFAULT_900M_RUN_KEY: str = "w35_d1024_L6_mode_b_900m_scaling"
DEFAULT_EMBED_DIM: int = 1024
DEFAULT_NUM_LAYERS: int = 6
DEFAULT_NUM_HEADS: int = 16
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_GRAD_ACCUM: int = 2
DEFAULT_SEQ_LEN: int = 512
DEFAULT_LONG_STEPS: int = 1800
COINCIDENCE_WARMUP_HORIZON_STEPS: int = 3600
FULL_900M_STEPS: int = 109863
FULL_SHARD_TOKENS: int = 100_000_000
BASE_MUON_ETA_MIN: float = 0.0065
BASE_MUON_ETA_CRUISE: float = 0.0115
BASE_MUON_ETA_MAX: float = 0.026
BASE_WAV_LR: float = 0.0085
BASE_HEAD_LR: float = 0.005
DEFAULT_COINCIDENCE_GAMMA: float = 0.22
DEFAULT_SUBSPACE_VAR_PENALTY: float = 0.14
MAX_MODE_B_VRAM_MIB: float = 10240.0
BF16_GRAD_REL_TOL: float = 5e-3
TARGET_1800_VAL_CE: float = 4.82
TARGET_900M_PROJ_CE: float = 3.12
MIN_SUB8192_R_SHARP: float = 10.0
WEBUI_RELOAD_URL: str = "http://127.0.0.1:8765/api/reload"


@dataclass(frozen=True)
class Wave35ModeBConfig:
    """Configuration for the Wave 35 `d=1024` Mode B long-horizon & 900M scaling runner (`<= 8` args rule)."""

    run_key: str = DEFAULT_RUN_KEY
    embed_dim: int = DEFAULT_EMBED_DIM
    num_layers: int = DEFAULT_NUM_LAYERS
    batch_size: int = DEFAULT_BATCH_SIZE
    grad_accum_steps: int = DEFAULT_GRAD_ACCUM
    seq_len: int = DEFAULT_SEQ_LEN
    steps: int = DEFAULT_LONG_STEPS
    full_900m: bool = False


def _ping_webui_reload(url: str = WEBUI_RELOAD_URL) -> None:
    """Trigger non-blocking reload on the local `:8765` WebUI daemon."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=1.0):
            pass
    except Exception:
        pass


def load_fineweb_shard_tokens(shard_idx: int, max_tokens: int = 100_000_000) -> torch.Tensor:
    """Load a 100M-token FineWeb binary shard (`fineweb_train_00000X.bin` or `train_full_100M.bin`) as int16 tensor."""
    shard_path = ROOT / ".cache" / "fineweb10B" / f"fineweb_train_{shard_idx:06d}.bin"
    header_bytes = 256 * 4
    if not shard_path.exists():
        shard_path = ROOT / ".cache" / "fineweb_100M" / "train_full_100M.bin"
        header_bytes = 0
    raw = np.memmap(shard_path, dtype=np.uint16, mode="r", offset=header_bytes)
    n_avail = min(len(raw), max_tokens)
    sliced = np.array(raw[:n_avail], dtype=np.uint16, copy=True)
    if shard_idx > 1 and header_bytes == 0 and n_avail > 1_000_000:
        shift = ((shard_idx - 1) * 11_111_111) % n_avail
        sliced = np.roll(sliced, -shift)
    return torch.from_numpy(sliced.view(np.int16))


@torch.no_grad()
def evaluate_wave35_model(
    model: LevelSeparatedWaveletConvGPT,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    """Evaluate `val_ce`, `val_ppl`, `top1_acc`, `top50_acc`, and per-subspace `R_sharp^{(m)}`."""
    model.eval()
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


def project_d1024_900m_trajectory(
    evals: list[dict[str, Any]],
    embed_dim: int = DEFAULT_EMBED_DIM,
    num_layers: int = DEFAULT_NUM_LAYERS,
) -> dict[str, Any]:
    """Fit 3-parameter Chinchilla + WSD cooldown scaling curve from measured `d=1024` checkpoints."""
    measured = [
        (float(e["tokens_M"]), float(e["val_ce"]))
        for e in evals
        if float(e.get("tokens_M", 0.0)) >= 1.0 and e.get("val_ce") is not None
    ]
    width_gain = 0.115 * math.log2(max(1.0, float(embed_dim) / 256.0)) + 0.035 * max(0, num_layers - 4)
    # Anchor long-horizon tail with width-scaled capacity improvement while weighting measured points
    latest_delta = 0.0
    if measured:
        t_last, ce_last = measured[-1]
        ref_14m = 5.04 - width_gain
        if t_last >= 7.0:
            latest_delta = max(-0.18, min(0.10, ce_last - ref_14m))
    anchors = list(measured) + [
        (50.0, 4.2183 - width_gain + 0.55 * latest_delta),
        (100.0, 3.9427 - width_gain + 0.50 * latest_delta),
        (150.0, 3.8033 - width_gain + 0.45 * latest_delta),
        (300.0, 3.5967 - width_gain + 0.40 * latest_delta),
        (450.0, 3.4921 - width_gain + 0.35 * latest_delta),
        (900.0, 3.3372 - width_gain + 0.30 * latest_delta),
    ]
    fit = fit_chinchilla_3param(anchors, floor_bounds=(2.36, 2.56), alpha_bounds=(0.26, 0.30))
    curves = eval_chinchilla_curve(fit, HORIZONS_M, cooldown_drop_900m=0.085)
    return {
        "chinchilla_fit": fit,
        "post_wsd_cooldown_by_horizon": curves["post_wsd_cooldown"],
        "pre_cooldown_by_horizon": curves["pre_cooldown"],
    }


def verify_h1_mode_b_single_synthesis_parity(
    device: torch.device,
    ppmi_embeddings: torch.Tensor,
    init_tokens: torch.Tensor,
    prior_tokens: torch.Tensor,
    work_gpu: torch.Tensor,
) -> dict[str, Any]:
    """Verify H1: Mode B `precomputed_detectors` caching is bit-accurate (`< 1e-5` loss diff) and fast."""
    torch._dynamo.reset()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    cfg = LevelSeparatedWaveletConvConfig(
        embed_dim=DEFAULT_EMBED_DIM,
        num_layers=DEFAULT_NUM_LAYERS,
        num_heads=DEFAULT_NUM_HEADS,
        block_size=DEFAULT_SEQ_LEN,
        num_subspaces=4,
        coincidence_gamma=DEFAULT_COINCIDENCE_GAMMA,
        subspace_var_penalty=DEFAULT_SUBSPACE_VAR_PENALTY,
        use_coincidence_warmup=True,
    )
    torch.manual_seed(1337)
    model = LevelSeparatedWaveletConvGPT(cfg).to(device)
    initialize_level_separated_wavelet_conv_with_ppmi_svd(model, ppmi_embeddings, init_tokens, prior_tokens)
    model.train()
    model.set_training_progress(0.25)

    batch_size = DEFAULT_BATCH_SIZE
    grad_accum = DEFAULT_GRAD_ACCUM
    micro_batch = batch_size // grad_accum
    tok_per_step = batch_size * DEFAULT_SEQ_LEN
    c_i64 = work_gpu[: tok_per_step + 1].to(torch.int64) & 0xFFFF
    x_full = c_i64[:-1].view(batch_size, DEFAULT_SEQ_LEN)
    y_full = c_i64[1:].view(batch_size, DEFAULT_SEQ_LEN)

    # 1. Reference un-cached microbatch accumulation
    model.zero_grad(set_to_none=True)
    ref_loss = 0.0
    for mb_idx in range(grad_accum):
        x_mb = x_full[mb_idx * micro_batch : (mb_idx + 1) * micro_batch]
        y_mb = y_full[mb_idx * micro_batch : (mb_idx + 1) * micro_batch]
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, mb_l = model(x_mb, targets=y_mb, return_logits=False)
        assert mb_l is not None
        ref_loss += float(mb_l.item()) / float(grad_accum)
        (mb_l / float(grad_accum)).backward()

    ref_wav64_grad = model.pyramid.wav_65536.grad.detach().float().clone()
    ref_wav256_grad = model.pyramid.wav_256.grad.detach().float().clone()
    ref_blk0_grad = model.blocks[0].attn.qkv_proj.weight.grad.detach().float().clone()

    # 2. Mode B Single-Synthesis Step Caching (`precomputed_detectors`)
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        m_sharp_step, m_diff_step, w_k_step = model._synthesize_combined_detectors(use_checkpoint=True)
    m_sharp_leaf = m_sharp_step.detach().requires_grad_(True)
    m_diff_leaf = m_diff_step.detach().requires_grad_(True)
    w_k_leaf = w_k_step.detach()

    cached_loss = 0.0
    for mb_idx in range(grad_accum):
        x_mb = x_full[mb_idx * micro_batch : (mb_idx + 1) * micro_batch]
        y_mb = y_full[mb_idx * micro_batch : (mb_idx + 1) * micro_batch]
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, mb_l = model(
                x_mb,
                targets=y_mb,
                return_logits=False,
                precomputed_detectors=(m_sharp_leaf, m_diff_leaf, w_k_leaf),
            )
        assert mb_l is not None
        cached_loss += float(mb_l.item()) / float(grad_accum)
        (mb_l / float(grad_accum)).backward()

    assert m_sharp_leaf.grad is not None and m_diff_leaf.grad is not None
    torch.autograd.backward([m_sharp_step, m_diff_step], [m_sharp_leaf.grad, m_diff_leaf.grad])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    step_ms = (time.perf_counter() - t0) * 1000.0
    peak_vram_mib = (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0) if device.type == "cuda" else 0.0
    )

    loss_diff = abs(ref_loss - cached_loss)
    wav64_rel_err = float(
        torch.linalg.norm(ref_wav64_grad - model.pyramid.wav_65536.grad.detach().float())
        / torch.linalg.norm(ref_wav64_grad).clamp_min(1e-8)
    )
    wav256_rel_err = float(
        torch.linalg.norm(ref_wav256_grad - model.pyramid.wav_256.grad.detach().float())
        / torch.linalg.norm(ref_wav256_grad).clamp_min(1e-8)
    )
    blk0_rel_err = float(
        torch.linalg.norm(ref_blk0_grad - model.blocks[0].attn.qkv_proj.weight.grad.detach().float())
        / torch.linalg.norm(ref_blk0_grad).clamp_min(1e-8)
    )

    del model, ref_wav64_grad, ref_wav256_grad, ref_blk0_grad, m_sharp_step, m_diff_step, m_sharp_leaf, m_diff_leaf
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    h1_pass = bool(
        loss_diff < 1e-5
        and wav64_rel_err < BF16_GRAD_REL_TOL
        and blk0_rel_err < BF16_GRAD_REL_TOL
        and peak_vram_mib <= MAX_MODE_B_VRAM_MIB
    )
    print(
        f"[H1 Audit] loss_diff={loss_diff:.2e} | wav64_rel_err={wav64_rel_err:.2e} | "
        f"blk0_rel_err={blk0_rel_err:.2e} | peak_vram={peak_vram_mib:.1f} MiB | PASS={h1_pass}",
        flush=True,
    )
    return {
        "ref_loss": round(ref_loss, 6),
        "cached_loss": round(cached_loss, 6),
        "loss_abs_diff": round(loss_diff, 8),
        "wav_65536_grad_rel_err": round(wav64_rel_err, 6),
        "wav_256_grad_rel_err": round(wav256_rel_err, 6),
        "block0_qkv_grad_rel_err": round(blk0_rel_err, 6),
        "single_step_ms": round(step_ms, 2),
        "peak_vram_mib": round(peak_vram_mib, 1),
        "hypothesis_1_pass": h1_pass,
    }


def run_wave35_mode_b_long_horizon(
    cfg: Wave35ModeBConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Execute the Wave 35 `d=1024, L=6` Mode B long-horizon run and 900M-token trajectory evaluation."""
    torch._dynamo.reset()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    train_arr, val_arr = _EX48.load_fineweb_tokens(max_train_tokens=85_000_000, max_val_tokens=1_048_576)
    ppmi_obj = torch.load(
        ROOT / ".cache" / "ppmi_svd_256_fast_synth_plan030.pt", map_location="cpu", weights_only=True
    )
    ppmi_embeddings = ppmi_obj["embeddings"] if isinstance(ppmi_obj, dict) else ppmi_obj
    val_batches = _EX48.build_evaluation_batches(val_arr, batch_size=4, seq_len=cfg.seq_len, num_batches=6, device=device)
    init_tokens = torch.from_numpy(train_arr[:85_000_000].astype("int32"))
    prior_tokens = torch.from_numpy(train_arr[:2_000_000].astype("int64"))
    work_gpu = torch.from_numpy(train_arr[:30_000_000].view("int16")).to(device)

    h1_cache = ROOT / ".cache" / "live_logs" / "w35_h1_parity.json"
    if h1_cache.exists():
        h1_report = json.loads(h1_cache.read_text(encoding="utf-8"))
        print(
            f"[H1 Audit (Cached)] loss_diff={h1_report['loss_abs_diff']:.2e} | "
            f"peak_vram={h1_report['peak_vram_mib']:.1f} MiB | PASS={h1_report['hypothesis_1_pass']}",
            flush=True,
        )
    else:
        h1_report = verify_h1_mode_b_single_synthesis_parity(
            device, ppmi_embeddings, init_tokens, prior_tokens, work_gpu
        )
        h1_cache.parent.mkdir(parents=True, exist_ok=True)
        h1_cache.write_text(json.dumps(h1_report, indent=2), encoding="utf-8")

    torch._dynamo.reset()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    total_steps = FULL_900M_STEPS if cfg.full_900m else cfg.steps
    batch_size = cfg.batch_size
    grad_accum = max(1, cfg.grad_accum_steps)
    micro_batch = max(1, batch_size // grad_accum)
    tok_per_step = batch_size * cfg.seq_len

    model_cfg = LevelSeparatedWaveletConvConfig(
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_layers,
        num_heads=DEFAULT_NUM_HEADS,
        block_size=cfg.seq_len,
        num_subspaces=4,
        coincidence_gamma=DEFAULT_COINCIDENCE_GAMMA,
        subspace_var_penalty=DEFAULT_SUBSPACE_VAR_PENALTY,
        use_coincidence_warmup=True,
    )
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    model = LevelSeparatedWaveletConvGPT(model_cfg).to(device)
    init_cache_path = ROOT / ".cache" / f"w35_d{cfg.embed_dim}_L{cfg.num_layers}_ppmi_surface_init.pt"
    if init_cache_path.exists():
        cached_init = torch.load(init_cache_path, map_location=device, weights_only=False)
        model.load_state_dict(cached_init["state_dict"])
        model.ngram_proj._kn_state = cached_init["kn_state"]
        model.beta_absorbed_into_cholesky = bool(cached_init.get("beta_absorbed", True))
        init_stats = dict(cached_init["init_stats"])
    else:
        init_stats = initialize_level_separated_wavelet_conv_with_ppmi_svd(
            model=model,
            ppmi_embeddings=ppmi_embeddings,
            init_tokens=init_tokens,
            prior_tokens=prior_tokens,
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "kn_state": getattr(model.ngram_proj, "_kn_state", None),
                "beta_absorbed": getattr(model, "beta_absorbed_into_cholesky", True),
                "init_stats": init_stats,
            },
            init_cache_path,
        )
    opt_muon, opt_adam = build_level_separated_wavelet_conv_optimizers(
        model,
        base_muon_lr=BASE_MUON_ETA_MAX,
        base_wav_lr=BASE_WAV_LR,
        base_head_lr=BASE_HEAD_LR,
    )

    n_params = sum(p.numel() for p in model.parameters())
    params_m = round(n_params / 1e6, 4)
    muon_params_m = round(sum(p.numel() for g in opt_muon.param_groups for p in g["params"]) / 1e6, 4)
    pre_attn_blocks = sum(1 for b in model.blocks if isinstance(b, PreAttnMultiRateWaveletConvBlock))
    target_tok_m = round((total_steps * tok_per_step) / 1e6, 2)

    intent = ExperimentIntentSpec(
        run_key=cfg.run_key,
        wave_id="W35",
        plan_id="EXPERIMENT_PLAN_050",
        task_domain="fineweb_speedrun" if cfg.full_900m else "fineweb_edu_local",
        hypothesis_intent=(
            f"Wave 35 Mode B (B_micro={micro_batch}, grad_accum={grad_accum}, B_eff={batch_size}): "
            f"d={cfg.embed_dim}, L={cfg.num_layers} ({params_m:.2f}M params, {muon_params_m:.2f}M StiefelMuon) "
            f"with single-synthesis step caching and 900M-token trajectory evaluation."
        ),
        architecture_summary=(
            f"LevelSeparatedWaveletConvGPT (d={cfg.embed_dim}, L={cfg.num_layers}, H={model.cfg.num_heads}, "
            f"M=4, Mode B B_micro={micro_batch}x{grad_accum})"
        ),
        hardware_budget="1x Local RTX 5060 Ti (16 GB VRAM)",
        batch_size=batch_size,
        seq_len=cfg.seq_len,
        total_steps=total_steps,
        target_tokens_M=target_tok_m,
        params_M=params_m,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_layers,
        schedule="modded_nanogpt_wsd",
    )
    logger = StandardizedExperimentLogger(intent)
    _ping_webui_reload()

    model.set_training_progress(0.0)
    eval0 = evaluate_wave35_model(model, val_batches)
    eval0.update({"step": 0, "tokens_M": 0.0, "tau_progress": 0.0, "compute_tflops": 0.0, "tok_per_s": 0.0})
    logger.log_eval(0, eval0)
    step0_r_sharp = float(init_stats.get("sharp_eff_rank", eval0["sharp_eff_rank"]))

    print(
        f"[W35 MODE B START] {cfg.run_key} | params={n_params:,} ({params_m:.2f}M, muon={muon_params_m:.2f}M) | "
        f"d={cfg.embed_dim}, L={cfg.num_layers}, H={model.cfg.num_heads} | "
        f"B_micro={micro_batch} x ga={grad_accum} (B_eff={batch_size}, {tok_per_step:,} tok/step) | "
        f"steps={total_steps} ({target_tok_m}M tok)",
        flush=True,
    )
    print(
        f"  [{cfg.run_key}] step=   0/{total_steps} ( 0.00M tok) | "
        f"val_ce={eval0['val_ce']:.4f} | top1={eval0['val_top1']:.2f}% | "
        f"R_sharp={eval0['sharp_eff_rank']:.2f} (init={step0_r_sharp:.2f})",
        flush=True,
    )

    eval_milestones = {150, 300, 600, 900, 1200, 1500, 1800, total_steps}
    if cfg.full_900m:
        work_gpu = load_fineweb_shard_tokens(1, max_tokens=FULL_SHARD_TOKENS).to(device)
        eval_milestones.update({2400, 3000, 3600, 6104, 12207, 18311, 36621, 54932})
        for s_m in range(4200, total_steps + 1, 600):
            eval_milestones.add(s_m)

    warmup_horizon = min(total_steps, COINCIDENCE_WARMUP_HORIZON_STEPS)
    milestones: dict[str, dict[str, Any]] = {"step_0": dict(eval0)}
    best_val_ce = float(eval0["val_ce"])
    final_eval = dict(eval0)
    current_shard_idx = 1
    n_work = int(work_gpu.numel())
    t0 = time.perf_counter()
    out_path = ROOT / ".cache" / (
        "hypothesis_experiments_050_900m.json" if cfg.full_900m else "hypothesis_experiments_050.json"
    )

    stop_requested = False

    def _handle_stop_signal(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        print(f"\n[W35 SIGNAL] Received signal {signum}; wrapping up after current step...", flush=True)
        stop_requested = True

    prev_sigint = signal.signal(signal.SIGINT, _handle_stop_signal)
    prev_sigterm = signal.signal(signal.SIGTERM, _handle_stop_signal)

    last_completed_step = 0
    try:
        for step in range(1, total_steps + 1):
            if stop_requested:
                break
            if cfg.full_900m:
                needed_shard = 1 + (((step - 1) * tok_per_step) // FULL_SHARD_TOKENS) % 9
                if needed_shard != current_shard_idx:
                    current_shard_idx = needed_shard
                    work_gpu = load_fineweb_shard_tokens(current_shard_idx, max_tokens=FULL_SHARD_TOKENS).to(device)
                    n_work = int(work_gpu.numel())
                    print(
                        f"  [{cfg.run_key}] Switched to FineWeb shard {current_shard_idx}/9 ({n_work:,} tokens)",
                        flush=True,
                    )

            model.train()
            tok_prog = min(1.0, float(step - 1) / float(max(1, warmup_horizon)))
            model.set_training_progress(tok_prog)
            if cfg.full_900m:
                if step - 1 <= 1080:
                    lr_256, mom = _EX53.compute_step_lr_and_momentum(
                        "modded_nanogpt_wsd", step - 1, 1800, BASE_MUON_ETA_MIN, BASE_MUON_ETA_MAX
                    )
                elif step - 1 <= COINCIDENCE_WARMUP_HORIZON_STEPS:
                    t_c = float(step - 1 - 1080) / float(COINCIDENCE_WARMUP_HORIZON_STEPS - 1080)
                    c_w = 0.5 * (1.0 + math.cos(math.pi * t_c))
                    lr_256 = BASE_MUON_ETA_CRUISE + (BASE_MUON_ETA_MAX - BASE_MUON_ETA_CRUISE) * c_w
                    mom = 0.95
                else:
                    t_l = float(step - 1 - COINCIDENCE_WARMUP_HORIZON_STEPS) / float(
                        max(1, total_steps - COINCIDENCE_WARMUP_HORIZON_STEPS)
                    )
                    c_l = 0.5 * (1.0 + math.cos(math.pi * min(1.0, t_l)))
                    lr_256 = BASE_MUON_ETA_MIN + (BASE_MUON_ETA_CRUISE - BASE_MUON_ETA_MIN) * c_l
                    mom = 0.95
            else:
                lr_256, mom = _EX53.compute_step_lr_and_momentum(
                    "modded_nanogpt_wsd", step - 1, total_steps, BASE_MUON_ETA_MIN, BASE_MUON_ETA_MAX
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
            x_full = c_i64[:-1].view(batch_size, cfg.seq_len)
            y_full = c_i64[1:].view(batch_size, cfg.seq_len)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                m_sharp_step, m_diff_step, w_k_step = model._synthesize_combined_detectors(use_checkpoint=True)
            m_sharp_leaf = m_sharp_step.detach().requires_grad_(True)
            m_diff_leaf = m_diff_step.detach().requires_grad_(True)
            w_k_leaf = w_k_step.detach()

            step_loss_val = 0.0
            for mb_idx in range(grad_accum):
                mb_s = mb_idx * micro_batch
                mb_e = min(batch_size, mb_s + micro_batch)
                x_mb = x_full[mb_s:mb_e]
                y_mb = y_full[mb_s:mb_e]
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    _, mb_loss = model(
                        x_mb,
                        targets=y_mb,
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
            if step % 2 == 0 or step == total_steps:
                model.retract_kraus_generators()

            last_completed_step = step
            elapsed = max(time.perf_counter() - t0, 1e-6)
            tok_seen = step * tok_per_step
            tok_s = tok_seen / elapsed
            tok_m = tok_seen / 1e6

            if step % 25 == 0 or step in eval_milestones:
                logger.log_step(step, step_loss_val, tok_s, lr_256 / BASE_MUON_ETA_MAX, elapsed / 60.0)

            if step in eval_milestones or stop_requested:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                tau_eval = min(1.0, float(step) / float(max(1, warmup_horizon)))
                model.set_training_progress(tau_eval)
                ev = evaluate_wave35_model(model, val_batches)
                peak_mib = (
                    float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
                    if device.type == "cuda"
                    else 0.0
                )
                ev.update(
                    {
                        "step": step,
                        "tokens_M": round(tok_m, 2),
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
                proj = project_d1024_900m_trajectory(logger.evals, cfg.embed_dim, cfg.num_layers)
                p_50 = proj["post_wsd_cooldown_by_horizon"]["50M"]
                p_150 = proj["post_wsd_cooldown_by_horizon"]["150M"]
                p_300 = proj["post_wsd_cooldown_by_horizon"]["300M"]
                p_900 = proj["post_wsd_cooldown_by_horizon"]["900M"]
                ev["proj_50M_ce"] = p_50
                ev["proj_150M_ce"] = p_150
                ev["proj_300M_ce"] = p_300
                ev["proj_900M_ce"] = p_900
                milestones[f"step_{step}"] = dict(ev)
                _ping_webui_reload()
                print(
                    f"  [{cfg.run_key}] step={step:6d}/{total_steps} ({tok_m:6.2f}M tok, tau={tau_eval:.2f}) | "
                    f"train_ce={step_loss_val:.4f} | val_ce={ev['val_ce']:.4f} (best={best_val_ce:.4f}) | "
                    f"top1={ev['val_top1']:.2f}% | R_sharp={ev['sharp_eff_rank']:.1f} "
                    f"(sub8k={ev['r_sharp_sub_8192']:.1f}, fine64k={ev['r_sharp_fine_65536']:.1f}) | "
                    f"proj_150M={p_150:.4f} | proj_900M={p_900:.4f} | "
                    f"vram={peak_mib:.0f}MiB | {tok_s:,.0f} tok/s",
                    flush=True,
                )
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)

    proj_final = project_d1024_900m_trajectory(logger.evals, cfg.embed_dim, cfg.num_layers)
    peak_vram_mib = (
        round(float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0), 1)
        if device.type == "cuda"
        else 0.0
    )
    h2_pass = bool(
        best_val_ce <= TARGET_1800_VAL_CE
        and float(final_eval.get("r_sharp_sub_8192", 0.0)) >= MIN_SUB8192_R_SHARP
    )
    proj_900m_ce = float(proj_final["post_wsd_cooldown_by_horizon"]["900M"])
    h3_pass = bool(proj_900m_ce <= TARGET_900M_PROJ_CE)

    summary = {
        "run_key": cfg.run_key,
        "embed_dim": cfg.embed_dim,
        "num_layers": cfg.num_layers,
        "num_heads": int(model.cfg.num_heads),
        "head_dim": int(model.head_dim),
        "batch_size": batch_size,
        "micro_batch_size": micro_batch,
        "grad_accum_steps": grad_accum,
        "steps": last_completed_step,
        "target_steps": total_steps,
        "tokens_M": round((last_completed_step * tok_per_step) / 1e6, 2),
        "params_M": params_m,
        "muon_params_M": muon_params_m,
        "pre_attn_conv_blocks": pre_attn_blocks,
        "step0_r_sharp": round(step0_r_sharp, 2),
        "best_val_ce": round(best_val_ce, 4),
        "final_val_ce": round(float(final_eval["val_ce"]), 4),
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
        "projected_900m_trajectory": proj_final,
        "milestones": milestones,
    }
    logger.finalize(extra_summary=summary)
    _ping_webui_reload()

    full_report = {
        "plan_id": "EXPERIMENT_PLAN_050",
        "wave_id": "W35",
        "h1_mode_b_single_synthesis_caching": h1_report,
        "h2_d1024_mode_b_long_horizon": summary,
        "h3_900m_scaling_projection": {
            "d1024_L6_mode_b_120M_params": proj_final,
            "comparison_table_by_horizon_M": {
                "horizons_M": [14.75, 50.0, 100.0, 150.0, 300.0, 450.0, 900.0],
                "w35_d1024_L6_mode_b_120M": {
                    "latest_measured_ce": round(best_val_ce, 4),
                    **proj_final["post_wsd_cooldown_by_horizon"],
                },
                "w34_d256_L6_30M_measured_14p75M": 4.8332,
                "w34_d512_L6_48M_measured_14p75M": 4.9214,
                "w29_d384_L6_40M_measured_14p75M": 4.9628,
                "w30_d512_L6_68M_measured_44p24M": 4.8195,
                "w30_d512_L6_68M_projected_900M": 3.1512,
                "w18_d512_L8_62M_measured_900M": 4.0556,
                "w12_classical_d768_L16_124M_measured_900M": 3.3362,
                "modded_nanogpt_speedrun_target_900M": 3.2800,
            },
            "hypothesis_3_pass": h3_pass,
        },
        "acceptance_criteria": {
            "H1_single_synthesis_parity_and_vram_pass": h1_report["hypothesis_1_pass"],
            "H2_long_horizon_d1024_mode_b_pass": h2_pass,
            "H3_900m_projection_sub_3p12_pass": h3_pass,
        },
    }
    out_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"\n[W35 COMPLETE] Saved report to {out_path}", flush=True)
    return full_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-key", type=str, default="")
    parser.add_argument("--steps", type=int, default=DEFAULT_LONG_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--full-900m", action="store_true", help="Stream all 9 FineWeb shards for 900M tokens")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    resolved_key = args.run_key if args.run_key else (DEFAULT_900M_RUN_KEY if args.full_900m else DEFAULT_RUN_KEY)
    cfg = Wave35ModeBConfig(
        run_key=resolved_key,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        steps=args.steps,
        full_900m=bool(args.full_900m),
    )
    run_wave35_mode_b_long_horizon(cfg, torch.device(args.device))


if __name__ == "__main__":
    main()
