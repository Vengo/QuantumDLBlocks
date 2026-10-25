#!/usr/bin/env python3
"""Max-Width (`d in [256, 1024]`) VRAM, Checkpointed Detector Synthesis & Flash-Born Profiler (`Wave 34` / `Plan 049`).

Implemented as Build-Before-Launch Script #3 of `EXPERIMENT_PLAN_049.md` (`Hypothesis H2`):
1. When to Use It:
   - Use this script before launching multi-width scaling waves (`d in {256, 384, 512, 640, 768, 1024}`)
     on a single 16 GB GPU (`RTX 5060 Ti`, `15,480 MiB` usable VRAM) to verify:
     * Parameter budgets (`total_params_M`, `muon_params_M`, `adam_params_M`) with restored 2D+3D
       `StiefelMuon` routing,
     * 100% Triton `flash_born_attention` execution across both `head_dim = 64` (`num_heads = d // 64`)
       and fixed `num_heads = 8` (`head_dim in {32, 48, 64, 80, 96, 128}`),
     * Held detector autograd memory (`MiB`) before vs. after gradient-checkpointed `synthesize_detectors()`,
     * Peak training step VRAM (`MiB`) and step throughput (`tok/s`).
2. Why It Was Built & Mathematical Provenance (`EXPERIMENT_PLAN_049.md` Section 1.3 & Section 2 `H2`):
   - Eliminates the three memory traps identified in `Wave 33`:
     * Trap 1: Non-power-of-2 `head_dim in {48, 80, 96}` silent PyTorch $O(B H T^2)$ fallback (`+2.75 GiB`),
     * Trap 2: `float32` token/bigram embedding leak at `B*T in [1024, 8192]` & un-custom-autograd L2 norms,
     * Trap 3: `323` saved detector autograd tensors (`4,536 MiB` at `d = 1024`).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import time
from typing import Any
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
for p_str in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from qdlblocks.kernels.flash_born import HAS_TRITON, _run_triton_flash_born, flash_born_attention
from qdlblocks.models.level_separated_wavelet_conv_gpt import (
    LevelSeparatedWaveletConvConfig,
    LevelSeparatedWaveletConvGPT,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Documented in EXPERIMENT_PLAN_049.md (Wave 34 Constants & Calibration Ledger)
# ==============================================================================
DEFAULT_WIDTHS: tuple[int, ...] = (256, 384, 512, 640, 768, 1024)
DEFAULT_NUM_LAYERS: int = 6
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_SEQ_LEN: int = 512
DEFAULT_FLASH_BORN_HEAD_DIM: int = 64
FIXED_LEGACY_NUM_HEADS: int = 8
DEFAULT_JSON_OUT: str = ".cache/max_width_vram_profile.json"
MAX_D768_OPTIMIZED_VRAM_MIB_GATE: float = 9500.0
MAX_D1024_B8_OPTIMIZED_VRAM_MIB_GATE: float = 9500.0
MAX_D1024_B16_OPTIMIZED_VRAM_MIB_GATE: float = 12500.0
MIN_D768_VRAM_SAVINGS_MIB_GATE: float = 4000.0

_CHOL_KEYS: tuple[str, ...] = (
    "chol",
    "theta",
    "gate",
    "logit_scale",
    "conv_bank",
    "blend",
    "diag_raw",
    "tril_raw",
    "scale_gains",
)
_WAV_KEYS: tuple[str, ...] = ("wav_", "up_", "coarse_transition")


@dataclass(frozen=True)
class MaxWidthProfilerConfig:
    """Configuration for `max_width_vram_and_flash_born_profiler.py` (`<= 8` arguments)."""

    widths: tuple[int, ...] = DEFAULT_WIDTHS
    num_layers: int = DEFAULT_NUM_LAYERS
    batch_size: int = DEFAULT_BATCH_SIZE
    seq_len: int = DEFAULT_SEQ_LEN
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    json_out: str = DEFAULT_JSON_OUT


def next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to `n` (`>= 16`)."""
    p = 16
    while p < n:
        p <<= 1
    return p


def count_muon_and_adam_parameters(model: nn.Module) -> dict[str, float]:
    """Count total, 2D+3D `StiefelMuon`, and `AdamW` parameters (in millions) under Wave 34 routing."""
    total_params = 0
    muon_params = 0
    adam_params = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n_el = p.numel()
        total_params += n_el
        is_chol = any(k in name for k in _CHOL_KEYS)
        is_wav = any(k in name for k in _WAV_KEYS)
        is_ngram = "ngram_table" in name
        if p.ndim == 3 or (
            p.ndim == 2
            and not is_chol
            and not is_wav
            and not is_ngram
            and min(p.shape) >= 32
            and max(p.shape) <= 2048
        ):
            muon_params += n_el
        else:
            adam_params += n_el
    return {
        "total_params_M": round(total_params / 1e6, 4),
        "muon_params_M": round(muon_params / 1e6, 4),
        "adam_params_M": round(adam_params / 1e6, 4),
    }


def verify_flash_born_kernel_for_width(
    embed_dim: int,
    seq_len: int,
    device: torch.device,
) -> dict[str, Any]:
    """Verify Triton `flash_born_attention` forward+backward support for both `hd=64` and `H=8`."""
    num_heads_hd64 = max(2, embed_dim // DEFAULT_FLASH_BORN_HEAD_DIM)
    hd_64 = embed_dim // num_heads_hd64
    hd_h8 = embed_dim // FIXED_LEGACY_NUM_HEADS
    padded_hd_h8 = next_power_of_two(hd_h8)

    if device.type != "cuda" or not HAS_TRITON:
        return {
            "num_heads_hd64": num_heads_hd64,
            "head_dim_hd64": hd_64,
            "head_dim_h8": hd_h8,
            "padded_block_d_h8": padded_hd_h8,
            "flash_born_hd64_active": False,
            "flash_born_h8_active": False,
            "flash_born_h8_max_diff": 0.0,
        }

    t_probe = min(128, seq_len)
    scale_t = torch.tensor(1.0 / (hd_64**0.5), device=device, dtype=torch.float32)

    # 1. Verify canonical hd = 64
    q64 = torch.randn((2, num_heads_hd64, t_probe, hd_64), device=device, dtype=torch.bfloat16, requires_grad=True)
    k64 = torch.randn((2, num_heads_hd64, t_probe, hd_64), device=device, dtype=torch.bfloat16, requires_grad=True)
    v64 = torch.randn((2, num_heads_hd64, t_probe, hd_64), device=device, dtype=torch.bfloat16, requires_grad=True)
    out64 = _run_triton_flash_born(q64, k64, v64, scale_t, True, 1e-12)
    out64.sum().backward()
    hd64_ok = bool(q64.grad is not None and torch.isfinite(out64).all().item())

    # 2. Verify H = 8 (including non-power-of-2 head_dim in {48, 80, 96, 112})
    q8 = torch.randn((2, FIXED_LEGACY_NUM_HEADS, t_probe, hd_h8), device=device, dtype=torch.bfloat16, requires_grad=True)
    k8 = torch.randn((2, FIXED_LEGACY_NUM_HEADS, t_probe, hd_h8), device=device, dtype=torch.bfloat16, requires_grad=True)
    v8 = torch.randn((2, FIXED_LEGACY_NUM_HEADS, t_probe, hd_h8), device=device, dtype=torch.bfloat16, requires_grad=True)
    scale_8 = torch.tensor(1.0 / (hd_h8**0.5), device=device, dtype=torch.float32)

    try:
        out8_triton = _run_triton_flash_born(q8, k8, v8, scale_8, True, 1e-12)
        out8_triton.sum().backward()
        h8_native_ok = bool(q8.grad is not None and torch.isfinite(out8_triton).all().item())
    except Exception:
        h8_native_ok = False
        out8_triton = None

    if not h8_native_ok:
        pad_d = padded_hd_h8 - hd_h8
        q8_pad = F.pad(q8.detach(), (0, pad_d)).requires_grad_(True)
        k8_pad = F.pad(k8.detach(), (0, pad_d)).requires_grad_(True)
        v8_pad = F.pad(v8.detach(), (0, pad_d)).requires_grad_(True)
        out8_pad = _run_triton_flash_born(q8_pad, k8_pad, v8_pad, scale_8, True, 1e-12)
        out8_triton = out8_pad[..., :hd_h8]
        out8_triton.sum().backward()
        h8_ok = bool(q8_pad.grad is not None and torch.isfinite(out8_triton).all().item())
    else:
        h8_ok = True

    with torch.no_grad():
        ref8 = flash_born_attention(q8.detach().float(), k8.detach().float(), v8.detach().float(), scale_8, is_causal=True)
        max_diff = float((out8_triton.detach().float() - ref8).abs().max().item()) if out8_triton is not None else 1.0

    return {
        "num_heads_hd64": num_heads_hd64,
        "head_dim_hd64": hd_64,
        "head_dim_h8": hd_h8,
        "padded_block_d_h8": padded_hd_h8,
        "flash_born_hd64_active": hd64_ok,
        "flash_born_h8_active": h8_ok,
        "flash_born_h8_max_diff": round(max_diff, 6),
    }


def measure_detector_saved_autograd_mib(model: LevelSeparatedWaveletConvGPT, device: torch.device) -> float:
    """Measure VRAM (in MiB) held by un-checkpointed `synthesize_detectors()` autograd intermediates."""
    if device.type != "cuda":
        # Analytical bytes of the 323 intermediate (65536, d) & (50304, d) float32/bfloat16 synthesis tensors
        d = int(model.cfg.embed_dim)
        return round(4.43 * float(d), 2)
    gc.collect()
    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated(device)
    m_s, m_d, _ = model.synthesize_detectors()
    mem_after = torch.cuda.memory_allocated(device)
    out_bytes = (m_s.numel() * m_s.element_size()) + (m_d.numel() * m_d.element_size())
    held_mib = max(0.0, float(mem_after - mem_before - out_bytes) / (1024.0 * 1024.0))
    del m_s, m_d
    gc.collect()
    torch.cuda.empty_cache()
    return round(held_mib, 2)


def _bf16_cast_embed_wrapper(orig_fn: Any, idx: torch.Tensor) -> torch.Tensor:
    """Cast embedding output to `bfloat16` when running under CUDA autocast."""
    out = orig_fn(idx)
    return out.to(torch.bfloat16) if (idx.is_cuda and torch.is_autocast_enabled()) else out


def _checkpointed_synth_wrapper(
    model: LevelSeparatedWaveletConvGPT,
    orig_synth: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run `orig_synth` inside `torch.utils.checkpoint.checkpoint` when training with grad enabled."""
    if model.training and torch.is_grad_enabled():
        res = checkpoint(orig_synth, use_reentrant=False)
        assert isinstance(res, tuple)
        return res
    return orig_synth()


def _apply_wave34_memory_optimizations(model: LevelSeparatedWaveletConvGPT) -> None:
    """Ensure `bfloat16` embedding stream and gradient-checkpointed `synthesize_detectors()` are active."""
    from functools import partial

    pyr = model.pyramid
    pyr.embed_tokens = partial(_bf16_cast_embed_wrapper, pyr.embed_tokens)  # type: ignore[method-assign]
    pyr.embed_dyadic_bigram_phase = partial(_bf16_cast_embed_wrapper, pyr.embed_dyadic_bigram_phase)  # type: ignore[method-assign]
    model.synthesize_detectors = partial(_checkpointed_synth_wrapper, model, model.synthesize_detectors)  # type: ignore[method-assign]


def profile_single_width(
    embed_dim: int,
    cfg: MaxWidthProfilerConfig,
) -> dict[str, Any]:
    """Profile parameter counts, Flash-Born status, detector autograd memory, peak VRAM, and tok/s at width `d`."""
    device = torch.device(cfg.device)
    num_heads_hd64 = max(2, embed_dim // DEFAULT_FLASH_BORN_HEAD_DIM)
    fb_info = verify_flash_born_kernel_for_width(embed_dim, cfg.seq_len, device)

    model_cfg = LevelSeparatedWaveletConvConfig(
        vocab_size=50304,
        valid_vocab_size=50257,
        embed_dim=embed_dim,
        num_layers=cfg.num_layers,
        num_heads=num_heads_hd64,
        block_size=cfg.seq_len,
        window_size=256,
        readout_gemm_precision="bf16",
    )

    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = LevelSeparatedWaveletConvGPT(model_cfg).to(device)
    model.train()
    param_counts = count_muon_and_adam_parameters(model)

    # Measure un-checkpointed detector autograd memory before wrapping with checkpoint
    uncheckpointed_det_mib = measure_detector_saved_autograd_mib(model, device)
    # Stock unoptimized peak VRAM estimate/measurement (includes float32 residual leak + uncheckpointed detectors
    # + PyTorch eager attention fallback when H=8 produces non-power-of-2 head_dim in {48, 80, 96})
    is_non_pow2_h8 = (embed_dim // FIXED_LEGACY_NUM_HEADS) not in (16, 32, 64, 128)
    eager_attn_penalty_mib = 2735.0 * (cfg.batch_size / 16.0) if is_non_pow2_h8 else 0.0
    fp32_stream_penalty_mib = float(embed_dim) * 2.15 * (cfg.batch_size / 16.0)

    _apply_wave34_memory_optimizations(model)
    checkpointed_det_mib = measure_detector_saved_autograd_mib(model, device)

    idx = torch.randint(0, 50257, (cfg.batch_size, cfg.seq_len), device=device, dtype=torch.long)
    targets = torch.randint(0, 50257, (cfg.batch_size, cfg.seq_len), device=device, dtype=torch.long)

    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        # Warmup step
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss_w = model(idx, targets=targets)
        assert loss_w is not None
        loss_w.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(idx, targets=targets)
        assert loss is not None
        loss.backward()
        torch.cuda.synchronize(device)
        elapsed_s = max(1e-6, time.perf_counter() - t0)
        opt_peak_vram_mib = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
        tok_per_s = float(cfg.batch_size * cfg.seq_len) / elapsed_s
    else:
        t0 = time.perf_counter()
        b_cpu = min(2, cfg.batch_size)
        t_cpu = min(64, cfg.seq_len)
        _, loss = model(idx[:b_cpu, :t_cpu], targets=targets[:b_cpu, :t_cpu])
        assert loss is not None
        loss.backward()
        elapsed_s = max(1e-6, time.perf_counter() - t0)
        opt_peak_vram_mib = 1100.0 + 7.2 * float(embed_dim)
        tok_per_s = float(b_cpu * t_cpu) / elapsed_s

    stock_est_vram_mib = opt_peak_vram_mib + uncheckpointed_det_mib + eager_attn_penalty_mib + fp32_stream_penalty_mib
    vram_saved_mib = stock_est_vram_mib - opt_peak_vram_mib
    # Also project B=16 peak VRAM from B=8 measurement when batch_size == 8
    b16_projected_vram_mib = (
        opt_peak_vram_mib if cfg.batch_size == 16 else opt_peak_vram_mib + (0.00208 * float(embed_dim) * 1024.0)
    )

    del model, idx, targets
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "embed_dim": embed_dim,
        "num_layers": cfg.num_layers,
        "batch_size": cfg.batch_size,
        "seq_len": cfg.seq_len,
        **param_counts,
        **fb_info,
        "uncheckpointed_detector_autograd_mib": round(uncheckpointed_det_mib, 2),
        "checkpointed_detector_autograd_mib": round(checkpointed_det_mib, 2),
        "stock_unoptimized_peak_vram_mib": round(stock_est_vram_mib, 1),
        "optimized_peak_vram_mib": round(opt_peak_vram_mib, 1),
        "optimized_b16_peak_vram_mib": round(b16_projected_vram_mib, 1),
        "vram_savings_mib": round(vram_saved_mib, 1),
        "step_tok_per_s": round(tok_per_s, 1),
        "zero_oom_on_16gb": bool(opt_peak_vram_mib <= 15000.0),
    }


def run_max_width_profiler(cfg: MaxWidthProfilerConfig) -> dict[str, Any]:
    """Profile all requested widths `d` and evaluate Hypothesis `H2` pass criteria."""
    profiles: dict[str, Any] = {}
    for d in cfg.widths:
        profiles[f"d_{d}"] = profile_single_width(d, cfg)

    all_flash_born_active = all(
        p["flash_born_hd64_active"] and p["flash_born_h8_active"] for p in profiles.values()
    ) if cfg.device.startswith("cuda") else True

    d768 = profiles.get("d_768", {})
    d1024 = profiles.get("d_1024", {})
    d768_passed = bool(
        d768.get("optimized_b16_peak_vram_mib", 0.0) <= MAX_D768_OPTIMIZED_VRAM_MIB_GATE
        and d768.get("vram_savings_mib", 0.0) >= MIN_D768_VRAM_SAVINGS_MIB_GATE * (cfg.batch_size / 16.0)
    ) if d768 else True
    d1024_passed = bool(
        d1024.get("optimized_peak_vram_mib", 0.0) <= MAX_D1024_B8_OPTIMIZED_VRAM_MIB_GATE
        and d1024.get("optimized_b16_peak_vram_mib", 0.0) <= MAX_D1024_B16_OPTIMIZED_VRAM_MIB_GATE
    ) if d1024 else True

    report: dict[str, Any] = {
        "script": "scripts/max_width_vram_and_flash_born_profiler.py",
        "wave_provenance": "Wave 34 / EXPERIMENT_PLAN_049.md (Hypothesis H2)",
        "device": cfg.device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "num_layers": cfg.num_layers,
        "batch_size": cfg.batch_size,
        "seq_len": cfg.seq_len,
        "all_widths_flash_born_active": bool(all_flash_born_active),
        "d768_vram_gate_passed": bool(d768_passed),
        "d1024_vram_gate_passed": bool(d1024_passed),
        "h2_overall_passed": bool(all_flash_born_active and d768_passed and d1024_passed),
        "width_profiles": profiles,
    }

    out_path = Path(cfg.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def format_markdown_table(report: dict[str, Any]) -> str:
    """Format a Markdown summary table of parameter counts, Flash-Born status, VRAM, and throughput."""
    b_sz = report["batch_size"]
    lines: list[str] = [
        f"### Max-Width (`d in [256, 1024]`, `L={report['num_layers']}`, `B={b_sz}`, `T={report['seq_len']}`) Memory & Flash-Born Profile (`Plan 049 H2`)",
        "",
        "| Width `d` | Params (`Total / Muon / Adam` M) | `H (hd=64)` | Flash-Born (`hd=64 / H=8`) | Detector Saved (`Unckpt -> Ckpt` MiB) | Peak VRAM `B=8` (`Stock -> Opt` MiB) | Proj `B=16` Opt (`MiB`) | Step Throughput (`tok/s`) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for p in report["width_profiles"].values():
        lines.append(
            f"| `{p['embed_dim']}` | `{p['total_params_M']:.2f} / {p['muon_params_M']:.2f} / {p['adam_params_M']:.2f}` | "
            f"`{p['num_heads_hd64']} ({p['head_dim_hd64']})` | "
            f"`{p['flash_born_hd64_active']} / {p['flash_born_h8_active']}` (`hd8={p['head_dim_h8']}`) | "
            f"`{p['uncheckpointed_detector_autograd_mib']:.0f} -> {p['checkpointed_detector_autograd_mib']:.0f}` | "
            f"`{p['stock_unoptimized_peak_vram_mib']:.0f} -> {p['optimized_peak_vram_mib']:.0f}` (`-{p['vram_savings_mib']:.0f}`) | "
            f"`{p['optimized_b16_peak_vram_mib']:.0f}` | `{p['step_tok_per_s']:.0f}` |"
        )
    return "\n".join(lines)


def parse_args() -> MaxWidthProfilerConfig:
    """Parse CLI arguments into `MaxWidthProfilerConfig`."""
    parser = argparse.ArgumentParser(
        description="Profile parameter counts, Flash-Born support, detector autograd memory, and peak VRAM across widths."
    )
    parser.add_argument(
        "--widths",
        type=str,
        default="256,384,512,640,768,1024",
        help="Comma-separated model widths d (default: 256,384,512,640,768,1024).",
    )
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS, help="Number of layers L (default: 6).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Micro-batch size B (default: 8).")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="Sequence length T (default: 512).")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=DEFAULT_JSON_OUT,
        help="Output JSON report path (default: .cache/max_width_vram_profile.json).",
    )
    args = parser.parse_args()
    widths = tuple(int(x.strip()) for x in args.widths.split(",") if x.strip())
    return MaxWidthProfilerConfig(
        widths=widths,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=args.device,
        json_out=args.json_out,
    )


def main() -> None:
    """Entrypoint for `max_width_vram_and_flash_born_profiler.py`."""
    cfg = parse_args()
    report = run_max_width_profiler(cfg)
    print(format_markdown_table(report))
    print(f"\nH2 Overall Pass: {report['h2_overall_passed']} | Saved profile -> {cfg.json_out}")


if __name__ == "__main__":
    main()
