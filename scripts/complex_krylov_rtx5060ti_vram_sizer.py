"""RTX 5060 Ti 16 GB VRAM Sizer & Small-`d_complex` Multiples-of-32 Profiler (`Waves 38–39` / `EXPERIMENT_PLAN_053` & `054`).

Supports two profiling modes via `--mode`:
1. `--mode max_vram` (default): Profiles `d_complex in {128, 256, 384, 512, 640, 768}` on the 16 GB RTX 5060 Ti.
2. `--mode small_d_sweep`: Profiles all multiples of 32 `d_complex in {32, 64, 96, 128, 160, 192}` (`L=6, B=16, T=512`),
   reporting `total_params_M`, `float32_activation_leak_count`, `peak_vram_mib`, `tok_per_s`, `est_5min_steps`,
   and `est_5min_tokens_M`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qdlblocks.layers.packed_complex_register_density_attn import complex_l2_normalize
from qdlblocks.models.gemini_packed_complex_rope_gpt import (
    GeminiPackedComplexRoPEConfig,
    GeminiPackedComplexRoPEGPT,
    build_gemini_complex_optimizers,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
DEFAULT_SEQ_LEN: int = 512
DEFAULT_TARGET_MIN_VRAM_MIB: float = 8000.0
DEFAULT_TARGET_MAX_VRAM_MIB: float = 15000.0
WAVE37_BASELINE_VRAM_MIB: float = 5355.0
WAVE37_BASELINE_TOK_S: float = 7952.0
FIVE_MIN_WALL_CLOCK_S: float = 300.0
MIB_DIVISOR: float = float(1024 * 1024)


@dataclass(frozen=True)
class ComplexSizerSpec:
    """Candidate `GeminiPackedComplexRoPEGPT` sizing tier for the 16 GB RTX 5060 Ti."""

    tier_name: str
    d_complex: int
    n_registers: int
    d_register: int
    num_layers: int
    batch_size: int
    ffn_expansion: int
    checkpoint_blocks: bool


CANDIDATE_TIERS: tuple[ComplexSizerSpec, ...] = (
    ComplexSizerSpec("d128c_H4_L6_B16", 128, 4, 32, 6, 16, 2, False),
    ComplexSizerSpec("d256c_H8_L8_B16", 256, 8, 32, 8, 16, 2, False),
    ComplexSizerSpec("d384c_H12_L8_B16", 384, 12, 32, 8, 16, 2, False),
    ComplexSizerSpec("d512c_H16_L10_B16_ckpt", 512, 16, 32, 10, 16, 2, True),
    ComplexSizerSpec("d512c_H16_L10_B16_nockpt", 512, 16, 32, 10, 16, 2, False),
    ComplexSizerSpec("d640c_H20_L12_B16_ckpt", 640, 20, 32, 12, 16, 2, True),
    ComplexSizerSpec("d640c_H20_L12_B16_nockpt", 640, 20, 32, 12, 16, 2, False),
    ComplexSizerSpec("d768c_H24_L12_B16_ckpt", 768, 24, 32, 12, 16, 2, True),
    ComplexSizerSpec("d768c_H24_L12_B16_nockpt", 768, 24, 32, 12, 16, 2, False),
)

SMALL_D_SWEEP_TIERS: tuple[ComplexSizerSpec, ...] = (
    ComplexSizerSpec("d32c_H2_dh16_L6_B16", 32, 2, 16, 6, 16, 2, False),
    ComplexSizerSpec("d64c_H2_dh32_L6_B16", 64, 2, 32, 6, 16, 2, False),
    ComplexSizerSpec("d96c_H3_dh32_L6_B16", 96, 3, 32, 6, 16, 2, False),
    ComplexSizerSpec("d128c_H4_dh32_L6_B16", 128, 4, 32, 6, 16, 2, False),
    ComplexSizerSpec("d160c_H5_dh32_L6_B16", 160, 5, 32, 6, 16, 2, False),
    ComplexSizerSpec("d192c_H6_dh32_L6_B16", 192, 6, 32, 6, 16, 2, False),
)


def count_float32_activation_leaks(
    model: GeminiPackedComplexRoPEGPT,
    idx: torch.Tensor,
) -> int:
    """Count how many activation tensors leak to `float32` under CUDA `bfloat16` autocast."""
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        det_re, det_im = model.synthesize_complex_detectors()
        b_sz, seq_t = idx.shape
        d_c = model.cfg.d_complex
        psi0_re = model.embed_re.index_select(0, idx.reshape(-1)).view(b_sz, seq_t, d_c)
        psi0_im = model.embed_im.index_select(0, idx.reshape(-1)).view(b_sz, seq_t, d_c)
        psi_re, psi_im = complex_l2_normalize(psi0_re, psi0_im, dim=-1)
        dtypes = [det_re.dtype, det_im.dtype, psi_re.dtype, psi_im.dtype]
        v0_re, v0_im = None, None
        for l_idx, block in enumerate(model.blocks):
            psi_re, psi_im, vr, vi = block(psi_re, psi_im, v0_re=v0_re, v0_im=v0_im)
            dtypes.extend([psi_re.dtype, psi_im.dtype, vr.dtype, vi.dtype])
            if l_idx == 0:
                v0_re, v0_im = vr, vi
    return sum(1 for dt in dtypes if dt == torch.float32)


def benchmark_single_tier(
    spec: ComplexSizerSpec,
    seq_len: int,
    warmup_steps: int,
    bench_steps: int,
) -> dict[str, Any]:
    """Benchmark a single `ComplexSizerSpec` on CUDA (`<= 8` args)."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")

    cfg = GeminiPackedComplexRoPEConfig(
        d_complex=spec.d_complex,
        n_registers=spec.n_registers,
        d_register=spec.d_register,
        num_layers=spec.num_layers,
        block_size=seq_len,
        ffn_expansion=spec.ffn_expansion,
        checkpoint_blocks=spec.checkpoint_blocks,
    )
    tokens_per_step = int(spec.batch_size * seq_len)
    try:
        model = GeminiPackedComplexRoPEGPT(cfg).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        opt_muon, opt_adam = build_gemini_complex_optimizers(model)
        muon_params = sum(p.numel() for g in opt_muon.param_groups for p in g["params"])
        adam_params = sum(p.numel() for g in opt_adam.param_groups for p in g["params"])

        idx = torch.randint(0, cfg.valid_vocab_size, (spec.batch_size, seq_len), device=device)
        targets = torch.randint(0, cfg.valid_vocab_size, (spec.batch_size, seq_len), device=device)

        leak_count = count_float32_activation_leaks(model, idx[:2])
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        for _ in range(warmup_steps):
            opt_muon.zero_grad(set_to_none=True)
            opt_adam.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(idx, targets=targets)
            assert loss is not None
            loss.backward()
            opt_muon.step()
            opt_adam.step()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for s_idx in range(bench_steps):
            opt_muon.zero_grad(set_to_none=True)
            opt_adam.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(idx, targets=targets)
            assert loss is not None
            loss.backward()
            opt_muon.step()
            opt_adam.step()
            if (s_idx + 1) % 2 == 0:
                model.retract_complex_unitary_weights(rate=0.04)
        torch.cuda.synchronize()
        elapsed = max(1e-6, time.perf_counter() - t0)

        peak_vram_mib = float(torch.cuda.max_memory_allocated() / MIB_DIVISOR)
        tok_per_s = float((tokens_per_step * bench_steps) / elapsed)
        est_5min_tokens = tok_per_s * FIVE_MIN_WALL_CLOCK_S
        est_5min_steps = int(est_5min_tokens // tokens_per_step)
        result = {
            **asdict(spec),
            "real_equiv_dim": cfg.real_equiv_dim,
            "total_params_M": round(total_params / 1e6, 3),
            "muon_params_M": round(muon_params / 1e6, 3),
            "adam_params_M": round(adam_params / 1e6, 3),
            "float32_activation_leak_count": int(leak_count),
            "peak_vram_mib": round(peak_vram_mib, 1),
            "tok_per_s": round(tok_per_s, 1),
            "est_5min_steps": est_5min_steps,
            "est_5min_tokens_M": round(est_5min_tokens / 1e6, 2),
            "status": "OK",
        }
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            result = {
                **asdict(spec),
                "real_equiv_dim": 2 * spec.d_complex,
                "total_params_M": 0.0,
                "muon_params_M": 0.0,
                "adam_params_M": 0.0,
                "float32_activation_leak_count": 0,
                "peak_vram_mib": 16384.0,
                "tok_per_s": 0.0,
                "est_5min_steps": 0,
                "est_5min_tokens_M": 0.0,
                "status": "OOM",
            }
        else:
            raise
    finally:
        del cfg
        gc.collect()
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="RTX 5060 Ti 16 GB VRAM Sizer for GeminiPackedComplexRoPEGPT.")
    parser.add_argument("--mode", type=str, choices=("max_vram", "small_d_sweep"), default="max_vram")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--bench-steps", type=int, default=6)
    parser.add_argument("--target-min-vram-mib", type=float, default=DEFAULT_TARGET_MIN_VRAM_MIB)
    parser.add_argument("--target-max-vram-mib", type=float, default=DEFAULT_TARGET_MAX_VRAM_MIB)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    tiers = SMALL_D_SWEEP_TIERS if args.mode == "small_d_sweep" else CANDIDATE_TIERS
    default_out = (
        REPO_ROOT / ".cache" / "complex_krylov_small_d_sweep.json"
        if args.mode == "small_d_sweep"
        else REPO_ROOT / ".cache" / "complex_krylov_rtx5060ti_vram_sizer.json"
    )
    out_path = Path(args.json_out) if args.json_out else default_out

    rows: list[dict[str, Any]] = []
    print(
        "| Tier | d_complex | H | d_h | d_real | L | Params (M) | F32 Leaks | Peak VRAM (MiB) | Tok/s | 5-Min Steps | 5-Min Tok (M) | Status |"
    )
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for spec in tiers:
        row = benchmark_single_tier(spec, args.seq_len, args.warmup_steps, args.bench_steps)
        rows.append(row)
        print(
            f"| `{row['tier_name']}` | {row['d_complex']} | {row['n_registers']} | {row['d_register']} | "
            f"{row['real_equiv_dim']} | {row['num_layers']} | {row['total_params_M']:.2f}M | "
            f"{row['float32_activation_leak_count']} | {row['peak_vram_mib']:.1f} | {row['tok_per_s']:.0f} | "
            f"{row['est_5min_steps']} | {row['est_5min_tokens_M']:.2f}M | {row['status']} |"
        )

    d128_rows = [r for r in rows if r["d_complex"] == 128 and r["status"] == "OK"]
    baseline_row = d128_rows[0] if d128_rows else rows[0]

    if args.mode == "small_d_sweep":
        small_rows = [r for r in rows if r["d_complex"] <= 96 and r["status"] == "OK"]
        max_small_tok_s = max((r["tok_per_s"] for r in small_rows), default=0.0)
        total_leaks = sum(int(r["float32_activation_leak_count"]) for r in rows)
        report = {
            "mode": "small_d_sweep",
            "d128_tok_per_s": baseline_row["tok_per_s"],
            "d128_vram_mib": baseline_row["peak_vram_mib"],
            "max_small_d_tok_per_s": max_small_tok_s,
            "all_float32_leaks": total_leaks,
            "h2_pass": bool(
                total_leaks == 0 and baseline_row["tok_per_s"] >= 28000.0 and max_small_tok_s >= 38000.0
            ),
            "all_tiers": rows,
        }
    else:
        in_envelope = [
            r
            for r in rows
            if r["status"] == "OK"
            and r["d_complex"] >= 512
            and args.target_min_vram_mib <= r["peak_vram_mib"] <= args.target_max_vram_mib
        ]
        if not in_envelope:
            in_envelope = [
                r
                for r in rows
                if r["status"] == "OK"
                and r["d_complex"] >= 512
                and r["peak_vram_mib"] <= args.target_max_vram_mib
            ]
        in_envelope.sort(key=lambda r: (r["total_params_M"], r["tok_per_s"]), reverse=True)
        selected = in_envelope[0] if in_envelope else baseline_row
        report = {
            "mode": "max_vram",
            "wave37_baseline_vram_mib": WAVE37_BASELINE_VRAM_MIB,
            "wave37_baseline_tok_per_s": WAVE37_BASELINE_TOK_S,
            "wave38_d128_vram_mib": baseline_row["peak_vram_mib"],
            "wave38_d128_tok_per_s": baseline_row["tok_per_s"],
            "vram_compression_ratio_d128": round(
                WAVE37_BASELINE_VRAM_MIB / max(1.0, baseline_row["peak_vram_mib"]), 2
            ),
            "throughput_speedup_ratio_d128": round(baseline_row["tok_per_s"] / WAVE37_BASELINE_TOK_S, 2),
            "selected_max_5060ti_tier": selected,
            "all_tiers": rows,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved sizer report to {out_path}")


if __name__ == "__main__":
    main()
