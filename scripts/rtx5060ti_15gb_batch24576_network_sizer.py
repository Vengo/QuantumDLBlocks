#!/usr/bin/env python3
"""RTX 5060 Ti 15 GB VRAM & 24,576-Token Batch Network Sizer (`EXPERIMENT_PLAN_041.md`).

Audits `Radical2Pow16WaveletCholeskyGPT` under `torch.amp.autocast("cuda", dtype=torch.bfloat16)`
and `torch.cuda.set_per_process_memory_fraction(0.95, device=0)` (~15.0 GB VRAM ceiling):
1. Verifies that all intermediate activations (`psi_in`, `psi`, `blk_diff`, `m_sharp`, `m_diff`)
   remain in `torch.bfloat16` without silent `float32` promotion.
2. Benchmarks forward+backward+optimizer step throughput (`tok/s`) and peak VRAM (`GB`)
   at `B_tok = 24,576` (`B = 48, T = 512`) across network scales (`d in {256, 384, 512}`)
   and readout GEMM precisions (`bf16`, `fp8`, `int8`).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import time
from typing import Any

import torch

from qdlblocks.models.wavelet_cholesky_2pow16_gpt import (
    Radical2Pow16Config,
    Radical2Pow16WaveletCholeskyGPT,
    build_radical_2pow16_optimizers,
)


@dataclass(frozen=True)
class CandidateScaleSpec:
    """Candidate architecture scale for the 15 GB / 24,576-token batch RTX 5060 Ti sweep."""

    name: str
    batch_size: int
    seq_len: int
    embed_dim: int
    num_layers: int
    num_heads: int
    mid_detail_dim: int
    fine_detail_dim: int
    diffuse_rank: int
    interferometer_expansion: float
    use_dyadic_hierarchical_born: bool = True
    readout_gemm_precision: str = "bf16"


def verify_bfloat16_activations(device: torch.device) -> dict[str, str]:
    """Verify that all activations in `Radical2Pow16WaveletCholeskyGPT` stay in `bfloat16` under autocast."""
    cfg = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet",
        num_layers=4,
        embed_dim=256,
        num_heads=8,
        use_fast_cholesky_and_compile=True,
        use_convex_traceless_born=True,
        use_dyadic_hierarchical_born=True,
    )
    model = Radical2Pow16WaveletCholeskyGPT(cfg).to(device)
    idx = torch.randint(0, cfg.valid_vocab_size, (2, 64), device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        psi_in = model.pyramid.embed_tokens(idx)
        bigram_psi = model.pyramid.embed_dyadic_bigram_phase(idx)
        v_0 = model.pyramid.synthesize_value_anchor(psi_in, bigram_psi)
        cos, sin = model.rope.get_cos_sin(idx.size(1))
        cos, sin = cos.to(device), sin.to(device)
        psi_blk, diff_blk, _ = model.blocks[0](psi_in, cos=cos, sin=sin, v_0=v_0, doc_ids=None)
        m_sharp, m_diff, _ = model.pyramid.synthesize_vocab_detectors()
    dtypes = {
        "psi_in": str(psi_in.dtype),
        "bigram_psi": str(bigram_psi.dtype),
        "v_0": str(v_0.dtype),
        "psi_blk": str(psi_blk.dtype),
        "diff_blk": str(diff_blk.dtype),
        "m_sharp": str(m_sharp.dtype),
        "m_diff": str(m_diff.dtype),
    }
    del model, idx, psi_in, bigram_psi, v_0, psi_blk, diff_blk, m_sharp, m_diff
    torch.cuda.empty_cache()
    return dtypes


def profile_candidate_spec(
    spec: CandidateScaleSpec,
    device: torch.device,
    warmup_steps: int = 2,
    timed_steps: int = 5,
) -> dict[str, Any]:
    """Profile parameter count, peak VRAM, and training throughput for one `CandidateScaleSpec`."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    cfg = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet",
        vocab_size=50304,
        valid_vocab_size=50257,
        ambient_n=65536,
        num_layers=spec.num_layers,
        embed_dim=spec.embed_dim,
        num_heads=spec.num_heads,
        block_size=spec.seq_len,
        window_size=min(384, spec.seq_len),
        fine_detail_dim=spec.fine_detail_dim,
        mid_detail_dim=spec.mid_detail_dim,
        diffuse_rank=spec.diffuse_rank,
        interferometer_expansion=spec.interferometer_expansion,
        num_fine_up_banks=1,
        use_kraus_retraction_and_lambda_anneal=True,
        use_fast_cholesky_and_compile=True,
        use_convex_traceless_born=True,
        traceless_gamma=0.25,
        use_dyadic_hierarchical_born=spec.use_dyadic_hierarchical_born,
        dyadic_coarse_alpha=0.22,
        dyadic_coarse_ce_weight=0.08,
        readout_gemm_precision=spec.readout_gemm_precision,  # type: ignore[arg-type]
    )
    tokens_per_step = spec.batch_size * spec.seq_len
    model = muon_opt = adam_opt = x = y = loss = None
    try:
        model = Radical2Pow16WaveletCholeskyGPT(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        muon_opt, adam_opt = build_radical_2pow16_optimizers(model)
        x = torch.randint(0, cfg.valid_vocab_size, (spec.batch_size, spec.seq_len), device=device)
        y = torch.randint(0, cfg.valid_vocab_size, (spec.batch_size, spec.seq_len), device=device)

        for s in range(warmup_steps + timed_steps):
            if s == warmup_steps:
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
            muon_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, targets=y, return_logits=False)
            assert loss is not None
            loss.backward()
            muon_opt.step()
            adam_opt.step()
            if s % 2 == 0:
                model.pyramid.retract_kraus_generators()

        torch.cuda.synchronize(device)
        dt = (time.perf_counter() - t0) / float(max(1, timed_steps))
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        tok_s = tokens_per_step / dt
        loss_val = float(loss.detach().item())
        del model, muon_opt, adam_opt, x, y, loss
        gc.collect()
        torch.cuda.empty_cache()
        return {
            **asdict(spec),
            "status": "OK",
            "params_m": round(n_params / 1e6, 3),
            "tokens_per_step": tokens_per_step,
            "step_ms": round(dt * 1000.0, 2),
            "tok_per_s": round(tok_s, 1),
            "peak_vram_gb": round(peak_gb, 3),
            "loss": round(loss_val, 4),
        }
    except RuntimeError as exc:
        err_msg = str(exc)[:100]
        exc.__traceback__ = None
        del model, muon_opt, adam_opt, x, y, loss
        gc.collect()
        torch.cuda.empty_cache()
        return {
            **asdict(spec),
            "status": f"OOM_OR_ERR: {err_msg}",
            "params_m": 0.0,
            "tokens_per_step": tokens_per_step,
            "step_ms": 0.0,
            "tok_per_s": 0.0,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated(device) / (1024**3), 3),
            "loss": 0.0,
        }


def default_candidate_specs() -> list[CandidateScaleSpec]:
    """Return candidate configurations for the 15 GB / 24,576-token batch sizing study."""
    return [
        CandidateScaleSpec("w25_baseline_b16_d256_L6_bf16", 16, 512, 256, 6, 8, 64, 64, 16, 1.5, False, "bf16"),
        CandidateScaleSpec("w26_b48_d256_L6_dyadic_bf16", 48, 512, 256, 6, 8, 64, 64, 16, 1.5, True, "bf16"),
        CandidateScaleSpec("w26_b48_d256_L6_dyadic_fp8", 48, 512, 256, 6, 8, 64, 64, 16, 1.5, True, "fp8"),
        CandidateScaleSpec("w26_b48_d256_L6_dyadic_int8", 48, 512, 256, 6, 8, 64, 64, 16, 1.5, True, "int8"),
        CandidateScaleSpec("w26_b48_d384_L6_dyadic_bf16", 48, 512, 384, 6, 6, 96, 64, 8, 1.5, True, "bf16"),
        CandidateScaleSpec("w26_b48_d384_L8_dyadic_bf16", 48, 512, 384, 8, 6, 96, 64, 8, 1.5, True, "bf16"),
        CandidateScaleSpec("w26_b48_d384_L8_dyadic_fp8", 48, 512, 384, 8, 6, 96, 64, 8, 1.5, True, "fp8"),
        CandidateScaleSpec("w26_b48_d512_L8_dyadic_bf16", 48, 512, 512, 8, 8, 128, 64, 8, 1.5, True, "bf16"),
        CandidateScaleSpec("w26_b48_d512_L8_dyadic_fp8", 48, 512, 512, 8, 8, 128, 64, 8, 1.5, True, "fp8"),
        CandidateScaleSpec("w26_b48_d512_L12_dyadic_bf16", 48, 512, 512, 12, 8, 128, 64, 8, 1.5, True, "bf16"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="RTX 5060 Ti 15 GB / 24,576-Token Batch Network Sizer.")
    parser.add_argument("--vram-fraction", type=float, default=0.95)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=5)
    parser.add_argument("--json-out", type=str, default=".cache/rtx5060ti_15gb_batch24576_network_sizer.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(args.vram_fraction), device=0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dtypes = verify_bfloat16_activations(device) if device.type == "cuda" else {}
    print("=== Activation Dtype Audit under `torch.amp.autocast('cuda', dtype=torch.bfloat16)` ===")
    for k, v in dtypes.items():
        print(f"  {k:15s}: {v}")

    rows: list[dict[str, Any]] = []
    print("\n| Candidate Config | Params (`M`) | Batch Tokens | Prec | Step (`ms`) | Throughput (`tok/s`) | Peak VRAM (`GB`) | Status |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for spec in default_candidate_specs():
        res = profile_candidate_spec(spec, device, warmup_steps=args.warmup_steps, timed_steps=args.timed_steps)
        rows.append(res)
        print(
            f"| `{res['name']}` | `{res['params_m']:.2f}M` | `{res['tokens_per_step']}` | `{res['readout_gemm_precision']}` "
            f"| `{res['step_ms']:.1f}` | **`{res['tok_per_s']:,.0f}`** | `{res['peak_vram_gb']:.2f} GB` | `{res['status']}` |"
        )

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"activation_dtypes": dtypes, "candidates": rows}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
