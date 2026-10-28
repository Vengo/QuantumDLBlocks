#!/usr/bin/env python3
"""Multi-Head Superposition Attention, Readout Backward, & Eval Speed Profiler (Plan 059 / Wave 44).

Benchmarks:
1. Attention step latency (ms/step), throughput (`tok/s`), TFLOP/s, total `torch.matmul` count,
   and `(T x T)` `torch.matmul` count across `attn_evolution_mode`:
   - `"multihead_gram"` (3 `(T x T)` matmuls + 2 `(K x K)` Gram matmuls = 5 total `torch.matmul` calls)
   - `"legacy_krylov_11mm"` (11 `(T x T)` matmuls + 2 `(K x K)` Gram matmuls = 13 total `torch.matmul` calls)
   - `"adhoc_poly_gamma"` (poly-gamma coherent ablation)
   at sequence lengths `T in {512, 1024}` and subhead counts `K in {2, 4}`.
2. Readout forward+backward latency and single-pass `pow` call reduction in
   `MixedStateKroneckerBornChunkedNLLFn` vs. `CalibratedKroneckerBornChunkedNLLFn`.
3. Validation evaluation throughput (`evaluate_modded_nanogpt_validation`).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile
import time
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from qdlblocks.layers.multihead_superposition_mixed_state_attn import (
    MultiHeadSuperpositionAttnConfig,
    MultiHeadSuperpositionMixedStateAttention,
)
from qdlblocks.models.calibrated_kronecker_born_gpt import (
    CalibratedKroneckerBornChunkedNLLFn,
)
from qdlblocks.models.multihead_mixed_state_kronecker_gpt import (
    MixedStateKroneckerBornChunkedNLLFn,
    MultiHeadMixedStateKroneckerConfig,
    MultiHeadMixedStateKroneckerGPT,
)
from qdlblocks.models.ngram_calibrated_kronecker_born_gpt import (
    evaluate_modded_nanogpt_validation,
)

# ==============================================================================
# Rule 7 Named Constants (Plan 059 / Wave 44)
# ==============================================================================
DEFAULT_SEQ_LENS_CSV: str = "512,1024"
DEFAULT_MODES_CSV: str = "multihead_gram,legacy_krylov_11mm,adhoc_poly_gamma"
DEFAULT_SUBHEADS_CSV: str = "2,4"
DEFAULT_PROFILE_BATCH_SIZE: int = 2
DEFAULT_PROFILE_NUM_MEANINGS: int = 2
DEFAULT_PROFILE_NUM_FACTORS: int = 4
DEFAULT_PROFILE_D_REGISTER: int = 16
DEFAULT_PROFILE_VOCAB_SIZE: int = 512
DEFAULT_PROFILE_CHUNK_SIZE: int = 128
DEFAULT_WARMUP_STEPS: int = 2
DEFAULT_BENCH_STEPS: int = 4
SMOKE_SEQ_LEN: int = 64
SMOKE_VAL_TOKENS: int = 2048
PROFILE_VAL_TOKENS: int = 8192
FLOPS_TO_TFLOPS: float = 1e-12


@dataclass(frozen=True)
class AttentionProfileRecord:
    """Telemetry record for one attention evolution mode and sequence length."""

    attn_evolution_mode: str
    seq_len: int
    n_subheads: int
    batch_size: int
    total_matmul_calls: int
    txt_matmul_calls: int
    fwd_bwd_ms_per_step: float
    tokens_per_sec: float
    approx_tflops_per_sec: float
    subhead_purity_mean: float
    reduced_state_purity_mean: float
    gram_trace_mean: float


@dataclass(frozen=True)
class ReadoutBackwardProfileRecord:
    """Telemetry record comparing readout forward+backward op counts and latency."""

    readout_variant: str
    tokens_count: int
    vocab_size: int
    polysemy_rank: int
    pow_call_count: int
    bmm_call_count: int
    fwd_bwd_ms: float


def count_attention_matmul_calls(
    attn_layer: MultiHeadSuperpositionMixedStateAttention,
    psi_re: torch.Tensor,
    psi_im: torch.Tensor,
) -> tuple[int, int]:
    """Count total `torch.matmul` calls and `(T x T)` `torch.matmul` calls in one forward pass."""
    orig_matmul = torch.matmul
    total_calls = 0
    txt_calls = 0
    seq_len = psi_re.size(1)

    def _counting_matmul(a: torch.Tensor, b: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal total_calls, txt_calls
        total_calls += 1
        if a.ndim >= 2 and b.ndim >= 2:
            if (a.size(-2) == seq_len and a.size(-1) == seq_len) or (
                a.size(-2) == seq_len and b.size(-1) == seq_len
            ):
                txt_calls += 1
        return orig_matmul(a, b, *args, **kwargs)

    with torch.no_grad(), patch("torch.matmul", side_effect=_counting_matmul):
        attn_layer(psi_re, psi_im)

    return total_calls, txt_calls


def profile_single_attention_config(
    mode: str,
    seq_len: int,
    n_subheads: int,
    batch_size: int,
    warmup_steps: int,
    bench_steps: int,
    device: torch.device,
) -> AttentionProfileRecord:
    """Benchmark a single `(mode, seq_len, n_subheads)` attention configuration."""
    cfg = MultiHeadSuperpositionAttnConfig(
        num_meanings=DEFAULT_PROFILE_NUM_MEANINGS,
        num_factors=DEFAULT_PROFILE_NUM_FACTORS,
        d_register=DEFAULT_PROFILE_D_REGISTER,
        n_subheads_per_factor=n_subheads,
        attn_evolution_mode=mode,
        max_seq_len=max(seq_len, 64),
    )
    layer = MultiHeadSuperpositionMixedStateAttention(cfg).to(device)
    d_complex = cfg.d_model
    psi_re = torch.randn(batch_size, seq_len, d_complex, device=device, requires_grad=True)
    psi_im = torch.randn(batch_size, seq_len, d_complex, device=device, requires_grad=True)

    total_mm, txt_mm = count_attention_matmul_calls(layer, psi_re.detach(), psi_im.detach())

    for _ in range(warmup_steps):
        out_re, out_im, _, _ = layer(psi_re, psi_im)
        loss = (out_re.square() + out_im.square()).mean()
        loss.backward()
        layer.zero_grad(set_to_none=True)
        psi_re.grad = None
        psi_im.grad = None

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(bench_steps):
        out_re, out_im, _, _ = layer(psi_re, psi_im)
        loss = (out_re.square() + out_im.square()).mean()
        loss.backward()
        layer.zero_grad(set_to_none=True)
        psi_re.grad = None
        psi_im.grad = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = max(time.perf_counter() - t0, 1e-9)

    ms_per_step = (elapsed_s * 1000.0) / float(bench_steps)
    total_tokens = batch_size * seq_len * bench_steps
    tok_per_s = float(total_tokens) / elapsed_s

    flops_per_step = (
        6.0
        * float(batch_size)
        * float(cfg.n_subheads_total)
        * float(txt_mm)
        * float(seq_len)
        * float(seq_len)
        * float(cfg.d_register)
    )
    tflops_per_s = (flops_per_step * float(bench_steps) * FLOPS_TO_TFLOPS) / elapsed_s
    tel = layer.compute_subhead_purity_telemetry()

    return AttentionProfileRecord(
        attn_evolution_mode=mode,
        seq_len=seq_len,
        n_subheads=n_subheads,
        batch_size=batch_size,
        total_matmul_calls=total_mm,
        txt_matmul_calls=txt_mm,
        fwd_bwd_ms_per_step=ms_per_step,
        tokens_per_sec=tok_per_s,
        approx_tflops_per_sec=tflops_per_s,
        subhead_purity_mean=float(tel["subhead_gram_purity_mean"]),
        reduced_state_purity_mean=float(tel["reduced_density_purity_mean"]),
        gram_trace_mean=float(tel["subhead_gram_trace_mean"]),
    )


def profile_readout_backward_comparison(
    tokens_count: int,
    vocab_size: int,
    polysemy_rank: int,
    device: torch.device,
) -> list[ReadoutBackwardProfileRecord]:
    """Compare `pow` and `bmm` call counts in MixedState vs. Calibrated Kronecker chunked NLL."""
    phi_re = torch.randn(tokens_count, polysemy_rank, 4, 16, device=device)
    phi_im = torch.randn(tokens_count, polysemy_rank, 4, 16, device=device)
    norms = torch.sqrt((phi_re.square() + phi_im.square()).sum(dim=-1, keepdim=True).clamp_min(1e-8))
    phi_factors = torch.cat([phi_re / norms, phi_im / norms], dim=-1).detach().requires_grad_(True)

    psi_re = torch.randn(vocab_size, polysemy_rank, 4, 16, device=device)
    psi_im = torch.randn(vocab_size, polysemy_rank, 4, 16, device=device)
    p_norms = torch.sqrt((psi_re.square() + psi_im.square()).sum(dim=-1, keepdim=True).clamp_min(1e-8))
    psi_factors = torch.cat([psi_re / p_norms, psi_im / p_norms], dim=-1).detach().requires_grad_(True)

    inv_sqrt_r = 1.0 / float(polysemy_rank) ** 0.5
    branch_beta = torch.full((tokens_count, polysemy_rank), inv_sqrt_r, device=device, requires_grad=True)
    branch_alpha = torch.full((vocab_size, polysemy_rank), inv_sqrt_r, device=device, requires_grad=True)
    povm_weights = torch.ones((vocab_size,), device=device, requires_grad=True)
    targets = torch.randint(0, vocab_size, (tokens_count,), device=device)
    hyperparams = torch.tensor([2.0, 0.85, 0.30, 0.15], device=device)

    records: list[ReadoutBackwardProfileRecord] = []
    for variant_name, fn_cls in (
        ("mixed_density", MixedStateKroneckerBornChunkedNLLFn),
        ("calibrated_coherent", CalibratedKroneckerBornChunkedNLLFn),
    ):
        fn_cls.vocab_chunk_size = DEFAULT_PROFILE_CHUNK_SIZE
        pow_calls = 0
        bmm_calls = 0
        orig_pow = torch.pow
        orig_bmm = torch.bmm

        def _counting_pow(self_t: torch.Tensor, exponent: Any) -> torch.Tensor:
            nonlocal pow_calls
            pow_calls += 1
            return orig_pow(self_t, exponent)

        def _counting_bmm(a: torch.Tensor, b: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            nonlocal bmm_calls
            bmm_calls += 1
            return orig_bmm(a, b, *args, **kwargs)

        t0 = time.perf_counter()
        with patch("torch.pow", side_effect=_counting_pow), patch("torch.bmm", side_effect=_counting_bmm):
            loss = fn_cls.apply(
                phi_factors,
                psi_factors,
                branch_beta,
                branch_alpha,
                povm_weights,
                targets,
                hyperparams,
                None,
            )
            loss.backward()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        phi_factors.grad = None
        psi_factors.grad = None
        branch_beta.grad = None
        branch_alpha.grad = None
        povm_weights.grad = None

        records.append(
            ReadoutBackwardProfileRecord(
                readout_variant=variant_name,
                tokens_count=tokens_count,
                vocab_size=vocab_size,
                polysemy_rank=polysemy_rank,
                pow_call_count=pow_calls,
                bmm_call_count=bmm_calls,
                fwd_bwd_ms=elapsed_ms,
            )
        )
    return records


def profile_validation_throughput(
    vocab_size: int,
    seq_len: int,
    val_tokens: int,
    device: torch.device,
) -> dict[str, float]:
    """Benchmark `evaluate_modded_nanogpt_validation` throughput on a synthetic FineWeb shard."""
    cfg = MultiHeadMixedStateKroneckerConfig(
        vocab_size=vocab_size,
        valid_vocab_size=vocab_size,
        num_meanings=2,
        num_factors=4,
        factor_dim=16,
        num_layers=1,
        n_subheads_per_factor=2,
        max_seq_len=max(seq_len, 64),
        vocab_chunk_size=128,
    )
    model = MultiHeadMixedStateKroneckerGPT(cfg).to(device)

    with tempfile.TemporaryDirectory() as tmp_dir:
        shard_path = Path(tmp_dir) / "fineweb_val_000000.bin"
        header = np.zeros(256, dtype=np.int32)
        header[0] = 20240520
        header[1] = 1
        header[2] = val_tokens + 16
        tokens = np.random.randint(0, vocab_size, size=(val_tokens + 16,), dtype=np.uint16)
        with shard_path.open("wb") as f:
            f.write(header.tobytes())
            f.write(tokens.tobytes())

        t0 = time.perf_counter()
        metrics = evaluate_modded_nanogpt_validation(
            model=model,
            val_tokens=shard_path,
            max_eval_tokens=val_tokens,
            batch_size=2,
            seq_len=seq_len,
        )
        elapsed_s = max(time.perf_counter() - t0, 1e-9)

    eval_tok = float(metrics.get("eval_tokens", metrics.get("tokens_evaluated", val_tokens)))
    return {
        "val_ce": float(metrics["val_ce"]),
        "val_ppl": float(metrics["val_ppl"]),
        "tokens_evaluated": eval_tok,
        "eval_wall_s": elapsed_s,
        "eval_tokens_per_sec": eval_tok / elapsed_s,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the Multi-Head Mixed-State & Eval Speed Profiler."""
    parser = argparse.ArgumentParser(
        description="Profile Multi-Head Superposition Attention, Mixed-State Readout, and Validation Speed."
    )
    parser.add_argument(
        "--seq-lens",
        type=str,
        default=DEFAULT_SEQ_LENS_CSV,
        help="Comma-separated sequence lengths to profile (e.g. '512,1024').",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=DEFAULT_MODES_CSV,
        help="Comma-separated attention evolution modes.",
    )
    parser.add_argument(
        "--subheads",
        type=str,
        default=DEFAULT_SUBHEADS_CSV,
        help="Comma-separated subhead counts per factor K.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_PROFILE_BATCH_SIZE,
        help="Microbatch size B.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help="Warmup iterations per configuration.",
    )
    parser.add_argument(
        "--bench-steps",
        type=int,
        default=DEFAULT_BENCH_STEPS,
        help="Timed benchmark iterations per configuration.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a fast smoke profile at small T.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to write the structured JSON benchmark summary.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the attention, readout backward, and validation speed benchmarks."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke:
        seq_lens = [SMOKE_SEQ_LEN]
        subheads_list = [2]
        warmup_steps = 1
        bench_steps = 2
        val_tokens = SMOKE_VAL_TOKENS
    else:
        seq_lens = [int(x.strip()) for x in args.seq_lens.split(",") if x.strip()]
        subheads_list = [int(x.strip()) for x in args.subheads.split(",") if x.strip()]
        warmup_steps = args.warmup_steps
        bench_steps = args.bench_steps
        val_tokens = PROFILE_VAL_TOKENS

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    attn_records: list[AttentionProfileRecord] = []
    for seq_len in seq_lens:
        for mode in modes:
            k_candidates = subheads_list if mode == "multihead_gram" else [subheads_list[0]]
            for k_sub in k_candidates:
                rec = profile_single_attention_config(
                    mode=mode,
                    seq_len=seq_len,
                    n_subheads=k_sub,
                    batch_size=args.batch_size,
                    warmup_steps=warmup_steps,
                    bench_steps=bench_steps,
                    device=device,
                )
                attn_records.append(rec)
                print(
                    f"[ATTN] mode={rec.attn_evolution_mode:<18s} T={rec.seq_len:<4d} K={rec.n_subheads} | "
                    f"matmuls={rec.total_matmul_calls:2d} (TxT={rec.txt_matmul_calls:2d}) | "
                    f"ms/step={rec.fwd_bwd_ms_per_step:7.2f} | tok/s={rec.tokens_per_sec:9.1f} | "
                    f"purity={rec.reduced_state_purity_mean:.4f}"
                )

    readout_records = profile_readout_backward_comparison(
        tokens_count=min(seq_lens[0] * args.batch_size, 256),
        vocab_size=DEFAULT_PROFILE_VOCAB_SIZE,
        polysemy_rank=DEFAULT_PROFILE_NUM_MEANINGS,
        device=device,
    )
    for r_rec in readout_records:
        print(
            f"[READOUT] variant={r_rec.readout_variant:<20s} | "
            f"pow_calls={r_rec.pow_call_count:2d} | bmm_calls={r_rec.bmm_call_count:2d} | "
            f"fwd_bwd_ms={r_rec.fwd_bwd_ms:6.2f}"
        )

    eval_summary = profile_validation_throughput(
        vocab_size=DEFAULT_PROFILE_VOCAB_SIZE,
        seq_len=seq_lens[0],
        val_tokens=val_tokens,
        device=device,
    )
    print(
        f"[EVAL] val_ce={eval_summary['val_ce']:.4f} | "
        f"tokens={int(eval_summary['tokens_evaluated'])} | "
        f"tok/s={eval_summary['eval_tokens_per_sec']:.1f}"
    )

    payload = {
        "device": str(device),
        "attention_profiles": [asdict(r) for r in attn_records],
        "readout_backward_profiles": [asdict(r) for r in readout_records],
        "validation_throughput": eval_summary,
    }
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[SAVED] Wrote JSON profile summary to {out_path}")


if __name__ == "__main__":
    main()
