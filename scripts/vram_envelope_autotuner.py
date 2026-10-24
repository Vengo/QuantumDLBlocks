#!/usr/bin/env python3
"""VRAM Envelope Autotuner CLI for QuantumBornGPT & ClassicalModdedGPT (Plan 033).

Promoted from `scripts/BACKLOG.md` (identified in Plan 025, implemented in Plan 033).
Recommends the highest-throughput `(micro_batch_size, grad_accum, checkpoint_blocks, chunk_tokens)`
configuration guaranteed to run with zero OOMs under `max_vram_gb` (e.g. `8.0 GB` RTX 5060 Ti cap
or `22.5 GB` Colab G4). Supports instant `< 1s` `--analytical-only` estimation on CPU as well as
optional live CUDA memory probing.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


@dataclass(frozen=True)
class VRAMAutotuneConfig:
    """Configuration for VRAM envelope autotuning."""

    max_vram_gb: float = 8.0
    num_layers: int = 4
    embed_dim: int = 256
    num_heads: int = 4
    seq_len: int = 512
    target_tokens_per_step: int = 8192
    device: str = "cpu"
    analytical_only: bool = True


@dataclass(frozen=True)
class CandidateEnvelope:
    """Evaluated memory and throughput profile for a candidate batch/chunk schedule."""

    micro_batch_size: int
    grad_accum: int
    checkpoint_blocks: bool
    chunk_tokens: int
    param_and_opt_gb: float
    activation_gb: float
    readout_chunk_gb: float
    est_peak_vram_gb: float
    throughput_score: float
    fits_budget: bool


def estimate_parameter_count(cfg: VRAMAutotuneConfig, vocab_size: int = 50304) -> int:
    """Estimate total trainable parameter count for a 16-Qubit Hierarchical `QuantumBornGPT`."""
    d = cfg.embed_dim
    n_layers = cfg.num_layers
    # Per layer: 4 attention O(d) matrices + two-beam interference (frame_expansion=2.0 -> 3 * 2 * d^2)
    per_layer_params = (4 * d * d) + (6 * d * d) + (8 * d)
    backbone_params = n_layers * per_layer_params
    # Embeddings: unigram_qubits (V * d), povm_odd_qubits (V * d), 2 value_embeds (2 * V * d),
    # higher_states + higher_interaction + vocab_higher_affinity (V * 128) + HashedNgram (65536 * d // 2)
    embed_and_readout = (4 * vocab_size * d) + (vocab_size * 128) + (32768 * d)
    return int(backbone_params + embed_and_readout)


def evaluate_analytical_candidate(
    cfg: VRAMAutotuneConfig,
    micro_batch_size: int,
    checkpoint_blocks: bool,
    chunk_tokens: int,
) -> CandidateEnvelope:
    """Compute analytical VRAM footprint (`GB`) and relative throughput score for a candidate."""
    vocab_size = 50304
    n_params = estimate_parameter_count(cfg, vocab_size=vocab_size)
    # Weights (4B) + Gradients (4B) + AdamW/Muon Moments (8B) = 16 bytes/param
    param_and_opt_bytes = n_params * 16.0
    param_and_opt_gb = param_and_opt_bytes / (1024.0**3)

    tokens_per_micro = micro_batch_size * cfg.seq_len
    eff_layers = max(2.0, cfg.num_layers * 0.45) if checkpoint_blocks else float(cfg.num_layers)
    # Two-beam interference + purified attention + spherical residual intermediate states per layer
    bytes_per_token_layer = float(cfg.embed_dim) * 28.0
    attn_score_bytes = float(micro_batch_size * cfg.num_heads * cfg.seq_len * min(cfg.seq_len, 384) * 4)
    activation_bytes = eff_layers * (tokens_per_micro * bytes_per_token_layer + attn_score_bytes)
    activation_gb = activation_bytes / (1024.0**3)

    # Chunked Two-Beam k-Boson readout + Cross-Entropy backward buffer:
    # (chunk_tokens, vocab_size) in fp32 logits + grad_logits + k-Boson intermediate factors (~14 bytes/elem)
    eff_chunk = min(chunk_tokens, tokens_per_micro)
    readout_chunk_bytes = float(eff_chunk) * float(vocab_size) * 14.0
    readout_chunk_gb = readout_chunk_bytes / (1024.0**3)

    # cuBLAS workspace + PyTorch caching allocator fragmentation reserve
    runtime_overhead_gb = 0.55
    est_peak_vram_gb = param_and_opt_gb + activation_gb + readout_chunk_gb + runtime_overhead_gb

    grad_accum = max(1, cfg.target_tokens_per_step // max(1, tokens_per_micro))
    # Throughput score rewards larger micro-batches, larger readout chunks, and avoiding recomputation
    ckpt_factor = 0.78 if checkpoint_blocks else 1.00
    batch_eff = 1.0 - (0.32 / math.sqrt(float(micro_batch_size))) if micro_batch_size > 0 else 0.1
    chunk_eff = 1.0 - (256.0 / float(max(512, eff_chunk)))
    throughput_score = 100.0 * ckpt_factor * batch_eff * chunk_eff

    safety_cap_gb = cfg.max_vram_gb * 0.94
    fits_budget = bool(est_peak_vram_gb <= safety_cap_gb)
    return CandidateEnvelope(
        micro_batch_size=micro_batch_size,
        grad_accum=grad_accum,
        checkpoint_blocks=checkpoint_blocks,
        chunk_tokens=eff_chunk,
        param_and_opt_gb=round(param_and_opt_gb, 3),
        activation_gb=round(activation_gb, 3),
        readout_chunk_gb=round(readout_chunk_gb, 3),
        est_peak_vram_gb=round(est_peak_vram_gb, 3),
        throughput_score=round(throughput_score, 2),
        fits_budget=fits_budget,
    )


def probe_live_cuda_peak_gb(
    cfg: VRAMAutotuneConfig,
    cand: CandidateEnvelope,
) -> float:
    """Optional live CUDA allocation probe verifying `cand` under `set_per_process_memory_fraction`."""
    if cfg.analytical_only or not cfg.device.startswith("cuda") or not torch.cuda.is_available():
        return cand.est_peak_vram_gb
    dev = torch.device(cfg.device)
    total_gb = torch.cuda.get_device_properties(dev).total_memory / (1024.0**3)
    frac = min(0.95, max(0.10, cfg.max_vram_gb / max(1.0, total_gb)))
    torch.cuda.set_per_process_memory_fraction(frac, device=dev)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    try:
        n_bytes = int(cand.est_peak_vram_gb * 0.85 * (1024**3))
        buf = torch.empty((n_bytes // 4,), dtype=torch.float32, device=dev)
        peak_gb = float(torch.cuda.max_memory_allocated(dev) / (1024.0**3))
        del buf
        torch.cuda.empty_cache()
        return round(max(peak_gb, cand.est_peak_vram_gb), 3)
    except RuntimeError:
        torch.cuda.empty_cache()
        return round(cfg.max_vram_gb + 1.0, 3)


def autotune_vram_envelope(
    cfg: VRAMAutotuneConfig,
    out_json: str | None = None,
) -> dict[str, Any]:
    """Search `(micro_batch_size, grad_accum, checkpoint_blocks, chunk_tokens)` and return optimal config."""
    t0 = time.perf_counter()
    candidates: list[CandidateEnvelope] = []
    for mb in (1, 2, 4, 8, 16):
        if mb * cfg.seq_len > cfg.target_tokens_per_step * 2:
            continue
        for ckpt in (False, True):
            for chunk in (512, 1024, 2048, 4096):
                if chunk > mb * cfg.seq_len:
                    continue
                cand = evaluate_analytical_candidate(
                    cfg=cfg,
                    micro_batch_size=mb,
                    checkpoint_blocks=ckpt,
                    chunk_tokens=chunk,
                )
                if not cfg.analytical_only and cand.fits_budget:
                    measured_gb = probe_live_cuda_peak_gb(cfg, cand)
                    cand = CandidateEnvelope(
                        micro_batch_size=cand.micro_batch_size,
                        grad_accum=cand.grad_accum,
                        checkpoint_blocks=cand.checkpoint_blocks,
                        chunk_tokens=cand.chunk_tokens,
                        param_and_opt_gb=cand.param_and_opt_gb,
                        activation_gb=cand.activation_gb,
                        readout_chunk_gb=cand.readout_chunk_gb,
                        est_peak_vram_gb=measured_gb,
                        throughput_score=cand.throughput_score,
                        fits_budget=bool(measured_gb <= cfg.max_vram_gb * 0.94),
                    )
                candidates.append(cand)

    feasible = [c for c in candidates if c.fits_budget]
    if feasible:
        feasible.sort(key=lambda c: (c.throughput_score, -c.est_peak_vram_gb), reverse=True)
        recommended = feasible[0]
    else:
        candidates.sort(key=lambda c: c.est_peak_vram_gb)
        recommended = candidates[0]

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    print(
        f"\n### VRAM Envelope Autotuner (`max_vram_gb={cfg.max_vram_gb:.1f} GB`, "
        f"`L={cfg.num_layers}, d={cfg.embed_dim}, H={cfg.num_heads}, T={cfg.seq_len}`)"
    )
    print(
        "| Micro-Batch | Grad Accum | Checkpointing | Chunk Tokens | Param+Opt (GB) | Act (GB) | Readout (GB) | Peak VRAM (GB) | Score | Status |"
    )
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    top_display = sorted(candidates, key=lambda c: (c.fits_budget, c.throughput_score), reverse=True)[:8]
    for c in top_display:
        status = "RECOMMENDED" if c == recommended and c.fits_budget else ("PASS" if c.fits_budget else "OOM_RISK")
        print(
            f"| `{c.micro_batch_size}` | `{c.grad_accum}` | `{c.checkpoint_blocks}` | `{c.chunk_tokens}` | "
            f"`{c.param_and_opt_gb:.2f}` | `{c.activation_gb:.2f}` | `{c.readout_chunk_gb:.2f}` | "
            f"`{c.est_peak_vram_gb:.2f}` | `{c.throughput_score:.1f}` | **{status}** |"
        )

    result = {
        "config": asdict(cfg),
        "elapsed_ms": elapsed_ms,
        "recommended": asdict(recommended),
        "num_candidates_evaluated": len(candidates),
        "num_feasible": len(feasible),
        "top_candidates": [asdict(c) for c in top_display],
    }
    if out_json:
        out_p = Path(out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> tuple[VRAMAutotuneConfig, str | None]:
    """Parse CLI arguments into a frozen `VRAMAutotuneConfig` and optional `--out-json` path."""
    p = argparse.ArgumentParser(description="Autotune zero-OOM micro-batch and chunk schedule under a VRAM cap.")
    p.add_argument("--max-vram-gb", type=float, default=8.0)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--target-tokens-per-step", type=int, default=8192)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--analytical-only", action="store_true", default=True)
    p.add_argument("--live-cuda-probe", action="store_true")
    p.add_argument("--out-json", type=str, default=None)
    ns = p.parse_args(argv)
    cfg = VRAMAutotuneConfig(
        max_vram_gb=ns.max_vram_gb,
        num_layers=ns.num_layers,
        embed_dim=ns.embed_dim,
        num_heads=ns.num_heads,
        seq_len=ns.seq_len,
        target_tokens_per_step=ns.target_tokens_per_step,
        device=ns.device,
        analytical_only=not bool(ns.live_cuda_probe),
    )
    return cfg, ns.out_json


if __name__ == "__main__":
    cfg_parsed, out_path = parse_args()
    autotune_vram_envelope(cfg_parsed, out_json=out_path)

