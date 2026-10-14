"""Example 6: Fused Triton Flash-Born Attention vs. Materialized Born Attention Benchmark.

Demonstrates:
1. Forward + Backward numerical equivalence between Materialized Born Attention and
   the custom SRAM-tiled Triton Flash-Born Attention kernel (`use_flash_born=True`).
2. Peak GPU VRAM savings (eliminating the B x H x T x T attention matrix allocation)
   and execution speed across context lengths T = 512, 1024, 2048.
"""

from __future__ import annotations

import copy
import time

import torch

from qdlblocks import HAS_TRITON, SubspaceProjectorBornAttention


def benchmark_attention_mode(
    attn: SubspaceProjectorBornAttention,
    x: torch.Tensor,
    psi_0: torch.Tensor,
    warmup: int = 3,
    iters: int = 10,
) -> tuple[float, float]:
    """Measure average forward+backward step latency (ms) and peak allocated VRAM (MB)."""
    device = x.device
    for _ in range(warmup):
        attn.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        out = attn(x, psi_0=psi_0)
        out.sum().backward()
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    base_mem = torch.cuda.memory_allocated(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        attn.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        out = attn(x, psi_0=psi_0)
        out.sum().backward()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iters

    peak_delta_mb = (torch.cuda.max_memory_allocated(device) - base_mem) / (1024**2)
    return elapsed_ms, peak_delta_mb


def run_flash_born_benchmark() -> None:
    if not (HAS_TRITON and torch.cuda.is_available()):
        print("CUDA GPU and Triton required for Flash-Born benchmark.")
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print("====================================================================================")
    print("   Fused Triton Flash-Born Attention Benchmark (Zero T x T Matrix in VRAM)")
    print(f"   GPU: {gpu_name}")
    print("====================================================================================")
    print(
        f"{'Batch x Heads x SeqLen':24s} | {'Materialized (ms / MB)':24s} | "
        f"{'Flash-Born Triton (ms / MB)':27s} | {'VRAM Saved'}"
    )
    print("-" * 88)

    B = 8
    embed_dim = 1024
    num_heads = 16  # head_dim = 64

    for seq_len in (512, 1024, 2048):
        torch.manual_seed(42)
        attn_std = SubspaceProjectorBornAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            block_size=seq_len,
            is_causal=True,
            learnable_temperature=True,
            value_residual_mixing=True,
            use_flash_born=False,
        ).to(device, dtype=torch.bfloat16)

        attn_flash = copy.deepcopy(attn_std)
        attn_flash.use_flash_born = True

        x = torch.randn(B, seq_len, embed_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        psi_0 = torch.randn(B, seq_len, embed_dim, device=device, dtype=torch.bfloat16)

        std_ms, std_mb = benchmark_attention_mode(attn_std, x, psi_0)
        flash_ms, flash_mb = benchmark_attention_mode(attn_flash, x, psi_0)

        reduction = (1.0 - flash_mb / max(std_mb, 1e-5)) * 100.0
        config_str = f"B={B}, H={num_heads}, T={seq_len:<4d}"
        std_str = f"{std_ms:6.2f} ms / {std_mb:7.1f} MB"
        flash_str = f"{flash_ms:6.2f} ms / {flash_mb:7.1f} MB"
        print(f"{config_str:24s} | {std_str:24s} | {flash_str:27s} | {reduction:6.1f}% less VRAM")

    print("-" * 88)


if __name__ == "__main__":
    run_flash_born_benchmark()
