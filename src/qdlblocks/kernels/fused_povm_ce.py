"""Analytic-gradient L2-tiled POVM cross-entropy kernel and roofline estimator.

Eliminates full ``[N, V]`` logit tensor materialization in HBM by decomposing both
the forward log-sum-exp reduction and the exact analytical backward pass into
L2-cache-sized token tiles ``[B_tile, V]``. Only the 1D log-sum-exp vector
``lse`` of shape ``(N,)`` is saved between forward and backward passes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def estimate_povm_head_roofline(
    num_tokens: int,
    seed_dim: int,
    vocab_size: int,
    tile_size: int,
    *,
    dtype_bytes: int = 2,
    l2_cache_bytes: int = 96 * 1024 * 1024,
) -> dict[str, float | bool]:
    """Estimate FLOPs, HBM byte traffic, and arithmetic intensity for tiled POVM CE.

    Args:
        num_tokens: Total tokens ``N = B * T`` in the micro-batch.
        seed_dim: Seed Hilbert embedding dimension ``D_seed``.
        vocab_size: Vocabulary size ``V``.
        tile_size: Token tile size ``B_tile``.
        dtype_bytes: Element size in bytes for GEMM operands/buffers (default: 2 for BF16).
        l2_cache_bytes: On-chip L2 cache capacity in bytes (default: 96 MiB for Blackwell).

    Returns:
        Dictionary with ``gemm_flops``, ``total_hbm_bytes``, ``arithmetic_intensity``,
        ``fits_in_l2``, ``peak_logit_buffer_mb``, and ``saved_activation_kb``.
    """
    if num_tokens < 1:
        raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
    if seed_dim < 1:
        raise ValueError(f"seed_dim must be >= 1, got {seed_dim}")
    if vocab_size < 1:
        raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
    if tile_size < 1:
        raise ValueError(f"tile_size must be >= 1, got {tile_size}")
    if dtype_bytes < 1:
        raise ValueError(f"dtype_bytes must be >= 1, got {dtype_bytes}")
    if l2_cache_bytes < 1:
        raise ValueError(f"l2_cache_bytes must be >= 1, got {l2_cache_bytes}")

    eff_tile = min(num_tokens, tile_size)
    gemm_flops = float(8 * num_tokens * seed_dim * vocab_size)
    tile_working_set_bytes = float(eff_tile * vocab_size * dtype_bytes)
    fits_in_l2 = bool(tile_working_set_bytes <= float(l2_cache_bytes))
    base_hbm_bytes = float(
        (4 * num_tokens * seed_dim + 4 * vocab_size * seed_dim) * dtype_bytes
        + num_tokens * 12
    )
    if fits_in_l2:
        total_hbm_bytes = base_hbm_bytes
    else:
        spill_hbm_bytes = float(6 * num_tokens * vocab_size * dtype_bytes)
        total_hbm_bytes = base_hbm_bytes + spill_hbm_bytes
    arithmetic_intensity = gemm_flops / max(1.0, total_hbm_bytes)
    peak_logit_buffer_mb = tile_working_set_bytes / (1024.0 * 1024.0)
    saved_activation_kb = float(num_tokens * 4) / 1024.0
    return {
        "gemm_flops": gemm_flops,
        "total_hbm_bytes": total_hbm_bytes,
        "arithmetic_intensity": round(arithmetic_intensity, 2),
        "fits_in_l2": fits_in_l2,
        "peak_logit_buffer_mb": round(peak_logit_buffer_mb, 3),
        "saved_activation_kb": round(saved_activation_kb, 3),
    }


measure_povm_ce_roofline = estimate_povm_head_roofline


class _FusedTiledPOVMCEFn(torch.autograd.Function):
    """Custom autograd function for analytic-gradient L2-tiled POVM cross-entropy."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        z: torch.Tensor,
        m_vocab: torch.Tensor,
        inv_temp: torch.Tensor,
        targets: torch.Tensor,
        tile_size: int,
    ) -> torch.Tensor:
        N = z.size(0)
        lse = torch.empty(N, device=z.device, dtype=torch.float32)
        loss_sum = torch.zeros((), device=z.device, dtype=torch.float32)
        beta_f = float(inv_temp.detach().float().item())
        m_f = m_vocab.detach().float()
        for start in range(0, N, tile_size):
            end = min(start + tile_size, N)
            z_b = z[start:end].detach().float()
            y_b = targets[start:end].long()
            logits_b = F.linear(z_b, m_f) * beta_f
            lse_b = torch.logsumexp(logits_b, dim=-1)
            tgt_logits_b = logits_b.gather(1, y_b.unsqueeze(1)).squeeze(1)
            lse[start:end] = lse_b
            loss_sum = loss_sum + torch.sum(lse_b - tgt_logits_b)
        ctx.save_for_backward(z, m_vocab, inv_temp, targets, lse)
        ctx.tile_size = tile_size
        return loss_sum / float(N)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        z, m_vocab, inv_temp, targets, lse = ctx.saved_tensors
        N, D = z.shape
        tile_size = ctx.tile_size
        beta_f = float(inv_temp.detach().float().item())
        scale = float(grad_output.detach().float().item()) / float(N)
        m_f = m_vocab.detach().float()
        dz = torch.empty((N, D), device=z.device, dtype=torch.float32)
        dm = torch.zeros_like(m_f)
        dbeta = torch.zeros((), device=z.device, dtype=torch.float32)
        for start in range(0, N, tile_size):
            end = min(start + tile_size, N)
            z_b = z[start:end].detach().float()
            y_b = targets[start:end].long()
            lse_b = lse[start:end].unsqueeze(1)
            raw_dot = F.linear(z_b, m_f)
            probs_b = torch.exp(raw_dot * beta_f - lse_b)
            probs_b.scatter_add_(
                1,
                y_b.unsqueeze(1),
                torch.full(
                    (end - start, 1), -1.0, device=z.device, dtype=torch.float32
                ),
            )
            dbeta = dbeta + torch.sum(probs_b * raw_dot) * scale
            probs_b.mul_(scale * beta_f)
            dz[start:end] = torch.matmul(probs_b, m_f)
            dm.addmm_(probs_b.transpose(0, 1), z_b)
        return (
            dz.to(z.dtype),
            dm.to(m_vocab.dtype),
            dbeta.reshape_as(inv_temp).to(inv_temp.dtype),
            None,
            None,
        )


def fused_tiled_povm_cross_entropy(
    z: torch.Tensor,
    m_vocab: torch.Tensor,
    inv_temp: torch.Tensor,
    targets: torch.Tensor,
    *,
    tile_size: int = 1024,
) -> torch.Tensor:
    """Compute POVM Born cross-entropy loss with L2-tiled analytic gradients.

    Args:
        z: Seed Hilbert states of shape ``(N, D)``.
        m_vocab: Vocabulary POVM dictionary of shape ``(V, D)``.
        inv_temp: Scalar inverse Born temperature ``beta > 0``.
        targets: Target token indices of shape ``(N,)`` in ``[0, V - 1]``.
        tile_size: Token tile size ``B_tile >= 1``.

    Returns:
        Scalar FP32 mean cross-entropy loss.
    """
    if tile_size < 1:
        raise ValueError(f"tile_size must be >= 1, got {tile_size}")
    if z.ndim != 2:
        raise ValueError(f"z must be 2D (N, D), got shape {tuple(z.shape)}")
    if m_vocab.ndim != 2:
        raise ValueError(f"m_vocab must be 2D (V, D), got shape {tuple(m_vocab.shape)}")
    if z.size(0) < 1 or m_vocab.size(0) < 1:
        raise ValueError("z and m_vocab must have non-zero leading dimensions")
    if z.size(1) != m_vocab.size(1):
        raise ValueError(
            f"Feature dimension mismatch: z has D={z.size(1)}, m_vocab has D={m_vocab.size(1)}"
        )
    if targets.ndim != 1 or targets.size(0) != z.size(0):
        raise ValueError(
            f"targets must be 1D of length N={z.size(0)}, got shape {tuple(targets.shape)}"
        )
    if inv_temp.numel() != 1:
        raise ValueError(
            f"inv_temp must be a scalar or single-element tensor, got shape {tuple(inv_temp.shape)}"
        )

    return _FusedTiledPOVMCEFn.apply(z, m_vocab, inv_temp, targets, int(tile_size))


__all__ = [
    "estimate_povm_head_roofline",
    "measure_povm_ce_roofline",
    "fused_tiled_povm_cross_entropy",
]
