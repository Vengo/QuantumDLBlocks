"""JAX analytic-gradient L2-tiled POVM cross-entropy with custom VJP and lax.scan.

Computes POVM Born cross-entropy over token tiles of size ``tile_size`` using
``jax.lax.scan`` and ``@jax.custom_vjp``, saving only the 1D per-token
log-sum-exp array ``lse`` between forward and backward passes.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


def estimate_povm_head_roofline_jax(
    num_tokens: int,
    seed_dim: int,
    vocab_size: int,
    tile_size: int,
    *,
    dtype_bytes: int = 2,
    l2_cache_bytes: int = 96 * 1024 * 1024,
) -> dict[str, float | bool]:
    """Estimate FLOPs, HBM byte traffic, and arithmetic intensity for tiled POVM CE."""
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


estimate_povm_head_roofline = estimate_povm_head_roofline_jax
measure_povm_ce_roofline = estimate_povm_head_roofline_jax
measure_povm_ce_roofline_jax = estimate_povm_head_roofline_jax


def _forward_eval(
    z: jax.Array,
    m_vocab: jax.Array,
    inv_temp: jax.Array,
    targets: jax.Array,
    tile_size: int,
) -> tuple[jax.Array, jax.Array]:
    N, D = z.shape
    n_tiles = N // tile_size
    z_tiles = z.astype(jnp.float32).reshape(n_tiles, tile_size, D)
    y_tiles = targets.reshape(n_tiles, tile_size)
    m_f = m_vocab.astype(jnp.float32)
    beta_f = inv_temp.astype(jnp.float32).reshape(())

    def _scan_fwd(
        loss_acc: jax.Array,
        tile_inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        z_b, y_b = tile_inputs
        logits_b = (
            jnp.matmul(z_b, m_f.T, precision=jax.lax.Precision.HIGHEST) * beta_f
        )
        lse_b = jax.nn.logsumexp(logits_b, axis=-1)
        tgt_logits_b = jnp.take_along_axis(
            logits_b, y_b[:, None].astype(jnp.int32), axis=-1
        ).squeeze(-1)
        loss_acc = loss_acc + jnp.sum(lse_b - tgt_logits_b)
        return loss_acc, lse_b

    loss_sum, lse = jax.lax.scan(
        _scan_fwd,
        jnp.zeros((), dtype=jnp.float32),
        (z_tiles, y_tiles),
    )
    return loss_sum / float(N), lse


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def _fused_tiled_povm_ce_jax(
    z: jax.Array,
    m_vocab: jax.Array,
    inv_temp: jax.Array,
    targets: jax.Array,
    tile_size: int,
) -> jax.Array:
    loss, _ = _forward_eval(z, m_vocab, inv_temp, targets, tile_size)
    return loss


def _fused_tiled_povm_ce_jax_fwd(
    z: jax.Array,
    m_vocab: jax.Array,
    inv_temp: jax.Array,
    targets: jax.Array,
    tile_size: int,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
    loss, lse = _forward_eval(z, m_vocab, inv_temp, targets, tile_size)
    return loss, (z, m_vocab, inv_temp, targets, lse)


def _fused_tiled_povm_ce_jax_bwd(
    tile_size: int,
    res: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    g_out: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, None]:
    z, m_vocab, inv_temp, targets, lse = res
    N, D = z.shape
    V = m_vocab.shape[0]
    n_tiles = N // tile_size

    z_tiles = z.astype(jnp.float32).reshape(n_tiles, tile_size, D)
    y_tiles = targets.reshape(n_tiles, tile_size)
    m_f = m_vocab.astype(jnp.float32)
    beta_f = inv_temp.astype(jnp.float32).reshape(())
    scale = g_out.astype(jnp.float32).reshape(()) / float(N)

    def _scan_bwd(
        carry: tuple[jax.Array, jax.Array],
        tile_inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        dm_acc, dbeta_acc = carry
        z_b, y_b, lse_b = tile_inputs
        raw_dot = jnp.matmul(z_b, m_f.T, precision=jax.lax.Precision.HIGHEST)
        probs_b = jnp.exp(raw_dot * beta_f - lse_b[:, None])
        diff_b = probs_b - jax.nn.one_hot(y_b, V, dtype=jnp.float32)
        dbeta_acc = dbeta_acc + jnp.sum(diff_b * raw_dot) * scale
        g_b = diff_b * (scale * beta_f)
        dz_b = jnp.matmul(g_b, m_f, precision=jax.lax.Precision.HIGHEST)
        dm_acc = dm_acc + jnp.matmul(
            g_b.T, z_b, precision=jax.lax.Precision.HIGHEST
        )
        return (dm_acc, dbeta_acc), dz_b

    init_carry = (
        jnp.zeros((V, D), dtype=jnp.float32),
        jnp.zeros((), dtype=jnp.float32),
    )
    (dm, dbeta), dz_tiles = jax.lax.scan(
        _scan_bwd,
        init_carry,
        (z_tiles, y_tiles, lse),
    )
    dz = dz_tiles.reshape(N, D).astype(z.dtype)
    dm_out = dm.astype(m_vocab.dtype)
    dbeta_out = dbeta.reshape(inv_temp.shape).astype(inv_temp.dtype)
    return dz, dm_out, dbeta_out, None


_fused_tiled_povm_ce_jax.defvjp(
    _fused_tiled_povm_ce_jax_fwd,
    _fused_tiled_povm_ce_jax_bwd,
)


def fused_tiled_povm_cross_entropy_jax(
    z: jax.Array,
    m_vocab: jax.Array,
    inv_temp: jax.Array,
    targets: jax.Array,
    *,
    tile_size: int = 1024,
) -> jax.Array:
    """Compute POVM Born cross-entropy in JAX with L2-tiled custom VJP.

    Args:
        z: Seed Hilbert states of shape ``(N, D)``.
        m_vocab: Vocabulary POVM dictionary of shape ``(V, D)``.
        inv_temp: Scalar inverse Born temperature ``beta > 0``.
        targets: Target token indices of shape ``(N,)`` in ``[0, V - 1]``.
        tile_size: Token tile size ``B_tile >= 1`` dividing ``N``.

    Returns:
        Scalar FP32 mean cross-entropy loss.
    """
    if tile_size < 1:
        raise ValueError(f"tile_size must be >= 1, got {tile_size}")
    if z.ndim != 2:
        raise ValueError(f"z must be 2D (N, D), got shape {z.shape}")
    if m_vocab.ndim != 2:
        raise ValueError(f"m_vocab must be 2D (V, D), got shape {m_vocab.shape}")
    if z.shape[0] < 1 or m_vocab.shape[0] < 1:
        raise ValueError("z and m_vocab must have non-zero leading dimensions")
    if z.shape[1] != m_vocab.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: z has D={z.shape[1]}, m_vocab has D={m_vocab.shape[1]}"
        )
    if targets.ndim != 1 or targets.shape[0] != z.shape[0]:
        raise ValueError(
            f"targets must be 1D of length N={z.shape[0]}, got shape {targets.shape}"
        )
    if z.shape[0] % tile_size != 0:
        raise ValueError(
            f"num_tokens N={z.shape[0]} must be divisible by tile_size={tile_size} for jax.lax.scan"
        )
    if inv_temp.size != 1:
        raise ValueError(
            f"inv_temp must be a scalar or single-element array, got shape {inv_temp.shape}"
        )

    return _fused_tiled_povm_ce_jax(z, m_vocab, inv_temp, targets, int(tile_size))


__all__ = [
    "estimate_povm_head_roofline",
    "estimate_povm_head_roofline_jax",
    "measure_povm_ce_roofline",
    "measure_povm_ce_roofline_jax",
    "fused_tiled_povm_cross_entropy_jax",
]
