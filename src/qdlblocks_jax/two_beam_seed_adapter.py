"""Pure JAX functional implementation of TwoBeamSeedReadoutAdapter."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from .functional import l2_normalize as l2_normalize_jax

__all__ = [
    "two_beam_seed_adapter_forward_jax",
]


def two_beam_seed_adapter_forward_jax(
    psi_final: jax.Array,
    phi_linear: jax.Array,
    U_beam1: jax.Array,
    U_beam2: jax.Array,
    W_down: jax.Array,
    seed_gate: jax.Array,
    beam_eta: jax.Array,
    eps: float = 1e-12,
) -> jax.Array:
    """Compute TwoBeamSeedReadoutAdapter forward pass in pure JAX.

    Args:
        psi_final: Normalized contextual state on S^{D-1} of shape `(..., embed_dim)`.
        phi_linear: Base linear seed state on S^{D_seed-1} of shape `(..., seed_dim)`.
        U_beam1: First overcomplete tight-frame matrix of shape `(frame_dim, embed_dim)`.
        U_beam2: Second orthogonal overcomplete tight-frame matrix of shape `(frame_dim, embed_dim)`.
        W_down: Down-projection matrix of shape `(seed_dim, frame_dim)`.
        seed_gate: Channel gate vector of shape `(seed_dim,)`.
        beam_eta: Scalar cross-beam interference mixing weight.
        eps: Numerical stability floor.

    Returns:
        Normalized seed state on S^{D_seed-1} of shape `(..., seed_dim)`.
    """
    u1 = l2_normalize_jax(U_beam1, axis=-1, eps=eps).astype(psi_final.dtype)
    u2 = l2_normalize_jax(U_beam2, axis=-1, eps=eps).astype(psi_final.dtype)
    z1 = jnp.matmul(
        psi_final,
        jnp.swapaxes(u1, -1, -2),
        precision=jax.lax.Precision.HIGHEST,
    )
    z2 = jnp.matmul(
        psi_final,
        jnp.swapaxes(u2, -1, -2),
        precision=jax.lax.Precision.HIGHEST,
    )

    embed_dim = float(psi_final.shape[-1])
    seed_dim = float(phi_linear.shape[-1])
    sqrt_d_seed = math.sqrt(seed_dim)
    scale_q = embed_dim / sqrt_d_seed
    scale_b = (embed_dim**1.5) / sqrt_d_seed

    z1_sq = z1 * z1
    q = (z1_sq - (1.0 / embed_dim)) * scale_q
    b = z1_sq * jax.nn.silu(z2) * scale_b

    combined = q + jnp.asarray(beam_eta, dtype=psi_final.dtype) * b
    delta_seed = (
        jnp.matmul(
            combined,
            jnp.swapaxes(W_down.astype(psi_final.dtype), -1, -2),
            precision=jax.lax.Precision.HIGHEST,
        )
        * seed_gate.astype(psi_final.dtype)
    )
    return l2_normalize_jax(phi_linear + delta_seed, axis=-1, eps=eps)
