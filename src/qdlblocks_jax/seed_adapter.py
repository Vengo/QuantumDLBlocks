"""Pure JAX functional implementation of OvercompleteSeedReadoutAdapter."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from .functional import l2_normalize as l2_normalize_jax

__all__ = [
    "overcomplete_seed_adapter_forward_jax",
]


def overcomplete_seed_adapter_forward_jax(
    psi_final: jax.Array,
    phi_linear: jax.Array,
    U_frame: jax.Array,
    W_down: jax.Array,
    seed_gate: jax.Array,
    eps: float = 1e-12,
) -> jax.Array:
    """Compute OvercompleteSeedReadoutAdapter forward pass in pure JAX.

    Args:
        psi_final: Normalized contextual state on S^{D-1} of shape `(..., embed_dim)`.
        phi_linear: Base linear seed state on S^{D_seed-1} of shape `(..., seed_dim)`.
        U_frame: Overcomplete frame matrix of shape `(frame_dim, embed_dim)`.
        W_down: Down-projection matrix of shape `(seed_dim, frame_dim)`.
        seed_gate: Channel gate vector of shape `(seed_dim,)`.
        eps: Numerical stability floor.

    Returns:
        Normalized seed state on S^{D_seed-1} of shape `(..., seed_dim)`.
    """
    u_unit = l2_normalize_jax(U_frame, axis=-1, eps=eps).astype(psi_final.dtype)
    z = jnp.matmul(
        psi_final,
        jnp.swapaxes(u_unit, -1, -2),
        precision=jax.lax.Precision.HIGHEST,
    )
    embed_dim = float(psi_final.shape[-1])
    seed_dim = float(phi_linear.shape[-1])
    scale = embed_dim / math.sqrt(seed_dim)
    q = (z * z - (1.0 / embed_dim)) * scale
    delta_seed = (
        jnp.matmul(
            q,
            jnp.swapaxes(W_down.astype(psi_final.dtype), -1, -2),
            precision=jax.lax.Precision.HIGHEST,
        )
        * seed_gate.astype(psi_final.dtype)
    )
    return l2_normalize_jax(phi_linear + delta_seed, axis=-1, eps=eps)
