"""JAX/XLA Speedrun Primitives: Batched 3D Newton-Schulz, Half-Truncated RoPE, and Document IDs."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def zeropower_via_newtonschulz5_batched_jax(
    G: jax.Array,
    steps: int = 5,
    eps: float = 1e-7,
) -> jax.Array:
    """Compute the nearest semi-orthogonal matrices for a 3D batch of identically shaped 2D matrices in JAX.

    Args:
        G: Input 3D array of shape `(K, M, N)`.
        steps: Number of quintic Newton-Schulz iterations (default: 5).
        eps: Numerical floor for Frobenius norm scaling.

    Returns:
        Semi-orthogonalized 3D array of shape `(K, M, N)` in `G.dtype`.
    """
    if G.ndim != 3:
        raise ValueError(f"Expected 3D tensor (K, M, N) for batched Newton-Schulz, got {G.ndim}D")

    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.astype(jnp.bfloat16) if G.dtype == jnp.bfloat16 else G.astype(jnp.float32)
    prec = jax.lax.Precision.DEFAULT if G.dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST

    transposed = G.shape[1] > G.shape[2]
    if transposed:
        X = jnp.swapaxes(X, -2, -1)

    norms = jnp.linalg.norm(X.astype(jnp.float32), axis=(-2, -1), keepdims=True)
    X = X / (norms.astype(X.dtype) + eps)

    for _ in range(steps):
        A = jnp.matmul(X, jnp.swapaxes(X, -2, -1), precision=prec)
        B = b * A + c * jnp.matmul(A, A, precision=prec)
        X = a * X + jnp.matmul(B, X, precision=prec)

    if transposed:
        X = jnp.swapaxes(X, -2, -1)

    return X.astype(G.dtype)


def half_truncated_rope_cos_sin_jax(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
) -> tuple[jax.Array, jax.Array]:
    """Precompute half-truncated Unitary RoPE `(cos, sin)` tables in JAX.

    Only the first `rot_d = (head_dim // 2) // 2` frequencies rotate with position;
    the upper half `freqs[rot_d:]` is zeroed (`cos = 1`, `sin = 0`), creating a
    dedicated position-invariant subspace inside every attention head.

    Args:
        seq_len: Sequence length `T`.
        head_dim: Head dimension `d_head` (must be even).
        base: Geometric progression base (default: `10000.0`).

    Returns:
        Tuple `(cos, sin)` each of shape `(1, 1, seq_len, head_dim // 2)`.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for UnitaryRoPE, got {head_dim}")
    half_d = head_dim // 2
    rot_d = max(1, half_d // 2)
    freqs_active = 1.0 / (base ** (jnp.arange(0, rot_d * 2, 2, dtype=jnp.float32) / float(rot_d * 2)))
    freqs = jnp.zeros((half_d,), dtype=jnp.float32)
    freqs = freqs.at[:rot_d].set(freqs_active)
    t = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(t, freqs)
    cos = jnp.cos(angles)[None, None, :, :]
    sin = jnp.sin(angles)[None, None, :, :]
    return cos, sin


def compute_document_ids_jax(
    idx: jax.Array,
    eot_token_id: int = 50256,
) -> jax.Array:
    """Compute monotonic per-token document IDs from EOT boundaries in JAX.

    An EOT token at position `t` belongs to the current document (ending it),
    and increments the document ID starting at position `t + 1`.

    Args:
        idx: Token ID array of shape `(B, T)`.
        eot_token_id: End-of-text token ID (default: `50256`).

    Returns:
        Integer array `doc_ids` of shape `(B, T)` with `jnp.int32` dtype.
    """
    is_eot = (idx == eot_token_id).astype(jnp.int32)
    shifted = jnp.pad(is_eot[:, :-1], ((0, 0), (1, 0)))
    return jnp.cumsum(shifted, axis=-1, dtype=jnp.int32)
