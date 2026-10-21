"""JAX/XLA counterparts for Overcomplete Tight-Frame Penalties and Welch-Bound Coherence Metrics."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

__all__ = [
    "welch_bound_coherence_jax",
    "tight_frame_penalty_jax",
    "frame_coherence_metrics_jax",
]


def welch_bound_coherence_jax(num_vectors: int, dim: int) -> float:
    """Compute the theoretical Welch lower bound on root-mean-square off-diagonal coherence."""
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if num_vectors <= dim:
        return 0.0
    return math.sqrt((num_vectors - dim) / float(dim * (num_vectors - 1)))


def tight_frame_penalty_jax(w: jax.Array, target_iso_scale: float = 1.25) -> jax.Array:
    """Compute the differentiable Parseval tight-frame potential penalty in JAX."""
    if w.ndim != 2:
        raise ValueError(f"tight_frame_penalty_jax expects a 2D array, got shape {tuple(w.shape)}")
    wf = w.astype(jnp.float32)
    if wf.shape[0] < wf.shape[1]:
        wf = wf.T
    m, d = wf.shape
    gamma = (float(target_iso_scale) ** 2) * (float(m) / float(d))
    gram_dim = wf.T @ wf
    eye_d = jnp.eye(d, dtype=jnp.float32)
    diff = gram_dim - gamma * eye_d
    return jnp.sum(diff * diff) / float(d)


def frame_coherence_metrics_jax(w: jax.Array, eps: float = 1e-6) -> dict[str, float]:
    """Compute overcomplete frame coherence, tight-frame error, and naive Stiefel isometry error in JAX."""
    if w.ndim != 2:
        raise ValueError(f"frame_coherence_metrics_jax expects a 2D array, got shape {tuple(w.shape)}")
    wf = w.astype(jnp.float32)
    if wf.shape[0] < wf.shape[1]:
        wf = wf.T
    m, d = wf.shape
    w_hat = wf * jax.lax.rsqrt(jnp.sum(wf * wf, axis=-1, keepdims=True) + float(eps))
    redundancy_ratio = float(m) / float(d)

    eye_d = jnp.eye(d, dtype=jnp.float32)
    gram_raw = wf.T @ wf
    iso_err_vs_identity = float(jnp.linalg.norm(gram_raw - eye_d, ord="fro") / math.sqrt(float(d)))

    gram_unit = w_hat.T @ w_hat
    tight_frame_err = float(
        jnp.linalg.norm(gram_unit - redundancy_ratio * eye_d, ord="fro") / math.sqrt(float(d))
    )

    if m <= 1:
        rms_coherence = 0.0
        max_coherence = 0.0
    else:
        g = w_hat @ w_hat.T
        g_off = g - jnp.diag(jnp.diag(g))
        rms_coherence = float(jnp.sqrt(jnp.sum(g_off * g_off) / float(m * (m - 1))))
        max_coherence = float(jnp.max(jnp.abs(g_off)))

    return {
        "redundancy_ratio": redundancy_ratio,
        "iso_err_vs_identity": iso_err_vs_identity,
        "tight_frame_err": tight_frame_err,
        "rms_coherence": rms_coherence,
        "max_coherence": max_coherence,
        "welch_bound": welch_bound_coherence_jax(m, d),
    }
