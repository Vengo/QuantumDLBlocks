"""JAX/XLA counterparts for Overcomplete Two-Beam Frame Penalties and Coherence Metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .overcomplete_frame import frame_coherence_metrics_jax

__all__ = [
    "two_beam_frame_penalty_jax",
    "overcomplete_two_beam_metrics_jax",
]


def _l2_normalize_jax(x: jax.Array, axis: int = -1, eps: float = 1e-12) -> jax.Array:
    return x * jax.lax.rsqrt(jnp.sum(x * x, axis=axis, keepdims=True) + float(eps))


def _unit_tight_frame_penalty_jax(w: jax.Array, eps: float = 1e-12) -> jax.Array:
    if w.ndim != 2:
        raise ValueError(f"Expected a 2D array of shape (M, d), got shape {tuple(w.shape)}")
    wf = w.astype(jnp.float32)
    m, d = wf.shape
    u = _l2_normalize_jax(wf, axis=-1, eps=eps)
    rho = float(m) / float(d)
    eye_d = jnp.eye(d, dtype=jnp.float32)
    diff = (u.T @ u) - rho * eye_d
    return jnp.sum(diff * diff) / ((rho**2) * float(d))


def two_beam_frame_penalty_jax(
    W_beam1: jax.Array,
    W_beam2: jax.Array,
    eps: float = 1e-12,
) -> jax.Array:
    """Compute unified two-beam tight-frame and paired-row orthogonality penalty in JAX."""
    if W_beam1.ndim != 2 or W_beam2.ndim != 2 or W_beam1.shape != W_beam2.shape:
        raise ValueError(
            f"W_beam1 and W_beam2 must be 2D arrays of identical shape (M, d), "
            f"got {tuple(W_beam1.shape)} and {tuple(W_beam2.shape)}"
        )
    p1 = _unit_tight_frame_penalty_jax(W_beam1, eps=eps)
    p2 = _unit_tight_frame_penalty_jax(W_beam2, eps=eps)
    u1 = _l2_normalize_jax(W_beam1.astype(jnp.float32), axis=-1, eps=eps)
    u2 = _l2_normalize_jax(W_beam2.astype(jnp.float32), axis=-1, eps=eps)
    p_pair = jnp.mean(jnp.sum(u1 * u2, axis=-1) ** 2)
    return 0.5 * (p1 + p2) + p_pair


def overcomplete_two_beam_metrics_jax(
    W_beam1: jax.Array,
    W_beam2: jax.Array,
    eps: float = 1e-12,
) -> dict[str, jax.Array]:
    """Compute Welch-bound coherence, tight-frame error, paired-beam coherence, and naive isometry error in JAX."""
    m1 = frame_coherence_metrics_jax(W_beam1, eps=eps)
    m2 = frame_coherence_metrics_jax(W_beam2, eps=eps)
    u1 = _l2_normalize_jax(W_beam1.astype(jnp.float32), axis=-1, eps=eps)
    u2 = _l2_normalize_jax(W_beam2.astype(jnp.float32), axis=-1, eps=eps)
    paired_rms = jnp.sqrt(jnp.mean(jnp.sum(u1 * u2, axis=-1) ** 2))
    return {
        "beam1_tight_frame_err": jnp.asarray(m1["tight_frame_err"], dtype=jnp.float32),
        "beam2_tight_frame_err": jnp.asarray(m2["tight_frame_err"], dtype=jnp.float32),
        "beam1_rms_coherence": jnp.asarray(m1["rms_coherence"], dtype=jnp.float32),
        "beam2_rms_coherence": jnp.asarray(m2["rms_coherence"], dtype=jnp.float32),
        "welch_bound": jnp.asarray(m1["welch_bound"], dtype=jnp.float32),
        "paired_beam_rms_coherence": jnp.asarray(paired_rms, dtype=jnp.float32),
        "naive_iso_err": jnp.asarray(
            0.5 * (m1["iso_err_vs_identity"] + m2["iso_err_vs_identity"]), dtype=jnp.float32
        ),
    }
