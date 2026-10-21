"""JAX/Equinox/Haliax Contextual Bloch-Purity POVM Sharpness & Logit-Norm Diagnostics (Plan 013)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def compute_logit_norm_variation_metrics_jax(
    logits: jnp.ndarray,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Compute per-token logit L2-norm statistics, coefficient of variation, and output entropy in JAX."""
    if logits.ndim < 1 or logits.shape[-1] < 1 or logits.size == 0:
        raise ValueError(f"Expected non-empty logits array (..., V), got shape {tuple(logits.shape)}.")
    v_dim = logits.shape[-1]
    flat = jnp.asarray(logits, dtype=jnp.float32).reshape(-1, v_dim)
    norms = jnp.sqrt(jnp.sum(jnp.square(flat), axis=-1))
    n_mean = jnp.mean(norms)
    n_std = jnp.std(norms)
    n_cv = n_std / (n_mean + eps)

    probs = jax.nn.softmax(flat, axis=-1)
    entropies = -jnp.sum(probs * jnp.log(probs + eps), axis=-1)
    max_probs = jnp.max(probs, axis=-1)

    return {
        "logit_norm_mean": float(n_mean),
        "logit_norm_std": float(n_std),
        "logit_norm_cv": float(n_cv),
        "mean_entropy_nats": float(jnp.mean(entropies)),
        "mean_max_prob": float(jnp.mean(max_probs)),
    }


def contextual_bloch_purity_beta_jax(
    psi: jnp.ndarray,
    w1: jnp.ndarray,
    b1: jnp.ndarray,
    w2: jnp.ndarray,
    b2: jnp.ndarray,
    base_inv_temp: float | jnp.ndarray = 28.57,
    r_min: float = 0.25,
    r_max: float = 4.0,
    init_bias: float = 1.0,
    max_beta: float = 160.0,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Compute per-token Bloch-purity inverse temperature `beta_{b,t}` of shape `(..., 1)` in JAX."""
    psi_f = jnp.asarray(psi, dtype=jnp.float32)
    w1_f = jnp.asarray(w1, dtype=jnp.float32)
    b1_f = jnp.asarray(b1, dtype=jnp.float32)
    w2_f = jnp.asarray(w2, dtype=jnp.float32)
    b2_f = jnp.asarray(b2, dtype=jnp.float32)

    psi_norm = psi_f / jnp.sqrt(jnp.sum(jnp.square(psi_f), axis=-1, keepdims=True) + eps)
    h = jax.nn.silu(jnp.matmul(psi_norm, jnp.swapaxes(w1_f, -1, -2)) + b1_f)
    u = jnp.matmul(h, jnp.swapaxes(w2_f, -1, -2)) + b2_f
    s0 = jax.nn.softplus(jnp.asarray(init_bias, dtype=jnp.float32))
    r = jnp.clip(jax.nn.softplus(u) / s0, r_min, r_max)
    base_t = jnp.asarray(base_inv_temp, dtype=jnp.float32)
    return jnp.clip(base_t * r, 1.0, max_beta)
