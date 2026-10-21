r"""JAX Counterparts for Early-Epoch Quantum Convergence & Purified Born Attention (Plan 014).

Provides JAX implementations with exact numerical parity against PyTorch (`qdlblocks.layers.early_epoch_quantum`):
1. `compute_self_repetition_metrics_jax`: Measures self-repetition probability $P(y_t = x_t | x_{\le t})$.
2. `purified_born_attention_weights_jax`: Quartic Coherence-Purified Born Attention
   $w_{ij}^{\text{purified}} = \text{ReLU}(\gamma_h (s_{ij} - \tau_h))^4 + \epsilon_{\text{base}} \text{ReLU}(s_{ij})^2$.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "compute_self_repetition_metrics_jax",
    "purified_born_attention_weights_jax",
]


def compute_self_repetition_metrics_jax(
    logits: jax.Array,
    input_ids: jax.Array,
    eps: float = 1e-8,
) -> dict[str, jax.Array]:
    r"""Compute self-repetition diagnostics $P(y_t = x_t | x_{\le t})$ in JAX.

    Args:
        logits: ``(B, T, V)`` unnormalized token logits.
        input_ids: ``(B, T)`` integer input token indices.
        eps: Numerical floor for safe division.

    Returns:
        Dictionary of scalar JAX arrays matching PyTorch `compute_self_repetition_metrics`:
        - ``mean_self_repeat_prob``: Mean probability assigned to the current input token $x_t$.
        - ``top1_self_repeat_rate``: Fraction of positions where $\arg\max_v P(v|x_{\le t}) == x_t$.
        - ``mean_entropy_nats``: Mean predictive Shannon entropy across the vocabulary.
    """
    logits_f = logits.astype(jnp.float32)
    probs = jax.nn.softmax(logits_f, axis=-1)
    idx = jnp.expand_dims(input_ids.astype(jnp.int32), axis=-1)
    self_probs = jnp.squeeze(jnp.take_along_axis(probs, idx, axis=-1), axis=-1)
    top1_ids = jnp.argmax(logits_f, axis=-1)

    mean_self_repeat_prob = jnp.mean(self_probs)
    top1_self_repeat_rate = jnp.mean((top1_ids == input_ids).astype(jnp.float32))
    entropies = -jnp.sum(probs * jnp.log(probs + eps), axis=-1)
    mean_entropy_nats = jnp.mean(entropies)

    return {
        "mean_self_repeat_prob": mean_self_repeat_prob,
        "top1_self_repeat_rate": top1_self_repeat_rate,
        "mean_entropy_nats": mean_entropy_nats,
    }


def purified_born_attention_weights_jax(
    q: jax.Array,
    k: jax.Array,
    gamma_h: float | jax.Array = 8.0,
    tau_h: float | jax.Array = 0.0625,
    eps_base: float = 0.05,
    causal: bool = True,
    eps: float = 1e-8,
    *,
    gamma: float | jax.Array | None = None,
    tau: float | jax.Array | None = None,
) -> jax.Array:
    r"""Compute Quartic Coherence-Purified Born Attention weights in JAX.

    Implements the canonical unnormalized weight formula:
    $$w_{ij}^{\text{purified}} = \text{ReLU}(\gamma_h (s_{ij} - \tau_h))^4 + \epsilon_{\text{base}} \text{ReLU}(s_{ij})^2$$
    followed by causal masking and row-wise $L_1$ normalization.

    Args:
        q: ``(B, H, T, d_h)`` unit-normalized query states.
        k: ``(B, H, S, d_h)`` unit-normalized key states.
        gamma_h: Scalar or ``(1, H, 1, 1)`` learnable sharpness gain $\gamma_h$.
        tau_h: Scalar or ``(1, H, 1, 1)`` learnable coherence threshold $\tau_h$.
        eps_base: Quadratic Born residual coefficient $\epsilon_{\text{base}}$.
        causal: Whether to apply causal lower-triangular masking.
        eps: Numerical floor for safe row normalization.

    Returns:
        Normalized attention probability matrix ``(B, H, T, S)``.
    """
    if gamma is not None:
        gamma_h = gamma
    if tau is not None:
        tau_h = tau

    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    s = jnp.matmul(q_f32, jnp.swapaxes(k_f32, -1, -2))

    gamma_t = jnp.asarray(gamma_h, dtype=jnp.float32)
    tau_t = jnp.asarray(tau_h, dtype=jnp.float32)

    coherent = jax.nn.relu(gamma_t * (s - tau_t))
    prob = jnp.square(jnp.square(coherent)) + float(eps_base) * jnp.square(jax.nn.relu(s))

    if causal:
        t_len, s_len = s.shape[-2], s.shape[-1]
        causal_mask = jnp.tril(jnp.ones((t_len, s_len), dtype=prob.dtype))
        prob = prob * causal_mask

    weights = prob / (jnp.sum(prob, axis=-1, keepdims=True) + eps)
    return weights.astype(q.dtype)
