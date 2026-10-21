"""Pure JAX counterparts for Two-Beam Bilinear Interference and Seed-Space Scaling Diagnostics."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from .functional import (
    effective_quantum_rank as effective_quantum_rank_jax,
    l2_normalize as l2_normalize_jax,
    purity_participation_ratio as purity_participation_ratio_jax,
)

__all__ = [
    "single_beam_signed_born_features_jax",
    "two_beam_bilinear_born_features_jax",
    "compute_interference_feature_rank_jax",
    "compute_cross_beam_coherence_jax",
    "compute_paired_beam_orthogonality_stats_jax",
    "compute_coupled_seed_pullback_rank_jax",
    "compute_effective_seed_utilization_ratio_jax",
]


def single_beam_signed_born_features_jax(
    psi: jax.Array,
    W_up: jax.Array,
    eps: float = 1e-12,
) -> jax.Array:
    """Compute degree-2 signed Born features for a single overcomplete frame in pure JAX."""
    w_unit = l2_normalize_jax(W_up, axis=-1, eps=eps).astype(psi.dtype)
    z = jnp.matmul(
        psi,
        jnp.swapaxes(w_unit, -1, -2),
        precision=jax.lax.Precision.HIGHEST,
    )
    return (z * z) * jnp.sign(z)


def two_beam_bilinear_born_features_jax(
    psi: jax.Array,
    W_beam1: jax.Array,
    W_beam2: jax.Array,
    eps: float = 1e-12,
) -> jax.Array:
    """Compute degree-3 cross-subspace bilinear Born features for two paired beams in pure JAX."""
    w1_unit = l2_normalize_jax(W_beam1, axis=-1, eps=eps).astype(psi.dtype)
    w2_unit = l2_normalize_jax(W_beam2, axis=-1, eps=eps).astype(psi.dtype)
    z1 = jnp.matmul(
        psi,
        jnp.swapaxes(w1_unit, -1, -2),
        precision=jax.lax.Precision.HIGHEST,
    )
    z2 = jnp.matmul(
        psi,
        jnp.swapaxes(w2_unit, -1, -2),
        precision=jax.lax.Precision.HIGHEST,
    )
    return (z1 * z1) * jax.nn.silu(z2)


def compute_interference_feature_rank_jax(
    features: jax.Array,
    eps: float = 1e-12,
) -> dict[str, jax.Array]:
    """Compute spectral Shannon effective rank, participation ratio, and trace in pure JAX."""
    if features.ndim < 1 or features.shape[-1] < 1 or features.size == 0:
        raise ValueError(
            f"compute_interference_feature_rank_jax expects non-empty array (..., M), got {features.shape}"
        )
    flat = features.reshape(-1, features.shape[-1]).astype(jnp.float32)
    n = int(flat.shape[0])
    if n <= 1:
        return {
            "feature_eff_rank": jnp.asarray(1.0, dtype=jnp.float32),
            "feature_participation_ratio": jnp.asarray(1.0, dtype=jnp.float32),
            "feature_trace": jnp.asarray(0.0, dtype=jnp.float32),
        }
    f_c = flat - jnp.mean(flat, axis=0, keepdims=True)
    c_mat = jnp.matmul(f_c.T, f_c, precision=jax.lax.Precision.HIGHEST) / float(n - 1)
    eigvals = jnp.maximum(jnp.linalg.eigvalsh(c_mat), 0.0)
    trace = jnp.sum(eigvals)
    p = eigvals / (trace + float(eps))
    eff_rank = jnp.exp(-jnp.sum(p * jnp.log(p + float(eps))))
    pr = 1.0 / (jnp.sum(p * p) + float(eps))
    return {
        "feature_eff_rank": eff_rank,
        "feature_participation_ratio": pr,
        "feature_trace": trace,
    }


def compute_cross_beam_coherence_jax(
    W_beam1: jax.Array,
    W_beam2: jax.Array,
    eps: float = 1e-12,
) -> dict[str, jax.Array]:
    """Compute root-mean-square and maximum paired-row coherence between two beam matrices in pure JAX."""
    if W_beam1.ndim != 2 or W_beam2.ndim != 2 or W_beam1.shape != W_beam2.shape:
        raise ValueError(
            f"compute_cross_beam_coherence_jax expects 2D arrays of identical shape (M, D), "
            f"got {W_beam1.shape} and {W_beam2.shape}"
        )
    w1_u = l2_normalize_jax(W_beam1.astype(jnp.float32), axis=-1, eps=eps)
    w2_u = l2_normalize_jax(W_beam2.astype(jnp.float32), axis=-1, eps=eps)
    row_cos = jnp.sum(w1_u * w2_u, axis=-1)
    return {
        "rms_paired_beam_coherence": jnp.sqrt(jnp.mean(row_cos * row_cos)),
        "max_paired_beam_coherence": jnp.max(jnp.abs(row_cos)),
    }


def compute_paired_beam_orthogonality_stats_jax(
    W_beam1: jax.Array,
    W_beam2: jax.Array,
    eps: float = 1e-12,
) -> dict[str, jax.Array]:
    """Alias for `compute_cross_beam_coherence_jax` returning paired-beam coherence statistics."""
    return compute_cross_beam_coherence_jax(W_beam1=W_beam1, W_beam2=W_beam2, eps=eps)


def compute_coupled_seed_pullback_rank_jax(
    fn: Callable[[jax.Array], jax.Array],
    psi: jax.Array,
    eps: float = 1e-12,
) -> dict[str, jax.Array]:
    """Compute the input-sensitivity pullback metric effective rank R_eff(G) in pure JAX."""
    if psi.ndim < 2 or psi.shape[-1] < 1 or psi.size == 0:
        raise ValueError(
            f"compute_coupled_seed_pullback_rank_jax expects psi of shape (N, D), got {psi.shape}"
        )
    d = int(psi.shape[-1])
    n = int(psi.size // d)
    x = psi.reshape(n, d).astype(jnp.float32)
    jac_batch = jax.vmap(jax.jacrev(fn))(x)  # (N, D_seed, D)
    g_mat = jnp.einsum("nkd,nke->de", jac_batch, jac_batch, precision=jax.lax.Precision.HIGHEST) / float(n)
    evals = jnp.maximum(jnp.linalg.eigvalsh(g_mat), 0.0)
    trace_g = jnp.sum(evals)
    p = evals / (trace_g + float(eps))
    eff_rank = jnp.exp(-jnp.sum(p * jnp.log(p + float(eps))))
    pr = 1.0 / jnp.maximum(jnp.sum(p * p), float(eps))
    return {
        "pullback_eff_rank": eff_rank,
        "pullback_participation_ratio": pr,
        "trace_G": trace_g,
    }


def compute_effective_seed_utilization_ratio_jax(
    phi_seed: jax.Array,
    eps: float = 1e-8,
) -> dict[str, jax.Array]:
    """Compute effective quantum rank and normalized dimension utilization ratio of seed states in pure JAX."""
    if phi_seed.ndim < 1 or phi_seed.shape[-1] < 1 or phi_seed.size == 0:
        raise ValueError(
            f"compute_effective_seed_utilization_ratio_jax expects non-empty (..., D_seed), got {phi_seed.shape}"
        )
    d_seed = float(phi_seed.shape[-1])
    r_eff = effective_quantum_rank_jax(phi_seed, eps=eps)
    pr = purity_participation_ratio_jax(phi_seed, eps=eps)
    return {
        "seed_eff_rank": r_eff,
        "seed_participation_ratio": pr,
        "seed_dim": jnp.asarray(d_seed, dtype=jnp.float32),
        "seed_utilization_ratio": r_eff / d_seed,
        "seed_pr_utilization_ratio": pr / d_seed,
    }
