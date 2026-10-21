"""Pure JAX Multi-Beam Kraus Subspace POVM Readout and Logit Matrix Effective Rank."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from .functional import l2_normalize

__all__ = [
    "compute_logit_matrix_effective_rank_jax",
    "init_multibeam_kraus_params",
    "multibeam_kraus_forward_jax",
]


def compute_logit_matrix_effective_rank_jax(
    logits: jax.Array,
    eps: float = 1e-8,
) -> jax.Array:
    """Compute the Logit Matrix Effective Singular Rank R_sigma(L) in [1, min(N, V)] in JAX.

    Args:
        logits: Logit array of shape `(..., V)` with `logits.ndim >= 2`,
            where `N = logits.size // V >= 2` and `V >= 2`.
        eps: Numerical stability floor.

    Returns:
        Scalar JAX array containing the effective singular rank R_sigma(L).
    """
    if logits.ndim < 2:
        raise ValueError(
            f"compute_logit_matrix_effective_rank_jax requires logits.ndim >= 2, got {logits.ndim}."
        )
    v = logits.shape[-1]
    n = logits.size // max(v, 1)
    if n < 2 or v < 2:
        raise ValueError(
            f"compute_logit_matrix_effective_rank_jax requires N >= 2 and V >= 2, got N={n}, V={v}."
        )
    x = logits.reshape(n, v).astype(jnp.float32)
    x_c = x - jnp.mean(x, axis=0, keepdims=True)
    s = jnp.linalg.svd(x_c, compute_uv=False)
    s_sq = s**2
    p = s_sq / (jnp.sum(s_sq) + eps)
    ent = -jnp.sum(p * jnp.log(jnp.maximum(p, eps)))
    return jnp.exp(ent)


def _orthogonal_matrix_jax(key: jax.Array, rows: int, cols: int, gain: float = 1.0) -> jax.Array:
    """Generate a semi-orthogonal matrix of shape `(rows, cols)` scaled by `gain`."""
    transposed = rows < cols
    r_dim, c_dim = (cols, rows) if transposed else (rows, cols)
    raw = jax.random.normal(key, (r_dim, c_dim), dtype=jnp.float32)
    q, r = jnp.linalg.qr(raw)
    diag_sign = jnp.sign(jnp.diag(r))
    diag_sign = jnp.where(diag_sign == 0.0, 1.0, diag_sign)
    q = q * diag_sign[None, :]
    if transposed:
        q = q.T
    return q * float(gain)


def init_multibeam_kraus_params(
    key: jax.Array,
    vocab_size: int,
    embed_dim: int,
    seed_dim: int,
    num_beams: int = 4,
    beam_adapter_rank: int = 16,
    adapter_init_scale: float = 1.0,
    povm_temperature: float = 0.035,
) -> dict[str, jax.Array]:
    """Initialize parameters for pure-JAX Multi-Beam Kraus Readout.

    Args:
        key: JAX PRNGKey.
        vocab_size: Vocabulary size `V >= 2`.
        embed_dim: Backbone Hilbert space dimension `d >= seed_dim`.
        seed_dim: Seed subspace dimension `D_seed >= 1`.
        num_beams: Number of orthogonal readout beams `K >= 1` (`K * D_seed <= d`).
        beam_adapter_rank: Low-rank Kraus perturbation rank `r_b >= 1`.
        adapter_init_scale: Initial scale for vocabulary adapter factors `> 0.0`.
        povm_temperature: Initial POVM measurement temperature.

    Returns:
        Dictionary of JAX parameter arrays matching `MultiBeamKrausReadout`.
    """
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2, got {vocab_size}.")
    if num_beams < 1:
        raise ValueError(f"num_beams must be >= 1, got {num_beams}.")
    if seed_dim < 1:
        raise ValueError(f"seed_dim must be >= 1, got {seed_dim}.")
    if embed_dim < seed_dim:
        raise ValueError(f"embed_dim ({embed_dim}) must be >= seed_dim ({seed_dim}).")
    if num_beams * seed_dim > embed_dim:
        raise ValueError(
            f"num_beams * seed_dim ({num_beams * seed_dim}) must be <= embed_dim ({embed_dim})."
        )
    if beam_adapter_rank < 1:
        raise ValueError(f"beam_adapter_rank must be >= 1, got {beam_adapter_rank}.")
    if adapter_init_scale <= 0.0:
        raise ValueError(f"adapter_init_scale must be > 0.0, got {adapter_init_scale}.")

    k_rot, k_a, k_b = jax.random.split(key, 3)
    readout_rot = _orthogonal_matrix_jax(k_rot, num_beams * seed_dim, embed_dim, gain=1.0)

    init_beta = 1.0 / max(povm_temperature, 1e-4)
    params: dict[str, jax.Array] = {
        "readout_rot": readout_rot,
        "inv_temp": jnp.array(float(init_beta), dtype=jnp.float32),
    }
    if num_beams > 1:
        k_aux = num_beams - 1
        gain_a = 2.0 * adapter_init_scale * math.sqrt(float(vocab_size) / float(beam_adapter_rank))
        if vocab_size >= k_aux * beam_adapter_rank:
            a_stacked = _orthogonal_matrix_jax(
                k_a, vocab_size, k_aux * beam_adapter_rank, gain=gain_a
            )
            beam_vocab_a = jnp.transpose(
                a_stacked.reshape(vocab_size, k_aux, beam_adapter_rank), (1, 0, 2)
            )
        else:
            keys_a = jax.random.split(k_a, k_aux)
            beam_vocab_a = jnp.stack(
                [
                    _orthogonal_matrix_jax(keys_a[i], vocab_size, beam_adapter_rank, gain=gain_a)
                    for i in range(k_aux)
                ],
                axis=0,
            )

        keys_b = jax.random.split(k_b, k_aux)
        beam_seed_b = jnp.stack(
            [
                _orthogonal_matrix_jax(keys_b[i], seed_dim, beam_adapter_rank, gain=1.0)
                for i in range(k_aux)
            ],
            axis=0,
        )
        params["beam_vocab_a"] = beam_vocab_a
        params["beam_seed_b"] = beam_seed_b
        params["beam_weights"] = jnp.ones((num_beams,), dtype=jnp.float32) / math.sqrt(
            float(num_beams)
        )
    return params


def multibeam_kraus_forward_jax(
    params: dict[str, jax.Array],
    psi: jax.Array,
    base_wte: jax.Array,
    eps: float = 1e-12,
) -> jax.Array:
    """Compute Multi-Beam Kraus POVM readout logits in pure JAX.

    Args:
        params: Parameter dictionary from `init_multibeam_kraus_params`.
        psi: Contextual Hilbert state array of shape `(..., embed_dim)`.
        base_wte: Base seed vocabulary dictionary of shape `(vocab_size, seed_dim)`.
        eps: Numerical stability floor.

    Returns:
        Logit array of shape `(..., vocab_size)`.
    """
    rot = params["readout_rot"].astype(psi.dtype)
    d_seed = base_wte.shape[-1]
    k_beams = rot.shape[0] // d_seed
    beams = jnp.matmul(psi, rot.T, precision=jax.lax.Precision.HIGHEST).reshape(
        *psi.shape[:-1], k_beams, d_seed
    )
    beams = l2_normalize(beams, axis=-1, eps=eps)
    m_0 = l2_normalize(base_wte, axis=-1, eps=eps).astype(psi.dtype)
    beta = jnp.clip(params["inv_temp"], 1.0, 60.0).astype(psi.dtype)
    if k_beams == 1:
        return (
            jnp.matmul(
                jnp.squeeze(beams, axis=-2),
                m_0.T,
                precision=jax.lax.Precision.HIGHEST,
            )
            * beta
        )
    w = params["beam_weights"].astype(psi.dtype)
    w = w / jnp.maximum(jnp.linalg.norm(w), eps)
    logits = w[0] * jnp.matmul(
        beams[..., 0, :],
        m_0.T,
        precision=jax.lax.Precision.HIGHEST,
    )
    for k in range(1, k_beams):
        a_k = params["beam_vocab_a"][k - 1].astype(psi.dtype)
        b_k = params["beam_seed_b"][k - 1].astype(psi.dtype)
        delta_m = jnp.matmul(
            a_k,
            jnp.swapaxes(b_k, -2, -1),
            precision=jax.lax.Precision.HIGHEST,
        )
        m_k = l2_normalize(m_0 + delta_m, axis=-1, eps=eps)
        logits = logits + w[k] * jnp.matmul(
            beams[..., k, :],
            m_k.T,
            precision=jax.lax.Precision.HIGHEST,
        )
    return logits * beta
