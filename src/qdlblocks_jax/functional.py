"""Pure JAX functional quantum and Stiefel manifold operations for qdlblocks_jax."""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "l2_normalize",
    "riemannian_center",
    "precompute_rope_cos_sin",
    "apply_unitary_rope",
    "zeropower_via_newtonschulz5",
    "stiefel_isometry_error",
    "ensure_complex",
    "normalize_trace",
    "state_to_density",
    "power_purify",
    "kraus_transform",
    "povm_measure",
    "von_neumann_entropy",
    "partial_trace_qubits",
    "bipartite_entanglement_entropy",
    "empirical_state_density",
    "effective_quantum_rank",
    "purity_participation_ratio",
]


def l2_normalize(x: jax.Array, axis: int = -1, eps: float = 1e-12) -> jax.Array:
    """Project real or complex vectors onto the unit hypersphere S^{d-1}."""
    sq_norm = jnp.sum(jnp.real(x * jnp.conj(x)), axis=axis, keepdims=True)
    norm = jnp.sqrt(jnp.maximum(sq_norm, eps * eps))
    return x / norm


def riemannian_center(x: jax.Array, axis: int = -1, eps: float = 1e-12) -> jax.Array:
    """Zero-mean center a vector and re-project onto S^{d-1}."""
    centered = x - jnp.mean(x, axis=axis, keepdims=True)
    return l2_normalize(centered, axis=axis, eps=eps)


def precompute_rope_cos_sin(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[jax.Array, jax.Array]:
    """Precompute unitary Rotary Position Embedding (RoPE) cosine and sine tables."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}.")
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    return jnp.cos(freqs).astype(dtype), jnp.sin(freqs).astype(dtype)


def apply_unitary_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Apply SO(2)^{D/2} unitary rotary transformation to (..., T, D)."""
    seq_len = x.shape[-2]
    d = x.shape[-1]
    if d % 2 != 0:
        raise ValueError(f"Last dimension of x must be even for RoPE, got {d}.")
    d_half = d // 2
    if cos.ndim == 2:
        cos_s = cos[:seq_len].reshape((1,) * (x.ndim - 2) + (seq_len, d_half))
        sin_s = sin[:seq_len].reshape((1,) * (x.ndim - 2) + (seq_len, d_half))
    else:
        cos_s = cos[..., :seq_len, :]
        sin_s = sin[..., :seq_len, :]
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]
    out = jnp.concatenate([x1 * cos_s - x2 * sin_s, x1 * sin_s + x2 * cos_s], axis=-1)
    return out.astype(x.dtype)


def zeropower_via_newtonschulz5(
    G: jax.Array,
    steps: int = 5,
    eps: float = 1e-7,
) -> jax.Array:
    """Compute the orthogonal polar factor via 5-step quintic Newton-Schulz iteration."""
    if G.ndim < 2:
        raise ValueError(f"Expected G to have at least 2 dimensions, got {G.ndim}.")
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.astype(jnp.float32)
    transposed = G.shape[-2] > G.shape[-1]
    if transposed:
        X = jnp.swapaxes(X, -1, -2)
    frob = jnp.sqrt(jnp.sum(X * X, axis=(-2, -1), keepdims=True))
    X = X / (frob + eps)
    for _ in range(steps):
        A = jnp.matmul(X, jnp.swapaxes(X, -1, -2), precision=jax.lax.Precision.HIGHEST)
        B = b * A + c * jnp.matmul(A, A, precision=jax.lax.Precision.HIGHEST)
        X = a * X + jnp.matmul(B, X, precision=jax.lax.Precision.HIGHEST)
    if transposed:
        X = jnp.swapaxes(X, -1, -2)
    return X.astype(G.dtype)


def _iso_err_2d(mat_2d: jax.Array) -> jax.Array:
    m, n = mat_2d.shape[0], mat_2d.shape[1]
    mat_f = mat_2d.astype(jnp.float32)
    if m >= n:
        diff = (
            jnp.matmul(mat_f.T, mat_f, precision=jax.lax.Precision.HIGHEST)
            - jnp.eye(n, dtype=jnp.float32)
        )
        return jnp.linalg.norm(diff) / jnp.sqrt(float(n))
    diff = (
        jnp.matmul(mat_f, mat_f.T, precision=jax.lax.Precision.HIGHEST)
        - jnp.eye(m, dtype=jnp.float32)
    )
    return jnp.linalg.norm(diff) / jnp.sqrt(float(m))


def stiefel_isometry_error(w: jax.Array) -> jax.Array:
    """Measure normalized departure from the Stiefel manifold for 2D or 3D tensors."""
    if w.ndim == 2:
        return _iso_err_2d(w)
    if w.ndim == 3:
        if w.shape[1] * w.shape[2] == w.shape[0]:
            return _iso_err_2d(w.reshape(w.shape[0], -1))
        if w.shape[0] * w.shape[1] == w.shape[2]:
            return _iso_err_2d(w.reshape(-1, w.shape[2]))
        return jnp.mean(jax.vmap(_iso_err_2d)(w))
    raise ValueError(f"stiefel_isometry_error supports 2D or 3D tensors, got shape {w.shape}.")


def ensure_complex(x: jax.Array) -> jax.Array:
    """Cast real arrays to complex64 while preserving existing complex dtypes."""
    if jnp.iscomplexobj(x):
        return x
    return x.astype(jnp.complex64)


def normalize_trace(rho: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Hermitize and normalize a batch of density matrices to unit trace."""
    rho_c = ensure_complex(rho)
    rho_h = 0.5 * (rho_c + jnp.conj(jnp.swapaxes(rho_c, -1, -2)))
    tr = jnp.real(jnp.trace(rho_h, axis1=-2, axis2=-1))[..., None, None]
    return rho_h / (tr + eps)


def state_to_density(psi: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Form pure-state density matrix |psi><psi| from state vectors psi."""
    psi_norm = l2_normalize(ensure_complex(psi), axis=-1, eps=eps)
    rho = psi_norm[..., :, None] * jnp.conj(psi_norm[..., None, :])
    return normalize_trace(rho, eps=eps)


def power_purify(rho: jax.Array, power: int = 2, eps: float = 1e-8) -> jax.Array:
    """Apply polynomial density matrix purification rho^p / Tr(rho^p)."""
    if power < 1:
        raise ValueError(f"power must be >= 1, got {power}.")
    if power == 1:
        return normalize_trace(rho, eps=eps)
    rho_c = ensure_complex(rho)
    rho_h = 0.5 * (rho_c + jnp.conj(jnp.swapaxes(rho_c, -1, -2)))
    rho_p = rho_h
    for _ in range(power - 1):
        rho_p = jnp.matmul(rho_p, rho_h, precision=jax.lax.Precision.HIGHEST)
    return normalize_trace(rho_p, eps=eps)


def kraus_transform(
    rho: jax.Array,
    kraus_ops: jax.Array,
    eps: float = 1e-8,
) -> jax.Array:
    """Apply a completely positive Kraus superoperator sum_k K_k rho K_k^dagger."""
    rho_c = ensure_complex(rho)
    k_c = ensure_complex(kraus_ops)
    evolved = jnp.einsum(
        "koi,...ij,kqj->...oq",
        k_c,
        rho_c,
        jnp.conj(k_c),
        precision=jax.lax.Precision.HIGHEST,
    )
    return normalize_trace(evolved, eps=eps)


def povm_measure(rho: jax.Array, effects_factors: jax.Array) -> jax.Array:
    """Compute Born rule measurement outcomes p_c = Tr(rho M_c^dagger M_c)."""
    rho_c = ensure_complex(rho)
    m_c = ensure_complex(effects_factors)
    effects = jnp.einsum(
        "cik,cjk->cij",
        m_c,
        jnp.conj(m_c),
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.real(
        jnp.einsum(
            "...ij,cji->...c",
            rho_c,
            effects,
            precision=jax.lax.Precision.HIGHEST,
        )
    )


def von_neumann_entropy(rho: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Compute von Neumann entropy S(rho) = -Tr(rho log rho) in nats."""
    rho_c = ensure_complex(rho)
    rho_h = 0.5 * (rho_c + jnp.conj(jnp.swapaxes(rho_c, -1, -2)))
    eigvals = jnp.linalg.eigvalsh(rho_h)
    p = jnp.maximum(jnp.real(eigvals), 0.0)
    p = p / (jnp.sum(p, axis=-1, keepdims=True) + eps)
    return -jnp.sum(p * jnp.log(jnp.maximum(p, eps)), axis=-1)


def partial_trace_qubits(
    psi_or_rho: jax.Array,
    num_qubits: int,
    keep_qubits: tuple[int, ...],
    eps: float = 1e-8,
) -> jax.Array:
    """Compute reduced density matrix rho_A = Tr_B(rho) over a subset of qubits.

    When `psi_or_rho` is a pure state vector `(..., 2^n)`, uses the exact Schmidt
    reshape `M in C^(..., 2^k, 2^(n-k))` to compute `rho_A = M @ M^dagger` in
    `O(2^(n+k))` time and `O(4^k)` memory without materializing the `2^n x 2^n`
    full density matrix.
    """
    if num_qubits < 1:
        raise ValueError(f"num_qubits must be >= 1, got {num_qubits}.")
    keep = tuple(keep_qubits)
    if len(keep) == 0 or len(keep) > num_qubits:
        raise ValueError(f"keep_qubits must have 1..{num_qubits} indices, got {keep}.")
    if len(set(keep)) != len(keep):
        raise ValueError(f"keep_qubits must not contain duplicates, got {keep}.")
    if any(q < 0 or q >= num_qubits for q in keep):
        raise ValueError(f"keep_qubits indices must be in [0, {num_qubits - 1}], got {keep}.")

    dim = 2**num_qubits
    is_density = (
        psi_or_rho.ndim >= 2
        and psi_or_rho.shape[-1] == dim
        and psi_or_rho.shape[-2] == dim
    )
    if not is_density and psi_or_rho.shape[-1] == dim:
        psi_norm = l2_normalize(ensure_complex(psi_or_rho), axis=-1, eps=eps)
        batch_shape = psi_norm.shape[:-1]
        n_batch = len(batch_shape)
        k = len(keep)
        trace_out = tuple(q for q in range(num_qubits) if q not in keep)
        psi_q = psi_norm.reshape(batch_shape + (2,) * num_qubits)
        perm = (
            list(range(n_batch))
            + [n_batch + q for q in keep]
            + [n_batch + q for q in trace_out]
        )
        psi_perm = jnp.transpose(psi_q, perm)
        schmidt_mat = psi_perm.reshape(batch_shape + (2**k, 2 ** (num_qubits - k)))
        rho_red = jnp.matmul(
            schmidt_mat,
            jnp.conj(jnp.swapaxes(schmidt_mat, -1, -2)),
            precision=jax.lax.Precision.HIGHEST,
        )
        return normalize_trace(rho_red, eps=eps)

    if is_density:
        rho = normalize_trace(ensure_complex(psi_or_rho), eps=eps)
    else:
        raise ValueError(
            f"Expected state or density with dimension {dim} (2**{num_qubits}), got shape {psi_or_rho.shape}."
        )

    batch_shape = rho.shape[:-2]
    n_batch = len(batch_shape)
    k = len(keep)
    trace_out = tuple(q for q in range(num_qubits) if q not in keep)
    if len(trace_out) == 0 and keep == tuple(range(num_qubits)):
        return rho

    rho_q = rho.reshape(batch_shape + (2,) * num_qubits + (2,) * num_qubits)
    n_cur = num_qubits
    for q_tr in reversed(sorted(trace_out)):
        rho_q = jnp.trace(rho_q, axis1=n_batch + q_tr, axis2=n_batch + n_cur + q_tr)
        n_cur -= 1

    sorted_keep = sorted(keep)
    if list(keep) != sorted_keep:
        perm_local = [sorted_keep.index(q) for q in keep]
        perm = (
            list(range(n_batch))
            + [n_batch + i for i in perm_local]
            + [n_batch + k + i for i in perm_local]
        )
        rho_q = jnp.transpose(rho_q, perm)

    rho_red = rho_q.reshape(batch_shape + (2**k, 2**k))
    return normalize_trace(rho_red, eps=eps)


def bipartite_entanglement_entropy(
    psi_or_rho: jax.Array,
    num_qubits: int,
    keep_qubits: tuple[int, ...],
    eps: float = 1e-8,
) -> jax.Array:
    """Compute bipartite von Neumann entanglement entropy S(rho_A) for keep_qubits."""
    rho_a = partial_trace_qubits(
        psi_or_rho,
        num_qubits=num_qubits,
        keep_qubits=keep_qubits,
        eps=eps,
    )
    return von_neumann_entropy(rho_a, eps=eps)


def empirical_state_density(psi: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Compute the empirical ensemble density matrix rho = (1/N) sum_{i=1}^N |psi_i><psi_i|."""
    if psi.ndim < 1 or psi.shape[-1] < 1 or psi.size == 0:
        raise ValueError(f"Expected non-empty state array of shape (..., D), got shape {psi.shape}.")
    d = psi.shape[-1]
    n = psi.size // d
    psi_norm = l2_normalize(ensure_complex(psi.reshape(n, d)), axis=-1, eps=eps)
    rho = jnp.matmul(jnp.swapaxes(jnp.conj(psi_norm), -2, -1), psi_norm, precision=jax.lax.Precision.HIGHEST) / float(n)
    rho = 0.5 * (rho + jnp.conj(jnp.swapaxes(rho, -2, -1)))
    return normalize_trace(rho, eps=eps)


def _resolve_density_matrix_jax(
    psi_or_rho: jax.Array,
    eps: float = 1e-8,
    *,
    is_density: bool = False,
) -> jax.Array:
    if is_density:
        if psi_or_rho.ndim < 2 or psi_or_rho.shape[-2] != psi_or_rho.shape[-1]:
            raise ValueError(
                f"When is_density=True, expected square density matrix (..., D, D), got shape {psi_or_rho.shape}."
            )
        rho = ensure_complex(psi_or_rho)
        rho = 0.5 * (rho + jnp.conj(jnp.swapaxes(rho, -2, -1)))
        return normalize_trace(rho, eps=eps)
    return empirical_state_density(psi_or_rho, eps=eps)


def effective_quantum_rank(
    psi_or_rho: jax.Array,
    eps: float = 1e-8,
    *,
    is_density: bool = False,
) -> jax.Array:
    """Compute the Effective Quantum Rank R_eff(rho) = exp(S(rho)) in [1, D]."""
    rho = _resolve_density_matrix_jax(psi_or_rho, eps=eps, is_density=is_density)
    return jnp.exp(von_neumann_entropy(rho, eps=eps))


def purity_participation_ratio(
    psi_or_rho: jax.Array,
    eps: float = 1e-8,
    *,
    is_density: bool = False,
) -> jax.Array:
    """Compute the Purity Participation Ratio PR(rho) = 1 / Tr(rho^2) in [1, D]."""
    rho = _resolve_density_matrix_jax(psi_or_rho, eps=eps, is_density=is_density)
    purity = jnp.sum(jnp.abs(rho) ** 2, axis=(-2, -1))
    return 1.0 / jnp.maximum(purity, eps)

