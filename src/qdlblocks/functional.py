"""Stateless functional quantum operations on pure state vectors, density matrices, and Stiefel manifolds."""

from __future__ import annotations

import torch
import torch.nn.functional as F


# =====================================================================
# 1. Pure-State Hyperspherical & Stiefel Manifold Operations
# =====================================================================
def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """Project state vectors onto the unit hypersphere S^{d-1} (||psi||_2 = 1).

    Args:
        x: Input tensor of any shape.
        dim: Dimension along which to normalize (default: -1).
        eps: Small constant for numerical stability.

    Returns:
        Unit-norm tensor on S^{d-1}.
    """
    return x * torch.rsqrt(torch.sum(x * x, dim=dim, keepdim=True) + eps)


def riemannian_center(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """Zero-mean center state vectors before projecting onto the unit hypersphere S^{d-1}.

    Mitigates anisotropic cone collapse (hyperspherical dimensional collapse) across deep layers.

    Args:
        x: Input state tensor.
        dim: Feature dimension to center and normalize.
        eps: Numerical stability constant.

    Returns:
        Centered and unit-normalized tensor on S^{d-1}.
    """
    centered = x - x.mean(dim=dim, keepdim=True)
    return l2_normalize(centered, dim=dim, eps=eps)


def apply_unitary_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply Unitary Rotary Position Embedding (RoPE) on SO(2)^{d/2}.

    Strictly preserves L2 state norms ||psi||_2 = 1.

    Args:
        x: Query or Key tensor of shape `(..., d)`.
        cos: Cosine frequency table broadcastable to `(..., d // 2)`.
        sin: Sine frequency table broadcastable to `(..., d // 2)`.

    Returns:
        Rotated tensor of shape `(..., d)` with identical L2 norm.
    """
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Quintic Newton-Schulz iteration computing the orthogonal polar factor on the Stiefel manifold.

    Computes the nearest semi-orthogonal matrix U = G (G^T G)^{-1/2} without an SVD:
        X_0 = G / ||G||_F
        X_{k+1} = a X_k + b (X_k X_k^T) X_k + c (X_k X_k^T)^2 X_k
    where (a, b, c) = (3.4445, -4.7750, 2.0315).

    Args:
        G: 2D matrix tensor of shape `(M, N)`.
        steps: Number of quintic Newton-Schulz iterations (default: 5).
        eps: Norm stability floor.

    Returns:
        Orthogonalized polar factor tensor with the same shape and dtype as `G`.
    """
    if G.ndim != 2:
        raise ValueError(f"zeropower_via_newtonschulz5 expects a 2D tensor, got {G.ndim}D")
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()
    X = X / (X.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.type_as(G)


# =====================================================================
# 2. Complex Density-Matrix & Open Quantum System Operations
# =====================================================================
def ensure_complex(x: torch.Tensor) -> torch.Tensor:
    """Ensure a tensor is complex-valued (`complex64` or `complex128`).

    Args:
        x: Input real or complex tensor.

    Returns:
        Complex tensor with matching precision.
    """
    if x.is_complex():
        return x
    if x.dtype == torch.float64:
        return x.to(torch.complex128)
    return x.to(torch.complex64)


def normalize_trace(rho: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize a batch of density matrices to have unit trace Tr(rho) = 1.

    Args:
        rho: Complex tensor of shape `(..., N, N)`.
        eps: Small constant for numerical stability.

    Returns:
        Trace-normalized density matrix of shape `(..., N, N)`.
    """
    tr = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1).real
    tr = tr.unsqueeze(-1).unsqueeze(-1)
    return rho / (tr + eps)


def state_to_density(psi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Form a normalized rank-1 pure state density matrix from state ket(s).

    Computes:
        rho = |psi><psi| / (||psi||^2 + eps)

    Args:
        psi: Complex state vector tensor of shape `(..., N)`.
        eps: Small constant to prevent division by zero.

    Returns:
        Pure density matrix tensor of shape `(..., N, N)`.
    """
    psi = ensure_complex(psi)
    psi_ket = psi.unsqueeze(-1)          # (..., N, 1)
    psi_bra = psi.conj().unsqueeze(-2)   # (..., 1, N)
    unnorm_rho = torch.matmul(psi_ket, psi_bra)  # (..., N, N)

    norm_sq = torch.sum(psi.real.square() + psi.imag.square(), dim=-1, keepdim=True).unsqueeze(-1)
    return unnorm_rho / (norm_sq + eps)


def power_purify(rho: torch.Tensor, power: int = 2, eps: float = 1e-12) -> torch.Tensor:
    """Apply non-linear Power State Purification to a density matrix.

    Computes:
        rho -> rho^p / (Tr(rho^p) + eps)

    On mixed states (rank > 1), this acts as a spectral non-linearity that
    amplifies dominant eigenstates and suppresses low-eigenvalue noise states,
    reducing von Neumann entropy.

    Args:
        rho: Complex density matrix tensor of shape `(..., N, N)`.
        power: Integer power p >= 1 (default: 2).
        eps: Small constant for trace normalization stability.

    Returns:
        Purified density matrix tensor of shape `(..., N, N)`.
    """
    if power < 1:
        raise ValueError(f"power must be >= 1, got {power}")
    if power == 1:
        return normalize_trace(rho, eps=eps)
    elif power == 2:
        rho_p = torch.matmul(rho, rho)
    else:
        rho_p = torch.linalg.matrix_power(rho, power)

    return normalize_trace(rho_p, eps=eps)


def convex_density_mixture(
    rhos: torch.Tensor,
    weights_logits: torch.Tensor,
    head_dim: int = -3,
) -> torch.Tensor:
    """Compute a convex combination of density matrices across heads.

    Computes:
        rho_mix = sum_h w_h rho_h,  where w = Softmax(weights_logits)

    Args:
        rhos: Complex density matrices with a head axis, e.g. `(B, H, N, N)` or `(B, H, L, N, N)`.
        weights_logits: Real 1D tensor of shape `(H,)` representing unnormalized head logits.
        head_dim: Axis corresponding to the heads (default: -3 for `(..., H, N, N)`).

    Returns:
        Mixed density matrix with `head_dim` reduced, e.g. `(B, N, N)` or `(B, L, N, N)`.
    """
    w = F.softmax(weights_logits, dim=0).to(rhos.dtype)
    ndim = rhos.dim()
    actual_dim = head_dim if head_dim >= 0 else ndim + head_dim
    view_shape = [1] * ndim
    view_shape[actual_dim] = w.numel()
    w_expanded = w.view(*view_shape)
    rho_mix = torch.sum(rhos * w_expanded, dim=actual_dim)
    rho_mix = 0.5 * (rho_mix + rho_mix.conj().transpose(-2, -1))
    return normalize_trace(rho_mix)


def kraus_transform(
    rho: torch.Tensor,
    kraus_ops: torch.Tensor,
    normalize: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Apply a quantum Kraus channel transformation to a density matrix.

    Computes:
        E(rho) = sum_{k=1}^K A_k rho A_k^dagger

    Supports both square (N -> N) and rectangular (N_in -> N_out) Kraus operators.

    Args:
        rho: Input density matrix of shape `(..., N_in, N_in)`.
        kraus_ops: Complex Kraus operators of shape `(K, N_out, N_in)`.
        normalize: Whether to renormalize the output trace to 1 (default: True).
        eps: Numerical stability constant for trace normalization.

    Returns:
        Transformed density matrix of shape `(..., N_out, N_out)`.
    """
    rho = ensure_complex(rho)
    kraus_ops = ensure_complex(kraus_ops)

    rho_out = torch.einsum("kim,...mn,kjn->...ij", kraus_ops, rho, kraus_ops.conj())
    rho_out = 0.5 * (rho_out + rho_out.conj().transpose(-2, -1))

    if normalize:
        rho_out = normalize_trace(rho_out, eps=eps)
    return rho_out


def povm_measure(
    rho: torch.Tensor,
    l_factors: torch.Tensor,
    return_operators: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute POVM measurement logits via Born's rule Tr(rho @ M_c).

    Each measurement operator is parameterized as M_c = L_c @ L_c^dagger >= 0,
    guaranteeing Hermiticity and positive semi-definiteness.
    Uses an O(B * C * N^2) einsum trace contraction instead of O(B * C * N^3) matmul.

    Args:
        rho: Density matrix tensor of shape `(..., N, N)`.
        l_factors: Complex operator factors of shape `(C, N, R)` (typically `R = N`).
        return_operators: If True, also returns the constructed `M` operators `(C, N, N)`.

    Returns:
        Real measurement logits of shape `(..., C)` (and `M` if `return_operators=True`).
    """
    rho = ensure_complex(rho)
    l_factors = ensure_complex(l_factors)

    m_ops = torch.matmul(l_factors, l_factors.conj().transpose(-2, -1))
    logits = torch.einsum("...ij,cji->...c", rho, m_ops).real

    if return_operators:
        return logits, m_ops
    return logits
