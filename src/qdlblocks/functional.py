"""Stateless functional quantum operations on pure state vectors, density matrices, and Stiefel manifolds."""

from __future__ import annotations

import torch
import torch.nn.functional as F


# =====================================================================
# 1. Pure-State Hyperspherical & Stiefel Manifold Operations
# =====================================================================
class _L2NormalizeFn(torch.autograd.Function):
    """Memory-efficient L2 normalization saving only `y` (in `x.dtype`) and `r_inv` (`(..., 1)`)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int, eps: float) -> torch.Tensor:  # type: ignore[override]
        x_f = x.float()
        r_inv = torch.rsqrt(torch.sum(x_f * x_f, dim=dim, keepdim=True) + eps)
        y = (x_f * r_inv).to(x.dtype)
        ctx.save_for_backward(y, r_inv)
        ctx.dim = dim
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        y, r_inv = ctx.saved_tensors
        gy = grad_out.float()
        yf = y.float()
        dot = torch.sum(gy * yf, dim=ctx.dim, keepdim=True)
        return ((gy - dot * yf) * r_inv).to(y.dtype), None, None


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """Project state vectors onto the unit hypersphere S^{d-1} (||psi||_2 = 1).

    Args:
        x: Input tensor of any shape.
        dim: Dimension along which to normalize (default: -1).
        eps: Small constant for numerical stability.

    Returns:
        Unit-norm tensor on S^{d-1} preserving `x.dtype`.
    """
    if x.is_complex():
        norm_sq = torch.sum(x.real.float().square() + x.imag.float().square(), dim=dim, keepdim=True)
        return x * torch.rsqrt(norm_sq + eps).to(x.real.dtype)
    if (
        x.requires_grad
        and x.ndim == 3
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.numel() > 10_000_000
        and not torch.compiler.is_compiling()
    ):
        return _L2NormalizeFn.apply(x, int(dim), float(eps))  # type: ignore[return-value]
    x_f = x.float()
    return (x_f * torch.rsqrt(torch.sum(x_f * x_f, dim=dim, keepdim=True) + eps)).type_as(x)



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
        Rotated tensor of shape `(..., d)` with identical L2 norm and dtype.
    """
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    c = cos.to(x.dtype)
    s = sin.to(x.dtype)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


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


def zeropower_via_newtonschulz5_batched(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Batched 3D/2D Quintic Newton-Schulz iteration computing orthogonal polar factors via `torch.bmm`.

    Accepts a 2D matrix `(M, N)` or a 3D batch of same-shape matrices `(K, M, N)` and
    executes the 5-step quintic Newton-Schulz iteration in a single batched `torch.bmm` loop.

    Args:
        G: 2D `(M, N)` or 3D `(K, M, N)` tensor.
        steps: Number of quintic Newton-Schulz iterations (default: 5).
        eps: Norm stability floor.

    Returns:
        Orthogonalized polar factor tensor with the same shape and dtype as `G`.
    """
    if G.ndim == 2:
        return zeropower_via_newtonschulz5(G, steps=steps, eps=eps)
    if G.ndim != 3:
        raise ValueError(f"zeropower_via_newtonschulz5_batched expects a 2D or 3D tensor, got {G.ndim}D")
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()
    norms = torch.linalg.matrix_norm(X, ord="fro", keepdim=True) + eps
    X = X / norms
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True
    for _ in range(steps):
        A = torch.bmm(X, X.mT)
        B = b * A + c * torch.bmm(A, A)
        X = a * X + torch.bmm(B, X)
    if transposed:
        X = X.mT
    return X.type_as(G)


ANVIL_QUINTIC_MAPS: tuple[tuple[float, float, float], ...] = (
    (3.923798038567, -6.095026865488, 3.905234618423),
    (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615),
    (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461),
    (2.256537145403, -2.166840097229, 0.929501253245),
)


def zeropower_via_polarexpress5(
    G: torch.Tensor,
    steps: int = 5,
    frobenius_margin: float = 1.05,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Non-stationary PolarExpress / ANVIL quintic polar orthogonalization.

    Uses per-step Remez-equioscillating quintic coefficients (`ANVIL_QUINTIC_MAPS`)
    to pull small singular values up to 1.0 faster than stationary Newton-Schulz.
    """
    if G.ndim != 2:
        raise ValueError(f"zeropower_via_polarexpress5 expects a 2D tensor, got {G.ndim}D")
    X = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()
    X = X / (float(frobenius_margin) * X.norm() + eps)
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True
    n_steps = max(1, int(steps))
    for t in range(n_steps):
        a, b, c = ANVIL_QUINTIC_MAPS[min(t, len(ANVIL_QUINTIC_MAPS) - 1)]
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.type_as(G)


def zeropower_via_polarexpress5_batched(
    G: torch.Tensor,
    steps: int = 5,
    frobenius_margin: float = 1.05,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Batched 3D/2D PolarExpress / ANVIL quintic polar orthogonalization via `torch.bmm`."""
    if G.ndim == 2:
        return zeropower_via_polarexpress5(G, steps=steps, frobenius_margin=frobenius_margin, eps=eps)
    if G.ndim != 3:
        raise ValueError(f"zeropower_via_polarexpress5_batched expects a 2D or 3D tensor, got {G.ndim}D")
    X = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()
    norms = torch.linalg.matrix_norm(X, ord="fro", keepdim=True) * float(frobenius_margin) + eps
    X = X / norms
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True
    n_steps = max(1, int(steps))
    for t in range(n_steps):
        a, b, c = ANVIL_QUINTIC_MAPS[min(t, len(ANVIL_QUINTIC_MAPS) - 1)]
        A = torch.bmm(X, X.mT)
        B = b * A + c * torch.bmm(A, A)
        X = a * X + torch.bmm(B, X)
    if transposed:
        X = X.mT
    return X.type_as(G)


def normuon_equalize(
    U: torch.Tensor,
    lane_energy: torch.Tensor | None = None,
    beta2: float = 0.95,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-lane energy equalization (NorMuon) after polar orthogonalization."""
    red_dim = -2 if U.size(-2) >= U.size(-1) else -1
    lane_power = U.float().square().mean(dim=red_dim, keepdim=True)
    lane_len = float(U.size(red_dim))
    pre_norm = (lane_power.sum(dim=(-2, -1), keepdim=True) * lane_len).sqrt()
    if lane_energy is None or lane_energy.shape != lane_power.shape:
        lane_energy = lane_power.clone()
    else:
        lane_energy.lerp_(lane_power.to(lane_energy.dtype), 1.0 - float(beta2))
    lane_gain = lane_energy.float().clamp_min(eps).rsqrt()
    post_norm = ((lane_power * lane_len) * lane_gain.square()).sum(dim=(-2, -1), keepdim=True).sqrt()
    eq_scale = lane_gain * (pre_norm / post_norm.clamp_min(eps))
    return (U * eq_scale.type_as(U)), lane_energy




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


def partial_trace_qubits(
    psi_or_rho: torch.Tensor,
    num_qubits: int,
    keep_qubits: tuple[int, ...],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the reduced density matrix rho_A = Tr_B(rho) over a subset of qubits in PyTorch.

    When `psi_or_rho` is a pure state vector `(..., 2^n)`, uses the exact Schmidt
    decomposition reshape `M in C^(..., 2^k, 2^(n-k))` to compute `rho_A = M @ M^dagger`
    in `O(2^(n+k))` time and `O(4^k)` memory without materializing a `2^n x 2^n`
    density matrix (e.g., for a 10-qubit `5+5` split, multiplies a `32 x 32` Schmidt
    matrix instead of allocating a `1024 x 1024` density matrix).

    Args:
        psi_or_rho: Pure state vector `(..., 2^n)` or density matrix `(..., 2^n, 2^n)`.
        num_qubits: Total number of qubits `n >= 1`.
        keep_qubits: Tuple of qubit indices in `[0, n-1]` to retain in subsystem A.
        eps: Numerical stability floor for trace normalization.

    Returns:
        Complex reduced density matrix `rho_A` of shape `(..., 2^k, 2^k)`.
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
        psi_norm = l2_normalize(ensure_complex(psi_or_rho), dim=-1, eps=eps)
        batch_shape = tuple(psi_norm.shape[:-1])
        n_batch = len(batch_shape)
        k = len(keep)
        trace_out = tuple(q for q in range(num_qubits) if q not in keep)
        psi_q = psi_norm.reshape(*batch_shape, *([2] * num_qubits))
        perm = (
            list(range(n_batch))
            + [n_batch + q for q in keep]
            + [n_batch + q for q in trace_out]
        )
        psi_perm = psi_q.permute(*perm)
        schmidt_mat = psi_perm.reshape(*batch_shape, 2**k, 2 ** (num_qubits - k))
        rho_red = torch.matmul(schmidt_mat, schmidt_mat.conj().transpose(-2, -1))
        return normalize_trace(rho_red, eps=eps)

    if is_density:
        rho = normalize_trace(ensure_complex(psi_or_rho), eps=eps)
    else:
        raise ValueError(
            f"Expected state or density with dimension {dim} (2**{num_qubits}), got shape {tuple(psi_or_rho.shape)}."
        )

    batch_shape = tuple(rho.shape[:-2])
    n_batch = len(batch_shape)
    k = len(keep)
    trace_out = tuple(q for q in range(num_qubits) if q not in keep)
    if len(trace_out) == 0 and keep == tuple(range(num_qubits)):
        return rho

    rho_q = rho.reshape(*batch_shape, *([2] * num_qubits), *([2] * num_qubits))
    n_cur = num_qubits
    for q_tr in reversed(sorted(trace_out)):
        rho_q = torch.diagonal(rho_q, dim1=n_batch + q_tr, dim2=n_batch + n_cur + q_tr).sum(dim=-1)
        n_cur -= 1

    sorted_keep = sorted(keep)
    if list(keep) != sorted_keep:
        perm_local = [sorted_keep.index(q) for q in keep]
        perm = (
            list(range(n_batch))
            + [n_batch + i for i in perm_local]
            + [n_batch + k + i for i in perm_local]
        )
        rho_q = rho_q.permute(*perm)

    rho_red = rho_q.reshape(*batch_shape, 2**k, 2**k)
    return normalize_trace(rho_red, eps=eps)


def bipartite_entanglement_entropy(
    psi_or_rho: torch.Tensor,
    num_qubits: int,
    keep_qubits: tuple[int, ...],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute bipartite von Neumann entanglement entropy S(rho_A) = -Tr(rho_A log rho_A) in nats."""
    rho_a = partial_trace_qubits(
        psi_or_rho,
        num_qubits=num_qubits,
        keep_qubits=keep_qubits,
        eps=eps,
    )
    rho_sym = 0.5 * (rho_a + rho_a.conj().transpose(-2, -1))
    if not torch.compiler.is_compiling() and not torch.isfinite(rho_sym).all():
        return torch.zeros(rho_sym.shape[:-2], device=rho_sym.device, dtype=torch.float32)
    try:
        evals = torch.linalg.eigvalsh(rho_sym).real
    except RuntimeError:
        return torch.zeros(rho_sym.shape[:-2], device=rho_sym.device, dtype=torch.float32)
    p = torch.clamp(evals, min=0.0)
    p = p / (torch.sum(p, dim=-1, keepdim=True) + eps)
    return -torch.sum(p * torch.log(torch.clamp(p, min=eps)), dim=-1)


def empirical_state_density(psi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute the empirical ensemble density matrix rho = (1/N) sum_{i=1}^N |psi_i><psi_i|.

    Args:
        psi: Real or complex state tensor of shape `(..., D)`.
        eps: Numerical stability floor for state and trace normalization.

    Returns:
        Hermitian, unit-trace complex density matrix of shape `(D, D)`.
    """
    if psi.ndim < 1 or psi.shape[-1] < 1 or psi.numel() == 0:
        raise ValueError(f"Expected non-empty state tensor of shape (..., D), got shape {tuple(psi.shape)}.")
    d = psi.shape[-1]
    n = psi.numel() // d
    psi_norm = l2_normalize(ensure_complex(psi.reshape(n, d)), dim=-1, eps=eps)
    rho = torch.matmul(psi_norm.conj().transpose(-2, -1), psi_norm) / float(n)
    rho = 0.5 * (rho + rho.conj().transpose(-2, -1))
    return normalize_trace(rho, eps=eps)


def _resolve_density_matrix(
    psi_or_rho: torch.Tensor,
    eps: float = 1e-8,
    *,
    is_density: bool = False,
) -> torch.Tensor:
    if is_density:
        if psi_or_rho.ndim < 2 or psi_or_rho.shape[-2] != psi_or_rho.shape[-1]:
            raise ValueError(
                f"When is_density=True, expected square density matrix (..., D, D), got shape {tuple(psi_or_rho.shape)}."
            )
        rho = ensure_complex(psi_or_rho)
        rho = 0.5 * (rho + rho.conj().transpose(-2, -1))
        return normalize_trace(rho, eps=eps)
    return empirical_state_density(psi_or_rho, eps=eps)


def effective_quantum_rank(
    psi_or_rho: torch.Tensor,
    eps: float = 1e-8,
    *,
    is_density: bool = False,
) -> torch.Tensor:
    """Compute the Effective Quantum Rank R_eff(rho) = exp(S(rho)) in [1, D].

    Args:
        psi_or_rho: Batch of state vectors `(..., D)` when `is_density=False`,
            or square density matrix `(..., D, D)` when `is_density=True`.
        eps: Numerical stability floor.
        is_density: Static flag indicating whether `psi_or_rho` is already a density matrix.

    Returns:
        Real tensor of effective quantum rank values in `[1, D]`.
    """
    rho = _resolve_density_matrix(psi_or_rho, eps=eps, is_density=is_density)
    evals = torch.linalg.eigvalsh(rho).real
    p = torch.clamp(evals, min=0.0)
    p = p / (torch.sum(p, dim=-1, keepdim=True) + eps)
    entropy = -torch.sum(p * torch.log(torch.clamp(p, min=eps)), dim=-1)
    return torch.exp(entropy)


def purity_participation_ratio(
    psi_or_rho: torch.Tensor,
    eps: float = 1e-8,
    *,
    is_density: bool = False,
) -> torch.Tensor:
    """Compute the Purity Participation Ratio PR(rho) = 1 / Tr(rho^2) in [1, D].

    Args:
        psi_or_rho: Batch of state vectors `(..., D)` when `is_density=False`,
            or square density matrix `(..., D, D)` when `is_density=True`.
        eps: Numerical stability floor.
        is_density: Static flag indicating whether `psi_or_rho` is already a density matrix.

    Returns:
        Real tensor of participation ratios in `[1, D]`.
    """
    rho = _resolve_density_matrix(psi_or_rho, eps=eps, is_density=is_density)
    pur = torch.sum(torch.abs(rho) ** 2, dim=(-2, -1))
    return 1.0 / torch.clamp(pur, min=eps)

