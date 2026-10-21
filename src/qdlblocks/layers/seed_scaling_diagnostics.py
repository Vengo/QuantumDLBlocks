"""Diagnostics for Two-Beam Bilinear Interference vs. Single-Beam Overcomplete Frames and Seed-Space Scaling."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import effective_quantum_rank, l2_normalize, purity_participation_ratio
from .seed_adapter import compute_input_sensitivity_effective_rank

__all__ = [
    "single_beam_signed_born_features",
    "two_beam_bilinear_born_features",
    "compute_interference_feature_rank",
    "compute_cross_beam_coherence",
    "compute_paired_beam_orthogonality_stats",
    "compute_cross_subspace_regression_mse",
    "compute_coupled_seed_pullback_rank",
    "compute_effective_seed_utilization_ratio",
]


def single_beam_signed_born_features(
    psi: torch.Tensor,
    W_up: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute degree-2 signed Born features for a single overcomplete frame.

    Computes:
        f_1beam(psi) = (W_unit psi)^{odot 2} odot sign(W_unit psi)

    Args:
        psi: Input quantum state tensor of shape `(..., D)`.
        W_up: Frame projection matrix of shape `(M, D)`.
        eps: Numerical stability floor for row normalization.

    Returns:
        Signed quadratic Born feature tensor of shape `(..., M)`.
    """
    w_unit = l2_normalize(W_up, dim=-1, eps=eps).to(psi.dtype)
    z = F.linear(psi, w_unit)
    return (z * z) * torch.sign(z)


def two_beam_bilinear_born_features(
    psi: torch.Tensor,
    W_beam1: torch.Tensor,
    W_beam2: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute degree-3 cross-subspace bilinear Born features for two paired beams.

    Computes:
        f_2beam(psi) = (W1_unit psi)^{odot 2} odot SiLU(W2_unit psi)

    Args:
        psi: Input quantum state tensor of shape `(..., D)`.
        W_beam1: First beam projection matrix of shape `(M, D)`.
        W_beam2: Second beam projection matrix of shape `(M, D)`.
        eps: Numerical stability floor for row normalization.

    Returns:
        Two-beam bilinear Born feature tensor of shape `(..., M)`.
    """
    w1_unit = l2_normalize(W_beam1, dim=-1, eps=eps).to(psi.dtype)
    w2_unit = l2_normalize(W_beam2, dim=-1, eps=eps).to(psi.dtype)
    z1 = F.linear(psi, w1_unit)
    z2 = F.linear(psi, w2_unit)
    return (z1 * z1) * F.silu(z2)


def compute_interference_feature_rank(
    features: torch.Tensor,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute spectral Shannon effective rank, participation ratio, and trace of interference features.

    Args:
        features: Feature tensor of shape `(..., M)`.
        eps: Numerical stability floor.

    Returns:
        Dictionary with `"feature_eff_rank"`, `"feature_participation_ratio"`, and `"feature_trace"`.
    """
    if features.ndim < 1 or features.shape[-1] < 1 or features.numel() == 0:
        raise ValueError(
            f"compute_interference_feature_rank expects non-empty tensor (..., M), got {tuple(features.shape)}"
        )
    flat = features.detach().reshape(-1, features.shape[-1]).to(torch.float64)
    n = int(flat.shape[0])
    if n <= 1:
        return {
            "feature_eff_rank": 1.0,
            "feature_participation_ratio": 1.0,
            "feature_trace": 0.0,
        }
    f_c = flat - flat.mean(dim=0, keepdim=True)
    c_mat = (f_c.T @ f_c) / float(n - 1)
    eigvals = torch.linalg.eigvalsh(c_mat).clamp(min=0.0)
    trace = eigvals.sum()
    p = eigvals / (trace + float(eps))
    eff_rank = torch.exp(-(p * torch.log(p + float(eps))).sum())
    pr = 1.0 / ((p * p).sum() + float(eps))
    return {
        "feature_eff_rank": float(eff_rank.item()),
        "feature_participation_ratio": float(pr.item()),
        "feature_trace": float(trace.item()),
    }


def compute_cross_beam_coherence(
    W_beam1: torch.Tensor,
    W_beam2: torch.Tensor,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute root-mean-square and maximum paired-row coherence between two beam matrices.

    Args:
        W_beam1: First beam matrix of shape `(M, D)`.
        W_beam2: Second beam matrix of shape `(M, D)`.
        eps: Numerical stability floor.

    Returns:
        Dictionary with `"rms_paired_beam_coherence"` and `"max_paired_beam_coherence"`.
    """
    if W_beam1.ndim != 2 or W_beam2.ndim != 2 or W_beam1.shape != W_beam2.shape:
        raise ValueError(
            f"compute_cross_beam_coherence expects 2D matrices of identical shape (M, D), "
            f"got {tuple(W_beam1.shape)} and {tuple(W_beam2.shape)}"
        )
    w1_u = l2_normalize(W_beam1.detach().float(), dim=-1, eps=eps)
    w2_u = l2_normalize(W_beam2.detach().float(), dim=-1, eps=eps)
    row_cos = (w1_u * w2_u).sum(dim=-1)
    return {
        "rms_paired_beam_coherence": float(torch.sqrt((row_cos * row_cos).mean()).item()),
        "max_paired_beam_coherence": float(row_cos.abs().max().item()),
    }


def compute_paired_beam_orthogonality_stats(
    W_beam1: torch.Tensor,
    W_beam2: torch.Tensor,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Alias for `compute_cross_beam_coherence` returning paired-beam coherence statistics."""
    return compute_cross_beam_coherence(W_beam1=W_beam1, W_beam2=W_beam2, eps=eps)


def compute_cross_subspace_regression_mse(
    features_train: torch.Tensor,
    targets_train: torch.Tensor,
    features_test: torch.Tensor,
    targets_test: torch.Tensor,
    ridge_lambda: float = 1e-4,
) -> dict[str, float]:
    """Evaluate closed-form ridge regression MSE from interference features to targets.

    Args:
        features_train: Training features of shape `(N_train, M)` or `(..., M)`.
        targets_train: Training targets of shape `(N_train, K)` or `(..., K)`.
        features_test: Held-out test features of shape `(N_test, M)` or `(..., M)`.
        targets_test: Held-out test targets of shape `(N_test, K)` or `(..., K)`.
        ridge_lambda: L2 regularization parameter lambda > 0.

    Returns:
        Dictionary with `"train_mse"` and `"test_mse"`.
    """
    f_tr_raw = features_train.detach().reshape(-1, features_train.shape[-1]).to(torch.float64)
    y_tr_raw = targets_train.detach().reshape(-1, targets_train.shape[-1]).to(torch.float64)
    f_te_raw = features_test.detach().reshape(-1, features_test.shape[-1]).to(torch.float64)
    y_te_raw = targets_test.detach().reshape(-1, targets_test.shape[-1]).to(torch.float64)

    m = int(f_tr_raw.shape[1])
    f_mean = f_tr_raw.mean(dim=0, keepdim=True)
    y_mean = y_tr_raw.mean(dim=0, keepdim=True)

    f_tr = f_tr_raw - f_mean
    y_tr = y_tr_raw - y_mean
    f_te = f_te_raw - f_mean
    y_te = y_te_raw - y_mean

    gram = f_tr.T @ f_tr + float(ridge_lambda) * torch.eye(m, dtype=torch.float64, device=f_tr.device)
    rhs = f_tr.T @ y_tr
    w_star = torch.linalg.solve(gram, rhs)

    train_mse = float(((f_tr @ w_star - y_tr) ** 2).mean().item())
    test_mse = float(((f_te @ w_star - y_te) ** 2).mean().item())
    return {
        "train_mse": train_mse,
        "test_mse": test_mse,
    }


def compute_coupled_seed_pullback_rank(
    module: nn.Module | Callable[[torch.Tensor], torch.Tensor],
    psi: torch.Tensor,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute the input-sensitivity pullback rank of a coupled backbone + seed readout map."""
    return compute_input_sensitivity_effective_rank(module=module, psi=psi, eps=eps)


def compute_effective_seed_utilization_ratio(
    phi_seed: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Compute effective quantum rank and normalized dimension utilization ratio of seed states."""
    if phi_seed.ndim < 1 or phi_seed.shape[-1] < 1 or phi_seed.numel() == 0:
        raise ValueError(
            f"compute_effective_seed_utilization_ratio expects non-empty (..., D_seed), got {tuple(phi_seed.shape)}"
        )
    d_seed = int(phi_seed.shape[-1])
    r_eff = float(effective_quantum_rank(phi_seed, eps=eps).item())
    pr = float(purity_participation_ratio(phi_seed, eps=eps).item())
    return {
        "seed_eff_rank": r_eff,
        "seed_participation_ratio": pr,
        "seed_dim": float(d_seed),
        "seed_utilization_ratio": r_eff / float(d_seed),
        "seed_pr_utilization_ratio": pr / float(d_seed),
    }
