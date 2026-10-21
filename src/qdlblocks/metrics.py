"""Quantum information metrics, Stiefel manifold geometry checks, and spectral diagnostics."""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ReductionType = Literal["mean", "sum", "none"]


def _reduce(values: torch.Tensor, reduction: ReductionType) -> torch.Tensor:
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    raise ValueError(f"Unsupported reduction: {reduction!r}. Expected 'mean', 'sum', or 'none'.")


# =====================================================================
# 1. Density-Matrix Quantum Information Metrics
# =====================================================================
def von_neumann_entropy(
    rho: torch.Tensor,
    eps: float = 1e-12,
    reduction: ReductionType = "mean",
) -> torch.Tensor:
    """Compute the von Neumann entropy S(rho) = -Tr(rho log rho) = -sum_i lambda_i log(lambda_i).

    Args:
        rho: Hermitian density matrix tensor of shape `(..., N, N)`.
        eps: Minimum clamp threshold for eigenvalues before taking `log`.
        reduction: `'mean'`, `'sum'`, or `'none'`.

    Returns:
        Real tensor of von Neumann entropies (in nats). For pure states, S(rho) = 0;
        for maximally mixed states in C^N, S(rho) = log(N).
    """
    rho_sym = 0.5 * (rho + rho.conj().transpose(-2, -1))
    evals = torch.linalg.eigvalsh(rho_sym)
    evals_clamped = torch.clamp(evals, min=eps)
    entropy = -torch.sum(evals_clamped * torch.log(evals_clamped), dim=-1)
    return _reduce(entropy, reduction)


def purity(
    rho: torch.Tensor,
    reduction: ReductionType = "mean",
) -> torch.Tensor:
    """Compute quantum state purity gamma(rho) = Tr(rho^2) in O(N^2) time.

    Args:
        rho: Density matrix tensor of shape `(..., N, N)`.
        reduction: `'mean'`, `'sum'`, or `'none'`.

    Returns:
        Real tensor of state purities bounded in [1/N, 1].
    """
    tr_rho_sq = torch.einsum("...ij,...ji->...", rho, rho).real
    return _reduce(tr_rho_sq, reduction)


def hs_inner_product(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Compute Hilbert-Schmidt inner product Tr(rho @ sigma) between two Hermitian operators."""
    return torch.einsum("...ij,...ji->...", rho, sigma).real


def is_valid_density_matrix(
    rho: torch.Tensor,
    atol: float = 1e-4,
) -> tuple[bool, dict[str, float]]:
    """Verify whether a batch of matrices satisfies quantum density matrix axioms.

    Checks:
        1. Hermiticity: ||rho - rho^dagger||_max <= atol
        2. Unit Trace: |Tr(rho) - 1| <= atol and |Im(Tr(rho))| <= atol
        3. Positive Semi-Definiteness (PSD): min_eigenvalue >= -atol
    """
    with torch.no_grad():
        herm_diff = (rho - rho.conj().transpose(-2, -1)).abs().max().item()
        traces = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1)
        trace_real_err = (traces.real - 1.0).abs().max().item()
        trace_imag_err = traces.imag.abs().max().item()

        rho_sym = 0.5 * (rho + rho.conj().transpose(-2, -1))
        min_eval = torch.linalg.eigvalsh(rho_sym).min().item()

        is_valid = (
            herm_diff <= atol
            and trace_real_err <= atol
            and trace_imag_err <= atol
            and min_eval >= -atol
        )
        diagnostics = {
            "max_hermiticity_error": herm_diff,
            "max_trace_real_error": trace_real_err,
            "max_trace_imag_error": trace_imag_err,
            "min_eigenvalue": min_eval,
        }
        return is_valid, diagnostics


# =====================================================================
# 2. Stiefel Manifold & Quantum-Geometric Born Telemetry Metrics
# =====================================================================
@torch.no_grad()
def stiefel_isometry_error(weight: torch.Tensor) -> float:
    """Compute deviation from Stiefel manifold orthogonality: ||W^T W - I||_F / sqrt(d).

    For rectangular matrices W in R^{M x N}, evaluates W^T W - I_N if M >= N,
    or W W^T - I_M if M < N.

    Args:
        weight: 2D weight tensor.

    Returns:
        Normalized Frobenius isometry error (0.0 for exact Stiefel isometries).
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got {weight.ndim}D")
    W = weight.detach().float()
    if W.size(0) >= W.size(1):
        gram = W.t() @ W
    else:
        gram = W @ W.t()
    ident = torch.eye(gram.size(0), device=W.device, dtype=W.dtype)
    err = torch.norm(gram - ident, p="fro") / math.sqrt(gram.size(0))
    return float(err.item())


@torch.no_grad()
def model_isometry_error(
    model: nn.Module,
    param_filter: Callable[[str, nn.Parameter], bool] | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute per-layer and mean Stiefel isometry error across 2D weight matrices of a model.

    By default, inspects all 2D parameters excluding token embedding tables (`wte`)
    and vocabulary POVM readout frames (`povm_head`).

    Args:
        model: PyTorch module to inspect.
        param_filter: Optional predicate `(name, param) -> bool`.

    Returns:
        Tuple `(mean_isometry_error, per_parameter_error_dict)`.
    """
    if param_filter is None:
        _non_stiefel = (
            "wte",
            "povm_head",
            "embed",
            "value_embeds",
            "unet_lambdas",
            "vocab_higher_affinity",
            "token_radial_scale",
            "kraus",
            "kboson",
            "diffuse",
            "gate",
            "mudd",
            "smear",
            "purity",
            "adapter",
            "lambdas",
        )
        def param_filter(name: str, p: nn.Parameter) -> bool:
            return p.ndim == 2 and not any(sub in name for sub in _non_stiefel)

    errors: dict[str, float] = {}
    for name, p in model.named_parameters():
        if param_filter(name, p):
            errors[name] = stiefel_isometry_error(p)

    mean_err = float(np.mean(list(errors.values()))) if errors else 0.0
    return mean_err, errors


@torch.no_grad()
def effective_rank(
    activations: torch.Tensor,
    max_rows: int = 4096,
    eps: float = 1e-12,
) -> float:
    """Compute the Effective Dimensional Rank of hidden state activations via singular value entropy.

    Defined as:
        Rank_eff = exp( -sum_i p_i ln(p_i) ),  where p_i = sigma_i / sum_j sigma_j

    Detects hyperspherical dimensional collapse (where state vectors cluster into a narrow cone).

    Args:
        activations: Hidden state tensor of shape `(..., D)`.
        max_rows: Maximum flattened token rows to subsample for fast SVD.
        eps: Numerical stability floor.

    Returns:
        Effective rank in `[1.0, D]`.
    """
    H_flat = activations.detach().reshape(-1, activations.size(-1)).float()
    if H_flat.size(0) > max_rows:
        H_flat = H_flat[:max_rows]
    _, S, _ = torch.linalg.svd(H_flat, full_matrices=False)
    S_norm = S / (torch.sum(S) + eps)
    entropy = -torch.sum(S_norm * torch.log(S_norm + eps)).item()
    return float(math.exp(entropy))


@torch.no_grad()
def attention_entropy_and_sparsity(
    born_weights: torch.Tensor,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute Shannon entropy H(A) and exact zero sparsity of Born attention probabilities.

    Args:
        born_weights: Normalized Born attention matrix of shape `(..., T, T)`.
        eps: Floor for log computation.

    Returns:
        Dict with `'entropy'` (in nats) and `'sparsity'` (fraction of exact zeros in `[0, 1]`).
    """
    w = born_weights.detach().float()
    sparsity = float((w == 0.0).float().mean().item())
    entropy = -torch.sum(w * torch.log(w + eps), dim=-1).mean().item()
    return {
        "entropy": float(entropy),
        "sparsity": sparsity,
    }


@torch.no_grad()
def logit_margin_stats(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float | int]:
    """Compute predictive margin spectrum, target token rank, and Top-1/Top-5 accuracies.

    Diagnoses readout rank bottlenecks (e.g., high Top-5 accuracy with stalled Top-1 accuracy).

    Args:
        logits: Model output logits of shape `(B, T, V)` or `(N, V)`.
        targets: Ground-truth class/token indices of shape `(B, T)` or `(N,)`.

    Returns:
        Dictionary containing:
        - `mean_top1_top2_margin`: float
        - `max_top1_top2_margin`: float
        - `median_target_rank`: int (0-indexed rank; 0 means Top-1 is correct)
        - `top1_accuracy`: float (percentage in `[0, 100]`)
        - `top5_accuracy`: float (percentage in `[0, 100]`)
        - `eval_cross_entropy`: float
    """
    logits_flat = logits.detach().reshape(-1, logits.size(-1)).float()
    y_flat = targets.detach().reshape(-1).long()

    loss = F.cross_entropy(logits_flat, y_flat).item()

    top2_vals, _ = torch.topk(logits_flat, k=min(2, logits_flat.size(-1)), dim=-1)
    if top2_vals.size(-1) >= 2:
        margins = (top2_vals[:, 0] - top2_vals[:, 1]).cpu().numpy()
    else:
        margins = np.zeros(logits_flat.size(0), dtype=np.float32)

    target_logits = logits_flat[torch.arange(len(y_flat), device=logits_flat.device), y_flat]
    ranks = (logits_flat > target_logits.unsqueeze(-1)).sum(dim=-1).cpu().numpy()

    top1_acc = float(np.mean(ranks == 0) * 100.0)
    top5_acc = float(np.mean(ranks < 5) * 100.0)

    return {
        "mean_top1_top2_margin": float(np.mean(margins)),
        "max_top1_top2_margin": float(np.max(margins)),
        "median_target_rank": int(np.median(ranks)),
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "eval_cross_entropy": float(loss),
    }


@torch.no_grad()
def model_quantum_scale_telemetry(model: nn.Module) -> dict[str, float | list[float]]:
    """Extract per-head Born attention scales gamma_h^(l), POVM inverse temperature beta, and interferometry gates.

    Args:
        model: `QuantumBornGPT` or compatible module.

    Returns:
        Dictionary with `gamma_head_mean`, `gamma_head_min`, `gamma_head_max`,
        `gamma_layer_means`, `povm_beta`, and `unet_gate_scales`.
    """
    raw = getattr(model, "_orig_mod", model)
    blocks = getattr(raw, "blocks", None)
    all_gammas: list[float] = []
    layer_means: list[float] = []
    if blocks is not None:
        for blk in blocks:
            attn = getattr(blk, "attn", None)
            if attn is not None:
                hs = getattr(attn, "head_scale", None)
                if hs is not None:
                    vals = torch.clamp(hs.detach().float().view(-1), min=1.0, max=100.0).cpu().tolist()
                else:
                    scale_val = float(getattr(attn, "scale", 1.0))
                    n_h = int(getattr(attn, "num_heads", 1))
                    vals = [scale_val] * n_h
                all_gammas.extend(vals)
                layer_means.append(round(float(np.mean(vals)), 4))

    povm_beta = 1.0 / 0.035
    if getattr(raw, "tied_inv_temp", None) is not None:
        povm_beta = float(torch.clamp(raw.tied_inv_temp.detach().float(), min=1.0, max=60.0).item())
    elif getattr(raw, "povm_readout", None) is not None:
        inv_t = raw.povm_readout.inv_temp
        povm_beta = float(inv_t.detach().float().item()) if isinstance(inv_t, torch.Tensor) else float(inv_t)
    elif hasattr(raw, "_dual_inv_temp"):
        povm_beta = float(getattr(raw, "_dual_inv_temp"))

    unet_scales: list[float] = []
    if getattr(raw, "unet_scales", None) is not None:
        u_t = 0.35 * torch.tanh(raw.unet_scales.detach().float())
        unet_scales = [round(float(v), 4) for v in u_t.cpu().tolist()]
    elif getattr(raw, "unet_lambdas", None) is not None:
        u_t = raw.unet_lambdas[:, 1].detach().float()
        unet_scales = [round(float(v), 4) for v in u_t.cpu().tolist()]

    return {
        "gamma_head_mean": round(float(np.mean(all_gammas)), 4) if all_gammas else 0.0,
        "gamma_head_min": round(float(np.min(all_gammas)), 4) if all_gammas else 0.0,
        "gamma_head_max": round(float(np.max(all_gammas)), 4) if all_gammas else 0.0,
        "gamma_layer_means": layer_means,
        "povm_beta": round(povm_beta, 4),
        "unet_gate_scales": unet_scales,
    }

