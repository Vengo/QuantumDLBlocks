"""Probabilistic Calibration (ECE, Brier Score) and Out-of-Distribution (OOD) Detection Metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def probs_from_logits_or_probs(
    inputs: torch.Tensor,
    from_logits: bool | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convert raw logits or probability vectors into normalized probability distributions."""
    if from_logits is None:
        ones = torch.ones(inputs.shape[:-1], device=inputs.device, dtype=inputs.dtype)
        is_probs = bool(
            (inputs.min().item() >= -1e-5)
            and torch.allclose(inputs.sum(dim=-1), ones, atol=1e-3)
        )
    else:
        is_probs = not from_logits

    if is_probs:
        p = inputs.clamp(min=0.0)
        return p / (p.sum(dim=-1, keepdim=True) + eps)
    return F.softmax(inputs, dim=-1)


def expected_calibration_error(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 15,
    from_logits: bool | None = None,
) -> float:
    """Compute Expected Calibration Error (ECE) over equal-width confidence bins in [0, 1]."""
    probs = probs_from_logits_or_probs(inputs, from_logits=from_logits)
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == targets).float()

    boundaries = torch.linspace(0.0, 1.0, num_bins + 1, device=inputs.device, dtype=probs.dtype)
    ece = torch.zeros(1, device=inputs.device, dtype=probs.dtype)

    for b in range(num_bins):
        if b == 0:
            mask = (confidences >= boundaries[b]) & (confidences <= boundaries[b + 1])
        else:
            mask = (confidences > boundaries[b]) & (confidences <= boundaries[b + 1])
        if mask.any():
            prop = mask.float().mean()
            acc_bin = accuracies[mask].mean()
            conf_bin = confidences[mask].mean()
            ece = ece + prop * torch.abs(acc_bin - conf_bin)

    return float(ece.item())


def brier_score(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    from_logits: bool | None = None,
) -> float:
    """Compute multi-class Brier score E[sum_c (p_c - y_c)^2]."""
    probs = probs_from_logits_or_probs(inputs, from_logits=from_logits)
    one_hot = F.one_hot(targets, num_classes=probs.size(-1)).to(probs.dtype)
    return float(((probs - one_hot).square().sum(dim=-1).mean()).item())


def ood_auroc_from_scores(
    id_uncertainty: torch.Tensor,
    ood_uncertainty: torch.Tensor,
) -> float:
    """Compute exact Mann-Whitney U / Wilcoxon AUROC distinguishing OOD from ID uncertainty."""
    id_u = id_uncertainty.detach().reshape(-1).float()
    ood_u = ood_uncertainty.detach().reshape(-1).float()
    diff = ood_u[:, None] - id_u[None, :]
    wins = (diff > 0).float() + 0.5 * (diff == 0).float()
    return float(wins.mean().item())


ood_auroc = ood_auroc_from_scores
