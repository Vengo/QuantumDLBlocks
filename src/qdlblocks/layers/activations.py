"""Quantum non-linear activations for density matrices and pure-state projective channels."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import power_purify


class PowerPurification(nn.Module):
    """Non-linear Power State Purification layer for density matrices.

    Applies the non-linear map:
        rho -> rho^p / (Tr(rho^p) + eps)

    When applied to mixed quantum states (rank(rho) > 1)—such as after incoherent
    token pooling, convex head mixing, or a Kraus channel—this map sharpens the
    eigenvalue spectrum $\\lambda_i \\mapsto \\lambda_i^p / \\sum_j \\lambda_j^p$,
    amplifying dominant eigenstates and suppressing noise while strictly preserving
    Hermiticity, positive semi-definiteness, and unit trace.
    """

    def __init__(self, power: int = 2, eps: float = 1e-12) -> None:
        super().__init__()
        if power < 1:
            raise ValueError(f"power must be >= 1, got {power}")
        self.power = power
        self.eps = eps

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        return power_purify(rho, power=self.power, eps=self.eps)

    def extra_repr(self) -> str:
        return f"power={self.power}, eps={self.eps}"


class ReLUSquared(nn.Module):
    """Projective ReLU^2 multi-particle channel activation.

    Applies positive-quadrant subspace filtering followed by Born amplitude squaring:
        f(x) = (max(0, x))^2

    Provides non-vanishing linear gradients 2 * max(0, x) for aligned states while
    filtering out anti-correlated components.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()
