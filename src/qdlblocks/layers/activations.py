"""Quantum non-linear activations for density matrices and pure-state projective channels."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import l2_normalize, power_purify


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


class TwoBeamQuantumInterference(nn.Module):
    """Two-Beam Bilinear Quantum Wave Interference Channel on S^{d-1}.

    Replaces classical elementwise activations (GELU/SiLU) with multiplicative two-beam
    Stiefel wave interference on the unit hypersphere:
        |phi_mlp> = N( W_down ( (W_a |psi>) * N(W_b |psi>) * sqrt(D_ff) ) )
    where W_a, W_b in St(D_ff, d) and W_down in St(D_ff, d) are orthogonal Stiefel
    projections. Constructive and destructive interference between the two orthogonal
    Stiefel paths gates information purely via quantum wave mechanics.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        normalize_output: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.normalize_output = normalize_output
        self.eps = eps

        self.w_a = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w_b = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, embed_dim, bias=False)
        nn.init.orthogonal_(self.w_a.weight)
        nn.init.orthogonal_(self.w_b.weight)
        nn.init.orthogonal_(self.w_down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        psi = l2_normalize(x, dim=-1, eps=self.eps)
        a = self.w_a(psi)
        b = l2_normalize(self.w_b(psi), dim=-1, eps=self.eps)
        interfered = a * b * math.sqrt(self.hidden_dim)
        out = self.w_down(interfered)
        if self.normalize_output:
            return l2_normalize(out, dim=-1, eps=self.eps)
        return out

