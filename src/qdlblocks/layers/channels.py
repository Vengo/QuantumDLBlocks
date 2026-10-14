"""Quantum channel transformations and convex ensemble mixing layers."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..functional import convex_density_mixture, kraus_transform


class ConvexHeadMixture(nn.Module):
    """Learnable convex combination of multi-head density matrices.

    Computes:
        rho_global = sum_{h=1}^H w_h rho_h,  where w = Softmax(head_weights)

    Because a convex combination of Hermitian PSD unit-trace matrices is also
    Hermitian, PSD, and unit-trace, this forms a valid mixed quantum state.
    """

    def __init__(self, num_heads: int, head_dim: int = -3) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)

    @property
    def mixture_probabilities(self) -> torch.Tensor:
        """Return normalized Softmax head weights of shape `(H,)`."""
        return torch.softmax(self.head_weights, dim=0)

    def forward(self, rhos: torch.Tensor) -> torch.Tensor:
        return convex_density_mixture(rhos, self.head_weights, head_dim=self.head_dim)

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}"


class KrausChannel(nn.Module):
    """Trainable Quantum Kraus Channel transformation layer.

    Maps an input density matrix `rho_in` in C^{N_in x N_in} to an output density
    matrix `rho_out` in C^{N_out x N_out} via K Kraus operators:
        E(rho) = sum_{k=1}^K A_k rho A_k^dagger
        rho_out = E(rho) / (Tr(E(rho)) + eps)

    Supports:
    - Square Hilbert space transitions (`in_hilbert_dim == out_hilbert_dim`)
    - Dimension-changing transitions (`in_hilbert_dim != out_hilbert_dim`)
    - Optional learnable convex residual connection when `in_hilbert_dim == out_hilbert_dim`
    """

    def __init__(
        self,
        in_hilbert_dim: int,
        out_hilbert_dim: int | None = None,
        num_kraus: int = 4,
        init_std: float = 0.1,
        residual: bool = False,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.in_hilbert_dim = in_hilbert_dim
        self.out_hilbert_dim = out_hilbert_dim if out_hilbert_dim is not None else in_hilbert_dim
        self.num_kraus = num_kraus
        self.eps = eps

        if residual and self.in_hilbert_dim != self.out_hilbert_dim:
            raise ValueError(
                "Residual connection requires in_hilbert_dim == out_hilbert_dim, "
                f"got {self.in_hilbert_dim} != {self.out_hilbert_dim}"
            )
        self.residual = residual

        self.A_r = nn.Parameter(
            torch.randn(num_kraus, self.out_hilbert_dim, self.in_hilbert_dim) * init_std
        )
        self.A_i = nn.Parameter(
            torch.randn(num_kraus, self.out_hilbert_dim, self.in_hilbert_dim) * init_std
        )

        if self.residual:
            # Sigmoid gate initialized at 0.0 -> 50/50 convex blend of identity and Kraus output
            self.residual_logit = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("residual_logit", None)

    @property
    def kraus_operators(self) -> torch.Tensor:
        """Return complex Kraus operators of shape `(K, N_out, N_in)`."""
        return torch.complex(self.A_r, self.A_i)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        rho_out = kraus_transform(rho, self.kraus_operators, normalize=True, eps=self.eps)
        if self.residual and self.residual_logit is not None:
            alpha = torch.sigmoid(self.residual_logit).to(rho_out.dtype)
            rho_out = alpha * rho + (1.0 - alpha) * rho_out
        return rho_out

    def extra_repr(self) -> str:
        return (
            f"in_hilbert_dim={self.in_hilbert_dim}, out_hilbert_dim={self.out_hilbert_dim}, "
            f"num_kraus={self.num_kraus}, residual={self.residual}"
        )
