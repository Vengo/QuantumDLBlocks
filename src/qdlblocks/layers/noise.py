"""Quantum noise layers for training regularization and robustness."""

from __future__ import annotations

import torch
import torch.nn as nn


class SPAMNoise(nn.Module):
    """State Preparation and Measurement (SPAM) Gaussian noise augmentation layer.

    During training (`model.train()`), injects zero-mean Gaussian noise with standard
    deviation `std` into input features, optionally clamping to `clamp_range`.
    Acts as an identity operation during evaluation (`model.eval()`).
    """

    def __init__(
        self,
        std: float = 0.02,
        clamp_range: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        self.std = std
        self.clamp_range = clamp_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.std <= 0.0:
            return x
        noisy = x + torch.randn_like(x) * self.std
        if self.clamp_range is not None:
            noisy = torch.clamp(noisy, self.clamp_range[0], self.clamp_range[1])
        return noisy

    def extra_repr(self) -> str:
        return f"std={self.std}, clamp_range={self.clamp_range}"


class DepolarizingNoise(nn.Module):
    """Quantum Depolarizing Channel noise layer for density matrices.

    During training (`model.train()`), mixes the density matrix with the maximally
    mixed state I / N with probability p in [0, 1]:
        rho -> (1 - p) rho + p (I / N)

    Strictly preserves Hermiticity, positive semi-definiteness, and unit trace, while
    increasing von Neumann entropy (simulating environmental decoherence).
    """

    def __init__(self, p: float = 0.05) -> None:
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Depolarizing probability p must be in [0, 1], got {p}")
        self.p = p

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return rho
        n = rho.shape[-1]
        eye = torch.eye(n, dtype=rho.dtype, device=rho.device)
        return (1.0 - self.p) * rho + (self.p / n) * eye

    def extra_repr(self) -> str:
        return f"p={self.p}"
