"""POVM (Positive Operator-Valued Measure) quantum readout layers for density matrices and pure states."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..functional import l2_normalize, povm_measure


class POVMMeasurement(nn.Module):
    """Density-Matrix Positive Operator-Valued Measure (POVM) readout layer.

    Parameterizes `num_classes` positive semi-definite measurement operators
    in C^{N x N} via Cholesky-like factors:
        M_c = L_c @ L_c^dagger  (for c = 0, ..., num_classes - 1)

    Computes real observable expectation values / class logits via Born's trace rule:
        logits_c = Re(Tr(rho @ M_c))
    """

    def __init__(
        self,
        hilbert_dim: int,
        num_classes: int,
        operator_rank: int | None = None,
        init_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.hilbert_dim = hilbert_dim
        self.num_classes = num_classes
        self.operator_rank = operator_rank if operator_rank is not None else hilbert_dim

        self.L_r = nn.Parameter(
            torch.randn(num_classes, hilbert_dim, self.operator_rank) * init_std
        )
        self.L_i = nn.Parameter(
            torch.randn(num_classes, hilbert_dim, self.operator_rank) * init_std
        )

    @property
    def l_factors(self) -> torch.Tensor:
        """Return complex factor matrices `L` of shape `(num_classes, N, rank)`."""
        return torch.complex(self.L_r, self.L_i)

    @property
    def measurement_operators(self) -> torch.Tensor:
        """Return positive semi-definite POVM operators `M_c = L_c @ L_c^dagger` of shape `(C, N, N)`."""
        L = self.l_factors
        return torch.matmul(L, L.conj().transpose(-2, -1))

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """Measure density matrix batch `rho` of shape `(..., N, N)` -> logits `(..., num_classes)`."""
        return povm_measure(rho, self.l_factors, return_operators=False)

    def extra_repr(self) -> str:
        return (
            f"hilbert_dim={self.hilbert_dim}, num_classes={self.num_classes}, "
            f"operator_rank={self.operator_rank}"
        )


class ProjectivePOVMHead(nn.Module):
    """Pure-State Projective POVM Measurement Readout Head on S^{d-1}.

    Evaluates Cauchy-Schwarz bounded measurement overlaps `a_w = <M_w | psi_final>` in `[-1, 1]`
    against unit-normalized detector vectors `|M_w> in S^{d-1}`, scaled by inverse temperature
    `beta = 1 / povm_temperature`:
        logits_w = beta * <M_w | psi_final>

    Supports:
    - **Full-Rank Asymmetric Readout**: Allocates an independent detector frame `M` in the full
      Hilbert space `R^{vocab_size x embed_dim}`, avoiding low-rank readout bottlenecks.
    - **Learnable Inverse Temperature**: When `learnable_temperature=True`, optimizes `beta`
      clamped within `[min_inv_temp, max_inv_temp]` (default `[10.0, 100.0]`).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 1024,
        povm_temperature: float = 0.035,
        learnable_temperature: bool = False,
        min_inv_temp: float = 10.0,
        max_inv_temp: float = 100.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.min_inv_temp = min_inv_temp
        self.max_inv_temp = max_inv_temp
        self.eps = eps

        init_inv_temp = 1.0 / povm_temperature
        if learnable_temperature:
            self.inv_temp_param = nn.Parameter(torch.tensor(float(init_inv_temp)))
        else:
            self.register_parameter("inv_temp_param", None)
        self._fixed_inv_temp = float(init_inv_temp)

        self.povm_head = nn.Parameter(
            torch.randn(vocab_size, embed_dim) * (1.0 / math.sqrt(embed_dim))
        )
        with torch.no_grad():
            self.povm_head.copy_(l2_normalize(self.povm_head, dim=1, eps=eps))
        # Allow both head.effects and head.povm_head.effects to access the POVM matrix
        self.povm_head.effects = self.povm_head  # type: ignore[attr-defined]

    @property
    def effects(self) -> nn.Parameter:
        return self.povm_head

    @property
    def inv_temp(self) -> torch.Tensor | float:
        if self.inv_temp_param is not None:
            return torch.clamp(self.inv_temp_param, min=self.min_inv_temp, max=self.max_inv_temp)
        return self._fixed_inv_temp

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """Compute bounded Born measurement logits `(..., vocab_size)` from state `(..., embed_dim)`."""
        psi_final = l2_normalize(psi, dim=-1, eps=self.eps)
        M_norm = l2_normalize(self.povm_head, dim=1, eps=self.eps)
        amplitudes = torch.matmul(psi_final, M_norm.t())
        return amplitudes * self.inv_temp

