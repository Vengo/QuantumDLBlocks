"""Overcomplete Born Seed Readout Adapter and Input-Sensitivity Pullback Rank Telemetry."""

from __future__ import annotations

from collections.abc import Callable
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import l2_normalize
from .overcomplete_frame import (
    frame_coherence_metrics,
    optimize_overcomplete_tight_frame,
)

__all__ = [
    "compute_input_sensitivity_effective_rank",
    "OvercompleteSeedReadoutAdapter",
]


def compute_input_sensitivity_effective_rank(
    module: nn.Module | Callable[[torch.Tensor], torch.Tensor],
    psi: torch.Tensor,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute the input-sensitivity pullback metric effective rank R_eff(G) on S^{D-1}.

    Forms the empirical pullback metric over N input states {psi_i}_{i=1}^N in R^D:
        G = (1 / N) sum_{i=1}^N J(psi_i)^T J(psi_i) in R^{D x D}
    where J(psi_i) = d phi(psi_i) / d psi_i in R^{D_seed x D}, and computes the
    spectral Shannon effective rank and participation ratio of G.

    Args:
        module: Callable or `nn.Module` mapping `(N, D) -> (N, D_seed)`.
        psi: Input state batch of shape `(N, D)` (or `(..., D)`).
        eps: Numerical stability floor.

    Returns:
        Dictionary with keys `"pullback_eff_rank"`, `"pullback_participation_ratio"`,
        and `"trace_G"`.
    """
    if psi.ndim < 2 or psi.shape[-1] < 1 or psi.numel() == 0:
        raise ValueError(
            f"compute_input_sensitivity_effective_rank expects psi of shape (N, D) with N >= 1, D >= 1; "
            f"got {tuple(psi.shape)}."
        )
    d = int(psi.shape[-1])
    n = int(psi.numel() // d)

    with torch.enable_grad():
        x = psi.detach().reshape(n, d).float().clone().requires_grad_(True)
        out = module(x)
        if out.ndim != 2 or out.shape[0] != n:
            out = out.reshape(n, -1)
        d_seed = int(out.shape[1])

        g_mat = torch.zeros((d, d), device=x.device, dtype=torch.float64)
        for k in range(d_seed):
            (grad_k,) = torch.autograd.grad(
                out[:, k].sum(),
                x,
                retain_graph=(k < d_seed - 1),
                create_graph=False,
                allow_unused=True,
            )
            if grad_k is not None:
                gk_64 = grad_k.to(torch.float64)
                g_mat = g_mat + (gk_64.transpose(0, 1) @ gk_64) / float(n)

    evals = torch.linalg.eigvalsh(g_mat).clamp(min=0.0)
    trace_g = torch.sum(evals)
    p = evals / (trace_g + float(eps))
    pullback_eff_rank = torch.exp(-torch.sum(p * torch.log(p + float(eps))))
    pullback_pr = 1.0 / torch.clamp(torch.sum(p * p), min=float(eps))

    return {
        "pullback_eff_rank": float(pullback_eff_rank.item()),
        "pullback_participation_ratio": float(pullback_pr.item()),
        "trace_G": float(trace_g.item()),
    }


class OvercompleteSeedReadoutAdapter(nn.Module):
    """Overcomplete Born Seed Readout Adapter breaking the (D - D_seed) linear nullspace bottleneck.

    Augments the linear Stiefel seed collapse phi_lin(psi) = Normalize(W_lift^T psi_final)
    with a degree-2 spherical harmonic quadratic Born interference branch over an
    overcomplete tight frame U_frame in R^{M x D} (M = frame_expansion * D):
        q(psi) = (D / sqrt(D_seed)) * ((U_unit psi)^{odot 2} - (1 / D) 1_M)
        phi_over(psi) = Normalize(phi_lin + seed_gate odot (W_down q(psi)))
    """

    def __init__(
        self,
        embed_dim: int,
        seed_dim: int,
        frame_expansion: int = 2,
        init_gate: float = 0.5,
        tight_frame_init_steps: int = 80,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {embed_dim}")
        if seed_dim < 1:
            raise ValueError(f"seed_dim must be >= 1, got {seed_dim}")
        if frame_expansion < 1:
            raise ValueError(f"frame_expansion must be >= 1, got {frame_expansion}")

        self.embed_dim = int(embed_dim)
        self.seed_dim = int(seed_dim)
        self.frame_expansion = int(frame_expansion)
        self.frame_dim = self.frame_expansion * self.embed_dim
        self.init_gate = float(init_gate)
        self.tight_frame_init_steps = int(tight_frame_init_steps)
        self.eps = float(eps)

        w_init = optimize_overcomplete_tight_frame(
            self.frame_dim,
            self.embed_dim,
            steps=self.tight_frame_init_steps,
            lr=0.05,
        )
        self.U_frame = nn.Parameter(w_init.clone())
        self.W_down = nn.Parameter(torch.empty(self.seed_dim, self.frame_dim, dtype=torch.float32))
        nn.init.orthogonal_(self.W_down)
        self.seed_gate = nn.Parameter(
            torch.full((self.seed_dim,), float(init_gate), dtype=torch.float32)
        )

    def forward(self, psi_final: torch.Tensor, phi_linear: torch.Tensor) -> torch.Tensor:
        """Fuse linear seed state `phi_linear` with overcomplete quadratic Born features of `psi_final`."""
        u_unit = l2_normalize(self.U_frame, dim=-1, eps=self.eps).to(psi_final.dtype)
        z = F.linear(psi_final, u_unit)
        scale = float(self.embed_dim) / math.sqrt(float(self.seed_dim))
        q = (z * z - (1.0 / float(self.embed_dim))) * scale
        delta_seed = F.linear(q, self.W_down.to(psi_final.dtype)) * self.seed_gate.to(psi_final.dtype)
        return l2_normalize(phi_linear + delta_seed, dim=-1, eps=self.eps)

    def frame_metrics(self) -> dict[str, float]:
        """Return overcomplete frame coherence metrics and mean absolute seed gate magnitude."""
        metrics = frame_coherence_metrics(self.U_frame, eps=self.eps)
        metrics["mean_abs_seed_gate"] = float(self.seed_gate.detach().abs().mean().item())
        return metrics
