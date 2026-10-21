"""Phase-Sensitive Two-Beam Bilinear Seed Readout Adapter and Nullspace Antipodal Separation Telemetry."""

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
    "compute_nullspace_antipodal_separation",
    "optimize_orthogonal_pair_tight_frames",
    "TwoBeamSeedReadoutAdapter",
]


def compute_nullspace_antipodal_separation(
    adapter_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    W_lin: torch.Tensor,
    num_samples: int = 256,
    seed: int = 42,
    eps: float = 1e-12,
) -> float:
    """Compute the Nullspace Antipodal Separation metric Delta_antipodal(phi) on ker(W_lin) cap S^{D-1}.

    Given a linear seed projection matrix `W_lin` of shape `(D_seed, D)` with `D > D_seed`,
    computes an orthonormal basis Q_perp in R^{(D - D_seed) x D} for ker(W_lin) via SVD,
    samples `num_samples` unit vectors psi_perp in ker(W_lin) cap S^{D-1}, and evaluates:
        Delta_antipodal(phi) = E_{psi_perp}[ || phi(psi_perp, 0) - phi(-psi_perp, 0) ||_2 ]

    Args:
        adapter_fn: Callable `(psi_final, phi_linear) -> phi_seed` mapping `(N, D) x (N, D_seed) -> (N, D_seed)`.
        W_lin: Linear seed projection matrix of shape `(D_seed, D)` with `D > D_seed`.
        num_samples: Number of random nullspace unit vectors to sample.
        seed: Random generator seed for reproducibility.
        eps: Numerical stability floor.

    Returns:
        Mean Euclidean separation between antipodal nullspace states `phi(psi_perp)` and `phi(-psi_perp)`.
    """
    if W_lin.ndim != 2:
        raise ValueError(f"W_lin must be a 2D tensor of shape (D_seed, D), got {tuple(W_lin.shape)}")
    d_seed, d = int(W_lin.shape[0]), int(W_lin.shape[1])
    if d <= d_seed:
        raise ValueError(f"Require D > D_seed for non-trivial ker(W_lin), got D_seed={d_seed}, D={d}")
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    with torch.no_grad():
        _, _, vh = torch.linalg.svd(W_lin.detach().float(), full_matrices=True)
        q_perp = vh[d_seed:].to(device=W_lin.device, dtype=W_lin.dtype)  # (D - D_seed, D)

        gen = torch.Generator(device=W_lin.device)
        gen.manual_seed(int(seed))
        coeffs = torch.randn(
            (int(num_samples), d - d_seed),
            generator=gen,
            device=W_lin.device,
            dtype=W_lin.dtype,
        )
        psi_perp = l2_normalize(coeffs @ q_perp, dim=-1, eps=eps)
        zeros_seed = torch.zeros(
            (int(num_samples), d_seed),
            device=W_lin.device,
            dtype=W_lin.dtype,
        )
        phi_pos = adapter_fn(psi_perp, zeros_seed)
        phi_neg = adapter_fn(-psi_perp, zeros_seed)
        sep = torch.linalg.norm(phi_pos - phi_neg, dim=-1).mean()
        return float(sep.item())


def optimize_orthogonal_pair_tight_frames(
    num_vectors: int,
    dim: int,
    steps: int = 80,
    lr: float = 0.05,
    seed: int = 42,
    device: torch.device | str = "cpu",
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a pair of overcomplete tight frames `(U1, U2)` in R^{num_vectors x dim} with mutually orthogonal rows."""
    u1 = optimize_overcomplete_tight_frame(
        num_vectors=num_vectors,
        dim=dim,
        steps=steps,
        lr=lr,
        seed=seed,
        device=device,
    )
    u2 = optimize_overcomplete_tight_frame(
        num_vectors=num_vectors,
        dim=dim,
        steps=max(int(steps) // 2, 1),
        lr=lr,
        seed=int(seed) + 1000,
        device=device,
    )
    if dim <= 1:
        return u1, u2

    # Project rows of u2 orthogonal to rows of u1 and refine tight-frame conditioning
    u2 = u2 - torch.sum(u2 * u1, dim=-1, keepdim=True) * u1
    u2 = l2_normalize(u2, dim=-1, eps=eps)

    dev = torch.device(device)
    rho = float(num_vectors) / float(dim)
    eye_d = torch.eye(dim, device=dev, dtype=torch.float32)
    refine_steps = max(int(steps) // 2, 1)

    for _ in range(refine_steps):
        if num_vectors > dim:
            g_err = u2.transpose(0, 1) @ u2 - rho * eye_d
            grad_u2 = (4.0 / float(dim)) * (u2 @ g_err)
            u2 = u2 - float(lr) * grad_u2
        u2 = u2 - torch.sum(u2 * u1, dim=-1, keepdim=True) * u1
        u2 = l2_normalize(u2, dim=-1, eps=eps)

    return u1, u2


class TwoBeamSeedReadoutAdapter(nn.Module):
    """Phase-Sensitive Two-Beam Bilinear Seed Readout Adapter breaking antipodal sign degeneracy on ker(W_lin).

    Equips the D -> D_seed seed collapse interface with two mutually orthogonal Welch-bounded
    overcomplete tight frames U_1, U_2 in R^{M x D} (M = frame_expansion * D):
        q(psi) = (D / sqrt(D_seed)) * ((U_1 psi)^{odot 2} - (1 / D) 1_M)
        b(psi) = (D^{3/2} / sqrt(D_seed)) * ((U_1 psi)^{odot 2} odot SiLU(U_2 psi))
        phi_2beam(psi) = Normalize(phi_lin + seed_gate odot W_down (q(psi) + beam_eta * b(psi)))
    """

    def __init__(
        self,
        embed_dim: int,
        seed_dim: int,
        frame_expansion: int = 2,
        init_gate: float = 0.5,
        init_beam_eta: float = 0.6,
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
        self.init_beam_eta = float(init_beam_eta)
        self.tight_frame_init_steps = int(tight_frame_init_steps)
        self.eps = float(eps)

        u1_init, u2_init = optimize_orthogonal_pair_tight_frames(
            num_vectors=self.frame_dim,
            dim=self.embed_dim,
            steps=self.tight_frame_init_steps,
            lr=0.05,
            eps=self.eps,
        )
        self.U_beam1 = nn.Parameter(u1_init.clone())
        self.U_beam2 = nn.Parameter(u2_init.clone())
        self.W_down = nn.Parameter(torch.empty(self.seed_dim, self.frame_dim, dtype=torch.float32))
        nn.init.orthogonal_(self.W_down)
        self.seed_gate = nn.Parameter(
            torch.full((self.seed_dim,), float(init_gate), dtype=torch.float32)
        )
        self.beam_eta = nn.Parameter(torch.tensor(float(init_beam_eta), dtype=torch.float32))

    def forward(self, psi_final: torch.Tensor, phi_linear: torch.Tensor) -> torch.Tensor:
        """Fuse linear seed state `phi_linear` with degree-2 and phase-sensitive degree-3 two-beam features."""
        u1 = l2_normalize(self.U_beam1, dim=-1, eps=self.eps).to(psi_final.dtype)
        u2 = l2_normalize(self.U_beam2, dim=-1, eps=self.eps).to(psi_final.dtype)
        z1 = F.linear(psi_final, u1)
        z2 = F.linear(psi_final, u2)

        d_f = float(self.embed_dim)
        sqrt_d_seed = math.sqrt(float(self.seed_dim))
        scale_q = d_f / sqrt_d_seed
        scale_b = (d_f**1.5) / sqrt_d_seed

        z1_sq = z1 * z1
        q = (z1_sq - (1.0 / d_f)) * scale_q
        b = z1_sq * F.silu(z2) * scale_b

        combined = q + self.beam_eta.to(psi_final.dtype) * b
        delta_seed = F.linear(combined, self.W_down.to(psi_final.dtype)) * self.seed_gate.to(psi_final.dtype)
        return l2_normalize(phi_linear + delta_seed, dim=-1, eps=self.eps)

    def frame_metrics(self) -> dict[str, float]:
        """Return tight-frame coherence metrics, cross-beam orthogonality, and gate/eta telemetry."""
        m1 = frame_coherence_metrics(self.U_beam1, eps=self.eps)
        m2 = frame_coherence_metrics(self.U_beam2, eps=self.eps)
        with torch.no_grad():
            u1_hat = l2_normalize(self.U_beam1.detach().float(), dim=-1, eps=self.eps)
            u2_hat = l2_normalize(self.U_beam2.detach().float(), dim=-1, eps=self.eps)
            cross_cos = float(torch.sum(u1_hat * u2_hat, dim=-1).abs().mean().item())
        return {
            "redundancy_ratio": m1["redundancy_ratio"],
            "tight_frame_err": 0.5 * (m1["tight_frame_err"] + m2["tight_frame_err"]),
            "rms_coherence": 0.5 * (m1["rms_coherence"] + m2["rms_coherence"]),
            "max_coherence": max(m1["max_coherence"], m2["max_coherence"]),
            "welch_bound": m1["welch_bound"],
            "beam1_tight_frame_err": m1["tight_frame_err"],
            "beam2_tight_frame_err": m2["tight_frame_err"],
            "cross_beam_abs_cos": cross_cos,
            "mean_abs_seed_gate": float(self.seed_gate.detach().abs().mean().item()),
            "beam_eta": float(self.beam_eta.detach().item()),
        }
