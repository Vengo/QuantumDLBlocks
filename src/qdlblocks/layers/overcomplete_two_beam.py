"""Overcomplete Two-Beam Quantum Interference Layer with Welch-Bounded Tight Frames and Paired-Beam Orthogonality."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import l2_normalize
from .overcomplete_frame import frame_coherence_metrics

__all__ = [
    "two_beam_frame_penalty",
    "overcomplete_two_beam_metrics",
    "optimize_orthogonal_pair_tight_frames",
    "OvercompleteTwoBeamInterference",
]


def _unit_tight_frame_penalty(w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute scale-normalized Parseval tight-frame Gram penalty on row-normalized frame vectors."""
    if w.ndim != 2:
        raise ValueError(f"Expected a 2D weight tensor of shape (M, d), got shape {tuple(w.shape)}")
    wf = w.float()
    m, d = wf.shape
    u = l2_normalize(wf, dim=-1, eps=eps)
    rho = float(m) / float(d)
    eye_d = torch.eye(d, device=wf.device, dtype=wf.dtype)
    diff = (u.transpose(0, 1) @ u) - rho * eye_d
    return torch.sum(diff * diff) / ((rho**2) * float(d))


def two_beam_frame_penalty(
    W_beam1: torch.Tensor,
    W_beam2: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute unified two-beam tight-frame and paired-row orthogonality penalty.

    Args:
        W_beam1: First beam frame matrix of shape `(M, d)`.
        W_beam2: Second beam frame matrix of shape `(M, d)`.
        eps: Numerical floor for row normalization.

    Returns:
        Scalar penalty tensor `0.5 * (L_frame(W_beam1) + L_frame(W_beam2)) + mean_m(<u1_m, u2_m>^2)`.
    """
    if W_beam1.ndim != 2 or W_beam2.ndim != 2 or W_beam1.shape != W_beam2.shape:
        raise ValueError(
            f"W_beam1 and W_beam2 must be 2D tensors of identical shape (M, d), "
            f"got {tuple(W_beam1.shape)} and {tuple(W_beam2.shape)}"
        )
    p1 = _unit_tight_frame_penalty(W_beam1, eps=eps)
    p2 = _unit_tight_frame_penalty(W_beam2, eps=eps)
    u1 = l2_normalize(W_beam1.float(), dim=-1, eps=eps)
    u2 = l2_normalize(W_beam2.float(), dim=-1, eps=eps)
    p_pair = ((u1 * u2).sum(dim=-1) ** 2).mean()
    return 0.5 * (p1 + p2) + p_pair


def overcomplete_two_beam_metrics(
    W_beam1: torch.Tensor,
    W_beam2: torch.Tensor,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute Welch-bound coherence, tight-frame error, paired-beam coherence, and naive isometry error."""
    m1 = frame_coherence_metrics(W_beam1, eps=eps)
    m2 = frame_coherence_metrics(W_beam2, eps=eps)
    u1 = l2_normalize(W_beam1.detach().float(), dim=-1, eps=eps)
    u2 = l2_normalize(W_beam2.detach().float(), dim=-1, eps=eps)
    paired_rms = float(torch.sqrt(((u1 * u2).sum(dim=-1) ** 2).mean()).item())
    return {
        "beam1_tight_frame_err": m1["tight_frame_err"],
        "beam2_tight_frame_err": m2["tight_frame_err"],
        "beam1_rms_coherence": m1["rms_coherence"],
        "beam2_rms_coherence": m2["rms_coherence"],
        "welch_bound": m1["welch_bound"],
        "paired_beam_rms_coherence": paired_rms,
        "naive_iso_err": 0.5 * (m1["iso_err_vs_identity"] + m2["iso_err_vs_identity"]),
    }


def optimize_orthogonal_pair_tight_frames(
    num_vectors: int,
    dim: int,
    steps: int = 200,
    lr: float = 0.08,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a pair of unit-row tight frames `(W1, W2)` in `R^{num_vectors x dim}` with orthogonal paired rows."""
    if num_vectors < 1 or dim < 1:
        raise ValueError(f"num_vectors and dim must be >= 1, got num_vectors={num_vectors}, dim={dim}")
    dev = torch.device(device)
    gen = torch.Generator(device=dev).manual_seed(int(seed))

    with torch.enable_grad():
        w1 = torch.randn((num_vectors, dim), generator=gen, device=dev, dtype=torch.float32)
        w2 = torch.randn((num_vectors, dim), generator=gen, device=dev, dtype=torch.float32)
        w1 = l2_normalize(w1, dim=-1)
        w2 = l2_normalize(w2 - (w2 * w1).sum(dim=-1, keepdim=True) * w1, dim=-1)

        if num_vectors <= dim:
            q, r = torch.linalg.qr(w1.transpose(0, 1))
            d_diag = torch.diagonal(r)
            signs = torch.where(d_diag >= 0, torch.ones_like(d_diag), -torch.ones_like(d_diag))
            w1_ortho = l2_normalize((q * signs.unsqueeze(0)).transpose(0, 1).contiguous(), dim=-1)
            if num_vectors >= 2:
                w2_ortho = torch.roll(w1_ortho, shifts=1, dims=0)
            else:
                w2_ortho = l2_normalize(w2 - (w2 * w1_ortho).sum(dim=-1, keepdim=True) * w1_ortho, dim=-1)
            return l2_normalize(w1_ortho.detach(), dim=-1), l2_normalize(w2_ortho.detach(), dim=-1)

        if steps > 0:
            p1 = nn.Parameter(w1.clone())
            p2 = nn.Parameter(w2.clone())
            opt = torch.optim.Adam([p1, p2], lr=float(lr))
            for _ in range(int(steps)):
                opt.zero_grad()
                loss = two_beam_frame_penalty(p1, p2)
                loss.backward()
                opt.step()
            w1 = p1.detach()
            w2 = p2.detach()

    return l2_normalize(w1.detach(), dim=-1), l2_normalize(w2.detach(), dim=-1)


class OvercompleteTwoBeamInterference(nn.Module):
    """Overcomplete Two-Beam Quantum Interference Layer with Welch-Bounded 2d Tight Frames."""

    def __init__(
        self,
        embed_dim: int,
        frame_expansion: float = 2.0,
        target_iso_scale: float = 1.25,
        init_steps: int = 200,
        seed: int = 42,
        eps: float = 1e-12,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.frame_expansion = float(frame_expansion)
        self.target_iso_scale = float(target_iso_scale)
        self.init_steps = int(init_steps)
        self.seed = int(seed)
        self.eps = float(eps)
        self.frame_dim = int(round(self.embed_dim * self.frame_expansion))

        w1_init, w2_init = optimize_orthogonal_pair_tight_frames(
            self.frame_dim,
            self.embed_dim,
            steps=self.init_steps,
            seed=self.seed,
            device=device or "cpu",
        )
        self.W_beam1 = nn.Parameter(w1_init * self.target_iso_scale)
        self.W_beam2 = nn.Parameter(w2_init * self.target_iso_scale)
        self.W_out = nn.Parameter(
            w1_init.t().contiguous()
            * (self.target_iso_scale * math.sqrt(float(self.embed_dim) / float(self.frame_dim)))
        )
        self.post_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.last_beam2: torch.Tensor | None = None

    def forward(
        self,
        psi: torch.Tensor,
        return_second_beam: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        u1 = l2_normalize(self.W_beam1, dim=-1, eps=self.eps).to(psi.dtype)
        u2 = l2_normalize(self.W_beam2, dim=-1, eps=self.eps).to(psi.dtype)
        scale = math.sqrt(float(self.embed_dim))
        z1 = F.linear(psi, u1) * scale
        z2 = F.linear(psi, u2) * scale
        interf = (z1 * z1) * F.silu(z2)
        out_1 = F.linear(interf, self.W_out.to(psi.dtype))
        psi_main = l2_normalize(self.post_norm(out_1.float()).to(psi.dtype), dim=-1, eps=self.eps)

        b2 = (z2 * z2) * F.silu(z1)
        out_2 = F.linear(b2, u2.t().to(psi.dtype)) * (
            self.target_iso_scale * math.sqrt(float(self.embed_dim) / float(self.frame_dim))
        )
        psi_beam2 = l2_normalize(out_2 - 0.25 * out_1, dim=-1, eps=self.eps)
        self.last_beam2 = psi_beam2
        if return_second_beam:
            return psi_main, psi_beam2
        return psi_main

    def frame_penalty(self) -> torch.Tensor:
        return two_beam_frame_penalty(self.W_beam1, self.W_beam2, eps=self.eps)

    def coherence_metrics(self) -> dict[str, float]:
        return overcomplete_two_beam_metrics(self.W_beam1, self.W_beam2, eps=self.eps)
