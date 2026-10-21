"""Overcomplete Tight Frames, Welch-Bound Coherence Metrics, and Relaxed-Isometry Interference Layers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import l2_normalize

__all__ = [
    "welch_bound_coherence",
    "tight_frame_penalty",
    "frame_coherence_metrics",
    "optimize_overcomplete_tight_frame",
    "OvercompleteFrameInterference",
]


def _extract_weight_tensor(w: torch.Tensor | nn.Module) -> torch.Tensor:
    if isinstance(w, torch.Tensor):
        return w
    if hasattr(w, "weight") and isinstance(w.weight, torch.Tensor):
        return w.weight
    raise TypeError(f"Expected a 2D torch.Tensor or module with .weight, got {type(w)!r}")


def welch_bound_coherence(num_vectors: int, dim: int) -> float:
    """Compute the theoretical Welch lower bound on root-mean-square off-diagonal coherence.

    For M = num_vectors unit vectors in R^d (with M > d):
        mu_Welch(M, d) = sqrt((M - d) / (d * (M - 1)))
    For M <= d, orthogonal packing is achievable and the bound is 0.0.
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if num_vectors <= dim:
        return 0.0
    return math.sqrt((num_vectors - dim) / float(dim * (num_vectors - 1)))


def tight_frame_penalty(w: torch.Tensor | nn.Module, target_iso_scale: float = 1.25) -> torch.Tensor:
    """Compute the differentiable Parseval tight-frame potential penalty.

    Orients `w` as (M, d) with M >= d and penalizes deviation from the scaled
    Parseval resolution of the identity:
        R_frame(W) = (1 / d) * || W^T W - (target_iso_scale^2 * M / d) I_d ||_F^2
    """
    wt = _extract_weight_tensor(w)
    if wt.ndim != 2:
        raise ValueError(f"tight_frame_penalty expects a 2D weight tensor, got shape {tuple(wt.shape)}")
    wf = wt.float()
    if wf.shape[0] < wf.shape[1]:
        wf = wf.transpose(0, 1)
    m, d = wf.shape
    gamma = (float(target_iso_scale) ** 2) * (float(m) / float(d))
    gram_dim = wf.transpose(0, 1) @ wf
    eye_d = torch.eye(d, device=wf.device, dtype=wf.dtype)
    diff = gram_dim - gamma * eye_d
    return torch.sum(diff * diff) / float(d)


def frame_coherence_metrics(w: torch.Tensor | nn.Module, eps: float = 1e-6) -> dict[str, float]:
    """Compute overcomplete frame coherence, tight-frame error, and naive Stiefel isometry error."""
    wt = _extract_weight_tensor(w)
    if wt.ndim != 2:
        raise ValueError(f"frame_coherence_metrics expects a 2D weight tensor, got shape {tuple(wt.shape)}")
    wf = wt.detach().float()
    if wf.shape[0] < wf.shape[1]:
        wf = wf.transpose(0, 1)
    m, d = wf.shape
    w_hat = l2_normalize(wf, dim=-1, eps=eps)
    redundancy_ratio = float(m) / float(d)

    eye_d = torch.eye(d, device=wf.device, dtype=torch.float32)
    gram_raw = wf.transpose(0, 1) @ wf
    iso_err_vs_identity = float(
        (torch.linalg.norm(gram_raw - eye_d, ord="fro") / math.sqrt(float(d))).item()
    )

    gram_unit = w_hat.transpose(0, 1) @ w_hat
    tight_frame_err = float(
        (torch.linalg.norm(gram_unit - redundancy_ratio * eye_d, ord="fro") / math.sqrt(float(d))).item()
    )

    if m <= 1:
        rms_coherence = 0.0
        max_coherence = 0.0
    else:
        g = w_hat @ w_hat.transpose(0, 1)
        g_off = g - torch.diag(torch.diag(g))
        rms_coherence = float(torch.sqrt(torch.sum(g_off * g_off) / float(m * (m - 1))).item())
        max_coherence = float(torch.max(torch.abs(g_off)).item())

    return {
        "redundancy_ratio": redundancy_ratio,
        "iso_err_vs_identity": iso_err_vs_identity,
        "tight_frame_err": tight_frame_err,
        "rms_coherence": rms_coherence,
        "max_coherence": max_coherence,
        "welch_bound": welch_bound_coherence(m, d),
    }


def optimize_overcomplete_tight_frame(
    num_vectors: int,
    dim: int,
    steps: int = 250,
    lr: float = 0.1,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Construct a unit-row overcomplete Parseval tight frame in R^{num_vectors x dim}.

    Minimizes the Benedetto-Fickus frame potential (1/d) ||W^T W - (M/d) I_d||_F^2 on the
    oblique manifold of unit-norm rows using closed-form Riemannian/projected gradient steps,
    so it executes safely both inside and outside `torch.no_grad()`.
    """
    if dim < 1 or num_vectors < dim:
        raise ValueError(
            f"Require num_vectors >= dim >= 1, got num_vectors={num_vectors}, dim={dim}"
        )
    dev = torch.device(device)
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))

    if num_vectors == dim:
        z = torch.randn((dim, dim), generator=gen, device=dev, dtype=torch.float32)
        q, r = torch.linalg.qr(z)
        d_diag = torch.diagonal(r)
        signs = torch.where(d_diag >= 0, torch.ones_like(d_diag), -torch.ones_like(d_diag))
        q = q * signs.unsqueeze(0)
        return l2_normalize(q, dim=-1)

    w = torch.randn((num_vectors, dim), generator=gen, device=dev, dtype=torch.float32)
    w = l2_normalize(w, dim=-1)
    rho = float(num_vectors) / float(dim)
    eye_d = torch.eye(dim, device=dev, dtype=torch.float32)

    for _ in range(int(steps)):
        g_err = w.transpose(0, 1) @ w - rho * eye_d
        grad_w = (4.0 / float(dim)) * (w @ g_err)
        w = l2_normalize(w - float(lr) * grad_w, dim=-1)

    return w


class OvercompleteFrameInterference(nn.Module):
    """Overcomplete 2d Tight-Frame Quantum Interference Layer with Controlled Isometry Scale."""

    def __init__(
        self,
        embed_dim: int,
        frame_expansion: float = 2.0,
        target_iso_scale: float = 1.25,
        seed: int = 42,
        eps: float = 1e-6,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.frame_expansion = float(frame_expansion)
        self.target_iso_scale = float(target_iso_scale)
        self.eps = float(eps)
        self.num_vectors = max(self.embed_dim, int(round(self.embed_dim * self.frame_expansion)))

        wa_unit = optimize_overcomplete_tight_frame(
            self.num_vectors, self.embed_dim, steps=80, seed=int(seed), device=device
        )
        wb_unit = optimize_overcomplete_tight_frame(
            self.num_vectors, self.embed_dim, steps=80, seed=int(seed) + 1, device=device
        )

        self.w_a = nn.Parameter((wa_unit * self.target_iso_scale).clone())
        self.w_b = nn.Parameter((wb_unit * self.target_iso_scale).clone())
        down_scale = math.sqrt(float(self.embed_dim) / float(self.num_vectors))
        self.w_down = nn.Parameter((wa_unit.transpose(0, 1) * down_scale).contiguous().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u_a = F.linear(x, self.w_a.to(x.dtype))
        u_b = F.linear(x, self.w_b.to(x.dtype))
        h = (F.silu(u_a) * u_b + 0.25 * (u_a.square() - u_b.square())) * float(self.num_vectors)
        y = F.linear(h, self.w_down.to(x.dtype))
        return l2_normalize(x + y, dim=-1, eps=self.eps)

    def frame_penalty(self) -> torch.Tensor:
        return 0.5 * (
            tight_frame_penalty(self.w_a, self.target_iso_scale)
            + tight_frame_penalty(self.w_b, self.target_iso_scale)
        )
