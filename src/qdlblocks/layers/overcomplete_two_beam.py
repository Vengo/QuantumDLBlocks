"""Overcomplete Two-Beam Quantum Interference Layer with Welch-Bounded Tight Frames and Paired-Beam Orthogonality."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import l2_normalize
from .overcomplete_frame import frame_coherence_metrics
from .stereographic_lift import apply_dimension_preserving_stereographic_rotation

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
        use_stereographic_lift: bool = False,
        use_cosine_phase_beam_gate: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.frame_expansion = float(frame_expansion)
        self.target_iso_scale = float(target_iso_scale)
        self.init_steps = int(init_steps)
        self.seed = int(seed)
        self.eps = float(eps)
        self.frame_dim = int(round(self.embed_dim * self.frame_expansion))
        self.use_stereographic_lift = bool(use_stereographic_lift)
        self.use_cosine_phase_beam_gate = bool(use_cosine_phase_beam_gate)

        dev_init = torch.device(device) if device is not None else torch.device("cpu")
        w1_init, w2_init = optimize_orthogonal_pair_tight_frames(
            self.frame_dim,
            self.embed_dim,
            steps=self.init_steps,
            seed=self.seed,
            device=dev_init,
        )
        self.W_beam1 = nn.Parameter(w1_init * self.target_iso_scale)
        self.W_beam2 = nn.Parameter(w2_init * self.target_iso_scale)
        self.W_out = nn.Parameter(
            w1_init.t().contiguous()
            * (self.target_iso_scale * math.sqrt(float(self.embed_dim) / float(self.frame_dim)))
        )
        self.post_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

        if self.use_cosine_phase_beam_gate:
            self.phase_omega: nn.Parameter | None = nn.Parameter(
                torch.full((self.frame_dim,), 0.50, dtype=torch.float32, device=dev_init)
            )
            self.phase_phi: nn.Parameter | None = nn.Parameter(
                torch.zeros(self.frame_dim, dtype=torch.float32, device=dev_init)
            )
            self.phase_gate_scale: nn.Parameter | None = nn.Parameter(
                torch.tensor(0.35, dtype=torch.float32, device=dev_init)
            )
        else:
            self.register_parameter("phase_omega", None)
            self.register_parameter("phase_phi", None)
            self.register_parameter("phase_gate_scale", None)

        if self.use_stereographic_lift:
            gen_s = torch.Generator(device="cpu").manual_seed(self.seed + 303)
            p1_init = l2_normalize(
                torch.randn(self.embed_dim, generator=gen_s, dtype=torch.float32), dim=-1
            ).to(dev_init)
            p2_init = l2_normalize(
                torch.randn(self.embed_dim, generator=gen_s, dtype=torch.float32), dim=-1
            ).to(dev_init)
            self.stereo_pole_1: nn.Parameter | None = nn.Parameter(p1_init)
            self.stereo_pole_2: nn.Parameter | None = nn.Parameter(p2_init)
            self.stereo_tau: nn.Parameter | None = nn.Parameter(
                torch.tensor(0.55, dtype=torch.float32, device=dev_init)
            )
            self.stereo_gain: nn.Parameter | None = nn.Parameter(
                torch.tensor(0.40, dtype=torch.float32, device=dev_init)
            )
        else:
            self.register_parameter("stereo_pole_1", None)
            self.register_parameter("stereo_pole_2", None)
            self.register_parameter("stereo_tau", None)
            self.register_parameter("stereo_gain", None)

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
        b2 = (z2 * z2) * F.silu(z1)

        if (
            self.use_cosine_phase_beam_gate
            and self.phase_omega is not None
            and self.phase_phi is not None
            and self.phase_gate_scale is not None
        ):
            delta_theta = self.phase_omega.to(z1.dtype) * (z1 - z2) + self.phase_phi.to(z1.dtype)
            alpha_p = (0.50 * torch.tanh(self.phase_gate_scale)).to(z1.dtype)
            interf = interf * (1.0 + alpha_p * torch.cos(delta_theta))
            b2 = b2 * (1.0 + alpha_p * torch.sin(delta_theta))

        out_1 = F.linear(interf, self.W_out.to(psi.dtype))
        normed_1 = self.post_norm(out_1.float()).to(psi.dtype)
        if (
            self.use_stereographic_lift
            and self.stereo_pole_1 is not None
            and self.stereo_tau is not None
            and self.stereo_gain is not None
        ):
            out1_radial = out_1.float().norm(p=2, dim=-1, keepdim=True)
            psi_main = apply_dimension_preserving_stereographic_rotation(
                normed_1,
                self.stereo_pole_1,
                self.stereo_tau,
                self.stereo_gain,
                eps=self.eps,
                radial_norm_override=out1_radial,
            )
        else:
            psi_main = l2_normalize(normed_1, dim=-1, eps=self.eps)

        out_2 = F.linear(b2, u2.t().to(psi.dtype)) * (
            self.target_iso_scale * math.sqrt(float(self.embed_dim) / float(self.frame_dim))
        )
        raw_b2 = out_2 - 0.25 * out_1
        if (
            self.use_stereographic_lift
            and self.stereo_pole_2 is not None
            and self.stereo_tau is not None
            and self.stereo_gain is not None
        ):
            psi_beam2 = apply_dimension_preserving_stereographic_rotation(
                raw_b2,
                self.stereo_pole_2,
                self.stereo_tau,
                self.stereo_gain,
                eps=self.eps,
                radial_norm_override=raw_b2.float().norm(p=2, dim=-1, keepdim=True),
            )
        else:
            psi_beam2 = l2_normalize(raw_b2, dim=-1, eps=self.eps)

        if self.use_cosine_phase_beam_gate:
            b2_f = psi_beam2.float()
            m_f = psi_main.float()
            psi_beam2 = l2_normalize(
                b2_f - (b2_f * m_f).sum(dim=-1, keepdim=True) * m_f,
                dim=-1,
                eps=self.eps,
            ).to(psi.dtype)

        if not torch.compiler.is_compiling():
            self.last_beam2 = psi_beam2
        if return_second_beam:
            return psi_main, psi_beam2
        return psi_main

    def frame_penalty(self) -> torch.Tensor:
        return two_beam_frame_penalty(self.W_beam1, self.W_beam2, eps=self.eps)

    def coherence_metrics(self) -> dict[str, float]:
        return overcomplete_two_beam_metrics(self.W_beam1, self.W_beam2, eps=self.eps)

