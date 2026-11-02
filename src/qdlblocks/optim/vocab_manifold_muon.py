"""Active-Token Vocabulary Manifold Optimizers for Standalone Quantum Born GPT.

Provenance: Wave 48 (`EXPERIMENT_PLAN_063.md`) & *"Top-1 Accuracy vs Cross-Entropy Divergence"* (Pages 1-15).

Implements:
1. `BatchSubspaceTikhonovGramMuon`: Active-token `(Q, N_act, A)` Tikhonov-damped feature-Gram preconditioned
   Muon operating on the $d \\times d$ ($768 \\times 768$) active feature subspace with coupled Newton-Schulz
   inverse square root $A^{-1/2}$, SNR trust-region gating, and unit-RMS sphere retraction onto $\\sqrt{d}\\,\\mathbb{S}^{d-1}$.
2. `RiemannianSphereTangentSNROptimizer`: Oblique sphere tangent momentum with parallel transport, per-token
   SNR trust-region gating $\\theta_v = \\eta_{\\text{sphere}} \\frac{\\text{SNR}_v}{\\text{SNR}_v + \\tau_{\\text{snr}}}$, and
   exact great-circle geodesic retraction on $\\sqrt{d}\\,\\mathbb{S}^{d-1}$.
"""

from __future__ import annotations

import math
from typing import Any

import torch

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_063.md Section 3
# ==============================================================================
DEFAULT_SPHERE_SNR_TRUST_TAU: float = 0.50
DEFAULT_SPHERE_GEODESIC_LR: float = 0.08
DEFAULT_BATCH_GRAM_MUON_LR: float = 0.06
DEFAULT_TIKHONOV_GRAM_LAMBDA: float = 1e-3
DEFAULT_GRAM_WHITENING_MIX: float = 0.25
DEFAULT_VOCAB_MOMENTUM_BETA1: float = 0.90
DEFAULT_VOCAB_MOMENTUM_BETA2: float = 0.99
DEFAULT_COUPLED_NS_STEPS: int = 5
DEFAULT_VOCAB_RADIAL_LR_RATIO: float = 0.25
DEFAULT_VOCAB_RADIAL_MIN: float = 0.75
DEFAULT_VOCAB_RADIAL_MAX: float = 1.50
VOCAB_MANIFOLD_EPS: float = 1e-7


def project_rows_to_rms_sphere_(
    weight: torch.Tensor,
    eps: float = VOCAB_MANIFOLD_EPS,
) -> torch.Tensor:
    """Project each row of `weight` `(..., d)` in-place onto the unit-RMS sphere $\\sqrt{d}\\,\\mathbb{S}^{d-1}$."""
    with torch.no_grad():
        d_float = float(weight.size(-1))
        target_radius = math.sqrt(d_float)
        row_norm = weight.float().norm(dim=-1, keepdim=True).clamp_min(eps)
        scale = (target_radius / row_norm).to(weight.dtype)
        weight.mul_(scale)
    return weight


def _project_to_sphere_tangent(
    vec: torch.Tensor,
    anchor: torch.Tensor,
    eps: float = VOCAB_MANIFOLD_EPS,
) -> torch.Tensor:
    """Project `vec` `(N, d)` onto the sphere tangent space $T_{\\text{anchor}}\\mathbb{S}^{d-1}$ ($\\langle v_{\\perp}, w\\rangle = 0$)."""
    anchor_norm_sq = anchor.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    radial_coeff = (vec * anchor).sum(dim=-1, keepdim=True) / anchor_norm_sq
    return vec - radial_coeff * anchor


def _newton_schulz_coupled_inv_sqrt(
    gram_a: torch.Tensor,
    steps: int = DEFAULT_COUPLED_NS_STEPS,
) -> torch.Tensor:
    """Compute $A^{-1/2}$ on symmetric positive-definite $A \\in \\mathbb{R}^{d \\times d}$ via Frobenius-normalized coupled Newton-Schulz.

    Normalizing $\\hat{A} = A / \\|A\\|_F$ guarantees $\\|\\hat{A}\\|_2 \\le \\|\\hat{A}\\|_F = 1 < 3$ while elevating
    the spectrum by $\\sqrt{d_{\\text{eff}}}$ over trace normalization, accelerating cubic convergence of:
      $Y_0 = \\hat{A}, \\quad Z_0 = I_d$
      $T_k = \\frac{1}{2}(3 I_d - Z_k Y_k), \\quad Y_{k+1} = Y_k T_k, \\quad Z_{k+1} = T_k Z_k$
      $A^{-1/2} = \\|A\\|_F^{-1/2} Z_K$
    """
    d = gram_a.size(-1)
    fro_a = torch.linalg.matrix_norm(gram_a, ord="fro").clamp_min(VOCAB_MANIFOLD_EPS)
    eye = torch.eye(d, device=gram_a.device, dtype=gram_a.dtype)
    y_mat = gram_a / fro_a
    z_mat = eye.clone()
    for _ in range(max(6, int(steps) + 1)):
        t_mat = 0.5 * (3.0 * eye - torch.matmul(z_mat, y_mat))
        y_mat = torch.matmul(y_mat, t_mat)
        z_mat = torch.matmul(t_mat, z_mat)
    return z_mat * torch.rsqrt(fro_a)


def _compute_tikhonov_gram_inv_sqrt(
    m_tan: torch.Tensor,
    tik_lam: float,
    ns_steps: int,
) -> torch.Tensor:
    """Compute well-conditioned $d \\times d$ Tikhonov Gram inverse square root $A^{-1/2}$ from tangent momentum `m_tan`."""
    d_dim = m_tan.size(-1)
    eye_d = torch.eye(d_dim, device=m_tan.device, dtype=torch.float32)
    if m_tan.is_cuda and m_tan.size(0) >= 1024 and torch.cuda.is_bf16_supported():
        m_bf16 = m_tan.bfloat16()
        raw_gram = torch.matmul(m_bf16.transpose(0, 1), m_bf16).float()
    else:
        raw_gram = torch.matmul(m_tan.transpose(0, 1), m_tan)
    mean_eig = (raw_gram.diagonal().sum() / float(max(1, d_dim))).clamp_min(VOCAB_MANIFOLD_EPS)
    norm_gram = raw_gram / mean_eig
    mix = DEFAULT_GRAM_WHITENING_MIX
    gram_a = (1.0 - mix + tik_lam) * eye_d + mix * norm_gram
    return _newton_schulz_coupled_inv_sqrt(gram_a, steps=ns_steps)


def _apply_polar_radial_vocab_step(
    w_unit: torch.Tensor,
    w_unit_next: torch.Tensor,
    r_cur: torch.Tensor,
    m_tan: torch.Tensor,
    m_rad: torch.Tensor,
    theta: torch.Tensor,
    radial_lr_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update per-row vocabulary radial scale $r_v \\in [r_{\\min}, r_{\\max}]$ and pack orthogonal tangent+radial momentum (`<= 8` args)."""
    target_radius = math.sqrt(float(w_unit.size(-1)))
    m_rad_scalar = (m_rad * w_unit).sum(dim=-1, keepdim=True) / target_radius
    tot_m_norm = (m_tan.square().sum(dim=-1, keepdim=True) + m_rad_scalar.square()).sqrt().clamp_min(VOCAB_MANIFOLD_EPS)
    u_r_unit = m_rad_scalar / tot_m_norm
    r_scale = (r_cur / target_radius) - (float(radial_lr_ratio) * theta) * u_r_unit
    r_scale = r_scale.clamp(DEFAULT_VOCAB_RADIAL_MIN, DEFAULT_VOCAB_RADIAL_MAX)
    r_scale = r_scale / r_scale.square().mean().sqrt().clamp_min(VOCAB_MANIFOLD_EPS)
    m_rad_next = (m_rad_scalar / target_radius) * w_unit_next
    m_stored = _project_to_sphere_tangent(m_tan, w_unit_next) + m_rad_next
    return w_unit_next * r_scale, m_stored


def _step_full_single_param(
    p: torch.Tensor,
    state: dict[str, Any],
    lr: float,
    beta1: float,
    tik_lam: float,
    ns_steps: int,
    idx_act: torch.Tensor | None,
    radial_lr_ratio: float = 0.0,
) -> None:
    """Execute one dense-gradient vocabulary matrix step (sphere tangent or polar radial+tangent, `<= 8` args)."""
    if p.grad is None or p.ndim != 2:
        return
    g_f = p.grad.float()
    if not torch.isfinite(g_f).all():
        return
    if "momentum_buffer" not in state:
        state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros(p.size(0), 1, device=p.device, dtype=torch.float32)
    mom_buf = state["momentum_buffer"]
    exp_avg_sq = state["exp_avg_sq"]

    w_f = p.float()
    target_radius = math.sqrt(float(w_f.size(-1)))
    r_cur = w_f.norm(dim=-1, keepdim=True).clamp_min(VOCAB_MANIFOLD_EPS)
    w_unit = w_f * (target_radius / r_cur) if radial_lr_ratio > 0.0 else w_f

    g_tan = _project_to_sphere_tangent(g_f, w_unit)
    g_rad = g_f - g_tan
    m_prev_tan = _project_to_sphere_tangent(mom_buf, w_unit)
    m_prev_rad = mom_buf - m_prev_tan
    m_tan = _project_to_sphere_tangent(beta1 * m_prev_tan + (1.0 - beta1) * g_tan, w_unit)
    m_rad = beta1 * m_prev_rad + (1.0 - beta1) * g_rad
    exp_avg_sq.mul_(DEFAULT_VOCAB_MOMENTUM_BETA2).add_(
        g_tan.square().sum(dim=-1, keepdim=True), alpha=1.0 - DEFAULT_VOCAB_MOMENTUM_BETA2
    )

    if float(torch.linalg.matrix_norm(m_tan, ord="fro").item()) < VOCAB_MANIFOLD_EPS:
        return
    inv_sqrt_a = _compute_tikhonov_gram_inv_sqrt(m_tan, tik_lam=tik_lam, ns_steps=ns_steps)
    if m_tan.is_cuda and m_tan.size(0) >= 1024 and torch.cuda.is_bf16_supported():
        u_all = torch.matmul(m_tan.bfloat16(), inv_sqrt_a.bfloat16()).float()
    else:
        u_all = torch.matmul(m_tan, inv_sqrt_a)
    u_tan = _project_to_sphere_tangent(u_all, w_unit)

    u_norm = u_tan.norm(dim=-1, keepdim=True)
    snr = m_tan.norm(dim=-1, keepdim=True) / (exp_avg_sq.sqrt() + VOCAB_MANIFOLD_EPS)
    rms_u = u_norm.square().mean().sqrt().clamp_min(VOCAB_MANIFOLD_EPS)
    rel_mag = (u_norm / rms_u).clamp_max(2.5)
    if idx_act is not None:
        row_lr = torch.full((w_f.size(0), 1), lr * 0.25, device=w_f.device, dtype=w_f.dtype)
        row_lr.index_fill_(0, idx_act.to(w_f.device), lr)
        theta = row_lr * rel_mag * (snr / (snr + DEFAULT_SPHERE_SNR_TRUST_TAU))
    else:
        theta = lr * rel_mag * (snr / (snr + DEFAULT_SPHERE_SNR_TRUST_TAU))

    u_dir = (target_radius / u_norm.clamp_min(VOCAB_MANIFOLD_EPS)) * u_tan
    w_raw = torch.cos(theta) * w_unit - torch.sin(theta) * u_dir
    w_unit_next = (target_radius / w_raw.norm(dim=-1, keepdim=True).clamp_min(VOCAB_MANIFOLD_EPS)) * w_raw

    if radial_lr_ratio > 0.0:
        w_next, m_stored = _apply_polar_radial_vocab_step(
            w_unit, w_unit_next, r_cur, m_tan, m_rad, theta, radial_lr_ratio
        )
        mom_buf.copy_(m_stored)
        p.copy_(w_next.to(p.dtype))
    else:
        mom_buf.copy_(_project_to_sphere_tangent(m_tan, w_unit_next))
        p.copy_(w_unit_next.to(p.dtype))


def _compute_batched_tikhonov_gram_inv_sqrt_3d(
    m_tan_bmd: torch.Tensor,
    tik_lam: float,
    ns_steps: int,
) -> torch.Tensor:
    """Compute batched `(M, d_h, d_h)` Tikhonov Gram inverse square roots $A_m^{-1/2}$ for 3D wavepacket embeddings."""
    m_mnd = m_tan_bmd.permute(1, 0, 2)
    m_modes, _, d_h = m_mnd.shape
    eye = torch.eye(d_h, device=m_mnd.device, dtype=torch.float32).unsqueeze(0)
    raw_gram = torch.bmm(m_mnd.transpose(1, 2), m_mnd)
    mean_eig = (
        raw_gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1) / float(max(1, d_h))
    ).clamp_min(VOCAB_MANIFOLD_EPS)
    norm_gram = raw_gram / mean_eig
    mix = DEFAULT_GRAM_WHITENING_MIX
    gram_a = (1.0 - mix + tik_lam) * eye + mix * norm_gram
    fro_a = torch.linalg.matrix_norm(gram_a, ord="fro", dim=(-2, -1), keepdim=True).clamp_min(VOCAB_MANIFOLD_EPS)
    y_mat = gram_a / fro_a
    z_mat = eye.expand(m_modes, d_h, d_h).clone()
    for _ in range(max(6, int(ns_steps) + 1)):
        t_mat = 0.5 * (3.0 * eye - torch.bmm(z_mat, y_mat))
        y_mat = torch.bmm(y_mat, t_mat)
        z_mat = torch.bmm(t_mat, z_mat)
    return z_mat * torch.rsqrt(fro_a)


def _step_active_wavepacket_3d(
    p: torch.Tensor,
    state: dict[str, Any],
    idx_dev: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    tik_lam: float,
    ns_steps: int,
) -> None:
    """Update active 3D `(V, M, d_h)` wavepacket rows on $(\\sqrt{d_h}\\,\\mathbb{S}^{d_h-1})^M$ (`8` args)."""
    if p.grad is None:
        return
    g_act = p.grad.index_select(0, idx_dev).float()
    if not torch.isfinite(g_act).all():
        return
    if "momentum_buffer" not in state:
        state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros(p.size(0), p.size(1), 1, device=p.device, dtype=torch.float32)
    mom_buf = state["momentum_buffer"]
    exp_avg_sq = state["exp_avg_sq"]

    w_act = p.index_select(0, idx_dev).float()
    target_radius = math.sqrt(float(w_act.size(-1)))
    g_tan = _project_to_sphere_tangent(g_act, w_act)
    g_norm_sq = g_tan.square().sum(dim=-1, keepdim=True)
    m_prev = mom_buf.index_select(0, idx_dev)
    v_prev = exp_avg_sq.index_select(0, idx_dev)

    m_new = beta1 * m_prev + (1.0 - beta1) * g_tan
    v_new = beta2 * v_prev + (1.0 - beta2) * g_norm_sq
    m_tan = _project_to_sphere_tangent(m_new, w_act)
    if float(m_tan.square().sum().sqrt().item()) < VOCAB_MANIFOLD_EPS:
        return

    inv_sqrt_a = _compute_batched_tikhonov_gram_inv_sqrt_3d(m_tan, tik_lam=tik_lam, ns_steps=ns_steps)
    u_act = torch.bmm(m_tan.permute(1, 0, 2), inv_sqrt_a).permute(1, 0, 2)
    u_tan = _project_to_sphere_tangent(u_act, w_act)

    u_norm = u_tan.norm(dim=-1, keepdim=True)
    snr = m_tan.norm(dim=-1, keepdim=True) / (v_new.sqrt() + VOCAB_MANIFOLD_EPS)
    rms_u = u_norm.square().mean().sqrt().clamp_min(VOCAB_MANIFOLD_EPS)
    rel_mag = (u_norm / rms_u).clamp_max(2.5)
    theta = lr * rel_mag * (snr / (snr + DEFAULT_SPHERE_SNR_TRUST_TAU))

    u_dir = (target_radius / u_norm.clamp_min(VOCAB_MANIFOLD_EPS)) * u_tan
    w_raw = torch.cos(theta) * w_act - torch.sin(theta) * u_dir
    w_next = (target_radius / w_raw.norm(dim=-1, keepdim=True).clamp_min(VOCAB_MANIFOLD_EPS)) * w_raw

    mom_buf.index_copy_(0, idx_dev, _project_to_sphere_tangent(m_tan, w_next))
    exp_avg_sq.index_copy_(0, idx_dev, v_new)
    p.index_copy_(0, idx_dev, w_next.to(p.dtype))


class BatchSubspaceTikhonovGramMuon(torch.optim.Optimizer):
    """Active-token `(Q, N_act, A)` Tikhonov-damped Gram Muon on the unit-RMS sphere $\\sqrt{d}\\,\\mathbb{S}^{d-1}$."""

    def __init__(
        self,
        params: Any,
        lr: float = DEFAULT_BATCH_GRAM_MUON_LR,
        beta: float = DEFAULT_VOCAB_MOMENTUM_BETA1,
        tikhonov_lambda: float = DEFAULT_TIKHONOV_GRAM_LAMBDA,
        ns_steps: int = DEFAULT_COUPLED_NS_STEPS,
        radial_lr_ratio: float = 0.0,
    ) -> None:
        self.radial_lr_ratio = float(radial_lr_ratio)
        defaults = {
            "lr": float(lr),
            "beta": float(beta),
            "tikhonov_lambda": float(tikhonov_lambda),
            "ns_steps": int(ns_steps),
        }
        super().__init__(params, defaults)

    def set_momentum(self, beta: float) -> None:
        """Update active sphere tangent momentum coefficient `beta` across all parameter groups."""
        for group in self.param_groups:
            group["beta"] = float(beta)

    @torch.no_grad()
    def step_active(
        self,
        active_idx: torch.Tensor,
        lr_scale: float = 1.0,
    ) -> None:
        """Update strictly the active vocabulary rows indexed by `active_idx`."""
        valid_mask = active_idx >= 0
        idx_act = torch.unique(active_idx[valid_mask].reshape(-1)).long()
        if idx_act.numel() == 0:
            return

        for group in self.param_groups:
            lr = float(group["lr"]) * float(lr_scale)
            beta1 = float(group["beta"])
            beta2 = DEFAULT_VOCAB_MOMENTUM_BETA2
            tik_lam = float(group["tikhonov_lambda"])
            ns_steps = int(group["ns_steps"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                idx_dev = idx_act.to(p.device)
                if p.ndim == 3:
                    _step_active_wavepacket_3d(
                        p, self.state[p], idx_dev, lr, beta1, beta2, tik_lam, ns_steps
                    )
                    continue
                if p.ndim != 2:
                    continue
                g_act = p.grad.index_select(0, idx_dev).float()
                if not torch.isfinite(g_act).all():
                    continue

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros(p.size(0), 1, device=p.device, dtype=torch.float32)
                mom_buf = state["momentum_buffer"]
                exp_avg_sq = state["exp_avg_sq"]

                w_act = p.index_select(0, idx_dev).float()
                d_dim = w_act.size(-1)
                target_radius = math.sqrt(float(d_dim))

                g_tan = _project_to_sphere_tangent(g_act, w_act)
                g_norm_sq = g_tan.square().sum(dim=-1, keepdim=True)
                m_prev = mom_buf.index_select(0, idx_dev)
                v_prev = exp_avg_sq.index_select(0, idx_dev)

                m_new = beta1 * m_prev + (1.0 - beta1) * g_tan
                v_new = beta2 * v_prev + (1.0 - beta2) * g_norm_sq
                m_tan = _project_to_sphere_tangent(m_new, w_act)

                fro_norm = torch.linalg.matrix_norm(m_tan, ord="fro")
                if float(fro_norm.item()) < VOCAB_MANIFOLD_EPS:
                    continue

                inv_sqrt_a = _compute_tikhonov_gram_inv_sqrt(m_tan, tik_lam=tik_lam, ns_steps=ns_steps)
                u_act = torch.matmul(m_tan, inv_sqrt_a)
                u_tan = _project_to_sphere_tangent(u_act, w_act)

                u_norm = u_tan.norm(dim=-1, keepdim=True)
                snr = m_tan.norm(dim=-1, keepdim=True) / (v_new.sqrt() + VOCAB_MANIFOLD_EPS)
                rms_u = u_norm.square().mean().sqrt().clamp_min(VOCAB_MANIFOLD_EPS)
                rel_mag = (u_norm / rms_u).clamp_max(2.5)
                theta = lr * rel_mag * (snr / (snr + DEFAULT_SPHERE_SNR_TRUST_TAU))

                u_dir = (target_radius / u_norm.clamp_min(VOCAB_MANIFOLD_EPS)) * u_tan
                w_raw = torch.cos(theta) * w_act - torch.sin(theta) * u_dir
                w_next = (target_radius / w_raw.norm(dim=-1, keepdim=True).clamp_min(VOCAB_MANIFOLD_EPS)) * w_raw

                m_transported = _project_to_sphere_tangent(m_tan, w_next)
                mom_buf.index_copy_(0, idx_dev, m_transported)
                exp_avg_sq.index_copy_(0, idx_dev, v_new)
                p.index_copy_(0, idx_dev, w_next.to(p.dtype))

    @torch.no_grad()
    def step_full(
        self,
        lr_scale: float = 1.0,
        active_idx: torch.Tensor | None = None,
    ) -> None:
        """Update all rows of dense-gradient vocabulary readout matrices in a single unified tangent pass."""
        idx_act: torch.Tensor | None = None
        if active_idx is not None:
            valid_mask = active_idx >= 0
            u_idx = torch.unique(active_idx[valid_mask].reshape(-1)).long()
            if u_idx.numel() > 0:
                idx_act = u_idx

        for group in self.param_groups:
            lr = float(group["lr"]) * float(lr_scale)
            beta1 = float(group["beta"])
            tik_lam = float(group["tikhonov_lambda"])
            ns_steps = int(group["ns_steps"])

            for p in group["params"]:
                _step_full_single_param(
                    p, self.state[p], lr, beta1, tik_lam, ns_steps, idx_act, self.radial_lr_ratio
                )

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.step_full(lr_scale=1.0)
        return loss


class RiemannianSphereTangentSNROptimizer(torch.optim.Optimizer):
    """Oblique sphere tangent momentum with parallel transport and SNR trust-region geodesic retraction."""

    def __init__(
        self,
        params: Any,
        lr: float = DEFAULT_SPHERE_GEODESIC_LR,
        beta1: float = DEFAULT_VOCAB_MOMENTUM_BETA1,
        beta2: float = DEFAULT_VOCAB_MOMENTUM_BETA2,
        snr_tau: float = DEFAULT_SPHERE_SNR_TRUST_TAU,
    ) -> None:
        defaults = {
            "lr": float(lr),
            "beta1": float(beta1),
            "beta2": float(beta2),
            "snr_tau": float(snr_tau),
        }
        super().__init__(params, defaults)

    def set_momentum(self, beta: float) -> None:
        """Update active sphere tangent momentum coefficient `beta` across all parameter groups."""
        for group in self.param_groups:
            group["beta"] = float(beta)

    @torch.no_grad()
    def step_active(
        self,
        active_idx: torch.Tensor,
        lr_scale: float = 1.0,
    ) -> None:
        """Update strictly the active vocabulary rows indexed by `active_idx` via SNR-gated great-circle steps."""
        valid_mask = active_idx >= 0
        idx_act = torch.unique(active_idx[valid_mask].reshape(-1)).long()
        if idx_act.numel() == 0:
            return

        for group in self.param_groups:
            lr = float(group["lr"]) * float(lr_scale)
            beta1 = float(group["beta1"])
            beta2 = float(group["beta2"])
            snr_tau = float(group["snr_tau"])

            for p in group["params"]:
                if p.grad is None or p.ndim not in (2, 3):
                    continue
                idx_dev = idx_act.to(p.device)
                g_act = p.grad.index_select(0, idx_dev).float()
                if not torch.isfinite(g_act).all():
                    continue

                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros(*p.shape[:-1], 1, device=p.device, dtype=torch.float32)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                w_act = p.index_select(0, idx_dev).float()
                d_dim = w_act.size(-1)
                target_radius = math.sqrt(float(d_dim))

                g_tan = _project_to_sphere_tangent(g_act, w_act)
                g_norm_sq = g_tan.square().sum(dim=-1, keepdim=True)

                m_prev = exp_avg.index_select(0, idx_dev)
                v_prev = exp_avg_sq.index_select(0, idx_dev)

                m_raw = beta1 * m_prev + (1.0 - beta1) * g_tan
                v_new = beta2 * v_prev + (1.0 - beta2) * g_norm_sq
                m_tan = _project_to_sphere_tangent(m_raw, w_act)

                m_norm = m_tan.norm(dim=-1, keepdim=True)
                rms_m_norm = m_norm.square().mean().sqrt().clamp_min(VOCAB_MANIFOLD_EPS)
                rel_mag = (m_norm / rms_m_norm).clamp_max(2.5)
                snr = m_norm / (v_new.sqrt() + VOCAB_MANIFOLD_EPS)
                theta = lr * rel_mag * (snr / (snr + snr_tau))

                m_dir = (target_radius / m_norm.clamp_min(VOCAB_MANIFOLD_EPS)) * m_tan
                w_raw = torch.cos(theta) * w_act - torch.sin(theta) * m_dir
                w_next = (target_radius / w_raw.norm(dim=-1, keepdim=True).clamp_min(VOCAB_MANIFOLD_EPS)) * w_raw

                m_transported = _project_to_sphere_tangent(m_tan, w_next)
                exp_avg.index_copy_(0, idx_dev, m_transported)
                exp_avg_sq.index_copy_(0, idx_dev, v_new)
                p.index_copy_(0, idx_dev, w_next.to(p.dtype))

    @torch.no_grad()
    def step_full(
        self,
        lr_scale: float = 1.0,
        active_idx: torch.Tensor | None = None,
    ) -> None:
        """Update all rows of dense-gradient vocabulary readout matrices via SNR-gated great-circle steps."""
        idx_act: torch.Tensor | None = None
        if active_idx is not None:
            valid_mask = active_idx >= 0
            u_idx = torch.unique(active_idx[valid_mask].reshape(-1)).long()
            if u_idx.numel() > 0:
                idx_act = u_idx

        for group in self.param_groups:
            lr = float(group["lr"]) * float(lr_scale)
            beta1 = float(group["beta1"])
            beta2 = float(group["beta2"])
            snr_tau = float(group["snr_tau"])

            for p in group["params"]:
                if p.grad is None or p.ndim != 2:
                    continue
                g_f = p.grad.float()
                if not torch.isfinite(g_f).all():
                    continue

                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros(p.size(0), 1, device=p.device, dtype=torch.float32)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                w_f = p.float()
                d_dim = w_f.size(-1)
                target_radius = math.sqrt(float(d_dim))

                g_tan = _project_to_sphere_tangent(g_f, w_f)
                g_norm_sq = g_tan.square().sum(dim=-1, keepdim=True)
                exp_avg.mul_(beta1).add_(g_tan, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).add_(g_norm_sq, alpha=1.0 - beta2)
                m_tan = _project_to_sphere_tangent(exp_avg, w_f)

                m_norm = m_tan.norm(dim=-1, keepdim=True)
                rms_m_norm = m_norm.square().mean().sqrt().clamp_min(VOCAB_MANIFOLD_EPS)
                rel_mag = (m_norm / rms_m_norm).clamp_max(2.5)
                snr = m_norm / (exp_avg_sq.sqrt() + VOCAB_MANIFOLD_EPS)
                if idx_act is not None:
                    row_lr = torch.full((w_f.size(0), 1), lr * 0.25, device=w_f.device, dtype=w_f.dtype)
                    row_lr.index_fill_(0, idx_act.to(w_f.device), lr)
                    theta = row_lr * rel_mag * (snr / (snr + snr_tau))
                else:
                    theta = lr * rel_mag * (snr / (snr + snr_tau))

                m_dir = (target_radius / m_norm.clamp_min(VOCAB_MANIFOLD_EPS)) * m_tan
                w_raw = torch.cos(theta) * w_f - torch.sin(theta) * m_dir
                w_next = (target_radius / w_raw.norm(dim=-1, keepdim=True).clamp_min(VOCAB_MANIFOLD_EPS)) * w_raw

                exp_avg.copy_(_project_to_sphere_tangent(m_tan, w_next))
                p.copy_(w_next.to(p.dtype))

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.step_full(lr_scale=1.0)
        return loss
