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
DEFAULT_BALL_TRUST_TAU: float = 0.18
DEFAULT_BALL_MAX_RMS: float = 1.50
DEFAULT_COMPLEX_SPHERE_MAX_ANGLE: float = 0.20
DEFAULT_POSITIVE_GEODESIC_MAX_STEP: float = 0.22
DEFAULT_BOUNDED_TANGENT_MAX_STEP: float = 0.12
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
    m_row_norm = m_tan.norm(dim=-1, keepdim=True)
    snr = m_row_norm / (exp_avg_sq.sqrt() + VOCAB_MANIFOLD_EPS)
    rms_u = u_norm.square().mean().sqrt().clamp_min(VOCAB_MANIFOLD_EPS)
    rel_mag = (u_norm / rms_u).clamp_max(2.5)
    g_tau = float(state.get("grad_norm_gate_tau", 0.0))
    mag_gate: torch.Tensor | float = (
        (m_row_norm.square() / (m_row_norm.square() + g_tau * g_tau)).sqrt() if g_tau > 0.0 else 1.0
    )
    if idx_act is not None:
        row_lr = torch.full((w_f.size(0), 1), lr * 0.25, device=w_f.device, dtype=w_f.dtype)
        row_lr.index_fill_(0, idx_act.to(w_f.device), lr)
        theta = row_lr * rel_mag * (snr / (snr + DEFAULT_SPHERE_SNR_TRUST_TAU)) * mag_gate
    else:
        theta = lr * rel_mag * (snr / (snr + DEFAULT_SPHERE_SNR_TRUST_TAU)) * mag_gate

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
    """Update active 3D `(V, M, d_h)` or 4D complex `(V, M_c, 2, d_h)` wavepacket rows on the sphere (`8` args)."""
    if p.grad is None:
        return
    is_complex_pair = bool(getattr(p, "_complex_wavepacket_pair", False)) or (p.ndim == 4)
    v_sz = p.size(0)
    m_eff = (p.size(1) // 2) if (p.ndim == 3 and is_complex_pair) else p.size(1)
    d_eff = (2 * p.size(-1)) if is_complex_pair else p.size(-1)

    g_act = p.grad.index_select(0, idx_dev).float().reshape(-1, m_eff, d_eff)
    if not torch.isfinite(g_act).all():
        return
    if "momentum_buffer" not in state:
        state["momentum_buffer"] = torch.zeros((v_sz, m_eff, d_eff), device=p.device, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros((v_sz, m_eff, 1), device=p.device, dtype=torch.float32)
    mom_buf = state["momentum_buffer"]
    exp_avg_sq = state["exp_avg_sq"]

    w_act = p.index_select(0, idx_dev).float().reshape(-1, m_eff, d_eff)
    target_radius = math.sqrt(float(d_eff))
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
    slice_mult = state.get("slice_lr_mult")
    if slice_mult is None:
        slice_mult = getattr(p, "_slice_lr_mult", None)
    if slice_mult is not None:
        theta = theta * slice_mult.to(device=theta.device, dtype=theta.dtype).view(1, -1, 1)

    u_dir = (target_radius / u_norm.clamp_min(VOCAB_MANIFOLD_EPS)) * u_tan
    w_raw = torch.cos(theta) * w_act - torch.sin(theta) * u_dir
    w_next = (target_radius / w_raw.norm(dim=-1, keepdim=True).clamp_min(VOCAB_MANIFOLD_EPS)) * w_raw

    mom_buf.index_copy_(0, idx_dev, _project_to_sphere_tangent(m_tan, w_next))
    exp_avg_sq.index_copy_(0, idx_dev, v_new)
    p.index_copy_(0, idx_dev, w_next.reshape(idx_dev.numel(), *p.shape[1:]).to(p.dtype))


def _step_active_zero_init_ball(
    p: torch.Tensor,
    state: dict[str, Any],
    lr: float,
    beta1: float,
    beta2: float,
    tik_lam: float,
    ns_steps: int,
) -> None:
    """Update active rows of zero-init N-gram/register tables via `d x d` Tikhonov Gram whitening inside the unit-RMS ball (`<= 8` args)."""
    if p.grad is None or p.ndim != 2:
        return
    g_row_sq = p.grad.float().square().sum(dim=-1)
    idx_nz = torch.nonzero(g_row_sq > 1e-18, as_tuple=False).reshape(-1)
    if idx_nz.numel() == 0:
        return
    g_act = p.grad.index_select(0, idx_nz).float()
    if not torch.isfinite(g_act).all():
        return

    if "momentum_buffer" not in state:
        state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        state["step_count"] = torch.zeros(p.size(0), 1, device=p.device, dtype=torch.float32)
    mom_buf = state["momentum_buffer"]
    exp_avg_sq = state["exp_avg_sq"]
    step_buf = state["step_count"]

    m_prev = mom_buf.index_select(0, idx_nz)
    v_prev = exp_avg_sq.index_select(0, idx_nz)
    s_new = step_buf.index_select(0, idx_nz) + 1.0

    m_new = beta1 * m_prev + (1.0 - beta1) * g_act
    v_new = beta2 * v_prev + (1.0 - beta2) * g_act.square()
    if float(torch.linalg.matrix_norm(m_new, ord="fro").item()) < VOCAB_MANIFOLD_EPS:
        return

    bc1 = (1.0 - (beta1 ** s_new)).clamp_min(1e-3) if beta1 > 0.0 else torch.ones_like(s_new)
    bc2 = (1.0 - (beta2 ** s_new)).clamp_min(1e-3)
    u_coord = (m_new / bc1) / ((v_new / bc2).sqrt().add_(1e-8))

    inv_sqrt_a = _compute_tikhonov_gram_inv_sqrt(m_new, tik_lam=tik_lam, ns_steps=ns_steps)
    u_gram = torch.matmul(m_new, inv_sqrt_a)
    u_gram_unit = u_gram / u_gram.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(VOCAB_MANIFOLD_EPS)
    u_comb = 0.85 * u_coord + 0.15 * u_gram_unit

    delta = -DEFAULT_BALL_TRUST_TAU * torch.tanh((lr * u_comb) / DEFAULT_BALL_TRUST_TAU)
    w_act = p.index_select(0, idx_nz).float()
    w_raw = w_act + delta
    w_rms = w_raw.square().mean(dim=-1, keepdim=True).sqrt()
    w_next = w_raw * (DEFAULT_BALL_MAX_RMS / w_rms.clamp_min(DEFAULT_BALL_MAX_RMS))

    mom_buf.index_copy_(0, idx_nz, m_new)
    exp_avg_sq.index_copy_(0, idx_nz, v_new)
    step_buf.index_copy_(0, idx_nz, s_new)
    p.index_copy_(0, idx_nz, w_next.to(p.dtype))


class BatchSubspaceTikhonovGramMuon(torch.optim.Optimizer):
    """Active-token `(Q, N_act, A)` Tikhonov-damped Gram Muon on the unit-RMS sphere $\\sqrt{d}\\,\\mathbb{S}^{d-1}$ or soft RMS ball."""

    def __init__(
        self,
        params: Any,
        lr: float = DEFAULT_BATCH_GRAM_MUON_LR,
        beta: float = DEFAULT_VOCAB_MOMENTUM_BETA1,
        tikhonov_lambda: float = DEFAULT_TIKHONOV_GRAM_LAMBDA,
        ns_steps: int = DEFAULT_COUPLED_NS_STEPS,
        radial_lr_ratio: float = 0.0,
        zero_init_ball: bool = False,
        grad_norm_gate_tau: float = 0.0,
    ) -> None:
        self.radial_lr_ratio = float(radial_lr_ratio)
        self.grad_norm_gate_tau = float(grad_norm_gate_tau)
        defaults = {
            "lr": float(lr),
            "beta": float(beta),
            "tikhonov_lambda": float(tikhonov_lambda),
            "ns_steps": int(ns_steps),
            "zero_init_ball": bool(zero_init_ball),
            "grad_norm_gate_tau": float(grad_norm_gate_tau),
        }
        super().__init__(params, defaults)

    def set_momentum(self, beta: float) -> None:
        """Update active sphere tangent momentum coefficient `beta` across all parameter groups."""
        for group in self.param_groups:
            group["beta"] = float(beta)

    def configure_packet_lr_scales(
        self,
        param: torch.Tensor,
        m_sem: int,
        id_lr_scale: float = 0.25,
    ) -> None:
        """Configure per-slice geodesic learning-rate multipliers (`1.0` on `0..m_sem-1`, `id_lr_scale` on `m_sem..M_c-1`)."""
        m_c = int(param.size(1))
        m_sem_eff = max(0, min(int(m_sem), m_c))
        mult = torch.ones(m_c, device=param.device, dtype=torch.float32)
        if m_sem_eff < m_c:
            mult[m_sem_eff:] = float(id_lr_scale)
        self.state[param]["slice_lr_mult"] = mult
        setattr(param, "_slice_lr_mult", mult)

    @torch.no_grad()
    def step_active(
        self,
        active_idx: torch.Tensor,
        lr_scale: float = 1.0,
    ) -> None:
        """Update strictly the active vocabulary or N-gram bucket rows."""
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
            is_ball = bool(group.get("zero_init_ball", False))
            g_tau = float(group.get("grad_norm_gate_tau", self.grad_norm_gate_tau))

            for p in group["params"]:
                if p.grad is None:
                    continue
                if is_ball:
                    _step_active_zero_init_ball(p, self.state[p], lr, beta1, beta2, tik_lam, ns_steps)
                    continue
                idx_dev = idx_act.to(p.device)
                if p.ndim in (3, 4):
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
                m_row_norm = m_tan.norm(dim=-1, keepdim=True)
                snr = m_row_norm / (v_new.sqrt() + VOCAB_MANIFOLD_EPS)
                rms_u = u_norm.square().mean().sqrt().clamp_min(VOCAB_MANIFOLD_EPS)
                rel_mag = (u_norm / rms_u).clamp_max(2.5)
                mag_gate: torch.Tensor | float = (
                    (m_row_norm.square() / (m_row_norm.square() + g_tau * g_tau)).sqrt() if g_tau > 0.0 else 1.0
                )
                theta = lr * rel_mag * (snr / (snr + DEFAULT_SPHERE_SNR_TRUST_TAU)) * mag_gate

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
            is_ball = bool(group.get("zero_init_ball", False))
            g_tau = float(group.get("grad_norm_gate_tau", self.grad_norm_gate_tau))

            for p in group["params"]:
                if is_ball:
                    _step_active_zero_init_ball(
                        p, self.state[p], lr, beta1, DEFAULT_VOCAB_MOMENTUM_BETA2, tik_lam, ns_steps
                    )
                else:
                    self.state[p]["grad_norm_gate_tau"] = g_tau
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


def _step_complex_sphere_pair(
    p_re: torch.Tensor,
    p_im: torch.Tensor,
    state: dict[str, Any],
    lr: float,
    beta1: float,
    beta2: float,
    max_angle: float,
) -> None:
    """Update paired complex kets `(p_re, p_im)` on $\\mathbb{S}^{2 d_h - 1}$ via tangent-projected great-circle rotation (`<= 8` args)."""
    if p_re.grad is None or p_im.grad is None:
        return
    d_h = p_re.size(-1)
    w = torch.cat([p_re.float(), p_im.float()], dim=-1)
    g = torch.cat([p_re.grad.float(), p_im.grad.float()], dim=-1)
    if not torch.isfinite(g).all():
        return

    target_r = w.norm(dim=-1, keepdim=True).clamp_min(VOCAB_MANIFOLD_EPS)
    g_tan = _project_to_sphere_tangent(g, w)
    g_sq = g_tan.square().sum(dim=-1, keepdim=True)

    if "exp_avg" not in state:
        state["exp_avg"] = torch.zeros_like(w, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros(*w.shape[:-1], 1, device=w.device, dtype=torch.float32)
        state["step"] = 0
    state["step"] += 1
    t = int(state["step"])
    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]

    exp_avg.mul_(beta1).add_(g_tan, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).add_(g_sq, alpha=1.0 - beta2)
    m_tan = _project_to_sphere_tangent(exp_avg, w)

    bc1 = max(1e-3, 1.0 - (beta1 ** t)) if beta1 > 0.0 else 1.0
    bc2 = max(1e-3, 1.0 - (beta2 ** t))
    m_norm = m_tan.norm(dim=-1, keepdim=True)
    snr = ((m_norm / bc1) / ((exp_avg_sq / bc2).sqrt() + VOCAB_MANIFOLD_EPS)).clamp_max(2.0)
    theta = max_angle * torch.tanh((lr * snr) / max_angle)

    u_dir = (target_r / m_norm.clamp_min(VOCAB_MANIFOLD_EPS)) * m_tan
    w_raw = torch.cos(theta) * w - torch.sin(theta) * u_dir
    w_next = (target_r / w_raw.norm(dim=-1, keepdim=True).clamp_min(VOCAB_MANIFOLD_EPS)) * w_raw

    exp_avg.copy_(_project_to_sphere_tangent(m_tan, w_next))
    p_re.copy_(w_next[..., :d_h].to(p_re.dtype))
    p_im.copy_(w_next[..., d_h:].to(p_im.dtype))


def _step_real_sphere_ket(
    p: torch.Tensor,
    state: dict[str, Any],
    lr: float,
    beta1: float,
    beta2: float,
    max_angle: float,
) -> None:
    """Update real-packed sphere ket `p` `(..., d)` on $\\sqrt{d}\\,\\mathbb{S}^{d-1}$ via great-circle geodesic rotation (`<= 8` args)."""
    if p.grad is None:
        return
    g = p.grad.float()
    if not torch.isfinite(g).all():
        return
    w = p.float()
    target_r = math.sqrt(float(w.size(-1)))
    g_tan = _project_to_sphere_tangent(g, w)
    g_sq = g_tan.square().sum(dim=-1, keepdim=True)

    if "exp_avg" not in state:
        state["exp_avg"] = torch.zeros_like(w, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros(*w.shape[:-1], 1, device=w.device, dtype=torch.float32)
        state["step"] = 0
    state["step"] += 1
    t = int(state["step"])
    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]

    exp_avg.mul_(beta1).add_(g_tan, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).add_(g_sq, alpha=1.0 - beta2)
    m_tan = _project_to_sphere_tangent(exp_avg, w)

    bc1 = max(1e-3, 1.0 - (beta1 ** t)) if beta1 > 0.0 else 1.0
    bc2 = max(1e-3, 1.0 - (beta2 ** t))
    m_norm = m_tan.norm(dim=-1, keepdim=True)
    snr = ((m_norm / bc1) / ((exp_avg_sq / bc2).sqrt() + VOCAB_MANIFOLD_EPS)).clamp_max(2.0)
    theta = max_angle * torch.tanh((lr * snr) / max_angle)

    u_dir = (target_r / m_norm.clamp_min(VOCAB_MANIFOLD_EPS)) * m_tan
    w_raw = torch.cos(theta) * w - torch.sin(theta) * u_dir
    w_next = (target_r / w_raw.norm(dim=-1, keepdim=True).clamp_min(VOCAB_MANIFOLD_EPS)) * w_raw

    exp_avg.copy_(_project_to_sphere_tangent(m_tan, w_next))
    p.copy_(w_next.to(p.dtype))


def _step_positive_geodesic_param(
    p: torch.Tensor,
    state: dict[str, Any],
    lr: float,
    beta1: float,
    beta2: float,
    ref_scale: float,
    max_log_step: float,
) -> None:
    """Update positive scale `p > 0` on $(\\mathbb{R}_{>0}, ds = (1/p + 1/\\beta_{\\text{ref}})dp)$ with zero quenching (`<= 8` args)."""
    if p.grad is None:
        return
    g_f = p.grad.float()
    if not torch.isfinite(g_f).all():
        return
    if "exp_avg" not in state:
        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        state["step"] = 0
    state["step"] += 1
    t = int(state["step"])
    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]

    exp_avg.mul_(beta1).add_(g_f, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(g_f, g_f, value=1.0 - beta2)

    bc1 = max(1e-3, 1.0 - (beta1 ** t)) if beta1 > 0.0 else 1.0
    bc2 = max(1e-3, 1.0 - (beta2 ** t))
    u_hat = (exp_avg / bc1) / ((exp_avg_sq / bc2).sqrt().add_(1e-8))

    cold_ramp = 0.50 if t == 1 else 1.0
    delta_xi = -max_log_step * torch.tanh((cold_ramp * lr * u_hat) / max_log_step)
    p_f = p.float().clamp_min(1e-4)
    eff_log_step = delta_xi / (1.0 + p_f / float(max(1e-3, ref_scale)))
    p.copy_((p_f * torch.exp(eff_log_step)).to(p.dtype))


def _step_sparse_vocab_param(
    p: torch.Tensor,
    state: dict[str, Any],
    idx_act: torch.Tensor | None,
    lr: float,
    beta1: float,
    beta2: float,
    max_step: float,
    is_positive: bool,
) -> None:
    """Update active vocabulary coordinates `p[idx_act]` `(V,)` via sparse geodesic/tangent steps (`<= 8` args)."""
    if p.grad is None or p.ndim != 1:
        return
    if idx_act is None or idx_act.numel() == 0:
        return
    idx_dev = idx_act.to(p.device)
    g_act = p.grad.index_select(0, idx_dev).float()
    if not torch.isfinite(g_act).all():
        return

    if "exp_avg" not in state:
        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        state["step_count"] = torch.zeros_like(p, dtype=torch.float32)
    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]
    step_buf = state["step_count"]

    m_new = beta1 * exp_avg.index_select(0, idx_dev) + (1.0 - beta1) * g_act
    v_new = beta2 * exp_avg_sq.index_select(0, idx_dev) + (1.0 - beta2) * g_act.square()
    s_new = step_buf.index_select(0, idx_dev) + 1.0

    bc1 = (1.0 - (beta1 ** s_new)).clamp_min(1e-3) if beta1 > 0.0 else torch.ones_like(s_new)
    bc2 = (1.0 - (beta2 ** s_new)).clamp_min(1e-3)
    u_hat = (m_new / bc1) / ((v_new / bc2).sqrt().add_(1e-8))
    delta = -max_step * torch.tanh((lr * u_hat) / max_step)

    p_act = p.index_select(0, idx_dev).float()
    p_next = p_act.clamp_min(1e-4) * torch.exp(delta / (1.0 + p_act.clamp_min(1e-4))) if is_positive else (p_act + delta)
    exp_avg.index_copy_(0, idx_dev, m_new)
    exp_avg_sq.index_copy_(0, idx_dev, v_new)
    step_buf.index_copy_(0, idx_dev, s_new)
    p.index_copy_(0, idx_dev, p_next.to(p.dtype))


def _step_bounded_tangent_param(
    p: torch.Tensor,
    state: dict[str, Any],
    lr: float,
    beta1: float,
    beta2: float,
    max_step: float,
) -> None:
    """Update signed coupling scalar, phase $\\in \\mathbb{S}^1$, or small 2D gate matrix with bounded tangent step (`<= 8` args)."""
    if p.grad is None:
        return
    g_f = p.grad.float()
    if not torch.isfinite(g_f).all():
        return
    if "exp_avg" not in state:
        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        state["step"] = 0
    state["step"] += 1
    t = int(state["step"])
    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]

    exp_avg.mul_(beta1).add_(g_f, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(g_f, g_f, value=1.0 - beta2)

    bc1 = max(1e-3, 1.0 - (beta1 ** t)) if beta1 > 0.0 else 1.0
    bc2 = max(1e-3, 1.0 - (beta2 ** t))
    u_hat = (exp_avg / bc1) / ((exp_avg_sq / bc2).sqrt().add_(1e-8))
    delta = -max_step * torch.tanh((lr * u_hat) / max_step)
    w_raw = p.float() + delta
    if p.ndim == 2:
        w_rms = w_raw.square().mean(dim=-1, keepdim=True).sqrt()
        w_raw = w_raw * (DEFAULT_BALL_MAX_RMS / w_rms.clamp_min(DEFAULT_BALL_MAX_RMS))
    p.copy_(w_raw.to(p.dtype))


class RiemannianGeodesicScalarAndProbeOptimizer(torch.optim.Optimizer):
    """100% `AdamW`-free Riemannian optimizer for complex $\\mathbb{S}^{31}$ probe/vacuum kets, $\\mathbb{R}_{>0}$ geodesic scales, sparse priors, and bounded gates."""

    def __init__(
        self,
        params: Any,
        lr: float = 0.008,
        beta1: float = 0.80,
        beta2: float = 0.95,
    ) -> None:
        defaults = {
            "lr": float(lr),
            "beta1": float(beta1),
            "beta2": float(beta2),
            "kind": "bounded_tangent",
            "ref_scale": 1.0,
            "max_step": DEFAULT_BOUNDED_TANGENT_MAX_STEP,
        }
        super().__init__(params, defaults)

    def set_momentum(self, beta1: float) -> None:
        """Update first-moment tangent decay `beta1` across all parameter groups."""
        b1 = 0.0 if beta1 <= 0.0 else float(beta1)
        for group in self.param_groups:
            group["beta1"] = b1

    @torch.no_grad()
    def step_active(
        self,
        active_idx: torch.Tensor | None = None,
        lr_scale: float = 1.0,
    ) -> None:
        """Execute one coordinated geodesic/tangent step across all parameter groups."""
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
            kind = str(group.get("kind", "bounded_tangent"))
            ref_scale = float(group.get("ref_scale", 1.0))
            max_step = float(group.get("max_step", DEFAULT_BOUNDED_TANGENT_MAX_STEP))
            params = group["params"]

            if kind == "complex_sphere_pair":
                for i in range(0, len(params) - 1, 2):
                    _step_complex_sphere_pair(
                        params[i], params[i + 1], self.state[params[i]], lr, beta1, beta2, max_step
                    )
            elif kind == "real_sphere_ket":
                for p in params:
                    _step_real_sphere_ket(p, self.state[p], lr, beta1, beta2, max_step)
            elif kind == "positive_geodesic":
                for p in params:
                    _step_positive_geodesic_param(p, self.state[p], lr, beta1, beta2, ref_scale, max_step)
            elif kind in ("sparse_vocab_positive", "sparse_vocab_bias"):
                is_pos = kind == "sparse_vocab_positive"
                for p in params:
                    _step_sparse_vocab_param(p, self.state[p], idx_act, lr, beta1, beta2, max_step, is_pos)
            else:
                for p in params:
                    _step_bounded_tangent_param(p, self.state[p], lr, beta1, beta2, max_step)

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.step_active(active_idx=None, lr_scale=1.0)
        return loss

