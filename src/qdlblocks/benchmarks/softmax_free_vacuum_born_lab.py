r"""100% `softmax`-Free Vacuum-Sharpened Coincidence Born Readout & Trunk No-Click Born Attention Laboratory (`EXPERIMENT_PLAN_080.md` / `Wave 60`).

Mathematical & Architectural Motivation
---------------------------------------
1. **Why Ripping Out `softmax` Requires the Orthogonal Vacuum Subspace $\mathcal{H}_{\text{vac}}$**:
   - On a standard unit sphere $\mathbb{S}^{d-1}$ ($d_{\text{vac}} = 0$), every fiber vector ($h_{\text{sem}} \in \mathbb{S}^{447}$,
     $h_{\text{id}, r} \in \mathbb{S}^{63}$) is forced by `RMSNorm` to have fixed full length $\sqrt{d}$. Under a polynomial
     Born rule without `softmax`, raising $(1 + z/(2p))_+^{2p}$ to a high power $2p$ to sharpen confident predictions causes
     uncertain tokens and low-energy excited Cholesky modes ($r \ge 1$) to inject severe false-positive spikes from random
     unit-sphere overlaps ($\pm 0.58$ in `64D`).
   - Embedding the readout fibers into the **Vacuum-Augmented Sphere** $\mathcal{H}_{\text{vocab}} \oplus \mathcal{H}_{\text{vac}}$
     ($d_{\text{vac}} = 4$) where vocabulary detectors $|w_v\rangle$ have zero support ($\langle w_v \mid h^{\text{vac}}\rangle \equiv 0$)
     modulates every vocabulary inner product by the **Vacuum Angle** $\cos\theta_{\text{vac}}(t, r) \in [0, 1]$:
     $$z_{\text{id}, t, r, v} \;=\; \tau_{\text{id}} \cos\theta_{\text{vac}}^{\text{id}}(t, r) \, \frac{\langle \hat{h}_{\text{id}, t, r}^{\text{vocab}} \mid w_{\text{id}, v}\rangle}{64}$$
   - Under the $2p$-boson Fock Coincidence Born amplitude $\Phi_{2p}(z) = \big(1 + \frac{z}{2p}\big)_+^{2p}$:
     - When mode $r$ (or the identity fiber) rotates into $\mathcal{H}_{\text{vac}}$ ($\cos\theta_{\text{vac}} \to 0$),
       $\Phi_{2p}(0) = (1 + 0)_+^{2p} \equiv 1$ across all $V = 50,304$ tokens — smoothly turning off the mode with zero false positives!
     - When mode $r$ rotates into $\mathcal{H}_{\text{vocab}}$ ($\cos\theta_{\text{vac}} \to 1$), the full $2p$-boson polynomial
       sharpness engages, while negatively aligned tokens ($z \le -2p$) hit exact destructive interference $\Phi_{2p}(z) = 0$.

2. **Single-Denominator Factorized Coincidence Born Readout (`_VacuumCoincidenceBornCEFn`)**:
   - Unlike a `softmax` mixture $\sum_r \pi_{t,r} \text{softmax}_v(z_{t,r,v})$ which must evaluate $R$ separate `50,304`-way
     `log_softmax` partition functions per token, quantum Born probability is linear in the density matrix
     $\rho_t = \sum_{r=0}^{R-1} \pi_{t,r} |\psi_{t,r}\rangle\langle\psi_{t,r}|$:
     $$B_{t, v} \;=\; I_{\text{sem}, t, v} \sum_{r=0}^{R-1} \pi_{t, r} \, \Phi_{2p_{\text{id}}}(z_{\text{id}, t, r, v}), \qquad P_{\text{Born}}(v \mid x_{\le t}) \;=\; \frac{B_{t, v}}{\sum_{v'=1}^V B_{t, v'}}$$
   - Combining $h_{\text{sem0}, t} + \bar{\beta}_t h_{\text{sem1}, t}$ before the vocabulary projection eliminates 3 full
     `448 x 50,304` GEMMs per step and requires only a single scalar partition function $Z_t = \sum_v B_{t,v}$ per token.

3. **100% `softmax`-Free Trunk Causal Hermitian-Born Attention with Quantum No-Click Vacuum Key**:
   - Replaces causal `softmax` attention with $2p_{\text{attn}} = 4$ polynomial Born intensities normalized against the
     learned Vacuum Key $|\hat{k}_{\text{vac}, h}\rangle \in \mathbb{C}^{32}$ in the denominator:
     $$A_{t, s}^{(h)} \;=\; \frac{(1 + S_{t, s}^{(h)} / 4)_+^4}{\sum_{s' \le t} (1 + S_{t, s'}^{(h)} / 4)_+^4 + W_{t, \text{vac}}^{(h)} + \varepsilon}$$
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.fused_quantum_wavepacket_attn import (
     _FusedHermitianBornRhoKetAttnFn, _get_compiled_chol_gram_fn, associative_low_rank_cholesky_gram_modulation,
)
from ..layers.impure_density_mirror_block import TRACE_NORM_EPS
from ..layers.quantum_statmech_channels import QuantumMomentumScheduleMode, compute_riemannian_momentum_beta
from ..layers.quantum_svd_stiefel_graft import apply_fubini_study_self_ortho
from ..models.compiled_leaderboard_classical_gpt import apply_native_leaderboard_rotary
from .born_schur_density_lab import DEFAULT_FINEWEB_EVAL_INTERVAL, DEFAULT_FINEWEB_STEPS, DEFAULT_SCHUR_EPS
from .experiment_logger import ExperimentIntentSpec, StandardizedExperimentLogger
from .noclick_vacuum_and_smith_sgd_lab import (
    DEFAULT_INIT_NOCLICK_POS_LOGIT, DEFAULT_INIT_NOCLICK_SINK_BIAS, DEFAULT_NOCLICK_HIGHWAY_SHIELD_STRENGTH,
    _apply_global_optimizer_step, _NoClickVacuumCholeskyWavepacketBlock, compute_noclick_vacuum_sink_scores,
)
from .shallow_longseq_wavepacket_lm_lab import (
    DEFAULT_SHALLOW_EMBED_DIM, DEFAULT_SHALLOW_MLP_HIDDEN, DEFAULT_SHALLOW_NUM_HEADS, DEFAULT_SHALLOW_PKT_COUNT,
    DEFAULT_SHALLOW_PKT_DIM, LongSeqFineWebStream,
)
from .vacuum_subspace_cholesky_lab import DEFAULT_MULTIBRANCH_TOKEN_CHUNK, DEFAULT_VACUUM_SUBSPACE_DIM
from .variable_rank_cholesky_vacuum_lab import (
    DEFAULT_WINNING_CHOLESKY_RANK, VariableRankCholeskySpec, VariableRankVacuumCholeskyGPT, VariableRankVariantMode,
    _build_variable_rank_cholesky_f_flat, build_variable_rank_cholesky_optimizers,
    initialize_and_calibrate_variable_rank_cholesky_model_,
)
from .wavefunction_outer_density_lab import DEFAULT_INIT_SCHUR_GATE_SCALE

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_080.md
# ==============================================================================
DEFAULT_BORN_SEM_DEGREE: int = 16
DEFAULT_BORN_ID_DEGREE: int = 16
DEFAULT_BORN_MODE_DEGREE: int = 4
DEFAULT_BORN_ATTN_DEGREE: int = 4
DEFAULT_BORN_READOUT_EPS: float = 1e-7
DEFAULT_BORN_MAX_COORD: float = 28.0


class SoftmaxFreeBornVariantMode(str, Enum):
    """Strategy enum for 100% `softmax`-Free Vacuum-Sharpened Coincidence Born experiments (`B0..B2`, Rule 3)."""

    B0_SOFTMAX_FREE_BORN_NO_VACUUM_R8 = "b0_softmax_free_born_no_vacuum_r8"
    B1_SOFTMAX_FREE_VACUUM_BORN_R8 = "b1_softmax_free_vacuum_born_r8"
    B2_SOFTMAX_FREE_VACUUM_BORN_R12 = "b2_softmax_free_vacuum_born_r12"


@dataclass(frozen=True)
class SoftmaxFreeVacuumBornSpec:
    r"""Configuration for `SoftmaxFreeVacuumBornGPT` (`<= 8` fields, Rule 2)."""

    variant_mode: SoftmaxFreeBornVariantMode = SoftmaxFreeBornVariantMode.B1_SOFTMAX_FREE_VACUUM_BORN_R8
    num_cholesky_modes: int = DEFAULT_WINNING_CHOLESKY_RANK
    vacuum_dim: int = DEFAULT_VACUUM_SUBSPACE_DIM
    use_noclick_vacuum_sink: bool = True
    use_softmax_free_trunk_attn: bool = True
    sem_born_degree: int = DEFAULT_BORN_SEM_DEGREE
    id_born_degree: int = DEFAULT_BORN_ID_DEGREE
    seed: int = 1337


def _poly_power_even(u_pos: torch.Tensor, degree: int) -> torch.Tensor:
    """Compute `u_pos ** degree` for `degree in {4, 8, 16}` using fast repeated squaring."""
    if degree == 16:
        u2 = u_pos.square()
        u4 = u2.square()
        u8 = u4.square()
        return u8.square()
    if degree == 8:
        u2 = u_pos.square()
        u4 = u2.square()
        return u4.square()
    if degree == 4:
        return u_pos.square().square()
    return u_pos.pow(float(degree))


def compute_softmax_free_mode_weights(
    alpha_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Compute `softmax`-free 4-boson Born mode mixture weights $\pi_{t, r} = \frac{w_{t, r}}{\sum_{r'} w_{t, r'}}$ (`<= 8` args)."""
    u_m = (1.0 + 0.25 * (alpha_flat + 1.5)).clamp_min(0.0)
    w_m = u_m.square().square() + DEFAULT_BORN_READOUT_EPS
    w_sum = w_m.sum(dim=-1, keepdim=True).clamp_min(DEFAULT_BORN_READOUT_EPS)
    pi_m = w_m / w_sum
    return pi_m, u_m, w_sum


class _VacuumCoincidenceBornCEFn(torch.autograd.Function):
    r"""Rule-9 Single-Pass Chunked Autograd Function for 100% `softmax`-Free Vacuum-Sharpened Coincidence Born Readout.

    Evaluates:
      $$B_{t, v} = I_{\text{sem}, t, v} \sum_{r=0}^{R-1} \pi_{t, r} \Phi_{2p_{\text{id}}}(z_{\text{id}, t, r, v}), \qquad
        \mathcal{L}_t = \ln\!\left(\sum_{v=1}^V B_{t, v}\right) - \ln I_{\text{sem}, t, y_t} - \ln M_{\text{id}, t, y_t}$$
    with zero `exp`, zero `softmax`, zero `logsumexp`, and zero `tanh` softcap (`bwd_flops / fwd_flops <= 2.2 <= 3.5`).
    """

    @staticmethod
    def forward(
        ctx: Any,
        h_base_464: torch.Tensor,
        h_sem_branches: torch.Tensor,
        h_id_branches: torch.Tensor,
        branch_controls: torch.Tensor,
        w_aug_528: torch.Tensor,
        targets: torch.Tensor,
        born_degrees: tuple[int, int] = (DEFAULT_BORN_SEM_DEGREE, DEFAULT_BORN_ID_DEGREE),
    ) -> torch.Tensor:
        """Compute single-pass chunked forward Coincidence Born NLL loss without `exp` or `softmax`."""
        bsz, seq_len, _ = h_base_464.shape
        n_total, num_branches, v_sz = bsz * seq_len, h_id_branches.size(-2), w_aug_528.size(0)
        sem_dim, id_dim = h_sem_branches.size(-1), h_id_branches.size(-1)
        car_dim, work_dtype = h_base_464.size(-1) - sem_dim, h_base_464.dtype
        deg_sem, deg_id = int(born_degrees[0]), int(born_degrees[1])
        inv_deg_sem, inv_deg_id = 1.0 / float(deg_sem), 1.0 / float(deg_id)

        hb_flat = h_base_464.reshape(n_total, sem_dim + car_dim)
        hs_flat = h_sem_branches.reshape(n_total, sem_dim)
        hid_flat = h_id_branches.reshape(n_total, num_branches, id_dim)
        ctrl_flat = branch_controls.reshape(n_total, num_branches, 3).float()
        beta_flat, tau_flat, alpha_flat = ctrl_flat[:, :, 0], ctrl_flat[:, :, 1], ctrl_flat[:, :, 2]
        y_flat = targets.reshape(n_total)

        bar_beta = beta_flat[:, 1] if num_branches > 1 else beta_flat[:, 0]
        sem_eff = hb_flat[:, :sem_dim] + bar_beta.to(work_dtype).unsqueeze(1) * hs_flat
        heff_flat = torch.cat([sem_eff, hb_flat[:, sem_dim:]], dim=-1).contiguous()

        w_sem, w_id = w_aug_528[:, :sem_dim], w_aug_528[:, sem_dim : sem_dim + id_dim]
        w_car = w_aug_528[:, sem_dim + id_dim : sem_dim + id_dim + car_dim]
        w_base_t = torch.cat([w_sem, w_car], dim=-1).t().contiguous()
        w_id_t = w_id.t().contiguous()

        pi_flat, _, _ = compute_softmax_free_mode_weights(alpha_flat)
        raw_chunk = (DEFAULT_MULTIBRANCH_TOKEN_CHUNK * 2) // max(1, num_branches)
        chunk_sz = max(128, (raw_chunk // 64) * 64)

        total_nll = torch.zeros((), device=h_base_464.device, dtype=torch.float32)
        n_valid = int((y_flat >= 0).sum().item())
        all_valid = bool(n_valid == n_total)

        for start in range(0, n_total, chunk_sz):
            end = min(start + chunk_sz, n_total)
            c_len, yc = end - start, y_flat[start:end]
            if not all_valid and not bool((yc >= 0).any().item()):
                continue

            z_sem = torch.matmul(heff_flat[start:end], w_base_t).float().clamp(-DEFAULT_BORN_MAX_COORD, DEFAULT_BORN_MAX_COORD)
            u_sem = (1.0 + z_sem * inv_deg_sem).clamp_min_(0.0)
            i_sem = _poly_power_even(u_sem, deg_sem).add_(DEFAULT_BORN_READOUT_EPS)
            del z_sem, u_sem

            u_id = torch.matmul(hid_flat[start:end].reshape(c_len * num_branches, id_dim), w_id_t).float().view(c_len, num_branches, v_sz)
            u_id.mul_(tau_flat[start:end].unsqueeze(-1)).clamp_(-DEFAULT_BORN_MAX_COORD, DEFAULT_BORN_MAX_COORD)
            u_id.mul_(inv_deg_id).add_(1.0).clamp_min_(0.0)
            phi_id = _poly_power_even(u_id, deg_id).add_(DEFAULT_BORN_READOUT_EPS)
            del u_id

            z_tr = torch.bmm(phi_id, i_sem.unsqueeze(-1)).squeeze(-1)
            yc_clamped = yc.clamp_min(0)
            yc_idx = yc_clamped.view(c_len, 1, 1).expand(c_len, num_branches, 1)
            phi_id_y = phi_id.gather(dim=-1, index=yc_idx).squeeze(-1)
            del phi_id

            pi_c = pi_flat[start:end]
            z_tot = (pi_c * z_tr).sum(dim=-1).clamp_min(DEFAULT_BORN_READOUT_EPS)
            m_id_y = (pi_c * phi_id_y).sum(dim=-1).clamp_min(DEFAULT_BORN_READOUT_EPS)
            i_sem_y = i_sem.gather(dim=-1, index=yc_clamped.unsqueeze(1)).squeeze(1).clamp_min(DEFAULT_BORN_READOUT_EPS)
            del i_sem

            token_nll = torch.log(z_tot) - torch.log(i_sem_y) - torch.log(m_id_y)
            total_nll = total_nll + (token_nll.sum() if all_valid else token_nll[yc >= 0].sum())

        ctx.save_for_backward(h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug_528, targets)
        ctx.n_valid, ctx.all_valid = max(1, n_valid), all_valid
        ctx.deg_sem, ctx.deg_id = deg_sem, deg_id
        return total_nll / float(max(1, n_valid))

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """Single-pass chunked backward for 100% `softmax`-free Coincidence Born Readout (Rule 9)."""
        h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug_528, targets = ctx.saved_tensors
        bsz, seq_len, _ = h_base_464.shape
        n_total, num_branches, v_sz = bsz * seq_len, h_id_branches.size(-2), w_aug_528.size(0)
        sem_dim, id_dim = h_sem_branches.size(-1), h_id_branches.size(-1)
        car_dim, work_dtype = h_base_464.size(-1) - sem_dim, h_base_464.dtype
        all_valid = bool(getattr(ctx, "all_valid", False))
        deg_sem, deg_id = int(ctx.deg_sem), int(ctx.deg_id)
        inv_deg_sem, inv_deg_id = 1.0 / float(deg_sem), 1.0 / float(deg_id)

        hb_flat = h_base_464.reshape(n_total, sem_dim + car_dim)
        hs_flat = h_sem_branches.reshape(n_total, sem_dim)
        hid_flat = h_id_branches.reshape(n_total, num_branches, id_dim)
        ctrl_flat = branch_controls.reshape(n_total, num_branches, 3).float()
        beta_flat, tau_flat, alpha_flat = ctrl_flat[:, :, 0], ctrl_flat[:, :, 1], ctrl_flat[:, :, 2]
        y_flat = targets.reshape(n_total)

        beta_col_idx = 1 if num_branches > 1 else 0
        bar_beta = beta_flat[:, beta_col_idx]
        sem_eff = hb_flat[:, :sem_dim] + bar_beta.to(work_dtype).unsqueeze(1) * hs_flat
        heff_flat = torch.cat([sem_eff, hb_flat[:, sem_dim:]], dim=-1).contiguous()

        w_sem, w_id = w_aug_528[:, :sem_dim], w_aug_528[:, sem_dim : sem_dim + id_dim]
        w_car = w_aug_528[:, sem_dim + id_dim : sem_dim + id_dim + car_dim]
        w_base = torch.cat([w_sem, w_car], dim=-1).contiguous()
        w_base_t, w_id_t = w_base.t().contiguous(), w_id.t().contiguous()

        pi_flat, u_m_flat, w_sum_flat = compute_softmax_free_mode_weights(alpha_flat)

        grad_hb = torch.zeros_like(hb_flat)
        grad_hs = torch.zeros_like(hs_flat)
        grad_hid = torch.zeros_like(hid_flat)
        grad_ctrl = torch.zeros_like(ctrl_flat, dtype=branch_controls.dtype)
        grad_w_base = torch.zeros_like(w_base)
        grad_w_id = torch.zeros_like(w_id)

        scale = float(grad_output.item()) / float(ctx.n_valid)
        raw_chunk = (DEFAULT_MULTIBRANCH_TOKEN_CHUNK * 2) // max(1, num_branches)
        chunk_sz = max(128, (raw_chunk // 64) * 64)

        for start in range(0, n_total, chunk_sz):
            end = min(start + chunk_sz, n_total)
            c_len, yc = end - start, y_flat[start:end]
            valid_c = yc >= 0
            if not all_valid and not bool(valid_c.any().item()):
                continue

            heff_c, hs_c, hid_c = heff_flat[start:end], hs_flat[start:end], hid_flat[start:end]
            t_c, pi_c = tau_flat[start:end], pi_flat[start:end]
            u_m_c, w_sum_c = u_m_flat[start:end], w_sum_flat[start:end]
            valid_f = torch.ones((c_len, 1), device=yc.device, dtype=torch.float32) if all_valid else valid_c.float().unsqueeze(-1)
            s_vec = scale * valid_f.squeeze(-1)

            z_sem = torch.matmul(heff_c, w_base_t).float()
            sem_in_bounds = (z_sem > -DEFAULT_BORN_MAX_COORD) & (z_sem < DEFAULT_BORN_MAX_COORD)
            z_sem_c = z_sem.clamp(-DEFAULT_BORN_MAX_COORD, DEFAULT_BORN_MAX_COORD)
            u_sem = (1.0 + z_sem_c * inv_deg_sem).clamp_min_(0.0)
            u_sem_pow = _poly_power_even(u_sem, deg_sem)
            i_sem = u_sem_pow + DEFAULT_BORN_READOUT_EPS
            d_isem_dz = torch.where(
                (u_sem > 1e-6) & sem_in_bounds,
                u_sem_pow / u_sem.clamp_min(1e-6),
                torch.zeros_like(u_sem),
            )
            del z_sem, z_sem_c, u_sem, u_sem_pow

            u_id = torch.matmul(hid_c.reshape(c_len * num_branches, id_dim), w_id_t).float().view(c_len, num_branches, v_sz)
            u_id.mul_(t_c.unsqueeze(-1))
            id_in_bounds = (u_id > -DEFAULT_BORN_MAX_COORD) & (u_id < DEFAULT_BORN_MAX_COORD)
            u_id.clamp_(-DEFAULT_BORN_MAX_COORD, DEFAULT_BORN_MAX_COORD).mul_(inv_deg_id).add_(1.0).clamp_min_(0.0)
            u_id_pow = _poly_power_even(u_id, deg_id)
            phi_id = u_id_pow + DEFAULT_BORN_READOUT_EPS

            z_tr = torch.bmm(phi_id, i_sem.unsqueeze(-1)).squeeze(-1)
            m_id_v = torch.bmm(pi_c.unsqueeze(1), phi_id).squeeze(1)

            yc_clamped = yc.clamp_min(0)
            yc_idx = yc_clamped.view(c_len, 1, 1).expand(c_len, num_branches, 1)
            phi_id_y = phi_id.gather(dim=-1, index=yc_idx).squeeze(-1)
            del phi_id

            z_tot = (pi_c * z_tr).sum(dim=-1).clamp_min(DEFAULT_BORN_READOUT_EPS)
            m_id_y = (pi_c * phi_id_y).sum(dim=-1).clamp_min(DEFAULT_BORN_READOUT_EPS)

            # 1. Mode weight gradient dL / d(alpha_{t,r})
            g_pi = s_vec.unsqueeze(-1) * (z_tr / z_tot.unsqueeze(-1) - phi_id_y / m_id_y.unsqueeze(-1))
            dot_pi_g = (pi_c * g_pi).sum(dim=-1, keepdim=True)
            g_w = (g_pi - dot_pi_g) / w_sum_c
            g_alpha = g_w * u_m_c.pow(3) * (u_m_c > 0.0).float()
            grad_ctrl[start:end, :, 2] = g_alpha.to(branch_controls.dtype)

            # 2. Semantic + Carrier gradient dL / d(z_sem)
            g_sem = (i_sem * m_id_v) / z_tot.unsqueeze(-1)
            g_sem.scatter_add_(dim=-1, index=yc_clamped.unsqueeze(1), src=torch.full((c_len, 1), -1.0, device=g_sem.device, dtype=g_sem.dtype))
            g_sem.mul_(s_vec.unsqueeze(-1)).mul_(d_isem_dz / i_sem)
            del d_isem_dz, m_id_v

            g_sem_w = g_sem.to(work_dtype)
            del g_sem
            grad_heff_c = torch.matmul(g_sem_w, w_base)
            grad_w_base.add_(torch.matmul(g_sem_w.t(), heff_c))
            del g_sem_w

            grad_hb[start:end] = grad_heff_c
            grad_sem_part = grad_heff_c[:, :sem_dim]
            grad_hs[start:end] = grad_sem_part * bar_beta[start:end].to(work_dtype).unsqueeze(1)
            grad_ctrl[start:end, beta_col_idx, 0] = (grad_sem_part.float() * hs_c.float()).sum(dim=-1).to(branch_controls.dtype)

            # 3. Identity Cholesky mode gradients dL / d(z_{id, t, r, v})
            q_tv = i_sem / z_tot.unsqueeze(-1)
            del i_sem
            neg_inv_mid_y = (-1.0 / m_id_y).unsqueeze(1)
            q_tv.scatter_add_(dim=-1, index=yc_clamped.unsqueeze(1), src=neg_inv_mid_y)
            q_tv.mul_(s_vec.unsqueeze(-1))

            d_phi_dz = torch.where(
                (u_id > 1e-6) & id_in_bounds,
                u_id_pow / u_id.clamp_min(1e-6),
                torch.zeros_like(u_id_pow),
            )
            del u_id, u_id_pow, id_in_bounds
            d_phi_dz.mul_(pi_c.unsqueeze(-1)).mul_(q_tv.unsqueeze(1))
            del q_tv
            g_u_w = d_phi_dz.to(work_dtype)
            del d_phi_dz

            grad_hid_unscaled = torch.matmul(g_u_w.reshape(c_len * num_branches, v_sz), w_id).view(c_len, num_branches, id_dim)
            grad_hid[start:end] = grad_hid_unscaled * t_c.unsqueeze(-1).to(work_dtype)
            grad_ctrl[start:end, :, 1] = (grad_hid_unscaled.float() * hid_c.float()).sum(dim=-1).to(branch_controls.dtype)
            g_id_flat = (g_u_w * t_c.unsqueeze(-1).to(work_dtype)).reshape(c_len * num_branches, v_sz)
            grad_w_id.add_(torch.matmul(g_id_flat.t(), hid_c.reshape(c_len * num_branches, id_dim)))
            del g_u_w, grad_hid_unscaled

        grad_w_aug = torch.cat([grad_w_base[:, :sem_dim], grad_w_id, grad_w_base[:, sem_dim:]], dim=-1)
        return (
            grad_hb.view_as(h_base_464),
            grad_hs.view_as(h_sem_branches),
            grad_hid.view_as(h_id_branches),
            grad_ctrl.view_as(branch_controls),
            grad_w_aug,
            None,
            None,
        )


def fused_softmax_free_noclick_cholesky_ket_attention(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    r_re_im: tuple[torch.Tensor, torch.Tensor],
    v_mixed: torch.Tensor,
    high_mode_gates: torch.Tensor,
    noclick_params: tuple[torch.Tensor, torch.Tensor, bool],
    phase_and_born: tuple[torch.Tensor, torch.Tensor, float, bool],
    k_chol: int = DEFAULT_WINNING_CHOLESKY_RANK,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Single-pass Cholesky + Ket Attention with 100% `softmax`-free Polynomial Born No-Click Vacuum Sink (`<= 8` args)."""
    r_re, r_im = r_re_im
    bsz, num_heads, seq_len, v_rank, d_h = r_re.shape
    r_modes = max(1, int(k_chol))
    head_dim = q_rot.size(-1)
    k_vac_ket, vac_sink_bias, use_sink = noclick_params
    berry_phase, born_beta, sdpa_scale, use_hermitian_born = phase_and_born

    if q_rot.requires_grad or v_mixed.requires_grad:
        f_flat = torch.utils.checkpoint.checkpoint(
            _build_variable_rank_cholesky_f_flat, q_rot, k_rot, r_re, r_im, v_mixed, high_mode_gates, r_modes, use_reentrant=False
        )
    else:
        f_flat = _build_variable_rank_cholesky_f_flat(q_rot, k_rot, r_re, r_im, v_mixed, high_mode_gates, r_modes)

    chol_width = 2 * r_modes * d_h
    v_ket_flat = v_mixed.view(bsz, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    payload_packed = torch.cat([f_flat, v_ket_flat], dim=-1).contiguous()

    out_payload = _FusedHermitianBornRhoKetAttnFn.apply(
        q_rot, k_rot, payload_packed, None, berry_phase, born_beta, (float(sdpa_scale), bool(use_hermitian_born))
    )
    l_packed = out_payload[..., :chol_width].view(bsz, num_heads, seq_len, r_modes, 2, d_h)
    a_v = out_payload[..., chol_width:].view(bsz, num_heads, seq_len, v_rank, 2, d_h)
    l_re, l_im = l_packed[..., 0, :], l_packed[..., 1, :]
    v_bar_re, v_bar_im = a_v[..., 0, :], a_v[..., 1, :]

    if not use_sink:
        p_zero = torch.zeros((bsz, num_heads, seq_len), device=q_rot.device, dtype=q_rot.dtype)
        return l_re, l_im, v_bar_re, v_bar_im, p_zero

    # Softmax-free 4-boson polynomial Born competition between window value energy and Vacuum Key |k_{vac, h}>
    s_vac_bht = compute_noclick_vacuum_sink_scores(q_rot, k_vac_ket, vac_sink_bias, berry_phase, born_beta, float(sdpa_scale))
    v_bar_rms = (
        (v_bar_re.detach().float().square().mean(dim=(-2, -1)) + v_bar_im.detach().float().square().mean(dim=(-2, -1)))
        .clamp_min(1e-6)
        .sqrt()
    )
    w_vac = (1.0 + 0.25 * s_vac_bht.clamp(-4.0, 12.0)).clamp_min(0.0).square().square()
    w_win = (1.0 + 0.25 * v_bar_rms.clamp(-4.0, 12.0)).clamp_min(0.0).square().square() + DEFAULT_BORN_READOUT_EPS
    p_noclick = (w_vac / (w_win + w_vac)).to(q_rot.dtype)

    click_prob = (1.0 - p_noclick).unsqueeze(-1).unsqueeze(-1)
    v_bar_re_out = v_bar_re * click_prob
    v_bar_im_out = v_bar_im * click_prob

    if r_modes > 1:
        exc_shield = (1.0 - DEFAULT_NOCLICK_HIGHWAY_SHIELD_STRENGTH * p_noclick).unsqueeze(-1).unsqueeze(-1)
        l_re_out = torch.cat([l_re[:, :, :, :1, :], l_re[:, :, :, 1:, :] * exc_shield], dim=-2)
        l_im_out = torch.cat([l_im[:, :, :, :1, :], l_im[:, :, :, 1:, :] * exc_shield], dim=-2)
    else:
        l_re_out, l_im_out = l_re, l_im

    return l_re_out, l_im_out, v_bar_re_out, v_bar_im_out, p_noclick


class _SoftmaxFreeNoClickWavepacketBlock(_NoClickVacuumCholeskyWavepacketBlock):
    r"""Wavepacket Block with 100% `softmax`-free Polynomial Born No-Click Vacuum Sink."""

    def forward(
        self,
        x: torch.Tensor,
        x_anchor: torch.Tensor,
        cos_4d: torch.Tensor,
        sin_4d: torch.Tensor,
        in_cholesky: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Execute variable-rank Cholesky attention with `softmax`-free Polynomial Born No-Click Vacuum Sink."""
        if not self.use_noclick_vacuum_sink:
            self.last_mean_noclick_prob = self.last_id_head_noclick_prob = self.last_sem_head_noclick_prob = 0.0
            return super().forward(x, x_anchor, cos_4d, sin_4d, in_cholesky=in_cholesky)

        bsz, seq_len, d_model = x.shape
        work_dtype = x.dtype
        l0, l1 = self.block_lambdas[0].to(work_dtype), self.block_lambdas[1].to(work_dtype)
        x_in = l0 * x + l1 * x_anchor
        h_in = F.rms_norm(x_in, (d_model,), eps=1e-6)

        q_raw = self.q_proj(h_in).view(bsz, seq_len, self.num_heads, self.head_dim)
        k_raw = self.k_proj(h_in).view(bsz, seq_len, self.num_heads, self.head_dim)
        v_wp = self.v_proj(h_in).view(bsz, seq_len, self.pkt_count, self.d_h)

        g_ve = (2.0 * torch.sigmoid(F.linear(h_in[..., : self.gate_in_ch], self.ve_gate.weight.to(work_dtype)))).unsqueeze(-1)
        v_anc_h = x_anchor.view(bsz, seq_len, self.num_heads, self.head_dim) * g_ve
        s0, s1 = self.sa_lambdas[0].to(work_dtype), self.sa_lambdas[1].to(work_dtype)
        v_mixed_h = s0 * v_wp.view(bsz, seq_len, self.num_heads, self.head_dim) + s1 * v_anc_h

        q_rot = apply_native_leaderboard_rotary(F.rms_norm(q_raw, (self.head_dim,), eps=1e-6), cos_4d, sin_4d)
        k_rot = apply_native_leaderboard_rotary(F.rms_norm(k_raw, (self.head_dim,), eps=1e-6), cos_4d, sin_4d)
        if seq_len > 1:
            d_half = self.head_dim // 2
            k_shifted = torch.cat([k_rot[:, :1, :, d_half:], k_rot[:, :-1, :, d_half:]], dim=1)
            k_rot = torch.cat([k_rot[..., :d_half], k_shifted], dim=-1)

        v_c = v_mixed_h.view(bsz, seq_len, self.num_heads, self.v_rank, 2, self.d_h).permute(0, 2, 1, 3, 4, 5)
        vn_re, vn_im = v_c[..., 0, :], v_c[..., 1, :]
        nsq = (
            (vn_re.float().square().sum(dim=(-2, -1), keepdim=True) + vn_im.float().square().sum(dim=(-2, -1), keepdim=True))
            .clamp_min(TRACE_NORM_EPS)
            .rsqrt()
            .to(work_dtype)
        )
        r_re, r_im = vn_re * nsq, vn_im * nsq
        l_re, l_im, v_bar_re, v_bar_im, p_noclick = fused_softmax_free_noclick_cholesky_ket_attention(
            q_rot,
            k_rot,
            (r_re, r_im),
            v_mixed_h,
            self.high_mode_gates,
            (self.k_vac_ket, self.vac_sink_bias, self.use_noclick_vacuum_sink),
            (self.berry_qk_phase, self.born_qk_beta, self.sdpa_scale, True),
            k_chol=self.num_cholesky_modes,
        )
        with torch.no_grad():
            p_nc_f = p_noclick.detach().float()
            self.last_mean_noclick_prob = float(p_nc_f.mean().item())
            self.last_id_head_noclick_prob = float(p_nc_f[:, -1, :].mean().item())
            self.last_sem_head_noclick_prob = float(p_nc_f[:, :-1, :].mean().item())

        if self.training and l_re.requires_grad:
            out_l_r, out_l_i = torch.utils.checkpoint.checkpoint(self._cholesky_schur_step, l_re, l_im, use_reentrant=False)
        else:
            out_l_r, out_l_i = self._cholesky_schur_step(l_re, l_im)

        if in_cholesky is not None:
            if self.training and out_l_r.requires_grad:
                mod_l_r, mod_l_i = torch.utils.checkpoint.checkpoint(
                    self._merge_uncollapsed_cholesky_highway, in_cholesky[0], in_cholesky[1], out_l_r, out_l_i, use_reentrant=False
                )
            else:
                mod_l_r, mod_l_i = self._merge_uncollapsed_cholesky_highway(in_cholesky[0], in_cholesky[1], out_l_r, out_l_i)
        else:
            mod_l_r, mod_l_i = out_l_r, out_l_i
        self.last_cholesky_pre_u = (mod_l_r, mod_l_i)

        rho_scale = torch.tanh(self.rho_coupling).view(1, self.num_heads, 1, 1, 1)
        chol_fn = _get_compiled_chol_gram_fn() if x.is_cuda else associative_low_rank_cholesky_gram_modulation
        y_heads, pur_t = chol_fn(
            mod_l_r, mod_l_i, v_bar_re, v_bar_im, v_mixed_h, (self.u_evol_re, self.u_evol_im), (rho_scale, self.obs_scale, True)
        )
        self.last_mean_purity = float(pur_t.mean().item())
        self.last_mean_eff_rank = float((1.0 / pur_t.clamp_min(1e-4)).mean().item())

        v_head_rms = v_mixed_h.float().pow(2).mean(dim=-1, keepdim=True).clamp_min(TRACE_NORM_EPS).sqrt().to(work_dtype)
        y_scaled = y_heads.view(bsz, seq_len, self.num_heads, self.head_dim) * v_head_rms
        y_ortho = apply_fubini_study_self_ortho(y_scaled, v_mixed_h, self.self_ortho_alpha)
        g_at = (2.0 * torch.sigmoid(F.linear(h_in[..., : self.gate_in_ch], self.q_attn_gate.weight.to(work_dtype)))).unsqueeze(-1)
        y_flat = (y_ortho * g_at).reshape(bsz, seq_len, d_model)
        return self.mz_mlp(x_in + self.obs_proj(y_flat))


class SoftmaxFreeVacuumBornGPT(VariableRankVacuumCholeskyGPT):
    r"""100% `softmax`-Free Vacuum-Sharpened Coincidence Born GPT (`B0..B2`).

    Replaces readout `softmax`/`logsumexp`/`tanh`, Cholesky mode-mixture `softmax`, and (in `B1..B2`) No-Click sink `sigmoid`
    with $2p$-boson polynomial Fock Born amplitudes $\Phi_{2p}(z) = (1 + z / (2p))_+^{2p}$ on $\mathcal{H}_{\text{vocab}} \oplus \mathcal{H}_{\text{vac}}$.
    """

    def __init__(self, sf_spec: SoftmaxFreeVacuumBornSpec) -> None:
        v_mode = (
            VariableRankVariantMode.R0_RANK8_NO_VACUUM_SPHERE
            if sf_spec.variant_mode == SoftmaxFreeBornVariantMode.B0_SOFTMAX_FREE_BORN_NO_VACUUM_R8
            else VariableRankVariantMode.R4_RANK8_VACUUM_CHOLESKY
        )
        r_spec = VariableRankCholeskySpec(
            variant_mode=v_mode,
            num_cholesky_modes=sf_spec.num_cholesky_modes,
            vacuum_dim=0 if sf_spec.variant_mode == SoftmaxFreeBornVariantMode.B0_SOFTMAX_FREE_BORN_NO_VACUUM_R8 else sf_spec.vacuum_dim,
            use_layer0_input_cholesky=True,
            use_independent_rmode_512d=False,
            seed=sf_spec.seed,
        )
        super().__init__(r_spec)
        self.sf_spec = sf_spec
        self.born_degrees = (int(sf_spec.sem_born_degree), int(sf_spec.id_born_degree))

        noclick_cfg = (
            bool(sf_spec.use_noclick_vacuum_sink),
            float(DEFAULT_NOCLICK_HIGHWAY_SHIELD_STRENGTH),
            float(DEFAULT_INIT_NOCLICK_SINK_BIAS),
            float(DEFAULT_INIT_NOCLICK_POS_LOGIT),
        )
        base_blocks = list(self.quantum_blocks)
        new_blocks: list[_SoftmaxFreeNoClickWavepacketBlock] = []
        for b_old in base_blocks:
            b_new = _SoftmaxFreeNoClickWavepacketBlock(
                self.embed_dim,
                self.num_heads,
                self.pkt_count,
                self.pkt_dim,
                self.mlp_hidden,
                init_schur_scale=DEFAULT_INIT_SCHUR_GATE_SCALE,
                num_cholesky_modes=self.num_branches,
                noclick_cfg=noclick_cfg,
            )
            b_new.load_state_dict(b_old.state_dict(), strict=False)
            new_blocks.append(b_new)
        self.quantum_blocks = nn.ModuleList(new_blocks)

    def get_noclick_telemetry(self) -> dict[str, float]:
        """Return mean, identity-head, and semantic-head polynomial Born No-Click probabilities across all layers."""
        if not self.sf_spec.use_noclick_vacuum_sink:
            return {"mean_noclick_prob": 0.0, "id_head_noclick_prob": 0.0, "sem_head_noclick_prob": 0.0}
        n_l = float(max(1, len(self.quantum_blocks)))
        mean_nc = sum(float(getattr(b, "last_mean_noclick_prob", 0.0)) for b in self.quantum_blocks) / n_l
        id_nc = sum(float(getattr(b, "last_id_head_noclick_prob", 0.0)) for b in self.quantum_blocks) / n_l
        sem_nc = sum(float(getattr(b, "last_sem_head_noclick_prob", 0.0)) for b in self.quantum_blocks) / n_l
        return {
            "mean_noclick_prob": round(mean_nc, 4),
            "id_head_noclick_prob": round(id_nc, 4),
            "sem_head_noclick_prob": round(sem_nc, 4),
        }

    def forward_loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute 100% `softmax`-free Vacuum-Sharpened Coincidence Born cross-entropy loss."""
        h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug = self.build_multibranch_readout_tensors(idx)
        return _VacuumCoincidenceBornCEFn.apply(
            h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug, targets, self.born_degrees
        )


@torch.no_grad()
def evaluate_softmax_free_vacuum_born_model(
    model: SoftmaxFreeVacuumBornGPT, stream: LongSeqFineWebStream
) -> dict[str, float]:
    """Evaluate `SoftmaxFreeVacuumBornGPT` on unseen Shard-0 validation batches (`65,536` tokens, 100% `softmax`-free)."""
    was_training = model.training
    model.eval()

    deg_sem, deg_id = model.born_degrees
    inv_deg_sem, inv_deg_id = 1.0 / float(deg_sem), 1.0 / float(deg_id)
    total_neural_loss, branch_entropy_sum = 0.0, 0.0
    anchor_cos_sum, excited_cos_sum, core_exc_cos_sum, high_exc_cos_sum = 0.0, 0.0, 0.0, 0.0
    total_top1, total_top5, total_tokens, n_batches = 0, 0, 0, 0

    for idx_in, targets in stream.val_batches:
        bsz, seq_len = idx_in.shape
        n_tok = int(targets.numel())
        h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug_528 = model.build_multibranch_readout_tensors(idx_in)

        num_r, v_sz = h_id_branches.size(-2), w_aug_528.size(0)
        sem_dim, id_dim = h_sem_branches.size(-1), h_id_branches.size(-1)
        car_dim = h_base_464.size(-1) - sem_dim
        work_dtype = h_base_464.dtype

        sem1_gates, id_scales, gate_logits = branch_controls[:, :, :, 0], branch_controls[:, :, :, 1], branch_controls[:, :, :, 2]
        bar_beta = sem1_gates[:, :, 1] if num_r > 1 else sem1_gates[:, :, 0]
        sem_eff = h_base_464[:, :, :sem_dim] + bar_beta.to(work_dtype).unsqueeze(-1) * h_sem_branches
        heff_464 = torch.cat([sem_eff, h_base_464[:, :, sem_dim:]], dim=-1).contiguous()

        w_sem, w_id = w_aug_528[:, :sem_dim], w_aug_528[:, sem_dim : sem_dim + id_dim]
        w_car = w_aug_528[:, sem_dim + id_dim : sem_dim + id_dim + car_dim]
        w_base_t = torch.cat([w_sem, w_car], dim=-1).t().contiguous()
        w_id_t = w_id.t().contiguous()

        pi_b, _, _ = compute_softmax_free_mode_weights(gate_logits.float())
        ent_b = -(pi_b * torch.log(pi_b.clamp_min(1e-8))).sum(dim=-1).mean()
        branch_entropy_sum += float(ent_b.item())
        anchor_cos_sum += float(model.last_anchor_id_cos_vac)
        excited_cos_sum += float(model.last_excited_id_cos_vac)
        core_exc_cos_sum += float(model.last_core_excited_cos_vac)
        high_exc_cos_sum += float(model.last_high_excited_cos_vac)
        n_batches += 1

        t_step = max(8, (32 * 4) // max(1, num_r))
        for t_start in range(0, seq_len, t_step):
            t_end = min(t_start + t_step, seq_len)
            c_len = bsz * (t_end - t_start)
            heff_c = heff_464[:, t_start:t_end].reshape(c_len, sem_dim + car_dim)
            hid_c = h_id_branches[:, t_start:t_end].reshape(c_len * num_r, id_dim)
            t_c = id_scales[:, t_start:t_end].reshape(c_len, num_r).float()
            pi_c = pi_b[:, t_start:t_end].reshape(c_len, num_r).float()
            y_c = targets[:, t_start:t_end].reshape(c_len)

            z_sem = torch.matmul(heff_c, w_base_t).float().clamp(-DEFAULT_BORN_MAX_COORD, DEFAULT_BORN_MAX_COORD)
            u_sem = (1.0 + z_sem * inv_deg_sem).clamp_min_(0.0)
            i_sem = _poly_power_even(u_sem, deg_sem).add_(DEFAULT_BORN_READOUT_EPS)
            del z_sem, u_sem

            u_id = torch.matmul(hid_c, w_id_t).float().view(c_len, num_r, v_sz)
            u_id.mul_(t_c.unsqueeze(-1)).clamp_(-DEFAULT_BORN_MAX_COORD, DEFAULT_BORN_MAX_COORD)
            u_id.mul_(inv_deg_id).add_(1.0).clamp_min_(0.0)
            phi_id = _poly_power_even(u_id, deg_id).add_(DEFAULT_BORN_READOUT_EPS)
            del u_id

            m_id_v = torch.bmm(pi_c.unsqueeze(1), phi_id).squeeze(1).clamp_min(DEFAULT_BORN_READOUT_EPS)
            del phi_id
            b_mix = i_sem * m_id_v
            z_tot = b_mix.sum(dim=-1).clamp_min(DEFAULT_BORN_READOUT_EPS)
            b_target = b_mix.gather(dim=-1, index=y_c.unsqueeze(1)).squeeze(1).clamp_min(DEFAULT_BORN_READOUT_EPS)

            nll_neural = (torch.log(z_tot) - torch.log(b_target)).sum()
            total_neural_loss += float(nll_neural.item())

            top5_preds = b_mix.topk(5, dim=-1).indices
            total_top1 += int((top5_preds[:, 0] == y_c).sum().item())
            total_top5 += int((top5_preds == y_c.unsqueeze(1)).any(dim=-1).sum().item())
            del i_sem, m_id_v, b_mix
        del h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug_528, heff_464
        total_tokens += n_tok

    if stream.device.type == "cuda":
        torch.cuda.empty_cache()
    if was_training:
        model.train()

    mean_ce = total_neural_loss / float(max(1, total_tokens))
    mean_ent = branch_entropy_sum / float(max(1, n_batches))
    denom = float(max(1, n_batches))
    res = {
        "val_ce": round(mean_ce, 4),
        "neural_val_ce": round(mean_ce, 4),
        "val_ppl": round(math.exp(min(20.0, mean_ce)), 2),
        "top1_acc": round(total_top1 / float(max(1, total_tokens)), 4),
        "top5_acc": round(total_top5 / float(max(1, total_tokens)), 4),
        "branch_gate_entropy_nats": round(mean_ent, 4),
        "effective_active_branches": round(math.exp(mean_ent), 2),
        "mean_anchor_id_cos_theta_vac": round(anchor_cos_sum / denom, 4),
        "mean_excited_id_cos_theta_vac": round(excited_cos_sum / denom, 4),
        "mean_core_excited_cos_theta_vac": round(core_exc_cos_sum / denom, 4),
        "mean_high_excited_cos_theta_vac": round(high_exc_cos_sum / denom, 4),
    }
    res.update(model.get_noclick_telemetry())
    return res


def run_softmax_free_vacuum_born_ablation_benchmark(
    steps: int = DEFAULT_FINEWEB_STEPS,
    eval_interval: int = DEFAULT_FINEWEB_EVAL_INTERVAL,
    device_str: str = "cuda",
) -> list[dict[str, Any]]:
    r"""Run the 300-step (`2.46M`-token) 100% `softmax`-Free Vacuum-Sharpened Coincidence Born ablation (`B0..B2`, `<= 8` args)."""
    device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    stream = LongSeqFineWebStream(device=device)
    vac_d = DEFAULT_VACUUM_SUBSPACE_DIM
    tokens_per_step = stream.batch_size * stream.seq_len

    variants: list[tuple[str, SoftmaxFreeVacuumBornSpec, str]] = [
        (
            "B0_SOFTMAX_FREE_BORN_NO_VACUUM_R8",
            SoftmaxFreeVacuumBornSpec(SoftmaxFreeBornVariantMode.B0_SOFTMAX_FREE_BORN_NO_VACUUM_R8, 8, 0, False),
            "100% Softmax-Free Coincidence Born Readout WITHOUT Vacuum Subspace (R=8, d_vac=0, forced S^63)",
        ),
        (
            "B1_SOFTMAX_FREE_VACUUM_BORN_R8",
            SoftmaxFreeVacuumBornSpec(SoftmaxFreeBornVariantMode.B1_SOFTMAX_FREE_VACUUM_BORN_R8, 8, vac_d, True),
            "100% Softmax-Free Vacuum-Sharpened Coincidence Born Readout + Trunk No-Click Born Sink (R=8, d_vac=4)",
        ),
        (
            "B2_SOFTMAX_FREE_VACUUM_BORN_R12",
            SoftmaxFreeVacuumBornSpec(SoftmaxFreeBornVariantMode.B2_SOFTMAX_FREE_VACUUM_BORN_R12, 12, vac_d, True),
            "100% Softmax-Free Vacuum-Sharpened Coincidence Born Readout + Trunk No-Click Born Sink (R=12, d_vac=4)",
        ),
    ]

    results: list[dict[str, Any]] = []
    for variant_id, sf_spec, desc in variants:
        torch.manual_seed(sf_spec.seed)
        model = SoftmaxFreeVacuumBornGPT(sf_spec).to(device=device)
        init_ms, _ = initialize_and_calibrate_variable_rank_cholesky_model_(model, stream)
        model.clamp_identity_dispersion_and_scalars_()

        run_key = f"plan080_{variant_id.lower()}_{steps}steps"
        intent = ExperimentIntentSpec(
            run_key=run_key,
            wave_id="W60_BORN",
            plan_id="EXPERIMENT_PLAN_080",
            task_domain="fineweb_speedrun",
            hypothesis_intent=desc,
            architecture_summary=f"SoftmaxFreeVacuumBornGPT(L=4, d=512, R={sf_spec.num_cholesky_modes}, d_vac={model.d_vac}, sink={sf_spec.use_noclick_vacuum_sink})",
            hardware_budget="1x RTX 5060 Ti 16GB",
            batch_size=stream.batch_size,
            seq_len=stream.seq_len,
            total_steps=steps,
            target_tokens_M=round((steps * tokens_per_step) / 1e6, 3),
        )
        exp_logger = StandardizedExperimentLogger(intent)
        optimizers = build_variable_rank_cholesky_optimizers(model)

        step0_eval = evaluate_softmax_free_vacuum_born_model(model, stream)
        step0_eval["step"], step0_eval["tokens_M"] = 0, 0.0
        exp_logger.log_eval(0, {**step0_eval, "top1_acc": round(100.0 * step0_eval["top1_acc"], 2), "top5_acc": round(100.0 * step0_eval["top5_acc"], 2)})
        print(
            f"[{variant_id} (R={sf_spec.num_cholesky_modes}, d_vac={model.d_vac}, sink={sf_spec.use_noclick_vacuum_sink})] "
            f"Step    0 | Softmax-Free Born Val CE: {step0_eval['val_ce']:.4f} (PPL {step0_eval['val_ppl']:.2f}) | "
            f"Modes: {step0_eval['effective_active_branches']:.2f}/{sf_spec.num_cholesky_modes} | p_noclick: {step0_eval['mean_noclick_prob']:.3f}",
            flush=True,
        )

        curve: list[dict[str, Any]] = [step0_eval]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        model.train()

        for step in range(1, steps + 1):
            warm_steps = max(1, int(0.08 * steps))
            lr_mult = (
                0.20 + 0.80 * (float(step - 1) / float(max(1, warm_steps - 1)))
                if step <= warm_steps
                else 0.12 + 0.88 * 0.5 * (1.0 + math.cos(math.pi * float(step - warm_steps) / float(max(1, steps - warm_steps))))
            )
            mom_beta = compute_riemannian_momentum_beta(step, steps, QuantumMomentumScheduleMode.LOW_STEADY_MOMENTUM_070)
            idx_in, targets = stream.sample_train_batch(step)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss = model.forward_loss(idx_in, targets)
            train_loss_val = float(loss.detach().item())
            loss.backward()
            del loss
            _apply_global_optimizer_step(model, optimizers, idx_in, targets, lr_mult=lr_mult, mom_beta=mom_beta)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            if step % eval_interval == 0 or step == steps:
                ev = evaluate_softmax_free_vacuum_born_model(model, stream)
                ev["step"], ev["tokens_M"] = step, round((step * tokens_per_step) / 1e6, 3)
                ev["train_loss"], ev["lr_mult"], ev["mom_beta"] = round(train_loss_val, 4), round(lr_mult, 4), round(mom_beta, 4)
                curve.append(ev)
                exp_logger.log_eval(step, {**ev, "top1_acc": round(100.0 * ev["top1_acc"], 2), "top5_acc": round(100.0 * ev["top5_acc"], 2)})
                print(
                    f"[{variant_id} (R={sf_spec.num_cholesky_modes}, d_vac={model.d_vac}, sink={sf_spec.use_noclick_vacuum_sink})] "
                    f"Step {step:4d} ({ev['tokens_M']:5.2f}M tok, lr={lr_mult:.2f}x) | "
                    f"Train CE: {ev['train_loss']:.4f} | Softmax-Free Born Val CE: {ev['val_ce']:.4f} (PPL {ev['val_ppl']:.2f}) | "
                    f"Top1: {ev['top1_acc']*100:.2f}% | Top5: {ev['top5_acc']*100:.2f}% | "
                    f"Modes: {ev['effective_active_branches']:.2f}/{sf_spec.num_cholesky_modes} | "
                    f"cos(vac) [anc/exc]: {ev['mean_anchor_id_cos_theta_vac']:.3f}/{ev['mean_excited_id_cos_theta_vac']:.3f} | "
                    f"p_noclick: {ev['mean_noclick_prob']:.3f}",
                    flush=True,
                )

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - t_start
        results.append({
            "variant_id": variant_id,
            "variant_mode": sf_spec.variant_mode.value,
            "description": desc,
            "num_cholesky_modes": sf_spec.num_cholesky_modes,
            "vacuum_dim": model.d_vac,
            "use_noclick_vacuum_sink": sf_spec.use_noclick_vacuum_sink,
            "momentum_beta": 0.70,
            "init_ms": round(init_ms, 2),
            "elapsed_sec": round(elapsed_s, 2),
            "step0": step0_eval,
            "final": curve[-1],
            "curve": curve,
        })
        del model, optimizers
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


__all__ = [
    "DEFAULT_BORN_ATTN_DEGREE",
    "DEFAULT_BORN_ID_DEGREE",
    "DEFAULT_BORN_MAX_COORD",
    "DEFAULT_BORN_MODE_DEGREE",
    "DEFAULT_BORN_READOUT_EPS",
    "DEFAULT_BORN_SEM_DEGREE",
    "SoftmaxFreeBornVariantMode",
    "SoftmaxFreeVacuumBornGPT",
    "SoftmaxFreeVacuumBornSpec",
    "_SoftmaxFreeNoClickWavepacketBlock",
    "_VacuumCoincidenceBornCEFn",
    "compute_softmax_free_mode_weights",
    "evaluate_softmax_free_vacuum_born_model",
    "fused_softmax_free_noclick_cholesky_ket_attention",
    "run_softmax_free_vacuum_born_ablation_benchmark",
]
