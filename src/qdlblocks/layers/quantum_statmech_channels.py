"""Wave 49 (`EXPERIMENT_PLAN_064.md`): Reusable Quantum Statistical-Mechanics Channels & Purification Operators.

Factors the core quantum density-matrix evolution, $p$-boson Born coincidence weighting, Bures-Wasserstein /
Fermi-Dirac purification, Uhlmann parallel-transport Krylov extraction, and open-system CPTP Kraus channels
out of `standalone_quantum_born_gpt.py` into reusable, strategy-enum-controlled library primitives:

1. **Thermodynamic Density Purification (`PurificationMode`)**:
   - `QUADRATIC_MCWEENY`: 1-step Bures-Wasserstein McWeeny map $\\rho^{(1)} = \\rho_0^2 / \\operatorname{Tr}(\\rho_0^2)$.
   - `CUBIC_FERMI_DIRAC`: Symmetric zero-temperature Fermi-Dirac spectral step
     $\\tilde{\\rho} = 3\\rho_0^2 - 2\\rho_0^3 = \\rho_0^2(3I - 2\\rho_0)$, $\\rho^{(1)} = \\tilde{\\rho} / \\operatorname{Tr}(\\tilde{\\rho})$.
   - `LEARNABLE_TEMP_MCWEENY`: Per-head learnable convex polynomial combination over
     $\\{\\rho_0,\\; \\rho_{\\text{quad}},\\; \\rho_{\\text{FD}}\\}$ parameterized by per-head softmax logits $\\theta_h \\in \\mathbb{R}^3$.
2. **Closed Unitary $U(d_h)$ vs. Open CPTP Kraus Channel Evolution (`QuantumChannelMode`)**:
   - `UNITARY_U16`: Closed-system unitary conjugation $\\mathcal{U}_h(\\rho) = U_h \\rho U_h^\\dagger$ ($U_h \\in U(d_h)$).
   - `CPTP_KRAUS_RANK_K`: Completely Positive Trace-Preserving operator-sum channel
     $\\mathcal{E}_h(\\rho) = \\sum_{k=1}^K A_{h,k} \\rho A_{h,k}^\\dagger$ where $A_h = [A_{h,1}^\\top, \\dots, A_{h,K}^\\top]^\\top \\in \\mathrm{St}_{\\mathbb{C}}(K d_h, d_h)$
     satisfies $\\sum_{k=1}^K A_{h,k}^\\dagger A_{h,k} = I_{d_h}$.
3. **Detached Purity / Tsallis-2 Entropy Gate (`EntropyGateMode`)**:
   - `NONE`: Standard fixed $1/\\sqrt{2}$ isometric fusion of `packed_rho_out` and `psi_u_flat`.
   - `DETACHED_TSALLIS_PURITY_GATE`: Uses detached purity $\\mathcal{P}_{\\text{det}} = \\operatorname{Tr}(\\rho^2).\\text{detach}()$
     and Tsallis-2 linear entropy $S_2 = 1 - \\mathcal{P}_{\\text{det}}$ to route between coherent wavepackets
     and mixed density/Krylov features while weighting head confidence without radial collapse gradients.
"""

from __future__ import annotations

from enum import Enum
import math
from typing import Any

import torch
import torch.nn.functional as F

from .impure_density_mirror_block import (
    DEFAULT_NS5_STEPS,
    DEFAULT_RIEMANNIAN_MOMENTUM_BETA,
    DEFAULT_RIEMANNIAN_MUON_LR,
    DEFAULT_TIKHONOV_GRAM_LAMBDA,
    DEFAULT_UHLMANN_PHASE_EPS,
    DEFAULT_UNITARY_GEODESIC_LR,
    INV_SQRT2_ISOMETRIC_SCALE,
    SQRT2_ISOMETRIC_OFFDIAG_SCALE,
    TRACE_NORM_EPS,
    _complex_newton_schulz5,
    _newton_schulz5_tall_real,
    bures_wasserstein_mcweeny_step,
    extract_rank2_krylov_eigenpair_uhlmann,
    unpack_hermitian_isometric,
)

# ---------------------------------------------------------------------------
# Top-of-File Named Constants (Rule 7 — Synchronized with EXPERIMENT_PLAN_064.md)
# ---------------------------------------------------------------------------
DEFAULT_STATMECH_MCWEENY_STEPS: int = 1
DEFAULT_KRAUS_CHANNEL_RANK: int = 4
DEFAULT_KRAUS_DOMINANT_WEIGHT: float = 0.85
DEFAULT_TSALLIS_PURITY_CENTER: float = 0.35
DEFAULT_COLUMN_SCALE_LR_RATIO: float = 1.00
DEFAULT_COLUMN_SCALE_MIN: float = 0.50
DEFAULT_COLUMN_SCALE_MAX: float = 2.00


class PurificationMode(str, Enum):
    """Strategy enum selecting quantum statistical-mechanics purification polynomial (Rule 3)."""

    QUADRATIC_MCWEENY = "quadratic_mcweeny"
    CUBIC_FERMI_DIRAC = "cubic_fermi_dirac"
    LEARNABLE_TEMP_MCWEENY = "learnable_temp_mcweeny"


class QuantumChannelMode(str, Enum):
    """Strategy enum selecting closed $U(d_h)$ unitary evolution vs. open Rank-$K$ CPTP Kraus channel (Rule 3)."""

    UNITARY_U16 = "unitary_u16"
    CPTP_KRAUS_RANK_K = "cptp_kraus_rank_k"


class EntropyGateMode(str, Enum):
    """Strategy enum selecting fixed isometric fusion, normalized branches, or detached Tsallis-2 gating (Rule 3)."""

    NONE = "none"
    DETACHED_TSALLIS_PURITY_GATE = "detached_tsallis_purity_gate"
    ISOMETRIC_NORMALIZED_BRANCHES = "isometric_normalized_branches"
    ISOMETRIC_NORMALIZED_PLUS_TSALLIS = "isometric_normalized_plus_tsallis"


class StiefelMomentumMode(str, Enum):
    """Strategy enum selecting ambient Euclidean, tangent-transported, column-scaled, or unnormalized-boundary momentum (Rule 3)."""

    EUCLIDEAN_AMBIENT = "euclidean_ambient"
    RIEMANNIAN_TANGENT_TRANSPORTED = "riemannian_tangent_transported"
    RIEMANNIAN_COLUMN_SCALED_STIEFEL = "riemannian_column_scaled_stiefel"
    UNNORMALIZED_FINAL_READOUT_HEAD = "unnormalized_final_readout_head"
    UNNORMALIZED_FINAL_BLOCK_AND_HEAD = "unnormalized_final_block_and_head"


_COL_SCALED_STIEFEL_MODES = (
    StiefelMomentumMode.RIEMANNIAN_COLUMN_SCALED_STIEFEL,
    StiefelMomentumMode.UNNORMALIZED_FINAL_READOUT_HEAD,
    StiefelMomentumMode.UNNORMALIZED_FINAL_BLOCK_AND_HEAD,
)


def causal_born_p3_weights_impl(ip_re: torch.Tensor, ip_im: torch.Tensor) -> torch.Tensor:
    """Compute causal $p=3$ polynomial Born coincidence weights $|\\langle q_t \\mid k_s\\rangle|^6$."""
    seq_len = ip_re.size(-1)
    born_mod_sq = (ip_re.float().square() + ip_im.float().square()).clamp_min(0.0)
    born_scores = born_mod_sq * born_mod_sq.square()
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=ip_re.device, dtype=torch.bool))
    born_scores = born_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), 0.0)
    return (born_scores / born_scores.sum(dim=-1, keepdim=True).clamp_min(TRACE_NORM_EPS)).to(ip_re.dtype)


_COMPILED_BORN_P3_HELPER: Any = None


def causal_born_p3_weights(
    ip_re: torch.Tensor,
    ip_im: torch.Tensor,
    use_compile: bool = False,
) -> torch.Tensor:
    """Execute `causal_born_p3_weights_impl` with optional `torch.compile` fusion."""
    global _COMPILED_BORN_P3_HELPER
    if use_compile and ip_re.is_cuda:
        if _COMPILED_BORN_P3_HELPER is None:
            _COMPILED_BORN_P3_HELPER = torch.compile(causal_born_p3_weights_impl, dynamic=False)
        return _COMPILED_BORN_P3_HELPER(ip_re, ip_im)
    return causal_born_p3_weights_impl(ip_re, ip_im)


def cubic_fermi_dirac_purification_step(
    rho_re: torch.Tensor,
    rho_im: torch.Tensor,
    eps: float = TRACE_NORM_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply symmetric cubic Fermi-Dirac spectral purification $\\tilde{\\rho} = 3\\rho^2 - 2\\rho^3$, $\\rho^{(1)} = \\tilde{\\rho}/\\operatorname{Tr}(\\tilde{\\rho})$."""
    tr_cur = torch.diagonal(rho_re, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(eps)
    cur_re = rho_re / tr_cur
    cur_im = rho_im / tr_cur

    sq_re = torch.matmul(cur_re, cur_re) - torch.matmul(cur_im, cur_im)
    sq_im = torch.matmul(cur_re, cur_im) + torch.matmul(cur_im, cur_re)
    sq_re = 0.5 * (sq_re + sq_re.transpose(-2, -1))
    sq_im = 0.5 * (sq_im - sq_im.transpose(-2, -1))

    cub_re = torch.matmul(sq_re, cur_re) - torch.matmul(sq_im, cur_im)
    cub_im = torch.matmul(sq_re, cur_im) + torch.matmul(sq_im, cur_re)
    fd_raw_re = 3.0 * sq_re - 2.0 * cub_re
    fd_raw_im = 3.0 * sq_im - 2.0 * cub_im
    fd_raw_re = 0.5 * (fd_raw_re + fd_raw_re.transpose(-2, -1))
    fd_raw_im = 0.5 * (fd_raw_im - fd_raw_im.transpose(-2, -1))

    tr_fd = torch.diagonal(fd_raw_re, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(eps)
    out_re = fd_raw_re / tr_fd
    out_im = fd_raw_im / tr_fd
    purity = (out_re.square().sum(dim=(-2, -1)) + out_im.square().sum(dim=(-2, -1))).clamp(eps, 1.0)
    eff_rank = (1.0 / purity).clamp(1.0, float(rho_re.size(-1)))
    return out_re, out_im, purity, eff_rank


def learnable_temp_purification_step(
    rho_re: torch.Tensor,
    rho_im: torch.Tensor,
    purify_logits_h3: torch.Tensor,
    eps: float = TRACE_NORM_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply per-head learnable convex polynomial purification over $\\{\\rho_0, \\rho_{\\text{quad}}, \\rho_{\\text{FD}}\\}$."""
    tr_cur = torch.diagonal(rho_re, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(eps)
    cur_re = rho_re / tr_cur
    cur_im = rho_im / tr_cur

    sq_re = torch.matmul(cur_re, cur_re) - torch.matmul(cur_im, cur_im)
    sq_im = torch.matmul(cur_re, cur_im) + torch.matmul(cur_im, cur_re)
    sq_re = 0.5 * (sq_re + sq_re.transpose(-2, -1))
    sq_im = 0.5 * (sq_im - sq_im.transpose(-2, -1))
    tr_sq = torch.diagonal(sq_re, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(eps)
    quad_re = sq_re / tr_sq
    quad_im = sq_im / tr_sq

    cub_re = torch.matmul(sq_re, cur_re) - torch.matmul(sq_im, cur_im)
    cub_im = torch.matmul(sq_re, cur_im) + torch.matmul(sq_im, cur_re)
    fd_raw_re = 3.0 * sq_re - 2.0 * cub_re
    fd_raw_im = 3.0 * sq_im - 2.0 * cub_im
    fd_raw_re = 0.5 * (fd_raw_re + fd_raw_re.transpose(-2, -1))
    fd_raw_im = 0.5 * (fd_raw_im - fd_raw_im.transpose(-2, -1))
    tr_fd = torch.diagonal(fd_raw_re, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(eps)
    fd_re = fd_raw_re / tr_fd
    fd_im = fd_raw_im / tr_fd

    weights = F.softmax(purify_logits_h3.float(), dim=-1).to(rho_re.dtype)
    w_lin = weights[:, 0].view(1, -1, 1, 1, 1)
    w_quad = weights[:, 1].view(1, -1, 1, 1, 1)
    w_fd = weights[:, 2].view(1, -1, 1, 1, 1)

    out_re = w_lin * cur_re + w_quad * quad_re + w_fd * fd_re
    out_im = w_lin * cur_im + w_quad * quad_im + w_fd * fd_im
    purity = (out_re.square().sum(dim=(-2, -1)) + out_im.square().sum(dim=(-2, -1))).clamp(eps, 1.0)
    eff_rank = (1.0 / purity).clamp(1.0, float(rho_re.size(-1)))
    return out_re, out_im, purity, eff_rank


def purify_density_statmech(
    rho_re: torch.Tensor,
    rho_im: torch.Tensor,
    mode: PurificationMode = PurificationMode.QUADRATIC_MCWEENY,
    purify_logits_h3: torch.Tensor | None = None,
    eps: float = TRACE_NORM_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch density purification across `QUADRATIC_MCWEENY`, `CUBIC_FERMI_DIRAC`, and `LEARNABLE_TEMP_MCWEENY`."""
    if mode == PurificationMode.CUBIC_FERMI_DIRAC:
        return cubic_fermi_dirac_purification_step(rho_re, rho_im, eps=eps)
    if mode == PurificationMode.LEARNABLE_TEMP_MCWEENY and purify_logits_h3 is not None:
        return learnable_temp_purification_step(rho_re, rho_im, purify_logits_h3, eps=eps)
    return bures_wasserstein_mcweeny_step(
        rho_re, rho_im, num_steps=DEFAULT_STATMECH_MCWEENY_STEPS, eps=eps
    )


def evolve_head_unitary_functional(
    rho_re: torch.Tensor,
    rho_im: torch.Tensor,
    head_u_re: torch.Tensor,
    head_u_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-head complex unitary conjugation $U_h \\rho_{t,h} U_h^\\dagger$ where $U_h \\in U(d_h)$."""
    u_re = head_u_re.to(rho_re.dtype).unsqueeze(0).unsqueeze(2)
    u_im = head_u_im.to(rho_re.dtype).unsqueeze(0).unsqueeze(2)
    ut_re = u_re.transpose(-2, -1)
    ut_im = -u_im.transpose(-2, -1)

    tmp_re = torch.matmul(u_re, rho_re) - torch.matmul(u_im, rho_im)
    tmp_im = torch.matmul(u_re, rho_im) + torch.matmul(u_im, rho_re)
    out_re = torch.matmul(tmp_re, ut_re) - torch.matmul(tmp_im, ut_im)
    out_im = torch.matmul(tmp_re, ut_im) + torch.matmul(tmp_im, ut_re)
    return 0.5 * (out_re + out_re.transpose(-2, -1)), 0.5 * (out_im - out_im.transpose(-2, -1))


def evolve_head_unitary_traceless_functional(
    rho_re: torch.Tensor,
    rho_im: torch.Tensor,
    head_u_re: torch.Tensor,
    head_u_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply traceless complex unitary conjugation $I_{d_h}/d_h + U_h(\\rho - I_{d_h}/d_h)U_h^\\dagger$ eliminating identity-trace radial gradients."""
    d_h = rho_re.size(-1)
    inv_d = 1.0 / float(d_h)
    eye_h = torch.eye(d_h, device=rho_re.device, dtype=rho_re.dtype)
    rot_tl_re, rot_tl_im = evolve_head_unitary_functional(rho_re - inv_d * eye_h, rho_im, head_u_re, head_u_im)
    return rot_tl_re + inv_d * eye_h, rot_tl_im


def initialize_complex_kraus_stiefel_pair(
    num_heads: int,
    d_h: int,
    kraus_rank: int = DEFAULT_KRAUS_CHANNEL_RANK,
    dominant_weight: float = DEFAULT_KRAUS_DOMINANT_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize tall complex Stiefel Kraus matrices $A_h \\in \\mathrm{St}_{\\mathbb{C}}(K d_h, d_h)$ with $\\sum_{k=1}^K A_{h,k}^\\dagger A_{h,k} = I_{d_h}$."""
    k_rank = max(1, int(kraus_rank))
    phase_angles = torch.linspace(-0.25, 0.25, d_h, dtype=torch.float32)
    base_re = torch.diag(torch.cos(phase_angles)).unsqueeze(0).expand(num_heads, -1, -1).clone()
    base_im = torch.diag(torch.sin(phase_angles)).unsqueeze(0).expand(num_heads, -1, -1).clone()
    if k_rank == 1:
        return base_re, base_im

    w1 = math.sqrt(max(0.5, min(0.99, float(dominant_weight))))
    w_rest = math.sqrt((1.0 - w1 * w1) / float(k_rank - 1))
    blocks_re = [w1 * base_re]
    blocks_im = [w1 * base_im]
    for k_idx in range(1, k_rank):
        shift_angles = phase_angles + 0.37 * float(k_idx)
        perm = torch.roll(torch.arange(d_h), shifts=k_idx)
        blk_re = torch.diag(torch.cos(shift_angles))[:, perm].unsqueeze(0).expand(num_heads, -1, -1).clone()
        blk_im = torch.diag(torch.sin(shift_angles))[:, perm].unsqueeze(0).expand(num_heads, -1, -1).clone()
        blocks_re.append(w_rest * blk_re)
        blocks_im.append(w_rest * blk_im)

    tall_re = torch.cat(blocks_re, dim=-2)
    tall_im = torch.cat(blocks_im, dim=-2)
    return _complex_newton_schulz5(tall_re, tall_im, steps=8, tikhonov_lambda=0.0)


def evolve_head_cptp_kraus_functional(
    rho_re: torch.Tensor,
    rho_im: torch.Tensor,
    kraus_stiefel_re: torch.Tensor,
    kraus_stiefel_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rank-$K$ CPTP Kraus channel $\\mathcal{E}_h(\\rho) = \\sum_{k=1}^K A_{h,k} \\rho A_{h,k}^\\dagger$ where $A_h \\in \\mathrm{St}_{\\mathbb{C}}(K d_h, d_h)$."""
    d_h = rho_re.size(-1)
    k_rank = kraus_stiefel_re.size(-2) // d_h
    sum_re = torch.zeros_like(rho_re)
    sum_im = torch.zeros_like(rho_im)
    for k_idx in range(k_rank):
        s = k_idx * d_h
        ak_re = kraus_stiefel_re[:, s : s + d_h, :]
        ak_im = kraus_stiefel_im[:, s : s + d_h, :]
        br_re, br_im = evolve_head_unitary_functional(rho_re, rho_im, ak_re, ak_im)
        sum_re = sum_re + br_re
        sum_im = sum_im + br_im
    out_re = 0.5 * (sum_re + sum_re.transpose(-2, -1))
    out_im = 0.5 * (sum_im - sum_im.transpose(-2, -1))
    tr_out = torch.diagonal(out_re, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(TRACE_NORM_EPS)
    return out_re / tr_out, out_im / tr_out


def project_real_stiefel_tangent(vec: torch.Tensor, weight_tall: torch.Tensor) -> torch.Tensor:
    """Project `vec` `(..., D, d)` ($D \\ge d$) onto the canonical Stiefel tangent space $T_W\\mathrm{St}(D, d)$ ($V - W\\,\\mathrm{sym}(W^\\top V)$)."""
    wt_v = torch.matmul(weight_tall.transpose(-2, -1), vec)
    sym_part = 0.5 * (wt_v + wt_v.transpose(-2, -1))
    return vec - torch.matmul(weight_tall, sym_part)


def project_complex_stiefel_tangent(
    v_re: torch.Tensor,
    v_im: torch.Tensor,
    a_re: torch.Tensor,
    a_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project complex `(v_re, v_im)` `(..., D, d)` ($D \\ge d$) onto $T_A\\mathrm{St}_{\\mathbb{C}}(D, d)$ ($V - A\\,\\mathrm{Herm}(A^\\dagger V)$)."""
    at_re = a_re.transpose(-2, -1)
    at_im = -a_im.transpose(-2, -1)
    p_re = torch.matmul(at_re, v_re) - torch.matmul(at_im, v_im)
    p_im = torch.matmul(at_re, v_im) + torch.matmul(at_im, v_re)
    sym_re = 0.5 * (p_re + p_re.transpose(-2, -1))
    sym_im = 0.5 * (p_im - p_im.transpose(-2, -1))
    t_re = v_re - (torch.matmul(a_re, sym_re) - torch.matmul(a_im, sym_im))
    t_im = v_im - (torch.matmul(a_re, sym_im) + torch.matmul(a_im, sym_re))
    return t_re, t_im


def compute_real_stiefel_vector_diagnostics(
    w_prev: torch.Tensor,
    g_raw: torch.Tensor,
    m_pre: torch.Tensor,
    m_updated: torch.Tensor,
    m_stored: torch.Tensor,
    u_tan: torch.Tensor,
    w_next: torch.Tensor,
) -> dict[str, float]:
    """Compare parameter gradient, momentum, Newton-Schulz tangent velocity, and retraction chord vectors on $\\mathrm{St}(D, d)$ (`<= 8` args)."""
    eps = 1e-8
    wp, gr, mp, mu, ms, ut, wn = (
        w_prev.float(),
        g_raw.float(),
        m_pre.float(),
        m_updated.float(),
        m_stored.float(),
        u_tan.float(),
        w_next.float(),
    )
    if wp.size(-2) < wp.size(-1):
        wp, gr, mp, mu, ms, ut, wn = (
            wp.transpose(-2, -1),
            gr.transpose(-2, -1),
            mp.transpose(-2, -1),
            mu.transpose(-2, -1),
            ms.transpose(-2, -1),
            ut.transpose(-2, -1),
            wn.transpose(-2, -1),
        )

    wp_unit = wp / wp.norm(dim=-2, keepdim=True).clamp_min(eps)
    wn_unit = wn / wn.norm(dim=-2, keepdim=True).clamp_min(eps)

    g_tan = project_real_stiefel_tangent(gr, wp_unit)
    g_norm = gr - g_tan
    wt_g = torch.matmul(wp_unit.transpose(-2, -1), gr)
    g_skew_vert = torch.matmul(wp_unit, 0.5 * (wt_g - wt_g.transpose(-2, -1)))
    gr_fro = torch.linalg.matrix_norm(gr, ord="fro").clamp_min(eps)
    grad_normal_frac = (torch.linalg.matrix_norm(g_norm, ord="fro") / gr_fro).mean()
    grad_skew_tangent_frac = (torch.linalg.matrix_norm(g_skew_vert, ord="fro") / gr_fro).mean()

    m_pre_tan = project_real_stiefel_tangent(mp, wp_unit)
    m_pre_norm = mp - m_pre_tan
    pre_mom_normal_frac = (
        torch.linalg.matrix_norm(m_pre_norm, ord="fro") / torch.linalg.matrix_norm(mp, ord="fro").clamp_min(eps)
    ).mean()

    m_tan_w = project_real_stiefel_tangent(mu, wp_unit)
    m_norm_w = mu - m_tan_w
    leak_into_next_tan = project_real_stiefel_tangent(m_norm_w, wn_unit)
    total_next_tan = project_real_stiefel_tangent(mu, wn_unit)
    normal_to_tangent_leakage_ratio = (
        torch.linalg.matrix_norm(leak_into_next_tan, ord="fro")
        / torch.linalg.matrix_norm(total_next_tan, ord="fro").clamp_min(eps)
    ).mean()

    ms_tan_next = project_real_stiefel_tangent(ms, wn_unit)
    ms_norm_next = ms - ms_tan_next
    post_mom_normal_frac = (
        torch.linalg.matrix_norm(ms_norm_next, ord="fro") / torch.linalg.matrix_norm(ms, ord="fro").clamp_min(eps)
    ).mean()

    def _batch_cos(a: torch.Tensor, b: torch.Tensor) -> float:
        dot = (a * b).sum(dim=(-2, -1))
        na = torch.linalg.matrix_norm(a, ord="fro").clamp_min(eps)
        nb = torch.linalg.matrix_norm(b, ord="fro").clamp_min(eps)
        return float((dot / (na * nb)).mean().item())

    delta_w = wn - wp
    delta_w_tan = project_real_stiefel_tangent(delta_w, wp_unit)
    delta_w_norm = delta_w - delta_w_tan
    retract_normal_frac = (
        torch.linalg.matrix_norm(delta_w_norm, ord="fro") / torch.linalg.matrix_norm(delta_w, ord="fro").clamp_min(eps)
    ).mean()

    return {
        "grad_normal_frac": float(grad_normal_frac.item()),
        "grad_skew_tangent_frac": float(grad_skew_tangent_frac.item()),
        "pre_mom_normal_frac": float(pre_mom_normal_frac.item()),
        "post_mom_normal_at_w_next_frac": float(post_mom_normal_frac.item()),
        "normal_to_tangent_leakage_ratio": float(normal_to_tangent_leakage_ratio.item()),
        "cos_gtan_mtan": _batch_cos(g_tan, m_tan_w),
        "cos_gtan_utan": _batch_cos(g_tan, ut),
        "cos_mtan_utan": _batch_cos(m_tan_w, ut),
        "cos_utan_retract": _batch_cos(-ut, delta_w),
        "retract_normal_frac": float(retract_normal_frac.item()),
    }


def riemannian_stiefel_tangent_transport_muon_step(
    weight: torch.Tensor,
    grad: torch.Tensor,
    momentum_buf: torch.Tensor,
    lr: float = DEFAULT_RIEMANNIAN_MUON_LR,
    beta: float = DEFAULT_RIEMANNIAN_MOMENTUM_BETA,
    ns_steps: int = DEFAULT_NS5_STEPS,
    tikhonov_lambda: float = 0.0,
    momentum_mode: StiefelMomentumMode = StiefelMomentumMode.RIEMANNIAN_TANGENT_TRANSPORTED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute 1 batched Stiefel-Muon step supporting `EUCLIDEAN_AMBIENT`, `RIEMANNIAN_TANGENT_TRANSPORTED`, and `RIEMANNIAN_COLUMN_SCALED_STIEFEL` (`<= 8` args)."""
    g_clean = torch.nan_to_num(grad.float(), nan=0.0, posinf=0.0, neginf=0.0)
    w_f32 = weight.float()
    m_f32 = momentum_buf.float()

    transposed = w_f32.size(-2) < w_f32.size(-1)
    if transposed:
        w_f32 = w_f32.transpose(-2, -1)
        g_clean = g_clean.transpose(-2, -1)
        m_f32 = m_f32.transpose(-2, -1)

    if momentum_mode in _COL_SCALED_STIEFEL_MODES:
        s_col = w_f32.norm(dim=-2, keepdim=True).clamp_min(TRACE_NORM_EPS)
        w_unit = w_f32 / s_col
        g_tan = project_real_stiefel_tangent(g_clean, w_unit)
        m_prev_tan = project_real_stiefel_tangent(m_f32, w_unit)
        m_updated = float(beta) * m_prev_tan + g_tan
        m_tan = g_tan + float(beta) * m_updated

        g_s_raw = (w_unit * g_clean).sum(dim=-2, keepdim=True)
        m_s_prev = (w_unit * m_f32).sum(dim=-2, keepdim=True)
        s_norm_sq = s_col.square().sum(dim=-1, keepdim=True).clamp_min(TRACE_NORM_EPS)
        g_s_tan = g_s_raw - (g_s_raw * s_col).sum(dim=-1, keepdim=True) / s_norm_sq * s_col
        m_s_prev_tan = m_s_prev - (m_s_prev * s_col).sum(dim=-1, keepdim=True) / s_norm_sq * s_col
        m_s_updated = float(beta) * m_s_prev_tan + g_s_tan
        u_s_raw = g_s_tan + float(beta) * m_s_updated
        col_fro_scale = (
            (m_tan.square().sum(dim=(-2, -1), keepdim=True) + u_s_raw.square().sum(dim=(-2, -1), keepdim=True))
            / float(w_unit.size(-1))
        ).sqrt().clamp_min(TRACE_NORM_EPS)
        u_s_unit = u_s_raw / col_fro_scale
    elif momentum_mode == StiefelMomentumMode.RIEMANNIAN_TANGENT_TRANSPORTED:
        w_unit = w_f32
        g_tan = project_real_stiefel_tangent(g_clean, w_unit)
        m_prev_tan = project_real_stiefel_tangent(m_f32, w_unit)
        m_updated = float(beta) * m_prev_tan + g_tan
        m_tan = g_tan + float(beta) * m_updated
    else:
        w_unit = w_f32
        m_updated = float(beta) * m_f32 + g_clean
        g_nest = g_clean + float(beta) * m_updated
        m_tan = project_real_stiefel_tangent(g_nest, w_unit)

    cols = w_unit.size(-1)
    u_raw = _newton_schulz5_tall_real(m_tan, steps=ns_steps, tikhonov_lambda=tikhonov_lambda)
    if w_unit.size(-2) != cols:
        eye_c = torch.eye(cols, device=w_unit.device, dtype=w_unit.dtype)
        u_raw = 0.5 * torch.matmul(u_raw, 3.0 * eye_c - torch.matmul(u_raw.transpose(-2, -1), u_raw))
    u_tan = project_real_stiefel_tangent(u_raw, w_unit)

    retract_scale = 1.0 / math.sqrt(1.0 + float(lr) * float(lr))
    w_unit_next = (w_unit - float(lr) * u_tan) * retract_scale
    eye_f32 = torch.eye(cols, device=w_unit_next.device, dtype=w_unit_next.dtype)
    w_unit_next = 0.5 * torch.matmul(
        w_unit_next, 3.0 * eye_f32 - torch.matmul(w_unit_next.transpose(-2, -1), w_unit_next)
    )

    if momentum_mode in _COL_SCALED_STIEFEL_MODES:
        col_lr = float(lr) * DEFAULT_COLUMN_SCALE_LR_RATIO
        s_cand = (s_col - col_lr * u_s_unit).clamp(DEFAULT_COLUMN_SCALE_MIN, DEFAULT_COLUMN_SCALE_MAX)
        s_rms = s_cand.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(TRACE_NORM_EPS)
        s_next = (s_cand / s_rms).clamp(DEFAULT_COLUMN_SCALE_MIN, DEFAULT_COLUMN_SCALE_MAX)
        s_next_rms = s_next.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(TRACE_NORM_EPS)
        s_next = s_next / s_next_rms
        w_next = w_unit_next * s_next
        s_next_sq = s_next.square().sum(dim=-1, keepdim=True).clamp_min(TRACE_NORM_EPS)
        m_s_stored = m_s_updated - (m_s_updated * s_next).sum(dim=-1, keepdim=True) / s_next_sq * s_next
        m_stored = project_real_stiefel_tangent(m_updated, w_unit_next) + w_unit_next * m_s_stored
    elif momentum_mode == StiefelMomentumMode.RIEMANNIAN_TANGENT_TRANSPORTED:
        w_next = w_unit_next
        m_stored = project_real_stiefel_tangent(m_updated, w_next)
    else:
        w_next = w_unit_next
        m_stored = m_updated

    if transposed:
        w_next = w_next.transpose(-2, -1)
        m_stored = m_stored.transpose(-2, -1)
        m_updated = m_updated.transpose(-2, -1)
        u_tan = u_tan.transpose(-2, -1)

    weight.copy_(w_next.to(weight.dtype))
    momentum_buf.copy_(m_stored.to(momentum_buf.dtype))
    return u_tan, m_updated


def euclidean_frobenius_normed_muon_step(
    weight: torch.Tensor,
    grad: torch.Tensor,
    momentum_buf: torch.Tensor,
    lr: float = DEFAULT_RIEMANNIAN_MUON_LR,
    beta: float = DEFAULT_RIEMANNIAN_MOMENTUM_BETA,
    ns_steps: int = DEFAULT_NS5_STEPS,
    tikhonov_lambda: float = DEFAULT_TIKHONOV_GRAM_LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute 1 batched Frobenius-normalized Euclidean Muon step capturing 100% of tangent + symmetric normal gradients (`<= 8` args)."""
    g_clean = torch.nan_to_num(grad.float(), nan=0.0, posinf=0.0, neginf=0.0)
    w_f32 = weight.float()
    m_f32 = momentum_buf.float()
    transposed = w_f32.size(-2) < w_f32.size(-1)
    if transposed:
        w_f32 = w_f32.transpose(-2, -1)
        g_clean = g_clean.transpose(-2, -1)
        m_f32 = m_f32.transpose(-2, -1)

    m_updated = float(beta) * m_f32 + g_clean
    u_ns = g_clean + float(beta) * m_updated
    u_ortho = _newton_schulz5_tall_real(u_ns, steps=ns_steps, tikhonov_lambda=tikhonov_lambda)
    cols = w_f32.size(-1)
    target_fro = math.sqrt(float(cols))
    eff_lr = float(lr) * SQRT2_ISOMETRIC_OFFDIAG_SCALE
    w_cand = w_f32 - eff_lr * u_ortho
    fro_norm = torch.linalg.matrix_norm(w_cand, ord="fro", dim=(-2, -1), keepdim=True).clamp_min(TRACE_NORM_EPS)
    w_next = w_cand * (target_fro / fro_norm)

    if transposed:
        w_next = w_next.transpose(-2, -1)
        m_updated = m_updated.transpose(-2, -1)
        u_ortho = u_ortho.transpose(-2, -1)
    weight.copy_(w_next.to(weight.dtype))
    momentum_buf.copy_(m_updated.to(momentum_buf.dtype))
    return u_ortho, m_updated


def riemannian_complex_kraus_stiefel_muon_step(
    a_re: torch.Tensor,
    a_im: torch.Tensor,
    grad_re: torch.Tensor,
    grad_im: torch.Tensor,
    mom_re: torch.Tensor,
    mom_im: torch.Tensor,
    lr: float = DEFAULT_UNITARY_GEODESIC_LR,
    tikhonov_lambda: float = DEFAULT_TIKHONOV_GRAM_LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute 1 batched Riemannian Complex Stiefel Muon step on $A \\in \\mathrm{St}_{\\mathbb{C}}(D, d)$ ($D \\ge d$, $A^\\dagger A = I_d$)."""
    riemannian_complex_tangent_transport_muon_step(
        a_re, a_im, grad_re, grad_im, mom_re, mom_im, lr=lr, momentum_mode=StiefelMomentumMode.EUCLIDEAN_AMBIENT
    )
    return a_re, a_im


def riemannian_complex_tangent_transport_muon_step(
    a_re: torch.Tensor,
    a_im: torch.Tensor,
    grad_re: torch.Tensor,
    grad_im: torch.Tensor,
    mom_re: torch.Tensor,
    mom_im: torch.Tensor,
    lr: float = DEFAULT_UNITARY_GEODESIC_LR,
    momentum_mode: StiefelMomentumMode = StiefelMomentumMode.EUCLIDEAN_AMBIENT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute 1 batched Complex Unitary/Stiefel Muon step supporting `EUCLIDEAN_AMBIENT` vs. `RIEMANNIAN_TANGENT_TRANSPORTED` (`<= 8` args)."""
    g_in_re = torch.nan_to_num(grad_re.float(), nan=0.0, posinf=0.0, neginf=0.0)
    g_in_im = torch.nan_to_num(grad_im.float(), nan=0.0, posinf=0.0, neginf=0.0)
    ar = a_re.float()
    ai = a_im.float()
    beta = DEFAULT_RIEMANNIAN_MOMENTUM_BETA

    if momentum_mode != StiefelMomentumMode.EUCLIDEAN_AMBIENT:
        g_tan_re, g_tan_im = project_complex_stiefel_tangent(g_in_re, g_in_im, ar, ai)
        m_pre_tan_re, m_pre_tan_im = project_complex_stiefel_tangent(mom_re.float(), mom_im.float(), ar, ai)
        mom_re.copy_(float(beta) * m_pre_tan_re + g_tan_re)
        mom_im.copy_(float(beta) * m_pre_tan_im + g_tan_im)
        g_re = g_tan_re + float(beta) * mom_re.float()
        g_im = g_tan_im + float(beta) * mom_im.float()
    else:
        mom_re.mul_(beta).add_(g_in_re)
        mom_im.mul_(beta).add_(g_in_im)
        g_re = g_in_re + float(beta) * mom_re.float()
        g_im = g_in_im + float(beta) * mom_im.float()

    t_re, t_im = project_complex_stiefel_tangent(g_re, g_im, ar, ai)
    z_re, z_im = _complex_newton_schulz5(t_re, t_im, steps=DEFAULT_NS5_STEPS, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA)
    tan_re, tan_im = project_complex_stiefel_tangent(z_re, z_im, ar, ai)

    scale = 1.0 / math.sqrt(1.0 + float(lr) * float(lr))
    nx_re = (ar - float(lr) * tan_re) * scale
    nx_im = (ai - float(lr) * tan_im) * scale

    eye = torch.eye(ar.size(-1), device=ar.device, dtype=ar.dtype)
    gram_re = torch.matmul(nx_re.transpose(-2, -1), nx_re) + torch.matmul(nx_im.transpose(-2, -1), nx_im)
    gram_im = torch.matmul(nx_re.transpose(-2, -1), nx_im) - torch.matmul(nx_im.transpose(-2, -1), nx_re)
    tm_re = 3.0 * eye - gram_re
    tm_im = -gram_im
    pol_re = 0.5 * (torch.matmul(nx_re, tm_re) - torch.matmul(nx_im, tm_im))
    pol_im = 0.5 * (torch.matmul(nx_re, tm_im) + torch.matmul(nx_im, tm_re))

    if momentum_mode != StiefelMomentumMode.EUCLIDEAN_AMBIENT:
        tr_m_re, tr_m_im = project_complex_stiefel_tangent(mom_re.float(), mom_im.float(), pol_re, pol_im)
        mom_re.copy_(tr_m_re.to(mom_re.dtype))
        mom_im.copy_(tr_m_im.to(mom_im.dtype))

    a_re.copy_(pol_re.to(a_re.dtype))
    a_im.copy_(pol_im.to(a_im.dtype))
    return a_re, a_im


def apply_detached_tsallis_entropy_gate(
    packed_rho_out: torch.Tensor,
    psi_u_flat: torch.Tensor,
    krylov_64: torch.Tensor,
    purity: torch.Tensor,
    entropy_gate_params: torch.Tensor,
) -> torch.Tensor:
    """Route coherent wavepackets vs. mixed density/Krylov features using detached purity $\\mathcal{P}_{\\text{det}} = \\operatorname{Tr}(\\rho^2).\\text{detach}()$."""
    work_dtype = packed_rho_out.dtype
    d_sq = packed_rho_out.size(-1)
    v_branch_dim = psi_u_flat.size(-1)
    c_dim = min(v_branch_dim, d_sq)

    p_det = purity.detach().to(work_dtype).unsqueeze(-1)
    p_delta = p_det - float(DEFAULT_TSALLIS_PURITY_CENTER)
    tsallis_delta = -p_delta

    g_params = entropy_gate_params.to(work_dtype)
    slope_coh = g_params[:, 0].view(1, -1, 1, 1)
    slope_mix = g_params[:, 1].view(1, -1, 1, 1)
    slope_conf = g_params[:, 2].view(1, -1, 1, 1)

    w_coh = math.sqrt(2.0) * torch.sigmoid(slope_coh * p_delta)
    w_mix = math.sqrt(2.0) * torch.sigmoid(slope_mix * tsallis_delta)
    w_conf = (1.0 + 0.5 * torch.tanh(slope_conf) * p_delta).clamp(0.5, 1.5)

    rho_gated = w_mix * packed_rho_out
    psi_gated = w_coh * psi_u_flat
    rho_plus_psi = torch.cat(
        [
            INV_SQRT2_ISOMETRIC_SCALE * (rho_gated[..., :c_dim] + psi_gated[..., :c_dim]),
            rho_gated[..., c_dim:],
        ],
        dim=-1,
    )
    krylov_gated = w_mix * krylov_64
    return w_conf * torch.cat([rho_plus_psi, krylov_gated], dim=-1)


def statmech_purify_evolve_uhlmann_krylov_impl(
    packed_all: torch.Tensor,
    op_re: torch.Tensor,
    op_im: torch.Tensor,
    re_pack_mask: torch.Tensor,
    im_pack_mask: torch.Tensor,
    purify_logits_h3: torch.Tensor | None = None,
    entropy_gate_params: torch.Tensor | None = None,
    modes: tuple[str, str, str] = (
        PurificationMode.QUADRATIC_MCWEENY.value,
        QuantumChannelMode.UNITARY_U16.value,
        EntropyGateMode.NONE.value,
    ),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack attention output, run purification + Unitary/Kraus evolution, and extract Uhlmann Rank-2 Krylov features (`<= 8` args)."""
    work_dtype = packed_all.dtype
    d_h = op_re.size(-1)
    d_sq = d_h * d_h
    base_pack_dim = d_sq + 2 * d_h
    purify_mode = PurificationMode(modes[0])
    channel_mode = QuantumChannelMode(modes[1])
    gate_mode = EntropyGateMode(modes[2])
    use_clean_iso = gate_mode in (
        EntropyGateMode.ISOMETRIC_NORMALIZED_BRANCHES,
        EntropyGateMode.ISOMETRIC_NORMALIZED_PLUS_TSALLIS,
    )

    packed_rho0 = packed_all[..., :d_sq]
    v_bar_re = packed_all[..., d_sq : d_sq + d_h]
    v_bar_im = packed_all[..., d_sq + d_h : base_pack_dim]

    rho0_re, rho0_im = unpack_hermitian_isometric(packed_rho0, d_h)
    rho1_re, rho1_im, purity, eff_rank = purify_density_statmech(
        rho0_re, rho0_im, mode=purify_mode, purify_logits_h3=purify_logits_h3, eps=TRACE_NORM_EPS
    )

    u0_re = op_re[:, :d_h, :].to(work_dtype).unsqueeze(0).unsqueeze(2)
    u0_im = op_im[:, :d_h, :].to(work_dtype).unsqueeze(0).unsqueeze(2)
    ut_re = u0_re.transpose(-2, -1)
    ut_im = u0_im.transpose(-2, -1)

    if channel_mode == QuantumChannelMode.CPTP_KRAUS_RANK_K and op_re.size(-2) > d_h:
        rho_out_re, rho_out_im = evolve_head_cptp_kraus_functional(rho1_re, rho1_im, op_re, op_im)
    elif use_clean_iso:
        rho_out_re, rho_out_im = evolve_head_unitary_traceless_functional(
            rho1_re, rho1_im, op_re[:, :d_h, :], op_im[:, :d_h, :]
        )
    else:
        rho_out_re, rho_out_im = evolve_head_unitary_functional(rho1_re, rho1_im, op_re[:, :d_h, :], op_im[:, :d_h, :])

    packed_rho_trace1 = (
        rho_out_re * re_pack_mask.to(work_dtype) + rho_out_im * im_pack_mask.to(work_dtype)
    ).flatten(start_dim=-2, end_dim=-1)

    if use_clean_iso and not (channel_mode == QuantumChannelMode.CPTP_KRAUS_RANK_K and op_re.size(-2) > d_h):
        inv_rho_rms = torch.rsqrt(purity.to(work_dtype).unsqueeze(-1) / float(d_sq) + TRACE_NORM_EPS)
        packed_rho_out = packed_rho_trace1 * inv_rho_rms

        lam1, lam2, phi1_0_re, phi1_0_im, phi2_0_re, phi2_0_im = extract_rank2_krylov_eigenpair_uhlmann(
            rho1_re, rho1_im, v_bar_re, v_bar_im, eps=DEFAULT_UHLMANN_PHASE_EPS
        )
        s1 = torch.sqrt(lam1.clamp_min(DEFAULT_UHLMANN_PHASE_EPS)).unsqueeze(-1).to(work_dtype)
        s2 = torch.sqrt(lam2.clamp_min(DEFAULT_UHLMANN_PHASE_EPS)).unsqueeze(-1).to(work_dtype)
        phi_0_re = torch.stack([s1 * phi1_0_re.to(work_dtype), s2 * phi2_0_re.to(work_dtype)], dim=-2)
        phi_0_im = torch.stack([s1 * phi1_0_im.to(work_dtype), s2 * phi2_0_im.to(work_dtype)], dim=-2)
        krylov_0_rms_inv = torch.rsqrt(
            (
                phi_0_re.float().square().sum(dim=(-2, -1), keepdim=True)
                + phi_0_im.float().square().sum(dim=(-2, -1), keepdim=True)
            )
            / float(4 * d_h)
            + TRACE_NORM_EPS
        ).to(work_dtype)
        phi_0_re = phi_0_re * krylov_0_rms_inv
        phi_0_im = phi_0_im * krylov_0_rms_inv
        phi_u_re = torch.matmul(phi_0_re, ut_re) - torch.matmul(phi_0_im, ut_im)
        phi_u_im = torch.matmul(phi_0_re, ut_im) + torch.matmul(phi_0_im, ut_re)
        krylov_64 = torch.cat(
            [phi_u_re[..., 0, :], phi_u_im[..., 0, :], phi_u_re[..., 1, :], phi_u_im[..., 1, :]],
            dim=-1,
        )
    else:
        packed_rho_out = F.rms_norm(packed_rho_trace1, (d_sq,), eps=TRACE_NORM_EPS)
        v_u_re = torch.matmul(v_bar_re.unsqueeze(-2), ut_re).squeeze(-2) - torch.matmul(
            v_bar_im.unsqueeze(-2), ut_im
        ).squeeze(-2)
        v_u_im = torch.matmul(v_bar_re.unsqueeze(-2), ut_im).squeeze(-2) + torch.matmul(
            v_bar_im.unsqueeze(-2), ut_re
        ).squeeze(-2)
        lam1, lam2, phi1_re, phi1_im, phi2_re, phi2_im = extract_rank2_krylov_eigenpair_uhlmann(
            rho_out_re, rho_out_im, v_u_re, v_u_im, eps=DEFAULT_UHLMANN_PHASE_EPS
        )
        s1 = torch.sqrt(lam1.clamp_min(DEFAULT_UHLMANN_PHASE_EPS)).unsqueeze(-1).to(work_dtype)
        s2 = torch.sqrt(lam2.clamp_min(DEFAULT_UHLMANN_PHASE_EPS)).unsqueeze(-1).to(work_dtype)
        krylov_raw = torch.cat(
            [
                s1 * phi1_re.to(work_dtype),
                s1 * phi1_im.to(work_dtype),
                s2 * phi2_re.to(work_dtype),
                s2 * phi2_im.to(work_dtype),
            ],
            dim=-1,
        )
        krylov_64 = F.rms_norm(krylov_raw, (4 * d_h,), eps=TRACE_NORM_EPS)

    if packed_all.size(-1) > base_pack_dim:
        psi_tail = packed_all[..., base_pack_dim:]
        v_branch_dim = psi_tail.size(-1)
        v_rank = v_branch_dim // (2 * d_h)
        if channel_mode == QuantumChannelMode.CPTP_KRAUS_RANK_K and op_re.size(-2) == v_rank * d_h:
            psi_bar = psi_tail.view(*psi_tail.shape[:-1], v_rank, 2, d_h)
            k_re = op_re.to(work_dtype).view(1, op_re.size(0), 1, v_rank, d_h, d_h).transpose(-2, -1)
            k_im = op_im.to(work_dtype).view(1, op_im.size(0), 1, v_rank, d_h, d_h).transpose(-2, -1)
            p_re_col = psi_bar[..., 0, :].unsqueeze(-2)
            p_im_col = psi_bar[..., 1, :].unsqueeze(-2)
            psi_u_re = (torch.matmul(p_re_col, k_re) - torch.matmul(p_im_col, k_im)).squeeze(-2)
            psi_u_im = (torch.matmul(p_re_col, k_im) + torch.matmul(p_im_col, k_re)).squeeze(-2)
            psi_u_flat = F.rms_norm(
                torch.cat([psi_u_re, psi_u_im], dim=-1).flatten(start_dim=-2, end_dim=-1),
                (v_branch_dim,),
                eps=TRACE_NORM_EPS,
            )
        elif use_clean_iso:
            psi_bar = F.rms_norm(psi_tail, (v_branch_dim,), eps=TRACE_NORM_EPS).view(
                *psi_tail.shape[:-1], v_rank, 2, d_h
            )
            psi_u_re = torch.matmul(psi_bar[..., 0, :], ut_re) - torch.matmul(psi_bar[..., 1, :], ut_im)
            psi_u_im = torch.matmul(psi_bar[..., 0, :], ut_im) + torch.matmul(psi_bar[..., 1, :], ut_re)
            psi_u_flat = torch.cat([psi_u_re, psi_u_im], dim=-1).flatten(start_dim=-2, end_dim=-1)
        else:
            psi_bar = psi_tail.view(*psi_tail.shape[:-1], v_rank, 2, d_h)
            psi_u_re = torch.matmul(psi_bar[..., 0, :], ut_re) - torch.matmul(psi_bar[..., 1, :], ut_im)
            psi_u_im = torch.matmul(psi_bar[..., 0, :], ut_im) + torch.matmul(psi_bar[..., 1, :], ut_re)
            psi_u_flat = F.rms_norm(
                torch.cat([psi_u_re, psi_u_im], dim=-1).flatten(start_dim=-2, end_dim=-1),
                (v_branch_dim,),
                eps=TRACE_NORM_EPS,
            )
        if (
            gate_mode in (EntropyGateMode.DETACHED_TSALLIS_PURITY_GATE, EntropyGateMode.ISOMETRIC_NORMALIZED_PLUS_TSALLIS)
            and entropy_gate_params is not None
        ):
            obs_320 = apply_detached_tsallis_entropy_gate(
                packed_rho_out, psi_u_flat, krylov_64, purity, entropy_gate_params
            )
        else:
            c_dim = min(v_branch_dim, d_sq)
            rho_plus_psi = torch.cat(
                [
                    INV_SQRT2_ISOMETRIC_SCALE * (packed_rho_out[..., :c_dim] + psi_u_flat[..., :c_dim]),
                    packed_rho_out[..., c_dim:],
                ],
                dim=-1,
            )
            obs_320 = torch.cat([rho_plus_psi, krylov_64], dim=-1)
    else:
        obs_320 = torch.cat([packed_rho_out, krylov_64], dim=-1)
    return obs_320, packed_rho_out, packed_rho_trace1, lam1, lam2, purity, eff_rank


_COMPILED_STATMECH_HELPERS: dict[tuple[str, str, str], Any] = {}


def compiled_statmech_purify_evolve_uhlmann_krylov_step(
    packed_all: torch.Tensor,
    op_re: torch.Tensor,
    op_im: torch.Tensor,
    re_pack_mask: torch.Tensor,
    im_pack_mask: torch.Tensor,
    purify_logits_h3: torch.Tensor | None = None,
    entropy_gate_params: torch.Tensor | None = None,
    modes: tuple[str, str, str, bool] = (
        PurificationMode.QUADRATIC_MCWEENY.value,
        QuantumChannelMode.UNITARY_U16.value,
        EntropyGateMode.NONE.value,
        False,
    ),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute `statmech_purify_evolve_uhlmann_krylov_impl` with mode-specialized `torch.compile` caching (`<= 8` args)."""
    mode_key = (str(modes[0]), str(modes[1]), str(modes[2]))
    use_compile = bool(modes[3])
    if use_compile and packed_all.is_cuda:
        if mode_key not in _COMPILED_STATMECH_HELPERS:
            _COMPILED_STATMECH_HELPERS[mode_key] = torch.compile(
                statmech_purify_evolve_uhlmann_krylov_impl, dynamic=False
            )
        return _COMPILED_STATMECH_HELPERS[mode_key](
            packed_all, op_re, op_im, re_pack_mask, im_pack_mask, purify_logits_h3, entropy_gate_params, mode_key
        )
    return statmech_purify_evolve_uhlmann_krylov_impl(
        packed_all, op_re, op_im, re_pack_mask, im_pack_mask, purify_logits_h3, entropy_gate_params, mode_key
    )


__all__ = [
    "DEFAULT_COLUMN_SCALE_LR_RATIO",
    "DEFAULT_COLUMN_SCALE_MAX",
    "DEFAULT_COLUMN_SCALE_MIN",
    "DEFAULT_KRAUS_CHANNEL_RANK",
    "DEFAULT_KRAUS_DOMINANT_WEIGHT",
    "DEFAULT_STATMECH_MCWEENY_STEPS",
    "DEFAULT_TSALLIS_PURITY_CENTER",
    "EntropyGateMode",
    "PurificationMode",
    "QuantumChannelMode",
    "StiefelMomentumMode",
    "apply_detached_tsallis_entropy_gate",
    "causal_born_p3_weights",
    "causal_born_p3_weights_impl",
    "compiled_statmech_purify_evolve_uhlmann_krylov_step",
    "compute_real_stiefel_vector_diagnostics",
    "cubic_fermi_dirac_purification_step",
    "evolve_head_cptp_kraus_functional",
    "evolve_head_unitary_functional",
    "evolve_head_unitary_traceless_functional",
    "euclidean_frobenius_normed_muon_step",
    "initialize_complex_kraus_stiefel_pair",
    "learnable_temp_purification_step",
    "project_complex_stiefel_tangent",
    "project_real_stiefel_tangent",
    "purify_density_statmech",
    "riemannian_complex_kraus_stiefel_muon_step",
    "riemannian_complex_tangent_transport_muon_step",
    "riemannian_stiefel_tangent_transport_muon_step",
    "statmech_purify_evolve_uhlmann_krylov_impl",
]
