r"""Wave 67 (`EXPERIMENT_PLAN_087.md`): Selective Exact Quotienting, Strictly Read-Only Riemannian Shadow EMA, & Wide `256D` Subspace Readout.

Mathematical & Architectural Overview
-------------------------------------
Resolves the over-projection ("frozen capacity") and mid-training rollback issues discovered in the Wave 66 (`200`-step)
horizon analysis while retaining 100% of the exact gauge-quotient benefits:

1. **Exact $\mathfrak{su}(16)$ Horizontal Lift for Gauge-Locked MUB Tomography ($U_h^{(2)} \equiv F_{16} U_h^{(1)}$)**:
   - Tying $U_h^{(2)} = F_{16} U_h^{(1)}$ via the unitary $16 \times 16$ DFT matrix guarantees exact mutually unbiased
     overlap $|\langle c_{h,m}^{(1)} \mid c_{h,n}^{(2)}\rangle|^2 \equiv \frac{1}{16}$ (`MUBErr = 0.00e+00`) at every step.
   - Because relative row phases of $U_h^{(1)}$ do not commute with $F_{16}$ (and actively steer the second MUB basis $U_h^{(2)}$),
     only the 1D global phase center $U_h^{(1)} \mapsto e^{i\phi_h} U_h^{(1)}$ ($\operatorname{Tr}(\operatorname{Im}(U_h^\dagger \xi_h)) = 0$)
     is a true gauge symmetry of the tied pair $(U_h^{(1)}, F_{16} U_h^{(1)})$. Projecting onto $\mathfrak{su}(16) = \mathfrak{u}(16)/\mathfrak{u}(1)$
     preserves `100.0%` of the physical gradient (`raw_vertical_grad_leak_frac = 0.00e+00`).

2. **Exact Lyapunov $\mathfrak{u}(R)$ Horizontal Lift for Rank-$R$ CPTP Kraus Operators ($\mathbf{K}_h \in \operatorname{St}_{\mathbb{C}}(R d_h, d_h)$)**:
   - Solves the $R \times R$ Hermitian Lyapunov equation $\frac{1}{2}(A_h G_K + G_K A_h) = S_{\text{skew}}(\Xi_h)$ on the branch Gram matrix
     $G_K = K_{\text{flat}} K_{\text{flat}}^\dagger$, stripping `100%` of the vertical $\mathfrak{u}(R)$ branch-mixing gauge fiber while keeping
     $\Xi_h^{\text{horiz}} = \Xi_h - A_h K_{\text{flat}}$ inside $T_{\mathbf{K}_h}\operatorname{St}_{\mathbb{C}}(R d_h, d_h)$.

3. **Spectral-Degeneracy-Weighted Post-Newton–Schulz Co-Rotation Filter for Coupled SVD-Stiefel Pairs $(U, V)$**:
   - For $W = V \operatorname{diag}(\vec{\sigma}) U^\top$, simultaneous right-rotation $(U e^{\epsilon\Omega}, V e^{\epsilon\Omega})$ produces
     $(V^\top \delta W U)_{ij} = (\sigma_j - \sigma_i)\Omega_{ij}$. Raw gradients retain `100.0%` of their Stiefel tangent components
     (stripping only exact degenerate pairs $|\sigma_i - \sigma_j| \le 10^{-6}$ where $\Omega_{\text{sym}, ij}(\nabla\mathcal{L}) \equiv 0$),
     while post-Newton–Schulz artificial co-rotation spin is filtered by the Lorentzian spectral degeneracy kernel
     $\Gamma_{ij}(\vec{\sigma}) = \tau_\sigma^2 / (\tau_\sigma^2 + (\sigma_i - \sigma_j)^2)$.

4. **Tied-Subspace Excess Common-Fiber Filter & Widened `256D` Readout Frame ($U_{\text{in}}, U_{\text{out}} \in \operatorname{St}(512, 256)$)**:
   - Preserves `100.0%` of the raw Stiefel gradients on $(U_{\text{in}}, U_{\text{out}})$ (keeping relative input-to-output transitions
     $\Omega_{\text{in}} - \Omega_{\text{out}} \in \mathfrak{so}(d_{\text{voc}})$ and vocabulary alignment unblocked) while damping only
     the post-Newton–Schulz excess common-fiber spin $\Delta \Omega_{\text{common}}^{\text{NS}}$.
   - Widens `subspace_dim` from `128 -> 256` (`W67_S2`, `W67_S3`) to eliminate the rank-128 readout bottleneck, and scales trunk capacity
     to `kraus_rank = 4, mlp_hidden = 1536` (`W67_S3`).

5. **Strictly Read-Only Riemannian Shadow EMA (`ReadOnlyQuenchRiemannianShadowEMA`)**:
   - Zero mid-training parameter rollbacks (`triggered = False`) and zero `lr_scale` damping (`lr_scale = 1.00x` throughout the plateau).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import gc
import math
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_ckpt

from qdlblocks.benchmarks.experiment_logger import ExperimentIntentSpec, StandardizedExperimentLogger
from qdlblocks.benchmarks.explicit_density_tomographic_512d_lab import (
    DEFAULT_INIT_FILTER_SCALE, DEFAULT_INIT_TOMO_SCALE, DEFAULT_INIT_VACUUM_REST_MASS,
    DEFAULT_TOK_NGRAM_BUCKETS, DEFAULT_TOMO_EMBED_DIM, DEFAULT_TOMO_HEAD_COMPLEX_DIM,
    DEFAULT_TOMO_MLP_HIDDEN, DEFAULT_TOMO_NUM_HEADS, DEFAULT_TOMO_NUM_LAYERS,
    DEFAULT_TOMO_VACUUM_DIM, _complex_hermitian_mv, build_layer0_explicit_outer_density,
)
from qdlblocks.benchmarks.horizontal_grassmann_gauge_free_lab import (
    _retract_complex_tall_stiefel_step, _retract_real_tall_stiefel_step,
    compute_coupled_svd_joint_spin_fraction, compute_gauge_locked_mub_probe_expectations,
    compute_kraus_branch_gauge_fraction, compute_mub_unbiasedness_max_error,
    project_kraus_branch_gauge_horizontal,
)
from qdlblocks.benchmarks.kraus_mub_thermostat_8l_lab import (
    DEFAULT_ANALYTICAL_BETA_TARGET, DEFAULT_KRAUS_BRANCH_RANK, DEFAULT_THERMOSTAT_EMA_DECAY,
    DEFAULT_THERMOSTAT_RETRACT_CADENCE, RiemannianShadowEMAThermostat, _KrausDensityHighwayBlock,
    build_initial_kraus_stiefel_operators, build_mub_two_basis_unitary_probes,
    compute_analytical_parseval_beta, compute_mub_two_basis_probe_expectations,
)
from qdlblocks.benchmarks.quantum_subsystem_vocab_lab import (
    DEFAULT_SHARED_VOCAB_SUBSPACE_DIM, SharedStiefelSubspaceFrameBoundary,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import (
    DEFAULT_SHALLOW_NGRAM_LR, DEFAULT_SHALLOW_READOUT_LR, DEFAULT_SHALLOW_SCALAR_LR,
    DEFAULT_SHALLOW_STIEFEL_LR, DEFAULT_SHALLOW_WTE_LR, LongSeqFineWebStream,
)
from qdlblocks.layers.impure_density_mirror_block import (
    DEFAULT_NS5_STEPS, DEFAULT_TIKHONOV_GRAM_LAMBDA, _complex_newton_schulz5,
    _newton_schulz5_tall_real, complex_l2_normalize, unpack_hermitian_isometric,
)
from qdlblocks.layers.quantum_statmech_channels import (
    DEFAULT_MUON_GRAD_NORM_GATE_TAU, project_complex_stiefel_tangent, project_real_stiefel_tangent,
)
from qdlblocks.layers.quantum_svd_stiefel_graft import BiorthogonalSVDStiefelProjection
from qdlblocks.models.compiled_leaderboard_classical_gpt import NativeLeaderboardRotary
from qdlblocks.optim.vocab_manifold_muon import (
    BatchSubspaceTikhonovGramMuon, RiemannianGeodesicScalarAndProbeOptimizer,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 — Synchronized with EXPERIMENT_PLAN_087.md)
# ==============================================================================
DEFAULT_NARROW_SUBSPACE_DIM: int = DEFAULT_SHARED_VOCAB_SUBSPACE_DIM
DEFAULT_WIDE_SUBSPACE_DIM: int = 256
DEFAULT_WIDE_KRAUS_RANK: int = 4
DEFAULT_WIDE_MLP_HIDDEN: int = 1536
DEFAULT_SVD_DEGENERACY_TAU: float = 0.05
DEFAULT_COMMON_BRIDGE_EXCESS_DAMPING: float = 0.50
DEFAULT_QUENCH_FRACTION: float = 0.15
DEFAULT_SELECTIVE_MOMENTUM_BETA: float = 0.92


class Wave67SelectiveVariantMode(str, Enum):
    """Ablation variants for `EXPERIMENT_PLAN_087.md` (`Wave 67`, Rule 3)."""

    W67_S0_AMBIENT_RATCHET_BASELINE = "W67_S0_AMBIENT_RATCHET_BASELINE"
    W67_S1_SELECTIVE_EXACT_QUOTIENT_128D = "W67_S1_SELECTIVE_EXACT_QUOTIENT_128D"
    W67_S2_SELECTIVE_QUOTIENT_256D_READOUT = "W67_S2_SELECTIVE_QUOTIENT_256D_READOUT"
    W67_S3_SELECTIVE_QUOTIENT_256D_WIDE_TRUNK = "W67_S3_SELECTIVE_QUOTIENT_256D_WIDE_TRUNK"


@dataclass(frozen=True)
class Wave67SelectiveQuotientSpec:
    r"""Configuration for `SelectiveExactQuotientGPT` (`<= 8` fields, Rule 2)."""

    variant_mode: Wave67SelectiveVariantMode = Wave67SelectiveVariantMode.W67_S1_SELECTIVE_EXACT_QUOTIENT_128D
    num_layers: int = DEFAULT_TOMO_NUM_LAYERS
    kraus_rank: int = DEFAULT_KRAUS_BRANCH_RANK
    mlp_hidden: int = DEFAULT_TOMO_MLP_HIDDEN
    subspace_dim: int = DEFAULT_NARROW_SUBSPACE_DIM
    enable_selective_quotient: bool = True
    enable_read_only_shadow_ema: bool = True
    seed: int = 1337


def build_wave67_variant_spec(
    variant: Wave67SelectiveVariantMode | str,
    num_layers: int = DEFAULT_TOMO_NUM_LAYERS,
    seed: int = 1337,
) -> Wave67SelectiveQuotientSpec:
    """Resolve `Wave67SelectiveQuotientSpec` from `Wave67SelectiveVariantMode` (`<= 8` args)."""
    mode = Wave67SelectiveVariantMode(variant)
    if mode == Wave67SelectiveVariantMode.W67_S0_AMBIENT_RATCHET_BASELINE:
        return Wave67SelectiveQuotientSpec(
            variant_mode=mode, num_layers=num_layers, kraus_rank=DEFAULT_KRAUS_BRANCH_RANK,
            mlp_hidden=DEFAULT_TOMO_MLP_HIDDEN, subspace_dim=DEFAULT_NARROW_SUBSPACE_DIM,
            enable_selective_quotient=False, enable_read_only_shadow_ema=False, seed=seed,
        )
    if mode == Wave67SelectiveVariantMode.W67_S1_SELECTIVE_EXACT_QUOTIENT_128D:
        return Wave67SelectiveQuotientSpec(
            variant_mode=mode, num_layers=num_layers, kraus_rank=DEFAULT_KRAUS_BRANCH_RANK,
            mlp_hidden=DEFAULT_TOMO_MLP_HIDDEN, subspace_dim=DEFAULT_NARROW_SUBSPACE_DIM,
            enable_selective_quotient=True, enable_read_only_shadow_ema=True, seed=seed,
        )
    if mode == Wave67SelectiveVariantMode.W67_S2_SELECTIVE_QUOTIENT_256D_READOUT:
        return Wave67SelectiveQuotientSpec(
            variant_mode=mode, num_layers=num_layers, kraus_rank=DEFAULT_KRAUS_BRANCH_RANK,
            mlp_hidden=DEFAULT_TOMO_MLP_HIDDEN, subspace_dim=DEFAULT_WIDE_SUBSPACE_DIM,
            enable_selective_quotient=True, enable_read_only_shadow_ema=True, seed=seed,
        )
    return Wave67SelectiveQuotientSpec(
        variant_mode=mode, num_layers=num_layers, kraus_rank=DEFAULT_WIDE_KRAUS_RANK,
        mlp_hidden=DEFAULT_WIDE_MLP_HIDDEN, subspace_dim=DEFAULT_WIDE_SUBSPACE_DIM,
        enable_selective_quotient=True, enable_read_only_shadow_ema=True, seed=seed,
    )


# ==============================================================================
# 1. Selective Exact Quotient Projections (100% Horizontal Gradient Retention)
# ==============================================================================
def project_su_dh_mub_horizontal(
    v_re: torch.Tensor, v_im: torch.Tensor, u_re: torch.Tensor, u_im: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Project `(v_re, v_im)` onto $T_{U_h}SU(d_h)$ (`\operatorname{Tr}(\operatorname{Im}(U_h^\dagger \xi_h)) = 0`, `<= 8` args).

    When the second MUB basis is gauge-locked via $U_h^{(2)} \equiv F_{16} U_h^{(1)}$, only the 1D global phase center
    $U_h^{(1)} \mapsto e^{i\phi_h} U_h^{(1)}$ leaves all 32 rank-1 MUB projectors identically invariant. Stripping
    only the trace $\bar{\phi}_h = \frac{1}{d_h}\operatorname{Tr}(\operatorname{Im}(U_h^\dagger \xi_h))$ preserves 100.0%
    of the 255 physical $\mathfrak{su}(16)$ gradient directions per head (`raw_vertical_grad_leak_frac = 0.00e+00`).
    """
    ut_re = u_re.transpose(-2, -1)
    ut_im = -u_im.transpose(-2, -1)
    p_re = torch.matmul(ut_re, v_re) - torch.matmul(ut_im, v_im)
    p_im = torch.matmul(ut_re, v_im) + torch.matmul(ut_im, v_re)
    omega_re = 0.5 * (p_re - p_re.transpose(-2, -1))
    omega_im = 0.5 * (p_im + p_im.transpose(-2, -1))
    d_h = omega_im.size(-1)
    mean_tr_im = torch.diagonal(omega_im, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
    eye_d = torch.eye(d_h, device=omega_im.device, dtype=omega_im.dtype)
    omega_im_su = omega_im - mean_tr_im * eye_d
    h_re = torch.matmul(u_re, omega_re) - torch.matmul(u_im, omega_im_su)
    h_im = torch.matmul(u_re, omega_im_su) + torch.matmul(u_im, omega_re)
    return h_re, h_im


def compute_su_dh_global_phase_fraction(
    v_re: torch.Tensor, v_im: torch.Tensor, u_re: torch.Tensor, u_im: torch.Tensor
) -> float:
    r"""Measure fraction of tangent norm residing in the 1D global $U(1)$ phase fiber $\frac{1}{\sqrt{d_h}}|\operatorname{Tr}(\operatorname{Im}(U_h^\dagger \xi_h))| / \|\xi_h\|_F$ (`<= 8` args)."""
    ut_re = u_re.float().transpose(-2, -1)
    ut_im = -u_im.float().transpose(-2, -1)
    vr, vi = v_re.float(), v_im.float()
    p_im = torch.matmul(ut_re, vi) + torch.matmul(ut_im, vr)
    omega_im = 0.5 * (p_im + p_im.transpose(-2, -1))
    d_h = float(omega_im.size(-1))
    tr_im = torch.diagonal(omega_im, dim1=-2, dim2=-1).sum(dim=-1).abs() / math.sqrt(d_h)
    den = (vr.square().sum(dim=(-2, -1)) + vi.square().sum(dim=(-2, -1))).sqrt().clamp_min(1e-8)
    return float((tr_im / den).mean().item())


def compute_lorentzian_degeneracy_mask(
    sigma: torch.Tensor, tau: float = DEFAULT_SVD_DEGENERACY_TAU
) -> torch.Tensor:
    r"""Compute $\Gamma_{ij}(\vec{\sigma}) = \frac{\tau^2}{\tau^2 + (\sigma_i - \sigma_j)^2} \in (0, 1]$ (`<= 8` args)."""
    sig = sigma.float()
    diff_sq = (sig.unsqueeze(-1) - sig.unsqueeze(-2)).square()
    tau_sq = float(tau) * float(tau)
    return tau_sq / (tau_sq + diff_sq)


def project_coupled_svd_raw_exact_degenerate(
    xi_u: torch.Tensor, xi_v: torch.Tensor, u_mat: torch.Tensor, v_mat: torch.Tensor, sigma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Project `(xi_u, xi_v)` onto $T_U\operatorname{St} \times T_V\operatorname{St}$ and strip only exact degenerate co-rotations $|\sigma_i - \sigma_j| \le 10^{-6}$ (`<= 8` args).

    Because $(V^\top \delta W U)_{ij} = (\sigma_j - \sigma_i)\Omega_{ij}$, any pair $(i, j)$ with $|\sigma_i - \sigma_j| \le 10^{-6}$
    has $\Omega_{\text{sym}, ij}(\nabla \mathcal{L}) \equiv 0$ by the chain rule, so this projection preserves 100.0%
    of the physical gradient for both isotropic ($\vec{\sigma} = \vec{1}$) and split ($\sigma_i \neq \sigma_j$) spectra.
    """
    tu = project_real_stiefel_tangent(xi_u, u_mat)
    tv = project_real_stiefel_tangent(xi_v, v_mat)
    ut_tu = torch.matmul(u_mat.transpose(-2, -1), tu)
    vt_tv = torch.matmul(v_mat.transpose(-2, -1), tv)
    skew_u = 0.5 * (ut_tu - ut_tu.transpose(-2, -1))
    skew_v = 0.5 * (vt_tv - vt_tv.transpose(-2, -1))
    omega_sym = 0.5 * (skew_u + skew_v)
    deg_mask = ((sigma.float().unsqueeze(-1) - sigma.float().unsqueeze(-2)).abs() <= 1e-6).to(omega_sym.dtype)
    omega_deg = omega_sym * deg_mask
    return tu - torch.matmul(u_mat, omega_deg), tv - torch.matmul(v_mat, omega_deg)


def filter_coupled_svd_post_ns_corotation(
    zu_ns: torch.Tensor, zv_ns: torch.Tensor, xi_u_ref: torch.Tensor, xi_v_ref: torch.Tensor,
    u_mat: torch.Tensor, v_mat: torch.Tensor, sigma: torch.Tensor, tau: float = DEFAULT_SVD_DEGENERACY_TAU,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Combine Horizontal SVD-Stiefel step `(zu_ns, zv_ns)` ($\Omega_{\text{sym}} = 0$) with Lorentzian split-spectrum co-rotation $(1 - \Gamma_{ij}(\vec{\sigma}))\Omega_{\text{ref}, ij}$ (`<= 8` args)."""
    tu_ns = project_real_stiefel_tangent(zu_ns, u_mat)
    tv_ns = project_real_stiefel_tangent(zv_ns, v_mat)
    ut_z = torch.matmul(u_mat.transpose(-2, -1), tu_ns)
    vt_z = torch.matmul(v_mat.transpose(-2, -1), tv_ns)
    omega_ns = 0.25 * ((ut_z - ut_z.transpose(-2, -1)) + (vt_z - vt_z.transpose(-2, -1)))
    tu_base = tu_ns - torch.matmul(u_mat, omega_ns)
    tv_base = tv_ns - torch.matmul(v_mat, omega_ns)

    norm_ns = (tu_base.square().sum(dim=(-2, -1), keepdim=True) + tv_base.square().sum(dim=(-2, -1), keepdim=True)).sqrt()
    norm_ref = (xi_u_ref.square().sum(dim=(-2, -1), keepdim=True) + xi_v_ref.square().sum(dim=(-2, -1), keepdim=True)).sqrt().clamp_min(1e-8)
    ut_r = torch.matmul(u_mat.transpose(-2, -1), xi_u_ref)
    vt_r = torch.matmul(v_mat.transpose(-2, -1), xi_v_ref)
    omega_ref = 0.25 * ((ut_r - ut_r.transpose(-2, -1)) + (vt_r - vt_r.transpose(-2, -1))) * (norm_ns / norm_ref)

    gamma_deg = compute_lorentzian_degeneracy_mask(sigma, tau=tau).to(omega_ns.dtype)
    omega_split = (1.0 - gamma_deg) * omega_ref
    return tu_base + torch.matmul(u_mat, omega_split), tv_base + torch.matmul(v_mat, omega_split)


def compute_degenerate_svd_spin_fraction(
    xi_u: torch.Tensor, xi_v: torch.Tensor, u_mat: torch.Tensor, v_mat: torch.Tensor,
    sigma: torch.Tensor, tau: float = DEFAULT_SVD_DEGENERACY_TAU,
) -> float:
    r"""Measure degeneracy-weighted co-rotation spin $\|\Gamma(\vec{\sigma}) \odot \Omega_{\text{sym}}(\xi_U, \xi_V)\|_F / \sqrt{\|\xi_U\|_F^2 + \|\xi_V\|_F^2}$ (`<= 8` args)."""
    ut_u = torch.matmul(u_mat.float().transpose(-2, -1), xi_u.float())
    vt_v = torch.matmul(v_mat.float().transpose(-2, -1), xi_v.float())
    omega_sym = 0.25 * ((ut_u - ut_u.transpose(-2, -1)) + (vt_v - vt_v.transpose(-2, -1)))
    gamma_deg = compute_lorentzian_degeneracy_mask(sigma, tau=tau)
    num = torch.linalg.matrix_norm(gamma_deg * omega_sym, ord="fro")
    den = (xi_u.float().square().sum(dim=(-2, -1)) + xi_v.float().square().sum(dim=(-2, -1))).sqrt().clamp_min(1e-8)
    return float((num / den).mean().item())


def filter_tied_bridge_post_ns_common_spin(
    z_in_ns: torch.Tensor, z_out_ns: torch.Tensor, xi_in_ref: torch.Tensor, xi_out_ref: torch.Tensor,
    u_in: torch.Tensor, u_out: torch.Tensor, excess_damping: float = DEFAULT_COMMON_BRIDGE_EXCESS_DAMPING,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Preserve 100% of relative bridge rotation $\Omega_{\text{in}} - \Omega_{\text{out}}$ while damping excess post-NS common spin $\Delta\Omega_{\text{com}}^{\text{NS}}$ (`<= 8` args)."""
    tin = project_real_stiefel_tangent(z_in_ns, u_in)
    tout = project_real_stiefel_tangent(z_out_ns, u_out)
    ut_in = torch.matmul(u_in.transpose(-2, -1), tin)
    ut_out = torch.matmul(u_out.transpose(-2, -1), tout)
    om_com_ns = 0.25 * ((ut_in - ut_in.transpose(-2, -1)) + (ut_out - ut_out.transpose(-2, -1)))

    norm_ns = (tin.square().sum(dim=(-2, -1), keepdim=True) + tout.square().sum(dim=(-2, -1), keepdim=True)).sqrt()
    norm_ref = (xi_in_ref.square().sum(dim=(-2, -1), keepdim=True) + xi_out_ref.square().sum(dim=(-2, -1), keepdim=True)).sqrt().clamp_min(1e-8)
    ur_in = torch.matmul(u_in.transpose(-2, -1), xi_in_ref)
    ur_out = torch.matmul(u_out.transpose(-2, -1), xi_out_ref)
    om_com_ref = 0.25 * ((ur_in - ur_in.transpose(-2, -1)) + (ur_out - ur_out.transpose(-2, -1))) * (norm_ns / norm_ref)
    delta_com = float(excess_damping) * (om_com_ns - om_com_ref)
    return tin - torch.matmul(u_in, delta_com), tout - torch.matmul(u_out, delta_com)


def _clean_and_snr_gate_real_tangent(
    g_tan: torch.Tensor, m_prev_tan: torch.Tensor, lr: float, beta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Clean opposing momentum ($\langle M, G\rangle < 0$) and compute tangent SNR + magnitude trust-region `step_lr` (`<= 8` args)."""
    g_sq = g_tan.square().sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    m_sq = m_prev_tan.square().sum(dim=(-2, -1), keepdim=True)
    dot_mg = (m_prev_tan * g_tan).sum(dim=(-2, -1), keepdim=True)
    m_clean = m_prev_tan - torch.clamp_max(dot_mg / g_sq, 0.0) * g_tan
    m_upd = float(beta) * m_clean + g_tan
    xi = g_tan + float(beta) * m_upd
    g_n, m_n = g_sq.sqrt(), m_sq.sqrt()
    cos_mg = dot_mg / (m_n * g_n + 1e-8)
    coh = m_upd.norm(dim=(-2, -1), keepdim=True) / (float(beta) * m_n + g_n + 1e-8)
    snr_tan = (coh * (1.0 + 0.5 * cos_mg)).clamp(0.05, 1.50)
    xi_sq = xi.square().sum(dim=(-2, -1), keepdim=True)
    tau_sq = float(DEFAULT_MUON_GRAD_NORM_GATE_TAU * DEFAULT_MUON_GRAD_NORM_GATE_TAU)
    step_lr = float(lr) * (snr_tan / (snr_tan + 0.35)) * (xi_sq / (xi_sq + tau_sq)).sqrt()
    return m_upd, xi, step_lr


def _clean_and_snr_gate_complex_tangent(
    gr_tan: torch.Tensor, gi_tan: torch.Tensor, mr_prev: torch.Tensor, mi_prev: torch.Tensor, lr: float, beta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clean opposing complex momentum and compute complex tangent SNR + magnitude trust-region `step_lr` (`<= 8` args)."""
    g_sq = (gr_tan.square() + gi_tan.square()).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    m_sq = (mr_prev.square() + mi_prev.square()).sum(dim=(-2, -1), keepdim=True)
    dot_mg = (mr_prev * gr_tan + mi_prev * gi_tan).sum(dim=(-2, -1), keepdim=True)
    neg_proj = torch.clamp_max(dot_mg / g_sq, 0.0)
    mr_clean = mr_prev - neg_proj * gr_tan
    mi_clean = mi_prev - neg_proj * gi_tan
    mr_upd = float(beta) * mr_clean + gr_tan
    mi_upd = float(beta) * mi_clean + gi_tan
    xi_re = gr_tan + float(beta) * mr_upd
    xi_im = gi_tan + float(beta) * mi_upd
    g_n, m_n = g_sq.sqrt(), m_sq.sqrt()
    m_upd_n = (mr_upd.square() + mi_upd.square()).sum(dim=(-2, -1), keepdim=True).sqrt()
    cos_mg = dot_mg / (m_n * g_n + 1e-8)
    coh = m_upd_n / (float(beta) * m_n + g_n + 1e-8)
    snr_tan = (coh * (1.0 + 0.5 * cos_mg)).clamp(0.05, 1.50)
    xi_sq = (xi_re.square() + xi_im.square()).sum(dim=(-2, -1), keepdim=True)
    tau_sq = float(DEFAULT_MUON_GRAD_NORM_GATE_TAU * DEFAULT_MUON_GRAD_NORM_GATE_TAU)
    step_lr = float(lr) * (snr_tan / (snr_tan + 0.35)) * (xi_sq / (xi_sq + tau_sq)).sqrt()
    return mr_upd, mi_upd, xi_re, xi_im, step_lr


def _whiten_tall_real_stiefel(xi: torch.Tensor, ns_steps: int, tik_lam: float) -> torch.Tensor:
    """Run `_newton_schulz5_tall_real` plus 1 quadratic polar step when `rows != cols` (`<= 8` args)."""
    u_raw = _newton_schulz5_tall_real(xi, steps=ns_steps, tikhonov_lambda=tik_lam)
    cols = xi.size(-1)
    if xi.size(-2) != cols:
        eye_c = torch.eye(cols, device=xi.device, dtype=xi.dtype)
        u_raw = 0.5 * torch.matmul(u_raw, 3.0 * eye_c - torch.matmul(u_raw.transpose(-2, -1), u_raw))
    return u_raw


def _retract_real_tall_stiefel_tensor_lr(w_tall: torch.Tensor, u_tan: torch.Tensor, step_lr: torch.Tensor | float) -> torch.Tensor:
    """Retract `w_tall` `(B, D, k)` along `u_tan` with tensor or scalar `step_lr` (`<= 8` args)."""
    cols = w_tall.size(-1)
    scale = torch.rsqrt(1.0 + step_lr * step_lr) if isinstance(step_lr, torch.Tensor) else (1.0 / math.sqrt(1.0 + float(step_lr) * float(step_lr)))
    w_next = (w_tall - step_lr * u_tan) * scale
    eye = torch.eye(cols, device=w_next.device, dtype=w_next.dtype)
    for _ in range(2):
        w_next = 0.5 * torch.matmul(w_next, 3.0 * eye - torch.matmul(w_next.transpose(-2, -1), w_next))
    return w_next


def _retract_complex_tall_stiefel_tensor_lr(
    a_re: torch.Tensor, a_im: torch.Tensor, tan_re: torch.Tensor, tan_im: torch.Tensor, step_lr: torch.Tensor | float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retract complex `(a_re, a_im)` `(B, D, d)` along `(tan_re, tan_im)` with tensor or scalar `step_lr` (`<= 8` args)."""
    scale = torch.rsqrt(1.0 + step_lr * step_lr) if isinstance(step_lr, torch.Tensor) else (1.0 / math.sqrt(1.0 + float(step_lr) * float(step_lr)))
    nx_re = (a_re - step_lr * tan_re) * scale
    nx_im = (a_im - step_lr * tan_im) * scale
    eye = torch.eye(a_re.size(-1), device=a_re.device, dtype=a_re.dtype)
    for _ in range(2):
        g_re = torch.matmul(nx_re.transpose(-2, -1), nx_re) + torch.matmul(nx_im.transpose(-2, -1), nx_im)
        g_im = torch.matmul(nx_re.transpose(-2, -1), nx_im) - torch.matmul(nx_im.transpose(-2, -1), nx_re)
        tm_re, tm_im = 3.0 * eye - g_re, -g_im
        nx_re, nx_im = (
            0.5 * (torch.matmul(nx_re, tm_re) - torch.matmul(nx_im, tm_im)),
            0.5 * (torch.matmul(nx_re, tm_im) + torch.matmul(nx_im, tm_re)),
        )
    return nx_re, nx_im


# ==============================================================================
# 2. Selective Exact Quotient Riemannian Optimizer (`SelectiveExactQuotientMuon`)
# ==============================================================================
class SelectiveExactQuotientMuon(torch.optim.Optimizer):
    r"""Riemannian Optimizer enforcing Selective Exact Quotienting across Tied Bridges, Coupled SVD-Stiefel Triples, Rank-$R$ Kraus Channels, and $\mathfrak{su}(16)$ MUB Probes."""

    def __init__(
        self,
        bridge_pairs: list[tuple[nn.Parameter, nn.Parameter]],
        coupled_svd_triples: list[tuple[nn.Parameter, nn.Parameter, nn.Parameter]],
        kraus_pairs: list[tuple[nn.Parameter, nn.Parameter]],
        mub_su_pairs: list[tuple[nn.Parameter, nn.Parameter]],
        stiefel_lr: float = DEFAULT_SHALLOW_STIEFEL_LR,
        unitary_lr: float = DEFAULT_SHALLOW_STIEFEL_LR,
        beta: float = DEFAULT_SELECTIVE_MOMENTUM_BETA,
    ) -> None:
        all_params: list[nn.Parameter] = []
        for u_in, u_out in bridge_pairs:
            all_params.extend([u_in, u_out])
        for u_p, v_p, _ in coupled_svd_triples:
            all_params.extend([u_p, v_p])
        for kr_p, ki_p in kraus_pairs:
            all_params.extend([kr_p, ki_p])
        for mr_p, mi_p in mub_su_pairs:
            all_params.extend([mr_p, mi_p])
        defaults = dict(
            lr=stiefel_lr, stiefel_lr=stiefel_lr, unitary_lr=unitary_lr,
            beta=beta, ns_steps=DEFAULT_NS5_STEPS, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA,
        )
        super().__init__(all_params, defaults)
        self.bridge_pairs = list(bridge_pairs)
        self.coupled_svd_triples = list(coupled_svd_triples)
        self.kraus_pairs = list(kraus_pairs)
        self.mub_su_pairs = list(mub_su_pairs)
        self.enable_selective_quotient: bool = True
        self.last_gauge_diagnostics: dict[str, float] = {}

    @torch.no_grad()
    def _step_bridge_pairs(self, stiefel_lr: float, beta: float, ns_steps: int, tik_lam: float) -> None:
        """Update tied vocabulary subspace bridges `(U_in, U_out)` with SNR-gated Stiefel whitening + excess common-spin damping (`<= 8` args)."""
        active = [(u_in, u_out) for u_in, u_out in self.bridge_pairs if u_in.grad is not None and u_out.grad is not None]
        if not active:
            return
        uin_s = torch.stack([p.data.float() for p, _ in active], dim=0)
        uout_s = torch.stack([p.data.float() for _, p in active], dim=0)
        gin_s = torch.stack([torch.nan_to_num(p.grad.float(), 0.0, 0.0, 0.0) for p, _ in active], dim=0)
        gout_s = torch.stack([torch.nan_to_num(p.grad.float(), 0.0, 0.0, 0.0) for _, p in active], dim=0)
        min_l = [self.state[p].setdefault("momentum_buffer", torch.zeros_like(p, dtype=torch.float32)) for p, _ in active]
        mout_l = [self.state[p].setdefault("momentum_buffer", torch.zeros_like(p, dtype=torch.float32)) for _, p in active]
        min_s, mout_s = torch.stack(min_l, dim=0), torch.stack(mout_l, dim=0)

        tin_g = project_real_stiefel_tangent(gin_s, uin_s)
        tout_g = project_real_stiefel_tangent(gout_s, uout_s)
        min_pre = project_real_stiefel_tangent(min_s, uin_s)
        mout_pre = project_real_stiefel_tangent(mout_s, uout_s)
        min_upd, xi_in, lr_in = _clean_and_snr_gate_real_tangent(tin_g, min_pre, stiefel_lr, beta)
        mout_upd, xi_out, lr_out = _clean_and_snr_gate_real_tangent(tout_g, mout_pre, stiefel_lr, beta)
        zin_ns = _whiten_tall_real_stiefel(xi_in, ns_steps=ns_steps, tik_lam=tik_lam)
        zout_ns = _whiten_tall_real_stiefel(xi_out, ns_steps=ns_steps, tik_lam=tik_lam)
        if self.enable_selective_quotient:
            tan_in, tan_out = filter_tied_bridge_post_ns_common_spin(zin_ns, zout_ns, xi_in, xi_out, uin_s, uout_s)
        else:
            tan_in = project_real_stiefel_tangent(zin_ns, uin_s)
            tan_out = project_real_stiefel_tangent(zout_ns, uout_s)
        uin_next = _retract_real_tall_stiefel_tensor_lr(uin_s, tan_in, lr_in)
        uout_next = _retract_real_tall_stiefel_tensor_lr(uout_s, tan_out, lr_out)
        min_stored = project_real_stiefel_tangent(min_upd, uin_next)
        mout_stored = project_real_stiefel_tangent(mout_upd, uout_next)

        self.last_gauge_diagnostics["bridge_common_spin_frac"] = compute_coupled_svd_joint_spin_fraction(tan_in, tan_out, uin_s, uout_s)
        torch._foreach_copy_([p.data for p, _ in active], list(uin_next.unbind(0)))
        torch._foreach_copy_([p.data for _, p in active], list(uout_next.unbind(0)))
        torch._foreach_copy_(min_l, list(min_stored.unbind(0)))
        torch._foreach_copy_(mout_l, list(mout_stored.unbind(0)))

    @torch.no_grad()
    def _step_coupled_svd_triples(self, stiefel_lr: float, beta: float, ns_steps: int, tik_lam: float) -> None:
        """Update coupled `(U, V, sigma)` triples with SNR-gated Stiefel whitening + Lorentzian split-spectrum co-rotation (`<= 8` args)."""
        buckets: dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[nn.Parameter, nn.Parameter, nn.Parameter]]] = {}
        for u_p, v_p, s_p in self.coupled_svd_triples:
            if u_p.grad is not None and v_p.grad is not None:
                key = ((int(u_p.size(0)), int(u_p.size(1))), (int(v_p.size(0)), int(v_p.size(1))))
                buckets.setdefault(key, []).append((u_p, v_p, s_p))

        deg_spins, unwt_spins = [], []
        for _, triples in buckets.items():
            u_stack = torch.stack([u.data.float() for u, _, _ in triples], dim=0)
            v_stack = torch.stack([v.data.float() for _, v, _ in triples], dim=0)
            s_stack = torch.stack([s.data.float() for _, _, s in triples], dim=0)
            gu_stack = torch.stack([torch.nan_to_num(u.grad.float(), 0.0, 0.0, 0.0) for u, _, _ in triples], dim=0)
            gv_stack = torch.stack([torch.nan_to_num(v.grad.float(), 0.0, 0.0, 0.0) for _, v, _ in triples], dim=0)
            mu_list = [self.state[u].setdefault("momentum_buffer", torch.zeros_like(u, dtype=torch.float32)) for u, _, _ in triples]
            mv_list = [self.state[v].setdefault("momentum_buffer", torch.zeros_like(v, dtype=torch.float32)) for _, v, _ in triples]
            mu_stack, mv_stack = torch.stack(mu_list, dim=0), torch.stack(mv_list, dim=0)

            if self.enable_selective_quotient:
                gu_h, gv_h = project_coupled_svd_raw_exact_degenerate(gu_stack, gv_stack, u_stack, v_stack, s_stack)
                mu_pre, mv_pre = project_coupled_svd_raw_exact_degenerate(mu_stack, mv_stack, u_stack, v_stack, s_stack)
                mu_upd, xi_u, lr_u = _clean_and_snr_gate_real_tangent(gu_h, mu_pre, stiefel_lr, beta)
                mv_upd, xi_v, lr_v = _clean_and_snr_gate_real_tangent(gv_h, mv_pre, stiefel_lr, beta)
                zu_ns = _whiten_tall_real_stiefel(xi_u, ns_steps=ns_steps, tik_lam=tik_lam)
                zv_ns = _whiten_tall_real_stiefel(xi_v, ns_steps=ns_steps, tik_lam=tik_lam)
                tu_tan, tv_tan = filter_coupled_svd_post_ns_corotation(zu_ns, zv_ns, xi_u, xi_v, u_stack, v_stack, s_stack)
            else:
                gu_t = project_real_stiefel_tangent(gu_stack, u_stack)
                gv_t = project_real_stiefel_tangent(gv_stack, v_stack)
                mu_pre = project_real_stiefel_tangent(mu_stack, u_stack)
                mv_pre = project_real_stiefel_tangent(mv_stack, v_stack)
                mu_upd, xi_u, lr_u = _clean_and_snr_gate_real_tangent(gu_t, mu_pre, stiefel_lr, beta)
                mv_upd, xi_v, lr_v = _clean_and_snr_gate_real_tangent(gv_t, mv_pre, stiefel_lr, beta)
                zu_ns = _whiten_tall_real_stiefel(xi_u, ns_steps=ns_steps, tik_lam=tik_lam)
                zv_ns = _whiten_tall_real_stiefel(xi_v, ns_steps=ns_steps, tik_lam=tik_lam)
                tu_tan = project_real_stiefel_tangent(zu_ns, u_stack)
                tv_tan = project_real_stiefel_tangent(zv_ns, v_stack)

            u_next = _retract_real_tall_stiefel_tensor_lr(u_stack, tu_tan, lr_u)
            v_next = _retract_real_tall_stiefel_tensor_lr(v_stack, tv_tan, lr_v)
            mu_stored = project_real_stiefel_tangent(mu_upd, u_next)
            mv_stored = project_real_stiefel_tangent(mv_upd, v_next)

            deg_spins.append(compute_degenerate_svd_spin_fraction(tu_tan, tv_tan, u_stack, v_stack, s_stack))
            unwt_spins.append(compute_coupled_svd_joint_spin_fraction(tu_tan, tv_tan, u_stack, v_stack))
            torch._foreach_copy_([u.data for u, _, _ in triples], list(u_next.unbind(0)))
            torch._foreach_copy_([v.data for _, v, _ in triples], list(v_next.unbind(0)))
            torch._foreach_copy_(mu_list, list(mu_stored.unbind(0)))
            torch._foreach_copy_(mv_list, list(mv_stored.unbind(0)))

        if deg_spins:
            self.last_gauge_diagnostics["svd_degenerate_spin_frac"] = float(sum(deg_spins) / len(deg_spins))
            self.last_gauge_diagnostics["svd_unweighted_spin_frac"] = float(sum(unwt_spins) / len(unwt_spins))

    @torch.no_grad()
    def _step_complex_kraus_and_mub(self, unitary_lr: float, beta: float, ns_steps: int, tik_lam: float) -> None:
        r"""Update Rank-$R$ Kraus channels ($\mathfrak{u}(R)$ Lyapunov quotient) and MUB probes ($\mathfrak{su}(16)$ quotient) (`<= 8` args)."""
        for is_mub, pair_list in ((False, self.kraus_pairs), (True, self.mub_su_pairs)):
            active = [(ur, ui) for ur, ui in pair_list if ur.grad is not None and ui.grad is not None]
            if not active:
                continue
            ur_cat = torch.cat([ur.data.float() for ur, _ in active], dim=0)
            ui_cat = torch.cat([ui.data.float() for _, ui in active], dim=0)
            gr_cat = torch.cat([torch.nan_to_num(ur.grad.float(), 0.0, 0.0, 0.0) for ur, _ in active], dim=0)
            gi_cat = torch.cat([torch.nan_to_num(ui.grad.float(), 0.0, 0.0, 0.0) for _, ui in active], dim=0)
            mr_list = [self.state[ur].setdefault("momentum_buffer", torch.zeros_like(ur, dtype=torch.float32)) for ur, _ in active]
            mi_list = [self.state[ui].setdefault("momentum_buffer", torch.zeros_like(ui, dtype=torch.float32)) for _, ui in active]
            mr_cat, mi_cat = torch.cat(mr_list, dim=0), torch.cat(mi_list, dim=0)

            proj_fn = (project_su_dh_mub_horizontal if is_mub else project_kraus_branch_gauge_horizontal) if self.enable_selective_quotient else project_complex_stiefel_tangent
            gr_h, gi_h = proj_fn(gr_cat, gi_cat, ur_cat, ui_cat)
            mr_pre, mi_pre = proj_fn(mr_cat, mi_cat, ur_cat, ui_cat)
            mr_upd, mi_upd, xi_re, xi_im, step_lr = _clean_and_snr_gate_complex_tangent(
                gr_h, gi_h, mr_pre, mi_pre, unitary_lr, beta
            )
            zr, zi = _complex_newton_schulz5(xi_re, xi_im, steps=ns_steps, tikhonov_lambda=tik_lam)
            tan_re, tan_im = proj_fn(zr, zi, ur_cat, ui_cat)
            ur_next, ui_next = _retract_complex_tall_stiefel_tensor_lr(ur_cat, ui_cat, tan_re, tan_im, step_lr)
            mr_stored, mi_stored = proj_fn(mr_upd, mi_upd, ur_next, ui_next)

            if is_mub:
                self.last_gauge_diagnostics["su16_step_phase_frac"] = compute_su_dh_global_phase_fraction(tan_re, tan_im, ur_cat, ui_cat)
            else:
                self.last_gauge_diagnostics["kraus_step_gauge_frac"] = compute_kraus_branch_gauge_fraction(tan_re, tan_im, ur_cat, ui_cat)

            splits = [ur.size(0) for ur, _ in active]
            torch._foreach_copy_([ur.data for ur, _ in active], list(ur_next.split(splits, dim=0)))
            torch._foreach_copy_([ui.data for _, ui in active], list(ui_next.split(splits, dim=0)))
            torch._foreach_copy_(mr_list, list(mr_stored.split(splits, dim=0)))
            torch._foreach_copy_(mi_list, list(mi_stored.split(splits, dim=0)))

    def set_momentum(self, beta: float) -> None:
        """Set Riemannian momentum factor `beta`."""
        for g in self.param_groups:
            g["beta"] = float(beta)

    def set_lr_scale(self, lr_scale: float) -> None:
        """Set learning-rate multiplier `lr_scale`."""
        for g in self.param_groups:
            g["lr_scale"] = float(lr_scale)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """Execute 1 selective-quotient Riemannian step across Bridges, Coupled-SVD, Kraus, and MUB manifolds."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        g0 = self.param_groups[0]
        lr_scale = float(g0.get("lr_scale", 1.0))
        stiefel_lr = float(g0.get("stiefel_lr", g0["lr"])) * lr_scale
        unitary_lr = float(g0.get("unitary_lr", g0["lr"])) * lr_scale
        beta = float(g0.get("beta", DEFAULT_SELECTIVE_MOMENTUM_BETA))
        ns_steps = int(g0.get("ns_steps", DEFAULT_NS5_STEPS))
        tik_lam = float(g0.get("tikhonov_lambda", DEFAULT_TIKHONOV_GRAM_LAMBDA))

        self._step_bridge_pairs(stiefel_lr, beta, ns_steps, tik_lam)
        self._step_coupled_svd_triples(stiefel_lr, beta, ns_steps, tik_lam)
        self._step_complex_kraus_and_mub(unitary_lr, beta, ns_steps, tik_lam)
        return loss


# ==============================================================================
# 3. Strictly Read-Only Riemannian Shadow EMA (Zero Rollbacks / Zero LR Ratchet)
# ==============================================================================
class ReadOnlyQuenchRiemannianShadowEMA(RiemannianShadowEMAThermostat):
    r"""Strictly Read-Only Geodesic Polyak–Ruppert Shadow EMA (`0` mid-training rollbacks, `lr_scale = 1.00x`)."""

    def __init__(
        self,
        hot_model: Any,
        ema_decay: float = DEFAULT_THERMOSTAT_EMA_DECAY,
        eval_fn: Any = None,
        retract_cadence: int = DEFAULT_THERMOSTAT_RETRACT_CADENCE,
    ) -> None:
        super().__init__(
            hot_model,
            ema_decay=ema_decay,
            gap_threshold=1e9,
            gain_kappa=0.0,
            min_scale_step=1.0,
            eval_fn=eval_fn,
            retract_cadence=retract_cadence,
        )

    @torch.no_grad()
    def check_and_thermostat(
        self,
        hot_model: Any,
        stream: Any,
        current_lr_scale: float,
        step: int,
        apply_feedback: bool = True,
    ) -> dict[str, Any]:
        """Retract and evaluate `ema_model` alongside `hot_model` in strictly read-only mode (`triggered = False`, `lr_scale = 1.00x`, `<= 8` args)."""
        del step, apply_feedback
        self.retract_shadow_ema_manifold_(hot_model=hot_model)
        hot_ev = self.eval_fn(hot_model, stream)
        ema_ev = self.eval_fn(self.ema_model, stream)
        hot_ce, ema_ce = float(hot_ev["val_ce"]), float(ema_ev["val_ce"])
        thermal_gap = hot_ce - ema_ce
        lr_keep = float(current_lr_scale)
        active_ev = ema_ev if ema_ce <= hot_ce else hot_ev
        return {
            "triggered": False,
            "hot_val_ce": round(hot_ce, 4),
            "ema_val_ce": round(ema_ce, 4),
            "thermal_gap_ce": round(thermal_gap, 4),
            "lr_scale_before": round(lr_keep, 4),
            "lr_scale_after": round(lr_keep, 4),
            "retract_ms": round(self.last_retract_ms, 3),
            "active_eval": active_ev,
            "hot_eval": hot_ev,
            "ema_eval": ema_ev,
        }


# ==============================================================================
# 4. Wave 67 Selective Exact Quotient Quantum GPT (`SelectiveExactQuotientGPT`)
# ==============================================================================
class SelectiveExactQuotientGPT(nn.Module):
    r"""Wave 67 Quantum Density-Matrix Language Model with Selective Exact Quotienting, $\mathfrak{su}(16)$ MUB Tomography, and Configurable `128D`/`256D` Subspace Readout."""

    def __init__(self, spec: Wave67SelectiveQuotientSpec) -> None:
        super().__init__()
        self.spec, self.variant_mode = spec, Wave67SelectiveVariantMode(spec.variant_mode)
        self.embed_dim, self.num_heads = int(DEFAULT_TOMO_EMBED_DIM), int(DEFAULT_TOMO_NUM_HEADS)
        self.d_h, self.head_real_dim = int(DEFAULT_TOMO_HEAD_COMPLEX_DIM), 2 * int(DEFAULT_TOMO_HEAD_COMPLEX_DIM)
        self.num_layers, self.kraus_rank = int(spec.num_layers), max(1, int(spec.kraus_rank))
        self.mlp_hidden, self.subspace_dim = int(spec.mlp_hidden), int(spec.subspace_dim)
        self.d_vac = int(DEFAULT_TOMO_VACUUM_DIM)
        depth_scale = math.sqrt(float(DEFAULT_TOMO_NUM_LAYERS) / float(max(1, self.num_layers)))

        self.boundary = SharedStiefelSubspaceFrameBoundary(self.embed_dim, self.subspace_dim)

        self.tok_ngram_in = nn.Embedding(DEFAULT_TOK_NGRAM_BUCKETS, self.embed_dim)
        self.tok_ngram_out = nn.Embedding(DEFAULT_TOK_NGRAM_BUCKETS, self.embed_dim)
        nn.init.zeros_(self.tok_ngram_in.weight)
        nn.init.zeros_(self.tok_ngram_out.weight)
        self.ngram_out_scale = nn.Parameter(torch.tensor(0.85, dtype=torch.float32))
        self.lag_phase = nn.Parameter(torch.linspace(-0.35, 0.35, self.num_heads, dtype=torch.float32))
        self.layer0_mix_logits = nn.Parameter(torch.tensor([1.10, 0.20, -0.10], dtype=torch.float32))
        self.smear_gate = nn.Linear(16, self.embed_dim, bias=False)
        self.smear_lambda = nn.Parameter(torch.tensor(0.22, dtype=torch.float32))
        nn.init.zeros_(self.smear_gate.weight)

        self.rope = NativeLeaderboardRotary(self.head_real_dim, max_seq_len=2048)
        self.quantum_blocks = nn.ModuleList([
            _KrausDensityHighwayBlock(
                embed_dim=self.embed_dim, num_heads=self.num_heads, head_complex_dim=self.d_h,
                mlp_hidden=self.mlp_hidden, kraus_rank=self.kraus_rank, depth_scale=depth_scale,
            )
            for _ in range(self.num_layers)
        ])
        self.exit_bridge = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)
        self.tomo_proj = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)

        u1_re, u1_im, u2_re, u2_im = build_mub_two_basis_unitary_probes(self.num_heads, self.d_h, seed=spec.seed)
        self.probe_u1_re, self.probe_u1_im = nn.Parameter(u1_re), nn.Parameter(u1_im)
        if spec.enable_selective_quotient:
            self.register_parameter("probe_u2_re", None)
            self.register_parameter("probe_u2_im", None)
        else:
            self.probe_u2_re, self.probe_u2_im = nn.Parameter(u2_re), nn.Parameter(u2_im)

        idx = torch.arange(self.d_h, dtype=torch.float32)
        angles = (2.0 * math.pi / float(self.d_h)) * torch.outer(idx, idx)
        scale = 1.0 / math.sqrt(float(self.d_h))
        self.register_buffer("dft_f_re", (torch.cos(angles) * scale).unsqueeze(0), persistent=False)
        self.register_buffer("dft_f_im", (torch.sin(angles) * scale).unsqueeze(0), persistent=False)

        self.tomo_scale = nn.Parameter(torch.tensor(DEFAULT_INIT_TOMO_SCALE, dtype=torch.float32))
        self.filter_scale = nn.Parameter(torch.tensor(DEFAULT_INIT_FILTER_SCALE, dtype=torch.float32))
        self.vac_rest_mass = nn.Parameter(torch.tensor(DEFAULT_INIT_VACUUM_REST_MASS, dtype=torch.float32))
        self.vac_proj = nn.Linear(16, self.d_vac, bias=False)
        nn.init.zeros_(self.vac_proj.weight)
        self.register_buffer("tokens_seen_buf", torch.zeros((), dtype=torch.long), persistent=True)

    @torch.no_grad()
    def initialize_model_(self) -> float:
        """Initialize trunk and fixed-once vocabulary boundary at Step 0 with zero training tokens scanned."""
        device = self.tomo_scale.device
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.boundary.initialize_boundary_(seed=self.spec.seed)
        eye_d = torch.eye(self.embed_dim, device=device, dtype=torch.float32)
        for br in (self.exit_bridge, self.tomo_proj):
            br.u_stiefel.copy_(eye_d)
            br.v_stiefel.copy_(eye_d)
            br.sigma.fill_(1.0)
        u1_re, u1_im, u2_re, u2_im = build_mub_two_basis_unitary_probes(
            self.num_heads, self.d_h, seed=self.spec.seed, device=device
        )
        self.probe_u1_re.copy_(u1_re)
        self.probe_u1_im.copy_(u1_im)
        if self.probe_u2_re is not None and self.probe_u2_im is not None:
            self.probe_u2_re.copy_(u2_re)
            self.probe_u2_im.copy_(u2_im)
        for blk in self.quantum_blocks:
            kr, ki = build_initial_kraus_stiefel_operators(
                self.num_heads, self.d_h, kraus_rank=self.kraus_rank, device=device
            )
            blk.k_evol_re.copy_(kr)
            blk.k_evol_im.copy_(ki)
        self.tokens_seen_buf.zero_()
        self.restrict_scalars_()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    def current_analytical_beta(self) -> torch.Tensor:
        r"""Return $\mu\text{P}$ subspace-calibrated Parseval inverse temperature $\beta_0(N_{\text{tok}})\sqrt{d_{\text{sub}}/128}$."""
        base_val = compute_analytical_parseval_beta(int(self.tokens_seen_buf.item()), beta_target=float(DEFAULT_ANALYTICAL_BETA_TARGET))
        sub_mup = math.sqrt(float(self.subspace_dim) / float(DEFAULT_NARROW_SUBSPACE_DIM))
        return torch.tensor(base_val * sub_mup, device=self.tokens_seen_buf.device, dtype=torch.float32)

    def bridge_parameter_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        r"""Return tied vocabulary subspace bridge pair $(U_{\text{in}}, U_{\text{out}}) \in \operatorname{St}(512, d_{\text{voc}})^2$."""
        return [(self.boundary.u_in_stiefel, self.boundary.u_out_stiefel)]

    def coupled_svd_parameter_triples(self) -> list[tuple[nn.Parameter, nn.Parameter, nn.Parameter]]:
        r"""Return all coupled `(u_stiefel, v_stiefel, sigma)` triples across `BiorthogonalSVDStiefelProjection` modules."""
        triples: list[tuple[nn.Parameter, nn.Parameter, nn.Parameter]] = [
            (self.exit_bridge.u_stiefel, self.exit_bridge.v_stiefel, self.exit_bridge.sigma),
            (self.tomo_proj.u_stiefel, self.tomo_proj.v_stiefel, self.tomo_proj.sigma),
        ]
        for q_blk in self.quantum_blocks:
            for proj in (q_blk.q_proj, q_blk.k_proj, q_blk.v_proj, q_blk.obs_proj, q_blk.mz_mlp.up_proj, q_blk.mz_mlp.down_proj):
                triples.append((proj.u_stiefel, proj.v_stiefel, proj.sigma))
        return triples

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Return flat list of all real Stiefel matrices for `RiemannianShadowEMAThermostat` retraction."""
        params: list[nn.Parameter] = list(self.boundary.real_stiefel_parameters())
        for u_p, v_p, _ in self.coupled_svd_parameter_triples():
            params.extend([u_p, v_p])
        return params

    def kraus_unitary_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        r"""Return paired `(k_evol_re, k_evol_im)` $\operatorname{St}_{\mathbb{C}}(R d_h, d_h)$ Kraus operators."""
        pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
        for q_blk in self.quantum_blocks:
            pairs.extend(q_blk.complex_unitary_pairs())
        return pairs

    def mub_su_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        r"""Return MUB probe parameter pairs on $SU(16) = U(16)/U(1)$."""
        pairs: list[tuple[nn.Parameter, nn.Parameter]] = [(self.probe_u1_re, self.probe_u1_im)]
        if self.probe_u2_re is not None and self.probe_u2_im is not None:
            pairs.append((self.probe_u2_re, self.probe_u2_im))
        return pairs

    def complex_unitary_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        """Return all complex Stiefel/Unitary parameter pairs for `RiemannianShadowEMAThermostat` retraction."""
        return self.mub_su_pairs() + self.kraus_unitary_pairs()

    @torch.no_grad()
    def restrict_scalars_(self) -> None:
        """Clamp trunk and boundary scalars to their physically valid ranges."""
        self.boundary.restrict_scalars_()
        self.tomo_scale.clamp_(0.05, 2.50)
        self.filter_scale.clamp_(0.05, 2.50)
        self.vac_rest_mass.clamp_(0.02, 2.00)
        self.ngram_out_scale.clamp_(0.10, 2.50)
        self.smear_lambda.clamp_(0.0, 0.85)
        for br in (self.exit_bridge, self.tomo_proj):
            br.sigma.clamp_(0.25, 2.50)
        for blk in self.quantum_blocks:
            blk.restrict_scalars_()

    def current_mub_unbiasedness_error(self) -> float:
        r"""Compute $\max_{h,m,n} \big| |\langle c_{h,m}^{(1)} \mid c_{h,n}^{(2)}\rangle|^2 - \frac{1}{16} \big|$."""
        if self.probe_u2_re is not None and self.probe_u2_im is not None:
            u2_re, u2_im = self.probe_u2_re, self.probe_u2_im
        else:
            u2_re = torch.matmul(self.dft_f_re, self.probe_u1_re) - torch.matmul(self.dft_f_im, self.probe_u1_im)
            u2_im = torch.matmul(self.dft_f_re, self.probe_u1_im) + torch.matmul(self.dft_f_im, self.probe_u1_re)
        return compute_mub_unbiasedness_max_error(self.probe_u1_re, self.probe_u1_im, u2_re, u2_im)

    @torch.no_grad()
    def get_manifold_telemetry(self) -> dict[str, float]:
        """Measure Kraus Stiefel, $SU(16)$ MUB unbiasedness, and Stiefel bridge orthogonality errors."""
        kraus_errs = []
        for blk in self.quantum_blocks:
            kr, ki = blk.k_evol_re.detach().float(), blk.k_evol_im.detach().float()
            g_re = torch.bmm(kr.transpose(-2, -1), kr) + torch.bmm(ki.transpose(-2, -1), ki)
            g_im = torch.bmm(kr.transpose(-2, -1), ki) - torch.bmm(ki.transpose(-2, -1), kr)
            eye = torch.eye(self.d_h, device=kr.device, dtype=torch.float32).unsqueeze(0)
            kraus_errs.append(float(((g_re - eye).square().sum() + g_im.square().sum()).sqrt().item() / math.sqrt(self.num_heads * self.d_h)))
        gr_errs = []
        for p_st in self.boundary.real_stiefel_parameters():
            w = p_st.detach().float()
            eye_c = torch.eye(w.size(-1), device=w.device, dtype=torch.float32)
            gr_errs.append(float((torch.matmul(w.t(), w) - eye_c).norm().item() / math.sqrt(float(w.size(-1)))))
        return {
            "analytical_beta": round(float(self.current_analytical_beta().item()), 4),
            "kraus_stiefel_err": float(f"{max(kraus_errs, default=0.0):.3e}"),
            "boundary_stiefel_err": float(f"{max(gr_errs, default=0.0):.3e}"),
            "mub_unbiased_err": round(self.current_mub_unbiasedness_error(), 6),
        }

    def encode_trunk(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        r"""Encode `idx` `(B, T)` through the $L$-layer Kraus highway and gauge-locked $\mathfrak{su}(16)$ MUB readout (`<= 8` args)."""
        bsz, seq_len = idx.shape
        work_dtype = torch.bfloat16 if (idx.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
        idx_l = idx.to(torch.long)
        idx_p1 = torch.cat([idx_l[:, :1], idx_l[:, :-1]], dim=1) if seq_len > 1 else idx_l
        idx_p2 = torch.cat([idx_l[:, :2], idx_l[:, :-2]], dim=1) if seq_len > 2 else idx_p1
        h_tok = (idx_l * 1000003 + idx_p1 * 65537 + idx_p2 * 257) & (DEFAULT_TOK_NGRAM_BUCKETS - 1)

        in_mup = math.sqrt(float(DEFAULT_NARROW_SUBSPACE_DIM) / float(self.subspace_dim))
        raw_emb = self.boundary.encode_tokens(idx) * in_mup
        ng_in = self.tok_ngram_in(h_tok).float()
        raw_emb = raw_emb + 0.30 * (ng_in * torch.rsqrt(ng_in.square().mean(dim=-1, keepdim=True) + 0.0625)).to(work_dtype)
        if seq_len > 1:
            s_gate = self.smear_lambda.to(work_dtype) * torch.sigmoid(F.linear(raw_emb[:, 1:, :16], self.smear_gate.weight.to(work_dtype)))
            raw_emb = torch.cat([raw_emb[:, :1, :], raw_emb[:, 1:, :] + s_gate * raw_emb[:, :-1, :]], dim=1)

        x_0 = F.rms_norm(raw_emb, (self.embed_dim,), eps=1e-6)
        psi_0 = x_0.view(bsz, seq_len, self.num_heads, 2, self.d_h).permute(0, 2, 1, 3, 4)
        psi0_re, psi0_im = complex_l2_normalize(psi_0[..., 0, :], psi_0[..., 1, :])
        rho_packed = build_layer0_explicit_outer_density(psi0_re, psi0_im, self.lag_phase, self.layer0_mix_logits)

        x = x_0
        cos_4d, sin_4d = self.rope.get_cos_sin(seq_len, dtype=work_dtype)
        use_ckpt = self.training and (self.num_layers > DEFAULT_TOMO_NUM_LAYERS) and torch.is_grad_enabled()
        for layer_idx, q_blk in enumerate(self.quantum_blocks):
            x, rho_packed = (
                torch_ckpt.checkpoint(q_blk, x, x_0, cos_4d, sin_4d, rho_packed, use_reentrant=False)
                if (use_ckpt and (layer_idx % 2 == 1)) else q_blk(x, x_0, cos_4d, sin_4d, rho_packed=rho_packed)
            )

        h_pre = F.rms_norm(x, (self.embed_dim,), eps=1e-6)
        h_exit = F.rms_norm(self.exit_bridge(h_pre), (self.embed_dim,), eps=1e-6)
        ng_out = self.tok_ngram_out(h_tok).float()
        ng_out_norm = (ng_out * torch.rsqrt(ng_out.square().mean(dim=-1, keepdim=True) + 0.0625)).to(work_dtype)

        psi_exit = h_exit.view(bsz, seq_len, self.num_heads, 2, self.d_h).permute(0, 2, 1, 3, 4)
        pe_re, pe_im = complex_l2_normalize(psi_exit[..., 0, :], psi_exit[..., 1, :])
        rho_re_final, rho_im_final = unpack_hermitian_isometric(rho_packed, self.d_h)
        filt_re, filt_im = _complex_hermitian_mv(rho_re_final, rho_im_final, pe_re, pe_im)
        empty_rho = torch.empty(0, device=idx.device, dtype=work_dtype)
        rho_re_final, rho_im_final = empty_rho, empty_rho
        h_filt_norm = F.rms_norm(
            torch.stack([filt_re, filt_im], dim=-2).permute(0, 2, 1, 3, 4).reshape(bsz, seq_len, self.embed_dim),
            (self.embed_dim,), eps=1e-6,
        )

        if self.probe_u2_re is not None and self.probe_u2_im is not None:
            q_512 = compute_mub_two_basis_probe_expectations(
                rho_packed, self.probe_u1_re, self.probe_u1_im, self.probe_u2_re, self.probe_u2_im
            )
        else:
            q_512 = compute_gauge_locked_mub_probe_expectations(
                rho_packed, self.probe_u1_re, self.probe_u1_im, self.dft_f_re, self.dft_f_im
            )
        q_proj = F.rms_norm(self.tomo_proj(q_512), (self.embed_dim,), eps=1e-6)
        ng_mod = (1.0 + 0.30 * torch.tanh(q_proj)) * ng_out_norm

        u_vocab = (
            h_exit
            + self.filter_scale.to(work_dtype) * h_filt_norm
            + self.tomo_scale.to(work_dtype) * q_proj
            + self.ngram_out_scale.to(work_dtype) * ng_mod
        )
        u_vac = F.linear(h_pre[:, :, :16], self.vac_proj.weight.to(work_dtype))
        return u_vocab, u_vac, rho_re_final, rho_im_final

    def forward_loss_and_metrics(self, idx: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Run trunk + fixed-once subsystem readout and return `(loss, metrics_dict)`."""
        u_vocab, u_vac, rho_re, rho_im = self.encode_trunk(idx)
        return self.boundary.forward_nll_and_metrics(
            u_vocab, u_vac, self.vac_rest_mass, rho_re, rho_im, self.current_analytical_beta(), targets
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Alias for `forward_loss_and_metrics` (`<= 8` args)."""
        return self.forward_loss_and_metrics(idx, targets)


def build_wave67_selective_optimizers(
    model: SelectiveExactQuotientGPT,
) -> tuple[SelectiveExactQuotientMuon, BatchSubspaceTikhonovGramMuon, RiemannianGeodesicScalarAndProbeOptimizer]:
    """Build 100% `AdamW`-free Selective-Quotient Stiefel+Kraus+MUB, Vocab Sphere, and Geodesic optimizers (`<= 8` args)."""
    q_trunk_opt = SelectiveExactQuotientMuon(
        model.bridge_parameter_pairs(),
        model.coupled_svd_parameter_triples(),
        model.kraus_unitary_pairs(),
        model.mub_su_pairs(),
        stiefel_lr=DEFAULT_SHALLOW_STIEFEL_LR * 1.35,
        unitary_lr=0.028,
        beta=DEFAULT_SELECTIVE_MOMENTUM_BETA,
    )
    q_trunk_opt.enable_selective_quotient = bool(model.spec.enable_selective_quotient)

    sub_mup = math.sqrt(float(model.subspace_dim) / float(DEFAULT_NARROW_SUBSPACE_DIM))
    voc_lr = DEFAULT_SHALLOW_WTE_LR * 1.35
    gate_tau = DEFAULT_MUON_GRAD_NORM_GATE_TAU / sub_mup
    sub_groups = model.boundary.sphere_and_ball_groups(
        base_wte_lr=voc_lr, base_readout_lr=DEFAULT_SHALLOW_READOUT_LR * 1.35
    )
    sub_groups.append({"params": [model.tok_ngram_in.weight, model.tok_ngram_out.weight], "lr": DEFAULT_SHALLOW_NGRAM_LR * 1.5, "zero_init_ball": True})
    vocab_opt = BatchSubspaceTikhonovGramMuon(
        sub_groups, lr=voc_lr, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA, grad_norm_gate_tau=gate_tau
    )

    stiefel_ids = {id(p) for p in model.real_stiefel_parameters()}
    for ur, ui in model.complex_unitary_pairs():
        stiefel_ids.update({id(ur), id(ui)})
    vocab_ids = {id(p) for g in sub_groups for p in g["params"]}
    vac_kets = [blk.k_vac_ket for blk in model.quantum_blocks]
    vac_ids = {id(p) for p in vac_kets}

    geo_groups: list[dict[str, Any]] = [{"params": vac_kets, "lr": 0.025, "kind": "real_sphere_ket", "max_step": 0.20}]
    other_scalars = [p for p in model.parameters() if id(p) not in stiefel_ids and id(p) not in vocab_ids and id(p) not in vac_ids]
    geo_groups.append({"params": other_scalars, "lr": DEFAULT_SHALLOW_SCALAR_LR, "kind": "bounded_tangent", "max_step": 0.08})
    geo_opt = RiemannianGeodesicScalarAndProbeOptimizer(geo_groups, lr=DEFAULT_SHALLOW_SCALAR_LR, beta1=0.80, beta2=0.95)
    return q_trunk_opt, vocab_opt, geo_opt


def step_wave67_optimizers(
    model: SelectiveExactQuotientGPT,
    optimizers: tuple[SelectiveExactQuotientMuon, BatchSubspaceTikhonovGramMuon, RiemannianGeodesicScalarAndProbeOptimizer],
    tokens_in_step: int,
    lr_mult: float = 1.0,
    mom_beta: float = DEFAULT_SELECTIVE_MOMENTUM_BETA,
) -> None:
    """Execute one coordinated Selective-Quotient Stiefel+Kraus+MUB + Vocab Sphere + Geodesic step (`<= 8` args)."""
    q_trunk_opt, vocab_opt, geo_opt = optimizers
    q_trunk_opt.set_momentum(mom_beta)
    q_trunk_opt.set_lr_scale(lr_mult)
    q_trunk_opt.step()

    vocab_opt.set_momentum(mom_beta)
    vocab_opt.step_full(lr_scale=lr_mult)

    geo_opt.set_momentum(0.0 if mom_beta <= 0.0 else 0.80)
    geo_opt.step_active(active_idx=None, lr_scale=lr_mult)
    model.restrict_scalars_()
    with torch.no_grad():
        model.tokens_seen_buf.add_(int(tokens_in_step))


@torch.no_grad()
def evaluate_wave67_model(
    model: SelectiveExactQuotientGPT, stream_or_batches: Any
) -> dict[str, float]:
    """Evaluate `SelectiveExactQuotientGPT` on Shard-0 validation batches (`<= 8` args)."""
    was_training = model.training
    model.eval()
    t0 = time.perf_counter()
    val_batches = stream_or_batches.val_batches if hasattr(stream_or_batches, "val_batches") else stream_or_batches

    total_nll, total_top1, total_top5, total_batches = 0.0, 0.0, 0.0, 0
    for idx_in, targets in val_batches:
        loss_t, m_dict = model.forward_loss_and_metrics(idx_in, targets)
        total_nll += float(loss_t.item())
        total_top1 += float(m_dict["top1_acc"])
        total_top5 += float(m_dict["top5_acc"])
        total_batches += 1

    if was_training:
        model.train()
    denom = float(max(1, total_batches))
    val_ce = total_nll / denom
    telem = model.get_manifold_telemetry()
    return {
        "val_ce": round(val_ce, 4),
        "val_ppl": round(math.exp(min(20.0, val_ce)), 2),
        "top1_acc": round(total_top1 / denom, 4),
        "top5_acc": round(total_top5 / denom, 4),
        "eval_wall_sec": round(time.perf_counter() - t0, 2),
        **telem,
    }


def run_wave67_selective_quotient_comparison_suite(
    stream: LongSeqFineWebStream,
    steps: int = 60,
    eval_interval: int = 20,
    wave_id: str = "W67",
    num_layers: int = DEFAULT_TOMO_NUM_LAYERS,
    grad_accum_steps: int = 1,
    only_variant: str = "",
) -> dict[str, Any]:
    """Run the Wave 67 Selective Exact Quotient, Read-Only Shadow EMA, & Wide `256D` Readout comparison suite (`<= 8` args)."""
    device = stream.device
    k_accum = max(1, int(grad_accum_steps))
    tokens_per_step = k_accum * stream.batch_size * stream.seq_len
    quench_steps = max(1, int(round(float(steps) * float(DEFAULT_QUENCH_FRACTION))))
    plateau_steps = max(1, steps - quench_steps)

    modes: list[tuple[Wave67SelectiveVariantMode, str]] = [
        (Wave67SelectiveVariantMode.W67_S0_AMBIENT_RATCHET_BASELINE, "W66_G0 Reference: Ambient Stiefel 128D + Separate MUB U1/U2 + Standard Ratchet Thermostat"),
        (Wave67SelectiveVariantMode.W67_S1_SELECTIVE_EXACT_QUOTIENT_128D, "Selective Exact Quotient 128D (su(16) MUB + Lyapunov u(R) Kraus + Degeneracy SVD + Tied Bridge) + Read-Only Shadow EMA"),
        (Wave67SelectiveVariantMode.W67_S2_SELECTIVE_QUOTIENT_256D_READOUT, "W67_S1 + Widened 256D Tied Stiefel Subspace Frame (Eliminating Rank-128 Readout Bottleneck)"),
        (Wave67SelectiveVariantMode.W67_S3_SELECTIVE_QUOTIENT_256D_WIDE_TRUNK, "W67_S2 + Widened Quantum Trunk Capacity (Rank-4 CPTP Kraus Channels + 1536D Mach-Zehnder MLP)"),
    ]
    filter_str = only_variant.strip().upper()
    if filter_str:
        modes = [m for m in modes if filter_str in m[0].value.upper()]

    results: list[dict[str, Any]] = []
    for mode, desc in modes:
        spec = build_wave67_variant_spec(mode, num_layers=int(num_layers), seed=1337)
        torch.manual_seed(spec.seed)
        model = SelectiveExactQuotientGPT(spec).to(device=device)
        init_ms = model.initialize_model_()
        total_params = sum(p.numel() for p in model.parameters())

        sfx = f"_l{int(num_layers)}_ga{k_accum}" if (int(num_layers) != DEFAULT_TOMO_NUM_LAYERS or k_accum > 1) else ""
        run_key = f"plan087_{mode.value.lower()}_{steps}steps{sfx}"
        intent = ExperimentIntentSpec(
            run_key=run_key, wave_id=wave_id, plan_id="EXPERIMENT_PLAN_087", task_domain="fineweb_speedrun",
            hypothesis_intent=desc,
            architecture_summary=f"SelectiveExactQuotientGPT(mode={mode.value}, L={int(num_layers)}, d_sub={spec.subspace_dim}, R={spec.kraus_rank}, params={total_params:,})",
            hardware_budget=str(device), batch_size=stream.batch_size * k_accum, seq_len=stream.seq_len,
            total_steps=steps, target_tokens_M=round((steps * tokens_per_step) / 1e6, 3),
        )
        exp_logger = StandardizedExperimentLogger(intent)
        optimizers = build_wave67_selective_optimizers(model)
        q_trunk_opt = optimizers[0]
        ema_d = DEFAULT_THERMOSTAT_EMA_DECAY
        retr_cad = 8 if steps >= 1000 else DEFAULT_THERMOSTAT_RETRACT_CADENCE
        thermostat: RiemannianShadowEMAThermostat = (
            ReadOnlyQuenchRiemannianShadowEMA(model, ema_decay=ema_d, eval_fn=evaluate_wave67_model, retract_cadence=retr_cad)
            if spec.enable_read_only_shadow_ema
            else RiemannianShadowEMAThermostat(model, ema_decay=ema_d, eval_fn=evaluate_wave67_model, retract_cadence=retr_cad)
        )

        step0_ev = evaluate_wave67_model(model, stream)
        step0_ev["step"], step0_ev["tokens_M"] = 0, 0.0
        exp_logger.log_eval(0, {**step0_ev, "top1_acc": round(100.0 * step0_ev["top1_acc"], 2), "top5_acc": round(100.0 * step0_ev["top5_acc"], 2)})
        print(
            f"[{mode.value} (init={init_ms:.1f}ms, d_sub={spec.subspace_dim}, R={spec.kraus_rank}, params={total_params:,}, "
            f"mub_err={step0_ev['mub_unbiased_err']:.2e})] Step    0 | Val CE: {step0_ev['val_ce']:.4f}",
            flush=True,
        )

        curve: list[dict[str, Any]] = [step0_ev]
        thermostat_lr_scale: float = 1.0
        warmup_steps = 5
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.train()

        for step in range(1, steps + 1):
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss_acc_t = torch.zeros((), device=device, dtype=torch.float32)
            inv_k = 1.0 / float(k_accum)
            for m_idx in range(k_accum):
                idx_in, targets = stream.sample_train_batch((step - 1) * k_accum + m_idx + 1)
                loss, _ = model.forward_loss_and_metrics(idx_in, targets)
                loss_acc_t = loss_acc_t + loss.detach().float() * inv_k
                (loss * inv_k).backward()
                del loss

            anneal = (1.0 / math.sqrt(1.0 + max(0.0, float(step - 200)) / 300.0)) if steps >= 1000 else 1.0
            if step <= warmup_steps:
                warm = float(step) / float(warmup_steps)
                lr_mult, mom_beta = (0.25 + 0.75 * warm) * thermostat_lr_scale, 0.50 * warm
            elif step <= plateau_steps:
                lr_mult, mom_beta = anneal * thermostat_lr_scale, DEFAULT_SELECTIVE_MOMENTUM_BETA
            else:
                anneal_end = (1.0 / math.sqrt(1.0 + max(0.0, float(plateau_steps - 200)) / 300.0)) if steps >= 1000 else 1.0
                q_p = float(step - plateau_steps) / float(max(1, quench_steps))
                cos_q = 0.5 * (1.0 + math.cos(math.pi * min(1.0, q_p)))
                lr_mult = anneal_end * (0.15 + 0.85 * cos_q) * thermostat_lr_scale
                mom_beta = 0.85 + 0.10 * (1.0 - cos_q)

            step_wave67_optimizers(model, optimizers, tokens_in_step=tokens_per_step, lr_mult=lr_mult, mom_beta=mom_beta)
            thermostat.update_after_step(model)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            if step % 10 == 0 or step == 1:
                el_s = max(1e-3, time.perf_counter() - t0)
                tps = (float(step) * float(tokens_per_step)) / el_s
                exp_logger.log_step(step, train_ce=float(loss_acc_t.item()), tok_per_s=tps, lr_mult=lr_mult)

            if step % eval_interval == 0 or step == steps:
                train_ce = float(loss_acc_t.item())
                th_out = thermostat.check_and_thermostat(
                    model, stream, current_lr_scale=thermostat_lr_scale, step=step, apply_feedback=True
                )
                thermostat_lr_scale = float(th_out["lr_scale_after"])
                gd = {k: round(float(v), 6) for k, v in q_trunk_opt.last_gauge_diagnostics.items()}
                ev = dict(th_out["active_eval"])
                ev["step"], ev["tokens_M"] = step, round((step * tokens_per_step) / 1e6, 3)
                ev["train_ce"] = round(train_ce, 4)
                ev["hot_val_ce"] = th_out["hot_val_ce"]
                ev["ema_val_ce"] = th_out["ema_val_ce"]
                ev["thermal_gap_ce"] = th_out["thermal_gap_ce"]
                ev["thermostat_triggered"] = th_out["triggered"]
                ev["thermostat_lr_scale"] = round(lr_mult, 4)
                ev["retract_ms"] = th_out["retract_ms"]
                ev.update(gd)
                curve.append(ev)
                exp_logger.log_eval(step, {**ev, "top1_acc": round(100.0 * ev["top1_acc"], 2), "top5_acc": round(100.0 * ev["top5_acc"], 2)})
                print(
                    f"[{mode.value}] Step {step:4d} ({ev['tokens_M']:5.2f}M tok, lr={lr_mult:.3f}x) | "
                    f"Hot CE: {th_out['hot_val_ce']:.4f} | EMA CE: {th_out['ema_val_ce']:.4f} "
                    f"(gap={th_out['thermal_gap_ce']:+.4f}, trig={th_out['triggered']}) | "
                    f"Top1: {100.0 * ev['top1_acc']:.2f}% | MUBErr: {ev['mub_unbiased_err']:.2e} | "
                    f"DegSVD: {gd.get('svd_degenerate_spin_frac', 0.0):.2e} | "
                    f"SU16Phase: {gd.get('su16_step_phase_frac', 0.0):.2e} | KrausGauge: {gd.get('kraus_step_gauge_frac', 0.0):.2e}",
                    flush=True,
                )

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
        exp_logger.finalize()
        results.append({
            "variant_id": mode.value, "description": desc, "total_params": total_params,
            "thermostat_triggers": thermostat.trigger_events,
            "gauge_diagnostics": {k: round(float(v), 6) for k, v in q_trunk_opt.last_gauge_diagnostics.items()},
            "elapsed_sec": round(wall_s, 2), "ms_per_step": round((wall_s * 1000.0) / max(1, steps), 2),
            "step0": step0_ev, "final": curve[-1], "curve": curve,
        })
        del model, optimizers, thermostat
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"variants": results}
