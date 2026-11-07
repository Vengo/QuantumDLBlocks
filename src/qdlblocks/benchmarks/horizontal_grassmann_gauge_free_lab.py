r"""Wave 66 (`EXPERIMENT_PLAN_086.md`): Horizontal-Grassmannian Lift, Gauge-Quotiented Coupled SVD/Kraus/$\mathbb{CP}^{15}$ Manifolds, & Spike-Guarded Shadow EMA Thermostat.

Mathematical & Architectural Overview
-------------------------------------
Resolves the four parameter-manifold gauge collisions and the late-horizon thermostat "ratchet trap" identified in
the `229.38M`-token `Wave 65` (`W65_V3_SHARED_STIEFEL_128D_FRAME`) scaling run:

1. **Intrinsic Horizontal-Lift on $\operatorname{Gr}(k, D)$ for Subspace Bridges ($U_{\text{in}}, U_{\text{out}} \in \operatorname{St}(512, 128)$)**:
   - Projects both gradients and momentum buffers onto the **Horizontal Grassmannian Tangent Space**
     $\mathcal{H}_U = \{\Delta \in \mathbb{R}^{D \times k} : U^\top \Delta = 0\}$:
     $$G_{\text{Grassmann}} = (I_D - U U^\top) \nabla_U \mathcal{L}, \qquad M_{\text{Grassmann}} = (I_D - U U^\top) M$$
     stripping out both the symmetric normal drift and the skew-symmetric vertical $O(k)$ gauge-fiber spin
     $U \operatorname{skew}(U^\top G)$ before and after 5-step Newton–Schulz.

2. **Joint $O(r)$ Gauge-Quotient Projection for Coupled Biorthogonal SVD-Stiefel Pairs $(U, V)$**:
   - In every `BiorthogonalSVDStiefelProjection` ($W(x) = ((x V) \odot \vec{\sigma}) U^\top$), quotients out the
     joint right-action gauge fiber $\{(U \Omega, V \Omega) : \Omega \in \mathfrak{so}(r)\}$ via:
     $$\Omega_{\text{gauge}}(\xi_U, \xi_V) = \frac{1}{2}\Big(\operatorname{skew}(U^\top \xi_U) + \operatorname{skew}(V^\top \xi_V)\Big), \qquad (\xi_U^{\text{horiz}}, \xi_V^{\text{horiz}}) = (\xi_U - U \Omega_{\text{gauge}},\; \xi_V - V \Omega_{\text{gauge}})$$
     guaranteeing $\operatorname{skew}(U^\top \xi_U^{\text{horiz}}) + \operatorname{skew}(V^\top \xi_V^{\text{horiz}}) \equiv 0$.

3. **Gauge-Locked $\mathbb{CP}^{15}$ MUB Tomography & $U(R)$-Quotiented CPTP Kraus Channels**:
   - Locks the second MUB probe basis analytically via the $16 \times 16$ unitary DFT matrix $U_h^{(2)} \equiv F_{16} U_h^{(1)}$
     (guaranteeing $|\langle c_{h,m}^{(1)} \mid c_{h,n}^{(2)}\rangle|^2 \equiv \frac{1}{16}$ at every step), zeroes out the
     diagonal vertical $U(1)^{16}$ phase fiber $\operatorname{diag}(\operatorname{Im}(U_h^\dagger \xi)) = 0$ on the Flag Manifold
     $U(16)/U(1)^{16}$, and quotients out the $\mathfrak{u}(R)$ inter-branch Kraus gauge fiber on $\mathbf{K}_h \in \operatorname{St}_{\mathbb{C}}(R d_h, d_h)$.

4. **Spike-Guarded Polyak–Ruppert Shadow EMA Thermostat**:
   - Evaluates the manifold-retracted Shadow EMA weights $W_{\text{ema}}$ at every checkpoint while triggering hard
     active-weight rollbacks $W \leftarrow W_{\text{ema}}$ **only** on genuine hot-loss spikes ($\mathcal{L}_{\text{hot}}(t) > \mathcal{L}_{\text{hot}}(t-1) + \delta_{\text{spike}}$
     or $\mathcal{L}_{\text{hot}}(t) - \mathcal{L}_{\text{ema}}(t) \ge \delta_{\text{cat}}$), preventing the every-250-step
     momentum-killing ratchet trap at long horizons.
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
    compute_tomographic_512d_probe_expectations,
)
from qdlblocks.benchmarks.kraus_mub_thermostat_8l_lab import (
    DEFAULT_ANALYTICAL_BETA_TARGET, DEFAULT_KRAUS_BRANCH_RANK, DEFAULT_THERMOSTAT_EMA_DECAY,
    DEFAULT_THERMOSTAT_GAIN_KAPPA, DEFAULT_THERMOSTAT_GAP_THRESHOLD,
    DEFAULT_THERMOSTAT_MIN_SCALE_STEP, DEFAULT_THERMOSTAT_RETRACT_CADENCE,
    RiemannianShadowEMAThermostat, _KrausDensityHighwayBlock,
    build_initial_kraus_stiefel_operators, build_mub_two_basis_unitary_probes,
    compute_analytical_parseval_beta, compute_mub_two_basis_probe_expectations,
)
from qdlblocks.benchmarks.quantum_subsystem_vocab_lab import (
    DEFAULT_SHARED_VOCAB_SUBSPACE_DIM, SharedStiefelSubspaceFrameBoundary,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import (
    DEFAULT_SHALLOW_NGRAM_LR, DEFAULT_SHALLOW_READOUT_LR, DEFAULT_SHALLOW_SCALAR_LR,
    DEFAULT_SHALLOW_STIEFEL_LR, DEFAULT_SHALLOW_VOCAB_SIZE, DEFAULT_SHALLOW_WTE_LR,
    LongSeqFineWebStream,
)
from qdlblocks.layers.impure_density_mirror_block import (
    DEFAULT_NS5_STEPS, DEFAULT_TIKHONOV_GRAM_LAMBDA, TRACE_NORM_EPS,
    _complex_newton_schulz5, _newton_schulz5_tall_real, complex_l2_normalize,
    unpack_hermitian_isometric,
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
# Top-of-File Named Constants (Rule 7 — Synchronized with EXPERIMENT_PLAN_086.md)
# ==============================================================================
DEFAULT_GRASSMANN_SUBSPACE_DIM: int = DEFAULT_SHARED_VOCAB_SUBSPACE_DIM
DEFAULT_COUPLED_GAUGE_QUOTIENT_SCALE: float = 0.5
DEFAULT_HORIZ_MOMENTUM_BETA: float = 0.92
DEFAULT_SPIKE_ROLLBACK_TOL_NATS: float = 0.030
DEFAULT_CATASTROPHIC_GAP_TOL_NATS: float = 0.042
DEFAULT_GENTLE_COOL_GAP_NATS: float = 0.022
DEFAULT_GENTLE_LR_DECAY_FACTOR: float = 0.96


class Wave66GaugeVariantMode(str, Enum):
    """Ablation variants for `EXPERIMENT_PLAN_086.md` (`Wave 66`, Rule 3)."""

    W66_G0_STIEFEL_AMBIENT_BASELINE = "W66_G0_STIEFEL_AMBIENT_BASELINE"
    W66_G1_HORIZONTAL_GRASSMANN_VOCAB_AND_BRIDGES = "W66_G1_HORIZONTAL_GRASSMANN_VOCAB_AND_BRIDGES"
    W66_G2_GAUGE_LOCKED_CP15_MUB_AND_KRAUS = "W66_G2_GAUGE_LOCKED_CP15_MUB_AND_KRAUS"
    W66_G3_FULL_GAUGE_FREE_SPIKE_ONLY_THERMOSTAT = "W66_G3_FULL_GAUGE_FREE_SPIKE_ONLY_THERMOSTAT"


@dataclass(frozen=True)
class Wave66GaugeFreeSpec:
    r"""Configuration for `HorizontalGrassmannGaugeFreeGPT` (`<= 8` fields, Rule 2)."""

    variant_mode: Wave66GaugeVariantMode = Wave66GaugeVariantMode.W66_G3_FULL_GAUGE_FREE_SPIKE_ONLY_THERMOSTAT
    num_layers: int = DEFAULT_TOMO_NUM_LAYERS
    kraus_rank: int = DEFAULT_KRAUS_BRANCH_RANK
    subspace_dim: int = DEFAULT_GRASSMANN_SUBSPACE_DIM
    enable_horizontal_grassmann: bool = True
    enable_gauge_locked_mub_kraus: bool = True
    enable_spike_guarded_thermostat: bool = True
    seed: int = 1337


def build_wave66_variant_spec(
    variant: Wave66GaugeVariantMode | str,
    num_layers: int = DEFAULT_TOMO_NUM_LAYERS,
    seed: int = 1337,
) -> Wave66GaugeFreeSpec:
    """Resolve `Wave66GaugeFreeSpec` from `Wave66GaugeVariantMode` (`<= 8` args)."""
    mode = Wave66GaugeVariantMode(variant)
    if mode == Wave66GaugeVariantMode.W66_G0_STIEFEL_AMBIENT_BASELINE:
        return Wave66GaugeFreeSpec(
            variant_mode=mode, num_layers=num_layers,
            enable_horizontal_grassmann=False, enable_gauge_locked_mub_kraus=False,
            enable_spike_guarded_thermostat=False, seed=seed,
        )
    if mode == Wave66GaugeVariantMode.W66_G1_HORIZONTAL_GRASSMANN_VOCAB_AND_BRIDGES:
        return Wave66GaugeFreeSpec(
            variant_mode=mode, num_layers=num_layers,
            enable_horizontal_grassmann=True, enable_gauge_locked_mub_kraus=False,
            enable_spike_guarded_thermostat=False, seed=seed,
        )
    if mode == Wave66GaugeVariantMode.W66_G2_GAUGE_LOCKED_CP15_MUB_AND_KRAUS:
        return Wave66GaugeFreeSpec(
            variant_mode=mode, num_layers=num_layers,
            enable_horizontal_grassmann=True, enable_gauge_locked_mub_kraus=True,
            enable_spike_guarded_thermostat=False, seed=seed,
        )
    return Wave66GaugeFreeSpec(
        variant_mode=mode, num_layers=num_layers,
        enable_horizontal_grassmann=True, enable_gauge_locked_mub_kraus=True,
        enable_spike_guarded_thermostat=True, seed=seed,
    )


# ==============================================================================
# 1. Intrinsic Horizontal-Grassmannian & Coupled-SVD Gauge-Quotient Primitives
# ==============================================================================
def project_real_grassmann_horizontal(vec: torch.Tensor, v_tall: torch.Tensor) -> torch.Tensor:
    r"""Project `vec` `(..., D, k)` ($D > k$) onto the Horizontal Grassmannian space $\mathcal{H}_V = (I_D - V V^\top)\,\text{vec}$ (`<= 8` args).

    Removes both the symmetric normal drift $V \operatorname{sym}(V^\top \text{vec})$ and the vertical $O(k)$
    gauge-fiber spin $V \operatorname{skew}(V^\top \text{vec})$ so that $V^\top G_{\text{Grassmann}} \equiv 0$.
    """
    return vec - torch.matmul(v_tall, torch.matmul(v_tall.transpose(-2, -1), vec))


def compute_grassmann_vertical_spin_fraction(vec: torch.Tensor, v_tall: torch.Tensor) -> float:
    r"""Return $\|V \operatorname{skew}(V^\top \text{vec})\|_F / \|\text{vec}\|_F$ measuring vertical $O(k)$ gauge-fiber spin (`<= 8` args)."""
    vf = vec.float()
    vt_v = torch.matmul(v_tall.float().transpose(-2, -1), vf)
    skew = 0.5 * (vt_v - vt_v.transpose(-2, -1))
    vert = torch.matmul(v_tall.float(), skew)
    num = torch.linalg.matrix_norm(vert, ord="fro")
    den = torch.linalg.matrix_norm(vf, ord="fro").clamp_min(1e-8)
    return float((num / den).mean().item())


def project_coupled_svd_pair_horizontal(
    xi_u: torch.Tensor, xi_v: torch.Tensor, u_mat: torch.Tensor, v_mat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Project `(xi_u, xi_v)` onto $T_U\operatorname{St} \times T_V\operatorname{St}$ and quotient out the joint $O(r)$ gauge fiber (`<= 8` args).

    For $W(x) = ((x V) \odot \vec{\sigma}) U^\top$, simultaneous right-rotation $(U Q, V Q)$ ($Q \in O(r)$) generates
    the vertical gauge orbit $\{(U\Omega, V\Omega) : \Omega \in \mathfrak{so}(r)\}$. Subtracting
    $\Omega_{\text{gauge}} = \frac{1}{2}(\operatorname{skew}(U^\top \xi_U) + \operatorname{skew}(V^\top \xi_V))$
    enforces $\operatorname{skew}(U^\top \xi_U^{\text{horiz}}) + \operatorname{skew}(V^\top \xi_V^{\text{horiz}}) \equiv 0$.
    """
    tu = project_real_stiefel_tangent(xi_u, u_mat)
    tv = project_real_stiefel_tangent(xi_v, v_mat)
    ut_tu = torch.matmul(u_mat.transpose(-2, -1), tu)
    vt_tv = torch.matmul(v_mat.transpose(-2, -1), tv)
    skew_u = 0.5 * (ut_tu - ut_tu.transpose(-2, -1))
    skew_v = 0.5 * (vt_tv - vt_tv.transpose(-2, -1))
    omega_gauge = float(DEFAULT_COUPLED_GAUGE_QUOTIENT_SCALE) * (skew_u + skew_v)
    return tu - torch.matmul(u_mat, omega_gauge), tv - torch.matmul(v_mat, omega_gauge)


def compute_coupled_svd_joint_spin_fraction(
    xi_u: torch.Tensor, xi_v: torch.Tensor, u_mat: torch.Tensor, v_mat: torch.Tensor
) -> float:
    r"""Return $\|\operatorname{skew}(U^\top \xi_U) + \operatorname{skew}(V^\top \xi_V)\|_F / \sqrt{\|\xi_U\|_F^2 + \|\xi_V\|_F^2}$ (`<= 8` args)."""
    ut_u = torch.matmul(u_mat.float().transpose(-2, -1), xi_u.float())
    vt_v = torch.matmul(v_mat.float().transpose(-2, -1), xi_v.float())
    joint_skew = 0.5 * ((ut_u - ut_u.transpose(-2, -1)) + (vt_v - vt_v.transpose(-2, -1)))
    num = torch.linalg.matrix_norm(joint_skew, ord="fro")
    den = (xi_u.float().square().sum(dim=(-2, -1)) + xi_v.float().square().sum(dim=(-2, -1))).sqrt().clamp_min(1e-8)
    return float((num / den).mean().item())


# ==============================================================================
# 2. Complex Projective CP^{15} MUB & U(R)-Quotiented Kraus Primitives
# ==============================================================================
def project_cp15_flag_horizontal(
    v_re: torch.Tensor, v_im: torch.Tensor, u_re: torch.Tensor, u_im: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Project `(v_re, v_im)` onto the Horizontal Flag Manifold $U(d_h)/U(1)^{d_h}$ (`\operatorname{diag}(\operatorname{Im}(U^\dagger \xi)) = 0`, `<= 8` args)."""
    ut_re = u_re.transpose(-2, -1)
    ut_im = -u_im.transpose(-2, -1)
    p_re = torch.matmul(ut_re, v_re) - torch.matmul(ut_im, v_im)
    p_im = torch.matmul(ut_re, v_im) + torch.matmul(ut_im, v_re)
    omega_re = 0.5 * (p_re - p_re.transpose(-2, -1))
    omega_im = 0.5 * (p_im + p_im.transpose(-2, -1))
    diag_mask = 1.0 - torch.eye(omega_im.size(-1), device=omega_im.device, dtype=omega_im.dtype)
    omega_im_zero_diag = omega_im * diag_mask
    h_re = torch.matmul(u_re, omega_re) - torch.matmul(u_im, omega_im_zero_diag)
    h_im = torch.matmul(u_re, omega_im_zero_diag) + torch.matmul(u_im, omega_re)
    return h_re, h_im


def compute_cp15_diag_phase_fraction(
    v_re: torch.Tensor, v_im: torch.Tensor, u_re: torch.Tensor, u_im: torch.Tensor
) -> float:
    r"""Measure fraction of complex tangent norm residing in the vertical $U(1)^{d_h}$ diagonal phase fiber (`<= 8` args)."""
    ut_re = u_re.float().transpose(-2, -1)
    ut_im = -u_im.float().transpose(-2, -1)
    vr, vi = v_re.float(), v_im.float()
    p_im = torch.matmul(ut_re, vi) + torch.matmul(ut_im, vr)
    omega_im = 0.5 * (p_im + p_im.transpose(-2, -1))
    diag_im = torch.diagonal(omega_im, dim1=-2, dim2=-1)
    num = diag_im.square().sum(dim=-1).sqrt()
    den = (vr.square().sum(dim=(-2, -1)) + vi.square().sum(dim=(-2, -1))).sqrt().clamp_min(1e-8)
    return float((num / den).mean().item())


def project_kraus_branch_gauge_horizontal(
    v_re: torch.Tensor, v_im: torch.Tensor, k_re: torch.Tensor, k_im: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Project `(v_re, v_im)` onto $T_{\mathbf{K}}\operatorname{St}_{\mathbb{C}}(R d_h, d_h)$ and quotient out the left $\mathfrak{u}(R)$ branch-mixing fiber (`<= 8` args).

    For $\mathcal{E}_h(\rho) = \sum_{r=1}^R K_{h,r}\rho K_{h,r}^\dagger$, any $A \in \mathfrak{u}(R)$ ($A^\dagger = -A$) generates a
    vertical gauge tangent $\delta K_{\text{flat}} = A K_{\text{flat}}$ which automatically lies inside
    $T_{\mathbf{K}}\operatorname{St}_{\mathbb{C}}(R d_h, d_h)$ because $\mathbf{K}^\dagger (A \otimes I_{d_h})\mathbf{K}$ is skew-Hermitian.
    Solving the $R \times R$ Lyapunov equation $\frac{1}{2}(A G_K + G_K A) = S_{\text{skew}}(\Xi)$ with $G_K = K_{\text{flat}} K_{\text{flat}}^\dagger$
    yields $\Xi^{\text{horiz}} = \Xi - A K_{\text{flat}}$ satisfying $S_{\text{skew}}(\Xi^{\text{horiz}}) \equiv 0$ to machine precision.
    """
    t_re, t_im = project_complex_stiefel_tangent(v_re, v_im, k_re, k_im)
    num_h, rd_h, d_h = k_re.shape
    r_rank = rd_h // d_h
    if r_rank <= 1:
        return t_re, t_im

    kr_flat = k_re.float().view(num_h, r_rank, d_h * d_h)
    ki_flat = k_im.float().view(num_h, r_rank, d_h * d_h)
    tr_flat = t_re.float().view(num_h, r_rank, d_h * d_h)
    ti_flat = t_im.float().view(num_h, r_rank, d_h * d_h)

    k_c = torch.complex(kr_flat, ki_flat)
    t_c = torch.complex(tr_flat, ti_flat)
    eye_r = torch.eye(r_rank, device=k_re.device, dtype=torch.float32).unsqueeze(0)
    gram_k = torch.matmul(k_c, k_c.conj().transpose(-2, -1)) + (1e-5 * eye_r).to(k_c.dtype)
    m_c = torch.matmul(t_c, k_c.conj().transpose(-2, -1))
    s_skew = 0.5 * (m_c - m_c.conj().transpose(-2, -1))

    evals, evecs = torch.linalg.eigh(gram_k)
    evals_clamped = evals.clamp_min(1e-5)
    denom = evals_clamped.unsqueeze(-1) + evals_clamped.unsqueeze(-2)
    s_eig = torch.matmul(evecs.conj().transpose(-2, -1), torch.matmul(s_skew, evecs))
    a_eig = (2.0 * s_eig) / denom.to(s_eig.dtype)
    a_c = torch.matmul(evecs, torch.matmul(a_eig, evecs.conj().transpose(-2, -1)))
    a_c = 0.5 * (a_c - a_c.conj().transpose(-2, -1))

    horiz_c = t_c - torch.matmul(a_c, k_c)
    return horiz_c.real.view(num_h, rd_h, d_h).to(v_re.dtype), horiz_c.imag.view(num_h, rd_h, d_h).to(v_im.dtype)


def compute_kraus_branch_gauge_fraction(
    v_re: torch.Tensor, v_im: torch.Tensor, k_re: torch.Tensor, k_im: torch.Tensor
) -> float:
    r"""Measure normalized $\mathfrak{u}(R)$ branch-mixing gauge moment $\|\operatorname{skew}_{\mathbb{C}}(\Xi_{\text{flat}} K_{\text{flat}}^\dagger)\|_F / (\|\Xi\|_F \|K_{\text{flat}}\|_F)$ (`<= 8` args)."""
    num_h, rd_h, d_h = k_re.shape
    r_rank = rd_h // d_h
    if r_rank <= 1:
        return 0.0
    kr_flat = k_re.float().view(num_h, r_rank, d_h * d_h)
    ki_flat = k_im.float().view(num_h, r_rank, d_h * d_h)
    vr_flat = v_re.float().view(num_h, r_rank, d_h * d_h)
    vi_flat = v_im.float().view(num_h, r_rank, d_h * d_h)
    m_re = torch.matmul(vr_flat, kr_flat.transpose(-2, -1)) + torch.matmul(vi_flat, ki_flat.transpose(-2, -1))
    m_im = torch.matmul(vi_flat, kr_flat.transpose(-2, -1)) - torch.matmul(vr_flat, ki_flat.transpose(-2, -1))
    s_re = 0.5 * (m_re - m_re.transpose(-2, -1))
    s_im = 0.5 * (m_im + m_im.transpose(-2, -1))
    num = (s_re.square().sum(dim=(-2, -1)) + s_im.square().sum(dim=(-2, -1))).sqrt()
    v_norm = (vr_flat.square().sum(dim=(-2, -1)) + vi_flat.square().sum(dim=(-2, -1))).sqrt().clamp_min(1e-8)
    k_norm = (kr_flat.square().sum(dim=(-2, -1)) + ki_flat.square().sum(dim=(-2, -1))).sqrt().clamp_min(1e-8)
    return float((num / (v_norm * k_norm)).mean().item())


def compute_gauge_locked_mub_probe_expectations(
    rho_packed: torch.Tensor, u1_re: torch.Tensor, u1_im: torch.Tensor, dft_re: torch.Tensor, dft_im: torch.Tensor
) -> torch.Tensor:
    r"""Evaluate $16 \times 32 = 512$ Born expectations with analytically gauge-locked second MUB basis $U_h^{(2)} \equiv F_{16} U_h^{(1)}$ (`<= 8` args)."""
    u2_re = torch.matmul(dft_re, u1_re) - torch.matmul(dft_im, u1_im)
    u2_im = torch.matmul(dft_re, u1_im) + torch.matmul(dft_im, u1_re)
    return compute_tomographic_512d_probe_expectations(
        rho_packed, torch.cat([u1_re, u2_re], dim=1), torch.cat([u1_im, u2_im], dim=1)
    )


def compute_mub_unbiasedness_max_error(
    u1_re: torch.Tensor, u1_im: torch.Tensor, u2_re: torch.Tensor, u2_im: torch.Tensor
) -> float:
    r"""Return $\max_{h,m,n} \big| |\langle c_{h,m}^{(1)} \mid c_{h,n}^{(2)}\rangle|^2 - \frac{1}{d_h} \big|$ (`<= 8` args)."""
    d_h = u1_re.size(-1)
    ov_re = torch.matmul(u1_re.float(), u2_re.float().transpose(-2, -1)) + torch.matmul(u1_im.float(), u2_im.float().transpose(-2, -1))
    ov_im = torch.matmul(u1_im.float(), u2_re.float().transpose(-2, -1)) - torch.matmul(u1_re.float(), u2_im.float().transpose(-2, -1))
    mod_sq = ov_re.square() + ov_im.square()
    return float((mod_sq - (1.0 / float(d_h))).abs().max().item())


# ==============================================================================
# 3. Horizontal-Grassmannian & Gauge-Quotiented Riemannian Optimizer
# ==============================================================================
def _retract_real_tall_stiefel_step(w_tall: torch.Tensor, u_tan: torch.Tensor, lr: float) -> torch.Tensor:
    """Retract `w_tall` `(B, D, k)` along `u_tan` via normalized chord + 2-step quadratic polar refinement (`<= 8` args)."""
    cols = w_tall.size(-1)
    scale = 1.0 / math.sqrt(1.0 + float(lr) * float(lr))
    w_next = (w_tall - float(lr) * u_tan) * scale
    eye = torch.eye(cols, device=w_next.device, dtype=w_next.dtype)
    for _ in range(2):
        w_next = 0.5 * torch.matmul(w_next, 3.0 * eye - torch.matmul(w_next.transpose(-2, -1), w_next))
    return w_next


def _retract_complex_tall_stiefel_step(
    a_re: torch.Tensor, a_im: torch.Tensor, tan_re: torch.Tensor, tan_im: torch.Tensor, lr: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retract complex `(a_re, a_im)` `(B, D, d)` along `(tan_re, tan_im)` via 2-step complex polar refinement (`<= 8` args)."""
    scale = 1.0 / math.sqrt(1.0 + float(lr) * float(lr))
    nx_re = (a_re - float(lr) * tan_re) * scale
    nx_im = (a_im - float(lr) * tan_im) * scale
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


class HorizontalGrassmannGaugeFreeMuon(torch.optim.Optimizer):
    r"""Riemannian Optimizer enforcing Horizontal-Grassmannian Lift on $\operatorname{Gr}(k, D)$, Joint $O(r)$ SVD Gauge Quotient, $\mathbb{CP}^{15}$ Flag Projection, and $\mathfrak{u}(R)$ Kraus Quotient."""

    def __init__(
        self,
        grassmann_params: list[nn.Parameter],
        coupled_svd_pairs: list[tuple[nn.Parameter, nn.Parameter]],
        kraus_pairs: list[tuple[nn.Parameter, nn.Parameter]],
        mub_flag_pairs: list[tuple[nn.Parameter, nn.Parameter]],
        stiefel_lr: float = DEFAULT_SHALLOW_STIEFEL_LR,
        unitary_lr: float = DEFAULT_SHALLOW_STIEFEL_LR,
        beta: float = DEFAULT_HORIZ_MOMENTUM_BETA,
    ) -> None:
        all_params: list[nn.Parameter] = list(grassmann_params)
        for u_p, v_p in coupled_svd_pairs:
            all_params.extend([u_p, v_p])
        for kr_p, ki_p in kraus_pairs:
            all_params.extend([kr_p, ki_p])
        for mr_p, mi_p in mub_flag_pairs:
            all_params.extend([mr_p, mi_p])
        defaults = dict(
            lr=stiefel_lr, stiefel_lr=stiefel_lr, unitary_lr=unitary_lr,
            beta=beta, ns_steps=DEFAULT_NS5_STEPS, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA,
        )
        super().__init__(all_params, defaults)
        self.grassmann_params = list(grassmann_params)
        self.coupled_svd_pairs = list(coupled_svd_pairs)
        self.kraus_pairs = list(kraus_pairs)
        self.mub_flag_pairs = list(mub_flag_pairs)
        self.enable_horizontal_grassmann: bool = True
        self.enable_gauge_locked_mub_kraus: bool = True
        self.last_gauge_diagnostics: dict[str, float] = {}

    @torch.no_grad()
    def _step_grassmann_params(self, stiefel_lr: float, beta: float, ns_steps: int, tik_lam: float) -> None:
        r"""Update tall Stiefel embeddings on $\operatorname{Gr}(k, D)$ via horizontal projection $G_{\text{Gr}} = (I_D - U U^\top) G$ (`<= 8` args)."""
        active = [p for p in self.grassmann_params if p.grad is not None]
        if not active:
            return
        w_stack = torch.stack([p.data.float() for p in active], dim=0)
        g_stack = torch.stack([torch.nan_to_num(p.grad.float(), 0.0, 0.0, 0.0) for p in active], dim=0)
        m_list = [self.state[p].setdefault("momentum_buffer", torch.zeros_like(p, dtype=torch.float32)) for p in active]
        m_stack = torch.stack(m_list, dim=0)

        raw_spin = compute_grassmann_vertical_spin_fraction(g_stack, w_stack)
        if self.enable_horizontal_grassmann:
            g_proj = project_real_grassmann_horizontal(g_stack, w_stack)
            m_prev = project_real_grassmann_horizontal(m_stack, w_stack)
            m_upd = float(beta) * m_prev + g_proj
            u_ns = _newton_schulz5_tall_real(g_proj + float(beta) * m_upd, steps=ns_steps, tikhonov_lambda=tik_lam)
            u_tan = project_real_grassmann_horizontal(u_ns, w_stack)
            w_next = _retract_real_tall_stiefel_step(w_stack, u_tan, stiefel_lr)
            m_stored = project_real_grassmann_horizontal(m_upd, w_next)
        else:
            m_upd = float(beta) * m_stack + g_stack
            u_in = project_real_stiefel_tangent(g_stack + float(beta) * m_upd, w_stack)
            u_ns = _newton_schulz5_tall_real(u_in, steps=ns_steps, tikhonov_lambda=tik_lam)
            u_tan = project_real_stiefel_tangent(u_ns, w_stack)
            w_next = _retract_real_tall_stiefel_step(w_stack, u_tan, stiefel_lr)
            m_stored = m_upd

        self.last_gauge_diagnostics["grassmann_raw_grad_spin_frac"] = raw_spin
        self.last_gauge_diagnostics["grassmann_step_spin_frac"] = compute_grassmann_vertical_spin_fraction(u_tan, w_stack)
        self.last_gauge_diagnostics["grassmann_mom_spin_frac"] = compute_grassmann_vertical_spin_fraction(m_stored, w_next)
        torch._foreach_copy_([p.data for p in active], list(w_next.unbind(0)))
        torch._foreach_copy_(m_list, list(m_stored.unbind(0)))

    @torch.no_grad()
    def _step_coupled_svd_pairs(self, stiefel_lr: float, beta: float, ns_steps: int, tik_lam: float) -> None:
        """Update coupled `(U, V)` pairs in `BiorthogonalSVDStiefelProjection` with joint $O(r)$ gauge quotient (`<= 8` args)."""
        buckets: dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[nn.Parameter, nn.Parameter]]] = {}
        for u_p, v_p in self.coupled_svd_pairs:
            if u_p.grad is not None and v_p.grad is not None:
                key = ((int(u_p.size(0)), int(u_p.size(1))), (int(v_p.size(0)), int(v_p.size(1))))
                buckets.setdefault(key, []).append((u_p, v_p))

        raw_spins, step_spins, mom_spins = [], [], []
        for _, pairs in buckets.items():
            u_stack = torch.stack([u.data.float() for u, _ in pairs], dim=0)
            v_stack = torch.stack([v.data.float() for _, v in pairs], dim=0)
            gu_stack = torch.stack([torch.nan_to_num(u.grad.float(), 0.0, 0.0, 0.0) for u, _ in pairs], dim=0)
            gv_stack = torch.stack([torch.nan_to_num(v.grad.float(), 0.0, 0.0, 0.0) for _, v in pairs], dim=0)
            mu_list = [self.state[u].setdefault("momentum_buffer", torch.zeros_like(u, dtype=torch.float32)) for u, _ in pairs]
            mv_list = [self.state[v].setdefault("momentum_buffer", torch.zeros_like(v, dtype=torch.float32)) for _, v in pairs]
            mu_stack = torch.stack(mu_list, dim=0)
            mv_stack = torch.stack(mv_list, dim=0)

            raw_spins.append(compute_coupled_svd_joint_spin_fraction(gu_stack, gv_stack, u_stack, v_stack))
            if self.enable_horizontal_grassmann:
                gu_h, gv_h = project_coupled_svd_pair_horizontal(gu_stack, gv_stack, u_stack, v_stack)
                mu_pre, mv_pre = project_coupled_svd_pair_horizontal(mu_stack, mv_stack, u_stack, v_stack)
                mu_upd = float(beta) * mu_pre + gu_h
                mv_upd = float(beta) * mv_pre + gv_h
                zu_ns = _newton_schulz5_tall_real(gu_h + float(beta) * mu_upd, steps=ns_steps, tikhonov_lambda=tik_lam)
                zv_ns = _newton_schulz5_tall_real(gv_h + float(beta) * mv_upd, steps=ns_steps, tikhonov_lambda=tik_lam)
                tu_tan, tv_tan = project_coupled_svd_pair_horizontal(zu_ns, zv_ns, u_stack, v_stack)
                u_next = _retract_real_tall_stiefel_step(u_stack, tu_tan, stiefel_lr)
                v_next = _retract_real_tall_stiefel_step(v_stack, tv_tan, stiefel_lr)
                mu_stored, mv_stored = project_coupled_svd_pair_horizontal(mu_upd, mv_upd, u_next, v_next)
            else:
                mu_upd = float(beta) * mu_stack + gu_stack
                mv_upd = float(beta) * mv_stack + gv_stack
                tu_in = project_real_stiefel_tangent(gu_stack + float(beta) * mu_upd, u_stack)
                tv_in = project_real_stiefel_tangent(gv_stack + float(beta) * mv_upd, v_stack)
                zu_ns = _newton_schulz5_tall_real(tu_in, steps=ns_steps, tikhonov_lambda=tik_lam)
                zv_ns = _newton_schulz5_tall_real(tv_in, steps=ns_steps, tikhonov_lambda=tik_lam)
                tu_tan = project_real_stiefel_tangent(zu_ns, u_stack)
                tv_tan = project_real_stiefel_tangent(zv_ns, v_stack)
                u_next = _retract_real_tall_stiefel_step(u_stack, tu_tan, stiefel_lr)
                v_next = _retract_real_tall_stiefel_step(v_stack, tv_tan, stiefel_lr)
                mu_stored, mv_stored = mu_upd, mv_upd

            step_spins.append(compute_coupled_svd_joint_spin_fraction(tu_tan, tv_tan, u_stack, v_stack))
            mom_spins.append(compute_coupled_svd_joint_spin_fraction(mu_stored, mv_stored, u_next, v_next))
            torch._foreach_copy_([u.data for u, _ in pairs], list(u_next.unbind(0)))
            torch._foreach_copy_([v.data for _, v in pairs], list(v_next.unbind(0)))
            torch._foreach_copy_(mu_list, list(mu_stored.unbind(0)))
            torch._foreach_copy_(mv_list, list(mv_stored.unbind(0)))

        if raw_spins:
            self.last_gauge_diagnostics["coupled_svd_raw_spin_frac"] = float(sum(raw_spins) / len(raw_spins))
            self.last_gauge_diagnostics["coupled_svd_step_spin_frac"] = float(sum(step_spins) / len(step_spins))
            self.last_gauge_diagnostics["coupled_svd_mom_spin_frac"] = float(sum(mom_spins) / len(mom_spins))

    @torch.no_grad()
    def _step_complex_kraus_and_mub(self, unitary_lr: float, beta: float, ns_steps: int, tik_lam: float) -> None:
        r"""Update complex Rank-$R$ Kraus channels ($\mathfrak{u}(R)$ quotient) and MUB probes ($\mathbb{CP}^{15}$ Flag quotient) (`<= 8` args)."""
        for is_mub, pair_list in ((False, self.kraus_pairs), (True, self.mub_flag_pairs)):
            active = [(ur, ui) for ur, ui in pair_list if ur.grad is not None and ui.grad is not None]
            if not active:
                continue
            ur_cat = torch.cat([ur.data.float() for ur, _ in active], dim=0)
            ui_cat = torch.cat([ui.data.float() for _, ui in active], dim=0)
            gr_cat = torch.cat([torch.nan_to_num(ur.grad.float(), 0.0, 0.0, 0.0) for ur, _ in active], dim=0)
            gi_cat = torch.cat([torch.nan_to_num(ui.grad.float(), 0.0, 0.0, 0.0) for _, ui in active], dim=0)
            mr_list = [self.state[ur].setdefault("momentum_buffer", torch.zeros_like(ur, dtype=torch.float32)) for ur, _ in active]
            mi_list = [self.state[ui].setdefault("momentum_buffer", torch.zeros_like(ui, dtype=torch.float32)) for _, ui in active]
            mr_cat = torch.cat(mr_list, dim=0)
            mi_cat = torch.cat(mi_list, dim=0)

            proj_fn = project_cp15_flag_horizontal if is_mub else project_kraus_branch_gauge_horizontal
            if self.enable_gauge_locked_mub_kraus:
                gr_h, gi_h = proj_fn(gr_cat, gi_cat, ur_cat, ui_cat)
                mr_pre, mi_pre = proj_fn(mr_cat, mi_cat, ur_cat, ui_cat)
                mr_upd = float(beta) * mr_pre + gr_h
                mi_upd = float(beta) * mi_pre + gi_h
                zr, zi = _complex_newton_schulz5(
                    gr_h + float(beta) * mr_upd, gi_h + float(beta) * mi_upd, steps=ns_steps, tikhonov_lambda=tik_lam
                )
                tan_re, tan_im = proj_fn(zr, zi, ur_cat, ui_cat)
                ur_next, ui_next = _retract_complex_tall_stiefel_step(ur_cat, ui_cat, tan_re, tan_im, unitary_lr)
                mr_stored, mi_stored = proj_fn(mr_upd, mi_upd, ur_next, ui_next)
            else:
                mr_upd = float(beta) * mr_cat + gr_cat
                mi_upd = float(beta) * mi_cat + gi_cat
                tr, ti = project_complex_stiefel_tangent(gr_cat + float(beta) * mr_upd, gi_cat + float(beta) * mi_upd, ur_cat, ui_cat)
                zr, zi = _complex_newton_schulz5(tr, ti, steps=ns_steps, tikhonov_lambda=tik_lam)
                tan_re, tan_im = project_complex_stiefel_tangent(zr, zi, ur_cat, ui_cat)
                ur_next, ui_next = _retract_complex_tall_stiefel_step(ur_cat, ui_cat, tan_re, tan_im, unitary_lr)
                mr_stored, mi_stored = mr_upd, mi_upd

            if is_mub:
                self.last_gauge_diagnostics["cp15_step_phase_frac"] = compute_cp15_diag_phase_fraction(tan_re, tan_im, ur_cat, ui_cat)
                self.last_gauge_diagnostics["cp15_mom_phase_frac"] = compute_cp15_diag_phase_fraction(mr_stored, mi_stored, ur_next, ui_next)
            else:
                self.last_gauge_diagnostics["kraus_step_gauge_frac"] = compute_kraus_branch_gauge_fraction(tan_re, tan_im, ur_cat, ui_cat)
                self.last_gauge_diagnostics["kraus_mom_gauge_frac"] = compute_kraus_branch_gauge_fraction(mr_stored, mi_stored, ur_next, ui_next)

            splits = [ur.size(0) for ur, _ in active]
            torch._foreach_copy_([ur.data for ur, _ in active], list(ur_next.split(splits, dim=0)))
            torch._foreach_copy_([ui.data for _, ui in active], list(ui_next.split(splits, dim=0)))
            torch._foreach_copy_(mr_list, list(mr_stored.split(splits, dim=0)))
            torch._foreach_copy_(mi_list, list(mi_stored.split(splits, dim=0)))

    def set_momentum(self, beta: float) -> None:
        """Set horizontal Riemannian momentum factor `beta`."""
        for g in self.param_groups:
            g["beta"] = float(beta)

    def set_lr_scale(self, lr_scale: float) -> None:
        """Set learning-rate multiplier `lr_scale` across Stiefel and Unitary steps."""
        for g in self.param_groups:
            g["lr_scale"] = float(lr_scale)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """Execute 1 gauge-quotiented Riemannian step across Grassmannian, Coupled-SVD, Kraus, and Flag manifolds."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        g0 = self.param_groups[0]
        lr_scale = float(g0.get("lr_scale", 1.0))
        stiefel_lr = float(g0.get("stiefel_lr", g0["lr"])) * lr_scale
        unitary_lr = float(g0.get("unitary_lr", g0["lr"])) * lr_scale
        beta = float(g0.get("beta", DEFAULT_HORIZ_MOMENTUM_BETA))
        ns_steps = int(g0.get("ns_steps", DEFAULT_NS5_STEPS))
        tik_lam = float(g0.get("tikhonov_lambda", DEFAULT_TIKHONOV_GRAM_LAMBDA))

        self._step_grassmann_params(stiefel_lr, beta, ns_steps, tik_lam)
        self._step_coupled_svd_pairs(stiefel_lr, beta, ns_steps, tik_lam)
        self._step_complex_kraus_and_mub(unitary_lr, beta, ns_steps, tik_lam)
        return loss


# ==============================================================================
# 4. Spike-Guarded Shadow EMA Thermostat (Eliminating the Ratchet Trap)
# ==============================================================================
class SpikeGuardedRiemannianShadowEMAThermostat(RiemannianShadowEMAThermostat):
    r"""Polyak–Ruppert Riemannian Shadow EMA Thermostat that only hard-resets active weights on genuine hot-loss spikes."""

    def __init__(
        self,
        hot_model: Any,
        ema_decay: float = DEFAULT_THERMOSTAT_EMA_DECAY,
        spike_rollback_tol_nats: float = DEFAULT_SPIKE_ROLLBACK_TOL_NATS,
        catastrophic_gap_tol_nats: float = DEFAULT_CATASTROPHIC_GAP_TOL_NATS,
        gentle_cool_gap_nats: float = DEFAULT_GENTLE_COOL_GAP_NATS,
        gain_kappa: float = DEFAULT_THERMOSTAT_GAIN_KAPPA,
        eval_fn: Any = None,
        retract_cadence: int = DEFAULT_THERMOSTAT_RETRACT_CADENCE,
    ) -> None:
        super().__init__(
            hot_model,
            ema_decay=ema_decay,
            gap_threshold=catastrophic_gap_tol_nats,
            gain_kappa=gain_kappa,
            min_scale_step=DEFAULT_THERMOSTAT_MIN_SCALE_STEP,
            eval_fn=eval_fn,
            retract_cadence=retract_cadence,
        )
        self.spike_rollback_tol_nats = float(spike_rollback_tol_nats)
        self.catastrophic_gap_tol_nats = float(catastrophic_gap_tol_nats)
        self.gentle_cool_gap_nats = float(gentle_cool_gap_nats)
        self.prev_hot_val_ce: float | None = None

    @torch.no_grad()
    def check_and_thermostat(
        self,
        hot_model: Any,
        stream: Any,
        current_lr_scale: float,
        step: int,
        apply_feedback: bool = True,
    ) -> dict[str, Any]:
        """Evaluate `hot_model` vs. manifold-retracted `ema_model`; gently cool LR on thermal gap, but only rollback weights on true spikes (`<= 8` args)."""
        self.retract_shadow_ema_manifold_(hot_model=hot_model)
        hot_ev = self.eval_fn(hot_model, stream)
        ema_ev = self.eval_fn(self.ema_model, stream)

        hot_ce, ema_ce = float(hot_ev["val_ce"]), float(ema_ev["val_ce"])
        thermal_gap = hot_ce - ema_ce
        hot_spike = (hot_ce - self.prev_hot_val_ce) if self.prev_hot_val_ce is not None else 0.0
        is_genuine_spike = bool(
            apply_feedback
            and (
                (self.prev_hot_val_ce is not None and hot_spike >= self.spike_rollback_tol_nats and thermal_gap > 0.0)
                or (thermal_gap >= self.catastrophic_gap_tol_nats)
            )
        )
        lr_before = float(current_lr_scale)

        if is_genuine_spike:
            gamma = max(self.min_scale_step, 1.0 / math.sqrt(1.0 + self.gain_kappa * max(0.0, thermal_gap)))
            lr_after = max(0.60, lr_before * gamma)
            hot_model.load_state_dict(self.ema_model.state_dict())
            self.prev_hot_val_ce = ema_ce
            self.trigger_events.append({
                "step": int(step), "hot_val_ce_before_bank": round(hot_ce, 4), "ema_val_ce_banked": round(ema_ce, 4),
                "thermal_gap_ce": round(thermal_gap, 4), "hot_spike_ce": round(hot_spike, 4),
                "lr_scale_before": round(lr_before, 4), "lr_scale_after": round(lr_after, 4),
                "reduction_factor": round(gamma, 4),
            })
            active_ev = ema_ev
        else:
            self.prev_hot_val_ce = hot_ce
            if apply_feedback and thermal_gap >= self.gentle_cool_gap_nats and lr_before > 0.60:
                lr_after = max(0.60, lr_before * float(DEFAULT_GENTLE_LR_DECAY_FACTOR))
            elif apply_feedback and thermal_gap < 0.012 and lr_before < 1.00:
                lr_after = min(1.00, lr_before / float(DEFAULT_GENTLE_LR_DECAY_FACTOR))
            else:
                lr_after = lr_before
            active_ev = ema_ev if ema_ce <= hot_ce else hot_ev

        return {
            "triggered": bool(is_genuine_spike),
            "hot_val_ce": round(hot_ce, 4),
            "ema_val_ce": round(ema_ce, 4),
            "thermal_gap_ce": round(thermal_gap, 4),
            "hot_spike_ce": round(hot_spike, 4),
            "lr_scale_before": round(lr_before, 4),
            "lr_scale_after": round(lr_after, 4),
            "retract_ms": round(self.last_retract_ms, 3),
            "active_eval": active_ev,
            "hot_eval": hot_ev,
            "ema_eval": ema_ev,
        }


# ==============================================================================
# 5. Wave 66 Horizontal-Grassmannian & Gauge-Free Quantum GPT
# ==============================================================================
class HorizontalGrassmannGaugeFreeGPT(nn.Module):
    r"""Wave 66 Quantum Density-Matrix Language Model with Horizontal-Grassmannian Frame Bridges, Gauge-Quotiented Coupled SVD/Kraus Operators, and Gauge-Locked $\mathbb{CP}^{15}$ MUB Tomography."""

    def __init__(self, spec: Wave66GaugeFreeSpec) -> None:
        super().__init__()
        self.spec, self.variant_mode = spec, Wave66GaugeVariantMode(spec.variant_mode)
        self.embed_dim, self.num_heads = int(DEFAULT_TOMO_EMBED_DIM), int(DEFAULT_TOMO_NUM_HEADS)
        self.d_h, self.head_real_dim = int(DEFAULT_TOMO_HEAD_COMPLEX_DIM), 2 * int(DEFAULT_TOMO_HEAD_COMPLEX_DIM)
        self.num_layers, self.kraus_rank = int(spec.num_layers), max(1, int(spec.kraus_rank))
        self.d_vac = int(DEFAULT_TOMO_VACUUM_DIM)
        depth_scale = math.sqrt(float(DEFAULT_TOMO_NUM_LAYERS) / float(max(1, self.num_layers)))

        self.boundary = SharedStiefelSubspaceFrameBoundary(self.embed_dim, spec.subspace_dim)

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
                mlp_hidden=DEFAULT_TOMO_MLP_HIDDEN, kraus_rank=self.kraus_rank, depth_scale=depth_scale,
            )
            for _ in range(self.num_layers)
        ])
        self.exit_bridge = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)
        self.tomo_proj = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)

        u1_re, u1_im, u2_re, u2_im = build_mub_two_basis_unitary_probes(self.num_heads, self.d_h, seed=spec.seed)
        self.probe_u1_re, self.probe_u1_im = nn.Parameter(u1_re), nn.Parameter(u1_im)
        if spec.enable_gauge_locked_mub_kraus:
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
        r"""Return current deterministic Parseval inverse temperature $\beta_0(N_{\text{tok}})$."""
        val = compute_analytical_parseval_beta(int(self.tokens_seen_buf.item()), beta_target=float(DEFAULT_ANALYTICAL_BETA_TARGET))
        return torch.tensor(val, device=self.tokens_seen_buf.device, dtype=torch.float32)

    def grassmann_parameters(self) -> list[nn.Parameter]:
        r"""Return tall subspace frame bridges $U_{\text{in}}, U_{\text{out}} \in \operatorname{Gr}(128, 512)$."""
        return self.boundary.real_stiefel_parameters()

    def coupled_svd_parameter_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        r"""Return all coupled `(u_stiefel, v_stiefel)` pairs across `BiorthogonalSVDStiefelProjection` modules."""
        pairs: list[tuple[nn.Parameter, nn.Parameter]] = [
            (self.exit_bridge.u_stiefel, self.exit_bridge.v_stiefel),
            (self.tomo_proj.u_stiefel, self.tomo_proj.v_stiefel),
        ]
        for q_blk in self.quantum_blocks:
            for proj in (q_blk.q_proj, q_blk.k_proj, q_blk.v_proj, q_blk.obs_proj, q_blk.mz_mlp.up_proj, q_blk.mz_mlp.down_proj):
                pairs.append((proj.u_stiefel, proj.v_stiefel))
        return pairs

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Return flat list of all real Stiefel/Grassmann matrices for `RiemannianShadowEMAThermostat` retraction."""
        params: list[nn.Parameter] = list(self.grassmann_parameters())
        for u_p, v_p in self.coupled_svd_parameter_pairs():
            params.extend([u_p, v_p])
        return params

    def kraus_unitary_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        r"""Return paired `(k_evol_re, k_evol_im)` $\operatorname{St}_{\mathbb{C}}(R d_h, d_h)$ Kraus operators."""
        pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
        for q_blk in self.quantum_blocks:
            pairs.extend(q_blk.complex_unitary_pairs())
        return pairs

    def mub_flag_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        r"""Return MUB probe parameter pairs on $U(16) / U(1)^{16}$."""
        pairs: list[tuple[nn.Parameter, nn.Parameter]] = [(self.probe_u1_re, self.probe_u1_im)]
        if self.probe_u2_re is not None and self.probe_u2_im is not None:
            pairs.append((self.probe_u2_re, self.probe_u2_im))
        return pairs

    def complex_unitary_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        """Return all complex Stiefel/Unitary parameter pairs for `RiemannianShadowEMAThermostat` retraction."""
        return self.mub_flag_pairs() + self.kraus_unitary_pairs()

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
        """Measure Kraus Stiefel, $U(16)$ MUB unbiasedness, and Grassmannian Stiefel orthogonality errors."""
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
        r"""Encode `idx` `(B, T)` through the $L$-layer Kraus highway and gauge-locked $\mathbb{CP}^{15}$ MUB readout (`<= 8` args)."""
        bsz, seq_len = idx.shape
        work_dtype = torch.bfloat16 if (idx.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
        idx_l = idx.to(torch.long)
        idx_p1 = torch.cat([idx_l[:, :1], idx_l[:, :-1]], dim=1) if seq_len > 1 else idx_l
        idx_p2 = torch.cat([idx_l[:, :2], idx_l[:, :-2]], dim=1) if seq_len > 2 else idx_p1
        h_tok = (idx_l * 1000003 + idx_p1 * 65537 + idx_p2 * 257) & (DEFAULT_TOK_NGRAM_BUCKETS - 1)

        raw_emb = self.boundary.encode_tokens(idx)
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


def build_wave66_gauge_free_optimizers(
    model: HorizontalGrassmannGaugeFreeGPT,
) -> tuple[HorizontalGrassmannGaugeFreeMuon, BatchSubspaceTikhonovGramMuon, RiemannianGeodesicScalarAndProbeOptimizer]:
    """Build 100% `AdamW`-free Horizontal-Grassmannian/Gauge-Free Stiefel+Kraus+MUB, Vocab Sphere, and Geodesic optimizers (`<= 8` args)."""
    q_trunk_opt = HorizontalGrassmannGaugeFreeMuon(
        model.grassmann_parameters(),
        model.coupled_svd_parameter_pairs(),
        model.kraus_unitary_pairs(),
        model.mub_flag_pairs(),
        stiefel_lr=DEFAULT_SHALLOW_STIEFEL_LR * 1.35,
        unitary_lr=0.028,
        beta=DEFAULT_HORIZ_MOMENTUM_BETA,
    )
    q_trunk_opt.enable_horizontal_grassmann = bool(model.spec.enable_horizontal_grassmann)
    q_trunk_opt.enable_gauge_locked_mub_kraus = bool(model.spec.enable_gauge_locked_mub_kraus)

    sub_groups = model.boundary.sphere_and_ball_groups(
        base_wte_lr=DEFAULT_SHALLOW_WTE_LR * 1.35, base_readout_lr=DEFAULT_SHALLOW_READOUT_LR * 1.35
    )
    sub_groups.append({"params": [model.tok_ngram_in.weight, model.tok_ngram_out.weight], "lr": DEFAULT_SHALLOW_NGRAM_LR * 1.5, "zero_init_ball": True})
    vocab_opt = BatchSubspaceTikhonovGramMuon(
        sub_groups, lr=DEFAULT_SHALLOW_WTE_LR * 1.35, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA, grad_norm_gate_tau=DEFAULT_MUON_GRAD_NORM_GATE_TAU
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


def step_wave66_optimizers(
    model: HorizontalGrassmannGaugeFreeGPT,
    optimizers: tuple[HorizontalGrassmannGaugeFreeMuon, BatchSubspaceTikhonovGramMuon, RiemannianGeodesicScalarAndProbeOptimizer],
    tokens_in_step: int,
    lr_mult: float = 1.0,
    mom_beta: float = DEFAULT_HORIZ_MOMENTUM_BETA,
) -> None:
    """Execute one coordinated Horizontal-Grassmannian + Gauge-Free Kraus/MUB + Vocab Sphere + Geodesic step (`<= 8` args)."""
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
def evaluate_wave66_model(
    model: HorizontalGrassmannGaugeFreeGPT, stream_or_batches: Any
) -> dict[str, float]:
    """Evaluate `HorizontalGrassmannGaugeFreeGPT` on Shard-0 validation batches (`<= 8` args)."""
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


def run_wave66_gauge_free_comparison_suite(
    stream: LongSeqFineWebStream,
    steps: int = 60,
    eval_interval: int = 20,
    wave_id: str = "W66",
    num_layers: int = DEFAULT_TOMO_NUM_LAYERS,
    grad_accum_steps: int = 1,
    only_variant: str = "",
) -> dict[str, Any]:
    """Run the Wave 66 Horizontal-Grassmannian & Gauge-Free Quantum Density-Matrix comparison suite (`<= 8` args)."""
    device = stream.device
    k_accum = max(1, int(grad_accum_steps))
    tokens_per_step = k_accum * stream.batch_size * stream.seq_len
    quench_steps = max(1, steps // 5)
    plateau_steps = max(1, steps - quench_steps)

    modes: list[tuple[Wave66GaugeVariantMode, str]] = [
        (Wave66GaugeVariantMode.W66_G0_STIEFEL_AMBIENT_BASELINE, "W65_V3 Baseline: Ambient Stiefel Momentum + Separate MUB U1/U2 + Standard Ratchet Thermostat"),
        (Wave66GaugeVariantMode.W66_G1_HORIZONTAL_GRASSMANN_VOCAB_AND_BRIDGES, "Horizontal Grassmannian Gr(128, 512) Subspace Bridges + Joint O(r) Coupled SVD-Stiefel Gauge Quotient"),
        (Wave66GaugeVariantMode.W66_G2_GAUGE_LOCKED_CP15_MUB_AND_KRAUS, "W66_G1 + Gauge-Locked CP15 Flag MUB Tomography (U2 = F16 U1) + u(R) Kraus Branch Quotient"),
        (Wave66GaugeVariantMode.W66_G3_FULL_GAUGE_FREE_SPIKE_ONLY_THERMOSTAT, "Full Gauge-Free Architecture (W66_G2) + Spike-Guarded Polyak-Ruppert Shadow EMA Thermostat"),
    ]
    filter_str = only_variant.strip().upper()
    if filter_str:
        modes = [m for m in modes if filter_str in m[0].value.upper()]

    results: list[dict[str, Any]] = []
    for mode, desc in modes:
        spec = build_wave66_variant_spec(mode, num_layers=int(num_layers), seed=1337)
        torch.manual_seed(spec.seed)
        model = HorizontalGrassmannGaugeFreeGPT(spec).to(device=device)
        init_ms = model.initialize_model_()
        total_params = sum(p.numel() for p in model.parameters())

        sfx = f"_l{int(num_layers)}_ga{k_accum}" if (int(num_layers) != DEFAULT_TOMO_NUM_LAYERS or k_accum > 1) else ""
        run_key = f"plan086_{mode.value.lower()}_{steps}steps{sfx}"
        intent = ExperimentIntentSpec(
            run_key=run_key, wave_id=wave_id, plan_id="EXPERIMENT_PLAN_086", task_domain="fineweb_speedrun",
            hypothesis_intent=desc,
            architecture_summary=f"HorizontalGrassmannGaugeFreeGPT(mode={mode.value}, L={int(num_layers)}, params={total_params:,})",
            hardware_budget=str(device), batch_size=stream.batch_size * k_accum, seq_len=stream.seq_len,
            total_steps=steps, target_tokens_M=round((steps * tokens_per_step) / 1e6, 3),
        )
        exp_logger = StandardizedExperimentLogger(intent)
        optimizers = build_wave66_gauge_free_optimizers(model)
        q_trunk_opt = optimizers[0]
        thermostat: RiemannianShadowEMAThermostat = (
            SpikeGuardedRiemannianShadowEMAThermostat(model, eval_fn=evaluate_wave66_model)
            if spec.enable_spike_guarded_thermostat
            else RiemannianShadowEMAThermostat(model, eval_fn=evaluate_wave66_model)
        )

        step0_ev = evaluate_wave66_model(model, stream)
        step0_ev["step"], step0_ev["tokens_M"] = 0, 0.0
        exp_logger.log_eval(0, {**step0_ev, "top1_acc": round(100.0 * step0_ev["top1_acc"], 2), "top5_acc": round(100.0 * step0_ev["top5_acc"], 2)})
        print(
            f"[{mode.value} (init={init_ms:.1f}ms, L={int(num_layers)}, params={total_params:,}, "
            f"mub_err={step0_ev['mub_unbiased_err']:.2e})] Step    0 | Val CE: {step0_ev['val_ce']:.4f}",
            flush=True,
        )

        curve: list[dict[str, Any]] = [step0_ev]
        thermostat_lr_scale: float = 1.0
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

            if step <= 5:
                warm = float(step) / 5.0
                lr_mult, mom_beta = (0.25 + 0.75 * warm) * thermostat_lr_scale, 0.50 * warm
            elif step <= plateau_steps:
                lr_mult, mom_beta = 1.0 * thermostat_lr_scale, DEFAULT_HORIZ_MOMENTUM_BETA
            else:
                q_p = float(step - plateau_steps) / float(max(1, quench_steps))
                cos_q = 0.5 * (1.0 + math.cos(math.pi * min(1.0, q_p)))
                lr_mult, mom_beta = (0.08 + 0.92 * cos_q) * thermostat_lr_scale, 0.85 + 0.10 * (1.0 - cos_q)

            step_wave66_optimizers(model, optimizers, tokens_in_step=tokens_per_step, lr_mult=lr_mult, mom_beta=mom_beta)
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
                ev["thermostat_lr_scale"] = round(thermostat_lr_scale, 4)
                ev["retract_ms"] = th_out["retract_ms"]
                ev.update(gd)
                curve.append(ev)
                exp_logger.log_eval(step, {**ev, "top1_acc": round(100.0 * ev["top1_acc"], 2), "top5_acc": round(100.0 * ev["top5_acc"], 2)})
                print(
                    f"[{mode.value}] Step {step:4d} ({ev['tokens_M']:5.2f}M tok, lr={thermostat_lr_scale:.2f}x) | "
                    f"Hot CE: {th_out['hot_val_ce']:.4f} | EMA CE: {th_out['ema_val_ce']:.4f} "
                    f"(gap={th_out['thermal_gap_ce']:+.4f}, trig={th_out['triggered']}) | "
                    f"Top1: {100.0 * ev['top1_acc']:.2f}% | MUBErr: {ev['mub_unbiased_err']:.2e} | "
                    f"GrSpin: {gd.get('grassmann_step_spin_frac', 0.0):.2e} | SVDSpin: {gd.get('coupled_svd_step_spin_frac', 0.0):.2e} | "
                    f"CP15Phase: {gd.get('cp15_step_phase_frac', 0.0):.2e} | KrausGauge: {gd.get('kraus_step_gauge_frac', 0.0):.2e}",
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


