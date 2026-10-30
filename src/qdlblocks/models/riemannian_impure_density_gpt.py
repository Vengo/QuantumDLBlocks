"""Wave 47 (`EXPERIMENT_PLAN_062.md`): All-Layer `RiemannianStiefelMuon` Impure Density-Matrix GPT.

Synthesizes the quantum-geometric mechanisms validated in Wave 46 (`EXPERIMENT_PLAN_061.md`) and
the Gemini Notebook (*"Top-1 Accuracy vs Cross-Entropy Divergence"*, Pages 18-36) with the Wave 45
(`EXPERIMENT_PLAN_060.md`) leaderboard architecture into a full multi-layer language model:

1. **100% All-Layer `RiemannianStiefelMuon` Optimization (`Real Stiefel` + `Joint-Complex Unitary` $U(16)$)**:
   - Every 2D weight matrix in every layer (`w_q`, `w_k`, `w_v`, `w_obs_u`, `w_obs_vt`, `w_beam_a`,
     `w_beam_b`, `w_chan_u`, `w_chan_vt`) lives on a real Stiefel manifold $\\text{St}(D, d)$ ($W^T W = I_d$)
     and is updated via `riemannian_stiefel_muon_step` ($1 / \\sqrt{1 + \\eta^2}$ geodesic retraction).
   - Every 3D per-head complex unitary channel `(head_unitary_re, head_unitary_im)` $\\in U(16)^H$ is
     updated on the complex unitary manifold $U(16)$ via `riemannian_complex_unitary_muon_step`
     (skew-Hermitian tangent projection $\\mathfrak{u}(16)$ + complex Newton-Schulz + $1/\\sqrt{1+\\eta^2}$ retraction).
   - Readouts are parameterized in **Stiefel-SVD form** $M = U \\operatorname{diag}(\\sigma) V^T$
     ($U \\in \\text{St}(D, d), \\sigma \\in \\mathbb{R}^d, V^T \\in O(d)$), allowing exact zero-error
     representation of Golub-Pereyra ridge solutions while keeping all 2D matrices strictly on the
     Stiefel manifold (`||W^T W - I||_F / sqrt(d) < 1e-6`).
2. **Fast Rank-$R$ ($R=4$) Token Density Contraction + Precomputed Isometric $\\sqrt{2}$ Packing**:
   - Exploits the cancellation $p_{s,h,r} |\\hat{v}_{s,h}^{(r)}\\rangle\\langle \\hat{v}_{s,h}^{(r)}| =
     |v_{s,h}^{(r)}\\rangle\\langle v_{s,h}^{(r)}| / \\sum_{r\'} \\|v_{s,h}^{(r\')}\\|_2^2$ to compute the
     trace-1 rank-$R$ token density matrix $\\rho_{s,h}^{\\text{tok}} = \\tilde{V}_{s,h}^\\dagger \\tilde{V}_{s,h}
     \\in \\mathcal{D}(\\mathbb{C}^{16})$ via $(16 \\times 4) \\times (4 \\times 16)$ batched matmuls without
     materializing a 6D `(B, H, T, R, 16, 16)` tensor.
3. **Cross-Layer Quantum State Highway ($x_0$ Residual Anchor + $V_0$ Multi-Branch Value Ket Highway)**:
   - Propagates Layer-0 multi-branch value states $V_0 = x_{\\text{norm}}^{(0)} W_V^{(0)}$ to all deeper
     quantum layers ($V_{\\text{raw}}^{(l)} = \\lambda_{v,0}^{(l)} (x_{\\text{norm}}^{(l)} W_V^{(l)}) + \\lambda_{v,1}^{(l)} V_0$)
     prior to density-matrix synthesis and Bures-Wasserstein McWeeny purification $\\rho \\leftarrow \\rho^2 / \\text{Tr}(\\rho^2)$.
4. **Stationary Input Wavefunction Boundary Gauge (`InputBoundaryMode`, Gemini Notebook Pages 30-34)**:
   - Supports unit-sphere anchored token wavefunctions paired with a Stiefel input rotation $W_{\\text{in}} \\in O(d)$
     (`InputBoundaryMode.ANCHORED_SPHERE`), frozen classical-teacher boundary embeddings/head
     (`InputBoundaryMode.TEACHER_FROZEN_BOUNDARY`), or unconstrained Euclidean embeddings
     (`InputBoundaryMode.EUCLIDEAN_UNANCHORED`) for controlled ablation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.impure_density_mirror_block import (
    DEFAULT_NS5_STEPS,
    DEFAULT_RIEMANNIAN_MOMENTUM_BETA,
    DEFAULT_ROPE_BASE_THETA,
    INV_SQRT2_ISOMETRIC_SCALE,
    SQRT2_ISOMETRIC_OFFDIAG_SCALE,
    TRACE_NORM_EPS,
    bures_wasserstein_mcweeny_step,
    complex_l2_normalize,
    initialize_stiefel_parameter,
    pack_hermitian_isometric,
    pack_rank1_ket_outer_isometric,
    riemannian_stiefel_muon_step,
    unpack_hermitian_isometric,
)
from ..layers.isometric import apply_partial_key_offset
from .compiled_leaderboard_classical_gpt import _CompiledChunkedSoftcapLinearCEFn

# ---------------------------------------------------------------------------
# Top-of-File Named Constants (Rule 7 — Synchronized with EXPERIMENT_PLAN_062.md)
# ---------------------------------------------------------------------------
DEFAULT_RIEMANNIAN_VOCAB_SIZE: int = 50304
DEFAULT_RIEMANNIAN_VALID_VOCAB: int = 50257
DEFAULT_RIEMANNIAN_EMBED_DIM: int = 768
DEFAULT_RIEMANNIAN_NUM_LAYERS: int = 12
DEFAULT_RIEMANNIAN_NUM_HEADS: int = 6
DEFAULT_RIEMANNIAN_HEAD_COMPLEX_DIM: int = 16
DEFAULT_RIEMANNIAN_VALUE_BRANCHES: int = 4
DEFAULT_RIEMANNIAN_BORN_POWER_P: int = 2
DEFAULT_RIEMANNIAN_MCWEENY_STEPS: int = 1
DEFAULT_STIEFEL_GEODESIC_LR: float = 0.025
DEFAULT_UNITARY_GEODESIC_LR: float = 0.020
DEFAULT_READOUT_STIEFEL_LR_MULT: float = 1.0
DEFAULT_V0_HIGHWAY_INIT_LAMBDA: float = 0.15
DEFAULT_LOGIT_SOFTCAP: float = 15.0
DEFAULT_CE_CHUNK_TOKENS: int = 8192
DEFAULT_INIT_RESIDUAL_SIGMA: float = 0.08


class InputBoundaryMode(str, Enum):
    """Strategy enum controlling the input token wavefunction boundary gauge (Rule 3)."""

    ANCHORED_SPHERE = "anchored_sphere"
    EUCLIDEAN_UNANCHORED = "euclidean_unanchored"
    TEACHER_FROZEN_BOUNDARY = "teacher_frozen_boundary"


class DensityStateMode(str, Enum):
    """Strategy enum selecting multi-branch impure McWeeny density states vs. Rank-1 pure kets (Rule 3)."""

    IMPURE_MCWEENY_R4 = "impure_mcweeny_r4"
    PURE_KET_R1 = "pure_ket_r1"


@dataclass(frozen=True)
class RiemannianImpureDensityConfig:
    """Configuration for `RiemannianImpureDensityGPT` (`<= 8` fields, Rule 2)."""

    vocab_size: int = DEFAULT_RIEMANNIAN_VOCAB_SIZE
    embed_dim: int = DEFAULT_RIEMANNIAN_EMBED_DIM
    num_layers: int = DEFAULT_RIEMANNIAN_NUM_LAYERS
    num_heads: int = DEFAULT_RIEMANNIAN_NUM_HEADS
    head_complex_dim: int = DEFAULT_RIEMANNIAN_HEAD_COMPLEX_DIM
    boundary_mode: InputBoundaryMode = InputBoundaryMode.ANCHORED_SPHERE
    density_mode: DensityStateMode = DensityStateMode.IMPURE_MCWEENY_R4
    use_v0_highway: bool = True


def _complex_newton_schulz5(
    z_re: torch.Tensor,
    z_im: torch.Tensor,
    steps: int = DEFAULT_NS5_STEPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute unit-singular-value complex polar factor of `(z_re, z_im)` `(..., d, d)` via cubic Newton-Schulz."""
    x_re = z_re.float()
    x_im = z_im.float()
    fro_sq = (x_re.square().sum(dim=(-2, -1), keepdim=True) + x_im.square().sum(dim=(-2, -1), keepdim=True)).clamp_min(
        TRACE_NORM_EPS
    )
    inv_fro = torch.rsqrt(fro_sq)
    x_re = x_re * inv_fro
    x_im = x_im * inv_fro
    eye = torch.eye(x_re.size(-1), device=x_re.device, dtype=x_re.dtype)

    for _ in range(max(6, steps + 1)):
        gram_re = torch.matmul(x_re.transpose(-2, -1), x_re) + torch.matmul(x_im.transpose(-2, -1), x_im)
        gram_im = torch.matmul(x_re.transpose(-2, -1), x_im) - torch.matmul(x_im.transpose(-2, -1), x_re)
        three_minus_re = 3.0 * eye - gram_re
        three_minus_im = -gram_im
        next_re = 0.5 * (torch.matmul(x_re, three_minus_re) - torch.matmul(x_im, three_minus_im))
        next_im = 0.5 * (torch.matmul(x_re, three_minus_im) + torch.matmul(x_im, three_minus_re))
        x_re, x_im = next_re, next_im
    return x_re, x_im


def riemannian_complex_unitary_muon_step(
    u_re: torch.Tensor,
    u_im: torch.Tensor,
    grad_re: torch.Tensor,
    grad_im: torch.Tensor,
    mom_re: torch.Tensor,
    mom_im: torch.Tensor,
    lr: float = DEFAULT_UNITARY_GEODESIC_LR,
    beta: float = DEFAULT_RIEMANNIAN_MOMENTUM_BETA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute 1 exact Joint-Complex Unitary `RiemannianStiefelMuon` step on $U = U_{\\text{re}} + i U_{\\text{im}} \\in U(d_h)$.

    1. Accumulates complex Nesterov momentum $\\tilde{G} = G + \\beta M$.
    2. Projects $\\tilde{G}$ onto the skew-Hermitian Lie algebra $\\mathfrak{u}(d_h)$ at $T_U U(d_h)$:
       $\\Omega = \\frac{1}{2}(U^\\dagger \\tilde{G} - \\tilde{G}^\\dagger U)$, $T = U \\Omega$.
    3. Orthogonalizes $T$ via complex Newton-Schulz so $\\hat{T}^\\dagger \\hat{T} = I$ and $U^\\dagger \\hat{T} + \\hat{T}^\\dagger U = 0$.
    4. Retracts along the exact scalar geodesic $U_{\\text{next}} = (U - \\eta \\hat{T}) / \\sqrt{1 + \\eta^2}$ + 1 Schulz polish step.
    """
    if not (torch.isfinite(grad_re).all() and torch.isfinite(grad_im).all()):
        return u_re, u_im

    mom_re.mul_(beta).add_(grad_re.float())
    mom_im.mul_(beta).add_(grad_im.float())
    g_re = grad_re.float() + float(beta) * mom_re.float()
    g_im = grad_im.float() + float(beta) * mom_im.float()

    g_norm = (g_re.square().sum(dim=(-2, -1)) + g_im.square().sum(dim=(-2, -1))).sqrt().max()
    if float(g_norm.item()) < TRACE_NORM_EPS:
        return u_re, u_im

    ur = u_re.float()
    ui = u_im.float()
    ut_re = ur.transpose(-2, -1)
    ut_im = -ui.transpose(-2, -1)

    p_re = torch.matmul(ut_re, g_re) - torch.matmul(ut_im, g_im)
    p_im = torch.matmul(ut_re, g_im) + torch.matmul(ut_im, g_re)
    omega_re = 0.5 * (p_re - p_re.transpose(-2, -1))
    omega_im = 0.5 * (p_im + p_im.transpose(-2, -1))

    t_re = torch.matmul(ur, omega_re) - torch.matmul(ui, omega_im)
    t_im = torch.matmul(ur, omega_im) + torch.matmul(ui, omega_re)
    t_norm = (t_re.square().sum(dim=(-2, -1)) + t_im.square().sum(dim=(-2, -1))).sqrt().max()
    if float(t_norm.item()) < TRACE_NORM_EPS:
        return u_re, u_im

    z_re, z_im = _complex_newton_schulz5(t_re, t_im)
    pz_re = torch.matmul(ut_re, z_re) - torch.matmul(ut_im, z_im)
    pz_im = torch.matmul(ut_re, z_im) + torch.matmul(ut_im, z_re)
    om_re = 0.5 * (pz_re - pz_re.transpose(-2, -1))
    om_im = 0.5 * (pz_im + pz_im.transpose(-2, -1))
    tan_re = torch.matmul(ur, om_re) - torch.matmul(ui, om_im)
    tan_im = torch.matmul(ur, om_im) + torch.matmul(ui, om_re)

    scale = 1.0 / math.sqrt(1.0 + float(lr) * float(lr))
    nx_re = (ur - float(lr) * tan_re) * scale
    nx_im = (ui - float(lr) * tan_im) * scale

    eye = torch.eye(ur.size(-1), device=ur.device, dtype=ur.dtype)
    gram_re = torch.matmul(nx_re.transpose(-2, -1), nx_re) + torch.matmul(nx_im.transpose(-2, -1), nx_im)
    gram_im = torch.matmul(nx_re.transpose(-2, -1), nx_im) - torch.matmul(nx_im.transpose(-2, -1), nx_re)
    tm_re = 3.0 * eye - gram_re
    tm_im = -gram_im
    pol_re = 0.5 * (torch.matmul(nx_re, tm_re) - torch.matmul(nx_im, tm_im))
    pol_im = 0.5 * (torch.matmul(nx_re, tm_im) + torch.matmul(nx_im, tm_re))

    u_re.copy_(pol_re.to(u_re.dtype))
    u_im.copy_(pol_im.to(u_im.dtype))
    return u_re, u_im


def polar_factorize_stiefel_svd(
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Factor unconstrained matrix $M \\in \\mathbb{R}^{D \\times d}$ ($D \\ge d$) into $(U, \\sigma, V^T)$.

    Guarantees:
    - $U \\in \\text{St}(D, d)$ with $U^T U = I_d$,
    - $\\sigma \\in \\mathbb{R}^d$ singular value spectrum,
    - $V^T \\in O(d)$ with $V^T V = I_d$,
    - $U \\operatorname{diag}(\\sigma) V^T = M$ to machine precision (`< 3e-7` Stiefel error via `float64` SVD).
    """
    u_mat, sigma, vt_mat = torch.linalg.svd(matrix.double(), full_matrices=False)
    return u_mat.float(), sigma.float(), vt_mat.float()


def fast_rank_r_token_density_pack(
    v_re: torch.Tensor,
    v_im: torch.Tensor,
    re_pack_mask: torch.Tensor,
    im_pack_mask: torch.Tensor,
    eps: float = TRACE_NORM_EPS,
) -> torch.Tensor:
    """Compute trace-1 rank-$R$ token density matrix $\\rho_{s,h}^{\\text{tok}} = \\sum_{r=1}^R p_r |\\hat{v}^{(r)}\\rangle\\langle \\hat{v}^{(r)}|$ and pack into $\\mathbb{R}^{d_h^2}$.

    Uses the exact algebraic cancellation $p_{s,h,r} |\\hat{v}_{s,h}^{(r)}\\rangle\\langle \\hat{v}_{s,h}^{(r)}|
    = |v_{s,h}^{(r)}\\rangle\\langle v_{s,h}^{(r)}| / \\sum_{r\'} \\|v_{s,h}^{(r\')}\\|_2^2$ to contract over
    $R$ via $(d_h \\times R) \\times (R \\times d_h)$ matmuls without materializing a 6D `(B, H, T, R, d_h, d_h)` tensor.
    """
    orig_dtype = v_re.dtype
    total_energy = (
        v_re.float().square().sum(dim=(-2, -1), keepdim=True) + v_im.float().square().sum(dim=(-2, -1), keepdim=True)
    ).clamp_min(eps)
    inv_sqrt_e = torch.rsqrt(total_energy).to(orig_dtype)
    v_re_s = v_re * inv_sqrt_e
    v_im_s = v_im * inv_sqrt_e

    vt_re = v_re_s.transpose(-2, -1)
    vt_im = v_im_s.transpose(-2, -1)
    rho_re = torch.matmul(vt_re, v_re_s) + torch.matmul(vt_im, v_im_s)
    rho_im = torch.matmul(vt_im, v_re_s) - torch.matmul(vt_re, v_im_s)

    packed_mat = rho_re * re_pack_mask.to(orig_dtype) + rho_im * im_pack_mask.to(orig_dtype)
    return packed_mat.flatten(start_dim=-2, end_dim=-1)


class RiemannianImpureDensityBlock(nn.Module):
    """Quantum layer where 100% of 2D/3D operator matrices live on Stiefel/Unitary manifolds (`Wave 47`)."""

    def __init__(
        self,
        cfg: RiemannianImpureDensityConfig,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.embed_dim = cfg.embed_dim
        self.num_heads = cfg.num_heads
        self.d_h = cfg.head_complex_dim
        self.packed_head_dim = self.d_h * self.d_h
        self.total_packed_dim = self.num_heads * self.packed_head_dim
        self.ket_real_dim = self.num_heads * (2 * self.d_h)
        self.num_value_branches = (
            max(1, self.embed_dim // self.ket_real_dim)
            if cfg.density_mode == DensityStateMode.IMPURE_MCWEENY_R4
            else 1
        )
        self.v_real_dim = self.num_heads * self.num_value_branches * (2 * self.d_h)
        self.channel_hidden_dim = self.embed_dim * 2
        self.is_short_window_layer = (self.layer_idx % 4 != 3) and (self.layer_idx < cfg.num_layers - 1)
        self.is_long_window_layer: bool = not self.is_short_window_layer

        self.block_lambdas = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float32))
        v0_init = DEFAULT_V0_HIGHWAY_INIT_LAMBDA if (cfg.use_v0_highway and self.layer_idx > 0) else 0.0
        self.v_lambdas = nn.Parameter(torch.tensor([1.0, v0_init], dtype=torch.float32))

        self.w_q = nn.Parameter(torch.empty(self.embed_dim, self.ket_real_dim))
        self.w_k = nn.Parameter(torch.empty(self.embed_dim, self.ket_real_dim))
        self.w_v = nn.Parameter(torch.empty(self.embed_dim, self.v_real_dim))
        initialize_stiefel_parameter(self.w_q)
        initialize_stiefel_parameter(self.w_k)
        initialize_stiefel_parameter(self.w_v)

        phase_angles = torch.linspace(-0.25, 0.25, self.d_h, dtype=torch.float32)
        diag_re = torch.diag(torch.cos(phase_angles)).unsqueeze(0).expand(self.num_heads, -1, -1).clone()
        diag_im = torch.diag(torch.sin(phase_angles)).unsqueeze(0).expand(self.num_heads, -1, -1).clone()
        self.head_unitary_re = nn.Parameter(diag_re)
        self.head_unitary_im = nn.Parameter(diag_im)

        self.w_obs_u = nn.Parameter(torch.empty(self.total_packed_dim, self.embed_dim))
        self.w_obs_vt = nn.Parameter(torch.empty(self.embed_dim, self.embed_dim))
        initialize_stiefel_parameter(self.w_obs_u)
        initialize_stiefel_parameter(self.w_obs_vt)
        self.w_obs_u._is_svd_readout_stiefel = True  # type: ignore[attr-defined]
        self.w_obs_vt._is_svd_readout_stiefel = True  # type: ignore[attr-defined]
        init_sig = 0.25 / math.sqrt(float(max(1, cfg.num_layers)))
        self.obs_sigma = nn.Parameter(torch.full((self.embed_dim,), init_sig))

        self.w_beam_a = nn.Parameter(torch.empty(self.embed_dim, self.channel_hidden_dim))
        self.w_beam_b = nn.Parameter(torch.empty(self.embed_dim, self.channel_hidden_dim))
        self.w_chan_u = nn.Parameter(torch.empty(self.channel_hidden_dim, self.embed_dim))
        self.w_chan_vt = nn.Parameter(torch.empty(self.embed_dim, self.embed_dim))
        initialize_stiefel_parameter(self.w_beam_a)
        initialize_stiefel_parameter(self.w_beam_b)
        initialize_stiefel_parameter(self.w_chan_u)
        initialize_stiefel_parameter(self.w_chan_vt)
        self.w_chan_u._is_svd_readout_stiefel = True  # type: ignore[attr-defined]
        self.w_chan_vt._is_svd_readout_stiefel = True  # type: ignore[attr-defined]
        self.chan_sigma = nn.Parameter(torch.full((self.embed_dim,), init_sig))

        inv_freq = 1.0 / (
            DEFAULT_ROPE_BASE_THETA ** (torch.arange(0, self.d_h, dtype=torch.float32) / float(self.d_h))
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

        re_mask = torch.diag(torch.ones(self.d_h)) + torch.triu(
            torch.full((self.d_h, self.d_h), SQRT2_ISOMETRIC_OFFDIAG_SCALE), diagonal=1
        )
        im_mask = torch.tril(torch.full((self.d_h, self.d_h), -SQRT2_ISOMETRIC_OFFDIAG_SCALE), diagonal=-1)
        self.register_buffer("re_pack_mask", re_mask, persistent=False)
        self.register_buffer("im_pack_mask", im_mask, persistent=False)
        self.last_telemetry: dict[str, float] = {"mean_purity": 1.0, "mean_eff_rank": 1.0}

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Return all 2D real Stiefel matrices in this layer (`W^T W = I` or `W W^T = I`)."""
        return [
            self.w_q,
            self.w_k,
            self.w_v,
            self.w_obs_u,
            self.w_obs_vt,
            self.w_beam_a,
            self.w_beam_b,
            self.w_chan_u,
            self.w_chan_vt,
        ]

    def complex_unitary_pair(self) -> tuple[nn.Parameter, nn.Parameter]:
        """Return the 3D complex head unitary pair `(head_unitary_re, head_unitary_im)` $\\in U(d_h)^H$."""
        return self.head_unitary_re, self.head_unitary_im

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return 1D singular-value spectra and scalar residual/value highway weights."""
        return [
            self.obs_sigma,
            self.chan_sigma,
            self.block_lambdas,
            self.v_lambdas,
        ]

    def _apply_complex_rope(
        self,
        z_re: torch.Tensor,
        z_im: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply complex rotary phase $e^{i t \\omega_k}$ to `(z_re, z_im)` `(B, H, T, d_h)`."""
        seq_len = z_re.size(-2)
        pos = torch.arange(seq_len, device=z_re.device, dtype=torch.float32)
        angles = torch.outer(pos, self.rope_inv_freq).unsqueeze(0).unsqueeze(0)
        cos_a = torch.cos(angles).to(z_re.dtype)
        sin_a = torch.sin(angles).to(z_re.dtype)
        return z_re * cos_a - z_im * sin_a, z_re * sin_a + z_im * cos_a

    def _evolve_head_unitary(
        self,
        rho_re: torch.Tensor,
        rho_im: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply exact per-head complex unitary conjugation $U_h \\rho_{t,h} U_h^\\dagger$ where $U_h \\in U(d_h)$."""
        u_re = self.head_unitary_re.to(rho_re.dtype).unsqueeze(0).unsqueeze(2)
        u_im = self.head_unitary_im.to(rho_re.dtype).unsqueeze(0).unsqueeze(2)
        ut_re = u_re.transpose(-2, -1)
        ut_im = -u_im.transpose(-2, -1)

        tmp_re = torch.matmul(u_re, rho_re) - torch.matmul(u_im, rho_im)
        tmp_im = torch.matmul(u_re, rho_im) + torch.matmul(u_im, rho_re)
        out_re = torch.matmul(tmp_re, ut_re) - torch.matmul(tmp_im, ut_im)
        out_im = torch.matmul(tmp_re, ut_im) + torch.matmul(tmp_im, ut_re)
        return 0.5 * (out_re + out_re.transpose(-2, -1)), 0.5 * (out_im - out_im.transpose(-2, -1))

    def compute_packed_density_features(
        self,
        x_base: torch.Tensor,
        v_0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute isometric $\\sqrt{2}$-packed contextual density features `(B, T, 1536)` and `v_0_next`."""
        bsz, seq_len, _ = x_base.shape
        work_dtype = x_base.dtype
        h_norm = F.rms_norm(x_base, (self.embed_dim,), eps=TRACE_NORM_EPS)

        q_raw = torch.matmul(h_norm, self.w_q.to(work_dtype)).view(bsz, seq_len, self.num_heads, 2, self.d_h)
        k_raw = torch.matmul(h_norm, self.w_k.to(work_dtype)).view(bsz, seq_len, self.num_heads, 2, self.d_h)
        v_proj = torch.matmul(h_norm, self.w_v.to(work_dtype))

        if v_0 is None:
            v_0_next = v_proj
            v_mixed = v_proj
        else:
            v_0_next = v_0
            if self.cfg.use_v0_highway and v_0.shape == v_proj.shape:
                lam_v0 = self.v_lambdas[0].to(work_dtype)
                lam_v1 = self.v_lambdas[1].to(work_dtype)
                v_mixed = lam_v0 * v_proj + lam_v1 * v_0
            else:
                v_mixed = v_proj

        q_re, q_im = q_raw[..., 0, :].transpose(1, 2), q_raw[..., 1, :].transpose(1, 2)
        k_re, k_im = k_raw[..., 0, :].transpose(1, 2), k_raw[..., 1, :].transpose(1, 2)
        q_re, q_im = complex_l2_normalize(q_re, q_im)
        k_re, k_im = complex_l2_normalize(k_re, k_im)
        q_re, q_im = self._apply_complex_rope(q_re, q_im)
        k_re, k_im = self._apply_complex_rope(k_re, k_im)

        if self.is_short_window_layer and seq_len > 1:
            k_re = apply_partial_key_offset(k_re.transpose(1, 2)).transpose(1, 2)
            k_im = apply_partial_key_offset(k_im.transpose(1, 2)).transpose(1, 2)

        ip_re = torch.matmul(q_re, k_re.transpose(-2, -1)) + torch.matmul(q_im, k_im.transpose(-2, -1))
        ip_im = torch.matmul(q_im, k_re.transpose(-2, -1)) - torch.matmul(q_re, k_im.transpose(-2, -1))
        born_mod_sq = (ip_re.float().square() + ip_im.float().square()).clamp_min(0.0)
        born_scores = born_mod_sq.square()

        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x_base.device, dtype=torch.bool))
        born_scores = born_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), 0.0)
        attn_weights = (born_scores / born_scores.sum(dim=-1, keepdim=True).clamp_min(TRACE_NORM_EPS)).to(work_dtype)

        if self.cfg.density_mode == DensityStateMode.IMPURE_MCWEENY_R4:
            v_raw = v_mixed.view(bsz, seq_len, self.num_heads, self.num_value_branches, 2, self.d_h)
            v_re = v_raw[..., 0, :].permute(0, 2, 1, 3, 4)
            v_im = v_raw[..., 1, :].permute(0, 2, 1, 3, 4)
            packed_tok_rho = fast_rank_r_token_density_pack(
                v_re, v_im, self.re_pack_mask, self.im_pack_mask, eps=TRACE_NORM_EPS
            )
            packed_rho0 = torch.matmul(attn_weights, packed_tok_rho)
            rho0_re, rho0_im = unpack_hermitian_isometric(packed_rho0.float(), self.d_h)
            rho1_re, rho1_im, purity, eff_rank = bures_wasserstein_mcweeny_step(
                rho0_re, rho0_im, num_steps=DEFAULT_RIEMANNIAN_MCWEENY_STEPS, eps=TRACE_NORM_EPS
            )
            with torch.no_grad():
                self.last_telemetry = {
                    "mean_purity": float(purity.mean().item()),
                    "mean_eff_rank": float(eff_rank.mean().item()),
                }
        else:
            v_raw = v_mixed.view(bsz, seq_len, self.num_heads, 2, self.d_h)
            v_re, v_im = complex_l2_normalize(v_raw[..., 0, :].transpose(1, 2), v_raw[..., 1, :].transpose(1, 2))
            psi_re = torch.matmul(attn_weights, v_re)
            psi_im = torch.matmul(attn_weights, v_im)
            psi_re, psi_im = complex_l2_normalize(psi_re.float(), psi_im.float())
            rho1_re = psi_re.unsqueeze(-1) * psi_re.unsqueeze(-2) + psi_im.unsqueeze(-1) * psi_im.unsqueeze(-2)
            rho1_im = psi_im.unsqueeze(-1) * psi_re.unsqueeze(-2) - psi_re.unsqueeze(-1) * psi_im.unsqueeze(-2)
            with torch.no_grad():
                self.last_telemetry = {"mean_purity": 1.0, "mean_eff_rank": 1.0}

        rho_out_re, rho_out_im = self._evolve_head_unitary(rho1_re, rho1_im)
        packed_out = (
            rho_out_re * self.re_pack_mask + rho_out_im * self.im_pack_mask
        ).flatten(start_dim=-2, end_dim=-1) * float(self.d_h)
        packed_concat = packed_out.to(work_dtype).transpose(1, 2).reshape(bsz, seq_len, self.total_packed_dim)
        return packed_concat, v_0_next

    def compute_channel_features(self, x_mid: torch.Tensor) -> torch.Tensor:
        """Compute Stiefel two-beam Mach-Zehnder quantum interference features `(B, T, channel_hidden_dim)`."""
        work_dtype = x_mid.dtype
        m_norm = F.rms_norm(x_mid, (self.embed_dim,), eps=TRACE_NORM_EPS)
        beam_a = torch.matmul(m_norm, self.w_beam_a.to(work_dtype))
        beam_b = torch.matmul(m_norm, self.w_beam_b.to(work_dtype))
        return F.relu(beam_a + beam_b).square() - 0.25 * F.relu(beam_a - beam_b).square()

    def _apply_stiefel_svd_readout(
        self,
        features: torch.Tensor,
        w_u: torch.Tensor,
        sigma: torch.Tensor,
        w_vt: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Stiefel-SVD readout $(features \\cdot U \\cdot \\operatorname{diag}(\\sigma)) \\cdot V^T$."""
        work_dtype = features.dtype
        h_u = torch.matmul(features, w_u.to(work_dtype))
        h_scaled = h_u * sigma.to(work_dtype)
        return torch.matmul(h_scaled, w_vt.to(work_dtype))

    @torch.no_grad()
    def fit_stiefel_observables_polar_ridge(
        self,
        x_in_list: list[torch.Tensor],
        x_out_list: list[torch.Tensor],
        x_0_list: list[torch.Tensor] | None = None,
        v_0_list: list[torch.Tensor] | None = None,
        ridge_lambda: float = 0.05,
    ) -> None:
        """Solve closed-form Golub-Pereyra ridge regression and factor into exact Stiefel-SVD $(U, \\sigma, V^T)$."""
        phi_chunks: list[torch.Tensor] = []
        delta_targets: list[torch.Tensor] = []
        x_bases: list[torch.Tensor] = []

        for idx, (x_in, x_out) in enumerate(zip(x_in_list, x_out_list)):
            x_f = x_in.float()
            if x_0_list is not None:
                x_base = self.block_lambdas[0].float() * x_f + self.block_lambdas[1].float() * x_0_list[idx].float()
            else:
                x_base = x_f
            v0_cur = v_0_list[idx].float() if v_0_list is not None else None
            phi, _ = self.compute_packed_density_features(x_base, v_0=v0_cur)
            phi_chunks.append(phi.float().reshape(-1, self.total_packed_dim))
            delta_targets.append((x_out.float() - x_base).reshape(-1, self.embed_dim))
            x_bases.append(x_base)

        phi_all = torch.cat(phi_chunks, dim=0)
        y_all = torch.cat(delta_targets, dim=0)
        n_samples = float(phi_all.size(0))

        gram_attn = torch.matmul(phi_all.T, phi_all) + (ridge_lambda * n_samples) * torch.eye(
            self.total_packed_dim, device=phi_all.device, dtype=phi_all.dtype
        )
        w_obs_full = torch.linalg.solve(gram_attn, torch.matmul(phi_all.T, y_all))
        u_obs, sig_obs, vt_obs = polar_factorize_stiefel_svd(w_obs_full)
        self.w_obs_u.copy_(u_obs.to(self.w_obs_u.dtype))
        self.obs_sigma.copy_(sig_obs.to(self.obs_sigma.dtype))
        self.w_obs_vt.copy_(vt_obs.to(self.w_obs_vt.dtype))

        chan_chunks: list[torch.Tensor] = []
        rem_targets: list[torch.Tensor] = []
        for idx, x_base in enumerate(x_bases):
            phi_b = phi_chunks[idx].view(x_base.size(0), x_base.size(1), self.total_packed_dim)
            attn_delta = self._apply_stiefel_svd_readout(
                phi_b, self.w_obs_u.float(), self.obs_sigma.float(), self.w_obs_vt.float()
            )
            x_mid = x_base + attn_delta
            chan_feat = self.compute_channel_features(x_mid)
            chan_chunks.append(chan_feat.float().reshape(-1, self.channel_hidden_dim))
            rem_targets.append((x_out_list[idx].float() - x_mid).reshape(-1, self.embed_dim))

        chan_all = torch.cat(chan_chunks, dim=0)
        rem_all = torch.cat(rem_targets, dim=0)
        gram_chan = torch.matmul(chan_all.T, chan_all) + (ridge_lambda * n_samples) * torch.eye(
            self.channel_hidden_dim, device=chan_all.device, dtype=chan_all.dtype
        )
        w_chan_full = torch.linalg.solve(gram_chan, torch.matmul(chan_all.T, rem_all))
        u_ch, sig_ch, vt_ch = polar_factorize_stiefel_svd(w_chan_full)
        self.w_chan_u.copy_(u_ch.to(self.w_chan_u.dtype))
        self.chan_sigma.copy_(sig_ch.to(self.chan_sigma.dtype))
        self.w_chan_vt.copy_(vt_ch.to(self.w_chan_vt.dtype))

    def forward(
        self,
        x: torch.Tensor,
        x_0: torch.Tensor | None = None,
        cos_4d: torch.Tensor | None = None,
        sin_4d: torch.Tensor | None = None,
        v_0: torch.Tensor | None = None,
        ve_l: torch.Tensor | None = None,
        block_mask: Any = None,
        q_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass compatible with both standalone `RiemannianImpureDensityGPT` and `CompiledLeaderboardClassicalGPT`."""
        _ = (cos_4d, sin_4d, ve_l, block_mask, q_scale)
        work_dtype = x.dtype
        if x_0 is not None:
            x_cur = self.block_lambdas[0].to(work_dtype) * x + self.block_lambdas[1].to(work_dtype) * x_0.to(work_dtype)
        else:
            x_cur = x

        packed_concat, v_0_next = self.compute_packed_density_features(x_cur, v_0=v_0)
        attn_delta = self._apply_stiefel_svd_readout(packed_concat, self.w_obs_u, self.obs_sigma, self.w_obs_vt)
        x_mid = x_cur + attn_delta

        interference = self.compute_channel_features(x_mid)
        chan_delta = self._apply_stiefel_svd_readout(interference, self.w_chan_u, self.chan_sigma, self.w_chan_vt)
        x_out = x_mid + chan_delta
        return x_out, v_0_next


class AllLayerRiemannianStiefelMuon(torch.optim.Optimizer):
    """Unified optimizer stepping all 2D real Stiefel matrices and 3D complex $U(d_h)$ unitaries across all layers."""

    def __init__(
        self,
        real_stiefel_params: list[nn.Parameter],
        complex_unitary_pairs: list[tuple[nn.Parameter, nn.Parameter]],
        stiefel_lr: float = DEFAULT_STIEFEL_GEODESIC_LR,
        unitary_lr: float = DEFAULT_UNITARY_GEODESIC_LR,
        beta: float = DEFAULT_RIEMANNIAN_MOMENTUM_BETA,
        ns_steps: int = DEFAULT_NS5_STEPS,
        use_geodesic_retraction: bool = True,
    ) -> None:
        self.complex_unitary_pairs = list(complex_unitary_pairs)
        self.unitary_lr = float(unitary_lr)
        self.use_geodesic_retraction = bool(use_geodesic_retraction)
        all_params = list(real_stiefel_params)
        for u_re, u_im in self.complex_unitary_pairs:
            all_params.extend([u_re, u_im])
        defaults = {
            "lr": float(stiefel_lr),
            "unitary_lr": float(unitary_lr),
            "beta": float(beta),
            "ns_steps": int(ns_steps),
        }
        super().__init__(all_params, defaults)
        self._unitary_ids = {id(u_re) for u_re, _ in self.complex_unitary_pairs} | {
            id(u_im) for _, u_im in self.complex_unitary_pairs
        }

    def set_lr_Scale(self, lr_mult: float) -> None:
        """Scale geodesic step size by `lr_mult`."""
        for group in self.param_groups:
            group["lr_mult"] = float(lr_mult)

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr_mult = float(group.get("lr_mult", 1.0))
        stiefel_lr = float(group["lr"]) * lr_mult
        unitary_lr = float(group["unitary_lr"]) * lr_mult
        beta = float(group["beta"])
        ns_steps = int(group["ns_steps"])

        for p in group["params"]:
            if id(p) in self._unitary_ids or p.grad is None:
                continue
            eff_lr = (
                stiefel_lr * DEFAULT_READOUT_STIEFEL_LR_MULT
                if getattr(p, "_is_svd_readout_stiefel", False)
                else stiefel_lr
            )
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
            buf = state["momentum_buffer"]
            if self.use_geodesic_retraction:
                riemannian_stiefel_muon_step(p, p.grad, buf, lr=eff_lr, beta=beta, ns_steps=ns_steps)
            else:
                # Euclidean Muon baseline (chord step without tangent projection or 1/sqrt(1+lr^2) retraction)
                buf.mul_(beta).add_(p.grad.float())
                g_nest = p.grad.float() + beta * buf
                w_f = p.float()
                transposed = w_f.size(-2) < w_f.size(-1)
                if transposed:
                    g_nest = g_nest.transpose(-2, -1)
                from ..layers.impure_density_mirror_block import _newton_schulz5_tall_real

                u_step = _newton_schulz5_tall_real(g_nest, steps=ns_steps)
                if transposed:
                    u_step = u_step.transpose(-2, -1)
                p.add_(u_step.to(p.dtype), alpha=-eff_lr)

        for u_re, u_im in self.complex_unitary_pairs:
            if u_re.grad is None or u_im.grad is None:
                continue
            st_re = self.state[u_re]
            st_im = self.state[u_im]
            if "momentum_buffer" not in st_re:
                st_re["momentum_buffer"] = torch.zeros_like(u_re, dtype=torch.float32)
            if "momentum_buffer" not in st_im:
                st_im["momentum_buffer"] = torch.zeros_like(u_im, dtype=torch.float32)
            if self.use_geodesic_retraction:
                for h_idx in range(u_re.size(0)):
                    riemannian_complex_unitary_muon_step(
                        u_re[h_idx],
                        u_im[h_idx],
                        u_re.grad[h_idx],
                        u_im.grad[h_idx],
                        st_re["momentum_buffer"][h_idx],
                        st_im["momentum_buffer"][h_idx],
                        lr=unitary_lr,
                        beta=beta,
                    )
            else:
                st_re["momentum_buffer"].mul_(beta).add_(u_re.grad.float())
                st_im["momentum_buffer"].mul_(beta).add_(u_im.grad.float())
                g_re = u_re.grad.float() + beta * st_re["momentum_buffer"]
                g_im = u_im.grad.float() + beta * st_im["momentum_buffer"]
                z_re, z_im = _complex_newton_schulz5(g_re, g_im, steps=ns_steps)
                u_re.add_(z_re.to(u_re.dtype), alpha=-unitary_lr)
                u_im.add_(z_im.to(u_im.dtype), alpha=-unitary_lr)

        return loss


class RiemannianImpureDensityGPT(nn.Module):
    """Full $L$-layer Quantum Language Model with `RiemannianStiefelMuon` in all layers (`Wave 47`)."""

    def __init__(self, cfg: RiemannianImpureDensityConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else RiemannianImpureDensityConfig()
        self.vocab_size = self.cfg.vocab_size
        self.embed_dim = self.cfg.embed_dim
        self.num_layers = self.cfg.num_layers

        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)
        with torch.no_grad():
            nn.init.normal_(self.wte.weight, mean=0.0, std=1.0 / math.sqrt(self.embed_dim))
            self.wte.weight.copy_(
                F.normalize(self.wte.weight, p=2, dim=-1) * math.sqrt(float(self.embed_dim))
            )

        self.w_in_stiefel = nn.Parameter(torch.eye(self.embed_dim, dtype=torch.float32))
        self.in_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        if self.cfg.boundary_mode in (
            InputBoundaryMode.ANCHORED_SPHERE,
            InputBoundaryMode.TEACHER_FROZEN_BOUNDARY,
        ):
            self.wte.weight.requires_grad_(False)

        self.blocks = nn.ModuleList([
            RiemannianImpureDensityBlock(self.cfg, layer_idx=l_idx) for l_idx in range(self.num_layers)
        ])

        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(
                F.normalize(self.wte.weight, p=2, dim=-1) * 0.5
            )
        if self.cfg.boundary_mode == InputBoundaryMode.TEACHER_FROZEN_BOUNDARY:
            self.lm_head.weight.requires_grad_(False)

    def all_real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Collect all 2D real Stiefel parameters across the input rotation and all $L$ layers."""
        params: list[nn.Parameter] = []
        if self.w_in_stiefel.requires_grad:
            params.append(self.w_in_stiefel)
        for blk in self.blocks:
            params.extend(blk.real_stiefel_parameters())
        return params

    def all_complex_unitary_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        """Collect all 3D complex head unitary pairs `(U_re, U_im)` across all $L$ layers."""
        return [blk.complex_unitary_pair() for blk in self.blocks]

    def all_euclidean_parameters(self) -> list[nn.Parameter]:
        """Collect all 1D singular-value spectra, residual scales, and trainable boundary parameters."""
        params: list[nn.Parameter] = [self.in_scale]
        if self.wte.weight.requires_grad:
            params.append(self.wte.weight)
        for blk in self.blocks:
            params.extend(blk.scalar_parameters())
        if self.lm_head.weight.requires_grad:
            params.append(self.lm_head.weight)
        return params

    @torch.no_grad()
    def compute_manifold_and_purity_telemetry(self) -> dict[str, float]:
        """Measure mean Stiefel error, complex $U(d_h)$ unitary error, and McWeeny purity across all $L$ layers."""
        stiefel_errs: list[float] = []
        for w in self.all_real_stiefel_parameters():
            wf = w.float()
            if wf.size(0) < wf.size(1):
                wf = wf.T
            d_cols = wf.size(1)
            eye = torch.eye(d_cols, device=wf.device, dtype=wf.dtype)
            err = float(torch.linalg.matrix_norm(wf.T @ wf - eye, ord="fro").item()) / math.sqrt(float(d_cols))
            stiefel_errs.append(err)

        unitary_errs: list[float] = []
        for u_re, u_im in self.all_complex_unitary_pairs():
            ur = u_re.float()
            ui = u_im.float()
            d_h = ur.size(-1)
            eye_h = torch.eye(d_h, device=ur.device, dtype=ur.dtype).unsqueeze(0)
            gram_re = torch.matmul(ur.transpose(-2, -1), ur) + torch.matmul(ui.transpose(-2, -1), ui)
            gram_im = torch.matmul(ur.transpose(-2, -1), ui) - torch.matmul(ui.transpose(-2, -1), ur)
            u_err = float(
                ((gram_re - eye_h).square().sum(dim=(-2, -1)) + gram_im.square().sum(dim=(-2, -1))).sqrt().mean().item()
            ) / math.sqrt(float(d_h))
            unitary_errs.append(u_err)

        purities = [blk.last_telemetry.get("mean_purity", 1.0) for blk in self.blocks]
        eff_ranks = [blk.last_telemetry.get("mean_eff_rank", 1.0) for blk in self.blocks]
        return {
            "mean_stiefel_err": float(sum(stiefel_errs) / max(1, len(stiefel_errs))),
            "max_stiefel_err": float(max(stiefel_errs) if stiefel_errs else 0.0),
            "mean_unitary_err": float(sum(unitary_errs) / max(1, len(unitary_errs))),
            "mean_purity": float(sum(purities) / max(1, len(purities))),
            "mean_eff_rank": float(sum(eff_ranks) / max(1, len(eff_ranks))),
        }

    def forward_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        """Compute final normalized hidden states `(B, T, d)` across all $L$ quantum density blocks."""
        work_dtype = torch.bfloat16 if (idx.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
        tok_emb = self.wte(idx).to(work_dtype)
        if self.cfg.boundary_mode == InputBoundaryMode.ANCHORED_SPHERE:
            tok_unit = F.normalize(tok_emb.float(), p=2, dim=-1).to(work_dtype) * math.sqrt(float(self.embed_dim))
            x = torch.matmul(tok_unit, self.w_in_stiefel.to(work_dtype)) * self.in_scale.to(work_dtype)
        elif self.cfg.boundary_mode == InputBoundaryMode.TEACHER_FROZEN_BOUNDARY:
            x = F.rms_norm(tok_emb, (self.embed_dim,), eps=TRACE_NORM_EPS)
            x = torch.matmul(x, self.w_in_stiefel.to(work_dtype))
        else:
            x = torch.matmul(tok_emb, self.w_in_stiefel.to(work_dtype)) * self.in_scale.to(work_dtype)
        x_0 = x
        v_0: torch.Tensor | None = None

        for blk in self.blocks:
            x, v_0 = blk(x, x_0=x_0, v_0=v_0)

        return F.rms_norm(x, (self.embed_dim,), eps=TRACE_NORM_EPS)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Forward pass using zero-extra-VRAM in-place Triton softcap chunked CE when `targets` is provided."""
        h_final = self.forward_hidden(idx)
        loss: torch.Tensor | None = None
        logits: torch.Tensor | None = None

        if targets is not None:
            loss = _CompiledChunkedSoftcapLinearCEFn.apply(
                h_final.reshape(-1, self.embed_dim),
                self.lm_head.weight,
                targets.reshape(-1),
                DEFAULT_LOGIT_SOFTCAP,
                DEFAULT_CE_CHUNK_TOKENS,
            )

        if return_logits or targets is None:
            raw_logits = F.linear(h_final, self.lm_head.weight.to(h_final.dtype)).float()
            logits = DEFAULT_LOGIT_SOFTCAP * torch.tanh(raw_logits / DEFAULT_LOGIT_SOFTCAP)

        return logits, loss


def build_all_layer_riemannian_optimizers(
    model: RiemannianImpureDensityGPT,
    stiefel_lr: float = DEFAULT_STIEFEL_GEODESIC_LR,
    unitary_lr: float = DEFAULT_UNITARY_GEODESIC_LR,
    scalar_lr: float = 0.005,
    head_lr: float = 0.004,
    use_geodesic_retraction: bool = True,
) -> tuple[AllLayerRiemannianStiefelMuon, torch.optim.Optimizer]:
    """Build `AllLayerRiemannianStiefelMuon` for all 2D/3D Stiefel/Unitary matrices + `AdamW` for 1D scalars/head."""
    stiefel_opt = AllLayerRiemannianStiefelMuon(
        real_stiefel_params=model.all_real_stiefel_parameters(),
        complex_unitary_pairs=model.all_complex_unitary_pairs(),
        stiefel_lr=stiefel_lr,
        unitary_lr=unitary_lr,
        use_geodesic_retraction=use_geodesic_retraction,
    )

    scalar_params: list[nn.Parameter] = [model.in_scale]
    for blk in model.blocks:
        scalar_params.extend(blk.scalar_parameters())

    euclid_groups: list[dict[str, Any]] = [
        {"params": [p for p in scalar_params if p.requires_grad], "lr": float(scalar_lr), "weight_decay": 0.0},
    ]
    if model.wte.weight.requires_grad:
        euclid_groups.append({"params": [model.wte.weight], "lr": float(head_lr) * 2.0, "weight_decay": 0.0})
    if model.lm_head.weight.requires_grad:
        euclid_groups.append({"params": [model.lm_head.weight], "lr": float(head_lr), "weight_decay": 0.0})

    euclid_opt = torch.optim.AdamW(euclid_groups, betas=(0.9, 0.95), eps=1e-8)
    return stiefel_opt, euclid_opt


__all__ = [
    "AllLayerRiemannianStiefelMuon",
    "DensityStateMode",
    "InputBoundaryMode",
    "RiemannianImpureDensityBlock",
    "RiemannianImpureDensityConfig",
    "RiemannianImpureDensityGPT",
    "build_all_layer_riemannian_optimizers",
    "fast_rank_r_token_density_pack",
    "polar_factorize_stiefel_svd",
    "riemannian_complex_unitary_muon_step",
]
