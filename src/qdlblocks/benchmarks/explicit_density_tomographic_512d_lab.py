r"""Architectural Consolidation: Explicit $\mathbb{C}^{16 \times 16}$ Outer-Product Density-Matrix Highway ($H = 16$ Heads), `16 × 32 = 512D` Quantum State Tomography Readout ($\operatorname{Tr}(\rho_t E_v)$), & Zero-Synthesis Single-GEMM `512D` Sphere (`EXPERIMENT_PLAN_083.md` / `Wave 63`).

Mathematical & Architectural Overview
-------------------------------------
Consolidates the multi-wave lessons from `Waves 58–62` (`EXPERIMENT_PLAN_078`–`082`) into a **single, flat,
zero-inheritance module** with strictly **power-of-2 tensor dimensions** (`d = 512` real = $H = 16$ heads of $\mathbb{C}^{16}$):

1. **Power-of-2 Complex Wavepackets & Explicit $\mathbb{C}^{16 \times 16}$ Outer-Product Density Highway**:
   - Each token $v \in \{0, \dots, 50303\}$ looks up a `512D` real embedding reshaped into $H = 16$ unit-norm
     complex wavefunctions $|\psi_{t,h}^{(0)}\rangle \in \mathbb{C}^{16}$ (`16 × 16 × 2 = 512` floats).
   - At Layer 0, each head forms the explicit $16 \times 16$ Hermitian outer-product density matrix $\rho_{t,h}^{(0)} \in \operatorname{Herm}(\mathbb{C}^{16})$
     from the local causal wavepacket window ($|\psi_{t,h}^{(0)}\rangle$, phase-transported $|\tilde{\psi}_{t-1,h}^{(0)}\rangle$,
     and their normalized complex Schur interference ket $|\xi_{t,h}^{(0)}\rangle$), packed bijectively via
     [`pack_rank1_ket_outer_isometric`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/layers/impure_density_mirror_block.py#L125-L130)
     into a **`256D` real Frobenius-isometric vector** $\tilde{\boldsymbol{\rho}}_{t,h}^{(0)} = \tilde{P}(\rho_{t,h}^{(0)}) \in \mathbb{R}^{256}$:
     $$\big\langle \tilde{P}(A),\; \tilde{P}(B) \big\rangle_{\mathbb{R}^{256}} \;=\; \operatorname{Tr}(AB), \qquad \|\tilde{P}(\rho_{t,h})\|_2^2 \;=\; \operatorname{Tr}(\rho_{t,h}^2)$$
   - In each of the $L = 4$ Quantum Density Highway blocks, causal Hermitian-Born attention (equipped with the
     **Trunk "No-Click" POVM Attention Vacuum Sink** $p_{\text{noclick},t,h}$) mixes both the `32D` complex value ket
     and the `256D` packed density matrix in a **single fused `(B, 16, T, 288)` Tensor-Core BMM**, followed by
     per-head $U(16)$ unitary conjugation $\rho_{t,h} \mapsto U_h \rho_{t,h} U_h^\dagger$ and a Mach-Zehnder Stiefel interferometer.

2. **`16 × 32 = 512D` Quantum State Tomography Readout ($\langle \mathbf{w}_v, \mathbf{q}_t \rangle = \sum_h \operatorname{Tr}(\rho_{t,h} E_{v,h})$)**:
   - Each head $h \in \{0, \dots, 15\}$ measures $\rho_{t,h} \in \operatorname{Herm}(\mathbb{C}^{16})$ against
     $K = 32$ learned unit-norm complex probe kets $|c_{k,h}\rangle \in \mathbb{C}^{16}$ (`16 × 32 = 512` real expectations):
     $$q_{t,h,k} \;=\; \langle c_{k,h} \mid \rho_{t,h} \mid c_{k,h} \rangle \;=\; \big\langle \tilde{\boldsymbol{\rho}}_{t,h},\; \tilde{P}(|c_{k,h}\rangle\langle c_{k,h}|) \big\rangle_{\mathbb{R}^{256}} \;\in\; \mathbb{R}^{512}$$
   - Normalized on the unified `512D` sphere with orthogonal vacuum subspace $\mathcal{H}_{\text{vac}}^{(4)}$ in the
     denominator, a **single `(N, 512) @ (512, 50304)` GEMM** evaluates $\sum_{h=0}^{15} \operatorname{Tr}(\rho_{t,h} E_{v,h})$
     with **zero `50,304 × 512` detector matrix re-synthesis** during training or `10.48M`-token validation.
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

from qdlblocks.benchmarks.experiment_logger import ExperimentIntentSpec, StandardizedExperimentLogger
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import (
    DEFAULT_SHALLOW_NGRAM_LR, DEFAULT_SHALLOW_READOUT_LR, DEFAULT_SHALLOW_SCALAR_LR, DEFAULT_SHALLOW_STIEFEL_LR,
    DEFAULT_SHALLOW_VOCAB_SIZE, DEFAULT_SHALLOW_WTE_LR, DEFAULT_SOFTCAP_LOGIT, RAW_GPT2_VOCAB_SIZE, LongSeqFineWebStream,
)
from qdlblocks.benchmarks.zero_ppmi_byte_pair_cholesky_lab import (
    OnlineCausalZipfTracker, _compute_zero_ppmi_step_lr_and_momentum, build_zero_data_bpe_rank_zipf_prior,
)
from qdlblocks.kernels.fused_quantum_wavepacket_attn import _FusedHermitianBornRhoKetAttnFn
from qdlblocks.layers.impure_density_mirror_block import (
    DEFAULT_TIKHONOV_GRAM_LAMBDA, TRACE_NORM_EPS, _complex_hermitian_mv, _get_hermitian_isometric_masks,
    _pack_rank1_ket_outer_masked, complex_l2_normalize, pack_hermitian_isometric,
    pack_rank1_ket_outer_isometric, unpack_hermitian_isometric,
)
from qdlblocks.layers.quantum_statmech_channels import StiefelMomentumMode
from qdlblocks.layers.quantum_svd_stiefel_graft import BiorthogonalSVDStiefelProjection, QuantumStiefelProjMode
from qdlblocks.models.compiled_leaderboard_classical_gpt import NativeLeaderboardRotary, apply_native_leaderboard_rotary
from qdlblocks.models.modular_quantum_statmech_gpt import ModularRiemannianStiefelKrausMuon
from qdlblocks.models.quantum_full_backbone_graft_gpt import SVDStiefelMachZehnderInterferometer
from qdlblocks.optim.vocab_manifold_muon import BatchSubspaceTikhonovGramMuon, RiemannianGeodesicScalarAndProbeOptimizer

# ==============================================================================
# Top-of-File Named Constants (Rule 7 — Synchronized with EXPERIMENT_PLAN_083.md)
# ==============================================================================
DEFAULT_TOMO_EMBED_DIM: int = 512
DEFAULT_TOMO_NUM_HEADS: int = 16
DEFAULT_TOMO_HEAD_COMPLEX_DIM: int = 16
DEFAULT_TOMO_PACKED_RHO_DIM: int = 256
DEFAULT_PROBES_PER_HEAD: int = 32
DEFAULT_TOMO_NUM_LAYERS: int = 4
DEFAULT_TOMO_MLP_HIDDEN: int = 1024
DEFAULT_TOMO_VACUUM_DIM: int = 4
DEFAULT_INIT_VACUUM_REST_MASS: float = 0.35
DEFAULT_INIT_NOCLICK_BIAS: float = -2.20
DEFAULT_INIT_NOCLICK_POS_LOGIT: float = -1.10
DEFAULT_NOCLICK_RHO_DAMPING: float = 0.50
DEFAULT_INIT_VALUE_EXCITATION_ALPHA: float = 0.30
DEFAULT_INIT_HIGHWAY_BLEND_GAMMA: float = 0.35
DEFAULT_INIT_MLP_RHO_INJECT: float = 0.20
DEFAULT_INIT_TOMO_SCALE: float = 0.45
DEFAULT_INIT_FILTER_SCALE: float = 0.35
DEFAULT_INIT_READOUT_TEMP: float = 0.50
DEFAULT_INIT_BILINEAR_REGISTER_SCALE: float = 0.00
DEFAULT_REGISTER_BUCKETS: int = 256
DEFAULT_TOK_NGRAM_BUCKETS: int = 4096
TOMO_CE_TOKEN_CHUNK: int = 2048

_COMPILED_LAYER0_OUTER_FN: Any = None
_COMPILED_CONJ_UNITARY_FN: Any = None


class ExplicitDensityVariantMode(str, Enum):
    """Ablation and scaling modes for `EXPERIMENT_PLAN_083.md` (`Wave 63`, Rule 3)."""

    T0_PURE_WAVEPACKET_RANK1_BASELINE = "t0_pure_wavepacket_rank1_baseline"
    T1_EXPLICIT_OUTER_DENSITY_TOMOGRAPHIC_512D = "t1_explicit_outer_density_tomographic_512d"
    T2_DENSITY_TOMOGRAPHY_PLUS_NOCLICK_AND_VACUUM = "t2_density_tomography_plus_noclick_and_vacuum"
    T3_ZERO_SYNTHESIS_TWO_REGISTER_BILINEAR_TRACE = "t3_zero_synthesis_two_register_bilinear_trace"
    T4_CONSOLIDATED_TOMOGRAPHIC_SCALING_1200 = "t4_consolidated_tomographic_scaling_1200"


@dataclass(frozen=True)
class ExplicitDensityTomographicSpec:
    r"""Configuration for `ExplicitDensityTomographicGPT` (`<= 8` fields, Rule 2)."""

    variant_mode: ExplicitDensityVariantMode = ExplicitDensityVariantMode.T2_DENSITY_TOMOGRAPHY_PLUS_NOCLICK_AND_VACUUM
    enable_density_highway: bool = True
    enable_tomographic_readout: bool = True
    enable_noclick_and_vacuum: bool = True
    enable_two_register_bilinear: bool = False
    num_layers: int = DEFAULT_TOMO_NUM_LAYERS
    vacuum_dim: int = DEFAULT_TOMO_VACUUM_DIM
    seed: int = 1337


def _build_layer0_outer_masked(
    psi_re: torch.Tensor, psi_im: torch.Tensor, lag_phase: torch.Tensor,
    mix_logits: torch.Tensor, pack_re: torch.Tensor, pack_im_t: torch.Tensor,
) -> torch.Tensor:
    """Pure tensor Layer-0 3-ket outer-product mixture for Inductor fusion (`<= 8` args)."""
    work_dtype = psi_re.dtype
    pi = F.softmax(mix_logits.float(), dim=0).to(work_dtype)
    p0_packed = _pack_rank1_ket_outer_masked(psi_re, psi_im, pack_re, pack_im_t)
    if psi_re.size(2) <= 1:
        return p0_packed

    prev_re = torch.cat([psi_re[:, :, :1, :], psi_re[:, :, :-1, :]], dim=2)
    prev_im = torch.cat([psi_im[:, :, :1, :], psi_im[:, :, :-1, :]], dim=2)
    cos_p = torch.cos(lag_phase.float()).to(work_dtype).view(1, -1, 1, 1)
    sin_p = torch.sin(lag_phase.float()).to(work_dtype).view(1, -1, 1, 1)
    rot_re = prev_re * cos_p - prev_im * sin_p
    rot_im = prev_re * sin_p + prev_im * cos_p
    p1_packed = _pack_rank1_ket_outer_masked(rot_re, rot_im, pack_re, pack_im_t)

    schur_re = psi_re * rot_re - psi_im * rot_im
    schur_im = psi_re * rot_im + psi_im * rot_re
    xi_re, xi_im = complex_l2_normalize(schur_re, schur_im)
    p2_packed = _pack_rank1_ket_outer_masked(xi_re, xi_im, pack_re, pack_im_t)
    return (pi[0] * p0_packed + pi[1] * p1_packed + pi[2] * p2_packed).contiguous()


def build_layer0_explicit_outer_density(
    psi_re: torch.Tensor, psi_im: torch.Tensor, lag_phase: torch.Tensor, mix_logits: torch.Tensor
) -> torch.Tensor:
    r"""Assemble Layer-0 explicit $\operatorname{Herm}(\mathbb{C}^{16})$ outer-product density matrix packed in $\mathbb{R}^{256}$ (`<= 8` args)."""
    global _COMPILED_LAYER0_OUTER_FN
    d_h = psi_re.size(-1)
    pack_re, pack_im_t, _, _ = _get_hermitian_isometric_masks(d_h, psi_re.device, psi_re.dtype)
    if psi_re.is_cuda and psi_re.size(2) > 1 and not torch.compiler.is_compiling():
        if _COMPILED_LAYER0_OUTER_FN is None:
            _COMPILED_LAYER0_OUTER_FN = torch.compile(_build_layer0_outer_masked, dynamic=False)
        return _COMPILED_LAYER0_OUTER_FN(psi_re, psi_im, lag_phase, mix_logits, pack_re, pack_im_t)
    return _build_layer0_outer_masked(psi_re, psi_im, lag_phase, mix_logits, pack_re, pack_im_t)


def _conjugate_packed_hermitian_masked(
    rho_packed: torch.Tensor, u_re: torch.Tensor, u_im: torch.Tensor,
    pack_re: torch.Tensor, pack_im_t: torch.Tensor, unpack_re_half: torch.Tensor, unpack_im_low: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure tensor 4D unitary conjugation $U_h \rho_{t,h} U_h^\dagger$ for Inductor fusion (`<= 8` args)."""
    bsz, num_h, seq_len, _ = rho_packed.shape
    work_dtype = rho_packed.dtype
    d_h = u_re.size(-1)
    mat = rho_packed.unflatten(dim=-1, sizes=(d_h, d_h))
    half_re = mat * unpack_re_half
    low_im = mat * unpack_im_low
    rho_re = half_re + half_re.transpose(-2, -1)
    rho_im = low_im.transpose(-2, -1) - low_im

    urt = u_re.to(work_dtype).transpose(-2, -1).unsqueeze(0)
    uit = u_im.to(work_dtype).transpose(-2, -1).unsqueeze(0)

    r_re_flat = rho_re.reshape(bsz, num_h, seq_len * d_h, d_h)
    r_im_flat = rho_im.reshape(bsz, num_h, seq_len * d_h, d_h)
    n_re = (torch.matmul(r_re_flat, urt) + torch.matmul(r_im_flat, uit)).view(bsz, num_h, seq_len, d_h, d_h)
    n_im = (torch.matmul(r_im_flat, urt) - torch.matmul(r_re_flat, uit)).view(bsz, num_h, seq_len, d_h, d_h)

    nt_re_flat = n_re.transpose(-2, -1).reshape(bsz, num_h, seq_len * d_h, d_h)
    nt_im_flat = n_im.transpose(-2, -1).reshape(bsz, num_h, seq_len * d_h, d_h)
    rot_t_re = (torch.matmul(nt_re_flat, urt) - torch.matmul(nt_im_flat, uit)).view(bsz, num_h, seq_len, d_h, d_h)
    rot_t_im = (torch.matmul(nt_re_flat, uit) + torch.matmul(nt_im_flat, urt)).view(bsz, num_h, seq_len, d_h, d_h)

    rot_re = 0.5 * (rot_t_re.transpose(-2, -1) + rot_t_re)
    rot_im = 0.5 * (rot_t_im.transpose(-2, -1) - rot_t_im)
    rot_packed = (rot_re * pack_re + rot_im.transpose(-2, -1) * pack_im_t).flatten(start_dim=-2, end_dim=-1)
    return rot_packed, rot_re, rot_im


def conjugate_packed_hermitian_by_unitary(
    rho_packed: torch.Tensor, u_re: torch.Tensor, u_im: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Apply per-head $U(16)$ unitary evolution $\rho_{t,h}^{\text{rot}} = U_h \rho_{t,h} U_h^\dagger$ (`<= 8` args)."""
    global _COMPILED_CONJ_UNITARY_FN
    d_h = u_re.size(-1)
    p_re, p_im_t, u_re_h, u_im_l = _get_hermitian_isometric_masks(d_h, rho_packed.device, rho_packed.dtype)
    if rho_packed.is_cuda and not torch.compiler.is_compiling():
        if _COMPILED_CONJ_UNITARY_FN is None:
            _COMPILED_CONJ_UNITARY_FN = torch.compile(_conjugate_packed_hermitian_masked, dynamic=False)
        return _COMPILED_CONJ_UNITARY_FN(rho_packed, u_re, u_im, p_re, p_im_t, u_re_h, u_im_l)
    return _conjugate_packed_hermitian_masked(rho_packed, u_re, u_im, p_re, p_im_t, u_re_h, u_im_l)


def compute_tomographic_512d_probe_expectations(
    rho_packed: torch.Tensor, probe_re: torch.Tensor, probe_im: torch.Tensor
) -> torch.Tensor:
    r"""Evaluate $16 \times 32 = 512$ Born probe expectations $q_{t,h,k} = \langle c_{k,h} \mid \rho_{t,h} \mid c_{k,h} \rangle$ (`<= 8` args).

    Uses the Frobenius-isometric identity:
    $$\langle c_{k,h} \mid \rho_{t,h} \mid c_{k,h} \rangle = \big\langle \tilde{P}(\rho_{t,h}),\; \tilde{P}(|c_{k,h}\rangle\langle c_{k,h}|) \big\rangle_{\mathbb{R}^{256}}$$
    so packing happens once on the `(16, 32, 16)` probe parameters and contracts with `rho_packed` `(B, 16, T, 256)`
    via a single `(B, 16, T, 256) @ (1, 16, 256, 32)` matmul.
    """
    bsz, num_h, seq_len, _ = rho_packed.shape
    c_re, c_im = complex_l2_normalize(probe_re.float(), probe_im.float())
    c_outer_packed = pack_rank1_ket_outer_isometric(c_re, c_im).to(rho_packed.dtype)
    q_bhtk = torch.matmul(rho_packed, c_outer_packed.transpose(-1, -2).unsqueeze(0))
    return q_bhtk.transpose(1, 2).reshape(bsz, seq_len, num_h * c_re.size(1)).contiguous()


def apply_unified_512d_vacuum_sphere_norm(
    u_vocab_512: torch.Tensor, u_vac: torch.Tensor | None, vac_rest_mass: torch.Tensor,
    readout_temp: torch.Tensor, enable_vacuum: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Normalize `u_vocab_512` `(B, T, 512)` on $\mathcal{H}_{\text{vocab}}^{(512)} \oplus \mathcal{H}_{\text{vac}}^{(d_{\text{vac}})}$ (`<= 8` args)."""
    ms_vocab = u_vocab_512.float().square().mean(dim=-1, keepdim=True)
    if enable_vacuum:
        m_sq = vac_rest_mass.float().square()
        e_vac = m_sq + u_vac.float().square().mean(dim=-1, keepdim=True) if (u_vac is not None and u_vac.numel() > 0) else m_sq
        denom = torch.sqrt((ms_vocab + e_vac).clamp_min(1e-6))
        cos_theta = torch.sqrt(ms_vocab.clamp_min(1e-8) / (ms_vocab + e_vac).clamp_min(1e-6))
        mean_cos_t = cos_theta.detach().mean()
    else:
        denom = torch.sqrt(ms_vocab.clamp_min(1e-6))
        mean_cos_t = torch.ones((), device=u_vocab_512.device, dtype=torch.float32)

    temp = readout_temp.float().clamp(0.05, 45.0) / float(u_vocab_512.size(-1))
    h_norm = ((u_vocab_512.float() / denom) * temp).to(u_vocab_512.dtype).contiguous()
    return h_norm, mean_cos_t


class _ExplicitDensityHighwayBlock(nn.Module):
    r"""Quantum Density Highway Block evolving $H = 16$ heads of $\mathbb{C}^{16}$ wavepackets and $\operatorname{Herm}(\mathbb{C}^{16})$ density matrices."""

    def __init__(
        self, embed_dim: int = DEFAULT_TOMO_EMBED_DIM, num_heads: int = DEFAULT_TOMO_NUM_HEADS,
        head_complex_dim: int = DEFAULT_TOMO_HEAD_COMPLEX_DIM, mlp_hidden: int = DEFAULT_TOMO_MLP_HIDDEN,
        enable_density_highway: bool = True, enable_noclick_sink: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim, self.num_heads, self.d_h = int(embed_dim), int(num_heads), int(head_complex_dim)
        self.head_real_dim = 2 * self.d_h
        self.packed_rho_dim = self.d_h * self.d_h
        self.enable_density_highway = bool(enable_density_highway)
        self.enable_noclick_sink = bool(enable_noclick_sink)
        self.gate_in_ch = min(16, self.embed_dim)
        self.sdpa_scale = 1.0 / math.sqrt(float(self.d_h))

        self.block_lambdas = nn.Parameter(torch.tensor([1.0, 0.15], dtype=torch.float32))
        self.sa_lambdas = nn.Parameter(torch.tensor([1.0, 0.35], dtype=torch.float32))

        self.q_proj = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)
        self.k_proj = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)
        self.v_proj = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)
        self.obs_proj = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)

        eye_h = torch.eye(self.d_h, dtype=torch.float32).unsqueeze(0).expand(self.num_heads, self.d_h, self.d_h).clone()
        self.u_evol_re = nn.Parameter(eye_h)
        self.u_evol_im = nn.Parameter(torch.zeros(self.num_heads, self.d_h, self.d_h, dtype=torch.float32))

        self.berry_qk_phase = nn.Parameter(torch.zeros(self.num_heads, 1, self.d_h, dtype=torch.float32))
        self.born_qk_beta = nn.Parameter(torch.full((self.num_heads,), 0.20, dtype=torch.float32))
        self.val_exc_alpha = nn.Parameter(torch.full((self.num_heads,), DEFAULT_INIT_VALUE_EXCITATION_ALPHA, dtype=torch.float32))
        self.highway_gamma = nn.Parameter(torch.full((self.num_heads,), DEFAULT_INIT_HIGHWAY_BLEND_GAMMA, dtype=torch.float32))
        self.rho_ket_coupling = nn.Parameter(torch.full((self.num_heads,), 0.30, dtype=torch.float32))
        self.mlp_rho_inject = nn.Parameter(torch.full((self.num_heads,), DEFAULT_INIT_MLP_RHO_INJECT, dtype=torch.float32))
        self.obs_scale = nn.Parameter(torch.tensor(0.45, dtype=torch.float32))

        # Trunk No-Click POVM Attention Vacuum Sink parameters in C^{16}
        vac_init = F.normalize(torch.randn(self.num_heads, self.head_real_dim, dtype=torch.float32), dim=-1)
        self.k_vac_ket = nn.Parameter(vac_init * math.sqrt(float(self.head_real_dim)))
        self.noclick_bias = nn.Parameter(torch.full((self.num_heads,), DEFAULT_INIT_NOCLICK_BIAS, dtype=torch.float32))
        self.noclick_pos_logit = nn.Parameter(torch.full((self.num_heads,), DEFAULT_INIT_NOCLICK_POS_LOGIT, dtype=torch.float32))

        self.ve_gate = nn.Linear(self.gate_in_ch, self.num_heads, bias=False)
        self.q_attn_gate = nn.Linear(self.gate_in_ch, self.num_heads, bias=False)
        nn.init.zeros_(self.ve_gate.weight)
        nn.init.zeros_(self.q_attn_gate.weight)

        self.mz_mlp = SVDStiefelMachZehnderInterferometer(
            self.embed_dim, int(mlp_hidden), QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R256
        )
        self._last_noclick_t: torch.Tensor | None = None
        self._last_purity_t: torch.Tensor | None = None

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Return all 2D real Stiefel matrices in this block."""
        params: list[nn.Parameter] = []
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.obs_proj):
            params.extend(proj.real_stiefel_parameters())
        params.extend(self.mz_mlp.real_stiefel_parameters())
        return params

    def complex_unitary_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        """Return paired `(u_evol_re, u_evol_im)` $U(16)$ parameters (Wave 44 Invariant 1)."""
        return [(self.u_evol_re, self.u_evol_im)]

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return scalar gates, singular values, and No-Click POVM parameters."""
        params: list[nn.Parameter] = [self.block_lambdas, self.sa_lambdas]
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.obs_proj):
            params.extend(proj.scalar_parameters())
        params.extend([
            self.berry_qk_phase, self.born_qk_beta, self.val_exc_alpha, self.highway_gamma,
            self.rho_ket_coupling, self.mlp_rho_inject, self.obs_scale, self.k_vac_ket,
            self.noclick_bias, self.noclick_pos_logit, self.ve_gate.weight, self.q_attn_gate.weight,
        ])
        params.extend(self.mz_mlp.scalar_parameters())
        return params

    @torch.no_grad()
    def restrict_scalars_(self) -> None:
        """Clamp singular values and coupling scalars to physical ranges."""
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.obs_proj):
            proj.clamp_sigma_()
        self.block_lambdas.clamp_(-2.5, 2.5)
        self.sa_lambdas.clamp_(-2.5, 2.5)
        self.berry_qk_phase.clamp_(-math.pi, math.pi)
        self.born_qk_beta.clamp_(-1.5, 1.5)
        self.val_exc_alpha.clamp_(0.02, 0.85)
        self.highway_gamma.clamp_(0.05, 0.90)
        self.rho_ket_coupling.clamp_(-1.5, 1.5)
        self.mlp_rho_inject.clamp_(0.01, 0.65)
        self.obs_scale.clamp_(0.05, 3.0)
        self.noclick_bias.clamp_(-5.0, 1.5)
        self.noclick_pos_logit.clamp_(-4.0, 2.0)
        k_norm = F.normalize(self.k_vac_ket.data.float(), dim=-1, eps=1e-6) * math.sqrt(float(self.head_real_dim))
        self.k_vac_ket.data.copy_(k_norm)
        self.mz_mlp.restrict_scalars_()

    def _compute_noclick_sink_prob(
        self, q_re: torch.Tensor, q_im: torch.Tensor, v_bar_re: torch.Tensor, v_bar_im: torch.Tensor
    ) -> torch.Tensor:
        """Compute Trunk 'No-Click' POVM Vacuum Sink probability `p_noclick` `(B, H, T, 1)` (`<= 8` args)."""
        if not self.enable_noclick_sink:
            self._last_noclick_t = None
            return torch.zeros((q_re.size(0), self.num_heads, q_re.size(2), 1), device=q_re.device, dtype=q_re.dtype)

        seq_len = q_re.size(2)
        kv = self.k_vac_ket.float().view(1, self.num_heads, 1, 2, self.d_h)
        kv_re, kv_im = kv[..., 0, :], kv[..., 1, :]
        ov_re = (q_re.float() * kv_re + q_im.float() * kv_im).sum(dim=-1, keepdim=True) * self.sdpa_scale
        ov_im = (q_im.float() * kv_re - q_re.float() * kv_im).sum(dim=-1, keepdim=True) * self.sdpa_scale

        phi = self.berry_qk_phase.float().mean(dim=(-2, -1)).view(1, self.num_heads, 1, 1)
        beta = self.born_qk_beta.float().view(1, self.num_heads, 1, 1)
        s_vac = torch.cos(phi) * ov_re - torch.sin(phi) * ov_im + beta * (ov_re.square() + ov_im.square())

        pos_log = torch.log(torch.arange(1, seq_len + 1, device=q_re.device, dtype=torch.float32)).view(1, 1, seq_len, 1)
        gamma_pos = torch.sigmoid(self.noclick_pos_logit.float()).view(1, self.num_heads, 1, 1)
        b_vac = self.noclick_bias.float().view(1, self.num_heads, 1, 1)
        v_rms = torch.sqrt((v_bar_re.float().square().mean(dim=-1, keepdim=True) + v_bar_im.float().square().mean(dim=-1, keepdim=True)).clamp_min(1e-6))
        p_noclick = torch.sigmoid(b_vac + gamma_pos * pos_log + s_vac - v_rms)
        self._last_noclick_t = p_noclick.detach().mean()
        return p_noclick.to(q_re.dtype)

    def forward(
        self, x: torch.Tensor, x_anchor: torch.Tensor, cos_4d: torch.Tensor, sin_4d: torch.Tensor,
        rho_packed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""Evolve coherent `512D` stream `x` and explicit $\operatorname{Herm}(\mathbb{C}^{16})$ density highway `rho_packed` (`<= 8` args)."""
        bsz, seq_len, d_model = x.shape
        work_dtype = x.dtype
        x_in = self.block_lambdas[0].to(work_dtype) * x + self.block_lambdas[1].to(work_dtype) * x_anchor
        h_in = F.rms_norm(x_in, (d_model,), eps=1e-6)

        q_raw = self.q_proj(h_in).view(bsz, seq_len, self.num_heads, self.head_real_dim)
        k_raw = self.k_proj(h_in).view(bsz, seq_len, self.num_heads, self.head_real_dim)
        v_raw = self.v_proj(h_in).view(bsz, seq_len, self.num_heads, self.head_real_dim)

        g_ve = (2.0 * torch.sigmoid(F.linear(h_in[..., : self.gate_in_ch], self.ve_gate.weight.to(work_dtype)))).unsqueeze(-1)
        v_anc = x_anchor.view(bsz, seq_len, self.num_heads, self.head_real_dim) * g_ve
        v_mixed = self.sa_lambdas[0].to(work_dtype) * v_raw + self.sa_lambdas[1].to(work_dtype) * v_anc

        q_rot = apply_native_leaderboard_rotary(F.rms_norm(q_raw, (self.head_real_dim,), eps=1e-6), cos_4d, sin_4d)
        k_rot = apply_native_leaderboard_rotary(F.rms_norm(k_raw, (self.head_real_dim,), eps=1e-6), cos_4d, sin_4d)
        if seq_len > 1:
            d_half = self.head_real_dim // 2
            k_shifted = torch.cat([k_rot[:, :1, :, d_half:], k_rot[:, :-1, :, d_half:]], dim=1)
            k_rot = torch.cat([k_rot[..., :d_half], k_shifted], dim=-1)

        q_c = q_rot.view(bsz, seq_len, self.num_heads, 2, self.d_h).permute(0, 2, 1, 3, 4)
        v_c = v_mixed.view(bsz, seq_len, self.num_heads, 2, self.d_h).permute(0, 2, 1, 3, 4)
        q_re, q_im = q_c[..., 0, :], q_c[..., 1, :]
        v_re, v_im = complex_l2_normalize(v_c[..., 0, :], v_c[..., 1, :])

        if self.enable_density_highway and rho_packed is not None:
            v_outer_packed = pack_rank1_ket_outer_isometric(v_re, v_im)
            a_exc = self.val_exc_alpha.to(work_dtype).view(1, self.num_heads, 1, 1)
            rho_in = (1.0 - a_exc) * rho_packed + a_exc * v_outer_packed
            payload_288 = torch.cat([v_re, v_im, rho_in], dim=-1).contiguous()
            attn_288 = _FusedHermitianBornRhoKetAttnFn.apply(
                q_rot, k_rot, payload_288, None, self.berry_qk_phase, self.born_qk_beta, (float(self.sdpa_scale), True)
            )
            v_bar_re, v_bar_im = attn_288[..., : self.d_h], attn_288[..., self.d_h : 2 * self.d_h]
            rho_attn_packed = attn_288[..., 2 * self.d_h :]

            p_nc = self._compute_noclick_sink_prob(q_re, q_im, v_bar_re, v_bar_im)
            v_bar_re, v_bar_im = (1.0 - p_nc) * v_bar_re, (1.0 - p_nc) * v_bar_im
            nc_damp = float(DEFAULT_NOCLICK_RHO_DAMPING) * p_nc
            rho_damped = (1.0 - nc_damp) * rho_attn_packed + nc_damp * rho_packed

            rho_rot_packed, rho_rot_re, rho_rot_im = conjugate_packed_hermitian_by_unitary(
                rho_damped, self.u_evol_re, self.u_evol_im
            )
            rv_re, rv_im = _complex_hermitian_mv(rho_rot_re, rho_rot_im, v_bar_re, v_bar_im)
            k_rho = self.rho_ket_coupling.to(work_dtype).view(1, self.num_heads, 1, 1)
            v_out_re, v_out_im = v_bar_re + k_rho * rv_re, v_bar_im + k_rho * rv_im

            g_hw = self.highway_gamma.to(work_dtype).view(1, self.num_heads, 1, 1)
            rho_next = (1.0 - g_hw) * rho_packed + g_hw * rho_rot_packed
        else:
            payload_32 = torch.cat([v_re, v_im], dim=-1).contiguous()
            v_bar = _FusedHermitianBornRhoKetAttnFn.apply(
                q_rot, k_rot, payload_32, None, self.berry_qk_phase, self.born_qk_beta, (float(self.sdpa_scale), True)
            )
            v_bar_re, v_bar_im = v_bar[..., : self.d_h], v_bar[..., self.d_h :]
            p_nc = self._compute_noclick_sink_prob(q_re, q_im, v_bar_re, v_bar_im)
            v_out_re, v_out_im = (1.0 - p_nc) * v_bar_re, (1.0 - p_nc) * v_bar_im
            rho_next = None

        attn_heads = torch.stack([v_out_re, v_out_im], dim=-2).permute(0, 2, 1, 3, 4).reshape(bsz, seq_len, self.num_heads, self.head_real_dim)
        g_at = (2.0 * torch.sigmoid(F.linear(h_in[..., : self.gate_in_ch], self.q_attn_gate.weight.to(work_dtype)))).unsqueeze(-1)
        x_mid = x_in + self.obs_scale.to(work_dtype) * self.obs_proj((attn_heads * g_at).reshape(bsz, seq_len, d_model))
        x_out = self.mz_mlp(x_mid)

        if self.enable_density_highway and rho_next is not None:
            psi_mlp = x_out.view(bsz, seq_len, self.num_heads, 2, self.d_h).permute(0, 2, 1, 3, 4)
            pm_re, pm_im = complex_l2_normalize(psi_mlp[..., 0, :], psi_mlp[..., 1, :])
            mlp_outer = pack_rank1_ket_outer_isometric(pm_re, pm_im)
            inj = self.mlp_rho_inject.to(work_dtype).view(1, self.num_heads, 1, 1)
            rho_next = (1.0 - inj) * rho_next + inj * mlp_outer
            self._last_purity_t = rho_next.detach().float().square().sum(dim=-1).mean()

        return x_out, rho_next


class _SingleGemm512DTomographicCEFn(torch.autograd.Function):
    r"""Rule-9 Single-Pass Chunked `512D` Tomographic Cross-Entropy Function (`1` Fwd GEMM, `2` Bwd GEMMs)."""

    @staticmethod
    def forward(
        ctx: Any, h_512: torch.Tensor, w_vocab: torch.Tensor, b_uni: torch.Tensor,
        reg_hi: torch.Tensor | None, reg_lo: torch.Tensor | None, targets: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, d_dim = h_512.shape
        n_pos = bsz * seq_len
        vocab_size = w_vocab.size(0)
        n_valid = min(RAW_GPT2_VOCAB_SIZE, vocab_size)

        h_flat = h_512.detach().reshape(n_pos, d_dim)
        w_det = w_vocab.detach()
        w_t = w_det.t().contiguous()
        b_row = b_uni.detach().float().unsqueeze(0)
        y_flat = targets.reshape(n_pos)

        has_reg = reg_hi is not None and reg_lo is not None
        if has_reg:
            assert reg_hi is not None and reg_lo is not None
            k_rank = reg_hi.size(2)
            r_hi_flat = reg_hi.detach().reshape(n_pos, k_rank, DEFAULT_REGISTER_BUCKETS)
            r_lo_flat = reg_lo.detach().reshape(n_pos, k_rank, DEFAULT_REGISTER_BUCKETS)
        else:
            r_hi_flat, r_lo_flat = None, None

        cap = float(DEFAULT_SOFTCAP_LOGIT)
        inv_cap = 1.0 / cap
        n_cells = DEFAULT_REGISTER_BUCKETS * DEFAULT_REGISTER_BUCKETS
        total_nll = torch.zeros((), device=h_512.device, dtype=torch.float32)

        for start in range(0, n_pos, TOMO_CE_TOKEN_CHUNK):
            end = min(start + TOMO_CE_TOKEN_CHUNK, n_pos)
            c_len = end - start
            y_c = y_flat[start:end].clamp(0, n_valid - 1)

            z_c = torch.matmul(h_flat[start:end], w_t).float() + b_row
            if has_reg and r_hi_flat is not None and r_lo_flat is not None:
                grid_c = torch.bmm(r_hi_flat[start:end].transpose(1, 2), r_lo_flat[start:end]).float()
                z_c.add_(grid_c.reshape(c_len, n_cells)[:, :vocab_size])

            z_c.mul_(inv_cap).tanh_().mul_(cap)
            if vocab_size > n_valid:
                z_c[:, n_valid:] = -1e9
            log_z = torch.logsumexp(z_c, dim=-1)
            target_z = z_c.gather(dim=-1, index=y_c.unsqueeze(-1)).squeeze(-1)
            total_nll = total_nll + (log_z - target_z).sum()

        empty_t = torch.empty(0, device=h_512.device)
        ctx.save_for_backward(
            h_512.detach(), w_det, b_uni.detach(),
            reg_hi.detach() if reg_hi is not None else empty_t,
            reg_lo.detach() if reg_lo is not None else empty_t,
            targets,
        )
        ctx.has_reg = has_reg
        return total_nll / float(max(1, n_pos))

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, None]:
        h_512, w_det, b_uni, reg_hi_saved, reg_lo_saved, targets = ctx.saved_tensors
        has_reg: bool = ctx.has_reg
        bsz, seq_len, d_dim = h_512.shape
        n_pos = bsz * seq_len
        vocab_size = w_det.size(0)
        n_valid = min(RAW_GPT2_VOCAB_SIZE, vocab_size)
        work_dtype = h_512.dtype

        h_flat = h_512.reshape(n_pos, d_dim)
        w_t = w_det.t().contiguous()
        b_row = b_uni.float().unsqueeze(0)
        y_flat = targets.reshape(n_pos)

        grad_h_flat = torch.empty_like(h_flat)
        grad_w_f32 = torch.zeros((vocab_size, d_dim), device=h_512.device, dtype=torch.float32)
        grad_b_f32 = torch.zeros((vocab_size,), device=h_512.device, dtype=torch.float32)

        if has_reg:
            k_rank = reg_hi_saved.size(2)
            r_hi_flat = reg_hi_saved.reshape(n_pos, k_rank, DEFAULT_REGISTER_BUCKETS)
            r_lo_flat = reg_lo_saved.reshape(n_pos, k_rank, DEFAULT_REGISTER_BUCKETS)
            grad_r_hi, grad_r_lo = torch.empty_like(r_hi_flat), torch.empty_like(r_lo_flat)
        else:
            r_hi_flat, r_lo_flat, grad_r_hi, grad_r_lo = None, None, None, None

        cap = float(DEFAULT_SOFTCAP_LOGIT)
        inv_cap = 1.0 / cap
        n_cells = DEFAULT_REGISTER_BUCKETS * DEFAULT_REGISTER_BUCKETS
        scale = grad_out.float() / float(max(1, n_pos))

        for start in range(0, n_pos, TOMO_CE_TOKEN_CHUNK):
            end = min(start + TOMO_CE_TOKEN_CHUNK, n_pos)
            c_len = end - start
            h_c = h_flat[start:end]
            y_c = y_flat[start:end].clamp(0, n_valid - 1)

            z_raw = torch.matmul(h_c, w_t).float() + b_row
            if has_reg and r_hi_flat is not None and r_lo_flat is not None:
                grid_c = torch.bmm(r_hi_flat[start:end].transpose(1, 2), r_lo_flat[start:end]).float()
                z_raw.add_(grid_c.reshape(c_len, n_cells)[:, :vocab_size])

            tanh_u = torch.tanh(z_raw * inv_cap)
            z_cap = tanh_u * cap
            if vocab_size > n_valid:
                z_cap[:, n_valid:] = -1e9
            probs = F.softmax(z_cap, dim=-1)
            probs.scatter_add_(1, y_c.unsqueeze(1), torch.full((c_len, 1), -1.0, device=h_512.device, dtype=torch.float32))
            if vocab_size > n_valid:
                probs[:, n_valid:] = 0.0

            d_z = probs * (1.0 - tanh_u * tanh_u) * scale
            grad_b_f32.add_(d_z.sum(dim=0))
            d_z_w = d_z.to(work_dtype)
            grad_h_flat[start:end] = torch.matmul(d_z_w, w_det)
            grad_w_f32.add_(torch.matmul(d_z_w.t(), h_c).float())

            if has_reg and r_hi_flat is not None and r_lo_flat is not None and grad_r_hi is not None and grad_r_lo is not None:
                d_grid = torch.zeros((c_len, n_cells), device=h_512.device, dtype=work_dtype)
                d_grid[:, :vocab_size] = d_z_w
                d_grid_3d = d_grid.view(c_len, DEFAULT_REGISTER_BUCKETS, DEFAULT_REGISTER_BUCKETS)
                grad_r_hi[start:end] = torch.bmm(r_lo_flat[start:end], d_grid_3d.transpose(1, 2))
                grad_r_lo[start:end] = torch.bmm(r_hi_flat[start:end], d_grid_3d)

        return (
            grad_h_flat.view(bsz, seq_len, d_dim), grad_w_f32.to(w_det.dtype), grad_b_f32.to(b_uni.dtype),
            grad_r_hi.view_as(reg_hi_saved) if grad_r_hi is not None else None,
            grad_r_lo.view_as(reg_lo_saved) if grad_r_lo is not None else None, None,
        )


class ExplicitDensityTomographicGPT(nn.Module):
    r"""Flat, Consolidated $H = 16, \mathbb{C}^{16 \times 16}$ Explicit Outer-Product Density-Matrix & `512D` Tomographic Readout GPT (`Wave 63`)."""

    def __init__(self, spec: ExplicitDensityTomographicSpec) -> None:
        super().__init__()
        self.spec = spec
        self.variant_mode = ExplicitDensityVariantMode(spec.variant_mode)
        self.vocab_size = int(DEFAULT_SHALLOW_VOCAB_SIZE)
        self.embed_dim = int(DEFAULT_TOMO_EMBED_DIM)
        self.num_heads = int(DEFAULT_TOMO_NUM_HEADS)
        self.d_h = int(DEFAULT_TOMO_HEAD_COMPLEX_DIM)
        self.head_real_dim = 2 * self.d_h
        self.packed_rho_dim = int(DEFAULT_TOMO_PACKED_RHO_DIM)
        self.probes_per_head = int(DEFAULT_PROBES_PER_HEAD)
        self.num_layers = int(spec.num_layers)
        self.d_vac = int(spec.vacuum_dim) if spec.enable_noclick_and_vacuum else 0

        # 1. Power-of-2 (50304, 512) Input Wavepacket, Output Detector, & (4096, 512) Zero-Init Causal Context Tables
        self.wte_512 = nn.Parameter(torch.empty(self.vocab_size, self.embed_dim, dtype=torch.float32))
        self.w_det_512 = nn.Parameter(torch.empty(self.vocab_size, self.embed_dim, dtype=torch.float32))
        self.unigram_bias = nn.Parameter(torch.zeros(self.vocab_size, dtype=torch.float32))
        self.radial_scale = nn.Parameter(torch.ones(self.vocab_size, dtype=torch.float32))
        self.prior_gate = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.online_zipf = OnlineCausalZipfTracker(self.vocab_size, enable_online_ema=True)
        self.tok_ngram_in = nn.Embedding(DEFAULT_TOK_NGRAM_BUCKETS, self.embed_dim)
        self.tok_ngram_out = nn.Embedding(DEFAULT_TOK_NGRAM_BUCKETS, self.embed_dim)
        self.tok_unigram_out = nn.Embedding(DEFAULT_TOK_NGRAM_BUCKETS, self.embed_dim)
        nn.init.zeros_(self.tok_ngram_in.weight)
        nn.init.zeros_(self.tok_ngram_out.weight)
        nn.init.zeros_(self.tok_unigram_out.weight)
        self.ngram_out_scale = nn.Parameter(torch.tensor(0.85, dtype=torch.float32))

        # 2. Layer-0 Causal Wavepacket Dispersion & Explicit Outer-Product Density Seeding
        self.lag_phase = nn.Parameter(torch.linspace(-0.35, 0.35, self.num_heads, dtype=torch.float32))
        self.layer0_mix_logits = nn.Parameter(torch.tensor([1.10, 0.20, -0.10], dtype=torch.float32))
        self.smear_gate = nn.Linear(16, self.embed_dim, bias=False)
        self.smear_lambda = nn.Parameter(torch.tensor(0.22, dtype=torch.float32))
        nn.init.zeros_(self.smear_gate.weight)

        # 3. 4x Quantum Density Highway Blocks
        self.rope = NativeLeaderboardRotary(self.head_real_dim, max_seq_len=2048)
        self.quantum_blocks = nn.ModuleList([
            _ExplicitDensityHighwayBlock(
                embed_dim=self.embed_dim, num_heads=self.num_heads, head_complex_dim=self.d_h,
                mlp_hidden=DEFAULT_TOMO_MLP_HIDDEN, enable_density_highway=spec.enable_density_highway,
                enable_noclick_sink=spec.enable_noclick_and_vacuum,
            )
            for _ in range(self.num_layers)
        ])

        # 4. Stiefel Exit Bridge + 16 x 32 = 512D Quantum State Tomography Probes + Unified 512D Vacuum Sphere
        self.exit_bridge = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)
        self.tomo_proj = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)

        gen = torch.Generator()
        gen.manual_seed(int(spec.seed) + 17)
        p_raw = torch.randn(self.num_heads, self.probes_per_head, 2, self.d_h, generator=gen, dtype=torch.float32)
        p_re, p_im = complex_l2_normalize(p_raw[..., 0, :], p_raw[..., 1, :])
        self.probe_re = nn.Parameter(p_re.contiguous())
        self.probe_im = nn.Parameter(p_im.contiguous())

        self.tomo_scale = nn.Parameter(torch.tensor(DEFAULT_INIT_TOMO_SCALE, dtype=torch.float32))
        self.filter_scale = nn.Parameter(torch.tensor(DEFAULT_INIT_FILTER_SCALE, dtype=torch.float32))
        self.readout_temp = nn.Parameter(torch.tensor(DEFAULT_INIT_READOUT_TEMP, dtype=torch.float32))
        self.vac_rest_mass = nn.Parameter(torch.tensor(DEFAULT_INIT_VACUUM_REST_MASS, dtype=torch.float32))
        self.vac_proj = nn.Linear(16, max(1, self.d_vac), bias=False)
        nn.init.zeros_(self.vac_proj.weight)

        # 5. Optional T3 Zero-Synthesis Power-of-2 (256 x 256) Two-Register Bilinear Density Trace
        if spec.enable_two_register_bilinear:
            scale_dh = 1.0 / math.sqrt(float(self.d_h))
            self.in_reg_hi = nn.Parameter(torch.randn(DEFAULT_REGISTER_BUCKETS, self.embed_dim, generator=gen, dtype=torch.float32) * 0.02)
            self.in_reg_lo = nn.Parameter(torch.randn(DEFAULT_REGISTER_BUCKETS, self.embed_dim, generator=gen, dtype=torch.float32) * 0.02)
            self.phi_hi_re = nn.Parameter(torch.randn(self.num_heads, DEFAULT_REGISTER_BUCKETS, self.d_h, generator=gen, dtype=torch.float32) * scale_dh)
            self.phi_hi_im = nn.Parameter(torch.randn(self.num_heads, DEFAULT_REGISTER_BUCKETS, self.d_h, generator=gen, dtype=torch.float32) * scale_dh)
            self.phi_lo_re = nn.Parameter(torch.randn(self.num_heads, DEFAULT_REGISTER_BUCKETS, self.d_h, generator=gen, dtype=torch.float32) * scale_dh)
            self.phi_lo_im = nn.Parameter(torch.randn(self.num_heads, DEFAULT_REGISTER_BUCKETS, self.d_h, generator=gen, dtype=torch.float32) * scale_dh)
            self.u_hi_readout = nn.Parameter(torch.zeros(DEFAULT_REGISTER_BUCKETS, self.embed_dim, dtype=torch.float32))
            self.bilinear_scale = nn.Parameter(torch.tensor(DEFAULT_INIT_BILINEAR_REGISTER_SCALE, dtype=torch.float32))
        else:
            for name in ("in_reg_hi", "in_reg_lo", "phi_hi_re", "phi_hi_im", "phi_lo_re", "phi_lo_im", "u_hi_readout", "bilinear_scale"):
                self.register_parameter(name, None)

        self.init_tokens_scanned: int = 0
        self._last_cos_vac_t: torch.Tensor | None = None
        self._last_rho_rank_t: torch.Tensor | None = None
        self._last_probe_ent_t: torch.Tensor | None = None

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Return all 2D real Stiefel matrices in the trunk and readout bridges."""
        params: list[nn.Parameter] = []
        for q_blk in self.quantum_blocks:
            params.extend(q_blk.real_stiefel_parameters())
        params.extend(self.exit_bridge.real_stiefel_parameters())
        params.extend(self.tomo_proj.real_stiefel_parameters())
        return params

    def complex_unitary_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        """Return all paired complex $U(16)$ matrices across the 4 highway blocks."""
        pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
        for q_blk in self.quantum_blocks:
            pairs.extend(q_blk.complex_unitary_pairs())
        return pairs

    def scalar_and_probe_parameters(self) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
        """Return `(scalar_params, fast_temp_params, register_table_params)` for `AdamW`."""
        scalar_params: list[nn.Parameter] = [
            self.unigram_bias, self.radial_scale, self.prior_gate, self.lag_phase, self.layer0_mix_logits,
            self.smear_gate.weight, self.smear_lambda, self.probe_re, self.probe_im, self.vac_rest_mass,
            self.vac_proj.weight, self.ngram_out_scale,
        ]
        scalar_params.extend(self.exit_bridge.scalar_parameters())
        scalar_params.extend(self.tomo_proj.scalar_parameters())
        for q_blk in self.quantum_blocks:
            scalar_params.extend(q_blk.scalar_parameters())

        fast_temp_params: list[nn.Parameter] = [self.readout_temp, self.tomo_scale, self.filter_scale]
        reg_tables: list[nn.Parameter] = [
            self.tok_ngram_in.weight, self.tok_ngram_out.weight, self.tok_unigram_out.weight,
        ]
        if self.spec.enable_two_register_bilinear:
            assert self.in_reg_hi is not None and self.in_reg_lo is not None
            assert self.phi_hi_re is not None and self.phi_hi_im is not None
            assert self.phi_lo_re is not None and self.phi_lo_im is not None
            assert self.u_hi_readout is not None and self.bilinear_scale is not None
            reg_tables.extend([
                self.in_reg_hi, self.in_reg_lo, self.phi_hi_re, self.phi_hi_im,
                self.phi_lo_re, self.phi_lo_im, self.u_hi_readout,
            ])
            fast_temp_params.append(self.bilinear_scale)
        return scalar_params, fast_temp_params, reg_tables

    @torch.no_grad()
    def restrict_scalars_(self) -> None:
        """Clamp model temperatures, vacuum rest mass, and block parameters to physical bounds."""
        for q_blk in self.quantum_blocks:
            q_blk.restrict_scalars_()
        self.exit_bridge.clamp_sigma_()
        self.tomo_proj.clamp_sigma_()
        self.readout_temp.clamp_(0.05, 40.0)
        self.tomo_scale.clamp_(-2.0, 2.5)
        self.filter_scale.clamp_(-2.0, 2.5)
        self.ngram_out_scale.clamp_(0.05, 2.5)
        self.vac_rest_mass.clamp_(0.02, 0.85)
        self.smear_lambda.clamp_(-1.0, 1.0)
        self.radial_scale.clamp_(0.25, 2.20)
        p_re, p_im = complex_l2_normalize(self.probe_re.data.float(), self.probe_im.data.float())
        self.probe_re.data.copy_(p_re)
        self.probe_im.data.copy_(p_im)
        if self.bilinear_scale is not None:
            self.bilinear_scale.clamp_(-1.5, 1.5)

    def _compute_two_register_bilinear_factors(
        self, u_rms_512: torch.Tensor, psi_exit_re: torch.Tensor, psi_exit_im: torch.Tensor,
        rho_re: torch.Tensor | None, rho_im: torch.Tensor | None, log_p_hi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute `(reg_hi, reg_lo)` of shape `(B, T, 17, 256)` for the zero-synthesis $256 \times 256$ trace (`<= 8` args)."""
        assert self.phi_hi_re is not None and self.phi_hi_im is not None
        assert self.phi_lo_re is not None and self.phi_lo_im is not None
        assert self.u_hi_readout is not None and self.bilinear_scale is not None

        bsz, seq_len, _ = u_rms_512.shape
        work_dtype = u_rms_512.dtype
        if rho_re is not None and rho_im is not None:
            bra_re, bra_im = _complex_hermitian_mv(rho_re, rho_im, psi_exit_re, psi_exit_im)
        else:
            bra_re, bra_im = psi_exit_re, psi_exit_im

        phi_hi_r, phi_hi_i = self.phi_hi_re.to(work_dtype).transpose(-1, -2).unsqueeze(0), self.phi_hi_im.to(work_dtype).transpose(-1, -2).unsqueeze(0)
        phi_lo_r, phi_lo_i = self.phi_lo_re.to(work_dtype).transpose(-1, -2).unsqueeze(0), self.phi_lo_im.to(work_dtype).transpose(-1, -2).unsqueeze(0)

        a_h = torch.matmul(bra_re, phi_hi_r) + torch.matmul(bra_im, phi_hi_i)
        b_h = torch.matmul(psi_exit_re, phi_lo_r) + torch.matmul(psi_exit_im, phi_lo_i)

        a_bt = (torch.tanh(a_h * 0.5) * self.bilinear_scale.to(work_dtype)).transpose(1, 2)
        b_bt = torch.tanh(b_h * 0.5).transpose(1, 2)

        # Add rank-1 coarse high-byte logit + online high-byte prior shift in slot 16
        inv_sqrt_d = 1.0 / math.sqrt(float(self.embed_dim))
        u_hi_score = F.linear(u_rms_512, self.u_hi_readout.to(work_dtype)) * inv_sqrt_d
        hi_prior_pad = torch.zeros((DEFAULT_REGISTER_BUCKETS,), device=u_rms_512.device, dtype=work_dtype)
        n_hi = min(DEFAULT_REGISTER_BUCKETS, log_p_hi.numel())
        hi_prior_pad[:n_hi] = (log_p_hi[:n_hi] - log_p_hi[:n_hi].mean()).to(work_dtype) * 0.12
        a_extra = (u_hi_score + hi_prior_pad.view(1, 1, DEFAULT_REGISTER_BUCKETS)).unsqueeze(2)
        b_ones = torch.ones((bsz, seq_len, 1, DEFAULT_REGISTER_BUCKETS), device=u_rms_512.device, dtype=work_dtype)

        return torch.cat([a_bt, a_extra], dim=2).contiguous(), torch.cat([b_bt, b_ones], dim=2).contiguous()

    def build_readout_tensors(
        self, idx: torch.Tensor, include_vocab_params: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        r"""Encode `idx` `(B, T)` through the 4-layer $\mathbb{C}^{16 \times 16}$ highway and `512D` tomographic readout (`<= 8` args)."""
        if self.training:
            self.online_zipf.update_from_training_batch_(idx)
            log_p_tok_cur, _ = self.online_zipf.current_log_priors()
            with torch.no_grad():
                n_seen = int(self.online_zipf.batches_seen.item())
                self.unigram_bias.lerp_(log_p_tok_cur, 0.25 / (1.0 + 0.05 * float(n_seen)))

        bsz, seq_len = idx.shape
        work_dtype = torch.bfloat16 if (idx.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
        idx_l = idx.to(torch.long)
        idx_p1 = torch.cat([idx_l[:, :1], idx_l[:, :-1]], dim=1) if seq_len > 1 else idx_l
        idx_p2 = torch.cat([idx_l[:, :2], idx_l[:, :-2]], dim=1) if seq_len > 2 else idx_p1
        h_tok = (idx_l * 1000003 + idx_p1 * 65537 + idx_p2 * 257) & (DEFAULT_TOK_NGRAM_BUCKETS - 1)
        h_uni = (idx_l * 2654435761) & (DEFAULT_TOK_NGRAM_BUCKETS - 1)

        raw_emb = F.embedding(idx, self.wte_512).to(work_dtype)
        ng_in = self.tok_ngram_in(h_tok).float()
        raw_emb = raw_emb + 0.30 * (ng_in * torch.rsqrt(ng_in.square().mean(dim=-1, keepdim=True) + 0.0625)).to(work_dtype)
        if self.spec.enable_two_register_bilinear and self.in_reg_hi is not None and self.in_reg_lo is not None:
            b_hi = (idx >> 8).clamp(0, DEFAULT_REGISTER_BUCKETS - 1)
            b_lo = idx & 0xFF
            raw_emb = raw_emb + 0.20 * F.embedding(b_hi, self.in_reg_hi.to(work_dtype)) + 0.15 * F.embedding(b_lo, self.in_reg_lo.to(work_dtype))

        if seq_len > 1:
            s_gate = self.smear_lambda.to(work_dtype) * torch.sigmoid(F.linear(raw_emb[:, 1:, :16], self.smear_gate.weight.to(work_dtype)))
            raw_emb = torch.cat([raw_emb[:, :1, :], raw_emb[:, 1:, :] + s_gate * raw_emb[:, :-1, :]], dim=1)

        x_0 = F.rms_norm(raw_emb, (self.embed_dim,), eps=1e-6)
        psi_0 = x_0.view(bsz, seq_len, self.num_heads, 2, self.d_h).permute(0, 2, 1, 3, 4)
        psi0_re, psi0_im = complex_l2_normalize(psi_0[..., 0, :], psi_0[..., 1, :])

        rho_packed = (
            build_layer0_explicit_outer_density(psi0_re, psi0_im, self.lag_phase, self.layer0_mix_logits)
            if self.spec.enable_density_highway else None
        )

        x = x_0
        cos_4d, sin_4d = self.rope.get_cos_sin(seq_len, dtype=work_dtype)
        for q_blk in self.quantum_blocks:
            x, rho_packed = q_blk(x, x_0, cos_4d, sin_4d, rho_packed=rho_packed)

        h_pre = F.rms_norm(x, (self.embed_dim,), eps=1e-6)
        h_exit = F.rms_norm(self.exit_bridge(h_pre), (self.embed_dim,), eps=1e-6)
        ng_out = self.tok_ngram_out(h_tok).float() + 0.65 * self.tok_unigram_out(h_uni).float()
        ng_out_norm = (ng_out * torch.rsqrt(ng_out.square().mean(dim=-1, keepdim=True) + 0.0625)).to(work_dtype)

        psi_exit = h_exit.view(bsz, seq_len, self.num_heads, 2, self.d_h).permute(0, 2, 1, 3, 4)
        pe_re, pe_im = complex_l2_normalize(psi_exit[..., 0, :], psi_exit[..., 1, :])

        rho_re_final, rho_im_final = None, None
        if self.spec.enable_density_highway and rho_packed is not None:
            rho_re_final, rho_im_final = unpack_hermitian_isometric(rho_packed, self.d_h)
            filt_re, filt_im = _complex_hermitian_mv(rho_re_final, rho_im_final, pe_re, pe_im)
            h_filt = torch.stack([filt_re, filt_im], dim=-2).permute(0, 2, 1, 3, 4).reshape(bsz, seq_len, self.embed_dim)
            h_filt_norm = F.rms_norm(h_filt, (self.embed_dim,), eps=1e-6)

            if self.spec.enable_tomographic_readout:
                q_512 = compute_tomographic_512d_probe_expectations(rho_packed, self.probe_re, self.probe_im)
                q_proj = F.rms_norm(self.tomo_proj(q_512), (self.embed_dim,), eps=1e-6)
                ng_mod = (1.0 + 0.30 * torch.tanh(q_proj)) * ng_out_norm
                u_vocab = (
                    h_exit + self.filter_scale.to(work_dtype) * h_filt_norm
                    + self.tomo_scale.to(work_dtype) * q_proj + self.ngram_out_scale.to(work_dtype) * ng_mod
                )
                with torch.no_grad():
                    q_pos = q_512.detach().float().clamp_min(1e-6)
                    q_prob = q_pos / q_pos.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                    self._last_probe_ent_t = -(q_prob * torch.log(q_prob.clamp_min(1e-8))).sum(dim=-1).mean()
            else:
                u_vocab = h_exit + self.filter_scale.to(work_dtype) * h_filt_norm + self.ngram_out_scale.to(work_dtype) * ng_out_norm
                self._last_probe_ent_t = None

            with torch.no_grad():
                tr_rho = torch.diagonal(rho_re_final.detach().float(), dim1=-2, dim2=-1).sum(dim=-1).clamp_min(1e-5)
                tr_rho_sq = rho_packed.detach().float().square().sum(dim=-1).clamp_min(1e-6)
                purity_norm = (tr_rho_sq / tr_rho.square()).clamp(1.0 / float(self.d_h), 1.0)
                self._last_rho_rank_t = (1.0 / purity_norm).mean()
        else:
            u_vocab = h_exit + self.ngram_out_scale.to(work_dtype) * ng_out_norm
            self._last_rho_rank_t = None
            self._last_probe_ent_t = None

        u_vac = F.linear(h_pre[:, :, :16], self.vac_proj.weight.to(work_dtype)) if (self.spec.enable_noclick_and_vacuum and self.d_vac > 0) else None
        h_norm_512, cos_theta_t = apply_unified_512d_vacuum_sphere_norm(
            u_vocab, u_vac, self.vac_rest_mass, self.readout_temp, enable_vacuum=self.spec.enable_noclick_and_vacuum
        )
        self._last_cos_vac_t = cos_theta_t

        log_p_online, log_p_hi = self.online_zipf.current_log_priors()
        if include_vocab_params:
            w_vocab = (F.rms_norm(self.w_det_512.to(work_dtype), (self.embed_dim,), eps=1e-6) * self.radial_scale.to(work_dtype).unsqueeze(1)).contiguous()
            b_uni = (0.50 * self.unigram_bias + 0.50 * log_p_online) * self.prior_gate
        else:
            w_vocab = self.w_det_512[:0]
            b_uni = self.unigram_bias[:0]

        reg_hi, reg_lo = (
            self._compute_two_register_bilinear_factors(
                F.rms_norm(u_vocab, (self.embed_dim,), eps=1e-6), pe_re, pe_im, rho_re_final, rho_im_final, log_p_hi
            )
            if self.spec.enable_two_register_bilinear else (None, None)
        )
        return h_norm_512, w_vocab, b_uni, reg_hi, reg_lo

    def forward_loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Single-GEMM `512D` Tomographic Cross-Entropy loss over `50,257` valid GPT-2 tokens."""
        h_norm_512, w_vocab, b_uni, reg_hi, reg_lo = self.build_readout_tensors(idx, include_vocab_params=True)
        return _SingleGemm512DTomographicCEFn.apply(h_norm_512, w_vocab, b_uni, reg_hi, reg_lo, targets)

    def get_noclick_and_density_telemetry(self) -> dict[str, float]:
        """Return trunk No-Click POVM sink probability, density effective rank, and vacuum angle telemetry."""
        nc_vals = [float(blk._last_noclick_t.item()) for blk in self.quantum_blocks if blk._last_noclick_t is not None]
        mean_nc = float(sum(nc_vals) / len(nc_vals)) if nc_vals else 0.0
        rho_rank = float(self._last_rho_rank_t.item()) if self._last_rho_rank_t is not None else 1.0
        cos_vac = float(self._last_cos_vac_t.item()) if self._last_cos_vac_t is not None else 1.0
        probe_ent = float(self._last_probe_ent_t.item()) if self._last_probe_ent_t is not None else 0.0
        return {
            "mean_noclick_prob": round(mean_nc, 4),
            "rho_effective_rank": round(rho_rank, 3),
            "cos_theta_vac": round(cos_vac, 4),
            "probe_entropy_nats": round(probe_ent, 4),
        }


def build_explicit_density_tomographic_optimizers(
    model: ExplicitDensityTomographicGPT,
) -> tuple[
    ModularRiemannianStiefelKrausMuon, BatchSubspaceTikhonovGramMuon,
    BatchSubspaceTikhonovGramMuon, RiemannianGeodesicScalarAndProbeOptimizer,
]:
    """Build 100% `AdamW`-free Riemannian Stiefel+$U(16)$, Vocab+N-Gram Gram-Muon, and Geodesic Scalar+Probe optimizers (`<= 8` args)."""
    scalar_params, fast_temp_params, reg_tables = model.scalar_and_probe_parameters()
    q_trunk_opt = ModularRiemannianStiefelKrausMuon(
        real_stiefel_params=model.real_stiefel_parameters(),
        complex_operator_pairs=model.complex_unitary_pairs(),
        stiefel_lr=DEFAULT_SHALLOW_STIEFEL_LR, unitary_lr=0.025,
        tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA,
        momentum_mode=StiefelMomentumMode.RIEMANNIAN_TANGENT_SNR_GATED,
    )
    reg_2d = [p for p in reg_tables if p.ndim == 2]
    wte_groups: list[dict[str, Any]] = [{"params": [model.wte_512], "lr": DEFAULT_SHALLOW_WTE_LR, "zero_init_ball": False}]
    if reg_2d:
        wte_groups.append({"params": reg_2d, "lr": DEFAULT_SHALLOW_NGRAM_LR, "zero_init_ball": True})
    wte_opt = BatchSubspaceTikhonovGramMuon(wte_groups, lr=DEFAULT_SHALLOW_WTE_LR, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA)
    head_opt = BatchSubspaceTikhonovGramMuon([model.w_det_512], lr=DEFAULT_SHALLOW_READOUT_LR, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA)

    complex_pairs = [model.probe_re, model.probe_im] + ([p for p in reg_tables if p.ndim == 3] if model.spec.enable_two_register_bilinear else [])
    vac_kets = [blk.k_vac_ket for blk in model.quantum_blocks]
    pos_ids = {id(model.vac_rest_mass), id(model.ngram_out_scale), id(model.prior_gate), id(model.exit_bridge.sigma), id(model.tomo_proj.sigma)}
    for blk in model.quantum_blocks:
        pos_ids.update({
            id(blk.val_exc_alpha), id(blk.highway_gamma), id(blk.mlp_rho_inject), id(blk.obs_scale),
            id(blk.q_proj.sigma), id(blk.k_proj.sigma), id(blk.v_proj.sigma), id(blk.obs_proj.sigma),
            id(blk.mz_mlp.up_proj.sigma), id(blk.mz_mlp.down_proj.sigma), id(blk.mz_mlp.mz_scale),
        })
    skip_ids = {id(p) for p in (*complex_pairs, *vac_kets, model.unigram_bias, model.radial_scale)}
    pos_scalars = [p for p in scalar_params if id(p) not in skip_ids and id(p) in pos_ids]
    signed_scalars = [p for p in scalar_params if id(p) not in skip_ids and id(p) not in pos_ids]
    fast_signed = [p for p in fast_temp_params if p is not model.readout_temp]

    geo_groups: list[dict[str, Any]] = [
        {"params": complex_pairs, "lr": 0.025, "kind": "complex_sphere_pair", "max_step": 0.20},
        {"params": vac_kets, "lr": 0.025, "kind": "real_sphere_ket", "max_step": 0.20},
        {"params": [model.readout_temp], "lr": 0.095, "kind": "positive_geodesic", "ref_scale": 4.0, "max_step": 0.22},
        {"params": fast_signed, "lr": 0.080, "kind": "bounded_tangent", "max_step": 0.14},
        {"params": [model.radial_scale], "lr": DEFAULT_SHALLOW_SCALAR_LR, "kind": "sparse_vocab_positive", "max_step": 0.08},
        {"params": [model.unigram_bias], "lr": DEFAULT_SHALLOW_SCALAR_LR, "kind": "sparse_vocab_bias", "max_step": 0.08},
        {"params": pos_scalars, "lr": DEFAULT_SHALLOW_SCALAR_LR, "kind": "positive_geodesic", "ref_scale": 1.0, "max_step": 0.12},
        {"params": signed_scalars, "lr": DEFAULT_SHALLOW_SCALAR_LR, "kind": "bounded_tangent", "max_step": 0.08},
    ]
    geo_opt = RiemannianGeodesicScalarAndProbeOptimizer(geo_groups, lr=DEFAULT_SHALLOW_SCALAR_LR, beta1=0.80, beta2=0.95)
    return q_trunk_opt, wte_opt, head_opt, geo_opt


@torch.no_grad()
def initialize_explicit_density_tomographic_model_(
    model: ExplicitDensityTomographicGPT,
) -> tuple[float, dict[str, Any]]:
    """Initialize `ExplicitDensityTomographicGPT` at Step 0 with **zero training tokens scanned** (`< 30 ms`)."""
    device = model.wte_512.device
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    gen = torch.Generator(device=device)
    gen.manual_seed(int(model.spec.seed))

    raw_wte = torch.randn((model.vocab_size, model.num_heads, 2, model.d_h), generator=gen, device=device, dtype=torch.float32)
    w_re, w_im = complex_l2_normalize(raw_wte[..., 0, :], raw_wte[..., 1, :])
    wte_norm = torch.stack([w_re, w_im], dim=-2).reshape(model.vocab_size, model.embed_dim)
    if model.spec.enable_two_register_bilinear and model.in_reg_hi is not None and model.in_reg_lo is not None:
        v_ids = torch.arange(model.vocab_size, device=device, dtype=torch.long)
        b_hi, b_lo = (v_ids >> 8).clamp(0, DEFAULT_REGISTER_BUCKETS - 1), v_ids & 0xFF
        wte_norm = F.normalize(
            0.75 * wte_norm + 0.15 * F.normalize(model.in_reg_hi[b_hi], dim=-1) + 0.10 * F.normalize(model.in_reg_lo[b_lo], dim=-1),
            dim=-1, eps=1e-6,
        )
    model.wte_512.copy_(wte_norm)

    log_p_tok, log_p_hi, radial = build_zero_data_bpe_rank_zipf_prior(model.vocab_size, device=device)
    model.unigram_bias.copy_(log_p_tok)
    model.radial_scale.copy_(radial)
    model.prior_gate.fill_(1.0)
    model.readout_temp.fill_(DEFAULT_INIT_READOUT_TEMP)

    p0 = torch.exp(log_p_tok).unsqueeze(1)
    centered_det = F.normalize(wte_norm - (p0 * wte_norm).sum(dim=0, keepdim=True), dim=-1, eps=1e-6)
    model.w_det_512.copy_(centered_det * math.sqrt(float(model.embed_dim)))

    eye_d = torch.eye(model.embed_dim, device=device, dtype=torch.float32)
    for br in (model.exit_bridge, model.tomo_proj):
        br.u_stiefel.copy_(eye_d)
        br.v_stiefel.copy_(eye_d)
        br.sigma.fill_(1.0)

    model.init_tokens_scanned = 0
    model.restrict_scalars_()

    if device.type == "cuda":
        torch.cuda.synchronize()
    init_ms = (time.perf_counter() - t0) * 1000.0
    return init_ms, {
        "init_tokens_scanned": 0, "init_ms": round(init_ms, 3), "num_heads": model.num_heads,
        "head_complex_dim": model.d_h, "packed_rho_dim": model.packed_rho_dim, "probes_per_head": model.probes_per_head,
        "step0_analytical_hi_entropy_nats": round(float(-(torch.exp(log_p_hi) * log_p_hi).sum().item()), 4),
    }


def _step_explicit_density_optimizers(
    model: ExplicitDensityTomographicGPT,
    optimizers: tuple[ModularRiemannianStiefelKrausMuon, BatchSubspaceTikhonovGramMuon, BatchSubspaceTikhonovGramMuon, RiemannianGeodesicScalarAndProbeOptimizer],
    idx_in: torch.Tensor, targets: torch.Tensor, lr_mult: float, mom_beta: float,
) -> None:
    """Execute one coordinated 100% `AdamW`-free Riemannian Stiefel+$U(16)$ + Vocab + Geodesic Scalar optimizer step (`<= 8` args)."""
    q_trunk_opt, wte_opt, head_opt, geo_opt = optimizers
    q_trunk_opt.set_momentum(mom_beta)
    q_trunk_opt.set_lr_scale(lr_mult)
    q_trunk_opt.step()

    active_ids = torch.unique(torch.cat([idx_in.reshape(-1), targets.reshape(-1)]))
    wte_opt.set_momentum(mom_beta)
    wte_opt.step_active(active_ids, lr_scale=lr_mult)
    head_opt.set_momentum(mom_beta)
    head_opt.step_full(lr_scale=lr_mult, active_idx=active_ids)

    geo_opt.set_momentum(0.0 if mom_beta <= 0.0 else 0.80)
    geo_opt.step_active(active_idx=active_ids, lr_scale=lr_mult)
    model.restrict_scalars_()


@torch.no_grad()
def evaluate_explicit_density_tomographic_model(
    model: ExplicitDensityTomographicGPT, stream: LongSeqFineWebStream
) -> dict[str, float]:
    """Evaluate `ExplicitDensityTomographicGPT` on `10,485,760` Shard-0 tokens with **zero `50,304 × 512` re-synthesis** (`<= 8` args)."""
    was_training = model.training
    model.eval()
    dev = stream.device
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_eval_start = time.perf_counter()

    total_nll_t = torch.zeros((), device=dev, dtype=torch.float64)
    total_top1_t = torch.zeros((), device=dev, dtype=torch.int64)
    total_top5_t = torch.zeros((), device=dev, dtype=torch.int64)
    total_tokens = 0
    cap = float(DEFAULT_SOFTCAP_LOGIT)
    inv_cap = 1.0 / cap
    n_cells = DEFAULT_REGISTER_BUCKETS * DEFAULT_REGISTER_BUCKETS

    # Hoist the (50257, 512) detector matrix and unigram bias ONCE outside the 1,280-batch validation loop!
    work_dtype = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    w_vocab_all = F.rms_norm(model.w_det_512.to(work_dtype), (model.embed_dim,), eps=1e-6) * model.radial_scale.to(work_dtype).unsqueeze(1)
    w_t_hoisted = w_vocab_all[:RAW_GPT2_VOCAB_SIZE].t().contiguous()
    log_p_online, _ = model.online_zipf.current_log_priors()
    b_uni_hoisted = ((0.50 * model.unigram_bias + 0.50 * log_p_online) * model.prior_gate)[:RAW_GPT2_VOCAB_SIZE].float().unsqueeze(0)

    for idx_in, targets in stream.val_batches:
        bsz, seq_len = idx_in.shape
        n_pos = bsz * seq_len
        h_norm_512, _, _, reg_hi, reg_lo = model.build_readout_tensors(idx_in, include_vocab_params=False)

        h_flat = h_norm_512.reshape(n_pos, model.embed_dim)
        if reg_hi is not None and reg_lo is not None:
            k_rank = reg_hi.size(2)
            r_hi_flat = reg_hi.reshape(n_pos, k_rank, DEFAULT_REGISTER_BUCKETS)
            r_lo_flat = reg_lo.reshape(n_pos, k_rank, DEFAULT_REGISTER_BUCKETS)
        else:
            r_hi_flat, r_lo_flat = None, None
        y_flat = targets.reshape(n_pos)

        for start in range(0, n_pos, TOMO_CE_TOKEN_CHUNK):
            end = min(start + TOMO_CE_TOKEN_CHUNK, n_pos)
            c_len = end - start
            y_c = y_flat[start:end]

            z_c = torch.matmul(h_flat[start:end], w_t_hoisted).float() + b_uni_hoisted
            if r_hi_flat is not None and r_lo_flat is not None:
                grid_c = torch.bmm(r_hi_flat[start:end].transpose(1, 2), r_lo_flat[start:end]).float()
                z_c.add_(grid_c.reshape(c_len, n_cells)[:, :RAW_GPT2_VOCAB_SIZE])
                del grid_c
            z_c.mul_(inv_cap).tanh_().mul_(cap)

            log_z = torch.logsumexp(z_c, dim=-1)
            target_z = z_c.gather(dim=-1, index=y_c.unsqueeze(-1)).squeeze(-1)
            total_nll_t += (log_z - target_z).sum().to(torch.float64)

            top5_idx = torch.topk(z_c, k=5, dim=-1).indices
            total_top1_t += (top5_idx[:, 0] == y_c).sum()
            total_top5_t += (top5_idx == y_c.unsqueeze(-1)).any(dim=-1).sum()
            total_tokens += c_len
            del z_c

        del h_norm_512, reg_hi, reg_lo

    if dev.type == "cuda":
        torch.cuda.synchronize()
    eval_wall_s = time.perf_counter() - t_eval_start
    if was_training:
        model.train()

    denom = float(max(1, total_tokens))
    val_ce = float(total_nll_t.item()) / denom
    telem = model.get_noclick_and_density_telemetry()
    ev = {
        "val_ce": round(val_ce, 4), "neural_val_ce": round(val_ce, 4), "val_ppl": round(math.exp(min(20.0, val_ce)), 2),
        "top1_acc": round(float(total_top1_t.item()) / denom, 4), "top5_acc": round(float(total_top5_t.item()) / denom, 4),
        "eval_wall_sec": round(eval_wall_s, 2), "mean_rho_eff_rank": telem["rho_effective_rank"],
        "mean_cos_theta_vac": telem["cos_theta_vac"], "mean_probe_entropy_nats": telem["probe_entropy_nats"],
        "online_batches_seen": float(model.online_zipf.batches_seen.item()),
    }
    ev.update(telem)
    return ev


def _train_and_eval_explicit_density_variants(
    stream: LongSeqFineWebStream, variants: list[tuple[str, ExplicitDensityTomographicSpec, str]],
    steps: int, eval_interval: int, wave_id: str = "W63",
) -> list[dict[str, Any]]:
    """Train and evaluate `ExplicitDensityTomographicGPT` variants (`T0..T4`, `<= 8` args)."""
    device = stream.device
    results: list[dict[str, Any]] = []
    tokens_per_step = stream.batch_size * stream.seq_len

    for variant_id, spec, desc in variants:
        torch.manual_seed(spec.seed)
        model = ExplicitDensityTomographicGPT(spec).to(device=device)
        init_ms, init_meta = initialize_explicit_density_tomographic_model_(model)

        run_key = f"plan083_{variant_id.lower()}_{steps}steps"
        arch_summary = f"ExplicitDensityTomographicGPT(L={spec.num_layers}, d=512, H=16xC16, rho={spec.enable_density_highway}, tomo={spec.enable_tomographic_readout}, vac={spec.enable_noclick_and_vacuum}, reg={spec.enable_two_register_bilinear})"
        intent = ExperimentIntentSpec(
            run_key=run_key, wave_id=wave_id, plan_id="EXPERIMENT_PLAN_083", task_domain="fineweb_speedrun",
            hypothesis_intent=desc, architecture_summary=arch_summary, hardware_budget="1x RTX 5060 Ti 16GB",
            batch_size=stream.batch_size, seq_len=stream.seq_len, total_steps=steps, target_tokens_M=round((steps * tokens_per_step) / 1e6, 3),
        )
        exp_logger = StandardizedExperimentLogger(intent)
        optimizers = build_explicit_density_tomographic_optimizers(model)

        step0_eval = evaluate_explicit_density_tomographic_model(model, stream)
        step0_eval["step"], step0_eval["tokens_M"] = 0, 0.0
        exp_logger.log_eval(0, {**step0_eval, "top1_acc": round(100.0 * step0_eval["top1_acc"], 2), "top5_acc": round(100.0 * step0_eval["top5_acc"], 2)})
        print(f"[{variant_id} (init={init_ms:.1f}ms)] Step    0 | Val CE: {step0_eval['val_ce']:.4f} (PPL {step0_eval['val_ppl']:.2f}) | Top1: {100.0 * step0_eval['top1_acc']:.2f}% | Eval: {step0_eval['eval_wall_sec']:.1f}s | RhoRank: {step0_eval['mean_rho_eff_rank']:.2f}/16", flush=True)

        curve: list[dict[str, Any]] = [step0_eval]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        model.train()

        for step in range(1, steps + 1):
            idx_in, targets = stream.sample_train_batch(step)
            lr_mult, mom_beta = _compute_zero_ppmi_step_lr_and_momentum(step, steps)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            loss = model.forward_loss(idx_in, targets)
            train_loss_val = float(loss.detach().item())
            loss.backward()
            del loss
            _step_explicit_density_optimizers(model, optimizers, idx_in, targets, lr_mult=lr_mult, mom_beta=mom_beta)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            if step % eval_interval == 0 or step == steps:
                ev = evaluate_explicit_density_tomographic_model(model, stream)
                ev["step"], ev["tokens_M"] = step, round((step * tokens_per_step) / 1e6, 3)
                ev["train_loss"], ev["lr_mult"] = round(train_loss_val, 4), round(lr_mult, 4)
                curve.append(ev)
                exp_logger.log_eval(step, {**ev, "top1_acc": round(100.0 * ev["top1_acc"], 2), "top5_acc": round(100.0 * ev["top5_acc"], 2)})
                print(f"[{variant_id}] Step {step:4d} ({ev['tokens_M']:5.2f}M tok, lr={lr_mult:.2f}x) | Train CE: {ev['train_loss']:.4f} | Val CE: {ev['val_ce']:.4f} (PPL {ev['val_ppl']:.2f}) | Top1: {100.0 * ev['top1_acc']:.2f}% | Top5: {100.0 * ev['top5_acc']:.2f}% | RhoRank: {ev['mean_rho_eff_rank']:.2f}/16 | p_nc: {ev['mean_noclick_prob']:.3f} | Eval: {ev['eval_wall_sec']:.1f}s", flush=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - t_start
        full_10m_ev = None
        if steps >= 1000 and stream.val_tokens.numel() < 10_485_760:
            full_stream = LongSeqFineWebStream(seq_len=stream.seq_len, batch_size=stream.batch_size, val_tokens=10_485_760, device=device)
            full_10m_ev = evaluate_explicit_density_tomographic_model(model, full_stream)
            print(f"[{variant_id} FULL 10.48M-TOK SHARD-0] Val CE: {full_10m_ev['val_ce']:.4f} (PPL {full_10m_ev['val_ppl']:.2f}) | Top1: {100.0 * full_10m_ev['top1_acc']:.2f}% | Eval: {full_10m_ev['eval_wall_sec']:.1f}s", flush=True)
            del full_stream
        results.append({
            "variant_id": variant_id, "variant_mode": spec.variant_mode.value, "description": desc,
            "enable_density_highway": spec.enable_density_highway, "enable_tomographic_readout": spec.enable_tomographic_readout,
            "enable_noclick_and_vacuum": spec.enable_noclick_and_vacuum, "enable_two_register_bilinear": spec.enable_two_register_bilinear,
            "init_tokens_scanned": init_meta["init_tokens_scanned"], "init_ms": round(init_ms, 2),
            "elapsed_sec": round(elapsed_s, 2), "step0": step0_eval, "final": curve[-1], "full_10m_val": full_10m_ev, "curve": curve,
        })
        del model, optimizers
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def run_explicit_density_tomographic_ablation_benchmark(
    stream: LongSeqFineWebStream, steps: int = 300, eval_interval: int = 100
) -> list[dict[str, Any]]:
    """Run the 300-step (`2.46M`-token) `T0..T3` ablation suite (`<= 8` args)."""
    variants = [
        ("T0_PURE_WAVEPACKET_RANK1_BASELINE", ExplicitDensityTomographicSpec(variant_mode=ExplicitDensityVariantMode.T0_PURE_WAVEPACKET_RANK1_BASELINE, enable_density_highway=False, enable_tomographic_readout=False, enable_noclick_and_vacuum=False, enable_two_register_bilinear=False), "Pure Rank-1 Complex Wavepacket (H=16xC16, Single 512D GEMM, Zero Density Highway, Zero Vacuum)"),
        ("T1_EXPLICIT_OUTER_DENSITY_TOMOGRAPHIC_512D", ExplicitDensityTomographicSpec(variant_mode=ExplicitDensityVariantMode.T1_EXPLICIT_OUTER_DENSITY_TOMOGRAPHIC_512D, enable_density_highway=True, enable_tomographic_readout=True, enable_noclick_and_vacuum=False, enable_two_register_bilinear=False), "Explicit C^{16x16} Outer-Product Density Highway + 16x32=512D Probe Tomography Tr(rho_t E_v) Readout"),
        ("T2_DENSITY_TOMOGRAPHY_PLUS_NOCLICK_AND_VACUUM", ExplicitDensityTomographicSpec(variant_mode=ExplicitDensityVariantMode.T2_DENSITY_TOMOGRAPHY_PLUS_NOCLICK_AND_VACUUM, enable_density_highway=True, enable_tomographic_readout=True, enable_noclick_and_vacuum=True, enable_two_register_bilinear=False), "Explicit C^{16x16} Density Tomography + Trunk No-Click POVM Sink + Unified 512D Vacuum Subspace (Zero Byte-Pairs)"),
        ("T3_ZERO_SYNTHESIS_TWO_REGISTER_BILINEAR_TRACE", ExplicitDensityTomographicSpec(variant_mode=ExplicitDensityVariantMode.T3_ZERO_SYNTHESIS_TWO_REGISTER_BILINEAR_TRACE, enable_density_highway=True, enable_tomographic_readout=True, enable_noclick_and_vacuum=True, enable_two_register_bilinear=True), "T2 + Zero-Synthesis 256x256 Two-Register Bilinear Density Trace <phi_hi | rho_t | phi_lo> (Zero 50304x512 Re-Synthesis)"),
    ]
    return _train_and_eval_explicit_density_variants(stream, variants, steps=steps, eval_interval=eval_interval)


def run_explicit_density_tomographic_scaling_benchmark(
    stream: LongSeqFineWebStream, steps: int = 1200, eval_interval: int = 150, enable_two_register_bilinear: bool = True
) -> list[dict[str, Any]]:
    """Run the 1,200-step (`9.83M`-token) `T4` scaling benchmark (`<= 8` args)."""
    variants = [
        ("T4_CONSOLIDATED_TOMOGRAPHIC_SCALING_1200", ExplicitDensityTomographicSpec(variant_mode=ExplicitDensityVariantMode.T4_CONSOLIDATED_TOMOGRAPHIC_SCALING_1200, enable_density_highway=True, enable_tomographic_readout=True, enable_noclick_and_vacuum=True, enable_two_register_bilinear=enable_two_register_bilinear), "1,200-Step (9.83M-Token) Consolidated Explicit C^{16x16} Density Highway + 512D Tomographic Readout Scaling"),
    ]
    return _train_and_eval_explicit_density_variants(stream, variants, steps=steps, eval_interval=eval_interval)
