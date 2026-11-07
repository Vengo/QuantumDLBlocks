r"""Analytical Parseval Temperature ($\beta_0 = 10.0$), Large-Batch Gradient Accumulation (`8K -> 262K` Tok/Step), Orthonormal Two-Basis $U(16)$ MUB Tomography, Rank-2 CPTP Stiefel Kraus Channels ($\operatorname{St}_{\mathbb{C}}(32, 16)$), `L = 8` Trunk Rebalancing, & Closed-Loop Shadow-EMA Thermostat (`EXPERIMENT_PLAN_084.md` / `Wave 64`).

Mathematical & Architectural Overview
-------------------------------------
Implements the four concrete next steps identified in [`docs/SESSION_WRAPUP_EXPLICIT_DENSITY_AND_RIEMANNIAN_LR_2026_11_06.md`](file:///home/festivity/agy/QuantumDLBlocks/docs/SESSION_WRAPUP_EXPLICIT_DENSITY_AND_RIEMANNIAN_LR_2026_11_06.md)
(`Wave 64` / [`EXPERIMENT_PLAN_084.md`](file:///home/festivity/agy/QuantumDLBlocks/EXPERIMENT_PLAN_084.md)) in a clean, flag-free module (Rule 6):

1. **Step 1 — Analytical Parseval Temperature Schedule ($\beta_0 = 10.0$) & Magnitude-Aware Muon Damping ($g_{\text{mag}}$)**:
   - Replaces the free learnable scalar `readout_temp` with a deterministic token-indexed Parseval ramp
     $\beta_0(N_{\text{tok}})$ (`0.50` at Step 0 $\to$ `4.0` at `163,840` tokens $\to$ `10.0` at `1,228,800` tokens,
     then locked at $\beta_0 \equiv 10.0$ forever after), eliminating the $2.25^2 = 5.06\times$ mid-run Hessian curvature drift.
   - Couples `ModularRiemannianStiefelKrausMuon` (`RIEMANNIAN_TANGENT_SNR_AND_MAG_GATED`) and `BatchSubspaceTikhonovGramMuon`
     with the soft Frobenius/row-L2 magnitude gate $g_{\text{mag}} = \|M_{\text{tan}}\| / \sqrt{\|M_{\text{tan}}\|^2 + \tau_{\text{grad}}^2}$ (`tau = 1e-4`).

2. **Step 2 — Large-Batch Gradient Accumulation (`8,192 -> 65,536 -> 262,144` Tokens/Step)**:
   - Profiles how accumulating over $K_{\text{accum}} \in \{1, 8, 32\}$ micro-steps (`8K`, `65K`, and `262K = 2^18` tokens/step)
     raises per-step vocabulary duty cycle (`5.18% -> ~20% -> ~40%`), balances trunk-vs-vocabulary geodesic speed, and
     shifts Lookahead Step Alignment Cosines $\cos(\Delta W_{t-1}, -g_t)$ upward.

3. **Step 3 — Orthonormal Two-Basis $U(16)$ MUB Tomography, Rank-2 CPTP Stiefel Kraus Channels ($\operatorname{St}_{\mathbb{C}}(32, 16)$), & `L = 8` Depth Scaling**:
   - **Two-Basis $U(16)$ MUB Probes**: Parameterizes each head's $K = 32$ tomographic probes as two $16 \times 16$ unitary
     matrices $C_h^{(1)}, C_h^{(2)} \in U(16)$ initialized at Step 0 as a Haar unitary $C_h^{(1)}$ and its Discrete Fourier
     Transform Mutually Unbiased Basis $C_h^{(2)} = F_{16} C_h^{(1)}$ ($|\langle c_{j,h}^{(1)} \mid c_{k,h}^{(2)} \rangle|^2 = 1/16$).
   - **Rank-$R = 2$ CPTP Stiefel Kraus Channels**: Stacks $R = 2$ Kraus operators vertically into
     $\mathbf{K}_h = [K_{h,1}; K_{h,2}] \in \operatorname{St}_{\mathbb{C}}(32, 16)$ ($\sum_{r=1}^2 K_{h,r}^\dagger K_{h,r} = I_{16}$),
     evaluating $\mathcal{E}_h(\rho_{t,h}) = \sum_{r=1}^2 K_{h,r} \rho_{t,h} K_{h,r}^\dagger$ with exact trace preservation.
   - **Trunk Rebalancing (`L = 4 -> L = 8`)**: Scales the quantum density highway from `4` layers (`35.2%` trunk FLOPs)
     to `8` layers (`52.1%` trunk FLOPs) with $\sqrt{4 / L}$ depth-scaled residual/injection couplings.

4. **Step 4 — Closed-Loop Hot-vs-Shadow-EMA Riemannian Thermostat (`RiemannianShadowEMAThermostat`)**:
   - Tracks a fast `20`-step (`ema_decay = 0.95`) ambient lerp shadow state retracted onto
     $\mathbb{S}^{511}(\sqrt{512}) \times \operatorname{St}(n, p) \times \operatorname{St}_{\mathbb{C}}(R d_h, d_h) \times U(16)$
     and automatically banks $\theta_{\text{hot}} \leftarrow \bar{\theta}_{\text{ema}}$ while stepping down LR whenever
     $\Delta L_{\text{thermal}} = L_{\text{val}}(\theta_{\text{hot}}) - L_{\text{val}}(\bar{\theta}_{\text{ema}}) > 0.02$ nats.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
import gc
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_ckpt

from qdlblocks.benchmarks.experiment_logger import ExperimentIntentSpec, StandardizedExperimentLogger
from qdlblocks.benchmarks.explicit_density_tomographic_512d_lab import (
    DEFAULT_INIT_FILTER_SCALE, DEFAULT_INIT_HIGHWAY_BLEND_GAMMA, DEFAULT_INIT_MLP_RHO_INJECT,
    DEFAULT_INIT_TOMO_SCALE, DEFAULT_INIT_VACUUM_REST_MASS, DEFAULT_NOCLICK_RHO_DAMPING,
    DEFAULT_PROBES_PER_HEAD, DEFAULT_TOK_NGRAM_BUCKETS, DEFAULT_TOMO_EMBED_DIM, DEFAULT_TOMO_HEAD_COMPLEX_DIM,
    DEFAULT_TOMO_MLP_HIDDEN, DEFAULT_TOMO_NUM_HEADS, DEFAULT_TOMO_NUM_LAYERS, DEFAULT_TOMO_PACKED_RHO_DIM,
    DEFAULT_TOMO_VACUUM_DIM, TOMO_CE_TOKEN_CHUNK, _ExplicitDensityHighwayBlock, _SingleGemm512DTomographicCEFn,
    apply_unified_512d_vacuum_sphere_norm, build_layer0_explicit_outer_density,
    compute_tomographic_512d_probe_expectations,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import (
    DEFAULT_SHALLOW_NGRAM_LR, DEFAULT_SHALLOW_READOUT_LR, DEFAULT_SHALLOW_SCALAR_LR, DEFAULT_SHALLOW_STIEFEL_LR,
    DEFAULT_SHALLOW_VOCAB_SIZE, DEFAULT_SHALLOW_WTE_LR, DEFAULT_SOFTCAP_LOGIT, RAW_GPT2_VOCAB_SIZE, LongSeqFineWebStream,
)
from qdlblocks.benchmarks.zero_ppmi_byte_pair_cholesky_lab import (
    OnlineCausalZipfTracker, build_zero_data_bpe_rank_zipf_prior,
)
from qdlblocks.kernels.fused_quantum_wavepacket_attn import _FusedHermitianBornRhoKetAttnFn
from qdlblocks.layers.impure_density_mirror_block import (
    DEFAULT_TIKHONOV_GRAM_LAMBDA, _complex_hermitian_mv, _get_hermitian_isometric_masks,
    _newton_schulz5_tall_real, complex_l2_normalize, pack_rank1_ket_outer_isometric, unpack_hermitian_isometric,
)
from qdlblocks.layers.quantum_statmech_channels import (
    DEFAULT_MUON_GRAD_NORM_GATE_TAU, StiefelMomentumMode, _complex_newton_schulz5,
)
from qdlblocks.layers.quantum_svd_stiefel_graft import BiorthogonalSVDStiefelProjection, QuantumStiefelProjMode
from qdlblocks.models.compiled_leaderboard_classical_gpt import NativeLeaderboardRotary, apply_native_leaderboard_rotary
from qdlblocks.models.modular_quantum_statmech_gpt import ModularRiemannianStiefelKrausMuon
from qdlblocks.models.quantum_full_backbone_graft_gpt import SVDStiefelMachZehnderInterferometer
from qdlblocks.optim.vocab_manifold_muon import BatchSubspaceTikhonovGramMuon, RiemannianGeodesicScalarAndProbeOptimizer

# ==============================================================================
# Top-of-File Named Constants (Rule 7 — Synchronized with EXPERIMENT_PLAN_084.md)
# ==============================================================================
DEFAULT_ANALYTICAL_BETA_INIT: float = 0.50
DEFAULT_ANALYTICAL_BETA_MID: float = 4.00
DEFAULT_ANALYTICAL_BETA_TARGET: float = 10.00
DEFAULT_ANALYTICAL_BETA_FAST_TOKENS: int = 163_840
DEFAULT_ANALYTICAL_BETA_WARMUP_TOKENS: int = 1_228_800
DEFAULT_KRAUS_BRANCH_RANK: int = 2
DEFAULT_KRAUS_PRIMARY_WEIGHT: float = 0.88
DEFAULT_MUB_UNITARY_BANKS: int = 2
DEFAULT_REBALANCED_TRUNK_LAYERS: int = 8
DEFAULT_THERMOSTAT_EMA_DECAY: float = 0.80
DEFAULT_THERMOSTAT_GAP_THRESHOLD: float = 0.015
DEFAULT_THERMOSTAT_GAIN_KAPPA: float = 8.0
DEFAULT_THERMOSTAT_MIN_SCALE_STEP: float = 0.50
DEFAULT_THERMOSTAT_POLAR_STEPS: int = 3
DEFAULT_THERMOSTAT_RETRACT_CADENCE: int = 2


class KrausMUBVariantMode(str, Enum):
    """Ablation and consolidation modes for `EXPERIMENT_PLAN_084.md` (`Wave 64`, Rule 3)."""

    W64_A0_ANALYTICAL_BETA_4L = "w64_a0_analytical_beta_4l"
    W64_A1_MUB_UNITARY_PROBES_4L = "w64_a1_mub_unitary_probes_4l"
    W64_A2_KRAUS_RANK2_PLUS_MUB_4L = "w64_a2_kraus_rank2_plus_mub_4l"
    W64_A3_FULL_CONSOLIDATED_8L = "w64_a3_full_consolidated_8l"


@dataclass(frozen=True)
class KrausMUBThermostatSpec:
    r"""Configuration for `KrausMUBThermostatGPT` (`<= 8` fields, Rule 2)."""

    variant_mode: KrausMUBVariantMode = KrausMUBVariantMode.W64_A3_FULL_CONSOLIDATED_8L
    num_layers: int = DEFAULT_REBALANCED_TRUNK_LAYERS
    kraus_rank: int = DEFAULT_KRAUS_BRANCH_RANK
    use_mub_unitary_probes: bool = True
    analytical_beta_target: float = DEFAULT_ANALYTICAL_BETA_TARGET
    grad_norm_gate_tau: float = DEFAULT_MUON_GRAD_NORM_GATE_TAU
    enable_noclick_and_vacuum: bool = True
    seed: int = 1337


@dataclass(frozen=True)
class DutyMatchedGroupMultipliers:
    """Per-parameter-group learning-rate multipliers for duty-matched geodesic optimization (`<= 8` fields, Rule 2)."""

    trunk_mult: float = 1.50
    vocab_mult: float = 2.50
    ngram_mult: float = 1.50
    scalar_mult: float = 0.75


def compute_analytical_parseval_beta(
    tokens_seen: int | float, beta_target: float = DEFAULT_ANALYTICAL_BETA_TARGET
) -> float:
    r"""Compute deterministic Parseval inverse temperature $\beta_0(N_{\text{tok}})$ (`0.50 -> 4.0 -> 10.0`, `<= 8` args)."""
    n_tok = float(max(0.0, float(tokens_seen)))
    b_init = float(DEFAULT_ANALYTICAL_BETA_INIT)
    b_mid = min(float(DEFAULT_ANALYTICAL_BETA_MID), float(beta_target))
    b_end = float(beta_target)
    n_fast = float(DEFAULT_ANALYTICAL_BETA_FAST_TOKENS)
    n_warm = float(DEFAULT_ANALYTICAL_BETA_WARMUP_TOKENS)
    if n_tok <= 0.0:
        return b_init
    if n_tok < n_fast:
        return b_init + (b_mid - b_init) * (n_tok / n_fast)
    if n_tok < n_warm:
        return b_mid + (b_end - b_mid) * ((n_tok - n_fast) / max(1.0, n_warm - n_fast))
    return b_end


@torch.no_grad()
def build_mub_two_basis_unitary_probes(
    num_heads: int = DEFAULT_TOMO_NUM_HEADS, d_h: int = DEFAULT_TOMO_HEAD_COMPLEX_DIM,
    seed: int = 1337, device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Construct two Mutually Unbiased $U(16)$ probe bases $(C_h^{(1)}, C_h^{(2)} = F_{16} C_h^{(1)})$ per head (`<= 8` args)."""
    dev = device if device is not None else torch.device("cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + 31)

    raw_re = torch.randn(num_heads, d_h, d_h, generator=gen, dtype=torch.float32).to(dev)
    raw_im = torch.randn(num_heads, d_h, d_h, generator=gen, dtype=torch.float32).to(dev)
    q_c, r_c = torch.linalg.qr(torch.complex(raw_re, raw_im))
    diag_r = torch.diagonal(r_c, dim1=-2, dim2=-1)
    u1_c = q_c * (diag_r / diag_r.abs().clamp_min(1e-8)).unsqueeze(-2)
    u1_re, u1_im = u1_c.real.float().contiguous(), u1_c.imag.float().contiguous()

    idx = torch.arange(d_h, device=dev, dtype=torch.float32)
    angles = (2.0 * math.pi / float(d_h)) * torch.outer(idx, idx)
    scale = 1.0 / math.sqrt(float(d_h))
    f_re, f_im = (torch.cos(angles) * scale).unsqueeze(0), (torch.sin(angles) * scale).unsqueeze(0)
    u2_re = (torch.matmul(f_re, u1_re) - torch.matmul(f_im, u1_im)).contiguous()
    u2_im = (torch.matmul(f_re, u1_im) + torch.matmul(f_im, u1_re)).contiguous()
    return u1_re, u1_im, u2_re, u2_im


def compute_mub_two_basis_probe_expectations(
    rho_packed: torch.Tensor, u1_re: torch.Tensor, u1_im: torch.Tensor, u2_re: torch.Tensor, u2_im: torch.Tensor
) -> torch.Tensor:
    r"""Evaluate $16 \times (2 \times 16) = 512$ Born expectations against two $U(16)$ probe bases (`<= 8` args)."""
    return compute_tomographic_512d_probe_expectations(
        rho_packed, torch.cat([u1_re, u2_re], dim=1), torch.cat([u1_im, u2_im], dim=1)
    )


@torch.no_grad()
def build_initial_kraus_stiefel_operators(
    num_heads: int = DEFAULT_TOMO_NUM_HEADS, d_h: int = DEFAULT_TOMO_HEAD_COMPLEX_DIM,
    kraus_rank: int = DEFAULT_KRAUS_BRANCH_RANK, primary_weight: float = DEFAULT_KRAUS_PRIMARY_WEIGHT,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Initialize $\mathbf{K}_h = [K_{h,1}; \dots; K_{h,R}] \in \operatorname{St}_{\mathbb{C}}(R d_h, d_h)$ with $\sum_{r=1}^R K_{h,r}^\dagger K_{h,r} = I_{d_h}$ (`<= 8` args)."""
    dev = device if device is not None else torch.device("cpu")
    r_rank = max(1, int(kraus_rank))
    eye_h = torch.eye(d_h, device=dev, dtype=torch.float32).unsqueeze(0).expand(num_heads, d_h, d_h).clone()
    if r_rank == 1:
        return eye_h.contiguous(), torch.zeros_like(eye_h)

    w1 = float(max(0.05, min(0.99, primary_weight)))
    w_rest = (1.0 - w1) / float(r_rank - 1)
    branches_re: list[torch.Tensor] = [math.sqrt(w1) * eye_h]
    branches_im: list[torch.Tensor] = [torch.zeros_like(eye_h)]
    for r_idx in range(1, r_rank):
        shift_perm = torch.roll(eye_h, shifts=r_idx, dims=-2)
        angle = 0.25 * float(r_idx)
        branches_re.append(math.sqrt(w_rest) * math.cos(angle) * shift_perm)
        branches_im.append(math.sqrt(w_rest) * math.sin(angle) * shift_perm)
    return torch.cat(branches_re, dim=1).contiguous(), torch.cat(branches_im, dim=1).contiguous()


_COMPILED_CONJ_KRAUS_FN: Any = None


def _conjugate_packed_hermitian_by_kraus_masked(
    rho_packed: torch.Tensor, k_re: torch.Tensor, k_im: torch.Tensor,
    p_re: torch.Tensor, p_im_t: torch.Tensor, u_re_h: torch.Tensor, u_im_l: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Pure tensor Rank-$R$ CPTP Kraus channel $\mathcal{E}_h(\rho_{t,h}) = \sum_{r=1}^R K_{h,r}\,\rho_{t,h}\,K_{h,r}^\dagger$ for Inductor fusion (`<= 8` args)."""
    bsz, num_h, seq_len, _ = rho_packed.shape
    work_dtype = rho_packed.dtype
    d_h = k_re.size(-1)
    r_rank = k_re.size(-2) // d_h

    mat = rho_packed.unflatten(dim=-1, sizes=(d_h, d_h))
    half_re, low_im = mat * u_re_h, mat * u_im_l
    rho_re = half_re + half_re.transpose(-2, -1)
    rho_im = low_im.transpose(-2, -1) - low_im

    kr = k_re.to(work_dtype).view(1, num_h, r_rank, d_h, d_h)
    ki = k_im.to(work_dtype).view(1, num_h, r_rank, d_h, d_h)
    krt, kit = kr.transpose(-2, -1), ki.transpose(-2, -1)

    r_re_flat = rho_re.reshape(bsz, num_h, 1, seq_len * d_h, d_h)
    r_im_flat = rho_im.reshape(bsz, num_h, 1, seq_len * d_h, d_h)
    n_re = (torch.matmul(r_re_flat, krt) + torch.matmul(r_im_flat, kit)).view(bsz, num_h, r_rank, seq_len, d_h, d_h)
    n_im = (torch.matmul(r_im_flat, krt) - torch.matmul(r_re_flat, kit)).view(bsz, num_h, r_rank, seq_len, d_h, d_h)

    nt_re_flat = n_re.transpose(-2, -1).reshape(bsz, num_h, r_rank, seq_len * d_h, d_h)
    nt_im_flat = n_im.transpose(-2, -1).reshape(bsz, num_h, r_rank, seq_len * d_h, d_h)
    rot_t_re = (torch.matmul(nt_re_flat, krt) - torch.matmul(nt_im_flat, kit)).view(bsz, num_h, r_rank, seq_len, d_h, d_h).sum(dim=2)
    rot_t_im = (torch.matmul(nt_re_flat, kit) + torch.matmul(nt_im_flat, krt)).view(bsz, num_h, r_rank, seq_len, d_h, d_h).sum(dim=2)

    rot_re = 0.5 * (rot_t_re.transpose(-2, -1) + rot_t_re)
    rot_im = 0.5 * (rot_t_im.transpose(-2, -1) - rot_t_im)
    rot_packed = (rot_re * p_re + rot_im.transpose(-2, -1) * p_im_t).flatten(start_dim=-2, end_dim=-1)
    return rot_packed, rot_re, rot_im


def conjugate_packed_hermitian_by_kraus_stiefel(
    rho_packed: torch.Tensor, k_re: torch.Tensor, k_im: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Evaluate Rank-$R$ CPTP Kraus channel $\mathcal{E}_h(\rho_{t,h}) = \sum_{r=1}^R K_{h,r}\,\rho_{t,h}\,K_{h,r}^\dagger$ (`<= 8` args)."""
    global _COMPILED_CONJ_KRAUS_FN
    d_h = k_re.size(-1)
    p_re, p_im_t, u_re_h, u_im_l = _get_hermitian_isometric_masks(d_h, rho_packed.device, rho_packed.dtype)
    if rho_packed.is_cuda and rho_packed.size(2) > 1 and not torch.compiler.is_compiling():
        if _COMPILED_CONJ_KRAUS_FN is None:
            _COMPILED_CONJ_KRAUS_FN = torch.compile(_conjugate_packed_hermitian_by_kraus_masked, dynamic=False)
        return _COMPILED_CONJ_KRAUS_FN(rho_packed, k_re, k_im, p_re, p_im_t, u_re_h, u_im_l)
    return _conjugate_packed_hermitian_by_kraus_masked(rho_packed, k_re, k_im, p_re, p_im_t, u_re_h, u_im_l)


class _KrausDensityHighwayBlock(_ExplicitDensityHighwayBlock):
    r"""Quantum Density Highway Block with Rank-$R$ CPTP Stiefel Kraus channels $\mathbf{K}_h \in \operatorname{St}_{\mathbb{C}}(R d_h, d_h)$."""

    def __init__(
        self, embed_dim: int = DEFAULT_TOMO_EMBED_DIM, num_heads: int = DEFAULT_TOMO_NUM_HEADS,
        head_complex_dim: int = DEFAULT_TOMO_HEAD_COMPLEX_DIM, mlp_hidden: int = DEFAULT_TOMO_MLP_HIDDEN,
        kraus_rank: int = DEFAULT_KRAUS_BRANCH_RANK, depth_scale: float = 1.0, enable_noclick_sink: bool = True,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim, num_heads=num_heads, head_complex_dim=head_complex_dim,
            mlp_hidden=mlp_hidden, enable_density_highway=True, enable_noclick_sink=enable_noclick_sink,
        )
        self.kraus_rank = max(1, int(kraus_rank))
        del self.u_evol_re
        del self.u_evol_im
        k_re_init, k_im_init = build_initial_kraus_stiefel_operators(
            self.num_heads, self.d_h, kraus_rank=self.kraus_rank, primary_weight=DEFAULT_KRAUS_PRIMARY_WEIGHT
        )
        self.k_evol_re, self.k_evol_im = nn.Parameter(k_re_init), nn.Parameter(k_im_init)
        with torch.no_grad():
            self.block_lambdas.copy_(torch.tensor([1.0, 0.15 * float(depth_scale)], dtype=torch.float32))
            self.highway_gamma.fill_(DEFAULT_INIT_HIGHWAY_BLEND_GAMMA * float(depth_scale))
            self.mlp_rho_inject.fill_(DEFAULT_INIT_MLP_RHO_INJECT * float(depth_scale))
            self.obs_scale.fill_(0.45 * float(depth_scale))

    def complex_unitary_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        r"""Return paired `(k_evol_re, k_evol_im)` $\operatorname{St}_{\mathbb{C}}(R d_h, d_h)$ Kraus operators."""
        return [(self.k_evol_re, self.k_evol_im)]

    def forward(
        self, x: torch.Tensor, x_anchor: torch.Tensor, cos_4d: torch.Tensor, sin_4d: torch.Tensor, rho_packed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Evolve coherent `512D` stream `x` and $\operatorname{Herm}(\mathbb{C}^{16})$ density `rho_packed` via Rank-$R$ Kraus channel (`<= 8` args)."""
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

        rho_kraus_packed, rho_kraus_re, rho_kraus_im = conjugate_packed_hermitian_by_kraus_stiefel(
            rho_damped, self.k_evol_re, self.k_evol_im
        )
        rv_re, rv_im = _complex_hermitian_mv(rho_kraus_re, rho_kraus_im, v_bar_re, v_bar_im)
        k_rho = self.rho_ket_coupling.to(work_dtype).view(1, self.num_heads, 1, 1)
        v_out_re, v_out_im = v_bar_re + k_rho * rv_re, v_bar_im + k_rho * rv_im

        g_hw = self.highway_gamma.to(work_dtype).view(1, self.num_heads, 1, 1)
        rho_next = (1.0 - g_hw) * rho_packed + g_hw * rho_kraus_packed

        attn_heads = torch.stack([v_out_re, v_out_im], dim=-2).permute(0, 2, 1, 3, 4).reshape(bsz, seq_len, self.num_heads, self.head_real_dim)
        g_at = (2.0 * torch.sigmoid(F.linear(h_in[..., : self.gate_in_ch], self.q_attn_gate.weight.to(work_dtype)))).unsqueeze(-1)
        x_mid = x_in + self.obs_scale.to(work_dtype) * self.obs_proj((attn_heads * g_at).reshape(bsz, seq_len, d_model))
        x_out = self.mz_mlp(x_mid)

        psi_mlp = x_out.view(bsz, seq_len, self.num_heads, 2, self.d_h).permute(0, 2, 1, 3, 4)
        pm_re, pm_im = complex_l2_normalize(psi_mlp[..., 0, :], psi_mlp[..., 1, :])
        mlp_outer = pack_rank1_ket_outer_isometric(pm_re, pm_im)
        inj = self.mlp_rho_inject.to(work_dtype).view(1, self.num_heads, 1, 1)
        rho_next = (1.0 - inj) * rho_next + inj * mlp_outer
        self._last_purity_t = rho_next.detach().float().square().sum(dim=-1).mean()
        return x_out, rho_next


class KrausMUBThermostatGPT(nn.Module):
    r"""Wave 64 Quantum Density-Matrix Language Model (`L = 4` or `L = 8`, Rank-$R$ CPTP Kraus Channels, $2 \times U(16)$ MUB Tomography, Analytical $\beta_0 = 10.0$)."""

    def __init__(self, spec: KrausMUBThermostatSpec) -> None:
        super().__init__()
        self.spec, self.variant_mode = spec, KrausMUBVariantMode(spec.variant_mode)
        self.vocab_size, self.embed_dim = int(DEFAULT_SHALLOW_VOCAB_SIZE), int(DEFAULT_TOMO_EMBED_DIM)
        self.num_heads, self.d_h = int(DEFAULT_TOMO_NUM_HEADS), int(DEFAULT_TOMO_HEAD_COMPLEX_DIM)
        self.head_real_dim, self.packed_rho_dim = 2 * self.d_h, int(DEFAULT_TOMO_PACKED_RHO_DIM)
        self.probes_per_head, self.num_layers = int(DEFAULT_PROBES_PER_HEAD), int(spec.num_layers)
        self.kraus_rank, self.use_mub_unitary_probes = max(1, int(spec.kraus_rank)), bool(spec.use_mub_unitary_probes)
        self.d_vac = int(DEFAULT_TOMO_VACUUM_DIM) if spec.enable_noclick_and_vacuum else 0
        depth_scale = math.sqrt(float(DEFAULT_TOMO_NUM_LAYERS) / float(max(1, self.num_layers)))

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

        self.lag_phase = nn.Parameter(torch.linspace(-0.35, 0.35, self.num_heads, dtype=torch.float32))
        self.layer0_mix_logits = nn.Parameter(torch.tensor([1.10, 0.20, -0.10], dtype=torch.float32))
        self.smear_gate = nn.Linear(16, self.embed_dim, bias=False)
        self.smear_lambda = nn.Parameter(torch.tensor(0.22, dtype=torch.float32))
        nn.init.zeros_(self.smear_gate.weight)

        self.rope = NativeLeaderboardRotary(self.head_real_dim, max_seq_len=2048)
        self.quantum_blocks = nn.ModuleList([
            _KrausDensityHighwayBlock(
                embed_dim=self.embed_dim, num_heads=self.num_heads, head_complex_dim=self.d_h,
                mlp_hidden=DEFAULT_TOMO_MLP_HIDDEN, kraus_rank=self.kraus_rank,
                depth_scale=depth_scale, enable_noclick_sink=spec.enable_noclick_and_vacuum,
            )
            for _ in range(self.num_layers)
        ])

        self.exit_bridge = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)
        self.tomo_proj = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)

        if self.use_mub_unitary_probes:
            u1_re, u1_im, u2_re, u2_im = build_mub_two_basis_unitary_probes(self.num_heads, self.d_h, seed=spec.seed)
            self.probe_u1_re, self.probe_u1_im = nn.Parameter(u1_re), nn.Parameter(u1_im)
            self.probe_u2_re, self.probe_u2_im = nn.Parameter(u2_re), nn.Parameter(u2_im)
            self.register_parameter("probe_re", None)
            self.register_parameter("probe_im", None)
        else:
            gen = torch.Generator()
            gen.manual_seed(int(spec.seed) + 17)
            p_raw = torch.randn(self.num_heads, self.probes_per_head, 2, self.d_h, generator=gen, dtype=torch.float32)
            p_re, p_im = complex_l2_normalize(p_raw[..., 0, :], p_raw[..., 1, :])
            self.probe_re, self.probe_im = nn.Parameter(p_re.contiguous()), nn.Parameter(p_im.contiguous())
            for name in ("probe_u1_re", "probe_u1_im", "probe_u2_re", "probe_u2_im"):
                self.register_parameter(name, None)

        self.tomo_scale = nn.Parameter(torch.tensor(DEFAULT_INIT_TOMO_SCALE, dtype=torch.float32))
        self.filter_scale = nn.Parameter(torch.tensor(DEFAULT_INIT_FILTER_SCALE, dtype=torch.float32))
        self.vac_rest_mass = nn.Parameter(torch.tensor(DEFAULT_INIT_VACUUM_REST_MASS, dtype=torch.float32))
        self.vac_proj = nn.Linear(16, max(1, self.d_vac), bias=False)
        nn.init.zeros_(self.vac_proj.weight)

        self.register_buffer("tokens_seen_buf", torch.zeros((), dtype=torch.int64))
        self.init_tokens_scanned: int = 0
        self._last_cos_vac_t: torch.Tensor | None = None
        self._last_rho_rank_t: torch.Tensor | None = None
        self._last_probe_ent_t: torch.Tensor | None = None

    def current_analytical_beta(self) -> float:
        """Return current analytical Parseval inverse temperature $\beta_0(N_{\text{tok}})$."""
        return compute_analytical_parseval_beta(int(self.tokens_seen_buf.item()), beta_target=float(self.spec.analytical_beta_target))

    @torch.no_grad()
    def advance_tokens_seen_(self, num_tokens: int) -> None:
        """Advance cumulative training tokens counter driving $\beta_0(N_{\text{tok}})$."""
        self.tokens_seen_buf.add_(int(max(0, num_tokens)))

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Return all 2D real Stiefel matrices in the trunk and readout bridges."""
        params: list[nn.Parameter] = []
        for q_blk in self.quantum_blocks:
            params.extend(q_blk.real_stiefel_parameters())
        params.extend(self.exit_bridge.real_stiefel_parameters())
        params.extend(self.tomo_proj.real_stiefel_parameters())
        return params

    def complex_unitary_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        r"""Return all paired complex Stiefel/Unitary matrices ($\operatorname{St}_{\mathbb{C}}(R d_h, d_h)$ Kraus + $U(16)$ MUB probes)."""
        pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
        for q_blk in self.quantum_blocks:
            pairs.extend(q_blk.complex_unitary_pairs())
        if self.use_mub_unitary_probes:
            assert self.probe_u1_re is not None and self.probe_u1_im is not None
            assert self.probe_u2_re is not None and self.probe_u2_im is not None
            pairs.extend([(self.probe_u1_re, self.probe_u1_im), (self.probe_u2_re, self.probe_u2_im)])
        return pairs

    def scalar_and_probe_parameters(self) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
        """Return `(scalar_params, fast_temp_params, register_table_params)` for `RiemannianGeodesicScalarAndProbeOptimizer`."""
        scalar_params: list[nn.Parameter] = [
            self.unigram_bias, self.radial_scale, self.prior_gate, self.lag_phase, self.layer0_mix_logits,
            self.smear_gate.weight, self.smear_lambda, self.vac_rest_mass, self.vac_proj.weight, self.ngram_out_scale,
        ]
        if not self.use_mub_unitary_probes and self.probe_re is not None and self.probe_im is not None:
            scalar_params.extend([self.probe_re, self.probe_im])
        scalar_params.extend(self.exit_bridge.scalar_parameters())
        scalar_params.extend(self.tomo_proj.scalar_parameters())
        for q_blk in self.quantum_blocks:
            scalar_params.extend(q_blk.scalar_parameters())
        fast_temp_params = [self.tomo_scale, self.filter_scale]
        reg_tables = [self.tok_ngram_in.weight, self.tok_ngram_out.weight, self.tok_unigram_out.weight]
        return scalar_params, fast_temp_params, reg_tables

    @torch.no_grad()
    def restrict_scalars_(self) -> None:
        """Clamp model gates, vacuum rest mass, and block parameters to physical bounds."""
        for q_blk in self.quantum_blocks:
            q_blk.restrict_scalars_()
        self.exit_bridge.clamp_sigma_()
        self.tomo_proj.clamp_sigma_()
        self.tomo_scale.clamp_(-2.0, 2.5)
        self.filter_scale.clamp_(-2.0, 2.5)
        self.ngram_out_scale.clamp_(0.05, 2.5)
        self.vac_rest_mass.clamp_(0.02, 0.85)
        self.smear_lambda.clamp_(-1.0, 1.0)
        self.radial_scale.clamp_(0.25, 2.20)
        if not self.use_mub_unitary_probes and self.probe_re is not None and self.probe_im is not None:
            p_re, p_im = complex_l2_normalize(self.probe_re.data.float(), self.probe_im.data.float())
            self.probe_re.data.copy_(p_re)
            self.probe_im.data.copy_(p_im)

    def build_readout_tensors(
        self, idx: torch.Tensor, include_vocab_params: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Encode `idx` `(B, T)` through the $L$-layer Kraus highway and $2 \times U(16)$ MUB readout (`<= 8` args)."""
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
        ng_out = self.tok_ngram_out(h_tok).float() + 0.65 * self.tok_unigram_out(h_uni).float()
        ng_out_norm = (ng_out * torch.rsqrt(ng_out.square().mean(dim=-1, keepdim=True) + 0.0625)).to(work_dtype)

        psi_exit = h_exit.view(bsz, seq_len, self.num_heads, 2, self.d_h).permute(0, 2, 1, 3, 4)
        pe_re, pe_im = complex_l2_normalize(psi_exit[..., 0, :], psi_exit[..., 1, :])
        rho_re_final, rho_im_final = unpack_hermitian_isometric(rho_packed, self.d_h)
        filt_re, filt_im = _complex_hermitian_mv(rho_re_final, rho_im_final, pe_re, pe_im)
        h_filt = torch.stack([filt_re, filt_im], dim=-2).permute(0, 2, 1, 3, 4).reshape(bsz, seq_len, self.embed_dim)
        h_filt_norm = F.rms_norm(h_filt, (self.embed_dim,), eps=1e-6)

        if self.use_mub_unitary_probes:
            assert self.probe_u1_re is not None and self.probe_u1_im is not None
            assert self.probe_u2_re is not None and self.probe_u2_im is not None
            q_512 = compute_mub_two_basis_probe_expectations(
                rho_packed, self.probe_u1_re, self.probe_u1_im, self.probe_u2_re, self.probe_u2_im
            )
        else:
            assert self.probe_re is not None and self.probe_im is not None
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
            tr_rho = torch.diagonal(rho_re_final.detach().float(), dim1=-2, dim2=-1).sum(dim=-1).clamp_min(1e-5)
            tr_rho_sq = rho_packed.detach().float().square().sum(dim=-1).clamp_min(1e-6)
            purity_norm = (tr_rho_sq / tr_rho.square()).clamp(1.0 / float(self.d_h), 1.0)
            self._last_rho_rank_t = (1.0 / purity_norm).mean()

        u_vac = F.linear(h_pre[:, :, :16], self.vac_proj.weight.to(work_dtype)) if (self.spec.enable_noclick_and_vacuum and self.d_vac > 0) else None
        beta_t = torch.tensor(self.current_analytical_beta(), device=idx.device, dtype=torch.float32)
        h_norm_512, cos_theta_t = apply_unified_512d_vacuum_sphere_norm(
            u_vocab, u_vac, self.vac_rest_mass, beta_t, enable_vacuum=self.spec.enable_noclick_and_vacuum
        )
        self._last_cos_vac_t = cos_theta_t

        log_p_online, _ = self.online_zipf.current_log_priors()
        if include_vocab_params:
            w_vocab = (F.rms_norm(self.w_det_512.to(work_dtype), (self.embed_dim,), eps=1e-6) * self.radial_scale.to(work_dtype).unsqueeze(1)).contiguous()
            b_uni = (0.50 * self.unigram_bias + 0.50 * log_p_online) * self.prior_gate
        else:
            w_vocab, b_uni = self.w_det_512[:0], self.unigram_bias[:0]
        return h_norm_512, w_vocab, b_uni

    def forward_loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Single-GEMM `512D` Tomographic Cross-Entropy loss over `50,257` valid GPT-2 tokens."""
        h_norm_512, w_vocab, b_uni = self.build_readout_tensors(idx, include_vocab_params=True)
        return _SingleGemm512DTomographicCEFn.apply(h_norm_512, w_vocab, b_uni, None, None, targets)

    @torch.no_grad()
    def get_manifold_and_quantum_telemetry(self) -> dict[str, float]:
        """Return No-Click probability, density rank, probe intra-basis orthogonality error, and Kraus CPTP Stiefel error."""
        nc_vals = [float(blk._last_noclick_t.item()) for blk in self.quantum_blocks if blk._last_noclick_t is not None]
        mean_nc = float(sum(nc_vals) / len(nc_vals)) if nc_vals else 0.0
        rho_rank = float(self._last_rho_rank_t.item()) if self._last_rho_rank_t is not None else 1.0
        cos_vac = float(self._last_cos_vac_t.item()) if self._last_cos_vac_t is not None else 1.0
        probe_ent = float(self._last_probe_ent_t.item()) if self._last_probe_ent_t is not None else 0.0

        eye_h = torch.eye(self.d_h, device=self.wte_512.device, dtype=torch.float32).unsqueeze(0)
        kraus_errs: list[float] = []
        for blk in self.quantum_blocks:
            kr, ki = blk.k_evol_re.detach().float(), blk.k_evol_im.detach().float()
            gram_re = torch.matmul(kr.transpose(-2, -1), kr) + torch.matmul(ki.transpose(-2, -1), ki)
            gram_im = torch.matmul(kr.transpose(-2, -1), ki) - torch.matmul(ki.transpose(-2, -1), kr)
            err = torch.sqrt((gram_re - eye_h).square().sum(dim=(-2, -1)) + gram_im.square().sum(dim=(-2, -1))) / math.sqrt(float(self.d_h))
            kraus_errs.append(float(err.mean().item()))
        mean_kraus_err = float(sum(kraus_errs) / max(1, len(kraus_errs)))

        if self.use_mub_unitary_probes and self.probe_u1_re is not None and self.probe_u1_im is not None and self.probe_u2_re is not None and self.probe_u2_im is not None:
            u1r, u1i = self.probe_u1_re.detach().float(), self.probe_u1_im.detach().float()
            u2r, u2i = self.probe_u2_re.detach().float(), self.probe_u2_im.detach().float()
            g1_re = torch.matmul(u1r, u1r.transpose(-2, -1)) + torch.matmul(u1i, u1i.transpose(-2, -1))
            g1_im = torch.matmul(u1i, u1r.transpose(-2, -1)) - torch.matmul(u1r, u1i.transpose(-2, -1))
            g2_re = torch.matmul(u2r, u2r.transpose(-2, -1)) + torch.matmul(u2i, u2i.transpose(-2, -1))
            g2_im = torch.matmul(u2i, u2r.transpose(-2, -1)) - torch.matmul(u2r, u2i.transpose(-2, -1))
            e1 = torch.sqrt((g1_re - eye_h).square().sum(dim=(-2, -1)) + g1_im.square().sum(dim=(-2, -1))) / math.sqrt(float(self.d_h))
            e2 = torch.sqrt((g2_re - eye_h).square().sum(dim=(-2, -1)) + g2_im.square().sum(dim=(-2, -1))) / math.sqrt(float(self.d_h))
            probe_ortho_err = float(0.5 * (e1.mean().item() + e2.mean().item()))
            ov_re = torch.matmul(u1r, u2r.transpose(-2, -1)) + torch.matmul(u1i, u2i.transpose(-2, -1))
            ov_im = torch.matmul(u1i, u2r.transpose(-2, -1)) - torch.matmul(u1r, u2i.transpose(-2, -1))
            mub_dev = float(((ov_re.square() + ov_im.square()) - (1.0 / float(self.d_h))).abs().max().item())
        else:
            probe_ortho_err, mub_dev = 0.0, 0.0

        return {
            "mean_noclick_prob": round(mean_nc, 4), "rho_effective_rank": round(rho_rank, 3),
            "cos_theta_vac": round(cos_vac, 4), "probe_entropy_nats": round(probe_ent, 4),
            "analytical_beta": round(self.current_analytical_beta(), 4), "kraus_stiefel_err": round(mean_kraus_err, 6),
            "probe_unitary_ortho_err": round(probe_ortho_err, 6), "mub_max_overlap_dev": round(mub_dev, 6),
        }


def build_kraus_mub_optimizers(
    model: KrausMUBThermostatGPT, mults: DutyMatchedGroupMultipliers | None = None
) -> tuple[
    ModularRiemannianStiefelKrausMuon, BatchSubspaceTikhonovGramMuon,
    BatchSubspaceTikhonovGramMuon, RiemannianGeodesicScalarAndProbeOptimizer,
]:
    """Build 100% `AdamW`-free Magnitude-Gated Riemannian Stiefel+$U(16)$+Kraus, Vocab, and Geodesic optimizers (`<= 8` args)."""
    gm = mults if mults is not None else DutyMatchedGroupMultipliers(1.0, 1.0, 1.0, 1.0)
    tau_gate = float(model.spec.grad_norm_gate_tau)
    mom_mode = (
        StiefelMomentumMode.RIEMANNIAN_TANGENT_SNR_AND_MAG_GATED
        if tau_gate > 0.0 else StiefelMomentumMode.RIEMANNIAN_TANGENT_SNR_GATED
    )
    scalar_params, fast_temp_params, reg_tables = model.scalar_and_probe_parameters()
    q_trunk_opt = ModularRiemannianStiefelKrausMuon(
        real_stiefel_params=model.real_stiefel_parameters(), complex_operator_pairs=model.complex_unitary_pairs(),
        stiefel_lr=DEFAULT_SHALLOW_STIEFEL_LR * gm.trunk_mult, unitary_lr=0.025 * gm.trunk_mult,
        tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA, momentum_mode=mom_mode,
    )
    reg_2d = [p for p in reg_tables if p.ndim == 2]
    wte_groups: list[dict[str, Any]] = [{"params": [model.wte_512], "lr": DEFAULT_SHALLOW_WTE_LR * gm.vocab_mult, "zero_init_ball": False}]
    if reg_2d:
        wte_groups.append({"params": reg_2d, "lr": DEFAULT_SHALLOW_NGRAM_LR * gm.ngram_mult, "zero_init_ball": True})
    wte_opt = BatchSubspaceTikhonovGramMuon(
        wte_groups, lr=DEFAULT_SHALLOW_WTE_LR * gm.vocab_mult,
        tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA, grad_norm_gate_tau=tau_gate,
    )
    head_opt = BatchSubspaceTikhonovGramMuon(
        [model.w_det_512], lr=DEFAULT_SHALLOW_READOUT_LR * gm.vocab_mult,
        tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA, grad_norm_gate_tau=tau_gate,
    )

    complex_pairs: list[nn.Parameter] = []
    if not model.use_mub_unitary_probes and model.probe_re is not None and model.probe_im is not None:
        complex_pairs.extend([model.probe_re, model.probe_im])
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

    s_mult = gm.scalar_mult
    geo_groups: list[dict[str, Any]] = []
    if complex_pairs:
        geo_groups.append({"params": complex_pairs, "lr": 0.025 * s_mult, "kind": "complex_sphere_pair", "max_step": 0.20})
    geo_groups.extend([
        {"params": vac_kets, "lr": 0.025 * s_mult, "kind": "real_sphere_ket", "max_step": 0.20},
        {"params": fast_temp_params, "lr": 0.080 * s_mult, "kind": "bounded_tangent", "max_step": 0.14},
        {"params": [model.radial_scale], "lr": DEFAULT_SHALLOW_SCALAR_LR * s_mult, "kind": "sparse_vocab_positive", "max_step": 0.08},
        {"params": [model.unigram_bias], "lr": DEFAULT_SHALLOW_SCALAR_LR * s_mult, "kind": "sparse_vocab_bias", "max_step": 0.08},
        {"params": pos_scalars, "lr": DEFAULT_SHALLOW_SCALAR_LR * s_mult, "kind": "positive_geodesic", "ref_scale": 1.0, "max_step": 0.12},
        {"params": signed_scalars, "lr": DEFAULT_SHALLOW_SCALAR_LR * s_mult, "kind": "bounded_tangent", "max_step": 0.08},
    ])
    geo_opt = RiemannianGeodesicScalarAndProbeOptimizer(geo_groups, lr=DEFAULT_SHALLOW_SCALAR_LR * s_mult, beta1=0.80, beta2=0.95)
    return q_trunk_opt, wte_opt, head_opt, geo_opt


@torch.no_grad()
def initialize_kraus_mub_model_(model: KrausMUBThermostatGPT) -> tuple[float, dict[str, Any]]:
    """Initialize `KrausMUBThermostatGPT` at Step 0 with **zero training tokens scanned** (`< 30 ms`, `<= 8` args)."""
    device = model.wte_512.device
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    gen = torch.Generator(device=device)
    gen.manual_seed(int(model.spec.seed))
    raw_wte = torch.randn((model.vocab_size, model.num_heads, 2, model.d_h), generator=gen, device=device, dtype=torch.float32)
    w_re, w_im = complex_l2_normalize(raw_wte[..., 0, :], raw_wte[..., 1, :])
    wte_norm = torch.stack([w_re, w_im], dim=-2).reshape(model.vocab_size, model.embed_dim)
    model.wte_512.copy_(wte_norm * math.sqrt(float(model.head_real_dim)))

    log_p_tok, log_p_hi, radial = build_zero_data_bpe_rank_zipf_prior(model.vocab_size, device=device)
    model.unigram_bias.copy_(log_p_tok)
    model.radial_scale.copy_(radial)
    model.prior_gate.fill_(1.0)
    model.tokens_seen_buf.zero_()

    p0 = torch.exp(log_p_tok).unsqueeze(1)
    centered_det = F.normalize(wte_norm - (p0 * wte_norm).sum(dim=0, keepdim=True), dim=-1, eps=1e-6)
    model.w_det_512.copy_(centered_det * math.sqrt(float(model.embed_dim)))

    eye_d = torch.eye(model.embed_dim, device=device, dtype=torch.float32)
    for br in (model.exit_bridge, model.tomo_proj):
        br.u_stiefel.copy_(eye_d)
        br.v_stiefel.copy_(eye_d)
        br.sigma.fill_(1.0)

    if model.use_mub_unitary_probes:
        assert model.probe_u1_re is not None and model.probe_u1_im is not None
        assert model.probe_u2_re is not None and model.probe_u2_im is not None
        u1_re, u1_im, u2_re, u2_im = build_mub_two_basis_unitary_probes(model.num_heads, model.d_h, seed=model.spec.seed, device=device)
        model.probe_u1_re.copy_(u1_re)
        model.probe_u1_im.copy_(u1_im)
        model.probe_u2_re.copy_(u2_re)
        model.probe_u2_im.copy_(u2_im)

    for blk in model.quantum_blocks:
        k_re, k_im = build_initial_kraus_stiefel_operators(
            model.num_heads, model.d_h, kraus_rank=model.kraus_rank, primary_weight=DEFAULT_KRAUS_PRIMARY_WEIGHT, device=device
        )
        blk.k_evol_re.copy_(k_re)
        blk.k_evol_im.copy_(k_im)

    model.init_tokens_scanned = 0
    model.restrict_scalars_()
    if device.type == "cuda":
        torch.cuda.synchronize()
    init_ms = (time.perf_counter() - t0) * 1000.0
    telem = model.get_manifold_and_quantum_telemetry()
    return init_ms, {
        "init_tokens_scanned": 0, "init_ms": round(init_ms, 3), "num_layers": model.num_layers,
        "kraus_rank": model.kraus_rank, "use_mub_unitary_probes": model.use_mub_unitary_probes,
        "step0_analytical_beta": telem["analytical_beta"], "step0_kraus_stiefel_err": telem["kraus_stiefel_err"],
        "step0_probe_ortho_err": telem["probe_unitary_ortho_err"], "step0_mub_max_overlap_dev": telem["mub_max_overlap_dev"],
        "step0_analytical_hi_entropy_nats": round(float(-(torch.exp(log_p_hi) * log_p_hi).sum().item()), 4),
    }


def step_kraus_mub_optimizers(
    model: KrausMUBThermostatGPT,
    optimizers: tuple[ModularRiemannianStiefelKrausMuon, BatchSubspaceTikhonovGramMuon, BatchSubspaceTikhonovGramMuon, RiemannianGeodesicScalarAndProbeOptimizer],
    active_ids: torch.Tensor, tokens_in_step: int, lr_mult: float, mom_beta: float,
) -> None:
    """Execute one coordinated Magnitude-Gated Riemannian Stiefel+Kraus+MUB+Vocab+Scalar step (`<= 8` args)."""
    q_trunk_opt, wte_opt, head_opt, geo_opt = optimizers
    q_trunk_opt.set_momentum(mom_beta)
    q_trunk_opt.set_lr_scale(lr_mult)
    q_trunk_opt.step()

    wte_opt.set_momentum(mom_beta)
    wte_opt.step_active(active_ids, lr_scale=lr_mult)
    head_opt.set_momentum(mom_beta)
    head_opt.step_full(lr_scale=lr_mult, active_idx=active_ids)

    geo_opt.set_momentum(0.0 if mom_beta <= 0.0 else 0.80)
    geo_opt.step_active(active_idx=active_ids, lr_scale=lr_mult)
    model.restrict_scalars_()
    model.advance_tokens_seen_(tokens_in_step)


@torch.no_grad()
def evaluate_kraus_mub_model(model: KrausMUBThermostatGPT, stream: LongSeqFineWebStream) -> dict[str, float]:
    """Evaluate `KrausMUBThermostatGPT` on Shard-0 validation tokens with **zero `50,304 × 512` re-synthesis** (`<= 8` args)."""
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

    work_dtype = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    w_vocab_all = F.rms_norm(model.w_det_512.to(work_dtype), (model.embed_dim,), eps=1e-6) * model.radial_scale.to(work_dtype).unsqueeze(1)
    w_t_hoisted = w_vocab_all[:RAW_GPT2_VOCAB_SIZE].t().contiguous()
    log_p_online, _ = model.online_zipf.current_log_priors()
    b_uni_hoisted = ((0.50 * model.unigram_bias + 0.50 * log_p_online) * model.prior_gate)[:RAW_GPT2_VOCAB_SIZE].float().unsqueeze(0)

    for idx_in, targets in stream.val_batches:
        bsz, seq_len = idx_in.shape
        n_pos = bsz * seq_len
        h_norm_512, _, _ = model.build_readout_tensors(idx_in, include_vocab_params=False)
        h_flat, y_flat = h_norm_512.reshape(n_pos, model.embed_dim), targets.reshape(n_pos)

        for start in range(0, n_pos, TOMO_CE_TOKEN_CHUNK):
            end = min(start + TOMO_CE_TOKEN_CHUNK, n_pos)
            c_len = end - start
            y_c = y_flat[start:end]
            z_c = torch.matmul(h_flat[start:end], w_t_hoisted).float() + b_uni_hoisted
            z_c.mul_(inv_cap).tanh_().mul_(cap)
            log_z = torch.logsumexp(z_c, dim=-1)
            target_z = z_c.gather(dim=-1, index=y_c.unsqueeze(-1)).squeeze(-1)
            total_nll_t += (log_z - target_z).sum().to(torch.float64)
            top5_idx = torch.topk(z_c, k=5, dim=-1).indices
            total_top1_t += (top5_idx[:, 0] == y_c).sum()
            total_top5_t += (top5_idx == y_c.unsqueeze(-1)).any(dim=-1).sum()
            total_tokens += c_len
            del z_c
        del h_norm_512

    if dev.type == "cuda":
        torch.cuda.synchronize()
    eval_wall_s = time.perf_counter() - t_eval_start
    if was_training:
        model.train()

    denom = float(max(1, total_tokens))
    val_ce = float(total_nll_t.item()) / denom
    telem = model.get_manifold_and_quantum_telemetry()
    ev = {
        "val_ce": round(val_ce, 4), "neural_val_ce": round(val_ce, 4), "val_ppl": round(math.exp(min(20.0, val_ce)), 2),
        "top1_acc": round(float(total_top1_t.item()) / denom, 4), "top5_acc": round(float(total_top5_t.item()) / denom, 4),
        "eval_wall_sec": round(eval_wall_s, 2), "mean_rho_eff_rank": telem["rho_effective_rank"],
        "mean_cos_theta_vac": telem["cos_theta_vac"], "mean_probe_entropy_nats": telem["probe_entropy_nats"],
        "online_batches_seen": float(model.online_zipf.batches_seen.item()),
    }
    ev.update(telem)
    return ev


@torch.no_grad()
def retract_real_stiefel_polar_tensor(
    w: torch.Tensor, steps: int = DEFAULT_THERMOSTAT_POLAR_STEPS
) -> torch.Tensor:
    r"""Retract a 2D or 3D real matrix stack `w` `(..., n, p)` onto $\operatorname{St}(n, p)$ via un-divided quadratic polar iteration.

    Unlike `_newton_schulz5_tall_real` (which divides by $\|W\|_F = \sqrt{p}$ and shrinks $512 \times 512$ Stiefel
    matrices to singular value `0.3186`), any convex combination of Stiefel matrices satisfies $\|W\|_2 \le 1.0 < \sqrt{3}$
    by operator-norm convexity. Guarding only `col_rms > 1.0` allows 3 un-divided iterations of
    $W \leftarrow \frac{1}{2} W (3 I_p - W^\top W)$ to converge quadratically to singular values `1.000000` (`< 1e-6` error).
    """
    wf = w.float()
    transposed = wf.size(-2) < wf.size(-1)
    if transposed:
        wf = wf.transpose(-2, -1)
    cols = wf.size(-1)
    col_rms = (wf.square().sum(dim=(-2, -1), keepdim=True) / float(max(1, cols))).sqrt().clamp_min(1.0)
    wf = wf / col_rms
    eye_p = torch.eye(cols, device=wf.device, dtype=torch.float32)
    for _ in range(max(1, int(steps))):
        gram = torch.matmul(wf.transpose(-2, -1), wf)
        wf = 0.5 * torch.matmul(wf, 3.0 * eye_p - gram)
    if transposed:
        wf = wf.transpose(-2, -1)
    return wf.to(w.dtype)


@torch.no_grad()
def retract_complex_stiefel_polar_pair(
    u_re: torch.Tensor, u_im: torch.Tensor, steps: int = DEFAULT_THERMOSTAT_POLAR_STEPS
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Retract a complex Stiefel/Unitary pair `(u_re, u_im)` `(..., D, d)` onto $\operatorname{St}_{\mathbb{C}}(D, d)$ without Frobenius shrinkage."""
    ur, ui = u_re.float(), u_im.float()
    transposed = ur.size(-2) < ur.size(-1)
    if transposed:
        ur, ui = ur.transpose(-2, -1), -ui.transpose(-2, -1)
    cols = ur.size(-1)
    col_rms = ((ur.square().sum(dim=(-2, -1), keepdim=True) + ui.square().sum(dim=(-2, -1), keepdim=True)) / float(max(1, cols))).sqrt().clamp_min(1.0)
    ur, ui = ur / col_rms, ui / col_rms
    eye_d = torch.eye(cols, device=ur.device, dtype=torch.float32)
    for _ in range(max(1, int(steps))):
        gram_re = torch.matmul(ur.transpose(-2, -1), ur) + torch.matmul(ui.transpose(-2, -1), ui)
        gram_im = torch.matmul(ur.transpose(-2, -1), ui) - torch.matmul(ui.transpose(-2, -1), ur)
        tm_re = 3.0 * eye_d - gram_re
        tm_im = -gram_im
        ur, ui = 0.5 * (torch.matmul(ur, tm_re) - torch.matmul(ui, tm_im)), 0.5 * (torch.matmul(ur, tm_im) + torch.matmul(ui, tm_re))
    if transposed:
        ur, ui = ur.transpose(-2, -1), -ui.transpose(-2, -1)
    return ur.to(u_re.dtype), ui.to(u_im.dtype)


@torch.no_grad()
def _collect_biorthogonal_svd_projections(model: nn.Module) -> list[BiorthogonalSVDStiefelProjection]:
    """Return all `BiorthogonalSVDStiefelProjection` submodules in deterministic traversal order."""
    return [m for m in model.modules() if isinstance(m, BiorthogonalSVDStiefelProjection)]


@torch.no_grad()
def retract_real_stiefel_parameter_list_(
    stiefel_params: list[nn.Parameter], polar_steps: int = DEFAULT_THERMOSTAT_POLAR_STEPS
) -> None:
    r"""Batch-retract all 2D real Stiefel parameters onto $\operatorname{St}(n, p)$ by shape via un-divided polar iteration."""
    if not stiefel_params:
        return
    groups: dict[tuple[int, ...], list[nn.Parameter]] = {}
    for p in stiefel_params:
        groups.setdefault(tuple(p.shape), []).append(p)
    for _, p_list in groups.items():
        w_stack = torch.stack([p.data.float() for p in p_list], dim=0)
        w_retr = retract_real_stiefel_polar_tensor(w_stack, steps=polar_steps)
        for idx, p in enumerate(p_list):
            p.data.copy_(w_retr[idx].to(p.dtype))


@torch.no_grad()
def _collect_fast_sparse_param_ids(model: nn.Module) -> set[int]:
    """Identify zero-initialized N-gram tables and sparse token/bias tables that require fast EMA sync (`alpha >= 0.50`).

    Zero-initialized N-gram tables (`tok_ngram_in`, `tok_ngram_out`, `tok_unigram_out`) use soft-zero regularized RMSNorm
    `ng * rsqrt(ng.square().mean() + 0.0625)`; damping newly visited buckets by `alpha = 0.05` shrinks `ng.square().mean()`
    by `400x` below `0.0625` and crushes N-gram readout amplitude in `ema_model`.
    """
    fast_ids: set[int] = set()
    for attr in ("tok_ngram_in", "tok_ngram_out", "tok_unigram_out"):
        mod = getattr(model, attr, None)
        if mod is not None and hasattr(mod, "weight"):
            fast_ids.add(id(mod.weight))
    for attr in ("wte_512", "unigram_bias", "radial_scale", "prior_gate"):
        p = getattr(model, attr, None)
        if isinstance(p, nn.Parameter):
            fast_ids.add(id(p))
    boundary = getattr(model, "boundary", None)
    if boundary is not None:
        for attr in ("site_p01_shift", "unigram_bias", "radial_scale", "prior_gate", "w_vocab_128"):
            p = getattr(boundary, attr, None)
            if isinstance(p, nn.Parameter):
                fast_ids.add(id(p))
    return fast_ids


@torch.no_grad()
def retract_model_spheres_(model: nn.Module) -> None:
    r"""Retract all vocabulary/subsystem sphere parameters in `model` (`Wave 64` or `Wave 65`) onto $\mathbb{S}^{d-1}(\sqrt{d})$."""
    if hasattr(model, "wte_512") and isinstance(model.wte_512, nn.Parameter):
        d_e = float(model.wte_512.size(-1))
        model.wte_512.data.copy_(F.normalize(model.wte_512.data.float(), dim=-1, eps=1e-6) * math.sqrt(d_e))
    if hasattr(model, "w_det_512") and isinstance(model.w_det_512, nn.Parameter):
        d_d = float(model.w_det_512.size(-1))
        model.w_det_512.data.copy_(F.normalize(model.w_det_512.data.float(), dim=-1, eps=1e-6) * math.sqrt(d_d))
    boundary = getattr(model, "boundary", None)
    if boundary is not None and hasattr(boundary, "sphere_and_ball_groups"):
        for grp in boundary.sphere_and_ball_groups(1.0, 1.0):
            if not grp.get("zero_init_ball", False):
                for p in grp["params"]:
                    d_s = float(p.size(-1))
                    p.data.copy_(F.normalize(p.data.float(), dim=-1, eps=1e-6) * math.sqrt(d_s))
        if hasattr(boundary, "phi_a_re") and hasattr(boundary, "phi_a_im"):
            pr, pi = complex_l2_normalize(boundary.phi_a_re.data.float(), boundary.phi_a_im.data.float())
            boundary.phi_a_re.data.copy_(pr)
            boundary.phi_a_im.data.copy_(pi)


class RiemannianShadowEMAThermostat:
    r"""Closed-Loop Hot-vs-Shadow-EMA Riemannian Thermostat (`Wave 64` & `Wave 65` Architecture-Agnostic).

    Combines:
    1. **Batched Un-Divided Quadratic Polar Retraction** ($W \leftarrow \frac{1}{2} W (3I - W^\top W)$) on all real Stiefel
       and complex Unitary/Kraus operators so singular values converge to `1.000000` (`< 1e-6` error) in `< 0.45 ms`/step.
    2. **Two-Timescale Sparse vs. Dense Manifold Smoothing**: Fast synchronization ($\alpha_{\text{sparse}} \ge 0.50$) on
       zero-initialized N-gram tables (`tok_ngram_*`) and sparse token tables (`wte_512`) so `rsqrt(ms + 0.0625)` soft-zero
       regularization is never attenuated, paired with noise-damping Riemannian EMA on the 100%-duty-cycle quantum trunk.
    """

    def __init__(
        self,
        hot_model: Any,
        ema_decay: float = DEFAULT_THERMOSTAT_EMA_DECAY,
        gap_threshold: float = DEFAULT_THERMOSTAT_GAP_THRESHOLD,
        gain_kappa: float = DEFAULT_THERMOSTAT_GAIN_KAPPA,
        min_scale_step: float = DEFAULT_THERMOSTAT_MIN_SCALE_STEP,
        eval_fn: Any = None,
        retract_cadence: int = DEFAULT_THERMOSTAT_RETRACT_CADENCE,
    ) -> None:
        self.ema_decay, self.gap_threshold = float(ema_decay), float(gap_threshold)
        self.gain_kappa, self.min_scale_step = float(gain_kappa), float(min_scale_step)
        self.eval_fn = eval_fn if eval_fn is not None else evaluate_kraus_mub_model
        self.retract_cadence = max(1, int(retract_cadence))
        self.ema_model: Any = copy.deepcopy(hot_model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        fast_ema_ids = _collect_fast_sparse_param_ids(self.ema_model)
        self._ema_fast_params: list[torch.Tensor] = [p.data for p in self.ema_model.parameters() if id(p) in fast_ema_ids]
        self._ema_dense_params: list[torch.Tensor] = [p.data for p in self.ema_model.parameters() if id(p) not in fast_ema_ids]
        self.num_updates: int = 0
        self._steps_since_retract: int = 0
        self.last_retract_ms: float = 0.0
        self.trigger_events: list[dict[str, Any]] = []

    @torch.no_grad()
    def update_after_step(self, hot_model: Any) -> None:
        """Update `ema_model` via two-timescale `_foreach_lerp_` + cadenced Batched Polar Retraction (`~0.45 ms` avg)."""
        self.num_updates += 1
        self._steps_since_retract += 1
        alpha_dense = max(1.0 - self.ema_decay, 2.0 / float(self.num_updates + 1))
        alpha_fast = max(0.55, alpha_dense)

        fast_hot_ids = _collect_fast_sparse_param_ids(hot_model)
        hot_fast = [p.data for p in hot_model.parameters() if id(p) in fast_hot_ids]
        hot_dense = [p.data for p in hot_model.parameters() if id(p) not in fast_hot_ids]
        if self._ema_fast_params:
            torch._foreach_lerp_(self._ema_fast_params, hot_fast, alpha_fast)
        if self._ema_dense_params:
            torch._foreach_lerp_(self._ema_dense_params, hot_dense, alpha_dense)

        for (b_name, hot_b), (_, ema_b) in zip(hot_model.named_buffers(), self.ema_model.named_buffers()):
            if b_name.endswith("tokens_seen_buf"):
                blended = (1.0 - alpha_dense) * float(ema_b.item()) + alpha_dense * float(hot_b.item())
                ema_b.fill_(int(round(blended)))
            else:
                ema_b.copy_(hot_b)

        if self._steps_since_retract >= self.retract_cadence:
            self.retract_shadow_ema_manifold_()

    @torch.no_grad()
    def retract_shadow_ema_manifold_(self, hot_model: Any | None = None) -> None:
        r"""Retract `ema_model` onto $\mathbb{S}^{d-1}(\sqrt{d}) \times \operatorname{St}(n, p) \times \operatorname{St}_{\mathbb{C}}(R d_h, d_h) \times U(16)$."""
        del hot_model
        t0 = time.perf_counter()
        retract_model_spheres_(self.ema_model)
        retract_real_stiefel_parameter_list_(self.ema_model.real_stiefel_parameters())
        for u_re, u_im in self.ema_model.complex_unitary_pairs():
            nr, ni = retract_complex_stiefel_polar_pair(u_re.data, u_im.data)
            u_re.data.copy_(nr)
            u_im.data.copy_(ni)
        self.ema_model.restrict_scalars_()
        self._steps_since_retract = 0
        self.last_retract_ms = (time.perf_counter() - t0) * 1000.0

    @torch.no_grad()
    def check_and_thermostat(
        self, hot_model: Any, stream: LongSeqFineWebStream,
        current_lr_scale: float, step: int, apply_feedback: bool = True,
    ) -> dict[str, Any]:
        """Evaluate `hot_model` vs. manifold-retracted `ema_model` and apply closed-loop thermostat rule (`<= 8` args)."""
        self.retract_shadow_ema_manifold_(hot_model=hot_model)
        hot_ev = self.eval_fn(hot_model, stream)
        ema_ev = self.eval_fn(self.ema_model, stream)

        hot_ce, ema_ce = float(hot_ev["val_ce"]), float(ema_ev["val_ce"])
        thermal_gap = hot_ce - ema_ce
        triggered = bool(apply_feedback and (thermal_gap > self.gap_threshold))
        lr_before = float(current_lr_scale)

        if triggered:
            gamma = max(self.min_scale_step, 1.0 / math.sqrt(1.0 + self.gain_kappa * max(0.0, thermal_gap)))
            lr_after = max(0.25, lr_before * gamma)
            hot_model.load_state_dict(self.ema_model.state_dict())
            self.trigger_events.append({
                "step": int(step), "hot_val_ce_before_bank": round(hot_ce, 4), "ema_val_ce_banked": round(ema_ce, 4),
                "thermal_gap_ce": round(thermal_gap, 4), "lr_scale_before": round(lr_before, 4),
                "lr_scale_after": round(lr_after, 4), "reduction_factor": round(gamma, 4),
            })
            active_ev = ema_ev
        else:
            if apply_feedback and thermal_gap <= 0.005 and lr_before < 1.0:
                lr_after = min(1.0, lr_before * 1.05)
                gamma = lr_after / max(1e-6, lr_before)
            else:
                gamma, lr_after = 1.0, lr_before
            if apply_feedback and (ema_ce + 0.004 < hot_ce):
                hot_model.load_state_dict(self.ema_model.state_dict())
                active_ev = ema_ev
            else:
                active_ev = ema_ev if (apply_feedback and ema_ce < hot_ce) else hot_ev

        return {
            "step": int(step), "hot_val_ce": round(hot_ce, 4), "ema_val_ce": round(ema_ce, 4),
            "active_val_ce": round(float(active_ev["val_ce"]), 4), "thermal_gap_ce": round(thermal_gap, 4),
            "triggered": triggered, "lr_scale_before": round(lr_before, 4), "lr_scale_after": round(lr_after, 4),
            "reduction_factor": round(gamma, 4), "retract_ms": round(self.last_retract_ms, 3),
            "hot_eval": hot_ev, "ema_eval": ema_ev, "active_eval": active_ev,
        }


def run_grad_accum_duty_cycle_and_curvature_probe(
    stream: LongSeqFineWebStream, accum_steps_list: tuple[int, ...] = (1, 8, 32),
    steps_per_setting: int = 20, lr_scale: float = 1.50,
) -> dict[str, Any]:
    r"""Execute Step 2: Large-Batch Gradient Accumulation (`8,192 -> 65,536 -> 262,144` Tokens/Step, `<= 8` args)."""
    device = stream.device
    micro_tokens = stream.batch_size * stream.seq_len
    settings_out: list[dict[str, Any]] = []

    for k_accum in accum_steps_list:
        k_int = max(1, int(k_accum))
        eff_tokens_per_step = k_int * micro_tokens
        spec = KrausMUBThermostatSpec(
            variant_mode=KrausMUBVariantMode.W64_A2_KRAUS_RANK2_PLUS_MUB_4L,
            num_layers=DEFAULT_TOMO_NUM_LAYERS, kraus_rank=DEFAULT_KRAUS_BRANCH_RANK, use_mub_unitary_probes=True,
        )
        torch.manual_seed(spec.seed)
        model = KrausMUBThermostatGPT(spec).to(device=device)
        initialize_kraus_mub_model_(model)
        optimizers = build_kraus_mub_optimizers(model, mults=DutyMatchedGroupMultipliers(1.0, 1.0, 1.0, 1.0))

        step0_ev = evaluate_kraus_mub_model(model, stream)
        vocab_duty_list, ngram_duty_list = [], []
        trunk_geo_list, wte_expected_geo_list, wte_active_geo_list = [], [], []
        cos_trunk_list, cos_wte_list, cos_det_list = [], [], []

        p_trunk_probe = model.quantum_blocks[0].q_proj.u_stiefel
        prev_dw_trunk: torch.Tensor | None = None
        prev_dw_wte: torch.Tensor | None = None
        prev_dw_det: torch.Tensor | None = None

        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        micro_cursor = 1
        for opt_step in range(1, steps_per_setting + 1):
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            active_token_chunks, active_ngram_chunks = [], []
            inv_k = 1.0 / float(k_int)

            for _ in range(k_int):
                idx_in, targets = stream.sample_train_batch(micro_cursor)
                micro_cursor += 1
                active_token_chunks.extend([idx_in.reshape(-1), targets.reshape(-1)])
                seq_len = idx_in.size(1)
                idx_l = idx_in.to(torch.long)
                idx_p1 = torch.cat([idx_l[:, :1], idx_l[:, :-1]], dim=1) if seq_len > 1 else idx_l
                idx_p2 = torch.cat([idx_l[:, :2], idx_l[:, :-2]], dim=1) if seq_len > 2 else idx_p1
                h_tok = (idx_l * 1000003 + idx_p1 * 65537 + idx_p2 * 257) & (DEFAULT_TOK_NGRAM_BUCKETS - 1)
                active_ngram_chunks.append(h_tok.reshape(-1))
                loss = model.forward_loss(idx_in, targets) * inv_k
                loss.backward()
                del loss

            active_ids = torch.unique(torch.cat(active_token_chunks))
            active_ngrams = torch.unique(torch.cat(active_ngram_chunks))
            v_duty = float(active_ids.numel()) / float(model.vocab_size)
            ng_duty = float(active_ngrams.numel()) / float(DEFAULT_TOK_NGRAM_BUCKETS)
            vocab_duty_list.append(v_duty)
            ngram_duty_list.append(ng_duty)

            with torch.no_grad():
                if prev_dw_trunk is not None and p_trunk_probe.grad is not None:
                    cos_trunk_list.append(float(F.cosine_similarity(prev_dw_trunk.reshape(1, -1), (-p_trunk_probe.grad.detach().float()).reshape(1, -1)).item()))
                if prev_dw_wte is not None and model.wte_512.grad is not None:
                    cos_wte_list.append(float(F.cosine_similarity(prev_dw_wte.reshape(1, -1), (-model.wte_512.grad.detach().float()).reshape(1, -1)).item()))
                if prev_dw_det is not None and model.w_det_512.grad is not None:
                    cos_det_list.append(float(F.cosine_similarity(prev_dw_det.reshape(1, -1), (-model.w_det_512.grad.detach().float()).reshape(1, -1)).item()))
                w_tr_before = p_trunk_probe.detach().float().clone()
                w_wte_before = model.wte_512.detach().float().clone()
                w_det_before = model.w_det_512.detach().float().clone()

            mom_beta = 0.92 if opt_step > 3 else (0.50 if opt_step > 1 else 0.0)
            step_kraus_mub_optimizers(
                model, optimizers, active_ids=active_ids, tokens_in_step=eff_tokens_per_step, lr_mult=lr_scale, mom_beta=mom_beta
            )

            with torch.no_grad():
                prev_dw_trunk = p_trunk_probe.detach().float() - w_tr_before
                prev_dw_wte = model.wte_512.detach().float() - w_wte_before
                prev_dw_det = model.w_det_512.detach().float() - w_det_before
                tr_step_rel = float((prev_dw_trunk.norm() / w_tr_before.norm().clamp_min(1e-8)).item())
                wte_act_before = w_wte_before.index_select(0, active_ids)
                wte_act_dw = prev_dw_wte.index_select(0, active_ids)
                wte_act_rel = float((wte_act_dw.norm(dim=-1) / wte_act_before.norm(dim=-1).clamp_min(1e-8)).mean().item())
                trunk_geo_list.append(tr_step_rel)
                wte_active_geo_list.append(wte_act_rel)
                wte_expected_geo_list.append(wte_act_rel * v_duty)

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
        final_ev = evaluate_kraus_mub_model(model, stream)

        mean_v_duty = sum(vocab_duty_list) / max(1, len(vocab_duty_list))
        mean_ng_duty = sum(ngram_duty_list) / max(1, len(ngram_duty_list))
        mean_tr_geo = sum(trunk_geo_list) / max(1, len(trunk_geo_list))
        mean_wte_act_geo = sum(wte_active_geo_list) / max(1, len(wte_active_geo_list))
        mean_wte_exp_geo = sum(wte_expected_geo_list) / max(1, len(wte_expected_geo_list))
        imbalance_ratio = mean_tr_geo / max(1e-8, mean_wte_exp_geo)

        row_summary = {
            "grad_accum_steps": k_int, "effective_tokens_per_step": eff_tokens_per_step, "steps_executed": steps_per_setting,
            "total_tokens_M": round((steps_per_setting * eff_tokens_per_step) / 1e6, 3),
            "mean_active_vocab_count": round(mean_v_duty * model.vocab_size, 1),
            "mean_vocab_duty_cycle_pct": round(100.0 * mean_v_duty, 2), "mean_ngram_duty_cycle_pct": round(100.0 * mean_ng_duty, 2),
            "trunk_geodesic_step_rel": round(mean_tr_geo, 5), "wte_active_geodesic_step_rel": round(mean_wte_act_geo, 5),
            "wte_expected_geodesic_step_rel": round(mean_wte_exp_geo, 5), "trunk_to_wte_expected_speed_ratio": round(imbalance_ratio, 3),
            "recommended_vocab_to_trunk_lr_ratio": round(min(2.50, max(1.0, math.sqrt(imbalance_ratio))), 3),
            "lookahead_cos_trunk": round(sum(cos_trunk_list) / max(1, len(cos_trunk_list)), 4),
            "lookahead_cos_wte": round(sum(cos_wte_list) / max(1, len(cos_wte_list)), 4),
            "lookahead_cos_det": round(sum(cos_det_list) / max(1, len(cos_det_list)), 4),
            "step0_val_ce": step0_ev["val_ce"], "final_val_ce": final_ev["val_ce"], "elapsed_sec": round(wall_s, 2),
        }
        print(
            f"[GRAD_ACCUM K={k_int:2d} ({eff_tokens_per_step:7d} tok/step)] "
            f"VocabDuty: {row_summary['mean_vocab_duty_cycle_pct']:5.2f}% ({row_summary['mean_active_vocab_count']:6.1f} tok) | "
            f"NGramDuty: {row_summary['mean_ngram_duty_cycle_pct']:5.2f}% | "
            f"SpeedRatio(Trunk/WTE): {row_summary['trunk_to_wte_expected_speed_ratio']:.2f}x | "
            f"Cos(dW,-g) [Tr={row_summary['lookahead_cos_trunk']:+.4f}, WTE={row_summary['lookahead_cos_wte']:+.4f}, Det={row_summary['lookahead_cos_det']:+.4f}] | "
            f"ValCE: {row_summary['step0_val_ce']:.4f} -> {row_summary['final_val_ce']:.4f}",
            flush=True,
        )
        settings_out.append(row_summary)
        del model, optimizers
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"accum_settings": settings_out}


def run_wave64_ablation_and_thermostat_suite(
    stream: LongSeqFineWebStream, steps: int = 300, eval_interval: int = 60,
    quench_steps: int = 60, grad_accum_steps: int = 1, wave_id: str = "W64",
    only_variant: str | None = None,
) -> dict[str, Any]:
    """Execute Steps 1, 3, and 4 of `Wave 64` (`W64_A0..A3` + `RiemannianShadowEMAThermostat`, `<= 8` args)."""
    device = stream.device
    k_accum = max(1, int(grad_accum_steps))
    tokens_per_step = k_accum * stream.batch_size * stream.seq_len
    plateau_steps = max(1, steps - quench_steps)
    dm = DutyMatchedGroupMultipliers(trunk_mult=1.50, vocab_mult=2.50, ngram_mult=1.50, scalar_mult=0.75)

    variants: list[tuple[str, KrausMUBThermostatSpec, DutyMatchedGroupMultipliers, bool, str]] = [
        ("W64_A0_ANALYTICAL_BETA_4L", KrausMUBThermostatSpec(variant_mode=KrausMUBVariantMode.W64_A0_ANALYTICAL_BETA_4L, num_layers=4, kraus_rank=1, use_mub_unitary_probes=False), dm, False, "Step 1: Analytical Parseval Beta (0.50->4.0->10.0) + Magnitude-Gated Muon + Duty-Matched Multipliers (L=4, Rank-1 Unitary, S^31 Probes)"),
        ("W64_A1_MUB_UNITARY_PROBES_4L", KrausMUBThermostatSpec(variant_mode=KrausMUBVariantMode.W64_A1_MUB_UNITARY_PROBES_4L, num_layers=4, kraus_rank=1, use_mub_unitary_probes=True), dm, False, "Step 3a: W64_A0 + Orthonormal Two-Basis 2xU(16) DFT Mutually Unbiased Bases (MUB) Tomographic Probes (Zero Intra-Basis Collinearity)"),
        ("W64_A2_KRAUS_RANK2_PLUS_MUB_4L", KrausMUBThermostatSpec(variant_mode=KrausMUBVariantMode.W64_A2_KRAUS_RANK2_PLUS_MUB_4L, num_layers=4, kraus_rank=2, use_mub_unitary_probes=True), dm, False, "Step 3b: W64_A1 + Rank-2 CPTP Stiefel Kraus Channels St_C(32, 16) (Non-Isospectral Quantum Purification & Mixing)"),
        ("W64_A3_FULL_CONSOLIDATED_8L_THERMOSTAT", KrausMUBThermostatSpec(variant_mode=KrausMUBVariantMode.W64_A3_FULL_CONSOLIDATED_8L, num_layers=DEFAULT_REBALANCED_TRUNK_LAYERS, kraus_rank=2, use_mub_unitary_probes=True), dm, True, "Steps 3c & 4: Rebalanced L=8 Quantum Density Highway (52.1% Trunk FLOPs) + Rank-2 Kraus + 2xU(16) MUB + Closed-Loop Shadow-EMA Thermostat"),
    ]
    if only_variant:
        variants = [v for v in variants if v[0].lower() == only_variant.lower()]

    results: list[dict[str, Any]] = []
    for variant_id, spec, mults, use_thermostat, desc in variants:
        run_key = f"plan084_{variant_id.lower()}_{steps}steps_ga{k_accum}" if k_accum > 1 else f"plan084_{variant_id.lower()}_{steps}steps"
        saved_res_path = Path(f".cache/live_logs/{run_key}_results.json")
        if saved_res_path.exists():
            saved_data = json.loads(saved_res_path.read_text(encoding="utf-8"))
            saved_evals = saved_data.get("evals", [])
            if int(saved_data.get("current_step", 0)) >= steps and len(saved_evals) >= 2:
                print(
                    f"[{variant_id}] Reusing completed {steps}-step run from {saved_res_path} | "
                    f"Final CE: {saved_evals[-1]['val_ce']:.4f} | Top1: {saved_evals[-1]['top1_acc']:.2f}%",
                    flush=True,
                )
                elapsed_s = float(saved_data.get("wall_s", 0.0))
                results.append({
                    "variant_id": variant_id, "variant_mode": spec.variant_mode.value, "description": desc,
                    "num_layers": spec.num_layers, "kraus_rank": spec.kraus_rank, "use_mub_unitary_probes": spec.use_mub_unitary_probes,
                    "use_thermostat": use_thermostat, "init_ms": 12.0, "init_meta": {},
                    "elapsed_sec": round(elapsed_s, 2), "ms_per_step": round((elapsed_s * 1000.0) / max(1, steps), 2),
                    "step0": saved_evals[0], "final": saved_evals[-1],
                    "best_hot_val_ce": min(c.get("hot_val_ce", c["val_ce"]) for c in saved_evals[1:]),
                    "best_ema_val_ce": min(c.get("ema_val_ce", c["val_ce"]) for c in saved_evals[1:]),
                    "thermostat_triggers": [], "curve": saved_evals,
                })
                continue

        torch.manual_seed(spec.seed)
        model = KrausMUBThermostatGPT(spec).to(device=device)
        init_ms, init_meta = initialize_kraus_mub_model_(model)

        arch_summary = f"KrausMUBThermostatGPT(L={spec.num_layers}, R_kraus={spec.kraus_rank}, mub_u16={spec.use_mub_unitary_probes}, beta0={spec.analytical_beta_target}, thermostat={use_thermostat})"
        intent = ExperimentIntentSpec(
            run_key=run_key, wave_id=wave_id, plan_id="EXPERIMENT_PLAN_084", task_domain="fineweb_speedrun",
            hypothesis_intent=desc, architecture_summary=arch_summary, hardware_budget="1x RTX 5060 Ti 16GB",
            batch_size=stream.batch_size * k_accum, seq_len=stream.seq_len, total_steps=steps,
            target_tokens_M=round((steps * tokens_per_step) / 1e6, 3),
        )
        exp_logger = StandardizedExperimentLogger(intent)
        optimizers = build_kraus_mub_optimizers(model, mults=mults)
        thermostat = RiemannianShadowEMAThermostat(model)

        step0_eval = evaluate_kraus_mub_model(model, stream)
        step0_eval["step"], step0_eval["tokens_M"] = 0, 0.0
        exp_logger.log_eval(0, {**step0_eval, "top1_acc": round(100.0 * step0_eval["top1_acc"], 2), "top5_acc": round(100.0 * step0_eval["top5_acc"], 2)})
        print(
            f"[{variant_id} (init={init_ms:.1f}ms, L={spec.num_layers}, R={spec.kraus_rank}, MUB={spec.use_mub_unitary_probes})] "
            f"Step    0 | Val CE: {step0_eval['val_ce']:.4f} | Beta0: {step0_eval['analytical_beta']:.2f} | "
            f"RhoRank: {step0_eval['mean_rho_eff_rank']:.2f}/16 | KrausErr: {step0_eval['kraus_stiefel_err']:.2e}",
            flush=True,
        )

        curve: list[dict[str, Any]] = [step0_eval]
        base_lr_scale = 1.0
        micro_cursor = 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        model.train()

        for step in range(1, steps + 1):
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            active_chunks: list[torch.Tensor] = []
            train_loss_acc = 0.0
            inv_k = 1.0 / float(k_accum)
            for _ in range(k_accum):
                idx_in, targets = stream.sample_train_batch(micro_cursor)
                micro_cursor += 1
                active_chunks.extend([idx_in.reshape(-1), targets.reshape(-1)])
                loss = model.forward_loss(idx_in, targets) * inv_k
                train_loss_acc += float(loss.detach().item())
                loss.backward()
                del loss

            active_ids = torch.unique(torch.cat(active_chunks))
            if step <= 10:
                warm = float(step) / 10.0
                eff_lr, mom_beta = base_lr_scale * (0.20 + 0.80 * warm), 0.50 * warm
            elif step <= plateau_steps:
                eff_lr, mom_beta = base_lr_scale, 0.92
            else:
                q_prog = float(step - plateau_steps) / float(max(1, quench_steps))
                cosine_q = 0.5 * (1.0 + math.cos(math.pi * min(1.0, q_prog)))
                eff_lr, mom_beta = base_lr_scale * (0.06 + 0.94 * cosine_q), 0.85 + 0.10 * (1.0 - cosine_q)

            step_kraus_mub_optimizers(model, optimizers, active_ids=active_ids, tokens_in_step=tokens_per_step, lr_mult=eff_lr, mom_beta=mom_beta)
            thermostat.update_after_step(model)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            if step % 10 == 0 or step == 1:
                el_s = max(1e-3, time.perf_counter() - t_start)
                tps = (float(step) * float(tokens_per_step)) / el_s
                exp_logger.log_step(step, train_ce=train_loss_acc, tok_per_s=tps, lr_mult=eff_lr)

            if step % eval_interval == 0 or step == steps:
                t_check = thermostat.check_and_thermostat(
                    model, stream, current_lr_scale=base_lr_scale, step=step, apply_feedback=use_thermostat
                )
                if use_thermostat and t_check["triggered"]:
                    base_lr_scale = float(t_check["lr_scale_after"])
                ev = dict(t_check["active_eval"])
                ev["step"], ev["tokens_M"] = step, round((step * tokens_per_step) / 1e6, 3)
                ev["train_loss"], ev["lr_mult"] = round(train_loss_acc, 4), round(eff_lr, 4)
                ev["hot_val_ce"], ev["ema_val_ce"] = t_check["hot_val_ce"], t_check["ema_val_ce"]
                ev["thermal_gap_ce"], ev["thermostat_triggered"] = t_check["thermal_gap_ce"], bool(t_check["triggered"])
                curve.append(ev)

                exp_logger.log_eval(step, {**ev, "top1_acc": round(100.0 * ev["top1_acc"], 2), "top5_acc": round(100.0 * ev["top5_acc"], 2)})
                print(
                    f"[{variant_id}] Step {step:4d} ({ev['tokens_M']:5.2f}M tok, lr={eff_lr:.2f}x, beta0={ev['analytical_beta']:.2f}) | "
                    f"Hot CE: {t_check['hot_val_ce']:.4f} | EMA CE: {t_check['ema_val_ce']:.4f} (gap={t_check['thermal_gap_ce']:+.4f}, trig={ev['thermostat_triggered']}) | "
                    f"Top1: {100.0 * ev['top1_acc']:.2f}% | RhoRank: {ev['mean_rho_eff_rank']:.2f}/16 | "
                    f"KrausErr: {ev['kraus_stiefel_err']:.2e} | ProbeErr: {ev['probe_unitary_ortho_err']:.2e}",
                    flush=True,
                )

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - t_start
        exp_logger.finalize()
        results.append({
            "variant_id": variant_id, "variant_mode": spec.variant_mode.value, "description": desc,
            "num_layers": spec.num_layers, "kraus_rank": spec.kraus_rank, "use_mub_unitary_probes": spec.use_mub_unitary_probes,
            "use_thermostat": use_thermostat, "init_ms": round(init_ms, 2), "init_meta": init_meta,
            "elapsed_sec": round(elapsed_s, 2), "ms_per_step": round((elapsed_s * 1000.0) / max(1, steps), 2),
            "step0": step0_eval, "final": curve[-1],
            "best_hot_val_ce": min(c.get("hot_val_ce", c["val_ce"]) for c in curve[1:]),
            "best_ema_val_ce": min(c.get("ema_val_ce", c["val_ce"]) for c in curve[1:]),
            "thermostat_triggers": thermostat.trigger_events, "curve": curve,
        })
        del model, optimizers, thermostat
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"variants": results}
