r"""Variable-Rank $R$-Mode Uncollapsed Cholesky Density & Vacuum-Subspace Dark Parking Laboratory (`EXPERIMENT_PLAN_079.md`).

Mathematical & Architectural Motivation
---------------------------------------
1. **Why the Vacuum Subspace $\mathcal{H}_{\text{vac}}$ Unlocks Unrestricted Cholesky Rank $R \ge 1$**:
   - On a standard unit sphere $\mathbb{S}^{d-1}$ ($d_{\text{vac}} = 0$), every Cholesky mode $r \in \{0, \dots, R-1\}$
     is forced by `RMSNorm` to have fixed full length $\|h_{\text{id}, r}\|_2 = \sqrt{64} = 8$. Increasing $R$ beyond
     $R = 4$ injects spurious extreme-value cosine overlaps ($\approx 0.58$) from low-energy modes and dilutes the
     anchor mode's trace share by $1/R$.
   - On the Vacuum-Augmented Sphere $\mathbb{S}^{(d + d_{\text{vac}}) - 1} = \mathcal{H}_{\text{vocab}} \oplus \mathcal{H}_{\text{vac}}$
     ($d_{\text{vac}} = 4$), coupled with:
     (a) **Rank-Invariant Excited Trace Scaling** $\gamma_{\text{trace}}(R) = \sqrt{3 / \max(1, R - 1)}$,
     (b) **Rank-Adaptive Coordinate Unit-RMS Scaling** $\alpha_{\text{chol}}(R) = \sqrt{R \cdot d_{\text{id}}} = 8\sqrt{R}$, and
     (c) **Progressive Vacuum Parking Rest-Mass** $\nu_{\text{vac}, r} = \nu_{\text{exc}}(1 + \delta_{\text{prog}} \frac{r-1}{\max(1, R-2)})$,
     any unused higher-order mode ($r \ge 4$) smoothly rotates into the Dark Parking Vacuum Subspace $\mathcal{H}_{\text{vac}}$
     ($\cos\theta_{\text{vac}}(t, r) \to 0$) while polysemous or phrase-coincidence contexts recruit up to $R$ modes!

2. **Multi-Horizon Causal & Schur Coincidence Basis for Unrestricted $R \in \{1, 2, 4, 6, 8, 12, 16, \dots\}$**:
   - Mode $r = 0$: Normalized primary value wavepacket $|r_{s,h}\rangle \in \mathbb{C}^{32}$.
   - Modes $r = 1, 2, 3$: Local value ket $|v_{s,h}\rangle$, rotary key ket $|k_{s,h}\rangle$, rotary query ket $|q_{s,h}\rangle$.
   - Modes $r = 4, 5$: Local Value-Key Schur coincidence $\sqrt{d_h}(|r_{s,h}\rangle \odot_{\mathbb{C}} |\hat{k}_{s,h}\rangle)$
     and Value-Query conjugate dipole $\sqrt{d_h}(|r_{s,h}\rangle \odot_{\mathbb{C}} |\hat{q}_{s,h}^*\rangle)$.
   - Modes $r = 6, 7$: Attended Causal Lag-1 ket $|\tilde{r}_{s-1, h}\rangle$ and Causal Bigram Schur coincidence
     $\sqrt{d_h}(|r_{s,h}\rangle \odot_{\mathbb{C}} |\tilde{r}_{s-1, h}\rangle)$ aggregated across the entire causal window $s \le t$.
   - Modes $r = 8..11$: Causal Lag-2, Skip-Bigram Schur, Query-Key Born, and Trigram Lag-1/Lag-2 Schur coincidences.
   - Modes $r \ge 12$: Cyclic sub-packet shifted & phase-rotated wavepackets supporting arbitrary $R \ge 1$.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Any

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.fused_quantum_wavepacket_attn import (
    _FusedHermitianBornRhoKetAttnFn, _get_compiled_chol_gram_fn, associative_low_rank_cholesky_gram_modulation,
)
from ..layers.impure_density_mirror_block import DEFAULT_TIKHONOV_GRAM_LAMBDA, TRACE_NORM_EPS
from ..layers.quantum_statmech_channels import QuantumMomentumScheduleMode, StiefelMomentumMode, compute_riemannian_momentum_beta
from ..layers.quantum_svd_stiefel_graft import apply_fubini_study_self_ortho
from ..models.compiled_leaderboard_classical_gpt import apply_native_leaderboard_rotary
from ..models.modular_quantum_statmech_gpt import ModularRiemannianStiefelKrausMuon
from ..optim.vocab_manifold_muon import BatchSubspaceTikhonovGramMuon
from .born_schur_density_lab import DEFAULT_FINEWEB_EVAL_INTERVAL, DEFAULT_FINEWEB_STEPS, DEFAULT_SCHUR_EPS
from .experiment_logger import ExperimentIntentSpec, StandardizedExperimentLogger
from .shallow_longseq_wavepacket_lm_lab import (
    DEFAULT_CARRIER_BIAS_DIM, DEFAULT_SHALLOW_EMBED_DIM, DEFAULT_SHALLOW_MLP_HIDDEN, DEFAULT_SHALLOW_NGRAM_LR,
    DEFAULT_SHALLOW_NUM_HEADS, DEFAULT_SHALLOW_PKT_COUNT, DEFAULT_SHALLOW_PKT_DIM, DEFAULT_SHALLOW_READOUT_LR,
    DEFAULT_SHALLOW_SCALAR_LR, DEFAULT_SHALLOW_STIEFEL_LR, DEFAULT_SHALLOW_UNITARY_LR, DEFAULT_SHALLOW_WTE_LR,
    DEFAULT_SOFTCAP_LOGIT, DEFAULT_STEP0_PROCRUSTES_BATCHES, DEFAULT_STEP0_PROCRUSTES_LAMBDA, RAW_GPT2_VOCAB_SIZE,
    LongSeqFineWebStream,
)
from .support_preserving_countsketch_lab import SupportSketchMode, SupportSketchSpec, compute_multishard_multilag_complex_wavefunction
from .vacuum_subspace_cholesky_lab import (
    DEFAULT_INIT_EXCITED_CHOL_VAC_MASS, DEFAULT_INIT_ID_VACUUM_MASS, DEFAULT_INIT_SEM_VACUUM_MASS,
    DEFAULT_LAYER0_CHOL_LAG1_SCALE, DEFAULT_LAYER0_CHOL_LAG2_SCALE, DEFAULT_LAYER0_CHOL_SCHUR_SCALE,
    DEFAULT_VACUUM_SUBSPACE_DIM, VacuumAwareNGramProjector, _VacuumFourMode512DWavepacketCEFn,
    vacuum_sphere_rms_norm,
)
from .wavefunction_outer_density_lab import (
    DEFAULT_CHOLESKY_ANCHOR_LOGIT_BIAS, DEFAULT_CHOLESKY_MODE_COHERENCE_MIX, DEFAULT_INIT_SCHUR_GATE_SCALE,
    DEFAULT_LONG_FINEWEB_EVAL_INTERVAL, DEFAULT_LONG_FINEWEB_STEPS, NativeCholeskyDensitySpec,
    NativeCholeskyDensityWavepacketGPT, _NativeCholeskySchurWavepacketBlock, phase_lock_cholesky_modes,
)

# Increase dynamo cache limit so sweeping across R in {1, 2, 4, 6, 8, 12, 16} never triggers recompilation warnings
torch._dynamo.config.cache_size_limit = max(64, getattr(torch._dynamo.config, "cache_size_limit", 8))

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_079.md
# ==============================================================================
REF_EXCITED_CHOLESKY_MODES: float = 3.0
DEFAULT_PROG_VAC_SLOPE: float = 0.35
DEFAULT_HIGH_MODE_INIT_GATE: float = -0.35
DEFAULT_LAYER0_HIGH_ROW_SCALE: float = -1.15
DEFAULT_RANK_SWEEP_MODES: tuple[int, ...] = (2, 4, 6, 8, 12)
DEFAULT_WINNING_CHOLESKY_RANK: int = 8


class VariableRankVariantMode(str, Enum):
    """Strategy enum for Variable-Rank $R$-Mode Cholesky & Vacuum Subspace experiments (`R0..R6`, Rule 3)."""

    R0_RANK8_NO_VACUUM_SPHERE = "r0_rank8_no_vacuum_sphere"
    R1_RANK2_VACUUM_CHOLESKY = "r1_rank2_vacuum_cholesky"
    R2_RANK4_VACUUM_CHOLESKY_REF = "r2_rank4_vacuum_cholesky_ref"
    R3_RANK6_VACUUM_CHOLESKY = "r3_rank6_vacuum_cholesky"
    R4_RANK8_VACUUM_CHOLESKY = "r4_rank8_vacuum_cholesky"
    R5_RANK12_VACUUM_CHOLESKY = "r5_rank12_vacuum_cholesky"
    R6_RANK8_VACUUM_CHOLESKY_SCALING_1200 = "r6_rank8_vacuum_cholesky_scaling_1200"


@dataclass(frozen=True)
class VariableRankCholeskySpec:
    r"""Configuration for `VariableRankVacuumCholeskyGPT` supporting unrestricted $R \ge 1$ (`<= 8` fields, Rule 2)."""

    variant_mode: VariableRankVariantMode = VariableRankVariantMode.R4_RANK8_VACUUM_CHOLESKY
    num_cholesky_modes: int = DEFAULT_WINNING_CHOLESKY_RANK
    vacuum_dim: int = DEFAULT_VACUUM_SUBSPACE_DIM
    use_layer0_input_cholesky: bool = True
    use_independent_rmode_512d: bool = False
    init_id_vac_mass: float = DEFAULT_INIT_ID_VACUUM_MASS
    init_excited_chol_vac_mass: float = DEFAULT_INIT_EXCITED_CHOL_VAC_MASS
    seed: int = 1337


def compute_rank_adaptive_cholesky_rms_scale(num_modes: int, id_real_dim: int = 64) -> float:
    r"""Return $\alpha_{\text{chol}}(R) = \sqrt{R \cdot d_{\text{id}}}$ so a mode with average energy $E_{t,r} = 1/R$ has unit coordinate RMS."""
    return math.sqrt(float(max(1, int(num_modes)) * int(id_real_dim)))


def compute_rank_invariant_excited_trace_scale(num_modes: int) -> float:
    r"""Return $\gamma_{\text{trace}}(R) = \sqrt{\frac{3}{\max(1, R - 1)}}$ so total excited trace is invariant to $R \ge 2$ (`1.0` at $R=4$)."""
    if int(num_modes) <= 1:
        return 0.0
    return math.sqrt(REF_EXCITED_CHOLESKY_MODES / float(max(1, int(num_modes) - 1)))


def _complex_schur_product(
    a_re: torch.Tensor, a_im: torch.Tensor, b_re: torch.Tensor, b_im: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled complex entrywise Hadamard/Schur product `scale * (a * b)` (`<= 8` args)."""
    return scale * (a_re * b_re - a_im * b_im), scale * (a_re * b_im + a_im * b_re)


def build_layer0_variable_rank_cholesky_factor(
    raw_wp_4d: torch.Tensor,
    wp_causal_re: torch.Tensor,
    wp_causal_im: torch.Tensor,
    row_scales: torch.Tensor,
    num_modes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Construct unit-Frobenius Layer-0 rank-$R$ Cholesky factor $L_{t,h}^{(0)} \in \mathbb{C}^{R \times 32}$ for any $R \ge 1$ (`<= 8` args).

    - Row $r = 0$: Current wavepacket $|\psi_{t,h}\rangle$ (active on all 8 packets including Identity Packet 7).
    - Rows $r = 1 \dots R-1$: Multi-horizon causal lag wavepackets ($t-1, t-2, t-3$) and entrywise Schur coincidences,
      zero-masked on Identity Packet $h = 7$ so lexical identity is strictly unblurred by neighbor tokens at Layer 0,
      and scaled by $\gamma_{\text{trace}}(R) = \sqrt{3 / \max(1, R - 1)}$ so anchor trace share $E_{t,0}$ is rank-invariant.
    """
    bsz, seq_len, n_heads, _, d_h = raw_wp_4d.shape
    r_modes = max(1, int(num_modes))
    work_dtype = raw_wp_4d.dtype
    re_0, im_0 = raw_wp_4d[:, :, :, 0, :].float(), raw_wp_4d[:, :, :, 1, :].float()

    n0 = torch.rsqrt(re_0.square().sum(dim=-1, keepdim=True) + im_0.square().sum(dim=-1, keepdim=True) + DEFAULT_SCHUR_EPS)
    u0_re, u0_im = re_0 * n0, im_0 * n0
    if r_modes == 1:
        return (
            u0_re.unsqueeze(-2).permute(0, 2, 1, 3, 4).to(work_dtype).contiguous(),
            u0_im.unsqueeze(-2).permute(0, 2, 1, 3, 4).to(work_dtype).contiguous(),
        )

    sem_mask = torch.ones((1, 1, n_heads, 1), device=raw_wp_4d.device, dtype=torch.float32)
    sem_mask[:, :, -1, :] = 0.0

    cr = torch.cos(wp_causal_re.float()).view(1, 1, n_heads, 1)
    si = torch.sin(wp_causal_im.float()).view(1, 1, n_heads, 1)
    rot_re = (u0_re * cr - u0_im * si) * sem_mask
    rot_im = (u0_re * si + u0_im * cr) * sem_mask

    z1 = torch.zeros_like(rot_re[:, :1])
    u1_re = torch.cat([z1, rot_re[:, :-1]], dim=1) if seq_len > 1 else torch.zeros_like(u0_re)
    u1_im = torch.cat([z1, rot_im[:, :-1]], dim=1) if seq_len > 1 else torch.zeros_like(u0_im)

    z2 = torch.zeros_like(rot_re[:, :2])
    u2_re = torch.cat([z2, rot_re[:, :-2]], dim=1) if seq_len > 2 else torch.zeros_like(u0_re)
    u2_im = torch.cat([z2, rot_im[:, :-2]], dim=1) if seq_len > 2 else torch.zeros_like(u0_im)

    sqrt_d = math.sqrt(float(d_h))
    s01_re, s01_im = _complex_schur_product(u0_re, u0_im, u1_re, u1_im, sqrt_d)
    s01_re, s01_im = s01_re * sem_mask, s01_im * sem_mask

    exc_re_list: list[torch.Tensor] = [u1_re, u2_re, s01_re]
    exc_im_list: list[torch.Tensor] = [u1_im, u2_im, s01_im]

    if r_modes > 4:
        z3 = torch.zeros_like(rot_re[:, :3])
        u3_re = torch.cat([z3, rot_re[:, :-3]], dim=1) if seq_len > 3 else torch.zeros_like(u0_re)
        u3_im = torch.cat([z3, rot_im[:, :-3]], dim=1) if seq_len > 3 else torch.zeros_like(u0_im)
        s02_re, s02_im = _complex_schur_product(u0_re, u0_im, u2_re, u2_im, sqrt_d)
        s12_re, s12_im = _complex_schur_product(u1_re, u1_im, u2_re, u2_im, sqrt_d)
        s03_re, s03_im = _complex_schur_product(u0_re, u0_im, u3_re, u3_im, sqrt_d)
        exc_re_list.extend([u3_re, s02_re * sem_mask, s12_re * sem_mask, s03_re * sem_mask])
        exc_im_list.extend([u3_im, s02_im * sem_mask, s12_im * sem_mask, s03_im * sem_mask])

        for extra_idx in range(len(exc_re_list), r_modes - 1):
            shift = (extra_idx - 6) % max(1, d_h - 1) + 1
            sh_re, sh_im = torch.roll(rot_re, shifts=shift, dims=-1), torch.roll(rot_im, shifts=shift, dims=-1)
            c_re, c_im = _complex_schur_product(sh_re, sh_im, u1_re, u1_im, sqrt_d)
            exc_re_list.append(c_re * sem_mask)
            exc_im_list.append(c_im * sem_mask)

    n_exc = r_modes - 1
    trace_gamma = compute_rank_invariant_excited_trace_scale(r_modes)
    g = (trace_gamma * 0.45 * torch.sigmoid(row_scales[:n_exc].float())).view(1, 1, 1, n_exc, 1)
    raw_exc_re, raw_exc_im = torch.stack(exc_re_list[:n_exc], dim=-2), torch.stack(exc_im_list[:n_exc], dim=-2)
    exc_sq = raw_exc_re.square().sum(dim=-1, keepdim=True) + raw_exc_im.square().sum(dim=-1, keepdim=True)
    inv_exc = torch.where(exc_sq > 1e-6, torch.rsqrt(exc_sq + DEFAULT_SCHUR_EPS), torch.zeros_like(exc_sq))
    r_exc_re, r_exc_im = raw_exc_re * inv_exc * g, raw_exc_im * inv_exc * g

    all_re = torch.cat([u0_re.unsqueeze(-2), r_exc_re], dim=-2).permute(0, 2, 1, 3, 4).contiguous()
    all_im = torch.cat([u0_im.unsqueeze(-2), r_exc_im], dim=-2).permute(0, 2, 1, 3, 4).contiguous()

    frob = torch.rsqrt(
        all_re.square().sum(dim=(-2, -1), keepdim=True) + all_im.square().sum(dim=(-2, -1), keepdim=True) + TRACE_NORM_EPS
    )
    return (all_re * frob).to(work_dtype), (all_im * frob).to(work_dtype)


def _build_variable_rank_cholesky_f_flat(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    r_re: torch.Tensor,
    r_im: torch.Tensor,
    v_mixed: torch.Tensor,
    high_mode_gates: torch.Tensor,
    r_modes: int,
) -> torch.Tensor:
    """Construct packed unit-Frobenius rank-$R$ Cholesky rows `f_flat` `(B, H, T, 2 * R * d_h)` (`<= 8` args)."""
    bsz, num_heads, seq_len, v_rank, d_h = r_re.shape
    work_dtype = r_re.dtype

    v_c = v_mixed.view(bsz, seq_len, num_heads, v_rank, 2, d_h).permute(0, 2, 1, 3, 4, 5)
    k_c = k_rot.view(bsz, seq_len, num_heads, v_rank, 2, d_h).permute(0, 2, 1, 3, 4, 5)
    q_c = q_rot.view(bsz, seq_len, num_heads, v_rank, 2, d_h).permute(0, 2, 1, 3, 4, 5)

    v_re, v_im = v_c[..., 0, :], v_c[..., 1, :]
    k_re, k_im = k_c[..., 0, :], k_c[..., 1, :]
    q_re, q_im = q_c[..., 0, :], q_c[..., 1, :]

    base_re_list: list[torch.Tensor] = [r_re, v_re, k_re, q_re]
    base_im_list: list[torch.Tensor] = [r_im, v_im, k_im, q_im]

    if r_modes > 4:
        kn = torch.rsqrt(k_re.float().square().sum(dim=-1, keepdim=True) + k_im.float().square().sum(dim=-1, keepdim=True) + DEFAULT_SCHUR_EPS).to(work_dtype)
        qn = torch.rsqrt(q_re.float().square().sum(dim=-1, keepdim=True) + q_im.float().square().sum(dim=-1, keepdim=True) + DEFAULT_SCHUR_EPS).to(work_dtype)
        ku_re, ku_im = k_re * kn, k_im * kn
        qu_re, qu_im = q_re * qn, q_im * qn
        amp = math.sqrt(float(2 * d_h)) * float(d_h) ** 0.5

        rk_re, rk_im = _complex_schur_product(r_re, r_im, ku_re, ku_im, amp)
        rq_re, rq_im = _complex_schur_product(r_re, r_im, qu_re, -qu_im, amp)

        z1 = torch.zeros_like(r_re[:, :, :1])
        r1_re = torch.cat([z1, r_re[:, :, :-1]], dim=2) if seq_len > 1 else torch.zeros_like(r_re)
        r1_im = torch.cat([z1, r_im[:, :, :-1]], dim=2) if seq_len > 1 else torch.zeros_like(r_im)
        bi_re, bi_im = _complex_schur_product(r_re, r_im, r1_re, r1_im, amp)
        r1_scaled_re, r1_scaled_im = r1_re * math.sqrt(float(2 * d_h)), r1_im * math.sqrt(float(2 * d_h))

        high_re_list: list[torch.Tensor] = [rk_re, rq_re, r1_scaled_re, bi_re]
        high_im_list: list[torch.Tensor] = [rk_im, rq_im, r1_scaled_im, bi_im]

        if r_modes > 8:
            z2 = torch.zeros_like(r_re[:, :, :2])
            r2_re = torch.cat([z2, r_re[:, :, :-2]], dim=2) if seq_len > 2 else torch.zeros_like(r_re)
            r2_im = torch.cat([z2, r_im[:, :, :-2]], dim=2) if seq_len > 2 else torch.zeros_like(r_im)
            r2_scaled_re, r2_scaled_im = r2_re * math.sqrt(float(2 * d_h)), r2_im * math.sqrt(float(2 * d_h))
            sk_re, sk_im = _complex_schur_product(r_re, r_im, r2_re, r2_im, amp)
            qk_re, qk_im = _complex_schur_product(qu_re, qu_im, ku_re, -ku_im, amp)
            tri_re, tri_im = _complex_schur_product(r1_re, r1_im, r2_re, r2_im, amp)
            high_re_list.extend([r2_scaled_re, sk_re, qk_re, tri_re])
            high_im_list.extend([r2_scaled_im, sk_im, qk_im, tri_im])

            for extra_idx in range(len(high_re_list), r_modes - 4):
                shift = (extra_idx - 7) % max(1, d_h - 1) + 1
                sh_re, sh_im = torch.roll(r_re, shifts=shift, dims=-1), torch.roll(r_im, shifts=shift, dims=-1)
                c_re, c_im = _complex_schur_product(sh_re, sh_im, ku_re, ku_im, amp)
                high_re_list.append(c_re)
                high_im_list.append(c_im)

        n_high = r_modes - 4
        g_high = (1.25 * torch.sigmoid(high_mode_gates[:n_high].float())).to(work_dtype).view(1, 1, 1, n_high, 1)
        h_re = torch.cat(high_re_list[:n_high], dim=-2) * g_high
        h_im = torch.cat(high_im_list[:n_high], dim=-2) * g_high
        f_all_re = torch.cat(base_re_list + [h_re], dim=-2)
        f_all_im = torch.cat(base_im_list[:4] + [h_im], dim=-2)
    else:
        f_all_re = torch.cat(base_re_list[:r_modes], dim=-2)
        f_all_im = torch.cat(base_im_list[:r_modes], dim=-2)

    if r_modes > 1 and r_modes != 4:
        gamma_tr = compute_rank_invariant_excited_trace_scale(r_modes)
        row_mult = torch.ones((1, 1, 1, r_modes, 1), device=r_re.device, dtype=work_dtype)
        row_mult[:, :, :, 1:, :] = float(gamma_tr)
        f_raw_re, f_raw_im = f_all_re * row_mult, f_all_im * row_mult
    else:
        f_raw_re, f_raw_im = f_all_re, f_all_im

    f_nsq = (
        (f_raw_re.float().square().sum(dim=(-2, -1), keepdim=True) + f_raw_im.float().square().sum(dim=(-2, -1), keepdim=True))
        .clamp_min(TRACE_NORM_EPS)
        .rsqrt()
        .to(work_dtype)
    )
    f_re, f_im = f_raw_re * f_nsq, f_raw_im * f_nsq
    return torch.stack([f_re, f_im], dim=-2).reshape(bsz, num_heads, seq_len, 2 * r_modes * d_h)


def fused_variable_rank_cholesky_ket_attention(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    r_re_im: tuple[torch.Tensor, torch.Tensor],
    v_mixed: torch.Tensor,
    high_mode_gates: torch.Tensor,
    phase_and_born: tuple[torch.Tensor | None, torch.Tensor | None, float, bool] = (None, None, 0.12, True),
    k_chol: int = DEFAULT_WINNING_CHOLESKY_RANK,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Single-pass fused Variable-Rank Cholesky factor $L_{t,h} \in \mathbb{C}^{R \times d_h}$ + ket attention (`<= 8` args)."""
    r_re, r_im = r_re_im
    bsz, num_heads, seq_len, v_rank, d_h = r_re.shape
    r_modes = max(1, int(k_chol))
    head_dim = q_rot.size(-1)
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
    return l_packed[..., 0, :], l_packed[..., 1, :], a_v[..., 0, :], a_v[..., 1, :]


class _VariableRankCholeskyWavepacketBlock(_NativeCholeskySchurWavepacketBlock):
    r"""Wavepacket Block that evolves and propagates an unrestricted rank-$R$ Cholesky factor $L_{t,h} \in \mathbb{C}^{R \times 32}$."""

    def __init__(
        self,
        embed_dim: int = DEFAULT_SHALLOW_EMBED_DIM,
        num_heads: int = DEFAULT_SHALLOW_NUM_HEADS,
        pkt_count: int = DEFAULT_SHALLOW_PKT_COUNT,
        pkt_dim: int = DEFAULT_SHALLOW_PKT_DIM,
        mlp_hidden: int = DEFAULT_SHALLOW_MLP_HIDDEN,
        init_schur_scale: float = DEFAULT_INIT_SCHUR_GATE_SCALE,
        num_cholesky_modes: int = DEFAULT_WINNING_CHOLESKY_RANK,
    ) -> None:
        super().__init__(embed_dim, num_heads, pkt_count, pkt_dim, mlp_hidden, init_schur_scale, uncollapsed_cholesky=True)
        self.num_cholesky_modes = max(1, int(num_cholesky_modes))
        self.chol_hwy_bias = nn.Parameter(torch.full((self.num_heads, self.num_cholesky_modes), 1.65, dtype=torch.float32))
        self.chol_hwy_scale = nn.Parameter(torch.full((self.num_heads, self.num_cholesky_modes), 1.25, dtype=torch.float32))
        n_high = max(1, self.num_cholesky_modes - 4)
        self.high_mode_gates = nn.Parameter(torch.full((n_high,), DEFAULT_HIGH_MODE_INIT_GATE, dtype=torch.float32))

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return scalar parameters including variable-rank highway and higher-order coincidence gates."""
        params = super().scalar_parameters()
        if self.num_cholesky_modes > 4:
            params.append(self.high_mode_gates)
        return params

    @torch.no_grad()
    def restrict_scalars_(self) -> None:
        """Clamp Schur and variable-rank highway scalars to stable bounds."""
        super().restrict_scalars_()
        self.high_mode_gates.clamp_(-3.5, 2.5)

    def _merge_uncollapsed_cholesky_highway(
        self, in_r: torch.Tensor, in_i: torch.Tensor, cur_r: torch.Tensor, cur_i: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge incoming rank-$R$ Cholesky factor `(in_r, in_i)` with `(cur_r, cur_i)` via phase-locked Born-Trace POVM gate."""
        work_dtype = cur_r.dtype
        mh_norm = torch.sqrt(
            self.m_hwy_r.float().square().sum(dim=-1, keepdim=True)
            + self.m_hwy_i.float().square().sum(dim=-1, keepdim=True)
            + DEFAULT_SCHUR_EPS
        )
        mhr = (self.m_hwy_r.float() / mh_norm).to(work_dtype).view(1, self.num_heads, 1, self.d_h, 1)
        mhi = (self.m_hwy_i.float() / mh_norm).to(work_dtype).view(1, self.num_heads, 1, self.d_h, 1)
        ov_r = torch.matmul(cur_r, mhr) + torch.matmul(cur_i, mhi)
        ov_i = torch.matmul(cur_i, mhr) - torch.matmul(cur_r, mhi)
        p_mode = (ov_r.float().square() + ov_i.float().square()).clamp(0.0, 1.0)
        b_h = self.chol_hwy_bias.float().view(1, self.num_heads, 1, self.num_cholesky_modes, 1)
        s_h = self.chol_hwy_scale.float().view(1, self.num_heads, 1, self.num_cholesky_modes, 1)
        alpha = torch.sigmoid(b_h + s_h * p_mode)
        w_old = torch.sqrt((1.0 - alpha).clamp_min(1e-6)).to(work_dtype)
        w_new = torch.sqrt(alpha.clamp_min(1e-6)).to(work_dtype)

        in_al_r, in_al_i = phase_lock_cholesky_modes(cur_r, cur_i, in_r, in_i)
        out_r = w_new * cur_r + w_old * in_al_r
        out_i = w_new * cur_i + w_old * in_al_i
        frob = (
            (out_r.float().square().sum(dim=(-2, -1), keepdim=True) + out_i.float().square().sum(dim=(-2, -1), keepdim=True))
            .clamp_min(TRACE_NORM_EPS)
            .rsqrt()
            .to(work_dtype)
        )
        return out_r * frob, out_i * frob

    def forward(
        self,
        x: torch.Tensor,
        x_anchor: torch.Tensor,
        cos_4d: torch.Tensor,
        sin_4d: torch.Tensor,
        in_cholesky: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Execute variable-rank $R$-mode Cholesky attention, Schur modulation, highway merge, and McWeeny Gram evolution."""
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
        l_re, l_im, v_bar_re, v_bar_im = fused_variable_rank_cholesky_ket_attention(
            q_rot, k_rot, (r_re, r_im), v_mixed_h, self.high_mode_gates,
            phase_and_born=(self.berry_qk_phase, self.born_qk_beta, self.sdpa_scale, True), k_chol=self.num_cholesky_modes,
        )

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


class VariableRankVacuumCholeskyGPT(NativeCholeskyDensityWavepacketGPT):
    r"""Unrestricted Rank-$R$ Uncollapsed Cholesky Density-Matrix GPT with Vacuum-Subspace Dark Parking (`R0..R6`)."""

    def __init__(self, r_spec: VariableRankCholeskySpec) -> None:
        nat_spec = NativeCholeskyDensitySpec(
            use_outer_product_input=False, uncollapsed_cholesky_highway=True, use_semantic_cholesky_readout=True,
            enable_kn_synergy_eval=False, init_schur_gate_scale=DEFAULT_INIT_SCHUR_GATE_SCALE,
            anchor_logit_bias=DEFAULT_CHOLESKY_ANCHOR_LOGIT_BIAS, mode_coherence_mix=DEFAULT_CHOLESKY_MODE_COHERENCE_MIX,
            seed=r_spec.seed,
        )
        super().__init__(nat_spec)
        self.r_spec = r_spec
        self.variant_mode = VariableRankVariantMode(r_spec.variant_mode)
        self.num_branches = max(1, int(r_spec.num_cholesky_modes))
        self.d_vac = 0 if self.variant_mode == VariableRankVariantMode.R0_RANK8_NO_VACUUM_SPHERE else max(0, int(r_spec.vacuum_dim))
        self.chol_rms_scale = compute_rank_adaptive_cholesky_rms_scale(self.num_branches, self.id_real_dim)

        self.sem_ngram = VacuumAwareNGramProjector(embed_dim=self.embed_dim, sem_dim=self.sem_real_dim)  # type: ignore[assignment]

        self.quantum_blocks = nn.ModuleList(
            [
                _VariableRankCholeskyWavepacketBlock(
                    self.embed_dim, self.num_heads, self.pkt_count, self.pkt_dim, self.mlp_hidden,
                    init_schur_scale=nat_spec.init_schur_gate_scale, num_cholesky_modes=self.num_branches,
                )
                for _ in range(self.num_layers)
            ]
        )

        r_m = self.num_branches
        n_exc = max(1, r_m - 1)
        prior_init = torch.zeros(r_m, dtype=torch.float32)
        prior_init[0] = float(nat_spec.anchor_logit_bias)
        self.chol_mode_log_prior = nn.Parameter(prior_init)
        if n_exc == 3:
            w_init = torch.tensor([0.50, 0.30, 0.20], dtype=torch.float32)
            b_init = torch.tensor([0.12, 0.10, 0.08], dtype=torch.float32)
        else:
            w_init = torch.linspace(0.55, 0.15, n_exc, dtype=torch.float32)
            b_init = torch.linspace(0.12, 0.06, n_exc, dtype=torch.float32)
        self.sem_chol_mode_weights = nn.Parameter(w_init)
        self.sem_chol_branch_gates = nn.Parameter(b_init)

        vac_d = max(1, self.d_vac)
        self.id_vac_mass = nn.Parameter(torch.full((vac_d,), float(r_spec.init_id_vac_mass), dtype=torch.float32))
        self.sem_vac_mass = nn.Parameter(torch.full((vac_d,), DEFAULT_INIT_SEM_VACUUM_MASS, dtype=torch.float32))

        m_id, m_exc = float(r_spec.init_id_vac_mass), float(r_spec.init_excited_chol_vac_mass)
        prog_rows: list[list[float]] = [[m_id * 0.5] * vac_d]
        for exc_i in range(max(1, r_m - 1)):
            frac = float(exc_i) / float(max(1, r_m - 2)) if r_m > 2 else 0.0
            slope = 0.30 if r_m == 4 else DEFAULT_PROG_VAC_SLOPE
            prog_rows.append([m_exc * (1.0 + slope * frac)] * vac_d)
        self.chol_id_vac_mass = nn.Parameter(torch.tensor(prog_rows[:r_m], dtype=torch.float32))
        self.chol_sem_vac_mass = nn.Parameter(torch.full((r_m, vac_d), DEFAULT_INIT_SEM_VACUUM_MASS * 1.5, dtype=torch.float32))

        self.id_vac_proj = nn.Linear(16, vac_d, bias=False)
        self.chol_vac_proj = nn.Linear(16, r_m * vac_d, bias=False)
        nn.init.zeros_(self.id_vac_proj.weight)
        nn.init.zeros_(self.chol_vac_proj.weight)

        base_l0 = [DEFAULT_LAYER0_CHOL_LAG1_SCALE, DEFAULT_LAYER0_CHOL_LAG2_SCALE, DEFAULT_LAYER0_CHOL_SCHUR_SCALE]
        while len(base_l0) < n_exc:
            base_l0.append(DEFAULT_LAYER0_HIGH_ROW_SCALE - 0.05 * float(len(base_l0) - 3))
        self.layer0_chol_row_scales = nn.Parameter(torch.tensor(base_l0[:n_exc], dtype=torch.float32))

        self.last_anchor_id_cos_vac: float = 1.0
        self.last_excited_id_cos_vac: float = 1.0
        self.last_core_excited_cos_vac: float = 1.0
        self.last_high_excited_cos_vac: float = 1.0
        self.last_sem_cos_vac: float = 1.0

    def vacuum_scalar_parameters(self) -> list[nn.Parameter]:
        """Return scalar and variable-rank vacuum parameters for `AdamW` optimization."""
        params = list(self.native_density_scalar_parameters())
        if self.d_vac > 0:
            params.extend([
                self.id_vac_mass, self.sem_vac_mass, self.chol_id_vac_mass, self.chol_sem_vac_mass,
                self.id_vac_proj.weight, self.chol_vac_proj.weight,
            ])
        if self.r_spec.use_layer0_input_cholesky and self.num_branches > 1:
            params.append(self.layer0_chol_row_scales)
        return params

    @torch.no_grad()
    def clamp_identity_dispersion_and_scalars_(self) -> None:
        """Keep Identity Packet 7 zero-dispersion and clamp variable-rank vacuum rest-mass coordinates."""
        super().clamp_identity_dispersion_and_scalars_()
        if self.d_vac > 0:
            self.id_vac_mass.clamp_(0.02, 0.32)
            self.sem_vac_mass.clamp_(0.01, 0.18)
            self.chol_id_vac_mass.clamp_(0.02, 0.75)
            self.chol_sem_vac_mass.clamp_(0.02, 0.45)
        self.layer0_chol_row_scales.clamp_(-3.5, 2.0)

    def _encode_multiscale_trunk(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]:
        r"""Encode `idx` `(B, T)` propagating uncollapsed rank-$R$ Cholesky factors $L_{t,h}^{(\ell)} \in \mathbb{C}^{R \times 32}$."""
        bsz, seq_len = idx.shape
        work_dtype = torch.bfloat16 if (idx.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
        tok_wp = F.embedding(idx, self.wte_3d.view(self.vocab_size, self.embed_dim)).view(bsz, seq_len, self.pkt_count, self.pkt_dim)
        raw_wp = tok_wp.to(work_dtype)
        if seq_len > 1:
            raw_wp = self._apply_dispersion(raw_wp)

        raw_flat = raw_wp.reshape(bsz, seq_len, self.embed_dim)
        if seq_len > 1:
            s_gate = self.smear_lambda.to(work_dtype) * torch.sigmoid(
                F.linear(raw_flat[:, 1:, : self.smear_in_ch], self.smear_gate.weight.to(work_dtype))
            )
            sem_curr = raw_flat[:, 1:, : self.sem_real_dim] + s_gate * raw_flat[:, :-1, : self.sem_real_dim]
            sem_full = torch.cat([raw_flat[:, :1, : self.sem_real_dim], sem_curr], dim=1)
            raw_flat = torch.cat([sem_full, raw_flat[:, :, self.sem_real_dim :]], dim=-1)

        readout_sem_448: torch.Tensor | None = None
        readout_id_64: torch.Tensor | None = None
        if self.sem_ngram is not None:
            in_sem_delta, readout_sem_448, readout_id_64 = self.sem_ngram(idx, work_dtype)
            raw_flat = raw_flat + in_sem_delta

        x_0 = F.rms_norm(raw_flat, (self.embed_dim,), eps=1e-6)

        chol_state: tuple[torch.Tensor, torch.Tensor] | None = None
        if self.r_spec.use_layer0_input_cholesky:
            blk0: _VariableRankCholeskyWavepacketBlock = self.quantum_blocks[0]  # type: ignore[assignment]
            v0_wp = blk0.v_proj(x_0).view(bsz, seq_len, self.num_heads, 2, self.pkt_dim)
            if self.training and v0_wp.requires_grad:
                chol_state = torch.utils.checkpoint.checkpoint(
                    build_layer0_variable_rank_cholesky_factor,
                    v0_wp, self.wp_causal_re, self.wp_causal_im, self.layer0_chol_row_scales, self.num_branches,
                    use_reentrant=False,
                )
            else:
                chol_state = build_layer0_variable_rank_cholesky_factor(
                    v0_wp, self.wp_causal_re, self.wp_causal_im, self.layer0_chol_row_scales, self.num_branches
                )

        x = x_0
        cos_4d, sin_4d = self.rope.get_cos_sin(seq_len, dtype=work_dtype)
        for q_blk in self.quantum_blocks:
            x = q_blk(x, x_0, cos_4d, sin_4d, in_cholesky=chol_state)  # type: ignore[call-arg]
            chol_state = q_blk.last_cholesky_pre_u  # type: ignore[assignment]
            q_blk.last_cholesky_pre_u = None  # type: ignore[assignment]

        h_pre = F.rms_norm(x, (self.embed_dim,), eps=1e-6)
        return h_pre, readout_sem_448, readout_id_64, chol_state

    def encode_pre_bridge_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        """Encode `idx` `(B, T)` using the variable-rank Cholesky trunk so Step-0 Procrustes calibration matches runtime."""
        h_pre, _, _, _ = self._encode_multiscale_trunk(idx)
        return h_pre

    def build_multibranch_readout_tensors(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Build $R$ Cholesky readout modes on the Vacuum-Augmented Sphere $\mathcal{H}_{\text{vocab}} \oplus \mathcal{H}_{\text{vac}}$."""
        h_pre, readout_sem_448, readout_id_64, final_chol = self._encode_multiscale_trunk(idx)
        bsz, seq_len, _ = h_pre.shape
        r_m = self.num_branches
        work_dtype = h_pre.dtype
        h_bridge = self.exit_bridge(h_pre)

        ctx_16 = h_pre[:, :, :16]
        if self.d_vac > 0:
            u_id_vac_0 = self.id_vac_mass.to(work_dtype).view(1, 1, self.d_vac) * (
                2.0 * torch.sigmoid(F.linear(ctx_16, self.id_vac_proj.weight.to(work_dtype)))
            )
            u_sem_vac_0 = self.sem_vac_mass.to(work_dtype).view(1, 1, self.d_vac).expand(bsz, seq_len, self.d_vac)
        else:
            u_id_vac_0 = torch.zeros((bsz, seq_len, 0), device=idx.device, dtype=work_dtype)
            u_sem_vac_0 = torch.zeros((bsz, seq_len, 0), device=idx.device, dtype=work_dtype)

        h_sem_tr = F.rms_norm(h_bridge[:, :, : self.sem_real_dim], (self.sem_real_dim,), eps=1e-6)
        h_id_tr = F.rms_norm(h_bridge[:, :, self.sem_real_dim :], (self.id_real_dim,), eps=1e-6)

        if readout_sem_448 is not None and readout_id_64 is not None:
            g_sem, g_id = self.trunk_sem_gate.to(work_dtype), self.trunk_id_gate.to(work_dtype)
            raw_sem_0, raw_id_0 = g_sem * h_sem_tr + readout_sem_448, g_id * h_id_tr + readout_id_64
        else:
            raw_sem_0, raw_id_0 = h_sem_tr, h_id_tr

        h_sem_0, _, sem_cos_0 = vacuum_sphere_rms_norm(raw_sem_0, u_sem_vac_0)
        h_id_0, h_id_0_vac, id_cos_0 = vacuum_sphere_rms_norm(raw_id_0, u_id_vac_0)
        self.last_anchor_id_cos_vac = float(id_cos_0.detach().mean().item())
        self.last_sem_cos_vac = float(sem_cos_0.detach().mean().item())

        sem_scale = (self.temp_sem_scale / float(self.sem_real_dim)).to(work_dtype)
        id_scale_scalar = (self.temp_id_scale / float(self.id_real_dim)).to(work_dtype)

        h_sem_0_scaled = h_sem_0 * sem_scale
        c0_batch = self.carrier_c0.to(work_dtype).view(1, 1, DEFAULT_CARRIER_BIAS_DIM).expand(bsz, seq_len, DEFAULT_CARRIER_BIAS_DIM)
        h_base_464 = torch.cat([h_sem_0_scaled, c0_batch], dim=-1).contiguous()

        w_dir = (self.w_det_512 * self.radial_scale.unsqueeze(1)).to(work_dtype)
        b_col = (self.prior_gate * self.unigram_bias).to(work_dtype).unsqueeze(1) * self.carrier_c0.to(work_dtype).unsqueeze(0)
        w_aug = torch.cat([w_dir, b_col], dim=-1).contiguous()

        if r_m == 1:
            self.last_excited_id_cos_vac = self.last_anchor_id_cos_vac
            self.last_core_excited_cos_vac = self.last_anchor_id_cos_vac
            self.last_high_excited_cos_vac = self.last_anchor_id_cos_vac
            h_id_branches = (h_id_0.unsqueeze(-2) * id_scale_scalar).contiguous()
            h_sem_branches = torch.zeros_like(h_sem_0)
            z_ctrl = torch.zeros((bsz, seq_len, 1), device=idx.device, dtype=work_dtype)
            o_ctrl = torch.ones((bsz, seq_len, 1), device=idx.device, dtype=work_dtype)
            branch_controls = torch.stack([z_ctrl, o_ctrl, z_ctrl], dim=-1)
            return h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug

        last_blk: _VariableRankCholeskyWavepacketBlock = self.quantum_blocks[-1]  # type: ignore[assignment]
        assert final_chol is not None
        mod_l_r, mod_l_i = final_chol
        u_r_t = last_blk.u_evol_re.to(work_dtype).view(1, self.num_heads, 1, last_blk.d_h, last_blk.d_h).transpose(-2, -1)
        u_i_t = last_blk.u_evol_im.to(work_dtype).view(1, self.num_heads, 1, last_blk.d_h, last_blk.d_h).transpose(-2, -1)
        l_all_re = torch.matmul(mod_l_r, u_r_t) - torch.matmul(mod_l_i, u_i_t)
        l_all_im = torch.matmul(mod_l_r, u_i_t) + torch.matmul(mod_l_i, u_r_t)

        l7_re, l7_im = l_all_re[:, -1], l_all_im[:, -1]
        id_energy = (l7_re.float().square().sum(dim=-1) + l7_im.float().square().sum(dim=-1)).clamp_min(1e-6)
        l_sem_re, l_sem_im = l_all_re[:, : self.sem_complex_pkts], l_all_im[:, : self.sem_complex_pkts]
        sem_energy = (l_sem_re.float().square().sum(dim=-1) + l_sem_im.float().square().sum(dim=-1)).mean(dim=1).clamp_min(1e-6)
        w_sem = torch.sigmoid(self.sem_energy_mix_logit.float())
        mode_energy = (1.0 - w_sem) * id_energy + w_sem * sem_energy
        mode_log_weights = torch.log(mode_energy) + self.chol_mode_log_prior.view(1, 1, r_m)

        # Scale by alpha_chol(R) = sqrt(R * 64) so MS = R * E_{t,r} across any rank R!
        l7_64_scaled = torch.stack([l7_re, l7_im], dim=-2).reshape(bsz, seq_len, r_m, self.id_real_dim) * self.chol_rms_scale
        if self.d_vac > 0:
            u_chol_vac = self.chol_id_vac_mass.to(work_dtype).view(1, 1, r_m, self.d_vac) * (
                2.0 * torch.sigmoid(F.linear(ctx_16, self.chol_vac_proj.weight.to(work_dtype)).view(bsz, seq_len, r_m, self.d_vac))
            )
        else:
            u_chol_vac = torch.zeros((bsz, seq_len, r_m, 0), device=idx.device, dtype=work_dtype)
        l7_vocab, l7_vac, _ = vacuum_sphere_rms_norm(l7_64_scaled, u_chol_vac)

        coh = torch.sigmoid(self.chol_mode_coherence).to(work_dtype)
        mix_exc_vocab = coh * h_id_0.unsqueeze(-2) + (1.0 - coh) * l7_vocab[:, :, 1:, :]
        mix_exc_vac = coh * h_id_0_vac.unsqueeze(-2) + (1.0 - coh) * l7_vac[:, :, 1:, :]
        excited_modes, _, exc_cos = vacuum_sphere_rms_norm(mix_exc_vocab, mix_exc_vac)
        self.last_excited_id_cos_vac = float(exc_cos.detach().mean().item())
        n_core_exc = min(3, r_m - 1)
        self.last_core_excited_cos_vac = float(exc_cos[:, :, :n_core_exc].detach().mean().item())
        self.last_high_excited_cos_vac = (
            float(exc_cos[:, :, 3:].detach().mean().item()) if (r_m - 1) > 3 else self.last_core_excited_cos_vac
        )

        cholesky_id_branches = torch.cat([h_id_0.unsqueeze(-2), excited_modes], dim=-2)
        h_id_branches = (cholesky_id_branches * id_scale_scalar).contiguous()

        n_exc = r_m - 1
        b_exc = (0.35 * torch.tanh(self.sem_chol_branch_gates[:n_exc].float())).view(1, 1, n_exc).expand(bsz, seq_len, n_exc)
        sem1_gates = torch.cat([torch.zeros((bsz, seq_len, 1), device=idx.device, dtype=torch.float32), b_exc], dim=-1)

        sem_exc_raw = (
            torch.stack([l_sem_re[:, :, :, 1:, :], l_sem_im[:, :, :, 1:, :]], dim=-2)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(bsz, seq_len, n_exc, self.sem_real_dim)
            * self.chol_rms_scale
        )
        w_m = F.softmax(self.sem_chol_mode_weights[:n_exc].float(), dim=0).to(work_dtype).view(1, 1, n_exc, 1)
        sem_super = (sem_exc_raw * w_m).sum(dim=2)
        if self.r_spec.use_independent_rmode_512d:
            s0_f = h_sem_0.float().unsqueeze(-2)
            s0_norm_sq = s0_f.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
            exc_sem_f = (coh * sem_super.unsqueeze(-2) + (1.0 - coh) * sem_exc_raw).float()
            proj_c = (exc_sem_f * s0_f).sum(dim=-1, keepdim=True) / s0_norm_sq
            ortho_sem_exc = (exc_sem_f - proj_c * s0_f).to(work_dtype)
            u_sem_exc_vac = (
                self.chol_sem_vac_mass[1:r_m].to(work_dtype).view(1, 1, n_exc, self.d_vac).expand(bsz, seq_len, n_exc, self.d_vac)
                if self.d_vac > 0
                else torch.zeros((bsz, seq_len, n_exc, 0), device=idx.device, dtype=work_dtype)
            )
            norm_sem_exc, _, _ = vacuum_sphere_rms_norm(ortho_sem_exc, u_sem_exc_vac)
            zero_sem_0 = torch.zeros((bsz, seq_len, 1, self.sem_real_dim), device=idx.device, dtype=work_dtype)
            h_sem_branches = (torch.cat([zero_sem_0, norm_sem_exc], dim=-2) * sem_scale).contiguous()
        else:
            u_f, s0_f_2d = sem_super.float(), h_sem_0.float()
            proj_c = (u_f * s0_f_2d).sum(dim=-1, keepdim=True) / s0_f_2d.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
            ortho_sem_1 = (u_f - proj_c * s0_f_2d).to(work_dtype)
            u_sem1_vac = (
                self.chol_sem_vac_mass[1].to(work_dtype).view(1, 1, self.d_vac).expand(bsz, seq_len, self.d_vac)
                if self.d_vac > 0
                else torch.zeros((bsz, seq_len, 0), device=idx.device, dtype=work_dtype)
            )
            h_sem_1, _, _ = vacuum_sphere_rms_norm(ortho_sem_1, u_sem1_vac)
            h_sem_branches = (h_sem_1 * sem_scale).contiguous()

        id_ones = torch.ones_like(mode_log_weights)
        branch_controls = torch.stack([sem1_gates, id_ones, mode_log_weights], dim=-1).to(work_dtype)
        return h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug

    def forward_loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Variable-Rank Vacuum-Subspace Cholesky cross-entropy loss."""
        h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug = self.build_multibranch_readout_tensors(idx)
        return _VacuumFourMode512DWavepacketCEFn.apply(h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug, targets)


@torch.no_grad()
def initialize_and_calibrate_variable_rank_cholesky_model_(
    model: VariableRankVacuumCholeskyGPT, stream: LongSeqFineWebStream
) -> tuple[float, dict[str, float]]:
    r"""Initialize `(50304, 8, 2, 32)` wavepackets from Shards `1..9` and calibrate Step-0 Variable-Rank Readout (`<= 8` args)."""
    device = stream.device
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    sketch_spec = SupportSketchSpec(
        mode=SupportSketchMode.I17_SHARDS1_TO_9_900M_STEP0_CALIBRATED,
        vocab_size=model.vocab_size, pkt_count=model.pkt_count, pkt_dim=model.pkt_dim, seed=model.spec.seed,
    )
    psi_4d, log_prior, radial, meta = compute_multishard_multilag_complex_wavefunction(
        sketch_spec, num_shards=model.spec.num_init_shards, device=device, rev_token_id=RAW_GPT2_VOCAB_SIZE
    )
    model.wte_3d.copy_(psi_4d.view(model.vocab_size, model.pkt_count, model.pkt_dim))
    model.unigram_bias.copy_(log_prior)
    model.radial_scale.copy_(radial)
    model.prior_gate.fill_(1.0)
    del psi_4d, log_prior, radial

    init_disp = torch.tensor([0.22, 0.22, 0.14, 0.08, 0.22, 0.22, 0.14, 0.00], device=device, dtype=torch.float32)
    model.wp_causal_re.copy_(init_disp)
    model.wp_causal_im.zero_()

    for q_blk in model.quantum_blocks:
        q_blk.obs_scale.fill_(0.45)  # type: ignore[union-attr]
        q_blk.mz_mlp.mz_scale.fill_(0.45)  # type: ignore[union-attr]

    wte_flat = model.wte_3d.detach().float().view(model.vocab_size, model.embed_dim)
    p0 = F.softmax(model.unigram_bias.detach().float(), dim=0).unsqueeze(1)
    sem_raw, id_raw = wte_flat[:, : model.sem_real_dim], wte_flat[:, model.sem_real_dim :]
    active_sem_mask = (sem_raw.norm(dim=-1, keepdim=True) > 1e-3).float()
    mean_sem = (p0 * sem_raw).sum(dim=0, keepdim=True)
    centered_sem = F.normalize(sem_raw - active_sem_mask * mean_sem, dim=-1, eps=1e-6)
    centered_id = F.normalize(id_raw - (p0 * id_raw).sum(dim=0, keepdim=True), dim=-1, eps=1e-6)

    target_sem_rms = centered_sem * math.sqrt(float(model.sem_real_dim))
    target_id_rms = centered_id * math.sqrt(float(model.id_real_dim))
    model.w_det_512.copy_(torch.cat([target_sem_rms, target_id_rms], dim=-1))
    del wte_flat, p0, sem_raw, id_raw, active_sem_mask, mean_sem, centered_sem, centered_id, target_sem_rms, target_id_rms
    target_table = model.w_det_512.detach().float() * model.radial_scale.detach().float().unsqueeze(1)

    if model.sem_ngram is not None:
        if isinstance(model.sem_ngram, VacuumAwareNGramProjector):
            model.sem_ngram.calibrate_from_training_shards_(stream, target_table)
        else:
            model.sem_ngram.calibrate_from_semantic_wavepackets_(stream.shard1_gpu, target_table)

    d_emb = model.embed_dim
    cov_hh = torch.zeros((d_emb, d_emb), device=device, dtype=torch.float32)
    cov_hy = torch.zeros((d_emb, d_emb), device=device, dtype=torch.float32)
    cal_batches: list[tuple[torch.Tensor, torch.Tensor]] = []

    was_training = model.training
    model.eval()
    n_cal = DEFAULT_STEP0_PROCRUSTES_BATCHES
    max_start = stream.shard1_gpu.numel() - stream.seq_len - 2
    stride = max(1, max_start // (n_cal * stream.batch_size))

    for b_idx in range(n_cal):
        base = b_idx * stream.batch_size * stride
        starts = (base + torch.arange(stream.batch_size, device=device) * stride) % max_start
        offsets = torch.arange(stream.seq_len + 1, device=device).unsqueeze(0)
        chunk = stream.shard1_gpu[starts.unsqueeze(1) + offsets]
        idx_in, targets = chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()
        if b_idx < 2:
            cal_batches.append((idx_in, targets))

        h_flat = model.encode_pre_bridge_hidden(idx_in).reshape(-1, d_emb).float()
        y_flat = target_table[targets.reshape(-1)]
        cov_hh.add_(torch.matmul(h_flat.t(), h_flat))
        cov_hy.add_(torch.matmul(h_flat.t(), y_flat))

    n_samples = float(n_cal * stream.batch_size * stream.seq_len)
    cov_hh.div_(n_samples)
    cov_hy.div_(n_samples)

    reg_eye = DEFAULT_STEP0_PROCRUSTES_LAMBDA * torch.eye(d_emb, device=device, dtype=torch.float32)
    w_star = torch.linalg.solve(cov_hh + reg_eye, cov_hy)
    u_svd, s_svd, vh_svd = torch.linalg.svd(w_star.t(), full_matrices=False)
    s_rms = s_svd.square().mean().sqrt().clamp_min(1e-6)
    s_norm = (s_svd / s_rms).clamp(0.05, 5.0)

    model.exit_bridge.u_stiefel.copy_(u_svd.contiguous())
    model.exit_bridge.v_stiefel.copy_(vh_svd.t().contiguous())
    model.exit_bridge.sigma.copy_(s_norm.contiguous())
    del cov_hh, cov_hy, w_star, u_svd, s_svd, vh_svd, target_table

    if model.d_vac > 0:
        best_sem_t, best_id_t, best_vac_m, best_loss = 5.5, 0.8, 0.14, float("inf")
        for cand_vac in (0.08, 0.14, 0.22):
            model.id_vac_mass.fill_(cand_vac)
            for cand_sem in (3.5, 4.8, 5.5, 7.0):
                for cand_id in (0.6, 1.0, 1.5, 2.2):
                    model.temp_sem_scale.fill_(cand_sem)
                    model.temp_id_scale.fill_(cand_id)
                    tot_l = sum(float(model.forward_loss(x_b, y_b).item()) for x_b, y_b in cal_batches)
                    if tot_l < best_loss:
                        best_loss, best_sem_t, best_id_t, best_vac_m = tot_l, cand_sem, cand_id, cand_vac
        model.id_vac_mass.fill_(best_vac_m)
        model.temp_sem_scale.fill_(best_sem_t)
        model.temp_id_scale.fill_(best_id_t)
    else:
        best_sem_t, best_id_t, best_loss = 5.5, 0.6, float("inf")
        for cand_sem in (3.5, 4.8, 5.5, 7.0):
            for cand_id in (0.6, 1.0, 1.5, 2.2):
                model.temp_sem_scale.fill_(cand_sem)
                model.temp_id_scale.fill_(cand_id)
                tot_l = sum(float(model.forward_loss(x_b, y_b).item()) for x_b, y_b in cal_batches)
                if tot_l < best_loss:
                    best_loss, best_sem_t, best_id_t = tot_l, cand_sem, cand_id
        model.temp_sem_scale.fill_(best_sem_t)
        model.temp_id_scale.fill_(best_id_t)

    del cal_batches
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if was_training:
        model.train()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms, meta


@torch.no_grad()
def evaluate_variable_rank_cholesky_model(
    model: VariableRankVacuumCholeskyGPT, stream: LongSeqFineWebStream
) -> dict[str, float]:
    """Evaluate `VariableRankVacuumCholeskyGPT` across unseen Shard-0 validation batches (`65,536` tokens, 100% Pure Neural)."""
    was_training = model.training
    model.eval()

    total_neural_loss, branch_entropy_sum = 0.0, 0.0
    anchor_cos_sum, excited_cos_sum, core_exc_cos_sum, high_exc_cos_sum = 0.0, 0.0, 0.0, 0.0
    total_top1, total_top5, total_tokens, n_batches = 0, 0, 0, 0
    cap, inv_cap = float(DEFAULT_SOFTCAP_LOGIT), 1.0 / float(DEFAULT_SOFTCAP_LOGIT)

    for idx_in, targets in stream.val_batches:
        bsz, seq_len = idx_in.shape
        n_tok = int(targets.numel())
        h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug_528 = model.build_multibranch_readout_tensors(idx_in)

        num_r, v_sz = h_id_branches.size(-2), w_aug_528.size(0)
        sem_dim, id_dim = h_sem_branches.size(-1), h_id_branches.size(-1)
        car_dim = h_base_464.size(-1) - sem_dim
        indep_sem = h_sem_branches.ndim == 4

        w_sem, w_id = w_aug_528[:, :sem_dim], w_aug_528[:, sem_dim : sem_dim + id_dim]
        w_car = w_aug_528[:, sem_dim + id_dim : sem_dim + id_dim + car_dim]
        w_base_t = torch.cat([w_sem, w_car], dim=-1).t().contiguous()
        w_sem_t, w_id_t = w_sem.t().contiguous(), w_id.t().contiguous()

        sem1_gates, id_scales, gate_logits = branch_controls[:, :, :, 0], branch_controls[:, :, :, 1], branch_controls[:, :, :, 2]

        pi_b = F.softmax(gate_logits.float(), dim=-1)
        ent_b = -(pi_b * torch.log(pi_b.clamp_min(1e-8))).sum(dim=-1).mean()
        branch_entropy_sum += float(ent_b.item())
        anchor_cos_sum += float(model.last_anchor_id_cos_vac)
        excited_cos_sum += float(model.last_excited_id_cos_vac)
        core_exc_cos_sum += float(model.last_core_excited_cos_vac)
        high_exc_cos_sum += float(model.last_high_excited_cos_vac)
        n_batches += 1

        has_sem1 = bool(sem1_gates.abs().max().item() > 1e-8)
        t_step = max(8, (32 * 4) // max(1, num_r))
        for t_start in range(0, seq_len, t_step):
            t_end = min(t_start + t_step, seq_len)
            c_len = bsz * (t_end - t_start)
            hb_c = h_base_464[:, t_start:t_end].reshape(c_len, sem_dim + car_dim)
            hid_c = h_id_branches[:, t_start:t_end].reshape(c_len * num_r, id_dim)
            b_c = sem1_gates[:, t_start:t_end].reshape(c_len, num_r).float()
            t_c = id_scales[:, t_start:t_end].reshape(c_len, num_r).float()
            a_c = gate_logits[:, t_start:t_end].reshape(c_len, num_r).float()
            y_c = targets[:, t_start:t_end].reshape(c_len)

            z_base = torch.matmul(hb_c, w_base_t).float()
            raw_r = torch.matmul(hid_c, w_id_t).float().view(c_len, num_r, v_sz)
            raw_r.mul_(t_c.unsqueeze(-1)).add_(z_base.unsqueeze(1))
            del z_base
            if has_sem1:
                if indep_sem:
                    hs_c = h_sem_branches[:, t_start:t_end].reshape(c_len * num_r, sem_dim)
                    z_sem1 = torch.matmul(hs_c, w_sem_t).float().view(c_len, num_r, v_sz)
                    raw_r.addcmul_(b_c.unsqueeze(-1), z_sem1)
                else:
                    hs_c = h_sem_branches[:, t_start:t_end].reshape(c_len, sem_dim)
                    z_sem1 = torch.matmul(hs_c, w_sem_t).float()
                    raw_r.addcmul_(b_c.unsqueeze(-1), z_sem1.unsqueeze(1))
                del z_sem1
            raw_r.mul_(inv_cap).tanh_().mul_(cap)

            log_pi = F.log_softmax(a_c, dim=-1)
            log_p_r = F.log_softmax(raw_r, dim=-1)
            del raw_r
            log_p_mix = torch.logsumexp(log_pi.unsqueeze(-1) + log_p_r, dim=1)
            del log_p_r

            nll_neural = -log_p_mix.gather(dim=-1, index=y_c.unsqueeze(1)).squeeze(1).sum()
            total_neural_loss += float(nll_neural.item())

            top5_preds = log_p_mix.topk(5, dim=-1).indices
            total_top1 += int((top5_preds[:, 0] == y_c).sum().item())
            total_top5 += int((top5_preds == y_c.unsqueeze(1)).any(dim=-1).sum().item())
            del log_p_mix
        del h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug_528
        total_tokens += n_tok

    if stream.device.type == "cuda":
        torch.cuda.empty_cache()
    if was_training:
        model.train()

    mean_ce = total_neural_loss / float(max(1, total_tokens))
    mean_ent = branch_entropy_sum / float(max(1, n_batches))
    denom = float(max(1, n_batches))
    return {
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


def build_variable_rank_cholesky_optimizers(
    model: VariableRankVacuumCholeskyGPT,
) -> tuple[
    ModularRiemannianStiefelKrausMuon,
    BatchSubspaceTikhonovGramMuon,
    BatchSubspaceTikhonovGramMuon,
    torch.optim.AdamW,
]:
    """Build Riemannian Stiefel, Vocab Gram-Muon, and AdamW optimizers for `VariableRankVacuumCholeskyGPT` (`<= 8` args)."""
    dev = model.wte_3d.device
    stiefel_params: list[nn.Parameter] = list(model.exit_bridge.real_stiefel_parameters())
    unitary_pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
    for q_blk in model.quantum_blocks:
        stiefel_params.extend(q_blk.real_stiefel_parameters())  # type: ignore[operator]
        unitary_pairs.extend(q_blk.complex_unitary_pairs())  # type: ignore[operator]

    fast_temp_params = [model.temp_sem_scale, model.temp_id_scale, model.trunk_sem_gate, model.trunk_id_gate]
    scalar_params = [
        model.wp_causal_re, model.wp_causal_im, model.smear_gate.weight, model.smear_lambda,
        model.radial_scale, model.unigram_bias, model.prior_gate,
    ]
    scalar_params.extend(model.exit_bridge.scalar_parameters())
    scalar_params.extend(model.vacuum_scalar_parameters())
    for q_blk in model.quantum_blocks:
        scalar_params.extend(q_blk.scalar_parameters())  # type: ignore[operator]

    q_trunk_opt = ModularRiemannianStiefelKrausMuon(
        real_stiefel_params=stiefel_params, complex_operator_pairs=unitary_pairs,
        stiefel_lr=DEFAULT_SHALLOW_STIEFEL_LR, unitary_lr=DEFAULT_SHALLOW_UNITARY_LR,
        tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA, momentum_mode=StiefelMomentumMode.RIEMANNIAN_TANGENT_SNR_GATED,
    )
    wte_opt = BatchSubspaceTikhonovGramMuon([model.wte_3d], lr=DEFAULT_SHALLOW_WTE_LR, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA)
    head_opt = BatchSubspaceTikhonovGramMuon([model.w_det_512], lr=DEFAULT_SHALLOW_READOUT_LR, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA)

    adam_groups: list[dict[str, Any]] = [
        {"params": scalar_params, "lr": DEFAULT_SHALLOW_SCALAR_LR, "weight_decay": 0.0},
        {"params": fast_temp_params, "lr": 0.06, "weight_decay": 0.0},
    ]
    if model.sem_ngram is not None:
        ngram_tables = [
            model.sem_ngram.unigram_next_table, model.sem_ngram.bigram_table,
            model.sem_ngram.trigram_table, model.sem_ngram.skipgram_table, model.sem_ngram.u_sem,
        ]
        ngram_scalars = [
            model.sem_ngram.in_scale, model.sem_ngram.readout_sem_scale,
            model.sem_ngram.readout_id_scale, model.sem_ngram.order_logits,
        ]
        if isinstance(model.sem_ngram, VacuumAwareNGramProjector):
            ngram_scalars.append(model.sem_ngram.purity_gate_gamma)
        adam_groups.append({"params": ngram_tables, "lr": DEFAULT_SHALLOW_NGRAM_LR, "weight_decay": 0.0})
        adam_groups.append({"params": ngram_scalars, "lr": DEFAULT_SHALLOW_SCALAR_LR, "weight_decay": 0.0})
    adam_opt = torch.optim.AdamW(adam_groups, betas=(0.80, 0.95), eps=1e-8, fused=(dev.type == "cuda"))
    return q_trunk_opt, wte_opt, head_opt, adam_opt


def _train_and_eval_variable_rank_variants(
    stream: LongSeqFineWebStream,
    variants: list[tuple[str, VariableRankCholeskySpec, str]],
    steps: int,
    eval_interval: int,
    wave_id: str = "W59",
) -> list[dict[str, Any]]:
    """Run training and evaluation loop across `VariableRankCholeskySpec` variants (`<= 8` args)."""
    device = stream.device
    results: list[dict[str, Any]] = []
    tokens_per_step = stream.batch_size * stream.seq_len

    for variant_id, r_spec, desc in variants:
        torch.manual_seed(r_spec.seed)
        model = VariableRankVacuumCholeskyGPT(r_spec).to(device=device)
        init_ms, _ = initialize_and_calibrate_variable_rank_cholesky_model_(model, stream)
        model.clamp_identity_dispersion_and_scalars_()

        run_key = f"plan079_{variant_id.lower()}_{steps}steps"
        intent = ExperimentIntentSpec(
            run_key=run_key, wave_id=wave_id, plan_id="EXPERIMENT_PLAN_079",
            task_domain="fineweb_speedrun", hypothesis_intent=desc,
            architecture_summary=f"VariableRankVacuumCholeskyGPT(L=4, d=512, R={r_spec.num_cholesky_modes}, d_vac={model.d_vac}, mode={r_spec.variant_mode.value})",
            hardware_budget="1x RTX 5060 Ti 16GB", batch_size=stream.batch_size, seq_len=stream.seq_len,
            total_steps=steps, target_tokens_M=round((steps * tokens_per_step) / 1e6, 3),
        )
        exp_logger = StandardizedExperimentLogger(intent)
        q_trunk_opt, wte_opt, head_opt, adam_opt = build_variable_rank_cholesky_optimizers(model)

        step0_eval = evaluate_variable_rank_cholesky_model(model, stream)
        step0_eval["step"] = 0
        step0_eval["tokens_M"] = 0.0
        exp_logger.log_eval(0, {**step0_eval, "top1_acc": round(100.0 * step0_eval["top1_acc"], 2), "top5_acc": round(100.0 * step0_eval["top5_acc"], 2)})
        print(
            f"[{variant_id} (R={r_spec.num_cholesky_modes}, d_vac={model.d_vac})] Step    0 | "
            f"Pure Neural Val CE: {step0_eval['val_ce']:.4f} (PPL {step0_eval['val_ppl']:.2f}) | "
            f"Top1: {step0_eval['top1_acc']*100:.2f}% | Top5: {step0_eval['top5_acc']*100:.2f}% | "
            f"Modes: {step0_eval['effective_active_branches']:.2f}/{r_spec.num_cholesky_modes} | "
            f"cos(vac) [anc/core/high]: {step0_eval['mean_anchor_id_cos_theta_vac']:.3f}/"
            f"{step0_eval['mean_core_excited_cos_theta_vac']:.3f}/{step0_eval['mean_high_excited_cos_theta_vac']:.3f}",
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
            idx_in, targets = stream.sample_train_batch(step)
            q_trunk_opt.zero_grad(set_to_none=True)
            wte_opt.zero_grad(set_to_none=True)
            head_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)

            loss = model.forward_loss(idx_in, targets)
            loss.backward()

            mom_beta = compute_riemannian_momentum_beta(step, steps, QuantumMomentumScheduleMode.LOW_STEADY_MOMENTUM_070)
            q_trunk_opt.set_momentum(mom_beta)
            q_trunk_opt.set_lr_scale(lr_mult)
            q_trunk_opt.step()

            active_ids = torch.unique(torch.cat([idx_in.reshape(-1), targets.reshape(-1)]))
            wte_opt.set_momentum(mom_beta)
            wte_opt.step_active(active_ids, lr_scale=lr_mult)
            head_opt.set_momentum(mom_beta)
            head_opt.step_full(lr_scale=lr_mult, active_idx=active_ids)

            adam_opt.param_groups[0]["lr"] = DEFAULT_SHALLOW_SCALAR_LR * lr_mult
            adam_opt.param_groups[1]["lr"] = 0.06 * lr_mult
            if len(adam_opt.param_groups) > 2:
                adam_opt.param_groups[2]["lr"] = DEFAULT_SHALLOW_NGRAM_LR * lr_mult
                adam_opt.param_groups[3]["lr"] = DEFAULT_SHALLOW_SCALAR_LR * lr_mult
            adam_opt.step()
            model.clamp_identity_dispersion_and_scalars_()

            if step % eval_interval == 0 or step == steps:
                ev = evaluate_variable_rank_cholesky_model(model, stream)
                ev["step"] = step
                ev["tokens_M"] = round((step * tokens_per_step) / 1e6, 3)
                ev["train_loss"] = round(float(loss.detach().item()), 4)
                ev["lr_mult"] = round(lr_mult, 4)
                curve.append(ev)
                exp_logger.log_eval(step, {**ev, "top1_acc": round(100.0 * ev["top1_acc"], 2), "top5_acc": round(100.0 * ev["top5_acc"], 2)})
                print(
                    f"[{variant_id} (R={r_spec.num_cholesky_modes}, d_vac={model.d_vac})] "
                    f"Step {step:4d} ({ev['tokens_M']:5.2f}M tok) | "
                    f"Train CE: {ev['train_loss']:.4f} | Pure Neural Val CE: {ev['val_ce']:.4f} (PPL {ev['val_ppl']:.2f}) | "
                    f"Top1: {ev['top1_acc']*100:.2f}% | Top5: {ev['top5_acc']*100:.2f}% | "
                    f"Modes: {ev['effective_active_branches']:.2f}/{r_spec.num_cholesky_modes} | "
                    f"cos(vac) [anc/core/high]: {ev['mean_anchor_id_cos_theta_vac']:.3f}/"
                    f"{ev['mean_core_excited_cos_theta_vac']:.3f}/{ev['mean_high_excited_cos_theta_vac']:.3f}",
                    flush=True,
                )

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - t_start
        results.append({
            "variant_id": variant_id, "variant_mode": r_spec.variant_mode.value, "description": desc,
            "num_cholesky_modes": r_spec.num_cholesky_modes, "vacuum_dim": model.d_vac,
            "use_layer0_input_cholesky": r_spec.use_layer0_input_cholesky,
            "use_independent_rmode_512d": r_spec.use_independent_rmode_512d,
            "init_ms": round(init_ms, 2), "elapsed_sec": round(elapsed_s, 2),
            "step0": step0_eval, "final": curve[-1], "curve": curve,
        })
        del model, q_trunk_opt, wte_opt, head_opt, adam_opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def run_variable_rank_cholesky_ablation_benchmark(
    steps: int = DEFAULT_FINEWEB_STEPS, eval_interval: int = DEFAULT_FINEWEB_EVAL_INTERVAL, device_str: str = "cuda",
) -> list[dict[str, Any]]:
    r"""Run the 300-step (`2.46M`-token) Variable-Rank $R \in \{2, 4, 6, 8, 12\}$ Cholesky & Vacuum ablation (`R0..R5`, `<= 8` args)."""
    device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    stream = LongSeqFineWebStream(device=device)
    vac_d = DEFAULT_VACUUM_SUBSPACE_DIM
    variants: list[tuple[str, VariableRankCholeskySpec, str]] = [
        ("R0_RANK8_NO_VACUUM_SPHERE", VariableRankCholeskySpec(VariableRankVariantMode.R0_RANK8_NO_VACUUM_SPHERE, 8, 0, True), "R=8 Uncollapsed Cholesky WITHOUT Vacuum Subspace (d_vac=0, forced S^63)"),
        ("R1_RANK2_VACUUM_CHOLESKY", VariableRankCholeskySpec(VariableRankVariantMode.R1_RANK2_VACUUM_CHOLESKY, 2, vac_d, True), "R=2 Uncollapsed Cholesky + Vacuum Subspace H_vac (d_vac=4)"),
        ("R3_RANK6_VACUUM_CHOLESKY", VariableRankCholeskySpec(VariableRankVariantMode.R3_RANK6_VACUUM_CHOLESKY, 6, vac_d, True), "R=6 Uncollapsed Cholesky + Vacuum Subspace H_vac (Core 4 + Value-Key/Query Schur Modes)"),
        ("R4_RANK8_VACUUM_CHOLESKY", VariableRankCholeskySpec(VariableRankVariantMode.R4_RANK8_VACUUM_CHOLESKY, 8, vac_d, True), "R=8 Uncollapsed Cholesky + Vacuum Subspace H_vac (Core 6 + Attended Lag-1 & Bigram Schur Modes)"),
        ("R5_RANK12_VACUUM_CHOLESKY", VariableRankCholeskySpec(VariableRankVariantMode.R5_RANK12_VACUUM_CHOLESKY, 12, vac_d, True), "R=12 Uncollapsed Cholesky + Vacuum Subspace H_vac (Core 8 + Lag-2, Skip-Gram, QK-Born, Trigram Modes)"),
    ]
    return _train_and_eval_variable_rank_variants(stream, variants, steps, eval_interval, wave_id="W59")


def run_variable_rank_cholesky_long_scaling_benchmark(
    steps: int = DEFAULT_LONG_FINEWEB_STEPS, eval_interval: int = DEFAULT_LONG_FINEWEB_EVAL_INTERVAL,
    num_cholesky_modes: int = DEFAULT_WINNING_CHOLESKY_RANK, device_str: str = "cuda",
) -> list[dict[str, Any]]:
    """Run the 1,200-step (`9.83M`-token) 100% Pure-Neural scaling run (`R6`) at unrestricted Cholesky rank $R=8$ (`<= 8` args)."""
    device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    stream = LongSeqFineWebStream(device=device)
    r_m = int(num_cholesky_modes)
    spec = VariableRankCholeskySpec(VariableRankVariantMode.R6_RANK8_VACUUM_CHOLESKY_SCALING_1200, r_m, DEFAULT_VACUUM_SUBSPACE_DIM, True, False)
    variants = [("R6_RANK8_VACUUM_CHOLESKY_SCALING_1200", spec, f"1,200-step (9.83M-token) 100% Pure-Neural Scaling Run with Unrestricted R={r_m} Vacuum Cholesky Highway")]
    return _train_and_eval_variable_rank_variants(stream, variants, steps, eval_interval, wave_id="W59_LONG")


__all__ = [
    "DEFAULT_HIGH_MODE_INIT_GATE", "DEFAULT_LAYER0_HIGH_ROW_SCALE", "DEFAULT_PROG_VAC_SLOPE",
    "DEFAULT_RANK_SWEEP_MODES", "DEFAULT_WINNING_CHOLESKY_RANK", "REF_EXCITED_CHOLESKY_MODES",
    "VariableRankCholeskySpec", "VariableRankVacuumCholeskyGPT", "VariableRankVariantMode",
    "_VariableRankCholeskyWavepacketBlock", "build_layer0_variable_rank_cholesky_factor",
    "build_variable_rank_cholesky_optimizers", "compute_rank_adaptive_cholesky_rms_scale",
    "compute_rank_invariant_excited_trace_scale", "evaluate_variable_rank_cholesky_model",
    "fused_variable_rank_cholesky_ket_attention", "initialize_and_calibrate_variable_rank_cholesky_model_",
    "run_variable_rank_cholesky_ablation_benchmark", "run_variable_rank_cholesky_long_scaling_benchmark",
]
