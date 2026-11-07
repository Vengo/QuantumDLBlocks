r"""Quantum "No-Click" POVM Attention Vacuum Sink, Per-Mode $R = 12$ vs. $R = 8$ Spectrum, and Leslie Smith Momentum-Free ($\beta = 0$) SGD Super-Convergence Laboratory (`EXPERIMENT_PLAN_080.md` / `Wave 60`).

Mathematical & Architectural Motivation
---------------------------------------
1. **Part 3 of the Vacuum Concept — Quantum "No-Click" POVM / Vacuum Sink Inside Trunk Causal Hermitian-Born Cholesky Attention**:
   - In standard causal softmax attention, causal weights over prior tokens $s \le t$ are forced to sum to $1$:
     $\sum_{s=0}^t A_{t,s}^{(h)} \equiv 1$. When no prior token in the window is relevant to head $h$ at position $t$,
     the head is still forced to inject a full convex combination of prior value kets $|v_{s,h}\rangle$ and excited
     Cholesky modes $L_{t,h,r \ge 1}$.
   - In quantum photodetection, an $M$-mode detector has a **Vacuum / No-Click POVM outcome** $E_{\text{vac}} = I - \sum_{s \le t} E_s$
     where the field occupies only the ground carrier state ($r = 0$) with **zero signal value excitation** ($|v_{\text{vac}, h}\rangle = \mathbf{0}$)
     and **zero excited Cholesky mode occupation** ($F_{\text{vac}, h, r \ge 1} = \mathbf{0}$).
   - We equip each wavepacket head $h \in \{0, \dots, 7\}$ with:
     (a) a learned unit-RMS complex **Vacuum Key ket** $|\hat{k}_{\text{vac}, h}\rangle \in \mathbb{C}^{32}$ ($\|\hat{k}_{\text{vac}, h}\|_2 = \sqrt{64} = 8$),
     (b) a learned **Vacuum Sink Log-Prior Bias** $b_{\text{vac}, h} \in \mathbb{R}$, and
     (c) a learned **Causal Horizon Entropy Exponent** $\gamma_{\text{pos}, h} = \sigma(\tilde{\gamma}_{\text{pos}, h}) \in (0, 1)$.
   - Query $|q_{t,h}\rangle \in \mathbb{C}^{32}$ evaluates its Hermitian-Born transition score against $|\hat{k}_{\text{vac}, h}\rangle$
     compared against the constructive phase coherence $\text{RMS}(|\bar{v}_{t,h}\rangle)$ of the attended window value ket:
     $$S_{t, \text{vac}}^{(h)} \;=\; b_{\text{vac}, h} \;+\; \gamma_{\text{pos}, h}\ln(t + 1) \;+\; \sigma_{\text{sdpa}}\!\Big(\operatorname{Re}\big(e^{i\phi_h}\langle q_{t,h}\mid \hat{k}_{\text{vac}, h}\rangle\big) + \beta_{\text{born}, h}\big|\langle q_{t,h}\mid \hat{k}_{\text{vac}, h}\rangle\big|^2\Big)$$
     $$p_{\text{noclick}, t, h} \;=\; \sigma\!\Big(S_{t, \text{vac}}^{(h)} - \text{RMS}\big(|\bar{v}_{t,h}\rangle\big)\Big) \;\in\; (0, 1)$$
   - **Dual Physical Effect of a No-Click Outcome**:
     1. **Window Value Ket Abstention ($|v_{\text{vac}, h}\rangle = \mathbf{0}$)**:
        $|\bar{v}_{t,h}^{\text{out}}\rangle = (1 - p_{\text{noclick}, t, h})\,|\bar{v}_{t,h}\rangle$.
     2. **Excited Cholesky Mode Vacuum Damping ($F_{\text{vac}, h, r \ge 1} = \mathbf{0}$)**:
        Excited Cholesky rows $r \in \{1, \dots, R-1\}$ are attenuated by $(1 - \kappa\,p_{\text{noclick}, t, h})$ relative to
        Anchor Row $r = 0$, purifying the local density factor $L_{t,h}$ toward the ground carrier when the window fails to click.

2. **Leslie Smith Momentum-Free ($\beta = 0$) SGD Step-Size Estimation & 1-Cycle Super-Convergence**:
   - Removing momentum ($\beta = 0$ across `ModularRiemannianStiefelKrausMuon`, `BatchSubspaceTikhonovGramMuon`, and scalar updates)
     eliminates the $\frac{1}{1-\beta} \approx 3.33\times$ ($\beta = 0.70$) velocity gain. Using Leslie Smith's (`arXiv:1506.01186`, `arXiv:1708.07120`)
     exponential **LR Range Test** with $\beta = 0$, we estimate the optimal momentum-free 1-Cycle peak step-size multiplier
     $\mu_{\text{peak}}^*(\beta = 0)$ and demonstrate **1-Cycle Super-Convergence** without momentum on the Quantum Wavepacket Cholesky manifold.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_NOCLICK_TRITON = True
except ImportError:
    HAS_NOCLICK_TRITON = False

from ..kernels.fused_quantum_wavepacket_attn import (
    MASK_NEG_INF_FILL, TRITON_MAX_FUSED_SEQ_LEN, _Backward_hermitian_qk_grams, _FusedHermitianBornRhoKetAttnFn,
    _compute_complex_hermitian_qk_grams, _get_compiled_chol_gram_fn, associative_low_rank_cholesky_gram_modulation,
)
from ..layers.impure_density_mirror_block import TRACE_NORM_EPS
from ..layers.quantum_statmech_channels import (
    QuantumMomentumScheduleMode, StiefelMomentumMode, compute_riemannian_momentum_beta,
)
from ..layers.quantum_svd_stiefel_graft import apply_fubini_study_self_ortho
from ..models.compiled_leaderboard_classical_gpt import apply_native_leaderboard_rotary
from ..models.modular_quantum_statmech_gpt import ModularRiemannianStiefelKrausMuon
from ..optim.vocab_manifold_muon import BatchSubspaceTikhonovGramMuon
from .born_schur_density_lab import DEFAULT_FINEWEB_EVAL_INTERVAL, DEFAULT_FINEWEB_STEPS, DEFAULT_SCHUR_EPS
from .experiment_logger import ExperimentIntentSpec, StandardizedExperimentLogger
from .shallow_longseq_wavepacket_lm_lab import (
    DEFAULT_SHALLOW_EMBED_DIM, DEFAULT_SHALLOW_MLP_HIDDEN, DEFAULT_SHALLOW_NGRAM_LR, DEFAULT_SHALLOW_NUM_HEADS,
    DEFAULT_SHALLOW_PKT_COUNT, DEFAULT_SHALLOW_PKT_DIM, DEFAULT_SHALLOW_READOUT_LR, DEFAULT_SHALLOW_SCALAR_LR,
    DEFAULT_SHALLOW_STIEFEL_LR, DEFAULT_SHALLOW_WTE_LR, DEFAULT_SOFTCAP_LOGIT, RAW_GPT2_VOCAB_SIZE,
    LongSeqFineWebStream,
)
from .vacuum_subspace_cholesky_lab import DEFAULT_VACUUM_SUBSPACE_DIM, vacuum_sphere_rms_norm
from .variable_rank_cholesky_vacuum_lab import (
    DEFAULT_WINNING_CHOLESKY_RANK, VariableRankCholeskySpec, VariableRankVacuumCholeskyGPT, VariableRankVariantMode,
    _build_variable_rank_cholesky_f_flat, _VariableRankCholeskyWavepacketBlock, build_variable_rank_cholesky_optimizers,
    evaluate_variable_rank_cholesky_model, initialize_and_calibrate_variable_rank_cholesky_model_,
)
from .wavefunction_outer_density_lab import (
    DEFAULT_INIT_SCHUR_GATE_SCALE, DEFAULT_LONG_FINEWEB_EVAL_INTERVAL, DEFAULT_LONG_FINEWEB_STEPS,
    phase_lock_cholesky_modes,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_080.md Section 3
# ==============================================================================
DEFAULT_INIT_NOCLICK_SINK_BIAS: float = -2.20
DEFAULT_INIT_NOCLICK_POS_LOGIT: float = -1.10
DEFAULT_NOCLICK_HIGHWAY_SHIELD_STRENGTH: float = 0.50
DEFAULT_SMITH_RANGE_STEPS: int = 80
DEFAULT_SMITH_MIN_LR_MULT: float = 0.02
DEFAULT_SMITH_MAX_LR_MULT: float = 12.0
DEFAULT_SMITH_1CYCLE_PCT_START: float = 0.30
DEFAULT_NUM_FREQ_DECILES: int = 10

CHOLESKY_MODE_ROLE_DESCRIPTIONS: tuple[str, ...] = (
    "r=0: Anchor Wavepacket |psi_{t,h}> (Primary Lexical + Semantic Carrier)",
    "r=1: Local Value Ket |v_{t,h}> (Window Value Excitation)",
    "r=2: Rotary Key Ket |k_{t,h}> (Window Key Boundary Detector)",
    "r=3: Rotary Query Ket |q_{t,h}> (Window Query Selector)",
    "r=4: Value-Key Schur Coincidence |r_{t,h}> * |k_{t,h}>",
    "r=5: Value-Query Conjugate Dipole |r_{t,h}> * |q_{t,h}^*>",
    "r=6: Attended Causal Lag-1 Ket |r_{t-1,h}>",
    "r=7: Causal Bigram Schur Coincidence |r_{t,h}> * |r_{t-1,h}>",
    "r=8: Attended Causal Lag-2 Ket |r_{t-2,h}>",
    "r=9: Skip-Bigram Schur Coincidence |r_{t,h}> * |r_{t-2,h}>",
    "r=10: Query-Key Born Dipole |q_{t,h}> * |k_{t,h}^*>",
    "r=11: Trigram Lag-1/Lag-2 Schur Coincidence |r_{t-1,h}> * |r_{t-2,h}>",
)


class NoClickVacuumVariantMode(str, Enum):
    """Strategy enum for Quantum No-Click POVM Vacuum Sink & Smith SGD experiments (`N0..N4`, `S0..S2`, Rule 3)."""

    N0_RANK8_NO_SINK_REF = "n0_rank8_no_sink_ref"
    N1_RANK8_NOCLICK_VACUUM_POVM = "n1_rank8_noclick_vacuum_povm"
    N2_RANK12_NOCLICK_VACUUM_POVM = "n2_rank12_noclick_vacuum_povm"
    N3_RANK12_VACUUM_SCALING_1200 = "n3_rank12_vacuum_scaling_1200"
    N4_RANK12_NOCLICK_VACUUM_SCALING_1200 = "n4_rank12_noclick_vacuum_scaling_1200"
    S0_MOMENTUM_FREE_SGD_BASELINE_LR = "s0_momentum_free_sgd_baseline_lr"
    S1_MOMENTUM_FREE_SGD_SMITH_1CYCLE = "s1_momentum_free_sgd_smith_1cycle"
    S2_MOMENTUM_070_REFERENCE = "s2_momentum_070_reference"


@dataclass(frozen=True)
class NoClickVacuumCholeskySpec:
    r"""Configuration for `NoClickVacuumCholeskyGPT` (`<= 8` fields, Rule 2)."""

    variant_mode: NoClickVacuumVariantMode = NoClickVacuumVariantMode.N1_RANK8_NOCLICK_VACUUM_POVM
    num_cholesky_modes: int = DEFAULT_WINNING_CHOLESKY_RANK
    vacuum_dim: int = DEFAULT_VACUUM_SUBSPACE_DIM
    use_noclick_vacuum_sink: bool = True
    noclick_highway_shield: float = DEFAULT_NOCLICK_HIGHWAY_SHIELD_STRENGTH
    init_noclick_bias: float = DEFAULT_INIT_NOCLICK_SINK_BIAS
    init_noclick_pos_logit: float = DEFAULT_INIT_NOCLICK_POS_LOGIT
    seed: int = 1337


if HAS_NOCLICK_TRITON:

    @triton.jit
    def _noclick_hermitian_born_fwd_kernel(
        c_re_ptr, c_im_ptr, s_vac_ptr, cos_phi_ptr, sin_phi_ptr, born_scale_ptr,
        a_out_ptr, p_vac_out_ptr, seq_t: int, num_heads: int, sdpa_scale: float,
        HAS_IMAG: tl.constexpr, BLOCK_T: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        t_idx = row_idx % seq_t
        h_idx = (row_idx // seq_t) % num_heads
        offs_s = tl.arange(0, BLOCK_T)
        valid_s = offs_s < seq_t
        active_mask = valid_s & (offs_s <= t_idx)
        row_base = row_idx * seq_t + offs_s

        cr = tl.load(c_re_ptr + row_base, mask=active_mask, other=0.0).to(tl.float32)
        if HAS_IMAG:
            ci = tl.load(c_im_ptr + row_base, mask=active_mask, other=0.0).to(tl.float32)
            c_p = tl.load(cos_phi_ptr + h_idx).to(tl.float32)
            s_p = tl.load(sin_phi_ptr + h_idx).to(tl.float32)
            b_sc = tl.load(born_scale_ptr + h_idx).to(tl.float32)
            raw_score = sdpa_scale * (c_p * cr + s_p * ci + b_sc * (cr * cr + ci * ci))
        else:
            raw_score = sdpa_scale * cr

        s_vac = tl.load(s_vac_ptr + row_idx).to(tl.float32)
        masked_score = tl.where(active_mask, raw_score, -1.0e20)
        max_s = tl.maximum(tl.max(masked_score, axis=0), s_vac)
        exp_s = tl.where(active_mask, tl.exp(masked_score - max_s), 0.0)
        exp_vac = tl.exp(s_vac - max_s)
        denom = tl.sum(exp_s, axis=0) + exp_vac + 1.0e-12
        a_val = exp_s / denom
        p_vac = exp_vac / denom
        tl.store(a_out_ptr + row_base, a_val.to(a_out_ptr.dtype.element_ty), mask=valid_s)
        tl.store(p_vac_out_ptr + row_idx, p_vac.to(p_vac_out_ptr.dtype.element_ty))

    @triton.jit
    def _noclick_hermitian_born_bwd_kernel(
        c_re_ptr, c_im_ptr, a_out_ptr, p_vac_ptr, grad_a_ptr, grad_p_vac_ptr,
        cos_phi_ptr, sin_phi_ptr, born_scale_ptr, grad_c_re_ptr, grad_c_im_ptr,
        grad_s_vac_ptr, grad_cos_sin_born_ptr, seq_t: int, num_heads: int, sdpa_scale: float,
        HAS_IMAG: tl.constexpr, BLOCK_T: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        h_idx = (row_idx // seq_t) % num_heads
        offs_s = tl.arange(0, BLOCK_T)
        valid_s = offs_s < seq_t
        row_base = row_idx * seq_t + offs_s

        a_val = tl.load(a_out_ptr + row_base, mask=valid_s, other=0.0).to(tl.float32)
        ga = tl.load(grad_a_ptr + row_base, mask=valid_s, other=0.0).to(tl.float32)
        p_vac = tl.load(p_vac_ptr + row_idx).to(tl.float32)
        gp_vac = tl.load(grad_p_vac_ptr + row_idx).to(tl.float32)

        dot_tot = tl.sum(a_val * ga, axis=0) + p_vac * gp_vac
        ds = a_val * (ga - dot_tot) * sdpa_scale
        tl.store(grad_s_vac_ptr + row_idx, p_vac * (gp_vac - dot_tot))

        if HAS_IMAG:
            cr = tl.load(c_re_ptr + row_base, mask=valid_s, other=0.0).to(tl.float32)
            ci = tl.load(c_im_ptr + row_base, mask=valid_s, other=0.0).to(tl.float32)
            c_p = tl.load(cos_phi_ptr + h_idx).to(tl.float32)
            s_p = tl.load(sin_phi_ptr + h_idx).to(tl.float32)
            two_b = 2.0 * tl.load(born_scale_ptr + h_idx).to(tl.float32)
            g_cr = ds * (c_p + two_b * cr)
            g_ci = ds * (s_p + two_b * ci)
            tl.store(grad_c_re_ptr + row_base, g_cr.to(grad_c_re_ptr.dtype.element_ty), mask=valid_s)
            tl.store(grad_c_im_ptr + row_base, g_ci.to(grad_c_im_ptr.dtype.element_ty), mask=valid_s)
            out_base = row_idx * 3
            tl.store(grad_cos_sin_born_ptr + out_base + 0, tl.sum(ds * cr, axis=0))
            tl.store(grad_cos_sin_born_ptr + out_base + 1, tl.sum(ds * ci, axis=0))
            tl.store(grad_cos_sin_born_ptr + out_base + 2, tl.sum(ds * (cr * cr + ci * ci), axis=0))
        else:
            tl.store(grad_c_re_ptr + row_base, ds.to(grad_c_re_ptr.dtype.element_ty), mask=valid_s)


class _FusedNoClickVacuumHermitianBornAttnFn(torch.autograd.Function):
    """Single-pass autograd function for Fused Causal Hermitian-Born Attention with a Quantum No-Click Vacuum Sink (Rule 9)."""

    @staticmethod
    def forward(
        ctx: Any, q_rot: torch.Tensor, k_rot: torch.Tensor, payload_packed: torch.Tensor, s_vac_bht: torch.Tensor,
        berry_phase_hrd: torch.Tensor | None, qk_born_beta_h: torch.Tensor | None, scale_and_hb: tuple[float, bool],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sdpa_scale, use_hermitian_born = scale_and_hb
        bsz, seq_len, num_heads, head_dim = q_rot.shape
        work_dtype = q_rot.dtype
        v_rank = berry_phase_hrd.size(1) if berry_phase_hrd is not None else 1
        d_h = berry_phase_hrd.size(2) if berry_phase_hrd is not None else (head_dim // 2)

        c_re, c_im = _compute_complex_hermitian_qk_grams(q_rot, k_rot, v_rank, d_h, need_imag=use_hermitian_born)

        if use_hermitian_born and berry_phase_hrd is not None and qk_born_beta_h is not None:
            phi_h = berry_phase_hrd.float().mean(dim=(-2, -1))
            cos_phi, sin_phi = torch.cos(phi_h), torch.sin(phi_h)
            born_scale = torch.tanh(qk_born_beta_h.float()) / float(head_dim)
        else:
            cos_phi = torch.ones((num_heads,), device=q_rot.device, dtype=torch.float32)
            sin_phi = torch.zeros((num_heads,), device=q_rot.device, dtype=torch.float32)
            born_scale = torch.zeros((num_heads,), device=q_rot.device, dtype=torch.float32)

        s_vac_c = s_vac_bht.float().contiguous()
        use_triton = HAS_NOCLICK_TRITON and q_rot.is_cuda and seq_len <= TRITON_MAX_FUSED_SEQ_LEN
        if use_triton:
            n_rows = bsz * num_heads * seq_len
            block_t = max(16, triton.next_power_of_2(seq_len))
            a_weights = torch.empty((bsz, num_heads, seq_len, seq_len), device=q_rot.device, dtype=work_dtype)
            p_noclick = torch.empty((bsz, num_heads, seq_len), device=q_rot.device, dtype=work_dtype)
            c_im_c = c_im if c_im is not None else c_re
            _noclick_hermitian_born_fwd_kernel[(n_rows,)](
                c_re, c_im_c, s_vac_c, cos_phi, sin_phi, born_scale, a_weights, p_noclick,
                seq_len, num_heads, float(sdpa_scale), HAS_IMAG=bool(use_hermitian_born), BLOCK_T=block_t,
            )
        else:
            cr_f = c_re.float()
            if use_hermitian_born and c_im is not None:
                ci_f = c_im.float()
                cp_v, sp_v, bs_v = cos_phi.view(1, num_heads, 1, 1), sin_phi.view(1, num_heads, 1, 1), born_scale.view(1, num_heads, 1, 1)
                scores = float(sdpa_scale) * (cp_v * cr_f + sp_v * ci_f + bs_v * (cr_f.square() + ci_f.square()))
            else:
                scores = float(sdpa_scale) * cr_f
            q_idx = torch.arange(seq_len, device=q_rot.device).unsqueeze(1)
            k_idx = torch.arange(seq_len, device=q_rot.device).unsqueeze(0)
            scores = scores.masked_fill(q_idx < k_idx, MASK_NEG_INF_FILL)
            scores_aug = torch.cat([scores, s_vac_c.unsqueeze(-1)], dim=-1)
            probs_aug = F.softmax(scores_aug, dim=-1).to(work_dtype)
            a_weights = probs_aug[..., :seq_len].contiguous()
            p_noclick = probs_aug[..., seq_len].contiguous()

        pay_w = payload_packed.to(work_dtype)
        out_payload = torch.matmul(a_weights, pay_w).to(work_dtype)
        ctx.save_for_backward(
            q_rot, k_rot, pay_w, a_weights, p_noclick, c_re,
            c_im if c_im is not None else torch.empty(0, device=q_rot.device),
            cos_phi, sin_phi, born_scale,
            berry_phase_hrd if berry_phase_hrd is not None else torch.empty(0, device=q_rot.device),
            qk_born_beta_h if qk_born_beta_h is not None else torch.empty(0, device=q_rot.device),
        )
        ctx.sdpa_scale, ctx.use_hermitian_born, ctx.use_triton = float(sdpa_scale), bool(use_hermitian_born), bool(use_triton)
        ctx.v_rank, ctx.d_h = int(v_rank), int(d_h)
        ctx.payload_orig_dtype, ctx.svac_orig_dtype = payload_packed.dtype, s_vac_bht.dtype
        return out_payload, p_noclick

    @staticmethod
    def backward(ctx: Any, grad_out_payload: torch.Tensor, grad_p_noclick: torch.Tensor) -> tuple[Any, ...]:
        (
            q_rot, k_rot, payload_packed, a_weights, p_noclick, c_re, c_im_saved,
            cos_phi, sin_phi, born_scale, berry_saved, born_beta_saved,
        ) = ctx.saved_tensors
        bsz, seq_len, num_heads, head_dim = q_rot.shape
        work_dtype = c_re.dtype
        use_hb = ctx.use_hermitian_born
        c_im = c_im_saved.to(work_dtype) if use_hb else None

        grad_out_c = grad_out_payload.to(work_dtype).contiguous()
        gp_vac_c = grad_p_noclick.float().contiguous() if grad_p_noclick is not None else torch.zeros_like(p_noclick, dtype=torch.float32)
        a_w, pay_w = a_weights.to(work_dtype), payload_packed.to(work_dtype)
        grad_payload = torch.matmul(a_w.transpose(-2, -1), grad_out_c).to(ctx.payload_orig_dtype)
        grad_a = torch.matmul(grad_out_c, pay_w.transpose(-2, -1))
        grad_berry: torch.Tensor | None = None
        grad_born_beta: torch.Tensor | None = None

        if ctx.use_triton:
            n_rows = bsz * num_heads * seq_len
            block_t = max(16, triton.next_power_of_2(seq_len))
            grad_c_re = torch.empty_like(c_re)
            grad_c_im = torch.empty_like(c_im) if (use_hb and c_im is not None) else grad_c_re
            grad_s_vac = torch.empty((bsz, num_heads, seq_len), device=q_rot.device, dtype=torch.float32)
            grad_csb = torch.empty((bsz, num_heads, seq_len, 3) if use_hb else (1,), device=q_rot.device, dtype=torch.float32)
            _noclick_hermitian_born_bwd_kernel[(n_rows,)](
                c_re, c_im if (use_hb and c_im is not None) else c_re, a_w, p_noclick, grad_a, gp_vac_c,
                cos_phi, sin_phi, born_scale, grad_c_re, grad_c_im, grad_s_vac, grad_csb,
                seq_len, num_heads, ctx.sdpa_scale, HAS_IMAG=bool(use_hb), BLOCK_T=block_t,
            )
            d_cos_h, d_sin_h, d_born_h = (
                (grad_csb[..., 0].sum(dim=(0, 2)), grad_csb[..., 1].sum(dim=(0, 2)), grad_csb[..., 2].sum(dim=(0, 2)))
                if use_hb else (None, None, None)
            )
        else:
            af, gaf, pvf = a_w.float(), grad_a.float(), p_noclick.float()
            dot_tot = (af * gaf).sum(dim=-1, keepdim=True) + (pvf * gp_vac_c).unsqueeze(-1)
            ds = af * (gaf - dot_tot) * ctx.sdpa_scale
            grad_s_vac = pvf * (gp_vac_c - dot_tot.squeeze(-1))
            if use_hb and c_im is not None:
                crf, cif = c_re.float(), c_im.float()
                cp_v, sp_v, two_b = cos_phi.view(1, num_heads, 1, 1), sin_phi.view(1, num_heads, 1, 1), 2.0 * born_scale.view(1, num_heads, 1, 1)
                grad_c_re = (ds * (cp_v + two_b * crf)).to(work_dtype)
                grad_c_im = (ds * (sp_v + two_b * cif)).to(work_dtype)
                d_cos_h, d_sin_h, d_born_h = (ds * crf).sum(dim=(0, 2, 3)), (ds * cif).sum(dim=(0, 2, 3)), (ds * (crf.square() + cif.square())).sum(dim=(0, 2, 3))
            else:
                grad_c_re, grad_c_im, d_cos_h, d_sin_h, d_born_h = ds.to(work_dtype), None, None, None, None

        if use_hb and d_cos_h is not None and d_sin_h is not None and d_born_h is not None:
            if berry_saved.numel() > 0:
                d_phi_h = -sin_phi * d_cos_h + cos_phi * d_sin_h
                scale_b = 1.0 / float(berry_saved.size(1) * berry_saved.size(2))
                grad_berry = (d_phi_h.view(num_heads, 1, 1) * scale_b).expand_as(berry_saved).to(berry_saved.dtype).contiguous()
            if born_beta_saved.numel() > 0:
                th = torch.tanh(born_beta_saved.float())
                grad_born_beta = (d_born_h * (1.0 - th.square()) / float(head_dim)).to(born_beta_saved.dtype)

        grad_q_rot, grad_k_rot = _Backward_hermitian_qk_grams(
            grad_c_re, grad_c_im if use_hb else None, q_rot, k_rot, ctx.v_rank, ctx.d_h
        )
        return grad_q_rot, grad_k_rot, grad_payload, grad_s_vac.to(ctx.svac_orig_dtype), grad_berry, grad_born_beta, None


def compute_noclick_vacuum_sink_scores(
    q_rot: torch.Tensor, k_vac_ket: torch.Tensor, vac_sink_bias: torch.Tensor,
    berry_phase: torch.Tensor, born_beta: torch.Tensor, sdpa_scale: float,
) -> torch.Tensor:
    r"""Compute per-query Quantum No-Click POVM score $S_{t, \text{vac}}^{(h)}$ `(B, H, T)` against $|\hat{k}_{\text{vac}, h}\rangle \in \mathbb{C}^{32}$ (`<= 8` args)."""
    bsz, seq_len, num_heads, head_dim = q_rot.shape
    d_h = head_dim // 2
    k_vac_unit = F.rms_norm(k_vac_ket.float(), (head_dim,), eps=1e-6).view(1, 1, num_heads, 2, d_h)
    q_c = q_rot.float().view(bsz, seq_len, num_heads, 2, d_h)
    q_re, q_im = q_c[..., 0, :], q_c[..., 1, :]
    kv_re, kv_im = k_vac_unit[..., 0, :], k_vac_unit[..., 1, :]

    c_vac_re = (q_re * kv_re + q_im * kv_im).sum(dim=-1).permute(0, 2, 1)
    c_vac_im = (q_re * kv_im - q_im * kv_re).sum(dim=-1).permute(0, 2, 1)

    phi_h = berry_phase.float().mean(dim=(-2, -1)).view(1, num_heads, 1)
    cos_p, sin_p = torch.cos(phi_h), torch.sin(phi_h)
    b_sc = (torch.tanh(born_beta.float()) / float(head_dim)).view(1, num_heads, 1)

    hb_term = float(sdpa_scale) * (cos_p * c_vac_re + sin_p * c_vac_im + b_sc * (c_vac_re.square() + c_vac_im.square()))
    b_vac = vac_sink_bias[:, 0].float().view(1, num_heads, 1)
    gamma_pos = torch.sigmoid(vac_sink_bias[:, 1].float()).view(1, num_heads, 1)
    log_pos = torch.log(torch.arange(1, seq_len + 1, device=q_rot.device, dtype=torch.float32)).view(1, 1, seq_len)
    return (b_vac + gamma_pos * log_pos + hb_term).contiguous()


def fused_noclick_vacuum_cholesky_ket_attention(
    q_rot: torch.Tensor, k_rot: torch.Tensor, r_re_im: tuple[torch.Tensor, torch.Tensor], v_mixed: torch.Tensor,
    high_mode_gates: torch.Tensor, noclick_params: tuple[torch.Tensor, torch.Tensor],
    phase_and_born: tuple[torch.Tensor, torch.Tensor, float, bool], k_chol: int = DEFAULT_WINNING_CHOLESKY_RANK,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Single-pass fused Variable-Rank Cholesky + Ket Attention with a Ground-Mode Quantum No-Click POVM Vacuum Sink (`<= 8` args)."""
    r_re, r_im = r_re_im
    bsz, num_heads, seq_len, v_rank, d_h = r_re.shape
    r_modes = max(1, int(k_chol))
    head_dim = q_rot.size(-1)
    k_vac_ket, vac_sink_bias = noclick_params
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

    s_vac_bht = compute_noclick_vacuum_sink_scores(q_rot, k_vac_ket, vac_sink_bias, berry_phase, born_beta, float(sdpa_scale))
    v_bar_rms = (
        (v_bar_re.detach().float().square().mean(dim=(-2, -1)) + v_bar_im.detach().float().square().mean(dim=(-2, -1)))
        .clamp_min(1e-6)
        .sqrt()
    )
    p_noclick = torch.sigmoid(s_vac_bht - v_bar_rms).to(q_rot.dtype)

    # 1. Attenuate window value ket |v_bar> by click probability (1 - p_noclick) since |v_vac> = 0
    click_prob = (1.0 - p_noclick).unsqueeze(-1).unsqueeze(-1)
    v_bar_re_out = v_bar_re * click_prob
    v_bar_im_out = v_bar_im * click_prob

    # 2. Attenuate excited Cholesky modes r >= 1 relative to Ground Mode r = 0 since F_{vac, r >= 1} = 0
    if r_modes > 1:
        exc_shield = (1.0 - DEFAULT_NOCLICK_HIGHWAY_SHIELD_STRENGTH * p_noclick).unsqueeze(-1).unsqueeze(-1)
        l_re_out = torch.cat([l_re[:, :, :, :1, :], l_re[:, :, :, 1:, :] * exc_shield], dim=-2)
        l_im_out = torch.cat([l_im[:, :, :, :1, :], l_im[:, :, :, 1:, :] * exc_shield], dim=-2)
    else:
        l_re_out, l_im_out = l_re, l_im

    return l_re_out, l_im_out, v_bar_re_out, v_bar_im_out, p_noclick


class _NoClickVacuumCholeskyWavepacketBlock(_VariableRankCholeskyWavepacketBlock):
    r"""Wavepacket Block with a learned Quantum No-Click POVM / Vacuum Key $|\hat{k}_{\text{vac}, h}\rangle \in \mathbb{C}^{32}$ per head."""

    def __init__(
        self,
        embed_dim: int = DEFAULT_SHALLOW_EMBED_DIM,
        num_heads: int = DEFAULT_SHALLOW_NUM_HEADS,
        pkt_count: int = DEFAULT_SHALLOW_PKT_COUNT,
        pkt_dim: int = DEFAULT_SHALLOW_PKT_DIM,
        mlp_hidden: int = DEFAULT_SHALLOW_MLP_HIDDEN,
        init_schur_scale: float = DEFAULT_INIT_SCHUR_GATE_SCALE,
        num_cholesky_modes: int = DEFAULT_WINNING_CHOLESKY_RANK,
        noclick_cfg: tuple[bool, float, float, float] = (
            True, DEFAULT_NOCLICK_HIGHWAY_SHIELD_STRENGTH, DEFAULT_INIT_NOCLICK_SINK_BIAS, DEFAULT_INIT_NOCLICK_POS_LOGIT,
        ),
    ) -> None:
        super().__init__(embed_dim, num_heads, pkt_count, pkt_dim, mlp_hidden, init_schur_scale, num_cholesky_modes)
        use_sink, shield_strength, init_bias, init_pos_logit = noclick_cfg
        self.use_noclick_vacuum_sink = bool(use_sink)
        self.noclick_highway_shield = float(shield_strength)

        k_vac_init = F.rms_norm(torch.randn((self.num_heads, self.head_dim), dtype=torch.float32), (self.head_dim,), eps=1e-6)
        self.k_vac_ket = nn.Parameter(k_vac_init)
        sink_bias_init = torch.stack(
            [torch.full((self.num_heads,), float(init_bias)), torch.full((self.num_heads,), float(init_pos_logit))], dim=-1
        )
        self.vac_sink_bias = nn.Parameter(sink_bias_init)
        self.last_mean_noclick_prob: float = 0.0
        self.last_id_head_noclick_prob: float = 0.0
        self.last_sem_head_noclick_prob: float = 0.0

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return scalar parameters including Quantum No-Click POVM Vacuum Key and sink bias."""
        params = super().scalar_parameters()
        if self.use_noclick_vacuum_sink:
            params.extend([self.k_vac_ket, self.vac_sink_bias])
        return params

    @torch.no_grad()
    def restrict_scalars_(self) -> None:
        """Clamp No-Click sink bias and re-normalize `k_vac_ket` to unit coordinate RMS."""
        super().restrict_scalars_()
        if self.use_noclick_vacuum_sink:
            self.vac_sink_bias[:, 0].clamp_(-4.5, 1.5)
            self.vac_sink_bias[:, 1].clamp_(-3.5, 2.5)
            self.k_vac_ket.copy_(F.rms_norm(self.k_vac_ket.float(), (self.head_dim,), eps=1e-6))

    def forward(
        self, x: torch.Tensor, x_anchor: torch.Tensor, cos_4d: torch.Tensor, sin_4d: torch.Tensor,
        in_cholesky: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Execute variable-rank Cholesky attention with optional Quantum No-Click POVM Vacuum Sink."""
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
        nsq = (vn_re.float().square().sum(dim=(-2, -1), keepdim=True) + vn_im.float().square().sum(dim=(-2, -1), keepdim=True)).clamp_min(TRACE_NORM_EPS).rsqrt().to(work_dtype)
        r_re, r_im = vn_re * nsq, vn_im * nsq
        l_re, l_im, v_bar_re, v_bar_im, p_noclick = fused_noclick_vacuum_cholesky_ket_attention(
            q_rot, k_rot, (r_re, r_im), v_mixed_h, self.high_mode_gates, (self.k_vac_ket, self.vac_sink_bias),
            (self.berry_qk_phase, self.born_qk_beta, self.sdpa_scale, True), k_chol=self.num_cholesky_modes,
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


class NoClickVacuumCholeskyGPT(VariableRankVacuumCholeskyGPT):
    r"""Unrestricted Rank-$R$ Cholesky Density-Matrix GPT with Readout Vacuum Sphere $\mathcal{H}_{\text{vac}}$ AND Trunk Quantum No-Click POVM Sink."""

    def __init__(self, nc_spec: NoClickVacuumCholeskySpec) -> None:
        r_spec = VariableRankCholeskySpec(
            variant_mode=VariableRankVariantMode.R4_RANK8_VACUUM_CHOLESKY,
            num_cholesky_modes=nc_spec.num_cholesky_modes, vacuum_dim=nc_spec.vacuum_dim,
            use_layer0_input_cholesky=True, use_independent_rmode_512d=False, seed=nc_spec.seed,
        )
        super().__init__(r_spec)
        self.nc_spec = nc_spec
        self.nc_variant_mode = NoClickVacuumVariantMode(nc_spec.variant_mode)
        noclick_cfg = (
            bool(nc_spec.use_noclick_vacuum_sink), float(nc_spec.noclick_highway_shield),
            float(nc_spec.init_noclick_bias), float(nc_spec.init_noclick_pos_logit),
        )
        base_blocks = list(self.quantum_blocks)
        new_blocks: list[_NoClickVacuumCholeskyWavepacketBlock] = []
        for b_old in base_blocks:
            b_new = _NoClickVacuumCholeskyWavepacketBlock(
                self.embed_dim, self.num_heads, self.pkt_count, self.pkt_dim, self.mlp_hidden,
                init_schur_scale=DEFAULT_INIT_SCHUR_GATE_SCALE, num_cholesky_modes=self.num_branches, noclick_cfg=noclick_cfg,
            )
            b_new.load_state_dict(b_old.state_dict(), strict=False)
            new_blocks.append(b_new)
        self.quantum_blocks = nn.ModuleList(new_blocks)

    def get_noclick_telemetry(self) -> dict[str, float]:
        """Return mean, identity-head, and semantic-head No-Click POVM probabilities across all layers."""
        if not self.nc_spec.use_noclick_vacuum_sink:
            return {"mean_noclick_prob": 0.0, "id_head_noclick_prob": 0.0, "sem_head_noclick_prob": 0.0}
        n_l = float(max(1, len(self.quantum_blocks)))
        mean_nc = sum(float(getattr(b, "last_mean_noclick_prob", 0.0)) for b in self.quantum_blocks) / n_l
        id_nc = sum(float(getattr(b, "last_id_head_noclick_prob", 0.0)) for b in self.quantum_blocks) / n_l
        sem_nc = sum(float(getattr(b, "last_sem_head_noclick_prob", 0.0)) for b in self.quantum_blocks) / n_l
        return {"mean_noclick_prob": round(mean_nc, 4), "id_head_noclick_prob": round(id_nc, 4), "sem_head_noclick_prob": round(sem_nc, 4)}


@torch.no_grad()
def evaluate_noclick_vacuum_cholesky_model(
    model: NoClickVacuumCholeskyGPT, stream: LongSeqFineWebStream
) -> dict[str, float]:
    """Evaluate `NoClickVacuumCholeskyGPT` on unseen Shard-0 validation batches and include No-Click POVM telemetry (`<= 8` args)."""
    ev = evaluate_variable_rank_cholesky_model(model, stream)
    ev.update(model.get_noclick_telemetry())
    return ev


@torch.no_grad()
def extract_per_mode_readout_diagnostics(
    model: VariableRankVacuumCholeskyGPT, idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Extract per-token, per-mode normalized energies $E_{t,r}$, mixture probabilities $\pi_{t,r}$, and vacuum angles $\cos\theta_{\text{vac}}(t, r)$ `(B, T, R)` (`<= 8` args)."""
    h_pre, _, readout_id_64, final_chol = model._encode_multiscale_trunk(idx)
    bsz, seq_len, _ = h_pre.shape
    r_m = model.num_branches
    work_dtype = h_pre.dtype
    h_bridge = model.exit_bridge(h_pre)

    ctx_16 = h_pre[:, :, :16]
    u_id_vac_0 = (
        model.id_vac_mass.to(work_dtype).view(1, 1, model.d_vac) * (2.0 * torch.sigmoid(F.linear(ctx_16, model.id_vac_proj.weight.to(work_dtype))))
        if model.d_vac > 0 else torch.zeros((bsz, seq_len, 0), device=idx.device, dtype=work_dtype)
    )
    h_id_tr = F.rms_norm(h_bridge[:, :, model.sem_real_dim :], (model.id_real_dim,), eps=1e-6)
    raw_id_0 = model.trunk_id_gate.to(work_dtype) * h_id_tr + readout_id_64 if readout_id_64 is not None else h_id_tr
    h_id_0, h_id_0_vac, id_cos_0 = vacuum_sphere_rms_norm(raw_id_0, u_id_vac_0)

    if r_m == 1:
        ones_e = torch.ones((bsz, seq_len, 1), device=idx.device, dtype=torch.float32)
        return ones_e, ones_e, id_cos_0.float().view(bsz, seq_len, 1)

    last_blk = model.quantum_blocks[-1]
    assert final_chol is not None
    mod_l_r, mod_l_i = final_chol
    u_r_t = last_blk.u_evol_re.to(work_dtype).view(1, model.num_heads, 1, last_blk.d_h, last_blk.d_h).transpose(-2, -1)  # type: ignore[union-attr]
    u_i_t = last_blk.u_evol_im.to(work_dtype).view(1, model.num_heads, 1, last_blk.d_h, last_blk.d_h).transpose(-2, -1)  # type: ignore[union-attr]
    l_all_re = torch.matmul(mod_l_r, u_r_t) - torch.matmul(mod_l_i, u_i_t)
    l_all_im = torch.matmul(mod_l_r, u_i_t) + torch.matmul(mod_l_i, u_r_t)

    l7_re, l7_im = l_all_re[:, -1], l_all_im[:, -1]
    id_energy = (l7_re.float().square().sum(dim=-1) + l7_im.float().square().sum(dim=-1)).clamp_min(1e-6)
    l_sem_re, l_sem_im = l_all_re[:, : model.sem_complex_pkts], l_all_im[:, : model.sem_complex_pkts]
    sem_energy = (l_sem_re.float().square().sum(dim=-1) + l_sem_im.float().square().sum(dim=-1)).mean(dim=1).clamp_min(1e-6)
    w_sem = torch.sigmoid(model.sem_energy_mix_logit.float())
    mode_energy = (1.0 - w_sem) * id_energy + w_sem * sem_energy
    norm_energy = mode_energy / mode_energy.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    pi_r = F.softmax(torch.log(mode_energy) + model.chol_mode_log_prior.view(1, 1, r_m), dim=-1)

    l7_64_scaled = torch.stack([l7_re, l7_im], dim=-2).reshape(bsz, seq_len, r_m, model.id_real_dim) * model.chol_rms_scale
    u_chol_vac = (
        model.chol_id_vac_mass.to(work_dtype).view(1, 1, r_m, model.d_vac) * (2.0 * torch.sigmoid(F.linear(ctx_16, model.chol_vac_proj.weight.to(work_dtype)).view(bsz, seq_len, r_m, model.d_vac)))
        if model.d_vac > 0 else torch.zeros((bsz, seq_len, r_m, 0), device=idx.device, dtype=work_dtype)
    )
    l7_vocab, l7_vac, _ = vacuum_sphere_rms_norm(l7_64_scaled, u_chol_vac)

    coh = torch.sigmoid(model.chol_mode_coherence).to(work_dtype)
    mix_exc_vocab = coh * h_id_0.unsqueeze(-2) + (1.0 - coh) * l7_vocab[:, :, 1:, :]
    mix_exc_vac = coh * h_id_0_vac.unsqueeze(-2) + (1.0 - coh) * l7_vac[:, :, 1:, :]
    _, _, exc_cos = vacuum_sphere_rms_norm(mix_exc_vocab, mix_exc_vac)
    cos_all = torch.cat([id_cos_0.float().view(bsz, seq_len, 1), exc_cos.float().view(bsz, seq_len, r_m - 1)], dim=-1)
    return norm_energy.float(), pi_r.float(), cos_all


def build_zipf_frequency_decile_lookup(unigram_bias: torch.Tensor, num_deciles: int = DEFAULT_NUM_FREQ_DECILES) -> torch.Tensor:
    """Assign each GPT-2 vocabulary token `0..50256` to a Zipf frequency decile `0` (most frequent) .. `num_deciles - 1` (rarest)."""
    v_total = int(unigram_bias.numel())
    n_valid = min(RAW_GPT2_VOCAB_SIZE, v_total)
    sorted_idx = torch.argsort(unigram_bias[:n_valid].detach().float(), descending=True)
    decile_map = torch.full((v_total,), num_deciles - 1, device=unigram_bias.device, dtype=torch.long)
    ranks = torch.arange(n_valid, device=unigram_bias.device, dtype=torch.long)
    decile_map[sorted_idx] = torch.clamp((ranks * num_deciles) // max(1, n_valid), 0, num_deciles - 1)
    return decile_map


@torch.no_grad()
def profile_per_mode_cholesky_spectrum_and_vacuum_angles(
    model: VariableRankVacuumCholeskyGPT, stream: LongSeqFineWebStream, max_val_batches: int = 8,
) -> dict[str, Any]:
    r"""Profile per-mode energy $E_{t,r}$, vacuum angle $\cos\theta_{\text{vac}}(t, r)$, and Zipf token-decile specialization across $r = 0 \dots R-1$ (`<= 8` args)."""
    was_training = model.training
    model.eval()
    r_m, n_dec = model.num_branches, DEFAULT_NUM_FREQ_DECILES
    decile_map = build_zipf_frequency_decile_lookup(model.unigram_bias, num_deciles=n_dec)

    energy_list, pi_list, cos_list = [], [], []
    dec_token_counts = torch.zeros((n_dec,), device=stream.device, dtype=torch.float64)
    dec_mode_post_sum = torch.zeros((n_dec, r_m), device=stream.device, dtype=torch.float64)
    dec_mode_nll_sum = torch.zeros((n_dec, r_m), device=stream.device, dtype=torch.float64)
    dom_exc_counts = torch.zeros((r_m,), device=stream.device, dtype=torch.float64)

    cap, inv_cap = float(DEFAULT_SOFTCAP_LOGIT), 1.0 / float(DEFAULT_SOFTCAP_LOGIT)
    batches_to_run = stream.val_batches[: max(1, min(len(stream.val_batches), int(max_val_batches)))]

    for idx_in, targets in batches_to_run:
        bsz, seq_len = idx_in.shape
        e_btr, pi_btr, cos_btr = extract_per_mode_readout_diagnostics(model, idx_in)
        energy_list.append(e_btr.reshape(-1, r_m).cpu())
        pi_list.append(pi_btr.reshape(-1, r_m).cpu())
        cos_list.append(cos_btr.reshape(-1, r_m).cpu())
        if r_m > 1:
            dom_exc = torch.argmax(pi_btr[:, :, 1:], dim=-1).reshape(-1) + 1
            dom_exc_counts.scatter_add_(0, dom_exc, torch.ones_like(dom_exc, dtype=torch.float64))
        del e_btr, pi_btr, cos_btr

        h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug_528 = model.build_multibranch_readout_tensors(idx_in)
        v_sz, sem_dim, id_dim = w_aug_528.size(0), h_sem_branches.size(-1), h_id_branches.size(-1)
        car_dim, indep_sem = h_base_464.size(-1) - sem_dim, h_sem_branches.ndim == 4
        w_sem, w_id = w_aug_528[:, :sem_dim], w_aug_528[:, sem_dim : sem_dim + id_dim]
        w_car = w_aug_528[:, sem_dim + id_dim : sem_dim + id_dim + car_dim]
        w_base_t, w_sem_t, w_id_t = torch.cat([w_sem, w_car], dim=-1).t().contiguous(), w_sem.t().contiguous(), w_id.t().contiguous()
        sem1_gates, id_scales, gate_logits = branch_controls[:, :, :, 0], branch_controls[:, :, :, 1], branch_controls[:, :, :, 2]
        has_sem1 = bool(sem1_gates.abs().max().item() > 1e-8)

        t_step = max(16, (64 * 4) // max(1, r_m))
        for t_start in range(0, seq_len, t_step):
            t_end = min(t_start + t_step, seq_len)
            c_len = bsz * (t_end - t_start)
            hb_c = h_base_464[:, t_start:t_end].reshape(c_len, sem_dim + car_dim)
            hid_c = h_id_branches[:, t_start:t_end].reshape(c_len * r_m, id_dim)
            b_c = sem1_gates[:, t_start:t_end].reshape(c_len, r_m).float()
            t_c = id_scales[:, t_start:t_end].reshape(c_len, r_m).float()
            a_c = gate_logits[:, t_start:t_end].reshape(c_len, r_m).float()
            y_c = targets[:, t_start:t_end].reshape(c_len)

            z_base = torch.matmul(hb_c, w_base_t).float()
            raw_r = torch.matmul(hid_c, w_id_t).float().view(c_len, r_m, v_sz)
            raw_r.mul_(t_c.unsqueeze(-1)).add_(z_base.unsqueeze(1))
            if has_sem1:
                if indep_sem:
                    hs_c = h_sem_branches[:, t_start:t_end].reshape(c_len * r_m, sem_dim)
                    raw_r.addcmul_(b_c.unsqueeze(-1), torch.matmul(hs_c, w_sem_t).float().view(c_len, r_m, v_sz))
                else:
                    hs_c = h_sem_branches[:, t_start:t_end].reshape(c_len, sem_dim)
                    raw_r.addcmul_(b_c.unsqueeze(-1), torch.matmul(hs_c, w_sem_t).float().unsqueeze(1))
            raw_r.mul_(inv_cap).tanh_().mul_(cap)

            log_pi = F.log_softmax(a_c, dim=-1)
            log_p_r = F.log_softmax(raw_r, dim=-1)
            log_p_y_r = log_p_r.gather(dim=-1, index=y_c.view(c_len, 1, 1).expand(c_len, r_m, 1)).squeeze(-1)
            posterior_r = F.softmax(log_pi + log_p_y_r, dim=-1)

            dec_y = decile_map[y_c]
            dec_token_counts.scatter_add_(0, dec_y, torch.ones_like(dec_y, dtype=torch.float64))
            dec_mode_post_sum.scatter_add_(0, dec_y.unsqueeze(1).expand(c_len, r_m), posterior_r.double())
            dec_mode_nll_sum.scatter_add_(0, dec_y.unsqueeze(1).expand(c_len, r_m), (-log_p_y_r).double())
            del z_base, raw_r, log_p_r
        del h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug_528

    if stream.device.type == "cuda":
        torch.cuda.empty_cache()
    if was_training:
        model.train()

    all_e, all_pi, all_cos = torch.cat(energy_list, dim=0), torch.cat(pi_list, dim=0), torch.cat(cos_list, dim=0)
    total_pos = float(max(1, all_e.size(0)))
    dec_denom = dec_token_counts.clamp_min(1.0).unsqueeze(1)
    mean_dec_post = (dec_mode_post_sum / dec_denom).cpu()
    mean_dec_nll = (dec_mode_nll_sum / dec_denom).cpu()
    dec_counts_cpu = dec_token_counts.cpu()

    per_mode_records: list[dict[str, Any]] = []
    for r_idx in range(r_m):
        e_col, pi_col, cos_col = all_e[:, r_idx], all_pi[:, r_idx], all_cos[:, r_idx]
        role = CHOLESKY_MODE_ROLE_DESCRIPTIONS[r_idx] if r_idx < len(CHOLESKY_MODE_ROLE_DESCRIPTIONS) else f"r={r_idx}: Higher-Order Cyclic Schur Mode"
        head_post = float(mean_dec_post[:3, r_idx].mean().item())
        mid_post = float(mean_dec_post[3:7, r_idx].mean().item())
        tail_post = float(mean_dec_post[7:, r_idx].mean().item())
        adv_vs_r0_tail = float((mean_dec_nll[7:, 0] - mean_dec_nll[7:, r_idx]).mean().item())
        per_mode_records.append({
            "mode_index": r_idx, "role": role,
            "mean_energy_share": round(float(e_col.mean().item()), 5), "std_energy_share": round(float(e_col.std().item()), 5),
            "mean_mixture_pi": round(float(pi_col.mean().item()), 5),
            "dominant_excited_frac": round(float(dom_exc_counts[r_idx].item() / total_pos), 4) if r_idx >= 1 else 0.0,
            "cos_theta_vac_mean": round(float(cos_col.mean().item()), 4), "cos_theta_vac_std": round(float(cos_col.std().item()), 4),
            "cos_theta_vac_p10": round(float(torch.quantile(cos_col, 0.10).item()), 4),
            "cos_theta_vac_p90": round(float(torch.quantile(cos_col, 0.90).item()), 4),
            "posterior_resp_head_deciles_0_2": round(head_post, 5), "posterior_resp_mid_deciles_3_6": round(mid_post, 5),
            "posterior_resp_tail_deciles_7_9": round(tail_post, 5),
            "tail_over_head_posterior_ratio": round(tail_post / max(1e-6, head_post), 4),
            "tail_nll_advantage_vs_mode0_nats": round(adv_vs_r0_tail, 4),
        })

    per_decile_records: list[dict[str, Any]] = []
    for d_idx in range(n_dec):
        post_row, nll_row = mean_dec_post[d_idx], mean_dec_nll[d_idx]
        best_exc = int(torch.argmin(nll_row[1:]).item() + 1) if r_m > 1 else 0
        per_decile_records.append({
            "decile": d_idx, "token_count": int(dec_counts_cpu[d_idx].item()),
            "mode0_posterior_share": round(float(post_row[0].item()), 4),
            "excited_posterior_share": round(float(post_row[1:].sum().item()), 4) if r_m > 1 else 0.0,
            "mode0_standalone_ce": round(float(nll_row[0].item()), 4),
            "best_excited_mode": best_exc, "best_excited_standalone_ce": round(float(nll_row[best_exc].item()), 4),
        })

    return {
        "num_cholesky_modes": r_m, "vacuum_dim": model.d_vac, "evaluated_tokens": int(total_pos),
        "per_mode_spectrum": per_mode_records, "per_decile_specialization": per_decile_records,
    }


def _apply_global_optimizer_step(
    model: VariableRankVacuumCholeskyGPT,
    optimizers: tuple[ModularRiemannianStiefelKrausMuon, BatchSubspaceTikhonovGramMuon, BatchSubspaceTikhonovGramMuon, torch.optim.AdamW],
    idx_in: torch.Tensor, targets: torch.Tensor, lr_mult: float, mom_beta: float,
) -> None:
    """Execute one coordinated Riemannian + Vocab + Scalar step with given `lr_mult` and `mom_beta` (`<= 8` args)."""
    q_trunk_opt, wte_opt, head_opt, adam_opt = optimizers
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
    adam_beta1 = 0.0 if mom_beta <= 0.0 else 0.80
    for pg in adam_opt.param_groups:
        pg["betas"] = (adam_beta1, 0.95)
    adam_opt.step()
    model.clamp_identity_dispersion_and_scalars_()


def run_smith_momentum_free_lr_range_test(
    stream: LongSeqFineWebStream, nc_spec: NoClickVacuumCholeskySpec,
    range_steps: int = DEFAULT_SMITH_RANGE_STEPS, min_lr_mult: float = DEFAULT_SMITH_MIN_LR_MULT,
    max_lr_mult: float = DEFAULT_SMITH_MAX_LR_MULT,
) -> dict[str, Any]:
    r"""Run Leslie Smith (`arXiv:1506.01186`) continuous exponential LR Range Test for $\beta = 0.0$ vs. $\beta = 0.70$ (`<= 8` args)."""
    device = stream.device
    torch.manual_seed(nc_spec.seed)
    base_model = NoClickVacuumCholeskyGPT(nc_spec).to(device=device)
    initialize_and_calibrate_variable_rank_cholesky_model_(base_model, stream)
    base_model.clamp_identity_dispersion_and_scalars_()
    base_state = copy.deepcopy(base_model.state_dict())

    val_x, val_y = stream.val_batches[0]
    sweep_reports: dict[str, Any] = {}

    for regime_name, beta_val, stiefel_mode in [
        ("momentum_free_beta_000", 0.0, StiefelMomentumMode.RIEMANNIAN_TANGENT_TRANSPORTED),
        ("momentum_ref_beta_070", 0.70, StiefelMomentumMode.RIEMANNIAN_TANGENT_SNR_GATED),
    ]:
        model = copy.deepcopy(base_model)
        model.load_state_dict(base_state)
        optimizers = build_variable_rank_cholesky_optimizers(model)
        optimizers[0].momentum_mode = stiefel_mode

        lr_mults, train_losses, probe_val_losses = [], [], []
        model.train()

        for s_idx in range(1, range_steps + 1):
            frac = float(s_idx - 1) / float(max(1, range_steps - 1))
            mu = float(min_lr_mult) * ((float(max_lr_mult) / float(min_lr_mult)) ** frac)
            idx_in, targets = stream.sample_train_batch(s_idx)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss = model.forward_loss(idx_in, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            _apply_global_optimizer_step(model, optimizers, idx_in, targets, lr_mult=mu, mom_beta=beta_val)

            with torch.no_grad():
                model.eval()
                v_loss = float(model.forward_loss(val_x, val_y).item())
                model.train()

            lr_mults.append(round(mu, 6))
            train_losses.append(round(float(loss.detach().item()), 5))
            probe_val_losses.append(round(v_loss, 5))

        log_mu = np.log(np.array(lr_mults, dtype=np.float64))
        val_np = np.array(probe_val_losses, dtype=np.float64)
        kernel = np.array([0.12, 0.22, 0.32, 0.22, 0.12], dtype=np.float64)
        smoothed = np.convolve(np.pad(val_np, (2, 2), mode="edge"), kernel, mode="valid")
        d_loss_d_log_mu = np.gradient(smoothed, log_mu)

        min_idx = int(np.argmin(smoothed))
        post_init = min(max(1, range_steps // 6), max(1, min_idx - 1))
        steep_idx = post_init + int(np.argmin(d_loss_d_log_mu[post_init : max(post_init + 1, min_idx + 1)]))
        mu_steep, mu_min_loss = float(lr_mults[steep_idx]), float(lr_mults[min_idx])
        mu_peak_star = float(np.clip(0.85 * mu_min_loss, 0.50, 4.50))
        mu_base_star = round(mu_peak_star / 4.0, 4)

        sweep_reports[regime_name] = {
            "momentum_beta": beta_val, "range_steps": range_steps, "min_lr_mult": min_lr_mult, "max_lr_mult": max_lr_mult,
            "step0_probe_val_ce": probe_val_losses[0], "min_smoothed_probe_val_ce": round(float(smoothed[min_idx]), 4),
            "mu_steepest_descent": round(mu_steep, 4), "mu_min_loss_before_divergence": round(mu_min_loss, 4),
            "recommended_1cycle_peak_lr_mult": round(mu_peak_star, 4), "recommended_1cycle_base_lr_mult": mu_base_star,
            "effective_stiefel_peak_lr": round(DEFAULT_SHALLOW_STIEFEL_LR * mu_peak_star, 5),
            "effective_wte_peak_lr": round(DEFAULT_SHALLOW_WTE_LR * mu_peak_star, 5),
            "effective_readout_peak_lr": round(DEFAULT_SHALLOW_READOUT_LR * mu_peak_star, 5),
            "trajectory_sample": [
                {"step": i + 1, "lr_mult": lr_mults[i], "train_ce": train_losses[i], "probe_val_ce": probe_val_losses[i], "smoothed_val_ce": round(float(smoothed[i]), 4)}
                for i in range(0, range_steps, max(1, range_steps // 16))
            ] + [{"step": range_steps, "lr_mult": lr_mults[-1], "train_ce": train_losses[-1], "probe_val_ce": probe_val_losses[-1], "smoothed_val_ce": round(float(smoothed[-1]), 4)}],
        }
        del model, optimizers
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del base_model, base_state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    b0_peak = sweep_reports["momentum_free_beta_000"]["recommended_1cycle_peak_lr_mult"]
    b70_peak = sweep_reports["momentum_ref_beta_070"]["recommended_1cycle_peak_lr_mult"]
    sweep_reports["beta0_over_beta070_peak_lr_ratio"] = round(float(b0_peak / max(1e-6, b70_peak)), 4)
    return sweep_reports


def compute_smith_1cycle_lr_mult(
    step: int, total_steps: int, peak_lr_mult: float, pct_start: float = DEFAULT_SMITH_1CYCLE_PCT_START,
) -> float:
    """Compute Leslie Smith's (`arXiv:1708.07120`) 1-Cycle Super-Convergence LR multiplier at `step` (`<= 8` args)."""
    up_steps = max(1, int(float(pct_start) * total_steps))
    base_mult, min_mult = float(peak_lr_mult) / 4.0, float(peak_lr_mult) * 0.08
    if step <= up_steps:
        return base_mult + (float(peak_lr_mult) - base_mult) * (float(step - 1) / float(max(1, up_steps - 1)))
    down_frac = float(step - up_steps) / float(max(1, total_steps - up_steps))
    return min_mult + (float(peak_lr_mult) - min_mult) * 0.5 * (1.0 + math.cos(math.pi * down_frac))


def _train_and_eval_noclick_and_smith_variants(
    stream: LongSeqFineWebStream, variants: list[tuple[str, NoClickVacuumCholeskySpec, str, float, float, bool]],
    steps: int, eval_interval: int, wave_id: str = "W60",
) -> list[dict[str, Any]]:
    """Run training and evaluation across `NoClickVacuumCholeskySpec` and Smith SGD variants (`<= 8` args)."""
    device = stream.device
    results: list[dict[str, Any]] = []
    tokens_per_step = stream.batch_size * stream.seq_len

    for variant_id, nc_spec, desc, fixed_beta, peak_lr_mult, use_smith_1cycle in variants:
        torch.manual_seed(nc_spec.seed)
        model = NoClickVacuumCholeskyGPT(nc_spec).to(device=device)
        init_ms, _ = initialize_and_calibrate_variable_rank_cholesky_model_(model, stream)
        model.clamp_identity_dispersion_and_scalars_()

        run_key = f"plan080_{variant_id.lower()}_{steps}steps"
        intent = ExperimentIntentSpec(
            run_key=run_key, wave_id=wave_id, plan_id="EXPERIMENT_PLAN_080", task_domain="fineweb_speedrun",
            hypothesis_intent=desc,
            architecture_summary=f"NoClickVacuumCholeskyGPT(L=4, d=512, R={nc_spec.num_cholesky_modes}, d_vac={model.d_vac}, sink={nc_spec.use_noclick_vacuum_sink}, beta={fixed_beta})",
            hardware_budget="1x RTX 5060 Ti 16GB", batch_size=stream.batch_size, seq_len=stream.seq_len,
            total_steps=steps, target_tokens_M=round((steps * tokens_per_step) / 1e6, 3),
        )
        exp_logger = StandardizedExperimentLogger(intent)
        optimizers = build_variable_rank_cholesky_optimizers(model)
        if fixed_beta == 0.0:
            optimizers[0].momentum_mode = StiefelMomentumMode.RIEMANNIAN_TANGENT_TRANSPORTED

        step0_eval = evaluate_noclick_vacuum_cholesky_model(model, stream)
        step0_eval["step"], step0_eval["tokens_M"] = 0, 0.0
        exp_logger.log_eval(0, {**step0_eval, "top1_acc": round(100.0 * step0_eval["top1_acc"], 2), "top5_acc": round(100.0 * step0_eval["top5_acc"], 2)})
        print(
            f"[{variant_id} (R={nc_spec.num_cholesky_modes}, sink={nc_spec.use_noclick_vacuum_sink}, beta={fixed_beta:.2f})] "
            f"Step    0 | Pure Neural Val CE: {step0_eval['val_ce']:.4f} (PPL {step0_eval['val_ppl']:.2f}) | "
            f"Modes: {step0_eval['effective_active_branches']:.2f}/{nc_spec.num_cholesky_modes} | p_noclick: {step0_eval['mean_noclick_prob']:.3f}",
            flush=True,
        )

        curve: list[dict[str, Any]] = [step0_eval]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        model.train()

        for step in range(1, steps + 1):
            if use_smith_1cycle:
                lr_mult = compute_smith_1cycle_lr_mult(step, steps, peak_lr_mult=peak_lr_mult)
            else:
                warm_steps = max(1, int(0.08 * steps))
                base_sched = (
                    0.20 + 0.80 * (float(step - 1) / float(max(1, warm_steps - 1)))
                    if step <= warm_steps
                    else 0.12 + 0.88 * 0.5 * (1.0 + math.cos(math.pi * float(step - warm_steps) / float(max(1, steps - warm_steps))))
                )
                lr_mult = float(peak_lr_mult) * base_sched

            mom_beta = float(fixed_beta) if fixed_beta == 0.0 else compute_riemannian_momentum_beta(step, steps, QuantumMomentumScheduleMode.LOW_STEADY_MOMENTUM_070)
            idx_in, targets = stream.sample_train_batch(step)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss = model.forward_loss(idx_in, targets)
            train_loss_val = float(loss.detach().item())
            loss.backward()
            del loss
            if peak_lr_mult > 1.25:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            _apply_global_optimizer_step(model, optimizers, idx_in, targets, lr_mult=lr_mult, mom_beta=mom_beta)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            if step % eval_interval == 0 or step == steps:
                ev = evaluate_noclick_vacuum_cholesky_model(model, stream)
                ev["step"], ev["tokens_M"] = step, round((step * tokens_per_step) / 1e6, 3)
                ev["train_loss"], ev["lr_mult"], ev["mom_beta"] = round(train_loss_val, 4), round(lr_mult, 4), round(mom_beta, 4)
                curve.append(ev)
                exp_logger.log_eval(step, {**ev, "top1_acc": round(100.0 * ev["top1_acc"], 2), "top5_acc": round(100.0 * ev["top5_acc"], 2)})
                print(
                    f"[{variant_id} (R={nc_spec.num_cholesky_modes}, sink={nc_spec.use_noclick_vacuum_sink}, beta={mom_beta:.2f})] "
                    f"Step {step:4d} ({ev['tokens_M']:5.2f}M tok, lr={lr_mult:.2f}x) | "
                    f"Train CE: {ev['train_loss']:.4f} | Pure Neural Val CE: {ev['val_ce']:.4f} (PPL {ev['val_ppl']:.2f}) | "
                    f"Top1: {ev['top1_acc']*100:.2f}% | Top5: {ev['top5_acc']*100:.2f}% | "
                    f"Modes: {ev['effective_active_branches']:.2f}/{nc_spec.num_cholesky_modes} | "
                    f"p_noclick [mean/id/sem]: {ev['mean_noclick_prob']:.3f}/{ev['id_head_noclick_prob']:.3f}/{ev['sem_head_noclick_prob']:.3f}",
                    flush=True,
                )

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - t_start
        results.append({
            "variant_id": variant_id, "variant_mode": nc_spec.variant_mode.value, "description": desc,
            "num_cholesky_modes": nc_spec.num_cholesky_modes, "vacuum_dim": model.d_vac,
            "use_noclick_vacuum_sink": nc_spec.use_noclick_vacuum_sink, "momentum_beta": fixed_beta,
            "peak_lr_mult": peak_lr_mult, "use_smith_1cycle": use_smith_1cycle,
            "init_ms": round(init_ms, 2), "elapsed_sec": round(elapsed_s, 2),
            "step0": step0_eval, "final": curve[-1], "curve": curve,
        })
        del model, optimizers
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def run_noclick_vacuum_ablation_benchmark(
    steps: int = DEFAULT_FINEWEB_STEPS, eval_interval: int = DEFAULT_FINEWEB_EVAL_INTERVAL, device_str: str = "cuda",
) -> list[dict[str, Any]]:
    """Run the 300-step (`2.46M`-token) Quantum No-Click POVM / Vacuum Sink ablation (`N1, N2`, `<= 8` args)."""
    device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    stream = LongSeqFineWebStream(device=device)
    vac_d = DEFAULT_VACUUM_SUBSPACE_DIM
    variants = [
        ("N1_RANK8_NOCLICK_VACUUM_POVM", NoClickVacuumCholeskySpec(NoClickVacuumVariantMode.N1_RANK8_NOCLICK_VACUUM_POVM, 8, vac_d, True), "R=8 Uncollapsed Cholesky + Readout H_vac + Trunk Quantum No-Click POVM Vacuum Sink", 0.70, 1.0, False),
        ("N2_RANK12_NOCLICK_VACUUM_POVM", NoClickVacuumCholeskySpec(NoClickVacuumVariantMode.N2_RANK12_NOCLICK_VACUUM_POVM, 12, vac_d, True), "R=12 Uncollapsed Cholesky + Readout H_vac + Trunk Quantum No-Click POVM Vacuum Sink", 0.70, 1.0, False),
    ]
    return _train_and_eval_noclick_and_smith_variants(stream, variants, steps, eval_interval, wave_id="W60")


def run_rank12_vs_rank8_long_scaling_benchmark(
    steps: int = DEFAULT_LONG_FINEWEB_STEPS, eval_interval: int = DEFAULT_LONG_FINEWEB_EVAL_INTERVAL, device_str: str = "cuda",
) -> list[dict[str, Any]]:
    """Run the 1,200-step (`9.83M`-token) $R = 12$ scaling run (`N4` with Readout Vacuum + No-Click Sink, `<= 8` args)."""
    device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    stream = LongSeqFineWebStream(device=device)
    vac_d = DEFAULT_VACUUM_SUBSPACE_DIM
    variants = [
        ("N4_RANK12_NOCLICK_VACUUM_SCALING_1200", NoClickVacuumCholeskySpec(NoClickVacuumVariantMode.N4_RANK12_NOCLICK_VACUUM_SCALING_1200, 12, vac_d, True), "1,200-step (9.83M-token) 100% Pure-Neural Scaling Run at R=12 + Readout H_vac + Trunk Quantum No-Click POVM Vacuum Sink", 0.70, 1.0, False),
    ]
    return _train_and_eval_noclick_and_smith_variants(stream, variants, steps, eval_interval, wave_id="W60_LONG")


def run_smith_momentum_free_superconvergence_benchmark(
    steps: int = DEFAULT_FINEWEB_STEPS, eval_interval: int = DEFAULT_FINEWEB_EVAL_INTERVAL,
    range_steps: int = DEFAULT_SMITH_RANGE_STEPS, device_str: str = "cuda",
) -> dict[str, Any]:
    r"""Run Leslie Smith $\beta = 0$ Exponential LR Range Test + 300-step 1-Cycle Super-Convergence comparison (`S0, S1`, `<= 8` args)."""
    device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    stream = LongSeqFineWebStream(device=device)
    vac_d = DEFAULT_VACUUM_SUBSPACE_DIM
    probe_spec = NoClickVacuumCholeskySpec(NoClickVacuumVariantMode.S1_MOMENTUM_FREE_SGD_SMITH_1CYCLE, 8, vac_d, True)
    range_report = run_smith_momentum_free_lr_range_test(stream, probe_spec, range_steps=range_steps)
    opt_peak_mult = float(range_report["momentum_free_beta_000"]["recommended_1cycle_peak_lr_mult"])

    variants = [
        ("S0_MOMENTUM_FREE_SGD_BASELINE_LR", NoClickVacuumCholeskySpec(NoClickVacuumVariantMode.S0_MOMENTUM_FREE_SGD_BASELINE_LR, 8, vac_d, True), "Momentum-Free (beta=0.0) Riemannian/Spectral SGD at Unscaled Baseline Step Size (1.0x)", 0.0, 1.0, False),
        ("S1_MOMENTUM_FREE_SGD_SMITH_1CYCLE", NoClickVacuumCholeskySpec(NoClickVacuumVariantMode.S1_MOMENTUM_FREE_SGD_SMITH_1CYCLE, 8, vac_d, True), f"Momentum-Free (beta=0.0) Riemannian/Spectral SGD with Leslie Smith 1-Cycle Super-Convergence (peak={opt_peak_mult:.2f}x)", 0.0, opt_peak_mult, True),
    ]
    runs = _train_and_eval_noclick_and_smith_variants(stream, variants, steps, eval_interval, wave_id="W60_SMITH")
    return {"lr_range_test": range_report, "superconvergence_runs": runs}


__all__ = [
    "CHOLESKY_MODE_ROLE_DESCRIPTIONS", "DEFAULT_INIT_NOCLICK_POS_LOGIT", "DEFAULT_INIT_NOCLICK_SINK_BIAS",
    "DEFAULT_NOCLICK_HIGHWAY_SHIELD_STRENGTH", "DEFAULT_NUM_FREQ_DECILES", "DEFAULT_SMITH_1CYCLE_PCT_START",
    "DEFAULT_SMITH_MAX_LR_MULT", "DEFAULT_SMITH_MIN_LR_MULT", "DEFAULT_SMITH_RANGE_STEPS",
    "NoClickVacuumCholeskyGPT", "NoClickVacuumCholeskySpec", "NoClickVacuumVariantMode",
    "_FusedNoClickVacuumHermitianBornAttnFn", "_NoClickVacuumCholeskyWavepacketBlock",
    "build_zipf_frequency_decile_lookup", "compute_noclick_vacuum_sink_scores", "compute_smith_1cycle_lr_mult",
    "evaluate_noclick_vacuum_cholesky_model", "extract_per_mode_readout_diagnostics",
    "fused_noclick_vacuum_cholesky_ket_attention", "profile_per_mode_cholesky_spectrum_and_vacuum_angles",
    "run_noclick_vacuum_ablation_benchmark", "run_rank12_vs_rank8_long_scaling_benchmark",
    "run_smith_momentum_free_lr_range_test", "run_smith_momentum_free_superconvergence_benchmark",
]
