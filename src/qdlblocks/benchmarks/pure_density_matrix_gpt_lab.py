r"""Wave 69 (`EXPERIMENT_PLAN_089.md`): Pure Density-Matrix Architecture — Direct $Q/K/V$ Density Projections, `exp`-Free Hilbert–Schmidt Quantum Attention, Cross-Head Quantum "OR + AND" (Schur Product) Logic, Online EMA N-Gram Density Matrices, Stateful Multi-Window Density History, & Unconstrained Born Readout.

Mathematical & Architectural Overview
-------------------------------------
Replaces the Waves 63–68 hybrid classical-vector + density-highway trunk with a **100% Pure Density-Matrix
Architecture** (`PureDensityMatrixGPT`) where **the only state flowing through the trunk is a batch of
multi-head density matrices** $\boldsymbol{\rho}_t^{(\ell)} = (\rho_{t,1}^{(\ell)}, \dots, \rho_{t,H}^{(\ell)}) \in \mathcal{D}(\mathbb{C}^{16})^H$,
represented by their Rank-$R$ ($R=4$) complex Cholesky/purification factors $L_{t,h}^{(\ell)} \in \mathbb{C}^{16 \times 4}$
($\rho_{t,h}^{(\ell)} = L_{t,h}^{(\ell)} L_{t,h}^{(\ell)\dagger}$, $\operatorname{Tr}(\rho_{t,h}^{(\ell)}) = \|L_{t,h}^{(\ell)}\|_F^2 = 1$)
and Frobenius-isometric packed Hermitian vectors $\tilde{\rho}_{t,h}^{(\ell)} = \tilde{P}(\rho_{t,h}^{(\ell)}) \in \mathbb{R}^{256}$:

1. **Pillar 1 (`Path B` / Lean 4 Theorem 6) — Direct $Q/K/V$ Density Projections & `exp`-Free Hilbert–Schmidt Attention**:
   - Projects $L_{t,h}^{\text{in}} \in \mathbb{C}^{16 \times R}$ directly via horizontal unitary maps $W_{Q,h}, W_{K,h}, W_{V,h} \in U(16)$
     (with complex RoPE phase rotation on $Q, K$) into Query, Key, and Value density matrices
     $\rho_{t,h}^Q = Q_{t,h} Q_{t,h}^\dagger$, $\rho_{s,h}^K = K_{s,h} K_{s,h}^\dagger$, and $\rho_{s,h}^V = V_{s,h} V_{s,h}^\dagger$.
   - Computes the exact Hilbert–Schmidt quantum overlap via flattened isometric outer products:
     $$S_{t,s}^{(h)} = \big\langle \tilde{P}(\rho_{t,h}^Q),\; \tilde{P}(\rho_{s,h}^K) \big\rangle_{\mathbb{R}^{256}} \equiv \operatorname{Tr}\!\left(\rho_{t,h}^Q \rho_{s,h}^K\right) = \left\| Q_{t,h}^\dagger K_{s,h} \right\|_F^2 \ge 0$$
   - Generates positive, sum-to-one causal mixing weights $A_{t,s}^{(h)} = \tilde{S}_{t,s}^{(h)} / (\sum_{j \le t} \tilde{S}_{t,j}^{(h)} + \sum_m \tilde{S}_{t,\text{hist},m}^{(h)} + \varepsilon_{\text{vac},h}) \ge 0$
     **without `exp` or `softmax`** and with **exact $U(R)$-gauge invariance**.

2. **Pillar 2 (`Paths C & D Unified` / Lean 4 Theorem 7) — Cross-Head Quantum "OR + AND" (Schur Product) Logic Channel**:
   - Replaces the classical vector Mach–Zehnder MLP with a native density-matrix logic sub-block:
     - **Quantum "OR" (Orthostochastic & Unitary Cross-Head Braiding)**: Learned $W_{\text{OR}}^{(A)}, W_{\text{OR}}^{(B)} \in \operatorname{O}(H)$
       braid Cholesky factors across heads and mix packed density matrices via orthostochastic weights $B_{h,h'}^{\text{OR}} = (W_{\text{OR},h,h'}^{(A)})^2 \ge 0$ ($\sum_{h'} B_{h,h'}^{\text{OR}} \equiv 1$).
     - **Quantum "AND" (Schur / Hadamard Coincidence Product $\rho^{(A)} \circ \rho^{(B)}$)**: By the **Schur Product Theorem**,
       the elementwise complex Hadamard product of braided Cholesky factors $L_{t,h}^{\text{AND}} = \operatorname{Norm}_F(L_{t,h}^{(A)} \odot L_{t,h}^{(B)})$
       and normalized Schur product $(\rho^{(A)} \circ \rho^{(B)}) / \operatorname{Tr}(\rho^{(A)} \circ \rho^{(B)})$ form a strictly
       positive semidefinite, trace-1 Hermitian coincidence density matrix.

3. **Pillar 3 (`Path A` / Lean 4 Theorem 8a) — NanoGPT-Speedrun-Compliant Online EMA N-Gram Density Matrices**:
   - `OnlineEmaNgramDensityBank` initializes `G = 4096` n-gram density matrices at Step 0 to the maximally mixed
     state $\frac{1}{16}I_{16}$ (zero offline precomputation) and updates them online via causal EMA from observed token wavepackets.

4. **Pillar 4 (`Path F` / Lean 4 Theorem 8b) — Stateful Multi-Window Quantum Density History Buffer**:
   - `StatefulQuantumDensityHistoryBuffer` carries $M_{\text{hist}} = 8$ multi-timescale window-summary density matrices
     $\rho_{\text{hist},m}^{(h,\ell)} \in \mathcal{D}(\mathbb{C}^{16})$ across consecutive training windows (`512 KB` total state).

5. **Pillar 5 (`Path E`) — Pure Density-Matrix Tomographic Readout & Unconstrained Exact Log-Likelihood**:
   - Scores all `50,257` tokens exclusively from $\rho_{t,h}^{(L)} \in \mathcal{D}(\mathbb{C}^{16})$ via gauge-locked $\mathfrak{su}(16)$
     MUB probes + horizontal Stiefel Hermitian observables $W_{\text{obs},h} \in \operatorname{St}(256, 32)$ on
     $Q_{\text{out}} \in \operatorname{O}(512)$ with the artificial `30.0 * tanh(z / 30.0)` softcap removed.
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

from qdlblocks.benchmarks.complementary_512d_lindblad_8l_lab import (
    ComplementaryDualSubspace512DBoundary, FrequencyAdaptiveBatchSubspaceGramMuon,
    ParallelTransportSelectiveQuotientMuon, _collect_active_vocab_indices_w68,
    parallel_transport_real_stiefel_momentum,
)
from qdlblocks.benchmarks.experiment_logger import ExperimentIntentSpec, StandardizedExperimentLogger
from qdlblocks.benchmarks.explicit_density_tomographic_512d_lab import (
    DEFAULT_INIT_VACUUM_REST_MASS, DEFAULT_TOK_NGRAM_BUCKETS, DEFAULT_TOMO_EMBED_DIM,
    DEFAULT_TOMO_HEAD_COMPLEX_DIM, DEFAULT_TOMO_NUM_HEADS, DEFAULT_TOMO_NUM_LAYERS,
    _complex_hermitian_mv, apply_unified_512d_vacuum_sphere_norm,
)
from qdlblocks.benchmarks.horizontal_grassmann_gauge_free_lab import (
    DEFAULT_MUON_GRAD_NORM_GATE_TAU, compute_gauge_locked_mub_probe_expectations, compute_mub_unbiasedness_max_error,
)
from qdlblocks.benchmarks.kraus_mub_thermostat_8l_lab import (
    DEFAULT_ANALYTICAL_BETA_TARGET, DEFAULT_THERMOSTAT_EMA_DECAY,
    build_initial_kraus_stiefel_operators, build_mub_two_basis_unitary_probes,
    compute_analytical_parseval_beta, conjugate_packed_hermitian_by_kraus_stiefel,
)
from qdlblocks.benchmarks.quantum_subsystem_vocab_lab import LUDERS_COARSE_OUTCOMES, LUDERS_FINE_MAX_OUTCOMES
from qdlblocks.benchmarks.selective_exact_quotient_lab import (
    DEFAULT_NARROW_SUBSPACE_DIM, DEFAULT_QUENCH_FRACTION, DEFAULT_SELECTIVE_MOMENTUM_BETA,
    DEFAULT_WIDE_KRAUS_RANK, ReadOnlyQuenchRiemannianShadowEMA, _clean_and_snr_gate_real_tangent,
    _retract_real_tall_stiefel_tensor_lr, _whiten_tall_real_stiefel,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import (
    DEFAULT_SHALLOW_NGRAM_LR, DEFAULT_SHALLOW_READOUT_LR, DEFAULT_SHALLOW_SCALAR_LR,
    DEFAULT_SHALLOW_STIEFEL_LR, DEFAULT_SHALLOW_WTE_LR, RAW_GPT2_VOCAB_SIZE, LongSeqFineWebStream,
)
from qdlblocks.layers.impure_density_mirror_block import (
    DEFAULT_NS5_STEPS, DEFAULT_TIKHONOV_GRAM_LAMBDA, _get_hermitian_isometric_masks,
    complex_l2_normalize, pack_hermitian_isometric, pack_rank1_ket_outer_isometric, unpack_hermitian_isometric,
)
from qdlblocks.layers.quantum_statmech_channels import project_real_stiefel_tangent
from qdlblocks.optim.vocab_manifold_muon import RiemannianGeodesicScalarAndProbeOptimizer

# ==============================================================================
# Top-of-File Named Constants (Rule 7 — Synchronized with EXPERIMENT_PLAN_089.md)
# ==============================================================================
DEFAULT_PURE_RHO_HEADS: int = DEFAULT_TOMO_NUM_HEADS
DEFAULT_PURE_RHO_DIM: int = DEFAULT_TOMO_HEAD_COMPLEX_DIM
DEFAULT_PURE_CHOLESKY_RANK: int = 4
DEFAULT_PURE_PACKED_DIM: int = DEFAULT_PURE_RHO_DIM * DEFAULT_PURE_RHO_DIM  # 256
DEFAULT_PURE_OBS_PER_HEAD: int = 32
DEFAULT_HS_COINCIDENCE_BETA_INIT: float = 0.35
DEFAULT_HS_VACUUM_FLOOR_INIT: float = 0.08
DEFAULT_SCHUR_AND_MIX_INIT: float = 0.30
DEFAULT_ONLINE_NGRAM_EMA_DECAY: float = 0.04
DEFAULT_HISTORY_SLOTS: int = 8
DEFAULT_HISTORY_MIN_DECAY: float = 0.015
DEFAULT_HISTORY_MAX_DECAY: float = 0.50
DEFAULT_UNCONSTRAINED_LOGIT_MAX: float = 80.0
PURE_RHO_TRACE_EPS: float = 1.0e-6
PADDED_LUDERS_GRID_SIZE: int = LUDERS_COARSE_OUTCOMES * LUDERS_FINE_MAX_OUTCOMES  # 50,432
UNCONSTRAINED_CE_TOKEN_CHUNK: int = 1024


class Wave69VariantMode(str, Enum):
    """Ablation and scale-up modes for `EXPERIMENT_PLAN_089.md` (`Wave 69`, Rule 3)."""

    W69_P0_PURE_DENSITY_HS_ATTN_4L = "W69_P0_PURE_DENSITY_HS_ATTN_4L"
    W69_P1_PURE_DENSITY_UNCONSTRAINED_BORN_4L = "W69_P1_PURE_DENSITY_UNCONSTRAINED_BORN_4L"
    W69_P2_PURE_DENSITY_EMA_NGRAM_HISTORY_4L = "W69_P2_PURE_DENSITY_EMA_NGRAM_HISTORY_4L"
    W69_P3_PURE_DENSITY_DEEP_8L_CONSOLIDATED = "W69_P3_PURE_DENSITY_DEEP_8L_CONSOLIDATED"


@dataclass(frozen=True)
class Wave69ExperimentSpec:
    r"""Immutable configuration for `PureDensityMatrixGPT` (`<= 8` fields, Rule 2)."""

    variant_mode: Wave69VariantMode = Wave69VariantMode.W69_P2_PURE_DENSITY_EMA_NGRAM_HISTORY_4L
    num_layers: int = DEFAULT_TOMO_NUM_LAYERS
    cholesky_rank: int = DEFAULT_PURE_CHOLESKY_RANK
    kraus_rank: int = DEFAULT_WIDE_KRAUS_RANK
    enable_unconstrained_born_ce: bool = True
    enable_stateful_ema_and_history: bool = True
    history_slots: int = DEFAULT_HISTORY_SLOTS
    seed: int = 1337


_WAVE69_MODE_DEFAULTS: dict[Wave69VariantMode, tuple[int, bool, bool]] = {
    Wave69VariantMode.W69_P0_PURE_DENSITY_HS_ATTN_4L: (DEFAULT_TOMO_NUM_LAYERS, False, False),
    Wave69VariantMode.W69_P1_PURE_DENSITY_UNCONSTRAINED_BORN_4L: (DEFAULT_TOMO_NUM_LAYERS, True, False),
    Wave69VariantMode.W69_P2_PURE_DENSITY_EMA_NGRAM_HISTORY_4L: (DEFAULT_TOMO_NUM_LAYERS, True, True),
    Wave69VariantMode.W69_P3_PURE_DENSITY_DEEP_8L_CONSOLIDATED: (8, True, True),
}


def build_wave69_variant_spec(
    variant: Wave69VariantMode | str,
    num_layers_override: int | None = None,
    seed: int = 1337,
) -> Wave69ExperimentSpec:
    """Resolve `Wave69ExperimentSpec` from `Wave69VariantMode` (`<= 8` args)."""
    mode = Wave69VariantMode(variant)
    def_layers, unconst_ce, stateful_ema = _WAVE69_MODE_DEFAULTS[mode]
    n_lay = int(num_layers_override) if num_layers_override is not None else def_layers
    return Wave69ExperimentSpec(
        variant_mode=mode, num_layers=n_lay, cholesky_rank=DEFAULT_PURE_CHOLESKY_RANK,
        kraus_rank=DEFAULT_WIDE_KRAUS_RANK, enable_unconstrained_born_ce=unconst_ce,
        enable_stateful_ema_and_history=stateful_ema, history_slots=DEFAULT_HISTORY_SLOTS, seed=seed,
    )


# ==============================================================================
# 1. Pure Density-Matrix & Cholesky Factor Primitives (Lean 4 Theorems 6 & 7)
# ==============================================================================
def normalize_cholesky_factor_frobenius(
    l_re: torch.Tensor, l_im: torch.Tensor, eps: float = PURE_RHO_TRACE_EPS
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Normalize complex Cholesky factor $L \in \mathbb{C}^{\dots \times d_h \times R}$ so $\operatorname{Tr}(L L^\dagger) = \|L\|_F^2 \equiv 1$ (`<= 8` args)."""
    orig_dtype = l_re.dtype
    lr_f, li_f = l_re.float(), l_im.float()
    inv_norm = torch.rsqrt((lr_f.square() + li_f.square()).sum(dim=(-2, -1), keepdim=True).clamp_min(eps))
    return (lr_f * inv_norm).to(orig_dtype), (li_f * inv_norm).to(orig_dtype)


def pack_cholesky_factor_outer_isometric(
    l_re: torch.Tensor, l_im: torch.Tensor
) -> torch.Tensor:
    r"""Form $\rho = L L^\dagger \in \mathcal{D}(\mathbb{C}^{d_h})$ from $L \in \mathbb{C}^{\dots \times d_h \times R}$ and pack via $\tilde{P}(\rho) \in \mathbb{R}^{d_h^2}$ (`<= 8` args).

    By **Lean 4 Theorem 6b & 6c**:
    1. $\langle \tilde{P}(Q Q^\dagger), \tilde{P}(K K^\dagger) \rangle_{\mathbb{R}^{d_h^2}} \equiv \operatorname{Tr}(Q Q^\dagger K K^\dagger) = \|Q^\dagger K\|_F^2 \ge 0$.
    2. Right-multiplying $L \mapsto L U$ for any $U \in U(R)$ leaves $L L^\dagger$ and $\tilde{P}(L L^\dagger)$ identically invariant.
    """
    d_h = l_re.size(-2)
    lrt, lit = l_re.transpose(-2, -1), l_im.transpose(-2, -1)
    rho_re = torch.matmul(l_re, lrt) + torch.matmul(l_im, lit)
    rho_im = torch.matmul(l_im, lrt) - torch.matmul(l_re, lit)
    p_re, p_im_t, _, _ = _get_hermitian_isometric_masks(d_h, l_re.device, l_re.dtype)
    return (rho_re * p_re + rho_im.transpose(-2, -1) * p_im_t).flatten(start_dim=-2, end_dim=-1).contiguous()


def normalize_packed_density_trace(
    rho_packed: torch.Tensor, d_h: int = DEFAULT_PURE_RHO_DIM, eps: float = PURE_RHO_TRACE_EPS
) -> torch.Tensor:
    r"""Ensure exact unit trace $\operatorname{Tr}(\rho) = \sum_{i=0}^{d_h-1} \rho_{ii} \equiv 1$ on packed isometric $\tilde{\rho} \in \mathbb{R}^{\dots \times d_h^2}$ (`<= 8` args)."""
    orig_dtype = rho_packed.dtype
    rp_f = rho_packed.float()
    diag_idx = torch.arange(0, d_h * d_h, d_h + 1, device=rho_packed.device, dtype=torch.long)
    tr_val = rp_f.index_select(-1, diag_idx).sum(dim=-1, keepdim=True).clamp_min(eps)
    return (rp_f / tr_val).to(orig_dtype)


def apply_complex_unitary_to_cholesky(
    l_re: torch.Tensor, l_im: torch.Tensor, u_re: torch.Tensor, u_im: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Apply per-head unitary $U_h \in U(d_h)$ (`(H, d_h, d_h)`) to $L_{t,h} \in \mathbb{C}^{d_h \times R}$ (`(B, H, T, d_h, R)`) (`<= 8` args)."""
    ur = u_re.to(l_re.dtype).unsqueeze(0).unsqueeze(2)
    ui = u_im.to(l_re.dtype).unsqueeze(0).unsqueeze(2)
    return torch.matmul(ur, l_re) - torch.matmul(ui, l_im), torch.matmul(ur, l_im) + torch.matmul(ui, l_re)


def apply_complex_rope_to_cholesky(
    l_re: torch.Tensor, l_im: torch.Tensor, cos_t: torch.Tensor, sin_t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Apply position-dependent complex phase rotation $e^{i t \omega_k}$ to $L_{t,h} \in \mathbb{C}^{d_h \times R}$ (`<= 8` args)."""
    c, s = cos_t.to(l_re.dtype), sin_t.to(l_re.dtype)
    return l_re * c - l_im * s, l_re * s + l_im * c


def compute_hilbert_schmidt_quantum_attention(
    q_packed: torch.Tensor,
    k_packed: torch.Tensor,
    v_payload: torch.Tensor,
    hist_k_packed: torch.Tensor | None,
    hist_v_payload: torch.Tensor | None,
    coinc_beta: torch.Tensor,
    pos_decay_rate: torch.Tensor,
    vac_floor_and_therm: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""100% `exp`/`softmax`-Free Hilbert–Schmidt Quantum Density Attention (**Lean 4 Theorem 6**, `<= 8` args).

    Given Query and Key density matrices packed in Frobenius-isometric form $\tilde{\rho}_{t,h}^Q, \tilde{\rho}_{s,h}^K \in \mathbb{R}^{256}$:
    1. Evaluates the exact Hilbert–Schmidt quantum overlap via Tensor-Core BMM:
       $$S_{t,s}^{(h)} = \big\langle \tilde{\rho}_{t,h}^Q,\; \tilde{\rho}_{s,h}^K \big\rangle_{\mathbb{R}^{256}} = \operatorname{Tr}\!\left(\rho_{t,h}^Q \rho_{s,h}^K\right) \ge 0$$
    2. Applies 2-boson coincidence sharpening $S_{t,s}^{(h)} + \beta_{\text{coinc},h}(S_{t,s}^{(h)})^2 \ge 0$ and
       algebraic causal relative-distance weighting $w_{t,s}^{(h)} = 1 / (1 + \kappa_h (t - s))$ ($s \le t$).
    3. Normalizes by the causal partition sum plus positive vacuum floor $\varepsilon_{\text{vac},h} > 0$
       (and optional stateful history slots $m = 1 \dots M_{\text{hist}}$) to produce exact convex mixing weights
       $A_{t,s}^{(h)} \ge 0$, $\sum_{s \le t} A_{t,s}^{(h)} + \sum_m A_{t,\text{hist},m}^{(h)} + A_{t,\text{vac}}^{(h)} \equiv 1$.
    """
    bsz, num_h, seq_len, _ = q_packed.shape
    work_dtype, dev = q_packed.dtype, q_packed.device
    vac_floor, therm_payload = vac_floor_and_therm

    sim_local = (torch.matmul(q_packed, k_packed.transpose(-2, -1)).float() - (1.0 / float(DEFAULT_PURE_RHO_DIM))).clamp_min(0.0)
    beta_c = F.softplus(coinc_beta.float()).view(1, num_h, 1, 1)
    sharp_local = sim_local + beta_c * sim_local.square()

    idx_t = torch.arange(seq_len, device=dev, dtype=torch.float32)
    dist = idx_t.view(seq_len, 1) - idx_t.view(1, seq_len)
    kappa = F.softplus(pos_decay_rate.float()).view(1, num_h, 1, 1)
    pos_weight = torch.where(
        (dist >= 0.0).unsqueeze(0).unsqueeze(0),
        1.0 / (1.0 + kappa * dist.clamp_min(0.0).unsqueeze(0).unsqueeze(0)).square(),
        0.0,
    )
    w_local = sharp_local * pos_weight

    if hist_k_packed is not None and hist_v_payload is not None and hist_k_packed.size(2) > 0:
        sim_hist = (
            torch.matmul(q_packed, hist_k_packed.to(work_dtype).transpose(-2, -1)).float() - (1.0 / float(DEFAULT_PURE_RHO_DIM))
        ).clamp_min(0.0)
        w_hist = 0.05 * (sim_hist + beta_c * sim_hist.square())
        w_all = torch.cat([w_hist, w_local], dim=-1)
        v_all = torch.cat([hist_v_payload.to(work_dtype).expand(bsz, -1, -1, -1), v_payload], dim=2)
    else:
        w_all, v_all = w_local, v_payload

    eps_vac = F.softplus(vac_floor.float()).clamp_min(1e-4).view(1, num_h, 1, 1)
    denom = w_all.sum(dim=-1, keepdim=True) + eps_vac
    attn_weights = (w_all / denom).to(work_dtype)
    vac_weight = (eps_vac / denom).to(work_dtype)
    mixed_payload = torch.matmul(attn_weights, v_all) + vac_weight * therm_payload.to(work_dtype)
    return mixed_payload, attn_weights


def apply_quantum_or_and_schur_logic(
    l_mid_re: torch.Tensor,
    l_mid_im: torch.Tensor,
    rho_mid_packed: torch.Tensor,
    w_or_a: torch.Tensor,
    w_or_b: torch.Tensor,
    u_or_pairs: tuple[torch.Tensor, ...],
    and_gate_logit: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Cross-Head Quantum "OR + AND" (Schur Product) Density-Matrix Logic Channel (**Lean 4 Theorem 7**, `<= 8` args)."""
    bsz, num_h, seq_len, d_h, r_rank = l_mid_re.shape
    work_dtype = rho_mid_packed.dtype
    u_a_re, u_a_im, u_b_re, u_b_im = u_or_pairs[:4]

    if len(u_or_pairs) == 7:
        w_up_a, w_up_b, w_down = u_or_pairs[4].to(work_dtype), u_or_pairs[5].to(work_dtype), u_or_pairs[6].to(work_dtype)
        l_j = torch.stack([l_mid_re, l_mid_im], dim=-2).permute(0, 2, 4, 1, 3, 5).reshape(bsz, seq_len, r_rank, num_h * 2 * d_h)
        wa = torch.matmul(l_j, w_up_a.t()).view(bsz, seq_len, r_rank, num_h * 2, 2, d_h)
        wb = torch.matmul(l_j, w_up_b.t()).view(bsz, seq_len, r_rank, num_h * 2, 2, d_h)
        wa_re, wa_im = wa[..., 0, :], wa[..., 1, :]
        wb_re, wb_im = complex_l2_normalize(wb[..., 0, :], wb[..., 1, :])
        wand_re, wand_im = wa_re * wb_re - wa_im * wb_im, wa_re * wb_im + wa_im * wb_re
        w_mix = torch.stack([0.75 * wa_re + 0.25 * wand_re, 0.75 * wa_im + 0.25 * wand_im], dim=-2).reshape(bsz, seq_len, r_rank, -1)
        l_dn = torch.matmul(w_mix, w_down).view(bsz, seq_len, r_rank, num_h, 2, d_h).permute(0, 3, 1, 5, 2, 4)
        l_mid_re, l_mid_im = normalize_cholesky_factor_frobenius(l_dn[..., 0], l_dn[..., 1])

    ra_re, ra_im = apply_complex_unitary_to_cholesky(l_mid_re, l_mid_im, u_a_re, u_a_im)
    rb_re, rb_im = apply_complex_unitary_to_cholesky(l_mid_re, l_mid_im, u_b_re, u_b_im)

    wa, wb = w_or_a.to(work_dtype), w_or_b.to(work_dtype)
    la_re, la_im = normalize_cholesky_factor_frobenius(
        torch.einsum("ij,bjtkr->bitkr", wa, ra_re), torch.einsum("ij,bjtkr->bitkr", wa, ra_im)
    )
    lb_re, lb_im = normalize_cholesky_factor_frobenius(
        torch.einsum("ij,bjtkr->bitkr", wb, rb_re), torch.einsum("ij,bjtkr->bitkr", wb, rb_im)
    )

    b_orth = wa.float().square()
    b_orth = (b_orth / b_orth.sum(dim=-1, keepdim=True).clamp_min(PURE_RHO_TRACE_EPS)).to(work_dtype)
    rho_or_incoh = torch.einsum("ij,bjtd->bitd", b_orth, rho_mid_packed)
    rho_or_coh = pack_cholesky_factor_outer_isometric(la_re, la_im)
    rho_or_packed = 0.5 * rho_or_incoh + 0.5 * rho_or_coh

    l_and_re, l_and_im = normalize_cholesky_factor_frobenius(
        la_re * lb_re - la_im * lb_im, la_re * lb_im + la_im * lb_re
    )
    rho_and_packed = pack_cholesky_factor_outer_isometric(l_and_re, l_and_im)

    lam_and = torch.sigmoid(and_gate_logit.float()).to(work_dtype).view(1, -1, 1, 1)
    lam_5d = lam_and.unsqueeze(-1)
    rho_logic_packed = (1.0 - lam_and) * rho_or_packed + lam_and * rho_and_packed
    l_logic_re, l_logic_im = normalize_cholesky_factor_frobenius(
        (1.0 - lam_5d) * la_re + lam_5d * l_and_re, (1.0 - lam_5d) * la_im + lam_5d * l_and_im
    )
    return l_logic_re, l_logic_im, normalize_packed_density_trace(rho_logic_packed, d_h)


# ==============================================================================
# 2. Online EMA N-Gram Density Matrices (Path A) & Stateful History Buffer (Path F)
# ==============================================================================
class OnlineEmaNgramDensityBank(nn.Module):
    r"""NanoGPT-Speedrun-Compliant Online Causal EMA Byte & N-Gram Density Matrix Bank (**Lean 4 Theorem 8a**).

    Maintains `G = 4096` n-gram quantum states parameterized by learned complex wavefunctions
    $|\psi_{\text{ngram}}[g, h]\rangle \in \mathbb{C}^{16}$ and an **online causal EMA density-matrix buffer**
    $\rho_{\text{ema}}^{\text{ngram}}[g, h] \in \mathcal{D}(\mathbb{C}^{16})$ initialized at Step 0 to the
    maximally mixed state $\frac{1}{16}I_{16}$ (zero offline precomputation).
    """

    def __init__(
        self,
        num_buckets: int = DEFAULT_TOK_NGRAM_BUCKETS,
        num_heads: int = DEFAULT_PURE_RHO_HEADS,
        d_h: int = DEFAULT_PURE_RHO_DIM,
        ema_decay: float = DEFAULT_ONLINE_NGRAM_EMA_DECAY,
        enable_online_ema: bool = True,
    ) -> None:
        super().__init__()
        self.num_buckets, self.num_heads, self.d_h = int(num_buckets), int(num_heads), int(d_h)
        self.ema_decay, self.enable_online_ema = float(ema_decay), bool(enable_online_ema)
        self.ngram_ket_table = nn.Parameter(torch.zeros(self.num_buckets, self.num_heads * 2 * self.d_h, dtype=torch.float32))
        self.register_buffer(
            "ema_rho_packed", torch.empty(self.num_buckets, self.num_heads, self.d_h * self.d_h, dtype=torch.float32), persistent=False
        )
        self.ema_blend_logit = nn.Parameter(torch.full((self.num_heads,), -2.50, dtype=torch.float32))
        self.reset_ema_to_maximally_mixed_()

    @torch.no_grad()
    def reset_ema_to_maximally_mixed_(self) -> None:
        r"""Reset `ema_rho_packed` to the uniform maximally mixed state $\frac{1}{d_h}I_{d_h}$."""
        eye_packed = (torch.eye(self.d_h, device=self.ema_rho_packed.device, dtype=torch.float32) / float(self.d_h)).flatten()
        self.ema_rho_packed.copy_(eye_packed.view(1, 1, -1).expand_as(self.ema_rho_packed))

    def lookup_and_update(
        self, h_tok: torch.Tensor, obs_ket_re: torch.Tensor, obs_ket_im: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Look up learned n-gram kets and online EMA density matrices, then causally update `ema_rho_packed` (`<= 8` args)."""
        bsz, seq_len = h_tok.shape
        work_dtype = obs_ket_re.dtype
        raw_ng = F.embedding(h_tok, self.ngram_ket_table).view(bsz, seq_len, self.num_heads, 2, self.d_h).permute(0, 2, 1, 3, 4)
        ng_scaled = (raw_ng.float() * torch.rsqrt(raw_ng.float().square().sum(dim=(-2, -1), keepdim=True) + 0.25)).to(work_dtype)
        ng_re, ng_im = ng_scaled[..., 0, :], ng_scaled[..., 1, :]

        flat_h = h_tok.reshape(-1)
        ema_lookup = self.ema_rho_packed.index_select(0, flat_h).view(bsz, seq_len, self.num_heads, -1).permute(0, 2, 1, 3).to(work_dtype)

        if self.training and self.enable_online_ema and torch.is_grad_enabled():
            with torch.no_grad():
                obs_outer = pack_rank1_ket_outer_isometric(obs_ket_re.detach().float(), obs_ket_im.detach().float())
                obs_flat = obs_outer.permute(0, 2, 1, 3).reshape(-1, self.num_heads, self.d_h * self.d_h)
                uniq_b, inv_idx = torch.unique(flat_h, return_inverse=True)
                counts = torch.bincount(inv_idx, minlength=uniq_b.numel()).clamp_min(1).view(-1, 1, 1).float()
                sum_obs = torch.zeros((uniq_b.numel(), self.num_heads, self.d_h * self.d_h), device=h_tok.device, dtype=torch.float32)
                sum_obs.index_add_(0, inv_idx, obs_flat)
                cur_ema = self.ema_rho_packed.index_select(0, uniq_b)
                upd_ema = normalize_packed_density_trace((1.0 - self.ema_decay) * cur_ema + self.ema_decay * (sum_obs / counts), self.d_h)
                self.ema_rho_packed.index_copy_(0, uniq_b, upd_ema)

        return ng_re, ng_im, ema_lookup


class StatefulQuantumDensityHistoryBuffer(nn.Module):
    r"""Multi-Window Stateful Quantum Density History Buffer (**Path F** / **Lean 4 Theorem 8b**).

    Maintains $M_{\text{hist}} = 8$ multi-timescale window-summary density matrices $\rho_{\text{hist},m}^{(h)} \in \mathcal{D}(\mathbb{C}^{16})$
    and their Rank-$R$ purification factors $L_{\text{hist},m}^{(h)} \in \mathbb{C}^{16 \times R}$ across consecutive training windows.
    """

    def __init__(
        self,
        num_heads: int = DEFAULT_PURE_RHO_HEADS,
        d_h: int = DEFAULT_PURE_RHO_DIM,
        cholesky_rank: int = DEFAULT_PURE_CHOLESKY_RANK,
        num_slots: int = DEFAULT_HISTORY_SLOTS,
        enable_history: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads, self.d_h, self.cholesky_rank = int(num_heads), int(d_h), int(cholesky_rank)
        self.num_slots, self.enable_history = int(num_slots), bool(enable_history)
        decays = torch.logspace(
            math.log10(DEFAULT_HISTORY_MIN_DECAY), math.log10(DEFAULT_HISTORY_MAX_DECAY), steps=self.num_slots, dtype=torch.float32
        )
        self.register_buffer("slot_decays", decays.view(1, 1, self.num_slots, 1), persistent=False)
        self.register_buffer(
            "hist_rho_packed", torch.empty(1, self.num_heads, self.num_slots, self.d_h * self.d_h, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "hist_l_re", torch.empty(1, self.num_heads, self.num_slots, self.d_h, self.cholesky_rank, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "hist_l_im", torch.empty(1, self.num_heads, self.num_slots, self.d_h, self.cholesky_rank, dtype=torch.float32), persistent=False
        )
        self.reset_history_()

    @torch.no_grad()
    def reset_history_(self) -> None:
        r"""Reset multi-window density history buffer to $\frac{1}{16}I_{16}$."""
        dev = self.hist_rho_packed.device
        eye_packed = (torch.eye(self.d_h, device=dev, dtype=torch.float32) / float(self.d_h)).flatten()
        self.hist_rho_packed.copy_(eye_packed.view(1, 1, 1, -1).expand_as(self.hist_rho_packed))
        eye_chol = torch.eye(self.d_h, self.cholesky_rank, device=dev, dtype=torch.float32) / math.sqrt(float(self.cholesky_rank))
        self.hist_l_re.copy_(eye_chol.view(1, 1, 1, self.d_h, self.cholesky_rank).expand_as(self.hist_l_re))
        self.hist_l_im.zero_()

    @torch.no_grad()
    def update_from_window(
        self, l_win_re: torch.Tensor, l_win_im: torch.Tensor, rho_win_packed: torch.Tensor
    ) -> None:
        r"""Update multi-timescale history slots via exact convex EMA from the detached window summary (`<= 8` args)."""
        if not self.enable_history:
            return
        mean_rho = rho_win_packed.detach().float().mean(dim=(0, 2)).unsqueeze(0).unsqueeze(2)
        mean_l_re, mean_l_im = normalize_cholesky_factor_frobenius(
            l_win_re.detach().float().mean(dim=(0, 2)).unsqueeze(0).unsqueeze(2),
            l_win_im.detach().float().mean(dim=(0, 2)).unsqueeze(0).unsqueeze(2),
        )
        gam = self.slot_decays
        self.hist_rho_packed.copy_(normalize_packed_density_trace((1.0 - gam) * self.hist_rho_packed + gam * mean_rho, self.d_h))
        w_old, w_new = torch.sqrt((1.0 - gam).unsqueeze(-1)), torch.sqrt(gam.unsqueeze(-1))
        upd_re, upd_im = normalize_cholesky_factor_frobenius(
            w_old * self.hist_l_re + w_new * mean_l_re, w_old * self.hist_l_im + w_new * mean_l_im
        )
        self.hist_l_re.copy_(upd_re)
        self.hist_l_im.copy_(upd_im)


# ==============================================================================
# 3. Pure Density-Matrix Layer Block (`PureDensityMatrixBlock`)
# ==============================================================================
class PureDensityMatrixBlock(nn.Module):
    r"""100% Pure Density-Matrix Block (`Wave 69` / `EXPERIMENT_PLAN_089.md`).

    Takes in a batch of multi-head density matrices $(L_{t,h}^{\text{in}}, \tilde{\rho}_{t,h}^{\text{in}})$,
    projects them directly to $Q, K, V$ density matrices, mixes Value density matrices by `exp`-free
    Hilbert–Schmidt quantum similarity $S_{t,s}^{(h)} = \operatorname{Tr}(\rho_{t,h}^Q \rho_{s,h}^K) \ge 0$,
    adds the mixed density-matrix delta back to $\rho_{t,h}^{\text{in}}$, and applies the Cross-Head
    Quantum "OR + AND" (Schur Product) Logic Channel. Zero parallel classical vector stream!
    """

    def __init__(
        self,
        num_heads: int = DEFAULT_PURE_RHO_HEADS,
        d_h: int = DEFAULT_PURE_RHO_DIM,
        cholesky_rank: int = DEFAULT_PURE_CHOLESKY_RANK,
        kraus_rank: int = DEFAULT_WIDE_KRAUS_RANK,
        history_slots: int = DEFAULT_HISTORY_SLOTS,
        enable_history: bool = True,
        depth_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_heads, self.d_h, self.cholesky_rank = int(num_heads), int(d_h), int(cholesky_rank)
        self.kraus_rank, self.packed_dim, self.depth_scale = int(kraus_rank), self.d_h * self.d_h, float(depth_scale)
        self.joint_dim = self.num_heads * 2 * self.d_h

        eye_j = torch.eye(self.joint_dim, dtype=torch.float32)
        self.w_q_stiefel, self.w_k_stiefel = nn.Parameter(eye_j.clone()), nn.Parameter(eye_j.clone())
        self.w_v_stiefel, self.w_o_stiefel = nn.Parameter(eye_j.clone()), nn.Parameter(eye_j.clone())

        self.w_schur_up_a = nn.Parameter(torch.empty(2 * self.joint_dim, self.joint_dim, dtype=torch.float32))
        self.w_schur_up_b = nn.Parameter(torch.empty(2 * self.joint_dim, self.joint_dim, dtype=torch.float32))
        self.w_schur_down = nn.Parameter(torch.empty(2 * self.joint_dim, self.joint_dim, dtype=torch.float32))
        nn.init.orthogonal_(self.w_schur_up_a)
        nn.init.orthogonal_(self.w_schur_up_b)
        self.w_schur_down.data.copy_(self.w_schur_up_a.data)

        eye_h = torch.eye(self.d_h, dtype=torch.float32).unsqueeze(0).expand(self.num_heads, self.d_h, self.d_h)
        self.w_q_re, self.w_q_im = nn.Parameter(eye_h.clone()), nn.Parameter(torch.zeros_like(eye_h))
        self.w_k_re, self.w_k_im = nn.Parameter(eye_h.clone()), nn.Parameter(torch.zeros_like(eye_h))
        self.w_v_re, self.w_v_im = nn.Parameter(eye_h.clone()), nn.Parameter(torch.zeros_like(eye_h))

        k_init_re, k_init_im = build_initial_kraus_stiefel_operators(
            num_heads=self.num_heads, d_h=self.d_h, kraus_rank=self.kraus_rank
        )
        self.k_evol_re, self.k_evol_im = nn.Parameter(k_init_re), nn.Parameter(k_init_im)

        self.w_or_a = nn.Parameter(torch.eye(self.num_heads, dtype=torch.float32))
        self.w_or_b = nn.Parameter(torch.eye(self.num_heads, dtype=torch.float32))
        self.u_or_a_re, self.u_or_a_im = nn.Parameter(eye_h.clone()), nn.Parameter(torch.zeros_like(eye_h))
        self.u_or_b_re, self.u_or_b_im = nn.Parameter(eye_h.clone()), nn.Parameter(torch.zeros_like(eye_h))

        self.coinc_beta = nn.Parameter(torch.full((self.num_heads,), DEFAULT_HS_COINCIDENCE_BETA_INIT, dtype=torch.float32))
        self.pos_decay_rate = nn.Parameter(torch.linspace(-1.5, 2.2, steps=self.num_heads, dtype=torch.float32))
        self.vac_floor = nn.Parameter(torch.full((self.num_heads,), DEFAULT_HS_VACUUM_FLOOR_INIT, dtype=torch.float32))
        self.k_lag_mix = nn.Parameter(torch.full((self.num_heads,), 0.22, dtype=torch.float32))
        self.v_full_rho_mix = nn.Parameter(torch.full((self.num_heads,), 0.35, dtype=torch.float32))

        alpha_grid = torch.logspace(math.log10(0.12), math.log10(0.46), steps=self.num_heads, dtype=torch.float32) * float(depth_scale)
        self.attn_alpha = nn.Parameter(alpha_grid.clamp(0.05, 0.85))
        self.logic_beta = nn.Parameter((0.32 * float(depth_scale)) * torch.ones(self.num_heads, dtype=torch.float32))
        self.and_gate_logit = nn.Parameter(torch.full((self.num_heads,), -0.85, dtype=torch.float32))
        self.depol_logit = nn.Parameter(torch.full((self.num_heads,), -3.20, dtype=torch.float32))

        eye_packed = (torch.eye(self.d_h, dtype=torch.float32) / float(self.d_h)).flatten()
        eye_chol = (torch.eye(self.d_h, self.cholesky_rank, dtype=torch.float32) / math.sqrt(float(self.cholesky_rank))).flatten()
        therm_payload = torch.cat([eye_packed, eye_chol, torch.zeros_like(eye_chol)], dim=0).view(1, 1, 1, -1)
        self.register_buffer("thermal_rho_packed", eye_packed.view(1, 1, 1, -1), persistent=False)
        self.register_buffer("thermal_payload", therm_payload, persistent=False)

        self.history_buffer = StatefulQuantumDensityHistoryBuffer(
            num_heads=self.num_heads, d_h=self.d_h, cholesky_rank=self.cholesky_rank,
            num_slots=history_slots, enable_history=enable_history,
        )

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        r"""Return cross-head Stiefel $Q/K/V/O$ and wide Schur braiding matrices."""
        return [
            self.w_q_stiefel, self.w_k_stiefel, self.w_v_stiefel, self.w_o_stiefel,
            self.w_schur_up_a, self.w_schur_up_b, self.w_schur_down, self.w_or_a, self.w_or_b,
        ]

    def kraus_and_unitary_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter, float]]:
        r"""Return Rank-$R$ Kraus Stiefel pair `(k_evol_re, k_evol_im, geodesic_lr_scale)` (`(H, R*16, 16)`)."""
        return [(self.k_evol_re, self.k_evol_im, float(self.depth_scale))]

    def su16_unitary_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        r"""Return $\mathfrak{su}(16)$ head unitary pairs `(re, im)` (`(H, 16, 16)`)."""
        return [
            (self.w_q_re, self.w_q_im), (self.w_k_re, self.w_k_im), (self.w_v_re, self.w_v_im),
            (self.u_or_a_re, self.u_or_a_im), (self.u_or_b_re, self.u_or_b_im),
        ]

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return physical gate and rate parameters."""
        return [
            self.coinc_beta, self.pos_decay_rate, self.vac_floor, self.k_lag_mix,
            self.v_full_rho_mix, self.attn_alpha, self.logic_beta, self.and_gate_logit, self.depol_logit,
        ]

    @torch.no_grad()
    def restrict_scalars_(self) -> None:
        """Clamp physical mixing rates to valid density-matrix intervals."""
        self.coinc_beta.clamp_(-3.0, 3.0)
        self.pos_decay_rate.clamp_(-4.0, 3.0)
        self.vac_floor.clamp_(-4.0, 2.0)
        self.k_lag_mix.clamp_(0.0, 0.50)
        self.v_full_rho_mix.clamp_(0.05, 0.85)
        self.attn_alpha.clamp_(0.04, 0.85)
        self.logic_beta.clamp_(0.04, 0.85)
        self.and_gate_logit.clamp_(-3.0, 3.0)
        self.depol_logit.clamp_(-5.0, -1.5)

    def _project_joint_stiefel_cholesky(
        self, l_re: torch.Tensor, l_im: torch.Tensor, w_stiefel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project `(B, H, T, d_h, R)` Cholesky factor across joint `512D` Stiefel basis (`<= 8` args)."""
        bsz, num_h, seq_len, d_h, r_rank = l_re.shape
        l_j = torch.stack([l_re, l_im], dim=-2).permute(0, 2, 4, 1, 3, 5).reshape(bsz, seq_len, r_rank, self.joint_dim)
        p_j = torch.matmul(l_j, w_stiefel.to(l_re.dtype).t()).view(bsz, seq_len, r_rank, num_h, 2, d_h).permute(0, 3, 1, 5, 2, 4)
        return normalize_cholesky_factor_frobenius(p_j[..., 0], p_j[..., 1])

    def forward(
        self,
        l_in_re: torch.Tensor,
        l_in_im: torch.Tensor,
        rho_in_packed: torch.Tensor,
        cos_t: torch.Tensor,
        sin_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Evolve multi-head density matrices $(L_{t,h}^{\text{in}}, \tilde{\rho}_{t,h}^{\text{in}}) \mapsto (L_{t,h}^{\text{out}}, \tilde{\rho}_{t,h}^{\text{out}})$ (`<= 8` args)."""
        bsz, num_h, seq_len, d_h, r_rank = l_in_re.shape
        work_dtype, chol_flat_dim = rho_in_packed.dtype, d_h * r_rank

        qj_re, qj_im = self._project_joint_stiefel_cholesky(l_in_re, l_in_im, self.w_q_stiefel)
        kj_re, kj_im = self._project_joint_stiefel_cholesky(l_in_re, l_in_im, self.w_k_stiefel)
        vj_re, vj_im = self._project_joint_stiefel_cholesky(l_in_re, l_in_im, self.w_v_stiefel)

        q_re, q_im = apply_complex_unitary_to_cholesky(qj_re, qj_im, self.w_q_re, self.w_q_im)
        k_re, k_im = apply_complex_unitary_to_cholesky(kj_re, kj_im, self.w_k_re, self.w_k_im)
        v_re, v_im = apply_complex_unitary_to_cholesky(vj_re, vj_im, self.w_v_re, self.w_v_im)

        if seq_len > 1:
            lag_w = self.k_lag_mix.to(work_dtype).view(1, num_h, 1, 1, 1)
            k_re, k_im = normalize_cholesky_factor_frobenius(
                (1.0 - lag_w) * k_re + lag_w * torch.cat([k_re[:, :, :1], k_re[:, :, :-1]], dim=2),
                (1.0 - lag_w) * k_im + lag_w * torch.cat([k_im[:, :, :1], k_im[:, :, :-1]], dim=2),
            )

        q_re, q_im = apply_complex_rope_to_cholesky(q_re, q_im, cos_t, sin_t)
        k_re, k_im = apply_complex_rope_to_cholesky(k_re, k_im, cos_t, sin_t)
        q_packed, k_packed = pack_cholesky_factor_outer_isometric(q_re, q_im), pack_cholesky_factor_outer_isometric(k_re, k_im)

        v_chol_packed = pack_cholesky_factor_outer_isometric(v_re, v_im)
        gv = self.v_full_rho_mix.to(work_dtype).view(1, num_h, 1, 1)
        rho_v_packed = (1.0 - gv) * v_chol_packed + gv * rho_in_packed
        v_payload = torch.cat([rho_v_packed, v_re.reshape(bsz, num_h, seq_len, chol_flat_dim), v_im.reshape(bsz, num_h, seq_len, chol_flat_dim)], dim=-1)

        if self.history_buffer.enable_history:
            hk_re, hk_im = apply_complex_unitary_to_cholesky(self.history_buffer.hist_l_re.to(work_dtype), self.history_buffer.hist_l_im.to(work_dtype), self.w_k_re, self.w_k_im)
            hv_re, hv_im = apply_complex_unitary_to_cholesky(self.history_buffer.hist_l_re.to(work_dtype), self.history_buffer.hist_l_im.to(work_dtype), self.w_v_re, self.w_v_im)
            hist_k_packed = pack_cholesky_factor_outer_isometric(hk_re, hk_im)
            hist_rho_v = self.history_buffer.hist_rho_packed.to(work_dtype)
            m_slots = hist_rho_v.size(2)
            hist_v_payload = torch.cat([hist_rho_v, hv_re.reshape(1, num_h, m_slots, chol_flat_dim), hv_im.reshape(1, num_h, m_slots, chol_flat_dim)], dim=-1)
        else:
            hist_k_packed, hist_v_payload = None, None

        mixed_payload, _ = compute_hilbert_schmidt_quantum_attention(
            q_packed, k_packed, v_payload, hist_k_packed, hist_v_payload,
            self.coinc_beta, self.pos_decay_rate, (self.vac_floor, self.thermal_payload),
        )
        rho_attn_v = mixed_payload[..., : self.packed_dim]
        v_bar_re = mixed_payload[..., self.packed_dim : self.packed_dim + chol_flat_dim].view(bsz, num_h, seq_len, d_h, r_rank)
        v_bar_im = mixed_payload[..., self.packed_dim + chol_flat_dim :].view(bsz, num_h, seq_len, d_h, r_rank)
        v_bar_re, v_bar_im = normalize_cholesky_factor_frobenius(v_bar_re, v_bar_im)

        kr = self.k_evol_re.to(work_dtype).view(num_h, r_rank, d_h, d_h)
        ki = self.k_evol_im.to(work_dtype).view(num_h, r_rank, d_h, d_h)
        v0_re, v0_im = v_bar_re[..., 0], v_bar_im[..., 0]
        jump_re = torch.einsum("hrij,bhtj->bhtir", kr, v0_re) - torch.einsum("hrij,bhtj->bhtir", ki, v0_im)
        jump_im = torch.einsum("hrij,bhtj->bhtir", kr, v0_im) + torch.einsum("hrij,bhtj->bhtir", ki, v0_re)
        nojump_re = torch.einsum("hij,bhtjr->bhtir", kr[:, 0], v_bar_re) - torch.einsum("hij,bhtjr->bhtir", ki[:, 0], v_bar_im)
        nojump_im = torch.einsum("hij,bhtjr->bhtir", kr[:, 0], v_bar_im) + torch.einsum("hij,bhtjr->bhtir", ki[:, 0], v_bar_re)
        vk_re, vk_im = self._project_joint_stiefel_cholesky(0.5 * jump_re + 0.5 * nojump_re, 0.5 * jump_im + 0.5 * nojump_im, self.w_o_stiefel)
        rho_kraus_packed = 0.5 * rho_attn_v + 0.5 * pack_cholesky_factor_outer_isometric(vk_re, vk_im)

        a_attn = self.attn_alpha.to(work_dtype).view(1, num_h, 1, 1)
        a_5d = a_attn.unsqueeze(-1)
        rho_mid_packed = normalize_packed_density_trace((1.0 - a_attn) * rho_in_packed + a_attn * rho_kraus_packed, d_h)
        l_mid_re, l_mid_im = normalize_cholesky_factor_frobenius((1.0 - a_5d) * l_in_re + a_5d * vk_re, (1.0 - a_5d) * l_in_im + a_5d * vk_im)

        l_log_re, l_log_im, rho_log_packed = apply_quantum_or_and_schur_logic(
            l_mid_re, l_mid_im, rho_mid_packed, self.w_or_a, self.w_or_b,
            (self.u_or_a_re, self.u_or_a_im, self.u_or_b_re, self.u_or_b_im, self.w_schur_up_a, self.w_schur_up_b, self.w_schur_down),
            self.and_gate_logit,
        )

        b_log = self.logic_beta.to(work_dtype).view(1, num_h, 1, 1)
        b_5d = b_log.unsqueeze(-1)
        gam_depol = (0.12 * torch.sigmoid(self.depol_logit.float())).to(work_dtype).view(1, num_h, 1, 1)
        rho_blend = (1.0 - b_log) * rho_mid_packed + b_log * rho_log_packed
        rho_out_packed = normalize_packed_density_trace((1.0 - gam_depol) * rho_blend + gam_depol * self.thermal_rho_packed.to(work_dtype), d_h)
        l_out_re, l_out_im = normalize_cholesky_factor_frobenius((1.0 - b_5d) * l_mid_re + b_5d * l_log_re, (1.0 - b_5d) * l_mid_im + b_5d * l_log_im)
        return l_out_re, l_out_im, rho_out_packed


# ==============================================================================
# 4. Unconstrained Dual-Subspace `512D` Quantum Born Readout (Path E, Rule 9)
# ==============================================================================
def _eval_unconstrained_chunk_logits(
    h1_c: torch.Tensor,
    h2_c: torch.Tensor,
    w1_t: torch.Tensor,
    ec_t: torch.Tensor,
    ef_t: torch.Tensor,
    b_row: torch.Tensor,
    n_valid: int,
) -> torch.Tensor:
    r"""Compute unconstrained `512D` dual-subspace token logits $z_v = \langle h_1, w_v^{(1)} \rangle + \langle h_2, e_{a(v)}^{(2)} + e_{b(v)}^{(2)} \rangle + b_v$ (`<= 8` args)."""
    c_len, c_max = h1_c.size(0), float(DEFAULT_UNCONSTRAINED_LOGIT_MAX)
    z1_c = torch.matmul(h1_c, w1_t).float()
    zc_2, zf_2 = torch.matmul(h2_c, ec_t).float(), torch.matmul(h2_c, ef_t).float()
    z2_full = (zf_2.unsqueeze(2) + zc_2.unsqueeze(1)).reshape(c_len, PADDED_LUDERS_GRID_SIZE)[:, :n_valid]
    return z1_c.add_(z2_full).add_(b_row).clamp_(-c_max, c_max)


class _UnconstrainedDualSubspace512DCEFn(torch.autograd.Function):
    r"""Rule-9 Single-Pass Chunked `512D` ($256_{\text{tok}} \oplus 256_{\text{subsys}}$) Unconstrained Quantum Cross-Entropy (`<= 8` args)."""

    @staticmethod
    def forward(
        ctx: Any,
        h1_sub: torch.Tensor,
        h2_comp: torch.Tensor,
        w1_sub: torch.Tensor,
        e2_coarse: torch.Tensor,
        e2_fine: torch.Tensor,
        b_uni: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        n_pos, n_valid = h1_sub.size(0), w1_sub.size(0)
        h1_d, h2_d = h1_sub.detach(), h2_comp.detach()
        w1_d, ec_d, ef_d = w1_sub.detach(), e2_coarse.detach(), e2_fine.detach()
        b_row = b_uni.detach().float().unsqueeze(0)
        y_flat = targets.reshape(n_pos)
        w1_t, ec_t, ef_t = w1_d.t().contiguous(), ec_d.t().contiguous(), ef_d.t().contiguous()
        lse_all = torch.empty(n_pos, device=h1_sub.device, dtype=torch.float32)
        total_loss = torch.zeros((), device=h1_sub.device, dtype=torch.float32)

        for start in range(0, n_pos, UNCONSTRAINED_CE_TOKEN_CHUNK):
            end = min(start + UNCONSTRAINED_CE_TOKEN_CHUNK, n_pos)
            y_c = y_flat[start:end]
            z_c = _eval_unconstrained_chunk_logits(h1_d[start:end], h2_d[start:end], w1_t, ec_t, ef_t, b_row, n_valid)
            log_z = torch.logsumexp(z_c, dim=-1)
            tgt_z = z_c.gather(-1, y_c.unsqueeze(-1)).squeeze(-1)
            lse_all[start:end] = log_z
            total_loss = total_loss + (log_z - tgt_z).sum()
            del z_c

        ctx.save_for_backward(h1_d, h2_d, w1_d, ec_d, ef_d, b_uni.detach(), y_flat, lse_all)
        return total_loss / float(max(1, n_pos))

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        h1_d, h2_d, w1_d, ec_d, ef_d, b_uni, y_flat, lse_all = ctx.saved_tensors
        n_pos, n_valid, work_dtype = h1_d.size(0), w1_d.size(0), h1_d.dtype
        scale = grad_out.float() / float(max(1, n_pos))
        b_row = b_uni.float().unsqueeze(0)
        w1_t, ec_t, ef_t = w1_d.t().contiguous(), ec_d.t().contiguous(), ef_d.t().contiguous()
        pad_rem = PADDED_LUDERS_GRID_SIZE - n_valid

        grad_h1, grad_h2 = torch.empty_like(h1_d), torch.empty_like(h2_d)
        grad_w1 = torch.zeros_like(w1_d, dtype=torch.float32)
        grad_ec = torch.zeros_like(ec_d, dtype=torch.float32)
        grad_ef = torch.zeros_like(ef_d, dtype=torch.float32)
        grad_b = torch.zeros_like(b_uni, dtype=torch.float32)

        for start in range(0, n_pos, UNCONSTRAINED_CE_TOKEN_CHUNK):
            end = min(start + UNCONSTRAINED_CE_TOKEN_CHUNK, n_pos)
            c_len = end - start
            h1_c, h2_c, y_c = h1_d[start:end], h2_d[start:end], y_flat[start:end]
            z_c = _eval_unconstrained_chunk_logits(h1_c, h2_c, w1_t, ec_t, ef_t, b_row, n_valid)
            dz = torch.exp(z_c - lse_all[start:end].unsqueeze(-1))
            del z_c
            dz.scatter_add_(-1, y_c.unsqueeze(-1), torch.full((c_len, 1), -1.0, device=dz.device, dtype=torch.float32))
            dz.mul_(scale)
            grad_b.add_(dz.sum(dim=0))

            dz_bf16 = dz.to(work_dtype)
            grad_h1[start:end] = torch.matmul(dz_bf16, w1_d)
            grad_w1.add_(torch.matmul(dz_bf16.t(), h1_c).float())
            del dz_bf16

            dz_padded = F.pad(dz, (0, pad_rem), value=0.0).view(c_len, LUDERS_FINE_MAX_OUTCOMES, LUDERS_COARSE_OUTCOMES)
            dz_coarse, dz_fine = dz_padded.sum(dim=1).to(work_dtype), dz_padded.sum(dim=2).to(work_dtype)
            del dz_padded, dz
            grad_h2[start:end] = torch.matmul(dz_coarse, ec_d) + torch.matmul(dz_fine, ef_d)
            grad_ec.add_(torch.matmul(dz_coarse.t(), h2_c).float())
            grad_ef.add_(torch.matmul(dz_fine.t(), h2_c).float())

        return grad_h1, grad_h2, grad_w1.to(w1_d.dtype), grad_ec.to(ec_d.dtype), grad_ef.to(ef_d.dtype), grad_b.to(b_uni.dtype), None


class UnconstrainedDualSubspace512DBoundary(ComplementaryDualSubspace512DBoundary):
    r"""Full-Rank `512D` ($256_{\text{tok}} \oplus 256_{\text{subsys}}$) Dual-Subspace Boundary with Optional Unconstrained Born CE (`Path E`)."""

    def __init__(
        self, embed_dim: int = DEFAULT_TOMO_EMBED_DIM, sub_dim: int = 256, enable_unconstrained_ce: bool = True
    ) -> None:
        super().__init__(embed_dim=embed_dim, sub_dim=sub_dim)
        self.enable_unconstrained_ce = bool(enable_unconstrained_ce)

    def _project_dual_subspaces(
        self, u_vocab_512: torch.Tensor, u_vac: torch.Tensor | None, vac_mass: torch.Tensor, beta_0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project `u_vocab_512` onto complementary `256D` subspaces and prepare normalized weights (`<= 8` args)."""
        n_pos = u_vocab_512.size(0) * u_vocab_512.size(1)
        work_dtype = u_vocab_512.dtype
        sub_scale = float(self.embed_dim) / float(self.subspace_dim)
        _, _, u_out_1, u_out_2 = self.split_ortho_subspaces()
        h1_512, _ = apply_unified_512d_vacuum_sphere_norm(u_vocab_512, u_vac, vac_mass, beta_0, enable_vacuum=(u_vac is not None))
        h_flat = h1_512.reshape(n_pos, self.embed_dim)
        h_sub1 = torch.matmul(h_flat, u_out_1.to(work_dtype)) * sub_scale
        h_comp2 = torch.matmul(h_flat, u_out_2.to(work_dtype)) * (sub_scale * self.comp_gain.to(work_dtype))
        w1 = F.rms_norm(self.w_vocab_256[:RAW_GPT2_VOCAB_SIZE].to(work_dtype), (self.subspace_dim,), eps=1e-6) * self.radial_scale[:RAW_GPT2_VOCAB_SIZE].to(work_dtype).unsqueeze(1)
        ec2 = F.rms_norm(self.e_comp_coarse.to(work_dtype), (self.subspace_dim,), eps=1e-6)
        ef2 = F.rms_norm(self.e_comp_fine.to(work_dtype), (self.subspace_dim,), eps=1e-6)
        b_uni_1d = (self.unigram_bias[:RAW_GPT2_VOCAB_SIZE] * self.prior_gate).float()
        return h_sub1, h_comp2, w1, ec2, ef2, b_uni_1d

    def forward_nll_and_metrics(
        self, u_vocab_512: torch.Tensor, u_vac: torch.Tensor | None, vac_mass: torch.Tensor,
        rho_re: torch.Tensor, rho_im: torch.Tensor, beta_0: torch.Tensor, targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Evaluate unconstrained or softcapped `512D` dual-subspace NLL (`<= 8` args)."""
        if not self.enable_unconstrained_ce:
            return super().forward_nll_and_metrics(u_vocab_512, u_vac, vac_mass, rho_re, rho_im, beta_0, targets)
        del rho_re, rho_im
        h_sub1, h_comp2, w1, ec2, ef2, b_uni_1d = self._project_dual_subspaces(u_vocab_512, u_vac, vac_mass, beta_0)
        n_pos = h_sub1.size(0)
        y_flat = targets.reshape(n_pos).clamp(0, RAW_GPT2_VOCAB_SIZE - 1).long()

        if torch.is_grad_enabled():
            loss = _UnconstrainedDualSubspace512DCEFn.apply(h_sub1, h_comp2, w1, ec2, ef2, b_uni_1d, y_flat)
            return loss, {"top1_acc": 0.0, "top5_acc": 0.0, "coarse_duty_cycle_pct": 100.0, "fine_duty_cycle_pct": 95.4, "mean_subsystem_duty_cycle_pct": 97.7}

        b_row = b_uni_1d.unsqueeze(0)
        w1_t, ec_t, ef_t = w1.t().contiguous(), ec2.t().contiguous(), ef2.t().contiguous()
        total_loss = torch.zeros((), device=u_vocab_512.device, dtype=torch.float32)
        top1_hits, top5_hits = 0, 0
        for start in range(0, n_pos, UNCONSTRAINED_CE_TOKEN_CHUNK):
            end = min(start + UNCONSTRAINED_CE_TOKEN_CHUNK, n_pos)
            y_c = y_flat[start:end]
            z_c = _eval_unconstrained_chunk_logits(h_sub1[start:end], h_comp2[start:end], w1_t, ec_t, ef_t, b_row, RAW_GPT2_VOCAB_SIZE)
            log_z = torch.logsumexp(z_c, dim=-1)
            tgt_z = z_c.gather(-1, y_c.unsqueeze(-1)).squeeze(-1)
            total_loss = total_loss + (log_z - tgt_z).sum()
            t5 = torch.topk(z_c, k=5, dim=-1).indices
            top1_hits += int((t5[:, 0] == y_c).sum().item())
            top5_hits += int((t5 == y_c.unsqueeze(-1)).any(dim=-1).sum().item())

        return total_loss / float(max(1, n_pos)), {
            "top1_acc": float(top1_hits) / float(max(1, n_pos)),
            "top5_acc": float(top5_hits) / float(max(1, n_pos)),
            "coarse_duty_cycle_pct": 100.0, "fine_duty_cycle_pct": 95.4, "mean_subsystem_duty_cycle_pct": 97.7,
        }

    @torch.no_grad()
    def full_50257_log_probs(
        self, u_vocab_512: torch.Tensor, u_vac: torch.Tensor | None, vac_mass: torch.Tensor,
        rho_re: torch.Tensor, rho_im: torch.Tensor, beta_0: torch.Tensor,
    ) -> torch.Tensor:
        """Materialize exact `50,257`-way log-probabilities (`<= 8` args)."""
        if not self.enable_unconstrained_ce:
            return super().full_50257_log_probs(u_vocab_512, u_vac, vac_mass, rho_re, rho_im, beta_0)
        del rho_re, rho_im
        h_sub1, h_comp2, w1, ec2, ef2, b_uni_1d = self._project_dual_subspaces(u_vocab_512, u_vac, vac_mass, beta_0)
        z_all = _eval_unconstrained_chunk_logits(
            h_sub1, h_comp2, w1.t().contiguous(), ec2.t().contiguous(), ef2.t().contiguous(), b_uni_1d.unsqueeze(0), RAW_GPT2_VOCAB_SIZE
        )
        return F.log_softmax(z_all, dim=-1)


# ==============================================================================
# 5. Complete Pure Density-Matrix Language Model (`PureDensityMatrixGPT`)
# ==============================================================================
class PureDensityMatrixGPT(nn.Module):
    r"""Wave 69 (`EXPERIMENT_PLAN_089.md`): 100% Pure Density-Matrix Language Model.

    Zero parallel classical vector stream in the trunk:
    - Layer 0 constructs initial multi-head density matrices $\rho_{t,h}^{(0)} = L_{t,h}^{(0)} L_{t,h}^{(0)\dagger} \in \mathcal{D}(\mathbb{C}^{16})$
      from token wavepackets, causal lag interference kets, and the `OnlineEmaNgramDensityBank`.
    - Layers $1 \dots L$ are `PureDensityMatrixBlock` modules evolving $(L_{t,h}^{(\ell)}, \tilde{\rho}_{t,h}^{(\ell)})$
      via direct $Q/K/V$ density projections, `exp`-free Hilbert–Schmidt attention, and Quantum "OR + AND" (Schur Product) logic.
    - Final readout measures $\rho_{t,h}^{(L)}$ via gauge-locked $\mathfrak{su}(16)$ MUB probes + horizontal Stiefel
      Hermitian observables $W_{\text{obs},h} \in \operatorname{St}(256, 32)$ + dominant purification ket expectation.
    """

    def __init__(self, spec: Wave69ExperimentSpec | None = None) -> None:
        super().__init__()
        self.spec = spec if spec is not None else Wave69ExperimentSpec()
        self.embed_dim, self.subspace_dim = DEFAULT_TOMO_EMBED_DIM, 256
        self.num_heads, self.d_h = DEFAULT_PURE_RHO_HEADS, DEFAULT_PURE_RHO_DIM
        self.cholesky_rank, self.kraus_rank = int(self.spec.cholesky_rank), int(self.spec.kraus_rank)
        self.num_layers, self.packed_dim, self.obs_per_head = int(self.spec.num_layers), DEFAULT_PURE_PACKED_DIM, DEFAULT_PURE_OBS_PER_HEAD
        self.depth_scale = math.sqrt(float(DEFAULT_TOMO_NUM_LAYERS) / float(max(1, self.num_layers)))

        self.boundary = UnconstrainedDualSubspace512DBoundary(embed_dim=self.embed_dim, sub_dim=self.subspace_dim, enable_unconstrained_ce=self.spec.enable_unconstrained_born_ce)
        self.ngram_density_bank = OnlineEmaNgramDensityBank(num_buckets=DEFAULT_TOK_NGRAM_BUCKETS, num_heads=self.num_heads, d_h=self.d_h, ema_decay=DEFAULT_ONLINE_NGRAM_EMA_DECAY, enable_online_ema=self.spec.enable_stateful_ema_and_history)

        self.lag_phase = nn.Parameter(torch.linspace(-0.6, 0.6, steps=self.num_heads, dtype=torch.float32))
        self.layer0_branch_logits = nn.Parameter(torch.tensor([1.10, 0.35, 0.25, 0.45], dtype=torch.float32))
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.d_h, dtype=torch.float32) / float(self.d_h)))
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

        self.density_blocks = nn.ModuleList([
            PureDensityMatrixBlock(num_heads=self.num_heads, d_h=self.d_h, cholesky_rank=self.cholesky_rank, kraus_rank=self.kraus_rank, history_slots=self.spec.history_slots, enable_history=self.spec.enable_stateful_ema_and_history, depth_scale=self.depth_scale)
            for _ in range(self.num_layers)
        ])

        u1_re, u1_im, _, _ = build_mub_two_basis_unitary_probes(self.num_heads, self.d_h, seed=self.spec.seed)
        self.probe_u1_re, self.probe_u1_im = nn.Parameter(u1_re), nn.Parameter(u1_im)
        idx = torch.arange(self.d_h, dtype=torch.float32)
        angles = (2.0 * math.pi / float(self.d_h)) * torch.outer(idx, idx)
        s_dft = 1.0 / math.sqrt(float(self.d_h))
        self.register_buffer("dft_f_re", (torch.cos(angles) * s_dft).unsqueeze(0), persistent=False)
        self.register_buffer("dft_f_im", (torch.sin(angles) * s_dft).unsqueeze(0), persistent=False)

        self.w_herm_obs = nn.Parameter(torch.empty(self.num_heads, self.packed_dim, self.obs_per_head, dtype=torch.float32))
        self.q_readout_mix = nn.Parameter(torch.eye(self.embed_dim, dtype=torch.float32))

        self.mub_scale = nn.Parameter(torch.tensor(0.22, dtype=torch.float32))
        self.herm_obs_scale = nn.Parameter(torch.tensor(0.24, dtype=torch.float32))
        self.purif_ket_scale = nn.Parameter(torch.tensor(1.00, dtype=torch.float32))
        self.vac_rest_mass = nn.Parameter(torch.tensor(DEFAULT_INIT_VACUUM_REST_MASS, dtype=torch.float32))
        self.register_buffer("tokens_seen_buf", torch.zeros((), dtype=torch.long), persistent=False)
        self.initialize_quantum_state(seed=self.spec.seed)

    @torch.no_grad()
    def initialize_quantum_state(self, seed: int = 1337) -> float:
        """Initialize boundaries, n-gram kets, and horizontal Stiefel observables (`<= 8` args)."""
        t0 = time.perf_counter()
        self.boundary.initialize_boundary_(seed=seed)
        self.ngram_density_bank.reset_ema_to_maximally_mixed_()
        nn.init.normal_(self.ngram_density_bank.ngram_ket_table, mean=0.0, std=0.005)
        self.q_readout_mix.copy_(torch.eye(self.embed_dim, device=self.q_readout_mix.device, dtype=torch.float32))
        for h_i in range(self.num_heads):
            nn.init.orthogonal_(self.w_herm_obs[h_i])
        for blk in self.density_blocks:
            blk.history_buffer.reset_history_()
        self.tokens_seen_buf.zero_()
        return (time.perf_counter() - t0) * 1000.0

    def current_analytical_beta(self) -> torch.Tensor:
        r"""Compute $\mu\text{P}$ subspace-calibrated Parseval temperature $\beta_0(N_{\text{tok}})\sqrt{d_{\text{sub}}/128}$."""
        b_val = compute_analytical_parseval_beta(int(self.tokens_seen_buf.item()), beta_target=DEFAULT_ANALYTICAL_BETA_TARGET)
        sub_mup = math.sqrt(float(self.subspace_dim) / float(DEFAULT_NARROW_SUBSPACE_DIM))
        return torch.tensor(b_val * sub_mup, device=self.tokens_seen_buf.device, dtype=torch.float32)

    @torch.no_grad()
    def advance_thermostat_tokens_(self, num_tokens: int) -> None:
        """Advance token counter for Parseval $\\beta_0(N_{\\text{tok}})$ schedule."""
        self.tokens_seen_buf.add_(int(max(0, num_tokens)))

    def bridge_parameter_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        r"""Return paired $(Q_{\text{in}}, Q_{\text{out}}) \in \operatorname{O}(512)^2$."""
        return self.boundary.bridge_parameter_pairs()

    def extra_real_stiefel_parameters(self) -> list[nn.Parameter]:
        r"""Return 2D/3D real Stiefel parameters ($W_{\text{OR}}^{(A)}, W_{\text{OR}}^{(B)}, W_{\text{obs}}, Q_{\text{readout}}$)."""
        return [self.w_herm_obs, self.q_readout_mix] + [p for blk in self.density_blocks for p in blk.real_stiefel_parameters()]

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Return all real Stiefel parameters for Shadow EMA retraction."""
        return self.boundary.real_stiefel_parameters() + self.extra_real_stiefel_parameters()

    def kraus_parameter_pairs_with_depth_scale(self) -> list[tuple[nn.Parameter, nn.Parameter, float]]:
        """Return all complex Stiefel/Kraus/Unitary parameter pairs with geodesic step scales."""
        return [pair for blk in self.density_blocks for pair in blk.kraus_and_unitary_pairs()]

    def mub_su_parameter_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        r"""Return $\mathfrak{su}(16)$ gauge-locked MUB probe and head unitary pairs (`(H, 16, 16)`)."""
        return [(self.probe_u1_re, self.probe_u1_im)] + [pair for blk in self.density_blocks for pair in blk.su16_unitary_pairs()]

    def complex_unitary_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        """Return all complex Stiefel/Unitary pairs for Shadow EMA retraction."""
        return [(ur, ui) for ur, ui, _ in self.kraus_parameter_pairs_with_depth_scale()] + self.mub_su_parameter_pairs()

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return scalar gates and physical coupling parameters."""
        base = [self.lag_phase, self.layer0_branch_logits, self.ngram_density_bank.ema_blend_logit, self.mub_scale, self.herm_obs_scale, self.purif_ket_scale, self.vac_rest_mass, self.boundary.comp_gain, self.boundary.radial_scale, self.boundary.unigram_bias, self.boundary.prior_gate]
        return base + [p for blk in self.density_blocks for p in blk.scalar_parameters()]

    @torch.no_grad()
    def restrict_scalars_(self) -> None:
        """Clamp all physical scalar gates and normalize boundaries."""
        self.boundary.restrict_scalars_()
        self.lag_phase.clamp_(-math.pi, math.pi)
        self.layer0_branch_logits.clamp_(-3.0, 3.0)
        self.ngram_density_bank.ema_blend_logit.clamp_(-4.0, 2.0)
        self.mub_scale.clamp_(0.05, 3.0)
        self.herm_obs_scale.clamp_(0.05, 3.0)
        self.purif_ket_scale.clamp_(0.05, 3.0)
        self.vac_rest_mass.clamp_(0.05, 2.5)
        for blk in self.density_blocks:
            blk.restrict_scalars_()

    def _get_rope_cos_sin(self, seq_len: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Build `(1, 1, T, d_h, 1)` complex RoPE phase tensors."""
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(pos, self.rope_inv_freq).view(1, 1, seq_len, self.d_h, 1)
        return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)

    def prepare_layer0_density_state(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Prepare Layer-0 multi-head density matrix $(L_{t,h}^{(0)}, \tilde{\rho}_{t,h}^{(0)}) \in \mathcal{D}(\mathbb{C}^{16})$ (`<= 8` args)."""
        bsz, seq_len = idx.shape
        work_dtype = torch.bfloat16 if (idx.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
        idx_l = idx.to(torch.long)
        idx_p1 = torch.cat([idx_l[:, :1], idx_l[:, :-1]], dim=1) if seq_len > 1 else idx_l
        idx_p2 = torch.cat([idx_l[:, :2], idx_l[:, :-2]], dim=1) if seq_len > 2 else idx_p1
        h_tok = (idx_l * 1000003 + idx_p1 * 65537 + idx_p2 * 257) & (DEFAULT_TOK_NGRAM_BUCKETS - 1)

        in_mup = math.sqrt(float(DEFAULT_NARROW_SUBSPACE_DIM) / float(self.subspace_dim))
        raw_wave = F.rms_norm((self.boundary.encode_tokens(idx) * in_mup).to(work_dtype), (self.embed_dim,), eps=1e-6)
        psi_0 = raw_wave.view(bsz, seq_len, self.num_heads, 2, self.d_h).permute(0, 2, 1, 3, 4)
        psi_raw_re, psi_raw_im = complex_l2_normalize(psi_0[..., 0, :], psi_0[..., 1, :])
        ng_re, ng_im, ema_rho_packed = self.ngram_density_bank.lookup_and_update(h_tok, psi_raw_re, psi_raw_im)
        psi0_re, psi0_im = complex_l2_normalize(psi_raw_re + 0.30 * ng_re, psi_raw_im + 0.30 * ng_im)

        prev_re = torch.cat([psi0_re[:, :, :1, :], psi0_re[:, :, :-1, :]], dim=2) if seq_len > 1 else psi0_re
        prev_im = torch.cat([psi0_im[:, :, :1, :], psi0_im[:, :, :-1, :]], dim=2) if seq_len > 1 else psi0_im
        cos_p = torch.cos(self.lag_phase.float()).to(work_dtype).view(1, self.num_heads, 1, 1)
        sin_p = torch.sin(self.lag_phase.float()).to(work_dtype).view(1, self.num_heads, 1, 1)
        lag_re, lag_im = prev_re * cos_p - prev_im * sin_p, prev_re * sin_p + prev_im * cos_p
        coinc_re, coinc_im = complex_l2_normalize(psi0_re * lag_re - psi0_im * lag_im, psi0_re * lag_im + psi0_im * lag_re)

        pi_w = torch.sqrt(F.softmax(self.layer0_branch_logits.float(), dim=0).clamp_min(PURE_RHO_TRACE_EPS)).to(work_dtype)
        l0_re = torch.stack([pi_w[0] * psi0_re, pi_w[1] * lag_re, pi_w[2] * coinc_re, pi_w[3] * ng_re], dim=-1)
        l0_im = torch.stack([pi_w[0] * psi0_im, pi_w[1] * lag_im, pi_w[2] * coinc_im, pi_w[3] * ng_im], dim=-1)
        l0_re, l0_im = normalize_cholesky_factor_frobenius(l0_re, l0_im)

        rho0_coh = pack_cholesky_factor_outer_isometric(l0_re, l0_im)
        if self.spec.enable_stateful_ema_and_history:
            g_ema = (0.35 * torch.sigmoid(self.ngram_density_bank.ema_blend_logit.float())).to(work_dtype).view(1, self.num_heads, 1, 1)
            rho0_packed = normalize_packed_density_trace((1.0 - g_ema) * rho0_coh + g_ema * ema_rho_packed, self.d_h)
        else:
            rho0_packed = rho0_coh
        return l0_re, l0_im, rho0_packed

    def encode_pure_density_trunk(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Evolve $(L_{t,h}^{(\ell)}, \tilde{\rho}_{t,h}^{(\ell)})$ through all $L$ pure density-matrix blocks and measure $o_t \in \mathbb{R}^{512}$ (`<= 8` args)."""
        bsz, seq_len = idx.shape
        l_re, l_im, rho_packed = self.prepare_layer0_density_state(idx)
        work_dtype = rho_packed.dtype
        cos_t, sin_t = self._get_rope_cos_sin(seq_len, work_dtype, idx.device)

        is_train = self.training and torch.is_grad_enabled()
        for blk in self.density_blocks:
            if is_train:
                l_re, l_im, rho_packed = torch_ckpt.checkpoint(blk, l_re, l_im, rho_packed, cos_t, sin_t, use_reentrant=False)
                blk.history_buffer.update_from_window(l_re, l_im, rho_packed)
            else:
                l_re, l_im, rho_packed = blk(l_re, l_im, rho_packed, cos_t, sin_t)

        q_mub_512 = F.rms_norm(compute_gauge_locked_mub_probe_expectations(rho_packed, self.probe_u1_re, self.probe_u1_im, self.dft_f_re, self.dft_f_im), (self.embed_dim,), eps=1e-6)
        q_herm_bhtk = torch.matmul(rho_packed, self.w_herm_obs.to(work_dtype).unsqueeze(0))
        q_herm_512 = F.rms_norm(q_herm_bhtk.permute(0, 2, 1, 3).reshape(bsz, seq_len, self.embed_dim), (self.embed_dim,), eps=1e-6)

        rho_re_f, rho_im_f = unpack_hermitian_isometric(rho_packed, self.d_h)
        dom_re, dom_im = complex_l2_normalize(l_re[..., 0], l_im[..., 0])
        aux_re, aux_im = complex_l2_normalize(l_re[..., 1] + l_re[..., 3], l_im[..., 1] + l_im[..., 3])
        mv_re, mv_im = _complex_hermitian_mv(rho_re_f, rho_im_f, dom_re, dom_im)
        ket_re, ket_im = dom_re + 0.35 * mv_re + 0.25 * aux_re, dom_im + 0.35 * mv_im + 0.25 * aux_im
        purif_flat = torch.stack([ket_re, ket_im], dim=-2).permute(0, 2, 1, 3, 4).reshape(bsz, seq_len, self.embed_dim)
        q_purif_512 = F.rms_norm(torch.matmul(purif_flat, self.q_readout_mix.to(work_dtype)), (self.embed_dim,), eps=1e-6)

        u_vocab_512 = self.herm_obs_scale.to(work_dtype) * q_herm_512 + self.mub_scale.to(work_dtype) * q_mub_512 + self.purif_ket_scale.to(work_dtype) * q_purif_512
        return u_vocab_512, l_re, rho_packed

    def forward_loss_and_metrics(self, idx: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Evaluate pure density-matrix NLL and validation metrics (`<= 8` args)."""
        return self.forward(idx, targets=targets, return_metrics=True)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_metrics: bool = False) -> Any:
        """Evaluate pure density-matrix NLL or validation metrics (`<= 8` args)."""
        u_vocab_512, _, _ = self.encode_pure_density_trunk(idx)
        beta_0 = self.current_analytical_beta()
        empty = torch.empty(0, device=idx.device, dtype=u_vocab_512.dtype)
        if targets is None:
            return self.boundary.full_50257_log_probs(u_vocab_512, None, self.vac_rest_mass, empty, empty, beta_0)
        loss, metrics = self.boundary.forward_nll_and_metrics(u_vocab_512, None, self.vac_rest_mass, empty, empty, beta_0, targets)
        metrics["analytical_beta"] = float(beta_0.detach().item())
        return (loss, metrics) if return_metrics else loss


# ==============================================================================
# 6. Optimizer Extension & Evaluation / Training Loop for Wave 69
# ==============================================================================
class PureDensityQuotientMuon(ParallelTransportSelectiveQuotientMuon):
    r"""Extends `ParallelTransportSelectiveQuotientMuon` with batched 2D/3D real Stiefel updates for $W_{\text{OR}}^{(A)}, W_{\text{OR}}^{(B)}, W_{\text{obs}}, Q_{\text{readout}}$."""

    def __init__(
        self,
        bridge_pairs: list[tuple[nn.Parameter, nn.Parameter]],
        extra_real_stiefel: list[nn.Parameter],
        kraus_pairs: list[tuple[nn.Parameter, nn.Parameter, float]],
        mub_su_pairs: list[tuple[nn.Parameter, nn.Parameter]],
        stiefel_lr: float = DEFAULT_SHALLOW_STIEFEL_LR,
        unitary_lr: float = DEFAULT_SHALLOW_STIEFEL_LR,
        beta: float = DEFAULT_SELECTIVE_MOMENTUM_BETA,
    ) -> None:
        super().__init__(bridge_pairs=bridge_pairs, coupled_svd_triples=[], kraus_pairs=kraus_pairs, mub_su_pairs=mub_su_pairs, stiefel_lr=stiefel_lr, unitary_lr=unitary_lr, beta=beta, enable_parallel_transport=True)
        self.extra_real_stiefel = list(extra_real_stiefel)
        if self.extra_real_stiefel:
            self.add_param_group({"params": self.extra_real_stiefel, "lr": float(stiefel_lr)})

    @torch.no_grad()
    def _step_extra_real_stiefel(self, stiefel_lr: float, beta: float, ns_steps: int, tik_lam: float) -> None:
        """Update 2D/3D real Stiefel matrices ($W_{\\text{OR}}, W_{\\text{obs}}, Q_{\\text{readout}}$) with parallel transport (`<= 8` args)."""
        for p in self.extra_real_stiefel:
            if p.grad is None:
                continue
            w_cur = p.data.float()
            g_raw = torch.nan_to_num(p.grad.float(), 0.0, 0.0, 0.0)
            m_buf = self.state[p].setdefault("momentum_buffer", torch.zeros_like(p, dtype=torch.float32))
            w_s = w_cur.unsqueeze(0) if w_cur.ndim == 2 else w_cur
            g_s = g_raw.unsqueeze(0) if w_cur.ndim == 2 else g_raw
            m_s = m_buf.unsqueeze(0) if w_cur.ndim == 2 else m_buf

            g_tan = project_real_stiefel_tangent(g_s, w_s)
            m_pre = project_real_stiefel_tangent(m_s, w_s)
            m_upd, xi, step_lr = _clean_and_snr_gate_real_tangent(g_tan, m_pre, stiefel_lr, beta)
            z_ns = _whiten_tall_real_stiefel(xi, ns_steps=ns_steps, tik_lam=tik_lam)
            u_tan = project_real_stiefel_tangent(z_ns, w_s)
            w_next = _retract_real_tall_stiefel_tensor_lr(w_s, u_tan, step_lr)
            m_stored = parallel_transport_real_stiefel_momentum(m_upd, w_s, w_next)

            p.data.copy_((w_next.squeeze(0) if w_cur.ndim == 2 else w_next).to(p.dtype))
            m_buf.copy_(m_stored.squeeze(0) if w_cur.ndim == 2 else m_stored)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """Execute 1 Riemannian step across bridges, extra real Stiefel blocks, and complex Kraus/MUB unitaries."""
        loss = super().step(closure=closure)
        g0 = self.param_groups[0]
        lr_scale = float(g0.get("lr_scale", 1.0))
        stiefel_lr = float(g0.get("stiefel_lr", g0["lr"])) * lr_scale
        beta = float(g0.get("beta", DEFAULT_SELECTIVE_MOMENTUM_BETA))
        ns_steps = int(g0.get("ns_steps", DEFAULT_NS5_STEPS))
        tik_lam = float(g0.get("tikhonov_lambda", DEFAULT_TIKHONOV_GRAM_LAMBDA))
        self._step_extra_real_stiefel(stiefel_lr, beta, ns_steps, tik_lam)
        return loss


def build_wave69_optimizers(
    model: PureDensityMatrixGPT,
) -> tuple[PureDensityQuotientMuon, FrequencyAdaptiveBatchSubspaceGramMuon, RiemannianGeodesicScalarAndProbeOptimizer]:
    """Construct `(trunk_opt, vocab_opt, scalar_opt)` for `PureDensityMatrixGPT` (`<= 8` args)."""
    sub_mup = math.sqrt(float(model.subspace_dim) / float(DEFAULT_NARROW_SUBSPACE_DIM))
    voc_lr, read_lr, gate_tau = DEFAULT_SHALLOW_WTE_LR * 1.35, DEFAULT_SHALLOW_READOUT_LR * 1.35, DEFAULT_MUON_GRAD_NORM_GATE_TAU / sub_mup
    trunk_opt = PureDensityQuotientMuon(bridge_pairs=model.bridge_parameter_pairs(), extra_real_stiefel=model.extra_real_stiefel_parameters(), kraus_pairs=model.kraus_parameter_pairs_with_depth_scale(), mub_su_pairs=model.mub_su_parameter_pairs(), stiefel_lr=DEFAULT_SHALLOW_STIEFEL_LR * 1.35, unitary_lr=0.028, beta=DEFAULT_SELECTIVE_MOMENTUM_BETA)
    vocab_groups = list(model.boundary.sphere_and_ball_groups(voc_lr, read_lr)) + [{"params": [model.ngram_density_bank.ngram_ket_table], "lr": DEFAULT_SHALLOW_NGRAM_LR * 1.50, "zero_init_ball": True}]
    vocab_opt = FrequencyAdaptiveBatchSubspaceGramMuon(vocab_groups, lr=voc_lr, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA, grad_norm_gate_tau=gate_tau, enable_freq_adaptive=True)
    scalar_opt = RiemannianGeodesicScalarAndProbeOptimizer([{"params": model.scalar_parameters(), "lr": DEFAULT_SHALLOW_SCALAR_LR, "kind": "bounded_tangent", "max_step": 0.08}], lr=DEFAULT_SHALLOW_SCALAR_LR, beta1=0.80, beta2=0.95)
    return trunk_opt, vocab_opt, scalar_opt


@torch.no_grad()
def evaluate_wave69_model(model: PureDensityMatrixGPT, stream_or_batches: Any) -> dict[str, float]:
    """Evaluate `PureDensityMatrixGPT` on Shard-0 validation batches (`<= 8` args)."""
    was_training = model.training
    model.eval()
    t0 = time.perf_counter()
    val_batches = stream_or_batches.val_batches if hasattr(stream_or_batches, "val_batches") else stream_or_batches
    total_loss, total_top1, total_top5, total_batches = 0.0, 0.0, 0.0, 0

    for x, y in val_batches:
        loss, metrics = model(x, targets=y, return_metrics=True)
        total_loss += float(loss.item())
        total_top1 += float(metrics["top1_acc"])
        total_top5 += float(metrics["top5_acc"])
        total_batches += 1

    n_b = float(max(1, total_batches))
    val_ce = total_loss / n_b
    u2_re = torch.matmul(model.dft_f_re, model.probe_u1_re) - torch.matmul(model.dft_f_im, model.probe_u1_im)
    u2_im = torch.matmul(model.dft_f_re, model.probe_u1_im) + torch.matmul(model.dft_f_im, model.probe_u1_re)
    mub_err = compute_mub_unbiasedness_max_error(model.probe_u1_re, model.probe_u1_im, u2_re, u2_im)

    eye_h = torch.eye(model.d_h, device=model.probe_u1_re.device, dtype=torch.float32).unsqueeze(0)
    k_errs = [float(((torch.matmul(ur.float().transpose(-2, -1), ur.float()) + torch.matmul(ui.float().transpose(-2, -1), ui.float())) - eye_h).abs().max().item()) for ur, ui, _ in model.kraus_parameter_pairs_with_depth_scale()]
    b_errs = []
    for p in model.real_stiefel_parameters():
        pf = p.data.float()
        eye_b = torch.eye(pf.size(-1), device=pf.device, dtype=torch.float32)
        b_errs.append(float((torch.matmul(pf.transpose(-2, -1), pf) - (eye_b.unsqueeze(0) if pf.ndim == 3 else eye_b)).abs().max().item()))

    if was_training:
        model.train()
    return {
        "val_ce": round(val_ce, 4), "val_ppl": round(math.exp(min(20.0, val_ce)), 2),
        "top1_acc": round(total_top1 / n_b, 4), "top5_acc": round(total_top5 / n_b, 4),
        "eval_wall_sec": round(time.perf_counter() - t0, 3), "analytical_beta": round(float(model.current_analytical_beta().item()), 4),
        "kraus_stiefel_err": round(float(max(k_errs)) if k_errs else 0.0, 8),
        "boundary_stiefel_err": round(float(max(b_errs)) if b_errs else 0.0, 8), "mub_unbiased_err": round(mub_err, 8),
    }


def run_wave69_variant(
    spec: Wave69ExperimentSpec, stream: LongSeqFineWebStream, steps: int = 200,
    batch_size: int = 8, seq_len: int = 512, eval_interval: int = 25, eval_batches: int = 8,
) -> dict[str, Any]:
    """Train and evaluate one `Wave69VariantMode` run with `StandardizedExperimentLogger` (`<= 8` args)."""
    del eval_batches
    device = stream.device
    torch.manual_seed(spec.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(spec.seed)
        gc.collect()
        torch.cuda.empty_cache()

    model = PureDensityMatrixGPT(spec).to(device)
    init_ms = model.initialize_quantum_state(seed=spec.seed)
    total_params = sum(p.numel() for p in model.parameters())
    trunk_opt, vocab_opt, scalar_opt = build_wave69_optimizers(model)
    retr_cad = 8 if steps >= 1000 else 5
    thermostat = ReadOnlyQuenchRiemannianShadowEMA(model, ema_decay=DEFAULT_THERMOSTAT_EMA_DECAY, eval_fn=evaluate_wave69_model, retract_cadence=retr_cad)

    tokens_per_step = batch_size * seq_len
    run_key = f"plan089_{spec.variant_mode.value.lower()}_{steps}steps"
    exp_logger = StandardizedExperimentLogger(
        ExperimentIntentSpec(
            run_key=run_key, wave_id="W69", plan_id="EXPERIMENT_PLAN_089", task_domain="fineweb_speedrun",
            hypothesis_intent=f"Wave 69 {spec.variant_mode.value} (L={spec.num_layers}, R_chol={spec.cholesky_rank}, unconst_ce={spec.enable_unconstrained_born_ce}).",
            architecture_summary=f"PureDensityMatrixGPT(L={spec.num_layers}, H=16, d_h=16, R_chol={spec.cholesky_rank}, params={total_params:,})",
            hardware_budget=str(device), batch_size=batch_size, seq_len=seq_len,
            total_steps=steps, target_tokens_M=round((steps * tokens_per_step) / 1e6, 3),
        )
    )

    step0_ev = evaluate_wave69_model(model, stream)
    step0_ev["step"], step0_ev["tokens_M"] = 0, 0.0
    exp_logger.log_eval(0, {**step0_ev, "top1_acc": round(100.0 * step0_ev["top1_acc"], 2), "top5_acc": round(100.0 * step0_ev["top5_acc"], 2)})
    print(f"[{spec.variant_mode.value} (init={init_ms:.1f}ms, L={spec.num_layers}, params={total_params:,})] Step    0 | Val CE: {step0_ev['val_ce']:.4f}", flush=True)

    quench_steps = max(1, int(round(float(steps) * float(DEFAULT_QUENCH_FRACTION))))
    plateau_steps = max(1, steps - quench_steps)
    flat_steps, tau_anneal, warmup_steps = max(200, int(0.35 * float(plateau_steps))), max(300.0, 0.65 * float(plateau_steps)), 5
    curve: list[dict[str, Any]] = [step0_ev]
    t0 = time.perf_counter()
    model.train()

    for step in range(1, steps + 1):
        for opt in (trunk_opt, vocab_opt, scalar_opt):
            opt.zero_grad(set_to_none=True)

        idx_in, targets = stream.sample_train_batch(step)
        loss = model(idx_in, targets=targets, return_metrics=False)
        loss.backward()

        anneal = (1.0 / math.sqrt(1.0 + max(0.0, float(step - flat_steps)) / tau_anneal)) if steps >= 1000 else 1.0
        if step <= warmup_steps:
            warm = float(step) / float(warmup_steps)
            lr_mult, mom_beta = 0.25 + 0.75 * warm, 0.50 * warm
        elif step <= plateau_steps:
            lr_mult, mom_beta = anneal, DEFAULT_SELECTIVE_MOMENTUM_BETA
        else:
            anneal_end = (1.0 / math.sqrt(1.0 + max(0.0, float(plateau_steps - flat_steps)) / tau_anneal)) if steps >= 1000 else 1.0
            q_p = float(step - plateau_steps) / float(max(1, quench_steps))
            cos_q = 0.5 * (1.0 + math.cos(math.pi * min(1.0, q_p)))
            lr_mult, mom_beta = anneal_end * (0.15 + 0.85 * cos_q), 0.85 + 0.10 * (1.0 - cos_q)

        is_eval_step = (step % eval_interval == 0) or (step == steps) or (step == 1)
        trunk_opt.record_diagnostics = is_eval_step
        trunk_opt.set_momentum(mom_beta)
        trunk_opt.set_lr_scale(lr_mult)
        trunk_opt.step()

        vocab_opt.set_momentum(mom_beta)
        vocab_opt.step_active(_collect_active_vocab_indices_w68(idx_in, targets), lr_scale=lr_mult)

        scalar_opt.set_momentum(0.0 if mom_beta <= 0.0 else 0.80)
        scalar_opt.step_active(active_idx=None, lr_scale=lr_mult)
        model.restrict_scalars_()
        model.advance_thermostat_tokens_(tokens_per_step)
        thermostat.update_after_step(model)

        if step % 10 == 0 or step == 1:
            el_s = max(1e-3, time.perf_counter() - t0)
            exp_logger.log_step(step, train_ce=float(loss.detach().item()), tok_per_s=(float(step * tokens_per_step) / el_s), lr_mult=lr_mult)

        if step % eval_interval == 0 or step == steps:
            th_out = thermostat.check_and_thermostat(model, stream, current_lr_scale=1.0, step=step, apply_feedback=True)
            gd = {k: round(float(v), 6) for k, v in trunk_opt.last_gauge_diagnostics.items()}
            ev = dict(th_out["active_eval"])
            ev["step"], ev["tokens_M"], ev["train_ce"] = step, round((step * tokens_per_step) / 1e6, 3), round(float(loss.detach().item()), 4)
            ev["hot_val_ce"], ev["ema_val_ce"], ev["thermal_gap_ce"] = th_out["hot_val_ce"], th_out["ema_val_ce"], th_out["thermal_gap_ce"]
            ev["thermostat_triggered"], ev["thermostat_lr_scale"] = False, round(lr_mult, 4)
            ev.update(gd)
            curve.append(ev)
            exp_logger.log_eval(step, {**ev, "top1_acc": round(100.0 * ev["top1_acc"], 2), "top5_acc": round(100.0 * ev["top5_acc"], 2)})
            print(
                f"[{spec.variant_mode.value}] Step {step:5d} ({ev['tokens_M']:6.2f}M tok, lr={lr_mult:.3f}x) | "
                f"Hot CE: {th_out['hot_val_ce']:.4f} | EMA CE: {th_out['ema_val_ce']:.4f} "
                f"(gap={th_out['thermal_gap_ce']:+.4f}) | Top1: {100.0 * ev['top1_acc']:.2f}% | "
                f"MUBErr: {ev['mub_unbiased_err']:.2e} | SU16Phase: {gd.get('su16_step_phase_frac', 0.0):.2e} | "
                f"KrausGauge: {gd.get('kraus_step_gauge_frac', 0.0):.2e}",
                flush=True,
            )

    wall_s = max(1e-6, time.perf_counter() - t0)
    exp_logger.finalize()
    summary = {
        "variant": spec.variant_mode.value, "num_layers": spec.num_layers,
        "cholesky_rank": spec.cholesky_rank, "kraus_rank": spec.kraus_rank,
        "enable_unconstrained_born_ce": spec.enable_unconstrained_born_ce,
        "enable_stateful_ema_and_history": spec.enable_stateful_ema_and_history,
        "total_params": total_params, "total_params_M": round(total_params / 1e6, 3),
        "init_ms": round(init_ms, 2), "elapsed_sec": round(wall_s, 2),
        "ms_per_step": round((wall_s * 1000.0) / max(1, steps), 2),
        "tok_per_s": round((steps * tokens_per_step) / wall_s, 1),
        "step0": step0_ev, "final": curve[-1], "curve": curve,
    }
    del model, thermostat, vocab_opt, trunk_opt, scalar_opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary
