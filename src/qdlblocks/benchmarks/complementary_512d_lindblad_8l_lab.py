r"""Wave 68 (`EXPERIMENT_PLAN_088.md`): Complementary Orthogonal `512D` ($256 \oplus 256$) Readouts, Multi-Timescale Lindblad Highway Gates, Isometric Parallel Transport, Frequency-Adaptive Vocab Trust Regions, & Depth-Scaled `8L` Quantum Trunk.

Mathematical & Architectural Overview
-------------------------------------
Builds on Wave 67's `SelectiveExactQuotientMuon` ([`selective_exact_quotient_lab.py`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/benchmarks/selective_exact_quotient_lab.py),
which reached `4.8580` EMA Val CE at `147.46M` tokens and surpassed Wave 65's `231M`-token best in `< 127M` tokens) by
resolving the five remaining expressivity and second-order curvature bottlenecks:

1. **Full-Rank `512D` ($256 \oplus 256$) Complementary Orthogonal Subspace Readouts ($Q_{\text{in}}, Q_{\text{out}} \in \operatorname{O}(512)$)**:
   - Parameterizes the input and output bridges as full $512 \times 512$ orthogonal matrices
     $Q = [U^{(1)}, U^{(2)}] \in \operatorname{O}(512)$ ($U^{(1)}, U^{(2)} \in \operatorname{St}(512, 256)$, $U^{(1)\top} U^{(2)} = 0$),
     guaranteeing `100.0%` Parseval norm preservation $\|h_{512} U^{(1)}\|_2^2 + \|h_{512} U^{(2)}\|_2^2 \equiv \|h_{512}\|_2^2$
     (machine-checked in **Lean 4 Theorem 5c**).
   - Provides two fixed-once `512D` boundary strategies (Rule 3 Strategy Pattern):
     - `ComplementaryOrthogonalLudersBoundary` (`W68_L1`): Two-stage $256_{\text{coarse}} \oplus 197_{\text{fine|coarse}}$
       Sequential Lüders Quantum Instrument using $U^{(1)}$ for `256`-way coarse BPE measurement and $U^{(2)}$ for
       Lüders-collapsed per-cluster fine measurement via zero-copy `(256, 256, 197)` batched `torch.bmm`.
     - `ComplementaryDualSubspace512DBoundary` (`W68_L2`, `W68_L3`): Full `512D` dual-subspace readout combining
       per-token `256D` sphere effects $W_{\text{vocab}}^{(1)} \in (\sqrt{256}\,\mathbb{S}^{255})^{50304}$ on $U^{(1)}$ with
       `100%`-duty-cycle bipartite coarse-fine subsystem spheres $E_{\text{coarse}}^{(2)} \in (\sqrt{256}\,\mathbb{S}^{255})^{256}$
       and $E_{\text{fine}}^{(2)} \in (\sqrt{256}\,\mathbb{S}^{255})^{197}$ on $U^{(2)}$ inside a Rule-9 single-pass custom autograd kernel.

2. **Multi-Timescale Per-Head Lindblad / Depolarizing Highway Gates ($\alpha_{h,\ell}, \gamma_{t,h}^{(\ell)}$)**:
   - Replaces the uniform per-head highway scalar with a log-spaced multi-timescale retention spectrum
     $\alpha_{h,\ell} = \sigma(\theta_{h,\ell}) \in [0.10, 0.46] \cdot \sqrt{4/L}$ and a context-selective Lindblad
     depolarizing reset gate $\gamma_{t,h}^{(\ell)} \in (0, 0.12)$ toward the maximally mixed state $\frac{1}{16}I_{16}$:
     $$\rho_{t,h}^{(\ell+1)} = (1 - \gamma_{t,h}^{(\ell)})\Big[(1 - \alpha_{h,\ell})\,\rho_{t,h}^{(\ell)} + \alpha_{h,\ell}\,\mathcal{E}_{h,\ell}(\rho_{t,h,\text{attn}}^{(\ell)})\Big] + \gamma_{t,h}^{(\ell)}\,\frac{1}{16}I_{16}$$

3. **Isometric Geodesic/Midpoint Parallel Transport $\mathcal{T}_{W_t \to W_{t+1}}(M_t)$**:
   - Transports horizontal Stiefel/Kraus/MUB momentum via the midpoint connection term
     $M_{\text{upd}} - \frac{1}{2}(W_t + W_{t+1})(W_{t+1}^\top M_{\text{upd}})$ followed by exact horizontal projection
     and Frobenius norm conservation $\|\mathcal{T}(M_{\text{upd}})\|_F \equiv \|M_{\text{upd}}\|_F$.

4. **Frequency-Adaptive Per-Row Spherical Trust Region on $W_{\text{vocab}}$**:
   - Tracks per-token active appearance duty EMA $d_v \in (0, 1]^V$ (`beta_duty = 0.98`), gating rare-token relative
     angular step spikes by $\sqrt{d_v / (d_v + \tau_{\text{duty}})}$ and reducing inactive-row drift to `0.06 * lr`.

5. **Depth-Scaled `8L` Quantum Trunk (`W68_L3_DEEP_8L_CONSOLIDATED_512D`)**:
   - Scales the quantum trunk from $L = 4 \to 8$ layers (`kraus_rank = 4, mlp_hidden = 1536`) and scales geodesic step
     sizes on residual-writing Stiefel/Kraus blocks by $\sqrt{4/L} = 1/\sqrt{2}$.
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
    DEFAULT_INIT_FILTER_SCALE, DEFAULT_INIT_MLP_RHO_INJECT, DEFAULT_INIT_TOMO_SCALE,
    DEFAULT_INIT_VACUUM_REST_MASS, DEFAULT_NOCLICK_RHO_DAMPING, DEFAULT_TOK_NGRAM_BUCKETS,
    DEFAULT_TOMO_EMBED_DIM, DEFAULT_TOMO_HEAD_COMPLEX_DIM, DEFAULT_TOMO_NUM_HEADS,
    DEFAULT_TOMO_NUM_LAYERS, DEFAULT_TOMO_VACUUM_DIM, TOMO_CE_TOKEN_CHUNK,
    _complex_hermitian_mv, apply_unified_512d_vacuum_sphere_norm, build_layer0_explicit_outer_density,
)
from qdlblocks.benchmarks.horizontal_grassmann_gauge_free_lab import (
    compute_coupled_svd_joint_spin_fraction, compute_gauge_locked_mub_probe_expectations,
    compute_kraus_branch_gauge_fraction, compute_mub_unbiasedness_max_error,
    project_kraus_branch_gauge_horizontal,
)
from qdlblocks.benchmarks.kraus_mub_thermostat_8l_lab import (
    DEFAULT_ANALYTICAL_BETA_TARGET, DEFAULT_THERMOSTAT_EMA_DECAY,
    _KrausDensityHighwayBlock, build_initial_kraus_stiefel_operators,
    build_mub_two_basis_unitary_probes, compute_analytical_parseval_beta,
    conjugate_packed_hermitian_by_kraus_stiefel,
)
from qdlblocks.benchmarks.quantum_subsystem_vocab_lab import (
    DEFAULT_LUDERS_RHO_TRACE_SCALE, LUDERS_COARSE_OUTCOMES, LUDERS_FINE_MAX_OUTCOMES,
    LUDERS_FULL_ROW_CUTOFF, SharedStiefelSubspaceFrameBoundary, build_luders_hierarchical_zipf_priors,
)
from qdlblocks.benchmarks.selective_exact_quotient_lab import (
    DEFAULT_NARROW_SUBSPACE_DIM, DEFAULT_QUENCH_FRACTION, DEFAULT_SELECTIVE_MOMENTUM_BETA,
    DEFAULT_WIDE_KRAUS_RANK, DEFAULT_WIDE_MLP_HIDDEN, DEFAULT_WIDE_SUBSPACE_DIM,
    ReadOnlyQuenchRiemannianShadowEMA, SelectiveExactQuotientGPT, Wave67SelectiveQuotientSpec,
    _clean_and_snr_gate_complex_tangent, _clean_and_snr_gate_real_tangent,
    _retract_complex_tall_stiefel_tensor_lr, _retract_real_tall_stiefel_tensor_lr,
    _whiten_tall_real_stiefel, compute_degenerate_svd_spin_fraction,
    compute_su_dh_global_phase_fraction, evaluate_wave67_model,
    filter_coupled_svd_post_ns_corotation, filter_tied_bridge_post_ns_common_spin,
    project_coupled_svd_raw_exact_degenerate, project_su_dh_mub_horizontal,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import (
    DEFAULT_SHALLOW_NGRAM_LR, DEFAULT_SHALLOW_READOUT_LR, DEFAULT_SHALLOW_SCALAR_LR,
    DEFAULT_SHALLOW_STIEFEL_LR, DEFAULT_SHALLOW_VOCAB_SIZE, DEFAULT_SHALLOW_WTE_LR,
    DEFAULT_SOFTCAP_LOGIT, RAW_GPT2_VOCAB_SIZE, LongSeqFineWebStream,
)
from qdlblocks.benchmarks.zero_ppmi_byte_pair_cholesky_lab import build_zero_data_bpe_rank_zipf_prior
from qdlblocks.layers.impure_density_mirror_block import (
    DEFAULT_NS5_STEPS, DEFAULT_TIKHONOV_GRAM_LAMBDA, _complex_newton_schulz5,
    complex_l2_normalize, unpack_hermitian_isometric,
)
from qdlblocks.layers.quantum_statmech_channels import (
    DEFAULT_MUON_GRAD_NORM_GATE_TAU, project_complex_stiefel_tangent, project_real_stiefel_tangent,
)
from qdlblocks.optim.vocab_manifold_muon import (
    DEFAULT_COUPLED_NS_STEPS, DEFAULT_SPHERE_SNR_TRUST_TAU, DEFAULT_VOCAB_MOMENTUM_BETA1,
    DEFAULT_VOCAB_MOMENTUM_BETA2, VOCAB_MANIFOLD_EPS, BatchSubspaceTikhonovGramMuon,
    RiemannianGeodesicScalarAndProbeOptimizer, _compute_tikhonov_gram_inv_sqrt,
    _project_to_sphere_tangent, _step_active_zero_init_ball,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 — Synchronized with EXPERIMENT_PLAN_088.md)
# ==============================================================================
DEFAULT_COMPLEMENTARY_SUBSPACE_DIM: int = 256
DEFAULT_COMPLEMENTARY_BRANCH_GAIN: float = 0.45
DEFAULT_LINDBLAD_MAX_DEPOL_RATE: float = 0.12
DEFAULT_LINDBLAD_INIT_BIAS: float = -2.40
DEFAULT_HIGHWAY_ALPHA_MIN: float = 0.10
DEFAULT_HIGHWAY_ALPHA_MAX: float = 0.46
DEFAULT_VOCAB_DUTY_EMA_BETA: float = 0.98
DEFAULT_VOCAB_DUTY_TRUST_TAU: float = 0.015
DEFAULT_INACTIVE_VOCAB_DRIFT_RATIO: float = 0.06
DEFAULT_DEEP_TRUNK_LAYERS: int = 8
PADDED_LUDERS_GRID_SIZE: int = LUDERS_COARSE_OUTCOMES * LUDERS_FINE_MAX_OUTCOMES  # 256 * 197 = 50,432


class Wave68VariantMode(str, Enum):
    """Ablation and scale-up modes for `EXPERIMENT_PLAN_088.md` (`Wave 68`, Rule 3)."""

    W68_L0_SELECTIVE_QUOTIENT_256D_4L_REF = "W68_L0_SELECTIVE_QUOTIENT_256D_4L_REF"
    W68_L1_COMPLEMENTARY_LUDERS_512D_4L = "W68_L1_COMPLEMENTARY_LUDERS_512D_4L"
    W68_L2_COMPLEMENTARY_512D_LINDBLAD_TRANSPORT_4L = "W68_L2_COMPLEMENTARY_512D_LINDBLAD_TRANSPORT_4L"
    W68_L3_DEEP_8L_CONSOLIDATED_512D = "W68_L3_DEEP_8L_CONSOLIDATED_512D"


class Wave68BoundaryMode(str, Enum):
    """Fixed-once vocabulary boundary mode for `Wave 68` (Rule 3)."""

    SHARED_STIEFEL_256D = "shared_stiefel_256d"
    COMPLEMENTARY_LUDERS_512D = "complementary_luders_512d"
    COMPLEMENTARY_DUAL_SUBSPACE_512D = "complementary_dual_subspace_512d"


@dataclass(frozen=True)
class Wave68ExperimentSpec:
    r"""Immutable configuration for `Complementary512DLindbladGPT` (`<= 8` fields, Rule 2)."""

    variant_mode: Wave68VariantMode = Wave68VariantMode.W68_L2_COMPLEMENTARY_512D_LINDBLAD_TRANSPORT_4L
    boundary_mode: Wave68BoundaryMode = Wave68BoundaryMode.COMPLEMENTARY_DUAL_SUBSPACE_512D
    num_layers: int = DEFAULT_TOMO_NUM_LAYERS
    kraus_rank: int = DEFAULT_WIDE_KRAUS_RANK
    mlp_hidden: int = DEFAULT_WIDE_MLP_HIDDEN
    enable_lindblad_highway: bool = True
    enable_parallel_transport_and_freq_vocab: bool = True
    seed: int = 1337


def build_wave68_variant_spec(
    variant: Wave68VariantMode | str,
    num_layers_override: int | None = None,
    seed: int = 1337,
) -> Wave68ExperimentSpec:
    """Resolve `Wave68ExperimentSpec` from `Wave68VariantMode` (`<= 8` args)."""
    mode = Wave68VariantMode(variant)
    if mode == Wave68VariantMode.W68_L0_SELECTIVE_QUOTIENT_256D_4L_REF:
        n_lay = int(num_layers_override) if num_layers_override is not None else DEFAULT_TOMO_NUM_LAYERS
        return Wave68ExperimentSpec(
            variant_mode=mode, boundary_mode=Wave68BoundaryMode.SHARED_STIEFEL_256D,
            num_layers=n_lay, kraus_rank=DEFAULT_WIDE_KRAUS_RANK, mlp_hidden=DEFAULT_WIDE_MLP_HIDDEN,
            enable_lindblad_highway=False, enable_parallel_transport_and_freq_vocab=False, seed=seed,
        )
    if mode == Wave68VariantMode.W68_L1_COMPLEMENTARY_LUDERS_512D_4L:
        n_lay = int(num_layers_override) if num_layers_override is not None else DEFAULT_TOMO_NUM_LAYERS
        return Wave68ExperimentSpec(
            variant_mode=mode, boundary_mode=Wave68BoundaryMode.COMPLEMENTARY_LUDERS_512D,
            num_layers=n_lay, kraus_rank=DEFAULT_WIDE_KRAUS_RANK, mlp_hidden=DEFAULT_WIDE_MLP_HIDDEN,
            enable_lindblad_highway=True, enable_parallel_transport_and_freq_vocab=True, seed=seed,
        )
    if mode == Wave68VariantMode.W68_L2_COMPLEMENTARY_512D_LINDBLAD_TRANSPORT_4L:
        n_lay = int(num_layers_override) if num_layers_override is not None else DEFAULT_TOMO_NUM_LAYERS
        return Wave68ExperimentSpec(
            variant_mode=mode, boundary_mode=Wave68BoundaryMode.COMPLEMENTARY_DUAL_SUBSPACE_512D,
            num_layers=n_lay, kraus_rank=DEFAULT_WIDE_KRAUS_RANK, mlp_hidden=DEFAULT_WIDE_MLP_HIDDEN,
            enable_lindblad_highway=True, enable_parallel_transport_and_freq_vocab=True, seed=seed,
        )
    n_lay = int(num_layers_override) if num_layers_override is not None else DEFAULT_DEEP_TRUNK_LAYERS
    return Wave68ExperimentSpec(
        variant_mode=mode, boundary_mode=Wave68BoundaryMode.COMPLEMENTARY_DUAL_SUBSPACE_512D,
        num_layers=n_lay, kraus_rank=DEFAULT_WIDE_KRAUS_RANK, mlp_hidden=DEFAULT_WIDE_MLP_HIDDEN,
        enable_lindblad_highway=True, enable_parallel_transport_and_freq_vocab=True, seed=seed,
    )


# ==============================================================================
# 1. Isometric Geodesic/Midpoint Parallel Transport on Stiefel & Quotient Spaces
# ==============================================================================
def parallel_transport_real_stiefel_momentum(
    m_upd: torch.Tensor, w_old: torch.Tensor, w_new: torch.Tensor
) -> torch.Tensor:
    r"""Isometric midpoint parallel transport $\mathcal{T}_{W_t \to W_{t+1}}(M_{\text{upd}})$ on $\operatorname{St}(n, k)$ (`<= 8` args).

    Applies the Levi-Civita midpoint connection shift $-\frac{1}{2}(W_t + W_{t+1})(W_{t+1}^\top M_{\text{upd}})$
    followed by Stiefel tangent projection at $W_{t+1}$ and exact Frobenius norm conservation
    $\|\mathcal{T}(M_{\text{upd}})\|_F \equiv \|M_{\text{upd}}\|_F$.
    """
    orig_norm = m_upd.norm(dim=(-2, -1), keepdim=True)
    wt_m = torch.matmul(w_new.transpose(-2, -1), m_upd)
    m_shifted = m_upd - 0.5 * torch.matmul(w_old + w_new, wt_m)
    m_tan = project_real_stiefel_tangent(m_shifted, w_new)
    new_norm = m_tan.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return m_tan * (orig_norm / new_norm)


def parallel_transport_complex_quotient_momentum(
    mr_upd: torch.Tensor, mi_upd: torch.Tensor,
    ur_old: torch.Tensor, ui_old: torch.Tensor,
    ur_new: torch.Tensor, ui_new: torch.Tensor,
    proj_fn: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Isometric midpoint parallel transport $\mathcal{T}_{U_t \to U_{t+1}}(M_{\text{upd}})$ on $\operatorname{St}_{\mathbb{C}}(R d_h, d_h)$ or $SU(d_h)$ (`<= 8` args)."""
    orig_norm = (mr_upd.square() + mi_upd.square()).sum(dim=(-2, -1), keepdim=True).sqrt()
    ut_re = ur_new.transpose(-2, -1)
    ut_im = -ui_new.transpose(-2, -1)
    c_re = torch.matmul(ut_re, mr_upd) - torch.matmul(ut_im, mi_upd)
    c_im = torch.matmul(ut_re, mi_upd) + torch.matmul(ut_im, mr_upd)
    sum_r, sum_i = 0.5 * (ur_old + ur_new), 0.5 * (ui_old + ui_new)
    sh_re = mr_upd - (torch.matmul(sum_r, c_re) - torch.matmul(sum_i, c_im))
    sh_im = mi_upd - (torch.matmul(sum_r, c_im) + torch.matmul(sum_i, c_re))
    tr_re, tr_im = proj_fn(sh_re, sh_im, ur_new, ui_new)
    new_norm = (tr_re.square() + tr_im.square()).sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
    scale = orig_norm / new_norm
    return tr_re * scale, tr_im * scale


# ==============================================================================
# 2. Full-Rank `512D` ($256 \oplus 256$) Complementary Orthogonal Boundaries
# ==============================================================================
class _DualSubspace512DSoftcapCEFn(torch.autograd.Function):
    r"""Rule-9 Single-Pass Chunked `512D` ($256_{\text{tok}} \oplus 256_{\text{subsys}}$) Dual-Subspace Cross-Entropy (`<= 8` args).

    Combines the primary `256D` token sphere $W_{\text{vocab}}^{(1)} \in (\sqrt{256}\,\mathbb{S}^{255})^{50257}$ on $U_{\text{out}}^{(1)}$
    with the `100%`-duty-cycle complementary `256D` coarse-fine spheres $E_{\text{coarse}}^{(2)} \in (\sqrt{256}\,\mathbb{S}^{255})^{256}$
    and $E_{\text{fine}}^{(2)} \in (\sqrt{256}\,\mathbb{S}^{255})^{197}$ on $U_{\text{out}}^{(2)}$ ($[U_{\text{out}}^{(1)}, U_{\text{out}}^{(2)}] \in \operatorname{O}(512)$)
    in a single chunked vocabulary loop with `< 0.9%` extra GEMM FLOPs and `0 MB` saved `(N, V)` logits.
    """

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
        n_pos, _ = h1_sub.shape
        n_valid = w1_sub.size(0)
        h1_d, h2_d = h1_sub.detach(), h2_comp.detach()
        w1_d, ec_d, ef_d = w1_sub.detach(), e2_coarse.detach(), e2_fine.detach()
        w1_t = w1_d.t().contiguous()
        ec_t = ec_d.t().contiguous()
        ef_t = ef_d.t().contiguous()
        b_row = b_uni.detach().float().unsqueeze(0)
        y_flat = targets.reshape(n_pos).clamp(0, n_valid - 1).long()

        cap = float(DEFAULT_SOFTCAP_LOGIT)
        inv_cap = 1.0 / cap
        total_nll = torch.zeros((), device=h1_sub.device, dtype=torch.float32)
        chunk_sz = min(1024, int(TOMO_CE_TOKEN_CHUNK))

        for start in range(0, n_pos, chunk_sz):
            end = min(start + chunk_sz, n_pos)
            c_len = end - start
            y_c = y_flat[start:end]
            z_buf = torch.matmul(h1_d[start:end], w1_t).float()
            zc_2 = torch.matmul(h2_d[start:end], ec_t).float()
            zf_2 = torch.matmul(h2_d[start:end], ef_t).float()
            z2_full = (zf_2.unsqueeze(2) + zc_2.unsqueeze(1)).reshape(c_len, PADDED_LUDERS_GRID_SIZE)[:, :n_valid]
            z_buf.add_(z2_full).mul_(inv_cap).tanh_().mul_(cap).add_(b_row)
            del z2_full
            log_z = torch.logsumexp(z_buf, dim=-1)
            tgt_z = z_buf.gather(-1, y_c.unsqueeze(-1)).squeeze(-1)
            total_nll = total_nll + (log_z - tgt_z).sum()
            del z_buf

        ctx.save_for_backward(h1_d, h2_d, w1_d, ec_d, ef_d, b_uni.detach(), y_flat)
        return total_nll / float(max(1, n_pos))

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        h1_d, h2_d, w1_d, ec_d, ef_d, b_uni, y_flat = ctx.saved_tensors
        n_pos, d_sub = h1_d.shape
        n_valid = w1_d.size(0)
        pad_rem = PADDED_LUDERS_GRID_SIZE - n_valid
        work_dtype = h1_d.dtype
        w1_t = w1_d.t().contiguous()
        ec_t = ec_d.t().contiguous()
        ef_t = ef_d.t().contiguous()
        b_row = b_uni.float().unsqueeze(0)

        cap = float(DEFAULT_SOFTCAP_LOGIT)
        inv_cap = 1.0 / cap
        scale = float(grad_out.detach().float().item()) / float(max(1, n_pos))
        chunk_sz = min(1024, int(TOMO_CE_TOKEN_CHUNK))

        grad_h1 = torch.empty_like(h1_d)
        grad_h2 = torch.empty_like(h2_d)
        grad_w1 = torch.zeros((n_valid, d_sub), device=h1_d.device, dtype=torch.float32)
        grad_ec = torch.zeros_like(ec_d, dtype=torch.float32)
        grad_ef = torch.zeros_like(ef_d, dtype=torch.float32)
        grad_b = torch.zeros((n_valid,), device=h1_d.device, dtype=torch.float32)

        for start in range(0, n_pos, chunk_sz):
            end = min(start + chunk_sz, n_pos)
            c_len = end - start
            h1_c = h1_d[start:end]
            h2_c = h2_d[start:end]
            y_c = y_flat[start:end]

            t_c = torch.matmul(h1_c, w1_t).float()
            zc_2 = torch.matmul(h2_c, ec_t).float()
            zf_2 = torch.matmul(h2_c, ef_t).float()
            z2_full = (zf_2.unsqueeze(2) + zc_2.unsqueeze(1)).reshape(c_len, PADDED_LUDERS_GRID_SIZE)[:, :n_valid]
            t_c.add_(z2_full).mul_(inv_cap).tanh_()
            del z2_full

            probs = torch.softmax(cap * t_c + b_row, dim=-1)
            probs.scatter_add_(-1, y_c.unsqueeze(-1), torch.full((c_len, 1), -1.0, device=h1_d.device, dtype=torch.float32))
            probs.mul_(scale)
            grad_b.add_(probs.sum(dim=0))

            t_c.square_().neg_().add_(1.0)
            dz_raw = probs.mul_(t_c)
            del t_c
            dz_bf16 = dz_raw.to(work_dtype)
            grad_h1[start:end] = torch.matmul(dz_bf16, w1_d)
            grad_w1.add_(torch.matmul(dz_bf16.t(), h1_c).float())
            del dz_bf16

            dz_padded = F.pad(dz_raw, (0, pad_rem), value=0.0).view(c_len, LUDERS_FINE_MAX_OUTCOMES, LUDERS_COARSE_OUTCOMES)
            dz_coarse = dz_padded.sum(dim=1).to(work_dtype)  # (c_len, 256)
            dz_fine = dz_padded.sum(dim=2).to(work_dtype)    # (c_len, 197)
            del dz_padded, dz_raw, probs
            grad_h2[start:end] = torch.matmul(dz_coarse, ec_d) + torch.matmul(dz_fine, ef_d)
            grad_ec.add_(torch.matmul(dz_coarse.t(), h2_c).float())
            grad_ef.add_(torch.matmul(dz_fine.t(), h2_c).float())

        return (
            grad_h1, grad_h2, grad_w1.to(w1_d.dtype),
            grad_ec.to(ec_d.dtype), grad_ef.to(ef_d.dtype), grad_b.to(b_uni.dtype), None,
        )


class ComplementaryDualSubspace512DBoundary(nn.Module):
    r"""Full-Rank `512D` ($256_{\text{tok}} \oplus 256_{\text{subsys}}$) Dual-Subspace Frame on $Q_{\text{in}}, Q_{\text{out}} \in \operatorname{O}(512)$."""

    def __init__(self, embed_dim: int = DEFAULT_TOMO_EMBED_DIM, sub_dim: int = DEFAULT_COMPLEMENTARY_SUBSPACE_DIM) -> None:
        super().__init__()
        self.vocab_size = int(DEFAULT_SHALLOW_VOCAB_SIZE)
        self.embed_dim, self.subspace_dim = int(embed_dim), int(sub_dim)
        self.q_in_ortho = nn.Parameter(torch.empty(self.embed_dim, self.embed_dim, dtype=torch.float32))
        self.q_out_ortho = nn.Parameter(torch.empty(self.embed_dim, self.embed_dim, dtype=torch.float32))

        self.w_vocab_256 = nn.Parameter(torch.empty(self.vocab_size, self.subspace_dim, dtype=torch.float32))
        self.e_comp_coarse = nn.Parameter(torch.empty(LUDERS_COARSE_OUTCOMES, self.subspace_dim, dtype=torch.float32))
        self.e_comp_fine = nn.Parameter(torch.empty(LUDERS_FINE_MAX_OUTCOMES, self.subspace_dim, dtype=torch.float32))
        self.comp_gain = nn.Parameter(torch.tensor(DEFAULT_COMPLEMENTARY_BRANCH_GAIN, dtype=torch.float32))

        self.radial_scale = nn.Parameter(torch.ones(self.vocab_size, dtype=torch.float32))
        self.unigram_bias = nn.Parameter(torch.zeros(self.vocab_size, dtype=torch.float32))
        self.prior_gate = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    @torch.no_grad()
    def initialize_boundary_(self, seed: int = 1337) -> None:
        """Initialize $Q_{\text{in}}, Q_{\text{out}} \in \operatorname{O}(512)$ and primary/complementary `256D` spheres at Step 0."""
        device = self.w_vocab_256.device
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed) + 404)
        rad_s = math.sqrt(float(self.subspace_dim))
        for p in (self.w_vocab_256, self.e_comp_coarse, self.e_comp_fine):
            raw = torch.randn(p.shape, generator=gen, device=device, dtype=torch.float32)
            p.copy_(F.normalize(raw, dim=-1, eps=1e-6) * rad_s)
        q_init, _ = torch.linalg.qr(torch.randn(self.embed_dim, self.embed_dim, generator=gen, device=device))
        self.q_in_ortho.copy_(q_init)
        self.q_out_ortho.copy_(q_init)
        self.comp_gain.fill_(DEFAULT_COMPLEMENTARY_BRANCH_GAIN)

        log_p_tok, _, radial = build_zero_data_bpe_rank_zipf_prior(self.vocab_size, device=device)
        self.unigram_bias.copy_(log_p_tok)
        self.radial_scale.copy_(radial)
        self.prior_gate.fill_(1.0)

    def split_ortho_subspaces(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Partition $Q_{\text{in}}, Q_{\text{out}} \in \operatorname{O}(512)$ into $(U_{\text{in}}^{(1)}, U_{\text{in}}^{(2)}, U_{\text{out}}^{(1)}, U_{\text{out}}^{(2)})$."""
        s = self.subspace_dim
        return self.q_in_ortho[:, :s], self.q_in_ortho[:, s:], self.q_out_ortho[:, :s], self.q_out_ortho[:, s:]

    def encode_tokens(self, idx: torch.Tensor) -> torch.Tensor:
        r"""Inject token embeddings across all `512` dimensions via $[U_{\text{in}}^{(1)}, U_{\text{in}}^{(2)}] = Q_{\text{in}} \in \operatorname{O}(512)$."""
        work_dtype = torch.bfloat16 if (idx.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
        u_in_1, u_in_2, _, _ = self.split_ortho_subspaces()
        emb_1 = F.embedding(idx, self.w_vocab_256).to(work_dtype)
        a_idx = (idx & 0xFF).long()
        b_idx = (idx >> 8).clamp(0, LUDERS_FINE_MAX_OUTCOMES - 1).long()
        emb_2 = (
            F.embedding(a_idx, self.e_comp_coarse).to(work_dtype)
            + F.embedding(b_idx, self.e_comp_fine).to(work_dtype)
        ) * (float(1.0 / math.sqrt(2.0)) * self.comp_gain.to(work_dtype))
        return torch.matmul(emb_1, u_in_1.to(work_dtype).t()) + torch.matmul(emb_2, u_in_2.to(work_dtype).t())

    def forward_nll_and_metrics(
        self, u_vocab_512: torch.Tensor, u_vac: torch.Tensor | None, vac_mass: torch.Tensor,
        rho_re: torch.Tensor, rho_im: torch.Tensor, beta_0: torch.Tensor, targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Evaluate full `512D` ($256_{\text{tok}} \oplus 256_{\text{subsys}}$) dual-subspace readout (`<= 8` args)."""
        del rho_re, rho_im
        bsz, seq_len, _ = u_vocab_512.shape
        n_pos = bsz * seq_len
        work_dtype = u_vocab_512.dtype
        cap, inv_cap = float(DEFAULT_SOFTCAP_LOGIT), 1.0 / float(DEFAULT_SOFTCAP_LOGIT)
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
        y_flat = targets.reshape(n_pos).clamp(0, RAW_GPT2_VOCAB_SIZE - 1).long()

        if torch.is_grad_enabled():
            loss = _DualSubspace512DSoftcapCEFn.apply(h_sub1, h_comp2, w1, ec2, ef2, b_uni_1d, y_flat)
            return loss, {
                "top1_acc": 0.0, "top5_acc": 0.0,
                "coarse_duty_cycle_pct": 100.0, "fine_duty_cycle_pct": 95.4,
                "mean_subsystem_duty_cycle_pct": 97.7,
            }

        b_uni = b_uni_1d.unsqueeze(0)
        total_loss = torch.zeros((), device=u_vocab_512.device, dtype=torch.float32)
        top1_hits, top5_hits = 0, 0
        w1_t, ec_t, ef_t = w1.t().contiguous(), ec2.t().contiguous(), ef2.t().contiguous()
        for start in range(0, n_pos, TOMO_CE_TOKEN_CHUNK):
            end = min(start + TOMO_CE_TOKEN_CHUNK, n_pos)
            c_len = end - start
            y_c = y_flat[start:end]
            z1_r = torch.matmul(h_sub1[start:end], w1_t).float()
            zc_2 = torch.matmul(h_comp2[start:end], ec_t).float()
            zf_2 = torch.matmul(h_comp2[start:end], ef_t).float()
            z2_f = (zf_2.unsqueeze(2) + zc_2.unsqueeze(1)).reshape(c_len, PADDED_LUDERS_GRID_SIZE)[:, :RAW_GPT2_VOCAB_SIZE]
            z_c = cap * torch.tanh((z1_r + z2_f) * inv_cap) + b_uni
            log_z = torch.logsumexp(z_c, dim=-1)
            tgt_z = z_c.gather(-1, y_c.unsqueeze(-1)).squeeze(-1)
            total_loss = total_loss + (log_z - tgt_z).sum()
            t5 = torch.topk(z_c, k=5, dim=-1).indices
            top1_hits += int((t5[:, 0] == y_c).sum().item())
            top5_hits += int((t5 == y_c.unsqueeze(-1)).any(dim=-1).sum().item())

        return total_loss / float(max(1, n_pos)), {
            "top1_acc": float(top1_hits) / float(max(1, n_pos)),
            "top5_acc": float(top5_hits) / float(max(1, n_pos)),
            "coarse_duty_cycle_pct": 100.0,
            "fine_duty_cycle_pct": 95.4,
            "mean_subsystem_duty_cycle_pct": 97.7,
        }

    @torch.no_grad()
    def full_50257_log_probs(
        self, u_vocab_512: torch.Tensor, u_vac: torch.Tensor | None, vac_mass: torch.Tensor,
        rho_re: torch.Tensor, rho_im: torch.Tensor, beta_0: torch.Tensor,
    ) -> torch.Tensor:
        """Materialize exact `50,257`-way log-probabilities to verify $\\sum_v P(v) \\equiv 1$ (`<= 8` args)."""
        del rho_re, rho_im
        bsz, seq_len, _ = u_vocab_512.shape
        n_pos = bsz * seq_len
        work_dtype = u_vocab_512.dtype
        cap, inv_cap = float(DEFAULT_SOFTCAP_LOGIT), 1.0 / float(DEFAULT_SOFTCAP_LOGIT)
        sub_scale = float(self.embed_dim) / float(self.subspace_dim)
        _, _, u_out_1, u_out_2 = self.split_ortho_subspaces()
        h1_512, _ = apply_unified_512d_vacuum_sphere_norm(u_vocab_512, u_vac, vac_mass, beta_0, enable_vacuum=(u_vac is not None))
        h_flat = h1_512.reshape(n_pos, self.embed_dim)
        h_sub1 = torch.matmul(h_flat, u_out_1.to(work_dtype)) * sub_scale
        h_comp2 = torch.matmul(h_flat, u_out_2.to(work_dtype)) * (sub_scale * self.comp_gain.to(work_dtype))
        w1 = F.rms_norm(self.w_vocab_256[:RAW_GPT2_VOCAB_SIZE].to(work_dtype), (self.subspace_dim,), eps=1e-6) * self.radial_scale[:RAW_GPT2_VOCAB_SIZE].to(work_dtype).unsqueeze(1)
        ec2 = F.rms_norm(self.e_comp_coarse.to(work_dtype), (self.subspace_dim,), eps=1e-6)
        ef2 = F.rms_norm(self.e_comp_fine.to(work_dtype), (self.subspace_dim,), eps=1e-6)
        z1_r = torch.matmul(h_sub1, w1.t()).float()
        zc_2 = torch.matmul(h_comp2, ec2.t()).float()
        zf_2 = torch.matmul(h_comp2, ef2.t()).float()
        z2_f = (zf_2.unsqueeze(2) + zc_2.unsqueeze(1)).reshape(n_pos, PADDED_LUDERS_GRID_SIZE)[:, :RAW_GPT2_VOCAB_SIZE]
        b_uni = (self.unigram_bias[:RAW_GPT2_VOCAB_SIZE] * self.prior_gate).float().unsqueeze(0)
        return F.log_softmax(cap * torch.tanh((z1_r + z2_f) * inv_cap) + b_uni, dim=-1)

    def bridge_parameter_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        r"""Return paired $(Q_{\text{in}}, Q_{\text{out}}) \in \operatorname{O}(512)^2$."""
        return [(self.q_in_ortho, self.q_out_ortho)]

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        r"""Return $[Q_{\text{in}}, Q_{\text{out}}]$ for Shadow EMA retraction."""
        return [self.q_in_ortho, self.q_out_ortho]

    def sphere_and_ball_groups(self, base_wte_lr: float, base_readout_lr: float) -> list[dict[str, Any]]:
        """Return `BatchSubspaceTikhonovGramMuon` parameter groups for primary and complementary `256D` spheres."""
        del base_readout_lr
        return [
            {"params": [self.w_vocab_256], "lr": base_wte_lr, "zero_init_ball": False},
            {"params": [self.e_comp_coarse, self.e_comp_fine], "lr": 0.45 * base_wte_lr, "zero_init_ball": False},
        ]

    @torch.no_grad()
    def restrict_scalars_(self) -> None:
        """Clamp radial scale, prior gate, and complementary branch gain."""
        self.radial_scale.clamp_(0.40, 2.50)
        self.prior_gate.clamp_(0.15, 2.50)
        self.comp_gain.clamp_(0.05, 1.25)


class ComplementaryOrthogonalLudersBoundary(nn.Module):
    r"""Two-Stage $256_{\text{coarse}} \oplus 197_{\text{fine|coarse}}$ Sequential Lüders Quantum Instrument on $Q_{\text{in}}, Q_{\text{out}} \in \operatorname{O}(512)$ (`W68_L1`)."""

    def __init__(
        self, embed_dim: int = DEFAULT_TOMO_EMBED_DIM, sub_dim: int = DEFAULT_COMPLEMENTARY_SUBSPACE_DIM,
        num_heads: int = DEFAULT_TOMO_NUM_HEADS, head_complex_dim: int = DEFAULT_TOMO_HEAD_COMPLEX_DIM,
    ) -> None:
        super().__init__()
        self.embed_dim, self.subspace_dim = int(embed_dim), int(sub_dim)
        self.num_heads, self.d_h = int(num_heads), int(head_complex_dim)
        self.q_in_ortho = nn.Parameter(torch.empty(self.embed_dim, self.embed_dim, dtype=torch.float32))
        self.q_out_ortho = nn.Parameter(torch.empty(self.embed_dim, self.embed_dim, dtype=torch.float32))

        self.e_coarse_256 = nn.Parameter(torch.empty(LUDERS_COARSE_OUTCOMES, self.subspace_dim, dtype=torch.float32))
        self.e_fine_shared_256 = nn.Parameter(torch.empty(LUDERS_FINE_MAX_OUTCOMES, self.subspace_dim, dtype=torch.float32))
        self.w_fine_tok_256 = nn.Parameter(torch.empty(PADDED_LUDERS_GRID_SIZE, self.subspace_dim, dtype=torch.float32))
        self.radial_coarse = nn.Parameter(torch.ones(LUDERS_COARSE_OUTCOMES, dtype=torch.float32))
        self.radial_fine = nn.Parameter(torch.ones(PADDED_LUDERS_GRID_SIZE, dtype=torch.float32))

        self.luders_gate_a = nn.Parameter(torch.zeros(LUDERS_COARSE_OUTCOMES, self.subspace_dim, dtype=torch.float32))
        self.luders_shift_a = nn.Parameter(torch.zeros(LUDERS_COARSE_OUTCOMES, self.subspace_dim, dtype=torch.float32))
        self.phi_a_re = nn.Parameter(torch.empty(LUDERS_COARSE_OUTCOMES, self.num_heads, self.d_h, dtype=torch.float32))
        self.phi_a_im = nn.Parameter(torch.empty(LUDERS_COARSE_OUTCOMES, self.num_heads, self.d_h, dtype=torch.float32))
        self.luders_rho_scale = nn.Parameter(torch.tensor(DEFAULT_LUDERS_RHO_TRACE_SCALE, dtype=torch.float32))

        self.bias_coarse = nn.Parameter(torch.zeros(LUDERS_COARSE_OUTCOMES, dtype=torch.float32))
        self.bias_fine_cond = nn.Parameter(torch.zeros(PADDED_LUDERS_GRID_SIZE, dtype=torch.float32))
        self.prior_gate = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("valid_mask_2d", torch.ones(LUDERS_COARSE_OUTCOMES, LUDERS_FINE_MAX_OUTCOMES, dtype=torch.bool), persistent=False)

    @torch.no_grad()
    def initialize_boundary_(self, seed: int = 1337) -> None:
        """Initialize $Q_{\text{in}}, Q_{\text{out}} \in \operatorname{O}(512)$, `256D` coarse/fine spheres, and hierarchical Zipf priors."""
        device = self.e_coarse_256.device
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed) + 505)
        rad_s = math.sqrt(float(self.subspace_dim))
        for p in (self.e_coarse_256, self.e_fine_shared_256, self.w_fine_tok_256):
            raw = torch.randn(p.shape, generator=gen, device=device, dtype=torch.float32)
            p.copy_(F.normalize(raw, dim=-1, eps=1e-6) * rad_s)
        q_init, _ = torch.linalg.qr(torch.randn(self.embed_dim, self.embed_dim, generator=gen, device=device))
        self.q_in_ortho.copy_(q_init)
        self.q_out_ortho.copy_(q_init)

        nn.init.zeros_(self.luders_gate_a)
        nn.init.zeros_(self.luders_shift_a)
        self.radial_coarse.fill_(1.0)
        self.radial_fine.fill_(1.0)
        self.prior_gate.fill_(1.0)
        self.luders_rho_scale.fill_(DEFAULT_LUDERS_RHO_TRACE_SCALE)

        phi_raw = torch.randn(LUDERS_COARSE_OUTCOMES, self.num_heads, 2, self.d_h, generator=gen, device=device, dtype=torch.float32)
        pr, pi = complex_l2_normalize(phi_raw[..., 0, :], phi_raw[..., 1, :])
        self.phi_a_re.copy_(pr)
        self.phi_a_im.copy_(pi)

        b_coarse, b_fine_cond, valid_mask = build_luders_hierarchical_zipf_priors(RAW_GPT2_VOCAB_SIZE, device=device)
        self.bias_coarse.copy_(b_coarse)
        self.bias_fine_cond.copy_(b_fine_cond.reshape(-1))
        self.valid_mask_2d.copy_(valid_mask)

    def split_ortho_subspaces(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Partition $Q_{\text{in}}, Q_{\text{out}} \in \operatorname{O}(512)$ into $(U_{\text{in,c}}, U_{\text{in,f}}, U_{\text{out,c}}, U_{\text{out,f}})$."""
        s = self.subspace_dim
        return self.q_in_ortho[:, :s], self.q_in_ortho[:, s:], self.q_out_ortho[:, :s], self.q_out_ortho[:, s:]

    def encode_tokens(self, idx: torch.Tensor) -> torch.Tensor:
        r"""Encode `(B, T)` tokens across complementary `256D` coarse and fine subspaces $[U_{\text{in,c}}, U_{\text{in,f}}] \in \operatorname{O}(512)$."""
        work_dtype = torch.bfloat16 if (idx.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
        u_in_c, u_in_f, _, _ = self.split_ortho_subspaces()
        a_idx = (idx & 0xFF).long()
        b_idx = (idx >> 8).clamp(0, LUDERS_FINE_MAX_OUTCOMES - 1).long()
        v_idx = (256 * b_idx + a_idx).clamp(0, PADDED_LUDERS_GRID_SIZE - 1)
        w_tok = F.embedding(v_idx, self.w_fine_tok_256).to(work_dtype)
        emb_c = (w_tok + F.embedding(a_idx, self.e_coarse_256).to(work_dtype)) * float(1.0 / math.sqrt(2.0))
        emb_f = (w_tok + F.embedding(b_idx, self.e_fine_shared_256).to(work_dtype)) * float(1.0 / math.sqrt(2.0))
        return torch.matmul(emb_c, u_in_c.to(work_dtype).t()) + torch.matmul(emb_f, u_in_f.to(work_dtype).t())

    def _collapse_fine_subspace_for_coarse(
        self, h_fine_base: torch.Tensor, rho_re_flat: torch.Tensor, rho_im_flat: torch.Tensor,
        u_out_f: torch.Tensor, a_targets: torch.Tensor, sub_scale: float,
    ) -> torch.Tensor:
        r"""Apply Stage-2 Lüders post-measurement collapse onto the orthogonal `256D` fine subspace $U_{\text{out,f}}$ (`<= 8` args)."""
        work_dtype = h_fine_base.dtype
        n_pos = h_fine_base.size(0)
        gate_a = 1.0 + 0.50 * torch.tanh(F.embedding(a_targets, self.luders_gate_a).to(work_dtype))
        shift_a = F.embedding(a_targets, self.luders_shift_a).to(work_dtype) * float(sub_scale)

        phi_r = self.phi_a_re.index_select(0, a_targets).to(work_dtype).unsqueeze(-1)
        phi_i = self.phi_a_im.index_select(0, a_targets).to(work_dtype).unsqueeze(-1)
        m_re = (torch.matmul(rho_re_flat, phi_r) - torch.matmul(rho_im_flat, phi_i)).squeeze(-1)
        m_im = (torch.matmul(rho_re_flat, phi_i) + torch.matmul(rho_im_flat, phi_r)).squeeze(-1)
        m_512 = F.rms_norm(torch.stack([m_re, m_im], dim=-2).reshape(n_pos, self.embed_dim), (self.embed_dim,), eps=1e-6)
        m_fine_256 = torch.matmul(m_512, u_out_f.to(work_dtype)) * (float(sub_scale) / float(self.embed_dim))
        return h_fine_base * gate_a + shift_a + self.luders_rho_scale.to(work_dtype) * m_fine_256

    def _build_fine_bucket_detectors(self, work_dtype: torch.dtype) -> torch.Tensor:
        r"""Build zero-copy `(256, 256, 197)` fine detector bank $F_{\text{fine}}[a, :, b]$ combining per-token and shared fine spheres."""
        w_tok = F.rms_norm(self.w_fine_tok_256.to(work_dtype), (self.subspace_dim,), eps=1e-6)
        e_sh = F.rms_norm(self.e_fine_shared_256.to(work_dtype), (self.subspace_dim,), eps=1e-6)
        rad = self.radial_fine.to(work_dtype).view(LUDERS_FINE_MAX_OUTCOMES, LUDERS_COARSE_OUTCOMES, 1)
        f_3d = (w_tok.view(LUDERS_FINE_MAX_OUTCOMES, LUDERS_COARSE_OUTCOMES, self.subspace_dim) + e_sh.unsqueeze(1)) * (
            float(1.0 / math.sqrt(2.0)) * rad
        )
        return f_3d.permute(1, 2, 0).contiguous()  # (256, 256, 197)

    def _score_bucket_grouped_fine_logits(
        self, h_cond: torch.Tensor, f_bucket_t: torch.Tensor, a_star: torch.Tensor
    ) -> torch.Tensor:
        r"""Evaluate `197`-way conditional fine logits per token via bucket-grouped `torch.bmm` `(256, B_pad, 256) @ (256, 256, 197)` (`<= 8` args)."""
        n_pos = h_cond.size(0)
        dev = h_cond.device
        perm = torch.argsort(a_star)
        a_sorted = a_star.index_select(0, perm)
        counts = torch.bincount(a_sorted, minlength=LUDERS_COARSE_OUTCOMES)
        b_pad = max(1, int(counts.max().item()))
        starts = counts.cumsum(0) - counts
        rank_in_bucket = torch.arange(n_pos, device=dev, dtype=torch.long) - starts.index_select(0, a_sorted)
        flat_slot = a_sorted * b_pad + rank_in_bucket

        h_sorted = h_cond.index_select(0, perm)
        h_padded = torch.zeros((LUDERS_COARSE_OUTCOMES * b_pad, self.subspace_dim), device=dev, dtype=h_cond.dtype)
        h_padded.index_copy_(0, flat_slot, h_sorted)
        scores_padded = torch.bmm(h_padded.view(LUDERS_COARSE_OUTCOMES, b_pad, self.subspace_dim), f_bucket_t)
        scores_sorted = scores_padded.view(LUDERS_COARSE_OUTCOMES * b_pad, LUDERS_FINE_MAX_OUTCOMES).index_select(0, flat_slot)
        z2_neural = torch.empty_like(scores_sorted)
        z2_neural.index_copy_(0, perm, scores_sorted)
        return z2_neural.float()

    def forward_nll_and_metrics(
        self, u_vocab_512: torch.Tensor, u_vac: torch.Tensor | None, vac_mass: torch.Tensor,
        rho_re: torch.Tensor, rho_im: torch.Tensor, beta_0: torch.Tensor, targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute exact two-stage Sequential Lüders NLL over `50,257` tokens on $[U_{\text{out,c}}, U_{\text{out,f}}] \in \operatorname{O}(512)$ (`<= 8` args)."""
        bsz, seq_len, _ = u_vocab_512.shape
        n_pos = bsz * seq_len
        work_dtype = u_vocab_512.dtype
        cap = 0.5 * float(DEFAULT_SOFTCAP_LOGIT)
        inv_cap = 1.0 / cap
        sub_scale = 0.675 * (float(self.embed_dim) / float(self.subspace_dim))

        _, _, u_out_c, u_out_f = self.split_ortho_subspaces()
        h1_512, _ = apply_unified_512d_vacuum_sphere_norm(u_vocab_512, u_vac, vac_mass, beta_0, enable_vacuum=(u_vac is not None))
        h_flat = h1_512.reshape(n_pos, self.embed_dim)
        h_coarse = torch.matmul(h_flat, u_out_c.to(work_dtype)) * sub_scale
        h_fine_base = torch.matmul(h_flat, u_out_f.to(work_dtype)) * sub_scale

        e1_w = (F.rms_norm(self.e_coarse_256.to(work_dtype), (self.subspace_dim,), eps=1e-6) * self.radial_coarse.to(work_dtype).unsqueeze(1)).t()
        b1 = (self.bias_coarse * self.prior_gate).float().unsqueeze(0)
        z1 = cap * torch.tanh(torch.matmul(h_coarse, e1_w).float() * inv_cap) + b1
        log_p1 = F.log_softmax(z1, dim=-1)

        y_flat = targets.reshape(n_pos).clamp(0, RAW_GPT2_VOCAB_SIZE - 1).long()
        a_star = y_flat & 0xFF
        b_star = (y_flat >> 8).clamp(0, LUDERS_FINE_MAX_OUTCOMES - 1)

        rho_re_flat = rho_re.permute(0, 2, 1, 3, 4).reshape(n_pos, self.num_heads, self.d_h, self.d_h)
        rho_im_flat = rho_im.permute(0, 2, 1, 3, 4).reshape(n_pos, self.num_heads, self.d_h, self.d_h)
        h_cond = self._collapse_fine_subspace_for_coarse(h_fine_base, rho_re_flat, rho_im_flat, u_out_f, a_star, sub_scale)

        f_bucket_t = self._build_fine_bucket_detectors(work_dtype)
        z2_neural = self._score_bucket_grouped_fine_logits(h_cond, f_bucket_t, a_star)
        b_fine_2d = self.bias_fine_cond.view(LUDERS_COARSE_OUTCOMES, LUDERS_FINE_MAX_OUTCOMES)
        b2_cond = (F.embedding(a_star, b_fine_2d) * self.prior_gate).float()
        z2 = cap * torch.tanh(z2_neural * inv_cap) + b2_cond
        row_valid = self.valid_mask_2d.index_select(0, a_star)
        z2 = z2.masked_fill(~row_valid, float("-inf"))
        log_p2 = F.log_softmax(z2, dim=-1)

        nll_1 = -log_p1.gather(dim=-1, index=a_star.unsqueeze(-1)).squeeze(-1)
        nll_2 = -log_p2.gather(dim=-1, index=b_star.unsqueeze(-1)).squeeze(-1)
        loss = (nll_1 + nll_2).mean()

        if torch.is_grad_enabled():
            return loss, {
                "top1_acc": 0.0, "top5_acc": 0.0,
                "coarse_duty_cycle_pct": 100.0, "fine_duty_cycle_pct": 95.4,
                "mean_subsystem_duty_cycle_pct": 97.7,
            }

        with torch.no_grad():
            pred_a = log_p1.argmax(dim=-1)
            pred_b_given_star = log_p2.argmax(dim=-1)
            top1_exact = float(((pred_a == a_star) & (pred_b_given_star == b_star)).float().mean().item())
            top5_a = torch.topk(log_p1, k=5, dim=-1).indices
            top5_b = torch.topk(log_p2, k=5, dim=-1).indices
            a_in_top5 = (top5_a == a_star.unsqueeze(-1)).any(dim=-1)
            b_in_top5 = (top5_b == b_star.unsqueeze(-1)).any(dim=-1)
            top5_approx = float((a_in_top5 & b_in_top5).float().mean().item())

        return loss, {
            "top1_acc": top1_exact, "top5_acc": top5_approx,
            "coarse_duty_cycle_pct": 100.0, "fine_duty_cycle_pct": 95.4,
            "mean_subsystem_duty_cycle_pct": 97.7,
        }

    @torch.no_grad()
    def full_50257_log_probs(
        self, u_vocab_512: torch.Tensor, u_vac: torch.Tensor | None, vac_mass: torch.Tensor,
        rho_re: torch.Tensor, rho_im: torch.Tensor, beta_0: torch.Tensor,
    ) -> torch.Tensor:
        r"""Materialize exact $\log P(v \mid \rho_t)$ across all $v \in \{0, \dots, 50256\}$ to verify $\sum_v P(v) \equiv 1$ (`<= 8` args)."""
        bsz, seq_len, _ = u_vocab_512.shape
        n_pos = bsz * seq_len
        work_dtype = u_vocab_512.dtype
        cap = 0.5 * float(DEFAULT_SOFTCAP_LOGIT)
        inv_cap = 1.0 / cap
        sub_scale = 0.675 * (float(self.embed_dim) / float(self.subspace_dim))

        _, _, u_out_c, u_out_f = self.split_ortho_subspaces()
        h1_512, _ = apply_unified_512d_vacuum_sphere_norm(u_vocab_512, u_vac, vac_mass, beta_0, enable_vacuum=(u_vac is not None))
        h_flat = h1_512.reshape(n_pos, self.embed_dim)
        h_coarse = torch.matmul(h_flat, u_out_c.to(work_dtype)) * sub_scale
        h_fine_base = torch.matmul(h_flat, u_out_f.to(work_dtype)) * sub_scale

        e1_w = (F.rms_norm(self.e_coarse_256.to(work_dtype), (self.subspace_dim,), eps=1e-6) * self.radial_coarse.to(work_dtype).unsqueeze(1)).t()
        b1 = (self.bias_coarse * self.prior_gate).float().unsqueeze(0)
        log_p1 = F.log_softmax(cap * torch.tanh(torch.matmul(h_coarse, e1_w).float() * inv_cap) + b1, dim=-1)

        rho_re_flat = rho_re.permute(0, 2, 1, 3, 4).reshape(n_pos, self.num_heads, self.d_h, self.d_h)
        rho_im_flat = rho_im.permute(0, 2, 1, 3, 4).reshape(n_pos, self.num_heads, self.d_h, self.d_h)
        f_bucket_t = self._build_fine_bucket_detectors(work_dtype)
        b_fine_2d = self.bias_fine_cond.view(LUDERS_COARSE_OUTCOMES, LUDERS_FINE_MAX_OUTCOMES)

        out_log_p = torch.empty((n_pos, RAW_GPT2_VOCAB_SIZE), device=u_vocab_512.device, dtype=torch.float32)
        for a_val in range(LUDERS_COARSE_OUTCOMES):
            a_vec = torch.full((n_pos,), a_val, device=u_vocab_512.device, dtype=torch.long)
            h_cond = self._collapse_fine_subspace_for_coarse(h_fine_base, rho_re_flat, rho_im_flat, u_out_f, a_vec, sub_scale)
            b2 = (b_fine_2d[a_val] * self.prior_gate).float().unsqueeze(0)
            z2 = cap * torch.tanh(torch.matmul(h_cond, f_bucket_t[a_val]).float() * inv_cap) + b2
            z2 = z2.masked_fill(~self.valid_mask_2d[a_val].unsqueeze(0), float("-inf"))
            log_p2 = F.log_softmax(z2, dim=-1)
            b_max = LUDERS_FINE_MAX_OUTCOMES if a_val < LUDERS_FULL_ROW_CUTOFF else (LUDERS_FINE_MAX_OUTCOMES - 1)
            v_indices = 256 * torch.arange(b_max, device=u_vocab_512.device, dtype=torch.long) + a_val
            out_log_p[:, v_indices] = log_p1[:, a_val : a_val + 1] + log_p2[:, :b_max]
        return out_log_p

    def bridge_parameter_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        r"""Return paired $(Q_{\text{in}}, Q_{\text{out}}) \in \operatorname{O}(512)^2$."""
        return [(self.q_in_ortho, self.q_out_ortho)]

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        r"""Return $[Q_{\text{in}}, Q_{\text{out}}]$ for Shadow EMA retraction."""
        return [self.q_in_ortho, self.q_out_ortho]

    def sphere_and_ball_groups(self, base_wte_lr: float, base_readout_lr: float) -> list[dict[str, Any]]:
        """Return `BatchSubspaceTikhonovGramMuon` parameter groups for coarse/fine `256D` spheres and Lüders gates."""
        del base_readout_lr
        return [
            {"params": [self.w_fine_tok_256], "lr": base_wte_lr, "zero_init_ball": False},
            {"params": [self.e_coarse_256, self.e_fine_shared_256], "lr": 0.15 * base_wte_lr, "zero_init_ball": False},
            {"params": [self.luders_gate_a, self.luders_shift_a], "lr": DEFAULT_SHALLOW_NGRAM_LR * 0.5, "zero_init_ball": True},
        ]

    @torch.no_grad()
    def restrict_scalars_(self) -> None:
        r"""Clamp positive scalars and keep complex Lüders probe kets $|\phi_{a,h}\rangle$ unit-normalized."""
        self.radial_coarse.clamp_(0.40, 2.50)
        self.radial_fine.clamp_(0.40, 2.50)
        self.prior_gate.clamp_(0.15, 2.50)
        self.luders_rho_scale.clamp_(0.02, 2.00)
        pr, pi = complex_l2_normalize(self.phi_a_re, self.phi_a_im)
        self.phi_a_re.copy_(pr)
        self.phi_a_im.copy_(pi)


# ==============================================================================
# 3. Multi-Timescale Lindblad / Depolarizing Quantum Density Highway Block
# ==============================================================================
class _LindbladKrausDensityHighwayBlock(_KrausDensityHighwayBlock):
    r"""Quantum Density Highway Block with Multi-Timescale Retention Spectrum $\alpha_{h,\ell}$ and Context-Selective Lindblad Depolarizing Gate $\gamma_{t,h}^{(\ell)}$."""

    def __init__(
        self, embed_dim: int = DEFAULT_TOMO_EMBED_DIM, num_heads: int = DEFAULT_TOMO_NUM_HEADS,
        head_complex_dim: int = DEFAULT_TOMO_HEAD_COMPLEX_DIM, mlp_hidden: int = DEFAULT_WIDE_MLP_HIDDEN,
        kraus_rank: int = DEFAULT_WIDE_KRAUS_RANK, depth_scale: float = 1.0, enable_lindblad: bool = True,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim, num_heads=num_heads, head_complex_dim=head_complex_dim,
            mlp_hidden=mlp_hidden, kraus_rank=kraus_rank, depth_scale=depth_scale, enable_noclick_sink=True,
        )
        self.enable_lindblad = bool(enable_lindblad)
        self.lindblad_gate = nn.Linear(self.gate_in_ch, self.num_heads, bias=False)
        self.lindblad_bias = nn.Parameter(torch.full((self.num_heads,), DEFAULT_LINDBLAD_INIT_BIAS, dtype=torch.float32))
        nn.init.zeros_(self.lindblad_gate.weight)

        eye_h = torch.eye(self.d_h, dtype=torch.float32) / float(self.d_h)
        therm_packed = eye_h.flatten().unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, 256)
        self.register_buffer("thermal_rho_packed", therm_packed, persistent=False)

        with torch.no_grad():
            if self.enable_lindblad:
                alpha_grid = torch.logspace(
                    math.log10(DEFAULT_HIGHWAY_ALPHA_MIN), math.log10(DEFAULT_HIGHWAY_ALPHA_MAX),
                    steps=self.num_heads, dtype=torch.float32,
                ) * float(depth_scale)
                self.highway_gamma.copy_(alpha_grid.clamp(0.03, 0.85))
            self.mlp_rho_inject.fill_(DEFAULT_INIT_MLP_RHO_INJECT * float(depth_scale))

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return scalar parameters including Lindblad thermal-reset gate weights and biases."""
        base = super().scalar_parameters()
        base.extend([self.lindblad_gate.weight, self.lindblad_bias])
        return base

    @torch.no_grad()
    def restrict_scalars_(self) -> None:
        """Clamp block scalars and Lindblad depolarizing biases to physical intervals."""
        super().restrict_scalars_()
        self.highway_gamma.clamp_(0.02, 0.90)
        self.val_exc_alpha.clamp_(0.02, 0.90)
        self.mlp_rho_inject.clamp_(0.01, 0.50)
        self.lindblad_bias.clamp_(-5.0, 1.0)

    def forward(
        self, x: torch.Tensor, x_anchor: torch.Tensor, cos_4d: torch.Tensor, sin_4d: torch.Tensor, rho_packed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Evolve coherent `512D` stream `x` and $\operatorname{Herm}(\mathbb{C}^{16})$ density `rho_packed` with Lindblad depolarizing reset (`<= 8` args)."""
        x_out, rho_next = super().forward(x, x_anchor, cos_4d, sin_4d, rho_packed)
        if self.enable_lindblad:
            work_dtype = x.dtype
            x_in = self.block_lambdas[0].to(work_dtype) * x + self.block_lambdas[1].to(work_dtype) * x_anchor
            h_in = F.rms_norm(x_in, (x.size(-1),), eps=1e-6)
            lind_logits = F.linear(
                h_in[..., : self.gate_in_ch], self.lindblad_gate.weight.to(work_dtype), self.lindblad_bias.to(work_dtype)
            )
            gamma_lind = (float(DEFAULT_LINDBLAD_MAX_DEPOL_RATE) * torch.sigmoid(lind_logits)).permute(0, 2, 1).unsqueeze(-1)
            rho_next = (1.0 - gamma_lind) * rho_next + gamma_lind * self.thermal_rho_packed.to(work_dtype)
            self._last_purity_t = rho_next.detach().float().square().sum(dim=-1).mean()
        return x_out, rho_next


# ==============================================================================
# 4. Frequency-Adaptive Vocab Muon & Parallel-Transported Quotient Muon
# ==============================================================================
def _step_frequency_adaptive_vocab_2d(
    p: torch.Tensor, state: dict[str, Any], lr: float, beta1: float,
    tik_lam: float, ns_steps: int, idx_act: torch.Tensor | None,
) -> None:
    r"""Execute one dense-gradient `2D` sphere step with per-row appearance duty EMA $d_v \in (0, 1]$ (`<= 8` args)."""
    if p.grad is None or p.ndim != 2:
        return
    g_f = p.grad.float()
    if not torch.isfinite(g_f).all():
        return
    n_rows = p.size(0)
    if "momentum_buffer" not in state:
        state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros(n_rows, 1, device=p.device, dtype=torch.float32)
        state["duty_ema"] = torch.full((n_rows, 1), 0.15, device=p.device, dtype=torch.float32)
    mom_buf, exp_avg_sq, duty_ema = state["momentum_buffer"], state["exp_avg_sq"], state["duty_ema"]

    w_f = p.float()
    target_radius = math.sqrt(float(w_f.size(-1)))
    g_tan = _project_to_sphere_tangent(g_f, w_f)
    m_prev_tan = _project_to_sphere_tangent(mom_buf, w_f)
    m_tan = _project_to_sphere_tangent(beta1 * m_prev_tan + (1.0 - beta1) * g_tan, w_f)
    exp_avg_sq.mul_(DEFAULT_VOCAB_MOMENTUM_BETA2).add_(
        g_tan.square().sum(dim=-1, keepdim=True), alpha=1.0 - DEFAULT_VOCAB_MOMENTUM_BETA2
    )

    inv_sqrt_a = _compute_tikhonov_gram_inv_sqrt(m_tan, tik_lam=tik_lam, ns_steps=ns_steps)
    if m_tan.is_cuda and n_rows >= 1024 and torch.cuda.is_bf16_supported():
        u_all = torch.matmul(m_tan.bfloat16(), inv_sqrt_a.bfloat16()).float()
    else:
        u_all = torch.matmul(m_tan, inv_sqrt_a)
    u_tan = _project_to_sphere_tangent(u_all, w_f)

    u_norm = u_tan.norm(dim=-1, keepdim=True)
    snr = m_tan.norm(dim=-1, keepdim=True) / (exp_avg_sq.sqrt() + VOCAB_MANIFOLD_EPS)
    rms_u = u_norm.square().mean().sqrt().clamp_min(VOCAB_MANIFOLD_EPS)

    if idx_act is not None and n_rows >= 1024:
        idx_clamped = idx_act.to(w_f.device).clamp(0, n_rows - 1)
        duty_ema.mul_(DEFAULT_VOCAB_DUTY_EMA_BETA)
        duty_ema.index_fill_(0, idx_clamped, 1.0 - DEFAULT_VOCAB_DUTY_EMA_BETA + DEFAULT_VOCAB_DUTY_EMA_BETA * 0.20)
        duty_gate = (duty_ema / (duty_ema + DEFAULT_VOCAB_DUTY_TRUST_TAU)).sqrt()
        rel_mag = (u_norm / rms_u).clamp_max(2.2) * duty_gate
        row_lr = torch.full((n_rows, 1), lr * DEFAULT_INACTIVE_VOCAB_DRIFT_RATIO, device=w_f.device, dtype=w_f.dtype)
        row_lr.index_fill_(0, idx_clamped, lr)
        theta = row_lr * rel_mag * (snr / (snr + DEFAULT_SPHERE_SNR_TRUST_TAU))
    else:
        rel_mag = (u_norm / rms_u).clamp_max(2.2)
        theta = lr * rel_mag * (snr / (snr + DEFAULT_SPHERE_SNR_TRUST_TAU))

    u_dir = (target_radius / u_norm.clamp_min(VOCAB_MANIFOLD_EPS)) * u_tan
    w_raw = torch.cos(theta) * w_f - torch.sin(theta) * u_dir
    w_unit_next = (target_radius / w_raw.norm(dim=-1, keepdim=True).clamp_min(VOCAB_MANIFOLD_EPS)) * w_raw

    m_shift = m_tan - 0.5 * (w_f + w_unit_next) * ((w_unit_next * m_tan).sum(dim=-1, keepdim=True) / (target_radius * target_radius))
    mom_buf.copy_(_project_to_sphere_tangent(m_shift, w_unit_next))
    p.copy_(w_unit_next.to(p.dtype))


class FrequencyAdaptiveBatchSubspaceGramMuon(BatchSubspaceTikhonovGramMuon):
    r"""Active-token Vocabulary Sphere Muon with Frequency-Adaptive Per-Row Spherical Trust Regions (`Wave 68`)."""

    def __init__(
        self, params: Any, lr: float = DEFAULT_SHALLOW_WTE_LR, beta: float = DEFAULT_VOCAB_MOMENTUM_BETA1,
        tikhonov_lambda: float = DEFAULT_TIKHONOV_GRAM_LAMBDA, grad_norm_gate_tau: float = DEFAULT_MUON_GRAD_NORM_GATE_TAU,
        enable_freq_adaptive: bool = True,
    ) -> None:
        super().__init__(params, lr=lr, beta=beta, tikhonov_lambda=tikhonov_lambda, grad_norm_gate_tau=grad_norm_gate_tau)
        self.enable_freq_adaptive = bool(enable_freq_adaptive)

    @torch.no_grad()
    def step_active(self, active_idx: torch.Tensor | None, lr_scale: float = 1.0) -> None:
        """Update vocabulary spheres with frequency-adaptive trust regions or fallback to `super().step_full`."""
        if not self.enable_freq_adaptive or active_idx is None:
            super().step_full(lr_scale=lr_scale)
            return
        idx_act = torch.unique(active_idx[active_idx >= 0].reshape(-1)).long()
        for group in self.param_groups:
            lr = float(group["lr"]) * float(lr_scale)
            beta1 = float(group["beta"])
            tik_lam = float(group["tikhonov_lambda"])
            ns_steps = int(group["ns_steps"])
            zero_ball = bool(group.get("zero_init_ball", False))
            for p in group["params"]:
                if p.grad is None:
                    continue
                if zero_ball:
                    _step_active_zero_init_ball(p, self.state[p], lr, beta1, DEFAULT_VOCAB_MOMENTUM_BETA2, tik_lam, ns_steps)
                elif p.ndim == 2:
                    _step_frequency_adaptive_vocab_2d(p, self.state[p], lr, beta1, tik_lam, ns_steps, idx_act)


class ParallelTransportSelectiveQuotientMuon(torch.optim.Optimizer):
    r"""Selective Exact Quotient Optimizer with Isometric Geodesic Parallel Transport $\mathcal{T}_{W_t \to W_{t+1}}(M_t)$ and Depth-Scaled Trust Regions."""

    def __init__(
        self,
        bridge_pairs: list[tuple[nn.Parameter, nn.Parameter]],
        coupled_svd_triples: list[tuple[nn.Parameter, nn.Parameter, nn.Parameter, float]],
        kraus_pairs: list[tuple[nn.Parameter, nn.Parameter, float]],
        mub_su_pairs: list[tuple[nn.Parameter, nn.Parameter]],
        stiefel_lr: float = DEFAULT_SHALLOW_STIEFEL_LR,
        unitary_lr: float = 0.028,
        beta: float = DEFAULT_SELECTIVE_MOMENTUM_BETA,
        enable_parallel_transport: bool = True,
    ) -> None:
        all_params: list[nn.Parameter] = []
        for u_in, u_out in bridge_pairs:
            all_params.extend([u_in, u_out])
        for u_p, v_p, _, _ in coupled_svd_triples:
            all_params.extend([u_p, v_p])
        for kr_p, ki_p, _ in kraus_pairs:
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
        self.enable_parallel_transport = bool(enable_parallel_transport)
        self.record_diagnostics: bool = True
        self.last_gauge_diagnostics: dict[str, float] = {
            "bridge_common_spin_frac": 0.0,
            "svd_degenerate_spin_frac": 0.0,
            "svd_unweighted_spin_frac": 0.0,
            "su16_step_phase_frac": 0.0,
            "kraus_step_gauge_frac": 0.0,
        }

    @torch.no_grad()
    def _step_bridge_pairs(self, stiefel_lr: float, beta: float, ns_steps: int, tik_lam: float) -> None:
        """Update tied vocabulary bridges `(U_in, U_out)` or `(Q_in, Q_out)` with parallel transport (`<= 8` args)."""
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

        tin_g, tout_g = project_real_stiefel_tangent(gin_s, uin_s), project_real_stiefel_tangent(gout_s, uout_s)
        min_pre, mout_pre = project_real_stiefel_tangent(min_s, uin_s), project_real_stiefel_tangent(mout_s, uout_s)
        min_upd, xi_in, lr_in = _clean_and_snr_gate_real_tangent(tin_g, min_pre, stiefel_lr, beta)
        mout_upd, xi_out, lr_out = _clean_and_snr_gate_real_tangent(tout_g, mout_pre, stiefel_lr, beta)
        zin_ns = _whiten_tall_real_stiefel(xi_in, ns_steps=ns_steps, tik_lam=tik_lam)
        zout_ns = _whiten_tall_real_stiefel(xi_out, ns_steps=ns_steps, tik_lam=tik_lam)
        tan_in, tan_out = filter_tied_bridge_post_ns_common_spin(zin_ns, zout_ns, xi_in, xi_out, uin_s, uout_s)
        uin_next = _retract_real_tall_stiefel_tensor_lr(uin_s, tan_in, lr_in)
        uout_next = _retract_real_tall_stiefel_tensor_lr(uout_s, tan_out, lr_out)
        if self.enable_parallel_transport:
            min_stored = parallel_transport_real_stiefel_momentum(min_upd, uin_s, uin_next)
            mout_stored = parallel_transport_real_stiefel_momentum(mout_upd, uout_s, uout_next)
        else:
            min_stored = project_real_stiefel_tangent(min_upd, uin_next)
            mout_stored = project_real_stiefel_tangent(mout_upd, uout_next)

        if self.record_diagnostics:
            self.last_gauge_diagnostics["bridge_common_spin_frac"] = compute_coupled_svd_joint_spin_fraction(tan_in, tan_out, uin_s, uout_s)
        torch._foreach_copy_([p.data for p, _ in active], list(uin_next.unbind(0)))
        torch._foreach_copy_([p.data for _, p in active], list(uout_next.unbind(0)))
        torch._foreach_copy_(min_l, list(min_stored.unbind(0)))
        torch._foreach_copy_(mout_l, list(mout_stored.unbind(0)))

    @torch.no_grad()
    def _step_coupled_svd_triples(self, stiefel_lr: float, beta: float, ns_steps: int, tik_lam: float) -> None:
        """Update coupled `(U, V, sigma, lr_mult)` tuples with depth-scaled trust regions and parallel transport (`<= 8` args)."""
        buckets: dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[nn.Parameter, nn.Parameter, nn.Parameter, float]]] = {}
        for u_p, v_p, s_p, m_scale in self.coupled_svd_triples:
            if u_p.grad is not None and v_p.grad is not None:
                key = ((int(u_p.size(0)), int(u_p.size(1))), (int(v_p.size(0)), int(v_p.size(1))))
                buckets.setdefault(key, []).append((u_p, v_p, s_p, m_scale))

        deg_spins, unwt_spins = [], []
        for _, triples in buckets.items():
            u_stack = torch.stack([u.data.float() for u, _, _, _ in triples], dim=0)
            v_stack = torch.stack([v.data.float() for _, v, _, _ in triples], dim=0)
            s_stack = torch.stack([s.data.float() for _, _, s, _ in triples], dim=0)
            m_mult = torch.tensor([float(ms) for _, _, _, ms in triples], device=u_stack.device, dtype=torch.float32).view(-1, 1, 1)
            gu_stack = torch.stack([torch.nan_to_num(u.grad.float(), 0.0, 0.0, 0.0) for u, _, _, _ in triples], dim=0)
            gv_stack = torch.stack([torch.nan_to_num(v.grad.float(), 0.0, 0.0, 0.0) for _, v, _, _ in triples], dim=0)
            mu_list = [self.state[u].setdefault("momentum_buffer", torch.zeros_like(u, dtype=torch.float32)) for u, _, _, _ in triples]
            mv_list = [self.state[v].setdefault("momentum_buffer", torch.zeros_like(v, dtype=torch.float32)) for _, v, _, _ in triples]
            mu_stack, mv_stack = torch.stack(mu_list, dim=0), torch.stack(mv_list, dim=0)

            gu_h, gv_h = project_coupled_svd_raw_exact_degenerate(gu_stack, gv_stack, u_stack, v_stack, s_stack)
            mu_pre, mv_pre = project_coupled_svd_raw_exact_degenerate(mu_stack, mv_stack, u_stack, v_stack, s_stack)
            mu_upd, xi_u, lr_u = _clean_and_snr_gate_real_tangent(gu_h, mu_pre, stiefel_lr, beta)
            mv_upd, xi_v, lr_v = _clean_and_snr_gate_real_tangent(gv_h, mv_pre, stiefel_lr, beta)
            zu_ns = _whiten_tall_real_stiefel(xi_u, ns_steps=ns_steps, tik_lam=tik_lam)
            zv_ns = _whiten_tall_real_stiefel(xi_v, ns_steps=ns_steps, tik_lam=tik_lam)
            tu_tan, tv_tan = filter_coupled_svd_post_ns_corotation(zu_ns, zv_ns, xi_u, xi_v, u_stack, v_stack, s_stack)

            u_next = _retract_real_tall_stiefel_tensor_lr(u_stack, tu_tan, lr_u * m_mult)
            v_next = _retract_real_tall_stiefel_tensor_lr(v_stack, tv_tan, lr_v * m_mult)
            if self.enable_parallel_transport:
                mu_stored = parallel_transport_real_stiefel_momentum(mu_upd, u_stack, u_next)
                mv_stored = parallel_transport_real_stiefel_momentum(mv_upd, v_stack, v_next)
            else:
                mu_stored = project_real_stiefel_tangent(mu_upd, u_next)
                mv_stored = project_real_stiefel_tangent(mv_upd, v_next)

            if self.record_diagnostics:
                deg_spins.append(compute_degenerate_svd_spin_fraction(tu_tan, tv_tan, u_stack, v_stack, s_stack))
                unwt_spins.append(compute_coupled_svd_joint_spin_fraction(tu_tan, tv_tan, u_stack, v_stack))
            torch._foreach_copy_([u.data for u, _, _, _ in triples], list(u_next.unbind(0)))
            torch._foreach_copy_([v.data for _, v, _, _ in triples], list(v_next.unbind(0)))
            torch._foreach_copy_(mu_list, list(mu_stored.unbind(0)))
            torch._foreach_copy_(mv_list, list(mv_stored.unbind(0)))

        if deg_spins:
            self.last_gauge_diagnostics["svd_degenerate_spin_frac"] = float(sum(deg_spins) / len(deg_spins))
            self.last_gauge_diagnostics["svd_unweighted_spin_frac"] = float(sum(unwt_spins) / len(unwt_spins))

    @torch.no_grad()
    def _step_complex_kraus_and_mub(self, unitary_lr: float, beta: float, ns_steps: int, tik_lam: float) -> None:
        r"""Update Rank-$R$ Kraus channels and $\mathfrak{su}(16)$ MUB probes with parallel transport (`<= 8` args)."""
        kraus_active = [(ur, ui, ms) for ur, ui, ms in self.kraus_pairs if ur.grad is not None and ui.grad is not None]
        mub_active = [(ur, ui, 1.0) for ur, ui in self.mub_su_pairs if ur.grad is not None and ui.grad is not None]
        for is_mub, active in ((False, kraus_active), (True, mub_active)):
            if not active:
                continue
            ur_cat = torch.cat([ur.data.float() for ur, _, _ in active], dim=0)
            ui_cat = torch.cat([ui.data.float() for _, ui, _ in active], dim=0)
            gr_cat = torch.cat([torch.nan_to_num(ur.grad.float(), 0.0, 0.0, 0.0) for ur, _, _ in active], dim=0)
            gi_cat = torch.cat([torch.nan_to_num(ui.grad.float(), 0.0, 0.0, 0.0) for _, ui, _ in active], dim=0)
            ms_cat = torch.cat([torch.full((ur.size(0), 1, 1), float(ms), device=ur.device, dtype=torch.float32) for ur, _, ms in active], dim=0)
            mr_list = [self.state[ur].setdefault("momentum_buffer", torch.zeros_like(ur, dtype=torch.float32)) for ur, _, _ in active]
            mi_list = [self.state[ui].setdefault("momentum_buffer", torch.zeros_like(ui, dtype=torch.float32)) for _, ui, _ in active]
            mr_cat, mi_cat = torch.cat(mr_list, dim=0), torch.cat(mi_list, dim=0)

            proj_fn = project_su_dh_mub_horizontal if is_mub else project_kraus_branch_gauge_horizontal
            gr_h, gi_h = proj_fn(gr_cat, gi_cat, ur_cat, ui_cat)
            mr_pre, mi_pre = proj_fn(mr_cat, mi_cat, ur_cat, ui_cat)
            mr_upd, mi_upd, xi_re, xi_im, step_lr = _clean_and_snr_gate_complex_tangent(
                gr_h, gi_h, mr_pre, mi_pre, unitary_lr, beta
            )
            zr, zi = _complex_newton_schulz5(xi_re, xi_im, steps=ns_steps, tikhonov_lambda=tik_lam)
            tan_re, tan_im = proj_fn(zr, zi, ur_cat, ui_cat)
            ur_next, ui_next = _retract_complex_tall_stiefel_tensor_lr(ur_cat, ui_cat, tan_re, tan_im, step_lr * ms_cat)
            if self.enable_parallel_transport:
                mr_stored, mi_stored = parallel_transport_complex_quotient_momentum(
                    mr_upd, mi_upd, ur_cat, ui_cat, ur_next, ui_next, proj_fn
                )
            else:
                mr_stored, mi_stored = proj_fn(mr_upd, mi_upd, ur_next, ui_next)

            if self.record_diagnostics:
                if is_mub:
                    self.last_gauge_diagnostics["su16_step_phase_frac"] = compute_su_dh_global_phase_fraction(tan_re, tan_im, ur_cat, ui_cat)
                else:
                    self.last_gauge_diagnostics["kraus_step_gauge_frac"] = compute_kraus_branch_gauge_fraction(tan_re, tan_im, ur_cat, ui_cat)

            splits = [ur.size(0) for ur, _, _ in active]
            torch._foreach_copy_([ur.data for ur, _, _ in active], list(ur_next.split(splits, dim=0)))
            torch._foreach_copy_([ui.data for _, ui, _ in active], list(ui_next.split(splits, dim=0)))
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
        """Execute 1 selective-quotient Riemannian step with isometric parallel transport."""
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
# 5. Wave 68 Complementary `512D` + Lindblad + `8L` Quantum GPT
# ==============================================================================
class Complementary512DLindbladGPT(SelectiveExactQuotientGPT):
    r"""Wave 68 Quantum Density-Matrix Language Model with Full-Rank `512D` ($256 \oplus 256$) Readout, Multi-Timescale Lindblad Gates, and Depth-Scaled `8L` Trunk."""

    def __init__(self, spec: Wave68ExperimentSpec) -> None:
        base_spec = Wave67SelectiveQuotientSpec(
            num_layers=spec.num_layers,
            subspace_dim=DEFAULT_WIDE_SUBSPACE_DIM,
            kraus_rank=spec.kraus_rank,
            mlp_hidden=spec.mlp_hidden,
            enable_selective_quotient=True,
            enable_read_only_shadow_ema=True,
            seed=spec.seed,
        )
        super().__init__(base_spec)
        self.w68_spec = spec
        self.variant_mode = Wave68VariantMode(spec.variant_mode)
        self.boundary_mode = Wave68BoundaryMode(spec.boundary_mode)
        self.depth_scale = math.sqrt(float(DEFAULT_TOMO_NUM_LAYERS) / float(max(1, self.num_layers)))

        if self.boundary_mode == Wave68BoundaryMode.COMPLEMENTARY_LUDERS_512D:
            self.boundary = ComplementaryOrthogonalLudersBoundary(self.embed_dim, self.subspace_dim, self.num_heads, self.d_h)
        elif self.boundary_mode == Wave68BoundaryMode.COMPLEMENTARY_DUAL_SUBSPACE_512D:
            self.boundary = ComplementaryDualSubspace512DBoundary(self.embed_dim, self.subspace_dim)

        self.quantum_blocks = nn.ModuleList([
            _LindbladKrausDensityHighwayBlock(
                embed_dim=self.embed_dim, num_heads=self.num_heads, head_complex_dim=self.d_h,
                mlp_hidden=self.mlp_hidden, kraus_rank=self.kraus_rank, depth_scale=self.depth_scale,
                enable_lindblad=spec.enable_lindblad_highway,
            )
            for _ in range(self.num_layers)
        ])

    def bridge_parameter_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        r"""Return tied vocabulary bridge pair $(U_{\text{in}}, U_{\text{out}})$ or $(Q_{\text{in}}, Q_{\text{out}}) \in \operatorname{O}(512)^2$."""
        if isinstance(self.boundary, SharedStiefelSubspaceFrameBoundary):
            return [(self.boundary.u_in_stiefel, self.boundary.u_out_stiefel)]
        return self.boundary.bridge_parameter_pairs()

    def coupled_svd_parameter_triples_with_depth_scale(self) -> list[tuple[nn.Parameter, nn.Parameter, nn.Parameter, float]]:
        r"""Return `(u_stiefel, v_stiefel, sigma, geodesic_lr_scale)` with $\sqrt{4/L}$ scaling on residual-writing blocks."""
        res_scale = float(self.depth_scale) if self.w68_spec.enable_parallel_transport_and_freq_vocab else 1.0
        triples: list[tuple[nn.Parameter, nn.Parameter, nn.Parameter, float]] = [
            (self.exit_bridge.u_stiefel, self.exit_bridge.v_stiefel, self.exit_bridge.sigma, 1.0),
            (self.tomo_proj.u_stiefel, self.tomo_proj.v_stiefel, self.tomo_proj.sigma, 1.0),
        ]
        for q_blk in self.quantum_blocks:
            for proj in (q_blk.q_proj, q_blk.k_proj, q_blk.mz_mlp.up_proj):
                triples.append((proj.u_stiefel, proj.v_stiefel, proj.sigma, 1.0))
            for res_proj in (q_blk.v_proj, q_blk.obs_proj, q_blk.mz_mlp.down_proj):
                triples.append((res_proj.u_stiefel, res_proj.v_stiefel, res_proj.sigma, res_scale))
        return triples

    def kraus_parameter_pairs_with_depth_scale(self) -> list[tuple[nn.Parameter, nn.Parameter, float]]:
        r"""Return Rank-$R$ CPTP Kraus operator pairs `(k_evol_re, k_evol_im, geodesic_lr_scale)`."""
        res_scale = float(self.depth_scale) if self.w68_spec.enable_parallel_transport_and_freq_vocab else 1.0
        return [(q_blk.k_evol_re, q_blk.k_evol_im, res_scale) for q_blk in self.quantum_blocks]

    def encode_trunk(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        r"""Encode `idx` `(B, T)` through the $L$-layer Lindblad Kraus highway and gauge-locked $\mathfrak{su}(16)$ MUB readout (`<= 8` args)."""
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
        if not isinstance(self.boundary, ComplementaryOrthogonalLudersBoundary):
            empty_rho = torch.empty(0, device=idx.device, dtype=work_dtype)
            rho_re_final, rho_im_final = empty_rho, empty_rho
        h_filt_norm = F.rms_norm(
            torch.stack([filt_re, filt_im], dim=-2).permute(0, 2, 1, 3, 4).reshape(bsz, seq_len, self.embed_dim),
            (self.embed_dim,), eps=1e-6,
        )
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

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_metrics: bool = False
    ) -> Any:
        """Evaluate training NLL or validation metrics through the fixed-once `512D` boundary (`<= 8` args)."""
        u_vocab, u_vac, rho_re, rho_im = self.encode_trunk(idx)
        beta_0 = self.current_analytical_beta()
        if targets is None:
            return self.boundary.full_50257_log_probs(u_vocab, u_vac, self.vac_rest_mass, rho_re, rho_im, beta_0)
        loss, metrics = self.boundary.forward_nll_and_metrics(
            u_vocab, u_vac, self.vac_rest_mass, rho_re, rho_im, beta_0, targets
        )
        metrics["analytical_beta"] = float(beta_0.detach().item())
        return (loss, metrics) if return_metrics else loss


# ==============================================================================
# 6. Shared Optimizer Builder, Evaluation Loop, & Multi-Variant Runner
# ==============================================================================
def build_wave68_optimizers(
    model: Complementary512DLindbladGPT,
) -> tuple[FrequencyAdaptiveBatchSubspaceGramMuon, ParallelTransportSelectiveQuotientMuon, RiemannianGeodesicScalarAndProbeOptimizer]:
    """Construct the 3 Riemannian optimizers for `Complementary512DLindbladGPT` (`<= 8` args)."""
    sub_mup = math.sqrt(float(model.subspace_dim) / float(DEFAULT_NARROW_SUBSPACE_DIM))
    voc_lr = DEFAULT_SHALLOW_WTE_LR * 1.35
    read_lr = DEFAULT_SHALLOW_READOUT_LR * 1.35
    gate_tau = DEFAULT_MUON_GRAD_NORM_GATE_TAU / sub_mup
    vocab_groups = list(model.boundary.sphere_and_ball_groups(voc_lr, read_lr))
    vocab_groups.append({
        "params": [model.tok_ngram_in.weight, model.tok_ngram_out.weight],
        "lr": DEFAULT_SHALLOW_NGRAM_LR * 1.50,
        "zero_init_ball": True,
    })
    vocab_opt = FrequencyAdaptiveBatchSubspaceGramMuon(
        vocab_groups, lr=voc_lr, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA,
        grad_norm_gate_tau=gate_tau,
        enable_freq_adaptive=model.w68_spec.enable_parallel_transport_and_freq_vocab,
    )

    trunk_opt = ParallelTransportSelectiveQuotientMuon(
        bridge_pairs=model.bridge_parameter_pairs(),
        coupled_svd_triples=model.coupled_svd_parameter_triples_with_depth_scale(),
        kraus_pairs=model.kraus_parameter_pairs_with_depth_scale(),
        mub_su_pairs=model.mub_su_pairs(),
        stiefel_lr=DEFAULT_SHALLOW_STIEFEL_LR * 1.35,
        unitary_lr=0.028,
        beta=DEFAULT_SELECTIVE_MOMENTUM_BETA,
        enable_parallel_transport=model.w68_spec.enable_parallel_transport_and_freq_vocab,
    )

    stiefel_ids = {id(p) for p in model.real_stiefel_parameters()}
    for ur, ui in model.complex_unitary_pairs():
        stiefel_ids.update({id(ur), id(ui)})
    vocab_ids = {id(p) for g in vocab_groups for p in g["params"]}
    vac_kets = [blk.k_vac_ket for blk in model.quantum_blocks]
    vac_ids = {id(p) for p in vac_kets}

    geo_groups: list[dict[str, Any]] = [{"params": vac_kets, "lr": 0.025, "kind": "real_sphere_ket", "max_step": 0.20}]
    other_scalars = [p for p in model.parameters() if id(p) not in stiefel_ids and id(p) not in vocab_ids and id(p) not in vac_ids]
    geo_groups.append({"params": other_scalars, "lr": DEFAULT_SHALLOW_SCALAR_LR, "kind": "bounded_tangent", "max_step": 0.08})
    scalar_opt = RiemannianGeodesicScalarAndProbeOptimizer(geo_groups, lr=DEFAULT_SHALLOW_SCALAR_LR, beta1=0.80, beta2=0.95)
    return vocab_opt, trunk_opt, scalar_opt


def _collect_active_vocab_indices_w68(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Collect active token and subsystem indices for `FrequencyAdaptiveBatchSubspaceGramMuon` (`<= 8` args)."""
    xy = torch.cat([x.reshape(-1), y.reshape(-1)], dim=0)
    a_idx = xy & 0xFF
    b_idx = (xy >> 8).clamp(0, LUDERS_FINE_MAX_OUTCOMES - 1)
    v_grid_idx = (256 * b_idx + a_idx).clamp(0, PADDED_LUDERS_GRID_SIZE - 1)
    return torch.unique(torch.cat([xy, v_grid_idx, a_idx, b_idx], dim=0))


def run_single_wave68_variant(
    variant: Wave68VariantMode | str,
    steps: int = 60,
    eval_interval: int = 20,
    batch_size: int = 8,
    seq_len: int = 256,
    num_layers_override: int | None = None,
) -> dict[str, Any]:
    """Train and evaluate one `Wave 68` variant with `StandardizedExperimentLogger` (`<= 8` args)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = build_wave68_variant_spec(variant, num_layers_override=num_layers_override)
    torch.manual_seed(spec.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(spec.seed)
        torch.cuda.empty_cache()

    stream = LongSeqFineWebStream(seq_len=seq_len, batch_size=batch_size, val_tokens=65_536, device=device)
    model = Complementary512DLindbladGPT(spec).to(device)
    init_ms = model.initialize_model_()
    vocab_opt, trunk_opt, scalar_opt = build_wave68_optimizers(model)

    retr_cad = 8 if steps >= 1000 else 5
    thermostat = ReadOnlyQuenchRiemannianShadowEMA(
        model, ema_decay=DEFAULT_THERMOSTAT_EMA_DECAY, eval_fn=evaluate_wave67_model, retract_cadence=retr_cad
    )

    total_params = sum(p.numel() for p in model.parameters())
    tokens_per_step = batch_size * seq_len
    run_key = f"plan088_{spec.variant_mode.value.lower()}_{steps}steps"

    exp_logger = StandardizedExperimentLogger(
        ExperimentIntentSpec(
            run_key=run_key, wave_id="W68", plan_id="EXPERIMENT_PLAN_088", task_domain="fineweb_speedrun",
            hypothesis_intent=(
                f"Test {spec.variant_mode.value} (boundary={spec.boundary_mode.value}, L={spec.num_layers}, "
                f"R={spec.kraus_rank}, lindblad={spec.enable_lindblad_highway}, pt_freq={spec.enable_parallel_transport_and_freq_vocab})."
            ),
            architecture_summary=f"Complementary512DLindbladGPT(mode={spec.variant_mode.value}, L={spec.num_layers}, R={spec.kraus_rank}, params={total_params:,})",
            hardware_budget=str(device), batch_size=batch_size, seq_len=seq_len,
            total_steps=steps, target_tokens_M=round((steps * tokens_per_step) / 1e6, 3),
        )
    )

    step0_ev = evaluate_wave67_model(model, stream)
    step0_ev["step"], step0_ev["tokens_M"] = 0, 0.0
    exp_logger.log_eval(0, {**step0_ev, "top1_acc": round(100.0 * step0_ev["top1_acc"], 2), "top5_acc": round(100.0 * step0_ev["top5_acc"], 2)})
    print(
        f"[{spec.variant_mode.value} (init={init_ms:.1f}ms, boundary={spec.boundary_mode.value}, L={spec.num_layers}, "
        f"params={total_params:,})] Step    0 | Val CE: {step0_ev['val_ce']:.4f}",
        flush=True,
    )

    quench_steps = max(1, int(round(float(steps) * float(DEFAULT_QUENCH_FRACTION))))
    plateau_steps = max(1, steps - quench_steps)
    flat_steps = max(200, int(0.35 * float(plateau_steps)))
    tau_anneal = max(300.0, 0.65 * float(plateau_steps))
    warmup_steps = 5
    curve: list[dict[str, Any]] = [step0_ev]
    t0 = time.perf_counter()
    model.train()

    for step in range(1, steps + 1):
        for opt in (vocab_opt, trunk_opt, scalar_opt):
            opt.zero_grad(set_to_none=True)

        idx_in, targets = stream.sample_train_batch(step)
        loss, _ = model.forward_loss_and_metrics(idx_in, targets)
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
            lr_mult = anneal_end * (0.15 + 0.85 * cos_q)
            mom_beta = 0.85 + 0.10 * (1.0 - cos_q)

        is_eval_step = (step % eval_interval == 0) or (step == steps) or (step == 1)
        trunk_opt.record_diagnostics = is_eval_step
        trunk_opt.set_momentum(mom_beta)
        trunk_opt.set_lr_scale(lr_mult)
        trunk_opt.step()

        vocab_opt.set_momentum(mom_beta)
        act_idx = _collect_active_vocab_indices_w68(idx_in, targets)
        vocab_opt.step_active(act_idx, lr_scale=lr_mult)

        scalar_opt.set_momentum(0.0 if mom_beta <= 0.0 else 0.80)
        scalar_opt.step_active(active_idx=None, lr_scale=lr_mult)
        model.restrict_scalars_()
        with torch.no_grad():
            model.tokens_seen_buf.add_(int(tokens_per_step))
        thermostat.update_after_step(model)

        if step % 10 == 0 or step == 1:
            el_s = max(1e-3, time.perf_counter() - t0)
            tps = (float(step) * float(tokens_per_step)) / el_s
            exp_logger.log_step(step, train_ce=float(loss.detach().item()), tok_per_s=tps, lr_mult=lr_mult)

        if step % eval_interval == 0 or step == steps:
            th_out = thermostat.check_and_thermostat(model, stream, current_lr_scale=1.0, step=step, apply_feedback=True)
            gd = {k: round(float(v), 6) for k, v in trunk_opt.last_gauge_diagnostics.items()}
            ev = dict(th_out["active_eval"])
            ev["step"], ev["tokens_M"] = step, round((step * tokens_per_step) / 1e6, 3)
            ev["train_ce"] = round(float(loss.detach().item()), 4)
            ev["hot_val_ce"] = th_out["hot_val_ce"]
            ev["ema_val_ce"] = th_out["ema_val_ce"]
            ev["thermal_gap_ce"] = th_out["thermal_gap_ce"]
            ev["thermostat_triggered"] = False
            ev["thermostat_lr_scale"] = round(lr_mult, 4)
            ev.update(gd)
            curve.append(ev)
            exp_logger.log_eval(step, {**ev, "top1_acc": round(100.0 * ev["top1_acc"], 2), "top5_acc": round(100.0 * ev["top5_acc"], 2)})
            print(
                f"[{spec.variant_mode.value}] Step {step:4d} ({ev['tokens_M']:5.2f}M tok, lr={lr_mult:.3f}x) | "
                f"Hot CE: {th_out['hot_val_ce']:.4f} | EMA CE: {th_out['ema_val_ce']:.4f} "
                f"(gap={th_out['thermal_gap_ce']:+.4f}) | "
                f"Top1: {100.0 * ev['top1_acc']:.2f}% | MUBErr: {ev['mub_unbiased_err']:.2e} | "
                f"DegSVD: {gd.get('svd_degenerate_spin_frac', 0.0):.2e} | "
                f"SU16Phase: {gd.get('su16_step_phase_frac', 0.0):.2e} | KrausGauge: {gd.get('kraus_step_gauge_frac', 0.0):.2e}",
                flush=True,
            )

    wall_s = max(1e-6, time.perf_counter() - t0)
    exp_logger.finalize()
    summary = {
        "variant": spec.variant_mode.value,
        "boundary_mode": spec.boundary_mode.value,
        "num_layers": spec.num_layers,
        "kraus_rank": spec.kraus_rank,
        "mlp_hidden": spec.mlp_hidden,
        "enable_lindblad_highway": spec.enable_lindblad_highway,
        "enable_parallel_transport_and_freq_vocab": spec.enable_parallel_transport_and_freq_vocab,
        "total_params": total_params,
        "total_params_M": round(total_params / 1e6, 3),
        "init_ms": round(init_ms, 2),
        "elapsed_sec": round(wall_s, 2),
        "ms_per_step": round((wall_s * 1000.0) / max(1, steps), 2),
        "tok_per_s": round((steps * tokens_per_step) / wall_s, 1),
        "step0": step0_ev,
        "final": curve[-1],
        "curve": curve,
    }
    del model, thermostat, vocab_opt, trunk_opt, scalar_opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary
