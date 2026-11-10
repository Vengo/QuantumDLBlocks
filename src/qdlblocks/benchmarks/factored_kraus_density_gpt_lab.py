"""Wave 70 (`EXPERIMENT_PLAN_090.md`): Factored Cross-Head Kraus Superoperator & Multi-Branch Pure Density-Matrix GPT.

Implements Theorems 9 & 10 from `lean/QuantumDLBlocksLean/FactoredCrossHeadKrausSuperoperator.lean`:
1. Factored Associative Cross-Head Kraus Superoperator (`Trick A`, Theorem 9a): computes the exact pairwise
   congruence $\\sum_{h',m} w_{h,h',m} (K_{\\text{out},h,m} K_{\\text{in},h',m}) \\rho_{t,h'} (K_{\\text{out},h,m} K_{\\text{in},h',m})^\\dagger$
   in $O(H M)$ congruences + packed $\\mathbb{R}^{256}$ routing without materializing the $(B, T, H, H, M, 16, 16)$ tensor.
2. Attention-Sum Commuted Cross-Head Kraus Superoperator (`Trick B`, Theorem 9b): commutes $K_{\\text{in},h',m}$
   through $\\sum_{s \\le t} A_{t,s}^{(h')}$ so it acts on the Rank-$R$ ($16 \\times 4$) Value Cholesky factors $(V_{\\text{re}}, V_{\\text{im}})$.
3. Multi-Branch Cross-Head Stiefel Kraus Superoperator (Theorem 10b): applies $M=4$ joint $\\operatorname{O}(512)$
   Stiefel branches in one fused GEMM to amplify single-token Rank-$4$ states into full Rank-$16$ mixed density matrices.
"""

from __future__ import annotations

import enum
import gc
import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from qdlblocks.benchmarks.complementary_512d_lindblad_8l_lab import _collect_active_vocab_indices_w68
from qdlblocks.benchmarks.experiment_logger import (
    ExperimentIntentSpec,
    StandardizedExperimentLogger,
)
from qdlblocks.benchmarks.pure_density_matrix_gpt_lab import (
    DEFAULT_PURE_CHOLESKY_RANK,
    DEFAULT_PURE_RHO_DIM,
    DEFAULT_PURE_RHO_HEADS,
    PURE_RHO_TRACE_EPS,
    PureDensityMatrixBlock,
    PureDensityMatrixGPT,
    Wave69ExperimentSpec,
    Wave69VariantMode,
    apply_complex_rope_to_cholesky,
    apply_complex_unitary_to_cholesky,
    apply_quantum_or_and_schur_logic,
    build_wave69_optimizers,
    compute_hilbert_schmidt_quantum_attention,
    evaluate_wave69_model,
    normalize_cholesky_factor_frobenius,
    normalize_packed_density_trace,
    pack_cholesky_factor_outer_isometric,
)
from qdlblocks.benchmarks.selective_exact_quotient_lab import (
    DEFAULT_QUENCH_FRACTION,
    DEFAULT_THERMOSTAT_EMA_DECAY,
    ReadOnlyQuenchRiemannianShadowEMA,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import LongSeqFineWebStream
from qdlblocks.layers.impure_density_mirror_block import (
    pack_hermitian_isometric,
    unpack_hermitian_isometric,
)

DEFAULT_FACTORED_KRAUS_BRANCHES: int = 4
DEFAULT_FACTORED_KRAUS_BLEND_INIT: float = 0.25
DEFAULT_MULTIBRANCH_KRAUS_BLEND_INIT: float = 0.22
FULL_RANK_EIGVAL_THRESHOLD: float = 1e-5
DEFAULT_RESTORED_MOMENTUM_WARMUP_START: float = 0.85
DEFAULT_RESTORED_MOMENTUM_PLATEAU_BETA: float = 0.95
DEFAULT_RESTORED_MOMENTUM_QUENCH_END: float = 0.97
DEFAULT_MOMENTUM_WARMUP_STEPS: int = 300


def fast_pack_hermitian_to_isometric_real(rho_re: torch.Tensor, rho_im: torch.Tensor) -> torch.Tensor:
    r"""Frobenius-isometric Hermitian packing $\tilde{P}: \operatorname{Herm}(\mathbb{C}^{d_h}) \to \mathbb{R}^{d_h^2}$ (`<= 8` args)."""
    return pack_hermitian_isometric(rho_re, rho_im)


def fast_unpack_isometric_real_to_hermitian(
    packed: torch.Tensor,
    d_h: int = DEFAULT_PURE_RHO_DIM,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Frobenius-isometric Hermitian unpacking $\tilde{P}^{-1}: \mathbb{R}^{d_h^2} \to \operatorname{Herm}(\mathbb{C}^{d_h})$ (`<= 8` args)."""
    return unpack_hermitian_isometric(packed, d_h)


def normalize_kraus_routing_weights(w_route_raw: torch.Tensor) -> torch.Tensor:
    r"""Compute non-negative sum-to-1 routing weights $w_{h,h',m} = W_{h,h',m}^2 / \sum_{j,n} W_{h,j,n}^2$ (`<= 8` args)."""
    w_sq = w_route_raw.float().square() + 1e-8
    denom = w_sq.sum(dim=(-2, -1), keepdim=True).clamp_min(PURE_RHO_TRACE_EPS)
    return w_sq / denom


def _complex_congruence_from_real_imag(
    rho_re: torch.Tensor,
    rho_im: torch.Tensor,
    k_re: torch.Tensor,
    k_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute Hermitian congruence $K \rho K^\dagger$ for complex $K = K_{\text{re}} + i K_{\text{im}}$ (`<= 8` args)."""
    kr_r = torch.matmul(k_re, rho_re) - torch.matmul(k_im, rho_im)
    kr_i = torch.matmul(k_re, rho_im) + torch.matmul(k_im, rho_re)
    kt_re = k_re.transpose(-2, -1)
    kt_im = k_im.transpose(-2, -1)
    out_re = torch.matmul(kr_r, kt_re) + torch.matmul(kr_i, kt_im)
    out_im = torch.matmul(kr_i, kt_re) - torch.matmul(kr_r, kt_im)
    return out_re, out_im


def apply_naive_pairwise_cross_head_kraus(
    rho_packed: torch.Tensor,
    k_in_pair: tuple[torch.Tensor, torch.Tensor],
    k_out_pair: tuple[torch.Tensor, torch.Tensor],
    w_route_raw: torch.Tensor,
) -> torch.Tensor:
    r"""Reference $O(H^2 M)$ pairwise congruence loop $\mathcal{E}_{\text{naive}}(\boldsymbol{\rho})$ (`Theorem 9a`, `<= 8` args).

    Materializes the composite operators $K_{h,h',m} = K_{\text{out},h,m} K_{\text{in},h',m}$ of shape `(H, H, M, 16, 16)`
    for exact numerical verification against `apply_factored_associative_cross_head_kraus`.
    """
    orig_dtype = rho_packed.dtype
    k_in_re, k_in_im = k_in_pair[0].float(), k_in_pair[1].float()
    k_out_re, k_out_im = k_out_pair[0].float(), k_out_pair[1].float()
    w_norm = normalize_kraus_routing_weights(w_route_raw)

    ko_r, ko_i = k_out_re.unsqueeze(1), k_out_im.unsqueeze(1)
    ki_r, ki_i = k_in_re.unsqueeze(0), k_in_im.unsqueeze(0)
    k_comp_re = torch.matmul(ko_r, ki_r) - torch.matmul(ko_i, ki_i)
    k_comp_im = torch.matmul(ko_r, ki_i) + torch.matmul(ko_i, ki_r)

    rho_re, rho_im = fast_unpack_isometric_real_to_hermitian(rho_packed.float())
    rho_re_exp = rho_re.permute(0, 2, 1, 3, 4).unsqueeze(2).unsqueeze(4)
    rho_im_exp = rho_im.permute(0, 2, 1, 3, 4).unsqueeze(2).unsqueeze(4)
    kc_re = k_comp_re.unsqueeze(0).unsqueeze(0)
    kc_im = k_comp_im.unsqueeze(0).unsqueeze(0)

    cong_re, cong_im = _complex_congruence_from_real_imag(rho_re_exp, rho_im_exp, kc_re, kc_im)
    w_exp = w_norm.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    mix_re = (cong_re * w_exp).sum(dim=(3, 4)).permute(0, 2, 1, 3, 4)
    mix_im = (cong_im * w_exp).sum(dim=(3, 4)).permute(0, 2, 1, 3, 4)
    return fast_pack_hermitian_to_isometric_real(mix_re, mix_im).to(orig_dtype)


def apply_factored_associative_cross_head_kraus(
    rho_packed: torch.Tensor,
    k_in_pair: tuple[torch.Tensor, torch.Tensor],
    k_out_pair: tuple[torch.Tensor, torch.Tensor],
    w_route_raw: torch.Tensor,
) -> torch.Tensor:
    r"""Factored $O(H M)$ associative Cross-Head Kraus Superoperator (`Trick A`, Lean 4 Theorem 9a, `<= 8` args).

    Computes:
    $$\mathcal{E}_{\text{factored}}(\boldsymbol{\rho})_h = \sum_{m=1}^{M} K_{\text{out},h,m}
      \left(\sum_{h'=1}^{H} w_{h,h',m} \left(K_{\text{in},h',m} \rho_{t,h'} K_{\text{in},h',m}^\dagger\right)\right)
      K_{\text{out},h,m}^\dagger$$
    in 3 stages (`In-Congruence -> Packed R^256 Linear Routing -> Out-Congruence`) without materializing
    $(H, H, M, 16, 16)$ or $(B, T, H, M, 16, 16)$ intermediate tensors.
    """
    work_dtype = rho_packed.dtype
    k_in_re, k_in_im = k_in_pair[0].to(work_dtype), k_in_pair[1].to(work_dtype)
    k_out_re, k_out_im = k_out_pair[0].to(work_dtype), k_out_pair[1].to(work_dtype)
    w_norm = normalize_kraus_routing_weights(w_route_raw).to(work_dtype)

    rho_re, rho_im = fast_unpack_isometric_real_to_hermitian(rho_packed)
    m_branches = k_in_re.size(1)
    out_packed = torch.zeros_like(rho_packed)

    for m in range(m_branches):
        ki_re = k_in_re[:, m].unsqueeze(0).unsqueeze(2)
        ki_im = k_in_im[:, m].unsqueeze(0).unsqueeze(2)
        sig_re, sig_im = _complex_congruence_from_real_imag(rho_re, rho_im, ki_re, ki_im)
        sig_packed = fast_pack_hermitian_to_isometric_real(sig_re, sig_im)

        routed_packed = torch.einsum("hj,bjtd->bhtd", w_norm[:, :, m], sig_packed)
        bar_re, bar_im = fast_unpack_isometric_real_to_hermitian(routed_packed)

        ko_re = k_out_re[:, m].unsqueeze(0).unsqueeze(2)
        ko_im = k_out_im[:, m].unsqueeze(0).unsqueeze(2)
        out_re_m, out_im_m = _complex_congruence_from_real_imag(bar_re, bar_im, ko_re, ko_im)
        out_packed = out_packed + fast_pack_hermitian_to_isometric_real(out_re_m, out_im_m)
    return out_packed


def apply_attention_commuted_cross_head_kraus(
    v_chol_pair: tuple[torch.Tensor, torch.Tensor],
    attn_weights: torch.Tensor,
    k_in_pair: tuple[torch.Tensor, torch.Tensor],
    k_out_pair: tuple[torch.Tensor, torch.Tensor],
    w_route_raw: torch.Tensor,
) -> torch.Tensor:
    r"""Attention-Sum Commuted Cross-Head Kraus Superoperator (`Trick B`, Lean 4 Theorem 9b, `<= 8` args).

    Commutes $K_{\text{in},h',m}$ inside $\rho_{t,h'}^{\text{attn}} = \sum_{s \le t} A_{t,s}^{(h')} V_{s,h'} V_{s,h'}^\dagger$
    so $K_{\text{in},h',m}$ multiplies the Rank-$R$ ($16 \times 4$) Value Cholesky factor $V_{s,h'}$ once before forming
    packed outer products and contracting with `attn_weights`.
    """
    orig_dtype = v_chol_pair[0].dtype
    v_re = v_chol_pair[0].float().unsqueeze(2)
    v_im = v_chol_pair[1].float().unsqueeze(2)
    k_in_re, k_in_im = k_in_pair[0].float(), k_in_pair[1].float()
    k_out_re, k_out_im = k_out_pair[0].float(), k_out_pair[1].float()
    w_norm = normalize_kraus_routing_weights(w_route_raw)

    ki_re, ki_im = k_in_re.unsqueeze(0).unsqueeze(3), k_in_im.unsqueeze(0).unsqueeze(3)
    kv_re = torch.matmul(ki_re, v_re) - torch.matmul(ki_im, v_im)
    kv_im = torch.matmul(ki_re, v_im) + torch.matmul(ki_im, v_re)
    kv_packed = pack_cholesky_factor_outer_isometric(kv_re, kv_im)

    sig_packed_bhtm = torch.matmul(attn_weights.float().unsqueeze(2), kv_packed)
    sig_packed = sig_packed_bhtm.permute(0, 3, 1, 2, 4)

    routed_packed = torch.einsum("hjm,btjmd->bthmd", w_norm, sig_packed)
    bar_re, bar_im = fast_unpack_isometric_real_to_hermitian(routed_packed)

    ko_re, ko_im = k_out_re.unsqueeze(0).unsqueeze(0), k_out_im.unsqueeze(0).unsqueeze(0)
    out_re_m, out_im_m = _complex_congruence_from_real_imag(bar_re, bar_im, ko_re, ko_im)
    out_re = out_re_m.sum(dim=3).permute(0, 2, 1, 3, 4)
    out_im = out_im_m.sum(dim=3).permute(0, 2, 1, 3, 4)
    return fast_pack_hermitian_to_isometric_real(out_re, out_im).to(orig_dtype)


def apply_commuted_cholesky_cross_head_kraus(
    v_bar_pair: tuple[torch.Tensor, torch.Tensor],
    k_in_pair: tuple[torch.Tensor, torch.Tensor],
    k_out_pair: tuple[torch.Tensor, torch.Tensor],
    w_route_raw: torch.Tensor,
) -> torch.Tensor:
    r"""Fast $O(T)$ Cholesky-Factor Cross-Head Kraus Superoperator (`Trick B` on `v_bar`, `<= 8` args).

    Applies $K_{\text{in},h',m}$ directly to the attention-summed Rank-$R$ ($16 \times 4$) Value Cholesky factor
    $\bar{V}_{t,h'}$ in $O(T)$ time (avoiding $M$ redundant $O(T^2)$ sequence matmuls and $(B, T, H, M, 16, 16)$ tensors).
    """
    v_re, v_im = v_bar_pair[0], v_bar_pair[1]
    work_dtype = v_re.dtype
    k_in_re, k_in_im = k_in_pair[0].to(work_dtype), k_in_pair[1].to(work_dtype)
    k_out_re, k_out_im = k_out_pair[0].to(work_dtype), k_out_pair[1].to(work_dtype)
    w_norm = normalize_kraus_routing_weights(w_route_raw).to(work_dtype)

    b_sz, n_h, seq_l, d_h, _ = v_re.shape
    m_branches = k_in_re.size(1)
    out_packed = torch.zeros((b_sz, n_h, seq_l, d_h * d_h), device=v_re.device, dtype=work_dtype)

    for m in range(m_branches):
        ki_re = k_in_re[:, m].unsqueeze(0).unsqueeze(2)
        ki_im = k_in_im[:, m].unsqueeze(0).unsqueeze(2)
        kv_re = torch.matmul(ki_re, v_re) - torch.matmul(ki_im, v_im)
        kv_im = torch.matmul(ki_re, v_im) + torch.matmul(ki_im, v_re)
        sig_packed = pack_cholesky_factor_outer_isometric(kv_re, kv_im)

        routed_packed = torch.einsum("hj,bjtd->bhtd", w_norm[:, :, m], sig_packed)
        bar_re, bar_im = fast_unpack_isometric_real_to_hermitian(routed_packed)

        ko_re = k_out_re[:, m].unsqueeze(0).unsqueeze(2)
        ko_im = k_out_im[:, m].unsqueeze(0).unsqueeze(2)
        out_re_m, out_im_m = _complex_congruence_from_real_imag(bar_re, bar_im, ko_re, ko_im)
        out_packed = out_packed + fast_pack_hermitian_to_isometric_real(out_re_m, out_im_m)
    return out_packed


def apply_multibranch_stiefel_kraus_superoperator(
    l_re: torch.Tensor,
    l_im: torch.Tensor,
    w_kraus_branches: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Multi-Branch Cross-Head Stiefel Kraus Superoperator (`Lean 4 Theorem 10b`, `<= 8` args).

    Applies $M=4$ joint cross-head orthogonal Stiefel transformations $\mathcal{K}_1, \dots, \mathcal{K}_M \in \operatorname{O}(512)$
    in a single fused GEMM `(B*T*R, 512) @ (512, M*512)` to amplify a Rank-$R$ ($R=4$) purification factor into a
    Rank-$(M \cdot R) = \text{Rank-}16$ full-rank mixed density matrix $\rho_{t,h}^{\text{kraus}} = \frac{1}{M}\sum_{m=1}^M Z_{t,h,m} Z_{t,h,m}^\dagger$.
    """
    b_sz, n_h, seq_l, d_h, r_chol = l_re.shape
    n_branches = w_kraus_branches.size(0)
    flat_dim = n_h * 2 * d_h
    work_dtype = l_re.dtype

    l_joint = torch.stack([l_re, l_im], dim=-2).permute(0, 2, 5, 1, 4, 3).reshape(b_sz * seq_l * r_chol, flat_dim)
    w_fused = w_kraus_branches.to(work_dtype).permute(1, 0, 2).reshape(flat_dim, n_branches * flat_dim)
    y_btr = torch.matmul(l_joint, w_fused)

    z_branches = y_btr.view(b_sz, seq_l, r_chol, n_branches, n_h, 2, d_h).permute(0, 4, 1, 3, 6, 2, 5).contiguous()
    z_re, z_im = normalize_cholesky_factor_frobenius(z_branches[..., 0], z_branches[..., 1])

    scale_m = 1.0 / math.sqrt(float(n_branches))
    z_full_re = z_re.permute(0, 1, 2, 4, 3, 5).reshape(b_sz, n_h, seq_l, d_h, n_branches * r_chol) * scale_m
    z_full_im = z_im.permute(0, 1, 2, 4, 3, 5).reshape(b_sz, n_h, seq_l, d_h, n_branches * r_chol) * scale_m
    rho_kraus_packed = pack_cholesky_factor_outer_isometric(z_full_re, z_full_im)
    l_prin_re, l_prin_im = normalize_cholesky_factor_frobenius(z_re.mean(dim=3), z_im.mean(dim=3))
    return l_prin_re, l_prin_im, normalize_packed_density_trace(rho_kraus_packed, d_h)


class Wave70VariantMode(str, enum.Enum):
    """Ablation modes for Wave 70 (`EXPERIMENT_PLAN_090.md`)."""

    W70_K0_LUT_PACKED_BASELINE_4L = "W70_K0_LUT_PACKED_BASELINE_4L"
    W70_K1_FACTORED_KRAUS_TRICK_A_4L = "W70_K1_FACTORED_KRAUS_TRICK_A_4L"
    W70_K2_COMMUTED_AND_MULTIBRANCH_4L = "W70_K2_COMMUTED_AND_MULTIBRANCH_4L"
    W70_K3_DEEP_8L_FACTORED_KRAUS = "W70_K3_DEEP_8L_FACTORED_KRAUS"


@dataclass(frozen=True)
class Wave70ExperimentSpec:
    """Configuration for `FactoredKrausPureDensityGPT` (`Rule 2`, `<= 8` fields)."""

    variant_mode: Wave70VariantMode = Wave70VariantMode.W70_K2_COMMUTED_AND_MULTIBRANCH_4L
    num_layers: int = 4
    kraus_branches: int = DEFAULT_FACTORED_KRAUS_BRANCHES
    enable_factored_kraus_attn: bool = True
    enable_commuted_trick_b: bool = True
    enable_multibranch_stiefel_kraus: bool = True
    seed: int = 1337


class FactoredKrausPureDensityBlock(PureDensityMatrixBlock):
    r"""Pure Density-Matrix Block with Factored Cross-Head Kraus Superoperator (`Theorems 9 & 10`)."""

    def __init__(
        self,
        spec: Wave70ExperimentSpec | None = None,
        depth_scale: float = 1.0,
    ) -> None:
        self.w70_spec = spec or Wave70ExperimentSpec()
        super().__init__(
            num_heads=DEFAULT_PURE_RHO_HEADS,
            d_h=DEFAULT_PURE_RHO_DIM,
            cholesky_rank=DEFAULT_PURE_CHOLESKY_RANK,
            enable_history=True,
            depth_scale=depth_scale,
        )
        self.m_branches = self.w70_spec.kraus_branches
        hm = self.num_heads * self.m_branches

        if self.w70_spec.enable_factored_kraus_attn:
            eye_hm = torch.eye(self.d_h, dtype=torch.float32).unsqueeze(0).expand(hm, self.d_h, self.d_h)
            self.k_in_re = nn.Parameter(eye_hm.clone())
            self.k_in_im = nn.Parameter(torch.zeros_like(eye_hm))
            self.k_out_re = nn.Parameter(eye_hm.clone())
            self.k_out_im = nn.Parameter(torch.zeros_like(eye_hm))
            self.w_kraus_route = nn.Parameter(
                torch.full((self.num_heads, self.num_heads, self.m_branches), 1.0 / math.sqrt(float(hm)), dtype=torch.float32)
            )
        else:
            self.register_parameter("k_in_re", None)
            self.register_parameter("k_in_im", None)
            self.register_parameter("k_out_re", None)
            self.register_parameter("k_out_im", None)
            self.register_parameter("w_kraus_route", None)

        if self.w70_spec.enable_multibranch_stiefel_kraus:
            self.w_multibranch_stiefel = nn.Parameter(
                torch.stack([torch.eye(self.joint_dim, dtype=torch.float32) for _ in range(self.m_branches)], dim=0)
            )
        else:
            self.register_parameter("w_multibranch_stiefel", None)

        self.kraus_attn_blend_logit = nn.Parameter(
            torch.full((self.num_heads,), math.log(DEFAULT_FACTORED_KRAUS_BLEND_INIT / (1.0 - DEFAULT_FACTORED_KRAUS_BLEND_INIT)), dtype=torch.float32)
        )
        self.multibranch_blend_logit = nn.Parameter(
            torch.full((self.num_heads,), math.log(DEFAULT_MULTIBRANCH_KRAUS_BLEND_INIT / (1.0 - DEFAULT_MULTIBRANCH_KRAUS_BLEND_INIT)), dtype=torch.float32)
        )

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Return real Stiefel parameters including `w_multibranch_stiefel` (`<= 8` args)."""
        params = super().real_stiefel_parameters()
        if self.w_multibranch_stiefel is not None:
            params.append(self.w_multibranch_stiefel)
        return params

    def su16_unitary_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        r"""Return $\mathfrak{su}(16)$ unitary pairs including factored Kraus `k_in` and `k_out` (`<= 8` args)."""
        pairs = super().su16_unitary_pairs()
        if self.k_in_re is not None and self.k_in_im is not None:
            pairs.append((self.k_in_re, self.k_in_im))
        if self.k_out_re is not None and self.k_out_im is not None:
            pairs.append((self.k_out_re, self.k_out_im))
        return pairs

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return bounded scalar/routing parameters (`<= 8` args)."""
        params = super().scalar_parameters()
        params.extend([self.kraus_attn_blend_logit, self.multibranch_blend_logit])
        if self.w_kraus_route is not None:
            params.append(self.w_kraus_route)
        return params

    def restrict_scalars_(self) -> None:
        """Clamp bounded scalar parameters in-place (`<= 8` args)."""
        super().restrict_scalars_()
        with torch.no_grad():
            self.kraus_attn_blend_logit.clamp_(-4.0, 4.0)
            self.multibranch_blend_logit.clamp_(-4.0, 4.0)

    def forward(
        self,
        l_in_re: torch.Tensor,
        l_in_im: torch.Tensor,
        rho_in_packed: torch.Tensor,
        cos_t: torch.Tensor,
        sin_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Evolve multi-head density matrices via HS attention + Factored Cross-Head Kraus + Schur logic (`<= 8` args)."""
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
        v_payload = torch.cat(
            [rho_v_packed, v_re.reshape(bsz, num_h, seq_len, chol_flat_dim), v_im.reshape(bsz, num_h, seq_len, chol_flat_dim)],
            dim=-1,
        )

        if self.history_buffer.enable_history:
            hk_re, hk_im = apply_complex_unitary_to_cholesky(
                self.history_buffer.hist_l_re.to(work_dtype), self.history_buffer.hist_l_im.to(work_dtype), self.w_k_re, self.w_k_im
            )
            hv_re, hv_im = apply_complex_unitary_to_cholesky(
                self.history_buffer.hist_l_re.to(work_dtype), self.history_buffer.hist_l_im.to(work_dtype), self.w_v_re, self.w_v_im
            )
            hist_k_packed = pack_cholesky_factor_outer_isometric(hk_re, hk_im)
            hist_rho_v = self.history_buffer.hist_rho_packed.to(work_dtype)
            m_slots = hist_rho_v.size(2)
            hist_v_payload = torch.cat(
                [hist_rho_v, hv_re.reshape(1, num_h, m_slots, chol_flat_dim), hv_im.reshape(1, num_h, m_slots, chol_flat_dim)],
                dim=-1,
            )
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

        if self.w70_spec.enable_factored_kraus_attn and self.k_in_re is not None and self.w_kraus_route is not None:
            ki_r = self.k_in_re.view(num_h, self.m_branches, d_h, d_h)
            ki_i = self.k_in_im.view(num_h, self.m_branches, d_h, d_h)
            ko_r = self.k_out_re.view(num_h, self.m_branches, d_h, d_h)
            ko_i = self.k_out_im.view(num_h, self.m_branches, d_h, d_h)
            if self.w70_spec.enable_commuted_trick_b:
                if torch.is_grad_enabled():
                    rho_fk = torch.utils.checkpoint.checkpoint(
                        apply_commuted_cholesky_cross_head_kraus,
                        (v_bar_re, v_bar_im), (ki_r, ki_i), (ko_r, ko_i), self.w_kraus_route,
                        use_reentrant=False,
                    )
                else:
                    rho_fk = apply_commuted_cholesky_cross_head_kraus(
                        (v_bar_re, v_bar_im), (ki_r, ki_i), (ko_r, ko_i), self.w_kraus_route
                    )
            else:
                if torch.is_grad_enabled():
                    rho_fk = torch.utils.checkpoint.checkpoint(
                        apply_factored_associative_cross_head_kraus,
                        rho_attn_v, (ki_r, ki_i), (ko_r, ko_i), self.w_kraus_route,
                        use_reentrant=False,
                    )
                else:
                    rho_fk = apply_factored_associative_cross_head_kraus(
                        rho_attn_v, (ki_r, ki_i), (ko_r, ko_i), self.w_kraus_route
                    )
            lam_k = torch.sigmoid(self.kraus_attn_blend_logit.float()).to(work_dtype).view(1, num_h, 1, 1)
            rho_kraus_packed = (1.0 - lam_k) * rho_kraus_packed + lam_k * rho_fk

        a_attn = self.attn_alpha.to(work_dtype).view(1, num_h, 1, 1)
        a_5d = a_attn.unsqueeze(-1)
        rho_mid_packed = normalize_packed_density_trace((1.0 - a_attn) * rho_in_packed + a_attn * rho_kraus_packed, d_h)
        l_mid_re, l_mid_im = normalize_cholesky_factor_frobenius(
            (1.0 - a_5d) * l_in_re + a_5d * vk_re, (1.0 - a_5d) * l_in_im + a_5d * vk_im
        )

        if torch.is_grad_enabled():
            l_log_re, l_log_im, rho_log_packed = torch.utils.checkpoint.checkpoint(
                apply_quantum_or_and_schur_logic,
                l_mid_re, l_mid_im, rho_mid_packed, self.w_or_a, self.w_or_b,
                (self.u_or_a_re, self.u_or_a_im, self.u_or_b_re, self.u_or_b_im, self.w_schur_up_a, self.w_schur_up_b, self.w_schur_down),
                self.and_gate_logit,
                use_reentrant=False,
            )
        else:
            l_log_re, l_log_im, rho_log_packed = apply_quantum_or_and_schur_logic(
                l_mid_re, l_mid_im, rho_mid_packed, self.w_or_a, self.w_or_b,
                (self.u_or_a_re, self.u_or_a_im, self.u_or_b_re, self.u_or_b_im, self.w_schur_up_a, self.w_schur_up_b, self.w_schur_down),
                self.and_gate_logit,
            )

        if self.w70_spec.enable_multibranch_stiefel_kraus and self.w_multibranch_stiefel is not None:
            if torch.is_grad_enabled():
                l_mb_re, l_mb_im, rho_mb_packed = torch.utils.checkpoint.checkpoint(
                    apply_multibranch_stiefel_kraus_superoperator,
                    l_mid_re, l_mid_im, self.w_multibranch_stiefel,
                    use_reentrant=False,
                )
            else:
                l_mb_re, l_mb_im, rho_mb_packed = apply_multibranch_stiefel_kraus_superoperator(
                    l_mid_re, l_mid_im, self.w_multibranch_stiefel
                )
            lam_mb = torch.sigmoid(self.multibranch_blend_logit.float()).to(work_dtype).view(1, num_h, 1, 1)
            lam_mb_5d = lam_mb.unsqueeze(-1)
            rho_log_packed = normalize_packed_density_trace((1.0 - lam_mb) * rho_log_packed + lam_mb * rho_mb_packed, d_h)
            l_log_re, l_log_im = normalize_cholesky_factor_frobenius(
                (1.0 - lam_mb_5d) * l_log_re + lam_mb_5d * l_mb_re,
                (1.0 - lam_mb_5d) * l_log_im + lam_mb_5d * l_mb_im,
            )

        b_log = self.logic_beta.to(work_dtype).view(1, num_h, 1, 1)
        b_5d = b_log.unsqueeze(-1)
        gam_depol = (0.12 * torch.sigmoid(self.depol_logit.float())).to(work_dtype).view(1, num_h, 1, 1)
        rho_blend = (1.0 - b_log) * rho_mid_packed + b_log * rho_log_packed
        rho_out_packed = normalize_packed_density_trace(
            (1.0 - gam_depol) * rho_blend + gam_depol * self.thermal_rho_packed.to(work_dtype), d_h
        )
        l_out_re, l_out_im = normalize_cholesky_factor_frobenius(
            (1.0 - b_5d) * l_mid_re + b_5d * l_log_re, (1.0 - b_5d) * l_mid_im + b_5d * l_log_im
        )
        return l_out_re, l_out_im, rho_out_packed


class FactoredKrausPureDensityGPT(PureDensityMatrixGPT):
    r"""Wave 70 Pure Density-Matrix GPT with Factored Cross-Head Kraus Superoperators (`Theorems 9 & 10`)."""

    def __init__(self, spec: Wave70ExperimentSpec | None = None) -> None:
        self.w70_spec = spec or Wave70ExperimentSpec()
        super().__init__(
            Wave69ExperimentSpec(
                variant_mode=Wave69VariantMode.W69_P2_PURE_DENSITY_EMA_NGRAM_HISTORY_4L,
                num_layers=self.w70_spec.num_layers,
                seed=self.w70_spec.seed,
            )
        )
        self.depth_scale = 1.0 / math.sqrt(float(max(1, self.w70_spec.num_layers)))
        self.density_blocks = nn.ModuleList(
            [FactoredKrausPureDensityBlock(spec=self.w70_spec, depth_scale=self.depth_scale) for _ in range(self.w70_spec.num_layers)]
        )
        self.initialize_quantum_state(seed=self.w70_spec.seed)


def build_wave70_variant_spec(mode: Wave70VariantMode, seed: int = 1337) -> Wave70ExperimentSpec:
    """Resolve `Wave70ExperimentSpec` from `Wave70VariantMode` (`<= 8` args)."""
    if mode == Wave70VariantMode.W70_K0_LUT_PACKED_BASELINE_4L:
        return Wave70ExperimentSpec(
            variant_mode=mode,
            num_layers=4,
            enable_factored_kraus_attn=False,
            enable_commuted_trick_b=False,
            enable_multibranch_stiefel_kraus=False,
            seed=seed,
        )
    if mode == Wave70VariantMode.W70_K1_FACTORED_KRAUS_TRICK_A_4L:
        return Wave70ExperimentSpec(
            variant_mode=mode,
            num_layers=4,
            enable_factored_kraus_attn=True,
            enable_commuted_trick_b=False,
            enable_multibranch_stiefel_kraus=False,
            seed=seed,
        )
    if mode == Wave70VariantMode.W70_K2_COMMUTED_AND_MULTIBRANCH_4L:
        return Wave70ExperimentSpec(
            variant_mode=mode,
            num_layers=4,
            enable_factored_kraus_attn=True,
            enable_commuted_trick_b=True,
            enable_multibranch_stiefel_kraus=True,
            seed=seed,
        )
    return Wave70ExperimentSpec(
        variant_mode=mode,
        num_layers=8,
        enable_factored_kraus_attn=True,
        enable_commuted_trick_b=True,
        enable_multibranch_stiefel_kraus=True,
        seed=seed,
    )


def compute_restored_riemannian_momentum_schedule(step: int, steps: int) -> tuple[float, float]:
    r"""Compute `(lr_mult, mom_beta)` with restored `0.85 -> 0.95` Riemannian momentum warmup (`Theorem 11`, `<= 8` args).

    Warms `mom_beta` from `DEFAULT_RESTORED_MOMENTUM_WARMUP_START` (`0.85`) to
    `DEFAULT_RESTORED_MOMENTUM_PLATEAU_BETA` (`0.95`) over `DEFAULT_MOMENTUM_WARMUP_STEPS` (`300`),
    maintains `0.95` across the main plateau, and transitions to `0.97` during final cosine quench.
    """
    quench_steps = max(1, int(round(float(steps) * float(DEFAULT_QUENCH_FRACTION))))
    plateau_steps = max(1, steps - quench_steps)
    flat_steps = max(200, int(0.35 * float(plateau_steps)))
    tau_anneal = max(300.0, 0.65 * float(plateau_steps))
    lr_warmup_steps = 5
    mom_warmup_steps = min(DEFAULT_MOMENTUM_WARMUP_STEPS, max(5, plateau_steps // 2))

    anneal = (1.0 / math.sqrt(1.0 + max(0.0, float(step - flat_steps)) / tau_anneal)) if steps >= 1000 else 1.0
    if step <= lr_warmup_steps:
        warm_lr = float(step) / float(lr_warmup_steps)
        lr_mult = 0.25 + 0.75 * warm_lr
    elif step <= plateau_steps:
        lr_mult = anneal
    else:
        anneal_end = (1.0 / math.sqrt(1.0 + max(0.0, float(plateau_steps - flat_steps)) / tau_anneal)) if steps >= 1000 else 1.0
        q_p = float(step - plateau_steps) / float(max(1, quench_steps))
        cos_q = 0.5 * (1.0 + math.cos(math.pi * min(1.0, q_p)))
        lr_mult = anneal_end * (0.15 + 0.85 * cos_q)

    if step <= mom_warmup_steps:
        frac = float(step) / float(max(1, mom_warmup_steps))
        mom_beta = DEFAULT_RESTORED_MOMENTUM_WARMUP_START + (
            DEFAULT_RESTORED_MOMENTUM_PLATEAU_BETA - DEFAULT_RESTORED_MOMENTUM_WARMUP_START
        ) * frac
    elif step <= plateau_steps:
        mom_beta = DEFAULT_RESTORED_MOMENTUM_PLATEAU_BETA
    else:
        q_p = float(step - plateau_steps) / float(max(1, quench_steps))
        cos_q = 0.5 * (1.0 + math.cos(math.pi * min(1.0, q_p)))
        mom_beta = DEFAULT_RESTORED_MOMENTUM_PLATEAU_BETA + (
            DEFAULT_RESTORED_MOMENTUM_QUENCH_END - DEFAULT_RESTORED_MOMENTUM_PLATEAU_BETA
        ) * (1.0 - cos_q)
    return lr_mult, mom_beta


def audit_riemannian_momentum_buffer_properties(device: torch.device) -> dict[str, float]:
    r"""Numerically verify Lean 4 Theorem 11 (`Theorem 11a, 11b, 11c`) on the restored `0.95` momentum buffer (`<= 8` args).

    Verifies:
    1. Theorem 11a: Exact quotient horizontality of stored momentum buffers $M_t$ across $\mathfrak{su}(16)$,
       Kraus $\mathfrak{u}(R)$, and real Stiefel blocks at $\beta = 0.95$, plus 100% active buffer coverage on
       `FactoredKrausPureDensityGPT`.
    2. Theorem 11b: Opposing-momentum cleaning (`_clean_and_snr_gate_real_tangent`) guarantees exact orthogonality
       $\langle M_{\text{clean}}, G\rangle = 0$, Pythagorean energy contraction
       $\|M_{\text{clean}}\|^2 + \frac{\langle M_{\text{pre}}, G\rangle^2}{\|G\|^2} = \|M_{\text{pre}}\|^2$,
       and strict first-order descent $\langle \Xi_{\text{clean}}, G\rangle / \|G\|^2 = 1 + \beta = 1.95 > 0$
       (whereas uncleaned opposing momentum at $\beta = 0.95$ strictly ascends: $\langle \Xi_{\text{unclean}}, G\rangle / \|G\|^2 < 0$).
    3. Theorem 11c: Isometric horizontal parallel transport preserves new-point horizontality and kinetic energy $\|M\|_F$.
    """
    from qdlblocks.benchmarks.complementary_512d_lindblad_8l_lab import parallel_transport_complex_quotient_momentum
    from qdlblocks.benchmarks.horizontal_grassmann_gauge_free_lab import compute_kraus_branch_gauge_fraction
    from qdlblocks.benchmarks.selective_exact_quotient_lab import (
        _clean_and_snr_gate_real_tangent,
        _retract_complex_tall_stiefel_tensor_lr,
        compute_su_dh_global_phase_fraction,
        project_su_dh_mub_horizontal,
    )

    torch.manual_seed(2026)
    g_vec = torch.randn(1, 64, 16, device=device, dtype=torch.float32)
    h_orth = torch.randn(1, 64, 16, device=device, dtype=torch.float32)
    h_orth = h_orth - (torch.sum(h_orth * g_vec) / torch.sum(g_vec * g_vec)) * g_vec
    m_opposing = -2.5 * g_vec + 0.5 * h_orth
    beta = DEFAULT_RESTORED_MOMENTUM_PLATEAU_BETA

    g_norm_sq = float(torch.sum(g_vec * g_vec).item())
    m_pre_norm_sq = float(torch.sum(m_opposing * m_opposing).item())
    inner_pre_g = float(torch.sum(m_opposing * g_vec).item())

    m_unclean = beta * m_opposing + g_vec
    xi_unclean = g_vec + beta * m_unclean
    uncleaned_descent_ratio = float(torch.sum(xi_unclean * g_vec).item()) / max(1e-12, g_norm_sq)

    m_upd, xi_clean, _ = _clean_and_snr_gate_real_tangent(g_vec, m_opposing, 0.028, beta)
    m_clean = (m_upd - g_vec) / beta
    m_clean_norm_sq = float(torch.sum(m_clean * m_clean).item())
    cleaned_orth_err = abs(float(torch.sum(m_clean * g_vec).item())) / max(1e-12, math.sqrt(m_clean_norm_sq * g_norm_sq))
    proj_energy = (inner_pre_g * inner_pre_g) / max(1e-12, g_norm_sq)
    cleaned_pythagorean_err = abs((m_clean_norm_sq + proj_energy) - m_pre_norm_sq) / max(1e-12, m_pre_norm_sq)
    cleaned_descent_ratio = float(torch.sum(xi_clean * g_vec).item()) / max(1e-12, g_norm_sq)

    u_re = torch.eye(16, device=device, dtype=torch.float32).unsqueeze(0).expand(4, 16, 16).clone()
    u_im = torch.zeros_like(u_re)
    raw_m_re, raw_m_im = torch.randn_like(u_re), torch.randn_like(u_im)
    m_h_re, m_h_im = project_su_dh_mub_horizontal(raw_m_re, raw_m_im, u_re, u_im)
    m_h_norm = math.sqrt(float((m_h_re.square().sum() + m_h_im.square().sum()).item()))
    u_new_re, u_new_im = _retract_complex_tall_stiefel_tensor_lr(u_re, u_im, m_h_re, m_h_im, 0.05)
    u_new_re, u_new_im = _retract_complex_tall_stiefel_tensor_lr(
        u_new_re, u_new_im, torch.zeros_like(u_new_re), torch.zeros_like(u_new_im), 0.0
    )
    m_tr_re, m_tr_im = parallel_transport_complex_quotient_momentum(
        m_h_re, m_h_im, u_re, u_im, u_new_re, u_new_im, project_su_dh_mub_horizontal
    )
    tr_norm_after = math.sqrt(float((m_tr_re.square().sum() + m_tr_im.square().sum()).item()))
    transported_norm_rel_err = abs(tr_norm_after - m_h_norm) / max(1e-12, m_h_norm)
    transported_vertical_leak = float(compute_su_dh_global_phase_fraction(m_tr_re, m_tr_im, u_new_re, u_new_im))

    spec = build_wave70_variant_spec(Wave70VariantMode.W70_K2_COMMUTED_AND_MULTIBRANCH_4L, seed=1337)
    model = FactoredKrausPureDensityGPT(spec).to(device)
    trunk_opt, vocab_opt, scalar_opt = build_wave69_optimizers(model)
    for _ in range(2):
        for opt in (trunk_opt, vocab_opt, scalar_opt):
            opt.zero_grad(set_to_none=True)
        idx = torch.randint(0, 2048, (2, 16), device=device)
        tgt = torch.randint(0, 2048, (2, 16), device=device)
        loss = model(idx, targets=tgt, return_metrics=False)
        loss.backward()
        trunk_opt.record_diagnostics = True
        trunk_opt.set_momentum(beta)
        trunk_opt.set_lr_scale(1.0)
        trunk_opt.step()

    max_su16_mom_leak = 0.0
    for u_r, u_i in trunk_opt.mub_su_pairs:
        mr = trunk_opt.state[u_r]["momentum_buffer"]
        mi = trunk_opt.state[u_i]["momentum_buffer"]
        leak = compute_su_dh_global_phase_fraction(mr, mi, u_r.data.float(), u_i.data.float())
        max_su16_mom_leak = max(max_su16_mom_leak, float(leak))

    max_kraus_mom_leak = 0.0
    for k_r, k_i, _ in trunk_opt.kraus_pairs:
        mr = trunk_opt.state[k_r]["momentum_buffer"]
        mi = trunk_opt.state[k_i]["momentum_buffer"]
        leak = compute_kraus_branch_gauge_fraction(mr, mi, k_r.data.float(), k_i.data.float())
        max_kraus_mom_leak = max(max_kraus_mom_leak, float(leak))

    active_stiefel_buffers = sum(1 for p in trunk_opt.extra_real_stiefel if "momentum_buffer" in trunk_opt.state.get(p, {}))
    active_su16_buffers = sum(1 for p_r, _ in trunk_opt.mub_su_pairs if "momentum_buffer" in trunk_opt.state.get(p_r, {}))
    _, mom_at_warmup_end = compute_restored_riemannian_momentum_schedule(DEFAULT_MOMENTUM_WARMUP_STEPS, 18311)

    del model, trunk_opt, vocab_opt, scalar_opt
    return {
        "plateau_momentum_beta": float(mom_at_warmup_end),
        "uncleaned_opposing_descent_ratio": float(uncleaned_descent_ratio),
        "cleaned_opposing_descent_ratio": float(cleaned_descent_ratio),
        "cleaned_orthogonality_err": float(cleaned_orth_err),
        "cleaned_pythagorean_err": float(cleaned_pythagorean_err),
        "transported_norm_rel_err": float(transported_norm_rel_err),
        "transported_vertical_leak": float(transported_vertical_leak),
        "max_su16_mom_vertical_leak": float(max_su16_mom_leak),
        "max_kraus_mom_vertical_leak": float(max_kraus_mom_leak),
        "max_mub_mom_vertical_leak": float(max_su16_mom_leak),
        "active_stiefel_mom_buffers": float(active_stiefel_buffers),
        "active_su16_mom_buffers": float(active_su16_buffers),
    }


def run_wave70_variant(
    spec: Wave70ExperimentSpec,
    stream: LongSeqFineWebStream,
    steps: int = 200,
    batch_size: int = 8,
    seq_len: int = 512,
    eval_interval: int = 25,
) -> dict[str, Any]:
    """Train and evaluate one `Wave70VariantMode` run with `StandardizedExperimentLogger` (`<= 8` args)."""
    device = stream.device
    torch.manual_seed(spec.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(spec.seed)
        gc.collect()
        torch.cuda.empty_cache()

    model = FactoredKrausPureDensityGPT(spec).to(device)
    init_ms = model.initialize_quantum_state(seed=spec.seed)
    total_params = sum(p.numel() for p in model.parameters())
    trunk_opt, vocab_opt, scalar_opt = build_wave69_optimizers(model)
    retr_cad = 8 if steps >= 1000 else 5
    thermostat = ReadOnlyQuenchRiemannianShadowEMA(
        model, ema_decay=DEFAULT_THERMOSTAT_EMA_DECAY, eval_fn=evaluate_wave69_model, retract_cadence=retr_cad
    )

    tokens_per_step = batch_size * seq_len
    run_key = f"plan090_{spec.variant_mode.value.lower()}_{steps}steps"
    exp_logger = StandardizedExperimentLogger(
        ExperimentIntentSpec(
            run_key=run_key,
            wave_id="W70",
            plan_id="EXPERIMENT_PLAN_090",
            task_domain="fineweb_speedrun",
            hypothesis_intent=f"Wave 70 {spec.variant_mode.value} (L={spec.num_layers}, kraus_attn={spec.enable_factored_kraus_attn}, trick_b={spec.enable_commuted_trick_b}, multibranch={spec.enable_multibranch_stiefel_kraus}, mom_beta=0.85->0.95).",
            architecture_summary=f"FactoredKrausPureDensityGPT(L={spec.num_layers}, M={spec.kraus_branches}, params={total_params:,})",
            hardware_budget=str(device),
            batch_size=batch_size,
            seq_len=seq_len,
            total_steps=steps,
            target_tokens_M=round((steps * tokens_per_step) / 1e6, 3),
        )
    )

    step0_ev = evaluate_wave69_model(model, stream)
    step0_ev["step"], step0_ev["tokens_M"] = 0, 0.0
    exp_logger.log_eval(0, {**step0_ev, "top1_acc": round(100.0 * step0_ev["top1_acc"], 2), "top5_acc": round(100.0 * step0_ev["top5_acc"], 2)})
    print(f"[{spec.variant_mode.value} (init={init_ms:.1f}ms, L={spec.num_layers}, params={total_params:,})] Step    0 | Val CE: {step0_ev['val_ce']:.4f}", flush=True)

    curve: list[dict[str, Any]] = [step0_ev]
    t0 = time.perf_counter()
    model.train()

    for step in range(1, steps + 1):
        for opt in (trunk_opt, vocab_opt, scalar_opt):
            opt.zero_grad(set_to_none=True)
        idx_in, targets = stream.sample_train_batch(step)
        loss = model(idx_in, targets=targets, return_metrics=False)
        loss.backward()

        lr_mult, mom_beta = compute_restored_riemannian_momentum_schedule(step, steps)
        is_eval_step = (step % eval_interval == 0) or (step == steps) or (step == 1)
        trunk_opt.record_diagnostics = is_eval_step
        trunk_opt.set_momentum(mom_beta)
        trunk_opt.set_lr_scale(lr_mult)
        trunk_opt.step()

        vocab_opt.set_momentum(mom_beta)
        vocab_opt.step_active(_collect_active_vocab_indices_w68(idx_in, targets), lr_scale=lr_mult)
        scalar_opt.set_momentum(0.0 if mom_beta <= 0.0 else min(0.90, mom_beta))
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
            ev["thermostat_triggered"], ev["thermostat_lr_scale"], ev["mom_beta"] = False, round(lr_mult, 4), round(mom_beta, 4)
            ev.update(gd)
            curve.append(ev)
            exp_logger.log_eval(step, {**ev, "top1_acc": round(100.0 * ev["top1_acc"], 2), "top5_acc": round(100.0 * ev["top5_acc"], 2)})
            print(
                f"[{spec.variant_mode.value}] Step {step:5d} ({ev['tokens_M']:6.2f}M tok, lr={lr_mult:.3f}x, mom={mom_beta:.3f}) | "
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
