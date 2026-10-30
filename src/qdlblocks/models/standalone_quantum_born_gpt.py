"""Wave 48 (`EXPERIMENT_PLAN_063.md`): Standalone Teacher-Free Complex Quantum Born GPT.

Synthesizes the quantum-geometric mechanisms from *"Top-1 Accuracy vs Cross-Entropy Divergence"*
(Pages 1-36) into a 100% standalone, end-to-end differentiable language model (`StandaloneQuantumBornGPT`)
with zero classical teacher dependencies and zero Golub-Pereyra ridge-regression refits:

1. **Fused `288`-Width Isometric Density + First-Moment Ket Attention Packing**:
   - Packs the $\\sqrt{2}$-isometric Rank-$R=4$ Hermitian outer product $\\tilde{P}(\\rho_{s,h}^{\\text{tok}}) \\in \\mathbb{R}^{256}$
     and the unit-norm first-moment coherent value ket $|\\hat{v}_{s,h}^{\\text{coh}}\\rangle \\in \\mathbb{C}^{16} \\cong \\mathbb{R}^{32}$
     into a single contiguous `288`-dim vector per head, contracting both in one causal $p$-boson Born
     attention GEMM (`(B, H, T, T) @ (B, H, T, 288)`).
2. **Uhlmann Phase-Locked Rank-2 Lanczos-Krylov Eigenpair Extraction (`320`-Dim/Head Readout)**:
   - After 1-step Bures-Wasserstein McWeeny purification $\\rho \\leftarrow \\rho^2 / \\operatorname{Tr}(\\rho^2)$
     and per-head $U(16)$ unitary evolution, seeds a 2-step Lanczos-Krylov recursion with the contextual
     first-moment ket $|\\bar{v}_{t,h}\\rangle$ and locks the Uhlmann parallel-transport phase
     $\\langle \\bar{v}_{t,h} \\mid \\phi_{1,t,h}\\rangle \\in \\mathbb{R}_{\\ge 0}$.
   - Concatenates the unit-isometric-scale purified density $\\tilde{P}(\\rho_{\\text{out}}) \\in \\mathbb{R}^{256}$
     with both scaled Krylov eigenmodes $[\\sqrt{\\lambda_1}\\phi_1, \\sqrt{\\lambda_2}\\phi_2] \\in \\mathbb{R}^{64}$
     (`320` dims/head, `1,920` total across 6 heads) into a single direct Stiefel readout
     $W_{\\text{obs}} \\in \\text{St}(1920, 768)$ with scalar residual gain `obs_scale`.
3. **Tikhonov-Damped `RiemannianStiefelMuon` Across All 2D Stiefel & 3D $U(16)$ Layers**:
   - Updates 100% of 2D real Stiefel matrices and 3D complex head unitaries using Tikhonov-damped
     Newton-Schulz orthogonalization ($\\lambda_{\\text{tik}} = 10^{-3}$) and $1 / \\sqrt{1 + \\eta^2}$
     geodesic retraction.
4. **Active-Token Vocabulary Manifold Optimization & Rank-2 Krylov Born Readout**:
   - Supports `BatchSubspaceTikhonovGramMuon` (`(Q, N_act, A)` feature-Gram Muon on $\\sqrt{d}\\,\\mathbb{S}^{d-1}$),
     `RiemannianSphereTangentSNROptimizer`, and `AdamW` across `RANK2_KRYLOV_BORN` (`Rank2KrylovBornChunkedCEFn`,
     Rule-9 single-pass chunked autograd) and `SOFTCAP_LINEAR_CE` (`_CompiledChunkedSoftcapLinearCEFn`).
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
    DEFAULT_FUSED_HEAD_PACK_DIM,
    DEFAULT_KRYLOV_OBSERVABLE_HEAD_DIM,
    DEFAULT_NS5_STEPS,
    DEFAULT_RIEMANNIAN_MOMENTUM_BETA,
    DEFAULT_ROPE_BASE_THETA,
    DEFAULT_TIKHONOV_GRAM_LAMBDA,
    DEFAULT_UHLMANN_PHASE_EPS,
    INV_SQRT2_ISOMETRIC_SCALE,
    SQRT2_ISOMETRIC_OFFDIAG_SCALE,
    TRACE_NORM_EPS,
    _complex_newton_schulz5,
    _newton_schulz5_tall_real,
    bures_wasserstein_mcweeny_step,
    complex_l2_normalize,
    extract_rank2_krylov_eigenpair_uhlmann,
    fast_rank_r_hermitian_and_ket_pack,
    initialize_stiefel_parameter,
    riemannian_complex_unitary_muon_step,
    riemannian_stiefel_muon_step,
    unpack_hermitian_isometric,
)
from ..layers.isometric import apply_partial_key_offset
from ..optim.vocab_manifold_muon import (
    DEFAULT_BATCH_GRAM_MUON_LR,
    DEFAULT_SPHERE_GEODESIC_LR,
    DEFAULT_SPHERE_SNR_TRUST_TAU,
    BatchSubspaceTikhonovGramMuon,
    RiemannianSphereTangentSNROptimizer,
    project_rows_to_rms_sphere_,
)
from .compiled_leaderboard_classical_gpt import _CompiledChunkedSoftcapLinearCEFn

# ---------------------------------------------------------------------------
# Top-of-File Named Constants (Rule 7 — Synchronized with EXPERIMENT_PLAN_063.md)
# ---------------------------------------------------------------------------
DEFAULT_STANDALONE_VOCAB_SIZE: int = 50304
DEFAULT_STANDALONE_VALID_VOCAB: int = 50257
DEFAULT_STANDALONE_EMBED_DIM: int = 768
DEFAULT_STANDALONE_NUM_LAYERS: int = 12
DEFAULT_STANDALONE_NUM_HEADS: int = 6
DEFAULT_STANDALONE_HEAD_COMPLEX_DIM: int = 16
DEFAULT_STANDALONE_VALUE_BRANCHES: int = 4
DEFAULT_STANDALONE_MCWEENY_STEPS: int = 1

DEFAULT_STIEFEL_GEODESIC_LR: float = 0.025
DEFAULT_UNITARY_GEODESIC_LR: float = 0.020
DEFAULT_SCALAR_ADAM_LR: float = 0.005
DEFAULT_ADAMW_VOCAB_LR: float = 0.004
DEFAULT_V0_HIGHWAY_INIT_LAMBDA: float = 0.15
DEFAULT_RESIDUAL_GAIN_COEFF: float = 0.25

DEFAULT_KRYLOV_BORN_LOG_TEMP: float = 6.0
DEFAULT_BORN_LOG_INTENSITY_EPS: float = 0.01
DEFAULT_LOGIT_SOFTCAP: float = 15.0
DEFAULT_CE_CHUNK_TOKENS: int = 2048


class QuantumAttentionMode(str, Enum):
    """Strategy enum selecting Uhlmann-Krylov 320-dim readout vs. phase-erased 256-dim density readout (Rule 3)."""

    UHLMANN_KRYLOV_320 = "uhlmann_krylov_320"
    PHASE_ERASED_DENSITY_256 = "phase_erased_density_256"


class VocabularyReadoutMode(str, Enum):
    """Strategy enum selecting Rank-2 Krylov Born log-intensity CE vs. linear softcap CE (Rule 3)."""

    RANK2_KRYLOV_BORN = "rank2_krylov_born"
    SOFTCAP_LINEAR_CE = "softcap_linear_ce"


class VocabOptimizerMode(str, Enum):
    """Strategy enum selecting active-token vocabulary manifold optimizer vs. AdamW baseline (Rule 3)."""

    BATCH_SUBSPACE_GRAM_MUON = "batch_subspace_gram_muon"
    SPHERE_TANGENT_SNR_MUON = "sphere_tangent_snr_muon"
    ADAMW_BASELINE = "adamw_baseline"


@dataclass(frozen=True)
class StandaloneQuantumBornConfig:
    """Configuration for `StandaloneQuantumBornGPT` (`<= 8` fields, Rule 2)."""

    vocab_size: int = DEFAULT_STANDALONE_VOCAB_SIZE
    n_layer: int = DEFAULT_STANDALONE_NUM_LAYERS
    n_head: int = DEFAULT_STANDALONE_NUM_HEADS
    n_embd: int = DEFAULT_STANDALONE_EMBED_DIM
    v_rank: int = DEFAULT_STANDALONE_VALUE_BRANCHES
    attention_mode: QuantumAttentionMode = QuantumAttentionMode.UHLMANN_KRYLOV_320
    readout_mode: VocabularyReadoutMode = VocabularyReadoutMode.RANK2_KRYLOV_BORN
    vocab_optim_mode: VocabOptimizerMode = VocabOptimizerMode.BATCH_SUBSPACE_GRAM_MUON


def _evolve_head_unitary_functional(
    rho_re: torch.Tensor,
    rho_im: torch.Tensor,
    head_u_re: torch.Tensor,
    head_u_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-head complex unitary conjugation $U_h \\rho_{t,h} U_h^\\dagger$ where $U_h \\in U(d_h)$."""
    u_re = head_u_re.to(rho_re.dtype).unsqueeze(0).unsqueeze(2)
    u_im = head_u_im.to(rho_re.dtype).unsqueeze(0).unsqueeze(2)
    ut_re = u_re.transpose(-2, -1)
    ut_im = -u_im.transpose(-2, -1)

    tmp_re = torch.matmul(u_re, rho_re) - torch.matmul(u_im, rho_im)
    tmp_im = torch.matmul(u_re, rho_im) + torch.matmul(u_im, rho_re)
    out_re = torch.matmul(tmp_re, ut_re) - torch.matmul(tmp_im, ut_im)
    out_im = torch.matmul(tmp_re, ut_im) + torch.matmul(tmp_im, ut_re)
    return 0.5 * (out_re + out_re.transpose(-2, -1)), 0.5 * (out_im - out_im.transpose(-2, -1))


def _causal_born_p3_weights_impl(ip_re: torch.Tensor, ip_im: torch.Tensor) -> torch.Tensor:
    """Compute causal $p=3$ polynomial Born coincidence weights $|\\langle q_t \\mid k_s\\rangle|^6$."""
    seq_len = ip_re.size(-1)
    born_mod_sq = (ip_re.float().square() + ip_im.float().square()).clamp_min(0.0)
    born_scores = born_mod_sq * born_mod_sq.square()
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=ip_re.device, dtype=torch.bool))
    born_scores = born_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), 0.0)
    return (born_scores / born_scores.sum(dim=-1, keepdim=True).clamp_min(TRACE_NORM_EPS)).to(ip_re.dtype)


def _mcweeny_uhlmann_krylov_step_impl(
    packed_288: torch.Tensor,
    head_u_re: torch.Tensor,
    head_u_im: torch.Tensor,
    re_pack_mask: torch.Tensor,
    im_pack_mask: torch.Tensor,
    eps: float = DEFAULT_UHLMANN_PHASE_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack attention output, run McWeeny + $U(16)$ evolution, and extract Uhlmann Rank-2 Krylov features."""
    work_dtype = packed_288.dtype
    d_h = head_u_re.size(-1)
    d_sq = d_h * d_h
    base_pack_dim = d_sq + 2 * d_h

    packed_rho0 = packed_288[..., :d_sq]
    v_bar_re = packed_288[..., d_sq : d_sq + d_h]
    v_bar_im = packed_288[..., d_sq + d_h : base_pack_dim]

    rho0_re, rho0_im = unpack_hermitian_isometric(packed_rho0, d_h)
    rho1_re, rho1_im, purity, eff_rank = bures_wasserstein_mcweeny_step(
        rho0_re, rho0_im, num_steps=DEFAULT_STANDALONE_MCWEENY_STEPS, eps=TRACE_NORM_EPS
    )
    rho_out_re, rho_out_im = _evolve_head_unitary_functional(rho1_re, rho1_im, head_u_re, head_u_im)

    u_re = head_u_re.to(work_dtype).unsqueeze(0).unsqueeze(2)
    u_im = head_u_im.to(work_dtype).unsqueeze(0).unsqueeze(2)
    ut_re = u_re.transpose(-2, -1)
    ut_im = u_im.transpose(-2, -1)
    v_u_re = torch.matmul(v_bar_re.unsqueeze(-2), ut_re).squeeze(-2) - torch.matmul(v_bar_im.unsqueeze(-2), ut_im).squeeze(-2)
    v_u_im = torch.matmul(v_bar_re.unsqueeze(-2), ut_im).squeeze(-2) + torch.matmul(v_bar_im.unsqueeze(-2), ut_re).squeeze(-2)

    packed_rho_raw = (
        rho_out_re * re_pack_mask.to(work_dtype) + rho_out_im * im_pack_mask.to(work_dtype)
    ).flatten(start_dim=-2, end_dim=-1)
    packed_rho_out = F.rms_norm(packed_rho_raw, (d_sq,), eps=TRACE_NORM_EPS)

    lam1, lam2, phi1_re, phi1_im, phi2_re, phi2_im = extract_rank2_krylov_eigenpair_uhlmann(
        rho_out_re, rho_out_im, v_u_re, v_u_im, eps=eps
    )

    s1 = torch.sqrt(lam1.clamp_min(eps)).unsqueeze(-1).to(work_dtype)
    s2 = torch.sqrt(lam2.clamp_min(eps)).unsqueeze(-1).to(work_dtype)
    krylov_raw = torch.cat(
        [s1 * phi1_re.to(work_dtype), s1 * phi1_im.to(work_dtype), s2 * phi2_re.to(work_dtype), s2 * phi2_im.to(work_dtype)],
        dim=-1,
    )
    krylov_64 = F.rms_norm(krylov_raw, (4 * d_h,), eps=TRACE_NORM_EPS)

    if packed_288.size(-1) > base_pack_dim:
        psi_tail = packed_288[..., base_pack_dim:]
        v_branch_dim = psi_tail.size(-1)
        v_rank = v_branch_dim // (2 * d_h)
        psi_bar = psi_tail.view(*psi_tail.shape[:-1], v_rank, 2, d_h)
        psi_u_re = torch.matmul(psi_bar[..., 0, :], ut_re) - torch.matmul(psi_bar[..., 1, :], ut_im)
        psi_u_im = torch.matmul(psi_bar[..., 0, :], ut_im) + torch.matmul(psi_bar[..., 1, :], ut_re)
        psi_u_flat = F.rms_norm(
            torch.cat([psi_u_re, psi_u_im], dim=-1).flatten(start_dim=-2, end_dim=-1),
            (v_branch_dim,),
            eps=TRACE_NORM_EPS,
        )
        c_dim = min(v_branch_dim, d_sq)
        rho_plus_psi = torch.cat(
            [
                INV_SQRT2_ISOMETRIC_SCALE * (packed_rho_out[..., :c_dim] + psi_u_flat[..., :c_dim]),
                packed_rho_out[..., c_dim:],
            ],
            dim=-1,
        )
        obs_320 = torch.cat([rho_plus_psi, krylov_64], dim=-1)
    else:
        obs_320 = torch.cat([packed_rho_out, krylov_64], dim=-1)
    return obs_320, packed_rho_out, lam1, lam2, purity, eff_rank


_COMPILED_KRYLOV_HELPER: Any = None
_COMPILED_BORN_WEIGHTS_HELPER: Any = None


def _compiled_mcweeny_uhlmann_krylov_step(
    packed_288: torch.Tensor,
    head_u_re: torch.Tensor,
    head_u_im: torch.Tensor,
    re_pack_mask: torch.Tensor,
    im_pack_mask: torch.Tensor,
    eps: float = DEFAULT_UHLMANN_PHASE_EPS,
    use_compile: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute `_mcweeny_uhlmann_krylov_step_impl` with optional `torch.compile` kernel fusion."""
    global _COMPILED_KRYLOV_HELPER
    if use_compile and packed_288.is_cuda:
        if _COMPILED_KRYLOV_HELPER is None:
            _COMPILED_KRYLOV_HELPER = torch.compile(_mcweeny_uhlmann_krylov_step_impl, dynamic=False)
        return _COMPILED_KRYLOV_HELPER(packed_288, head_u_re, head_u_im, re_pack_mask, im_pack_mask, eps)
    return _mcweeny_uhlmann_krylov_step_impl(packed_288, head_u_re, head_u_im, re_pack_mask, im_pack_mask, eps)


class StandaloneQuantumBornBlock(nn.Module):
    """Standalone Teacher-Free Quantum Layer with Fused 288-Width Packing and Direct Stiefel Readouts (`Wave 48`)."""

    def __init__(self, cfg: StandaloneQuantumBornConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.embed_dim = cfg.n_embd
        self.num_heads = cfg.n_head
        self.d_h = DEFAULT_STANDALONE_HEAD_COMPLEX_DIM
        self.v_rank = max(1, int(cfg.v_rank))
        self.fused_pack_dim = self.d_h * self.d_h + 2 * self.d_h
        self.obs_head_dim = (
            DEFAULT_KRYLOV_OBSERVABLE_HEAD_DIM
            if cfg.attention_mode == QuantumAttentionMode.UHLMANN_KRYLOV_320
            else (self.d_h * self.d_h)
        )
        self.total_obs_dim = self.num_heads * self.obs_head_dim
        self.qk_complex_dim = self.v_rank * self.d_h
        self.ket_real_dim = self.num_heads * (2 * self.qk_complex_dim)
        self.v_real_dim = self.num_heads * self.v_rank * (2 * self.d_h)
        self.channel_hidden_dim = self.embed_dim * 2
        self.is_long_window_layer = (self.layer_idx % 4 == 3) or (self.layer_idx == cfg.n_layer - 1)
        self.use_compiled_krylov: bool = False

        self.block_lambdas = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float32))
        v0_init = DEFAULT_V0_HIGHWAY_INIT_LAMBDA if self.layer_idx > 0 else 0.0
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

        init_gain = DEFAULT_RESIDUAL_GAIN_COEFF / math.sqrt(float(max(1, cfg.n_layer)))
        self.w_obs = nn.Parameter(torch.empty(self.total_obs_dim, self.embed_dim))
        initialize_stiefel_parameter(self.w_obs)
        self.obs_scale = nn.Parameter(torch.tensor(init_gain, dtype=torch.float32))

        self.w_beam_a = nn.Parameter(torch.empty(self.embed_dim, self.channel_hidden_dim))
        self.w_beam_b = nn.Parameter(torch.empty(self.embed_dim, self.channel_hidden_dim))
        self.w_chan = nn.Parameter(torch.empty(self.channel_hidden_dim, self.embed_dim))
        initialize_stiefel_parameter(self.w_beam_a)
        initialize_stiefel_parameter(self.w_beam_b)
        initialize_stiefel_parameter(self.w_chan)
        self.chan_scale = nn.Parameter(torch.tensor(init_gain, dtype=torch.float32))

        inv_freq = 1.0 / (
            DEFAULT_ROPE_BASE_THETA
            ** (torch.arange(0, self.qk_complex_dim, dtype=torch.float32) / float(self.qk_complex_dim))
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

        re_mask = torch.diag(torch.ones(self.d_h)) + torch.triu(
            torch.full((self.d_h, self.d_h), SQRT2_ISOMETRIC_OFFDIAG_SCALE), diagonal=1
        )
        im_mask = torch.tril(torch.full((self.d_h, self.d_h), -SQRT2_ISOMETRIC_OFFDIAG_SCALE), diagonal=-1)
        self.register_buffer("re_pack_mask", re_mask, persistent=False)
        self.register_buffer("im_pack_mask", im_mask, persistent=False)
        self.last_telemetry: dict[str, float] = {
            "mean_purity": 1.0,
            "mean_eff_rank": 1.0,
            "mean_lambda1": 0.5,
            "mean_lambda2": 0.5,
        }

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Return all 7 direct 2D real Stiefel matrices in this layer (`W^T W = I` or `W W^T = I`)."""
        return [self.w_q, self.w_k, self.w_v, self.w_obs, self.w_beam_a, self.w_beam_b, self.w_chan]

    def complex_unitary_pair(self) -> tuple[nn.Parameter, nn.Parameter]:
        """Return the 3D complex head unitary pair `(head_unitary_re, head_unitary_im)` $\\in U(d_h)^H$."""
        return self.head_unitary_re, self.head_unitary_im

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return 1D scalar residual gains and highway weights."""
        return [self.obs_scale, self.chan_scale, self.block_lambdas, self.v_lambdas]

    def _apply_complex_rope(self, z_re: torch.Tensor, z_im: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply complex rotary phase $e^{i t \\omega_k}$ to `(z_re, z_im)` `(B, H, T, d_h)`."""
        seq_len = z_re.size(-2)
        pos = torch.arange(seq_len, device=z_re.device, dtype=torch.float32)
        angles = torch.outer(pos, self.rope_inv_freq).unsqueeze(0).unsqueeze(0)
        cos_a = torch.cos(angles).to(z_re.dtype)
        sin_a = torch.sin(angles).to(z_re.dtype)
        return z_re * cos_a - z_im * sin_a, z_re * sin_a + z_im * cos_a

    def compute_quantum_attention_features(
        self,
        x_base: torch.Tensor,
        v_0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute `(obs_concat, v_0_next, krylov_weights_bt2)` via fused 288-width Born attention."""
        bsz, seq_len, _ = x_base.shape
        work_dtype = x_base.dtype
        h_norm = F.rms_norm(x_base, (self.embed_dim,), eps=TRACE_NORM_EPS)

        q_raw = torch.matmul(h_norm, self.w_q.to(work_dtype)).view(
            bsz, seq_len, self.num_heads, 2, self.qk_complex_dim
        )
        k_raw = torch.matmul(h_norm, self.w_k.to(work_dtype)).view(
            bsz, seq_len, self.num_heads, 2, self.qk_complex_dim
        )
        v_proj = torch.matmul(h_norm, self.w_v.to(work_dtype))

        if v_0 is None:
            v_0_next = v_proj
            v_mixed = v_proj
        else:
            v_0_next = v_0
            if v_0.shape == v_proj.shape:
                v_mixed = self.v_lambdas[0].to(work_dtype) * v_proj + self.v_lambdas[1].to(work_dtype) * v_0
            else:
                v_mixed = v_proj

        q_re, q_im = q_raw[..., 0, :].transpose(1, 2), q_raw[..., 1, :].transpose(1, 2)
        k_re, k_im = k_raw[..., 0, :].transpose(1, 2), k_raw[..., 1, :].transpose(1, 2)
        q_re, q_im = complex_l2_normalize(q_re, q_im)
        k_re, k_im = complex_l2_normalize(k_re, k_im)
        q_re, q_im = self._apply_complex_rope(q_re, q_im)
        k_re, k_im = self._apply_complex_rope(k_re, k_im)

        if self.is_long_window_layer and seq_len > 1:
            k_re = apply_partial_key_offset(k_re.transpose(1, 2)).transpose(1, 2)
            k_im = apply_partial_key_offset(k_im.transpose(1, 2)).transpose(1, 2)

        q_cat = torch.cat([q_re, q_im], dim=-1)
        k_cat = torch.cat([k_re, k_im], dim=-1)
        k_skew = torch.cat([-k_im, k_re], dim=-1)
        ip_re = torch.matmul(q_cat, k_cat.transpose(-2, -1))
        ip_im = torch.matmul(q_cat, k_skew.transpose(-2, -1))
        global _COMPILED_BORN_WEIGHTS_HELPER
        if self.use_compiled_krylov and ip_re.is_cuda:
            if _COMPILED_BORN_WEIGHTS_HELPER is None:
                _COMPILED_BORN_WEIGHTS_HELPER = torch.compile(_causal_born_p3_weights_impl, dynamic=False)
            attn_weights = _COMPILED_BORN_WEIGHTS_HELPER(ip_re, ip_im)
        else:
            attn_weights = _causal_born_p3_weights_impl(ip_re, ip_im)

        v_raw = v_mixed.view(bsz, seq_len, self.num_heads, self.v_rank, 2, self.d_h)
        v_re = v_raw[..., 0, :].permute(0, 2, 1, 3, 4)
        v_im = v_raw[..., 1, :].permute(0, 2, 1, 3, 4)
        v_branch_dim = self.v_rank * 2 * self.d_h
        v_branch_flat = torch.cat([v_re, v_im], dim=-1).reshape(bsz, self.num_heads, seq_len, v_branch_dim)

        packed_tok_288 = fast_rank_r_hermitian_and_ket_pack(
            v_re, v_im, self.re_pack_mask, self.im_pack_mask, eps=TRACE_NORM_EPS
        )
        packed_ctx_all = torch.matmul(attn_weights, torch.cat([packed_tok_288, v_branch_flat], dim=-1))

        obs_320, packed_rho_256, lam1, lam2, purity, eff_rank = _compiled_mcweeny_uhlmann_krylov_step(
            packed_ctx_all,
            self.head_unitary_re,
            self.head_unitary_im,
            self.re_pack_mask,
            self.im_pack_mask,
            eps=DEFAULT_UHLMANN_PHASE_EPS,
            use_compile=self.use_compiled_krylov,
        )

        with torch.no_grad():
            self.last_telemetry = {
                "mean_purity": float(purity.mean().item()),
                "mean_eff_rank": float(eff_rank.mean().item()),
                "mean_lambda1": float(lam1.mean().item()),
                "mean_lambda2": float(lam2.mean().item()),
            }

        lam1_bar, lam2_bar = lam1.mean(dim=1), lam2.mean(dim=1)
        lam_sum = (lam1_bar + lam2_bar).clamp_min(DEFAULT_UHLMANN_PHASE_EPS)
        krylov_weights_bt2 = torch.stack([lam1_bar / lam_sum, lam2_bar / lam_sum], dim=-1)

        if self.cfg.attention_mode == QuantumAttentionMode.UHLMANN_KRYLOV_320:
            head_features = obs_320
        else:
            head_features = packed_rho_256

        obs_concat = head_features.to(work_dtype).transpose(1, 2).reshape(bsz, seq_len, self.total_obs_dim)
        return obs_concat, v_0_next, krylov_weights_bt2

    def compute_channel_features(self, x_mid: torch.Tensor) -> torch.Tensor:
        """Compute Stiefel two-beam Mach-Zehnder quantum interference features `(B, T, channel_hidden_dim)`."""
        work_dtype = x_mid.dtype
        m_norm = F.rms_norm(x_mid, (self.embed_dim,), eps=TRACE_NORM_EPS)
        beam_a = torch.matmul(m_norm, self.w_beam_a.to(work_dtype))
        beam_b = torch.matmul(m_norm, self.w_beam_b.to(work_dtype))
        return F.relu(beam_a + beam_b).square() - 0.25 * F.relu(beam_a - beam_b).square()

    def forward(
        self,
        x: torch.Tensor,
        x_0: torch.Tensor | None = None,
        v_0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning `(x_out, v_0_next, krylov_weights_bt2)`."""
        work_dtype = x.dtype
        if x_0 is not None:
            x_cur = self.block_lambdas[0].to(work_dtype) * x + self.block_lambdas[1].to(work_dtype) * x_0.to(work_dtype)
        else:
            x_cur = x

        obs_concat, v_0_next, krylov_weights_bt2 = self.compute_quantum_attention_features(x_cur, v_0=v_0)
        attn_delta = torch.matmul(obs_concat, self.w_obs.to(work_dtype)) * self.obs_scale.to(work_dtype)
        x_mid = x_cur + attn_delta

        interference = self.compute_channel_features(x_mid)
        chan_delta = torch.matmul(interference, self.w_chan.to(work_dtype)) * self.chan_scale.to(work_dtype)
        x_out = x_mid + chan_delta
        return x_out, v_0_next, krylov_weights_bt2


class Rank2KrylovBornChunkedCEFn(torch.autograd.Function):
    """Rule-9 single-pass chunked Rank-2 Krylov Born log-intensity + tanh-softcap cross-entropy autograd function.

    Computes all forward losses and analytical gradients `(dh, d_krylov_weights, dw_lm, d_tau)` inside
    a single chunked pass over tokens so `backward()` performs 0 matmuls and 0 `.pow()` calls.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        h_flat: torch.Tensor,
        krylov_weights_flat: torch.Tensor,
        w_lm: torch.Tensor,
        targets_flat: torch.Tensor,
        tau_born: torch.Tensor,
        softcap: float = DEFAULT_LOGIT_SOFTCAP,
        chunk_tokens: int = DEFAULT_CE_CHUNK_TOKENS,
    ) -> torch.Tensor:
        with torch.no_grad():
            n_tok, d_model = h_flat.shape
            d_sub = d_model // 2
            inv_sqrt_sub = 1.0 / math.sqrt(float(max(1, d_sub)))
            flat_y = targets_flat.reshape(-1).long()
            chunk_sz = max(1, min(int(chunk_tokens), n_tok))

            work_dtype = (
                torch.bfloat16
                if (h_flat.is_cuda and (h_flat.dtype == torch.bfloat16 or torch.is_autocast_enabled()))
                else h_flat.dtype
            )
            h_work = h_flat if h_flat.dtype == work_dtype else h_flat.to(work_dtype)
            w_work = w_lm if w_lm.dtype == work_dtype else w_lm.to(work_dtype)
            kw_f32 = krylov_weights_flat.float()

            h1_all = h_work[:, :d_sub] * inv_sqrt_sub
            h2_all = h_work[:, d_sub:] * inv_sqrt_sub
            w1 = w_work[:, :d_sub]
            w2 = w_work[:, d_sub:]

            dh1 = torch.empty_like(h1_all)
            dh2 = torch.empty_like(h2_all)
            dkw = torch.empty_like(kw_f32)
            dw1 = torch.zeros_like(w1)
            dw2 = torch.zeros_like(w2)
            d_tau = torch.zeros((), device=h_flat.device, dtype=torch.float32)
            total_loss = torch.zeros((), device=h_flat.device, dtype=torch.float32)

            inv_n_tok = 1.0 / float(max(1, n_tok))
            softcap_f = float(softcap)
            inv_softcap = 1.0 / softcap_f
            tau_f = float(tau_born.float().item())

            for start in range(0, n_tok, chunk_sz):
                end = min(start + chunk_sz, n_tok)
                h1_c = h1_all[start:end]
                h2_c = h2_all[start:end]
                kw_c = kw_f32[start:end]
                yc = flat_y[start:end]

                coh1 = torch.matmul(h1_c, w1.t()).float().mul_(0.5).add_(1.0)
                coh2 = torch.matmul(h2_c, w2.t()).float().mul_(0.5).add_(1.0)
                a1_sq = coh1.square()
                a2_sq = coh2.square()

                lam1_c = kw_c[:, 0:1]
                lam2_c = kw_c[:, 1:2]
                intensity = lam1_c * a1_sq + lam2_c * a2_sq + DEFAULT_BORN_LOG_INTENSITY_EPS
                log_int = torch.log(intensity)
                u_val = torch.tanh((tau_f * inv_softcap) * log_int)
                z_logits = u_val * softcap_f

                lse = torch.logsumexp(z_logits, dim=-1, keepdim=True)
                target_z = z_logits.gather(1, yc.unsqueeze(1))
                total_loss.add_((lse - target_z).sum() * inv_n_tok)

                z_logits.sub_(lse).exp_()
                z_logits.scatter_add_(1, yc.unsqueeze(1), torch.full_like(target_z, -1.0))
                z_logits.mul_(inv_n_tok)
                u_val.square_().neg_().add_(1.0)
                grad_pre_tanh = z_logits.mul_(u_val)
                d_tau.add_((grad_pre_tanh * log_int).sum())
                del log_int, u_val

                grad_int = grad_pre_tanh.mul_(tau_f).div_(intensity)
                del intensity
                dkw[start:end, 0] = (grad_int * a1_sq).sum(dim=-1)
                dkw[start:end, 1] = (grad_int * a2_sq).sum(dim=-1)
                del a1_sq, a2_sq

                coh1.mul_(grad_int).mul_(lam1_c)
                coh2.mul_(grad_int).mul_(lam2_c)
                del grad_int
                ga1 = coh1.to(work_dtype)
                ga2 = coh2.to(work_dtype)
                del coh1, coh2

                dh1[start:end] = torch.matmul(ga1, w1) * inv_sqrt_sub
                dh2[start:end] = torch.matmul(ga2, w2) * inv_sqrt_sub
                dw1.add_(torch.matmul(ga1.t(), h1_c))
                dw2.add_(torch.matmul(ga2.t(), h2_c))

            dh_full = torch.cat([dh1, dh2], dim=-1).to(h_flat.dtype)
            dw_full = torch.cat([dw1, dw2], dim=-1).to(w_lm.dtype)
            ctx.save_for_backward(dh_full, dkw.to(krylov_weights_flat.dtype), dw_full, d_tau.to(tau_born.dtype))
            return total_loss

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, torch.Tensor, None, None]:
        dh_full, dkw, dw_full, d_tau = ctx.saved_tensors
        scale_h = grad_output.to(dh_full.dtype)
        scale_kw = grad_output.to(dkw.dtype)
        scale_w = grad_output.to(dw_full.dtype)
        scale_tau = grad_output.to(d_tau.dtype)
        return dh_full * scale_h, dkw * scale_kw, dw_full * scale_w, None, d_tau * scale_tau, None, None


def eval_rank2_krylov_born_logits(
    h_bt: torch.Tensor,
    krylov_weights_bt: torch.Tensor,
    w_lm: torch.Tensor,
    tau_born: torch.Tensor,
    softcap: float = DEFAULT_LOGIT_SOFTCAP,
) -> torch.Tensor:
    """Evaluate Rank-2 Krylov Born softcapped logits `(B, T, V)` for validation or generation."""
    d_model = h_bt.size(-1)
    d_sub = d_model // 2
    inv_sqrt_sub = 1.0 / math.sqrt(float(max(1, d_sub)))
    work_dtype = h_bt.dtype
    w_work = w_lm.to(work_dtype)

    h1 = h_bt[..., :d_sub] * inv_sqrt_sub
    h2 = h_bt[..., d_sub:] * inv_sqrt_sub
    a1 = F.linear(h1, w_work[:, :d_sub]).float()
    a2 = F.linear(h2, w_work[:, d_sub:]).float()
    coh1 = 1.0 + 0.5 * a1
    coh2 = 1.0 + 0.5 * a2

    lam1 = krylov_weights_bt[..., 0:1].float()
    lam2 = krylov_weights_bt[..., 1:2].float()
    intensity = lam1 * coh1.square() + lam2 * coh2.square() + DEFAULT_BORN_LOG_INTENSITY_EPS
    tau_f = tau_born.float()
    raw_logits = tau_f * torch.log(intensity)
    return float(softcap) * torch.tanh(raw_logits / float(softcap))


class AllLayerTikhonovRiemannianStiefelMuon(torch.optim.Optimizer):
    """Tikhonov-damped `RiemannianStiefelMuon` optimizer for all 2D real Stiefel and 3D $U(d_h)$ unitary parameters."""

    def __init__(
        self,
        real_stiefel_params: list[nn.Parameter],
        complex_unitary_pairs: list[tuple[nn.Parameter, nn.Parameter]],
        stiefel_lr: float = DEFAULT_STIEFEL_GEODESIC_LR,
        unitary_lr: float = DEFAULT_UNITARY_GEODESIC_LR,
        beta: float = DEFAULT_RIEMANNIAN_MOMENTUM_BETA,
        ns_steps: int = DEFAULT_NS5_STEPS,
        tikhonov_lambda: float = DEFAULT_TIKHONOV_GRAM_LAMBDA,
    ) -> None:
        self.complex_unitary_pairs = list(complex_unitary_pairs)
        all_params = list(real_stiefel_params)
        for u_re, u_im in self.complex_unitary_pairs:
            all_params.extend([u_re, u_im])
        defaults = {
            "lr": float(stiefel_lr),
            "unitary_lr": float(unitary_lr),
            "beta": float(beta),
            "ns_steps": int(ns_steps),
            "tikhonov_lambda": float(tikhonov_lambda),
        }
        super().__init__(all_params, defaults)
        self._unitary_ids = {id(u_re) for u_re, _ in self.complex_unitary_pairs} | {
            id(u_im) for _, u_im in self.complex_unitary_pairs
        }

    def set_lr_scale(self, lr_mult: float) -> None:
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
        tik_lam = float(group["tikhonov_lambda"])

        for p in group["params"]:
            if id(p) in self._unitary_ids or p.grad is None:
                continue
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
            riemannian_stiefel_muon_step(
                p, p.grad, state["momentum_buffer"], lr=stiefel_lr, beta=beta, ns_steps=ns_steps, tikhonov_lambda=tik_lam
            )

        for u_re, u_im in self.complex_unitary_pairs:
            if u_re.grad is None or u_im.grad is None:
                continue
            st_re = self.state[u_re]
            st_im = self.state[u_im]
            if "momentum_buffer" not in st_re:
                st_re["momentum_buffer"] = torch.zeros_like(u_re, dtype=torch.float32)
            if "momentum_buffer" not in st_im:
                st_im["momentum_buffer"] = torch.zeros_like(u_im, dtype=torch.float32)
            riemannian_complex_unitary_muon_step(
                u_re, u_im, u_re.grad, u_im.grad, st_re["momentum_buffer"], st_im["momentum_buffer"], lr=unitary_lr, tikhonov_lambda=tik_lam
            )

        return loss


class StandaloneQuantumBornGPT(nn.Module):
    """Standalone Teacher-Free $L$-Layer Complex Quantum Born Language Model (`Wave 48` / `EXPERIMENT_PLAN_063.md`)."""

    def __init__(self, cfg: StandaloneQuantumBornConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else StandaloneQuantumBornConfig()
        self.vocab_size = self.cfg.vocab_size
        self.embed_dim = self.cfg.n_embd
        self.num_layers = self.cfg.n_layer

        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)
        with torch.no_grad():
            nn.init.normal_(self.wte.weight, mean=0.0, std=1.0 / math.sqrt(float(self.embed_dim)))
            project_rows_to_rms_sphere_(self.wte.weight)

        self.w_in_stiefel = nn.Parameter(torch.eye(self.embed_dim, dtype=torch.float32))
        self.in_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.blocks = nn.ModuleList(
            [StandaloneQuantumBornBlock(self.cfg, layer_idx=l_idx) for l_idx in range(self.num_layers)]
        )

        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)
        with torch.no_grad():
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=1.0 / math.sqrt(float(self.embed_dim)))
            project_rows_to_rms_sphere_(self.lm_head.weight)

        self.logit_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(float(self.embed_dim)), dtype=torch.float32))
        self.born_tau = nn.Parameter(torch.tensor(DEFAULT_KRYLOV_BORN_LOG_TEMP, dtype=torch.float32))

    def set_compiled_krylov(self, enabled: bool = True) -> None:
        """Enable or disable `torch.compile` fusion on the per-layer McWeeny + Uhlmann-Krylov step."""
        for blk in self.blocks:
            blk.use_compiled_krylov = bool(enabled)

    def all_real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Collect all 2D real Stiefel parameters across `w_in_stiefel` and all $L$ blocks (`1 + 7L` total)."""
        params: list[nn.Parameter] = [self.w_in_stiefel]
        for blk in self.blocks:
            params.extend(blk.real_stiefel_parameters())
        return params

    def all_complex_unitary_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        """Collect all 3D complex head unitary pairs `(head_unitary_re, head_unitary_im)` across all $L$ blocks."""
        return [blk.complex_unitary_pair() for blk in self.blocks]

    def all_scalar_parameters(self) -> list[nn.Parameter]:
        """Collect all 1D scalar gain, residual highway, and temperature parameters."""
        params: list[nn.Parameter] = [self.in_scale, self.logit_scale, self.born_tau]
        for blk in self.blocks:
            params.extend(blk.scalar_parameters())
        return params

    @torch.no_grad()
    def compute_manifold_and_purity_telemetry(self) -> dict[str, float]:
        """Measure Stiefel error, $U(d_h)$ unitary error, sphere RMS error, purity, and Krylov Rayleigh quotients."""
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

        target_radius = math.sqrt(float(self.embed_dim))
        wte_rms_err = float((self.wte.weight.float().norm(dim=-1) / target_radius - 1.0).abs().max().item())
        head_rms_err = float((self.lm_head.weight.float().norm(dim=-1) / target_radius - 1.0).abs().max().item())

        purities = [blk.last_telemetry.get("mean_purity", 1.0) for blk in self.blocks]
        eff_ranks = [blk.last_telemetry.get("mean_eff_rank", 1.0) for blk in self.blocks]
        lam1s = [blk.last_telemetry.get("mean_lambda1", 0.5) for blk in self.blocks]
        lam2s = [blk.last_telemetry.get("mean_lambda2", 0.5) for blk in self.blocks]
        return {
            "mean_stiefel_err": float(sum(stiefel_errs) / max(1, len(stiefel_errs))),
            "max_stiefel_err": float(max(stiefel_errs) if stiefel_errs else 0.0),
            "mean_unitary_err": float(sum(unitary_errs) / max(1, len(unitary_errs))),
            "max_wte_sphere_rms_err": wte_rms_err,
            "max_head_sphere_rms_err": head_rms_err,
            "mean_purity": float(sum(purities) / max(1, len(purities))),
            "mean_eff_rank": float(sum(eff_ranks) / max(1, len(eff_ranks))),
            "mean_lambda1": float(sum(lam1s) / max(1, len(lam1s))),
            "mean_lambda2": float(sum(lam2s) / max(1, len(lam2s))),
        }

    def forward_hidden(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute final RMS-normalized hidden states `(B, T, d)` and final-block Krylov weights `(B, T, 2)`."""
        work_dtype = torch.bfloat16 if (idx.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
        tok_emb = F.rms_norm(self.wte(idx).to(work_dtype), (self.embed_dim,), eps=TRACE_NORM_EPS)
        x = torch.matmul(tok_emb, self.w_in_stiefel.to(work_dtype)) * self.in_scale.to(work_dtype)
        x_0 = x
        v_0: torch.Tensor | None = None
        krylov_weights_bt2 = torch.full(
            (idx.size(0), idx.size(1), 2),
            0.5,
            device=idx.device,
            dtype=torch.float32,
        )

        for blk in self.blocks:
            x, v_0, krylov_weights_bt2 = blk(x, x_0=x_0, v_0=v_0)

        h_final = F.rms_norm(x, (self.embed_dim,), eps=TRACE_NORM_EPS)
        return h_final, krylov_weights_bt2

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Forward pass supporting `RANK2_KRYLOV_BORN` and `SOFTCAP_LINEAR_CE` readout modes."""
        h_final, krylov_weights_bt2 = self.forward_hidden(idx)
        loss: torch.Tensor | None = None
        logits: torch.Tensor | None = None

        if self.cfg.readout_mode == VocabularyReadoutMode.RANK2_KRYLOV_BORN:
            if targets is not None:
                loss = Rank2KrylovBornChunkedCEFn.apply(
                    h_final.reshape(-1, self.embed_dim),
                    krylov_weights_bt2.reshape(-1, 2),
                    self.lm_head.weight,
                    targets.reshape(-1),
                    self.born_tau,
                    DEFAULT_LOGIT_SOFTCAP,
                    DEFAULT_CE_CHUNK_TOKENS,
                )
            if return_logits or targets is None:
                logits = eval_rank2_krylov_born_logits(
                    h_final,
                    krylov_weights_bt2,
                    self.lm_head.weight,
                    self.born_tau,
                    softcap=DEFAULT_LOGIT_SOFTCAP,
                )
        else:
            min_scale = 0.75 / math.sqrt(float(self.embed_dim))
            eff_scale = self.logit_scale.clamp(min=min_scale, max=8.0 * min_scale).to(self.lm_head.weight.dtype)
            w_lm_scaled = self.lm_head.weight * eff_scale
            if targets is not None:
                loss = _CompiledChunkedSoftcapLinearCEFn.apply(
                    h_final.reshape(-1, self.embed_dim),
                    w_lm_scaled,
                    targets.reshape(-1),
                    DEFAULT_LOGIT_SOFTCAP,
                    DEFAULT_CE_CHUNK_TOKENS,
                )
            if return_logits or targets is None:
                raw_logits = F.linear(h_final, w_lm_scaled.to(h_final.dtype)).float()
                logits = DEFAULT_LOGIT_SOFTCAP * torch.tanh(raw_logits / DEFAULT_LOGIT_SOFTCAP)

        return logits, loss


class StandaloneRiemannianOptimizerBundle:
    """Unified optimizer bundle coordinating Stiefel/Unitary trunk, active-token vocabulary manifold, and 1D scalars."""

    def __init__(
        self,
        trunk_opt: AllLayerTikhonovRiemannianStiefelMuon,
        scalar_opt: torch.optim.Optimizer,
        vocab_optim_mode: VocabOptimizerMode,
        wte_opt: BatchSubspaceTikhonovGramMuon | RiemannianSphereTangentSNROptimizer | None = None,
        head_opt: BatchSubspaceTikhonovGramMuon | RiemannianSphereTangentSNROptimizer | None = None,
    ) -> None:
        self.trunk_opt = trunk_opt
        self.scalar_opt = scalar_opt
        self.vocab_optim_mode = vocab_optim_mode
        self.wte_opt = wte_opt
        self.head_opt = head_opt
        self._base_scalar_lrs = [float(g["lr"]) for g in self.scalar_opt.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients across all underlying optimizers."""
        self.trunk_opt.zero_grad(set_to_none=set_to_none)
        self.scalar_opt.zero_grad(set_to_none=set_to_none)
        if self.wte_opt is not None:
            self.wte_opt.zero_grad(set_to_none=set_to_none)
        if self.head_opt is not None:
            self.head_opt.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(
        self,
        idx: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        lr_scale: float = 1.0,
    ) -> None:
        """Step all trunk Stiefel/Unitary parameters, active vocabulary rows, and 1D scalar parameters."""
        self.trunk_opt.set_lr_scale(lr_scale)
        self.trunk_opt.step()

        for base_lr, group in zip(self._base_scalar_lrs, self.scalar_opt.param_groups):
            group["lr"] = base_lr * float(lr_scale)
        self.scalar_opt.step()

        if self.wte_opt is not None and idx is not None:
            idx_in = torch.unique(idx.reshape(-1))
            self.wte_opt.step_active(idx_in, lr_scale=lr_scale)
        elif self.wte_opt is not None:
            self.wte_opt.step()

        if self.head_opt is not None:
            if idx is not None and targets is not None:
                act_ids = torch.unique(torch.cat([idx.reshape(-1), targets.reshape(-1)]))
                self.head_opt.step_active(act_ids, lr_scale=lr_scale * 0.75)
                self.head_opt.step_full(lr_scale=lr_scale * 0.25)
            else:
                self.head_opt.step_full(lr_scale=lr_scale)


def build_standalone_quantum_born_optimizers(
    model: StandaloneQuantumBornGPT,
    stiefel_lr: float = DEFAULT_STIEFEL_GEODESIC_LR,
    unitary_lr: float = DEFAULT_UNITARY_GEODESIC_LR,
    vocab_lr: float | None = None,
    scalar_lr: float = DEFAULT_SCALAR_ADAM_LR,
    tikhonov_lambda: float = DEFAULT_TIKHONOV_GRAM_LAMBDA,
) -> StandaloneRiemannianOptimizerBundle:
    """Build `StandaloneRiemannianOptimizerBundle` for `StandaloneQuantumBornGPT` (`<= 8` args, Rule 2)."""
    trunk_opt = AllLayerTikhonovRiemannianStiefelMuon(
        real_stiefel_params=model.all_real_stiefel_parameters(),
        complex_unitary_pairs=model.all_complex_unitary_pairs(),
        stiefel_lr=stiefel_lr,
        unitary_lr=unitary_lr,
        tikhonov_lambda=tikhonov_lambda,
    )

    scalar_groups: list[dict[str, Any]] = [
        {
            "params": [p for p in model.all_scalar_parameters() if p.requires_grad],
            "lr": float(scalar_lr),
            "weight_decay": 0.0,
        }
    ]
    wte_opt: BatchSubspaceTikhonovGramMuon | RiemannianSphereTangentSNROptimizer | None = None
    head_opt: BatchSubspaceTikhonovGramMuon | RiemannianSphereTangentSNROptimizer | None = None

    mode = model.cfg.vocab_optim_mode
    if mode == VocabOptimizerMode.BATCH_SUBSPACE_GRAM_MUON:
        eff_vocab_lr = float(vocab_lr) if vocab_lr is not None else DEFAULT_BATCH_GRAM_MUON_LR
        wte_opt = BatchSubspaceTikhonovGramMuon([model.wte.weight], lr=eff_vocab_lr, tikhonov_lambda=tikhonov_lambda)
        head_opt = BatchSubspaceTikhonovGramMuon([model.lm_head.weight], lr=eff_vocab_lr, tikhonov_lambda=tikhonov_lambda)
    elif mode == VocabOptimizerMode.SPHERE_TANGENT_SNR_MUON:
        eff_vocab_lr = float(vocab_lr) if vocab_lr is not None else DEFAULT_SPHERE_GEODESIC_LR
        wte_opt = RiemannianSphereTangentSNROptimizer([model.wte.weight], lr=eff_vocab_lr, snr_tau=DEFAULT_SPHERE_SNR_TRUST_TAU)
        head_opt = RiemannianSphereTangentSNROptimizer([model.lm_head.weight], lr=eff_vocab_lr, snr_tau=DEFAULT_SPHERE_SNR_TRUST_TAU)
    else:
        eff_vocab_lr = float(vocab_lr) if vocab_lr is not None else DEFAULT_ADAMW_VOCAB_LR
        d_scale = math.sqrt(float(model.embed_dim))
        head_mult = d_scale if model.cfg.readout_mode == VocabularyReadoutMode.SOFTCAP_LINEAR_CE else 4.0
        scalar_groups.append({"params": [model.wte.weight], "lr": eff_vocab_lr * d_scale, "weight_decay": 0.0})
        scalar_groups.append({"params": [model.lm_head.weight], "lr": eff_vocab_lr * head_mult, "weight_decay": 0.0})

    scalar_opt = torch.optim.AdamW(scalar_groups, betas=(0.9, 0.95), eps=1e-8)
    return StandaloneRiemannianOptimizerBundle(
        trunk_opt=trunk_opt, scalar_opt=scalar_opt, vocab_optim_mode=mode, wte_opt=wte_opt, head_opt=head_opt
    )


__all__ = [
    "DEFAULT_BORN_LOG_INTENSITY_EPS",
    "DEFAULT_FUSED_HEAD_PACK_DIM",
    "DEFAULT_KRYLOV_BORN_LOG_TEMP",
    "DEFAULT_KRYLOV_OBSERVABLE_HEAD_DIM",
    "DEFAULT_SPHERE_GEODESIC_LR",
    "DEFAULT_SPHERE_SNR_TRUST_TAU",
    "DEFAULT_TIKHONOV_GRAM_LAMBDA",
    "DEFAULT_UHLMANN_PHASE_EPS",
    "AllLayerTikhonovRiemannianStiefelMuon",
    "QuantumAttentionMode",
    "Rank2KrylovBornChunkedCEFn",
    "StandaloneQuantumBornBlock",
    "StandaloneQuantumBornConfig",
    "StandaloneQuantumBornGPT",
    "StandaloneRiemannianOptimizerBundle",
    "VocabOptimizerMode",
    "VocabularyReadoutMode",
    "_compiled_mcweeny_uhlmann_krylov_step",
    "build_standalone_quantum_born_optimizers",
    "eval_rank2_krylov_born_logits",
]
