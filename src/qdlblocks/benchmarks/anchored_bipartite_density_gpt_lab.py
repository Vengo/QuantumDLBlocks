"""Wave 71 (`EXPERIMENT_PLAN_091.md`): Sensory-Anchored Bipartite Pure Density-Matrix GPT.

Implements Theorems 12a, 12b, & 12c from `lean/QuantumDLBlocksLean/SensoryAnchoredBipartiteDensity.lean`:
1. Dual-Unitary Layer-0 Sensory Anchor (`Theorem 12b`): passes the immutable Layer-0 token density state
   $(L_{t,h}^{(0)}, \\tilde{\\rho}_{t,h}^{(0)})$ into every layer $\\ell = 1 \\dots L$ and projects it via dedicated
   horizontal unitaries $W_{Q,\\text{anc}}^{(\\ell)}, W_{K,\\text{anc}}^{(\\ell)}, W_{V,\\text{anc}}^{(\\ell)} \\in U(16)$
   initialized orthogonally to $W_{Q,\\text{evol}}^{(\\ell)}, W_{K,\\text{evol}}^{(\\ell)}, W_{V,\\text{evol}}^{(\\ell)}$,
   eliminating state collisions between observed token identity and evolved next-token predictions.
2. Bipartite Sensory/Predictive Register Split (`Theorem 12c`): partitions $L_{t,h}^{(\\ell)} \\in \\mathbb{C}^{16 \\times 4}$
   into a 2-mode sensory-tethered sub-register (`cols 0:2`) and a 2-mode predictive sub-register (`cols 2:4`).
3. Decoupled Input Preparation vs. Output Measurement Frames (`Theorem 12c`): initializes and optimizes
   $Q_{\\text{in}}, Q_{\\text{out}} \\in \\operatorname{O}(512)$ and $Q_{\\text{readout}} \\in \\operatorname{O}(512)$ as independent
   orthogonal frames, eliminating the Step-0 $x_{t+1} = x_t$ self-copy attractor.
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
import torch.nn.functional as F
import torch.utils.checkpoint as torch_ckpt

from qdlblocks.benchmarks.complementary_512d_lindblad_8l_lab import _collect_active_vocab_indices_w68
from qdlblocks.benchmarks.experiment_logger import (
    ExperimentIntentSpec,
    StandardizedExperimentLogger,
)
from qdlblocks.benchmarks.factored_kraus_density_gpt_lab import (
    DEFAULT_FACTORED_KRAUS_BRANCHES,
    FactoredKrausPureDensityBlock,
    FactoredKrausPureDensityGPT,
    Wave70ExperimentSpec,
    Wave70VariantMode,
    apply_commuted_cholesky_cross_head_kraus,
    apply_factored_associative_cross_head_kraus,
    apply_multibranch_stiefel_kraus_superoperator,
    compute_restored_riemannian_momentum_schedule,
)
from qdlblocks.benchmarks.explicit_density_tomographic_512d_lab import _complex_hermitian_mv
from qdlblocks.benchmarks.horizontal_grassmann_gauge_free_lab import compute_gauge_locked_mub_probe_expectations
from qdlblocks.benchmarks.pure_density_matrix_gpt_lab import (
    DEFAULT_PURE_CHOLESKY_RANK,
    DEFAULT_PURE_RHO_DIM,
    DEFAULT_PURE_RHO_HEADS,
    PURE_RHO_TRACE_EPS,
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
    DEFAULT_THERMOSTAT_EMA_DECAY,
    ReadOnlyQuenchRiemannianShadowEMA,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import LongSeqFineWebStream
from qdlblocks.layers.impure_density_mirror_block import (
    complex_l2_normalize,
    unpack_hermitian_isometric,
)

DEFAULT_ANCHOR_QKV_MIX_INIT: float = 0.30
DEFAULT_ANCHOR_RHO_V_MIX_INIT: float = 0.25
DEFAULT_BIPARTITE_SENSORY_TETHER: float = 0.60
DEFAULT_BIPARTITE_SENSORY_MODES: int = 2
DECOUPLED_Q_OUT_SEED_OFFSET: int = 909
DECOUPLED_READOUT_MIX_SEED_OFFSET: int = 1212


def _logit_from_prob(prob: float) -> float:
    """Convert probability `prob` in `(0, 1)` to unconstrained logit (`<= 8` args)."""
    p_c = max(1e-4, min(1.0 - 1e-4, float(prob)))
    return math.log(p_c / (1.0 - p_c))


def _build_deterministic_haar_unitary_stack(
    num_heads: int,
    d_h: int,
    seed: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct `num_heads` deterministic Haar-like unitary matrices in $SU(d_h)$ (`<= 8` args)."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    re_list, im_list = [], []
    for _ in range(num_heads):
        ar = torch.randn(d_h, d_h, generator=gen, dtype=torch.float32)
        ai = torch.randn(d_h, d_h, generator=gen, dtype=torch.float32)
        q_c, r_c = torch.linalg.qr(torch.complex(ar, ai))
        diag_r = torch.diagonal(r_c)
        ph = diag_r / diag_r.abs().clamp_min(1e-8)
        u_c = q_c * ph.unsqueeze(0)
        det_u = torch.linalg.det(u_c)
        det_phase = torch.angle(det_u) / float(d_h)
        u_su = u_c * torch.polar(torch.ones_like(det_phase), -det_phase).unsqueeze(-1).unsqueeze(-1)
        re_list.append(u_su.real)
        im_list.append(u_su.imag)
    u_re = torch.stack(re_list, dim=0)
    u_im = torch.stack(im_list, dim=0)
    if device is not None:
        return u_re.to(device), u_im.to(device)
    return u_re, u_im


def apply_dual_unitary_sensory_anchored_qkv(
    l_evol_pair: tuple[torch.Tensor, torch.Tensor],
    l_anc_pair: tuple[torch.Tensor, torch.Tensor],
    u_evol_pair: tuple[torch.Tensor, torch.Tensor],
    u_anc_pair: tuple[torch.Tensor, torch.Tensor] | None,
    anchor_gate_logit: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Dual-Unitary Sensory-Anchored Projection (`Lean 4 Theorem 12b`, `<= 8` args).

    Computes:
    $$\tilde{L}_{t,h}^{(\ell)} = \operatorname{Norm}_F\!\Big(
        \sqrt{1 - \gamma_h}\, \big(W_{\text{evol}}^{(\ell)} L_{t,h}^{\text{evol}}\big)
        + \sqrt{\gamma_h}\, \big(W_{\text{anc}}^{(\ell)} L_{t,h}^{(0)}\big)
    \Big)$$
    where $W_{\text{anc}}^{(\ell)} \in SU(16)$ is a dedicated sensory-anchor unitary distinct from $W_{\text{evol}}^{(\ell)}$.
    """
    evol_re, evol_im = apply_complex_unitary_to_cholesky(l_evol_pair[0], l_evol_pair[1], u_evol_pair[0], u_evol_pair[1])
    if u_anc_pair is None or anchor_gate_logit is None:
        return normalize_cholesky_factor_frobenius(evol_re, evol_im)

    work_dtype = evol_re.dtype
    num_h = evol_re.size(1)
    anc_re, anc_im = apply_complex_unitary_to_cholesky(l_anc_pair[0], l_anc_pair[1], u_anc_pair[0], u_anc_pair[1])
    gam = torch.sigmoid(anchor_gate_logit.float()).view(1, num_h, 1, 1, 1)
    w_evol = torch.sqrt((1.0 - gam).clamp_min(PURE_RHO_TRACE_EPS)).to(work_dtype)
    w_anc = torch.sqrt(gam.clamp_min(PURE_RHO_TRACE_EPS)).to(work_dtype)
    return normalize_cholesky_factor_frobenius(w_evol * evol_re + w_anc * anc_re, w_evol * evol_im + w_anc * anc_im)


def apply_bipartite_sensory_predictive_register_split(
    l_new_pair: tuple[torch.Tensor, torch.Tensor],
    l_anc_pair: tuple[torch.Tensor, torch.Tensor],
    sensory_tether_logit: torch.Tensor,
    sensory_modes: int = DEFAULT_BIPARTITE_SENSORY_MODES,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Bipartite Sensory/Predictive Register Split (`Lean 4 Theorem 12c`, `<= 8` args).

    Partitions $L_{t,h}^{(\ell)} \in \mathbb{C}^{16 \times 4}$ into:
    - Sensory-tethered sub-register `cols 0:sensory_modes` (tethered to $L_{t,h,:,0:2}^{(0)}$ with weight $\tau_h$), and
    - Predictive sub-register `cols sensory_modes:R` (evolving freely toward $x_{t+1}$),
    normalizing each half to Frobenius norm $1/\sqrt{2}$ so $\|L_{t,h}^{(\ell)}\|_F^2 \equiv 1$.
    """
    l_re, l_im = l_new_pair[0], l_new_pair[1]
    l0_re, l0_im = l_anc_pair[0], l_anc_pair[1]
    work_dtype = l_re.dtype
    num_h = l_re.size(1)
    tau = torch.sigmoid(sensory_tether_logit.float()).to(work_dtype).view(1, num_h, 1, 1, 1)

    sens_raw_re = (1.0 - tau) * l_re[..., :sensory_modes] + tau * l0_re[..., :sensory_modes]
    sens_raw_im = (1.0 - tau) * l_im[..., :sensory_modes] + tau * l0_im[..., :sensory_modes]
    sens_re, sens_im = normalize_cholesky_factor_frobenius(sens_raw_re, sens_raw_im)
    pred_re, pred_im = normalize_cholesky_factor_frobenius(l_re[..., sensory_modes:], l_im[..., sensory_modes:])

    half_scale = float(1.0 / math.sqrt(2.0))
    out_re = torch.cat([sens_re * half_scale, pred_re * half_scale], dim=-1)
    out_im = torch.cat([sens_im * half_scale, pred_im * half_scale], dim=-1)
    return out_re, out_im


class Wave71VariantMode(str, enum.Enum):
    """Ablation modes for Wave 71 (`EXPERIMENT_PLAN_091.md`)."""

    W71_A0_UNANCHORED_W70_REF_4L = "W71_A0_UNANCHORED_W70_REF_4L"
    W71_A1_SENSORY_ANCHORED_QKV_4L = "W71_A1_SENSORY_ANCHORED_QKV_4L"
    W71_A2_ANCHORED_BIPARTITE_DECOUPLED_4L = "W71_A2_ANCHORED_BIPARTITE_DECOUPLED_4L"
    W71_A3_DEEP_8L_ANCHORED_BIPARTITE = "W71_A3_DEEP_8L_ANCHORED_BIPARTITE"


@dataclass(frozen=True)
class Wave71ExperimentSpec:
    """Configuration for `AnchoredBipartitePureDensityGPT` (`Rule 2`, `<= 8` fields)."""

    variant_mode: Wave71VariantMode = Wave71VariantMode.W71_A2_ANCHORED_BIPARTITE_DECOUPLED_4L
    num_layers: int = 4
    kraus_branches: int = DEFAULT_FACTORED_KRAUS_BRANCHES
    enable_sensory_anchor: bool = True
    enable_bipartite_register: bool = True
    enable_decoupled_io_frames: bool = True
    seed: int = 1337


class SensoryAnchoredBipartiteBlock(FactoredKrausPureDensityBlock):
    r"""Pure Density-Matrix Block with Dual-Unitary Layer-0 Sensory Anchor & Bipartite Register Split (`Theorem 12`)."""

    def __init__(
        self,
        spec: Wave71ExperimentSpec | None = None,
        layer_idx: int = 0,
        depth_scale: float = 1.0,
    ) -> None:
        self.w71_spec = spec or Wave71ExperimentSpec()
        self.layer_idx = int(layer_idx)
        super().__init__(
            spec=Wave70ExperimentSpec(
                variant_mode=Wave70VariantMode.W70_K2_COMMUTED_AND_MULTIBRANCH_4L,
                num_layers=self.w71_spec.num_layers,
                kraus_branches=self.w71_spec.kraus_branches,
                enable_factored_kraus_attn=True,
                enable_commuted_trick_b=True,
                enable_multibranch_stiefel_kraus=True,
                seed=self.w71_spec.seed,
            ),
            depth_scale=depth_scale,
        )

        if self.w71_spec.enable_sensory_anchor:
            base_seed = int(self.w71_spec.seed) + 1000 + 97 * self.layer_idx
            wq_r, wq_i = _build_deterministic_haar_unitary_stack(self.num_heads, self.d_h, base_seed + 11)
            wk_r, wk_i = _build_deterministic_haar_unitary_stack(self.num_heads, self.d_h, base_seed + 23)
            wv_r, wv_i = _build_deterministic_haar_unitary_stack(self.num_heads, self.d_h, base_seed + 37)
            self.w_q_anc_re, self.w_q_anc_im = nn.Parameter(wq_r), nn.Parameter(wq_i)
            self.w_k_anc_re, self.w_k_anc_im = nn.Parameter(wk_r), nn.Parameter(wk_i)
            self.w_v_anc_re, self.w_v_anc_im = nn.Parameter(wv_r), nn.Parameter(wv_i)

            qkv_logit_init = _logit_from_prob(DEFAULT_ANCHOR_QKV_MIX_INIT)
            rhov_logit_init = _logit_from_prob(DEFAULT_ANCHOR_RHO_V_MIX_INIT)
            self.q_anc_gate_logit = nn.Parameter(torch.full((self.num_heads,), qkv_logit_init, dtype=torch.float32))
            self.k_anc_gate_logit = nn.Parameter(torch.full((self.num_heads,), qkv_logit_init, dtype=torch.float32))
            self.v_anc_gate_logit = nn.Parameter(torch.full((self.num_heads,), qkv_logit_init, dtype=torch.float32))
            self.rho_v_anc_gate_logit = nn.Parameter(torch.full((self.num_heads,), rhov_logit_init, dtype=torch.float32))
        else:
            for name in (
                "w_q_anc_re", "w_q_anc_im", "w_k_anc_re", "w_k_anc_im",
                "w_v_anc_re", "w_v_anc_im", "q_anc_gate_logit", "k_anc_gate_logit",
                "v_anc_gate_logit", "rho_v_anc_gate_logit",
            ):
                self.register_parameter(name, None)

        if self.w71_spec.enable_bipartite_register:
            tether_init = _logit_from_prob(DEFAULT_BIPARTITE_SENSORY_TETHER)
            self.bipartite_tether_logit = nn.Parameter(torch.full((self.num_heads,), tether_init, dtype=torch.float32))
        else:
            self.register_parameter("bipartite_tether_logit", None)

    def su16_unitary_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        r"""Return $\mathfrak{su}(16)$ unitary pairs including sensory-anchor unitaries (`<= 8` args)."""
        pairs = super().su16_unitary_pairs()
        if self.w_q_anc_re is not None and self.w_q_anc_im is not None:
            pairs.extend([
                (self.w_q_anc_re, self.w_q_anc_im),
                (self.w_k_anc_re, self.w_k_anc_im),
                (self.w_v_anc_re, self.w_v_anc_im),
            ])
        return pairs

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return bounded scalar/routing parameters including sensory-anchor gates (`<= 8` args)."""
        params = super().scalar_parameters()
        for p in (
            self.q_anc_gate_logit, self.k_anc_gate_logit,
            self.v_anc_gate_logit, self.rho_v_anc_gate_logit,
            self.bipartite_tether_logit,
        ):
            if p is not None:
                params.append(p)
        return params

    def restrict_scalars_(self) -> None:
        """Clamp bounded scalar logits in-place (`<= 8` args)."""
        super().restrict_scalars_()
        with torch.no_grad():
            for p in (
                self.q_anc_gate_logit, self.k_anc_gate_logit,
                self.v_anc_gate_logit, self.rho_v_anc_gate_logit,
                self.bipartite_tether_logit,
            ):
                if p is not None:
                    p.clamp_(-3.5, 3.5)

    def forward_anchored(
        self,
        l_in_re: torch.Tensor,
        l_in_im: torch.Tensor,
        rho_in_packed: torch.Tensor,
        l0_re: torch.Tensor,
        l0_im: torch.Tensor,
        rho0_packed: torch.Tensor,
        cos_t: torch.Tensor,
        sin_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Evolve density state with Layer-0 Dual-Unitary Sensory Anchor + Bipartite Register Split (`<= 8` args)."""
        bsz, num_h, seq_len, d_h, r_rank = l_in_re.shape
        work_dtype, chol_flat_dim = rho_in_packed.dtype, d_h * r_rank

        qj_re, qj_im = self._project_joint_stiefel_cholesky(l_in_re, l_in_im, self.w_q_stiefel)
        kj_re, kj_im = self._project_joint_stiefel_cholesky(l_in_re, l_in_im, self.w_k_stiefel)
        vj_re, vj_im = self._project_joint_stiefel_cholesky(l_in_re, l_in_im, self.w_v_stiefel)

        u_q_anc = (self.w_q_anc_re, self.w_q_anc_im) if self.w_q_anc_re is not None else None
        u_k_anc = (self.w_k_anc_re, self.w_k_anc_im) if self.w_k_anc_re is not None else None
        u_v_anc = (self.w_v_anc_re, self.w_v_anc_im) if self.w_v_anc_re is not None else None

        q_re, q_im = apply_dual_unitary_sensory_anchored_qkv(
            (qj_re, qj_im), (l0_re, l0_im), (self.w_q_re, self.w_q_im), u_q_anc, self.q_anc_gate_logit
        )
        k_re, k_im = apply_dual_unitary_sensory_anchored_qkv(
            (kj_re, kj_im), (l0_re, l0_im), (self.w_k_re, self.w_k_im), u_k_anc, self.k_anc_gate_logit
        )
        v_re, v_im = apply_dual_unitary_sensory_anchored_qkv(
            (vj_re, vj_im), (l0_re, l0_im), (self.w_v_re, self.w_v_im), u_v_anc, self.v_anc_gate_logit
        )

        if seq_len > 1:
            lag_w = self.k_lag_mix.to(work_dtype).view(1, num_h, 1, 1, 1)
            k_re, k_im = normalize_cholesky_factor_frobenius(
                (1.0 - lag_w) * k_re + lag_w * torch.cat([k_re[:, :, :1], k_re[:, :, :-1]], dim=2),
                (1.0 - lag_w) * k_im + lag_w * torch.cat([k_im[:, :, :1], k_im[:, :, :-1]], dim=2),
            )

        q_re, q_im = apply_complex_rope_to_cholesky(q_re, q_im, cos_t, sin_t)
        k_re, k_im = apply_complex_rope_to_cholesky(k_re, k_im, cos_t, sin_t)
        q_packed = pack_cholesky_factor_outer_isometric(q_re, q_im)
        k_packed = pack_cholesky_factor_outer_isometric(k_re, k_im)

        v_chol_packed = pack_cholesky_factor_outer_isometric(v_re, v_im)
        gv = self.v_full_rho_mix.to(work_dtype).view(1, num_h, 1, 1)
        rho_v_packed = (1.0 - gv) * v_chol_packed + gv * rho_in_packed
        if self.rho_v_anc_gate_logit is not None:
            g_anc_v = torch.sigmoid(self.rho_v_anc_gate_logit.float()).to(work_dtype).view(1, num_h, 1, 1)
            rho_v_packed = normalize_packed_density_trace((1.0 - g_anc_v) * rho_v_packed + g_anc_v * rho0_packed, d_h)

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
                    rho_fk = torch_ckpt.checkpoint(
                        apply_commuted_cholesky_cross_head_kraus,
                        (v_bar_re, v_bar_im), (ki_r, ki_i), (ko_r, ko_i), self.w_kraus_route,
                        use_reentrant=False,
                    )
                else:
                    rho_fk = apply_commuted_cholesky_cross_head_kraus(
                        (v_bar_re, v_bar_im), (ki_r, ki_i), (ko_r, ko_i), self.w_kraus_route
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
            l_log_re, l_log_im, rho_log_packed = torch_ckpt.checkpoint(
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
                l_mb_re, l_mb_im, rho_mb_packed = torch_ckpt.checkpoint(
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

        if self.bipartite_tether_logit is not None:
            l_out_re, l_out_im = apply_bipartite_sensory_predictive_register_split(
                (l_out_re, l_out_im), (l0_re, l0_im), self.bipartite_tether_logit
            )
        return l_out_re, l_out_im, rho_out_packed


class AnchoredBipartitePureDensityGPT(FactoredKrausPureDensityGPT):
    r"""Wave 71 (`EXPERIMENT_PLAN_091.md`): Sensory-Anchored Bipartite Pure Density-Matrix GPT."""

    def __init__(self, spec: Wave71ExperimentSpec | None = None) -> None:
        self.w71_spec = spec or Wave71ExperimentSpec()
        super().__init__(
            Wave70ExperimentSpec(
                variant_mode=Wave70VariantMode.W70_K2_COMMUTED_AND_MULTIBRANCH_4L,
                num_layers=self.w71_spec.num_layers,
                kraus_branches=self.w71_spec.kraus_branches,
                seed=self.w71_spec.seed,
            )
        )
        self.depth_scale = 1.0 / math.sqrt(float(max(1, self.w71_spec.num_layers)))
        self.density_blocks = nn.ModuleList([
            SensoryAnchoredBipartiteBlock(spec=self.w71_spec, layer_idx=i, depth_scale=self.depth_scale)
            for i in range(self.w71_spec.num_layers)
        ])
        self.initialize_quantum_state(seed=self.w71_spec.seed)

    @torch.no_grad()
    def initialize_quantum_state(self, seed: int = 1337) -> float:
        """Initialize boundary frames and decoupled readout basis (`<= 8` args)."""
        elapsed_ms = super().initialize_quantum_state(seed=seed)
        if getattr(self, "w71_spec", None) is not None and self.w71_spec.enable_decoupled_io_frames:
            device = self.boundary.q_out_ortho.device
            gen_out = torch.Generator(device=device).manual_seed(int(seed) + DECOUPLED_Q_OUT_SEED_OFFSET)
            q_out, _ = torch.linalg.qr(torch.randn(self.embed_dim, self.embed_dim, generator=gen_out, device=device))
            self.boundary.q_out_ortho.copy_(q_out)

            gen_mix = torch.Generator(device=device).manual_seed(int(seed) + DECOUPLED_READOUT_MIX_SEED_OFFSET)
            q_mix, _ = torch.linalg.qr(torch.randn(self.embed_dim, self.embed_dim, generator=gen_mix, device=device))
            self.q_readout_mix.copy_(q_mix)
        return elapsed_ms

    def bridge_parameter_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        """Decouple `(q_in_ortho, q_out_ortho)` when `enable_decoupled_io_frames` is active (`<= 8` args)."""
        if self.w71_spec.enable_decoupled_io_frames:
            return []
        return super().bridge_parameter_pairs()

    def extra_real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Include decoupled `q_in_ortho` and `q_out_ortho` as independent real Stiefel manifolds (`<= 8` args)."""
        params = super().extra_real_stiefel_parameters()
        if self.w71_spec.enable_decoupled_io_frames:
            params.extend([self.boundary.q_in_ortho, self.boundary.q_out_ortho])
        return params

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Return deduplicated real Stiefel parameters for Shadow EMA retraction (`<= 8` args)."""
        if self.w71_spec.enable_decoupled_io_frames:
            return self.extra_real_stiefel_parameters()
        return super().real_stiefel_parameters()

    def encode_pure_density_trunk(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Evolve $(L_{t,h}^{(\ell)}, \tilde{\rho}_{t,h}^{(\ell)})$ with persistent Layer-0 sensory anchor $(L_{t,h}^{(0)}, \tilde{\rho}_{t,h}^{(0)})$ (`<= 8` args)."""
        bsz, seq_len = idx.shape
        l0_re, l0_im, rho0_packed = self.prepare_layer0_density_state(idx)
        l_re, l_im, rho_packed = l0_re, l0_im, rho0_packed
        work_dtype = rho_packed.dtype
        cos_t, sin_t = self._get_rope_cos_sin(seq_len, work_dtype, idx.device)

        is_train = self.training and torch.is_grad_enabled()
        for blk in self.density_blocks:
            if is_train:
                l_re, l_im, rho_packed = torch_ckpt.checkpoint(
                    blk.forward_anchored,
                    l_re, l_im, rho_packed, l0_re, l0_im, rho0_packed, cos_t, sin_t,
                    use_reentrant=False,
                )
                blk.history_buffer.update_from_window(l_re, l_im, rho_packed)
            else:
                l_re, l_im, rho_packed = blk.forward_anchored(
                    l_re, l_im, rho_packed, l0_re, l0_im, rho0_packed, cos_t, sin_t
                )

        q_mub_512 = F.rms_norm(
            compute_gauge_locked_mub_probe_expectations(rho_packed, self.probe_u1_re, self.probe_u1_im, self.dft_f_re, self.dft_f_im),
            (self.embed_dim,), eps=1e-6,
        )
        q_herm_bhtk = torch.matmul(rho_packed, self.w_herm_obs.to(work_dtype).unsqueeze(0))
        q_herm_512 = F.rms_norm(q_herm_bhtk.permute(0, 2, 1, 3).reshape(bsz, seq_len, self.embed_dim), (self.embed_dim,), eps=1e-6)

        rho_re_f, rho_im_f = unpack_hermitian_isometric(rho_packed, self.d_h)
        if self.w71_spec.enable_bipartite_register:
            dom_re, dom_im = complex_l2_normalize(l_re[..., 2], l_im[..., 2])
            aux_re, aux_im = complex_l2_normalize(l_re[..., 3] + 0.5 * l_re[..., 1], l_im[..., 3] + 0.5 * l_im[..., 1])
        else:
            dom_re, dom_im = complex_l2_normalize(l_re[..., 0], l_im[..., 0])
            aux_re, aux_im = complex_l2_normalize(l_re[..., 1] + l_re[..., 3], l_im[..., 1] + l_im[..., 3])

        mv_re, mv_im = _complex_hermitian_mv(rho_re_f, rho_im_f, dom_re, dom_im)
        ket_re, ket_im = dom_re + 0.35 * mv_re + 0.25 * aux_re, dom_im + 0.35 * mv_im + 0.25 * aux_im
        purif_flat = torch.stack([ket_re, ket_im], dim=-2).permute(0, 2, 1, 3, 4).reshape(bsz, seq_len, self.embed_dim)
        q_purif_512 = F.rms_norm(torch.matmul(purif_flat, self.q_readout_mix.to(work_dtype)), (self.embed_dim,), eps=1e-6)

        u_vocab_512 = (
            self.herm_obs_scale.to(work_dtype) * q_herm_512
            + self.mub_scale.to(work_dtype) * q_mub_512
            + self.purif_ket_scale.to(work_dtype) * q_purif_512
        )
        return u_vocab_512, l_re, rho_packed


def build_wave71_variant_spec(mode: Wave71VariantMode, seed: int = 1337) -> Wave71ExperimentSpec:
    """Resolve `Wave71ExperimentSpec` from `Wave71VariantMode` (`<= 8` args)."""
    if mode == Wave71VariantMode.W71_A0_UNANCHORED_W70_REF_4L:
        return Wave71ExperimentSpec(
            variant_mode=mode, num_layers=4,
            enable_sensory_anchor=False, enable_bipartite_register=False, enable_decoupled_io_frames=False,
            seed=seed,
        )
    if mode == Wave71VariantMode.W71_A1_SENSORY_ANCHORED_QKV_4L:
        return Wave71ExperimentSpec(
            variant_mode=mode, num_layers=4,
            enable_sensory_anchor=True, enable_bipartite_register=False, enable_decoupled_io_frames=False,
            seed=seed,
        )
    if mode == Wave71VariantMode.W71_A2_ANCHORED_BIPARTITE_DECOUPLED_4L:
        return Wave71ExperimentSpec(
            variant_mode=mode, num_layers=4,
            enable_sensory_anchor=True, enable_bipartite_register=True, enable_decoupled_io_frames=True,
            seed=seed,
        )
    return Wave71ExperimentSpec(
        variant_mode=mode, num_layers=8,
        enable_sensory_anchor=True, enable_bipartite_register=True, enable_decoupled_io_frames=True,
        seed=seed,
    )


def run_wave71_variant(
    spec: Wave71ExperimentSpec,
    stream: LongSeqFineWebStream,
    steps: int = 200,
    batch_size: int = 8,
    seq_len: int = 512,
    eval_interval: int = 25,
) -> dict[str, Any]:
    """Train and evaluate one `Wave71VariantMode` run with `StandardizedExperimentLogger` (`<= 8` args)."""
    device = stream.device
    torch.manual_seed(spec.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(spec.seed)
        gc.collect()
        torch.cuda.empty_cache()

    model = AnchoredBipartitePureDensityGPT(spec).to(device)
    init_ms = model.initialize_quantum_state(seed=spec.seed)
    total_params = sum(p.numel() for p in model.parameters())
    trunk_opt, vocab_opt, scalar_opt = build_wave69_optimizers(model)
    retr_cad = 8 if steps >= 1000 else 5
    thermostat = ReadOnlyQuenchRiemannianShadowEMA(
        model, ema_decay=DEFAULT_THERMOSTAT_EMA_DECAY, eval_fn=evaluate_wave69_model, retract_cadence=retr_cad
    )

    tokens_per_step = batch_size * seq_len
    run_key = f"plan091_{spec.variant_mode.value.lower()}_{steps}steps"
    exp_logger = StandardizedExperimentLogger(
        ExperimentIntentSpec(
            run_key=run_key,
            wave_id="W71",
            plan_id="EXPERIMENT_PLAN_091",
            task_domain="fineweb_speedrun",
            hypothesis_intent=(
                f"Wave 71 {spec.variant_mode.value} (L={spec.num_layers}, anchor={spec.enable_sensory_anchor}, "
                f"bipartite={spec.enable_bipartite_register}, decoupled_io={spec.enable_decoupled_io_frames}, mom_beta=0.85->0.95)."
            ),
            architecture_summary=f"AnchoredBipartitePureDensityGPT(L={spec.num_layers}, params={total_params:,})",
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
