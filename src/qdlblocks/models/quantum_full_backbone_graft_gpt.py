r"""Full-Backbone (`Embeddings through Layer 11`) Quantum Wavepacket Replacement (`EXPERIMENT_PLAN_073.md`).

Implements **Wave 53A** (`EXPERIMENT_PLAN_073.md`):
1. **Progressive & Full-Backbone Quantum Wavepacket Replacement (`QuantumBackboneDepthScope`)**:
   - `EMB_PLUS_L0_TO_L2_ATTN`: Replaces `wte` + Layers `0..2` (`25%` of backbone) with Quantum Wavepacket Outer-Product Attention.
   - `EMB_PLUS_L0_TO_L5_ATTN`: Replaces `wte` + Layers `0..5` (`50%` U-Net encoder half) with Quantum Wavepacket Attention.
   - `EMB_PLUS_ALL_12_ATTN`: Replaces `wte` + **all attention layers across Layers `0..11`** (`100%` of attention through the last layer).
   - `EMB_PLUS_ALL_12_FULL_BLOCKS_MZ`: Replaces **100% of the network from embeddings through Layer 11** (`q_wte`, polar-factorized
     sphere `q_value_embeds[0..2]`, `q_smear_gate`, `q_skip_weights`, all 12 layers' quantum wavepacket attention, and all 12 layers'
     `SVDStiefelMachZehnderInterferometer` MLPs), leaving only `lm_head` at the vocabulary readout boundary.
2. **Entropy-Preserving Coherent Ket + Traceless Outer-Product Density Operator (`QuantumKetAmplitudeMode`)**:
   - Preserves attention-entropy amplitude modulation in the $V$-ket branch while evolving a trace-1 purified outer-product density
     matrix $\rho_{\text{evol}, t, h}$ and applying its traceless Hermitian component $\mathring{\rho}_{\text{evol}, t, h} = \rho_{\text{evol}, t, h} - \frac{1}{d_h}I_{d_h}$
     directly onto the rotated $V$-space wavefunctions.
3. **Biorthogonal SVD-Stiefel Two-Beam Mach-Zehnder Interferometer (`SVDStiefelMachZehnderInterferometer`)**:
   - Parameterizes each layer's feed-forward channel on $\mathrm{St}(3072, r_{\text{mlp}}) \times \mathbb{R}_+^{r_{\text{mlp}}} \times \mathrm{St}(768, r_{\text{mlp}})$
     with coherent beam-splitter quadrature interference $A = S + D(\theta_{\text{mz}}), B = S - D(\theta_{\text{mz}})$.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..kernels.fused_quantum_wavepacket_attn import (
    QuantumAttnKernelMode,
    run_coherent_ket_wavepacket_attention,
)
from ..layers.impure_density_mirror_block import (
    DEFAULT_TIKHONOV_GRAM_LAMBDA,
    TRACE_NORM_EPS,
    initialize_stiefel_parameter,
)
from ..layers.quantum_frontend_graft import (
    DEFAULT_FINAL_FEAT_ALIGN_SCALE,
    DEFAULT_TEACHER_300M_CKPT_PATH,
    ConfigurableWavepacketStatMechBlock,
    QuantumAttnInteriorMode,
    QuantumQKPhaseMode,
    QuantumRecoveryScheduleMode,
    QuantumSelfOrthoMode,
    QuantumStateIOMode,
    QuantumStiefelProjMode,
    QuantumValueEmbedCouplingMode,
    QuantumWavefunctionShape,
    QuantumWteInitMode,
    _normalized_cosine_mse,
    resolve_wavefunction_shape_spec,
)
from ..layers.quantum_statmech_channels import (
    PurificationMode,
    QuantumChannelMode,
    QuantumMomentumScheduleMode,
    StiefelMomentumMode,
    compute_riemannian_momentum_beta,
)
from ..layers.quantum_svd_stiefel_graft import (
    DEFAULT_CALIBRATED_LAMBDA_BOUND,
    DEFAULT_PARTIAL_KEY_SHIFT_HALF,
    BiorthogonalSVDStiefelProjection,
    QuantumDepthLRMode,
    QuantumDocSWCMode,
    SVDStiefelWavepacketStatMechBlock,
    apply_berry_qk_phase,
    apply_boundary_stiefel_bridge,
    apply_fubini_study_self_ortho,
    build_doc_swc_sdpa_masks,
    compute_depth_lr_scale,
    compute_liouville_commutator,
    compute_quantum_backbone_lr_multiplier,
    initialize_arbitrary_shape_ppmi_sphere_embeddings_,
    initialize_teacher_procrustes_sphere_embeddings_,
    map_teacher_768_to_quantum_packets,
    resolve_svd_stiefel_rank,
)
from ..layers.wavepacket_statmech_block import (
    DEFAULT_PERSISTENT_ALIGN_FLOOR_RATIO,
    apply_causal_wavepacket_dispersion,
)
from .compiled_leaderboard_classical_gpt import (
    CompiledLeaderboardClassicalConfig,
    CompiledLeaderboardClassicalGPT,
    _CompiledChunkedSoftcapLinearCEFn,
    apply_native_leaderboard_rotary,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
DEFAULT_COHERENT_OBS_SCALE_INIT: float = 1.0
DEFAULT_BASE_RHO_COUPLING_INIT: float = 0.25
DEFAULT_DOC_SWC_RHO_COUPLING_INIT: float = 0.05
DEFAULT_MZ_SCALE_INIT: float = 1.0
DEFAULT_MZ_SCALE_MIN: float = 0.05
DEFAULT_MZ_SCALE_MAX: float = 5.00
DEFAULT_MZ_THETA_INIT: float = 0.04
DEFAULT_DOC_SWC_MZ_THETA_INIT: float = 0.015
DEFAULT_MZ_THETA_BOUND: float = 1.50
DEFAULT_MZ_GAMMA_INIT: float = 0.25
DEFAULT_DOC_SWC_MZ_GAMMA_INIT: float = 0.20
DEFAULT_MZ_GAMMA_BOUND: float = 2.00
DEFAULT_VE_BANK_SCALE_MIN: float = 0.20
DEFAULT_VE_BANK_SCALE_MAX: float = 5.00
DEFAULT_FULL_BACKBONE_WARMUP_STEPS: int = 250
DEFAULT_FULL_BACKBONE_ALIGN_WEIGHT: float = 0.50
DEFAULT_FULL_BACKBONE_WTE_LR: float = 0.004
DEFAULT_FULL_BACKBONE_STIEFEL_LR: float = 0.008
DEFAULT_FULL_BACKBONE_UNITARY_LR: float = 0.012
DEFAULT_FULL_BACKBONE_SCALAR_LR: float = 0.006
DEFAULT_TEACHER_0M_CKPT_PATH: Path = DEFAULT_TEACHER_300M_CKPT_PATH.parent / "ckpt_step00000_0000M_m0000M.pt"
DEFAULT_TEACHER_100M_CKPT_PATH: Path = DEFAULT_TEACHER_300M_CKPT_PATH.parent / "ckpt_step00382_0100M_m0100M.pt"
DEFAULT_QUANTUM_TAIL_EMA_START_FRAC: float = 0.70
DEFAULT_QUANTUM_TAIL_EMA_RATE: float = 0.05
DEFAULT_QUANTUM_TAIL_EMA_BLEND: float = 0.70
DEFAULT_FROM_SCRATCH_HEAD_LR: float = 0.008
DEFAULT_FROM_SCRATCH_MLP_MUON_LR: float = 0.04
DEFAULT_FROM_SCRATCH_STIEFEL_LR: float = 0.025
DEFAULT_FROM_SCRATCH_UNITARY_LR: float = 0.025
DEFAULT_FROM_SCRATCH_WTE_LR: float = 0.060
DEFAULT_FROM_SCRATCH_SCALAR_LR: float = 0.020


class QuantumBackboneDepthScope(str, Enum):
    """Strategy enum selecting how much of the 12-layer backbone is replaced by quantum blocks (Rule 3)."""

    EMB_PLUS_L0_TO_L2_ATTN = "emb_plus_l0_to_l2_attn"
    EMB_PLUS_L0_TO_L5_ATTN = "emb_plus_l0_to_l5_attn"
    EMB_PLUS_ALL_12_ATTN = "emb_plus_all_12_attn"
    EMB_PLUS_ALL_12_FULL_BLOCKS_MZ = "emb_plus_all_12_full_blocks_mz"


class QuantumKetAmplitudeMode(str, Enum):
    """Strategy enum selecting post-RMSNorm ket vs. entropy-preserving coherent ket outer-product coupling (Rule 3)."""

    POST_RMSNORM_KET = "post_rmsnorm_ket"
    ENTROPY_PRESERVING_COHERENT_KET = "entropy_preserving_coherent_ket"


class QuantumMLPChannelMode(str, Enum):
    """Strategy enum selecting frozen classical MLP vs. SVD-Stiefel two-beam Mach-Zehnder interferometer (Rule 3)."""

    FROZEN_CLASSICAL_MLP = "frozen_classical_mlp"
    SVD_STIEFEL_MACH_ZEHNDER = "svd_stiefel_mach_zehnder"


@dataclass
class QuantumFullBackboneGraftConfig:
    """Configuration for progressive and full-backbone (`Embeddings through Layer 11`) quantum wavepacket replacement."""

    wf_shape: QuantumWavefunctionShape = QuantumWavefunctionShape.SHAPE_12x2x32
    depth_scope: QuantumBackboneDepthScope = QuantumBackboneDepthScope.EMB_PLUS_ALL_12_ATTN
    ket_amplitude_mode: QuantumKetAmplitudeMode = QuantumKetAmplitudeMode.ENTROPY_PRESERVING_COHERENT_KET
    mlp_channel_mode: QuantumMLPChannelMode = QuantumMLPChannelMode.FROZEN_CLASSICAL_MLP
    doc_swc_mode: QuantumDocSWCMode = QuantumDocSWCMode.FULL_CAUSAL_1024
    depth_lr_mode: QuantumDepthLRMode = QuantumDepthLRMode.UNIFORM_FLAT_THEN_COSINE
    state_io_mode: QuantumStateIOMode = QuantumStateIOMode.WF_IN_WF_OUT
    attn_interior_mode: QuantumAttnInteriorMode = QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V
    purification_mode: PurificationMode = PurificationMode.QUADRATIC_MCWEENY
    channel_mode: QuantumChannelMode = QuantumChannelMode.UNITARY_U16
    wte_init_mode: QuantumWteInitMode = QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE
    ve_coupling_mode: QuantumValueEmbedCouplingMode = QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED
    recovery_schedule_mode: QuantumRecoveryScheduleMode = QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE
    stiefel_proj_mode: QuantumStiefelProjMode = QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384
    mlp_stiefel_proj_mode: QuantumStiefelProjMode = QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R384
    self_ortho_mode: QuantumSelfOrthoMode = QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO
    qk_phase_mode: QuantumQKPhaseMode = QuantumQKPhaseMode.BERRY_PHASE_COMPLEX_QK
    attn_kernel_mode: QuantumAttnKernelMode = QuantumAttnKernelMode.EXPANDED_SDPA_CHUNKS
    stiefel_momentum_mode: StiefelMomentumMode = StiefelMomentumMode.RIEMANNIAN_TANGENT_TRANSPORTED
    momentum_schedule_mode: QuantumMomentumScheduleMode = QuantumMomentumScheduleMode.FIXED_HIGH_MOMENTUM_090
    interface_warmup_steps: int = DEFAULT_FULL_BACKBONE_WARMUP_STEPS
    interface_align_weight: float = DEFAULT_FULL_BACKBONE_ALIGN_WEIGHT
    wte_lr: float = DEFAULT_FULL_BACKBONE_WTE_LR
    stiefel_lr: float = DEFAULT_FULL_BACKBONE_STIEFEL_LR
    unitary_lr: float = DEFAULT_FULL_BACKBONE_UNITARY_LR
    scalar_lr: float = DEFAULT_FULL_BACKBONE_SCALAR_LR


class SVDStiefelMachZehnderInterferometer(nn.Module):
    r"""Biorthogonal SVD-Stiefel Two-Beam Mach-Zehnder Quantum Interferometer replacing `blocks[l].mlp`."""

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        proj_mode: QuantumStiefelProjMode,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_hidden = int(d_hidden)
        self.proj_mode = QuantumStiefelProjMode(proj_mode)
        self.rank = resolve_svd_stiefel_rank(self.proj_mode, self.d_model, self.d_hidden)
        self.eps = float(eps)

        self.up_proj = BiorthogonalSVDStiefelProjection(self.d_model, self.d_hidden, self.rank)
        self.down_proj = BiorthogonalSVDStiefelProjection(self.d_hidden, self.d_model, self.rank)
        self.mz_scale = nn.Parameter(torch.tensor(DEFAULT_MZ_SCALE_INIT, dtype=torch.float32))
        self.mz_theta = nn.Parameter(torch.tensor(DEFAULT_MZ_THETA_INIT, dtype=torch.float32))
        self.mz_gamma = nn.Parameter(torch.tensor(DEFAULT_MZ_GAMMA_INIT, dtype=torch.float32))

    @torch.no_grad()
    def calibrate_from_teacher_block_(self, teacher_blk: Any) -> None:
        """Calibrate `up_proj` and `down_proj` on Stiefel manifolds from `teacher_blk.fc1` and `teacher_blk.fc2`."""
        self.up_proj.initialize_from_teacher_matrix_(teacher_blk.fc1.weight.detach().float())
        self.down_proj.initialize_from_teacher_matrix_(teacher_blk.fc2.weight.detach().float())

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Return the 4 tall Stiefel manifold matrices `[up.u, up.v, down.u, down.v]`."""
        return self.up_proj.real_stiefel_parameters() + self.down_proj.real_stiefel_parameters()

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return singular value spectra and interferometer scalar angles `[up.sigma, down.sigma, scale, theta, gamma]`."""
        params: list[nn.Parameter] = []
        params.extend(self.up_proj.scalar_parameters())
        params.extend(self.down_proj.scalar_parameters())
        params.extend([self.mz_scale, self.mz_theta, self.mz_gamma])
        return params

    @torch.no_grad()
    def restrict_scalars_(self) -> None:
        """Clamp singular values and Mach-Zehnder interferometer scalars to bounded physical intervals."""
        self.up_proj.clamp_sigma_()
        self.down_proj.clamp_sigma_()
        self.mz_scale.clamp_(DEFAULT_MZ_SCALE_MIN, DEFAULT_MZ_SCALE_MAX)
        self.mz_theta.clamp_(-DEFAULT_MZ_THETA_BOUND, DEFAULT_MZ_THETA_BOUND)
        self.mz_gamma.clamp_(-DEFAULT_MZ_GAMMA_BOUND, DEFAULT_MZ_GAMMA_BOUND)

    def forward(self, x_mid: torch.Tensor) -> torch.Tensor:
        """Execute two-beam Mach-Zehnder quadrature interference and return `x_out` (`(B, T, d_model)`)."""
        work_dtype = x_mid.dtype
        h_mlp = F.rms_norm(x_mid, (self.d_model,), eps=self.eps)
        s_exc = self.up_proj(h_mlp)
        s_pairs = s_exc.view(*s_exc.shape[:-1], self.d_hidden // 2, 2)
        s_re, s_im = s_pairs[..., 0], s_pairs[..., 1]
        th = torch.tanh(self.mz_theta).to(work_dtype)
        d_quad = torch.stack([-th * s_im, th * s_re], dim=-1).reshape_as(s_exc)
        beam_a = s_exc + d_quad
        beam_b = s_exc - d_quad
        ia = F.relu(beam_a).square()
        ib = F.relu(beam_b).square()
        gam = torch.tanh(self.mz_gamma).to(work_dtype)
        z_mz = 0.5 * (ia + ib) + 0.25 * gam * (ia - ib)
        return x_mid + self.mz_scale.to(work_dtype) * self.down_proj(z_mz)


def _run_entropy_preserving_coherent_ket_attention(
    svd_blk: SVDStiefelWavepacketStatMechBlock,
    v_wp: torch.Tensor,
    v_mixed: torch.Tensor,
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Execute trace-1 outer-product density evolution + coherent $V$-ket attention via `run_coherent_ket_wavepacket_attention`."""
    return run_coherent_ket_wavepacket_attention(svd_blk, v_wp, v_mixed, q_rot, k_rot, attn_mask=attn_mask)


def execute_quantum_wavepacket_attn_layer(
    svd_blk: SVDStiefelWavepacketStatMechBlock,
    x_768: torch.Tensor,
    x_0_768: torch.Tensor,
    cos_4d: torch.Tensor,
    sin_4d: torch.Tensor,
    q_scale: torch.Tensor,
    ve_or_v0_wp: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute 1 calibrated quantum wavepacket attention sublayer returning `(x_mid_768, v_wp)` (`<= 8` args)."""
    bsz, seq_len, _ = x_768.shape
    work_dtype = x_768.dtype
    l0, l1 = svd_blk.block_lambdas[0].to(work_dtype), svd_blk.block_lambdas[1].to(work_dtype)
    x_in_768 = l0 * x_768 + l1 * x_0_768.to(work_dtype)
    h_768 = F.rms_norm(x_in_768, (svd_blk.d_model,), eps=TRACE_NORM_EPS)

    q_wp = svd_blk.q_proj(h_768).view(bsz, seq_len, svd_blk.num_pkts, svd_blk.d_h)
    k_wp = svd_blk.k_proj(h_768).view(bsz, seq_len, svd_blk.num_pkts, svd_blk.d_h)
    v_wp = svd_blk.v_proj(h_768).view(bsz, seq_len, svd_blk.num_pkts, svd_blk.d_h)

    s0, s1 = svd_blk.sa_lambdas[0].to(work_dtype), svd_blk.sa_lambdas[1].to(work_dtype)
    if ve_or_v0_wp is not None:
        v_mixed = s0 * v_wp + s1 * ve_or_v0_wp.to(work_dtype)
    else:
        v_mixed = s0 * v_wp

    q_heads = F.rms_norm(q_wp.view(bsz, seq_len, svd_blk.num_heads, svd_blk.head_dim), (svd_blk.head_dim,), eps=TRACE_NORM_EPS)
    k_heads = F.rms_norm(k_wp.view(bsz, seq_len, svd_blk.num_heads, svd_blk.head_dim), (svd_blk.head_dim,), eps=TRACE_NORM_EPS)
    q_rot = apply_native_leaderboard_rotary(q_heads, cos_4d, sin_4d) * q_scale.to(work_dtype)
    if (
        svd_blk.qk_phase_mode == QuantumQKPhaseMode.BERRY_PHASE_COMPLEX_QK
        and svd_blk.attn_kernel_mode != QuantumAttnKernelMode.FUSED_HERMITIAN_BORN_RHO_KET.value
    ):
        q_rot = apply_berry_qk_phase(q_rot, svd_blk.berry_qk_phase)
    k_rot = apply_native_leaderboard_rotary(k_heads, cos_4d, sin_4d)
    if svd_blk.use_partial_key_offset and seq_len > 1 and svd_blk.head_dim == 2 * DEFAULT_PARTIAL_KEY_SHIFT_HALF:
        d_half = DEFAULT_PARTIAL_KEY_SHIFT_HALF
        k_shifted = torch.cat([k_rot[:, :1, :, d_half:], k_rot[:, :-1, :, d_half:]], dim=1)
        k_rot = torch.cat([k_rot[..., :d_half], k_shifted], dim=-1)

    v_mixed_h = v_mixed.view(bsz, seq_len, svd_blk.num_heads, svd_blk.head_dim)
    if svd_blk.ket_amplitude_mode == QuantumKetAmplitudeMode.ENTROPY_PRESERVING_COHERENT_KET.value:
        y_scaled = _run_entropy_preserving_coherent_ket_attention(svd_blk, v_wp, v_mixed, q_rot, k_rot, attn_mask=attn_mask)
    else:
        h_pkt = F.rms_norm(v_wp, (svd_blk.d_h,), eps=TRACE_NORM_EPS)
        y_wp = svd_blk.core_block._run_attention_and_evolution(
            h_pkt, q_rot.view(bsz, seq_len, svd_blk.num_pkts, svd_blk.d_h), q_rot, k_rot, v_mixed
        )
        v_head_rms = v_mixed_h.float().pow(2).mean(dim=-1, keepdim=True).clamp_min(TRACE_NORM_EPS).sqrt().to(work_dtype)
        y_scaled = y_wp.view(bsz, seq_len, svd_blk.num_heads, svd_blk.head_dim) * v_head_rms * svd_blk.obs_scale.to(work_dtype)

    if svd_blk.self_ortho_mode == QuantumSelfOrthoMode.FUBINI_STUDY_SELF_ORTHO:
        y_ortho = apply_fubini_study_self_ortho(y_scaled, v_mixed_h, svd_blk.q_xsa_alpha)
    else:
        y_ortho = y_scaled

    g_attn = (
        2.0 * torch.sigmoid(F.linear(h_768[..., : svd_blk.gate_in_ch], svd_blk.q_attn_gate.weight.to(work_dtype)))
    ).unsqueeze(-1)
    y_gated = (y_ortho * g_attn).reshape(bsz, seq_len, svd_blk.d_emb)
    d_attn_768 = svd_blk.obs_proj(y_gated)
    return x_in_768 + d_attn_768, v_wp


class QuantumFullBackboneGraftGPT(nn.Module):
    r"""Progressive & Full-Backbone (`Embeddings through Layer 11`) Quantum Wavepacket GPT (`EXPERIMENT_PLAN_073/074.md`)."""

    def __init__(
        self,
        classical_model: CompiledLeaderboardClassicalGPT,
        graft_cfg: QuantumFullBackboneGraftConfig | None = None,
        ppmi_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.classical = classical_model
        self.cfg = classical_model.cfg
        self.graft_cfg = graft_cfg if graft_cfg is not None else QuantumFullBackboneGraftConfig()
        self.vocab_size = int(self.classical.vocab_size)
        self.embed_dim = int(self.classical.embed_dim)
        self.num_heads = int(self.classical.num_heads)
        self.num_layers = int(self.classical.num_layers)
        self.hidden_dim = int(self.classical.hidden_dim)
        self.lm_head = self.classical.lm_head
        self.training_step: int = 0
        self.from_scratch_mode: bool = False

        for p in self.classical.parameters():
            p.requires_grad_(False)
        object.__setattr__(self.classical.cfg, "use_flex_doc_swc", False)

        spec = resolve_wavefunction_shape_spec(self.graft_cfg.wf_shape)
        self.wte_slices, self.wte_sphere_dim, self.d_emb, self.pkt_count, self.pkt_dim, self.q_heads = spec

        self.w_in_bridge: nn.Parameter | None = None
        if self.d_emb != self.embed_dim:
            rows, cols = max(self.d_emb, self.embed_dim), min(self.d_emb, self.embed_dim)
            w_in_b = torch.empty(rows, cols, dtype=torch.float32)
            initialize_stiefel_parameter(w_in_b)
            self.w_in_bridge = nn.Parameter(w_in_b)

        q_wte = torch.randn(self.vocab_size, self.wte_slices, self.wte_sphere_dim, dtype=torch.float32)
        if self.graft_cfg.wte_init_mode == QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE:
            initialize_teacher_procrustes_sphere_embeddings_(q_wte, self.classical.wte.weight, self.w_in_bridge, None)
        else:
            initialize_arbitrary_shape_ppmi_sphere_embeddings_(q_wte, ppmi_embeddings=ppmi_embeddings)
        self.q_wte = nn.Parameter(q_wte)
        self.in_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        wp_complex_modes = max(1, self.pkt_count // 2)
        self.wp_causal_re = nn.Parameter(torch.zeros(wp_complex_modes, dtype=torch.float32))
        self.wp_causal_im = nn.Parameter(torch.zeros(wp_complex_modes, dtype=torch.float32))

        self.full_blocks_replaced: bool = (
            self.graft_cfg.depth_scope == QuantumBackboneDepthScope.EMB_PLUS_ALL_12_FULL_BLOCKS_MZ
            or self.graft_cfg.mlp_channel_mode == QuantumMLPChannelMode.SVD_STIEFEL_MACH_ZEHNDER
        )
        self.replaced_attn_layers: tuple[int, ...] = self._resolve_replaced_attn_layers()
        self.replaced_mlp_layers: tuple[int, ...] = tuple(range(self.num_layers)) if self.full_blocks_replaced else ()

        self.quantum_attn_blocks = nn.ModuleDict()
        n_q_attn = max(1, len(self.replaced_attn_layers))
        use_doc_swc = self.graft_cfg.doc_swc_mode == QuantumDocSWCMode.TEACHER_DOC_SWC_MASK
        base_rho = DEFAULT_DOC_SWC_RHO_COUPLING_INIT if use_doc_swc else DEFAULT_BASE_RHO_COUPLING_INIT
        rho_init = base_rho / math.sqrt(float(n_q_attn))
        for l_idx in self.replaced_attn_layers:
            t_blk = self.classical.blocks[l_idx]
            core = ConfigurableWavepacketStatMechBlock(
                self.pkt_count,
                self.q_heads,
                self.pkt_dim,
                self.num_layers,
                self.graft_cfg.state_io_mode,
                self.graft_cfg.attn_interior_mode,
                self.graft_cfg.purification_mode,
                self.graft_cfg.channel_mode,
            )
            core.disable_mach_zehnder_()
            svd_blk = SVDStiefelWavepacketStatMechBlock(
                core,
                self.embed_dim,
                self.d_emb,
                self.graft_cfg.stiefel_proj_mode,
                self.graft_cfg.self_ortho_mode,
                gate_in_ch=int(t_blk.gate_in_ch),
                qk_phase_mode=self.graft_cfg.qk_phase_mode,
            )
            svd_blk.ket_amplitude_mode = self.graft_cfg.ket_amplitude_mode.value
            svd_blk.attn_kernel_mode = QuantumAttnKernelMode(self.graft_cfg.attn_kernel_mode).value
            if svd_blk.attn_kernel_mode == QuantumAttnKernelMode.FUSED_HERMITIAN_BORN_RHO_KET.value:
                svd_blk.berry_qk_phase.requires_grad_(True)
                svd_blk.qk_born_beta.requires_grad_(True)
            if self.graft_cfg.wte_init_mode == QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE:
                svd_blk.calibrate_from_teacher_block0_(t_blk, self.w_in_bridge)
            else:
                svd_blk.calibrate_cold_start_block_(t_blk)
            if self.graft_cfg.ket_amplitude_mode == QuantumKetAmplitudeMode.ENTROPY_PRESERVING_COHERENT_KET:
                with torch.no_grad():
                    if self.graft_cfg.wte_init_mode == QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE:
                        svd_blk.obs_scale.fill_(DEFAULT_COHERENT_OBS_SCALE_INIT)
                    svd_blk.rho_coupling_scale.fill_(rho_init)
                svd_blk.rho_coupling_scale.requires_grad_(True)
            self.quantum_attn_blocks[str(l_idx)] = svd_blk

        self.quantum_mlp_blocks = nn.ModuleDict()
        for l_idx in self.replaced_mlp_layers:
            t_blk = self.classical.blocks[l_idx]
            mz_mlp = SVDStiefelMachZehnderInterferometer(
                d_model=self.embed_dim,
                d_hidden=self.hidden_dim,
                proj_mode=self.graft_cfg.mlp_stiefel_proj_mode,
                eps=self.cfg.eps,
            )
            if self.graft_cfg.wte_init_mode == QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE:
                mz_mlp.calibrate_from_teacher_block_(t_blk)
            if use_doc_swc:
                with torch.no_grad():
                    mz_mlp.mz_theta.fill_(DEFAULT_DOC_SWC_MZ_THETA_INIT)
                    mz_mlp.mz_gamma.fill_(DEFAULT_DOC_SWC_MZ_GAMMA_INIT)
            self.quantum_mlp_blocks[str(l_idx)] = mz_mlp

        self.q_value_embeds: nn.ParameterList | None = None
        self.ve_bank_scales: nn.Parameter | None = None
        self.q_smear_gate: nn.Linear | None = None
        self.q_smear_lambda: nn.Parameter | None = None
        self.q_skip_weights: nn.Parameter | None = None
        self.q_skip_attn_lambdas = nn.ParameterDict()
        self.q_rope = copy.deepcopy(self.classical.rope)
        if self.full_blocks_replaced:
            self._init_standalone_full_backbone_modules()

    def _resolve_replaced_attn_layers(self) -> tuple[int, ...]:
        """Resolve which layer indices have their attention sublayer replaced by quantum wavepacket attention."""
        scope = self.graft_cfg.depth_scope
        if scope == QuantumBackboneDepthScope.EMB_PLUS_L0_TO_L2_ATTN:
            max_l = min(3, self.num_layers)
        elif scope == QuantumBackboneDepthScope.EMB_PLUS_L0_TO_L5_ATTN:
            max_l = min(6, self.num_layers)
        else:
            max_l = self.num_layers
        return tuple(l_idx for l_idx in range(max_l) if not self.classical.blocks[l_idx].skip_attention)

    @torch.no_grad()
    def _init_standalone_full_backbone_modules(self) -> None:
        """Initialize polar-factorized sphere `q_value_embeds`, `q_smear_gate`, `q_skip_weights`, and Layer-7 lambdas."""
        from ..optim.vocab_manifold_muon import project_rows_to_rms_sphere_

        if self.classical.value_embeds is not None and len(self.classical.value_embeds) > 0:
            q_ve_list: list[nn.Parameter] = []
            for b_idx, ve_emb in enumerate(self.classical.value_embeds):
                ve_t = ve_emb.weight.detach().to(device="cpu", dtype=torch.float32).clone()
                if self.w_in_bridge is not None:
                    b_cpu = self.w_in_bridge.detach().to(device="cpu", dtype=torch.float32)
                    ve_proj = torch.matmul(ve_t, b_cpu if ve_t.size(-1) == b_cpu.size(0) else b_cpu.t())
                else:
                    ve_proj = ve_t
                ve_3d = ve_proj.reshape(self.vocab_size, self.wte_slices, self.wte_sphere_dim).clone()
                slice_rms = ve_3d.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt().contiguous()
                project_rows_to_rms_sphere_(ve_3d)
                q_ve_list.append(nn.Parameter(ve_3d))
                self.register_buffer(f"ve_slice_rms_{b_idx}", slice_rms, persistent=True)
            self.q_value_embeds = nn.ParameterList(q_ve_list)
            self.ve_bank_scales = nn.Parameter(torch.ones(len(q_ve_list), dtype=torch.float32))

        if self.classical.smear_gate is not None and self.classical.smear_lambda is not None:
            self.q_smear_gate = copy.deepcopy(self.classical.smear_gate)
            self.q_smear_gate.weight.requires_grad_(True)
            self.q_smear_lambda = nn.Parameter(self.classical.smear_lambda.detach().float().clone())

        self.q_skip_weights = nn.Parameter(self.classical.skip_weights.detach().float().clone())
        for l_idx in range(self.num_layers):
            if self.classical.blocks[l_idx].skip_attention:
                self.q_skip_attn_lambdas[str(l_idx)] = nn.Parameter(
                    self.classical.blocks[l_idx].block_lambdas.detach().float().clone()
                )

    @classmethod
    def from_teacher_checkpoint(
        cls,
        ckpt_path: str | Path = DEFAULT_TEACHER_300M_CKPT_PATH,
        graft_cfg: QuantumFullBackboneGraftConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> QuantumFullBackboneGraftGPT:
        """Load a pretrained `CompiledLeaderboardClassicalGPT` teacher checkpoint and construct the quantum backbone."""
        ckpt = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
        raw_cfg = dict(ckpt["config"])
        for k_tup in ("skip_attn_layers", "long_window_layers"):
            if k_tup in raw_cfg:
                raw_cfg[k_tup] = tuple(raw_cfg[k_tup])
        c_cfg = CompiledLeaderboardClassicalConfig(**raw_cfg)
        teacher = CompiledLeaderboardClassicalGPT(c_cfg)
        teacher.load_state_dict(ckpt["model_state_dict"], strict=True)
        ws_blocks = int(ckpt.get("current_ws_blocks", c_cfg.initial_ws_blocks))
        teacher.set_sliding_window_blocks(ws_blocks, apply_yarn=True)
        return cls(teacher, graft_cfg=graft_cfg).to(device)

    @classmethod
    def from_scratch(
        cls,
        ckpt0_path: str | Path = DEFAULT_TEACHER_0M_CKPT_PATH,
        graft_cfg: QuantumFullBackboneGraftConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> QuantumFullBackboneGraftGPT:
        """Construct a cold-start 12-layer Quantum Wavepacket GPT with zero pretrained teacher weights (`step=0`)."""
        cfg_eff = copy.deepcopy(graft_cfg) if graft_cfg is not None else QuantumFullBackboneGraftConfig()
        cfg_eff.wte_init_mode = QuantumWteInitMode.PPMI_SVD_SPHERE
        cfg_eff.recovery_schedule_mode = QuantumRecoveryScheduleMode.PURE_CE
        cfg_eff.interface_warmup_steps = 0

        p0 = Path(ckpt0_path)
        if p0.exists():
            ckpt = torch.load(p0, map_location="cpu", weights_only=False)
            raw_cfg = dict(ckpt["config"])
            for k_tup in ("skip_attn_layers", "long_window_layers"):
                if k_tup in raw_cfg:
                    raw_cfg[k_tup] = tuple(raw_cfg[k_tup])
            c_cfg = CompiledLeaderboardClassicalConfig(**raw_cfg)
            base_model = CompiledLeaderboardClassicalGPT(c_cfg)
            base_model.load_state_dict(ckpt["model_state_dict"], strict=True)
            ws_blocks = int(ckpt.get("current_ws_blocks", c_cfg.initial_ws_blocks))
            base_model.set_sliding_window_blocks(ws_blocks, apply_yarn=True)
        else:
            base_model = CompiledLeaderboardClassicalGPT()

        model = cls(base_model, graft_cfg=cfg_eff)
        model.from_scratch_mode = True
        if not model.full_blocks_replaced:
            model._init_standalone_full_backbone_modules()
        model.classical.lm_head.weight.requires_grad_(True)
        for l_idx, blk in enumerate(model.classical.blocks):
            if str(l_idx) not in model.quantum_mlp_blocks:
                blk.fc1.weight.requires_grad_(True)
                blk.fc2.weight.requires_grad_(True)
        return model.to(device)

    @torch.no_grad()
    def set_sliding_window_blocks(
        self,
        ws_blocks: int,
        ws_final_blocks: int | None = None,
        apply_yarn: bool = True,
    ) -> None:
        """Update Long/Short/Final Doc-SWC window buffers and YaRN scale in-place (`<= 8` args)."""
        old_ws = int(self.classical.current_ws_blocks)
        new_ws = max(1, int(ws_blocks))
        self.classical.set_sliding_window_blocks(new_ws, ws_final_blocks=ws_final_blocks, apply_yarn=apply_yarn)
        if apply_yarn and self.q_rope is not None and new_ws != old_ws:
            if new_ws > old_ws:
                self.q_rope.apply_yarn(old_ws, new_ws, block_size=128)
            else:
                self.q_rope.reset_yarn()

    def _ve_bank_index(self, layer_idx: int) -> int | None:
        """Map layer index `l` to value embedding bank index in `[0, n_ve_banks)`."""
        n_banks = int(self.classical.n_ve_banks)
        if n_banks <= 0:
            return None
        if layer_idx < n_banks:
            return layer_idx
        if layer_idx >= self.num_layers - n_banks:
            return layer_idx - (self.num_layers - n_banks)
        return None

    def _lookup_value_embedding_768(self, bank_idx: int, idx: torch.Tensor, work_dtype: torch.dtype) -> torch.Tensor:
        """Look up token value embedding bank `bank_idx` in `(B, T, 768)` from `q_value_embeds` or frozen teacher."""
        bsz, seq_len = idx.shape
        if self.q_value_embeds is not None and self.ve_bank_scales is not None and bank_idx < len(self.q_value_embeds):
            q_ve = F.embedding(idx, self.q_value_embeds[bank_idx].view(self.vocab_size, self.d_emb)).view(
                bsz, seq_len, self.wte_slices, self.wte_sphere_dim
            )
            rms_buf = getattr(self, f"ve_slice_rms_{bank_idx}")
            rms_tok = F.embedding(idx, rms_buf.view(self.vocab_size, self.wte_slices)).unsqueeze(-1)
            ve_wp = (q_ve.to(work_dtype) * rms_tok.to(work_dtype)) * self.ve_bank_scales[bank_idx].to(work_dtype)
            return apply_boundary_stiefel_bridge(ve_wp.reshape(bsz, seq_len, self.d_emb), self.w_in_bridge)
        with torch.no_grad():
            assert self.classical.value_embeds is not None
            return self.classical.value_embeds[bank_idx](idx).to(work_dtype)

    def _prepare_quantum_ve_or_v0_packets(
        self,
        layer_idx: int,
        svd_blk: SVDStiefelWavepacketStatMechBlock,
        x_cur: torch.Tensor,
        x_0: torch.Tensor,
        ve_768: torch.Tensor | None,
        v_0_wp: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Prepare gated token value embedding or interior `v_0` wavepacket tensor `(B, T, M_pkt, d_h)`."""
        bsz, seq_len, _ = x_cur.shape
        work_dtype = x_cur.dtype
        if (
            self.graft_cfg.ve_coupling_mode == QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED
            and ve_768 is not None
        ):
            ve_wp = map_teacher_768_to_quantum_packets(ve_768, self.w_in_bridge, self.pkt_count, self.pkt_dim)
            if svd_blk.q_ve_gate is not None:
                l0, l1 = svd_blk.block_lambdas[0].to(work_dtype), svd_blk.block_lambdas[1].to(work_dtype)
                h_ve = F.rms_norm(l0 * x_cur + l1 * x_0, (self.embed_dim,), eps=self.cfg.eps)
                c_ve = svd_blk.ve_part_ch
                ve_in = torch.cat([h_ve[..., :c_ve], ve_768[..., :c_ve]], dim=-1)
                g_ve = (2.0 * torch.sigmoid(F.linear(ve_in, svd_blk.q_ve_gate.weight.to(work_dtype)))).view(
                    bsz, seq_len, svd_blk.num_heads, 1, 1
                )
                ve_wp = (ve_wp.view(bsz, seq_len, svd_blk.num_heads, -1, self.pkt_dim) * g_ve).reshape(
                    bsz, seq_len, self.pkt_count, self.pkt_dim
                )
            return ve_wp
        if layer_idx > 0 and self.cfg.interior_value_mode == "v0_residual" and v_0_wp is not None:
            return v_0_wp
        return None

    def _select_layer_attn_mask(
        self,
        l_idx: int,
        is_long: bool,
        masks: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
    ) -> torch.Tensor | None:
        """Select `(mask_long, mask_short, mask_final)` for layer index `l_idx`."""
        m_long, m_short, m_final = masks
        if l_idx == self.num_layers - 1:
            return m_final
        return m_long if is_long else m_short

    def _run_classical_block_sdpa(
        self,
        t_blk: Any,
        x: torch.Tensor,
        x_0: torch.Tensor,
        cos_4d: torch.Tensor,
        sin_4d: torch.Tensor,
        v_0: torch.Tensor | None,
        ve_l: torch.Tensor | None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Execute a classical transformer block with optional `(B, 1, T, T)` boolean `Doc-SWC` mask (`<= 8` args)."""
        q_scale = self.classical.rope.yarn_q_scale
        if attn_mask is None:
            return t_blk(x, x_0, cos_4d, sin_4d, v_0=v_0, ve_l=ve_l, block_mask=None, q_scale=q_scale)
        x = t_blk.block_lambdas[0].to(x.dtype) * x + t_blk.block_lambdas[1].to(x.dtype) * x_0
        v_0_out = v_0
        if not t_blk.skip_attention and t_blk.qkv_proj is not None and t_blk.out_proj is not None:
            bsz, seq_len, d_model = x.shape
            h = F.rms_norm(x, (d_model,), eps=t_blk.eps)
            q, k, v = t_blk.qkv_proj(h).view(bsz, seq_len, 3, t_blk.num_heads, t_blk.head_dim).unbind(dim=2)
            if v_0_out is None:
                v_0_out = v
            if t_blk.sa_lambdas is not None:
                s0, s1 = t_blk.sa_lambdas[0].to(v.dtype), t_blk.sa_lambdas[1].to(v.dtype)
                if ve_l is not None:
                    ve_heads = ve_l.view(bsz, seq_len, t_blk.num_heads, t_blk.head_dim)
                    if t_blk.ve_gate is not None:
                        c_ve = t_blk.ve_part_ch
                        ve_g = (2.0 * torch.sigmoid(t_blk.ve_gate(torch.cat([h[..., :c_ve], ve_l[..., :c_ve]], dim=-1)))).unsqueeze(-1)
                        ve_heads = ve_heads * ve_g
                    v = s0 * v + s1 * ve_heads
                elif t_blk.interior_value_mode == "v0_residual" and v_0_out is not None:
                    v = s0 * v + s1 * v_0_out
                else:
                    v = s0 * v
            q = apply_native_leaderboard_rotary(F.rms_norm(q, (t_blk.head_dim,), eps=t_blk.eps), cos_4d, sin_4d) * q_scale.to(q.dtype)
            k = apply_native_leaderboard_rotary(F.rms_norm(k, (t_blk.head_dim,), eps=t_blk.eps), cos_4d, sin_4d)
            if t_blk.use_partial_key_offset and seq_len > 1:
                d_half = t_blk.head_dim // 2
                k = torch.cat([k[..., :d_half], torch.cat([k[:, :1, :, d_half:], k[:, :-1, :, d_half:]], dim=1)], dim=-1)
            q_b, k_b, v_b = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            attn_out = F.scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=attn_mask, is_causal=False, scale=t_blk.attn_scale)
            if t_blk.xsa_alpha is not None:
                v_unit = F.normalize(v_b, dim=-1, eps=1e-5)
                alpha_h = torch.tanh(t_blk.xsa_alpha).view(1, t_blk.num_heads, 1, 1).to(attn_out.dtype)
                attn_out = attn_out - alpha_h * ((attn_out * v_unit).sum(dim=-1, keepdim=True) * v_unit)
            attn_heads = attn_out.transpose(1, 2)
            if t_blk.attn_gate is not None:
                attn_heads = attn_heads * (2.0 * torch.sigmoid(t_blk.attn_gate(h[..., : t_blk.gate_in_ch]))).unsqueeze(-1)
            x = x + t_blk.out_proj(attn_heads.reshape(bsz, seq_len, d_model))
        h_mlp = F.rms_norm(x, (t_blk.embed_dim,), eps=t_blk.eps)
        return x + t_blk.fc2(F.relu(t_blk.fc1(h_mlp)).square()), v_0_out

    def encode_all_layers(
        self, idx: torch.Tensor, collect_trajectories: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Run the 12-layer U-Net backbone returning `(x_final_norm, x_0, layer_outputs)`."""
        bsz, seq_len = idx.shape
        work_dtype = (
            torch.bfloat16
            if (idx.is_cuda and self.classical.lm_head.weight.dtype == torch.bfloat16)
            else torch.float32
        )
        raw_wp = (
            F.embedding(idx, self.q_wte.view(self.vocab_size, self.d_emb))
            .view(bsz, seq_len, self.pkt_count, self.pkt_dim)
            .to(work_dtype)
        )
        if seq_len > 1:
            raw_wp = apply_causal_wavepacket_dispersion(raw_wp, self.wp_causal_re, self.wp_causal_im)
        raw_768 = apply_boundary_stiefel_bridge(raw_wp.reshape(bsz, seq_len, self.d_emb), self.w_in_bridge)

        use_standalone_gates = self.full_blocks_replaced or self.from_scratch_mode
        smear_gate = self.q_smear_gate if use_standalone_gates else self.classical.smear_gate
        smear_lam = self.q_smear_lambda if use_standalone_gates else self.classical.smear_lambda
        if smear_gate is not None and smear_lam is not None and seq_len > 1:
            s_in = raw_768[:, 1:, : self.classical.smear_in_ch]
            s_gate = smear_lam.to(work_dtype) * torch.sigmoid(F.linear(s_in, smear_gate.weight.to(work_dtype)))
            raw_768 = torch.cat([raw_768[:, :1], raw_768[:, 1:] + s_gate * raw_768[:, :-1]], dim=1)

        x_0 = F.rms_norm(raw_768, (self.embed_dim,), eps=self.cfg.eps) * self.in_scale.to(work_dtype)
        rope_mod = self.q_rope if use_standalone_gates else self.classical.rope
        cos_4d, sin_4d = rope_mod.get_cos_sin(seq_len, dtype=work_dtype)
        q_scale = rope_mod.yarn_q_scale
        if self.graft_cfg.doc_swc_mode == QuantumDocSWCMode.TEACHER_DOC_SWC_MASK:
            masks: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] = build_doc_swc_sdpa_masks(
                idx,
                int(self.classical.long_window_buf.item()),
                int(self.classical.short_window_buf.item()),
                int(self.classical.final_window_buf.item()),
            )
        else:
            masks = (None, None, None)

        n_ve = int(self.classical.n_ve_banks)
        ve_banks_768 = [self._lookup_value_embedding_768(b, idx, work_dtype) for b in range(n_ve)]

        n_half = self.num_layers // 2
        skip_w = self.q_skip_weights if (use_standalone_gates and self.q_skip_weights is not None) else self.classical.skip_weights
        encoder_states: list[torch.Tensor] = []
        layer_outputs: list[torch.Tensor] = []
        x = x_0
        v_0_wp: torch.Tensor | None = None
        v_0_heads: torch.Tensor | None = None

        for l_idx in range(self.num_layers):
            if l_idx >= n_half and encoder_states:
                dec_i = l_idx - n_half
                skip_g = (
                    torch.sigmoid(skip_w[dec_i]).to(work_dtype)
                    if self.cfg.use_sigmoid_skip
                    else skip_w[dec_i].to(work_dtype)
                )
                x = x + skip_g * encoder_states.pop()

            bank_idx = self._ve_bank_index(l_idx)
            ve_l = ve_banks_768[bank_idx] if (bank_idx is not None and bank_idx < len(ve_banks_768)) else None
            l_key = str(l_idx)

            if l_key in self.quantum_attn_blocks:
                svd_blk: SVDStiefelWavepacketStatMechBlock = self.quantum_attn_blocks[l_key]  # type: ignore[assignment]
                ve_or_v0_wp = self._prepare_quantum_ve_or_v0_packets(l_idx, svd_blk, x, x_0, ve_l, v_0_wp)
                l_mask = self._select_layer_attn_mask(l_idx, svd_blk.is_long_window_layer, masks)
                need_ckpt = (
                    len(self.replaced_attn_layers) > 3
                    and self.training
                    and torch.is_grad_enabled()
                    and (svd_blk.d_h > 32 or (l_idx % 3 == 0))
                )
                if need_ckpt:
                    x_mid, v_wp = checkpoint(
                        execute_quantum_wavepacket_attn_layer,
                        svd_blk,
                        x,
                        x_0,
                        cos_4d,
                        sin_4d,
                        q_scale,
                        ve_or_v0_wp,
                        l_mask,
                        use_reentrant=False,
                    )
                else:
                    x_mid, v_wp = execute_quantum_wavepacket_attn_layer(
                        svd_blk,
                        x,
                        x_0,
                        cos_4d,
                        sin_4d,
                        q_scale,
                        ve_or_v0_wp=ve_or_v0_wp,
                        attn_mask=l_mask,
                    )
                if v_0_wp is None:
                    v_0_wp = v_wp
                    v_0_proj = apply_boundary_stiefel_bridge(v_wp.reshape(bsz, seq_len, self.d_emb), self.w_in_bridge)
                    v_0_heads = v_0_proj.view(bsz, seq_len, self.num_heads, self.embed_dim // self.num_heads)
                if l_key in self.quantum_mlp_blocks:
                    x = self.quantum_mlp_blocks[l_key](x_mid)
                else:
                    t_blk = self.classical.blocks[l_idx]
                    h_mlp = F.rms_norm(x_mid, (self.embed_dim,), eps=self.cfg.eps)
                    x = x_mid + t_blk.fc2(F.relu(t_blk.fc1(h_mlp)).square())
            elif l_key in self.quantum_mlp_blocks:
                lam_7 = self.q_skip_attn_lambdas[l_key]
                x_mid = lam_7[0].to(work_dtype) * x + lam_7[1].to(work_dtype) * x_0
                x = self.quantum_mlp_blocks[l_key](x_mid)
            else:
                t_blk = self.classical.blocks[l_idx]
                if l_key in self.q_skip_attn_lambdas and t_blk.skip_attention:
                    lam_7 = self.q_skip_attn_lambdas[l_key]
                    x_mid = lam_7[0].to(work_dtype) * x + lam_7[1].to(work_dtype) * x_0
                    h_mlp = F.rms_norm(x_mid, (self.embed_dim,), eps=self.cfg.eps)
                    x = x_mid + t_blk.fc2(F.relu(t_blk.fc1(h_mlp)).square())
                else:
                    l_mask = self._select_layer_attn_mask(l_idx, t_blk.is_long_window_layer, masks)
                    x, v_0_heads = self._run_classical_block_sdpa(
                        t_blk, x, x_0, cos_4d, sin_4d, v_0_heads, ve_l, attn_mask=l_mask
                    )

            if collect_trajectories:
                layer_outputs.append(x)
            if l_idx < n_half:
                encoder_states.append(x)

        x_final = F.rms_norm(x, (self.embed_dim,), eps=self.cfg.eps)
        return x_final, x_0, layer_outputs

    def forward_features(self, idx: torch.Tensor) -> torch.Tensor:
        """Encode token indices `(B, T)` into final RMSNormed hidden states `(B, T, 768)`."""
        x_final, _, _ = self.encode_all_layers(idx, collect_trajectories=False)
        return x_final

    @torch.no_grad()
    def _extract_teacher_doc_swc_trajectories(self, idx: torch.Tensor) -> dict[str, Any]:
        """Extract classical teacher trajectories using exact boolean `Doc-SWC` masks when enabled."""
        if self.graft_cfg.doc_swc_mode != QuantumDocSWCMode.TEACHER_DOC_SWC_MASK:
            return self.classical.extract_layer_trajectories(idx)
        _, seq_len = idx.shape
        raw_emb = self.classical.wte(idx)
        if self.classical.smear_gate is not None and self.classical.smear_lambda is not None and seq_len > 1:
            s_gate = self.classical.smear_lambda.to(raw_emb.dtype) * torch.sigmoid(
                self.classical.smear_gate(raw_emb[:, 1:, : self.classical.smear_in_ch])
            )
            raw_emb = torch.cat([raw_emb[:, :1], raw_emb[:, 1:] + s_gate * raw_emb[:, :-1]], dim=1)
        x_0 = F.rms_norm(raw_emb, (self.embed_dim,), eps=self.cfg.eps)
        x = x_0
        cos_4d, sin_4d = self.classical.rope.get_cos_sin(seq_len, dtype=x.dtype)
        masks = build_doc_swc_sdpa_masks(
            idx,
            int(self.classical.long_window_buf.item()),
            int(self.classical.short_window_buf.item()),
            int(self.classical.final_window_buf.item()),
        )
        ve_banks = [ve_emb(idx) for ve_emb in self.classical.value_embeds] if self.classical.value_embeds is not None else []
        n_half = self.num_layers // 2
        enc_states: list[torch.Tensor] = []
        outs_c: list[torch.Tensor] = []
        v_0_c: torch.Tensor | None = None
        for l_idx, block in enumerate(self.classical.blocks):
            if l_idx >= n_half and enc_states:
                dec_i = l_idx - n_half
                skip_g = torch.sigmoid(self.classical.skip_weights[dec_i]).to(x.dtype) if self.cfg.use_sigmoid_skip else self.classical.skip_weights[dec_i].to(x.dtype)
                x = x + skip_g * enc_states.pop()
            b_idx = self._ve_bank_index(l_idx)
            ve_l = ve_banks[b_idx] if (b_idx is not None and b_idx < len(ve_banks)) else None
            l_mask = self._select_layer_attn_mask(l_idx, block.is_long_window_layer, masks)
            x, v_0_c = self._run_classical_block_sdpa(block, x, x_0, cos_4d, sin_4d, v_0_c, ve_l, attn_mask=l_mask)
            outs_c.append(x)
            if l_idx < n_half:
                enc_states.append(x)
        return {"x_0": x_0, "layer_outputs": outs_c, "final_norm_x": F.rms_norm(x, (self.embed_dim,), eps=self.cfg.eps)}

    def _compute_multilayer_alignment_loss(
        self,
        idx: torch.Tensor,
        x_0_q: torch.Tensor,
        layer_outputs_q: list[torch.Tensor],
        x_final_q: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine trajectory co-distillation loss across all replaced quantum layers."""
        with torch.no_grad():
            t_traj = self._extract_teacher_doc_swc_trajectories(idx)
            x_0_c = t_traj["x_0"].detach()
            outs_c = [t.detach() for t in t_traj["layer_outputs"]]
            x_final_c = t_traj["final_norm_x"].detach()

        replaced_set = sorted(set(self.replaced_attn_layers) | set(self.replaced_mlp_layers))
        layer_terms = [_normalized_cosine_mse(layer_outputs_q[l_i], outs_c[l_i]) for l_i in replaced_set]
        mean_layer_align = sum(layer_terms) / float(max(1, len(layer_terms)))
        raw_align = (
            0.5 * _normalized_cosine_mse(x_0_q, x_0_c)
            + mean_layer_align
            + DEFAULT_FINAL_FEAT_ALIGN_SCALE * _normalized_cosine_mse(x_final_q, x_final_c)
        )

        t_align = max(1, int(self.graft_cfg.interface_warmup_steps))
        w_init = float(self.graft_cfg.interface_align_weight)
        is_persistent = self.graft_cfg.recovery_schedule_mode == QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE
        w_floor = DEFAULT_PERSISTENT_ALIGN_FLOOR_RATIO * w_init if is_persistent else 0.0
        prog = min(1.0, float(max(0, self.training_step)) / float(t_align))
        lam_t = w_floor + 0.5 * (w_init - w_floor) * (1.0 + math.cos(math.pi * prog))
        return lam_t * raw_align

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Forward pass returning `(logits_or_none, total_loss_or_none)`."""
        t_align = int(self.graft_cfg.interface_warmup_steps)
        sched = self.graft_cfg.recovery_schedule_mode
        is_persistent = sched == QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE
        use_align = (
            self.training
            and targets is not None
            and t_align > 0
            and (0 <= self.training_step < t_align or is_persistent)
            and sched != QuantumRecoveryScheduleMode.PURE_CE
        )

        x_final, x_0, layer_outputs = self.encode_all_layers(idx, collect_trajectories=use_align)
        align_loss = self._compute_multilayer_alignment_loss(idx, x_0, layer_outputs, x_final) if use_align else None

        if targets is not None and not return_logits:
            ce_loss = _CompiledChunkedSoftcapLinearCEFn.apply(
                x_final, self.classical.lm_head.weight, targets, self.cfg.logit_softcap, self.cfg.ce_chunk_tokens, True
            )
            return None, (ce_loss + align_loss if align_loss is not None else ce_loss)

        logits = self.classical.compute_logits(x_final)
        total_loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1).long(), ignore_index=-1
            )
            total_loss = ce_loss + align_loss if align_loss is not None else ce_loss
        return logits, total_loss

    def quantum_sphere_embedding_parameters(self) -> list[nn.Parameter]:
        """Return 3D unit-RMS sphere embedding tables (`[q_wte]` plus `q_value_embeds` when full blocks replaced)."""
        params = [self.q_wte]
        if self.q_value_embeds is not None:
            params.extend(list(self.q_value_embeds))
        return [p for p in params if p.requires_grad]

    def quantum_real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Return all active real Stiefel manifold matrices across all replaced attention and MLP blocks."""
        params: list[nn.Parameter] = []
        if self.w_in_bridge is not None and self.w_in_bridge.requires_grad:
            params.append(self.w_in_bridge)
        for svd_blk in self.quantum_attn_blocks.values():
            params.extend(svd_blk.real_stiefel_parameters())  # type: ignore[operator]
        for mz_blk in self.quantum_mlp_blocks.values():
            params.extend(mz_blk.real_stiefel_parameters())  # type: ignore[operator]
        return params

    def quantum_complex_unitary_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        """Return complex head Unitary/Kraus pairs across all replaced quantum attention layers."""
        return [svd_blk.complex_unitary_pair() for svd_blk in self.quantum_attn_blocks.values()]  # type: ignore[misc]

    def quantum_scalar_parameters(self) -> list[nn.Parameter]:
        """Return all active 1D scalar and gate parameters across the quantum backbone."""
        params: list[nn.Parameter] = [self.in_scale, self.wp_causal_re, self.wp_causal_im]
        if self.ve_bank_scales is not None and self.ve_bank_scales.requires_grad:
            params.append(self.ve_bank_scales)
        if self.q_smear_lambda is not None and self.q_smear_lambda.requires_grad:
            params.append(self.q_smear_lambda)
        if self.q_smear_gate is not None:
            params.extend([p for p in self.q_smear_gate.parameters() if p.requires_grad])
        if self.q_skip_weights is not None and self.q_skip_weights.requires_grad:
            params.append(self.q_skip_weights)
        for lam_p in self.q_skip_attn_lambdas.values():
            if lam_p.requires_grad:
                params.append(lam_p)
        for svd_blk in self.quantum_attn_blocks.values():
            params.extend(svd_blk.scalar_parameters())  # type: ignore[operator]
        for mz_blk in self.quantum_mlp_blocks.values():
            params.extend(mz_blk.scalar_parameters())  # type: ignore[operator]
        return [p for p in params if p.requires_grad]

    def active_quantum_parameters(self) -> list[nn.Parameter]:
        """Return all trainable quantum parameters across the entire replaced backbone."""
        params: list[nn.Parameter] = []
        params.extend(self.quantum_sphere_embedding_parameters())
        params.extend(self.quantum_real_stiefel_parameters())
        for u_re, u_im in self.quantum_complex_unitary_pairs():
            if u_re.requires_grad:
                params.append(u_re)
            if u_im.requires_grad:
                params.append(u_im)
        params.extend(self.quantum_scalar_parameters())
        return params

    @torch.no_grad()
    def restrict_quantum_scalars_(self) -> None:
        """Clamp all quantum backbone scalar parameters to bounded physical intervals."""
        self.in_scale.clamp_(0.50, 2.00)
        self.wp_causal_re.clamp_(-1.50, 1.50)
        self.wp_causal_im.clamp_(-1.50, 1.50)
        if self.ve_bank_scales is not None:
            self.ve_bank_scales.clamp_(DEFAULT_VE_BANK_SCALE_MIN, DEFAULT_VE_BANK_SCALE_MAX)
        if self.q_smear_lambda is not None:
            self.q_smear_lambda.clamp_(-5.0, 5.0)
        if self.q_skip_weights is not None:
            self.q_skip_weights.clamp_(-10.0, 10.0)
        for lam_p in self.q_skip_attn_lambdas.values():
            lam_p.clamp_(-DEFAULT_CALIBRATED_LAMBDA_BOUND, DEFAULT_CALIBRATED_LAMBDA_BOUND)
        for svd_blk in self.quantum_attn_blocks.values():
            svd_blk.restrict_scalars_()  # type: ignore[operator]
        for mz_blk in self.quantum_mlp_blocks.values():
            mz_blk.restrict_scalars_()  # type: ignore[operator]

    @torch.no_grad()
    def compute_quantum_graft_telemetry(self) -> dict[str, float]:
        """Compute max Stiefel orthogonality error, sphere embedding RMS error, and mean quantum purity/rank."""
        st_errs: list[float] = []
        for w in self.quantum_real_stiefel_parameters():
            wf = w.float()
            if wf.size(-2) < wf.size(-1):
                wf = wf.transpose(-2, -1)
            d_cols = wf.size(-1)
            eye = torch.eye(d_cols, device=wf.device, dtype=wf.dtype)
            gram = torch.matmul(wf.transpose(-2, -1), wf)
            st_errs.append(float(((gram - eye).norm(dim=(-2, -1)) / math.sqrt(float(d_cols))).max().item()))
        u_errs: list[float] = []
        for ur, ui in self.quantum_complex_unitary_pairs():
            urf, uif = ur.float(), ui.float()
            d_u = urf.size(-1)
            eye_u = torch.eye(d_u, device=urf.device, dtype=urf.dtype)
            gram_re = torch.matmul(urf, urf.transpose(-2, -1)) + torch.matmul(uif, uif.transpose(-2, -1))
            gram_im = torch.matmul(uif, urf.transpose(-2, -1)) - torch.matmul(urf, uif.transpose(-2, -1))
            err_u = ((gram_re - eye_u).square() + gram_im.square()).sum(dim=(-2, -1)).sqrt() / math.sqrt(float(d_u))
            u_errs.append(float(err_u.max().item()))
        sphere_errs: list[float] = []
        for emb_p in self.quantum_sphere_embedding_parameters():
            rms = emb_p.float().pow(2).mean(dim=-1).sqrt()
            sphere_errs.append(float((rms - 1.0).abs().max().item()))
        purities: list[float] = []
        ranks: list[float] = []
        for svd_blk in self.quantum_attn_blocks.values():
            tel = svd_blk.get_last_telemetry_dict()  # type: ignore[operator]
            purities.append(float(tel["mean_purity"]))
            ranks.append(float(tel["mean_eff_rank"]))
        return {
            "q_max_stiefel_err": max(st_errs) if st_errs else 0.0,
            "q_max_unitary_err": max(u_errs) if u_errs else 0.0,
            "q_wte_sphere_err": max(sphere_errs) if sphere_errs else 0.0,
            "q_l0_purity": sum(purities) / float(max(1, len(purities))),
            "q_l0_eff_rank": sum(ranks) / float(max(1, len(ranks))),
        }


@torch.no_grad()
def retract_quantum_backbone_manifolds_(model: QuantumFullBackboneGraftGPT) -> None:
    """Retract all real Stiefel matrices, complex head unitaries, and 3D sphere embeddings onto their exact manifolds."""
    from ..optim.vocab_manifold_muon import project_rows_to_rms_sphere_

    for w in model.quantum_real_stiefel_parameters():
        wf = w.detach().float()
        transposed = wf.size(-2) < wf.size(-1)
        if transposed:
            wf = wf.transpose(-2, -1)
        cols = wf.size(-1)
        eye = torch.eye(cols, device=wf.device, dtype=wf.dtype)
        gram0 = torch.matmul(wf.transpose(-2, -1), wf)
        scale0 = gram0.norm(dim=(-2, -1), keepdim=True).div(math.sqrt(float(cols))).clamp_min(1.0).sqrt()
        wf = wf / scale0
        for _ in range(6):
            wf = 0.5 * torch.matmul(wf, 3.0 * eye - torch.matmul(wf.transpose(-2, -1), wf))
        if transposed:
            wf = wf.transpose(-2, -1)
        w.copy_(wf.to(w.dtype))

    for u_re, u_im in model.quantum_complex_unitary_pairs():
        ur, ui = u_re.detach().float(), u_im.detach().float()
        d_u = ur.size(-1)
        eye_u = torch.eye(d_u, device=ur.device, dtype=ur.dtype)
        g0_re = torch.matmul(ur.transpose(-2, -1), ur) + torch.matmul(ui.transpose(-2, -1), ui)
        scale_u = g0_re.norm(dim=(-2, -1), keepdim=True).div(math.sqrt(float(d_u))).clamp_min(1.0).sqrt()
        ur, ui = ur / scale_u, ui / scale_u
        for _ in range(6):
            g_re = torch.matmul(ur.transpose(-2, -1), ur) + torch.matmul(ui.transpose(-2, -1), ui)
            g_im = torch.matmul(ur.transpose(-2, -1), ui) - torch.matmul(ui.transpose(-2, -1), ur)
            m_re = 3.0 * eye_u - g_re
            m_im = -g_im
            nr = 0.5 * (torch.matmul(ur, m_re) - torch.matmul(ui, m_im))
            ni = 0.5 * (torch.matmul(ur, m_im) + torch.matmul(ui, m_re))
            ur, ui = nr, ni
        u_re.copy_(ur.to(u_re.dtype))
        u_im.copy_(ui.to(u_im.dtype))

    for emb_p in model.quantum_sphere_embedding_parameters():
        project_rows_to_rms_sphere_(emb_p)


class QuantumFullBackboneOptimizerBundle:
    """Riemannian optimizer bundle stepping all active quantum backbone parameters (`q_trunk_opt`, `q_wte_opt`, `q_scalar_opt`, plus optional cold-start `mlp_muon_opt` and `head_adam_opt`)."""

    def __init__(
        self,
        model: QuantumFullBackboneGraftGPT,
        q_trunk_opt: Any,
        q_wte_opt: Any,
        q_scalar_opt: torch.optim.Optimizer,
        base_scalar_lr: float = DEFAULT_FULL_BACKBONE_SCALAR_LR,
        mlp_muon_opt: Any = None,
        head_adam_opt: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.q_trunk_opt = q_trunk_opt
        self.q_wte_opt = q_wte_opt
        self.q_scalar_opt = q_scalar_opt
        self.base_scalar_lr = float(base_scalar_lr)
        self.mlp_muon_opt = mlp_muon_opt
        self.head_adam_opt = head_adam_opt

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients across all active quantum and cold-start optimizer groups."""
        for opt in (self.q_trunk_opt, self.q_wte_opt, self.q_scalar_opt, self.mlp_muon_opt, self.head_adam_opt):
            if opt is not None:
                opt.zero_grad(set_to_none=set_to_none)

    def step(self, idx: torch.Tensor, lr_mult: float = 1.0, momentum_beta: float | None = None) -> None:
        """Step Stiefel/Unitary Muon, active-row 3D sphere `q_wte`/`q_value_embeds`, scalar AdamW, and cold-start MLP/head."""
        if self.q_trunk_opt is not None:
            if momentum_beta is not None:
                self.q_trunk_opt.set_momentum(momentum_beta)
            self.q_trunk_opt.set_lr_scale(lr_mult)
            self.q_trunk_opt.step()
        if self.q_wte_opt is not None:
            if momentum_beta is not None:
                self.q_wte_opt.set_momentum(momentum_beta)
            idx_in = torch.unique(idx.reshape(-1))
            self.q_wte_opt.step_active(idx_in, lr_scale=lr_mult)
        if self.q_scalar_opt is not None:
            for grp in self.q_scalar_opt.param_groups:
                grp["lr"] = self.base_scalar_lr * lr_mult
            self.q_scalar_opt.step()
            self.model.restrict_quantum_scalars_()
        if self.mlp_muon_opt is not None:
            for grp in self.mlp_muon_opt.param_groups:
                grp["lr"] = DEFAULT_FROM_SCRATCH_MLP_MUON_LR * lr_mult
            self.mlp_muon_opt.step()
        if self.head_adam_opt is not None:
            for grp in self.head_adam_opt.param_groups:
                grp["lr"] = DEFAULT_FROM_SCRATCH_HEAD_LR * lr_mult
            self.head_adam_opt.step()
        self.model.training_step += 1


def build_quantum_full_backbone_optimizers(
    model: QuantumFullBackboneGraftGPT,
    stiefel_lr: float | None = None,
    unitary_lr: float | None = None,
    wte_lr: float | None = None,
    scalar_lr: float | None = None,
) -> QuantumFullBackboneOptimizerBundle:
    """Build Riemannian Stiefel/Unitary + Sphere + Scalar optimizers for `QuantumFullBackboneGraftGPT` (`<= 8` args)."""
    from ..optim.muon import StiefelMuon, StiefelMuonConfig
    from ..optim.vocab_manifold_muon import BatchSubspaceTikhonovGramMuon
    from .modular_quantum_statmech_gpt import ModularRiemannianStiefelKrausMuon

    cfg = model.graft_cfg
    n_sub = len(model.replaced_attn_layers) + len(model.replaced_mlp_layers)
    is_from_scratch = bool(getattr(model, "from_scratch_mode", False))
    use_depth_scale = (not is_from_scratch) and cfg.depth_lr_mode in (
        QuantumDepthLRMode.DEPTH_SQRT_WARMUP_COSINE,
        QuantumDepthLRMode.DEPTH_SQRT_WARMUP_WSD_COOLDOWN,
    )
    d_scale = compute_depth_lr_scale(n_sub) if use_depth_scale else 1.0
    eff_stiefel_lr = float(stiefel_lr if stiefel_lr is not None else cfg.stiefel_lr) * d_scale
    eff_unitary_lr = float(unitary_lr if unitary_lr is not None else cfg.unitary_lr) * d_scale
    eff_wte_lr = float(wte_lr if wte_lr is not None else cfg.wte_lr) * d_scale
    eff_scalar_lr = float(scalar_lr if scalar_lr is not None else cfg.scalar_lr) * d_scale

    q_stiefel = model.quantum_real_stiefel_parameters()
    q_trunk_opt = (
        ModularRiemannianStiefelKrausMuon(
            real_stiefel_params=q_stiefel,
            complex_operator_pairs=model.quantum_complex_unitary_pairs(),
            stiefel_lr=eff_stiefel_lr,
            unitary_lr=eff_unitary_lr,
            tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA,
            momentum_mode=StiefelMomentumMode(cfg.stiefel_momentum_mode),
        )
        if q_stiefel
        else None
    )
    q_wte_opt = BatchSubspaceTikhonovGramMuon(
        model.quantum_sphere_embedding_parameters(),
        lr=eff_wte_lr,
        tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA,
    )
    q_scalar_opt = torch.optim.AdamW(
        [{"params": model.quantum_scalar_parameters(), "lr": eff_scalar_lr, "weight_decay": 0.0}],
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    mlp_muon_opt: Any = None
    head_adam_opt: torch.optim.Adam | None = None
    if getattr(model, "from_scratch_mode", False):
        mlp_params: list[nn.Parameter] = []
        for l_idx, blk in enumerate(model.classical.blocks):
            if str(l_idx) not in model.quantum_mlp_blocks:
                if blk.fc1.weight.requires_grad:
                    mlp_params.append(blk.fc1.weight)
                if blk.fc2.weight.requires_grad:
                    mlp_params.append(blk.fc2.weight)
        if mlp_params:
            mlp_muon_opt = StiefelMuon(
                mlp_params,
                lr=DEFAULT_FROM_SCRATCH_MLP_MUON_LR,
                momentum=0.95,
                weight_decay=0.0,
                decay_mode="none",
                aspect_scale=True,
                config=StiefelMuonConfig(
                    nesterov=True,
                    ns_steps=5,
                    polar_method="polar_express",
                    stiefel_penalty=0.0,
                    use_normuon=True,
                    use_twin_rail=True,
                ),
                momentum_warmup_steps=300,
                momentum_init=0.85,
                ema_momentum=True,
                batched_ns=True,
            )
        if model.classical.lm_head.weight.requires_grad:
            is_cuda = model.classical.lm_head.weight.is_cuda
            head_adam_opt = torch.optim.Adam(
                [{"params": [model.classical.lm_head.weight], "lr": DEFAULT_FROM_SCRATCH_HEAD_LR}],
                betas=(0.80, 0.95),
                eps=1e-8,
                fused=is_cuda,
            )

    return QuantumFullBackboneOptimizerBundle(
        model,
        q_trunk_opt,
        q_wte_opt,
        q_scalar_opt,
        base_scalar_lr=eff_scalar_lr,
        mlp_muon_opt=mlp_muon_opt,
        head_adam_opt=head_adam_opt,
    )


__all__ = [
    "DEFAULT_BASE_RHO_COUPLING_INIT",
    "DEFAULT_COHERENT_OBS_SCALE_INIT",
    "DEFAULT_DOC_SWC_MZ_GAMMA_INIT",
    "DEFAULT_DOC_SWC_MZ_THETA_INIT",
    "DEFAULT_DOC_SWC_RHO_COUPLING_INIT",
    "DEFAULT_FROM_SCRATCH_HEAD_LR",
    "DEFAULT_FROM_SCRATCH_MLP_MUON_LR",
    "DEFAULT_FROM_SCRATCH_SCALAR_LR",
    "DEFAULT_FROM_SCRATCH_STIEFEL_LR",
    "DEFAULT_FROM_SCRATCH_UNITARY_LR",
    "DEFAULT_FROM_SCRATCH_WTE_LR",
    "DEFAULT_FULL_BACKBONE_ALIGN_WEIGHT",
    "DEFAULT_FULL_BACKBONE_SCALAR_LR",
    "DEFAULT_FULL_BACKBONE_STIEFEL_LR",
    "DEFAULT_FULL_BACKBONE_UNITARY_LR",
    "DEFAULT_FULL_BACKBONE_WARMUP_STEPS",
    "DEFAULT_FULL_BACKBONE_WTE_LR",
    "DEFAULT_QUANTUM_TAIL_EMA_BLEND",
    "DEFAULT_QUANTUM_TAIL_EMA_RATE",
    "DEFAULT_QUANTUM_TAIL_EMA_START_FRAC",
    "DEFAULT_TEACHER_0M_CKPT_PATH",
    "DEFAULT_TEACHER_100M_CKPT_PATH",
    "QuantumAttnKernelMode",
    "QuantumBackboneDepthScope",
    "QuantumDepthLRMode",
    "QuantumDocSWCMode",
    "QuantumFullBackboneGraftConfig",
    "QuantumFullBackboneGraftGPT",
    "QuantumFullBackboneOptimizerBundle",
    "QuantumKetAmplitudeMode",
    "QuantumMLPChannelMode",
    "QuantumMomentumScheduleMode",
    "SVDStiefelMachZehnderInterferometer",
    "build_doc_swc_sdpa_masks",
    "build_quantum_full_backbone_optimizers",
    "compute_depth_lr_scale",
    "compute_quantum_backbone_lr_multiplier",
    "compute_riemannian_momentum_beta",
    "execute_quantum_wavepacket_attn_layer",
    "retract_quantum_backbone_manifolds_",
]
