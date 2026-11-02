r"""Surgical Quantum Front-End Graft onto a Frozen Pretrained Classical GPT (`EXPERIMENT_PLAN_065.md` & `066.md`).

Implements the configurable Quantum Front-End (`q_wte` + Quantum Layer 0) grafted onto a frozen
pretrained `CompiledLeaderboardClassicalGPT` teacher (`blocks[1..11]`, `value_embeds`, `skip_weights`, `lm_head`):

1. **Axis A — Wavefunction Shape Taxonomy (`QuantumWavefunctionShape`)**:
   - `SHAPE_24x2x16`: Physical 3D `(V, 48, 16)` (`D_emb = 768`), Layer 0 `(B, T, 48, 16)` ($H_q=6, R=4, d_h=16$).
   - `SHAPE_1x2x256`: Physical 3D `(V, 2, 256)` (`D_emb = 512`), Layer 0 `(B, T, 32, 16)` ($H_q=4, R=4, d_h=16$).
   - `SHAPE_1x2x512`: Physical 3D `(V, 2, 512)` (`D_emb = 1024`), Layer 0 `(B, T, 64, 16)` ($H_q=8, R=4, d_h=16$).
   - `SHAPE_256x2x2`: Physical 3D `(V, 256, 4)` (`D_emb = 1024`), Layer 0 `(B, T, 256, 4)` ($H_q=8 \times 32$ qubits, $d_h=2$).
2. **Axis B — State I/O Contract (`QuantumStateIOMode`)**:
   - `WF_IN_WF_OUT`, `DM_IN_DM_OUT`, and `WF_IN_DENSITY_OBS_OUT`.
3. **Axis C — Interior Attention Operations (`QuantumAttnInteriorMode`)**:
   - `ATTENDED_RHO_ON_ATTENDED_V`, `ATTENDED_RHO_ON_LOCAL_V`, `KV_CROSS_OUTER_ON_Q`, and `PRE_ATTN_LOCAL_OUTER_ON_V`.
4. **Axis D — Layer-0 Replacement Scope (`QuantumReplacementScope`)**:
   - `TEACHER_PASS_THROUGH`, `FULL_BLOCK0_QUANTUM`, and `ATTN0_QUANTUM_KEEP_FROZEN_MLP0`.
5. **Axes E, F, G (`EXPERIMENT_PLAN_066.md`)**:
   - `QuantumWteInitMode`: `PPMI_SVD_SPHERE` vs. `TEACHER_PROCRUSTES_SPHERE`.
   - `QuantumValueEmbedCouplingMode`: `NONE` vs. `FROZEN_TEACHER_VE0_COUPLED`.
   - `QuantumRecoveryScheduleMode`: `PURE_CE`, `INTERFACE_WARMUP_TO_PURE_CE`, `INTERFACE_AND_FINAL_FEAT_COSINE_TO_CE`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .impure_density_mirror_block import (
    DEFAULT_ROPE_BASE_THETA, DEFAULT_TIKHONOV_GRAM_LAMBDA, TRACE_NORM_EPS, initialize_stiefel_parameter,
)  # fmt: skip
from .isometric import apply_partial_key_offset
from .quantum_statmech_channels import (
    DEFAULT_KRAUS_CHANNEL_RANK, PurificationMode, QuantumChannelMode, StiefelMomentumMode,
    evolve_head_cptp_kraus_functional, evolve_head_unitary_traceless_functional,
    initialize_complex_kraus_stiefel_pair, purify_density_statmech,
)  # fmt: skip
from .quantum_svd_stiefel_graft import (
    BiorthogonalSVDStiefelProjection, DEFAULT_SVD_STIEFEL_OBS_SCALE_INIT, QuantumQKPhaseMode,
    QuantumSelfOrthoMode, QuantumStiefelProjMode, SVDStiefelWavepacketStatMechBlock,
    apply_berry_qk_phase, apply_boundary_stiefel_bridge, apply_fubini_study_self_ortho,
    apply_interior_rho_v_coupling, compute_liouville_commutator, compute_teacher_layer1_target,
    encode_svd_stiefel_layer0, encode_svd_stiefel_layer1,
    initialize_arbitrary_shape_ppmi_sphere_embeddings_, initialize_teacher_procrustes_sphere_embeddings_,
    map_teacher_768_to_quantum_packets, resolve_svd_stiefel_rank,
)  # fmt: skip
from .wavepacket_statmech_block import (
    DEFAULT_DM_LAG_MIX_INIT, DEFAULT_DM_RES_MIX_INIT, DEFAULT_PERSISTENT_ALIGN_FLOOR_RATIO,
    DEFAULT_WAVEPACKET_HILBERT_DIM, DEFAULT_WAVEPACKET_RESIDUAL_GAIN,
    DEFAULT_WAVEPACKET_SDPA_SCALE, apply_causal_wavepacket_dispersion, apply_factored_wavepacket_stiefel,
    build_configurable_rho_and_ket_sdpa_chunks, form_wavepacket_outer_product_and_v, initialize_packet_stiefel_3d,
)  # fmt: skip

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
_TEACHER_CKPT_DIR: Path = _REPO_ROOT / ".cache" / "checkpoints" / "w45b_classical_teacher_ckpt50m_l12_d768_b262k"
DEFAULT_TEACHER_300M_CKPT_PATH: Path = _TEACHER_CKPT_DIR / "ckpt_step01145_0300M_m0300M.pt"
DEFAULT_TEACHER_50M_CKPT_PATH: Path = _TEACHER_CKPT_DIR / "ckpt_step00191_0050M_m0050M.pt"

DEFAULT_WF_LARGE_256_REAL_DIM: int = 512
DEFAULT_WF_LARGE_512_REAL_DIM: int = 1024
DEFAULT_QUBIT_SPINOR_COUNT: int = 256
DEFAULT_QUBIT_SPHERE_DIM: int = 4
DEFAULT_QUBIT_HEADS: int = 8
DEFAULT_GRAFT_INTERFACE_ALIGN_WEIGHT: float = 1.0
DEFAULT_GRAFT_V0_ALIGN_SCALE: float = 0.25
DEFAULT_FINAL_FEAT_ALIGN_SCALE: float = 0.50
DEFAULT_VE0_LAMBDA_INIT: float = 0.25
DEFAULT_GRAFT_DIAG_OBS_INIT: float = 0.25
DEFAULT_GRAFT_STIEFEL_LR: float = 0.04
DEFAULT_GRAFT_UNITARY_LR: float = 0.02
DEFAULT_GRAFT_WTE_LR: float = 0.12
DEFAULT_GRAFT_SCALAR_LR: float = 0.04


class QuantumWavefunctionShape(str, Enum):
    """Axis A, H & Y enum selecting the token wavefunction factorization geometry (Rule 3)."""

    SHAPE_24x2x16 = "shape_24x2x16"
    SHAPE_1x2x256 = "shape_1x2x256"
    SHAPE_1x2x512 = "shape_1x2x512"
    SHAPE_256x2x2 = "shape_256x2x2"
    SHAPE_48x2x8 = "shape_48x2x8"
    SHAPE_12x2x32 = "shape_12x2x32"
    SHAPE_6x2x64 = "shape_6x2x64"
    SHAPE_1x2x512_PKT32 = "shape_1x2x512_pkt32"


class QuantumStateIOMode(str, Enum):
    """Axis B & I enum selecting complex wavefunction I/O vs. density-matrix Liouville I/O (Rule 3)."""

    WF_IN_WF_OUT = "wf_in_wf_out"
    DM_IN_DM_OUT = "dm_in_dm_out"
    WF_IN_DENSITY_OBS_OUT = "wf_in_density_obs_out"
    CAUSAL_ENSEMBLE_DM_IO = "causal_ensemble_dm_io"


class QuantumAttnInteriorMode(str, Enum):
    """Axis C & O enum selecting how outer products of wavefunctions combine $V$-space projections (Rule 3)."""

    ATTENDED_RHO_ON_ATTENDED_V = "attended_rho_on_attended_v"
    ATTENDED_RHO_ON_LOCAL_V = "attended_rho_on_local_v"
    KV_CROSS_OUTER_ON_Q = "kv_cross_outer_on_q"
    PRE_ATTN_LOCAL_OUTER_ON_V = "pre_attn_local_outer_on_v"
    SYMMETRIZED_LOCAL_CONTEXT_RHO_V = "symmetrized_local_context_rho_v"
    LIOUVILLE_COMMUTATOR_RHO_V = "liouville_commutator_rho_v"


class QuantumReplacementScope(str, Enum):
    """Axis D & P enum selecting whether to replace `blocks[0]`, `blocks[0].attn`, or `blocks[0..1].attn` (Rule 3)."""

    TEACHER_PASS_THROUGH = "teacher_pass_through"
    FULL_BLOCK0_QUANTUM = "full_block0_quantum"
    ATTN0_QUANTUM_KEEP_FROZEN_MLP0 = "attn0_quantum_keep_frozen_mlp0"
    EMB_PLUS_L0_L1 = "emb_plus_l0_l1"


class QuantumWteInitMode(str, Enum):
    """Axis E enum selecting PPMI-SVD sphere initialization vs. Teacher-Procrustes sphere alignment (Rule 3)."""

    PPMI_SVD_SPHERE = "ppmi_svd_sphere"
    TEACHER_PROCRUSTES_SPHERE = "teacher_procrustes_sphere"


class QuantumValueEmbedCouplingMode(str, Enum):
    """Axis F enum selecting whether Quantum Layer 0 couples `classical.value_embeds[0]` into $V$-space (Rule 3)."""

    NONE = "none"
    FROZEN_TEACHER_VE0_COUPLED = "frozen_teacher_ve0_coupled"


class QuantumRecoveryScheduleMode(str, Enum):
    """Axis G & J enum selecting pure CE vs. two-stage or persistent subspace co-distillation (Rule 3)."""

    PURE_CE = "pure_ce"
    INTERFACE_WARMUP_TO_PURE_CE = "interface_warmup_to_pure_ce"
    INTERFACE_AND_FINAL_FEAT_COSINE_TO_CE = "interface_and_final_feat_cosine_to_ce"
    PERSISTENT_COSINE_FEAT_AND_CE = "persistent_cosine_feat_and_ce"


@dataclass
class QuantumFrontEndGraftConfig:
    """Configuration for surgical quantum front-end replacement on a frozen classical trunk."""

    wf_shape: QuantumWavefunctionShape = QuantumWavefunctionShape.SHAPE_24x2x16
    state_io_mode: QuantumStateIOMode = QuantumStateIOMode.WF_IN_WF_OUT
    attn_interior_mode: QuantumAttnInteriorMode = QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V
    replacement_scope: QuantumReplacementScope = QuantumReplacementScope.FULL_BLOCK0_QUANTUM
    purification_mode: PurificationMode = PurificationMode.QUADRATIC_MCWEENY
    channel_mode: QuantumChannelMode = QuantumChannelMode.UNITARY_U16
    wte_init_mode: QuantumWteInitMode = QuantumWteInitMode.PPMI_SVD_SPHERE
    ve_coupling_mode: QuantumValueEmbedCouplingMode = QuantumValueEmbedCouplingMode.NONE
    recovery_schedule_mode: QuantumRecoveryScheduleMode = QuantumRecoveryScheduleMode.INTERFACE_WARMUP_TO_PURE_CE
    interface_warmup_steps: int = 0
    interface_align_weight: float = DEFAULT_GRAFT_INTERFACE_ALIGN_WEIGHT
    wte_lr: float = DEFAULT_GRAFT_WTE_LR
    stiefel_lr: float = DEFAULT_GRAFT_STIEFEL_LR
    stiefel_proj_mode: QuantumStiefelProjMode = QuantumStiefelProjMode.KRONEKER_FACTORED_STIEFEL
    self_ortho_mode: QuantumSelfOrthoMode = QuantumSelfOrthoMode.NONE
    qk_phase_mode: QuantumQKPhaseMode = QuantumQKPhaseMode.STANDARD_REAL_DOT


def resolve_wavefunction_shape_spec(wf_shape: QuantumWavefunctionShape) -> tuple[int, int, int, int, int, int]:
    """Return `(wte_slices, wte_sphere_dim, d_emb, pkt_count, pkt_dim, num_heads)` for `wf_shape`."""
    shape_mode = QuantumWavefunctionShape(wf_shape)
    if shape_mode == QuantumWavefunctionShape.SHAPE_1x2x256:
        return (2, 256, DEFAULT_WF_LARGE_256_REAL_DIM, 32, DEFAULT_WAVEPACKET_HILBERT_DIM, 4)
    if shape_mode == QuantumWavefunctionShape.SHAPE_1x2x512:
        return (2, 512, DEFAULT_WF_LARGE_512_REAL_DIM, 64, DEFAULT_WAVEPACKET_HILBERT_DIM, 8)
    if shape_mode == QuantumWavefunctionShape.SHAPE_1x2x512_PKT32:
        return (2, 512, DEFAULT_WF_LARGE_512_REAL_DIM, 32, 32, 8)
    if shape_mode == QuantumWavefunctionShape.SHAPE_48x2x8:
        return (96, 8, 768, 96, 8, 6)
    if shape_mode == QuantumWavefunctionShape.SHAPE_12x2x32:
        return (24, 32, 768, 24, 32, 6)
    if shape_mode == QuantumWavefunctionShape.SHAPE_6x2x64:
        return (12, 64, 768, 12, 64, 6)
    if shape_mode == QuantumWavefunctionShape.SHAPE_256x2x2:
        n_q, d_s = DEFAULT_QUBIT_SPINOR_COUNT, DEFAULT_QUBIT_SPHERE_DIM
        return (n_q, d_s, n_q * d_s, n_q, d_s, DEFAULT_QUBIT_HEADS)
    return (48, DEFAULT_WAVEPACKET_HILBERT_DIM, 768, 48, DEFAULT_WAVEPACKET_HILBERT_DIM, 6)


def _normalized_cosine_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Scale-invariant RMS-normalized MSE `0.5 * mean((rms_norm(a) - rms_norm(b))^2) = 1 - cos(a, b)` in `[0, 2]`."""
    d_last = a.size(-1)
    return 0.5 * F.mse_loss(F.rms_norm(a.float(), (d_last,), eps=1e-6), F.rms_norm(b.float(), (d_last,), eps=1e-6))


class _BaseQuantumGraftLayer0(nn.Module):
    """Shared base class for Quantum Layer-0 blocks providing factored Stiefel projections and Mach-Zehnder tail."""

    def __init__(self, num_pkts: int, d_pkt: int, num_heads: int, num_layers: int) -> None:
        super().__init__()
        self.num_pkts, self.d_pkt, self.num_heads = int(num_pkts), int(d_pkt), int(num_heads)
        self.beam_pkts = 2 * self.num_pkts
        self.include_mach_zehnder: bool = True

        self.block_lambdas = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float32))
        self.w_q_pkt = nn.Parameter(torch.empty(self.num_pkts, self.d_pkt, self.d_pkt))
        self.w_k_pkt = nn.Parameter(torch.empty(self.num_pkts, self.d_pkt, self.d_pkt))
        self.w_v_pkt = nn.Parameter(torch.empty(self.num_pkts, self.d_pkt, self.d_pkt))
        self.w_obs_pkt = nn.Parameter(torch.empty(self.num_pkts, self.d_pkt, self.d_pkt))
        self.w_beam_a_pkt = nn.Parameter(torch.empty(self.num_pkts, self.d_pkt, self.d_pkt))
        self.w_beam_b_pkt = nn.Parameter(torch.empty(self.num_pkts, self.d_pkt, self.d_pkt))
        self.w_chan_pkt = nn.Parameter(torch.empty(self.beam_pkts, self.d_pkt, self.d_pkt))
        self.m_q_mode = nn.Parameter(torch.empty(self.d_pkt, self.num_pkts, self.num_pkts))
        self.m_k_mode = nn.Parameter(torch.empty(self.d_pkt, self.num_pkts, self.num_pkts))
        self.m_v_mode = nn.Parameter(torch.empty(self.d_pkt, self.num_pkts, self.num_pkts))
        self.m_obs_mode = nn.Parameter(torch.empty(self.d_pkt, self.num_pkts, self.num_pkts))
        self.m_beam_a_mode = nn.Parameter(torch.empty(self.d_pkt, self.num_pkts, self.beam_pkts))
        self.m_beam_b_mode = nn.Parameter(torch.empty(self.d_pkt, self.num_pkts, self.beam_pkts))
        self.m_chan_mode = nn.Parameter(torch.empty(self.d_pkt, self.beam_pkts, self.num_pkts))
        for p3 in self._all_stiefel_tensors():
            initialize_packet_stiefel_3d(p3)

        init_gain = DEFAULT_WAVEPACKET_RESIDUAL_GAIN / math.sqrt(float(max(1, num_layers)))
        self.obs_scale = nn.Parameter(torch.tensor(init_gain, dtype=torch.float32))
        self.chan_scale = nn.Parameter(torch.tensor(init_gain, dtype=torch.float32))
        self.head_unitary_re: nn.Parameter
        self.head_unitary_im: nn.Parameter
        self._last_telemetry_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._last_rho0_re_im: tuple[torch.Tensor, torch.Tensor] | None = None

    def _all_stiefel_tensors(self) -> tuple[nn.Parameter, ...]:
        return (
            self.w_q_pkt, self.w_k_pkt, self.w_v_pkt, self.w_obs_pkt,
            self.w_beam_a_pkt, self.w_beam_b_pkt, self.w_chan_pkt,
            self.m_q_mode, self.m_k_mode, self.m_v_mode, self.m_obs_mode,
            self.m_beam_a_mode, self.m_beam_b_mode, self.m_chan_mode,
        )  # fmt: skip

    def disable_mach_zehnder_(self) -> None:
        """Disable and freeze Mach-Zehnder parameters when keeping frozen classical `blocks[0].mlp`."""
        self.include_mach_zehnder = False
        mz_params = (
            self.w_beam_a_pkt, self.w_beam_b_pkt, self.w_chan_pkt,
            self.m_beam_a_mode, self.m_beam_b_mode, self.m_chan_mode, self.chan_scale,
        )  # fmt: skip
        for p in mz_params:
            p.requires_grad_(False)

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Return active real Stiefel parameters in this block."""
        return [p for p in self._all_stiefel_tensors() if p.requires_grad]

    def complex_unitary_pair(self) -> tuple[nn.Parameter, nn.Parameter]:
        """Return the 3D complex head operator pair `(head_unitary_re, head_unitary_im)`."""
        return self.head_unitary_re, self.head_unitary_im

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return active 1D scalar parameters in this block."""
        params = [self.obs_scale, self.block_lambdas]
        if self.include_mach_zehnder and self.chan_scale.requires_grad:
            params.append(self.chan_scale)
        return params

    def get_last_telemetry_dict(self) -> dict[str, float]:
        """Convert detached GPU telemetry tensors to Python floats."""
        if self._last_telemetry_tensors is None:
            return {"mean_purity": 1.0, "mean_eff_rank": 1.0, "mean_lambda1": 0.5, "mean_lambda2": 0.5}
        p_t, er_t, l1_t, l2_t = self._last_telemetry_tensors
        return {
            "mean_purity": float(p_t.item()),
            "mean_eff_rank": float(er_t.item()),
            "mean_lambda1": float(l1_t.item()),
            "mean_lambda2": float(l2_t.item()),
        }

    def _expand_sdpa_qk(self, q_h: torch.Tensor, k_h: torch.Tensor, chunks: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand `(B, T, H, sdpa_dim)` query/key tensors across `chunks` operator rows per head."""
        bsz, seq_len, _, sdpa_dim = q_h.shape
        q_exp = (
            q_h.permute(0, 2, 1, 3)
            .unsqueeze(2)
            .expand(bsz, self.num_heads, chunks, seq_len, sdpa_dim)
            .reshape(bsz, self.num_heads * chunks, seq_len, sdpa_dim)
        )
        k_exp = (
            k_h.permute(0, 2, 1, 3)
            .unsqueeze(2)
            .expand(bsz, self.num_heads, chunks, seq_len, sdpa_dim)
            .reshape(bsz, self.num_heads * chunks, seq_len, sdpa_dim)
        )
        return q_exp, k_exp

    def _finish_readout_and_mach_zehnder(
        self, x_cur: torch.Tensor, y_wp: torch.Tensor, v_wp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project `y_wp` via `w_obs_pkt/m_obs_mode` and optionally apply the two-beam Mach-Zehnder tail."""
        work_dtype = x_cur.dtype
        d_obs = apply_factored_wavepacket_stiefel(y_wp, self.w_obs_pkt.to(work_dtype), self.m_obs_mode.to(work_dtype))
        d_obs_scaled = self.obs_scale.to(work_dtype) * d_obs
        x_mid = x_cur + d_obs_scaled
        if not self.include_mach_zehnder:
            return x_mid, d_obs_scaled, v_wp
        m_norm = F.rms_norm(x_mid, (self.d_pkt,), eps=TRACE_NORM_EPS)
        beam_a = apply_factored_wavepacket_stiefel(
            m_norm, self.w_beam_a_pkt.to(work_dtype), self.m_beam_a_mode.to(work_dtype)
        )
        beam_b = apply_factored_wavepacket_stiefel(
            m_norm, self.w_beam_b_pkt.to(work_dtype), self.m_beam_b_mode.to(work_dtype)
        )
        mz_wp = F.relu(beam_a + beam_b).square() - 0.25 * F.relu(beam_a - beam_b).square()
        d_chan = apply_factored_wavepacket_stiefel(mz_wp, self.w_chan_pkt.to(work_dtype), self.m_chan_mode.to(work_dtype))
        return x_mid + self.chan_scale.to(work_dtype) * d_chan, d_obs_scaled, v_wp


class QubitSpinorBlochStatMechBlock(_BaseQuantumGraftLayer0):
    r"""256-Qubit Spinor `(B, T, 256, 4)` Layer 0 with Analytical $2 \times 2$ Bloch McWeeny Purification & $U(2)$ Evolution."""

    def __init__(self, num_layers: int = 12, include_mach_zehnder: bool = True) -> None:
        super().__init__(
            num_pkts=DEFAULT_QUBIT_SPINOR_COUNT,
            d_pkt=DEFAULT_QUBIT_SPHERE_DIM,
            num_heads=DEFAULT_QUBIT_HEADS,
            num_layers=num_layers,
        )
        self.num_qubits = DEFAULT_QUBIT_SPINOR_COUNT
        self.qubits_per_head = self.num_qubits // self.num_heads
        self.d_h = self.d_pkt
        self.v_rank = self.qubits_per_head // 2
        self.state_io_mode = QuantumStateIOMode.WF_IN_WF_OUT
        self.channel_mode = QuantumChannelMode.UNITARY_U16
        if not include_mach_zehnder:
            self.disable_mach_zehnder_()

        angles = torch.linspace(-0.25, 0.25, self.num_qubits, dtype=torch.float32)
        u_re = torch.zeros(self.num_qubits, 2, 2, dtype=torch.float32)
        u_im = torch.zeros(self.num_qubits, 2, 2, dtype=torch.float32)
        u_re[:, 0, 0] = u_re[:, 1, 1] = torch.cos(angles)
        u_im[:, 0, 0], u_im[:, 1, 1] = torch.sin(angles), -torch.sin(angles)
        self.head_unitary_re, self.head_unitary_im = nn.Parameter(u_re), nn.Parameter(u_im)

        qk_complex_dim = self.qubits_per_head * 2
        inv_freq = 1.0 / (
            DEFAULT_ROPE_BASE_THETA ** (torch.arange(0, qk_complex_dim, dtype=torch.float32) / float(qk_complex_dim))
        )
        self.register_buffer("rope_inv_freq", inv_freq.view(1, 1, 1, self.qubits_per_head, 2), persistent=False)

    def _apply_qubit_rope(self, z_btmd: torch.Tensor) -> torch.Tensor:
        """Apply complex rotary phase to `(B, T, 256, 4)` viewed as `(B, T, 8, 32, 2, 2)` `(c_idx, Re/Im)`."""
        bsz, seq_len, _, _ = z_btmd.shape
        z_c = z_btmd.view(bsz, seq_len, self.num_heads, self.qubits_per_head, 2, 2)
        z_re, z_im = z_c[..., 0], z_c[..., 1]
        pos = torch.arange(seq_len, device=z_btmd.device, dtype=torch.float32).view(1, seq_len, 1, 1, 1)
        angles = pos * self.rope_inv_freq
        cos_a, sin_a = torch.cos(angles).to(z_btmd.dtype), torch.sin(angles).to(z_btmd.dtype)
        return torch.stack([z_re * cos_a - z_im * sin_a, z_re * sin_a + z_im * cos_a], dim=-1).reshape(
            bsz, seq_len, self.num_qubits, self.d_pkt
        )

    def _run_attention_and_evolution(
        self, h_norm: torch.Tensor, q_rot: torch.Tensor, q_h: torch.Tensor, k_h: torch.Tensor, v_wp: torch.Tensor
    ) -> torch.Tensor:
        """Execute causal Bloch outer-product attention, analytical 2x2 McWeeny purification, and U(2) evolution."""
        bsz, seq_len, _, _ = v_wp.shape
        work_dtype = v_wp.dtype
        sdpa_dim = self.qubits_per_head * self.d_pkt
        vn = v_wp * torch.rsqrt(v_wp.square().sum(dim=-1, keepdim=True).clamp_min(TRACE_NORM_EPS))
        ar, ai, br, bi = vn[..., 0], vn[..., 1], vn[..., 2], vn[..., 3]
        bx, by, bz = 2.0 * (ar * br + ai * bi), 2.0 * (ar * bi - ai * br), (ar * ar + ai * ai) - (br * br + bi * bi)
        bloch_4 = torch.stack([bx, by, bz, torch.ones_like(bz)], dim=-1).view(bsz, seq_len, self.num_heads, 1, sdpa_dim)
        ket_4 = vn.view(bsz, seq_len, self.num_heads, 1, sdpa_dim)
        sdpa_v = torch.cat([bloch_4, ket_4], dim=3).reshape(bsz, seq_len, self.num_heads * 2, sdpa_dim).permute(0, 2, 1, 3).contiguous()
        sdpa_q, sdpa_k = self._expand_sdpa_qk(q_h, k_h, chunks=2)
        a_all = F.scaled_dot_product_attention(
            sdpa_q, sdpa_k, sdpa_v, is_causal=True, scale=DEFAULT_WAVEPACKET_SDPA_SCALE
        ).view(bsz, self.num_heads, 2, seq_len, self.qubits_per_head, 4)

        b_att = a_all[:, :, 0].permute(0, 2, 1, 3, 4).reshape(bsz, seq_len, self.num_qubits, 4).float()
        v_att = a_all[:, :, 1].permute(0, 2, 1, 3, 4).reshape(bsz, seq_len, self.num_qubits, 4)
        v_bar = v_att * torch.rsqrt(v_att.square().sum(dim=-1, keepdim=True).clamp_min(TRACE_NORM_EPS))

        r_vec = b_att[..., :3] / b_att[..., 3:4].clamp_min(TRACE_NORM_EPS)
        r_sq = r_vec.square().sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        purity = 0.5 * (1.0 + r_sq.squeeze(-1))
        eff_rank = 1.0 / purity.clamp_min(TRACE_NORM_EPS)
        r_pure = (2.0 * r_vec / (1.0 + r_sq)).to(work_dtype)
        rx, ry, rz = r_pure[..., 0], r_pure[..., 1], r_pure[..., 2]
        half = torch.tensor(0.5, device=v_wp.device, dtype=work_dtype)
        one = torch.tensor(1.0, device=v_wp.device, dtype=work_dtype)
        zero = torch.zeros_like(rx)

        rho_pure_re = half * torch.stack(
            [torch.stack([one + rz, rx], dim=-1), torch.stack([rx, one - rz], dim=-1)], dim=-2
        )
        rho_pure_im = half * torch.stack([torch.stack([zero, -ry], dim=-1), torch.stack([ry, zero], dim=-1)], dim=-2)

        ur = self.head_unitary_re.to(work_dtype).view(1, 1, self.num_qubits, 2, 2)
        ui = self.head_unitary_im.to(work_dtype).view(1, 1, self.num_qubits, 2, 2)
        urt, uit = ur.transpose(-2, -1), -ui.transpose(-2, -1)
        tmp_re = torch.matmul(ur, rho_pure_re) - torch.matmul(ui, rho_pure_im)
        tmp_im = torch.matmul(ur, rho_pure_im) + torch.matmul(ui, rho_pure_re)
        evol_re = torch.matmul(tmp_re, urt) - torch.matmul(tmp_im, uit)
        evol_im = torch.matmul(tmp_re, uit) + torch.matmul(tmp_im, urt)

        v_c = v_bar.view(bsz, seq_len, self.num_qubits, 2, 2)
        vr, vi = v_c[..., 0].unsqueeze(-1), v_c[..., 1].unsqueeze(-1)
        vur, vui = torch.matmul(ur, vr) - torch.matmul(ui, vi), torch.matmul(ur, vi) + torch.matmul(ui, vr)
        out_r = vur + 0.5 * (torch.matmul(evol_re, vur) - torch.matmul(evol_im, vui))
        out_i = vui + 0.5 * (torch.matmul(evol_re, vui) + torch.matmul(evol_im, vur))
        y_wp = F.rms_norm(
            torch.stack([out_r.squeeze(-1), out_i.squeeze(-1)], dim=-1).reshape(bsz, seq_len, self.num_qubits, 4),
            (4,),
            eps=TRACE_NORM_EPS,
        )

        with torch.no_grad():
            r0_x, r0_y, r0_z = r_vec[..., 0], r_vec[..., 1], r_vec[..., 2]
            z0 = torch.zeros_like(r0_x)
            rho0_re = 0.5 * torch.stack(
                [torch.stack([1.0 + r0_z, r0_x], dim=-1), torch.stack([r0_x, 1.0 - r0_z], dim=-1)], dim=-2
            )
            rho0_im = 0.5 * torch.stack([torch.stack([z0, -r0_y], dim=-1), torch.stack([r0_y, z0], dim=-1)], dim=-2)
            self._last_rho0_re_im = (rho0_re.detach(), rho0_im.detach())
            lam1 = (0.5 * (1.0 + r_sq.squeeze(-1).sqrt())).clamp(0.0, 1.0)
            self._last_telemetry_tensors = (
                purity.detach().mean(),
                eff_rank.detach().mean(),
                lam1.detach().mean(),
                (1.0 - lam1).clamp(0.0, 1.0).detach().mean(),
            )
        return y_wp

    def forward(
        self, x: torch.Tensor, x_0: torch.Tensor | None = None,
        ve_wp: torch.Tensor | None = None, v_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
        """Return `(x_out_or_mid, attn_delta_scaled, v_wp)` on `(B, T, 256, 4)`."""
        bsz, seq_len, _, _ = x.shape
        work_dtype = x.dtype
        l0, l1 = self.block_lambdas[0].to(work_dtype), self.block_lambdas[1].to(work_dtype)
        x_cur = l0 * x + l1 * x_0.to(work_dtype) if x_0 is not None else x
        h_norm = F.rms_norm(x_cur, (self.d_pkt,), eps=TRACE_NORM_EPS)
        q_wp = apply_factored_wavepacket_stiefel(h_norm, self.w_q_pkt.to(work_dtype), self.m_q_mode.to(work_dtype))
        k_wp = apply_factored_wavepacket_stiefel(h_norm, self.w_k_pkt.to(work_dtype), self.m_k_mode.to(work_dtype))
        v_wp = apply_factored_wavepacket_stiefel(h_norm, self.w_v_pkt.to(work_dtype), self.m_v_mode.to(work_dtype))
        if ve_wp is not None and v_scale is not None:
            v_wp = v_scale.to(work_dtype) * v_wp + ve_wp.to(work_dtype)

        q_rot, k_rot = self._apply_qubit_rope(q_wp), self._apply_qubit_rope(k_wp)
        sdpa_dim = self.qubits_per_head * self.d_pkt
        q_h = F.rms_norm(q_rot.view(bsz, seq_len, self.num_heads, sdpa_dim), (sdpa_dim,), eps=TRACE_NORM_EPS)
        k_h = F.rms_norm(k_rot.view(bsz, seq_len, self.num_heads, sdpa_dim), (sdpa_dim,), eps=TRACE_NORM_EPS)
        if seq_len > 1:
            k_h = apply_partial_key_offset(k_h)
        y_wp = self._run_attention_and_evolution(h_norm, q_rot, q_h, k_h, v_wp)
        return self._finish_readout_and_mach_zehnder(x_cur, y_wp, v_wp)


class ConfigurableWavepacketStatMechBlock(_BaseQuantumGraftLayer0):
    r"""Configurable `(B, T, M_pkt, 16)` Layer 0 supporting all State I/O and Interior Attention Modes."""

    def __init__(
        self, num_pkts: int = 48, num_heads: int = 6, d_h: int = DEFAULT_WAVEPACKET_HILBERT_DIM, num_layers: int = 12,
        state_io_mode: QuantumStateIOMode = QuantumStateIOMode.WF_IN_WF_OUT,
        attn_interior_mode: QuantumAttnInteriorMode = QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
        purification_mode: PurificationMode = PurificationMode.QUADRATIC_MCWEENY,
        channel_mode: QuantumChannelMode = QuantumChannelMode.UNITARY_U16,
    ) -> None:  # fmt: skip
        super().__init__(num_pkts=num_pkts, d_pkt=d_h, num_heads=num_heads, num_layers=num_layers)
        self.d_h = int(d_h)
        self.complex_heads = max(1, self.num_pkts // 2)
        self.v_rank = max(1, self.complex_heads // self.num_heads)
        self.state_io_mode = QuantumStateIOMode(state_io_mode)
        self.attn_interior_mode = QuantumAttnInteriorMode(attn_interior_mode)
        self.purification_mode = PurificationMode(purification_mode)
        self.channel_mode = QuantumChannelMode(channel_mode)

        if self.channel_mode == QuantumChannelMode.CPTP_KRAUS_RANK_K:
            kr_re, kr_im = initialize_complex_kraus_stiefel_pair(
                self.num_heads, self.d_h, kraus_rank=DEFAULT_KRAUS_CHANNEL_RANK
            )
            self.head_unitary_re, self.head_unitary_im = nn.Parameter(kr_re), nn.Parameter(kr_im)
        else:
            phase_angles = torch.linspace(-0.25, 0.25, self.d_h, dtype=torch.float32)
            diag_re = torch.diag(torch.cos(phase_angles)).unsqueeze(0).expand(self.num_heads, -1, -1).clone()
            diag_im = torch.diag(torch.sin(phase_angles)).unsqueeze(0).expand(self.num_heads, -1, -1).clone()
            self.head_unitary_re, self.head_unitary_im = nn.Parameter(diag_re), nn.Parameter(diag_im)

        self.diag_obs_gamma = nn.Parameter(
            torch.full((self.num_heads,), DEFAULT_GRAFT_DIAG_OBS_INIT, dtype=torch.float32),
            requires_grad=(self.state_io_mode == QuantumStateIOMode.WF_IN_DENSITY_OBS_OUT),
        )
        self.dm_res_mix = nn.Parameter(
            torch.tensor(DEFAULT_DM_RES_MIX_INIT, dtype=torch.float32),
            requires_grad=(self.state_io_mode == QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO),
        )
        self.dm_lag_mix = nn.Parameter(
            torch.tensor(DEFAULT_DM_LAG_MIX_INIT, dtype=torch.float32),
            requires_grad=(self.state_io_mode == QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO),
        )
        init_purify = torch.tensor([-2.0, 2.0, 0.0], dtype=torch.float32).unsqueeze(0).expand(self.num_heads, -1).clone()
        self.purify_logits = nn.Parameter(
            init_purify, requires_grad=(self.purification_mode == PurificationMode.LEARNABLE_TEMP_MCWEENY)
        )
        self.local_rho_mix_logit = nn.Parameter(
            torch.full((self.num_heads,), -1.10, dtype=torch.float32),
            requires_grad=(self.attn_interior_mode == QuantumAttnInteriorMode.SYMMETRIZED_LOCAL_CONTEXT_RHO_V),
        )
        self.liouville_beta = nn.Parameter(
            torch.full((self.num_heads,), 0.25, dtype=torch.float32),
            requires_grad=(self.attn_interior_mode == QuantumAttnInteriorMode.LIOUVILLE_COMMUTATOR_RHO_V),
        )

        qk_complex_dim = self.v_rank * self.d_h
        inv_freq = 1.0 / (
            DEFAULT_ROPE_BASE_THETA ** (torch.arange(0, qk_complex_dim, dtype=torch.float32) / float(qk_complex_dim))
        )
        self.register_buffer("rope_inv_freq", inv_freq.view(1, 1, 1, self.v_rank, self.d_h), persistent=False)

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return active 1D scalar parameters in this block."""
        params = super().scalar_parameters()
        for p in (
            self.diag_obs_gamma, self.dm_res_mix, self.dm_lag_mix,
            self.purify_logits, self.local_rho_mix_logit, self.liouville_beta,
        ):  # fmt: skip
            if p.requires_grad:
                params.append(p)
        return params

    def _apply_wavepacket_complex_rope(self, z_btmd: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale complex rotary phase to `(B, T, M_pkt, 16)`."""
        bsz, seq_len, _, _ = z_btmd.shape
        z_c = z_btmd.view(bsz, seq_len, self.num_heads, self.v_rank, 2, self.d_h)
        z_re, z_im = z_c[..., 0, :], z_c[..., 1, :]
        pos = torch.arange(seq_len, device=z_btmd.device, dtype=torch.float32).view(1, seq_len, 1, 1, 1)
        angles = pos * self.rope_inv_freq
        cos_a, sin_a = torch.cos(angles).to(z_btmd.dtype), torch.sin(angles).to(z_btmd.dtype)
        return torch.stack([z_re * cos_a - z_im * sin_a, z_re * sin_a + z_im * cos_a], dim=-2).reshape(
            bsz, seq_len, self.num_pkts, self.d_h
        )

    def _normalize_mode_kets(self, z_btmd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(zn_re, zn_im)` of shape `(B, H, T, R, d_h)` normalized so $\\sum_{r=1}^R \\|z_{r}\\|_2^2 = 1$."""
        bsz, seq_len, _, _ = z_btmd.shape
        zc = z_btmd.view(bsz, seq_len, self.num_heads, self.v_rank, 2, self.d_h).permute(0, 2, 1, 3, 4, 5)
        zr, zi = zc[..., 0, :], zc[..., 1, :]
        nsq = (zr.square().sum(dim=-1, keepdim=True) + zi.square().sum(dim=-1, keepdim=True)).clamp_min(TRACE_NORM_EPS)
        inv_n = torch.rsqrt(nsq * float(self.v_rank))
        return zr * inv_n, zi * inv_n

    def _evolve_purified_density(
        self, rho0_re: torch.Tensor, rho0_im: torch.Tensor, work_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Purify `(rho0_re, rho0_im)` `(B, H, T, d_h, d_h)` and evolve via Unitary or CPTP Kraus channel."""
        pure_re, pure_im, purity, eff_rank = purify_density_statmech(
            rho0_re, rho0_im, mode=self.purification_mode, purify_logits_h3=self.purify_logits, eps=TRACE_NORM_EPS
        )
        if self.channel_mode == QuantumChannelMode.CPTP_KRAUS_RANK_K and self.head_unitary_re.size(-2) > self.d_h:
            evol_re, evol_im = evolve_head_cptp_kraus_functional(
                pure_re, pure_im, self.head_unitary_re, self.head_unitary_im
            )
            u0_re = self.head_unitary_re[:, : self.d_h, :].to(work_dtype).view(1, self.num_heads, 1, self.d_h, self.d_h)
            u0_im = self.head_unitary_im[:, : self.d_h, :].to(work_dtype).view(1, self.num_heads, 1, self.d_h, self.d_h)
        else:
            evol_re, evol_im = evolve_head_unitary_traceless_functional(
                pure_re, pure_im, self.head_unitary_re, self.head_unitary_im
            )
            u0_re = self.head_unitary_re.to(work_dtype).view(1, self.num_heads, 1, self.d_h, self.d_h)
            u0_im = self.head_unitary_im.to(work_dtype).view(1, self.num_heads, 1, self.d_h, self.d_h)

        with torch.no_grad():
            diag_evol = torch.diagonal(evol_re, dim1=-2, dim2=-1)
            lam1 = diag_evol.max(dim=-1).values.clamp(0.0, 1.0)
            self._last_telemetry_tensors = (
                purity.detach().mean(),
                eff_rank.detach().mean(),
                lam1.detach().mean(),
                (1.0 - lam1).clamp(0.0, 1.0).detach().mean(),
            )
        return evol_re, evol_im, u0_re, u0_im

    def _run_attention_and_evolution(
        self, h_norm: torch.Tensor, q_rot: torch.Tensor, q_h: torch.Tensor, k_h: torch.Tensor, v_wp: torch.Tensor
    ) -> torch.Tensor:
        """Execute configured interior attention mode and state I/O contract returning `y_wp` `(B, T, M_pkt, d_h)`."""
        bsz, seq_len, _, _ = h_norm.shape
        work_dtype = h_norm.dtype
        sdpa_dim = 2 * self.v_rank * self.d_h
        k_rho = max(1, self.d_h // self.v_rank)
        row_chunk = self.v_rank

        if self.attn_interior_mode == QuantumAttnInteriorMode.PRE_ATTN_LOCAL_OUTER_ON_V:
            in_re, in_im = self._normalize_mode_kets(h_norm)
            rho0_re = torch.matmul(in_re.transpose(-2, -1), in_re) + torch.matmul(in_im.transpose(-2, -1), in_im)
            rho0_im = torch.matmul(in_im.transpose(-2, -1), in_re) - torch.matmul(in_re.transpose(-2, -1), in_im)
            evol_re, evol_im, u0_re, u0_im = self._evolve_purified_density(rho0_re, rho0_im, work_dtype)
            v_re, v_im = self._normalize_mode_kets(v_wp)
            vu_re = torch.matmul(v_re, u0_re.transpose(-2, -1)) - torch.matmul(v_im, u0_im.transpose(-2, -1))
            vu_im = torch.matmul(v_re, u0_im.transpose(-2, -1)) + torch.matmul(v_im, u0_re.transpose(-2, -1))
            mod_re = vu_re + 0.5 * (torch.matmul(vu_re, evol_re) + torch.matmul(vu_im, evol_im))
            mod_im = vu_im + 0.5 * (torch.matmul(vu_im, evol_re) - torch.matmul(vu_re, evol_im))
            sdpa_v1 = torch.stack([mod_re, mod_im], dim=-2).reshape(bsz, self.num_heads, seq_len, sdpa_dim).contiguous()
            a_v1 = F.scaled_dot_product_attention(
                q_h.permute(0, 2, 1, 3), k_h.permute(0, 2, 1, 3), sdpa_v1, is_causal=True, scale=DEFAULT_WAVEPACKET_SDPA_SCALE
            )
            return F.rms_norm(
                a_v1.permute(0, 2, 1, 3).reshape(bsz, seq_len, self.num_pkts, self.d_h), (self.d_h,), eps=TRACE_NORM_EPS
            )

        if self.attn_interior_mode == QuantumAttnInteriorMode.KV_CROSS_OUTER_ON_Q:
            v_re, v_im = self._normalize_mode_kets(v_wp)
            in_re, in_im = self._normalize_mode_kets(h_norm)
            om_re = torch.matmul(v_re.transpose(-2, -1), in_re) + torch.matmul(v_im.transpose(-2, -1), in_im)
            om_im = torch.matmul(v_im.transpose(-2, -1), in_re) - torch.matmul(v_re.transpose(-2, -1), in_im)
            sdpa_om = (
                torch.stack([om_re.view(bsz, self.num_heads, seq_len, k_rho, row_chunk, self.d_h), om_im.view(bsz, self.num_heads, seq_len, k_rho, row_chunk, self.d_h)], dim=4)
                .permute(0, 1, 3, 2, 4, 5, 6).reshape(bsz, self.num_heads * k_rho, seq_len, sdpa_dim).contiguous()
            )
            sdpa_q4, sdpa_k4 = self._expand_sdpa_qk(q_h, k_h, chunks=k_rho)
            a_om = F.scaled_dot_product_attention(sdpa_q4, sdpa_k4, sdpa_om, is_causal=True, scale=DEFAULT_WAVEPACKET_SDPA_SCALE).view(bsz, self.num_heads, k_rho, seq_len, 2, row_chunk, self.d_h).permute(0, 1, 3, 2, 4, 5, 6)
            bar_om_re = a_om[:, :, :, :, 0].reshape(bsz, self.num_heads, seq_len, self.d_h, self.d_h)
            bar_om_im = a_om[:, :, :, :, 1].reshape(bsz, self.num_heads, seq_len, self.d_h, self.d_h)
            gram_re = torch.matmul(bar_om_re, bar_om_re.transpose(-2, -1)) + torch.matmul(bar_om_im, bar_om_im.transpose(-2, -1))
            gram_im = torch.matmul(bar_om_im, bar_om_re.transpose(-2, -1)) - torch.matmul(bar_om_re, bar_om_im.transpose(-2, -1))
            gram_re, gram_im = 0.5 * (gram_re + gram_re.transpose(-2, -1)), 0.5 * (gram_im - gram_im.transpose(-2, -1))
            tr_g = torch.diagonal(gram_re, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(TRACE_NORM_EPS)
            evol_re, evol_im, u0_re, u0_im = self._evolve_purified_density(gram_re / tr_g, gram_im / tr_g, work_dtype)
            q_re, q_im = self._normalize_mode_kets(q_rot)
            qv_re = torch.matmul(q_re, bar_om_re.transpose(-2, -1)) - torch.matmul(q_im, bar_om_im.transpose(-2, -1))
            qv_im = torch.matmul(q_re, bar_om_im.transpose(-2, -1)) + torch.matmul(q_im, bar_om_re.transpose(-2, -1))
            qv_nsq = (qv_re.square().sum(dim=-1, keepdim=True) + qv_im.square().sum(dim=-1, keepdim=True)).clamp_min(TRACE_NORM_EPS)
            inv_qv = torch.rsqrt(qv_nsq * float(self.v_rank))
            p_re, p_im = qv_re * inv_qv, qv_im * inv_qv
            vu_re = torch.matmul(p_re, u0_re.transpose(-2, -1)) - torch.matmul(p_im, u0_im.transpose(-2, -1))
            vu_im = torch.matmul(p_re, u0_im.transpose(-2, -1)) + torch.matmul(p_im, u0_re.transpose(-2, -1))
            wp_u_re = vu_re + 0.5 * (torch.matmul(vu_re, evol_re) + torch.matmul(vu_im, evol_im))
            wp_u_im = vu_im + 0.5 * (torch.matmul(vu_im, evol_re) - torch.matmul(vu_re, evol_im))
            return F.rms_norm(torch.stack([wp_u_re, wp_u_im], dim=-2).permute(0, 2, 1, 3, 4, 5).reshape(bsz, seq_len, self.num_pkts, self.d_h), (self.d_h,), eps=TRACE_NORM_EPS)

        if self.state_io_mode == QuantumStateIOMode.DM_IN_DM_OUT:
            sdpa_v_full = form_wavepacket_outer_product_and_v(v_wp, num_heads=self.num_heads, v_rank=self.v_rank).view(bsz, seq_len, self.num_heads, k_rho + 1, sdpa_dim)
            sdpa_rho_only = sdpa_v_full[:, :, :, :k_rho].reshape(bsz, seq_len, self.num_heads * k_rho, sdpa_dim).permute(0, 2, 1, 3).contiguous()
            sdpa_q4, sdpa_k4 = self._expand_sdpa_qk(q_h, k_h, chunks=k_rho)
            a_rho = F.scaled_dot_product_attention(sdpa_q4, sdpa_k4, sdpa_rho_only, is_causal=True, scale=DEFAULT_WAVEPACKET_SDPA_SCALE).view(bsz, self.num_heads, k_rho, seq_len, 2, row_chunk, self.d_h).permute(0, 1, 3, 2, 4, 5, 6)
            rho0_re = a_rho[:, :, :, :, 0].reshape(bsz, self.num_heads, seq_len, self.d_h, self.d_h)
            rho0_im = a_rho[:, :, :, :, 1].reshape(bsz, self.num_heads, seq_len, self.d_h, self.d_h)
            evol_re, evol_im, u0_re, u0_im = self._evolve_purified_density(rho0_re, rho0_im, work_dtype)
            probe_re = u0_re[:, :, :, :, : self.v_rank].transpose(-2, -1)
            probe_im = u0_im[:, :, :, :, : self.v_rank].transpose(-2, -1)
            wp_u_re = probe_re + 0.5 * (torch.matmul(probe_re, evol_re) + torch.matmul(probe_im, evol_im))
            wp_u_im = probe_im + 0.5 * (torch.matmul(probe_im, evol_re) - torch.matmul(probe_re, evol_im))
            return F.rms_norm(torch.stack([wp_u_re, wp_u_im], dim=-2).permute(0, 2, 1, 3, 4, 5).reshape(bsz, seq_len, self.num_pkts, self.d_h), (self.d_h,), eps=TRACE_NORM_EPS)

        chunks = k_rho + 1
        sdpa_v = build_configurable_rho_and_ket_sdpa_chunks(
            v_wp, h_norm, self.num_heads, self.v_rank, self.dm_res_mix, self.dm_lag_mix,
            use_causal_ensemble=(self.state_io_mode == QuantumStateIOMode.CAUSAL_ENSEMBLE_DM_IO),
            use_h_norm_for_rho=(self.attn_interior_mode == QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V),
        )
        sdpa_q, sdpa_k = self._expand_sdpa_qk(q_h, k_h, chunks=chunks)
        a_all = F.scaled_dot_product_attention(sdpa_q, sdpa_k, sdpa_v, is_causal=True, scale=DEFAULT_WAVEPACKET_SDPA_SCALE).view(bsz, self.num_heads, chunks, seq_len, sdpa_dim)
        a_rho = a_all[:, :, :k_rho].permute(0, 1, 3, 2, 4).reshape(bsz, self.num_heads, seq_len, k_rho, 2, row_chunk, self.d_h)
        rho0_re = a_rho[:, :, :, :, 0].reshape(bsz, self.num_heads, seq_len, self.d_h, self.d_h)
        rho0_im = a_rho[:, :, :, :, 1].reshape(bsz, self.num_heads, seq_len, self.d_h, self.d_h)
        a_v = a_all[:, :, k_rho].view(bsz, self.num_heads, seq_len, self.v_rank, 2, self.d_h)
        v_bar_re, v_bar_im = a_v[..., 0, :], a_v[..., 1, :]
        v_norm_sq = (v_bar_re.square().sum(dim=-1, keepdim=True) + v_bar_im.square().sum(dim=-1, keepdim=True)).clamp_min(TRACE_NORM_EPS)
        inv_vn = torch.rsqrt(v_norm_sq * float(self.v_rank))
        p_re, p_im = v_bar_re * inv_vn, v_bar_im * inv_vn

        evol_re, evol_im, u0_re, u0_im = self._evolve_purified_density(rho0_re, rho0_im, work_dtype)
        vu_re = torch.matmul(p_re, u0_re.transpose(-2, -1)) - torch.matmul(p_im, u0_im.transpose(-2, -1))
        vu_im = torch.matmul(p_re, u0_im.transpose(-2, -1)) + torch.matmul(p_im, u0_re.transpose(-2, -1))
        loc_re, loc_im = self._normalize_mode_kets(v_wp)
        lu_re = torch.matmul(loc_re, u0_re.transpose(-2, -1)) - torch.matmul(loc_im, u0_im.transpose(-2, -1))
        lu_im = torch.matmul(loc_re, u0_im.transpose(-2, -1)) + torch.matmul(loc_im, u0_re.transpose(-2, -1))
        head_p = (
            self.local_rho_mix_logit
            if self.attn_interior_mode == QuantumAttnInteriorMode.SYMMETRIZED_LOCAL_CONTEXT_RHO_V
            else (self.liouville_beta if self.attn_interior_mode == QuantumAttnInteriorMode.LIOUVILLE_COMMUTATOR_RHO_V else None)
        )
        wp_u_re, wp_u_im = apply_interior_rho_v_coupling(
            vu_re, vu_im, lu_re, lu_im, evol_re, evol_im, self.attn_interior_mode.value, head_p
        )

        y_wp = F.rms_norm(
            torch.stack([wp_u_re, wp_u_im], dim=-2).permute(0, 2, 1, 3, 4, 5).reshape(bsz, seq_len, self.num_pkts, self.d_h),
            (self.d_h,), eps=TRACE_NORM_EPS,
        )  # fmt: skip
        if self.state_io_mode == QuantumStateIOMode.WF_IN_DENSITY_OBS_OUT:
            diag_p = torch.diagonal(evol_re, dim1=-2, dim2=-1).permute(0, 2, 1, 3).unsqueeze(3)
            g_diag = torch.tanh(self.diag_obs_gamma).to(work_dtype).view(1, 1, self.num_heads, 1, 1)
            mod_factor = 1.0 + g_diag * (float(self.d_h) * diag_p - 1.0)
            y_wp = F.rms_norm(
                (y_wp.view(bsz, seq_len, self.num_heads, 2 * self.v_rank, self.d_h) * mod_factor).reshape(bsz, seq_len, self.num_pkts, self.d_h),
                (self.d_h,), eps=TRACE_NORM_EPS,
            )  # fmt: skip
        return y_wp

    def forward(
        self, x: torch.Tensor, x_0: torch.Tensor | None = None,
        ve_wp: torch.Tensor | None = None, v_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
        """Return `(x_out_or_mid, attn_delta_scaled, v_wp)` on `(B, T, M_pkt, 16)`."""
        bsz, seq_len, _, _ = x.shape
        work_dtype = x.dtype
        l0, l1 = self.block_lambdas[0].to(work_dtype), self.block_lambdas[1].to(work_dtype)
        x_cur = l0 * x + l1 * x_0.to(work_dtype) if x_0 is not None else x
        h_norm = F.rms_norm(x_cur, (self.d_h,), eps=TRACE_NORM_EPS)
        q_wp = apply_factored_wavepacket_stiefel(h_norm, self.w_q_pkt.to(work_dtype), self.m_q_mode.to(work_dtype))
        k_wp = apply_factored_wavepacket_stiefel(h_norm, self.w_k_pkt.to(work_dtype), self.m_k_mode.to(work_dtype))
        v_wp = apply_factored_wavepacket_stiefel(h_norm, self.w_v_pkt.to(work_dtype), self.m_v_mode.to(work_dtype))
        if ve_wp is not None and v_scale is not None:
            v_wp = v_scale.to(work_dtype) * v_wp + ve_wp.to(work_dtype)

        q_rot, k_rot = self._apply_wavepacket_complex_rope(q_wp), self._apply_wavepacket_complex_rope(k_wp)
        sdpa_dim = 2 * self.v_rank * self.d_h
        q_h = F.rms_norm(q_rot.view(bsz, seq_len, self.num_heads, sdpa_dim), (sdpa_dim,), eps=TRACE_NORM_EPS)
        k_h = F.rms_norm(k_rot.view(bsz, seq_len, self.num_heads, sdpa_dim), (sdpa_dim,), eps=TRACE_NORM_EPS)
        if seq_len > 1:
            k_h = apply_partial_key_offset(k_h)

        y_wp = self._run_attention_and_evolution(h_norm, q_rot, q_h, k_h, v_wp)
        return self._finish_readout_and_mach_zehnder(x_cur, y_wp, v_wp)


class FrozenTrunkQuantumFrontEndGPT(nn.Module):
    """Frozen Pretrained `CompiledLeaderboardClassicalGPT` with Surgical Quantum Front-End Replacement."""

    def __init__(
        self,
        classical_model: Any,
        graft_cfg: QuantumFrontEndGraftConfig | None = None,
        ppmi_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.classical = classical_model
        self.cfg = classical_model.cfg
        self.graft_cfg = graft_cfg if graft_cfg is not None else QuantumFrontEndGraftConfig()
        self.vocab_size = int(self.classical.vocab_size)
        self.embed_dim = int(self.classical.embed_dim)
        self.num_heads = int(self.classical.num_heads)
        self.num_layers = int(self.classical.num_layers)
        self.lm_head = self.classical.lm_head
        self.training_step: int = 0

        for p in self.classical.parameters():
            p.requires_grad_(False)

        spec = resolve_wavefunction_shape_spec(self.graft_cfg.wf_shape)
        self.wte_slices, self.wte_sphere_dim, self.d_emb, self.pkt_count, self.pkt_dim, self.q_heads = spec

        self.q_wte: nn.Parameter | None = None
        self.w_in_pkt: nn.Parameter | None = None
        self.w_in_mode: nn.Parameter | None = None
        self.w_in_bridge: nn.Parameter | None = None
        self.w_out_bridge: nn.Parameter | None = None
        self.in_scale: nn.Parameter | None = None
        self.wp_causal_re: nn.Parameter | None = None
        self.wp_causal_im: nn.Parameter | None = None
        self.ve0_lambdas: nn.Parameter | None = None
        self.quantum_block0: (
            QubitSpinorBlochStatMechBlock | ConfigurableWavepacketStatMechBlock | SVDStiefelWavepacketStatMechBlock | None
        ) = None
        self.quantum_block1: SVDStiefelWavepacketStatMechBlock | None = None

        if self.graft_cfg.replacement_scope != QuantumReplacementScope.TEACHER_PASS_THROUGH:
            is_svd_stiefel = self.graft_cfg.stiefel_proj_mode != QuantumStiefelProjMode.KRONEKER_FACTORED_STIEFEL
            if self.d_emb != self.embed_dim:
                rows, cols = max(self.d_emb, self.embed_dim), min(self.d_emb, self.embed_dim)
                w_in_b = torch.empty(rows, cols, dtype=torch.float32)
                w_out_b = torch.empty(rows, cols, dtype=torch.float32)
                initialize_stiefel_parameter(w_in_b)
                initialize_stiefel_parameter(w_out_b)
                self.w_in_bridge = nn.Parameter(w_in_b)
                self.w_out_bridge = nn.Parameter(w_out_b, requires_grad=(not is_svd_stiefel))

            q_wte = torch.randn(self.vocab_size, self.wte_slices, self.wte_sphere_dim, dtype=torch.float32)
            if self.graft_cfg.wte_init_mode == QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE:
                initialize_teacher_procrustes_sphere_embeddings_(
                    q_wte, self.classical.wte.weight, self.w_in_bridge, self.w_out_bridge
                )
            else:
                initialize_arbitrary_shape_ppmi_sphere_embeddings_(q_wte, ppmi_embeddings=ppmi_embeddings)
            self.q_wte = nn.Parameter(q_wte)
            self.in_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

            wp_complex_modes = max(1, self.pkt_count // 2)
            n_groups = max(1, (wp_complex_modes + 3) // 4)
            init_wp_re = torch.tensor([0.15, 0.10, 0.08, 0.05], dtype=torch.float32).repeat(n_groups)[:wp_complex_modes]
            init_wp_im = torch.tensor([0.00, 0.10, -0.08, 0.05], dtype=torch.float32).repeat(n_groups)[:wp_complex_modes]
            if self.graft_cfg.wte_init_mode == QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE:
                init_wp_re.zero_()
                init_wp_im.zero_()
            self.wp_causal_re, self.wp_causal_im = nn.Parameter(init_wp_re), nn.Parameter(init_wp_im)

            eye_pkt = torch.eye(self.pkt_dim, dtype=torch.float32).unsqueeze(0).expand(self.pkt_count, -1, -1).clone()
            eye_mode = torch.eye(self.pkt_count, dtype=torch.float32).unsqueeze(0).expand(self.pkt_dim, -1, -1).clone()
            self.w_in_pkt = nn.Parameter(eye_pkt, requires_grad=(not is_svd_stiefel))
            self.w_in_mode = nn.Parameter(eye_mode, requires_grad=(not is_svd_stiefel))

            has_ve = self.classical.value_embeds is not None and len(self.classical.value_embeds) > 0
            if not is_svd_stiefel and self.graft_cfg.ve_coupling_mode == QuantumValueEmbedCouplingMode.FROZEN_TEACHER_VE0_COUPLED and has_ve:
                self.ve0_lambdas = nn.Parameter(torch.tensor([1.0, DEFAULT_VE0_LAMBDA_INIT], dtype=torch.float32))

            inc_mz = self.graft_cfg.replacement_scope == QuantumReplacementScope.FULL_BLOCK0_QUANTUM
            if self.graft_cfg.wf_shape == QuantumWavefunctionShape.SHAPE_256x2x2:
                blk0: QubitSpinorBlochStatMechBlock | ConfigurableWavepacketStatMechBlock = QubitSpinorBlochStatMechBlock(self.num_layers, include_mach_zehnder=inc_mz)
            else:
                blk0 = ConfigurableWavepacketStatMechBlock(self.pkt_count, self.q_heads, self.pkt_dim, self.num_layers, self.graft_cfg.state_io_mode, self.graft_cfg.attn_interior_mode, self.graft_cfg.purification_mode, self.graft_cfg.channel_mode)
                if not inc_mz:
                    blk0.disable_mach_zehnder_()
            if is_svd_stiefel:
                svd_blk0 = SVDStiefelWavepacketStatMechBlock(blk0, self.embed_dim, self.d_emb, self.graft_cfg.stiefel_proj_mode, self.graft_cfg.self_ortho_mode, gate_in_ch=int(self.classical.blocks[0].gate_in_ch), qk_phase_mode=self.graft_cfg.qk_phase_mode)
                if self.graft_cfg.wte_init_mode == QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE:
                    svd_blk0.calibrate_from_teacher_block0_(self.classical.blocks[0], self.w_in_bridge)
                self.quantum_block0 = svd_blk0
                if self.graft_cfg.replacement_scope == QuantumReplacementScope.EMB_PLUS_L0_L1 and self.num_layers > 1:
                    core1 = ConfigurableWavepacketStatMechBlock(self.pkt_count, self.q_heads, self.pkt_dim, self.num_layers, self.graft_cfg.state_io_mode, self.graft_cfg.attn_interior_mode, self.graft_cfg.purification_mode, self.graft_cfg.channel_mode)
                    core1.disable_mach_zehnder_()
                    svd_blk1 = SVDStiefelWavepacketStatMechBlock(core1, self.embed_dim, self.d_emb, self.graft_cfg.stiefel_proj_mode, self.graft_cfg.self_ortho_mode, gate_in_ch=int(self.classical.blocks[1].gate_in_ch), qk_phase_mode=self.graft_cfg.qk_phase_mode)
                    if self.graft_cfg.wte_init_mode == QuantumWteInitMode.TEACHER_PROCRUSTES_SPHERE:
                        svd_blk1.calibrate_from_teacher_block0_(self.classical.blocks[1], self.w_in_bridge)
                    self.quantum_block1 = svd_blk1
            else:
                self.quantum_block0 = blk0

    @classmethod
    def from_teacher_checkpoint(
        cls, ckpt_path: str | Path = DEFAULT_TEACHER_300M_CKPT_PATH,
        graft_cfg: QuantumFrontEndGraftConfig | None = None, device: str | torch.device = "cpu",
    ) -> FrozenTrunkQuantumFrontEndGPT:  # fmt: skip
        """Load a pretrained `CompiledLeaderboardClassicalGPT` teacher checkpoint and attach the quantum graft."""
        from ..models.compiled_leaderboard_classical_gpt import (
            CompiledLeaderboardClassicalConfig, CompiledLeaderboardClassicalGPT,
        )  # fmt: skip

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

    def quantum_real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Collect all active real Stiefel parameters in the quantum front-end."""
        in_st = (self.w_in_pkt, self.w_in_mode, self.w_in_bridge, self.w_out_bridge)
        params = [p for p in in_st if p is not None and p.requires_grad]
        for q_blk in (self.quantum_block0, self.quantum_block1):
            if q_blk is not None:
                params.extend(q_blk.real_stiefel_parameters())
        return params

    def quantum_complex_unitary_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        """Collect complex head Unitary/Kraus pairs from Quantum Layer 0 (and Layer 1 if active)."""
        return [q_blk.complex_unitary_pair() for q_blk in (self.quantum_block0, self.quantum_block1) if q_blk is not None]

    def quantum_scalar_parameters(self) -> list[nn.Parameter]:
        """Collect active 1D scalar parameters from the quantum front-end."""
        in_sc = (self.in_scale, self.wp_causal_re, self.wp_causal_im, self.ve0_lambdas)
        params = [p for p in in_sc if p is not None and p.requires_grad]
        for q_blk in (self.quantum_block0, self.quantum_block1):
            if q_blk is not None:
                params.extend(q_blk.scalar_parameters())
        return params

    def active_quantum_parameters(self) -> list[nn.Parameter]:
        """Return all trainable quantum parameters (`q_wte` + Stiefel + Unitary + Scalars)."""
        params: list[nn.Parameter] = []
        if self.q_wte is not None and self.q_wte.requires_grad:
            params.append(self.q_wte)
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
        """Clamp quantum front-end scalar parameters to their bounded physical intervals."""
        if self.in_scale is not None:
            self.in_scale.clamp_(0.50, 2.00)
        if self.wp_causal_re is not None:
            self.wp_causal_re.clamp_(-1.50, 1.50)
        if self.wp_causal_im is not None:
            self.wp_causal_im.clamp_(-1.50, 1.50)
        if self.ve0_lambdas is not None:
            self.ve0_lambdas[0].clamp_(0.10, 2.00)
            self.ve0_lambdas[1].clamp_(-1.00, 1.00)
        for q_blk in (self.quantum_block0, self.quantum_block1):
            if isinstance(q_blk, SVDStiefelWavepacketStatMechBlock):
                q_blk.restrict_scalars_()
            elif q_blk is not None:
                q_blk.obs_scale.clamp_(0.02, 0.60)
                if q_blk.chan_scale.requires_grad:
                    q_blk.chan_scale.clamp_(0.02, 0.60)
                q_blk.block_lambdas[0].clamp_(0.50, 1.50)
                q_blk.block_lambdas[1].clamp_(-0.50, 0.50)
                if isinstance(q_blk, ConfigurableWavepacketStatMechBlock):
                    q_blk.dm_res_mix.clamp_(0.01, 1.00)
                    q_blk.dm_lag_mix.clamp_(0.01, 1.00)
                    q_blk.local_rho_mix_logit.clamp_(-4.0, 4.0)
                    q_blk.liouville_beta.clamp_(-2.0, 2.0)

    @torch.no_grad()
    def compute_teacher_layer0_targets(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute detached classical teacher Layer-0 states `(x_0_c, x_1_c, v_0_c)` for interface warmup."""
        _, seq_len = idx.shape
        raw_emb = self.classical.wte(idx)
        if self.classical.smear_gate is not None and self.classical.smear_lambda is not None and seq_len > 1:
            s_gate = self.classical.smear_lambda.to(raw_emb.dtype) * torch.sigmoid(
                self.classical.smear_gate(raw_emb[:, 1:, : self.classical.smear_in_ch])
            )
            raw_emb = torch.cat([raw_emb[:, :1], raw_emb[:, 1:] + s_gate * raw_emb[:, :-1]], dim=1)
        x_0_c = F.rms_norm(raw_emb, (self.embed_dim,), eps=self.cfg.eps)
        cos_4d, sin_4d = self.classical.rope.get_cos_sin(seq_len, dtype=x_0_c.dtype)
        q_scale = self.classical.rope.yarn_q_scale
        bm_long, bm_short = self.classical._build_flex_block_masks(idx)
        ve_0 = self.classical.value_embeds[0](idx) if self.classical.value_embeds is not None else None
        blk0 = self.classical.blocks[0]
        layer_bm = bm_long if blk0.is_long_window_layer else bm_short
        x_1_c, v_0_c = blk0(x_0_c, x_0_c, cos_4d, sin_4d, v_0=None, ve_l=ve_0, block_mask=layer_bm, q_scale=q_scale)
        assert v_0_c is not None
        return x_0_c.detach(), x_1_c.detach(), v_0_c.detach()

    def encode_quantum_layer0(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute quantum front-end `(x_0, x_1, v_0)` at the Layer-0 boundary."""
        assert self.q_wte is not None and self.in_scale is not None
        assert self.wp_causal_re is not None and self.wp_causal_im is not None
        assert self.w_in_pkt is not None and self.w_in_mode is not None and self.quantum_block0 is not None

        bsz, seq_len = idx.shape
        work_dtype = torch.bfloat16 if (idx.is_cuda and self.classical.lm_head.weight.dtype == torch.bfloat16) else torch.float32
        raw_wp = F.embedding(idx, self.q_wte.view(self.vocab_size, self.d_emb)).view(bsz, seq_len, self.pkt_count, self.pkt_dim).to(work_dtype)
        if seq_len > 1:
            raw_wp = apply_causal_wavepacket_dispersion(raw_wp, self.wp_causal_re, self.wp_causal_im)

        if isinstance(self.quantum_block0, SVDStiefelWavepacketStatMechBlock):
            return encode_svd_stiefel_layer0(self, idx, raw_wp)

        tok_wp = F.rms_norm(raw_wp, (self.pkt_dim,), eps=TRACE_NORM_EPS)
        x_0_wp = apply_factored_wavepacket_stiefel(tok_wp, self.w_in_pkt.to(work_dtype), self.w_in_mode.to(work_dtype)) * self.in_scale.to(work_dtype)
        x_0_proj = apply_boundary_stiefel_bridge(x_0_wp.reshape(bsz, seq_len, self.d_emb), self.w_in_bridge)
        x_0 = F.rms_norm(x_0_proj, (self.embed_dim,), eps=self.cfg.eps) * self.in_scale.to(work_dtype)

        ve_wp: torch.Tensor | None = None
        v_scale: torch.Tensor | None = None
        if self.ve0_lambdas is not None and self.classical.value_embeds is not None:
            with torch.no_grad():
                ve0_768 = self.classical.value_embeds[0](idx).to(work_dtype)
            ve0_pkt = map_teacher_768_to_quantum_packets(ve0_768, self.w_in_bridge, self.pkt_count, self.pkt_dim)
            ve_wp = self.ve0_lambdas[1].to(work_dtype) * F.rms_norm(ve0_pkt, (self.pkt_dim,), eps=TRACE_NORM_EPS)
            v_scale = self.ve0_lambdas[0].to(work_dtype)

        x_out_wp, _, v_wp = self.quantum_block0(x_0_wp, x_0=x_0_wp, ve_wp=ve_wp, v_scale=v_scale)
        v_0_proj = apply_boundary_stiefel_bridge(v_wp.reshape(bsz, seq_len, self.d_emb), self.w_in_bridge)
        v_0 = v_0_proj.view(bsz, seq_len, self.num_heads, self.embed_dim // self.num_heads)
        x_mid_or_out_768 = apply_boundary_stiefel_bridge(x_out_wp.reshape(bsz, seq_len, self.d_emb), self.w_out_bridge)

        if self.graft_cfg.replacement_scope in (QuantumReplacementScope.ATTN0_QUANTUM_KEEP_FROZEN_MLP0, QuantumReplacementScope.EMB_PLUS_L0_L1):
            blk0 = self.classical.blocks[0]
            h_mlp = F.rms_norm(x_mid_or_out_768, (self.embed_dim,), eps=self.cfg.eps)
            x_1 = x_mid_or_out_768 + blk0.fc2(F.relu(blk0.fc1(h_mlp)).square())
        else:
            x_1 = x_mid_or_out_768

        return x_0, x_1, v_0

    def _run_frozen_classical_tail(
        self, idx: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, v_0: torch.Tensor
    ) -> torch.Tensor:
        """Execute frozen classical U-Net layers `1 .. num_layers - 1` (or `quantum_block1` + `2 .. num_layers - 1`)."""
        _, seq_len = idx.shape
        cos_4d, sin_4d = self.classical.rope.get_cos_sin(seq_len, dtype=x_1.dtype)
        q_scale = self.classical.rope.yarn_q_scale
        bm_long, bm_short = self.classical._build_flex_block_masks(idx)
        with torch.no_grad():
            ve_banks = [ve_emb(idx).to(x_1.dtype) for ve_emb in self.classical.value_embeds] if self.classical.value_embeds is not None else []

        n_half = self.num_layers // 2
        encoder_states: list[torch.Tensor] = [x_1]
        x = x_1
        v_0_cur: torch.Tensor | None = v_0
        start_layer = 1
        if self.quantum_block1 is not None and self.num_layers > 1:
            x = encode_svd_stiefel_layer1(self, idx, x_0, x_1, v_0)
            if 1 < n_half:
                encoder_states.append(x)
            start_layer = 2

        for layer_idx in range(start_layer, self.num_layers):
            block = self.classical.blocks[layer_idx]
            if layer_idx >= n_half and encoder_states:
                dec_i = layer_idx - n_half
                skip_g = torch.sigmoid(self.classical.skip_weights[dec_i]).to(x.dtype) if self.cfg.use_sigmoid_skip else self.classical.skip_weights[dec_i].to(x.dtype)
                x = x + skip_g * encoder_states.pop()

            bank_idx = self.classical._ve_bank_index(layer_idx)
            ve_l = ve_banks[bank_idx] if (bank_idx is not None and bank_idx < len(ve_banks)) else None
            layer_bm = bm_long if block.is_long_window_layer else bm_short
            x, v_0_cur = block(x, x_0, cos_4d, sin_4d, v_0=v_0_cur, ve_l=ve_l, block_mask=layer_bm, q_scale=q_scale)
            if layer_idx < n_half:
                encoder_states.append(x)

        return F.rms_norm(x, (self.embed_dim,), eps=self.cfg.eps)

    def forward_features(self, idx: torch.Tensor) -> torch.Tensor:
        """Encode token indices `(B, T)` into final RMSNormed hidden states `(B, T, embed_dim)`."""
        if self.graft_cfg.replacement_scope == QuantumReplacementScope.TEACHER_PASS_THROUGH:
            return self.classical.forward_features(idx)
        x_0, x_1, v_0 = self.encode_quantum_layer0(idx)
        return self._run_frozen_classical_tail(idx, x_0, x_1, v_0)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Forward pass returning `(logits_or_none, total_loss_or_none)`."""
        if self.graft_cfg.replacement_scope == QuantumReplacementScope.TEACHER_PASS_THROUGH:
            return self.classical(idx, targets=targets, return_logits=return_logits)

        from ..models.compiled_leaderboard_classical_gpt import _CompiledChunkedSoftcapLinearCEFn

        x_0, x_1, v_0 = self.encode_quantum_layer0(idx)
        x_final = self._run_frozen_classical_tail(idx, x_0, x_1, v_0)

        align_loss: torch.Tensor | None = None
        t_align = int(self.graft_cfg.interface_warmup_steps)
        sched = self.graft_cfg.recovery_schedule_mode
        is_persistent = sched == QuantumRecoveryScheduleMode.PERSISTENT_COSINE_FEAT_AND_CE
        if self.training and targets is not None and t_align > 0 and (0 <= self.training_step < t_align or is_persistent) and sched != QuantumRecoveryScheduleMode.PURE_CE:
            x_0_c, x_1_c, v_0_c = self.compute_teacher_layer0_targets(idx)
            w_init = float(self.graft_cfg.interface_align_weight)
            w_floor = DEFAULT_PERSISTENT_ALIGN_FLOOR_RATIO * w_init if is_persistent else 0.0
            prog = min(1.0, float(max(0, self.training_step)) / float(t_align))
            lam_t = w_floor + 0.5 * (w_init - w_floor) * (1.0 + math.cos(math.pi * prog))
            l_int = _normalized_cosine_mse(x_0, x_0_c) + _normalized_cosine_mse(x_1, x_1_c)
            if self.quantum_block1 is not None:
                x_2_q = encode_svd_stiefel_layer1(self, idx, x_0, x_1, v_0)
                x_2_c = compute_teacher_layer1_target(self, idx, x_0_c, x_1_c, v_0_c)
                l_int = l_int + _normalized_cosine_mse(x_2_q, x_2_c)
            if not (isinstance(self.quantum_block0, SVDStiefelWavepacketStatMechBlock) and self.cfg.interior_value_mode == "unit_v"):
                l_int = l_int + DEFAULT_GRAFT_V0_ALIGN_SCALE * _normalized_cosine_mse(v_0, v_0_c)
            if is_persistent or sched == QuantumRecoveryScheduleMode.INTERFACE_AND_FINAL_FEAT_COSINE_TO_CE:
                with torch.no_grad():
                    x_final_c = self.classical.forward_features(idx).detach()
                l_int = l_int + DEFAULT_FINAL_FEAT_ALIGN_SCALE * _normalized_cosine_mse(x_final, x_final_c)
            align_loss = lam_t * l_int

        if targets is not None and not return_logits:
            ce_loss = _CompiledChunkedSoftcapLinearCEFn.apply(
                x_final, self.classical.lm_head.weight, targets, self.cfg.logit_softcap, self.cfg.ce_chunk_tokens, True
            )
            return None, (ce_loss + align_loss if align_loss is not None else ce_loss)

        logits = self.classical.compute_logits(x_final)
        total_loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1).long(), ignore_index=-1)
            total_loss = ce_loss + align_loss if align_loss is not None else ce_loss
        return logits, total_loss

    @torch.no_grad()
    def compute_quantum_graft_telemetry(self) -> dict[str, float]:
        """Compute strict Stiefel orthogonality, `q_wte` sphere RMS error, and Layer-0 density telemetry."""
        if self.graft_cfg.replacement_scope == QuantumReplacementScope.TEACHER_PASS_THROUGH:
            return {}
        st_errs: list[float] = []
        for w in self.quantum_real_stiefel_parameters():
            wf = w.float()
            if wf.size(-2) < wf.size(-1):
                wf = wf.transpose(-2, -1)
            d_cols = wf.size(-1)
            eye = torch.eye(d_cols, device=wf.device, dtype=wf.dtype)
            gram = torch.matmul(wf.transpose(-2, -1), wf)
            st_errs.append(float(((gram - eye).norm(dim=(-2, -1)) / math.sqrt(float(d_cols))).max().item()))
        wte_err = 0.0
        if self.q_wte is not None:
            wte_rms = self.q_wte.float().pow(2).mean(dim=-1).sqrt()
            wte_err = float((wte_rms - 1.0).abs().max().item())
        out: dict[str, float] = {
            "q_max_stiefel_err": max(st_errs) if st_errs else 0.0,
            "q_wte_sphere_err": wte_err,
        }
        if self.quantum_block0 is not None:
            tel = self.quantum_block0.get_last_telemetry_dict()
            out["q_l0_purity"] = tel["mean_purity"]
            out["q_l0_eff_rank"] = tel["mean_eff_rank"]
        return out


class FrozenTrunkQuantumOptimizerBundle:
    """Optimizer bundle stepping ONLY the Quantum Front-End (`q_trunk_opt`, `q_wte_opt`, `q_scalar_opt`)."""

    def __init__(
        self, model: FrozenTrunkQuantumFrontEndGPT, q_trunk_opt: Any = None,
        q_wte_opt: Any = None, q_scalar_opt: torch.optim.Optimizer | None = None,
        base_scalar_lr: float = DEFAULT_GRAFT_SCALAR_LR,
    ) -> None:  # fmt: skip
        self.model = model
        self.q_trunk_opt = q_trunk_opt
        self.q_wte_opt = q_wte_opt
        self.q_scalar_opt = q_scalar_opt
        self.base_scalar_lr = float(base_scalar_lr)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients across all active quantum front-end optimizer groups."""
        for opt in (self.q_trunk_opt, self.q_wte_opt, self.q_scalar_opt):
            if opt is not None:
                opt.zero_grad(set_to_none=set_to_none)

    def step(self, idx: torch.Tensor, lr_mult: float = 1.0) -> None:
        """Step the Stiefel/Unitary optimizer, active-row 3D `q_wte` sphere optimizer, and scalar AdamW."""
        if self.q_trunk_opt is not None:
            self.q_trunk_opt.set_lr_scale(lr_mult)
            self.q_trunk_opt.step()
        if self.q_wte_opt is not None:
            idx_in = torch.unique(idx.reshape(-1))
            self.q_wte_opt.step_active(idx_in, lr_scale=lr_mult)
        if self.q_scalar_opt is not None:
            for grp in self.q_scalar_opt.param_groups:
                grp["lr"] = self.base_scalar_lr * lr_mult
            self.q_scalar_opt.step()
            self.model.restrict_quantum_scalars_()
        self.model.training_step += 1


def build_frozen_trunk_quantum_frontend_optimizers(
    model: FrozenTrunkQuantumFrontEndGPT, stiefel_lr: float | None = None,
    unitary_lr: float | None = None, wte_lr: float | None = None, scalar_lr: float | None = None,
) -> FrozenTrunkQuantumOptimizerBundle:  # fmt: skip
    """Build Riemannian optimizers for the trainable Quantum Front-End (`<= 8` args)."""
    from ..models.modular_quantum_statmech_gpt import ModularRiemannianStiefelKrausMuon
    from ..optim.vocab_manifold_muon import BatchSubspaceTikhonovGramMuon

    is_svd = model.graft_cfg.stiefel_proj_mode != QuantumStiefelProjMode.KRONEKER_FACTORED_STIEFEL
    eff_scalar_lr = float(scalar_lr if scalar_lr is not None else (0.008 if is_svd else DEFAULT_GRAFT_SCALAR_LR))
    if model.graft_cfg.replacement_scope == QuantumReplacementScope.TEACHER_PASS_THROUGH:
        return FrozenTrunkQuantumOptimizerBundle(model, base_scalar_lr=eff_scalar_lr)

    eff_stiefel_lr = float(stiefel_lr if stiefel_lr is not None else model.graft_cfg.stiefel_lr)
    eff_unitary_lr = float(unitary_lr if unitary_lr is not None else (0.015 if is_svd else DEFAULT_GRAFT_UNITARY_LR))
    eff_wte_lr = float(wte_lr if wte_lr is not None else model.graft_cfg.wte_lr)
    q_stiefel_params = model.quantum_real_stiefel_parameters()
    q_trunk_opt = (
        ModularRiemannianStiefelKrausMuon(
            real_stiefel_params=q_stiefel_params, complex_operator_pairs=model.quantum_complex_unitary_pairs(),
            stiefel_lr=eff_stiefel_lr, unitary_lr=eff_unitary_lr, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA,
            momentum_mode=StiefelMomentumMode.RIEMANNIAN_TANGENT_TRANSPORTED,
        )  # fmt: skip
        if q_stiefel_params
        else None
    )
    assert model.q_wte is not None
    q_wte_opt = BatchSubspaceTikhonovGramMuon([model.q_wte], lr=eff_wte_lr, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA)
    q_scalar_opt = torch.optim.AdamW(
        [{"params": model.quantum_scalar_parameters(), "lr": eff_scalar_lr, "weight_decay": 0.0}],
        betas=(0.9, 0.95), eps=1e-8,
    )  # fmt: skip
    return FrozenTrunkQuantumOptimizerBundle(model, q_trunk_opt, q_wte_opt, q_scalar_opt, base_scalar_lr=eff_scalar_lr)


__all__ = [
    "BiorthogonalSVDStiefelProjection", "ConfigurableWavepacketStatMechBlock", "DEFAULT_FINAL_FEAT_ALIGN_SCALE",
    "DEFAULT_GRAFT_DIAG_OBS_INIT", "DEFAULT_GRAFT_INTERFACE_ALIGN_WEIGHT", "DEFAULT_GRAFT_SCALAR_LR",
    "DEFAULT_GRAFT_STIEFEL_LR", "DEFAULT_GRAFT_UNITARY_LR", "DEFAULT_GRAFT_V0_ALIGN_SCALE", "DEFAULT_GRAFT_WTE_LR",
    "DEFAULT_QUBIT_HEADS", "DEFAULT_QUBIT_SPHERE_DIM", "DEFAULT_QUBIT_SPINOR_COUNT",
    "DEFAULT_SVD_STIEFEL_OBS_SCALE_INIT", "DEFAULT_TEACHER_300M_CKPT_PATH", "DEFAULT_TEACHER_50M_CKPT_PATH",
    "DEFAULT_VE0_LAMBDA_INIT", "DEFAULT_WF_LARGE_256_REAL_DIM", "DEFAULT_WF_LARGE_512_REAL_DIM",
    "FrozenTrunkQuantumFrontEndGPT", "FrozenTrunkQuantumOptimizerBundle", "QuantumAttnInteriorMode",
    "QuantumFrontEndGraftConfig", "QuantumQKPhaseMode", "QuantumRecoveryScheduleMode", "QuantumReplacementScope",
    "QuantumSelfOrthoMode", "QuantumStateIOMode", "QuantumStiefelProjMode", "QuantumValueEmbedCouplingMode",
    "QuantumWavefunctionShape", "QuantumWteInitMode", "QubitSpinorBlochStatMechBlock",
    "SVDStiefelWavepacketStatMechBlock", "apply_berry_qk_phase", "apply_boundary_stiefel_bridge",
    "apply_fubini_study_self_ortho", "build_frozen_trunk_quantum_frontend_optimizers",
    "initialize_arbitrary_shape_ppmi_sphere_embeddings_", "initialize_teacher_procrustes_sphere_embeddings_",
    "map_teacher_768_to_quantum_packets", "resolve_svd_stiefel_rank", "resolve_wavefunction_shape_spec",
]  # fmt: skip
