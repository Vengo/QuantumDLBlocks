"""Born-Schur Density-Matrix Attention & Density-Matrix Highway Laboratory (`Wave 50+` Local Exploration).

When to Use This Building Block:
- Use `BornSchurDensityAttention`, `DensityMatrixHighwayBlock`, `BornSchurMultiBranchWavepacketGPT`,
  `run_schur_thermodynamic_cooling_probe`, `run_born_schur_reversal_benchmark`, and
  `run_born_schur_fineweb_benchmark` to test shifting attention values $V$ (and optionally the
  entire residual highway) from raw wavepackets $|\\psi_{t,h}\\rangle \\in \\mathbb{C}^D$ to
  Hermitian Positive Semi-Definite (PSD) unit-trace density matrices $\\rho_{t,h} \\in \\mathbb{C}^{D \\times D}$
  assembled via **positive linear combinations of entrywise Schur (Hadamard) products**:
  $$\\tilde{\\rho}_{t,h}^{\\text{out}} = w_{t,h,0}\\, \\bar{\\rho}_{t,h} + \\sum_{(a,b)} w_{t,h,ab}\\left(\\bar{\\rho}_{t,a} \\odot_{\\mathbb{C}} \\bar{\\rho}_{t,b}\\right), \\qquad \\rho_{t,h}^{\\text{out}} = \\frac{\\tilde{\\rho}_{t,h}^{\\text{out}}}{\\operatorname{Tr}(\\tilde{\\rho}_{t,h}^{\\text{out}}) + \\varepsilon}$$

Why It Was Built & Mathematical Provenance:
- Directly implements and benchmarks the two architectures from the user's Density-Matrix Schur design:
  1. `BornSchurDensityAttention` (Wavepacket Highway `(B, T, H, 2, D)` + Density-Matrix `V` Context Gathering +
     Factored Bilinear Schur Product Mixer + Reference-Seeded Quantum Power Projection $|\\tilde{\\psi}^{\\text{out}}\\rangle = \\rho^{\\text{out}} |\\psi^{\\text{in}}\\rangle$,
     with both static squared-amplitude weights and dynamic **Born-Trace POVM Gating** $g_{t,h} = \\langle m_h | \\bar{\\rho}_{t,h} | m_h \\rangle \\in [0, 1]$).
  2. `DensityMatrixHighwayBlock` (Full Density-Matrix Residual Stream `(B, T, H, 2, D, D)` with
     Unitary $U_V$ Commutation past the attention sum, Born-Trace POVM Gating on the highway,
     and Self/Cross-Packet Schur Cooling to prevent von Neumann entropy "thermal death").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.fused_quantum_wavepacket_attn import (
    fused_low_rank_cholesky_ket_attention,
    fused_quantum_wavepacket_rho_ket_attention,
)
from ..layers.impure_density_mirror_block import DEFAULT_TIKHONOV_GRAM_LAMBDA, TRACE_NORM_EPS
from ..layers.packed_complex_register_density_attn import ComplexRoPE
from ..layers.quantum_statmech_channels import (
    QuantumMomentumScheduleMode,
    StiefelMomentumMode,
    compute_riemannian_momentum_beta,
)
from ..layers.quantum_svd_stiefel_graft import (
    BiorthogonalSVDStiefelProjection,
    QuantumStiefelProjMode,
    apply_fubini_study_self_ortho,
)
from ..models.compiled_leaderboard_classical_gpt import apply_native_leaderboard_rotary
from ..models.modular_quantum_statmech_gpt import ModularRiemannianStiefelKrausMuon
from ..models.quantum_full_backbone_graft_gpt import SVDStiefelMachZehnderInterferometer
from ..optim.vocab_manifold_muon import BatchSubspaceTikhonovGramMuon
from .micro_reversal_readout_lab import (
    DEFAULT_MICRO_BATCH_SIZE,
    DEFAULT_MICRO_NUM_HEADS,
    DEFAULT_MICRO_NUM_LAYERS,
    DEFAULT_MICRO_SCALAR_LR,
    DEFAULT_MICRO_STIEFEL_LR,
    DEFAULT_MICRO_UNITARY_LR,
    DEFAULT_MICRO_WTE_LR,
    FineWebSpanReversalDataset,
    MicroBackboneMode,
    MicroInitMode,
    MicroReadoutMode,
    MicroReversalConfig,
    MicroReversalGPT,
    _MicroQuantumWavepacketBlock,
    evaluate_micro_reversal_model,
)
from .multibranch_wavepacket_readout_lab import (
    MultiBranchMode,
    MultiBranchReadoutSpec,
    MultiBranchWavepacketGPT,
    calibrate_multibranch_residual_step0_,
    evaluate_multibranch_model,
)
from .shallow_longseq_wavepacket_lm_lab import (
    DEFAULT_SHALLOW_EMBED_DIM,
    DEFAULT_SHALLOW_MLP_HIDDEN,
    DEFAULT_SHALLOW_MUON_RETRACT_EVERY,
    DEFAULT_SHALLOW_NGRAM_LR,
    DEFAULT_SHALLOW_NUM_HEADS,
    DEFAULT_SHALLOW_PKT_COUNT,
    DEFAULT_SHALLOW_PKT_DIM,
    DEFAULT_SHALLOW_READOUT_LR,
    DEFAULT_SHALLOW_SCALAR_LR,
    DEFAULT_SHALLOW_STIEFEL_LR,
    DEFAULT_SHALLOW_UNITARY_LR,
    DEFAULT_SHALLOW_WTE_LR,
    LongSeqFineWebStream,
    _ShallowQuantumWavepacketBlock,
    initialize_and_calibrate_shallow_model_,
    project_complex_wavepackets_to_rms_sphere_,
)
from .support_preserving_countsketch_lab import (
    SupportSketchMode,
    SupportSketchSpec,
    apply_support_sketch_initialization_,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
DEFAULT_SCHUR_EPS: float = 1e-7
DEFAULT_PACKETS_H: int = 8
DEFAULT_COMPLEX_DIM_D: int = 32
DEFAULT_PROBE_RANK_R: int = 16
DEFAULT_LIN_NOISE_STD: float = 0.05
DEFAULT_SCHUR_NOISE_STD: float = 0.15
DEFAULT_PAIR_NOISE_STD: float = 0.18
DEFAULT_UNITARY_NOISE_STD: float = 0.05
DEFAULT_PROBE_INIT_STD: float = 0.15
DEFAULT_COOLING_PROBE_LAYERS: int = 8
DEFAULT_REVERSAL_STEPS: int = 200
DEFAULT_REVERSAL_EVAL_INTERVAL: int = 50
DEFAULT_FINEWEB_STEPS: int = 300
DEFAULT_FINEWEB_EVAL_INTERVAL: int = 100


class DensitySchurMode(str, Enum):
    """Strategy enum selecting McWeeny Matmul vs. Born-Schur Density Attention vs. Full Density Highway (Rule 3)."""

    MCWEENY_MATMUL_BASELINE = "mcweeny_matmul_baseline"
    BORN_SCHUR_STATIC_WEIGHTS = "born_schur_static_weights"
    BORN_SCHUR_DYNAMIC_POVM = "born_schur_dynamic_povm"
    FULL_DENSITY_MATRIX_HIGHWAY = "full_density_matrix_highway"


@dataclass(frozen=True)
class CoolingLayerMetrics:
    """Per-layer density-matrix thermodynamic diagnostics across depth (`<= 8` fields)."""

    layer_idx: int
    mean_purity_tr_rho2: float
    diag_coincidence_d_tr_schur: float
    effective_rank: float
    hermitian_err: float
    trace_err: float
    min_eigenvalue: float
    mean_schur_coincidence: float


def compute_born_trace_gate(
    rho_r: torch.Tensor,
    rho_i: torch.Tensor,
    mr: torch.Tensor,
    mi: torch.Tensor,
    eps: float = DEFAULT_SCHUR_EPS,
) -> torch.Tensor:
    r"""Compute physical Born trace expectation $\langle m \mid \rho \mid m \rangle = \operatorname{Tr}(\rho |m\rangle\langle m|) \in [0, 1]$."""
    norm = torch.sqrt(mr.square().sum(dim=-1, keepdim=True) + mi.square().sum(dim=-1, keepdim=True) + eps)
    ur, ui = mr / norm, mi / norm
    ur_4d, ui_4d = ur.unsqueeze(0).unsqueeze(2), ui.unsqueeze(0).unsqueeze(2)
    term1 = (torch.einsum("nhtcd,hd->nhtc", rho_r, ur) * ur_4d).sum(dim=-1)
    term2 = (torch.einsum("nhtcd,hd->nhtc", rho_r, ui) * ui_4d).sum(dim=-1)
    term3 = 2.0 * (torch.einsum("nhtcd,hd->nhtc", rho_i, ui) * ur_4d).sum(dim=-1)
    return (term1 + term2 - term3).clamp(min=0.0, max=1.0).unsqueeze(-1).unsqueeze(-1)


def project_complex_unitary_qr(
    ar: torch.Tensor,
    ai: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project batch of complex matrices `(H, D, D)` onto the unitary group $U(D)$ via real `2D x 2D` QR retraction."""
    d_state = ar.size(-1)
    orig_dtype = ar.dtype
    ar_f, ai_f = ar.float(), ai.float()
    block = torch.cat([torch.cat([ar_f, -ai_f], dim=-1), torch.cat([ai_f, ar_f], dim=-1)], dim=-2)
    q_mat, r_mat = torch.linalg.qr(block)
    diag_r = torch.diagonal(r_mat, dim1=-2, dim2=-1)
    diag_sign = torch.sign(diag_r) + (diag_r == 0).float()
    q_mat = q_mat * diag_sign.unsqueeze(-2)
    return q_mat[:, :d_state, :d_state].to(orig_dtype), q_mat[:, d_state:, :d_state].to(orig_dtype)


def pack_hermitian_density_matrix(rho_r: torch.Tensor, rho_i: torch.Tensor) -> torch.Tensor:
    """Pack Hermitian `(rho_r, rho_i)` of shape `(..., D, D)` into a single real `(..., D, D)` matrix."""
    d_state = rho_r.size(-1)
    triu_mask = torch.triu(torch.ones((d_state, d_state), device=rho_r.device, dtype=torch.bool), diagonal=0)
    return torch.where(triu_mask, rho_r, rho_i)


def unpack_hermitian_density_matrix(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack a single real `(..., D, D)` Hermitian-packed tensor back into `(rho_r, rho_i)`."""
    strict_upper = torch.triu(packed, diagonal=1)
    rho_r = torch.triu(packed, diagonal=0) + strict_upper.transpose(-2, -1)
    strict_lower = torch.tril(packed, diagonal=-1)
    rho_i = strict_lower - strict_lower.transpose(-2, -1)
    return rho_r, rho_i


def gather_pure_density_matrices_packed(
    probs: torch.Tensor,
    vr: torch.Tensor,
    vi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lift unit kets `(vr, vi)` of shape `(B, T, H, D)` to pure density matrices and gather via `probs` (`(B, H, T, T)`)."""
    bsz, seq_t, n_pkt, d_state = vr.shape
    vr_h, vi_h = vr.permute(0, 2, 1, 3).contiguous(), vi.permute(0, 2, 1, 3).contiguous()
    rho_v_r = vr_h.unsqueeze(-1) * vr_h.unsqueeze(-2) + vi_h.unsqueeze(-1) * vi_h.unsqueeze(-2)
    rho_v_i = vi_h.unsqueeze(-1) * vr_h.unsqueeze(-2) - vr_h.unsqueeze(-1) * vi_h.unsqueeze(-2)
    packed_v = pack_hermitian_density_matrix(rho_v_r, rho_v_i).view(bsz, n_pkt, seq_t, d_state * d_state)
    gathered_packed = torch.matmul(probs.to(dtype=packed_v.dtype), packed_v).view(bsz, n_pkt, seq_t, d_state, d_state)
    return unpack_hermitian_density_matrix(gathered_packed)


class BornSchurDensityAttention(nn.Module):
    r"""Standalone Born-Schur Density-Matrix Attention for `(B, T, 8, 2, D)` wavepackets (Section 5 of user design)."""

    def __init__(
        self,
        n_packets: int = DEFAULT_PACKETS_H,
        d_packet: int = DEFAULT_COMPLEX_DIM_D,
        use_dynamic_povm_gates: bool = True,
        eps: float = DEFAULT_SCHUR_EPS,
    ) -> None:
        super().__init__()
        self.H, self.D = int(n_packets), int(d_packet)
        self.use_dynamic_povm_gates = bool(use_dynamic_povm_gates)
        self.eps = float(eps)
        self.total_dim = self.H * self.D

        self.q_real = nn.Linear(self.total_dim, self.total_dim, bias=False)
        self.q_imag = nn.Linear(self.total_dim, self.total_dim, bias=False)
        self.k_real = nn.Linear(self.total_dim, self.total_dim, bias=False)
        self.k_imag = nn.Linear(self.total_dim, self.total_dim, bias=False)
        self.v_real = nn.Linear(self.total_dim, self.total_dim, bias=False)
        self.v_imag = nn.Linear(self.total_dim, self.total_dim, bias=False)
        self.rope = ComplexRoPE(d_register=self.D)

        self.raw_w_lin = nn.Parameter(torch.eye(self.H) + DEFAULT_LIN_NOISE_STD * torch.randn(self.H, self.H))
        self.raw_u_schur = nn.Parameter(torch.eye(self.H) + DEFAULT_SCHUR_NOISE_STD * torch.randn(self.H, self.H))
        self.raw_v_schur = nn.Parameter(
            torch.eye(self.H).roll(shifts=1, dims=1) + DEFAULT_SCHUR_NOISE_STD * torch.randn(self.H, self.H)
        )
        self.m_lin_r = nn.Parameter(torch.randn(self.H, self.D))
        self.m_lin_i = nn.Parameter(torch.randn(self.H, self.D))
        self.m_schur_r = nn.Parameter(torch.randn(self.H, self.D))
        self.m_schur_i = nn.Parameter(torch.randn(self.H, self.D))

    def _complex_proj(self, x: torch.Tensor, lin_r: nn.Linear, lin_i: nn.Linear) -> torch.Tensor:
        bsz, seq_t, n_pkt, _, d_state = x.shape
        xr = x[:, :, :, 0, :].reshape(bsz, seq_t, n_pkt * d_state)
        xi = x[:, :, :, 1, :].reshape(bsz, seq_t, n_pkt * d_state)
        yr = (lin_r(xr) - lin_i(xi)).view(bsz, seq_t, n_pkt, d_state)
        yi = (lin_r(xi) + lin_i(xr)).view(bsz, seq_t, n_pkt, d_state)
        norm = torch.sqrt(yr.float().square().sum(dim=-1, keepdim=True) + yi.float().square().sum(dim=-1, keepdim=True) + self.eps)
        return torch.stack([(yr.float() / norm).to(x.dtype), (yi.float() / norm).to(x.dtype)], dim=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_t, _, _, _ = x.shape
        q = self._complex_proj(x, self.q_real, self.q_imag)
        k = self._complex_proj(x, self.k_real, self.k_imag)
        v = self._complex_proj(x, self.v_real, self.v_imag)

        qr_h, qi_h = self.rope(q[:, :, :, 0, :].permute(0, 2, 1, 3), q[:, :, :, 1, :].permute(0, 2, 1, 3))
        kr_h, ki_h = self.rope(k[:, :, :, 0, :].permute(0, 2, 1, 3), k[:, :, :, 1, :].permute(0, 2, 1, 3))
        vr, vi = v[:, :, :, 0, :], v[:, :, :, 1, :]

        ov_real = torch.matmul(qr_h.float(), kr_h.float().transpose(-2, -1)) + torch.matmul(
            qi_h.float(), ki_h.float().transpose(-2, -1)
        )
        ov_imag = torch.matmul(qr_h.float(), ki_h.float().transpose(-2, -1)) - torch.matmul(
            qi_h.float(), kr_h.float().transpose(-2, -1)
        )
        born_scores = (ov_real.square() + ov_imag.square()).masked_fill(
            ~torch.tril(torch.ones((seq_t, seq_t), device=x.device, dtype=torch.bool)), 0.0
        )
        probs = born_scores / (born_scores.sum(dim=-1, keepdim=True) + self.eps)

        bar_rho_r, bar_rho_i = gather_pure_density_matrices_packed(probs, vr, vi)
        bar_rho_r, bar_rho_i = bar_rho_r.float(), bar_rho_i.float()

        w_lin, u_schur, v_schur = self.raw_w_lin.float().square(), self.raw_u_schur.float().square(), self.raw_v_schur.float().square()
        lin_r = torch.einsum("ha,natcd->nhtcd", w_lin, bar_rho_r)
        lin_i = torch.einsum("ha,natcd->nhtcd", w_lin, bar_rho_i)
        left_r = torch.einsum("ha,natcd->nhtcd", u_schur, bar_rho_r)
        left_i = torch.einsum("ha,natcd->nhtcd", u_schur, bar_rho_i)
        right_r = torch.einsum("hp,nptcd->nhtcd", v_schur, bar_rho_r)
        right_i = torch.einsum("hp,nptcd->nhtcd", v_schur, bar_rho_i)

        schur_r = left_r * right_r - left_i * right_i
        schur_i = left_r * right_i + left_i * right_r

        if self.use_dynamic_povm_gates:
            g_lin = compute_born_trace_gate(bar_rho_r, bar_rho_i, self.m_lin_r.float(), self.m_lin_i.float(), self.eps)
            g_schur = compute_born_trace_gate(bar_rho_r, bar_rho_i, self.m_schur_r.float(), self.m_schur_i.float(), self.eps)
            out_rho_r = (g_lin + self.eps) * lin_r + g_schur * schur_r
            out_rho_i = (g_lin + self.eps) * lin_i + g_schur * schur_i
        else:
            out_rho_r, out_rho_i = lin_r + schur_r, lin_i + schur_i

        trace = torch.diagonal(out_rho_r, dim1=-2, dim2=-1).sum(dim=-1).unsqueeze(-1).unsqueeze(-1)
        out_rho_r, out_rho_i = out_rho_r / (trace + self.eps), out_rho_i / (trace + self.eps)

        xr_in = x[:, :, :, 0, :].permute(0, 2, 1, 3).float()
        xi_in = x[:, :, :, 1, :].permute(0, 2, 1, 3).float()
        psi_out_r = torch.einsum("nhtcd,nhtd->nhtc", out_rho_r, xr_in) - torch.einsum("nhtcd,nhtd->nhtc", out_rho_i, xi_in)
        psi_out_i = torch.einsum("nhtcd,nhtd->nhtc", out_rho_r, xi_in) + torch.einsum("nhtcd,nhtd->nhtc", out_rho_i, xr_in)

        out_norm = torch.sqrt(psi_out_r.square().sum(dim=-1, keepdim=True) + psi_out_i.square().sum(dim=-1, keepdim=True) + self.eps)
        return torch.stack([psi_out_r / out_norm, psi_out_i / out_norm], dim=3).permute(0, 2, 1, 3, 4).to(dtype=x.dtype)


class DensityMatrixHighwayBlock(nn.Module):
    r"""Transformer-style CPTP Block operating directly on a Density-Matrix Highway `(B, T, H, 2, D, D)`."""

    def __init__(
        self,
        n_packets: int = DEFAULT_PACKETS_H,
        d_state: int = DEFAULT_COMPLEX_DIM_D,
        d_probe: int = DEFAULT_PROBE_RANK_R,
        enable_schur_cooling: bool = True,
        enable_mcweeny_purification: bool = True,
        eps: float = DEFAULT_SCHUR_EPS,
    ) -> None:
        super().__init__()
        self.H, self.D, self.R = int(n_packets), int(d_state), int(d_probe)
        self.enable_schur_cooling = bool(enable_schur_cooling)
        self.enable_mcweeny_purification = bool(enable_mcweeny_purification)
        self.eps = float(eps)

        self.q_probe_r = nn.Parameter(torch.randn(self.H, self.D, self.R) * DEFAULT_PROBE_INIT_STD)
        self.q_probe_i = nn.Parameter(torch.randn(self.H, self.D, self.R) * DEFAULT_PROBE_INIT_STD)
        self.k_probe_r = nn.Parameter(torch.randn(self.H, self.D, self.R) * DEFAULT_PROBE_INIT_STD)
        self.k_probe_i = nn.Parameter(torch.randn(self.H, self.D, self.R) * DEFAULT_PROBE_INIT_STD)
        self.rope = ComplexRoPE(d_register=self.R)

        eye_h = torch.eye(self.D).unsqueeze(0).repeat(self.H, 1, 1)
        self.v_unitary_r = nn.Parameter(eye_h + DEFAULT_UNITARY_NOISE_STD * torch.randn(self.H, self.D, self.D))
        self.v_unitary_i = nn.Parameter(DEFAULT_UNITARY_NOISE_STD * torch.randn(self.H, self.D, self.D))

        self.m_lin_r = nn.Parameter(torch.randn(self.H, self.D))
        self.m_lin_i = nn.Parameter(torch.randn(self.H, self.D))
        self.m_schur_r = nn.Parameter(torch.randn(self.H, self.D))
        self.m_schur_i = nn.Parameter(torch.randn(self.H, self.D))
        self.m_hwy_r = nn.Parameter(torch.randn(self.H, self.D))
        self.m_hwy_i = nn.Parameter(torch.randn(self.H, self.D))
        self.raw_pair_mix = nn.Parameter(torch.eye(self.H) + DEFAULT_PAIR_NOISE_STD * torch.randn(self.H, self.H))

    def complex_unitary_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        return [(self.v_unitary_r, self.v_unitary_i)]

    def scalar_parameters(self) -> list[nn.Parameter]:
        return [
            self.q_probe_r, self.q_probe_i, self.k_probe_r, self.k_probe_i,
            self.m_lin_r, self.m_lin_i, self.m_schur_r, self.m_schur_i,
            self.m_hwy_r, self.m_hwy_i, self.raw_pair_mix,
        ]

    def _extract_rotary_probe_kets(
        self,
        rho_r: torch.Tensor,
        rho_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_norm = torch.sqrt(self.q_probe_r.float().square().sum(dim=1, keepdim=True) + self.q_probe_i.float().square().sum(dim=1, keepdim=True) + self.eps)
        k_norm = torch.sqrt(self.k_probe_r.float().square().sum(dim=1, keepdim=True) + self.k_probe_i.float().square().sum(dim=1, keepdim=True) + self.eps)
        qr, qi = self.q_probe_r.float() / q_norm, self.q_probe_i.float() / q_norm
        kr, ki = self.k_probe_r.float() / k_norm, self.k_probe_i.float() / k_norm

        q_ket_r = torch.einsum("nhtcd,hdr->nhtr", rho_r, qr) + torch.einsum("nhtcd,hdr->nhtr", rho_i, qi)
        q_ket_i = torch.einsum("nhtcd,hdr->nhtr", rho_i, qr) - torch.einsum("nhtcd,hdr->nhtr", rho_r, qi)
        k_ket_r = torch.einsum("nhscd,hdr->nhsr", rho_r, kr) + torch.einsum("nhscd,hdr->nhsr", rho_i, ki)
        k_ket_i = torch.einsum("nhscd,hdr->nhsr", rho_i, kr) - torch.einsum("nhscd,hdr->nhsr", rho_r, ki)
        q_rot_r, q_rot_i = self.rope(q_ket_r, q_ket_i)
        k_rot_r, k_rot_i = self.rope(k_ket_r, k_ket_i)
        return q_rot_r, q_rot_i, k_rot_r, k_rot_i

    def forward(self, rho_hwy: torch.Tensor) -> tuple[torch.Tensor, float]:
        bsz, seq_t, n_pkt, _, d_state, _ = rho_hwy.shape
        rho_r = rho_hwy[:, :, :, 0, :, :].permute(0, 2, 1, 3, 4).float()
        rho_i = rho_hwy[:, :, :, 1, :, :].permute(0, 2, 1, 3, 4).float()

        q_rot_r, q_rot_i, k_rot_r, k_rot_i = self._extract_rotary_probe_kets(rho_r, rho_i)
        ov_r = torch.matmul(q_rot_r, k_rot_r.transpose(-2, -1)) + torch.matmul(q_rot_i, k_rot_i.transpose(-2, -1))
        ov_i = torch.matmul(q_rot_r, k_rot_i.transpose(-2, -1)) - torch.matmul(q_rot_i, k_rot_r.transpose(-2, -1))
        born_scores = (ov_r.square() + ov_i.square()).masked_fill(
            ~torch.tril(torch.ones((seq_t, seq_t), device=rho_hwy.device, dtype=torch.bool)), 0.0
        )
        probs = born_scores / (born_scores.sum(dim=-1, keepdim=True) + self.eps)

        packed_hwy = pack_hermitian_density_matrix(rho_r, rho_i).view(bsz, n_pkt, seq_t, d_state * d_state)
        gathered_r, gathered_i = unpack_hermitian_density_matrix(
            torch.matmul(probs, packed_hwy).view(bsz, n_pkt, seq_t, d_state, d_state)
        )

        ur, ui = project_complex_unitary_qr(self.v_unitary_r, self.v_unitary_i)
        ur_f, ui_f = ur.float(), ui.float()
        m_r = torch.einsum("hce,nhted->nhtcd", ur_f, gathered_r) - torch.einsum("hce,nhted->nhtcd", ui_f, gathered_i)
        m_i = torch.einsum("hce,nhted->nhtcd", ur_f, gathered_i) + torch.einsum("hce,nhted->nhtcd", ui_f, gathered_r)
        bar_rho_r = torch.einsum("nhtce,hde->nhtcd", m_r, ur_f) + torch.einsum("nhtce,hde->nhtcd", m_i, ui_f)
        bar_rho_i = torch.einsum("nhtce,hde->nhtcd", m_i, ur_f) - torch.einsum("nhtce,hde->nhtcd", m_r, ui_f)

        w_lin = compute_born_trace_gate(rho_r, rho_i, self.m_lin_r.float(), self.m_lin_i.float(), self.eps)
        w_schur = compute_born_trace_gate(rho_r, rho_i, self.m_schur_r.float(), self.m_schur_i.float(), self.eps)
        pair_weights = self.raw_pair_mix.float().square()
        partner_r = torch.einsum("ha,natcd->nhtcd", pair_weights, bar_rho_r)
        partner_i = torch.einsum("ha,natcd->nhtcd", pair_weights, bar_rho_i)

        schur_r = float(d_state) * (bar_rho_r * partner_r - bar_rho_i * partner_i)
        schur_i = float(d_state) * (bar_rho_r * partner_i + bar_rho_i * partner_r)
        coincidence = float(torch.diagonal(schur_r, dim1=-2, dim2=-1).sum(dim=-1).mean().detach().item())

        if self.enable_schur_cooling:
            block_r = (w_lin + self.eps) * bar_rho_r + w_schur * schur_r
            block_i = (w_lin + self.eps) * bar_rho_i + w_schur * schur_i
        else:
            block_r, block_i = bar_rho_r, bar_rho_i

        trace = torch.diagonal(block_r, dim1=-2, dim2=-1).sum(dim=-1).unsqueeze(-1).unsqueeze(-1)
        block_r, block_i = block_r / (trace + self.eps), block_i / (trace + self.eps)

        alpha = compute_born_trace_gate(rho_r, rho_i, self.m_hwy_r.float(), self.m_hwy_i.float(), self.eps)
        next_r = (1.0 - alpha) * rho_r + alpha * block_r
        next_i = (1.0 - alpha) * rho_i + alpha * block_i

        if self.enable_mcweeny_purification:
            # 1-step Hermitian McWeeny purification rho <- rho^2 / Tr(rho^2) (Section 4.2 of user design)
            # Since next_r is symmetric and next_i is skew-symmetric, Tr(next^2) = ||next_r||_F^2 + ||next_i||_F^2
            sq_r = torch.matmul(next_r, next_r) - torch.matmul(next_i, next_i)
            sq_i = torch.matmul(next_r, next_i) + torch.matmul(next_i, next_r)
            tr_sq = torch.diagonal(sq_r, dim1=-2, dim2=-1).sum(dim=-1).unsqueeze(-1).unsqueeze(-1)
            next_r = sq_r / (tr_sq + self.eps)
            next_i = sq_i / (tr_sq + self.eps)

        return torch.stack([next_r, next_i], dim=3).permute(0, 2, 1, 3, 4, 5).to(dtype=rho_hwy.dtype), coincidence


def assemble_born_schur_density_and_project(
    rho0_re: torch.Tensor,
    rho0_im: torch.Tensor,
    v_bar_re: torch.Tensor,
    v_bar_im: torch.Tensor,
    v_mixed: torch.Tensor,
    u_evol_pair: tuple[torch.Tensor, torch.Tensor],
    schur_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    scales_and_flags: tuple[torch.Tensor, torch.Tensor, bool],
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Assemble Inter-Packet Schur Products $\bar{\rho}_a \odot_{\mathbb{C}} \bar{\rho}_b$ and apply McWeeny + Reference-Seeded Power Projection."""
    from ..kernels.fused_quantum_wavepacket_attn import associative_mcweeny_unitary_ket_modulation

    d_h = rho0_re.size(-1)
    work_dtype = v_bar_re.dtype
    tr0 = torch.diagonal(rho0_re.float(), dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(TRACE_NORM_EPS).to(work_dtype)
    rr, ri = rho0_re.to(work_dtype) / tr0, rho0_im.to(work_dtype) / tr0

    raw_w_lin, raw_pair_mix, m_lin_r, m_lin_i, m_schur_r, m_schur_i = schur_params
    rho_scale, obs_scale, use_dyn_povm = scales_and_flags

    w_lin = raw_w_lin.float().square().to(work_dtype)
    w_pair = raw_pair_mix.float().square().to(work_dtype)
    lin_r = torch.einsum("ha,natcd->nhtcd", w_lin, rr)
    lin_i = torch.einsum("ha,natcd->nhtcd", w_lin, ri)
    partner_r = torch.einsum("ha,natcd->nhtcd", w_pair, rr)
    partner_i = torch.einsum("ha,natcd->nhtcd", w_pair, ri)

    schur_r = float(d_h) * (rr * partner_r - ri * partner_i)
    schur_i = float(d_h) * (rr * partner_i + ri * partner_r)

    if use_dyn_povm:
        g_lin = compute_born_trace_gate(rr.float(), ri.float(), m_lin_r.float(), m_lin_i.float(), DEFAULT_SCHUR_EPS).to(work_dtype)
        g_schur = compute_born_trace_gate(rr.float(), ri.float(), m_schur_r.float(), m_schur_i.float(), DEFAULT_SCHUR_EPS).to(work_dtype)
        tilde_r = (g_lin + DEFAULT_SCHUR_EPS) * lin_r + g_schur * schur_r
        tilde_i = (g_lin + DEFAULT_SCHUR_EPS) * lin_i + g_schur * schur_i
    else:
        tilde_r, tilde_i = lin_r + schur_r, lin_i + schur_i

    tr_val = torch.diagonal(tilde_r.float(), dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(TRACE_NORM_EPS).to(work_dtype)
    out_rho_r = tilde_r / tr_val
    out_rho_i = tilde_i / tr_val

    return associative_mcweeny_unitary_ket_modulation(
        out_rho_r,
        out_rho_i,
        v_bar_re,
        v_bar_im,
        v_mixed,
        u_evol_pair,
        (rho_scale, obs_scale, True),
    )


class _BornSchurMicroReversalBlock(_MicroQuantumWavepacketBlock):
    """Subclass of `_MicroQuantumWavepacketBlock` replacing McWeeny matmul with Inter-Packet Born-Schur Assembly."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        pkt_count: int,
        pkt_dim: int,
        mlp_hidden: int,
        use_dynamic_povm: bool = True,
    ) -> None:
        super().__init__(embed_dim, num_heads, pkt_count, pkt_dim, mlp_hidden)
        self.use_dynamic_povm = bool(use_dynamic_povm)
        self.raw_w_lin = nn.Parameter(torch.eye(self.num_heads) + DEFAULT_LIN_NOISE_STD * torch.randn(self.num_heads, self.num_heads))
        self.raw_pair_mix = nn.Parameter(torch.eye(self.num_heads) + DEFAULT_PAIR_NOISE_STD * torch.randn(self.num_heads, self.num_heads))
        self.m_lin_r = nn.Parameter(torch.randn(self.num_heads, self.d_h))
        self.m_lin_i = nn.Parameter(torch.randn(self.num_heads, self.d_h))
        self.m_schur_r = nn.Parameter(torch.randn(self.num_heads, self.d_h))
        self.m_schur_i = nn.Parameter(torch.randn(self.num_heads, self.d_h))

    def scalar_parameters(self) -> list[nn.Parameter]:
        return super().scalar_parameters() + [
            self.raw_w_lin, self.raw_pair_mix, self.m_lin_r, self.m_lin_i, self.m_schur_r, self.m_schur_i
        ]

    def forward(self, x: torch.Tensor, cos_4d: torch.Tensor, sin_4d: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        h_in = F.rms_norm(x, (d_model,), eps=1e-6)
        q_raw = self.q_proj(h_in).view(bsz, seq_len, self.num_heads, self.head_dim)
        k_raw = self.k_proj(h_in).view(bsz, seq_len, self.num_heads, self.head_dim)
        v_wp = self.v_proj(h_in).view(bsz, seq_len, self.pkt_count, self.d_h)

        q_rot = apply_native_leaderboard_rotary(F.rms_norm(q_raw, (self.head_dim,), eps=1e-6), cos_4d, sin_4d)
        k_rot = apply_native_leaderboard_rotary(F.rms_norm(k_raw, (self.head_dim,), eps=1e-6), cos_4d, sin_4d)

        v_c = v_wp.view(bsz, seq_len, self.num_heads, self.v_rank, 2, self.d_h).permute(0, 2, 1, 3, 4, 5)
        vn_re, vn_im = v_c[..., 0, :], v_c[..., 1, :]
        nsq = (
            (vn_re.float().square().sum(dim=(-2, -1), keepdim=True) + vn_im.float().square().sum(dim=(-2, -1), keepdim=True))
            .clamp_min(TRACE_NORM_EPS).rsqrt().to(v_wp.dtype)
        )
        r_re, r_im = vn_re * nsq, vn_im * nsq
        tok_rho_re = torch.matmul(r_re.transpose(-2, -1), r_re) + torch.matmul(r_im.transpose(-2, -1), r_im)
        tok_rho_im = torch.matmul(r_im.transpose(-2, -1), r_re) - torch.matmul(r_re.transpose(-2, -1), r_im)

        v_mixed = v_wp.view(bsz, seq_len, self.num_heads, self.head_dim)
        rho0_re, rho0_im, v_bar_re, v_bar_im = fused_quantum_wavepacket_rho_ket_attention(
            q_rot, k_rot, tok_rho_re, tok_rho_im, v_mixed, attn_mask=None,
            phase_and_born=(self.berry_qk_phase, self.born_qk_beta, self.sdpa_scale, True),
        )
        rho_scale = torch.tanh(self.rho_coupling).view(1, self.num_heads, 1, 1, 1)
        y_heads, pur_t = assemble_born_schur_density_and_project(
            rho0_re, rho0_im, v_bar_re, v_bar_im, v_mixed,
            (self.u_evol_re, self.u_evol_im),
            (self.raw_w_lin, self.raw_pair_mix, self.m_lin_r, self.m_lin_i, self.m_schur_r, self.m_schur_i),
            (rho_scale, self.obs_scale, self.use_dynamic_povm),
        )
        self.last_mean_purity = float(pur_t.mean().item())
        self.last_mean_eff_rank = float((1.0 / pur_t.clamp_min(1e-4)).mean().item())

        y_ortho = apply_fubini_study_self_ortho(y_heads, v_mixed, self.self_ortho_alpha)
        y_flat = F.rms_norm(y_ortho.reshape(bsz, seq_len, self.pkt_count, self.d_h), (self.d_h,), eps=1e-6).reshape(bsz, seq_len, d_model)
        return self.mz_mlp(x + self.obs_proj(y_flat))


class _DensityHighwayReversalAdapterBlock(_BornSchurMicroReversalBlock):
    """Subclass of `_BornSchurMicroReversalBlock` adding Commuted Unitary Conjugation + Density-Matrix Highway Gate (`D3`)."""

    def __init__(self, embed_dim: int, num_heads: int, pkt_count: int, pkt_dim: int, mlp_hidden: int) -> None:
        super().__init__(embed_dim, num_heads, pkt_count, pkt_dim, mlp_hidden, use_dynamic_povm=True)
        self.m_hwy_r = nn.Parameter(torch.randn(self.num_heads, self.d_h))
        self.m_hwy_i = nn.Parameter(torch.randn(self.num_heads, self.d_h))

    def scalar_parameters(self) -> list[nn.Parameter]:
        return super().scalar_parameters() + [self.m_hwy_r, self.m_hwy_i]

    def forward(self, x: torch.Tensor, cos_4d: torch.Tensor, sin_4d: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        h_in = F.rms_norm(x, (d_model,), eps=1e-6)
        q_raw = self.q_proj(h_in).view(bsz, seq_len, self.num_heads, self.head_dim)
        k_raw = self.k_proj(h_in).view(bsz, seq_len, self.num_heads, self.head_dim)
        v_wp = self.v_proj(h_in).view(bsz, seq_len, self.pkt_count, self.d_h)

        q_rot = apply_native_leaderboard_rotary(F.rms_norm(q_raw, (self.head_dim,), eps=1e-6), cos_4d, sin_4d)
        k_rot = apply_native_leaderboard_rotary(F.rms_norm(k_raw, (self.head_dim,), eps=1e-6), cos_4d, sin_4d)

        v_c = v_wp.view(bsz, seq_len, self.num_heads, self.v_rank, 2, self.d_h).permute(0, 2, 1, 3, 4, 5)
        vn_re, vn_im = v_c[..., 0, :], v_c[..., 1, :]
        nsq = (
            (vn_re.float().square().sum(dim=(-2, -1), keepdim=True) + vn_im.float().square().sum(dim=(-2, -1), keepdim=True))
            .clamp_min(TRACE_NORM_EPS).rsqrt().to(v_wp.dtype)
        )
        r_re, r_im = vn_re * nsq, vn_im * nsq
        tok_rho_re = torch.matmul(r_re.transpose(-2, -1), r_re) + torch.matmul(r_im.transpose(-2, -1), r_im)
        tok_rho_im = torch.matmul(r_im.transpose(-2, -1), r_re) - torch.matmul(r_re.transpose(-2, -1), r_im)

        v_mixed = v_wp.view(bsz, seq_len, self.num_heads, self.head_dim)
        rho0_re, rho0_im, v_bar_re, v_bar_im = fused_quantum_wavepacket_rho_ket_attention(
            q_rot, k_rot, tok_rho_re, tok_rho_im, v_mixed, attn_mask=None,
            phase_and_born=(self.berry_qk_phase, self.born_qk_beta, self.sdpa_scale, True),
        )
        # Born-Trace Highway gate <m_hwy | rho_tok | m_hwy> == |<m_hwy | r>|^2 + convex highway blend
        mh_norm = torch.sqrt(self.m_hwy_r.float().square().sum(dim=-1, keepdim=True) + self.m_hwy_i.float().square().sum(dim=-1, keepdim=True) + DEFAULT_SCHUR_EPS)
        mhr = (self.m_hwy_r.float() / mh_norm).view(1, self.num_heads, 1, 1, self.d_h)
        mhi = (self.m_hwy_i.float() / mh_norm).view(1, self.num_heads, 1, 1, self.d_h)
        ov_r = (r_re.float() * mhr + r_im.float() * mhi).sum(dim=-1, keepdim=True)
        ov_i = (r_im.float() * mhr - r_re.float() * mhi).sum(dim=-1, keepdim=True)
        alpha_hwy = (0.50 + 0.45 * (ov_r.square() + ov_i.square()).clamp(0.0, 1.0)).to(rho0_re.dtype)
        hwy_r = torch.lerp(tok_rho_re, rho0_re, alpha_hwy)
        hwy_i = torch.lerp(tok_rho_im, rho0_im, alpha_hwy)

        rho_scale = torch.tanh(self.rho_coupling).view(1, self.num_heads, 1, 1, 1)
        y_heads, pur_t = assemble_born_schur_density_and_project(
            hwy_r, hwy_i, v_bar_re, v_bar_im, v_mixed,
            (self.u_evol_re, self.u_evol_im),
            (self.raw_w_lin, self.raw_pair_mix, self.m_lin_r, self.m_lin_i, self.m_schur_r, self.m_schur_i),
            (rho_scale, self.obs_scale, True),
        )
        self.last_mean_purity = float(pur_t.mean().item())
        self.last_mean_eff_rank = float((1.0 / pur_t.clamp_min(1e-4)).mean().item())

        y_ortho = apply_fubini_study_self_ortho(y_heads, v_mixed, self.self_ortho_alpha)
        y_flat = F.rms_norm(y_ortho.reshape(bsz, seq_len, self.pkt_count, self.d_h), (self.d_h,), eps=1e-6).reshape(bsz, seq_len, d_model)
        return self.mz_mlp(x + self.obs_proj(y_flat))


class _ShallowBornSchurWavepacketBlock(_ShallowQuantumWavepacketBlock):
    r"""Subclass of `_ShallowQuantumWavepacketBlock` adding Inter-Packet Born-Schur Density Assembly (`D4`)."""

    def __init__(
        self,
        embed_dim: int = DEFAULT_SHALLOW_EMBED_DIM,
        num_heads: int = DEFAULT_SHALLOW_NUM_HEADS,
        pkt_count: int = DEFAULT_SHALLOW_PKT_COUNT,
        pkt_dim: int = DEFAULT_SHALLOW_PKT_DIM,
        mlp_hidden: int = DEFAULT_SHALLOW_MLP_HIDDEN,
    ) -> None:
        super().__init__(embed_dim, num_heads, pkt_count, pkt_dim, mlp_hidden)
        self.raw_w_lin = nn.Parameter(torch.eye(self.num_heads) + DEFAULT_LIN_NOISE_STD * torch.randn(self.num_heads, self.num_heads))
        self.raw_pair_mix = nn.Parameter(torch.eye(self.num_heads) + DEFAULT_PAIR_NOISE_STD * torch.randn(self.num_heads, self.num_heads))
        self.m_lin_r = nn.Parameter(torch.randn(self.num_heads, self.d_h))
        self.m_lin_i = nn.Parameter(torch.randn(self.num_heads, self.d_h))
        self.m_schur_r = nn.Parameter(torch.randn(self.num_heads, self.d_h))
        self.m_schur_i = nn.Parameter(torch.randn(self.num_heads, self.d_h))

    def scalar_parameters(self) -> list[nn.Parameter]:
        return super().scalar_parameters() + [
            self.raw_w_lin, self.raw_pair_mix, self.m_lin_r, self.m_lin_i, self.m_schur_r, self.m_schur_i
        ]

    def _cholesky_schur_step(
        self,
        l_re: torch.Tensor,
        l_im: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = l_re.size(0)
        work_dtype = l_re.dtype
        w_pair = self.raw_pair_mix.float().square().to(work_dtype)
        part_l_r = torch.matmul(w_pair, l_re.reshape(bsz, self.num_heads, -1)).view_as(l_re)
        part_l_i = torch.matmul(w_pair, l_im.reshape(bsz, self.num_heads, -1)).view_as(l_im)

        sqrt_d = math.sqrt(float(self.d_h))
        schur_l_r = sqrt_d * (l_re * part_l_r - l_im * part_l_i)
        schur_l_i = sqrt_d * (l_re * part_l_i + l_im * part_l_r)

        ml_norm = torch.sqrt(self.m_lin_r.float().square().sum(dim=-1, keepdim=True) + self.m_lin_i.float().square().sum(dim=-1, keepdim=True) + DEFAULT_SCHUR_EPS)
        ms_norm = torch.sqrt(self.m_schur_r.float().square().sum(dim=-1, keepdim=True) + self.m_schur_i.float().square().sum(dim=-1, keepdim=True) + DEFAULT_SCHUR_EPS)
        mlr = (self.m_lin_r.float() / ml_norm).to(work_dtype).view(1, self.num_heads, 1, self.d_h, 1)
        mli = (self.m_lin_i.float() / ml_norm).to(work_dtype).view(1, self.num_heads, 1, self.d_h, 1)
        msr = (self.m_schur_r.float() / ms_norm).to(work_dtype).view(1, self.num_heads, 1, self.d_h, 1)
        msi = (self.m_schur_i.float() / ms_norm).to(work_dtype).view(1, self.num_heads, 1, self.d_h, 1)

        ov_lin_r = torch.matmul(l_re, mlr) + torch.matmul(l_im, mli)
        ov_lin_i = torch.matmul(l_im, mlr) - torch.matmul(l_re, mli)
        ov_sch_r = torch.matmul(l_re, msr) + torch.matmul(l_im, msi)
        ov_sch_i = torch.matmul(l_im, msr) - torch.matmul(l_re, msi)
        g_lin = torch.sqrt((ov_lin_r.float().square() + ov_lin_i.float().square()).sum(dim=-2, keepdim=True).clamp(1e-5, 1.0)).to(work_dtype)
        g_schur = torch.sqrt((ov_sch_r.float().square() + ov_sch_i.float().square()).sum(dim=-2, keepdim=True).clamp(1e-5, 1.0)).to(work_dtype)

        out_l_r = (g_lin + DEFAULT_SCHUR_EPS) * l_re + g_schur * schur_l_r
        out_l_i = (g_lin + DEFAULT_SCHUR_EPS) * l_im + g_schur * schur_l_i
        frob = torch.sqrt(out_l_r.float().square().sum(dim=(-2, -1), keepdim=True) + out_l_i.float().square().sum(dim=(-2, -1), keepdim=True) + TRACE_NORM_EPS).to(work_dtype)
        return out_l_r / frob, out_l_i / frob

    def forward(
        self,
        x: torch.Tensor,
        x_anchor: torch.Tensor,
        cos_4d: torch.Tensor,
        sin_4d: torch.Tensor,
    ) -> torch.Tensor:
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
            .clamp_min(TRACE_NORM_EPS).rsqrt().to(work_dtype)
        )
        r_re, r_im = vn_re * nsq, vn_im * nsq

        l_re, l_im, v_bar_re, v_bar_im = fused_low_rank_cholesky_ket_attention(
            q_rot, k_rot, (r_re, r_im), v_mixed_h, attn_mask=None,
            phase_and_born=(self.berry_qk_phase, self.born_qk_beta, self.sdpa_scale, True), k_chol=4,
        )
        from ..kernels.fused_quantum_wavepacket_attn import associative_low_rank_cholesky_gram_modulation

        if self.training and l_re.requires_grad:
            out_l_r, out_l_i = torch.utils.checkpoint.checkpoint(self._cholesky_schur_step, l_re, l_im, use_reentrant=False)
        else:
            out_l_r, out_l_i = self._cholesky_schur_step(l_re, l_im)

        rho_scale = torch.tanh(self.rho_coupling).view(1, self.num_heads, 1, 1, 1)
        y_heads, pur_t = associative_low_rank_cholesky_gram_modulation(
            out_l_r, out_l_i, v_bar_re, v_bar_im, v_mixed_h,
            (self.u_evol_re, self.u_evol_im),
            (rho_scale, self.obs_scale, True),
        )
        self.last_mean_purity = float(pur_t.mean().item())
        self.last_mean_eff_rank = float((1.0 / pur_t.clamp_min(1e-4)).mean().item())

        v_head_rms = v_mixed_h.float().pow(2).mean(dim=-1, keepdim=True).clamp_min(TRACE_NORM_EPS).sqrt().to(work_dtype)
        y_scaled = y_heads.view(bsz, seq_len, self.num_heads, self.head_dim) * v_head_rms
        y_ortho = apply_fubini_study_self_ortho(y_scaled, v_mixed_h, self.self_ortho_alpha)
        g_at = (2.0 * torch.sigmoid(F.linear(h_in[..., : self.gate_in_ch], self.q_attn_gate.weight.to(work_dtype)))).unsqueeze(-1)
        y_flat = (y_ortho * g_at).reshape(bsz, seq_len, d_model)
        return self.mz_mlp(x_in + self.obs_proj(y_flat))


class BornSchurMultiBranchWavepacketGPT(MultiBranchWavepacketGPT):
    """`MultiBranchWavepacketGPT` with `_ShallowBornSchurWavepacketBlock` layers (`D4`)."""

    def __init__(self, spec: MultiBranchReadoutSpec) -> None:
        super().__init__(spec)
        self.quantum_blocks = nn.ModuleList(
            [
                _ShallowBornSchurWavepacketBlock(
                    self.embed_dim, self.num_heads, self.pkt_count, self.pkt_dim, self.mlp_hidden
                )
                for _ in range(self.num_layers)
            ]
        )


def run_schur_thermodynamic_cooling_probe(
    n_packets: int = DEFAULT_PACKETS_H,
    d_state: int = DEFAULT_COMPLEX_DIM_D,
    num_layers: int = DEFAULT_COOLING_PROBE_LAYERS,
    device: str = "cuda",
) -> dict[str, Any]:
    """Verify Hermitian/PSD/Trace invariants, Pure vs. Mixed Schur Identity, and Schur Cooling vs. Thermal Death."""
    torch.manual_seed(1337)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    u_vec = torch.randn(n_packets, 2, d_state, device=dev)
    v_vec = torch.randn(n_packets, 2, d_state, device=dev)
    u_vec = u_vec / torch.sqrt(u_vec.square().sum(dim=(1, 2), keepdim=True))
    v_vec = v_vec / torch.sqrt(v_vec.square().sum(dim=(1, 2), keepdim=True))

    ur, ui = u_vec[:, 0], u_vec[:, 1]
    vr, vi = v_vec[:, 0], v_vec[:, 1]
    rho_u_r = ur.unsqueeze(-1) * ur.unsqueeze(-2) + ui.unsqueeze(-1) * ui.unsqueeze(-2)
    rho_u_i = ui.unsqueeze(-1) * ur.unsqueeze(-2) - ur.unsqueeze(-1) * ui.unsqueeze(-2)
    rho_v_r = vr.unsqueeze(-1) * vr.unsqueeze(-2) + vi.unsqueeze(-1) * vi.unsqueeze(-2)
    rho_v_i = vi.unsqueeze(-1) * vr.unsqueeze(-2) - vr.unsqueeze(-1) * vi.unsqueeze(-2)

    lhs_r = rho_u_r * rho_v_r - rho_u_i * rho_v_i
    lhs_i = rho_u_r * rho_v_i + rho_u_i * rho_v_r
    uv_r, uv_i = ur * vr - ui * vi, ur * vi + ui * vr
    rhs_r = uv_r.unsqueeze(-1) * uv_r.unsqueeze(-2) + uv_i.unsqueeze(-1) * uv_i.unsqueeze(-2)
    rhs_i = uv_i.unsqueeze(-1) * uv_r.unsqueeze(-2) - uv_r.unsqueeze(-1) * uv_i.unsqueeze(-2)
    pure_schur_identity_err = float(torch.sqrt((lhs_r - rhs_r).square().mean() + (lhs_i - rhs_i).square().mean()).item())

    bsz, seq_t = 2, 32
    probs = torch.softmax(torch.randn(bsz, n_packets, seq_t, seq_t, device=dev), dim=-1)
    v_seq = torch.randn(bsz, seq_t, n_packets, 2, d_state, device=dev)
    v_seq = v_seq / torch.sqrt(v_seq.square().sum(dim=(3, 4), keepdim=True))
    bar_r_packed, bar_i_packed = gather_pure_density_matrices_packed(probs, v_seq[:, :, :, 0], v_seq[:, :, :, 1])

    vr_h, vi_h = v_seq[:, :, :, 0].permute(0, 2, 1, 3), v_seq[:, :, :, 1].permute(0, 2, 1, 3)
    explicit_r = torch.einsum("bhts,bhsc,bhsd->bhtcd", probs, vr_h, vr_h) + torch.einsum("bhts,bhsc,bhsd->bhtcd", probs, vi_h, vi_h)
    explicit_i = torch.einsum("bhts,bhsc,bhsd->bhtcd", probs, vi_h, vr_h) - torch.einsum("bhts,bhsc,bhsd->bhtcd", probs, vr_h, vi_h)
    packed_gemm_err = float(
        torch.sqrt((bar_r_packed - explicit_r).square().mean() + (bar_i_packed - explicit_i).square().mean()).item()
    )

    psi0 = torch.randn(bsz, seq_t, n_packets, 2, d_state, device=dev)
    psi0 = psi0 / torch.sqrt(psi0.square().sum(dim=(3, 4), keepdim=True))
    p0_r, p0_i = psi0[:, :, :, 0], psi0[:, :, :, 1]
    rho0_r = p0_r.unsqueeze(-1) * p0_r.unsqueeze(-2) + p0_i.unsqueeze(-1) * p0_i.unsqueeze(-2)
    rho0_i = p0_i.unsqueeze(-1) * p0_r.unsqueeze(-2) - p0_r.unsqueeze(-1) * p0_i.unsqueeze(-2)
    rho0 = torch.stack([rho0_r, rho0_i], dim=3)

    results_by_mode: dict[str, list[dict[str, float]]] = {}
    for mode_name, enable_cooling, enable_mcw in [
        ("convex_only_no_schur", False, False),
        ("schur_only_highway", True, False),
        ("schur_plus_mcweeny_highway", True, True),
    ]:
        torch.manual_seed(2026)
        blocks = [
            DensityMatrixHighwayBlock(
                n_packets=n_packets,
                d_state=d_state,
                d_probe=DEFAULT_PROBE_RANK_R,
                enable_schur_cooling=enable_cooling,
                enable_mcweeny_purification=enable_mcw,
            ).to(dev)
            for _ in range(num_layers)
        ]
        rho_curr = rho0.clone()
        layer_stats: list[dict[str, float]] = []
        with torch.no_grad():
            for l_idx, blk in enumerate(blocks, start=1):
                rho_curr, coinc = blk(rho_curr)
                rr, ri = rho_curr[:, :, :, 0], rho_curr[:, :, :, 1]
                purity = (rr.square().sum(dim=(-2, -1)) + ri.square().sum(dim=(-2, -1))).mean().item()
                diag_pur = float(d_state * torch.diagonal(rr, dim1=-2, dim2=-1).square().sum(dim=-1).mean().item())
                eff_rank = 1.0 / max(purity, DEFAULT_SCHUR_EPS)
                herm_err = float(
                    torch.sqrt((rr - rr.transpose(-2, -1)).square().mean() + (ri + ri.transpose(-2, -1)).square().mean()).item()
                )
                trace_err = float((torch.diagonal(rr, dim1=-2, dim2=-1).sum(dim=-1) - 1.0).abs().max().item())
                min_eig = float(torch.linalg.eigvalsh(torch.complex(rr[0, -1, 0], ri[0, -1, 0])).min().item())
                layer_stats.append(
                    asdict(
                        CoolingLayerMetrics(
                            layer_idx=l_idx,
                            mean_purity_tr_rho2=round(purity, 4),
                            diag_coincidence_d_tr_schur=round(diag_pur, 4),
                            effective_rank=round(eff_rank, 2),
                            hermitian_err=round(herm_err, 8),
                            trace_err=round(trace_err, 8),
                            min_eigenvalue=round(min_eig, 6),
                            mean_schur_coincidence=round(coinc, 4),
                        )
                    )
                )
        results_by_mode[mode_name] = layer_stats

    return {
        "pure_schur_identity_err": pure_schur_identity_err,
        "packed_gemm_equivalence_err": packed_gemm_err,
        "cooling_vs_thermal_death": results_by_mode,
    }


def run_born_schur_reversal_benchmark(
    steps: int = DEFAULT_REVERSAL_STEPS,
    eval_interval: int = DEFAULT_REVERSAL_EVAL_INTERVAL,
    device: str = "cuda",
    variant_filter: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Compare McWeeny Matmul vs. Born-Schur Density Attention vs. Full Density Highway on `50,304`-vocab reversal."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    dataset = FineWebSpanReversalDataset(vocab_size=50304, device=dev)
    eff_spec = SupportSketchSpec(
        mode=SupportSketchMode.I11_COMPLEX_WAVEFUNCTION_8x2x32_ID2, vocab_size=50304, pkt_count=16, pkt_dim=32, seed=1337
    )

    variants = [
        ("D0_MCWEENY_MATMUL_BASELINE", DensitySchurMode.MCWEENY_MATMUL_BASELINE, "Within-head McWeeny matmul (rho_h^2)"),
        ("D1_BORN_SCHUR_STATIC_ATTN", DensitySchurMode.BORN_SCHUR_STATIC_WEIGHTS, "Static squared-amplitude linear + Schur products"),
        ("D2_BORN_SCHUR_DYNAMIC_POVM", DensitySchurMode.BORN_SCHUR_DYNAMIC_POVM, "Dynamic Born-Trace POVM gates + Schur coincidence"),
        ("D3_FULL_DENSITY_MATRIX_HIGHWAY", DensitySchurMode.FULL_DENSITY_MATRIX_HIGHWAY, "Commuted Unitary Conjugation + Born-Trace Highway Gate + Schur"),
    ]

    results: list[dict[str, Any]] = []
    for var_id, mode, desc in variants:
        if variant_filter is not None and var_id not in variant_filter:
            continue
        cfg = MicroReversalConfig(
            backbone_mode=MicroBackboneMode.QUANTUM_WAVEPACKET,
            init_mode=MicroInitMode.GPU_UNIGRAM_PRIOR,
            readout_mode=MicroReadoutMode.R4_HERMITIAN_QUADRATIC_BORN,
            vocab_size=50304, span_len=dataset.span_len, embed_dim=512,
            num_heads=8, num_layers=DEFAULT_MICRO_NUM_LAYERS,
            mlp_hidden=1536, pkt_count=16, pkt_dim=32, seed=1337,
        )
        model = MicroReversalGPT(cfg, seq_len=dataset.seq_len).to(dev)
        if mode in (DensitySchurMode.BORN_SCHUR_STATIC_WEIGHTS, DensitySchurMode.BORN_SCHUR_DYNAMIC_POVM):
            use_dyn = mode == DensitySchurMode.BORN_SCHUR_DYNAMIC_POVM
            model.quantum_blocks = nn.ModuleList(
                [_BornSchurMicroReversalBlock(512, 8, 16, 32, 1536, use_dynamic_povm=use_dyn).to(dev) for _ in range(DEFAULT_MICRO_NUM_LAYERS)]
            )
        elif mode == DensitySchurMode.FULL_DENSITY_MATRIX_HIGHWAY:
            model.quantum_blocks = nn.ModuleList(
                [_DensityHighwayReversalAdapterBlock(512, 8, 16, 32, 1536).to(dev) for _ in range(DEFAULT_MICRO_NUM_LAYERS)]
            )

        apply_support_sketch_initialization_(model, dataset, eff_spec)

        stiefel_params: list[nn.Parameter] = list(model.readout.real_stiefel_parameters())
        unitary_pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
        for q_blk in model.quantum_blocks:
            stiefel_params.extend(q_blk.real_stiefel_parameters())  # type: ignore[operator]
            unitary_pairs.extend(q_blk.complex_unitary_pairs())  # type: ignore[operator]

        q_trunk_opt = ModularRiemannianStiefelKrausMuon(
            real_stiefel_params=stiefel_params, complex_operator_pairs=unitary_pairs,
            stiefel_lr=DEFAULT_MICRO_STIEFEL_LR, unitary_lr=DEFAULT_MICRO_UNITARY_LR,
            tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA, momentum_mode=StiefelMomentumMode.RIEMANNIAN_TANGENT_SNR_GATED,
        )
        wte_opt = BatchSubspaceTikhonovGramMuon(
            [model.wte_3d] + model.readout.sphere_parameters(), lr=DEFAULT_MICRO_WTE_LR, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA
        )
        scalar_params: list[nn.Parameter] = [model.wpe_3d, model.pos_scale, model.wp_causal_re, model.wp_causal_im]
        scalar_params.extend(model.readout.scalar_parameters())
        for q_blk in model.quantum_blocks:
            scalar_params.extend(q_blk.scalar_parameters())  # type: ignore[operator]
        scalar_opt = torch.optim.AdamW([{"params": scalar_params, "lr": DEFAULT_MICRO_SCALAR_LR, "weight_decay": 0.0}], betas=(0.9, 0.95), eps=1e-8)

        s0 = evaluate_micro_reversal_model(model, dataset)
        s0["step"] = 0
        curve = [s0]
        print(f"[{var_id}] Step   0 | Val CE: {s0['val_ce']:.4f} | Top1: {100.0 * s0['top1_acc']:.2f}% | Top5: {100.0 * s0['top5_acc']:.2f}%", flush=True)
        t0 = time.perf_counter()
        model.train()
        for step in range(1, steps + 1):
            warm_steps = max(1, int(0.10 * steps))
            lr_mult = (
                0.15 + 0.85 * (float(step - 1) / float(max(1, warm_steps - 1)))
                if step <= warm_steps
                else 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * float(step - warm_steps) / float(max(1, steps - warm_steps))))
            )
            mom_beta = compute_riemannian_momentum_beta(step, steps, QuantumMomentumScheduleMode.LOW_STEADY_MOMENTUM_070)
            q_trunk_opt.zero_grad(set_to_none=True)
            wte_opt.zero_grad(set_to_none=True)
            scalar_opt.zero_grad(set_to_none=True)

            idx_in, targets = dataset.sample_train_batch(step, batch_size=DEFAULT_MICRO_BATCH_SIZE)
            _, loss = model(idx_in, targets)
            assert loss is not None
            loss.backward()

            q_trunk_opt.set_momentum(mom_beta)
            q_trunk_opt.set_lr_scale(lr_mult)
            q_trunk_opt.step()

            active_ids = torch.unique(torch.cat([idx_in.reshape(-1), targets.reshape(-1)]))
            wte_opt.set_momentum(mom_beta)
            wte_opt.step_active(active_ids, lr_scale=lr_mult)

            scalar_opt.param_groups[0]["lr"] = DEFAULT_MICRO_SCALAR_LR * lr_mult
            scalar_opt.step()

            project_complex_wavepackets_to_rms_sphere_(model.wpe_3d.view(model.seq_len, model.pkt_count // 2, 2, model.pkt_dim))
            model.restrict_all_scalars_()

            if step % eval_interval == 0 or step == steps:
                ev = evaluate_micro_reversal_model(model, dataset)
                ev["step"] = step
                curve.append(ev)
                print(f"[{var_id}] Step {step:3d} | Val CE: {ev['val_ce']:.4f} | Top1: {100.0 * ev['top1_acc']:.2f}% | Top5: {100.0 * ev['top5_acc']:.2f}%", flush=True)

        elapsed = time.perf_counter() - t0
        results.append(
            {
                "variant_id": var_id, "mode": mode.value, "description": desc,
                "step0_val_ce": round(curve[0]["val_ce"], 4), "final_val_ce": round(curve[-1]["val_ce"], 4),
                "final_top1_acc": round(100.0 * curve[-1]["top1_acc"], 2), "final_top5_acc": round(100.0 * curve[-1]["top5_acc"], 2),
                "elapsed_sec": round(elapsed, 2), "curve": curve,
            }
        )
        del model, q_trunk_opt, wte_opt, scalar_opt
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    return results


def run_born_schur_fineweb_benchmark(
    steps: int = DEFAULT_FINEWEB_STEPS,
    eval_interval: int = DEFAULT_FINEWEB_EVAL_INTERVAL,
    device: str = "cuda",
) -> dict[str, Any]:
    """Benchmark `BornSchurMultiBranchWavepacketGPT` (`D4`) on FineWeb (`L=4, d=512, T=512`)."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    stream = LongSeqFineWebStream(device=dev)
    spec = MultiBranchReadoutSpec(
        mode=MultiBranchMode.M1_FIBER_STIEFEL_FRAME_R4, seed=1337
    )
    torch.manual_seed(int(spec.seed))
    model = BornSchurMultiBranchWavepacketGPT(spec).to(dev)
    saved_mb_head = model.mb_head
    model.mb_head = None
    init_ms, _ = initialize_and_calibrate_shallow_model_(model, stream)
    model.mb_head = saved_mb_head
    if model.mb_head is not None:
        t_res0 = time.perf_counter()
        calibrate_multibranch_residual_step0_(model, stream)
        init_ms += (time.perf_counter() - t_res0) * 1000.0
    model.clamp_identity_dispersion_and_scalars_()

    stiefel_params: list[nn.Parameter] = list(model.exit_bridge.real_stiefel_parameters())
    unitary_pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
    for q_blk in model.quantum_blocks:
        stiefel_params.extend(q_blk.real_stiefel_parameters())  # type: ignore[operator]
        unitary_pairs.extend(q_blk.complex_unitary_pairs())  # type: ignore[operator]

    fast_temp_params: list[nn.Parameter] = [
        model.temp_sem_scale, model.temp_id_scale, model.trunk_sem_gate, model.trunk_id_gate,
    ]
    scalar_params: list[nn.Parameter] = [
        model.wp_causal_re, model.wp_causal_im, model.smear_gate.weight,
        model.smear_lambda, model.radial_scale, model.unigram_bias, model.prior_gate,
    ]
    scalar_params.extend(model.exit_bridge.scalar_parameters())
    for q_blk in model.quantum_blocks:
        scalar_params.extend(q_blk.scalar_parameters())  # type: ignore[operator]

    mb_stiefel_params: list[nn.Parameter] = []
    if model.mb_head is not None:
        stiefel_params.extend(model.mb_head.real_stiefel_parameters())
        scalar_params.extend(model.mb_head.scalar_parameters())
        mb_stiefel_params.append(model.mb_head.id_branch_stiefel)

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
        adam_groups.append({"params": ngram_tables, "lr": DEFAULT_SHALLOW_NGRAM_LR, "weight_decay": 0.0})
        adam_groups.append({"params": ngram_scalars, "lr": DEFAULT_SHALLOW_SCALAR_LR, "weight_decay": 0.0})
    if mb_stiefel_params:
        adam_groups.append({"params": mb_stiefel_params, "lr": DEFAULT_SHALLOW_STIEFEL_LR, "weight_decay": 0.0})
    adam_opt = torch.optim.AdamW(adam_groups, betas=(0.80, 0.95), eps=1e-8, fused=(dev.type == "cuda"))

    s0 = evaluate_multibranch_model(model, stream)
    s0["step"] = 0
    curve = [s0]
    print(
        f"[D4_BORN_SCHUR_CONE_R4] Step   0 | Val CE: {s0['val_ce']:.4f} (Neural={s0['neural_val_ce']:.4f}) (PPL {s0['val_ppl']:.2f}) | "
        f"Top1: {100.0 * s0['top1_acc']:.2f}% | Top5: {100.0 * s0['top5_acc']:.2f}% | Branches: {s0['effective_active_branches']:.2f}",
        flush=True,
    )

    t0 = time.perf_counter()
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
        if len(adam_opt.param_groups) > 4:
            adam_opt.param_groups[4]["lr"] = DEFAULT_SHALLOW_STIEFEL_LR * lr_mult
        adam_opt.step()

        model.clamp_identity_dispersion_and_scalars_()

        if step % eval_interval == 0 or step == steps:
            ev = evaluate_multibranch_model(model, stream)
            ev["step"] = step
            ev["train_loss"] = round(float(loss.detach().item()), 4)
            curve.append(ev)
            print(
                f"[D4_BORN_SCHUR_CONE_R4] Step {step:3d} | Train CE: {ev['train_loss']:.4f} | Val CE: {ev['val_ce']:.4f} "
                f"(Neural={ev['neural_val_ce']:.4f}) (PPL {ev['val_ppl']:.2f}) | "
                f"Top1: {100.0 * ev['top1_acc']:.2f}% | Top5: {100.0 * ev['top5_acc']:.2f}% | Branches: {ev['effective_active_branches']:.2f}",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    return {
        "variant_id": "D4_BORN_SCHUR_POVM_CONE_R4",
        "init_ms": round(init_ms, 2),
        "elapsed_sec": round(elapsed, 2),
        "step0": curve[0],
        "final": curve[-1],
        "curve": curve,
    }
