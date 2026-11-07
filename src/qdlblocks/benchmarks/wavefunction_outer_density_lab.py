r"""Wavefunction Outer-Product Input Density & Cone-Free Native Cholesky Density Readout Laboratory.

Mathematical & Architectural Motivation
---------------------------------------
1. **Input Wavefunction -> Outer-Product Density Construction (`W0..W3`)**:
   - When each vocabulary item $v \in \{0, \dots, V-1\}$ is represented as a pure multi-packet wavefunction
     $|\psi_v\rangle = (|\psi_{v,0}\rangle, \dots, |\psi_{v,7}\rangle) \in (\mathbb{C}^{32})^8$ (`(50304, 8, 2, 32)`),
     adding local context wavefunctions as vectors ($|\psi_{x_t}\rangle + |\psi_{x_{t-1}}\rangle + |\psi_{\text{pos}, t}\rangle$)
     suffers from arbitrary global phase collisions: multiplying $|\psi_{x_{t-1}}\rangle$ by $e^{i\theta}$ flips
     constructive reinforcement into destructive cancellation.
   - By contrast, lifting each constituent wavefunction to its rank-1 outer-product density matrix first:
     $$\rho_{t,h}^{\text{curr}} = |\psi_{x_t, h}\rangle\langle\psi_{x_t, h}|, \qquad
       \rho_{t,h}^{\text{prev}} = U_{\Delta=1, h}\, |\psi_{x_{t-1}, h}\rangle\langle\psi_{x_{t-1}, h}|\, U_{\Delta=1, h}^\dagger$$
     and forming the initial state via **Incoherent Density Mixture + Entrywise Schur Coincidence**:
     $$\rho_{t,h}^{(0)} = (1 - \alpha_{t,h})\, \rho_{t,h}^{\text{curr}}
       + \alpha_{t,h}\, \rho_{t,h}^{\text{prev}}
       + \gamma_{t,h}\, D \left(\rho_{t,h}^{\text{curr}} \odot_{\mathbb{C}} \rho_{t,h}^{\text{prev}}\right)$$
     where $\alpha_{t,h}, \gamma_{t,h} \in [0, 1]$ are Born-Trace POVM measurements
     $\langle m_{\text{in}, h} \mid \rho_{t,h}^{\text{curr}} \mid m_{\text{in}, h}\rangle = |\langle m_{\text{in}, h} \mid \psi_{x_t, h}\rangle|^2$,
     is **100% invariant to global phase gauge noise** on every neighbor token!
   - Furthermore, projecting $\rho_{t,h}^{(0)}$ onto the reference ket $|\psi_{x_t, h}\rangle$ admits an exact
     $O(D)$ closed-form operator-ket expression that never materializes a $D \times D$ matrix:
     $$\rho_{t,h}^{(0)} |\psi_{x_t, h}\rangle
       = (1 - \alpha_{t,h})\, |\psi_{x_t, h}\rangle
       + \alpha_{t,h}\, |\tilde{\psi}_{x_{t-1}, h}\rangle \langle\tilde{\psi}_{x_{t-1}, h} \mid \psi_{x_t, h}\rangle
       + \gamma_{t,h}\, D \left(|\tilde{\psi}_{x_{t-1}, h}|^2 \odot |\psi_{x_t, h}\rangle\right).$$

2. **Eliminating the `D4` Rank-4 -> Rank-1 -> Rank-4 Bottleneck (`D5` & `D6` Cone-Free Native Cholesky Readout)**:
   - In `D4_BORN_SCHUR_POVM_CONE_R4`, each attention layer already computed a rank-$k$ ($k=4$) Cholesky factor
     $L_{t,h} \in \mathbb{C}^{4 \times 32}$ ($\rho_{t,h} = L_{t,h}^\dagger L_{t,h}$), collapsed it to a rank-1 ket,
     and then used a separate `mb_head` (`id_branch_stiefel` $3 \times 64 \times 512$, `branch_gate_proj`,
     Gram-Schmidt, and a fixed $30^\circ$ geodesic cone) to recreate 4 readout branches.
   - Because any rank-$k$ Cholesky factor $L_{t,7}^{(L)} \in \mathbb{C}^{4 \times 32}$ on Identity Packet 7
     is already a spectral sum of $k=4$ outer products:
     $$\rho_{t,7}^{(L)} = \sum_{r=0}^{3} |l_{t,7,r}^{(L)}\rangle\langle l_{t,7,r}^{(L)}|
       = \sum_{r=0}^{3} \pi_{t,r}\, |u_{t,7,r}^{(L)}\rangle\langle u_{t,7,r}^{(L)}|, \qquad
       \pi_{t,r} = \frac{\|l_{t,7,r}^{(L)}\|_2^2}{\sum_{r'=0}^{3} \|l_{t,7,r'}^{(L)}\|_2^2 + \varepsilon},$$
     we can **delete `mb_head` (`-99K` params), delete the $30^\circ$ cone heuristic, and delete Step-0 SVD
     residual calibration**, reading out the 4 Cholesky eigen-modes and their endogenous trace weights $\pi_{t,r}$
     directly against the pure vocabulary wavefunction outer products $|e_v\rangle\langle e_v|$!
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.fused_quantum_wavepacket_attn import (
    _FusedHermitianBornRhoKetAttnFn,
    _get_compiled_chol_gram_fn,
    associative_low_rank_cholesky_gram_modulation,
    fused_low_rank_cholesky_ket_attention,
)
from ..layers.impure_density_mirror_block import DEFAULT_TIKHONOV_GRAM_LAMBDA, TRACE_NORM_EPS
from ..layers.quantum_statmech_channels import (
    QuantumMomentumScheduleMode,
    StiefelMomentumMode,
    compute_riemannian_momentum_beta,
)
from ..layers.quantum_svd_stiefel_graft import apply_fubini_study_self_ortho
from ..models.compiled_leaderboard_classical_gpt import apply_native_leaderboard_rotary
from ..models.modular_quantum_statmech_gpt import ModularRiemannianStiefelKrausMuon
from ..optim.vocab_manifold_muon import BatchSubspaceTikhonovGramMuon
from .born_schur_density_lab import (
    DEFAULT_FINEWEB_EVAL_INTERVAL,
    DEFAULT_FINEWEB_STEPS,
    DEFAULT_PAIR_NOISE_STD,
    DEFAULT_REVERSAL_EVAL_INTERVAL,
    DEFAULT_REVERSAL_STEPS,
    DEFAULT_SCHUR_EPS,
    _DensityHighwayReversalAdapterBlock,
)
from .micro_reversal_readout_lab import (
    DEFAULT_MICRO_BATCH_SIZE,
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
    evaluate_micro_reversal_model,
)
from .multibranch_wavepacket_readout_lab import (
    MultiBranchMode,
    MultiBranchReadoutSpec,
    MultiBranchWavepacketGPT,
    _FactoredMultiBranchWavepacketCEFn,
    evaluate_multibranch_model,
)
from .shallow_longseq_wavepacket_lm_lab import (
    DEFAULT_CARRIER_BIAS_DIM,
    DEFAULT_SHALLOW_EMBED_DIM,
    DEFAULT_SHALLOW_MLP_HIDDEN,
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
)
from .support_preserving_countsketch_lab import (
    SupportSketchMode,
    SupportSketchSpec,
    apply_support_sketch_initialization_,
    project_complex_wavepackets_to_rms_sphere_,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
DEFAULT_INIT_SCHUR_GATE_SCALE: float = 0.15
DEFAULT_CHOLESKY_ANCHOR_LOGIT_BIAS: float = 1.45
DEFAULT_CHOLESKY_MODE_COHERENCE_MIX: float = 0.78
DEFAULT_INPUT_OUTER_MIX_INIT: float = 0.22
DEFAULT_INPUT_SCHUR_MIX_INIT: float = 0.12
DEFAULT_LONG_FINEWEB_STEPS: int = 1200
DEFAULT_LONG_FINEWEB_EVAL_INTERVAL: int = 150


class WavefunctionInputMode(str, Enum):
    """Strategy enum selecting how token wavefunctions form the initial density state at Layer 0 (Rule 3)."""

    W0_VECTOR_SUM_THEN_OUTER = "w0_vector_sum_then_outer"
    W1_OUTER_PHASE_ALIGNED_MIXTURE = "w1_outer_phase_aligned_mixture"
    W2_OUTER_PLUS_BIGRAM_SCHUR_INPUT = "w2_outer_plus_bigram_schur_input"
    W3_UNITARY_POS_OUTER_DENSITY_INPUT = "w3_unitary_pos_outer_density_input"


@dataclass(frozen=True)
class NativeCholeskyDensitySpec:
    """Configuration for Cone-Free Native & Uncollapsed Cholesky Density models (`D5..D9`) (`<= 8` fields)."""

    use_outer_product_input: bool = False
    uncollapsed_cholesky_highway: bool = False
    use_semantic_cholesky_readout: bool = False
    enable_kn_synergy_eval: bool = False
    init_schur_gate_scale: float = DEFAULT_INIT_SCHUR_GATE_SCALE
    anchor_logit_bias: float = DEFAULT_CHOLESKY_ANCHOR_LOGIT_BIAS
    mode_coherence_mix: float = DEFAULT_CHOLESKY_MODE_COHERENCE_MIX
    seed: int = 1337


def apply_phase_invariant_outer_density_input_lift(
    curr_wp: torch.Tensor,
    prev_wp: torch.Tensor,
    m_mix_re: torch.Tensor,
    m_mix_im: torch.Tensor,
    m_schur_re: torch.Tensor,
    m_schur_im: torch.Tensor,
    scales: tuple[torch.Tensor, torch.Tensor, bool],
) -> torch.Tensor:
    r"""Apply input outer-product density operator $\rho_{t,h}^{(0)}$ to $|\psi_{x_t, h}\rangle$ in $O(D)$ ops (`<= 8` args).

    Given current wavepacket $|\psi_t\rangle \in \mathbb{C}^D$ and phase-transported previous wavepacket
    $|\tilde{\psi}_{t-1}\rangle \in \mathbb{C}^D$ (both of shape `(B, T, H, 2, D)`):
      1. Pure outer-product projection:
         $$\left(|\tilde{\psi}_{t-1}\rangle\langle\tilde{\psi}_{t-1}|\right)|\psi_t\rangle
           = |\tilde{\psi}_{t-1}\rangle \langle\tilde{\psi}_{t-1} \mid \psi_t\rangle$$
         which cancels any global phase $e^{i\theta}$ on $|\tilde{\psi}_{t-1}\rangle$ via $e^{i\theta}e^{-i\theta} = 1$.
      2. Entrywise Schur outer-product coincidence:
         $$D \left(|\psi_t\rangle\langle\psi_t| \odot_{\mathbb{C}} |\tilde{\psi}_{t-1}\rangle\langle\tilde{\psi}_{t-1}|\right)|\psi_t\rangle
           \propto D \left(|\tilde{\psi}_{t-1}|^2 \odot |\psi_t\rangle\right)$$
         which modulates each basis coordinate by the phase-invariant Born probability density $|\tilde{\psi}_{t-1, d}|^2$.
      3. Both terms are gated by physical Born-Trace POVM measurements $|\langle m_h \mid \psi_{t,h}\rangle|^2 \in [0, 1]$.
    """
    mix_scale, schur_scale, enable_input_schur = scales
    work_dtype = curr_wp.dtype
    d_state = curr_wp.size(-1)

    cr, ci = curr_wp[..., 0, :].float(), curr_wp[..., 1, :].float()
    pr, pi = prev_wp[..., 0, :].float(), prev_wp[..., 1, :].float()

    # Unit-normalize kets per packet so outer products have exact unit trace Tr(|psi><psi|) = 1
    c_norm = torch.sqrt(cr.square().sum(dim=-1, keepdim=True) + ci.square().sum(dim=-1, keepdim=True) + DEFAULT_SCHUR_EPS)
    p_norm = torch.sqrt(pr.square().sum(dim=-1, keepdim=True) + pi.square().sum(dim=-1, keepdim=True) + DEFAULT_SCHUR_EPS)
    uc_r, uc_i = cr / c_norm, ci / c_norm
    up_r, up_i = pr / p_norm, pi / p_norm

    # Born-Trace POVM gate alpha_{t,h} = |<m_mix | u_c>|^2 in [0, 1]
    mm_norm = torch.sqrt(m_mix_re.float().square().sum(dim=-1, keepdim=True) + m_mix_im.float().square().sum(dim=-1, keepdim=True) + DEFAULT_SCHUR_EPS)
    mmr, mmi = (m_mix_re.float() / mm_norm).unsqueeze(0).unsqueeze(0), (m_mix_im.float() / mm_norm).unsqueeze(0).unsqueeze(0)
    ov_m_r = (uc_r * mmr + uc_i * mmi).sum(dim=-1, keepdim=True)
    ov_m_i = (uc_i * mmr - uc_r * mmi).sum(dim=-1, keepdim=True)
    g_mix = torch.sigmoid(mix_scale.float()).view(1, 1, -1, 1) * (ov_m_r.square() + ov_m_i.square()).clamp(0.0, 1.0)

    # Phase-aligned outer-product projection (|u_p><u_p|) |u_c> = |u_p> <u_p | u_c>
    ip_r = (up_r * uc_r + up_i * uc_i).sum(dim=-1, keepdim=True)
    ip_i = (up_r * uc_i - up_i * uc_r).sum(dim=-1, keepdim=True)
    proj_r = up_r * ip_r - up_i * ip_i
    proj_i = up_r * ip_i + up_i * ip_r

    out_r = (1.0 - g_mix) * uc_r + g_mix * proj_r
    out_i = (1.0 - g_mix) * uc_i + g_mix * proj_i

    if enable_input_schur:
        ms_norm = torch.sqrt(m_schur_re.float().square().sum(dim=-1, keepdim=True) + m_schur_im.float().square().sum(dim=-1, keepdim=True) + DEFAULT_SCHUR_EPS)
        msr, msi = (m_schur_re.float() / ms_norm).unsqueeze(0).unsqueeze(0), (m_schur_im.float() / ms_norm).unsqueeze(0).unsqueeze(0)
        ov_s_r = (uc_r * msr + uc_i * msi).sum(dim=-1, keepdim=True)
        ov_s_i = (uc_i * msr - uc_r * msi).sum(dim=-1, keepdim=True)
        g_schur = torch.sigmoid(schur_scale.float()).view(1, 1, -1, 1) * (ov_s_r.square() + ov_s_i.square()).clamp(0.0, 1.0)

        # Diagonal probability density of previous wavepacket |u_p|^2 modulating |u_c>
        prob_prev = float(d_state) * (up_r.square() + up_i.square())
        out_r = out_r + g_schur * (prob_prev * uc_r)
        out_i = out_i + g_schur * (prob_prev * uc_i)

    # Restore original RMS scale
    out_stack = torch.stack([out_r * c_norm, out_i * c_norm], dim=-2)
    return out_stack.to(work_dtype)


class WavefunctionOuterInputReversalGPT(MicroReversalGPT):
    """`MicroReversalGPT` with configurable Wavefunction -> Outer-Product Density Input (`W0..W3`)."""

    def __init__(self, cfg: MicroReversalConfig, seq_len: int, input_mode: WavefunctionInputMode) -> None:
        super().__init__(cfg, seq_len=seq_len)
        self.input_mode = WavefunctionInputMode(input_mode)
        n_complex = self.pkt_count // 2
        self.quantum_blocks = nn.ModuleList(
            [_DensityHighwayReversalAdapterBlock(512, 8, 16, 32, 1536) for _ in range(int(cfg.num_layers))]
        )
        self.in_m_mix_re = nn.Parameter(torch.randn(n_complex, self.pkt_dim) * 0.25)
        self.in_m_mix_im = nn.Parameter(torch.randn(n_complex, self.pkt_dim) * 0.25)
        self.in_m_schur_re = nn.Parameter(torch.randn(n_complex, self.pkt_dim) * 0.25)
        self.in_m_schur_im = nn.Parameter(torch.randn(n_complex, self.pkt_dim) * 0.25)
        self.in_mix_logit = nn.Parameter(torch.full((n_complex,), -1.20, dtype=torch.float32))
        self.in_schur_logit = nn.Parameter(torch.full((n_complex,), -1.80, dtype=torch.float32))

    def extra_input_scalar_parameters(self) -> list[nn.Parameter]:
        """Return input Born-Trace POVM and outer-product gate parameters."""
        return [
            self.in_m_mix_re,
            self.in_m_mix_im,
            self.in_m_schur_re,
            self.in_m_schur_im,
            self.in_mix_logit,
            self.in_schur_logit,
        ]

    def forward(self, idx_in: torch.Tensor, targets_rev: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, seq_len = idx_in.shape
        work_dtype = torch.bfloat16 if (idx_in.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
        n_complex = self.pkt_count // 2

        tok_wp = F.embedding(idx_in, self.wte_3d.view(self.vocab_size, self.embed_dim)).view(
            bsz, seq_len, n_complex, 2, self.pkt_dim
        )
        pos_wp = self.wpe_3d[:seq_len].view(1, seq_len, n_complex, 2, self.pkt_dim)

        if self.input_mode == WavefunctionInputMode.W0_VECTOR_SUM_THEN_OUTER:
            return super().forward(idx_in, targets_rev)

        if self.input_mode == WavefunctionInputMode.W3_UNITARY_POS_OUTER_DENSITY_INPUT:
            # Unitary Positional Phase Conjugation U_pos(t) |psi_x> + outer-product pos alignment
            p_re, p_im = pos_wp[..., 0, :].float(), pos_wp[..., 1, :].float()
            p_mag = torch.sqrt(p_re.square() + p_im.square() + DEFAULT_SCHUR_EPS)
            cos_p, sin_p = p_re / p_mag, p_im / p_mag
            t_re, t_im = tok_wp[..., 0, :].float(), tok_wp[..., 1, :].float()
            rot_re = t_re * cos_p - t_im * sin_p + self.pos_scale * p_re
            rot_im = t_re * sin_p + t_im * cos_p + self.pos_scale * p_im
            base_wp = torch.stack([rot_re, rot_im], dim=-2).to(work_dtype)
        else:
            base_wp = (tok_wp + self.pos_scale * pos_wp).to(work_dtype)

        if seq_len > 1:
            # Construct phase-shifted neighbor wavepacket and apply Outer-Product Density Lift
            disp_wp = self._transport_prev_wavepacket(base_wp)
            prev_wp = torch.cat([base_wp[:, :1], disp_wp[:, :-1]], dim=1)
            use_schur = self.input_mode in (
                WavefunctionInputMode.W2_OUTER_PLUS_BIGRAM_SCHUR_INPUT,
                WavefunctionInputMode.W3_UNITARY_POS_OUTER_DENSITY_INPUT,
            )
            lifted_wp = apply_phase_invariant_outer_density_input_lift(
                base_wp,
                prev_wp,
                self.in_m_mix_re,
                self.in_m_mix_im,
                self.in_m_schur_re,
                self.in_m_schur_im,
                (self.in_mix_logit, self.in_schur_logit, use_schur),
            )
        else:
            lifted_wp = base_wp

        x = F.rms_norm(lifted_wp.reshape(bsz, seq_len, self.embed_dim), (self.embed_dim,), eps=1e-6)
        cos_4d, sin_4d = self.rope.get_cos_sin(seq_len, dtype=work_dtype)
        for q_blk in self.quantum_blocks:
            x = q_blk(x, cos_4d, sin_4d)

        x_final = F.rms_norm(x, (self.embed_dim,), eps=1e-6)
        span_len = self.cfg.span_len
        h_rev = x_final[:, span_len : 2 * span_len, :].contiguous()
        logits_rev = self.readout(h_rev)

        loss: torch.Tensor | None = None
        if targets_rev is not None:
            loss = F.cross_entropy(logits_rev.reshape(-1, self.vocab_size), targets_rev.reshape(-1).long())
        return logits_rev, loss

    def _transport_prev_wavepacket(self, wp_5d: torch.Tensor) -> torch.Tensor:
        """Apply learned complex phase rotation `(wp_causal_re, wp_causal_im)` to previous wavepacket."""
        cr = torch.cos(self.wp_causal_re).view(1, 1, -1, 1).to(wp_5d.dtype)
        si = torch.sin(self.wp_causal_im).view(1, 1, -1, 1).to(wp_5d.dtype)
        re, im = wp_5d[..., 0, :], wp_5d[..., 1, :]
        return torch.stack([re * cr - im * si, re * si + im * cr], dim=-2)


def phase_lock_cholesky_modes(
    ref_re: torch.Tensor,
    ref_im: torch.Tensor,
    cand_re: torch.Tensor,
    cand_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Gauge-rotate each Cholesky row $r$ of `cand` by $e^{i\phi_r} = \frac{\langle \text{cand}_r \mid \text{ref}_r\rangle}{|\langle \text{cand}_r \mid \text{ref}_r\rangle| + \varepsilon}$ (`<= 8` args).

    Because $\sum_r (e^{i\phi_r} c_r)^\dagger (e^{i\phi_r} c_r) = \sum_r c_r^\dagger c_r$, this $U(1)^k$ row gauge
    transformation leaves the density matrix $\rho = L^\dagger L$ strictly invariant while guaranteeing constructive
    overlap $\operatorname{Re}\langle \text{ref}_r \mid \text{cand}_r^{\text{aligned}}\rangle = |\langle \text{ref}_r \mid \text{cand}_r\rangle| \ge 0$.
    """
    ip_re = (cand_re.float() * ref_re.float() + cand_im.float() * ref_im.float()).sum(dim=-1, keepdim=True)
    ip_im = (cand_re.float() * ref_im.float() - cand_im.float() * ref_re.float()).sum(dim=-1, keepdim=True)
    mag = torch.sqrt(ip_re.square() + ip_im.square() + DEFAULT_SCHUR_EPS)
    cos_p = (ip_re / mag).to(cand_re.dtype)
    sin_p = (ip_im / mag).to(cand_re.dtype)
    return cand_re * cos_p - cand_im * sin_p, cand_re * sin_p + cand_im * cos_p


class _NativeCholeskySchurWavepacketBlock(_ShallowQuantumWavepacketBlock):
    r"""Wavepacket Block that evolves and propagates its rank-$k$ ($k=4$) Cholesky density factor $L_{t,h} \in \mathbb{C}^{4 \times 32}$."""

    def __init__(
        self,
        embed_dim: int = DEFAULT_SHALLOW_EMBED_DIM,
        num_heads: int = DEFAULT_SHALLOW_NUM_HEADS,
        pkt_count: int = DEFAULT_SHALLOW_PKT_COUNT,
        pkt_dim: int = DEFAULT_SHALLOW_PKT_DIM,
        mlp_hidden: int = DEFAULT_SHALLOW_MLP_HIDDEN,
        init_schur_scale: float = DEFAULT_INIT_SCHUR_GATE_SCALE,
        uncollapsed_cholesky: bool = False,
    ) -> None:
        super().__init__(embed_dim, num_heads, pkt_count, pkt_dim, mlp_hidden)
        self.uncollapsed_cholesky = bool(uncollapsed_cholesky)
        self.raw_pair_mix = nn.Parameter(
            torch.eye(self.num_heads) + DEFAULT_PAIR_NOISE_STD * torch.randn(self.num_heads, self.num_heads)
        )
        self.m_lin_r = nn.Parameter(torch.randn(self.num_heads, self.d_h) * 0.25)
        self.m_lin_i = nn.Parameter(torch.randn(self.num_heads, self.d_h) * 0.25)
        self.m_schur_r = nn.Parameter(torch.randn(self.num_heads, self.d_h) * 0.25)
        self.m_schur_i = nn.Parameter(torch.randn(self.num_heads, self.d_h) * 0.25)
        self.schur_mix_scale = nn.Parameter(torch.full((self.num_heads,), float(init_schur_scale), dtype=torch.float32))

        # Uncollapsed 4-mode Cholesky inter-layer highway POVM gate parameters (per head and per mode r in {0..3})
        self.m_hwy_r = nn.Parameter(torch.randn(self.num_heads, self.d_h) * 0.25)
        self.m_hwy_i = nn.Parameter(torch.randn(self.num_heads, self.d_h) * 0.25)
        self.chol_hwy_bias = nn.Parameter(torch.full((self.num_heads, 4), 1.65, dtype=torch.float32))
        self.chol_hwy_scale = nn.Parameter(torch.full((self.num_heads, 4), 1.25, dtype=torch.float32))

        self.last_cholesky_pre_u: tuple[torch.Tensor, torch.Tensor] | None = None
        self.last_cholesky_all_re: torch.Tensor | None = None
        self.last_cholesky_all_im: torch.Tensor | None = None
        self.last_cholesky_id_re: torch.Tensor | None = None
        self.last_cholesky_id_im: torch.Tensor | None = None

    def scalar_parameters(self) -> list[nn.Parameter]:
        params = super().scalar_parameters() + [
            self.raw_pair_mix,
            self.m_lin_r,
            self.m_lin_i,
            self.m_schur_r,
            self.m_schur_i,
            self.schur_mix_scale,
        ]
        if self.uncollapsed_cholesky:
            params.extend([self.m_hwy_r, self.m_hwy_i, self.chol_hwy_bias, self.chol_hwy_scale])
        return params

    @torch.no_grad()
    def restrict_scalars_(self) -> None:
        super().restrict_scalars_()
        self.schur_mix_scale.clamp_(0.0, 1.50)
        self.chol_hwy_scale.clamp_(-4.0, 4.0)

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
        s_scale = self.schur_mix_scale.to(work_dtype).view(1, self.num_heads, 1, 1, 1)
        g_schur = s_scale * torch.sqrt((ov_sch_r.float().square() + ov_sch_i.float().square()).sum(dim=-2, keepdim=True).clamp(1e-5, 1.0)).to(work_dtype)

        out_l_r = (g_lin + DEFAULT_SCHUR_EPS) * l_re + g_schur * schur_l_r
        out_l_i = (g_lin + DEFAULT_SCHUR_EPS) * l_im + g_schur * schur_l_i
        frob = torch.sqrt(out_l_r.float().square().sum(dim=(-2, -1), keepdim=True) + out_l_i.float().square().sum(dim=(-2, -1), keepdim=True) + TRACE_NORM_EPS).to(work_dtype)
        return out_l_r / frob, out_l_i / frob

    def _merge_uncollapsed_cholesky_highway(
        self,
        in_r: torch.Tensor,
        in_i: torch.Tensor,
        cur_r: torch.Tensor,
        cur_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge incoming uncollapsed Cholesky factor `(in_r, in_i)` with current `(cur_r, cur_i)` via phase-locked per-mode Born-Trace POVM gate."""
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
        b_h = self.chol_hwy_bias.float().view(1, self.num_heads, 1, 4, 1)
        s_h = self.chol_hwy_scale.float().view(1, self.num_heads, 1, 4, 1)
        alpha = torch.sigmoid(b_h + s_h * p_mode)
        w_old = torch.sqrt((1.0 - alpha).clamp_min(1e-6)).to(work_dtype)
        w_new = torch.sqrt(alpha.clamp_min(1e-6)).to(work_dtype)

        # Lock incoming mode phases to current layer's mode phases (leaves rho^{(l-1)} = L^\dagger L invariant)
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

        if self.training and l_re.requires_grad:
            out_l_r, out_l_i = torch.utils.checkpoint.checkpoint(self._cholesky_schur_step, l_re, l_im, use_reentrant=False)
        else:
            out_l_r, out_l_i = self._cholesky_schur_step(l_re, l_im)

        if self.uncollapsed_cholesky:
            if in_cholesky is not None:
                mod_l_r, mod_l_i = self._merge_uncollapsed_cholesky_highway(in_cholesky[0], in_cholesky[1], out_l_r, out_l_i)
            else:
                mod_l_r, mod_l_i = out_l_r, out_l_i
            self.last_cholesky_pre_u = (mod_l_r, mod_l_i)
            u_r_t = self.u_evol_re.to(work_dtype).view(1, self.num_heads, 1, self.d_h, self.d_h).transpose(-2, -1)
            u_i_t = self.u_evol_im.to(work_dtype).view(1, self.num_heads, 1, self.d_h, self.d_h).transpose(-2, -1)
            self.last_cholesky_all_re = torch.matmul(mod_l_r, u_r_t) - torch.matmul(mod_l_i, u_i_t)
            self.last_cholesky_all_im = torch.matmul(mod_l_r, u_i_t) + torch.matmul(mod_l_i, u_r_t)
            self.last_cholesky_id_re = self.last_cholesky_all_re[:, -1]
            self.last_cholesky_id_im = self.last_cholesky_all_im[:, -1]
        else:
            mod_l_r, mod_l_i = out_l_r, out_l_i
            u7_r = self.u_evol_re[-1].to(work_dtype).view(1, 1, self.d_h, self.d_h)
            u7_i = self.u_evol_im[-1].to(work_dtype).view(1, 1, self.d_h, self.d_h)
            l7_r, l7_i = out_l_r[:, -1], out_l_i[:, -1]
            self.last_cholesky_id_re = torch.matmul(l7_r, u7_r.transpose(-2, -1)) - torch.matmul(l7_i, u7_i.transpose(-2, -1))
            self.last_cholesky_id_im = torch.matmul(l7_r, u7_i.transpose(-2, -1)) + torch.matmul(l7_i, u7_r.transpose(-2, -1))

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


class NativeCholeskyDensityWavepacketGPT(MultiBranchWavepacketGPT):
    r"""Cone-Free Native & Uncollapsed 4-Mode Cholesky Density-Matrix Language Model (`D5..D9`).

    Strips out `mb_head` (`MultiBranchWavepacketReadoutHead` = `None`), strips out `id_branch_stiefel`
    (`3 x 64 x 512`), strips out `branch_gate_proj`, strips out the $30^\circ$ Geodesic Cone heuristic,
    and strips out Step-0 SVD residual calibration. When `uncollapsed_cholesky_highway=True`, all 4 Cholesky
    modes $L_{t,h}^{(\ell)} \in \mathbb{C}^{4 \times 32}$ propagate across all layers $\ell = 1 \dots L$
    without collapsing to a rank-1 ket between blocks.
    """

    def __init__(self, native_spec: NativeCholeskyDensitySpec) -> None:
        mb_mode = (
            MultiBranchMode.M4_COUPLED_R8_CALIBRATED_SYNERGY
            if native_spec.enable_kn_synergy_eval
            else MultiBranchMode.M1_FIBER_STIEFEL_FRAME_R4
        )
        super().__init__(MultiBranchReadoutSpec(mode=mb_mode, seed=native_spec.seed))
        self.native_spec = native_spec
        self.mb_head = None
        self.num_branches = 4

        self.quantum_blocks = nn.ModuleList(
            [
                _NativeCholeskySchurWavepacketBlock(
                    self.embed_dim, self.num_heads, self.pkt_count, self.pkt_dim, self.mlp_hidden,
                    init_schur_scale=native_spec.init_schur_gate_scale,
                    uncollapsed_cholesky=native_spec.uncollapsed_cholesky_highway,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.chol_mode_log_prior = nn.Parameter(torch.tensor([native_spec.anchor_logit_bias, 0.0, 0.0, 0.0], dtype=torch.float32))
        self.chol_mode_coherence = nn.Parameter(torch.tensor(native_spec.mode_coherence_mix, dtype=torch.float32))
        self.sem_energy_mix_logit = nn.Parameter(torch.tensor(-0.85, dtype=torch.float32))
        self.sem_chol_mode_weights = nn.Parameter(torch.tensor([0.50, 0.30, 0.20], dtype=torch.float32))
        self.sem_chol_branch_gates = nn.Parameter(torch.tensor([0.12, 0.10, 0.08], dtype=torch.float32))

        n_sem_complex = self.sem_complex_pkts
        self.in_m_mix_re = nn.Parameter(torch.randn(n_sem_complex, self.pkt_dim) * 0.25)
        self.in_m_mix_im = nn.Parameter(torch.randn(n_sem_complex, self.pkt_dim) * 0.25)
        self.in_m_schur_re = nn.Parameter(torch.randn(n_sem_complex, self.pkt_dim) * 0.25)
        self.in_m_schur_im = nn.Parameter(torch.randn(n_sem_complex, self.pkt_dim) * 0.25)
        self.in_mix_logit = nn.Parameter(torch.full((n_sem_complex,), -1.25, dtype=torch.float32))
        self.in_schur_logit = nn.Parameter(torch.full((n_sem_complex,), -1.85, dtype=torch.float32))

    def native_density_scalar_parameters(self) -> list[nn.Parameter]:
        """Return scalar parameters for native/uncollapsed Cholesky readout and outer-product input lift."""
        params = [self.chol_mode_log_prior, self.chol_mode_coherence]
        if self.native_spec.uncollapsed_cholesky_highway:
            params.append(self.sem_energy_mix_logit)
        if self.native_spec.use_semantic_cholesky_readout:
            params.extend([self.sem_chol_mode_weights, self.sem_chol_branch_gates])
        if self.native_spec.use_outer_product_input:
            params.extend([self.in_m_mix_re, self.in_m_mix_im, self.in_m_schur_re, self.in_m_schur_im, self.in_mix_logit, self.in_schur_logit])
        return params

    def _encode_multiscale_trunk(
        self,
        idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Encode `idx` `(B, T)` while propagating uncollapsed 4-mode Cholesky factors across layers when enabled."""
        bsz, seq_len = idx.shape
        work_dtype = torch.bfloat16 if (idx.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
        tok_wp = F.embedding(idx, self.wte_3d.view(self.vocab_size, self.embed_dim)).view(bsz, seq_len, self.pkt_count, self.pkt_dim)
        raw_wp = tok_wp.to(work_dtype)
        if seq_len > 1:
            raw_wp = self._apply_dispersion(raw_wp)

        if self.native_spec.use_outer_product_input and seq_len > 1:
            sem_curr = raw_wp[:, 1:, : 2 * self.sem_complex_pkts, :].view(bsz, seq_len - 1, self.sem_complex_pkts, 2, self.pkt_dim)
            sem_prev = raw_wp[:, :-1, : 2 * self.sem_complex_pkts, :].view(bsz, seq_len - 1, self.sem_complex_pkts, 2, self.pkt_dim)
            sem_lifted = apply_phase_invariant_outer_density_input_lift(
                sem_curr, sem_prev, self.in_m_mix_re, self.in_m_mix_im, self.in_m_schur_re, self.in_m_schur_im, (self.in_mix_logit, self.in_schur_logit, True)
            ).reshape(bsz, seq_len - 1, self.sem_real_dim)
            raw_flat_disp = raw_wp.reshape(bsz, seq_len, self.embed_dim)
            sem_full = torch.cat([raw_flat_disp[:, :1, : self.sem_real_dim], sem_lifted], dim=1)
            raw_flat = torch.cat([sem_full, raw_flat_disp[:, :, self.sem_real_dim :]], dim=-1)
        else:
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
        x = x_0
        cos_4d, sin_4d = self.rope.get_cos_sin(seq_len, dtype=work_dtype)
        mid_idx = max(0, (len(self.quantum_blocks) // 2) - 1)
        h_mid = x_0
        chol_state: tuple[torch.Tensor, torch.Tensor] | None = None
        for b_idx, q_blk in enumerate(self.quantum_blocks):
            x = q_blk(x, x_0, cos_4d, sin_4d, in_cholesky=chol_state)  # type: ignore[call-arg]
            if self.native_spec.uncollapsed_cholesky_highway:
                chol_state = q_blk.last_cholesky_pre_u
            if b_idx == mid_idx:
                h_mid = F.rms_norm(x, (self.embed_dim,), eps=1e-6)

        h_pre = F.rms_norm(x, (self.embed_dim,), eps=1e-6)
        return h_pre, h_mid, readout_sem_448, readout_id_64

    def build_multibranch_readout_tensors(
        self,
        idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build 4 readout modes directly from the final layer's rank-4 Cholesky factor $L_{t}^{(L)}$ (Zero `mb_head`!)."""
        h_pre, _h_mid, readout_sem_448, readout_id_64 = self._encode_multiscale_trunk(idx)
        bsz, seq_len, _ = h_pre.shape
        work_dtype = h_pre.dtype
        h_bridge = self.exit_bridge(h_pre)

        h_sem_tr = F.rms_norm(h_bridge[:, :, : self.sem_real_dim], (self.sem_real_dim,), eps=1e-6)
        h_id_tr = F.rms_norm(h_bridge[:, :, self.sem_real_dim :], (self.id_real_dim,), eps=1e-6)
        if readout_sem_448 is not None and readout_id_64 is not None:
            g_sem, g_id = self.trunk_sem_gate.to(work_dtype), self.trunk_id_gate.to(work_dtype)
            h_sem_0 = F.rms_norm(g_sem * h_sem_tr + readout_sem_448, (self.sem_real_dim,), eps=1e-6)
            h_id_0 = F.rms_norm(g_id * h_id_tr + readout_id_64, (self.id_real_dim,), eps=1e-6)
        else:
            h_sem_0, h_id_0 = h_sem_tr, h_id_tr

        sem_scale = (self.temp_sem_scale / float(self.sem_real_dim)).to(work_dtype)
        id_scale_scalar = (self.temp_id_scale / float(self.id_real_dim)).to(work_dtype)

        h_sem_0_scaled = h_sem_0 * sem_scale
        c0_batch = self.carrier_c0.to(work_dtype).view(1, 1, DEFAULT_CARRIER_BIAS_DIM).expand(
            bsz, seq_len, DEFAULT_CARRIER_BIAS_DIM
        )
        h_base_464 = torch.cat([h_sem_0_scaled, c0_batch], dim=-1).contiguous()

        w_dir = (self.w_det_512 * self.radial_scale.unsqueeze(1)).to(work_dtype)
        b_col = (self.prior_gate * self.unigram_bias).to(work_dtype).unsqueeze(1) * self.carrier_c0.to(
            work_dtype
        ).unsqueeze(0)
        w_aug = torch.cat([w_dir, b_col], dim=-1).contiguous()

        last_blk: _NativeCholeskySchurWavepacketBlock = self.quantum_blocks[-1]  # type: ignore[assignment]
        assert last_blk.last_cholesky_id_re is not None and last_blk.last_cholesky_id_im is not None
        l7_re, l7_im = last_blk.last_cholesky_id_re, last_blk.last_cholesky_id_im

        id_energy = (l7_re.float().square().sum(dim=-1) + l7_im.float().square().sum(dim=-1)).clamp_min(1e-6)
        if self.native_spec.uncollapsed_cholesky_highway and last_blk.last_cholesky_all_re is not None:
            l_sem_re = last_blk.last_cholesky_all_re[:, : self.sem_complex_pkts]
            l_sem_im = last_blk.last_cholesky_all_im[:, : self.sem_complex_pkts]  # type: ignore[index]
            sem_energy = (
                l_sem_re.float().square().sum(dim=-1) + l_sem_im.float().square().sum(dim=-1)
            ).mean(dim=1).clamp_min(1e-6)
            w_sem = torch.sigmoid(self.sem_energy_mix_logit.float())
            mode_energy = (1.0 - w_sem) * id_energy + w_sem * sem_energy
        else:
            mode_energy = id_energy
        mode_log_weights = torch.log(mode_energy) + self.chol_mode_log_prior.view(1, 1, 4)

        l7_64 = torch.stack([l7_re, l7_im], dim=-2).reshape(bsz, seq_len, 4, self.id_real_dim)
        l7_unit = F.rms_norm(l7_64, (self.id_real_dim,), eps=1e-6)
        coh = torch.sigmoid(self.chol_mode_coherence).to(work_dtype)

        excited_modes = F.rms_norm(coh * h_id_0.unsqueeze(-2) + (1.0 - coh) * l7_unit[:, :, 1:, :], (self.id_real_dim,), eps=1e-6)
        cholesky_id_branches = torch.cat([h_id_0.unsqueeze(-2), excited_modes], dim=-2)
        h_id_branches = (cholesky_id_branches * id_scale_scalar).contiguous()

        if self.native_spec.use_semantic_cholesky_readout and last_blk.last_cholesky_all_re is not None:
            l_sem_re = last_blk.last_cholesky_all_re[:, : self.sem_complex_pkts]
            l_sem_im = last_blk.last_cholesky_all_im[:, : self.sem_complex_pkts]  # type: ignore[index]
            sem_exc = torch.stack([l_sem_re[:, :, :, 1:, :], l_sem_im[:, :, :, 1:, :]], dim=-2)
            w_m = F.softmax(self.sem_chol_mode_weights.float(), dim=0).to(work_dtype).view(1, 1, 1, 3, 1, 1)
            sem_super = (sem_exc * w_m).sum(dim=3).permute(0, 2, 1, 3, 4).reshape(bsz, seq_len, self.sem_real_dim)
            u_f, s0_f = sem_super.float(), h_sem_0.float()
            proj_c = (u_f * s0_f).sum(dim=-1, keepdim=True) / s0_f.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
            h_sem_1 = F.rms_norm((u_f - proj_c * s0_f).to(work_dtype), (self.sem_real_dim,), eps=1e-6)
            h_sem_1_out = (h_sem_1 * sem_scale).contiguous()
            b_exc = (0.35 * torch.tanh(self.sem_chol_branch_gates.float())).view(1, 1, 3).expand(bsz, seq_len, 3)
            sem1_gates = torch.cat([torch.zeros((bsz, seq_len, 1), device=idx.device, dtype=torch.float32), b_exc], dim=-1)
        else:
            h_sem_1_out = torch.zeros_like(h_sem_0_scaled)
            sem1_gates = torch.zeros_like(mode_log_weights)

        id_ones = torch.ones_like(mode_log_weights)
        branch_controls = torch.stack([sem1_gates, id_ones, mode_log_weights], dim=-1).to(work_dtype)
        return h_base_464, h_sem_1_out, h_id_branches, branch_controls, w_aug

    def forward_loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Native Cholesky Density Born readout loss without `mb_head`."""
        h_base_464, h_sem1_448, h_id_branches, branch_controls, w_aug = self.build_multibranch_readout_tensors(idx)
        return _FactoredMultiBranchWavepacketCEFn.apply(h_base_464, h_sem1_448, h_id_branches, branch_controls, w_aug, targets)


def run_wavefunction_outer_input_reversal_benchmark(
    steps: int = DEFAULT_REVERSAL_STEPS,
    eval_interval: int = DEFAULT_REVERSAL_EVAL_INTERVAL,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """Benchmark `W0..W3` Wavefunction -> Outer-Product Density Input modes on `50,304`-vocab reversal."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    dataset = FineWebSpanReversalDataset(vocab_size=50304, device=dev)
    eff_spec = SupportSketchSpec(
        mode=SupportSketchMode.I11_COMPLEX_WAVEFUNCTION_8x2x32_ID2, vocab_size=50304, pkt_count=16, pkt_dim=32, seed=1337
    )

    variants = [
        ("W0_VECTOR_SUM_THEN_OUTER", WavefunctionInputMode.W0_VECTOR_SUM_THEN_OUTER, "Vector sum |psi_t> + |pos_t> + |psi_{t-1}> before outer product"),
        ("W1_OUTER_PHASE_ALIGNED_MIXTURE", WavefunctionInputMode.W1_OUTER_PHASE_ALIGNED_MIXTURE, "Phase-aligned outer-product mixture (|psi_{t-1}><psi_{t-1}|)|psi_t> + Born-Trace gate"),
        ("W2_OUTER_PLUS_BIGRAM_SCHUR_INPUT", WavefunctionInputMode.W2_OUTER_PLUS_BIGRAM_SCHUR_INPUT, "Outer-product mixture + Input Bigram Schur coincidence D*(rho_t * rho_{t-1})"),
        ("W3_UNITARY_POS_OUTER_DENSITY_INPUT", WavefunctionInputMode.W3_UNITARY_POS_OUTER_DENSITY_INPUT, "Unitary positional phase conjugation + Input Outer-Product + Bigram Schur"),
    ]

    results: list[dict[str, Any]] = []
    for var_id, in_mode, desc in variants:
        cfg = MicroReversalConfig(
            backbone_mode=MicroBackboneMode.QUANTUM_WAVEPACKET,
            init_mode=MicroInitMode.GPU_UNIGRAM_PRIOR,
            readout_mode=MicroReadoutMode.R4_HERMITIAN_QUADRATIC_BORN,
            vocab_size=50304,
            span_len=dataset.span_len,
            embed_dim=512,
            num_heads=8,
            num_layers=DEFAULT_MICRO_NUM_LAYERS,
            mlp_hidden=1536,
            pkt_count=16,
            pkt_dim=32,
            seed=1337,
        )
        model = WavefunctionOuterInputReversalGPT(cfg, seq_len=dataset.seq_len, input_mode=in_mode).to(dev)
        apply_support_sketch_initialization_(model, dataset, eff_spec)

        stiefel_params: list[nn.Parameter] = list(model.readout.real_stiefel_parameters())
        unitary_pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
        for q_blk in model.quantum_blocks:
            stiefel_params.extend(q_blk.real_stiefel_parameters())  # type: ignore[operator]
            unitary_pairs.extend(q_blk.complex_unitary_pairs())  # type: ignore[operator]

        q_trunk_opt = ModularRiemannianStiefelKrausMuon(
            real_stiefel_params=stiefel_params,
            complex_operator_pairs=unitary_pairs,
            stiefel_lr=DEFAULT_MICRO_STIEFEL_LR,
            unitary_lr=DEFAULT_MICRO_UNITARY_LR,
            tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA,
            momentum_mode=StiefelMomentumMode.RIEMANNIAN_TANGENT_SNR_GATED,
        )
        wte_opt = BatchSubspaceTikhonovGramMuon(
            [model.wte_3d] + model.readout.sphere_parameters(), lr=DEFAULT_MICRO_WTE_LR, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA
        )
        scalar_params: list[nn.Parameter] = [model.wpe_3d, model.pos_scale, model.wp_causal_re, model.wp_causal_im]
        scalar_params.extend(model.extra_input_scalar_parameters())
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
                "variant_id": var_id,
                "input_mode": in_mode.value,
                "description": desc,
                "step0_val_ce": round(curve[0]["val_ce"], 4),
                "step50_val_ce": round(curve[1]["val_ce"], 4) if len(curve) > 1 else round(curve[-1]["val_ce"], 4),
                "step50_top1_acc": round(100.0 * curve[1]["top1_acc"], 2) if len(curve) > 1 else round(100.0 * curve[-1]["top1_acc"], 2),
                "final_val_ce": round(curve[-1]["val_ce"], 4),
                "final_top1_acc": round(100.0 * curve[-1]["top1_acc"], 2),
                "final_top5_acc": round(100.0 * curve[-1]["top5_acc"], 2),
                "elapsed_sec": round(elapsed, 2),
                "curve": curve,
            }
        )
        del model, q_trunk_opt, wte_opt, scalar_opt
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    return results


def _train_and_eval_native_cholesky_variants(
    variants: list[tuple[str, NativeCholeskyDensitySpec, str]],
    stream: LongSeqFineWebStream,
    steps: int,
    eval_interval: int,
) -> list[dict[str, Any]]:
    """Shared FineWeb training and Shard-0 evaluation loop for `NativeCholeskyDensityWavepacketGPT` variants (`<= 8` args)."""
    dev = stream.device
    results: list[dict[str, Any]] = []
    for var_id, nat_spec, desc in variants:
        torch.manual_seed(int(nat_spec.seed))
        model = NativeCholeskyDensityWavepacketGPT(nat_spec).to(dev)
        init_ms, _ = initialize_and_calibrate_shallow_model_(model, stream)
        model.clamp_identity_dispersion_and_scalars_()

        stiefel_params: list[nn.Parameter] = list(model.exit_bridge.real_stiefel_parameters())
        unitary_pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
        for q_blk in model.quantum_blocks:
            stiefel_params.extend(q_blk.real_stiefel_parameters())  # type: ignore[operator]
            unitary_pairs.extend(q_blk.complex_unitary_pairs())  # type: ignore[operator]

        fast_temp_params = [model.temp_sem_scale, model.temp_id_scale, model.trunk_sem_gate, model.trunk_id_gate]
        scalar_params = [
            model.wp_causal_re,
            model.wp_causal_im,
            model.smear_gate.weight,
            model.smear_lambda,
            model.radial_scale,
            model.unigram_bias,
            model.prior_gate,
        ]
        scalar_params.extend(model.exit_bridge.scalar_parameters())
        scalar_params.extend(model.native_density_scalar_parameters())
        for q_blk in model.quantum_blocks:
            scalar_params.extend(q_blk.scalar_parameters())  # type: ignore[operator]

        q_trunk_opt = ModularRiemannianStiefelKrausMuon(
            real_stiefel_params=stiefel_params,
            complex_operator_pairs=unitary_pairs,
            stiefel_lr=DEFAULT_SHALLOW_STIEFEL_LR,
            unitary_lr=DEFAULT_SHALLOW_UNITARY_LR,
            tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA,
            momentum_mode=StiefelMomentumMode.RIEMANNIAN_TANGENT_SNR_GATED,
        )
        wte_opt = BatchSubspaceTikhonovGramMuon([model.wte_3d], lr=DEFAULT_SHALLOW_WTE_LR, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA)
        head_opt = BatchSubspaceTikhonovGramMuon([model.w_det_512], lr=DEFAULT_SHALLOW_READOUT_LR, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA)

        adam_groups: list[dict[str, Any]] = [
            {"params": scalar_params, "lr": DEFAULT_SHALLOW_SCALAR_LR, "weight_decay": 0.0},
            {"params": fast_temp_params, "lr": 0.06, "weight_decay": 0.0},
        ]
        if model.sem_ngram is not None:
            ngram_tables = [
                model.sem_ngram.unigram_next_table,
                model.sem_ngram.bigram_table,
                model.sem_ngram.trigram_table,
                model.sem_ngram.skipgram_table,
                model.sem_ngram.u_sem,
            ]
            ngram_scalars = [
                model.sem_ngram.in_scale,
                model.sem_ngram.readout_sem_scale,
                model.sem_ngram.readout_id_scale,
                model.sem_ngram.order_logits,
            ]
            adam_groups.append({"params": ngram_tables, "lr": DEFAULT_SHALLOW_NGRAM_LR, "weight_decay": 0.0})
            adam_groups.append({"params": ngram_scalars, "lr": DEFAULT_SHALLOW_SCALAR_LR, "weight_decay": 0.0})
        adam_opt = torch.optim.AdamW(adam_groups, betas=(0.80, 0.95), eps=1e-8, fused=(dev.type == "cuda"))

        s0 = evaluate_multibranch_model(model, stream)
        s0["step"] = 0
        curve = [s0]
        print(
            f"[{var_id}] Step    0 | Val CE: {s0['val_ce']:.4f} (Neural {s0['neural_val_ce']:.4f}, PPL {s0['val_ppl']:.2f}) | "
            f"Top1: {100.0 * s0['top1_acc']:.2f}% | Top5: {100.0 * s0['top5_acc']:.2f}% | CholeskyModes: {s0['effective_active_branches']:.2f}",
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
            adam_opt.step()

            model.clamp_identity_dispersion_and_scalars_()

            if step % eval_interval == 0 or step == steps:
                ev = evaluate_multibranch_model(model, stream)
                ev["step"] = step
                ev["tokens_M"] = round(float(step * stream.batch_size * stream.seq_len) / 1.0e6, 3)
                ev["train_loss"] = round(float(loss.detach().item()), 4)
                curve.append(ev)
                print(
                    f"[{var_id}] Step {step:4d} ({ev['tokens_M']:5.2f}M tok) | Train CE: {ev['train_loss']:.4f} | "
                    f"Val CE: {ev['val_ce']:.4f} (Neural {ev['neural_val_ce']:.4f}, PPL {ev['val_ppl']:.2f}) | "
                    f"Top1: {100.0 * ev['top1_acc']:.2f}% | Top5: {100.0 * ev['top5_acc']:.2f}% | CholeskyModes: {ev['effective_active_branches']:.2f}",
                    flush=True,
                )

        elapsed = time.perf_counter() - t0
        results.append(
            {
                "variant_id": var_id,
                "description": desc,
                "uncollapsed_cholesky_highway": nat_spec.uncollapsed_cholesky_highway,
                "use_semantic_cholesky_readout": nat_spec.use_semantic_cholesky_readout,
                "enable_kn_synergy_eval": nat_spec.enable_kn_synergy_eval,
                "init_ms": round(init_ms, 2),
                "elapsed_sec": round(elapsed, 2),
                "step0": curve[0],
                "final": curve[-1],
                "curve": curve,
            }
        )
        del model, q_trunk_opt, wte_opt, head_opt, adam_opt
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    return results


def run_native_cholesky_density_fineweb_benchmark(
    steps: int = DEFAULT_FINEWEB_STEPS,
    eval_interval: int = DEFAULT_FINEWEB_EVAL_INTERVAL,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """Benchmark Cone-Free `D5` & `D6` (`NativeCholeskyDensityWavepacketGPT`) on FineWeb (`L=4, d=512, T=512`)."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    stream = LongSeqFineWebStream(device=dev)
    variants = [
        ("D5_NATIVE_CHOLESKY_DENSITY_NO_CONE", NativeCholeskyDensitySpec(use_outer_product_input=False, seed=1337), "Zero mb_head / Zero 30-deg Cone / Zero SVD Calib: direct rank-4 Cholesky density readout"),
        ("D6_OUTER_INPUT_NATIVE_CHOLESKY_NO_CONE", NativeCholeskyDensitySpec(use_outer_product_input=True, seed=1337), "Phase-invariant Outer-Product + Bigram Schur Input + Cone-Free Native Cholesky Readout"),
    ]
    return _train_and_eval_native_cholesky_variants(variants, stream, steps, eval_interval)


def run_uncollapsed_cholesky_ablation_benchmark(
    steps: int = DEFAULT_FINEWEB_STEPS,
    eval_interval: int = DEFAULT_FINEWEB_EVAL_INTERVAL,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """Benchmark Uncollapsed 4-Mode Cholesky Propagation (`D7`, `D8`) on FineWeb (`L=4, d=512, T=512`, `300` steps)."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    stream = LongSeqFineWebStream(device=dev)
    variants = [
        (
            "D7_UNCOLLAPSED_CHOLESKY_4MODE_HIGHWAY",
            NativeCholeskyDensitySpec(uncollapsed_cholesky_highway=True, use_semantic_cholesky_readout=False, seed=1337),
            "Uncollapsed 4-Mode Cholesky Highway across all L=4 layers (4x4 Gram McWeeny + Phase-Locked Born-Trace Highway)",
        ),
        (
            "D8_UNCOLLAPSED_CHOLESKY_SEM_ID_READOUT",
            NativeCholeskyDensitySpec(uncollapsed_cholesky_highway=True, use_semantic_cholesky_readout=True, seed=1337),
            "Uncollapsed 4-Mode Cholesky Highway + Joint Semantic (448D) & Identity (64D) Cholesky Readout",
        ),
    ]
    return _train_and_eval_native_cholesky_variants(variants, stream, steps, eval_interval)


def run_uncollapsed_cholesky_long_scaling_benchmark(
    steps: int = DEFAULT_LONG_FINEWEB_STEPS,
    eval_interval: int = DEFAULT_LONG_FINEWEB_EVAL_INTERVAL,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """Run longer FineWeb scaling benchmark (`1,200` steps = `9.83M` tokens) with Uncollapsed 4-Mode Cholesky Highway."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    stream = LongSeqFineWebStream(device=dev)
    variants = [
        (
            "D9_UNCOLLAPSED_CHOLESKY_LONG_SCALING_1200",
            NativeCholeskyDensitySpec(
                uncollapsed_cholesky_highway=True,
                use_semantic_cholesky_readout=True,
                enable_kn_synergy_eval=True,
                seed=1337,
            ),
            "1,200-step (9.83M-token) FineWeb scaling run with Uncollapsed 4-Mode Cholesky Highway (reports both Neural & KN-Synergy CE)",
        ),
    ]
    return _train_and_eval_native_cholesky_variants(variants, stream, steps, eval_interval)

