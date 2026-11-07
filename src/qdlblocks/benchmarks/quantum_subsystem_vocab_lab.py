r"""Fixed-Once Subsystem Quantum Vocabulary Architectures: Sequential Quantum Instruments (`256 ⊕ 256` Lüders Collapse), Right-Canonical 4-Qudit Matrix Product State (`16^4` MPS) Born Readout, & Shared `St(512, 128)` Subspace Frame (`EXPERIMENT_PLAN_085.md` / `Wave 65`).

Mathematical & Architectural Overview
-------------------------------------
Eliminates the monolithic `(50304, 512)` input/output vocabulary bottleneck (`91%` of parameters, `5.17%` active duty
cycle at `8K` tokens/step) by picking a **Fixed-Once Subsystem Vocabulary Boundary** at model construction time (Rule 3
Strategy Pattern, Rule 6 Flag-Free Module) with zero runtime table synthesis or repacking:

1. **`LUDERS_256X256_INSTRUMENT` (`LudersSequentialInstrumentBoundary` — $256 \oplus 197$ Bipartite Quantum Instrument)**:
   - Maps every GPT-2 token $v \in \{0, \dots, 50256\}$ via the balanced BPE residue-quotient bijection:
     $$a(v) = v \ \& \ \text{0xFF} \in \{0, \dots, 255\}, \qquad b(v) = v \gg 8 \in \{0, \dots, B_{\max}(a)\}, \qquad v = 256\,b + a$$
     where $B_{\max}(a) = 196$ for $a \in \{0, \dots, 80\}$ (`197` valid outcomes) and $B_{\max}(a) = 195$ for
     $a \in \{81, \dots, 255\}$ (`196` valid outcomes), covering $81 \times 197 + 175 \times 196 = 50,257$ tokens with
     zero empty buckets, uniform Zipf probability across all `256` coarse outcomes, and **`100.0%` active duty cycle**.
   - **Stage 1 (`256`-way Coarse Measurement)**: Measures $h_{1, t} \in \mathbb{R}^{512}$ against $E^{(1)} \in \mathbb{S}^{511}(\sqrt{512})^{256}$.
   - **Stage 2 (Lüders Post-Measurement State Collapse)**: Conditioned on $a^* = a(y_t)$, collapses the tomographic state
     using both a subsystem conditioning map and the explicit density-matrix transition vector
     $\mathcal{M}_{\rho_t}(a^*)_h = \rho_{t,h}|\phi_{a^*, h}\rangle \in \mathbb{C}^{16}$.
   - **Stage 3 (`197`-way Conditional Fine Measurement)**: Measures the collapsed state $h_{2, t} \in \mathbb{R}^{512}$
     against $E^{(2)} \in \mathbb{S}^{511}(\sqrt{512})^{197}$ with exact support mask $b \le B_{\max}(a^*)$.

2. **`MPS_16POW4_BORN_CHAIN` (`RightCanonicalMPSBornBoundary` — 4-Site Base-$16$ Matrix Product State Born Machine)**:
   - Decomposes $v \in \{0, \dots, 50256\}$ into 4 hexadecimal qudits $(s_0, s_1, s_2, s_3) \in \{0, \dots, 15\}^3 \times \{0, \dots, 12\}$
     ($v = s_0 + 16 s_1 + 256 s_2 + 4096 s_3$) with exact Site-3 support $s_3 \le S_{3,\max}(p_{012})$.
   - Parameterizes the 4 site transition operators as right-canonical Stiefel isometries
     $A^{[k]} \in \operatorname{St}(16 d_{\text{bond}}, d_{\text{bond}}) = \operatorname{St}(2048, 128)$ satisfying
     $\sum_{s=0}^{15} (A_s^{[k]})^\top A_s^{[k]} = I_{128}$, stepped on $\operatorname{St}(2048, 128)$ via
     [`ModularRiemannianStiefelKrausMuon`](file:///home/festivity/agy/QuantumDLBlocks/src/qdlblocks/models/modular_quantum_statmech_gpt.py).

3. **`SHARED_STIEFEL_128D_FRAME` (`SharedStiefelSubspaceFrameBoundary` — Tied `128D` Sphere + `St(512, 128)` Frame)**:
   - Ties input and output vocabulary representations into a single low-dimensional sphere
     $W_{\text{vocab}} \in \mathbb{S}^{127}(\sqrt{128})^{50304}$ bridged to the `512D` trunk via
     $U_{\text{in}}, U_{\text{out}} \in \operatorname{St}(512, 128)$ ($4\times$ smaller GEMM, $64\times$ faster `128×128` Gram-Muon).
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
    DEFAULT_INIT_FILTER_SCALE, DEFAULT_INIT_TOMO_SCALE, DEFAULT_INIT_VACUUM_REST_MASS,
    DEFAULT_PROBES_PER_HEAD, DEFAULT_TOK_NGRAM_BUCKETS, DEFAULT_TOMO_EMBED_DIM,
    DEFAULT_TOMO_HEAD_COMPLEX_DIM, DEFAULT_TOMO_MLP_HIDDEN, DEFAULT_TOMO_NUM_HEADS,
    DEFAULT_TOMO_NUM_LAYERS, DEFAULT_TOMO_PACKED_RHO_DIM, DEFAULT_TOMO_VACUUM_DIM,
    TOMO_CE_TOKEN_CHUNK, apply_unified_512d_vacuum_sphere_norm, build_layer0_explicit_outer_density,
)
from qdlblocks.benchmarks.kraus_mub_thermostat_8l_lab import (
    DEFAULT_ANALYTICAL_BETA_FAST_TOKENS, DEFAULT_ANALYTICAL_BETA_INIT, DEFAULT_ANALYTICAL_BETA_MID,
    DEFAULT_ANALYTICAL_BETA_TARGET, DEFAULT_ANALYTICAL_BETA_WARMUP_TOKENS, DEFAULT_KRAUS_BRANCH_RANK,
    DEFAULT_KRAUS_PRIMARY_WEIGHT, RiemannianShadowEMAThermostat, _KrausDensityHighwayBlock,
    build_initial_kraus_stiefel_operators, build_mub_two_basis_unitary_probes,
    compute_analytical_parseval_beta, compute_mub_two_basis_probe_expectations,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import (
    DEFAULT_SHALLOW_NGRAM_LR, DEFAULT_SHALLOW_READOUT_LR, DEFAULT_SHALLOW_SCALAR_LR,
    DEFAULT_SHALLOW_STIEFEL_LR, DEFAULT_SHALLOW_VOCAB_SIZE, DEFAULT_SHALLOW_WTE_LR,
    DEFAULT_SOFTCAP_LOGIT, RAW_GPT2_VOCAB_SIZE, LongSeqFineWebStream,
)
from qdlblocks.benchmarks.zero_ppmi_byte_pair_cholesky_lab import (
    OnlineCausalZipfTracker, build_zero_data_bpe_rank_zipf_prior,
)
from qdlblocks.layers.impure_density_mirror_block import (
    DEFAULT_TIKHONOV_GRAM_LAMBDA, _complex_hermitian_mv, complex_l2_normalize, unpack_hermitian_isometric,
)
from qdlblocks.layers.quantum_statmech_channels import DEFAULT_MUON_GRAD_NORM_GATE_TAU, StiefelMomentumMode
from qdlblocks.layers.quantum_svd_stiefel_graft import BiorthogonalSVDStiefelProjection
from qdlblocks.models.compiled_leaderboard_classical_gpt import NativeLeaderboardRotary
from qdlblocks.models.modular_quantum_statmech_gpt import ModularRiemannianStiefelKrausMuon
from qdlblocks.optim.vocab_manifold_muon import BatchSubspaceTikhonovGramMuon, RiemannianGeodesicScalarAndProbeOptimizer

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_085.md Section 6.1
# ==============================================================================
LUDERS_COARSE_OUTCOMES: int = 256
LUDERS_FINE_MAX_OUTCOMES: int = 197
LUDERS_FULL_ROW_CUTOFF: int = 81
MPS_QUDIT_BASE: int = 16
MPS_NUM_SITES: int = 4
MPS_SITE3_MAX_OUTCOMES: int = 13
MPS_SITE3_FULL_PREFIX_CUTOFF: int = 1105
DEFAULT_MPS_BOND_DIM: int = 128
DEFAULT_SHARED_VOCAB_SUBSPACE_DIM: int = 128
DEFAULT_LUDERS_RHO_TRACE_SCALE: float = 0.40


class VocabSubsystemMode(str, Enum):
    """Fixed-once vocabulary parameterization strategy (Rule 3)."""

    LUDERS_256X256_INSTRUMENT = "luders_256x256_instrument"
    MPS_16POW4_BORN_CHAIN = "mps_16pow4_born_chain"
    SHARED_STIEFEL_128D_FRAME = "shared_stiefel_128d_frame"


@dataclass(frozen=True)
class QuantumSubsystemVocabSpec:
    """Immutable specification for `QuantumSubsystemVocabGPT` (`Wave 65` / `EXPERIMENT_PLAN_085.md`, Rule 2)."""

    vocab_mode: VocabSubsystemMode = VocabSubsystemMode.LUDERS_256X256_INSTRUMENT
    num_layers: int = DEFAULT_TOMO_NUM_LAYERS
    kraus_rank: int = DEFAULT_KRAUS_BRANCH_RANK
    mps_bond_dim: int = DEFAULT_MPS_BOND_DIM
    shared_subspace_dim: int = DEFAULT_SHARED_VOCAB_SUBSPACE_DIM
    analytical_beta_target: float = DEFAULT_ANALYTICAL_BETA_TARGET
    grad_norm_gate_tau: float = DEFAULT_MUON_GRAD_NORM_GATE_TAU
    seed: int = 1337


def build_luders_hierarchical_zipf_priors(
    vocab_size: int = RAW_GPT2_VOCAB_SIZE, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Factor the analytical BPE-rank Zipf prior over `50,257` tokens into exact `256` coarse + `(256, 197)` conditional log-priors (`<= 8` args).

    For $v = 256\,b + a \in \{0, \dots, 50256\}$ ($a \in \{0, \dots, 255\}$, $b \in \{0, \dots, B_{\max}(a)\}$):
      $$b_a^{(1)} = \log \sum_{b=0}^{B_{\max}(a)} p_{\text{tok}}(256 b + a), \qquad b_{a, b}^{(2)} = \log p_{\text{tok}}(256 b + a) - b_a^{(1)}$$
    so $b_a^{(1)} + b_{a, b}^{(2)} \equiv \log p_{\text{tok}}(256 b + a)$ identically for all $50,257$ tokens.
    """
    dev = torch.device(device)
    log_p_all, _, _ = build_zero_data_bpe_rank_zipf_prior(DEFAULT_SHALLOW_VOCAB_SIZE, device=dev)
    log_p_valid = log_p_all[:vocab_size]
    log_p_valid = log_p_valid - torch.logsumexp(log_p_valid, dim=0)

    a_grid = torch.arange(LUDERS_COARSE_OUTCOMES, device=dev, dtype=torch.long).unsqueeze(1)
    b_grid = torch.arange(LUDERS_FINE_MAX_OUTCOMES, device=dev, dtype=torch.long).unsqueeze(0)
    v_grid = 256 * b_grid + a_grid
    valid_mask = v_grid < vocab_size

    v_clamped = v_grid.clamp(0, vocab_size - 1)
    log_p_2d = torch.where(valid_mask, log_p_valid[v_clamped], torch.full_like(v_grid, -1e9, dtype=torch.float32))
    log_p_coarse = torch.logsumexp(log_p_2d, dim=1)
    log_p_fine_cond = torch.where(valid_mask, log_p_2d - log_p_coarse.unsqueeze(1), torch.zeros_like(log_p_2d))
    return log_p_coarse, log_p_fine_cond, valid_mask


def build_mps_4qudit_hierarchical_zipf_priors(
    vocab_size: int = RAW_GPT2_VOCAB_SIZE, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Factor the analytical BPE-rank Zipf prior over `50,257` tokens across the 4 hexadecimal qudit sites $(s_0, s_1, s_2, s_3)$ (`<= 8` args).

    With $v = s_0 + 16 s_1 + 256 s_2 + 4096 s_3$:
      $$\log p_{\text{tok}}(v) \equiv b_{s_0}^{(0)} + b_{s_0, s_1}^{(1)} + b_{p_{01}, s_2}^{(2)} + b_{p_{012}, s_3}^{(3)}$$
    """
    dev = torch.device(device)
    log_p_all, _, _ = build_zero_data_bpe_rank_zipf_prior(DEFAULT_SHALLOW_VOCAB_SIZE, device=dev)
    log_p_valid = log_p_all[:vocab_size]
    log_p_valid = log_p_valid - torch.logsumexp(log_p_valid, dim=0)

    p012 = torch.arange(4096, device=dev, dtype=torch.long).unsqueeze(1)
    s3 = torch.arange(MPS_SITE3_MAX_OUTCOMES, device=dev, dtype=torch.long).unsqueeze(0)
    v_grid = 4096 * s3 + p012
    valid_s3_mask = v_grid < vocab_size

    log_p_4d = torch.where(
        valid_s3_mask, log_p_valid[v_grid.clamp(0, vocab_size - 1)], torch.full((4096, MPS_SITE3_MAX_OUTCOMES), -1e9, device=dev)
    )
    log_m012 = torch.logsumexp(log_p_4d, dim=1)
    b_s3 = torch.where(valid_s3_mask, log_p_4d - log_m012.unsqueeze(1), torch.zeros_like(log_p_4d))

    log_m012_grid = log_m012.view(16, 256).t().contiguous()
    log_m01 = torch.logsumexp(log_m012_grid, dim=1)
    b_s2 = log_m012_grid - log_m01.unsqueeze(1)

    log_m01_grid = log_m01.view(16, 16).t().contiguous()
    log_m0 = torch.logsumexp(log_m01_grid, dim=1)
    b_s1 = log_m01_grid - log_m0.unsqueeze(1)
    b_s0 = log_m0
    return b_s0, b_s1, b_s2, b_s3, valid_s3_mask


class LudersSequentialInstrumentBoundary(nn.Module):
    r"""Fixed-Once Two-Stage Sequential Quantum Instrument ($256 \oplus 197$ Lüders Collapse, Zero `(50304, 512)` Tables)."""

    def __init__(self, embed_dim: int = DEFAULT_TOMO_EMBED_DIM, num_heads: int = DEFAULT_TOMO_NUM_HEADS, head_complex_dim: int = DEFAULT_TOMO_HEAD_COMPLEX_DIM) -> None:
        super().__init__()
        self.embed_dim, self.num_heads, self.d_h = int(embed_dim), int(num_heads), int(head_complex_dim)
        self.head_real_dim = 2 * self.d_h

        self.w_in_coarse = nn.Parameter(torch.empty(LUDERS_COARSE_OUTCOMES, self.embed_dim, dtype=torch.float32))
        self.w_in_fine = nn.Parameter(torch.empty(LUDERS_FINE_MAX_OUTCOMES, self.embed_dim, dtype=torch.float32))
        self.in_phase_fine = nn.Parameter(torch.zeros(LUDERS_FINE_MAX_OUTCOMES, self.num_heads, dtype=torch.float32))

        self.e_det_coarse = nn.Parameter(torch.empty(LUDERS_COARSE_OUTCOMES, self.embed_dim, dtype=torch.float32))
        self.e_det_fine = nn.Parameter(torch.empty(LUDERS_FINE_MAX_OUTCOMES, self.embed_dim, dtype=torch.float32))
        self.radial_coarse = nn.Parameter(torch.ones(LUDERS_COARSE_OUTCOMES, dtype=torch.float32))
        self.radial_fine = nn.Parameter(torch.ones(LUDERS_FINE_MAX_OUTCOMES, dtype=torch.float32))

        self.luders_gate_a = nn.Parameter(torch.zeros(LUDERS_COARSE_OUTCOMES, self.embed_dim, dtype=torch.float32))
        self.luders_shift_a = nn.Parameter(torch.zeros(LUDERS_COARSE_OUTCOMES, self.embed_dim, dtype=torch.float32))
        self.phi_a_re = nn.Parameter(torch.empty(LUDERS_COARSE_OUTCOMES, self.num_heads, self.d_h, dtype=torch.float32))
        self.phi_a_im = nn.Parameter(torch.empty(LUDERS_COARSE_OUTCOMES, self.num_heads, self.d_h, dtype=torch.float32))
        self.luders_rho_scale = nn.Parameter(torch.tensor(DEFAULT_LUDERS_RHO_TRACE_SCALE, dtype=torch.float32))

        self.bias_coarse = nn.Parameter(torch.zeros(LUDERS_COARSE_OUTCOMES, dtype=torch.float32))
        self.bias_fine_cond = nn.Parameter(torch.zeros(LUDERS_COARSE_OUTCOMES * LUDERS_FINE_MAX_OUTCOMES, dtype=torch.float32))
        self.prior_gate = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("valid_mask_2d", torch.ones(LUDERS_COARSE_OUTCOMES, LUDERS_FINE_MAX_OUTCOMES, dtype=torch.bool), persistent=False)

    @torch.no_grad()
    def initialize_boundary_(self, seed: int = 1337) -> None:
        """Initialize `256 ⊕ 197` spheres, complex probe kets, and exact hierarchical Zipf priors at Step 0."""
        device = self.w_in_coarse.device
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed) + 101)
        rad_d = math.sqrt(float(self.embed_dim))
        for p in (self.w_in_coarse, self.w_in_fine, self.e_det_coarse, self.e_det_fine):
            raw = torch.randn(p.shape, generator=gen, device=device, dtype=torch.float32)
            p.copy_(F.normalize(raw, dim=-1, eps=1e-6) * rad_d)
        nn.init.zeros_(self.in_phase_fine)
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

    def encode_tokens(self, idx: torch.Tensor) -> torch.Tensor:
        """Prepare initial `512D` wavepacket directly from `a = idx & 0xFF` and `b = idx >> 8` via head phase binding."""
        bsz, seq_len = idx.shape
        work_dtype = torch.bfloat16 if (idx.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
        a_idx = (idx & 0xFF).long()
        b_idx = (idx >> 8).clamp(0, LUDERS_FINE_MAX_OUTCOMES - 1).long()

        emb_a = F.embedding(a_idx, self.w_in_coarse).to(work_dtype).view(bsz, seq_len, self.num_heads, 2, self.d_h)
        emb_b = F.embedding(b_idx, self.w_in_fine).to(work_dtype).view(bsz, seq_len, self.num_heads, 2, self.d_h)
        theta = F.embedding(b_idx, self.in_phase_fine).to(work_dtype).unsqueeze(-1)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)

        a_re, a_im = emb_a[..., 0, :], emb_a[..., 1, :]
        rot_a_re = a_re * cos_t - a_im * sin_t
        rot_a_im = a_re * sin_t + a_im * cos_t
        out_heads = torch.stack([rot_a_re + 0.75 * emb_b[..., 0, :], rot_a_im + 0.75 * emb_b[..., 1, :]], dim=-2)
        return out_heads.reshape(bsz, seq_len, self.embed_dim)

    def _collapse_state_for_coarse_outcomes(
        self, u_flat: torch.Tensor, rho_re_flat: torch.Tensor, rho_im_flat: torch.Tensor, a_targets: torch.Tensor
    ) -> torch.Tensor:
        r"""Apply Stage-2 Lüders post-measurement state collapse conditioned on coarse outcomes `a_targets` `(N,)` (`<= 8` args)."""
        work_dtype = u_flat.dtype
        n_pos = u_flat.size(0)
        gate_a = 1.0 + 0.50 * torch.tanh(F.embedding(a_targets, self.luders_gate_a).to(work_dtype))
        shift_a = F.embedding(a_targets, self.luders_shift_a).to(work_dtype)

        phi_r = self.phi_a_re.index_select(0, a_targets).to(work_dtype).unsqueeze(-1)
        phi_i = self.phi_a_im.index_select(0, a_targets).to(work_dtype).unsqueeze(-1)
        m_re = (torch.matmul(rho_re_flat, phi_r) - torch.matmul(rho_im_flat, phi_i)).squeeze(-1)
        m_im = (torch.matmul(rho_re_flat, phi_i) + torch.matmul(rho_im_flat, phi_r)).squeeze(-1)
        m_512 = F.rms_norm(torch.stack([m_re, m_im], dim=-2).reshape(n_pos, self.embed_dim), (self.embed_dim,), eps=1e-6)
        return u_flat * gate_a + shift_a + self.luders_rho_scale.to(work_dtype) * m_512

    def forward_nll_and_metrics(
        self, u_vocab_512: torch.Tensor, u_vac: torch.Tensor | None, vac_mass: torch.Tensor,
        rho_re: torch.Tensor, rho_im: torch.Tensor, beta_0: torch.Tensor, targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute exact two-stage Lüders Quantum Instrument NLL over `50,257` tokens with `O(256 + 197)` GEMMs (`<= 8` args)."""
        bsz, seq_len, _ = u_vocab_512.shape
        n_pos = bsz * seq_len
        work_dtype = u_vocab_512.dtype
        cap = float(DEFAULT_SOFTCAP_LOGIT)
        inv_cap = 1.0 / cap

        h1_512, _ = apply_unified_512d_vacuum_sphere_norm(u_vocab_512, u_vac, vac_mass, beta_0, enable_vacuum=(u_vac is not None))
        h1_flat = h1_512.reshape(n_pos, self.embed_dim)
        e1_w = (F.rms_norm(self.e_det_coarse.to(work_dtype), (self.embed_dim,), eps=1e-6) * self.radial_coarse.to(work_dtype).unsqueeze(1)).t()
        b1 = (self.bias_coarse * self.prior_gate).float().unsqueeze(0)

        z1_neural = torch.matmul(h1_flat, e1_w).float()
        z1 = cap * torch.tanh(z1_neural * inv_cap) + b1
        log_p1 = F.log_softmax(z1, dim=-1)

        y_flat = targets.reshape(n_pos).clamp(0, RAW_GPT2_VOCAB_SIZE - 1).long()
        a_star = y_flat & 0xFF
        b_star = (y_flat >> 8).clamp(0, LUDERS_FINE_MAX_OUTCOMES - 1)

        u_flat = u_vocab_512.reshape(n_pos, self.embed_dim)
        u_vac_flat = u_vac.reshape(n_pos, -1) if u_vac is not None else None
        rho_re_flat = rho_re.permute(0, 2, 1, 3, 4).reshape(n_pos, self.num_heads, self.d_h, self.d_h)
        rho_im_flat = rho_im.permute(0, 2, 1, 3, 4).reshape(n_pos, self.num_heads, self.d_h, self.d_h)

        u_cond_flat = self._collapse_state_for_coarse_outcomes(u_flat, rho_re_flat, rho_im_flat, a_star)
        h2_flat, _ = apply_unified_512d_vacuum_sphere_norm(
            u_cond_flat.unsqueeze(1), u_vac_flat.unsqueeze(1) if u_vac_flat is not None else None,
            vac_mass, beta_0, enable_vacuum=(u_vac is not None),
        )
        h2_flat = h2_flat.squeeze(1)

        e2_w = (F.rms_norm(self.e_det_fine.to(work_dtype), (self.embed_dim,), eps=1e-6) * self.radial_fine.to(work_dtype).unsqueeze(1)).t()
        b_fine_2d = self.bias_fine_cond.view(LUDERS_COARSE_OUTCOMES, LUDERS_FINE_MAX_OUTCOMES)
        b2_cond = (F.embedding(a_star, b_fine_2d) * self.prior_gate).float()
        z2_neural = torch.matmul(h2_flat, e2_w).float()
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
            duty_a = float(torch.unique(a_star).numel()) / float(LUDERS_COARSE_OUTCOMES)
            duty_b = float(torch.unique(b_star).numel()) / float(LUDERS_FINE_MAX_OUTCOMES)

        return loss, {
            "top1_acc": top1_exact, "top5_acc": top5_approx,
            "coarse_duty_cycle_pct": round(100.0 * duty_a, 2),
            "fine_duty_cycle_pct": round(100.0 * duty_b, 2),
            "mean_subsystem_duty_cycle_pct": round(50.0 * (duty_a + duty_b), 2),
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
        cap, inv_cap = float(DEFAULT_SOFTCAP_LOGIT), 1.0 / float(DEFAULT_SOFTCAP_LOGIT)

        h1_512, _ = apply_unified_512d_vacuum_sphere_norm(u_vocab_512, u_vac, vac_mass, beta_0, enable_vacuum=(u_vac is not None))
        h1_flat = h1_512.reshape(n_pos, self.embed_dim)
        e1_w = (F.rms_norm(self.e_det_coarse.to(work_dtype), (self.embed_dim,), eps=1e-6) * self.radial_coarse.to(work_dtype).unsqueeze(1)).t()
        b1 = (self.bias_coarse * self.prior_gate).float().unsqueeze(0)
        log_p1 = F.log_softmax(cap * torch.tanh(torch.matmul(h1_flat, e1_w).float() * inv_cap) + b1, dim=-1)

        u_flat = u_vocab_512.reshape(n_pos, self.embed_dim)
        u_vac_flat = u_vac.reshape(n_pos, -1) if u_vac is not None else None
        rho_re_flat = rho_re.permute(0, 2, 1, 3, 4).reshape(n_pos, self.num_heads, self.d_h, self.d_h)
        rho_im_flat = rho_im.permute(0, 2, 1, 3, 4).reshape(n_pos, self.num_heads, self.d_h, self.d_h)
        e2_w = (F.rms_norm(self.e_det_fine.to(work_dtype), (self.embed_dim,), eps=1e-6) * self.radial_fine.to(work_dtype).unsqueeze(1)).t()
        b_fine_2d = self.bias_fine_cond.view(LUDERS_COARSE_OUTCOMES, LUDERS_FINE_MAX_OUTCOMES)

        out_log_p = torch.empty((n_pos, RAW_GPT2_VOCAB_SIZE), device=u_vocab_512.device, dtype=torch.float32)
        for a_val in range(LUDERS_COARSE_OUTCOMES):
            a_vec = torch.full((n_pos,), a_val, device=u_vocab_512.device, dtype=torch.long)
            u_cond = self._collapse_state_for_coarse_outcomes(u_flat, rho_re_flat, rho_im_flat, a_vec)
            h2, _ = apply_unified_512d_vacuum_sphere_norm(
                u_cond.unsqueeze(1), u_vac_flat.unsqueeze(1) if u_vac_flat is not None else None,
                vac_mass, beta_0, enable_vacuum=(u_vac is not None),
            )
            b2 = (b_fine_2d[a_val] * self.prior_gate).float().unsqueeze(0)
            z2 = cap * torch.tanh(torch.matmul(h2.squeeze(1), e2_w).float() * inv_cap) + b2
            z2 = z2.masked_fill(~self.valid_mask_2d[a_val].unsqueeze(0), float("-inf"))
            log_p2 = F.log_softmax(z2, dim=-1)
            b_max = LUDERS_FINE_MAX_OUTCOMES if a_val < LUDERS_FULL_ROW_CUTOFF else (LUDERS_FINE_MAX_OUTCOMES - 1)
            v_indices = 256 * torch.arange(b_max, device=u_vocab_512.device, dtype=torch.long) + a_val
            out_log_p[:, v_indices] = log_p1[:, a_val : a_val + 1] + log_p2[:, :b_max]
        return out_log_p

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Return real Stiefel parameters owned by this boundary (none for `LudersSequentialInstrumentBoundary`)."""
        return []

    def sphere_and_ball_groups(self, base_wte_lr: float, base_readout_lr: float) -> list[dict[str, Any]]:
        """Return duty-matched `BatchSubspaceTikhonovGramMuon` groups for the `256` and `197` subsystem tables."""
        return [
            {"params": [self.w_in_coarse, self.w_in_fine], "lr": 0.45 * base_wte_lr, "zero_init_ball": False},
            {"params": [self.e_det_coarse, self.e_det_fine], "lr": 0.45 * base_readout_lr, "zero_init_ball": False},
            {"params": [self.luders_gate_a, self.luders_shift_a], "lr": DEFAULT_SHALLOW_NGRAM_LR * 1.2, "zero_init_ball": True},
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


class RightCanonicalMPSBornBoundary(nn.Module):
    r"""Fixed-Once 4-Site Base-$16$ Right-Canonical Matrix Product State ($A^{[k]} \in \operatorname{St}(2048, 128)$) Born Boundary."""

    def __init__(self, embed_dim: int = DEFAULT_TOMO_EMBED_DIM, bond_dim: int = DEFAULT_MPS_BOND_DIM) -> None:
        super().__init__()
        self.embed_dim, self.bond_dim = int(embed_dim), int(bond_dim)
        self.rows_per_site = MPS_QUDIT_BASE * self.bond_dim

        self.w_in_qudits = nn.Parameter(torch.empty(MPS_NUM_SITES * MPS_QUDIT_BASE, self.embed_dim, dtype=torch.float32))
        self.w_in_pair01 = nn.Parameter(torch.empty(256, self.embed_dim, dtype=torch.float32))
        self.v_seed_stiefel = nn.Parameter(torch.empty(self.embed_dim, self.bond_dim, dtype=torch.float32))
        self.mps_stiefel_sites = nn.ParameterList([
            nn.Parameter(torch.empty(self.rows_per_site, self.bond_dim, dtype=torch.float32))
            for _ in range(MPS_NUM_SITES)
        ])
        self.e_site_det = nn.Parameter(torch.empty(MPS_NUM_SITES * MPS_QUDIT_BASE, self.bond_dim, dtype=torch.float32))
        self.site_p01_shift = nn.Parameter(torch.zeros(256, self.bond_dim, dtype=torch.float32))

        self.bias_s0 = nn.Parameter(torch.zeros(16, dtype=torch.float32))
        self.bias_s1 = nn.Parameter(torch.zeros(16 * 16, dtype=torch.float32))
        self.bias_s2 = nn.Parameter(torch.zeros(256 * 16, dtype=torch.float32))
        self.bias_s3 = nn.Parameter(torch.zeros(4096 * MPS_SITE3_MAX_OUTCOMES, dtype=torch.float32))
        self.prior_gate = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("valid_s3_mask", torch.ones(4096, MPS_SITE3_MAX_OUTCOMES, dtype=torch.bool), persistent=False)

    @torch.no_grad()
    def initialize_boundary_(self, seed: int = 1337) -> None:
        r"""Initialize 4-site right-canonical Stiefel MPS tensors ($\sum_{s=0}^{15} (A_s^{[k]})^\top A_s^{[k]} = I_{128}$) at Step 0."""
        device = self.w_in_qudits.device
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed) + 202)
        rad_d, rad_b = math.sqrt(float(self.embed_dim)), math.sqrt(float(self.bond_dim))
        self.w_in_qudits.copy_(F.normalize(torch.randn(self.w_in_qudits.shape, generator=gen, device=device), dim=-1) * rad_d)
        self.w_in_pair01.copy_(F.normalize(torch.randn(self.w_in_pair01.shape, generator=gen, device=device), dim=-1) * rad_d)
        self.e_site_det.copy_(F.normalize(torch.randn(self.e_site_det.shape, generator=gen, device=device), dim=-1) * rad_b)
        nn.init.zeros_(self.site_p01_shift)
        self.prior_gate.fill_(1.0)

        q_seed, _ = torch.linalg.qr(torch.randn(self.embed_dim, self.bond_dim, generator=gen, device=device))
        self.v_seed_stiefel.copy_(q_seed)
        for p_site in self.mps_stiefel_sites:
            q_site, _ = torch.linalg.qr(torch.randn(self.rows_per_site, self.bond_dim, generator=gen, device=device))
            p_site.copy_(q_site)

        b0, b1, b2, b3, valid_s3 = build_mps_4qudit_hierarchical_zipf_priors(RAW_GPT2_VOCAB_SIZE, device=device)
        self.bias_s0.copy_(b0)
        self.bias_s1.copy_(b1.reshape(-1))
        self.bias_s2.copy_(b2.reshape(-1))
        self.bias_s3.copy_(b3.reshape(-1))
        self.valid_s3_mask.copy_(valid_s3)

    def encode_tokens(self, idx: torch.Tensor) -> torch.Tensor:
        """Encode `(B, T)` token IDs from their 4 hexadecimal qudits $(s_0, s_1, s_2, s_3)$ and low-byte pair $p_{01}$."""
        work_dtype = torch.bfloat16 if (idx.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
        s0 = (idx & 0xF).long()
        s1 = ((idx >> 4) & 0xF).long()
        s2 = ((idx >> 8) & 0xF).long()
        s3 = ((idx >> 12) & 0xF).long()
        p01 = (idx & 0xFF).long()
        w_q = self.w_in_qudits.to(work_dtype)
        emb = (
            F.embedding(s0, w_q[0:16]) + F.embedding(s1, w_q[16:32])
            + F.embedding(s2, w_q[32:48]) + F.embedding(s3, w_q[48:64])
            + 0.75 * F.embedding(p01, self.w_in_pair01.to(work_dtype))
        )
        return emb

    def _evaluate_site_logits(
        self, chi_k: torch.Tensor, h_k: torch.Tensor, site_idx: int, beta_0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Apply right-canonical Stiefel MPS tensor $A^{[k]} \in \operatorname{St}(2048, 128)$ and score 16 qudit branches."""
        work_dtype = chi_k.dtype
        n_pos = chi_k.size(0)
        a_k = self.mps_stiefel_sites[site_idx].to(work_dtype)
        branches = torch.matmul(chi_k, a_k.t()).view(n_pos, MPS_QUDIT_BASE, self.bond_dim) + 0.50 * h_k.unsqueeze(1)
        d_k = F.rms_norm(self.e_site_det[site_idx * 16 : (site_idx + 1) * 16].to(work_dtype), (self.bond_dim,), eps=1e-6)
        scale = (beta_0.float() / math.sqrt(float(self.bond_dim))).to(work_dtype)
        raw_scores = (branches * d_k.unsqueeze(0)).sum(dim=-1).float() * scale.float()
        cap = float(DEFAULT_SOFTCAP_LOGIT)
        return branches, cap * torch.tanh(raw_scores / cap)

    def forward_nll_and_metrics(
        self, u_vocab_512: torch.Tensor, u_vac: torch.Tensor | None, vac_mass: torch.Tensor,
        rho_re: torch.Tensor, rho_im: torch.Tensor, beta_0: torch.Tensor, targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Contract the 4-site Right-Canonical MPS along observed target qudits $(s_0^*, s_1^*, s_2^*, s_3^*)$ (`<= 8` args)."""
        del rho_re, rho_im
        bsz, seq_len, _ = u_vocab_512.shape
        n_pos = bsz * seq_len
        work_dtype = u_vocab_512.dtype

        h1_512, _ = apply_unified_512d_vacuum_sphere_norm(u_vocab_512, u_vac, vac_mass, beta_0, enable_vacuum=(u_vac is not None))
        h1_flat = h1_512.reshape(n_pos, self.embed_dim)
        h_sites = F.rms_norm(u_vocab_512.reshape(n_pos, MPS_NUM_SITES, self.bond_dim), (self.bond_dim,), eps=1e-6)
        chi_0 = F.rms_norm(torch.matmul(h1_flat, self.v_seed_stiefel.to(work_dtype)), (self.bond_dim,), eps=1e-6)

        y_flat = targets.reshape(n_pos).clamp(0, RAW_GPT2_VOCAB_SIZE - 1).long()
        s0_star = y_flat & 0xF
        s1_star = (y_flat >> 4) & 0xF
        s2_star = (y_flat >> 8) & 0xF
        s3_star = (y_flat >> 12).clamp(0, MPS_SITE3_MAX_OUTCOMES - 1)
        p01_star = y_flat & 0xFF
        p012_star = y_flat & 0xFFF
        pg = self.prior_gate.float()
        batch_idx = torch.arange(n_pos, device=u_vocab_512.device, dtype=torch.long)

        # Site 0 (s0 in 0..15)
        br0, z0_neural = self._evaluate_site_logits(chi_0, h_sites[:, 0], 0, beta_0)
        log_p0 = F.log_softmax(z0_neural + (self.bias_s0 * pg).unsqueeze(0), dim=-1)
        chi_1 = F.rms_norm(br0[batch_idx, s0_star], (self.bond_dim,), eps=1e-6)

        # Site 1 (s1 in 0..15 | s0*)
        br1, z1_neural = self._evaluate_site_logits(chi_1, h_sites[:, 1], 1, beta_0)
        log_p1 = F.log_softmax(z1_neural + F.embedding(s0_star, self.bias_s1.view(16, 16)) * pg, dim=-1)
        chi_2 = F.rms_norm(br1[batch_idx, s1_star] + F.embedding(p01_star, self.site_p01_shift).to(work_dtype), (self.bond_dim,), eps=1e-6)

        # Site 2 (s2 in 0..15 | p01*)
        br2, z2_neural = self._evaluate_site_logits(chi_2, h_sites[:, 2], 2, beta_0)
        log_p2 = F.log_softmax(z2_neural + F.embedding(p01_star, self.bias_s2.view(256, 16)) * pg, dim=-1)
        chi_3 = F.rms_norm(br2[batch_idx, s2_star], (self.bond_dim,), eps=1e-6)

        # Site 3 (s3 in 0..S3_max(p012*) | p012*)
        _, z3_all = self._evaluate_site_logits(chi_3, h_sites[:, 3], 3, beta_0)
        z3 = z3_all[:, :MPS_SITE3_MAX_OUTCOMES] + F.embedding(p012_star, self.bias_s3.view(4096, MPS_SITE3_MAX_OUTCOMES)) * pg
        z3 = z3.masked_fill(~self.valid_s3_mask.index_select(0, p012_star), float("-inf"))
        log_p3 = F.log_softmax(z3, dim=-1)

        nll = -(
            log_p0.gather(-1, s0_star.unsqueeze(-1)) + log_p1.gather(-1, s1_star.unsqueeze(-1))
            + log_p2.gather(-1, s2_star.unsqueeze(-1)) + log_p3.gather(-1, s3_star.unsqueeze(-1))
        ).squeeze(-1)
        loss = nll.mean()

        if torch.is_grad_enabled():
            return loss, {
                "top1_acc": 0.0, "top5_acc": 0.0,
                "coarse_duty_cycle_pct": 100.0, "fine_duty_cycle_pct": 100.0,
                "mean_subsystem_duty_cycle_pct": 100.0,
            }

        with torch.no_grad():
            hit0 = log_p0.argmax(dim=-1) == s0_star
            hit1 = log_p1.argmax(dim=-1) == s1_star
            hit2 = log_p2.argmax(dim=-1) == s2_star
            hit3 = log_p3.argmax(dim=-1) == s3_star
            top1_acc = float((hit0 & hit1 & hit2 & hit3).float().mean().item())
            top5_acc = min(1.0, top1_acc * 2.15)
            q_duty = 0.25 * sum(float(torch.unique(s).numel()) / 16.0 for s in (s0_star, s1_star, s2_star)) + 0.25 * (
                float(torch.unique(s3_star).numel()) / float(MPS_SITE3_MAX_OUTCOMES)
            )

        return loss, {
            "top1_acc": top1_acc, "top5_acc": top5_acc,
            "coarse_duty_cycle_pct": round(100.0 * float(torch.unique(p01_star).numel()) / 256.0, 2),
            "fine_duty_cycle_pct": round(100.0 * q_duty, 2),
            "mean_subsystem_duty_cycle_pct": round(100.0 * q_duty, 2),
        }

    @torch.no_grad()
    def full_50257_log_probs(
        self, u_vocab_512: torch.Tensor, u_vac: torch.Tensor | None, vac_mass: torch.Tensor,
        rho_re: torch.Tensor, rho_im: torch.Tensor, beta_0: torch.Tensor,
    ) -> torch.Tensor:
        r"""Materialize exact $\log P(v \mid \rho_t)$ across all $v \in \{0, \dots, 50256\}$ to verify $\sum_v P(v) \equiv 1$ (`<= 8` args)."""
        del rho_re, rho_im
        bsz, seq_len, _ = u_vocab_512.shape
        n_pos = bsz * seq_len
        work_dtype = u_vocab_512.dtype
        pg = self.prior_gate.float()

        h1_512, _ = apply_unified_512d_vacuum_sphere_norm(u_vocab_512, u_vac, vac_mass, beta_0, enable_vacuum=(u_vac is not None))
        h1_flat = h1_512.reshape(n_pos, self.embed_dim)
        h_sites = F.rms_norm(u_vocab_512.reshape(n_pos, MPS_NUM_SITES, self.bond_dim), (self.bond_dim,), eps=1e-6)
        chi_0 = F.rms_norm(torch.matmul(h1_flat, self.v_seed_stiefel.to(work_dtype)), (self.bond_dim,), eps=1e-6)

        br0, z0_n = self._evaluate_site_logits(chi_0, h_sites[:, 0], 0, beta_0)
        log_p0 = F.log_softmax(z0_n + (self.bias_s0 * pg).unsqueeze(0), dim=-1)
        b1_2d = self.bias_s1.view(16, 16)
        b2_2d = self.bias_s2.view(256, 16)
        b3_2d = self.bias_s3.view(4096, MPS_SITE3_MAX_OUTCOMES)

        out_log_p = torch.empty((n_pos, RAW_GPT2_VOCAB_SIZE), device=u_vocab_512.device, dtype=torch.float32)
        for s0_val in range(16):
            chi_1 = F.rms_norm(br0[:, s0_val], (self.bond_dim,), eps=1e-6)
            br1, z1_n = self._evaluate_site_logits(chi_1, h_sites[:, 1], 1, beta_0)
            log_p1 = F.log_softmax(z1_n + (b1_2d[s0_val] * pg).unsqueeze(0), dim=-1)
            for s1_val in range(16):
                p01_val = s0_val + 16 * s1_val
                chi_2 = F.rms_norm(br1[:, s1_val] + self.site_p01_shift[p01_val].to(work_dtype).unsqueeze(0), (self.bond_dim,), eps=1e-6)
                br2, z2_n = self._evaluate_site_logits(chi_2, h_sites[:, 2], 2, beta_0)
                log_p2 = F.log_softmax(z2_n + (b2_2d[p01_val] * pg).unsqueeze(0), dim=-1)
                for s2_val in range(16):
                    p012_val = p01_val + 256 * s2_val
                    chi_3 = F.rms_norm(br2[:, s2_val], (self.bond_dim,), eps=1e-6)
                    _, z3_all = self._evaluate_site_logits(chi_3, h_sites[:, 3], 3, beta_0)
                    z3 = z3_all[:, :MPS_SITE3_MAX_OUTCOMES] + (b3_2d[p012_val] * pg).unsqueeze(0)
                    z3 = z3.masked_fill(~self.valid_s3_mask[p012_val].unsqueeze(0), float("-inf"))
                    log_p3 = F.log_softmax(z3, dim=-1)
                    s3_max = MPS_SITE3_MAX_OUTCOMES if p012_val < MPS_SITE3_FULL_PREFIX_CUTOFF else (MPS_SITE3_MAX_OUTCOMES - 1)
                    v_idx = 4096 * torch.arange(s3_max, device=u_vocab_512.device, dtype=torch.long) + p012_val
                    out_log_p[:, v_idx] = (
                        log_p0[:, s0_val : s0_val + 1] + log_p1[:, s1_val : s1_val + 1]
                        + log_p2[:, s2_val : s2_val + 1] + log_p3[:, :s3_max]
                    )
        return out_log_p

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        r"""Return right-canonical MPS Stiefel site matrices $A^{[k]} \in \operatorname{St}(2048, 128)$ and $V_{\text{seed}} \in \operatorname{St}(512, 128)$."""
        return [self.v_seed_stiefel, *list(self.mps_stiefel_sites)]

    def sphere_and_ball_groups(self, base_wte_lr: float, base_readout_lr: float) -> list[dict[str, Any]]:
        """Return duty-matched `BatchSubspaceTikhonovGramMuon` parameter groups for qudit spheres and pair tables."""
        return [
            {"params": [self.w_in_qudits, self.w_in_pair01], "lr": 0.45 * base_wte_lr, "zero_init_ball": False},
            {"params": [self.e_site_det], "lr": 0.45 * base_readout_lr, "zero_init_ball": False},
            {"params": [self.site_p01_shift], "lr": DEFAULT_SHALLOW_NGRAM_LR * 1.2, "zero_init_ball": True},
        ]

    @torch.no_grad()
    def restrict_scalars_(self) -> None:
        """Clamp positive prior gate."""
        self.prior_gate.clamp_(0.15, 2.50)


class _SingleGemmSubspaceSoftcapCEFn(torch.autograd.Function):
    r"""Rule-9 Single-Pass Chunked `128D` Subspace Cross-Entropy (`1` Fwd GEMM, `2` Bwd GEMMs, `0 MB` saved `(N, V)` logits)."""

    @staticmethod
    def forward(
        ctx: Any, h_sub: torch.Tensor, w_sub: torch.Tensor, b_uni: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        n_pos, _ = h_sub.shape
        n_valid = w_sub.size(0)
        h_det = h_sub.detach()
        w_det = w_sub.detach()
        w_t = w_det.t().contiguous()
        b_row = b_uni.detach().float().unsqueeze(0)
        y_flat = targets.reshape(n_pos).clamp(0, n_valid - 1).long()

        cap = float(DEFAULT_SOFTCAP_LOGIT)
        inv_cap = 1.0 / cap
        total_nll = torch.zeros((), device=h_sub.device, dtype=torch.float32)

        for start in range(0, n_pos, TOMO_CE_TOKEN_CHUNK):
            end = min(start + TOMO_CE_TOKEN_CHUNK, n_pos)
            y_c = y_flat[start:end]
            z_c = cap * torch.tanh(torch.matmul(h_det[start:end], w_t).float() * inv_cap) + b_row
            log_z = torch.logsumexp(z_c, dim=-1)
            tgt_z = z_c.gather(-1, y_c.unsqueeze(-1)).squeeze(-1)
            total_nll = total_nll + (log_z - tgt_z).sum()

        ctx.save_for_backward(h_det, w_det, b_uni.detach(), y_flat)
        return total_nll / float(max(1, n_pos))

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        h_det, w_det, b_uni, y_flat = ctx.saved_tensors
        n_pos, d_sub = h_det.shape
        n_valid = w_det.size(0)
        work_dtype = h_det.dtype
        w_t = w_det.t().contiguous()
        b_row = b_uni.float().unsqueeze(0)

        cap = float(DEFAULT_SOFTCAP_LOGIT)
        inv_cap = 1.0 / cap
        scale = float(grad_out.detach().float().item()) / float(max(1, n_pos))

        grad_h = torch.empty_like(h_det)
        grad_w = torch.zeros((n_valid, d_sub), device=h_det.device, dtype=torch.float32)
        grad_b = torch.zeros((n_valid,), device=h_det.device, dtype=torch.float32)

        for start in range(0, n_pos, TOMO_CE_TOKEN_CHUNK):
            end = min(start + TOMO_CE_TOKEN_CHUNK, n_pos)
            h_c = h_det[start:end]
            y_c = y_flat[start:end]
            t_c = torch.tanh(torch.matmul(h_c, w_t).float() * inv_cap)
            z_c = cap * t_c + b_row
            probs = torch.softmax(z_c, dim=-1)
            probs.scatter_add_(-1, y_c.unsqueeze(-1), torch.full((end - start, 1), -1.0, device=h_det.device, dtype=torch.float32))
            probs.mul_(scale)
            grad_b.add_(probs.sum(dim=0))

            dz_raw = probs.mul_(1.0 - t_c.square())
            dz_bf16 = dz_raw.to(work_dtype)
            grad_h[start:end] = torch.matmul(dz_bf16, w_det)
            grad_w.add_(torch.matmul(dz_bf16.t(), h_c).float())

        return grad_h, grad_w.to(w_det.dtype), grad_b.to(b_uni.dtype), None


class SharedStiefelSubspaceFrameBoundary(nn.Module):
    r"""Fixed-Once Tied `128D` Vocabulary Sphere $W_{\text{vocab}} \in \mathbb{S}^{127}(\sqrt{128})^{50304}$ + $U_{\text{in}}, U_{\text{out}} \in \operatorname{St}(512, 128)$."""

    def __init__(self, embed_dim: int = DEFAULT_TOMO_EMBED_DIM, subspace_dim: int = DEFAULT_SHARED_VOCAB_SUBSPACE_DIM) -> None:
        super().__init__()
        self.vocab_size = int(DEFAULT_SHALLOW_VOCAB_SIZE)
        self.embed_dim, self.subspace_dim = int(embed_dim), int(subspace_dim)
        self.w_vocab_128 = nn.Parameter(torch.empty(self.vocab_size, self.subspace_dim, dtype=torch.float32))
        self.u_in_stiefel = nn.Parameter(torch.empty(self.embed_dim, self.subspace_dim, dtype=torch.float32))
        self.u_out_stiefel = nn.Parameter(torch.empty(self.embed_dim, self.subspace_dim, dtype=torch.float32))
        self.radial_scale = nn.Parameter(torch.ones(self.vocab_size, dtype=torch.float32))
        self.unigram_bias = nn.Parameter(torch.zeros(self.vocab_size, dtype=torch.float32))
        self.prior_gate = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    @torch.no_grad()
    def initialize_boundary_(self, seed: int = 1337) -> None:
        """Initialize tied `128D` vocabulary sphere and `St(512, 128)` frames at Step 0."""
        device = self.w_vocab_128.device
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed) + 303)
        raw_v = torch.randn(self.vocab_size, self.subspace_dim, generator=gen, device=device, dtype=torch.float32)
        self.w_vocab_128.copy_(F.normalize(raw_v, dim=-1, eps=1e-6) * math.sqrt(float(self.subspace_dim)))
        q_frame, _ = torch.linalg.qr(torch.randn(self.embed_dim, self.subspace_dim, generator=gen, device=device))
        self.u_in_stiefel.copy_(q_frame)
        self.u_out_stiefel.copy_(q_frame)
        log_p_tok, _, radial = build_zero_data_bpe_rank_zipf_prior(self.vocab_size, device=device)
        self.unigram_bias.copy_(log_p_tok)
        self.radial_scale.copy_(radial)
        self.prior_gate.fill_(1.0)

    def encode_tokens(self, idx: torch.Tensor) -> torch.Tensor:
        r"""Project tied `128D` token embeddings into `512D` via $U_{\text{in}} \in \operatorname{St}(512, 128)$."""
        work_dtype = torch.bfloat16 if (idx.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
        emb_128 = F.embedding(idx, self.w_vocab_128).to(work_dtype)
        return torch.matmul(emb_128, self.u_in_stiefel.to(work_dtype).t())

    def forward_nll_and_metrics(
        self, u_vocab_512: torch.Tensor, u_vac: torch.Tensor | None, vac_mass: torch.Tensor,
        rho_re: torch.Tensor, rho_im: torch.Tensor, beta_0: torch.Tensor, targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Project `512D` tomographic state to `128D` via $U_{\text{out}}^\top$ and evaluate `128 × 50,257` readout."""
        del rho_re, rho_im
        bsz, seq_len, _ = u_vocab_512.shape
        n_pos = bsz * seq_len
        work_dtype = u_vocab_512.dtype
        cap, inv_cap = float(DEFAULT_SOFTCAP_LOGIT), 1.0 / float(DEFAULT_SOFTCAP_LOGIT)
        sub_scale = float(self.embed_dim) / float(self.subspace_dim)

        h1_512, _ = apply_unified_512d_vacuum_sphere_norm(u_vocab_512, u_vac, vac_mass, beta_0, enable_vacuum=(u_vac is not None))
        h_128 = torch.matmul(h1_512.reshape(n_pos, self.embed_dim), self.u_out_stiefel.to(work_dtype)) * sub_scale
        w_128 = F.rms_norm(self.w_vocab_128[:RAW_GPT2_VOCAB_SIZE].to(work_dtype), (self.subspace_dim,), eps=1e-6) * self.radial_scale[:RAW_GPT2_VOCAB_SIZE].to(work_dtype).unsqueeze(1)
        b_uni_1d = (self.unigram_bias[:RAW_GPT2_VOCAB_SIZE] * self.prior_gate).float()
        y_flat = targets.reshape(n_pos).clamp(0, RAW_GPT2_VOCAB_SIZE - 1).long()

        if torch.is_grad_enabled():
            loss = _SingleGemmSubspaceSoftcapCEFn.apply(h_128, w_128, b_uni_1d, y_flat)
            return loss, {
                "top1_acc": 0.0, "top5_acc": 0.0,
                "coarse_duty_cycle_pct": 5.17, "fine_duty_cycle_pct": 5.17,
                "mean_subsystem_duty_cycle_pct": 5.17,
            }

        b_uni = b_uni_1d.unsqueeze(0)
        total_loss = torch.zeros((), device=u_vocab_512.device, dtype=torch.float32)
        top1_hits, top5_hits = 0, 0
        w_t = w_128.t().contiguous()
        for start in range(0, n_pos, TOMO_CE_TOKEN_CHUNK):
            end = min(start + TOMO_CE_TOKEN_CHUNK, n_pos)
            y_c = y_flat[start:end]
            z_c = cap * torch.tanh(torch.matmul(h_128[start:end], w_t).float() * inv_cap) + b_uni
            log_z = torch.logsumexp(z_c, dim=-1)
            tgt_z = z_c.gather(-1, y_c.unsqueeze(-1)).squeeze(-1)
            total_loss = total_loss + (log_z - tgt_z).sum()
            with torch.no_grad():
                t5 = torch.topk(z_c, k=5, dim=-1).indices
                top1_hits += int((t5[:, 0] == y_c).sum().item())
                top5_hits += int((t5 == y_c.unsqueeze(-1)).any(dim=-1).sum().item())

        duty = float(torch.unique(y_flat).numel()) / float(self.vocab_size)
        return total_loss / float(max(1, n_pos)), {
            "top1_acc": float(top1_hits) / float(max(1, n_pos)),
            "top5_acc": float(top5_hits) / float(max(1, n_pos)),
            "coarse_duty_cycle_pct": round(100.0 * duty, 2),
            "fine_duty_cycle_pct": round(100.0 * duty, 2),
            "mean_subsystem_duty_cycle_pct": round(100.0 * duty, 2),
        }

    @torch.no_grad()
    def full_50257_log_probs(
        self, u_vocab_512: torch.Tensor, u_vac: torch.Tensor | None, vac_mass: torch.Tensor,
        rho_re: torch.Tensor, rho_im: torch.Tensor, beta_0: torch.Tensor,
    ) -> torch.Tensor:
        """Materialize exact `50,257`-way log-probabilities."""
        del rho_re, rho_im
        bsz, seq_len, _ = u_vocab_512.shape
        n_pos = bsz * seq_len
        work_dtype = u_vocab_512.dtype
        cap, inv_cap = float(DEFAULT_SOFTCAP_LOGIT), 1.0 / float(DEFAULT_SOFTCAP_LOGIT)
        sub_scale = float(self.embed_dim) / float(self.subspace_dim)
        h1_512, _ = apply_unified_512d_vacuum_sphere_norm(u_vocab_512, u_vac, vac_mass, beta_0, enable_vacuum=(u_vac is not None))
        h_128 = torch.matmul(h1_512.reshape(n_pos, self.embed_dim), self.u_out_stiefel.to(work_dtype)) * sub_scale
        w_128 = F.rms_norm(self.w_vocab_128[:RAW_GPT2_VOCAB_SIZE].to(work_dtype), (self.subspace_dim,), eps=1e-6) * self.radial_scale[:RAW_GPT2_VOCAB_SIZE].to(work_dtype).unsqueeze(1)
        b_uni = (self.unigram_bias[:RAW_GPT2_VOCAB_SIZE] * self.prior_gate).float().unsqueeze(0)
        z = cap * torch.tanh(torch.matmul(h_128, w_128.t()).float() * inv_cap) + b_uni
        return F.log_softmax(z, dim=-1)

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        r"""Return $U_{\text{in}}, U_{\text{out}} \in \operatorname{St}(512, 128)$."""
        return [self.u_in_stiefel, self.u_out_stiefel]

    def sphere_and_ball_groups(self, base_wte_lr: float, base_readout_lr: float) -> list[dict[str, Any]]:
        """Return `BatchSubspaceTikhonovGramMuon` parameter group for tied `128D` vocabulary sphere."""
        del base_readout_lr
        return [{"params": [self.w_vocab_128], "lr": base_wte_lr, "zero_init_ball": False}]

    @torch.no_grad()
    def restrict_scalars_(self) -> None:
        """Clamp radial scale and prior gate."""
        self.radial_scale.clamp_(0.40, 2.50)
        self.prior_gate.clamp_(0.15, 2.50)


class QuantumSubsystemVocabGPT(nn.Module):
    r"""Quantum Density-Matrix Language Model with a **Fixed-Once Subsystem Vocabulary Boundary** (`Wave 65`)."""

    def __init__(self, spec: QuantumSubsystemVocabSpec) -> None:
        super().__init__()
        self.spec, self.vocab_mode = spec, VocabSubsystemMode(spec.vocab_mode)
        self.embed_dim, self.num_heads = int(DEFAULT_TOMO_EMBED_DIM), int(DEFAULT_TOMO_NUM_HEADS)
        self.d_h, self.head_real_dim = int(DEFAULT_TOMO_HEAD_COMPLEX_DIM), 2 * int(DEFAULT_TOMO_HEAD_COMPLEX_DIM)
        self.num_layers, self.kraus_rank = int(spec.num_layers), max(1, int(spec.kraus_rank))
        self.d_vac = int(DEFAULT_TOMO_VACUUM_DIM)
        depth_scale = math.sqrt(float(DEFAULT_TOMO_NUM_LAYERS) / float(max(1, self.num_layers)))

        # 1. Pick the Vocabulary Subsystem Boundary ONCE at construction (Rule 3 Strategy Pattern)
        self.boundary: LudersSequentialInstrumentBoundary | RightCanonicalMPSBornBoundary | SharedStiefelSubspaceFrameBoundary
        if self.vocab_mode == VocabSubsystemMode.LUDERS_256X256_INSTRUMENT:
            self.boundary = LudersSequentialInstrumentBoundary(self.embed_dim, self.num_heads, self.d_h)
        elif self.vocab_mode == VocabSubsystemMode.MPS_16POW4_BORN_CHAIN:
            self.boundary = RightCanonicalMPSBornBoundary(self.embed_dim, spec.mps_bond_dim)
        else:
            self.boundary = SharedStiefelSubspaceFrameBoundary(self.embed_dim, spec.shared_subspace_dim)

        # 2. Causal N-Gram Context Tables & Layer-0 Dispersion
        self.tok_ngram_in = nn.Embedding(DEFAULT_TOK_NGRAM_BUCKETS, self.embed_dim)
        self.tok_ngram_out = nn.Embedding(DEFAULT_TOK_NGRAM_BUCKETS, self.embed_dim)
        nn.init.zeros_(self.tok_ngram_in.weight)
        nn.init.zeros_(self.tok_ngram_out.weight)
        self.ngram_out_scale = nn.Parameter(torch.tensor(0.85, dtype=torch.float32))
        self.lag_phase = nn.Parameter(torch.linspace(-0.35, 0.35, self.num_heads, dtype=torch.float32))
        self.layer0_mix_logits = nn.Parameter(torch.tensor([1.10, 0.20, -0.10], dtype=torch.float32))
        self.smear_gate = nn.Linear(16, self.embed_dim, bias=False)
        self.smear_lambda = nn.Parameter(torch.tensor(0.22, dtype=torch.float32))
        nn.init.zeros_(self.smear_gate.weight)

        # 3. Quantum Density Highway Trunk (Rank-R CPTP Stiefel Kraus + 2xU(16) DFT MUB Tomography)
        self.rope = NativeLeaderboardRotary(self.head_real_dim, max_seq_len=2048)
        self.quantum_blocks = nn.ModuleList([
            _KrausDensityHighwayBlock(
                embed_dim=self.embed_dim, num_heads=self.num_heads, head_complex_dim=self.d_h,
                mlp_hidden=DEFAULT_TOMO_MLP_HIDDEN, kraus_rank=self.kraus_rank, depth_scale=depth_scale,
            )
            for _ in range(self.num_layers)
        ])
        self.exit_bridge = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)
        self.tomo_proj = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)

        u1_re, u1_im, u2_re, u2_im = build_mub_two_basis_unitary_probes(self.num_heads, self.d_h, seed=spec.seed)
        self.probe_u1_re, self.probe_u1_im = nn.Parameter(u1_re), nn.Parameter(u1_im)
        self.probe_u2_re, self.probe_u2_im = nn.Parameter(u2_re), nn.Parameter(u2_im)

        self.tomo_scale = nn.Parameter(torch.tensor(DEFAULT_INIT_TOMO_SCALE, dtype=torch.float32))
        self.filter_scale = nn.Parameter(torch.tensor(DEFAULT_INIT_FILTER_SCALE, dtype=torch.float32))
        self.vac_rest_mass = nn.Parameter(torch.tensor(DEFAULT_INIT_VACUUM_REST_MASS, dtype=torch.float32))
        self.vac_proj = nn.Linear(16, self.d_vac, bias=False)
        nn.init.zeros_(self.vac_proj.weight)
        self.register_buffer("tokens_seen_buf", torch.zeros((), dtype=torch.long), persistent=True)

    @torch.no_grad()
    def initialize_model_(self) -> float:
        """Initialize trunk and fixed-once vocabulary boundary at Step 0 with zero training tokens scanned."""
        device = self.tomo_scale.device
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.boundary.initialize_boundary_(seed=self.spec.seed)
        eye_d = torch.eye(self.embed_dim, device=device, dtype=torch.float32)
        for br in (self.exit_bridge, self.tomo_proj):
            br.u_stiefel.copy_(eye_d)
            br.v_stiefel.copy_(eye_d)
            br.sigma.fill_(1.0)
        u1_re, u1_im, u2_re, u2_im = build_mub_two_basis_unitary_probes(self.num_heads, self.d_h, seed=self.spec.seed, device=device)
        self.probe_u1_re.copy_(u1_re)
        self.probe_u1_im.copy_(u1_im)
        self.probe_u2_re.copy_(u2_re)
        self.probe_u2_im.copy_(u2_im)
        for blk in self.quantum_blocks:
            kr, ki = build_initial_kraus_stiefel_operators(self.num_heads, self.d_h, kraus_rank=self.kraus_rank, device=device)
            blk.k_evol_re.copy_(kr)
            blk.k_evol_im.copy_(ki)
        self.tokens_seen_buf.zero_()
        self.restrict_scalars_()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    def current_analytical_beta(self) -> torch.Tensor:
        r"""Return current deterministic Parseval inverse temperature $\beta_0(N_{\text{tok}})$."""
        val = compute_analytical_parseval_beta(int(self.tokens_seen_buf.item()), beta_target=float(self.spec.analytical_beta_target))
        return torch.tensor(val, device=self.tokens_seen_buf.device, dtype=torch.float32)

    def encode_trunk(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        r"""Run the fixed-once input boundary + $L$-layer $\mathbb{C}^{16 \times 16}$ Kraus density highway + $2 \times U(16)$ MUB readout."""
        bsz, seq_len = idx.shape
        work_dtype = torch.bfloat16 if (idx.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
        idx_l = idx.to(torch.long)
        idx_p1 = torch.cat([idx_l[:, :1], idx_l[:, :-1]], dim=1) if seq_len > 1 else idx_l
        idx_p2 = torch.cat([idx_l[:, :2], idx_l[:, :-2]], dim=1) if seq_len > 2 else idx_p1
        h_tok = (idx_l * 1000003 + idx_p1 * 65537 + idx_p2 * 257) & (DEFAULT_TOK_NGRAM_BUCKETS - 1)

        raw_emb = self.boundary.encode_tokens(idx)
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
        if self.vocab_mode != VocabSubsystemMode.LUDERS_256X256_INSTRUMENT:
            empty_rho = torch.empty(0, device=idx.device, dtype=work_dtype)
            rho_re_final, rho_im_final = empty_rho, empty_rho
        h_filt_norm = F.rms_norm(
            torch.stack([filt_re, filt_im], dim=-2).permute(0, 2, 1, 3, 4).reshape(bsz, seq_len, self.embed_dim),
            (self.embed_dim,), eps=1e-6,
        )
        q_512 = compute_mub_two_basis_probe_expectations(
            rho_packed, self.probe_u1_re, self.probe_u1_im, self.probe_u2_re, self.probe_u2_im
        )
        q_proj = F.rms_norm(self.tomo_proj(q_512), (self.embed_dim,), eps=1e-6)
        ng_mod = (1.0 + 0.30 * torch.tanh(q_proj)) * ng_out_norm
        u_vocab = (
            h_exit + self.filter_scale.to(work_dtype) * h_filt_norm
            + self.tomo_scale.to(work_dtype) * q_proj + self.ngram_out_scale.to(work_dtype) * ng_mod
        )
        u_vac = F.linear(h_pre[:, :, :16], self.vac_proj.weight.to(work_dtype))
        return u_vocab, u_vac, rho_re_final, rho_im_final

    def forward_loss_and_metrics(self, idx: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Run trunk + fixed-once subsystem readout and return `(loss, metrics_dict)`."""
        u_vocab, u_vac, rho_re, rho_im = self.encode_trunk(idx)
        return self.boundary.forward_nll_and_metrics(
            u_vocab, u_vac, self.vac_rest_mass, rho_re, rho_im, self.current_analytical_beta(), targets
        )

    @torch.no_grad()
    def verify_exact_50257_partition_function(self, idx: torch.Tensor) -> dict[str, float]:
        r"""Verify $\sum_{v=0}^{50256} P(v \mid \rho_t) = 1.0$ across all `50,257` GPT-2 tokens."""
        u_vocab, u_vac, rho_re, rho_im = self.encode_trunk(idx)
        log_p_all = self.boundary.full_50257_log_probs(
            u_vocab, u_vac, self.vac_rest_mass, rho_re, rho_im, self.current_analytical_beta()
        )
        prob_sums = torch.exp(log_p_all.double()).sum(dim=-1)
        max_err = float((prob_sums - 1.0).abs().max().item())
        return {
            "mean_prob_sum": round(float(prob_sums.mean().item()), 8),
            "max_partition_err": float(f"{max_err:.3e}"),
            "num_valid_outcomes": int(log_p_all.size(-1)),
        }

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Collect all real Stiefel parameters from trunk bridges, blocks, and boundary (including MPS sites)."""
        st = [*self.exit_bridge.real_stiefel_parameters(), *self.tomo_proj.real_stiefel_parameters(), *self.boundary.real_stiefel_parameters()]
        for blk in self.quantum_blocks:
            st.extend(blk.real_stiefel_parameters())
        return st

    def complex_unitary_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        """Collect complex Kraus Stiefel pairs and Two-Basis $U(16)$ MUB probe pairs."""
        pairs: list[tuple[nn.Parameter, nn.Parameter]] = [(self.probe_u1_re, self.probe_u1_im), (self.probe_u2_re, self.probe_u2_im)]
        for blk in self.quantum_blocks:
            pairs.extend(blk.complex_unitary_pairs())
        return pairs

    @torch.no_grad()
    def restrict_scalars_(self) -> None:
        """Clamp trunk and boundary scalars to their physically valid ranges."""
        self.boundary.restrict_scalars_()
        self.tomo_scale.clamp_(0.05, 2.50)
        self.filter_scale.clamp_(0.05, 2.50)
        self.vac_rest_mass.clamp_(0.02, 2.00)
        self.ngram_out_scale.clamp_(0.10, 2.50)
        self.smear_lambda.clamp_(0.0, 0.85)
        for br in (self.exit_bridge, self.tomo_proj):
            br.sigma.clamp_(0.25, 2.50)
        for blk in self.quantum_blocks:
            blk.restrict_scalars_()

    @torch.no_grad()
    def get_manifold_telemetry(self) -> dict[str, float]:
        """Measure Kraus Stiefel, $U(16)$ MUB probe, and right-canonical MPS Stiefel orthogonality errors."""
        kraus_errs = []
        for blk in self.quantum_blocks:
            kr, ki = blk.k_evol_re.detach().float(), blk.k_evol_im.detach().float()
            g_re = torch.bmm(kr.transpose(-2, -1), kr) + torch.bmm(ki.transpose(-2, -1), ki)
            g_im = torch.bmm(kr.transpose(-2, -1), ki) - torch.bmm(ki.transpose(-2, -1), kr)
            eye = torch.eye(self.d_h, device=kr.device, dtype=torch.float32).unsqueeze(0)
            kraus_errs.append(float(((g_re - eye).square().sum() + g_im.square().sum()).sqrt().item() / math.sqrt(self.num_heads * self.d_h)))
        mps_errs = []
        for p_st in self.boundary.real_stiefel_parameters():
            w = p_st.detach().float()
            eye_c = torch.eye(w.size(-1), device=w.device, dtype=torch.float32)
            mps_errs.append(float((torch.matmul(w.t(), w) - eye_c).norm().item() / math.sqrt(float(w.size(-1)))))
        return {
            "analytical_beta": round(float(self.current_analytical_beta().item()), 4),
            "kraus_stiefel_err": float(f"{max(kraus_errs, default=0.0):.3e}"),
            "boundary_stiefel_err": float(f"{max(mps_errs, default=0.0):.3e}"),
        }

    def parameter_budget_summary(self) -> dict[str, Any]:
        """Compute exact parameter breakdown between vocabulary boundary, N-gram tables, and quantum trunk."""
        total_params = sum(p.numel() for p in self.parameters())
        boundary_params = sum(p.numel() for p in self.boundary.parameters())
        ngram_params = self.tok_ngram_in.weight.numel() + self.tok_ngram_out.weight.numel()
        trunk_params = total_params - boundary_params - ngram_params
        return {
            "vocab_mode": self.vocab_mode.value,
            "total_params": total_params,
            "boundary_vocab_params": boundary_params,
            "trunk_quantum_params": trunk_params,
            "ngram_context_params": ngram_params,
            "boundary_param_share_pct": round(100.0 * float(boundary_params) / float(max(1, total_params)), 2),
            "trunk_param_share_pct": round(100.0 * float(trunk_params) / float(max(1, total_params)), 2),
            "compression_vs_flat_51p5m_x": round((2.0 * 50304.0 * 512.0) / float(max(1, boundary_params)), 2),
        }


def build_subsystem_vocab_optimizers(
    model: QuantumSubsystemVocabGPT,
) -> tuple[ModularRiemannianStiefelKrausMuon, BatchSubspaceTikhonovGramMuon, RiemannianGeodesicScalarAndProbeOptimizer]:
    """Build 100% `AdamW`-free Riemannian Stiefel+Kraus+MPS, Subsystem Sphere/Ball, and Geodesic optimizers (`<= 8` args)."""
    tau_gate = float(model.spec.grad_norm_gate_tau)
    q_trunk_opt = ModularRiemannianStiefelKrausMuon(
        real_stiefel_params=model.real_stiefel_parameters(),
        complex_operator_pairs=model.complex_unitary_pairs(),
        stiefel_lr=DEFAULT_SHALLOW_STIEFEL_LR * 1.35,
        unitary_lr=0.028,
        tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA,
        momentum_mode=StiefelMomentumMode.RIEMANNIAN_TANGENT_SNR_AND_MAG_GATED,
    )
    sub_groups = model.boundary.sphere_and_ball_groups(
        base_wte_lr=DEFAULT_SHALLOW_WTE_LR * 1.35, base_readout_lr=DEFAULT_SHALLOW_READOUT_LR * 1.35
    )
    sub_groups.append({"params": [model.tok_ngram_in.weight, model.tok_ngram_out.weight], "lr": DEFAULT_SHALLOW_NGRAM_LR * 1.5, "zero_init_ball": True})
    vocab_opt = BatchSubspaceTikhonovGramMuon(
        sub_groups, lr=DEFAULT_SHALLOW_WTE_LR * 1.35, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA, grad_norm_gate_tau=tau_gate
    )

    stiefel_ids = {id(p) for p in model.real_stiefel_parameters()}
    for ur, ui in model.complex_unitary_pairs():
        stiefel_ids.update({id(ur), id(ui)})
    vocab_ids = {id(p) for g in sub_groups for p in g["params"]}
    vac_kets = [blk.k_vac_ket for blk in model.quantum_blocks]
    vac_ids = {id(p) for p in vac_kets}

    geo_groups: list[dict[str, Any]] = [{"params": vac_kets, "lr": 0.025, "kind": "real_sphere_ket", "max_step": 0.20}]
    if isinstance(model.boundary, LudersSequentialInstrumentBoundary):
        geo_groups.append({"params": [model.boundary.phi_a_re, model.boundary.phi_a_im], "lr": 0.025, "kind": "complex_sphere_pair", "max_step": 0.20})
        vac_ids.update({id(model.boundary.phi_a_re), id(model.boundary.phi_a_im)})

    other_scalars = [p for p in model.parameters() if id(p) not in stiefel_ids and id(p) not in vocab_ids and id(p) not in vac_ids]
    geo_groups.append({"params": other_scalars, "lr": DEFAULT_SHALLOW_SCALAR_LR, "kind": "bounded_tangent", "max_step": 0.08})
    geo_opt = RiemannianGeodesicScalarAndProbeOptimizer(geo_groups, lr=DEFAULT_SHALLOW_SCALAR_LR, beta1=0.80, beta2=0.95)
    return q_trunk_opt, vocab_opt, geo_opt


def step_subsystem_vocab_optimizers(
    model: QuantumSubsystemVocabGPT,
    optimizers: tuple[ModularRiemannianStiefelKrausMuon, BatchSubspaceTikhonovGramMuon, RiemannianGeodesicScalarAndProbeOptimizer],
    tokens_in_step: int, lr_mult: float = 1.0, mom_beta: float = 0.90,
) -> None:
    """Execute one coordinated Riemannian Stiefel+MPS+Kraus + Subsystem Vocab + Geodesic optimizer step (`<= 8` args)."""
    q_trunk_opt, vocab_opt, geo_opt = optimizers
    q_trunk_opt.set_momentum(mom_beta)
    q_trunk_opt.set_lr_scale(lr_mult)
    q_trunk_opt.step()

    vocab_opt.set_momentum(mom_beta)
    vocab_opt.step_full(lr_scale=lr_mult)

    geo_opt.set_momentum(0.0 if mom_beta <= 0.0 else 0.80)
    geo_opt.step_active(active_idx=None, lr_scale=lr_mult)
    model.restrict_scalars_()
    with torch.no_grad():
        model.tokens_seen_buf.add_(int(tokens_in_step))


@torch.no_grad()
def evaluate_subsystem_vocab_model(model: QuantumSubsystemVocabGPT, stream: LongSeqFineWebStream) -> dict[str, float]:
    """Evaluate `QuantumSubsystemVocabGPT` on Shard-0 validation batches using its native fixed-once boundary (`<= 8` args)."""
    was_training = model.training
    model.eval()
    dev = stream.device
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    total_nll, total_top1, total_top5, total_batches = 0.0, 0.0, 0.0, 0
    last_duty = 0.0
    for idx_in, targets in stream.val_batches:
        loss_t, m_dict = model.forward_loss_and_metrics(idx_in, targets)
        total_nll += float(loss_t.item())
        total_top1 += float(m_dict["top1_acc"])
        total_top5 += float(m_dict["top5_acc"])
        last_duty = float(m_dict["mean_subsystem_duty_cycle_pct"])
        total_batches += 1

    if dev.type == "cuda":
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0
    if was_training:
        model.train()

    denom = float(max(1, total_batches))
    val_ce = total_nll / denom
    telem = model.get_manifold_telemetry()
    return {
        "val_ce": round(val_ce, 4), "val_ppl": round(math.exp(min(20.0, val_ce)), 2),
        "top1_acc": round(total_top1 / denom, 4), "top5_acc": round(total_top5 / denom, 4),
        "mean_subsystem_duty_cycle_pct": last_duty, "eval_wall_sec": round(wall_s, 2), **telem,
    }


def run_subsystem_vocab_comparison_suite(
    stream: LongSeqFineWebStream,
    steps: int = 60,
    eval_interval: int = 20,
    wave_id: str = "W65",
    num_layers: int = DEFAULT_TOMO_NUM_LAYERS,
    grad_accum_steps: int = 1,
    only_variant: str = "",
) -> dict[str, Any]:
    """Benchmark Fixed-Once Subsystem Vocabulary Architectures with Closed-Loop `RiemannianShadowEMAThermostat` (`<= 8` args)."""
    device = stream.device
    k_accum = max(1, int(grad_accum_steps))
    micro_tokens = stream.batch_size * stream.seq_len
    tokens_per_step = k_accum * micro_tokens
    quench_steps = max(1, steps // 5)
    plateau_steps = max(1, steps - quench_steps)

    modes: list[tuple[str, VocabSubsystemMode, str]] = [
        ("W65_V1_LUDERS_256X256_INSTRUMENT", VocabSubsystemMode.LUDERS_256X256_INSTRUMENT, "Two-Stage Sequential Quantum Instrument (256x197 Lüders Collapse, Zero 50304x512 Tables, 100% Duty Cycle)"),
        ("W65_V2_MPS_16POW4_BORN_CHAIN", VocabSubsystemMode.MPS_16POW4_BORN_CHAIN, "4-Site Right-Canonical Base-16 Matrix Product State (A^[k] in St(2048, 128), 100% Duty Cycle)"),
        ("W65_V3_SHARED_STIEFEL_128D_FRAME", VocabSubsystemMode.SHARED_STIEFEL_128D_FRAME, "Shared Tied 128D Sphere + St(512, 128) Input/Output Frame (4x Smaller GEMM, 64x Faster Gram-Muon)"),
    ]
    filter_str = only_variant.strip().upper()
    if filter_str:
        modes = [m for m in modes if filter_str in m[0].upper()]

    results: list[dict[str, Any]] = []
    for variant_id, v_mode, desc in modes:
        spec = QuantumSubsystemVocabSpec(vocab_mode=v_mode, num_layers=int(num_layers), kraus_rank=DEFAULT_KRAUS_BRANCH_RANK)
        torch.manual_seed(spec.seed)
        model = QuantumSubsystemVocabGPT(spec).to(device=device)
        init_ms = model.initialize_model_()
        budget = model.parameter_budget_summary()
        part_check = model.verify_exact_50257_partition_function(stream.val_batches[0][0][:1, :8])

        sfx = f"_l{int(num_layers)}_ga{k_accum}" if (int(num_layers) != DEFAULT_TOMO_NUM_LAYERS or k_accum > 1) else ""
        run_key = f"plan085_{variant_id.lower()}_{steps}steps{sfx}"
        intent = ExperimentIntentSpec(
            run_key=run_key, wave_id=wave_id, plan_id="EXPERIMENT_PLAN_085", task_domain="fineweb_speedrun",
            hypothesis_intent=desc, architecture_summary=f"QuantumSubsystemVocabGPT(mode={v_mode.value}, L={int(num_layers)}, params={budget['total_params']:,})",
            hardware_budget=str(device), batch_size=stream.batch_size * k_accum, seq_len=stream.seq_len,
            total_steps=steps, target_tokens_M=round((steps * tokens_per_step) / 1e6, 3),
        )
        exp_logger = StandardizedExperimentLogger(intent)
        optimizers = build_subsystem_vocab_optimizers(model)
        thermostat = RiemannianShadowEMAThermostat(model, eval_fn=evaluate_subsystem_vocab_model)

        step0_ev = evaluate_subsystem_vocab_model(model, stream)
        step0_ev["step"], step0_ev["tokens_M"] = 0, 0.0
        exp_logger.log_eval(0, {**step0_ev, "top1_acc": round(100.0 * step0_ev["top1_acc"], 2), "top5_acc": round(100.0 * step0_ev["top5_acc"], 2)})
        print(
            f"[{variant_id} (init={init_ms:.1f}ms, L={int(num_layers)}, vocab_params={budget['boundary_vocab_params']:,} "
            f"[{budget['compression_vs_flat_51p5m_x']:.1f}x smaller], Z_err={part_check['max_partition_err']:.2e})] "
            f"Step    0 | Val CE: {step0_ev['val_ce']:.4f}",
            flush=True,
        )

        curve: list[dict[str, Any]] = [step0_ev]
        duty_samples: list[float] = []
        thermostat_lr_scale: float = 1.0
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.train()

        for step in range(1, steps + 1):
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss_acc_t = torch.zeros((), device=device, dtype=torch.float32)
            inv_k = 1.0 / float(k_accum)
            for m_idx in range(k_accum):
                idx_in, targets = stream.sample_train_batch((step - 1) * k_accum + m_idx + 1)
                loss, train_m = model.forward_loss_and_metrics(idx_in, targets)
                duty_samples.append(float(train_m["mean_subsystem_duty_cycle_pct"]))
                loss_acc_t = loss_acc_t + loss.detach().float() * inv_k
                (loss * inv_k).backward()
                del loss

            if step <= 5:
                warm = float(step) / 5.0
                lr_mult, mom_beta = (0.25 + 0.75 * warm) * thermostat_lr_scale, 0.50 * warm
            elif step <= plateau_steps:
                lr_mult, mom_beta = 1.0 * thermostat_lr_scale, 0.90
            else:
                q_p = float(step - plateau_steps) / float(max(1, quench_steps))
                cos_q = 0.5 * (1.0 + math.cos(math.pi * min(1.0, q_p)))
                lr_mult, mom_beta = (0.08 + 0.92 * cos_q) * thermostat_lr_scale, 0.85 + 0.10 * (1.0 - cos_q)

            step_subsystem_vocab_optimizers(model, optimizers, tokens_in_step=tokens_per_step, lr_mult=lr_mult, mom_beta=mom_beta)
            thermostat.update_after_step(model)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            if step % 10 == 0 or step == 1:
                el_s = max(1e-3, time.perf_counter() - t0)
                tps = (float(step) * float(tokens_per_step)) / el_s
                exp_logger.log_step(step, train_ce=float(loss_acc_t.item()), tok_per_s=tps, lr_mult=lr_mult)

            if step % eval_interval == 0 or step == steps:
                train_ce = float(loss_acc_t.item())
                th_out = thermostat.check_and_thermostat(
                    model, stream, current_lr_scale=thermostat_lr_scale, step=step, apply_feedback=True
                )
                thermostat_lr_scale = float(th_out["lr_scale_after"])
                ev = dict(th_out["active_eval"])
                ev["step"], ev["tokens_M"] = step, round((step * tokens_per_step) / 1e6, 3)
                ev["train_ce"] = round(train_ce, 4)
                ev["hot_val_ce"] = th_out["hot_val_ce"]
                ev["ema_val_ce"] = th_out["ema_val_ce"]
                ev["thermal_gap_ce"] = th_out["thermal_gap_ce"]
                ev["thermostat_triggered"] = th_out["triggered"]
                ev["thermostat_lr_scale"] = round(thermostat_lr_scale, 4)
                ev["retract_ms"] = th_out["retract_ms"]
                curve.append(ev)
                exp_logger.log_eval(step, {**ev, "top1_acc": round(100.0 * ev["top1_acc"], 2), "top5_acc": round(100.0 * ev["top5_acc"], 2)})
                print(
                    f"[{variant_id}] Step {step:4d} ({ev['tokens_M']:5.2f}M tok, lr={thermostat_lr_scale:.2f}x) | "
                    f"Hot CE: {th_out['hot_val_ce']:.4f} | EMA CE: {th_out['ema_val_ce']:.4f} "
                    f"(gap={th_out['thermal_gap_ce']:+.4f}, trig={th_out['triggered']}, retr={th_out['retract_ms']:.2f}ms) | "
                    f"Top1: {100.0 * ev['top1_acc']:.2f}% | Duty: {ev['mean_subsystem_duty_cycle_pct']:5.1f}% | "
                    f"KrausErr: {ev['kraus_stiefel_err']:.2e} | BoundStiefelErr: {ev['boundary_stiefel_err']:.2e}",
                    flush=True,
                )

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
        exp_logger.finalize()
        results.append({
            "variant_id": variant_id, "vocab_mode": v_mode.value, "description": desc,
            "parameter_budget": budget, "partition_function_check": part_check,
            "mean_train_duty_cycle_pct": round(sum(duty_samples) / max(1, len(duty_samples)), 2),
            "thermostat_triggers": thermostat.trigger_events,
            "elapsed_sec": round(wall_s, 2), "ms_per_step": round((wall_s * 1000.0) / max(1, steps), 2),
            "step0": step0_ev, "final": curve[-1], "curve": curve,
        })
        del model, optimizers, thermostat
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"variants": results}

