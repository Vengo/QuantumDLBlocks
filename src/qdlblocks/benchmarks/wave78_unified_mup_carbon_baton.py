r"""Wave 78 `R4c + R5B` Quantum Baton Unification (`UnifiedMuPCarbonBatonLM`, `EXPERIMENT_PLAN_098.md`).

Provenance & Mathematical Rationale
-----------------------------------
Unites the two winning architectures from `Wave 77` (`EXPERIMENT_PLAN_097.md`) while resolving the
three bottlenecks surfaced by our `Wave 77/78` mechanistic & calibration audits (`Rules 1–8` of
`docs/SCRIPT_BEST_PRACTICES.md`):

1. **Purity-Conditioned Hybrid Walsh–Hadamard + Semantic Copy Gate (`PurityConditionedHybridWalshCopyGate`)**:
   - Combines deterministic `32`-complex (`64` real) Walsh–Hadamard exact token-ID kets (giving Step-0
     orthogonality on unseen rare IDs in `50,257`-vocab Fuzzy Induction) with learned contextual
     `SmearGate` kets via a convex gate $\sigma(\alpha_{\text{wh}})$.
   - Suppresses diffuse function-word (`" the"`) false-copy impostor scatter (`+2.015` logits on `m=0`
     Recall in `Wave 77` `R4c`) using the Rényi-2 / Herfindahl copy concentration
     $\kappa_t = \frac{\sum_{s<t} w_{\text{copy}}(t,s)^2}{\sum_{s<t} w_{\text{copy}}(t,s) + \varepsilon} \in [0, 1]$,
     which equals `~0.90` for unique entity matches ($m=1$) and `~0.057` for `15`-way `" the"` soup.

2. **Zero-Sync Subsystem Complementary Schur Gate (`ZeroSyncComplementarySchurGate`)**:
   - Implements Carbon's trace-preserving `AND` ($p_{\text{and}}$) + phase-preserving partial-trace
     `Disagreement` ($1 - p_{\text{and}}$) quantum instrument across $G = d // 16$ subsystems of
     $\mathbb{C}^{16}$ with **zero synchronous GPU-to-CPU `.item()` stalls** during `forward()`.

3. **Horizon-Pooled Strictly Causal $K=4, \chi=8$ Virtual MTP Bwd-Fwd MPS Sweep (`HorizonPooledCausalMTPMPSSweep`)**:
   - Pools all $K=4$ backward-and-forward smoothed virtual future sites $\sum_{k=0}^{K-1} w_k \tilde{\phi}_{t,k}$
     and supervises the smoothed sites directly against stop-grad intra-document $x_{t+1..t+4}$,
     giving `100%` of backward ($K \to 1$) and forward ($1 \to K$) MPS tensors active gradients with
     `0.0` future leakage at inference.

4. **Depth-Scaled U-Net $\mu\text{P}$ + `StiefelMuon` Highway (`UnifiedMuPCarbonBatonLM`)**:
   - Supports `4L` (`2+2`) and `6L` (`3+3`) U-Net skip topologies with `v_0` value residual anchoring,
     `XSA` carrier-orthogonal projection, and $1/\sqrt{L/4}$ depth-aware residual scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .architecture_xray_diagnostics import (
    FINEWEB_BOS_TOKEN_ID,
    VALID_FINEWEB_VOCAB_SIZE,
    DiagnosticModelAdapter,
)
from .fused_bf16_four_compartment_baton_lab import DEFAULT_TRACE_EPS, normalize_complex_ket
from .wave77_baton_ladder_models import (
    DEFAULT_ONLINE_UNI_EMA_DECAY,
    DEFAULT_ONLINE_UNI_PRIOR_SCALE,
    DEFAULT_W77_ATTN_SCALE,
    DEFAULT_W77_CE_CHUNK_TOKENS,
    DEFAULT_W77_LOGIT_SOFTCAP,
    DEFAULT_W77_ROPE_BASE,
    MUP_BASE_REAL_DIM,
    W77_HEAD_BASE_LR_AT_D256,
    W77_MUON_BASE_LR,
    W77_MUON_MOMENTUM,
    W77_MUON_MOMENTUM_INIT,
    W77_SCALAR_BASE_LR,
    W77_WTE_BASE_LR,
    WSD_COOLDOWN_FRACTION,
)
from .wave77_quantum_routing_primitives import (
    CAUSAL_MTP_HORIZON_K,
    CAUSAL_MTP_MPS_BOND_CHI,
    COMP_SCHUR_BEAM_RANK,
    COMP_SCHUR_TRACE_EPS,
    COMPLEX_HEAD_DIM,
    DEFAULT_COPY_BOSON_POWER,
    DEFAULT_COPY_GATE_INIT_BIAS,
    DEFAULT_COPY_MAX_LOGIT_GAIN,
    DEFAULT_COPY_VACUUM_INIT_LOGIT,
    build_bos_anchor_sink_doc_masks,
    explicit_complementary_schur_matrix_pair,
    zero_matrix_complementary_schur_probe_action,
)
from ..optim.muon import StiefelMuon, StiefelMuonConfig

# ==============================================================================
# Top-of-File Named Constants (Synchronized with EXPERIMENT_PLAN_098.md Section 3)
# ==============================================================================
WALSH_COPY_DIM: int = 64
WALSH_CYCLIC_SEEDS_16BIT: tuple[int, int, int] = (0xB496, 0xD2A9, 0xE563)
DEFAULT_WH_EXACT_PRIOR_BIAS: float = 0.50
DEFAULT_PURITY_CONC_INIT_LOGIT: float = 0.85
MTP_HORIZON_INIT_LOGITS: tuple[float, float, float, float] = (2.0, 0.6, 0.0, -0.4)
SM_SMOOTHED_MTP_AUX_WEIGHT: float = 0.05
DEPTH_RESIDUAL_INIT_SCALE: float = 1.0
DEFAULT_W78_MLP_EXPANSION: int = 3

CopyGateKind = Literal["semantic_only", "hybrid_walsh_purity"]
SchurGateKind = Literal["none", "complementary_zero_sync"]


class Wave78VariantKind(str, Enum):
    """Strategy selector for the `Wave 78` `R4c + R5B` unification ablation matrix (Rule 3)."""

    U0_R4C_BASE_4L_D256 = "w78_u0_r4c_base_4l_d256"
    U1_HYBRID_WH_COPY_4L_D256 = "w78_u1_hybrid_wh_copy_4l_d256"
    U2_R4C_COMP_SCHUR_4L_D256 = "w78_u2_r4c_comp_schur_4l_d256"
    U3_UNIFIED_R4C_R5B_4L_D256 = "w78_u3_unified_r4c_r5b_4l_d256"
    U4_UNIFIED_R4C_R5B_6L_D256 = "w78_u4_unified_r4c_r5b_6l_d256"


@dataclass(frozen=True)
class UnifiedMuPCarbonConfig:
    """Frozen structural config for `UnifiedMuPCarbonBatonLM` (`<= 8` fields, Rule 2)."""

    vocab_size: int = VALID_FINEWEB_VOCAB_SIZE
    complex_dim: int = 256
    num_layers: int = 4
    copy_mode: CopyGateKind = "hybrid_walsh_purity"
    schur_mode: SchurGateKind = "complementary_zero_sync"
    use_horizon_mtp_mps: bool = True
    mlp_expansion: int = DEFAULT_W78_MLP_EXPANSION
    logit_softcap: float = DEFAULT_W77_LOGIT_SOFTCAP


def build_symmetric_walsh_hadamard_codebook(
    vocab_size: int = VALID_FINEWEB_VOCAB_SIZE,
    num_features: int = WALSH_COPY_DIM,
) -> Tensor:
    """Build deterministic `Z_16`-symmetric unit-norm `GF(2)` Walsh–Hadamard codebook `(V, 64)` (`<= 8` args).

    Combines `16` single-bit masks `(1 << b)` with `3` complete `16`-shift cyclic orbits of balanced
    weight-8 `16`-bit polynomials (`WALSH_CYCLIC_SEEDS_16BIT`), ensuring every token-ID bit `0..15`
    participates in `25` out of `64` parity checks (`<wh(v) | wh(v)> == 1.0` for all `v`, including
    completely unseen rare IDs).
    """
    masks: list[int] = [(1 << b) for b in range(16)]
    for seed_mask in WALSH_CYCLIC_SEEDS_16BIT:
        for shift in range(16):
            rot = ((seed_mask << shift) | (seed_mask >> (16 - shift))) & 0xFFFF
            masks.append(rot)
    while len(masks) < num_features:
        masks.append(((len(masks) * 40503) & 0xFFFF) | 1)
    mask_t = torch.tensor(masks[:num_features], dtype=torch.long)  # (64,)
    bit_idx = torch.arange(16, dtype=torch.long)
    mask_bits = ((mask_t.unsqueeze(0) >> bit_idx.unsqueeze(1)) & 1).to(torch.float32)  # (16, 64)

    tok_ids = torch.arange(int(vocab_size), dtype=torch.long)
    tok_bits = ((tok_ids.unsqueeze(1) >> bit_idx.unsqueeze(0)) & 1).to(torch.float32)  # (V, 16)
    gf2_parities = tok_bits.matmul(mask_bits).to(torch.long) & 1  # (V, 64)
    signs = (1.0 - 2.0 * gf2_parities.to(torch.float32)) * (1.0 / math.sqrt(float(num_features)))
    return signs.to(torch.bfloat16)


# ==============================================================================
# Pillar 1: Purity-Conditioned Hybrid Walsh–Hadamard + Semantic Associative Copy
# ==============================================================================
class PurityConditionedHybridWalshCopyGate(nn.Module):
    """`<BOS>`-masked $p=3$ associative copy uniting Walsh–Hadamard exact-ID kets & Rényi-2 purity gating."""

    def __init__(
        self,
        sem_ket_dim: int,
        ctx_dim: int,
        vocab_size: int = VALID_FINEWEB_VOCAB_SIZE,
        copy_mode: CopyGateKind = "hybrid_walsh_purity",
        boson_power: float = DEFAULT_COPY_BOSON_POWER,
        max_logit_gain: float = DEFAULT_COPY_MAX_LOGIT_GAIN,
    ) -> None:
        super().__init__()
        self.sem_ket_dim = int(sem_ket_dim)
        self.ctx_dim = int(ctx_dim)
        self.copy_mode: CopyGateKind = copy_mode
        self.boson_power = float(boson_power)
        self.max_logit_gain = float(max_logit_gain)

        real_sem = 2 * self.sem_ket_dim
        self.q_sem_proj = nn.Linear(real_sem, real_sem, bias=False)
        self.k_sem_proj = nn.Linear(real_sem, real_sem, bias=False)
        nn.init.eye_(self.q_sem_proj.weight)
        nn.init.eye_(self.k_sem_proj.weight)

        if copy_mode == "hybrid_walsh_purity":
            self.register_buffer(
                "wh_codebook",
                build_symmetric_walsh_hadamard_codebook(vocab_size, WALSH_COPY_DIM),
                persistent=False,
            )
            init_m = math.log(math.expm1(1.0))
            self.wh_metric_logits: nn.Parameter | None = nn.Parameter(
                torch.full((WALSH_COPY_DIM,), init_m, dtype=torch.float32)
            )
            self.wh_blend_logit: nn.Parameter | None = nn.Parameter(
                torch.tensor(DEFAULT_WH_EXACT_PRIOR_BIAS, dtype=torch.float32)
            )
            self.purity_conc_logit: nn.Parameter | None = nn.Parameter(
                torch.tensor(DEFAULT_PURITY_CONC_INIT_LOGIT, dtype=torch.float32)
            )
            gate_in_dim = self.ctx_dim + 3
        else:
            self.wh_metric_logits = None
            self.wh_blend_logit = None
            self.purity_conc_logit = None
            gate_in_dim = self.ctx_dim

        self.ctx_gate_proj = nn.Linear(gate_in_dim, 1, bias=True)
        nn.init.zeros_(self.ctx_gate_proj.weight)
        nn.init.constant_(self.ctx_gate_proj.bias, DEFAULT_COPY_GATE_INIT_BIAS)

        self.vacuum_logit = nn.Parameter(
            torch.tensor(DEFAULT_COPY_VACUUM_INIT_LOGIT, dtype=torch.float32)
        )

    def _compute_pboson_prev_overlap(
        self, q_re: Tensor, q_im: Tensor, k_re: Tensor, k_im: Tensor
    ) -> Tensor:
        """Compute causal `|<q_t | k_{s-1}>|^{2p}` of shape `(B, T, T)` (`<= 8` args)."""
        prev_k_re = F.pad(k_re[:, :-1, :], (0, 0, 1, 0))
        prev_k_im = F.pad(k_im[:, :-1, :], (0, 0, 1, 0))
        ov_re = torch.matmul(q_re, prev_k_re.transpose(-2, -1)) + torch.matmul(
            q_im, prev_k_im.transpose(-2, -1)
        )
        ov_im = torch.matmul(q_im, prev_k_re.transpose(-2, -1)) - torch.matmul(
            q_re, prev_k_im.transpose(-2, -1)
        )
        ov_sq = (ov_re.float().square() + ov_im.float().square()).clamp(0.0, 1.0)
        return ov_sq.pow(self.boson_power) if abs(self.boson_power - 1.0) > 1e-5 else ov_sq

    def forward(
        self,
        idx: Tensor,
        sem_ket_re: Tensor,
        sem_ket_im: Tensor,
        ctx_feat: Tensor,
        doc_same: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return `(copy_log_odds, copy_conf)` of shapes `(B, T, T)` and `(B, T)` (`<= 8` args)."""
        _, seq_len = idx.shape
        d_s = self.sem_ket_dim
        sem_pack = torch.cat([sem_ket_re, sem_ket_im], dim=-1)
        qs_raw = self.q_sem_proj(sem_pack)
        ks_raw = self.k_sem_proj(sem_pack)
        qs_re, qs_im = normalize_complex_ket(qs_raw[..., :d_s], qs_raw[..., d_s:])
        ks_re, ks_im = normalize_complex_ket(ks_raw[..., :d_s], ks_raw[..., d_s:])
        sem_sharp = self._compute_pboson_prev_overlap(qs_re, qs_im, ks_re, ks_im)

        if self.copy_mode == "hybrid_walsh_purity":
            assert self.wh_metric_logits is not None and self.wh_blend_logit is not None
            safe_idx = idx.clamp(0, self.wh_codebook.size(0) - 1)
            wh_raw = F.embedding(safe_idx, self.wh_codebook).float()
            metric_sqrt = torch.sqrt(F.softplus(self.wh_metric_logits.float()).clamp_min(1.0e-5))
            wh_ket = F.normalize(wh_raw * metric_sqrt, dim=-1, eps=1.0e-6).to(ctx_feat.dtype)
            prev_wh = F.pad(wh_ket[:, :-1, :], (0, 0, 1, 0))
            wh_ov_sq = torch.matmul(wh_ket, prev_wh.transpose(-2, -1)).float().square().clamp(0.0, 1.0)
            wh_sharp = (
                wh_ov_sq.pow(self.boson_power) if abs(self.boson_power - 1.0) > 1e-5 else wh_ov_sq
            )
            w_wh = torch.sigmoid(self.wh_blend_logit.float())
            sharp_ov = w_wh * wh_sharp + (1.0 - w_wh) * sem_sharp
        else:
            sharp_ov = sem_sharp

        t_idx = torch.arange(seq_len, device=idx.device).unsqueeze(1)
        s_idx = torch.arange(seq_len, device=idx.device).unsqueeze(0)
        strict_past = ((t_idx > s_idx) & (s_idx >= 1)).unsqueeze(0)
        doc_prev = F.pad(doc_same[:, :, :-1], (1, 0), value=False)
        valid_pair = (strict_past & doc_same & doc_prev).to(torch.float32)

        masked_scores = sharp_ov * valid_pair
        eps_vac = F.softplus(self.vacuum_logit.float()).clamp_min(1.0e-4)
        w_copy = masked_scores / (masked_scores.sum(dim=-1, keepdim=True) + eps_vac)
        copy_conf = w_copy.sum(dim=-1)

        if self.copy_mode == "hybrid_walsh_purity":
            assert self.purity_conc_logit is not None
            copy_purity = w_copy.square().sum(dim=-1)
            herf_conc = copy_purity / copy_conf.clamp_min(1.0e-5)
            stats_cat = torch.stack([copy_conf, copy_purity, herf_conc], dim=-1).to(ctx_feat.dtype)
            gate_in = torch.cat([ctx_feat, stats_cat], dim=-1)
            raw_gain = self.max_logit_gain * torch.sigmoid(self.ctx_gate_proj(gate_in).float())
            conc_mix = torch.sigmoid(self.purity_conc_logit.float())
            eff_scale = conc_mix * herf_conc.unsqueeze(-1) + (1.0 - conc_mix)
            copy_log_odds = (raw_gain * eff_scale * w_copy).to(ctx_feat.dtype)
        else:
            raw_gain = self.max_logit_gain * torch.sigmoid(self.ctx_gate_proj(ctx_feat).float())
            copy_log_odds = (raw_gain * w_copy).to(ctx_feat.dtype)

        return copy_log_odds, copy_conf.to(ctx_feat.dtype)


# ==============================================================================
# Pillar 2: Zero-Sync Subsystem-Blocked ($d_h = 16$) Complementary Schur Gate
# ==============================================================================
class ZeroSyncComplementarySchurGate(nn.Module):
    """Trace-preserving Complementary Schur Gate with zero GPU-to-CPU `.item()` syncs in `forward()`."""

    def __init__(
        self,
        real_dim: int,
        beam_rank: int = COMP_SCHUR_BEAM_RANK,
    ) -> None:
        super().__init__()
        if real_dim % (2 * COMPLEX_HEAD_DIM) != 0:
            raise ValueError(f"real_dim ({real_dim}) must be divisible by {2 * COMPLEX_HEAD_DIM}")
        self.real_dim = int(real_dim)
        self.num_groups = self.real_dim // (2 * COMPLEX_HEAD_DIM)
        self.beam_rank = int(beam_rank)

        self.beam_a_proj = nn.Linear(self.real_dim, self.beam_rank * self.real_dim, bias=False)
        self.beam_b_proj = nn.Linear(self.real_dim, self.beam_rank * self.real_dim, bias=False)
        self.probe_proj = nn.Linear(self.real_dim, self.real_dim, bias=False)
        self.even_odd_out_proj = nn.Linear(2 * self.real_dim, self.real_dim, bias=False)

        std = 0.5 / math.sqrt(float(self.real_dim))
        nn.init.normal_(self.beam_a_proj.weight, std=std)
        nn.init.normal_(self.beam_b_proj.weight, std=std)
        nn.init.eye_(self.probe_proj.weight)
        nn.init.zeros_(self.even_odd_out_proj.weight)

        self._last_ua_sample: Tensor | None = None
        self._last_ub_sample: Tensor | None = None
        self._last_pand_mean: Tensor | None = None

    def _pack_trace_unit_beams(self, raw: Tensor, bsz: int, seq_len: int) -> tuple[Tensor, Tensor]:
        """Reshape `(B, T, R * 2d)` into trace-1 `(B, T, G, R, 16)` `(u_re, u_im)` (`<= 8` args)."""
        g = self.num_groups
        r = self.beam_rank
        d_h = COMPLEX_HEAD_DIM
        shaped = raw.view(bsz, seq_len, g, r, 2 * d_h).float()
        u_re = shaped[..., :d_h]
        u_im = shaped[..., d_h:]
        tr = (u_re.square() + u_im.square()).sum(dim=(-2, -1), keepdim=True).clamp_min(DEFAULT_TRACE_EPS)
        inv_s = torch.rsqrt(tr)
        return u_re * inv_s, u_im * inv_s

    @torch.no_grad()
    def snapshot_purity_metrics(self) -> dict[str, float]:
        """Materialize `mean_p_and`, `even_purity`, and `odd_purity` ONLY at evaluation time (`<= 8` args)."""
        if self._last_pand_mean is None or self._last_ua_sample is None or self._last_ub_sample is None:
            return {"mean_p_and": 0.0, "even_purity": 0.0, "odd_purity": 0.0}
        rho_a = torch.matmul(self._last_ua_sample.transpose(-2, -1), self._last_ua_sample.conj())
        rho_b = torch.matmul(self._last_ub_sample.transpose(-2, -1), self._last_ub_sample.conj())
        rho_and, rho_comp, _ = explicit_complementary_schur_matrix_pair(rho_a, rho_b)
        pur_e = (rho_and.real.square() + rho_and.imag.square()).sum(dim=(-2, -1)).mean()
        pur_o = (rho_comp.real.square() + rho_comp.imag.square()).sum(dim=(-2, -1)).mean()
        return {
            "mean_p_and": round(float(self._last_pand_mean.item()), 5),
            "even_purity": round(float(pur_e.item()), 5),
            "odd_purity": round(float(pur_o.item()), 5),
        }

    def forward(self, state_a: Tensor, state_b: Tensor) -> Tensor:
        """Intersect `state_a` and `state_b` `(B, T, D)` with zero `.item()` CUDA stalls (`<= 8` args)."""
        bsz, seq_len, _ = state_a.shape
        g = self.num_groups
        d_h = COMPLEX_HEAD_DIM

        u_a_re, u_a_im = self._pack_trace_unit_beams(self.beam_a_proj(state_a), bsz, seq_len)
        u_b_re, u_b_im = self._pack_trace_unit_beams(self.beam_b_proj(state_b), bsz, seq_len)

        pr_raw = self.probe_proj(0.5 * (state_a + state_b)).view(bsz, seq_len, g, 2 * d_h).float()
        psi_re, psi_im = normalize_complex_ket(pr_raw[..., :d_h], pr_raw[..., d_h:])

        and_re, and_im, comp_re, comp_im, p_and = zero_matrix_complementary_schur_probe_action(
            u_a_re, u_a_im, u_b_re, u_b_im, psi_re, psi_im
        )
        if not self.training:
            self._last_pand_mean = p_and.detach().mean()
            self._last_ua_sample = torch.complex(u_a_re[:2, :16].detach(), u_a_im[:2, :16].detach())
            self._last_ub_sample = torch.complex(u_b_re[:2, :16].detach(), u_b_im[:2, :16].detach())

        p_w = p_and.unsqueeze(-1)
        even_pkt = (p_w * torch.cat([and_re, and_im], dim=-1)).reshape(bsz, seq_len, self.real_dim)
        odd_pkt = ((1.0 - p_w) * torch.cat([comp_re, comp_im], dim=-1)).reshape(
            bsz, seq_len, self.real_dim
        )
        joined = torch.cat([even_pkt, odd_pkt], dim=-1).to(state_a.dtype)
        return self.even_odd_out_proj(joined)


# ==============================================================================
# Pillar 3: Horizon-Pooled Causal K=4 Virtual MTP Bwd-Fwd MPS Sweep (Zero Sync)
# ==============================================================================
class HorizonPooledCausalMTPMPSSweep(nn.Module):
    """Strictly causal $K=4, \\chi=8$ backward-then-forward MPS sweep pooling all $k=0..K-1$ smoothed sites."""

    def __init__(
        self,
        real_dim: int,
        horizon_k: int = CAUSAL_MTP_HORIZON_K,
        bond_chi: int = CAUSAL_MTP_MPS_BOND_CHI,
    ) -> None:
        super().__init__()
        if real_dim % (2 * COMPLEX_HEAD_DIM) != 0:
            raise ValueError(f"real_dim ({real_dim}) must be divisible by {2 * COMPLEX_HEAD_DIM}")
        self.real_dim = int(real_dim)
        self.num_groups = self.real_dim // (2 * COMPLEX_HEAD_DIM)
        self.horizon_k = int(horizon_k)
        self.bond_chi = int(bond_chi)

        site_real = 2 * COMPLEX_HEAD_DIM
        self.virtual_rollout_proj = nn.Linear(self.real_dim, self.horizon_k * self.real_dim, bias=False)
        self.fwd_bond_init = nn.Linear(site_real, self.bond_chi, bias=False)
        self.bwd_site_to_bond = nn.Parameter(
            torch.randn(self.horizon_k, site_real, self.bond_chi) / math.sqrt(float(site_real))
        )
        self.bwd_bond_transfer = nn.Parameter(
            torch.eye(self.bond_chi).unsqueeze(0).repeat(self.horizon_k, 1, 1)
        )
        self.bwd_bond_to_diag = nn.Parameter(
            torch.randn(self.horizon_k, self.bond_chi, COMPLEX_HEAD_DIM) / math.sqrt(float(self.bond_chi))
        )
        self.fwd_bond_to_site = nn.Parameter(
            torch.randn(self.horizon_k, self.bond_chi, site_real) / math.sqrt(float(self.bond_chi))
        )
        n_fwd_trans = max(1, self.horizon_k - 1)
        self.fwd_site_to_bond = nn.Parameter(
            torch.randn(n_fwd_trans, site_real, self.bond_chi) / math.sqrt(float(site_real))
        )
        init_h = list(MTP_HORIZON_INIT_LOGITS[: self.horizon_k])
        while len(init_h) < self.horizon_k:
            init_h.append(-0.5)
        self.horizon_pool_logits = nn.Parameter(torch.tensor(init_h, dtype=torch.float32))
        self.mtp_out_proj = nn.Linear(self.real_dim, self.real_dim, bias=False)

        std = 0.5 / math.sqrt(float(self.real_dim))
        nn.init.normal_(self.virtual_rollout_proj.weight, std=std)
        nn.init.normal_(self.fwd_bond_init.weight, std=1.0 / math.sqrt(float(site_real)))
        nn.init.zeros_(self.mtp_out_proj.weight)
        self._last_mps_pand: Tensor | None = None

    def forward(
        self,
        x_causal: Tensor,
        future_target_embeds: Tensor | None = None,
        future_targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Run backward ($K \\to 1$) then Schur-intersected forward ($1 \\to K$) causal virtual MPS sweep."""
        bsz, seq_len, _ = x_causal.shape
        g = self.num_groups
        d_h = COMPLEX_HEAD_DIM
        site_real = 2 * d_h

        virt_raw = self.virtual_rollout_proj(x_causal).view(
            bsz, seq_len, self.horizon_k, g, site_real
        )
        virt_f = F.normalize(virt_raw.float(), dim=-1, eps=1.0e-6)

        r_bwd = torch.full(
            (bsz, seq_len, g, self.bond_chi),
            1.0 / math.sqrt(float(self.bond_chi)),
            device=x_causal.device,
            dtype=torch.float32,
        )
        rho_bwd_diags: list[Tensor] = [torch.empty(0, device=x_causal.device)] * self.horizon_k
        for k in range(self.horizon_k - 1, -1, -1):
            site_drive = torch.matmul(virt_f[:, :, k], self.bwd_site_to_bond[k].float())
            bond_prop = torch.matmul(r_bwd, self.bwd_bond_transfer[k].float())
            r_bwd = F.normalize(bond_prop + site_drive, dim=-1, eps=1.0e-6)
            diag_raw = torch.matmul(r_bwd, self.bwd_bond_to_diag[k].float()).square() + (
                0.10 / float(d_h)
            )
            rho_bwd_diags[k] = diag_raw / diag_raw.sum(dim=-1, keepdim=True).clamp_min(DEFAULT_TRACE_EPS)

        x_heads = x_causal.view(bsz, seq_len, g, site_real)
        l_fwd = F.normalize(self.fwd_bond_init(x_heads).float(), dim=-1, eps=1.0e-6)
        smoothed_sites: list[Tensor] = []
        p_mps_list: list[Tensor] = []

        for k in range(self.horizon_k):
            phi_raw = virt_f[:, :, k] + torch.matmul(l_fwd, self.fwd_bond_to_site[k].float())
            phi_re, phi_im = normalize_complex_ket(phi_raw[..., :d_h], phi_raw[..., d_h:])
            rho_fwd_diag = phi_re.square() + phi_im.square()
            bwd_diag_k = rho_bwd_diags[k]

            p_mps = (rho_fwd_diag * bwd_diag_k).sum(dim=-1, keepdim=True).clamp_min(COMP_SCHUR_TRACE_EPS)
            p_mps_list.append(p_mps.detach().mean())
            bwd_amp = torch.sqrt(bwd_diag_k.clamp_min(DEFAULT_TRACE_EPS))
            inv_norm = torch.rsqrt(p_mps)
            smooth_site = torch.cat([phi_re * bwd_amp * inv_norm, phi_im * bwd_amp * inv_norm], dim=-1)
            smoothed_sites.append(smooth_site)
            if k + 1 < self.horizon_k:
                l_fwd = F.normalize(
                    l_fwd + torch.matmul(smooth_site, self.fwd_site_to_bond[k].float()),
                    dim=-1,
                    eps=1.0e-6,
                )

        if not self.training:
            self._last_mps_pand = torch.stack(p_mps_list).mean()

        stacked_smooth = torch.stack(smoothed_sites, dim=2)  # (B, T, K, G, 32)
        h_weights = F.softmax(self.horizon_pool_logits.float(), dim=0).view(1, 1, self.horizon_k, 1, 1)
        pooled = (stacked_smooth * h_weights).sum(dim=2).reshape(bsz, seq_len, self.real_dim)
        delta_x = self.mtp_out_proj(pooled.to(x_causal.dtype))

        aux_loss: Tensor | None = None
        if future_target_embeds is not None:
            tgt_norm = F.normalize(future_target_embeds.detach().float(), dim=-1, eps=1.0e-6)
            pred_flat = F.normalize(
                stacked_smooth.reshape(bsz, seq_len, self.horizon_k, self.real_dim),
                dim=-1,
                eps=1.0e-6,
            )
            bos_csum_pad: Tensor | None = None
            if future_targets is not None:
                bos_csum = future_targets.eq(FINEWEB_BOS_TOKEN_ID).to(torch.long).cumsum(dim=1)
                bos_csum_pad = F.pad(bos_csum, (1, 0), value=0)
            step_losses: list[Tensor] = []
            for k in range(self.horizon_k):
                if seq_len <= k:
                    break
                p_slice = pred_flat[:, : seq_len - k, k, :]
                t_slice = tgt_norm[:, k:, :]
                cos_err = 1.0 - (p_slice * t_slice).sum(dim=-1)
                if bos_csum_pad is not None:
                    same_k = (bos_csum_pad[:, k + 1 :] == bos_csum_pad[:, : seq_len - k]).to(
                        torch.float32
                    )
                    step_losses.append((cos_err * same_k).sum() / same_k.sum().clamp_min(1.0))
                else:
                    step_losses.append(cos_err.mean())
            if step_losses:
                aux_loss = SM_SMOOTHED_MTP_AUX_WEIGHT * torch.stack(step_losses).mean()

        return delta_x, aux_loss


# ==============================================================================
# Pillar 4: Unified U-Net MuP + Carbon Quantum Baton Block & Language Model
# ==============================================================================
def _apply_complex_head_rope_u78(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply complex unitary RoPE to `(B, H, T, 32)` (`16` complex pairs per head, `<= 8` args)."""
    half = x.size(-1) // 2
    x_re, x_im = x[..., :half], x[..., half:]
    c, s = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x_re * c - x_im * s, x_re * s + x_im * c], dim=-1)


class UnifiedMuPCarbonBlock(nn.Module):
    """U-Net Hilbert-highway block combining `<BOS>`-masked XSA, Complementary Schur, & Quantum MLP."""

    def __init__(self, cfg: UnifiedMuPCarbonConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.real_dim = 2 * cfg.complex_dim
        self.head_real_dim = 2 * COMPLEX_HEAD_DIM
        self.num_heads = self.real_dim // self.head_real_dim
        self.depth_scale = DEPTH_RESIDUAL_INIT_SCALE / math.sqrt(max(1.0, float(cfg.num_layers) / 4.0))

        init_std = 0.5 / math.sqrt(float(self.real_dim))
        self.block_lambdas = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float32))
        self.qkv_proj = nn.Linear(self.real_dim, 3 * self.real_dim, bias=False)
        self.out_proj = nn.Linear(self.real_dim, self.real_dim, bias=False)
        nn.init.normal_(self.qkv_proj.weight, std=init_std)
        nn.init.zeros_(self.out_proj.weight)

        self.v0_mix_logit = (
            nn.Parameter(torch.tensor(-0.5, dtype=torch.float32)) if layer_idx > 0 else None
        )
        self.xsa_alpha = nn.Parameter(torch.zeros((1, self.num_heads, 1, 1), dtype=torch.float32))

        self.schur_gate: ZeroSyncComplementarySchurGate | None = (
            ZeroSyncComplementarySchurGate(real_dim=self.real_dim)
            if cfg.schur_mode == "complementary_zero_sync"
            else None
        )

        hidden = cfg.mlp_expansion * self.real_dim
        self.mlp_up = nn.Linear(self.real_dim, hidden, bias=False)
        self.mlp_phase_gate = nn.Linear(self.real_dim, hidden, bias=False)
        self.mlp_down = nn.Linear(hidden, self.real_dim, bias=False)
        nn.init.normal_(self.mlp_up.weight, std=init_std)
        nn.init.normal_(self.mlp_phase_gate.weight, std=init_std)
        nn.init.zeros_(self.mlp_down.weight)

    def forward(
        self,
        x: Tensor,
        x_0: Tensor,
        cos_4d: Tensor,
        sin_4d: Tensor,
        causal_doc_mask: Tensor,
        same_prev_tok: Tensor,
        v_0: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Run `<BOS>`-masked XSA + Zero-Sync Complementary Schur + Quantum Interference MLP (`<= 8` args)."""
        bsz, seq_len, d_real = x.shape
        lam = self.block_lambdas.to(x.dtype)
        x_in = lam[0] * x + lam[1] * x_0

        h_norm = F.rms_norm(x_in, (d_real,), eps=1.0e-6)
        q_raw, k_raw, v_raw = self.qkv_proj(h_norm).chunk(3, dim=-1)
        q = q_raw.view(bsz, seq_len, self.num_heads, self.head_real_dim).permute(0, 2, 1, 3)
        k = k_raw.view(bsz, seq_len, self.num_heads, self.head_real_dim).permute(0, 2, 1, 3)
        v = v_raw.view(bsz, seq_len, self.num_heads, self.head_real_dim).permute(0, 2, 1, 3)

        q = _apply_complex_head_rope_u78(F.rms_norm(q, (self.head_real_dim,), eps=1.0e-6), cos_4d, sin_4d)
        k = _apply_complex_head_rope_u78(F.rms_norm(k, (self.head_real_dim,), eps=1.0e-6), cos_4d, sin_4d)

        if seq_len > 1:
            sp = same_prev_tok.unsqueeze(1)
            k = torch.cat(
                [
                    k[..., :24],
                    torch.cat([k[:, :, :1, 24:], torch.where(sp, k[:, :, :-1, 24:], k[:, :, 1:, 24:])], dim=2),
                ],
                dim=-1,
            )

        if v_0 is None:
            v_next_0 = v
        else:
            v_next_0 = v_0
            assert self.v0_mix_logit is not None
            v = torch.lerp(v, v_0, torch.sigmoid(self.v0_mix_logit).to(v.dtype))

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=causal_doc_mask, scale=DEFAULT_W77_ATTN_SCALE
        )
        v_unit = F.normalize(v.float(), dim=-1, eps=1.0e-6).to(attn_out.dtype)
        echo_proj = (attn_out * v_unit).sum(dim=-1, keepdim=True) * v_unit
        attn_clean = (
            attn_out - torch.sigmoid(self.xsa_alpha).to(attn_out.dtype) * echo_proj
        ).permute(0, 2, 1, 3).reshape(bsz, seq_len, d_real)

        ds = self.depth_scale
        x_mid = x_in + ds * self.out_proj(attn_clean)
        if self.schur_gate is not None:
            x_mid = x_mid + ds * self.schur_gate(h_norm, F.rms_norm(attn_clean, (d_real,), eps=1.0e-6))

        m_in = F.rms_norm(x_mid, (d_real,), eps=1.0e-6)
        boson_amp = F.relu(self.mlp_up(m_in)).square()
        phase_mod = 1.0 + 0.25 * torch.tanh(self.mlp_phase_gate(m_in))
        x_out = x_mid + ds * self.mlp_down(boson_amp * phase_mod)
        return x_out, v_next_0


class UnifiedMuPCarbonBatonLM(nn.Module):
    """Flag-free strategy-configured `Wave 78` `R4c + R5B` Unified Quantum Baton Language Model."""

    def __init__(self, cfg: UnifiedMuPCarbonConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.complex_dim = int(cfg.complex_dim)
        self.real_dim = 2 * self.complex_dim
        self.num_layers = int(cfg.num_layers)

        self.wte_ket = nn.Embedding(cfg.vocab_size, self.real_dim)
        nn.init.normal_(self.wte_ket.weight, std=1.0)

        self.smear_in_dim = min(16, self.real_dim)
        self.smear_gate = nn.Linear(self.smear_in_dim, 1, bias=False)
        nn.init.zeros_(self.smear_gate.weight)
        self.smear_lambda = nn.Parameter(torch.zeros((1,), dtype=torch.float32))

        self.blocks = nn.ModuleList(
            [UnifiedMuPCarbonBlock(cfg, layer_idx=i) for i in range(self.num_layers)]
        )
        n_dec = max(1, self.num_layers // 2)
        self.skip_weights = nn.Parameter(torch.zeros((n_dec,), dtype=torch.float32))

        self.causal_mtp_sweep: HorizonPooledCausalMTPMPSSweep | None = (
            HorizonPooledCausalMTPMPSSweep(real_dim=self.real_dim)
            if cfg.use_horizon_mtp_mps
            else None
        )
        self.copy_gate = PurityConditionedHybridWalshCopyGate(
            sem_ket_dim=self.complex_dim,
            ctx_dim=self.real_dim,
            vocab_size=cfg.vocab_size,
            copy_mode=cfg.copy_mode,
        )
        self.lm_head = nn.Linear(self.real_dim, cfg.vocab_size, bias=False)
        nn.init.zeros_(self.lm_head.weight)

        self.uni_prior_logit = nn.Parameter(
            torch.tensor(math.log(math.expm1(DEFAULT_ONLINE_UNI_PRIOR_SCALE)), dtype=torch.float32)
        )
        self.register_buffer(
            "online_uni_token_probs",
            torch.full((cfg.vocab_size,), 1.0 / float(cfg.vocab_size), dtype=torch.float32),
        )
        self.register_buffer("online_ingested_tokens", torch.tensor(0, dtype=torch.long))
        self._rope_cache: dict[tuple[int, torch.device], tuple[Tensor, Tensor]] = {}

    def _get_rope_cos_sin(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Return `(cos_4d, sin_4d)` of shape `(1, 1, T, 16)` (`<= 8` args)."""
        key = (seq_len, device)
        if key in self._rope_cache:
            return self._rope_cache[key]
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = (1.0 / DEFAULT_W77_ROPE_BASE) ** torch.linspace(
            0.0, 1.0, steps=COMPLEX_HEAD_DIM, device=device, dtype=torch.float32
        )
        theta = torch.outer(pos, freqs)
        pair = (
            theta.cos().view(1, 1, seq_len, COMPLEX_HEAD_DIM),
            theta.sin().view(1, 1, seq_len, COMPLEX_HEAD_DIM),
        )
        self._rope_cache[key] = pair
        return pair

    @torch.no_grad()
    def ingest_online_unigram_batch(self, idx: Tensor) -> None:
        """Update streaming online unigram prior (`100%` speedrun-compliant, `0` offline scan)."""
        eta = DEFAULT_ONLINE_UNI_EMA_DECAY
        flat = idx.reshape(-1)
        inc = eta / float(max(1, flat.numel()))
        self.online_uni_token_probs.mul_(1.0 - eta)
        self.online_uni_token_probs.scatter_add_(
            0, flat, torch.full_like(flat, inc, dtype=torch.float32)
        )
        self.online_ingested_tokens.add_(flat.numel())

    def encode_trunk_features(
        self,
        idx: Tensor,
        targets_for_mtp: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, list[Tensor]]:
        """Encode `idx` `(B, T)` into `(x_final, copy_log_odds, copy_conf, mtp_aux_loss, layer_acts)`."""
        _, seq_len = idx.shape
        raw_emb = self.wte_ket(idx)
        same_prev = (idx[:, 1:] != FINEWEB_BOS_TOKEN_ID).unsqueeze(-1)
        if seq_len > 1:
            gate = (
                self.smear_lambda.to(raw_emb.dtype)
                * torch.sigmoid(self.smear_gate(raw_emb[:, 1:, : self.smear_in_dim]))
                * same_prev.to(raw_emb.dtype)
            )
            smear_emb = torch.cat([raw_emb[:, :1], raw_emb[:, 1:] + gate * raw_emb[:, :-1]], dim=1)
        else:
            smear_emb = raw_emb

        x_0 = F.rms_norm(smear_emb, (self.real_dim,), eps=1.0e-6)
        ket0_re, ket0_im = normalize_complex_ket(
            smear_emb[..., : self.complex_dim], smear_emb[..., self.complex_dim :]
        )

        mask_pack = build_bos_anchor_sink_doc_masks(idx)
        t_idx = torch.arange(seq_len, device=idx.device).unsqueeze(1)
        s_idx = torch.arange(seq_len, device=idx.device).unsqueeze(0)
        causal_doc_mask = ((s_idx <= t_idx).unsqueeze(0) & mask_pack.doc_same).unsqueeze(1)
        cos_4d, sin_4d = self._get_rope_cos_sin(seq_len, idx.device)

        x = x_0
        v_0: Tensor | None = None
        n_half = self.num_layers // 2
        enc_stack: list[Tensor] = []
        layer_acts: list[Tensor] = []

        for l_idx, blk in enumerate(self.blocks):
            if self.num_layers >= 4 and l_idx >= n_half and enc_stack:
                skip_g = torch.sigmoid(self.skip_weights[l_idx - n_half]).to(x.dtype)
                x = x + skip_g * enc_stack.pop()
            x, v_0 = blk(x, x_0, cos_4d, sin_4d, causal_doc_mask, same_prev, v_0)
            layer_acts.append(x)
            if self.num_layers >= 4 and l_idx < n_half:
                enc_stack.append(x)

        mtp_aux_loss: Tensor | None = None
        if self.causal_mtp_sweep is not None:
            fut_emb = self.wte_ket(targets_for_mtp).detach() if targets_for_mtp is not None else None
            delta_mtp, mtp_aux_loss = self.causal_mtp_sweep(x, fut_emb, targets_for_mtp)
            x = x + delta_mtp

        x_final = F.rms_norm(x, (self.real_dim,), eps=1.0e-6)
        copy_log_odds, copy_conf = self.copy_gate(
            idx, ket0_re, ket0_im, x_final, mask_pack.doc_same
        )
        return x_final, copy_log_odds, copy_conf, mtp_aux_loss, layer_acts

    def compute_logits(self, idx: Tensor, x_final: Tensor, copy_log_odds: Tensor) -> Tensor:
        """Compute full `(B, T, V)` softcapped logits (`<= 8` args)."""
        bsz, seq_len = idx.shape
        uni_scale = F.softplus(self.uni_prior_logit.float()).to(x_final.dtype)
        uni_log_p = torch.log(self.online_uni_token_probs.detach().clamp_min(1.0e-9)).to(x_final.dtype)
        base_logits = F.linear(x_final, self.lm_head.weight) + uni_scale * uni_log_p.view(1, 1, -1)
        copy_index = idx.unsqueeze(1).expand(bsz, seq_len, seq_len)
        combined = base_logits.scatter_add(
            dim=-1, index=copy_index, src=copy_log_odds.to(base_logits.dtype)
        )
        cap = self.cfg.logit_softcap
        return cap * torch.tanh(combined.float() * (1.0 / cap))

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        ingest_online: bool = False,
        return_logits: bool = False,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Run chunked `bfloat16` training/eval forward pass (`<= 8` args)."""
        bsz, seq_len = idx.shape
        x_final, copy_log_odds, _, mtp_aux_loss, _ = self.encode_trunk_features(
            idx, targets_for_mtp=targets if self.training else None
        )
        if return_logits or targets is None:
            logits = self.compute_logits(idx, x_final, copy_log_odds)
            loss: Tensor | None = None
            if targets is not None:
                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                loss = ce + mtp_aux_loss if mtp_aux_loss is not None else ce
            if ingest_online:
                self.ingest_online_unigram_batch(idx)
            return logits, loss

        chunk_b = max(1, DEFAULT_W77_CE_CHUNK_TOKENS // max(1, seq_len))
        uni_scale = F.softplus(self.uni_prior_logit.float()).to(x_final.dtype)
        uni_log_p = (
            torch.log(self.online_uni_token_probs.detach().clamp_min(1.0e-9))
            .to(x_final.dtype)
            .view(1, 1, -1)
        )
        copy_index = idx.unsqueeze(1).expand(bsz, seq_len, seq_len)
        cap = self.cfg.logit_softcap
        inv_cap = 1.0 / cap

        chunk_losses: list[Tensor] = []
        for b0 in range(0, bsz, chunk_b):
            b1 = min(bsz, b0 + chunk_b)
            lg_c = F.linear(x_final[b0:b1], self.lm_head.weight) + uni_scale * uni_log_p
            lg_c = lg_c.scatter_add(
                dim=-1, index=copy_index[b0:b1], src=copy_log_odds[b0:b1].to(lg_c.dtype)
            )
            capped_c = cap * torch.tanh(lg_c.float() * inv_cap)
            ce_c = F.cross_entropy(capped_c.reshape(-1, capped_c.size(-1)), targets[b0:b1].reshape(-1))
            chunk_losses.append(ce_c * float(b1 - b0))

        total_ce = torch.stack(chunk_losses).sum() / float(bsz)
        total_loss = total_ce + mtp_aux_loss if mtp_aux_loss is not None else total_ce
        if ingest_online:
            self.ingest_online_unigram_batch(idx)
        return None, total_loss

    @torch.no_grad()
    def collect_carbon_routing_telemetry(self) -> dict[str, Any]:
        """Collect eval-time `mean_p_and`, `even_purity`, `odd_purity`, & `horizon_weights` (`<= 8` args)."""
        per_layer: list[dict[str, float]] = []
        for idx_l, blk in enumerate(self.blocks):
            if blk.schur_gate is not None:
                snap = blk.schur_gate.snapshot_purity_metrics()
                snap["layer"] = float(idx_l)
                per_layer.append(snap)
        n_l = float(max(1, len(per_layer)))
        mps_p = (
            round(float(self.causal_mtp_sweep._last_mps_pand.item()), 5)
            if (self.causal_mtp_sweep is not None and self.causal_mtp_sweep._last_mps_pand is not None)
            else 0.0
        )
        h_w = (
            [
                round(float(v), 4)
                for v in F.softmax(self.causal_mtp_sweep.horizon_pool_logits.detach().float(), dim=0).tolist()
            ]
            if self.causal_mtp_sweep is not None
            else []
        )
        wh_w = (
            round(float(torch.sigmoid(self.copy_gate.wh_blend_logit.detach()).item()), 4)
            if self.copy_gate.wh_blend_logit is not None
            else 0.0
        )
        return {
            "has_schur_gate": bool(len(per_layer) > 0),
            "mean_p_and": round(sum(d["mean_p_and"] for d in per_layer) / n_l, 5) if per_layer else 0.0,
            "mean_even_purity": round(sum(d["even_purity"] for d in per_layer) / n_l, 5) if per_layer else 0.0,
            "mean_odd_purity": round(sum(d["odd_purity"] for d in per_layer) / n_l, 5) if per_layer else 0.0,
            "mps_mean_p_and": mps_p,
            "mtp_horizon_weights": h_w,
            "wh_exact_copy_weight": wh_w,
            "per_layer": per_layer,
        }


# ==============================================================================
# Optimizer Builder & Polymorphic Variant Adapter (`Wave 78`)
# ==============================================================================
def build_wave78_mup_optimizers(
    model: UnifiedMuPCarbonBatonLM,
    warmup_steps: int = 40,
) -> list[torch.optim.Optimizer]:
    """Build `PolarExpress` `StiefelMuon` + `1/D`-scaled untied `lm_head` `Adam` (`<= 8` args)."""
    muon_params: list[nn.Parameter] = []
    wte_params: list[nn.Parameter] = [model.wte_ket.weight]
    head_params: list[nn.Parameter] = [model.lm_head.weight]
    scalar_params: list[nn.Parameter] = []

    excluded_ids = {id(model.wte_ket.weight), id(model.lm_head.weight)}
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in excluded_ids:
            continue
        if p.ndim >= 2 and min(p.shape[-2], p.shape[-1]) >= 8 and "ctx_gate_proj" not in name and "smear_gate" not in name:
            muon_params.append(p)
        else:
            scalar_params.append(p)

    muon_opt = StiefelMuon(
        muon_params,
        lr=W77_MUON_BASE_LR,
        momentum=W77_MUON_MOMENTUM,
        weight_decay=0.01,
        decay_mode="cautious",
        aspect_scale=True,
        config=StiefelMuonConfig(
            nesterov=True,
            ns_steps=5,
            polar_method="polar_express",
            stiefel_penalty=0.0,
            use_normuon=True,
            use_twin_rail=False,
        ),
        momentum_warmup_steps=max(10, int(warmup_steps)),
        momentum_init=W77_MUON_MOMENTUM_INIT,
        ema_momentum=True,
        batched_ns=True,
    )
    head_lr = W77_HEAD_BASE_LR_AT_D256 * (float(MUP_BASE_REAL_DIM) / float(model.real_dim))
    is_cuda = next(model.parameters()).is_cuda
    adam_opt = torch.optim.Adam(
        [
            {"params": head_params, "lr": head_lr, "group_name": "lm_head"},
            {"params": wte_params, "lr": W77_WTE_BASE_LR, "group_name": "wte_ket"},
            {"params": scalar_params, "lr": W77_SCALAR_BASE_LR, "group_name": "scalars"},
        ],
        betas=(0.80, 0.95),
        eps=1.0e-8,
        fused=is_cuda,
    )
    return [muon_opt, adam_opt]


class Wave78UnifiedAdapter(DiagnosticModelAdapter):
    """Polymorphic `DiagnosticModelAdapter` wrapping `UnifiedMuPCarbonBatonLM` (`<= 8` args)."""

    def __init__(
        self,
        key: str,
        label: str,
        model: UnifiedMuPCarbonBatonLM,
        optimizers: list[torch.optim.Optimizer],
        device: torch.device,
        use_wsd_schedule: bool = False,
    ) -> None:
        super().__init__(
            key=key,
            label=label,
            model=model,
            device=device,
            embed_dim=model.real_dim,
            num_layers=model.num_layers,
        )
        self.unified_model = model
        self.optimizers = optimizers
        self.use_wsd_schedule = bool(use_wsd_schedule)
        self.base_lrs: list[list[float]] = [
            [float(pg["lr"]) for pg in opt.param_groups] for opt in optimizers
        ]

    def _compute_lr_multiplier(self, step: int, total_steps: int) -> float:
        """Compute warmup + cosine (sprint) or `WSD` linear quench (confirmation) LR mult."""
        warmup = max(10, int(0.06 * total_steps))
        if step <= warmup:
            return float(step) / float(max(1, warmup))
        if self.use_wsd_schedule:
            quench_start = int((1.0 - WSD_COOLDOWN_FRACTION) * total_steps)
            if step <= quench_start:
                return 1.0
            q_prog = float(step - quench_start) / float(max(1, total_steps - quench_start))
            return max(0.05, 1.0 - 0.95 * q_prog)
        prog = float(step - warmup) / float(max(1, total_steps - warmup))
        return 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * prog))

    def train_step(self, x_batch: Tensor, y_batch: Tensor, step: int, total_steps: int) -> float:
        """Execute one `bfloat16` forward + backward + optimizer step (`<= 8` args)."""
        self.unified_model.train()
        lr_mult = self._compute_lr_multiplier(step, total_steps)
        for opt, base_group_lrs in zip(self.optimizers, self.base_lrs):
            for pg, base_lr in zip(opt.param_groups, base_group_lrs):
                pg["lr"] = base_lr * lr_mult
            opt.zero_grad(set_to_none=True)

        amp_on = x_batch.is_cuda
        with torch.amp.autocast(
            device_type=x_batch.device.type, dtype=torch.bfloat16, enabled=amp_on
        ):
            _, loss = self.unified_model(
                x_batch, targets=y_batch, ingest_online=True, return_logits=False
            )
        assert loss is not None
        loss.backward()
        for opt in self.optimizers:
            opt.step()
        return float(loss.detach().item())

    @torch.no_grad()
    def predict_log_probs_and_span(self, x_batch: Tensor) -> tuple[Tensor, Tensor]:
        """Return `(log_probs, logit_span)` of shapes `(B, T, 50257)` and `(B, T)` (`<= 8` args)."""
        was_train = self.unified_model.training
        self.unified_model.eval()
        amp_on = x_batch.is_cuda
        with torch.amp.autocast(
            device_type=x_batch.device.type, dtype=torch.bfloat16, enabled=amp_on
        ):
            logits_raw, _ = self.unified_model(
                x_batch, targets=None, ingest_online=False, return_logits=True
            )
        if was_train:
            self.unified_model.train()
        assert logits_raw is not None
        logits = logits_raw[..., :VALID_FINEWEB_VOCAB_SIZE].float()
        span = logits.amax(dim=-1) - logits.median(dim=-1).values
        return F.log_softmax(logits, dim=-1), span


def build_wave78_variant_adapter(
    variant: Wave78VariantKind,
    device: torch.device,
    seed: int = 1337,
    use_wsd_schedule: bool = False,
    complex_dim_override: int | None = None,
) -> Wave78UnifiedAdapter:
    """Instantiate any `Wave78VariantKind` cleanly via explicit strategy dispatch (`<= 8` args)."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    d_c = int(complex_dim_override) if complex_dim_override is not None else 256
    spec_map: dict[Wave78VariantKind, tuple[str, UnifiedMuPCarbonConfig]] = {
        Wave78VariantKind.U0_R4C_BASE_4L_D256: (
            "U0: R4c Baseline U-Net Baton Mixer (4L, d=256 / 512D)",
            UnifiedMuPCarbonConfig(
                complex_dim=d_c,
                num_layers=4,
                copy_mode="semantic_only",
                schur_mode="none",
                use_horizon_mtp_mps=False,
            ),
        ),
        Wave78VariantKind.U1_HYBRID_WH_COPY_4L_D256: (
            "U1: R4c + Purity-Gated Hybrid Walsh–Hadamard Copy (4L, d=256)",
            UnifiedMuPCarbonConfig(
                complex_dim=d_c,
                num_layers=4,
                copy_mode="hybrid_walsh_purity",
                schur_mode="none",
                use_horizon_mtp_mps=False,
            ),
        ),
        Wave78VariantKind.U2_R4C_COMP_SCHUR_4L_D256: (
            "U2: R4c + Hybrid WH Copy + Zero-Sync Comp Schur (4L, d=256)",
            UnifiedMuPCarbonConfig(
                complex_dim=d_c,
                num_layers=4,
                copy_mode="hybrid_walsh_purity",
                schur_mode="complementary_zero_sync",
                use_horizon_mtp_mps=False,
            ),
        ),
        Wave78VariantKind.U3_UNIFIED_R4C_R5B_4L_D256: (
            "U3: Unified R4c + R5B (Comp Schur + Horizon MTP MPS + Hybrid Copy, 4L, d=256)",
            UnifiedMuPCarbonConfig(
                complex_dim=d_c,
                num_layers=4,
                copy_mode="hybrid_walsh_purity",
                schur_mode="complementary_zero_sync",
                use_horizon_mtp_mps=True,
            ),
        ),
        Wave78VariantKind.U4_UNIFIED_R4C_R5B_6L_D256: (
            "U4: Deep 6L Unified R4c + R5B U-Net (6L, d=256 / 512D)",
            UnifiedMuPCarbonConfig(
                complex_dim=d_c,
                num_layers=6,
                copy_mode="hybrid_walsh_purity",
                schur_mode="complementary_zero_sync",
                use_horizon_mtp_mps=True,
            ),
        ),
    }
    label, cfg = spec_map[variant]
    model = UnifiedMuPCarbonBatonLM(cfg).to(device)
    opts = build_wave78_mup_optimizers(model, warmup_steps=40)
    return Wave78UnifiedAdapter(
        key=variant.value,
        label=label,
        model=model,
        optimizers=opts,
        device=device,
        use_wsd_schedule=use_wsd_schedule,
    )
