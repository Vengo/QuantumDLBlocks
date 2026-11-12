r"""Wave 79 Fused In-Attention Complementary Schur, Bi-Trigger Walsh Induction, & Compiled `<BOS>` Baton (`EXPERIMENT_PLAN_099.md`).

Provenance & Mathematical Rationale
-----------------------------------
Built for `Wave 79` (`EXPERIMENT_PLAN_099.md`) following Rules 1–8 of `docs/SCRIPT_BEST_PRACTICES.md`
(clean, flag-free per-wave formulation leaving `Waves 77–78` frozen for reproducible comparison).
Resolves both the throughput bottleneck (`-42.8%` to `-59.6%` speed drop in `Wave 78` `U2–U4`) and
the `T=64/128` Induction noise-inversion / `m=0` Recall impostor-tax bottleneck discovered in `Wave 78`:

1. **Surprisal-Conditioned `128`-D Welch-Rectified Walsh Bi-Trigger Copy Gate (`SurprisalConditionedWalshBiTriggerCopyGate`)**:
   - Expands deterministic `Z_16`-balanced `GF(2)` Walsh–Hadamard token-ID kets to `128` dimensions
     and applies **Welch Collision-Floor Rectification**
     $M_1(t, s) = \bigl(\max(0, |\langle w(x_t) \mid w(x_{s-1})\rangle|^2 - \tau_{\text{welch}}) / (1 - \tau_{\text{welch}})\bigr)^p$
     ($\tau_{\text{welch}} = 0.36$), giving **strictly `0.0`** off-diagonal noise on random tokens and
     **exact `1.0`** on all `50,257` seen or unseen token IDs from Step 0.
   - Computes **zero-GEMM 2-token phrase coincidence** via diagonal shift $M_2(t, s) = M_1(t, s) M_1(t-1, s-1)$
     and weights lone 1-gram matches by **streaming online unigram surprisal**
     $\sigma(\gamma_{\text{idf}}(-\ln P_{\text{uni}}(x_t) - I_0))$ ($I_0 = 5.80$ nats, `0` offline scan):
     bare `D0` stopwords (`" the"`, $I \approx 3.0$ nats) are damped unless part of a 2-gram match $M_2$,
     while rare entities (`D2–D9`) and `50k`-vocab Induction/MQAR triggers ($I \approx 10.82$ nats) fire at full gain.
   - Normalizes `ctx_feat` to unit $L_2$ norm (`1 / sqrt(D)`) and pairs a shared unitary semantic anchor
     $U_{\text{sem}}$ with conservative `W79_COPY_SCALAR_LR = 0.004`, eliminating $\mu\text{P}$ gate saturation.

2. **Zero-Expansion In-Attention Head-Pair Complementary Schur Router (`InAttentionComplementarySchurRouter`)**:
   - Pairs the $H = D_{\text{real}} // 32$ complex $\mathbb{C}^{16}$ attention heads into $G_{\text{pair}} = H // 2$
     subsystems of Rank $R = 2$ ($G_{\text{pair}} \cdot R \cdot 32 \equiv D_{\text{real}}$, `1.0x` width,
     `19x` fewer parameters than `Wave 78`'s external `ZeroSyncComplementarySchurGate`).
   - Routes trace-preserving consensus $p_{\text{and}} \rho_{\text{AND}}|\psi\rangle$ into even head slots `2g`
     and phase-preserving partial-trace disagreement $(1 - p_{\text{and}}) \rho_{\text{comp}}|\psi\rangle$
     into odd head slots `2g+1` directly before `out_proj`.

3. **Compiled `<BOS>` Document `FlexAttention` + Fused Chunked Softcap Readout (`FusedUnifiedSchurBatonLM`)**:
   - Uses `torch.nn.attention.flex_attention` (`BLOCK_SIZE = 128`) to skip cross-document and upper-triangular
     blocks in hardware on CUDA, and compiles the pure tensor-in/tensor-out U-Net backbone + chunked
     `tanh`-softcap cross-entropy readout with zero Dynamo graph breaks or attribute mutations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

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
from .wave78_unified_mup_carbon_baton import (
    DEFAULT_W78_MLP_EXPANSION,
    DEPTH_RESIDUAL_INIT_SCALE,
    MTP_HORIZON_INIT_LOGITS,
    SM_SMOOTHED_MTP_AUX_WEIGHT,
    HorizonPooledCausalMTPMPSSweep,
)
from ..optim.muon import StiefelMuon, StiefelMuonConfig

# ==============================================================================
# Top-of-File Named Constants (Synchronized with EXPERIMENT_PLAN_099.md Section 3)
# ==============================================================================
WALSH_INDUCTION_DIM: int = 128
WALSH_CYCLIC_SEEDS_128D: tuple[int, ...] = (
    0xB496,
    0xD2A9,
    0xE563,
    0x9C37,
    0xA75C,
    0x69AD,
    0x3CF2,
)
WELCH_WALSH_FLOOR_SQ: float = 0.36
WELCH_SEMANTIC_FLOOR_SQ: float = 0.08
SURPRISAL_IDF_MIDPOINT_NATS: float = 5.80
SURPRISAL_IDF_SLOPE: float = 1.25
BIGRAM_PHRASE_BOOST_MULTIPLIER: float = 1.35
W79_COPY_SCALAR_LR: float = 0.004

IN_ATTN_SCHUR_PAIR_RANK: int = 2
IN_ATTN_SCHUR_INIT_LOGIT: float = -0.50
FLEX_DOC_MASK_BLOCK_SIZE: int = 128

SchurRoutingMode = Literal["none", "in_attn_head_pair"]
SchurEvalMaskMode = Literal["normal", "drop_odd", "drop_even"]
AttnBackendMode = Literal["auto_flex", "dense_sdpa"]


class Wave79VariantKind(str, Enum):
    """Strategy selector for the `Wave 79` verification & `300M` speedrun suite (Rule 3)."""

    V0_SURPRISAL_WH_INDUCTION_4L_D256 = "w79_v0_surprisal_wh_induction_4l_d256"
    V1_IN_ATTN_COMP_SCHUR_4L_D256 = "w79_v1_in_attn_comp_schur_4l_d256"
    V2_FUSED_UNIFIED_4L_D256 = "w79_v2_fused_unified_4l_d256"
    V3_FUSED_UNIFIED_6L_D256 = "w79_v3_fused_unified_6l_d256"


@dataclass(frozen=True)
class FusedUnifiedBatonConfig:
    """Frozen structural configuration for `FusedUnifiedSchurBatonLM` (`<= 8` fields, Rule 2)."""

    vocab_size: int = VALID_FINEWEB_VOCAB_SIZE
    complex_dim: int = 256
    num_layers: int = 4
    schur_mode: SchurRoutingMode = "in_attn_head_pair"
    use_horizon_mtp_mps: bool = True
    attn_backend: AttnBackendMode = "auto_flex"
    compile_backbone: bool = True
    logit_softcap: float = DEFAULT_W77_LOGIT_SOFTCAP


def build_balanced_128d_walsh_codebook(
    vocab_size: int = VALID_FINEWEB_VOCAB_SIZE,
    num_features: int = WALSH_INDUCTION_DIM,
) -> Tensor:
    """Build deterministic `Z_16`-symmetric unit-norm `GF(2)` Walsh–Hadamard codebook `(V, 128)` (`<= 8` args).

    Combines `16` single-bit indicators `(1 << b)` with `7` complete `16`-shift cyclic orbits of
    weight-8 `16`-bit polynomials (`WALSH_CYCLIC_SEEDS_128D` = `112` parity masks), giving `128`
    balanced `±1 / sqrt(128)` coordinates where every token ID `v` has exact unit norm `1.0` and
    distinct random token pairs stay tightly concentrated near `0` (below `WELCH_WALSH_FLOOR_SQ`).
    """
    masks: list[int] = [(1 << b) for b in range(16)]
    for seed_mask in WALSH_CYCLIC_SEEDS_128D:
        for shift in range(16):
            rot = ((seed_mask << shift) | (seed_mask >> (16 - shift))) & 0xFFFF
            masks.append(rot)
    while len(masks) < num_features:
        masks.append(((len(masks) * 40503) & 0xFFFF) | 1)
    mask_t = torch.tensor(masks[:num_features], dtype=torch.long)
    bit_idx = torch.arange(16, dtype=torch.long)
    mask_bits = ((mask_t.unsqueeze(0) >> bit_idx.unsqueeze(1)) & 1).to(torch.float32)

    tok_ids = torch.arange(int(vocab_size), dtype=torch.long)
    tok_bits = ((tok_ids.unsqueeze(1) >> bit_idx.unsqueeze(0)) & 1).to(torch.float32)
    gf2_parities = tok_bits.matmul(mask_bits).to(torch.long) & 1
    signs = (1.0 - 2.0 * gf2_parities.to(torch.float32)) * (1.0 / math.sqrt(float(num_features)))
    return signs.to(torch.bfloat16)


def build_bos_doc_flex_block_mask(
    idx: Tensor,
    bos_token_id: int = FINEWEB_BOS_TOKEN_ID,
    block_size: int = FLEX_DOC_MASK_BLOCK_SIZE,
) -> BlockMask | None:
    """Build a `<BOS>` intra-document causal `BlockMask` on CUDA when `T` aligns to `block_size` (`<= 8` args)."""
    bsz, seq_len = idx.shape
    if not (idx.is_cuda and seq_len >= block_size and seq_len % block_size == 0):
        return None
    docs = (idx == int(bos_token_id)).to(torch.int32).cumsum(dim=1)

    def doc_mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        return (q_idx >= kv_idx) & (docs[b, q_idx] == docs[b, kv_idx])

    return create_block_mask(
        doc_mask_mod,
        B=bsz,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=idx.device,
        BLOCK_SIZE=block_size,
    )


# ==============================================================================
# Pillar 1: Surprisal-Conditioned 128D Welch-Rectified Bi-Trigger Walsh Copy Gate
# ==============================================================================
class SurprisalConditionedWalshBiTriggerCopyGate(nn.Module):
    """`<BOS>`-masked $p=3$ associative induction gate with Welch floor subtraction & online IDF gating."""

    def __init__(
        self,
        sem_ket_dim: int,
        ctx_dim: int,
        vocab_size: int = VALID_FINEWEB_VOCAB_SIZE,
        boson_power: float = DEFAULT_COPY_BOSON_POWER,
        max_logit_gain: float = DEFAULT_COPY_MAX_LOGIT_GAIN,
    ) -> None:
        super().__init__()
        self.sem_ket_dim = int(sem_ket_dim)
        self.ctx_dim = int(ctx_dim)
        self.boson_power = float(boson_power)
        self.max_logit_gain = float(max_logit_gain)

        real_sem = 2 * self.sem_ket_dim
        # Shared unitary semantic anchor U_sem preserves <U_sem psi(A) | U_sem psi(A)> == 1.0!
        self.u_sem_proj = nn.Linear(real_sem, real_sem, bias=False)
        self.fuzzy_key_delta = nn.Linear(real_sem, real_sem, bias=False)
        nn.init.eye_(self.u_sem_proj.weight)
        nn.init.zeros_(self.fuzzy_key_delta.weight)

        self.register_buffer(
            "wh_codebook",
            build_balanced_128d_walsh_codebook(vocab_size, WALSH_INDUCTION_DIM),
            persistent=False,
        )
        self.wh_blend_logit = nn.Parameter(torch.tensor(0.85, dtype=torch.float32))
        self.surprisal_mid_shift = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.bigram_boost_logit = nn.Parameter(
            torch.tensor(math.log(math.expm1(BIGRAM_PHRASE_BOOST_MULTIPLIER)), dtype=torch.float32)
        )
        self.ctx_gate_proj = nn.Linear(self.ctx_dim, 1, bias=True)
        nn.init.zeros_(self.ctx_gate_proj.weight)
        nn.init.constant_(self.ctx_gate_proj.bias, DEFAULT_COPY_GATE_INIT_BIAS)

        self.vacuum_logit = nn.Parameter(
            torch.tensor(DEFAULT_COPY_VACUUM_INIT_LOGIT, dtype=torch.float32)
        )

    def _rectified_pboson_overlap(
        self, q_re: Tensor, q_im: Tensor, k_re: Tensor, k_im: Tensor, floor_sq: float
    ) -> Tensor:
        """Compute Welch-rectified causal `(|<q_t | k_{s-1}>|^2 - floor_sq)_+^p` (`<= 8` args)."""
        prev_k_re = F.pad(k_re[:, :-1, :], (0, 0, 1, 0))
        prev_k_im = F.pad(k_im[:, :-1, :], (0, 0, 1, 0))
        ov_re = torch.matmul(q_re, prev_k_re.transpose(-2, -1)) + torch.matmul(
            q_im, prev_k_im.transpose(-2, -1)
        )
        ov_im = torch.matmul(q_im, prev_k_re.transpose(-2, -1)) - torch.matmul(
            q_re, prev_k_im.transpose(-2, -1)
        )
        ov_sq = (ov_re.float().square() + ov_im.float().square()).clamp(0.0, 1.0)
        rect = ((ov_sq - floor_sq) * (1.0 / max(1.0e-5, 1.0 - floor_sq))).clamp(0.0, 1.0)
        return rect.pow(self.boson_power) if abs(self.boson_power - 1.0) > 1e-5 else rect

    def forward(
        self,
        idx: Tensor,
        sem_ket_re: Tensor,
        sem_ket_im: Tensor,
        ctx_feat: Tensor,
        doc_same: Tensor,
        online_uni_probs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return `(copy_log_odds, copy_conf)` of shapes `(B, T, T)` and `(B, T)` (`<= 8` args)."""
        _, seq_len = idx.shape
        d_s = self.sem_ket_dim
        sem_pack = torch.cat([sem_ket_re, sem_ket_im], dim=-1)
        q_raw = self.u_sem_proj(sem_pack)
        k_raw = q_raw + 0.25 * self.fuzzy_key_delta(sem_pack)
        qs_re, qs_im = normalize_complex_ket(q_raw[..., :d_s], q_raw[..., d_s:])
        ks_re, ks_im = normalize_complex_ket(k_raw[..., :d_s], k_raw[..., d_s:])
        sem_m1 = self._rectified_pboson_overlap(qs_re, qs_im, ks_re, ks_im, WELCH_SEMANTIC_FLOOR_SQ)

        safe_idx = idx.clamp(0, self.wh_codebook.size(0) - 1)
        wh_ket = F.embedding(safe_idx, self.wh_codebook).to(ctx_feat.dtype)
        prev_wh = F.pad(wh_ket[:, :-1, :], (0, 0, 1, 0))
        wh_ov_sq = torch.matmul(wh_ket, prev_wh.transpose(-2, -1)).float().square().clamp(0.0, 1.0)
        wh_rect = (
            (wh_ov_sq - WELCH_WALSH_FLOOR_SQ) * (1.0 / (1.0 - WELCH_WALSH_FLOOR_SQ))
        ).clamp(0.0, 1.0)
        wh_m1 = wh_rect.pow(self.boson_power) if abs(self.boson_power - 1.0) > 1e-5 else wh_rect

        w_wh = torch.sigmoid(self.wh_blend_logit.float())
        m1 = w_wh * wh_m1 + (1.0 - w_wh) * sem_m1

        # Zero-GEMM 2-token phrase coincidence: [x_{t-1}, x_t] == [x_{s-2}, x_{s-1}] via diagonal pad!
        if seq_len > 1:
            m1_prev_diag = F.pad(m1[:, :-1, :-1], (1, 0, 1, 0))
            m2 = m1 * m1_prev_diag
        else:
            m2 = torch.zeros_like(m1)

        # Streaming online unigram surprisal I(x_t) = -ln P_uni(x_t) damps lone D0 stopwords (" the")
        uni_idx = idx.clamp(0, online_uni_probs.numel() - 1)
        tok_p = online_uni_probs.detach()[uni_idx].clamp_min(1.0e-9)
        surprisal_t = -torch.log(tok_p).float()
        mid_nats = SURPRISAL_IDF_MIDPOINT_NATS + 0.5 * torch.tanh(self.surprisal_mid_shift.float())
        idf_gate = torch.sigmoid(SURPRISAL_IDF_SLOPE * (surprisal_t - mid_nats)).unsqueeze(-1)
        bi_mult = F.softplus(self.bigram_boost_logit.float())

        t_idx = torch.arange(seq_len, device=idx.device).unsqueeze(1)
        s_idx = torch.arange(seq_len, device=idx.device).unsqueeze(0)
        strict_past = ((t_idx > s_idx) & (s_idx >= 1)).unsqueeze(0)
        doc_prev = F.pad(doc_same[:, :, :-1], (1, 0), value=False)
        valid_pair = (strict_past & doc_same & doc_prev).to(torch.float32)

        doc_prev2 = F.pad(doc_prev[:, :, :-1], (1, 0), value=False)
        strict_past2 = ((t_idx > s_idx) & (s_idx >= 2)).unsqueeze(0)
        valid_bi = (strict_past2 & doc_same & doc_prev & doc_prev2).to(torch.float32)

        masked_scores = idf_gate * (m1 * valid_pair) + bi_mult * (m2 * valid_bi)
        eps_vac = F.softplus(self.vacuum_logit.float()).clamp_min(1.0e-4)
        w_copy = masked_scores / (masked_scores.sum(dim=-1, keepdim=True) + eps_vac)
        copy_conf = w_copy.sum(dim=-1)

        # MuP-exact unit-L2 normalization (1/sqrt(D)) before linear gate prevents logit explosion
        ctx_unit = ctx_feat * (1.0 / math.sqrt(float(self.ctx_dim)))
        raw_gain = self.max_logit_gain * torch.sigmoid(self.ctx_gate_proj(ctx_unit).float())
        copy_log_odds = (raw_gain * w_copy).to(ctx_feat.dtype)
        return copy_log_odds, copy_conf.to(ctx_feat.dtype)


# ==============================================================================
# Pillar 2: Zero-Expansion In-Attention Head-Pair Complementary Schur Router
# ==============================================================================
class InAttentionComplementarySchurRouter(nn.Module):
    """Rank-$R=2$ head-pair Complementary Schur `(AND, Disagreement)` router inside `1x` width `2d`."""

    def __init__(self, real_dim: int) -> None:
        super().__init__()
        pair_span = 2 * IN_ATTN_SCHUR_PAIR_RANK * COMPLEX_HEAD_DIM
        if real_dim % pair_span != 0:
            raise ValueError(
                f"real_dim ({real_dim}) must be divisible by {pair_span} for R=2 head-pair Schur routing"
            )
        self.real_dim = int(real_dim)
        self.num_pairs = self.real_dim // pair_span
        self.pair_rank = IN_ATTN_SCHUR_PAIR_RANK

        self.partner_proj = nn.Linear(self.real_dim, self.real_dim, bias=False)
        nn.init.normal_(self.partner_proj.weight, std=0.5 / math.sqrt(float(self.real_dim)))
        self.schur_mix_logit = nn.Parameter(torch.tensor(IN_ATTN_SCHUR_INIT_LOGIT, dtype=torch.float32))
        self.eval_channel_mode: SchurEvalMaskMode = "normal"

    def _to_trace_unit_pair_beams(self, raw: Tensor, bsz: int, seq_len: int) -> tuple[Tensor, Tensor]:
        """Reshape `(B, T, D)` into trace-1 `(B, T, G_pair, R=2, 16)` `(u_re, u_im)` (`<= 8` args)."""
        shaped = raw.view(bsz, seq_len, self.num_pairs, self.pair_rank, 2 * COMPLEX_HEAD_DIM).float()
        u_re = shaped[..., :COMPLEX_HEAD_DIM]
        u_im = shaped[..., COMPLEX_HEAD_DIM:]
        tr = (u_re.square() + u_im.square()).sum(dim=(-2, -1), keepdim=True).clamp_min(DEFAULT_TRACE_EPS)
        inv_s = torch.rsqrt(tr)
        return u_re * inv_s, u_im * inv_s

    def compute_even_odd_packets(
        self, attn_flat: Tensor, h_norm: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return `(even_pkt, odd_pkt, p_and, u_a_re, u_a_im, u_b_re, u_b_im)` (`<= 8` args)."""
        bsz, seq_len, _ = attn_flat.shape
        u_a_re, u_a_im = self._to_trace_unit_pair_beams(attn_flat, bsz, seq_len)
        u_b_re, u_b_im = self._to_trace_unit_pair_beams(self.partner_proj(h_norm), bsz, seq_len)

        # Coherent head-pair probe |psi> in C^{16} with zero extra linear projections
        pair_heads = attn_flat.view(
            bsz, seq_len, self.num_pairs, self.pair_rank, 2 * COMPLEX_HEAD_DIM
        ).float()
        mean_head = pair_heads.mean(dim=-2)
        psi_re, psi_im = normalize_complex_ket(
            mean_head[..., :COMPLEX_HEAD_DIM], mean_head[..., COMPLEX_HEAD_DIM:]
        )

        and_re, and_im, comp_re, comp_im, p_and = zero_matrix_complementary_schur_probe_action(
            u_a_re, u_a_im, u_b_re, u_b_im, psi_re, psi_im
        )
        p_w = p_and.unsqueeze(-1)
        even_pkt = p_w * torch.cat([and_re, and_im], dim=-1)  # (B, T, G_pair, 32)
        odd_pkt = (1.0 - p_w) * torch.cat([comp_re, comp_im], dim=-1)  # (B, T, G_pair, 32)
        return even_pkt, odd_pkt, p_and, u_a_re, u_a_im, u_b_re, u_b_im

    def forward(self, attn_flat: Tensor, h_norm: Tensor) -> Tensor:
        """Route `[even_consensus, odd_disagreement]` into head pairs `(2g, 2g+1)` (`0` graph breaks)."""
        bsz, seq_len, d_real = attn_flat.shape
        even_pkt, odd_pkt, _, _, _, _, _ = self.compute_even_odd_packets(attn_flat, h_norm)
        if self.eval_channel_mode == "drop_odd":
            odd_pkt = torch.zeros_like(odd_pkt)
        elif self.eval_channel_mode == "drop_even":
            even_pkt = torch.zeros_like(even_pkt)
        stacked = torch.stack([even_pkt, odd_pkt], dim=3).reshape(bsz, seq_len, d_real)
        mix = torch.sigmoid(self.schur_mix_logit).to(attn_flat.dtype)
        return attn_flat + mix * stacked.to(attn_flat.dtype)

    @torch.no_grad()
    def inspect_pair_purity_on_sample(self, attn_flat: Tensor, h_norm: Tensor) -> dict[str, float]:
        """Compute exact $16 \\times 16$ `mean_p_and`, `even_purity`, and `odd_purity` at eval time."""
        _, _, p_and, ua_re, ua_im, ub_re, ub_im = self.compute_even_odd_packets(attn_flat, h_norm)
        ua_c = torch.complex(ua_re[:2, :16], ua_im[:2, :16])
        ub_c = torch.complex(ub_re[:2, :16], ub_im[:2, :16])
        rho_a = torch.matmul(ua_c.transpose(-2, -1), ua_c.conj())
        rho_b = torch.matmul(ub_c.transpose(-2, -1), ub_c.conj())
        rho_and, rho_comp, _ = explicit_complementary_schur_matrix_pair(rho_a, rho_b)
        pur_e = (rho_and.real.square() + rho_and.imag.square()).sum(dim=(-2, -1)).mean()
        pur_o = (rho_comp.real.square() + rho_comp.imag.square()).sum(dim=(-2, -1)).mean()
        return {
            "mean_p_and": round(float(p_and.mean().item()), 5),
            "even_purity": round(float(pur_e.item()), 5),
            "odd_purity": round(float(pur_o.item()), 5),
            "schur_mix_gate": round(float(torch.sigmoid(self.schur_mix_logit).item()), 4),
        }


# ==============================================================================
# Pillar 3: Compiled <BOS> FlexAttention + In-Attention Schur U-Net Block & LM
# ==============================================================================
def _apply_complex_rope_bhtd(x_bhtd: Tensor, cos_4d: Tensor, sin_4d: Tensor) -> Tensor:
    """Apply complex unitary RoPE to `(B, H, T, 32)` in native `x_bhtd.dtype` (`<= 8` args)."""
    x_re, x_im = x_bhtd[..., :COMPLEX_HEAD_DIM], x_bhtd[..., COMPLEX_HEAD_DIM:]
    c, s = cos_4d.to(x_bhtd.dtype), sin_4d.to(x_bhtd.dtype)
    return torch.cat([x_re * c - x_im * s, x_re * s + x_im * c], dim=-1)


_COMPILED_FLEX_ATTN_FN: Any = None


def _get_compiled_flex_attn() -> Any:
    """Return cached `torch.compile(flex_attention)` for fused CUDA `<BOS>` eval/probe passes."""
    global _COMPILED_FLEX_ATTN_FN
    if _COMPILED_FLEX_ATTN_FN is None:
        _COMPILED_FLEX_ATTN_FN = torch.compile(flex_attention, dynamic=False)
    return _COMPILED_FLEX_ATTN_FN


class FusedUnifiedSchurBatonBlock(nn.Module):
    """U-Net Hilbert-highway block with `<BOS>` `FlexAttention`, In-Attn Comp Schur, & Quantum MLP."""

    def __init__(self, cfg: FusedUnifiedBatonConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.real_dim = 2 * cfg.complex_dim
        self.head_real_dim = 2 * COMPLEX_HEAD_DIM
        self.num_heads = self.real_dim // self.head_real_dim
        self.depth_scale = DEPTH_RESIDUAL_INIT_SCALE / math.sqrt(
            max(1.0, float(cfg.num_layers) / 4.0)
        )

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
        self.in_attn_schur: InAttentionComplementarySchurRouter | None = (
            InAttentionComplementarySchurRouter(real_dim=self.real_dim)
            if cfg.schur_mode == "in_attn_head_pair"
            else None
        )

        hidden = DEFAULT_W78_MLP_EXPANSION * self.real_dim
        self.mlp_up = nn.Linear(self.real_dim, hidden, bias=False)
        self.mlp_phase_gate = nn.Linear(self.real_dim, hidden, bias=False)
        self.mlp_down = nn.Linear(hidden, self.real_dim, bias=False)
        nn.init.normal_(self.mlp_up.weight, std=init_std)
        nn.init.normal_(self.mlp_phase_gate.weight, std=init_std)
        nn.init.zeros_(self.mlp_down.weight)

    def _encode_clean_attn_and_norm(
        self,
        x_in: Tensor,
        cos_4d: Tensor,
        sin_4d: Tensor,
        flex_bm: BlockMask | None,
        dense_doc_mask: Tensor | None,
        same_prev_tok: Tensor,
        v_0: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute `(attn_clean_flat, h_norm, v_next_0)` (`<= 8` args)."""
        bsz, seq_len, d_real = x_in.shape
        h_norm = F.rms_norm(x_in, (d_real,), eps=1.0e-6)
        q_raw, k_raw, v_raw = self.qkv_proj(h_norm).chunk(3, dim=-1)
        q = q_raw.view(bsz, seq_len, self.num_heads, self.head_real_dim).permute(0, 2, 1, 3)
        k = k_raw.view(bsz, seq_len, self.num_heads, self.head_real_dim).permute(0, 2, 1, 3)
        v = v_raw.view(bsz, seq_len, self.num_heads, self.head_real_dim).permute(0, 2, 1, 3)

        q = _apply_complex_rope_bhtd(F.rms_norm(q, (self.head_real_dim,), eps=1.0e-6), cos_4d, sin_4d)
        k = _apply_complex_rope_bhtd(F.rms_norm(k, (self.head_real_dim,), eps=1.0e-6), cos_4d, sin_4d)

        if seq_len > 1:
            sp = same_prev_tok.unsqueeze(1)
            k_shifted = torch.where(sp, k[:, :, :-1, 24:], k[:, :, 1:, 24:])
            k = torch.cat([k[..., :24], torch.cat([k[:, :, :1, 24:], k_shifted], dim=2)], dim=-1)

        if v_0 is None:
            v_next_0 = v
        else:
            v_next_0 = v_0
            assert self.v0_mix_logit is not None
            v = torch.lerp(v, v_0, torch.sigmoid(self.v0_mix_logit).to(v.dtype))

        if flex_bm is not None:
            flex_fn = (
                flex_attention
                if torch.compiler.is_compiling() or not q.is_cuda
                else _get_compiled_flex_attn()
            )
            attn_out = flex_fn(q, k, v, block_mask=flex_bm, scale=DEFAULT_W77_ATTN_SCALE)
        elif dense_doc_mask is not None:
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=dense_doc_mask, scale=DEFAULT_W77_ATTN_SCALE
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=DEFAULT_W77_ATTN_SCALE
            )

        v_unit = F.normalize(v.float(), dim=-1, eps=1.0e-6).to(attn_out.dtype)
        echo_proj = (attn_out * v_unit).sum(dim=-1, keepdim=True) * v_unit
        attn_clean = (
            attn_out - torch.sigmoid(self.xsa_alpha).to(attn_out.dtype) * echo_proj
        ).permute(0, 2, 1, 3).reshape(bsz, seq_len, d_real)
        return attn_clean, h_norm, v_next_0

    def forward(
        self,
        x: Tensor,
        x_0: Tensor,
        cos_4d: Tensor,
        sin_4d: Tensor,
        flex_bm: BlockMask | None,
        dense_doc_mask: Tensor | None,
        same_prev_tok: Tensor,
        v_0: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Execute one `<BOS>`-masked U-Net block with zero graph breaks (`<= 8` args)."""
        d_real = x.size(-1)
        lam = self.block_lambdas.to(x.dtype)
        x_in = lam[0] * x + lam[1] * x_0

        attn_clean, h_norm, v_next_0 = self._encode_clean_attn_and_norm(
            x_in, cos_4d, sin_4d, flex_bm, dense_doc_mask, same_prev_tok, v_0
        )
        attn_routed = (
            self.in_attn_schur(attn_clean, h_norm)
            if self.in_attn_schur is not None
            else attn_clean
        )
        ds = self.depth_scale
        x_mid = x_in + ds * self.out_proj(attn_routed)

        m_in = F.rms_norm(x_mid, (d_real,), eps=1.0e-6)
        boson_amp = F.relu(self.mlp_up(m_in)).square()
        phase_mod = 1.0 + 0.25 * torch.tanh(self.mlp_phase_gate(m_in))
        x_out = x_mid + ds * self.mlp_down(boson_amp * phase_mod)
        return x_out, v_next_0


def _softcap_scatter_ce_chunk_step(
    x_c: Tensor,
    w_head: Tensor,
    uni_bias: Tensor,
    copy_idx_c: Tensor,
    copy_lo_c: Tensor,
    targets_c: Tensor,
    softcap: float,
) -> Tensor:
    """Pure tensor chunk step for `linear + uni_prior + copy_scatter + softcap_tanh + CE` (`<= 8` args)."""
    lg = F.linear(x_c, w_head) + uni_bias
    lg = lg.scatter_add(dim=-1, index=copy_idx_c, src=copy_lo_c.to(lg.dtype))
    capped = softcap * torch.tanh(lg.float() * (1.0 / softcap))
    return F.cross_entropy(capped.reshape(-1, capped.size(-1)), targets_c.reshape(-1))


class FusedUnifiedSchurBatonLM(nn.Module):
    """Flag-free `Wave 79` Unified Quantum Baton LM with In-Attention Schur & Compiled `<BOS>` `FlexAttention`."""

    def __init__(self, cfg: FusedUnifiedBatonConfig) -> None:
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
            [FusedUnifiedSchurBatonBlock(cfg, layer_idx=i) for i in range(self.num_layers)]
        )
        n_dec = max(1, self.num_layers // 2)
        self.skip_weights = nn.Parameter(torch.zeros((n_dec,), dtype=torch.float32))

        self.causal_mtp_sweep: HorizonPooledCausalMTPMPSSweep | None = (
            HorizonPooledCausalMTPMPSSweep(real_dim=self.real_dim)
            if cfg.use_horizon_mtp_mps
            else None
        )
        self.copy_gate = SurprisalConditionedWalshBiTriggerCopyGate(
            sem_ket_dim=self.complex_dim,
            ctx_dim=self.real_dim,
            vocab_size=cfg.vocab_size,
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
        self._compiled_trunk_fn: Any = None
        self._compiled_ce_chunk_fn: Any = None

    def enable_compilation(self) -> None:
        """Enable `torch.compile` on the pure U-Net highway and chunked softcap readout (`<= 8` args)."""
        if self._compiled_trunk_fn is None:
            self._compiled_trunk_fn = torch.compile(self._run_unet_blocks_pure, dynamic=False)
        if self._compiled_ce_chunk_fn is None:
            self._compiled_ce_chunk_fn = torch.compile(_softcap_scatter_ce_chunk_step, dynamic=False)

    def set_schur_eval_channel_mode(self, mode: SchurEvalMaskMode) -> None:
        """Set `normal`, `drop_odd`, or `drop_even` across all `InAttentionComplementarySchurRouter` blocks."""
        for blk in self.blocks:
            if blk.in_attn_schur is not None:
                blk.in_attn_schur.eval_channel_mode = mode

    def _get_rope_cos_sin(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Return cached `(cos_4d, sin_4d)` of shape `(1, 1, T, 16)` (`<= 8` args)."""
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
        flat = idx.reshape(-1).clamp(0, self.online_uni_token_probs.numel() - 1)
        inc = eta / float(max(1, flat.numel()))
        self.online_uni_token_probs.mul_(1.0 - eta)
        self.online_uni_token_probs.scatter_add_(
            0, flat, torch.full_like(flat, inc, dtype=torch.float32)
        )
        self.online_ingested_tokens.add_(flat.numel())

    def _embed_and_smear(self, idx: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return `(x_0, ket0_re, ket0_im, same_prev)` from input `idx` `(B, T)` (`<= 8` args)."""
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
        return x_0, ket0_re, ket0_im, same_prev

    def _run_unet_blocks_pure(
        self,
        x_0: Tensor,
        cos_4d: Tensor,
        sin_4d: Tensor,
        flex_bm: BlockMask | None,
        dense_doc_mask: Tensor | None,
        same_prev: Tensor,
    ) -> Tensor:
        """Pure tensor-in/tensor-out U-Net block stack ideal for `torch.compile` (`<= 8` args)."""
        x = x_0
        v_0: Tensor | None = None
        n_half = self.num_layers // 2
        enc_stack: list[Tensor] = []
        for l_idx, blk in enumerate(self.blocks):
            if self.num_layers >= 4 and l_idx >= n_half and enc_stack:
                skip_g = torch.sigmoid(self.skip_weights[l_idx - n_half]).to(x.dtype)
                x = x + skip_g * enc_stack.pop()
            x, v_0 = blk(x, x_0, cos_4d, sin_4d, flex_bm, dense_doc_mask, same_prev, v_0)
            if self.num_layers >= 4 and l_idx < n_half:
                enc_stack.append(x)
        return x

    def encode_trunk_features(
        self,
        idx: Tensor,
        targets_for_mtp: Tensor | None = None,
        force_eager: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Encode `idx` `(B, T)` into `(x_final, copy_log_odds, copy_conf, mtp_aux_loss)` (`<= 8` args)."""
        _, seq_len = idx.shape
        x_0, ket0_re, ket0_im, same_prev = self._embed_and_smear(idx)
        mask_pack = build_bos_anchor_sink_doc_masks(idx)
        cos_4d, sin_4d = self._get_rope_cos_sin(seq_len, idx.device)

        flex_bm: BlockMask | None = None
        if self.cfg.attn_backend == "auto_flex" and not force_eager:
            flex_bm = build_bos_doc_flex_block_mask(idx)
        dense_doc_mask: Tensor | None = None
        if flex_bm is None:
            t_idx = torch.arange(seq_len, device=idx.device).unsqueeze(1)
            s_idx = torch.arange(seq_len, device=idx.device).unsqueeze(0)
            dense_doc_mask = ((s_idx <= t_idx).unsqueeze(0) & mask_pack.doc_same).unsqueeze(1)

        can_compile = (
            self._compiled_trunk_fn is not None
            and idx.is_cuda
            and not force_eager
            and all(
                blk.in_attn_schur is None or blk.in_attn_schur.eval_channel_mode == "normal"
                for blk in self.blocks
            )
        )
        if can_compile:
            x = self._compiled_trunk_fn(x_0, cos_4d, sin_4d, flex_bm, dense_doc_mask, same_prev)
        else:
            x = self._run_unet_blocks_pure(x_0, cos_4d, sin_4d, flex_bm, dense_doc_mask, same_prev)

        mtp_aux_loss: Tensor | None = None
        if self.causal_mtp_sweep is not None:
            fut_emb = self.wte_ket(targets_for_mtp).detach() if targets_for_mtp is not None else None
            delta_mtp, mtp_aux_loss = self.causal_mtp_sweep(x, fut_emb, targets_for_mtp)
            x = x + delta_mtp

        x_final = F.rms_norm(x, (self.real_dim,), eps=1.0e-6)
        copy_log_odds, copy_conf = self.copy_gate(
            idx, ket0_re, ket0_im, x_final, mask_pack.doc_same, self.online_uni_token_probs
        )
        return x_final, copy_log_odds, copy_conf, mtp_aux_loss

    def compute_logits(self, idx: Tensor, x_final: Tensor, copy_log_odds: Tensor) -> Tensor:
        """Compute full `(B, T, V)` softcapped logits for eval/probing (`<= 8` args)."""
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
        """Execute chunked `bfloat16` training or evaluation forward pass (`<= 8` args)."""
        bsz, seq_len = idx.shape
        x_final, copy_log_odds, _, mtp_aux_loss = self.encode_trunk_features(
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
        uni_bias = uni_scale * (
            torch.log(self.online_uni_token_probs.detach().clamp_min(1.0e-9))
            .to(x_final.dtype)
            .view(1, 1, -1)
        )
        copy_index = idx.unsqueeze(1).expand(bsz, seq_len, seq_len)
        ce_fn = (
            self._compiled_ce_chunk_fn
            if (self._compiled_ce_chunk_fn is not None and idx.is_cuda)
            else _softcap_scatter_ce_chunk_step
        )

        chunk_losses: list[Tensor] = []
        for b0 in range(0, bsz, chunk_b):
            b1 = min(bsz, b0 + chunk_b)
            ce_c = ce_fn(
                x_final[b0:b1],
                self.lm_head.weight,
                uni_bias,
                copy_index[b0:b1],
                copy_log_odds[b0:b1],
                targets[b0:b1],
                self.cfg.logit_softcap,
            )
            chunk_losses.append(ce_c * float(b1 - b0))

        total_ce = torch.stack(chunk_losses).sum() / float(bsz)
        total_loss = total_ce + mtp_aux_loss if mtp_aux_loss is not None else total_ce
        if ingest_online:
            self.ingest_online_unigram_batch(idx)
        return None, total_loss

    @torch.no_grad()
    def inspect_routing_telemetry_on_sample(self, x_sample: Tensor) -> dict[str, Any]:
        """Eagerly inspect layerwise `mean_p_and`, `even_purity`, `odd_purity`, & `mtp` weights (`<= 8` args)."""
        was_tr = self.training
        self.eval()
        xb = x_sample[:2]
        _, seq_len = xb.shape
        x_0, ket0_re, ket0_im, same_prev = self._embed_and_smear(xb)
        mask_pack = build_bos_anchor_sink_doc_masks(xb)
        t_idx = torch.arange(seq_len, device=xb.device).unsqueeze(1)
        s_idx = torch.arange(seq_len, device=xb.device).unsqueeze(0)
        dense_doc_mask = ((s_idx <= t_idx).unsqueeze(0) & mask_pack.doc_same).unsqueeze(1)
        cos_4d, sin_4d = self._get_rope_cos_sin(seq_len, xb.device)

        x = x_0
        v_0: Tensor | None = None
        n_half = self.num_layers // 2
        enc_stack: list[Tensor] = []
        per_layer: list[dict[str, float]] = []

        for l_idx, blk in enumerate(self.blocks):
            if self.num_layers >= 4 and l_idx >= n_half and enc_stack:
                skip_g = torch.sigmoid(self.skip_weights[l_idx - n_half]).to(x.dtype)
                x = x + skip_g * enc_stack.pop()
            lam = blk.block_lambdas.to(x.dtype)
            x_in = lam[0] * x + lam[1] * x_0
            attn_clean, h_norm, v_0 = blk._encode_clean_attn_and_norm(
                x_in, cos_4d, sin_4d, None, dense_doc_mask, same_prev, v_0
            )
            if blk.in_attn_schur is not None:
                snap = blk.in_attn_schur.inspect_pair_purity_on_sample(attn_clean, h_norm)
                snap["layer"] = float(l_idx)
                per_layer.append(snap)
            x, _ = blk(x, x_0, cos_4d, sin_4d, None, dense_doc_mask, same_prev, v_0 if l_idx > 0 else None)
            if self.num_layers >= 4 and l_idx < n_half:
                enc_stack.append(x)

        mps_p = 0.0
        h_w: list[float] = []
        if self.causal_mtp_sweep is not None:
            x, _ = self.causal_mtp_sweep(x)
            if self.causal_mtp_sweep._last_mps_pand is not None:
                mps_p = round(float(self.causal_mtp_sweep._last_mps_pand.item()), 5)
            h_w = [
                round(float(v), 4)
                for v in F.softmax(self.causal_mtp_sweep.horizon_pool_logits.detach().float(), dim=0).tolist()
            ]
        x_final = F.rms_norm(x, (self.real_dim,), eps=1.0e-6)
        _, copy_conf = self.copy_gate(
            xb, ket0_re, ket0_im, x_final, mask_pack.doc_same, self.online_uni_token_probs
        )
        if was_tr:
            self.train()
        n_l = float(max(1, len(per_layer)))
        return {
            "has_schur_gate": bool(len(per_layer) > 0),
            "mean_p_and": round(sum(d["mean_p_and"] for d in per_layer) / n_l, 5) if per_layer else 0.0,
            "mean_even_purity": round(sum(d["even_purity"] for d in per_layer) / n_l, 5) if per_layer else 0.0,
            "mean_odd_purity": round(sum(d["odd_purity"] for d in per_layer) / n_l, 5) if per_layer else 0.0,
            "mps_mean_p_and": mps_p,
            "mtp_horizon_weights": h_w,
            "wh_exact_copy_weight": round(float(torch.sigmoid(self.copy_gate.wh_blend_logit).item()), 4),
            "mean_copy_confidence": round(float(copy_conf.float().mean().item()), 5),
            "per_layer": per_layer,
        }


# ==============================================================================
# MuP + PolarExpress StiefelMuon Optimizer & Adapter Builder (`Wave 79`)
# ==============================================================================
def build_wave79_mup_optimizers(
    model: FusedUnifiedSchurBatonLM,
    warmup_steps: int = 40,
) -> list[torch.optim.Optimizer]:
    """Build `PolarExpress` `StiefelMuon` + $\\mu\\text{P}$-calibrated `Adam` groups (`<= 8` args)."""
    muon_params: list[nn.Parameter] = []
    wte_params: list[nn.Parameter] = [model.wte_ket.weight]
    head_params: list[nn.Parameter] = [model.lm_head.weight]
    copy_params: list[nn.Parameter] = []
    scalar_params: list[nn.Parameter] = []

    excluded_ids = {id(model.wte_ket.weight), id(model.lm_head.weight)}
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in excluded_ids:
            continue
        if (
            p.ndim >= 2
            and min(p.shape[-2], p.shape[-1]) >= 8
            and "ctx_gate_proj" not in name
            and "smear_gate" not in name
        ):
            muon_params.append(p)
        elif name.startswith("copy_gate."):
            copy_params.append(p)
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
            {"params": copy_params, "lr": W79_COPY_SCALAR_LR, "group_name": "copy_gate_scalars"},
            {"params": scalar_params, "lr": W77_SCALAR_BASE_LR, "group_name": "scalars"},
        ],
        betas=(0.80, 0.95),
        eps=1.0e-8,
        fused=is_cuda,
    )
    return [muon_opt, adam_opt]


class Wave79FusedAdapter(DiagnosticModelAdapter):
    """Polymorphic `DiagnosticModelAdapter` wrapping `FusedUnifiedSchurBatonLM` (`<= 8` args)."""

    def __init__(
        self,
        key: str,
        label: str,
        model: FusedUnifiedSchurBatonLM,
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
        self.fused_model = model
        self.optimizers = optimizers
        self.use_wsd_schedule = bool(use_wsd_schedule)
        self.base_lrs: list[list[float]] = [
            [float(pg["lr"]) for pg in opt.param_groups] for opt in optimizers
        ]

    def _compute_lr_multiplier(self, step: int, total_steps: int) -> float:
        """Compute warmup + cosine (sprint) or `WSD` linear quench (speedrun) LR multiplier."""
        warmup = max(10, min(250, int(0.05 * total_steps)))
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
        self.fused_model.train()
        lr_mult = self._compute_lr_multiplier(step, total_steps)
        for opt, base_group_lrs in zip(self.optimizers, self.base_lrs):
            for pg, base_lr in zip(opt.param_groups, base_group_lrs):
                pg["lr"] = base_lr * lr_mult
            opt.zero_grad(set_to_none=True)

        amp_on = x_batch.is_cuda
        with torch.amp.autocast(
            device_type=x_batch.device.type, dtype=torch.bfloat16, enabled=amp_on
        ):
            _, loss = self.fused_model(
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
        was_train = self.fused_model.training
        self.fused_model.eval()
        amp_on = x_batch.is_cuda
        with torch.amp.autocast(
            device_type=x_batch.device.type, dtype=torch.bfloat16, enabled=amp_on
        ):
            logits_raw, _ = self.fused_model(
                x_batch, targets=None, ingest_online=False, return_logits=True
            )
        if was_train:
            self.fused_model.train()
        assert logits_raw is not None
        logits = logits_raw[..., :VALID_FINEWEB_VOCAB_SIZE].float()
        span = logits.amax(dim=-1) - logits.median(dim=-1).values
        return F.log_softmax(logits, dim=-1), span


def build_wave79_variant_adapter(
    variant: Wave79VariantKind,
    device: torch.device,
    seed: int = 1337,
    use_wsd_schedule: bool = False,
    enable_compile: bool | None = None,
    complex_dim_override: int | None = None,
) -> Wave79FusedAdapter:
    """Instantiate any `Wave79VariantKind` cleanly via explicit strategy dispatch (`<= 8` args)."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    d_c = int(complex_dim_override) if complex_dim_override is not None else 256
    spec_map: dict[Wave79VariantKind, tuple[str, FusedUnifiedBatonConfig]] = {
        Wave79VariantKind.V0_SURPRISAL_WH_INDUCTION_4L_D256: (
            "V0: Surprisal-Weighted 128D Bi-Trigger Walsh Copy + R4c Baton (4L, d=256, Eager)",
            FusedUnifiedBatonConfig(
                complex_dim=d_c,
                num_layers=4,
                schur_mode="none",
                use_horizon_mtp_mps=False,
                attn_backend="dense_sdpa",
                compile_backbone=False,
            ),
        ),
        Wave79VariantKind.V1_IN_ATTN_COMP_SCHUR_4L_D256: (
            "V1: In-Attention Rank-2 Head-Pair Complementary Schur + Bi-Trigger Copy (4L, d=256, Eager)",
            FusedUnifiedBatonConfig(
                complex_dim=d_c,
                num_layers=4,
                schur_mode="in_attn_head_pair",
                use_horizon_mtp_mps=False,
                attn_backend="dense_sdpa",
                compile_backbone=False,
            ),
        ),
        Wave79VariantKind.V2_FUSED_UNIFIED_4L_D256: (
            "V2: Compiled FlexAttention + In-Attn Comp Schur + Horizon MTP MPS (4L, d=256)",
            FusedUnifiedBatonConfig(
                complex_dim=d_c,
                num_layers=4,
                schur_mode="in_attn_head_pair",
                use_horizon_mtp_mps=True,
                attn_backend="auto_flex",
                compile_backbone=True,
            ),
        ),
        Wave79VariantKind.V3_FUSED_UNIFIED_6L_D256: (
            "V3: Deep 6L Compiled FlexAttention + In-Attn Comp Schur + Horizon MTP MPS (6L, d=256)",
            FusedUnifiedBatonConfig(
                complex_dim=d_c,
                num_layers=6,
                schur_mode="in_attn_head_pair",
                use_horizon_mtp_mps=True,
                attn_backend="auto_flex",
                compile_backbone=True,
            ),
        ),
    }
    label, cfg = spec_map[variant]
    model = FusedUnifiedSchurBatonLM(cfg).to(device)
    should_compile = cfg.compile_backbone if enable_compile is None else bool(enable_compile)
    if should_compile and device.type == "cuda":
        model.enable_compilation()
    warmup = 200 if use_wsd_schedule else 40
    opts = build_wave79_mup_optimizers(model, warmup_steps=warmup)
    return Wave79FusedAdapter(
        key=variant.value,
        label=label,
        model=model,
        optimizers=opts,
        device=device,
        use_wsd_schedule=use_wsd_schedule,
    )
