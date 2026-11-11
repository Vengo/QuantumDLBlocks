"""Cross-Architecture Mechanistic X-Ray & Stratified Nat-Gap Attribution (`Wave 76` / `EXPERIMENT_PLAN_096.md`).

Provenance & Mathematical Rationale
-----------------------------------
Built for `Wave 76` (`EXPERIMENT_PLAN_096.md`) following Rules 1–8 of `docs/SCRIPT_BEST_PRACTICES.md`
to diagnose why shallow quantum statistical-mechanics language models (`Waves 73–75`, reaching
`129.7k–162.2k tok/s` via `Theorems 14–16`) plateau near `5.2158` Val CE (`300M` tokens, `d=64`)
and `5.7809` Val CE (`8.19M` tokens, `d=256`), and to isolate a quantitative path to `< 3.28` Val CE:

1. **When to Use It**:
   - Whenever evaluating any quantum, hybrid, or shallow architecture checkpoint on FineWeb Shard 0
     or comparing short-horizon (`2M–10M` token) runs across known repository architectures (`Waves 45–75`).
   - Decomposes single-scalar `Val CE` into exact nat-by-nat differential attribution against a
     pretrained `CompiledLeaderboardClassicalGPT` anchor (`550M @ 3.35 CE` / `300M @ 3.58 CE`).

2. **Stratified Attribution Axes (`compute_stratified_nat_gap_xray`)**:
   - **Axis 1 — In-Document Copy (`y_t in x_{<=t}` within current `<BOS>` document) vs. Parametric Recall**,
     plus immediate self-echo (`y_t == x_t`).
   - **Axis 2 — 10 Empirical Zipf Frequency Deciles (`D0..D9`)** + Head (`D0–D1`), Body (`D2–D5`), Tail (`D6–D9`).
   - **Axis 3 — Intra-Document Horizon (`pos_in_doc` in `[0..3], [4..31], [32..127], [128+]`)**.
   - **Axis 4 — Reference Surprisal / Entropy Regimes (`Sharp < 1.5`, `Medium 1.5..4.0`, `Ambiguous >= 4.0` nats)**.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..models.compiled_leaderboard_classical_gpt import (
    CompiledLeaderboardClassicalConfig,
    CompiledLeaderboardClassicalGPT,
    CompiledLeaderboardOptimizerConfig,
    build_compiled_leaderboard_optimizers,
)
from .exact_mixed_state_baton_width_scaling_lab import ExactMixedStateFourCompartmentBatonLM
from .fused_bf16_four_compartment_baton_lab import (
    DEFAULT_MPS_NGRAM_BLEND,
    DEFAULT_PURITY_FLOOR_OFFSET,
    DEFAULT_PURITY_MIN_CLAMP,
    DEFAULT_TRACE_EPS,
    DEFAULT_TWO_HOP_MIX_ALPHA,
    FusedBF16FourCompartmentBatonConfig,
    FusedBF16FourCompartmentBatonLayer,
    FusedBF16FourCompartmentBatonLM,
    apply_complex_rope,
    apply_rank_beams_to_probe,
    evaluate_rank_beams_born_gates,
    factorized_rank_beams_schur_intersect,
    normalize_complex_ket,
)
from .noclick_vacuum_and_smith_sgd_lab import _apply_global_optimizer_step
from .shallow_longseq_wavepacket_lm_lab import DEFAULT_SOFTCAP_LOGIT
from .zero_ppmi_byte_pair_cholesky_lab import (
    ZeroPPMIBytePairCholeskyGPT,
    ZeroPPMIBytePairCholeskySpec,
    ZeroPPMIBytePairVariantMode,
    build_zero_ppmi_byte_pair_optimizers,
    initialize_zero_ppmi_byte_pair_model_,
)

# ==============================================================================
# Top-of-File Named Constants (Synchronized with EXPERIMENT_PLAN_096.md Section 3)
# ==============================================================================
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
DEFAULT_FINEWEB_CACHE_DIR: Path = REPO_ROOT / ".cache" / "fineweb_100M"
DEFAULT_TEACHER_CKPT_DIR: Path = (
    REPO_ROOT / ".cache" / "checkpoints" / "w45b_classical_teacher_ckpt50m_l12_d768_b262k"
)
TEACHER_50M_CKPT_PATH: Path = DEFAULT_TEACHER_CKPT_DIR / "ckpt_step00191_0050M_m0050M.pt"
TEACHER_300M_CKPT_PATH: Path = DEFAULT_TEACHER_CKPT_DIR / "ckpt_step01145_0300M_m0300M.pt"
TEACHER_550M_CKPT_PATH: Path = DEFAULT_TEACHER_CKPT_DIR / "ckpt_step02099_0550M_m0550M.pt"

VALID_FINEWEB_VOCAB_SIZE: int = 50257
FINEWEB_BOS_TOKEN_ID: int = 50256
FINEWEB_HEADER_INT32_COUNT: int = 256
UNIGRAM_CALIBRATION_SCAN_TOKENS: int = 2_000_000
NUM_ZIPF_DECILES: int = 10
NUM_ECE_BINS: int = 10
DOC_POS_BUCKET_EDGES: tuple[int, int, int] = (4, 32, 128)
TEACHER_ENTROPY_SHARP_MAX: float = 1.5
TEACHER_ENTROPY_MEDIUM_MAX: float = 4.0

DEFAULT_EMA_FAST_DECAY: float = 0.980
DEFAULT_EMA_SLOW_DECAY: float = 0.995
ASYMPTOTE_ALPHA_MIN: float = 0.08
ASYMPTOTE_ALPHA_MAX: float = 0.85
ASYMPTOTE_ALPHA_STEPS: int = 78

READOUT_SCALE_SWEEP_VALUES: tuple[float, ...] = (8.0, 12.0, 18.0, 24.0, 32.0, 40.0)
RIDGE_PROBE_LAMBDA: float = 10.0
SYNTHETIC_INDUCTION_SEQ_LENS: tuple[int, ...] = (64, 128, 256)
SYNTHETIC_MQAR_PAIR_COUNTS: tuple[int, ...] = (4, 8)

DEFAULT_BATON_BASE_LR: float = 3.0e-3
DEFAULT_BATON_MIN_LR_RATIO: float = 0.15
DEFAULT_CHOLESKY_BASE_MOMENTUM: float = 0.70
LOG_PROB_FLOOR: float = -30.0


class KnownArchitectureKind(str, Enum):
    """Strategy selector for known repository architectures & pretrained anchors (Rule 3)."""

    TEACHER_50M = "classical_teacher_50m"
    TEACHER_300M = "classical_teacher_300m"
    TEACHER_550M = "classical_teacher_550m"
    W74_BATON_1L_D64 = "w74_baton_1l_d64"
    W74_BATON_2L_D64 = "w74_baton_2l_d64"
    W74_BATON_2L_D256 = "w74_baton_2l_d256"
    W76_SHARP_BATON_2L_D64_P3 = "w76_sharp_baton_2l_d64_p3_docmask"
    W75_EXACT_MIXED_2L_D128 = "w75_exact_mixed_2l_d128"
    W45_CLASSICAL_2L_D512 = "w45_classical_2l_d512"
    W61_ZERO_PPMI_CHOLESKY_4L_D512 = "w61_zero_ppmi_cholesky_4l_d512"


@dataclass(frozen=True)
class ShortTrainConfig:
    """Hyperparameters for controlled short-horizon architecture sprints (`<= 8` fields, Rule 2)."""

    steps: int = 400
    batch_size: int = 32
    seq_len: int = 256
    eval_interval: int = 100
    eval_tokens: int = 16384
    wave_id: str = "W76"
    plan_id: str = "EXPERIMENT_PLAN_096"
    seed: int = 1337


@dataclass(frozen=True)
class SyntheticProbeConfig:
    """Hyperparameters for synthetic Induction, MQAR, and Readout Ceiling probes (`<= 8` fields, Rule 2)."""

    induction_steps: int = 180
    mqar_steps: int = 220
    overfit_steps: int = 220
    batch_size: int = 32
    eval_batches: int = 8
    vocab_size: int = VALID_FINEWEB_VOCAB_SIZE
    seed: int = 2026


@dataclass(frozen=True)
class TeacherReferenceProfile:
    """Cached token-level NLL and predictive entropy from a reference anchor on Shard-0 sequences."""

    anchor_name: str
    anchor_val_ce: float
    token_nll: Tensor
    token_entropy: Tensor


# ==============================================================================
# Clean Diagnostic Subclass: p-Boson + <BOS> Document-Masked Baton Layer (Rule 6)
# ==============================================================================
class SharpenedPBosonDocMaskedBatonLayer(FusedBF16FourCompartmentBatonLayer):
    """Diagnostic Baton layer testing $p$-boson Born exponent $p \\ge 1$ and `<BOS>` document masking.

    Provenance (`Wave 76` / `EXPERIMENT_PLAN_096.md` Hypothesis H3):
    Isolates whether `Wave 74`'s $p=1$ polynomial Born washout ($Z_t \\approx 1 + (t-1)/d$) and
    cross-`<BOS>` leakage in sequence attention, transition copy, and token-ID routing cause
    long-context induction degradation, without mutating frozen `Wave 74` code.
    """

    def __init__(
        self,
        config: FusedBF16FourCompartmentBatonConfig,
        boson_power: float = 3.0,
        mask_bos_documents: bool = True,
    ) -> None:
        super().__init__(config)
        self.boson_power = float(boson_power)
        self.mask_bos_documents = bool(mask_bos_documents)
        self.current_doc_same_mask: Tensor | None = None

    def _resolve_doc_mask(self, aux_pack: dict[str, Tensor] | None = None) -> Tensor | None:
        """Return active `(B, T, T)` boolean/float `<BOS>` document equality mask if enabled."""
        if not self.mask_bos_documents:
            return None
        if self.current_doc_same_mask is not None:
            return self.current_doc_same_mask
        if aux_pack is not None and "doc_same_mask" in aux_pack:
            return aux_pack["doc_same_mask"]
        return None

    def _compute_fused_transition_beams(
        self,
        ket_re: Tensor,
        ket_im: Tensor,
        trans_ema_re: Tensor,
        trans_ema_im: Tensor,
        hash_ket_re: Tensor,
        hash_ket_im: Tensor,
        causal_decay: Tensor,
        strict_past_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute $p$-boson sharpened, `<BOS>`-document-masked transition beams & copy weights (`<= 8` args)."""
        prev_ket_re = F.pad(ket_re[:, :-1, :], (0, 0, 1, 0))
        prev_ket_im = F.pad(ket_im[:, :-1, :], (0, 0, 1, 0))

        ov_re = torch.matmul(ket_re, prev_ket_re.transpose(-2, -1)) + torch.matmul(
            ket_im, prev_ket_im.transpose(-2, -1)
        )
        ov_im = torch.matmul(ket_im, prev_ket_re.transpose(-2, -1)) - torch.matmul(
            ket_re, prev_ket_im.transpose(-2, -1)
        )
        ov_sq = (ov_re.square() + ov_im.square()).clamp_min(0.0)
        if abs(self.boson_power - 1.0) > 1e-5:
            sharp_ov = torch.pow(ov_sq.float(), self.boson_power).to(ket_re.dtype)
        else:
            sharp_ov = ov_sq

        doc_mask = self._resolve_doc_mask(None)
        pair_doc_valid = strict_past_mask.unsqueeze(0)
        if doc_mask is not None:
            dm = doc_mask.to(ket_re.dtype)
            dm_prev = F.pad(dm[:, :, :-1], (1, 0), value=0.0)
            pair_doc_valid = pair_doc_valid * dm * dm_prev

        born_pull = sharp_ov * causal_decay.unsqueeze(0) * pair_doc_valid
        ctx_w_re = ov_re * born_pull
        ctx_w_im = ov_im * born_pull
        ctx_ket_re = torch.matmul(ctx_w_re, ket_re) - torch.matmul(ctx_w_im, ket_im)
        ctx_ket_im = torch.matmul(ctx_w_re, ket_im) + torch.matmul(ctx_w_im, ket_re)
        ctx_norm_re, ctx_norm_im = normalize_complex_ket(ctx_ket_re, ctx_ket_im)

        op_re = trans_ema_re.to(ket_re.dtype) + self.trans_learned_re.to(ket_re.dtype)
        op_im = trans_ema_im.to(ket_re.dtype) + self.trans_learned_im.to(ket_re.dtype)
        ema_ket_re = torch.matmul(ket_re, op_re.transpose(-2, -1)) - torch.matmul(
            ket_im, op_im.transpose(-2, -1)
        )
        ema_ket_im = torch.matmul(ket_re, op_im.transpose(-2, -1)) + torch.matmul(
            ket_im, op_re.transpose(-2, -1)
        )
        ema_norm_re, ema_norm_im = normalize_complex_ket(
            ema_ket_re + 0.35 * hash_ket_re + 0.15 * ket_re,
            ema_ket_im + 0.35 * hash_ket_im + 0.15 * ket_im,
        )
        in_ctx_copy_weights = sharp_ov * pair_doc_valid
        return ctx_norm_re, ctx_norm_im, ema_norm_re, ema_norm_im, in_ctx_copy_weights

    def _compute_fused_exterior_sequence_beams(
        self,
        ket_re: Tensor,
        ket_im: Tensor,
        mps_beams_re: Tensor,
        mps_beams_im: Tensor,
        mps_purity: Tensor,
        aux_pack: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute $p$-boson sharpened, `<BOS>`-document-masked exterior sequence beams (`<= 8` args)."""
        d = ket_re.shape[-1]
        seq_len = ket_re.shape[1]
        cos_pos = aux_pack["cos_pos"]
        sin_pos = aux_pack["sin_pos"]
        head_masks = aux_pack["head_masks"]

        probes_re = self.motif_probes_re.to(ket_re.dtype)
        probes_im = self.motif_probes_im.to(ket_re.dtype)
        proj_mid_re = torch.matmul(mps_beams_re, probes_re.transpose(0, 1)) + torch.matmul(
            mps_beams_im, probes_im.transpose(0, 1)
        )
        proj_mid_im = torch.matmul(mps_beams_im, probes_re.transpose(0, 1)) - torch.matmul(
            mps_beams_re, probes_im.transpose(0, 1)
        )
        pm_re_t = proj_mid_re.transpose(-2, -1)
        pm_im_t = proj_mid_im.transpose(-2, -1)
        motif_re = torch.matmul(pm_re_t, mps_beams_re) + torch.matmul(pm_im_t, mps_beams_im)
        motif_im = torch.matmul(pm_re_t, mps_beams_im) - torch.matmul(pm_im_t, mps_beams_re)
        motif_re, motif_im = normalize_complex_ket(
            motif_re + 0.25 * ket_re.unsqueeze(-2), motif_im + 0.25 * ket_im.unsqueeze(-2)
        )

        bsz = ket_re.shape[0]
        h_cnt = motif_re.shape[2]
        motif_packed = (
            torch.cat([motif_re, motif_im], dim=-1)
            .permute(2, 0, 1, 3)
            .reshape(h_cnt, bsz * seq_len, 2 * d)
        )
        w_top = torch.cat(
            [self.wq_re, self.wq_im, self.wk_re, self.wk_im, self.wv_re, self.wv_im], dim=-1
        )
        w_bot = torch.cat(
            [-self.wq_im, self.wq_re, -self.wk_im, self.wk_re, -self.wv_im, self.wv_re], dim=-1
        )
        w_qkv_packed = torch.cat([w_top, w_bot], dim=1).to(motif_packed.dtype)
        qkv_out = (
            torch.bmm(motif_packed, w_qkv_packed)
            .view(h_cnt, bsz, seq_len, 6, d)
            .permute(1, 0, 2, 3, 4)
        )
        q_re, q_im = normalize_complex_ket(qkv_out[..., 0, :], qkv_out[..., 1, :])
        k_re, k_im = normalize_complex_ket(qkv_out[..., 2, :], qkv_out[..., 3, :])
        v_re, v_im = normalize_complex_ket(qkv_out[..., 4, :], qkv_out[..., 5, :])
        q_re, q_im = apply_complex_rope(q_re, q_im, cos_pos.unsqueeze(1), sin_pos.unsqueeze(1))
        k_re, k_im = apply_complex_rope(k_re, k_im, cos_pos.unsqueeze(1), sin_pos.unsqueeze(1))

        k_pack_t = torch.cat([k_re, k_im], dim=-1).transpose(-2, -1)
        inner_re = torch.matmul(torch.cat([q_re, q_im], dim=-1), k_pack_t)
        inner_im = torch.matmul(torch.cat([q_im, -q_re], dim=-1), k_pack_t)
        mag_sq = (inner_re.square() + inner_im.square()).clamp_min(0.0)
        if abs(self.boson_power - 1.0) > 1e-5:
            born_scores = torch.pow(mag_sq.float(), self.boson_power).to(ket_re.dtype)
        else:
            born_scores = mag_sq

        purity_rel = (mps_purity - (1.0 / float(d)) + DEFAULT_PURITY_FLOOR_OFFSET).clamp(
            min=DEFAULT_PURITY_MIN_CLAMP, max=1.0
        )
        doc_mask = self._resolve_doc_mask(aux_pack)
        eff_mask = head_masks.unsqueeze(0)
        if doc_mask is not None:
            eff_mask = eff_mask * doc_mask.to(eff_mask.dtype).unsqueeze(1)

        masked_scores = born_scores * purity_rel.unsqueeze(1).unsqueeze(2) * eff_mask
        attn_probs = masked_scores / (masked_scores.sum(dim=-1, keepdim=True) + DEFAULT_TRACE_EPS)

        if self.config.use_two_hop_triangle and seq_len > 1:
            local_op = aux_pack["local_op"]
            if doc_mask is not None:
                masked_loc = local_op * doc_mask.to(local_op.dtype).unsqueeze(1)
                masked_loc = masked_loc / masked_loc.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                two_hop = torch.matmul(attn_probs, masked_loc)
            else:
                two_hop = torch.matmul(attn_probs, local_op)
            attn_probs = (
                1.0 - DEFAULT_TWO_HOP_MIX_ALPHA
            ) * attn_probs + DEFAULT_TWO_HOP_MIX_ALPHA * two_hop

        seq_head_both = (
            torch.matmul(attn_probs, torch.cat([v_re, v_im], dim=-1))
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        seq_head_re, seq_head_im = normalize_complex_ket(
            seq_head_both[..., :d], seq_head_both[..., d:]
        )
        agg_ket_re, agg_ket_im = normalize_complex_ket(
            seq_head_re.mean(dim=-2), seq_head_im.mean(dim=-2)
        )
        return seq_head_re * 0.5, seq_head_im * 0.5, agg_ket_re, agg_ket_im, attn_probs

    def forward(
        self,
        ket_re: Tensor,
        ket_im: Tensor,
        uni_diag: Tensor,
        trans_ema_re: Tensor,
        trans_ema_im: Tensor,
        hash_ket_re: Tensor,
        hash_ket_im: Tensor,
        aux_pack: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Run sharpened $p$-boson + `<BOS>`-masked 4-Compartment Baton layer (`<= 8` args)."""
        head_masks = aux_pack["head_masks"]
        causal_decay = aux_pack["causal_decay"]
        strict_past_mask = aux_pack["strict_past_mask"]
        id_ket_re = aux_pack["id_ket_re"]

        mps_beams_re, mps_beams_im, mps_purity, mps_ket_re, mps_ket_im = self.mps_baton(
            ket_re, ket_im, uni_diag
        )
        ctx_ket_re, ctx_ket_im, ema_ket_re, ema_ket_im, in_ctx_copy_weights = (
            self._compute_fused_transition_beams(
                ket_re,
                ket_im,
                trans_ema_re,
                trans_ema_im,
                hash_ket_re,
                hash_ket_im,
                causal_decay,
                strict_past_mask,
            )
        )
        w_mps = math.sqrt(DEFAULT_MPS_NGRAM_BLEND)
        w_tr = math.sqrt(0.5 * (1.0 - DEFAULT_MPS_NGRAM_BLEND))
        ngram_beams_re = torch.cat(
            [w_mps * mps_beams_re[:, :, :2, :], w_tr * ctx_ket_re.unsqueeze(2), w_tr * ema_ket_re.unsqueeze(2)],
            dim=2,
        )
        ngram_beams_im = torch.cat(
            [w_mps * mps_beams_im[:, :, :2, :], w_tr * ctx_ket_im.unsqueeze(2), w_tr * ema_ket_im.unsqueeze(2)],
            dim=2,
        )
        inv_ng = torch.rsqrt(
            (ngram_beams_re.square() + ngram_beams_im.square()).sum(dim=(-2, -1), keepdim=True).clamp_min(DEFAULT_TRACE_EPS)
        )
        ngram_beams_re, ngram_beams_im = ngram_beams_re * inv_ng, ngram_beams_im * inv_ng

        seq_beams_re, seq_beams_im, seq_ket_re, seq_ket_im, seq_attn_probs = (
            self._compute_fused_exterior_sequence_beams(
                ket_re, ket_im, mps_beams_re, mps_beams_im, mps_purity, aux_pack
            )
        )
        deep_beams_re, deep_beams_im, _ = factorized_rank_beams_schur_intersect(
            seq_beams_re, seq_beams_im, ngram_beams_re, ngram_beams_im
        )

        q_id = F.normalize(torch.matmul(ket_re + id_ket_re, self.id_query_re.to(ket_re.dtype)), dim=-1)
        k_id = F.normalize(torch.matmul(ket_re + id_ket_re, self.id_key_re.to(ket_re.dtype)), dim=-1)
        raw_id_sq = torch.matmul(q_id, k_id.transpose(-2, -1)).square()
        if abs(self.boson_power - 1.0) > 1e-5:
            raw_id_sq = torch.pow(raw_id_sq.float(), self.boson_power).to(ket_re.dtype)
        id_mask = head_masks[-1].unsqueeze(0)
        doc_mask = self._resolve_doc_mask(aux_pack)
        if doc_mask is not None:
            id_mask = id_mask * doc_mask.to(id_mask.dtype)
        id_scores = raw_id_sq * id_mask
        id_attn_probs = id_scores / (id_scores.sum(dim=-1, keepdim=True) + DEFAULT_TRACE_EPS)
        id_routed_re = F.normalize(torch.matmul(id_attn_probs, id_ket_re), dim=-1)
        id_routed_im = torch.zeros_like(id_routed_re)

        gate_kets_re, gate_kets_im = normalize_complex_ket(
            self.povm_gates_re.to(ket_re.dtype), self.povm_gates_im.to(ket_re.dtype)
        )
        gates = evaluate_rank_beams_born_gates(
            ngram_beams_re, ngram_beams_im, gate_kets_re, gate_kets_im, self.povm_gate_bias
        )
        comb_ket_re = ket_re + mps_ket_re + seq_ket_re + 0.5 * (ctx_ket_re + ema_ket_re)
        comb_ket_im = ket_im + mps_ket_im + seq_ket_im + 0.5 * (ctx_ket_im + ema_ket_im)
        act_deep_re, act_deep_im = apply_rank_beams_to_probe(
            deep_beams_re, deep_beams_im, comb_ket_re, comb_ket_im
        )
        next_ket_re, next_ket_im = normalize_complex_ket(
            ket_re + act_deep_re + 0.25 * comb_ket_re, ket_im + act_deep_im + 0.25 * comb_ket_im
        )
        return {
            "deep_beams_re": deep_beams_re,
            "deep_beams_im": deep_beams_im,
            "ngram_beams_re": ngram_beams_re,
            "ngram_beams_im": ngram_beams_im,
            "id_routed_re": id_routed_re,
            "id_routed_im": id_routed_im,
            "gates": gates,
            "next_ket_re": next_ket_re,
            "next_ket_im": next_ket_im,
            "in_ctx_copy_weights": in_ctx_copy_weights,
            "id_attn_probs": id_attn_probs,
            "seq_attn_probs": seq_attn_probs,
            "mps_purity": mps_purity,
        }


class SharpenedPBosonDocMaskedBatonLM(FusedBF16FourCompartmentBatonLM):
    """2L/1L Baton LM using $p$-boson Born exponent and `<BOS>` document-boundary masking."""

    def __init__(
        self,
        config: FusedBF16FourCompartmentBatonConfig,
        boson_power: float = 3.0,
        mask_bos_documents: bool = True,
    ) -> None:
        super().__init__(config)
        self.boson_power = float(boson_power)
        self.mask_bos_documents = bool(mask_bos_documents)
        self.layers = nn.ModuleList(
            [
                SharpenedPBosonDocMaskedBatonLayer(
                    config, boson_power=boson_power, mask_bos_documents=mask_bos_documents
                )
                for _ in range(config.num_layers)
            ]
        )
        self._compiled_layers_list = list(self.layers)

    def forward_with_diagnostics(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        ingest_online: bool = False,
        return_logits: bool = True,
    ) -> dict[str, Any]:
        """Inject per-batch `<BOS>` document equality mask onto layers (`<= 8` args)."""
        doc_ids = (idx == FINEWEB_BOS_TOKEN_ID).cumsum(dim=1)
        doc_same = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)
        for lyr in self.layers:
            if isinstance(lyr, SharpenedPBosonDocMaskedBatonLayer):
                lyr.current_doc_same_mask = doc_same if self.mask_bos_documents else None
        try:
            return super().forward_with_diagnostics(
                idx=idx,
                targets=targets,
                ingest_online=ingest_online,
                return_logits=return_logits,
            )
        finally:
            for lyr in self.layers:
                if isinstance(lyr, SharpenedPBosonDocMaskedBatonLayer):
                    lyr.current_doc_same_mask = None


# ==============================================================================
# Data Loading, Zipf Decile Partitioning, & Document Horizon Helpers
# ==============================================================================
def resolve_fineweb_shard_path(shard_filename: str) -> Path | None:
    """Locate a FineWeb `.bin` shard across canonical repository cache directories (`<= 8` args)."""
    candidates = (
        DEFAULT_FINEWEB_CACHE_DIR / shard_filename,
        REPO_ROOT / ".cache" / "fineweb10B" / shard_filename,
        REPO_ROOT / ".cache" / "fineweb_100m" / shard_filename,
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def load_fineweb_token_slice(
    shard_filename: str,
    num_tokens: int,
    offset_tokens: int = 0,
    seed_fallback: int = 42,
) -> Tensor:
    """Load `num_tokens` int64 tokens from a FineWeb `.bin` shard (with deterministic fallback)."""
    path = resolve_fineweb_shard_path(shard_filename)
    if path is not None:
        mm = np.memmap(
            path, dtype=np.uint16, mode="r", offset=FINEWEB_HEADER_INT32_COUNT * 4
        )
        end_idx = min(len(mm), int(offset_tokens) + int(num_tokens))
        arr = np.asarray(mm[int(offset_tokens) : end_idx], dtype=np.int64)
        if arr.size >= num_tokens:
            return torch.from_numpy(arr[:num_tokens].copy())
    gen = torch.Generator().manual_seed(int(seed_fallback))
    ranks = torch.arange(1, VALID_FINEWEB_VOCAB_SIZE + 1, dtype=torch.float64)
    probs = ranks ** -1.06
    probs = (probs / probs.sum()).to(torch.float32)
    sampled = torch.multinomial(
        probs, num_samples=int(num_tokens), replacement=True, generator=gen
    )
    if num_tokens >= 128:
        sampled[::97] = FINEWEB_BOS_TOKEN_ID
    return sampled.to(torch.long)


def build_empirical_zipf_decile_map(device: torch.device) -> Tensor:
    """Assign each token $v \\in \\{0..50256\\}$ to one of 10 empirical frequency deciles (`D0` = most frequent)."""
    train_slice = load_fineweb_token_slice(
        "fineweb_train_000001.bin", num_tokens=UNIGRAM_CALIBRATION_SCAN_TOKENS
    )
    counts = torch.bincount(
        train_slice.clamp(0, VALID_FINEWEB_VOCAB_SIZE - 1),
        minlength=VALID_FINEWEB_VOCAB_SIZE,
    ).to(torch.float64)
    tie_breaker = torch.linspace(0.5, 0.0, VALID_FINEWEB_VOCAB_SIZE, dtype=torch.float64)
    order = torch.argsort(counts + tie_breaker, descending=True)
    decile_map = torch.zeros(VALID_FINEWEB_VOCAB_SIZE, dtype=torch.long)
    bucket_size = math.ceil(VALID_FINEWEB_VOCAB_SIZE / float(NUM_ZIPF_DECILES))
    for d_idx in range(NUM_ZIPF_DECILES):
        start = d_idx * bucket_size
        end = min(VALID_FINEWEB_VOCAB_SIZE, (d_idx + 1) * bucket_size)
        if start < end:
            decile_map[order[start:end]] = d_idx
    return decile_map.to(device=device)


def compute_document_horizon_and_copy_masks(
    x_seq: Tensor, y_seq: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute `(pos_in_doc, in_doc_copy_mask, self_echo_mask)` of shape `(B, T)` (`<= 8` args)."""
    bsz, seq_len = x_seq.shape
    is_bos = x_seq == FINEWEB_BOS_TOKEN_ID
    doc_ids = is_bos.to(torch.long).cumsum(dim=1)

    t_idx = torch.arange(seq_len, device=x_seq.device).unsqueeze(0).expand(bsz, seq_len)
    bos_positions = torch.where(is_bos, t_idx, torch.zeros_like(t_idx))
    last_bos_pos = torch.cummax(bos_positions, dim=1).values
    has_seen_bos = doc_ids > 0
    pos_in_doc = torch.where(has_seen_bos, t_idx - last_bos_pos, t_idx)

    causal_same_doc = (t_idx.unsqueeze(2) >= t_idx.unsqueeze(1)) & (
        doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)
    )
    token_match = y_seq.unsqueeze(2) == x_seq.unsqueeze(1)
    in_doc_copy_mask = (token_match & causal_same_doc).any(dim=2)
    self_echo_mask = y_seq == x_seq
    return pos_in_doc, in_doc_copy_mask, self_echo_mask


# ==============================================================================
# Polymorphic Architecture Adapters (Rule 3 Strategy Dispatch)
# ==============================================================================
class DiagnosticModelAdapter:
    """Base polymorphic adapter exposing uniform training & log-probability evaluation across waves."""

    def __init__(
        self,
        key: str,
        label: str,
        model: nn.Module,
        device: torch.device,
        embed_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        self.key = key
        self.label = label
        self.model = model
        self.device = device
        self.embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)

    def parameter_breakdown(self) -> dict[str, int]:
        """Return total, embedding/vocab, and non-embedding trunk parameter counts (`<= 8` args)."""
        total = 0
        vocab_p = 0
        vocab_keys = ("wte", "lm_head", "token_ket", "value_embeds", "w_det", "unigram_bias", "radial_scale")
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            n = int(param.numel())
            total += n
            if any(k in name for k in vocab_keys):
                vocab_p += n
        if total == 0:
            for name, param in self.model.named_parameters():
                n = int(param.numel())
                total += n
                if any(k in name for k in vocab_keys):
                    vocab_p += n
        return {
            "total_params": total,
            "vocab_params": vocab_p,
            "trunk_params": max(0, total - vocab_p),
        }

    def train_step(self, x_batch: Tensor, y_batch: Tensor, step: int, total_steps: int) -> float:
        """Execute one forward+backward+optimizer step and return scalar training loss (`<= 8` args)."""
        raise NotImplementedError

    @torch.no_grad()
    def predict_log_probs_and_span(self, x_batch: Tensor) -> tuple[Tensor, Tensor]:
        """Return `(log_probs, logit_span)` of shapes `(B, T, 50257)` and `(B, T)` in `float32`."""
        raise NotImplementedError


class PretrainedClassicalTeacherAdapter(DiagnosticModelAdapter):
    """Frozen `CompiledLeaderboardClassicalGPT` teacher checkpoint adapter (`50M / 300M / 550M`)."""

    def __init__(self, key: str, label: str, ckpt_path: Path, device: torch.device) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw_cfg = dict(ckpt["config"])
        for k_tup in ("skip_attn_layers", "long_window_layers"):
            if k_tup in raw_cfg:
                raw_cfg[k_tup] = tuple(raw_cfg[k_tup])
        cfg = CompiledLeaderboardClassicalConfig(**raw_cfg)
        teacher = CompiledLeaderboardClassicalGPT(cfg).to(device)
        teacher.load_state_dict(ckpt["model_state_dict"], strict=True)
        ws_blocks = int(ckpt.get("current_ws_blocks", 8))
        teacher.set_sliding_window_blocks(
            ws_blocks=max(ws_blocks, 8), ws_final_blocks=16, apply_yarn=True
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        super().__init__(
            key=key,
            label=label,
            model=teacher,
            device=device,
            embed_dim=cfg.embed_dim,
            num_layers=cfg.num_layers,
        )

    @torch.no_grad()
    def predict_log_probs_and_span(self, x_batch: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate frozen `CompiledLeaderboardClassicalGPT` logits truncated to `50,257` valid tokens."""
        assert isinstance(self.model, CompiledLeaderboardClassicalGPT)
        self.model.eval()
        amp_on = x_batch.is_cuda
        with torch.amp.autocast(
            device_type=x_batch.device.type, dtype=torch.bfloat16, enabled=amp_on
        ):
            feats = self.model.forward_features(x_batch)
            logits = self.model.compute_logits(feats)[..., :VALID_FINEWEB_VOCAB_SIZE].float()
        span = logits.amax(dim=-1) - logits.median(dim=-1).values
        return F.log_softmax(logits, dim=-1), span


class TrainableClassicalGPTAdapter(DiagnosticModelAdapter):
    """Trainable `CompiledLeaderboardClassicalGPT` (`2L, d=512` shallow reference)."""

    def __init__(
        self,
        key: str,
        label: str,
        cfg: CompiledLeaderboardClassicalConfig,
        device: torch.device,
    ) -> None:
        model = CompiledLeaderboardClassicalGPT(cfg).to(device)
        muon_opt, adam_opt = build_compiled_leaderboard_optimizers(
            model, CompiledLeaderboardOptimizerConfig(muon_warmup_steps=50)
        )
        super().__init__(
            key=key,
            label=label,
            model=model,
            device=device,
            embed_dim=cfg.embed_dim,
            num_layers=cfg.num_layers,
        )
        self.muon_opt = muon_opt
        self.adam_opt = adam_opt
        self.base_muon_lrs = [float(pg["lr"]) for pg in muon_opt.param_groups]
        self.base_adam_lrs = [float(pg["lr"]) for pg in adam_opt.param_groups]

    def train_step(self, x_batch: Tensor, y_batch: Tensor, step: int, total_steps: int) -> float:
        """Run one `StiefelMuon + Adam` training step with warmup + cosine cooldown (`<= 8` args)."""
        assert isinstance(self.model, CompiledLeaderboardClassicalGPT)
        self.model.train()
        warmup = max(10, int(0.08 * total_steps))
        if step <= warmup:
            lr_mult = float(step) / float(warmup)
        else:
            prog = float(step - warmup) / float(max(1, total_steps - warmup))
            lr_mult = 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * prog))

        for pg, base_lr in zip(self.muon_opt.param_groups, self.base_muon_lrs):
            pg["lr"] = base_lr * lr_mult
        for pg, base_lr in zip(self.adam_opt.param_groups, self.base_adam_lrs):
            pg["lr"] = base_lr * lr_mult

        self.muon_opt.zero_grad(set_to_none=True)
        self.adam_opt.zero_grad(set_to_none=True)
        amp_on = x_batch.is_cuda
        with torch.amp.autocast(
            device_type=x_batch.device.type, dtype=torch.bfloat16, enabled=amp_on
        ):
            _, loss = self.model(x_batch, targets=y_batch, return_logits=False)
        assert loss is not None
        loss.backward()
        self.muon_opt.step()
        self.adam_opt.step()
        return float(loss.detach().item())

    @torch.no_grad()
    def predict_log_probs_and_span(self, x_batch: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate `CompiledLeaderboardClassicalGPT` log-probs and logit span (`<= 8` args)."""
        assert isinstance(self.model, CompiledLeaderboardClassicalGPT)
        was_train = self.model.training
        self.model.eval()
        amp_on = x_batch.is_cuda
        with torch.amp.autocast(
            device_type=x_batch.device.type, dtype=torch.bfloat16, enabled=amp_on
        ):
            feats = self.model.forward_features(x_batch)
            logits = self.model.compute_logits(feats)[..., :VALID_FINEWEB_VOCAB_SIZE].float()
        if was_train:
            self.model.train()
        span = logits.amax(dim=-1) - logits.median(dim=-1).values
        return F.log_softmax(logits, dim=-1), span


class Baton4CompAdapter(DiagnosticModelAdapter):
    """Adapter for `FusedBF16FourCompartmentBatonLM`, `ExactMixedStateFourCompartmentBatonLM`, & `SharpenedPBosonDocMaskedBatonLM`."""

    def __init__(
        self,
        key: str,
        label: str,
        model: FusedBF16FourCompartmentBatonLM,
        device: torch.device,
        base_lr: float = DEFAULT_BATON_BASE_LR,
    ) -> None:
        super().__init__(
            key=key,
            label=label,
            model=model,
            device=device,
            embed_dim=2 * model.config.hilbert_dim,
            num_layers=model.config.num_layers,
        )
        self.base_lr = float(base_lr)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.base_lr, betas=(0.9, 0.95), weight_decay=0.01
        )

    def train_step(self, x_batch: Tensor, y_batch: Tensor, step: int, total_steps: int) -> float:
        """Run one `AdamW` + online bath ingestion step on a 4-Compartment Baton model (`<= 8` args)."""
        assert isinstance(self.model, FusedBF16FourCompartmentBatonLM)
        self.model.train()
        warmup = max(8, int(0.06 * total_steps))
        if step <= warmup:
            lr_mult = float(step) / float(warmup)
        else:
            prog = float(step - warmup) / float(max(1, total_steps - warmup))
            lr_mult = DEFAULT_BATON_MIN_LR_RATIO + (1.0 - DEFAULT_BATON_MIN_LR_RATIO) * 0.5 * (
                1.0 + math.cos(math.pi * prog)
            )
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.base_lr * lr_mult

        self.optimizer.zero_grad(set_to_none=True)
        out = self.model.forward_with_diagnostics(
            idx=x_batch, targets=y_batch, ingest_online=True, return_logits=False
        )
        loss = out["loss"]
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.detach().item())

    @torch.no_grad()
    def predict_log_probs_and_span(self, x_batch: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate 4-Compartment Baton log-probabilities and logit span (`<= 8` args)."""
        assert isinstance(self.model, FusedBF16FourCompartmentBatonLM)
        was_train = self.model.training
        self.model.eval()
        out = self.model.forward_with_diagnostics(
            idx=x_batch, targets=None, ingest_online=False, return_logits=True
        )
        logits = out["logits"]
        assert logits is not None
        logits_f = logits[..., :VALID_FINEWEB_VOCAB_SIZE].float()
        if was_train:
            self.model.train()
        span = logits_f.amax(dim=-1) - logits_f.median(dim=-1).values
        return F.log_softmax(logits_f, dim=-1), span


class ZeroPPMICholeskyAdapter(DiagnosticModelAdapter):
    """Adapter for `Wave 61` `ZeroPPMIBytePairCholeskyGPT` (`4L, d=512` Quantum Vacuum Cholesky)."""

    def __init__(self, key: str, label: str, device: torch.device, seed: int = 1337) -> None:
        spec = ZeroPPMIBytePairCholeskySpec(
            variant_mode=ZeroPPMIBytePairVariantMode.Z2_ZERO_PPMI_FACTORED_BYTE_PAIR_SCHUR,
            num_cholesky_modes=8,
            enable_online_ema=True,
            enable_byte_pair_schur=True,
            use_exact_chain_rule_readout=False,
            seed=seed,
        )
        model = ZeroPPMIBytePairCholeskyGPT(spec).to(device)
        initialize_zero_ppmi_byte_pair_model_(model)
        optimizers = build_zero_ppmi_byte_pair_optimizers(model)
        super().__init__(
            key=key,
            label=label,
            model=model,
            device=device,
            embed_dim=512,
            num_layers=4,
        )
        self.optimizers = optimizers

    def train_step(self, x_batch: Tensor, y_batch: Tensor, step: int, total_steps: int) -> float:
        """Run one Riemannian `StiefelMuon` + `BatchSubspaceTikhonovGramMuon` + `AdamW` step (`<= 8` args)."""
        assert isinstance(self.model, ZeroPPMIBytePairCholeskyGPT)
        self.model.train()
        warmup = max(8, int(0.08 * total_steps))
        if step <= warmup:
            lr_mult = 0.20 + 0.80 * (float(step) / float(warmup))
        else:
            prog = float(step - warmup) / float(max(1, total_steps - warmup))
            lr_mult = 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * prog))

        q_trunk_opt, wte_opt, head_opt, adam_opt = self.optimizers
        q_trunk_opt.zero_grad()
        wte_opt.zero_grad()
        head_opt.zero_grad()
        adam_opt.zero_grad(set_to_none=True)

        amp_on = x_batch.is_cuda
        with torch.amp.autocast(
            device_type=x_batch.device.type, dtype=torch.bfloat16, enabled=amp_on
        ):
            loss = self.model.forward_loss(x_batch, y_batch)
        loss.backward()
        _apply_global_optimizer_step(
            self.model,
            self.optimizers,
            x_batch,
            y_batch,
            lr_mult=lr_mult,
            mom_beta=DEFAULT_CHOLESKY_BASE_MOMENTUM,
        )
        return float(loss.detach().item())

    @torch.no_grad()
    def predict_log_probs_and_span(self, x_batch: Tensor) -> tuple[Tensor, Tensor]:
        """Compute exact multi-branch Cholesky log-mixture probabilities `(B, T, 50257)` (`<= 8` args)."""
        assert isinstance(self.model, ZeroPPMIBytePairCholeskyGPT)
        was_train = self.model.training
        self.model.eval()
        bsz, seq_len = x_batch.shape
        h_base, h_sem_b, h_id_b, ctrl, w_aug = self.model.build_multibranch_readout_tensors(x_batch)
        num_r = h_id_b.size(-2)
        sem_dim, id_dim = h_sem_b.size(-1), h_id_b.size(-1)
        car_dim = h_base.size(-1) - sem_dim
        indep_sem = h_sem_b.ndim == 4

        w_valid = w_aug[:VALID_FINEWEB_VOCAB_SIZE]
        w_sem = w_valid[:, :sem_dim]
        w_id = w_valid[:, sem_dim : sem_dim + id_dim]
        w_car = w_valid[:, sem_dim + id_dim : sem_dim + id_dim + car_dim]
        w_base_t = torch.cat([w_sem, w_car], dim=-1).t().contiguous()
        w_sem_t, w_id_t = w_sem.t().contiguous(), w_id.t().contiguous()

        sem1_gates, id_scales, gate_logits = ctrl[:, :, :, 0], ctrl[:, :, :, 1], ctrl[:, :, :, 2]
        has_sem1 = bool(sem1_gates.abs().max().item() > 1e-8)
        cap = float(DEFAULT_SOFTCAP_LOGIT)
        inv_cap = 1.0 / cap

        log_p_chunks: list[Tensor] = []
        span_chunks: list[Tensor] = []
        t_step = 8
        for t0 in range(0, seq_len, t_step):
            t1 = min(seq_len, t0 + t_step)
            c_len = bsz * (t1 - t0)
            hb_c = h_base[:, t0:t1].reshape(c_len, sem_dim + car_dim)
            hid_c = h_id_b[:, t0:t1].reshape(c_len * num_r, id_dim)
            b_c = sem1_gates[:, t0:t1].reshape(c_len, num_r).float()
            t_c = id_scales[:, t0:t1].reshape(c_len, num_r).float()
            a_c = gate_logits[:, t0:t1].reshape(c_len, num_r).float()

            z_base = torch.matmul(hb_c, w_base_t).float()
            raw_r = torch.matmul(hid_c, w_id_t).float().view(c_len, num_r, VALID_FINEWEB_VOCAB_SIZE)
            raw_r.mul_(t_c.unsqueeze(-1)).add_(z_base.unsqueeze(1))
            if has_sem1:
                if indep_sem:
                    hs_c = h_sem_b[:, t0:t1].reshape(c_len * num_r, sem_dim)
                    z_sem1 = (
                        torch.matmul(hs_c, w_sem_t)
                        .float()
                        .view(c_len, num_r, VALID_FINEWEB_VOCAB_SIZE)
                    )
                    raw_r.addcmul_(b_c.unsqueeze(-1), z_sem1)
                else:
                    hs_c = h_sem_b[:, t0:t1].reshape(c_len, sem_dim)
                    z_sem1 = torch.matmul(hs_c, w_sem_t).float()
                    raw_r.addcmul_(b_c.unsqueeze(-1), z_sem1.unsqueeze(1))
            raw_r.mul_(inv_cap).tanh_().mul_(cap)

            span_c = raw_r[:, 0, :].amax(dim=-1) - raw_r[:, 0, :].median(dim=-1).values
            log_pi = F.log_softmax(a_c, dim=-1)
            log_p_r = F.log_softmax(raw_r, dim=-1)
            log_p_mix = torch.logsumexp(log_pi.unsqueeze(-1) + log_p_r, dim=1)

            log_p_chunks.append(log_p_mix.view(bsz, t1 - t0, VALID_FINEWEB_VOCAB_SIZE))
            span_chunks.append(span_c.view(bsz, t1 - t0))

        if was_train:
            self.model.train()
        return torch.cat(log_p_chunks, dim=1), torch.cat(span_chunks, dim=1)


def build_known_architecture_adapter(
    kind: KnownArchitectureKind,
    device: torch.device,
    seed: int = 1337,
) -> DiagnosticModelAdapter:
    """Instantiate any known architecture or pretrained teacher checkpoint by `KnownArchitectureKind`."""
    torch.manual_seed(int(seed))
    if kind == KnownArchitectureKind.TEACHER_50M:
        return PretrainedClassicalTeacherAdapter(
            kind.value, "Classical Teacher 12L d=768 (50M tok)", TEACHER_50M_CKPT_PATH, device
        )
    if kind == KnownArchitectureKind.TEACHER_300M:
        return PretrainedClassicalTeacherAdapter(
            kind.value, "Classical Teacher 12L d=768 (300M tok)", TEACHER_300M_CKPT_PATH, device
        )
    if kind == KnownArchitectureKind.TEACHER_550M:
        return PretrainedClassicalTeacherAdapter(
            kind.value, "Classical Teacher 12L d=768 (550M tok)", TEACHER_550M_CKPT_PATH, device
        )
    if kind == KnownArchitectureKind.W74_BATON_1L_D64:
        cfg = FusedBF16FourCompartmentBatonConfig(hilbert_dim=64, num_layers=1)
        return Baton4CompAdapter(
            kind.value,
            "W74 4-Comp Baton 1L d=64 (128D)",
            FusedBF16FourCompartmentBatonLM(cfg).to(device),
            device,
        )
    if kind == KnownArchitectureKind.W74_BATON_2L_D64:
        cfg = FusedBF16FourCompartmentBatonConfig(hilbert_dim=64, num_layers=2)
        return Baton4CompAdapter(
            kind.value,
            "W74 4-Comp Baton 2L d=64 (128D)",
            FusedBF16FourCompartmentBatonLM(cfg).to(device),
            device,
        )
    if kind == KnownArchitectureKind.W74_BATON_2L_D256:
        cfg = FusedBF16FourCompartmentBatonConfig(hilbert_dim=256, num_layers=2)
        return Baton4CompAdapter(
            kind.value,
            "W75 4-Comp Baton 2L d=256 (512D)",
            FusedBF16FourCompartmentBatonLM(cfg).to(device),
            device,
            base_lr=2.2e-3,
        )
    if kind == KnownArchitectureKind.W76_SHARP_BATON_2L_D64_P3:
        cfg = FusedBF16FourCompartmentBatonConfig(hilbert_dim=64, num_layers=2)
        return Baton4CompAdapter(
            kind.value,
            "W76 Sharpened Baton 2L d=64 (p=3 + DocMask)",
            SharpenedPBosonDocMaskedBatonLM(cfg, boson_power=3.0, mask_bos_documents=True).to(
                device
            ),
            device,
        )
    if kind == KnownArchitectureKind.W75_EXACT_MIXED_2L_D128:
        cfg = FusedBF16FourCompartmentBatonConfig(hilbert_dim=128, num_layers=2)
        return Baton4CompAdapter(
            kind.value,
            "W75 Exact Mixed-State Thm16 2L d=128 (256D)",
            ExactMixedStateFourCompartmentBatonLM(cfg).to(device),
            device,
            base_lr=2.5e-3,
        )
    if kind == KnownArchitectureKind.W45_CLASSICAL_2L_D512:
        c_cfg = CompiledLeaderboardClassicalConfig(
            embed_dim=512,
            num_heads=4,
            num_layers=2,
            hidden_dim=1536,
            max_seq_len=512,
            num_value_embeddings=1,
            skip_attn_layers=(),
            long_window_layers=(0, 1),
            bfloat16_weights=device.type == "cuda",
        )
        return TrainableClassicalGPTAdapter(
            kind.value, "W45 Classical Leaderboard 2L d=512", c_cfg, device
        )
    if kind == KnownArchitectureKind.W61_ZERO_PPMI_CHOLESKY_4L_D512:
        return ZeroPPMICholeskyAdapter(
            kind.value, "W61 Zero-PPMI Vacuum Cholesky 4L d=512", device, seed=seed
        )
    raise ValueError(f"Unsupported architecture kind: {kind}")


# ==============================================================================
# Engine 1: Stratified Nat-Gap Attribution & Calibration X-Ray
# ==============================================================================
def compute_expected_calibration_error(max_probs: Tensor, correct: Tensor) -> float:
    """Compute `NUM_ECE_BINS`-bin Expected Calibration Error (`ECE`) over flattened tokens (`<= 8` args)."""
    total = float(max(1, max_probs.numel()))
    ece = 0.0
    edges = torch.linspace(0.0, 1.0, NUM_ECE_BINS + 1, device=max_probs.device)
    for b_idx in range(NUM_ECE_BINS):
        lo, hi = edges[b_idx], edges[b_idx + 1]
        if b_idx == NUM_ECE_BINS - 1:
            mask = (max_probs >= lo) & (max_probs <= hi)
        else:
            mask = (max_probs >= lo) & (max_probs < hi)
        cnt = int(mask.sum().item())
        if cnt > 0:
            conf = float(max_probs[mask].mean().item())
            acc = float(correct[mask].mean().item())
            ece += (float(cnt) / total) * abs(conf - acc)
    return ece


def _summarize_bucket(
    mask: Tensor, nll: Tensor, correct: Tensor, ref_nll: Tensor | None
) -> dict[str, float]:
    """Summarize token count, fraction, CE, Top-1 accuracy, and differential $\\Delta\\text{CE}$ on `mask`."""
    cnt = int(mask.sum().item())
    total = max(1, int(nll.numel()))
    if cnt == 0:
        return {"count": 0.0, "frac": 0.0, "ce": 0.0, "top1_acc": 0.0, "delta_ce_vs_ref": 0.0}
    ce_val = float(nll[mask].mean().item())
    acc_val = float(correct[mask].mean().item())
    delta = 0.0
    if ref_nll is not None:
        delta = float((nll[mask] - ref_nll[mask]).mean().item())
    return {
        "count": float(cnt),
        "frac": round(float(cnt) / float(total), 5),
        "ce": round(ce_val, 5),
        "top1_acc": round(acc_val, 5),
        "delta_ce_vs_ref": round(delta, 5),
    }


@torch.no_grad()
def build_teacher_reference_profile(
    anchor_adapter: DiagnosticModelAdapter,
    x_eval: Tensor,
    y_eval: Tensor,
    micro_batch: int = 8,
) -> TeacherReferenceProfile:
    """Precompute token-level NLL and entropy on `(x_eval, y_eval)` for a reference anchor (`<= 8` args)."""
    nll_list: list[Tensor] = []
    ent_list: list[Tensor] = []
    n_seq = x_eval.size(0)
    for start in range(0, n_seq, micro_batch):
        end = min(n_seq, start + micro_batch)
        bx = x_eval[start:end].to(anchor_adapter.device)
        by = y_eval[start:end].to(anchor_adapter.device)
        log_p, _ = anchor_adapter.predict_log_probs_and_span(bx)
        tok_nll = -log_p.gather(dim=-1, index=by.unsqueeze(-1)).squeeze(-1)
        probs = torch.exp(log_p)
        tok_ent = -(probs * log_p.clamp_min(LOG_PROB_FLOOR)).sum(dim=-1)
        nll_list.append(tok_nll.cpu())
        ent_list.append(tok_ent.cpu())
    full_nll = torch.cat(nll_list, dim=0)
    full_ent = torch.cat(ent_list, dim=0)
    return TeacherReferenceProfile(
        anchor_name=anchor_adapter.key,
        anchor_val_ce=float(full_nll.mean().item()),
        token_nll=full_nll,
        token_entropy=full_ent,
    )


@torch.no_grad()
def compute_stratified_nat_gap_xray(
    adapter: DiagnosticModelAdapter,
    x_eval: Tensor,
    y_eval: Tensor,
    decile_map: Tensor,
    ref_profile: TeacherReferenceProfile | None = None,
    micro_batch: int = 8,
) -> dict[str, Any]:
    """Run full Stratified Nat-Gap & Calibration X-Ray on `(x_eval, y_eval)` (`<= 8` args)."""
    n_seq = x_eval.size(0)
    nll_list: list[Tensor] = []
    top1_list: list[Tensor] = []
    top5_list: list[Tensor] = []
    ent_list: list[Tensor] = []
    maxp_list: list[Tensor] = []
    span_list: list[Tensor] = []

    for start in range(0, n_seq, micro_batch):
        end = min(n_seq, start + micro_batch)
        bx = x_eval[start:end].to(adapter.device)
        by = y_eval[start:end].to(adapter.device)
        log_p, span = adapter.predict_log_probs_and_span(bx)
        probs = torch.exp(log_p)
        tok_nll = -log_p.gather(dim=-1, index=by.unsqueeze(-1)).squeeze(-1)
        top5_idx = log_p.topk(5, dim=-1).indices
        top1_ok = (top5_idx[..., 0] == by).to(torch.float32)
        top5_ok = (top5_idx == by.unsqueeze(-1)).any(dim=-1).to(torch.float32)
        tok_ent = -(probs * log_p.clamp_min(LOG_PROB_FLOOR)).sum(dim=-1)
        tok_maxp = probs.amax(dim=-1)

        nll_list.append(tok_nll.cpu())
        top1_list.append(top1_ok.cpu())
        top5_list.append(top5_ok.cpu())
        ent_list.append(tok_ent.cpu())
        maxp_list.append(tok_maxp.cpu())
        span_list.append(span.cpu())

    nll = torch.cat(nll_list, dim=0).reshape(-1)
    correct = torch.cat(top1_list, dim=0).reshape(-1)
    top5 = torch.cat(top5_list, dim=0).reshape(-1)
    ent = torch.cat(ent_list, dim=0).reshape(-1)
    maxp = torch.cat(maxp_list, dim=0).reshape(-1)
    spans = torch.cat(span_list, dim=0).reshape(-1)

    x_cpu, y_cpu = x_eval.cpu(), y_eval.cpu()
    pos_doc, copy_mask_2d, echo_mask_2d = compute_document_horizon_and_copy_masks(x_cpu, y_cpu)
    pos_flat = pos_doc.reshape(-1)
    copy_flat = copy_mask_2d.reshape(-1)
    echo_flat = echo_mask_2d.reshape(-1)
    target_deciles = decile_map.cpu()[y_cpu.clamp(0, VALID_FINEWEB_VOCAB_SIZE - 1).reshape(-1)]

    ref_nll = ref_profile.token_nll.reshape(-1) if ref_profile is not None else None
    ref_ent = ref_profile.token_entropy.reshape(-1) if ref_profile is not None else None

    val_ce = float(nll.mean().item())
    ece = compute_expected_calibration_error(maxp, correct)

    copy_vs_recall = {
        "in_doc_copy": _summarize_bucket(copy_flat, nll, correct, ref_nll),
        "out_of_doc_recall": _summarize_bucket(~copy_flat, nll, correct, ref_nll),
        "immediate_self_echo": _summarize_bucket(echo_flat, nll, correct, ref_nll),
        "non_echo": _summarize_bucket(~echo_flat, nll, correct, ref_nll),
    }

    decile_rows: list[dict[str, float]] = []
    for d_idx in range(NUM_ZIPF_DECILES):
        row = _summarize_bucket(target_deciles == d_idx, nll, correct, ref_nll)
        row["decile"] = float(d_idx)
        decile_rows.append(row)
    zipf_tiers = {
        "head_d0_d1": _summarize_bucket(target_deciles <= 1, nll, correct, ref_nll),
        "body_d2_d5": _summarize_bucket(
            (target_deciles >= 2) & (target_deciles <= 5), nll, correct, ref_nll
        ),
        "tail_d6_d9": _summarize_bucket(target_deciles >= 6, nll, correct, ref_nll),
    }

    e0, e1, e2 = DOC_POS_BUCKET_EDGES
    doc_horizons = {
        "pos_0_3_cold_bos": _summarize_bucket(pos_flat < e0, nll, correct, ref_nll),
        "pos_4_31_short": _summarize_bucket(
            (pos_flat >= e0) & (pos_flat < e1), nll, correct, ref_nll
        ),
        "pos_32_127_medium": _summarize_bucket(
            (pos_flat >= e1) & (pos_flat < e2), nll, correct, ref_nll
        ),
        "pos_128_plus_long": _summarize_bucket(pos_flat >= e2, nll, correct, ref_nll),
    }

    surprisal_regimes: dict[str, Any] = {}
    if ref_ent is not None:
        surprisal_regimes = {
            "sharp_deterministic_lt_1p5": _summarize_bucket(
                ref_ent < TEACHER_ENTROPY_SHARP_MAX, nll, correct, ref_nll
            ),
            "medium_entropy_1p5_to_4p0": _summarize_bucket(
                (ref_ent >= TEACHER_ENTROPY_SHARP_MAX) & (ref_ent < TEACHER_ENTROPY_MEDIUM_MAX),
                nll,
                correct,
                ref_nll,
            ),
            "ambiguous_high_entropy_ge_4p0": _summarize_bucket(
                ref_ent >= TEACHER_ENTROPY_MEDIUM_MAX, nll, correct, ref_nll
            ),
        }

    p_info = adapter.parameter_breakdown()
    return {
        "key": adapter.key,
        "label": adapter.label,
        "total_params": p_info["total_params"],
        "trunk_params": p_info["trunk_params"],
        "vocab_params": p_info["vocab_params"],
        "eval_tokens": int(nll.numel()),
        "val_ce": round(val_ce, 5),
        "val_ppl": round(math.exp(min(20.0, val_ce)), 3),
        "delta_ce_vs_ref": (
            round(val_ce - ref_profile.anchor_val_ce, 5) if ref_profile is not None else 0.0
        ),
        "top1_acc": round(float(correct.mean().item()), 5),
        "top5_acc": round(float(top5.mean().item()), 5),
        "mean_pred_entropy_nats": round(float(ent.mean().item()), 4),
        "mean_max_prob": round(float(maxp.mean().item()), 4),
        "ece_10bin": round(ece, 5),
        "mean_logit_span_max_minus_median": round(float(spans.mean().item()), 4),
        "copy_vs_recall": copy_vs_recall,
        "zipf_tiers": zipf_tiers,
        "deciles": decile_rows,
        "doc_horizons": doc_horizons,
        "surprisal_regimes": surprisal_regimes,
    }
