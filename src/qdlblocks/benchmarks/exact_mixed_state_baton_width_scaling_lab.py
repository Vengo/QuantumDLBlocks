"""Wave 75 (`EXPERIMENT_PLAN_095.md`): Exact Mixed-State Attention Schur Contraction (`Theorem 16`),
Shadow-EMA vs. Hot Validation Telemetry, and Width-Scaling Speed Exploration (`d = 64, 128, 192, 256`).

Addresses three key questions from `Wave 74`:
1. **Exact Equivalence Audit (`Wave 73` vs `Wave 74` vs `Wave 75 Theorem 16`)**:
   - Proves and implements **Theorem 16 (`exact_mixed_state_attention_schur_action`)**, evaluating
     the exact `Wave 73` incoherent mixed-state attention Schur product
     `((1/H ∑_{h,s} A_{t,s}^{(h)} |v_{s,h}⟩⟨v_{s,h}|) ⊙ (∑_r |z_{t,r}⟩⟨z_{t,r}|)) |ψ_t⟩`
     and trace `Tr(ρ_t^{seq} ⊙ ρ_t^{ngram})` in `O(B H R T^2 d)` Tensor-Core GEMMs without ever
     allocating `(B, T, H, d, d)` density matrices (`< 1e-6` FP32 error vs `Wave 73`).
   - Implements `exact_token_id_mixed_state_action` for `Compartment 4` (`Packet 7`) mixed-state
     Token-ID routing `(∑_s A_{t,s}^{id} |id_s⟩⟨id_s|) |ψ_t⟩` without `(B, T, d, d)` tensors.
2. **Hot vs. Shadow-EMA Validation Telemetry (`ShadowEMAParameterTracker`)**:
   - Maintains an exponential moving average `θ_ema <- β_ema θ_ema + (1 - β_ema) θ_hot`
     (`β_ema = 0.995`) of all trainable parameters and measures `hot_val_ce`, `ema_val_ce`, and
     `thermal_gap_ce = hot_val_ce - ema_val_ce` at every evaluation checkpoint to diagnose whether
     the learning rate is running too hot.
3. **Width-Scaling Speed & FineWeb Benchmark (`d ∈ {64, 128, 192, 256}`)**:
   - Benchmarks throughput (`tok/s`), step latency (`ms`), parameter count, and peak VRAM (`MiB`)
     across `d ∈ {64, 128, 192, 256}` (`1L` and `2L`), and runs FineWeb training with live
     `Hot Val CE`, `EMA Val CE`, and `Thermal Gap` logging.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from qdlblocks.benchmarks.experiment_logger import (
    ExperimentIntentSpec,
    StandardizedExperimentLogger,
)
from qdlblocks.benchmarks.four_compartment_baton_lm_lab import (
    DEFAULT_BORN_READOUT_SCALE,
    DEFAULT_DOC_BATH_MIX,
    DEFAULT_IN_CONTEXT_COPY_SCALE,
    DEFAULT_LOGIT_SOFTCAP,
    DEFAULT_MPS_NGRAM_BLEND,
    DEFAULT_PURITY_FLOOR_OFFSET,
    DEFAULT_PURITY_MIN_CLAMP,
    DEFAULT_TOKEN_ID_COPY_SCALE,
    DEFAULT_TRACE_EPS,
    DEFAULT_UNIGRAM_PRIOR_WEIGHT,
    DEFAULT_VACUUM_EPS,
    DEFAULT_VAL_SHARD as FINEWEB_VAL_SHARD_PATH,
    DEFAULT_VOCAB_SIZE,
    NUM_SEQ_ATTN_PACKETS,
    REPO_ROOT,
    apply_complex_rope,
    complex_schur_product,
    encode_walsh_hadamard_token_id_kets,
    normalize_complex_ket,
    pure_ket_to_density,
    rank_beams_to_density,
    trace_normalize_density,
)
from qdlblocks.benchmarks.fused_bf16_four_compartment_baton_lab import (
    DEFAULT_SCHUR_BEAM_RANK,
    DEFAULT_TWO_HOP_MIX_ALPHA,
    FINEWEB_TRAIN_SHARDS_300M,
    FusedBF16FourCompartmentBatonConfig,
    FusedBF16FourCompartmentBatonLayer,
    FusedBF16FourCompartmentBatonLM,
    MultiShardFineWebStreamer,
    _get_softcap_cross_entropy_fn,
    apply_rank_beams_to_probe,
    diagonal_conjugated_bath_schur_action,
    evaluate_rank_beams_born_gates,
    factorized_rank_beams_schur_intersect,
    gram_matrix_mps_purity,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 — SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
DEFAULT_WIDTH_CHOICES: tuple[int, ...] = (64, 128, 192, 256)
DEFAULT_SHADOW_EMA_DECAY: float = 0.995
EXACT_EQUIVALENCE_FP32_TOL: float = 1e-5
DEFAULT_EVAL_BATCH_SIZE: int = 16
DEFAULT_EVAL_BATCHES: int = 12


# ==============================================================================
# Theorem 16 Primitives: Exact Mixed-State Attention Schur Contraction
# ==============================================================================
def exact_mixed_state_attention_schur_action(
    attn_probs: Tensor,
    v_re: Tensor,
    v_im: Tensor,
    ngram_beams_re: Tensor,
    ngram_beams_im: Tensor,
    probe_re: Tensor,
    probe_im: Tensor,
    eps: float = DEFAULT_TRACE_EPS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Evaluate exact `(ρ_t^{seq} ⊙ ρ_t^{ngram}) |ψ_t⟩` and `Tr(ρ_t^{seq} ⊙ ρ_t^{ngram})` via Theorem 16 (`<= 8` args).

    Here `ρ_t^{seq} = (1/H) ∑_{h=1}^H ∑_{s=1}^T A_{t,s}^{(h)} |v_{s,h}⟩⟨v_{s,h}|` is the exact
    incoherent mixed-state attention ensemble from `Wave 73` (`FourCompartmentBatonLayer`), and
    `ρ_t^{ngram} = ∑_{r=1}^R |z_{t,r}⟩⟨z_{t,r}|` is the Rank-`R` n-gram density operator.
    Computes the trace-normalized Schur action in `O(B H R T^2 d)` Tensor-Core GEMMs without ever
    materializing `(B, T, H, d, d)` matrices.
    """
    bsz, h_cnt, seq_len, d = v_re.shape
    r_cnt = ngram_beams_re.shape[2]
    inv_h = 1.0 / float(h_cnt)

    v_prob = v_re.square() + v_im.square()
    seq_diag = torch.matmul(attn_probs, v_prob).mean(dim=1)
    ng_diag = ngram_beams_re.square().sum(dim=2) + ngram_beams_im.square().sum(dim=2)
    raw_trace = (seq_diag * ng_diag).sum(dim=-1, keepdim=True) + eps
    tr_denom = raw_trace.clamp_min(eps)

    phi_re = (
        ngram_beams_re * probe_re.unsqueeze(2) + ngram_beams_im * probe_im.unsqueeze(2)
    ).permute(0, 2, 1, 3)
    phi_im = (
        ngram_beams_re * probe_im.unsqueeze(2) - ngram_beams_im * probe_re.unsqueeze(2)
    ).permute(0, 2, 1, 3)

    v_pack_t = torch.cat([v_re, v_im], dim=-1).transpose(-2, -1)
    v_both = torch.cat([v_re, v_im], dim=-1)

    u_re_list: list[Tensor] = []
    u_im_list: list[Tensor] = []
    for r_idx in range(r_cnt):
        pr_re = phi_re[:, r_idx : r_idx + 1, :, :]
        pr_im = phi_im[:, r_idx : r_idx + 1, :, :]
        ov_re = torch.matmul(torch.cat([pr_re, pr_im], dim=-1), v_pack_t)
        ov_im = torch.matmul(torch.cat([pr_im, -pr_re], dim=-1), v_pack_t)

        w_re = (attn_probs * inv_h) * ov_re
        w_im = (attn_probs * inv_h) * ov_im

        re_both = torch.matmul(w_re, v_both).sum(dim=1)
        im_both = torch.matmul(w_im, v_both).sum(dim=1)
        u_re_r = re_both[..., :d] - im_both[..., d:]
        u_im_r = re_both[..., d:] + im_both[..., :d]
        u_re_list.append(u_re_r)
        u_im_list.append(u_im_r)

    u_re = torch.stack(u_re_list, dim=2)
    u_im = torch.stack(u_im_list, dim=2)

    unnorm_re = (ngram_beams_re * u_re - ngram_beams_im * u_im).sum(dim=2) + (
        eps / float(d)
    ) * probe_re
    unnorm_im = (ngram_beams_re * u_im + ngram_beams_im * u_re).sum(dim=2) + (
        eps / float(d)
    ) * probe_im
    return unnorm_re / tr_denom, unnorm_im / tr_denom, raw_trace.squeeze(-1)


def exact_token_id_mixed_state_action(
    id_attn_probs: Tensor,
    id_ket_re: Tensor,
    probe_re: Tensor,
    probe_im: Tensor,
) -> tuple[Tensor, Tensor]:
    """Evaluate exact `(∑_s A_{t,s}^{id} |id_s⟩⟨id_s|) |ψ_t⟩` without `(B, T, d, d)` matrices (`<= 8` args)."""
    id_t = id_ket_re.transpose(-2, -1)
    ov_re = torch.matmul(probe_re, id_t)
    ov_im = torch.matmul(probe_im, id_t)
    act_re = torch.matmul(id_attn_probs * ov_re, id_ket_re)
    act_im = torch.matmul(id_attn_probs * ov_im, id_ket_re)
    return act_re, act_im


# ==============================================================================
# Wave 75 Exact Mixed-State 4-Compartment Baton Layer & Model (Theorem 16)
# ==============================================================================
class ExactMixedStateFourCompartmentBatonLayer(FusedBF16FourCompartmentBatonLayer):
    """4-Compartment Baton layer using Theorem 16 for exact `Wave 73` mixed-state Schur contraction."""

    def _compute_exact_exterior_sequence_mixed_state(
        self,
        ket_re: Tensor,
        ket_im: Tensor,
        mps_beams_re: Tensor,
        mps_beams_im: Tensor,
        mps_purity: Tensor,
        aux_pack: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute normalized value kets `(v_re, v_im)` and causal/2-hop attention weights (`<= 8` args)."""
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
        q_re, q_im = qkv_out[..., 0, :], qkv_out[..., 1, :]
        k_re, k_im = qkv_out[..., 2, :], qkv_out[..., 3, :]
        v_re, v_im = qkv_out[..., 4, :], qkv_out[..., 5, :]

        q_re, q_im = normalize_complex_ket(q_re, q_im)
        k_re, k_im = normalize_complex_ket(k_re, k_im)
        v_re, v_im = normalize_complex_ket(v_re, v_im)
        q_re, q_im = apply_complex_rope(q_re, q_im, cos_pos.unsqueeze(1), sin_pos.unsqueeze(1))
        k_re, k_im = apply_complex_rope(k_re, k_im, cos_pos.unsqueeze(1), sin_pos.unsqueeze(1))

        k_pack_t = torch.cat([k_re, k_im], dim=-1).transpose(-2, -1)
        inner_re = torch.matmul(torch.cat([q_re, q_im], dim=-1), k_pack_t)
        inner_im = torch.matmul(torch.cat([q_im, -q_re], dim=-1), k_pack_t)
        born_scores = inner_re.square() + inner_im.square()

        purity_rel = (mps_purity - (1.0 / float(d)) + DEFAULT_PURITY_FLOOR_OFFSET).clamp(
            min=DEFAULT_PURITY_MIN_CLAMP, max=1.0
        )
        born_scores = born_scores * purity_rel.unsqueeze(1).unsqueeze(2)
        masked_scores = born_scores * head_masks.unsqueeze(0)
        attn_probs = masked_scores / (masked_scores.sum(dim=-1, keepdim=True) + DEFAULT_TRACE_EPS)

        if self.config.use_two_hop_triangle and seq_len > 1:
            two_hop = torch.matmul(attn_probs, aux_pack["local_op"])
            attn_probs = (
                1.0 - DEFAULT_TWO_HOP_MIX_ALPHA
            ) * attn_probs + DEFAULT_TWO_HOP_MIX_ALPHA * two_hop

        seq_head_both = (
            torch.matmul(attn_probs, torch.cat([v_re, v_im], dim=-1))
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        seq_head_re, seq_head_im = seq_head_both[..., :d], seq_head_both[..., d:]
        agg_ket_re, agg_ket_im = normalize_complex_ket(
            seq_head_re.mean(dim=-2), seq_head_im.mean(dim=-2)
        )
        return v_re, v_im, agg_ket_re, agg_ket_im, attn_probs

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
        """Run one Exact Mixed-State 4-Compartment Baton layer via Theorem 16 (`<= 8` args)."""
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
            [
                w_mps * mps_beams_re[:, :, :2, :],
                w_tr * ctx_ket_re.unsqueeze(2),
                w_tr * ema_ket_re.unsqueeze(2),
            ],
            dim=2,
        )
        ngram_beams_im = torch.cat(
            [
                w_mps * mps_beams_im[:, :, :2, :],
                w_tr * ctx_ket_im.unsqueeze(2),
                w_tr * ema_ket_im.unsqueeze(2),
            ],
            dim=2,
        )
        ng_frob = (
            (ngram_beams_re.square() + ngram_beams_im.square())
            .sum(dim=(-2, -1), keepdim=True)
            .clamp_min(DEFAULT_TRACE_EPS)
        )
        inv_ng = torch.rsqrt(ng_frob)
        ngram_beams_re = ngram_beams_re * inv_ng
        ngram_beams_im = ngram_beams_im * inv_ng

        v_re, v_im, seq_ket_re, seq_ket_im, seq_attn_probs = (
            self._compute_exact_exterior_sequence_mixed_state(
                ket_re, ket_im, mps_beams_re, mps_beams_im, mps_purity, aux_pack
            )
        )

        q_id = F.normalize(torch.matmul(ket_re + id_ket_re, self.id_query_re), dim=-1)
        k_id = F.normalize(torch.matmul(ket_re + id_ket_re, self.id_key_re), dim=-1)
        id_scores = torch.matmul(q_id, k_id.transpose(-2, -1)).square() * head_masks[-1].unsqueeze(
            0
        )
        id_attn_probs = id_scores / (id_scores.sum(dim=-1, keepdim=True) + DEFAULT_TRACE_EPS)

        gate_kets_re, gate_kets_im = normalize_complex_ket(
            self.povm_gates_re.to(ket_re.dtype), self.povm_gates_im.to(ket_re.dtype)
        )
        gates = evaluate_rank_beams_born_gates(
            ngram_beams_re, ngram_beams_im, gate_kets_re, gate_kets_im, self.povm_gate_bias
        )

        comb_ket_re = ket_re + mps_ket_re + seq_ket_re + 0.5 * (ctx_ket_re + ema_ket_re)
        comb_ket_im = ket_im + mps_ket_im + seq_ket_im + 0.5 * (ctx_ket_im + ema_ket_im)
        act_deep_re, act_deep_im, schur_trace = exact_mixed_state_attention_schur_action(
            seq_attn_probs, v_re, v_im, ngram_beams_re, ngram_beams_im, comb_ket_re, comb_ket_im
        )
        next_ket_re, next_ket_im = normalize_complex_ket(
            ket_re + act_deep_re + 0.25 * comb_ket_re, ket_im + act_deep_im + 0.25 * comb_ket_im
        )

        return {
            "v_re": v_re,
            "v_im": v_im,
            "ngram_beams_re": ngram_beams_re,
            "ngram_beams_im": ngram_beams_im,
            "gates": gates,
            "next_ket_re": next_ket_re,
            "next_ket_im": next_ket_im,
            "act_deep_comb_re": act_deep_re,
            "act_deep_comb_im": act_deep_im,
            "schur_trace": schur_trace,
            "in_ctx_copy_weights": in_ctx_copy_weights,
            "id_attn_probs": id_attn_probs,
            "seq_attn_probs": seq_attn_probs,
            "mps_purity": mps_purity,
        }


class ExactMixedStateFourCompartmentBatonLM(FusedBF16FourCompartmentBatonLM):
    """1L/2L 4-Compartment Baton LM using Theorem 16 Exact Mixed-State Attention Schur Contraction."""

    def __init__(self, config: FusedBF16FourCompartmentBatonConfig) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList(
            [ExactMixedStateFourCompartmentBatonLayer(config) for _ in range(config.num_layers)]
        )

    def forward_with_diagnostics(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        ingest_online: bool = False,
        return_logits: bool = True,
    ) -> dict[str, Any]:
        """Execute the Exact Mixed-State (`Theorem 16`) 1L/2L 4-Compartment Baton highway (`<= 8` args)."""
        use_amp = self.config.use_bfloat16_amp and idx.is_cuda
        amp_dtype = torch.bfloat16 if use_amp else torch.float32

        with torch.amp.autocast(device_type=idx.device.type, dtype=amp_dtype, enabled=use_amp):
            bsz, seq_len = idx.shape
            d = self.config.hilbert_dim

            ket_re, ket_im = normalize_complex_ket(
                F.embedding(idx, self.token_ket_re).to(amp_dtype),
                F.embedding(idx, self.token_ket_im).to(amp_dtype),
            )
            steps = torch.arange(1, seq_len + 1, device=idx.device, dtype=amp_dtype).view(
                1, seq_len, 1
            )
            doc_ket_re, doc_ket_im = normalize_complex_ket(
                (torch.cumsum(ket_re, dim=1) / steps).to(amp_dtype),
                (torch.cumsum(ket_im, dim=1) / steps).to(amp_dtype),
            )
            doc_diag = doc_ket_re.square() + doc_ket_im.square()
            uni_bath_re_snap = self.online_uni_bath_re.detach().clone()
            uni_bath_im_snap = self.online_uni_bath_im.detach().clone()
            trans_ema_re_snap = self.online_trans_ema_re.detach().clone()
            trans_ema_im_snap = self.online_trans_ema_im.detach().clone()
            uni_probs_snap = self.online_uni_token_probs.detach().clone()

            uni_global_diag = torch.diagonal(uni_bath_re_snap).view(1, 1, d).to(amp_dtype)
            uni_diag = (1.0 - DEFAULT_DOC_BATH_MIX) * uni_global_diag + DEFAULT_DOC_BATH_MIX * doc_diag

            id_ket_re, id_ket_im = encode_walsh_hadamard_token_id_kets(
                idx, self.walsh_parity_pairs, d
            )
            id_ket_re = id_ket_re.to(amp_dtype)
            id_ket_im = id_ket_im.to(amp_dtype)

            hash_ids = self._compute_causal_ngram_hash_ids(idx)
            hash_ket_re = (
                F.embedding(hash_ids, self.online_ngram_hash_re).detach().clone().to(amp_dtype)
            )
            hash_ket_im = (
                F.embedding(hash_ids, self.online_ngram_hash_im).detach().clone().to(amp_dtype)
            )

            geom = self._get_cached_sequence_geometry(seq_len, idx.device, amp_dtype)
            aux_pack = {
                "cos_pos": geom["cos_pos"],
                "sin_pos": geom["sin_pos"],
                "head_masks": geom["head_masks"],
                "strict_past_mask": geom["strict_past_mask"],
                "causal_decay": geom["causal_decay"],
                "local_op": geom["local_op"],
                "id_ket_re": id_ket_re,
                "id_ket_im": id_ket_im,
            }

            curr_ket_re, curr_ket_im = ket_re, ket_im
            prev_act_deep_re: Tensor | None = None
            prev_act_deep_im: Tensor | None = None
            layer_out: dict[str, Tensor] = {}

            if use_amp and not hasattr(self, "_compiled_layers_list"):
                self._compiled_layers_list = [torch.compile(lyr) for lyr in self.layers]
            active_layers = self._compiled_layers_list if use_amp else self.layers

            for layer_idx, layer_fn in enumerate(active_layers):
                layer_out = layer_fn(
                    curr_ket_re,
                    curr_ket_im,
                    uni_diag,
                    trans_ema_re_snap,
                    trans_ema_im_snap,
                    hash_ket_re,
                    hash_ket_im,
                    aux_pack,
                )
                curr_ket_re = layer_out["next_ket_re"]
                curr_ket_im = layer_out["next_ket_im"]
                if layer_idx == 0:
                    prev_act_deep_re = layer_out["act_deep_comb_re"]
                    prev_act_deep_im = layer_out["act_deep_comb_im"]

            v_re = layer_out["v_re"]
            v_im = layer_out["v_im"]
            seq_attn_probs = layer_out["seq_attn_probs"]
            ngram_beams_re = layer_out["ngram_beams_re"]
            ngram_beams_im = layer_out["ngram_beams_im"]
            id_attn_probs = layer_out["id_attn_probs"].to(amp_dtype)
            gates = layer_out["gates"].to(amp_dtype)

            probe_re, probe_im = normalize_complex_ket(
                torch.matmul(curr_ket_re, self.readout_probe_re)
                - torch.matmul(curr_ket_im, self.readout_probe_im),
                torch.matmul(curr_ket_re, self.readout_probe_im)
                + torch.matmul(curr_ket_im, self.readout_probe_re),
            )

            act_deep_re, act_deep_im, _ = exact_mixed_state_attention_schur_action(
                seq_attn_probs, v_re, v_im, ngram_beams_re, ngram_beams_im, probe_re, probe_im
            )
            if (
                self.config.num_layers > 1
                and prev_act_deep_re is not None
                and prev_act_deep_im is not None
            ):
                act_deep_re = 0.70 * act_deep_re + 0.30 * prev_act_deep_re
                act_deep_im = 0.70 * act_deep_im + 0.30 * prev_act_deep_im

            bath_re_cast = uni_bath_re_snap.to(amp_dtype)
            bath_im_cast = uni_bath_im_snap.to(amp_dtype)
            act_ng_uni_re, act_ng_uni_im, _ = diagonal_conjugated_bath_schur_action(
                ngram_beams_re, ngram_beams_im, bath_re_cast, bath_im_cast, probe_re, probe_im
            )
            act_uni_re = torch.matmul(probe_re, bath_re_cast.transpose(-2, -1)) - torch.matmul(
                probe_im, bath_im_cast.transpose(-2, -1)
            )
            act_uni_im = torch.matmul(probe_re, bath_im_cast.transpose(-2, -1)) + torch.matmul(
                probe_im, bath_re_cast.transpose(-2, -1)
            )
            act_id_re, act_id_im = exact_token_id_mixed_state_action(
                id_attn_probs, id_ket_re, probe_re, probe_im
            )

            w_id = gates[..., 0:1]
            w_deep = gates[..., 1:2]
            w_ngram = gates[..., 2:3]
            w_uni = gates[..., 3:4]

            mix_re_cast = self.beam_readout_mix_re.to(amp_dtype)
            mix_im_cast = self.beam_readout_mix_im.to(amp_dtype)
            beam_mix_re = (ngram_beams_re * mix_re_cast - ngram_beams_im * mix_im_cast).sum(dim=-2)
            beam_mix_im = (ngram_beams_re * mix_im_cast + ngram_beams_im * mix_re_cast).sum(dim=-2)

            state_vec_re = (
                w_id * act_id_re
                + w_deep * (act_deep_re + 0.35 * beam_mix_re)
                + w_ngram * (act_ng_uni_re + 0.25 * hash_ket_re)
                + (w_uni + DEFAULT_VACUUM_EPS) * act_uni_re
            )
            state_vec_im = (
                w_id * act_id_im
                + w_deep * (act_deep_im + 0.35 * beam_mix_im)
                + w_ngram * (act_ng_uni_im + 0.25 * hash_ket_im)
                + (w_uni + DEFAULT_VACUUM_EPS) * act_uni_im
            )
            readout_re, readout_im = normalize_complex_ket(
                state_vec_re + 0.50 * curr_ket_re, state_vec_im + 0.50 * curr_ket_im
            )

            vocab_re, vocab_im = normalize_complex_ket(
                self.token_ket_re.to(amp_dtype), self.token_ket_im.to(amp_dtype)
            )
            packed_readout = torch.cat([readout_re, readout_im], dim=-1).to(amp_dtype)
            packed_vocab_t = (
                torch.cat([vocab_re, vocab_im], dim=-1).transpose(0, 1).contiguous().to(amp_dtype)
            )

            uni_log_prior = (
                torch.log(uni_probs_snap.clamp_min(1e-9)).view(1, 1, -1).to(amp_dtype)
            )
            in_ctx_weights = layer_out["in_ctx_copy_weights"].to(amp_dtype)
            combined_copy_mass = (
                DEFAULT_IN_CONTEXT_COPY_SCALE * in_ctx_weights
                + DEFAULT_TOKEN_ID_COPY_SCALE * w_id * id_attn_probs
            ).to(amp_dtype)
            copy_index = idx.unsqueeze(1).expand(bsz, seq_len, seq_len)
            ce_fn = _get_softcap_cross_entropy_fn(use_amp)

            logits: Tensor | None = None
            loss: Tensor | None = None
            top1_acc_val: float = 0.0
            if targets is not None and not return_logits and bsz >= 8:
                chunk_b = 16
                chunk_losses: list[Tensor] = []
                correct_count = 0
                for b_start in range(0, bsz, chunk_b):
                    b_end = min(bsz, b_start + chunk_b)
                    pr_c = packed_readout[b_start:b_end]
                    wu_c = w_uni[b_start:b_end]
                    ci_c = copy_index[b_start:b_end]
                    cm_c = combined_copy_mass[b_start:b_end]
                    tg_c = targets[b_start:b_end]

                    born_c = torch.matmul(pr_c, packed_vocab_t)
                    chunk_losses.append(
                        ce_fn(born_c, wu_c, uni_log_prior, ci_c, cm_c, tg_c)
                        * float(b_end - b_start)
                    )
                    if not self.training:
                        lg_eval = (
                            DEFAULT_BORN_READOUT_SCALE * born_c
                            + (DEFAULT_UNIGRAM_PRIOR_WEIGHT + 0.50 * wu_c) * uni_log_prior
                        ).scatter_add(dim=-1, index=ci_c, src=cm_c)
                        correct_count += int((lg_eval.argmax(dim=-1) == tg_c).sum().item())
                loss = torch.stack(chunk_losses).sum() / float(bsz)
                if not self.training:
                    top1_acc_val = float(correct_count) / float(bsz * seq_len)
            else:
                logits = (
                    DEFAULT_BORN_READOUT_SCALE * torch.matmul(packed_readout, packed_vocab_t)
                    + (DEFAULT_UNIGRAM_PRIOR_WEIGHT + 0.50 * w_uni) * uni_log_prior
                )
                logits = logits.scatter_add(dim=-1, index=copy_index, src=combined_copy_mass)
                logits = DEFAULT_LOGIT_SOFTCAP * torch.tanh(logits / DEFAULT_LOGIT_SOFTCAP)
                if targets is not None:
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
                    )
                    top1_acc_val = float(
                        (logits.argmax(dim=-1) == targets).to(torch.float32).mean().item()
                    )

        if ingest_online and self.training:
            self.ingest_online_token_batch(idx)

        out_purity = gram_matrix_mps_purity(ngram_beams_re.float(), ngram_beams_im.float())
        return {
            "logits": logits,
            "loss": loss,
            "top1_acc": top1_acc_val,
            "gates": gates.detach(),
            "gate_means": gates.detach().float().mean(dim=(0, 1)),
            "output_purity_mean": float(out_purity.detach().mean().item()),
            "mps_purity_mean": float(layer_out["mps_purity"].detach().float().mean().item()),
        }


# ==============================================================================
# Equivalence Audit: Wave 73 Explicit d×d vs Wave 75 Theorem 16 vs Wave 74
# ==============================================================================
@torch.no_grad()
def verify_wave73_exact_mixed_state_equivalence(
    batch_size: int = 2,
    seq_len: int = 16,
    hilbert_dim: int = 64,
    seed: int = 1337,
) -> dict[str, Any]:
    """Audit FP32 algebraic equivalence between `Wave 73` explicit `d × d` density matrices and `Wave 75` Theorem 16 (`<= 8` args)."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    h_cnt = NUM_SEQ_ATTN_PACKETS
    r_cnt = DEFAULT_SCHUR_BEAM_RANK

    raw_scores = torch.rand(batch_size, h_cnt, seq_len, seq_len, device=device, dtype=dtype)
    t_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    s_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    causal = (s_idx <= t_idx).to(dtype)
    masked = raw_scores * causal.unsqueeze(0).unsqueeze(0)
    attn_probs = masked / (masked.sum(dim=-1, keepdim=True) + DEFAULT_TRACE_EPS)

    v_re, v_im = normalize_complex_ket(
        torch.randn(batch_size, h_cnt, seq_len, hilbert_dim, device=device, dtype=dtype),
        torch.randn(batch_size, h_cnt, seq_len, hilbert_dim, device=device, dtype=dtype),
    )
    raw_ng_re = torch.randn(batch_size, seq_len, r_cnt, hilbert_dim, device=device, dtype=dtype)
    raw_ng_im = torch.randn(batch_size, seq_len, r_cnt, hilbert_dim, device=device, dtype=dtype)
    ng_frob = (raw_ng_re.square() + raw_ng_im.square()).sum(dim=(-2, -1), keepdim=True)
    ngram_beams_re = raw_ng_re * torch.rsqrt(ng_frob)
    ngram_beams_im = raw_ng_im * torch.rsqrt(ng_frob)

    probe_re, probe_im = normalize_complex_ket(
        torch.randn(batch_size, seq_len, hilbert_dim, device=device, dtype=dtype),
        torch.randn(batch_size, seq_len, hilbert_dim, device=device, dtype=dtype),
    )

    # 1. Explicit Wave 73 d×d Mixed-State Density Matrix Formulation
    val_rho_re, val_rho_im = pure_ket_to_density(v_re, v_im)
    seq_pkts_re = torch.einsum("bhts,bhsde->bthde", attn_probs, val_rho_re)
    seq_pkts_im = torch.einsum("bhts,bhsde->bthde", attn_probs, val_rho_im)
    seq_pkts_re, seq_pkts_im = trace_normalize_density(seq_pkts_re, seq_pkts_im)
    rho_seq_re = seq_pkts_re.mean(dim=2)
    rho_seq_im = seq_pkts_im.mean(dim=2)

    eye_d = torch.eye(hilbert_dim, device=device, dtype=dtype) / float(hilbert_dim)
    rho_ngram_re, rho_ngram_im = rank_beams_to_density(
        ngram_beams_re, ngram_beams_im, eye_d, torch.zeros_like(eye_d), eps=0.0
    )
    w73_deep_re, w73_deep_im = complex_schur_product(
        rho_seq_re, rho_seq_im, rho_ngram_re, rho_ngram_im
    )
    w73_act_re = torch.einsum("btde,bte->btd", w73_deep_re, probe_re) - torch.einsum(
        "btde,bte->btd", w73_deep_im, probe_im
    )
    w73_act_im = torch.einsum("btde,bte->btd", w73_deep_re, probe_im) + torch.einsum(
        "btde,bte->btd", w73_deep_im, probe_re
    )

    # 2. Wave 75 Theorem 16 Zero-(d, d) Exact Mixed-State Contraction
    w75_act_re, w75_act_im, w75_trace = exact_mixed_state_attention_schur_action(
        attn_probs, v_re, v_im, ngram_beams_re, ngram_beams_im, probe_re, probe_im
    )

    # 3. Wave 74 Coherent Ket-Average Approximation (for comparison)
    seq_head_re = torch.matmul(attn_probs, v_re).permute(0, 2, 1, 3)
    seq_head_im = torch.matmul(attn_probs, v_im).permute(0, 2, 1, 3)
    seq_head_re, seq_head_im = normalize_complex_ket(seq_head_re, seq_head_im)
    w74_deep_beams_re, w74_deep_beams_im, _ = factorized_rank_beams_schur_intersect(
        0.5 * seq_head_re, 0.5 * seq_head_im, ngram_beams_re, ngram_beams_im
    )
    w74_act_re, w74_act_im = apply_rank_beams_to_probe(
        w74_deep_beams_re, w74_deep_beams_im, probe_re, probe_im
    )

    # 4. Compartment 4 (Packet 7) Token-ID Mixed State Audit
    raw_id_scores = torch.rand(batch_size, seq_len, seq_len, device=device, dtype=dtype) * causal.unsqueeze(0)
    id_attn_probs = raw_id_scores / (raw_id_scores.sum(dim=-1, keepdim=True) + DEFAULT_TRACE_EPS)
    id_ket_re = F.normalize(
        torch.randn(batch_size, seq_len, hilbert_dim, device=device, dtype=dtype), dim=-1
    )
    id_ket_im = torch.zeros_like(id_ket_re)
    id_pure_re, id_pure_im = pure_ket_to_density(id_ket_re, id_ket_im)
    rho_id_re = torch.einsum("bts,bsde->btde", id_attn_probs, id_pure_re)
    rho_id_im = torch.einsum("bts,bsde->btde", id_attn_probs, id_pure_im)
    rho_id_re, rho_id_im = trace_normalize_density(rho_id_re, rho_id_im)
    w73_id_act_re = torch.einsum("btde,bte->btd", rho_id_re, probe_re)
    w73_id_act_im = torch.einsum("btde,bte->btd", rho_id_re, probe_im)
    w75_id_act_re, w75_id_act_im = exact_token_id_mixed_state_action(
        id_attn_probs, id_ket_re, probe_re, probe_im
    )

    w75_vs_w73_deep_max_err = float(
        torch.max(
            (w75_act_re - w73_act_re).abs().max(), (w75_act_im - w73_act_im).abs().max()
        ).item()
    )
    w75_vs_w73_id_max_err = float(
        torch.max(
            (w75_id_act_re - w73_id_act_re).abs().max(),
            (w75_id_act_im - w73_id_act_im).abs().max(),
        ).item()
    )
    w74_vs_w73_deep_diff = float(
        torch.max(
            (w74_act_re - w73_act_re).abs().max(), (w74_act_im - w73_act_im).abs().max()
        ).item()
    )

    passed = (
        w75_vs_w73_deep_max_err <= EXACT_EQUIVALENCE_FP32_TOL
        and w75_vs_w73_id_max_err <= EXACT_EQUIVALENCE_FP32_TOL
    )
    return {
        "passed": passed,
        "w75_theorem16_vs_w73_explicit_deep_max_abs_err": w75_vs_w73_deep_max_err,
        "w75_theorem16_vs_w73_explicit_id_max_abs_err": w75_vs_w73_id_max_err,
        "w74_coherent_ket_avg_vs_w73_explicit_diff": w74_vs_w73_deep_diff,
        "mean_schur_trace": float(w75_trace.mean().item()),
        "tolerance": EXACT_EQUIVALENCE_FP32_TOL,
    }


# ==============================================================================
# Shadow-EMA Parameter Tracker (`Hot Val CE`, `EMA Val CE`, `Thermal Gap`)
# ==============================================================================
class ShadowEMAParameterTracker:
    """Tracks exponential moving average `θ_ema` of trainable parameters to evaluate `Hot` vs `EMA` Val CE.

    At any checkpoint:
    - `hot_val_ce`: Validation Cross-Entropy of the live ("hot") optimizer state `θ_hot`.
    - `ema_val_ce`: Validation Cross-Entropy of the quenched shadow state `θ_ema`.
    - `thermal_gap_ce = hot_val_ce - ema_val_ce`: Positive gap measures how much high-LR
      parameter jitter ("thermal noise") is degrading the live weights relative to the smoothed trajectory.
    """

    def __init__(self, model: nn.Module, beta_ema: float = DEFAULT_SHADOW_EMA_DECAY) -> None:
        self.beta_ema = float(beta_ema)
        self.num_updates: int = 0
        self.shadow_params: dict[str, Tensor] = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow_params[name] = param.detach().float().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update `θ_ema <- β_ema * θ_ema + (1 - β_ema) * θ_hot` after an optimizer step (`<= 8` args)."""
        b = self.beta_ema
        one_minus_b = 1.0 - b
        self.num_updates += 1
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.shadow_params[name].mul_(b).add_(param.detach().float(), alpha=one_minus_b)

    @torch.no_grad()
    def _eval_current_weights(
        self,
        model: nn.Module,
        val_mm: np.memmap,
        eval_batches: int,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> dict[str, Any]:
        """Evaluate current model parameters on deterministic FineWeb validation batches (`<= 8` args)."""
        was_training = model.training
        model.eval()
        losses: list[float] = []
        accs: list[float] = []
        gate_sums = torch.zeros(4, dtype=torch.float32)
        purity_sum = 0.0
        span = batch_size * (seq_len + 1)

        for b_idx in range(eval_batches):
            start = b_idx * batch_size * seq_len
            raw = np.array(val_mm[start : start + span], dtype=np.int64)
            chunk = torch.from_numpy(raw).view(batch_size, seq_len + 1).to(device)
            x_val = chunk[:, :-1].contiguous()
            y_val = chunk[:, 1:].contiguous()

            out = model.forward_with_diagnostics(
                x_val, targets=y_val, ingest_online=False, return_logits=False
            )
            losses.append(float(out["loss"].item()))
            accs.append(float(out["top1_acc"]))
            if "gate_means" in out:
                gate_sums += out["gate_means"].cpu().float()
            elif "gates" in out:
                gate_sums += out["gates"].detach().float().mean(dim=(0, 1)).cpu()
            if "output_purity_mean" in out:
                purity_sum += float(out["output_purity_mean"])
            elif "out_purity" in out:
                purity_sum += float(out["out_purity"].detach().float().mean().item())

        if was_training:
            model.train()
        val_ce = sum(losses) / max(1, len(losses))
        val_ppl = math.exp(min(20.0, val_ce))
        top1_acc = sum(accs) / max(1, len(accs))
        gate_avg = (gate_sums / float(max(1, eval_batches))).tolist()
        return {
            "val_ce": val_ce,
            "val_ppl": val_ppl,
            "top1_acc": top1_acc,
            "gate_means": gate_avg,
            "output_purity": purity_sum / float(max(1, eval_batches)),
        }

    @torch.no_grad()
    def evaluate_hot_and_ema(
        self,
        model: nn.Module,
        val_mm: np.memmap,
        eval_batches: int = DEFAULT_EVAL_BATCHES,
        batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
        seq_len: int = 256,
    ) -> dict[str, Any]:
        """Evaluate both `Hot` (`θ_hot`) and `EMA` (`θ_ema`) parameters and restore `θ_hot` bit-for-bit (`<= 8` args)."""
        device = next(model.parameters()).device
        hot_metrics = self._eval_current_weights(
            model, val_mm, eval_batches, batch_size, seq_len, device
        )

        hot_backup: dict[str, Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                hot_backup[name] = param.detach().clone()
                param.copy_(self.shadow_params[name].to(dtype=param.dtype, device=param.device))

        ema_metrics = self._eval_current_weights(
            model, val_mm, eval_batches, batch_size, seq_len, device
        )

        for name, param in model.named_parameters():
            if name in hot_backup:
                param.copy_(hot_backup[name])

        hot_ce = hot_metrics["val_ce"]
        ema_ce = ema_metrics["val_ce"]
        thermal_gap_ce = hot_ce - ema_ce
        return {
            "hot_val_ce": hot_ce,
            "hot_val_ppl": hot_metrics["val_ppl"],
            "hot_top1_acc": hot_metrics["top1_acc"],
            "ema_val_ce": ema_ce,
            "ema_val_ppl": ema_metrics["val_ppl"],
            "ema_top1_acc": ema_metrics["top1_acc"],
            "thermal_gap_ce": thermal_gap_ce,
            "gate_means": hot_metrics["gate_means"],
            "output_purity": hot_metrics["output_purity"],
            "ema_num_updates": self.num_updates,
        }


# ==============================================================================
# Width-Scaling Throughput & VRAM Benchmark (`d ∈ {64, 128, 192, 256}`)
# ==============================================================================
def benchmark_width_scaling_throughput(
    widths: tuple[int, ...] = DEFAULT_WIDTH_CHOICES,
    num_layers_choices: tuple[int, ...] = (1, 2),
    batch_size: int = 32,
    seq_len: int = 256,
    warmup_steps: int = 3,
    timed_steps: int = 10,
) -> list[dict[str, Any]]:
    """Benchmark `tok/s`, step latency (`ms`), and peak VRAM (`MiB`) across `d ∈ {64, 128, 192, 256}` (`<= 8` args)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: list[dict[str, Any]] = []

    for mode_name, model_cls in [
        ("W74_FUSED_BF16_KET_AVG", FusedBF16FourCompartmentBatonLM),
        ("W75_EXACT_MIXED_STATE_THM16", ExactMixedStateFourCompartmentBatonLM),
    ]:
        for num_layers in num_layers_choices:
            for d in widths:
                torch._dynamo.reset()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)

                cfg = FusedBF16FourCompartmentBatonConfig(
                    vocab_size=DEFAULT_VOCAB_SIZE,
                    hilbert_dim=d,
                    num_layers=num_layers,
                    use_two_hop_triangle=True,
                    use_bfloat16_amp=True,
                )
                model = model_cls(cfg).to(device)
                model.train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
                param_count = sum(p.numel() for p in model.parameters())

                x = torch.randint(0, DEFAULT_VOCAB_SIZE, (batch_size, seq_len), device=device)
                y = torch.randint(0, DEFAULT_VOCAB_SIZE, (batch_size, seq_len), device=device)

                for _ in range(warmup_steps):
                    optimizer.zero_grad(set_to_none=True)
                    out = model.forward_with_diagnostics(
                        x, targets=y, ingest_online=True, return_logits=False
                    )
                    out["loss"].backward()
                    optimizer.step()

                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                for _ in range(timed_steps):
                    optimizer.zero_grad(set_to_none=True)
                    out = model.forward_with_diagnostics(
                        x, targets=y, ingest_online=True, return_logits=False
                    )
                    out["loss"].backward()
                    optimizer.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                elapsed = max(1e-6, time.perf_counter() - t0)

                tok_per_s = float(timed_steps * batch_size * seq_len) / elapsed
                ms_per_step = (elapsed * 1000.0) / float(timed_steps)
                peak_vram_mib = (
                    float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
                    if torch.cuda.is_available()
                    else 0.0
                )

                rec = {
                    "mode": mode_name,
                    "num_layers": num_layers,
                    "hilbert_dim": d,
                    "real_embedding_width": 2 * d,
                    "param_count": param_count,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "tok_per_s": tok_per_s,
                    "ms_per_step": ms_per_step,
                    "peak_vram_mib": peak_vram_mib,
                }
                results.append(rec)
                print(
                    f"[width_bench] mode={mode_name} {num_layers}L d={d} (2d={2*d}) "
                    f"params={param_count:,} tok/s={tok_per_s:,.1f} "
                    f"ms/step={ms_per_step:.2f} vram={peak_vram_mib:.1f} MiB",
                    flush=True,
                )
                del model, optimizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return results


# ==============================================================================
# Width-Scaling Training Runner with Hot & Shadow-EMA Val CE Logging
# ==============================================================================
@dataclass(frozen=True)
class Wave75WidthRunSpec:
    """Specification for a `Wave 75` width-scaled training run with Hot & EMA Val CE (`<= 8` fields)."""

    run_key: str
    hilbert_dim: int
    num_layers: int
    exact_mixed_state: bool
    batch_size: int
    seq_len: int
    total_steps: int
    eval_interval: int


def run_wave75_width_training_with_ema(
    spec: Wave75WidthRunSpec,
    lr_peak: float = 3.0e-3,
    lr_floor: float = 3.0e-4,
    beta_ema: float = DEFAULT_SHADOW_EMA_DECAY,
) -> dict[str, Any]:
    """Train a `Wave 75` width-scaled 4-Compartment Baton LM and log both `Hot` and `EMA` Val CE (`<= 8` args)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1337)
    torch._dynamo.reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    cfg = FusedBF16FourCompartmentBatonConfig(
        vocab_size=DEFAULT_VOCAB_SIZE,
        hilbert_dim=spec.hilbert_dim,
        num_layers=spec.num_layers,
        use_two_hop_triangle=True,
        use_bfloat16_amp=True,
    )
    model_cls = (
        ExactMixedStateFourCompartmentBatonLM
        if spec.exact_mixed_state
        else FusedBF16FourCompartmentBatonLM
    )
    model = model_cls(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    ema_tracker = ShadowEMAParameterTracker(model, beta_ema=beta_ema)
    streamer = MultiShardFineWebStreamer(FINEWEB_TRAIN_SHARDS_300M)
    val_mm = np.memmap(FINEWEB_VAL_SHARD_PATH, dtype=np.uint16, mode="r", offset=256 * 4)

    emb_params = [model.token_ket_re, model.token_ket_im]
    emb_ids = {id(p) for p in emb_params}
    matrix_params = [p for p in model.parameters() if id(p) not in emb_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": emb_params, "lr": lr_peak, "weight_decay": 0.0},
            {"params": matrix_params, "lr": lr_peak * 0.75, "weight_decay": 0.01},
        ],
        betas=(0.9, 0.95),
        fused=torch.cuda.is_available(),
    )

    tokens_per_step = spec.batch_size * spec.seq_len
    target_tokens_m = float(spec.total_steps * tokens_per_step) / 1e6
    arch_tag = "ExactMixedState-Thm16" if spec.exact_mixed_state else "FusedBF16-KetAvg"

    intent = ExperimentIntentSpec(
        run_key=spec.run_key,
        wave_id="W75",
        plan_id="EXPERIMENT_PLAN_095",
        task_domain="fineweb_speedrun",
        hypothesis_intent=(
            f"Test {arch_tag} {spec.num_layers}L at complex width d={spec.hilbert_dim} (real 2d={2*spec.hilbert_dim}) "
            f"with live Hot Val CE and Shadow-EMA Val CE (beta_ema={beta_ema}) tracking."
        ),
        architecture_summary=(
            f"{model_cls.__name__}(d={spec.hilbert_dim}, layers={spec.num_layers}, "
            f"params={param_count:,}, beta_ema={beta_ema})"
        ),
        hardware_budget="Local RTX 5060 Ti 16GB (bfloat16 + torch.compile)",
        batch_size=spec.batch_size,
        seq_len=spec.seq_len,
        total_steps=spec.total_steps,
        target_tokens_M=target_tokens_m,
    )
    logger = StandardizedExperimentLogger(intent)

    eval_history: list[dict[str, Any]] = []
    best_hot_ce = float("inf")
    best_ema_ce = float("inf")
    warmup_steps = max(20, int(0.05 * spec.total_steps))
    t_start = time.perf_counter()

    # Step 0 Cold Evaluation
    eval0 = ema_tracker.evaluate_hot_and_ema(
        model, val_mm, eval_batches=DEFAULT_EVAL_BATCHES, batch_size=DEFAULT_EVAL_BATCH_SIZE, seq_len=spec.seq_len
    )
    eval0["step"] = 0
    eval0["tokens_M"] = 0.0
    eval_history.append(eval0)
    logger.log_eval(
        step=0,
        metrics={
            "val_ce": round(eval0["ema_val_ce"], 4),
            "val_ppl": round(eval0["ema_val_ppl"], 2),
            "top1_acc": round(eval0["ema_top1_acc"], 4),
            "hot_val_ce": round(eval0["hot_val_ce"], 4),
            "ema_val_ce": round(eval0["ema_val_ce"], 4),
            "thermal_gap_ce": round(eval0["thermal_gap_ce"], 4),
        },
    )

    model.train()
    for step in range(1, spec.total_steps + 1):
        if step <= warmup_steps:
            lr_mult = float(step) / float(max(1, warmup_steps))
        else:
            prog = float(step - warmup_steps) / float(max(1, spec.total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
            min_ratio = lr_floor / lr_peak
            lr_mult = min_ratio + (1.0 - min_ratio) * cosine

        optimizer.param_groups[0]["lr"] = lr_peak * lr_mult
        optimizer.param_groups[1]["lr"] = (lr_peak * 0.75) * lr_mult

        x_b, y_b, _ = streamer.sample_sequential_batch(step - 1, spec.batch_size, spec.seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        out = model.forward_with_diagnostics(
            x_b, targets=y_b, ingest_online=True, return_logits=False
        )
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ema_tracker.update(model)

        if step % 100 == 0 or step == spec.total_steps:
            elapsed = max(1e-6, time.perf_counter() - t_start)
            tok_s = float(step * tokens_per_step) / elapsed
            logger.log_step(
                step=step,
                train_ce=float(loss.item()),
                tok_per_s=tok_s,
                lr_mult=lr_mult,
            )

        if step % spec.eval_interval == 0 or step == spec.total_steps:
            ev = ema_tracker.evaluate_hot_and_ema(
                model,
                val_mm,
                eval_batches=DEFAULT_EVAL_BATCHES,
                batch_size=DEFAULT_EVAL_BATCH_SIZE,
                seq_len=spec.seq_len,
            )
            tokens_m = float(step * tokens_per_step) / 1e6
            elapsed = max(1e-6, time.perf_counter() - t_start)
            tok_s = float(step * tokens_per_step) / elapsed
            ev["step"] = step
            ev["tokens_M"] = tokens_m
            ev["tok_per_s"] = tok_s
            ev["lr"] = optimizer.param_groups[0]["lr"]
            eval_history.append(ev)

            best_hot_ce = min(best_hot_ce, ev["hot_val_ce"])
            best_ema_ce = min(best_ema_ce, ev["ema_val_ce"])

            logger.log_eval(
                step=step,
                metrics={
                    "val_ce": round(ev["ema_val_ce"], 4),
                    "val_ppl": round(ev["ema_val_ppl"], 2),
                    "top1_acc": round(ev["ema_top1_acc"], 4),
                    "hot_val_ce": round(ev["hot_val_ce"], 4),
                    "ema_val_ce": round(ev["ema_val_ce"], 4),
                    "thermal_gap_ce": round(ev["thermal_gap_ce"], 4),
                    "hot_top1_acc": round(ev["hot_top1_acc"], 4),
                    "ema_top1_acc": round(ev["ema_top1_acc"], 4),
                },
            )
            print(
                f"[{spec.run_key}] step={step}/{spec.total_steps} ({tokens_m:.2f}M tok) | "
                f"Hot CE={ev['hot_val_ce']:.4f} | EMA CE={ev['ema_val_ce']:.4f} | "
                f"Thermal Gap={ev['thermal_gap_ce']:+.4f} | EMA Top1={ev['ema_top1_acc']*100:.2f}% | "
                f"tok/s={tok_s:,.1f}",
                flush=True,
            )

    total_wall_s = max(1e-6, time.perf_counter() - t_start)
    logger.finalize()
    avg_tok_s = float(spec.total_steps * tokens_per_step) / total_wall_s
    peak_vram_mib = (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
        if torch.cuda.is_available()
        else 0.0
    )
    final_ev = eval_history[-1]
    return {
        "spec": asdict(spec),
        "param_count": param_count,
        "best_hot_val_ce": best_hot_ce,
        "best_ema_val_ce": best_ema_ce,
        "final_hot_val_ce": final_ev["hot_val_ce"],
        "final_ema_val_ce": final_ev["ema_val_ce"],
        "final_thermal_gap_ce": final_ev["thermal_gap_ce"],
        "final_hot_top1_acc": final_ev["hot_top1_acc"],
        "final_ema_top1_acc": final_ev["ema_top1_acc"],
        "avg_tok_per_s": avg_tok_s,
        "total_wall_s": total_wall_s,
        "peak_vram_mib": peak_vram_mib,
        "eval_history": eval_history,
    }
