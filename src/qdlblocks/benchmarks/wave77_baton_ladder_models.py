"""Wave 77 5-Rung Quantum Baton Evidence Ladder Models, MuP Optimizers, & Adapters (`EXPERIMENT_PLAN_097.md`).

Provenance & Mathematical Rationale
-----------------------------------
Built for `Wave 77` (`EXPERIMENT_PLAN_097.md`) following Rules 1–8 of `docs/SCRIPT_BEST_PRACTICES.md`
(`<= 1,000` SLOC, `<= 8` function arguments, strategy enums over boolean-flag jungles, zero mutation
of frozen `Wave 74–76` baselines):

- **Rung 1 (`R1_CALIB_COPY_SINK_2L_D64`)**:
  Isolates `<BOS>` anchor-sink normalization (`row_sum == 1.0`, `0.0%` cross-`<BOS>` leakage) and
  vacuum-normalized confidence-gated `[0..12.5]` associative copy log-odds on the exact `Wave 76`
  `2L, d=64` Baton trunk (`CalibratedCopySinkBatonLM`).
- **Rung 2 (`R2_PRUNED_UNTIED_2L_D64`)**:
  Prunes dead `C2` (`MPS/N-gram`: `< 0.005` nats) and `C4` (`Walsh Token-ID`: `< 0.003` nats) readout
  ket dilution (`38%` wasted amplitude in `Wave 76` LOO) and decouples input ket embeddings (with
  `<BOS>`-safe 1-token `SmearGate`) from an untied `bfloat16` vocabulary readout (`lm_head`), still
  under `AdamW` with zero wide channel mixer.
- **Rung 3 (`R3_MUON_MIXER_2L_D64`)**:
  Adds the `StiefelMuon`-conditioned Two-Beam Quantum Interference / `ReLU²` Channel-Mixing Bank
  (`D -> 3D -> D`), Layer-0 value anchor (`v_0`), and Carrier/Self-Echo Orthogonal Projection (`XSA`,
  Wave 44 Invariant #3) at matched `2L, d=64` (`128D` real), closing the `Out-of-Doc Recall` deficit.
- **Rung 4 (`R4_MUP_2L_D128`, `R4_MUP_2L_D256`, `R4_MUP_4L_D256`)**:
  Applies zero-shot $\\mu\\text{P}$ + `StiefelMuon` width/depth transfer across $d \\in \\{128, 256\\}$
  and $L \\in \\{2, 4\\}$ (with U-Net encoder-decoder skip connections at $L = 4$).
- **Rung 5 (`R5_CTRL_SCHUR_BASE_2L_D128`, `R5A_CARBON_COMP_SCHUR_2L_D128`, `R5B_CARBON_COMP_SCHUR_MTP_2L_D128`)**:
  Executes Carbon's 3-way routing ablation at matched `2L, d=128` (`8` subsystem heads of $\\mathbb{C}^{16}$):
  `(a)` post-selected Schur mixer, `(b)` trace-preserving Complementary Schur Gate, and `(c)`
  Complementary Schur Gate + strictly causal $K=4, \\chi=8$ backward-then-forward virtual MTP MPS sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..optim.muon import StiefelMuon, StiefelMuonConfig
from .architecture_xray_diagnostics import (
    FINEWEB_BOS_TOKEN_ID,
    VALID_FINEWEB_VOCAB_SIZE,
    DiagnosticModelAdapter,
    SharpenedPBosonDocMaskedBatonLayer,
    SharpenedPBosonDocMaskedBatonLM,
)
from .fused_bf16_four_compartment_baton_lab import (
    DEFAULT_BORN_READOUT_SCALE,
    DEFAULT_DOC_BATH_MIX,
    DEFAULT_PURITY_FLOOR_OFFSET,
    DEFAULT_PURITY_MIN_CLAMP,
    DEFAULT_TOKEN_ID_COPY_SCALE,
    DEFAULT_TRACE_EPS,
    DEFAULT_TWO_HOP_MIX_ALPHA,
    DEFAULT_UNIGRAM_PRIOR_WEIGHT,
    DEFAULT_VACUUM_EPS,
    FusedBF16FourCompartmentBatonConfig,
    apply_rank_beams_to_probe,
    diagonal_conjugated_bath_schur_action,
    encode_walsh_hadamard_token_id_kets,
    factorized_rank_beams_schur_intersect,
    normalize_complex_ket,
)
from .wave77_quantum_routing_primitives import (
    COMPLEX_HEAD_DIM,
     DEFAULT_COPY_BOSON_POWER,
    DEFAULT_COPY_MAX_LOGIT_GAIN,
    CalibratedPBosonAssociativeCopyGate,
    CausalVirtualMTPBackwardForwardMPSSweep,
    ComplementarySchurSubsystemGate,
    apply_bos_anchor_sink_row_norm,
    build_bos_anchor_sink_doc_masks,
)

# ==============================================================================
# Top-of-File Named Constants (Synchronized with EXPERIMENT_PLAN_097.md Section 3)
# ==============================================================================
MUP_BASE_COMPLEX_DIM: int = 128
MUP_BASE_REAL_DIM: int = 2 * MUP_BASE_COMPLEX_DIM
NAIVE_SP_FIXED_ADAM_LR: float = 3.0e-3

W77_MUON_BASE_LR: float = 0.035
W77_WTE_BASE_LR: float = 0.20
W77_HEAD_BASE_LR_AT_D256: float = 0.016
W77_SCALAR_BASE_LR: float = 0.035
W77_MUON_MOMENTUM: float = 0.95
W77_MUON_MOMENTUM_INIT: float = 0.85

DEFAULT_W77_LOGIT_SOFTCAP: float = 20.0
DEFAULT_W77_CE_CHUNK_TOKENS: int = 4096
DEFAULT_ONLINE_UNI_PRIOR_SCALE: float = 0.35
DEFAULT_ONLINE_UNI_EMA_DECAY: float = 0.02
DEFAULT_W77_ROPE_BASE: float = 1024.0
DEFAULT_W77_ATTN_SCALE: float = 0.18
OFFICIAL_SHARD0_VAL_TOKENS: int = 10_485_760
WSD_COOLDOWN_FRACTION: float = 0.18

ChannelMixerMode = Literal["none", "quantum_interference_mlp"]
OptimizerSuiteMode = Literal["adamw_sp", "mup_stiefel_muon"]
SchurRoutingMode = Literal["none", "post_selected", "complementary"]


class Wave77RungKind(str, Enum):
    """Strategy selector for all `Wave 77` Evidence Ladder rungs & Carbon ablations (Rule 3)."""

    R1_CALIB_COPY_SINK_2L_D64 = "w77_r1_calib_copy_sink_2l_d64"
    R2_PRUNED_UNTIED_2L_D64 = "w77_r2_pruned_untied_2l_d64"
    R3_MUON_MIXER_2L_D64 = "w77_r3_muon_mixer_2l_d64"
    R4_MUP_2L_D128 = "w77_r4_mup_2l_d128"
    R4_MUP_2L_D256 = "w77_r4_mup_2l_d256"
    R4_MUP_4L_D256 = "w77_r4_mup_4l_d256"
    R5_CTRL_SCHUR_BASE_2L_D128 = "w77_r5_ctrl_post_schur_2l_d128"
    R5A_CARBON_COMP_SCHUR_2L_D128 = "w77_r5a_carbon_comp_schur_2l_d128"
    R5B_CARBON_COMP_SCHUR_MTP_2L_D128 = "w77_r5b_carbon_comp_schur_mtp_2l_d128"


@dataclass(frozen=True)
class Wave77MixerConfig:
    """Configuration for `QuantumBatonMixerLM` (`<= 8` fields, Rule 2)."""

    vocab_size: int = VALID_FINEWEB_VOCAB_SIZE
    complex_dim: int = 128
    num_layers: int = 2
    mlp_expansion: int = 3
    mixer_mode: ChannelMixerMode = "quantum_interference_mlp"
    schur_mode: SchurRoutingMode = "none"
    use_causal_mtp_mps: bool = False
    logit_softcap: float = DEFAULT_W77_LOGIT_SOFTCAP


# ==============================================================================
# Rung 1 Model: Calibrated Copy Log-Odds + <BOS> Anchor Sink on Wave 76 Baton
# ==============================================================================
class CalibratedCopySinkBatonLayer(SharpenedPBosonDocMaskedBatonLayer):
    """`Wave 76` $p=3$ Baton layer upgraded with `<BOS>` anchor sink (`row_sum == 1.0` everywhere)."""

    def __init__(self, config: FusedBF16FourCompartmentBatonConfig) -> None:
        super().__init__(config, boson_power=DEFAULT_COPY_BOSON_POWER, mask_bos_documents=True)
        self.current_doc_start_pos: Tensor | None = None

    def _compute_fused_exterior_sequence_beams(
        self,
        ket_re: Tensor,
        ket_im: Tensor,
        mps_beams_re: Tensor,
        mps_beams_im: Tensor,
        mps_purity: Tensor,
        aux_pack: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute $p=3$ `<BOS>`-masked sequence attention with `<BOS>` anchor sink (`<= 8` args)."""
        from .fused_bf16_four_compartment_baton_lab import apply_complex_rope

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
        born_scores = torch.pow(mag_sq.float(), self.boson_power).to(ket_re.dtype)

        purity_rel = (mps_purity - (1.0 / float(d)) + DEFAULT_PURITY_FLOOR_OFFSET).clamp(
            min=DEFAULT_PURITY_MIN_CLAMP, max=1.0
        )
        doc_mask = self._resolve_doc_mask(aux_pack)
        valid_mask = head_masks.unsqueeze(0) > 0.0
        if doc_mask is not None:
            valid_mask = valid_mask & doc_mask.unsqueeze(1)

        doc_start = (
            self.current_doc_start_pos
            if self.current_doc_start_pos is not None
            else torch.zeros((bsz, seq_len), dtype=torch.long, device=ket_re.device)
        )
        weighted_scores = born_scores * purity_rel.unsqueeze(1).unsqueeze(2)
        attn_probs = apply_bos_anchor_sink_row_norm(weighted_scores, valid_mask, doc_start)

        if self.config.use_two_hop_triangle and seq_len > 1:
            local_op = aux_pack["local_op"]
            if doc_mask is not None:
                loc_valid = (local_op > 0.0) & doc_mask.unsqueeze(1)
                masked_loc = apply_bos_anchor_sink_row_norm(local_op, loc_valid, doc_start)
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


class CalibratedCopySinkBatonLM(SharpenedPBosonDocMaskedBatonLM):
    """Rung 1 (`w77_r1_calib_copy_sink_2l_d64`): `Wave 76` Baton + Calibrated Copy + `<BOS>` Sink."""

    def __init__(self, config: FusedBF16FourCompartmentBatonConfig) -> None:
        super().__init__(config, boson_power=DEFAULT_COPY_BOSON_POWER, mask_bos_documents=True)
        d = config.hilbert_dim
        self.layers = nn.ModuleList(
            [CalibratedCopySinkBatonLayer(config) for _ in range(config.num_layers)]
        )
        self._compiled_layers_list = list(self.layers)
        self.calib_copy_gate = CalibratedPBosonAssociativeCopyGate(
            copy_ket_dim=d,
            ctx_dim=2 * d,
            boson_power=DEFAULT_COPY_BOSON_POWER,
            max_logit_gain=DEFAULT_COPY_MAX_LOGIT_GAIN,
        )

    def forward_with_diagnostics(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        ingest_online: bool = False,
        return_logits: bool = True,
    ) -> dict[str, Any]:
        """Run `Wave 76` Baton trunk with `<BOS>` anchor sink and calibrated copy log-odds (`<= 8` args)."""
        use_amp = self.config.use_bfloat16_amp and idx.is_cuda
        amp_dtype = torch.bfloat16 if use_amp else torch.float32
        mask_pack = build_bos_anchor_sink_doc_masks(idx)

        for lyr in self.layers:
            if isinstance(lyr, CalibratedCopySinkBatonLayer):
                lyr.current_doc_same_mask = mask_pack.doc_same
                lyr.current_doc_start_pos = mask_pack.doc_start_pos

        try:
            with torch.amp.autocast(device_type=idx.device.type, dtype=amp_dtype, enabled=use_amp):
                bsz, seq_len = idx.shape
                d = self.config.hilbert_dim

                ket0_re, ket0_im = normalize_complex_ket(
                    F.embedding(idx, self.token_ket_re).to(amp_dtype),
                    F.embedding(idx, self.token_ket_im).to(amp_dtype),
                )
                steps = torch.arange(1, seq_len + 1, device=idx.device, dtype=amp_dtype).view(
                    1, seq_len, 1
                )
                doc_ket_re, doc_ket_im = normalize_complex_ket(
                    (torch.cumsum(ket0_re, dim=1) / steps).to(amp_dtype),
                    (torch.cumsum(ket0_im, dim=1) / steps).to(amp_dtype),
                )
                doc_diag = doc_ket_re.square() + doc_ket_im.square()
                uni_bath_re_snap = self.online_uni_bath_re.detach().clone()
                uni_bath_im_snap = self.online_uni_bath_im.detach().clone()
                trans_ema_re_snap = self.online_trans_ema_re.detach().clone()
                trans_ema_im_snap = self.online_trans_ema_im.detach().clone()
                uni_probs_snap = self.online_uni_token_probs.detach().clone()

                uni_global_diag = torch.diagonal(uni_bath_re_snap).view(1, 1, d).to(amp_dtype)
                uni_diag = (
                    1.0 - DEFAULT_DOC_BATH_MIX
                ) * uni_global_diag + DEFAULT_DOC_BATH_MIX * doc_diag

                id_ket_re, id_ket_im = encode_walsh_hadamard_token_id_kets(
                    idx, self.walsh_parity_pairs, d
                )
                id_ket_re, id_ket_im = id_ket_re.to(amp_dtype), id_ket_im.to(amp_dtype)
                hash_ids = self._compute_causal_ngram_hash_ids(idx)
                hash_ket_re = F.embedding(hash_ids, self.online_ngram_hash_re).detach().to(amp_dtype)
                hash_ket_im = F.embedding(hash_ids, self.online_ngram_hash_im).detach().to(amp_dtype)

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
                    "doc_same_mask": mask_pack.doc_same,
                }

                curr_ket_re, curr_ket_im = ket0_re, ket0_im
                prev_deep_re: Tensor | None = None
                prev_deep_im: Tensor | None = None
                layer_out: dict[str, Tensor] = {}
                for layer_idx, layer_fn in enumerate(self.layers):
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
                    if layer_idx > 0 and prev_deep_re is not None and prev_deep_im is not None:
                        four_re, four_im, _ = factorized_rank_beams_schur_intersect(
                            prev_deep_re,
                            prev_deep_im,
                            layer_out["deep_beams_re"],
                            layer_out["deep_beams_im"],
                        )
                        blend_re = 0.65 * layer_out["deep_beams_re"] + 0.35 * four_re
                        blend_im = 0.65 * layer_out["deep_beams_im"] + 0.35 * four_im
                        frob = (
                            (blend_re.square() + blend_im.square())
                            .sum(dim=(-2, -1), keepdim=True)
                            .clamp_min(DEFAULT_TRACE_EPS)
                        )
                        inv_f = torch.rsqrt(frob)
                        layer_out["deep_beams_re"] = blend_re * inv_f
                        layer_out["deep_beams_im"] = blend_im * inv_f
                    prev_deep_re = layer_out["deep_beams_re"]
                    prev_deep_im = layer_out["deep_beams_im"]

                deep_beams_re = layer_out["deep_beams_re"]
                deep_beams_im = layer_out["deep_beams_im"]
                ngram_beams_re = layer_out["ngram_beams_re"]
                ngram_beams_im = layer_out["ngram_beams_im"]
                id_routed_re = layer_out["id_routed_re"]
                id_routed_im = layer_out["id_routed_im"]
                gates = layer_out["gates"].to(amp_dtype)

                probe_re, probe_im = normalize_complex_ket(
                    torch.matmul(curr_ket_re, self.readout_probe_re)
                    - torch.matmul(curr_ket_im, self.readout_probe_im),
                    torch.matmul(curr_ket_re, self.readout_probe_im)
                    + torch.matmul(curr_ket_im, self.readout_probe_re),
                )
                act_deep_re, act_deep_im = apply_rank_beams_to_probe(
                    deep_beams_re, deep_beams_im, probe_re, probe_im
                )
                bath_re_c = uni_bath_re_snap.to(amp_dtype)
                bath_im_c = uni_bath_im_snap.to(amp_dtype)
                act_ng_re, act_ng_im, _ = diagonal_conjugated_bath_schur_action(
                    ngram_beams_re, ngram_beams_im, bath_re_c, bath_im_c, probe_re, probe_im
                )
                act_uni_re = torch.matmul(probe_re, bath_re_c.transpose(-2, -1)) - torch.matmul(
                    probe_im, bath_im_c.transpose(-2, -1)
                )
                act_uni_im = torch.matmul(probe_re, bath_im_c.transpose(-2, -1)) + torch.matmul(
                    probe_im, bath_re_c.transpose(-2, -1)
                )
                id_ov = (id_routed_re * probe_re).sum(dim=-1, keepdim=True)
                act_id_re, act_id_im = id_routed_re * id_ov, id_routed_im * id_ov

                w_id, w_deep, w_ngram, w_uni = (
                    gates[..., 0:1],
                    gates[..., 1:2],
                    gates[..., 2:3],
                    gates[..., 3:4],
                )
                mix_re_c = self.beam_readout_mix_re.to(amp_dtype)
                mix_im_c = self.beam_readout_mix_im.to(amp_dtype)
                beam_mix_re = (deep_beams_re * mix_re_c - deep_beams_im * mix_im_c).sum(dim=-2)
                beam_mix_im = (deep_beams_re * mix_im_c + deep_beams_im * mix_re_c).sum(dim=-2)

                state_re = (
                    w_id * act_id_re
                    + w_deep * (act_deep_re + 0.35 * beam_mix_re)
                    + w_ngram * (act_ng_re + 0.25 * hash_ket_re)
                    + (w_uni + DEFAULT_VACUUM_EPS) * act_uni_re
                )
                state_im = (
                    w_id * act_id_im
                    + w_deep * (act_deep_im + 0.35 * beam_mix_im)
                    + w_ngram * (act_ng_im + 0.25 * hash_ket_im)
                    + (w_uni + DEFAULT_VACUUM_EPS) * act_uni_im
                )
                readout_re, readout_im = normalize_complex_ket(
                    state_re + 0.50 * curr_ket_re, state_im + 0.50 * curr_ket_im
                )
                packed_readout = torch.cat([readout_re, readout_im], dim=-1).to(amp_dtype)
                vocab_re, vocab_im = normalize_complex_ket(
                    self.token_ket_re.to(amp_dtype), self.token_ket_im.to(amp_dtype)
                )
                packed_vocab_t = (
                    torch.cat([vocab_re, vocab_im], dim=-1).transpose(0, 1).contiguous().to(amp_dtype)
                )

                copy_log_odds, copy_conf = self.calib_copy_gate(
                    ket0_re, ket0_im, packed_readout, mask_pack.doc_same
                )
                id_probs = layer_out["id_attn_probs"].to(amp_dtype)
                combined_copy = (
                    copy_log_odds + DEFAULT_TOKEN_ID_COPY_SCALE * w_id * id_probs
                ).to(amp_dtype)
                copy_index = idx.unsqueeze(1).expand(bsz, seq_len, seq_len)
                uni_log_prior = torch.log(uni_probs_snap.clamp_min(1e-9)).view(1, 1, -1).to(amp_dtype)

                raw_lg = (
                    DEFAULT_BORN_READOUT_SCALE * torch.matmul(packed_readout, packed_vocab_t)
                    + (DEFAULT_UNIGRAM_PRIOR_WEIGHT + 0.50 * w_uni) * uni_log_prior
                ).scatter_add(dim=-1, index=copy_index, src=combined_copy)
                softcap = DEFAULT_W77_LOGIT_SOFTCAP
                logits = softcap * torch.tanh(raw_lg.float() * (1.0 / softcap))
                loss: Tensor | None = None
                if targets is not None:
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                if ingest_online:
                    self.ingest_online_token_batch(idx)
                return {
                    "logits": logits if return_logits else None,
                    "loss": loss,
                    "mean_copy_conf": float(copy_conf.detach().mean().item()),
                    "seq_attn_probs": layer_out["seq_attn_probs"].detach(),
                }
        finally:
            for lyr in self.layers:
                if isinstance(lyr, CalibratedCopySinkBatonLayer):
                    lyr.current_doc_same_mask = None
                    lyr.current_doc_start_pos = None


# ==============================================================================
# Rungs 2–5: Pruned Untied Quantum Baton Mixer + StiefelMuon + Carbon Routing
# ==============================================================================
def _apply_complex_head_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply complex unitary RoPE to `(B, H, T, 32)` (`16` complex pairs per head, `<= 8` args)."""
    half = x.size(-1) // 2
    x_re = x[..., :half]
    x_im = x[..., half:]
    c = cos.to(x.dtype)
    s = sin.to(x.dtype)
    out_re = x_re * c - x_im * s
    out_im = x_re * s + x_im * c
    return torch.cat([out_re, out_im], dim=-1)


class QuantumBatonMixerBlock(nn.Module):
    """Single Hilbert-highway layer with `<BOS>`-masked Quantum Attention + `XSA` + Interference MLP."""

    def __init__(self, cfg: Wave77MixerConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.real_dim = 2 * cfg.complex_dim
        self.head_real_dim = 2 * COMPLEX_HEAD_DIM
        if self.real_dim % self.head_real_dim != 0:
            raise ValueError(
                f"2 * complex_dim ({self.real_dim}) must be divisible by {self.head_real_dim}"
            )
        self.num_heads = self.real_dim // self.head_real_dim

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

        self.schur_gate: ComplementarySchurSubsystemGate | None = None
        if cfg.schur_mode != "none":
            self.schur_gate = ComplementarySchurSubsystemGate(
                real_dim=self.real_dim,
                use_complementary_split=(cfg.schur_mode == "complementary"),
            )

        self.mlp_up: nn.Linear | None = None
        self.mlp_phase_gate: nn.Linear | None = None
        self.mlp_down: nn.Linear | None = None
        if cfg.mixer_mode == "quantum_interference_mlp":
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
        """Run `<BOS>`-masked attention + optional Schur split + interference MLP (`<= 8` args)."""
        bsz, seq_len, d_real = x.shape
        lam = self.block_lambdas.to(x.dtype)
        x_in = lam[0] * x + lam[1] * x_0

        h_norm = F.rms_norm(x_in, (d_real,), eps=1.0e-6)
        q_raw, k_raw, v_raw = self.qkv_proj(h_norm).chunk(3, dim=-1)
        q = q_raw.view(bsz, seq_len, self.num_heads, self.head_real_dim).permute(0, 2, 1, 3)
        k = k_raw.view(bsz, seq_len, self.num_heads, self.head_real_dim).permute(0, 2, 1, 3)
        v = v_raw.view(bsz, seq_len, self.num_heads, self.head_real_dim).permute(0, 2, 1, 3)

        q = _apply_complex_head_rope(F.rms_norm(q, (self.head_real_dim,), eps=1.0e-6), cos_4d, sin_4d)
        k = _apply_complex_head_rope(F.rms_norm(k, (self.head_real_dim,), eps=1.0e-6), cos_4d, sin_4d)

        # <BOS>-safe 1-token partial key offset across the upper 8 real channels per head
        if seq_len > 1:
            sp = same_prev_tok.unsqueeze(1)
            k_tail_prev = k[:, :, :-1, 24:]
            k_tail_cur = k[:, :, 1:, 24:]
            k = torch.cat(
                [
                    k[..., :24],
                    torch.cat([k[:, :, :1, 24:], torch.where(sp, k_tail_prev, k_tail_cur)], dim=2),
                ],
                dim=-1,
            )

        if v_0 is None:
            v_next_0 = v
        else:
            v_next_0 = v_0
            assert self.v0_mix_logit is not None
            mix = torch.sigmoid(self.v0_mix_logit).to(v.dtype)
            v = torch.lerp(v, v_0, mix)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=causal_doc_mask,
            scale=DEFAULT_W77_ATTN_SCALE,
        )
        # Wave 44 Invariant #3 / XSA: Carrier / Self-Value Orthogonal Projection inside H_perp
        v_unit = F.normalize(v.float(), dim=-1, eps=1.0e-6).to(attn_out.dtype)
        echo_proj = (attn_out * v_unit).sum(dim=-1, keepdim=True) * v_unit
        xsa_gate = torch.sigmoid(self.xsa_alpha).to(attn_out.dtype)
        attn_clean = (attn_out - xsa_gate * echo_proj).permute(0, 2, 1, 3).reshape(bsz, seq_len, d_real)

        x_mid = x_in + self.out_proj(attn_clean)
        if self.schur_gate is not None:
            x_mid = x_mid + self.schur_gate(h_norm, F.rms_norm(attn_clean, (d_real,), eps=1.0e-6))

        if self.mlp_up is not None and self.mlp_phase_gate is not None and self.mlp_down is not None:
            m_in = F.rms_norm(x_mid, (d_real,), eps=1.0e-6)
            boson_amp = F.relu(self.mlp_up(m_in)).square()
            phase_mod = 1.0 + 0.25 * torch.tanh(self.mlp_phase_gate(m_in))
            x_out = x_mid + self.mlp_down(boson_amp * phase_mod)
        else:
            x_out = x_mid
        return x_out, v_next_0


class QuantumBatonMixerLM(nn.Module):
    """Flag-free strategy-configured Quantum Baton Mixer LM (`Wave 77` Rungs 2–5)."""

    def __init__(self, cfg: Wave77MixerConfig) -> None:
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
            [QuantumBatonMixerBlock(cfg, layer_idx=i) for i in range(self.num_layers)]
        )
        n_dec = self.num_layers // 2
        self.skip_weights = nn.Parameter(torch.zeros((max(1, n_dec),), dtype=torch.float32))

        self.causal_mtp_sweep: CausalVirtualMTPBackwardForwardMPSSweep | None = (
            CausalVirtualMTPBackwardForwardMPSSweep(real_dim=self.real_dim)
            if cfg.use_causal_mtp_mps
            else None
        )

        self.copy_gate = CalibratedPBosonAssociativeCopyGate(
            copy_ket_dim=self.complex_dim,
            ctx_dim=self.real_dim,
            boson_power=DEFAULT_COPY_BOSON_POWER,
            max_logit_gain=DEFAULT_COPY_MAX_LOGIT_GAIN,
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
        cos_4d = theta.cos().view(1, 1, seq_len, COMPLEX_HEAD_DIM)
        sin_4d = theta.sin().view(1, 1, seq_len, COMPLEX_HEAD_DIM)
        self._rope_cache[key] = (cos_4d, sin_4d)
        return cos_4d, sin_4d

    @torch.no_grad()
    def ingest_online_unigram_batch(self, idx: Tensor) -> None:
        """Update streaming online unigram probability distribution (`100%` speedrun-compliant)."""
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
        bsz, seq_len = idx.shape
        raw_emb = self.wte_ket(idx)
        same_prev = (idx[:, 1:] != FINEWEB_BOS_TOKEN_ID).unsqueeze(-1)
        if seq_len > 1:
            gate = (
                self.smear_lambda.to(raw_emb.dtype)
                * torch.sigmoid(self.smear_gate(raw_emb[:, 1:, : self.smear_in_dim]))
                * same_prev.to(raw_emb.dtype)
            )
            smear_emb = torch.cat(
                [raw_emb[:, :1], raw_emb[:, 1:] + gate * raw_emb[:, :-1]], dim=1
            )
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
                dec_i = l_idx - n_half
                skip_g = torch.sigmoid(self.skip_weights[dec_i]).to(x.dtype)
                x = x + skip_g * enc_stack.pop()
            x, v_0 = blk(
                x,
                x_0,
                cos_4d,
                sin_4d,
                causal_doc_mask,
                same_prev,
                v_0,
            )
            layer_acts.append(x)
            if self.num_layers >= 4 and l_idx < n_half:
                enc_stack.append(x)

        mtp_aux_loss: Tensor | None = None
        if self.causal_mtp_sweep is not None:
            fut_emb = self.wte_ket(targets_for_mtp).detach() if targets_for_mtp is not None else None
            delta_mtp, mtp_aux_loss = self.causal_mtp_sweep(x, fut_emb, mask_pack.doc_same)
            x = x + delta_mtp

        x_final = F.rms_norm(x, (self.real_dim,), eps=1.0e-6)
        copy_log_odds, copy_conf = self.copy_gate(
            ket0_re, ket0_im, x_final, mask_pack.doc_same
        )
        return x_final, copy_log_odds, copy_conf, mtp_aux_loss, layer_acts

    def compute_logits(
        self,
        idx: Tensor,
        x_final: Tensor,
        copy_log_odds: Tensor,
    ) -> Tensor:
        """Compute full `(B, T, V)` softcapped logits (`<= 8` args)."""
        bsz, seq_len = idx.shape
        uni_scale = F.softplus(self.uni_prior_logit.float()).to(x_final.dtype)
        uni_log_p = torch.log(self.online_uni_token_probs.detach().clamp_min(1.0e-9)).to(x_final.dtype)

        base_logits = F.linear(x_final, self.lm_head.weight) + uni_scale * uni_log_p.view(1, 1, -1)
        copy_index = idx.unsqueeze(1).expand(bsz, seq_len, seq_len)
        combined = base_logits.scatter_add(
            dim=-1, index=copy_index, src=copy_log_odds.to(base_logits.dtype)
        )
        softcap = self.cfg.logit_softcap
        return softcap * torch.tanh(combined.float() * (1.0 / softcap))

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
        softcap = self.cfg.logit_softcap
        inv_cap = 1.0 / softcap

        chunk_losses: list[Tensor] = []
        for b0 in range(0, bsz, chunk_b):
            b1 = min(bsz, b0 + chunk_b)
            lg_c = F.linear(x_final[b0:b1], self.lm_head.weight) + uni_scale * uni_log_p
            lg_c = lg_c.scatter_add(
                dim=-1, index=copy_index[b0:b1], src=copy_log_odds[b0:b1].to(lg_c.dtype)
            )
            capped_c = softcap * torch.tanh(lg_c.float() * inv_cap)
            ce_c = F.cross_entropy(capped_c.reshape(-1, capped_c.size(-1)), targets[b0:b1].reshape(-1))
            chunk_losses.append(ce_c * float(b1 - b0))

        total_ce = torch.stack(chunk_losses).sum() / float(bsz)
        total_loss = total_ce + mtp_aux_loss if mtp_aux_loss is not None else total_ce
        if ingest_online:
            self.ingest_online_unigram_batch(idx)
        return None, total_loss

    def collect_carbon_routing_telemetry(self) -> dict[str, Any]:
        """Return per-layer `mean_p_and`, `even_purity`, `odd_purity`, and `mps_p_and` (`<= 8` args)."""
        per_layer: list[dict[str, float]] = []
        for idx_l, blk in enumerate(self.blocks):
            if blk.schur_gate is not None:
                per_layer.append(
                    {
                        "layer": float(idx_l),
                        "mean_p_and": round(blk.schur_gate.last_mean_p_and, 5),
                        "even_purity": round(blk.schur_gate.last_even_purity, 5),
                        "odd_purity": round(blk.schur_gate.last_odd_purity, 5),
                    }
                )
        mean_p = (
            float(sum(d["mean_p_and"] for d in per_layer)) / float(len(per_layer))
            if per_layer
            else 0.0
        )
        mean_even = (
            float(sum(d["even_purity"] for d in per_layer)) / float(len(per_layer))
            if per_layer
            else 0.0
        )
        mean_odd = (
            float(sum(d["odd_purity"] for d in per_layer)) / float(len(per_layer))
            if per_layer
            else 0.0
        )
        return {
            "has_schur_gate": bool(len(per_layer) > 0),
            "mean_p_and": round(mean_p, 5),
            "mean_even_purity": round(mean_even, 5),
            "mean_odd_purity": round(mean_odd, 5),
            "mps_mean_p_and": (
                round(self.causal_mtp_sweep.last_mean_mps_p_and, 5)
                if self.causal_mtp_sweep is not None
                else 0.0
            ),
            "per_layer": per_layer,
        }


# ==============================================================================
# MuP + StiefelMuon Optimizer Builder & Rung Strategy Factory
# ==============================================================================
def build_wave77_mup_optimizers(
    model: QuantumBatonMixerLM,
    suite_mode: OptimizerSuiteMode = "mup_stiefel_muon",
    warmup_steps: int = 40,
) -> list[torch.optim.Optimizer]:
    """Build either `naive_sp` (`AdamW` fixed LR) or `mup_stiefel_muon` (`1/D` head LR) (`<= 8` args)."""
    if suite_mode == "adamw_sp":
        return [
            torch.optim.AdamW(
                model.parameters(),
                lr=NAIVE_SP_FIXED_ADAM_LR,
                betas=(0.90, 0.95),
                weight_decay=0.01,
            )
        ]

    muon_params: list[nn.Parameter] = []
    wte_params: list[nn.Parameter] = [model.wte_ket.weight]
    head_params: list[nn.Parameter] = [model.lm_head.weight]
    scalar_params: list[nn.Parameter] = []

    excluded_ids = {id(model.wte_ket.weight), id(model.lm_head.weight)}
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in excluded_ids:
            continue
        if p.ndim >= 2 and min(p.shape[-2], p.shape[-1]) >= 8 and "gate" not in name:
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


class Wave77DiagnosticAdapter(DiagnosticModelAdapter):
    """Polymorphic `DiagnosticModelAdapter` wrapping all `Wave 77` rungs (`<= 8` args)."""

    def __init__(
        self,
        key: str,
        label: str,
        model: nn.Module,
        optimizers: list[torch.optim.Optimizer],
        device: torch.device,
        embed_dim: int,
        num_layers: int,
        use_wsd_schedule: bool = False,
    ) -> None:
        super().__init__(
            key=key,
            label=label,
            model=model,
            device=device,
            embed_dim=embed_dim,
            num_layers=num_layers,
        )
        self.optimizers = optimizers
        self.use_wsd_schedule = bool(use_wsd_schedule)
        self.base_lrs: list[list[float]] = [
            [float(pg["lr"]) for pg in opt.param_groups] for opt in optimizers
        ]

    def _compute_lr_multiplier(self, step: int, total_steps: int) -> float:
        """Compute warmup + cosine (sprint) or Warmup-Stable-Decay (`WSD`, confirmation spin) LR mult."""
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
        self.model.train()
        lr_mult = self._compute_lr_multiplier(step, total_steps)
        for opt, base_group_lrs in zip(self.optimizers, self.base_lrs):
            for pg, base_lr in zip(opt.param_groups, base_group_lrs):
                pg["lr"] = base_lr * lr_mult
            opt.zero_grad(set_to_none=True)

        amp_on = x_batch.is_cuda
        if isinstance(self.model, CalibratedCopySinkBatonLM):
            out = self.model.forward_with_diagnostics(
                x_batch, targets=y_batch, ingest_online=True, return_logits=False
            )
            loss = out["loss"]
        else:
            assert isinstance(self.model, QuantumBatonMixerLM)
            with torch.amp.autocast(
                device_type=x_batch.device.type, dtype=torch.bfloat16, enabled=amp_on
            ):
                _, loss = self.model(
                    x_batch, targets=y_batch, ingest_online=True, return_logits=False
                )
        assert loss is not None
        loss.backward()
        if isinstance(self.model, CalibratedCopySinkBatonLM):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        for opt in self.optimizers:
            opt.step()
        return float(loss.detach().item())

    @torch.no_grad()
    def predict_log_probs_and_span(self, x_batch: Tensor) -> tuple[Tensor, Tensor]:
        """Return `(log_probs, logit_span)` of shapes `(B, T, 50257)` and `(B, T)` (`<= 8` args)."""
        was_train = self.model.training
        self.model.eval()
        amp_on = x_batch.is_cuda
        if isinstance(self.model, CalibratedCopySinkBatonLM):
            out = self.model.forward_with_diagnostics(
                x_batch, targets=None, ingest_online=False, return_logits=True
            )
            logits = out["logits"][..., :VALID_FINEWEB_VOCAB_SIZE].float()
        else:
            assert isinstance(self.model, QuantumBatonMixerLM)
            with torch.amp.autocast(
                device_type=x_batch.device.type, dtype=torch.bfloat16, enabled=amp_on
            ):
                logits_raw, _ = self.model(
                    x_batch, targets=None, ingest_online=False, return_logits=True
                )
            assert logits_raw is not None
            logits = logits_raw[..., :VALID_FINEWEB_VOCAB_SIZE].float()
        if was_train:
            self.model.train()
        span = logits.amax(dim=-1) - logits.median(dim=-1).values
        return F.log_softmax(logits, dim=-1), span


def build_wave77_rung_adapter(
    rung: Wave77RungKind,
    device: torch.device,
    seed: int = 1337,
    use_wsd_schedule: bool = False,
) -> Wave77DiagnosticAdapter:
    """Instantiate any `Wave77RungKind` cleanly via explicit strategy dispatch (`<= 8` args)."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    if rung == Wave77RungKind.R1_CALIB_COPY_SINK_2L_D64:
        cfg_r1 = FusedBF16FourCompartmentBatonConfig(
            vocab_size=VALID_FINEWEB_VOCAB_SIZE,
            hilbert_dim=64,
            num_layers=2,
            use_two_hop_triangle=True,
            use_bfloat16_amp=(device.type == "cuda"),
        )
        m_r1 = CalibratedCopySinkBatonLM(cfg_r1).to(device)
        opt_r1 = torch.optim.AdamW(
            m_r1.parameters(), lr=NAIVE_SP_FIXED_ADAM_LR, betas=(0.90, 0.95), weight_decay=0.01
        )
        return Wave77DiagnosticAdapter(
            key=rung.value,
            label="R1: CalibCopy[0..12.5] + <BOS>-Sink Baton (2L, d=64)",
            model=m_r1,
            optimizers=[opt_r1],
            device=device,
            embed_dim=128,
            num_layers=2,
            use_wsd_schedule=use_wsd_schedule,
        )

    spec_table: dict[Wave77RungKind, tuple[str, Wave77MixerConfig, OptimizerSuiteMode]] = {
        Wave77RungKind.R2_PRUNED_UNTIED_2L_D64: (
            "R2: Pruned Readout + Untied Head + Copy (2L, d=64, AdamW)",
            Wave77MixerConfig(complex_dim=64, num_layers=2, mixer_mode="none"),
            "adamw_sp",
        ),
        Wave77RungKind.R3_MUON_MIXER_2L_D64: (
            "R3: +StiefelMuon Quantum Interference Bank + XSA (2L, d=64)",
            Wave77MixerConfig(
                complex_dim=64, num_layers=2, mixer_mode="quantum_interference_mlp"
            ),
            "mup_stiefel_muon",
        ),
        Wave77RungKind.R4_MUP_2L_D128: (
            "R4a: MuP Scaled Quantum Baton Mixer (2L, d=128 / 256D)",
            Wave77MixerConfig(
                complex_dim=128, num_layers=2, mixer_mode="quantum_interference_mlp"
            ),
            "mup_stiefel_muon",
        ),
        Wave77RungKind.R4_MUP_2L_D256: (
            "R4b: MuP Scaled Quantum Baton Mixer (2L, d=256 / 512D)",
            Wave77MixerConfig(
                complex_dim=256, num_layers=2, mixer_mode="quantum_interference_mlp"
            ),
            "mup_stiefel_muon",
        ),
        Wave77RungKind.R4_MUP_4L_D256: (
            "R4c: MuP Scaled U-Net Quantum Baton Mixer (4L, d=256 / 512D)",
            Wave77MixerConfig(
                complex_dim=256, num_layers=4, mixer_mode="quantum_interference_mlp"
            ),
            "mup_stiefel_muon",
        ),
        Wave77RungKind.R5_CTRL_SCHUR_BASE_2L_D128: (
            "R5-Ctrl(a): Post-Selected Schur Beam Gate (2L, d=128)",
            Wave77MixerConfig(
                complex_dim=128,
                num_layers=2,
                mixer_mode="quantum_interference_mlp",
                schur_mode="post_selected",
            ),
            "mup_stiefel_muon",
        ),
        Wave77RungKind.R5A_CARBON_COMP_SCHUR_2L_D128: (
            "R5A(b): Carbon Complementary Schur Gate Only (2L, d=128)",
            Wave77MixerConfig(
                complex_dim=128,
                num_layers=2,
                mixer_mode="quantum_interference_mlp",
                schur_mode="complementary",
            ),
            "mup_stiefel_muon",
        ),
        Wave77RungKind.R5B_CARBON_COMP_SCHUR_MTP_2L_D128: (
            "R5B(c): Carbon Comp Schur + Causal K=4 MTP Bwd-Fwd MPS (2L, d=128)",
            Wave77MixerConfig(
                complex_dim=128,
                num_layers=2,
                mixer_mode="quantum_interference_mlp",
                schur_mode="complementary",
                use_causal_mtp_mps=True,
            ),
            "mup_stiefel_muon",
        ),
    }
    label, cfg, opt_mode = spec_table[rung]
    model = QuantumBatonMixerLM(cfg).to(device)
    opts = build_wave77_mup_optimizers(model, suite_mode=opt_mode, warmup_steps=40)
    return Wave77DiagnosticAdapter(
        key=rung.value,
        label=label,
        model=model,
        optimizers=opts,
        device=device,
        embed_dim=model.real_dim,
        num_layers=cfg.num_layers,
        use_wsd_schedule=use_wsd_schedule,
    )

