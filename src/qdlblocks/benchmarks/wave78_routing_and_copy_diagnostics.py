r"""Wave 78 Copy-Gate Entity Calibration & Complementary Schur Purity Diagnostics (`EXPERIMENT_PLAN_098.md`).

Provenance & Mathematical Rationale
-----------------------------------
Promotes the two `Wave 77` (`EXPERIMENT_PLAN_097.md`) candidate diagnostics from `scripts/TOOLING_BACKLOG.md`
into reusable, unit-tested benchmark primitives obeying Rules 1–8 of `docs/SCRIPT_BEST_PRACTICES.md`:

1. **Copy-Gate Confidence & Rare-Entity Calibration Inspector (`inspect_copy_gate_confidence_and_entity_calibration`)**:
   - Decomposes per-token associative copy confidence $c_t = \sum_{s < t} w_{\text{copy}}(t, s)$,
     scattered target logit boost $\Delta z_{\text{copy}}(y_t)$, worst impostor false-copy boost
     $\max_{v \ne y_t} \Delta z_{\text{copy}}(v)$, and exact net cross-entropy savings
     $\Delta\text{CE}_{\text{copy}} = \text{CE}_{\text{no\_copy}} - \text{CE}_{\text{with\_copy}}$
     across Zipf rarity tiers (`D0`, `D1`, `D2–D5`, `D6–D9` rare named entities) and intra-document
     prior target multiplicity $m_t \in \{0, 1, 2\text{–}3, 4+\}$.

2. **Complementary Schur $p_{\text{and}}$ & Packet-Purity Layer Profiler (`profile_complementary_schur_pand_and_purity_layers`)**:
   - Profiles token-by-token coincidence probability $p_{\text{and}}(b, t, \ell, g)$ and exact $16 \times 16$
     subsystem purities $\operatorname{Tr}(\rho_{\text{AND}}^2), \operatorname{Tr}(\rho_{\text{comp}}^2)$
     against the `550M` Classical Leaderboard Teacher's predictive entropy and surprisal, plus
     leave-one-packet-out readout ablations ($\Delta\text{CE}_{\text{drop\_odd}}$, $\Delta\text{CE}_{\text{drop\_even}}$).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .architecture_xray_diagnostics import (
    TEACHER_ENTROPY_MEDIUM_MAX,
    TEACHER_ENTROPY_SHARP_MAX,
    VALID_FINEWEB_VOCAB_SIZE,
    ShortTrainConfig,
    TeacherReferenceProfile,
)
from .wave77_baton_ladder_models import (
    QuantumBatonMixerLM,
    Wave77DiagnosticAdapter,
    Wave77RungKind,
    build_wave77_rung_adapter,
)
from .wave77_evidence_ladder_runner import run_wave77_controlled_sprint
from .wave77_quantum_routing_primitives import (
    COMPLEX_HEAD_DIM,
    build_bos_anchor_sink_doc_masks,
    explicit_complementary_schur_matrix_pair,
)

# ==============================================================================
# Top-of-File Named Constants (Synchronized with EXPERIMENT_PLAN_098.md Section 3)
# ==============================================================================
DEFAULT_INSPECTOR_EVAL_TOKENS: int = 16384
COPY_MULTIPLICITY_BINS: tuple[int, int, int, int] = (0, 1, 3, 9999)
PEARSON_STD_EPS: float = 1.0e-8


def load_wave77_checkpoint_or_sprint_adapter(
    checkpoint_path: Path | None,
    rung: Wave77RungKind,
    sprint_steps: int,
    x_val: Tensor,
    y_val: Tensor,
    device: torch.device,
) -> Wave77DiagnosticAdapter:
    """Load a saved `.pt` `Wave 77` checkpoint or run a short FineWeb calibration sprint (`<= 8` args)."""
    if checkpoint_path is not None and checkpoint_path.exists():
        raw_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_rung = Wave77RungKind(str(raw_ckpt.get("rung", rung.value)))
        adapter = build_wave77_rung_adapter(ckpt_rung, device=device)
        adapter.model.load_state_dict(raw_ckpt["model_state_dict"], strict=True)
        return adapter

    if sprint_steps <= 0:
        return build_wave77_rung_adapter(rung, device=device)

    cfg = ShortTrainConfig(
        steps=int(sprint_steps),
        batch_size=32,
        seq_len=x_val.size(1),
        eval_interval=max(25, int(sprint_steps)),
        eval_tokens=int(x_val.numel()),
        wave_id="W78DIAG",
        plan_id="EXPERIMENT_PLAN_098",
    )
    adapter, _ = run_wave77_controlled_sprint(rung, cfg, x_val, y_val, device=device)
    return adapter


def _summarize_copy_mask_bucket(
    mask: Tensor,
    total_tokens: int,
    copy_conf: Tensor,
    tgt_gain: Tensor,
    imp_gain: Tensor,
    nll_with: Tensor,
    nll_no: Tensor,
    ok_with: Tensor,
) -> dict[str, float]:
    """Aggregate copy confidence, target/impostor logit gains, and nat savings on `mask` (`<= 8` args)."""
    cnt = int(mask.sum().item())
    if cnt == 0:
        return {
            "count": 0.0,
            "frac": 0.0,
            "mean_copy_conf": 0.0,
            "mean_target_logit_gain": 0.0,
            "mean_max_impostor_logit_gain": 0.0,
            "ce_with_copy": 0.0,
            "ce_without_copy": 0.0,
            "net_copy_nat_savings": 0.0,
            "top1_with_copy_pct": 0.0,
        }
    ce_w = float(nll_with[mask].mean().item())
    ce_n = float(nll_no[mask].mean().item())
    return {
        "count": float(cnt),
        "frac": round(float(cnt) / float(max(1, total_tokens)), 5),
        "mean_copy_conf": round(float(copy_conf[mask].mean().item()), 5),
        "mean_target_logit_gain": round(float(tgt_gain[mask].mean().item()), 5),
        "mean_max_impostor_logit_gain": round(float(imp_gain[mask].mean().item()), 5),
        "ce_with_copy": round(ce_w, 5),
        "ce_without_copy": round(ce_n, 5),
        "net_copy_nat_savings": round(ce_n - ce_w, 5),
        "top1_with_copy_pct": round(100.0 * float(ok_with[mask].float().mean().item()), 3),
    }


@torch.no_grad()
def inspect_copy_gate_confidence_and_entity_calibration(
    model: nn.Module,
    x_val: Tensor,
    y_val: Tensor,
    decile_map: Tensor,
    model_label: str = "quantum_baton_mixer",
    micro_batch: int = 8,
) -> dict[str, Any]:
    """Audit per-token copy confidence, logit gain, impostor noise, & nat savings (`<= 8` args)."""
    was_train = model.training
    model.eval()
    dev = next(model.parameters()).device
    amp_on = dev.type == "cuda"
    n_seq, seq_len = x_val.shape

    all_conf: list[Tensor] = []
    all_tgt_gain: list[Tensor] = []
    all_imp_gain: list[Tensor] = []
    all_nll_with: list[Tensor] = []
    all_nll_no: list[Tensor] = []
    all_ok_with: list[Tensor] = []
    all_mult: list[Tensor] = []

    t_idx = torch.arange(seq_len, device=dev).unsqueeze(1)
    s_idx = torch.arange(seq_len, device=dev).unsqueeze(0)
    strict_past = ((t_idx > s_idx) & (s_idx >= 1)).unsqueeze(0)

    for b0 in range(0, n_seq, micro_batch):
        b1 = min(n_seq, b0 + micro_batch)
        xb = x_val[b0:b1].to(dev)
        yb = y_val[b0:b1].to(dev)
        bsz = xb.size(0)

        mask_pack = build_bos_anchor_sink_doc_masks(xb)
        doc_prev = F.pad(mask_pack.doc_same[:, :, :-1], (1, 0), value=False)
        valid_copy_pos = strict_past & mask_pack.doc_same & doc_prev
        matches_y = (xb.unsqueeze(1) == yb.unsqueeze(2)) & valid_copy_pos
        mult_bt = matches_y.to(torch.long).sum(dim=-1)

        with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp_on):
            x_fin, copy_lo, copy_conf, _, _ = model.encode_trunk_features(xb)  # type: ignore[operator]
            uni_s = F.softplus(model.uni_prior_logit.float()).to(x_fin.dtype)  # type: ignore[attr-defined]
            uni_lp = (
                torch.log(model.online_uni_token_probs.detach().clamp_min(1.0e-9))  # type: ignore[attr-defined]
                .to(x_fin.dtype)
                .view(1, 1, -1)
            )
            base_lg = F.linear(x_fin, model.lm_head.weight) + uni_s * uni_lp  # type: ignore[attr-defined]

        copy_scatter = torch.zeros(
            (bsz, seq_len, VALID_FINEWEB_VOCAB_SIZE), device=dev, dtype=torch.float32
        )
        c_idx = xb.unsqueeze(1).expand(bsz, seq_len, seq_len)
        copy_scatter.scatter_add_(dim=-1, index=c_idx, src=copy_lo.float())

        tgt_gain = copy_scatter.gather(dim=-1, index=yb.unsqueeze(-1)).squeeze(-1)
        masked_imp = copy_scatter.clone()
        masked_imp.scatter_(dim=-1, index=yb.unsqueeze(-1), value=0.0)
        imp_gain = masked_imp.amax(dim=-1)

        cap = float(model.cfg.logit_softcap)  # type: ignore[attr-defined]
        inv_c = 1.0 / cap
        lg_no = cap * torch.tanh(base_lg.float() * inv_c)
        lg_with = cap * torch.tanh((base_lg.float() + copy_scatter) * inv_c)

        nll_no = F.cross_entropy(
            lg_no.reshape(-1, VALID_FINEWEB_VOCAB_SIZE), yb.reshape(-1), reduction="none"
        ).view(bsz, seq_len)
        nll_with = F.cross_entropy(
            lg_with.reshape(-1, VALID_FINEWEB_VOCAB_SIZE), yb.reshape(-1), reduction="none"
        ).view(bsz, seq_len)
        ok_with = lg_with.argmax(dim=-1) == yb

        all_conf.append(copy_conf.float().cpu())
        all_tgt_gain.append(tgt_gain.cpu())
        all_imp_gain.append(imp_gain.cpu())
        all_nll_with.append(nll_with.cpu())
        all_nll_no.append(nll_no.cpu())
        all_ok_with.append(ok_with.cpu())
        all_mult.append(mult_bt.cpu())

    if was_train:
        model.train()

    conf_flat = torch.cat(all_conf, dim=0).reshape(-1)
    tgt_flat = torch.cat(all_tgt_gain, dim=0).reshape(-1)
    imp_flat = torch.cat(all_imp_gain, dim=0).reshape(-1)
    nll_w_flat = torch.cat(all_nll_with, dim=0).reshape(-1)
    nll_n_flat = torch.cat(all_nll_no, dim=0).reshape(-1)
    ok_w_flat = torch.cat(all_ok_with, dim=0).reshape(-1)
    mult_flat = torch.cat(all_mult, dim=0).reshape(-1)

    y_cpu = y_val.cpu().reshape(-1).clamp(0, decile_map.numel() - 1)
    dec_flat = decile_map.cpu()[y_cpu]
    tot = int(y_cpu.numel())
    all_true = torch.ones_like(mult_flat, dtype=torch.bool)
    copyable = mult_flat >= 1

    tier_defs = {
        "D0_function_words_all": dec_flat == 0,
        "D0_function_words_copyable": (dec_flat == 0) & copyable,
        "D1_common_all": dec_flat == 1,
        "D2_D5_body_all": (dec_flat >= 2) & (dec_flat <= 5),
        "D2_D5_body_copyable": (dec_flat >= 2) & (dec_flat <= 5) & copyable,
        "D6_D9_rare_entities_all": dec_flat >= 6,
        "D6_D9_rare_entities_copyable": (dec_flat >= 6) & copyable,
    }
    b0, b1_m, b2_m, _ = COPY_MULTIPLICITY_BINS
    mult_defs = {
        "m_0_out_of_doc_recall": mult_flat == b0,
        "m_1_single_prior_mention": mult_flat == b1_m,
        "m_2_3_few_mentions": (mult_flat >= 2) & (mult_flat <= b2_m),
        "m_4plus_frequent_repeat": mult_flat >= (b2_m + 1),
    }

    return {
        "model_label": model_label,
        "eval_tokens": tot,
        "overall": _summarize_copy_mask_bucket(
            all_true, tot, conf_flat, tgt_flat, imp_flat, nll_w_flat, nll_n_flat, ok_w_flat
        ),
        "copyable_all_tiers": _summarize_copy_mask_bucket(
            copyable, tot, conf_flat, tgt_flat, imp_flat, nll_w_flat, nll_n_flat, ok_w_flat
        ),
        "by_zipf_and_entity_tier": {
            k: _summarize_copy_mask_bucket(
                m, tot, conf_flat, tgt_flat, imp_flat, nll_w_flat, nll_n_flat, ok_w_flat
            )
            for k, m in tier_defs.items()
        },
        "by_prior_doc_multiplicity": {
            k: _summarize_copy_mask_bucket(
                m, tot, conf_flat, tgt_flat, imp_flat, nll_w_flat, nll_n_flat, ok_w_flat
            )
            for k, m in mult_defs.items()
        },
    }


def _safe_pearson_corr(x: Tensor, y: Tensor) -> float:
    """Compute finite Pearson correlation between 1D float tensors `x` and `y` (`<= 8` args)."""
    xf = x.float() - x.float().mean()
    yf = y.float() - y.float().mean()
    denom = float((xf.norm() * yf.norm()).item())
    if denom < PEARSON_STD_EPS:
        return 0.0
    return round(float((xf * yf).sum().item() / denom), 5)


class _PacketMaskPreHook:
    """Top-level forward pre-hook masking even or odd packet channels (`Rule 5` zero closures)."""

    def __init__(self, mode_tag: str) -> None:
        self.mode_tag = mode_tag

    def __call__(self, _mod: nn.Module, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        joined = inputs[0].clone()
        half = joined.size(-1) // 2
        if self.mode_tag == "drop_even":
            joined[..., :half] = 0.0
        else:
            joined[..., half:] = 0.0
        return (joined,)


class _SchurLayerCaptureHook:
    """Top-level forward hook recording exact $16 \times 16$ $p_{\text{and}}$ and packet purities (`Rule 5`)."""

    def __init__(
        self,
        gate_mod: nn.Module,
        pand_sink: list[Tensor],
        even_sink: list[Tensor],
        odd_sink: list[Tensor],
    ) -> None:
        self.gate_mod = gate_mod
        self.pand_sink = pand_sink
        self.even_sink = even_sink
        self.odd_sink = odd_sink

    def __call__(self, _m: nn.Module, inputs: tuple[Tensor, ...], _out: Tensor) -> None:
        sa, sb = inputs[0], inputs[1]
        bsz, seq_t, _ = sa.shape
        u_a_re, u_a_im = self.gate_mod._pack_trace_unit_beams(  # type: ignore[operator]
            self.gate_mod.beam_a_proj(sa), bsz, seq_t  # type: ignore[operator]
        )
        u_b_re, u_b_im = self.gate_mod._pack_trace_unit_beams(  # type: ignore[operator]
            self.gate_mod.beam_b_proj(sb), bsz, seq_t  # type: ignore[operator]
        )
        ua = torch.complex(u_a_re.float(), u_a_im.float())
        ub = torch.complex(u_b_re.float(), u_b_im.float())
        rho_a = torch.matmul(ua.transpose(-2, -1), ua.conj())
        rho_b = torch.matmul(ub.transpose(-2, -1), ub.conj())
        rho_and, rho_comp, p_and = explicit_complementary_schur_matrix_pair(rho_a, rho_b)
        pur_e = (rho_and.real.square() + rho_and.imag.square()).sum(dim=(-2, -1)).mean(dim=-1)
        pur_o = (rho_comp.real.square() + rho_comp.imag.square()).sum(dim=(-2, -1)).mean(dim=-1)
        self.pand_sink.append(p_and.mean(dim=-1).cpu())
        self.even_sink.append(pur_e.cpu())
        self.odd_sink.append(pur_o.cpu())


@torch.no_grad()
def _eval_packet_ablation_ce_by_tier(
    model: nn.Module,
    x_val: Tensor,
    y_val: Tensor,
    decile_map: Tensor,
    ablate_mode: str,
    micro_batch: int = 8,
) -> dict[str, float]:
    """Evaluate overall and `Head / Body / Tail` CE under `full`, `drop_odd`, or `drop_even` (`<= 8` args)."""
    dev = next(model.parameters()).device
    amp_on = dev.type == "cuda"
    hooks = []

    if ablate_mode in ("drop_odd", "drop_even"):
        for blk in model.blocks:  # type: ignore[attr-defined]
            gate = getattr(blk, "schur_gate", None)
            if gate is not None:
                hooks.append(
                    gate.even_odd_out_proj.register_forward_pre_hook(_PacketMaskPreHook(ablate_mode))
                )

    nll_chunks: list[Tensor] = []
    try:
        for b0 in range(0, x_val.size(0), micro_batch):
            b1 = min(x_val.size(0), b0 + micro_batch)
            xb = x_val[b0:b1].to(dev)
            yb = y_val[b0:b1].to(dev)
            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp_on):
                logits_raw, _ = model(xb, targets=None, ingest_online=False, return_logits=True)
            assert logits_raw is not None
            lg = logits_raw[..., :VALID_FINEWEB_VOCAB_SIZE].float()
            nll = F.cross_entropy(
                lg.reshape(-1, VALID_FINEWEB_VOCAB_SIZE), yb.reshape(-1), reduction="none"
            )
            nll_chunks.append(nll.cpu())
    finally:
        for h in hooks:
            h.remove()

    nll_all = torch.cat(nll_chunks, dim=0)
    y_cpu = y_val.cpu().reshape(-1).clamp(0, decile_map.numel() - 1)
    dec = decile_map.cpu()[y_cpu]
    return {
        "overall_ce": round(float(nll_all.mean().item()), 5),
        "head_d0_d1_ce": round(float(nll_all[dec <= 1].mean().item()), 5),
        "body_d2_d5_ce": round(float(nll_all[(dec >= 2) & (dec <= 5)].mean().item()), 5),
        "tail_d6_d9_ce": round(float(nll_all[dec >= 6].mean().item()), 5),
    }


@torch.no_grad()
def profile_complementary_schur_pand_and_purity_layers(
    model: nn.Module,
    x_val: Tensor,
    y_val: Tensor,
    decile_map: Tensor,
    ref_profile: TeacherReferenceProfile,
    model_label: str = "carbon_comp_schur",
    micro_batch: int = 8,
) -> dict[str, Any]:
    """Profile token-level $p_{\text{and}}$, even/odd purities, & packet ablations (`<= 8` args)."""
    was_train = model.training
    model.eval()
    dev = next(model.parameters()).device
    amp_on = dev.type == "cuda"

    num_layers = len(model.blocks)  # type: ignore[attr-defined]
    layer_pand: list[list[Tensor]] = [[] for _ in range(num_layers)]
    layer_even_pur: list[list[Tensor]] = [[] for _ in range(num_layers)]
    layer_odd_pur: list[list[Tensor]] = [[] for _ in range(num_layers)]

    hooks = []
    for idx_l, blk in enumerate(model.blocks):  # type: ignore[attr-defined]
        gate = getattr(blk, "schur_gate", None)
        if gate is not None:
            hooks.append(
                gate.register_forward_hook(
                    _SchurLayerCaptureHook(
                        gate, layer_pand[idx_l], layer_even_pur[idx_l], layer_odd_pur[idx_l]
                    )
                )
            )

    try:
        for b0 in range(0, x_val.size(0), micro_batch):
            b1 = min(x_val.size(0), b0 + micro_batch)
            xb = x_val[b0:b1].to(dev)
            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp_on):
                model.encode_trunk_features(xb)  # type: ignore[operator]
    finally:
        for h in hooks:
            h.remove()

    ref_nll = ref_profile.token_nll.cpu().reshape(-1)
    ref_ent = ref_profile.token_entropy.cpu().reshape(-1)
    y_cpu = y_val.cpu().reshape(-1).clamp(0, decile_map.numel() - 1)
    dec = decile_map.cpu()[y_cpu]

    per_layer_stats: list[dict[str, Any]] = []
    for idx_l in range(num_layers):
        if not layer_pand[idx_l]:
            continue
        p_flat = torch.cat(layer_pand[idx_l], dim=0).reshape(-1)
        pe_flat = torch.cat(layer_even_pur[idx_l], dim=0).reshape(-1)
        po_flat = torch.cat(layer_odd_pur[idx_l], dim=0).reshape(-1)
        disagree = 1.0 - p_flat

        sharp_m = ref_ent < TEACHER_ENTROPY_SHARP_MAX
        med_m = (ref_ent >= TEACHER_ENTROPY_SHARP_MAX) & (ref_ent < TEACHER_ENTROPY_MEDIUM_MAX)
        high_m = ref_ent >= TEACHER_ENTROPY_MEDIUM_MAX

        per_layer_stats.append(
            {
                "layer": idx_l,
                "mean_p_and": round(float(p_flat.mean().item()), 5),
                "p10_p_and": round(float(torch.quantile(p_flat, 0.10).item()), 5),
                "p90_p_and": round(float(torch.quantile(p_flat, 0.90).item()), 5),
                "mean_disagreement_trace": round(float(disagree.mean().item()), 5),
                "mean_even_purity": round(float(pe_flat.mean().item()), 5),
                "mean_odd_purity": round(float(po_flat.mean().item()), 5),
                "corr_disagree_vs_teacher_nll": _safe_pearson_corr(disagree, ref_nll),
                "corr_disagree_vs_teacher_entropy": _safe_pearson_corr(disagree, ref_ent),
                "by_teacher_entropy": {
                    "sharp_lt_1p5": {
                        "mean_p_and": round(float(p_flat[sharp_m].mean().item()), 5),
                        "even_purity": round(float(pe_flat[sharp_m].mean().item()), 5),
                        "odd_purity": round(float(po_flat[sharp_m].mean().item()), 5),
                    },
                    "medium_1p5_to_4p0": {
                        "mean_p_and": round(float(p_flat[med_m].mean().item()), 5),
                        "even_purity": round(float(pe_flat[med_m].mean().item()), 5),
                        "odd_purity": round(float(po_flat[med_m].mean().item()), 5),
                    },
                    "high_ge_4p0": {
                        "mean_p_and": round(float(p_flat[high_m].mean().item()), 5),
                        "even_purity": round(float(pe_flat[high_m].mean().item()), 5),
                        "odd_purity": round(float(po_flat[high_m].mean().item()), 5),
                    },
                },
                "by_zipf_tier": {
                    "head_d0_d1_p_and": round(float(p_flat[dec <= 1].mean().item()), 5),
                    "body_d2_d5_p_and": round(float(p_flat[(dec >= 2) & (dec <= 5)].mean().item()), 5),
                    "tail_d6_d9_p_and": round(float(p_flat[dec >= 6].mean().item()), 5),
                },
            }
        )

    ce_full = _eval_packet_ablation_ce_by_tier(model, x_val, y_val, decile_map, "full", micro_batch)
    ce_drop_odd = _eval_packet_ablation_ce_by_tier(
        model, x_val, y_val, decile_map, "drop_odd", micro_batch
    )
    ce_drop_even = _eval_packet_ablation_ce_by_tier(
        model, x_val, y_val, decile_map, "drop_even", micro_batch
    )

    if was_train:
        model.train()

    return {
        "model_label": model_label,
        "complex_head_dim": COMPLEX_HEAD_DIM,
        "eval_tokens": int(y_val.numel()),
        "per_layer_stats": per_layer_stats,
        "packet_ablation": {
            "full": ce_full,
            "drop_odd_disagreement": ce_drop_odd,
            "drop_even_consensus": ce_drop_even,
            "delta_ce_drop_odd_overall": round(ce_drop_odd["overall_ce"] - ce_full["overall_ce"], 5),
            "delta_ce_drop_odd_tail_d6_d9": round(
                ce_drop_odd["tail_d6_d9_ce"] - ce_full["tail_d6_d9_ce"], 5
            ),
            "delta_ce_drop_even_overall": round(
                ce_drop_even["overall_ce"] - ce_full["overall_ce"], 5
            ),
        },
    }
