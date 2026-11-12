"""Wave 79 `<BOS>` FlexAttention Benchmark, MTP Horizon Inspector, & Copy/Schur Diagnostics (`EXPERIMENT_PLAN_099.md`).

Provenance & Mathematical Rationale
-----------------------------------
Houses the three standalone, reusable diagnostic and hardware-benchmark engines for `Wave 79`
(`EXPERIMENT_PLAN_099.md`), obeying Rules 1–8 of `docs/SCRIPT_BEST_PRACTICES.md`:

1. `benchmark_flex_attention_bos_docmask_parity_and_speed`: Backs `scripts/flex_attention_bos_docmask_speed_and_parity_bench.py`.
2. `inspect_mtp_horizon_weights_and_rollout_cosines`: Backs `scripts/mtp_horizon_site_weight_and_rollout_cosine_inspector.py`.
3. `audit_wave79_copy_and_schur_diagnostics`: Evaluates Zipf/multiplicity copy-gate nat savings (`m=0` vs `m>=1`)
   and In-Attention Complementary Schur leave-one-channel-out (`drop_odd`, `drop_even`) sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from .architecture_mechanistic_probes import evaluate_quick_ce_and_top1
from .architecture_xray_diagnostics import (
    FINEWEB_BOS_TOKEN_ID,
    VALID_FINEWEB_VOCAB_SIZE,
    load_fineweb_token_slice,
)
from .fused_bf16_four_compartment_baton_lab import DEFAULT_TRACE_EPS, normalize_complex_ket
from .wave77_baton_ladder_models import DEFAULT_W77_ATTN_SCALE
from .wave77_quantum_routing_primitives import (
    COMP_SCHUR_TRACE_EPS,
    COMPLEX_HEAD_DIM,
    build_bos_anchor_sink_doc_masks,
)
from .wave78_routing_and_copy_diagnostics import _summarize_copy_mask_bucket
from .wave78_unified_mup_carbon_baton import HorizonPooledCausalMTPMPSSweep
from .wave79_fused_schur_and_induction_baton import (
    FLEX_DOC_MASK_BLOCK_SIZE,
    FusedUnifiedSchurBatonLM,
    Wave79FusedAdapter,
    build_bos_doc_flex_block_mask,
)


@dataclass(frozen=True)
class FlexDocMaskBenchConfig:
    """Configuration for `benchmark_flex_attention_bos_docmask_parity_and_speed` (`<= 8` fields)."""

    seq_lens: tuple[int, ...] = (256, 512, 1024, 2048)
    num_heads: int = 16
    head_real_dim: int = 32
    total_batch_tokens: int = 16384
    warmup_iters: int = 4
    bench_iters: int = 15
    seed: int = 1337


def benchmark_flex_attention_bos_docmask_parity_and_speed(
    cfg: FlexDocMaskBenchConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Benchmark `<BOS>` `flex_attention` `BlockMask` vs dense boolean `SDPA` across `cfg.seq_lens`."""
    torch.manual_seed(cfg.seed)
    max_needed = max(cfg.seq_lens) * max(2, cfg.total_batch_tokens // min(cfg.seq_lens)) + 64
    val_tokens = load_fineweb_token_slice("fineweb_val_000000.bin", num_tokens=max_needed)

    compiled_flex = (
        torch.compile(flex_attention, dynamic=False) if device.type == "cuda" else flex_attention
    )
    rows: list[dict[str, Any]] = []

    for seq_t in cfg.seq_lens:
        bsz = max(2, cfg.total_batch_tokens // int(seq_t))
        idx = val_tokens[: bsz * seq_t].view(bsz, seq_t).to(device)
        mask_pack = build_bos_anchor_sink_doc_masks(idx)
        t_i = torch.arange(seq_t, device=device).unsqueeze(1)
        s_i = torch.arange(seq_t, device=device).unsqueeze(0)
        dense_mask = ((s_i <= t_i).unsqueeze(0) & mask_pack.doc_same).unsqueeze(1)
        active_pair_frac = float(dense_mask.float().mean().item())

        work_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        q0 = torch.randn(
            (bsz, cfg.num_heads, seq_t, cfg.head_real_dim), device=device, dtype=work_dtype
        )
        k0 = torch.randn_like(q0)
        v0 = torch.randn_like(q0)

        q_ref = q0.clone().requires_grad_(True)
        k_ref = k0.clone().requires_grad_(True)
        v_ref = v0.clone().requires_grad_(True)
        out_sdpa = F.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, attn_mask=dense_mask, scale=DEFAULT_W77_ATTN_SCALE
        )
        grad_seed = torch.randn_like(out_sdpa)
        (out_sdpa * grad_seed).sum().backward()

        bm = build_bos_doc_flex_block_mask(idx, block_size=FLEX_DOC_MASK_BLOCK_SIZE)
        q_fx = q0.clone().requires_grad_(True)
        k_fx = k0.clone().requires_grad_(True)
        v_fx = v0.clone().requires_grad_(True)
        out_flex = (
            compiled_flex(q_fx, k_fx, v_fx, block_mask=bm, scale=DEFAULT_W77_ATTN_SCALE)
            if bm is not None
            else F.scaled_dot_product_attention(
                q_fx, k_fx, v_fx, attn_mask=dense_mask, scale=DEFAULT_W77_ATTN_SCALE
            )
        )
        (out_flex * grad_seed).sum().backward()

        diff = (out_flex.detach().float() - out_sdpa.detach().float()).abs()
        out_cos = F.cosine_similarity(
            out_flex.detach().float().reshape(-1), out_sdpa.detach().float().reshape(-1), dim=0
        )
        assert q_fx.grad is not None and q_ref.grad is not None
        grad_cos = F.cosine_similarity(
            q_fx.grad.detach().float().reshape(-1), q_ref.grad.detach().float().reshape(-1), dim=0
        )

        for _ in range(cfg.warmup_iters):
            q_w = q0.detach().requires_grad_(True)
            F.scaled_dot_product_attention(
                q_w, k0, v0, attn_mask=dense_mask, scale=DEFAULT_W77_ATTN_SCALE
            ).sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        for _ in range(cfg.bench_iters):
            dm = (
                (s_i <= t_i).unsqueeze(0) & build_bos_anchor_sink_doc_masks(idx).doc_same
            ).unsqueeze(1)
            q_w = q0.detach().requires_grad_(True)
            k_w = k0.detach().requires_grad_(True)
            v_w = v0.detach().requires_grad_(True)
            y_w = F.scaled_dot_product_attention(
                q_w, k_w, v_w, attn_mask=dm, scale=DEFAULT_W77_ATTN_SCALE
            )
            (y_w * grad_seed).sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        sdpa_sec = max(1.0e-6, time.perf_counter() - t0)
        sdpa_vram_mb = (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0
        )

        for _ in range(cfg.warmup_iters):
            bm_w = build_bos_doc_flex_block_mask(idx, block_size=FLEX_DOC_MASK_BLOCK_SIZE)
            q_w = q0.detach().requires_grad_(True)
            y_w = (
                compiled_flex(q_w, k0, v0, block_mask=bm_w, scale=DEFAULT_W77_ATTN_SCALE)
                if bm_w is not None
                else F.scaled_dot_product_attention(
                    q_w, k0, v0, attn_mask=dense_mask, scale=DEFAULT_W77_ATTN_SCALE
                )
            )
            y_w.sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        t1 = time.perf_counter()
        bm_w = bm
        for idx_it in range(cfg.bench_iters):
            if idx_it % 4 == 0:
                bm_w = build_bos_doc_flex_block_mask(idx, block_size=FLEX_DOC_MASK_BLOCK_SIZE)
            q_w = q0.detach().requires_grad_(True)
            k_w = k0.detach().requires_grad_(True)
            v_w = v0.detach().requires_grad_(True)
            y_w = (
                compiled_flex(q_w, k_w, v_w, block_mask=bm_w, scale=DEFAULT_W77_ATTN_SCALE)
                if bm_w is not None
                else F.scaled_dot_product_attention(
                    q_w, k_w, v_w, attn_mask=dense_mask, scale=DEFAULT_W77_ATTN_SCALE
                )
            )
            (y_w * grad_seed).sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        flex_sec = max(1.0e-6, time.perf_counter() - t1)
        flex_vram_mb = (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0
        )

        tot_tok = float(bsz * seq_t * cfg.bench_iters)
        sdpa_tps = tot_tok / sdpa_sec
        flex_tps = tot_tok / flex_sec
        rows.append(
            {
                "seq_len": int(seq_t),
                "batch_size": int(bsz),
                "used_flex_block_mask": bool(bm is not None),
                "active_doc_causal_pair_frac": round(active_pair_frac, 5),
                "max_abs_diff": round(float(diff.max().item()), 6),
                "mean_abs_diff": round(float(diff.mean().item()), 6),
                "output_cosine_sim": round(float(out_cos.item()), 6),
                "grad_q_cosine_sim": round(float(grad_cos.item()), 6),
                "dense_sdpa_tok_per_s": round(sdpa_tps, 1),
                "flex_blockmask_tok_per_s": round(flex_tps, 1),
                "speedup_x": round(flex_tps / max(1.0, sdpa_tps), 4),
                "dense_sdpa_peak_vram_mb": round(sdpa_vram_mb, 2),
                "flex_blockmask_peak_vram_mb": round(flex_vram_mb, 2),
            }
        )
    return {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "block_size": FLEX_DOC_MASK_BLOCK_SIZE,
        "results": rows,
    }


@torch.no_grad()
def _evaluate_mtp_sweep_module_cosines(
    mtp_mod: HorizonPooledCausalMTPMPSSweep,
    x_pre_mtp: Tensor,
    future_embeds: Tensor,
    future_targets: Tensor,
) -> dict[str, Any]:
    """Compute per-horizon `w_k`, `p_mps_k`, and smoothed vs raw future cosines (`<= 8` args)."""
    bsz, seq_len, d_real = x_pre_mtp.shape
    g = mtp_mod.num_groups
    d_h = COMPLEX_HEAD_DIM
    site_real = 2 * d_h
    hk = mtp_mod.horizon_k

    virt_raw = mtp_mod.virtual_rollout_proj(x_pre_mtp).view(bsz, seq_len, hk, g, site_real)
    virt_f = F.normalize(virt_raw.float(), dim=-1, eps=1.0e-6)

    r_bwd = torch.full(
        (bsz, seq_len, g, mtp_mod.bond_chi),
        1.0 / math.sqrt(float(mtp_mod.bond_chi)),
        device=x_pre_mtp.device,
        dtype=torch.float32,
    )
    rho_bwd_diags: list[Tensor] = [torch.empty(0, device=x_pre_mtp.device)] * hk
    for k in range(hk - 1, -1, -1):
        site_drive = torch.matmul(virt_f[:, :, k], mtp_mod.bwd_site_to_bond[k].float())
        bond_prop = torch.matmul(r_bwd, mtp_mod.bwd_bond_transfer[k].float())
        r_bwd = F.normalize(bond_prop + site_drive, dim=-1, eps=1.0e-6)
        diag_raw = torch.matmul(r_bwd, mtp_mod.bwd_bond_to_diag[k].float()).square() + (
            0.10 / float(d_h)
        )
        rho_bwd_diags[k] = diag_raw / diag_raw.sum(dim=-1, keepdim=True).clamp_min(DEFAULT_TRACE_EPS)

    x_heads = x_pre_mtp.view(bsz, seq_len, g, site_real)
    l_fwd = F.normalize(mtp_mod.fwd_bond_init(x_heads).float(), dim=-1, eps=1.0e-6)
    smoothed_sites: list[Tensor] = []
    p_mps_per_k: list[float] = []

    for k in range(hk):
        phi_raw = virt_f[:, :, k] + torch.matmul(l_fwd, mtp_mod.fwd_bond_to_site[k].float())
        phi_re, phi_im = normalize_complex_ket(phi_raw[..., :d_h], phi_raw[..., d_h:])
        rho_fwd_diag = phi_re.square() + phi_im.square()
        bwd_diag_k = rho_bwd_diags[k]
        p_mps = (rho_fwd_diag * bwd_diag_k).sum(dim=-1, keepdim=True).clamp_min(COMP_SCHUR_TRACE_EPS)
        p_mps_per_k.append(round(float(p_mps.mean().item()), 5))
        bwd_amp = torch.sqrt(bwd_diag_k.clamp_min(DEFAULT_TRACE_EPS))
        inv_norm = torch.rsqrt(p_mps)
        smooth_site = torch.cat([phi_re * bwd_amp * inv_norm, phi_im * bwd_amp * inv_norm], dim=-1)
        smoothed_sites.append(smooth_site)
        if k + 1 < hk:
            l_fwd = F.normalize(
                l_fwd + torch.matmul(smooth_site, mtp_mod.fwd_site_to_bond[k].float()),
                dim=-1,
                eps=1.0e-6,
            )

    smooth_flat = F.normalize(
        torch.stack(smoothed_sites, dim=2).reshape(bsz, seq_len, hk, d_real), dim=-1, eps=1.0e-6
    )
    raw_flat = F.normalize(virt_f.reshape(bsz, seq_len, hk, d_real), dim=-1, eps=1.0e-6)
    persist_flat = F.normalize(x_pre_mtp.float(), dim=-1, eps=1.0e-6)
    tgt_flat = F.normalize(future_embeds.float(), dim=-1, eps=1.0e-6)

    bos_csum = future_targets.eq(FINEWEB_BOS_TOKEN_ID).to(torch.long).cumsum(dim=1)
    bos_pad = F.pad(bos_csum, (1, 0), value=0)
    h_weights = F.softmax(mtp_mod.horizon_pool_logits.detach().float(), dim=0).tolist()

    per_horizon: list[dict[str, float]] = []
    for k in range(hk):
        if seq_len <= k:
            break
        same_doc = (bos_pad[:, k + 1 :] == bos_pad[:, : seq_len - k]).to(torch.float32)
        denom = same_doc.sum().clamp_min(1.0)
        t_sl = tgt_flat[:, k:, :]
        sm_cos = ((smooth_flat[:, : seq_len - k, k, :] * t_sl).sum(dim=-1) * same_doc).sum() / denom
        rw_cos = ((raw_flat[:, : seq_len - k, k, :] * t_sl).sum(dim=-1) * same_doc).sum() / denom
        ps_cos = ((persist_flat[:, : seq_len - k, :] * t_sl).sum(dim=-1) * same_doc).sum() / denom
        per_horizon.append(
            {
                "horizon_step_ahead": float(k + 1),
                "pool_weight": round(float(h_weights[k]), 5),
                "mps_schur_p_and": p_mps_per_k[k],
                "smoothed_rollout_cosine": round(float(sm_cos.item()), 5),
                "unsmoothed_raw_cosine": round(float(rw_cos.item()), 5),
                "static_persistence_cosine": round(float(ps_cos.item()), 5),
                "mps_smoothing_cosine_gain": round(float((sm_cos - rw_cos).item()), 5),
                "valid_intra_doc_pairs": float(denom.item()),
            }
        )
    mtp_out_fro = float(mtp_mod.mtp_out_proj.weight.detach().float().norm().item())
    return {
        "horizon_k": int(hk),
        "bond_chi": int(mtp_mod.bond_chi),
        "mtp_out_proj_frobenius_norm": round(mtp_out_fro, 5),
        "horizon_pool_weights": [round(float(w), 5) for w in h_weights],
        "mean_mps_schur_p_and": round(sum(p_mps_per_k) / float(max(1, len(p_mps_per_k))), 5),
        "per_horizon": per_horizon,
    }


@torch.no_grad()
def inspect_mtp_horizon_weights_and_rollout_cosines(
    model: FusedUnifiedSchurBatonLM,
    x_eval: Tensor,
    y_eval: Tensor,
) -> dict[str, Any]:
    """Inspect `HorizonPooledCausalMTPMPSSweep` weights and future cosines on `(x_eval, y_eval)`."""
    if model.causal_mtp_sweep is None:
        return {"has_mtp_sweep": False, "per_horizon": []}
    was_tr = model.training
    model.eval()
    device = next(model.parameters()).device
    xb = x_eval[:8].to(device)
    yb = y_eval[:8].to(device)
    _, seq_len = xb.shape
    amp_on = xb.is_cuda

    with torch.amp.autocast(device_type=xb.device.type, dtype=torch.bfloat16, enabled=amp_on):
        x_0, _, _, same_prev = model._embed_and_smear(xb)
        mask_pack = build_bos_anchor_sink_doc_masks(xb)
        t_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        s_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        dense_doc_mask = ((s_idx <= t_idx).unsqueeze(0) & mask_pack.doc_same).unsqueeze(1)
        cos_4d, sin_4d = model._get_rope_cos_sin(seq_len, device)
        x_pre = model._run_unet_blocks_pure(x_0, cos_4d, sin_4d, None, dense_doc_mask, same_prev)
        fut_emb = model.wte_ket(yb)
        report = _evaluate_mtp_sweep_module_cosines(model.causal_mtp_sweep, x_pre, fut_emb, yb)

    if was_tr:
        model.train()
    report["has_mtp_sweep"] = True
    return report


@torch.no_grad()
def audit_wave79_copy_and_schur_diagnostics(
    adapter: Wave79FusedAdapter,
    x_eval: Tensor,
    y_eval: Tensor,
    decile_map: Tensor,
) -> dict[str, Any]:
    """Evaluate Zipf/multiplicity copy calibration (`m=0` vs `m>=1`) & Schur channel-drop sensitivity."""
    mod = adapter.fused_model
    was_tr = mod.training
    mod.eval()
    device = adapter.device
    cap = mod.cfg.logit_softcap
    inv_c = 1.0 / cap
    amp_on = device.type == "cuda"

    conf_list: list[Tensor] = []
    tgt_gain_list: list[Tensor] = []
    imp_gain_list: list[Tensor] = []
    nll_w_list: list[Tensor] = []
    nll_n_list: list[Tensor] = []
    ok_w_list: list[Tensor] = []
    mult_list: list[Tensor] = []
    dec_list: list[Tensor] = []

    for b0 in range(0, x_eval.size(0), 8):
        b1 = min(x_eval.size(0), b0 + 8)
        xb = x_eval[b0:b1].to(device)
        yb = y_eval[b0:b1].to(device)
        bsz, seq_len = xb.shape
        mask_pack = build_bos_anchor_sink_doc_masks(xb)
        t_i = torch.arange(seq_len, device=device).unsqueeze(1)
        s_i = torch.arange(seq_len, device=device).unsqueeze(0)
        strict_past = (s_i < t_i).unsqueeze(0) & mask_pack.doc_same
        same_tok = (yb.unsqueeze(2) == xb.unsqueeze(1)) & strict_past
        mult_list.append(same_tok.sum(dim=-1).cpu())
        dec_list.append(decile_map[yb.cpu()])

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_on):
            x_fin, copy_lo, copy_conf, _ = mod.encode_trunk_features(xb, force_eager=True)
            uni_s = F.softplus(mod.uni_prior_logit.float()).to(x_fin.dtype)
            uni_lp = (
                torch.log(mod.online_uni_token_probs.detach().clamp_min(1.0e-9))
                .to(x_fin.dtype)
                .view(1, 1, -1)
            )
            base_raw = (F.linear(x_fin, mod.lm_head.weight) + uni_s * uni_lp)[
                ..., :VALID_FINEWEB_VOCAB_SIZE
            ]
            copy_scatter = torch.zeros_like(base_raw, dtype=torch.float32)
            c_idx = xb.clamp(0, VALID_FINEWEB_VOCAB_SIZE - 1).unsqueeze(1).expand(bsz, seq_len, seq_len)
            copy_scatter.scatter_add_(dim=-1, index=c_idx, src=copy_lo.float())

            lg_no = cap * torch.tanh(base_raw.float() * inv_c)
            lg_with = cap * torch.tanh((base_raw.float() + copy_scatter) * inv_c)
            logp_no = F.log_softmax(lg_no, dim=-1)
            logp_with = F.log_softmax(lg_with, dim=-1)

            nll_no = -logp_no.gather(dim=-1, index=yb.unsqueeze(-1)).squeeze(-1)
            nll_with = -logp_with.gather(dim=-1, index=yb.unsqueeze(-1)).squeeze(-1)
            tgt_gain = copy_scatter.gather(dim=-1, index=yb.unsqueeze(-1)).squeeze(-1)
            imp_clone = copy_scatter.clone()
            imp_clone.scatter_(dim=-1, index=yb.unsqueeze(-1), value=0.0)
            imp_gain = imp_clone.amax(dim=-1)

            conf_list.append(copy_conf.float().cpu())
            tgt_gain_list.append(tgt_gain.cpu())
            imp_gain_list.append(imp_gain.cpu())
            nll_w_list.append(nll_with.cpu())
            nll_n_list.append(nll_no.cpu())
            ok_w_list.append((lg_with.argmax(dim=-1) == yb).cpu())

    conf_all = torch.cat(conf_list, dim=0).reshape(-1)
    tgt_all = torch.cat(tgt_gain_list, dim=0).reshape(-1)
    imp_all = torch.cat(imp_gain_list, dim=0).reshape(-1)
    nw_all = torch.cat(nll_w_list, dim=0).reshape(-1)
    nn_all = torch.cat(nll_n_list, dim=0).reshape(-1)
    ok_all = torch.cat(ok_w_list, dim=0).reshape(-1)
    mult_all = torch.cat(mult_list, dim=0).reshape(-1)
    dec_all = torch.cat(dec_list, dim=0).reshape(-1)
    n_tok = int(nw_all.numel())

    mult_buckets = {
        "m0_unseen_recall": _summarize_copy_mask_bucket(
            mult_all == 0, n_tok, conf_all, tgt_all, imp_all, nw_all, nn_all, ok_all
        ),
        "m1_single_mention": _summarize_copy_mask_bucket(
            mult_all == 1, n_tok, conf_all, tgt_all, imp_all, nw_all, nn_all, ok_all
        ),
        "m2_to_3_mentions": _summarize_copy_mask_bucket(
            (mult_all >= 2) & (mult_all <= 3),
            n_tok,
            conf_all,
            tgt_all,
            imp_all,
            nw_all,
            nn_all,
            ok_all,
        ),
        "m4plus_frequent": _summarize_copy_mask_bucket(
            mult_all >= 4, n_tok, conf_all, tgt_all, imp_all, nw_all, nn_all, ok_all
        ),
    }
    entity_buckets = {
        "d0_stopwords": _summarize_copy_mask_bucket(
            dec_all == 0, n_tok, conf_all, tgt_all, imp_all, nw_all, nn_all, ok_all
        ),
        "d6_to_d9_rare_entities_repeated": _summarize_copy_mask_bucket(
            (dec_all >= 6) & (mult_all >= 1),
            n_tok,
            conf_all,
            tgt_all,
            imp_all,
            nw_all,
            nn_all,
            ok_all,
        ),
    }

    schur_ablation: dict[str, float] = {}
    if mod.cfg.schur_mode == "in_attn_head_pair":
        ce_norm = float(nw_all.mean().item())
        mod.set_schur_eval_channel_mode("drop_odd")
        ce_drop_odd, _ = evaluate_quick_ce_and_top1(adapter, x_eval, y_eval)
        mod.set_schur_eval_channel_mode("drop_even")
        ce_drop_even, _ = evaluate_quick_ce_and_top1(adapter, x_eval, y_eval)
        mod.set_schur_eval_channel_mode("normal")
        schur_ablation = {
            "ce_normal": round(ce_norm, 5),
            "ce_drop_odd": round(ce_drop_odd, 5),
            "ce_drop_even": round(ce_drop_even, 5),
            "delta_ce_drop_odd": round(ce_drop_odd - ce_norm, 5),
            "delta_ce_drop_even": round(ce_drop_even - ce_norm, 5),
        }

    if was_tr:
        mod.train()
    return {
        "overall_net_copy_nat_savings": round(float((nn_all - nw_all).mean().item()), 5),
        "multiplicity_buckets": mult_buckets,
        "entity_buckets": entity_buckets,
        "in_attn_schur_channel_ablation": schur_ablation,
    }
