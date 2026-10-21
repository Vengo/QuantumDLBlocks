"""Target-Rank Cross-Entropy Decomposition & Layerwise Training Dynamics Probe (Plan 025).

Implements Section 4.1 of EXPERIMENT_PLAN_025.md:
1. `RANK_BUCKETS`: Canonical 5-bucket target-rank partition (`rank_1`, `ranks_2_5`, `ranks_6_50`,
   `ranks_51_1000`, `ranks_gt_1000`).
2. `compute_calibration_rank_decomposition`: GPU-vectorized, chunked decomposition of held-out
   Validation Cross-Entropy across target-rank buckets, top-k candidate probability mass allocation,
   prediction Shannon entropy, logit standard deviation, and Expected Calibration Error (`ECE`).
3. `compute_layerwise_dynamics_profile`: Extracts per-layer angular velocity (in degrees),
   Shannon spectral effective rank R_sigma(h_l), participation ratio R_PR(h_l), Stiefel manifold
   isometry error, and embedding/readout vs. backbone (StiefelMuon) gradient norms.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import l2_normalize
from ..optim.muon import split_muon_adamw_params

RANK_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("rank_1", 1, 1),
    ("ranks_2_5", 2, 5),
    ("ranks_6_50", 6, 50),
    ("ranks_51_1000", 51, 1000),
    ("ranks_gt_1000", 1001, 1_000_000_000),
)


@torch.no_grad()
def compute_calibration_rank_decomposition(
    logits_2d: torch.Tensor,
    targets_1d: torch.Tensor,
    valid_vocab_size: int = 50257,
    num_ece_bins: int = 10,
    chunk_size: int = 2048,
) -> dict[str, Any]:
    """Decompose Cross-Entropy and probability mass across the 5 canonical target-rank buckets."""
    if logits_2d.ndim < 2:
        raise ValueError(f"logits_2d must have at least 2 dimensions, got shape {tuple(logits_2d.shape)}")

    v_full = int(logits_2d.size(-1))
    v_eff = min(max(2, int(valid_vocab_size)), v_full)
    exec_device = targets_1d.device if targets_1d.device.type == "cuda" else logits_2d.device
    logits_flat = logits_2d.reshape(-1, v_full)[:, :v_eff]
    targets_flat = targets_1d.reshape(-1).to(device=logits_flat.device, dtype=torch.int64)

    if logits_flat.size(0) != targets_flat.size(0):
        raise ValueError(
            f"Token count mismatch between logits_2d ({logits_flat.size(0)}) and targets_1d ({targets_flat.size(0)})"
        )

    valid_mask = (targets_flat >= 0) & (targets_flat < v_eff)
    if not bool(valid_mask.all().item()):
        logits_flat = logits_flat[valid_mask]
        targets_flat = targets_flat[valid_mask]

    n_total = int(targets_flat.numel())
    if n_total == 0:
        raise ValueError("No valid target tokens in [0, valid_vocab_size)")

    device = exec_device
    ranks_all = torch.empty(n_total, device=device, dtype=torch.int64)
    nlls_all = torch.empty(n_total, device=device, dtype=torch.float32)
    target_probs_all = torch.empty(n_total, device=device, dtype=torch.float32)
    top1_probs_all = torch.empty(n_total, device=device, dtype=torch.float32)
    top1_correct_all = torch.empty(n_total, device=device, dtype=torch.float32)

    sum_ranks_2_5_mass = 0.0
    sum_ranks_6_50_mass = 0.0
    sum_tail_gt_50_mass = 0.0
    sum_entropy_nats = 0.0
    sum_logit_std = 0.0

    k_top50 = min(50, v_eff)
    k_top5 = min(5, v_eff)
    step_chunk = max(64, int(chunk_size))

    for start in range(0, n_total, step_chunk):
        end = min(start + step_chunk, n_total)
        lg_c = logits_flat[start:end].to(device=device, dtype=torch.float32)
        y_c = targets_flat[start:end].to(device=device)

        log_probs = F.log_softmax(lg_c, dim=-1)
        probs = log_probs.exp()

        target_logit = lg_c.gather(1, y_c.unsqueeze(1))
        r_c = 1 + (lg_c > target_logit).sum(dim=-1)
        nll_c = -log_probs.gather(1, y_c.unsqueeze(1)).squeeze(1)
        tprob_c = probs.gather(1, y_c.unsqueeze(1)).squeeze(1)

        ranks_all[start:end] = r_c
        nlls_all[start:end] = nll_c
        target_probs_all[start:end] = tprob_c

        top_vals, top_idx = torch.topk(probs, k=k_top50, dim=-1)
        top1_p = top_vals[:, 0]
        top1_probs_all[start:end] = top1_p
        top1_correct_all[start:end] = (top_idx[:, 0] == y_c).float()

        if k_top5 > 1:
            r2_5_p = top_vals[:, 1:k_top5].sum(dim=-1)
        else:
            r2_5_p = torch.zeros_like(top1_p)

        if k_top50 > 5:
            r6_50_p = top_vals[:, 5:k_top50].sum(dim=-1)
        else:
            r6_50_p = torch.zeros_like(top1_p)

        tail_p = (1.0 - (top1_p + r2_5_p + r6_50_p)).clamp_min(0.0)

        sum_ranks_2_5_mass += float(r2_5_p.sum().item())
        sum_ranks_6_50_mass += float(r6_50_p.sum().item())
        sum_tail_gt_50_mass += float(tail_p.sum().item())

        ent_c = -(probs * log_probs).sum(dim=-1)
        sum_entropy_nats += float(ent_c.sum().item())
        sum_logit_std += float(lg_c.std(dim=-1).sum().item())

    total_ce = float(nlls_all.mean().item())
    total_ppl = float(math.exp(min(20.0, total_ce)))

    buckets_out: dict[str, dict[str, float | int]] = {}
    bucket_ce_sum = 0.0
    bucket_frac_sum = 0.0

    for b_name, r_lo, r_hi in RANK_BUCKETS:
        mask = (ranks_all >= r_lo) & (ranks_all <= r_hi)
        cnt = int(mask.sum().item())
        frac = float(cnt) / float(n_total)
        if cnt > 0:
            mean_nll = float(nlls_all[mask].mean().item())
            ce_contrib = float(nlls_all[mask].sum().item()) / float(n_total)
            mean_tprob = float(target_probs_all[mask].mean().item())
        else:
            mean_nll = 0.0
            ce_contrib = 0.0
            mean_tprob = 0.0

        bucket_ce_sum += ce_contrib
        bucket_frac_sum += frac
        buckets_out[b_name] = {
            "count": cnt,
            "fraction": frac,
            "mean_nll": mean_nll,
            "ce_contribution": ce_contrib,
            "mean_target_prob": mean_tprob,
        }

    top1_acc_pct = 100.0 * float((ranks_all == 1).float().mean().item())
    top5_acc_pct = 100.0 * float((ranks_all <= 5).float().mean().item())
    top50_acc_pct = 100.0 * float((ranks_all <= 50).float().mean().item())
    top5_minus_top1_gap_pct = top5_acc_pct - top1_acc_pct

    mean_top1_prob_mass = float(top1_probs_all.mean().item())
    mean_ranks_2_5_prob_mass = sum_ranks_2_5_mass / float(n_total)
    mean_ranks_6_50_prob_mass = sum_ranks_6_50_mass / float(n_total)
    mean_tail_gt_50_prob_mass = sum_tail_gt_50_mass / float(n_total)

    # 10-bin Expected Calibration Error (ECE) on Top-1 confidence vs accuracy
    n_bins = max(2, int(num_ece_bins))
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=device, dtype=torch.float32)
    ece_accum = 0.0
    for b_idx in range(n_bins):
        lo = boundaries[b_idx]
        hi = boundaries[b_idx + 1]
        if b_idx == 0:
            b_mask = (top1_probs_all >= lo) & (top1_probs_all <= hi)
        else:
            b_mask = (top1_probs_all > lo) & (top1_probs_all <= hi)
        if bool(b_mask.any().item()):
            prop = float(b_mask.float().mean().item())
            acc_bin = float(top1_correct_all[b_mask].mean().item())
            conf_bin = float(top1_probs_all[b_mask].mean().item())
            ece_accum += prop * abs(acc_bin - conf_bin)

    return {
        "num_tokens": n_total,
        "valid_vocab_size": v_eff,
        "total_ce": total_ce,
        "total_ppl": total_ppl,
        "bucket_ce_sum": bucket_ce_sum,
        "bucket_fraction_sum": bucket_frac_sum,
        "ce_decomposition_error": abs(bucket_ce_sum - total_ce),
        "top1_acc_pct": top1_acc_pct,
        "top5_acc_pct": top5_acc_pct,
        "top50_acc_pct": top50_acc_pct,
        "top5_minus_top1_gap_pct": top5_minus_top1_gap_pct,
        "median_rank": float(ranks_all.float().median().item()),
        "mean_rank": float(ranks_all.float().mean().item()),
        "mean_top1_prob_mass": mean_top1_prob_mass,
        "mean_ranks_2_5_prob_mass": mean_ranks_2_5_prob_mass,
        "mean_ranks_6_50_prob_mass": mean_ranks_6_50_prob_mass,
        "mean_tail_gt_50_prob_mass": mean_tail_gt_50_prob_mass,
        "prob_mass_sum": (
            mean_top1_prob_mass
            + mean_ranks_2_5_prob_mass
            + mean_ranks_6_50_prob_mass
            + mean_tail_gt_50_prob_mass
        ),
        "mean_prediction_entropy_nats": sum_entropy_nats / float(n_total),
        "logit_std": sum_logit_std / float(n_total),
        "ece": float(ece_accum),
        "buckets": buckets_out,
    }


def _compute_matrix_spectral_ranks(states_3d: torch.Tensor) -> tuple[float, float]:
    """Compute Shannon spectral effective rank R_sigma(h) and participation ratio R_PR(h)."""
    with torch.no_grad():
        x = states_3d.detach().reshape(-1, states_3d.size(-1)).float()
        if x.size(0) > 512:
            x = x[:512]
        x_centered = x - x.mean(dim=0, keepdim=True)
        if float(x_centered.norm().item()) < 1e-7:
            x_centered = x
        s = torch.linalg.svdvals(x_centered)
        s_pos = s[s > 1e-7]
        if s_pos.numel() == 0:
            return 1.0, 1.0
        p = s_pos / s_pos.sum()
        ent = -(p * torch.log(p + 1e-12)).sum().item()
        r_eff = float(math.exp(ent))
        s2 = s_pos.square()
        r_pr = float((s2.sum().square() / (s2.square().sum() + 1e-12)).item())
        return r_eff, r_pr


@torch.no_grad()
def _compute_stiefel_iso_error(model: nn.Module) -> float:
    """Compute mean Frobenius orthogonality deviation ||W^T W - I||_F / sqrt(d) across Stiefel matrices."""
    errs: list[float] = []
    for name, p in model.named_parameters():
        if p.ndim == 3 and "higher_interaction" in name:
            w = p.detach().float()
            m_dim = w.size(-1)
            gram = torch.bmm(w.transpose(-2, -1), w)
            eye = torch.eye(m_dim, device=w.device, dtype=torch.float32).unsqueeze(0).expand_as(gram)
            err = float((torch.linalg.norm(gram - eye, dim=(-2, -1)) / math.sqrt(float(m_dim))).mean().item())
            errs.append(err)
        elif p.ndim == 2 and p.size(0) == p.size(1) and p.size(0) >= 32:
            if any(k in name for k in ("q_proj", "k_proj", "v_proj", "out_proj")):
                w = p.detach().float()
                d = w.size(0)
                gram = w.T @ w
                eye = torch.eye(d, device=w.device, dtype=torch.float32)
                err = float(torch.norm(gram - eye, p="fro").item() / math.sqrt(float(d)))
                errs.append(err)
                if len(errs) >= 4:
                    break
    return float(sum(errs) / max(1, len(errs))) if errs else 0.0


def compute_layerwise_dynamics_profile(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    valid_vocab_size: int = 50257,
) -> dict[str, Any]:
    """Extract per-layer angular velocity, spectral effective rank, participation ratio, and gradient norms."""
    was_training = model.training
    model.eval()

    use_cuda_amp = x.device.type == "cuda"
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda_amp):
        if hasattr(model, "forward_features"):
            final_state, layer_states = model.forward_features(x, return_layer_states=True)
        else:
            out = model(x, return_layer_states=True)
            layer_states = list(out[-1]) if isinstance(out, tuple) and len(out) >= 3 else []
            final_state = layer_states[-1] if layer_states else out[0]

        h_0 = getattr(model, "_last_psi_0", None)
        if h_0 is None:
            h_0 = getattr(model, "_last_x_0", None)
        if h_0 is None and hasattr(model, "wte"):
            h_0 = model.wte(x)
        if h_0 is None and layer_states:
            h_0 = layer_states[0]

        angular_velocities_deg: list[float] = []
        effective_ranks: list[float] = []
        participation_ratios: list[float] = []

        prev_h = h_0 if h_0 is not None else final_state
        for h_l in layer_states:
            u_prev = l2_normalize(prev_h.detach().float(), dim=-1, eps=1e-8)
            u_curr = l2_normalize(h_l.detach().float(), dim=-1, eps=1e-8)
            cos_sim = (u_prev * u_curr).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            theta_deg = float((torch.acos(cos_sim) * (180.0 / math.pi)).mean().item())
            r_eff, r_pr = _compute_matrix_spectral_ranks(h_l)

            angular_velocities_deg.append(round(theta_deg, 4))
            effective_ranks.append(round(r_eff, 4))
            participation_ratios.append(round(r_pr, 4))
            prev_h = h_l

        stiefel_iso_err = _compute_stiefel_iso_error(model)
        n_layers = len(layer_states)
        del final_state, layer_states, prev_h

    # Micro-batch forward + backward probe to measure gradient norm distribution
    t_probe = min(256, x.size(1)) if x.ndim >= 2 else x.size(0)
    x_probe = x[: min(2, x.size(0)), :t_probe] if x.ndim >= 2 else x[:t_probe]
    y_probe = y[: min(2, y.size(0)), :t_probe] if y.ndim >= 2 else y[:t_probe]
    model.zero_grad(set_to_none=True)

    with torch.enable_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda_amp):
        if hasattr(model, "forward_features") and hasattr(model, "compute_logits"):
            h_f, _ = model.forward_features(x_probe, return_layer_states=False)
            logits_p = model.compute_logits(h_f)
        else:
            out_p = model(x_probe, y_probe)
            logits_p = out_p[0]
            h_f = logits_p

        v_eff = min(int(valid_vocab_size), int(logits_p.size(-1)))
        loss_p = F.cross_entropy(
            logits_p[..., :v_eff].reshape(-1, v_eff).float(),
            y_probe.reshape(-1).clamp(0, v_eff - 1),
        )
        # If lm_head is zero-initialized at Step 0 (ClassicalModdedGPT leaderboard_exact=True),
        # include a tiny auxiliary directional probe on h_f so Step-0 backbone sensitivity is also captured.
        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            if float(lm_head.weight.detach().abs().max().item()) == 0.0:
                d_idx = torch.arange(1, h_f.size(-1) + 1, device=h_f.device, dtype=torch.float32)
                probe_dir = torch.sin(d_idx) / math.sqrt(float(h_f.size(-1)))
                loss_p = loss_p + (h_f.float() * probe_dir).sum(dim=-1).mean()

        loss_p.backward()
        del logits_p, h_f, loss_p

    muon_params, adamw_params = split_muon_adamw_params(model)
    muon_sq = 0.0
    for p in muon_params:
        if p.grad is not None:
            muon_sq += float(p.grad.detach().float().norm(p=2).square().item())

    adamw_sq = 0.0
    for p in adamw_params:
        if p.grad is not None:
            adamw_sq += float(p.grad.detach().float().norm(p=2).square().item())

    grad_norm_backbone_muon = math.sqrt(muon_sq)
    grad_norm_embed_readout = math.sqrt(adamw_sq)
    embed_to_backbone_grad_ratio = grad_norm_embed_readout / max(1e-8, grad_norm_backbone_muon)

    model.zero_grad(set_to_none=True)
    model.train(was_training)

    return {
        "num_layers": n_layers,
        "layer_angular_velocity_deg": angular_velocities_deg,
        "mean_angular_velocity_deg": round(
            sum(angular_velocities_deg) / max(1, len(angular_velocities_deg)), 4
        ),
        "layer_effective_ranks": effective_ranks,
        "final_layer_effective_rank": effective_ranks[-1] if effective_ranks else 1.0,
        "layer_participation_ratios": participation_ratios,
        "final_layer_participation_ratio": participation_ratios[-1] if participation_ratios else 1.0,
        "grad_norm_embed_readout": round(grad_norm_embed_readout, 6),
        "grad_norm_backbone_muon": round(grad_norm_backbone_muon, 6),
        "embed_to_backbone_grad_ratio": round(embed_to_backbone_grad_ratio, 4),
        "stiefel_iso_err": round(stiefel_iso_err, 4),
    }
