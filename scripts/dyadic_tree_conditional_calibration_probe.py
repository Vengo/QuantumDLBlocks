#!/usr/bin/env python3
"""Dyadic Tree Coarse-to-Fine Conditional Calibration Probe (Plan 040).

Promoted from `scripts/BACKLOG.md` (identified in Plan 039, implemented in Plan 040).
Audits multi-scale Wavelet-Cholesky tree calibration across `naive_kboson_born`,
`traceless_kboson_born`, `convex_traceless_born`, and `dyadic_hierarchical_born`:
1. Coarse-cluster NLL `-ln P_coarse(g(y) | x)` and Coarse Top-1 accuracy,
2. Conditional intra-cluster NLL `-ln P_fine(y | g(y), x)` and Conditional Top-1 accuracy,
3. Total NLL `-ln P(y | x)` (`= coarse_nll + conditional_fine_nll`) and Total Top-1 / Top-5 accuracy,
4. Exact probability normalization error `max_x |sum_v P(v | x) - 1|`,
5. Expected Calibration Error (`ECE`, 15-bin confidence calibration).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

READOUT_FAMILIES: tuple[str, ...] = (
    "naive_kboson_born",
    "traceless_kboson_born",
    "convex_traceless_born",
    "dyadic_hierarchical_born",
)


@dataclass(frozen=True)
class DyadicCalibrationProbeConfig:
    """Configuration for the multi-scale Wavelet-Cholesky tree calibration probe (`<= 8` args rule)."""

    num_classes: int = 1000
    num_coarse: int = 32
    embed_dim: int = 128
    num_samples: int = 512
    beta: float = 18.0
    k_copies: float = 20.0
    gamma: float = 1.0
    ece_bins: int = 15
    target_cos_min: float = 0.48
    target_cos_max: float = 0.76
    seed: int = 2040
    device: str = "cpu"
    checkpoint: str = ""
    json_out: str = ".cache/dyadic_tree_conditional_calibration_probe.json"


@dataclass(frozen=True)
class DyadicTreeSampleBundle:
    """Synthetic multi-scale Wavelet-Cholesky cosine scores and hierarchical class labels."""

    flat_cos: torch.Tensor
    coarse_cos: torch.Tensor
    fine_cos: torch.Tensor
    labels: torch.Tensor
    coarse_labels: torch.Tensor
    class_to_coarse: torch.Tensor
    coarse_member_mask: torch.Tensor


@torch.no_grad()
def build_dyadic_tree_samples(
    cfg: DyadicCalibrationProbeConfig,
    device: torch.device,
) -> DyadicTreeSampleBundle:
    """Synthesize `K`-class multi-scale Wavelet-Cholesky detectors and query states in `R^d`."""
    k_cls = int(cfg.num_classes)
    k_coarse = int(cfg.num_coarse)
    d = int(cfg.embed_dim)
    n_samp = int(cfg.num_samples)

    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
    class_to_coarse = torch.div(
        torch.arange(k_cls, device=device, dtype=torch.long) * k_coarse,
        k_cls,
        rounding_mode="floor",
    )
    coarse_member_mask = torch.zeros((k_coarse, k_cls), dtype=torch.bool, device=device)
    for c_idx in range(k_cls):
        g_idx = int(class_to_coarse[c_idx].item())
        coarse_member_mask[g_idx, c_idx] = True

    c_coarse = F.normalize(
        torch.randn((k_coarse, d), generator=gen, dtype=torch.float32, device=device),
        p=2.0,
        dim=-1,
    )
    w_fine = F.normalize(
        torch.randn((k_cls, d), generator=gen, dtype=torch.float32, device=device),
        p=2.0,
        dim=-1,
    )
    psi_flat = F.normalize(
        math.sqrt(0.50) * c_coarse.index_select(0, class_to_coarse) + math.sqrt(0.50) * w_fine,
        p=2.0,
        dim=-1,
    )

    labels = (torch.arange(n_samp, device=device, dtype=torch.long) * 11) % k_cls
    coarse_labels = class_to_coarse.index_select(0, labels)

    u_rand = torch.rand((n_samp,), generator=gen, dtype=torch.float32, device=device)
    c_target = float(cfg.target_cos_min) + (float(cfg.target_cos_max) - float(cfg.target_cos_min)) * u_rand

    target_vecs = psi_flat.index_select(0, labels)
    noise_raw = torch.randn((n_samp, d), generator=gen, dtype=torch.float32, device=device)
    noise_orth = F.normalize(
        noise_raw - (noise_raw * target_vecs).sum(dim=-1, keepdim=True) * target_vecs,
        p=2.0,
        dim=-1,
    )
    psi_queries = F.normalize(
        c_target.unsqueeze(-1) * target_vecs
        + torch.sqrt((1.0 - c_target * c_target).clamp_min(1e-6)).unsqueeze(-1) * noise_orth,
        p=2.0,
        dim=-1,
    )
    flat_cos = psi_queries @ psi_flat.t()

    target_coarse = c_coarse.index_select(0, coarse_labels)
    nc_raw = torch.randn((n_samp, d), generator=gen, dtype=torch.float32, device=device)
    nc_orth = F.normalize(
        nc_raw - (nc_raw * target_coarse).sum(dim=-1, keepdim=True) * target_coarse,
        p=2.0,
        dim=-1,
    )
    q_coarse = F.normalize(
        c_target.unsqueeze(-1) * target_coarse
        + torch.sqrt((1.0 - c_target * c_target).clamp_min(1e-6)).unsqueeze(-1) * nc_orth,
        p=2.0,
        dim=-1,
    )
    coarse_cos = q_coarse @ c_coarse.t()

    target_fine = w_fine.index_select(0, labels)
    nf_raw = torch.randn((n_samp, d), generator=gen, dtype=torch.float32, device=device)
    nf_orth = F.normalize(
        nf_raw - (nf_raw * target_fine).sum(dim=-1, keepdim=True) * target_fine,
        p=2.0,
        dim=-1,
    )
    q_fine = F.normalize(
        c_target.unsqueeze(-1) * target_fine
        + torch.sqrt((1.0 - c_target * c_target).clamp_min(1e-6)).unsqueeze(-1) * nf_orth,
        p=2.0,
        dim=-1,
    )
    fine_cos = q_fine @ w_fine.t()

    return DyadicTreeSampleBundle(
        flat_cos=flat_cos,
        coarse_cos=coarse_cos,
        fine_cos=fine_cos,
        labels=labels,
        coarse_labels=coarse_labels,
        class_to_coarse=class_to_coarse,
        coarse_member_mask=coarse_member_mask,
    )


@torch.no_grad()
def compute_traceless_observable(cos_scores: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute zero-trace convex-peak observable `O_hat = c_hat + gamma * (relu(c_hat)^2 - tau_+)`."""
    c_hat = cos_scores - cos_scores.mean(dim=-1, keepdim=True)
    pos_sq = F.relu(c_hat).square()
    tau_pos = pos_sq.mean(dim=-1, keepdim=True)
    return c_hat + float(gamma) * (pos_sq - tau_pos)


@torch.no_grad()
def compute_ece(probs: torch.Tensor, labels: torch.Tensor, num_bins: int = 15) -> float:
    """Compute Expected Calibration Error (`ECE`) across `num_bins` confidence intervals."""
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == labels).float()
    bin_boundaries = torch.linspace(0.0, 1.0, int(num_bins) + 1, device=probs.device)
    ece = torch.zeros((), dtype=torch.float32, device=probs.device)
    for b_idx in range(int(num_bins)):
        low = bin_boundaries[b_idx]
        high = bin_boundaries[b_idx + 1]
        if b_idx == 0:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences > low) & (confidences <= high)
        prop_in_bin = in_bin.float().mean()
        if float(prop_in_bin.item()) > 0.0:
            acc_in_bin = accuracies[in_bin].mean()
            conf_in_bin = confidences[in_bin].mean()
            ece = ece + torch.abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return float(ece.item())


@torch.no_grad()
def evaluate_tree_calibration(
    family: str,
    bundle: DyadicTreeSampleBundle,
    cfg: DyadicCalibrationProbeConfig,
) -> dict[str, Any]:
    """Audit coarse NLL/Top-1, conditional intra-cluster NLL/Top-1, total NLL/Top-1, sum error, and ECE."""
    n_samp, k_cls = bundle.flat_cos.shape
    row_idx = torch.arange(n_samp, device=bundle.flat_cos.device)
    r_tilde = min(float(cfg.beta) / float(cfg.k_copies), 0.95)
    k_val = float(cfg.k_copies)
    beta_val = float(cfg.beta)
    gamma = float(cfg.gamma)

    if family == "naive_kboson_born":
        flat_logits = k_val * torch.log(torch.clamp_min(1.0 + r_tilde * bundle.flat_cos, 1e-4))
        log_probs = F.log_softmax(flat_logits, dim=-1)
    elif family == "traceless_kboson_born":
        obs = compute_traceless_observable(bundle.flat_cos, gamma)
        flat_logits = k_val * torch.log(torch.clamp_min(1.0 + r_tilde * obs, 1e-4))
        log_probs = F.log_softmax(flat_logits, dim=-1)
    elif family == "convex_traceless_born":
        obs = compute_traceless_observable(bundle.flat_cos, gamma)
        purity_gain = 1.0 + 0.35 * obs.max(dim=-1, keepdim=True).values.clamp_min(0.0)
        flat_logits = (beta_val * purity_gain) * obs
        log_probs = F.log_softmax(flat_logits, dim=-1)
    elif family == "dyadic_hierarchical_born":
        obs_c = compute_traceless_observable(bundle.coarse_cos, gamma)
        obs_f = compute_traceless_observable(bundle.fine_cos, gamma)
        log_p_coarse = F.log_softmax(beta_val * obs_c, dim=-1)
        parent_mask = bundle.coarse_member_mask.index_select(0, bundle.class_to_coarse)
        fine_logits_all = beta_val * obs_f
        neg_inf = torch.full((n_samp, k_cls, k_cls), -1e9, device=obs_f.device, dtype=obs_f.dtype)
        masked_sibling_logits = torch.where(
            parent_mask.unsqueeze(0),
            fine_logits_all.unsqueeze(1).expand(n_samp, k_cls, k_cls),
            neg_inf,
        )
        log_z_sibling = torch.logsumexp(masked_sibling_logits, dim=-1)
        log_p_cond_all = fine_logits_all - log_z_sibling
        log_probs = log_p_coarse.index_select(1, bundle.class_to_coarse) + log_p_cond_all
    else:
        raise ValueError(f"Unknown readout family: {family}")

    probs = log_probs.exp()
    max_prob_sum_error = float((probs.sum(dim=-1) - 1.0).abs().max().item())

    # Marginal coarse-cluster probabilities P_coarse(g | x) = sum_{v in g} P(v | x)
    coarse_probs = probs @ bundle.coarse_member_mask.float().t()
    coarse_log_probs = torch.log(coarse_probs.clamp_min(1e-12))
    coarse_nll = float(F.nll_loss(coarse_log_probs, bundle.coarse_labels).item())
    coarse_top1_acc = float((coarse_probs.argmax(dim=-1) == bundle.coarse_labels).float().mean().item() * 100.0)

    # Conditional intra-cluster probabilities P_fine(v | g(y), x) inside true coarse cluster g(y)
    true_cluster_mask = bundle.coarse_member_mask.index_select(0, bundle.coarse_labels)
    cond_log_probs_true_g = torch.where(
        true_cluster_mask,
        log_probs - coarse_log_probs[row_idx, bundle.coarse_labels].unsqueeze(-1),
        torch.full_like(log_probs, -1e9),
    )
    cond_fine_nll = float((-cond_log_probs_true_g[row_idx, bundle.labels]).mean().item())
    cond_fine_top1_acc = float(
        (cond_log_probs_true_g.argmax(dim=-1) == bundle.labels).float().mean().item() * 100.0
    )

    total_nll = float(F.nll_loss(log_probs, bundle.labels).item())
    total_top1_acc = float((log_probs.argmax(dim=-1) == bundle.labels).float().mean().item() * 100.0)
    top5_k = min(5, k_cls)
    top5_preds = log_probs.topk(top5_k, dim=-1).indices
    total_top5_acc = float((top5_preds == bundle.labels.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0)
    ece_val = compute_ece(probs, bundle.labels, num_bins=cfg.ece_bins)
    chain_rule_residual = abs(total_nll - (coarse_nll + cond_fine_nll))

    return {
        "family": family,
        "coarse_nll_nats": round(coarse_nll, 4),
        "coarse_top1_acc_pct": round(coarse_top1_acc, 2),
        "conditional_fine_nll_nats": round(cond_fine_nll, 4),
        "conditional_fine_top1_acc_pct": round(cond_fine_top1_acc, 2),
        "total_nll_nats": round(total_nll, 4),
        "total_top1_acc_pct": round(total_top1_acc, 2),
        "total_top5_acc_pct": round(total_top5_acc, 2),
        "max_prob_sum_error": round(max_prob_sum_error, 8),
        "chain_rule_nll_residual": round(chain_rule_residual, 8),
        "ece": round(ece_val, 4),
    }


def run_dyadic_calibration_probe(cfg: DyadicCalibrationProbeConfig) -> dict[str, Any]:
    """Run the full multi-scale Wavelet-Cholesky tree conditional calibration probe."""
    device = torch.device(cfg.device)
    bundle = build_dyadic_tree_samples(cfg, device)
    rows: list[dict[str, Any]] = []
    by_family: dict[str, dict[str, Any]] = {}

    for fam in READOUT_FAMILIES:
        res = evaluate_tree_calibration(fam, bundle, cfg)
        rows.append(res)
        by_family[fam] = res

    print(
        f"\n### Multi-Scale Wavelet-Cholesky Dyadic Tree Conditional Calibration Probe "
        f"(`K={cfg.num_classes}`, `K_coarse={cfg.num_coarse}`, `d={cfg.embed_dim}`, `samples={cfg.num_samples}`)"
    )
    print(
        "| Readout Family | Coarse NLL | Coarse Top-1 | Cond Fine NLL | Cond Fine Top-1 | "
        "Total NLL | Total Top-1 | Max Sum Err | ECE |"
    )
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in rows:
        print(
            f"| `{r['family']}` | `{r['coarse_nll_nats']:.4f}` | `{r['coarse_top1_acc_pct']:.2f}%` | "
            f"`{r['conditional_fine_nll_nats']:.4f}` | `{r['conditional_fine_top1_acc_pct']:.2f}%` | "
            f"**`{r['total_nll_nats']:.4f}`** | `{r['total_top1_acc_pct']:.2f}%` | "
            f"`{r['max_prob_sum_error']:.2e}` | `{r['ece']:.4f}` |"
        )

    report = {
        "config": asdict(cfg),
        "summary": {
            "naive_total_nll_nats": by_family["naive_kboson_born"]["total_nll_nats"],
            "naive_ece": by_family["naive_kboson_born"]["ece"],
            "convex_total_nll_nats": by_family["convex_traceless_born"]["total_nll_nats"],
            "convex_ece": by_family["convex_traceless_born"]["ece"],
            "dyadic_total_nll_nats": by_family["dyadic_hierarchical_born"]["total_nll_nats"],
            "dyadic_coarse_nll_nats": by_family["dyadic_hierarchical_born"]["coarse_nll_nats"],
            "dyadic_conditional_fine_nll_nats": by_family["dyadic_hierarchical_born"]["conditional_fine_nll_nats"],
            "dyadic_ece": by_family["dyadic_hierarchical_born"]["ece"],
            "dyadic_max_prob_sum_error": by_family["dyadic_hierarchical_born"]["max_prob_sum_error"],
        },
        "families": by_family,
        "rows": rows,
    }
    if cfg.json_out:
        out_p = Path(cfg.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> DyadicCalibrationProbeConfig:
    """Parse CLI arguments into `DyadicCalibrationProbeConfig`."""
    p = argparse.ArgumentParser(
        description="Audit coarse-cluster vs. conditional intra-cluster NLL, Top-1 accuracy, and ECE."
    )
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--num-coarse", "--coarse-clusters", dest="num_coarse", type=int, default=32)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--beta", type=float, default=18.0)
    p.add_argument("--k-copies", type=float, default=20.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--ece-bins", type=int, default=15)
    p.add_argument("--seed", type=int, default=2040)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument(
        "--json-out",
        "--out-json",
        dest="json_out",
        type=str,
        default=".cache/dyadic_tree_conditional_calibration_probe.json",
    )
    ns = p.parse_args(argv)
    return DyadicCalibrationProbeConfig(
        num_classes=ns.num_classes,
        num_coarse=ns.num_coarse,
        embed_dim=ns.embed_dim,
        num_samples=ns.num_samples,
        beta=ns.beta,
        k_copies=ns.k_copies,
        gamma=ns.gamma,
        ece_bins=ns.ece_bins,
        seed=ns.seed,
        device=ns.device,
        checkpoint=ns.checkpoint,
        json_out=ns.json_out,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    run_dyadic_calibration_probe(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
