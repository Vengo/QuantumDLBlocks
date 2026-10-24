#!/usr/bin/env python3
"""Multi-Class Cross-Entropy Margin & Partition-Function Decomposer (Plan 040).

Promoted from `scripts/BACKLOG.md` (identified in Plan 039, implemented in Plan 040).
Decomposes multi-class Cross-Entropy:
    -ln P(y | x) = ln(1 + Z_pos_runnerup / exp(l_y) + Z_neg_tail / exp(l_y))
into:
1. `positive_runnerup_ce_tax_nats`: `ln(1 + Z_pos_runnerup / exp(l_y))` where
   `Z_pos_runnerup = sum_{v != y, c_hat_v > 0} exp(l_v)`.
2. `negative_tail_ce_tax_nats`: `ln(1 + Z_non_target / exp(l_y)) - ln(1 + Z_pos_runnerup / exp(l_y))`
   where `Z_non_target = Z_pos_runnerup + Z_neg_tail = sum_{v != y} exp(l_v)`.

Compares `naive_kboson_born`, `traceless_kboson_born`, `convex_traceless_born`,
and `dyadic_hierarchical_born` across `K in {100, 1000, 50257}`.
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
class MarginDecomposerConfig:
    """Configuration for the multi-class Cross-Entropy margin decomposer (`<= 8` args rule)."""

    num_classes_grid: tuple[int, ...] = (100, 1000, 50257)
    embed_dim: int = 128
    num_samples: int = 256
    beta: float = 18.0
    k_copies: float = 20.0
    gamma: float = 1.0
    target_cos_min: float = 0.50
    target_cos_max: float = 0.78
    seed: int = 2040
    device: str = "cpu"
    json_out: str = ".cache/multiclass_ce_margin_decomposer.json"


@dataclass(frozen=True)
class SyntheticScaleInputs:
    """Precomputed cosines and labels for one class count `K`."""

    num_classes: int
    coarse_clusters: int
    fine_clusters: int
    flat_cos: torch.Tensor
    coarse_cos: torch.Tensor
    fine_cos: torch.Tensor
    labels: torch.Tensor


def _choose_dyadic_factors(num_classes: int) -> tuple[int, int]:
    """Select `(K_coarse, K_fine)` such that `K_coarse * K_fine >= num_classes`."""
    if num_classes <= 128:
        k_coarse = 16
    elif num_classes <= 2048:
        k_coarse = 32
    else:
        k_coarse = 256
    k_fine = math.ceil(num_classes / float(k_coarse))
    return k_coarse, k_fine


@torch.no_grad()
def build_scale_inputs(
    num_classes: int,
    cfg: MarginDecomposerConfig,
    device: torch.device,
) -> SyntheticScaleInputs:
    """Synthesize unit-norm query states and `K`-class Wavelet-Cholesky detectors in `R^d`."""
    k_cls = int(num_classes)
    d = int(cfg.embed_dim)
    n_samp = int(cfg.num_samples)
    k_coarse, k_fine = _choose_dyadic_factors(k_cls)
    n_dyadic = k_coarse * k_fine

    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed) + k_cls)
    c_coarse = F.normalize(
        torch.randn((k_coarse, d), generator=gen, dtype=torch.float32, device=device),
        p=2.0,
        dim=-1,
    )
    w_fine = F.normalize(
        torch.randn((n_dyadic, d), generator=gen, dtype=torch.float32, device=device),
        p=2.0,
        dim=-1,
    ).reshape(k_coarse, k_fine, d)

    psi_dyadic = F.normalize(
        math.sqrt(0.50) * c_coarse.unsqueeze(1) + math.sqrt(0.50) * w_fine,
        p=2.0,
        dim=-1,
    ).reshape(n_dyadic, d)
    psi_flat = psi_dyadic[:k_cls]

    labels = (torch.arange(n_samp, device=device, dtype=torch.long) * 13) % k_cls
    coarse_labels = torch.div(labels, k_fine, rounding_mode="floor")
    fine_labels = labels % k_fine

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

    nf_raw = (
        torch.randn((n_samp, k_fine), generator=gen, dtype=torch.float32, device=device)
        * (1.0 / math.sqrt(float(d)))
    )
    row_idx = torch.arange(n_samp, device=device)
    nf_raw[row_idx, fine_labels] = c_target
    fine_cos = nf_raw.unsqueeze(1).expand(n_samp, k_coarse, k_fine).clone()

    return SyntheticScaleInputs(
        num_classes=k_cls,
        coarse_clusters=k_coarse,
        fine_clusters=k_fine,
        flat_cos=flat_cos,
        coarse_cos=coarse_cos,
        fine_cos=fine_cos,
        labels=labels,
    )


@torch.no_grad()
def compute_traceless_observable(cos_scores: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute zero-trace convex-peak observable `O_hat = c_hat + gamma * (relu(c_hat)^2 - tau_+)`."""
    c_hat = cos_scores - cos_scores.mean(dim=-1, keepdim=True)
    pos_sq = F.relu(c_hat).square()
    tau_pos = pos_sq.mean(dim=-1, keepdim=True)
    return c_hat + float(gamma) * (pos_sq - tau_pos)


@torch.no_grad()
def compute_family_logits(
    family: str,
    inputs: SyntheticScaleInputs,
    cfg: MarginDecomposerConfig,
) -> torch.Tensor:
    """Compute class logits `(N, K)` for the specified readout family."""
    n_samp, k_cls = inputs.flat_cos.shape
    r_tilde = min(float(cfg.beta) / float(cfg.k_copies), 0.95)
    k_val = float(cfg.k_copies)
    beta_val = float(cfg.beta)
    gamma = float(cfg.gamma)

    if family == "naive_kboson_born":
        return k_val * torch.log(torch.clamp_min(1.0 + r_tilde * inputs.flat_cos, 1e-4))
    if family == "traceless_kboson_born":
        obs = compute_traceless_observable(inputs.flat_cos, gamma)
        return k_val * torch.log(torch.clamp_min(1.0 + r_tilde * obs, 1e-4))
    if family == "convex_traceless_born":
        obs = compute_traceless_observable(inputs.flat_cos, gamma)
        purity_gain = 1.0 + 0.35 * obs.max(dim=-1, keepdim=True).values.clamp_min(0.0)
        return (beta_val * purity_gain) * obs
    if family == "dyadic_hierarchical_born":
        obs_c = compute_traceless_observable(inputs.coarse_cos, gamma)
        obs_f = compute_traceless_observable(inputs.fine_cos, gamma)
        log_p_c = F.log_softmax(beta_val * obs_c, dim=-1)
        log_p_f = F.log_softmax(beta_val * obs_f, dim=-1)
        joint_log_p = (log_p_c.unsqueeze(-1) + log_p_f).reshape(n_samp, -1)[:, :k_cls]
        return F.log_softmax(joint_log_p, dim=-1)
    raise ValueError(f"Unknown readout family: {family}")


@torch.no_grad()
def decompose_family_ce(
    family: str,
    inputs: SyntheticScaleInputs,
    cfg: MarginDecomposerConfig,
) -> dict[str, Any]:
    """Decompose multi-class Cross-Entropy into positive runner-up vs. negative-cosine tail taxes."""
    logits = compute_family_logits(family, inputs, cfg)
    n_samp, k_cls = logits.shape
    row_idx = torch.arange(n_samp, device=logits.device)

    c_hat = inputs.flat_cos - inputs.flat_cos.mean(dim=-1, keepdim=True)
    target_logits = logits[row_idx, inputs.labels]

    non_target_mask = torch.ones_like(logits, dtype=torch.bool)
    non_target_mask[row_idx, inputs.labels] = False
    pos_runnerup_mask = non_target_mask & (c_hat > 0.0)
    neg_tail_mask = non_target_mask & (c_hat <= 0.0)

    neg_inf = torch.full_like(logits, -1e9)
    log_z_non_target = torch.logsumexp(torch.where(non_target_mask, logits, neg_inf), dim=-1)
    log_z_pos = torch.logsumexp(torch.where(pos_runnerup_mask, logits, neg_inf), dim=-1)
    log_z_neg = torch.logsumexp(torch.where(neg_tail_mask, logits, neg_inf), dim=-1)

    total_ce_per_sample = F.softplus(log_z_non_target - target_logits)
    pos_tax_per_sample = F.softplus(log_z_pos - target_logits)
    neg_tax_per_sample = total_ce_per_sample - pos_tax_per_sample

    z_non_target = (log_z_non_target - target_logits).exp()
    z_pos_ratio = (log_z_pos - target_logits).exp()
    z_neg_ratio = (log_z_neg - target_logits).exp()
    pos_share = (z_pos_ratio / (z_non_target + 1e-12)).mean().item()
    neg_share = (z_neg_ratio / (z_non_target + 1e-12)).mean().item()

    masked_logits = torch.where(non_target_mask, logits, neg_inf)
    runner_up_logits = masked_logits.max(dim=-1).values
    mean_margin = float((target_logits - runner_up_logits).mean().item())

    top1_acc = float((logits.argmax(dim=-1) == inputs.labels).float().mean().item() * 100.0)
    top5_k = min(5, k_cls)
    top5_preds = logits.topk(top5_k, dim=-1).indices
    top5_acc = float((top5_preds == inputs.labels.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0)

    total_ce = float(total_ce_per_sample.mean().item())
    pos_tax = float(pos_tax_per_sample.mean().item())
    neg_tax = float(neg_tax_per_sample.mean().item())
    decomp_err = abs(total_ce - (pos_tax + neg_tax))

    return {
        "num_classes": int(k_cls),
        "family": family,
        "total_ce_nats": round(total_ce, 4),
        "positive_runnerup_ce_tax_nats": round(pos_tax, 4),
        "negative_tail_ce_tax_nats": round(neg_tax, 4),
        "positive_runnerup_partition_share": round(float(pos_share), 4),
        "negative_tail_partition_share": round(float(neg_share), 4),
        "mean_target_to_runner_up_margin": round(mean_margin, 4),
        "top1_acc_pct": round(top1_acc, 2),
        "top5_acc_pct": round(top5_acc, 2),
        "decomposition_residual": round(float(decomp_err), 8),
    }


def run_margin_decomposer(cfg: MarginDecomposerConfig) -> dict[str, Any]:
    """Run the multi-class Cross-Entropy margin decomposer across all `K` in `cfg.num_classes_grid`."""
    device = torch.device(cfg.device)
    rows: list[dict[str, Any]] = []
    by_k: dict[str, dict[str, Any]] = {}

    for k_cls in cfg.num_classes_grid:
        inputs = build_scale_inputs(int(k_cls), cfg, device)
        k_key = str(int(k_cls))
        by_k[k_key] = {}
        for fam in READOUT_FAMILIES:
            res = decompose_family_ce(fam, inputs, cfg)
            rows.append(res)
            by_k[k_key][fam] = res

    print(
        f"\n### Multi-Class Cross-Entropy Margin & Partition Tax Decomposer "
        f"(`d={cfg.embed_dim}`, `samples={cfg.num_samples}`, `beta={cfg.beta}`, `k={cfg.k_copies}`, `gamma={cfg.gamma}`)"
    )
    print(
        "| Classes `K` | Readout Family | Total CE (nats) | Pos Runner-Up Tax | Neg Tail Tax | "
        "Neg Tail Share | Target Margin | Top-1 Acc | Top-5 Acc |"
    )
    print("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in rows:
        print(
            f"| `{r['num_classes']}` | `{r['family']}` | **`{r['total_ce_nats']:.4f}`** | "
            f"`{r['positive_runnerup_ce_tax_nats']:.4f}` | `{r['negative_tail_ce_tax_nats']:.4f}` | "
            f"`{r['negative_tail_partition_share']:.4f}` | `{r['mean_target_to_runner_up_margin']:.3f}` | "
            f"`{r['top1_acc_pct']:.2f}%` | `{r['top5_acc_pct']:.2f}%` |"
        )

    max_k_key = str(int(cfg.num_classes_grid[-1]))
    naive_max_k = by_k[max_k_key]["naive_kboson_born"]
    convex_max_k = by_k[max_k_key]["convex_traceless_born"]
    dyadic_max_k = by_k[max_k_key]["dyadic_hierarchical_born"]

    report = {
        "config": asdict(cfg),
        "summary": {
            "largest_k": int(cfg.num_classes_grid[-1]),
            "naive_total_ce_at_largest_k": naive_max_k["total_ce_nats"],
            "naive_neg_tail_tax_at_largest_k": naive_max_k["negative_tail_ce_tax_nats"],
            "convex_total_ce_at_largest_k": convex_max_k["total_ce_nats"],
            "convex_neg_tail_tax_at_largest_k": convex_max_k["negative_tail_ce_tax_nats"],
            "dyadic_total_ce_at_largest_k": dyadic_max_k["total_ce_nats"],
            "dyadic_neg_tail_tax_at_largest_k": dyadic_max_k["negative_tail_ce_tax_nats"],
        },
        "by_num_classes": by_k,
        "rows": rows,
    }
    if cfg.json_out:
        out_p = Path(cfg.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> MarginDecomposerConfig:
    """Parse CLI arguments into `MarginDecomposerConfig`."""
    p = argparse.ArgumentParser(
        description="Decompose multi-class Cross-Entropy into positive runner-up vs. negative tail taxes."
    )
    p.add_argument("--num-classes", type=str, default="100,1000,50257")
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--beta", type=float, default=18.0)
    p.add_argument("--k-copies", type=float, default=20.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2040)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--json-out", "--out-json", dest="json_out", type=str, default=".cache/multiclass_ce_margin_decomposer.json")
    ns = p.parse_args(argv)
    k_grid = tuple(int(x.strip()) for x in ns.num_classes.split(",") if x.strip())
    return MarginDecomposerConfig(
        num_classes_grid=k_grid,
        embed_dim=ns.embed_dim,
        num_samples=ns.num_samples,
        beta=ns.beta,
        k_copies=ns.k_copies,
        gamma=ns.gamma,
        seed=ns.seed,
        device=ns.device,
        json_out=ns.json_out,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    run_margin_decomposer(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
