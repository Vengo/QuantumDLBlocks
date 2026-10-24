#!/usr/bin/env python3
"""Traceless Observable & Convex/Dyadic Born Rank-Bucket Calibration Probe (Plan 039).

Promoted from `scripts/BACKLOG.md` (identified in Plan 038, implemented in Plan 039).
Synthesizes unit-norm query states and `K`-class Wavelet-Cholesky detectors in `R^d` (`K > d`)
with planted target cosine signal `c_y in [0.45, 0.75]` and compares:
1. `naive_kboson_born`: `k * ln(1 + r_tilde * c_v)` (concave peak compression & high tail mass),
2. `traceless_kboson_born`: `k * ln(1 + r_tilde * O_hat_v)` with `O_hat_v = c_hat_v + gamma * (c_hat_{v,+}^2 - tau_+)`,
3. `convex_traceless_born`: `beta * [c_hat_v + gamma * (c_hat_{v,+}^2 - tau_+)]` (no concave `ln(1+x)` compression),
4. `dyadic_hierarchical_born`: Coarse-to-fine dyadic log-probability `ln P_coarse(g(v)|x) + ln P_fine(v|g(v), x)`
   across `K_coarse x K_fine = 32 x 32 = 1024 >= K`.
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

DEFAULT_GAMMA_GRID: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5)
READOUT_FAMILIES: tuple[str, ...] = (
    "naive_kboson_born",
    "traceless_kboson_born",
    "convex_traceless_born",
    "dyadic_hierarchical_born",
)


@dataclass(frozen=True)
class RankBucketProbeConfig:
    """Configuration for the multi-class Traceless / Convex / Dyadic Born rank-bucket probe."""

    num_classes: int = 1000
    embed_dim: int = 128
    num_samples: int = 512
    coarse_clusters: int = 32
    fine_clusters: int = 32
    gamma_grid: tuple[float, ...] = DEFAULT_GAMMA_GRID
    beta: float = 18.0
    k_copies: float = 20.0
    target_cos_min: float = 0.45
    target_cos_max: float = 0.75
    seed: int = 2039
    device: str = "cpu"
    json_out: str = ""


@dataclass(frozen=True)
class SyntheticPyramidBundle:
    """Synthetic dyadic Wavelet-Cholesky detectors and planted-signal unit query states."""

    flat_cos: torch.Tensor
    coarse_cos: torch.Tensor
    fine_cos: torch.Tensor
    labels: torch.Tensor
    coarse_labels: torch.Tensor
    fine_labels: torch.Tensor
    planted_target_cos: torch.Tensor


@torch.no_grad()
def _orthonormal_rows(num_rows: int, dim: int, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    """Construct `(num_rows, dim)` unit vectors that are exact orthonormal rows when `num_rows <= dim`."""
    raw = torch.randn((dim, max(num_rows, dim)), generator=gen, dtype=torch.float32).to(device)
    if num_rows <= dim:
        q_mat, _ = torch.linalg.qr(raw[:, :num_rows])
        return F.normalize(q_mat.t(), p=2.0, dim=-1)
    return F.normalize(raw.t()[:num_rows], p=2.0, dim=-1)


@torch.no_grad()
def build_synthetic_wavelet_pyramid_queries(
    cfg: RankBucketProbeConfig,
    device: torch.device,
) -> SyntheticPyramidBundle:
    """Synthesize `K`-class dyadic Wavelet-Cholesky detectors in `R^d` and `N` unit-norm queries."""
    k_cls = int(cfg.num_classes)
    d = int(cfg.embed_dim)
    n_samp = int(cfg.num_samples)
    k_coarse = int(cfg.coarse_clusters)
    k_fine = max(int(cfg.fine_clusters), math.ceil(k_cls / float(k_coarse)))
    n_dyadic = k_coarse * k_fine

    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
    c_coarse = _orthonormal_rows(k_coarse, d, gen, device)
    w_fine = torch.stack(
        [_orthonormal_rows(k_fine, d, gen, device) for _ in range(k_coarse)],
        dim=0,
    )

    coarse_weight = math.sqrt(0.50)
    fine_weight = math.sqrt(0.50)
    psi_dyadic = F.normalize(
        coarse_weight * c_coarse.unsqueeze(1) + fine_weight * w_fine,
        p=2.0,
        dim=-1,
    ).reshape(n_dyadic, d)
    psi_flat = psi_dyadic[:k_cls]

    labels = (torch.arange(n_samp, device=device, dtype=torch.long) * 7) % k_cls
    coarse_labels = torch.div(labels, k_fine, rounding_mode="floor")
    fine_labels = labels % k_fine

    u_rand = torch.rand((n_samp,), generator=gen, dtype=torch.float32).to(device)
    c_target = float(cfg.target_cos_min) + (float(cfg.target_cos_max) - float(cfg.target_cos_min)) * u_rand

    # Flat query states with exact planted cosine `c_target` to true class `y_i`
    target_vecs = psi_flat.index_select(0, labels)
    noise_raw = torch.randn((n_samp, d), generator=gen, dtype=torch.float32).to(device)
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

    # Coarse & fine query states at the same planted target cosine `c_target`
    target_coarse = c_coarse.index_select(0, coarse_labels)
    nc_raw = torch.randn((n_samp, d), generator=gen, dtype=torch.float32).to(device)
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

    nf_raw = torch.randn((n_samp, k_fine), generator=gen, dtype=torch.float32).to(device) * (1.0 / math.sqrt(float(d)))
    row_idx = torch.arange(n_samp, device=device)
    nf_raw[row_idx, fine_labels] = c_target
    fine_cos = nf_raw.unsqueeze(1).expand(n_samp, k_coarse, k_fine).clone()

    return SyntheticPyramidBundle(
        flat_cos=flat_cos,
        coarse_cos=coarse_cos,
        fine_cos=fine_cos,
        labels=labels,
        coarse_labels=coarse_labels,
        fine_labels=fine_labels,
        planted_target_cos=c_target,
    )


@torch.no_grad()
def compute_traceless_observable(cos_scores: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute zero-trace convex-peak observable `O_hat = c_hat + gamma * (relu(c_hat)^2 - tau_+)`."""
    c_hat = cos_scores - cos_scores.mean(dim=-1, keepdim=True)
    pos_sq = F.relu(c_hat).square()
    tau_pos = pos_sq.mean(dim=-1, keepdim=True)
    return c_hat + float(gamma) * (pos_sq - tau_pos)


@torch.no_grad()
def evaluate_readout_family(
    family: str,
    gamma: float,
    bundle: SyntheticPyramidBundle,
    cfg: RankBucketProbeConfig,
) -> dict[str, Any]:
    """Evaluate Cross-Entropy, Top-1, logit margin, and 5-bucket rank probability mass for one family."""
    n_samp, k_cls = bundle.flat_cos.shape
    row_idx = torch.arange(n_samp, device=bundle.flat_cos.device)
    r_tilde = min(float(cfg.beta) / float(cfg.k_copies), 0.95)
    k_val = float(cfg.k_copies)
    beta_val = float(cfg.beta)

    if family == "naive_kboson_born":
        logits = k_val * torch.log(torch.clamp_min(1.0 + r_tilde * bundle.flat_cos, 1e-4))
        slope_pos = (k_val * r_tilde) / (1.0 + r_tilde * 0.65)
        slope_neg = (k_val * r_tilde) / max(1e-4, 1.0 - r_tilde * 0.30)
    elif family == "traceless_kboson_born":
        obs = compute_traceless_observable(bundle.flat_cos, gamma)
        logits = k_val * torch.log(torch.clamp_min(1.0 + r_tilde * obs, 1e-4))
        slope_pos = ((k_val * r_tilde) / (1.0 + r_tilde * 0.65)) * (1.0 + 2.0 * float(gamma) * 0.65)
        slope_neg = (k_val * r_tilde) / max(1e-4, 1.0 - r_tilde * 0.30)
    elif family == "convex_traceless_born":
        obs = compute_traceless_observable(bundle.flat_cos, gamma)
        purity_gain = 1.0 + 0.35 * obs.max(dim=-1, keepdim=True).values.clamp_min(0.0)
        logits = (beta_val * purity_gain) * obs
        slope_pos = beta_val * (1.0 + 2.0 * float(gamma) * 0.65)
        slope_neg = beta_val
    elif family == "dyadic_hierarchical_born":
        obs_c = compute_traceless_observable(bundle.coarse_cos, gamma)
        obs_f = compute_traceless_observable(bundle.fine_cos, gamma)
        log_p_c = F.log_softmax(beta_val * obs_c, dim=-1)
        log_p_f = F.log_softmax(beta_val * obs_f, dim=-1)
        joint_log_p = (log_p_c.unsqueeze(-1) + log_p_f).reshape(n_samp, -1)[:, :k_cls]
        logits = F.log_softmax(joint_log_p, dim=-1)
        slope_pos = beta_val * (1.0 + 2.0 * float(gamma) * 0.65)
        slope_neg = beta_val
    else:
        raise ValueError(f"Unknown readout family: {family}")

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    ce_nats = float(F.nll_loss(log_probs, bundle.labels).item())
    top1_acc = float((logits.argmax(dim=-1) == bundle.labels).float().mean().item() * 100.0)
    top5_k = min(5, k_cls)
    top5_preds = logits.topk(top5_k, dim=-1).indices
    top5_acc = float((top5_preds == bundle.labels.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0)

    target_logits = logits[row_idx, bundle.labels]
    masked_logits = logits.clone()
    masked_logits[row_idx, bundle.labels] = -1e9
    runner_up_logits = masked_logits.max(dim=-1).values
    mean_margin = float((target_logits - runner_up_logits).mean().item())

    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    r1 = float(sorted_probs[:, :1].sum(dim=-1).mean().item())
    r2_5 = float(sorted_probs[:, 1:min(5, k_cls)].sum(dim=-1).mean().item())
    r6_20 = float(sorted_probs[:, min(5, k_cls):min(20, k_cls)].sum(dim=-1).mean().item())
    r21_100 = float(sorted_probs[:, min(20, k_cls):min(100, k_cls)].sum(dim=-1).mean().item())
    r_gt_100 = float(sorted_probs[:, min(100, k_cls):].sum(dim=-1).mean().item()) if k_cls > 100 else 0.0

    return {
        "family": family,
        "gamma": float(gamma),
        "cross_entropy_nats": round(ce_nats, 4),
        "top1_acc_pct": round(top1_acc, 2),
        "top5_acc_pct": round(top5_acc, 2),
        "mean_target_to_runner_up_margin": round(mean_margin, 4),
        "slope_ratio_peak_to_neg_tail": round(float(slope_pos / max(slope_neg, 1e-6)), 4),
        "rank_1_prob_mass": round(r1, 4),
        "ranks_2_to_5_prob_mass": round(r2_5, 4),
        "ranks_6_to_20_prob_mass": round(r6_20, 4),
        "ranks_21_to_100_prob_mass": round(r21_100, 4),
        "ranks_gt_100_prob_mass": round(r_gt_100, 4),
    }


def run_rank_bucket_probe(cfg: RankBucketProbeConfig) -> dict[str, Any]:
    """Run the full rank-bucket calibration probe across all readout families and `gamma` values."""
    device = torch.device(cfg.device)
    bundle = build_synthetic_wavelet_pyramid_queries(cfg, device)
    rows: list[dict[str, Any]] = []
    by_family: dict[str, dict[str, Any]] = {fam: {} for fam in READOUT_FAMILIES}

    for gamma in cfg.gamma_grid:
        for fam in READOUT_FAMILIES:
            res = evaluate_readout_family(fam, float(gamma), bundle, cfg)
            rows.append(res)
            by_family[fam][f"{float(gamma):.2f}"] = res

    print(
        f"\n### Traceless Observable & Convex/Dyadic Born Rank-Bucket Probe "
        f"(`K={cfg.num_classes}`, `d={cfg.embed_dim}`, `samples={cfg.num_samples}`, "
        f"`beta={cfg.beta}`, `k={cfg.k_copies}`)"
    )
    print(
        "| Readout Family | `gamma` | Cross-Entropy | Top-1 Acc | Target Margin | "
        "Slope Ratio (`+0.65/-0.30`) | Rank 1 Mass | Ranks 2–5 | Ranks 6–20 | Ranks 21–100 | Ranks > 100 |"
    )
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in rows:
        print(
            f"| `{r['family']}` | `{r['gamma']:.2f}` | **`{r['cross_entropy_nats']:.4f}`** | "
            f"`{r['top1_acc_pct']:.2f}%` | `{r['mean_target_to_runner_up_margin']:.3f}` | "
            f"`{r['slope_ratio_peak_to_neg_tail']:.3f}x` | `{r['rank_1_prob_mass']:.4f}` | "
            f"`{r['ranks_2_to_5_prob_mass']:.4f}` | `{r['ranks_6_to_20_prob_mass']:.4f}` | "
            f"`{r['ranks_21_to_100_prob_mass']:.4f}` | `{r['ranks_gt_100_prob_mass']:.4f}` |"
        )

    best_gamma_key = f"{float(cfg.gamma_grid[-1]):.2f}"
    naive_ce = by_family["naive_kboson_born"][best_gamma_key]["cross_entropy_nats"]
    convex_ce = by_family["convex_traceless_born"][best_gamma_key]["cross_entropy_nats"]
    dyadic_ce = by_family["dyadic_hierarchical_born"][best_gamma_key]["cross_entropy_nats"]

    report = {
        "config": asdict(cfg),
        "summary": {
            "naive_kboson_ce_at_max_gamma": naive_ce,
            "convex_traceless_ce_at_max_gamma": convex_ce,
            "dyadic_hierarchical_ce_at_max_gamma": dyadic_ce,
            "convex_ce_gain_vs_naive_nats": round(naive_ce - convex_ce, 4),
            "dyadic_ce_gain_vs_naive_nats": round(naive_ce - dyadic_ce, 4),
        },
        "families": by_family,
        "rows": rows,
    }
    if cfg.json_out:
        out_p = Path(cfg.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> RankBucketProbeConfig:
    """Parse CLI arguments into a frozen `RankBucketProbeConfig`."""
    p = argparse.ArgumentParser(
        description="Profile Cross-Entropy, logit margins, and rank-bucket probability mass across Born readouts."
    )
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--coarse-clusters", type=int, default=32)
    p.add_argument("--fine-clusters", type=int, default=32)
    p.add_argument("--gamma-grid", type=str, default="0.0,0.5,1.0,1.5")
    p.add_argument("--beta", type=float, default=18.0)
    p.add_argument("--k-copies", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=2039)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--json-out", "--out-json", dest="json_out", type=str, default="")
    ns = p.parse_args(argv)
    gammas = tuple(float(x.strip()) for x in ns.gamma_grid.split(",") if x.strip())
    return RankBucketProbeConfig(
        num_classes=ns.num_classes,
        embed_dim=ns.embed_dim,
        num_samples=ns.num_samples,
        coarse_clusters=ns.coarse_clusters,
        fine_clusters=ns.fine_clusters,
        gamma_grid=gammas,
        beta=ns.beta,
        k_copies=ns.k_copies,
        seed=ns.seed,
        device=ns.device,
        json_out=ns.json_out,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    run_rank_bucket_probe(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
