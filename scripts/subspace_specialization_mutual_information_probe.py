#!/usr/bin/env python3
"""Subspace Specialization, Cross-Level Norm Cannibalization, & Mutual Information Probe (`Plan 047`).

Promoted from `scripts/TOOLING_BACKLOG.md` (identified in Plan 046, implemented in Plan 047 / Wave 32).
Audits three foundational properties of multi-level wavelet and multi-subspace representations:
1. **Cross-Level L2 Norm Cannibalization (`merged_global_l2` vs. `explicit_level_separated`)**:
   Demonstrates mathematically and numerically why summing the 4 wavelet levels (`coarse_256`,
   `mid_2048`, `sub_8192`, `fine_65536`) under a single global `l2_normalize` crushes fine leaf
   energy (`< 5%`) when `wav_256` grows by `8x` during early semantic clustering, whereas explicit
   level-separated tensors (`WaveletScaleState`) preserve exact `25.0%` (`1/M`) energy per level.
2. **Singular-Value Spectrum Balance (`contiguous` vs. `strided` vs. `explicit_level_separated`)**:
   Quantifies how contiguous slicing `[0:64], [64:128], ...` of sorted PPMI-SVD coordinates creates
   severe inter-subspace variance skew (`> 10x`), whereas strided round-robin partitioning (`x[..., m::M]`)
   equalizes singular-value energy across all `M` subspaces.
3. **Pairwise Cross-Subspace Gaussian Mutual Information & Per-Level Participation Ratio `R_sharp^{(m)}`**:
   Computes `I(X^{(m)}; X^{(m')}) = -0.5 * log det(I - C_{m,m'} C_{m,m'}^T)` and `R_sharp^{(m)}` across
   the `M = 4` subspaces.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.functional import l2_normalize
from qdlblocks.layers.multirate_wavefunction_conv import (
    NUM_WAVELET_PYRAMID_LEVELS,
    WaveletScaleState,
)
from qdlblocks.layers.subspace_coincidence_born import subspace_l2_normalize

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Documented in EXPERIMENT_PLAN_047.md
# ==============================================================================
DEFAULT_PROBE_EMBED_DIM: int = 256
DEFAULT_PROBE_NUM_SUBSPACES: int = 4
DEFAULT_PROBE_VOCAB_SIZE: int = 8192
COARSE_GROWTH_MULTIPLIER: float = 8.0
SVD_SPECTRUM_DECAY_POWER: float = 0.85
MI_EIGENVALUE_CLAMP_MAX: float = 0.9999


@dataclass(frozen=True)
class SubspaceSpecializationProbeConfig:
    """Configuration for `subspace_specialization_mutual_information_probe.py` (`<= 8` args)."""

    embed_dim: int = DEFAULT_PROBE_EMBED_DIM
    num_subspaces: int = DEFAULT_PROBE_NUM_SUBSPACES
    vocab_size: int = DEFAULT_PROBE_VOCAB_SIZE
    coarse_growth_multiplier: float = COARSE_GROWTH_MULTIPLIER
    device: str = "cpu"
    seed: int = 42
    json_out: str = ".cache/subspace_specialization_mutual_information_probe.json"


def compute_participation_ratio(x: torch.Tensor) -> float:
    """Compute centered Gram participation ratio `(Tr G)^2 / ||G||_F^2` for `(V, d_s)` tensor."""
    x_c = x.detach().float()
    x_c = x_c - x_c.mean(dim=0, keepdim=True)
    gram = x_c.t() @ x_c
    tr = torch.diagonal(gram).sum()
    return float(((tr * tr) / (gram * gram).sum().clamp_min(1e-12)).item())


def compute_gaussian_mutual_information_nats(x_m: torch.Tensor, x_n: torch.Tensor) -> float:
    """Compute canonical-correlation Gaussian MI `I(X_m; X_n) = -0.5 * sum_i ln(1 - rho_i^2)` in nats."""
    a = x_m.detach().float()
    b = x_n.detach().float()
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    q_a, _ = torch.linalg.qr(a, mode="reduced")
    q_b, _ = torch.linalg.qr(b, mode="reduced")
    cross = q_a.t() @ q_b
    s_vals = torch.linalg.svdvals(cross).clamp(max=MI_EIGENVALUE_CLAMP_MAX)
    mi = -0.5 * torch.log1p(-(s_vals * s_vals)).sum().item()
    return float(max(0.0, mi))


def audit_cross_level_norm_cannibalization(cfg: SubspaceSpecializationProbeConfig) -> dict[str, Any]:
    """Compare fine-level (`wav_65536`) energy survival when `wav_256` grows by `8x`."""
    device = torch.device(cfg.device)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    v, d = int(cfg.vocab_size), int(cfg.embed_dim)
    d_s = d // NUM_WAVELET_PYRAMID_LEVELS

    # Simulate the 4 wavelet levels when coarse_256 has grown by COARSE_GROWTH_MULTIPLIER (8x)
    c3_raw = torch.randn((v, d), generator=gen, device=device) * float(cfg.coarse_growth_multiplier)
    c2_raw = torch.randn((v, d), generator=gen, device=device) * 2.0
    c1_raw = torch.randn((v, d), generator=gen, device=device) * 1.0
    c0_raw = torch.randn((v, d), generator=gen, device=device) * 1.0

    # 1. Legacy global L2 normalization over summed levels: psi = l2_normalize(c3 + c2 + c1 + c0)
    merged_sum = c3_raw + c2_raw + c1_raw + c0_raw
    merged_denom_sq = merged_sum.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
    merged_fine_energy_frac = float((c0_raw.square().sum(dim=-1, keepdim=True) / merged_denom_sq).mean().item())
    merged_coarse_energy_frac = float((c3_raw.square().sum(dim=-1, keepdim=True) / merged_denom_sq).mean().item())
    merged_unit = l2_normalize(merged_sum, dim=-1)

    # 2. Explicit Level-Separated Wavelet Tensors (`WaveletScaleState`)
    sep_state = WaveletScaleState(
        coarse_256=c3_raw[:, :d_s] * float(cfg.coarse_growth_multiplier),
        mid_2048=c2_raw[:, :d_s] * 2.0,
        sub_8192=c1_raw[:, :d_s],
        fine_65536=c0_raw[:, :d_s],
    ).normalize_levels()
    sep_fractions = sep_state.level_energy_fractions()
    sep_concat = sep_state.to_product_of_spheres()
    max_unit_norm_err = float((sep_concat.norm(dim=-1) - 1.0).abs().max().item())

    return {
        "coarse_growth_multiplier": float(cfg.coarse_growth_multiplier),
        "merged_global_l2_coarse_energy_fraction": merged_coarse_energy_frac,
        "merged_global_l2_fine_energy_fraction": merged_fine_energy_frac,
        "merged_global_l2_r_sharp": compute_participation_ratio(merged_unit),
        "explicit_level_separated_energy_fractions": sep_fractions,
        "explicit_level_separated_fine_energy_fraction": sep_fractions["fine_65536"],
        "explicit_level_separated_r_sharp": compute_participation_ratio(sep_concat),
        "max_unit_norm_err": max_unit_norm_err,
    }


def audit_svd_spectrum_and_mutual_information(cfg: SubspaceSpecializationProbeConfig) -> dict[str, Any]:
    """Compare contiguous vs. strided round-robin subspace partitioning on sorted PPMI-SVD coordinates."""
    device = torch.device(cfg.device)
    gen = torch.Generator(device=device).manual_seed(cfg.seed + 7)
    v, d, m = int(cfg.vocab_size), int(cfg.embed_dim), int(cfg.num_subspaces)
    d_sub = d // m

    # Sorted singular value spectrum sigma_j = (j + 1)^{-0.85} mimicking PPMI-SVD embeddings
    j_idx = torch.arange(1, d + 1, device=device, dtype=torch.float32)
    singular_profile = torch.pow(j_idx, -SVD_SPECTRUM_DECAY_POWER).unsqueeze(0)
    raw_ppmi = torch.randn((v, d), generator=gen, device=device) * singular_profile

    # Pre-normalization per-subspace energy skew: contiguous vs. strided
    contig_views = raw_ppmi.reshape(v, m, d_sub)
    contig_sub_vars = contig_views.square().sum(dim=-1).mean(dim=0)
    contig_skew_ratio = float((contig_sub_vars.max() / contig_sub_vars.min().clamp_min(1e-12)).item())

    strided_views = raw_ppmi.reshape(v, d_sub, m)
    strided_sub_vars = strided_views.square().sum(dim=-2).mean(dim=0)
    strided_skew_ratio = float((strided_sub_vars.max() / strided_sub_vars.min().clamp_min(1e-12)).item())

    # Normalized Product-of-Spheres states and pairwise mutual information
    strided_norm = subspace_l2_normalize(raw_ppmi, num_subspaces=m, partition_mode="strided")
    strided_3d = strided_norm.reshape(v, d_sub, m)
    per_sub_r_sharp = [compute_participation_ratio(strided_3d[:, :, i]) for i in range(m)]
    pairwise_mi: list[float] = []
    for i in range(m):
        for j in range(i + 1, m):
            pairwise_mi.append(compute_gaussian_mutual_information_nats(strided_3d[:, :, i], strided_3d[:, :, j]))

    return {
        "contiguous_pre_norm_energy_skew_ratio": contig_skew_ratio,
        "strided_pre_norm_energy_skew_ratio": strided_skew_ratio,
        "skew_reduction_factor": contig_skew_ratio / max(1e-9, strided_skew_ratio),
        "strided_per_subspace_r_sharp": per_sub_r_sharp,
        "mean_pairwise_mutual_information_nats": float(sum(pairwise_mi) / max(1, len(pairwise_mi))),
    }


def run_probe(cfg: SubspaceSpecializationProbeConfig) -> dict[str, Any]:
    """Execute the full Plan 047 subspace specialization & cross-level norm probe."""
    cannibal_stats = audit_cross_level_norm_cannibalization(cfg)
    svd_stats = audit_svd_spectrum_and_mutual_information(cfg)

    h1_energy_passed = (
        cannibal_stats["merged_global_l2_fine_energy_fraction"] < 0.05
        and abs(cannibal_stats["explicit_level_separated_fine_energy_fraction"] - 0.25) < 1e-5
        and cannibal_stats["max_unit_norm_err"] < 1e-6
    )
    h1_strided_passed = svd_stats["skew_reduction_factor"] >= 5.0

    print("\n### Plan 047 Subspace Specialization & Cross-Level Norm Cannibalization Audit")
    print("| Diagnostic Metric | Merged Global L2 (`M=1`) | Contiguous (`M=4`) | Strided / Explicit Level-Separated (`M=4`) |")
    print("| :--- | :---: | :---: | :---: |")
    print(
        f"| Fine `wav_65536` Energy Fraction (`8x` Coarse Growth) | "
        f"`{cannibal_stats['merged_global_l2_fine_energy_fraction'] * 100:.2f}%` | `25.00%` | "
        f"**`{cannibal_stats['explicit_level_separated_fine_energy_fraction'] * 100:.2f}%`** |"
    )
    print(
        f"| Pre-Norm PPMI-SVD Subspace Energy Skew Ratio | — | "
        f"`{svd_stats['contiguous_pre_norm_energy_skew_ratio']:.2f}x` | "
        f"**`{svd_stats['strided_pre_norm_energy_skew_ratio']:.2f}x`** (`{svd_stats['skew_reduction_factor']:.1f}x` lower) |"
    )
    print(
        f"| Mean Pairwise Cross-Subspace MI (nats) | — | — | "
        f"`{svd_stats['mean_pairwise_mutual_information_nats']:.4f}` |"
    )

    report = {
        "config": asdict(cfg),
        "cross_level_norm_cannibalization": cannibal_stats,
        "svd_spectrum_and_mutual_information": svd_stats,
        "H1_cross_level_energy_preservation_passed": bool(h1_energy_passed),
        "H1_strided_svd_balance_passed": bool(h1_strided_passed),
        "all_passed": bool(h1_energy_passed and h1_strided_passed),
    }
    if cfg.json_out:
        out_p = ROOT / cfg.json_out if not Path(cfg.json_out).is_absolute() else Path(cfg.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> SubspaceSpecializationProbeConfig:
    """Parse CLI arguments into `SubspaceSpecializationProbeConfig`."""
    p = argparse.ArgumentParser(description="Audit cross-level norm cannibalization and subspace specialization.")
    p.add_argument("--embed-dim", type=int, default=DEFAULT_PROBE_EMBED_DIM)
    p.add_argument("--num-subspaces", type=int, default=DEFAULT_PROBE_NUM_SUBSPACES)
    p.add_argument("--vocab-size", type=int, default=DEFAULT_PROBE_VOCAB_SIZE)
    p.add_argument("--coarse-growth-multiplier", type=float, default=COARSE_GROWTH_MULTIPLIER)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json-out", type=str, default=".cache/subspace_specialization_mutual_information_probe.json")
    ns = p.parse_args(argv)
    return SubspaceSpecializationProbeConfig(
        embed_dim=ns.embed_dim,
        num_subspaces=ns.num_subspaces,
        vocab_size=ns.vocab_size,
        coarse_growth_multiplier=ns.coarse_growth_multiplier,
        device=ns.device,
        seed=ns.seed,
        json_out=ns.json_out,
    )


if __name__ == "__main__":
    rep = run_probe(parse_args())
    sys.exit(0 if bool(rep["all_passed"]) else 1)
