#!/usr/bin/env python3
"""Multi-Scale Dyadic Wavelet Ablation, Parseval Overlap, & Banked Subtree Lift Probe (Plan 037).

Promoted from `scripts/BACKLOG.md` (identified in Plan 036, implemented in Plan 037).
Audits on CPU:
1. `num_fine_up_banks=1` (Plan 036 single `64x256` lift) vs. `num_fine_up_banks=16`
   (Plan 037 Fix 2 banked `16x64x256` subtree lifts).
2. Scale-by-scale ablation (`only_256_c3`, `only_2048_c2`, `only_8192_c1`, `only_65536_c0`,
   cumulative `c3 -> c3+c2 -> c3+c2+c1 -> c3+c2+c1+c0`, and leave-one-out drops):
   - `sharp_eff_rank` `R_sharp = (Tr G)^2 / ||G||_F^2`,
   - Parseval mean squared off-diagonal overlap `E_{u != v}[c_{u,v}^2]` and background sum `V / R_sharp`,
   - Dyadic sibling token separation `1 - |<psi_{2k} | psi_{2k+1}>|^2`.
3. Fix 4 (`use_fast_cholesky_and_compile=False` vs. `True`) detector synthesis equivalence and latency (`ms`).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.models.wavelet_cholesky_2pow16_gpt import (
    Radical2Pow16Config,
    Radical2Pow16WaveletCholeskyGPT,
    initialize_radical_2pow16_with_ppmi_svd,
)

SCALE_ABLATION_MASKS: dict[str, tuple[float, float, float, float]] = {
    "only_256_c3": (1.0, 0.0, 0.0, 0.0),
    "only_2048_c2": (0.0, 1.0, 0.0, 0.0),
    "only_8192_c1": (0.0, 0.0, 1.0, 0.0),
    "only_65536_c0": (0.0, 0.0, 0.0, 1.0),
    "cum_c3": (1.0, 0.0, 0.0, 0.0),
    "cum_c3_c2": (1.0, 1.0, 0.0, 0.0),
    "cum_c3_c2_c1": (1.0, 1.0, 1.0, 0.0),
    "cum_c3_c2_c1_c0": (1.0, 1.0, 1.0, 1.0),
    "drop_256_c3": (0.0, 1.0, 1.0, 1.0),
    "drop_2048_c2": (1.0, 0.0, 1.0, 1.0),
    "drop_8192_c1": (1.0, 1.0, 0.0, 1.0),
    "drop_65536_c0": (1.0, 1.0, 1.0, 0.0),
}


@dataclass(frozen=True)
class WaveletScaleAblationConfig:
    """Configuration for multi-scale wavelet ablation and Fix 2 / Fix 4 profiling."""

    vocab_size: int = 50304
    valid_vocab_size: int = 50257
    ambient_n: int = 65536
    embed_dim: int = 256
    num_layers: int = 2
    sibling_probe_pairs: int = 2048
    bench_iters: int = 3
    device: str = "cpu"
    out_json: str = ""


def build_synthetic_ppmi_and_stream(
    vocab_size: int, valid_vocab_size: int, embed_dim: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize isotropic multi-cluster PPMI embeddings and a full-vocabulary directed token stream."""
    gen = torch.Generator(device="cpu").manual_seed(3037)
    n_clust = 16
    svd_env = torch.arange(1, embed_dim + 1, dtype=torch.float32) ** -0.35
    q_mat, _ = torch.linalg.qr(torch.randn((n_clust, embed_dim), generator=gen).t())
    centers = F.normalize(q_mat.t()[:n_clust] * svd_env, p=2.0, dim=-1)
    cluster_of_v = torch.arange(vocab_size, dtype=torch.long) % n_clust
    noise = 0.85 * torch.randn((vocab_size, embed_dim), generator=gen) * svd_env
    ppmi = F.normalize(centers[cluster_of_v] + noise, p=2.0, dim=-1).to(device)

    steps = torch.arange(max(65536, valid_vocab_size + 4096), dtype=torch.long)
    toks = (((steps * 3) % n_clust + (steps // n_clust) * n_clust) % valid_vocab_size).to(device)
    return ppmi, toks


@torch.no_grad()
def evaluate_single_scale_mask(
    model: Radical2Pow16WaveletCholeskyGPT,
    mask: tuple[float, float, float, float],
    sibling_pairs: int,
) -> dict[str, float]:
    """Evaluate `sharp_eff_rank`, Parseval overlap, and sibling separation under a 4-scale gain mask."""
    pyr, v_eff = model.pyramid, model.cfg.valid_vocab_size
    saved_gains = pyr.scale_gains.detach().clone()
    saved_trans_gain = pyr.transition_gain.detach().clone()
    saved_pred_gain = pyr.pred_mix_gain.detach().clone()

    masked_gains = torch.where(
        torch.tensor(mask, device=saved_gains.device, dtype=torch.float32) > 0.5,
        saved_gains,
        torch.full_like(saved_gains, -1000.0),
    )
    pyr.scale_gains.copy_(masked_gains)
    if mask[0] < 0.5:
        pyr.transition_gain.zero_()
    if mask[1] < 0.5:
        pyr.pred_mix_gain.zero_()

    t0 = time.perf_counter()
    m_sharp, _, _ = pyr.synthesize_vocab_detectors()
    synth_ms = (time.perf_counter() - t0) * 1000.0

    r_sharp = pyr.compute_sharp_eff_rank(m_sharp)
    parseval_bg_sum = float(v_eff) / max(r_sharp, 1e-6)
    mean_offdiag_sq = max(0.0, (parseval_bg_sum - 1.0) / float(max(1, v_eff - 1)))

    n_pairs = min(max(1, sibling_pairs), model.cfg.ambient_n // 2)
    leaf_idx = torch.arange(n_pairs * 2, device=saved_gains.device, dtype=torch.long)
    psi_leaves = pyr._synthesize_leaf_vectors(leaf_idx).view(n_pairs, 2, -1)
    sib_cos_sq = (psi_leaves[:, 0, :] * psi_leaves[:, 1, :]).sum(dim=-1).pow(2)
    sibling_sep = float((1.0 - sib_cos_sq).mean().item())

    pyr.scale_gains.copy_(saved_gains)
    pyr.transition_gain.copy_(saved_trans_gain)
    pyr.pred_mix_gain.copy_(saved_pred_gain)

    return {
        "sharp_eff_rank": round(r_sharp, 4),
        "parseval_bg_sum": round(parseval_bg_sum, 4),
        "mean_offdiag_sq_overlap": round(mean_offdiag_sq, 8),
        "sibling_separation": round(sibling_sep, 6),
        "synthesis_ms": round(synth_ms, 3),
    }


@torch.no_grad()
def profile_bank_configuration(
    cfg: WaveletScaleAblationConfig,
    num_fine_up_banks: int,
    ppmi: torch.Tensor,
    toks: torch.Tensor,
) -> tuple[Radical2Pow16WaveletCholeskyGPT, dict[str, Any]]:
    """Build and profile all 12 individual, cumulative, and drop scale masks for a given `num_fine_up_banks`."""
    dev, d = torch.device(cfg.device), cfg.embed_dim
    r_cfg = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet", vocab_size=cfg.vocab_size, valid_vocab_size=cfg.valid_vocab_size,
        ambient_n=cfg.ambient_n, num_layers=cfg.num_layers, embed_dim=d, num_heads=max(1, min(4, d // 16)),
        block_size=32, window_size=32, fine_detail_dim=max(16, d // 4), mid_detail_dim=max(32, d // 2),
        diffuse_rank=16, num_fine_up_banks=num_fine_up_banks, use_kraus_retraction_and_lambda_anneal=True,
        use_fast_cholesky_and_compile=False,
    )
    model = Radical2Pow16WaveletCholeskyGPT(r_cfg).to(dev)
    init_stats = initialize_radical_2pow16_with_ppmi_svd(model, ppmi, toks, toks)
    ablation_rows = {
        m_name: evaluate_single_scale_mask(model, m_tuple, sibling_pairs=cfg.sibling_probe_pairs)
        for m_name, m_tuple in SCALE_ABLATION_MASKS.items()
    }
    return model, {
        "num_fine_up_banks": num_fine_up_banks,
        "total_params": sum(p.numel() for p in model.parameters()),
        "step0_init_stats": {k: round(float(v), 6) for k, v in init_stats.items()},
        "ablations": ablation_rows,
    }


@torch.no_grad()
def profile_fast_cholesky_detector_synthesis(
    slow_model: Radical2Pow16WaveletCholeskyGPT, bench_iters: int,
) -> dict[str, float]:
    """Compare `use_fast_cholesky_and_compile=False` vs. `True` detector synthesis accuracy and speed."""
    dev = next(slow_model.parameters()).device
    fast_cfg = Radical2Pow16Config(**{**asdict(slow_model.cfg), "use_fast_cholesky_and_compile": True})
    fast_model = Radical2Pow16WaveletCholeskyGPT(fast_cfg).to(dev)
    fast_model.load_state_dict(slow_model.state_dict(), strict=True)

    m_sharp_slow, m_diff_slow, _ = slow_model.pyramid.synthesize_vocab_detectors()
    m_sharp_fast, m_diff_fast, _ = fast_model.pyramid.synthesize_vocab_detectors()
    max_sharp_diff = float((m_sharp_slow - m_sharp_fast).abs().max().item())
    max_diff_diff = float((m_diff_slow - m_diff_fast).abs().max().item())

    n_it = max(1, int(bench_iters))
    t0 = time.perf_counter()
    for _ in range(n_it):
        _ = slow_model.pyramid.synthesize_vocab_detectors()
    slow_ms = (time.perf_counter() - t0) * 1000.0 / float(n_it)

    t1 = time.perf_counter()
    for _ in range(n_it):
        _ = fast_model.pyramid.synthesize_vocab_detectors()
    fast_ms = (time.perf_counter() - t1) * 1000.0 / float(n_it)

    return {
        "max_abs_m_sharp_diff": round(max_sharp_diff, 8),
        "max_abs_m_diff_diff": round(max_diff_diff, 8),
        "slow_synthesis_ms": round(slow_ms, 3),
        "fast_synthesis_ms": round(fast_ms, 3),
        "speedup_ratio": round(slow_ms / max(fast_ms, 1e-6), 3),
    }


@torch.no_grad()
def run_wavelet_scale_ablation_probe(cfg: WaveletScaleAblationConfig) -> dict[str, Any]:
    """Run the 1-bank vs. 16-bank wavelet scale ablation probe and print a Markdown report."""
    t0 = time.perf_counter()
    dev = torch.device(cfg.device)
    ppmi, toks = build_synthetic_ppmi_and_stream(cfg.vocab_size, cfg.valid_vocab_size, cfg.embed_dim, dev)

    _, bank1_res = profile_bank_configuration(cfg, num_fine_up_banks=1, ppmi=ppmi, toks=toks)
    model_16, bank16_res = profile_bank_configuration(cfg, num_fine_up_banks=16, ppmi=ppmi, toks=toks)
    fast_cmp = profile_fast_cholesky_detector_synthesis(model_16, bench_iters=cfg.bench_iters)

    r1_full = bank1_res["ablations"]["cum_c3_c2_c1_c0"]["sharp_eff_rank"]
    r16_full = bank16_res["ablations"]["cum_c3_c2_c1_c0"]["sharp_eff_rank"]
    bg1_full = bank1_res["ablations"]["cum_c3_c2_c1_c0"]["parseval_bg_sum"]
    bg16_full = bank16_res["ablations"]["cum_c3_c2_c1_c0"]["parseval_bg_sum"]

    comparison = {
        "bank1_full_sharp_eff_rank": r1_full,
        "bank16_full_sharp_eff_rank": r16_full,
        "sharp_eff_rank_gain_ratio": round(r16_full / max(r1_full, 1e-6), 4),
        "bank1_parseval_bg_sum": bg1_full,
        "bank16_parseval_bg_sum": bg16_full,
        "parseval_tail_reduction_ratio": round(bg1_full / max(bg16_full, 1e-6), 4),
        "bank16_only_c0_sharp_eff_rank": bank16_res["ablations"]["only_65536_c0"]["sharp_eff_rank"],
        "bank1_only_c0_sharp_eff_rank": bank1_res["ablations"]["only_65536_c0"]["sharp_eff_rank"],
        "fast_cholesky_synthesis": fast_cmp,
    }
    report = {
        "config": asdict(cfg),
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "bank1_profile": bank1_res,
        "bank16_profile": bank16_res,
        "comparison": comparison,
    }

    print(f"\n### Wavelet Scale Ablation & Parseval Tail Probe (`d={cfg.embed_dim}, V={cfg.valid_vocab_size}, N={cfg.ambient_n}`)")
    print(f"- **Fix 2 Rank Expansion**: `1` Bank `R_sharp = {r1_full:.2f}` (`V/R = {bg1_full:.1f}`) -> `16` Banks **`R_sharp = {r16_full:.2f}`** (`V/R = {bg16_full:.1f}`, **`{comparison['sharp_eff_rank_gain_ratio']:.2f}x`** rank gain)")
    print(f"- **Fix 4 Detector Synthesis**: `max_abs_diff = {fast_cmp['max_abs_m_sharp_diff']:.2e}` | Slow `{fast_cmp['slow_synthesis_ms']:.2f} ms` vs. Fast **`{fast_cmp['fast_synthesis_ms']:.2f} ms`** (`{fast_cmp['speedup_ratio']:.2f}x` speedup)")
    print("| Ablation Mask | `1`-Bank `R_sharp` | `16`-Bank `R_sharp` | `16`-Bank `V / R_sharp` | `16`-Bank `E[c_{u,v}^2]` | Sibling Sep |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for m_name in SCALE_ABLATION_MASKS:
        r1, r16 = bank1_res["ablations"][m_name], bank16_res["ablations"][m_name]
        print(f"| `{m_name}` | `{r1['sharp_eff_rank']:.2f}` | **`{r16['sharp_eff_rank']:.2f}`** | `{r16['parseval_bg_sum']:.1f}` | `{r16['mean_offdiag_sq_overlap']:.6f}` | `{r16['sibling_separation']:.4f}` |")

    if cfg.out_json:
        out_p = Path(cfg.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> WaveletScaleAblationConfig:
    """Parse CLI arguments into a frozen `WaveletScaleAblationConfig`."""
    p = argparse.ArgumentParser(description="Profile scale-by-scale wavelet ablations and 1-bank vs. 16-bank Parseval overlap.")
    p.add_argument("--vocab-size", type=int, default=50304)
    p.add_argument("--valid-vocab-size", type=int, default=50257)
    p.add_argument("--ambient-n", type=int, default=65536)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--sibling-probe-pairs", type=int, default=2048)
    p.add_argument("--bench-iters", type=int, default=3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-json", type=str, default="")
    ns = p.parse_args(argv)
    return WaveletScaleAblationConfig(
        vocab_size=ns.vocab_size, valid_vocab_size=ns.valid_vocab_size, ambient_n=ns.ambient_n,
        embed_dim=ns.embed_dim, num_layers=ns.num_layers, sibling_probe_pairs=ns.sibling_probe_pairs,
        bench_iters=ns.bench_iters, device=ns.device, out_json=ns.out_json,
    )


if __name__ == "__main__":
    run_wavelet_scale_ablation_probe(parse_args())
