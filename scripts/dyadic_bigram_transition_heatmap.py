#!/usr/bin/env python3
"""Dyadic Bigram Transition Heatmap & Predecessor Wavelet Specificity Auditor (Plan 037).

Promoted from `scripts/BACKLOG.md` (identified in Plan 036, implemented in Plan 037).
Audits on CPU:
1. The `256 x 256` coarse dyadic cluster transition matrix `pyramid.coarse_transition`
   (row entropy in nats, sparsity, and top coarse `c_prev -> c_next` cluster transitions).
2. The `2,048`-bucket predecessor wavelet states `pyramid.pred_wav_2048`, comparing
   Zipfian-centered predecessor accumulation against uncentered predecessor accumulation.
3. Step-0 exact self-overlap orthogonality (`max_self_overlap <= 0.05`) and `sharp_eff_rank`
   under Fix 2 (`num_fine_up_banks=16`).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.functional import l2_normalize
from qdlblocks.models.wavelet_cholesky_2pow16_gpt import (
    Radical2Pow16Config,
    Radical2Pow16WaveletCholeskyGPT,
    initialize_radical_2pow16_with_ppmi_svd,
)


@dataclass(frozen=True)
class BigramTransitionAuditConfig:
    """Configuration for auditing the 256x256 coarse transition matrix and 2048-bucket predecessor wavelets."""

    vocab_size: int = 50304
    valid_vocab_size: int = 50257
    ambient_n: int = 65536
    embed_dim: int = 64
    num_layers: int = 2
    num_fine_up_banks: int = 16
    num_clusters: int = 16
    stream_len: int = 16384
    top_k_transitions: int = 10
    ppmi_path: str = ""
    checkpoint_path: str = ""
    device: str = "cpu"
    out_json: str = ""


def build_or_load_ppmi_and_stream(cfg: BigramTransitionAuditConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Load or synthesize clustered PPMI embeddings and a Zipfian-skewed directed bigram token stream."""
    dev = torch.device(cfg.device)
    gen = torch.Generator(device="cpu").manual_seed(2037)
    d, n_clust = cfg.embed_dim, max(4, cfg.num_clusters)

    if cfg.ppmi_path and Path(cfg.ppmi_path).exists():
        loaded = torch.load(cfg.ppmi_path, map_location=dev, weights_only=True)
        if isinstance(loaded, dict) and "ppmi_embeddings" in loaded:
            loaded = loaded["ppmi_embeddings"]
        ppmi = l2_normalize(loaded[: cfg.vocab_size, :d].float().to(dev), dim=-1)
    else:
        svd_env = torch.arange(1, d + 1, dtype=torch.float32) ** -0.85
        q_mat, _ = torch.linalg.qr(torch.randn((n_clust, d), generator=gen).t())
        centers = F.normalize(q_mat.t()[:n_clust] * svd_env, p=2.0, dim=-1)
        cluster_of_v = torch.arange(cfg.vocab_size, dtype=torch.long) % n_clust
        noise = 0.025 * torch.randn((cfg.vocab_size, d), generator=gen) * svd_env
        ppmi = F.normalize(centers[cluster_of_v] + noise, p=2.0, dim=-1).to(dev)

    steps = torch.arange(cfg.stream_len, dtype=torch.long)
    c_seq = (steps * 3) % n_clust
    offset_seq = ((steps // n_clust) % 32) * n_clust
    toks = (c_seq + offset_seq) % cfg.valid_vocab_size
    toks = torch.where(steps % 10 == 0, torch.zeros_like(toks), toks)
    return ppmi, toks.to(dev)


@torch.no_grad()
def audit_coarse_transition_matrix(coarse_transition: torch.Tensor, top_k: int) -> dict[str, Any]:
    """Compute row entropy (nats), sparsity, and top `(c_prev -> c_next)` transitions of `coarse_transition`."""
    mat = coarse_transition.detach().float().cpu()
    n_rows, n_cols = mat.shape
    total_entries = n_rows * n_cols
    abs_mat = mat.abs()
    nonzero_count = int((abs_mat > 1e-6).sum().item())
    sparsity_fraction = 1.0 - (float(nonzero_count) / float(max(1, total_entries)))

    row_sums = abs_mat.sum(dim=-1)
    active_rows = row_sums > 1e-6
    num_active_rows = int(active_rows.sum().item())
    if num_active_rows > 0:
        probs = abs_mat[active_rows] / row_sums[active_rows].unsqueeze(-1).clamp_min(1e-12)
        row_entropies = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        mean_row_entropy_nats = float(row_entropies.mean().item())
        min_row_entropy_nats = float(row_entropies.min().item())
    else:
        mean_row_entropy_nats, min_row_entropy_nats = 0.0, 0.0

    uniform_entropy_nats = math.log(float(n_cols))
    k_eff = min(max(1, top_k), total_entries)
    top_vals, top_idx = torch.topk(mat.reshape(-1), k=k_eff, largest=True)

    top_transitions: list[dict[str, Any]] = []
    for rank_i in range(k_eff):
        flat_i = int(top_idx[rank_i].item())
        c_next, c_prev = flat_i // n_cols, flat_i % n_cols
        weight = float(top_vals[rank_i].item())
        r_sum = float(row_sums[c_next].item())
        top_transitions.append({
            "rank": rank_i + 1,
            "c_prev": c_prev,
            "c_next": c_next,
            "weight": round(weight, 6),
            "row_share": round(weight / r_sum if r_sum > 1e-12 else 0.0, 6),
        })

    return {
        "shape": [n_rows, n_cols],
        "frobenius_norm": round(float(mat.norm().item()), 6),
        "active_rows": num_active_rows,
        "nonzero_entries": nonzero_count,
        "sparsity_fraction": round(sparsity_fraction, 6),
        "mean_row_entropy_nats": round(mean_row_entropy_nats, 6),
        "min_row_entropy_nats": round(min_row_entropy_nats, 6),
        "uniform_entropy_nats": round(uniform_entropy_nats, 6),
        "entropy_reduction_nats": round(uniform_entropy_nats - mean_row_entropy_nats, 6),
        "top_transitions": top_transitions,
    }


@torch.no_grad()
def _compute_pair_overlap_ratio(
    model: Radical2Pow16WaveletCholeskyGPT, u_s: torch.Tensor, v_s: torch.Tensor,
) -> float:
    """Compute directed vs. random squared inner-product ratio `E[|<psi_u | m_v>|^2] / E[|<psi_u | m_rand>|^2]`."""
    pyr = model.pyramid
    psi_u = pyr.embed_tokens(u_s)
    m_sharp, _, _ = pyr.synthesize_vocab_detectors()
    bg_sq = float((psi_u * m_sharp.index_select(0, v_s)).sum(dim=-1).pow(2).mean().item())
    v_rand = torch.remainder(u_s + 257, model.cfg.valid_vocab_size)
    rnd_sq = max(float((psi_u * m_sharp.index_select(0, v_rand)).sum(dim=-1).pow(2).mean().item()), 1e-12)
    return float(bg_sq / rnd_sq)


@torch.no_grad()
def _compute_bucket_offdiag_cosine(pred_table: torch.Tensor) -> tuple[int, float]:
    """Compute count of active buckets and mean off-diagonal absolute cosine similarity across active buckets."""
    active = pred_table[pred_table.norm(dim=-1) > 1e-6]
    n_act = int(active.size(0))
    if n_act <= 1:
        return n_act, 0.0
    sample = l2_normalize(active[: min(256, n_act)].float(), dim=-1)
    n_s = int(sample.size(0))
    gram = (sample @ sample.t()).abs()
    eye = torch.eye(n_s, device=sample.device, dtype=sample.dtype)
    return n_act, float(((gram * (1.0 - eye)).sum() / float(n_s * (n_s - 1))).item())


@torch.no_grad()
def audit_predecessor_centering_effect(
    model: Radical2Pow16WaveletCholeskyGPT, token_stream: torch.Tensor,
) -> dict[str, Any]:
    """Compare Zipfian-centered `pred_wav_2048` against uncentered predecessor accumulation."""
    pyr, cfg = model.pyramid, model.cfg
    device = token_stream.device
    src = token_stream[:-1].clamp(0, cfg.vocab_size - 1)
    dst = token_stream[1:].clamp(0, cfg.vocab_size - 1)

    n_pairs = min(4096, int(src.numel()))
    u_s, v_s = src[:n_pairs], dst[:n_pairs]
    non_self = (u_s != v_s) & (u_s != 0) & (v_s != 0)
    if bool(non_self.any().item()):
        u_s, v_s = u_s[non_self], v_s[non_self]

    saved_pred = pyr.pred_wav_2048.detach().clone()
    saved_gain = pyr.pred_mix_gain.detach().clone()
    centered_act, centered_off_cos = _compute_bucket_offdiag_cosine(saved_pred)
    even_energy = float(saved_pred[:, 0::2].pow(2).sum().item())
    odd_energy = float(saved_pred[:, 1::2].pow(2).sum().item())

    pyr.pred_mix_gain.fill_(1.25)
    centered_ratio = _compute_pair_overlap_ratio(model, u_s, v_s)

    psi_all_v = pyr.embed_tokens(torch.arange(cfg.vocab_size, device=device))
    psi_src_uncentered = psi_all_v.index_select(0, src) + 0.85 * l2_normalize(psi_all_v[0:1], dim=-1)
    pred_uncentered = torch.zeros_like(saved_pred)
    pred_uncentered.index_add_(0, pyr.perm_v_to_leaf.index_select(0, dst) >> 5, psi_src_uncentered)
    pred_uncentered = l2_normalize(pred_uncentered, dim=-1) * 0.45

    _, uncentered_off_cos = _compute_bucket_offdiag_cosine(pred_uncentered)
    pyr.pred_wav_2048.copy_(pred_uncentered)
    uncentered_ratio = _compute_pair_overlap_ratio(model, u_s, v_s)

    pyr.pred_wav_2048.copy_(saved_pred)
    pyr.pred_mix_gain.copy_(saved_gain)
    default_centered_ratio = _compute_pair_overlap_ratio(model, u_s, v_s)

    return {
        "active_buckets_2048": centered_act,
        "default_directed_bigram_overlap_ratio": round(default_centered_ratio, 6),
        "centered_directed_bigram_overlap_ratio": round(centered_ratio, 6),
        "uncentered_directed_bigram_overlap_ratio": round(uncentered_ratio, 6),
        "centering_overlap_gain_ratio": round(centered_ratio / max(uncentered_ratio, 1e-6), 4),
        "centered_bucket_mean_abs_cos": round(centered_off_cos, 6),
        "uncentered_bucket_mean_abs_cos": round(uncentered_off_cos, 6),
        "odd_to_even_channel_energy_ratio": round(odd_energy / max(even_energy, 1e-12), 4),
    }


@torch.no_grad()
def run_bigram_transition_audit(cfg: BigramTransitionAuditConfig) -> dict[str, Any]:
    """Run the 256x256 coarse transition & 2048-bucket predecessor audit and print Markdown summary."""
    t0 = time.perf_counter()
    dev, d = torch.device(cfg.device), cfg.embed_dim
    r_cfg = Radical2Pow16Config(
        arch_mode="fused_kronecker_unet", vocab_size=cfg.vocab_size, valid_vocab_size=cfg.valid_vocab_size,
        ambient_n=cfg.ambient_n, num_layers=cfg.num_layers, embed_dim=d, num_heads=max(1, min(4, d // 16)),
        block_size=32, window_size=32, fine_detail_dim=max(16, d // 4), mid_detail_dim=max(32, d // 2),
        diffuse_rank=8, num_fine_up_banks=cfg.num_fine_up_banks, use_kraus_retraction_and_lambda_anneal=True,
    )
    model = Radical2Pow16WaveletCholeskyGPT(r_cfg).to(dev)
    ppmi, toks = build_or_load_ppmi_and_stream(cfg)
    init_stats = initialize_radical_2pow16_with_ppmi_svd(model, ppmi, toks, toks)

    if cfg.checkpoint_path and Path(cfg.checkpoint_path).exists():
        state = torch.load(cfg.checkpoint_path, map_location=dev, weights_only=True)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)

    transition_audit = audit_coarse_transition_matrix(model.pyramid.coarse_transition, top_k=cfg.top_k_transitions)
    predecessor_audit = audit_predecessor_centering_effect(model, toks)
    report = {
        "config": asdict(cfg),
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "step0_init_stats": {k: round(float(v), 6) for k, v in init_stats.items()},
        "coarse_transition_audit": transition_audit,
        "predecessor_2048_audit": predecessor_audit,
    }

    print(f"\n### Dyadic Bigram Transition & Predecessor Wavelet Audit (`d={d}, banks={cfg.num_fine_up_banks}, V={cfg.vocab_size}`)")
    print(f"- **Step-0 `max_self_overlap`**: `{init_stats['max_self_overlap']:.6f}` (`<= 0.05`) | **`sharp_eff_rank`**: `{init_stats['sharp_eff_rank']:.2f}` | **`kraus_stiefel_err`**: `{init_stats['kraus_stiefel_err']:.6f}`")
    print(f"- **`coarse_transition` (`256x256`)**: Active Rows `{transition_audit['active_rows']}` | Row Entropy `{transition_audit['mean_row_entropy_nats']:.3f}` nats (vs. uniform `{transition_audit['uniform_entropy_nats']:.3f}` nats) | Sparsity `{transition_audit['sparsity_fraction'] * 100.0:.2f}%`")
    print(f"- **Zipfian-Centered `pred_wav_2048`**: Centered Overlap Ratio **`{predecessor_audit['centered_directed_bigram_overlap_ratio']:.3f}x`** vs. Uncentered `{predecessor_audit['uncentered_directed_bigram_overlap_ratio']:.3f}x` (Gain **`{predecessor_audit['centering_overlap_gain_ratio']:.2f}x`**)")
    print("| Rank | Coarse Predecessor `c_prev` | Coarse Target `c_next` | Transition Weight | Row Share |")
    print("| :---: | :---: | :---: | :---: | :---: |")
    for row in transition_audit["top_transitions"][:6]:
        print(f"| `{row['rank']}` | `cluster_{row['c_prev']:03d}` | `cluster_{row['c_next']:03d}` | `{row['weight']:.4f}` | `{row['row_share'] * 100.0:.2f}%` |")

    if cfg.out_json:
        out_p = Path(cfg.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> BigramTransitionAuditConfig:
    """Parse CLI arguments into a frozen `BigramTransitionAuditConfig`."""
    p = argparse.ArgumentParser(description="Audit the 256x256 coarse dyadic transition matrix and 2048-bucket predecessor wavelets.")
    p.add_argument("--vocab-size", type=int, default=50304)
    p.add_argument("--valid-vocab-size", type=int, default=50257)
    p.add_argument("--ambient-n", type=int, default=65536)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-fine-up-banks", type=int, default=16)
    p.add_argument("--num-clusters", type=int, default=16)
    p.add_argument("--stream-len", type=int, default=16384)
    p.add_argument("--top-k-transitions", type=int, default=10)
    p.add_argument("--ppmi-path", type=str, default="")
    p.add_argument("--checkpoint-path", type=str, default="")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-json", type=str, default="")
    ns = p.parse_args(argv)
    return BigramTransitionAuditConfig(
        vocab_size=ns.vocab_size, valid_vocab_size=ns.valid_vocab_size, ambient_n=ns.ambient_n,
        embed_dim=ns.embed_dim, num_layers=ns.num_layers, num_fine_up_banks=ns.num_fine_up_banks,
        num_clusters=ns.num_clusters, stream_len=ns.stream_len, top_k_transitions=ns.top_k_transitions,
        ppmi_path=ns.ppmi_path, checkpoint_path=ns.checkpoint_path, device=ns.device, out_json=ns.out_json,
    )


if __name__ == "__main__":
    run_bigram_transition_audit(parse_args())
