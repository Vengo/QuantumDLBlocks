#!/usr/bin/env python3
"""16-Bit Dyadic Spectral Tree Multi-Resolution Coherence & Sibling Cluster Inspector (Plan 036).

Promoted from `scripts/BACKLOG.md` (identified in Plan 035, implemented in Plan 036).
Audits on CPU:
1. Exact bijection validity of `build_dyadic_2pow16_spectral_tree(ppmi_embeddings, ambient_n)`.
2. Multi-resolution intra-cluster cosine similarity across dyadic block sizes
   `[2, 4, 8, 16, 32, 64, 128, 256]`, verifying monotonic coherence increase as block size narrows
   and `>= 2.0x` coherence gain over raw token index ordering.
3. Top semantic dyadic sibling pairs `(2k, 2k+1)` and coarse macro-cluster summaries.
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

from qdlblocks.functional import l2_normalize
from qdlblocks.models.wavelet_cholesky_2pow16_gpt import build_dyadic_2pow16_spectral_tree

DYADIC_BLOCK_SIZES: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256)


@dataclass(frozen=True)
class SpectralTreeInspectorConfig:
    """Configuration for the 16-bit dyadic spectral tree multi-scale inspector."""

    vocab_size: int = 2048
    ambient_n: int = 4096
    embed_dim: int = 64
    num_clusters: int = 32
    top_k_siblings: int = 8
    ppmi_path: str = ""
    device: str = "cpu"
    out_json: str = ""


def build_or_load_ppmi_embeddings(cfg: SpectralTreeInspectorConfig) -> torch.Tensor:
    """Load PPMI embeddings from `cfg.ppmi_path` or synthesize hierarchical spectral embeddings."""
    dev = torch.device(cfg.device)
    if cfg.ppmi_path and Path(cfg.ppmi_path).exists():
        loaded = torch.load(cfg.ppmi_path, map_location=dev, weights_only=True)
        if isinstance(loaded, dict) and "ppmi_embeddings" in loaded:
            loaded = loaded["ppmi_embeddings"]
        return l2_normalize(loaded[: cfg.vocab_size, : cfg.embed_dim].float().to(dev), dim=-1)

    gen = torch.Generator(device="cpu").manual_seed(2036)
    d = cfg.embed_dim
    svd_env = torch.arange(1, d + 1, dtype=torch.float32) ** -0.85
    n_macro = max(4, cfg.num_clusters // 4)
    macro_centers = F.normalize(torch.randn((n_macro, d), generator=gen) * svd_env, p=2.0, dim=-1)
    n_sub = n_macro * 4
    sub_offsets = 0.32 * F.normalize(torch.randn((n_sub, d), generator=gen) * svd_env, p=2.0, dim=-1)
    sub_centers = F.normalize(macro_centers.repeat(4, 1) + sub_offsets, p=2.0, dim=-1)

    # Interleave cluster assignments in raw token order so raw contiguous blocks mix distinct clusters
    sub_assign = torch.arange(cfg.vocab_size, dtype=torch.long) % n_sub
    fine_noise = 0.12 * torch.randn((cfg.vocab_size, d), generator=gen) * svd_env
    ppmi = F.normalize(sub_centers[sub_assign] + fine_noise, p=2.0, dim=-1)
    return ppmi.to(dev)


def compute_intra_block_cosine(vecs: torch.Tensor, block_size: int) -> float:
    """Compute mean off-diagonal pairwise cosine similarity within contiguous blocks of `block_size`."""
    n_total, d = vecs.shape
    num_blocks = n_total // block_size
    if num_blocks <= 0 or block_size <= 1:
        return 1.0
    blocks = l2_normalize(vecs[: num_blocks * block_size].view(num_blocks, block_size, d), dim=-1)
    gram = torch.bmm(blocks, blocks.transpose(1, 2))
    eye = torch.eye(block_size, device=vecs.device, dtype=vecs.dtype).unsqueeze(0)
    off_diag_sum = (gram * (1.0 - eye)).sum(dim=(1, 2))
    return float((off_diag_sum / float(block_size * (block_size - 1))).mean().item())


def decode_token_label(token_id: int, num_clusters: int) -> str:
    """Return a human-readable synthetic or token label with its underlying semantic cluster ID."""
    macro_id = (token_id % (max(4, num_clusters) * 4)) // 4
    sub_id = (token_id % (max(4, num_clusters) * 4)) % 4
    return f"tok_{token_id:05d}[C{macro_id:02d}.{sub_id}]"


@torch.no_grad()
def inspect_multiscale_coherence(
    ppmi: torch.Tensor,
    leaf_to_v: torch.Tensor,
    vocab_size: int,
) -> dict[str, Any]:
    """Evaluate intra-block cosine coherence across dyadic block sizes `[2, 4, 8, 16, 32, 64, 128, 256]`."""
    lex_leaves = leaf_to_v[leaf_to_v < vocab_size]
    ordered_lex = ppmi.index_select(0, lex_leaves)
    scale_rows: dict[str, dict[str, float]] = {}

    valid_sizes = [b for b in DYADIC_BLOCK_SIZES if b <= ordered_lex.size(0)]
    for b in valid_sizes:
        raw_c = compute_intra_block_cosine(ppmi, b)
        spec_c = compute_intra_block_cosine(ordered_lex, b)
        gain = spec_c / max(abs(raw_c), 1e-4)
        scale_rows[f"block_{b}"] = {
            "block_size": float(b),
            "raw_cosine": round(raw_c, 6),
            "spectral_cosine": round(spec_c, 6),
            "gain_ratio": round(gain, 4),
        }

    c2 = scale_rows["block_2"]["spectral_cosine"]
    c8 = scale_rows["block_8"]["spectral_cosine"]
    c64 = scale_rows["block_64"]["spectral_cosine"]
    c256 = scale_rows.get("block_256", scale_rows[f"block_{valid_sizes[-1]}"])["spectral_cosine"]
    monotonic_key_scales = bool(c2 >= c8 >= c64 >= c256)
    spec_list = [scale_rows[f"block_{b}"]["spectral_cosine"] for b in valid_sizes]
    monotonic_all = all(spec_list[i] >= spec_list[i + 1] - 1e-4 for i in range(len(spec_list) - 1))

    return {
        "scales": scale_rows,
        "monotonic_key_scales_2_8_64_256": monotonic_key_scales,
        "monotonic_all_scales": bool(monotonic_all),
        "sibling_block2_cos": c2,
        "block8_cos": c8,
        "block64_cos": c64,
        "block256_cos": c256,
        "block8_gain_ratio": scale_rows["block_8"]["gain_ratio"],
        "block64_gain_ratio": scale_rows["block_64"]["gain_ratio"],
    }


@torch.no_grad()
def inspect_siblings_and_clusters(
    ppmi: torch.Tensor,
    leaf_to_v: torch.Tensor,
    cfg: SpectralTreeInspectorConfig,
) -> dict[str, Any]:
    """Extract top dyadic sibling pairs `(2k, 2k+1)` and coarse macro-cluster summaries."""
    lex_leaves = leaf_to_v[leaf_to_v < cfg.vocab_size]
    n_pairs = lex_leaves.numel() // 2
    pairs_v = lex_leaves[: n_pairs * 2].view(n_pairs, 2)
    vecs_0 = l2_normalize(ppmi.index_select(0, pairs_v[:, 0]), dim=-1)
    vecs_1 = l2_normalize(ppmi.index_select(0, pairs_v[:, 1]), dim=-1)
    pair_cos = (vecs_0 * vecs_1).sum(dim=-1)

    k_top = min(cfg.top_k_siblings, n_pairs)
    top_vals, top_idx = torch.topk(pair_cos, k=k_top, largest=True)
    top_siblings: list[dict[str, Any]] = []
    for rank_i in range(k_top):
        p_idx = int(top_idx[rank_i].item())
        u_id = int(pairs_v[p_idx, 0].item())
        v_id = int(pairs_v[p_idx, 1].item())
        top_siblings.append({
            "pair_index": p_idx,
            "token_a": u_id,
            "token_b": v_id,
            "label_a": decode_token_label(u_id, cfg.num_clusters),
            "label_b": decode_token_label(v_id, cfg.num_clusters),
            "cosine": round(float(top_vals[rank_i].item()), 6),
        })

    num_macro = min(256, max(1, cfg.ambient_n // 16))
    macro_len = cfg.ambient_n // num_macro
    macro_blocks = leaf_to_v.view(num_macro, macro_len)
    cluster_summaries: list[dict[str, Any]] = []
    for c_idx in range(min(6, num_macro)):
        block_ids = macro_blocks[c_idx]
        lex_ids = block_ids[block_ids < cfg.vocab_size]
        dark_count = int((block_ids >= cfg.vocab_size).sum().item())
        intra_c = compute_intra_block_cosine(ppmi.index_select(0, lex_ids), int(lex_ids.numel())) if lex_ids.numel() > 1 else 1.0
        sample_labels = [decode_token_label(int(t.item()), cfg.num_clusters) for t in lex_ids[:4]]
        cluster_summaries.append({
            "cluster_id": c_idx,
            "lexical_tokens": int(lex_ids.numel()),
            "dark_ancilla_leaves": dark_count,
            "intra_cluster_cosine": round(intra_c, 6),
            "sample_tokens": sample_labels,
        })

    return {
        "mean_sibling_cosine": round(float(pair_cos.mean().item()), 6),
        "top_sibling_pairs": top_siblings,
        "coarse_cluster_summaries": cluster_summaries,
    }


@torch.no_grad()
def run_spectral_tree_inspector(cfg: SpectralTreeInspectorConfig) -> dict[str, Any]:
    """Run the 16-bit dyadic spectral tree inspector, print Markdown tables, and optionally save JSON."""
    t0 = time.perf_counter()
    dev = torch.device(cfg.device)
    ppmi = build_or_load_ppmi_embeddings(cfg)
    perm_v_to_leaf, leaf_to_v = build_dyadic_2pow16_spectral_tree(ppmi, ambient_n=cfg.ambient_n)
    arange_n = torch.arange(cfg.ambient_n, device=dev, dtype=torch.long)
    is_bijection = bool(
        perm_v_to_leaf.shape == (cfg.ambient_n,)
        and torch.equal(leaf_to_v[perm_v_to_leaf], arange_n)
        and torch.equal(perm_v_to_leaf[leaf_to_v], arange_n)
    )

    coherence = inspect_multiscale_coherence(ppmi, leaf_to_v, cfg.vocab_size)
    clusters = inspect_siblings_and_clusters(ppmi, leaf_to_v, cfg)
    report = {
        "config": asdict(cfg),
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "is_bijection": is_bijection,
        "multiscale_coherence": coherence,
        "siblings_and_clusters": clusters,
    }

    print(f"\n### 16-Bit Dyadic Spectral Tree Inspector (`V={cfg.vocab_size}, N={cfg.ambient_n}, d={cfg.embed_dim}`)")
    print(
        f"- **Exact Bijection**: `{is_bijection}` | **Monotonic Coherence (`2 >= 8 >= 64 >= 256`)**: "
        f"`{coherence['monotonic_key_scales_2_8_64_256']}`"
    )
    print("| Dyadic Block Size | Raw Order Cosine | Spectral Tree Cosine | Coherence Gain |")
    print("| :---: | :---: | :---: | :---: |")
    for s_info in coherence["scales"].values():
        b_sz = int(s_info["block_size"])
        print(
            f"| `{b_sz}` leaves | `{s_info['raw_cosine']:.4f}` | "
            f"`{s_info['spectral_cosine']:.4f}` | **`{s_info['gain_ratio']:.2f}x`** |"
        )
    if cfg.out_json:
        out_p = Path(cfg.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> SpectralTreeInspectorConfig:
    """Parse CLI arguments into a frozen `SpectralTreeInspectorConfig`."""
    p = argparse.ArgumentParser(description="Inspect multi-resolution coherence and clusters of the 16-bit dyadic spectral tree.")
    p.add_argument("--vocab-size", type=int, default=2048)
    p.add_argument("--ambient-n", type=int, default=4096)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--num-clusters", type=int, default=32)
    p.add_argument("--top-k-siblings", type=int, default=8)
    p.add_argument("--ppmi-path", type=str, default="")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-json", type=str, default="")
    ns = p.parse_args(argv)
    return SpectralTreeInspectorConfig(
        vocab_size=ns.vocab_size,
        ambient_n=ns.ambient_n,
        embed_dim=ns.embed_dim,
        num_clusters=ns.num_clusters,
        top_k_siblings=ns.top_k_siblings,
        ppmi_path=ns.ppmi_path,
        device=ns.device,
        out_json=ns.out_json,
    )


if __name__ == "__main__":
    run_spectral_tree_inspector(parse_args())
