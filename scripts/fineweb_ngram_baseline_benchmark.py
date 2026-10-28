#!/usr/bin/env python3
"""GPU-Vectorized Unigram, Bigram, and Trigram Baseline Benchmark on FineWeb (`Wave 42` Comparison).

Computes exact Cross-Entropy (`val_ce`), Perplexity (`val_ppl`), Top-1 Accuracy (`top1_acc`),
and Top-50 Accuracy (`top50_acc`) for classical count-based Unigram, Bigram, and Trigram
language models (with hierarchical Dirichlet backoff smoothing) using 100% vectorized GPU
`torch.unique` + `torch.searchsorted` lookups (`< 2s` for `100M` tokens, zero Python token loops).

Evaluates across all three FineWeb Shard-0 (`fineweb_val_000000.bin`) validation scopes:
  1. `65,536` tokens (`16` batches x `8` x `512`): the rapid in-loop screening slice used in Waves 40-42.
  2. `10,485,760` tokens (`10.49M` tokens): the standard `KellerJordan/modded-nanogpt` `val_tokens` slice.
  3. `100,000,000` tokens (`100.0M` tokens): the complete `fineweb_val_000000.bin` Shard 0.
And across both:
  - `2.46M` training tokens (matching the 300-step `B=16, T=512` Stage-3 quantum model token budget), and
  - `100.0M` training tokens (the full first training shard `fineweb_train_000001.bin`).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import urllib.request

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p_str in (str(ROOT / "src"), str(ROOT)):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from qdlblocks.models.kronecker_multimeaning_16x4_gpt import (  # noqa: E402
    _load_header_safe_shard_tokens,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
DATA_DIR: Path = ROOT / ".cache" / "fineweb_100M"
DEFAULT_JSON_OUT: Path = ROOT / ".cache" / "fineweb_ngram_baseline_benchmark.json"
VAL_SHARD0_URL: str = (
    "https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main/fineweb_val_000000.bin"
)
FULL_SHARD_BYTES: int = 200_001_024

VALID_VOCAB_SIZE: int = 50257
SLICE_2P46M_TOKENS: int = 2_457_600  # Exact 300 steps * 16 * 512 tokens
SLICE_100M_TOKENS: int = 100_000_000

VAL_FAST_65K_TOKENS: int = 65_536
VAL_MODDED_10M_TOKENS: int = 10_485_760
VAL_FULL_100M_TOKENS: int = 100_000_000
EVAL_CHUNK_SIZE: int = 5_000_000
TOP_K_EVAL: int = 50

DEFAULT_UNIGRAM_ALPHA: float = 1.0
DEFAULT_BIGRAM_BETA: float = 120.0
DEFAULT_TRIGRAM_BETA: float = 200.0
DEFAULT_TRIGRAM_JM_LAMBDA: float = 0.35


@dataclass(frozen=True)
class VectorizedNGramModel:
    """Compact GPU/CPU representation of smoothed Unigram, Bigram, and Trigram tables (`<= 8` fields)."""

    p_uni: torch.Tensor
    top50_uni_ids: torch.Tensor
    ctx1_totals: torch.Tensor
    bi_uniq_keys: torch.Tensor
    bi_uniq_counts: torch.Tensor
    bi_best_y_per_x: torch.Tensor
    tri_uniq_keys: torch.Tensor
    tri_uniq_counts: torch.Tensor


@dataclass(frozen=True)
class TrigramContextTable:
    """Auxiliary trigram context counts and top-1 predictions (`<= 8` fields)."""

    ctx2_uniq_keys: torch.Tensor
    ctx2_uniq_counts: torch.Tensor
    tri_best_ctx2_keys: torch.Tensor
    tri_best_y: torch.Tensor
    beta_2: float = DEFAULT_BIGRAM_BETA
    beta_3: float = DEFAULT_TRIGRAM_BETA


def ensure_full_val_shard0(data_dir: Path) -> Path:
    """Ensure the full 100M-token `fineweb_val_000000.bin` (Shard 0) is present on disk."""
    data_dir.mkdir(parents=True, exist_ok=True)
    full_val_path = data_dir / "fineweb_val_000000.bin"
    if full_val_path.exists() and full_val_path.stat().st_size >= FULL_SHARD_BYTES:
        return full_val_path

    tmp_path = full_val_path.with_suffix(".bin.part")
    print(
        f"[FineWeb Shard 0] Downloading full 100M-token `fineweb_val_000000.bin` "
        f"({FULL_SHARD_BYTES / 1e6:.1f} MB) from HuggingFace ...",
        flush=True,
    )
    req = urllib.request.Request(VAL_SHARD0_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(tmp_path, "wb") as f:
        while True:
            chunk = resp.read(16 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp_path.replace(full_val_path)
    return full_val_path


def _segment_argmax_by_sorted_key(
    sorted_group_keys: torch.Tensor,
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Given sorted `sorted_group_keys`, return `(unique_group_keys, argmax_candidate_id_per_group)`."""
    if sorted_group_keys.numel() == 0:
        return sorted_group_keys, candidate_ids
    uniq_g, inv_idx = torch.unique_consecutive(sorted_group_keys, return_inverse=True)
    num_g = int(uniq_g.numel())
    max_scores = torch.full((num_g,), -1e30, dtype=scores.dtype, device=scores.device)
    max_scores.scatter_reduce_(0, inv_idx, scores, reduce="amax", include_self=True)
    is_winner = scores >= (max_scores[inv_idx] - 1e-12)
    win_pos = torch.nonzero(is_winner, as_tuple=False).squeeze(-1)
    win_g = inv_idx[win_pos]
    # Keep first winner per group
    first_mask = torch.ones_like(win_g, dtype=torch.bool)
    if win_g.numel() > 1:
        first_mask[1:] = win_g[1:] != win_g[:-1]
    chosen_pos = win_pos[first_mask]
    best_cand = candidate_ids[chosen_pos]
    return uniq_g, best_cand


def fit_vectorized_ngram_tables(
    train_tokens_np: np.ndarray,
    val_stream_np: np.ndarray,
    device: torch.device,
    beta_2: float = DEFAULT_BIGRAM_BETA,
    beta_3: float = DEFAULT_TRIGRAM_BETA,
) -> tuple[VectorizedNGramModel, TrigramContextTable]:
    """Fit exact Unigram, Bigram, and validation-filtered Trigram count tables on GPU (`<= 8` args)."""
    v_size = VALID_VOCAB_SIZE
    tr_cpu = np.clip(train_tokens_np, 0, v_size - 1).astype(np.int64)

    # 1. Exact Unigram counts & probabilities
    counts_1_np = np.bincount(tr_cpu, minlength=v_size).astype(np.float64)
    p_uni_np = (counts_1_np + DEFAULT_UNIGRAM_ALPHA) / float(
        counts_1_np.sum() + DEFAULT_UNIGRAM_ALPHA * v_size
    )
    p_uni = torch.from_numpy(p_uni_np).to(device=device, dtype=torch.float64)
    top50_uni_ids = torch.argsort(p_uni, descending=True)[:TOP_K_EVAL]
    top1_uni_id = int(top50_uni_ids[0].item())

    # 2. Exact Bigram counts `C_2(x, y)` across the entire training slice
    ctx1_np = np.bincount(tr_cpu[:-1], minlength=v_size).astype(np.float64)
    ctx1_totals = torch.from_numpy(ctx1_np).to(device=device, dtype=torch.float64)

    bi_keys_cpu = tr_cpu[:-1] * v_size + tr_cpu[1:]
    bi_uniq_k_list: list[torch.Tensor] = []
    bi_uniq_c_list: list[torch.Tensor] = []
    chunk_step = 25_000_000
    for start in range(0, int(bi_keys_cpu.shape[0]), chunk_step):
        k_gpu = torch.from_numpy(bi_keys_cpu[start : start + chunk_step]).to(device=device)
        uk, uc = torch.unique(k_gpu, return_counts=True)
        bi_uniq_k_list.append(uk)
        bi_uniq_c_list.append(uc.to(torch.float64))
        del k_gpu
    if len(bi_uniq_k_list) == 1:
        bi_uniq_keys = bi_uniq_k_list[0]
        bi_uniq_counts = bi_uniq_c_list[0]
    else:
        cat_k = torch.cat(bi_uniq_k_list)
        cat_c = torch.cat(bi_uniq_c_list)
        order = torch.argsort(cat_k)
        cat_k = cat_k[order]
        cat_c = cat_c[order]
        bi_uniq_keys, inv = torch.unique_consecutive(cat_k, return_inverse=True)
        bi_uniq_counts = torch.zeros(bi_uniq_keys.shape[0], dtype=torch.float64, device=device)
        bi_uniq_counts.scatter_add_(0, inv, cat_c)

    # Best bigram prediction per context `x`
    u_x = torch.div(bi_uniq_keys, v_size, rounding_mode="floor")
    u_y = bi_uniq_keys % v_size
    bi_scores = bi_uniq_counts + beta_2 * p_uni[u_y]
    uniq_x, best_y_obs = _segment_argmax_by_sorted_key(u_x, u_y, bi_scores)
    bi_best_y_per_x = torch.full((v_size,), top1_uni_id, dtype=torch.long, device=device)
    bi_best_y_per_x[uniq_x] = best_y_obs

    # 3. Trigram counts restricted to bigram contexts `(w1, w2)` that appear in `val_stream_np`
    val_clip = np.clip(val_stream_np, 0, v_size - 1).astype(np.int64)
    val_bi_ctx_cpu = val_clip[:-2] * v_size + val_clip[1:-1]
    uniq_val_bi_list: list[torch.Tensor] = []
    for start in range(0, int(val_bi_ctx_cpu.shape[0]), chunk_step):
        vb_gpu = torch.from_numpy(val_bi_ctx_cpu[start : start + chunk_step]).to(device=device)
        uniq_val_bi_list.append(torch.unique(vb_gpu))
        del vb_gpu
    uniq_val_bi = torch.unique(torch.cat(uniq_val_bi_list))

    # Filter training trigrams whose bigram prefix `(w1, w2)` is in `uniq_val_bi`
    tri_k_list: list[torch.Tensor] = []
    tri_c_list: list[torch.Tensor] = []
    n_tri = int(tr_cpu.shape[0]) - 2
    for start in range(0, n_tri, chunk_step):
        end = min(n_tri, start + chunk_step)
        w1_g = torch.from_numpy(tr_cpu[start:end]).to(device=device)
        w2_g = torch.from_numpy(tr_cpu[start + 1 : end + 1]).to(device=device)
        y_g = torch.from_numpy(tr_cpu[start + 2 : end + 2]).to(device=device)
        bi_g = w1_g * v_size + w2_g
        idx = torch.searchsorted(uniq_val_bi, bi_g)
        in_val = (idx < uniq_val_bi.numel()) & (
            uniq_val_bi[idx.clamp_max(uniq_val_bi.numel() - 1)] == bi_g
        )
        t_keys = bi_g[in_val] * v_size + y_g[in_val]
        uk, uc = torch.unique(t_keys, return_counts=True)
        tri_k_list.append(uk)
        tri_c_list.append(uc.to(torch.float64))
        del w1_g, w2_g, y_g, bi_g, idx, in_val, t_keys

    if len(tri_k_list) == 1:
        tri_uniq_keys = tri_k_list[0]
        tri_uniq_counts = tri_c_list[0]
    else:
        cat_tk = torch.cat(tri_k_list)
        cat_tc = torch.cat(tri_c_list)
        ord_t = torch.argsort(cat_tk)
        cat_tk = cat_tk[ord_t]
        cat_tc = cat_tc[ord_t]
        tri_uniq_keys, inv_t = torch.unique_consecutive(cat_tk, return_inverse=True)
        tri_uniq_counts = torch.zeros(tri_uniq_keys.shape[0], dtype=torch.float64, device=device)
        tri_uniq_counts.scatter_add_(0, inv_t, cat_tc)

    tri_ctx2 = torch.div(tri_uniq_keys, v_size, rounding_mode="floor")
    tri_y = tri_uniq_keys % v_size
    ctx2_uniq_keys, inv_c2 = torch.unique_consecutive(tri_ctx2, return_inverse=True)
    ctx2_uniq_counts = torch.zeros(ctx2_uniq_keys.shape[0], dtype=torch.float64, device=device)
    ctx2_uniq_counts.scatter_add_(0, inv_c2, tri_uniq_counts)

    # Compute trigram argmax candidate per observed `ctx2`
    tri_w2 = tri_ctx2 % v_size
    bi_lookup_keys = tri_w2 * v_size + tri_y
    b_idx = torch.searchsorted(bi_uniq_keys, bi_lookup_keys)
    b_match = (b_idx < bi_uniq_keys.numel()) & (
        bi_uniq_keys[b_idx.clamp_max(max(1, bi_uniq_keys.numel()) - 1)] == bi_lookup_keys
    )
    c_bi_for_tri = torch.where(
        b_match, bi_uniq_counts[b_idx.clamp_max(max(1, bi_uniq_keys.numel()) - 1)], 0.0
    )
    p_bi_for_tri = (c_bi_for_tri + beta_2 * p_uni[tri_y]) / (ctx1_totals[tri_w2] + beta_2)
    tri_scores = tri_uniq_counts + beta_3 * p_bi_for_tri
    tri_best_ctx2_keys, tri_best_y = _segment_argmax_by_sorted_key(tri_ctx2, tri_y, tri_scores)

    model = VectorizedNGramModel(
        p_uni=p_uni,
        top50_uni_ids=top50_uni_ids,
        ctx1_totals=ctx1_totals,
        bi_uniq_keys=bi_uniq_keys,
        bi_uniq_counts=bi_uniq_counts,
        bi_best_y_per_x=bi_best_y_per_x,
        tri_uniq_keys=tri_uniq_keys,
        tri_uniq_counts=tri_uniq_counts,
    )
    tri_tbl = TrigramContextTable(
        ctx2_uniq_keys=ctx2_uniq_keys,
        ctx2_uniq_counts=ctx2_uniq_counts,
        tri_best_ctx2_keys=tri_best_ctx2_keys,
        tri_best_y=tri_best_y,
        beta_2=beta_2,
        beta_3=beta_3,
    )
    return model, tri_tbl


def _lookup_sorted_counts(
    sorted_keys: torch.Tensor,
    sorted_vals: torch.Tensor,
    query_keys: torch.Tensor,
    default_val: float | int = 0,
) -> torch.Tensor:
    """Vectorized binary search lookup in `sorted_keys` (`<= 8` args)."""
    if sorted_keys.numel() == 0:
        return torch.full_like(query_keys, default_val, dtype=sorted_vals.dtype)
    idx = torch.searchsorted(sorted_keys, query_keys)
    clamped = idx.clamp_max(sorted_keys.numel() - 1)
    matched = (idx < sorted_keys.numel()) & (sorted_keys[clamped] == query_keys)
    return torch.where(
        matched,
        sorted_vals[clamped],
        torch.as_tensor(default_val, dtype=sorted_vals.dtype, device=sorted_vals.device),
    )


def evaluate_val_slice_vectorized(
    val_tokens_np: np.ndarray,
    model: VectorizedNGramModel,
    tri_tbl: TrigramContextTable,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate exact Unigram, Bigram, and Trigram CE, PPL, and Top-1 on `val_tokens_np` (`<= 8` args)."""
    v_size = VALID_VOCAB_SIZE
    val_clip = np.clip(val_tokens_np, 0, v_size - 1).astype(np.int64)
    n_pred = int(val_clip.shape[0]) - 2

    uni_nll_sum = 0.0
    uni_top1_hits = 0
    uni_top50_hits = 0

    bi_nll_sum = 0.0
    bi_top1_hits = 0
    bi_exact_hits = 0

    tri_nll_sum = 0.0
    tri_top1_hits = 0
    tri_exact_hits = 0

    top1_uni_id = int(model.top50_uni_ids[0].item())

    for start in range(0, n_pred, EVAL_CHUNK_SIZE):
        end = min(n_pred, start + EVAL_CHUNK_SIZE)
        w1 = torch.from_numpy(val_clip[start:end]).to(device=device)
        w2 = torch.from_numpy(val_clip[start + 1 : end + 1]).to(device=device)
        y = torch.from_numpy(val_clip[start + 2 : end + 2]).to(device=device)

        # 1. Unigram
        p_u = model.p_uni[y]
        uni_nll_sum += float((-torch.log(p_u.clamp_min(1e-15))).sum().item())
        uni_top1_hits += int((y == top1_uni_id).sum().item())
        uni_top50_hits += int(torch.isin(y, model.top50_uni_ids).sum().item())

        # 2. Bigram P_bi(y | w2)
        bi_k = w2 * v_size + y
        c_bi = _lookup_sorted_counts(model.bi_uniq_keys, model.bi_uniq_counts, bi_k, 0.0)
        bi_exact_hits += int((c_bi > 0).sum().item())
        p_bi = (c_bi + tri_tbl.beta_2 * p_u) / (model.ctx1_totals[w2] + tri_tbl.beta_2)
        bi_nll_sum += float((-torch.log(p_bi.clamp_min(1e-15))).sum().item())
        bi_pred_1 = model.bi_best_y_per_x[w2]
        bi_top1_hits += int((y == bi_pred_1).sum().item())

        # 3. Trigram P_tri(y | w1, w2): interpolated Dirichlet + Jelinek-Mercer backoff to P_bi(y | w2)
        ctx2_k = w1 * v_size + w2
        tri_k = ctx2_k * v_size + y
        c_tri = _lookup_sorted_counts(model.tri_uniq_keys, model.tri_uniq_counts, tri_k, 0.0)
        c_ctx2 = _lookup_sorted_counts(tri_tbl.ctx2_uniq_keys, tri_tbl.ctx2_uniq_counts, ctx2_k, 0.0)
        tri_exact_hits += int((c_tri > 0).sum().item())
        p_tri_dir = (c_tri + tri_tbl.beta_3 * p_bi) / (c_ctx2 + tri_tbl.beta_3)
        p_tri = (1.0 - DEFAULT_TRIGRAM_JM_LAMBDA) * p_tri_dir + DEFAULT_TRIGRAM_JM_LAMBDA * p_bi
        tri_nll_sum += float((-torch.log(p_tri.clamp_min(1e-15))).sum().item())

        # Trigram Top-1 prediction: observed trigram winner if ctx2 seen, else bigram winner
        tri_obs_y = _lookup_sorted_counts(tri_tbl.tri_best_ctx2_keys, tri_tbl.tri_best_y, ctx2_k, -1)
        tri_pred_1 = torch.where(tri_obs_y >= 0, tri_obs_y, bi_pred_1)
        tri_top1_hits += int((y == tri_pred_1).sum().item())

    uni_ce = uni_nll_sum / max(1, n_pred)
    bi_ce = bi_nll_sum / max(1, n_pred)
    tri_ce = tri_nll_sum / max(1, n_pred)
    return {
        "evaluated_tokens": n_pred,
        "unigram": {
            "val_ce": round(uni_ce, 4),
            "val_ppl": round(math.exp(min(20.0, uni_ce)), 2),
            "top1_acc": round(100.0 * uni_top1_hits / max(1, n_pred), 2),
            "top50_acc": round(100.0 * uni_top50_hits / max(1, n_pred), 2),
        },
        "bigram": {
            "beta_2": tri_tbl.beta_2,
            "val_ce": round(bi_ce, 4),
            "val_ppl": round(math.exp(min(20.0, bi_ce)), 2),
            "top1_acc": round(100.0 * bi_top1_hits / max(1, n_pred), 2),
            "exact_hit_rate_pct": round(100.0 * bi_exact_hits / max(1, n_pred), 2),
        },
        "trigram": {
            "beta_2": tri_tbl.beta_2,
            "beta_3": tri_tbl.beta_3,
            "val_ce": round(tri_ce, 4),
            "val_ppl": round(math.exp(min(20.0, tri_ce)), 2),
            "top1_acc": round(100.0 * tri_top1_hits / max(1, n_pred), 2),
            "exact_hit_rate_pct": round(100.0 * tri_exact_hits / max(1, n_pred), 2),
        },
    }


def main() -> None:
    """Download/load full 100M-token Shard 0 (`fineweb_val_000000.bin`) and benchmark N-gram baselines."""
    parser = argparse.ArgumentParser(description="Vectorized Unigram, Bigram, and Trigram baselines on FineWeb.")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT, help="Output JSON report path.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shard1 = DATA_DIR / "train_full_100M.bin"
    if not shard1.exists():
        shard1 = DATA_DIR / "fineweb_train_000001.bin"
    full_val_path = ensure_full_val_shard0(DATA_DIR)

    tokens_100m = _load_header_safe_shard_tokens(shard1, max_tokens=SLICE_100M_TOKENS)
    val_100m = _load_header_safe_shard_tokens(full_val_path, max_tokens=VAL_FULL_100M_TOKENS)

    val_slices = {
        "65.5K_fast_screening": val_100m[: VAL_FAST_65K_TOKENS + 2],
        "10.49M_modded_nanogpt": val_100m[: VAL_MODDED_10M_TOKENS + 2],
        "100.0M_full_shard0": val_100m,
    }
    train_slices = {
        "2.46M_tokens_300steps": tokens_100m[:SLICE_2P46M_TOKENS],
        "100.0M_tokens_shard1": tokens_100m,
    }

    report: dict[str, Any] = {"results": {}}
    for tr_name, tr_arr in train_slices.items():
        t0 = time.perf_counter()
        print(f"[Fitting N-Gram Tables] Training slice `{tr_name}` ({tr_arr.shape[0]:,} tokens) ...", flush=True)
        model, tri_tbl = fit_vectorized_ngram_tables(tr_arr, val_100m, device)
        fit_s = time.perf_counter() - t0
        tr_entry: dict[str, Any] = {
            "train_tokens": int(tr_arr.shape[0]),
            "fit_sec": round(fit_s, 2),
            "unique_bigrams_M": round(float(model.bi_uniq_keys.numel()) / 1e6, 3),
            "val_Matched_trigrams_M": round(float(model.tri_uniq_keys.numel()) / 1e6, 3),
            "validation_scopes": {},
        }
        for val_name, val_arr in val_slices.items():
            ev_res = evaluate_val_slice_vectorized(val_arr, model, tri_tbl, device)
            tr_entry["validation_scopes"][val_name] = ev_res
            print(
                f"  -> Val `{val_name}` ({ev_res['evaluated_tokens']:,} tok): "
                f"Uni CE={ev_res['unigram']['val_ce']:.4f} (Top1={ev_res['unigram']['top1_acc']:.2f}%) | "
                f"Bi CE={ev_res['bigram']['val_ce']:.4f} (Top1={ev_res['bigram']['top1_acc']:.2f}%) | "
                f"Tri CE={ev_res['trigram']['val_ce']:.4f} (Top1={ev_res['trigram']['top1_acc']:.2f}%)",
                flush=True,
            )
        report["results"][tr_name] = tr_entry
        del model, tri_tbl
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[Saved N-Gram Baseline Report] -> {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
