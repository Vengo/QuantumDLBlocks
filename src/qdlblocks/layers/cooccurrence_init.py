"""9-Shard (900M-Token) GPU Shifted PPMI + Randomized SVD Co-Occurrence Spectral Initialization.

Implements Section 4.1.1 of EXPERIMENT_PLAN_021.md:
- `compute_ppmi_svd_embeddings`: Single-pass vectorized GPU co-occurrence sketch across multi-shard
  token streams (`uint16` / `int64`), Context-Distribution-Smoothed Shifted PPMI (`cds_alpha=0.75`),
  deterministic GPU Randomized SVD (`torch.svd_lowrank`) with deterministic column sign convention
  and `Sigma^{1/2}` spectral weighting, and atomic `.pt` disk caching.
- `initialize_quantum_born_with_ppmi_svd`: Initializes `QuantumBornGPT` embeddings (`dm_seed`, `wte`,
  `povm_head.effects`, `value_embeds`) with the PPMI-SVD spectral basis while strictly preserving
  exact Step-0 input-output subspace orthogonality (`max_w |<m_w | psi_0(w)>| < 1e-6`) when
  `use_ortho_decouple=True`.
- `initialize_classical_with_ppmi_svd`: Initializes `ClassicalModdedGPT` token embeddings (`wte`)
  while preserving target per-row standard deviation.
- `compute_collocation_coherence_metrics`: Evaluates Step-0 cosine coherence and purified Born
  attention energy on top collocated token pairs vs. random token pairs.
"""

from __future__ import annotations

from collections.abc import Iterable
import itertools
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import l2_normalize
from .early_epoch_quantum import purified_born_attention_weights


def _slice_to_device_int64(
    arr_slice: np.ndarray | torch.Tensor,
    dev: torch.device,
    vocab_size: int,
) -> torch.Tensor:
    """Transfer a token slice to `dev` in compact 16/32-bit representation before casting to int64 on `dev`."""
    if isinstance(arr_slice, torch.Tensor):
        t = arr_slice.to(device=dev, non_blocking=True)
        if t.dtype != torch.int64:
            t = t.to(torch.int64)
        return t.reshape(-1).clamp_(0, vocab_size - 1)

    np_arr = np.asarray(arr_slice)
    if np_arr.dtype == np.uint16:
        # Transfer as 2-byte uint16/int16 over PCIe first (4x less H2D traffic than CPU int64),
        # then reconstruct unsigned values in int64 directly on the target device.
        if not np_arr.flags.c_contiguous or not np_arr.flags.writeable:
            np_arr = np.ascontiguousarray(np_arr)
        t_i16 = torch.from_numpy(np_arr.view(np.int16)).to(device=dev, non_blocking=True)
        t_i64 = t_i16.to(torch.int64) & 0xFFFF
        return t_i64.reshape(-1).clamp_(0, vocab_size - 1)

    if not np_arr.flags.c_contiguous or not np_arr.flags.writeable:
        np_arr = np.ascontiguousarray(np_arr)
    t_any = torch.from_numpy(np_arr).to(device=dev, non_blocking=True).to(torch.int64)
    return t_any.reshape(-1).clamp_(0, vocab_size - 1)


@torch.no_grad()
def compute_ppmi_svd_embeddings(
    token_arrays: Iterable[np.ndarray | torch.Tensor],
    vocab_size: int = 50304,
    embed_dim: int = 1024,
    num_context_buckets: int = 2048,
    window_size: int = 4,
    cds_alpha: float = 0.75,
    shift_kappa: float = 1.0,
    svd_oversample: int = 32,
    svd_niter: int = 4,
    chunk_tokens: int = 25_000_000,
    device: torch.device | str | None = None,
    cache_path: Path | str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute (or load from disk cache) unit-normalized Shifted PPMI + Randomized SVD token embeddings."""
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
    if embed_dim < 1:
        raise ValueError(f"embed_dim must be >= 1, got {embed_dim}")
    if num_context_buckets < 2:
        raise ValueError(f"num_context_buckets must be >= 2, got {num_context_buckets}")
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    dev = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    t0 = time.perf_counter()

    # 1. Disk Cache Check
    if cache_path is not None:
        cp = Path(cache_path)
        if cp.exists() and cp.stat().st_size > 0:
            cached = torch.load(cp, map_location=dev, weights_only=True)
            if (
                isinstance(cached, dict)
                and "embeddings" in cached
                and tuple(cached["embeddings"].shape) == (int(vocab_size), int(embed_dim))
            ):
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                load_s = float(time.perf_counter() - t0)
                out_cached = dict(cached)
                out_cached["elapsed_s"] = load_s
                out_cached["loaded_from_cache"] = True
                return out_cached

    it = iter(token_arrays)
    try:
        first_shard = next(it)
    except StopIteration:
        raise ValueError("token_arrays must contain at least one non-empty token array")

    all_shards = itertools.chain([first_shard], it)
    B = min(int(vocab_size), int(num_context_buckets))

    # 2. Deterministic Context Bucket Mapping (bucket_map: (vocab_size,) int64)
    if vocab_size <= num_context_buckets:
        bucket_map = torch.arange(vocab_size, device=dev, dtype=torch.int64)
    else:
        sample_len = min(5_000_000, len(first_shard))
        sample_i64 = _slice_to_device_int64(first_shard[:sample_len], dev=dev, vocab_size=vocab_size)
        sample_counts = torch.bincount(sample_i64, minlength=vocab_size)
        b_exact = B // 2
        b_rem = max(1, B - b_exact)
        top_ids = torch.argsort(-sample_counts, stable=True)[:b_exact]
        v_ids = torch.arange(vocab_size, device=dev, dtype=torch.int64)
        bucket_map = b_exact + ((v_ids * 36313) % b_rem)
        bucket_map[top_ids] = torch.arange(b_exact, device=dev, dtype=torch.int64)

    # 3. Vectorized Single-Pass Co-Occurrence Accumulation
    counts_flat = torch.zeros(vocab_size * B, dtype=torch.float32, device=dev)
    unigram_counts = torch.zeros(vocab_size, dtype=torch.float64, device=dev)
    total_tokens = 0
    chunk_sz = max(1024, int(chunk_tokens))

    # Pre-allocate weight buffer on CUDA to avoid allocating a 103M-element int64 bincount tensor per offset
    use_scatter_accum = (dev.type == "cuda") and (vocab_size * B >= 1_000_000)
    w_buf: torch.Tensor | None = None

    for shard in all_shards:
        shard_len = len(shard)
        for start in range(0, shard_len, chunk_sz):
            end = min(start + chunk_sz, shard_len)
            chunk_i64 = _slice_to_device_int64(shard[start:end], dev=dev, vocab_size=vocab_size)
            n_tok = int(chunk_i64.numel())
            if n_tok == 0:
                continue
            total_tokens += n_tok
            unigram_counts.add_(torch.bincount(chunk_i64, minlength=vocab_size).to(torch.float64))
            if n_tok <= 1:
                continue

            ctx_b = bucket_map[chunk_i64]
            if use_scatter_accum:
                if w_buf is None or w_buf.numel() < n_tok:
                    w_buf = torch.empty(n_tok, dtype=torch.float32, device=dev)
                for delta in range(1, window_size + 1):
                    if n_tok <= delta:
                        break
                    n_pairs = n_tok - delta
                    w_delta = 1.0 / float(delta)
                    w_slice = w_buf[:n_pairs]
                    w_slice.fill_(w_delta)
                    flat_fwd = chunk_i64[:-delta] * B + ctx_b[delta:]
                    flat_bwd = chunk_i64[delta:] * B + ctx_b[:-delta]
                    counts_flat.scatter_add_(0, flat_fwd, w_slice)
                    counts_flat.scatter_add_(0, flat_bwd, w_slice)
            else:
                for delta in range(1, window_size + 1):
                    if n_tok <= delta:
                        break
                    w_delta = 1.0 / float(delta)
                    flat_fwd = chunk_i64[:-delta] * B + ctx_b[delta:]
                    flat_bwd = chunk_i64[delta:] * B + ctx_b[:-delta]
                    counts_flat.add_(
                        torch.bincount(flat_fwd, minlength=vocab_size * B).to(torch.float32),
                        alpha=w_delta,
                    )
                    counts_flat.add_(
                        torch.bincount(flat_bwd, minlength=vocab_size * B).to(torch.float32),
                        alpha=w_delta,
                    )

    # 4. Shifted Positive Pointwise Mutual Information (SPPMI) Transform
    C = counts_flat.view(vocab_size, B)
    r_w = C.sum(dim=1, keepdim=True)  # (vocab_size, 1)
    c_b = C.sum(dim=0, keepdim=True)  # (1, B)

    c_pow = c_b.clamp_min(0.0).pow(float(cds_alpha))
    p_alpha = c_pow / (c_pow.sum() + 1e-12)  # (1, B)

    kappa = max(float(shift_kappa), 1e-6)
    expected_c = r_w.clamp_min(1e-12) * p_alpha * kappa + 1e-12
    sppmi = torch.log(C.clamp_min(1e-12) / expected_c).clamp_min_(0.0)

    active_rows = r_w.squeeze(-1) > 0.0
    sppmi[~active_rows] = 0.0

    # Anchor each active token w in its own context bucket b(w) so the Gram matrix
    # (I + A)(I + A)^T = I + A + A^T + A A^T captures both 1st-order adjacent collocations
    # and 2nd-order shared contexts in the top singular vectors.
    if active_rows.any():
        active_idx = torch.nonzero(active_rows, as_tuple=False).squeeze(-1)
        row_peaks = sppmi[active_idx].amax(dim=-1).clamp_min(1.0)
        sppmi[active_idx, bucket_map[active_idx]] += row_peaks

    # 5. Deterministic GPU Randomized SVD (torch.svd_lowrank)
    k_svd = min(int(embed_dim), int(vocab_size), int(B))
    q_svd = min(int(vocab_size), int(B), k_svd + max(0, int(svd_oversample)))

    fork_devs = [dev] if dev.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devs):
        torch.manual_seed(int(seed))
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        U, S, _ = torch.svd_lowrank(sppmi, q=q_svd, niter=max(1, int(svd_niter)))

    u_k = U[:, :k_svd]
    s_k = S[:k_svd].clamp_min(1e-12)

    # Deterministic Sign Convention: sign of argmax_i |U[i, j]| is positive (+1)
    max_row_idx = torch.argmax(u_k.abs(), dim=0)
    col_indices = torch.arange(k_svd, device=dev)
    col_signs = torch.where(u_k[max_row_idx, col_indices] >= 0.0, 1.0, -1.0)
    u_k = u_k * col_signs.unsqueeze(0)

    # Weight by Sigma^{1/2}
    u_weighted = u_k * torch.sqrt(s_k).unsqueeze(0)
    if k_svd < embed_dim:
        u_weighted = F.pad(u_weighted, (0, int(embed_dim) - k_svd))

    embeddings = torch.zeros((vocab_size, embed_dim), dtype=torch.float32, device=dev)
    if active_rows.any():
        embeddings[active_rows] = l2_normalize(u_weighted[active_rows], dim=-1)
    if (~active_rows).any():
        n_inactive = int((~active_rows).sum().item())
        g_inact = torch.Generator(device="cpu").manual_seed(int(seed) + 17)
        z_inact = torch.randn((n_inactive, embed_dim), generator=g_inact, dtype=torch.float32).to(dev)
        embeddings[~active_rows] = l2_normalize(z_inact, dim=-1)

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elapsed_s = float(time.perf_counter() - t0)

    result: dict[str, Any] = {
        "embeddings": embeddings,
        "singular_values": s_k,
        "unigram_counts": unigram_counts,
        "bucket_map": bucket_map,
        "total_tokens": int(total_tokens),
        "elapsed_s": elapsed_s,
        "loaded_from_cache": False,
    }

    if cache_path is not None:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp_cp = cp.with_suffix(cp.suffix + ".tmp")
        torch.save(result, tmp_cp)
        tmp_cp.replace(cp)

    return result


def _adapt_ppmi_shape(
    ppmi_embeddings: torch.Tensor,
    target_vocab: int,
    target_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Slice or zero-pad `ppmi_embeddings` to `(target_vocab, target_dim)` on `device`."""
    emb = ppmi_embeddings.detach().to(device=device, dtype=torch.float32)
    v_src, d_src = emb.shape
    if v_src < target_vocab:
        emb = F.pad(emb, (0, 0, 0, target_vocab - v_src))
    elif v_src > target_vocab:
        emb = emb[:target_vocab]
    if d_src < target_dim:
        emb = F.pad(emb, (0, target_dim - d_src))
    elif d_src > target_dim:
        emb = emb[:, :target_dim]
    return emb


@torch.no_grad()
def initialize_quantum_born_with_ppmi_svd(
    model: nn.Module,
    ppmi_embeddings: torch.Tensor | None = None,
    alpha: float = 0.65,
    seed: int = 42,
    *,
    token_shards: Sequence[np.ndarray | torch.Tensor] | None = None,
    alpha_ppmi: float | None = None,
    cache_path: str | Path | None = None,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Initialize `QuantumBornGPT` embeddings (`dm_seed`, `wte`, `povm_head.effects`, `value_embeds`) with PPMI-SVD.

    When `model.use_ortho_decouple` is True:
    - Places `(1 - alpha) * z_even + alpha * c_half` on even coordinates `0::2` of `dm_seed.weight`.
    - Places `(1 - alpha) * z_odd + alpha * c_half` on odd coordinates `1::2` of `povm_head.effects`.
    - Guarantees exact Step-0 input-output orthogonality `max_w |<m_w | psi_0(w)>| == 0 < 1e-6`
      while endowing co-occurring tokens with strong Step-0 inner-product coherence.
    """
    eff_alpha = float(alpha_ppmi if alpha_ppmi is not None else alpha)
    alpha_f = float(min(max(eff_alpha, 0.0), 1.0))

    # Resolve target device/dtype from model parameters
    first_param = next(model.parameters())
    dev = torch.device(device) if device is not None else first_param.device
    param_dtype = first_param.dtype

    if ppmi_embeddings is None:
        if token_shards is None:
            raise ValueError("Either `ppmi_embeddings` or `token_shards` must be provided")
        V_model = int(getattr(model, "vocab_size", 50304))
        d_model = int(getattr(model, "embed_dim", 512))
        use_ortho_flag = bool(getattr(model, "use_ortho_decouple", False))
        svd_dim = (d_model // 2) if use_ortho_flag else d_model
        ppmi_res = compute_ppmi_svd_embeddings(
            token_arrays=token_shards,
            vocab_size=V_model,
            embed_dim=svd_dim,
            seed=seed,
            device=dev,
            cache_path=cache_path,
        )
        ppmi_embeddings = ppmi_res["embeddings"]

    V = int(getattr(model, "vocab_size", ppmi_embeddings.size(0)))
    d = int(getattr(model, "embed_dim", ppmi_embeddings.size(1)))
    if d % 2 != 0:
        raise ValueError(f"embed_dim must be even for QuantumBornGPT PPMI-SVD initialization, got {d}")

    emb_vd = _adapt_ppmi_shape(ppmi_embeddings, target_vocab=V, target_dim=d, device=dev)
    active_mask = emb_vd.norm(dim=-1) > 1e-6
    active_frac = float(active_mask.float().mean().item())

    use_ortho = bool(getattr(model, "use_ortho_decouple", False))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    # Ensure model.dm_seed is available as a full-width (V, d) embedding when in asymmetric mode
    dm_seed_mod = getattr(model, "dm_seed", None)
    wte_mod = getattr(model, "wte", None)
    w_lift_mod = getattr(model, "W_lift", None)

    if (
        getattr(model, "readout_mode", "asymmetric") == "asymmetric"
        and wte_mod is not None
        and wte_mod.weight.size(1) != d
        and getattr(model, "seed_adapter", None) is None
    ):
        # Upgrade seed_dim to full embed_dim so dm_seed / wte holds full-width (V, d) states
        wte_mod.weight = nn.Parameter(torch.empty((V, d), device=dev, dtype=param_dtype))
        wte_mod.embedding_dim = d
        if hasattr(model, "hilbert_lift"):
            model.hilbert_lift.seed_dim = d
        model.seed_dim = d
        if w_lift_mod is not None:
            w_lift_mod.weight = nn.Parameter(torch.eye(d, device=dev, dtype=param_dtype))
            w_lift_mod.in_features = d
            w_lift_mod.out_features = d
        dm_seed_mod = wte_mod
        setattr(model, "dm_seed", dm_seed_mod)

    # Resolve POVM effects parameter (untied asymmetric head)
    effects_param: nn.Parameter | None = None
    povm_head_attr = getattr(model, "povm_head", None)
    if isinstance(povm_head_attr, nn.Parameter):
        effects_param = povm_head_attr
    elif povm_head_attr is not None and isinstance(getattr(povm_head_attr, "effects", None), nn.Parameter):
        effects_param = povm_head_attr.effects
    elif getattr(model, "povm_readout", None) is not None:
        pr = getattr(model, "povm_readout")
        if isinstance(getattr(pr, "povm_head", None), nn.Parameter):
            effects_param = pr.povm_head

    if use_ortho:
        d_half = d // 2
        c_half = torch.zeros((V, d_half), dtype=torch.float32, device=dev)
        if active_mask.any():
            c_half[active_mask] = l2_normalize(emb_vd[active_mask, :d_half], dim=-1)

        z_even = l2_normalize(
            torch.randn((V, d_half), generator=gen, dtype=torch.float32).to(dev),
            dim=-1,
        )
        z_odd = l2_normalize(
            torch.randn((V, d_half), generator=gen, dtype=torch.float32).to(dev),
            dim=-1,
        )

        u_even = l2_normalize((1.0 - alpha_f) * z_even + alpha_f * c_half, dim=-1)
        u_odd = l2_normalize((1.0 - alpha_f) * z_odd + alpha_f * c_half, dim=-1)

        u_full_in = torch.zeros((V, d), dtype=torch.float32, device=dev)
        u_full_in[:, 0::2] = u_even

        u_full_out = torch.zeros((V, d), dtype=torch.float32, device=dev)
        u_full_out[:, 1::2] = u_odd

        if dm_seed_mod is not None and hasattr(dm_seed_mod, "weight") and dm_seed_mod.weight.shape == (V, d):
            dm_seed_mod.weight.zero_()
            dm_seed_mod.weight[:, 0::2].copy_(u_even.to(dm_seed_mod.weight.dtype))

        if wte_mod is not None and hasattr(wte_mod, "weight"):
            if wte_mod.weight.shape == (V, d):
                wte_mod.weight.zero_()
                wte_mod.weight[:, 0::2].copy_(u_even.to(wte_mod.weight.dtype))
                if w_lift_mod is not None and w_lift_mod.weight.shape == (d, d):
                    w_lift_mod.weight.copy_(torch.eye(d, device=dev, dtype=w_lift_mod.weight.dtype))
            else:
                s_dim = wte_mod.weight.size(1)
                c_seed = torch.zeros((V, s_dim), dtype=torch.float32, device=dev)
                if active_mask.any():
                    c_seed[active_mask] = l2_normalize(emb_vd[active_mask, :s_dim], dim=-1)
                z_seed = l2_normalize(
                    torch.randn((V, s_dim), generator=gen, dtype=torch.float32).to(dev),
                    dim=-1,
                )
                u_seed = l2_normalize((1.0 - alpha_f) * z_seed + alpha_f * c_seed, dim=-1)
                wte_mod.weight.copy_(u_seed.to(wte_mod.weight.dtype))
                if w_lift_mod is not None and w_lift_mod.weight.shape == (d, s_dim):
                    w_lift_mod.weight.zero_()
                    n_map = min(d_half, s_dim)
                    w_lift_mod.weight[0 : 2 * n_map : 2, :n_map] = torch.eye(
                        n_map, device=dev, dtype=w_lift_mod.weight.dtype
                    )

        if effects_param is not None and effects_param.shape == (V, d):
            effects_param.zero_()
            effects_param[:, 1::2].copy_(u_odd.to(effects_param.dtype))
    else:
        c_full = torch.zeros((V, d), dtype=torch.float32, device=dev)
        if active_mask.any():
            c_full[active_mask] = l2_normalize(emb_vd[active_mask], dim=-1)
        z_full = l2_normalize(
            torch.randn((V, d), generator=gen, dtype=torch.float32).to(dev),
            dim=-1,
        )
        u_full_in = l2_normalize((1.0 - alpha_f) * z_full + alpha_f * c_full, dim=-1)
        u_full_out = u_full_in

        if dm_seed_mod is not None and hasattr(dm_seed_mod, "weight") and dm_seed_mod.weight.shape == (V, d):
            dm_seed_mod.weight.copy_(u_full_in.to(dm_seed_mod.weight.dtype))
        if wte_mod is not None and hasattr(wte_mod, "weight"):
            if wte_mod.weight.shape == (V, d):
                wte_mod.weight.copy_(u_full_in.to(wte_mod.weight.dtype))
                if w_lift_mod is not None and w_lift_mod.weight.shape == (d, d):
                    w_lift_mod.weight.copy_(torch.eye(d, device=dev, dtype=w_lift_mod.weight.dtype))
            else:
                s_dim = wte_mod.weight.size(1)
                c_seed = torch.zeros((V, s_dim), dtype=torch.float32, device=dev)
                if active_mask.any():
                    c_seed[active_mask] = l2_normalize(emb_vd[active_mask, :s_dim], dim=-1)
                z_seed = l2_normalize(
                    torch.randn((V, s_dim), generator=gen, dtype=torch.float32).to(dev),
                    dim=-1,
                )
                u_seed = l2_normalize((1.0 - alpha_f) * z_seed + alpha_f * c_seed, dim=-1)
                wte_mod.weight.copy_(u_seed.to(wte_mod.weight.dtype))
        if effects_param is not None and effects_param.shape == (V, d):
            effects_param.copy_(u_full_out.to(effects_param.dtype))

    # Initialize multi-bank value embeddings from input seed state
    value_embeds = getattr(model, "value_embeds", None)
    if value_embeds is not None:
        for ve_emb in value_embeds:
            if hasattr(ve_emb, "weight") and ve_emb.weight.shape == (V, d):
                ve_emb.weight.copy_(u_full_in.to(ve_emb.weight.dtype))

    # Compute exact Step-0 max self-overlap max_w |<m_w | psi_0(w)>|
    if dm_seed_mod is not None and hasattr(dm_seed_mod, "weight") and dm_seed_mod.weight.shape == (V, d):
        psi_0_all = l2_normalize(dm_seed_mod.weight.float(), dim=-1)
    else:
        psi_0_all = l2_normalize(u_full_in.float(), dim=-1)

    if effects_param is not None and effects_param.shape == (V, d):
        m_all = l2_normalize(effects_param.float(), dim=-1)
    else:
        m_all = l2_normalize(u_full_out.float(), dim=-1)

    max_self_overlap = float((psi_0_all * m_all).sum(dim=-1).abs().max().item())
    row_norms = psi_0_all.norm(dim=-1)
    return {
        "max_self_overlap": max_self_overlap,
        "min_row_norm": float(row_norms.min().item()),
        "max_row_norm": float(row_norms.max().item()),
        "active_vocab_fraction": active_frac,
        "alpha": alpha_f,
    }


@torch.no_grad()
def initialize_classical_with_ppmi_svd(
    model: nn.Module,
    ppmi_embeddings: torch.Tensor | None = None,
    alpha: float = 0.50,
    seed: int = 42,
    *,
    token_shards: Sequence[np.ndarray | torch.Tensor] | None = None,
    alpha_ppmi: float | None = None,
    cache_path: str | Path | None = None,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Initialize `ClassicalModdedGPT` token embeddings (`wte`) with PPMI-SVD while preserving row std."""
    wte = getattr(model, "wte", None)
    if wte is None or not hasattr(wte, "weight"):
        raise ValueError("Model must have a `wte` embedding layer with `.weight`")

    eff_alpha = float(alpha_ppmi if alpha_ppmi is not None else alpha)
    alpha_f = float(min(max(eff_alpha, 0.0), 1.0))
    w_orig = wte.weight.detach().float()
    V, d = w_orig.shape
    dev = torch.device(device) if device is not None else wte.weight.device

    if ppmi_embeddings is None:
        if token_shards is None:
            raise ValueError("Either `ppmi_embeddings` or `token_shards` must be provided")
        ppmi_res = compute_ppmi_svd_embeddings(
            token_arrays=token_shards,
            vocab_size=V,
            embed_dim=d,
            seed=seed,
            device=dev,
            cache_path=cache_path,
        )
        ppmi_embeddings = ppmi_res["embeddings"]

    target_std = float(w_orig.std().item())
    if target_std <= 1e-8:
        target_std = 1.0 if bool(getattr(model, "leaderboard_exact", False)) else (1.0 / math.sqrt(d))

    emb_vd = _adapt_ppmi_shape(ppmi_embeddings, target_vocab=V, target_dim=d, device=dev)
    active_mask = emb_vd.norm(dim=-1) > 1e-6
    active_frac = float(active_mask.float().mean().item())

    c_full = torch.zeros((V, d), dtype=torch.float32, device=dev)
    if active_mask.any():
        c_full[active_mask] = l2_normalize(emb_vd[active_mask], dim=-1) * math.sqrt(float(d))

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    z_std = torch.randn((V, d), generator=gen, dtype=torch.float32).to(dev)

    w_new = z_std.clone()
    if active_mask.any():
        z_scale = math.sqrt(max(0.0, 1.0 - alpha_f * alpha_f))
        w_new[active_mask] = z_scale * z_std[active_mask] + alpha_f * c_full[active_mask]
    w_new.mul_(target_std)

    wte.weight.copy_(w_new.to(wte.weight.dtype))
    new_std = float(wte.weight.detach().float().std().item())
    return {
        "target_std": target_std,
        "new_std": new_std,
        "wte_rms": new_std,
        "active_vocab_fraction": active_frac,
        "alpha": alpha_f,
    }


@torch.no_grad()
def compute_collocation_coherence_metrics(
    tokens_1d: np.ndarray | torch.Tensor,
    embeddings: torch.Tensor,
    top_k: int = 256,
    tau_h: float = 0.0625,
    gamma_h: float = 8.0,
    eps_base: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Compare Step-0 state cosine coherence and purified Born energy on top collocations vs random pairs."""
    emb_norm = l2_normalize(embeddings.detach().float(), dim=-1)
    V = emb_norm.size(0)
    dev = emb_norm.device

    sample_len = min(2_000_000, len(tokens_1d))
    tok_i64 = _slice_to_device_int64(tokens_1d[:sample_len], dev=dev, vocab_size=V)
    if tok_i64.numel() < 2:
        raise ValueError("tokens_1d must contain at least 2 tokens")

    u_tok = tok_i64[:-1]
    v_tok = tok_i64[1:]
    neq_mask = u_tok != v_tok
    u_tok = u_tok[neq_mask]
    v_tok = v_tok[neq_mask]

    pair_keys = u_tok * V + v_tok
    uniq_keys, counts = torch.unique(pair_keys, return_counts=True)
    k_eff = min(int(top_k), int(uniq_keys.numel()))
    top_idx = torch.argsort(-counts, stable=True)[:k_eff]
    top_pairs = uniq_keys[top_idx]
    colloc_u = torch.div(top_pairs, V, rounding_mode="floor")
    colloc_v = top_pairs % V

    # Active vocabulary for random baseline pairs
    active_ids = torch.unique(tok_i64)
    if active_ids.numel() < 4:
        active_ids = torch.arange(V, device=dev, dtype=torch.int64)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    rand_u_idx = torch.randint(0, active_ids.numel(), (k_eff,), generator=gen).to(dev)
    rand_v_idx = (rand_u_idx + torch.randint(1, max(2, active_ids.numel() - 1), (k_eff,), generator=gen).to(dev)) % active_ids.numel()
    rand_u = active_ids[rand_u_idx]
    rand_v = active_ids[rand_v_idx]

    colloc_dots = (emb_norm[colloc_u] * emb_norm[colloc_v]).sum(dim=-1)
    rand_dots = (emb_norm[rand_u] * emb_norm[rand_v]).sum(dim=-1)

    colloc_mean_cos = float(colloc_dots.mean().item())
    random_mean_cos = float(rand_dots.abs().mean().item())
    cos_ratio = colloc_mean_cos / max(random_mean_cos, 1e-6)

    # Purified Born attention unnormalized kernel energy: [gamma * ReLU(s - tau)]^4 + eps_base * ReLU(s)^2
    colloc_coh = F.relu(float(gamma_h) * (colloc_dots - float(tau_h)))
    colloc_energy = float((colloc_coh.pow(4) + float(eps_base) * F.relu(colloc_dots).square()).mean().item())

    rand_coh = F.relu(float(gamma_h) * (rand_dots - float(tau_h)))
    rand_energy = float((rand_coh.pow(4) + float(eps_base) * F.relu(rand_dots).square()).mean().item())

    # Also compute normalized purified Born target attention weight in a 32-token context window
    # where key j=0 is the collocated partner v and keys j=1..31 are random context tokens
    ctx_distractors = 31
    dist_idx = torch.randint(0, active_ids.numel(), (k_eff, ctx_distractors), generator=gen).to(dev)
    dist_tokens = active_ids[dist_idx]  # (k_eff, 31)
    q_seq = emb_norm[colloc_u].view(k_eff, 1, 1, -1)  # (k_eff, 1, 1, d)
    k_colloc = emb_norm[colloc_v].view(k_eff, 1, 1, -1)
    k_rand0 = emb_norm[rand_v].view(k_eff, 1, 1, -1)
    k_dist = emb_norm[dist_tokens].unsqueeze(1)  # (k_eff, 1, 31, d)

    k_seq_colloc = torch.cat([k_colloc, k_dist], dim=2)  # (k_eff, 1, 32, d)
    k_seq_rand = torch.cat([k_rand0, k_dist], dim=2)
    w_colloc = purified_born_attention_weights(
        q_seq, k_seq_colloc, gamma=gamma_h, tau=tau_h, eps_base=eps_base, causal=False
    )
    w_rand = purified_born_attention_weights(
        q_seq, k_seq_rand, gamma=gamma_h, tau=tau_h, eps_base=eps_base, causal=False
    )
    colloc_attn_weight = float(w_colloc[:, 0, 0, 0].mean().item())
    random_attn_weight = float(w_rand[:, 0, 0, 0].mean().item())

    return {
        "colloc_mean_cos": colloc_mean_cos,
        "random_mean_cos": random_mean_cos,
        "cos_ratio": cos_ratio,
        "colloc_purified_born_energy": colloc_energy,
        "random_purified_born_energy": rand_energy,
        "colloc_purified_attn_weight": colloc_attn_weight,
        "random_purified_attn_weight": random_attn_weight,
    }


__all__ = [
    "compute_ppmi_svd_embeddings",
    "initialize_quantum_born_with_ppmi_svd",
    "initialize_classical_with_ppmi_svd",
    "compute_collocation_coherence_metrics",
]
