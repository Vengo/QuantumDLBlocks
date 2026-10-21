"""JAX companion utilities for FineWeb Multi-Shard Streaming & Memorization Diagnostics (`qdlblocks_jax.fineweb_stream`).

Implements Section 3.2 of EXPERIMENT_PLAN_011.md:
- `compute_memorization_gap_metrics_jax`
- `extract_shard_batch_jax`
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def compute_memorization_gap_metrics_jax(
    train_ce_arr: jax.Array,
    val_ce_arr: jax.Array,
    boundary_pre_arr: jax.Array | None = None,
    boundary_post_arr: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """Compute memorization gap and pass-boundary drop diagnostics in pure JAX."""
    train_f = jnp.asarray(train_ce_arr, dtype=jnp.float32).reshape(-1)
    val_f = jnp.asarray(val_ce_arr, dtype=jnp.float32).reshape(-1)
    if train_f.size == 0 or val_f.size == 0:
        raise ValueError("train_ce_arr and val_ce_arr must be non-empty")
    if train_f.shape != val_f.shape:
        raise ValueError(f"Shape mismatch: {train_f.shape} vs {val_f.shape}")

    gaps = val_f - train_f
    if boundary_pre_arr is not None and boundary_post_arr is not None:
        pre_f = jnp.asarray(boundary_pre_arr, dtype=jnp.float32).reshape(-1)
        post_f = jnp.asarray(boundary_post_arr, dtype=jnp.float32).reshape(-1)
        mean_drop = jnp.mean(pre_f - post_f) if pre_f.size > 0 else jnp.zeros((), dtype=jnp.float32)
    else:
        mean_drop = jnp.zeros((), dtype=jnp.float32)

    return {
        "final_train_ce": train_f[-1],
        "final_val_ce": val_f[-1],
        "final_memorization_gap": gaps[-1],
        "max_memorization_gap": jnp.max(gaps),
        "mean_boundary_drop": mean_drop,
    }


def extract_shard_batch_jax(
    shard_tokens: np.ndarray,
    chunk_idx: int,
    batch_size: int,
    seq_len: int,
) -> tuple[jax.Array, jax.Array]:
    """Slice contiguous `(x, y)` token batch of shape `(batch_size, seq_len)` from `shard_tokens`."""
    B = int(batch_size)
    T = int(seq_len)
    tok_per_step = B * T
    start = int(chunk_idx) * tok_per_step
    buf = np.asarray(shard_tokens[start : start + tok_per_step + 1], dtype=np.int32)
    if buf.size < tok_per_step + 1:
        raise ValueError(
            f"Shard slice starting at {start} has {buf.size} tokens, need {tok_per_step + 1}"
        )
    x = jnp.asarray(buf[:-1].reshape(B, T), dtype=jnp.int32)
    y = jnp.asarray(buf[1:].reshape(B, T), dtype=jnp.int32)
    return x, y


__all__ = [
    "compute_memorization_gap_metrics_jax",
    "extract_shard_batch_jax",
]
