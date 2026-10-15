"""Memory-efficient Flash-Born Attention in JAX using chunked lax.scan + custom_vjp and optional Pallas."""

from __future__ import annotations

from functools import partial
import math
from typing import Literal

import jax
import jax.numpy as jnp

try:
    from jax.experimental import pallas as pl

    HAS_PALLAS = True
except Exception:
    HAS_PALLAS = False

__all__ = [
    "flash_born_attention_jax",
]


def _flash_born_scan_forward(
    q_pad: jax.Array,
    k_pad: jax.Array,
    v_pad: jax.Array,
    scale: float,
    is_causal: bool,
    eps: float,
    block_size: int,
    valid_len: int,
) -> tuple[jax.Array, jax.Array]:
    """Chunked O(T) memory forward pass over padded (B, H, T_pad, D) tensors."""
    B, H, T_pad, D = q_pad.shape
    num_blocks = T_pad // block_size

    q_blocks = q_pad.reshape(B, H, num_blocks, block_size, D)
    k_blocks = k_pad.reshape(B, H, num_blocks, block_size, D)
    v_blocks = v_pad.reshape(B, H, num_blocks, block_size, D)
    idx_blocks = jnp.arange(T_pad, dtype=jnp.int32).reshape(num_blocks, block_size)

    # Transpose block axis to leading dimension for jax.vmap / jax.lax.scan
    q_b = jnp.moveaxis(q_blocks, 2, 0)  # (num_blocks, B, H, B_sz, D)
    k_b = jnp.moveaxis(k_blocks, 2, 0)  # (num_blocks, B, H, B_sz, D)
    v_b = jnp.moveaxis(v_blocks, 2, 0)  # (num_blocks, B, H, B_sz, D)

    def eval_query_block(q_blk: jax.Array, q_idx: jax.Array) -> tuple[jax.Array, jax.Array]:
        init_o = jnp.zeros((B, H, block_size, D), dtype= jnp.float32)
        init_z = jnp.zeros((B, H, block_size, 1), dtype=jnp.float32)

        def body_fn(carry, kv_inputs):
            o_num, z_den = carry
            k_blk, v_blk, k_idx = kv_inputs
            scores = (
                jnp.einsum(
                    "bhid,bhjd->bhij",
                    q_blk.astype(jnp.float32),
                    k_blk.astype(jnp.float32),
                    precision=jax.lax.Precision.HIGHEST,
                )
                * scale
            )
            valid = (q_idx[:, None] < valid_len) & (k_idx[None, :] < valid_len)
            if is_causal:
                valid = valid & (q_idx[:, None] >= k_idx[None, :])
            a = jnp.where(valid[None, None, :, :], jnp.maximum(scores, 0.0), 0.0)
            p = a * a
            o_num = o_num + jnp.einsum(
                "bhij,bhjd->bhid",
                p,
                v_blk.astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )
            z_den = z_den + jnp.sum(p, axis=-1, keepdims=True)
            return (o_num, z_den), None

        (o_num, z_den), _ = jax.lax.scan(body_fn, (init_o, init_z), (k_b, v_b, idx_blocks))
        z_total = z_den + eps
        o_blk = (o_num / z_total).astype(q_pad.dtype)
        return o_blk, z_total

    o_b, z_b = jax.vmap(eval_query_block, in_axes=(0, 0))(q_b, idx_blocks)
    # o_b: (num_blocks, B, H, B_sz, D) -> (B, H, T_pad, D)
    o_pad = jnp.moveaxis(o_b, 0, 2).reshape(B, H, T_pad, D)
    z_pad = jnp.moveaxis(z_b, 0, 2).reshape(B, H, T_pad, 1)
    return o_pad, z_pad


def _flash_born_pallas_forward(
    q_pad: jax.Array,
    k_pad: jax.Array,
    v_pad: jax.Array,
    scale: float,
    is_causal: bool,
    eps: float,
    block_size: int,
    valid_len: int,
) -> tuple[jax.Array, jax.Array]:
    """Pallas kernel forward pass with automatic interpret=True on CPU."""
    if not HAS_PALLAS:
        return _flash_born_scan_forward(
            q_pad, k_pad, v_pad, scale, is_causal, eps, block_size, valid_len
        )

    B, H, T_pad, D = q_pad.shape
    num_blocks = T_pad // block_size
    BH = B * H
    q_flat = q_pad.reshape(BH, T_pad, D).astype(jnp.float32)
    k_flat = k_pad.reshape(BH, T_pad, D).astype(jnp.float32)
    v_flat = v_pad.reshape(BH, T_pad, D).astype(jnp.float32)

    def pallas_kernel(q_ref, k_ref, v_ref, o_ref, z_ref):
        m_blk = pl.program_id(1)
        q_blk = q_ref[0, :, :]
        q_idx = m_blk * block_size + jnp.arange(block_size, dtype=jnp.int32)

        def loop_body(n_blk, carry):
            o_acc, z_acc = carry
            k_blk = pl.load(
                k_ref,
                (0, pl.dslice(n_blk * block_size, block_size), slice(None)),
            )
            v_blk = pl.load(
                v_ref,
                (0, pl.dslice(n_blk * block_size, block_size), slice(None)),
            )
            k_idx = n_blk * block_size + jnp.arange(block_size, dtype=jnp.int32)
            scores = (
                jnp.matmul(q_blk, jnp.swapaxes(k_blk, -1, -2), precision=jax.lax.Precision.HIGHEST)
                * scale
            )
            valid = (q_idx[:, None] < valid_len) & (k_idx[None, :] < valid_len)
            if is_causal:
                valid = valid & (q_idx[:, None] >= k_idx[None, :])
            a = jnp.where(valid, jnp.maximum(scores, 0.0), 0.0)
            p = a * a
            o_acc = o_acc + jnp.matmul(p, v_blk, precision=jax.lax.Precision.HIGHEST)
            z_acc = z_acc + jnp.sum(p, axis=-1, keepdims=True)
            return o_acc, z_acc

        init_o = jnp.zeros((block_size, D), dtype=jnp.float32)
        init_z = jnp.zeros((block_size, 1), dtype=jnp.float32)
        o_num, z_den = jax.lax.fori_loop(0, num_blocks, loop_body, (init_o, init_z))
        z_tot = z_den + eps
        o_ref[0, :, :] = o_num / z_tot
        z_ref[0, :, :] = z_tot

    is_cpu = all(d.platform == "cpu" for d in q_pad.devices()) if q_pad.devices() else True
    try:
        o_flat, z_flat = pl.pallas_call(
            pallas_kernel,
            out_shape=[
                jax.ShapeDtypeStruct((BH, T_pad, D), jnp.float32),
                jax.ShapeDtypeStruct((BH, T_pad, 1), jnp.float32),
            ],
            grid=(BH, num_blocks),
            in_specs=[
                pl.BlockSpec((1, block_size, D), lambda bh, m: (bh, m, 0)),
                pl.BlockSpec((1, T_pad, D), lambda bh, m: (bh, 0, 0)),
                pl.BlockSpec((1, T_pad, D), lambda bh, m: (bh, 0, 0)),
            ],
            out_specs=[
                pl.BlockSpec((1, block_size, D), lambda bh, m: (bh, m, 0)),
                pl.BlockSpec((1, block_size, 1), lambda bh, m: (bh, m, 0)),
            ],
            interpret=is_cpu,
        )(q_flat, k_flat, v_flat)
        return (
            o_flat.reshape(B, H, T_pad, D).astype(q_pad.dtype),
            z_flat.reshape(B, H, T_pad, 1),
        )
    except Exception:
        return _flash_born_scan_forward(
            q_pad, k_pad, v_pad, scale, is_causal, eps, block_size, valid_len
        )


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8))
def _flash_born_core(
    q_pad: jax.Array,
    k_pad: jax.Array,
    v_pad: jax.Array,
    scale: float,
    is_causal: bool,
    eps: float,
    block_size: int,
    valid_len: int,
    use_pallas: bool,
) -> jax.Array:
    if use_pallas:
        o_pad, _ = _flash_born_pallas_forward(
            q_pad, k_pad, v_pad, scale, is_causal, eps, block_size, valid_len
        )
    else:
        o_pad, _ = _flash_born_scan_forward(
            q_pad, k_pad, v_pad, scale, is_causal, eps, block_size, valid_len
        )
    return o_pad


def _flash_born_core_fwd(
    q_pad: jax.Array,
    k_pad: jax.Array,
    v_pad: jax.Array,
    scale: float,
    is_causal: bool,
    eps: float,
    block_size: int,
    valid_len: int,
    use_pallas: bool,
):
    if use_pallas:
        o_pad, z_pad = _flash_born_pallas_forward(
            q_pad, k_pad, v_pad, scale, is_causal, eps, block_size, valid_len
        )
    else:
        o_pad, z_pad = _flash_born_scan_forward(
            q_pad, k_pad, v_pad, scale, is_causal, eps, block_size, valid_len
        )
    return o_pad, (q_pad, k_pad, v_pad, o_pad, z_pad)


def _flash_born_core_bwd(
    scale: float,
    is_causal: bool,
    eps: float,
    block_size: int,
    valid_len: int,
    use_pallas: bool,
    res,
    do_pad: jax.Array,
):
    del eps, use_pallas
    q_pad, k_pad, v_pad, o_pad, z_pad = res
    B, H, T_pad, D = q_pad.shape
    num_blocks = T_pad // block_size

    q_f = q_pad.astype(jnp.float32)
    k_f = k_pad.astype(jnp.float32)
    v_f = v_pad.astype(jnp.float32)
    o_f = o_pad.astype(jnp.float32)
    do_f = do_pad.astype(jnp.float32)
    d_row = jnp.sum(do_f * o_f, axis=-1, keepdims=True)  # (B, H, T_pad, 1)

    q_b = jnp.moveaxis(q_f.reshape(B, H, num_blocks, block_size, D), 2, 0)
    k_b = jnp.moveaxis(k_f.reshape(B, H, num_blocks, block_size, D), 2, 0)
    v_b = jnp.moveaxis(v_f.reshape(B, H, num_blocks, block_size, D), 2, 0)
    do_b = jnp.moveaxis(do_f.reshape(B, H, num_blocks, block_size, D), 2, 0)
    z_b = jnp.moveaxis(z_pad.reshape(B, H, num_blocks, block_size, 1), 2, 0)
    d_b = jnp.moveaxis(d_row.reshape(B, H, num_blocks, block_size, 1), 2, 0)
    idx_blocks = jnp.arange(T_pad, dtype=jnp.int32).reshape(num_blocks, block_size)

    # 1. Compute dq_b by vmapping over query blocks and scanning over key blocks
    def compute_dq_block(q_blk, do_blk, z_blk, d_blk, q_idx):
        init_dq = jnp.zeros((B, H, block_size, D), dtype=jnp.float32)

        def scan_k(dq_acc, kv_inputs):
            k_blk, v_blk, k_idx = kv_inputs
            scores = (
                jnp.einsum(
                    "bhid,bhjd->bhij",
                    q_blk,
                    k_blk,
                    precision=jax.lax.Precision.HIGHEST,
                )
                * scale
            )
            valid = (q_idx[:, None] < valid_len) & (k_idx[None, :] < valid_len)
            if is_causal:
                valid = valid & (q_idx[:, None] >= k_idx[None, :])
            a = jnp.where(valid[None, None, :, :], jnp.maximum(scores, 0.0), 0.0)
            dp_raw = jnp.einsum(
                "bhid,bhjd->bhij",
                do_blk,
                v_blk,
                precision=jax.lax.Precision.HIGHEST,
            )
            dS = (2.0 * a / z_blk) * (dp_raw - d_blk) * scale
            dq_acc = dq_acc + jnp.einsum(
                "bhij,bhjd->bhid",
                dS,
                k_blk,
                precision=jax.lax.Precision.HIGHEST,
            )
            return dq_acc, None

        dq_blk, _ = jax.lax.scan(scan_k, init_dq, (k_b, v_b, idx_blocks))
        return dq_blk

    dq_b = jax.vmap(compute_dq_block, in_axes=(0, 0, 0, 0, 0))(
        q_b, do_b, z_b, d_b, idx_blocks
    )

    # 2. Compute dk_b and dv_b by vmapping over key blocks and scanning over query blocks
    def compute_dkv_block(k_blk, v_blk, k_idx):
        init_dk = jnp.zeros((B, H, block_size, D), dtype=jnp.float32)
        init_dv = jnp.zeros((B, H, block_size, D), dtype=jnp.float32)

        def scan_q(carry, q_inputs):
            dk_acc, dv_acc = carry
            q_blk, do_blk, z_blk, d_blk, q_idx = q_inputs
            scores = (
                jnp.einsum(
                    "bhid,bhjd->bhij",
                    q_blk,
                    k_blk,
                    precision=jax.lax.Precision.HIGHEST,
                )
                * scale
            )
            valid = (q_idx[:, None] < valid_len) & (k_idx[None, :] < valid_len)
            if is_causal:
                valid = valid & (q_idx[:, None] >= k_idx[None, :])
            a = jnp.where(valid[None, None, :, :], jnp.maximum(scores, 0.0), 0.0)
            p = a * a
            w = p / z_blk
            dv_acc = dv_acc + jnp.einsum(
                "bhij,bhid->bhjd",
                w,
                do_blk,
                precision=jax.lax.Precision.HIGHEST,
            )
            dp_raw = jnp.einsum(
                "bhid,bhjd->bhij",
                do_blk,
                v_blk,
                precision=jax.lax.Precision.HIGHEST,
            )
            dS = (2.0 * a / z_blk) * (dp_raw - d_blk) * scale
            dk_acc = dk_acc + jnp.einsum(
                "bhij,bhid->bhjd",
                dS,
                q_blk,
                precision=jax.lax.Precision.HIGHEST,
            )
            return (dk_acc, dv_acc), None

        (dk_blk, dv_blk), _ = jax.lax.scan(
            scan_q, (init_dk, init_dv), (q_b, do_b, z_b, d_b, idx_blocks)
        )
        return dk_blk, dv_blk

    dk_b, dv_b = jax.vmap(compute_dkv_block, in_axes=(0, 0, 0))(k_b, v_b, idx_blocks)

    dq_pad = jnp.moveaxis(dq_b, 0, 2).reshape(B, H, T_pad, D).astype(q_pad.dtype)
    dk_pad = jnp.moveaxis(dk_b, 0, 2).reshape(B, H, T_pad, D).astype(k_pad.dtype)
    dv_pad = jnp.moveaxis(dv_b, 0, 2).reshape(B, H, T_pad, D).astype(v_pad.dtype)
    return dq_pad, dk_pad, dv_pad


_flash_born_core.defvjp(_flash_born_core_fwd, _flash_born_core_bwd)


def flash_born_attention_jax(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float | None = None,
    is_causal: bool = True,
    eps: float = 1e-6,
    block_size: int = 64,
    backend: Literal["auto", "pallas", "scan"] = "auto",
) -> jax.Array:
    """Memory-efficient O(T) Flash-Born Positive-Cone Attention in JAX.

    Never materializes the full (B, H, T, T) attention matrix in forward or backward.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"Expected 4D (B, H, T, D) inputs, got {q.shape}, {k.shape}, {v.shape}.")
    B, H, T, D = q.shape
    if scale is None:
        scale = math.sqrt(D)
    b_sz = min(block_size, max(T, 1))
    rem = T % b_sz
    pad_t = (b_sz - rem) if rem != 0 else 0
    if pad_t > 0:
        q_pad = jnp.pad(q, ((0, 0), (0, 0), (0, pad_t), (0, 0)))
        k_pad = jnp.pad(k, ((0, 0), (0, 0), (0, pad_t), (0, 0)))
        v_pad = jnp.pad(v, ((0, 0), (0, 0), (0, pad_t), (0, 0)))
    else:
        q_pad, k_pad, v_pad = q, k, v

    use_pallas = backend == "pallas"
    o_pad = _flash_born_core(
        q_pad,
        k_pad,
        v_pad,
        float(scale),
        bool(is_causal),
        float(eps),
        int(b_sz),
        int(T),
        bool(use_pallas),
    )
    return o_pad[:, :, :T, :]
