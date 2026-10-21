"""Memory-efficient Flash-Born Attention in JAX using Pallas TPU/VMEM kernels and chunked lax.scan."""

from __future__ import annotations

from functools import partial
import math
from typing import Any, Literal

import jax
import jax.numpy as jnp

try:
    from jax.experimental import pallas as pl

    HAS_PALLAS = True
except Exception:
    HAS_PALLAS = False

try:
    from jax.experimental.pallas import tpu as pltpu

    HAS_PALLAS_TPU = True
except Exception:
    HAS_PALLAS_TPU = False

__all__ = [
    "HAS_PALLAS",
    "HAS_PALLAS_TPU",
    "flash_born_attention_jax",
]


def _tpu_compiler_params() -> Any:
    if HAS_PALLAS_TPU:
        try:
            return pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))
        except Exception:
            return None
    return None


def _flash_born_scan_forward(
    q_pad: jax.Array,
    k_pad: jax.Array,
    v_pad: jax.Array,
    scale_vec: jax.Array,
    is_causal: bool,
    eps: float,
    block_size: int,
    valid_len: int,
) -> tuple[jax.Array, jax.Array]:
    """Chunked O(T) memory forward pass over padded `(B, H, T_pad, D)` tensors with per-head `scale_vec` `(H,)`."""
    B, H, T_pad, D = q_pad.shape
    num_blocks = T_pad // block_size

    q_blocks = q_pad.reshape(B, H, num_blocks, block_size, D)
    k_blocks = k_pad.reshape(B, H, num_blocks, block_size, D)
    v_blocks = v_pad.reshape(B, H, num_blocks, block_size, D)
    idx_blocks = jnp.arange(T_pad, dtype=jnp.int32).reshape(num_blocks, block_size)
    scale_4d = scale_vec.astype(jnp.float32).reshape(1, H, 1, 1)

    q_b = jnp.moveaxis(q_blocks, 2, 0)  # (num_blocks, B, H, B_sz, D)
    k_b = jnp.moveaxis(k_blocks, 2, 0)  # (num_blocks, B, H, B_sz, D)
    v_b = jnp.moveaxis(v_blocks, 2, 0)  # (num_blocks, B, H, B_sz, D)

    def eval_query_block(q_blk: jax.Array, q_idx: jax.Array) -> tuple[jax.Array, jax.Array]:
        init_o = jnp.zeros((B, H, block_size, D), dtype=jnp.float32)
        init_z = jnp.zeros((B, H, block_size, 1), dtype=jnp.float32)

        def body_fn(carry, kv_inputs):
            o_num, z_den = carry
            k_blk, v_blk, k_idx = kv_inputs
            raw_dot = jnp.einsum(
                "bhid,bhjd->bhij",
                q_blk,
                k_blk,
                preferred_element_type=jnp.float32,
            )
            scores = raw_dot * scale_4d
            valid = (q_idx[:, None] < valid_len) & (k_idx[None, :] < valid_len)
            if is_causal:
                valid = valid & (q_idx[:, None] >= k_idx[None, :])
            a = jnp.where(valid[None, None, :, :], jnp.maximum(scores, 0.0), 0.0)
            p = a * a
            o_num = o_num + jnp.einsum(
                "bhij,bhjd->bhid",
                p.astype(v_blk.dtype),
                v_blk,
                preferred_element_type=jnp.float32,
            )
            z_den = z_den + jnp.sum(p, axis=-1, keepdims=True)
            return (o_num, z_den), None

        (o_num, z_den), _ = jax.lax.scan(body_fn, (init_o, init_z), (k_b, v_b, idx_blocks))
        z_total = z_den + eps
        o_blk = (o_num / z_total).astype(q_pad.dtype)
        return o_blk, z_total

    o_b, z_b = jax.vmap(eval_query_block, in_axes=(0, 0))(q_b, idx_blocks)
    o_pad = jnp.moveaxis(o_b, 0, 2).reshape(B, H, T_pad, D)
    z_pad = jnp.moveaxis(z_b, 0, 2).reshape(B, H, T_pad, 1)
    return o_pad, z_pad


def _flash_born_pallas_forward(
    q_pad: jax.Array,
    k_pad: jax.Array,
    v_pad: jax.Array,
    scale_vec: jax.Array,
    is_causal: bool,
    eps: float,
    block_size: int,
    valid_len: int,
) -> tuple[jax.Array, jax.Array]:
    """VMEM-tiled Pallas Flash-Born forward kernel with causal block skipping and per-head scale."""
    if not HAS_PALLAS:
        return _flash_born_scan_forward(
            q_pad, k_pad, v_pad, scale_vec, is_causal, eps, block_size, valid_len
        )

    B, H, T_pad, D = q_pad.shape
    num_blocks = T_pad // block_size
    BH = B * H
    q_flat = q_pad.reshape(BH, T_pad, D)
    k_flat = k_pad.reshape(BH, T_pad, D)
    v_flat = v_pad.reshape(BH, T_pad, D)
    scale_bh = jnp.broadcast_to(scale_vec.astype(jnp.float32)[None, :], (B, H)).reshape(BH, 1, 1)

    def pallas_fwd_kernel(q_ref, k_ref, v_ref, s_ref, o_ref, z_ref):
        m_blk = pl.program_id(1)
        q_blk = q_ref[0, :, :]
        scale_val = s_ref[0, 0, 0].astype(jnp.float32)
        q_idx = m_blk * block_size + jnp.arange(block_size, dtype=jnp.int32)
        max_n_blk = (m_blk + 1) if is_causal else num_blocks

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
            raw_dot = jnp.matmul(
                q_blk,
                jnp.swapaxes(k_blk, -1, -2),
                preferred_element_type=jnp.float32,
            )
            scores = raw_dot * scale_val
            valid = (q_idx[:, None] < valid_len) & (k_idx[None, :] < valid_len)
            if is_causal:
                valid = valid & (q_idx[:, None] >= k_idx[None, :])
            a = jnp.where(valid, jnp.maximum(scores, 0.0), 0.0)
            p = a * a
            o_acc = o_acc + jnp.matmul(
                p.astype(v_blk.dtype),
                v_blk,
                preferred_element_type=jnp.float32,
            )
            z_acc = z_acc + jnp.sum(p, axis=-1, keepdims=True)
            return o_acc, z_acc

        init_o = jnp.zeros((block_size, D), dtype=jnp.float32)
        init_z = jnp.zeros((block_size, 1), dtype=jnp.float32)
        o_num, z_den = jax.lax.fori_loop(0, max_n_blk, loop_body, (init_o, init_z))
        z_tot = z_den + eps
        o_ref[0, :, :] = (o_num / z_tot).astype(o_ref.dtype)
        z_ref[0, :, :] = z_tot

    backend_plat = jax.default_backend()
    is_cpu = backend_plat == "cpu"
    is_tpu = backend_plat == "tpu"
    compiler_params = _tpu_compiler_params() if is_tpu else None

    try:
        call_kwargs: dict[str, Any] = dict(
            out_shape=[
                jax.ShapeDtypeStruct((BH, T_pad, D), q_pad.dtype),
                jax.ShapeDtypeStruct((BH, T_pad, 1), jnp.float32),
            ],
            grid=(BH, num_blocks),
            in_specs=[
                pl.BlockSpec((1, block_size, D), lambda bh, m: (bh, m, 0)),
                pl.BlockSpec((1, T_pad, D), lambda bh, m: (bh, 0, 0)),
                pl.BlockSpec((1, T_pad, D), lambda bh, m: (bh, 0, 0)),
                pl.BlockSpec((1, 1, 1), lambda bh, m: (bh, 0, 0)),
            ],
            out_specs=[
                pl.BlockSpec((1, block_size, D), lambda bh, m: (bh, m, 0)),
                pl.BlockSpec((1, block_size, 1), lambda bh, m: (bh, m, 0)),
            ],
            interpret=is_cpu,
        )
        if compiler_params is not None:
            call_kwargs["compiler_params"] = compiler_params

        o_flat, z_flat = pl.pallas_call(pallas_fwd_kernel, **call_kwargs)(
            q_flat, k_flat, v_flat, scale_bh
        )
        return (
            o_flat.reshape(B, H, T_pad, D).astype(q_pad.dtype),
            z_flat.reshape(B, H, T_pad, 1),
        )
    except Exception:
        return _flash_born_scan_forward(
            q_pad, k_pad, v_pad, scale_vec, is_causal, eps, block_size, valid_len
        )


def _flash_born_pallas_backward(
    q_pad: jax.Array,
    k_pad: jax.Array,
    v_pad: jax.Array,
    o_pad: jax.Array,
    z_pad: jax.Array,
    do_pad: jax.Array,
    scale_vec: jax.Array,
    is_causal: bool,
    block_size: int,
    valid_len: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array] | None:
    """Atomic-free VMEM-tiled Pallas Flash-Born backward pass (`dQ` kernel + `dK/dV/dScale` kernel)."""
    if not HAS_PALLAS:
        return None

    B, H, T_pad, D = q_pad.shape
    num_blocks = T_pad // block_size
    BH = B * H

    q_flat = q_pad.reshape(BH, T_pad, D)
    k_flat = k_pad.reshape(BH, T_pad, D)
    v_flat = v_pad.reshape(BH, T_pad, D)
    o_flat = o_pad.reshape(BH, T_pad, D)
    do_flat = do_pad.reshape(BH, T_pad, D)
    z_flat = z_pad.reshape(BH, T_pad, 1).astype(jnp.float32)
    d_flat = jnp.sum(do_flat.astype(jnp.float32) * o_flat.astype(jnp.float32), axis=-1, keepdims=True)
    scale_bh = jnp.broadcast_to(scale_vec.astype(jnp.float32)[None, :], (B, H)).reshape(BH, 1, 1)

    def pallas_bwd_dq_kernel(q_ref, k_ref, v_ref, do_ref, z_ref, d_ref, s_ref, dq_ref):
        m_blk = pl.program_id(1)
        q_blk = q_ref[0, :, :]
        do_blk = do_ref[0, :, :]
        z_blk = z_ref[0, :, :]
        d_blk = d_ref[0, :, :]
        scale_val = s_ref[0, 0, 0].astype(jnp.float32)
        q_idx = m_blk * block_size + jnp.arange(block_size, dtype=jnp.int32)
        max_n_blk = (m_blk + 1) if is_causal else num_blocks

        def loop_n(n_blk, dq_acc):
            k_blk = pl.load(k_ref, (0, pl.dslice(n_blk * block_size, block_size), slice(None)))
            v_blk = pl.load(v_ref, (0, pl.dslice(n_blk * block_size, block_size), slice(None)))
            k_idx = n_blk * block_size + jnp.arange(block_size, dtype=jnp.int32)
            raw_dot = jnp.matmul(
                q_blk,
                jnp.swapaxes(k_blk, -1, -2),
                preferred_element_type=jnp.float32,
            )
            scores = raw_dot * scale_val
            valid = (q_idx[:, None] < valid_len) & (k_idx[None, :] < valid_len)
            if is_causal:
                valid = valid & (q_idx[:, None] >= k_idx[None, :])
            a = jnp.where(valid, jnp.maximum(scores, 0.0), 0.0)
            dp_raw = jnp.matmul(
                do_blk,
                jnp.swapaxes(v_blk, -1, -2),
                preferred_element_type=jnp.float32,
            )
            ds = jnp.where(valid, (2.0 * a / z_blk) * (dp_raw - d_blk), 0.0)
            dq_step = (
                jnp.matmul(ds.astype(k_blk.dtype), k_blk, preferred_element_type=jnp.float32)
                * scale_val
            )
            return dq_acc + dq_step

        init_dq = jnp.zeros((block_size, D), dtype=jnp.float32)
        dq_out = jax.lax.fori_loop(0, max_n_blk, loop_n, init_dq)
        dq_ref[0, :, :] = dq_out.astype(dq_ref.dtype)

    def pallas_bwd_dkv_kernel(q_ref, k_ref, v_ref, do_ref, z_ref, d_ref, s_ref, dk_ref, dv_ref, ds_ref):
        n_blk = pl.program_id(1)
        k_blk = k_ref[0, :, :]
        v_blk = v_ref[0, :, :]
        scale_val = s_ref[0, 0, 0].astype(jnp.float32)
        k_idx = n_blk * block_size + jnp.arange(block_size, dtype=jnp.int32)
        start_m_blk = n_blk if is_causal else 0

        def loop_m(m_blk, carry):
            dk_acc, dv_acc, dscale_acc = carry
            q_blk = pl.load(q_ref, (0, pl.dslice(m_blk * block_size, block_size), slice(None)))
            do_blk = pl.load(do_ref, (0, pl.dslice(m_blk * block_size, block_size), slice(None)))
            z_blk = pl.load(z_ref, (0, pl.dslice(m_blk * block_size, block_size), slice(None)))
            d_blk = pl.load(d_ref, (0, pl.dslice(m_blk * block_size, block_size), slice(None)))
            q_idx = m_blk * block_size + jnp.arange(block_size, dtype=jnp.int32)

            raw_dot = jnp.matmul(
                q_blk,
                jnp.swapaxes(k_blk, -1, -2),
                preferred_element_type=jnp.float32,
            )
            scores = raw_dot * scale_val
            valid = (q_idx[:, None] < valid_len) & (k_idx[None, :] < valid_len)
            if is_causal:
                valid = valid & (q_idx[:, None] >= k_idx[None, :])
            a = jnp.where(valid, jnp.maximum(scores, 0.0), 0.0)
            p = a * a
            w = p / z_blk
            dv_step = jnp.matmul(
                jnp.swapaxes(w.astype(do_blk.dtype), -1, -2),
                do_blk,
                preferred_element_type=jnp.float32,
            )
            dp_raw = jnp.matmul(
                do_blk,
                jnp.swapaxes(v_blk, -1, -2),
                preferred_element_type=jnp.float32,
            )
            ds = jnp.where(valid, (2.0 * a / z_blk) * (dp_raw - d_blk), 0.0)
            dscale_step = jnp.sum(ds * raw_dot)
            dk_step = (
                jnp.matmul(
                    jnp.swapaxes(ds.astype(q_blk.dtype), -1, -2),
                    q_blk,
                    preferred_element_type=jnp.float32,
                )
                * scale_val
            )
            return dk_acc + dk_step, dv_acc + dv_step, dscale_acc + dscale_step

        init_dk = jnp.zeros((block_size, D), dtype=jnp.float32)
        init_dv = jnp.zeros((block_size, D), dtype=jnp.float32)
        init_ds = jnp.zeros((1, 1), dtype=jnp.float32)
        dk_out, dv_out, ds_out = jax.lax.fori_loop(
            start_m_blk, num_blocks, loop_m, (init_dk, init_dv, init_ds)
        )
        dk_ref[0, :, :] = dk_out.astype(dk_ref.dtype)
        dv_ref[0, :, :] = dv_out.astype(dv_ref.dtype)
        ds_ref[0, :, :] = ds_out

    backend_plat = jax.default_backend()
    is_cpu = backend_plat == "cpu"
    is_tpu = backend_plat == "tpu"
    compiler_params = _tpu_compiler_params() if is_tpu else None

    try:
        dq_kwargs: dict[str, Any] = dict(
            out_shape=jax.ShapeDtypeStruct((BH, T_pad, D), q_pad.dtype),
            grid=(BH, num_blocks),
            in_specs=[
                pl.BlockSpec((1, block_size, D), lambda bh, m: (bh, m, 0)),
                pl.BlockSpec((1, T_pad, D), lambda bh, m: (bh, 0, 0)),
                pl.BlockSpec((1, T_pad, D), lambda bh, m: (bh, 0, 0)),
                pl.BlockSpec((1, block_size, D), lambda bh, m: (bh, m, 0)),
                pl.BlockSpec((1, block_size, 1), lambda bh, m: (bh, m, 0)),
                pl.BlockSpec((1, block_size, 1), lambda bh, m: (bh, m, 0)),
                pl.BlockSpec((1, 1, 1), lambda bh, m: (bh, 0, 0)),
            ],
            out_specs=pl.BlockSpec((1, block_size, D), lambda bh, m: (bh, m, 0)),
            interpret=is_cpu,
        )
        dkv_kwargs: dict[str, Any] = dict(
            out_shape=[
                jax.ShapeDtypeStruct((BH, T_pad, D), k_pad.dtype),
                jax.ShapeDtypeStruct((BH, T_pad, D), v_pad.dtype),
                jax.ShapeDtypeStruct((BH, num_blocks, 1), jnp.float32),
            ],
            grid=(BH, num_blocks),
            in_specs=[
                pl.BlockSpec((1, T_pad, D), lambda bh, n: (bh, 0, 0)),
                pl.BlockSpec((1, block_size, D), lambda bh, n: (bh, n, 0)),
                pl.BlockSpec((1, block_size, D), lambda bh, n: (bh, n, 0)),
                pl.BlockSpec((1, T_pad, D), lambda bh, n: (bh, 0, 0)),
                pl.BlockSpec((1, T_pad, 1), lambda bh, n: (bh, 0, 0)),
                pl.BlockSpec((1, T_pad, 1), lambda bh, n: (bh, 0, 0)),
                pl.BlockSpec((1, 1, 1), lambda bh, n: (bh, 0, 0)),
            ],
            out_specs=[
                pl.BlockSpec((1, block_size, D), lambda bh, n: (bh, n, 0)),
                pl.BlockSpec((1, block_size, D), lambda bh, n: (bh, n, 0)),
                pl.BlockSpec((1, 1, 1), lambda bh, n: (bh, n, 0)),
            ],
            interpret=is_cpu,
        )
        if compiler_params is not None:
            dq_kwargs["compiler_params"] = compiler_params
            dkv_kwargs["compiler_params"] = compiler_params

        dq_flat = pl.pallas_call(pallas_bwd_dq_kernel, **dq_kwargs)(
            q_flat, k_flat, v_flat, do_flat, z_flat, d_flat, scale_bh
        )
        dk_flat, dv_flat, ds_blocks = pl.pallas_call(pallas_bwd_dkv_kernel, **dkv_kwargs)(
            q_flat, k_flat, v_flat, do_flat, z_flat, d_flat, scale_bh
        )
        dscale_vec = jnp.sum(ds_blocks.reshape(B, H, num_blocks), axis=(0, 2)).astype(scale_vec.dtype)
        return (
            dq_flat.reshape(B, H, T_pad, D).astype(q_pad.dtype),
            dk_flat.reshape(B, H, T_pad, D).astype(k_pad.dtype),
            dv_flat.reshape(B, H, T_pad, D).astype(v_pad.dtype),
            dscale_vec,
        )
    except Exception:
        return None


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8))
def _flash_born_core(
    q_pad: jax.Array,
    k_pad: jax.Array,
    v_pad: jax.Array,
    scale_vec: jax.Array,
    is_causal: bool,
    eps: float,
    block_size: int,
    valid_len: int,
    use_pallas: bool,
) -> jax.Array:
    if use_pallas:
        o_pad, _ = _flash_born_pallas_forward(
            q_pad, k_pad, v_pad, scale_vec, is_causal, eps, block_size, valid_len
        )
    else:
        o_pad, _ = _flash_born_scan_forward(
            q_pad, k_pad, v_pad, scale_vec, is_causal, eps, block_size, valid_len
        )
    return o_pad


def _flash_born_core_fwd(
    q_pad: jax.Array,
    k_pad: jax.Array,
    v_pad: jax.Array,
    scale_vec: jax.Array,
    is_causal: bool,
    eps: float,
    block_size: int,
    valid_len: int,
    use_pallas: bool,
):
    if use_pallas:
        o_pad, z_pad = _flash_born_pallas_forward(
            q_pad, k_pad, v_pad, scale_vec, is_causal, eps, block_size, valid_len
        )
    else:
        o_pad, z_pad = _flash_born_scan_forward(
            q_pad, k_pad, v_pad, scale_vec, is_causal, eps, block_size, valid_len
        )
    return o_pad, (q_pad, k_pad, v_pad, o_pad, z_pad, scale_vec)


def _flash_born_core_bwd(
    is_causal: bool,
    eps: float,
    block_size: int,
    valid_len: int,
    use_pallas: bool,
    res,
    do_pad: jax.Array,
):
    del eps
    q_pad, k_pad, v_pad, o_pad, z_pad, scale_vec = res

    if use_pallas:
        pallas_grads = _flash_born_pallas_backward(
            q_pad,
            k_pad,
            v_pad,
            o_pad,
            z_pad,
            do_pad,
            scale_vec,
            is_causal,
            block_size,
            valid_len,
        )
        if pallas_grads is not None:
            return pallas_grads

    B, H, T_pad, D = q_pad.shape
    num_blocks = T_pad // block_size
    scale_4d = scale_vec.astype(jnp.float32).reshape(1, H, 1, 1)

    q_f = q_pad
    k_f = k_pad
    v_f = v_pad
    o_f = o_pad.astype(jnp.float32)
    do_f = do_pad
    d_row = jnp.sum(do_f.astype(jnp.float32) * o_f, axis=-1, keepdims=True)  # (B, H, T_pad, 1)

    q_b = jnp.moveaxis(q_f.reshape(B, H, num_blocks, block_size, D), 2, 0)
    k_b = jnp.moveaxis(k_f.reshape(B, H, num_blocks, block_size, D), 2, 0)
    v_b = jnp.moveaxis(v_f.reshape(B, H, num_blocks, block_size, D), 2, 0)
    do_b = jnp.moveaxis(do_f.reshape(B, H, num_blocks, block_size, D), 2, 0)
    z_b = jnp.moveaxis(z_pad.reshape(B, H, num_blocks, block_size, 1), 2, 0)
    d_b = jnp.moveaxis(d_row.reshape(B, H, num_blocks, block_size, 1), 2, 0)
    idx_blocks = jnp.arange(T_pad, dtype=jnp.int32).reshape(num_blocks, block_size)

    def compute_dq_block(q_blk, do_blk, z_blk, d_blk, q_idx):
        init_dq = jnp.zeros((B, H, block_size, D), dtype=jnp.float32)

        def scan_k(dq_acc, kv_inputs):
            k_blk, v_blk, k_idx = kv_inputs
            raw_dot = jnp.einsum(
                "bhid,bhjd->bhij",
                q_blk,
                k_blk,
                preferred_element_type=jnp.float32,
            )
            scores = raw_dot * scale_4d
            valid = (q_idx[:, None] < valid_len) & (k_idx[None, :] < valid_len)
            if is_causal:
                valid = valid & (q_idx[:, None] >= k_idx[None, :])
            a = jnp.where(valid[None, None, :, :], jnp.maximum(scores, 0.0), 0.0)
            dp_raw = jnp.einsum(
                "bhid,bhjd->bhij",
                do_blk,
                v_blk,
                preferred_element_type=jnp.float32,
            )
            dS = jnp.where(valid[None, None, :, :], (2.0 * a / z_blk) * (dp_raw - d_blk), 0.0) * scale_4d
            dq_acc = dq_acc + jnp.einsum(
                "bhij,bhjd->bhid",
                dS.astype(k_blk.dtype),
                k_blk,
                preferred_element_type=jnp.float32,
            )
            return dq_acc, None

        dq_blk, _ = jax.lax.scan(scan_k, init_dq, (k_b, v_b, idx_blocks))
        return dq_blk

    dq_b = jax.vmap(compute_dq_block, in_axes=(0, 0, 0, 0, 0))(
        q_b, do_b, z_b, d_b, idx_blocks
    )

    def compute_dkv_block(k_blk, v_blk, k_idx):
        init_dk = jnp.zeros((B, H, block_size, D), dtype=jnp.float32)
        init_dv = jnp.zeros((B, H, block_size, D), dtype=jnp.float32)
        init_ds = jnp.zeros((H,), dtype=jnp.float32)

        def scan_q(carry, q_inputs):
            dk_acc, dv_acc, ds_acc = carry
            q_blk, do_blk, z_blk, d_blk, q_idx = q_inputs
            raw_dot = jnp.einsum(
                "bhid,bhjd->bhij",
                q_blk,
                k_blk,
                preferred_element_type=jnp.float32,
            )
            scores = raw_dot * scale_4d
            valid = (q_idx[:, None] < valid_len) & (k_idx[None, :] < valid_len)
            if is_causal:
                valid = valid & (q_idx[:, None] >= k_idx[None, :])
            a = jnp.where(valid[None, None, :, :], jnp.maximum(scores, 0.0), 0.0)
            p = a * a
            w = p / z_blk
            dv_acc = dv_acc + jnp.einsum(
                "bhij,bhid->bhjd",
                w.astype(do_blk.dtype),
                do_blk,
                preferred_element_type=jnp.float32,
            )
            dp_raw = jnp.einsum(
                "bhid,bhjd->bhij",
                do_blk,
                v_blk,
                preferred_element_type=jnp.float32,
            )
            ds_unscaled = jnp.where(valid[None, None, :, :], (2.0 * a / z_blk) * (dp_raw - d_blk), 0.0)
            ds_acc = ds_acc + jnp.sum(ds_unscaled * raw_dot, axis=(0, 2, 3))
            dS = ds_unscaled * scale_4d
            dk_acc = dk_acc + jnp.einsum(
                "bhij,bhid->bhjd",
                dS.astype(q_blk.dtype),
                q_blk,
                preferred_element_type=jnp.float32,
            )
            return (dk_acc, dv_acc, ds_acc), None

        (dk_blk, dv_blk, ds_blk), _ = jax.lax.scan(
            scan_q, (init_dk, init_dv, init_ds), (q_b, do_b, z_b, d_b, idx_blocks)
        )
        return dk_blk, dv_blk, ds_blk

    dk_b, dv_b, ds_b = jax.vmap(compute_dkv_block, in_axes=(0, 0, 0))(k_b, v_b, idx_blocks)

    dq_pad = jnp.moveaxis(dq_b, 0, 2).reshape(B, H, T_pad, D).astype(q_pad.dtype)
    dk_pad = jnp.moveaxis(dk_b, 0, 2).reshape(B, H, T_pad, D).astype(k_pad.dtype)
    dv_pad = jnp.moveaxis(dv_b, 0, 2).reshape(B, H, T_pad, D).astype(v_pad.dtype)
    dscale_vec = jnp.sum(ds_b, axis=0).astype(scale_vec.dtype)
    return dq_pad, dk_pad, dv_pad, dscale_vec


_flash_born_core.defvjp(_flash_born_core_fwd, _flash_born_core_bwd)


def flash_born_attention_jax(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float | jax.Array | None = None,
    is_causal: bool = True,
    eps: float = 1e-6,
    block_size: int = 64,
    backend: Literal["auto", "pallas", "scan"] = "auto",
) -> jax.Array:
    """Memory-efficient O(T) Flash-Born Positive-Cone Attention in JAX (Pallas TPU & lax.scan).

    Never materializes the full `(B, H, T, T)` attention matrix in HBM2e/VRAM during
    forward or backward execution, and supports both scalar `scale` and differentiable
    per-head `scale` tensors of shape `(H,)` or `(1, H, 1, 1)`.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"Expected 4D (B, H, T, D) inputs, got {q.shape}, {k.shape}, {v.shape}.")
    B, H, T, D = q.shape
    if scale is None:
        scale_vec = jnp.full((H,), math.sqrt(D), dtype=jnp.float32)
    elif isinstance(scale, (int, float)):
        scale_vec = jnp.full((H,), float(scale), dtype=jnp.float32)
    else:
        scale_arr = jnp.asarray(scale, dtype=jnp.float32).reshape(-1)
        if scale_arr.size == 1:
            scale_vec = jnp.broadcast_to(scale_arr, (H,))
        elif scale_arr.size == H:
            scale_vec = scale_arr
        else:
            raise ValueError(f"Expected scale with 1 or H={H} elements, got shape {scale.shape}.")

    b_sz = min(block_size, max(T, 1))
    rem = T % b_sz
    pad_t = (b_sz - rem) if rem != 0 else 0
    if pad_t > 0:
        q_pad = jnp.pad(q, ((0, 0), (0, 0), (0, pad_t), (0, 0)))
        k_pad = jnp.pad(k, ((0, 0), (0, 0), (0, pad_t), (0, 0)))
        v_pad = jnp.pad(v, ((0, 0), (0, 0), (0, pad_t), (0, 0)))
    else:
        q_pad, k_pad, v_pad = q, k, v

    if backend == "pallas":
        use_pallas = HAS_PALLAS
    elif backend == "scan":
        use_pallas = False
    else:
        on_tpu = jax.default_backend() == "tpu"
        use_pallas = bool(HAS_PALLAS and on_tpu)

    o_pad = _flash_born_core(
        q_pad,
        k_pad,
        v_pad,
        scale_vec,
        bool(is_causal),
        float(eps),
        int(b_sz),
        int(T),
        bool(use_pallas),
    )
    return o_pad[:, :, :T, :]
