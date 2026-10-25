"""Fused Flash-Born Attention Triton Kernel (Zero T x T Attention Matrix Materialization in VRAM).

Computes Subspace Projector Born Attention:
    a_ij = max(0, scale * <Q_i, K_j>) * 1(j <= i)
    p_ij = a_ij^2
    O_i  = (sum_j p_ij V_j) / (sum_j p_ij + eps)

Entirely inside GPU SRAM tiles in both the forward and backward passes, reducing
attention memory complexity from O(B * H * T^2) to O(B * H * T).
Supports block-sparse sliding-window (`window_size > 0`) key-tile skipping and
intra-document boundary masking (`doc_ids`).
"""

from __future__ import annotations

from typing import Any

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

IS_HIP: bool = getattr(torch.version, "hip", None) is not None


def get_flash_born_launch_meta(
    seq_len: int,
    is_hip: bool | None = None,
    head_dim: int = 64,
) -> dict[str, int]:
    """Return architecture-tuned Triton launch parameters for CUDA (Wave32 warp) vs ROCm RDNA 4 (Wave32 EU)."""
    use_hip = IS_HIP if is_hip is None else bool(is_hip)
    eff_dim = triton.next_power_of_2(head_dim) if HAS_TRITON else (1 << (max(1, head_dim) - 1).bit_length())
    if eff_dim >= 256:
        block_m = 16
        block_n = 16
        num_stages = 1
    elif eff_dim >= 128:
        block_m = 32 if seq_len >= 32 else 16
        block_n = 32 if seq_len >= 32 else 16
        num_stages = 1
    else:
        block_m = 64 if seq_len >= 64 else (32 if seq_len >= 32 else 16)
        block_n = 64 if seq_len >= 64 else (32 if seq_len >= 32 else 16)
        num_stages = 1 if use_hip else 2
    if use_hip:
        return {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "num_warps": 4,
            "num_stages": 1,
            "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16,
        }
    return {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "num_warps": 4,
        "num_stages": num_stages,
    }


def get_rdna4_autotune_configs() -> list[dict[str, int]]:
    """Return candidate RDNA 4 (`gfx1201`, Wave32 16x16x16 BF16 WMMA) Triton autotune configurations."""
    return [
        {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
        {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
        {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
        {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 2, "num_stages": 1, "waves_per_eu": 4, "matrix_instr_nonkdim": 16},
    ]


if HAS_TRITON:

    @triton.jit
    def _flash_born_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        Z_ptr,
        Scale_ptr,
        DOC_IDS_ptr,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        stride_zb,
        stride_zh,
        stride_zm,
        stride_doc_b,
        stride_doc_t,
        T: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        PER_HEAD_SCALE: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        HAS_DOC_IDS: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_h = tl.program_id(2)

        start_m = pid_m * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < HEAD_DIM

        q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
        k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
        o_base = Out_ptr + pid_b * stride_ob + pid_h * stride_oh
        z_base = Z_ptr + pid_b * stride_zb + pid_h * stride_zh

        if PER_HEAD_SCALE:
            scale = tl.load(Scale_ptr + pid_h).to(tl.float32)
        else:
            scale = tl.load(Scale_ptr).to(tl.float32)

        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        mask_m = offs_m < T
        q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

        if HAS_DOC_IDS:
            doc_base = DOC_IDS_ptr + pid_b * stride_doc_b
            doc_m = tl.load(doc_base + offs_m * stride_doc_t, mask=mask_m, other=-1)

        o_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        z_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

        lo = 0 if WINDOW_SIZE <= 0 else tl.maximum(0, (start_m - WINDOW_SIZE + 1) & ~(BLOCK_N - 1))
        max_n = (pid_m + 1) * BLOCK_M if IS_CAUSAL else T
        max_n = tl.minimum(max_n, T)

        for start_n in range(lo, max_n, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < T

            k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

            k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

            s = tl.dot(q, tl.trans(k), allow_tf32=False) * scale
            a = tl.maximum(s, 0.0)

            if IS_CAUSAL:
                valid_mask = (offs_m[:, None] >= offs_n[None, :]) & mask_n[None, :]
            else:
                valid_mask = mask_n[None, :]
            if WINDOW_SIZE > 0:
                valid_mask = valid_mask & ((offs_m[:, None] - offs_n[None, :]) < WINDOW_SIZE)
            if HAS_DOC_IDS:
                doc_n = tl.load(doc_base + offs_n * stride_doc_t, mask=mask_n, other=-2)
                valid_mask = valid_mask & (doc_m[:, None] == doc_n[None, :])
            a = tl.where(valid_mask, a, 0.0)

            p = a * a

            z_acc += tl.sum(p, axis=1)
            o_acc += tl.dot(p.to(v.dtype), v, allow_tf32=False)

        z_denom = z_acc + EPS
        o_out = o_acc / z_denom[:, None]

        o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        tl.store(o_ptrs, o_out.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])

        z_ptrs = z_base + offs_m * stride_zm
        tl.store(z_ptrs, z_denom, mask=mask_m)

    @triton.jit
    def _flash_born_bwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        dO_ptr,
        Z_ptr,
        Scale_ptr,
        DOC_IDS_ptr,
        dK_ptr,
        dV_ptr,
        dScale_ptr,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_zb,
        stride_zh,
        stride_zm,
        stride_doc_b,
        stride_doc_t,
        T: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        PER_HEAD_SCALE: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        HAS_DOC_IDS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_h = tl.program_id(2)

        start_n = pid_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        mask_n = offs_n < T
        mask_d = offs_d < HEAD_DIM

        q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
        k_base = K_ptr + pid_b * stride_qb + pid_h * stride_qh
        v_base = V_ptr + pid_b * stride_qb + pid_h * stride_qh
        o_base = Out_ptr + pid_b * stride_qb + pid_h * stride_qh
        do_base = dO_ptr + pid_b * stride_qb + pid_h * stride_qh
        dk_base = dK_ptr + pid_b * stride_qb + pid_h * stride_qh
        dv_base = dV_ptr + pid_b * stride_qb + pid_h * stride_qh
        z_base = Z_ptr + pid_b * stride_zb + pid_h * stride_zh

        if PER_HEAD_SCALE:
            scale = tl.load(Scale_ptr + pid_h).to(tl.float32)
        else:
            scale = tl.load(Scale_ptr).to(tl.float32)

        k_ptrs = k_base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qd
        v_ptrs = v_base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qd

        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        if HAS_DOC_IDS:
            doc_base = DOC_IDS_ptr + pid_b * stride_doc_b
            doc_n = tl.load(doc_base + offs_n * stride_doc_t, mask=mask_n, other=-2)

        dk_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        dv_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        dscale_acc = 0.0

        start_m = start_n if IS_CAUSAL else 0
        end_m = T if (WINDOW_SIZE <= 0 or not IS_CAUSAL) else tl.minimum(T, start_n + BLOCK_N + WINDOW_SIZE)

        for curr_m in range(start_m, end_m, BLOCK_M):
            offs_m = curr_m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < T

            q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
            o_ptrs = o_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
            do_ptrs = do_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
            z_ptrs = z_base + offs_m * stride_zm

            q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
            o = tl.load(o_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
            do_f32 = do.to(tl.float32)
            z_denom = tl.load(z_ptrs, mask=mask_m, other=1.0).to(tl.float32)

            raw_dot = tl.dot(q, tl.trans(k), allow_tf32=False)
            s = raw_dot * scale
            a = tl.maximum(s, 0.0)

            if IS_CAUSAL:
                valid_mask = (offs_m[:, None] >= offs_n[None, :]) & mask_m[:, None] & mask_n[None, :]
            else:
                valid_mask = mask_m[:, None] & mask_n[None, :]
            if WINDOW_SIZE > 0:
                valid_mask = valid_mask & ((offs_m[:, None] - offs_n[None, :]) < WINDOW_SIZE)
            if HAS_DOC_IDS:
                doc_m = tl.load(doc_base + offs_m * stride_doc_t, mask=mask_m, other=-1)
                valid_mask = valid_mask & (doc_m[:, None] == doc_n[None, :])
            a = tl.where(valid_mask, a, 0.0)

            p = a * a
            w = p / z_denom[:, None]

            dv_acc += tl.dot(tl.trans(w.to(do.dtype)), do, allow_tf32=False)

            delta_i = tl.sum(do_f32 * o, axis=1)
            dov = tl.dot(do, tl.trans(v), allow_tf32=False)
            dp = (dov - delta_i[:, None]) / z_denom[:, None]

            ds = 2.0 * a * dp
            ds = tl.where(valid_mask, ds, 0.0)

            dscale_acc += tl.sum(ds * raw_dot)

            ds_typed = ds.to(q.dtype)
            dk_acc += tl.dot(tl.trans(ds_typed), q, allow_tf32=False) * scale

        dk_ptrs = dk_base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qd
        dv_ptrs = dv_base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qd
        tl.store(dk_ptrs, dk_acc.to(dK_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_d[None, :])
        tl.store(dv_ptrs, dv_acc.to(dV_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_d[None, :])

        if PER_HEAD_SCALE:
            tl.atomic_add(dScale_ptr + pid_h, dscale_acc)
        else:
            tl.atomic_add(dScale_ptr, dscale_acc)

    @triton.jit
    def _flash_born_bwd_dq_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        dO_ptr,
        Z_ptr,
        Scale_ptr,
        DOC_IDS_ptr,
        dQ_ptr,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_zb,
        stride_zh,
        stride_zm,
        stride_doc_b,
        stride_doc_t,
        T: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        PER_HEAD_SCALE: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        HAS_DOC_IDS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_h = tl.program_id(2)

        start_m = pid_m * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < T
        mask_d = offs_d < HEAD_DIM

        q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
        k_base = K_ptr + pid_b * stride_qb + pid_h * stride_qh
        v_base = V_ptr + pid_b * stride_qb + pid_h * stride_qh
        o_base = Out_ptr + pid_b * stride_qb + pid_h * stride_qh
        do_base = dO_ptr + pid_b * stride_qb + pid_h * stride_qh
        dq_base = dQ_ptr + pid_b * stride_qb + pid_h * stride_qh
        z_base = Z_ptr + pid_b * stride_zb + pid_h * stride_zh

        if PER_HEAD_SCALE:
            scale = tl.load(Scale_ptr + pid_h).to(tl.float32)
        else:
            scale = tl.load(Scale_ptr).to(tl.float32)

        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        o_ptrs = o_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        do_ptrs = do_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        z_ptrs = z_base + offs_m * stride_zm

        q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        do_f32 = do.to(tl.float32)
        z_denom = tl.load(z_ptrs, mask=mask_m, other=1.0).to(tl.float32)
        delta_i = tl.sum(do_f32 * o, axis=1)

        if HAS_DOC_IDS:
            doc_base = DOC_IDS_ptr + pid_b * stride_doc_b
            doc_m = tl.load(doc_base + offs_m * stride_doc_t, mask=mask_m, other=-1)

        dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        lo = 0 if WINDOW_SIZE <= 0 else tl.maximum(0, (start_m - WINDOW_SIZE + 1) & ~(BLOCK_N - 1))
        max_n = (pid_m + 1) * BLOCK_M if IS_CAUSAL else T
        max_n = tl.minimum(max_n, T)

        for start_n in range(lo, max_n, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < T

            k_ptrs = k_base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qd
            v_ptrs = v_base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qd

            k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

            s = tl.dot(q, tl.trans(k), allow_tf32=False) * scale
            a = tl.maximum(s, 0.0)

            if IS_CAUSAL:
                valid_mask = (offs_m[:, None] >= offs_n[None, :]) & mask_m[:, None] & mask_n[None, :]
            else:
                valid_mask = mask_m[:, None] & mask_n[None, :]
            if WINDOW_SIZE > 0:
                valid_mask = valid_mask & ((offs_m[:, None] - offs_n[None, :]) < WINDOW_SIZE)
            if HAS_DOC_IDS:
                doc_n = tl.load(doc_base + offs_n * stride_doc_t, mask=mask_n, other=-2)
                valid_mask = valid_mask & (doc_m[:, None] == doc_n[None, :])
            a = tl.where(valid_mask, a, 0.0)

            dov = tl.dot(do, tl.trans(v), allow_tf32=False)
            dp = (dov - delta_i[:, None]) / z_denom[:, None]
            ds = 2.0 * a * dp
            ds = tl.where(valid_mask, ds, 0.0)

            dq_acc += tl.dot(ds.to(k.dtype), k, allow_tf32=False) * scale

        dq_ptrs = dq_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        tl.store(dq_ptrs, dq_acc.to(dQ_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])


class _FlashBornAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: torch.Tensor,
        is_causal: bool,
        eps: float,
        window_size: int = 0,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, H, T, D = q.shape
        target_dtype = torch.get_autocast_gpu_dtype() if (q.is_cuda and torch.is_autocast_enabled()) else v.dtype
        q_c = q.to(target_dtype).contiguous()
        k_c = k.to(target_dtype).contiguous()
        v_c = v.to(target_dtype).contiguous()

        per_head_scale = scale.numel() > 1
        scale_flat = scale.contiguous().view(-1).to(dtype=torch.float32, device=q.device)

        has_doc_ids = doc_ids is not None
        if has_doc_ids:
            doc_ids_c = doc_ids.to(device=q.device, dtype=torch.int32).contiguous()
            doc_ptr = doc_ids_c
            stride_doc_b = doc_ids_c.stride(0)
            stride_doc_t = doc_ids_c.stride(1)
        else:
            doc_ids_c = torch.empty(0, device=q.device, dtype=torch.int32)
            doc_ptr = q_c
            stride_doc_b = 0
            stride_doc_t = 0

        out = torch.empty_like(q_c)
        z_denom = torch.empty((B, H, T), dtype=torch.float32, device=q.device)

        launch_meta = get_flash_born_launch_meta(T, head_dim=D)
        block_m = launch_meta["BLOCK_M"]
        block_d = triton.next_power_of_2(D)
        grid = (triton.cdiv(T, block_m), B, H)

        _flash_born_fwd_kernel[grid](
            q_c,
            k_c,
            v_c,
            out,
            z_denom,
            scale_flat,
            doc_ptr,
            q_c.stride(0),
            q_c.stride(1),
            q_c.stride(2),
            q_c.stride(3),
            k_c.stride(0),
            k_c.stride(1),
            k_c.stride(2),
            k_c.stride(3),
            v_c.stride(0),
            v_c.stride(1),
            v_c.stride(2),
            v_c.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            z_denom.stride(0),
            z_denom.stride(1),
            z_denom.stride(2),
            stride_doc_b,
            stride_doc_t,
            T=T,
            HEAD_DIM=D,
            BLOCK_D=block_d,
            IS_CAUSAL=is_causal,
            PER_HEAD_SCALE=per_head_scale,
            WINDOW_SIZE=int(window_size),
            HAS_DOC_IDS=has_doc_ids,
            EPS=eps,
            **launch_meta,
        )

        ctx.save_for_backward(q_c, k_c, v_c, out, z_denom, scale_flat, doc_ids_c)
        ctx.is_causal = is_causal
        ctx.per_head_scale = per_head_scale
        ctx.window_size = int(window_size)
        ctx.has_doc_ids = has_doc_ids
        ctx.scale_shape = scale.shape
        ctx.scale_dtype = scale.dtype
        ctx.orig_q_dtype = q.dtype
        ctx.orig_k_dtype = k.dtype
        ctx.orig_v_dtype = v.dtype
        return out

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q_c, k_c, v_c, out, z_denom, scale_flat, doc_ids_c = ctx.saved_tensors
        B, H, T, D = q_c.shape

        do_c = do.to(v_c.dtype).contiguous()
        dq = torch.empty_like(q_c)
        dk = torch.empty_like(k_c)
        dv = torch.empty_like(v_c)
        dscale_f32 = torch.zeros_like(scale_flat, dtype=torch.float32)

        if ctx.has_doc_ids:
            doc_ptr = doc_ids_c
            stride_doc_b = doc_ids_c.stride(0)
            stride_doc_t = doc_ids_c.stride(1)
        else:
            doc_ptr = q_c
            stride_doc_b = 0
            stride_doc_t = 0

        launch_meta = get_flash_born_launch_meta(T, head_dim=D)
        block_m = launch_meta["BLOCK_M"]
        block_n = launch_meta["BLOCK_N"]
        block_d = triton.next_power_of_2(D)
        grid_n = (triton.cdiv(T, block_n), B, H)
        grid_m = (triton.cdiv(T, block_m), B, H)

        _flash_born_bwd_kernel[grid_n](
            q_c,
            k_c,
            v_c,
            out,
            do_c,
            z_denom,
            scale_flat,
            doc_ptr,
            dk,
            dv,
            dscale_f32,
            q_c.stride(0),
            q_c.stride(1),
            q_c.stride(2),
            q_c.stride(3),
            z_denom.stride(0),
            z_denom.stride(1),
            z_denom.stride(2),
            stride_doc_b,
            stride_doc_t,
            T=T,
            HEAD_DIM=D,
            BLOCK_D=block_d,
            IS_CAUSAL=ctx.is_causal,
            PER_HEAD_SCALE=ctx.per_head_scale,
            WINDOW_SIZE=ctx.window_size,
            HAS_DOC_IDS=ctx.has_doc_ids,
            **launch_meta,
        )

        _flash_born_bwd_dq_kernel[grid_m](
            q_c,
            k_c,
            v_c,
            out,
            do_c,
            z_denom,
            scale_flat,
            doc_ptr,
            dq,
            q_c.stride(0),
            q_c.stride(1),
            q_c.stride(2),
            q_c.stride(3),
            z_denom.stride(0),
            z_denom.stride(1),
            z_denom.stride(2),
            stride_doc_b,
            stride_doc_t,
            T=T,
            HEAD_DIM=D,
            BLOCK_D=block_d,
            IS_CAUSAL=ctx.is_causal,
            PER_HEAD_SCALE=ctx.per_head_scale,
            WINDOW_SIZE=ctx.window_size,
            HAS_DOC_IDS=ctx.has_doc_ids,
            **launch_meta,
        )

        dq_out = dq.to(ctx.orig_q_dtype)
        dk_out = dk.to(ctx.orig_k_dtype)
        dv_out = dv.to(ctx.orig_v_dtype)
        dscale = dscale_f32.view(ctx.scale_shape).to(ctx.scale_dtype)
        return dq_out, dk_out, dv_out, dscale, None, None, None, None


@torch.compiler.disable
def _run_triton_flash_born(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale_t: torch.Tensor,
    is_causal: bool,
    eps: float,
    window_size: int = 0,
    doc_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    return _FlashBornAttentionFn.apply(q, k, v, scale_t, is_causal, eps, int(window_size), doc_ids)


def _apply_window_and_doc_mask(
    prob: torch.Tensor,
    is_causal: bool,
    window_size: int,
    doc_ids: torch.Tensor | None,
) -> torch.Tensor:
    T = prob.shape[-1]
    device = prob.device
    idx = torch.arange(T, device=device)
    diff = idx[:, None] - idx[None, :]
    mask = (diff >= 0) if is_causal else torch.ones((T, T), device=device, dtype=torch.bool)
    if window_size > 0:
        mask = mask & (diff < int(window_size))
    mask_4d = mask.view(1, 1, T, T)
    if doc_ids is not None:
        doc_eq = (doc_ids[:, None, :, None] == doc_ids[:, None, None, :])
        mask_4d = mask_4d & doc_eq
    return prob * mask_4d.to(prob.dtype)


def flash_born_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | torch.Tensor,
    is_causal: bool = True,
    eps: float = 1e-12,
    window_size: int = 0,
    doc_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute SRAM-tiled Flash-Born Attention via Triton on CUDA (or fallback on CPU).

    Supports sliding-window tile skipping (`window_size > 0`), document boundary
    masking (`doc_ids` of shape `(B, T)`), and any `HEAD_DIM % 16 == 0` in `[16, 256]`.
    """
    D = q.shape[-1]
    is_supported_d = (D >= 16) and (D <= 256) and (D % 16 == 0)

    if not isinstance(scale, torch.Tensor):
        scale_t = torch.tensor(float(scale), dtype=torch.float32, device=q.device)
    else:
        scale_t = scale

    if HAS_TRITON and q.is_cuda and is_supported_d:
        return _run_triton_flash_born(q, k, v, scale_t, is_causal, eps, window_size=window_size, doc_ids=doc_ids)

    s_v = scale_t.to(v.dtype)
    if s_v.ndim == 1:
        s_v = s_v.view(1, -1, 1, 1)
    scaled_overlap = torch.matmul(q.to(v.dtype), k.to(v.dtype).transpose(-1, -2)) * s_v
    projected_amp = torch.relu(scaled_overlap)
    prob = projected_amp.square()
    if is_causal or window_size > 0 or doc_ids is not None:
        prob = _apply_window_and_doc_mask(prob, is_causal=is_causal, window_size=window_size, doc_ids=doc_ids)
    born_weights = prob / (torch.sum(prob, dim=-1, keepdim=True) + eps)
    return torch.matmul(born_weights.to(v.dtype), v)


if HAS_TRITON:

    @triton.jit
    def _flash_purified_born_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        Z_ptr,
        Scale_ptr,
        Tau_ptr,
        DOC_IDS_ptr,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        stride_zb,
        stride_zh,
        stride_zm,
        stride_doc_b,
        stride_doc_t,
        T: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        PER_HEAD_SCALE: tl.constexpr,
        PER_HEAD_TAU: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        HAS_DOC_IDS: tl.constexpr,
        EPS_BASE: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_h = tl.program_id(2)

        start_m = pid_m * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < HEAD_DIM

        q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
        k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
        o_base = Out_ptr + pid_b * stride_ob + pid_h * stride_oh
        z_base = Z_ptr + pid_b * stride_zb + pid_h * stride_zh

        scale = tl.load(Scale_ptr + pid_h if PER_HEAD_SCALE else Scale_ptr).to(tl.float32)
        tau = tl.load(Tau_ptr + pid_h if PER_HEAD_TAU else Tau_ptr).to(tl.float32)

        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        mask_m = offs_m < T
        q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

        if HAS_DOC_IDS:
            doc_base = DOC_IDS_ptr + pid_b * stride_doc_b
            doc_m = tl.load(doc_base + offs_m * stride_doc_t, mask=mask_m, other=-1)

        o_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        z_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

        lo = 0 if WINDOW_SIZE <= 0 else tl.maximum(0, (start_m - WINDOW_SIZE + 1) & ~(BLOCK_N - 1))
        max_n = (pid_m + 1) * BLOCK_M if IS_CAUSAL else T
        max_n = tl.minimum(max_n, T)

        for start_n in range(lo, max_n, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < T

            k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

            k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

            raw_dot = tl.dot(q, tl.trans(k), allow_tf32=False)
            u = tl.maximum(scale * (raw_dot - tau), 0.0)
            b = tl.maximum(raw_dot, 0.0)

            if IS_CAUSAL:
                valid_mask = (offs_m[:, None] >= offs_n[None, :]) & mask_n[None, :]
            else:
                valid_mask = mask_n[None, :]
            if WINDOW_SIZE > 0:
                valid_mask = valid_mask & ((offs_m[:, None] - offs_n[None, :]) < WINDOW_SIZE)
            if HAS_DOC_IDS:
                doc_n = tl.load(doc_base + offs_n * stride_doc_t, mask=mask_n, other=-2)
                valid_mask = valid_mask & (doc_m[:, None] == doc_n[None, :])
            u = tl.where(valid_mask, u, 0.0)
            b = tl.where(valid_mask, b, 0.0)

            u2 = u * u
            p = u2 * u2 + EPS_BASE * (b * b)

            z_acc += tl.sum(p, axis=1)
            o_acc += tl.dot(p, v.to(tl.float32), allow_tf32=False)

        z_denom = z_acc + EPS
        o_out = o_acc / z_denom[:, None]

        o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        tl.store(o_ptrs, o_out.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])

        z_ptrs = z_base + offs_m * stride_zm
        tl.store(z_ptrs, z_denom, mask=mask_m)

    @triton.jit
    def _flash_purified_born_bwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        dO_ptr,
        Z_ptr,
        Scale_ptr,
        Tau_ptr,
        DOC_IDS_ptr,
        dK_ptr,
        dV_ptr,
        dScale_ptr,
        dTau_ptr,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_zb,
        stride_zh,
        stride_zm,
        stride_doc_b,
        stride_doc_t,
        T: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        PER_HEAD_SCALE: tl.constexpr,
        PER_HEAD_TAU: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        HAS_DOC_IDS: tl.constexpr,
        EPS_BASE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_h = tl.program_id(2)

        start_n = pid_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        mask_n = offs_n < T
        mask_d = offs_d < HEAD_DIM

        q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
        k_base = K_ptr + pid_b * stride_qb + pid_h * stride_qh
        v_base = V_ptr + pid_b * stride_qb + pid_h * stride_qh
        o_base = Out_ptr + pid_b * stride_qb + pid_h * stride_qh
        do_base = dO_ptr + pid_b * stride_qb + pid_h * stride_qh
        dk_base = dK_ptr + pid_b * stride_qb + pid_h * stride_qh
        dv_base = dV_ptr + pid_b * stride_qb + pid_h * stride_qh
        z_base = Z_ptr + pid_b * stride_zb + pid_h * stride_zh

        scale = tl.load(Scale_ptr + pid_h if PER_HEAD_SCALE else Scale_ptr).to(tl.float32)
        tau = tl.load(Tau_ptr + pid_h if PER_HEAD_TAU else Tau_ptr).to(tl.float32)

        k_ptrs = k_base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qd
        v_ptrs = v_base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qd

        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        if HAS_DOC_IDS:
            doc_base = DOC_IDS_ptr + pid_b * stride_doc_b
            doc_n = tl.load(doc_base + offs_n * stride_doc_t, mask=mask_n, other=-2)

        dk_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        dv_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        dscale_acc = 0.0
        dtau_acc = 0.0

        start_m = start_n if IS_CAUSAL else 0
        end_m = T if (WINDOW_SIZE <= 0 or not IS_CAUSAL) else tl.minimum(T, start_n + BLOCK_N + WINDOW_SIZE)

        for curr_m in range(start_m, end_m, BLOCK_M):
            offs_m = curr_m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < T
            mask_md = mask_m[:, None] & mask_d[None, :]

            q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
            o_ptrs = o_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
            do_ptrs = do_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
            z_ptrs = z_base + offs_m * stride_zm

            q = tl.load(q_ptrs, mask=mask_md, other=0.0)
            o = tl.load(o_ptrs, mask=mask_md, other=0.0).to(tl.float32)
            do = tl.load(do_ptrs, mask=mask_md, other=0.0)
            do_f32 = do.to(tl.float32)
            z_denom = tl.load(z_ptrs, mask=mask_m, other=1.0).to(tl.float32)

            raw_dot = tl.dot(q, tl.trans(k), allow_tf32=False)
            u = tl.maximum(scale * (raw_dot - tau), 0.0)
            b = tl.maximum(raw_dot, 0.0)

            if IS_CAUSAL:
                valid_mask = (offs_m[:, None] >= offs_n[None, :]) & mask_m[:, None] & mask_n[None, :]
            else:
                valid_mask = mask_m[:, None] & mask_n[None, :]
            if WINDOW_SIZE > 0:
                valid_mask = valid_mask & ((offs_m[:, None] - offs_n[None, :]) < WINDOW_SIZE)
            if HAS_DOC_IDS:
                doc_m = tl.load(doc_base + offs_m * stride_doc_t, mask=mask_m, other=-1)
                valid_mask = valid_mask & (doc_m[:, None] == doc_n[None, :])
            u = tl.where(valid_mask, u, 0.0)
            b = tl.where(valid_mask, b, 0.0)

            u2 = u * u
            u3 = u2 * u
            p = u2 * u2 + EPS_BASE * (b * b)
            w = p / z_denom[:, None]

            dv_acc += tl.dot(tl.trans(w.to(do.dtype)), do, allow_tf32=False)

            delta_i = tl.sum(do_f32 * o, axis=1)
            dov = tl.dot(do, tl.trans(v), allow_tf32=False)
            dp = (dov - delta_i[:, None]) / z_denom[:, None]

            du_dp = 4.0 * u3 * dp
            d_raw_dot = du_dp * scale + 2.0 * EPS_BASE * b * dp
            d_raw_dot = tl.where(valid_mask, d_raw_dot, 0.0)

            dscale_acc += tl.sum(du_dp * (raw_dot - tau))
            dtau_acc += tl.sum(-du_dp * scale)

            dk_acc += tl.dot(tl.trans(d_raw_dot.to(q.dtype)), q, allow_tf32=False)

        dk_ptrs = dk_base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qd
        dv_ptrs = dv_base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qd
        tl.store(dk_ptrs, dk_acc.to(dK_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_d[None, :])
        tl.store(dv_ptrs, dv_acc.to(dV_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_d[None, :])

        tl.atomic_add(dScale_ptr + pid_h if PER_HEAD_SCALE else dScale_ptr, dscale_acc)
        tl.atomic_add(dTau_ptr + pid_h if PER_HEAD_TAU else dTau_ptr, dtau_acc)

    @triton.jit
    def _flash_purified_born_bwd_dq_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        dO_ptr,
        Z_ptr,
        Scale_ptr,
        Tau_ptr,
        DOC_IDS_ptr,
        dQ_ptr,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_zb,
        stride_zh,
        stride_zm,
        stride_doc_b,
        stride_doc_t,
        T: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        PER_HEAD_SCALE: tl.constexpr,
        PER_HEAD_TAU: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        HAS_DOC_IDS: tl.constexpr,
        EPS_BASE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_h = tl.program_id(2)

        start_m = pid_m * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < T
        mask_d = offs_d < HEAD_DIM
        mask_md = mask_m[:, None] & mask_d[None, :]

        q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
        k_base = K_ptr + pid_b * stride_qb + pid_h * stride_qh
        v_base = V_ptr + pid_b * stride_qb + pid_h * stride_qh
        o_base = Out_ptr + pid_b * stride_qb + pid_h * stride_qh
        do_base = dO_ptr + pid_b * stride_qb + pid_h * stride_qh
        dq_base = dQ_ptr + pid_b * stride_qb + pid_h * stride_qh
        z_base = Z_ptr + pid_b * stride_zb + pid_h * stride_zh

        scale = tl.load(Scale_ptr + pid_h if PER_HEAD_SCALE else Scale_ptr).to(tl.float32)
        tau = tl.load(Tau_ptr + pid_h if PER_HEAD_TAU else Tau_ptr).to(tl.float32)

        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        o_ptrs = o_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        do_ptrs = do_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        z_ptrs = z_base + offs_m * stride_zm

        q = tl.load(q_ptrs, mask=mask_md, other=0.0)
        o = tl.load(o_ptrs, mask=mask_md, other=0.0).to(tl.float32)
        do = tl.load(do_ptrs, mask=mask_md, other=0.0)
        do_f32 = do.to(tl.float32)
        z_denom = tl.load(z_ptrs, mask=mask_m, other=1.0).to(tl.float32)
        delta_i = tl.sum(do_f32 * o, axis=1)

        if HAS_DOC_IDS:
            doc_base = DOC_IDS_ptr + pid_b * stride_doc_b
            doc_m = tl.load(doc_base + offs_m * stride_doc_t, mask=mask_m, other=-1)

        dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        lo = 0 if WINDOW_SIZE <= 0 else tl.maximum(0, (start_m - WINDOW_SIZE + 1) & ~(BLOCK_N - 1))
        max_n = (pid_m + 1) * BLOCK_M if IS_CAUSAL else T
        max_n = tl.minimum(max_n, T)

        for start_n in range(lo, max_n, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < T

            k_ptrs = k_base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qd
            v_ptrs = v_base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qd

            k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

            raw_dot = tl.dot(q, tl.trans(k), allow_tf32=False)
            u = tl.maximum(scale * (raw_dot - tau), 0.0)
            b = tl.maximum(raw_dot, 0.0)

            if IS_CAUSAL:
                valid_mask = (offs_m[:, None] >= offs_n[None, :]) & mask_m[:, None] & mask_n[None, :]
            else:
                valid_mask = mask_m[:, None] & mask_n[None, :]
            if WINDOW_SIZE > 0:
                valid_mask = valid_mask & ((offs_m[:, None] - offs_n[None, :]) < WINDOW_SIZE)
            if HAS_DOC_IDS:
                doc_n = tl.load(doc_base + offs_n * stride_doc_t, mask=mask_n, other=-2)
                valid_mask = valid_mask & (doc_m[:, None] == doc_n[None, :])
            u = tl.where(valid_mask, u, 0.0)
            b = tl.where(valid_mask, b, 0.0)

            u3 = u * u * u
            dov = tl.dot(do, tl.trans(v), allow_tf32=False)
            dp = (dov - delta_i[:, None]) / z_denom[:, None]
            du_dp = 4.0 * u3 * dp
            d_raw_dot = du_dp * scale + 2.0 * EPS_BASE * b * dp
            d_raw_dot = tl.where(valid_mask, d_raw_dot, 0.0)

            dq_acc += tl.dot(d_raw_dot.to(k.dtype), k, allow_tf32=False)

        dq_ptrs = dq_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        tl.store(dq_ptrs, dq_acc.to(dQ_ptr.dtype.element_ty), mask=mask_md)


class _FlashPurifiedBornAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: torch.Tensor,
        tau: torch.Tensor,
        eps_base: float,
        is_causal: bool,
        eps: float,
        window_size: int = 0,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, H, T, D = q.shape
        target_dtype = torch.get_autocast_gpu_dtype() if (q.is_cuda and torch.is_autocast_enabled()) else v.dtype
        q_c = q.to(target_dtype).contiguous()
        k_c = k.to(target_dtype).contiguous()
        v_c = v.to(target_dtype).contiguous()

        per_head_scale = scale.numel() > 1
        per_head_tau = tau.numel() > 1
        scale_flat = scale.contiguous().view(-1).to(dtype=torch.float32, device=q.device)
        tau_flat = tau.contiguous().view(-1).to(dtype=torch.float32, device=q.device)

        has_doc_ids = doc_ids is not None
        if has_doc_ids:
            doc_ids_c = doc_ids.to(device=q.device, dtype=torch.int32).contiguous()
            doc_ptr = doc_ids_c
            stride_doc_b = doc_ids_c.stride(0)
            stride_doc_t = doc_ids_c.stride(1)
        else:
            doc_ids_c = torch.empty(0, device=q.device, dtype=torch.int32)
            doc_ptr = q_c
            stride_doc_b = 0
            stride_doc_t = 0

        out_f32 = torch.empty((B, H, T, D), dtype=torch.float32, device=q.device)
        z_denom = torch.empty((B, H, T), dtype=torch.float32, device=q.device)

        launch_meta = get_flash_born_launch_meta(T, head_dim=D)
        block_m = launch_meta["BLOCK_M"]
        block_d = triton.next_power_of_2(D)
        grid = (triton.cdiv(T, block_m), B, H)

        _flash_purified_born_fwd_kernel[grid](
            q_c,
            k_c,
            v_c,
            out_f32,
            z_denom,
            scale_flat,
            tau_flat,
            doc_ptr,
            q_c.stride(0),
            q_c.stride(1),
            q_c.stride(2),
            q_c.stride(3),
            k_c.stride(0),
            k_c.stride(1),
            k_c.stride(2),
            k_c.stride(3),
            v_c.stride(0),
            v_c.stride(1),
            v_c.stride(2),
            v_c.stride(3),
            out_f32.stride(0),
            out_f32.stride(1),
            out_f32.stride(2),
            out_f32.stride(3),
            z_denom.stride(0),
            z_denom.stride(1),
            z_denom.stride(2),
            stride_doc_b,
            stride_doc_t,
            T=T,
            HEAD_DIM=D,
            BLOCK_D=block_d,
            IS_CAUSAL=is_causal,
            PER_HEAD_SCALE=per_head_scale,
            PER_HEAD_TAU=per_head_tau,
            WINDOW_SIZE=int(window_size),
            HAS_DOC_IDS=has_doc_ids,
            EPS_BASE=float(eps_base),
            EPS=float(eps),
            **launch_meta,
        )

        ctx.save_for_backward(q_c, k_c, v_c, out_f32, z_denom, scale_flat, tau_flat, doc_ids_c)
        ctx.is_causal = is_causal
        ctx.per_head_scale = per_head_scale
        ctx.per_head_tau = per_head_tau
        ctx.window_size = int(window_size)
        ctx.has_doc_ids = has_doc_ids
        ctx.eps_base = float(eps_base)
        ctx.scale_shape = scale.shape
        ctx.scale_dtype = scale.dtype
        ctx.tau_shape = tau.shape
        ctx.tau_dtype = tau.dtype
        ctx.orig_q_dtype = q.dtype
        ctx.orig_k_dtype = k.dtype
        ctx.orig_v_dtype = v.dtype
        return out_f32.to(target_dtype)

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q_c, k_c, v_c, out, z_denom, scale_flat, tau_flat, doc_ids_c = ctx.saved_tensors
        B, H, T, D = q_c.shape

        do_c = do.to(v_c.dtype).contiguous()
        dq = torch.empty_like(q_c)
        dk = torch.empty_like(k_c)
        dv = torch.empty_like(v_c)
        dscale_f32 = torch.zeros_like(scale_flat, dtype=torch.float32)
        dtau_f32 = torch.zeros_like(tau_flat, dtype=torch.float32)

        if ctx.has_doc_ids:
            doc_ptr = doc_ids_c
            stride_doc_b = doc_ids_c.stride(0)
            stride_doc_t = doc_ids_c.stride(1)
        else:
            doc_ptr = q_c
            stride_doc_b = 0
            stride_doc_t = 0

        launch_meta = get_flash_born_launch_meta(T, head_dim=D)
        block_m = launch_meta["BLOCK_M"]
        block_n = launch_meta["BLOCK_N"]
        block_d = triton.next_power_of_2(D)
        grid_n = (triton.cdiv(T, block_n), B, H)
        grid_m = (triton.cdiv(T, block_m), B, H)

        _flash_purified_born_bwd_kernel[grid_n](
            q_c,
            k_c,
            v_c,
            out,
            do_c,
            z_denom,
            scale_flat,
            tau_flat,
            doc_ptr,
            dk,
            dv,
            dscale_f32,
            dtau_f32,
            q_c.stride(0),
            q_c.stride(1),
            q_c.stride(2),
            q_c.stride(3),
            z_denom.stride(0),
            z_denom.stride(1),
            z_denom.stride(2),
            stride_doc_b,
            stride_doc_t,
            T=T,
            HEAD_DIM=D,
            BLOCK_D=block_d,
            IS_CAUSAL=ctx.is_causal,
            PER_HEAD_SCALE=ctx.per_head_scale,
            PER_HEAD_TAU=ctx.per_head_tau,
            WINDOW_SIZE=ctx.window_size,
            HAS_DOC_IDS=ctx.has_doc_ids,
            EPS_BASE=ctx.eps_base,
            **launch_meta,
        )

        _flash_purified_born_bwd_dq_kernel[grid_m](
            q_c,
            k_c,
            v_c,
            out,
            do_c,
            z_denom,
            scale_flat,
            tau_flat,
            doc_ptr,
            dq,
            q_c.stride(0),
            q_c.stride(1),
            q_c.stride(2),
            q_c.stride(3),
            z_denom.stride(0),
            z_denom.stride(1),
            z_denom.stride(2),
            stride_doc_b,
            stride_doc_t,
            T=T,
            HEAD_DIM=D,
            BLOCK_D=block_d,
            IS_CAUSAL=ctx.is_causal,
            PER_HEAD_SCALE=ctx.per_head_scale,
            PER_HEAD_TAU=ctx.per_head_tau,
            WINDOW_SIZE=ctx.window_size,
            HAS_DOC_IDS=ctx.has_doc_ids,
            EPS_BASE=ctx.eps_base,
            **launch_meta,
        )

        dq_out = dq.to(ctx.orig_q_dtype)
        dk_out = dk.to(ctx.orig_k_dtype)
        dv_out = dv.to(ctx.orig_v_dtype)
        dscale = dscale_f32.view(ctx.scale_shape).to(ctx.scale_dtype)
        dtau = dtau_f32.view(ctx.tau_shape).to(ctx.tau_dtype)
        return dq_out, dk_out, dv_out, dscale, dtau, None, None, None, None, None


@torch.compiler.disable
def _run_triton_flash_purified_born(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale_t: torch.Tensor,
    tau_t: torch.Tensor,
    eps_base: float,
    is_causal: bool,
    eps: float,
    window_size: int = 0,
    doc_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    return _FlashPurifiedBornAttentionFn.apply(
        q, k, v, scale_t, tau_t, eps_base, is_causal, eps, int(window_size), doc_ids
    )


def flash_purified_born_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | torch.Tensor | None = None,
    tau: float | torch.Tensor | None = None,
    eps_base: float = 0.05,
    is_causal: bool = True,
    eps: float = 1e-8,
    window_size: int = 0,
    doc_ids: torch.Tensor | None = None,
    *,
    gamma_h: float | torch.Tensor | None = None,
    tau_h: float | torch.Tensor | None = None,
    causal: bool | None = None,
) -> torch.Tensor:
    """Execute SRAM-tiled Coherence-Thresholded Purified Born Attention via Triton on CUDA.

    Supports sliding-window tile skipping (`window_size > 0`) and document boundary
    masking (`doc_ids` of shape `(B, T)`).
    """
    if scale is None:
        scale = gamma_h if gamma_h is not None else 8.0
    if tau is None:
        tau = tau_h if tau_h is not None else 0.0625
    if causal is not None:
        is_causal = bool(causal)

    D = q.shape[-1]
    is_supported_d = (D >= 16) and (D <= 256) and (D % 16 == 0)

    scale_t = scale if isinstance(scale, torch.Tensor) else torch.tensor(float(scale), dtype=torch.float32, device=q.device)
    tau_t = tau if isinstance(tau, torch.Tensor) else torch.tensor(float(tau), dtype=torch.float32, device=q.device)

    if HAS_TRITON and q.is_cuda and is_supported_d:
        return _run_triton_flash_purified_born(
            q, k, v, scale_t, tau_t, eps_base, is_causal, eps, window_size=window_size, doc_ids=doc_ids
        )

    s = torch.matmul(q.to(v.dtype), k.to(v.dtype).transpose(-1, -2))
    g_v = scale_t.to(s.dtype)
    if g_v.ndim == 1:
        g_v = g_v.view(1, -1, 1, 1)
    t_v = tau_t.to(s.dtype)
    if t_v.ndim == 1:
        t_v = t_v.view(1, -1, 1, 1)
    u = torch.relu(g_v * (s - t_v))
    prob = u.square().square() + float(eps_base) * torch.relu(s).square()
    if is_causal or window_size > 0 or doc_ids is not None:
        prob = _apply_window_and_doc_mask(prob, is_causal=is_causal, window_size=window_size, doc_ids=doc_ids)
    born_weights = prob / (torch.sum(prob, dim=-1, keepdim=True) + eps)
    return torch.matmul(born_weights.to(v.dtype), v)
