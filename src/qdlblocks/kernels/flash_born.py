"""Fused Flash-Born Attention Triton Kernel (Zero T x T Attention Matrix Materialization in VRAM).

Computes Subspace Projector Born Attention:
    a_ij = max(0, scale * <Q_i, K_j>) * 1(j <= i)
    p_ij = a_ij^2
    O_i  = (sum_j p_ij V_j) / (sum_j p_ij + eps)

Entirely inside GPU SRAM tiles in both the forward and backward passes, reducing
attention memory complexity from O(B * H * T^2) to O(B * H * T).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _flash_born_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        Z_ptr,
        Scale_ptr,
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
        T: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        PER_HEAD_SCALE: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_h = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)

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
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

        o_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        z_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

        max_n = (pid_m + 1) * BLOCK_M if IS_CAUSAL else T
        max_n = tl.minimum(max_n, T)

        for start_n in range(0, max_n, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < T

            k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

            # Compute scaled overlap S_ij = scale * <Q_i, K_j> in SRAM (Tensor Cores for bf16/fp16)
            s = tl.dot(q, tl.trans(k), allow_tf32=False) * scale

            # Positive-cone projection a_ij = max(0, S_ij)
            a = tl.maximum(s, 0.0)

            # Mask out-of-bounds keys and causal future tokens (j > i)
            if IS_CAUSAL:
                valid_mask = (offs_m[:, None] >= offs_n[None, :]) & mask_n[None, :]
            else:
                valid_mask = mask_n[None, :]
            a = tl.where(valid_mask, a, 0.0)

            # Born rule amplitude squaring: p_ij = a_ij^2
            p = a * a

            z_acc += tl.sum(p, axis=1)
            o_acc += tl.dot(p.to(v.dtype), v, allow_tf32=False)

        z_denom = z_acc + EPS
        o_out = o_acc / z_denom[:, None]

        o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        tl.store(o_ptrs, o_out.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None])

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
        dQ_ptr,
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
        T: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        PER_HEAD_SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_h = tl.program_id(2)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        mask_n = offs_n < T

        q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
        k_base = K_ptr + pid_b * stride_qb + pid_h * stride_qh
        v_base = V_ptr + pid_b * stride_qb + pid_h * stride_qh
        o_base = Out_ptr + pid_b * stride_qb + pid_h * stride_qh
        do_base = dO_ptr + pid_b * stride_qb + pid_h * stride_qh
        dq_base = dQ_ptr + pid_b * stride_qb + pid_h * stride_qh
        dk_base = dK_ptr + pid_b * stride_qb + pid_h * stride_qh
        dv_base = dV_ptr + pid_b * stride_qb + pid_h * stride_qh
        z_base = Z_ptr + pid_b * stride_zb + pid_h * stride_zh

        if PER_HEAD_SCALE:
            scale = tl.load(Scale_ptr + pid_h).to(tl.float32)
        else:
            scale = tl.load(Scale_ptr).to(tl.float32)

        k_ptrs = k_base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qd
        v_ptrs = v_base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qd

        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        dk_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
        dv_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
        dscale_acc = 0.0

        start_m = pid_n * BLOCK_N if IS_CAUSAL else 0

        for curr_m in range(start_m, T, BLOCK_M):
            offs_m = curr_m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < T

            q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
            o_ptrs = o_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
            do_ptrs = do_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
            z_ptrs = z_base + offs_m * stride_zm

            q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
            o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
            do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
            do_f32 = do.to(tl.float32)
            z_denom = tl.load(z_ptrs, mask=mask_m, other=1.0).to(tl.float32)

            # Recompute overlap in SRAM: raw_dot = <Q_i, K_j>
            raw_dot = tl.dot(q, tl.trans(k), allow_tf32=False)
            s = raw_dot * scale
            a = tl.maximum(s, 0.0)

            if IS_CAUSAL:
                valid_mask = (offs_m[:, None] >= offs_n[None, :]) & mask_m[:, None] & mask_n[None, :]
            else:
                valid_mask = mask_m[:, None] & mask_n[None, :]
            a = tl.where(valid_mask, a, 0.0)

            p = a * a
            w = p / z_denom[:, None]  # Normalized Born weights in SRAM

            # dV += w^T @ dO
            dv_acc += tl.dot(tl.trans(w.to(do.dtype)), do, allow_tf32=False)

            # dp_ij = (<dO_i, V_j> - <dO_i, O_i>) / z_denom_i
            delta_i = tl.sum(do_f32 * o, axis=1)
            dov = tl.dot(do, tl.trans(v), allow_tf32=False)
            dp = (dov - delta_i[:, None]) / z_denom[:, None]

            # ds_ij = 2 * a_ij * dp_ij
            ds = 2.0 * a * dp
            ds = tl.where(valid_mask, ds, 0.0)

            # Accumulate gradient wrt scale: dL/d(scale) = sum_{i,j} ds_{ij} * <Q_i, K_j>
            dscale_acc += tl.sum(ds * raw_dot)

            ds_typed = ds.to(q.dtype)
            # dK += scale * ds^T @ Q
            dk_acc += tl.dot(tl.trans(ds_typed), q, allow_tf32=False) * scale

            # Atomic add to dQ: dQ += scale * ds @ K
            dq_tile = tl.dot(ds_typed, k, allow_tf32=False) * scale
            dq_ptrs = dq_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
            tl.atomic_add(dq_ptrs, dq_tile, mask=mask_m[:, None])

        dk_ptrs = dk_base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qd
        dv_ptrs = dv_base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qd
        tl.store(dk_ptrs, dk_acc.to(dK_ptr.dtype.element_ty), mask=mask_n[:, None])
        tl.store(dv_ptrs, dv_acc.to(dV_ptr.dtype.element_ty), mask=mask_n[:, None])

        if PER_HEAD_SCALE:
            tl.atomic_add(dScale_ptr + pid_h, dscale_acc)
        else:
            tl.atomic_add(dScale_ptr, dscale_acc)


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
    ) -> torch.Tensor:
        B, H, T, D = q.shape
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()

        per_head_scale = scale.numel() > 1
        scale_flat = scale.contiguous().view(-1).to(dtype=torch.float32, device=q.device)

        out = torch.empty_like(q_c)
        z_denom = torch.empty((B, H, T), dtype=torch.float32, device=q.device)

        block_m = 32 if T >= 32 else 16
        block_n = 32 if T >= 32 else 16
        grid = (triton.cdiv(T, block_m), B, H)

        _flash_born_fwd_kernel[grid](
            q_c,
            k_c,
            v_c,
            out,
            z_denom,
            scale_flat,
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
            T=T,
            HEAD_DIM=D,
            IS_CAUSAL=is_causal,
            PER_HEAD_SCALE=per_head_scale,
            EPS=eps,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
        )

        ctx.save_for_backward(q_c, k_c, v_c, out, z_denom, scale_flat)
        ctx.is_causal = is_causal
        ctx.per_head_scale = per_head_scale
        ctx.scale_shape = scale.shape
        ctx.scale_dtype = scale.dtype
        return out

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q_c, k_c, v_c, out, z_denom, scale_flat = ctx.saved_tensors
        B, H, T, D = q_c.shape

        do_c = do.contiguous()
        # dQ uses atomic_add in float32 for precision
        dq_f32 = torch.zeros((B, H, T, D), dtype=torch.float32, device=q_c.device)
        dk = torch.empty_like(k_c)
        dv = torch.empty_like(v_c)
        dscale_f32 = torch.zeros_like(scale_flat, dtype=torch.float32)

        block_m = 32 if T >= 32 else 16
        block_n = 32 if T >= 32 else 16
        grid = (triton.cdiv(T, block_n), B, H)

        _flash_born_bwd_kernel[grid](
            q_c,
            k_c,
            v_c,
            out,
            do_c,
            z_denom,
            scale_flat,
            dq_f32,
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
            T=T,
            HEAD_DIM=D,
            IS_CAUSAL=ctx.is_causal,
            PER_HEAD_SCALE=ctx.per_head_scale,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
        )

        dq = dq_f32.to(q_c.dtype)
        dscale = dscale_f32.view(ctx.scale_shape).to(ctx.scale_dtype)
        return dq, dk, dv, dscale, None, None


def flash_born_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | torch.Tensor,
    is_causal: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Execute SRAM-tiled Flash-Born Attention via Triton on CUDA (or fallback on CPU).

    Computes:
        a_ij = max(0, scale * <Q_i, K_j>) * 1(j <= i)
        O_i  = (sum_j a_ij^2 V_j) / (sum_j a_ij^2 + eps)

    Never materializes the (B, H, T, T) attention matrix in GPU global memory!

    Args:
        q: Unit-normalized query tensor `(B, H, T, D)` on CUDA (with D a power of 2 >= 16).
        k: Unit-normalized key tensor `(B, H, T, D)`.
        v: Value tensor `(B, H, T, D)`.
        scale: Scalar float or per-head tensor `(1, H, 1, 1)` / `(H,)`.
        is_causal: Whether to apply lower-triangular causal masking.
        eps: Denominator stability floor.

    Returns:
        Attention output tensor `(B, H, T, D)`.
    """
    D = q.shape[-1]
    is_pow2_d = (D >= 16) and ((D & (D - 1)) == 0)

    if not isinstance(scale, torch.Tensor):
        scale_t = torch.tensor(float(scale), dtype=torch.float32, device=q.device)
    else:
        scale_t = scale

    if HAS_TRITON and q.is_cuda and is_pow2_d:
        return _FlashBornAttentionFn.apply(q, k, v, scale_t, is_causal, eps)

    # Fallback PyTorch implementation when on CPU or non-power-of-2 head_dim
    scaled_overlap = torch.matmul(q, k.transpose(-1, -2)) * scale_t
    projected_amp = torch.relu(scaled_overlap)
    prob = projected_amp.square()
    if is_causal:
        T = q.shape[-2]
        mask = torch.tril(torch.ones(T, T, device=q.device, dtype=prob.dtype)).view(1, 1, T, T)
        prob = prob * mask
    born_weights = prob / (torch.sum(prob, dim=-1, keepdim=True) + eps)
    return torch.matmul(born_weights, v)
