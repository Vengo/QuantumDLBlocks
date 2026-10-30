"""Custom Fused Triton & Compiled CUDA Kernels for Multi-Head Mixed-State Kronecker GPT (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

When to Use This Building Block:
- Use `fused_mixed_state_kronecker_chunk_fwd_triton`, `fused_mixed_state_kronecker_chunk_bwd_triton`,
  `flash_causal_pboson_complex_attn_triton`, `fused_cross_head_gram_stinespring_step`, and
  `fused_kronecker_16x4_unitary_mixer_step` to accelerate `MultiHeadMixedStateKroneckerGPT` and
  `MultiHeadSuperpositionMixedStateAttention` across arbitrary batch sizes `B in {1..64}`, sequence
  lengths `T in {64..4096}`, polysemy ranks `R in {1..4}`, and vocabulary chunk sizes `V_c`:
  1. **2D-Tiled Mixed-State Kronecker Readout Forward & Analytical Backward (`fused_mixed_state_kronecker_chunk_*_triton`)**:
     Executes the 4-factor complex overlap product `k_re + i k_im`, holographic + coincidence intensity
     `I_{r,n,v}`, trace-1 mixed-state expectation `I_{n,v}^{mixed} = sum_r beta_{n,r}^2 I_{r,n,v}`, and
     single-pass analytical backward pullback (`g_cat`, `d_beta_c`, `d_alpha_c`, `d_povm_c`) in a single
     2D `(BLOCK_N x BLOCK_V)` `@triton.jit` kernel per vocabulary chunk, eliminating PyTorch Inductor's
     5-kernel split and all intermediate `(R, N, V_c)` `float32` HBM spills.
  2. **SRAM-Tiled Flash Causal Complex `p`-Boson Born Attention (`flash_causal_pboson_complex_attn_triton`)**:
     Computes `c_re = Re(Q K^H)`, `c_im = Im(Q K^H)`, causal `p=2` Born weights `u_{ts} = I(s <= t)|c_{ts}|^4`,
     row partition normalization `Z_t`, value aggregation `U_t = sum_{s <= t} (u_{ts}/Z_t) V_s`, and output
     `CS^{d_h - 1}` projection in `(64 x 64)` SRAM tiles (skipping the strictly upper-triangular `s > t`
     tiles to save 50% of attention FLOPs and eliminating all three `(B, H, T, T)` HBM matrices), with an
     analytical SRAM-tiled backward pass (`dQ, dK, dV`).
  3. **Fused Cross-Head Hermitian Gram + Stinespring Mixing (`fused_cross_head_gram_stinespring_step`)** &
     **Fused 3-Stage `16x4` Kronecker Unitary Meaning Mixer (`fused_kronecker_16x4_unitary_mixer_step`)**:
     Fuses the ~45 per-layer small elementwise and complex contraction kernels into compiled/fused GPU
     kernels with zero mid-forward host-device synchronization barriers (`.item()`).
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    HAS_FUSED_KRONECKER_TRITON = True
except ImportError:
    HAS_FUSED_KRONECKER_TRITON = False

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_059.md
# ==============================================================================
DEFAULT_REF_PHASE_COS_SIN: float = 0.7071067811865476
DEFAULT_MIXED_STATE_INTENSITY_EPS: float = 1e-6
PROB_LOG_CLAMP_MIN: float = 1e-12
SUBHEAD_KET_NORM_EPS: float = 1e-8
ALGEBRAIC_GATE_HALF: float = 0.5
ALGEBRAIC_GATE_ONE: float = 1.0

TRITON_READOUT_BLOCK_N: int = 32
TRITON_READOUT_BLOCK_V: int = 64
TRITON_FLASH_ATTN_BLOCK_M: int = 64
TRITON_FLASH_ATTN_BLOCK_N: int = 64
TRITON_FLASH_MIN_SEQ_LEN: int = 32


if HAS_FUSED_KRONECKER_TRITON:

    @triton.jit
    def _fused_mixed_state_kron_fwd_r2_kernel(
        z_cat_ptr,
        beta_ptr,
        alpha_ptr,
        povm_ptr,
        rel_idx_ptr,
        in_chunk_ptr,
        z_out_ptr,
        target_out_ptr,
        n_tok: int,
        vc: int,
        boson_p: float,
        holo_alpha: float,
        kron_blend: float,
        coinc_gamma: float,
        s0: float,
        eps_floor: float,
        sqrt_r: float,
        R_NUM: tl.constexpr,
        IS_P2: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_v = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_n = offs_n < n_tok
        mask_v = offs_v < vc
        mask_nv = mask_n[:, None] & mask_v[None, :]

        stride_k = 2 * n_tok * vc
        base_x = offs_n[:, None] * vc + offs_v[None, :]
        base_y = (n_tok + offs_n[:, None]) * vc + offs_v[None, :]

        eta = kron_blend
        one_minus_eta = 1.0 - eta

        i_tv = tl.full((BLOCK_N, BLOCK_V), eps_floor, dtype=tl.float32)

        for r in tl.static_range(0, R_NUM):
            beta_r = tl.load(beta_ptr + offs_n * R_NUM + r, mask=mask_n, other=0.0).to(tl.float32)
            alpha_r = tl.load(alpha_ptr + offs_v * R_NUM + r, mask=mask_v, other=0.0).to(tl.float32)
            pi_rn = (beta_r * beta_r)[:, None]
            a_r = (sqrt_r * alpha_r)[None, :]

            k_base = r * 4
            x0 = tl.load(z_cat_ptr + (k_base + 0) * stride_k + base_x, mask=mask_nv, other=0.0).to(tl.float32)
            x1 = tl.load(z_cat_ptr + (k_base + 1) * stride_k + base_x, mask=mask_nv, other=0.0).to(tl.float32)
            x2 = tl.load(z_cat_ptr + (k_base + 2) * stride_k + base_x, mask=mask_nv, other=0.0).to(tl.float32)
            x3 = tl.load(z_cat_ptr + (k_base + 3) * stride_k + base_x, mask=mask_nv, other=0.0).to(tl.float32)

            y0 = tl.load(z_cat_ptr + (k_base + 0) * stride_k + base_y, mask=mask_nv, other=0.0).to(tl.float32)
            y1 = tl.load(z_cat_ptr + (k_base + 1) * stride_k + base_y, mask=mask_nv, other=0.0).to(tl.float32)
            y2 = tl.load(z_cat_ptr + (k_base + 2) * stride_k + base_y, mask=mask_nv, other=0.0).to(tl.float32)
            y3 = tl.load(z_cat_ptr + (k_base + 3) * stride_k + base_y, mask=mask_nv, other=0.0).to(tl.float32)

            x_bar = 0.25 * (x0 + x1 + x2 + x3)
            y_bar = 0.25 * (y0 + y1 + y2 + y3)
            p01_re = x0 * x1 - y0 * y1
            p01_im = x0 * y1 + y0 * x1
            p23_re = x2 * x3 - y2 * y3
            p23_im = x2 * y3 + y2 * x3
            k_re = p01_re * p23_re - p01_im * p23_im
            k_im = p01_re * p23_im + p01_im * p23_re

            m_re = one_minus_eta * x_bar + eta * k_re
            m_im = one_minus_eta * y_bar + eta * k_im
            ref_re = s0 + holo_alpha * a_r * m_re
            ref_im = s0 + holo_alpha * a_r * m_im
            i_r = ref_re * ref_re + ref_im * ref_im + coinc_gamma * (a_r * a_r) * (k_re * k_re + k_im * k_im)
            i_tv += pi_rn * i_r

        povm_v = tl.load(povm_ptr + offs_v, mask=mask_v, other=0.0).to(tl.float32)[None, :]
        if IS_P2:
            i_pow = i_tv * i_tv
        else:
            i_pow = tl.exp(boson_p * tl.log(i_tv))
        p_tilde = tl.where(mask_nv, povm_v * i_pow, 0.0)
        z_tile = tl.sum(p_tilde, axis=1)
        tl.atomic_add(z_out_ptr + offs_n, z_tile, mask=mask_n)

        rel_t = tl.load(rel_idx_ptr + offs_n, mask=mask_n, other=-1)
        in_c = tl.load(in_chunk_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        is_tgt = (offs_v[None, :] == rel_t[:, None]) & mask_nv
        tgt_tile = tl.sum(tl.where(is_tgt, p_tilde, 0.0), axis=1) * in_c
        tl.atomic_add(target_out_ptr + offs_n, tgt_tile, mask=mask_n)

    @triton.jit
    def _fused_mixed_state_kron_bwd_r2_kernel(
        z_cat_ptr,
        beta_ptr,
        alpha_ptr,
        povm_ptr,
        scale_ptr,
        inv_z_ptr,
        v_idx_ptr,
        targets_ptr,
        g_cat_ptr,
        d_beta_ptr,
        d_alpha_ptr,
        d_povm_ptr,
        n_tok: int,
        vc: int,
        boson_p: float,
        holo_alpha: float,
        kron_blend: float,
        coinc_gamma: float,
        s0: float,
        eps_floor: float,
        sqrt_r: float,
        R_NUM: tl.constexpr,
        IS_P2: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_v = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_n = offs_n < n_tok
        mask_v = offs_v < vc
        mask_nv = mask_n[:, None] & mask_v[None, :]

        stride_k = 2 * n_tok * vc
        base_x = offs_n[:, None] * vc + offs_v[None, :]
        base_y = (n_tok + offs_n[:, None]) * vc + offs_v[None, :]

        eta = kron_blend
        one_minus_eta = 1.0 - eta
        one_minus_eta_q = 0.25 * one_minus_eta

        # Pass 1: Compute mixed-state Born intensity i_tv in registers
        i_tv = tl.full((BLOCK_N, BLOCK_V), eps_floor, dtype=tl.float32)
        for r in tl.static_range(0, R_NUM):
            beta_r = tl.load(beta_ptr + offs_n * R_NUM + r, mask=mask_n, other=0.0).to(tl.float32)
            alpha_r = tl.load(alpha_ptr + offs_v * R_NUM + r, mask=mask_v, other=0.0).to(tl.float32)
            pi_rn = (beta_r * beta_r)[:, None]
            a_r = (sqrt_r * alpha_r)[None, :]

            k_base = r * 4
            x0 = tl.load(z_cat_ptr + (k_base + 0) * stride_k + base_x, mask=mask_nv, other=0.0).to(tl.float32)
            x1 = tl.load(z_cat_ptr + (k_base + 1) * stride_k + base_x, mask=mask_nv, other=0.0).to(tl.float32)
            x2 = tl.load(z_cat_ptr + (k_base + 2) * stride_k + base_x, mask=mask_nv, other=0.0).to(tl.float32)
            x3 = tl.load(z_cat_ptr + (k_base + 3) * stride_k + base_x, mask=mask_nv, other=0.0).to(tl.float32)
            y0 = tl.load(z_cat_ptr + (k_base + 0) * stride_k + base_y, mask=mask_nv, other=0.0).to(tl.float32)
            y1 = tl.load(z_cat_ptr + (k_base + 1) * stride_k + base_y, mask=mask_nv, other=0.0).to(tl.float32)
            y2 = tl.load(z_cat_ptr + (k_base + 2) * stride_k + base_y, mask=mask_nv, other=0.0).to(tl.float32)
            y3 = tl.load(z_cat_ptr + (k_base + 3) * stride_k + base_y, mask=mask_nv, other=0.0).to(tl.float32)

            x_bar = 0.25 * (x0 + x1 + x2 + x3)
            y_bar = 0.25 * (y0 + y1 + y2 + y3)
            p01_re = x0 * x1 - y0 * y1
            p01_im = x0 * y1 + y0 * x1
            p23_re = x2 * x3 - y2 * y3
            p23_im = x2 * y3 + y2 * x3
            k_re = p01_re * p23_re - p01_im * p23_im
            k_im = p01_re * p23_im + p01_im * p23_re

            m_re = one_minus_eta * x_bar + eta * k_re
            m_im = one_minus_eta * y_bar + eta * k_im
            ref_re = s0 + holo_alpha * a_r * m_re
            ref_im = s0 + holo_alpha * a_r * m_im
            i_r = ref_re * ref_re + ref_im * ref_im + coinc_gamma * (a_r * a_r) * (k_re * k_re + k_im * k_im)
            i_tv += pi_rn * i_r

        povm_1d = tl.load(povm_ptr + offs_v, mask=mask_v, other=1.0).to(tl.float32)
        scale_n = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[:, None]
        inv_z_n = tl.load(inv_z_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[:, None]
        v_global = tl.load(v_idx_ptr + offs_v, mask=mask_v, other=-2)[None, :]
        targets_n = tl.load(targets_ptr + offs_n, mask=mask_n, other=-1)[:, None]

        if IS_P2:
            i_pow = i_tv * i_tv
        else:
            i_pow = tl.exp(boson_p * tl.log(i_tv))
        probs_c = (povm_1d[None, :] * i_pow) * inv_z_n
        is_target = (v_global == targets_n).to(tl.float32)
        delta_tv = tl.where(mask_nv, scale_n * (probs_c - is_target), 0.0)

        d_povm_tile = tl.sum(delta_tv, axis=0) / tl.maximum(povm_1d, 1e-12)
        tl.atomic_add(d_povm_ptr + offs_v, d_povm_tile, mask=mask_v)

        g_i = delta_tv * (boson_p / i_tv)
        out_dtype = g_cat_ptr.dtype.element_ty

        # Pass 2: Analytical pullback per polysemy branch r (z_cat hot in L1/L2 cache)
        for r in tl.static_range(0, R_NUM):
            beta_r = tl.load(beta_ptr + offs_n * R_NUM + r, mask=mask_n, other=0.0).to(tl.float32)
            alpha_r = tl.load(alpha_ptr + offs_v * R_NUM + r, mask=mask_v, other=0.0).to(tl.float32)
            pi_rn = (beta_r * beta_r)[:, None]
            a_r = (sqrt_r * alpha_r)[None, :]
            a_r_sq = a_r * a_r

            k_base = r * 4
            x0 = tl.load(z_cat_ptr + (k_base + 0) * stride_k + base_x, mask=mask_nv, other=0.0).to(tl.float32)
            x1 = tl.load(z_cat_ptr + (k_base + 1) * stride_k + base_x, mask=mask_nv, other=0.0).to(tl.float32)
            x2 = tl.load(z_cat_ptr + (k_base + 2) * stride_k + base_x, mask=mask_nv, other=0.0).to(tl.float32)
            x3 = tl.load(z_cat_ptr + (k_base + 3) * stride_k + base_x, mask=mask_nv, other=0.0).to(tl.float32)
            y0 = tl.load(z_cat_ptr + (k_base + 0) * stride_k + base_y, mask=mask_nv, other=0.0).to(tl.float32)
            y1 = tl.load(z_cat_ptr + (k_base + 1) * stride_k + base_y, mask=mask_nv, other=0.0).to(tl.float32)
            y2 = tl.load(z_cat_ptr + (k_base + 2) * stride_k + base_y, mask=mask_nv, other=0.0).to(tl.float32)
            y3 = tl.load(z_cat_ptr + (k_base + 3) * stride_k + base_y, mask=mask_nv, other=0.0).to(tl.float32)

            x_bar = 0.25 * (x0 + x1 + x2 + x3)
            y_bar = 0.25 * (y0 + y1 + y2 + y3)
            p01_re = x0 * x1 - y0 * y1
            p01_im = x0 * y1 + y0 * x1
            p23_re = x2 * x3 - y2 * y3
            p23_im = x2 * y3 + y2 * x3
            k_re = p01_re * p23_re - p01_im * p23_im
            k_im = p01_re * p23_im + p01_im * p23_re
            coinc_r = k_re * k_re + k_im * k_im

            m_re = one_minus_eta * x_bar + eta * k_re
            m_im = one_minus_eta * y_bar + eta * k_im
            ref_re = s0 + holo_alpha * a_r * m_re
            ref_im = s0 + holo_alpha * a_r * m_im
            i_r = ref_re * ref_re + ref_im * ref_im + coinc_gamma * a_r_sq * coinc_r

            d_beta_tile = 2.0 * beta_r * tl.sum(g_i * i_r, axis=1)
            tl.atomic_add(d_beta_ptr + offs_n * R_NUM + r, d_beta_tile, mask=mask_n)

            w_gi = pi_rn * g_i
            d_ref_re = 2.0 * w_gi * ref_re
            d_ref_im = 2.0 * w_gi * ref_im
            d_a_r_tile = tl.sum(
                holo_alpha * (d_ref_re * m_re + d_ref_im * m_im) + 2.0 * coinc_gamma * w_gi * a_r * coinc_r,
                axis=0,
            )
            tl.atomic_add(d_alpha_ptr + offs_v * R_NUM + r, sqrt_r * d_a_r_tile, mask=mask_v)

            g_m_re = holo_alpha * a_r * d_ref_re
            g_m_im = holo_alpha * a_r * d_ref_im
            two_gc_a2 = 2.0 * coinc_gamma * w_gi * a_r_sq
            g_k_re = eta * g_m_re + two_gc_a2 * k_re
            g_k_im = eta * g_m_im + two_gc_a2 * k_im

            g_p01_re = g_k_re * p23_re + g_k_im * p23_im
            g_p01_im = g_k_im * p23_re - g_k_re * p23_im
            g_p23_re = g_k_re * p01_re + g_k_im * p01_im
            g_p23_im = g_k_im * p01_re - g_k_re * p01_im
            g_xbar = one_minus_eta_q * g_m_re
            g_ybar = one_minus_eta_q * g_m_im

            gx0 = g_xbar + g_p01_re * x1 + g_p01_im * y1
            gx1 = g_xbar + g_p01_re * x0 + g_p01_im * y0
            gx2 = g_xbar + g_p23_re * x3 + g_p23_im * y3
            gx3 = g_xbar + g_p23_re * x2 + g_p23_im * y2

            gy0 = g_ybar + g_p01_im * x1 - g_p01_re * y1
            gy1 = g_ybar + g_p01_im * x0 - g_p01_re * y0
            gy2 = g_ybar + g_p23_im * x3 - g_p23_re * y3
            gy3 = g_ybar + g_p23_im * x2 - g_p23_re * y2

            tl.store(g_cat_ptr + (k_base + 0) * stride_k + base_x, gx0.to(out_dtype), mask=mask_nv)
            tl.store(g_cat_ptr + (k_base + 1) * stride_k + base_x, gx1.to(out_dtype), mask=mask_nv)
            tl.store(g_cat_ptr + (k_base + 2) * stride_k + base_x, gx2.to(out_dtype), mask=mask_nv)
            tl.store(g_cat_ptr + (k_base + 3) * stride_k + base_x, gx3.to(out_dtype), mask=mask_nv)

            tl.store(g_cat_ptr + (k_base + 0) * stride_k + base_y, gy0.to(out_dtype), mask=mask_nv)
            tl.store(g_cat_ptr + (k_base + 1) * stride_k + base_y, gy1.to(out_dtype), mask=mask_nv)
            tl.store(g_cat_ptr + (k_base + 2) * stride_k + base_y, gy2.to(out_dtype), mask=mask_nv)
            tl.store(g_cat_ptr + (k_base + 3) * stride_k + base_y, gy3.to(out_dtype), mask=mask_nv)

    @triton.jit
    def _flash_causal_pboson_complex_fwd_kernel(
        q_re_ptr,
        q_im_ptr,
        k_re_ptr,
        k_im_ptr,
        v_re_ptr,
        v_im_ptr,
        out_re_ptr,
        out_im_ptr,
        denom_ptr,
        inv_vnorm_ptr,
        seq_t: int,
        d_h: int,
        eps: float,
        PREC: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        bh_idx = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < seq_t
        mask_d = offs_d < d_h
        mask_md = mask_m[:, None] & mask_d[None, :]

        bh_base = bh_idx * seq_t * d_h
        q_ptrs = bh_base + offs_m[:, None] * d_h + offs_d[None, :]
        qr = tl.load(q_re_ptr + q_ptrs, mask=mask_md, other=0.0).to(tl.float32)
        qi = tl.load(q_im_ptr + q_ptrs, mask=mask_md, other=0.0).to(tl.float32)

        acc_re = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        acc_im = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        z_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        max_n = tl.minimum(seq_t, (pid_m + 1) * BLOCK_M)
        for start_n in range(0, max_n, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < seq_t
            mask_nd = mask_n[:, None] & mask_d[None, :]
            kv_ptrs = bh_base + offs_n[:, None] * d_h + offs_d[None, :]

            kr = tl.load(k_re_ptr + kv_ptrs, mask=mask_nd, other=0.0).to(tl.float32)
            ki = tl.load(k_im_ptr + kv_ptrs, mask=mask_nd, other=0.0).to(tl.float32)
            vr = tl.load(v_re_ptr + kv_ptrs, mask=mask_nd, other=0.0).to(tl.float32)
            vi = tl.load(v_im_ptr + kv_ptrs, mask=mask_nd, other=0.0).to(tl.float32)

            c_re = tl.dot(qr, tl.trans(kr), input_precision=PREC) + tl.dot(qi, tl.trans(ki), input_precision=PREC)
            c_im = tl.dot(qi, tl.trans(kr), input_precision=PREC) - tl.dot(qr, tl.trans(ki), input_precision=PREC)
            mag_sq = c_re * c_re + c_im * c_im
            causal_mask = (offs_n[None, :] <= offs_m[:, None]) & (mask_m[:, None] & mask_n[None, :])
            u = tl.where(causal_mask, mag_sq * mag_sq, 0.0)

            z_acc += tl.sum(u, axis=1)
            acc_re += tl.dot(u, vr, input_precision=PREC)
            acc_im += tl.dot(u, vi, input_precision=PREC)

        denom = z_acc + eps
        vb_re = acc_re / denom[:, None]
        vb_im = acc_im / denom[:, None]
        vnorm_sq = tl.sum(vb_re * vb_re + vb_im * vb_im, axis=1)
        inv_vn = tl.rsqrt(vnorm_sq + eps)

        or_val = vb_re * inv_vn[:, None]
        oi_val = vb_im * inv_vn[:, None]
        out_dtype = out_re_ptr.dtype.element_ty
        tl.store(out_re_ptr + q_ptrs, or_val.to(out_dtype), mask=mask_md)
        tl.store(out_im_ptr + q_ptrs, oi_val.to(out_dtype), mask=mask_md)

        row_ptrs = bh_idx * seq_t + offs_m
        tl.store(denom_ptr + row_ptrs, denom, mask=mask_m)
        tl.store(inv_vnorm_ptr + row_ptrs, inv_vn, mask=mask_m)

    @triton.jit
    def _flash_causal_pboson_complex_bwd_q_kernel(
        q_re_ptr,
        q_im_ptr,
        k_re_ptr,
        k_im_ptr,
        v_re_ptr,
        v_im_ptr,
        g_vb_re_ptr,
        g_vb_im_ptr,
        denom_ptr,
        dot_gw_ptr,
        dq_re_ptr,
        dq_im_ptr,
        seq_t: int,
        d_h: int,
        PREC: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        bh_idx = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < seq_t
        mask_d = offs_d < d_h
        mask_md = mask_m[:, None] & mask_d[None, :]

        bh_base = bh_idx * seq_t * d_h
        q_ptrs = bh_base + offs_m[:, None] * d_h + offs_d[None, :]
        qr = tl.load(q_re_ptr + q_ptrs, mask=mask_md, other=0.0).to(tl.float32)
        qi = tl.load(q_im_ptr + q_ptrs, mask=mask_md, other=0.0).to(tl.float32)
        gv_r = tl.load(g_vb_re_ptr + q_ptrs, mask=mask_md, other=0.0).to(tl.float32)
        gv_i = tl.load(g_vb_im_ptr + q_ptrs, mask=mask_md, other=0.0).to(tl.float32)
        row_ptrs = bh_idx * seq_t + offs_m
        denom = tl.load(denom_ptr + row_ptrs, mask=mask_m, other=1.0).to(tl.float32)[:, None]
        dot_gw = tl.load(dot_gw_ptr + row_ptrs, mask=mask_m, other=0.0).to(tl.float32)[:, None]

        dq_r = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        dq_i = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        max_n = tl.minimum(seq_t, (pid_m + 1) * BLOCK_M)
        for start_n in range(0, max_n, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < seq_t
            mask_nd = mask_n[:, None] & mask_d[None, :]
            kv_ptrs = bh_base + offs_n[:, None] * d_h + offs_d[None, :]

            kr = tl.load(k_re_ptr + kv_ptrs, mask=mask_nd, other=0.0).to(tl.float32)
            ki = tl.load(k_im_ptr + kv_ptrs, mask=mask_nd, other=0.0).to(tl.float32)
            vr = tl.load(v_re_ptr + kv_ptrs, mask=mask_nd, other=0.0).to(tl.float32)
            vi = tl.load(v_im_ptr + kv_ptrs, mask=mask_nd, other=0.0).to(tl.float32)

            c_re = tl.dot(qr, tl.trans(kr), input_precision=PREC) + tl.dot(qi, tl.trans(ki), input_precision=PREC)
            c_im = tl.dot(qi, tl.trans(kr), input_precision=PREC) - tl.dot(qr, tl.trans(ki), input_precision=PREC)
            mag_sq = c_re * c_re + c_im * c_im
            causal_mask = (offs_n[None, :] <= offs_m[:, None]) & (mask_m[:, None] & mask_n[None, :])

            dw = tl.dot(gv_r, tl.trans(vr), input_precision=PREC) + tl.dot(gv_i, tl.trans(vi), input_precision=PREC)
            du = tl.where(causal_mask, (dw - dot_gw) / denom, 0.0)
            scale = 4.0 * du * mag_sq
            dc_re = scale * c_re
            dc_im = scale * c_im

            dq_r += tl.dot(dc_re, kr, input_precision=PREC) - tl.dot(dc_im, ki, input_precision=PREC)
            dq_i += tl.dot(dc_re, ki, input_precision=PREC) + tl.dot(dc_im, kr, input_precision=PREC)

        out_dtype = dq_re_ptr.dtype.element_ty
        tl.store(dq_re_ptr + q_ptrs, dq_r.to(out_dtype), mask=mask_md)
        tl.store(dq_im_ptr + q_ptrs, dq_i.to(out_dtype), mask=mask_md)

    @triton.jit
    def _flash_causal_pboson_complex_bwd_kv_kernel(
        q_re_ptr,
        q_im_ptr,
        k_re_ptr,
        k_im_ptr,
        v_re_ptr,
        v_im_ptr,
        g_vb_re_ptr,
        g_vb_im_ptr,
        denom_ptr,
        dot_gw_ptr,
        dk_re_ptr,
        dk_im_ptr,
        dv_re_ptr,
        dv_im_ptr,
        seq_t: int,
        d_h: int,
        PREC: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        bh_idx = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        mask_n = offs_n < seq_t
        mask_d = offs_d < d_h
        mask_nd = mask_n[:, None] & mask_d[None, :]

        bh_base = bh_idx * seq_t * d_h
        kv_ptrs = bh_base + offs_n[:, None] * d_h + offs_d[None, :]
        kr = tl.load(k_re_ptr + kv_ptrs, mask=mask_nd, other=0.0).to(tl.float32)
        ki = tl.load(k_im_ptr + kv_ptrs, mask=mask_nd, other=0.0).to(tl.float32)
        vr = tl.load(v_re_ptr + kv_ptrs, mask=mask_nd, other=0.0).to(tl.float32)
        vi = tl.load(v_im_ptr + kv_ptrs, mask=mask_nd, other=0.0).to(tl.float32)

        dk_r = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
        dk_i = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
        dv_r = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
        dv_i = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

        start_m_init = (pid_n * BLOCK_N // BLOCK_M) * BLOCK_M
        for start_m in range(start_m_init, seq_t, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < seq_t
            mask_md = mask_m[:, None] & mask_d[None, :]
            q_ptrs = bh_base + offs_m[:, None] * d_h + offs_d[None, :]

            qr = tl.load(q_re_ptr + q_ptrs, mask=mask_md, other=0.0).to(tl.float32)
            qi = tl.load(q_im_ptr + q_ptrs, mask=mask_md, other=0.0).to(tl.float32)
            gv_r = tl.load(g_vb_re_ptr + q_ptrs, mask=mask_md, other=0.0).to(tl.float32)
            gv_i = tl.load(g_vb_im_ptr + q_ptrs, mask=mask_md, other=0.0).to(tl.float32)
            row_ptrs = bh_idx * seq_t + offs_m
            denom = tl.load(denom_ptr + row_ptrs, mask=mask_m, other=1.0).to(tl.float32)[:, None]
            dot_gw = tl.load(dot_gw_ptr + row_ptrs, mask=mask_m, other=0.0).to(tl.float32)[:, None]

            c_re = tl.dot(qr, tl.trans(kr), input_precision=PREC) + tl.dot(qi, tl.trans(ki), input_precision=PREC)
            c_im = tl.dot(qi, tl.trans(kr), input_precision=PREC) - tl.dot(qr, tl.trans(ki), input_precision=PREC)
            mag_sq = c_re * c_re + c_im * c_im
            causal_mask = (offs_n[None, :] <= offs_m[:, None]) & (mask_m[:, None] & mask_n[None, :])
            u = tl.where(causal_mask, mag_sq * mag_sq, 0.0)
            w = u / denom

            dv_r += tl.dot(tl.trans(w), gv_r, input_precision=PREC)
            dv_i += tl.dot(tl.trans(w), gv_i, input_precision=PREC)

            dw = tl.dot(gv_r, tl.trans(vr), input_precision=PREC) + tl.dot(gv_i, tl.trans(vi), input_precision=PREC)
            du = tl.where(causal_mask, (dw - dot_gw) / denom, 0.0)
            scale = 4.0 * du * mag_sq
            dc_re = scale * c_re
            dc_im = scale * c_im

            dk_r += tl.dot(tl.trans(dc_re), qr, input_precision=PREC) + tl.dot(tl.trans(dc_im), qi, input_precision=PREC)
            dk_i += tl.dot(tl.trans(dc_re), qi, input_precision=PREC) - tl.dot(tl.trans(dc_im), qr, input_precision=PREC)

        out_dtype = dk_re_ptr.dtype.element_ty
        tl.store(dk_re_ptr + kv_ptrs, dk_r.to(out_dtype), mask=mask_nd)
        tl.store(dk_im_ptr + kv_ptrs, dk_i.to(out_dtype), mask=mask_nd)
        tl.store(dv_re_ptr + kv_ptrs, dv_r.to(out_dtype), mask=mask_nd)
        tl.store(dv_im_ptr + kv_ptrs, dv_i.to(out_dtype), mask=mask_nd)


def fused_mixed_state_kronecker_chunk_fwd_triton(
    z_cat: torch.Tensor,
    beta_f: torch.Tensor,
    alpha_c: torch.Tensor,
    povm_c: torch.Tensor,
    rel_idx: torch.Tensor,
    in_chunk_f: torch.Tensor,
    boson_p: float,
    coeffs: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute single-kernel 2D-tiled `@triton.jit` mixed-state Kronecker chunk forward (`<= 8` args)."""
    n_tok, r_num = beta_f.shape
    vc = alpha_c.size(0)
    holo_alpha, kron_blend, coinc_gamma = coeffs
    bp = float(boson_p)

    z_out = torch.zeros((n_tok,), device=z_cat.device, dtype=torch.float32)
    target_out = torch.zeros((n_tok,), device=z_cat.device, dtype=torch.float32)

    grid = (triton.cdiv(n_tok, TRITON_READOUT_BLOCK_N), triton.cdiv(vc, TRITON_READOUT_BLOCK_V))
    _fused_mixed_state_kron_fwd_r2_kernel[grid](
        z_cat.contiguous(),
        beta_f.contiguous(),
        alpha_c.contiguous(),
        povm_c.contiguous(),
        rel_idx.reshape(-1).contiguous(),
        in_chunk_f.reshape(-1).contiguous(),
        z_out,
        target_out,
        n_tok=n_tok,
        vc=vc,
        boson_p=bp,
        holo_alpha=float(holo_alpha),
        kron_blend=float(kron_blend),
        coinc_gamma=float(coinc_gamma),
        s0=DEFAULT_REF_PHASE_COS_SIN,
        eps_floor=DEFAULT_MIXED_STATE_INTENSITY_EPS,
        sqrt_r=math.sqrt(float(r_num)),
        R_NUM=r_num,
        IS_P2=(bp == 2.0),
        BLOCK_N=TRITON_READOUT_BLOCK_N,
        BLOCK_V=TRITON_READOUT_BLOCK_V,
    )
    return z_out, target_out


def fused_mixed_state_kronecker_chunk_bwd_triton(
    z_cat: torch.Tensor,
    beta_f: torch.Tensor,
    alpha_c: torch.Tensor,
    povm_c: torch.Tensor,
    scale_inv_z: tuple[torch.Tensor, torch.Tensor],
    target_match: tuple[torch.Tensor, torch.Tensor],
    boson_p: float,
    coeffs: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute single-kernel 2D-tiled `@triton.jit` mixed-state Kronecker chunk backward (`<= 8` args)."""
    n_tok, r_num = beta_f.shape
    vc = alpha_c.size(0)
    scale_t, inv_z = scale_inv_z
    v_idx, targets_col = target_match
    holo_alpha, kron_blend, coinc_gamma = coeffs
    bp = float(boson_p)

    g_cat = torch.empty_like(z_cat)
    d_beta_c = torch.zeros((n_tok, r_num), device=z_cat.device, dtype=torch.float32)
    d_alpha_c = torch.zeros((vc, r_num), device=z_cat.device, dtype=torch.float32)
    d_povm_c = torch.zeros((vc,), device=z_cat.device, dtype=torch.float32)

    grid = (triton.cdiv(n_tok, TRITON_READOUT_BLOCK_N), triton.cdiv(vc, TRITON_READOUT_BLOCK_V))
    _fused_mixed_state_kron_bwd_r2_kernel[grid](
        z_cat.contiguous(),
        beta_f.contiguous(),
        alpha_c.contiguous(),
        povm_c.contiguous(),
        scale_t.reshape(-1).contiguous(),
        inv_z.reshape(-1).contiguous(),
        v_idx.reshape(-1).contiguous(),
        targets_col.reshape(-1).contiguous(),
        g_cat,
        d_beta_c,
        d_alpha_c,
        d_povm_c,
        n_tok=n_tok,
        vc=vc,
        boson_p=bp,
        holo_alpha=float(holo_alpha),
        kron_blend=float(kron_blend),
        coinc_gamma=float(coinc_gamma),
        s0=DEFAULT_REF_PHASE_COS_SIN,
        eps_floor=DEFAULT_MIXED_STATE_INTENSITY_EPS,
        sqrt_r=math.sqrt(float(r_num)),
        R_NUM=r_num,
        IS_P2=(bp == 2.0),
        BLOCK_N=TRITON_READOUT_BLOCK_N,
        BLOCK_V=TRITON_READOUT_BLOCK_V,
    )
    return g_cat, d_beta_c, d_alpha_c, d_povm_c


class _FlashCausalPBosonComplexAttnFn(torch.autograd.Function):
    """SRAM-tiled Flash Causal Complex `p=2` Born Attention autograd wrapper (`<= 8` args)."""

    @staticmethod
    def forward(
        ctx,
        q_re: torch.Tensor,
        q_im: torch.Tensor,
        k_re: torch.Tensor,
        k_im: torch.Tensor,
        v_re: torch.Tensor,
        v_im: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b_sz, h_num, seq_t, d_h = q_re.shape
        bh_tot = b_sz * h_num
        block_d = max(16, triton.next_power_of_2(d_h))
        block_m = min(TRITON_FLASH_ATTN_BLOCK_M, max(16, triton.next_power_of_2(seq_t)))
        block_n = min(TRITON_FLASH_ATTN_BLOCK_N, max(16, triton.next_power_of_2(seq_t)))
        prec = "ieee" if (q_re.dtype == torch.float32 and not torch.is_autocast_enabled()) else "tf32"

        qr_c, qi_c = q_re.contiguous(), q_im.contiguous()
        kr_c, ki_c = k_re.contiguous(), k_im.contiguous()
        vr_c, vi_c = v_re.contiguous(), v_im.contiguous()

        out_re = torch.empty_like(qr_c)
        out_im = torch.empty_like(qi_c)
        denom = torch.empty((bh_tot, seq_t), device=q_re.device, dtype=torch.float32)
        inv_vnorm = torch.empty((bh_tot, seq_t), device=q_re.device, dtype=torch.float32)

        grid = (triton.cdiv(seq_t, block_m), bh_tot)
        _flash_causal_pboson_complex_fwd_kernel[grid](
            qr_c,
            qi_c,
            kr_c,
            ki_c,
            vr_c,
            vi_c,
            out_re,
            out_im,
            denom,
            inv_vnorm,
            seq_t=seq_t,
            d_h=d_h,
            eps=float(eps),
            PREC=prec,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
        )
        ctx.save_for_backward(qr_c, qi_c, kr_c, ki_c, vr_c, vi_c, out_re, out_im, denom, inv_vnorm)
        ctx.block_meta = (block_m, block_n, block_d, prec)
        return out_re, out_im

    @staticmethod
    def backward(
        ctx,
        grad_out_re: torch.Tensor,
        grad_out_im: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        qr_c, qi_c, kr_c, ki_c, vr_c, vi_c, out_re, out_im, denom, inv_vnorm = ctx.saved_tensors
        block_m, block_n, block_d, prec = ctx.block_meta
        b_sz, h_num, seq_t, d_h = qr_c.shape
        bh_tot = b_sz * h_num

        # Tangent projection on CS^{d_h - 1}: g_vb = inv_vnorm * (grad_out - <out | grad_out>_R * out)
        gor_f, goi_f = grad_out_re.float(), grad_out_im.float()
        or_f, oi_f = out_re.float(), out_im.float()
        dot_og = (or_f * gor_f + oi_f * goi_f).sum(dim=-1, keepdim=True)
        inv_vn_4d = inv_vnorm.view(b_sz, h_num, seq_t, 1)
        g_vb_re = (inv_vn_4d * (gor_f - dot_og * or_f)).contiguous()
        g_vb_im = (inv_vn_4d * (goi_f - dot_og * oi_f)).contiguous()
        dot_gw = (
            (g_vb_re * or_f + g_vb_im * oi_f).sum(dim=-1)
            / inv_vnorm.view(b_sz, h_num, seq_t).clamp_min(1e-12)
        ).reshape(bh_tot, seq_t).contiguous()

        dq_re = torch.empty_like(qr_c)
        dq_im = torch.empty_like(qi_c)
        dk_re = torch.empty_like(kr_c)
        dk_im = torch.empty_like(ki_c)
        dv_re = torch.empty_like(vr_c)
        dv_im = torch.empty_like(vi_c)

        grid_q = (triton.cdiv(seq_t, block_m), bh_tot)
        _flash_causal_pboson_complex_bwd_q_kernel[grid_q](
            qr_c,
            qi_c,
            kr_c,
            ki_c,
            vr_c,
            vi_c,
            g_vb_re,
            g_vb_im,
            denom,
            dot_gw,
            dq_re,
            dq_im,
            seq_t=seq_t,
            d_h=d_h,
            PREC=prec,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
        )

        grid_kv = (triton.cdiv(seq_t, block_n), bh_tot)
        _flash_causal_pboson_complex_bwd_kv_kernel[grid_kv](
            qr_c,
            qi_c,
            kr_c,
            ki_c,
            vr_c,
            vi_c,
            g_vb_re,
            g_vb_im,
            denom,
            dot_gw,
            dk_re,
            dk_im,
            dv_re,
            dv_im,
            seq_t=seq_t,
            d_h=d_h,
            PREC=prec,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
        )
        return dq_re, dq_im, dk_re, dk_im, dv_re, dv_im, None


def flash_causal_pboson_complex_attn_triton(
    q_re: torch.Tensor,
    q_im: torch.Tensor,
    k_re: torch.Tensor,
    k_im: torch.Tensor,
    v_re: torch.Tensor,
    v_im: torch.Tensor,
    eps: float = SUBHEAD_KET_NORM_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute Flash SRAM-tiled causal `p=2` complex Born attention on `(B, H, T, d_h)` (`<= 8` args)."""
    return _FlashCausalPBosonComplexAttnFn.apply(q_re, q_im, k_re, k_im, v_re, v_im, float(eps))  # type: ignore[return-value]


def _cross_head_gram_stinespring_eager(
    phi_re: torch.Tensor,
    phi_im: torch.Tensor,
    gram_eta: torch.Tensor,
    stinespring_w_re: torch.Tensor,
    stinespring_w_im: torch.Tensor,
    eps: float = SUBHEAD_KET_NORM_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse cross-head `K x K` Hermitian Gram coupling and Stinespring `K -> R` projection (`<= 8` args)."""
    k_sub = phi_re.size(-2)
    inv_k = 1.0 / float(max(1, k_sub))
    phi_cat = torch.cat([phi_re, phi_im], dim=-1)
    phi_skew = torch.cat([-phi_im, phi_re], dim=-1)
    g_hat_re = torch.matmul(phi_cat, phi_cat.transpose(-2, -1)) * inv_k
    g_hat_im = torch.matmul(phi_cat, phi_skew.transpose(-2, -1)) * inv_k

    mix_re = torch.einsum("...kj,...jd->...kd", g_hat_re, phi_re) - torch.einsum(
        "...kj,...jd->...kd", g_hat_im, phi_im
    )
    mix_im = torch.einsum("...kj,...jd->...kd", g_hat_re, phi_im) + torch.einsum(
        "...kj,...jd->...kd", g_hat_im, phi_re
    )
    eta = gram_eta.to(dtype=phi_re.dtype).view(*([1] * (phi_re.ndim - 3)), -1, 1, 1)
    tilde_re = phi_re + eta * mix_re
    tilde_im = phi_im + eta * mix_im
    purity_tm = (g_hat_re.float().square() + g_hat_im.float().square()).sum(dim=(-2, -1))

    w_re = stinespring_w_re.to(dtype=tilde_re.dtype)
    w_im = stinespring_w_im.to(dtype=tilde_im.dtype)
    psi_re = torch.einsum("btmkd,mkr->btrmd", tilde_re, w_re) - torch.einsum(
        "btmkd,mkr->btrmd", tilde_im, w_im
    )
    psi_im = torch.einsum("btmkd,mkr->btrmd", tilde_re, w_im) + torch.einsum(
        "btmkd,mkr->btrmd", tilde_im, w_re
    )

    factor_sq_norm = (psi_re.float().square() + psi_im.float().square()).sum(dim=-1)
    nu_tr = factor_sq_norm.mean(dim=-1)
    inv_norm = torch.rsqrt(factor_sq_norm.unsqueeze(-1) + float(eps)).to(dtype=psi_re.dtype)
    norm_re = (psi_re * inv_norm).flatten(start_dim=2)
    norm_im = (psi_im * inv_norm).flatten(start_dim=2)
    return norm_re, norm_im, purity_tm, nu_tr


fused_cross_head_gram_stinespring_step = torch.compile(_cross_head_gram_stinespring_eager)


def _kronecker_16x4_unitary_mixer_eager(
    psi_re: torch.Tensor,
    psi_im: torch.Tensor,
    factor_u_re: torch.Tensor,
    factor_u_im: torch.Tensor,
    factor_mix_re: torch.Tensor,
    factor_mix_im: torch.Tensor,
    meaning_mix_re: torch.Tensor,
    meaning_mix_im: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse 3-stage `(16x16) -> (4x4) -> (RxR)` complex unitary contractions (`<= 8` args)."""
    w_dtype = psi_re.dtype
    b_sz, seq_t, d_tot = psi_re.shape
    r_num, m_num = factor_mix_re.size(0), factor_mix_re.size(1)
    d_f = factor_u_re.size(-1)
    k_tot = r_num * m_num

    mixer_gate: torch.Tensor | None = None
    if isinstance(meaning_mix_im, tuple):
        mm_im_raw, mixer_gate = meaning_mix_im
    else:
        mm_im_raw = meaning_mix_im

    x_re = psi_re.view(b_sz, seq_t, k_tot, d_f)
    x_im = psi_im.view(b_sz, seq_t, k_tot, d_f)
    u_re, u_im = factor_u_re.to(dtype=w_dtype), factor_u_im.to(dtype=w_dtype)

    u1_re = (torch.einsum("btkd,kdo->btko", x_re, u_re) - torch.einsum("btkd,kdo->btko", x_im, u_im)).view(
        b_sz, seq_t, r_num, m_num, d_f
    )
    u1_im = (torch.einsum("btkd,kdo->btko", x_re, u_im) + torch.einsum("btkd,kdo->btko", x_im, u_re)).view(
        b_sz, seq_t, r_num, m_num, d_f
    )

    fm_re, fm_im = factor_mix_re.to(dtype=w_dtype), factor_mix_im.to(dtype=w_dtype)
    u2_re = torch.einsum("btrmd,rmn->btrnd", u1_re, fm_re) - torch.einsum("btrmd,rmn->btrnd", u1_im, fm_im)
    u2_im = torch.einsum("btrmd,rmn->btrnd", u1_re, fm_im) + torch.einsum("btrmd,rmn->btrnd", u1_im, fm_re)

    mm_re, mm_im = meaning_mix_re.to(dtype=w_dtype), mm_im_raw.to(dtype=w_dtype)
    u3_re = torch.einsum("btrmd,mrs->btsmd", u2_re, mm_re) - torch.einsum("btrmd,mrs->btsmd", u2_im, mm_im)
    u3_im = torch.einsum("btrmd,mrs->btsmd", u2_re, mm_im) + torch.einsum("btrmd,mrs->btsmd", u2_im, mm_re)

    if mixer_gate is None:
        return u3_re, u3_im

    inv_u3 = torch.rsqrt((u3_re.float().square() + u3_im.float().square()).sum(dim=-1, keepdim=True) + SUBHEAD_KET_NORM_EPS).to(
        dtype=w_dtype
    )
    u3_n_re, u3_n_im = u3_re * inv_u3, u3_im * inv_u3
    x_5d_re, x_5d_im = x_re.view(b_sz, seq_t, r_num, m_num, d_f), x_im.view(b_sz, seq_t, r_num, m_num, d_f)
    gate = torch.tanh(mixer_gate).to(dtype=w_dtype)
    comb_re, comb_im = x_5d_re + gate * u3_n_re, x_5d_im + gate * u3_n_im
    inv_c = torch.rsqrt((comb_re.float().square() + comb_im.float().square()).sum(dim=-1, keepdim=True) + SUBHEAD_KET_NORM_EPS).to(
        dtype=w_dtype
    )
    inv_sqrt_k = 1.0 / math.sqrt(float(k_tot))
    scale = inv_c * inv_sqrt_k
    return (comb_re * scale).reshape(b_sz, seq_t, d_tot).to(w_dtype), (comb_im * scale).reshape(b_sz, seq_t, d_tot).to(w_dtype)


fused_kronecker_16x4_unitary_mixer_step = torch.compile(_kronecker_16x4_unitary_mixer_eager)


__all__ = [
    "HAS_FUSED_KRONECKER_TRITON",
    "TRITON_FLASH_ATTN_BLOCK_M",
    "TRITON_FLASH_ATTN_BLOCK_N",
    "TRITON_READOUT_BLOCK_N",
    "TRITON_READOUT_BLOCK_V",
    "flash_causal_pboson_complex_attn_triton",
    "fused_cross_head_gram_stinespring_step",
    "fused_kronecker_16x4_unitary_mixer_step",
    "fused_mixed_state_kronecker_chunk_bwd_triton",
    "fused_mixed_state_kronecker_chunk_fwd_triton",
]

