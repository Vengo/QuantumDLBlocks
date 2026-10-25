"""Fused GPU Kernels for Parseval muP Convex-Traceless Born Cross-Entropy (`src/qdlblocks/kernels/`).

Provides reusable, script-agnostic fused GPU kernels for QuantumDLBlocks readouts:
1. `quantize_fp8_vocab_transposed`: Pre-quantizes `(V, d)` vocabulary detectors to `float8_e4m3fn`
   once before the token-chunk loop so all chunks share the transposed FP8 tensor-core weights.
2. `fused_fp8_or_bf16_chunk_gemm`: Executes native FP8 (`torch._scaled_mm`) or BF16 Tensor-Core GEMMs
   for a token chunk without FP32 intermediate materialization.
3. `compiled_parseval_convex_traceless_chunk_step`: `@torch.compile` (TorchInductor -> Triton) fused
   SRAM kernel that combines runtime cosine centering, rectified-quadratic peak boost, quadratic
   diffuse fidelity, online log-sum-exp cross-entropy, and analytical backward pullback into a
   single fused GPU kernel launch per chunk.
4. `compiled_m4_coincidence_chunk_step`: `@torch.compile` (TorchInductor -> Triton) fused SRAM
   kernel for 4-subspace (`M = 4`) geometric-mean coincidence cross-entropy (`Wave 34` /
   `EXPERIMENT_PLAN_049.md`), combining per-subspace cosine centering, smooth 4th-root coincidence
   gating, online log-sum-exp cross-entropy, and analytical backward pullback to all 4 subspace
   overlaps `(4, chunk_sz, V_active)`.
"""

from __future__ import annotations

import torch

FP8_E4M3_MAX_VAL: float = 448.0
FP8_SCALE_MIN_CLAMP: float = 1e-12
M4_GEOM_EPS_SQ: float = 1e-4
M4_GEOM_EPS_OFFSET: float = 1e-2
M4_SUBSPACE_SCALE: float = 4.0
M4_INV_SUBSPACE_SCALE: float = 0.25


def quantize_fp8_vocab_transposed(
    m_work: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a `(V, d)` detector table to transposed `float8_e4m3fn` `(d, V)` + FP32 scale."""
    scale = (m_work.abs().amax() / FP8_E4M3_MAX_VAL).float().clamp_min(FP8_SCALE_MIN_CLAMP)
    m_q_t = (m_work / scale).to(torch.float8_e4m3fn).t()
    return m_q_t, scale


def fused_fp8_or_bf16_chunk_gemm(
    psi_c: torch.Tensor,
    m_act: torch.Tensor,
    m_q_t: torch.Tensor | None,
    sc_m: torch.Tensor | None,
    buf_out: torch.Tensor,
    v_act: int,
) -> torch.Tensor:
    """Compute `(c_len, v_act)` chunk overlaps using FP8 or BF16 Tensor Cores."""
    c_len = psi_c.size(0)
    if m_q_t is not None and sc_m is not None and (c_len % 16 == 0):
        sc_p = (psi_c.abs().amax() / FP8_E4M3_MAX_VAL).float().clamp_min(FP8_SCALE_MIN_CLAMP)
        p_q = (psi_c / sc_p).to(torch.float8_e4m3fn)
        full_out = torch._scaled_mm(
            p_q, m_q_t, scale_a=sc_p, scale_b=sc_m, out_dtype=torch.bfloat16
        )
        return full_out[:, :v_act].to(psi_c.dtype)
    out_slice = buf_out[:c_len, :v_act]
    torch.mm(psi_c, m_act.t(), out=out_slice)
    return out_slice


@torch.compile
def compiled_parseval_convex_traceless_chunk_step(
    w_sharp: torch.Tensor,
    w_diff: torch.Tensor,
    yc_safe: torch.Tensor,
    valid_and_scale: tuple[torch.Tensor, torch.Tensor],
    scalars: tuple[float, float, float, float],
    log_prior_act: torch.Tensor | None,
    v_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused TorchInductor/Triton chunk kernel for Parseval muP Convex-Traceless Born CE.

    Fuses all ~25 elementwise and reduction ops over `(chunk_sz, V_active)` into a single
    GPU kernel pass, returning `(loss_c, dprior_c, grad_s, grad_d)` with `grad_s` and `grad_d`
    cast directly to `w_sharp.dtype` (`bfloat16`) for Tensor-Core backward GEMMs.
    """
    valid_f, row_scale = valid_and_scale
    beta_0, lam_sc, gamma, quad_w = scalars

    c_raw = w_sharp.float()
    c_hat = c_raw - c_raw.mean(dim=-1, keepdim=True)
    c_pos = torch.clamp_min(c_hat, 0.0)
    c_sq = c_pos * c_pos
    o_hat = c_hat + gamma * (c_sq - c_sq.mean(dim=-1, keepdim=True))

    u_raw = w_diff.float()
    u_pos = torch.clamp_min(u_raw, 0.0)
    f_quad = u_raw + quad_w * (u_pos * u_pos)
    f_hat = f_quad - f_quad.mean(dim=-1, keepdim=True)

    logits = beta_0 * o_hat + lam_sc * f_hat
    if log_prior_act is not None:
        logits = logits + log_prior_act

    max_logits = torch.amax(logits, dim=-1, keepdim=True)
    logits_shift = logits - max_logits
    target_logits = logits_shift.gather(1, yc_safe)
    exp_logits = torch.exp(logits_shift)
    sum_exp = exp_logits.sum(dim=-1, keepdim=True)
    loss_c = ((torch.log(sum_exp) - target_logits) * valid_f).sum()

    probs = exp_logits / sum_exp
    is_target = (v_idx == yc_safe).float()
    g_c = (probs - is_target) * row_scale
    dprior_c = g_c.sum(dim=0)

    u_s = g_c * beta_0
    u_s_mean = u_s.mean(dim=-1, keepdim=True)
    g_hat = u_s + (2.0 * gamma) * c_pos * (u_s - u_s_mean)
    grad_s = (g_hat - g_hat.mean(dim=-1, keepdim=True)).to(w_sharp.dtype)

    u_d = g_c * lam_sc
    u_d_cent = u_d - u_d.mean(dim=-1, keepdim=True)
    grad_d = (u_d_cent * (1.0 + (2.0 * quad_w) * u_pos)).to(w_diff.dtype)
    return loss_c, dprior_c, grad_s, grad_d


@torch.compile
def compiled_m4_coincidence_chunk_step(
    w_sharp_m: torch.Tensor,
    w_diff: torch.Tensor,
    yc_safe: torch.Tensor,
    valid_and_scale: tuple[torch.Tensor, torch.Tensor],
    scalars: tuple[float, ...],
    log_prior_act: torch.Tensor | None,
    v_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused TorchInductor/Triton chunk kernel for 4-subspace (`M = 4`) Coincidence Born CE.

    Developed and validated in `Wave 34` (`EXPERIMENT_PLAN_049.md` Section 3.2.1).
    Takes `w_sharp_m` of shape `(4, chunk_sz, V_active)` where each subspace slice has
    Product-of-Spheres norm `1 / sqrt(4) = 0.5`, computes per-subspace vocabulary-centered
    cosines `c_m`, mean quadratic peak `Q`, nested-sqrt 4-way geometric coincidence `G`,
    inter-subspace variance penalty `V_sub`, diffuse fidelity `f_hat`, online log-sum-exp CE,
    and analytical backward pullback to `(chunk_loss_sum, grad_s_m, grad_d, dprior_chunk)`.
    """
    valid_f, row_scale = valid_and_scale
    beta_0 = scalars[0]
    lam_sc = scalars[1]
    gamma = scalars[2]
    gamma_c = scalars[3]
    lam_var = scalars[4]
    sub_scale = scalars[5] if len(scalars) > 5 else M4_SUBSPACE_SCALE
    quad_w = scalars[6] if len(scalars) > 6 else 0.5

    u_m = sub_scale * w_sharp_m.float()
    c_m = u_m - u_m.mean(dim=-1, keepdim=True)
    c0, c1, c2, c3 = c_m[0], c_m[1], c_m[2], c_m[3]
    c_bar = M4_INV_SUBSPACE_SCALE * (c0 + c1 + c2 + c3)
    c_pos = torch.clamp_min(c_bar, 0.0)

    a0 = torch.clamp_min(c0, 0.0)
    a1 = torch.clamp_min(c1, 0.0)
    a2 = torch.clamp_min(c2, 0.0)
    a3 = torch.clamp_min(c3, 0.0)

    q_peak = c_pos * c_pos
    p01 = torch.sqrt(a0 * a1 + M4_GEOM_EPS_SQ)
    p23 = torch.sqrt(a2 * a3 + M4_GEOM_EPS_SQ)
    g_root = torch.sqrt(p01 * p23 + M4_GEOM_EPS_SQ)
    g_coinc = g_root - M4_GEOM_EPS_OFFSET
    v_sub = M4_INV_SUBSPACE_SCALE * (c0 * c0 + c1 * c1 + c2 * c2 + c3 * c3) - c_bar * c_bar

    o_raw = c_bar + gamma * q_peak + gamma_c * g_coinc - lam_var * v_sub
    o_hat = o_raw - o_raw.mean(dim=-1, keepdim=True)

    u_raw = w_diff.float()
    u_pos = torch.clamp_min(u_raw, 0.0)
    f_quad = u_raw + quad_w * (u_pos * u_pos)
    f_hat = f_quad - f_quad.mean(dim=-1, keepdim=True)

    logits = beta_0 * o_hat + lam_sc * f_hat
    if log_prior_act is not None:
        logits = logits + log_prior_act

    max_logits = torch.amax(logits, dim=-1, keepdim=True)
    logits_shift = logits - max_logits
    target_logits = logits_shift.gather(1, yc_safe)
    exp_logits = torch.exp(logits_shift)
    sum_exp = exp_logits.sum(dim=-1, keepdim=True)
    chunk_loss_sum = ((torch.log(sum_exp) - target_logits) * valid_f).sum()

    probs = exp_logits / sum_exp
    vocab_arange = torch.arange(probs.size(1), device=probs.device).unsqueeze(0)
    is_target = (vocab_arange == yc_safe).float()
    g_c = (probs - is_target) * row_scale
    dprior_chunk = g_c.sum(dim=0)

    g_o = beta_0 * (g_c - g_c.mean(dim=-1, keepdim=True))
    inv_2g = 0.5 / g_root
    f01 = (p23 * inv_2g) * (0.5 / p01)
    f23 = (p01 * inv_2g) * (0.5 / p23)

    dg_dc0 = torch.where(c0 > 0.0, f01 * a1, torch.zeros_like(c0))
    dg_dc1 = torch.where(c1 > 0.0, f01 * a0, torch.zeros_like(c1))
    dg_dc2 = torch.where(c2 > 0.0, f23 * a3, torch.zeros_like(c2))
    dg_dc3 = torch.where(c3 > 0.0, f23 * a2, torch.zeros_like(c3))

    dq_dc = (0.5 * gamma) * c_pos
    do_dc0 = M4_INV_SUBSPACE_SCALE + dq_dc + gamma_c * dg_dc0 - (0.5 * lam_var) * (c0 - c_bar)
    do_dc1 = M4_INV_SUBSPACE_SCALE + dq_dc + gamma_c * dg_dc1 - (0.5 * lam_var) * (c1 - c_bar)
    do_dc2 = M4_INV_SUBSPACE_SCALE + dq_dc + gamma_c * dg_dc2 - (0.5 * lam_var) * (c2 - c_bar)
    do_dc3 = M4_INV_SUBSPACE_SCALE + dq_dc + gamma_c * dg_dc3 - (0.5 * lam_var) * (c3 - c_bar)

    do_dc = torch.stack((do_dc0, do_dc1, do_dc2, do_dc3), dim=0)
    g_tilde = g_o.unsqueeze(0) * do_dc
    grad_s_m = (sub_scale * (g_tilde - g_tilde.mean(dim=-1, keepdim=True))).to(w_sharp_m.dtype)

    u_d = g_c * lam_sc
    u_d_cent = u_d - u_d.mean(dim=-1, keepdim=True)
    grad_d = (u_d_cent * (1.0 + (2.0 * quad_w) * u_pos)).to(w_diff.dtype)
    return chunk_loss_sum, grad_s_m, grad_d, dprior_chunk


