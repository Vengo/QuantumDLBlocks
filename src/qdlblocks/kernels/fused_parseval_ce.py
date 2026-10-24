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
"""

from __future__ import annotations

import torch


def quantize_fp8_vocab_transposed(
    m_work: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a `(V, d)` detector table to transposed `float8_e4m3fn` `(d, V)` + FP32 scale."""
    scale = (m_work.abs().amax() / 448.0).float().clamp_min(1e-12)
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
        sc_p = (psi_c.abs().amax() / 448.0).float().clamp_min(1e-12)
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
