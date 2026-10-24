"""Flag-Free Parseval muP sqrt(d) Convex-Traceless + Dyadic Chunked Cross-Entropy (Wave 28)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

from ..kernels.fused_parseval_ce import (
    compiled_parseval_convex_traceless_chunk_step,
    fused_fp8_or_bf16_chunk_gemm,
    quantize_fp8_vocab_transposed,
)


@dataclass(frozen=True)
class ParsevalChunkedCEReadoutSpec:
    """Frozen configuration for Flag-Free Parseval muP Dyadic Chunked Cross-Entropy."""

    beta_0: float
    lambda_diff_scaled: float
    traceless_gamma: float = 0.25
    dyadic_coarse_weight: float = 0.14
    chunk_size: int = 4096
    valid_vocab_size: int = 50257
    ignore_index: int = -100
    quad_diffuse_fidelity: float = 0.50
    gemm_precision: str = "bf16"
    num_coarse_buckets: int = 256
    v_to_coarse: torch.Tensor | None = None


def _fp8_or_bf16_mm(x_2d: torch.Tensor, w_2d: torch.Tensor, prec: str = "bf16") -> torch.Tensor:
    """Compute `x_2d @ w_2d.T` using native FP8 (`float8_e4m3fn`) tensor cores or standard GEMM."""
    if (
        x_2d.device.type == "cuda"
        and prec.lower() == "fp8"
        and (x_2d.size(0) % 16 == 0)
        and (x_2d.size(1) % 16 == 0)
        and (w_2d.size(0) % 16 == 0)
    ):
        sc_x = (x_2d.abs().amax() / 448.0).float() + 1e-12
        sc_w = (w_2d.abs().amax() / 448.0).float() + 1e-12
        xq = (x_2d / sc_x).to(torch.float8_e4m3fn)
        wqt = (w_2d / sc_w).to(torch.float8_e4m3fn).t()
        return torch._scaled_mm(xq, wqt, scale_a=sc_x, scale_b=sc_w, out_dtype=torch.bfloat16).to(x_2d.dtype)
    return torch.mm(x_2d, w_2d.t())


def _resolve_coarse_mapping(
    v_active: int, spec: ParsevalChunkedCEReadoutSpec, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build vocabulary-to-coarse bucket mapping `g(v)` and bucket normalizers `1 / |B_g|`."""
    n_c = max(1, min(int(spec.num_coarse_buckets), v_active))
    if spec.v_to_coarse is not None and spec.v_to_coarse.numel() >= v_active:
        v_map = torch.remainder(spec.v_to_coarse[:v_active].to(device=device, dtype=torch.long), n_c)
    else:
        v_map = torch.div(torch.arange(v_active, device=device, dtype=torch.long) * n_c, v_active, rounding_mode="floor")
    cnt = torch.bincount(v_map, minlength=n_c).to(dtype=dtype)
    return v_map, 1.0 / torch.maximum(cnt, torch.ones_like(cnt)), n_c


def _compute_chunk_logits_and_states(
    c_raw: torch.Tensor, u_raw: torch.Tensor, spec: ParsevalChunkedCEReadoutSpec,
    coarse_ctx: tuple[torch.Tensor, torch.Tensor, int], prior_act: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute unified Parseval muP chunk logits and intermediate states `(c_pos, u_pos, c_coarse_pos)`."""
    gamma, beta_0, alpha_c = float(spec.traceless_gamma), float(spec.beta_0), float(spec.dyadic_coarse_weight)
    v_map, inv_b_cnt, n_c = coarse_ctx
    c_raw.sub_(c_raw.mean(dim=-1, keepdim=True))
    c_pos = F.relu(c_raw)
    c_sq = c_pos.square()
    c_sq.sub_(c_sq.mean(dim=-1, keepdim=True))
    logits = c_raw.add(c_sq, alpha=gamma).mul_(beta_0)
    del c_sq
    u_pos = F.relu(u_raw)
    u_raw.addcmul_(u_pos, u_pos, value=float(spec.quad_diffuse_fidelity))
    u_raw.sub_(u_raw.mean(dim=-1, keepdim=True))
    logits.add_(u_raw, alpha=float(spec.lambda_diff_scaled))
    c_coarse_pos: torch.Tensor | None = None
    if alpha_c > 0.0:
        c_coarse = torch.zeros((c_raw.size(0), n_c), device=c_raw.device, dtype=c_raw.dtype)
        c_coarse.scatter_add_(1, v_map.unsqueeze(0).expand(c_raw.size(0), -1), c_raw)
        c_c_hat = c_coarse.mul_(inv_b_cnt.unsqueeze(0))
        c_c_hat.sub_(c_c_hat.mean(dim=-1, keepdim=True))
        c_coarse_pos = F.relu(c_c_hat)
        c_c_sq = c_coarse_pos.square()
        c_c_sq.sub_(c_c_sq.mean(dim=-1, keepdim=True))
        o_c = c_c_hat.add_(c_c_sq, alpha=gamma)
        logits.add_(o_c.index_select(1, v_map), alpha=beta_0 * alpha_c)
    if prior_act is not None:
        logits.add_(prior_act)
    return logits, c_pos, u_pos, c_coarse_pos


class _ParsevalMuPDyadicChunkedCEFn(torch.autograd.Function):
    """Single-pass chunked Parseval muP Convex-Traceless + Dyadic Coarse-to-Fine Born Cross-Entropy."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, psi_sharp: torch.Tensor, psi_diff: torch.Tensor, m_sharp: torch.Tensor,
        m_diff: torch.Tensor, targets: torch.Tensor, spec: ParsevalChunkedCEReadoutSpec,
        log_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        orig_s, orig_d = psi_sharp.shape, psi_diff.shape
        d, r = psi_sharp.size(-1), psi_diff.size(-1)
        psi_s_flat, psi_d_flat = psi_sharp.detach().reshape(-1, d), psi_diff.detach().reshape(-1, r)
        flat_y = targets.reshape(-1).long()
        n_tok, v_vocab = psi_s_flat.size(0), m_sharp.size(0)
        v_act = min(v_vocab, int(spec.valid_vocab_size)) if int(spec.valid_vocab_size) > 0 else v_vocab
        chunk_sz = max(1, min(int(spec.chunk_size), n_tok))
        if v_vocab >= 16384 and psi_s_flat.is_cuda:
            chunk_sz = min(chunk_sz, 2048 if n_tok > 8192 else 1024)
        acc_dt = torch.float64 if psi_sharp.dtype == torch.float64 else torch.float32
        work_dt = (
            torch.get_autocast_gpu_dtype()
            if (psi_s_flat.is_cuda and torch.is_autocast_enabled() and psi_sharp.dtype != torch.float64)
            else psi_s_flat.dtype
        )
        m_s_w, m_d_w = m_sharp.detach().to(work_dt), m_diff.detach().to(work_dt)
        m_s_act, m_d_act = m_s_w[:v_act], m_d_w[:v_act]
        prior_act = log_prior.detach().to(acc_dt).view(1, -1)[:, :v_act] if log_prior is not None else None

        beta_0, lam_sc = float(spec.beta_0), float(spec.lambda_diff_scaled)
        gamma, alpha_c, quad_w = float(spec.traceless_gamma), float(spec.dyadic_coarse_weight), float(spec.quad_diffuse_fidelity)
        ns, nd, nms, nmd = (bool(ctx.needs_input_grad[i]) for i in range(4))
        npr = bool(log_prior is not None and ctx.needs_input_grad[6])
        dpsi_s = torch.empty((n_tok, d), device=psi_s_flat.device, dtype=work_dt) if ns else None
        dpsi_d = torch.empty((n_tok, r), device=psi_d_flat.device, dtype=work_dt) if nd else None
        dm_s = torch.zeros((v_vocab, d), device=m_sharp.device, dtype=work_dt) if nms else None
        dm_d = torch.zeros((v_vocab, r), device=m_diff.device, dtype=work_dt) if nmd else None
        dprior = torch.zeros((v_vocab,), device=psi_s_flat.device, dtype=acc_dt) if npr else None

        with torch.no_grad():
            valid_all = (flat_y != int(spec.ignore_index)) & (flat_y >= 0) & (flat_y < v_act)
            inv_n = 1.0 / torch.maximum(valid_all.sum().to(acc_dt), torch.ones((), device=psi_s_flat.device, dtype=acc_dt))
            total_loss = torch.zeros((), device=psi_s_flat.device, dtype=acc_dt)
            use_fused = bool(psi_s_flat.is_cuda and acc_dt == torch.float32 and alpha_c <= 0.0)

            if use_fused:
                can_fp8 = bool(spec.gemm_precision.lower() == "fp8" and (v_vocab % 16 == 0) and (d % 16 == 0) and (r % 16 == 0))
                ms_qt, sc_ms = quantize_fp8_vocab_transposed(m_s_w) if can_fp8 else (None, None)
                md_qt, sc_md = quantize_fp8_vocab_transposed(m_d_w) if can_fp8 else (None, None)
                buf_ws = torch.empty((chunk_sz, v_vocab), device=psi_s_flat.device, dtype=work_dt)
                buf_wd = torch.empty((chunk_sz, v_vocab), device=psi_s_flat.device, dtype=work_dt)
                v_idx = torch.arange(v_act, device=psi_s_flat.device).unsqueeze(0)
                scalars = (beta_0, lam_sc, gamma, quad_w)
                for start in range(0, n_tok, chunk_sz):
                    end = min(start + chunk_sz, n_tok)
                    psi_s_c, psi_d_c = psi_s_flat[start:end].to(work_dt), psi_d_flat[start:end].to(work_dt)
                    yc_val = valid_all[start:end].unsqueeze(1)
                    yc_safe = torch.where(yc_val, flat_y[start:end].unsqueeze(1), torch.zeros_like(flat_y[start:end].unsqueeze(1)))
                    val_f = yc_val.to(acc_dt)
                    w_s = fused_fp8_or_bf16_chunk_gemm(psi_s_c, m_s_act, ms_qt, sc_ms, buf_ws, v_act)
                    w_d = fused_fp8_or_bf16_chunk_gemm(psi_d_c, m_d_act, md_qt, sc_md, buf_wd, v_act)
                    loss_c, dpr_c, grad_s, grad_d = compiled_parseval_convex_traceless_chunk_step(
                        w_s, w_d, yc_safe, (val_f, val_f * inv_n), scalars, prior_act, v_idx
                    )
                    total_loss.add_(loss_c)
                    if npr and dprior is not None:
                        dprior[:v_act].add_(dpr_c)
                    if ns and dpsi_s is not None:
                        torch.mm(grad_s, m_s_act, out=dpsi_s[start:end])
                    if nms and dm_s is not None:
                        dm_s[:v_act].addmm_(grad_s.t(), psi_s_c)
                    if nd and dpsi_d is not None:
                        torch.mm(grad_d, m_d_act, out=dpsi_d[start:end])
                    if nmd and dm_d is not None:
                        dm_d[:v_act].addmm_(grad_d.t(), psi_d_c)
            else:
                neg_one = torch.full((chunk_sz, 1), -1.0, device=psi_s_flat.device, dtype=acc_dt)
                coarse_ctx = _resolve_coarse_mapping(v_act, spec, psi_s_flat.device, acc_dt)
                v_map, inv_b_cnt, n_c = coarse_ctx
                for start in range(0, n_tok, chunk_sz):
                    end = min(start + chunk_sz, n_tok)
                    c_len = end - start
                    psi_s_c, psi_d_c = psi_s_flat[start:end].to(work_dt), psi_d_flat[start:end].to(work_dt)
                    yc_val = valid_all[start:end].unsqueeze(1)
                    yc_safe = torch.where(yc_val, flat_y[start:end].unsqueeze(1), torch.zeros_like(flat_y[start:end].unsqueeze(1)))
                    val_f = yc_val.to(acc_dt)
                    c_raw = _fp8_or_bf16_mm(psi_s_c, m_s_w, spec.gemm_precision)[:, :v_act].to(acc_dt)
                    u_raw = _fp8_or_bf16_mm(psi_d_c, m_d_w, spec.gemm_precision)[:, :v_act].to(acc_dt)
                    logits, c_pos, u_pos, c_c_pos = _compute_chunk_logits_and_states(c_raw, u_raw, spec, coarse_ctx, prior_act)
                    del c_raw, u_raw
                    logits.sub_(torch.amax(logits, dim=-1, keepdim=True))
                    target_l = logits.gather(1, yc_safe)
                    logits.exp_()
                    sum_exp = logits.sum(dim=-1, keepdim=True)
                    total_loss.add_(((torch.log(sum_exp) - target_l) * val_f).sum())
                    if ns or nd or nms or nmd or npr:
                        g_c = logits.div_(sum_exp)
                        g_c.scatter_add_(1, yc_safe, neg_one[:c_len])
                        g_c.mul_(val_f * inv_n)
                        if npr and dprior is not None:
                            dprior[:v_act].add_(g_c.sum(dim=0))
                        if ns or nms:
                            u_s = g_c * beta_0
                            c_pos.mul_(u_s - u_s.mean(dim=-1, keepdim=True)).mul_(2.0 * gamma)
                            g_hat = u_s.add_(c_pos)
                            if alpha_c > 0.0 and c_c_pos is not None:
                                u_b = torch.zeros((c_len, n_c), device=psi_s_flat.device, dtype=acc_dt)
                                u_b.scatter_add_(1, v_map.unsqueeze(0).expand(c_len, -1), g_c * (beta_0 * alpha_c))
                                g_b = u_b + (2.0 * gamma) * c_c_pos * (u_b - u_b.mean(dim=-1, keepdim=True))
                                g_c_c = (g_b - g_b.mean(dim=-1, keepdim=True)) * inv_b_cnt.unsqueeze(0)
                                g_hat.add_(g_c_c.index_select(1, v_map))
                            g_hat.sub_(g_hat.mean(dim=-1, keepdim=True))
                            grad_s = g_hat.to(work_dt)
                            del g_hat
                            if ns and dpsi_s is not None:
                                torch.mm(grad_s, m_s_act, out=dpsi_s[start:end])
                            if nms and dm_s is not None:
                                dm_s[:v_act].addmm_(grad_s.t(), psi_s_c)
                            del grad_s
                        if nd or nmd:
                            u_d = g_c.mul_(lam_sc)
                            u_d.sub_(u_d.mean(dim=-1, keepdim=True))
                            u_pos.mul_(2.0 * quad_w).add_(1.0)
                            u_d.mul_(u_pos)
                            grad_d = u_d.to(work_dt)
                            if nd and dpsi_d is not None:
                                torch.mm(grad_d, m_d_act, out=dpsi_d[start:end])
                            if nmd and dm_d is not None:
                                dm_d[:v_act].addmm_(grad_d.t(), psi_d_c)
                            del grad_d, u_d
                    del logits, c_pos, u_pos, c_c_pos

        empty = torch.empty(0, device=psi_s_flat.device)
        ctx.save_for_backward(
            dpsi_s.to(psi_sharp.dtype).view(orig_s) if dpsi_s is not None else empty,
            dpsi_d.to(psi_diff.dtype).view(orig_d) if dpsi_d is not None else empty,
            dm_s.to(m_sharp.dtype) if dm_s is not None else empty,
            dm_d.to(m_diff.dtype) if dm_d is not None else empty,
            dprior.view(log_prior.shape).to(log_prior.dtype) if (dprior is not None and log_prior is not None) else empty,
        )
        ctx.needs = (ns, nd, nms, nmd, npr)
        return total_loss * inv_n

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        saved = ctx.saved_tensors
        g = grad_out.to(torch.float64 if any(t.dtype == torch.float64 for t in saved if t.numel() > 0) else torch.float32)
        grads = [(t.to(g.dtype) * g).to(t.dtype) if need else None for t, need in zip(saved, ctx.needs)]
        return grads[0], grads[1], grads[2], grads[3], None, None, grads[4]


def parseval_mup_dyadic_chunked_ce(
    psi_sharp: torch.Tensor, psi_diff: torch.Tensor, m_sharp: torch.Tensor,
    m_diff: torch.Tensor, targets: torch.Tensor, spec: ParsevalChunkedCEReadoutSpec,
    log_prior: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute single-pass chunked Flag-Free Parseval muP Dyadic Born Cross-Entropy."""
    return _ParsevalMuPDyadicChunkedCEFn.apply(psi_sharp, psi_diff, m_sharp, m_diff, targets, spec, log_prior)


def eval_parseval_mup_full_logits(
    psi_sharp: torch.Tensor, psi_diff: torch.Tensor, m_sharp: torch.Tensor,
    m_diff: torch.Tensor, spec: ParsevalChunkedCEReadoutSpec,
    log_prior: torch.Tensor | None = None, targets: torch.Tensor | None = None,
) -> torch.Tensor | dict[str, Any]:
    """Compute full Parseval muP logits, or return validation metrics (`val_ce`, `top1`, `top50`, `R_sharp`)."""
    acc_dt = torch.float64 if psi_sharp.dtype == torch.float64 else torch.float32
    v_act = min(m_sharp.size(0), int(spec.valid_vocab_size)) if int(spec.valid_vocab_size) > 0 else m_sharp.size(0)
    c_raw = F.linear(psi_sharp.to(acc_dt), m_sharp[:v_act].to(acc_dt)).reshape(-1, v_act)
    u_raw = F.linear(psi_diff.to(acc_dt), m_diff[:v_act].to(acc_dt)).reshape(-1, v_act)
    prior_act = log_prior[:v_act].to(acc_dt).view(1, -1) if log_prior is not None else None
    coarse_ctx = _resolve_coarse_mapping(v_act, spec, psi_sharp.device, acc_dt)
    logits_2d, _, _, _ = _compute_chunk_logits_and_states(c_raw, u_raw, spec, coarse_ctx, prior_act)
    logits = logits_2d.view(*psi_sharp.shape[:-1], v_act)
    logits = logits if psi_sharp.dtype == torch.float64 else logits.float()
    if targets is None:
        return logits
    lg_flat, y_flat = logits.reshape(-1, v_act), targets.reshape(-1).long()
    val_ce = float(F.cross_entropy(lg_flat, y_flat, ignore_index=int(spec.ignore_index)).item())
    valid_mask = (y_flat != int(spec.ignore_index)) & (y_flat >= 0) & (y_flat < v_act)
    n_valid = max(1, int(valid_mask.sum().item()))
    tk_idx = torch.topk(lg_flat[valid_mask], k=min(50, v_act), dim=-1).indices
    yv = y_flat[valid_mask].unsqueeze(-1)
    top1 = 100.0 * float((tk_idx[:, :1] == yv).any(dim=-1).sum().item()) / float(n_valid)
    top50 = 100.0 * float((tk_idx == yv).any(dim=-1).sum().item()) / float(n_valid)
    m_f = m_sharp[:v_act].detach().float()
    m_f = m_f - m_f.mean(dim=0, keepdim=True)
    gram = m_f.t() @ m_f
    tr = torch.diagonal(gram).sum()
    r_sharp = float(((tr * tr) / ((gram * gram).sum() + 1e-12)).item())
    return {
        "logits": logits, "val_ce": val_ce, "val_ppl": math.exp(min(20.0, val_ce)),
        "top1": top1, "val_top1": top1, "top50": top50, "val_top50": top50,
        "R_sharp": r_sharp, "sharp_eff_rank": r_sharp,
    }


compute_parseval_mup_logits = eval_parseval_mup_full_logits

