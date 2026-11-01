"""Wave 49 (`EXPERIMENT_PLAN_064.md`): Vocabulary Readout Operators & Rank-Bucketed Divergence Telemetry.

Factors vocabulary Born-rule readouts out of `standalone_quantum_born_gpt.py` and implements the
**Mean-Centered Isometric Born Trace Readout** (Pages 33–37 of *"Top-1 Accuracy vs Cross-Entropy Divergence"*)
alongside rank-bucketed NLL divergence diagnostics:

1. **`VocabularyReadoutMode.SOFTCAP_LINEAR_CE`**:
   - Signed coherent linear inner product $c_{\\text{sharp}}(t,v) = s_{\\text{eff}}\\langle h_{\\text{final},t}, w_{\\text{lm},v}\\rangle$
     with `15.0 * tanh(z / 15.0)` softcapping via `_CompiledChunkedSoftcapLinearCEFn`.
2. **`VocabularyReadoutMode.RANK2_KRYLOV_BORN`**:
   - Two-mode Krylov-weighted Born log-intensity readout (`Rank2KrylovBornChunkedCEFn`).
3. **`VocabularyReadoutMode.MEAN_CENTERED_ISOMETRIC_TRACE`**:
   - Combines the signed coherent linear overlap $c_{\\text{sharp}}(t,v)$ with the register-averaged,
     mean-centered isometric Hilbert-Schmidt Born trace overlap:
     $$\\tilde{c}_{\\text{trace}}(t,v) = \\frac{1}{H}\\sum_{h=1}^H \\langle \\tilde{P}(\\rho_{t,h}^{\\text{eff}}), \\tilde{P}(E_{v,h})\\rangle - \\mu_t, \\qquad \\mu_t = \\frac{1}{V}\\sum_{v'=1}^V c_{\\text{trace}}(t,v') \\approx \\frac{1}{d_h}$$
     and softcaps $z(t,v) = C_{\\text{cap}}\\tanh\\!\\big((c_{\\text{sharp}}(t,v) + \\beta_{\\text{trace}}\\tilde{c}_{\\text{trace}}(t,v)) / C_{\\text{cap}}\\big)$
     inside a Rule-9 single-pass chunked autograd function (`MeanCenteredIsometricBornTraceCEFn`) whose
     `backward()` executes 0 matmuls and 0 `.pow()` calls.
4. **Rank-Bucketed Readout Divergence Telemetry (`evaluate_readout_divergence_batch`)**:
   - Decomposes validation cross-entropy by predicted target rank (`Rank 1`, `Rank 2–5`, `Rank 6–50`,
     `Rank > 50`) and measures `Top-1`, `Top-5`, `Gap(Top5 - Top1)`, and logit standard deviation.
"""

from __future__ import annotations

from enum import Enum
import math
from typing import Any

import torch
import torch.nn.functional as F

from .impure_density_mirror_block import TRACE_NORM_EPS

# ---------------------------------------------------------------------------
# Top-of-File Named Constants (Rule 7 — Synchronized with EXPERIMENT_PLAN_064.md)
# ---------------------------------------------------------------------------
DEFAULT_KRYLOV_BORN_LOG_TEMP: float = 6.0
DEFAULT_BORN_LOG_INTENSITY_EPS: float = 0.01
DEFAULT_LOGIT_SOFTCAP: float = 15.0
DEFAULT_CE_CHUNK_TOKENS: int = 2048
DEFAULT_ISOMETRIC_TRACE_UNIFORM_MEAN: float = 0.0625
DEFAULT_ISOMETRIC_TRACE_INIT_BETA: float = 2.0


class VocabularyReadoutMode(str, Enum):
    """Strategy enum selecting vocabulary readout operator (Rule 3)."""

    RANK2_KRYLOV_BORN = "rank2_krylov_born"
    SOFTCAP_LINEAR_CE = "softcap_linear_ce"
    MEAN_CENTERED_ISOMETRIC_TRACE = "mean_centered_isometric_trace"


def pack_multihead_wavepacket_density_isometric(
    x_vec: torch.Tensor,
    num_heads: int,
    v_rank: int,
    d_h: int,
    re_pack_mask: torch.Tensor,
    im_pack_mask: torch.Tensor,
    eps: float = TRACE_NORM_EPS,
) -> torch.Tensor:
    """Pack `(..., H * R * 2 * d_h)` wavepackets into `(..., H * d_h^2)` trace-1 isometric Hermitian vectors $\\tilde{P}(\\rho_h)$."""
    prefix_shape = x_vec.shape[:-1]
    d_in = x_vec.size(-1)
    head_ket_dim = num_heads * 2 * d_h
    eff_rank = max(1, d_in // head_ket_dim) if d_in != num_heads * v_rank * 2 * d_h else v_rank
    work_dtype = x_vec.dtype
    psi = x_vec[..., : eff_rank * head_ket_dim].reshape(*prefix_shape, num_heads, eff_rank, 2, d_h)
    v_re = psi[..., 0, :]
    v_im = psi[..., 1, :]

    energy = (v_re.float().square().sum(dim=(-2, -1), keepdim=True) + v_im.float().square().sum(dim=(-2, -1), keepdim=True)).clamp_min(eps)
    inv_norm = torch.rsqrt(energy).to(work_dtype)
    v_re_n = v_re * inv_norm
    v_im_n = v_im * inv_norm

    vt_re = v_re_n.transpose(-2, -1)
    vt_im = v_im_n.transpose(-2, -1)
    rho_re = torch.matmul(vt_re, v_re_n) + torch.matmul(vt_im, v_im_n)
    rho_im = torch.matmul(vt_re, v_im_n) - torch.matmul(vt_im, v_re_n)

    packed = rho_re * re_pack_mask.to(work_dtype) + rho_im * im_pack_mask.to(work_dtype)
    return packed.reshape(*prefix_shape, num_heads * d_h * d_h)


class MeanCenteredIsometricBornTraceCEFn(torch.autograd.Function):
    """Rule-9 single-pass chunked Mean-Centered Isometric Born Trace + Sharp Coherent Linear Softcap CE autograd function.

    Computes forward cross-entropy loss and all analytical gradients `(dh_flat, dp_rho_flat, dw_lm_scaled, dp_e_vocab, d_beta)`
    inside a single chunked forward loop over tokens so `backward()` executes 0 matmuls and 0 `.pow()` calls.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        h_flat: torch.Tensor,
        p_rho_flat: torch.Tensor,
        w_lm_scaled: torch.Tensor,
        p_e_vocab: torch.Tensor,
        targets_flat: torch.Tensor,
        trace_beta: torch.Tensor,
        num_heads: int = 6,
    ) -> torch.Tensor:
        with torch.no_grad():
            n_tok = h_flat.size(0)
            flat_y = targets_flat.reshape(-1).long()
            chunk_sz = max(1, min(int(DEFAULT_CE_CHUNK_TOKENS), n_tok))
            inv_heads = 1.0 / float(max(1, int(num_heads)))
            inv_n_tok = 1.0 / float(max(1, n_tok))
            softcap_f = float(DEFAULT_LOGIT_SOFTCAP)
            inv_softcap = 1.0 / softcap_f

            work_dtype = (
                torch.bfloat16
                if (h_flat.is_cuda and (h_flat.dtype == torch.bfloat16 or torch.is_autocast_enabled()))
                else h_flat.dtype
            )
            h_work = h_flat if h_flat.dtype == work_dtype else h_flat.to(work_dtype)
            pr_work = p_rho_flat if p_rho_flat.dtype == work_dtype else p_rho_flat.to(work_dtype)
            w_work = w_lm_scaled if w_lm_scaled.dtype == work_dtype else w_lm_scaled.to(work_dtype)
            pe_work = p_e_vocab if p_e_vocab.dtype == work_dtype else p_e_vocab.to(work_dtype)
            beta_f = trace_beta.detach().float()

            dh = torch.empty_like(h_work)
            dpr = torch.empty_like(pr_work)
            dw = torch.zeros_like(w_work)
            dpe = torch.zeros_like(pe_work)
            d_beta = torch.zeros((), device=h_flat.device, dtype=torch.float32)
            total_loss = torch.zeros((), device=h_flat.device, dtype=torch.float32)

            for start in range(0, n_tok, chunk_sz):
                end = min(start + chunk_sz, n_tok)
                hc = h_work[start:end]
                prc = pr_work[start:end]
                yc = flat_y[start:end]

                c_sharp = torch.matmul(hc, w_work.t()).float()
                c_trace = torch.matmul(prc, pe_work.t()).float().mul_(inv_heads)
                c_trace_centered = c_trace - c_trace.mean(dim=-1, keepdim=True)
                del c_trace

                z_raw = c_sharp.add(c_trace_centered, alpha=1.0) if False else (c_sharp + beta_f * c_trace_centered)
                del c_sharp
                u_val = torch.tanh(z_raw * inv_softcap)
                z_logits = u_val * softcap_f
                del z_raw

                lse = torch.logsumexp(z_logits, dim=-1, keepdim=True)
                target_z = z_logits.gather(1, yc.unsqueeze(1))
                total_loss.add_((lse - target_z).sum() * inv_n_tok)

                z_logits.sub_(lse).exp_()
                z_logits.scatter_add_(1, yc.unsqueeze(1), torch.full_like(target_z, -1.0))
                z_logits.mul_(inv_n_tok)
                u_val.square_().neg_().add_(1.0)
                grad_raw = z_logits.mul_(u_val)
                del u_val

                d_beta.add_((grad_raw * c_trace_centered).sum())
                del c_trace_centered

                g_sharp_bf = grad_raw.to(work_dtype)
                dh[start:end] = torch.matmul(g_sharp_bf, w_work)
                dw.add_(torch.matmul(g_sharp_bf.t(), hc))
                del g_sharp_bf

                grad_trace = (grad_raw - grad_raw.mean(dim=-1, keepdim=True)) * (beta_f * inv_heads)
                del grad_raw
                g_trace_bf = grad_trace.to(work_dtype)
                del grad_trace
                dpr[start:end] = torch.matmul(g_trace_bf, pe_work)
                dpe.add_(torch.matmul(g_trace_bf.t(), prc))
                del g_trace_bf

            ctx.save_for_backward(
                dh.to(h_flat.dtype),
                dpr.to(p_rho_flat.dtype),
                dw.to(w_lm_scaled.dtype),
                dpe.to(p_e_vocab.dtype),
                d_beta.to(trace_beta.dtype),
            )
            return total_loss

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, torch.Tensor, None]:
        dh, dpr, dw, dpe, d_beta = ctx.saved_tensors
        return (
            dh * grad_output.to(dh.dtype),
            dpr * grad_output.to(dpr.dtype),
            dw * grad_output.to(dw.dtype),
            dpe * grad_output.to(dpe.dtype),
            None,
            d_beta * grad_output.to(d_beta.dtype),
            None,
        )


def eval_mean_centered_isometric_trace_logits(
    h_bt: torch.Tensor,
    p_rho_bt: torch.Tensor,
    w_lm_scaled: torch.Tensor,
    p_e_vocab: torch.Tensor,
    trace_beta: torch.Tensor,
    num_heads: int = 6,
    softcap: float = DEFAULT_LOGIT_SOFTCAP,
) -> torch.Tensor:
    """Evaluate Mean-Centered Isometric Born Trace softcapped logits `(B, T, V)` for validation or generation (`<= 8` args)."""
    work_dtype = h_bt.dtype
    inv_heads = 1.0 / float(max(1, int(num_heads)))
    c_sharp = F.linear(h_bt, w_lm_scaled.to(work_dtype)).float()
    c_trace = F.linear(p_rho_bt.to(work_dtype), p_e_vocab.to(work_dtype)).float().mul_(inv_heads)
    c_trace_centered = c_trace - c_trace.mean(dim=-1, keepdim=True)
    raw_logits = c_sharp + trace_beta.float() * c_trace_centered
    return float(softcap) * torch.tanh(raw_logits / float(softcap))


class Rank2KrylovBornChunkedCEFn(torch.autograd.Function):
    """Rule-9 single-pass chunked Rank-2 Krylov Born log-intensity + tanh-softcap cross-entropy autograd function."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        h_flat: torch.Tensor,
        krylov_weights_flat: torch.Tensor,
        w_lm: torch.Tensor,
        targets_flat: torch.Tensor,
        tau_born: torch.Tensor,
        softcap: float = DEFAULT_LOGIT_SOFTCAP,
        chunk_tokens: int = DEFAULT_CE_CHUNK_TOKENS,
    ) -> torch.Tensor:
        with torch.no_grad():
            n_tok, d_model = h_flat.shape
            d_sub = d_model // 2
            inv_sqrt_sub = 1.0 / math.sqrt(float(max(1, d_sub)))
            flat_y = targets_flat.reshape(-1).long()
            chunk_sz = max(1, min(int(chunk_tokens), n_tok))

            work_dtype = (
                torch.bfloat16
                if (h_flat.is_cuda and (h_flat.dtype == torch.bfloat16 or torch.is_autocast_enabled()))
                else h_flat.dtype
            )
            h_work = h_flat if h_flat.dtype == work_dtype else h_flat.to(work_dtype)
            w_work = w_lm if w_lm.dtype == work_dtype else w_lm.to(work_dtype)
            kw_f32 = krylov_weights_flat.float()

            h1_all = h_work[:, :d_sub] * inv_sqrt_sub
            h2_all = h_work[:, d_sub:] * inv_sqrt_sub
            w1 = w_work[:, :d_sub]
            w2 = w_work[:, d_sub:]

            dh1 = torch.empty_like(h1_all)
            dh2 = torch.empty_like(h2_all)
            dkw = torch.empty_like(kw_f32)
            dw1 = torch.zeros_like(w1)
            dw2 = torch.zeros_like(w2)
            d_tau = torch.zeros((), device=h_flat.device, dtype=torch.float32)
            total_loss = torch.zeros((), device=h_flat.device, dtype=torch.float32)

            inv_n_tok = 1.0 / float(max(1, n_tok))
            softcap_f = float(softcap)
            inv_softcap = 1.0 / softcap_f
            tau_f = tau_born.detach().float()

            for start in range(0, n_tok, chunk_sz):
                end = min(start + chunk_sz, n_tok)
                h1_c = h1_all[start:end]
                h2_c = h2_all[start:end]
                kw_c = kw_f32[start:end]
                yc = flat_y[start:end]

                coh1 = torch.matmul(h1_c, w1.t()).float().mul_(0.5).add_(1.0)
                coh2 = torch.matmul(h2_c, w2.t()).float().mul_(0.5).add_(1.0)
                a1_sq = coh1.square()
                a2_sq = coh2.square()

                lam1_c = kw_c[:, 0:1]
                lam2_c = kw_c[:, 1:2]
                intensity = lam1_c * a1_sq + lam2_c * a2_sq + DEFAULT_BORN_LOG_INTENSITY_EPS
                log_int = torch.log(intensity)
                u_val = torch.tanh((tau_f * inv_softcap) * log_int)
                z_logits = u_val * softcap_f

                lse = torch.logsumexp(z_logits, dim=-1, keepdim=True)
                target_z = z_logits.gather(1, yc.unsqueeze(1))
                total_loss.add_((lse - target_z).sum() * inv_n_tok)

                z_logits.sub_(lse).exp_()
                z_logits.scatter_add_(1, yc.unsqueeze(1), torch.full_like(target_z, -1.0))
                z_logits.mul_(inv_n_tok)
                u_val.square_().neg_().add_(1.0)
                grad_pre_tanh = z_logits.mul_(u_val)
                d_tau.add_((grad_pre_tanh * log_int).sum())
                del log_int, u_val

                grad_int = grad_pre_tanh.mul_(tau_f).div_(intensity)
                del intensity
                dkw[start:end, 0] = (grad_int * a1_sq).sum(dim=-1)
                dkw[start:end, 1] = (grad_int * a2_sq).sum(dim=-1)
                del a1_sq, a2_sq

                coh1.mul_(grad_int).mul_(lam1_c)
                coh2.mul_(grad_int).mul_(lam2_c)
                del grad_int
                ga1 = coh1.to(work_dtype)
                ga2 = coh2.to(work_dtype)
                del coh1, coh2

                dh1[start:end] = torch.matmul(ga1, w1) * inv_sqrt_sub
                dh2[start:end] = torch.matmul(ga2, w2) * inv_sqrt_sub
                dw1.add_(torch.matmul(ga1.t(), h1_c))
                dw2.add_(torch.matmul(ga2.t(), h2_c))

            dh_full = torch.cat([dh1, dh2], dim=-1).to(h_flat.dtype)
            dw_full = torch.cat([dw1, dw2], dim=-1).to(w_lm.dtype)
            ctx.save_for_backward(dh_full, dkw.to(krylov_weights_flat.dtype), dw_full, d_tau.to(tau_born.dtype))
            return total_loss

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, torch.Tensor, None, None]:
        dh_full, dkw, dw_full, d_tau = ctx.saved_tensors
        return (
            dh_full * grad_output.to(dh_full.dtype),
            dkw * grad_output.to(dkw.dtype),
            dw_full * grad_output.to(dw_full.dtype),
            None,
            d_tau * grad_output.to(d_tau.dtype),
            None,
            None,
        )


def eval_rank2_krylov_born_logits(
    h_bt: torch.Tensor,
    krylov_weights_bt: torch.Tensor,
    w_lm: torch.Tensor,
    tau_born: torch.Tensor,
    softcap: float = DEFAULT_LOGIT_SOFTCAP,
) -> torch.Tensor:
    """Evaluate Rank-2 Krylov Born softcapped logits `(B, T, V)` for validation or generation."""
    d_model = h_bt.size(-1)
    d_sub = d_model // 2
    inv_sqrt_sub = 1.0 / math.sqrt(float(max(1, d_sub)))
    work_dtype = h_bt.dtype
    w_work = w_lm.to(work_dtype)

    h1 = h_bt[..., :d_sub] * inv_sqrt_sub
    h2 = h_bt[..., d_sub:] * inv_sqrt_sub
    a1 = F.linear(h1, w_work[:, :d_sub]).float()
    a2 = F.linear(h2, w_work[:, d_sub:]).float()
    coh1 = 1.0 + 0.5 * a1
    coh2 = 1.0 + 0.5 * a2

    lam1 = krylov_weights_bt[..., 0:1].float()
    lam2 = krylov_weights_bt[..., 1:2].float()
    intensity = lam1 * coh1.square() + lam2 * coh2.square() + DEFAULT_BORN_LOG_INTENSITY_EPS
    raw_logits = tau_born.float() * torch.log(intensity)
    return float(softcap) * torch.tanh(raw_logits / float(softcap))


@torch.no_grad()
def evaluate_readout_divergence_batch(
    logits_btv: torch.Tensor,
    targets_bt: torch.Tensor,
) -> dict[str, float]:
    """Compute rank-bucketed NLL divergence diagnostics (`Top-1`, `Top-5`, `Gap`, `NLL(1)`, `NLL(2-5)`, `NLL(6-50)`, `NLL(>50)`)."""
    z = logits_btv.reshape(-1, logits_btv.size(-1)).float()
    y = targets_bt.reshape(-1).long()
    lse = torch.logsumexp(z, dim=-1)
    target_z = z.gather(1, y.unsqueeze(1)).squeeze(1)
    token_nll = lse - target_z

    ranks = (z > target_z.unsqueeze(1)).sum(dim=-1) + 1
    mask_r1 = ranks == 1
    mask_r2_5 = (ranks >= 2) & (ranks <= 5)
    mask_r6_50 = (ranks >= 6) & (ranks <= 50)
    mask_r51_plus = ranks > 50

    top1 = float(mask_r1.float().mean().item())
    top5 = float((ranks <= 5).float().mean().item())
    val_ce = float(token_nll.mean().item())
    return {
        "val_ce": val_ce,
        "val_ppl": float(math.exp(min(20.0, val_ce))),
        "top1_acc": top1,
        "top5_acc": top5,
        "gap_top5_top1": float(top5 - top1),
        "nll_rank1": float(token_nll[mask_r1].mean().item()) if bool(mask_r1.any()) else 0.0,
        "nll_rank2_5": float(token_nll[mask_r2_5].mean().item()) if bool(mask_r2_5.any()) else 0.0,
        "nll_rank6_50": float(token_nll[mask_r6_50].mean().item()) if bool(mask_r6_50.any()) else 0.0,
        "nll_rank51_plus": float(token_nll[mask_r51_plus].mean().item()) if bool(mask_r51_plus.any()) else 0.0,
        "logit_std": float(z.std(dim=-1).mean().item()),
    }


__all__ = [
    "DEFAULT_BORN_LOG_INTENSITY_EPS",
    "DEFAULT_CE_CHUNK_TOKENS",
    "DEFAULT_ISOMETRIC_TRACE_INIT_BETA",
    "DEFAULT_ISOMETRIC_TRACE_UNIFORM_MEAN",
    "DEFAULT_KRYLOV_BORN_LOG_TEMP",
    "DEFAULT_LOGIT_SOFTCAP",
    "MeanCenteredIsometricBornTraceCEFn",
    "Rank2KrylovBornChunkedCEFn",
    "VocabularyReadoutMode",
    "eval_mean_centered_isometric_trace_logits",
    "eval_rank2_krylov_born_logits",
    "evaluate_readout_divergence_batch",
    "pack_multihead_wavepacket_density_isometric",
]
