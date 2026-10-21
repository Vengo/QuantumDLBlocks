"""Single-Pass Direct Chunked Cross-Entropy Autograd Functions (Experiment Plan 016).

Eliminates the second 50,304-vocabulary GEMM caused by wrapping chunked cross-entropy
in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`. Instead, computes
chunk logits, accumulates the scalar cross-entropy loss, and accumulates exact analytical
gradients `(dx, dw)` / `(dphi, dm_seed, d_inv_t, d_log_prior)` in a single forward loop
over `chunk_size` token slices, executing zero GEMMs during `.backward()`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..functional import l2_normalize


class _DirectChunkedSoftcapLinearCEFn(torch.autograd.Function):
    """Single-pass chunked linear + tanh-softcap cross-entropy with analytical gradients."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x_2d: torch.Tensor,
        w_lm: torch.Tensor,
        targets: torch.Tensor,
        softcap: float = 30.0,
        chunk_size: int = 16384,
        ignore_index: int = -1,
        valid_vocab_size: int | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        orig_x_shape = x_2d.shape
        d_model = x_2d.size(-1)
        x_flat = x_2d.reshape(-1, d_model)
        flat_y = targets.reshape(-1).long()
        n_tok = x_flat.size(0)
        chunk_sz = max(1, min(int(chunk_size), n_tok))
        softcap_f = float(softcap)
        ignore_idx = int(ignore_index)

        work_dtype = (
            torch.get_autocast_gpu_dtype()
            if (x_flat.is_cuda and torch.is_autocast_enabled())
            else x_flat.dtype
        )
        w_work = w_lm.to(work_dtype)
        bias_f = bias.float().view(1, -1) if bias is not None else None

        need_dx = bool(ctx.needs_input_grad[0])
        need_dw = bool(ctx.needs_input_grad[1])
        need_dbias = bool(bias is not None and ctx.needs_input_grad[7])

        v_vocab = w_work.size(0)
        v_active = (
            min(v_vocab, int(valid_vocab_size))
            if (valid_vocab_size is not None and int(valid_vocab_size) > 0)
            else v_vocab
        )
        dx_work = torch.empty((n_tok, d_model), device=x_flat.device, dtype=work_dtype) if need_dx else None
        dw_work = torch.zeros((v_vocab, d_model), device=w_lm.device, dtype=work_dtype) if need_dw else None
        dbias_accum = (
            torch.zeros((v_vocab,), device=x_flat.device, dtype=torch.float32)
            if need_dbias
            else None
        )
        total_loss = torch.zeros((), device=x_flat.device, dtype=torch.float32)

        valid_all = (flat_y != ignore_idx) & (flat_y >= 0) & (flat_y < v_active)
        valid_count = valid_all.sum().clamp(min=1).to(torch.float32)
        inv_n_tok = 1.0 / valid_count
        inv_softcap = (1.0 / softcap_f) if softcap_f > 0.0 else 0.0

        logits_buf = torch.empty((chunk_sz, v_vocab), device=x_flat.device, dtype=torch.float32)
        neg_one = torch.full((chunk_sz, 1), -1.0, device=x_flat.device, dtype=torch.float32)

        for start in range(0, n_tok, chunk_sz):
            end = min(start + chunk_sz, n_tok)
            c_len = end - start
            xc = x_flat[start:end]
            yc = flat_y[start:end]
            yc_valid = valid_all[start:end].unsqueeze(1)
            yc_safe = torch.where(yc_valid, yc.unsqueeze(1), torch.zeros_like(yc.unsqueeze(1)))
            valid_f = yc_valid.to(torch.float32)
            row_scale = valid_f * inv_n_tok
            xc_work = xc if xc.dtype == work_dtype else xc.to(work_dtype)

            logits = logits_buf[:c_len]

            # Single forward vocabulary projection GEMM per chunk via F.linear
            raw = F.linear(xc_work, w_work)

            if softcap_f > 0.0:
                if bias_f is not None:
                    logits.copy_(raw).add_(bias_f).mul_(inv_softcap).tanh_()
                    raw.copy_(logits)
                    logits.mul_(softcap_f)
                else:
                    raw.mul_(inv_softcap).tanh_()
                    logits.copy_(raw).mul_(softcap_f)
            else:
                logits.copy_(raw)
                if bias_f is not None:
                    logits.add_(bias_f)

            # Single-buffer in-place LogSumExp + Softmax
            max_logits = torch.amax(logits[:, :v_active], dim=-1, keepdim=True)
            logits.sub_(max_logits)
            target_logits = logits.gather(1, yc_safe)
            logits.exp_()
            if v_active < v_vocab:
                logits[:, v_active:].zero_()
            sum_exp = logits.sum(dim=-1, keepdim=True)
            total_loss.add_(((torch.log(sum_exp) - target_logits) * valid_f).sum())

            if need_dx or need_dw or need_dbias:
                logits.div_(sum_exp)
                logits.scatter_add_(1, yc_safe, neg_one[:c_len])
                if v_active < v_vocab:
                    logits[:, v_active:].zero_()
                if softcap_f > 0.0:
                    # raw currently holds tanh(u); compute (1 - tanh(u)^2) in-place
                    raw.square_().neg_().add_(1.0)
                    logits.mul_(raw).mul_(row_scale)
                else:
                    logits.mul_(row_scale)

                if need_dbias and dbias_accum is not None:
                    dbias_accum.add_(logits.sum(dim=0))

                if need_dx or need_dw:
                    raw.copy_(logits)
                    if need_dx and dx_work is not None:
                        torch.mm(raw, w_work, out=dx_work[start:end])
                    if need_dw and dw_work is not None:
                        dw_work.addmm_(raw.t(), xc_work)

        mean_loss = total_loss * inv_n_tok

        dx_saved = (
            dx_work.to(x_2d.dtype).view(orig_x_shape)
            if dx_work is not None
            else torch.empty(0, device=x_flat.device)
        )
        dw_saved = (
            dw_work.to(w_lm.dtype)
            if dw_work is not None
            else torch.empty(0, device=w_lm.device)
        )
        dbias_saved = (
            dbias_accum.view(bias.shape).to(bias.dtype)
            if (need_dbias and dbias_accum is not None and bias is not None)
            else torch.empty(0, device=x_flat.device)
        )
        ctx.save_for_backward(dx_saved, dw_saved, dbias_saved)
        ctx.need_dx = need_dx
        ctx.need_dw = need_dw
        ctx.need_dbias = need_dbias
        return mean_loss

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        dx_saved, dw_saved, dbias_saved = ctx.saved_tensors
        g = grad_out.to(torch.float32)
        dx = (dx_saved.float() * g).to(dx_saved.dtype) if ctx.need_dx else None
        dw = (dw_saved.float() * g).to(dw_saved.dtype) if ctx.need_dw else None
        dbias = (dbias_saved.float() * g).to(dbias_saved.dtype) if ctx.need_dbias else None
        return dx, dw, None, None, None, None, None, dbias


def direct_chunked_softcap_linear_ce(
    x_2d: torch.Tensor,
    w_lm: torch.Tensor,
    targets: torch.Tensor,
    softcap: float = 30.0,
    chunk_size: int = 16384,
    chunk_tokens: int | None = None,
    ignore_index: int = -1,
    valid_vocab_size: int | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute single-pass chunked linear + tanh-softcap cross-entropy loss with analytical gradients.

    Executes only 1 forward vocabulary GEMM (`F.linear`) per token chunk across the entire
    forward+backward pass (eliminating the second forward recomputation GEMM of `checkpoint`).
    """
    eff_chunk = int(chunk_tokens) if chunk_tokens is not None else int(chunk_size)
    return _DirectChunkedSoftcapLinearCEFn.apply(
        x_2d, w_lm, targets, float(softcap), eff_chunk, int(ignore_index), valid_vocab_size, bias
    )


class _DirectChunkedPOVMCEFn(torch.autograd.Function):
    """Single-pass chunked POVM cosine readout cross-entropy with exact spherical Jacobian pullback."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        phi_2d: torch.Tensor,
        m_seed: torch.Tensor,
        inv_t: torch.Tensor,
        targets: torch.Tensor,
        log_prior: torch.Tensor | None = None,
        chunk_size: int = 16384,
        eps: float = 1e-12,
        ignore_index: int = -1,
        valid_vocab_size: int | None = None,
    ) -> torch.Tensor:
        orig_phi_shape = phi_2d.shape
        d_out = phi_2d.size(-1)
        phi_flat = phi_2d.reshape(-1, d_out)
        flat_y = targets.reshape(-1).long()
        n_tok = phi_flat.size(0)
        chunk_sz = max(1, min(int(chunk_size), n_tok))
        eps_f = float(eps)
        ignore_idx = int(ignore_index)

        work_dtype = (
            torch.get_autocast_gpu_dtype()
            if (phi_flat.is_cuda and torch.is_autocast_enabled())
            else phi_flat.dtype
        )

        # Normalize POVM frame once before the chunk loop
        m_f = m_seed.float()
        r_inv = torch.rsqrt(torch.sum(m_f * m_f, dim=-1, keepdim=True) + eps_f)
        m_norm = m_f * r_inv
        m_work = m_norm.to(work_dtype)
        v_vocab = m_work.size(0)
        v_active = (
            min(v_vocab, int(valid_vocab_size))
            if (valid_vocab_size is not None and int(valid_vocab_size) > 0)
            else v_vocab
        )

        is_per_token_beta = inv_t.numel() > 1
        if is_per_token_beta:
            inv_t_flat = inv_t.float().reshape(n_tok, 1)
            inv_t_val = None
        else:
            inv_t_flat = None
            inv_t_val = inv_t.float().reshape(1, 1)

        log_prior_f = log_prior.float().view(1, -1) if log_prior is not None else None

        need_dphi = bool(ctx.needs_input_grad[0])
        need_dm = bool(ctx.needs_input_grad[1])
        need_dinv_t = bool(ctx.needs_input_grad[2])
        need_dprior = bool(log_prior is not None and ctx.needs_input_grad[4])

        dphi_work = torch.empty((n_tok, d_out), device=phi_flat.device, dtype=work_dtype) if need_dphi else None
        dm_work = torch.zeros((v_vocab, d_out), device=m_seed.device, dtype=work_dtype) if need_dm else None
        dinv_t_flat = (
            torch.empty((n_tok, 1), device=phi_flat.device, dtype=torch.float32)
            if (need_dinv_t and is_per_token_beta)
            else None
        )
        dinv_t_scalar = (
            torch.zeros((), device=phi_flat.device, dtype=torch.float32)
            if (need_dinv_t and not is_per_token_beta)
            else None
        )
        dprior_accum = (
            torch.zeros((v_vocab,), device=phi_flat.device, dtype=torch.float32)
            if need_dprior
            else None
        )

        total_loss = torch.zeros((), device=phi_flat.device, dtype=torch.float32)
        valid_all = (flat_y != ignore_idx) & (flat_y >= 0) & (flat_y < v_active)
        valid_count = valid_all.sum().clamp(min=1).to(torch.float32)
        inv_n_tok = 1.0 / valid_count

        logits_buf = torch.empty((chunk_sz, v_vocab), device=phi_flat.device, dtype=torch.float32)
        neg_one = torch.full((chunk_sz, 1), -1.0, device=phi_flat.device, dtype=torch.float32)

        for start in range(0, n_tok, chunk_sz):
            end = min(start + chunk_sz, n_tok)
            c_len = end - start
            phi_c = phi_flat[start:end]
            yc = flat_y[start:end]
            yc_valid = valid_all[start:end].unsqueeze(1)
            yc_safe = torch.where(yc_valid, yc.unsqueeze(1), torch.zeros_like(yc.unsqueeze(1)))
            valid_f = yc_valid.to(torch.float32)
            row_scale = valid_f * inv_n_tok
            phi_work = phi_c if phi_c.dtype == work_dtype else phi_c.to(work_dtype)

            logits = logits_buf[:c_len]

            # Single forward POVM cosine similarity GEMM per chunk via F.linear
            cos_c = F.linear(phi_work, m_work)
            beta_c = inv_t_flat[start:end] if (is_per_token_beta and inv_t_flat is not None) else inv_t_val
            assert beta_c is not None

            logits.copy_(cos_c).mul_(beta_c)
            if log_prior_f is not None:
                logits.add_(log_prior_f)

            max_logits = torch.amax(logits[:, :v_active], dim=-1, keepdim=True)
            logits.sub_(max_logits)
            target_logits = logits.gather(1, yc_safe)
            logits.exp_()
            if v_active < v_vocab:
                logits[:, v_active:].zero_()
            sum_exp = logits.sum(dim=-1, keepdim=True)
            total_loss.add_(((torch.log(sum_exp) - target_logits) * valid_f).sum())

            if need_dphi or need_dm or need_dinv_t or need_dprior:
                logits.div_(sum_exp)
                logits.scatter_add_(1, yc_safe, neg_one[:c_len])
                if v_active < v_vocab:
                    logits[:, v_active:].zero_()
                logits.mul_(row_scale)

                if need_dprior and dprior_accum is not None:
                    dprior_accum.add_(logits.sum(dim=0))

                if need_dinv_t:
                    if is_per_token_beta and dinv_t_flat is not None:
                        dinv_t_flat[start:end] = torch.sum(logits * cos_c.float(), dim=-1, keepdim=True)
                    elif dinv_t_scalar is not None:
                        dinv_t_scalar.add_(torch.sum(logits * cos_c.float()))

                logits.mul_(beta_c)
                cos_c.copy_(logits)
                if need_dphi and dphi_work is not None:
                    torch.mm(cos_c, m_work, out=dphi_work[start:end])
                if need_dm and dm_work is not None:
                    dm_work.addmm_(cos_c.t(), phi_work)

        mean_loss = total_loss * inv_n_tok

        # Exact spherical Jacobian pullback dm_seed = (dm_norm - m_norm * <dm_norm, m_norm>) / ||m_seed||_2
        if need_dm and dm_work is not None:
            dm_norm = dm_work.float()
            radial_dot = torch.sum(dm_norm * m_norm, dim=-1, keepdim=True)
            dm_seed = ((dm_norm - m_norm * radial_dot) * r_inv).to(m_seed.dtype)
        else:
            dm_seed = torch.empty(0, device=m_seed.device)

        if need_dinv_t:
            if is_per_token_beta and dinv_t_flat is not None:
                dinv_t_saved = dinv_t_flat.view(inv_t.shape).to(inv_t.dtype)
            elif dinv_t_scalar is not None:
                dinv_t_saved = dinv_t_scalar.view(inv_t.shape).to(inv_t.dtype)
            else:
                dinv_t_saved = torch.empty(0, device=phi_flat.device)
        else:
            dinv_t_saved = torch.empty(0, device=phi_flat.device)

        dphi_saved = (
            dphi_work.to(phi_2d.dtype).view(orig_phi_shape)
            if dphi_work is not None
            else torch.empty(0, device=phi_flat.device)
        )
        dprior_saved = (
            dprior_accum.view(log_prior.shape).to(log_prior.dtype)
            if (need_dprior and dprior_accum is not None and log_prior is not None)
            else torch.empty(0, device=phi_flat.device)
        )

        ctx.save_for_backward(dphi_saved, dm_seed, dinv_t_saved, dprior_saved)
        ctx.need_dphi = need_dphi
        ctx.need_dm = need_dm
        ctx.need_dinv_t = need_dinv_t
        ctx.need_dprior = need_dprior
        return mean_loss

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        dphi_saved, dm_seed, dinv_t_saved, dprior_saved = ctx.saved_tensors
        g = grad_out.to(torch.float32)
        dphi = (dphi_saved.float() * g).to(dphi_saved.dtype) if ctx.need_dphi else None
        dm = (dm_seed.float() * g).to(dm_seed.dtype) if ctx.need_dm else None
        dinv_t = (dinv_t_saved.float() * g).to(dinv_t_saved.dtype) if ctx.need_dinv_t else None
        dprior = (dprior_saved.float() * g).to(dprior_saved.dtype) if ctx.need_dprior else None
        return dphi, dm, dinv_t, None, dprior, None, None, None, None


def direct_chunked_povm_ce(
    phi_2d: torch.Tensor,
    m_seed: torch.Tensor,
    inv_t: torch.Tensor | float,
    targets: torch.Tensor,
    log_prior: torch.Tensor | None = None,
    chunk_size: int = 16384,
    eps: float = 1e-12,
    chunk_tokens: int | None = None,
    ignore_index: int = -1,
    valid_vocab_size: int | None = None,
) -> torch.Tensor:
    """Compute single-pass chunked POVM cross-entropy with exact analytical spherical gradients.

    Supports both scalar `inv_t` (`()` / `(1,)`) and dynamic per-token `inv_t` (`(N_tok, 1)`),
    optional `log_prior` bias `(V,)`, and exact tangent-space Jacobian pullback onto `m_seed`.
    """
    if not isinstance(inv_t, torch.Tensor):
        inv_t_t = torch.tensor(float(inv_t), device=phi_2d.device, dtype=torch.float32)
    else:
        inv_t_t = inv_t
    eff_chunk = int(chunk_tokens) if chunk_tokens is not None else int(chunk_size)
    return _DirectChunkedPOVMCEFn.apply(
        phi_2d,
        m_seed,
        inv_t_t,
        targets,
        log_prior,
        eff_chunk,
        float(eps),
        int(ignore_index),
        valid_vocab_size,
    )


def build_sampled_candidate_set(
    targets: torch.Tensor,
    vocab_size: int,
    num_candidates: int,
    step: int = 0,
    negative_stride: int = 20011,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a deterministic coprime-strided candidate set containing all active positive targets.

    Returns `(candidate_ids, remapped_targets)` where `candidate_ids` has shape `(P,)` sorted
    ascending in `[0, vocab_size)` and `remapped_targets` has shape `(N_tok,)` with valid
    targets mapped to their exact indices in `[0, P)` and ignored targets (`< 0`) preserved.
    """
    flat_targets = targets.reshape(-1).to(torch.int64)
    v_size = int(vocab_size)
    p_cand = int(num_candidates)

    if p_cand <= 0 or p_cand >= v_size:
        return (
            torch.arange(v_size, device=targets.device, dtype=torch.int64),
            flat_targets,
        )

    stride = int(negative_stride) % v_size
    while stride <= 0 or math.gcd(stride, v_size) != 1:
        stride = (stride + 1) % v_size
        if stride <= 0:
            stride = 1

    valid_mask = (flat_targets >= 0) & (flat_targets < v_size)
    u_targets = torch.unique(flat_targets[valid_mask], sorted=True)

    if u_targets.numel() >= p_cand:
        candidate_ids = u_targets
    else:
        n_needed = int(p_cand - u_targets.numel())
        n_probe = min(v_size, int(p_cand + u_targets.numel()))
        k = torch.arange(n_probe, device=targets.device, dtype=torch.int64) + int(step * p_cand)
        raw_neg = (k * stride) % v_size
        if u_targets.numel() > 0:
            idx = torch.searchsorted(u_targets, raw_neg).clamp(max=u_targets.numel() - 1)
            in_u = u_targets[idx] == raw_neg
            neg_pool = raw_neg[~in_u][:n_needed]
        else:
            neg_pool = raw_neg[:n_needed]
        candidate_ids, _ = torch.sort(torch.cat([u_targets, neg_pool], dim=0))

    remapped = torch.searchsorted(candidate_ids, flat_targets.clamp(min=0))
    remapped = torch.where(valid_mask, remapped, flat_targets)
    return candidate_ids, remapped


def sampled_softcap_linear_ce(
    x: torch.Tensor,
    lm_head_weight: torch.Tensor,
    targets: torch.Tensor,
    num_candidates: int = 10240,
    step: int = 0,
    softcap: float = 15.0,
    chunk_tokens: int = 4096,
    ignore_index: int = -1,
    negative_stride: int = 20011,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute sampled-softmax tanh-softcap linear cross-entropy over `num_candidates` vocabulary rows."""
    candidate_ids, remapped_targets = build_sampled_candidate_set(
        targets=targets,
        vocab_size=lm_head_weight.size(0),
        num_candidates=num_candidates,
        step=step,
        negative_stride=negative_stride,
    )
    w_sub = lm_head_weight[candidate_ids]
    bias_sub = bias[candidate_ids] if bias is not None else None
    return direct_chunked_softcap_linear_ce(
        x_2d=x,
        w_lm=w_sub,
        targets=remapped_targets,
        softcap=softcap,
        chunk_tokens=chunk_tokens,
        ignore_index=ignore_index,
        bias=bias_sub,
    )


def sampled_povm_ce(
    psi: torch.Tensor,
    dm_seed_weight: torch.Tensor,
    wte_weight: torch.Tensor | float | None = None,
    W_lift_weight: torch.Tensor | None = None,
    readout_ortho_weight: torch.Tensor | None = None,
    inv_t: torch.Tensor | float = 10.0,
    targets: torch.Tensor | None = None,
    num_candidates: int = 10240,
    step: int = 0,
    readout_mode: str = "asymmetric",
    log_prior: torch.Tensor | None = None,
    effects_weight: torch.Tensor | None = None,
    chunk_tokens: int = 4096,
    ignore_index: int = -1,
    eps: float = 1e-12,
    negative_stride: int = 20011,
) -> torch.Tensor:
    """Compute sampled-softmax POVM cross-entropy over `num_candidates` vocabulary effects.

    Supports both:
    1. Full 7-argument signature:
       `sampled_povm_ce(psi, dm_seed_weight, wte_weight, W_lift_weight, readout_ortho_weight, inv_t, targets, ...)`
    2. Direct 4-argument signature:
       `sampled_povm_ce(phi_2d, m_seed, inv_t, targets, ...)`
    """
    # Detect 4-positional-argument call: sampled_povm_ce(phi_2d, m_seed, inv_t, targets, ...)
    if targets is None and W_lift_weight is not None and (
        not isinstance(wte_weight, torch.Tensor) or wte_weight.ndim <= 1 or wte_weight.size(-1) == 1
    ):
        actual_inv_t: torch.Tensor | float = wte_weight if wte_weight is not None else inv_t
        actual_targets = W_lift_weight
        v_size = effects_weight.size(0) if effects_weight is not None else dm_seed_weight.size(0)
        candidate_ids, remapped_targets = build_sampled_candidate_set(
            targets=actual_targets,
            vocab_size=v_size,
            num_candidates=num_candidates,
            step=step,
            negative_stride=negative_stride,
        )
        m_sub = (
            effects_weight[candidate_ids]
            if effects_weight is not None
            else dm_seed_weight[candidate_ids]
        )
        prior_sub = log_prior[candidate_ids] if log_prior is not None else None
        return direct_chunked_povm_ce(
            phi_2d=psi,
            m_seed=m_sub,
            inv_t=actual_inv_t,
            targets=remapped_targets,
            log_prior=prior_sub,
            chunk_tokens=chunk_tokens,
            ignore_index=ignore_index,
            eps=eps,
        )

    if targets is None:
        raise ValueError("`targets` must be provided to `sampled_povm_ce`.")

    if effects_weight is not None:
        v_size = effects_weight.size(0)
    elif isinstance(wte_weight, torch.Tensor) and wte_weight.ndim == 2:
        v_size = wte_weight.size(0)
    else:
        v_size = dm_seed_weight.size(0)

    candidate_ids, remapped_targets = build_sampled_candidate_set(
        targets=targets,
        vocab_size=v_size,
        num_candidates=num_candidates,
        step=step,
        negative_stride=negative_stride,
    )
    prior_sub = log_prior[candidate_ids] if log_prior is not None else None

    if readout_mode == "asymmetric" and effects_weight is not None:
        phi_2d = l2_normalize(psi, dim=-1, eps=eps)
        m_sub = effects_weight[candidate_ids]
    else:
        if readout_ortho_weight is not None:
            psi_rot = l2_normalize(F.linear(psi, readout_ortho_weight.to(psi.dtype)), dim=-1, eps=eps)
        else:
            psi_rot = l2_normalize(psi, dim=-1, eps=eps)

        if W_lift_weight is not None and psi_rot.size(-1) != (
            dm_seed_weight.size(-1) if dm_seed_weight is not None else psi_rot.size(-1)
        ):
            if W_lift_weight.size(0) == psi_rot.size(-1):
                phi_2d = l2_normalize(
                    torch.matmul(psi_rot, W_lift_weight.to(psi_rot.dtype)), dim=-1, eps=eps
                )
            else:
                phi_2d = l2_normalize(
                    F.linear(psi_rot, W_lift_weight.to(psi_rot.dtype)), dim=-1, eps=eps
                )
        elif (
            W_lift_weight is not None
            and isinstance(wte_weight, torch.Tensor)
            and psi_rot.size(-1) != wte_weight.size(-1)
        ):
            if W_lift_weight.size(0) == psi_rot.size(-1):
                phi_2d = l2_normalize(
                    torch.matmul(psi_rot, W_lift_weight.to(psi_rot.dtype)), dim=-1, eps=eps
                )
            else:
                phi_2d = l2_normalize(
                    F.linear(psi_rot, W_lift_weight.to(psi_rot.dtype)), dim=-1, eps=eps
                )
        else:
            phi_2d = psi_rot

        if effects_weight is not None and effects_weight.size(-1) == phi_2d.size(-1):
            m_sub = effects_weight[candidate_ids]
        elif (
            isinstance(wte_weight, torch.Tensor)
            and wte_weight.ndim == 2
            and dm_seed_weight is not None
            and wte_weight.shape == dm_seed_weight.shape
        ):
            if readout_mode in ("tied_stiefel", "tied_vocab"):
                m_sub = wte_weight[candidate_ids] + dm_seed_weight[candidate_ids]
            else:
                m_sub = dm_seed_weight[candidate_ids]
        elif dm_seed_weight is not None:
            m_sub = dm_seed_weight[candidate_ids]
        elif isinstance(wte_weight, torch.Tensor):
            m_sub = wte_weight[candidate_ids]
        else:
            raise ValueError("No valid POVM seed or effects weight provided to `sampled_povm_ce`.")

    return direct_chunked_povm_ce(
        phi_2d=phi_2d,
        m_seed=m_sub,
        inv_t=inv_t,
        targets=remapped_targets,
        log_prior=prior_sub,
        chunk_tokens=chunk_tokens,
        ignore_index=ignore_index,
        eps=eps,
    )


class _DirectChunkedTwoBeamKBosonCEFn(torch.autograd.Function):
    """Single-pass chunked Two-Beam k-Boson mixed-state POVM cross-entropy with analytical gradients."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        psi_sharp: torch.Tensor,      # (..., d), unit-norm ket in S^{d-1}
        psi_diff: torch.Tensor,       # (..., r), compressed diffuse ket in R^r
        m_sharp: torch.Tensor,        # (V, d), unit-norm POVM effect directions in S^{d-1}
        m_diff: torch.Tensor,         # (V, r), compressed diffuse POVM effects in R^r
        beta_t: torch.Tensor,         # (..., 1) or scalar ()
        k_copies: torch.Tensor,       # scalar ()
        lambda_diff: torch.Tensor,    # scalar ()
        targets: torch.Tensor,        # (...,)
        log_prior: torch.Tensor | None = None,
        chunk_size: int = 8192,
        ignore_index: int = -1,
        valid_vocab_size: int | None = None,
    ) -> torch.Tensor:
        orig_psi_s_shape = psi_sharp.shape
        orig_psi_d_shape = psi_diff.shape
        d = psi_sharp.size(-1)
        r = psi_diff.size(-1)
        psi_s_flat = psi_sharp.reshape(-1, d)
        psi_d_flat = psi_diff.reshape(-1, r)
        flat_y = targets.reshape(-1).long()
        n_tok = psi_s_flat.size(0)
        v_vocab = m_sharp.size(0)
        v_active = (
            min(v_vocab, int(valid_vocab_size))
            if (valid_vocab_size is not None and int(valid_vocab_size) > 0)
            else v_vocab
        )
        chunk_sz = max(1, min(int(chunk_size), n_tok))
        ignore_idx = int(ignore_index)

        acc_dtype = torch.float64 if psi_sharp.dtype == torch.float64 else torch.float32
        if psi_sharp.dtype == torch.float64:
            work_dtype = torch.float64
        elif psi_s_flat.is_cuda and torch.is_autocast_enabled():
            work_dtype = torch.get_autocast_gpu_dtype()
        else:
            work_dtype = psi_s_flat.dtype

        m_s_work = m_sharp.to(work_dtype)
        m_d_work = m_diff.to(work_dtype)

        k_val = k_copies.to(acc_dtype).reshape(())
        lam_val = lambda_diff.to(acc_dtype).reshape(())
        is_per_token_beta = beta_t.numel() > 1
        beta_flat = (
            beta_t.to(acc_dtype).reshape(n_tok, 1)
            if is_per_token_beta
            else beta_t.to(acc_dtype).reshape(1, 1)
        )
        log_prior_f = log_prior.to(acc_dtype).view(1, -1) if log_prior is not None else None

        need_dpsi_s = bool(ctx.needs_input_grad[0])
        need_dpsi_d = bool(ctx.needs_input_grad[1])
        need_dm_s = bool(ctx.needs_input_grad[2])
        need_dm_d = bool(ctx.needs_input_grad[3])
        need_dbeta = bool(ctx.needs_input_grad[4])
        need_dk = bool(ctx.needs_input_grad[5])
        need_dlam = bool(ctx.needs_input_grad[6])
        need_dprior = bool(log_prior is not None and ctx.needs_input_grad[8])

        dpsi_s = (
            torch.empty((n_tok, d), device=psi_s_flat.device, dtype=work_dtype)
            if need_dpsi_s
            else None
        )
        dpsi_d = (
            torch.empty((n_tok, r), device=psi_d_flat.device, dtype=work_dtype)
            if need_dpsi_d
            else None
        )
        dm_s = (
            torch.zeros((v_vocab, d), device=m_sharp.device, dtype=work_dtype)
            if need_dm_s
            else None
        )
        dm_d = (
            torch.zeros((v_vocab, r), device=m_diff.device, dtype=work_dtype)
            if need_dm_d
            else None
        )
        dbeta = (
            torch.zeros_like(beta_flat)
            if need_dbeta
            else None
        )
        dk = torch.zeros((), device=psi_s_flat.device, dtype=acc_dtype) if need_dk else None
        dlam = torch.zeros((), device=psi_s_flat.device, dtype=acc_dtype) if need_dlam else None
        dprior = (
            torch.zeros((v_vocab,), device=psi_s_flat.device, dtype=acc_dtype)
            if need_dprior
            else None
        )

        valid_all = (flat_y != ignore_idx) & (flat_y >= 0) & (flat_y < v_active)
        valid_count = valid_all.sum().clamp(min=1).to(acc_dtype)
        inv_n_tok = 1.0 / valid_count
        total_loss = torch.zeros((), device=psi_s_flat.device, dtype=acc_dtype)

        for start in range(0, n_tok, chunk_sz):
            end = min(start + chunk_sz, n_tok)
            psi_s_c = psi_s_flat[start:end].to(work_dtype)
            psi_d_c = psi_d_flat[start:end].to(work_dtype)
            yc = flat_y[start:end]
            yc_valid = valid_all[start:end].unsqueeze(1)
            yc_safe = torch.where(yc_valid, yc.unsqueeze(1), torch.zeros_like(yc.unsqueeze(1)))
            valid_f = yc_valid.to(acc_dtype)
            row_scale = valid_f * inv_n_tok

            c_sharp = F.linear(psi_s_c, m_s_work).to(acc_dtype)
            c_diff = F.linear(psi_d_c, m_d_work).to(acc_dtype)
            beta_c = beta_flat[start:end] if is_per_token_beta else beta_flat
            r_raw = beta_c / k_val
            r_mask = ((r_raw > 0.0) & (r_raw < 0.95)).to(acc_dtype)
            r_tilde = torch.clamp(r_raw, min=0.0, max=0.95)

            z_raw = 1.0 + r_tilde * c_sharp + lam_val * c_diff
            z_mask = (z_raw > 1e-4).to(acc_dtype)
            z = torch.clamp(z_raw, min=1e-4)
            log_z = torch.log(z)
            logits = k_val * log_z
            if log_prior_f is not None:
                logits = logits + log_prior_f

            max_logits = torch.amax(logits[:, :v_active], dim=-1, keepdim=True)
            logits = logits - max_logits
            target_logits = logits.gather(1, yc_safe)
            exp_logits = torch.exp(logits)
            if v_active < v_vocab:
                exp_logits[:, v_active:] = 0.0
            sum_exp = exp_logits.sum(dim=-1, keepdim=True)
            total_loss.add_(((torch.log(sum_exp) - target_logits) * valid_f).sum())

            if any(
                (
                    need_dpsi_s,
                    need_dpsi_d,
                    need_dm_s,
                    need_dm_d,
                    need_dbeta,
                    need_dk,
                    need_dlam,
                    need_dprior,
                )
            ):
                probs = exp_logits / sum_exp
                probs.scatter_add_(
                    1,
                    yc_safe,
                    torch.full_like(yc_safe, -1.0, dtype=acc_dtype),
                )
                if v_active < v_vocab:
                    probs[:, v_active:] = 0.0
                g_c = probs * row_scale  # dL / d_logits_{t,v}

                if need_dprior and dprior is not None:
                    dprior.add_(g_c.sum(dim=0))

                q_c = g_c * (k_val / z) * z_mask  # dL / dz_raw_{t,v}
                dr_tilde = (q_c * c_sharp).sum(dim=-1, keepdim=True)

                if need_dk and dk is not None:
                    dk.add_(
                        (g_c * log_z).sum()
                        - (dr_tilde * (beta_c / (k_val * k_val)) * r_mask).sum()
                    )

                if need_dbeta and dbeta is not None:
                    dbeta_c = dr_tilde * (1.0 / k_val) * r_mask
                    if is_per_token_beta:
                        dbeta[start:end] = dbeta_c
                    else:
                        dbeta.add_(dbeta_c.sum())

                if need_dlam and dlam is not None:
                    dlam.add_((q_c * c_diff).sum())

                if need_dpsi_s or need_dm_s:
                    dc_sharp = (q_c * r_tilde).to(work_dtype)
                    if need_dpsi_s and dpsi_s is not None:
                        torch.mm(dc_sharp, m_s_work, out=dpsi_s[start:end])
                    if need_dm_s and dm_s is not None:
                        dm_s.addmm_(dc_sharp.t(), psi_s_c)

                if need_dpsi_d or need_dm_d:
                    dc_diff = (q_c * lam_val).to(work_dtype)
                    if need_dpsi_d and dpsi_d is not None:
                        torch.mm(dc_diff, m_d_work, out=dpsi_d[start:end])
                    if need_dm_d and dm_d is not None:
                        dm_d.addmm_(dc_diff.t(), psi_d_c)

        ctx.save_for_backward(
            dpsi_s.to(psi_sharp.dtype).view(orig_psi_s_shape)
            if dpsi_s is not None
            else torch.empty(0, device=psi_s_flat.device),
            dpsi_d.to(psi_diff.dtype).view(orig_psi_d_shape)
            if dpsi_d is not None
            else torch.empty(0, device=psi_d_flat.device),
            dm_s.to(m_sharp.dtype)
            if dm_s is not None
            else torch.empty(0, device=m_sharp.device),
            dm_d.to(m_diff.dtype)
            if dm_d is not None
            else torch.empty(0, device=m_diff.device),
            dbeta.view(beta_t.shape).to(beta_t.dtype)
            if dbeta is not None
            else torch.empty(0, device=psi_s_flat.device),
            dk.view(k_copies.shape).to(k_copies.dtype)
            if dk is not None
            else torch.empty(0, device=psi_s_flat.device),
            dlam.view(lambda_diff.shape).to(lambda_diff.dtype)
            if dlam is not None
            else torch.empty(0, device=psi_s_flat.device),
            dprior.view(log_prior.shape).to(log_prior.dtype)
            if (dprior is not None and log_prior is not None)
            else torch.empty(0, device=psi_s_flat.device),
        )
        ctx.needs = (
            need_dpsi_s,
            need_dpsi_d,
            need_dm_s,
            need_dm_d,
            need_dbeta,
            need_dk,
            need_dlam,
            need_dprior,
        )
        return total_loss * inv_n_tok

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        saved = ctx.saved_tensors
        g = grad_out.to(
            torch.float64
            if any(t.dtype == torch.float64 for t in saved if t.numel() > 0)
            else torch.float32
        )
        grads = [
            (t.to(g.dtype) * g).to(t.dtype) if need else None
            for t, need in zip(saved, ctx.needs)
        ]
        return (
            grads[0],
            grads[1],
            grads[2],
            grads[3],
            grads[4],
            grads[5],
            grads[6],
            None,
            grads[7],
            None,
            None,
            None,
        )


def direct_chunked_two_beam_kboson_ce(
    psi_sharp: torch.Tensor,
    psi_diff: torch.Tensor,
    m_sharp: torch.Tensor,
    m_diff: torch.Tensor,
    beta_t: torch.Tensor | float,
    k_copies: torch.Tensor | float,
    lambda_diff: torch.Tensor | float,
    targets: torch.Tensor,
    log_prior: torch.Tensor | None = None,
    chunk_size: int = 8192,
    chunk_tokens: int | None = None,
    ignore_index: int = -1,
    valid_vocab_size: int | None = None,
) -> torch.Tensor:
    """Compute single-pass chunked Two-Beam k-Boson mixed-state POVM cross-entropy with analytical gradients."""
    ref_dtype = torch.float64 if psi_sharp.dtype == torch.float64 else torch.float32
    beta_tensor = (
        beta_t
        if isinstance(beta_t, torch.Tensor)
        else torch.tensor(float(beta_t), device=psi_sharp.device, dtype=ref_dtype)
    )
    k_tensor = (
        k_copies
        if isinstance(k_copies, torch.Tensor)
        else torch.tensor(float(k_copies), device=psi_sharp.device, dtype=ref_dtype)
    )
    lam_tensor = (
        lambda_diff
        if isinstance(lambda_diff, torch.Tensor)
        else torch.tensor(float(lambda_diff), device=psi_sharp.device, dtype=ref_dtype)
    )
    eff_chunk = int(chunk_tokens) if chunk_tokens is not None else int(chunk_size)
    return _DirectChunkedTwoBeamKBosonCEFn.apply(
        psi_sharp,
        psi_diff,
        m_sharp,
        m_diff,
        beta_tensor,
        k_tensor,
        lam_tensor,
        targets,
        log_prior,
        eff_chunk,
        int(ignore_index),
        valid_vocab_size,
    )


def compute_two_beam_kboson_logits(
    psi_sharp: torch.Tensor,
    psi_diff: torch.Tensor,
    m_sharp: torch.Tensor,
    m_diff: torch.Tensor,
    beta_t: torch.Tensor | float,
    k_copies: torch.Tensor | float,
    lambda_diff: torch.Tensor | float,
    log_prior: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute unchunked Two-Beam k-Boson log-probabilities (up to additive normalization constant)."""
    acc_dtype = torch.float64 if psi_sharp.dtype == torch.float64 else torch.float32
    c_sharp = F.linear(psi_sharp.to(acc_dtype), m_sharp.to(acc_dtype))
    c_diff = F.linear(psi_diff.to(acc_dtype), m_diff.to(acc_dtype))
    beta_f = (
        beta_t.to(acc_dtype)
        if isinstance(beta_t, torch.Tensor)
        else torch.tensor(float(beta_t), device=psi_sharp.device, dtype=acc_dtype)
    )
    if beta_f.ndim > 0 and beta_f.ndim < c_sharp.ndim:
        beta_f = beta_f.unsqueeze(-1)
    k_f = (
        k_copies.to(acc_dtype)
        if isinstance(k_copies, torch.Tensor)
        else torch.tensor(float(k_copies), device=psi_sharp.device, dtype=acc_dtype)
    )
    lam_f = (
        lambda_diff.to(acc_dtype)
        if isinstance(lambda_diff, torch.Tensor)
        else torch.tensor(float(lambda_diff), device=psi_sharp.device, dtype=acc_dtype)
    )
    r_tilde = torch.clamp(beta_f / k_f, min=0.0, max=0.95)
    z = torch.clamp(1.0 + r_tilde * c_sharp + lam_f * c_diff, min=1e-4)
    logits = k_f * torch.log(z)
    if log_prior is not None:
        logits = logits + log_prior.to(acc_dtype)
    return logits if psi_sharp.dtype == torch.float64 else logits.float()


__all__ = [
    "_DirectChunkedSoftcapLinearCEFn",
    "direct_chunked_softcap_linear_ce",
    "_DirectChunkedPOVMCEFn",
    "direct_chunked_povm_ce",
    "_DirectChunkedTwoBeamKBosonCEFn",
    "direct_chunked_two_beam_kboson_ce",
    "compute_two_beam_kboson_logits",
    "build_sampled_candidate_set",
    "sampled_softcap_linear_ce",
    "sampled_povm_ce",
]


