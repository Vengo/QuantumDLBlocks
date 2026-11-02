r"""Fused Single-Pass Complex Hermitian/Born $QK$ + Outer-Product Density $\rho_{t,h}^{(0)}$ & Coherent Ket Attention Kernel (`Wave 54–55` / `EXPERIMENT_PLAN_075.md` & `EXPERIMENT_PLAN_076.md`).

When to Use This Building Block:
- Use `QuantumAttnKernelMode`, `fused_quantum_wavepacket_rho_ket_attention`,
  `associative_mcweeny_unitary_ket_modulation`, and `run_coherent_ket_wavepacket_attention` inside
  `SVDStiefelWavepacketStatMechBlock` and `QuantumFullBackboneGraftGPT` whenever aggregating trace-1
  outer-product density matrices
  $\rho_{t,h}^{(0)} = \sum_{s \le t} A_{t,s}^{(h)} \rho_{s,h}^{\text{tok}} \in \mathbb{C}^{d_h \times d_h}$
  and coherent value kets $|\bar{v}_{t,h,r}\rangle = \sum_{s \le t} A_{t,s}^{(h)} |v_{\text{mixed}, s, h, r}\rangle \in \mathbb{C}^{d_h}$:
  1. `QuantumAttnKernelMode.FUSED_SINGLE_PASS_RHO_KET`: Eliminates the $K_\rho + 1 = 17$ ($d_h=32$) or
     $K_\rho + 1 = 65$ ($d_h=64$) sub-head $QK^\top$ expansion required by standard SDPA. Packs the
     Hermitian token density matrix into a single real matrix $M_{s,h}^{\text{tok}} = \rho_{s,h,R}^{\text{tok}} + \rho_{s,h,I}^{\text{tok}} \in \mathbb{R}^{d_h \times d_h}$
     and recovers $\rho_R^{(0)} = \frac{1}{2}(M^{(0)} + M^{(0)\top})$, $\rho_I^{(0)} = \frac{1}{2}(M^{(0)} - M^{(0)\top})$
     after attention in a single Tensor-Core GEMM pass, matching `EXPANDED_SDPA_CHUNKS` to `< 1e-5` error.
  2. `QuantumAttnKernelMode.FUSED_HERMITIAN_BORN_RHO_KET`: Replaces external `apply_berry_qk_phase`
     (zero HBM query-rotation allocation) by computing both the real and imaginary parts
     $(C_{\text{re}}, C_{\text{im}}) = (\operatorname{Re}\langle Q_{t,h}|K_{s,h}\rangle_{\mathbb{C}},\, \operatorname{Im}\langle Q_{t,h}|K_{s,h}\rangle_{\mathbb{C}})$
     and scoring query-key pairs via a `@triton.jit` SRAM kernel combining $U(1)$ Hermitian phase rotation
     and second-order phase-invariant Born transition probability $|\langle Q_{t,h}|K_{s,h}\rangle_{\mathbb{C}}|^2 = C_{\text{re}}^2 + C_{\text{im}}^2$.
  3. `associative_mcweeny_unitary_ket_modulation`: Exploits matrix multiplication associativity to contract
     the $(R \times d_h)$ target ket with $U_h (\bar{\rho}^2 / \operatorname{Tr}(\bar{\rho}^2) - \frac{1}{d_h}I) U_h^\dagger$
     without forming $d_h \times d_h \times d_h$ matrix-matrix products (`2.6x` faster at $d_h=32$, `3.8x` faster at $d_h=64$).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_WAVEPACKET_TRITON = True
except ImportError:
    HAS_WAVEPACKET_TRITON = False

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_075.md & EXPERIMENT_PLAN_076.md
# ==============================================================================
TRITON_HERMITIAN_BLOCK_N: int = 64
TRITON_MAX_FUSED_SEQ_LEN: int = 2048
DEFAULT_QK_BORN_BETA_INIT: float = 0.0
DEFAULT_QK_BORN_BETA_MAX: float = 1.50
DEFAULT_LOW_RANK_CHOLESKY_K: int = 4
MASK_NEG_INF_FILL: float = -1.0e9

_COMPILED_ASSOC_KET_FN: Any = None
_COMPILED_CHOL_GRAM_FN: Any = None


class QuantumAttnKernelMode(str, Enum):
    """Axis W strategy enum selecting expanded SDPA chunks vs. single-pass fused outer-product kernels (Rule 3)."""

    EXPANDED_SDPA_CHUNKS = "expanded_sdpa_chunks"
    FUSED_SINGLE_PASS_RHO_KET = "fused_single_pass_rho_ket"
    FUSED_HERMITIAN_BORN_RHO_KET = "fused_hermitian_born_rho_ket"
    FUSED_HERMITIAN_BORN_LOW_RANK_CHOLESKY = "fused_hermitian_born_low_rank_cholesky"


if HAS_WAVEPACKET_TRITON:

    @triton.jit
    def _fused_hermitian_born_doc_swc_fwd_kernel(
        c_re_ptr,
        c_im_ptr,
        mask_ptr,
        cos_phi_ptr,
        sin_phi_ptr,
        born_scale_ptr,
        a_out_ptr,
        seq_t: int,
        num_heads: int,
        sdpa_scale: float,
        HAS_IMAG: tl.constexpr,
        HAS_MASK: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        t_idx = row_idx % seq_t
        bh_idx = row_idx // seq_t
        h_idx = bh_idx % num_heads
        b_idx = bh_idx // num_heads

        offs_s = tl.arange(0, BLOCK_T)
        valid_s = offs_s < seq_t
        row_base = row_idx * seq_t + offs_s

        if HAS_MASK:
            mask_base = (b_idx * seq_t + t_idx) * seq_t + offs_s
            allow = tl.load(mask_ptr + mask_base, mask=valid_s, other=0).to(tl.int1)
        else:
            allow = offs_s <= t_idx
        active_mask = valid_s & allow

        cr = tl.load(c_re_ptr + row_base, mask=active_mask, other=0.0).to(tl.float32)
        if HAS_IMAG:
            ci = tl.load(c_im_ptr + row_base, mask=active_mask, other=0.0).to(tl.float32)
            c_p = tl.load(cos_phi_ptr + h_idx).to(tl.float32)
            s_p = tl.load(sin_phi_ptr + h_idx).to(tl.float32)
            b_sc = tl.load(born_scale_ptr + h_idx).to(tl.float32)
            mag_sq = cr * cr + ci * ci
            raw_score = sdpa_scale * (c_p * cr + s_p * ci + b_sc * mag_sq)
        else:
            raw_score = sdpa_scale * cr

        masked_score = tl.where(active_mask, raw_score, -1.0e20)
        max_s = tl.max(masked_score, axis=0)
        exp_s = tl.where(active_mask, tl.exp(masked_score - max_s), 0.0)
        denom = tl.sum(exp_s, axis=0) + 1.0e-12
        a_val = exp_s / denom
        tl.store(a_out_ptr + row_base, a_val.to(a_out_ptr.dtype.element_ty), mask=valid_s)

    @triton.jit
    def _fused_hermitian_born_doc_swc_bwd_kernel(
        c_re_ptr,
        c_im_ptr,
        a_out_ptr,
        grad_a_ptr,
        cos_phi_ptr,
        sin_phi_ptr,
        born_scale_ptr,
        grad_c_re_ptr,
        grad_c_im_ptr,
        grad_cos_sin_born_ptr,
        seq_t: int,
        num_heads: int,
        sdpa_scale: float,
        HAS_IMAG: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        bh_idx = row_idx // seq_t
        h_idx = bh_idx % num_heads
        offs_s = tl.arange(0, BLOCK_T)
        valid_s = offs_s < seq_t
        row_base = row_idx * seq_t + offs_s

        a_val = tl.load(a_out_ptr + row_base, mask=valid_s, other=0.0).to(tl.float32)
        ga = tl.load(grad_a_ptr + row_base, mask=valid_s, other=0.0).to(tl.float32)
        dot_ag = tl.sum(a_val * ga, axis=0)
        ds = a_val * (ga - dot_ag) * sdpa_scale

        if HAS_IMAG:
            cr = tl.load(c_re_ptr + row_base, mask=valid_s, other=0.0).to(tl.float32)
            ci = tl.load(c_im_ptr + row_base, mask=valid_s, other=0.0).to(tl.float32)
            c_p = tl.load(cos_phi_ptr + h_idx).to(tl.float32)
            s_p = tl.load(sin_phi_ptr + h_idx).to(tl.float32)
            b_sc = tl.load(born_scale_ptr + h_idx).to(tl.float32)

            two_b = 2.0 * b_sc
            g_cr = ds * (c_p + two_b * cr)
            g_ci = ds * (s_p + two_b * ci)
            tl.store(grad_c_re_ptr + row_base, g_cr.to(grad_c_re_ptr.dtype.element_ty), mask=valid_s)
            tl.store(grad_c_im_ptr + row_base, g_ci.to(grad_c_im_ptr.dtype.element_ty), mask=valid_s)

            d_cos = tl.sum(ds * cr, axis=0)
            d_sin = tl.sum(ds * ci, axis=0)
            d_born = tl.sum(ds * (cr * cr + ci * ci), axis=0)
            out_base = row_idx * 3
            tl.store(grad_cos_sin_born_ptr + out_base + 0, d_cos)
            tl.store(grad_cos_sin_born_ptr + out_base + 1, d_sin)
            tl.store(grad_cos_sin_born_ptr + out_base + 2, d_born)
        else:
            tl.store(grad_c_re_ptr + row_base, ds.to(grad_c_re_ptr.dtype.element_ty), mask=valid_s)


def _compute_complex_hermitian_qk_grams(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    v_rank: int,
    d_h: int,
    need_imag: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute `c_re` (and `c_im` when `need_imag=True`) of shape `(B, H, T, T)` via Tensor-Core GEMMs (`<= 8` args)."""
    bsz, seq_len, num_heads, head_dim = q_rot.shape
    work_dtype = q_rot.dtype
    q_bh = q_rot.permute(0, 2, 1, 3).contiguous()
    k_bh_t = k_rot.to(work_dtype).permute(0, 2, 3, 1).contiguous()
    if not need_imag:
        return torch.matmul(q_bh, k_bh_t).to(work_dtype), None

    qc = q_bh.view(bsz, num_heads, seq_len, v_rank, 2, d_h)
    q_conj = torch.stack([-qc[..., 1, :], qc[..., 0, :]], dim=-2).reshape(bsz, num_heads, seq_len, head_dim)
    q_cat = torch.cat([q_bh, q_conj], dim=2)
    c_both = torch.matmul(q_cat, k_bh_t).to(work_dtype)
    return c_both[:, :, :seq_len, :].contiguous(), c_both[:, :, seq_len:, :].contiguous()


def _Backward_hermitian_qk_grams(
    grad_c_re: torch.Tensor,
    grad_c_im: torch.Tensor | None,
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    v_rank: int,
    d_h: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass backward for `_compute_complex_hermitian_qk_grams` returning `(grad_q_rot, grad_k_rot)` (`<= 8` args)."""
    bsz, seq_len, num_heads, head_dim = q_rot.shape
    dt = grad_c_re.dtype
    q_bh = q_rot.to(dt).permute(0, 2, 1, 3)
    k_bh = k_rot.to(dt).permute(0, 2, 1, 3)
    if grad_c_im is None:
        gq_bh = torch.matmul(grad_c_re, k_bh)
        gk_bh = torch.matmul(grad_c_re.transpose(-2, -1), q_bh)
        return (
            gq_bh.permute(0, 2, 1, 3).to(q_rot.dtype).contiguous(),
            gk_bh.permute(0, 2, 1, 3).to(k_rot.dtype).contiguous(),
        )

    gc_im = grad_c_im.to(dt)
    gq_re_part = torch.matmul(grad_c_re, k_bh)
    gq_im_part = torch.matmul(gc_im, k_bh)
    gqc_im = gq_im_part.view(bsz, num_heads, seq_len, v_rank, 2, d_h)
    gq_from_im = torch.stack([gqc_im[..., 1, :], -gqc_im[..., 0, :]], dim=-2).reshape(bsz, num_heads, seq_len, head_dim)
    gq_bh = gq_re_part + gq_from_im

    qc = q_bh.view(bsz, num_heads, seq_len, v_rank, 2, d_h)
    q_conj = torch.stack([-qc[..., 1, :], qc[..., 0, :]], dim=-2).reshape(bsz, num_heads, seq_len, head_dim)
    gk_bh = torch.matmul(grad_c_re.transpose(-2, -1), q_bh) + torch.matmul(gc_im.transpose(-2, -1), q_conj)
    return (
        gq_bh.permute(0, 2, 1, 3).to(q_rot.dtype).contiguous(),
        gk_bh.permute(0, 2, 1, 3).to(k_rot.dtype).contiguous(),
    )


class _FusedHermitianBornRhoKetAttnFn(torch.autograd.Function):
    """Single-pass autograd function for Fused Complex Hermitian/Born $QK$ + Packed $(M, v)$ Attention (Rule 9)."""

    @staticmethod
    def forward(
        ctx: Any,
        q_rot: torch.Tensor,
        k_rot: torch.Tensor,
        payload_packed: torch.Tensor,
        attn_mask: torch.Tensor | None,
        berry_phase_hrd: torch.Tensor | None,
        qk_born_beta_h: torch.Tensor | None,
        scale_and_hb: tuple[float, bool],
    ) -> torch.Tensor:
        sdpa_scale, use_hermitian_born = scale_and_hb
        bsz, seq_len, num_heads, head_dim = q_rot.shape
        work_dtype = q_rot.dtype
        v_rank = berry_phase_hrd.size(1) if berry_phase_hrd is not None else 1
        d_h = berry_phase_hrd.size(2) if berry_phase_hrd is not None else (head_dim // 2)

        c_re, c_im = _compute_complex_hermitian_qk_grams(q_rot, k_rot, v_rank, d_h, need_imag=use_hermitian_born)

        if use_hermitian_born and berry_phase_hrd is not None and qk_born_beta_h is not None:
            phi_h = berry_phase_hrd.float().mean(dim=(-2, -1))
            cos_phi = torch.cos(phi_h)
            sin_phi = torch.sin(phi_h)
            inv_d = 1.0 / float(head_dim)
            born_scale = torch.tanh(qk_born_beta_h.float()) * inv_d
        else:
            cos_phi = torch.ones((num_heads,), device=q_rot.device, dtype=torch.float32)
            sin_phi = torch.zeros((num_heads,), device=q_rot.device, dtype=torch.float32)
            born_scale = torch.zeros((num_heads,), device=q_rot.device, dtype=torch.float32)

        use_triton = (
            HAS_WAVEPACKET_TRITON
            and q_rot.is_cuda
            and seq_len <= TRITON_MAX_FUSED_SEQ_LEN
        )
        if use_triton:
            n_rows = bsz * num_heads * seq_len
            block_t = max(16, triton.next_power_of_2(seq_len))
            a_weights = torch.empty((bsz, num_heads, seq_len, seq_len), device=q_rot.device, dtype=work_dtype)
            mask_c = attn_mask.contiguous() if attn_mask is not None else c_re
            c_im_c = c_im if c_im is not None else c_re
            _fused_hermitian_born_doc_swc_fwd_kernel[(n_rows,)](
                c_re,
                c_im_c,
                mask_c,
                cos_phi,
                sin_phi,
                born_scale,
                a_weights,
                seq_len,
                num_heads,
                float(sdpa_scale),
                HAS_IMAG=bool(use_hermitian_born),
                HAS_MASK=bool(attn_mask is not None),
                BLOCK_T=block_t,
            )
        else:
            cr_f = c_re.float()
            if use_hermitian_born and c_im is not None:
                ci_f = c_im.float()
                cp_v = cos_phi.view(1, num_heads, 1, 1)
                sp_v = sin_phi.view(1, num_heads, 1, 1)
                bs_v = born_scale.view(1, num_heads, 1, 1)
                scores = float(sdpa_scale) * (cp_v * cr_f + sp_v * ci_f + bs_v * (cr_f.square() + ci_f.square()))
            else:
                scores = float(sdpa_scale) * cr_f

            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, MASK_NEG_INF_FILL)
            else:
                q_idx = torch.arange(seq_len, device=q_rot.device).unsqueeze(1)
                k_idx = torch.arange(seq_len, device=q_rot.device).unsqueeze(0)
                scores = scores.masked_fill(q_idx < k_idx, MASK_NEG_INF_FILL)
            a_weights = F.softmax(scores, dim=-1).to(work_dtype)

        pay_w = payload_packed.to(work_dtype)
        out_payload = torch.matmul(a_weights, pay_w).to(work_dtype)
        ctx.save_for_backward(
            q_rot,
            k_rot,
            pay_w,
            a_weights,
            c_re,
            c_im if c_im is not None else torch.empty(0, device=q_rot.device),
            cos_phi,
            sin_phi,
            born_scale,
            berry_phase_hrd if berry_phase_hrd is not None else torch.empty(0, device=q_rot.device),
            qk_born_beta_h if qk_born_beta_h is not None else torch.empty(0, device=q_rot.device),
        )
        ctx.sdpa_scale = float(sdpa_scale)
        ctx.use_hermitian_born = bool(use_hermitian_born)
        ctx.use_triton = bool(use_triton)
        ctx.v_rank = int(v_rank)
        ctx.d_h = int(d_h)
        ctx.payload_orig_dtype = payload_packed.dtype
        return out_payload

    @staticmethod
    def backward(ctx: Any, grad_out_payload: torch.Tensor) -> tuple[Any, ...]:
        (
            q_rot,
            k_rot,
            payload_packed,
            a_weights,
            c_re,
            c_im_saved,
            cos_phi,
            sin_phi,
            born_scale,
            berry_saved,
            born_beta_saved,
        ) = ctx.saved_tensors
        bsz, seq_len, num_heads, head_dim = q_rot.shape
        work_dtype = c_re.dtype
        use_hb = ctx.use_hermitian_born
        c_im = c_im_saved.to(work_dtype) if use_hb else None

        grad_out_c = grad_out_payload.to(work_dtype).contiguous()
        a_w = a_weights.to(work_dtype)
        pay_w = payload_packed.to(work_dtype)
        grad_payload = torch.matmul(a_w.transpose(-2, -1), grad_out_c).to(ctx.payload_orig_dtype)
        grad_a = torch.matmul(grad_out_c, pay_w.transpose(-2, -1))

        grad_berry: torch.Tensor | None = None
        grad_born_beta: torch.Tensor | None = None

        if ctx.use_triton:
            n_rows = bsz * num_heads * seq_len
            block_t = max(16, triton.next_power_of_2(seq_len))
            grad_c_re = torch.empty_like(c_re)
            grad_c_im = torch.empty_like(c_im) if (use_hb and c_im is not None) else grad_c_re
            grad_csb = (
                torch.empty((bsz, num_heads, seq_len, 3), device=q_rot.device, dtype=torch.float32)
                if use_hb
                else torch.empty((1,), device=q_rot.device, dtype=torch.float32)
            )
            _fused_hermitian_born_doc_swc_bwd_kernel[(n_rows,)](
                c_re,
                c_im if (use_hb and c_im is not None) else c_re,
                a_w,
                grad_a,
                cos_phi,
                sin_phi,
                born_scale,
                grad_c_re,
                grad_c_im,
                grad_csb,
                seq_len,
                num_heads,
                ctx.sdpa_scale,
                HAS_IMAG=bool(use_hb),
                BLOCK_T=block_t,
            )
            if use_hb:
                d_cos_h = grad_csb[..., 0].sum(dim=(0, 2))
                d_sin_h = grad_csb[..., 1].sum(dim=(0, 2))
                d_born_h = grad_csb[..., 2].sum(dim=(0, 2))
            else:
                d_cos_h = d_sin_h = d_born_h = None
        else:
            af = a_w.float()
            gaf = grad_a.float()
            dot_ag = (af * gaf).sum(dim=-1, keepdim=True)
            ds = af * (gaf - dot_ag) * ctx.sdpa_scale
            if use_hb and c_im is not None:
                crf = c_re.float()
                cif = c_im.float()
                cp_v = cos_phi.view(1, num_heads, 1, 1)
                sp_v = sin_phi.view(1, num_heads, 1, 1)
                two_b = 2.0 * born_scale.view(1, num_heads, 1, 1)
                grad_c_re = (ds * (cp_v + two_b * crf)).to(work_dtype)
                grad_c_im = (ds * (sp_v + two_b * cif)).to(work_dtype)
                d_cos_h = (ds * crf).sum(dim=(0, 2, 3))
                d_sin_h = (ds * cif).sum(dim=(0, 2, 3))
                d_born_h = (ds * (crf.square() + cif.square())).sum(dim=(0, 2, 3))
            else:
                grad_c_re = ds.to(work_dtype)
                grad_c_im = None
                d_cos_h = d_sin_h = d_born_h = None

        if use_hb and d_cos_h is not None and d_sin_h is not None and d_born_h is not None:
            if berry_saved.numel() > 0:
                d_phi_h = -sin_phi * d_cos_h + cos_phi * d_sin_h
                n_rd = float(berry_saved.size(1) * berry_saved.size(2))
                grad_berry = (
                    (d_phi_h / n_rd)
                    .view(num_heads, 1, 1)
                    .expand_as(berry_saved)
                    .to(berry_saved.dtype)
                )
            if born_beta_saved.numel() > 0:
                inv_d = 1.0 / float(head_dim)
                sech_sq = 1.0 - torch.tanh(born_beta_saved.float()).square()
                grad_born_beta = (d_born_h * inv_d * sech_sq).to(born_beta_saved.dtype)

        grad_q, grad_k = _Backward_hermitian_qk_grams(
            grad_c_re, grad_c_im if use_hb else None, q_rot, k_rot, ctx.v_rank, ctx.d_h
        )
        return grad_q, grad_k, grad_payload, None, grad_berry, grad_born_beta, None


def fused_quantum_wavepacket_rho_ket_attention(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    tok_rho_re: torch.Tensor,
    tok_rho_im: torch.Tensor,
    v_mixed: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    phase_and_born: tuple[torch.Tensor | None, torch.Tensor | None, float, bool] = (None, None, 0.12, False),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Single-pass fused $(\rho_{t,h}^{(0)}, |\bar{v}_{t,h}\rangle)$ attention using Zero-Mask Hermitian Sum Packing $M = \rho_R + \rho_I$ (`<= 8` args)."""
    bsz, num_heads, seq_len, d_h, _ = tok_rho_re.shape
    head_dim = q_rot.size(-1)
    v_rank = max(1, head_dim // (2 * d_h))
    berry_phase, born_beta, sdpa_scale, use_hermitian_born = phase_and_born

    d_sq = d_h * d_h
    tok_m_flat = (tok_rho_re + tok_rho_im).reshape(bsz, num_heads, seq_len, d_sq)
    v_ket_flat = v_mixed.view(bsz, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    payload_packed = torch.cat([tok_m_flat, v_ket_flat], dim=-1).contiguous()

    out_payload = _FusedHermitianBornRhoKetAttnFn.apply(
        q_rot,
        k_rot,
        payload_packed,
        attn_mask,
        berry_phase,
        born_beta,
        (float(sdpa_scale), bool(use_hermitian_born)),
    )
    m0 = out_payload[..., :d_sq].reshape(bsz, num_heads, seq_len, d_h, d_h)
    m0_t = m0.transpose(-2, -1)
    rho0_re = 0.5 * (m0 + m0_t)
    rho0_im = 0.5 * (m0 - m0_t)
    a_v = out_payload[..., d_sq:].view(bsz, num_heads, seq_len, v_rank, 2, d_h)
    return rho0_re, rho0_im, a_v[..., 0, :], a_v[..., 1, :]


def associative_mcweeny_unitary_ket_modulation(
    rho0_re: torch.Tensor,
    rho0_im: torch.Tensor,
    v_bar_re: torch.Tensor,
    v_bar_im: torch.Tensor,
    v_mixed: torch.Tensor,
    u_re_im: tuple[torch.Tensor, torch.Tensor],
    scales_and_mode: tuple[torch.Tensor, torch.Tensor, bool],
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Associative $(R \times d_h)$ ket-operator contraction for Quadratic McWeeny + Unitary evolution (`<= 8` args)."""
    from ..layers.impure_density_mirror_block import TRACE_NORM_EPS

    bsz, num_heads, seq_len, d_h, _ = rho0_re.shape
    v_rank = v_bar_re.size(-2)
    sdpa_dim = 2 * v_rank * d_h
    work_dtype = v_bar_re.dtype
    head_u_re, head_u_im = u_re_im
    rho_coupling_scale, obs_scale, use_local_v_target = scales_and_mode

    u0_re = head_u_re.to(work_dtype).view(1, num_heads, 1, d_h, d_h)
    u0_im = head_u_im.to(work_dtype).view(1, num_heads, 1, d_h, d_h)
    u0_re_t = u0_re.transpose(-2, -1)
    u0_im_t = u0_im.transpose(-2, -1)

    vu_re = torch.matmul(v_bar_re, u0_re_t) - torch.matmul(v_bar_im, u0_im_t)
    vu_im = torch.matmul(v_bar_re, u0_im_t) + torch.matmul(v_bar_im, u0_re_t)

    if use_local_v_target:
        v_loc_c = v_mixed.view(bsz, seq_len, num_heads, v_rank, 2, d_h).permute(0, 2, 1, 3, 4, 5)
        vl_re, vl_im = v_loc_c[..., 0, :], v_loc_c[..., 1, :]
        lu_raw_re = torch.matmul(vl_re, u0_re_t) - torch.matmul(vl_im, u0_im_t)
        lu_raw_im = torch.matmul(vl_re, u0_im_t) + torch.matmul(vl_im, u0_re_t)
        att_fro = (
            (
                vu_re.float().square().sum(dim=(-2, -1), keepdim=True)
                + vu_im.float().square().sum(dim=(-2, -1), keepdim=True)
            )
            .clamp_min(TRACE_NORM_EPS)
            .sqrt()
            .to(work_dtype)
        )
        loc_fro = (
            (
                lu_raw_re.float().square().sum(dim=(-2, -1), keepdim=True)
                + lu_raw_im.float().square().sum(dim=(-2, -1), keepdim=True)
            )
            .clamp_min(TRACE_NORM_EPS)
            .sqrt()
            .to(work_dtype)
        )
        env_ratio = att_fro / loc_fro
        tar_re, tar_im = lu_raw_re * env_ratio, lu_raw_im * env_ratio
    else:
        tar_re, tar_im = vu_re, vu_im

    tr_cur = torch.diagonal(rho0_re, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(TRACE_NORM_EPS)
    cur_re = rho0_re / tr_cur
    cur_im = rho0_im / tr_cur
    tr_sq = (
        (
            cur_re.float().square().sum(dim=(-2, -1), keepdim=True)
            + cur_im.float().square().sum(dim=(-2, -1), keepdim=True)
        )
        .clamp_min(TRACE_NORM_EPS)
        .to(work_dtype)
    )

    w0_re = torch.matmul(tar_re, u0_re) + torch.matmul(tar_im, u0_im)
    w0_im = torch.matmul(tar_im, u0_re) - torch.matmul(tar_re, u0_im)
    w1_re = torch.matmul(w0_re, cur_re) + torch.matmul(w0_im, cur_im)
    w1_im = torch.matmul(w0_im, cur_re) - torch.matmul(w0_re, cur_im)
    w2_re = torch.matmul(w1_re, cur_re) + torch.matmul(w1_im, cur_im)
    w2_im = torch.matmul(w1_im, cur_re) - torch.matmul(w1_re, cur_im)

    inv_tr_sq = 1.0 / tr_sq
    inv_d = 1.0 / float(d_h)
    w3_re = w2_re * inv_tr_sq - inv_d * w0_re
    w3_im = w2_im * inv_tr_sq - inv_d * w0_im
    rho_v_re = torch.matmul(w3_re, u0_re_t) - torch.matmul(w3_im, u0_im_t)
    rho_v_im = torch.matmul(w3_im, u0_re_t) + torch.matmul(w3_re, u0_im_t)

    beta_rho = rho_coupling_scale.to(work_dtype)
    wp_u_re = vu_re + beta_rho * rho_v_re
    wp_u_im = vu_im + beta_rho * rho_v_im
    y_heads = (
        torch.stack([wp_u_re, wp_u_im], dim=-2)
        .permute(0, 2, 1, 3, 4, 5)
        .reshape(bsz, seq_len, num_heads, sdpa_dim)
    )
    return y_heads * obs_scale.to(work_dtype), tr_sq.detach().float().clamp(TRACE_NORM_EPS, 1.0)


def _get_compiled_associative_ket_fn() -> Any:
    """Return lazily compiled `associative_mcweeny_unitary_ket_modulation`."""
    global _COMPILED_ASSOC_KET_FN
    if _COMPILED_ASSOC_KET_FN is None:
        _COMPILED_ASSOC_KET_FN = torch.compile(associative_mcweeny_unitary_ket_modulation, dynamic=False)
    return _COMPILED_ASSOC_KET_FN


def fused_low_rank_cholesky_ket_attention(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    r_re_im: tuple[torch.Tensor, torch.Tensor],
    v_mixed: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    phase_and_born: tuple[torch.Tensor | None, torch.Tensor | None, float, bool] = (None, None, 0.12, True),
    k_chol: int = DEFAULT_LOW_RANK_CHOLESKY_K,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Single-pass fused low-rank Cholesky factor $L_{t,h} \in \mathbb{C}^{k_{\text{chol}} \times d_h}$ + ket attention (`<= 8` args)."""
    from ..layers.impure_density_mirror_block import TRACE_NORM_EPS

    r_re, r_im = r_re_im
    bsz, num_heads, seq_len, v_rank, d_h = r_re.shape
    head_dim = q_rot.size(-1)
    berry_phase, born_beta, sdpa_scale, use_hermitian_born = phase_and_born

    v_c = v_mixed.view(bsz, seq_len, num_heads, v_rank, 2, d_h).permute(0, 2, 1, 3, 4, 5)
    k_c = k_rot.view(bsz, seq_len, num_heads, v_rank, 2, d_h).permute(0, 2, 1, 3, 4, 5)
    q_c = q_rot.view(bsz, seq_len, num_heads, v_rank, 2, d_h).permute(0, 2, 1, 3, 4, 5)
    f_raw_re = torch.cat([r_re, v_c[..., 0, :], k_c[..., 0, :], q_c[..., 0, :]], dim=-2)[..., :k_chol, :]
    f_raw_im = torch.cat([r_im, v_c[..., 1, :], k_c[..., 1, :], q_c[..., 1, :]], dim=-2)[..., :k_chol, :]
    f_nsq = (
        (
            f_raw_re.float().square().sum(dim=(-2, -1), keepdim=True)
            + f_raw_im.float().square().sum(dim=(-2, -1), keepdim=True)
        )
        .clamp_min(TRACE_NORM_EPS)
        .rsqrt()
        .to(r_re.dtype)
    )
    f_re = f_raw_re * f_nsq
    f_im = f_raw_im * f_nsq

    chol_width = 2 * k_chol * d_h
    f_flat = torch.stack([f_re, f_im], dim=-2).reshape(bsz, num_heads, seq_len, chol_width)
    v_ket_flat = v_mixed.view(bsz, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    payload_packed = torch.cat([f_flat, v_ket_flat], dim=-1).contiguous()

    out_payload = _FusedHermitianBornRhoKetAttnFn.apply(
        q_rot,
        k_rot,
        payload_packed,
        attn_mask,
        berry_phase,
        born_beta,
        (float(sdpa_scale), bool(use_hermitian_born)),
    )
    l_packed = out_payload[..., :chol_width].view(bsz, num_heads, seq_len, k_chol, 2, d_h)
    a_v = out_payload[..., chol_width:].view(bsz, num_heads, seq_len, v_rank, 2, d_h)
    return l_packed[..., 0, :], l_packed[..., 1, :], a_v[..., 0, :], a_v[..., 1, :]


def associative_low_rank_cholesky_gram_modulation(
    l_re: torch.Tensor,
    l_im: torch.Tensor,
    v_bar_re: torch.Tensor,
    v_bar_im: torch.Tensor,
    v_mixed: torch.Tensor,
    u_re_im: tuple[torch.Tensor, torch.Tensor],
    scales_and_mode: tuple[torch.Tensor, torch.Tensor, bool],
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Inversion-free $k_{\text{chol}} \times k_{\text{chol}}$ Gram McWeeny purification + Unitary ket modulation (`<= 8` args)."""
    from ..layers.impure_density_mirror_block import TRACE_NORM_EPS

    bsz, num_heads, seq_len, _, d_h = l_re.shape
    v_rank = v_bar_re.size(-2)
    sdpa_dim = 2 * v_rank * d_h
    work_dtype = v_bar_re.dtype
    head_u_re, head_u_im = u_re_im
    rho_coupling_scale, obs_scale, use_local_v_target = scales_and_mode

    u0_re = head_u_re.to(work_dtype).view(1, num_heads, 1, d_h, d_h)
    u0_im = head_u_im.to(work_dtype).view(1, num_heads, 1, d_h, d_h)
    u0_re_t = u0_re.transpose(-2, -1)
    u0_im_t = u0_im.transpose(-2, -1)

    vu_re = torch.matmul(v_bar_re, u0_re_t) - torch.matmul(v_bar_im, u0_im_t)
    vu_im = torch.matmul(v_bar_re, u0_im_t) + torch.matmul(v_bar_im, u0_re_t)

    if use_local_v_target:
        v_loc_c = v_mixed.view(bsz, seq_len, num_heads, v_rank, 2, d_h).permute(0, 2, 1, 3, 4, 5)
        vl_re, vl_im = v_loc_c[..., 0, :], v_loc_c[..., 1, :]
        lu_raw_re = torch.matmul(vl_re, u0_re_t) - torch.matmul(vl_im, u0_im_t)
        lu_raw_im = torch.matmul(vl_re, u0_im_t) + torch.matmul(vl_im, u0_re_t)
        att_fro = (
            (
                vu_re.float().square().sum(dim=(-2, -1), keepdim=True)
                + vu_im.float().square().sum(dim=(-2, -1), keepdim=True)
            )
            .clamp_min(TRACE_NORM_EPS)
            .sqrt()
            .to(work_dtype)
        )
        loc_fro = (
            (
                lu_raw_re.float().square().sum(dim=(-2, -1), keepdim=True)
                + lu_raw_im.float().square().sum(dim=(-2, -1), keepdim=True)
            )
            .clamp_min(TRACE_NORM_EPS)
            .sqrt()
            .to(work_dtype)
        )
        env_ratio = att_fro / loc_fro
        tar_re, tar_im = lu_raw_re * env_ratio, lu_raw_im * env_ratio
    else:
        tar_re, tar_im = vu_re, vu_im

    l_nsq = (
        (
            l_re.float().square().sum(dim=(-2, -1), keepdim=True)
            + l_im.float().square().sum(dim=(-2, -1), keepdim=True)
        )
        .clamp_min(TRACE_NORM_EPS)
        .rsqrt()
        .to(work_dtype)
    )
    lh_re = l_re * l_nsq
    lh_im = l_im * l_nsq
    lh_re_t = lh_re.transpose(-2, -1)
    lh_im_t = lh_im.transpose(-2, -1)

    g_re = torch.matmul(lh_re, lh_re_t) + torch.matmul(lh_im, lh_im_t)
    g_im = torch.matmul(lh_im, lh_re_t) - torch.matmul(lh_re, lh_im_t)
    tr_sq = (
        (
            g_re.float().square().sum(dim=(-2, -1), keepdim=True)
            + g_im.float().square().sum(dim=(-2, -1), keepdim=True)
        )
        .clamp_min(TRACE_NORM_EPS)
        .to(work_dtype)
    )

    w0_re = torch.matmul(tar_re, u0_re) + torch.matmul(tar_im, u0_im)
    w0_im = torch.matmul(tar_im, u0_re) - torch.matmul(tar_re, u0_im)
    c1_re = torch.matmul(w0_re, lh_re_t) + torch.matmul(w0_im, lh_im_t)
    c1_im = torch.matmul(w0_im, lh_re_t) - torch.matmul(w0_re, lh_im_t)
    c2_re = torch.matmul(c1_re, g_re) - torch.matmul(c1_im, g_im)
    c2_im = torch.matmul(c1_im, g_re) + torch.matmul(c1_re, g_im)
    w2_re = torch.matmul(c2_re, lh_re) - torch.matmul(c2_im, lh_im)
    w2_im = torch.matmul(c2_im, lh_re) + torch.matmul(c2_re, lh_im)

    inv_tr_sq = 1.0 / tr_sq
    inv_d = 1.0 / float(d_h)
    w3_re = w2_re * inv_tr_sq - inv_d * w0_re
    w3_im = w2_im * inv_tr_sq - inv_d * w0_im
    rho_v_re = torch.matmul(w3_re, u0_re_t) - torch.matmul(w3_im, u0_im_t)
    rho_v_im = torch.matmul(w3_im, u0_re_t) + torch.matmul(w3_re, u0_im_t)

    beta_rho = rho_coupling_scale.to(work_dtype)
    wp_u_re = vu_re + beta_rho * rho_v_re
    wp_u_im = vu_im + beta_rho * rho_v_im
    y_heads = (
        torch.stack([wp_u_re, wp_u_im], dim=-2)
        .permute(0, 2, 1, 3, 4, 5)
        .reshape(bsz, seq_len, num_heads, sdpa_dim)
    )
    return y_heads * obs_scale.to(work_dtype), tr_sq.detach().float().clamp(TRACE_NORM_EPS, 1.0)


def _get_compiled_chol_gram_fn() -> Any:
    """Return lazily compiled `associative_low_rank_cholesky_gram_modulation`."""
    global _COMPILED_CHOL_GRAM_FN
    if _COMPILED_CHOL_GRAM_FN is None:
        _COMPILED_CHOL_GRAM_FN = torch.compile(associative_low_rank_cholesky_gram_modulation, dynamic=False)
    return _COMPILED_CHOL_GRAM_FN


def run_coherent_ket_wavepacket_attention(
    svd_blk: Any,
    v_wp: torch.Tensor,
    v_mixed: torch.Tensor,
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Execute trace-1 outer-product density evolution + coherent $V$-ket attention via `svd_blk.attn_kernel_mode` (`<= 8` args)."""
    from ..layers.impure_density_mirror_block import TRACE_NORM_EPS
    from ..layers.quantum_frontend_graft import QuantumAttnInteriorMode
    from ..layers.quantum_statmech_channels import PurificationMode, QuantumChannelMode
    from ..layers.quantum_svd_stiefel_graft import compute_liouville_commutator

    core = svd_blk.core_block
    bsz, seq_len, _, _ = v_wp.shape
    work_dtype = v_wp.dtype
    num_heads = svd_blk.num_heads
    v_rank = svd_blk.v_rank
    d_h = svd_blk.d_h
    sdpa_dim = svd_blk.head_dim

    h_pkt = F.rms_norm(v_wp, (d_h,), eps=TRACE_NORM_EPS)
    rho_src = h_pkt if core.attn_interior_mode == QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V else v_mixed
    r_re, r_im = core._normalize_mode_kets(rho_src)

    kernel_mode = getattr(svd_blk, "attn_kernel_mode", QuantumAttnKernelMode.EXPANDED_SDPA_CHUNKS.value)
    use_low_rank_chol = kernel_mode in (
        QuantumAttnKernelMode.FUSED_HERMITIAN_BORN_LOW_RANK_CHOLESKY,
        QuantumAttnKernelMode.FUSED_HERMITIAN_BORN_LOW_RANK_CHOLESKY.value,
    )
    if use_low_rank_chol:
        l_re, l_im, v_bar_re, v_bar_im = fused_low_rank_cholesky_ket_attention(
            q_rot,
            k_rot,
            (r_re, r_im),
            v_mixed,
            attn_mask=attn_mask,
            phase_and_born=(svd_blk.berry_qk_phase, svd_blk.qk_born_beta, svd_blk.sdpa_scale, True),
        )
        can_use_chol_fast = (
            core.purification_mode == PurificationMode.QUADRATIC_MCWEENY
            and core.channel_mode != QuantumChannelMode.CPTP_KRAUS_RANK_K
            and core.attn_interior_mode in (
                QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
                QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
            )
        )
        if can_use_chol_fast:
            use_loc_v = core.attn_interior_mode == QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V
            chol_fn = _get_compiled_chol_gram_fn() if v_wp.is_cuda else associative_low_rank_cholesky_gram_modulation
            y_out, tr_sq_f = chol_fn(
                l_re,
                l_im,
                v_bar_re,
                v_bar_im,
                v_mixed,
                (core.head_unitary_re, core.head_unitary_im),
                (svd_blk.rho_coupling_scale, svd_blk.obs_scale, use_loc_v),
            )
            if not torch.compiler.is_compiling():
                with torch.no_grad():
                    p_pure = ((2.0 * tr_sq_f.square()) / (1.0 + tr_sq_f.square())).clamp(TRACE_NORM_EPS, 1.0)
                    r_pure = (1.0 / p_pure).clamp(1.0, float(d_h))
                    p_m = p_pure.mean()
                    r_m = r_pure.mean()
                    core._last_telemetry_tensors = (p_m, r_m, p_m, (1.0 - p_m).clamp(0.0, 1.0))
            return y_out
        rho0_re = torch.matmul(l_re.transpose(-2, -1), l_re) + torch.matmul(l_im.transpose(-2, -1), l_im)
        rho0_im = torch.matmul(l_im.transpose(-2, -1), l_re) - torch.matmul(l_re.transpose(-2, -1), l_im)
        use_fused_kernel = False
    else:
        tok_rho_re = torch.matmul(r_re.transpose(-2, -1), r_re) + torch.matmul(r_im.transpose(-2, -1), r_im)
        tok_rho_im = torch.matmul(r_im.transpose(-2, -1), r_re) - torch.matmul(r_re.transpose(-2, -1), r_im)
        use_fused_kernel = kernel_mode in (
            QuantumAttnKernelMode.FUSED_SINGLE_PASS_RHO_KET,
            QuantumAttnKernelMode.FUSED_SINGLE_PASS_RHO_KET.value,
            QuantumAttnKernelMode.FUSED_HERMITIAN_BORN_RHO_KET,
            QuantumAttnKernelMode.FUSED_HERMITIAN_BORN_RHO_KET.value,
        )
        if use_fused_kernel:
            use_hb = kernel_mode in (
                QuantumAttnKernelMode.FUSED_HERMITIAN_BORN_RHO_KET,
                QuantumAttnKernelMode.FUSED_HERMITIAN_BORN_RHO_KET.value,
            )
            rho0_re, rho0_im, v_bar_re, v_bar_im = fused_quantum_wavepacket_rho_ket_attention(
                q_rot,
                k_rot,
                tok_rho_re,
                tok_rho_im,
                v_mixed,
                attn_mask=attn_mask,
                phase_and_born=(svd_blk.berry_qk_phase, svd_blk.qk_born_beta, svd_blk.sdpa_scale, use_hb),
            )
        else:
            k_rho = max(1, d_h // v_rank)
            row_chunk = v_rank
            chunks = k_rho + 1
            rho_chunks = (
                torch.stack(
                    [
                        tok_rho_re.view(bsz, num_heads, seq_len, k_rho, row_chunk, d_h),
                        tok_rho_im.view(bsz, num_heads, seq_len, k_rho, row_chunk, d_h),
                    ],
                    dim=4,
                )
                .permute(0, 1, 3, 2, 4, 5, 6)
                .reshape(bsz, num_heads, k_rho, seq_len, sdpa_dim)
            )
            v_ket_chunk = v_mixed.view(bsz, seq_len, num_heads, 1, sdpa_dim).permute(0, 2, 3, 1, 4)
            sdpa_v = torch.cat([rho_chunks, v_ket_chunk], dim=2).reshape(bsz, num_heads * chunks, seq_len, sdpa_dim).contiguous()
            sdpa_q, sdpa_k = core._expand_sdpa_qk(q_rot, k_rot, chunks=chunks)
            a_all = F.scaled_dot_product_attention(
                sdpa_q, sdpa_k, sdpa_v, attn_mask=attn_mask, is_causal=(attn_mask is None), scale=svd_blk.sdpa_scale
            ).view(bsz, num_heads, chunks, seq_len, sdpa_dim)
            a_rho = a_all[:, :, :k_rho].permute(0, 1, 3, 2, 4).reshape(bsz, num_heads, seq_len, k_rho, 2, row_chunk, d_h)
            rho0_re = a_rho[:, :, :, :, 0].reshape(bsz, num_heads, seq_len, d_h, d_h)
            rho0_im = a_rho[:, :, :, :, 1].reshape(bsz, num_heads, seq_len, d_h, d_h)
            a_v = a_all[:, :, k_rho].view(bsz, num_heads, seq_len, v_rank, 2, d_h)
            v_bar_re, v_bar_im = a_v[..., 0, :], a_v[..., 1, :]

    can_use_assoc_fast_path = (
        use_fused_kernel
        and core.purification_mode == PurificationMode.QUADRATIC_MCWEENY
        and core.channel_mode != QuantumChannelMode.CPTP_KRAUS_RANK_K
        and core.attn_interior_mode in (
            QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V,
            QuantumAttnInteriorMode.ATTENDED_RHO_ON_ATTENDED_V,
        )
    )
    if can_use_assoc_fast_path:
        use_loc_v = core.attn_interior_mode == QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V
        fn = _get_compiled_associative_ket_fn() if v_wp.is_cuda else associative_mcweeny_unitary_ket_modulation
        y_out, tr_sq_f = fn(
            rho0_re,
            rho0_im,
            v_bar_re,
            v_bar_im,
            v_mixed,
            (core.head_unitary_re, core.head_unitary_im),
            (svd_blk.rho_coupling_scale, svd_blk.obs_scale, use_loc_v),
        )
        if not torch.compiler.is_compiling():
            with torch.no_grad():
                p_pure = ((2.0 * tr_sq_f.square()) / (1.0 + tr_sq_f.square())).clamp(TRACE_NORM_EPS, 1.0)
                r_pure = (1.0 / p_pure).clamp(1.0, float(d_h))
                p_m = p_pure.mean()
                r_m = r_pure.mean()
                core._last_telemetry_tensors = (p_m, r_m, p_m, (1.0 - p_m).clamp(0.0, 1.0))
        return y_out

    evol_re, evol_im, u0_re, u0_im = core._evolve_purified_density(rho0_re, rho0_im, work_dtype)
    eye_d = torch.eye(d_h, device=evol_re.device, dtype=work_dtype)
    rho_tl_re = evol_re - (1.0 / float(d_h)) * eye_d
    rho_tl_im = evol_im

    vu_re = torch.matmul(v_bar_re, u0_re.transpose(-2, -1)) - torch.matmul(v_bar_im, u0_im.transpose(-2, -1))
    vu_im = torch.matmul(v_bar_re, u0_im.transpose(-2, -1)) + torch.matmul(v_bar_im, u0_re.transpose(-2, -1))

    v_loc_c = v_mixed.view(bsz, seq_len, num_heads, v_rank, 2, d_h).permute(0, 2, 1, 3, 4, 5)
    vl_re, vl_im = v_loc_c[..., 0, :], v_loc_c[..., 1, :]
    lu_raw_re = torch.matmul(vl_re, u0_re.transpose(-2, -1)) - torch.matmul(vl_im, u0_im.transpose(-2, -1))
    lu_raw_im = torch.matmul(vl_re, u0_im.transpose(-2, -1)) + torch.matmul(vl_im, u0_re.transpose(-2, -1))
    att_fro = (
        (vu_re.float().square().sum(dim=(-2, -1), keepdim=True) + vu_im.float().square().sum(dim=(-2, -1), keepdim=True))
        .clamp_min(TRACE_NORM_EPS)
        .sqrt()
        .to(work_dtype)
    )
    loc_fro = (
        (
            lu_raw_re.float().square().sum(dim=(-2, -1), keepdim=True)
            + lu_raw_im.float().square().sum(dim=(-2, -1), keepdim=True)
        )
        .clamp_min(TRACE_NORM_EPS)
        .sqrt()
        .to(work_dtype)
    )
    env_ratio = att_fro / loc_fro
    lu_re, lu_im = lu_raw_re * env_ratio, lu_raw_im * env_ratio

    if core.attn_interior_mode == QuantumAttnInteriorMode.ATTENDED_RHO_ON_LOCAL_V:
        tar_re, tar_im = lu_re, lu_im
        op_re, op_im = rho_tl_re, rho_tl_im
    elif core.attn_interior_mode == QuantumAttnInteriorMode.LIOUVILLE_COMMUTATOR_RHO_V:
        loc_u_re, loc_u_im = core._normalize_mode_kets(v_mixed)
        a_re = torch.matmul(loc_u_re.transpose(-2, -1), loc_u_re) + torch.matmul(loc_u_im.transpose(-2, -1), loc_u_im)
        a_im = torch.matmul(loc_u_im.transpose(-2, -1), loc_u_re) - torch.matmul(loc_u_re.transpose(-2, -1), loc_u_im)
        c_re, c_im = compute_liouville_commutator(a_re, a_im, rho_tl_re, rho_tl_im)
        beta_l = torch.tanh(core.liouville_beta).to(work_dtype).view(1, -1, 1, 1, 1)
        tar_re, tar_im = vu_re, vu_im
        op_re, op_im = rho_tl_re + beta_l * c_re, rho_tl_im + beta_l * c_im
    else:
        tar_re, tar_im = vu_re, vu_im
        op_re, op_im = rho_tl_re, rho_tl_im

    rho_v_re = torch.matmul(tar_re, op_re) + torch.matmul(tar_im, op_im)
    rho_v_im = torch.matmul(tar_im, op_re) - torch.matmul(tar_re, op_im)
    beta_rho = svd_blk.rho_coupling_scale.to(work_dtype)
    wp_u_re = vu_re + beta_rho * rho_v_re
    wp_u_im = vu_im + beta_rho * rho_v_im
    y_heads = (
        torch.stack([wp_u_re, wp_u_im], dim=-2)
        .permute(0, 2, 1, 3, 4, 5)
        .reshape(bsz, seq_len, num_heads, sdpa_dim)
    )
    return y_heads * svd_blk.obs_scale.to(work_dtype)


__all__ = [
    "DEFAULT_LOW_RANK_CHOLESKY_K",
    "DEFAULT_QK_BORN_BETA_INIT",
    "DEFAULT_QK_BORN_BETA_MAX",
    "HAS_WAVEPACKET_TRITON",
    "QuantumAttnKernelMode",
    "TRITON_HERMITIAN_BLOCK_N",
    "TRITON_MAX_FUSED_SEQ_LEN",
    "associative_low_rank_cholesky_gram_modulation",
    "associative_mcweeny_unitary_ket_modulation",
    "fused_low_rank_cholesky_ket_attention",
    "fused_quantum_wavepacket_rho_ket_attention",
    "run_coherent_ket_wavepacket_attention",
]
