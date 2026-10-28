"""Standalone Pure-Complex Register Density GPT (`Wave 37` / `EXPERIMENT_PLAN_052.md`).

When to Use This Building Block:
- Use `GeminiPackedComplexRoPEGPT`, `GeminiPackedComplexRoPEBlock`, `ComplexUnitaryInterferenceFFN`,
  and `GeminiPackedComplexRoPEConfig` when you want to evaluate the standalone pure-complex
  architecture built directly around `GeminiBornComplexPackedWithRoPE.py` (with Refinements 1, 3, and 4,
  and zero `gamma_coh`):
  1. Every token state is a pure complex vector `psi_t = psi_re_t + i * psi_im_t in C^{d_complex}`
     partitioned into `H` complex registers of dimension `d_h = 32` (`d_complex = H * d_h`).
  2. Each layer applies `PackedComplexRegisterDensityAttention` (native `ComplexRoPE` on `Q` and `K` only,
     softmax-free `p`-boson Born weights, 1-GEMM Hermitian triangle packing + first-moment anchor ket
     of width `d_h^2 + 2 * d_h = 1088`, and `V`-space Krylov Power Iteration probing of the McWeeny-purified
     density matrix `rho_t^purified = rho_t^2 / Tr(rho_t^2)`), followed by a complex unitary two-beam
     interferometer `ComplexUnitaryInterferenceFFN`.
  3. Readout projects complex states onto tied orthogonalized complex vocabulary detectors
     `M_v = W_meas E_v in C^{d_complex}` using the Hermitian inner product `<psi_t | M_v>_C` and
     analytical Parseval scale `beta_0 = c^* * sqrt(2 * d_complex)` with exact chunked-train / full-eval parity.

Why It Was Built & Mathematical Provenance (`EXPERIMENT_PLAN_052.md` / `Wave 37`):
- Keeps `GeminiBornComplexPackedWithRoPE.py` untouched as the reference specification while providing a
  complete, standalone pure-complex GPT model to test Refinements 1, 3, and 4 head-to-head on `FineWeb-Edu`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..kernels.fused_parseval_ce import compiled_parseval_convex_traceless_chunk_step
from ..layers.packed_complex_register_density_attn import (
    ComplexLinear,
    ComplexProbeMode,
    DEFAULT_COMPLEX_NORM_EPS,
    DEFAULT_COMPLEX_REGISTER_DIM,
    DEFAULT_P_BOSON_POWER,
    PackedComplexRegisterDensityAttention,
    complex_l2_normalize,
)
from ..optim.muon import (
    DEFAULT_MUON_REFERENCE_BATCH_TOKENS,
    StiefelMuon,
    collect_named_complex_muon_pairs,
    compute_large_batch_muon_lr_scale,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Documented in EXPERIMENT_PLAN_052.md & EXPERIMENT_PLAN_053.md
# ==============================================================================
DEFAULT_COMPLEX_PARSEVAL_C: float = 1.48
DEFAULT_COMPLEX_FFN_EXPANSION: int = 2
DEFAULT_COMPLEX_RESIDUAL_INIT: float = 0.35
DEFAULT_COMPLEX_PEAK_GAMMA: float = 0.42
DEFAULT_COMPLEX_PRIOR_TEMP: float = 0.35
DEFAULT_COMPLEX_VOCAB_CHUNK: int = 4096
DEFAULT_COMPLEX_TOK_CHUNK: int = 512
DEFAULT_COMPLEX_MUON_LR: float = 0.015
DEFAULT_COMPLEX_EMBED_LR: float = 0.008
DEFAULT_COMPLEX_GATE_LR: float = 0.003
MIN_COMPLEX_MUON_DIM: int = 16
MAX_COMPLEX_MUON_DIM: int = 4096


@dataclass
class GeminiPackedComplexRoPEConfig:
    """Configuration for standalone `GeminiPackedComplexRoPEGPT` (`Waves 37–39` / `EXPERIMENT_PLAN_052`–`054`)."""

    vocab_size: int = 50304
    valid_vocab_size: int = 50257
    d_complex: int = 128
    n_registers: int = 4
    d_register: int = DEFAULT_COMPLEX_REGISTER_DIM
    num_layers: int = 6
    block_size: int = 512
    p_boson: int = DEFAULT_P_BOSON_POWER
    probe_mode: ComplexProbeMode = "v_krylov"
    normalize_krylov_ket: bool = True
    ffn_expansion: int = DEFAULT_COMPLEX_FFN_EXPANSION
    parseval_c: float = DEFAULT_COMPLEX_PARSEVAL_C
    peak_gamma: float = DEFAULT_COMPLEX_PEAK_GAMMA
    prior_temp: float = DEFAULT_COMPLEX_PRIOR_TEMP
    vocab_chunk_size: int = DEFAULT_COMPLEX_VOCAB_CHUNK
    checkpoint_blocks: bool = False

    def __post_init__(self) -> None:
        if self.n_registers * self.d_register != self.d_complex:
            if self.d_complex % self.d_register == 0:
                self.n_registers = max(1, self.d_complex // self.d_register)
            elif self.d_complex % self.n_registers == 0:
                self.d_register = max(8, self.d_complex // self.n_registers)
            if self.n_registers * self.d_register != self.d_complex:
                raise ValueError(
                    f"d_complex ({self.d_complex}) must equal n_registers ({self.n_registers}) * d_register ({self.d_register})"
                )

    @property
    def real_equiv_dim(self) -> int:
        """Return total real degrees of freedom per token (`2 * d_complex`)."""
        return 2 * int(self.d_complex)


class ComplexUnitaryInterferenceFFN(nn.Module):
    """Complex Unitary Two-Beam Phase-Gated Interferometer (`C^{d_complex} -> C^{m * d_complex} -> C^{d_complex}`)."""

    def __init__(self, d_complex: int, expansion: int = DEFAULT_COMPLEX_FFN_EXPANSION) -> None:
        super().__init__()
        self.d_complex = int(d_complex)
        self.d_hidden = int(d_complex) * int(expansion)
        self.beam1 = ComplexLinear(self.d_complex, self.d_hidden)
        self.beam2 = ComplexLinear(self.d_complex, self.d_hidden)
        self.proj_down = ComplexLinear(self.d_hidden, self.d_complex)
        self.phase_gain = nn.Parameter(torch.full((self.d_hidden,), 0.35))

    def forward(
        self,
        psi_re: torch.Tensor,
        psi_im: torch.Tensor,
        eps: float = DEFAULT_COMPLEX_NORM_EPS,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply two-beam complex interference and project back to `C^{d_complex}`."""
        b1_re, b1_im = self.beam1(psi_re, psi_im)
        b2_re, b2_im = self.beam2(psi_re, psi_im)
        b1_re, b1_im = complex_l2_normalize(b1_re, b1_im, dim=-1, eps=eps)
        b2_re, b2_im = complex_l2_normalize(b2_re, b2_im, dim=-1, eps=eps)

        # Cross-beam Hermitian coherence Re(b1^* * b2) and Im(b1^* * b2)
        coh_re = b1_re * b2_re + b1_im * b2_im
        coh_im = b1_re * b2_im - b1_im * b2_re
        gain = torch.tanh(self.phase_gain).to(dtype=b1_re.dtype)
        mod_re = 1.0 + gain * coh_re
        mod_im = gain * coh_im

        z_re = b1_re * mod_re - b1_im * mod_im
        z_im = b1_re * mod_im + b1_im * mod_re
        out_re, out_im = self.proj_down(z_re, z_im)
        return complex_l2_normalize(out_re, out_im, dim=-1, eps=eps)


class GeminiPackedComplexRoPEBlock(nn.Module):
    """Single complex transformer block: `PackedComplexRegisterDensityAttention` + `ComplexUnitaryInterferenceFFN`."""

    def __init__(self, cfg: GeminiPackedComplexRoPEConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = PackedComplexRegisterDensityAttention(
            d_model=cfg.d_complex,
            n_registers=cfg.n_registers,
            d_register=cfg.d_register,
            p_boson=cfg.p_boson,
            probe_mode=cfg.probe_mode,
            normalize_krylov_ket=cfg.normalize_krylov_ket,
            value_residual_mixing=(layer_idx > 0),
        )
        self.ffn = ComplexUnitaryInterferenceFFN(cfg.d_complex, expansion=cfg.ffn_expansion)
        init_scale = DEFAULT_COMPLEX_RESIDUAL_INIT / math.sqrt(float(max(1, layer_idx + 1)))
        self.attn_gate = nn.Parameter(torch.full((cfg.d_complex,), init_scale))
        self.ffn_gate = nn.Parameter(torch.full((cfg.d_complex,), init_scale))

    def forward(
        self,
        psi_re: torch.Tensor,
        psi_im: torch.Tensor,
        v0_re: torch.Tensor | None = None,
        v0_im: torch.Tensor | None = None,
        eps: float = DEFAULT_COMPLEX_NORM_EPS,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run complex register density attention and complex unitary interference FFN."""
        attn_re, attn_im, v_raw_re, v_raw_im = self.attn.forward_complex(
            psi_re, psi_im, v0_re=v0_re, v0_im=v0_im, eps=eps
        )
        attn_re, attn_im = complex_l2_normalize(attn_re, attn_im, dim=-1, eps=eps)
        g_a = torch.tanh(self.attn_gate).to(dtype=attn_re.dtype)
        p_re = psi_re.to(dtype=attn_re.dtype)
        p_im = psi_im.to(dtype=attn_im.dtype)
        mid_re, mid_im = complex_l2_normalize(p_re + g_a * attn_re, p_im + g_a * attn_im, dim=-1, eps=eps)

        ffn_re, ffn_im = self.ffn(mid_re, mid_im, eps=eps)
        g_f = torch.tanh(self.ffn_gate).to(dtype=ffn_re.dtype)
        out_re, out_im = complex_l2_normalize(mid_re + g_f * ffn_re, mid_im + g_f * ffn_im, dim=-1, eps=eps)
        return out_re, out_im, v_raw_re, v_raw_im


def _compute_complex_born_logits_from_overlaps(
    ov: torch.Tensor,
    beta: float,
    peak_gamma: float,
    prior_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute centered Convex-Peak Traceless Born logits from Hermitian overlaps `ov = Re(<psi_t | M_v>_C)`."""
    ov_f = ov.float()
    c_hat = ov_f - ov_f.mean(dim=-1, keepdim=True)
    pos_sq = F.relu(c_hat).square()
    tau_plus = pos_sq.mean(dim=-1, keepdim=True)
    obs = c_hat + float(peak_gamma) * (pos_sq - tau_plus)
    logits = float(beta) * obs
    if prior_bias is not None:
        logits = logits + prior_bias.float().unsqueeze(0)
    return logits


def _eager_complex_chunk_step(
    w_sharp: torch.Tensor,
    yc_safe: torch.Tensor,
    row_scale: float,
    beta_0: float,
    gamma: float,
    log_prior_row: torch.Tensor | None,
    v_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager fallback chunk step matching `compiled_parseval_convex_traceless_chunk_step` (`<= 8` args)."""
    c_raw = w_sharp.float()
    c_hat = c_raw - c_raw.mean(dim=-1, keepdim=True)
    c_pos = torch.clamp_min(c_hat, 0.0)
    c_sq = c_pos * c_pos
    o_hat = c_hat + gamma * (c_sq - c_sq.mean(dim=-1, keepdim=True))
    logits = beta_0 * o_hat
    if log_prior_row is not None:
        logits = logits + log_prior_row

    max_logits = torch.amax(logits, dim=-1, keepdim=True)
    logits_shift = logits - max_logits
    target_logits = logits_shift.gather(1, yc_safe)
    exp_logits = torch.exp(logits_shift)
    sum_exp = exp_logits.sum(dim=-1, keepdim=True)
    loss_c = (torch.log(sum_exp) - target_logits).sum()

    probs = exp_logits / sum_exp
    is_target = (v_idx == yc_safe).float()
    g_c = (probs - is_target) * row_scale
    u_s = g_c * beta_0
    u_s_mean = u_s.mean(dim=-1, keepdim=True)
    g_hat = u_s + (2.0 * gamma) * c_pos * (u_s - u_s_mean)
    grad_s = (g_hat - g_hat.mean(dim=-1, keepdim=True)).to(dtype=w_sharp.dtype)
    return loss_c, grad_s


class _ComplexBornChunkedCEFn(torch.autograd.Function):
    """Single-pass analytical-backward `bfloat16` Tensor-Core chunked CE for `GeminiPackedComplexRoPEGPT`."""

    @staticmethod
    def forward(
        ctx,
        psi_flat: torch.Tensor,
        det_flat: torch.Tensor,
        valid_targets: torch.Tensor,
        prior_b: torch.Tensor,
        beta: float,
        peak_gamma: float,
    ) -> torch.Tensor:
        compute_dtype = (
            torch.get_autocast_gpu_dtype()
            if (psi_flat.is_cuda and torch.is_autocast_enabled())
            else psi_flat.dtype
        )
        psi_w = psi_flat.to(dtype=compute_dtype).contiguous()
        det_w = det_flat.to(dtype=compute_dtype).contiguous()
        n_tok = psi_w.size(0)
        v_eff = det_w.size(0)
        tok_chunk = DEFAULT_COMPLEX_TOK_CHUNK
        inv_n = 1.0 / float(max(1, n_tok))

        total_loss = torch.zeros((), device=psi_w.device, dtype=torch.float32)
        grad_psi = torch.empty_like(psi_w)
        grad_det = torch.zeros_like(det_w)

        prior_row = prior_b.float().unsqueeze(0)
        v_idx = torch.arange(v_eff, device=psi_w.device).unsqueeze(0)
        use_compiled = bool(psi_w.is_cuda and not torch.compiler.is_compiling())
        w_sharp_buf = torch.empty((min(tok_chunk, n_tok), v_eff), device=psi_w.device, dtype=compute_dtype)
        w_diff_buf = torch.zeros((min(tok_chunk, n_tok), 1), device=psi_w.device, dtype=compute_dtype)
        valid_buf = torch.ones((min(tok_chunk, n_tok), 1), device=psi_w.device, dtype=torch.float32)
        scale_buf = torch.full((min(tok_chunk, n_tok), 1), inv_n, device=psi_w.device, dtype=torch.float32)
        det_w_t = det_w.t()

        for t0 in range(0, n_tok, tok_chunk):
            t1 = min(n_tok, t0 + tok_chunk)
            c_len = t1 - t0
            psi_c = psi_w[t0:t1]
            yc_safe = valid_targets[t0:t1].unsqueeze(1)
            w_sharp = w_sharp_buf[:c_len]
            torch.mm(psi_c, det_w_t, out=w_sharp)

            if use_compiled:
                loss_c, _, grad_s, _ = compiled_parseval_convex_traceless_chunk_step(
                    w_sharp,
                    w_diff_buf[:c_len],
                    yc_safe,
                    (valid_buf[:c_len], scale_buf[:c_len]),
                    (float(beta), 0.0, float(peak_gamma), 0.0),
                    prior_row,
                    v_idx,
                )
            else:
                loss_c, grad_s = _eager_complex_chunk_step(
                    w_sharp, yc_safe, inv_n, float(beta), float(peak_gamma), prior_row, v_idx
                )

            total_loss = total_loss + loss_c
            torch.mm(grad_s, det_w, out=grad_psi[t0:t1])
            grad_det.addmm_(grad_s.t(), psi_c, beta=1.0, alpha=1.0)

        ctx.save_for_backward(grad_psi, grad_det)
        ctx.psi_dtype = psi_flat.dtype
        ctx.det_dtype = det_flat.dtype
        return total_loss * inv_n

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_psi, grad_det = ctx.saved_tensors
        scale = grad_output.to(dtype=grad_psi.dtype)
        return (
            (grad_psi * scale).to(dtype=ctx.psi_dtype),
            (grad_det * scale).to(dtype=ctx.det_dtype),
            None,
            None,
            None,
            None,
        )


class GeminiPackedComplexRoPEGPT(nn.Module):
    """Standalone Pure-Complex Register Density GPT (`Waves 37–38` / `EXPERIMENT_PLAN_052` & `053`)."""

    def __init__(self, cfg: GeminiPackedComplexRoPEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_re = nn.Parameter(torch.randn(cfg.vocab_size, cfg.d_complex) * 0.02)
        self.embed_im = nn.Parameter(torch.randn(cfg.vocab_size, cfg.d_complex) * 0.02)
        with torch.no_grad():
            e_re, e_im = complex_l2_normalize(self.embed_re.data, self.embed_im.data, dim=-1)
            self.embed_re.data.copy_(e_re)
            self.embed_im.data.copy_(e_im)

        self.meas_proj = ComplexLinear(cfg.d_complex, cfg.d_complex)
        self.blocks = nn.ModuleList([GeminiPackedComplexRoPEBlock(cfg, i) for i in range(cfg.num_layers)])
        self.psi0_readout_gate = nn.Parameter(torch.full((cfg.d_complex,), 0.04))
        self.register_buffer("povm_log_prior", torch.zeros(cfg.vocab_size, dtype=torch.float32))
        self._training_progress: float = 0.0

    def set_training_progress(self, progress: float) -> None:
        """Record normalized training progress `tau in [0.0, 1.0]`."""
        self._training_progress = float(min(max(progress, 0.0), 1.0))

    def current_beta(self) -> float:
        """Return analytical Parseval inverse temperature calibrated to effective co-occurrence rank."""
        d_real = float(self.cfg.real_equiv_dim)
        d_eff = 256.0 * ((d_real / 256.0) ** 0.30) if d_real > 256.0 else d_real
        return float(self.cfg.parseval_c) * math.sqrt(d_eff)

    def _synthesize_detectors_step(
        self, embed_re_v: torch.Tensor, embed_im_v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inner Gram-Schmidt complex detector synthesis step in native/autocast dtype."""
        e_re, e_im = complex_l2_normalize(embed_re_v, embed_im_v, dim=-1)
        m_re, m_im = self.meas_proj(e_re, e_im)
        ov_re = (e_re.float() * m_re.float() + e_im.float() * m_im.float()).sum(dim=-1, keepdim=True).to(dtype=m_re.dtype)
        orth_re = m_re - ov_re * e_re
        orth_im = m_im - ov_re * e_im
        return complex_l2_normalize(orth_re, orth_im, dim=-1)

    def synthesize_complex_detectors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Synthesize unit-norm complex vocabulary detectors `(det_re, det_im)` orthogonalized against self-echo."""
        v_eff = int(self.cfg.valid_vocab_size)
        e_re_v = self.embed_re[:v_eff]
        e_im_v = self.embed_im[:v_eff]
        if self.training and torch.is_grad_enabled() and not torch.compiler.is_compiling():
            return checkpoint(self._synthesize_detectors_step, e_re_v, e_im_v, use_reentrant=False)
        return self._synthesize_detectors_step(e_re_v, e_im_v)

    @torch.no_grad()
    def compute_sharp_eff_rank(self) -> float:
        """Compute participation ratio `R_sharp` of the stacked `[det_re, det_im]` detector frame in `R^{2*d_complex}`."""
        det_re, det_im = self.synthesize_complex_detectors()
        m_full = torch.cat([det_re.detach().float(), det_im.detach().float()], dim=-1)
        m_cent = m_full - m_full.mean(dim=0, keepdim=True)
        gram = m_cent.t() @ m_cent
        tr = torch.diagonal(gram).sum()
        return float(((tr * tr) / (gram * gram).sum().clamp_min(1e-12)).item())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_logits: bool | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Execute pure-complex forward pass through `L` register density blocks and tied complex Born readout."""
        b_sz, seq_t = idx.shape
        d_c = int(self.cfg.d_complex)
        h_num, dh = int(self.cfg.n_registers), int(self.cfg.d_register)

        psi0_re = self.embed_re.index_select(0, idx.reshape(-1)).view(b_sz, seq_t, d_c)
        psi0_im = self.embed_im.index_select(0, idx.reshape(-1)).view(b_sz, seq_t, d_c)
        psi0_re, psi0_im = complex_l2_normalize(psi0_re, psi0_im, dim=-1)

        psi_re, psi_im = psi0_re, psi0_im
        v0_re: torch.Tensor | None = None
        v0_im: torch.Tensor | None = None
        use_blk_ckpt = bool(
            self.cfg.checkpoint_blocks
            and self.training
            and torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
        )

        for l_idx, block in enumerate(self.blocks):
            if use_blk_ckpt:
                psi_re, psi_im, v_raw_re, v_raw_im = checkpoint(
                    block, psi_re, psi_im, v0_re, v0_im, use_reentrant=False
                )
            else:
                psi_re, psi_im, v_raw_re, v_raw_im = block(psi_re, psi_im, v0_re=v0_re, v0_im=v0_im)
            if l_idx == 0:
                v0_re = v_raw_re.view(b_sz, h_num, seq_t, dh)
                v0_im = v_raw_im.view(b_sz, h_num, seq_t, dh)

        # Conjugate residual skip from initial complex ket (psi0_re - i * psi0_im)
        g0 = torch.tanh(self.psi0_readout_gate).to(dtype=psi_re.dtype)
        psi_re, psi_im = complex_l2_normalize(
            psi_re + g0 * psi0_re.to(dtype=psi_re.dtype),
            psi_im - g0 * psi0_im.to(dtype=psi_im.dtype),
            dim=-1,
        )

        det_re, det_im = self.synthesize_complex_detectors()
        v_eff = det_re.size(0)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        psi_flat = torch.cat([psi_re.reshape(-1, d_c), psi_im.reshape(-1, d_c)], dim=-1) * inv_sqrt2
        det_flat = torch.cat([det_re, det_im], dim=-1) * inv_sqrt2

        beta = self.current_beta()
        p_gamma = float(self.cfg.peak_gamma)
        prior_b = float(self.cfg.prior_temp) * self.povm_log_prior[:v_eff]
        want_logits = (targets is None) if return_logits is None else bool(return_logits)

        if targets is not None and not want_logits:
            valid_targets = targets.clamp(min=0, max=v_eff - 1).reshape(-1)
            loss = _ComplexBornChunkedCEFn.apply(psi_flat, det_flat, valid_targets, prior_b, beta, p_gamma)
            out_logits = loss if return_logits is None else None
            return out_logits, loss

        ov_all = torch.matmul(psi_flat, det_flat.t())
        logits_flat = _compute_complex_born_logits_from_overlaps(ov_all, beta=beta, peak_gamma=p_gamma, prior_bias=prior_b)
        logits_full = logits_flat.view(b_sz, seq_t, v_eff)

        loss = None
        if targets is not None:
            valid_targets = targets.clamp(min=0, max=v_eff - 1)
            loss = F.cross_entropy(logits_flat, valid_targets.reshape(-1))

        out_logits = logits_full if want_logits else (loss if (targets is not None and return_logits is None) else None)
        return out_logits, loss

    @torch.no_grad()
    def retract_complex_unitary_weights(self, rate: float = 0.05) -> float:
        """Apply soft complex Stiefel/unitary landing step `W <- W - rate * (W W^H - I) W` to all `ComplexLinear` layers."""
        max_err = 0.0
        for mod in self.modules():
            if isinstance(mod, ComplexLinear) and mod.in_features == mod.out_features:
                w_re = mod.weight_re.data.float()
                w_im = mod.weight_im.data.float()
                d_sz = w_re.size(0)
                eye = torch.eye(d_sz, device=w_re.device, dtype=torch.float32)
                gram_re = w_re @ w_re.t() + w_im @ w_im.t() - eye
                gram_im = w_im @ w_re.t() - w_re @ w_im.t()
                err = float(torch.sqrt(gram_re.square().mean() + gram_im.square().mean()).item())
                max_err = max(max_err, err)
                denom = 1.0 + math.sqrt(err)
                step_re = (gram_re @ w_re - gram_im @ w_im) / denom
                step_im = (gram_re @ w_im + gram_im @ w_re) / denom
                mod.weight_re.data.copy_((w_re - rate * step_re).to(mod.weight_re.dtype))
                mod.weight_im.data.copy_((w_im - rate * step_im).to(mod.weight_im.dtype))
        return max_err


@torch.no_grad()
def initialize_gemini_complex_gpt_with_ppmi_svd(
    model: GeminiPackedComplexRoPEGPT,
    ppmi_embeddings: torch.Tensor,
    prior_tokens: torch.Tensor,
) -> dict[str, float]:
    """Initialize `embed_re, embed_im` across all `H` complex registers from PPMI-SVD pairs and harmonics."""
    d_c = int(model.cfg.d_complex)
    v_eff = min(int(model.cfg.valid_vocab_size), ppmi_embeddings.size(0))
    if ppmi_embeddings.size(1) >= 2 * d_c:
        raw_re = ppmi_embeddings[:v_eff, 0 : 2 * d_c : 2].float()
        raw_im = ppmi_embeddings[:v_eff, 1 : 2 * d_c : 2].float()
    else:
        base_re = ppmi_embeddings[:v_eff, 0::2].float()
        base_im = ppmi_embeddings[:v_eff, 1::2].float()
        b_dim = base_re.size(1)
        rms_target = base_re.square().mean().sqrt().clamp_min(1e-6)
        n_tiles = (d_c + b_dim - 1) // b_dim
        re_tiles: list[torch.Tensor] = [base_re]
        im_tiles: list[torch.Tensor] = [base_im]
        for k in range(1, n_tiles):
            angle = (math.pi * float(k)) / float(2 * n_tiles)
            c_a, s_a = math.cos(angle), math.sin(angle)
            perm_re = torch.roll(base_re, shifts=32 * k, dims=-1)
            perm_im = torch.roll(base_im, shifts=32 * k, dims=-1)
            rot_re = c_a * perm_re - s_a * perm_im
            rot_im = s_a * perm_re + c_a * perm_im
            quad_re = base_re * torch.roll(base_im, shifts=7 * k, dims=-1)
            quad_im = base_im * torch.roll(base_re, shifts=11 * k, dims=-1)
            quad_re = quad_re - quad_re.mean(dim=0, keepdim=True)
            quad_im = quad_im - quad_im.mean(dim=0, keepdim=True)
            q_rms = quad_re.square().mean().sqrt().clamp_min(1e-6)
            quad_re = quad_re * (rms_target / q_rms)
            quad_im = quad_im * (rms_target / q_rms)
            re_tiles.append(0.75 * rot_re + 0.25 * quad_re)
            im_tiles.append(0.75 * rot_im + 0.25 * quad_im)
        raw_re = torch.cat(re_tiles, dim=-1)[:, :d_c].contiguous()
        raw_im = torch.cat(im_tiles, dim=-1)[:, :d_c].contiguous()
    e_re, e_im = complex_l2_normalize(raw_re.to(model.embed_re.device), raw_im.to(model.embed_im.device), dim=-1)
    model.embed_re.data[:v_eff].copy_(e_re.to(model.embed_re.dtype))
    model.embed_im.data[:v_eff].copy_(e_im.to(model.embed_im.dtype))

    flat_tok = prior_tokens.reshape(-1).long().clamp(0, v_eff - 1)
    counts = torch.bincount(flat_tok, minlength=v_eff).float().clamp_min(1.0)
    probs = counts / counts.sum()
    log_p = torch.log(probs)
    log_p = log_p - log_p.mean()
    model.povm_log_prior.zero_()
    model.povm_log_prior[:v_eff].copy_(log_p.to(model.povm_log_prior.device))

    r_sharp = model.compute_sharp_eff_rank()
    return {
        "sharp_eff_rank": r_sharp,
        "beta_0": model.current_beta(),
    }


def build_gemini_complex_optimizers(
    model: GeminiPackedComplexRoPEGPT,
    base_muon_lr: float = DEFAULT_COMPLEX_MUON_LR,
    base_embed_lr: float = DEFAULT_COMPLEX_EMBED_LR,
    base_gate_lr: float = DEFAULT_COMPLEX_GATE_LR,
    use_joint_complex_muon: bool = True,
    batch_tokens: int = DEFAULT_MUON_REFERENCE_BATCH_TOKENS,
    weight_decay: float = 0.0,
    stiefel_penalty: float = 0.02,
) -> tuple[StiefelMuon, torch.optim.AdamW]:
    """Build `StiefelMuon` (joint-complex or split-real) and `AdamW` (for complex embeddings & gates).

    Provenance (`Wave 44` / `EXPERIMENT_PLAN_059.md` Track 1):
    - When `use_joint_complex_muon=True` (default), pairs `(w_re, w_im)` of every 2D complex linear projection
      are routed together (`is_complex_pair_group=True`) so `StiefelMuon` orthogonalizes `G = G_re + i G_im`
      in `U(d)` via `complex_polar_express_orth` instead of orthogonalizing real and imaginary parts separately.
    - Scales `muon_lr` by `compute_large_batch_muon_lr_scale(batch_tokens)`.
    """
    d_real = float(max(32, model.cfg.real_equiv_dim))
    d_ratio = 256.0 / d_real
    depth_scale = math.sqrt(6.0 / float(max(6, model.cfg.num_layers)))
    if d_real >= 256.0:
        muon_scale = (d_ratio ** 0.15) * depth_scale
        embed_scale = d_ratio ** 0.75
        gate_scale = math.sqrt(d_ratio)
    else:
        muon_scale = min(1.30, d_ratio ** 0.15) * depth_scale
        embed_scale = min(1.45, d_ratio ** 0.30)
        gate_scale = min(1.35, d_ratio ** 0.25)

    batch_lr_scale = compute_large_batch_muon_lr_scale(batch_tokens=int(batch_tokens))
    muon_lr = float(base_muon_lr) * muon_scale * batch_lr_scale
    embed_lr = float(base_embed_lr) * embed_scale
    gate_lr = float(base_gate_lr) * gate_scale

    embed_params: list[nn.Parameter] = []
    gate_params: list[nn.Parameter] = []

    if use_joint_complex_muon:
        complex_pairs, paired_ids = collect_named_complex_muon_pairs(
            model,
            min_dim=MIN_COMPLEX_MUON_DIM,
            max_dim=MAX_COMPLEX_MUON_DIM,
        )
        for name, param in model.named_parameters():
            if not param.requires_grad or id(param) in paired_ids:
                continue
            if "embed_" in name:
                embed_params.append(param)
            else:
                gate_params.append(param)
        muon_groups = [
            {
                "params": complex_pairs,
                "lr": muon_lr,
                "base_lr": muon_lr,
                "is_complex_pair_group": True,
            }
        ]
    else:
        muon_params: list[nn.Parameter] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "embed_" in name:
                embed_params.append(param)
            elif (
                param.ndim == 2
                and min(param.shape) >= MIN_COMPLEX_MUON_DIM
                and max(param.shape) <= MAX_COMPLEX_MUON_DIM
            ):
                muon_params.append(param)
            else:
                gate_params.append(param)
        muon_groups = [{"params": muon_params, "lr": muon_lr, "base_lr": muon_lr}]

    opt_muon = StiefelMuon(
        muon_groups,
        lr=muon_lr,
        momentum=0.95,
        weight_decay=float(weight_decay),
        polar_method="polar_express",
        stiefel_penalty=float(stiefel_penalty),
        retract_every=2,
        use_normuon=True,
        use_twin_rail=True,
    )
    opt_adam = torch.optim.AdamW(
        [
            {"params": embed_params, "lr": embed_lr, "base_lr": embed_lr, "weight_decay": float(weight_decay)},
            {"params": gate_params, "lr": gate_lr, "base_lr": gate_lr, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    return opt_muon, opt_adam


__all__ = [
    "ComplexUnitaryInterferenceFFN",
    "DEFAULT_COMPLEX_EMBED_LR",
    "DEFAULT_COMPLEX_FFN_EXPANSION",
    "DEFAULT_COMPLEX_GATE_LR",
    "DEFAULT_COMPLEX_MUON_LR",
    "DEFAULT_COMPLEX_PARSEVAL_C",
    "DEFAULT_COMPLEX_PEAK_GAMMA",
    "DEFAULT_COMPLEX_PRIOR_TEMP",
    "DEFAULT_COMPLEX_RESIDUAL_INIT",
    "DEFAULT_COMPLEX_TOK_CHUNK",
    "DEFAULT_COMPLEX_VOCAB_CHUNK",
    "GeminiPackedComplexRoPEBlock",
    "GeminiPackedComplexRoPEConfig",
    "GeminiPackedComplexRoPEGPT",
    "MAX_COMPLEX_MUON_DIM",
    "MIN_COMPLEX_MUON_DIM",
    "build_gemini_complex_optimizers",
    "initialize_gemini_complex_gpt_with_ppmi_svd",
]
