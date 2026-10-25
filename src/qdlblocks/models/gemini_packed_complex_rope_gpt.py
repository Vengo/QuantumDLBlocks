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

from ..layers.packed_complex_register_density_attn import (
    ComplexLinear,
    ComplexProbeMode,
    DEFAULT_COMPLEX_NORM_EPS,
    DEFAULT_COMPLEX_REGISTER_DIM,
    DEFAULT_P_BOSON_POWER,
    PackedComplexRegisterDensityAttention,
    complex_l2_normalize,
)
from ..optim.muon import StiefelMuon

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Documented in EXPERIMENT_PLAN_052.md (Section 6.1)
# ==============================================================================
DEFAULT_COMPLEX_PARSEVAL_C: float = 1.48
DEFAULT_COMPLEX_FFN_EXPANSION: int = 2
DEFAULT_COMPLEX_RESIDUAL_INIT: float = 0.35
DEFAULT_COMPLEX_PEAK_GAMMA: float = 0.42
DEFAULT_COMPLEX_PRIOR_TEMP: float = 0.35
DEFAULT_COMPLEX_VOCAB_CHUNK: int = 4096
DEFAULT_COMPLEX_MUON_LR: float = 0.015
DEFAULT_COMPLEX_EMBED_LR: float = 0.008
DEFAULT_COMPLEX_GATE_LR: float = 0.003


@dataclass
class GeminiPackedComplexRoPEConfig:
    """Configuration for standalone `GeminiPackedComplexRoPEGPT` (`Wave 37` / `EXPERIMENT_PLAN_052.md`)."""

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

    def __post_init__(self) -> None:
        if self.n_registers * self.d_register != self.d_complex:
            self.n_registers = max(1, self.d_complex // self.d_register)
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
        gain = torch.tanh(self.phase_gain).to(dtype=psi_re.dtype)
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
        g_a = torch.tanh(self.attn_gate).to(dtype=psi_re.dtype)
        mid_re, mid_im = complex_l2_normalize(psi_re + g_a * attn_re, psi_im + g_a * attn_im, dim=-1, eps=eps)

        ffn_re, ffn_im = self.ffn(mid_re, mid_im, eps=eps)
        g_f = torch.tanh(self.ffn_gate).to(dtype=psi_re.dtype)
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


def _compute_complex_token_chunk_ce(
    psi_chunk: torch.Tensor,
    det_flat: torch.Tensor,
    targets_chunk: torch.Tensor,
    prior_b: torch.Tensor,
    beta: float,
    peak_gamma: float,
) -> torch.Tensor:
    """Compute cross-entropy sum on a `(T_chunk, V_eff)` token slice (`<= 8` args, checkpointable)."""
    ov_c = torch.matmul(psi_chunk, det_flat.t())
    logits_c = _compute_complex_born_logits_from_overlaps(ov_c, beta=beta, peak_gamma=peak_gamma, prior_bias=prior_b)
    return F.cross_entropy(logits_c, targets_chunk, reduction="sum")


class GeminiPackedComplexRoPEGPT(nn.Module):
    """Standalone Pure-Complex Register Density GPT (`Wave 37` / `EXPERIMENT_PLAN_052.md`)."""

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
        """Return analytical Parseval inverse temperature `beta_0 = c^* * sqrt(2 * d_complex)`."""
        return float(self.cfg.parseval_c) * math.sqrt(float(self.cfg.real_equiv_dim))

    def synthesize_complex_detectors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Synthesize unit-norm complex vocabulary detectors `(det_re, det_im)` orthogonalized against self-echo."""
        v_eff = int(self.cfg.valid_vocab_size)
        e_re, e_im = complex_l2_normalize(self.embed_re[:v_eff], self.embed_im[:v_eff], dim=-1)
        m_re, m_im = self.meas_proj(e_re, e_im)
        # Complex Gram-Schmidt orthogonalization: remove real self-overlap Re(<e_v | m_v>_C) so self-echo == 0
        ov_re = (e_re * m_re + e_im * m_im).sum(dim=-1, keepdim=True)
        orth_re = m_re - ov_re * e_re
        orth_im = m_im - ov_re * e_im
        return complex_l2_normalize(orth_re, orth_im, dim=-1)

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

        for l_idx, block in enumerate(self.blocks):
            psi_re, psi_im, v_raw_re, v_raw_im = block(psi_re, psi_im, v0_re=v0_re, v0_im=v0_im)
            if l_idx == 0:
                v0_re = v_raw_re.view(b_sz, h_num, seq_t, dh)
                v0_im = v_raw_im.view(b_sz, h_num, seq_t, dh)

        # Conjugate residual skip from initial complex ket (psi0_re - i * psi0_im)
        g0 = torch.tanh(self.psi0_readout_gate).to(dtype=psi_re.dtype)
        psi_re, psi_im = complex_l2_normalize(psi_re + g0 * psi0_re, psi_im - g0 * psi0_im, dim=-1)

        det_re, det_im = self.synthesize_complex_detectors()
        v_eff = det_re.size(0)
        psi_flat = torch.cat([psi_re.reshape(-1, d_c), psi_im.reshape(-1, d_c)], dim=-1)
        det_flat = torch.cat([det_re, det_im], dim=-1) * (1.0 / math.sqrt(2.0))
        psi_flat = psi_flat * (1.0 / math.sqrt(2.0))

        beta = self.current_beta()
        p_gamma = float(self.cfg.peak_gamma)
        prior_b = float(self.cfg.prior_temp) * self.povm_log_prior[:v_eff]
        want_logits = (targets is None) if return_logits is None else bool(return_logits)

        if targets is not None and not want_logits:
            valid_targets = targets.clamp(min=0, max=v_eff - 1).reshape(-1)
            n_tok = psi_flat.size(0)
            tok_chunk = 512
            total_loss = torch.zeros((), device=psi_flat.device, dtype=torch.float32)
            for t0 in range(0, n_tok, tok_chunk):
                t1 = min(n_tok, t0 + tok_chunk)
                if self.training and torch.is_grad_enabled() and not torch.compiler.is_compiling():
                    chunk_loss = checkpoint(
                        _compute_complex_token_chunk_ce,
                        psi_flat[t0:t1],
                        det_flat,
                        valid_targets[t0:t1],
                        prior_b,
                        beta,
                        p_gamma,
                        use_reentrant=False,
                    )
                else:
                    chunk_loss = _compute_complex_token_chunk_ce(
                        psi_flat[t0:t1], det_flat, valid_targets[t0:t1], prior_b, beta, p_gamma
                    )
                total_loss = total_loss + chunk_loss
            loss = total_loss / float(max(1, n_tok))
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
    """Initialize `embed_re, embed_im` from PPMI-SVD pairs and set unigram log-prior `povm_log_prior`."""
    d_c = int(model.cfg.d_complex)
    v_eff = min(int(model.cfg.valid_vocab_size), ppmi_embeddings.size(0))
    if ppmi_embeddings.size(1) >= 2 * d_c:
        raw_re = ppmi_embeddings[:v_eff, 0 : 2 * d_c : 2].float()
        raw_im = ppmi_embeddings[:v_eff, 1 : 2 * d_c : 2].float()
    else:
        raw_re = ppmi_embeddings[:v_eff, :d_c].float()
        raw_im = torch.roll(raw_re, shifts=1, dims=-1)
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
) -> tuple[StiefelMuon, torch.optim.AdamW]:
    """Build `StiefelMuon` (for 2D complex linear matrices) and `AdamW` (for complex embeddings & gates)."""
    mup_scale = math.sqrt(256.0 / float(max(256, model.cfg.real_equiv_dim)))
    muon_params: list[nn.Parameter] = []
    embed_params: list[nn.Parameter] = []
    gate_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "embed_" in name:
            embed_params.append(param)
        elif param.ndim == 2 and min(param.shape) >= 32 and max(param.shape) <= 2048:
            muon_params.append(param)
        else:
            gate_params.append(param)

    muon_lr = base_muon_lr * mup_scale
    embed_lr = base_embed_lr * mup_scale
    gate_lr = base_gate_lr * mup_scale

    opt_muon = StiefelMuon(
        [{"params": muon_params, "lr": muon_lr, "base_lr": muon_lr}],
        lr=muon_lr,
        momentum=0.95,
        polar_method="polar_express",
        stiefel_penalty=0.02,
        retract_every=2,
        use_normuon=True,
        use_twin_rail=True,
    )
    opt_adam = torch.optim.AdamW(
        [
            {"params": embed_params, "lr": embed_lr, "base_lr": embed_lr, "weight_decay": 0.0},
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
    "DEFAULT_COMPLEX_VOCAB_CHUNK",
    "GeminiPackedComplexRoPEBlock",
    "GeminiPackedComplexRoPEConfig",
    "GeminiPackedComplexRoPEGPT",
    "build_gemini_complex_optimizers",
    "initialize_gemini_complex_gpt_with_ppmi_svd",
]
