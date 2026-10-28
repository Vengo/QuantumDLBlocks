"""Tied Token-Wavefunction & Operator-Ket Holographic Complex Born GPT (`Wave 40` / `EXPERIMENT_PLAN_055.md`).

When to Use This Building Block:
- Use `TiedHoloBornComplexRoPEGPT`, `TiedHoloBornConfig`, `TiedHoloBornChunkedNLLFn`,
  `eval_tied_holo_born_probs`, `compute_soft_manifold_and_isotropy_penalties`,
  `build_tied_holo_born_optimizers`, and `RareTokenImportanceWindowSampler` when training
  pure-complex register-density language models (`GeminiBornComplexPackedWithRoPE` family)
  across long token horizons (`50M-900M` FineWeb tokens) where vocabulary rank preservation
  (`R_emb >= 140`) and 100% softmax/exp-free quantum Born probability normalization are required.

Why It Was Built & Mathematical Provenance (`EXPERIMENT_PLAN_055.md` / `Wave 40`):
- Post-mortem of `Wave 39` (`w39_winner_d128c_H4_dh32_L6_900Mtok`, which plateaued at `5.37` Val CE
  as detector effective rank collapsed `243 -> 27`) identified four structural root causes that this
  flag-free module resolves simultaneously:
  1. **100% Softmax-Free Holographic Complex Born Readout (`eval_tied_holo_born_probs` & `TiedHoloBornChunkedNLLFn`)**:
     Eliminates the legacy Boltzmann `exp` / `log_softmax` vocabulary readout, the Real-part-only overlap
     discard of `Im(<e_v | Phi_t>_C)`, and the erroneous `1 / sqrt(2)` double-shrinkage that halved self-overlap
     to `0.50`. Replaces them with coherent interference between a unit reference wave `1` and the centered
     complex transition amplitude `alpha * <e_v | Phi_t>_C^cent`, raised to polynomial boson power `p` via
     `torch.pow(I_tv, boson_p)` in `float32` with zero `exp`, `softmax`, or `logsumexp` calls.
  2. **Tied Initial Token Wavefunctions `<e_v|` + Sequence-Side Operator Output `|Phi_t> = O_hat_t |psi_t>`**:
     Ties the normalized input token wavefunctions `|e_v> in C^{d_complex}` directly as the final measurement
     basis and applies the learned unitary readout operator `out_operator` + self-echo Gram-Schmidt projection
     on the `B * T` sequence states (`8,192` positions/step) instead of `V = 50,257` vocabulary rows (`6.13x`
     fewer operator GEMM rows per step, zero vocabulary-side activation checkpointing).
  3. **Smooth Differentiable Unit-Sphere, Complex-Unitary, and Parseval-Isotropy Penalties**:
     Replaces conflicting real-half `StiefelMuon` penalties (`W_re W_re^T = I`) and hard periodic retractions
     with smooth differentiable penalties over `valid_vocab_size = 50,257`: unit-sphere norm penalty
     `L_sphere`, square complex-unitary penalty `L_unitary` (`W W^H -> I`, excluding rectangular FFN beams),
     and Parseval frame isotropy penalty `L_iso` (`Sigma_E -> (1 / 2d) I_{2d}`).
  4. **Unbiased Rare-Token Importance Window Sampling (`RareTokenImportanceWindowSampler`)**:
     Counters severe FineWeb Zipf token starvation (where the bottom `50%` of the vocabulary receives only
     `3.44%` of updates) by sampling candidate windows with probability proportional to mean Laplace-smoothed
     token rarity `P_0(y_t)^{-alpha_rare}` and applying exact inverse-probability importance weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..layers.packed_complex_register_density_attn import (
    ComplexLinear,
    DEFAULT_COMPLEX_NORM_EPS,
    DEFAULT_COMPLEX_REGISTER_DIM,
    DEFAULT_P_BOSON_POWER,
    PackedComplexRegisterDensityAttention,
    complex_l2_normalize,
)
from ..optim.muon import StiefelMuon
from .gemini_packed_complex_rope_gpt import ComplexUnitaryInterferenceFFN

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_055.md (Wave 40 Constants Ledger)
# ==============================================================================
DEFAULT_HOLO_BOSON_P: float = 12.0
DEFAULT_HOLO_ALPHA: float = 0.85
DEFAULT_HOLO_EPS: float = 1e-6
DEFAULT_PRIOR_TEMP: float = 0.65
DEFAULT_SPHERE_PENALTY_WEIGHT: float = 2.0
DEFAULT_UNITARY_PENALTY_WEIGHT: float = 0.08
DEFAULT_ISO_PENALTY_WEIGHT: float = 0.50
DEFAULT_VOCAB_CHUNK_SIZE: int = 4096
DEFAULT_MUON_LR: float = 0.015
DEFAULT_EMBED_LR: float = 0.006
DEFAULT_GATE_LR: float = 0.003
DEFAULT_RARE_ALPHA: float = 0.28
DEFAULT_POOL_MULTIPLIER: int = 8
DEFAULT_POOL_CONTRAST_POWER: float = 6.0
DEFAULT_LAPLACE_EPS: float = 1.0
DEFAULT_PSI0_GATE_INIT: float = 0.20
DEFAULT_RESIDUAL_INIT: float = 0.35
DEFAULT_FFN_EXPANSION: int = 2
MIN_COMPLEX_MUON_DIM: int = 16
MAX_COMPLEX_MUON_DIM: int = 4096
PROB_LOG_CLAMP_MIN: float = 1e-12
PARTITION_CLAMP_MIN: float = 1e-30


@dataclass(frozen=True)
class TiedHoloBornConfig:
    """Frozen configuration for `TiedHoloBornComplexRoPEGPT` (`Wave 40` / `EXPERIMENT_PLAN_055.md`)."""

    vocab_size: int = 50304
    valid_vocab_size: int = 50257
    d_complex: int = 128
    n_registers: int = 4
    d_register: int = DEFAULT_COMPLEX_REGISTER_DIM
    num_layers: int = 6
    block_size: int = 512
    holo_boson_p: float = DEFAULT_HOLO_BOSON_P
    holo_alpha: float = DEFAULT_HOLO_ALPHA
    prior_temp: float = DEFAULT_PRIOR_TEMP
    sphere_penalty_weight: float = DEFAULT_SPHERE_PENALTY_WEIGHT
    unitary_penalty_weight: float = DEFAULT_UNITARY_PENALTY_WEIGHT
    iso_penalty_weight: float = DEFAULT_ISO_PENALTY_WEIGHT
    vocab_chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE
    checkpoint_blocks: bool = False

    def __post_init__(self) -> None:
        if self.n_registers * self.d_register != self.d_complex:
            if self.d_complex % self.d_register == 0:
                object.__setattr__(self, "n_registers", max(1, self.d_complex // self.d_register))
            elif self.d_complex % self.n_registers == 0:
                object.__setattr__(self, "d_register", max(8, self.d_complex // self.n_registers))
            if self.n_registers * self.d_register != self.d_complex:
                raise ValueError(
                    f"d_complex ({self.d_complex}) must equal "
                    f"n_registers ({self.n_registers}) * d_register ({self.d_register})"
                )

    @property
    def real_equiv_dim(self) -> int:
        """Return total real degrees of freedom per token (`2 * d_complex`)."""
        return 2 * int(self.d_complex)


def _init_haar_complex_unitary(mod: ComplexLinear) -> None:
    """Initialize a square `ComplexLinear` layer as an exact Haar complex unitary matrix `W W^H = I`."""
    if mod.in_features != mod.out_features:
        return
    d_c = int(mod.in_features)
    z_re = torch.randn(d_c, d_c, device=mod.weight_re.device, dtype=torch.float32)
    z_im = torch.randn(d_c, d_c, device=mod.weight_im.device, dtype=torch.float32)
    z_complex = torch.complex(z_re, z_im)
    q_mat, r_mat = torch.linalg.qr(z_complex)
    diag_r = torch.diagonal(r_mat)
    phase = diag_r / diag_r.abs().clamp_min(DEFAULT_HOLO_EPS)
    u_mat = q_mat * phase.unsqueeze(0)
    mod.weight_re.data.copy_(u_mat.real.to(dtype=mod.weight_re.dtype))
    mod.weight_im.data.copy_(u_mat.imag.to(dtype=mod.weight_im.dtype))


def eval_tied_energy_born_probs(
    phi_re: torch.Tensor,
    phi_im: torch.Tensor,
    e_re: torch.Tensor,
    e_im: torch.Tensor,
    prior_weights: torch.Tensor,
    boson_p: float = DEFAULT_HOLO_BOSON_P,
) -> torch.Tensor:
    """Compute pure `p`-boson Born energy probabilities `P(v|t) ~ w_v (|<e_v | Phi_t>_C|^2 + eps)^p` (`<= 8` args).

    Uses 100% polynomial `torch.pow` in `float32` with zero `exp`, `softmax`, or `logsumexp` calls.
    """
    phi_cat = torch.cat([phi_re.float(), phi_im.float()], dim=-1)
    e_cat = torch.cat([e_re.float(), e_im.float()], dim=-1)
    e_skew_cat = torch.cat([-e_im.float(), e_re.float()], dim=-1)

    c_re = torch.matmul(phi_cat, e_cat.t())
    c_im = -torch.matmul(phi_cat, e_skew_cat.t())
    energy_tv = c_re.square() + c_im.square() + DEFAULT_HOLO_EPS
    w_prior = prior_weights.float().view(*([1] * (energy_tv.ndim - 1)), -1)
    p_tilde = w_prior * torch.pow(energy_tv, float(boson_p))
    z_t = p_tilde.sum(dim=-1, keepdim=True).clamp_min(PARTITION_CLAMP_MIN)
    return p_tilde / z_t


def eval_tied_holo_born_probs(
    phi_re: torch.Tensor,
    phi_im: torch.Tensor,
    e_re: torch.Tensor,
    e_im: torch.Tensor,
    prior_weights: torch.Tensor,
    boson_p: float = DEFAULT_HOLO_BOSON_P,
    holo_alpha: float = DEFAULT_HOLO_ALPHA,
) -> torch.Tensor:
    """Compute 100% softmax-free Holographic Reference-Wave Complex Born probabilities (`<= 8` args).

    Mathematical Formulation (`EXPERIMENT_PLAN_055.md` Section 3.1.3):
    - `c_{t,v}^{re} = Phi_{re,t}^T e_{re,v} + Phi_{im,t}^T e_{im,v}`
    - `c_{t,v}^{im} = Phi_{re,t}^T e_{im,v} - Phi_{im,t}^T e_{re,v}`
    - `c_{t,v}^{re,cent} = c_{t,v}^{re} - (1/V) sum_{v'} c_{t,v'}^{re}`
    - `I_{t,v} = (1 + alpha * c_{t,v}^{re,cent})^2 + (alpha * c_{t,v}^{im})^2 + eps_0`
    - `P_{HoloBorn}(v | t) = w_{prior,v} * (I_{t,v})^p / sum_{v'} w_{prior,v'} * (I_{t,v'})^p`
    Evaluated in `float32` via `torch.pow(I_tv, boson_p)` with zero `exp`, `softmax`, or `logsumexp`.
    """
    phi_cat = torch.cat([phi_re.float(), phi_im.float()], dim=-1)
    e_cat = torch.cat([e_re.float(), e_im.float()], dim=-1)
    e_skew_cat = torch.cat([-e_im.float(), e_re.float()], dim=-1)

    c_re = torch.matmul(phi_cat, e_cat.t())
    c_im = -torch.matmul(phi_cat, e_skew_cat.t())
    c_re_cent = c_re - c_re.mean(dim=-1, keepdim=True)

    alpha = float(holo_alpha)
    i_tv = (1.0 + alpha * c_re_cent).square() + (alpha * c_im).square() + DEFAULT_HOLO_EPS
    w_prior = prior_weights.float().view(*([1] * (i_tv.ndim - 1)), -1)
    p_tilde = w_prior * torch.pow(i_tv, float(boson_p))
    z_t = p_tilde.sum(dim=-1, keepdim=True).clamp_min(PARTITION_CLAMP_MIN)
    return p_tilde / z_t


class TiedHoloBornChunkedNLLFn(torch.autograd.Function):
    """Memory-efficient vocabulary-chunked Holographic Complex Born NLL with exact centering pullback.

    Chunks over the vocabulary `V` (`vocab_chunk_size = 4096` by default) so full `(N, V)` float32
    intensity and probability tables never need to be materialized simultaneously in GPU HBM.
    """

    vocab_chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE

    @staticmethod
    def forward(
        ctx: Any,
        phi_cat: torch.Tensor,
        e_cat: torch.Tensor,
        targets: torch.Tensor,
        prior_weights: torch.Tensor,
        boson_p: float,
        holo_alpha: float,
        token_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute chunked Holographic Born NLL `sum_t w_tilde_t * (ln Z_t - ln P_tilde_{t, y_t})` (`<= 8` args)."""
        n_tok = phi_cat.size(0)
        d_real = phi_cat.size(1)
        d_c = d_real // 2
        v_eff = e_cat.size(0)
        v_chunk = max(1, int(TiedHoloBornChunkedNLLFn.vocab_chunk_size))

        gemm_dtype = phi_cat.dtype
        phi_w = phi_cat.contiguous()
        e_w = e_cat.to(dtype=gemm_dtype).contiguous()
        targets_flat = targets.reshape(-1).long().clamp(0, v_eff - 1)
        prior_f = prior_weights.float().contiguous()

        if token_weights is None:
            w_norm = torch.full(
                (n_tok,), 1.0 / float(max(1, n_tok)), device=phi_cat.device, dtype=torch.float32
            )
        else:
            tw_f = token_weights.reshape(-1).float()
            w_norm = tw_f / tw_f.sum().clamp_min(PROB_LOG_CLAMP_MIN)

        # Precompute exact global vocabulary mean ket e_bar and per-token mean real overlap mu_t^{re}
        e_mean = e_cat.float().mean(dim=0)
        mu_re = torch.mv(phi_cat.float(), e_mean)

        z_t = torch.zeros((n_tok,), device=phi_cat.device, dtype=torch.float32)
        target_p_tilde = torch.ones((n_tok,), device=phi_cat.device, dtype=torch.float32)
        alpha = float(holo_alpha)
        p_val = float(boson_p)

        for v0 in range(0, v_eff, v_chunk):
            v1 = min(v_eff, v0 + v_chunk)
            e_c = e_w[v0:v1]
            e_skew_c = torch.cat([-e_c[:, d_c:], e_c[:, :d_c]], dim=-1)

            c_re = torch.matmul(phi_w, e_c.t()).float()
            c_im = -torch.matmul(phi_w, e_skew_c.t()).float()
            c_re_cent = c_re - mu_re.unsqueeze(1)

            i_tv = (1.0 + alpha * c_re_cent).square() + (alpha * c_im).square() + DEFAULT_HOLO_EPS
            p_tilde_c = prior_f[v0:v1].unsqueeze(0) * torch.pow(i_tv, p_val)
            z_t.add_(p_tilde_c.sum(dim=-1))

            in_chunk = (targets_flat >= v0) & (targets_flat < v1)
            if in_chunk.any():
                rel_idx = (targets_flat - v0).clamp(0, (v1 - v0) - 1).unsqueeze(1)
                gathered = p_tilde_c.gather(1, rel_idx).squeeze(1)
                target_p_tilde = torch.where(in_chunk, gathered, target_p_tilde)

        z_clamped = z_t.clamp_min(PARTITION_CLAMP_MIN)
        target_clamped = target_p_tilde.clamp_min(PARTITION_CLAMP_MIN)
        nll_t = torch.log(z_clamped) - torch.log(target_clamped)
        loss = (w_norm * nll_t).sum()

        ctx.save_for_backward(phi_w, e_w, targets_flat, prior_f, w_norm, e_mean, mu_re, z_clamped)
        ctx.boson_p = p_val
        ctx.holo_alpha = alpha
        ctx.v_chunk = v_chunk
        ctx.phi_dtype = phi_cat.dtype
        ctx.e_dtype = e_cat.dtype
        return loss

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None, None, None]:
        """Compute exact chunked gradients wrt `phi_cat` and `e_cat` including `mu_t^{re}` centering pullback."""
        phi_w, e_w, targets_flat, prior_f, w_norm, e_mean, mu_re, z_clamped = ctx.saved_tensors
        p_val = float(ctx.boson_p)
        alpha = float(ctx.holo_alpha)
        v_chunk = int(ctx.v_chunk)

        n_tok, d_real = phi_w.shape
        d_c = d_real // 2
        v_eff = e_w.size(0)
        gemm_dtype = phi_w.dtype

        phi_skew_w = torch.cat([-phi_w[:, d_c:], phi_w[:, :d_c]], dim=-1).contiguous()
        d_phi_raw = torch.zeros((n_tok, d_real), device=phi_w.device, dtype=torch.float32)
        d_e_raw = torch.zeros((v_eff, d_real), device=e_w.device, dtype=torch.float32)
        s_t = torch.zeros((n_tok, 1), device=phi_w.device, dtype=torch.float32)

        scale_t = (grad_out.float() * w_norm).unsqueeze(1)
        inv_z = (1.0 / z_clamped).unsqueeze(1)
        two_p_alpha = 2.0 * p_val * alpha
        two_p_alpha_sq = 2.0 * p_val * alpha * alpha

        for v0 in range(0, v_eff, v_chunk):
            v1 = min(v_eff, v0 + v_chunk)
            e_c = e_w[v0:v1]
            e_skew_c = torch.cat([-e_c[:, d_c:], e_c[:, :d_c]], dim=-1)

            c_re = torch.matmul(phi_w, e_c.t()).float()
            c_im = -torch.matmul(phi_w, e_skew_c.t()).float()
            c_re_cent = c_re - mu_re.unsqueeze(1)

            one_plus = 1.0 + alpha * c_re_cent
            i_tv = one_plus.square() + (alpha * c_im).square() + DEFAULT_HOLO_EPS
            p_tilde_c = prior_f[v0:v1].unsqueeze(0) * torch.pow(i_tv, p_val)
            probs_c = p_tilde_c * inv_z

            in_chunk = (targets_flat >= v0) & (targets_flat < v1)
            if in_chunk.any():
                rel_idx = (targets_flat - v0).clamp(0, (v1 - v0) - 1).unsqueeze(1)
                probs_c.scatter_add_(
                    1,
                    rel_idx,
                    (-in_chunk.float()).unsqueeze(1),
                )

            delta_c = scale_t * probs_c
            inv_i = 1.0 / i_tv
            g_re_cent = delta_c * (two_p_alpha * one_plus * inv_i)
            g_im = delta_c * (two_p_alpha_sq * c_im * inv_i)

            s_t.add_(g_re_cent.sum(dim=1, keepdim=True))

            g_re_w = g_re_cent.to(dtype=gemm_dtype)
            g_im_w = g_im.to(dtype=gemm_dtype)
            d_phi_step = torch.matmul(g_re_w, e_c).float() - torch.matmul(g_im_w, e_skew_c).float()
            d_phi_raw.add_(d_phi_step)

            d_e_step = torch.matmul(g_re_w.t(), phi_w).float() + torch.matmul(g_im_w.t(), phi_skew_w).float()
            d_e_raw[v0:v1] = d_e_step

        # Exact centering pullback for mu_t^{re} = (1/V) sum_{v'} c_{t,v'}^{re} = phi_cat[t]^T e_bar
        d_phi_cat = d_phi_raw - s_t * e_mean.unsqueeze(0)
        mean_s_phi = torch.matmul(s_t.t(), phi_w.float()) / float(v_eff)
        d_e_cat = d_e_raw - mean_s_phi

        return (
            d_phi_cat.to(dtype=ctx.phi_dtype),
            d_e_cat.to(dtype=ctx.e_dtype),
            None,
            None,
            None,
            None,
            None,
        )


class TiedHoloBornComplexBlock(nn.Module):
    """Single complex register block (`PackedComplexRegisterDensityAttention` + `ComplexUnitaryInterferenceFFN`)."""

    def __init__(self, cfg: TiedHoloBornConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = PackedComplexRegisterDensityAttention(
            d_model=cfg.d_complex,
            n_registers=cfg.n_registers,
            d_register=cfg.d_register,
            p_boson=DEFAULT_P_BOSON_POWER,
            probe_mode="v_krylov",
            normalize_krylov_ket=True,
            value_residual_mixing=(layer_idx > 0),
        )
        self.ffn = ComplexUnitaryInterferenceFFN(cfg.d_complex, expansion=DEFAULT_FFN_EXPANSION)
        init_scale = DEFAULT_RESIDUAL_INIT / math.sqrt(float(max(1, layer_idx + 1)))
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
        """Run complex register density attention and unitary two-beam interference FFN (`<= 8` args)."""
        attn_re, attn_im, v_raw_re, v_raw_im = self.attn.forward_complex(
            psi_re, psi_im, v0_re=v0_re, v0_im=v0_im, eps=eps
        )
        attn_re, attn_im = complex_l2_normalize(attn_re, attn_im, dim=-1, eps=eps)
        g_a = torch.tanh(self.attn_gate).to(dtype=attn_re.dtype)
        p_re = psi_re.to(dtype=attn_re.dtype)
        p_im = psi_im.to(dtype=attn_im.dtype)
        mid_re, mid_im = complex_l2_normalize(
            p_re + g_a * attn_re, p_im + g_a * attn_im, dim=-1, eps=eps
        )

        ffn_re, ffn_im = self.ffn(mid_re, mid_im, eps=eps)
        g_f = torch.tanh(self.ffn_gate).to(dtype=ffn_re.dtype)
        out_re, out_im = complex_l2_normalize(
            mid_re + g_f * ffn_re, mid_im + g_f * ffn_im, dim=-1, eps=eps
        )
        return out_re, out_im, v_raw_re, v_raw_im


class TiedHoloBornComplexRoPEGPT(nn.Module):
    """Flag-Free Tied Token-Wavefunction & Operator-Ket Holographic Complex Born GPT (`Wave 40`)."""

    def __init__(self, cfg: TiedHoloBornConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_re = nn.Parameter(torch.empty(cfg.vocab_size, cfg.d_complex))
        self.embed_im = nn.Parameter(torch.empty(cfg.vocab_size, cfg.d_complex))
        self.register_buffer("prior_weights", torch.ones(cfg.vocab_size, dtype=torch.float32))

        self.blocks = nn.ModuleList([TiedHoloBornComplexBlock(cfg, i) for i in range(cfg.num_layers)])
        self.out_operator = ComplexLinear(cfg.d_complex, cfg.d_complex)
        self.psi0_readout_gate = nn.Parameter(torch.tensor(DEFAULT_PSI0_GATE_INIT, dtype=torch.float32))
        self._training_progress: float = 0.0

        self.reset_isotropic_unit_sphere_embeddings()
        self._init_square_unitary_operators()

    @torch.no_grad()
    def _init_square_unitary_operators(self) -> None:
        """Initialize all square `ComplexLinear` operators as exact Haar complex unitaries `W W^H = I`."""
        for mod in self.modules():
            if isinstance(mod, ComplexLinear) and mod.in_features == mod.out_features:
                _init_haar_complex_unitary(mod)

    @torch.no_grad()
    def reset_isotropic_unit_sphere_embeddings(self, seed: int = 1337) -> None:
        """Initialize active vocabulary rows isotropically on `S^{2*d_complex - 1}` and zero padding rows."""
        v_eff = int(self.cfg.valid_vocab_size)
        d_c = int(self.cfg.d_complex)
        gen = torch.Generator(device=self.embed_re.device)
        gen.manual_seed(int(seed))
        raw_re = torch.randn((v_eff, d_c), generator=gen, device=self.embed_re.device, dtype=torch.float32)
        raw_im = torch.randn((v_eff, d_c), generator=gen, device=self.embed_im.device, dtype=torch.float32)
        norm_sq = raw_re.square().sum(dim=-1, keepdim=True) + raw_im.square().sum(dim=-1, keepdim=True)
        inv_norm = torch.rsqrt(norm_sq + DEFAULT_HOLO_EPS)
        self.embed_re.data.zero_()
        self.embed_im.data.zero_()
        self.embed_re.data[:v_eff].copy_((raw_re * inv_norm).to(dtype=self.embed_re.dtype))
        self.embed_im.data[:v_eff].copy_((raw_im * inv_norm).to(dtype=self.embed_im.dtype))

    def set_training_progress(self, progress: float) -> None:
        """Record normalized training progress `tau in [0.0, 1.0]`."""
        self._training_progress = float(min(max(progress, 0.0), 1.0))

    @torch.no_grad()
    def initialize_uniform_or_empirical_prior(self, prior_tokens: torch.Tensor | None = None) -> None:
        """Initialize polynomial POVM measure weights `w_{prior,v} = (P_0(v) / max P_0)^{tau_prior}` (`<= 8` args)."""
        v_eff = int(self.cfg.valid_vocab_size)
        self.prior_weights.fill_(1.0)
        if prior_tokens is None or prior_tokens.numel() == 0:
            return
        flat_tok = prior_tokens.reshape(-1).long().clamp(0, v_eff - 1)
        counts = torch.bincount(flat_tok, minlength=v_eff).float() + DEFAULT_LAPLACE_EPS
        p0 = counts / counts.sum().clamp_min(DEFAULT_HOLO_EPS)
        w_prior = torch.pow(p0 / p0.max().clamp_min(DEFAULT_HOLO_EPS), float(self.cfg.prior_temp))
        self.prior_weights[:v_eff].copy_(w_prior.to(device=self.prior_weights.device, dtype=torch.float32))
        if self.cfg.vocab_size > v_eff:
            self.prior_weights[v_eff:].zero_()

    @torch.no_grad()
    def initialize_isotropic_haar(
        self,
        prior_tokens: torch.Tensor | None = None,
        seed: int = 1337,
    ) -> dict[str, float]:
        """Initialize `embed_re[:V], embed_im[:V]` isotropically on `S^{2*d_complex-1}` (`<= 8` args)."""
        self.reset_isotropic_unit_sphere_embeddings(seed=seed)
        self._init_square_unitary_operators()
        self.initialize_uniform_or_empirical_prior(prior_tokens)
        return {
            "sharp_eff_rank": self.compute_sharp_eff_rank(),
        }

    @torch.no_grad()
    def initialize_with_ppmi_svd(
        self,
        ppmi_embeddings: torch.Tensor,
        prior_tokens: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Initialize `embed_re[:V], embed_im[:V]` from PPMI-SVD pairs/harmonics and set `prior_weights` (`<= 8` args)."""
        d_c = int(self.cfg.d_complex)
        v_eff = min(int(self.cfg.valid_vocab_size), int(ppmi_embeddings.size(0)))
        if ppmi_embeddings.size(1) >= 2 * d_c:
            raw_re = ppmi_embeddings[:v_eff, 0 : 2 * d_c : 2].float()
            raw_im = ppmi_embeddings[:v_eff, 1 : 2 * d_c : 2].float()
        else:
            base_re = ppmi_embeddings[:v_eff, 0::2].float()
            base_im = ppmi_embeddings[:v_eff, 1::2].float()
            b_dim = base_re.size(1)
            rms_target = base_re.square().mean().sqrt().clamp_min(DEFAULT_HOLO_EPS)
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
                q_rms = quad_re.square().mean().sqrt().clamp_min(DEFAULT_HOLO_EPS)
                quad_re = quad_re * (rms_target / q_rms)
                quad_im = quad_im * (rms_target / q_rms)
                re_tiles.append(0.75 * rot_re + 0.25 * quad_re)
                im_tiles.append(0.75 * rot_im + 0.25 * quad_im)
            raw_re = torch.cat(re_tiles, dim=-1)[:, :d_c].contiguous()
            raw_im = torch.cat(im_tiles, dim=-1)[:, :d_c].contiguous()

        norm_sq = raw_re.square().sum(dim=-1, keepdim=True) + raw_im.square().sum(dim=-1, keepdim=True)
        inv_norm = torch.rsqrt(norm_sq + DEFAULT_HOLO_EPS)
        self.embed_re.data.zero_()
        self.embed_im.data.zero_()
        self.embed_re.data[:v_eff].copy_((raw_re * inv_norm).to(device=self.embed_re.device, dtype=self.embed_re.dtype))
        self.embed_im.data[:v_eff].copy_((raw_im * inv_norm).to(device=self.embed_im.device, dtype=self.embed_im.dtype))
        self._init_square_unitary_operators()
        self.initialize_uniform_or_empirical_prior(prior_tokens)
        return {
            "sharp_eff_rank": self.compute_sharp_eff_rank(),
        }

    def get_tied_vocab_wavefunctions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unit-norm complex vocabulary wavefunctions `(e_re, e_im)` over `0 .. valid_vocab_size - 1`."""
        v_eff = int(self.cfg.valid_vocab_size)
        return complex_l2_normalize(self.embed_re[:v_eff], self.embed_im[:v_eff], dim=-1)

    def apply_readout_operator(
        self,
        psi_re: torch.Tensor,
        psi_im: torch.Tensor,
        psi0_re: torch.Tensor,
        psi0_im: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply conjugate skip, sequence-side `out_operator`, and self-echo orthogonalization on `(B, T, d_complex)`."""
        g0 = torch.tanh(self.psi0_readout_gate).to(dtype=psi_re.dtype)
        p0_re = psi0_re.to(dtype=psi_re.dtype)
        p0_im = psi0_im.to(dtype=psi_im.dtype)
        mixed_re, mixed_im = complex_l2_normalize(
            psi_re + g0 * p0_re,
            psi_im - g0 * p0_im,
            dim=-1,
        )
        op_re, op_im = self.out_operator(mixed_re, mixed_im)
        ov_re = (
            p0_re.float() * op_re.float() + p0_im.float() * op_im.float()
        ).sum(dim=-1, keepdim=True).to(dtype=op_re.dtype)
        orth_re = op_re - ov_re * p0_re
        orth_im = op_im - ov_re * p0_im
        return complex_l2_normalize(orth_re, orth_im, dim=-1)

    @torch.no_grad()
    def compute_sharp_eff_rank(self) -> float:
        """Compute participation ratio `R_emb` of tied normalized vocabulary wavefunctions `[e_re, e_im]`."""
        v_eff = int(self.cfg.valid_vocab_size)
        e_re_raw = self.embed_re[:v_eff].detach().float()
        e_im_raw = self.embed_im[:v_eff].detach().float()
        norm_sq = e_re_raw.square().sum(dim=-1, keepdim=True) + e_im_raw.square().sum(dim=-1, keepdim=True)
        inv_n = torch.rsqrt(norm_sq + DEFAULT_HOLO_EPS)
        e_cat = torch.cat([e_re_raw * inv_n, e_im_raw * inv_n], dim=-1)
        e_cent = e_cat - e_cat.mean(dim=0, keepdim=True)
        gram = e_cent.t() @ e_cent
        tr = torch.diagonal(gram).sum()
        return float(((tr * tr) / (gram * gram).sum().clamp_min(PROB_LOG_CLAMP_MIN)).item())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_logits: bool | None = None,
        token_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Run `L` complex register blocks, sequence-side `out_operator`, and Tied Holographic Born readout."""
        b_sz, seq_t = idx.shape
        d_c = int(self.cfg.d_complex)
        h_num, dh = int(self.cfg.n_registers), int(self.cfg.d_register)
        v_eff = int(self.cfg.valid_vocab_size)

        idx_clamped = idx.clamp(min=0, max=v_eff - 1)
        psi0_re = self.embed_re.index_select(0, idx_clamped.reshape(-1)).view(b_sz, seq_t, d_c)
        psi0_im = self.embed_im.index_select(0, idx_clamped.reshape(-1)).view(b_sz, seq_t, d_c)
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

        phi_re, phi_im = self.apply_readout_operator(psi_re, psi_im, psi0_re, psi0_im)
        e_re, e_im = self.get_tied_vocab_wavefunctions()
        prior_w = self.prior_weights[:v_eff]
        want_logits = (targets is None) if return_logits is None else bool(return_logits)

        if targets is not None and not want_logits:
            phi_cat = torch.cat([phi_re.reshape(-1, d_c), phi_im.reshape(-1, d_c)], dim=-1)
            e_cat = torch.cat([e_re, e_im], dim=-1)
            valid_targets = targets.clamp(min=0, max=v_eff - 1).reshape(-1)
            TiedHoloBornChunkedNLLFn.vocab_chunk_size = int(self.cfg.vocab_chunk_size)
            loss = TiedHoloBornChunkedNLLFn.apply(
                phi_cat,
                e_cat,
                valid_targets,
                prior_w,
                float(self.cfg.holo_boson_p),
                float(self.cfg.holo_alpha),
                token_weights,
            )
            out_logits = loss if return_logits is None else None
            return out_logits, loss

        probs = eval_tied_holo_born_probs(
            phi_re,
            phi_im,
            e_re,
            e_im,
            prior_w,
            boson_p=float(self.cfg.holo_boson_p),
            holo_alpha=float(self.cfg.holo_alpha),
        )
        log_probs = torch.log(probs.clamp_min(PROB_LOG_CLAMP_MIN))

        loss = None
        if targets is not None:
            valid_targets = targets.clamp(min=0, max=v_eff - 1)
            token_nll = -log_probs.gather(-1, valid_targets.unsqueeze(-1)).squeeze(-1).reshape(-1)
            if token_weights is None:
                loss = token_nll.mean()
            else:
                tw_f = token_weights.reshape(-1).float()
                w_norm = tw_f / tw_f.sum().clamp_min(PROB_LOG_CLAMP_MIN)
                loss = (w_norm * token_nll).sum()

        out_logits = log_probs if want_logits else (loss if (targets is not None and return_logits is None) else None)
        return out_logits, loss


def compute_soft_manifold_and_isotropy_penalties(
    model: TiedHoloBornComplexRoPEGPT,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute smooth Unit-Sphere, Square Complex-Unitary, and Parseval-Isotropy penalties (`<= 8` args).

    Slices strictly over `0 .. valid_vocab_size - 1` (`50,257` rows) so unused padding rows never
    distort embedding norms or frame covariance, and penalizes only square `ComplexLinear` matrices
    (`in_features == out_features`), leaving rectangular FFN layers unconstrained.
    """
    v_eff = int(model.cfg.valid_vocab_size)
    e_re_raw = model.embed_re[:v_eff].float()
    e_im_raw = model.embed_im[:v_eff].float()
    e_norm_sq = e_re_raw.square().sum(dim=-1) + e_im_raw.square().sum(dim=-1)
    sphere_pen = (e_norm_sq - 1.0).square().mean()

    with torch.no_grad():
        e_norms = torch.sqrt(e_norm_sq.detach().clamp_min(PROB_LOG_CLAMP_MIN))
        embed_norm_median = float(e_norms.median().item())
        embed_norm_max = float(e_norms.max().item())

    unitary_terms: list[torch.Tensor] = []
    max_unitary_err = 0.0
    for mod in model.modules():
        if isinstance(mod, ComplexLinear) and mod.in_features == mod.out_features:
            w_re = mod.weight_re.float()
            w_im = mod.weight_im.float()
            d_c = w_re.size(0)
            eye = torch.eye(d_c, device=w_re.device, dtype=torch.float32)
            gram_re = w_re @ w_re.t() + w_im @ w_im.t() - eye
            gram_im = w_im @ w_re.t() - w_re @ w_im.t()
            u_pen = (gram_re.square().sum() + gram_im.square().sum()) / float(d_c)
            unitary_terms.append(u_pen)
            err_w = float(torch.sqrt(u_pen.detach().clamp_min(0.0)).item())
            max_unitary_err = max(max_unitary_err, err_w)

    if unitary_terms:
        unitary_pen = torch.stack(unitary_terms).mean()
        unitary_sum_full = torch.stack(unitary_terms).sum() * float(model.cfg.d_complex)
    else:
        unitary_pen = torch.zeros((), device=e_re_raw.device, dtype=torch.float32)
        unitary_sum_full = unitary_pen

    inv_norm = torch.rsqrt(e_norm_sq.unsqueeze(-1) + DEFAULT_HOLO_EPS)
    e_cat = torch.cat([e_re_raw * inv_norm, e_im_raw * inv_norm], dim=-1)
    e_cent = e_cat - e_cat.mean(dim=0, keepdim=True)
    cov = (e_cent.t() @ e_cent) / float(v_eff)
    d_real = 2 * int(model.cfg.d_complex)
    target_cov = torch.eye(d_real, device=cov.device, dtype=torch.float32) / float(d_real)
    iso_pen = float(d_real) * (cov - target_cov).square().sum()

    total_penalty = (
        float(model.cfg.sphere_penalty_weight) * sphere_pen
        + float(model.cfg.unitary_penalty_weight) * unitary_sum_full
        + float(model.cfg.iso_penalty_weight) * iso_pen
    )
    stats = {
        "sphere_pen": float(sphere_pen.detach().item()),
        "unitary_pen": float(unitary_pen.detach().item()),
        "iso_pen": float(iso_pen.detach().item()),
        "total_pen": float(total_penalty.detach().item()),
        "embed_norm_median": embed_norm_median,
        "embed_norm_max": embed_norm_max,
        "complex_unitary_err": max_unitary_err,
    }
    return total_penalty, stats


def build_tied_holo_born_optimizers(
    model: TiedHoloBornComplexRoPEGPT,
    base_muon_lr: float = DEFAULT_MUON_LR,
    base_embed_lr: float = DEFAULT_EMBED_LR,
    base_gate_lr: float = DEFAULT_GATE_LR,
    weight_decay: float = 0.0,
) -> tuple[StiefelMuon, torch.optim.AdamW]:
    """Build `StiefelMuon` (`stiefel_penalty=0.0`) for 2D `ComplexLinear` matrices and `AdamW` for embeddings/gates."""
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

    muon_params: list[nn.Parameter] = []
    embed_params: list[nn.Parameter] = []
    gate_params: list[nn.Parameter] = []

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

    muon_lr = float(base_muon_lr) * muon_scale
    embed_lr = float(base_embed_lr) * embed_scale
    gate_lr = float(base_gate_lr) * gate_scale

    opt_muon = StiefelMuon(
        [{"params": muon_params, "lr": muon_lr, "base_lr": muon_lr}],
        lr=muon_lr,
        momentum=0.95,
        polar_method="polar_express",
        stiefel_penalty=0.0,
        retract_every=1,
        use_normuon=True,
        use_twin_rail=True,
    )
    opt_adam = torch.optim.AdamW(
        [
            {"params": embed_params, "lr": embed_lr, "base_lr": embed_lr, "weight_decay": 0.0},
            {"params": gate_params, "lr": gate_lr, "base_lr": gate_lr, "weight_decay": float(weight_decay)},
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    return opt_muon, opt_adam


class RareTokenImportanceWindowSampler:
    """Unbiased Rare-Token Importance Window Sampler (`Wave 40` / `EXPERIMENT_PLAN_055.md` Section 3.1.7).

    Computes Laplace-smoothed unigram probabilities `P_0(v) = (count(v) + eps_laplace) / sum(count + eps_laplace)`
    so unseen tokens (`count(v) == 0`) have strictly positive probability and finite rarity `r(v) = P_0(v)^{-alpha_rare}`.
    Candidate sequence windows are sampled with probability `q_i = S_i / sum_j S_j` and paired with exact
    inverse-probability importance weights `w_i = 1 / (M * q_i)`.
    """

    def __init__(
        self,
        reference_tokens: np.ndarray | torch.Tensor,
        valid_vocab_size: int = 50257,
        alpha_rare: float = DEFAULT_RARE_ALPHA,
        pool_multiplier: int = DEFAULT_POOL_MULTIPLIER,
        laplace_eps: float = DEFAULT_LAPLACE_EPS,
    ) -> None:
        self.valid_vocab_size = int(valid_vocab_size)
        self.alpha_rare = float(alpha_rare)
        self.pool_multiplier = max(1, int(pool_multiplier))
        self.laplace_eps = float(laplace_eps)

        if isinstance(reference_tokens, np.ndarray):
            ref_t = torch.from_numpy(np.asarray(reference_tokens, dtype=np.int64))
        else:
            ref_t = reference_tokens.detach().cpu().to(torch.int64)
        flat_tok = ref_t.reshape(-1).clamp(0, self.valid_vocab_size - 1)
        counts = torch.bincount(flat_tok, minlength=self.valid_vocab_size).to(torch.float64)
        self.token_counts = counts.to(torch.int64)

        smoothed = counts + float(self.laplace_eps)
        p0 = smoothed / smoothed.sum().clamp_min(DEFAULT_HOLO_EPS)
        self.unigram_probs = p0.to(torch.float32)

        sorted_idx = torch.argsort(counts, stable=True)
        bottom_half = self.valid_vocab_size - (self.valid_vocab_size // 2)
        rare_mask = torch.zeros((self.valid_vocab_size,), dtype=torch.bool)
        rare_mask[sorted_idx[:bottom_half]] = True
        self.rare_bottom50_mask = rare_mask

        if self.alpha_rare == 0.0:
            self.token_rarity = torch.ones((self.valid_vocab_size,), dtype=torch.float32)
            self._tail_rarity = torch.ones((self.valid_vocab_size,), dtype=torch.float32)
        else:
            raw_r = torch.pow(p0, -self.alpha_rare).to(torch.float32)
            self.token_rarity = raw_r
            med_r = float(raw_r[sorted_idx[bottom_half]].item())
            tail_r = torch.clamp_min(raw_r - med_r, 0.0) * rare_mask.float()
            self._tail_rarity = tail_r + DEFAULT_HOLO_EPS
        self._device_rarity_cache: dict[torch.device, torch.Tensor] = {}

    def _get_device_rarity(self, device: torch.device) -> torch.Tensor:
        """Return `_tail_rarity` cached on `device`."""
        cached = self._device_rarity_cache.get(device)
        if cached is None or cached.device != device:
            cached = self._tail_rarity.to(device=device, non_blocking=True)
            self._device_rarity_cache[device] = cached
        return cached

    def compute_pool_distribution(
        self,
        cand_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(q_probs, unnorm_inv_weights)` over a candidate target pool `cand_y` of shape `(M, T)`."""
        m_pool = cand_y.size(0)
        if self.alpha_rare == 0.0:
            q_probs = torch.full((m_pool,), 1.0 / float(max(1, m_pool)), device=cand_y.device, dtype=torch.float32)
            unnorm_inv_weights = torch.ones((m_pool,), device=cand_y.device, dtype=torch.float32)
            return q_probs, unnorm_inv_weights

        rarity_dev = self._get_device_rarity(cand_y.device)
        y_clamped = cand_y.long().clamp(0, self.valid_vocab_size - 1)
        win_tail = rarity_dev[y_clamped].float().mean(dim=-1).clamp_min(DEFAULT_HOLO_EPS)
        rel_tail = win_tail / win_tail.mean().clamp_min(DEFAULT_HOLO_EPS)
        scores = torch.pow(rel_tail, DEFAULT_POOL_CONTRAST_POWER).clamp_min(DEFAULT_HOLO_EPS)
        q_probs = scores / scores.sum().clamp_min(DEFAULT_HOLO_EPS)
        unnorm_inv_weights = 1.0 / (float(m_pool) * q_probs.clamp_min(PARTITION_CLAMP_MIN))
        return q_probs, unnorm_inv_weights

    def sample_batch(
        self,
        token_buffer: torch.Tensor,
        batch_size: int,
        seq_len: int,
        step_seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample `(x_batch, y_batch, token_weights)` of shapes `(B, T), (B, T), (B, T)` (`<= 8` args)."""
        dev = token_buffer.device
        n_buf = int(token_buffer.numel())
        b_sz = int(batch_size)
        t_len = int(seq_len)
        m_pool = self.pool_multiplier * b_sz
        max_start = max(1, n_buf - t_len - 1)

        gen = torch.Generator(device=dev)
        gen.manual_seed(int(step_seed))
        starts = torch.randint(0, max_start, (m_pool,), generator=gen, device=dev)
        offsets = starts.unsqueeze(1) + torch.arange(t_len + 1, device=dev).unsqueeze(0)

        raw_windows = token_buffer[offsets]
        if token_buffer.dtype == torch.int16:
            cand_tokens = (raw_windows.to(torch.int64) & 0xFFFF).clamp(0, self.valid_vocab_size - 1)
        else:
            cand_tokens = raw_windows.to(torch.int64).clamp(0, self.valid_vocab_size - 1)

        cand_x = cand_tokens[:, :-1]
        cand_y = cand_tokens[:, 1:]
        q_probs, unnorm_inv_w = self.compute_pool_distribution(cand_y)

        chosen_idx = torch.multinomial(q_probs, num_samples=b_sz, replacement=True, generator=gen)
        x_batch = cand_x[chosen_idx].contiguous()
        y_batch = cand_y[chosen_idx].contiguous()

        chosen_w = unnorm_inv_w[chosen_idx]
        norm_w = chosen_w / chosen_w.mean().clamp_min(PROB_LOG_CLAMP_MIN)
        token_weights = norm_w.unsqueeze(1).expand(b_sz, t_len).contiguous()
        return x_batch, y_batch, token_weights


__all__ = [
    "DEFAULT_EMBED_LR",
    "DEFAULT_FFN_EXPANSION",
    "DEFAULT_GATE_LR",
    "DEFAULT_HOLO_ALPHA",
    "DEFAULT_HOLO_BOSON_P",
    "DEFAULT_HOLO_EPS",
    "DEFAULT_ISO_PENALTY_WEIGHT",
    "DEFAULT_LAPLACE_EPS",
    "DEFAULT_MUON_LR",
    "DEFAULT_POOL_MULTIPLIER",
    "DEFAULT_PRIOR_TEMP",
    "DEFAULT_PSI0_GATE_INIT",
    "DEFAULT_RARE_ALPHA",
    "DEFAULT_RESIDUAL_INIT",
    "DEFAULT_SPHERE_PENALTY_WEIGHT",
    "DEFAULT_UNITARY_PENALTY_WEIGHT",
    "DEFAULT_VOCAB_CHUNK_SIZE",
    "RareTokenImportanceWindowSampler",
    "TiedHoloBornChunkedNLLFn",
    "TiedHoloBornComplexBlock",
    "TiedHoloBornComplexRoPEGPT",
    "TiedHoloBornConfig",
    "build_tied_holo_born_optimizers",
    "compute_soft_manifold_and_isotropy_penalties",
    "eval_tied_energy_born_probs",
    "eval_tied_holo_born_probs",
]
