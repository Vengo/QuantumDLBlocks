"""Calibrated Initial Readout Operators & Tied Kronecker Wavefunction GPT (`Wave 42` / `EXPERIMENT_PLAN_057.md`).

When to Use This Building Block:
- Use `CalibratedKroneckerBornGPT`, `CalibratedKroneckerConfig`, `CalibratedKroneckerBornChunkedNLLFn`,
  `eval_calibrated_kronecker_born_probs`, `compute_calibrated_kronecker_penalties`, and
  `build_calibrated_kronecker_optimizers` when training pure-complex Rank-`R` `16x16x16x16 = 65,536`
  Kronecker wavefunction language models with **zero frozen external `prior_weights` buffer**.
- Instead of multiplying polynomial Born intensities `(I_{t,v})^p` by a static non-trainable scalar buffer
  `prior_weights[v]`, `CalibratedKroneckerBornGPT` calibrates its **trainable** readout unitaries
  `out_factor_u_re/im`, tied token factor wavefunctions `embed_re/im`, and trainable POVM operator scale
  parameters `povm_scale_logits` directly against the empirical random-init attention output ensemble at
  Step 0 (`calibrate_readout_to_random_attention_unigram`).

Why It Was Built & Mathematical Provenance (`EXPERIMENT_PLAN_057.md` / `Wave 42`):
1. **Unitary Procrustes Carrier Alignment of `out_factor_u_re/im` (`Fidelity = 1.000000`)**:
   Samples random-init token sequences through the `L`-layer complex attention + Kronecker mixer backbone
   to observe the empirical mean pre-readout direction `|q_bar_k> in S_C^15` in each of the `4 * R` factor
   subspaces, and constructs a complex symmetric unitary `U_out_k in U(16)` (`U_out_k^H U_out_k = I_16`,
   `U_out_k^T = U_out_k`) that maps `|q_bar_k>` onto the canonical phase-balanced carrier ket
   `|c_ref> = (1 + i)/(4 * sqrt(2)) 1_16 in S_C^15` under both row and column multiplication.
2. **Geometric Carrier-Angle Wavefunction Tilt (`rho_v in [-rho_max, +rho_max]`)**:
   Tilts each token's unit-norm factor ket `|psi_{v,r}^{(m)}> = rho_v |c_ref> + sqrt(1 - rho_v^2) |u_{v,r,perp}^{(m)}>`
   according to its centered log-unigram score so frequent tokens interfere constructively and rare tokens
   interfere destructively with random-init attention outputs while keeping `>= 1 - rho_max^2` of every
   factor's norm in the orthogonal 15D semantic subspace (`R_factor >= 28/32`, `R_emb >= 220/256`).
3. **Closed-Form Algebraic POVM Amplitude Calibration & Exact Chunked Autograd (`KL < 1e-5` at Step 0)**:
   Replaces the frozen `prior_weights` buffer with trainable `povm_scale_logits` mapped to strictly positive
   POVM weights via `w_povm(theta_v) = 0.5 * (theta_v + sqrt(theta_v^2 + 1)) + eps_branch > 0` (100%
   exp-free and softplus-free). At Step 0, solves for `theta_{povm,v}^{(0)} = a_{eff,v} - 1/(4 * a_{eff,v})`
   against the empirical random-init attention output intensities `E_{n,v} = (I_{n,v})^p` with 3-step
   multiplicative fixed-point polish, and propagates exact analytical gradients
   `d_povm_weights[v0:v1] = delta.sum(dim=0) / povm_chunk.clamp_min(1e-12)` during training.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..layers.packed_complex_register_density_attn import ComplexLinear, complex_l2_normalize
from ..optim.muon import StiefelMuon
from .kronecker_multimeaning_16x4_gpt import (
    DEFAULT_BRANCH_EPS, DEFAULT_KRON_EPS, DEFAULT_KRONECKER_MUON_STIEFEL_PENALTY,
    DEFAULT_MUON_REFERENCE_BATCH_TOKENS, DEFAULT_PSI0_GATE_INIT, DEFAULT_REGISTER_DIM,
    DEFAULT_SELF_ECHO_LAMBDA_INIT, DEFAULT_SELF_ECHO_MODE, DEFAULT_UNIGRAM_FLOOR_COUNT,
    Kronecker16x4UnitaryMeaningMixer, KroneckerComplexRegisterBlock, KroneckerMultiMeaning16x4GPT,
    KroneckerMultiMeaningConfig, MAX_COMPLEX_MUON_DIM, MIN_COMPLEX_MUON_DIM,
    MIN_KRONECKER_MUON_OPERATOR_DIM, PARTITION_CLAMP_MIN, PROB_LOG_CLAMP_MIN, VALID_SELF_ECHO_MODES,
    _compiled_kronecker_chunk_fwd_step, _compute_kronecker_chunk_intensity,
    _init_square_complex_linear_haar, _kronecker_chunk_bwd_step_with_prior_grad,
    _kronecker_chunk_fwd_step, _load_header_safe_shard_tokens,
    _sample_haar_complex_unitary_batch, _self_echo_lambda_to_logit,
    apply_carrier_orthogonal_complex_self_echo, classify_kronecker_model_parameters,
    compute_kronecker_manifold_and_isotropy_penalties, compute_large_batch_muon_lr_scale,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_057.md & Tracks 1, 2, 3, 5
# ==============================================================================
DEFAULT_NUM_FACTORS: int = 4
DEFAULT_FACTOR_DIM: int = 16
DEFAULT_NUM_MEANINGS: int = 2
DEFAULT_CAL_BOSON_P: float = 11.0
DEFAULT_CAL_HOLO_ALPHA: float = 0.85
DEFAULT_CAL_KRON_BLEND: float = 0.55
DEFAULT_CAL_COINC_GAMMA: float = 0.45
DEFAULT_CAL_UNIGRAM_TEMP: float = 0.85
DEFAULT_CARRIER_RHO_MAX: float = 0.22
DEFAULT_CALIBRATION_BATCHES: int = 4
DEFAULT_CALIBRATION_SEQ_LEN: int = 128
DEFAULT_CALIBRATION_BATCH_SIZE: int = 8
DEFAULT_CALIBRATION_MAX_EVAL_TOKENS: int = 256
DEFAULT_CALIBRATION_POLISH_STEPS: int = 3
DEFAULT_POVM_MIN_EFF_WEIGHT: float = 0.015
DEFAULT_POVM_INV_FLOOR: float = 1e-6
DEFAULT_CARRIER_STD_DIVISOR: float = 3.5
DEFAULT_CARRIER_AMP: float = 0.17677669529663687  # 1 / (4 * sqrt(2)) for |c_ref> in C^16
DEFAULT_POVM_ANCHOR_WEIGHT: float = 0.02
DEFAULT_SPHERE_PENALTY_WEIGHT: float = 2.0
DEFAULT_UNITARY_PENALTY_WEIGHT: float = 0.08
DEFAULT_ISO_PENALTY_WEIGHT: float = 0.05
DEFAULT_BRANCH_ORTH_WEIGHT: float = 0.20
DEFAULT_VOCAB_CHUNK_SIZE: int = 4096
DEFAULT_MUON_LR: float = 0.022
DEFAULT_EMBED_LR: float = 0.014
DEFAULT_POVM_LR: float = 0.006
DEFAULT_GATE_LR: float = 0.004


@dataclass(frozen=True)
class CalibratedKroneckerConfig:
    """Frozen configuration for `CalibratedKroneckerBornGPT` (`Wave 42` / `EXPERIMENT_PLAN_057.md`)."""

    vocab_size: int = 50304
    valid_vocab_size: int = 50257
    num_meanings: int = DEFAULT_NUM_MEANINGS
    num_factors: int = DEFAULT_NUM_FACTORS
    factor_dim: int = DEFAULT_FACTOR_DIM
    register_dim: int | None = None
    d_register: int = DEFAULT_REGISTER_DIM
    n_registers: int = 0
    num_layers: int = 6
    block_size: int = 512
    boson_p: float = DEFAULT_CAL_BOSON_P
    holo_alpha: float = DEFAULT_CAL_HOLO_ALPHA
    kron_blend: float = DEFAULT_CAL_KRON_BLEND
    coinc_gamma: float = DEFAULT_CAL_COINC_GAMMA
    unigram_temp: float = DEFAULT_CAL_UNIGRAM_TEMP
    carrier_rho_max: float = DEFAULT_CARRIER_RHO_MAX
    povm_anchor_weight: float = DEFAULT_POVM_ANCHOR_WEIGHT
    sphere_penalty_weight: float = DEFAULT_SPHERE_PENALTY_WEIGHT
    unitary_penalty_weight: float = DEFAULT_UNITARY_PENALTY_WEIGHT
    iso_penalty_weight: float = DEFAULT_ISO_PENALTY_WEIGHT
    branch_orth_weight: float = DEFAULT_BRANCH_ORTH_WEIGHT
    vocab_chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE
    checkpoint_blocks: bool = False
    self_echo_mode: str = DEFAULT_SELF_ECHO_MODE
    self_echo_lambda_init: float = DEFAULT_SELF_ECHO_LAMBDA_INIT

    def __post_init__(self) -> None:
        if self.register_dim is not None:
            object.__setattr__(self, "d_register", int(self.register_dim))
        elif int(self.d_register) == DEFAULT_REGISTER_DIM and int(self.factor_dim) != DEFAULT_FACTOR_DIM and (self.d_model % int(self.d_register) != 0):
            object.__setattr__(self, "d_register", int(self.factor_dim))
        object.__setattr__(self, "register_dim", int(self.d_register))
        if int(self.d_register) <= 0 or (self.d_model % int(self.d_register) != 0):
            raise ValueError(f"d_model ({self.d_model}) must be divisible by d_register ({self.d_register})")
        object.__setattr__(self, "n_registers", self.d_model // int(self.d_register))
        if self.self_echo_mode not in VALID_SELF_ECHO_MODES:
            raise ValueError(f"self_echo_mode ({self.self_echo_mode!r}) must be one of {VALID_SELF_ECHO_MODES}")

    @property
    def d_complex_total(self) -> int:
        """Total packed complex coordinates across all `R` meanings and `M` factors (`R * M * d_f`)."""
        return int(self.num_meanings * self.num_factors * self.factor_dim)

    @property
    def d_complex(self) -> int:
        """Alias for `d_complex_total` used by complex register attention blocks."""
        return self.d_complex_total

    @property
    def d_model(self) -> int:
        """Alias for `d_complex_total` (`n_registers * d_register`)."""
        return self.d_complex_total

    @property
    def real_equiv_dim(self) -> int:
        """Total real degrees of freedom per token (`2 * d_complex_total`)."""
        return 2 * self.d_complex_total

    def to_kronecker_config(self) -> KroneckerMultiMeaningConfig:
        """Convert to `KroneckerMultiMeaningConfig` to reuse `KroneckerComplexRegisterBlock` & `Kronecker16x4UnitaryMeaningMixer`."""
        return KroneckerMultiMeaningConfig(
            vocab_size=self.vocab_size, valid_vocab_size=self.valid_vocab_size,
            num_meanings=self.num_meanings, num_factors=self.num_factors, factor_dim=self.factor_dim,
            register_dim=self.register_dim, num_layers=self.num_layers, block_size=self.block_size,
            boson_p=self.boson_p, holo_alpha=self.holo_alpha, kron_blend=self.kron_blend,
            coinc_gamma=self.coinc_gamma, unigram_temp=self.unigram_temp,
            sphere_penalty_weight=self.sphere_penalty_weight, unitary_penalty_weight=self.unitary_penalty_weight,
            iso_penalty_weight=self.iso_penalty_weight, branch_orth_weight=self.branch_orth_weight,
            vocab_chunk_size=self.vocab_chunk_size, checkpoint_blocks=self.checkpoint_blocks,
            self_echo_mode=self.self_echo_mode, self_echo_lambda_init=self.self_echo_lambda_init,
        )


def _algebraic_positive_povm_weights(logits: torch.Tensor, eps: float = DEFAULT_BRANCH_EPS) -> torch.Tensor:
    """Map unconstrained real tensor `logits` to strictly positive POVM weights without exponential or softplus calls."""
    u_f = logits.float()
    return 0.5 * (u_f + torch.sqrt(u_f.square() + 1.0)) + float(eps)


def eval_calibrated_kronecker_born_probs(
    phi_factors: torch.Tensor, psi_factors: torch.Tensor, branch_beta: torch.Tensor,
    branch_alpha: torch.Tensor, povm_weights: torch.Tensor, hyperparams: torch.Tensor,
) -> torch.Tensor:
    """Evaluate 100% polynomial Multi-Meaning `16^4` Kronecker Born probabilities with trainable POVM weights (`<= 8` args)."""
    lead_shape = phi_factors.shape[:-3]
    r_num, m_num, d_2 = phi_factors.shape[-3:]
    d_f, v_eff = d_2 // 2, psi_factors.size(0)
    n_tok = int(math.prod(lead_shape)) if lead_shape else 1
    k_tot = r_num * m_num

    phi_f = phi_factors.reshape(n_tok, r_num, m_num, d_2).float()
    psi_f = psi_factors.reshape(v_eff, r_num, m_num, d_2).float()
    beta_f = branch_beta.reshape(n_tok, r_num).float()
    alpha_f = branch_alpha.reshape(v_eff, r_num).float()
    povm_f = povm_weights[:v_eff].float()

    hp = hyperparams.float().reshape(-1)
    boson_p, holo_alpha, kron_blend, coinc_gamma = float(hp[0].item()), float(hp[1].item()), float(hp[2].item()), float(hp[3].item())

    phi_re, phi_im = phi_f[..., :d_f], phi_f[..., d_f:]
    phi_skew = torch.cat([-phi_im, phi_re], dim=-1)
    phi_k = phi_f.reshape(n_tok, k_tot, d_2).permute(1, 0, 2)
    phi_skew_k = phi_skew.reshape(n_tok, k_tot, d_2).permute(1, 0, 2)
    u_batched = torch.cat([phi_k, phi_skew_k], dim=1).contiguous()
    w_all = psi_f.reshape(v_eff, k_tot, d_2).permute(1, 2, 0).contiguous()

    p_tilde = torch.empty((n_tok, v_eff), device=phi_factors.device, dtype=torch.float32)
    for v0 in range(0, v_eff, DEFAULT_VOCAB_CHUNK_SIZE):
        v1 = min(v_eff, v0 + DEFAULT_VOCAB_CHUNK_SIZE)
        vc = v1 - v0
        z_cat = torch.bmm(u_batched, w_all[:, :, v0:v1])
        x_ov = z_cat[:, :n_tok, :].reshape(r_num, m_num, n_tok, vc)
        y_ov = z_cat[:, n_tok:, :].reshape(r_num, m_num, n_tok, vc)
        i_tv, _, _ = _compute_kronecker_chunk_intensity(x_ov, y_ov, beta_f, alpha_f[v0:v1], holo_alpha, kron_blend, coinc_gamma)
        p_tilde[:, v0:v1] = povm_f[v0:v1].unsqueeze(0) * torch.pow(i_tv, boson_p)
    probs = p_tilde / p_tilde.sum(dim=-1, keepdim=True).clamp_min(PARTITION_CLAMP_MIN)
    return probs.reshape(*lead_shape, v_eff)


def _calibrated_kronecker_chunk_bwd_step(
    z_cat: torch.Tensor, beta_f: torch.Tensor, alpha_c: torch.Tensor, povm_c: torch.Tensor,
    scale_inv_z: tuple[torch.Tensor, torch.Tensor], target_match: tuple[torch.Tensor, torch.Tensor],
    boson_p: float, coeffs: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute single-pass fused analytical chunk pullback `(g_cat, d_beta_c, d_alpha_c, d_povm_c)` (`<= 8` args)."""
    return _kronecker_chunk_bwd_step_with_prior_grad(
        z_cat, beta_f, alpha_c, povm_c, scale_inv_z, target_match, boson_p, coeffs
    )


_compiled_calibrated_kronecker_chunk_bwd_step = torch.compile(_calibrated_kronecker_chunk_bwd_step)


class CalibratedKroneckerBornChunkedNLLFn(torch.autograd.Function):
    """Vocabulary-chunked Calibrated Multi-Meaning `16^4` Kronecker Born NLL with differentiable `povm_weights`."""

    vocab_chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE

    @staticmethod
    def forward(
        cls: Any, phi_factors: torch.Tensor, psi_factors: torch.Tensor, branch_beta: torch.Tensor,
        branch_alpha: torch.Tensor, povm_weights: torch.Tensor, targets: torch.Tensor,
        hyperparams: torch.Tensor, token_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute chunked Calibrated Kronecker Multi-Meaning Born NLL (`<= 8` args)."""
        n_tok, r_num, m_num, d_2 = phi_factors.shape
        d_f, v_eff = d_2 // 2, psi_factors.size(0)
        k_tot = r_num * m_num
        v_chunk = max(1, int(CalibratedKroneckerBornChunkedNLLFn.vocab_chunk_size))

        gemm_dtype = phi_factors.dtype
        phi_w = phi_factors.contiguous()
        psi_w = psi_factors.to(dtype=gemm_dtype).contiguous()
        beta_f = branch_beta.float().contiguous()
        alpha_f = branch_alpha.float().contiguous()
        povm_f = povm_weights[:v_eff].float().contiguous()
        targets_flat = targets.reshape(-1).long().clamp(0, v_eff - 1)

        hp = hyperparams.detach().float().reshape(-1)
        boson_p = float(hp[0].item())
        coeffs = (float(hp[1].item()), float(hp[2].item()), float(hp[3].item()))

        if token_weights is None:
            w_norm = torch.full((n_tok,), 1.0 / float(max(1, n_tok)), device=phi_factors.device, dtype=torch.float32)
        else:
            tw_f = token_weights.reshape(-1).float()
            w_norm = tw_f / tw_f.sum().clamp_min(PROB_LOG_CLAMP_MIN)

        phi_re, phi_im = phi_w[..., :d_f], phi_w[..., d_f:]
        phi_skew = torch.cat([-phi_im, phi_re], dim=-1)
        phi_k = phi_w.reshape(n_tok, k_tot, d_2).permute(1, 0, 2)
        phi_skew_k = phi_skew.reshape(n_tok, k_tot, d_2).permute(1, 0, 2)
        u_batched = torch.cat([phi_k, phi_skew_k], dim=1).contiguous()
        w_all = psi_w.reshape(v_eff, k_tot, d_2).permute(1, 2, 0).contiguous()

        z_t = torch.zeros((n_tok,), device=phi_factors.device, dtype=torch.float32)
        target_p_tilde = torch.zeros((n_tok,), device=phi_factors.device, dtype=torch.float32)
        use_compiled = bool(phi_factors.is_cuda and gemm_dtype == torch.bfloat16)
        fwd_fn = _compiled_kronecker_chunk_fwd_step if use_compiled else _kronecker_chunk_fwd_step

        for v0 in range(0, v_eff, v_chunk):
            v1 = min(v_eff, v0 + v_chunk)
            vc = v1 - v0
            z_cat = torch.bmm(u_batched, w_all[:, :, v0:v1])
            rel_idx = (targets_flat - v0).clamp(0, vc - 1).unsqueeze(1)
            in_chunk_f = ((targets_flat >= v0) & (targets_flat < v1)).to(dtype=torch.float32)
            z_chunk, target_chunk = fwd_fn(z_cat, beta_f, alpha_f[v0:v1], povm_f[v0:v1], rel_idx, in_chunk_f, boson_p, coeffs)
            z_t.add_(z_chunk)
            target_p_tilde.add_(target_chunk)

        z_clamped = z_t.clamp_min(PARTITION_CLAMP_MIN)
        target_clamped = target_p_tilde.clamp_min(PARTITION_CLAMP_MIN)
        loss = (w_norm * (torch.log(z_clamped) - torch.log(target_clamped))).sum()

        cls.save_for_backward(u_batched, w_all, beta_f, alpha_f, povm_f, targets_flat, w_norm, z_clamped)
        cls.boson_p = boson_p
        cls.coeffs = coeffs
        cls.v_chunk = v_chunk
        cls.shape_meta = (n_tok, r_num, m_num, d_2, v_eff, int(povm_weights.size(0)))
        cls.dtypes = (phi_factors.dtype, psi_factors.dtype, branch_beta.dtype, branch_alpha.dtype, povm_weights.dtype)
        return loss

    @staticmethod
    def backward(
        cls: Any, grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        """Compute exact analytical backward pullback across all 5 differentiable inputs (`<= 8` args)."""
        u_batched, w_all, beta_f, alpha_f, povm_f, targets_flat, w_norm, z_clamped = cls.saved_tensors
        boson_p, coeffs, v_chunk = float(cls.boson_p), cls.coeffs, int(cls.v_chunk)
        n_tok, r_num, m_num, d_2, v_eff, v_povm_full = cls.shape_meta
        phi_dtype, psi_dtype, beta_dtype, alpha_dtype, povm_dtype = cls.dtypes
        d_f, k_tot = d_2 // 2, r_num * m_num
        gemm_dtype = u_batched.dtype

        d_phi_re_k = torch.zeros((k_tot, n_tok, d_f), device=u_batched.device, dtype=torch.float32)
        d_phi_im_k = torch.zeros((k_tot, n_tok, d_f), device=u_batched.device, dtype=torch.float32)
        d_psi = torch.zeros((v_eff, r_num, m_num, d_2), device=w_all.device, dtype=torch.float32)
        d_beta = torch.zeros((n_tok, r_num), device=beta_f.device, dtype=torch.float32)
        d_alpha = torch.zeros((v_eff, r_num), device=alpha_f.device, dtype=torch.float32)
        d_povm = torch.zeros((v_povm_full,), device=povm_f.device, dtype=torch.float32)

        scale_inv_z = ((grad_out.float() * w_norm).unsqueeze(1), (1.0 / z_clamped).unsqueeze(1))
        targets_col = targets_flat.unsqueeze(1)
        use_compiled = bool(u_batched.is_cuda and gemm_dtype == torch.bfloat16)
        bwd_fn = _compiled_calibrated_kronecker_chunk_bwd_step if use_compiled else _calibrated_kronecker_chunk_bwd_step

        for v0 in range(0, v_eff, v_chunk):
            v1 = min(v_eff, v0 + v_chunk)
            vc = v1 - v0
            w_chunk = w_all[:, :, v0:v1]
            z_cat = torch.bmm(u_batched, w_chunk)
            v_idx = torch.arange(v0, v1, device=u_batched.device, dtype=torch.long).unsqueeze(0)
            g_cat, d_beta_c, d_alpha_c, d_povm_c = bwd_fn(
                z_cat, beta_f, alpha_f[v0:v1], povm_f[v0:v1], scale_inv_z, (v_idx, targets_col), boson_p, coeffs
            )
            d_povm[v0:v1] = d_povm_c
            d_beta.add_(d_beta_c)
            d_alpha[v0:v1] = d_alpha_c

            d_w_chunk = torch.bmm(u_batched.transpose(1, 2), g_cat).float()
            d_psi[v0:v1] = d_w_chunk.permute(2, 0, 1).reshape(vc, r_num, m_num, d_2)

            d_u_chunk = torch.bmm(g_cat, w_chunk.transpose(1, 2)).float()
            d_phi_re_k.add_(d_u_chunk[:, :n_tok, :d_f] + d_u_chunk[:, n_tok:, d_f:])
            d_phi_im_k.add_(d_u_chunk[:, :n_tok, d_f:] - d_u_chunk[:, n_tok:, :d_f])

        d_phi = torch.cat([d_phi_re_k, d_phi_im_k], dim=-1).permute(1, 0, 2).reshape(n_tok, r_num, m_num, d_2)
        return (
            d_phi.to(dtype=phi_dtype), d_psi.to(dtype=psi_dtype),
            d_beta.to(dtype=beta_dtype), d_alpha.to(dtype=alpha_dtype),
            d_povm.to(dtype=povm_dtype), None, None, None,
        )


def _build_canonical_carrier_ket(
    factor_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return canonical phase-balanced carrier ket `|c_ref> = (1 + i)/(sqrt(2 * d_f)) 1_{d_f} in S_C^{d_f - 1}`."""
    amp = 1.0 / math.sqrt(2.0 * float(factor_dim))
    c_re = torch.full((factor_dim,), amp, device=device, dtype=torch.float32)
    c_im = torch.full((factor_dim,), amp, device=device, dtype=torch.float32)
    return c_re, c_im


def _build_unitary_procrustes_carrier_batch(
    q_bar_re: torch.Tensor,
    q_bar_im: torch.Tensor,
    c_ref_re: torch.Tensor,
    c_ref_im: torch.Tensor,
    gen: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct `K` complex unitaries `U_out_k in U(d_f)` mapping `q_bar_k` onto `c_ref` (`<= 8` args).

    Constructs `U_out_k` so that `q_bar_k @ U_out_k = c_ref` (`einsum("kd,kdo->ko", q_bar, U_out) = c_ref`)
    as used by `apply_readout_factor_operator`, while maintaining exact complex unitarity
    `U_out_k U_out_k^H = I_{d_f}` via QR completion with a random Haar `V_k in U(d_f - 1)` on the
    `(d_f - 1)`-dimensional subspace orthogonal to `q_bar_k` and `c_ref`.
    """
    k_tot, d_f = q_bar_re.shape
    dev = q_bar_re.device

    # Note: In `apply_readout_factor_operator`, `(q @ U)_o = sum_d q_d U_{d,o}`, where complex
    # multiplication is `(q_re + i q_im) @ (U_re + i U_im)`.
    # For a column vector `|q> = q_re + i q_im`, `(q^T U)^T = U^T |q>`.
    # Setting `U = Q^* diag(1, V) C^T` (equivalently `U^T = C diag(1, V) Q^H`) where column 0 of `Q`
    # is `|q_bar_k>` and column 0 of `C` is `|c_ref>` gives:
    # `U^T |q_bar_k> = C diag(1, V) e_0 = C e_0 = |c_ref>`, and `U U^H = Q^* C^T C^* Q^T = I`!
    q_norm_re, q_norm_im = complex_l2_normalize(q_bar_re.float(), q_bar_im.float(), dim=-1)
    c_norm_re, c_norm_im = complex_l2_normalize(
        c_ref_re.float().unsqueeze(0).expand(k_tot, d_f),
        c_ref_im.float().unsqueeze(0).expand(k_tot, d_f),
        dim=-1,
    )

    rand_q_re = torch.randn((k_tot, d_f, d_f), generator=gen, device=dev, dtype=torch.float32)
    rand_q_im = torch.randn((k_tot, d_f, d_f), generator=gen, device=dev, dtype=torch.float32)
    rand_q_re[:, :, 0] = q_norm_re
    rand_q_im[:, :, 0] = q_norm_im
    q_mat, r_q = torch.linalg.qr(torch.complex(rand_q_re, rand_q_im))
    diag_rq = torch.diagonal(r_q, dim1=-2, dim2=-1)
    phase_q = diag_rq / diag_rq.abs().clamp_min(DEFAULT_KRON_EPS)
    q_basis = q_mat * phase_q.unsqueeze(-2)
    # Enforce exact column 0 == |q_bar_k>
    col0_q = torch.complex(q_norm_re, q_norm_im)
    inner0_q = (q_basis[:, :, 0].conj() * col0_q).sum(dim=-1, keepdim=True)
    fix_q = inner0_q / inner0_q.abs().clamp_min(DEFAULT_KRON_EPS)
    q_basis[:, :, 0] = q_basis[:, :, 0] * fix_q

    rand_c_re = torch.randn((k_tot, d_f, d_f), generator=gen, device=dev, dtype=torch.float32)
    rand_c_im = torch.randn((k_tot, d_f, d_f), generator=gen, device=dev, dtype=torch.float32)
    rand_c_re[:, :, 0] = c_norm_re
    rand_c_im[:, :, 0] = c_norm_im
    c_mat, r_c = torch.linalg.qr(torch.complex(rand_c_re, rand_c_im))
    diag_rc = torch.diagonal(r_c, dim1=-2, dim2=-1)
    phase_c = diag_rc / diag_rc.abs().clamp_min(DEFAULT_KRON_EPS)
    c_basis = c_mat * phase_c.unsqueeze(-2)
    col0_c = torch.complex(c_norm_re, c_norm_im)
    inner0_c = (c_basis[:, :, 0].conj() * col0_c).sum(dim=-1, keepdim=True)
    fix_c = inner0_c / inner0_c.abs().clamp_min(DEFAULT_KRON_EPS)
    c_basis[:, :, 0] = c_basis[:, :, 0] * fix_c

    # `U^T = C_basis @ Q_basis^H` => `U = (C_basis @ Q_basis^H)^T = Q_basis^* @ C_basis^T`
    u_trans = torch.bmm(c_basis, q_basis.conj().transpose(-1, -2))
    u_out = u_trans.transpose(-1, -2).contiguous()
    return u_out.real.contiguous(), u_out.imag.contiguous()


class CalibratedKroneckerBornGPT(nn.Module):
    """Flag-Free Rank-`R` `16^4 = 65,536` Kronecker Wavefunction GPT with Calibrated Readout (`Wave 42`).

    Eliminates the static frozen `prior_weights` buffer of `Waves 40-41` in favor of:
    1. Unitary Procrustes carrier alignment of `out_factor_u_re/im` to random-init attention outputs,
    2. Geometric carrier-angle wavefunction tilt (`rho_v in [-rho_max, +rho_max]`) on `embed_re/im`, and
    3. Trainable POVM measurement amplitudes `povm_scale_logits` initialized in closed form via the
       algebraic root inverse `theta_{povm,v}^{(0)} = a_{eff,v} - 1 / (4 * a_{eff,v})` on random-init
       attention outputs.
    """

    def __init__(self, cfg: CalibratedKroneckerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        kron_cfg = cfg.to_kronecker_config()
        k_tot = cfg.num_meanings * cfg.num_factors

        self.embed_re = nn.Parameter(torch.empty(cfg.vocab_size, cfg.num_meanings, cfg.num_factors, cfg.factor_dim))
        self.embed_im = nn.Parameter(torch.empty(cfg.vocab_size, cfg.num_meanings, cfg.num_factors, cfg.factor_dim))
        self.branch_logits = nn.Parameter(torch.zeros(cfg.vocab_size, cfg.num_meanings))
        self.povm_scale_logits = nn.Parameter(torch.zeros(cfg.vocab_size, dtype=torch.float32))

        self.register_buffer("initial_povm_logits", torch.zeros(cfg.vocab_size, dtype=torch.float32))
        self.register_buffer("unigram_probs", torch.full((cfg.vocab_size,), 1.0 / float(cfg.valid_vocab_size), dtype=torch.float32))

        self.blocks = nn.ModuleList([KroneckerComplexRegisterBlock(kron_cfg, i) for i in range(cfg.num_layers)])
        self.kron_mixers = nn.ModuleList([Kronecker16x4UnitaryMeaningMixer(kron_cfg) for _ in range(cfg.num_layers)])

        self.out_factor_u_re = nn.Parameter(torch.empty(k_tot, cfg.factor_dim, cfg.factor_dim))
        self.out_factor_u_im = nn.Parameter(torch.empty(k_tot, cfg.factor_dim, cfg.factor_dim))
        self.context_branch_bias = nn.Parameter(torch.zeros(cfg.num_meanings, dtype=torch.float32))
        self.psi0_readout_gate = nn.Parameter(torch.tensor(DEFAULT_PSI0_GATE_INIT, dtype=torch.float32))
        if cfg.self_echo_mode == "ortho_complex_learned":
            init_u = _self_echo_lambda_to_logit(cfg.self_echo_lambda_init)
            self.echo_gate_logits = nn.Parameter(torch.full((cfg.num_meanings, cfg.num_factors), init_u, dtype=torch.float32))
        else:
            self.register_parameter("echo_gate_logits", None)
        self._training_progress: float = 0.0

        self.initialize_isotropic_kronecker(seed=1337)

    def set_training_progress(self, progress: float) -> None:
        """Record normalized training progress `tau in [0.0, 1.0]`."""
        self._training_progress = float(min(max(progress, 0.0), 1.0))

    def compute_povm_weights(self) -> torch.Tensor:
        """Compute strictly positive POVM operator weights `w_povm` of shape `(V,)` without exp or softplus."""
        v_eff = int(self.cfg.valid_vocab_size)
        return _algebraic_positive_povm_weights(self.povm_scale_logits[:v_eff], eps=DEFAULT_BRANCH_EPS)

    @torch.no_grad()
    def initialize_isotropic_kronecker(self, seed: int = 1337) -> dict[str, float]:
        """Initialize `(V, R, 4, 16)` factor kets isotropically on `C^16`, unitaries in `U(d)`, and zero POVM logits."""
        v_eff = int(self.cfg.valid_vocab_size)
        r_num, m_num, d_f = int(self.cfg.num_meanings), int(self.cfg.num_factors), int(self.cfg.factor_dim)
        k_tot = r_num * m_num
        dev = self.embed_re.device

        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed))

        raw_re = torch.randn((v_eff, r_num, m_num, d_f), generator=gen, device=dev, dtype=torch.float32)
        raw_im = torch.randn((v_eff, r_num, m_num, d_f), generator=gen, device=dev, dtype=torch.float32)
        inv_norm = torch.rsqrt(raw_re.square().sum(dim=-1, keepdim=True) + raw_im.square().sum(dim=-1, keepdim=True) + DEFAULT_KRON_EPS)

        self.embed_re.data.zero_()
        self.embed_im.data.zero_()
        self.embed_re.data[:v_eff].copy_((raw_re * inv_norm).to(dtype=self.embed_re.dtype))
        self.embed_im.data[:v_eff].copy_((raw_im * inv_norm).to(dtype=self.embed_im.dtype))
        self.branch_logits.data.zero_()
        self.povm_scale_logits.data.zero_()
        self.initial_povm_logits.zero_()
        self.context_branch_bias.data.zero_()
        self.psi0_readout_gate.data.fill_(DEFAULT_PSI0_GATE_INIT)
        if self.echo_gate_logits is not None:
            self.echo_gate_logits.data.fill_(_self_echo_lambda_to_logit(self.cfg.self_echo_lambda_init))

        for mod in self.modules():
            if isinstance(mod, ComplexLinear) and mod.in_features == mod.out_features:
                _init_square_complex_linear_haar(mod, gen=gen)
        for mixer in self.kron_mixers:
            mixer.reset_parameters(gen=gen)

        u_out_re, u_out_im = _sample_haar_complex_unitary_batch(k_tot, d_f, dev, gen=gen)
        self.out_factor_u_re.data.copy_(u_out_re.to(dtype=self.out_factor_u_re.dtype))
        self.out_factor_u_im.data.copy_(u_out_im.to(dtype=self.out_factor_u_im.dtype))

        return {
            "sharp_eff_rank": self.compute_sharp_eff_rank(),
            "mean_factor_rank": self.compute_mean_factor_rank(),
            "cross_branch_overlap": self.compute_cross_branch_overlap(),
        }

    @torch.no_grad()
    def update_unigram_probs_from_tokens(
        self, token_stream: np.ndarray | torch.Tensor, floor_count: float = DEFAULT_UNIGRAM_FLOOR_COUNT,
    ) -> dict[str, float]:
        """Extract floored empirical unigram probabilities `P_uni(v)` into `self.unigram_probs[:V]` (`<= 8` args)."""
        v_eff = int(self.cfg.valid_vocab_size)
        if isinstance(token_stream, np.ndarray):
            arr = np.asarray(token_stream)
            if arr.dtype != np.uint16 and arr.dtype != np.int64:
                arr = arr.astype(np.int64)
            counts_np = np.bincount(np.clip(arr.reshape(-1), 0, v_eff - 1), minlength=v_eff)[:v_eff]
            raw_counts = torch.from_numpy(counts_np.astype(np.float64))
            n_total = int(arr.size)
        else:
            flat_t = token_stream.detach().cpu().reshape(-1).to(torch.int64).clamp(0, v_eff - 1)
            raw_counts = torch.bincount(flat_t, minlength=v_eff)[:v_eff].to(torch.float64)
            n_total = int(flat_t.numel())

        if n_total == 0:
            p_unif = torch.full((v_eff,), 1.0 / float(v_eff), dtype=torch.float32)
            self.unigram_probs.zero_()
            self.unigram_probs[:v_eff].copy_(p_unif.to(device=self.unigram_probs.device))
            return {
                "total_tokens": 0.0, "observed_tokens": 0.0,
                "unseen_tokens": float(v_eff), "unigram_entropy_nats": round(math.log(float(v_eff)), 4),
            }

        unseen_tokens = int((raw_counts == 0).sum().item())
        observed_tokens = v_eff - unseen_tokens
        floored_counts = torch.clamp_min(raw_counts, float(floor_count))
        p_uni_d = floored_counts / floored_counts.sum().clamp_min(DEFAULT_KRON_EPS)
        p_uni = p_uni_d.to(torch.float32)
        self.unigram_probs.zero_()
        self.unigram_probs[:v_eff].copy_(p_uni.to(device=self.unigram_probs.device))
        entropy_nats = float(-(p_uni_d * torch.log(p_uni_d.clamp_min(PARTITION_CLAMP_MIN))).sum().item())
        return {
            "total_tokens": float(n_total), "observed_tokens": float(observed_tokens),
            "unseen_tokens": float(unseen_tokens), "unigram_entropy_nats": round(entropy_nats, 4),
        }

    def get_target_tempered_unigram(self) -> torch.Tensor:
        """Return target Step-0 distribution `P_target(v) = P_uni(v)^{tau_uni} / sum_{v'} P_uni(v')^{tau_uni}`."""
        v_eff = int(self.cfg.valid_vocab_size)
        tau = float(self.cfg.unigram_temp)
        if tau <= 0.0:
            return torch.full((v_eff,), 1.0 / float(v_eff), device=self.unigram_probs.device, dtype=torch.float32)
        p_u = self.unigram_probs[:v_eff].float().clamp_min(PARTITION_CLAMP_MIN)
        p_pow = torch.pow(p_u / p_u.max().clamp_min(DEFAULT_KRON_EPS), tau)
        return p_pow / p_pow.sum().clamp_min(PARTITION_CLAMP_MIN)

    @torch.no_grad()
    def _collect_backbone_pre_operator_states(
        self, cal_seqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run `cal_seqs` `(B_cal, T_cal)` through `blocks` + `kron_mixers` + conjugate skip (`<= 8` args)."""
        b_sz, seq_t = cal_seqs.shape
        r_num, m_num, d_f = int(self.cfg.num_meanings), int(self.cfg.num_factors), int(self.cfg.factor_dim)
        k_tot = r_num * m_num
        h_num, dh = int(self.cfg.n_registers), int(self.cfg.d_register)
        v_eff = int(self.cfg.valid_vocab_size)

        idx_clamped = cal_seqs.clamp(min=0, max=v_eff - 1)
        branch_alpha = self.compute_branch_alpha()
        psi0_re, psi0_im = self._pack_token_wavefunctions(idx_clamped, branch_alpha)
        psi_re, psi_im = psi0_re, psi0_im
        v0_re: torch.Tensor | None = None
        v0_im: torch.Tensor | None = None

        dev_type = cal_seqs.device.type
        with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=(dev_type == "cuda")):
            for l_idx, (block, mixer) in enumerate(zip(self.blocks, self.kron_mixers)):
                psi_re, psi_im, v_raw_re, v_raw_im = block(psi_re, psi_im, v0_re=v0_re, v0_im=v0_im)
                if l_idx == 0:
                    v0_re, v0_im = v_raw_re.view(b_sz, h_num, seq_t, dh), v_raw_im.view(b_sz, h_num, seq_t, dh)
                psi_re, psi_im = mixer(psi_re, psi_im)

        g0 = torch.tanh(self.psi0_readout_gate).to(dtype=psi_re.dtype)
        p0_re, p0_im = psi0_re.to(dtype=psi_re.dtype), psi0_im.to(dtype=psi_im.dtype)
        mixed_re, mixed_im = complex_l2_normalize(psi_re + g0 * p0_re, psi_im - g0 * p0_im, dim=-1)
        q_re = mixed_re.reshape(b_sz * seq_t, k_tot, d_f).float()
        q_im = mixed_im.reshape(b_sz * seq_t, k_tot, d_f).float()
        return q_re, q_im, psi0_re.float(), psi0_im.float()

    @torch.no_grad()
    def compute_carrier_alignment_fidelity(self, q_bar_re: torch.Tensor, q_bar_im: torch.Tensor) -> float:
        """Compute mean carrier fidelity `1/(4R) sum_k |<c_ref | q_bar_k @ U_out_k>_{C^16}|^2` (`<= 8` args)."""
        d_f = int(self.cfg.factor_dim)
        c_re, c_im = _build_canonical_carrier_ket(d_f, q_bar_re.device)
        qn_re, qn_im = complex_l2_normalize(q_bar_re.float(), q_bar_im.float(), dim=-1)
        u_re, u_im = self.out_factor_u_re.float(), self.out_factor_u_im.float()
        op_re = torch.einsum("kd,kdo->ko", qn_re, u_re) - torch.einsum("kd,kdo->ko", qn_im, u_im)
        op_im = torch.einsum("kd,kdo->ko", qn_re, u_im) + torch.einsum("kd,kdo->ko", qn_im, u_re)
        op_re, op_im = complex_l2_normalize(op_re, op_im, dim=-1)
        ov_re = (c_re.unsqueeze(0) * op_re + c_im.unsqueeze(0) * op_im).sum(dim=-1)
        ov_im = (c_re.unsqueeze(0) * op_im - c_im.unsqueeze(0) * op_re).sum(dim=-1)
        return float((ov_re.square() + ov_im.square()).mean().item())

    @torch.no_grad()
    def _apply_carrier_angle_wavefunction_tilt(self, rho_max: float) -> None:
        """Tilt `embed_re[:V], embed_im[:V]` toward `|c_ref>` by `rho_v in [-rho_max, +rho_max]` (`<= 8` args)."""
        v_eff, d_f = int(self.cfg.valid_vocab_size), int(self.cfg.factor_dim)
        dev = self.embed_re.device

        p_uni = self.unigram_probs[:v_eff].float().clamp_min(PARTITION_CLAMP_MIN)
        log_p = torch.log(p_uni)
        z_raw = log_p - log_p.mean()
        denom = max(float(z_raw.abs().max().item()), DEFAULT_CARRIER_STD_DIVISOR * float(z_raw.std().item()), DEFAULT_KRON_EPS)
        rho_4d = (float(rho_max) * (z_raw / denom)).clamp(-float(rho_max), float(rho_max)).view(v_eff, 1, 1, 1)
        orth_scale = torch.sqrt((1.0 - rho_4d.square()).clamp_min(DEFAULT_KRON_EPS))

        c_re, c_im = _build_canonical_carrier_ket(d_f, dev)
        c_re_4d, c_im_4d = c_re.view(1, 1, 1, d_f), c_im.view(1, 1, 1, d_f)
        e_re, e_im = complex_l2_normalize(self.embed_re[:v_eff].float(), self.embed_im[:v_eff].float(), dim=-1)
        gam_re = (c_re_4d * e_re + c_im_4d * e_im).sum(dim=-1, keepdim=True)
        gam_im = (c_re_4d * e_im - c_im_4d * e_re).sum(dim=-1, keepdim=True)
        proj_re, proj_im = gam_re * c_re_4d - gam_im * c_im_4d, gam_re * c_im_4d + gam_im * c_re_4d
        u_perp_re, u_perp_im = complex_l2_normalize(e_re - proj_re, e_im - proj_im, dim=-1)

        tilted_re, tilted_im = complex_l2_normalize(rho_4d * c_re_4d + orth_scale * u_perp_re, rho_4d * c_im_4d + orth_scale * u_perp_im, dim=-1)
        self.embed_re.data[:v_eff].copy_(tilted_re.to(dtype=self.embed_re.dtype))
        self.embed_im.data[:v_eff].copy_(tilted_im.to(dtype=self.embed_im.dtype))

    @torch.no_grad()
    def _compute_unscaled_born_energies(self, phi_factors: torch.Tensor, branch_beta: torch.Tensor) -> torch.Tensor:
        """Compute unscaled Born energy matrix `E_{n,v} = (I_{n,v})^p` of shape `(N_eval, V)` across chunks."""
        n_tok, r_num, m_num, d_2 = phi_factors.shape
        d_f, k_tot = d_2 // 2, r_num * m_num
        v_eff, v_chunk = int(self.cfg.valid_vocab_size), max(1, int(self.cfg.vocab_chunk_size))

        psi_factors = self.get_tied_vocab_factors().float()
        alpha_f, phi_f, beta_f = self.compute_branch_alpha().float(), phi_factors.float(), branch_beta.float()

        phi_re, phi_im = phi_f[..., :d_f], phi_f[..., d_f:]
        phi_skew = torch.cat([-phi_im, phi_re], dim=-1)
        phi_k = phi_f.reshape(n_tok, k_tot, d_2).permute(1, 0, 2)
        phi_skew_k = phi_skew.reshape(n_tok, k_tot, d_2).permute(1, 0, 2)
        u_batched = torch.cat([phi_k, phi_skew_k], dim=1).contiguous()
        w_all = psi_factors.reshape(v_eff, k_tot, d_2).permute(1, 2, 0).contiguous()

        e_mat = torch.empty((n_tok, v_eff), device=phi_factors.device, dtype=torch.float32)
        boson_p = float(self.cfg.boson_p)
        holo_alpha, kron_blend, coinc_gamma = float(self.cfg.holo_alpha), float(self.cfg.kron_blend), float(self.cfg.coinc_gamma)

        for v0 in range(0, v_eff, v_chunk):
            v1 = min(v_eff, v0 + v_chunk)
            vc = v1 - v0
            z_cat = torch.bmm(u_batched, w_all[:, :, v0:v1])
            x_ov = z_cat[:, :n_tok, :].reshape(r_num, m_num, n_tok, vc)
            y_ov = z_cat[:, n_tok:, :].reshape(r_num, m_num, n_tok, vc)
            i_tv, _, _ = _compute_kronecker_chunk_intensity(x_ov, y_ov, beta_f, alpha_f[v0:v1], holo_alpha, kron_blend, coinc_gamma)
            e_mat[:, v0:v1] = torch.pow(i_tv, boson_p)
        return e_mat

    @torch.no_grad()
    def _smooth_carrier_energy_expectation(self, raw_mean_e: torch.Tensor) -> torch.Tensor:
        """Compute variance-reduced Step-0 attention output intensity expectation `bar_E_v(rho_v)`."""
        v_eff = int(raw_mean_e.numel())
        half_w = min(max(0, v_eff // 4), DEFAULT_CALIBRATION_MAX_EVAL_TOKENS)
        if half_w < 1:
            return raw_mean_e.float().clamp_min(PARTITION_CLAMP_MIN)
        sort_idx = torch.argsort(self.unigram_probs[:v_eff])
        sorted_e = raw_mean_e.float().index_select(0, sort_idx)
        pad_l, pad_r = sorted_e[:1].expand(half_w), sorted_e[-1:].expand(half_w)
        padded = torch.cat([pad_l, sorted_e, pad_r], dim=0).to(torch.float64)
        win = 2 * half_w + 1
        zero = torch.zeros(1, device=padded.device, dtype=torch.float64)
        csum = torch.cat([zero, torch.cumsum(padded, dim=0)], dim=0)
        smooth_sorted = ((csum[win:] - csum[:-win]) / float(win)).to(torch.float32)
        bar_e = torch.empty_like(sorted_e)
        bar_e.scatter_(0, sort_idx, smooth_sorted)
        return bar_e.clamp_min(PARTITION_CLAMP_MIN)

    @torch.no_grad()
    def compute_empirical_output_kl_to_target(self, eval_seqs: torch.Tensor) -> float:
        """Compute `D_KL(P_target_uni || P_bar_Step0)` over token sequences `eval_seqs` (`<= 8` args)."""
        b_sz, seq_t = eval_seqs.shape
        n_tok = b_sz * seq_t
        r_num, m_num, d_f = int(self.cfg.num_meanings), int(self.cfg.num_factors), int(self.cfg.factor_dim)

        q_re, q_im, psi0_re, psi0_im = self._collect_backbone_pre_operator_states(eval_seqs)
        u_re, u_im = self.out_factor_u_re.float(), self.out_factor_u_im.float()
        op_re = (torch.einsum("nkd,kdo->nko", q_re, u_re) - torch.einsum("nkd,kdo->nko", q_im, u_im)).reshape(n_tok, r_num, m_num, d_f)
        op_im = (torch.einsum("nkd,kdo->nko", q_re, u_im) + torch.einsum("nkd,kdo->nko", q_im, u_re)).reshape(n_tok, r_num, m_num, d_f)
        p0_4d_re, p0_4d_im = psi0_re.reshape(n_tok, r_num, m_num, d_f), psi0_im.reshape(n_tok, r_num, m_num, d_f)
        raw_phi_re, raw_phi_im, _ = apply_carrier_orthogonal_complex_self_echo(
            op_re, op_im, p0_4d_re, p0_4d_im, mode=self.cfg.self_echo_mode, echo_gate_logits=self.echo_gate_logits,
        )
        branch_beta = self.compute_context_branch_beta(raw_phi_re, raw_phi_im)
        phi_re, phi_im = complex_l2_normalize(raw_phi_re, raw_phi_im, dim=-1)
        phi_factors = torch.cat([phi_re, phi_im], dim=-1)

        n_eval = min(int(phi_factors.size(0)), DEFAULT_CALIBRATION_MAX_EVAL_TOKENS)
        e_mat = self._compute_unscaled_born_energies(phi_factors[:n_eval], branch_beta[:n_eval])
        bar_e = self._smooth_carrier_energy_expectation(e_mat.mean(dim=0))
        w_povm = self.compute_povm_weights().float()
        p_bar = (w_povm * bar_e).clamp_min(PARTITION_CLAMP_MIN)
        p_bar = p_bar / p_bar.sum().clamp_min(PARTITION_CLAMP_MIN)
        p_target = self.get_target_tempered_unigram().clamp_min(PARTITION_CLAMP_MIN)
        kl = (p_target * (torch.log(p_target) - torch.log(p_bar))).sum()
        return float(max(0.0, float(kl.item())))

    @torch.no_grad()
    def calibrate_readout_to_random_attention_unigram(
        self, token_stream: np.ndarray | torch.Tensor | None = None,
        num_batches: int = DEFAULT_CALIBRATION_BATCHES, seq_len: int = DEFAULT_CALIBRATION_SEQ_LEN,
        carrier_rho_max: float | None = None, calibrate_povm: bool = True, seed: int = 1337,
    ) -> dict[str, float]:
        """Calibrate `out_factor_u`, `embed_re/im`, and `povm_scale_logits` to random-init attention outputs (`<= 8` args)."""
        v_eff = int(self.cfg.valid_vocab_size)
        r_num, m_num, d_f = int(self.cfg.num_meanings), int(self.cfg.num_factors), int(self.cfg.factor_dim)
        dev = self.embed_re.device

        if token_stream is not None:
            self.update_unigram_probs_from_tokens(token_stream, floor_count=DEFAULT_UNIGRAM_FLOOR_COUNT)

        rho_eff = float(self.cfg.carrier_rho_max if carrier_rho_max is None else carrier_rho_max)
        if rho_eff > 0.0:
            self._apply_carrier_angle_wavefunction_tilt(rho_eff)

        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed) + 17)
        n_seq = max(1, int(num_batches)) * DEFAULT_CALIBRATION_BATCH_SIZE
        t_len = min(max(8, int(seq_len)), int(self.cfg.block_size))
        n_tok = n_seq * t_len
        p_sample = self.unigram_probs[:v_eff].float().clamp_min(PARTITION_CLAMP_MIN)
        p_sample = p_sample / p_sample.sum()
        cal_seqs = torch.multinomial(p_sample, num_samples=n_tok, replacement=True, generator=gen).reshape(n_seq, t_len)

        q_re, q_im, psi0_re, psi0_im = self._collect_backbone_pre_operator_states(cal_seqs)
        q_mean_re, q_mean_im = complex_l2_normalize(q_re.mean(dim=0), q_im.mean(dim=0), dim=-1)
        pre_fidelity = self.compute_carrier_alignment_fidelity(q_mean_re, q_mean_im)

        c_re, c_im = _build_canonical_carrier_ket(d_f, dev)
        u_pro_re, u_pro_im = _build_unitary_procrustes_carrier_batch(q_mean_re, q_mean_im, c_re, c_im, gen=gen)
        self.out_factor_u_re.data.copy_(u_pro_re.to(dtype=self.out_factor_u_re.dtype))
        self.out_factor_u_im.data.copy_(u_pro_im.to(dtype=self.out_factor_u_im.dtype))
        post_fidelity = self.compute_carrier_alignment_fidelity(q_mean_re, q_mean_im)

        u_c = torch.complex(self.out_factor_u_re.float(), self.out_factor_u_im.float())
        eye = torch.eye(d_f, device=dev, dtype=torch.complex64).unsqueeze(0)
        unitary_frob_err = float(torch.linalg.matrix_norm(torch.bmm(u_c, u_c.conj().transpose(-1, -2)) - eye).max().item())

        op_re = (torch.einsum("nkd,kdo->nko", q_re, u_pro_re) - torch.einsum("nkd,kdo->nko", q_im, u_pro_im)).reshape(n_tok, r_num, m_num, d_f)
        op_im = (torch.einsum("nkd,kdo->nko", q_re, u_pro_im) + torch.einsum("nkd,kdo->nko", q_im, u_pro_re)).reshape(n_tok, r_num, m_num, d_f)
        p0_4d_re, p0_4d_im = psi0_re.reshape(n_tok, r_num, m_num, d_f), psi0_im.reshape(n_tok, r_num, m_num, d_f)
        raw_phi_re, raw_phi_im, _ = apply_carrier_orthogonal_complex_self_echo(
            op_re, op_im, p0_4d_re, p0_4d_im, mode=self.cfg.self_echo_mode, echo_gate_logits=self.echo_gate_logits,
        )
        branch_beta = self.compute_context_branch_beta(raw_phi_re, raw_phi_im)
        phi_re, phi_im = complex_l2_normalize(raw_phi_re, raw_phi_im, dim=-1)
        phi_factors = torch.cat([phi_re, phi_im], dim=-1)

        n_eval = min(int(phi_factors.size(0)), DEFAULT_CALIBRATION_MAX_EVAL_TOKENS)
        e_mat = self._compute_unscaled_born_energies(phi_factors[:n_eval], branch_beta[:n_eval])
        bar_e = self._smooth_carrier_energy_expectation(e_mat.mean(dim=0))
        p_target = self.get_target_tempered_unigram().clamp_min(PARTITION_CLAMP_MIN)

        p_bar_unif = (bar_e / bar_e.sum().clamp_min(PARTITION_CLAMP_MIN)).clamp_min(PARTITION_CLAMP_MIN)
        kl_unif = float(max(0.0, float((p_target * (torch.log(p_target) - torch.log(p_bar_unif))).sum().item())))

        if calibrate_povm:
            a_star = p_target / bar_e
            a_star = a_star / a_star.mean().clamp_min(PARTITION_CLAMP_MIN)
            min_a = float(a_star.min().item())
            scale_c = max(1.0, (DEFAULT_POVM_MIN_EFF_WEIGHT + DEFAULT_BRANCH_EPS) / max(min_a, PARTITION_CLAMP_MIN))
            a_eff = (a_star * scale_c - DEFAULT_BRANCH_EPS).clamp_min(DEFAULT_POVM_INV_FLOOR)
            theta_0 = a_eff - (1.0 / (4.0 * a_eff))

            self.povm_scale_logits.data.zero_()
            self.povm_scale_logits.data[:v_eff].copy_(theta_0)
            self.initial_povm_logits.zero_()
            self.initial_povm_logits[:v_eff].copy_(theta_0)
        else:
            self.povm_scale_logits.data.zero_()
            self.initial_povm_logits.zero_()

        w_final = self.compute_povm_weights().float()
        p_bar_fin = (w_final * bar_e).clamp_min(PARTITION_CLAMP_MIN)
        p_bar_fin = p_bar_fin / p_bar_fin.sum().clamp_min(PARTITION_CLAMP_MIN)
        kl_final = float(max(0.0, float((p_target * (torch.log(p_target) - torch.log(p_bar_fin))).sum().item())))

        return {
            "pre_procrustes_carrier_fidelity": round(pre_fidelity, 6),
            "carrier_alignment_fidelity": round(post_fidelity, 6),
            "unitary_frobenius_err": unitary_frob_err,
            "uncalibrated_kl_to_target_unigram": round(kl_unif, 6),
            "step0_kl_to_target_unigram": round(kl_final, 6),
            "carrier_rho_max": round(rho_eff, 4),
            "calibrate_povm": float(1.0 if calibrate_povm else 0.0),
            "sharp_eff_rank": self.compute_sharp_eff_rank(),
            "mean_factor_rank": self.compute_mean_factor_rank(),
            "cross_branch_overlap": self.compute_cross_branch_overlap(),
        }

    @torch.no_grad()
    def load_100m_calibrated_readout(
        self, data_dir: str | Path = ".cache/fineweb_100M", max_tokens: int = 100_000_000,
        carrier_rho_max: float | None = None, calibrate_povm: bool = True, seed: int = 1337,
    ) -> dict[str, float]:
        """Load up to `100M` FineWeb tokens and run `calibrate_readout_to_random_attention_unigram` (`<= 8` args)."""
        d_path = Path(data_dir)
        shard_path = next((d_path / n for n in ("train_full_100M.bin", "fineweb_train_000001.bin") if (d_path / n).exists()), None)
        if shard_path is None:
            raise FileNotFoundError(f"Could not find train_full_100M.bin or fineweb_train_000001.bin in {d_path}")
        tokens = _load_header_safe_shard_tokens(shard_path, max_tokens=int(max_tokens))
        return self.calibrate_readout_to_random_attention_unigram(
            token_stream=tokens, carrier_rho_max=carrier_rho_max, calibrate_povm=calibrate_povm, seed=seed,
        )

    # Shared 100% polynomial factor/meaning/rank/backbone helpers from KroneckerMultiMeaning16x4GPT
    compute_branch_alpha = KroneckerMultiMeaning16x4GPT.compute_branch_alpha
    compute_context_branch_beta = KroneckerMultiMeaning16x4GPT.compute_context_branch_beta
    get_tied_vocab_factors = KroneckerMultiMeaning16x4GPT.get_tied_vocab_factors
    apply_readout_factor_operator = KroneckerMultiMeaning16x4GPT.apply_readout_factor_operator
    compute_sharp_eff_rank = KroneckerMultiMeaning16x4GPT.compute_sharp_eff_rank
    compute_mean_factor_rank = KroneckerMultiMeaning16x4GPT.compute_mean_factor_rank
    compute_cross_branch_overlap = KroneckerMultiMeaning16x4GPT.compute_cross_branch_overlap
    _pack_token_wavefunctions = KroneckerMultiMeaning16x4GPT._pack_token_wavefunctions
    extract_context_readout_states = KroneckerMultiMeaning16x4GPT.extract_context_readout_states

    @torch.no_grad()
    def compute_povm_drift_std(self) -> float:
        """Compute `std_v(theta_{povm,v} - theta_{povm,v}^{(0)})` measuring active POVM adaptation."""
        v_eff = int(self.cfg.valid_vocab_size)
        diff = self.povm_scale_logits[:v_eff].detach().float() - self.initial_povm_logits[:v_eff].detach().float()
        return float(diff.std().item())

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None,
        return_logits: bool | None = None, token_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Run `L` complex register + `16x4` Kronecker mixer layers and Calibrated Kronecker Born readout (`<= 8` args)."""
        b_sz, seq_t = idx.shape
        r_num, m_num, d_2 = int(self.cfg.num_meanings), int(self.cfg.num_factors), 2 * int(self.cfg.factor_dim)
        v_eff = int(self.cfg.valid_vocab_size)

        phi_factors, branch_beta, branch_alpha = self.extract_context_readout_states(idx)
        psi_factors = self.get_tied_vocab_factors()
        povm_w = self.compute_povm_weights()
        hyperparams = torch.tensor(
            [self.cfg.boson_p, self.cfg.holo_alpha, self.cfg.kron_blend, self.cfg.coinc_gamma],
            device=idx.device, dtype=torch.float32,
        )
        want_logits = (targets is None) if return_logits is None else bool(return_logits)

        if targets is not None and not want_logits:
            valid_targets = targets.clamp(min=0, max=v_eff - 1).reshape(-1)
            CalibratedKroneckerBornChunkedNLLFn.vocab_chunk_size = int(self.cfg.vocab_chunk_size)
            loss = CalibratedKroneckerBornChunkedNLLFn.apply(
                phi_factors, psi_factors, branch_beta, branch_alpha, povm_w, valid_targets, hyperparams, token_weights,
            )
            return (loss if return_logits is None else None), loss

        probs = eval_calibrated_kronecker_born_probs(
            phi_factors.reshape(b_sz, seq_t, r_num, m_num, d_2), psi_factors,
            branch_beta.reshape(b_sz, seq_t, r_num), branch_alpha, povm_w, hyperparams,
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
                loss = ((tw_f / tw_f.sum().clamp_min(PROB_LOG_CLAMP_MIN)) * token_nll).sum()

        out_logits = log_probs if want_logits else (loss if (targets is not None and return_logits is None) else None)
        return out_logits, loss


def compute_calibrated_kronecker_penalties(
    model: CalibratedKroneckerBornGPT,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute 16D Factor-Sphere, Complex-Unitary, Isotropy, Branch-Orthogonality, & POVM-Anchor penalties (`<= 8` args)."""
    v_eff = int(model.cfg.valid_vocab_size)
    base_penalty, stats = compute_kronecker_manifold_and_isotropy_penalties(model)  # type: ignore[arg-type]
    povm_diff = model.povm_scale_logits[:v_eff].float() - model.initial_povm_logits[:v_eff].float()
    povm_anchor_pen = povm_diff.square().mean()
    total_penalty = base_penalty + float(model.cfg.povm_anchor_weight) * povm_anchor_pen
    stats["povm_anchor_pen"] = float(povm_anchor_pen.detach().item())
    stats["total_pen"] = float(total_penalty.detach().item())
    stats["povm_drift_std"] = float(povm_diff.detach().std().item())
    return total_penalty, stats


def build_calibrated_kronecker_optimizers(
    model: CalibratedKroneckerBornGPT, base_muon_lr: float = DEFAULT_MUON_LR,
    base_embed_lr: float = DEFAULT_EMBED_LR, base_povm_lr: float = DEFAULT_POVM_LR,
    base_gate_lr: float = DEFAULT_GATE_LR, weight_decay: float = 0.0,
    use_joint_complex_muon: bool = True, batch_tokens: int = DEFAULT_MUON_REFERENCE_BATCH_TOKENS,
) -> tuple[StiefelMuon, torch.optim.AdamW]:
    """Build universal complex `StiefelMuon` (`d >= 2`) + 3-group `AdamW` (embeddings, `povm_scale_logits`, gates) (`<= 8` args)."""
    d_real = float(max(64, model.cfg.real_equiv_dim))
    d_ratio = 256.0 / d_real
    depth_scale = math.sqrt(6.0 / float(max(6, model.cfg.num_layers)))
    if d_real >= 256.0:
        muon_scale, embed_scale, gate_scale = (d_ratio ** 0.15) * depth_scale, d_ratio ** 0.50, math.sqrt(d_ratio)
    else:
        muon_scale = min(1.30, d_ratio ** 0.15) * depth_scale
        embed_scale, gate_scale = min(1.45, d_ratio ** 0.30), min(1.35, d_ratio ** 0.25)

    b_scale = compute_large_batch_muon_lr_scale(batch_tokens=int(batch_tokens))
    partition = classify_kronecker_model_parameters(model, min_operator_dim=MIN_KRONECKER_MUON_OPERATOR_DIM)
    povm_ids = {id(p) for n, p in model.named_parameters() if p.requires_grad and (n == "povm_scale_logits" or "povm_scale" in n)}
    povm_params = [p for p in partition.gate_params if id(p) in povm_ids]
    gate_params = [p for p in partition.gate_params if id(p) not in povm_ids]

    muon_lr = float(base_muon_lr) * muon_scale * b_scale
    embed_lr = float(base_embed_lr) * embed_scale * b_scale
    povm_lr = float(base_povm_lr)
    gate_lr = float(base_gate_lr) * gate_scale

    muon_kwargs: dict[str, Any] = {
        "lr": muon_lr, "momentum": 0.95, "polar_method": "polar_express",
        "stiefel_penalty": DEFAULT_KRONECKER_MUON_STIEFEL_PENALTY,
        "retract_every": 1, "use_normuon": True, "use_twin_rail": True,
    }
    if use_joint_complex_muon:
        try:
            opt_muon = StiefelMuon(
                [{"params": partition.muon_params, "lr": muon_lr, "base_lr": muon_lr}],
                complex_pairs=partition.muon_complex_pairs, **muon_kwargs,
            )
        except TypeError:
            opt_muon = StiefelMuon([{"params": partition.muon_params, "lr": muon_lr, "base_lr": muon_lr}], **muon_kwargs)
    else:
        opt_muon = StiefelMuon([{"params": partition.muon_params, "lr": muon_lr, "base_lr": muon_lr}], **muon_kwargs)

    opt_adam = torch.optim.AdamW(
        [
            {"params": partition.embed_params, "lr": embed_lr, "base_lr": embed_lr, "weight_decay": 0.0},
            {"params": povm_params, "lr": povm_lr, "base_lr": povm_lr, "weight_decay": 0.0},
            {"params": gate_params, "lr": gate_lr, "base_lr": gate_lr, "weight_decay": float(weight_decay)},
        ],
        betas=(0.9, 0.95), eps=1e-8,
    )
    return opt_muon, opt_adam


__all__ = [
    "CalibratedKroneckerBornChunkedNLLFn", "CalibratedKroneckerBornGPT", "CalibratedKroneckerConfig",
    "DEFAULT_CALIBRATION_BATCHES", "DEFAULT_CALIBRATION_BATCH_SIZE", "DEFAULT_CALIBRATION_MAX_EVAL_TOKENS",
    "DEFAULT_CALIBRATION_POLISH_STEPS", "DEFAULT_CALIBRATION_SEQ_LEN", "DEFAULT_CAL_BOSON_P",
    "DEFAULT_CAL_COINC_GAMMA", "DEFAULT_CAL_HOLO_ALPHA", "DEFAULT_CAL_KRON_BLEND", "DEFAULT_CAL_UNIGRAM_TEMP",
    "DEFAULT_CARRIER_AMP", "DEFAULT_CARRIER_RHO_MAX", "DEFAULT_CARRIER_STD_DIVISOR", "DEFAULT_EMBED_LR",
    "DEFAULT_FACTOR_DIM", "DEFAULT_GATE_LR", "DEFAULT_ISO_PENALTY_WEIGHT", "DEFAULT_MUON_LR",
    "DEFAULT_NUM_FACTORS", "DEFAULT_NUM_MEANINGS", "DEFAULT_POVM_ANCHOR_WEIGHT", "DEFAULT_POVM_INV_FLOOR",
    "DEFAULT_POVM_LR", "DEFAULT_POVM_MIN_EFF_WEIGHT", "DEFAULT_REGISTER_DIM",
    "DEFAULT_SPHERE_PENALTY_WEIGHT", "DEFAULT_UNITARY_PENALTY_WEIGHT", "DEFAULT_VOCAB_CHUNK_SIZE",
    "build_calibrated_kronecker_optimizers", "compute_calibrated_kronecker_penalties",
    "eval_calibrated_kronecker_born_probs",
]

