"""Multi-Head Stinespring Mixed-State `16^4` Kronecker Born GPT (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

When to Use This Building Block:
- Use `MultiHeadMixedStateKroneckerGPT`, `MultiHeadMixedStateKroneckerConfig`,
  `MultiHeadMixedStateKroneckerBlock`, `MixedStateKroneckerBornChunkedNLLFn`,
  `apply_carrier_orthogonal_complex_self_echo`, `normalize_raw_phi_per_factor_r_invariant`,
  `compute_phase_sensitive_branch_beta`, and `build_multihead_mixed_state_kronecker_optimizers`
  when training pure-complex Rank-`R` `16^4 = 65,536` Kronecker wavefunction language models on
  `900M`-token `FineWeb` at sequence lengths `T in {512, 1024}` with:
  1. **Track 3 — Carrier-Orthogonal Complex Self-Echo Projection (`apply_carrier_orthogonal_complex_self_echo`)**:
     Projects the current input token `x_t`'s factor ket `|p_0^{(r,m)}>` onto the 15D carrier-orthogonal
     subspace `H_perp = {|u> in C^16 : <c_ref | u> = 0}` and removes the full complex Hermitian overlap
     `<u_{0,perp}^{(r,m)} | op^{(r,m)}>_C = ov_re + i ov_im` with learned algebraic gate `lambda_{r,m} in (0, 1)`,
     leaving `<c_ref | raw_phi^{(r,m)}>_C == <c_ref | op^{(r,m)}>_C` strictly invariant.
  2. **Track 4 — Non-Diluted Co-Rotated Polysemy Branch Initialization, `R`-Invariant Factor Norm, &
     Phase-Sensitive Context Router (`normalize_raw_phi_per_factor_r_invariant` & `compute_phase_sensitive_branch_beta`)**:
     Eliminates the legacy `0.22x` (`98.1%` noise) secondary branch suppression (`secondary_branch_scale = 1.0`),
     normalizes `raw_phi` per `(r, m)` factor so the carrier-to-N-gram amplitude ratio is invariant across
     `R in {2, 3, 4}`, and routes polysemy branches via a phase-sensitive complex directional router
     initialized to exact uniform `beta_{t,r}^{(0)} = 1 / sqrt(R)`.
  3. **Track 6 — Multi-Head Stinespring Superposition Attention & Incoherent Mixed-State Density Readout
     (`MultiHeadSuperpositionMixedStateAttention` & `MixedStateKroneckerBornChunkedNLLFn`)**:
     Replaces the `11` `(T x T)` intra-head Krylov GEMMs per layer with `3`-matmul single-pass causal Born
     attention + trace-1 `K x K` cross-head Hermitian Gram coupling + Stinespring `C^{K x R}` projection,
     and evaluates trace-1 rank-`R` mixed-state Born intensities `I_{n,c}^{mixed} = sum_r beta_{n,r}^2 I_{r,n,c}`
     with a single-pass analytical backward pullback (`zero` duplicate `torch.pow` calls).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..layers.multihead_superposition_mixed_state_attn import (
    DEFAULT_ATTN_EVOLUTION_MODE, DEFAULT_FACTOR_HEAD_DIM, DEFAULT_GRAM_COUPLING_ETA_INIT,
    DEFAULT_MAX_SEQ_LEN, DEFAULT_SUBHEADS_PER_FACTOR, SUBHEAD_KET_NORM_EPS, VALID_ATTN_EVOLUTION_MODES,
    MultiHeadSuperpositionAttnConfig, MultiHeadSuperpositionMixedStateAttention,
)
from ..layers.packed_complex_register_density_attn import ComplexLinear, complex_l2_normalize
from ..optim.muon import StiefelMuon
from .calibrated_kronecker_born_gpt import (
    CalibratedKroneckerBornChunkedNLLFn, DEFAULT_CALIBRATION_MAX_EVAL_TOKENS,
    DEFAULT_CARRIER_STD_DIVISOR, _build_canonical_carrier_ket, eval_calibrated_kronecker_born_probs,
)
from .gemini_packed_complex_rope_gpt import ComplexUnitaryInterferenceFFN
from .kronecker_multimeaning_16x4_gpt import (
    DEFAULT_BRANCH_EPS, DEFAULT_FFN_EXPANSION, DEFAULT_KRON_EPS, DEFAULT_PSI0_GATE_INIT,
    DEFAULT_REF_PHASE_COS_SIN, DEFAULT_RESIDUAL_INIT, MAX_COMPLEX_MUON_DIM, PARTITION_CLAMP_MIN,
    PROB_LOG_CLAMP_MIN, Kronecker16x4UnitaryMeaningMixer, _batch_complex_unitary_terms,
    _compute_kronecker_chunk_intensity, _init_square_complex_linear_haar,
    _sample_haar_complex_unitary_batch, compute_kronecker_manifold_and_isotropy_penalties,
)
from .ngram_calibrated_kronecker_born_gpt import (
    DEFAULT_EMBED_LR, DEFAULT_GATE_LR, DEFAULT_MUON_LR, DEFAULT_NGRAM_BOSON_P,
    DEFAULT_NGRAM_COINC_GAMMA, DEFAULT_NGRAM_GATE_BI_INIT, DEFAULT_NGRAM_GATE_OP_INIT,
    DEFAULT_NGRAM_GATE_TRI_INIT, DEFAULT_NGRAM_HOLO_ALPHA, DEFAULT_NGRAM_KRON_BLEND,
    DEFAULT_POVM_LR, DEFAULT_SMEAR_GATE_INIT, DEFAULT_VOCAB_CHUNK_SIZE,
    Ngram900MCalibrationData, NgramCalibratedKroneckerBornGPT, NgramCalibratedKroneckerConfig,
    _apply_factor_unitary_bank, _deterministic_orthogonal_matrix, _get_helmert_zero_sum_basis,
    embed_real30_to_carrier_ortho_c16,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_059.md (Tracks 3, 4, & 6 Constants Ledger)
# ==============================================================================
# Track 4: Non-Diluted Polysemy Branch Init, R-Invariance, & Phase-Sensitive Router
DEFAULT_NON_DILUTED_SECONDARY_BRANCH_SCALE: float = 1.0
LEGACY_DILUTED_SECONDARY_BRANCH_SCALE: float = 0.22
DEFAULT_BRANCH_ROUTER_MODE: str = "phase_sensitive"
VALID_BRANCH_ROUTER_MODES: tuple[str, ...] = ("phase_sensitive", "legacy_norm_only")
REFERENCE_NGRAM_CALIBRATION_BRANCH_COUNT: int = 2
REFERENCE_RAW_PHI_FACTOR_SCALE: float = 0.3535533905932738  # 1.0 / math.sqrt(4 * 2)
ROUTER_DIRECTIONAL_SCORE_NORM: float = 8.0                  # math.sqrt(4 * 16)
ROUTER_ALGEBRAIC_POS_HALF: float = 0.5
ROUTER_ALGEBRAIC_POS_ONE: float = 1.0
ROUTER_BETA_NORM_EPS: float = 1e-8
ROUTER_HARMONIC_GAIN_SLOPE: float = 0.5
ROUTER_HARMONIC_PHASE_SCALE: float = 0.5

# Track 3: Carrier-Orthogonal Complex Self-Echo Projection
DEFAULT_SELF_ECHO_MODE: str = "ortho_complex_learned"
VALID_SELF_ECHO_MODES: tuple[str, ...] = (
    "legacy_ambient_real", "no_self_echo", "ortho_complex_full",
    "ortho_complex_learned", "carrier_ortho_complex", "fixed_carrier_ortho",
)
DEFAULT_SELF_ECHO_LAMBDA_INIT: float = 0.50
DEFAULT_ECHO_GATE_LOGIT_INIT: float = 0.0
SELF_ECHO_NORM_EPS: float = 1e-8
ALGEBRAIC_GATE_DENOM_ONE: float = 1.0
ALGEBRAIC_GATE_HALF: float = 0.5

# Track 6: Incoherent Mixed-State Density Readout & Optimizer Routing
DEFAULT_READOUT_DENSITY_MODE: str = "mixed_density"
VALID_READOUT_DENSITY_MODES: tuple[str, ...] = ("mixed_density", "coherent_pure", "coherent_superposition")
DEFAULT_MIXED_STATE_INTENSITY_EPS: float = 1e-6
MIN_STIEFEL_MUON_MATRIX_DIM: int = 2
DEFAULT_W44_STIEFEL_PENALTY: float = 0.01
BRANCH_ROTATION_SEED_BASE: int = 2000
BRANCH_ROTATION_SEED_STRIDE: int = 17


def _lambda_to_algebraic_logit(lam: float) -> float:
    """Invert `g(u) = 0.5 * (1 + u / sqrt(1 + u^2)) = lam` to `u = (2*lam - 1) / sqrt(1 - (2*lam - 1)^2)`."""
    clamped = min(max(float(lam), 1e-4), 1.0 - 1e-4)
    s = 2.0 * clamped - 1.0
    return s / math.sqrt(max(1e-8, 1.0 - s * s))


def apply_carrier_orthogonal_complex_self_echo(
    op_re: torch.Tensor, op_im: torch.Tensor, p0_re: torch.Tensor, p0_im: torch.Tensor,
    self_echo_mode: str = DEFAULT_SELF_ECHO_MODE,
    echo_gate_logits: torch.Tensor | float | None = None,
    eps: float = SELF_ECHO_NORM_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project `p0` onto `H_perp subset C^16` and attenuate complex Hermitian self-echo (`<= 8` args).

    Guarantees exact carrier invariance `<c_ref | raw_phi^{(r,m)}>_C == <c_ref | op^{(r,m)}>_C`
    for all `lambda_{r,m} in [0, 1]` while scaling the `H_perp` overlap to
    `(1 - lambda_{r,m}) <u_{0,perp}^{(r,m)} | op^{(r,m)}>_C`.
    """
    if self_echo_mode not in VALID_SELF_ECHO_MODES:
        raise ValueError(f"Unknown self_echo_mode {self_echo_mode!r}; expected one of {VALID_SELF_ECHO_MODES}")
    if self_echo_mode == "no_self_echo":
        return op_re, op_im

    orig_dtype = op_re.dtype
    op_r, op_i = op_re.float(), op_im.float()
    p0_r, p0_i = p0_re.float(), p0_im.float()

    if self_echo_mode == "legacy_ambient_real":
        reduce_dims = (-3, -2, -1) if op_r.ndim >= 4 else (-1,)
        echo_proj = (p0_r * op_r + p0_i * op_i).sum(dim=reduce_dims, keepdim=True)
        return (op_r - echo_proj * p0_r).to(dtype=orig_dtype), (op_i - echo_proj * p0_i).to(dtype=orig_dtype)

    # Project p0 onto the 15D carrier-orthogonal subspace H_perp (zero-mean along d_f=16)
    u0_perp_re = p0_r - p0_r.mean(dim=-1, keepdim=True)
    u0_perp_im = p0_i - p0_i.mean(dim=-1, keepdim=True)
    u0_norm = torch.sqrt((u0_perp_re.square() + u0_perp_im.square()).sum(dim=-1, keepdim=True) + float(eps))
    u0_hat_re, u0_hat_im = u0_perp_re / u0_norm, u0_perp_im / u0_norm

    # Full complex Hermitian inner product <u0_hat | op>_C = ov_re + i * ov_im
    ov_re = (u0_hat_re * op_r + u0_hat_im * op_i).sum(dim=-1, keepdim=True)
    ov_im = (u0_hat_re * op_i - u0_hat_im * op_r).sum(dim=-1, keepdim=True)

    # Rank-1 projection ket (ov_re + i * ov_im) |u0_hat> in H_perp
    proj_re = ov_re * u0_hat_re - ov_im * u0_hat_im
    proj_im = ov_re * u0_hat_im + ov_im * u0_hat_re

    if self_echo_mode in ("ortho_complex_full", "fixed_carrier_ortho"):
        lam_s = float(echo_gate_logits) if isinstance(echo_gate_logits, (int, float)) else 1.0
        return (op_r - lam_s * proj_re).to(dtype=orig_dtype), (op_i - lam_s * proj_im).to(dtype=orig_dtype)

    if echo_gate_logits is None or isinstance(echo_gate_logits, (int, float)):
        lam_val = DEFAULT_SELF_ECHO_LAMBDA_INIT if echo_gate_logits is None else float(echo_gate_logits)
        return (op_r - lam_val * proj_re).to(dtype=orig_dtype), (op_i - lam_val * proj_im).to(dtype=orig_dtype)

    u_gate = echo_gate_logits.float()
    lam_gate = ALGEBRAIC_GATE_HALF * (u_gate * torch.rsqrt(ALGEBRAIC_GATE_DENOM_ONE + u_gate * u_gate) + ALGEBRAIC_GATE_DENOM_ONE)
    if op_r.ndim >= 4 and lam_gate.ndim == 2:
        lam_gate = lam_gate.view(*([1] * (op_r.ndim - 3)), op_r.size(-3), op_r.size(-2), 1)
    else:
        while lam_gate.ndim < op_r.ndim:
            lam_gate = lam_gate.unsqueeze(-1)
    return (op_r - lam_gate * proj_re).to(dtype=orig_dtype), (op_i - lam_gate * proj_im).to(dtype=orig_dtype)


def normalize_raw_phi_per_factor_r_invariant(
    raw_phi_re: torch.Tensor, raw_phi_im: torch.Tensor,
    eps: float = ROUTER_BETA_NORM_EPS, reference_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize each `(r, m)` `C^16` factor slice of `(raw_phi_re, raw_phi_im)` so norm is `R`-invariant (`<= 8` args).

    Eliminates the `1 / sqrt(4 * R)` norm dilution when scaling `R in {2, 3, 4}` by normalizing
    each `16`-dimensional complex factor independently along `dim=-1`.
    """
    orig_shape, orig_dtype = raw_phi_re.shape, raw_phi_re.dtype
    if raw_phi_re.ndim < 4 and orig_shape[-1] % DEFAULT_FACTOR_HEAD_DIM == 0:
        r_re = raw_phi_re.reshape(*orig_shape[:-1], -1, DEFAULT_FACTOR_HEAD_DIM).float()
        r_im = raw_phi_im.reshape(*orig_shape[:-1], -1, DEFAULT_FACTOR_HEAD_DIM).float()
        scale = float(reference_scale) / torch.sqrt((r_re.square() + r_im.square()).sum(dim=-1, keepdim=True) + float(eps))
        return (r_re * scale).reshape(orig_shape).to(dtype=orig_dtype), (r_im * scale).reshape(orig_shape).to(dtype=orig_dtype)

    r_re, r_im = raw_phi_re.float(), raw_phi_im.float()
    scale = float(reference_scale) / torch.sqrt((r_re.square() + r_im.square()).sum(dim=-1, keepdim=True) + float(eps))
    return (r_re * scale).to(dtype=orig_dtype), (r_im * scale).to(dtype=orig_dtype)


def compute_phase_sensitive_branch_beta(
    psi_re: torch.Tensor, psi_im: torch.Tensor, psi0_re: torch.Tensor, psi0_im: torch.Tensor,
    gate_w_re: torch.Tensor, gate_w_im: torch.Tensor,
    gate_b: torch.Tensor | None = None, eps: float = ROUTER_BETA_NORM_EPS,
) -> torch.Tensor:
    """Compute 100% `exp`-free phase-sensitive context branch amplitudes `beta` of shape `(..., R)` (`<= 8` args).

    Supports both:
    1. Canonical 8-argument call `(psi_re, psi_im, psi0_re, psi0_im, gate_w_re, gate_w_im, gate_b, eps)`, and
    2. 6/7-argument call `(phi_re, phi_im, router_w_re, router_w_im, router_gain, router_bias, eps)`.
    At Step 0 when `gate_w_re == 0`, `gate_w_im == 0`, and `gate_b == 0`, returns exact uniform
    `beta_{t,r} = 1 / sqrt(R)`. When `gate_w_re/im != 0`, differentiates `+1`, `-1`, `+i`, and `-i`
    complex phases via both direct state projection and relative Hermitian phase overlap `psi * conj(psi0)`.
    """
    orig_dtype = psi_re.dtype
    if gate_w_re.ndim == 1 and psi0_re.shape != psi_re.shape:
        w_re_raw, w_im_raw = psi0_re.float(), psi0_im.float()
        r_count = int(w_re_raw.size(0))
        gain_vec, bias_vec = gate_w_re.float().view(1, r_count), gate_w_im.float().view(1, r_count)
        eff_eps = float(gate_b) if isinstance(gate_b, (int, float)) else float(eps)
        lead_shape = psi_re.shape[:-3] if psi_re.ndim >= 4 else psi_re.shape[:-1]
        pr = psi_re.float().reshape(-1, r_count, w_re_raw[0].numel())
        pi = psi_im.float().reshape(-1, r_count, w_im_raw[0].numel())
        p0_r, p0_i = torch.zeros_like(pr), torch.zeros_like(pi)
        w_re, w_im = w_re_raw.reshape(r_count, -1), w_im_raw.reshape(r_count, -1)
    else:
        r_count = int(gate_w_re.size(0))
        w_re, w_im = gate_w_re.float().reshape(r_count, -1), gate_w_im.float().reshape(r_count, -1)
        d_branch = int(w_re.size(1))
        lead_shape = psi_re.shape[:-3] if (psi_re.ndim >= 4 and psi_re.size(-3) == r_count) else psi_re.shape[:-1]
        pr, pi = psi_re.float().reshape(-1, r_count, d_branch), psi_im.float().reshape(-1, r_count, d_branch)
        p0_r, p0_i = psi0_re.float().reshape(-1, r_count, d_branch), psi0_im.float().reshape(-1, r_count, d_branch)
        gain_vec = torch.ones((1, r_count), device=psi_re.device, dtype=torch.float32)
        bias_vec = gate_b.float().view(1, r_count) if isinstance(gate_b, torch.Tensor) else torch.zeros((1, r_count), device=psi_re.device, dtype=torch.float32)
        eff_eps = float(eps)

    # Combine state (pr, pi) with relative Hermitian phase product psi * conj(psi0)
    feat_re = pr + (pr * p0_r + pi * p0_i)
    feat_im = pi + (pi * p0_r - pr * p0_i)

    # Branch-harmonic complex preconditioning breaks inter-branch symmetry while remaining zero at Step 0
    r_idx = torch.arange(r_count, device=psi_re.device, dtype=torch.float32).unsqueeze(1)
    r_denom = float(max(1, r_count))
    theta_r = (ROUTER_HARMONIC_PHASE_SCALE * math.pi) * (r_idx / r_denom)
    kappa_r = 1.0 + ROUTER_HARMONIC_GAIN_SLOPE * (r_idx / r_denom)
    cos_r, sin_r = torch.cos(theta_r), torch.sin(theta_r)
    w_eff_re = (kappa_r * (cos_r * w_re - sin_r * w_im)).unsqueeze(0)
    w_eff_im = (kappa_r * (sin_r * w_re + cos_r * w_im)).unsqueeze(0)

    d_norm = math.sqrt(float(max(1, feat_re.size(-1))))
    dir_score = (w_eff_re * feat_re + w_eff_im * feat_im).sum(dim=-1) / d_norm
    u_nr = gain_vec * dir_score + bias_vec

    raw_b = ROUTER_ALGEBRAIC_POS_HALF * (u_nr + torch.sqrt(u_nr * u_nr + ROUTER_ALGEBRAIC_POS_ONE))
    beta = (raw_b / torch.sqrt(raw_b.square().sum(dim=-1, keepdim=True) + eff_eps)).to(dtype=orig_dtype)
    return beta.reshape(*lead_shape, r_count) if lead_shape else beta.squeeze(0)


def _compute_mixed_state_kronecker_chunk_intensity(
    x_ov_or_z_cat: torch.Tensor, y_ov_or_beta: torch.Tensor,
    branch_beta_or_alpha: torch.Tensor, alpha_chunk_or_coeffs: torch.Tensor | tuple[float, ...],
    holo_alpha: float = DEFAULT_NGRAM_HOLO_ALPHA, kron_blend: float = DEFAULT_NGRAM_KRON_BLEND,
    coinc_gamma: float = DEFAULT_NGRAM_COINC_GAMMA, eps_floor: float = DEFAULT_MIXED_STATE_INTENSITY_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute trace-1 incoherent mixed-state 4-factor Kronecker Born intensity `I_tv^{mixed}` (`<= 8` args).

    Evaluates per-branch holographic + coincidence intensity:
      `I_{r,n,c} = (s_0 + holo_alpha * a_{r,c} M_{re,r,n,c})^2`
                `+ (s_0 + holo_alpha * a_{r,c} M_{im,r,n,c})^2`
                `+ coinc_gamma * a_{r,c}^2 (K_{re,r,n,c}^2 + K_{im,r,n,c}^2)`
    where `a_{r,c} = sqrt(R) * alpha_{c,r}`, and forms the trace-1 mixed-state expectation:
      `I_{n,c}^{mixed} = sum_{r=0}^{R-1} beta_{n,r}^2 I_{r,n,c} + eps_floor`.
    """
    if isinstance(alpha_chunk_or_coeffs, (tuple, list)) or x_ov_or_z_cat.ndim == 3:
        z_f, beta_f, alpha_c = x_ov_or_z_cat.float(), y_ov_or_beta.float(), branch_beta_or_alpha.float()
        coeffs = tuple(alpha_chunk_or_coeffs) if isinstance(alpha_chunk_or_coeffs, (tuple, list)) else (float(holo_alpha), float(kron_blend), float(coinc_gamma))
        h_alpha, k_blend, c_gamma = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        e_floor = float(coeffs[3]) if len(coeffs) >= 4 else float(eps_floor)
        n_tok, r_num = beta_f.shape
        vc = alpha_c.size(0)
        x_ov, y_ov = z_f[:, :n_tok, :].reshape(r_num, 4, n_tok, vc), z_f[:, n_tok:, :].reshape(r_num, 4, n_tok, vc)
    else:
        x_ov, y_ov = x_ov_or_z_cat.float(), y_ov_or_beta.float()
        beta_f, alpha_c = branch_beta_or_alpha.float(), alpha_chunk_or_coeffs.float()
        h_alpha, k_blend, c_gamma, e_floor = float(holo_alpha), float(kron_blend), float(coinc_gamma), float(eps_floor)
        r_num = x_ov.size(0)

    x_bar = 0.25 * (x_ov[:, 0] + x_ov[:, 1] + x_ov[:, 2] + x_ov[:, 3])
    y_bar = 0.25 * (y_ov[:, 0] + y_ov[:, 1] + y_ov[:, 2] + y_ov[:, 3])
    p01_re, p01_im = x_ov[:, 0] * x_ov[:, 1] - y_ov[:, 0] * y_ov[:, 1], x_ov[:, 0] * y_ov[:, 1] + y_ov[:, 0] * x_ov[:, 1]
    p23_re, p23_im = x_ov[:, 2] * x_ov[:, 3] - y_ov[:, 2] * y_ov[:, 3], x_ov[:, 2] * y_ov[:, 3] + y_ov[:, 2] * x_ov[:, 3]
    k_re, k_im = p01_re * p23_re - p01_im * p23_im, p01_re * p23_im + p01_im * p23_re

    m_re = (1.0 - k_blend) * x_bar + k_blend * k_re
    m_im = (1.0 - k_blend) * y_bar + k_blend * k_im
    s0 = DEFAULT_REF_PHASE_COS_SIN
    a_r = math.sqrt(float(r_num)) * alpha_c.t().unsqueeze(1)
    ref_re_r, ref_im_r = s0 + h_alpha * a_r * m_re, s0 + h_alpha * a_r * m_im
    i_r = ref_re_r.square() + ref_im_r.square() + c_gamma * a_r.square() * (k_re.square() + k_im.square())
    i_tv = (beta_f.t().square().unsqueeze(2) * i_r).sum(dim=0) + e_floor
    return i_tv, ref_re_r, ref_im_r


def _mixed_state_kronecker_chunk_fwd_step(
    z_cat: torch.Tensor, beta_f: torch.Tensor, alpha_c: torch.Tensor, povm_c: torch.Tensor,
    rel_idx: torch.Tensor, in_chunk_f: torch.Tensor, boson_p: float, coeffs: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-chunk mixed-state partition sum `z_chunk` and target `target_chunk` (`<= 8` args)."""
    n_tok, r_num = beta_f.shape
    vc = alpha_c.size(0)
    holo_alpha, kron_blend, coinc_gamma = coeffs
    z_f = z_cat.float()
    x_ov, y_ov = z_f[:, :n_tok, :].reshape(r_num, 4, n_tok, vc), z_f[:, n_tok:, :].reshape(r_num, 4, n_tok, vc)
    i_tv, _, _ = _compute_mixed_state_kronecker_chunk_intensity(x_ov, y_ov, beta_f, alpha_c, holo_alpha, kron_blend, coinc_gamma)
    p_tilde_c = povm_c.unsqueeze(0) * torch.pow(i_tv, boson_p)
    return p_tilde_c.sum(dim=-1), in_chunk_f * p_tilde_c.gather(1, rel_idx).squeeze(1)


def _mixed_state_kronecker_chunk_bwd_step(
    z_cat: torch.Tensor, beta_f: torch.Tensor, alpha_c: torch.Tensor, povm_c: torch.Tensor,
    scale_inv_z: tuple[torch.Tensor, torch.Tensor], target_match: tuple[torch.Tensor, torch.Tensor],
    boson_p: float, coeffs: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute single-pass analytical mixed-state chunk pullback `(g_cat, d_beta_c, d_alpha_c, d_povm_c)` (`<= 8` args).

    Executes only a single `torch.pow(i_tv, boson_p)` call per vocabulary chunk and returns
    `d_povm_c = delta_tv.sum(dim=0) / povm_c.clamp_min(PROB_LOG_CLAMP_MIN)` in the same pass.
    """
    n_tok, r_num = beta_f.shape
    vc, k_tot = alpha_c.size(0), r_num * 4
    scale_t, inv_z = scale_inv_z
    v_idx, targets_col = target_match
    holo_alpha, kron_blend, coinc_gamma = coeffs
    s0, eta = DEFAULT_REF_PHASE_COS_SIN, float(kron_blend)
    one_minus_eta_quarter, sqrt_r = 0.25 * (1.0 - eta), math.sqrt(float(r_num))

    z_f = z_cat.float()
    x_ov, y_ov = z_f[:, :n_tok, :].reshape(r_num, 4, n_tok, vc), z_f[:, n_tok:, :].reshape(r_num, 4, n_tok, vc)
    x0, x1, x2, x3 = x_ov[:, 0], x_ov[:, 1], x_ov[:, 2], x_ov[:, 3]
    y0, y1, y2, y3 = y_ov[:, 0], y_ov[:, 1], y_ov[:, 2], y_ov[:, 3]
    x_bar, y_bar = 0.25 * (x0 + x1 + x2 + x3), 0.25 * (y0 + y1 + y2 + y3)

    p01_re, p01_im = x0 * x1 - y0 * y1, x0 * y1 + y0 * x1
    p23_re, p23_im = x2 * x3 - y2 * y3, x2 * y3 + y2 * x3
    k_re, k_im = p01_re * p23_re - p01_im * p23_im, p01_re * p23_im + p01_im * p23_re
    coinc_r = k_re.square() + k_im.square()
    m_re, m_im = (1.0 - eta) * x_bar + eta * k_re, (1.0 - eta) * y_bar + eta * k_im

    a_r = sqrt_r * alpha_c.t().unsqueeze(1)
    ref_re_r, ref_im_r = s0 + float(holo_alpha) * a_r * m_re, s0 + float(holo_alpha) * a_r * m_im
    i_r = ref_re_r.square() + ref_im_r.square() + float(coinc_gamma) * a_r.square() * coinc_r
    pi_rn = beta_f.t().square().unsqueeze(2)
    i_tv = (pi_rn * i_r).sum(dim=0) + DEFAULT_MIXED_STATE_INTENSITY_EPS

    probs_c = (povm_c.unsqueeze(0) * torch.pow(i_tv, boson_p)) * inv_z
    is_target = (v_idx == targets_col).to(dtype=torch.float32)
    delta_tv = scale_t * (probs_c - is_target)
    d_povm_c = delta_tv.sum(dim=0) / povm_c.clamp_min(PROB_LOG_CLAMP_MIN)

    g_i_b = (delta_tv * (float(boson_p) / i_tv)).unsqueeze(0)
    d_beta_c = 2.0 * beta_f * (g_i_b * i_r).sum(dim=2).t()

    w_gi = pi_rn * g_i_b
    d_ref_re, d_ref_im = 2.0 * w_gi * ref_re_r, 2.0 * w_gi * ref_im_r
    d_a_r = (float(holo_alpha) * (d_ref_re * m_re + d_ref_im * m_im) + 2.0 * float(coinc_gamma) * w_gi * a_r * coinc_r).sum(dim=1)
    d_alpha_c = (sqrt_r * d_a_r).t()

    g_m_re, g_m_im = float(holo_alpha) * a_r * d_ref_re, float(holo_alpha) * a_r * d_ref_im
    two_gc_a2 = 2.0 * float(coinc_gamma) * w_gi * a_r.square()
    g_k_re, g_k_im = eta * g_m_re + two_gc_a2 * k_re, eta * g_m_im + two_gc_a2 * k_im

    g_p01_re, g_p01_im = g_k_re * p23_re + g_k_im * p23_im, g_k_im * p23_re - g_k_re * p23_im
    g_p23_re, g_p23_im = g_k_re * p01_re + g_k_im * p01_im, g_k_im * p01_re - g_k_re * p01_im
    g_xbar, g_ybar = one_minus_eta_quarter * g_m_re, one_minus_eta_quarter * g_m_im

    g_x = torch.stack([
        g_xbar + g_p01_re * x1 + g_p01_im * y1, g_xbar + g_p01_re * x0 + g_p01_im * y0,
        g_xbar + g_p23_re * x3 + g_p23_im * y3, g_xbar + g_p23_re * x2 + g_p23_im * y2,
    ], dim=1).reshape(k_tot, n_tok, vc)
    g_y = torch.stack([
        g_ybar + g_p01_im * x1 - g_p01_re * y1, g_ybar + g_p01_im * x0 - g_p01_re * y0,
        g_ybar + g_p23_im * x3 - g_p23_re * y3, g_ybar + g_p23_im * x2 - g_p23_re * y2,
    ], dim=1).reshape(k_tot, n_tok, vc)
    return torch.cat([g_x, g_y], dim=1).to(dtype=z_cat.dtype), d_beta_c, d_alpha_c, d_povm_c


_compiled_mixed_state_chunk_fwd_step = torch.compile(_mixed_state_kronecker_chunk_fwd_step)
_compiled_mixed_state_chunk_bwd_step = torch.compile(_mixed_state_kronecker_chunk_bwd_step)


def eval_mixed_state_kronecker_born_probs(
    phi_factors: torch.Tensor, psi_factors: torch.Tensor, branch_beta: torch.Tensor,
    branch_alpha: torch.Tensor, povm_weights: torch.Tensor, hyperparams: torch.Tensor,
) -> torch.Tensor:
    """Evaluate 100% polynomial mixed-state `16^4` Kronecker Born probabilities (`<= 8` args)."""
    lead_shape = phi_factors.shape[:-3]
    r_num, m_num, d_2 = phi_factors.shape[-3:]
    d_f, v_eff = d_2 // 2, psi_factors.size(0)
    n_tok = int(math.prod(lead_shape)) if lead_shape else 1
    k_tot = r_num * m_num

    phi_f, psi_f = phi_factors.reshape(n_tok, r_num, m_num, d_2).float(), psi_factors.reshape(v_eff, r_num, m_num, d_2).float()
    beta_f, alpha_f, povm_f = branch_beta.reshape(n_tok, r_num).float(), branch_alpha.reshape(v_eff, r_num).float(), povm_weights[:v_eff].float()
    hp = hyperparams.float().reshape(-1)
    boson_p, holo_alpha, kron_blend, coinc_gamma = float(hp[0].item()), float(hp[1].item()), float(hp[2].item()), float(hp[3].item())

    phi_re, phi_im = phi_f[..., :d_f], phi_f[..., d_f:]
    phi_skew = torch.cat([-phi_im, phi_re], dim=-1)
    u_batched = torch.cat([phi_f.reshape(n_tok, k_tot, d_2).permute(1, 0, 2), phi_skew.reshape(n_tok, k_tot, d_2).permute(1, 0, 2)], dim=1).contiguous()
    w_all = psi_f.reshape(v_eff, k_tot, d_2).permute(1, 2, 0).contiguous()

    p_tilde = torch.empty((n_tok, v_eff), device=phi_factors.device, dtype=torch.float32)
    for v0 in range(0, v_eff, DEFAULT_VOCAB_CHUNK_SIZE):
        v1 = min(v_eff, v0 + DEFAULT_VOCAB_CHUNK_SIZE)
        vc = v1 - v0
        z_cat = torch.bmm(u_batched, w_all[:, :, v0:v1])
        x_ov, y_ov = z_cat[:, :n_tok, :].reshape(r_num, m_num, n_tok, vc), z_cat[:, n_tok:, :].reshape(r_num, m_num, n_tok, vc)
        i_tv, _, _ = _compute_mixed_state_kronecker_chunk_intensity(x_ov, y_ov, beta_f, alpha_f[v0:v1], holo_alpha, kron_blend, coinc_gamma)
        p_tilde[:, v0:v1] = povm_f[v0:v1].unsqueeze(0) * torch.pow(i_tv, boson_p)
    probs = p_tilde / p_tilde.sum(dim=-1, keepdim=True).clamp_min(PARTITION_CLAMP_MIN)
    return probs.reshape(*lead_shape, v_eff)


class MixedStateKroneckerBornChunkedNLLFn(torch.autograd.Function):
    """Vocabulary-chunked Mixed-State `16^4` Kronecker Born NLL with single-pass analytical backward."""

    vocab_chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE

    @staticmethod
    def forward(
        cls: Any, phi_factors: torch.Tensor, psi_factors: torch.Tensor,
        branch_beta: torch.Tensor, branch_alpha: torch.Tensor, povm_weights: torch.Tensor,
        targets: torch.Tensor, hyperparams: torch.Tensor | float = DEFAULT_NGRAM_BOSON_P,
        token_weights: torch.Tensor | tuple[float, ...] | None = None,
    ) -> torch.Tensor:
        """Compute chunked Mixed-State Kronecker Born NLL (`<= 8` args)."""
        n_tok, r_num, m_num, d_2 = phi_factors.shape
        d_f, v_eff, k_tot = d_2 // 2, psi_factors.size(0), r_num * m_num
        v_chunk = max(1, int(MixedStateKroneckerBornChunkedNLLFn.vocab_chunk_size))

        gemm_dtype = phi_factors.dtype
        phi_w, psi_w = phi_factors.contiguous(), psi_factors.to(dtype=gemm_dtype).contiguous()
        beta_f, alpha_f = branch_beta.float().contiguous(), branch_alpha.float().contiguous()
        povm_f = povm_weights[:v_eff].float().contiguous()
        targets_flat = targets.reshape(-1).long().clamp(0, v_eff - 1)

        if isinstance(hyperparams, torch.Tensor):
            hp = hyperparams.detach().float().reshape(-1)
            boson_p = float(hp[0].item())
            coeffs = (float(hp[1].item()), float(hp[2].item()), float(hp[3].item()))
            tw_tensor = token_weights if isinstance(token_weights, torch.Tensor) else None
        else:
            boson_p = float(hyperparams)
            if isinstance(token_weights, (tuple, list)):
                coeffs = (float(token_weights[0]), float(token_weights[1]), float(token_weights[2]))
                tw_tensor = None
            else:
                coeffs = (DEFAULT_NGRAM_HOLO_ALPHA, DEFAULT_NGRAM_KRON_BLEND, DEFAULT_NGRAM_COINC_GAMMA)
                tw_tensor = token_weights if isinstance(token_weights, torch.Tensor) else None

        if tw_tensor is None:
            w_norm = torch.full((n_tok,), 1.0 / float(max(1, n_tok)), device=phi_factors.device, dtype=torch.float32)
        else:
            tw_f = tw_tensor.reshape(-1).float()
            w_norm = tw_f / tw_f.sum().clamp_min(PROB_LOG_CLAMP_MIN)

        phi_re, phi_im = phi_w[..., :d_f], phi_w[..., d_f:]
        phi_skew = torch.cat([-phi_im, phi_re], dim=-1)
        u_batched = torch.cat([phi_w.reshape(n_tok, k_tot, d_2).permute(1, 0, 2), phi_skew.reshape(n_tok, k_tot, d_2).permute(1, 0, 2)], dim=1).contiguous()
        w_all = psi_w.reshape(v_eff, k_tot, d_2).permute(1, 2, 0).contiguous()

        z_t = torch.zeros((n_tok,), device=phi_factors.device, dtype=torch.float32)
        target_p_tilde = torch.zeros((n_tok,), device=phi_factors.device, dtype=torch.float32)
        use_compiled = bool(phi_factors.is_cuda and gemm_dtype == torch.bfloat16 and not torch.compiler.is_compiling())
        fwd_fn = _compiled_mixed_state_chunk_fwd_step if use_compiled else _mixed_state_kronecker_chunk_fwd_step

        for v0 in range(0, v_eff, v_chunk):
            v1 = min(v_eff, v0 + v_chunk)
            vc = v1 - v0
            z_cat = torch.bmm(u_batched, w_all[:, :, v0:v1])
            rel_idx = (targets_flat - v0).clamp(0, vc - 1).unsqueeze(1)
            in_chunk_f = ((targets_flat >= v0) & (targets_flat < v1)).to(dtype=torch.float32)
            z_chunk, target_chunk = fwd_fn(z_cat, beta_f, alpha_f[v0:v1], povm_f[v0:v1], rel_idx, in_chunk_f, boson_p, coeffs)
            z_t.add_(z_chunk)
            target_p_tilde.add_(target_chunk)
            del z_cat, z_chunk, target_chunk

        z_clamped = z_t.clamp_min(PARTITION_CLAMP_MIN)
        target_clamped = target_p_tilde.clamp_min(PARTITION_CLAMP_MIN)
        loss = (w_norm * (torch.log(z_clamped) - torch.log(target_clamped))).sum()

        cls.save_for_backward(u_batched, w_all, beta_f, alpha_f, povm_f, targets_flat, w_norm, z_clamped)
        cls.boson_p, cls.coeffs, cls.v_chunk = boson_p, coeffs, v_chunk
        cls.shape_meta = (n_tok, r_num, m_num, d_2, v_eff, int(povm_weights.size(0)))
        cls.dtypes = (phi_factors.dtype, psi_factors.dtype, branch_beta.dtype, branch_alpha.dtype, povm_weights.dtype)
        return loss

    @staticmethod
    def backward(
        cls: Any, grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        """Compute single-pass analytical backward pullback across all 5 differentiable inputs (`<= 8` args)."""
        u_batched, w_all, beta_f, alpha_f, povm_f, targets_flat, w_norm, z_clamped = cls.saved_tensors
        boson_p, coeffs, v_chunk = float(cls.boson_p), cls.coeffs, int(cls.v_chunk)
        n_tok, r_num, m_num, d_2, v_eff, v_povm_full = cls.shape_meta
        phi_dtype, psi_dtype, beta_dtype, alpha_dtype, povm_dtype = cls.dtypes
        d_f, k_tot, gemm_dtype = d_2 // 2, r_num * m_num, u_batched.dtype

        d_phi_re_k = torch.zeros((k_tot, n_tok, d_f), device=u_batched.device, dtype=torch.float32)
        d_phi_im_k = torch.zeros((k_tot, n_tok, d_f), device=u_batched.device, dtype=torch.float32)
        d_psi = torch.zeros((v_eff, r_num, m_num, d_2), device=w_all.device, dtype=torch.float32)
        d_beta = torch.zeros((n_tok, r_num), device=beta_f.device, dtype=torch.float32)
        d_alpha = torch.zeros((v_eff, r_num), device=alpha_f.device, dtype=torch.float32)
        d_povm = torch.zeros((v_povm_full,), device=povm_f.device, dtype=torch.float32)

        scale_inv_z = ((grad_out.float() * w_norm).unsqueeze(1), (1.0 / z_clamped).unsqueeze(1))
        targets_col = targets_flat.unsqueeze(1)
        use_compiled = bool(u_batched.is_cuda and gemm_dtype == torch.bfloat16 and not torch.compiler.is_compiling())
        bwd_fn = _compiled_mixed_state_chunk_bwd_step if use_compiled else _mixed_state_kronecker_chunk_bwd_step

        for v0 in range(0, v_eff, v_chunk):
            v1 = min(v_eff, v0 + v_chunk)
            vc = v1 - v0
            w_chunk = w_all[:, :, v0:v1]
            z_cat = torch.bmm(u_batched, w_chunk)
            v_idx = torch.arange(v0, v1, device=u_batched.device, dtype=torch.long).unsqueeze(0)
            g_cat, d_beta_c, d_alpha_c, d_povm_c = bwd_fn(
                z_cat, beta_f, alpha_f[v0:v1], povm_f[v0:v1], scale_inv_z, (v_idx, targets_col), boson_p, coeffs
            )
            del z_cat
            d_povm[v0:v1], d_alpha[v0:v1] = d_povm_c, d_alpha_c
            d_beta.add_(d_beta_c)
            del d_beta_c, d_alpha_c, d_povm_c

            d_w_chunk = torch.bmm(u_batched.transpose(1, 2), g_cat).float()
            d_psi[v0:v1] = d_w_chunk.permute(2, 0, 1).reshape(vc, r_num, m_num, d_2)
            del d_w_chunk

            d_u_chunk = torch.bmm(g_cat, w_chunk.transpose(1, 2)).float()
            del g_cat
            d_phi_re_k.add_(d_u_chunk[:, :n_tok, :d_f] + d_u_chunk[:, n_tok:, d_f:])
            d_phi_im_k.add_(d_u_chunk[:, :n_tok, d_f:] - d_u_chunk[:, n_tok:, :d_f])
            del d_u_chunk

        d_phi = torch.cat([d_phi_re_k, d_phi_im_k], dim=-1).permute(1, 0, 2).reshape(n_tok, r_num, m_num, d_2)
        return (
            d_phi.to(dtype=phi_dtype), d_psi.to(dtype=psi_dtype), d_beta.to(dtype=beta_dtype),
            d_alpha.to(dtype=alpha_dtype), d_povm.to(dtype=povm_dtype), None, None, None,
        )


@dataclass(frozen=True)
class MultiHeadMixedStateKroneckerConfig(NgramCalibratedKroneckerConfig):
    """Frozen configuration for `MultiHeadMixedStateKroneckerGPT` (`Wave 44` / `EXPERIMENT_PLAN_059.md`)."""

    d_register: int = DEFAULT_FACTOR_HEAD_DIM
    n_subheads_per_factor: int = DEFAULT_SUBHEADS_PER_FACTOR
    subheads_per_factor: int | None = None
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    attn_evolution_mode: str = DEFAULT_ATTN_EVOLUTION_MODE
    attn_variant: str | None = None
    gram_eta_init: float = DEFAULT_GRAM_COUPLING_ETA_INIT
    readout_density_mode: str = DEFAULT_READOUT_DENSITY_MODE
    self_echo_mode: str = DEFAULT_SELF_ECHO_MODE
    self_echo_lambda_init: float = DEFAULT_SELF_ECHO_LAMBDA_INIT
    secondary_branch_scale: float = DEFAULT_NON_DILUTED_SECONDARY_BRANCH_SCALE
    branch_router_mode: str = DEFAULT_BRANCH_ROUTER_MODE

    def __post_init__(self) -> None:
        eff_k = int(self.subheads_per_factor if self.subheads_per_factor is not None else self.n_subheads_per_factor)
        eff_attn = str(self.attn_variant if self.attn_variant is not None else self.attn_evolution_mode)
        if eff_attn not in VALID_ATTN_EVOLUTION_MODES:
            raise ValueError(f"Unknown attn_evolution_mode {eff_attn!r}; expected {VALID_ATTN_EVOLUTION_MODES}")
        if self.readout_density_mode not in VALID_READOUT_DENSITY_MODES:
            raise ValueError(f"Unknown readout_density_mode {self.readout_density_mode!r}; expected {VALID_READOUT_DENSITY_MODES}")
        if self.self_echo_mode not in VALID_SELF_ECHO_MODES:
            raise ValueError(f"Unknown self_echo_mode {self.self_echo_mode!r}; expected {VALID_SELF_ECHO_MODES}")
        if self.branch_router_mode not in VALID_BRANCH_ROUTER_MODES:
            raise ValueError(f"Unknown branch_router_mode {self.branch_router_mode!r}; expected {VALID_BRANCH_ROUTER_MODES}")
        object.__setattr__(self, "n_subheads_per_factor", eff_k)
        object.__setattr__(self, "subheads_per_factor", eff_k)
        object.__setattr__(self, "attn_evolution_mode", eff_attn)
        object.__setattr__(self, "attn_variant", eff_attn)

    @property
    def readout_type(self) -> str:
        """Return `"mixed_density"` or `"pure_ket"` for `evaluate_modded_nanogpt_validation` compatibility."""
        return "mixed_density" if self.readout_density_mode == "mixed_density" else "pure_ket"

    def to_attn_config(self) -> MultiHeadSuperpositionAttnConfig:
        """Build `MultiHeadSuperpositionAttnConfig` for transformer blocks."""
        return MultiHeadSuperpositionAttnConfig(
            num_meanings=int(self.num_meanings), num_factors=int(self.num_factors),
            d_register=int(self.d_register), n_subheads_per_factor=int(self.n_subheads_per_factor),
            attn_evolution_mode=str(self.attn_evolution_mode), max_seq_len=int(self.max_seq_len),
            boson_p=2, gram_eta_init=float(self.gram_eta_init), value_residual_mixing=True,
        )


class MultiHeadMixedStateKroneckerBlock(nn.Module):
    """Single complex block (`MultiHeadSuperpositionMixedStateAttention` + `ComplexUnitaryInterferenceFFN`)."""

    def __init__(self, cfg: MultiHeadMixedStateKroneckerConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.attn = MultiHeadSuperpositionMixedStateAttention(cfg.to_attn_config(), layer_idx=self.layer_idx)
        self.ffn = ComplexUnitaryInterferenceFFN(cfg.d_complex_total, expansion=DEFAULT_FFN_EXPANSION)
        init_scale = DEFAULT_RESIDUAL_INIT / math.sqrt(float(max(1, self.layer_idx + 1)))
        self.attn_gate = nn.Parameter(torch.full((cfg.d_complex_total,), init_scale))
        self.ffn_gate = nn.Parameter(torch.full((cfg.d_complex_total,), init_scale))

    def forward(
        self, psi_re: torch.Tensor, psi_im: torch.Tensor,
        v0_re: torch.Tensor | None = None, v0_im: torch.Tensor | None = None,
        eps: float = SUBHEAD_KET_NORM_EPS,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run multi-head Stinespring attention and unitary two-beam interference FFN (`<= 8` args)."""
        attn_re, attn_im, v_raw_re, v_raw_im = self.attn.forward_complex(psi_re, psi_im, v0_re=v0_re, v0_im=v0_im, eps=eps)
        attn_re, attn_im = complex_l2_normalize(attn_re, attn_im, dim=-1, eps=eps)
        g_a = torch.tanh(self.attn_gate).to(dtype=attn_re.dtype)
        p_re, p_im = psi_re.to(dtype=attn_re.dtype), psi_im.to(dtype=attn_im.dtype)
        mid_re, mid_im = complex_l2_normalize(p_re + g_a * attn_re, p_im + g_a * attn_im, dim=-1, eps=eps)

        ffn_re, ffn_im = self.ffn(mid_re, mid_im, eps=eps)
        g_f = torch.tanh(self.ffn_gate).to(dtype=ffn_re.dtype)
        out_re, out_im = complex_l2_normalize(mid_re + g_f * ffn_re, mid_im + g_f * ffn_im, dim=-1, eps=eps)
        return out_re, out_im, v_raw_re, v_raw_im


class MultiHeadMixedStateKroneckerGPT(NgramCalibratedKroneckerBornGPT):
    """Flag-Free Rank-`R` `16^4 = 65,536` Multi-Head Mixed-State Kronecker Wavefunction GPT (`Wave 44`)."""

    cfg: MultiHeadMixedStateKroneckerConfig

    def __init__(self, cfg: MultiHeadMixedStateKroneckerConfig | None = None) -> None:
        nn.Module.__init__(self)
        self.cfg = cfg if cfg is not None else MultiHeadMixedStateKroneckerConfig()
        kron_cfg = self.cfg.to_kronecker_config()
        v_sz, r_num, m_num, d_f = int(self.cfg.vocab_size), int(self.cfg.num_meanings), int(self.cfg.num_factors), int(self.cfg.factor_dim)
        k_tot = r_num * m_num
        k_bi = min(int(self.cfg.direct_bigram_slots), int(self.cfg.valid_vocab_size))
        h_tri = int(self.cfg.hashed_trigram_buckets)

        self.embed_re = nn.Parameter(torch.empty(v_sz, r_num, m_num, d_f))
        self.embed_im = nn.Parameter(torch.empty(v_sz, r_num, m_num, d_f))
        self.branch_logits = nn.Parameter(torch.zeros(v_sz, r_num))
        self.povm_scale_logits = nn.Parameter(torch.zeros(v_sz, dtype=torch.float32))
        self.bi_ctx_wave_re = nn.Parameter(torch.zeros(k_bi + 1, m_num, d_f))
        self.bi_ctx_wave_im = nn.Parameter(torch.zeros(k_bi + 1, m_num, d_f))
        self.tri_ctx_wave_re = nn.Parameter(torch.zeros(h_tri, m_num, d_f))
        self.tri_ctx_wave_im = nn.Parameter(torch.zeros(h_tri, m_num, d_f))

        for uname in ("out_factor_u", "bi_factor_u", "tri_factor_u", "smear_u"):
            setattr(self, f"{uname}_re", nn.Parameter(torch.empty(k_tot, d_f, d_f)))
            setattr(self, f"{uname}_im", nn.Parameter(torch.empty(k_tot, d_f, d_f)))

        self.ngram_branch_gates = nn.Parameter(torch.zeros(r_num, 3, dtype=torch.float32))
        self.smear_gate_logit = nn.Parameter(torch.tensor(DEFAULT_SMEAR_GATE_INIT, dtype=torch.float32))
        self.context_branch_bias = nn.Parameter(torch.zeros(r_num, dtype=torch.float32))
        self.psi0_readout_gate = nn.Parameter(torch.tensor(DEFAULT_PSI0_GATE_INIT, dtype=torch.float32))

        # Track 4: Zero-initialized phase-sensitive complex directional router parameters
        self.router_w_re = nn.Parameter(torch.zeros(r_num, m_num, d_f, dtype=torch.float32))
        self.router_w_im = nn.Parameter(torch.zeros(r_num, m_num, d_f, dtype=torch.float32))
        self.branch_router_gain = nn.Parameter(torch.ones(r_num, dtype=torch.float32))
        self.branch_router_bias = nn.Parameter(torch.zeros(r_num, dtype=torch.float32))

        # Track 3: Per-factor carrier-orthogonal complex self-echo gate logits
        if self.cfg.self_echo_mode in ("ortho_complex_learned", "carrier_ortho_complex"):
            u0_echo = _lambda_to_algebraic_logit(float(self.cfg.self_echo_lambda_init))
            self.echo_gate_logits: nn.Parameter | None = nn.Parameter(torch.full((r_num, m_num), u0_echo, dtype=torch.float32))
        else:
            self.register_parameter("echo_gate_logits", None)

        self.register_buffer("initial_povm_logits", torch.zeros(v_sz, dtype=torch.float32))
        self.register_buffer("initial_bi_ctx_re", torch.zeros(k_bi + 1, m_num, d_f))
        self.register_buffer("initial_bi_ctx_im", torch.zeros(k_bi + 1, m_num, d_f))
        self.register_buffer("branch_ortho_rot", torch.eye(d_f).view(1, 1, d_f, d_f).repeat(r_num, m_num, 1, 1))
        self.register_buffer("unigram_probs", torch.full((v_sz,), 1.0 / float(self.cfg.valid_vocab_size)))
        self.register_buffer("vocab_to_bi_slot", torch.zeros(v_sz, dtype=torch.int64))

        self.blocks = nn.ModuleList([MultiHeadMixedStateKroneckerBlock(self.cfg, i) for i in range(self.cfg.num_layers)])
        self.kron_mixers = nn.ModuleList([Kronecker16x4UnitaryMeaningMixer(kron_cfg) for _ in range(self.cfg.num_layers)])
        self._active_ngram_mode = str(self.cfg.ngram_mode)
        self._training_progress = 0.0
        self._last_branch_beta: torch.Tensor | None = None
        self._last_phi_factors: torch.Tensor | None = None

        self.initialize_isotropic_kronecker(seed=1337)

    @property
    def gate_w_re(self) -> nn.Parameter:
        """Alias for `self.router_w_re`."""
        return self.router_w_re

    @property
    def gate_w_im(self) -> nn.Parameter:
        """Alias for `self.router_w_im`."""
        return self.router_w_im

    @property
    def gate_b(self) -> nn.Parameter:
        """Alias for `self.branch_router_bias`."""
        return self.branch_router_bias

    @torch.no_grad()
    def compute_povm_drift_std(self) -> float:
        """Return standard deviation of `povm_scale_logits[:V] - initial_povm_logits[:V]`."""
        v_eff = int(self.cfg.valid_vocab_size)
        diff = self.povm_scale_logits[:v_eff].detach().float() - self.initial_povm_logits[:v_eff].detach().float()
        return float(diff.std().item())

    @torch.no_grad()
    def initialize_isotropic_kronecker(self, seed: int = 1337) -> dict[str, float]:
        """Initialize `(V, R, 4, 16)` factor kets, non-diluted branch gates, and unitary banks (`<= 8` args)."""
        v_eff, r_num = int(self.cfg.valid_vocab_size), int(self.cfg.num_meanings)
        m_num, d_f = int(self.cfg.num_factors), int(self.cfg.factor_dim)
        k_tot, dev = r_num * m_num, self.embed_re.device
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed))

        raw_re = torch.randn((self.cfg.vocab_size, r_num, m_num, d_f), generator=gen, device=dev, dtype=torch.float32)
        raw_im = torch.randn((self.cfg.vocab_size, r_num, m_num, d_f), generator=gen, device=dev, dtype=torch.float32)
        inv_norm = torch.rsqrt(raw_re.square().sum(dim=-1, keepdim=True) + raw_im.square().sum(dim=-1, keepdim=True) + DEFAULT_KRON_EPS)
        self.embed_re.data.copy_((raw_re * inv_norm).to(dtype=self.embed_re.dtype))
        self.embed_im.data.copy_((raw_im * inv_norm).to(dtype=self.embed_im.dtype))

        for t_zero in (
            self.branch_logits.data, self.povm_scale_logits.data, self.initial_povm_logits,
            self.bi_ctx_wave_re.data, self.bi_ctx_wave_im.data, self.tri_ctx_wave_re.data,
            self.tri_ctx_wave_im.data, self.initial_bi_ctx_re, self.initial_bi_ctx_im,
            self.context_branch_bias.data, self.router_w_re.data, self.router_w_im.data, self.branch_router_bias.data,
        ):
            t_zero.zero_()
        self.branch_router_gain.data.fill_(1.0)
        self.psi0_readout_gate.data.fill_(DEFAULT_PSI0_GATE_INIT)
        self.smear_gate_logit.data.fill_(DEFAULT_SMEAR_GATE_INIT)
        if self.echo_gate_logits is not None:
            self.echo_gate_logits.data.fill_(_lambda_to_algebraic_logit(float(self.cfg.self_echo_lambda_init)))

        h_mat, c_proj = _get_helmert_zero_sum_basis(d_f, dev), torch.full((d_f, d_f), 1.0 / float(d_f), device=dev)
        for r in range(r_num):
            for m in range(m_num):
                if r == 0:
                    self.branch_ortho_rot[r, m].copy_(torch.eye(d_f, device=dev, dtype=torch.float32))
                else:
                    o_15 = _deterministic_orthogonal_matrix(d_f - 1, seed=BRANCH_ROTATION_SEED_BASE + BRANCH_ROTATION_SEED_STRIDE * r + m, device=dev)
                    self.branch_ortho_rot[r, m].copy_(h_mat.t() @ o_15 @ h_mat + c_proj)

        g_init = torch.tensor([DEFAULT_NGRAM_GATE_BI_INIT, DEFAULT_NGRAM_GATE_TRI_INIT, DEFAULT_NGRAM_GATE_OP_INIT], device=dev)
        self.ngram_branch_gates.data.copy_(g_init.unsqueeze(0).repeat(r_num, 1))
        if r_num > 1:
            self.ngram_branch_gates.data[1:] *= float(self.cfg.secondary_branch_scale)

        for mod in self.modules():
            if isinstance(mod, ComplexLinear):
                if mod.in_features == mod.out_features:
                    _init_square_complex_linear_haar(mod, gen=gen)
                else:
                    mod.reset_parameters()
        for blk in self.blocks:
            blk.attn.reset_stinespring_parameters()
        for mixer in self.kron_mixers:
            mixer.reset_parameters(gen=gen)
        for u_re_p, u_im_p in (
            (self.out_factor_u_re, self.out_factor_u_im), (self.bi_factor_u_re, self.bi_factor_u_im),
            (self.tri_factor_u_re, self.tri_factor_u_im), (self.smear_u_re, self.smear_u_im),
        ):
            ur, ui = _sample_haar_complex_unitary_batch(k_tot, d_f, dev, gen=gen)
            u_re_p.data.copy_(ur.to(dtype=u_re_p.dtype))
            u_im_p.data.copy_(ui.to(dtype=u_im_p.dtype))

        return {
            "sharp_eff_rank": self.compute_sharp_eff_rank(), "mean_factor_rank": self.compute_mean_factor_rank(),
            "cross_branch_overlap": self.compute_cross_branch_overlap(), "valid_vocab_size": float(v_eff),
        }

    @torch.no_grad()
    def _initialize_spectral_wavefunctions_and_tables_from_cache(
        self, cache: Ngram900MCalibrationData, rho_eff: float, seed: int = 1337,
    ) -> None:
        """Embed `900M`-token spectral coordinates using `cfg.secondary_branch_scale` (`1.0` default) (`<= 8` args)."""
        super()._initialize_spectral_wavefunctions_and_tables_from_cache(cache, rho_eff=rho_eff, seed=seed)
        v_eff, r_num = int(self.cfg.valid_vocab_size), int(self.cfg.num_meanings)
        m_num, d_f = int(self.cfg.num_factors), int(self.cfg.factor_dim)
        d_ortho, dev = int(self.cfg.ortho_real_dim_per_factor), self.embed_re.device
        if self._active_ngram_mode not in ("bigram", "trigram") or r_num <= 1:
            return

        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed) + 101)
        total_ortho_dim = r_num * m_num * d_ortho
        rand_raw = torch.randn((v_eff, total_ortho_dim), generator=gen, device=dev, dtype=torch.float32)
        rand_raw.sub_(rand_raw.mean(dim=0, keepdim=True))
        if v_eff >= total_ortho_dim:
            q_rand, _ = torch.linalg.qr(rand_raw, mode="reduced")
            rand_ortho = (q_rand * math.sqrt(float(v_eff))).reshape(v_eff, r_num, m_num, d_ortho)
        else:
            rand_ortho = rand_raw.reshape(v_eff, r_num, m_num, d_ortho)
        rand_ortho = rand_ortho * torch.rsqrt(rand_ortho.square().sum(dim=-1, keepdim=True).clamp_min(ROUTER_BETA_NORM_EPS))

        alpha_spec = float(self.cfg.spectral_wave_blend)
        base_re, base_im = embed_real30_to_carrier_ortho_c16(cache.spectral_vocab_ortho[:v_eff].to(dev), factor_dim=d_f)
        sec_scale = float(self.cfg.secondary_branch_scale)
        for r in range(r_num):
            alpha_r = alpha_spec if r == 0 else (alpha_spec * sec_scale)
            noise_scale = math.sqrt(max(0.0, 1.0 - alpha_r * alpha_r))
            for m in range(m_num):
                rot_16 = self.branch_ortho_rot[r, m].float()
                nr_re, nr_im = embed_real30_to_carrier_ortho_c16(rand_ortho[:, r, m], factor_dim=d_f)
                u_re, u_im = complex_l2_normalize(
                    alpha_r * (base_re[:, m, :] @ rot_16) + noise_scale * nr_re,
                    alpha_r * (base_im[:, m, :] @ rot_16) + noise_scale * nr_im, dim=-1,
                )
                self.embed_re.data[:v_eff, r, m].copy_(u_re.to(dtype=self.embed_re.dtype))
                self.embed_im.data[:v_eff, r, m].copy_(u_im.to(dtype=self.embed_im.dtype))

        if rho_eff > 0.0:
            log_p = torch.log(self.get_target_tempered_unigram().clamp_min(PARTITION_CLAMP_MIN))
            z_raw = (log_p - log_p.mean()) / max(float(log_p.std().item()), DEFAULT_CARRIER_STD_DIVISOR)
            rho_4d = (float(rho_eff) * (z_raw * torch.rsqrt(1.0 + z_raw.square()))).view(v_eff, 1, 1, 1)
            orth_scale = torch.sqrt((1.0 - rho_4d.square()).clamp_min(DEFAULT_KRON_EPS))
            c_re, c_im = _build_canonical_carrier_ket(d_f, dev)
            u_perp_re, u_perp_im = complex_l2_normalize(self.embed_re.data[:v_eff].float(), self.embed_im.data[:v_eff].float(), dim=-1)
            tilted_re, tilted_im = complex_l2_normalize(
                rho_4d * c_re.view(1, 1, 1, d_f) + orth_scale * u_perp_re,
                rho_4d * c_im.view(1, 1, 1, d_f) + orth_scale * u_perp_im, dim=-1,
            )
            self.embed_re.data[:v_eff].copy_(tilted_re.to(dtype=self.embed_re.dtype))
            self.embed_im.data[:v_eff].copy_(tilted_im.to(dtype=self.embed_im.dtype))

    def _run_smeared_backbone(
        self, idx_clamped: torch.Tensor, branch_alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run smeared Layer-0 + multi-head Stinespring `blocks` + `kron_mixers` (`<= 8` args)."""
        psi_re, psi_im, psi0_re, psi0_im = self._pack_smeared_token_wavefunctions(idx_clamped, branch_alpha)
        v0_re: torch.Tensor | None = None
        v0_im: torch.Tensor | None = None
        use_ckpt = bool(self.cfg.checkpoint_blocks and self.training and torch.is_grad_enabled() and not torch.compiler.is_compiling())
        for l_idx, (block, mixer) in enumerate(zip(self.blocks, self.kron_mixers)):
            if use_ckpt:
                psi_re, psi_im, vr, vi = checkpoint(block, psi_re, psi_im, v0_re, v0_im, use_reentrant=False)
            else:
                psi_re, psi_im, vr, vi = block(psi_re, psi_im, v0_re=v0_re, v0_im=v0_im)
            if l_idx == 0:
                v0_re, v0_im = vr, vi
            psi_re, psi_im = mixer(psi_re, psi_im)
        return psi_re, psi_im, psi0_re, psi0_im

    def build_calibrated_readout_factors(
        self, psi_re: torch.Tensor, psi_im: torch.Tensor, psi0_re: torch.Tensor,
        psi0_im: torch.Tensor, idx_clamped: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply `out_factor_u`, `H_perp` complex self-echo, `R`-invariant norm, and phase router (`<= 8` args)."""
        b_sz, seq_t, _ = psi_re.shape
        n_tok = b_sz * seq_t
        r_num, m_num, d_f = int(self.cfg.num_meanings), int(self.cfg.num_factors), int(self.cfg.factor_dim)
        k_tot = r_num * m_num

        g0 = torch.tanh(self.psi0_readout_gate).to(dtype=psi_re.dtype)
        p0_re, p0_im = psi0_re.to(dtype=psi_re.dtype), psi0_im.to(dtype=psi_im.dtype)
        mixed_re, mixed_im = complex_l2_normalize(psi_re + g0 * p0_re, psi_im - g0 * p0_im, dim=-1)

        op_k_re, op_k_im = _apply_factor_unitary_bank(
            mixed_re.reshape(n_tok, k_tot, d_f), mixed_im.reshape(n_tok, k_tot, d_f),
            self.out_factor_u_re.to(dtype=mixed_re.dtype), self.out_factor_u_im.to(dtype=mixed_re.dtype),
        )
        op_4d_re, op_4d_im = op_k_re.reshape(n_tok, r_num, m_num, d_f), op_k_im.reshape(n_tok, r_num, m_num, d_f)
        p0_4d_re, p0_4d_im = p0_re.reshape(n_tok, r_num, m_num, d_f), p0_im.reshape(n_tok, r_num, m_num, d_f)

        # Track 3: Carrier-orthogonal complex self-echo projection in H_perp
        echo_re, echo_im = apply_carrier_orthogonal_complex_self_echo(
            op_4d_re, op_4d_im, p0_4d_re, p0_4d_im,
            self_echo_mode=self.cfg.self_echo_mode, echo_gate_logits=self.echo_gate_logits, eps=SELF_ECHO_NORM_EPS,
        )

        # Track 4: R-invariant per-factor normalization before adding N-gram delta
        raw_phi_re, raw_phi_im = normalize_raw_phi_per_factor_r_invariant(echo_re, echo_im, eps=ROUTER_BETA_NORM_EPS, reference_scale=1.0)
        ng_re, ng_im = self._compute_ngram_carrier_ortho_delta(idx_clamped, out_dtype=raw_phi_re.dtype)
        inv_ref = 1.0 / REFERENCE_RAW_PHI_FACTOR_SCALE
        comb_re, comb_im = raw_phi_re + ng_re * inv_ref, raw_phi_im + ng_im * inv_ref

        phi_re, phi_im = complex_l2_normalize(comb_re, comb_im, dim=-1)
        if self.cfg.branch_router_mode == "legacy_norm_only":
            branch_beta = self.compute_context_branch_beta(comb_re, comb_im)
        else:
            p0_unit_re, p0_unit_im = complex_l2_normalize(p0_4d_re, p0_4d_im, dim=-1)
            branch_beta = compute_phase_sensitive_branch_beta(
                phi_re, phi_im, p0_unit_re, p0_unit_im,
                self.router_w_re, self.router_w_im, self.branch_router_bias, eps=ROUTER_BETA_NORM_EPS,
            )

        phi_factors = torch.cat([phi_re, phi_im], dim=-1)
        if not torch.compiler.is_compiling():
            self._last_branch_beta = branch_beta.detach()
            self._last_phi_factors = phi_factors.detach()
        return phi_factors, branch_beta

    @torch.no_grad()
    def _compute_unscaled_born_energies(
        self, phi_factors: torch.Tensor, branch_beta: torch.Tensor,
    ) -> torch.Tensor:
        """Compute unscaled Born energy matrix `E_{n,v} = (I_{n,v})^p` matching `readout_density_mode`."""
        if self.cfg.readout_density_mode != "mixed_density":
            return super()._compute_unscaled_born_energies(phi_factors, branch_beta)

        n_tok, r_num, m_num, d_2 = phi_factors.shape
        d_f, k_tot = d_2 // 2, r_num * m_num
        v_eff, v_chunk = int(self.cfg.valid_vocab_size), max(1, int(self.cfg.vocab_chunk_size))

        psi_factors, alpha_f = self.get_tied_vocab_factors().float(), self.compute_branch_alpha().float()
        phi_f, beta_f = phi_factors.float(), branch_beta.float()
        phi_re, phi_im = phi_f[..., :d_f], phi_f[..., d_f:]
        phi_skew = torch.cat([-phi_im, phi_re], dim=-1)
        u_batched = torch.cat([phi_f.reshape(n_tok, k_tot, d_2).permute(1, 0, 2), phi_skew.reshape(n_tok, k_tot, d_2).permute(1, 0, 2)], dim=1).contiguous()
        w_all = psi_factors.reshape(v_eff, k_tot, d_2).permute(1, 2, 0).contiguous()

        e_mat = torch.empty((n_tok, v_eff), device=phi_factors.device, dtype=torch.float32)
        boson_p, holo_alpha, kron_blend, coinc_gamma = float(self.cfg.boson_p), float(self.cfg.holo_alpha), float(self.cfg.kron_blend), float(self.cfg.coinc_gamma)
        for v0 in range(0, v_eff, v_chunk):
            v1 = min(v_eff, v0 + v_chunk)
            vc = v1 - v0
            z_cat = torch.bmm(u_batched, w_all[:, :, v0:v1])
            x_ov, y_ov = z_cat[:, :n_tok, :].reshape(r_num, m_num, n_tok, vc), z_cat[:, n_tok:, :].reshape(r_num, m_num, n_tok, vc)
            i_tv, _, _ = _compute_mixed_state_kronecker_chunk_intensity(x_ov, y_ov, beta_f, alpha_f[v0:v1], holo_alpha, kron_blend, coinc_gamma)
            e_mat[:, v0:v1] = torch.pow(i_tv, boson_p)
        return e_mat

    @torch.no_grad()
    def compute_empirical_output_kl_to_target(self, eval_seqs: torch.Tensor) -> float:
        """Compute `D_KL(P_target_uni || P_bar_Step0)` self-consistently with `build_calibrated_readout_factors`."""
        v_eff = int(self.cfg.valid_vocab_size)
        idx_clamped = eval_seqs.clamp(min=0, max=v_eff - 1)
        _, _, psi_re, psi_im, p0_re, p0_im = self._collect_backbone_pre_operator_states(eval_seqs)
        phi_factors, branch_beta = self.build_calibrated_readout_factors(psi_re, psi_im, p0_re, p0_im, idx_clamped)
        n_eval = min(int(phi_factors.size(0)), DEFAULT_CALIBRATION_MAX_EVAL_TOKENS)
        e_mat = self._compute_unscaled_born_energies(phi_factors[:n_eval], branch_beta[:n_eval])
        bar_e = self._smooth_carrier_energy_expectation(e_mat.mean(dim=0))
        p_bar = (self.compute_povm_weights().float() * bar_e).clamp_min(PARTITION_CLAMP_MIN)
        p_bar = p_bar / p_bar.sum().clamp_min(PARTITION_CLAMP_MIN)
        p_target = self.get_target_tempered_unigram().clamp_min(PARTITION_CLAMP_MIN)
        return float(max(0.0, float((p_target * (torch.log(p_target) - torch.log(p_bar))).sum().item())))

    def extract_context_readout_states(
        self, idx: torch.Tensor, branch_alpha: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run smeared backbone and calibrated readout factors returning `(phi_factors, branch_beta, alpha_eff)` (`<= 8` args)."""
        idx_clamped = idx.clamp(min=0, max=int(self.cfg.valid_vocab_size) - 1)
        alpha_eff = self.compute_branch_alpha() if branch_alpha is None else branch_alpha
        psi_re, psi_im, psi0_re, psi0_im = self._run_smeared_backbone(idx_clamped, alpha_eff)
        phi_factors, branch_beta = self.build_calibrated_readout_factors(psi_re, psi_im, psi0_re, psi0_im, idx_clamped)
        return phi_factors, branch_beta, alpha_eff

    def extract_readout_states(
        self, idx: torch.Tensor, branch_alpha: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(phi_factors, psi_factors, branch_beta, branch_alpha)` (`<= 8` args)."""
        phi_factors, branch_beta, alpha_eff = self.extract_context_readout_states(idx, branch_alpha=branch_alpha)
        return phi_factors, self.get_tied_vocab_factors(), branch_beta, alpha_eff

    @torch.no_grad()
    def compute_subhead_purity_telemetry(self, idx: torch.Tensor | None = None) -> dict[str, float]:
        """Aggregate cross-head Gram and reduced-density purity telemetry across all layers (`<= 8` args)."""
        if idx is not None:
            _ = self.extract_context_readout_states(idx)

        tels = [blk.attn.compute_subhead_purity_telemetry() for blk in self.blocks]
        denom = float(max(1, len(tels)))
        mean_p = sum(float(t["subhead_gram_purity_mean"]) for t in tels) / denom
        mean_r = sum(float(t["subhead_effective_rank"]) for t in tels) / denom
        mean_s2 = sum(float(t["subhead_renyi2_entropy"]) for t in tels) / denom
        max_tr_err = max((float(t["gram_trace_err"]) for t in tels), default=0.0)
        mean_nu = sum(float(t["branch_interference_nu_mean"]) for t in tels) / denom

        branch_red_purity = mean_p
        if self._last_phi_factors is not None and self._last_branch_beta is not None:
            d_f = int(self.cfg.factor_dim)
            phi_f, pi_nr = self._last_phi_factors.float(), self._last_branch_beta.float().square()
            pr, pi = phi_f[..., :d_f], phi_f[..., d_f:]
            ov_re = torch.einsum("nrmd,nsmd->nmrs", pr, pr) + torch.einsum("nrmd,nsmd->nmrs", pi, pi)
            ov_im = torch.einsum("nrmd,nsmd->nmrs", pi, pr) - torch.einsum("nrmd,nsmd->nmrs", pr, pi)
            w_rs = pi_nr.unsqueeze(1).unsqueeze(-1) * pi_nr.unsqueeze(1).unsqueeze(-2)
            branch_red_purity = float((w_rs * (ov_re.square() + ov_im.square())).sum(dim=(-2, -1)).mean().item())

        return {
            "subhead_gram_purity_mean": mean_p, "head_purity_mean": mean_p, "purity_mean": mean_p,
            "reduced_density_purity_mean": branch_red_purity, "subhead_gram_trace_mean": 1.0,
            "gram_trace_err": max_tr_err, "subhead_renyi2_entropy": mean_s2, "renyi2_entropy_nats": mean_s2,
            "subhead_effective_rank": mean_r, "mixed_rank_mean": mean_r, "branch_interference_nu_mean": mean_nu,
        }

    @torch.no_grad()
    def compute_branch_routing_telemetry(self, idx: torch.Tensor | None = None) -> dict[str, float]:
        """Return polysemy branch purity, context routing variation, and router norm telemetry (`<= 8` args)."""
        if idx is not None:
            _, branch_beta, _ = self.extract_context_readout_states(idx)
        elif self._last_branch_beta is not None:
            branch_beta = self._last_branch_beta
        else:
            r_num = int(self.cfg.num_meanings)
            branch_beta = torch.full((1, r_num), 1.0 / math.sqrt(float(r_num)), device=self.embed_re.device)

        beta_f = branch_beta.float()
        pi_nr = beta_f.square()
        branch_purity = pi_nr.square().sum(dim=-1).mean()
        branch_eff_rank = (1.0 / pi_nr.square().sum(dim=-1).clamp_min(ROUTER_BETA_NORM_EPS)).mean()
        beta_std = beta_f.std(dim=0).mean() if beta_f.size(0) > 1 else torch.tensor(0.0)
        beta0_std = beta_f[:, 0].std() if beta_f.size(0) > 1 else torch.tensor(0.0)
        w_norm = torch.sqrt(self.router_w_re.float().square().sum() + self.router_w_im.float().square().sum())

        v_eff, r_num = int(self.cfg.valid_vocab_size), int(self.cfg.num_meanings)
        e_re, e_im = complex_l2_normalize(self.embed_re[:v_eff].float(), self.embed_im[:v_eff].float(), dim=-1)
        ket_purity = (e_re.square() + e_im.square()).sum(dim=-1).square().mean()
        gate_ratio = (
            float((self.ngram_branch_gates[1:].float().norm() / (math.sqrt(float(max(1, r_num - 1))) * self.ngram_branch_gates[0].float().norm().clamp_min(ROUTER_BETA_NORM_EPS))).item())
            if r_num > 1 else 1.0
        )

        return {
            "step0_branch_ket_purity": float(ket_purity.item()),
            "secondary_branch_scale": float(self.cfg.secondary_branch_scale),
            "secondary_to_primary_gate_ratio": gate_ratio,
            "branch_purity_mean": float(branch_purity.item()),
            "branch_effective_rank": float(branch_eff_rank.item()),
            "branch_beta_std": float(beta_std.item()),
            "branch_beta0_std": float(beta0_std.item()),
            "router_weight_frobenius_norm": float(w_norm.item()),
            "cross_branch_overlap": float(self.compute_cross_branch_overlap()),
        }

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None,
        return_logits: bool | None = None, token_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Run `L` multi-head Stinespring blocks + Kronecker mixers and Mixed-State Born readout (`<= 8` args)."""
        b_sz, seq_t = idx.shape
        r_num, m_num, d_2 = int(self.cfg.num_meanings), int(self.cfg.num_factors), 2 * int(self.cfg.factor_dim)
        v_eff = int(self.cfg.valid_vocab_size)

        phi_factors, psi_factors, branch_beta, branch_alpha = self.extract_readout_states(idx)
        povm_w = self.compute_povm_weights()
        hyperparams = torch.tensor(
            [self.cfg.boson_p, self.cfg.holo_alpha, self.cfg.kron_blend, self.cfg.coinc_gamma],
            device=idx.device, dtype=torch.float32,
        )
        want_logits = (targets is None) if return_logits is None else bool(return_logits)
        use_mixed = self.cfg.readout_density_mode == "mixed_density"

        if targets is not None and not want_logits:
            valid_targets = targets.clamp(min=0, max=v_eff - 1).reshape(-1)
            fn_cls = MixedStateKroneckerBornChunkedNLLFn if use_mixed else CalibratedKroneckerBornChunkedNLLFn
            fn_cls.vocab_chunk_size = int(self.cfg.vocab_chunk_size)
            loss = fn_cls.apply(phi_factors, psi_factors, branch_beta, branch_alpha, povm_w, valid_targets, hyperparams, token_weights)
            return (loss if return_logits is None else None), loss

        prob_fn = eval_mixed_state_kronecker_born_probs if use_mixed else eval_calibrated_kronecker_born_probs
        probs = prob_fn(
            phi_factors.reshape(b_sz, seq_t, r_num, m_num, d_2), psi_factors,
            branch_beta.reshape(b_sz, seq_t, r_num), branch_alpha, povm_w, hyperparams,
        )
        log_probs = torch.log(probs.clamp_min(PROB_LOG_CLAMP_MIN))
        loss = None
        if targets is not None:
            token_nll = -log_probs.gather(-1, targets.clamp(0, v_eff - 1).unsqueeze(-1)).squeeze(-1).reshape(-1)
            tw_f = token_weights.reshape(-1).float() if token_weights is not None else None
            loss = token_nll.mean() if tw_f is None else ((tw_f / tw_f.sum().clamp_min(PROB_LOG_CLAMP_MIN)) * token_nll).sum()
        out_logits = log_probs if want_logits else (loss if (targets is not None and return_logits is None) else None)
        return out_logits, loss


def compute_multihead_mixed_state_kronecker_penalties(
    model: MultiHeadMixedStateKroneckerGPT,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute Sphere, Unitary (including Stinespring `W_m`), Isotropy, Branch-Orth, & Anchor penalties (`<= 8` args)."""
    v_eff = int(model.cfg.valid_vocab_size)
    base_penalty, stats = compute_kronecker_manifold_and_isotropy_penalties(model)  # type: ignore[arg-type]

    u_bi = _batch_complex_unitary_terms(model.bi_factor_u_re, model.bi_factor_u_im)
    u_tri = _batch_complex_unitary_terms(model.tri_factor_u_re, model.tri_factor_u_im)
    u_sm = _batch_complex_unitary_terms(model.smear_u_re, model.smear_u_im)
    stinespring_terms = [
        _batch_complex_unitary_terms(blk.attn.stinespring_w_re.transpose(-1, -2), -blk.attn.stinespring_w_im.transpose(-1, -2))
        if blk.attn.stinespring_w_re.size(-2) >= blk.attn.stinespring_w_re.size(-1)
        else _batch_complex_unitary_terms(blk.attn.stinespring_w_re, blk.attn.stinespring_w_im)
        for blk in model.blocks
    ]
    extra_u = torch.cat([u_bi, u_tri, u_sm, *stinespring_terms], dim=0)

    povm_diff = model.povm_scale_logits[:v_eff].float() - model.initial_povm_logits[:v_eff].float()
    povm_anchor_pen = povm_diff.square().mean()

    ctx_diff_re = model.bi_ctx_wave_re.float() - model.initial_bi_ctx_re.float()
    ctx_diff_im = model.bi_ctx_wave_im.float() - model.initial_bi_ctx_im.float()
    tri_energy = (model.tri_ctx_wave_re.float().square() + model.tri_ctx_wave_im.float().square()).mean()
    ctx_anchor_pen = (ctx_diff_re.square() + ctx_diff_im.square()).mean() + 0.25 * tri_energy

    w_u, w_p, w_c = float(model.cfg.unitary_penalty_weight), float(model.cfg.povm_anchor_weight), float(model.cfg.ctx_anchor_weight)
    total_penalty = base_penalty + w_u * extra_u.sum() + w_p * povm_anchor_pen + w_c * ctx_anchor_pen
    stats["unitary_pen"] = float(0.5 * (stats["unitary_pen"] + float(extra_u.mean().detach().item())))
    stats["povm_anchor_pen"] = float(povm_anchor_pen.detach().item())
    stats["ctx_anchor_pen"] = float(ctx_anchor_pen.detach().item())
    stats["total_pen"] = float(total_penalty.detach().item())
    stats["povm_drift_std"] = float(povm_diff.detach().std().item())
    return total_penalty, stats


def build_multihead_mixed_state_kronecker_optimizers(
    model: MultiHeadMixedStateKroneckerGPT, muon_lr: float = DEFAULT_MUON_LR,
    embed_lr: float = DEFAULT_EMBED_LR, povm_lr: float = DEFAULT_POVM_LR,
    gate_lr: float = DEFAULT_GATE_LR, weight_decay: float = 0.0,
    stiefel_penalty: float = DEFAULT_W44_STIEFEL_PENALTY,
) -> tuple[StiefelMuon, torch.optim.AdamW]:
    """Route paired 2D/3D complex unitary matrices (`dim >= 2`) to `StiefelMuon` & 3 groups to `AdamW` (`<= 8` args)."""
    d_real = float(max(64, model.cfg.real_equiv_dim))
    d_ratio = 256.0 / d_real
    depth_scale = math.sqrt(6.0 / float(max(6, model.cfg.num_layers)))
    if d_real >= 256.0:
        muon_scale, embed_scale, gate_scale = (d_ratio**0.15) * depth_scale, d_ratio**0.50, math.sqrt(d_ratio)
    else:
        muon_scale, embed_scale, gate_scale = min(1.30, d_ratio**0.15) * depth_scale, min(1.45, d_ratio**0.30), min(1.35, d_ratio**0.25)

    param_dict = {name: p for name, p in model.named_parameters() if p.requires_grad}
    used_ids: set[int] = set()
    muon_params: list[nn.Parameter] = []
    embed_params: list[nn.Parameter] = []
    povm_params: list[nn.Parameter] = []
    gate_params: list[nn.Parameter] = []

    emb_names = {"embed_re", "embed_im", "bi_ctx_wave_re", "bi_ctx_wave_im", "tri_ctx_wave_re", "tri_ctx_wave_im"}
    excluded_muon_prefixes = ("embed_", "bi_ctx_wave_", "tri_ctx_wave_", "router_w_", "branch_logits", "ngram_branch_gates", "echo_gate_logits")

    for name, p in param_dict.items():
        if name in emb_names:
            embed_params.append(p)
            used_ids.add(id(p))
        elif name == "povm_scale_logits" or "povm_scale" in name:
            povm_params.append(p)
            used_ids.add(id(p))

    for name, p_re in param_dict.items():
        if id(p_re) in used_ids or any(name.startswith(pref) or f".{pref}" in name for pref in excluded_muon_prefixes):
            continue
        im_name = f"{name[:-10]}.weight_im" if name.endswith(".weight_re") else (f"{name[:-3]}_im" if name.endswith("_re") else None)
        if im_name is None or im_name not in param_dict:
            continue
        p_im = param_dict[im_name]
        if (
            id(p_im) not in used_ids and p_re.shape == p_im.shape and p_re.ndim in (2, 3)
            and min(p_re.shape[-2:]) >= MIN_STIEFEL_MUON_MATRIX_DIM and max(p_re.shape[-2:]) <= MAX_COMPLEX_MUON_DIM
        ):
            muon_params.extend((p_re, p_im))
            used_ids.update((id(p_re), id(p_im)))

    for p in param_dict.values():
        if id(p) not in used_ids:
            gate_params.append(p)
            used_ids.add(id(p))

    eff_muon_lr, eff_embed_lr, eff_povm_lr, eff_gate_lr = float(muon_lr) * muon_scale, float(embed_lr) * embed_scale, float(povm_lr), float(gate_lr) * gate_scale
    opt_muon = StiefelMuon(
        [{"params": muon_params, "lr": eff_muon_lr, "base_lr": eff_muon_lr, "is_complex_pair_group": True}],
        lr=eff_muon_lr, momentum=0.95, polar_method="polar_express",
        stiefel_penalty=float(stiefel_penalty), retract_every=1, use_normuon=True, use_twin_rail=True,
    )
    opt_adam = torch.optim.AdamW(
        [
            {"params": embed_params, "lr": eff_embed_lr, "base_lr": eff_embed_lr, "weight_decay": 0.0},
            {"params": povm_params, "lr": eff_povm_lr, "base_lr": eff_povm_lr, "weight_decay": 0.0},
            {"params": gate_params, "lr": eff_gate_lr, "base_lr": eff_gate_lr, "weight_decay": float(weight_decay)},
        ],
        betas=(0.9, 0.95), eps=1e-8,
    )
    return opt_muon, opt_adam


__all__ = [
    "DEFAULT_BRANCH_ROUTER_MODE", "DEFAULT_ECHO_GATE_LOGIT_INIT", "DEFAULT_MIXED_STATE_INTENSITY_EPS",
    "DEFAULT_NON_DILUTED_SECONDARY_BRANCH_SCALE", "DEFAULT_READOUT_DENSITY_MODE", "DEFAULT_SELF_ECHO_LAMBDA_INIT",
    "DEFAULT_SELF_ECHO_MODE", "LEGACY_DILUTED_SECONDARY_BRANCH_SCALE", "REFERENCE_NGRAM_CALIBRATION_BRANCH_COUNT",
    "REFERENCE_RAW_PHI_FACTOR_SCALE", "ROUTER_BETA_NORM_EPS", "ROUTER_DIRECTIONAL_SCORE_NORM", "SELF_ECHO_NORM_EPS",
    "VALID_BRANCH_ROUTER_MODES", "VALID_READOUT_DENSITY_MODES", "VALID_SELF_ECHO_MODES",
    "MixedStateKroneckerBornChunkedNLLFn", "MultiHeadMixedStateKroneckerBlock",
    "MultiHeadMixedStateKroneckerConfig", "MultiHeadMixedStateKroneckerGPT",
    "_compute_mixed_state_kronecker_chunk_intensity", "_mixed_state_kronecker_chunk_bwd_step",
    "_mixed_state_kronecker_chunk_fwd_step", "apply_carrier_orthogonal_complex_self_echo",
    "build_multihead_mixed_state_kronecker_optimizers", "compute_multihead_mixed_state_kronecker_penalties",
    "compute_phase_sensitive_branch_beta", "eval_mixed_state_kronecker_born_probs",
    "normalize_raw_phi_per_factor_r_invariant",
]
