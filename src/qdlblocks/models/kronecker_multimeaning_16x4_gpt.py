"""Rank-R Multi-Meaning 16x16x16x16 = 65,536 Kronecker Wavefunction GPT (`Wave 41` / `EXPERIMENT_PLAN_056.md`).

When to Use This Building Block:
- Use `KroneckerMultiMeaning16x4GPT`, `KroneckerMultiMeaningConfig`, `Kronecker16x4UnitaryMeaningMixer`,
  `KroneckerMultiMeaningBornChunkedNLLFn`, `eval_kronecker_multimeaning_born_probs`,
  `compute_kronecker_manifold_and_isotropy_penalties`, and `build_kronecker_multimeaning_optimizers`
  when training pure-complex quantum language models where:
  1. Vocabulary tokens (`V = 50,257`) live in a vocabulary-scale Hilbert space
     `H_Kron = C^16 (x) C^16 (x) C^16 (x) C^16 ~= C^{65,536}` (`5` hierarchical levels:
     `1 -> 16 -> 256 -> 4,096 -> 65,536`) using only `4 * 16 = 64` complex amplitudes per meaning branch
     (`512x` single-branch background noise suppression `(1/16)^4 = 1/65,536` vs. `1/128` for flat `C^128`).
  2. Each polysemous word is represented as a Rank-`R` entangled superposition of `R` meaning branches
     (`|Psi_v> = sum_{r=0}^{R-1} alpha_{v,r} (x)_{m=0}^3 |psi_{v,r}^{(m)}>`), acting as a quantum OR-gate
     across `R` word senses and a quantum AND-gate across the `4` Kronecker factors.
  3. Readout uses 100% `exp`/`softmax`/`logsumexp`-free polynomial `p`-boson Born probabilities weighted by
     the empirical `100M`-token FineWeb unigram prior (`49,850` observed tokens + `407` unseen tokens floored
     at `1.0` count) and a phase-balanced reference wave `e^{i*pi/4} = (1 + i)/sqrt(2)`.

Why It Was Built & Mathematical Provenance (`EXPERIMENT_PLAN_056.md` / `Wave 41`):
- Resolves the three post-mortem bottlenecks identified at the close of `Wave 40` (`EXPERIMENT_PLAN_055.md`):
  1. **Phase-Balanced Reference Wave (`e^{i*pi/4}`) + Multi-Scale Kronecker Coincidence**:
     Eliminates the `16x` real-vs-imaginary gradient asymmetry near the origin while combining a 1-factor
     linear gradient highway `(1 - eta) * z_bar` with the 4-factor Kronecker product `eta * prod_{m=0}^3 z^{(m)}`
     and 8th-degree coincidence energy `gamma_c * prod_{m=0}^3 |z^{(m)}|^2`.
  2. **Relaxed Per-Factor Parseval Isotropy (`lambda_iso = 0.05`) + Cross-Meaning Branch Orthogonality**:
     Replaces the over-rigid global `lambda_iso = 0.50` penalty with per-factor 16D isotropy (`R_factor >= 20/32`)
     and an explicit 4-factor Kronecker branch orthogonality penalty (`O_branch < 1e-3`).
  3. **Single-`torch.bmm` Complex Factor Overlap Kernel**:
     Evaluates both real and imaginary parts of all `4 * R` 16-dimensional complex inner products in a single
     batched Tensor-Core GEMM `torch.bmm(U_batched, W_chunk)` per vocabulary chunk.
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

from ..kernels.fused_kronecker_mixed_state_kernels import fused_kronecker_16x4_unitary_mixer_step
from ..layers.packed_complex_register_density_attn import (
    ComplexLinear, DEFAULT_COMPLEX_NORM_EPS, DEFAULT_COMPLEX_REGISTER_DIM,
    DEFAULT_P_BOSON_POWER, PackedComplexRegisterDensityAttention, complex_l2_normalize,
)
from ..optim import muon as _muon_mod
from ..optim.muon import StiefelMuon
from .gemini_packed_complex_rope_gpt import ComplexUnitaryInterferenceFFN

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_056.md & Tracks 1, 2, 3, 5
# ==============================================================================
DEFAULT_NUM_FACTORS: int = 4
DEFAULT_FACTOR_DIM: int = 16
DEFAULT_REGISTER_DIM: int = 16
DEFAULT_KRONECKER_REGISTER_DIM: int = DEFAULT_REGISTER_DIM
DEFAULT_NUM_MEANINGS: int = 2
DEFAULT_KRON_BOSON_P: float = 11.0
DEFAULT_KRON_HOLO_ALPHA: float = 0.85
DEFAULT_KRON_BLEND: float = 0.55
DEFAULT_KRON_COINC_GAMMA: float = 0.45
DEFAULT_KRON_EPS: float = 1e-6
DEFAULT_REF_PHASE_COS_SIN: float = 0.7071067811865476  # 1 / sqrt(2) for e^{i*pi/4}
DEFAULT_UNIGRAM_TEMP: float = 0.85
DEFAULT_UNIGRAM_FLOOR_COUNT: float = 1.0
DEFAULT_BRANCH_EPS: float = 1e-4
DEFAULT_SPHERE_PENALTY_WEIGHT: float = 2.0
DEFAULT_UNITARY_PENALTY_WEIGHT: float = 0.08
DEFAULT_ISO_PENALTY_WEIGHT: float = 0.05
DEFAULT_BRANCH_ORTH_WEIGHT: float = 0.20
DEFAULT_VOCAB_CHUNK_SIZE: int = 8192
DEFAULT_MUON_LR: float = 0.015
DEFAULT_EMBED_LR: float = 0.008
DEFAULT_GATE_LR: float = 0.003
DEFAULT_MIXER_GATE_INIT: float = 0.20
DEFAULT_MIXER_NOISE_STD: float = 0.02
DEFAULT_PSI0_GATE_INIT: float = 0.20
DEFAULT_RESIDUAL_INIT: float = 0.35
DEFAULT_FFN_EXPANSION: int = 2
MIN_COMPLEX_MUON_DIM: int = 2
MIN_KRONECKER_MUON_OPERATOR_DIM: int = 2
MAX_COMPLEX_MUON_DIM: int = 4096
DEFAULT_KRONECKER_MUON_STIEFEL_PENALTY: float = 0.01
DEFAULT_MUON_REFERENCE_BATCH_TOKENS: int = 8192
DEFAULT_MUON_BATCH_LR_EXPONENT: float = 0.50
MAX_MUON_BATCH_LR_MULTIPLIER: float = 3.0
DEFAULT_SELF_ECHO_MODE: str = "ortho_complex_learned"
VALID_SELF_ECHO_MODES: tuple[str, ...] = (
    "none", "no_self_echo", "legacy_real", "legacy_ambient_real", "raw_conjugate",
    "ortho_real_only", "ortho_complex_full", "ortho_complex_learned",
)
DEFAULT_SELF_ECHO_LAMBDA_INIT: float = 0.50
DEFAULT_SELF_ECHO_LOGIT_INIT: float = 0.0
DEFAULT_ECHO_GATE_LOGIT_INIT: float = 0.0
SELF_ECHO_NORM_EPS: float = 1e-6
ALGEBRAIC_GATE_DENOM_ONE: float = 1.0
ALGEBRAIC_GATE_HALF: float = 0.5
PROB_LOG_CLAMP_MIN: float = 1e-12
PARTITION_CLAMP_MIN: float = 1e-30
FINEWEB_HEADER_INTS: int = 256
FINEWEB_HEADER_BYTES: int = 1024
FINEWEB_HEADER_MAGIC: int = 20240520

LOOKUP_TABLE_PARAM_SUFFIXES: tuple[str, ...] = (
    "embed_re", "embed_im", "bi_ctx_wave_re", "bi_ctx_wave_im", "tri_ctx_wave_re", "tri_ctx_wave_im",
)
POVM_SCALE_PARAM_SUFFIXES: tuple[str, ...] = ("povm_scale_logits",)
NON_OPERATOR_GATE_PARAM_TOKENS: tuple[str, ...] = (
    "branch_logits", "ngram_branch_gates", "router_w_re", "router_w_im", "branch_router_gain",
    "branch_router_bias", "echo_gate_logits", "gram_eta", "eta_g", "krylov_eta", "purity_gamma",
    "norm_gain", "resid_scale", "v_embed_mix", "smear_gate_logit", "carrier_logit_bias",
)
KRONECKER_EMBED_PARAM_NAMES: tuple[str, ...] = LOOKUP_TABLE_PARAM_SUFFIXES
KRONECKER_EMBED_SUBSTRINGS: tuple[str, ...] = ("embed_", "_ctx_wave_")
KRONECKER_GATE_PARAM_NAMES: tuple[str, ...] = POVM_SCALE_PARAM_SUFFIXES
KRONECKER_GATE_SUBSTRINGS: tuple[str, ...] = NON_OPERATOR_GATE_PARAM_TOKENS
COMPLEX_MATRIX_OPERATOR_BASENAMES: tuple[str, ...] = (
    "w_q", "w_k", "w_v", "w_out", "q_proj", "k_proj", "v_proj", "out_proj", "beam1", "beam2",
    "proj_down", "factor_u", "factor_mix", "meaning_mix", "out_factor_u", "bi_factor_u",
    "tri_factor_u", "smear_u", "stinespring_w",
)


@dataclass(frozen=True)
class KroneckerMultiMeaningConfig:
    """Frozen configuration for `KroneckerMultiMeaning16x4GPT` (`Wave 41` / `EXPERIMENT_PLAN_056.md`)."""

    vocab_size: int = 50304
    valid_vocab_size: int = 50257
    num_meanings: int = DEFAULT_NUM_MEANINGS
    num_factors: int = DEFAULT_NUM_FACTORS
    factor_dim: int = DEFAULT_FACTOR_DIM
    num_layers: int = 6
    block_size: int = 512
    boson_p: float = DEFAULT_KRON_BOSON_P
    holo_alpha: float = DEFAULT_KRON_HOLO_ALPHA
    kron_blend: float = DEFAULT_KRON_BLEND
    coinc_gamma: float = DEFAULT_KRON_COINC_GAMMA
    unigram_temp: float = DEFAULT_UNIGRAM_TEMP
    sphere_penalty_weight: float = DEFAULT_SPHERE_PENALTY_WEIGHT
    unitary_penalty_weight: float = DEFAULT_UNITARY_PENALTY_WEIGHT
    iso_penalty_weight: float = DEFAULT_ISO_PENALTY_WEIGHT
    branch_orth_weight: float = DEFAULT_BRANCH_ORTH_WEIGHT
    vocab_chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE
    checkpoint_blocks: bool = False
    register_dim: int | None = None
    d_register: int = DEFAULT_REGISTER_DIM
    n_registers: int = 0
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
            raise ValueError(f"Unknown self_echo_mode: {self.self_echo_mode!r}; expected one of {VALID_SELF_ECHO_MODES}")

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
        """Alias for `d_complex_total` representing the complex model width."""
        return self.d_complex_total

    @property
    def real_equiv_dim(self) -> int:
        """Total real degrees of freedom per token (`2 * d_complex_total`)."""
        return 2 * self.d_complex_total


def _sample_haar_complex_unitary_batch(
    batch_k: int,
    dim_d: int,
    device: torch.device,
    gen: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch_k` independent Haar complex unitary matrices in `U(dim_d)`."""
    z_re = torch.randn((batch_k, dim_d, dim_d), generator=gen, device=device, dtype=torch.float32)
    z_im = torch.randn((batch_k, dim_d, dim_d), generator=gen, device=device, dtype=torch.float32)
    z_c = torch.complex(z_re, z_im)
    q_mat, r_mat = torch.linalg.qr(z_c)
    diag_r = torch.diagonal(r_mat, dim1=-2, dim2=-1)
    phase = diag_r / diag_r.abs().clamp_min(DEFAULT_KRON_EPS)
    u_mat = q_mat * phase.unsqueeze(-2)
    return u_mat.real.contiguous(), u_mat.imag.contiguous()


def _sample_near_identity_complex_unitary_batch(
    batch_k: int,
    dim_d: int,
    noise_std: float,
    device: torch.device,
    gen: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize a batch of `(batch_k, dim_d, dim_d)` complex unitaries near `I + noise_std * Z`."""
    eye = torch.eye(dim_d, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_k, 1, 1)
    n_re = float(noise_std) * torch.randn((batch_k, dim_d, dim_d), generator=gen, device=device, dtype=torch.float32)
    n_im = float(noise_std) * torch.randn((batch_k, dim_d, dim_d), generator=gen, device=device, dtype=torch.float32)
    z_c = torch.complex(eye + n_re, n_im)
    q_mat, r_mat = torch.linalg.qr(z_c)
    diag_r = torch.diagonal(r_mat, dim1=-2, dim2=-1)
    phase = diag_r / diag_r.abs().clamp_min(DEFAULT_KRON_EPS)
    u_mat = q_mat * phase.unsqueeze(-2)
    return u_mat.real.contiguous(), u_mat.imag.contiguous()


def _init_square_complex_linear_haar(
    mod: ComplexLinear,
    gen: torch.Generator | None = None,
) -> None:
    """Initialize a square `ComplexLinear` layer as an exact Haar complex unitary matrix `W W^H = I`."""
    if mod.in_features != mod.out_features:
        return
    d_c = int(mod.in_features)
    u_re, u_im = _sample_haar_complex_unitary_batch(1, d_c, mod.weight_re.device, gen=gen)
    mod.weight_re.data.copy_(u_re[0].to(dtype=mod.weight_re.dtype))
    mod.weight_im.data.copy_(u_im[0].to(dtype=mod.weight_im.dtype))


def _algebraic_positive_unit_l2(
    logits: torch.Tensor,
    dim: int = -1,
    eps: float = DEFAULT_BRANCH_EPS,
) -> torch.Tensor:
    """Map unconstrained real tensor `logits` to strictly positive unit-L2 amplitudes without `exp`/`softplus`.

    Uses the smooth algebraic rational root map `a^+ = 0.5 * (u + sqrt(u^2 + 1)) + eps_branch > 0`
    followed by L2 normalization along `dim` so `sum_r alpha_r^2 == 1.0`.
    """
    u_f = logits.float()
    pos = 0.5 * (u_f + torch.sqrt(u_f.square() + 1.0)) + float(eps)
    inv_norm = torch.rsqrt(pos.square().sum(dim=dim, keepdim=True).clamp_min(PARTITION_CLAMP_MIN))
    return pos * inv_norm


def _self_echo_lambda_to_logit(lambda_val: float) -> float:
    """Invert `lambda(u) = 0.5 * (1 + u / sqrt(u^2 + 1))` to `u = (2 * lambda - 1) / (2 * sqrt(lambda * (1 - lambda)))`."""
    lam = min(max(float(lambda_val), 1e-4), 1.0 - 1e-4)
    return (2.0 * lam - 1.0) / (2.0 * math.sqrt(lam * (1.0 - lam)))


def apply_carrier_orthogonal_complex_self_echo(
    phi_re: torch.Tensor, phi_im: torch.Tensor, psi0_re: torch.Tensor, psi0_im: torch.Tensor,
    mode: str = DEFAULT_SELF_ECHO_MODE, echo_gate_logits: torch.Tensor | None = None,
    eps: float = SELF_ECHO_NORM_EPS, self_echo_mode: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply carrier-orthogonal complex self-echo gating per `(r, m)` factor (`<= 8` args).

    Mathematical Formulation (`EXPERIMENT_PLAN_059.md` Track 3):
    - In `"ortho_complex"` / `"ortho_complex_learned"`, leaves coordinate `0` (`c_0 = 1/sqrt(16)`,
      the Helmert unigram carrier) untouched and projects out the complex overlap only along AC
      coordinates `1..d_f-1` (`P_perp^{AC}`), preserving unigram marginals while suppressing self-echo.
    """
    eff_mode = str(self_echo_mode if self_echo_mode is not None else mode)
    if eff_mode not in VALID_SELF_ECHO_MODES:
        raise ValueError(f"Unsupported self_echo_mode={eff_mode!r}; expected one of {VALID_SELF_ECHO_MODES}")

    p0_re, p0_im = psi0_re.to(dtype=phi_re.dtype), psi0_im.to(dtype=phi_im.dtype)
    if eff_mode in ("none", "no_self_echo"):
        lam = torch.zeros((1, phi_re.size(1), phi_re.size(2), 1), device=phi_re.device, dtype=phi_re.dtype)
        return phi_re, phi_im, lam

    if eff_mode in ("legacy_real", "legacy_ambient_real", "raw_conjugate"):
        ov_re = (p0_re.float() * phi_re.float() + p0_im.float() * phi_im.float()).sum(dim=-1, keepdim=True).to(phi_re.dtype)
        lam = torch.ones((1, phi_re.size(1), phi_re.size(2), 1), device=phi_re.device, dtype=phi_re.dtype)
        return phi_re - ov_re * p0_re, phi_im - ov_re * p0_im, lam

    if eff_mode == "ortho_complex_learned" and echo_gate_logits is not None:
        u_g = echo_gate_logits.float()
        lam_f = 0.5 * (1.0 + u_g * torch.rsqrt(u_g.square() + 1.0))
        lam = lam_f.to(dtype=phi_re.dtype).view(1, phi_re.size(1), phi_re.size(2), 1)
    else:
        lam = torch.full((1, phi_re.size(1), phi_re.size(2), 1), DEFAULT_SELF_ECHO_LAMBDA_INIT, device=phi_re.device, dtype=phi_re.dtype)

    phi_ac_re, phi_ac_im = phi_re[..., 1:].float(), phi_im[..., 1:].float()
    p0_ac_re, p0_ac_im = p0_re[..., 1:].float(), p0_im[..., 1:].float()
    p0_ac_norm_sq = (p0_ac_re.square().sum(dim=-1, keepdim=True) + p0_ac_im.square().sum(dim=-1, keepdim=True)).clamp_min(float(eps))
    ov_ac_re = (p0_ac_re * phi_ac_re + p0_ac_im * phi_ac_im).sum(dim=-1, keepdim=True) / p0_ac_norm_sq
    if eff_mode == "ortho_real_only":
        proj_re, proj_im = (ov_ac_re * p0_ac_re).to(dtype=phi_re.dtype), (ov_ac_re * p0_ac_im).to(dtype=phi_im.dtype)
    else:
        ov_ac_im = (p0_ac_re * phi_ac_im - p0_ac_im * phi_ac_re).sum(dim=-1, keepdim=True) / p0_ac_norm_sq
        proj_re = (ov_ac_re * p0_ac_re - ov_ac_im * p0_ac_im).to(dtype=phi_re.dtype)
        proj_im = (ov_ac_re * p0_ac_im + ov_ac_im * p0_ac_re).to(dtype=phi_im.dtype)
    out_re = torch.cat([phi_re[..., :1], phi_re[..., 1:] - lam * proj_re], dim=-1)
    out_im = torch.cat([phi_im[..., :1], phi_im[..., 1:] - lam * proj_im], dim=-1)
    return out_re, out_im, lam


class Kronecker16x4UnitaryMeaningMixer(nn.Module):
    """Batched `16x16` Factor Unitary, `4x4` Cross-Factor, & `RxR` Cross-Meaning Mixer (`Wave 41`)."""

    def __init__(self, cfg: KroneckerMultiMeaningConfig) -> None:
        super().__init__()
        self.num_meanings, self.num_factors, self.factor_dim = int(cfg.num_meanings), int(cfg.num_factors), int(cfg.factor_dim)
        k_tot = self.num_meanings * self.num_factors
        self.factor_u_re = nn.Parameter(torch.empty(k_tot, self.factor_dim, self.factor_dim))
        self.factor_u_im = nn.Parameter(torch.empty(k_tot, self.factor_dim, self.factor_dim))
        self.factor_mix_re = nn.Parameter(torch.empty(self.num_meanings, self.num_factors, self.num_factors))
        self.factor_mix_im = nn.Parameter(torch.empty(self.num_meanings, self.num_factors, self.num_factors))
        self.meaning_mix_re = nn.Parameter(torch.empty(self.num_factors, self.num_meanings, self.num_meanings))
        self.meaning_mix_im = nn.Parameter(torch.empty(self.num_factors, self.num_meanings, self.num_meanings))
        self.mixer_gate = nn.Parameter(torch.tensor(DEFAULT_MIXER_GATE_INIT, dtype=torch.float32))
        self.use_fused_kernels: bool = True
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, gen: torch.Generator | None = None) -> None:
        """Initialize `16x16` factor unitaries as Haar `U(16)` and `4x4` / `RxR` mixers near identity."""
        k_tot = self.num_meanings * self.num_factors
        dev = self.factor_u_re.device
        u_re, u_im = _sample_haar_complex_unitary_batch(k_tot, self.factor_dim, dev, gen=gen)
        self.factor_u_re.data.copy_(u_re.to(dtype=self.factor_u_re.dtype))
        self.factor_u_im.data.copy_(u_im.to(dtype=self.factor_u_im.dtype))
        fm_re, fm_im = _sample_near_identity_complex_unitary_batch(self.num_meanings, self.num_factors, DEFAULT_MIXER_NOISE_STD, dev, gen=gen)
        self.factor_mix_re.data.copy_(fm_re.to(dtype=self.factor_mix_re.dtype))
        self.factor_mix_im.data.copy_(fm_im.to(dtype=self.factor_mix_im.dtype))
        mm_re, mm_im = _sample_near_identity_complex_unitary_batch(self.num_factors, self.num_meanings, DEFAULT_MIXER_NOISE_STD, dev, gen=gen)
        self.meaning_mix_re.data.copy_(mm_re.to(dtype=self.meaning_mix_re.dtype))
        self.meaning_mix_im.data.copy_(mm_im.to(dtype=self.meaning_mix_im.dtype))
        self.mixer_gate.data.fill_(DEFAULT_MIXER_GATE_INIT)

    def forward(self, psi_re: torch.Tensor, psi_im: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply batched `16x16` factor unitaries, `4x4` factor entangler, and `RxR` meaning interferometer (`<= 8` args)."""
        if getattr(self, "use_fused_kernels", False) and psi_re.is_cuda and not torch.compiler.is_compiling():
            return fused_kronecker_16x4_unitary_mixer_step(
                psi_re, psi_im, self.factor_u_re, self.factor_u_im,
                self.factor_mix_re, self.factor_mix_im, self.meaning_mix_re, (self.meaning_mix_im, self.mixer_gate),
            )
        w_dtype = psi_re.dtype
        b_sz, seq_t, d_tot = psi_re.shape
        r_num, m_num, d_f = self.num_meanings, self.num_factors, self.factor_dim
        k_tot = r_num * m_num

        x_re, x_im = psi_re.view(b_sz, seq_t, k_tot, d_f), psi_im.view(b_sz, seq_t, k_tot, d_f)
        u_re, u_im = self.factor_u_re.to(dtype=w_dtype), self.factor_u_im.to(dtype=w_dtype)
        u1_re = (torch.einsum("btkd,kdo->btko", x_re, u_re) - torch.einsum("btkd,kdo->btko", x_im, u_im)).view(b_sz, seq_t, r_num, m_num, d_f)
        u1_im = (torch.einsum("btkd,kdo->btko", x_re, u_im) + torch.einsum("btkd,kdo->btko", x_im, u_re)).view(b_sz, seq_t, r_num, m_num, d_f)

        fm_re, fm_im = self.factor_mix_re.to(dtype=w_dtype), self.factor_mix_im.to(dtype=w_dtype)
        u2_re = torch.einsum("btrmd,rmn->btrnd", u1_re, fm_re) - torch.einsum("btrmd,rmn->btrnd", u1_im, fm_im)
        u2_im = torch.einsum("btrmd,rmn->btrnd", u1_re, fm_im) + torch.einsum("btrmd,rmn->btrnd", u1_im, fm_re)

        mm_re, mm_im = self.meaning_mix_re.to(dtype=w_dtype), self.meaning_mix_im.to(dtype=w_dtype)
        u3_re = torch.einsum("btrmd,mrs->btsmd", u2_re, mm_re) - torch.einsum("btrmd,mrs->btsmd", u2_im, mm_im)
        u3_im = torch.einsum("btrmd,mrs->btsmd", u2_re, mm_im) + torch.einsum("btrmd,mrs->btsmd", u2_im, mm_re)

        u3_re, u3_im = complex_l2_normalize(u3_re, u3_im, dim=-1)
        x_5d_re, x_5d_im = x_re.view(b_sz, seq_t, r_num, m_num, d_f), x_im.view(b_sz, seq_t, r_num, m_num, d_f)
        gate = torch.tanh(self.mixer_gate).to(dtype=w_dtype)
        mix_re, mix_im = complex_l2_normalize(x_5d_re + gate * u3_re, x_5d_im + gate * u3_im, dim=-1)
        inv_sqrt_k = 1.0 / math.sqrt(float(k_tot))
        return (mix_re * inv_sqrt_k).reshape(b_sz, seq_t, d_tot).to(w_dtype), (mix_im * inv_sqrt_k).reshape(b_sz, seq_t, d_tot).to(w_dtype)


def _compute_kronecker_chunk_intensity(
    x_ov: torch.Tensor, y_ov: torch.Tensor, branch_beta: torch.Tensor, alpha_chunk: torch.Tensor,
    holo_alpha: float, kron_blend: float, coinc_gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute multi-meaning 4-factor Kronecker amplitude `(A_re, A_im)` and intensity `I_tv` (`<= 8` args)."""
    x_f, y_f = x_ov.float(), y_ov.float()
    x_bar, y_bar = 0.25 * (x_f[:, 0] + x_f[:, 1] + x_f[:, 2] + x_f[:, 3]), 0.25 * (y_f[:, 0] + y_f[:, 1] + y_f[:, 2] + y_f[:, 3])
    p01_re, p01_im = x_f[:, 0] * x_f[:, 1] - y_f[:, 0] * y_f[:, 1], x_f[:, 0] * y_f[:, 1] + y_f[:, 0] * x_f[:, 1]
    p23_re, p23_im = x_f[:, 2] * x_f[:, 3] - y_f[:, 2] * y_f[:, 3], x_f[:, 2] * y_f[:, 3] + y_f[:, 2] * x_f[:, 3]
    k_re, k_im = p01_re * p23_re - p01_im * p23_im, p01_re * p23_im + p01_im * p23_re

    eta = float(kron_blend)
    m_re, m_im = (1.0 - eta) * x_bar + eta * k_re, (1.0 - eta) * y_bar + eta * k_im

    w_r = branch_beta.t().unsqueeze(2) * alpha_chunk.t().unsqueeze(1)
    a_re, a_im = (w_r * m_re).sum(dim=0), (w_r * m_im).sum(dim=0)
    coinc = (w_r.square() * (k_re.square() + k_im.square())).sum(dim=0)

    s0, alpha = DEFAULT_REF_PHASE_COS_SIN, float(holo_alpha)
    i_tv = (s0 + alpha * a_re).square() + (s0 + alpha * a_im).square() + float(coinc_gamma) * coinc + DEFAULT_KRON_EPS
    return i_tv, a_re, a_im


_compiled_kronecker_chunk_intensity = torch.compile(_compute_kronecker_chunk_intensity)


def _compute_mixed_state_eval_intensity(
    x_ov: torch.Tensor, y_ov: torch.Tensor, beta_f: torch.Tensor, alpha_c: torch.Tensor,
    holo_alpha: float, kron_blend: float, coinc_gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute trace-1 incoherent mixed-state 4-factor Kronecker Born intensity for chunked evaluation (`<= 8` args)."""
    r_num, x_f, y_f = x_ov.size(0), x_ov.float(), y_ov.float()
    x_bar, y_bar = 0.25 * (x_f[:, 0] + x_f[:, 1] + x_f[:, 2] + x_f[:, 3]), 0.25 * (y_f[:, 0] + y_f[:, 1] + y_f[:, 2] + y_f[:, 3])
    p01_re, p01_im = x_f[:, 0] * x_f[:, 1] - y_f[:, 0] * y_f[:, 1], x_f[:, 0] * y_f[:, 1] + y_f[:, 0] * x_f[:, 1]
    p23_re, p23_im = x_f[:, 2] * x_f[:, 3] - y_f[:, 2] * y_f[:, 3], x_f[:, 2] * y_f[:, 3] + y_f[:, 2] * x_f[:, 3]
    k_re, k_im = p01_re * p23_re - p01_im * p23_im, p01_re * p23_im + p01_im * p23_re
    m_re, m_im = (1.0 - kron_blend) * x_bar + kron_blend * k_re, (1.0 - kron_blend) * y_bar + kron_blend * k_im
    s0, a_r = DEFAULT_REF_PHASE_COS_SIN, math.sqrt(float(r_num)) * alpha_c.t().unsqueeze(1)
    ref_re, ref_im = s0 + holo_alpha * a_r * m_re, s0 + holo_alpha * a_r * m_im
    i_r = ref_re.square() + ref_im.square() + coinc_gamma * a_r.square() * (k_re.square() + k_im.square())
    i_tv = (beta_f.t().square().unsqueeze(2) * i_r).sum(dim=0) + DEFAULT_KRON_EPS
    return i_tv, ref_re, ref_im


_compiled_mixed_state_eval_intensity = torch.compile(_compute_mixed_state_eval_intensity)


def eval_kronecker_multimeaning_born_probs(
    phi_factors: torch.Tensor, psi_factors: torch.Tensor, branch_beta: torch.Tensor,
    branch_alpha: torch.Tensor, prior_weights: torch.Tensor, hyperparams: torch.Tensor,
) -> torch.Tensor:
    """Evaluate 100% `exp`/`softmax`-free Multi-Meaning `16^4` Kronecker Born probabilities (`<= 8` args)."""
    lead_shape = phi_factors.shape[:-3]
    r_num, m_num, d_2 = phi_factors.shape[-3:]
    d_f, v_eff = d_2 // 2, psi_factors.size(0)
    n_tok = int(math.prod(lead_shape)) if lead_shape else 1
    k_tot = r_num * m_num

    phi_f = phi_factors.reshape(n_tok, r_num, m_num, d_2).float()
    psi_f = psi_factors.reshape(v_eff, r_num, m_num, d_2).float()
    beta_f = branch_beta.reshape(n_tok, r_num).float()
    alpha_f = branch_alpha.reshape(v_eff, r_num).float()
    prior_f = prior_weights[:v_eff].float()

    hp = hyperparams.float().reshape(-1)
    boson_p, holo_alpha, kron_blend, coinc_gamma = float(hp[0].item()), float(hp[1].item()), float(hp[2].item()), float(hp[3].item())

    phi_re, phi_im = phi_f[..., :d_f], phi_f[..., d_f:]
    phi_skew = torch.cat([-phi_im, phi_re], dim=-1)
    phi_k = phi_f.reshape(n_tok, k_tot, d_2).permute(1, 0, 2)
    phi_skew_k = phi_skew.reshape(n_tok, k_tot, d_2).permute(1, 0, 2)
    u_batched = torch.cat([phi_k, phi_skew_k], dim=1).contiguous()
    w_all = psi_f.reshape(v_eff, k_tot, d_2).permute(1, 2, 0).contiguous()

    z_cat = torch.bmm(u_batched, w_all)
    x_ov = z_cat[:, :n_tok, :].reshape(r_num, m_num, n_tok, v_eff)
    y_ov = z_cat[:, n_tok:, :].reshape(r_num, m_num, n_tok, v_eff)
    i_tv, _, _ = _compute_kronecker_chunk_intensity(x_ov, y_ov, beta_f, alpha_f, holo_alpha, kron_blend, coinc_gamma)
    p_tilde = prior_f.unsqueeze(0) * torch.pow(i_tv, boson_p)
    probs = p_tilde / p_tilde.sum(dim=-1, keepdim=True).clamp_min(PARTITION_CLAMP_MIN)
    return probs.reshape(*lead_shape, v_eff)


def _kronecker_chunk_fwd_step(
    z_cat: torch.Tensor, beta_f: torch.Tensor, alpha_c: torch.Tensor, prior_c: torch.Tensor,
    rel_idx: torch.Tensor, in_chunk_f: torch.Tensor, boson_p: float, coeffs: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-chunk partition sum `z_chunk` and target unnormalized probability `target_chunk` (`<= 8` args)."""
    n_tok, r_num = beta_f.shape
    vc = alpha_c.size(0)
    holo_alpha, kron_blend, coinc_gamma = coeffs
    z_f = z_cat.float()
    x_ov = z_f[:, :n_tok, :].reshape(r_num, 4, n_tok, vc)
    y_ov = z_f[:, n_tok:, :].reshape(r_num, 4, n_tok, vc)
    i_tv, _, _ = _compute_kronecker_chunk_intensity(x_ov, y_ov, beta_f, alpha_c, holo_alpha, kron_blend, coinc_gamma)
    p_tilde_c = prior_c.unsqueeze(0) * torch.pow(i_tv, boson_p)
    z_chunk = p_tilde_c.sum(dim=-1)
    target_chunk = in_chunk_f * p_tilde_c.gather(1, rel_idx).squeeze(1)
    return z_chunk, target_chunk


def _kronecker_chunk_bwd_step_with_prior_grad(
    z_cat: torch.Tensor, beta_f: torch.Tensor, alpha_c: torch.Tensor, prior_c: torch.Tensor,
    scale_inv_z: tuple[torch.Tensor, torch.Tensor], target_match: tuple[torch.Tensor, torch.Tensor],
    boson_p: float, coeffs: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-pass 2-level binary complex product tree pullback returning `(g_cat, d_beta_c, d_alpha_c, d_prior_c)` (`<= 8` args)."""
    n_tok, r_num = beta_f.shape
    vc = alpha_c.size(0)
    k_tot = r_num * 4
    scale_t, inv_z = scale_inv_z
    v_idx, targets_col = target_match
    holo_alpha, kron_blend, coinc_gamma = coeffs
    s0, eta = DEFAULT_REF_PHASE_COS_SIN, kron_blend
    one_minus_eta_quarter = 0.25 * (1.0 - eta)

    z_f = z_cat.float()
    x_ov = z_f[:, :n_tok, :].reshape(r_num, 4, n_tok, vc)
    y_ov = z_f[:, n_tok:, :].reshape(r_num, 4, n_tok, vc)
    x0, x1, x2, x3 = x_ov[:, 0], x_ov[:, 1], x_ov[:, 2], x_ov[:, 3]
    y0, y1, y2, y3 = y_ov[:, 0], y_ov[:, 1], y_ov[:, 2], y_ov[:, 3]
    x_bar, y_bar = 0.25 * (x0 + x1 + x2 + x3), 0.25 * (y0 + y1 + y2 + y3)

    p01_re, p01_im = x0 * x1 - y0 * y1, x0 * y1 + y0 * x1
    p23_re, p23_im = x2 * x3 - y2 * y3, x2 * y3 + y2 * x3
    k_re, k_im = p01_re * p23_re - p01_im * p23_im, p01_re * p23_im + p01_im * p23_re
    k_mag_sq = k_re.square() + k_im.square()
    m_re, m_im = (1.0 - eta) * x_bar + eta * k_re, (1.0 - eta) * y_bar + eta * k_im

    w_r = beta_f.t().unsqueeze(2) * alpha_c.t().unsqueeze(1)
    a_re, a_im = (w_r * m_re).sum(dim=0), (w_r * m_im).sum(dim=0)
    coinc = (w_r.square() * k_mag_sq).sum(dim=0)

    ref_re, ref_im = s0 + holo_alpha * a_re, s0 + holo_alpha * a_im
    i_tv = ref_re.square() + ref_im.square() + coinc_gamma * coinc + DEFAULT_KRON_EPS
    p_tilde_c = prior_c.unsqueeze(0) * torch.pow(i_tv, boson_p)
    probs_c = p_tilde_c * inv_z
    is_target = (v_idx == targets_col).to(dtype=torch.float32)

    delta_tv = scale_t * (probs_c - is_target)
    d_prior_c = delta_tv.sum(dim=0) / prior_c.clamp_min(DEFAULT_KRON_EPS)
    g_i = delta_tv * (boson_p / i_tv)
    g_a_re_b = (g_i * (2.0 * holo_alpha * ref_re)).unsqueeze(0)
    g_a_im_b = (g_i * (2.0 * holo_alpha * ref_im)).unsqueeze(0)
    g_coinc_b = (g_i * coinc_gamma).unsqueeze(0)

    g_w_r = g_a_re_b * m_re + g_a_im_b * m_im + 2.0 * g_coinc_b * w_r * k_mag_sq
    d_beta_c = (g_w_r * alpha_c.t().unsqueeze(1)).sum(dim=2).t()
    d_alpha_c = (g_w_r * beta_f.t().unsqueeze(2)).sum(dim=1).t()

    g_m_re, g_m_im = w_r * g_a_re_b, w_r * g_a_im_b
    two_gc_w2 = 2.0 * g_coinc_b * w_r.square()
    g_k_re, g_k_im = eta * g_m_re + two_gc_w2 * k_re, eta * g_m_im + two_gc_w2 * k_im

    g_p01_re, g_p01_im = g_k_re * p23_re + g_k_im * p23_im, g_k_im * p23_re - g_k_re * p23_im
    g_p23_re, g_p23_im = g_k_re * p01_re + g_k_im * p01_im, g_k_im * p01_re - g_k_re * p01_im
    g_xbar, g_ybar = one_minus_eta_quarter * g_m_re, one_minus_eta_quarter * g_m_im

    g_x0, g_y0 = g_xbar + g_p01_re * x1 + g_p01_im * y1, g_ybar + g_p01_im * x1 - g_p01_re * y1
    g_x1, g_y1 = g_xbar + g_p01_re * x0 + g_p01_im * y0, g_ybar + g_p01_im * x0 - g_p01_re * y0
    g_x2, g_y2 = g_xbar + g_p23_re * x3 + g_p23_im * y3, g_ybar + g_p23_im * x3 - g_p23_re * y3
    g_x3, g_y3 = g_xbar + g_p23_re * x2 + g_p23_im * y2, g_ybar + g_p23_im * x2 - g_p23_re * y2

    g_x = torch.stack([g_x0, g_x1, g_x2, g_x3], dim=1).reshape(k_tot, n_tok, vc)
    g_y = torch.stack([g_y0, g_y1, g_y2, g_y3], dim=1).reshape(k_tot, n_tok, vc)
    g_cat = torch.cat([g_x, g_y], dim=1).to(dtype=z_cat.dtype)
    return g_cat, d_beta_c, d_alpha_c, d_prior_c


def _kronecker_chunk_bwd_step(
    z_cat: torch.Tensor, beta_f: torch.Tensor, alpha_c: torch.Tensor, prior_c: torch.Tensor,
    scale_inv_z: tuple[torch.Tensor, torch.Tensor], target_match: tuple[torch.Tensor, torch.Tensor],
    boson_p: float, coeffs: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute analytical 2-level binary complex product tree pullback for one vocabulary chunk (`<= 8` args)."""
    return _kronecker_chunk_bwd_step_with_prior_grad(
        z_cat, beta_f, alpha_c, prior_c, scale_inv_z, target_match, boson_p, coeffs
    )[:3]


_compiled_kronecker_chunk_fwd_step = torch.compile(_kronecker_chunk_fwd_step)
_compiled_kronecker_chunk_bwd_step = torch.compile(_kronecker_chunk_bwd_step)
_compiled_kronecker_chunk_bwd_step_with_prior_grad = torch.compile(_kronecker_chunk_bwd_step_with_prior_grad)


class KroneckerMultiMeaningBornChunkedNLLFn(torch.autograd.Function):
    """Vocabulary-chunked Multi-Meaning `16^4` Kronecker Born NLL with single-`torch.bmm` complex overlap kernel.

    Chunks over `V = 50,257` (`vocab_chunk_size = 8192`) and evaluates all `R * 4` complex 16D inner
    products per chunk in a single `torch.bmm` call, followed by the `@torch.compile`-fused 2-level
    binary complex product tree forward intensity and analytical backward pullback.
    """

    vocab_chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE

    @staticmethod
    def forward(
        cls: Any, phi_factors: torch.Tensor, psi_factors: torch.Tensor, branch_beta: torch.Tensor,
        branch_alpha: torch.Tensor, targets: torch.Tensor, prior_weights: torch.Tensor,
        hyperparams: torch.Tensor, token_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute chunked Kronecker Multi-Meaning Born NLL (`<= 8` args)."""
        n_tok, r_num, m_num, d_2 = phi_factors.shape
        d_f = d_2 // 2
        v_eff = psi_factors.size(0)
        k_tot = r_num * m_num
        v_chunk = max(1, int(KroneckerMultiMeaningBornChunkedNLLFn.vocab_chunk_size))

        gemm_dtype = phi_factors.dtype
        phi_w = phi_factors.contiguous()
        psi_w = psi_factors.to(dtype=gemm_dtype).contiguous()
        beta_f = branch_beta.float().contiguous()
        alpha_f = branch_alpha.float().contiguous()
        prior_f = prior_weights[:v_eff].float().contiguous()
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
            z_chunk, target_chunk = fwd_fn(z_cat, beta_f, alpha_f[v0:v1], prior_f[v0:v1], rel_idx, in_chunk_f, boson_p, coeffs)
            z_t.add_(z_chunk)
            target_p_tilde.add_(target_chunk)

        z_clamped = z_t.clamp_min(PARTITION_CLAMP_MIN)
        target_clamped = target_p_tilde.clamp_min(PARTITION_CLAMP_MIN)
        loss = (w_norm * (torch.log(z_clamped) - torch.log(target_clamped))).sum()

        cls.save_for_backward(u_batched, w_all, beta_f, alpha_f, targets_flat, prior_f, w_norm, z_clamped)
        cls.boson_p = boson_p
        cls.coeffs = coeffs
        cls.v_chunk = v_chunk
        cls.shape_meta = (n_tok, r_num, m_num, d_2, v_eff)
        cls.dtypes = (phi_factors.dtype, psi_factors.dtype, branch_beta.dtype, branch_alpha.dtype)
        return loss

    @staticmethod
    def backward(
        cls: Any, grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None]:
        """Compute exact analytical backward pullback through the 2-level binary complex product tree."""
        u_batched, w_all, beta_f, alpha_f, targets_flat, prior_f, w_norm, z_clamped = cls.saved_tensors
        boson_p = float(cls.boson_p)
        coeffs = cls.coeffs
        v_chunk = int(cls.v_chunk)
        n_tok, r_num, m_num, d_2, v_eff = cls.shape_meta
        phi_dtype, psi_dtype, beta_dtype, alpha_dtype = cls.dtypes
        d_f = d_2 // 2
        k_tot = r_num * m_num
        gemm_dtype = u_batched.dtype

        d_phi_re_k = torch.zeros((k_tot, n_tok, d_f), device=u_batched.device, dtype=torch.float32)
        d_phi_im_k = torch.zeros((k_tot, n_tok, d_f), device=u_batched.device, dtype=torch.float32)
        d_psi = torch.zeros((v_eff, r_num, m_num, d_2), device=w_all.device, dtype=torch.float32)
        d_beta = torch.zeros((n_tok, r_num), device=beta_f.device, dtype=torch.float32)
        d_alpha = torch.zeros((v_eff, r_num), device=alpha_f.device, dtype=torch.float32)

        scale_inv_z = ((grad_out.float() * w_norm).unsqueeze(1), (1.0 / z_clamped).unsqueeze(1))
        targets_col = targets_flat.unsqueeze(1)
        use_compiled = bool(u_batched.is_cuda and gemm_dtype == torch.bfloat16)
        bwd_fn = _compiled_kronecker_chunk_bwd_step if use_compiled else _kronecker_chunk_bwd_step

        for v0 in range(0, v_eff, v_chunk):
            v1 = min(v_eff, v0 + v_chunk)
            vc = v1 - v0
            w_chunk = w_all[:, :, v0:v1]
            z_cat = torch.bmm(u_batched, w_chunk)
            v_idx = torch.arange(v0, v1, device=u_batched.device, dtype=torch.long).unsqueeze(0)
            g_cat, d_beta_c, d_alpha_c = bwd_fn(
                z_cat, beta_f, alpha_f[v0:v1], prior_f[v0:v1], scale_inv_z, (v_idx, targets_col), boson_p, coeffs
            )
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
            None, None, None, None,
        )


class KroneckerComplexRegisterBlock(nn.Module):
    """Single complex register block (`PackedComplexRegisterDensityAttention` + `ComplexUnitaryInterferenceFFN`)."""

    def __init__(self, cfg: KroneckerMultiMeaningConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = PackedComplexRegisterDensityAttention(
            d_model=cfg.d_complex_total, n_registers=cfg.n_registers, d_register=cfg.d_register,
            p_boson=DEFAULT_P_BOSON_POWER, probe_mode="v_krylov", normalize_krylov_ket=True,
            value_residual_mixing=(layer_idx > 0),
        )
        self.ffn = ComplexUnitaryInterferenceFFN(cfg.d_complex_total, expansion=DEFAULT_FFN_EXPANSION)
        init_scale = DEFAULT_RESIDUAL_INIT / math.sqrt(float(max(1, layer_idx + 1)))
        self.attn_gate = nn.Parameter(torch.full((cfg.d_complex_total,), init_scale))
        self.ffn_gate = nn.Parameter(torch.full((cfg.d_complex_total,), init_scale))

    def forward(
        self, psi_re: torch.Tensor, psi_im: torch.Tensor,
        v0_re: torch.Tensor | None = None, v0_im: torch.Tensor | None = None, eps: float = DEFAULT_COMPLEX_NORM_EPS,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run complex register density attention and unitary two-beam interference FFN (`<= 8` args)."""
        attn_re, attn_im, v_raw_re, v_raw_im = self.attn.forward_complex(psi_re, psi_im, v0_re=v0_re, v0_im=v0_im, eps=eps)
        attn_re, attn_im = complex_l2_normalize(attn_re, attn_im, dim=-1, eps=eps)
        g_a = torch.tanh(self.attn_gate).to(dtype=attn_re.dtype)
        p_re, p_im = psi_re.to(dtype=attn_re.dtype), psi_im.to(dtype=attn_im.dtype)
        mid_re, mid_im = complex_l2_normalize(p_re + g_a * attn_re, p_im + g_a * attn_im, dim=-1, eps=eps)

        ffn_re, ffn_im = self.ffn(mid_re, mid_im, eps=eps)
        g_f = torch.tanh(self.ffn_gate).to(dtype=ffn_re.dtype)
        out_re, out_im = complex_l2_normalize(mid_re + g_f * ffn_re, mid_im + g_f * ffn_im, dim=-1, eps=eps)
        return out_re, out_im, v_raw_re, v_raw_im


def _load_header_safe_shard_tokens(shard_path: Path, max_tokens: int) -> np.ndarray:
    """Load up to `max_tokens` uint16 token IDs from a FineWeb `.bin` shard, skipping the 1024B header if present."""
    file_size = shard_path.stat().st_size
    offset = 0
    if file_size >= FINEWEB_HEADER_BYTES:
        hdr = np.fromfile(shard_path, dtype=np.int32, count=FINEWEB_HEADER_INTS)
        if hdr.shape[0] == FINEWEB_HEADER_INTS and (
            int(hdr[0]) == FINEWEB_HEADER_MAGIC or bool(np.all(hdr[4:] == 0))
        ):
            offset = FINEWEB_HEADER_BYTES
    raw = np.memmap(shard_path, dtype=np.uint16, mode="r", offset=offset)
    n_take = min(int(max_tokens), int(raw.shape[0]))
    return np.array(raw[:n_take], dtype=np.uint16, copy=True)


class KroneckerMultiMeaning16x4GPT(nn.Module):
    """Flag-Free Rank-`R` Multi-Meaning `16x16x16x16 = 65,536` Kronecker Wavefunction GPT (`Wave 41`)."""

    def __init__(self, cfg: KroneckerMultiMeaningConfig) -> None:
        super().__init__()
        self.cfg = cfg
        k_tot = cfg.num_meanings * cfg.num_factors

        self.embed_re = nn.Parameter(torch.empty(cfg.vocab_size, cfg.num_meanings, cfg.num_factors, cfg.factor_dim))
        self.embed_im = nn.Parameter(torch.empty(cfg.vocab_size, cfg.num_meanings, cfg.num_factors, cfg.factor_dim))
        self.branch_logits = nn.Parameter(torch.zeros(cfg.vocab_size, cfg.num_meanings))

        self.register_buffer("prior_weights", torch.ones(cfg.vocab_size, dtype=torch.float32))
        self.register_buffer("unigram_probs", torch.full((cfg.vocab_size,), 1.0 / float(cfg.valid_vocab_size), dtype=torch.float32))

        self.blocks = nn.ModuleList([KroneckerComplexRegisterBlock(cfg, i) for i in range(cfg.num_layers)])
        self.kron_mixers = nn.ModuleList([Kronecker16x4UnitaryMeaningMixer(cfg) for _ in range(cfg.num_layers)])

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

    @torch.no_grad()
    def initialize_isotropic_kronecker(self, seed: int = 1337) -> dict[str, float]:
        """Initialize `(V, R, 4, 16)` factor kets isotropically on `C^16` and all unitaries in `U(d)` (`<= 8` args)."""
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
    def initialize_unigram_prior_from_tokens(
        self, token_stream: np.ndarray | torch.Tensor, floor_count: float = DEFAULT_UNIGRAM_FLOOR_COUNT,
    ) -> dict[str, float]:
        """Compute empirical unigram probabilities `P_uni(v)` and Born weights `w_uni(v)` (`<= 8` args)."""
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

        if n_total == 0 or float(self.cfg.unigram_temp) == 0.0:
            p_unif = torch.full((v_eff,), 1.0 / float(v_eff), dtype=torch.float32)
            self.unigram_probs.zero_()
            self.unigram_probs[:v_eff].copy_(p_unif.to(device=self.unigram_probs.device))
            self.prior_weights.zero_()
            self.prior_weights[:v_eff].fill_(1.0)
            unseen = int((raw_counts == 0).sum().item()) if n_total > 0 else v_eff
            return {
                "total_tokens": float(n_total), "observed_tokens": float(v_eff - unseen), "unseen_tokens": float(unseen),
                "unigram_entropy_nats": round(math.log(float(v_eff)), 4), "min_prior_weight": 1.0, "max_prior_weight": 1.0,
            }

        unseen_tokens = int((raw_counts == 0).sum().item())
        observed_tokens = v_eff - unseen_tokens
        floored_counts = torch.clamp_min(raw_counts, float(floor_count))
        p_uni_d = floored_counts / floored_counts.sum().clamp_min(DEFAULT_KRON_EPS)
        p_uni = p_uni_d.to(torch.float32)
        w_uni = torch.pow(p_uni / p_uni.max().clamp_min(DEFAULT_KRON_EPS), float(self.cfg.unigram_temp))

        self.unigram_probs.zero_()
        self.unigram_probs[:v_eff].copy_(p_uni.to(device=self.unigram_probs.device))
        self.prior_weights.zero_()
        self.prior_weights[:v_eff].copy_(w_uni.to(device=self.prior_weights.device))

        entropy_nats = float(-(p_uni_d * torch.log(p_uni_d.clamp_min(PARTITION_CLAMP_MIN))).sum().item())
        return {
            "total_tokens": float(n_total), "observed_tokens": float(observed_tokens), "unseen_tokens": float(unseen_tokens),
            "unigram_entropy_nats": round(entropy_nats, 4),
            "min_prior_weight": float(w_uni.min().item()), "max_prior_weight": float(w_uni.max().item()),
        }

    @torch.no_grad()
    def load_100m_unigram_prior(
        self, data_dir: str | Path = ".cache/fineweb_100M", max_tokens: int = 100_000_000,
    ) -> dict[str, float]:
        """Load up to `100M` tokens from FineWeb Shard 1 (`train_full_100M.bin` or `fineweb_train_000001.bin`)."""
        d_path = Path(data_dir)
        shard_path = next((d_path / n for n in ("train_full_100M.bin", "fineweb_train_000001.bin") if (d_path / n).exists()), None)
        if shard_path is None:
            raise FileNotFoundError(f"Could not find train_full_100M.bin or fineweb_train_000001.bin in {d_path}")
        tokens = _load_header_safe_shard_tokens(shard_path, max_tokens=int(max_tokens))
        return self.initialize_unigram_prior_from_tokens(tokens, floor_count=DEFAULT_UNIGRAM_FLOOR_COUNT)

    def compute_branch_alpha(self) -> torch.Tensor:
        """Compute 100% `exp`/`softmax`-free unit-L2 vocabulary meaning amplitudes `alpha` of shape `(V, R)`."""
        v_eff = int(self.cfg.valid_vocab_size)
        return _algebraic_positive_unit_l2(self.branch_logits[:v_eff], dim=-1, eps=DEFAULT_BRANCH_EPS)

    def compute_context_branch_beta(self, raw_phi_re: torch.Tensor, raw_phi_im: torch.Tensor) -> torch.Tensor:
        """Compute 100% `exp`/`softmax`-free unit-L2 context meaning amplitudes `beta` of shape `(N, R)`."""
        rms_r = torch.sqrt((raw_phi_re.float().square().sum(dim=-1) + raw_phi_im.float().square().sum(dim=-1)).mean(dim=-1) + 1e-8)
        u_b = self.context_branch_bias.float()
        b_pos = 0.5 * (u_b + torch.sqrt(u_b.square() + 1.0))
        beta_tilde = rms_r * b_pos.unsqueeze(0) + DEFAULT_BRANCH_EPS
        inv_norm = torch.rsqrt(beta_tilde.square().sum(dim=-1, keepdim=True).clamp_min(PARTITION_CLAMP_MIN))
        return beta_tilde * inv_norm

    def get_tied_vocab_factors(self) -> torch.Tensor:
        """Return unit-norm packed vocabulary factor kets `psi_factors` of shape `(V, R, M, 2 * d_f)`."""
        v_eff = int(self.cfg.valid_vocab_size)
        e_re, e_im = complex_l2_normalize(self.embed_re[:v_eff], self.embed_im[:v_eff], dim=-1)
        return torch.cat([e_re, e_im], dim=-1)

    def apply_readout_factor_operator(
        self, psi_re: torch.Tensor, psi_im: torch.Tensor, psi0_re: torch.Tensor, psi0_im: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply conjugate skip, batched `16x16` `out_factor_u`, and carrier-orthogonal self-echo gating (`<= 8` args)."""
        b_sz, seq_t, _ = psi_re.shape
        n_tok = b_sz * seq_t
        r_num, m_num, d_f = int(self.cfg.num_meanings), int(self.cfg.num_factors), int(self.cfg.factor_dim)
        k_tot = r_num * m_num

        g0 = torch.tanh(self.psi0_readout_gate).to(dtype=psi_re.dtype)
        p0_re, p0_im = psi0_re.to(dtype=psi_re.dtype), psi0_im.to(dtype=psi_im.dtype)
        mixed_re, mixed_im = complex_l2_normalize(psi_re + g0 * p0_re, psi_im - g0 * p0_im, dim=-1)

        q_re, q_im = mixed_re.reshape(n_tok, k_tot, d_f), mixed_im.reshape(n_tok, k_tot, d_f)
        u_out_re, u_out_im = self.out_factor_u_re.to(dtype=q_re.dtype), self.out_factor_u_im.to(dtype=q_re.dtype)
        op_re = (torch.einsum("nkd,kdo->nko", q_re, u_out_re) - torch.einsum("nkd,kdo->nko", q_im, u_out_im)).reshape(n_tok, r_num, m_num, d_f)
        op_im = (torch.einsum("nkd,kdo->nko", q_re, u_out_im) + torch.einsum("nkd,kdo->nko", q_im, u_out_re)).reshape(n_tok, r_num, m_num, d_f)
        p0_4d_re, p0_4d_im = p0_re.reshape(n_tok, r_num, m_num, d_f), p0_im.reshape(n_tok, r_num, m_num, d_f)

        raw_phi_re, raw_phi_im, _ = apply_carrier_orthogonal_complex_self_echo(
            op_re, op_im, p0_4d_re, p0_4d_im, mode=self.cfg.self_echo_mode, echo_gate_logits=self.echo_gate_logits,
        )
        branch_beta = self.compute_context_branch_beta(raw_phi_re, raw_phi_im)
        phi_re, phi_im = complex_l2_normalize(raw_phi_re, raw_phi_im, dim=-1)
        return torch.cat([phi_re, phi_im], dim=-1), branch_beta

    @torch.no_grad()
    def compute_sharp_eff_rank(self) -> float:
        """Compute participation ratio `R_emb in [1, 2 * d_complex_total]` of flattened normalized factor kets."""
        v_eff = int(self.cfg.valid_vocab_size)
        e_re, e_im = self.embed_re[:v_eff].detach().float(), self.embed_im[:v_eff].detach().float()
        inv_n = torch.rsqrt(e_re.square().sum(dim=-1, keepdim=True) + e_im.square().sum(dim=-1, keepdim=True) + DEFAULT_KRON_EPS)
        e_flat = torch.cat([e_re * inv_n, e_im * inv_n], dim=-1).reshape(v_eff, -1)
        e_cent = e_flat - e_flat.mean(dim=0, keepdim=True)
        gram = e_cent.t() @ e_cent
        tr = torch.diagonal(gram).sum()
        return float(((tr * tr) / (gram * gram).sum().clamp_min(PROB_LOG_CLAMP_MIN)).item())

    @torch.no_grad()
    def compute_mean_factor_rank(self) -> float:
        """Compute mean participation ratio `R_factor in [1, 2 * d_f]` across all `R * M` factor tables."""
        v_eff = int(self.cfg.valid_vocab_size)
        r_num, m_num = int(self.cfg.num_meanings), int(self.cfg.num_factors)
        d_2, k_tot = 2 * int(self.cfg.factor_dim), r_num * m_num
        e_re, e_im = self.embed_re[:v_eff].detach().float(), self.embed_im[:v_eff].detach().float()
        inv_n = torch.rsqrt(e_re.square().sum(dim=-1, keepdim=True) + e_im.square().sum(dim=-1, keepdim=True) + DEFAULT_KRON_EPS)
        psi_k = torch.cat([e_re * inv_n, e_im * inv_n], dim=-1).reshape(v_eff, k_tot, d_2).permute(1, 0, 2)
        psi_cent = psi_k - psi_k.mean(dim=1, keepdim=True)
        cov_k = torch.bmm(psi_cent.transpose(1, 2), psi_cent) / float(v_eff)
        tr_k = torch.diagonal(cov_k, dim1=-2, dim2=-1).sum(dim=-1)
        frob_sq_k = cov_k.square().sum(dim=(-2, -1)).clamp_min(PROB_LOG_CLAMP_MIN)
        return float(((tr_k * tr_k) / frob_sq_k).mean().item())

    @torch.no_grad()
    def compute_cross_branch_overlap(self) -> float:
        """Compute mean 4-factor Kronecker overlap `prod_{m=0}^3 |<psi_{v,r}^{(m)} | psi_{v,r'}^{(m)}>|^2` across `r < r'`."""
        r_num = int(self.cfg.num_meanings)
        if r_num < 2:
            return 0.0
        v_eff = int(self.cfg.valid_vocab_size)
        e_re, e_im = self.embed_re[:v_eff].detach().float(), self.embed_im[:v_eff].detach().float()
        inv_n = torch.rsqrt(e_re.square().sum(dim=-1, keepdim=True) + e_im.square().sum(dim=-1, keepdim=True) + DEFAULT_KRON_EPS)
        u_re, u_im = e_re * inv_n, e_im * inv_n

        pair_means: list[torch.Tensor] = []
        for r1 in range(r_num):
            for r2 in range(r1 + 1, r_num):
                ov_re = (u_re[:, r1] * u_re[:, r2] + u_im[:, r1] * u_im[:, r2]).sum(dim=-1)
                ov_im = (u_re[:, r1] * u_im[:, r2] - u_im[:, r1] * u_re[:, r2]).sum(dim=-1)
                pair_means.append((ov_re.square() + ov_im.square()).prod(dim=-1).mean())
        return float(torch.stack(pair_means).mean().item())

    def _pack_token_wavefunctions(self, idx_clamped: torch.Tensor, branch_alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up `(B, T, R, M, d_f)` token factor kets, normalize on `C^16`, weight by `alpha / sqrt(M)`, and pack."""
        b_sz, seq_t = idx_clamped.shape
        r_num, m_num, d_f = int(self.cfg.num_meanings), int(self.cfg.num_factors), int(self.cfg.factor_dim)
        d_c = int(self.cfg.d_complex_total)
        flat_idx = idx_clamped.reshape(-1)
        raw_re = self.embed_re.index_select(0, flat_idx).view(b_sz, seq_t, r_num, m_num, d_f)
        raw_im = self.embed_im.index_select(0, flat_idx).view(b_sz, seq_t, r_num, m_num, d_f)
        f_re, f_im = complex_l2_normalize(raw_re, raw_im, dim=-1)
        alpha_tok = branch_alpha.index_select(0, flat_idx).view(b_sz, seq_t, r_num, 1, 1)
        scale_tok = (alpha_tok * (1.0 / math.sqrt(float(m_num)))).to(dtype=f_re.dtype)
        return (f_re * scale_tok).reshape(b_sz, seq_t, d_c), (f_im * scale_tok).reshape(b_sz, seq_t, d_c)

    def extract_context_readout_states(
        self, idx: torch.Tensor, branch_alpha: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the `L`-layer complex register backbone and return `(phi_factors, branch_beta, branch_alpha)` (`<= 8` args)."""
        b_sz, seq_t = idx.shape
        h_num, dh = int(self.cfg.n_registers), int(self.cfg.d_register)
        v_eff = int(self.cfg.valid_vocab_size)
        idx_clamped = idx.clamp(min=0, max=v_eff - 1)
        alpha_eff = self.compute_branch_alpha() if branch_alpha is None else branch_alpha
        psi0_re, psi0_im = self._pack_token_wavefunctions(idx_clamped, alpha_eff)

        psi_re, psi_im = psi0_re, psi0_im
        v0_re: torch.Tensor | None = None
        v0_im: torch.Tensor | None = None
        use_blk_ckpt = bool(self.cfg.checkpoint_blocks and self.training and torch.is_grad_enabled() and not torch.compiler.is_compiling())

        for l_idx, (block, mixer) in enumerate(zip(self.blocks, self.kron_mixers)):
            if use_blk_ckpt:
                psi_re, psi_im, v_raw_re, v_raw_im = checkpoint(block, psi_re, psi_im, v0_re, v0_im, use_reentrant=False)
            else:
                psi_re, psi_im, v_raw_re, v_raw_im = block(psi_re, psi_im, v0_re=v0_re, v0_im=v0_im)
            if l_idx == 0:
                v0_re, v0_im = v_raw_re.view(b_sz, h_num, seq_t, dh), v_raw_im.view(b_sz, h_num, seq_t, dh)
            psi_re, psi_im = mixer(psi_re, psi_im)

        phi_factors, branch_beta = self.apply_readout_factor_operator(psi_re, psi_im, psi0_re, psi0_im)
        return phi_factors, branch_beta, alpha_eff

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None,
        return_logits: bool | None = None, token_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Run `L` complex register + `16x4` Kronecker mixer layers and Multi-Meaning Kronecker Born readout (`<= 8` args)."""
        b_sz, seq_t = idx.shape
        r_num, m_num, d_2 = int(self.cfg.num_meanings), int(self.cfg.num_factors), 2 * int(self.cfg.factor_dim)
        v_eff = int(self.cfg.valid_vocab_size)

        phi_factors, branch_beta, branch_alpha = self.extract_context_readout_states(idx)
        psi_factors = self.get_tied_vocab_factors()
        prior_w = self.prior_weights[:v_eff]
        hyperparams = torch.tensor(
            [self.cfg.boson_p, self.cfg.holo_alpha, self.cfg.kron_blend, self.cfg.coinc_gamma],
            device=idx.device, dtype=torch.float32,
        )
        want_logits = (targets is None) if return_logits is None else bool(return_logits)

        if targets is not None and not want_logits:
            valid_targets = targets.clamp(min=0, max=v_eff - 1).reshape(-1)
            KroneckerMultiMeaningBornChunkedNLLFn.vocab_chunk_size = int(self.cfg.vocab_chunk_size)
            loss = KroneckerMultiMeaningBornChunkedNLLFn.apply(
                phi_factors, psi_factors, branch_beta, branch_alpha, valid_targets, prior_w, hyperparams, token_weights,
            )
            return (loss if return_logits is None else None), loss

        probs = eval_kronecker_multimeaning_born_probs(
            phi_factors.reshape(b_sz, seq_t, r_num, m_num, d_2), psi_factors,
            branch_beta.reshape(b_sz, seq_t, r_num), branch_alpha, prior_w, hyperparams,
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


def _batch_complex_unitary_terms(w_re: torch.Tensor, w_im: torch.Tensor) -> torch.Tensor:
    """Compute per-matrix complex unitary deviation `(||W_re W_re^T + W_im W_im^T - I||_F^2 + ||W_im W_re^T - W_re W_im^T||_F^2) / d`."""
    wf_re, wf_im = w_re.float(), w_im.float()
    d_w = wf_re.size(-2)
    eye = torch.eye(d_w, device=wf_re.device, dtype=torch.float32)
    gram_re = torch.matmul(wf_re, wf_re.transpose(-1, -2)) + torch.matmul(wf_im, wf_im.transpose(-1, -2)) - eye
    gram_im = torch.matmul(wf_im, wf_re.transpose(-1, -2)) - torch.matmul(wf_re, wf_im.transpose(-1, -2))
    return ((gram_re.square().sum(dim=(-2, -1)) + gram_im.square().sum(dim=(-2, -1))) / float(d_w)).reshape(-1)


def compute_kronecker_manifold_and_isotropy_penalties(
    model: KroneckerMultiMeaning16x4GPT, include_stats: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute smooth 16D Factor-Sphere, Complex-Unitary, Per-Factor Isotropy, & Branch-Orthogonality penalties (`<= 8` args)."""
    v_eff = int(model.cfg.valid_vocab_size)
    r_num, m_num, d_f = int(model.cfg.num_meanings), int(model.cfg.num_factors), int(model.cfg.factor_dim)
    d_2, k_tot = 2 * d_f, r_num * m_num

    e_re, e_im = model.embed_re[:v_eff].float(), model.embed_im[:v_eff].float()
    factor_norm_sq = e_re.square().sum(dim=-1) + e_im.square().sum(dim=-1)
    sphere_pen = (factor_norm_sq - 1.0).square().mean()

    u_batches: list[torch.Tensor] = [_batch_complex_unitary_terms(model.out_factor_u_re, model.out_factor_u_im)]
    for mod in model.modules():
        if isinstance(mod, ComplexLinear) and mod.in_features == mod.out_features:
            u_batches.append(_batch_complex_unitary_terms(mod.weight_re, mod.weight_im))
    for mixer in model.kron_mixers:
        u_batches.append(_batch_complex_unitary_terms(mixer.factor_u_re, mixer.factor_u_im))
        u_batches.append(_batch_complex_unitary_terms(mixer.factor_mix_re, mixer.factor_mix_im))
        u_batches.append(_batch_complex_unitary_terms(mixer.meaning_mix_re, mixer.meaning_mix_im))

    all_u_terms = torch.cat(u_batches, dim=0)
    unitary_raw_sum, unitary_pen = all_u_terms.sum(), all_u_terms.mean()

    inv_norm = torch.rsqrt(factor_norm_sq.unsqueeze(-1) + DEFAULT_KRON_EPS)
    u_re, u_im = e_re * inv_norm, e_im * inv_norm
    psi_k = torch.cat([u_re, u_im], dim=-1).reshape(v_eff, k_tot, d_2).permute(1, 0, 2)
    psi_cent = psi_k - psi_k.mean(dim=1, keepdim=True)
    cov_k = torch.bmm(psi_cent.transpose(1, 2), psi_cent) / float(v_eff)
    target_cov = (torch.eye(d_2, device=cov_k.device, dtype=torch.float32) / float(d_2)).unsqueeze(0)
    factor_iso_pen = (float(d_2) * (cov_k - target_cov).square().sum(dim=(-2, -1))).mean()
    e_flat = psi_cent.permute(1, 0, 2).reshape(v_eff, k_tot * d_2)
    cov_global = (e_flat.t() @ e_flat) / float(v_eff)
    target_global = torch.eye(k_tot * d_2, device=cov_k.device, dtype=torch.float32) / float(d_2)
    iso_pen = factor_iso_pen + float(d_2) * (cov_global - target_global).square().sum()

    if r_num >= 2:
        pair_terms: list[torch.Tensor] = []
        for r1 in range(r_num):
            for r2 in range(r1 + 1, r_num):
                ov_re = (u_re[:, r1] * u_re[:, r2] + u_im[:, r1] * u_im[:, r2]).sum(dim=-1)
                ov_im = (u_re[:, r1] * u_im[:, r2] - u_im[:, r1] * u_re[:, r2]).sum(dim=-1)
                pair_terms.append((ov_re.square() + ov_im.square()).prod(dim=-1).mean())
        branch_orth_pen = torch.stack(pair_terms).mean()
    else:
        branch_orth_pen = torch.zeros((), device=e_re.device, dtype=torch.float32)

    total_penalty = (
        float(model.cfg.sphere_penalty_weight) * sphere_pen
        + float(model.cfg.unitary_penalty_weight) * unitary_raw_sum
        + float(model.cfg.iso_penalty_weight) * iso_pen
        + float(model.cfg.branch_orth_weight) * branch_orth_pen
    )
    if not include_stats:
        return total_penalty, {}
    with torch.no_grad():
        f_norms = torch.sqrt(factor_norm_sq.detach().clamp_min(PROB_LOG_CLAMP_MIN))
        embed_norm_median, embed_norm_max = float(f_norms.median().item()), float(f_norms.max().item())
        complex_unitary_err = float(torch.sqrt(all_u_terms.detach().clamp_min(0.0)).mean().item())
    stats = {
        "sphere_pen": float(sphere_pen.detach().item()), "unitary_pen": float(unitary_pen.detach().item()),
        "iso_pen": float(iso_pen.detach().item()), "branch_orth_pen": float(branch_orth_pen.detach().item()),
        "total_pen": float(total_penalty.detach().item()), "embed_norm_median": embed_norm_median,
        "embed_norm_max": embed_norm_max, "complex_unitary_err": complex_unitary_err,
    }
    return total_penalty, stats


@dataclass(frozen=True)
class KroneckerParameterPartition:
    """Explicit 3-bucket parameter partition for Kronecker complex models (`EXPERIMENT_PLAN_059.md` Track 2)."""

    muon_params: list[nn.Parameter]
    embed_params: list[nn.Parameter]
    gate_params: list[nn.Parameter]
    complex_matrix_pairs: list[tuple[str, nn.Parameter, nn.Parameter]]

    @property
    def muon_complex_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        """Return `(p_re, p_im)` pairs for compatibility with `StiefelMuon(..., complex_pairs=...)`."""
        return [(p_re, p_im) for _, p_re, p_im in self.complex_matrix_pairs]

    @property
    def unclassified_names(self) -> tuple[str, ...]:
        """All trainable parameters are exhaustively routed to `muon`, `embed`, or `gate` (`0` unclassified)."""
        return ()


def collect_named_complex_muon_pairs(
    model: nn.Module, min_dim: int = MIN_KRONECKER_MUON_OPERATOR_DIM,
) -> list[tuple[str, nn.Parameter, nn.Parameter]]:
    """Collect named `(stem, p_re, p_im)` complex matrix operator pairs eligible for `StiefelMuon` (`<= 8` args)."""
    param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    pairs: list[tuple[str, nn.Parameter, nn.Parameter]] = []
    for name, p_re in param_dict.items():
        if not name.endswith("_re"):
            continue
        if name in KRONECKER_EMBED_PARAM_NAMES or any(tok in name for tok in KRONECKER_EMBED_SUBSTRINGS):
            continue
        if name in KRONECKER_GATE_PARAM_NAMES or any(tok in name for tok in KRONECKER_GATE_SUBSTRINGS):
            continue
        if p_re.ndim not in (2, 3) or min(p_re.shape[-2:]) < int(min_dim):
            continue
        stem = name[:-3]
        p_im = param_dict.get(f"{stem}_im")
        if p_im is not None and p_im.shape == p_re.shape:
            pairs.append((stem, p_re, p_im))
    return pairs


def classify_kronecker_model_parameters(
    model: nn.Module, min_operator_dim: int = MIN_KRONECKER_MUON_OPERATOR_DIM,
) -> KroneckerParameterPartition:
    """Partition all trainable Kronecker parameters into `muon_params`, `embed_params`, and `gate_params` (`<= 8` args)."""
    named_pairs = collect_named_complex_muon_pairs(model, min_dim=min_operator_dim)
    paired_ids = {id(p) for _, p_re, p_im in named_pairs for p in (p_re, p_im)}
    muon_params: list[nn.Parameter] = [p for _, p_re, p_im in named_pairs for p in (p_re, p_im)]
    embed_params: list[nn.Parameter] = []
    gate_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in paired_ids:
            continue
        if name in KRONECKER_EMBED_PARAM_NAMES or any(tok in name for tok in KRONECKER_EMBED_SUBSTRINGS):
            embed_params.append(param)
        elif name in KRONECKER_GATE_PARAM_NAMES or any(tok in name for tok in KRONECKER_GATE_SUBSTRINGS):
            gate_params.append(param)
        elif param.ndim in (2, 3) and min(param.shape[-2:]) >= int(min_operator_dim):
            muon_params.append(param)
        else:
            gate_params.append(param)

    return KroneckerParameterPartition(
        muon_params=muon_params, embed_params=embed_params, gate_params=gate_params, complex_matrix_pairs=named_pairs,
    )


def compute_large_batch_muon_lr_scale(
    batch_tokens: int = DEFAULT_MUON_REFERENCE_BATCH_TOKENS,
    reference_batch_tokens: int = DEFAULT_MUON_REFERENCE_BATCH_TOKENS,
) -> float:
    """Compute square-root batch-token learning-rate scaling `clamp(sqrt(B / B_ref), 1.0, 2.5)` (`<= 8` args)."""
    ratio = float(max(1, int(batch_tokens))) / float(max(1, int(reference_batch_tokens)))
    return float(min(max(math.sqrt(ratio), 1.0), 2.5))


def build_kronecker_multimeaning_optimizers(
    model: KroneckerMultiMeaning16x4GPT, base_muon_lr: float = DEFAULT_MUON_LR,
    base_embed_lr: float = DEFAULT_EMBED_LR, base_gate_lr: float = DEFAULT_GATE_LR,
    weight_decay: float = 0.0, stiefel_penalty: float = DEFAULT_KRONECKER_MUON_STIEFEL_PENALTY,
    use_joint_complex_muon: bool = True, batch_tokens: int = DEFAULT_MUON_REFERENCE_BATCH_TOKENS,
) -> tuple[StiefelMuon, torch.optim.AdamW]:
    """Build universal complex `StiefelMuon` (`d >= 2`, `complex_pairs`) + `AdamW` (`<= 8` args)."""
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
    muon_lr = float(base_muon_lr) * muon_scale * b_scale
    embed_lr = float(base_embed_lr) * embed_scale * b_scale
    gate_lr = float(base_gate_lr) * gate_scale

    muon_kwargs: dict[str, Any] = {
        "lr": muon_lr, "momentum": 0.95, "polar_method": "polar_express",
        "stiefel_penalty": float(stiefel_penalty), "retract_every": 1, "use_normuon": True, "use_twin_rail": True,
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
            {"params": partition.gate_params, "lr": gate_lr, "base_lr": gate_lr, "weight_decay": float(weight_decay)},
        ],
        betas=(0.9, 0.95), eps=1e-8,
    )
    return opt_muon, opt_adam


__all__ = [
    "DEFAULT_BRANCH_EPS", "DEFAULT_BRANCH_ORTH_WEIGHT", "DEFAULT_EMBED_LR", "DEFAULT_FACTOR_DIM",
    "DEFAULT_GATE_LR", "DEFAULT_ISO_PENALTY_WEIGHT", "DEFAULT_KRON_BLEND", "DEFAULT_KRON_BOSON_P",
    "DEFAULT_KRON_COINC_GAMMA", "DEFAULT_KRON_EPS", "DEFAULT_KRON_HOLO_ALPHA",
    "DEFAULT_KRONECKER_MUON_STIEFEL_PENALTY", "DEFAULT_KRONECKER_REGISTER_DIM",
    "DEFAULT_MUON_LR", "DEFAULT_MUON_REFERENCE_BATCH_TOKENS", "DEFAULT_NUM_FACTORS",
    "DEFAULT_NUM_MEANINGS", "DEFAULT_REF_PHASE_COS_SIN", "DEFAULT_REGISTER_DIM",
    "DEFAULT_SELF_ECHO_LAMBDA_INIT", "DEFAULT_SELF_ECHO_LOGIT_INIT", "DEFAULT_SELF_ECHO_MODE",
    "DEFAULT_SPHERE_PENALTY_WEIGHT", "DEFAULT_UNIGRAM_FLOOR_COUNT", "DEFAULT_UNIGRAM_TEMP",
    "DEFAULT_UNITARY_PENALTY_WEIGHT", "DEFAULT_VOCAB_CHUNK_SIZE", "KRONECKER_EMBED_PARAM_NAMES",
    "KRONECKER_EMBED_SUBSTRINGS", "KRONECKER_GATE_PARAM_NAMES", "KRONECKER_GATE_SUBSTRINGS",
    "Kronecker16x4UnitaryMeaningMixer", "KroneckerComplexRegisterBlock", "KroneckerMultiMeaning16x4GPT",
    "KroneckerMultiMeaningBornChunkedNLLFn", "KroneckerMultiMeaningConfig", "KroneckerParameterPartition",
    "MIN_KRONECKER_MUON_OPERATOR_DIM", "SELF_ECHO_NORM_EPS", "VALID_SELF_ECHO_MODES",
    "apply_carrier_orthogonal_complex_self_echo", "build_kronecker_multimeaning_optimizers",
    "classify_kronecker_model_parameters", "collect_named_complex_muon_pairs",
    "compute_kronecker_manifold_and_isotropy_penalties", "compute_large_batch_muon_lr_scale",
    "eval_kronecker_multimeaning_born_probs",
]

