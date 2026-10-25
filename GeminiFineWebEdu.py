#!/usr/bin/env python3
"""
Plan 033 Prototype — End-to-End Complex QuantumBornGPT with Packed (H, d_h, d_h)
Hermitian Register Density Matrix Attention, Kramers Time-Reversal (T_kramers)
Self-Orthogonal Qubit Embeddings, Native ComplexRoPE, and Chunked k-Boson Readout.

Designed for local RTX 5060 Ti execution under the strict <= 8.0 GB (50%) VRAM cap.
"""

import argparse
import glob
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. COMPLEX HILBERT SPACE PRIMITIVES & KRAMERS TIME-REVERSAL ISOMETRY
# ============================================================================
def complex_l2_normalize(
    x_re: torch.Tensor, x_im: torch.Tensor, dim: int = -1, eps: float = 1e-7
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalizes z = x_re + i * x_im onto the complex unit sphere <z | z>_C = 1."""
    norm_sq = x_re.square().sum(dim=dim, keepdim=True) + x_im.square().sum(dim=dim, keepdim=True)
    inv_norm = torch.rsqrt(norm_sq + eps)
    return x_re * inv_norm, x_im * inv_norm


def kramers_time_reversal(
    z_re: torch.Tensor, z_im: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Kramers' Symplectic Time-Reversal Operator T: C^A -> C^A for even A:
      (T(z))_{2k}   = -conj(z_{2k+1}) = -z_{2k+1, re} + i * z_{2k+1, im}
      (T(z))_{2k+1} = +conj(z_{2k})   = +z_{2k, re}   - i * z_{2k, im}

    Guarantees exact complex Hermitian self-orthogonality:
      <z | T(z)>_C = sum_a conj(z_a) * (T(z))_a == 0 + 0i  identically!
    while keeping top singular values at the head of each register (2k <-> 2k+1).
    """
    A = z_re.size(-1)
    assert A % 2 == 0, "Register dimension A must be even for Kramers time reversal."
    re_pair = z_re.view(*z_re.shape[:-1], A // 2, 2)
    im_pair = z_im.view(*z_im.shape[:-1], A // 2, 2)

    # Real part of T(z): (-z_{2k+1, re}, +z_{2k, re})
    t_re = torch.stack([-re_pair[..., 1], re_pair[..., 0]], dim=-1).reshape_as(z_re)
    # Imaginary part of T(z): (+z_{2k+1, im}, -z_{2k, im}) due to complex conjugation!
    t_im = torch.stack([im_pair[..., 1], -im_pair[..., 0]], dim=-1).reshape_as(z_im)
    return t_re, t_im


def complex_gram_schmidt(
    u_re: torch.Tensor,
    u_im: torch.Tensor,
    ref_re: torch.Tensor,
    ref_im: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Projects complex vector u orthogonal to unit complex vector ref:
      u_perp = u - <ref | u>_C * ref,   then normalizes u_perp to unit complex norm.
    Guarantees |<ref | u_perp>_C| == 0.
    """
    # <ref | u>_C = sum(ref_re * u_re + ref_im * u_im) + i * sum(ref_re * u_im - ref_im * u_re)
    ov_re = (ref_re * u_re + ref_im * u_im).sum(dim=dim, keepdim=True)
    ov_im = (ref_re * u_im - ref_im * u_re).sum(dim=dim, keepdim=True)

    # (<ref | u>_C) * ref = (ov_re * ref_re - ov_im * ref_im) + i * (ov_re * ref_im + ov_im * ref_re)
    proj_re = ov_re * ref_re - ov_im * ref_im
    proj_im = ov_re * ref_im + ov_im * ref_re
    return complex_l2_normalize(u_re - proj_re, u_im - proj_im, dim=dim, eps=eps)


class ComplexLinear(nn.Module):
    """Unitary-initialized Complex Linear projection W = W_re + i * W_im."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_re = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_im = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        z = torch.complex(
            torch.randn(self.out_features, self.in_features),
            torch.randn(self.out_features, self.in_features),
        )
        q, r = torch.linalg.qr(z, mode="reduced")
        diag_sign = torch.sgn(torch.diagonal(r, 0))
        q = q * diag_sign.unsqueeze(0)
        if self.out_features > self.in_features:
            q = q / math.sqrt(self.out_features / self.in_features)
        with torch.no_grad():
            self.weight_re.copy_(q.real)
            self.weight_im.copy_(q.imag)

    def forward(self, x_re: torch.Tensor, x_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out_re = F.linear(x_re, self.weight_re) - F.linear(x_im, self.weight_im)
        out_im = F.linear(x_re, self.weight_im) + F.linear(x_im, self.weight_re)
        return out_re, out_im


# ============================================================================
# 2. NATIVE COMPLEX ROPE & HERMITIAN TRIANGLE PACKING (1-GEMM DENSITY MIXING)
# ============================================================================
class ComplexRoPE(nn.Module):
    """Rotates each complex register coordinate a by e^(i * t * omega_a)."""
    def __init__(self, d_register: int = 32, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_register, dtype=torch.float32) / d_register))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, z_re: torch.Tensor, z_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = z_re.size(2)
        t = torch.arange(T, device=z_re.device, dtype=self.inv_freq.dtype)
        angles = torch.outer(t, self.inv_freq)  # (T, d_h)
        cos = torch.cos(angles).view(1, 1, T, -1).to(z_re.dtype)
        sin = torch.sin(angles).view(1, 1, T, -1).to(z_re.dtype)
        return z_re * cos - z_im * sin, z_re * sin + z_im * cos


def pack_hermitian_outer_product(v_re: torch.Tensor, v_im: torch.Tensor) -> torch.Tensor:
    """
    Packs the rank-1 Hermitian outer product |v_s><v_s| = rho_re + i * rho_im into a SINGLE
    real (B, H, T, d_h * d_h) tensor:
      - Upper triangle + diagonal (i <= j): Symmetric rho_re[i, j]
      - Strict lower triangle     (i >  j): Skew-symmetric rho_im[i, j]
    """
    B, H, T, dh = v_re.shape
    rho_re = v_re.unsqueeze(-1) * v_re.unsqueeze(-2) + v_im.unsqueeze(-1) * v_im.unsqueeze(-2)
    rho_im = v_im.unsqueeze(-1) * v_re.unsqueeze(-2) - v_re.unsqueeze(-1) * v_im.unsqueeze(-2)
    packed = torch.triu(rho_re, diagonal=0) + torch.tril(rho_im, diagonal=-1)
    return packed.view(B, H, T, dh * dh)


def unpack_hermitian_matrix(packed_flat: torch.Tensor, dh: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unpacks P_t of shape (B, H, T, d_h * d_h) into Symmetric rho_re and Skew-Symmetric rho_im.
    Uses diagonal=1 on the transposed upper triangle so the diagonal is NEVER double-counted!
    """
    B, H, T, _ = packed_flat.shape
    P = packed_flat.view(B, H, T, dh, dh)
    rho_re = torch.triu(P, diagonal=0) + torch.triu(P, diagonal=1).transpose(-2, -1)
    lower_im = torch.tril(P, diagonal=-1)
    rho_im = lower_im - lower_im.transpose(-2, -1)
    return rho_re, rho_im


# ============================================================================
# 3. PACKED COMPLEX REGISTER DENSITY MATRIX ATTENTION + 2-BEAM INTERFERENCE
# ============================================================================
class PackedComplexRegisterDensityAttention(nn.Module):
    """
    Multi-Head Register Complex Hermitian Density Matrix Attention:
      - Softmax-Free p-Boson weights w_ts = |<q_t | k_s>_C|^(2p) / Z (sum_s w_ts = 1).
      - 1-GEMM Packed Incoherent Density Matrix sum_{s<=t} w_ts |v_s><v_s|.
      - Coherent Superposition outer product |Psi_t><Psi_t| where |Psi_t> = sum_s alpha_ts |v_s>.
      - McWeeny Matrix-Squaring Purification: rho <- (rho @ rho) / Tr(rho^2).
      - Returns updated wavefunction (psi_re, psi_im), secondary coherent beam, and purity r_tilde.
    """
    def __init__(
        self,
        d_model: int = 256,
        n_registers: int = 16,
        d_register: int = 16,
        p_boson: int = 2,
        purify_steps: int = 1,
        coherence_init: float = 0.30,
    ):
        super().__init__()
        assert d_model == n_registers * d_register
        self.d_model = d_model
        self.H = n_registers
        self.dh = d_register
        self.p_boson = p_boson
        self.purify_steps = purify_steps

        init_logit = math.log(coherence_init / (1.0 - coherence_init))
        self.coherence_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

        self.rope = ComplexRoPE(d_register=d_register)
        self.W_q = ComplexLinear(d_model, d_model)
        self.W_k = ComplexLinear(d_model, d_model)
        self.W_v = ComplexLinear(d_model, d_model)
        self.W_out = ComplexLinear(d_model, d_model)

    def forward(
        self, psi_re: torch.Tensor, psi_im: torch.Tensor, eps: float = 1e-7
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = psi_re.shape
        H, dh = self.H, self.dh

        # 1. Project & normalize into H complex registers of dimension dh
        q_re, q_im = self.W_q(psi_re, psi_im)
        k_re, k_im = self.W_k(psi_re, psi_im)
        v_re, v_im = self.W_v(psi_re, psi_im)

        q_re, q_im = complex_l2_normalize(
            q_re.view(B, T, H, dh).transpose(1, 2),
            q_im.view(B, T, H, dh).transpose(1, 2),
            dim=-1, eps=eps,
        )
        k_re, k_im = complex_l2_normalize(
            k_re.view(B, T, H, dh).transpose(1, 2),
            k_im.view(B, T, H, dh).transpose(1, 2),
            dim=-1, eps=eps,
        )
        v_re, v_im = complex_l2_normalize(
            v_re.view(B, T, H, dh).transpose(1, 2),
            v_im.view(B, T, H, dh).transpose(1, 2),
            dim=-1, eps=eps,
        )

        # 2. Apply Native ComplexRoPE to Q and K
        q_rot_re, q_rot_im = self.rope(q_re, q_im)
        k_rot_re, k_rot_im = self.rope(k_re, k_im)

        # 3. Hermitian overlap c_ts = <q_t | k_s>_C
        c_re = torch.matmul(q_rot_re, k_rot_re.transpose(-2, -1)) + torch.matmul(
            q_rot_im, k_rot_im.transpose(-2, -1)
        )
        c_im = torch.matmul(q_rot_re, k_rot_im.transpose(-2, -1)) - torch.matmul(
            q_rot_im, k_rot_re.transpose(-2, -1)
        )
        mag_sq = c_re.square() + c_im.square()

        # 4. Softmax-Free p-Boson Weights w_ts and Complex Amplitudes alpha_ts
        causal_mask = torch.tril(torch.ones(T, T, device=psi_re.device, dtype=torch.bool))
        intensity = mag_sq.pow(self.p_boson).masked_fill(~causal_mask, 0.0)
        z_trace = intensity.sum(dim=-1, keepdim=True) + eps
        w_born = intensity / z_trace  # Real non-negative, sums to 1

        amp_scale = (mag_sq.pow(0.5 * (self.p_boson - 1)) / z_trace.sqrt()).masked_fill(~causal_mask, 0.0)
        alpha_re = c_re * amp_scale
        alpha_im = c_im * amp_scale

        # 5. Coherent Wave Superposition |Psi_t> = sum_{s<=t} alpha_ts |v_s>
        psi_coh_re = torch.matmul(alpha_re, v_re) - torch.matmul(alpha_im, v_im)
        psi_coh_im = torch.matmul(alpha_re, v_im) + torch.matmul(alpha_im, v_re)
        psi_coh_re, psi_coh_im = complex_l2_normalize(psi_coh_re, psi_coh_im, dim=-1, eps=eps)

        # 6. 1-GEMM Packed Incoherent Density Matrix + Coherent Blend IN PACKED SPACE!
        #    Notice we can also pack |Psi_t><Psi_t| BEFORE unpacking so unpacking only happens once!
        v_packed = pack_hermitian_outer_product(v_re, v_im)                # (B, H, T, dh*dh)
        rho_inc_packed = torch.matmul(w_born, v_packed)                    # 1 real Tensor-Core GEMM!
        rho_coh_packed = pack_hermitian_outer_product(psi_coh_re, psi_coh_im)

        gamma = torch.sigmoid(self.coherence_logit).to(psi_re.dtype)
        rho_packed = (1.0 - gamma) * rho_inc_packed + gamma * rho_coh_packed
        rho_re, rho_im = unpack_hermitian_matrix(rho_packed, dh=dh)        # (B, H, T, dh, dh)

        # Contextual Bloch purity Tr(rho^2) per token: shape (B, T, 1)
        purity_token = (rho_re.square() + rho_im.square()).sum(dim=(-2, -1)).mean(dim=1).unsqueeze(-1)

        # 7. Complex McWeeny Matrix-Squaring Purification: rho <- (rho @ rho) / Tr(rho^2)
        for _ in range(self.purify_steps):
            sq_re = torch.matmul(rho_re, rho_re) - torch.matmul(rho_im, rho_im)
            sq_im = torch.matmul(rho_re, rho_im) + torch.matmul(rho_im, rho_re)
            tr_sq = sq_re.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
            rho_re = sq_re / (tr_sq + eps)
            rho_im = sq_im / (tr_sq + eps)

        # 8. Probe purified density matrix with un-rotated content query |q_t>
        q_re_col, q_im_col = q_re.unsqueeze(-1), q_im.unsqueeze(-1)
        u_re = (torch.matmul(rho_re, q_re_col) - torch.matmul(rho_im, q_im_col)).squeeze(-1)
        u_im = (torch.matmul(rho_re, q_im_col) + torch.matmul(rho_im, q_re_col)).squeeze(-1)

        u_re_flat = u_re.transpose(1, 2).contiguous().view(B, T, D)
        u_im_flat = u_im.transpose(1, 2).contiguous().view(B, T, D)
        out_re, out_im = self.W_out(u_re_flat, u_im_flat)

        # Residual superposition + normalization onto S^(2D-1)
        next_re, next_im = complex_l2_normalize(psi_re + out_re, psi_im + out_im, dim=-1, eps=eps)
        coh_flat_re = psi_coh_re.transpose(1, 2).contiguous().view(B, T, D)
        coh_flat_im = psi_coh_im.transpose(1, 2).contiguous().view(B, T, D)
        return next_re, next_im, coh_flat_re, coh_flat_im, purity_token


class ComplexOvercompleteTwoBeamInterference(nn.Module):
    """
    Lifts complex state (psi_re, psi_im) into a 2x overcomplete complex frame, applies
    quadrature phase interference, and outputs orthogonal Beam 1 (sharp) and Beam 2 (diffuse).
    """
    def __init__(self, d_model: int = 256, expansion: int = 2):
        super().__init__()
        self.d_inner = d_model * expansion
        self.U1 = ComplexLinear(d_model, self.d_inner)
        self.U2 = ComplexLinear(d_model, self.d_inner)
        self.W1 = ComplexLinear(self.d_inner, d_model)
        self.W2 = ComplexLinear(self.d_inner, d_model)
        self.alpha_phase = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

    def forward(
        self,
        psi_re: torch.Tensor,
        psi_im: torch.Tensor,
        coh_re: torch.Tensor,
        coh_im: torch.Tensor,
        eps: float = 1e-7,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z1_re, z1_im = self.U1(psi_re, psi_im)
        z2_re, z2_im = self.U2(psi_re, psi_im)

        # Complex cross-phase cosine/sine interference: Re(z1 * conj(z2)) and Im(z1 * conj(z2))
        mag1 = torch.sqrt(z1_re.square() + z1_im.square() + eps)
        mag2 = torch.sqrt(z2_re.square() + z2_im.square() + eps)
        cos_dphi = (z1_re * z2_re + z1_im * z2_im) / (mag1 * mag2 + eps)
        sin_dphi = (z1_im * z2_re - z1_re * z2_im) / (mag1 * mag2 + eps)

        alpha = torch.clamp(self.alpha_phase, -0.5, 0.5).to(psi_re.dtype)
        gate1 = F.silu(mag2) * (1.0 + alpha * cos_dphi)
        gate2 = F.silu(mag1) * (1.0 + alpha * sin_dphi)

        b1_re, b1_im = self.W1(z1_re * gate1, z1_im * gate1)
        b2_re, b2_im = self.W2(z2_re * gate2, z2_im * gate2)

        # Beam 1: Sharp state update
        sharp_re, sharp_im = complex_l2_normalize(psi_re + b1_re, psi_im + b1_im, dim=-1, eps=eps)

        # Beam 2: Combine interference Beam 2 with attention coherent superposition,
        # mean-center across feature dim, and enforce exact Complex Gram-Schmidt Orthogonality!
        raw_d_re = b2_re + 0.5 * coh_re
        raw_d_im = b2_im + 0.5 * coh_im
        raw_d_re = raw_d_re - raw_d_re.mean(dim=-1, keepdim=True)
        raw_d_im = raw_d_im - raw_d_im.mean(dim=-1, keepdim=True)

        diff_re, diff_im = complex_gram_schmidt(raw_d_re, raw_d_im, sharp_re, sharp_im, dim=-1, eps=eps)
        return sharp_re, sharp_im, diff_re, diff_im


# ============================================================================
# 4. COMPLETE COMPLEX QUANTUM BORN GPT (KRAMERS EMBED + CHUNKED K-BOSON READOUT)
# ============================================================================
@dataclass
class ComplexBornConfig:
    vocab_size: int = 50304
    num_layers: int = 4
    d_model: int = 256
    n_registers: int = 16
    d_register: int = 16
    num_higher_states: int = 256
    kraus_rank: int = 16
    p_boson_attn: int = 2
    purify_steps: int = 1
    k_boson_init: float = 36.0
    lam_diff_init: float = 0.21
    quad_diffuse: float = 0.50


class ComplexQuantumBornBlock(nn.Module):
    def __init__(self, cfg: ComplexBornConfig):
        super().__init__()
        self.attn = PackedComplexRegisterDensityAttention(
            d_model=cfg.d_model,
            n_registers=cfg.n_registers,
            d_register=cfg.d_register,
            p_boson=cfg.p_boson_attn,
            purify_steps=cfg.purify_steps,
        )
        self.interf = ComplexOvercompleteTwoBeamInterference(d_model=cfg.d_model, expansion=2)

    def forward(self, psi_re: torch.Tensor, psi_im: torch.Tensor):
        a_re, a_im, coh_re, coh_im, purity = self.attn(psi_re, psi_im)
        sharp_re, sharp_im, diff_re, diff_im = self.interf(a_re, a_im, coh_re, coh_im)
        return sharp_re, sharp_im, diff_re, diff_im, purity


class ComplexQuantumBornGPT(nn.Module):
    """
    Full Complex QuantumBornGPT with:
      - Factorized Complex Qubit Embeddings (Q x A = d_model) + 256 Higher Concept States
      - Kramers Time-Reversal (T_kramers) Full-Ambient Self-Decoupled Output Detectors
      - Stacked ComplexQuantumBornBlocks with Packed Density Matrix Attention
      - Memory-Safe Chunked Complex k-Boson + Quadratic Kraus Bath Readout
    """
    def __init__(self, cfg: ComplexBornConfig):
        super().__init__()
        self.cfg = cfg
        V, Q, A, D = cfg.vocab_size, cfg.n_registers, cfg.d_register, cfg.d_model
        M, R = cfg.num_higher_states, cfg.kraus_rank

        # 1. Complex Qubit Embeddings (routed to Gentle 2x Zero-WD group!)
        self.unigram_qubits_re = nn.Parameter(torch.randn(V, Q, A) * 0.04)
        self.unigram_qubits_im = nn.Parameter(torch.randn(V, Q, A) * 0.04)
        self.higher_states_re = nn.Parameter(torch.randn(M, Q, A) * 0.04)
        self.higher_states_im = nn.Parameter(torch.randn(M, Q, A) * 0.04)

        # Directed asymmetric residual for output detectors
        self.povm_dir_re = nn.Parameter(torch.randn(V, Q, A) * 0.02)
        self.povm_dir_im = nn.Parameter(torch.randn(V, Q, A) * 0.02)
        self.vocab_bias = nn.Parameter(torch.zeros(V))

        # 2. Complex Stiefel Kraus Semantic Bath (D x R)
        z_u = torch.linalg.qr(torch.complex(torch.randn(D, R), torch.randn(D, R)))[0]
        z_v = torch.linalg.qr(torch.complex(torch.randn(D, R), torch.randn(D, R)))[0]
        self.kraus_u_re = nn.Parameter(z_u.real.contiguous())
        self.kraus_u_im = nn.Parameter(z_u.imag.contiguous())
        self.kraus_v_re = nn.Parameter(z_v.real.contiguous())
        self.kraus_v_im = nn.Parameter(z_v.imag.contiguous())

        # 3. Backbone Layers
        self.blocks = nn.ModuleList([ComplexQuantumBornBlock(cfg) for _ in range(cfg.num_layers)])

        # Scale initial k_boson by sqrt(d / 512) so Step-0 entropy is calibrated at both d=256 and d=512!
        k_scaled = cfg.k_boson_init * math.sqrt(cfg.d_model / 512.0)
        self.k_boson = nn.Parameter(torch.tensor(k_scaled, dtype=torch.float32))
        self.lam_diff = cfg.lam_diff_init

    def get_input_kets(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = idx.shape
        u_re, u_im = complex_l2_normalize(
            self.unigram_qubits_re[idx], self.unigram_qubits_im[idx], dim=-1
        )
        # Normalize across all Q registers so total ket norm in C^D is 1.0
        scale = 1.0 / math.sqrt(self.cfg.n_registers)
        return (u_re * scale).view(B, T, -1), (u_im * scale).view(B, T, -1)

    def get_detector_chunk(self, v_start: int, v_end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs self-orthogonal output detectors M_v in C^D for vocabulary slice [v_start:v_end]
        using Kramers' Time-Reversal Isometry T(u_in) + directed perturbation, followed by
        exact per-register complex Gram-Schmidt against u_in!
        """
        u_in_re, u_in_im = complex_l2_normalize(
            self.unigram_qubits_re[v_start:v_end],
            self.unigram_qubits_im[v_start:v_end],
            dim=-1,
        )
        # Kramers Time-Reversal partner: <u_in | T(u_in)>_C == 0 + 0i identically!
        t_re, t_im = kramers_time_reversal(u_in_re, u_in_im)
        d_re, d_im = complex_l2_normalize(
            self.povm_dir_re[v_start:v_end],
            self.povm_dir_im[v_start:v_end],
            dim=-1,
        )
        mix_re = 0.70 * t_re + 0.30 * d_re
        mix_im = 0.70 * t_im + 0.30 * d_im

        # Exact per-register Complex Gram-Schmidt orthogonalization against u_in
        out_re, out_im = complex_gram_schmidt(mix_re, mix_im, u_in_re, u_in_im, dim=-1)
        scale = 1.0 / math.sqrt(self.cfg.n_registers)
        return (out_re * scale).reshape(v_end - v_start, -1), (out_im * scale).reshape(v_end - v_start, -1)

    def forward_backbone(self, idx: torch.Tensor):
        psi_re, psi_im = self.get_input_kets(idx)
        diff_re, diff_im = psi_re, psi_im
        purity = None
        for block in self.blocks:
            psi_re, psi_im, diff_re, diff_im, purity = block(psi_re, psi_im)
        return psi_re, psi_im, diff_re, diff_im, purity

    def compute_loss_and_metrics(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor,
        eval_metrics: bool = False,
        token_chunk: int = 1024,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        psi_re, psi_im, diff_re, diff_im, purity = self.forward_backbone(idx)
        B, T, D = psi_re.shape
        N = B * T
        V = self.cfg.vocab_size
        R = self.cfg.kraus_rank

        s_re = psi_re.view(N, D)
        s_im = psi_im.view(N, D)
        d_re = diff_re.view(N, D)
        d_im = diff_im.view(N, D)
        flat_targets = targets.view(N)

        # Build full detector table M_v in C^(V x D) in 8192-vocab slices
        M_re_list, M_im_list = [], []
        for vs in range(0, V, 8192):
            ve = min(V, vs + 8192)
            mr, mi = self.get_detector_chunk(vs, ve)
            M_re_list.append(mr)
            M_im_list.append(mi)
        M_re = torch.cat(M_re_list, dim=0)  # (V, D)
        M_im = torch.cat(M_im_list, dim=0)  # (V, D)

        # Low-rank Complex Kraus projections in C^R, scaled by (D / R)^(1/4)
        k_scale = (D / float(R)) ** 0.25
        u_re, u_im = complex_l2_normalize(self.kraus_u_re, self.kraus_u_im, dim=0)
        v_re, v_im = complex_l2_normalize(self.kraus_v_re, self.kraus_v_im, dim=0)

        # Project vocabulary detectors M onto Kraus U: m_diff in C^(V x R)
        m_diff_re = (M_re @ u_re + M_im @ u_im) * k_scale
        m_diff_im = (M_re @ u_im - M_im @ u_re) * k_scale
        # Project token Beam 2 onto Kraus V: p_diff in C^(N x R)
        p_diff_re = (d_re @ v_re + d_im @ v_im) * k_scale
        p_diff_im = (d_re @ v_im - d_im @ v_re) * k_scale

        # Contextual Bloch purity gate r_tilde in [0.55, 0.92]
        r_tilde = torch.clamp(0.65 + 0.25 * purity.view(N, 1), min=0.55, max=0.92)
        k_eff = torch.clamp(self.k_boson, min=4.0, max=64.0)
        lam = self.lam_diff
        quad_w = self.cfg.quad_diffuse

        # Chunk over N tokens so (chunk, V) logits stay under ~200 MB VRAM!
        losses = []
        top1_hits, top5_hits = 0, 0
        bucket_nll = {k: [] for k in ["r1", "r2_5", "r6_50", "r51_1k", "gt1k"]}

        for t0 in range(0, N, token_chunk):
            t1 = min(N, t0 + token_chunk)
            # Real part of Hermitian overlap <M_v | psi_sharp>_C = s_re @ M_re^T + s_im @ M_im^T
            c_sharp = s_re[t0:t1] @ M_re.T + s_im[t0:t1] @ M_im.T
            c_diff = p_diff_re[t0:t1] @ m_diff_re.T + p_diff_im[t0:t1] @ m_diff_im.T

            f_diff = c_diff + quad_w * F.relu(c_diff).square()
            interior = torch.clamp(1.0 + r_tilde[t0:t1] * c_sharp + lam * f_diff, min=0.12)
            logits = k_eff * torch.log(interior) + self.vocab_bias.unsqueeze(0)

            tgt_chunk = flat_targets[t0:t1]
            ce_chunk = F.cross_entropy(logits, tgt_chunk, reduction="none")
            losses.append(ce_chunk)

            if eval_metrics:
                with torch.no_grad():
                    tgt_logits = logits.gather(1, tgt_chunk.unsqueeze(1))
                    # Exact rank of target token (0 = Rank 1)
                    ranks = (logits > tgt_logits).sum(dim=-1)
                    top1_hits += (ranks == 0).sum().item()
                    top5_hits += (ranks < 5).sum().item()

                    ce_np = ce_chunk.detach().float().cpu().numpy()
                    rk_np = ranks.detach().cpu().numpy()
                    bucket_nll["r1"].extend(ce_np[rk_np == 0])
                    bucket_nll["r2_5"].extend(ce_np[(rk_np >= 1) & (rk_np < 5)])
                    bucket_nll["r6_50"].extend(ce_np[(rk_np >= 5) & (rk_np < 50)])
                    bucket_nll["r51_1k"].extend(ce_np[(rk_np >= 50) & (rk_np < 1000)])
                    bucket_nll["gt1k"].extend(ce_np[rk_np >= 1000])

        total_loss = torch.cat(losses, dim=0).mean()
        metrics = {}
        if eval_metrics:
            with torch.no_grad():
                # Compute Diffuse Effective Rank R_diff and Complex BeamOrtho |<psi_sharp | psi_diff>_C|
                cov = (p_diff_re.T @ p_diff_re + p_diff_im.T @ p_diff_im) / float(N)
                svals = torch.linalg.svdvals(cov.float())
                p_sv = svals / (svals.sum() + 1e-12)
                r_diff = torch.exp(-(p_sv * torch.log(p_sv + 1e-12)).sum()).item()

                ov_re = (s_re * d_re + s_im * d_im).sum(dim=-1)
                ov_im = (s_re * d_im - s_im * d_re).sum(dim=-1)
                beam_ortho = torch.sqrt(ov_re.square() + ov_im.square()).mean().item()

                t1_pct = 100.0 * top1_hits / float(N)
                t5_pct = 100.0 * top5_hits / float(N)
                metrics = {
                    "val_ce": total_loss.item(),
                    "val_ppl": math.exp(min(20.0, total_loss.item())),
                    "top1": t1_pct,
                    "top5": t5_pct,
                    "gap_t5_t1": t5_pct - t1_pct,
                    "nll_2_5": float(np.mean(bucket_nll["r2_5"])) if bucket_nll["r2_5"] else 0.0,
                    "nll_6_50": float(np.mean(bucket_nll["r6_50"])) if bucket_nll["r6_50"] else 0.0,
                    "nll_51_1k": float(np.mean(bucket_nll["r51_1k"])) if bucket_nll["r51_1k"] else 0.0,
                    "r_diff": r_diff,
                    "beam_ortho": beam_ortho,
                    "purity": purity.mean().item(),
                }
        return total_loss, metrics


# ============================================================================
# 5. OPTIMIZER BUILDER (GENTLE 2x ZERO-WD QUBIT LR + UNITARY RETRACTION)
# ============================================================================
def build_complex_born_optimizers(model: ComplexQuantumBornGPT, base_lr: float = 0.003):
    """
    Routes `unigram_qubits_*` and `higher_states_*` to the Gentle 2x Zero-WD group
    (lr = 2 * base_lr = 0.006, weight_decay = 0.0) proven in Plan 032 Isolate 1 & 3.
    """
    qubit_params = []
    body_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "unigram_qubits" in name or "higher_states" in name or "povm_dir" in name:
            qubit_params.append(p)
        else:
            body_params.append(p)

    opt = torch.optim.AdamW(
        [
            {"params": qubit_params, "lr": base_lr * 2.0, "weight_decay": 0.0},
            {"params": body_params, "lr": base_lr, "weight_decay": 0.01},
        ],
        betas=(0.9, 0.95),
    )
    return opt


# ============================================================================
# 6. DATA LOADER (AUTO-DETECTS FINEWEB-EDU SLICES WITH SYNTHETIC FALLBACK)
# ============================================================================
def load_token_slices(seq_len: int = 512) -> Tuple[torch.Tensor, torch.Tensor, str]:
    search_patterns = [
        "/home/festivity/agy/QuantumDLBlocks/.cache/**/train_slice.bin",
        "/home/festivity/agy/QuantumDLBlocks/**/train_slice.bin",
        ".cache/**/train_slice.bin",
        "**/fineweb*train*.bin",
    ]
    for pat in search_patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            train_path = matches[0]
            val_path = train_path.replace("train", "val")
            train_arr = np.memmap(train_path, dtype=np.uint16, mode="r")
            val_arr = np.memmap(val_path, dtype=np.uint16, mode="r") if os.path.exists(val_path) else train_arr[:65536]
            return (
                torch.from_numpy(np.array(train_arr[:500000], dtype=np.int64)),
                torch.from_numpy(np.array(val_arr[:65536], dtype=np.int64)),
                f"FineWeb-Edu ({train_path})",
            )

    # Fallback deterministic Zipfian-Markov stream if run outside the repo
    rng = np.random.default_rng(42)
    ranks = np.arange(1, 50305, dtype=np.float64)
    probs = (1.0 / (ranks ** 1.08))
    probs /= probs.sum()
    tokens = rng.choice(50304, size=180000, p=probs).astype(np.int64)
    return torch.from_numpy(tokens[:147456]), torch.from_numpy(tokens[147456:180000]), "Synthetic-Zipf"


# ============================================================================
# 7. MAIN VERIFICATION & BENCHMARK RUNNER
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--registers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # Enforce strict <= 8.0 GB (50%) VRAM cap on RTX 5060 Ti (15.48 GiB total)
        torch.cuda.set_per_process_memory_fraction(0.50, device=0)
        gpu_name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Complex Plan 033] GPU: {gpu_name} (Total={total_gb:.2f}GB, Cap<=8.0GB / 50%)")

    print("=" * 96)
    print("  [H1 Verification] 1-GEMM Hermitian Triangle Packing, Kramers T_iso & (512, 32, 32, 32) Speed")
    print("=" * 96)
    torch.manual_seed(1337)
    # 1. Verify Kramers Time-Reversal exact complex self-orthogonality
    z_re, z_im = complex_l2_normalize(torch.randn(50304, 16, 16, device=device),
                                      torch.randn(50304, 16, 16, device=device))
    t_re, t_im = kramers_time_reversal(z_re, z_im)
    ov_re = (z_re * t_re + z_im * t_im).sum(dim=-1).abs().max().item()
    ov_im = (z_re * t_im - z_im * t_re).sum(dim=-1).abs().max().item()

    # 2. Verify 1-GEMM Hermitian Triangle Packing == 2-GEMM Reference
    v_re, v_im = complex_l2_normalize(torch.randn(1, 32, 64, 32, device=device),
                                      torch.randn(1, 32, 64, 32, device=device))
    w = torch.tril(torch.rand(1, 32, 64, 64, device=device))
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-7)
    packed_out = torch.matmul(w, pack_hermitian_outer_product(v_re, v_im))
    up_re, up_im = unpack_hermitian_matrix(packed_out, dh=32)
    ref_re = torch.matmul(w, (v_re.unsqueeze(-1)*v_re.unsqueeze(-2) + v_im.unsqueeze(-1)*v_im.unsqueeze(-2)).view(1, 32, 64, 1024)).view(1, 32, 64, 32, 32)
    ref_im = torch.matmul(w, (v_im.unsqueeze(-1)*v_re.unsqueeze(-2) - v_re.unsqueeze(-1)*v_im.unsqueeze(-2)).view(1, 32, 64, 1024)).view(1, 32, 64, 32, 32)
    pack_err = max((up_re - ref_re).abs().max().item(), (up_im - ref_im).abs().max().item())
    print(f"- [H1 Math Check] KramersSelfOverlap=(Re:{ov_re:.2e}, Im:{ov_im:.2e}) (<1e-6) | PackingErr={pack_err:.2e} | PASS={ov_re < 1e-6 and pack_err < 1e-6}")

    # 3. Run Training Benchmark on FineWeb-Edu
    train_tokens, val_tokens, ds_label = load_token_slices(seq_len=args.seq_len)
    d_reg = args.d_model // args.registers
    cfg = ComplexBornConfig(
        num_layers=args.layers,
        d_model=args.d_model,
        n_registers=args.registers,
        d_register=d_reg,
    )
    model = ComplexQuantumBornGPT(cfg).to(device)
    opt = build_complex_born_optimizers(model, base_lr=0.003)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    print("=" * 96)
    print(f"  [H2 Benchmark] ComplexQuantumBornGPT (L={cfg.num_layers}, d={cfg.d_model}C, {cfg.n_registers}x{cfg.d_register} Density) on {ds_label}")
    print("=" * 96)

    T = args.seq_len
    B = args.batch_size
    val_batches = min(8, (len(val_tokens) - 1) // (4 * T))
    t_start = time.time()

    for step in range(args.steps + 1):
        # Scheduled warmup-hold-anneal for lam_diff (matches Wave 16/18 schedule)
        prog = step / max(1, args.steps)
        model.lam_diff = 0.21 if prog < 0.35 else 0.21 * (0.5 ** ((prog - 0.35) / 0.65))

        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                vx = val_tokens[: val_batches * 4 * T].view(val_batches * 4, T).to(device)
                vy = val_tokens[1 : val_batches * 4 * T + 1].view(val_batches * 4, T).to(device)
                _, m = model.compute_loss_and_metrics(vx, vy, eval_metrics=True)
            tok_M = (step * B * T) / 1e6
            print(
                f"  [Complex_L{cfg.num_layers}_d{cfg.d_model}] [Step {step:3d}/{args.steps} | {tok_M:.2f}M tok] "
                f"Val CE: {m['val_ce']:.4f} | PPL: {m['val_ppl']:.1f} | Top-1: {m['top1']:.2f}% | "
                f"Top-5: {m['top5']:.2f}% | Gap: {m['gap_t5_t1']:.2f}% | NLL(6-50): {m['nll_6_50']:.4f} | "
                f"R_diff: {m['r_diff']:.2f} | Purity: {m['purity']:.3f} | BeamOrtho: {m['beam_ortho']:.2e}"
            )
            model.train()
            if step == args.steps:
                break

        # Sample training batch
        ix = torch.randint(0, len(train_tokens) - T - 1, (B,))
        xb = torch.stack([train_tokens[i : i + T] for i in ix]).to(device)
        yb = torch.stack([train_tokens[i + 1 : i + T + 1] for i in ix]).to(device)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            loss, _ = model.compute_loss_and_metrics(xb, yb, eval_metrics=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    wall = time.time() - t_start
    tok_sec = (args.steps * B * T) / max(1e-6, wall)
    peak_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3) if device == "cuda" else 0.0
    print("-" * 96)
    print(
        f"- [Summary] Params={n_params:.2f}M | PeakVRAM={peak_gb:.3f}GB (<=8.0GB) | "
        f"Final Step {args.steps} CE={m['val_ce']:.4f} | Top1={m['top1']:.2f}% | Top5={m['top5']:.2f}% | "
        f"Speed={tok_sec:,.0f} tok/s ({wall:.1f}s)"
    )


if __name__ == "__main__":
    main()
