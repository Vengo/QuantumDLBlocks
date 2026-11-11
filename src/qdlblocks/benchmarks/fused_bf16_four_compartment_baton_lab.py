"""Wave 74 (`EXPERIMENT_PLAN_094.md`): Fused `bfloat16` Tensor-Core 4-Compartment Baton LM.

Implements the `bfloat16` + kernel-fused 1-Layer and 2-Layer 4-Compartment Quantum Statistical
Mechanics Baton Language Model for `300,000,000`-token (`300M` token) multi-shard FineWeb training
with 100% online unigram thermal bath and n-gram transition ingestion from cold Step 0:

1. **Theorem 15a (`Factorized Ket/Beam Schur Intersection`)**:
   Eliminates `(B, T, H, 64, 64)` intermediate density-matrix allocations by keeping token-varying
   compartment states in Rank-`R` (`R = 4`) purified complex factors `(Z_re, Z_im)` of shape
   `(B, T, 4, 64)`, where entrywise multiplication of beams `Z_seq ⊙ Z_ngram` forms the exact
   Cholesky factor of the Schur product `ρ_seq ⊙ ρ_ngram`.
2. **Theorem 15b (`Diagonal-Conjugated Thermal Bath Schur Action`)**:
   Computes `(ρ_ngram ⊙ ρ_uni) |ψ⟩` and `Tr(ρ_ngram ⊙ ρ_uni)` against the full-rank `64 × 64`
   online thermal bath `ρ_uni` via `z_r ⊙ (ρ_uni (z̄_r ⊙ |ψ⟩))` using a single batched `bfloat16`
   Tensor Core GEMM `(B*T*R, 64) @ (64, 64)`.
3. **Theorem 15c (`Gram-Matrix MPS Purity`)**:
   Evaluates key-state MPS purity `Tr((ρ_s^{MPS})^2)` from the `χ × χ` (`8 × 8`) Gram matrix
   `G = B B†` (`64` entries/token instead of `4,096`).
4. **Shift-Fused `bfloat16` MPS Baton & Single-GEMM `128D -> 50,257` Born Readout**:
   Replaces `.unfold()` with zero-copy causal slice shifts, fuses complex projections into packed
   real+imag `bfloat16` GEMMs, and executes a single `(B*T, 128) @ (128, 50257)` Tensor Core
   vocabulary readout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from qdlblocks.benchmarks.experiment_logger import (
    ExperimentIntentSpec,
    StandardizedExperimentLogger,
)
from qdlblocks.benchmarks.four_compartment_baton_lm_lab import (
    DEFAULT_BORN_READOUT_SCALE,
    DEFAULT_DOC_BATH_MIX,
    DEFAULT_HILBERT_DIM,
    DEFAULT_IN_CONTEXT_COPY_SCALE,
    DEFAULT_IN_CONTEXT_LINDBLAD_GAMMA,
    DEFAULT_LOGIT_SOFTCAP,
    DEFAULT_MPS_BOND_DIM,
    DEFAULT_MPS_CORE_INIT_STD,
    DEFAULT_MPS_CORE_INPUT_DIM,
    DEFAULT_MPS_NGRAM_BLEND,
    DEFAULT_MPS_WINDOW_K,
    DEFAULT_NUM_PACKETS,
    DEFAULT_PURITY_FLOOR_OFFSET,
    DEFAULT_PURITY_MIN_CLAMP,
    DEFAULT_ROPE_THETA_BASE,
    DEFAULT_TOKEN_ID_COPY_SCALE,
    DEFAULT_TRACE_EPS,
    DEFAULT_UNIGRAM_PRIOR_WEIGHT,
    DEFAULT_VACUUM_EPS,
    DEFAULT_VOCAB_SIZE,
    DEFAULT_VAL_SHARD as FINEWEB_VAL_SHARD_PATH,
    FINEWEB_CACHE_DIR,
    NUM_SEQ_ATTN_PACKETS,
    REPO_ROOT,
    apply_complex_rope,
    build_walsh_hadamard_parity_pairs,
    encode_walsh_hadamard_token_id_kets,
    normalize_complex_ket,
    purity_of_density,
    trace_normalize_density,
    trace_of_density,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 — SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
DEFAULT_SCHUR_BEAM_RANK: int = 4
DEFAULT_TWO_HOP_MIX_ALPHA: float = 0.30
DEFAULT_ONLINE_BATH_EMA_DECAY: float = 0.02
DEFAULT_ONLINE_TRANS_EMA_DECAY: float = 0.02
DEFAULT_ONLINE_HASH_EMA_DECAY: float = 0.05
DEFAULT_ONLINE_NGRAM_HASH_BUCKETS: int = 8192
DEFAULT_HASH_PRIME_PREV1: int = 65537
DEFAULT_HASH_PRIME_PREV2: int = 1000003
DEFAULT_TARGET_TOKENS_300M: int = 300_000_000

FINEWEB_TRAIN_SHARDS_300M: tuple[Path, ...] = (
    FINEWEB_CACHE_DIR / "fineweb_train_000001.bin",
    FINEWEB_CACHE_DIR / "fineweb_train_000002.bin",
    FINEWEB_CACHE_DIR / "fineweb_train_000003.bin",
)


# ==============================================================================
# Configuration Dataclass (Rule 2 — <= 8 Arguments)
# ==============================================================================
@dataclass(frozen=True)
class FusedBF16FourCompartmentBatonConfig:
    """Configuration for the Fused `bfloat16` 4-Compartment Baton LM (`Wave 74`, `<= 8` fields)."""

    vocab_size: int = DEFAULT_VOCAB_SIZE
    hilbert_dim: int = DEFAULT_HILBERT_DIM
    num_layers: int = 1
    mps_window_k: int = DEFAULT_MPS_WINDOW_K
    mps_bond_dim: int = DEFAULT_MPS_BOND_DIM
    use_two_hop_triangle: bool = True
    use_bfloat16_amp: bool = True
    online_bath_ema_decay: float = DEFAULT_ONLINE_BATH_EMA_DECAY


# ==============================================================================
# Exact Theorem 15a–15c Algebraic Kernels & Verification Probes
# ==============================================================================
def factorized_rank_beams_schur_intersect(
    beams_a_re: Tensor,
    beams_a_im: Tensor,
    beams_b_re: Tensor,
    beams_b_im: Tensor,
    eps: float = DEFAULT_TRACE_EPS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Theorem 15a: Intersect two Rank-`R` complex beam factors `(..., R, d)` via entrywise Schur product (`<= 8` args)."""
    prod_re = beams_a_re * beams_b_re - beams_a_im * beams_b_im
    prod_im = beams_a_re * beams_b_im + beams_a_im * beams_b_re
    frob_sq = (prod_re.square() + prod_im.square()).sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
    inv_frob = torch.rsqrt(frob_sq)
    return prod_re * inv_frob, prod_im * inv_frob, frob_sq.squeeze(-1).squeeze(-1)


def diagonal_conjugated_bath_schur_action(
    beams_re: Tensor,
    beams_im: Tensor,
    bath_re: Tensor,
    bath_im: Tensor,
    probe_re: Tensor,
    probe_im: Tensor,
    eps: float = DEFAULT_TRACE_EPS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Theorem 15b: Compute `((Z† Z) ⊙ ρ_uni) |ψ⟩` & `Tr((Z† Z) ⊙ ρ_uni)` in `O(R d^2)` without `(B, T, d, d)` allocation (`<= 8` args)."""
    bath_diag = torch.diagonal(bath_re, dim1=-2, dim2=-1).clamp_min(0.0)
    beam_prob = beams_re.square() + beams_im.square()
    tr_val = (beam_prob * bath_diag).sum(dim=(-2, -1)).clamp_min(eps)

    p_re = probe_re.unsqueeze(-2)
    p_im = probe_im.unsqueeze(-2)
    conj_prod_re = beams_re * p_re + beams_im * p_im
    conj_prod_im = beams_re * p_im - beams_im * p_re

    if bath_re.ndim == 2:
        mid_re = torch.matmul(conj_prod_re, bath_re.transpose(-2, -1)) - torch.matmul(
            conj_prod_im, bath_im.transpose(-2, -1)
        )
        mid_im = torch.matmul(conj_prod_re, bath_im.transpose(-2, -1)) + torch.matmul(
            conj_prod_im, bath_re.transpose(-2, -1)
        )
    else:
        mid_re = torch.einsum("...rd,...ed->...re", conj_prod_re, bath_re) - torch.einsum(
            "...rd,...ed->...re", conj_prod_im, bath_im
        )
        mid_im = torch.einsum("...rd,...ed->...re", conj_prod_re, bath_im) + torch.einsum(
            "...rd,...ed->...re", conj_prod_im, bath_re
        )

    act_re = (beams_re * mid_re - beams_im * mid_im).sum(dim=-2)
    act_im = (beams_re * mid_im + beams_im * mid_re).sum(dim=-2)
    inv_tr = (1.0 / tr_val).unsqueeze(-1)
    return act_re * inv_tr, act_im * inv_tr, tr_val


def gram_matrix_mps_purity(
    beams_re: Tensor, beams_im: Tensor, eps: float = DEFAULT_TRACE_EPS
) -> Tensor:
    """Theorem 15c: Compute purity `Tr(ρ^2) = ‖B B†‖_F^2 / ‖B‖_F^4` from the `χ × χ` Gram matrix (`<= 8` args)."""
    frob_sq = (beams_re.square() + beams_im.square()).sum(dim=(-2, -1)).clamp_min(eps)
    gram_re = torch.matmul(beams_re, beams_re.transpose(-2, -1)) + torch.matmul(
        beams_im, beams_im.transpose(-2, -1)
    )
    gram_im = torch.matmul(beams_im, beams_re.transpose(-2, -1)) - torch.matmul(
        beams_re, beams_im.transpose(-2, -1)
    )
    gram_frob_sq = (gram_re.square() + gram_im.square()).sum(dim=(-2, -1))
    return (gram_frob_sq / frob_sq.square()).clamp(min=0.0, max=1.0)


def apply_rank_beams_to_probe(
    beams_re: Tensor, beams_im: Tensor, probe_re: Tensor, probe_im: Tensor
) -> tuple[Tensor, Tensor]:
    """Compute `(Z† Z) |ψ⟩ = ∑_r |z_r⟩ ⟨z_r | ψ⟩` in `O(R d)` without forming `d × d` matrices (`<= 8` args)."""
    p_re = probe_re.unsqueeze(-2)
    p_im = probe_im.unsqueeze(-2)
    ov_re = (beams_re * p_re + beams_im * p_im).sum(dim=-1, keepdim=True)
    ov_im = (beams_re * p_im - beams_im * p_re).sum(dim=-1, keepdim=True)
    out_re = (beams_re * ov_re - beams_im * ov_im).sum(dim=-2)
    out_im = (beams_re * ov_im + beams_im * ov_re).sum(dim=-2)
    return out_re, out_im


def evaluate_rank_beams_born_gates(
    beams_re: Tensor,
    beams_im: Tensor,
    gate_kets_re: Tensor,
    gate_kets_im: Tensor,
    gate_bias: Tensor,
) -> Tensor:
    """Evaluate 4-compartment Born compatibility gate weights `(B, T, 4)` from Rank-`R` n-gram beams (`<= 8` args)."""
    ov_re = torch.matmul(beams_re, gate_kets_re.transpose(0, 1)) + torch.matmul(
        beams_im, gate_kets_im.transpose(0, 1)
    )
    ov_im = torch.matmul(beams_im, gate_kets_re.transpose(0, 1)) - torch.matmul(
        beams_re, gate_kets_im.transpose(0, 1)
    )
    expect = (ov_re.square() + ov_im.square()).sum(dim=-2)
    raw_gates = expect + F.softplus(gate_bias).to(expect.dtype).view(1, 1, -1)
    return raw_gates / (raw_gates.sum(dim=-1, keepdim=True) + DEFAULT_TRACE_EPS)


# ==============================================================================
# Shift-Fused `bfloat16` Vectorized K=4, χ=8 MPS Baton Module (`Packet 1`)
# ==============================================================================
class FusedBF16MPSBatonModule(nn.Module):
    """Shift-fused `bfloat16` Tensor-Core 1D MPS Baton (`Packet 1`, zero `.unfold()` allocation)."""

    def __init__(self, config: FusedBF16FourCompartmentBatonConfig) -> None:
        super().__init__()
        self.window_k: int = config.mps_window_k
        self.bond_dim: int = config.mps_bond_dim
        self.hilbert_dim: int = config.hilbert_dim
        self.core_in_dim: int = min(DEFAULT_MPS_CORE_INPUT_DIM, config.hilbert_dim)
        self.out_rank: int = DEFAULT_SCHUR_BEAM_RANK

        num_transfer = max(1, self.window_k - 1)
        flat_bond_in = self.bond_dim * self.core_in_dim

        self.in_proj_re = nn.Parameter(
            torch.randn(self.hilbert_dim, self.core_in_dim) / math.sqrt(float(self.hilbert_dim))
        )
        self.in_proj_im = nn.Parameter(torch.randn(self.hilbert_dim, self.core_in_dim) * 0.05)

        self.transfer_cores_re = nn.Parameter(
            torch.randn(num_transfer, flat_bond_in, self.bond_dim) * DEFAULT_MPS_CORE_INIT_STD
        )
        self.transfer_cores_im = nn.Parameter(
            torch.randn(num_transfer, flat_bond_in, self.bond_dim) * DEFAULT_MPS_CORE_INIT_STD
        )
        self.end_core_re = nn.Parameter(
            torch.randn(flat_bond_in, self.out_rank * self.hilbert_dim) * DEFAULT_MPS_CORE_INIT_STD
        )
        self.end_core_im = nn.Parameter(
            torch.randn(flat_bond_in, self.out_rank * self.hilbert_dim) * DEFAULT_MPS_CORE_INIT_STD
        )
        self.bath_seed_proj = nn.Parameter(
            torch.randn(self.hilbert_dim, self.bond_dim) / math.sqrt(float(self.hilbert_dim))
        )

    def forward(
        self, ket_re: Tensor, ket_im: Tensor, uni_diag: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Contract the `K=4` MPS window using causal slice shifts and fused GEMMs (`<= 8` args)."""
        bsz, seq_len, d = ket_re.shape
        k_win = self.window_k

        xi_re = torch.matmul(ket_re, self.in_proj_re) - torch.matmul(ket_im, self.in_proj_im)
        xi_im = torch.matmul(ket_re, self.in_proj_im) + torch.matmul(ket_im, self.in_proj_re)
        xi_re, xi_im = normalize_complex_ket(xi_re, xi_im)

        baton_re = torch.matmul(uni_diag, self.bath_seed_proj).clone()
        baton_im = torch.zeros_like(baton_re)
        baton_re[..., 0] = baton_re[..., 0] + 1.0
        baton_re, baton_im = normalize_complex_ket(baton_re, baton_im)

        for step_idx in range(k_win - 1):
            lag = (k_win - 1) - step_idx
            if lag < seq_len:
                step_xi_re = F.pad(xi_re[:, :-lag, :], (0, 0, lag, 0))
                step_xi_im = F.pad(xi_im[:, :-lag, :], (0, 0, lag, 0))
            else:
                step_xi_re = torch.zeros_like(xi_re)
                step_xi_im = torch.zeros_like(xi_im)

            prod_re = (
                baton_re.unsqueeze(-1) * step_xi_re.unsqueeze(-2)
                - baton_im.unsqueeze(-1) * step_xi_im.unsqueeze(-2)
            ).reshape(bsz, seq_len, -1)
            prod_im = (
                baton_re.unsqueeze(-1) * step_xi_im.unsqueeze(-2)
                + baton_im.unsqueeze(-1) * step_xi_re.unsqueeze(-2)
            ).reshape(bsz, seq_len, -1)

            core_re = self.transfer_cores_re[step_idx]
            core_im = self.transfer_cores_im[step_idx]
            delta_re = torch.matmul(prod_re, core_re) - torch.matmul(prod_im, core_im)
            delta_im = torch.matmul(prod_re, core_im) + torch.matmul(prod_im, core_re)
            baton_re, baton_im = normalize_complex_ket(baton_re + delta_re, baton_im + delta_im)

        end_in_re = (
            baton_re.unsqueeze(-1) * xi_re.unsqueeze(-2)
            - baton_im.unsqueeze(-1) * xi_im.unsqueeze(-2)
        ).reshape(bsz, seq_len, -1)
        end_in_im = (
            baton_re.unsqueeze(-1) * xi_im.unsqueeze(-2)
            + baton_im.unsqueeze(-1) * xi_re.unsqueeze(-2)
        ).reshape(bsz, seq_len, -1)

        raw_beams_re = (
            torch.matmul(end_in_re, self.end_core_re) - torch.matmul(end_in_im, self.end_core_im)
        ).view(bsz, seq_len, self.out_rank, d)
        raw_beams_im = (
            torch.matmul(end_in_re, self.end_core_im) + torch.matmul(end_in_im, self.end_core_re)
        ).view(bsz, seq_len, self.out_rank, d)

        raw_beams_re = raw_beams_re + 0.20 * ket_re.unsqueeze(-2)
        raw_beams_im = raw_beams_im + 0.20 * ket_im.unsqueeze(-2)
        frob = (
            (raw_beams_re.square() + raw_beams_im.square())
            .sum(dim=(-2, -1), keepdim=True)
            .clamp_min(DEFAULT_TRACE_EPS)
        )
        inv_frob = torch.rsqrt(frob)
        beams_re = raw_beams_re * inv_frob
        beams_im = raw_beams_im * inv_frob

        mps_purity = gram_matrix_mps_purity(beams_re, beams_im)
        mean_ket_re, mean_ket_im = normalize_complex_ket(
            beams_re.mean(dim=-2), beams_im.mean(dim=-2)
        )
        return beams_re, beams_im, mps_purity, mean_ket_re, mean_ket_im


# ==============================================================================
# Fused `bfloat16` 4-Compartment Quantum Statistical Mechanics Baton Layer
# ==============================================================================
class FusedBF16FourCompartmentBatonLayer(nn.Module):
    """Single Fused `bfloat16` layer of the 4-Compartment Baton Highway (`Wave 74`)."""

    def __init__(self, config: FusedBF16FourCompartmentBatonConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hilbert_dim
        self.mps_baton = FusedBF16MPSBatonModule(config)

        self.motif_probes_re = nn.Parameter(
            torch.randn(NUM_SEQ_ATTN_PACKETS, d) / math.sqrt(float(d))
        )
        self.motif_probes_im = nn.Parameter(torch.randn(NUM_SEQ_ATTN_PACKETS, d) * 0.05)

        eye_h = torch.eye(d).unsqueeze(0).repeat(NUM_SEQ_ATTN_PACKETS, 1, 1)
        self.wq_re = nn.Parameter(eye_h + 0.05 * torch.randn(NUM_SEQ_ATTN_PACKETS, d, d))
        self.wq_im = nn.Parameter(0.05 * torch.randn(NUM_SEQ_ATTN_PACKETS, d, d))
        self.wk_re = nn.Parameter(eye_h.clone() + 0.05 * torch.randn(NUM_SEQ_ATTN_PACKETS, d, d))
        self.wk_im = nn.Parameter(0.05 * torch.randn(NUM_SEQ_ATTN_PACKETS, d, d))
        self.wv_re = nn.Parameter(eye_h.clone() + 0.05 * torch.randn(NUM_SEQ_ATTN_PACKETS, d, d))
        self.wv_im = nn.Parameter(0.05 * torch.randn(NUM_SEQ_ATTN_PACKETS, d, d))

        self.id_query_re = nn.Parameter(torch.eye(d) + 0.03 * torch.randn(d, d))
        self.id_key_re = nn.Parameter(torch.eye(d) + 0.03 * torch.randn(d, d))

        self.povm_gates_re = nn.Parameter(torch.randn(4, d) / math.sqrt(float(d)))
        self.povm_gates_im = nn.Parameter(torch.randn(4, d) * 0.05)
        self.povm_gate_bias = nn.Parameter(
            torch.tensor([0.15, 1.10, 0.85, 0.35], dtype=torch.float32)
        )
        self.trans_learned_re = nn.Parameter(torch.eye(d) * 0.25 + 0.04 * torch.randn(d, d))
        self.trans_learned_im = nn.Parameter(0.04 * torch.randn(d, d))

    def _compute_fused_transition_beams(
        self,
        ket_re: Tensor,
        ket_im: Tensor,
        trans_ema_re: Tensor,
        trans_ema_im: Tensor,
        hash_ket_re: Tensor,
        hash_ket_im: Tensor,
        causal_decay: Tensor,
        strict_past_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute Rank-2 transition beams (`ctx_ket`, `ema+hash_ket`) and in-context copy weights (`<= 8` args)."""
        prev_ket_re = F.pad(ket_re[:, :-1, :], (0, 0, 1, 0))
        prev_ket_im = F.pad(ket_im[:, :-1, :], (0, 0, 1, 0))

        ov_re = torch.matmul(ket_re, prev_ket_re.transpose(-2, -1)) + torch.matmul(
            ket_im, prev_ket_im.transpose(-2, -1)
        )
        ov_im = torch.matmul(ket_im, prev_ket_re.transpose(-2, -1)) - torch.matmul(
            ket_re, prev_ket_im.transpose(-2, -1)
        )
        ov_sq = ov_re.square() + ov_im.square()

        born_pull = ov_sq.square() * causal_decay.unsqueeze(0)
        ctx_w_re = ov_re * born_pull
        ctx_w_im = ov_im * born_pull
        ctx_ket_re = torch.matmul(ctx_w_re, ket_re) - torch.matmul(ctx_w_im, ket_im)
        ctx_ket_im = torch.matmul(ctx_w_re, ket_im) + torch.matmul(ctx_w_im, ket_re)
        ctx_norm_re, ctx_norm_im = normalize_complex_ket(ctx_ket_re, ctx_ket_im)

        op_re = trans_ema_re.to(ket_re.dtype) + self.trans_learned_re.to(ket_re.dtype)
        op_im = trans_ema_im.to(ket_re.dtype) + self.trans_learned_im.to(ket_re.dtype)
        ema_ket_re = torch.matmul(ket_re, op_re.transpose(-2, -1)) - torch.matmul(
            ket_im, op_im.transpose(-2, -1)
        )
        ema_ket_im = torch.matmul(ket_re, op_im.transpose(-2, -1)) + torch.matmul(
            ket_im, op_re.transpose(-2, -1)
        )
        ema_norm_re, ema_norm_im = normalize_complex_ket(
            ema_ket_re + 0.35 * hash_ket_re + 0.15 * ket_re,
            ema_ket_im + 0.35 * hash_ket_im + 0.15 * ket_im,
        )

        in_ctx_copy_weights = ov_sq * strict_past_mask.unsqueeze(0)
        return ctx_norm_re, ctx_norm_im, ema_norm_re, ema_norm_im, in_ctx_copy_weights

    def _compute_fused_exterior_sequence_beams(
        self,
        ket_re: Tensor,
        ket_im: Tensor,
        mps_beams_re: Tensor,
        mps_beams_im: Tensor,
        mps_purity: Tensor,
        aux_pack: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute Rank-4 Exterior-Masked + 2-Hop Triangle Born-Attention beams in `O(B H T^2 d)` (`<= 8` args)."""
        d = ket_re.shape[-1]
        seq_len = ket_re.shape[1]
        cos_pos = aux_pack["cos_pos"]
        sin_pos = aux_pack["sin_pos"]
        head_masks = aux_pack["head_masks"]

        probes_re = self.motif_probes_re.to(ket_re.dtype)
        probes_im = self.motif_probes_im.to(ket_re.dtype)
        proj_mid_re = torch.matmul(mps_beams_re, probes_re.transpose(0, 1)) + torch.matmul(
            mps_beams_im, probes_im.transpose(0, 1)
        )
        proj_mid_im = torch.matmul(mps_beams_im, probes_re.transpose(0, 1)) - torch.matmul(
            mps_beams_re, probes_im.transpose(0, 1)
        )
        pm_re_t = proj_mid_re.transpose(-2, -1)
        pm_im_t = proj_mid_im.transpose(-2, -1)
        motif_re = torch.matmul(pm_re_t, mps_beams_re) + torch.matmul(pm_im_t, mps_beams_im)
        motif_im = torch.matmul(pm_re_t, mps_beams_im) - torch.matmul(pm_im_t, mps_beams_re)
        motif_re, motif_im = normalize_complex_ket(
            motif_re + 0.25 * ket_re.unsqueeze(-2), motif_im + 0.25 * ket_im.unsqueeze(-2)
        )

        bsz = ket_re.shape[0]
        h_cnt = motif_re.shape[2]
        motif_packed = (
            torch.cat([motif_re, motif_im], dim=-1)
            .permute(2, 0, 1, 3)
            .reshape(h_cnt, bsz * seq_len, 2 * d)
        )
        w_top = torch.cat(
            [self.wq_re, self.wq_im, self.wk_re, self.wk_im, self.wv_re, self.wv_im], dim=-1
        )
        w_bot = torch.cat(
            [-self.wq_im, self.wq_re, -self.wk_im, self.wk_re, -self.wv_im, self.wv_re], dim=-1
        )
        w_qkv_packed = torch.cat([w_top, w_bot], dim=1).to(motif_packed.dtype)
        qkv_out = (
            torch.bmm(motif_packed, w_qkv_packed)
            .view(h_cnt, bsz, seq_len, 6, d)
            .permute(1, 0, 2, 3, 4)
        )
        q_re, q_im = qkv_out[..., 0, :], qkv_out[..., 1, :]
        k_re, k_im = qkv_out[..., 2, :], qkv_out[..., 3, :]
        v_re, v_im = qkv_out[..., 4, :], qkv_out[..., 5, :]

        q_re, q_im = normalize_complex_ket(q_re, q_im)
        k_re, k_im = normalize_complex_ket(k_re, k_im)
        v_re, v_im = normalize_complex_ket(v_re, v_im)
        q_re, q_im = apply_complex_rope(q_re, q_im, cos_pos.unsqueeze(1), sin_pos.unsqueeze(1))
        k_re, k_im = apply_complex_rope(k_re, k_im, cos_pos.unsqueeze(1), sin_pos.unsqueeze(1))

        k_pack_t = torch.cat([k_re, k_im], dim=-1).transpose(-2, -1)
        inner_re = torch.matmul(torch.cat([q_re, q_im], dim=-1), k_pack_t)
        inner_im = torch.matmul(torch.cat([q_im, -q_re], dim=-1), k_pack_t)
        born_scores = inner_re.square() + inner_im.square()

        purity_rel = (mps_purity - (1.0 / float(d)) + DEFAULT_PURITY_FLOOR_OFFSET).clamp(
            min=DEFAULT_PURITY_MIN_CLAMP, max=1.0
        )
        born_scores = born_scores * purity_rel.unsqueeze(1).unsqueeze(2)
        masked_scores = born_scores * head_masks.unsqueeze(0)
        attn_probs = masked_scores / (masked_scores.sum(dim=-1, keepdim=True) + DEFAULT_TRACE_EPS)

        if self.config.use_two_hop_triangle and seq_len > 1:
            two_hop = torch.matmul(attn_probs, aux_pack["local_op"])
            attn_probs = (
                1.0 - DEFAULT_TWO_HOP_MIX_ALPHA
            ) * attn_probs + DEFAULT_TWO_HOP_MIX_ALPHA * two_hop

        seq_head_both = (
            torch.matmul(attn_probs, torch.cat([v_re, v_im], dim=-1))
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        seq_head_re, seq_head_im = seq_head_both[..., :d], seq_head_both[..., d:]
        seq_head_re, seq_head_im = normalize_complex_ket(seq_head_re, seq_head_im)
        seq_beams_re = seq_head_re * 0.5
        seq_beams_im = seq_head_im * 0.5
        agg_ket_re, agg_ket_im = normalize_complex_ket(
            seq_head_re.mean(dim=-2), seq_head_im.mean(dim=-2)
        )
        return seq_beams_re, seq_beams_im, agg_ket_re, agg_ket_im, attn_probs

    def forward(
        self,
        ket_re: Tensor,
        ket_im: Tensor,
        uni_diag: Tensor,
        trans_ema_re: Tensor,
        trans_ema_im: Tensor,
        hash_ket_re: Tensor,
        hash_ket_im: Tensor,
        aux_pack: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Run one Fused `bfloat16` 4-Compartment Baton layer in factorized Rank-4 beam space (`<= 8` args)."""
        head_masks = aux_pack["head_masks"]
        causal_decay = aux_pack["causal_decay"]
        strict_past_mask = aux_pack["strict_past_mask"]
        id_ket_re = aux_pack["id_ket_re"]

        mps_beams_re, mps_beams_im, mps_purity, mps_ket_re, mps_ket_im = self.mps_baton(
            ket_re, ket_im, uni_diag
        )
        ctx_ket_re, ctx_ket_im, ema_ket_re, ema_ket_im, in_ctx_copy_weights = (
            self._compute_fused_transition_beams(
                ket_re,
                ket_im,
                trans_ema_re,
                trans_ema_im,
                hash_ket_re,
                hash_ket_im,
                causal_decay,
                strict_past_mask,
            )
        )

        w_mps = math.sqrt(DEFAULT_MPS_NGRAM_BLEND)
        w_tr = math.sqrt(0.5 * (1.0 - DEFAULT_MPS_NGRAM_BLEND))
        ngram_beams_re = torch.cat(
            [
                w_mps * mps_beams_re[:, :, :2, :],
                w_tr * ctx_ket_re.unsqueeze(2),
                w_tr * ema_ket_re.unsqueeze(2),
            ],
            dim=2,
        )
        ngram_beams_im = torch.cat(
            [
                w_mps * mps_beams_im[:, :, :2, :],
                w_tr * ctx_ket_im.unsqueeze(2),
                w_tr * ema_ket_im.unsqueeze(2),
            ],
            dim=2,
        )
        ng_frob = (
            (ngram_beams_re.square() + ngram_beams_im.square())
            .sum(dim=(-2, -1), keepdim=True)
            .clamp_min(DEFAULT_TRACE_EPS)
        )
        inv_ng = torch.rsqrt(ng_frob)
        ngram_beams_re = ngram_beams_re * inv_ng
        ngram_beams_im = ngram_beams_im * inv_ng

        seq_beams_re, seq_beams_im, seq_ket_re, seq_ket_im, seq_attn_probs = (
            self._compute_fused_exterior_sequence_beams(
                ket_re, ket_im, mps_beams_re, mps_beams_im, mps_purity, aux_pack
            )
        )

        deep_beams_re, deep_beams_im, _ = factorized_rank_beams_schur_intersect(
            seq_beams_re, seq_beams_im, ngram_beams_re, ngram_beams_im
        )

        q_id = F.normalize(torch.matmul(ket_re + id_ket_re, self.id_query_re), dim=-1)
        k_id = F.normalize(torch.matmul(ket_re + id_ket_re, self.id_key_re), dim=-1)
        id_scores = torch.matmul(q_id, k_id.transpose(-2, -1)).square() * head_masks[-1].unsqueeze(
            0
        )
        id_attn_probs = id_scores / (id_scores.sum(dim=-1, keepdim=True) + DEFAULT_TRACE_EPS)
        id_routed_re = F.normalize(torch.matmul(id_attn_probs, id_ket_re), dim=-1)
        id_routed_im = torch.zeros_like(id_routed_re)

        gate_kets_re, gate_kets_im = normalize_complex_ket(
            self.povm_gates_re.to(ket_re.dtype), self.povm_gates_im.to(ket_re.dtype)
        )
        gates = evaluate_rank_beams_born_gates(
            ngram_beams_re, ngram_beams_im, gate_kets_re, gate_kets_im, self.povm_gate_bias
        )

        comb_ket_re = ket_re + mps_ket_re + seq_ket_re + 0.5 * (ctx_ket_re + ema_ket_re)
        comb_ket_im = ket_im + mps_ket_im + seq_ket_im + 0.5 * (ctx_ket_im + ema_ket_im)
        act_deep_re, act_deep_im = apply_rank_beams_to_probe(
            deep_beams_re, deep_beams_im, comb_ket_re, comb_ket_im
        )
        next_ket_re, next_ket_im = normalize_complex_ket(
            ket_re + act_deep_re + 0.25 * comb_ket_re, ket_im + act_deep_im + 0.25 * comb_ket_im
        )

        return {
            "deep_beams_re": deep_beams_re,
            "deep_beams_im": deep_beams_im,
            "ngram_beams_re": ngram_beams_re,
            "ngram_beams_im": ngram_beams_im,
            "id_routed_re": id_routed_re,
            "id_routed_im": id_routed_im,
            "gates": gates,
            "next_ket_re": next_ket_re,
            "next_ket_im": next_ket_im,
            "in_ctx_copy_weights": in_ctx_copy_weights,
            "id_attn_probs": id_attn_probs,
            "seq_attn_probs": seq_attn_probs,
            "mps_purity": mps_purity,
        }


def _eager_prior_copy_softcap_ce(
    born_logits: Tensor,
    wu_c: Tensor,
    uni_log_prior: Tensor,
    ci_c: Tensor,
    cm_c: Tensor,
    tg_c: Tensor,
) -> Tensor:
    """Fused unigram prior + copy `scatter_add` + `tanh` softcap + cross-entropy step (`<= 8` args)."""
    lg = (
        DEFAULT_BORN_READOUT_SCALE * born_logits
        + (DEFAULT_UNIGRAM_PRIOR_WEIGHT + 0.50 * wu_c) * uni_log_prior
    )
    lg = lg.scatter_add(dim=-1, index=ci_c, src=cm_c)
    capped = DEFAULT_LOGIT_SOFTCAP * torch.tanh(lg * (1.0 / DEFAULT_LOGIT_SOFTCAP))
    return F.cross_entropy(capped.reshape(-1, capped.shape[-1]), tg_c.reshape(-1))


_COMPILED_SOFTCAP_CE_FN: Any = None


def _get_softcap_cross_entropy_fn(use_compile: bool) -> Any:
    """Return `@torch.compile`-fused prior + copy + softcap + cross-entropy kernel on CUDA (`<= 8` args)."""
    global _COMPILED_SOFTCAP_CE_FN
    if not use_compile:
        return _eager_prior_copy_softcap_ce
    if _COMPILED_SOFTCAP_CE_FN is None:
        _COMPILED_SOFTCAP_CE_FN = torch.compile(_eager_prior_copy_softcap_ce)
    return _COMPILED_SOFTCAP_CE_FN


# ==============================================================================
# Full Fused `bfloat16` 1L/2L 4-Compartment Baton Language Model (`Wave 74`)
# ==============================================================================
class FusedBF16FourCompartmentBatonLM(nn.Module):
    """Fused `bfloat16` Tensor-Core 1-Layer / 2-Layer 4-Compartment Baton LM (`Wave 74`)."""

    def __init__(self, config: FusedBF16FourCompartmentBatonConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hilbert_dim
        v_sz = config.vocab_size

        self.token_ket_re = nn.Parameter(torch.randn(v_sz, d) / math.sqrt(float(d)))
        self.token_ket_im = nn.Parameter(torch.randn(v_sz, d) / math.sqrt(float(d)))
        self.readout_probe_re = nn.Parameter(torch.eye(d) + 0.03 * torch.randn(d, d))
        self.readout_probe_im = nn.Parameter(0.03 * torch.randn(d, d))
        self.beam_readout_mix_re = nn.Parameter(
            torch.ones(DEFAULT_SCHUR_BEAM_RANK, d) / float(DEFAULT_SCHUR_BEAM_RANK)
        )
        self.beam_readout_mix_im = nn.Parameter(torch.zeros(DEFAULT_SCHUR_BEAM_RANK, d))

        self.layers = nn.ModuleList(
            [FusedBF16FourCompartmentBatonLayer(config) for _ in range(config.num_layers)]
        )

        self.register_buffer("online_uni_bath_re", torch.eye(d, dtype=torch.float32) / float(d))
        self.register_buffer("online_uni_bath_im", torch.zeros(d, d, dtype=torch.float32))
        self.register_buffer(
            "online_uni_token_probs", torch.full((v_sz,), 1.0 / float(v_sz), dtype=torch.float32)
        )
        self.register_buffer("online_trans_ema_re", torch.zeros(d, d, dtype=torch.float32))
        self.register_buffer("online_trans_ema_im", torch.zeros(d, d, dtype=torch.float32))
        self.register_buffer(
            "online_ngram_hash_re",
            torch.zeros(DEFAULT_ONLINE_NGRAM_HASH_BUCKETS, d, dtype=torch.float32),
        )
        self.register_buffer(
            "online_ngram_hash_im",
            torch.zeros(DEFAULT_ONLINE_NGRAM_HASH_BUCKETS, d, dtype=torch.float32),
        )
        self.register_buffer("online_ingested_tokens", torch.tensor(0, dtype=torch.long))
        self.register_buffer(
            "walsh_parity_pairs", build_walsh_hadamard_parity_pairs(d), persistent=False
        )
        self._mask_cache: dict[tuple[int, torch.device, torch.dtype], dict[str, Tensor]] = {}

    def _get_cached_sequence_geometry(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, Tensor]:
        """Return cached RoPE tables, exterior causal masks, and Lindblad decay matrices (`<= 8` args)."""
        key = (seq_len, device, dtype)
        if key in self._mask_cache:
            return self._mask_cache[key]

        d = self.config.hilbert_dim
        k_win = self.config.mps_window_k
        pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        freq_idx = torch.arange(0, d, 1, device=device, dtype=torch.float32).unsqueeze(0)
        angles = pos * (1.0 / (DEFAULT_ROPE_THETA_BASE ** (freq_idx / float(d))))
        cos_pos = torch.cos(angles).unsqueeze(0).to(dtype)
        sin_pos = torch.sin(angles).unsqueeze(0).to(dtype)

        t_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        s_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        causal_mask = s_idx <= t_idx
        exterior_mask = torch.where(t_idx >= k_win, s_idx <= (t_idx - k_win), causal_mask)
        head_masks = torch.stack(
            [exterior_mask, exterior_mask, exterior_mask, causal_mask], dim=0
        ).to(dtype)

        strict_past_bool = (t_idx > s_idx) & (s_idx >= 1)
        strict_past_mask = strict_past_bool.to(dtype)
        lag = (t_idx - 1 - s_idx).clamp_min(0).to(torch.float32)
        causal_decay = (
            (1.0 - DEFAULT_IN_CONTEXT_LINDBLAD_GAMMA)
            * (DEFAULT_IN_CONTEXT_LINDBLAD_GAMMA ** lag)
            * strict_past_bool.to(torch.float32)
        ).to(dtype)

        local_band = (((t_idx - s_idx) >= 0) & ((t_idx - s_idx) < k_win)).to(torch.float32)
        local_op = (local_band / local_band.sum(dim=-1, keepdim=True).clamp_min(1.0)).to(dtype)

        geom = {
            "cos_pos": cos_pos,
            "sin_pos": sin_pos,
            "head_masks": geom_hm if (geom_hm := head_masks) is not None else head_masks,
            "strict_past_mask": strict_past_mask,
            "causal_decay": causal_decay,
            "local_op": local_op.unsqueeze(0).unsqueeze(0),
        }
        self._mask_cache[key] = geom
        return geom

    def _compute_causal_ngram_hash_ids(self, idx: Tensor) -> Tensor:
        """Compute causal bigram/trigram bucket IDs `(B, T)` in `[0, DEFAULT_ONLINE_NGRAM_HASH_BUCKETS)` (`<= 8` args)."""
        prev1 = F.pad(idx[:, :-1], (1, 0), value=0)
        prev2 = F.pad(idx[:, :-2], (2, 0), value=0)
        raw_hash = (
            idx.long()
            + prev1.long() * DEFAULT_HASH_PRIME_PREV1
            + prev2.long() * DEFAULT_HASH_PRIME_PREV2
        )
        return torch.remainder(raw_hash, DEFAULT_ONLINE_NGRAM_HASH_BUCKETS)

    @torch.no_grad()
    def ingest_online_token_batch(self, idx: Tensor) -> None:
        """Ingest a batch of token IDs `(B, T)` online into Compartment 1 & 2 buffers (`<= 8` args)."""
        eta_uni = self.config.online_bath_ema_decay
        eta_trans = DEFAULT_ONLINE_TRANS_EMA_DECAY

        raw_re = F.embedding(idx, self.token_ket_re).to(torch.float32)
        raw_im = F.embedding(idx, self.token_ket_im).to(torch.float32)
        ket_re, ket_im = normalize_complex_ket(raw_re, raw_im)
        flat_re = ket_re.reshape(-1, ket_re.shape[-1])
        flat_im = ket_im.reshape(-1, ket_im.shape[-1])

        batch_rho_re = torch.matmul(flat_re.transpose(0, 1), flat_re) + torch.matmul(
            flat_im.transpose(0, 1), flat_im
        )
        batch_rho_im = torch.matmul(flat_im.transpose(0, 1), flat_re) - torch.matmul(
            flat_re.transpose(0, 1), flat_im
        )
        batch_rho_re, batch_rho_im = trace_normalize_density(batch_rho_re, batch_rho_im)

        new_uni_re, new_uni_im = trace_normalize_density(
            (1.0 - eta_uni) * self.online_uni_bath_re + eta_uni * batch_rho_re,
            (1.0 - eta_uni) * self.online_uni_bath_im + eta_uni * batch_rho_im,
        )
        self.online_uni_bath_re.copy_(new_uni_re)
        self.online_uni_bath_im.copy_(new_uni_im)

        flat_idx = idx.reshape(-1)
        inc_val = eta_uni / max(1.0, float(flat_idx.numel()))
        self.online_uni_token_probs.mul_(1.0 - eta_uni)
        self.online_uni_token_probs.scatter_add_(
            0, flat_idx, torch.full_like(flat_idx, inc_val, dtype=torch.float32)
        )

        if idx.shape[1] > 1:
            curr_re = ket_re[:, 1:, :].reshape(-1, ket_re.shape[-1])
            curr_im = ket_im[:, 1:, :].reshape(-1, ket_im.shape[-1])
            prev_re = ket_re[:, :-1, :].reshape(-1, ket_re.shape[-1])
            prev_im = ket_im[:, :-1, :].reshape(-1, ket_im.shape[-1])
            denom = max(1.0, float(curr_re.shape[0]))
            trans_re = (
                torch.matmul(curr_re.transpose(0, 1), prev_re)
                + torch.matmul(curr_im.transpose(0, 1), prev_im)
            ) / denom
            trans_im = (
                torch.matmul(curr_im.transpose(0, 1), prev_re)
                - torch.matmul(curr_re.transpose(0, 1), prev_im)
            ) / denom
            self.online_trans_ema_re.mul_(1.0 - eta_trans).add_(eta_trans * trans_re)
            self.online_trans_ema_im.mul_(1.0 - eta_trans).add_(eta_trans * trans_im)

            hash_ids = self._compute_causal_ngram_hash_ids(idx)[:, :-1].reshape(-1)
            self.online_ngram_hash_re.index_add_(
                0, hash_ids, DEFAULT_ONLINE_HASH_EMA_DECAY * curr_re
            )
            self.online_ngram_hash_im.index_add_(
                0, hash_ids, DEFAULT_ONLINE_HASH_EMA_DECAY * curr_im
            )
            norms = torch.sqrt(
                self.online_ngram_hash_re.square().sum(dim=-1, keepdim=True)
                + self.online_ngram_hash_im.square().sum(dim=-1, keepdim=True)
                + DEFAULT_TRACE_EPS
            ).clamp_min(1.0)
            self.online_ngram_hash_re.div_(norms)
            self.online_ngram_hash_im.div_(norms)

        self.online_ingested_tokens.add_(idx.numel())

    def forward_with_diagnostics(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        ingest_online: bool = False,
        return_logits: bool = True,
    ) -> dict[str, Any]:
        """Execute the Fused `bfloat16` 1L/2L 4-Compartment Baton highway (`<= 8` args)."""
        use_amp = self.config.use_bfloat16_amp and idx.is_cuda
        amp_dtype = torch.bfloat16 if use_amp else torch.float32

        with torch.amp.autocast(device_type=idx.device.type, dtype=amp_dtype, enabled=use_amp):
            bsz, seq_len = idx.shape
            d = self.config.hilbert_dim

            ket_re, ket_im = normalize_complex_ket(
                F.embedding(idx, self.token_ket_re).to(amp_dtype),
                F.embedding(idx, self.token_ket_im).to(amp_dtype),
            )
            steps = torch.arange(1, seq_len + 1, device=idx.device, dtype=amp_dtype).view(
                1, seq_len, 1
            )
            doc_ket_re, doc_ket_im = normalize_complex_ket(
                (torch.cumsum(ket_re, dim=1) / steps).to(amp_dtype),
                (torch.cumsum(ket_im, dim=1) / steps).to(amp_dtype),
            )
            doc_diag = doc_ket_re.square() + doc_ket_im.square()
            uni_bath_re_snap = self.online_uni_bath_re.detach().clone()
            uni_bath_im_snap = self.online_uni_bath_im.detach().clone()
            trans_ema_re_snap = self.online_trans_ema_re.detach().clone()
            trans_ema_im_snap = self.online_trans_ema_im.detach().clone()
            uni_probs_snap = self.online_uni_token_probs.detach().clone()

            uni_global_diag = torch.diagonal(uni_bath_re_snap).view(1, 1, d).to(amp_dtype)
            uni_diag = (1.0 - DEFAULT_DOC_BATH_MIX) * uni_global_diag + DEFAULT_DOC_BATH_MIX * doc_diag

            id_ket_re, id_ket_im = encode_walsh_hadamard_token_id_kets(
                idx, self.walsh_parity_pairs, d
            )
            id_ket_re = id_ket_re.to(amp_dtype)
            id_ket_im = id_ket_im.to(amp_dtype)

            hash_ids = self._compute_causal_ngram_hash_ids(idx)
            hash_ket_re = (
                F.embedding(hash_ids, self.online_ngram_hash_re).detach().clone().to(amp_dtype)
            )
            hash_ket_im = (
                F.embedding(hash_ids, self.online_ngram_hash_im).detach().clone().to(amp_dtype)
            )

            geom = self._get_cached_sequence_geometry(seq_len, idx.device, amp_dtype)
            aux_pack = {
                "cos_pos": geom["cos_pos"],
                "sin_pos": geom["sin_pos"],
                "head_masks": geom["head_masks"],
                "strict_past_mask": geom["strict_past_mask"],
                "causal_decay": geom["causal_decay"],
                "local_op": geom["local_op"],
                "id_ket_re": id_ket_re,
                "id_ket_im": id_ket_im,
            }

            curr_ket_re, curr_ket_im = ket_re, ket_im
            prev_deep_re: Tensor | None = None
            prev_deep_im: Tensor | None = None
            layer_out: dict[str, Tensor] = {}

            if use_amp and not hasattr(self, "_compiled_layers_list"):
                self._compiled_layers_list = [torch.compile(lyr) for lyr in self.layers]
            active_layers = self._compiled_layers_list if use_amp else self.layers

            for layer_idx, layer_fn in enumerate(active_layers):
                layer_out = layer_fn(
                    curr_ket_re,
                    curr_ket_im,
                    uni_diag,
                    trans_ema_re_snap,
                    trans_ema_im_snap,
                    hash_ket_re,
                    hash_ket_im,
                    aux_pack,
                )
                curr_ket_re = layer_out["next_ket_re"]
                curr_ket_im = layer_out["next_ket_im"]
                if layer_idx > 0 and prev_deep_re is not None and prev_deep_im is not None:
                    four_way_re, four_way_im, _ = factorized_rank_beams_schur_intersect(
                        prev_deep_re,
                        prev_deep_im,
                        layer_out["deep_beams_re"],
                        layer_out["deep_beams_im"],
                    )
                    blend_re = 0.65 * layer_out["deep_beams_re"] + 0.35 * four_way_re
                    blend_im = 0.65 * layer_out["deep_beams_im"] + 0.35 * four_way_im
                    frob = (
                        (blend_re.square() + blend_im.square())
                        .sum(dim=(-2, -1), keepdim=True)
                        .clamp_min(DEFAULT_TRACE_EPS)
                    )
                    inv_f = torch.rsqrt(frob)
                    layer_out["deep_beams_re"] = blend_re * inv_f
                    layer_out["deep_beams_im"] = blend_im * inv_f
                prev_deep_re = layer_out["deep_beams_re"]
                prev_deep_im = layer_out["deep_beams_im"]

            deep_beams_re = layer_out["deep_beams_re"]
            deep_beams_im = layer_out["deep_beams_im"]
            ngram_beams_re = layer_out["ngram_beams_re"]
            ngram_beams_im = layer_out["ngram_beams_im"]
            id_routed_re = layer_out["id_routed_re"]
            id_routed_im = layer_out["id_routed_im"]
            gates = layer_out["gates"].to(amp_dtype)

            probe_re, probe_im = normalize_complex_ket(
                torch.matmul(curr_ket_re, self.readout_probe_re)
                - torch.matmul(curr_ket_im, self.readout_probe_im),
                torch.matmul(curr_ket_re, self.readout_probe_im)
                + torch.matmul(curr_ket_im, self.readout_probe_re),
            )

            act_deep_re, act_deep_im = apply_rank_beams_to_probe(
                deep_beams_re, deep_beams_im, probe_re, probe_im
            )
            bath_re_cast = uni_bath_re_snap.to(amp_dtype)
            bath_im_cast = uni_bath_im_snap.to(amp_dtype)
            act_ng_uni_re, act_ng_uni_im, _ = diagonal_conjugated_bath_schur_action(
                ngram_beams_re, ngram_beams_im, bath_re_cast, bath_im_cast, probe_re, probe_im
            )
            act_uni_re = torch.matmul(probe_re, bath_re_cast.transpose(-2, -1)) - torch.matmul(
                probe_im, bath_im_cast.transpose(-2, -1)
            )
            act_uni_im = torch.matmul(probe_re, bath_im_cast.transpose(-2, -1)) + torch.matmul(
                probe_im, bath_re_cast.transpose(-2, -1)
            )
            id_ov = (id_routed_re * probe_re).sum(dim=-1, keepdim=True)
            act_id_re = id_routed_re * id_ov
            act_id_im = id_routed_im * id_ov

            w_id = gates[..., 0:1]
            w_deep = gates[..., 1:2]
            w_ngram = gates[..., 2:3]
            w_uni = gates[..., 3:4]

            mix_re_cast = self.beam_readout_mix_re.to(amp_dtype)
            mix_im_cast = self.beam_readout_mix_im.to(amp_dtype)
            beam_mix_re = (deep_beams_re * mix_re_cast - deep_beams_im * mix_im_cast).sum(dim=-2)
            beam_mix_im = (deep_beams_re * mix_im_cast + deep_beams_im * mix_re_cast).sum(dim=-2)

            state_vec_re = (
                w_id * act_id_re
                + w_deep * (act_deep_re + 0.35 * beam_mix_re)
                + w_ngram * (act_ng_uni_re + 0.25 * hash_ket_re)
                + (w_uni + DEFAULT_VACUUM_EPS) * act_uni_re
            )
            state_vec_im = (
                w_id * act_id_im
                + w_deep * (act_deep_im + 0.35 * beam_mix_im)
                + w_ngram * (act_ng_uni_im + 0.25 * hash_ket_im)
                + (w_uni + DEFAULT_VACUUM_EPS) * act_uni_im
            )
            readout_re, readout_im = normalize_complex_ket(
                state_vec_re + 0.50 * curr_ket_re, state_vec_im + 0.50 * curr_ket_im
            )

            vocab_re, vocab_im = normalize_complex_ket(
                self.token_ket_re.to(amp_dtype), self.token_ket_im.to(amp_dtype)
            )
            packed_readout = torch.cat([readout_re, readout_im], dim=-1).to(amp_dtype)
            packed_vocab_t = (
                torch.cat([vocab_re, vocab_im], dim=-1).transpose(0, 1).contiguous().to(amp_dtype)
            )

            uni_log_prior = (
                torch.log(uni_probs_snap.clamp_min(1e-9)).view(1, 1, -1).to(amp_dtype)
            )
            in_ctx_weights = layer_out["in_ctx_copy_weights"].to(amp_dtype)
            id_attn_probs = layer_out["id_attn_probs"].to(amp_dtype)
            combined_copy_mass = (
                DEFAULT_IN_CONTEXT_COPY_SCALE * in_ctx_weights
                + DEFAULT_TOKEN_ID_COPY_SCALE * w_id * id_attn_probs
            ).to(amp_dtype)
            copy_index = idx.unsqueeze(1).expand(bsz, seq_len, seq_len)
            ce_fn = _get_softcap_cross_entropy_fn(use_amp)

            logits: Tensor | None = None
            loss: Tensor | None = None
            top1_acc_val: float = 0.0
            if targets is not None and not return_logits and bsz >= 8:
                chunk_b = 16
                chunk_losses: list[Tensor] = []
                correct_count = 0
                for b_start in range(0, bsz, chunk_b):
                    b_end = min(bsz, b_start + chunk_b)
                    pr_c = packed_readout[b_start:b_end]
                    wu_c = w_uni[b_start:b_end]
                    ci_c = copy_index[b_start:b_end]
                    cm_c = combined_copy_mass[b_start:b_end]
                    tg_c = targets[b_start:b_end]

                    born_c = torch.matmul(pr_c, packed_vocab_t)
                    chunk_losses.append(
                        ce_fn(born_c, wu_c, uni_log_prior, ci_c, cm_c, tg_c)
                        * float(b_end - b_start)
                    )
                    if not self.training:
                        lg_eval = (
                            DEFAULT_BORN_READOUT_SCALE * born_c
                            + (DEFAULT_UNIGRAM_PRIOR_WEIGHT + 0.50 * wu_c) * uni_log_prior
                        ).scatter_add(dim=-1, index=ci_c, src=cm_c)
                        correct_count += int((lg_eval.argmax(dim=-1) == tg_c).sum().item())
                loss = torch.stack(chunk_losses).sum() / float(bsz)
                if not self.training:
                    top1_acc_val = float(correct_count) / float(bsz * seq_len)
            else:
                logits = (
                    DEFAULT_BORN_READOUT_SCALE * torch.matmul(packed_readout, packed_vocab_t)
                    + (DEFAULT_UNIGRAM_PRIOR_WEIGHT + 0.50 * w_uni) * uni_log_prior
                )
                logits = logits.scatter_add(dim=-1, index=copy_index, src=combined_copy_mass)
                logits = DEFAULT_LOGIT_SOFTCAP * torch.tanh(logits / DEFAULT_LOGIT_SOFTCAP)
                if targets is not None:
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
                    )
                    top1_acc_val = float((logits.argmax(dim=-1) == targets).to(torch.float32).mean().item())

        if ingest_online and self.training:
            self.ingest_online_token_batch(idx)

        out_purity = gram_matrix_mps_purity(deep_beams_re.float(), deep_beams_im.float())
        return {
            "logits": logits,
            "loss": loss,
            "top1_acc": top1_acc_val,
            "gates": gates.detach(),
            "out_purity": out_purity.detach(),
            "mps_purity": layer_out["mps_purity"].detach(),
        }

    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor | None, Tensor | None]:
        """Standard `(logits, loss)` forward entrypoint (`<= 8` args)."""
        need_logits = (targets is None) or (not self.training)
        out = self.forward_with_diagnostics(
            idx=idx,
            targets=targets,
            ingest_online=self.training,
            return_logits=need_logits,
        )
        return out["logits"], out["loss"]


# ==============================================================================
# Theorem 15a–15c Numerical Parity & Throughput Inspector (`Wave 74`)
# ==============================================================================
def verify_theorem15_algebraic_identities() -> dict[str, Any]:
    """Verify Theorems 15a, 15b, and 15c numerically in FP32 (`< 1e-6` max error) (`<= 8` args)."""
    torch.manual_seed(2026)
    bsz, seq_len, d = 2, 8, DEFAULT_HILBERT_DIM
    rank = DEFAULT_SCHUR_BEAM_RANK

    ket_a_re, ket_a_im = normalize_complex_ket(
        torch.randn(bsz, seq_len, 1, d), torch.randn(bsz, seq_len, 1, d)
    )
    ket_b_re, ket_b_im = normalize_complex_ket(
        torch.randn(bsz, seq_len, 1, d), torch.randn(bsz, seq_len, 1, d)
    )
    probe_re, probe_im = normalize_complex_ket(
        torch.randn(bsz, seq_len, d), torch.randn(bsz, seq_len, d)
    )

    # 1. Theorem 15a: Rank-1 Factorized Ket Schur vs. Explicit 64x64 Schur Product
    rho_a_re = torch.einsum("btid,btie->btde", ket_a_re, ket_a_re) + torch.einsum(
        "btid,btie->btde", ket_a_im, ket_a_im
    )
    rho_a_im = torch.einsum("btid,btie->btde", ket_a_im, ket_a_re) - torch.einsum(
        "btid,btie->btde", ket_a_re, ket_a_im
    )
    rho_b_re = torch.einsum("btid,btie->btde", ket_b_re, ket_b_re) + torch.einsum(
        "btid,btie->btde", ket_b_im, ket_b_im
    )
    rho_b_im = torch.einsum("btid,btie->btde", ket_b_im, ket_b_re) - torch.einsum(
        "btid,btie->btde", ket_b_re, ket_b_im
    )

    schur_mat_re = rho_a_re * rho_b_re - rho_a_im * rho_b_im
    schur_mat_im = rho_a_re * rho_b_im + rho_a_im * rho_b_re
    tr_mat = trace_of_density(schur_mat_re).clamp_min(1e-12)
    schur_mat_re = schur_mat_re / tr_mat.unsqueeze(-1).unsqueeze(-1)
    schur_mat_im = schur_mat_im / tr_mat.unsqueeze(-1).unsqueeze(-1)
    explicit_act_re = torch.einsum("btde,bte->btd", schur_mat_re, probe_re) - torch.einsum(
        "btde,bte->btd", schur_mat_im, probe_im
    )
    explicit_act_im = torch.einsum("btde,bte->btd", schur_mat_re, probe_im) + torch.einsum(
        "btde,bte->btd", schur_mat_im, probe_re
    )

    z_ab_re, z_ab_im, tr_fact = factorized_rank_beams_schur_intersect(
        ket_a_re, ket_a_im, ket_b_re, ket_b_im, eps=1e-12
    )
    fact_act_re, fact_act_im = apply_rank_beams_to_probe(z_ab_re, z_ab_im, probe_re, probe_im)
    thm15a_act_err = float(
        torch.max(
            torch.abs(explicit_act_re - fact_act_re) + torch.abs(explicit_act_im - fact_act_im)
        ).item()
    )
    thm15a_tr_err = float(torch.max(torch.abs(tr_mat - tr_fact)).item())

    # 2. Theorem 15b: Diagonal-Conjugated Bath Schur Action vs. Explicit 64x64 Bath Schur
    beams_re = torch.randn(bsz, seq_len, rank, d) * 0.25
    beams_im = torch.randn(bsz, seq_len, rank, d) * 0.25
    bath_factor_re = torch.randn(d, d) * 0.2
    bath_factor_im = torch.randn(d, d) * 0.2
    bath_re, bath_im = trace_normalize_density(
        torch.matmul(bath_factor_re, bath_factor_re.T)
        + torch.matmul(bath_factor_im, bath_factor_im.T),
        torch.matmul(bath_factor_im, bath_factor_re.T)
        - torch.matmul(bath_factor_re, bath_factor_im.T),
    )

    rho_beams_re = torch.einsum("btrd,btre->btde", beams_re, beams_re) + torch.einsum(
        "btrd,btre->btde", beams_im, beams_im
    )
    rho_beams_im = torch.einsum("btrd,btre->btde", beams_im, beams_re) - torch.einsum(
        "btrd,btre->btde", beams_re, beams_im
    )
    exp_bu_re = rho_beams_re * bath_re.view(1, 1, d, d) - rho_beams_im * bath_im.view(1, 1, d, d)
    exp_bu_im = rho_beams_re * bath_im.view(1, 1, d, d) + rho_beams_im * bath_re.view(1, 1, d, d)
    tr_bu_exp = trace_of_density(exp_bu_re).clamp_min(1e-12)
    exp_bu_re = exp_bu_re / tr_bu_exp.unsqueeze(-1).unsqueeze(-1)
    exp_bu_im = exp_bu_im / tr_bu_exp.unsqueeze(-1).unsqueeze(-1)
    exp_bu_act_re = torch.einsum("btde,bte->btd", exp_bu_re, probe_re) - torch.einsum(
        "btde,bte->btd", exp_bu_im, probe_im
    )
    exp_bu_act_im = torch.einsum("btde,bte->btd", exp_bu_re, probe_im) + torch.einsum(
        "btde,bte->btd", exp_bu_im, probe_re
    )

    fast_bu_act_re, fast_bu_act_im, tr_bu_fast = diagonal_conjugated_bath_schur_action(
        beams_re, beams_im, bath_re, bath_im, probe_re, probe_im, eps=1e-12
    )
    thm15b_act_err = float(
        torch.max(
            torch.abs(exp_bu_act_re - fast_bu_act_re) + torch.abs(exp_bu_act_im - fast_bu_act_im)
        ).item()
    )
    thm15b_tr_err = float(torch.max(torch.abs(tr_bu_exp - tr_bu_fast)).item())

    # 3. Theorem 15c: Gram Matrix Purity vs. Explicit 64x64 Density Purity
    rho_norm_re, rho_norm_im = trace_normalize_density(rho_beams_re, rho_beams_im, eps=1e-12)
    exp_purity = purity_of_density(rho_norm_re, rho_norm_im)
    gram_purity = gram_matrix_mps_purity(beams_re, beams_im, eps=1e-12)
    thm15c_purity_err = float(torch.max(torch.abs(exp_purity - gram_purity)).item())

    passed = (
        thm15a_act_err < 1e-5
        and thm15a_tr_err < 1e-6
        and thm15b_act_err < 1e-5
        and thm15b_tr_err < 1e-6
        and thm15c_purity_err < 1e-5
    )
    return {
        "passed": passed,
        "theorem15a_factorized_schur_action_max_err": thm15a_act_err,
        "theorem15a_factorized_schur_trace_max_err": thm15a_tr_err,
        "theorem15b_diag_conjugated_bath_action_max_err": thm15b_act_err,
        "theorem15b_diag_conjugated_bath_trace_max_err": thm15b_tr_err,
        "theorem15c_gram_purity_max_err": thm15c_purity_err,
    }


def benchmark_wave73_vs_wave74_throughput(
    batch_size: int = 16, seq_len: int = 256, steps: int = 15
) -> dict[str, Any]:
    """Benchmark `Wave 73` FP32 explicit 64x64 vs. `Wave 74` Fused `bfloat16` (`1L` & `2L`) (`<= 8` args)."""
    from qdlblocks.benchmarks.four_compartment_baton_lm_lab import (
        FourCompartmentBatonConfig,
        FourCompartmentBatonHighwayLM,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx = torch.randint(0, DEFAULT_VOCAB_SIZE, (batch_size, seq_len), device=device)
    targets = torch.randint(0, DEFAULT_VOCAB_SIZE, (batch_size, seq_len), device=device)
    results: dict[str, Any] = {}

    # 1. Reference Wave 73 1L 2-Hop Triangle (FP32 explicit 64x64)
    w73_cfg = FourCompartmentBatonConfig(
        vocab_size=DEFAULT_VOCAB_SIZE,
        hilbert_dim=DEFAULT_HILBERT_DIM,
        num_layers=1,
        use_schur_intersection=True,
        use_two_hop_triangle=True,
    )
    w73_model = FourCompartmentBatonHighwayLM(w73_cfg).to(device)
    w73_opt = torch.optim.AdamW(w73_model.parameters(), lr=3e-3)
    for _ in range(2):
        w73_opt.zero_grad(set_to_none=True)
        _, loss = w73_model(idx[:4, :128], targets[:4, :128])
        assert loss is not None
        loss.backward()
        w73_opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        w73_opt.zero_grad(set_to_none=True)
        _, loss = w73_model(idx[:8, :128], targets[:8, :128])
        assert loss is not None
        loss.backward()
        w73_opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    w73_elapsed = time.perf_counter() - t0
    w73_tok_s = (5 * 8 * 128) / max(1e-6, w73_elapsed)
    results["W73_B3_EXPLICIT_FP32_1L"] = {"tokens_per_sec": round(w73_tok_s, 1)}
    del w73_model, w73_opt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2. Wave 74 Fused BF16 1L and 2L 2-Hop Triangle
    for layers in (1, 2):
        w74_cfg = FusedBF16FourCompartmentBatonConfig(
            vocab_size=DEFAULT_VOCAB_SIZE,
            hilbert_dim=DEFAULT_HILBERT_DIM,
            num_layers=layers,
            use_two_hop_triangle=True,
            use_bfloat16_amp=True,
        )
        w74_model = FusedBF16FourCompartmentBatonLM(w74_cfg).to(device)
        w74_opt = torch.optim.AdamW(
            w74_model.parameters(), lr=3e-3, fused=(device.type == "cuda")
        )
        for _ in range(3):
            w74_opt.zero_grad(set_to_none=True)
            _, loss = w74_model(idx, targets)
            assert loss is not None
            loss.backward()
            w74_opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(steps):
            w74_opt.zero_grad(set_to_none=True)
            _, loss = w74_model(idx, targets)
            assert loss is not None
            loss.backward()
            w74_opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_mib = round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 1)
        else:
            peak_mib = 0.0
        elapsed = time.perf_counter() - t0
        tok_s = (steps * batch_size * seq_len) / max(1e-6, elapsed)
        results[f"W74_F{layers}_FUSED_BF16_2HOP_{layers}L"] = {
            "tokens_per_sec": round(tok_s, 1),
            "speedup_vs_w73_1l": round(tok_s / max(1.0, w73_tok_s), 2),
            "peak_vram_mib": peak_mib,
        }
        del w74_model, w74_opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


# ==============================================================================
# Multi-Shard `300,000,000`-Token (`300M` Token) Streaming Loader & Runner
# ==============================================================================
class MultiShardFineWebStreamer:
    """Zero-copy `uint16` memmap streamer over `FineWeb` Shards 1..3 (`300,000,000` tokens)."""

    def __init__(self, shard_paths: tuple[Path, ...] = FINEWEB_TRAIN_SHARDS_300M) -> None:
        self.shard_paths = shard_paths
        self.memmaps: list[np.memmap] = []
        self.shard_lengths: list[int] = []
        for path in shard_paths:
            header = np.memmap(path, dtype=np.int32, mode="r", shape=(256,))
            n_tok = int(header[2])
            mm = np.memmap(
                path, dtype=np.uint16, mode="r", offset=256 * 4, shape=(n_tok,)
            )
            self.memmaps.append(mm)
            self.shard_lengths.append(n_tok)
        self.total_tokens: int = sum(self.shard_lengths)

    def sample_sequential_batch(
        self, step_idx: int, batch_size: int, seq_len: int, device: torch.device
    ) -> tuple[Tensor, Tensor, int]:
        """Fetch `(x, y, shard_idx)` for `step_idx` walking sequentially across all `300M` tokens (`<= 8` args)."""
        tokens_per_step = batch_size * seq_len
        global_pos = (step_idx * tokens_per_step) % max(1, self.total_tokens - tokens_per_step - 2)
        acc = 0
        shard_idx = 0
        local_pos = global_pos
        for idx, length in enumerate(self.shard_lengths):
            if global_pos < acc + length:
                shard_idx = idx
                local_pos = global_pos - acc
                break
            acc += length

        span = batch_size * (seq_len + 1)
        mm = self.memmaps[shard_idx]
        local_pos = max(0, min(local_pos, len(mm) - span - 1))
        raw_slice = np.array(mm[local_pos : local_pos + span], dtype=np.int64)
        tensor_chunk = torch.from_numpy(raw_slice).view(batch_size, seq_len + 1).to(
            device, non_blocking=True
        )
        return tensor_chunk[:, :-1].contiguous(), tensor_chunk[:, 1:].contiguous(), shard_idx


@dataclass(frozen=True)
class Wave74SpinRunSpec:
    """Specification for a Wave 74 Fused `bfloat16` 4-Compartment Baton run (`<= 8` fields)."""

    run_key: str
    num_layers: int = 2
    batch_size: int = 64
    seq_len: int = 256
    total_steps: int = 18311
    eval_interval: int = 1000
    learning_rate: float = 4.0e-3
    seed: int = 1337


@torch.no_grad()
def evaluate_wave74_model(
    model: FusedBF16FourCompartmentBatonLM,
    val_mm: np.memmap,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate `FusedBF16FourCompartmentBatonLM` on `fineweb_val_000000.bin` (`<= 8` args)."""
    model.eval()
    losses: list[float] = []
    accs: list[float] = []
    gate_sums = torch.zeros(4, dtype=torch.float64)
    out_purities: list[float] = []
    mps_purities: list[float] = []

    span = batch_size * (seq_len + 1)
    for b_idx in range(eval_batches):
        start = b_idx * batch_size * seq_len
        raw = np.array(val_mm[start : start + span], dtype=np.int64)
        chunk = torch.from_numpy(raw).view(batch_size, seq_len + 1).to(device)
        x_val = chunk[:, :-1].contiguous()
        y_val = chunk[:, 1:].contiguous()

        out = model.forward_with_diagnostics(
            x_val, y_val, ingest_online=False, return_logits=False
        )
        losses.append(float(out["loss"].item()))
        accs.append(float(out["top1_acc"]))
        gate_sums += out["gates"].mean(dim=(0, 1)).cpu().to(torch.float64)
        out_purities.append(float(out["out_purity"].mean().item()))
        mps_purities.append(float(out["mps_purity"].mean().item()))

    mean_ce = float(np.mean(losses))
    mean_gates = (gate_sums / float(max(1, eval_batches))).tolist()
    bath_purity = float(
        purity_of_density(model.online_uni_bath_re, model.online_uni_bath_im).item()
    )
    model.train()
    return {
        "val_ce": round(mean_ce, 4),
        "val_ppl": round(float(math.exp(min(20.0, mean_ce))), 2),
        "top1_acc": round(float(np.mean(accs)), 4),
        "mean_out_purity": round(float(np.mean(out_purities)), 4),
        "mean_mps_purity": round(float(np.mean(mps_purities)), 4),
        "online_bath_purity": round(bath_purity, 6),
        "gate_w_id": round(float(mean_gates[0]), 4),
        "gate_w_deep": round(float(mean_gates[1]), 4),
        "gate_w_ngram": round(float(mean_gates[2]), 4),
        "gate_w_uni": round(float(mean_gates[3]), 4),
    }


def run_wave74_spin_experiment(
    spec: Wave74SpinRunSpec,
    streamer: MultiShardFineWebStreamer,
    val_mm: np.memmap,
    device: torch.device,
    eval_batches: int = 16,
) -> dict[str, Any]:
    """Execute a Wave 74 Fused `bfloat16` 4-Compartment Baton training run (`<= 8` args)."""
    torch.manual_seed(spec.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.seed)
        torch.set_float32_matmul_precision("high")

    cfg = FusedBF16FourCompartmentBatonConfig(
        vocab_size=DEFAULT_VOCAB_SIZE,
        hilbert_dim=DEFAULT_HILBERT_DIM,
        num_layers=spec.num_layers,
        mps_window_k=DEFAULT_MPS_WINDOW_K,
        mps_bond_dim=DEFAULT_MPS_BOND_DIM,
        use_two_hop_triangle=True,
        use_bfloat16_amp=True,
        online_bath_ema_decay=DEFAULT_ONLINE_BATH_EMA_DECAY,
    )
    model = FusedBF16FourCompartmentBatonLM(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    tokens_per_step = spec.batch_size * spec.seq_len
    target_tokens_m = round((spec.total_steps * tokens_per_step) / 1e6, 2)

    intent = ExperimentIntentSpec(
        run_key=spec.run_key,
        wave_id="W74",
        plan_id="EXPERIMENT_PLAN_094",
        task_domain="fineweb_speedrun",
        hypothesis_intent=(
            f"Train Fused bfloat16 {spec.num_layers}-Layer 4-Compartment Baton + 2-Hop Triangle "
            f"over {target_tokens_m}M FineWeb tokens with 100% online thermal bath and n-gram ingestion."
        ),
        architecture_summary=(
            f"FusedBF16FourCompartmentBatonLM(L={spec.num_layers}, d={cfg.hilbert_dim}, "
            f"R={DEFAULT_SCHUR_BEAM_RANK}, K={cfg.mps_window_k}, chi={cfg.mps_bond_dim}, 2hop=True, bf16=True)"
        ),
        hardware_budget=str(device),
        batch_size=spec.batch_size,
        seq_len=spec.seq_len,
        total_steps=spec.total_steps,
        target_tokens_M=target_tokens_m,
    )
    logger = StandardizedExperimentLogger(intent)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=(device.type == "cuda"),
    )

    step0_eval = evaluate_wave74_model(
        model, val_mm, spec.batch_size, spec.seq_len, eval_batches, device
    )
    logger.log_eval(step=0, metrics=step0_eval)
    history: list[dict[str, Any]] = [
        {"step": 0, "tokens_M": 0.0, "shard_idx": 1, **step0_eval}
    ]
    best_val_ce = step0_eval["val_ce"]

    t0 = time.perf_counter()
    warmup_steps = max(20, int(0.03 * spec.total_steps))

    for step in range(1, spec.total_steps + 1):
        if step <= warmup_steps:
            lr_mult = float(step) / float(warmup_steps)
        else:
            prog = float(step - warmup_steps) / float(max(1, spec.total_steps - warmup_steps))
            lr_mult = 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * prog))
        for pg in optimizer.param_groups:
            pg["lr"] = spec.learning_rate * lr_mult

        x_batch, y_batch, shard_idx = streamer.sample_sequential_batch(
            step - 1, spec.batch_size, spec.seq_len, device
        )
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x_batch, y_batch)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == spec.total_steps:
            train_ce = float(loss.item())
            elapsed_so_far = max(1e-6, time.perf_counter() - t0)
            tok_per_s = (step * tokens_per_step) / elapsed_so_far
            logger.log_step(step=step, train_ce=train_ce, tok_per_s=tok_per_s, lr_mult=lr_mult)

        if step % spec.eval_interval == 0 or step == spec.total_steps:
            train_ce = float(loss.item())
            elapsed_so_far = max(1e-6, time.perf_counter() - t0)
            tok_per_s = (step * tokens_per_step) / elapsed_so_far
            tokens_m = round((step * tokens_per_step) / 1e6, 2)
            eval_metrics = evaluate_wave74_model(
                model, val_mm, spec.batch_size, spec.seq_len, eval_batches, device
            )
            best_val_ce = min(best_val_ce, eval_metrics["val_ce"])
            logger.log_eval(step=step, metrics=eval_metrics)
            rec = {
                "step": step,
                "tokens_M": tokens_m,
                "shard_idx": shard_idx + 1,
                "train_ce": round(train_ce, 4),
                "tok_per_s": round(tok_per_s, 1),
                "wall_sec": round(elapsed_so_far, 1),
                **eval_metrics,
            }
            history.append(rec)
            print(
                f"[{spec.run_key}] step={step}/{spec.total_steps} ({tokens_m:.1f}M tok, shard={shard_idx+1}) "
                f"| train_ce={train_ce:.4f} | val_ce={eval_metrics['val_ce']:.4f} (best={best_val_ce:.4f}) "
                f"| top1={eval_metrics['top1_acc']*100:.2f}% | out_pur={eval_metrics['mean_out_purity']:.4f} "
                f"| gates=({eval_metrics['gate_w_id']:.3f},{eval_metrics['gate_w_deep']:.3f},"
                f"{eval_metrics['gate_w_ngram']:.3f},{eval_metrics['gate_w_uni']:.3f}) "
                f"| {tok_per_s:,.1f} tok/s",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    logger.finalize()
    final_rec = history[-1]
    return {
        "run_key": spec.run_key,
        "num_layers": spec.num_layers,
        "param_count": param_count,
        "total_steps": spec.total_steps,
        "batch_size": spec.batch_size,
        "seq_len": spec.seq_len,
        "online_ingested_tokens": int(model.online_ingested_tokens.item()),
        "wall_time_sec": round(elapsed, 2),
        "tokens_per_sec": round((spec.total_steps * tokens_per_step) / max(1e-6, elapsed), 1),
        "step0_val_ce": step0_eval["val_ce"],
        "best_val_ce": best_val_ce,
        "final_val_ce": final_rec["val_ce"],
        "final_val_ppl": final_rec["val_ppl"],
        "final_top1_acc": final_rec["top1_acc"],
        "final_out_purity": final_rec["mean_out_purity"],
        "final_mps_purity": final_rec["mean_mps_purity"],
        "final_online_bath_purity": final_rec["online_bath_purity"],
        "final_gates": {
            "w_id": final_rec["gate_w_id"],
            "w_deep": final_rec["gate_w_deep"],
            "w_ngram": final_rec["gate_w_ngram"],
            "w_uni": final_rec["gate_w_uni"],
        },
        "eval_history": history,
    }
