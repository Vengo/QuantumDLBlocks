"""Wave 73 (`EXPERIMENT_PLAN_093.md`): 1-Layer & 2-Layer 4-Compartment Quantum Statistical
Mechanics Baton Language Model Laboratory (`four_compartment_baton_lm_lab.py`).

Implements the 4-Compartment Quantum Statistical Mechanics Baton Architecture synthesized from
the shared Gemini notebook **"Branch • Branch • Quantum States and the Born Rule"**
(Pages 48–72: `HybridMPSBornSchurBlock`, `CodingHairpinHybridNet`, the 4-Compartment Baton &
Quantum Statistical Backoff, and the $O(T d^2)$ Distributive Law Shortcut for 1–2 layers),
adapted for **Modded-NanoGPT Speedrun** compliance (100% online unigram thermal bath and dual
MPS + Lindblad n-gram learning as tokens are ingested; zero offline pre-step corpus scanning).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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

# ==============================================================================
# Top-of-File Named Constants (Rule 7 — Synchronized with EXPERIMENT_PLAN_093.md)
# ==============================================================================
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
FINEWEB_CACHE_DIR: Path = REPO_ROOT / ".cache" / "fineweb_100M"
DEFAULT_TRAIN_SHARD: Path = FINEWEB_CACHE_DIR / "fineweb_train_000001.bin"
DEFAULT_VAL_SHARD: Path = FINEWEB_CACHE_DIR / "fineweb_val_000000.bin"

FINEWEB_HEADER_INT32_COUNT: int = 256
FINEWEB_MAGIC_NUMBER: int = 20240520
DEFAULT_VOCAB_SIZE: int = 50257

DEFAULT_HILBERT_DIM: int = 64
DEFAULT_NUM_PACKETS: int = 8
PACKET_IDX_UNIGRAM_BATH: int = 0
PACKET_IDX_MPS_BATON: int = 1
PACKET_IDX_LINDBLAD_TRANS: int = 2
PACKET_SLICE_SEQ_START: int = 3
PACKET_SLICE_SEQ_END: int = 7
PACKET_IDX_TOKEN_ID: int = 7
NUM_SEQ_ATTN_PACKETS: int = 4

DEFAULT_MPS_WINDOW_K: int = 4
DEFAULT_MPS_BOND_DIM: int = 8
DEFAULT_MPS_CORE_INPUT_DIM: int = 16
DEFAULT_MPS_CORE_INIT_STD: float = 0.08

DEFAULT_ONLINE_BATH_EMA_DECAY: float = 0.05
DEFAULT_ONLINE_TRANS_EMA_DECAY: float = 0.05
DEFAULT_DOC_BATH_MIX: float = 0.20
DEFAULT_IN_CONTEXT_LINDBLAD_GAMMA: float = 0.88
DEFAULT_MPS_NGRAM_BLEND: float = 0.65

DEFAULT_VACUUM_EPS: float = 1e-4
DEFAULT_TRACE_EPS: float = 1e-6
DEFAULT_NORM_EPS: float = 1e-6
DEFAULT_PURITY_FLOOR_OFFSET: float = 0.02
DEFAULT_PURITY_MIN_CLAMP: float = 0.01
DEFAULT_ROPE_THETA_BASE: float = 10000.0

DEFAULT_LOGIT_SOFTCAP: float = 30.0
DEFAULT_BORN_READOUT_SCALE: float = 14.0
DEFAULT_UNIGRAM_PRIOR_WEIGHT: float = 0.35
DEFAULT_IN_CONTEXT_COPY_SCALE: float = 4.5
DEFAULT_TOKEN_ID_COPY_SCALE: float = 8.0

NUM_BINARY_ID_BITS: int = 16
NUM_WALSH_PARITY_FEATURES: int = 48

VARIANT_B0_ADDITIVE_NO_SCHUR_1L: str = "W73_B0_ADDITIVE_NO_SCHUR_1L"
VARIANT_B1_FOUR_COMPARTMENT_1L: str = "W73_B1_FOUR_COMPARTMENT_1L"
VARIANT_B2_FOUR_COMPARTMENT_2L: str = "W73_B2_FOUR_COMPARTMENT_2L"
VARIANT_B3_FOUR_COMPARTMENT_2HOP_TRIANGLE_1L: str = "W73_B3_FOUR_COMPARTMENT_2HOP_TRIANGLE_1L"

ALL_WAVE73_VARIANTS: tuple[str, ...] = (
    VARIANT_B0_ADDITIVE_NO_SCHUR_1L,
    VARIANT_B1_FOUR_COMPARTMENT_1L,
    VARIANT_B2_FOUR_COMPARTMENT_2L,
    VARIANT_B3_FOUR_COMPARTMENT_2HOP_TRIANGLE_1L,
)


# ==============================================================================
# Frozen Configuration Dataclasses (Rule 2 & Rule 3 — <= 8 Fields/Args)
# ==============================================================================
@dataclass(frozen=True)
class FourCompartmentBatonConfig:
    """Frozen structural configuration for `FourCompartmentBatonHighwayLM` (`<= 8` fields)."""

    vocab_size: int = DEFAULT_VOCAB_SIZE
    hilbert_dim: int = DEFAULT_HILBERT_DIM
    num_layers: int = 1
    mps_window_k: int = DEFAULT_MPS_WINDOW_K
    mps_bond_dim: int = DEFAULT_MPS_BOND_DIM
    use_schur_intersection: bool = True
    use_two_hop_triangle: bool = False
    online_bath_ema_decay: float = DEFAULT_ONLINE_BATH_EMA_DECAY


@dataclass(frozen=True)
class Wave73TrainSpec:
    """Frozen training and evaluation specification for Wave 73 FineWeb runs (`<= 8` fields)."""

    variant_id: str = VARIANT_B1_FOUR_COMPARTMENT_1L
    total_steps: int = 80
    eval_interval: int = 20
    batch_size: int = 8
    seq_len: int = 128
    learning_rate: float = 3.5e-3
    eval_batches: int = 12
    seed: int = 1337


def build_wave73_config_for_variant(
    variant_id: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> FourCompartmentBatonConfig:
    """Map a canonical Wave 73 variant ID to its `FourCompartmentBatonConfig` (`<= 8` args)."""
    if variant_id == VARIANT_B0_ADDITIVE_NO_SCHUR_1L:
        return FourCompartmentBatonConfig(
            vocab_size=vocab_size, num_layers=1, use_schur_intersection=False, use_two_hop_triangle=False
        )
    if variant_id == VARIANT_B1_FOUR_COMPARTMENT_1L:
        return FourCompartmentBatonConfig(
            vocab_size=vocab_size, num_layers=1, use_schur_intersection=True, use_two_hop_triangle=False
        )
    if variant_id == VARIANT_B2_FOUR_COMPARTMENT_2L:
        return FourCompartmentBatonConfig(
            vocab_size=vocab_size, num_layers=2, use_schur_intersection=True, use_two_hop_triangle=False
        )
    if variant_id == VARIANT_B3_FOUR_COMPARTMENT_2HOP_TRIANGLE_1L:
        return FourCompartmentBatonConfig(
            vocab_size=vocab_size, num_layers=1, use_schur_intersection=True, use_two_hop_triangle=True
        )
    raise ValueError(f"Unknown Wave 73 variant_id: {variant_id!r}")


# ==============================================================================
# Complex Density-Matrix & Walsh–Hadamard Primitives (`d = 64`)
# ==============================================================================
def normalize_complex_ket(
    ket_re: Tensor, ket_im: Tensor, eps: float = DEFAULT_NORM_EPS
) -> tuple[Tensor, Tensor]:
    """L2-normalize a complex vector `(ket_re, ket_im)` along its last dimension (`<= 8` args)."""
    norm_sq = ket_re.square().sum(dim=-1, keepdim=True) + ket_im.square().sum(dim=-1, keepdim=True)
    inv_norm = torch.rsqrt(norm_sq + eps)
    return ket_re * inv_norm, ket_im * inv_norm


def pure_ket_to_density(ket_re: Tensor, ket_im: Tensor) -> tuple[Tensor, Tensor]:
    """Form unit-trace Hermitian density matrix `|ψ⟩⟨ψ|` from normalized `(ket_re, ket_im)` (`<= 8` args)."""
    rho_re = ket_re.unsqueeze(-1) * ket_re.unsqueeze(-2) + ket_im.unsqueeze(-1) * ket_im.unsqueeze(-2)
    rho_im = ket_im.unsqueeze(-1) * ket_re.unsqueeze(-2) - ket_re.unsqueeze(-1) * ket_im.unsqueeze(-2)
    return rho_re, rho_im


def trace_of_density(rho_re: Tensor) -> Tensor:
    """Return real trace `Tr(ρ) = ∑_i Re(ρ)_{ii}` along the last two dimensions (`<= 8` args)."""
    return torch.diagonal(rho_re, dim1=-2, dim2=-1).sum(dim=-1)


def purity_of_density(rho_re: Tensor, rho_im: Tensor) -> Tensor:
    """Return quantum state purity `Tr(ρ^2) = ‖Re(ρ)‖_F^2 + ‖Im(ρ)‖_F^2` for Hermitian `ρ` (`<= 8` args)."""
    return rho_re.square().sum(dim=(-2, -1)) + rho_im.square().sum(dim=(-2, -1))


def trace_normalize_density(
    rho_re: Tensor, rho_im: Tensor, eps: float = DEFAULT_TRACE_EPS
) -> tuple[Tensor, Tensor]:
    """Normalize a Hermitian PSD matrix `(rho_re, rho_im)` to unit trace `Tr(ρ) = 1` (`<= 8` args)."""
    tr_val = trace_of_density(rho_re).clamp_min(eps).unsqueeze(-1).unsqueeze(-1)
    return rho_re / tr_val, rho_im / tr_val


def rank_beams_to_density(
    beams_re: Tensor,
    beams_im: Tensor,
    fallback_re: Tensor,
    fallback_im: Tensor,
    eps: float = DEFAULT_TRACE_EPS,
) -> tuple[Tensor, Tensor]:
    """Contract rank-`χ` complex beams `(..., χ, d)` into a unit-trace `d × d` density matrix (`<= 8` args)."""
    rho_re = torch.einsum("...ld,...le->...de", beams_re, beams_re) + torch.einsum(
        "...ld,...le->...de", beams_im, beams_im
    )
    rho_im = torch.einsum("...ld,...le->...de", beams_im, beams_re) - torch.einsum(
        "...ld,...le->...de", beams_re, beams_im
    )
    return trace_normalize_density(rho_re + eps * fallback_re, rho_im + eps * fallback_im, eps=eps)


def complex_schur_product(
    rho_a_re: Tensor,
    rho_a_im: Tensor,
    rho_b_re: Tensor,
    rho_b_im: Tensor,
    eps: float = DEFAULT_TRACE_EPS,
) -> tuple[Tensor, Tensor]:
    """Compute entrywise complex Schur product `(ρ_A ⊙ ρ_B)` and trace-normalize (`<= 8` args)."""
    out_re = rho_a_re * rho_b_re - rho_a_im * rho_b_im
    out_im = rho_a_re * rho_b_im + rho_a_im * rho_b_re
    d = out_re.shape[-1]
    eye = torch.eye(d, device=out_re.device, dtype=out_re.dtype)
    return trace_normalize_density(out_re + (eps / float(d)) * eye, out_im, eps=eps)


def born_quadratic_expectation(
    rho_re: Tensor, rho_im: Tensor, ket_re: Tensor, ket_im: Tensor
) -> Tensor:
    """Evaluate Born quadratic expectation `⟨ψ | ρ | ψ⟩ ≥ 0` for Hermitian `ρ` (`<= 8` args)."""
    ra = torch.einsum("...de,...e->...d", rho_re, ket_re)
    rb = torch.einsum("...de,...e->...d", rho_re, ket_im)
    ib = torch.einsum("...de,...e->...d", rho_im, ket_im)
    val = (ket_re * ra).sum(dim=-1) + (ket_im * rb).sum(dim=-1) - 2.0 * (ket_re * ib).sum(dim=-1)
    return val.clamp_min(0.0)


def extract_motif_ket_from_density(
    rho_re: Tensor,
    rho_im: Tensor,
    probe_re: Tensor,
    probe_im: Tensor,
    fallback_re: Tensor,
    fallback_im: Tensor,
) -> tuple[Tensor, Tensor]:
    """Project Hermitian density matrix `ρ` along complex probe `|u⟩` to extract a normalized ket (`<= 8` args)."""
    proj_re = torch.einsum("...de,he->...hd", rho_re, probe_re) - torch.einsum(
        "...de,he->...hd", rho_im, probe_im
    )
    proj_im = torch.einsum("...de,he->...hd", rho_re, probe_im) + torch.einsum(
        "...de,he->...hd", rho_im, probe_re
    )
    return normalize_complex_ket(
        proj_re + 0.25 * fallback_re.unsqueeze(-2), proj_im + 0.25 * fallback_im.unsqueeze(-2)
    )


def build_walsh_hadamard_parity_pairs(hilbert_dim: int = DEFAULT_HILBERT_DIM) -> Tensor:
    """Build deterministic 2-bit and 3-bit index tuples `(d - 16, 3)` for the Walsh–Hadamard Token-ID basis (`<= 8` args)."""
    needed = max(0, hilbert_dim - NUM_BINARY_ID_BITS)
    triples: list[tuple[int, int, int]] = []
    for i in range(NUM_BINARY_ID_BITS):
        for j in range(i + 1, NUM_BINARY_ID_BITS):
            triples.append((i, j, -1))
            if len(triples) >= needed:
                return torch.tensor(triples[:needed], dtype=torch.long)
    for i in range(NUM_BINARY_ID_BITS):
        for j in range(i + 1, NUM_BINARY_ID_BITS):
            for k in range(j + 1, NUM_BINARY_ID_BITS):
                triples.append((i, j, k))
                if len(triples) >= needed:
                    return torch.tensor(triples[:needed], dtype=torch.long)
    return torch.tensor(triples[:needed], dtype=torch.long)


def encode_walsh_hadamard_token_id_kets(
    token_ids: Tensor, parity_pairs: Tensor, hilbert_dim: int = DEFAULT_HILBERT_DIM
) -> tuple[Tensor, Tensor]:
    """Map token IDs `(B, T)` parameter-free to unit-norm `d`-dimensional Walsh–Hadamard phase kets (`<= 8` args)."""
    device = token_ids.device
    bit_shifts = torch.arange(NUM_BINARY_ID_BITS, device=device, dtype=torch.long)
    bits = ((token_ids.unsqueeze(-1).long() >> bit_shifts) & 1).to(torch.float32)
    signs = 1.0 - 2.0 * bits
    if hilbert_dim <= NUM_BINARY_ID_BITS:
        raw = signs[..., :hilbert_dim]
    else:
        idx_a = parity_pairs[:, 0].to(device)
        idx_b = parity_pairs[:, 1].to(device)
        idx_c = parity_pairs[:, 2].to(device)
        parities = signs[..., idx_a] * signs[..., idx_b]
        has_third = (idx_c >= 0).to(signs.dtype)
        third_factor = torch.where(idx_c >= 0, signs[..., idx_c.clamp_min(0)], torch.ones_like(parities))
        parities = parities * (has_third * third_factor + (1.0 - has_third))
        raw = torch.cat([signs, parities], dim=-1)[..., :hilbert_dim]
    ket_re = raw * (1.0 / math.sqrt(float(raw.shape[-1])))
    return ket_re, torch.zeros_like(ket_re)



def apply_complex_rope(
    ket_re: Tensor, ket_im: Tensor, cos_pos: Tensor, sin_pos: Tensor
) -> tuple[Tensor, Tensor]:
    """Apply unitary Complex RoPE phase rotation `(ket_re + i ket_im) * (cos + i sin)` (`<= 8` args)."""
    return ket_re * cos_pos - ket_im * sin_pos, ket_re * sin_pos + ket_im * cos_pos


# ==============================================================================
# Compartment 2a (`Packet 1`): Vectorized K=4, χ=8 1D MPS Baton Module
# ==============================================================================
class VectorizedMPSBatonModule(nn.Module):
    """Vectorized 1D Matrix Product State (MPS) transfer chain (`Packet 1`, Gemini chat Pages 51–53)."""

    def __init__(self, config: FourCompartmentBatonConfig) -> None:
        super().__init__()
        self.window_k: int = config.mps_window_k
        self.bond_dim: int = config.mps_bond_dim
        self.hilbert_dim: int = config.hilbert_dim
        self.core_in_dim: int = min(DEFAULT_MPS_CORE_INPUT_DIM, config.hilbert_dim)

        num_transfer = max(1, self.window_k - 1)
        self.in_proj_re = nn.Parameter(
            torch.randn(self.hilbert_dim, self.core_in_dim) / math.sqrt(float(self.hilbert_dim))
        )
        self.in_proj_im = nn.Parameter(torch.randn(self.hilbert_dim, self.core_in_dim) * 0.05)
        self.transfer_cores_re = nn.Parameter(
            torch.randn(num_transfer, self.bond_dim, self.core_in_dim, self.bond_dim)
            * DEFAULT_MPS_CORE_INIT_STD
        )
        self.transfer_cores_im = nn.Parameter(
            torch.randn(num_transfer, self.bond_dim, self.core_in_dim, self.bond_dim)
            * DEFAULT_MPS_CORE_INIT_STD
        )
        self.end_core_re = nn.Parameter(
            torch.randn(self.bond_dim, self.core_in_dim, self.bond_dim, self.hilbert_dim)
            * DEFAULT_MPS_CORE_INIT_STD
        )
        self.end_core_im = nn.Parameter(
            torch.randn(self.bond_dim, self.core_in_dim, self.bond_dim, self.hilbert_dim)
            * DEFAULT_MPS_CORE_INIT_STD
        )
        self.bath_seed_proj = nn.Parameter(
            torch.randn(self.hilbert_dim, self.bond_dim) / math.sqrt(float(self.hilbert_dim))
        )

    def forward(
        self, ket_re: Tensor, ket_im: Tensor, rho_uni_re: Tensor, rho_uni_im: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Contract the causal `K`-gram window across all `T` positions simultaneously (`<= 8` args)."""
        k_win = self.window_k
        xi_re = torch.matmul(ket_re, self.in_proj_re) - torch.matmul(ket_im, self.in_proj_im)
        xi_im = torch.matmul(ket_re, self.in_proj_im) + torch.matmul(ket_im, self.in_proj_re)
        xi_re, xi_im = normalize_complex_ket(xi_re, xi_im)

        pad_re = F.pad(xi_re, (0, 0, k_win - 1, 0))
        pad_im = F.pad(xi_im, (0, 0, k_win - 1, 0))
        win_re = pad_re.unfold(dimension=1, size=k_win, step=1).permute(0, 1, 3, 2).contiguous()
        win_im = pad_im.unfold(dimension=1, size=k_win, step=1).permute(0, 1, 3, 2).contiguous()

        uni_diag = torch.diagonal(rho_uni_re, dim1=-2, dim2=-1)
        baton_re = torch.matmul(uni_diag, self.bath_seed_proj)
        baton_im = torch.zeros_like(baton_re)
        baton_re[:, :, 0] = baton_re[:, :, 0] + 1.0
        baton_re, baton_im = normalize_complex_ket(baton_re, baton_im)

        for step_idx in range(k_win - 1):
            step_xi_re = win_re[:, :, step_idx, :]
            step_xi_im = win_im[:, :, step_idx, :]
            core_re = self.transfer_cores_re[step_idx]
            core_im = self.transfer_cores_im[step_idx]

            prod_re = baton_re.unsqueeze(-1) * step_xi_re.unsqueeze(-2) - baton_im.unsqueeze(
                -1
            ) * step_xi_im.unsqueeze(-2)
            prod_im = baton_re.unsqueeze(-1) * step_xi_im.unsqueeze(-2) + baton_im.unsqueeze(
                -1
            ) * step_xi_re.unsqueeze(-2)

            delta_re = torch.einsum("btam,amc->btc", prod_re, core_re) - torch.einsum(
                "btam,amc->btc", prod_im, core_im
            )
            delta_im = torch.einsum("btam,amc->btc", prod_re, core_im) + torch.einsum(
                "btam,amc->btc", prod_im, core_re
            )
            baton_re, baton_im = normalize_complex_ket(baton_re + delta_re, baton_im + delta_im)

        last_xi_re = win_re[:, :, -1, :]
        last_xi_im = win_im[:, :, -1, :]
        end_in_re = baton_re.unsqueeze(-1) * last_xi_re.unsqueeze(-2) - baton_im.unsqueeze(
            -1
        ) * last_xi_im.unsqueeze(-2)
        end_in_im = baton_re.unsqueeze(-1) * last_xi_im.unsqueeze(-2) + baton_im.unsqueeze(
            -1
        ) * last_xi_re.unsqueeze(-2)

        beams_re = torch.einsum("btam,amld->btld", end_in_re, self.end_core_re) - torch.einsum(
            "btam,amld->btld", end_in_im, self.end_core_im
        )
        beams_im = torch.einsum("btam,amld->btld", end_in_re, self.end_core_im) + torch.einsum(
            "btam,amld->btld", end_in_im, self.end_core_re
        )
        beams_re = beams_re + 0.20 * ket_re.unsqueeze(-2)
        beams_im = beams_im + 0.20 * ket_im.unsqueeze(-2)

        rho_mps_re, rho_mps_im = rank_beams_to_density(
            beams_re, beams_im, rho_uni_re, rho_uni_im
        )
        mean_beam_re, mean_beam_im = normalize_complex_ket(
            beams_re.mean(dim=-2), beams_im.mean(dim=-2)
        )
        return rho_mps_re, rho_mps_im, mean_beam_re, mean_beam_im


# ==============================================================================
# Single 4-Compartment Quantum Statistical Mechanics Baton Layer
# ==============================================================================
class FourCompartmentBatonLayer(nn.Module):
    """A single layer of the 4-Compartment Quantum Statistical Mechanics Baton Highway (`d = 64, H = 8`)."""

    def __init__(self, config: FourCompartmentBatonConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hilbert_dim
        self.mps_baton = VectorizedMPSBatonModule(config)

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

    def _compute_in_context_and_streaming_transition_packet(
        self,
        ket_re: Tensor,
        ket_im: Tensor,
        trans_ema_re: Tensor,
        trans_ema_im: Tensor,
        rho_uni_re: Tensor,
        rho_uni_im: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute `Packet 2` (`ρ_t^{trans}`) via causal in-context Lindblad transitions + cross-batch `T_bi^{ema}` (`<= 8` args)."""
        seq_len = ket_re.shape[1]
        device = ket_re.device
        dtype = ket_re.dtype

        prev_ket_re = F.pad(ket_re[:, :-1, :], (0, 0, 1, 0))
        prev_ket_im = F.pad(ket_im[:, :-1, :], (0, 0, 1, 0))

        ov_re = torch.einsum("btd,bsd->bts", ket_re, prev_ket_re) + torch.einsum(
            "btd,bsd->bts", ket_im, prev_ket_im
        )
        ov_im = torch.einsum("btd,bsd->bts", ket_im, prev_ket_re) - torch.einsum(
            "btd,bsd->bts", ket_re, prev_ket_im
        )
        ov_sq = ov_re.square() + ov_im.square()

        idx = torch.arange(seq_len, device=device)
        strict_past_mask = (idx.unsqueeze(1) > idx.unsqueeze(0)) & (idx.unsqueeze(0) >= 1)
        lag = (idx.unsqueeze(1) - 1 - idx.unsqueeze(0)).clamp_min(0).to(dtype)
        decay = (
            (1.0 - DEFAULT_IN_CONTEXT_LINDBLAD_GAMMA)
            * (DEFAULT_IN_CONTEXT_LINDBLAD_GAMMA ** lag)
        ) * strict_past_mask.to(dtype)

        born_pull = (ov_sq.square()) * decay.unsqueeze(0)
        ctx_weights_re = ov_re * born_pull
        ctx_weights_im = ov_im * born_pull
        ctx_ket_re = torch.einsum("bts,bsd->btd", ctx_weights_re, ket_re) - torch.einsum(
            "bts,bsd->btd", ctx_weights_im, ket_im
        )
        ctx_ket_im = torch.einsum("bts,bsd->btd", ctx_weights_re, ket_im) + torch.einsum(
            "bts,bsd->btd", ctx_weights_im, ket_re
        )
        ctx_norm_re, ctx_norm_im = normalize_complex_ket(ctx_ket_re, ctx_ket_im)

        op_re = trans_ema_re + self.trans_learned_re
        op_im = trans_ema_im + self.trans_learned_im
        ema_ket_re = torch.matmul(ket_re, op_re.transpose(-2, -1)) - torch.matmul(
            ket_im, op_im.transpose(-2, -1)
        )
        ema_ket_im = torch.matmul(ket_re, op_im.transpose(-2, -1)) + torch.matmul(
            ket_im, op_re.transpose(-2, -1)
        )
        ema_norm_re, ema_norm_im = normalize_complex_ket(
            ema_ket_re + 0.15 * ket_re, ema_ket_im + 0.15 * ket_im
        )

        ctx_rho_re, ctx_rho_im = pure_ket_to_density(ctx_norm_re, ctx_norm_im)
        ema_rho_re, ema_rho_im = pure_ket_to_density(ema_norm_re, ema_norm_im)
        rho_trans_re, rho_trans_im = trace_normalize_density(
            0.70 * ctx_rho_re + 0.25 * ema_rho_re + 0.05 * rho_uni_re,
            0.70 * ctx_rho_im + 0.25 * ema_rho_im + 0.05 * rho_uni_im,
        )
        comb_re, comb_im = normalize_complex_ket(
            0.70 * ctx_norm_re + 0.30 * ema_norm_re,
            0.70 * ctx_norm_im + 0.30 * ema_norm_im,
        )
        in_ctx_copy_weights = ov_sq * strict_past_mask.to(dtype).unsqueeze(0)
        return rho_trans_re, rho_trans_im, comb_re, comb_im, in_ctx_copy_weights

    def _compute_exterior_born_sequence_packets(
        self,
        ket_re: Tensor,
        ket_im: Tensor,
        rho_mps_re: Tensor,
        rho_mps_im: Tensor,
        cos_pos: Tensor,
        sin_pos: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute `Packets 3–6` (`Compartment 3`: Exterior-Masked Motif-Preconditioned Born-Attention) (`<= 8` args)."""
        _, seq_len, d = ket_re.shape
        device = ket_re.device
        dtype = ket_re.dtype

        motif_re, motif_im = extract_motif_ket_from_density(
            rho_mps_re, rho_mps_im, self.motif_probes_re, self.motif_probes_im, ket_re, ket_im
        )
        q_re = torch.einsum("bthd,hde->bthe", motif_re, self.wq_re) - torch.einsum(
            "bthd,hde->bthe", motif_im, self.wq_im
        )
        q_im = torch.einsum("bthd,hde->bthe", motif_re, self.wq_im) + torch.einsum(
            "bthd,hde->bthe", motif_im, self.wq_re
        )
        k_re = torch.einsum("bthd,hde->bthe", motif_re, self.wk_re) - torch.einsum(
            "bthd,hde->bthe", motif_im, self.wk_im
        )
        k_im = torch.einsum("bthd,hde->bthe", motif_re, self.wk_im) + torch.einsum(
            "bthd,hde->bthe", motif_im, self.wk_re
        )
        v_re = torch.einsum("bthd,hde->bthe", motif_re, self.wv_re) - torch.einsum(
            "bthd,hde->bthe", motif_im, self.wv_im
        )
        v_im = torch.einsum("bthd,hde->bthe", motif_re, self.wv_im) + torch.einsum(
            "bthd,hde->bthe", motif_im, self.wv_re
        )

        q_re, q_im = normalize_complex_ket(q_re, q_im)
        k_re, k_im = normalize_complex_ket(k_re, k_im)
        v_re, v_im = normalize_complex_ket(v_re, v_im)
        q_re, q_im = apply_complex_rope(q_re, q_im, cos_pos.unsqueeze(2), sin_pos.unsqueeze(2))
        k_re, k_im = apply_complex_rope(k_re, k_im, cos_pos.unsqueeze(2), sin_pos.unsqueeze(2))

        inner_re = torch.einsum("bthd,bshd->bhts", q_re, k_re) + torch.einsum(
            "bthd,bshd->bhts", q_im, k_im
        )
        inner_im = torch.einsum("bthd,bshd->bhts", q_im, k_re) - torch.einsum(
            "bthd,bshd->bhts", q_re, k_im
        )
        born_scores = inner_re.square() + inner_im.square()

        mps_purity = purity_of_density(rho_mps_re, rho_mps_im)
        purity_rel = (mps_purity - (1.0 / float(d)) + DEFAULT_PURITY_FLOOR_OFFSET).clamp(
            min=DEFAULT_PURITY_MIN_CLAMP, max=1.0
        )
        born_scores = born_scores * purity_rel.unsqueeze(1).unsqueeze(2)

        t_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        s_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        causal_mask = s_idx <= t_idx
        k_win = self.config.mps_window_k
        exterior_mask = torch.where(t_idx >= k_win, s_idx <= (t_idx - k_win), causal_mask)

        head_masks = torch.stack(
            [exterior_mask, exterior_mask, exterior_mask, causal_mask], dim=0
        ).to(dtype)
        masked_scores = born_scores * head_masks.unsqueeze(0)
        attn_probs = masked_scores / (masked_scores.sum(dim=-1, keepdim=True) + DEFAULT_TRACE_EPS)

        if self.config.use_two_hop_triangle and seq_len > 1:
            local_band = ((t_idx - s_idx) >= 0) & ((t_idx - s_idx) < k_win)
            local_op = local_band.to(dtype)
            local_op = local_op / local_op.sum(dim=-1, keepdim=True).clamp_min(1.0)
            two_hop = torch.matmul(attn_probs, local_op.unsqueeze(0).unsqueeze(0))
            attn_probs = 0.70 * attn_probs + 0.30 * two_hop

        val_rho_re, val_rho_im = pure_ket_to_density(v_re, v_im)
        seq_pkts_re = torch.einsum("bhts,bshde->bthde", attn_probs, val_rho_re)
        seq_pkts_im = torch.einsum("bhts,bshde->bthde", attn_probs, val_rho_im)
        seq_pkts_re, seq_pkts_im = trace_normalize_density(seq_pkts_re, seq_pkts_im)

        agg_ket_re, agg_ket_im = normalize_complex_ket(
            torch.einsum("bhts,bshd->bthd", attn_probs, v_re).mean(dim=2),
            torch.einsum("bhts,bshd->bthd", attn_probs, v_im).mean(dim=2),
        )
        return seq_pkts_re, seq_pkts_im, agg_ket_re, agg_ket_im, attn_probs

    def _compute_token_id_packet(
        self, ket_re: Tensor, id_ket_re: Tensor, id_ket_im: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute `Packet 7` (`Compartment 4`: Deterministic 64D Walsh–Hadamard Token-ID Routing with `U_V = I`) (`<= 8` args)."""
        seq_len = ket_re.shape[1]
        device = ket_re.device
        dtype = ket_re.dtype

        q_id = F.normalize(torch.matmul(ket_re + id_ket_re, self.id_query_re), dim=-1)
        k_id = F.normalize(torch.matmul(ket_re + id_ket_re, self.id_key_re), dim=-1)
        scores = torch.einsum("btd,bsd->bts", q_id, k_id).square()

        t_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        s_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        causal_mask = (s_idx <= t_idx).to(dtype)
        scores = scores * causal_mask.unsqueeze(0)
        id_attn_probs = scores / (scores.sum(dim=-1, keepdim=True) + DEFAULT_TRACE_EPS)

        id_pure_re, id_pure_im = pure_ket_to_density(id_ket_re, id_ket_im)
        rho_id_re = torch.einsum("bts,bsde->btde", id_attn_probs, id_pure_re)
        rho_id_im = torch.einsum("bts,bsde->btde", id_attn_probs, id_pure_im)
        rho_id_re, rho_id_im = trace_normalize_density(rho_id_re, rho_id_im)
        return rho_id_re, rho_id_im, id_attn_probs

    def forward(
        self,
        ket_re: Tensor,
        ket_im: Tensor,
        rho_uni_re: Tensor,
        rho_uni_im: Tensor,
        trans_ema_re: Tensor,
        trans_ema_im: Tensor,
        id_ket_re: Tensor,
        id_ket_im: Tensor,
    ) -> dict[str, Tensor]:
        """Execute one 4-Compartment Baton layer and assemble the 8 highway packets + backoff output (`<= 8` args)."""
        bsz, seq_len, d = ket_re.shape
        device = ket_re.device
        dtype = ket_re.dtype

        pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        freq_idx = torch.arange(0, d, 1, device=device, dtype=dtype).unsqueeze(0)
        angles = pos * (1.0 / (DEFAULT_ROPE_THETA_BASE ** (freq_idx / float(d))))
        cos_pos = torch.cos(angles).unsqueeze(0)
        sin_pos = torch.sin(angles).unsqueeze(0)

        rho_mps_re, rho_mps_im, mps_ket_re, mps_ket_im = self.mps_baton(
            ket_re, ket_im, rho_uni_re, rho_uni_im
        )
        rho_trans_re, rho_trans_im, trans_ket_re, trans_ket_im, in_ctx_copy_weights = (
            self._compute_in_context_and_streaming_transition_packet(
                ket_re, ket_im, trans_ema_re, trans_ema_im, rho_uni_re, rho_uni_im
            )
        )
        rho_ngram_re, rho_ngram_im = trace_normalize_density(
            DEFAULT_MPS_NGRAM_BLEND * rho_mps_re + (1.0 - DEFAULT_MPS_NGRAM_BLEND) * rho_trans_re,
            DEFAULT_MPS_NGRAM_BLEND * rho_mps_im + (1.0 - DEFAULT_MPS_NGRAM_BLEND) * rho_trans_im,
        )

        seq_pkts_re, seq_pkts_im, seq_ket_re, seq_ket_im, seq_attn_probs = (
            self._compute_exterior_born_sequence_packets(
                ket_re, ket_im, rho_mps_re, rho_mps_im, cos_pos, sin_pos
            )
        )
        rho_seq_re = seq_pkts_re.mean(dim=2)
        rho_seq_im = seq_pkts_im.mean(dim=2)
        rho_id_re, rho_id_im, id_attn_probs = self._compute_token_id_packet(
            ket_re, id_ket_re, id_ket_im
        )

        gate_kets_re, gate_kets_im = normalize_complex_ket(self.povm_gates_re, self.povm_gates_im)
        born_gates = []
        for g_idx in range(4):
            gk_re = gate_kets_re[g_idx].view(1, 1, d).expand(bsz, seq_len, d)
            gk_im = gate_kets_im[g_idx].view(1, 1, d).expand(bsz, seq_len, d)
            expect_val = born_quadratic_expectation(rho_ngram_re, rho_ngram_im, gk_re, gk_im)
            born_gates.append(expect_val + F.softplus(self.povm_gate_bias[g_idx]))
        gates = torch.stack(born_gates, dim=-1)
        gates = gates / (gates.sum(dim=-1, keepdim=True) + DEFAULT_TRACE_EPS)

        w_id = gates[..., 0:1].unsqueeze(-1)
        w_deep = gates[..., 1:2].unsqueeze(-1)
        w_ngram = gates[..., 2:3].unsqueeze(-1)
        w_uni = gates[..., 3:4].unsqueeze(-1)

        if self.config.use_schur_intersection:
            deep_re, deep_im = complex_schur_product(
                rho_seq_re, rho_seq_im, rho_ngram_re, rho_ngram_im
            )
            ng_uni_re, ng_uni_im = complex_schur_product(
                rho_ngram_re, rho_ngram_im, rho_uni_re, rho_uni_im
            )
        else:
            deep_re, deep_im = trace_normalize_density(
                0.5 * (rho_seq_re + rho_ngram_re), 0.5 * (rho_seq_im + rho_ngram_im)
            )
            ng_uni_re, ng_uni_im = trace_normalize_density(
                0.5 * (rho_ngram_re + rho_uni_re), 0.5 * (rho_ngram_im + rho_uni_im)
            )

        out_re = (
            w_id * rho_id_re
            + w_deep * deep_re
            + w_ngram * ng_uni_re
            + (w_uni + DEFAULT_VACUUM_EPS) * rho_uni_re
        )
        out_im = (
            w_id * rho_id_im
            + w_deep * deep_im
            + w_ngram * ng_uni_im
            + (w_uni + DEFAULT_VACUUM_EPS) * rho_uni_im
        )
        rho_out_re, rho_out_im = trace_normalize_density(out_re, out_im)

        packets_re = torch.cat(
            [
                rho_uni_re.unsqueeze(2),
                rho_mps_re.unsqueeze(2),
                rho_trans_re.unsqueeze(2),
                seq_pkts_re,
                rho_id_re.unsqueeze(2),
            ],
            dim=2,
        )
        packets_im = torch.cat(
            [
                rho_uni_im.unsqueeze(2),
                rho_mps_im.unsqueeze(2),
                rho_trans_im.unsqueeze(2),
                seq_pkts_im,
                rho_id_im.unsqueeze(2),
            ],
            dim=2,
        )

        comb_ket_re = ket_re + mps_ket_re + seq_ket_re + trans_ket_re
        comb_ket_im = ket_im + mps_ket_im + seq_ket_im + trans_ket_im
        next_ket_re = torch.einsum("btde,bte->btd", rho_out_re, comb_ket_re)
        next_ket_im = torch.einsum("btde,bte->btd", rho_out_im, comb_ket_im)
        next_ket_re, next_ket_im = normalize_complex_ket(
            ket_re + next_ket_re, ket_im + next_ket_im
        )

        return {
            "rho_out_re": rho_out_re,
            "rho_out_im": rho_out_im,
            "packets_re": packets_re,
            "packets_im": packets_im,
            "gates": gates,
            "next_ket_re": next_ket_re,
            "next_ket_im": next_ket_im,
            "in_ctx_copy_weights": in_ctx_copy_weights,
            "id_attn_probs": id_attn_probs,
            "seq_attn_probs": seq_attn_probs,
            "rho_deep_re": deep_re,
            "rho_deep_im": deep_im,
        }


# ==============================================================================
# Full 1-Layer / 2-Layer 4-Compartment Baton Language Model
# ==============================================================================
class FourCompartmentBatonHighwayLM(nn.Module):
    """1-Layer and 2-Layer 4-Compartment Quantum Statistical Mechanics Baton Language Model (`Wave 73`)."""

    def __init__(self, config: FourCompartmentBatonConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hilbert_dim
        v_sz = config.vocab_size

        self.token_ket_re = nn.Parameter(torch.randn(v_sz, d) / math.sqrt(float(d)))
        self.token_ket_im = nn.Parameter(torch.randn(v_sz, d) / math.sqrt(float(d)))
        self.readout_probe_re = nn.Parameter(torch.eye(d) + 0.03 * torch.randn(d, d))
        self.readout_probe_im = nn.Parameter(0.03 * torch.randn(d, d))
        self.layers = nn.ModuleList(
            [FourCompartmentBatonLayer(config) for _ in range(config.num_layers)]
        )

        self.register_buffer("online_uni_bath_re", torch.eye(d, dtype=torch.float32) / float(d))
        self.register_buffer("online_uni_bath_im", torch.zeros(d, d, dtype=torch.float32))
        self.register_buffer(
            "online_uni_token_probs", torch.full((v_sz,), 1.0 / float(v_sz), dtype=torch.float32)
        )
        self.register_buffer("online_trans_ema_re", torch.zeros(d, d, dtype=torch.float32))
        self.register_buffer("online_trans_ema_im", torch.zeros(d, d, dtype=torch.float32))
        self.register_buffer("online_ingested_tokens", torch.tensor(0, dtype=torch.long))
        self.register_buffer(
            "walsh_parity_pairs", build_walsh_hadamard_parity_pairs(d), persistent=False
        )

    @torch.no_grad()
    def reset_online_baths_to_cold_step0(self) -> None:
        """Reset all online thermal bath and transition buffers to exact cold Step-0 state (`<= 8` args)."""
        d = self.config.hilbert_dim
        v_sz = self.config.vocab_size
        self.online_uni_bath_re.copy_(
            torch.eye(d, device=self.online_uni_bath_re.device) / float(d)
        )
        self.online_uni_bath_im.zero_()
        self.online_uni_token_probs.fill_(1.0 / float(v_sz))
        self.online_trans_ema_re.zero_()
        self.online_trans_ema_im.zero_()
        self.online_ingested_tokens.zero_()

    @torch.no_grad()
    def ingest_online_token_batch(self, idx: Tensor) -> None:
        """Ingest a batch of token IDs `(B, T)` online into Compartment 1 & 2 buffers (`<= 8` args)."""
        eta_uni = self.config.online_bath_ema_decay
        eta_trans = DEFAULT_ONLINE_TRANS_EMA_DECAY

        raw_re = F.embedding(idx, self.token_ket_re)
        raw_im = F.embedding(idx, self.token_ket_im)
        ket_re, ket_im = normalize_complex_ket(raw_re, raw_im)

        batch_rho_re = torch.einsum("btd,bte->de", ket_re, ket_re) + torch.einsum(
            "btd,bte->de", ket_im, ket_im
        )
        batch_rho_im = torch.einsum("btd,bte->de", ket_im, ket_re) - torch.einsum(
            "btd,bte->de", ket_re, ket_im
        )
        batch_rho_re, batch_rho_im = trace_normalize_density(batch_rho_re, batch_rho_im)

        new_uni_re, new_uni_im = trace_normalize_density(
            (1.0 - eta_uni) * self.online_uni_bath_re + eta_uni * batch_rho_re,
            (1.0 - eta_uni) * self.online_uni_bath_im + eta_uni * batch_rho_im,
        )
        self.online_uni_bath_re.copy_(new_uni_re)
        self.online_uni_bath_im.copy_(new_uni_im)

        flat_idx = idx.reshape(-1)
        counts = torch.bincount(flat_idx, minlength=self.config.vocab_size).to(torch.float32)
        batch_probs = counts / counts.sum().clamp_min(1.0)
        self.online_uni_token_probs.mul_(1.0 - eta_uni).add_(eta_uni * batch_probs)
        self.online_uni_token_probs.div_(
            self.online_uni_token_probs.sum().clamp_min(DEFAULT_TRACE_EPS)
        )

        if idx.shape[1] > 1:
            curr_re, curr_im = ket_re[:, 1:, :], ket_im[:, 1:, :]
            prev_re, prev_im = ket_re[:, :-1, :], ket_im[:, :-1, :]
            denom = float(curr_re.shape[0] * curr_re.shape[1])
            trans_re = (
                torch.einsum("btd,bte->de", curr_re, prev_re)
                + torch.einsum("btd,bte->de", curr_im, prev_im)
            ) / max(1.0, denom)
            trans_im = (
                torch.einsum("btd,bte->de", curr_im, prev_re)
                - torch.einsum("btd,bte->de", curr_re, prev_im)
            ) / max(1.0, denom)
            self.online_trans_ema_re.mul_(1.0 - eta_trans).add_(eta_trans * trans_re)
            self.online_trans_ema_im.mul_(1.0 - eta_trans).add_(eta_trans * trans_im)

        self.online_ingested_tokens.add_(idx.numel())

    def _build_compartment1_unigram_bath_packet(
        self, ket_re: Tensor, ket_im: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Build `Packet 0` (`Compartment 1`: Online Unigram Thermal Bath + Causal Document Topic Bath) (`<= 8` args)."""
        bsz, seq_len, _ = ket_re.shape
        steps = torch.arange(1, seq_len + 1, device=ket_re.device, dtype=ket_re.dtype).view(
            1, seq_len, 1
        )
        doc_ket_re, doc_ket_im = normalize_complex_ket(
            torch.cumsum(ket_re, dim=1) / steps, torch.cumsum(ket_im, dim=1) / steps
        )
        doc_rho_re, doc_rho_im = pure_ket_to_density(doc_ket_re, doc_ket_im)

        uni_re = self.online_uni_bath_re.view(
            1, 1, self.config.hilbert_dim, self.config.hilbert_dim
        ).expand(bsz, seq_len, -1, -1)
        uni_im = self.online_uni_bath_im.view(
            1, 1, self.config.hilbert_dim, self.config.hilbert_dim
        ).expand(bsz, seq_len, -1, -1)
        return trace_normalize_density(
            (1.0 - DEFAULT_DOC_BATH_MIX) * uni_re + DEFAULT_DOC_BATH_MIX * doc_rho_re,
            (1.0 - DEFAULT_DOC_BATH_MIX) * uni_im + DEFAULT_DOC_BATH_MIX * doc_rho_im,
        )

    def forward_with_diagnostics(
        self, idx: Tensor, targets: Tensor | None = None, ingest_online: bool = False
    ) -> dict[str, Any]:
        """Run the 1L/2L 4-Compartment Baton highway and return logits, loss, and 8-packet states (`<= 8` args)."""
        ket_re, ket_im = normalize_complex_ket(
            F.embedding(idx, self.token_ket_re), F.embedding(idx, self.token_ket_im)
        )
        rho_uni_re, rho_uni_im = self._build_compartment1_unigram_bath_packet(ket_re, ket_im)
        id_ket_re, id_ket_im = encode_walsh_hadamard_token_id_kets(
            idx, self.walsh_parity_pairs, self.config.hilbert_dim
        )

        layer_out: dict[str, Tensor] = {}
        curr_ket_re, curr_ket_im = ket_re, ket_im
        prev_deep_re: Tensor | None = None
        prev_deep_im: Tensor | None = None

        for layer_idx, layer in enumerate(self.layers):
            layer_out = layer(
                curr_ket_re,
                curr_ket_im,
                rho_uni_re,
                rho_uni_im,
                self.online_trans_ema_re,
                self.online_trans_ema_im,
                id_ket_re,
                id_ket_im,
            )
            curr_ket_re = layer_out["next_ket_re"]
            curr_ket_im = layer_out["next_ket_im"]
            if layer_idx > 0 and prev_deep_re is not None and prev_deep_im is not None:
                if self.config.use_schur_intersection:
                    four_way_re, four_way_im = complex_schur_product(
                        prev_deep_re,
                        prev_deep_im,
                        layer_out["rho_deep_re"],
                        layer_out["rho_deep_im"],
                    )
                    rho_out_re, rho_out_im = trace_normalize_density(
                        0.65 * layer_out["rho_out_re"] + 0.35 * four_way_re,
                        0.65 * layer_out["rho_out_im"] + 0.35 * four_way_im,
                    )
                    layer_out["rho_out_re"] = rho_out_re
                    layer_out["rho_out_im"] = rho_out_im
            prev_deep_re = layer_out["rho_deep_re"]
            prev_deep_im = layer_out["rho_deep_im"]

        rho_out_re = layer_out["rho_out_re"]
        rho_out_im = layer_out["rho_out_im"]
        gates = layer_out["gates"]

        probe_re, probe_im = normalize_complex_ket(
            torch.matmul(curr_ket_re, self.readout_probe_re)
            - torch.matmul(curr_ket_im, self.readout_probe_im),
            torch.matmul(curr_ket_re, self.readout_probe_im)
            + torch.matmul(curr_ket_im, self.readout_probe_re),
        )
        state_vec_re = torch.einsum("btde,bte->btd", rho_out_re, probe_re) - torch.einsum(
            "btde,bte->btd", rho_out_im, probe_im
        )
        state_vec_im = torch.einsum("btde,bte->btd", rho_out_re, probe_im) + torch.einsum(
            "btde,bte->btd", rho_out_im, probe_re
        )
        readout_re, readout_im = normalize_complex_ket(
            state_vec_re + 0.50 * curr_ket_re, state_vec_im + 0.50 * curr_ket_im
        )

        vocab_re, vocab_im = normalize_complex_ket(self.token_ket_re, self.token_ket_im)
        born_sim_re = torch.matmul(readout_re, vocab_re.transpose(0, 1)) + torch.matmul(
            readout_im, vocab_im.transpose(0, 1)
        )
        born_logits = DEFAULT_BORN_READOUT_SCALE * born_sim_re

        w_uni_gate = gates[..., 3:4]
        uni_log_prior = torch.log(self.online_uni_token_probs.clamp_min(1e-9)).view(1, 1, -1)
        logits = born_logits + (DEFAULT_UNIGRAM_PRIOR_WEIGHT + 0.50 * w_uni_gate) * uni_log_prior

        in_ctx_weights = layer_out["in_ctx_copy_weights"]
        id_attn_probs = layer_out["id_attn_probs"]
        w_id_gate = gates[..., 0:1]
        combined_copy_mass = (
            DEFAULT_IN_CONTEXT_COPY_SCALE * in_ctx_weights
            + DEFAULT_TOKEN_ID_COPY_SCALE * w_id_gate * id_attn_probs
        )
        copy_index = idx.unsqueeze(1).expand(idx.shape[0], idx.shape[1], idx.shape[1])
        logits = logits.scatter_add(
            dim=-1, index=copy_index, src=combined_copy_mass.to(logits.dtype)
        )
        logits = DEFAULT_LOGIT_SOFTCAP * torch.tanh(logits / DEFAULT_LOGIT_SOFTCAP)

        loss: Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        if ingest_online and self.training:
            self.ingest_online_token_batch(idx)

        return {
            "logits": logits,
            "loss": loss,
            "packets_re": layer_out["packets_re"],
            "packets_im": layer_out["packets_im"],
            "rho_out_re": rho_out_re,
            "rho_out_im": rho_out_im,
            "gates": gates,
        }

    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Standard `(logits, loss)` forward entrypoint (`<= 8` args)."""
        out = self.forward_with_diagnostics(idx=idx, targets=targets, ingest_online=self.training)
        return out["logits"], out["loss"]


# ==============================================================================
# 5-Regime Quantum Statistical Mechanics Diagnostic Probe (`Wave 73`)
# ==============================================================================
def _probe_regime1_cold_bos_and_online_unigram_bath() -> dict[str, Any]:
    """Regime 1: Cold `<BOS>` Step-0 Unigram Thermal Bath & Online Ingestion + Trace Rescue (`<= 8` args)."""
    torch.manual_seed(1337)
    cfg = FourCompartmentBatonConfig(vocab_size=256, hilbert_dim=DEFAULT_HILBERT_DIM, num_layers=1)
    model = FourCompartmentBatonHighwayLM(cfg)
    model.eval()

    cold_tr = float(trace_of_density(model.online_uni_bath_re).item())
    cold_purity = float(
        purity_of_density(model.online_uni_bath_re, model.online_uni_bath_im).item()
    )

    skewed_batch = torch.full((4, 32), 42, dtype=torch.long)
    skewed_batch[:, ::4] = 7
    for _ in range(12):
        model.ingest_online_token_batch(skewed_batch)

    post_tr = float(trace_of_density(model.online_uni_bath_re).item())
    p_tok42 = float(model.online_uni_token_probs[42].item())
    p_tok99 = float(model.online_uni_token_probs[99].item())

    bos_idx = torch.zeros((2, 8), dtype=torch.long)
    out = model.forward_with_diagnostics(bos_idx, ingest_online=False)
    pkt_traces = trace_of_density(out["packets_re"])
    out_trace = trace_of_density(out["rho_out_re"])
    max_trace_err = float((pkt_traces - 1.0).abs().max().item())
    out_trace_err = float((out_trace - 1.0).abs().max().item())

    probs_bos = F.softmax(out["logits"][0, 0], dim=-1)
    bos_prefers_ingested = bool(probs_bos[42].item() > probs_bos[99].item())
    passed = (
        abs(cold_tr - 1.0) < 1e-5
        and abs(cold_purity - (1.0 / float(DEFAULT_HILBERT_DIM))) < 1e-5
        and abs(post_tr - 1.0) < 1e-5
        and max_trace_err < 1e-4
        and out_trace_err < 1e-4
        and p_tok42 > 10.0 * p_tok99
        and bos_prefers_ingested
    )
    return {
        "passed": passed,
        "cold_step0_bath_trace": round(cold_tr, 6),
        "cold_step0_bath_purity": round(cold_purity, 6),
        "post_ingest_bath_trace": round(post_tr, 6),
        "online_prob_tok42": round(p_tok42, 6),
        "online_prob_tok99": round(p_tok99, 6),
        "max_8packet_trace_err": round(max_trace_err, 7),
        "bos_prefers_ingested_hot_token": bos_prefers_ingested,
    }


def _probe_regime2_mps_stop_codon_destructive_cancellation() -> dict[str, Any]:
    """Regime 2: Vectorized `K=4, χ=8` MPS Destructive Phase Cancellation on Stop Codons (`TAA, TAG, TGA`) (`<= 8` args)."""
    torch.manual_seed(1337)
    cfg = FourCompartmentBatonConfig(vocab_size=4, hilbert_dim=DEFAULT_HILBERT_DIM, num_layers=1)
    mps = VectorizedMPSBatonModule(cfg)
    ket_emb = nn.Embedding(4, DEFAULT_HILBERT_DIM)
    readout = nn.Linear(DEFAULT_HILBERT_DIM, 4, bias=False)
    opt = torch.optim.AdamW(
        list(mps.parameters()) + list(ket_emb.parameters()) + list(readout.parameters()), lr=2.5e-2
    )

    train_x = torch.tensor([[0, 1, 3, 0], [2, 1, 3, 0], [1, 2, 3, 2], [0, 2, 3, 2]], dtype=torch.long)
    train_y = torch.tensor([1, 3, 2, 1], dtype=torch.long)
    rho_uni_re = (
        (torch.eye(DEFAULT_HILBERT_DIM) / float(DEFAULT_HILBERT_DIM))
        .view(1, 1, DEFAULT_HILBERT_DIM, DEFAULT_HILBERT_DIM)
        .expand(4, 4, -1, -1)
    )
    rho_uni_im = torch.zeros_like(rho_uni_re)

    for _ in range(90):
        opt.zero_grad()
        k_re, k_im = normalize_complex_ket(ket_emb(train_x), torch.zeros_like(ket_emb(train_x)))
        rho_mps_re, _, mean_re, _ = mps(k_re, k_im, rho_uni_re, rho_uni_im)
        feat = torch.einsum("bde,be->bd", rho_mps_re[:, -1], mean_re[:, -1])
        loss = F.cross_entropy(readout(feat), train_y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        k_re, k_im = normalize_complex_ket(ket_emb(train_x), torch.zeros_like(ket_emb(train_x)))
        rho_mps_re, _, mean_re, _ = mps(k_re, k_im, rho_uni_re, rho_uni_im)
        feat = torch.einsum("bde,be->bd", rho_mps_re[:, -1], mean_re[:, -1])
        probs = F.softmax(readout(feat), dim=-1)
        ta_forbidden_mass = float((probs[0, 0] + probs[0, 2]).item())
        tg_forbidden_mass = float(probs[2, 0].item())

    return {
        "passed": bool(ta_forbidden_mass < 0.01 and tg_forbidden_mass < 0.01),
        "prefix_TA_forbidden_TAA_TAG_prob": round(ta_forbidden_mass, 6),
        "prefix_TG_forbidden_TGA_prob": round(tg_forbidden_mass, 6),
    }


def _probe_regime3_in_context_lindblad_transition_induction() -> dict[str, Any]:
    """Regime 3: Online In-Context Lindblad Transition Matrix `M_t^{ctx}` 1-Layer Phrase Induction (`<= 8` args)."""
    torch.manual_seed(1337)
    cfg = FourCompartmentBatonConfig(vocab_size=512, hilbert_dim=DEFAULT_HILBERT_DIM, num_layers=1)
    model = FourCompartmentBatonHighwayLM(cfg)
    model.eval()

    seq = torch.tensor([[10, 101, 202, 30, 101, 202, 40, 101]], dtype=torch.long)
    with torch.no_grad():
        out = model.forward_with_diagnostics(seq, ingest_online=False)
        rho_trans_re = out["packets_re"][:, -1, PACKET_IDX_LINDBLAD_TRANS]
        rho_trans_im = out["packets_im"][:, -1, PACKET_IDX_LINDBLAD_TRANS]

        v202_re, v202_im = normalize_complex_ket(
            model.token_ket_re[202:203], model.token_ket_im[202:203]
        )
        v333_re, v333_im = normalize_complex_ket(
            model.token_ket_re[333:334], model.token_ket_im[333:334]
        )
        fid_target_202 = float(
            born_quadratic_expectation(rho_trans_re, rho_trans_im, v202_re, v202_im).item()
        )
        fid_distractor_333 = float(
            born_quadratic_expectation(rho_trans_re, rho_trans_im, v333_re, v333_im).item()
        )
        logits_last = out["logits"][0, -1]
        logit_202 = float(logits_last[202].item())
        logit_333 = float(logits_last[333].item())

    return {
        "passed": bool(fid_target_202 > 2.5 * fid_distractor_333 and logit_202 > logit_333),
        "packet2_born_fidelity_target_202": round(fid_target_202, 6),
        "packet2_born_fidelity_distractor_333": round(fid_distractor_333, 6),
        "logit_target_202": round(logit_202, 4),
        "logit_distractor_333": round(logit_333, 4),
    }


def _probe_regime4_coding_hairpin_schur_vs_additive() -> dict[str, Any]:
    """Regime 4: `CodingHairpin` Conjunctive Subspace Intersection (`Schur ⊙` vs. `Additive +`) (`<= 8` args)."""
    eye = torch.eye(DEFAULT_HILBERT_DIM, dtype=torch.float32)
    ket_g, ket_c, ket_a = eye[0], eye[1], eye[2]
    zero_im = torch.zeros_like(ket_g)

    rho_g_re, rho_g_im = pure_ket_to_density(ket_g, zero_im)
    rho_c_re, rho_c_im = pure_ket_to_density(ket_c, zero_im)
    rho_a_re, rho_a_im = pure_ket_to_density(ket_a, zero_im)

    rho_ngram_re, rho_ngram_im = trace_normalize_density(
        0.5 * (rho_g_re + rho_c_re), 0.5 * (rho_g_im + rho_c_im)
    )
    rho_seq_re, rho_seq_im = trace_normalize_density(
        0.5 * (rho_g_re + rho_a_re), 0.5 * (rho_g_im + rho_a_im)
    )
    rho_schur_re, rho_schur_im = complex_schur_product(
        rho_seq_re, rho_seq_im, rho_ngram_re, rho_ngram_im, eps=1e-7
    )
    rho_add_re, rho_add_im = trace_normalize_density(
        0.5 * (rho_seq_re + rho_ngram_re), 0.5 * (rho_seq_im + rho_ngram_im)
    )

    schur_p_g = float(born_quadratic_expectation(rho_schur_re, rho_schur_im, ket_g, zero_im).item())
    schur_leak_ca = float(
        (
            born_quadratic_expectation(rho_schur_re, rho_schur_im, ket_c, zero_im)
            + born_quadratic_expectation(rho_schur_re, rho_schur_im, ket_a, zero_im)
        ).item()
    )
    schur_purity = float(purity_of_density(rho_schur_re, rho_schur_im).item())

    add_p_g = float(born_quadratic_expectation(rho_add_re, rho_add_im, ket_g, zero_im).item())
    add_leak_ca = float(
        (
            born_quadratic_expectation(rho_add_re, rho_add_im, ket_c, zero_im)
            + born_quadratic_expectation(rho_add_re, rho_add_im, ket_a, zero_im)
        ).item()
    )
    add_purity = float(purity_of_density(rho_add_re, rho_add_im).item())

    passed = (
        schur_p_g > 0.999
        and schur_leak_ca < 1e-4
        and schur_purity > 0.998
        and abs(add_p_g - 0.50) < 1e-4
        and abs(add_leak_ca - 0.50) < 1e-4
    )
    return {
        "passed": passed,
        "schur_target_prob_G": round(schur_p_g, 6),
        "schur_forbidden_leak_CA": round(schur_leak_ca, 6),
        "schur_purity": round(schur_purity, 6),
        "additive_target_prob_G": round(add_p_g, 6),
        "additive_forbidden_leak_CA": round(add_leak_ca, 6),
        "additive_purity": round(add_purity, 6),
    }


def _probe_regime5_zero_shot_walsh_hadamard_token_id_reversal() -> dict[str, Any]:
    """Regime 5: Zero-Shot Unseen Token-ID (`#49999, #48888, #47777`) Copying & Palindromic Reversal (`Packet 7`) (`<= 8` args)."""
    parity_pairs = build_walsh_hadamard_parity_pairs(DEFAULT_HILBERT_DIM)
    unseen_ids = torch.tensor([[49999, 48888, 47777]], dtype=torch.long)
    distractor_ids = torch.tensor([[12345, 23456, 34567]], dtype=torch.long)

    unseen_re, unseen_im = encode_walsh_hadamard_token_id_kets(
        unseen_ids, parity_pairs, DEFAULT_HILBERT_DIM
    )
    dist_re, _ = encode_walsh_hadamard_token_id_kets(
        distractor_ids, parity_pairs, DEFAULT_HILBERT_DIM
    )

    rev_perm = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]])
    pure_re, pure_im = pure_ket_to_density(unseen_re, unseen_im)
    routed_rho_re = torch.einsum("bts,bsde->btde", rev_perm, pure_re)
    routed_rho_im = torch.einsum("bts,bsde->btde", rev_perm, pure_im)

    expected_rev_re = unseen_re.flip(dims=[1])
    expected_rev_im = unseen_im.flip(dims=[1])

    self_fidelity = float(
        born_quadratic_expectation(routed_rho_re, routed_rho_im, expected_rev_re, expected_rev_im)
        .mean()
        .item()
    )
    cross_fidelity = float(
        born_quadratic_expectation(routed_rho_re, routed_rho_im, dist_re, expected_rev_im)
        .mean()
        .item()
    )
    return {
        "passed": bool(abs(self_fidelity - 1.0) < 1e-5 and cross_fidelity < 0.15),
        "zero_shot_reversed_self_born_fidelity": round(self_fidelity, 6),
        "zero_shot_cross_distractor_born_fidelity": round(cross_fidelity, 6),
    }


def run_five_regime_baton_probe() -> dict[str, Any]:
    """Run all 5 quantum statistical mechanics regime probes and return a structured report (`<= 8` args)."""
    r1 = _probe_regime1_cold_bos_and_online_unigram_bath()
    r2 = _probe_regime2_mps_stop_codon_destructive_cancellation()
    r3 = _probe_regime3_in_context_lindblad_transition_induction()
    r4 = _probe_regime4_coding_hairpin_schur_vs_additive()
    r5 = _probe_regime5_zero_shot_walsh_hadamard_token_id_reversal()
    return {
        "overall_passed": bool(
            r1["passed"] and r2["passed"] and r3["passed"] and r4["passed"] and r5["passed"]
        ),
        "regime1_cold_bos_and_online_unigram_bath": r1,
        "regime2_mps_stop_codon_destructive_cancellation": r2,
        "regime3_in_context_lindblad_transition_induction": r3,
        "regime4_coding_hairpin_schur_vs_additive": r4,
        "regime5_zero_shot_walsh_hadamard_token_id_reversal": r5,
    }


# ==============================================================================
# FineWeb Shard-0 Data Loading & Wave 73 Benchmark Runner
# ==============================================================================
def load_fineweb_tokens(shard_path: Path, max_tokens: int = 2_000_000) -> Tensor:
    """Load GPT-2 BPE uint16 tokens from an official FineWeb `.bin` shard (`<= 8` args)."""
    raw = np.memmap(shard_path, dtype=np.int32, mode="r", shape=(FINEWEB_HEADER_INT32_COUNT,))
    if int(raw[0]) != FINEWEB_MAGIC_NUMBER:
        raise ValueError(f"Invalid FineWeb shard magic {raw[0]} in {shard_path}")
    read_count = min(int(raw[2]), max_tokens)
    tokens_u16 = np.memmap(
        shard_path,
        dtype=np.uint16,
        mode="r",
        offset=FINEWEB_HEADER_INT32_COUNT * 4,
        shape=(read_count,),
    )
    return torch.from_numpy(np.array(tokens_u16, dtype=np.int64))


def sample_fineweb_batch(
    tokens: Tensor, batch_size: int, seq_len: int, step_idx: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Deterministic strided batch sampler over a loaded FineWeb token buffer (`<= 8` args)."""
    span = batch_size * (seq_len + 1)
    max_offset = max(1, tokens.numel() - span - 1)
    start = (step_idx * batch_size * seq_len) % max_offset
    chunk = tokens[start : start + span].view(batch_size, seq_len + 1).to(device)
    return chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()


@torch.no_grad()
def evaluate_wave73_model(
    model: FourCompartmentBatonHighwayLM,
    val_tokens: Tensor,
    spec: Wave73TrainSpec,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate validation CE, perplexity, top-1 accuracy, and compartment gate weights (`<= 8` args)."""
    model.eval()
    losses: list[float] = []
    accs: list[float] = []
    gate_sums = torch.zeros(4, dtype=torch.float64)
    purity_vals: list[float] = []

    for b_idx in range(spec.eval_batches):
        x_val, y_val = sample_fineweb_batch(
            val_tokens, spec.batch_size, spec.seq_len, b_idx, device
        )
        out = model.forward_with_diagnostics(x_val, y_val, ingest_online=False)
        losses.append(float(out["loss"].item()))
        preds = out["logits"].argmax(dim=-1)
        accs.append(float((preds == y_val).to(torch.float32).mean().item()))
        gate_sums += out["gates"].mean(dim=(0, 1)).detach().cpu().to(torch.float64)
        purity_vals.append(
            float(purity_of_density(out["rho_out_re"], out["rho_out_im"]).mean().item())
        )

    mean_ce = float(np.mean(losses))
    mean_gates = (gate_sums / float(max(1, spec.eval_batches))).tolist()
    model.train()
    return {
        "val_ce": round(mean_ce, 4),
        "val_ppl": round(float(math.exp(min(20.0, mean_ce))), 2),
        "top1_acc": round(float(np.mean(accs)), 4),
        "mean_out_purity": round(float(np.mean(purity_vals)), 4),
        "gate_w_id": round(float(mean_gates[0]), 4),
        "gate_w_deep": round(float(mean_gates[1]), 4),
        "gate_w_ngram": round(float(mean_gates[2]), 4),
        "gate_w_uni": round(float(mean_gates[3]), 4),
    }


def run_wave73_variant(
    spec: Wave73TrainSpec,
    train_tokens: Tensor,
    val_tokens: Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Train and evaluate a single Wave 73 variant with `StandardizedExperimentLogger` (`<= 8` args)."""
    torch.manual_seed(spec.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.seed)

    cfg = build_wave73_config_for_variant(spec.variant_id, vocab_size=DEFAULT_VOCAB_SIZE)
    model = FourCompartmentBatonHighwayLM(cfg).to(device)
    model.reset_online_baths_to_cold_step0()
    param_count = sum(p.numel() for p in model.parameters())

    intent = ExperimentIntentSpec(
        run_key=spec.variant_id.lower(),
        wave_id="W73",
        plan_id="EXPERIMENT_PLAN_093",
        task_domain="fineweb_speedrun",
        hypothesis_intent=(
            f"Evaluate {spec.variant_id} (layers={cfg.num_layers}, schur={cfg.use_schur_intersection}, "
            f"two_hop={cfg.use_two_hop_triangle}) with 100% online unigram thermal bath and dual MPS/Lindblad "
            f"n-gram ingestion on FineWeb Shard 0."
        ),
        architecture_summary=(
            f"FourCompartmentBatonHighwayLM(d={cfg.hilbert_dim}, H=8, L={cfg.num_layers}, "
            f"K={cfg.mps_window_k}, chi={cfg.mps_bond_dim}, schur={cfg.use_schur_intersection})"
        ),
        hardware_budget=str(device),
        batch_size=spec.batch_size,
        seq_len=spec.seq_len,
        total_steps=spec.total_steps,
        target_tokens_M=round((spec.total_steps * spec.batch_size * spec.seq_len) / 1e6, 4),
    )
    logger = StandardizedExperimentLogger(intent)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=0.01)

    step0_eval = evaluate_wave73_model(model, val_tokens, spec, device)
    history: list[dict[str, Any]] = [{"step": 0, **step0_eval}]
    t0 = time.perf_counter()

    for step in range(1, spec.total_steps + 1):
        x_batch, y_batch = sample_fineweb_batch(
            train_tokens, spec.batch_size, spec.seq_len, step - 1, device
        )
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x_batch, y_batch)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_ce = float(loss.item())
        elapsed_so_far = max(1e-6, time.perf_counter() - t0)
        tok_per_s = (step * spec.batch_size * spec.seq_len) / elapsed_so_far
        logger.log_step(step=step, train_ce=train_ce, tok_per_s=tok_per_s, lr_mult=1.0)

        if step % spec.eval_interval == 0 or step == spec.total_steps:
            eval_metrics = evaluate_wave73_model(model, val_tokens, spec, device)
            logger.log_eval(step=step, metrics=eval_metrics)
            history.append({"step": step, "train_ce": round(train_ce, 4), **eval_metrics})

    elapsed = time.perf_counter() - t0
    logger.finalize()
    final_eval = history[-1]
    return {
        "variant_id": spec.variant_id,
        "config": asdict(cfg),
        "param_count": param_count,
        "online_ingested_tokens": int(model.online_ingested_tokens.item()),
        "wall_time_sec": round(elapsed, 2),
        "tokens_per_sec": round(
            (spec.total_steps * spec.batch_size * spec.seq_len) / max(1e-6, elapsed), 1
        ),
        "step0_val_ce": step0_eval["val_ce"],
        "final_val_ce": final_eval["val_ce"],
        "final_val_ppl": final_eval["val_ppl"],
        "final_top1_acc": final_eval["top1_acc"],
        "final_out_purity": final_eval["mean_out_purity"],
        "final_gates": {
            "w_id": final_eval["gate_w_id"],
            "w_deep": final_eval["gate_w_deep"],
            "w_ngram": final_eval["gate_w_ngram"],
            "w_uni": final_eval["gate_w_uni"],
        },
        "eval_history": history,
    }
