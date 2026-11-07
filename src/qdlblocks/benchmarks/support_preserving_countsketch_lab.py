r"""Support-Preserving CountSketch & Meaning-vs-Identity Wavepacket Laboratory (`V = 50,304`).

Investigates and resolves the fundamental geometric tension on the Product-of-Spheres manifold
$(\sqrt{d_h}\,\mathbb{S}^{d_h-1})^{M_{\text{pkt}}}$ between:
1. **Semantic Sharing ("Sharing Meaning")**: Co-occurring or semantically related tokens need non-zero inner products
   so attention heads and Mach-Zehnder interferometers can route and generalize across related words.
2. **Unique Token Discrimination ("Identity Preservation")**: Distinct tokens—especially rare/unseen tail tokens in a
   `50,304`-token vocabulary and near-synonym peers—must retain an orthogonal identity component so readout projections
   never collapse or confuse them.

Implements three support-preserving millisecond GPU initializers:
- `I5_SUPPORT_GATED_COUNTSKETCH`: Active-only column centering + Gram-Schmidt orthogonal identity complement $\hat{z}_v \perp \hat{s}_v$
  blended via an empirical-Bayes count gate $\alpha_v = \alpha_{\max}\sqrt{c_v / (c_v + k_0)}$.
- `I6_SPLIT_MEANING_ID_WAVEPACKET`: Partitioned wavepacket bundle where packets `0 .. M/2 - 1` carry support-gated
  bigram semantics and packets `M/2 .. M - 1` carry `100%` orthogonal random-sphere identity barcodes.
- `I7_WHITENED_SUPPORT_COUNTSKETCH`: Fast GPU $d \times d$ covariance ZCA whitening ($C^{-1/2}$) of active CountSketch
  PPMI rows to eliminate hub-token anisotropy before support-gated orthogonal blending.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.impure_density_mirror_block import (
    DEFAULT_TIKHONOV_GRAM_LAMBDA,
    TRACE_NORM_EPS,
)
from ..layers.quantum_statmech_channels import (
    QuantumMomentumScheduleMode,
    StiefelMomentumMode,
    compute_riemannian_momentum_beta,
)
from ..layers.wavepacket_statmech_block import apply_causal_wavepacket_dispersion
from ..models.modular_quantum_statmech_gpt import ModularRiemannianStiefelKrausMuon
from ..optim.vocab_manifold_muon import (
    BatchSubspaceTikhonovGramMuon,
    project_rows_to_rms_sphere_,
)
from .micro_reversal_readout_lab import (
    DEFAULT_FINEWEB_DIR,
    DEFAULT_HASH_PRIME_1,
    DEFAULT_HASH_PRIME_2,
    DEFAULT_MICRO_BATCH_SIZE,
    DEFAULT_MICRO_EMBED_DIM,
    DEFAULT_MICRO_MLP_HIDDEN,
    DEFAULT_MICRO_NUM_HEADS,
    DEFAULT_MICRO_NUM_LAYERS,
    DEFAULT_MICRO_PKT_COUNT,
    DEFAULT_MICRO_PKT_DIM,
    DEFAULT_MICRO_SCALAR_LR,
    DEFAULT_MICRO_STIEFEL_LR,
    DEFAULT_MICRO_UNITARY_LR,
    DEFAULT_MICRO_WTE_LR,
    DEFAULT_ORTHO_BALANCE_SEED,
    DEFAULT_SVD_ANCHOR_COLS,
    FINEWEB_HEADER_INT32,
    FineWebSpanReversalDataset,
    MicroBackboneMode,
    MicroInitMode,
    MicroReadoutMode,
    MicroReversalConfig,
    MicroReversalGPT,
    compute_gpu_countsketch_bigram_sphere_embeddings,
    compute_gpu_svd_bigram_sphere_embeddings,
    compute_gpu_unigram_log_prior_and_radial,
    evaluate_micro_reversal_model,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
DEFAULT_FULL_FINEWEB_VOCAB_SIZE: int = 50304
DEFAULT_SUPPORT_BAYES_K0: float = 32.0
DEFAULT_SUPPORT_ALPHA_MAX: float = 0.60
DEFAULT_SPLIT_SEMANTIC_ALPHA_MAX: float = 0.85
DEFAULT_WHITENING_REG_EPS: float = 1e-2
DEFAULT_GEOM_SAMPLE_SIZE: int = 512

DEFAULT_SHARD_STREAM_CHUNK_TOKENS: int = 25_000_000
DEFAULT_BILATERAL_EXACT_ANCHORS: int = 768
DEFAULT_BILATERAL_TAIL_BUCKETS: int = 256
DEFAULT_BILATERAL_TOTAL_COLS: int = DEFAULT_BILATERAL_EXACT_ANCHORS + DEFAULT_BILATERAL_TAIL_BUCKETS
DEFAULT_MULTILAG_OFFSETS: tuple[int, ...] = (2, 4, 8)
DEFAULT_MULTILAG_WEIGHTS: tuple[float, ...] = (0.55, 0.30, 0.15)
DEFAULT_STEP0_RIDGE_LAMBDA: float = 0.08
DEFAULT_STEP0_COLD_OBS_SCALE: float = 0.12
_CACHE_DIR: Path = DEFAULT_FINEWEB_DIR.parent


class SupportSketchMode(str, Enum):
    """Strategy enum for comparing baseline vs. support-preserving GPU initializers (Rule 3)."""

    I2_UNIGRAM_PRIOR_RANDOM_SPHERE = "i2_unigram_prior_random_sphere"
    I3_RAW_COUNTSKETCH_SPHERE = "i3_raw_countsketch_sphere"
    I4_LOWRANK_SVD_SPHERE = "i4_lowrank_svd_sphere"
    I5_SUPPORT_GATED_COUNTSKETCH = "i5_support_gated_countsketch"
    I6_SPLIT_MEANING_ID_WAVEPACKET = "i6_split_meaning_id_wavepacket"
    I7_WHITENED_SUPPORT_COUNTSKETCH = "i7_whitened_support_countsketch"
    I8_SPLIT_MEANING_ID_ADAPTIVE_PKT_GATE = "i8_split_meaning_id_adaptive_pkt_gate"
    I9_COMPLEX_WAVEFUNCTION_4x2x32 = "i9_complex_wavefunction_4x2x32"
    I10_COMPLEX_WAVEFUNCTION_4x2x64 = "i10_complex_wavefunction_4x2x64"
    I11_COMPLEX_WAVEFUNCTION_8x2x32_ID2 = "i11_complex_wavefunction_8x2x32_id2"
    I12_COMPLEX_WAVEFUNCTION_8x2x32_ID1 = "i12_complex_wavefunction_8x2x32_id1"
    I13_COMPLEX_WAVEFUNCTION_16x2x16_ID4 = "i13_complex_wavefunction_16x2x16_id4"
    I14_COMPLEX_WAVEFUNCTION_16x2x16_ID2 = "i14_complex_wavefunction_16x2x16_id2"
    I15_SHARD1_100M_MULTILAG_SKETCH = "i15_shard1_100m_multilag_sketch"
    I16_SHARD1_100M_STEP0_CALIBRATED = "i16_shard1_100m_step0_calibrated"
    I17_SHARDS1_TO_9_900M_STEP0_CALIBRATED = "i17_shards1_to_9_900m_step0_calibrated"


# Maps each 4D complex wavefunction mode to (m_complex, d_h, sem_complex_packets)
_COMPLEX_WF_MODE_LAYOUTS: dict[SupportSketchMode, tuple[int, int, int]] = {
    SupportSketchMode.I9_COMPLEX_WAVEFUNCTION_4x2x32: (4, 32, 3),
    SupportSketchMode.I10_COMPLEX_WAVEFUNCTION_4x2x64: (4, 64, 3),
    SupportSketchMode.I11_COMPLEX_WAVEFUNCTION_8x2x32_ID2: (8, 32, 6),
    SupportSketchMode.I12_COMPLEX_WAVEFUNCTION_8x2x32_ID1: (8, 32, 7),
    SupportSketchMode.I13_COMPLEX_WAVEFUNCTION_16x2x16_ID4: (16, 16, 12),
    SupportSketchMode.I14_COMPLEX_WAVEFUNCTION_16x2x16_ID2: (16, 16, 14),
    SupportSketchMode.I15_SHARD1_100M_MULTILAG_SKETCH: (8, 32, 7),
    SupportSketchMode.I16_SHARD1_100M_STEP0_CALIBRATED: (8, 32, 7),
    SupportSketchMode.I17_SHARDS1_TO_9_900M_STEP0_CALIBRATED: (8, 32, 7),
}

_MULTISHARD_MODES: dict[SupportSketchMode, tuple[int, bool]] = {
    SupportSketchMode.I15_SHARD1_100M_MULTILAG_SKETCH: (1, False),
    SupportSketchMode.I16_SHARD1_100M_STEP0_CALIBRATED: (1, True),
    SupportSketchMode.I17_SHARDS1_TO_9_900M_STEP0_CALIBRATED: (9, True),
}


@dataclass(frozen=True)
class SupportSketchSpec:
    """Hyperparameters governing support-preserving CountSketch initialization (Rule 2)."""

    mode: SupportSketchMode = SupportSketchMode.I5_SUPPORT_GATED_COUNTSKETCH
    vocab_size: int = DEFAULT_FULL_FINEWEB_VOCAB_SIZE
    pkt_count: int = DEFAULT_MICRO_PKT_COUNT
    pkt_dim: int = DEFAULT_MICRO_PKT_DIM
    bayes_k0: float = DEFAULT_SUPPORT_BAYES_K0
    alpha_max: float = DEFAULT_SUPPORT_ALPHA_MAX
    split_sem_alpha_max: float = DEFAULT_SPLIT_SEMANTIC_ALPHA_MAX
    seed: int = 1337


@torch.no_grad()
def _compute_raw_active_ppmi_sketch(
    train_tokens: torch.Tensor,
    vocab_size: int,
    d_emb: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute raw 2-hash signed PPMI sketch `(V, d_emb)`, token counts `(V,)`, and active row mask `(V, 1)`."""
    device = train_tokens.device
    u_tok = train_tokens[:-1]
    v_tok = train_tokens[1:]
    n_pairs = float(u_tok.numel())

    v_ids = torch.arange(vocab_size, device=device, dtype=torch.int64)
    col_hash_fwd = ((v_ids * DEFAULT_HASH_PRIME_1 + 17) % (d_emb // 2)).long()
    col_hash_rev = (d_emb // 2) + ((v_ids * DEFAULT_HASH_PRIME_2 + 31) % (d_emb // 2)).long()
    sign_fwd = torch.where(((v_ids * DEFAULT_HASH_PRIME_2) & 1) == 0, 1.0, -1.0)
    sign_rev = torch.where(((v_ids * DEFAULT_HASH_PRIME_1) & 1) == 0, 1.0, -1.0)

    sketch_pos = torch.zeros((vocab_size, d_emb), device=device, dtype=torch.float32)
    flat_fwd_idx = u_tok * d_emb + col_hash_fwd[v_tok]
    flat_rev_idx = v_tok * d_emb + col_hash_rev[u_tok]
    ones_pairs = torch.ones_like(u_tok, dtype=torch.float32)
    sketch_pos.view(-1).scatter_add_(0, flat_fwd_idx, ones_pairs)
    sketch_pos.view(-1).scatter_add_(0, flat_rev_idx, ones_pairs)

    row_counts_2d = sketch_pos.sum(dim=1, keepdim=True)
    tok_counts = 0.5 * row_counts_2d.squeeze(1)
    col_counts = sketch_pos.sum(dim=0, keepdim=True).clamp_min(1.0)
    expected = (row_counts_2d.clamp_min(1.0) * col_counts) / (2.0 * n_pairs + 1.0)
    ppmi_sketch = F.relu(torch.log((sketch_pos + 0.5) / (expected + 0.5)))

    col_signs = torch.ones(d_emb, device=device, dtype=torch.float32)
    col_signs[: d_emb // 2] = sign_fwd[: d_emb // 2] if vocab_size >= d_emb // 2 else 1.0
    col_signs[d_emb // 2 :] = sign_rev[: d_emb // 2] if vocab_size >= d_emb // 2 else -1.0
    signed_sketch = ppmi_sketch * col_signs.unsqueeze(0)

    row_energy = signed_sketch.square().sum(dim=1, keepdim=True)
    active_mask = (row_energy > 1e-8).float()
    n_active = active_mask.sum().clamp_min(1.0)
    active_mean = (signed_sketch * active_mask).sum(dim=0, keepdim=True) / n_active
    centered_active = (signed_sketch - active_mean) * active_mask
    return centered_active, tok_counts, active_mask


@torch.no_grad()
def _orthogonal_gram_schmidt_blend(
    sem_3d: torch.Tensor,
    active_mask_1d: torch.Tensor,
    alpha_per_pkt: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Blend unit-RMS semantic packets `sem_3d` with Gram-Schmidt-orthogonal identity packets `z_ortho` on the sphere."""
    vocab_size, pkt_count, pkt_dim = sem_3d.shape
    device = sem_3d.device

    sem_unit = sem_3d.clone()
    project_rows_to_rms_sphere_(sem_unit)

    gen = torch.Generator(device=device).manual_seed(int(seed) + 911)
    z_raw = torch.randn((vocab_size, pkt_count, pkt_dim), generator=gen, device=device, dtype=torch.float32)
    project_rows_to_rms_sphere_(z_raw)

    # Project out component parallel to sem_unit within each packet so <sem_unit, z_ortho> == 0
    dot_sz = (z_raw * sem_unit).mean(dim=-1, keepdim=True)
    z_perp = z_raw - dot_sz * sem_unit
    project_rows_to_rms_sphere_(z_perp)

    eff_alpha = (alpha_per_pkt * active_mask_1d.view(vocab_size, 1, 1)).clamp(0.0, 0.98)
    id_weight = torch.sqrt((1.0 - eff_alpha.square()).clamp_min(1e-6))
    blended = eff_alpha * sem_unit + id_weight * z_perp
    project_rows_to_rms_sphere_(blended)
    return blended


@torch.no_grad()
def project_complex_wavepackets_to_rms_sphere_(
    psi_4d: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    r"""Project a 4D complex wavefunction tensor `(..., M_c, 2, d_h)` onto the complex sphere $\sqrt{2 d_h}\,\mathbb{S}^{2 d_h - 1}$.

    Normalizes over the joint `(2, d_h)` complex Hilbert dimensions so that for every complex wavepacket
    $\psi_{v,m} = \psi_{v,m}^{\text{re}} + i\,\psi_{v,m}^{\text{im}} \in \mathbb{C}^{d_h}$:
    $$\|\psi_{v,m}\|_{\mathbb{C}}^2 = \|\psi_{v,m}^{\text{re}}\|_2^2 + \|\psi_{v,m}^{\text{im}}\|_2^2 = 2 d_h$$
    which is strictly invariant under $U(1)$ phase rotations $e^{i\phi}\psi_{v,m}$ and $U(d_h)$ unitary evolution.
    """
    target_radius = math.sqrt(float(2 * psi_4d.size(-1)))
    c_norm = psi_4d.float().square().sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(eps)
    scale = (target_radius / c_norm).to(psi_4d.dtype)
    psi_4d.mul_(scale)
    return psi_4d


@torch.no_grad()
def _hermitian_gram_schmidt_complex_blend(
    sem_4d: torch.Tensor,
    active_mask_1d: torch.Tensor,
    alpha_per_cpkt: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    r"""Blend 4D complex semantic wavepackets `(V, M_c, 2, d_h)` with Hermitian-orthogonal identity wavepackets $\langle s \mid z^\perp \rangle_{\mathbb{C}} = 0 + 0i$."""
    vocab_size, m_complex, two_dim, d_h = sem_4d.shape
    assert two_dim == 2
    device = sem_4d.device

    sem_unit = sem_4d.clone()
    project_complex_wavepackets_to_rms_sphere_(sem_unit)

    gen = torch.Generator(device=device).manual_seed(int(seed) + 1911)
    z_raw = torch.randn((vocab_size, m_complex, 2, d_h), generator=gen, device=device, dtype=torch.float32)
    project_complex_wavepackets_to_rms_sphere_(z_raw)

    s_re, s_im = sem_unit[:, :, 0, :], sem_unit[:, :, 1, :]
    z_re, z_im = z_raw[:, :, 0, :], z_raw[:, :, 1, :]
    inv_2dh = 1.0 / float(2 * d_h)

    # Full complex Hermitian inner product <s | z>_C / (2 d_h) = c_re + i c_im
    c_re = ((s_re * z_re).sum(dim=-1, keepdim=True) + (s_im * z_im).sum(dim=-1, keepdim=True)) * inv_2dh
    c_im = ((s_re * z_im).sum(dim=-1, keepdim=True) - (s_im * z_re).sum(dim=-1, keepdim=True)) * inv_2dh

    # Subtract (c_re + i c_im) * (s_re + i s_im) so both Re<s|z_perp> == 0 and Im<s|z_perp> == 0
    zp_re = z_re - (c_re * s_re - c_im * s_im)
    zp_im = z_im - (c_re * s_im + c_im * s_re)
    z_perp = torch.stack([zp_re, zp_im], dim=2)
    project_complex_wavepackets_to_rms_sphere_(z_perp)

    eff_alpha = (alpha_per_cpkt * active_mask_1d.view(vocab_size, 1, 1, 1)).clamp(0.0, 0.98)
    id_weight = torch.sqrt((1.0 - eff_alpha.square()).clamp_min(1e-6))
    blended_4d = eff_alpha * sem_unit + id_weight * z_perp
    project_complex_wavepackets_to_rms_sphere_(blended_4d)
    return blended_4d


@torch.no_grad()
def compute_complex_wavefunction_split_sphere(
    train_tokens: torch.Tensor,
    spec: SupportSketchSpec,
    sem_complex_packets: int | None = None,
    sem_alpha_cap: float = 0.96,
) -> torch.Tensor:
    r"""`I9`..`I14`: Compute 4D Complex Wavefunction Product-Sphere Table `(V, M_c, 2, d_h)` ($M_c \in \{4, 8, 16\}$).

    - Maps forward bigram PPMI sketch columns (`v -> x_{t+1}`) to $\operatorname{Re}(\psi_{v,m})$ and reverse bigram
      PPMI sketch columns (`x_{t-1} -> v`) to $\operatorname{Im}(\psi_{v,m})$.
    - Balances across the $M_c$ complex wavepackets via a Haar-random complex unitary $U_M \in U(M_c)$.
    - Splits the $M_c$ complex wavepackets into $M_{\text{sem}}$ Semantic Complex Wavefunctions + $(M_c - M_{\text{sem}})$
      Pure Identity Complex Wavefunctions with Hermitian Gram-Schmidt orthogonalization $\langle s \mid z^\perp \rangle_{\mathbb{C}} = 0 + 0i$.
    """
    vocab_size = int(spec.vocab_size)
    mode = SupportSketchMode(spec.mode)
    if mode in _COMPLEX_WF_MODE_LAYOUTS:
        m_complex, d_h, default_n_sem = _COMPLEX_WF_MODE_LAYOUTS[mode]
    else:
        m_complex = max(1, int(spec.pkt_count) // 2)
        d_h = int(spec.pkt_dim)
        default_n_sem = max(1, (3 * m_complex) // 4)
    n_sem_c = default_n_sem if sem_complex_packets is None else max(1, min(m_complex - 1, int(sem_complex_packets)))
    d_emb = 2 * m_complex * d_h
    device = train_tokens.device

    centered_active, tok_counts, active_mask = _compute_raw_active_ppmi_sketch(train_tokens, vocab_size, d_emb)
    half_d = m_complex * d_h
    fwd_3d = centered_active[:, :half_d].view(vocab_size, d_h, m_complex)
    rev_3d = centered_active[:, half_d:].view(vocab_size, d_h, m_complex)

    # Haar-random complex unitary U_M = U_re + i U_im in U(M_c) via complex QR
    gen = torch.Generator(device=device).manual_seed(DEFAULT_ORTHO_BALANCE_SEED + 2 * m_complex + d_h)
    a_re = torch.randn((m_complex, m_complex), generator=gen, device=device, dtype=torch.float32)
    a_im = torch.randn((m_complex, m_complex), generator=gen, device=device, dtype=torch.float32)
    q_c, r_c = torch.linalg.qr(torch.complex(a_re, a_im))
    d_diag = torch.diagonal(r_c)
    phase_diag = d_diag / d_diag.abs().clamp_min(1e-8)
    q_c = q_c * phase_diag.unsqueeze(0)
    u_re, u_im = q_c.real.contiguous(), q_c.imag.contiguous()

    sem_re = (torch.matmul(fwd_3d, u_re) - torch.matmul(rev_3d, u_im)).transpose(1, 2).contiguous()
    sem_im = (torch.matmul(fwd_3d, u_im) + torch.matmul(rev_3d, u_re)).transpose(1, 2).contiguous()
    sem_4d = torch.stack([sem_re, sem_im], dim=2)

    alpha_sem = float(sem_alpha_cap) * torch.sqrt(tok_counts / (tok_counts + float(spec.bayes_k0)))
    alpha_per_cpkt = torch.zeros((vocab_size, m_complex, 1, 1), device=device, dtype=torch.float32)
    alpha_per_cpkt[:, :n_sem_c, 0, 0] = alpha_sem.unsqueeze(1)
    return _hermitian_gram_schmidt_complex_blend(sem_4d, active_mask.squeeze(1), alpha_per_cpkt, spec.seed)


@torch.no_grad()
def compute_support_gated_countsketch_sphere(
    train_tokens: torch.Tensor,
    spec: SupportSketchSpec,
) -> torch.Tensor:
    r"""`I5`: Support-Preserving Empirical-Bayes CountSketch (`~5.8 ms` at $V=50,304$).

    Centers only over active rows ($r_v > 0$), rotates across packets via $Q_M \in O(M_{\text{pkt}})$, and blends
    each token's semantic direction $\hat{s}_v$ with its Gram-Schmidt-orthogonal random identity direction $\hat{z}_v$:
    $$u_{v,m} = \alpha_v \hat{s}_{v,m} + \sqrt{1 - \alpha_v^2}\,\hat{z}_{v,m}, \qquad \alpha_v = \alpha_{\max}\sqrt{\frac{c_v}{c_v + k_0}}\,\mathbb{I}[r_v > 0]$$
    """
    vocab_size = int(spec.vocab_size)
    pkt_count = int(spec.pkt_count)
    pkt_dim = int(spec.pkt_dim)
    d_emb = pkt_count * pkt_dim
    device = train_tokens.device

    centered_active, tok_counts, active_mask = _compute_raw_active_ppmi_sketch(train_tokens, vocab_size, d_emb)
    gen = torch.Generator(device=device).manual_seed(DEFAULT_ORTHO_BALANCE_SEED + pkt_count)
    q_m, r_m = torch.linalg.qr(torch.randn((pkt_count, pkt_count), generator=gen, device=device, dtype=torch.float32))
    diag_sign = torch.sign(torch.diagonal(r_m))
    q_m = q_m * torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign).unsqueeze(0)

    sem_3d = torch.matmul(centered_active.view(vocab_size, pkt_dim, pkt_count), q_m).transpose(1, 2).contiguous()
    alpha_v = float(spec.alpha_max) * torch.sqrt(tok_counts / (tok_counts + float(spec.bayes_k0)))
    alpha_per_pkt = alpha_v.view(vocab_size, 1, 1).expand(vocab_size, pkt_count, 1)
    return _orthogonal_gram_schmidt_blend(sem_3d, active_mask.squeeze(1), alpha_per_pkt, spec.seed)


@torch.no_grad()
def compute_split_meaning_id_wavepacket_sphere(
    train_tokens: torch.Tensor,
    spec: SupportSketchSpec,
    sem_packets: int | None = None,
    sem_alpha_cap: float | None = None,
) -> torch.Tensor:
    r"""`I6` / `I8`: Split-Packet Wavepacket Bundle (`packets 0..M_sem-1` = Meaning, `packets M_sem..M-1` = Identity)."""
    vocab_size = int(spec.vocab_size)
    pkt_count = int(spec.pkt_count)
    pkt_dim = int(spec.pkt_dim)
    d_emb = pkt_count * pkt_dim
    device = train_tokens.device

    centered_active, tok_counts, active_mask = _compute_raw_active_ppmi_sketch(train_tokens, vocab_size, d_emb)
    gen = torch.Generator(device=device).manual_seed(DEFAULT_ORTHO_BALANCE_SEED + pkt_count)
    q_m, r_m = torch.linalg.qr(torch.randn((pkt_count, pkt_count), generator=gen, device=device, dtype=torch.float32))
    diag_sign = torch.sign(torch.diagonal(r_m))
    q_m = q_m * torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign).unsqueeze(0)

    sem_3d = torch.matmul(centered_active.view(vocab_size, pkt_dim, pkt_count), q_m).transpose(1, 2).contiguous()
    cap = float(spec.split_sem_alpha_max if sem_alpha_cap is None else sem_alpha_cap)
    alpha_sem = cap * torch.sqrt(tok_counts / (tok_counts + float(spec.bayes_k0)))

    alpha_per_pkt = torch.zeros((vocab_size, pkt_count, 1), device=device, dtype=torch.float32)
    n_sem = (pkt_count // 2) if sem_packets is None else int(sem_packets)
    alpha_per_pkt[:, :n_sem, 0] = alpha_sem.unsqueeze(1)
    return _orthogonal_gram_schmidt_blend(sem_3d, active_mask.squeeze(1), alpha_per_pkt, spec.seed)


@torch.no_grad()
def compute_whitened_support_countsketch_sphere(
    train_tokens: torch.Tensor,
    spec: SupportSketchSpec,
) -> torch.Tensor:
    r"""`I7`: ZCA-Whitened Support-Preserving CountSketch (`~6.8 ms` at $V=50,304$)."""
    vocab_size = int(spec.vocab_size)
    pkt_count = int(spec.pkt_count)
    pkt_dim = int(spec.pkt_dim)
    d_emb = pkt_count * pkt_dim
    device = train_tokens.device

    centered_active, tok_counts, active_mask = _compute_raw_active_ppmi_sketch(train_tokens, vocab_size, d_emb)
    row_norm = centered_active.norm(dim=1, keepdim=True).clamp_min(1e-6)
    unit_active = (centered_active / row_norm) * active_mask
    n_act = active_mask.sum().clamp_min(1.0)

    cov = torch.matmul(unit_active.t(), unit_active) / n_act
    evals, evecs = torch.linalg.eigh(cov)
    inv_sqrt_evals = (evals.clamp_min(0.0) + DEFAULT_WHITENING_REG_EPS).rsqrt()
    w_zca = torch.matmul(evecs * inv_sqrt_evals.unsqueeze(0), evecs.t())
    whitened_flat = torch.matmul(unit_active, w_zca)

    gen = torch.Generator(device=device).manual_seed(DEFAULT_ORTHO_BALANCE_SEED + pkt_count)
    q_m, r_m = torch.linalg.qr(torch.randn((pkt_count, pkt_count), generator=gen, device=device, dtype=torch.float32))
    diag_sign = torch.sign(torch.diagonal(r_m))
    q_m = q_m * torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign).unsqueeze(0)

    sem_3d = torch.matmul(whitened_flat.view(vocab_size, pkt_dim, pkt_count), q_m).transpose(1, 2).contiguous()
    alpha_v = float(spec.alpha_max) * torch.sqrt(tok_counts / (tok_counts + float(spec.bayes_k0)))
    alpha_per_pkt = alpha_v.view(vocab_size, 1, 1).expand(vocab_size, pkt_count, 1)
    return _orthogonal_gram_schmidt_blend(sem_3d, active_mask.squeeze(1), alpha_per_pkt, spec.seed)


@torch.no_grad()
def _sketch_pos_to_signed_active_ppmi(
    sketch_pos: torch.Tensor,
    col_signs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert raw positive CountSketch `(V, d_emb)` to active-centered signed PPMI `(V, d_emb)` and active mask `(V, 1)`."""
    n_total = sketch_pos.sum().clamp_min(1.0)
    row_counts_2d = sketch_pos.sum(dim=1, keepdim=True)
    col_counts = sketch_pos.sum(dim=0, keepdim=True).clamp_min(1.0)
    expected = (row_counts_2d.clamp_min(1.0) * col_counts) / n_total
    ppmi_sketch = F.relu(torch.log((sketch_pos + 0.5) / (expected + 0.5)))
    signed_sketch = ppmi_sketch * col_signs.unsqueeze(0)

    row_energy = signed_sketch.square().sum(dim=1, keepdim=True)
    active_mask = (row_energy > 1e-8).float()
    n_active = active_mask.sum().clamp_min(1.0)
    active_mean = (signed_sketch * active_mask).sum(dim=0, keepdim=True) / n_active
    centered_active = (signed_sketch - active_mean) * active_mask
    return centered_active, active_mask


@torch.no_grad()
def _stream_multishard_multilag_bilateral_stats(
    num_shards: int,
    vocab_size: int,
    d_emb: int,
    device: torch.device,
) -> dict[str, Any]:
    r"""Stream `num_shards` FineWeb training shards (`1` = `100M` Shard 1; `9` = `900M` Shards 1..9, zero Shard 0).

    Accumulates both immediate ($k = 1$) and multi-lag ($k \in \{2, 4, 8\}$) 2-hash signed Rademacher CountSketches
    on GPU and caches the resulting sufficient statistics in `.cache/` while recording `cold_stream_ms`.
    """
    cache_path = _CACHE_DIR / f"fineweb_multilag_sketch_v3_s{int(num_shards)}_v{int(vocab_size)}_d{int(d_emb)}.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location=device, weights_only=True)
        return payload

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    half_d = d_emb // 2
    v_ids = torch.arange(vocab_size, device=device, dtype=torch.int64)
    col_hash_fwd = ((v_ids * DEFAULT_HASH_PRIME_1 + 17) % half_d).long()
    col_hash_rev = half_d + ((v_ids * DEFAULT_HASH_PRIME_2 + 31) % half_d).long()
    sign_fwd = torch.where(((v_ids * DEFAULT_HASH_PRIME_2) & 1) == 0, 1.0, -1.0)
    sign_rev = torch.where(((v_ids * DEFAULT_HASH_PRIME_1) & 1) == 0, 1.0, -1.0)

    col_signs = torch.ones(d_emb, device=device, dtype=torch.float32)
    col_signs[:half_d] = sign_fwd[:half_d] if vocab_size >= half_d else 1.0
    col_signs[half_d:] = sign_rev[:half_d] if vocab_size >= half_d else -1.0

    unigram_counts = torch.zeros(vocab_size, device=device, dtype=torch.float64)
    sketch_k1 = torch.zeros((vocab_size, d_emb), device=device, dtype=torch.float32)
    sketch_long = torch.zeros((vocab_size, d_emb), device=device, dtype=torch.float32)
    total_tokens_streamed = 0

    for shard_idx in range(1, int(num_shards) + 1):
        shard_path = DEFAULT_FINEWEB_DIR / f"fineweb_train_{shard_idx:06d}.bin"
        if not shard_path.exists():
            continue
        raw_mmap = np.memmap(shard_path, dtype=np.uint16, mode="r", offset=FINEWEB_HEADER_INT32 * 4)
        n_shard = int(raw_mmap.shape[0])
        for offset in range(0, n_shard, DEFAULT_SHARD_STREAM_CHUNK_TOKENS):
            end = min(n_shard, offset + DEFAULT_SHARD_STREAM_CHUNK_TOKENS)
            chunk_i32 = np.array(raw_mmap[offset:end], dtype=np.int32)
            tok = torch.from_numpy(chunk_i32).to(device=device, dtype=torch.int64).clamp_(0, vocab_size - 2)
            total_tokens_streamed += int(tok.numel())

            c_chunk = torch.bincount(tok, minlength=vocab_size)
            unigram_counts.add_(c_chunk.to(torch.float64))

            u1, v1 = tok[:-1], tok[1:]
            ones1 = torch.ones(u1.numel(), device=device, dtype=torch.float32)
            sketch_k1.view(-1).scatter_add_(0, u1 * d_emb + col_hash_fwd[v1], ones1)
            sketch_k1.view(-1).scatter_add_(0, v1 * d_emb + col_hash_rev[u1], ones1)

            for lag, w_lag in zip(DEFAULT_MULTILAG_OFFSETS, DEFAULT_MULTILAG_WEIGHTS):
                uk, vk = tok[:-lag], tok[lag:]
                w_vec = torch.full((uk.numel(),), float(w_lag), device=device, dtype=torch.float32)
                sketch_long.view(-1).scatter_add_(0, uk * d_emb + col_hash_fwd[vk], w_vec)
                sketch_long.view(-1).scatter_add_(0, vk * d_emb + col_hash_rev[uk], w_vec)

    centered_k1, active_k1 = _sketch_pos_to_signed_active_ppmi(sketch_k1, col_signs)
    centered_long, _ = _sketch_pos_to_signed_active_ppmi(sketch_long + 0.5 * sketch_k1, col_signs)

    if device.type == "cuda":
        torch.cuda.synchronize()
    cold_stream_ms = (time.perf_counter() - t0) * 1000.0
    tok_counts_f32 = unigram_counts.float()
    active_vocab_count = int((tok_counts_f32 > 0).sum().item())

    payload = {
        "unigram_counts": tok_counts_f32.cpu(),
        "centered_k1": centered_k1.to(torch.float16).cpu(),
        "centered_long": centered_long.to(torch.float16).cpu(),
        "active_mask": active_k1.cpu(),
        "cold_stream_ms": float(cold_stream_ms),
        "total_tokens_streamed": int(total_tokens_streamed),
        "active_vocab_count": int(active_vocab_count),
    }
    torch.save(payload, cache_path)
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in payload.items()}


@torch.no_grad()
def compute_multishard_multilag_complex_wavefunction(
    spec: SupportSketchSpec,
    num_shards: int,
    device: torch.device,
    rev_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    r"""`I15`..`I17`: Build `(50304, 8, 2, 32)` complex wavepackets from `100M` (Shard 1) or `900M` (Shards 1..9) tokens.

    Combines immediate ($k=1$) and longer-sequence ($k \in \{2, 4, 8\}$) signed PPMI CountSketches across the `7`
    semantic complex wavepackets (`Packets 0..4` weighted toward immediate syntax $k=1$, `Packets 5..6` weighted
    toward multi-lag phrase/topic context $k \in \{2, 4, 8\}$) balanced via $U_M \in U(8)$, plus `1` Hermitian-orthogonal
    identity complex wavepacket (`Packet 7`).
    """
    vocab_size = int(spec.vocab_size)
    m_complex, d_h, n_sem_c = _COMPLEX_WF_MODE_LAYOUTS[SupportSketchMode(spec.mode)]
    d_emb = 2 * m_complex * d_h
    stats = _stream_multishard_multilag_bilateral_stats(num_shards, vocab_size, d_emb, device)

    tok_counts = stats["unigram_counts"].to(device=device, dtype=torch.float32)
    active_mask = stats["active_mask"].to(device=device, dtype=torch.float32)
    c_k1 = stats["centered_k1"].to(device=device, dtype=torch.float32)
    c_long = stats["centered_long"].to(device=device, dtype=torch.float32)

    counts_for_prior = tok_counts.clone()
    counts_for_prior[rev_token_id] = max(1.0, float(tok_counts.sum().item()) / 31.0)
    probs = (counts_for_prior + 1.0) / (counts_for_prior.sum() + float(vocab_size))
    log_p = torch.log(probs)
    log_prior = log_p - log_p.mean()
    raw_r = probs.pow(0.12)
    radial = (raw_r / raw_r.square().mean().sqrt().clamp_min(1e-6)).clamp(0.40, 2.50)

    # Blend immediate k=1 sketch (75%) with multi-lag k={2,4,8} sequence sketch (25%)
    centered_active = 0.75 * c_k1 + 0.25 * c_long
    half_d = m_complex * d_h
    fwd_3d = centered_active[:, :half_d].view(vocab_size, d_h, m_complex)
    rev_3d = centered_active[:, half_d:].view(vocab_size, d_h, m_complex)

    gen = torch.Generator(device=device).manual_seed(DEFAULT_ORTHO_BALANCE_SEED + 2 * m_complex + d_h)
    a_re = torch.randn((m_complex, m_complex), generator=gen, device=device, dtype=torch.float32)
    a_im = torch.randn((m_complex, m_complex), generator=gen, device=device, dtype=torch.float32)
    q_c, r_c = torch.linalg.qr(torch.complex(a_re, a_im))
    d_diag = torch.diagonal(r_c)
    phase_diag = d_diag / d_diag.abs().clamp_min(1e-8)
    q_c = q_c * phase_diag.unsqueeze(0)
    u_re, u_im = q_c.real.contiguous(), q_c.imag.contiguous()

    sem_re = (torch.matmul(fwd_3d, u_re) - torch.matmul(rev_3d, u_im)).transpose(1, 2).contiguous()
    sem_im = (torch.matmul(fwd_3d, u_im) + torch.matmul(rev_3d, u_re)).transpose(1, 2).contiguous()
    sem_4d = torch.stack([sem_re, sem_im], dim=2)

    corpus_scale = max(1.0, float(stats["total_tokens_streamed"]) / 2_000_000.0)
    k_eff = float(spec.bayes_k0) * corpus_scale
    alpha_sem = 0.96 * torch.sqrt(tok_counts / (tok_counts + k_eff))
    alpha_per_cpkt = torch.zeros((vocab_size, m_complex, 1, 1), device=device, dtype=torch.float32)
    alpha_per_cpkt[:, :n_sem_c, 0, 0] = alpha_sem.unsqueeze(1)
    psi_4d = _hermitian_gram_schmidt_complex_blend(sem_4d, active_mask.squeeze(1), alpha_per_cpkt, spec.seed)

    meta = {
        "cold_stream_ms": round(float(stats["cold_stream_ms"]), 2),
        "tokens_streamed_M": round(float(stats["total_tokens_streamed"]) / 1e6, 1),
        "active_vocab_count": int(stats["active_vocab_count"]),
        "active_vocab_pct": round(100.0 * float(stats["active_vocab_count"]) / 50257.0, 3),
    }
    return psi_4d, log_prior, radial, meta


def _encode_causal_lm_hidden(model: MicroReversalGPT, idx_in: torch.Tensor) -> torch.Tensor:
    """Encode causal sequence `idx_in` `(B, L)` through wavepacket dispersion and quantum blocks to `x_final` `(B, L, d)`."""
    bsz, seq_len = idx_in.shape
    work_dtype = torch.bfloat16 if (idx_in.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
    tok_wp = F.embedding(idx_in, model.wte_3d.view(model.vocab_size, model.embed_dim)).view(
        bsz, seq_len, model.pkt_count, model.pkt_dim
    )
    pos_wp = model.wpe_3d[:seq_len].unsqueeze(0) * model.pos_scale
    raw_wp = (tok_wp + pos_wp).to(work_dtype)
    if seq_len > 1:
        raw_wp = apply_causal_wavepacket_dispersion(raw_wp, model.wp_causal_re, model.wp_causal_im)
    x = F.rms_norm(raw_wp.reshape(bsz, seq_len, model.embed_dim), (model.embed_dim,), eps=1e-6)
    cos_4d, sin_4d = model.rope.get_cos_sin(seq_len, dtype=work_dtype)
    for q_blk in model.quantum_blocks:
        x = q_blk(x, cos_4d, sin_4d)
    return F.rms_norm(x, (model.embed_dim,), eps=1e-6)


@torch.no_grad()
def calibrate_step0_causal_readout_(
    model: MicroReversalGPT,
    dataset: FineWebSpanReversalDataset,
    n_sem_complex: int = 7,
    n_cal_batches: int = 96,
    batch_size: int = DEFAULT_MICRO_BATCH_SIZE,
) -> None:
    r"""Calibrate `wp_causal_re`, `exit_bridge` (`u_stiefel, sigma, v_stiefel`), and `temp_scale` at Step 0 on Shard 1 (`train_tokens`).

    Solves the Ridge-Procrustes SVD transition problem mapping Step-0 hidden states $h_t = x_{\text{final}}(t)$
    to the unigram-centered semantic wavepacket of the next token $\Delta\psi_{\text{sem}}(x_{t+1})$ strictly on
    training Shard 1 (`dataset.train_tokens`), lowering Step-0 Cross-Entropy before any gradient step is taken.
    """
    device = dataset.device
    d_emb = model.embed_dim
    n_sem_real_dim = 2 * int(n_sem_complex) * model.pkt_dim

    # 1. Initialize multi-lag causal dispersion on Semantic Packets 0..6; keep Identity Packet 7 at 0.0
    init_disp = torch.tensor([0.22, 0.22, 0.14, 0.08, 0.22, 0.22, 0.14, 0.00], device=device, dtype=torch.float32)
    model.wp_causal_re.copy_(init_disp[: model.wp_causal_re.numel()])
    model.wp_causal_im.zero_()

    # 2. Moderate Step-0 residual gain so untrained random attention layers do not drown out the calibrated state
    for q_blk in model.quantum_blocks:
        q_blk.obs_scale.fill_(0.20)  # type: ignore[union-attr]
        q_blk.mz_mlp.mz_scale.fill_(0.20)  # type: ignore[union-attr]

    # 3. Unigram-centered semantic target table: Delta_psi[v] = r_v * psi_v - E_{p_0}[r_v * psi_v] on semantic dims
    wte_flat = model.wte_3d.detach().float().view(model.vocab_size, d_emb)
    r_v = model.readout.radial_scale.detach().float().unsqueeze(1)
    weighted_psi = wte_flat * r_v
    p0 = F.softmax(model.readout.unigram_bias.detach().float(), dim=0).unsqueeze(1)
    mean_psi = (p0 * weighted_psi).sum(dim=0, keepdim=True)
    target_table = weighted_psi - mean_psi
    target_table[:, n_sem_real_dim:] = 0.0

    # 4. Accumulate Step-0 backbone covariance H^T H and transition cross-covariance H^T Y on Shard 1 training batches
    seq_len = dataset.span_len
    bsz = int(batch_size)
    max_start = dataset.train_tokens.numel() - seq_len - 2
    stride = max(1, max_start // (n_cal_batches * bsz))

    cov_hh = torch.zeros((d_emb, d_emb), device=device, dtype=torch.float32)
    cov_hy = torch.zeros((d_emb, d_emb), device=device, dtype=torch.float32)
    cal_inputs: list[torch.Tensor] = []
    cal_targets: list[torch.Tensor] = []

    was_training = model.training
    model.eval()
    for b_idx in range(n_cal_batches):
        base = b_idx * bsz * stride
        starts = (base + torch.arange(bsz, device=device) * stride) % max_start
        offsets = torch.arange(seq_len + 1, device=device).unsqueeze(0)
        chunk = dataset.train_tokens[starts.unsqueeze(1) + offsets]
        idx_in = chunk[:, :-1].contiguous()
        targets = chunk[:, 1:].contiguous()
        if b_idx < 4:
            cal_inputs.append(idx_in)
            cal_targets.append(targets)

        h_flat = _encode_causal_lm_hidden(model, idx_in).reshape(-1, d_emb).float()
        y_flat = target_table[targets.reshape(-1)]
        cov_hh.add_(torch.matmul(h_flat.t(), h_flat))
        cov_hy.add_(torch.matmul(h_flat.t(), y_flat))

    n_samples = float(n_cal_batches * bsz * seq_len)
    cov_hh.div_(n_samples)
    cov_hy.div_(n_samples)

    # 5. Solve Ridge-Procrustes system W_exit^* = (Cov_hh + lambda I)^{-1} Cov_hy and load SVD into exit_bridge
    reg_eye = DEFAULT_STEP0_RIDGE_LAMBDA * torch.eye(d_emb, device=device, dtype=torch.float32)
    w_star = torch.linalg.solve(cov_hh + reg_eye, cov_hy)
    u_svd, s_svd, vh_svd = torch.linalg.svd(w_star.t(), full_matrices=False)
    s_rms = s_svd.square().mean().sqrt().clamp_min(1e-6)
    s_norm = (s_svd / s_rms).clamp(0.05, 5.0)

    assert model.readout.exit_bridge is not None
    model.readout.exit_bridge.u_stiefel.copy_(u_svd.contiguous())
    model.readout.exit_bridge.v_stiefel.copy_(vh_svd.t().contiguous())
    model.readout.exit_bridge.sigma.copy_(s_norm.contiguous())

    # 6. Emphasize semantic packets at Step 0 and select optimal Step-0 temp_scale batch-wise on Shard 1
    model.readout.pkt_gate_logits[: 2 * n_sem_complex].fill_(1.35)
    model.readout.pkt_gate_logits[2 * n_sem_complex :].fill_(-1.65)
    model.readout.born_gamma.fill_(0.12)

    h_cal_list = [_encode_causal_lm_hidden(model, x_b) for x_b in cal_inputs]
    y_cal_list = [t_b.reshape(-1).long() for t_b in cal_targets]
    best_temp = 0.45
    best_loss = float("inf")
    for cand_temp in (0.05, 0.12, 0.25, 0.35, 0.45, 0.55, 0.70, 0.85, 1.05):
        model.readout.temp_scale.fill_(cand_temp)
        tot_loss = 0.0
        for h_b, y_b in zip(h_cal_list, y_cal_list):
            logits_b = model.readout(h_b).reshape(-1, model.vocab_size).float()
            tot_loss += float(F.cross_entropy(logits_b, y_b).item())
        if tot_loss < best_loss:
            best_loss = tot_loss
            best_temp = cand_temp
    model.readout.temp_scale.fill_(best_temp)
    if was_training:
        model.train()


@torch.no_grad()
def apply_support_sketch_initialization_(
    model: MicroReversalGPT,
    dataset: FineWebSpanReversalDataset,
    spec: SupportSketchSpec,
) -> tuple[float, dict[str, float]]:
    """Initialize `model` with `spec.mode` and return `(elapsed_ms, geometry_diagnostics)`."""
    tokens = dataset.train_tokens
    mode = SupportSketchMode(spec.mode)
    if tokens.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    log_prior, radial = compute_gpu_unigram_log_prior_and_radial(tokens, spec.vocab_size, dataset.rev_token_id)
    is_complex_wf = mode in _COMPLEX_WF_MODE_LAYOUTS
    multishard_meta: dict[str, float] = {}

    if mode == SupportSketchMode.I2_UNIGRAM_PRIOR_RANDOM_SPHERE:
        gen = torch.Generator(device=model.wte_3d.device).manual_seed(int(spec.seed) + 11)
        emb_3d = torch.randn(model.wte_3d.shape, generator=gen, device=model.wte_3d.device, dtype=torch.float32)
        project_rows_to_rms_sphere_(emb_3d)
    elif mode == SupportSketchMode.I3_RAW_COUNTSKETCH_SPHERE:
        emb_3d = compute_gpu_countsketch_bigram_sphere_embeddings(
            tokens, spec.vocab_size, spec.pkt_count, spec.pkt_dim
        )
    elif mode == SupportSketchMode.I4_LOWRANK_SVD_SPHERE:
        emb_3d = compute_gpu_svd_bigram_sphere_embeddings(
            tokens, spec.vocab_size, spec.pkt_count, spec.pkt_dim, DEFAULT_SVD_ANCHOR_COLS
        )
    elif mode == SupportSketchMode.I5_SUPPORT_GATED_COUNTSKETCH:
        emb_3d = compute_support_gated_countsketch_sphere(tokens, spec)
    elif mode == SupportSketchMode.I6_SPLIT_MEANING_ID_WAVEPACKET:
        emb_3d = compute_split_meaning_id_wavepacket_sphere(tokens, spec)
    elif mode == SupportSketchMode.I7_WHITENED_SUPPORT_COUNTSKETCH:
        emb_3d = compute_whitened_support_countsketch_sphere(tokens, spec)
    elif mode == SupportSketchMode.I8_SPLIT_MEANING_ID_ADAPTIVE_PKT_GATE:
        n_sem = max(1, (3 * int(spec.pkt_count)) // 4)
        emb_3d = compute_split_meaning_id_wavepacket_sphere(tokens, spec, sem_packets=n_sem, sem_alpha_cap=0.96)
    elif mode in _MULTISHARD_MODES:
        n_shards, _ = _MULTISHARD_MODES[mode]
        psi_4d, log_prior, radial, multishard_meta = compute_multishard_multilag_complex_wavefunction(
            spec, n_shards, tokens.device, dataset.rev_token_id
        )
        emb_3d = psi_4d.view(model.wte_3d.shape)
    elif is_complex_wf:
        _, _, n_sem_c = _COMPLEX_WF_MODE_LAYOUTS[mode]
        psi_4d = compute_complex_wavefunction_split_sphere(tokens, spec, sem_complex_packets=n_sem_c, sem_alpha_cap=0.96)
        emb_3d = psi_4d.view(model.wte_3d.shape)
    else:
        raise ValueError(f"Unsupported SupportSketchMode: {mode}")

    model.wte_3d.copy_(emb_3d.to(model.wte_3d.device, dtype=model.wte_3d.dtype))
    setattr(model.wte_3d, "_complex_wavepacket_pair", is_complex_wf)
    model.readout.initialize_readout_priors_(log_prior, radial, init_detectors_from_wte=True)
    model.readout.complex_sphere_mode = is_complex_wf
    if is_complex_wf:
        project_complex_wavepackets_to_rms_sphere_(
            model.wpe_3d.view(model.seq_len, model.pkt_count // 2, 2, model.pkt_dim)
        )
        _, _, n_sem_c = _COMPLEX_WF_MODE_LAYOUTS[mode]
        n_sem_real = 2 * n_sem_c
        model.readout.pkt_gate_logits[:n_sem_real].fill_(0.85)
        model.readout.pkt_gate_logits[n_sem_real:].fill_(-0.85)
        if mode in _MULTISHARD_MODES and _MULTISHARD_MODES[mode][1]:
            calibrate_step0_causal_readout_(model, dataset, n_sem_complex=n_sem_c)
    elif mode == SupportSketchMode.I8_SPLIT_MEANING_ID_ADAPTIVE_PKT_GATE:
        n_sem = max(1, (3 * int(spec.pkt_count)) // 4)
        model.readout.pkt_gate_logits[:n_sem].fill_(0.85)
        model.readout.pkt_gate_logits[n_sem:].fill_(-0.85)

    if tokens.is_cuda:
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    geom = measure_sphere_geometry_and_crowding(emb_3d, tokens, spec.vocab_size)
    geom.update(multishard_meta)
    return elapsed_ms, geom


@torch.no_grad()
def measure_sphere_geometry_and_crowding(
    emb_3d: torch.Tensor,
    train_tokens: torch.Tensor,
    vocab_size: int,
) -> dict[str, float]:
    """Measure rare-token collapse (`rare_max_cos`), frequent-token crowding (`freq_max_cos`), and semantic alignment."""
    d_emb = emb_3d.size(1) * emb_3d.size(2)
    flat_unit = F.normalize(emb_3d.detach().float().view(vocab_size, d_emb), dim=-1)
    counts = torch.bincount(train_tokens, minlength=vocab_size)

    # 1. Sample 512 unseen/rare tokens (count == 0) to check for rare-token collapse
    zero_ids = torch.nonzero(counts == 0, as_tuple=False).squeeze(-1)
    if zero_ids.numel() >= 16:
        sample_rare = zero_ids[: min(DEFAULT_GEOM_SAMPLE_SIZE, zero_ids.numel())]
    else:
        sample_rare = torch.topk(counts, k=min(DEFAULT_GEOM_SAMPLE_SIZE, vocab_size), largest=False).indices
    u_rare = flat_unit[sample_rare]
    gram_rare = torch.matmul(u_rare, u_rare.t())
    gram_rare.fill_diagonal_(-1.0)
    rare_max_cos = float(gram_rare.max().item())
    rare_mean_abs_cos = float(gram_rare.triu(diagonal=1).abs().mean().item())

    # 2. Top 512 frequent tokens: check max pairwise cosine (crowding) and semantic correlation with raw PPMI sketch
    top_ids = torch.topk(counts, k=min(DEFAULT_GEOM_SAMPLE_SIZE, vocab_size), largest=True).indices
    u_freq = flat_unit[top_ids]
    gram_freq = torch.matmul(u_freq, u_freq.t())
    gram_freq.fill_diagonal_(-1.0)
    freq_max_cos = float(gram_freq.max().item())

    centered_active, _, _ = _compute_raw_active_ppmi_sketch(train_tokens, vocab_size, d_emb)
    ref_freq = F.normalize(centered_active[top_ids], dim=-1)
    gram_ref = torch.matmul(ref_freq, ref_freq.t())
    mask_upper = torch.triu(torch.ones_like(gram_ref, dtype=torch.bool), diagonal=1)
    v_emb = gram_freq[mask_upper]
    v_ref = gram_ref[mask_upper]
    v_emb_c = v_emb - v_emb.mean()
    v_ref_c = v_ref - v_ref.mean()
    denom = (v_emb_c.norm() * v_ref_c.norm()).clamp_min(1e-8)
    semantic_gram_corr = float((v_emb_c * v_ref_c).sum().div(denom).item())

    return {
        "rare_max_cos": round(rare_max_cos, 4),
        "rare_mean_abs_cos": round(rare_mean_abs_cos, 4),
        "freq_max_cos": round(freq_max_cos, 4),
        "semantic_gram_corr": round(semantic_gram_corr, 4),
    }


@torch.no_grad()
def evaluate_causal_lm_on_dataset(
    model: MicroReversalGPT,
    dataset: FineWebSpanReversalDataset,
) -> dict[str, float]:
    """Evaluate `model` on causal next-token language modeling over `dataset.val_tokens` spans (`B x L`)."""
    was_training = model.training
    model.eval()
    seq_len = dataset.span_len
    bsz = DEFAULT_MICRO_BATCH_SIZE
    n_batches = 8
    max_start = dataset.val_tokens.numel() - seq_len - 2
    stride = max(1, max_start // (n_batches * bsz))

    nll_list: list[torch.Tensor] = []
    top1_list: list[torch.Tensor] = []
    top5_list: list[torch.Tensor] = []

    for b_idx in range(n_batches):
        base = b_idx * bsz * stride
        starts = (base + torch.arange(bsz, device=dataset.device) * stride) % max_start
        offsets = torch.arange(seq_len + 1, device=dataset.device).unsqueeze(0)
        chunk = dataset.val_tokens[starts.unsqueeze(1) + offsets]
        idx_in = chunk[:, :-1].contiguous()
        targets = chunk[:, 1:].contiguous()

        x_final = _encode_causal_lm_hidden(model, idx_in)
        logits = model.readout(x_final)

        z = logits.reshape(-1, model.vocab_size).float()
        y = targets.reshape(-1).long()
        lse = torch.logsumexp(z, dim=-1)
        target_z = z.gather(1, y.unsqueeze(1)).squeeze(1)
        nll_list.append(lse - target_z)
        ranks = (z > target_z.unsqueeze(1)).sum(dim=-1) + 1
        top1_list.append((ranks == 1).float())
        top5_list.append((ranks <= 5).float())

    if was_training:
        model.train()

    all_nll = torch.cat(nll_list, dim=0)
    val_ce = float(all_nll.mean().item())
    return {
        "val_ce": val_ce,
        "val_ppl": float(math.exp(min(20.0, val_ce))),
        "top1_acc": float(torch.cat(top1_list, dim=0).mean().item()),
        "top5_acc": float(torch.cat(top5_list, dim=0).mean().item()),
    }


def run_support_sketch_experiment(
    spec: SupportSketchSpec,
    dataset: FineWebSpanReversalDataset,
    task_mode: str = "reversal",
    total_steps: int = 300,
    eval_every: int = 100,
) -> dict[str, Any]:
    """Run 300-step training on either `reversal` (exact identity) or `causal_lm` (semantic + identity) at V=50,304."""
    device = dataset.device
    mode = SupportSketchMode(spec.mode)
    is_complex_wf = mode in _COMPLEX_WF_MODE_LAYOUTS
    if is_complex_wf:
        m_complex, eff_pkt_dim, _ = _COMPLEX_WF_MODE_LAYOUTS[mode]
        eff_pkt_count = 2 * m_complex
    else:
        eff_pkt_dim = int(spec.pkt_dim)
        eff_pkt_count = int(spec.pkt_count)
    eff_embed_dim = eff_pkt_count * eff_pkt_dim
    eff_mlp_hidden = 3 * eff_embed_dim
    eff_spec = SupportSketchSpec(
        mode=mode,
        vocab_size=spec.vocab_size,
        pkt_count=eff_pkt_count,
        pkt_dim=eff_pkt_dim,
        bayes_k0=spec.bayes_k0,
        alpha_max=spec.alpha_max,
        split_sem_alpha_max=spec.split_sem_alpha_max,
        seed=spec.seed,
    )
    cfg = MicroReversalConfig(
        backbone_mode=MicroBackboneMode.QUANTUM_WAVEPACKET,
        init_mode=MicroInitMode.GPU_UNIGRAM_PRIOR,
        readout_mode=MicroReadoutMode.R4_HERMITIAN_QUADRATIC_BORN,
        vocab_size=eff_spec.vocab_size,
        span_len=dataset.span_len,
        embed_dim=eff_embed_dim,
        num_heads=DEFAULT_MICRO_NUM_HEADS,
        num_layers=DEFAULT_MICRO_NUM_LAYERS,
        mlp_hidden=eff_mlp_hidden,
        pkt_count=eff_spec.pkt_count,
        pkt_dim=eff_spec.pkt_dim,
        seed=eff_spec.seed,
    )
    model = MicroReversalGPT(cfg, seq_len=dataset.seq_len).to(device)
    init_ms, geom = apply_support_sketch_initialization_(model, dataset, eff_spec)

    stiefel_params: list[nn.Parameter] = list(model.readout.real_stiefel_parameters())
    unitary_pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
    for q_blk in model.quantum_blocks:
        stiefel_params.extend(q_blk.real_stiefel_parameters())  # type: ignore[operator]
        unitary_pairs.extend(q_blk.complex_unitary_pairs())  # type: ignore[operator]

    q_trunk_opt = ModularRiemannianStiefelKrausMuon(
        real_stiefel_params=stiefel_params,
        complex_operator_pairs=unitary_pairs,
        stiefel_lr=DEFAULT_MICRO_STIEFEL_LR,
        unitary_lr=DEFAULT_MICRO_UNITARY_LR,
        tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA,
        momentum_mode=StiefelMomentumMode.RIEMANNIAN_TANGENT_SNR_GATED,
    )
    wte_opt = BatchSubspaceTikhonovGramMuon(
        [model.wte_3d] + model.readout.sphere_parameters(),
        lr=DEFAULT_MICRO_WTE_LR,
        tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA,
    )
    scalar_params: list[nn.Parameter] = [
        model.wpe_3d,
        model.pos_scale,
        model.wp_causal_re,
        model.wp_causal_im,
    ]
    scalar_params.extend(model.readout.scalar_parameters())
    for q_blk in model.quantum_blocks:
        scalar_params.extend(q_blk.scalar_parameters())  # type: ignore[operator]
    scalar_opt = torch.optim.AdamW(
        [{"params": scalar_params, "lr": DEFAULT_MICRO_SCALAR_LR, "weight_decay": 0.0}],
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    eval_fn = evaluate_micro_reversal_model if task_mode == "reversal" else evaluate_causal_lm_on_dataset
    checkpoints: list[dict[str, Any]] = []
    s0 = eval_fn(model, dataset)
    s0["step"] = 0
    checkpoints.append(s0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_train0 = time.perf_counter()

    lm_seq_len = dataset.span_len
    bsz = DEFAULT_MICRO_BATCH_SIZE
    max_lm_start = dataset.train_tokens.numel() - lm_seq_len - 2

    model.train()
    for step in range(1, int(total_steps) + 1):
        warm_steps = max(1, int(0.10 * total_steps))
        if step <= warm_steps:
            lr_mult = 0.15 + 0.85 * (float(step - 1) / float(max(1, warm_steps - 1)))
        else:
            prog = float(step - warm_steps) / float(max(1, total_steps - warm_steps))
            lr_mult = 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * prog))
        mom_beta = compute_riemannian_momentum_beta(
            step, total_steps, QuantumMomentumScheduleMode.LOW_STEADY_MOMENTUM_070
        )

        q_trunk_opt.zero_grad(set_to_none=True)
        wte_opt.zero_grad(set_to_none=True)
        scalar_opt.zero_grad(set_to_none=True)

        if task_mode == "reversal":
            idx_in, targets = dataset.sample_train_batch(step, batch_size=bsz)
            _, loss = model(idx_in, targets)
        else:
            starts = (step * bsz * lm_seq_len + torch.arange(bsz, device=device) * 997) % max_lm_start
            offsets = torch.arange(lm_seq_len + 1, device=device).unsqueeze(0)
            chunk = dataset.train_tokens[starts.unsqueeze(1) + offsets]
            idx_in = chunk[:, :-1].contiguous()
            targets = chunk[:, 1:].contiguous()
            x_final = _encode_causal_lm_hidden(model, idx_in)
            logits = model.readout(x_final)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), targets.reshape(-1).long())

        assert loss is not None
        loss.backward()

        q_trunk_opt.set_momentum(mom_beta)
        q_trunk_opt.set_lr_scale(lr_mult)
        q_trunk_opt.step()

        active_ids = torch.unique(torch.cat([idx_in.reshape(-1), targets.reshape(-1)]))
        wte_opt.set_momentum(mom_beta)
        wte_opt.step_active(active_ids, lr_scale=lr_mult)

        scalar_opt.param_groups[0]["lr"] = DEFAULT_MICRO_SCALAR_LR * lr_mult
        scalar_opt.step()

        if is_complex_wf:
            project_complex_wavepackets_to_rms_sphere_(
                model.wpe_3d.view(model.seq_len, model.pkt_count // 2, 2, model.pkt_dim)
            )
            if mode in _MULTISHARD_MODES:
                _, _, n_sem_c = _COMPLEX_WF_MODE_LAYOUTS[mode]
                with torch.no_grad():
                    model.wp_causal_re[n_sem_c:].zero_()
                    model.wp_causal_im[n_sem_c:].zero_()
        else:
            project_rows_to_rms_sphere_(model.wpe_3d)
        model.restrict_all_scalars_()

        if step % int(eval_every) == 0 or step == int(total_steps):
            ev = eval_fn(model, dataset)
            ev["step"] = int(step)
            ev["train_loss"] = float(loss.detach().item())
            checkpoints.append(ev)

    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - t_train0
    return {
        "mode": spec.mode.value,
        "task_mode": task_mode,
        "wte_shape": [eff_spec.vocab_size, eff_spec.pkt_count // 2, 2, eff_spec.pkt_dim]
        if is_complex_wf
        else [eff_spec.vocab_size, eff_spec.pkt_count, eff_spec.pkt_dim],
        "init_ms": round(init_ms, 3),
        "train_wall_s": round(wall_s, 3),
        "geometry": geom,
        "step0": checkpoints[0],
        "final": checkpoints[-1],
        "checkpoints": checkpoints,
    }
