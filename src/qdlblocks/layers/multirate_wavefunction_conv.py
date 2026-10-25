"""Explicit Level-Separated Wavelet Tensors & Multi-Rate 1D Wavefunction Convolution Bank (`Wave 32` / `Plan 047`).

When to Use This Building Block:
- Use `WaveletScaleState` whenever synthesizing or evolving a multi-scale wavelet representation
  (`coarse_256`, `mid_2048`, `sub_8192`, `fine_65536`) so that each wavelet level is stored and
  normalized as a separate tensor in code (`||psi^{(m)}||_2 = 1 / sqrt(M)`), completely preventing
  high-frequency gradient updates on coarse roots (`wav_256`) from inflating a shared global L2
  denominator and crushing fine leaf representations (`wav_65536`).
- Use `MultiRateWavefunctionConvBank` when you want tokens represented as a multi-rate causal 1D
  strided/dilated convolution over a bank of wavefunctions (at temporal dilations `4, 2, 1, 1`)
  that feeds *directly into* Quantum Born Attention (`SubspaceProjectorBornAttention`), rather than
  only applying a post-attention channel U-Net.

Why It Was Built & Mathematical Provenance (`EXPERIMENT_PLAN_047.md` / `Wave 32`):
1. **Resolving Global L2 Norm Cannibalization Across Wavelet Levels**:
   In `Tied2Pow16WaveletCholeskyPyramid._synthesize_leaf_vectors`, summing `g_0 c_3 + g_1 c_2 + g_2 c_1 + g_3 c_0`
   into a single `R^d` tensor before a single global `l2_normalize` meant that as `||c_3||_2` (`wav_256`)
   grew during semantic clustering, `c_0` (`wav_65536`) was divided down toward zero. Keeping the 4
   levels in separate tensors inside `WaveletScaleState` and normalizing each level independently onto
   `(1/2) S^{d_s - 1}` locks every scale at exact `25.0%` (`1/4`) energy (`||concat||_2 = 1` identically).
2. **Pre-Attention Multi-Rate 1D Wavefunction Convolutions vs. Post-Attention Channel U-Net**:
   Our earlier `FusedKroneckerWaveletUNetBlock` ran Sweldens wavelets across the *channel* axis (`d -> d/8`)
   *after* attention. `MultiRateWavefunctionConvBank` evolves the 4 separated wavelet-level tensors along
   the *sequence/time* axis (`t, t - s_m, t - 2 s_m` with strides/dilations `s = (4, 2, 1, 1)`) *before*
   attention so Born attention queries and keys `Q_t, K_t` carry multi-scale temporal wavefunction packets.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import l2_normalize
from .wavelet_cholesky_branch import BlockCholeskyLinear

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Documented in EXPERIMENT_PLAN_047.md (Wave 32 Constants & Calibration Ledger)
# ==============================================================================
NUM_WAVELET_PYRAMID_LEVELS: int = 4
DEFAULT_LEVEL_TEMPORAL_STRIDES: tuple[int, int, int, int] = (4, 2, 1, 1)
DEFAULT_CONV_MIX_INIT: float = 0.22
DEFAULT_CROSS_SCALE_PHASE_INIT: float = 0.15
DEFAULT_LEVEL_NORM_EPS: float = 1e-8
DEFAULT_MAX_CHOLESKY_BLOCK: int = 32
COARSE_LAG1_WEIGHT: float = 0.65
COARSE_LAG2_WEIGHT: float = 0.35
SYMPLECTIC_PAIR_DIM: int = 2


@dataclass
class WaveletScaleState:
    """Explicitly separated 4-level wavelet tensors `(coarse_256, mid_2048, sub_8192, fine_65536)`.

    Each tensor has shape `(..., d_s)` where `d_s = embed_dim // 4`. Keeping the 4 levels in
    separate tensors makes their scale distinction explicit in code and prevents a global L2 norm
    from allowing coarse levels to cannibalize fine leaf levels.
    """

    coarse_256: torch.Tensor
    mid_2048: torch.Tensor
    sub_8192: torch.Tensor
    fine_65536: torch.Tensor

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the 4 level tensors ordered from coarsest (`coarse_256`) to finest (`fine_65536`)."""
        return (self.coarse_256, self.mid_2048, self.sub_8192, self.fine_65536)

    def normalize_levels(self, eps: float = DEFAULT_LEVEL_NORM_EPS) -> WaveletScaleState:
        """Normalize each wavelet level tensor independently onto `(1 / sqrt(4)) S^{d_s - 1}`."""
        scale = 1.0 / math.sqrt(float(NUM_WAVELET_PYRAMID_LEVELS))
        return WaveletScaleState(
            coarse_256=l2_normalize(self.coarse_256, dim=-1, eps=eps) * scale,
            mid_2048=l2_normalize(self.mid_2048, dim=-1, eps=eps) * scale,
            sub_8192=l2_normalize(self.sub_8192, dim=-1, eps=eps) * scale,
            fine_65536=l2_normalize(self.fine_65536, dim=-1, eps=eps) * scale,
        )

    def to_product_of_spheres(self, eps: float = DEFAULT_LEVEL_NORM_EPS) -> torch.Tensor:
        """Normalize each level independently and concatenate into `M_{d, 4} subset S^{d-1}`."""
        normed = self.normalize_levels(eps=eps)
        return torch.cat(normed.as_tuple(), dim=-1)

    def to_interleaved_product_of_spheres(
        self,
        eps: float = DEFAULT_LEVEL_NORM_EPS,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Interleave symplectic `(even, odd)` pairs across the 4 levels (`[..., d_s // 2, 4, 2] -> [..., d]`).

        Updated in `Wave 34` (`EXPERIMENT_PLAN_049.md` Section 3.1.2) so that each level's `2k` and `2k+1`
        coordinates land at adjacent even/odd indices `8k + 2m` and `8k + 2m + 1` in the full `d`-dim vector,
        preserving symplectic `U(1)` gauge covariance for `embed_dyadic_bigram_phase`, `RoPE`, and
        `OvercompleteTwoBeamInterference`.
        """
        normed = self.normalize_levels(eps=eps) if normalize else self
        lead_shape = normed.coarse_256.shape[:-1]
        d_level = normed.coarse_256.size(-1)
        if d_level % SYMPLECTIC_PAIR_DIM == 0:
            n_pairs = d_level // SYMPLECTIC_PAIR_DIM
            c3 = normed.coarse_256.reshape(*lead_shape, n_pairs, SYMPLECTIC_PAIR_DIM)
            c2 = normed.mid_2048.reshape(*lead_shape, n_pairs, SYMPLECTIC_PAIR_DIM)
            c1 = normed.sub_8192.reshape(*lead_shape, n_pairs, SYMPLECTIC_PAIR_DIM)
            c0 = normed.fine_65536.reshape(*lead_shape, n_pairs, SYMPLECTIC_PAIR_DIM)
            stacked = torch.stack((c3, c2, c1, c0), dim=-2)
            return stacked.reshape(*lead_shape, d_level * NUM_WAVELET_PYRAMID_LEVELS)
        stacked = torch.stack(normed.as_tuple(), dim=-1)
        return stacked.reshape(*lead_shape, d_level * NUM_WAVELET_PYRAMID_LEVELS)

    @staticmethod
    def from_concat(x: torch.Tensor) -> WaveletScaleState:
        """Unpack a concatenated `(..., d)` tensor into 4 explicit level tensors of width `d // 4`."""
        c3, c2, c1, c0 = x.chunk(NUM_WAVELET_PYRAMID_LEVELS, dim=-1)
        return WaveletScaleState(coarse_256=c3, mid_2048=c2, sub_8192=c1, fine_65536=c0)

    @classmethod
    def from_interleaved(cls, x: torch.Tensor) -> WaveletScaleState:
        """Unpack a pair-strided interleaved `(..., d)` tensor `[..., d_s // 2, 4, 2]` into 4 level tensors (`Wave 34`)."""
        lead_shape = x.shape[:-1]
        d = x.size(-1)
        d_level = d // NUM_WAVELET_PYRAMID_LEVELS
        if d_level % SYMPLECTIC_PAIR_DIM == 0:
            n_pairs = d_level // SYMPLECTIC_PAIR_DIM
            unpacked = x.reshape(*lead_shape, n_pairs, NUM_WAVELET_PYRAMID_LEVELS, SYMPLECTIC_PAIR_DIM)
            return cls(
                coarse_256=unpacked[..., 0, :].reshape(*lead_shape, d_level),
                mid_2048=unpacked[..., 1, :].reshape(*lead_shape, d_level),
                sub_8192=unpacked[..., 2, :].reshape(*lead_shape, d_level),
                fine_65536=unpacked[..., 3, :].reshape(*lead_shape, d_level),
            )
        unpacked = x.reshape(*lead_shape, d_level, NUM_WAVELET_PYRAMID_LEVELS)
        return cls(
            coarse_256=unpacked[..., 0],
            mid_2048=unpacked[..., 1],
            sub_8192=unpacked[..., 2],
            fine_65536=unpacked[..., 3],
        )

    def level_energy_fractions(self, eps: float = DEFAULT_LEVEL_NORM_EPS) -> dict[str, float]:
        """Compute the mean L2 energy fraction `||psi^{(m)}||_2^2 / sum_k ||psi^{(k)}||_2^2` per level."""
        energies = [t.detach().float().square().sum(dim=-1) for t in self.as_tuple()]
        total = (energies[0] + energies[1] + energies[2] + energies[3]).clamp_min(eps)
        names = ("coarse_256", "mid_2048", "sub_8192", "fine_65536")
        return {name: float((e / total).mean().item()) for name, e in zip(names, energies)}


def _causal_shift(x: torch.Tensor, lag: int) -> torch.Tensor:
    """Shift sequence tensor `x` of shape `(B, T, d_s)` causally by `lag` steps with zero future leakage."""
    if lag <= 0 or x.ndim < 3 or x.size(1) <= lag:
        return torch.zeros_like(x)
    return F.pad(x[:, :-lag, :], (0, 0, lag, 0))


class MultiRateWavefunctionConvBank(nn.Module):
    """Causal 1D Strided/Dilated Wavefunction Convolution Filter Bank (`Wave 32` / `Plan 047`).

    Evolves the 4 separated wavelet levels `(coarse_256, mid_2048, sub_8192, fine_65536)` over
    sequence positions `t` at temporal dilations/strides `(4, 2, 1, 1)` before feeding into
    Quantum Born Attention:
    - `coarse_256` (`dilation=4`, 3-tap causal filter over `t, t-4, t-8` + stride-4 causal pool)
    - `mid_2048` (`dilation=2`, 2-tap causal filter over `t, t-2` + stride-2 causal average)
    - `sub_8192` (`dilation=1`, 2-tap causal filter over `t, t-1`)
    - `fine_65536` (`dilation=1`, 1-tapCholesky-conditioned lexical anchor + cross-scale phase gate)
    """

    def __init__(
        self,
        embed_dim: int,
        temporal_strides: tuple[int, int, int, int] = DEFAULT_LEVEL_TEMPORAL_STRIDES,
        max_block_size: int = DEFAULT_MAX_CHOLESKY_BLOCK,
        fast_spd_bmm: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % NUM_WAVELET_PYRAMID_LEVELS != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by {NUM_WAVELET_PYRAMID_LEVELS}")
        self.embed_dim = int(embed_dim)
        self.level_dim = self.embed_dim // NUM_WAVELET_PYRAMID_LEVELS
        self.temporal_strides = tuple(int(s) for s in temporal_strides)
        blk = min(int(max_block_size), self.level_dim)

        self.chol_lag1 = nn.ModuleList(
            [BlockCholeskyLinear(self.level_dim, max_block_size=blk, fast_spd_bmm=fast_spd_bmm) for _ in range(3)]
        )
        self.chol_coarse_lag2 = BlockCholeskyLinear(self.level_dim, max_block_size=blk, fast_spd_bmm=fast_spd_bmm)
        self.conv_mix_gates = nn.Parameter(
            torch.full((NUM_WAVELET_PYRAMID_LEVELS, self.level_dim), DEFAULT_CONV_MIX_INIT)
        )
        self.cross_scale_gates = nn.Parameter(
            torch.full((NUM_WAVELET_PYRAMID_LEVELS - 1, self.level_dim), DEFAULT_CROSS_SCALE_PHASE_INIT)
        )

    def forward_levels(self, state: WaveletScaleState) -> WaveletScaleState:
        """Apply multi-rate causal 1D wavefunction convolutions and return normalized `WaveletScaleState`."""
        s0, s1, s2, _ = self.temporal_strides
        c3, c2, c1, c0 = state.as_tuple()
        gates = torch.tanh(self.conv_mix_gates).to(c3.dtype)
        x_gates = torch.tanh(self.cross_scale_gates).to(c3.dtype)

        # Level 0 (coarse_256, stride/dilation s0=4): 3-tap strided causal wavefunction filter
        c3_lag1 = self.chol_lag1[0](_causal_shift(c3, s0))
        c3_lag2 = self.chol_coarse_lag2(_causal_shift(c3, 2 * s0))
        c3_conv = c3 + gates[0] * (COARSE_LAG1_WEIGHT * c3_lag1 + COARSE_LAG2_WEIGHT * c3_lag2)

        # Level 1 (mid_2048, stride/dilation s1=2): 2-tap strided causal filter + coarse phase coupling
        c2_lag1 = self.chol_lag1[1](_causal_shift(c2, s1))
        c2_conv = c2 + gates[1] * c2_lag1 + x_gates[0] * (c3_conv * torch.cos(c2))

        # Level 2 (sub_8192, stride/dilation s2=1): 2-tap local bigram wavefunction filter + mid phase coupling
        c1_lag1 = self.chol_lag1[2](_causal_shift(c1, s2))
        c1_conv = c1 + gates[2] * c1_lag1 + x_gates[1] * (c2_conv * torch.cos(c1))

        # Level 3 (fine_65536, stride=1): fine token wavefunction modulated by local sub_8192 envelope
        c0_conv = c0 + gates[3] * _causal_shift(c0, 1) + x_gates[2] * (c1_conv * torch.cos(c0))

        return WaveletScaleState(
            coarse_256=c3_conv,
            mid_2048=c2_conv,
            sub_8192=c1_conv,
            fine_65536=c0_conv,
        ).normalize_levels()

    def forward(self, x: torch.Tensor | WaveletScaleState) -> torch.Tensor | WaveletScaleState:
        """Unpack `(B, T, d)` into 4 wavelet levels (or accept `WaveletScaleState` directly), apply convs, and return."""
        if isinstance(x, WaveletScaleState):
            return self.forward_levels(x)
        state = WaveletScaleState.from_concat(x)
        out_state = self.forward_levels(state)
        return out_state.to_product_of_spheres()


__all__ = [
    "DEFAULT_CONV_MIX_INIT",
    "DEFAULT_CROSS_SCALE_PHASE_INIT",
    "DEFAULT_LEVEL_NORM_EPS",
    "DEFAULT_LEVEL_TEMPORAL_STRIDES",
    "MultiRateWavefunctionConvBank",
    "NUM_WAVELET_PYRAMID_LEVELS",
    "WaveletScaleState",
]
