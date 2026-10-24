"""Clean re-export of Wavelet-Cholesky multiscale branch primitives (`qdlblocks.layers.wavelet_cholesky_branch`)."""

from __future__ import annotations

import torch.nn as nn

from .wavelet_cholesky import (
    BlockCholeskyLinear,
    WaveletCholeskyConfig,
    WaveletCholeskyMultiscaleBranch,
    apply_register_cholesky_coupling,
    apply_soft_powerlaw_spectral_floor,
    build_lower_triangular_cholesky,
)

RMSNorm = nn.RMSNorm

__all__ = [
    "BlockCholeskyLinear",
    "RMSNorm",
    "WaveletCholeskyConfig",
    "WaveletCholeskyMultiscaleBranch",
    "apply_register_cholesky_coupling",
    "apply_soft_powerlaw_spectral_floor",
    "build_lower_triangular_cholesky",
]
