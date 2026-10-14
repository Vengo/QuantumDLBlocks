"""Isometric Hilbert space embeddings, Stiefel manifold lifts, Unitary RoPE, and spherical residuals."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..functional import apply_unitary_rope, l2_normalize, riemannian_center


class UnitSphereEmbedding(nn.Module):
    """Token embedding table calibrated and normalized on the unit hypersphere S^{d-1}.

    Initializes weights with standard deviation `1 / sqrt(embed_dim)` so that raw
    embedding vectors have expected Euclidean norm ~1.0 at initialization (avoiding
    the ~sqrt(d) gradient dilution of standard N(0, 1) tables), and projects lookups
    strictly onto S^{d-1}.
    """

    def __init__(self, vocab_size: int, embed_dim: int, eps: float = 1e-12) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.eps = eps

        self.wte = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.wte.weight, mean=0.0, std=1.0 / math.sqrt(embed_dim))

    @property
    def weight(self) -> torch.Tensor:
        return self.wte.weight

    def normalized_weight(self) -> torch.Tensor:
        """Return unit-normalized vocabulary frame on S^{d-1} of shape `(vocab_size, embed_dim)`."""
        return l2_normalize(self.wte.weight, dim=-1, eps=self.eps)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Lookup token IDs `idx` and project onto S^{embed_dim - 1}."""
        return l2_normalize(self.wte(idx), dim=-1, eps=self.eps)


class IsometricHilbertLift(nn.Module):
    """Factorized Seed-to-Hilbert-Space Stiefel Manifold Lifting Operator.

    1. Embeds tokens into a compact seed Hilbert space S^{seed_dim - 1}.
    2. Lifts seed states via an orthogonal Stiefel operator W_lift in R^{embed_dim x seed_dim}
       into the full contextual Hilbert space S^{embed_dim - 1}.
    """

    def __init__(
        self,
        vocab_size: int,
        seed_dim: int = 128,
        embed_dim: int = 1024,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seed_dim = seed_dim
        self.embed_dim = embed_dim
        self.eps = eps

        self.wte = nn.Embedding(vocab_size, seed_dim)
        nn.init.normal_(self.wte.weight, mean=0.0, std=1.0 / math.sqrt(seed_dim))

        self.W_lift = nn.Linear(seed_dim, embed_dim, bias=False)
        nn.init.orthogonal_(self.W_lift.weight)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Embed token IDs `(B, T)` into seed space and lift to `S^{embed_dim - 1}` `(B, T, embed_dim)`."""
        psi_seed = l2_normalize(self.wte(idx), dim=-1, eps=self.eps)
        psi = l2_normalize(self.W_lift(psi_seed), dim=-1, eps=self.eps)
        return psi

    def collapse_to_seed(self, psi: torch.Tensor) -> torch.Tensor:
        """Project contextual state `psi` `(..., embed_dim)` back into seed space `S^{seed_dim - 1}`."""
        return l2_normalize(torch.matmul(psi, self.W_lift.weight), dim=-1, eps=self.eps)


class UnitaryRoPE(nn.Module):
    """Precomputed Unitary Rotary Position Embedding (RoPE) on SO(2)^{head_dim/2}."""

    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 1024,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for UnitaryRoPE, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        half_d = head_dim // 2
        freqs = 1.0 / (base ** (torch.arange(0, half_d).float() / half_d))
        t = torch.arange(max_seq_len).float()
        angles = torch.outer(t, freqs)
        self.register_buffer(
            "rope_cos",
            torch.cos(angles).view(1, 1, max_seq_len, half_d),
            persistent=False,
        )
        self.register_buffer(
            "rope_sin",
            torch.sin(angles).view(1, 1, max_seq_len, half_d),
            persistent=False,
        )

    def get_cos_sin(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice `(cos, sin)` tables for sequence length `T <= max_seq_len`."""
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds UnitaryRoPE max_seq_len {self.max_seq_len}"
            )
        return self.rope_cos[:, :, :seq_len, :], self.rope_sin[:, :, :seq_len, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply unitary rotary position embedding to tensor of shape `(B, H, T, head_dim)`."""
        seq_len = x.size(-2)
        cos, sin = self.get_cos_sin(seq_len)
        return apply_unitary_rope(x, cos, sin)


class SphericalResidual(nn.Module):
    """Hyperspherical residual blending layer keeping state representations closed on S^{d-1}.

    Computes:
        psi_out = (x + sublayer_out) / ||x + sublayer_out||_2
    Optionally applies zero-mean Riemannian centering prior to projection to prevent
    anisotropic cone collapse.
    """

    def __init__(self, center: bool = False, eps: float = 1e-12) -> None:
        super().__init__()
        self.center = center
        self.eps = eps

    def forward(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        combined = x + delta
        if self.center:
            return riemannian_center(combined, dim=-1, eps=self.eps)
        return l2_normalize(combined, dim=-1, eps=self.eps)
