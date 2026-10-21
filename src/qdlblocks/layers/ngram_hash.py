"""Hashed Bigram + Trigram Embedding Table with Weinberger +-1 Sign Trick (Category B).

Implements Section 4.1.2 of EXPERIMENT_PLAN_021.md (`KellerJordan/modded-nanogpt` Record #360
`track_1_short/ngram_table.py`):
- `BIGRAM_ROW_MULS`, `TRIGRAM_ROW_MULS`, `BIGRAM_SIGN_MULS`, `TRIGRAM_SIGN_MULS`
- `compute_ngram_hashes_and_signs`: Vectorized 64-bit multiplicative-XOR row and sign-pool hashing.
- `HashedNgramEmbedding`: Compact GPU-resident bigram + trigram lookup table with pseudorandom
  `+-1` sign cancellation pool and per-layer injection scales.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BIGRAM_ROW_MULS: tuple[int, int] = (36313, 27191)
TRIGRAM_ROW_MULS: tuple[int, int, int] = (17351, 60961, 45259)
BIGRAM_SIGN_MULS: tuple[int, int] = (48271, 30011)
TRIGRAM_SIGN_MULS: tuple[int, int, int] = (58699, 39779, 26801)


def compute_ngram_hashes_and_signs(
    idx: torch.Tensor,
    ngram_vocab_size: int = 262144,
    sign_pool_rows: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute `(bigram_ids, trigram_ids, bigram_sign_ids, trigram_sign_ids)` for `idx` `(B, T)`.

    Args:
        idx: Integer token ID tensor of shape `(B, T)` on CPU or GPU.
        ngram_vocab_size: Total rows in the combined bigram + trigram table (must be >= 4).
        sign_pool_rows: Number of rows in the pseudorandom `+-1` sign pool (must be >= 2).

    Returns:
        Tuple `(bigram_ids, trigram_ids, bigram_sign_ids, trigram_sign_ids)` each of shape `(B, T)` (`int64`).
    """
    if idx.ndim != 2:
        raise ValueError(f"compute_ngram_hashes_and_signs expects 2D idx (B, T), got {tuple(idx.shape)}")
    if ngram_vocab_size < 4:
        raise ValueError(f"ngram_vocab_size must be >= 4, got {ngram_vocab_size}")
    if sign_pool_rows < 2:
        raise ValueError(f"sign_pool_rows must be >= 2, got {sign_pool_rows}")

    B, T = idx.shape
    dev = idx.device
    x = idx.to(torch.int64)

    half = int(ngram_vocab_size) // 2
    bigram_mod = max(1, half - 1)
    trigram_mod = max(1, int(ngram_vocab_size) - half - 1)
    is_pow2_sign = (sign_pool_rows & (sign_pool_rows - 1)) == 0

    bigram_ids = torch.full((B, T), bigram_mod, dtype=torch.int64, device=dev)
    trigram_ids = torch.full((B, T), half + trigram_mod, dtype=torch.int64, device=dev)
    bigram_sign_ids = torch.zeros((B, T), dtype=torch.int64, device=dev)
    trigram_sign_ids = torch.zeros((B, T), dtype=torch.int64, device=dev)

    if T >= 2:
        # Bigram row IDs: at t=1 prev_b is bigram_mod; for t >= 2 prev_b is x[:, t-1]
        prev_b = torch.empty((B, T - 1), dtype=torch.int64, device=dev)
        prev_b[:, 0] = bigram_mod
        if T >= 3:
            prev_b[:, 1:] = x[:, 1:-1]
        bi_hash = (BIGRAM_ROW_MULS[0] * x[:, 1:]) ^ (BIGRAM_ROW_MULS[1] * prev_b)
        bigram_ids[:, 1:] = torch.remainder(bi_hash, bigram_mod)

        # Bigram sign IDs: for t >= 1 uses (x[:, t], x[:, t-1])
        bi_sign_hash = (BIGRAM_SIGN_MULS[0] * x[:, 1:]) ^ (BIGRAM_SIGN_MULS[1] * x[:, :-1])
        if is_pow2_sign:
            bigram_sign_ids[:, 1:] = bi_sign_hash & (sign_pool_rows - 1)
        else:
            bigram_sign_ids[:, 1:] = torch.remainder(bi_sign_hash, sign_pool_rows)

    if T >= 3:
        tri_hash = (
            (TRIGRAM_ROW_MULS[0] * x[:, 2:])
            ^ (TRIGRAM_ROW_MULS[1] * x[:, 1:-1])
            ^ (TRIGRAM_ROW_MULS[2] * x[:, :-2])
        )
        trigram_ids[:, 2:] = torch.remainder(tri_hash, trigram_mod) + half

        tri_sign_hash = (
            (TRIGRAM_SIGN_MULS[0] * x[:, 2:])
            ^ (TRIGRAM_SIGN_MULS[1] * x[:, 1:-1])
            ^ (TRIGRAM_SIGN_MULS[2] * x[:, :-2])
        )
        if is_pow2_sign:
            trigram_sign_ids[:, 2:] = tri_sign_hash & (sign_pool_rows - 1)
        else:
            trigram_sign_ids[:, 2:] = torch.remainder(tri_sign_hash, sign_pool_rows)

    return bigram_ids, trigram_ids, bigram_sign_ids, trigram_sign_ids


class HashedNgramEmbedding(nn.Module):
    """Hashed Bigram + Trigram Embedding Table with Weinberger `+-1` Sign Pool."""

    sign_pool: torch.Tensor

    def __init__(
        self,
        ngram_vocab_size: int = 262144,
        embed_dim: int = 1024,
        num_layers: int = 12,
        sign_pool_rows: int = 8192,
        inject_scale_init: float = 0.05,
        zero_init: bool = True,
        seed: int = 1337,
    ) -> None:
        super().__init__()
        self.ngram_vocab_size = int(ngram_vocab_size)
        self.embed_dim = int(embed_dim)
        self.num_layers = max(1, int(num_layers))
        self.sign_pool_rows = int(sign_pool_rows)
        self.inject_scale_init = float(inject_scale_init)
        self.zero_init = bool(zero_init)
        self.seed = int(seed)

        self.table = nn.Embedding(self.ngram_vocab_size, self.embed_dim)
        if self.zero_init:
            nn.init.zeros_(self.table.weight)
        else:
            nn.init.normal_(self.table.weight, mean=0.0, std=1.0 / math.sqrt(self.embed_dim))

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)
        signs = torch.randint(
            0, 2, (self.sign_pool_rows, self.embed_dim), generator=gen, dtype=torch.int8
        )
        sign_pool = (signs.to(torch.float32) * 2.0) - 1.0
        self.register_buffer("sign_pool", sign_pool, persistent=False)

        self.layer_scales = nn.Parameter(
            torch.full((self.num_layers,), self.inject_scale_init, dtype=torch.float32)
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Compute sign-modulated bigram + trigram embeddings `(B, T, embed_dim)` for `idx` `(B, T)`."""
        bigram_ids, trigram_ids, bigram_sign_ids, trigram_sign_ids = compute_ngram_hashes_and_signs(
            idx,
            ngram_vocab_size=self.ngram_vocab_size,
            sign_pool_rows=self.sign_pool_rows,
        )
        w = self.table.weight
        sp = self.sign_pool.to(device=w.device, dtype=w.dtype)
        h_bi = F.embedding(bigram_ids, w) * F.embedding(bigram_sign_ids, sp)
        h_tri = F.embedding(trigram_ids, w) * F.embedding(trigram_sign_ids, sp)
        return h_bi + h_tri


__all__ = [
    "BIGRAM_ROW_MULS",
    "TRIGRAM_ROW_MULS",
    "BIGRAM_SIGN_MULS",
    "TRIGRAM_SIGN_MULS",
    "compute_ngram_hashes_and_signs",
    "HashedNgramEmbedding",
]
