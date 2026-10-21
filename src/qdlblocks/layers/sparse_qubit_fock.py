"""64-Qubit Sparse Fock Excitation Space (H_64^{(<=2)}), Sparse Hashed 1/2/3-Gram Qubit Encodings, & 4M-Parameter Internal Density Matrix Layers.

Implements Section 3.1 of EXPERIMENT_PLAN_022.md:
1. `build_64qubit_pair_index_tables`: Deterministic two-qubit pair index tables `(pair_i, pair_j)`
   for `0 <= i < j < num_qubits` mapped into Sector 2 (`num_qubits:fock_dim`).
2. `Sparse64QubitNgramEmbedder`: Compact 64-qubit unigram amplitude table (`50,304 x 64 = 3.22M` params)
   + sparse unigram pair excitations + sparse hashed bigram/trigram excitation table (`262,144 x 16 = 4.19M`
   params) with Weinberger `+-1` sign cancellation + explicit 2-qubit and 3-qubit partial-trace pair
   correlations in `D_Fock = 2048`.
3. `Sparse64QubitDensityLayer`: Internal 64-qubit density-matrix layer with an exact `2048 x 2048 = 4,194,304`
   (`4.19M`) parameter Stiefel/Hermitian channel operator, two-body `64 <-> 1984` creation/annihilation
   Fock coupler, and Rank-`R` mixed density-matrix branch expansion (`rho_t = sum_{r=1}^R p_{t,r} |psi_{t,r}><psi_{t,r}|`).
4. `Sparse64QubitPOVMReadout`: Born density-matrix trace readout `Tr(rho_t M_w)` across rank-`R` eigen-branches
   with purity-modulated inverse temperature.
5. `initialize_sparse_64qubit_with_ppmi_svd`: Initializes `Sparse64QubitNgramEmbedder` and POVM readout effects
   from a Shifted PPMI + Randomized SVD spectral sketch with exact even/odd Fock coordinate orthogonal decoupling
   (`max_w |<M_w | psi_0(w)>| < 1e-6` and Step-0 collocation cosine ratio `>= 3.0x`).
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import l2_normalize
from .cooccurrence_init import _adapt_ppmi_shape, compute_ppmi_svd_embeddings
from .ngram_hash import (
    BIGRAM_ROW_MULS,
    BIGRAM_SIGN_MULS,
    TRIGRAM_ROW_MULS,
    TRIGRAM_SIGN_MULS,
)

UNIGRAM_PAIR_PRIME_TOK: int = 1640531527
UNIGRAM_PAIR_PRIME_K: int = 1597334677
NGRAM_COORD_PRIME_ROW: int = 1140071481
NGRAM_COORD_STRIDE_K: int = 131


def build_64qubit_pair_index_tables(
    num_qubits: int = 64,
    max_pair_modes: int = 1984,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct deterministic `int64` index tables `(pair_i, pair_j)` for two-qubit excitations `0 <= i < j < num_qubits`.

    For `num_qubits = 64`, there are `comb(64, 2) = 2016` unique two-qubit pairs. When `max_pair_modes = 1984`
    (so `64 + 1984 = 2048 = D_Fock`), this selects the first `1984` pairs covering all 64 qubits; if
    `max_pair_modes > comb(num_qubits, 2)`, it cycles deterministically.
    """
    if num_qubits < 2:
        raise ValueError(f"num_qubits must be >= 2, got {num_qubits}")
    if max_pair_modes < 1:
        raise ValueError(f"max_pair_modes must be >= 1, got {max_pair_modes}")

    tri = torch.triu_indices(num_qubits, num_qubits, offset=1, dtype=torch.int64)
    total_pairs = int(tri.size(1))
    if max_pair_modes <= total_pairs:
        # Interleave across rows so all qubits 0..num_qubits-1 participate evenly even when truncated
        perm = (torch.arange(max_pair_modes, dtype=torch.int64) * 17) % total_pairs
        if math.gcd(17, total_pairs) != 1:
            perm = torch.arange(max_pair_modes, dtype=torch.int64)
        pair_i = tri[0, perm].contiguous()
        pair_j = tri[1, perm].contiguous()
    else:
        idx = torch.arange(max_pair_modes, dtype=torch.int64) % total_pairs
        pair_i = tri[0, idx].contiguous()
        pair_j = tri[1, idx].contiguous()
    return pair_i, pair_j


class Sparse64QubitNgramEmbedder(nn.Module):
    """Encodes token IDs `(B, T)` into the 64-qubit sparse `<=2`-excitation Fock space `S^{D_Fock - 1}` (`D_Fock = 2048`)."""

    pair_i: torch.Tensor
    pair_j: torch.Tensor
    k_offsets: torch.Tensor
    sign_pool: torch.Tensor

    def __init__(
        self,
        vocab_size: int = 50304,
        fock_dim: int = 2048,
        num_qubits: int = 64,
        sparse_k: int = 16,
        ngram_vocab_size: int = 262144,
        num_layers: int = 8,
        sign_pool_rows: int = 8192,
        inject_scale_init: float = 0.15,
        unigram_pair_scale_init: float = 0.05,
        eps: float = 1e-12,
        seed: int = 1337,
    ) -> None:
        super().__init__()
        if fock_dim < num_qubits:
            raise ValueError(f"fock_dim ({fock_dim}) must be >= num_qubits ({num_qubits})")
        if sparse_k < 1:
            raise ValueError(f"sparse_k must be >= 1, got {sparse_k}")

        self.vocab_size = int(vocab_size)
        self.fock_dim = int(fock_dim)
        self.num_qubits = int(num_qubits)
        self.pair_modes = max(0, self.fock_dim - self.num_qubits)
        self.sparse_k = int(sparse_k)
        self.ngram_vocab_size = max(4, int(ngram_vocab_size))
        self.num_layers = max(1, int(num_layers))
        self.sign_pool_rows = max(2, int(sign_pool_rows))
        self.inject_scale_init = float(inject_scale_init)
        self.unigram_pair_scale_init = float(unigram_pair_scale_init)
        self.eps = float(eps)
        self.seed = int(seed)

        # 1. 64-Qubit Unigram Single-Excitation Table: 50,304 x 64 = 3,219,456 (3.22M) parameters
        self.qubit_wte = nn.Embedding(self.vocab_size, self.num_qubits)
        nn.init.normal_(self.qubit_wte.weight, mean=0.0, std=1.0 / math.sqrt(max(1, self.num_qubits)))

        # 2. Sparse Unigram Pair Excitation Amplitudes: 50,304 x K parameters
        self.unigram_sparse_vals = nn.Embedding(self.vocab_size, self.sparse_k)
        nn.init.normal_(self.unigram_sparse_vals.weight, mean=0.0, std=1.0 / math.sqrt(max(1, self.sparse_k)))
        self.unigram_pair_scale = nn.Parameter(
            torch.tensor(self.unigram_pair_scale_init, dtype=torch.float32)
        )

        # 3. Sparse Hashed Bigram + Trigram Qubit Excitation Table: 262,144 x 16 = 4,194,304 (4.19M) parameters
        self.ngram_sparse_vals = nn.Embedding(self.ngram_vocab_size, self.sparse_k)
        nn.init.normal_(self.ngram_sparse_vals.weight, mean=0.0, std=1.0 / math.sqrt(max(1, self.sparse_k)))

        # 4. Learnable 2-Qubit & 3-Qubit Partial-Trace Pair Coupling Scales + Per-Layer N-Gram Scales
        self.pair_scale_2gram = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.pair_scale_3gram = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))
        self.layer_scales = nn.Parameter(
            torch.full((self.num_layers,), self.inject_scale_init, dtype=torch.float32)
        )

        # 5. Deterministic Pair Tables & Weinberger +-1 Sign Pool
        pair_i, pair_j = build_64qubit_pair_index_tables(
            num_qubits=max(2, self.num_qubits),
            max_pair_modes=max(1, self.pair_modes),
        )
        self.register_buffer("pair_i", pair_i, persistent=False)
        self.register_buffer("pair_j", pair_j, persistent=False)
        self.register_buffer(
            "k_offsets",
            torch.arange(1, self.sparse_k + 1, dtype=torch.int64),
            persistent=False,
        )

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)
        signs = torch.randint(
            0, 2, (self.sign_pool_rows, self.sparse_k), generator=gen, dtype=torch.int8
        )
        sign_pool = (signs.to(torch.float32) * 2.0) - 1.0
        self.register_buffer("sign_pool", sign_pool, persistent=False)

    def _compute_ngram_ids_and_signs(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute `(bigram_ids, trigram_ids, bigram_sign_ids, trigram_sign_ids)` for `idx: (B, T)`."""
        B, T = idx.shape
        dev = idx.device
        x = idx.to(torch.int64)

        half = self.ngram_vocab_size // 2
        bigram_mod = max(1, half - 1)
        trigram_mod = max(1, self.ngram_vocab_size - half - 1)
        is_pow2_sign = (self.sign_pool_rows & (self.sign_pool_rows - 1)) == 0

        bigram_ids = torch.full((B, T), bigram_mod, dtype=torch.int64, device=dev)
        trigram_ids = torch.full((B, T), half + trigram_mod, dtype=torch.int64, device=dev)
        bigram_sign_ids = torch.zeros((B, T), dtype=torch.int64, device=dev)
        trigram_sign_ids = torch.zeros((B, T), dtype=torch.int64, device=dev)

        if T >= 2:
            bi_hash = (BIGRAM_ROW_MULS[0] * x[:, 1:]) ^ (BIGRAM_ROW_MULS[1] * x[:, :-1])
            bigram_ids[:, 1:] = torch.remainder(bi_hash, bigram_mod)
            bi_sign_hash = (BIGRAM_SIGN_MULS[0] * x[:, 1:]) ^ (BIGRAM_SIGN_MULS[1] * x[:, :-1])
            if is_pow2_sign:
                bigram_sign_ids[:, 1:] = bi_sign_hash & (self.sign_pool_rows - 1)
            else:
                bigram_sign_ids[:, 1:] = torch.remainder(bi_sign_hash, self.sign_pool_rows)

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
                trigram_sign_ids[:, 2:] = tri_sign_hash & (self.sign_pool_rows - 1)
            else:
                trigram_sign_ids[:, 2:] = torch.remainder(tri_sign_hash, self.sign_pool_rows)

        return bigram_ids, trigram_ids, bigram_sign_ids, trigram_sign_ids

    def forward_ngram_delta(self, idx: torch.Tensor) -> torch.Tensor:
        """Compute the sparse hashed bigram + trigram excitation vector `(B, T, fock_dim)` with Weinberger `+-1` signs."""
        B, T = idx.shape
        bigram_ids, trigram_ids, bigram_sign_ids, trigram_sign_ids = self._compute_ngram_ids_and_signs(idx)

        w = self.ngram_sparse_vals.weight
        sp = self.sign_pool.to(device=w.device, dtype=w.dtype)
        k_off = self.k_offsets.view(1, 1, self.sparse_k)

        out = torch.zeros((B, T, self.fock_dim), device=w.device, dtype=w.dtype)
        if T >= 2:
            bi_vals = F.embedding(bigram_ids, w) * F.embedding(bigram_sign_ids, sp)  # (B, T, K)
            # Zero out t=0 where no bigram exists
            bi_mask = torch.ones((1, T, 1), device=w.device, dtype=w.dtype)
            bi_mask[:, 0, :] = 0.0
            bi_vals = bi_vals * bi_mask
            bi_base = torch.remainder(
                bigram_ids.unsqueeze(-1) * NGRAM_COORD_PRIME_ROW + bigram_sign_ids.unsqueeze(-1) * 31,
                self.fock_dim,
            )
            bi_coords = torch.remainder(bi_base + k_off * NGRAM_COORD_STRIDE_K, self.fock_dim)
            out.scatter_add_(2, bi_coords, bi_vals)

        if T >= 3:
            tri_vals = F.embedding(trigram_ids, w) * F.embedding(trigram_sign_ids, sp)  # (B, T, K)
            tri_mask = torch.ones((1, T, 1), device=w.device, dtype=w.dtype)
            tri_mask[:, :2, :] = 0.0
            tri_vals = tri_vals * tri_mask
            tri_base = torch.remainder(
                trigram_ids.unsqueeze(-1) * NGRAM_COORD_PRIME_ROW + trigram_sign_ids.unsqueeze(-1) * 37,
                self.fock_dim,
            )
            tri_coords = torch.remainder(tri_base + k_off * NGRAM_COORD_STRIDE_K, self.fock_dim)
            out.scatter_add_(2, tri_coords, tri_vals)

        return out

    def forward_unigram_fock(self, idx: torch.Tensor) -> torch.Tensor:
        """Encode single tokens `idx: (B, T)` or `(N,)` into Sector 1 (`0:num_qubits`) + sparse even-coordinate pair excitations."""
        squeeze_1d = idx.ndim == 1
        if squeeze_1d:
            idx = idx.unsqueeze(1)
        B, T = idx.shape
        u_t = l2_normalize(self.qubit_wte(idx), dim=-1, eps=self.eps)  # (B, T, num_qubits)
        if self.pair_modes <= 0:
            return u_t.squeeze(1) if squeeze_1d else u_t

        x = idx.to(torch.int64)
        k_off = self.k_offsets.view(1, 1, self.sparse_k)
        uni_raw = self.unigram_sparse_vals(idx)  # (B, T, K)
        uni_sign_hash = ((x.unsqueeze(-1) * 48271) ^ (k_off * 30011)) & 1
        uni_signs = (uni_sign_hash.to(uni_raw.dtype) * 2.0) - 1.0
        uni_vals = uni_raw * uni_signs * self.unigram_pair_scale.to(uni_raw.dtype)

        # Scatter into even pair coordinates in Sector 2 so Step-0 orthogonal decoupling on odd coordinates is exact
        half_pairs = max(1, self.pair_modes // 2)
        pair_base = torch.remainder(x.unsqueeze(-1) * UNIGRAM_PAIR_PRIME_TOK, half_pairs)
        pair_half_idx = torch.remainder(pair_base + k_off * 67, half_pairs)
        pair_coords = (pair_half_idx * 2).clamp(0, self.pair_modes - 1)

        sector2 = torch.zeros((B, T, self.pair_modes), device=u_t.device, dtype=u_t.dtype)
        sector2.scatter_add_(2, pair_coords, uni_vals)
        out = l2_normalize(torch.cat([u_t, sector2], dim=-1), dim=-1, eps=self.eps)
        return out.squeeze(1) if squeeze_1d else out

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Encode token IDs `idx: (B, T)` into a unit-norm 64-qubit sparse Fock state `psi_0: (B, T, fock_dim)`."""
        B, T = idx.shape
        u_t = l2_normalize(self.qubit_wte(idx), dim=-1, eps=self.eps)  # (B, T, num_qubits)
        if self.pair_modes <= 0:
            psi_raw = u_t
        else:
            # 1. Unigram sparse pair excitations in Sector 2
            x = idx.to(torch.int64)
            k_off = self.k_offsets.view(1, 1, self.sparse_k)
            uni_raw = self.unigram_sparse_vals(idx)
            uni_sign_hash = ((x.unsqueeze(-1) * 48271) ^ (k_off * 30011)) & 1
            uni_signs = (uni_sign_hash.to(uni_raw.dtype) * 2.0) - 1.0
            uni_vals = uni_raw * uni_signs * self.unigram_pair_scale.to(uni_raw.dtype)

            half_pairs = max(1, self.pair_modes // 2)
            pair_base = torch.remainder(x.unsqueeze(-1) * UNIGRAM_PAIR_PRIME_TOK, half_pairs)
            pair_half_idx = torch.remainder(pair_base + k_off * 67, half_pairs)
            pair_coords = (pair_half_idx * 2).clamp(0, self.pair_modes - 1)

            sector2 = torch.zeros((B, T, self.pair_modes), device=u_t.device, dtype=u_t.dtype)
            sector2.scatter_add_(2, pair_coords, uni_vals)

            # 2. Explicit 2-Qubit (Bigram) and 3-Qubit Partial-Trace (Trigram) Pair Excitations
            if T >= 2:
                u_prev1 = F.pad(u_t[:, :-1, :], (0, 0, 1, 0))
                u_ti = u_t[..., self.pair_i]
                u_tj = u_t[..., self.pair_j]
                u_p1i = u_prev1[..., self.pair_i]
                u_p1j = u_prev1[..., self.pair_j]
                g2 = self.pair_scale_2gram.to(u_t.dtype)
                pair_corr = g2 * (u_ti * u_p1j + u_p1i * u_tj)
                if T >= 3:
                    u_prev2 = F.pad(u_t[:, :-2, :], (0, 0, 2, 0))
                    u_p2i = u_prev2[..., self.pair_i]
                    u_p2j = u_prev2[..., self.pair_j]
                    g3 = self.pair_scale_3gram.to(u_t.dtype)
                    pair_corr = pair_corr + g3 * (u_ti * u_p2j + u_p2i * u_tj + u_p1i * u_p2j)
                sector2 = sector2 + pair_corr

            psi_raw = torch.cat([u_t, sector2], dim=-1)

        if T >= 2:
            h_ngram = self.forward_ngram_delta(idx)
            s0 = self.layer_scales[0].to(psi_raw.dtype)
            psi_raw = psi_raw + s0 * h_ngram.to(psi_raw.dtype)

        return l2_normalize(psi_raw, dim=-1, eps=self.eps)


class Sparse64QubitDensityLayer(nn.Module):
    """64-Qubit Internal Density Matrix Layer (`2048 x 2048 = 4.19M` params) + Two-Body Fock Coupler & Rank-`R` Mixed State."""

    pair_i: torch.Tensor
    pair_j: torch.Tensor

    def __init__(
        self,
        fock_dim: int = 2048,
        num_qubits: int = 64,
        density_rank: int = 4,
        use_fock_coupler: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if fock_dim < 2:
            raise ValueError(f"fock_dim must be >= 2, got {fock_dim}")
        self.fock_dim = int(fock_dim)
        self.num_qubits = min(int(num_qubits), self.fock_dim)
        self.pair_modes = max(0, self.fock_dim - self.num_qubits)
        self.density_rank = max(1, int(density_rank))
        self.use_fock_coupler = bool(use_fock_coupler) and (self.pair_modes > 0)
        self.eps = float(eps)

        # 1. Core 4.19M-Parameter (2048 x 2048 = 4,194,304) Internal Density Matrix / Channel Operator
        self.density_operator = nn.Linear(self.fock_dim, self.fock_dim, bias=False)
        self._init_parity_coupled_orthogonal(self.density_operator.weight, theta=0.85, mix_eps=0.10)
        self.eig_weights = nn.Parameter(torch.ones(self.fock_dim, dtype=torch.float32))
        self.mix_gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # 2. Two-Body 64-Qubit Creation & Partial-Trace Annihilation Fock Coupler (64 <-> 1984)
        pair_i, pair_j = build_64qubit_pair_index_tables(
            num_qubits=max(2, self.num_qubits),
            max_pair_modes=max(1, self.pair_modes),
        )
        self.register_buffer("pair_i", pair_i, persistent=False)
        self.register_buffer("pair_j", pair_j, persistent=False)

        if self.use_fock_coupler:
            self.qubit_unitary: nn.Linear | None = nn.Linear(self.num_qubits, self.num_qubits, bias=False)
            self._init_parity_coupled_orthogonal(self.qubit_unitary.weight, theta=0.95, mix_eps=0.12)
            self.coupler_create_gain: nn.Parameter | None = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
            self.coupler_trace_gain: nn.Parameter | None = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        else:
            self.qubit_unitary = None
            self.register_parameter("coupler_create_gain", None)
            self.register_parameter("coupler_trace_gain", None)

        # 3. Rank-R Mixed Density Matrix Branch Expansion & Contextual Purity Router
        init_filters = torch.ones((self.density_rank, self.fock_dim), dtype=torch.float32)
        if self.density_rank > 1:
            coord_grid = torch.linspace(0.0, math.pi, self.fock_dim, dtype=torch.float32)
            for r in range(1, self.density_rank):
                init_filters[r] = 1.0 + 0.35 * torch.cos(coord_grid * float(r))
        self.branch_spectral_filters = nn.Parameter(init_filters)

        self.purity_router = nn.Linear(self.num_qubits, self.density_rank, bias=True)
        nn.init.normal_(self.purity_router.weight, mean=0.0, std=0.08)
        with torch.no_grad():
            if self.density_rank == 1:
                self.purity_router.bias.zero_()
            elif self.density_rank == 4:
                # Target initial mixture probabilities ~ [0.45, 0.25, 0.18, 0.12] -> Purity Tr(rho^2) ~ 0.312
                prior_p = torch.tensor([0.45, 0.25, 0.18, 0.12], dtype=torch.float32)
                self.purity_router.bias.copy_(torch.log(prior_p))
            else:
                ranks = torch.arange(1, self.density_rank + 1, dtype=torch.float32)
                prior_p = (1.0 / ranks) / (1.0 / ranks).sum()
                self.purity_router.bias.copy_(torch.log(prior_p))

        self.last_purity_mean: float = 1.0 if self.density_rank == 1 else 0.3122
        self.last_pair_sector_energy: float = 0.0
        self._capture_branches: bool = False
        self._last_branches: torch.Tensor | None = None
        self._last_p_weights: torch.Tensor | None = None

    @staticmethod
    @torch.no_grad()
    def _init_parity_coupled_orthogonal(weight: torch.Tensor, theta: float = 0.85, mix_eps: float = 0.10) -> None:
        """Initialize `weight` on `O(D)` with 2x2 even-odd parity rotation composed with a random orthogonal perturbation."""
        d = int(weight.size(0))
        if weight.ndim != 2 or weight.size(1) != d:
            nn.init.orthogonal_(weight)
            return
        q_rand = torch.empty((d, d), dtype=torch.float32, device=weight.device)
        nn.init.orthogonal_(q_rand)
        # Construct exact 2x2 block rotation J on (2m, 2m+1)
        j_mat = torch.eye(d, dtype=torch.float32, device=weight.device)
        c = math.cos(theta)
        s = math.sin(theta)
        n_pairs = d // 2
        if n_pairs > 0:
            ev = torch.arange(0, 2 * n_pairs, 2, device=weight.device)
            od = ev + 1
            j_mat[ev, ev] = c
            j_mat[od, od] = c
            j_mat[od, ev] = s
            j_mat[ev, od] = -s
        # Blend J with q_rand and QR-retract onto O(D)
        blended = (1.0 - mix_eps) * j_mat + mix_eps * q_rand
        q_orth, r_orth = torch.linalg.qr(blended)
        diag_sign = torch.sign(torch.diagonal(r_orth))
        diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
        q_orth = q_orth * diag_sign.unsqueeze(0)
        weight.copy_(q_orth.to(weight.dtype))

    def _apply_fock_coupler(self, psi: torch.Tensor) -> torch.Tensor:
        """Couple Sector 1 (`0:num_qubits`) and Sector 2 (`num_qubits:fock_dim`) via two-body creation & partial trace."""
        if not self.use_fock_coupler or self.qubit_unitary is None or self.pair_modes <= 0:
            return psi

        q = psi[..., : self.num_qubits]  # (B, T, 64)
        P = psi[..., self.num_qubits :]  # (B, T, 1984)
        q_tilde = self.qubit_unitary(q)  # (B, T, 64)

        # Two-body creation: (q_i * q_tilde_j + q_j * q_tilde_i) / sqrt(num_qubits)
        inv_sqrt_q = 1.0 / math.sqrt(float(max(1, self.num_qubits)))
        delta_P = inv_sqrt_q * (
            q[..., self.pair_i] * q_tilde[..., self.pair_j]
            + q[..., self.pair_j] * q_tilde[..., self.pair_i]
        )

        # Partial-trace annihilation: sum_{m: i_m = i} P_m * q_tilde_{j_m}
        inv_sqrt_p = 1.0 / math.sqrt(float(max(1, self.pair_modes)))
        contrib_i = (P * q_tilde[..., self.pair_j]) * inv_sqrt_p
        contrib_j = (P * q_tilde[..., self.pair_i]) * inv_sqrt_p

        delta_q = torch.zeros_like(q)
        shape_View = [1] * (psi.ndim - 1) + [self.pair_modes]
        idx_i = self.pair_i.view(*shape_View).expand_as(P)
        idx_j = self.pair_j.view(*shape_View).expand_as(P)
        delta_q.scatter_add_(-1, idx_i, contrib_i)
        delta_q.scatter_add_(-1, idx_j, contrib_j)

        assert self.coupler_trace_gain is not None and self.coupler_create_gain is not None
        q_out = q + self.coupler_trace_gain.to(psi.dtype) * delta_q
        P_out = P + self.coupler_create_gain.to(psi.dtype) * delta_P
        return l2_normalize(torch.cat([q_out, P_out], dim=-1), dim=-1, eps=self.eps)

    def forward(
        self,
        psi: torch.Tensor,
        return_branches: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evolve `psi: (B, T, fock_dim)` through the 64-qubit Fock coupler, 4.19M density operator, and rank-`R` mixture."""
        psi_coupled = self._apply_fock_coupler(psi)
        eig_w = self.eig_weights.to(psi.dtype)
        h_rho = l2_normalize(self.density_operator(psi_coupled * eig_w), dim=-1, eps=self.eps)

        alpha_mix = torch.sigmoid(self.mix_gate).to(psi.dtype)
        psi_base = l2_normalize(
            (1.0 - alpha_mix) * psi_coupled + alpha_mix * h_rho,
            dim=-1,
            eps=self.eps,
        )

        filters = self.branch_spectral_filters.to(psi.dtype)  # (R, fock_dim)
        if self.density_rank == 1:
            branches = l2_normalize(psi_base.unsqueeze(-2) * filters.view(1, 1, 1, -1), dim=-1, eps=self.eps)
            p_weights = torch.ones((*psi.shape[:-1], 1), device=psi.device, dtype=psi.dtype)
            psi_out = branches.squeeze(-2)
        else:
            # Construct R distinct eigen-branches from spectral filter modulations of (psi_base, h_rho, psi_coupled)
            branch_list: list[torch.Tensor] = []
            sources = (
                psi_base,
                h_rho,
                psi_coupled,
                l2_normalize(F.silu(h_rho) + 0.5 * psi_coupled, dim=-1, eps=self.eps),
            )
            for r in range(self.density_rank):
                src_r = sources[r % len(sources)]
                f_r = filters[r].view(*([1] * (psi.ndim - 1)), self.fock_dim)
                branch_list.append(l2_normalize(src_r * f_r, dim=-1, eps=self.eps))
            branches = torch.stack(branch_list, dim=-2)  # (B, T, R, fock_dim)

            q_summary = psi_base[..., : self.num_qubits]
            router_logits = self.purity_router(q_summary)
            p_weights = F.softmax(router_logits, dim=-1)  # (B, T, R)
            psi_out = l2_normalize(
                (branches * p_weights.unsqueeze(-1)).sum(dim=-2),
                dim=-1,
                eps=self.eps,
            )

        if self._capture_branches and not torch.compiler.is_compiling():
            self._last_branches = branches
            self._last_p_weights = p_weights
        elif not torch.compiler.is_compiling():
            self._last_branches = None
            self._last_p_weights = None

        if not torch.compiler.is_compiling():
            with torch.no_grad():
                purity_t = (p_weights.float().square()).sum(dim=-1)
                self.last_purity_mean = float(purity_t.mean().item())
                if self.pair_modes > 0:
                    pair_e = (psi_out[..., self.num_qubits :].float().square()).sum(dim=-1)
                    self.last_pair_sector_energy = float(pair_e.mean().item())
                else:
                    self.last_pair_sector_energy = 0.0

        if return_branches:
            return psi_out, branches, p_weights
        return psi_out


class Sparse64QubitPOVMReadout(nn.Module):
    """Born density-matrix trace readout `Tr(rho_t M_w)` across rank-`R` 64-qubit Fock eigen-branches."""

    def __init__(
        self,
        vocab_size: int = 50304,
        fock_dim: int = 2048,
        num_qubits: int = 64,
        povm_temperature: float = 0.035,
        purity_temp_gain: float = 0.25,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.fock_dim = int(fock_dim)
        self.num_qubits = int(num_qubits)
        self.eps = float(eps)
        self.inv_temp = nn.Parameter(torch.tensor(1.0 / max(povm_temperature, 1e-4), dtype=torch.float32))
        self.purity_temp_gain = nn.Parameter(torch.tensor(float(purity_temp_gain), dtype=torch.float32))
        self.povm_odd_delta = nn.Parameter(
            torch.zeros((self.vocab_size, max(1, self.num_qubits // 2)), dtype=torch.float32)
        )

    def get_qubit_povm_matrix(self, embedder: Sparse64QubitNgramEmbedder) -> torch.Tensor:
        """Construct the compact `(vocab_size, num_qubits)` odd-coordinate single-qubit POVM detector matrix."""
        q_w = embedder.qubit_wte.weight  # (V, num_qubits)
        _, num_q = q_w.shape
        q_half = max(1, num_q // 2)
        q_out = torch.zeros_like(q_w)
        q_out[:, 1 : 2 * q_half : 2] = q_w[:, 0 : 2 * q_half : 2] + self.povm_odd_delta[:, :q_half].to(q_w.dtype)
        q_out[:, 0 : 2 * q_half : 2] = q_w[:, 1 : 2 * q_half : 2]
        return l2_normalize(q_out, dim=-1, eps=self.eps)

    def get_povm_matrix(self, embedder: Sparse64QubitNgramEmbedder) -> torch.Tensor:
        """Construct the `(vocab_size, fock_dim)` POVM effect matrix on odd Fock coordinates orthogonal to `psi_0(w)`."""
        q_w = embedder.qubit_wte.weight  # (V, num_qubits)
        V, num_q = q_w.shape
        q_out = self.get_qubit_povm_matrix(embedder)

        pair_modes = self.fock_dim - num_q
        if pair_modes <= 0:
            return q_out

        v_ids = torch.arange(V, device=q_w.device, dtype=torch.int64)
        k_off = embedder.k_offsets.view(1, embedder.sparse_k)
        uni_raw = embedder.unigram_sparse_vals.weight  # (V, K)
        uni_sign_hash = ((v_ids.unsqueeze(-1) * 48271) ^ (k_off * 30011)) & 1
        uni_signs = (uni_sign_hash.to(uni_raw.dtype) * 2.0) - 1.0
        uni_vals = uni_raw * uni_signs * embedder.unigram_pair_scale.to(uni_raw.dtype)

        half_pairs = max(1, pair_modes // 2)
        pair_base = torch.remainder(v_ids.unsqueeze(-1) * UNIGRAM_PAIR_PRIME_TOK, half_pairs)
        pair_half_idx = torch.remainder(pair_base + k_off * 67, half_pairs)
        odd_pair_coords = (pair_half_idx * 2 + 1).clamp(0, pair_modes - 1)

        sector2_odd = torch.zeros((V, pair_modes), device=q_w.device, dtype=q_w.dtype)
        sector2_odd.scatter_add_(1, odd_pair_coords, uni_vals)
        return l2_normalize(torch.cat([q_out, sector2_odd], dim=-1), dim=-1, eps=self.eps)

    def compute_effective_state_and_beta(
        self,
        psi: torch.Tensor,
        branches: torch.Tensor | None = None,
        p_weights: torch.Tensor | None = None,
        base_inv_temp: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the density-matrix expectation state `phi_rho: (B, T, fock_dim)` and purity-modulated `beta_t`."""
        inv_t_src = self.inv_temp if base_inv_temp is None else base_inv_temp
        if isinstance(inv_t_src, torch.Tensor):
            beta_base = torch.clamp(inv_t_src, min=1.0, max=60.0)
        else:
            beta_base = torch.tensor(float(min(max(inv_t_src, 1.0), 60.0)), device=psi.device, dtype=torch.float32)

        if branches is not None and p_weights is not None and branches.size(-2) > 1:
            R = branches.size(-2)
            branch_mix = l2_normalize((branches * p_weights.unsqueeze(-1)).sum(dim=-2), dim=-1, eps=self.eps)
            phi_rho = l2_normalize(psi + 0.08 * branch_mix, dim=-1, eps=self.eps)
            purity_t = (p_weights.float().square()).sum(dim=-1, keepdim=True)  # (B, T, 1) in [1/R, 1]
            min_p = 1.0 / float(R)
            norm_purity = (purity_t - min_p) / max(1e-6, 1.0 - min_p)
            gain = torch.tanh(self.purity_temp_gain.float())
            beta_t = torch.clamp(beta_base.float() * (1.0 + 0.25 * gain * (norm_purity - 0.5)), min=1.0, max=60.0)
            return phi_rho, beta_t

        phi_rho = l2_normalize(psi, dim=-1, eps=self.eps)
        return phi_rho, beta_base

    def forward(
        self,
        psi: torch.Tensor,
        effects: torch.Tensor,
        branches: torch.Tensor | None = None,
        p_weights: torch.Tensor | None = None,
        base_inv_temp: torch.Tensor | float | None = None,
        log_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute vocabulary logits `(B, T, vocab_size)` under the Born density-matrix trace rule."""
        phi_rho, beta_t = self.compute_effective_state_and_beta(
            psi, branches=branches, p_weights=p_weights, base_inv_temp=base_inv_temp
        )
        M_norm = l2_normalize(effects, dim=-1, eps=self.eps).to(phi_rho.dtype)
        cos_sim = F.linear(phi_rho, M_norm)
        logits = cos_sim * beta_t.to(phi_rho.dtype)
        if log_prior is not None:
            logits = logits + log_prior.to(logits.dtype)
        return logits


@torch.no_grad()
def initialize_sparse_64qubit_with_ppmi_svd(
    model: nn.Module,
    ppmi_embeddings: torch.Tensor | None = None,
    alpha: float = 0.75,
    seed: int = 42,
    *,
    token_shards: Sequence[np.ndarray | torch.Tensor] | None = None,
    window: int = 5,
    alpha_ppmi: float | None = None,
    cache_path: str | Path | None = None,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Initialize `Sparse64QubitNgramEmbedder` and POVM readout effects with a 64-Qubit PPMI-SVD spectral sketch.

    Uses even/odd coordinate orthogonal decoupling so:
    1. Input single-token 64-qubit Fock states `psi_0(w)` live strictly on even coordinates `0::2`
       (carrying the top PPMI-SVD singular vectors on `0:num_qubits:2`),
    2. Output POVM detector effects `M_w` live strictly on odd coordinates `1::2`,
    3. Step-0 self-repetition overlap is identically zero (`max_w |<M_w | psi_0(w)>| < 1e-6`), and
    4. Step-0 collocation cosine similarity ratio in the 2048-D Fock space is `>= 3.0x` over random pairs.
    """
    from .cooccurrence_init import compute_collocation_coherence_metrics

    eff_alpha = float(alpha_ppmi if alpha_ppmi is not None else alpha)
    alpha_f = float(min(max(eff_alpha, 0.0), 1.0))

    embedder: Sparse64QubitNgramEmbedder | None = None
    if isinstance(model, Sparse64QubitNgramEmbedder):
        embedder = model
    elif isinstance(getattr(model, "sparse_qubit_embed", None), Sparse64QubitNgramEmbedder):
        embedder = getattr(model, "sparse_qubit_embed")

    first_param = next(model.parameters())
    dev = torch.device(device) if device is not None else first_param.device

    if embedder is not None:
        V = int(embedder.vocab_size)
        num_q = int(embedder.num_qubits)
        fock_dim = int(embedder.fock_dim)
    else:
        V = int(getattr(model, "vocab_size", 50304))
        num_q = int(getattr(model, "num_qubits", 64))
        fock_dim = int(getattr(model, "embed_dim", 2048))

    q_half = max(1, num_q // 2)
    if ppmi_embeddings is None:
        if token_shards is None:
            raise ValueError("Either `ppmi_embeddings` or `token_shards` must be provided")
        ppmi_res = compute_ppmi_svd_embeddings(
            token_arrays=token_shards,
            vocab_size=V,
            embed_dim=max(q_half, 64),
            window_size=int(window),
            seed=seed,
            device=dev,
            cache_path=cache_path,
        )
        ppmi_embeddings = ppmi_res["embeddings"]

    emb_vq = _adapt_ppmi_shape(ppmi_embeddings, target_vocab=V, target_dim=q_half, device=dev)
    active_mask = emb_vq.norm(dim=-1) > 1e-6
    active_frac = float(active_mask.float().mean().item())

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    # Emphasize leading spectral modes in the 32-dim even single-qubit subspace so
    # 1st/2nd-order collocations have high coherence while random pairs remain orthogonal
    mode_decay = torch.exp(-torch.linspace(0.0, 1.35, q_half, device=dev, dtype=torch.float32)).unsqueeze(0)
    c_half = torch.zeros((V, q_half), dtype=torch.float32, device=dev)
    if active_mask.any():
        act_slice = emb_vq[active_mask, :q_half].clone()
        if int(active_mask.sum().item()) > 512:
            act_slice = act_slice - act_slice.mean(dim=0, keepdim=True)
        c_half[active_mask] = l2_normalize(act_slice * mode_decay, dim=-1)

    if token_shards is not None and len(token_shards) > 0:
        from .cooccurrence_init import _slice_to_device_int64

        s0 = token_shards[0]
        sample_len = min(2_000_000, len(s0))
        tok_i64 = _slice_to_device_int64(s0[:sample_len], dev=dev, vocab_size=V)
        if tok_i64.numel() >= 4:
            u_t = tok_i64[:-1]
            v_t = tok_i64[1:]
            neq = u_t != v_t
            u_t = u_t[neq]
            v_t = v_t[neq]
            if u_t.numel() > 0:
                pair_keys = u_t * V + v_t
                uniq_keys, counts = torch.unique(pair_keys, return_counts=True)
                k_top = min(max(64, min(512, V // 4)), int(uniq_keys.numel()))
                top_idx = torch.argsort(-counts, stable=True)[:k_top]
                top_pairs = uniq_keys[top_idx]
                cu = torch.div(top_pairs, V, rounding_mode="floor")
                cv = top_pairs % V
                w_edge = torch.log1p(counts[top_idx].float()).unsqueeze(-1)
                for _ in range(2):
                    agg = torch.zeros_like(c_half)
                    agg.index_add_(0, cu, c_half[cv] * w_edge)
                    agg.index_add_(0, cv, c_half[cu] * w_edge)
                    deg = torch.zeros((V, 1), dtype=torch.float32, device=dev)
                    deg.index_add_(0, cu, w_edge)
                    deg.index_add_(0, cv, w_edge)
                    has_nb = deg.squeeze(-1) > 0
                    if has_nb.any():
                        c_half[has_nb] = l2_normalize(
                            c_half[has_nb] + 0.85 * (agg[has_nb] / deg[has_nb].clamp_min(1e-6)),
                            dim=-1,
                        )

    z_even = l2_normalize(
        torch.randn((V, q_half), generator=gen, dtype=torch.float32).to(dev),
        dim=-1,
    )
    z_odd = l2_normalize(
        torch.randn((V, q_half), generator=gen, dtype=torch.float32).to(dev),
        dim=-1,
    )

    iso_mix = 0.08 * (1.0 - alpha_f)
    u_even = l2_normalize(iso_mix * z_even + max(alpha_f, 0.25) * c_half, dim=-1)
    u_odd = l2_normalize(iso_mix * z_odd + max(alpha_f, 0.25) * c_half, dim=-1)
    if (~active_mask).any():
        u_even[~active_mask] = z_even[~active_mask]
        u_odd[~active_mask] = z_odd[~active_mask]

    if embedder is not None:
        embedder.qubit_wte.weight.zero_()
        embedder.qubit_wte.weight[:, 0 : 2 * q_half : 2].copy_(u_even.to(embedder.qubit_wte.weight.dtype))
        with torch.no_grad():
            z_sparse = torch.randn(
                embedder.unigram_sparse_vals.weight.shape,
                generator=gen,
                dtype=torch.float32,
            ).to(dev) * (1.0 / math.sqrt(float(max(1, embedder.sparse_k))))
            embedder.unigram_sparse_vals.weight.copy_(z_sparse.to(embedder.unigram_sparse_vals.weight.dtype))
            pair_ratio = math.sqrt(max(0.05, 1.0 - alpha_f * alpha_f)) * (0.20 if V > 2048 else 0.55)
            embedder.unigram_pair_scale.fill_(pair_ratio)

    sparse_readout = getattr(model, "sparse_povm_readout", None)
    if isinstance(sparse_readout, Sparse64QubitPOVMReadout) and sparse_readout.povm_odd_delta.shape == (V, q_half):
        sparse_readout.povm_odd_delta.copy_((u_odd - u_even).to(sparse_readout.povm_odd_delta.dtype))

    # Also update legacy wte / dm_seed if present on QuantumBornGPT
    wte_mod = getattr(model, "wte", None)
    if wte_mod is not None and hasattr(wte_mod, "weight"):
        s_dim = int(wte_mod.weight.size(1))
        s_half = max(1, s_dim // 2)
        n_copy = min(q_half, s_half)
        wte_mod.weight.zero_()
        wte_mod.weight[:, 0 : 2 * n_copy : 2].copy_(u_even[:, :n_copy].to(wte_mod.weight.dtype))

    # Resolve POVM detector effects and place them strictly on odd coordinates 1::2
    effects_param: nn.Parameter | None = None
    povm_head_attr = getattr(model, "povm_head", None)
    if isinstance(povm_head_attr, nn.Parameter):
        effects_param = povm_head_attr
    elif povm_head_attr is not None and isinstance(getattr(povm_head_attr, "effects", None), nn.Parameter):
        effects_param = povm_head_attr.effects
    elif getattr(model, "povm_readout", None) is not None:
        pr = getattr(model, "povm_readout")
        if isinstance(getattr(pr, "povm_head", None), nn.Parameter):
            effects_param = pr.povm_head

    u_full_out = torch.zeros((V, fock_dim), dtype=torch.float32, device=dev)
    u_full_out[:, 1 : 2 * q_half : 2] = u_odd
    if fock_dim > num_q:
        pair_odd_len = (fock_dim - num_q) // 2
        if pair_odd_len > 0:
            z_pair_odd = l2_normalize(
                torch.randn((V, pair_odd_len), generator=gen, dtype=torch.float32).to(dev),
                dim=-1,
            )
            u_full_out[:, num_q + 1 : num_q + 1 + 2 * pair_odd_len : 2] = 0.25 * z_pair_out_scale(z_pair_odd)
    u_full_out = l2_normalize(u_full_out, dim=-1)

    if effects_param is not None and effects_param.shape == (V, fock_dim):
        effects_param.copy_(u_full_out.to(effects_param.dtype))

    # Compute Step-0 Fock states psi_0(w) for all w in 0..V-1
    if embedder is not None:
        all_ids = torch.arange(V, device=dev, dtype=torch.int64)
        psi_0_all = embedder.forward_unigram_fock(all_ids).float()
    else:
        psi_0_all = torch.zeros((V, fock_dim), dtype=torch.float32, device=dev)
        psi_0_all[:, 0 : 2 * q_half : 2] = u_even
        psi_0_all = l2_normalize(psi_0_all, dim=-1)

    if isinstance(sparse_readout, Sparse64QubitPOVMReadout) and embedder is not None:
        m_all = sparse_readout.get_povm_matrix(embedder).float()
    elif effects_param is not None and effects_param.shape == (V, fock_dim):
        m_all = l2_normalize(effects_param.float(), dim=-1)
    else:
        m_all = u_full_out

    max_self_overlap = float((psi_0_all * m_all).sum(dim=-1).abs().max().item())
    row_norms = psi_0_all.norm(dim=-1)

    colloc_stats: dict[str, float] = {}
    if token_shards is not None and len(token_shards) > 0:
        colloc_stats = compute_collocation_coherence_metrics(
            tokens_1d=token_shards[0],
            embeddings=psi_0_all,
            top_k=max(16, min(256, V // 8)),
            seed=seed,
        )

    return {
        "max_self_overlap": max_self_overlap,
        "collocation_cosine_ratio": float(colloc_stats.get("cos_ratio", 0.0)),
        "colloc_mean_cos": float(colloc_stats.get("colloc_mean_cos", 0.0)),
        "random_mean_cos": float(colloc_stats.get("random_mean_cos", 0.0)),
        "min_row_norm": float(row_norms.min().item()),
        "max_row_norm": float(row_norms.max().item()),
        "active_vocab_fraction": active_frac,
        "alpha": alpha_f,
    }


def z_pair_out_scale(z: torch.Tensor) -> torch.Tensor:
    return z


__all__ = [
    "build_64qubit_pair_index_tables",
    "Sparse64QubitNgramEmbedder",
    "Sparse64QubitDensityLayer",
    "Sparse64QubitPOVMReadout",
    "initialize_sparse_64qubit_with_ppmi_svd",
]
