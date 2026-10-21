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
        zero_half_freq: bool = False,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for UnitaryRoPE, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.zero_half_freq = bool(zero_half_freq)

        self._build_tables(max_seq_len, device=None, dtype=torch.float32)

    def _build_tables(
        self,
        seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        half_d = self.head_dim // 2
        if self.zero_half_freq:
            rot_d = max(1, half_d // 2)
            active_freqs = 1.0 / (
                self.base ** (torch.arange(0, rot_d, device=device, dtype=torch.float32) / float(rot_d))
            )
            zero_freqs = torch.zeros(half_d - rot_d, device=device, dtype=torch.float32)
            freqs = torch.cat([active_freqs, zero_freqs], dim=0)
        else:
            freqs = 1.0 / (
                self.base ** (torch.arange(0, half_d, device=device, dtype=torch.float32) / half_d)
            )
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(t, freqs)
        cos_buf = torch.cos(angles).to(dtype).view(1, 1, seq_len, half_d)
        sin_buf = torch.sin(angles).to(dtype).view(1, 1, seq_len, half_d)
        self.max_seq_len = seq_len
        if "rope_cos" in self._buffers:
            self.rope_cos = cos_buf
            self.rope_sin = sin_buf
        else:
            self.register_buffer("rope_cos", cos_buf, persistent=False)
            self.register_buffer("rope_sin", sin_buf, persistent=False)

    def get_cos_sin(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice `(cos, sin)` tables for sequence length `T` (dynamically extending if `T > max_seq_len`)."""
        if seq_len > self.max_seq_len:
            self._build_tables(
                seq_len,
                device=self.rope_cos.device,
                dtype=self.rope_cos.dtype,
            )
        return self.rope_cos[:, :, :seq_len, :], self.rope_sin[:, :, :seq_len, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply unitary rotary position embedding to tensor of shape `(B, H, T, head_dim)`."""
        seq_len = x.size(-2)
        cos, sin = self.get_cos_sin(seq_len)
        return apply_unitary_rope(x, cos, sin)


class YaRNRotarySchedule(nn.Module):
    """Dynamic YaRN rotary frequency interpolation and attention temperature schedule."""

    angular_freq: torch.Tensor
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 65536,
        base: float = 1024.0,
        base_attn_scale: float = 0.12,
        zero_half_freq: bool = True,
        alpha: float = 1.0,
        beta: float = 32.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for YaRNRotarySchedule, got {head_dim}")
        self.head_dim = int(head_dim)
        self.max_seq_len = int(max_seq_len)
        self.base = float(base)
        self.base_attn_scale = float(base_attn_scale)
        self.attn_scale = float(base_attn_scale)
        self.zero_half_freq = bool(zero_half_freq)
        self.alpha = float(alpha)
        self.beta = float(beta)

        half_d = self.head_dim // 2
        if self.zero_half_freq:
            quarter = max(1, self.head_dim // 4)
            active = (1.0 / self.base) ** torch.linspace(0.0, 1.0, steps=quarter, dtype=torch.float32)
            angular_freq = torch.cat([active, active.new_zeros(half_d - quarter)])
        else:
            angular_freq = 1.0 / (
                self.base ** (torch.arange(0, half_d, dtype=torch.float32) / float(half_d))
            )
        self.register_buffer("angular_freq", angular_freq, persistent=False)
        self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        t = torch.arange(self.max_seq_len, device=self.angular_freq.device, dtype=torch.float32)
        theta = torch.outer(t, self.angular_freq)
        cos_buf = theta.cos()
        sin_buf = theta.sin()
        if "cos_cached" in self._buffers:
            self.cos_cached = cos_buf
            self.sin_cached = sin_buf
        else:
            self.register_buffer("cos_cached", cos_buf, persistent=False)
            self.register_buffer("sin_cached", sin_buf, persistent=False)

    @torch.no_grad()
    def apply_yarn(
        self,
        old_window: int,
        new_window: int,
        alpha: float | None = None,
        beta: float | None = None,
        block_size: int = 128,
    ) -> float:
        """Update rotary frequencies when attention window expands from `old_window` to `new_window`."""
        a = self.alpha if alpha is None else float(alpha)
        b = self.beta if beta is None else float(beta)
        if int(new_window) > int(old_window) and int(old_window) > 0:
            rotations = float(block_size) * float(old_window) * self.angular_freq / (2.0 * math.pi)
            weight = torch.clamp((rotations - a) / max(1e-6, b - a), 0.0, 1.0)
            scaling_factor = float(old_window) / float(new_window)
            self.angular_freq = self.angular_freq * (
                scaling_factor + weight * (1.0 - scaling_factor)
            )
            self._rebuild_cache()
            self.attn_scale *= 0.2 * math.log(float(new_window) / float(old_window)) + 1.0
        return self.attn_scale

    @torch.no_grad()
    def step_window(
        self,
        prev_window: int,
        new_window: int,
        seq_len: int = 1024,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Alias returning `(cos, sin, angular_freq, attn_scale)`."""
        self.apply_yarn(prev_window, new_window)
        cos, sin = self.get_cos_sin(seq_len)
        return cos, sin, self.angular_freq, self.attn_scale

    def get_cos_sin(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self.max_seq_len = int(seq_len)
            self._rebuild_cache()
        half_d = self.head_dim // 2
        return (
            self.cos_cached[:seq_len].view(1, 1, seq_len, half_d),
            self.sin_cached[:seq_len].view(1, 1, seq_len, half_d),
        )

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor | int | None = None,
        seq_len: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply current YaRN rotary schedule to tensor `x` (or pair `(q, k)`) of shape `(B, H, T, head_dim)`."""
        if isinstance(k, torch.Tensor):
            t_len = int(seq_len) if seq_len is not None else int(x.size(-2))
            cos, sin = self.get_cos_sin(t_len)
            return apply_unitary_rope(x, cos, sin), apply_unitary_rope(k, cos, sin)
        eff_seq_len = int(k) if isinstance(k, int) else seq_len
        t_len = int(eff_seq_len) if eff_seq_len is not None else int(x.size(-2))
        cos, sin = self.get_cos_sin(t_len)
        return apply_unitary_rope(x, cos, sin)


def apply_partial_key_offset(
    k_bhtd: torch.Tensor,
    rotary_dim: int | None = 64,
    preserve_norm: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Shift stationary upper channels of `k_bhtd` `(B, H, T, D_h)` forward by 1 token position."""
    if k_bhtd.size(-2) <= 1:
        return k_bhtd
    d_h = k_bhtd.size(-1)
    r_dim = (d_h // 2) if rotary_dim is None else int(rotary_dim)
    if r_dim <= 0 or r_dim >= d_h:
        return k_bhtd
    k_out = k_bhtd.clone()
    k_out[:, :, 1:, r_dim:] = k_bhtd[:, :, :-1, r_dim:]
    if preserve_norm:
        in_norm = torch.linalg.vector_norm(k_bhtd.float(), dim=-1, keepdim=True)
        out_norm = torch.linalg.vector_norm(k_out.float(), dim=-1, keepdim=True)
        k_out = (k_out.float() * (in_norm / (out_norm + eps))).to(k_bhtd.dtype)
    return k_out



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

