"""Quantum Born Transformer Block with Subspace Projector Born Attention and ReLU^2 Channels."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..functional import l2_normalize, riemannian_center
from .activations import ReLUSquared
from .attention import SubspaceProjectorBornAttention


class QuantumBornBlock(nn.Module):
    """Quantum-Geometric Born Block operating on the unit hypersphere S^{d-1}.

    Combines:
    1. `SubspaceProjectorBornAttention` (positive-cone squared overlaps, zero Softmax).
    2. Bias-free orthogonal Stiefel feed-forward projections (`ff1`, `ff2`) with
       Projective `ReLUSquared` multi-particle activation `f(x) = max(0, x)^2`.
    3. Strict hyperspherical residual blending `x <- l2_normalize(x + sublayer(x))`
       (with optional zero-mean Riemannian centering and coherent input-state highway mixing).
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        block_size: int = 1024,
        hidden_dim: int | None = None,
        mlp_expansion: float = 2.5,
        is_causal: bool = True,
        strict_spherical_residual: bool = True,
        riemannian_centering: bool = False,
        learnable_temperature: bool = False,
        value_residual_mixing: bool = False,
        input_highway_mixing: bool = False,
        complex_phase_attention: bool = False,
        use_flash_born: bool = False,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim if hidden_dim is not None else int(embed_dim * mlp_expansion)
        self.strict_spherical_residual = strict_spherical_residual
        self.riemannian_centering = riemannian_centering
        self.input_highway_mixing = input_highway_mixing
        self.eps = eps

        self.attn = SubspaceProjectorBornAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            block_size=block_size,
            is_causal=is_causal,
            learnable_temperature=learnable_temperature,
            value_residual_mixing=value_residual_mixing,
            complex_phase=complex_phase_attention,
            use_flash_born=use_flash_born,
            eps=eps,
        )

        self.ff1 = nn.Linear(embed_dim, self.hidden_dim, bias=False)
        self.act = ReLUSquared()
        self.ff2 = nn.Linear(self.hidden_dim, embed_dim, bias=False)

        nn.init.orthogonal_(self.ff1.weight)
        nn.init.orthogonal_(self.ff2.weight)

        if input_highway_mixing:
            # Sigmoid gate initialized near ~0.12 reference-beam highway blend
            self.highway_logit = nn.Parameter(torch.tensor(-2.0))
        else:
            self.register_parameter("highway_logit", None)

    def _spherical_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.riemannian_centering:
            return riemannian_center(x, dim=-1, eps=self.eps)
        return l2_normalize(x, dim=-1, eps=self.eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        psi_0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass on S^{embed_dim - 1}.

        Args:
            x: Current layer state tensor `(B, T, embed_dim)`.
            cos: Optional RoPE cosine table.
            sin: Optional RoPE sine table.
            psi_0: Optional initial lifted state `(B, T, embed_dim)` for coherent reference-beam
                interference.
        """
        if self.input_highway_mixing and psi_0 is not None and self.highway_logit is not None:
            gamma = torch.sigmoid(self.highway_logit).to(x.dtype)
            x = self._spherical_norm((1.0 - gamma) * x + gamma * psi_0)

        if self.strict_spherical_residual:
            a = self.attn(x, cos=cos, sin=sin, psi_0=psi_0)
            x = self._spherical_norm(x + a)
            h = self.ff2(self.act(self.ff1(x)))
            x = self._spherical_norm(x + h)
        else:
            x = x + self.attn(self._spherical_norm(x), cos=cos, sin=sin, psi_0=psi_0)
            h = self.ff2(self.act(self.ff1(self._spherical_norm(x))))
            x = x + h

        return x
