"""Quantum-Geometric Vision Transformer (QuantumBornViT) on S^{d-1}.

Replaces LayerNorm, exponential Softmax, and GELU with:
- 2D Non-Overlapping Patch Projection onto Seed Sphere S^{d_seed - 1} + Orthogonal Stiefel Lift
- Unitary RoPE + SubspaceProjectorBornAttention (with optional Flash-Born & Complex Phase)
- Dense Projective ReLU^2 Stiefel Channels or Sparse QuantumBornMoE Instrument Routing
- Spherical Mean Pooling + Full-Rank ProjectivePOVMHead Readout
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..functional import l2_normalize
from ..layers.born_block import QuantumBornBlock
from ..layers.isometric import UnitaryRoPE
from ..layers.measurement import ProjectivePOVMHead


class QuantumBornViT(nn.Module):
    """Quantum-Geometric Vision Transformer operating strictly on the unit hypersphere S^{d-1}."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        seed_dim: int = 64,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_expansion: float = 2.5,
        num_experts: int = 1,
        top_k_experts: int = 2,
        povm_temperature: float = 0.05,
        learnable_temperature: bool = True,
        complex_phase_attention: bool = False,
        use_flash_born: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size ({image_size}) must be divisible by patch_size ({patch_size}).")
        if embed_dim % num_heads != 0 or (embed_dim // num_heads) % 2 != 0:
            raise ValueError(f"embed_dim ({embed_dim}) / num_heads ({num_heads}) must be an even integer.")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.seed_dim = seed_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_patches = (image_size // patch_size) ** 2
        self.eps = eps

        self.patch_proj = nn.Conv2d(
            in_chans, seed_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        nn.init.orthogonal_(self.patch_proj.weight)

        self.w_lift = nn.Linear(seed_dim, embed_dim, bias=False)
        nn.init.orthogonal_(self.w_lift.weight)

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.rope = UnitaryRoPE(head_dim=embed_dim // num_heads, max_seq_len=max(64, self.num_patches))

        self.blocks = nn.ModuleList(
            [
                QuantumBornBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    block_size=max(64, self.num_patches),
                    mlp_expansion=mlp_expansion,
                    num_experts=num_experts,
                    top_k_experts=top_k_experts,
                    is_causal=False,
                    strict_spherical_residual=True,
                    learnable_temperature=learnable_temperature,
                    value_residual_mixing=True,
                    input_highway_mixing=(idx > 0),
                    complex_phase_attention=complex_phase_attention,
                    use_flash_born=use_flash_born,
                    eps=eps,
                )
                for idx in range(num_layers)
            ]
        )

        self.povm_head = ProjectivePOVMHead(
            vocab_size=num_classes,
            embed_dim=embed_dim,
            povm_temperature=povm_temperature,
            learnable_temperature=learnable_temperature,
            eps=eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_proj(x).flatten(2).transpose(1, 2)  # (B, T, seed_dim)
        psi_seed = l2_normalize(patches, dim=-1, eps=self.eps)
        seq_len = psi_seed.size(1)
        psi_0 = l2_normalize(
            self.w_lift(psi_seed) + self.pos_embed[:, :seq_len, :], dim=-1, eps=self.eps
        )
        cos, sin = self.rope.get_cos_sin(seq_len)
        psi = psi_0
        for block in self.blocks:
            psi = block(psi, cos=cos, sin=sin, psi_0=psi_0)
        psi_pool = l2_normalize(psi.mean(dim=1), dim=-1, eps=self.eps)
        return self.povm_head(psi_pool)
