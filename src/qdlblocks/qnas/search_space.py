"""Candidate Operations, Search Space, and Discrete Quantum Backbone for Q-NAS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..functional import l2_normalize
from ..layers.activations import ReLUSquared
from ..layers.attention import SubspaceProjectorBornAttention
from ..layers.measurement import ProjectivePOVMHead
from ..layers.moe import QuantumBornMoE


OP_NAMES: tuple[str, ...] = (
    "spherical_skip",
    "born_attn_real",
    "born_attn_complex",
    "projective_mlp",
    "sparse_born_moe",
)


class ProjectiveMLPBlock(nn.Module):
    """Two-layer Stiefel MLP with ReLUSquared activation and spherical output projection."""

    def __init__(self, embed_dim: int, expansion: int = 2, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps
        self.fc1 = nn.Linear(embed_dim, expansion * embed_dim, bias=False)
        self.act = ReLUSquared()
        self.fc2 = nn.Linear(expansion * embed_dim, embed_dim, bias=False)
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2_normalize(self.fc2(self.act(self.fc1(x))), dim=-1, eps=self.eps)


def build_candidate_op(
    op_name: str,
    embed_dim: int,
    num_heads: int = 4,
    block_size: int = 64,
) -> nn.Module:
    """Instantiate a single candidate Q-NAS operation by name."""
    if op_name == "spherical_skip":
        return nn.Identity()
    if op_name == "born_attn_real":
        return SubspaceProjectorBornAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            block_size=block_size,
            is_causal=False,
            learnable_temperature=True,
            complex_phase=False,
            use_flash_born=False,
        )
    if op_name == "born_attn_complex":
        return SubspaceProjectorBornAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            block_size=block_size,
            is_causal=False,
            learnable_temperature=True,
            complex_phase=True,
            use_flash_born=False,
        )
    if op_name == "projective_mlp":
        return ProjectiveMLPBlock(embed_dim=embed_dim, expansion=2)
    if op_name == "sparse_born_moe":
        return QuantumBornMoE(
            embed_dim=embed_dim,
            expert_hidden_dim=embed_dim,
            num_experts=4,
            top_k=2,
        )
    raise ValueError(f"Unknown Q-NAS candidate op: {op_name!r}")


@dataclass(frozen=True)
class ArchConfig:
    in_features: int = 64
    num_tokens: int = 8
    embed_dim: int = 32
    num_heads: int = 4
    num_classes: int = 10
    cell_ops: tuple[str, ...] = ("born_attn_real", "projective_mlp", "born_attn_complex")


def sample_random_arch(
    rng: np.random.Generator,
    in_features: int = 64,
    num_tokens: int = 8,
    embed_dim: int = 32,
    num_heads: int = 4,
    num_classes: int = 10,
    min_cells: int = 2,
    max_cells: int = 4,
) -> ArchConfig:
    """Sample a random discrete Quantum-Geometric architecture configuration."""
    num_cells = int(rng.integers(min_cells, max_cells + 1))
    cell_ops = tuple(str(rng.choice(OP_NAMES)) for _ in range(num_cells))
    return ArchConfig(
        in_features=in_features,
        num_tokens=num_tokens,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_classes=num_classes,
        cell_ops=cell_ops,
    )


class DiscreteQuantumBackbone(nn.Module):
    """Discrete Quantum-Geometric network instantiated from an ArchConfig."""

    def __init__(self, config: ArchConfig) -> None:
        super().__init__()
        if config.in_features % config.num_tokens != 0:
            raise ValueError(
                f"in_features ({config.in_features}) must be divisible by num_tokens ({config.num_tokens})."
            )
        self.config = config
        token_dim = config.in_features // config.num_tokens
        self.stem = nn.Linear(token_dim, config.embed_dim, bias=False)
        nn.init.orthogonal_(self.stem.weight)

        self.cells = nn.ModuleList(
            [
                build_candidate_op(op, config.embed_dim, config.num_heads, block_size=max(64, config.num_tokens))
                for op in config.cell_ops
            ]
        )
        self.povm_head = ProjectivePOVMHead(
            vocab_size=config.num_classes,
            embed_dim=config.embed_dim,
            povm_temperature=0.05,
            learnable_temperature=True,
        )

    def forward(
        self, x: torch.Tensor, return_layer_states: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if x.ndim == 2:
            x_tok = x.reshape(x.size(0), self.config.num_tokens, -1)
        else:
            x_tok = x

        psi = l2_normalize(self.stem(x_tok), dim=-1, eps=1e-12)
        layer_states: list[torch.Tensor] = []
        for cell in self.cells:
            cell_out = l2_normalize(cell(psi), dim=-1, eps=1e-12)
            psi = l2_normalize(psi + cell_out, dim=-1, eps=1e-12)
            layer_states.append(psi)

        psi_pool = l2_normalize(psi.mean(dim=1), dim=-1, eps=1e-12)
        logits = self.povm_head(psi_pool)
        if return_layer_states:
            return logits, layer_states
        return logits
