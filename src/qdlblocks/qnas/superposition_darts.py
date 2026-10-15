"""Method 1: Coherent Superposition DARTS with PowerPurification Annealing."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import l2_normalize
from ..layers.measurement import ProjectivePOVMHead
from .search_space import OP_NAMES, ArchConfig, DiscreteQuantumBackbone, build_candidate_op


def anneal_purification_power(
    epoch: int,
    max_epochs: int,
    p_start: float = 1.0,
    p_end: float = 4.0,
) -> float:
    """Linearly anneal purification exponent p from p_start to p_end across search epochs."""
    progress = min(max(epoch / max(max_epochs - 1, 1), 0.0), 1.0)
    return float(p_start + progress * (p_end - p_start))


class QuantumSearchCell(nn.Module):
    """Superposition cell placing all candidate quantum primitives in coherent superposition."""

    def __init__(
        self,
        embed_dim: int = 32,
        num_heads: int = 4,
        block_size: int = 64,
        mixing_mode: Literal["quantum_superposition", "classical_softmax"] = "quantum_superposition",
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mixing_mode = mixing_mode
        self.eps = eps

        self.ops = nn.ModuleList(
            [build_candidate_op(op, embed_dim, num_heads, block_size) for op in OP_NAMES]
        )
        # Initialize active quantum channels slightly above identity skip to prevent early DARTS skip-collapse
        init_alpha = torch.tensor([0.16, 0.22, 0.225, 0.22, 0.225], dtype=torch.float32)
        self.alpha_raw = nn.Parameter(init_alpha)

    def get_op_probabilities(self, purification_power: float = 1.0) -> torch.Tensor:
        if self.mixing_mode == "classical_softmax":
            return F.softmax(self.alpha_raw * purification_power, dim=0)
        alpha_unit = l2_normalize(self.alpha_raw, dim=0, eps=self.eps)
        energy = (alpha_unit.abs() + 1e-8).pow(2.0 * purification_power)
        return energy / (energy.sum() + self.eps)

    def forward(self, x: torch.Tensor, purification_power: float = 1.0) -> torch.Tensor:
        was_2d = x.ndim == 2
        if was_2d:
            x = x.unsqueeze(1)

        probs = self.get_op_probabilities(purification_power)
        if self.mixing_mode == "quantum_superposition":
            coeffs = torch.sqrt(probs + self.eps)
        else:
            coeffs = probs

        mixed = torch.zeros_like(x)
        for m, op in enumerate(self.ops):
            op_out = l2_normalize(op(x), dim=-1, eps=self.eps)
            mixed = mixed + coeffs[m] * op_out

        out = l2_normalize(x + mixed, dim=-1, eps=self.eps)
        return out.squeeze(1) if was_2d else out


class SuperpositionDARTSNetwork(nn.Module):
    """Differentiable Quantum Supernet with PowerPurification Annealing & Discrete Export."""

    def __init__(
        self,
        in_features: int = 64,
        num_tokens: int = 8,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_cells: int = 3,
        num_classes: int = 10,
        mixing_mode: Literal["quantum_superposition", "classical_softmax"] = "quantum_superposition",
    ) -> None:
        super().__init__()
        if in_features % num_tokens != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by num_tokens ({num_tokens})."
            )
        self.in_features = in_features
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_cells = num_cells
        self.num_classes = num_classes
        self.mixing_mode = mixing_mode
        self.purification_power: float = 1.0

        token_dim = in_features // num_tokens
        self.stem = nn.Linear(token_dim, embed_dim, bias=False)
        nn.init.orthogonal_(self.stem.weight)

        self.cells = nn.ModuleList(
            [
                QuantumSearchCell(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    block_size=max(64, num_tokens),
                    mixing_mode=mixing_mode,
                )
                for _ in range(num_cells)
            ]
        )
        self.povm_head = ProjectivePOVMHead(
            vocab_size=num_classes,
            embed_dim=embed_dim,
            povm_temperature=0.05,
            learnable_temperature=True,
        )

    def forward(self, x: torch.Tensor, purification_power: float | None = None) -> torch.Tensor:
        p = self.purification_power if purification_power is None else purification_power
        if x.ndim == 2:
            x_tok = x.reshape(x.size(0), self.num_tokens, -1)
        else:
            x_tok = x
        psi = l2_normalize(self.stem(x_tok), dim=-1, eps=1e-12)
        for cell in self.cells:
            psi = cell(psi, purification_power=p)
        psi_pool = l2_normalize(psi.mean(dim=1), dim=-1, eps=1e-12)
        return self.povm_head(psi_pool)

    def selected_ops(self) -> tuple[str, ...]:
        return tuple(
            OP_NAMES[int(cell.get_op_probabilities(self.purification_power).argmax().item())]
            for cell in self.cells
        )

    def export_discrete_model(self) -> DiscreteQuantumBackbone:
        ops = self.selected_ops()
        cfg = ArchConfig(
            in_features=self.in_features,
            num_tokens=self.num_tokens,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_classes=self.num_classes,
            cell_ops=ops,
        )
        disc = DiscreteQuantumBackbone(cfg).to(
            device=self.stem.weight.device, dtype=self.stem.weight.dtype
        )
        disc.stem.load_state_dict(self.stem.state_dict())
        disc.povm_head.load_state_dict(self.povm_head.state_dict())
        for l_idx, cell in enumerate(self.cells):
            m_hat = OP_NAMES.index(ops[l_idx])
            disc.cells[l_idx].load_state_dict(cell.ops[m_hat].state_dict())
        return disc
