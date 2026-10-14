"""Multi-Layer Pure Quantum Attention Network operating entirely in Hilbert space."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..layers.activations import PowerPurification
from ..layers.attention import PureQuantumAttention, PurificationPlacement
from ..layers.channels import KrausChannel
from ..layers.measurement import POVMMeasurement


class DeepPureQuantumNetwork(nn.Module):
    """Deep Quantum Network operating purely in complex Hilbert space.

    Architecture (Zero Conv, Zero ReLU, Zero MaxPool, Zero Softmax on attention):
    - Layer 1: `PureQuantumAttention` mapping inputs `R^D -> C^{N x N}` density matrix `rho_1`.
    - Layer 2..K: Stack of `KrausChannel` transformations `rho_l -> sum_k A_k rho_{l-1} A_k^dagger`,
      optionally interleaved with `PowerPurification` non-linearities.
    - Final Layer: `POVMMeasurement` readout `logits_c = Re(Tr(rho_final @ M_c))`.
    """

    def __init__(
        self,
        in_features: int = 64,
        hilbert_dim: int = 16,
        num_classes: int = 10,
        num_heads: int = 4,
        num_kraus: int = 4,
        num_kraus_layers: int = 1,
        purification_placement: PurificationPlacement = "per_head",
        purify_after_kraus: bool = False,
        purification_power: int = 2,
        kraus_residual: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hilbert_dim = hilbert_dim
        self.num_classes = num_classes

        self.attention = PureQuantumAttention(
            in_features=in_features,
            hilbert_dim=hilbert_dim,
            num_heads=num_heads,
            purification_power=purification_power,
            purification_placement=purification_placement,
        )

        kraus_blocks: list[nn.Module] = []
        for _ in range(num_kraus_layers):
            kraus_blocks.append(
                KrausChannel(
                    in_hilbert_dim=hilbert_dim,
                    out_hilbert_dim=hilbert_dim,
                    num_kraus=num_kraus,
                    residual=kraus_residual,
                )
            )
            if purify_after_kraus:
                kraus_blocks.append(PowerPurification(power=purification_power))

        self.quantum_channels = nn.ModuleList(kraus_blocks)
        self.povm = POVMMeasurement(hilbert_dim=hilbert_dim, num_classes=num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_trajectory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Forward pass through pure Hilbert-space network.

        Args:
            x: Input feature tensor of shape `(B, in_features)`.
            return_trajectory: If True, also returns the list of intermediate density matrices
                `[rho_1, rho_2, ...]` across the network depth.

        Returns:
            `(logits, rho_final)` or `(logits, rho_final, trajectory)`.
        """
        rho = self.attention(x)
        trajectory = [rho]

        for block in self.quantum_channels:
            rho = block(rho)
            trajectory.append(rho)

        logits = self.povm(rho)

        if return_trajectory:
            return logits, rho, trajectory
        return logits, rho
