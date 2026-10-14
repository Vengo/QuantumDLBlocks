"""Hybrid Classical-Quantum Classifier combining spatial tokens with Complex Superposition Attention."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..layers.activations import PowerPurification
from ..layers.attention import ComplexSuperpositionAttention, PoolingMode, PurificationPlacement
from ..layers.channels import KrausChannel
from ..layers.measurement import POVMMeasurement


class HybridQuantumClassifier(nn.Module):
    """Hybrid Spatial Feature + Complex Superposition Quantum Network.

    Implements and generalizes `PureQuantumClassifier` from Script 2:
    1. Extracts spatial patch tokens `(B, L, D)` using a lightweight 2D CNN stem
       (or accepts pre-extracted sequences directly if `use_conv_stem=False`).
    2. Projects tokens into Hilbert space via `ComplexSuperpositionAttention`,
       computing phase-preserving complex inner product amplitudes without Softmax.
    3. Optionally evolves the resulting density matrix through a `KrausChannel`
       and `PowerPurification`.
    4. Performs `POVMMeasurement` readout `Tr(rho @ M_c)` for class logits.
    """

    def __init__(
        self,
        in_channels: int = 1,
        token_dim: int = 32,
        hilbert_dim: int = 16,
        num_heads: int = 4,
        num_classes: int = 10,
        grid_size: tuple[int, int] = (4, 4),
        pooling_mode: PoolingMode = "coherent",
        purification_placement: PurificationPlacement = "per_head",
        purification_power: int = 2,
        num_kraus: int = 0,
        use_conv_stem: bool = True,
    ) -> None:
        super().__init__()
        self.use_conv_stem = use_conv_stem
        self.token_dim = token_dim
        self.hilbert_dim = hilbert_dim
        self.num_classes = num_classes

        if use_conv_stem:
            self.feature_proj: nn.Module | None = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, token_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(token_dim),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(grid_size),
            )
        else:
            self.feature_proj = None

        self.quantum_head = ComplexSuperpositionAttention(
            in_features=token_dim,
            hilbert_dim=hilbert_dim,
            num_heads=num_heads,
            pooling_mode=pooling_mode,
            purification_power=purification_power,
            purification_placement=purification_placement,
        )

        if num_kraus > 0:
            self.kraus_block: nn.Module | None = nn.Sequential(
                KrausChannel(
                    in_hilbert_dim=hilbert_dim,
                    out_hilbert_dim=hilbert_dim,
                    num_kraus=num_kraus,
                ),
                PowerPurification(power=purification_power),
            )
        else:
            self.kraus_block = None

        self.povm = POVMMeasurement(hilbert_dim=hilbert_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass from image `(B, C, H, W)` or sequence `(B, L, D)` to `(logits, rho)`."""
        if self.feature_proj is not None:
            feat = self.feature_proj(x)  # (B, token_dim, H_g, W_g)
            b, c, h_f, w_f = feat.shape
            tokens = feat.permute(0, 2, 3, 1).reshape(b, h_f * w_f, c)  # (B, L, token_dim)
        else:
            tokens = x

        rho = self.quantum_head(tokens)  # (B, N, N)
        if self.kraus_block is not None:
            rho = self.kraus_block(rho)

        logits = self.povm(rho)          # (B, num_classes)
        return logits, rho
