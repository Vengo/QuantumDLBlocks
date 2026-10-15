"""Modular complex Hilbert-space, density-matrix, and Stiefel/Born quantum-geometric layers."""

from .activations import PowerPurification, ReLUSquared
from .attention import (
    ComplexSuperpositionAttention,
    PoolingMode,
    PureQuantumAttention,
    PurificationPlacement,
    SubspaceProjectorBornAttention,
)
from .born_block import QuantumBornBlock
from .channels import ConvexHeadMixture, KrausChannel
from .complex_linear import ComplexLinear, MultiHeadComplexProjection
from .isometric import IsometricHilbertLift, SphericalResidual, UnitSphereEmbedding, UnitaryRoPE
from .measurement import POVMMeasurement, ProjectivePOVMHead
from .moe import QuantumBornMoE, SparseKrausMoE
from .noise import DepolarizingNoise, SPAMNoise

__all__ = [
    "ComplexLinear",
    "MultiHeadComplexProjection",
    "PureQuantumAttention",
    "ComplexSuperpositionAttention",
    "SubspaceProjectorBornAttention",
    "PoolingMode",
    "PurificationPlacement",
    "PowerPurification",
    "ReLUSquared",
    "ConvexHeadMixture",
    "KrausChannel",
    "POVMMeasurement",
    "ProjectivePOVMHead",
    "UnitSphereEmbedding",
    "IsometricHilbertLift",
    "UnitaryRoPE",
    "SphericalResidual",
    "QuantumBornBlock",
    "QuantumBornMoE",
    "SparseKrausMoE",
    "SPAMNoise",
    "DepolarizingNoise",
]
