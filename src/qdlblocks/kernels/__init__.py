"""Custom GPU Triton kernels for Quantum-Geometric Deep Learning."""

from .flash_born import HAS_TRITON, flash_born_attention

__all__ = [
    "HAS_TRITON",
    "flash_born_attention",
]
