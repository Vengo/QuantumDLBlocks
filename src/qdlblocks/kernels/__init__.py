"""Custom GPU Triton kernels for Quantum-Geometric Deep Learning."""

from .flash_born import (
    HAS_TRITON,
    IS_HIP,
    flash_born_attention,
    flash_purified_born_attention,
    get_flash_born_launch_meta,
    get_rdna4_autotune_configs,
)
from .fused_parseval_ce import (
    compiled_parseval_convex_traceless_chunk_step,
    fused_fp8_or_bf16_chunk_gemm,
    quantize_fp8_vocab_transposed,
)

__all__ = [
    "HAS_TRITON",
    "IS_HIP",
    "compiled_parseval_convex_traceless_chunk_step",
    "flash_born_attention",
    "flash_purified_born_attention",
    "fused_fp8_or_bf16_chunk_gemm",
    "get_flash_born_launch_meta",
    "get_rdna4_autotune_configs",
    "quantize_fp8_vocab_transposed",
]

