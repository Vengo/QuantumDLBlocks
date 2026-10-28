"""Custom GPU Triton kernels for Quantum-Geometric Deep Learning."""

from .flash_born import (
    HAS_TRITON,
    IS_HIP,
    flash_born_attention,
    flash_purified_born_attention,
    get_flash_born_launch_meta,
    get_rdna4_autotune_configs,
)
from .flash_complex_krylov_triton import (
    DEFAULT_COMPLEX_KERNEL_EPS,
    DEFAULT_KRYLOV_HEAD_CHUNK,
    HAS_COMPLEX_TRITON,
    TRITON_COMPLEX_BLOCK_M,
    TRITON_COMPLEX_BLOCK_N,
    TRITON_COMPLEX_REG_DIM,
    factored_complex_krylov_step,
    flash_complex_krylov_attention,
    fused_causal_pboson_born_triton,
    fused_complex_norm_rope_triton,
)
from .fused_parseval_ce import (
    M4_GEOM_EPS_OFFSET,
    M4_GEOM_EPS_SQ,
    M4_SUBSPACE_SCALE,
    compiled_m4_coincidence_chunk_step,
    compiled_parseval_convex_traceless_chunk_step,
    fused_fp8_or_bf16_chunk_gemm,
    quantize_fp8_vocab_transposed,
)

__all__ = [
    "DEFAULT_COMPLEX_KERNEL_EPS",
    "DEFAULT_KRYLOV_HEAD_CHUNK",
    "HAS_COMPLEX_TRITON",
    "HAS_TRITON",
    "IS_HIP",
    "M4_GEOM_EPS_OFFSET",
    "M4_GEOM_EPS_SQ",
    "M4_SUBSPACE_SCALE",
    "TRITON_COMPLEX_BLOCK_M",
    "TRITON_COMPLEX_BLOCK_N",
    "TRITON_COMPLEX_REG_DIM",
    "compiled_m4_coincidence_chunk_step",
    "compiled_parseval_convex_traceless_chunk_step",
    "factored_complex_krylov_step",
    "flash_born_attention",
    "flash_complex_krylov_attention",
    "flash_purified_born_attention",
    "fused_causal_pboson_born_triton",
    "fused_complex_norm_rope_triton",
    "fused_fp8_or_bf16_chunk_gemm",
    "get_flash_born_launch_meta",
    "get_rdna4_autotune_configs",
    "quantize_fp8_vocab_transposed",
]



