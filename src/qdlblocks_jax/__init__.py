"""JAX, Equinox, Haliax, Optax, and Levanter Quantum-Geometric Library (qdlblocks_jax)."""

from __future__ import annotations

from .functional import (
    apply_unitary_rope,
    bipartite_entanglement_entropy,
    ensure_complex,
    kraus_transform,
    l2_normalize,
    normalize_trace,
    partial_trace_qubits,
    povm_measure,
    power_purify,
    precompute_rope_cos_sin,
    riemannian_center,
    state_to_density,
    stiefel_isometry_error,
    von_neumann_entropy,
    zeropower_via_newtonschulz5,
)
from .haliax_modules import (
    HaxIsometricHilbertLift,
    HaxProjectivePOVMHead,
    HaxQuantumBornBlock,
    HaxQuantumBornGPT,
    HaxQuantumBornMoE,
    HaxSubspaceBornAttention,
    hax_l2_normalize,
    orthogonal_init,
)
from .levanter_model import (
    HAS_LEVANTER,
    HaxQuantumBornLM,
    HaxQuantumBornLMConfig,
)
from .optim import (
    StiefelMuonState,
    build_quantum_optax_optimizer,
    scale_by_stiefel_muon,
)
from .pallas_flash_born import flash_born_attention_jax

__all__ = [
    "l2_normalize",
    "riemannian_center",
    "precompute_rope_cos_sin",
    "apply_unitary_rope",
    "zeropower_via_newtonschulz5",
    "stiefel_isometry_error",
    "ensure_complex",
    "normalize_trace",
    "state_to_density",
    "power_purify",
    "kraus_transform",
    "povm_measure",
    "von_neumann_entropy",
    "partial_trace_qubits",
    "bipartite_entanglement_entropy",
    "flash_born_attention_jax",
    "StiefelMuonState",
    "scale_by_stiefel_muon",
    "build_quantum_optax_optimizer",
    "hax_l2_normalize",
    "orthogonal_init",
    "HaxIsometricHilbertLift",
    "HaxSubspaceBornAttention",
    "HaxQuantumBornMoE",
    "HaxQuantumBornBlock",
    "HaxProjectivePOVMHead",
    "HaxQuantumBornGPT",
    "HAS_LEVANTER",
    "HaxQuantumBornLMConfig",
    "HaxQuantumBornLM",
]
