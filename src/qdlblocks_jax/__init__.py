"""JAX, Equinox, Haliax, Optax, and Levanter Quantum-Geometric Library (qdlblocks_jax)."""

from __future__ import annotations

import jax
import jax._src.numpy.einsum as _je
import jax._src.numpy.lax_numpy as _lax_numpy
import jax._src.tree_util as _jtu

if not hasattr(_jtu, "BuiltInKeyEntry"):
    _jtu.BuiltInKeyEntry = getattr(_jtu, "KeyEntry", object)  # type: ignore[attr-defined]
if not hasattr(jax.core, "is_constant_dim"):
    setattr(jax.core, "is_constant_dim", lambda d: isinstance(d, int))
for _attr in ("_einsum", "_poly_einsum_handlers", "_default_poly_einsum_handler"):
    if not hasattr(_lax_numpy, _attr) and hasattr(_je, _attr):
        setattr(_lax_numpy, _attr, getattr(_je, _attr))

from .functional import (
    apply_unitary_rope,
    bipartite_entanglement_entropy,
    effective_quantum_rank,
    empirical_state_density,
    ensure_complex,
    kraus_transform,
    l2_normalize,
    normalize_trace,
    partial_trace_qubits,
    povm_measure,
    power_purify,
    precompute_rope_cos_sin,
    purity_participation_ratio,
    riemannian_center,
    state_to_density,
    stiefel_isometry_error,
    von_neumann_entropy,
    zeropower_via_newtonschulz5,
)
from .fused_povm_ce import (
    estimate_povm_head_roofline_jax,
    fused_tiled_povm_cross_entropy_jax,
)
from .kraus_readout import (
    compute_logit_matrix_effective_rank_jax,
    init_multibeam_kraus_params,
    multibeam_kraus_forward_jax,
)
from .overcomplete_frame import (
    frame_coherence_metrics_jax,
    tight_frame_penalty_jax,
    welch_bound_coherence_jax,
)
from .overcomplete_two_beam import (
    overcomplete_two_beam_metrics_jax,
    two_beam_frame_penalty_jax,
)
from .pallas_flash_born import flash_born_attention_jax
from .seed_adapter import overcomplete_seed_adapter_forward_jax
from .seed_scaling_diagnostics import (
    compute_coupled_seed_pullback_rank_jax,
    compute_cross_beam_coherence_jax,
    compute_effective_seed_utilization_ratio_jax,
    compute_interference_feature_rank_jax,
    compute_paired_beam_orthogonality_stats_jax,
    single_beam_signed_born_features_jax,
    two_beam_bilinear_born_features_jax,
)
from .tangent_stiefel import (
    newton_schulz_5_jax,
    project_stiefel_tangent_jax,
    stiefel_quadratic_retract_jax,
    tangent_stiefel_muon_step_jax,
)
from .two_beam_seed_adapter import two_beam_seed_adapter_forward_jax
from .fineweb_stream import (
    compute_memorization_gap_metrics_jax,
    extract_shard_batch_jax,
)
from .parameter_scaling import (
    compute_depth_fidelity_and_geodesic_profile_jax,
    compute_depth_residual_scale_jax,
    default_overcomplete_target_iso_scale_jax,
)
from .dynamic_povm_purity import (
    compute_logit_norm_variation_metrics_jax,
    contextual_bloch_purity_beta_jax,
)
from .early_epoch_quantum import (
    compute_self_repetition_metrics_jax,
    purified_born_attention_weights_jax,
)
from .value_residual_and_norm_control import (
    compute_cross_layer_v_orthogonality_metrics_jax,
    compute_mlp_attn_homogeneity_metrics_jax,
)
from .speedrun_primitives import (
    compute_document_ids_jax,
    half_truncated_rope_cos_sin_jax,
    zeropower_via_newtonschulz5_batched_jax,
)

try:
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
    from .distributed import (
        build_quantum_axis_mapping,
        distributed_stiefel_muon_step,
        make_quantum_device_mesh,
        partition_stiefel_leaves,
        shard_quantum_model,
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
        wsd_lr_schedule_jax,
    )
except ImportError:
    pass

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
    "empirical_state_density",
    "effective_quantum_rank",
    "purity_participation_ratio",
    "flash_born_attention_jax",
    "fused_tiled_povm_cross_entropy_jax",
    "estimate_povm_head_roofline_jax",
    "compute_logit_matrix_effective_rank_jax",
    "init_multibeam_kraus_params",
    "multibeam_kraus_forward_jax",
    "welch_bound_coherence_jax",
    "tight_frame_penalty_jax",
    "frame_coherence_metrics_jax",
    "two_beam_frame_penalty_jax",
    "overcomplete_two_beam_metrics_jax",
    "overcomplete_seed_adapter_forward_jax",
    "two_beam_seed_adapter_forward_jax",
    "compute_memorization_gap_metrics_jax",
    "extract_shard_batch_jax",
    "default_overcomplete_target_iso_scale_jax",
    "compute_depth_residual_scale_jax",
    "compute_depth_fidelity_and_geodesic_profile_jax",
    "compute_logit_norm_variation_metrics_jax",
    "contextual_bloch_purity_beta_jax",
    "compute_self_repetition_metrics_jax",
    "purified_born_attention_weights_jax",
    "compute_cross_layer_v_orthogonality_metrics_jax",
    "compute_mlp_attn_homogeneity_metrics_jax",
    "zeropower_via_newtonschulz5_batched_jax",
    "half_truncated_rope_cos_sin_jax",
    "compute_document_ids_jax",
    "single_beam_signed_born_features_jax",
    "two_beam_bilinear_born_features_jax",
    "compute_interference_feature_rank_jax",
    "compute_cross_beam_coherence_jax",
    "compute_paired_beam_orthogonality_stats_jax",
    "compute_coupled_seed_pullback_rank_jax",
    "compute_effective_seed_utilization_ratio_jax",
    "project_stiefel_tangent_jax",
    "stiefel_quadratic_retract_jax",
    "newton_schulz_5_jax",
    "tangent_stiefel_muon_step_jax",
    "StiefelMuonState",
    "scale_by_stiefel_muon",
    "build_quantum_optax_optimizer",
    "wsd_lr_schedule_jax",
    "build_quantum_axis_mapping",
    "make_quantum_device_mesh",
    "shard_quantum_model",
    "partition_stiefel_leaves",
    "distributed_stiefel_muon_step",
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



