"""Modular complex Hilbert-space, density-matrix, and Stiefel/Born quantum-geometric layers."""

from .activations import PowerPurification, ReLUSquared, TwoBeamQuantumInterference
from .attention import (
    ComplexSuperpositionAttention,
    PoolingMode,
    PureQuantumAttention,
    PurificationPlacement,
    SubspaceProjectorBornAttention,
    compute_document_ids,
)
from .born_block import QuantumBornBlock
from .channels import ConvexHeadMixture, KrausChannel
from .complex_linear import ComplexLinear, MultiHeadComplexProjection
from .direct_chunked_ce import (
    _DirectChunkedPOVMCEFn,
    _DirectChunkedSoftcapLinearCEFn,
    _DirectChunkedTwoBeamKBosonCEFn,
    build_sampled_candidate_set,
    compute_two_beam_kboson_logits,
    direct_chunked_povm_ce,
    direct_chunked_softcap_linear_ce,
    direct_chunked_two_beam_kboson_ce,
    sampled_povm_ce,
    sampled_softcap_linear_ce,
)
from .cooccurrence_init import (
    compute_collocation_coherence_metrics,
    compute_ppmi_svd_embeddings,
    initialize_classical_with_ppmi_svd,
    initialize_quantum_born_with_ppmi_svd,
)
from .fp8_linear import (
    FP8LinearWrapper,
    convert_linear_layers_to_fp8,
    is_fp8_scaled_mm_supported,
)
from .isometric import (
    IsometricHilbertLift,
    SphericalResidual,
    UnitSphereEmbedding,
    UnitaryRoPE,
    YaRNRotarySchedule,
    apply_partial_key_offset,
)
from .measurement import POVMMeasurement, ProjectivePOVMHead
from .moe import QuantumBornMoE, SparseKrausMoE
from .mudd_and_gates import (
    ForwardSmearGate,
    InputChannelGate,
    MUDDDynamicSkip,
    apply_gated_xsa,
)
from .ngram_hash import (
    BIGRAM_ROW_MULS,
    BIGRAM_SIGN_MULS,
    TRIGRAM_ROW_MULS,
    TRIGRAM_SIGN_MULS,
    HashedNgramEmbedding,
    compute_ngram_hashes_and_signs,
)
from .noise import DepolarizingNoise, SPAMNoise
from .overcomplete_frame import (
    OvercompleteFrameInterference,
    frame_coherence_metrics,
    optimize_overcomplete_tight_frame,
    tight_frame_penalty,
    welch_bound_coherence,
)
from .seed_adapter import (
    OvercompleteSeedReadoutAdapter,
    compute_input_sensitivity_effective_rank,
)
from .overcomplete_two_beam import (
    OvercompleteTwoBeamInterference,
    optimize_orthogonal_pair_tight_frames,
    overcomplete_two_beam_metrics,
    two_beam_frame_penalty,
)
from .seed_scaling_diagnostics import (
    compute_coupled_seed_pullback_rank,
    compute_cross_beam_coherence,
    compute_cross_subspace_regression_mse,
    compute_effective_seed_utilization_ratio,
    compute_interference_feature_rank,
    compute_paired_beam_orthogonality_stats,
    single_beam_signed_born_features,
    two_beam_bilinear_born_features,
)
from .two_beam_seed_adapter import (
    TwoBeamSeedReadoutAdapter,
    compute_nullspace_antipodal_separation,
)
from .parameter_scaling import (
    compute_depth_fidelity_and_geodesic_profile,
    compute_depth_residual_scale,
    compute_parameter_scaling_breakdown,
    default_overcomplete_target_iso_scale,
)
from .dynamic_povm_purity import (
    ContextualBlochPurityGate,
    compute_logit_norm_variation_metrics,
)
from .early_epoch_quantum import (
    apply_orthogonal_highway_decoupling,
    compute_born_attention_snr_metrics,
    compute_self_repetition_metrics,
    purified_born_attention_weights,
)
from .value_residual_and_norm_control import (
    compute_cross_layer_v_orthogonality_metrics,
    compute_mlp_attn_homogeneity_metrics,
)
from .sparse_qubit_fock import (
    Sparse64QubitDensityLayer,
    Sparse64QubitNgramEmbedder,
    Sparse64QubitPOVMReadout,
    build_64qubit_pair_index_tables,
    initialize_sparse_64qubit_with_ppmi_svd,
)
from .hierarchical_qubit import (
    Hierarchical16QubitEmbedder,
    Hierarchical16QubitPOVMReadout,
    initialize_hierarchical_16qubit_with_ppmi_svd,
)
from .stereographic_lift import (
    ClassicalSoftmaxMLP,
    NaiveBornClassifier,
    StereographicHilbertLift,
    StereographicSingleBeamKBoson,
    StereographicTwoBeamKBoson,
    apply_dimension_preserving_stereographic_rotation,
    build_concentric_shells_dataset,
    load_fisher_iris_dataset,
    run_iris_and_shells_benchmark,
)

__all__ = [
    "ComplexLinear",
    "MultiHeadComplexProjection",
    "PureQuantumAttention",
    "ComplexSuperpositionAttention",
    "SubspaceProjectorBornAttention",
    "compute_document_ids",
    "_DirectChunkedSoftcapLinearCEFn",
    "direct_chunked_softcap_linear_ce",
    "_DirectChunkedPOVMCEFn",
    "direct_chunked_povm_ce",
    "_DirectChunkedTwoBeamKBosonCEFn",
    "direct_chunked_two_beam_kboson_ce",
    "compute_two_beam_kboson_logits",
    "build_sampled_candidate_set",
    "sampled_softcap_linear_ce",
    "sampled_povm_ce",
    "compute_ppmi_svd_embeddings",
    "initialize_quantum_born_with_ppmi_svd",
    "initialize_classical_with_ppmi_svd",
    "compute_collocation_coherence_metrics",
    "BIGRAM_ROW_MULS",
    "TRIGRAM_ROW_MULS",
    "BIGRAM_SIGN_MULS",
    "TRIGRAM_SIGN_MULS",
    "compute_ngram_hashes_and_signs",
    "HashedNgramEmbedding",
    "build_64qubit_pair_index_tables",
    "Sparse64QubitNgramEmbedder",
    "Sparse64QubitDensityLayer",
    "Sparse64QubitPOVMReadout",
    "initialize_sparse_64qubit_with_ppmi_svd",
    "Hierarchical16QubitEmbedder",
    "Hierarchical16QubitPOVMReadout",
    "initialize_hierarchical_16qubit_with_ppmi_svd",
    "StereographicHilbertLift",
    "apply_dimension_preserving_stereographic_rotation",
    "load_fisher_iris_dataset",
    "build_concentric_shells_dataset",
    "NaiveBornClassifier",
    "StereographicSingleBeamKBoson",
    "StereographicTwoBeamKBoson",
    "ClassicalSoftmaxMLP",
    "run_iris_and_shells_benchmark",
    "InputChannelGate",
    "apply_gated_xsa",
    "MUDDDynamicSkip",
    "ForwardSmearGate",
    "is_fp8_scaled_mm_supported",
    "FP8LinearWrapper",
    "convert_linear_layers_to_fp8",
    "PoolingMode",
    "PurificationPlacement",
    "PowerPurification",
    "ReLUSquared",
    "TwoBeamQuantumInterference",
    "OvercompleteFrameInterference",
    "welch_bound_coherence",
    "tight_frame_penalty",
    "frame_coherence_metrics",
    "optimize_overcomplete_tight_frame",
    "OvercompleteSeedReadoutAdapter",
    "compute_input_sensitivity_effective_rank",
    "OvercompleteTwoBeamInterference",
    "two_beam_frame_penalty",
    "overcomplete_two_beam_metrics",
    "optimize_orthogonal_pair_tight_frames",
    "TwoBeamSeedReadoutAdapter",
    "compute_nullspace_antipodal_separation",
    "default_overcomplete_target_iso_scale",
    "compute_depth_residual_scale",
    "compute_parameter_scaling_breakdown",
    "compute_depth_fidelity_and_geodesic_profile",
    "ContextualBlochPurityGate",
    "compute_logit_norm_variation_metrics",
    "compute_self_repetition_metrics",
    "purified_born_attention_weights",
    "compute_born_attention_snr_metrics",
    "apply_orthogonal_highway_decoupling",
    "compute_cross_layer_v_orthogonality_metrics",
    "compute_mlp_attn_homogeneity_metrics",
    "single_beam_signed_born_features",
    "two_beam_bilinear_born_features",
    "compute_interference_feature_rank",
    "compute_cross_beam_coherence",
    "compute_paired_beam_orthogonality_stats",
    "compute_cross_subspace_regression_mse",
    "compute_coupled_seed_pullback_rank",
    "compute_effective_seed_utilization_ratio",
    "ConvexHeadMixture",
    "KrausChannel",
    "POVMMeasurement",
    "ProjectivePOVMHead",
    "UnitSphereEmbedding",
    "IsometricHilbertLift",
    "UnitaryRoPE",
    "YaRNRotarySchedule",
    "apply_partial_key_offset",
    "SphericalResidual",
    "QuantumBornBlock",
    "QuantumBornMoE",
    "SparseKrausMoE",
    "SPAMNoise",
    "DepolarizingNoise",
]



