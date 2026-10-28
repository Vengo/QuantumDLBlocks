"""Manifold-aware optimizers for Quantum-Geometric Deep Learning."""

from .distributed_muon import DistributedStiefelMuon
from .muon import (
    DEFAULT_MUON_REFERENCE_BATCH_TOKENS,
    DecayMode,
    KroneckerParameterPartition,
    MAX_MUON_LR_SCALE,
    MIN_COMPLEX_MUON_MATRIX_DIM,
    MIN_MUON_LR_SCALE,
    PolarMethod,
    StiefelMuon,
    StiefelMuonConfig,
    classify_kronecker_model_parameters,
    collect_named_complex_muon_pairs,
    compute_large_batch_muon_lr_scale,
    polar_snap_weights,
    retract_complex_kraus_generators,
    split_muon_adamw_params,
)
from .schedules import (
    ProgressiveSequenceCurriculum,
    branch_wsd_cooldown_multiplier,
    cosine_lr_multiplier,
    wsd_lr_multiplier,
)
from .tail_ema import TailWeightAverager

__all__ = [
    "DEFAULT_MUON_REFERENCE_BATCH_TOKENS",
    "DecayMode",
    "KroneckerParameterPartition",
    "MAX_MUON_LR_SCALE",
    "MIN_COMPLEX_MUON_MATRIX_DIM",
    "MIN_MUON_LR_SCALE",
    "PolarMethod",
    "StiefelMuon",
    "StiefelMuonConfig",
    "DistributedStiefelMuon",
    "classify_kronecker_model_parameters",
    "collect_named_complex_muon_pairs",
    "compute_large_batch_muon_lr_scale",
    "polar_snap_weights",
    "retract_complex_kraus_generators",
    "split_muon_adamw_params",
    "cosine_lr_multiplier",
    "wsd_lr_multiplier",
    "branch_wsd_cooldown_multiplier",
    "ProgressiveSequenceCurriculum",
    "TailWeightAverager",
]



