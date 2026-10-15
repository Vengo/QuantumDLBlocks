"""Quantum-Geometric Neural Architecture Search (`qdlblocks.qnas`)."""

from .compare_runner import QNASMethodResult, format_qnas_table, load_qnas_dataset, run_qnas_comparison
from .entropy_adaptive import AdaptiveKrausBottleneckNetwork, EntropyAdaptiveNetwork
from .search_space import (
    OP_NAMES,
    ArchConfig,
    DiscreteQuantumBackbone,
    ProjectiveMLPBlock,
    build_candidate_op,
    sample_random_arch,
)
from .superposition_darts import (
    QuantumSearchCell,
    SuperpositionDARTSNetwork,
    anneal_purification_power,
)
from .zero_shot_proxy import (
    QuantumZeroShotNAS,
    compute_qnas_proxy_score,
    kendall_tau_corr,
    spearman_rank_corr,
)

__all__ = [
    "OP_NAMES",
    "ProjectiveMLPBlock",
    "build_candidate_op",
    "ArchConfig",
    "DiscreteQuantumBackbone",
    "sample_random_arch",
    "anneal_purification_power",
    "QuantumSearchCell",
    "SuperpositionDARTSNetwork",
    "EntropyAdaptiveNetwork",
    "AdaptiveKrausBottleneckNetwork",
    "compute_qnas_proxy_score",
    "QuantumZeroShotNAS",
    "spearman_rank_corr",
    "kendall_tau_corr",
    "QNASMethodResult",
    "format_qnas_table",
    "load_qnas_dataset",
    "run_qnas_comparison",
]
