"""Multi-Domain Quantum-Geometric Benchmark Suite (`qdlblocks.benchmarks`)."""

from .baselines import ClassicalGPTBaseline, ClassicalViTBaseline, MagnitudeMelAudioBaseline
from .metrics_calibration import (
    brier_score,
    expected_calibration_error,
    ood_auroc,
    ood_auroc_from_scores,
    probs_from_logits_or_probs,
)
from .runner import (
    BenchmarkResult,
    format_benchmark_table,
    generate_associative_recall_batch,
    generate_markov_zipf_tokens,
    generate_phase_modulated_audio_dataset,
    generate_synthetic_gabor_dataset,
    run_all_benchmarks,
    run_audio_benchmark,
    run_calibration_ood_benchmark,
    run_language_benchmark,
    run_librispeech_benchmark,
    run_lra_benchmark,
    run_vision_benchmark,
)

__all__ = [
    "BenchmarkResult",
    "format_benchmark_table",
    "probs_from_logits_or_probs",
    "expected_calibration_error",
    "brier_score",
    "ood_auroc_from_scores",
    "ood_auroc",
    "ClassicalViTBaseline",
    "MagnitudeMelAudioBaseline",
    "ClassicalGPTBaseline",
    "generate_synthetic_gabor_dataset",
    "generate_phase_modulated_audio_dataset",
    "generate_markov_zipf_tokens",
    "generate_associative_recall_batch",
    "run_vision_benchmark",
    "run_audio_benchmark",
    "run_librispeech_benchmark",
    "run_language_benchmark",
    "run_lra_benchmark",
    "run_calibration_ood_benchmark",
    "run_all_benchmarks",
]
