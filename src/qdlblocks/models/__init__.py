"""Pre-built pure, hybrid, and Quantum-Geometric Born deep learning architectures."""

from .audio_quantum import ComplexSuperpositionAudioClassifier
from .born_gpt import QuantumBornGPT, ReadoutMode
from .born_vit import QuantumBornViT
from .classical_modded_gpt import (
    ClassicalModdedBlock,
    ClassicalModdedGPT,
    LeaderboardRotary,
    apply_leaderboard_rotary,
)
from .hybrid_classifier import HybridQuantumClassifier
from .level_separated_wavelet_conv_gpt import (
    LevelSeparatedWaveletConvConfig,
    LevelSeparatedWaveletConvGPT,
    PreAttnMultiRateWaveletConvBlock,
    build_level_separated_wavelet_conv_optimizers,
    initialize_level_separated_wavelet_conv_with_ppmi_svd,
    synthesize_level_separated_wavelet_state,
)
from .parseval_mup_dyadic_gpt import (
    ParsevalMuPDyadicConfig,
    ParsevalMuPDyadicGPT,
    build_parseval_mup_optimizers,
    initialize_parseval_mup_with_ppmi_svd,
)
from .parseval_mup_ngram_speedrun_gpt import (
    HashedSurfaceNGramProjector,
    ParsevalMuPNGramSpeedrunConfig,
    ParsevalMuPNGramSpeedrunGPT,
    build_parseval_mup_ngram_optimizers,
    initialize_parseval_mup_ngram_with_ppmi_svd,
)
from .parseval_mup_subspace_coincidence_gpt import (
    ParsevalMuPSubspaceCoincidenceConfig,
    ParsevalMuPSubspaceCoincidenceGPT,
    build_subspace_coincidence_optimizers,
    initialize_subspace_coincidence_with_ppmi_svd,
)
from .pure_quantum_net import DeepPureQuantumNetwork
from .speech_ctc import (
    LIBRISPEECH_CHARS,
    ClassicalSpeechCTCBaseline,
    ComplexPhaseAcousticStem,
    ComplexQuantumSpeechCTC,
    compute_cer,
    compute_wer,
    ctc_greedy_decode,
    extract_librispeech_word_segments,
    indices_to_text,
    levenshtein_distance,
    load_librispeech_dataset,
    text_to_char_indices,
)
from .wavelet_cholesky_2pow16_gpt import (
    ConsistentKroneckerWaveletBlock,
    FusedKroneckerWaveletUNetBlock,
    Radical2Pow16Config,
    Radical2Pow16WaveletCholeskyGPT,
    Tied2Pow16WaveletCholeskyPyramid,
    WaveletCholeskyUNetBlock,
    build_dyadic_2pow16_spectral_tree,
    build_radical_2pow16_optimizers,
    initialize_radical_2pow16_with_ppmi_svd,
)
from .wavelet_cholesky_vision import (
    TiedDyadicWaveletClassPyramid,
    VisionDatasetBundle,
    VisionWaveletCholeskyConfig,
    WaveletCholeskyVisionBlock,
    WaveletCholeskyVisionClassifier,
    build_1000class_finegrained_dataset,
    build_vision_wavelet_optimizers,
    load_or_download_cifar100,
    load_or_download_imagenet1k_32x32,
)
from .gemini_packed_complex_rope_gpt import (
    ComplexUnitaryInterferenceFFN,
    GeminiPackedComplexRoPEBlock,
    GeminiPackedComplexRoPEConfig,
    GeminiPackedComplexRoPEGPT,
    build_gemini_complex_optimizers,
    initialize_gemini_complex_gpt_with_ppmi_svd,
)

__all__ = [
    "DeepPureQuantumNetwork",
    "HybridQuantumClassifier",
    "QuantumBornGPT",
    "QuantumBornViT",
    "ClassicalModdedBlock",
    "ClassicalModdedGPT",
    "LeaderboardRotary",
    "apply_leaderboard_rotary",
    "ComplexSuperpositionAudioClassifier",
    "ComplexPhaseAcousticStem",
    "ComplexQuantumSpeechCTC",
    "ClassicalSpeechCTCBaseline",
    "LIBRISPEECH_CHARS",
    "text_to_char_indices",
    "indices_to_text",
    "ctc_greedy_decode",
    "levenshtein_distance",
    "compute_cer",
    "compute_wer",
    "load_librispeech_dataset",
    "extract_librispeech_word_segments",
    "ReadoutMode",
    "LevelSeparatedWaveletConvConfig",
    "LevelSeparatedWaveletConvGPT",
    "PreAttnMultiRateWaveletConvBlock",
    "synthesize_level_separated_wavelet_state",
    "initialize_level_separated_wavelet_conv_with_ppmi_svd",
    "build_level_separated_wavelet_conv_optimizers",
    "ParsevalMuPDyadicConfig",
    "ParsevalMuPDyadicGPT",
    "initialize_parseval_mup_with_ppmi_svd",
    "build_parseval_mup_optimizers",
    "ParsevalMuPNGramSpeedrunConfig",
    "HashedSurfaceNGramProjector",
    "ParsevalMuPNGramSpeedrunGPT",
    "initialize_parseval_mup_ngram_with_ppmi_svd",
    "build_parseval_mup_ngram_optimizers",
    "ParsevalMuPSubspaceCoincidenceConfig",
    "ParsevalMuPSubspaceCoincidenceGPT",
    "initialize_subspace_coincidence_with_ppmi_svd",
    "build_subspace_coincidence_optimizers",
    "Radical2Pow16Config",
    "Radical2Pow16WaveletCholeskyGPT",
    "Tied2Pow16WaveletCholeskyPyramid",
    "WaveletCholeskyUNetBlock",
    "ConsistentKroneckerWaveletBlock",
    "FusedKroneckerWaveletUNetBlock",
    "build_dyadic_2pow16_spectral_tree",
    "initialize_radical_2pow16_with_ppmi_svd",
    "build_radical_2pow16_optimizers",
    "VisionWaveletCholeskyConfig",
    "VisionDatasetBundle",
    "TiedDyadicWaveletClassPyramid",
    "WaveletCholeskyVisionBlock",
    "WaveletCholeskyVisionClassifier",
    "build_vision_wavelet_optimizers",
    "load_or_download_cifar100",
    "build_1000class_finegrained_dataset",
    "load_or_download_imagenet1k_32x32",
    "ComplexUnitaryInterferenceFFN",
    "GeminiPackedComplexRoPEBlock",
    "GeminiPackedComplexRoPEConfig",
    "GeminiPackedComplexRoPEGPT",
    "build_gemini_complex_optimizers",
    "initialize_gemini_complex_gpt_with_ppmi_svd",
]



