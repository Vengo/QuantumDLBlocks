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
]
