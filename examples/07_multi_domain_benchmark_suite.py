#!/usr/bin/env python3
"""Example 07: Multi-Domain Quantum-Geometric Benchmark Suite.

Evaluates QuantumDLBlocks across 6 domains against parameter-matched Classical Softmax/LayerNorm/GELU
baselines:
1. Vision (2D Gabor Orientation/Frequency): QuantumBornViT (Dense & Sparse Born MoE) vs ClassicalViTBaseline
2. Audio (Phase-Modulated Multi-Harmonic Chords): ComplexSuperpositionAudioClassifier vs MagnitudeMelAudioBaseline
3. Speech Recognition (LibriSpeech 16 kHz ASR): ComplexQuantumSpeechCTC (Dense & Sparse Born MoE) vs ClassicalSpeechCTCBaseline
4. Language Modeling (Markov Zipfian Streams): QuantumBornGPT (Dense & QuantumBornMoE) vs ClassicalGPTBaseline
5. Long-Range Arena (Associative Recall / Induction Heads): Real & Complex Phase Flash-Born vs Softmax
6. Calibration & OOD Detection: Projective POVM + von Neumann Entropy S(rho) vs Unbounded Softmax
"""

from __future__ import annotations

import argparse
import torch

from qdlblocks.benchmarks import (
    format_benchmark_table,
    run_all_benchmarks,
    run_audio_benchmark,
    run_calibration_ood_benchmark,
    run_language_benchmark,
    run_librispeech_benchmark,
    run_lra_benchmark,
    run_vision_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Domain Quantum-Geometric Benchmark Suite")
    parser.add_argument(
        "--domain",
        type=str,
        default="all",
        choices=["all", "vision", "audio", "librispeech", "language", "lra", "calibration_ood"],
        help="Benchmark domain to execute.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        choices=["quick", "full"],
        help="Execution mode ('quick' for fast verification, 'full' for deeper convergence).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device ('cuda' or 'cpu').",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    print("=" * 88)
    print(f"QuantumDLBlocks Multi-Domain Benchmark Suite | domain={args.domain} | mode={args.mode} | device={args.device}")
    print("=" * 88)

    if args.domain == "all":
        report = run_all_benchmarks(device=args.device, mode=args.mode, seed=args.seed)  # type: ignore[arg-type]
        print("\n" + report["markdown_table"] + "\n")
    else:
        dispatch = {
            "vision": run_vision_benchmark,
            "audio": run_audio_benchmark,
            "librispeech": run_librispeech_benchmark,
            "language": run_language_benchmark,
            "lra": run_lra_benchmark,
            "calibration_ood": run_calibration_ood_benchmark,
        }
        res = dispatch[args.domain](device=args.device, mode=args.mode, seed=args.seed)  # type: ignore[arg-type]
        print("\n" + format_benchmark_table(res) + "\n")


if __name__ == "__main__":
    main()
