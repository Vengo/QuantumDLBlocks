#!/usr/bin/env python3
"""Example 08: Quantum-Geometric Neural Architecture Search (Q-NAS) Head-to-Head Study.

Compares all three Quantum-Geometric NAS paradigms plus Classical Softmax DARTS:
1. Classical Softmax DARTS (Baseline)
2. Coherent Superposition DARTS with PowerPurification Annealing (rho^p / Tr(rho^p))
3. Thermodynamic / von Neumann Entropy-Adaptive Kraus Network (dynamic Hilbert dimension adaptation)
4. Zero-Shot Quantum-Geometric Proxy Search (training-free expressibility, gradient isotropy & Stiefel retention)
"""

from __future__ import annotations

import argparse
import torch

from qdlblocks.qnas import run_qnas_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Q-NAS Head-to-Head Comparison Study")
    parser.add_argument("--epochs", type=int, default=16, help="Search / training epochs.")
    parser.add_argument(
        "--candidates",
        type=int,
        default=10,
        help="Number of random architectures evaluated by Zero-Shot Quantum Proxy NAS.",
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
    print(
        f"QuantumDLBlocks Q-NAS Comparison Study | epochs={args.epochs} | "
        f"candidates={args.candidates} | device={args.device}"
    )
    print("=" * 88)

    report = run_qnas_comparison(
        device=args.device,
        epochs=args.epochs,
        num_zero_shot_candidates=args.candidates,
        seed=args.seed,
    )

    print("\n### Head-to-Head Q-NAS Results\n")
    print(report["markdown_table"])

    corr = report["proxy_rank_correlations"]
    print("\n### Zero-Shot Quantum Proxy Rank Correlation vs. Trained Accuracy")
    print(f"- Spearman Rank Correlation (rho): {corr['spearman_rho']:+.4f}")
    print(f"- Kendall Tau Rank Correlation   : {corr['kendall_tau']:+.4f}\n")


if __name__ == "__main__":
    main()
