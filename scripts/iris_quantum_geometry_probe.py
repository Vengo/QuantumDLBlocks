#!/usr/bin/env python3
"""Fisher's Iris & 4D Concentric Radial Shells Quantum Geometry Probe (`scripts/iris_quantum_geometry_probe.py`).

Implements Section 5.2 & Hypothesis H1 of `EXPERIMENT_PLAN_028.md`:
- Enforces `torch.cuda.set_per_process_memory_fraction(0.5, device=0)` (`<= 8.0 GB` VRAM cap on RTX 5060 Ti).
- Runs the 4-model head-to-head geometric trap benchmark:
  1. `NaiveBornClassifier` (`l2_normalize` radial-erasing projection + pure Born rule)
  2. `StereographicSingleBeamKBoson` (`StereographicHilbertLift` + single-beam k-Boson Bloch readout)
  3. `StereographicTwoBeamKBoson` (`StereographicHilbertLift` + Cosine/Sine Quadrature Two-Beam
     Interference + Rank-r Kraus Bath with Positive-Part Quadratic Fidelity)
  4. `ClassicalSoftmaxMLP` (Parameter-matched 2-layer `Linear -> GELU -> Linear` Softmax MLP)
- Evaluates both:
  * **4D Concentric Radial Shells (`N=300, d=4, C=3`, radii `{1.0, 2.2, 3.6}`)**
  * **Fisher's Iris (`N=150, d=4, C=3`, *Setosa*, *Versicolor*, *Virginica*)**
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.layers.stereographic_lift import run_iris_and_shells_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Fisher's Iris & 4D Concentric Radial Shells quantum geometry benchmark (Plan 028)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Training epochs per tabular model (default: 250).",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        default=False,
        help="Run a fast 150-epoch smoke test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic PRNG seed (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--max-vram-gb",
        type=float,
        default=8.0,
        help="Maximum CUDA VRAM cap in GB on local GPU (default: 8.0).",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Optional output path to save JSON benchmark telemetry.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        frac = min(0.5, max(0.05, float(args.max_vram_gb) / max(total_gb, 1e-6)))
        torch.cuda.set_per_process_memory_fraction(frac, device=0)
        torch.cuda.reset_peak_memory_stats(0)

    eff_epochs = 150 if args.quick_test else int(args.epochs)
    res = run_iris_and_shells_benchmark(
        epochs=eff_epochs,
        seed=int(args.seed),
        device=args.device,
    )
    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated(0) / (1024**2), 2)
        if (torch.cuda.is_available() and str(args.device).startswith("cuda"))
        else 0.0
    )
    res["peak_vram_mb"] = peak_vram_mb

    shells = res["concentric_shells_4d"]
    iris = res["fisher_iris_4d"]

    print("=" * 98)
    print(
        f"  PLAN 028 QUANTUM GEOMETRY PROBE — 4D CONCENTRIC RADIAL SHELLS & FISHER'S IRIS "
        f"(epochs={eff_epochs}, peak_vram={peak_vram_mb:.1f} MB)"
    )
    print("=" * 98)
    print(f"\n[1] 4D Concentric Radial Shells (N=300, d=4, C=3, radii={shells['radii']})")
    print(
        f"  {'Model':<32} | {'Params':>7} | {'Train Acc':>9} | {'Test Acc':>9} | "
        f"{'Test CE':>8} | {'Radial Sens':>11} | {'Beam Ortho':>10}"
    )
    print("  " + "-" * 94)
    for m_name, m_stats in shells["models"].items():
        print(
            f"  {m_name:<32} | {m_stats['params']:>7,d} | {m_stats['train_acc_pct']:>8.2f}% | "
            f"{m_stats['test_acc_pct']:>8.2f}% | {m_stats['test_ce']:>8.4f} | "
            f"{m_stats['radial_state_sensitivity_l2']:>11.4f} | {m_stats['max_beam_orthogonality_abs']:>10.2e}"
        )

    print(
        f"\n[2] Fisher's Iris (N=150, d=4, C=3; Versicolor-Virginica ray cos={iris['versicolor_virginica_centroid_cos']:.4f}, "
        f"norm ratio={iris['virginica_to_versicolor_norm_ratio']:.3f}x)"
    )
    print(
        f"  {'Model':<32} | {'Params':>7} | {'Full Acc':>9} | {'Test Acc':>9} | "
        f"{'Test CE':>8} | {'Radial Sens':>11} | {'Beam Ortho':>10}"
    )
    print("  " + "-" * 94)
    for m_name, m_stats in iris["models"].items():
        print(
            f"  {m_name:<32} | {m_stats['params']:>7,d} | {m_stats['full_acc_pct']:>8.2f}% | "
            f"{m_stats['test_acc_pct']:>8.2f}% | {m_stats['test_ce']:>8.4f} | "
            f"{m_stats['radial_state_sensitivity_l2']:>11.4f} | {m_stats['max_beam_orthogonality_abs']:>10.2e}"
        )

    print("\n" + "=" * 98)
    print(
        f"  H1 Verdict: {'PASS' if res['passed'] else 'FAIL'} | "
        f"Shells: NaiveBorn={shells['naive_born_test_acc_pct']:.2f}% vs StereoTwoBeam={shells['stereographic_two_beam_test_acc_pct']:.2f}% "
        f"(+{shells['gain_percentage_points']:.2f} pp) | "
        f"Iris: NaiveBorn={iris['naive_born_acc_pct']:.2f}% vs StereoTwoBeam={iris['stereographic_two_beam_acc_pct']:.2f}% "
        f"(+{iris['gain_percentage_points']:.2f} pp)"
    )
    print("=" * 98)

    if args.out_json:
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        print(f"Saved JSON telemetry -> {out_p}")

    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
