#!/usr/bin/env python3
r"""CLI Benchmark for Small-Scale FineWeb Span Reversal: Initialization & Probability Readout Suite.

Executes on the local GPU:
1. **GPU Initialization Wall-Clock Latency Benchmark (`benchmark_gpu_init_timings_ms`)**:
   Measures exact GPU `torch.cuda.Event` latency (`ms`) for Unigram `bincount`, SVD-free GPU CountSketch
   Bigram Sphere projection, and GPU `torch.svd_lowrank` Anchor Bigram PPMI projection.
2. **Part 1 — 8-Arm Readout Tournament (`Classical` vs. `Quantum Wavepacket` $\times$ `R1..R4`)**:
   Compares `R1_LINEAR_SOFTCAP`, `R2_LEVEL_A_RADIAL_SPHERE`, `R3_TIED_WAVEPACKET_EXIT_BRIDGE`, and
   `R4_HERMITIAN_QUADRATIC_BORN` on FineWeb Span Reversal (`[t_1..t_31, <REV>, t_31..t_1]`).
3. **Part 2 — 4-Arm GPU Initialization Ablation (`RANDOM_SPHERE` vs. `GPU_UNIGRAM_PRIOR` vs.
   `GPU_COUNTSKETCH_BIGRAM_SPHERE` vs. `GPU_SVD_BIGRAM_SPHERE`)**.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch  # noqa: E402

from qdlblocks.benchmarks.micro_reversal_readout_lab import (  # noqa: E402
    DEFAULT_MICRO_VOCAB_SIZE,
    DEFAULT_SPAN_LEN,
    FineWebSpanReversalDataset,
    MicroBackboneMode,
    MicroInitMode,
    MicroReadoutMode,
    MicroReversalConfig,
    benchmark_gpu_init_timings_ms,
    run_micro_reversal_experiment,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
DEFAULT_OUT_JSON: Path = _REPO_ROOT / ".cache" / "micro_reversal_init_and_readout_results.json"
DEFAULT_BENCH_STEPS: int = 300
DEFAULT_EVAL_EVERY: int = 100


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the FineWeb Span Reversal readout and initialization benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_BENCH_STEPS, help="Training steps per arm.")
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY, help="Evaluation interval in steps.")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_MICRO_VOCAB_SIZE, help="Active FineWeb vocab size.")
    parser.add_argument("--span-len", type=int, default=DEFAULT_SPAN_LEN, help="Token span length to reverse.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUT_JSON, help="Output JSON artifact path.")
    return parser.parse_args()


def _format_arm_summary(arm_id: str, res: dict[str, Any]) -> str:
    """Format a single-line human-readable progress summary for an experiment arm."""
    s0 = res["step0"]
    fin = res["final"]
    return (
        f"[{arm_id}] params={res['total_params']:,} (readout={res['readout_params']:,}) "
        f"init={res['init_ms']:.2f}ms wall={res['train_wall_s']:.2f}s | "
        f"Step0 CE={s0['val_ce']:.4f} -> Final CE={fin['val_ce']:.4f} "
        f"Top1={fin['top1_acc'] * 100:.2f}% Top5={fin['top5_acc'] * 100:.2f}% "
        f"(Near={fin['near_top1_acc'] * 100:.2f}%, Far={fin['far_top1_acc'] * 100:.2f}%)"
    )


def main() -> None:
    """Run the GPU initialization timing benchmark, 8-arm readout tournament, and 4-arm init ablation."""
    args = parse_args()
    device = torch.device(args.device)
    print(f"[micro_reversal_init_and_readout_bench] Loading FineWeb Span Reversal dataset on {device}...")
    dataset = FineWebSpanReversalDataset(
        vocab_size=args.vocab_size,
        span_len=args.span_len,
        device=device,
    )

    init_timings = benchmark_gpu_init_timings_ms(dataset, repeats=5)
    print(f"[GPU Init Timings (ms)] {json.dumps(init_timings)}")

    readout_arms: list[tuple[str, MicroBackboneMode, MicroReadoutMode]] = [
        ("C_R1_CLASSICAL_LINEAR_SOFTCAP", MicroBackboneMode.CLASSICAL, MicroReadoutMode.R1_LINEAR_SOFTCAP),
        ("C_R2_CLASSICAL_LEVEL_A_RADIAL_SPHERE", MicroBackboneMode.CLASSICAL, MicroReadoutMode.R2_LEVEL_A_RADIAL_SPHERE),
        (
            "C_R3_CLASSICAL_TIED_EXIT_BRIDGE",
            MicroBackboneMode.CLASSICAL,
            MicroReadoutMode.R3_TIED_WAVEPACKET_EXIT_BRIDGE,
        ),
        (
            "C_R4_CLASSICAL_HERMITIAN_QUAD_BORN",
            MicroBackboneMode.CLASSICAL,
            MicroReadoutMode.R4_HERMITIAN_QUADRATIC_BORN,
        ),
        ("Q_R1_QUANTUM_LINEAR_SOFTCAP", MicroBackboneMode.QUANTUM_WAVEPACKET, MicroReadoutMode.R1_LINEAR_SOFTCAP),
        (
            "Q_R2_QUANTUM_LEVEL_A_RADIAL_SPHERE",
            MicroBackboneMode.QUANTUM_WAVEPACKET,
            MicroReadoutMode.R2_LEVEL_A_RADIAL_SPHERE,
        ),
        (
            "Q_R3_QUANTUM_TIED_EXIT_BRIDGE",
            MicroBackboneMode.QUANTUM_WAVEPACKET,
            MicroReadoutMode.R3_TIED_WAVEPACKET_EXIT_BRIDGE,
        ),
        (
            "Q_R4_QUANTUM_HERMITIAN_QUAD_BORN",
            MicroBackboneMode.QUANTUM_WAVEPACKET,
            MicroReadoutMode.R4_HERMITIAN_QUADRATIC_BORN,
        ),
    ]

    readout_results: dict[str, Any] = {}
    for arm_id, b_mode, r_mode in readout_arms:
        cfg = MicroReversalConfig(
            backbone_mode=b_mode,
            init_mode=MicroInitMode.GPU_COUNTSKETCH_BIGRAM_SPHERE,
            readout_mode=r_mode,
            vocab_size=args.vocab_size,
            span_len=args.span_len,
        )
        res = run_micro_reversal_experiment(cfg, dataset, total_steps=args.steps, eval_every=args.eval_every)
        readout_results[arm_id] = res
        print(_format_arm_summary(arm_id, res), flush=True)

    # Select best quantum readout mode by final val_ce for Part 2 initialization ablation
    best_q_readout = min(
        (
            MicroReadoutMode.R1_LINEAR_SOFTCAP,
            MicroReadoutMode.R2_LEVEL_A_RADIAL_SPHERE,
            MicroReadoutMode.R3_TIED_WAVEPACKET_EXIT_BRIDGE,
            MicroReadoutMode.R4_HERMITIAN_QUADRATIC_BORN,
        ),
        key=lambda rm: min(
            v["final"]["val_ce"]
            for k, v in readout_results.items()
            if k.startswith("Q_") and v["readout_mode"] == rm.value
        ),
    )

    init_arms: list[tuple[str, MicroInitMode]] = [
        ("I1_RANDOM_SPHERE_ZERO_PRIOR", MicroInitMode.RANDOM_SPHERE),
        ("I2_GPU_UNIGRAM_PRIOR_ONLY", MicroInitMode.GPU_UNIGRAM_PRIOR),
        ("I3_GPU_COUNTSKETCH_BIGRAM_SPHERE", MicroInitMode.GPU_COUNTSKETCH_BIGRAM_SPHERE),
        ("I4_GPU_LOWRANK_SVD_BIGRAM_SPHERE", MicroInitMode.GPU_SVD_BIGRAM_SPHERE),
    ]

    init_results: dict[str, Any] = {}
    for arm_id, i_mode in init_arms:
        cfg = MicroReversalConfig(
            backbone_mode=MicroBackboneMode.QUANTUM_WAVEPACKET,
            init_mode=i_mode,
            readout_mode=best_q_readout,
            vocab_size=args.vocab_size,
            span_len=args.span_len,
        )
        res = run_micro_reversal_experiment(cfg, dataset, total_steps=args.steps, eval_every=args.eval_every)
        init_results[arm_id] = res
        print(_format_arm_summary(arm_id, res), flush=True)

    payload = {
        "script": "micro_reversal_init_and_readout_bench.py",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "vocab_size": int(args.vocab_size),
        "span_len": int(args.span_len),
        "steps": int(args.steps),
        "gpu_init_timings_ms": init_timings,
        "best_quantum_readout_for_init_ablation": best_q_readout.value,
        "readout_tournament": readout_results,
        "initialization_ablation": init_results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[micro_reversal_init_and_readout_bench] Saved results to {args.json_out}")


if __name__ == "__main__":
    main()
