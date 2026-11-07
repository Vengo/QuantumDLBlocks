#!/usr/bin/env python3
r"""CLI Benchmark Runner for Quantum No-Click POVM Vacuum Sink, 100% `softmax`-Free Vacuum-Born Readout, $R = 12$ vs. $R = 8$ 1,200-Step Scaling, and Leslie Smith $\beta = 0$ SGD Super-Convergence (`EXPERIMENT_PLAN_080.md`).

Executes:
1. 300-step (`2.46M`-token) Quantum No-Click POVM / Vacuum Sink ablation (`N1` at $R = 8$, `N2` at $R = 12$, or loads cached results via `--skip-ablation`),
2. Optional 300-step (`2.46M`-token) 100% `softmax`-Free Vacuum-Sharpened Coincidence Born ablation (`B0, B1, B2` via `--run-softmax-free`),
3. Optional Leslie Smith Momentum-Free ($\beta = 0$) SGD Exponential LR Range Test & 300-step 1-Cycle Super-Convergence (`S0, S1` via `--run-smith-sgd`), and
4. Optional 1,200-step (`9.83M`-token) 100% Pure-Neural scaling run at $R = 12$ (`N4` via `--run-long-scaling`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.benchmarks.born_schur_density_lab import DEFAULT_FINEWEB_EVAL_INTERVAL, DEFAULT_FINEWEB_STEPS  # noqa: E402
from qdlblocks.benchmarks.noclick_vacuum_and_smith_sgd_lab import (  # noqa: E402
    DEFAULT_SMITH_RANGE_STEPS,
    run_noclick_vacuum_ablation_benchmark,
    run_rank12_vs_rank8_long_scaling_benchmark,
    run_smith_momentum_free_superconvergence_benchmark,
)
from qdlblocks.benchmarks.softmax_free_vacuum_born_lab import (  # noqa: E402
    run_softmax_free_vacuum_born_ablation_benchmark,
)
from qdlblocks.benchmarks.wavefunction_outer_density_lab import (  # noqa: E402
    DEFAULT_LONG_FINEWEB_EVAL_INTERVAL,
    DEFAULT_LONG_FINEWEB_STEPS,
)


def _load_cached_ablation_results() -> list[dict[str, Any]]:
    """Load already-completed `N1` and `N2` 300-step runs from `.cache/live_logs/` if present."""
    cached: list[dict[str, Any]] = []
    specs = [
        ("N1_RANK8_NOCLICK_VACUUM_POVM", "n1_rank8_noclick_vacuum_povm", 8, ".cache/live_logs/plan080_n1_rank8_noclick_vacuum_povm_300steps_results.json"),
        ("N2_RANK12_NOCLICK_VACUUM_POVM", "n2_rank12_noclick_vacuum_povm", 12, ".cache/live_logs/plan080_n2_rank12_noclick_vacuum_povm_300steps_results.json"),
    ]
    for var_id, var_mode, r_modes, rel_path in specs:
        p = ROOT / rel_path
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            evals = data.get("evals", [])
            if evals:
                for ev in evals:
                    if ev.get("top1_acc", 0.0) > 1.0:
                        ev["top1_acc"] = round(ev["top1_acc"] / 100.0, 4)
                    if ev.get("top5_acc", 0.0) > 1.0:
                        ev["top5_acc"] = round(ev["top5_acc"] / 100.0, 4)
                cached.append({
                    "variant_id": var_id,
                    "variant_mode": var_mode,
                    "description": data.get("hypothesis_intent", var_id),
                    "num_cholesky_modes": r_modes,
                    "vacuum_dim": 4,
                    "use_noclick_vacuum_sink": True,
                    "momentum_beta": 0.70,
                    "elapsed_sec": data.get("wall_s", 0.0),
                    "step0": evals[0],
                    "final": evals[-1],
                    "curve": evals,
                })
    return cached


def _save_checkpoint(
    out_path: Path,
    steps: int,
    eval_interval: int,
    t0: float,
    ablation_results: list[dict[str, Any]],
    softmax_free_results: list[dict[str, Any]],
    smith_results: dict[str, Any] | None,
    long_results: list[dict[str, Any]],
) -> None:
    """Atomically write incremental benchmark results JSON (`<= 8` args)."""
    payload = {
        "plan": "EXPERIMENT_PLAN_080",
        "wave": "W60",
        "steps": steps,
        "eval_interval": eval_interval,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "ablation_results": ablation_results,
        "softmax_free_born_results": softmax_free_results,
        "smith_momentum_free_sgd": smith_results,
        "long_scaling_results": long_results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EXPERIMENT_PLAN_080 No-Click POVM, Softmax-Free Born, R=12 Scaling, and Smith SGD benchmarks.")
    parser.add_argument("--steps", type=int, default=DEFAULT_FINEWEB_STEPS, help="Training steps for ablation runs (default: 300)")
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_FINEWEB_EVAL_INTERVAL, help="Evaluation interval (default: 100)")
    parser.add_argument("--skip-ablation", action="store_true", help="Load cached N1/N2 300-step results from .cache/live_logs/")
    parser.add_argument("--run-softmax-free", action="store_true", help="Run 100% softmax-free Vacuum-Sharpened Coincidence Born ablation (B0, B1, B2)")
    parser.add_argument("--run-smith-sgd", action="store_true", help="Run Leslie Smith beta=0 SGD LR Range Test & 1-Cycle Super-Convergence (S0, S1)")
    parser.add_argument("--smith-range-steps", type=int, default=DEFAULT_SMITH_RANGE_STEPS, help="Steps for Smith LR Range Test (default: 80)")
    parser.add_argument("--run-long-scaling", action="store_true", help="Run 1,200-step R=12 scaling run (N4)")
    parser.add_argument("--long-steps", type=int, default=DEFAULT_LONG_FINEWEB_STEPS, help="Steps for long scaling runs (default: 1200)")
    parser.add_argument("--long-eval-interval", type=int, default=DEFAULT_LONG_FINEWEB_EVAL_INTERVAL, help="Eval interval for long runs")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cuda or cpu)")
    parser.add_argument("--json-out", type=str, default=".cache/noclick_vacuum_and_smith_sgd_results.json", help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    out_path = Path(args.json_out)

    ablation_results: list[dict[str, Any]] = []
    if args.skip_ablation:
        ablation_results = _load_cached_ablation_results()
        print(f"=== Loaded {len(ablation_results)} cached 300-step No-Click POVM ablation runs (N1, N2) ===", flush=True)
    else:
        print(f"=== [EXPERIMENT_PLAN_080] 300-Step Quantum No-Click POVM Vacuum Sink Ablation (steps={args.steps}) ===", flush=True)
        ablation_results = run_noclick_vacuum_ablation_benchmark(
            steps=args.steps, eval_interval=args.eval_interval, device_str=args.device
        )

    softmax_free_results: list[dict[str, Any]] = []
    smith_results: dict[str, Any] | None = None
    long_results: list[dict[str, Any]] = []
    _save_checkpoint(out_path, args.steps, args.eval_interval, t0, ablation_results, softmax_free_results, smith_results, long_results)

    if args.run_softmax_free:
        print(f"\n=== [EXPERIMENT_PLAN_080] 100% Softmax-Free Vacuum-Sharpened Coincidence Born Ablation (steps={args.steps}) ===", flush=True)
        softmax_free_results = run_softmax_free_vacuum_born_ablation_benchmark(
            steps=args.steps, eval_interval=args.eval_interval, device_str=args.device
        )
        _save_checkpoint(out_path, args.steps, args.eval_interval, t0, ablation_results, softmax_free_results, smith_results, long_results)

    if args.run_smith_sgd:
        print(
            f"\n=== [EXPERIMENT_PLAN_080] Leslie Smith Momentum-Free (beta=0) SGD LR Range Test ({args.smith_range_steps} steps) "
            f"& 1-Cycle Super-Convergence ({args.steps} steps) ===",
            flush=True,
        )
        smith_results = run_smith_momentum_free_superconvergence_benchmark(
            steps=args.steps, eval_interval=args.eval_interval, range_steps=args.smith_range_steps, device_str=args.device
        )
        _save_checkpoint(out_path, args.steps, args.eval_interval, t0, ablation_results, softmax_free_results, smith_results, long_results)

    if args.run_long_scaling:
        print(f"\n=== [EXPERIMENT_PLAN_080] 1,200-Step R=12 vs R=8 Pure-Neural Scaling Run (steps={args.long_steps}) ===", flush=True)
        long_results = run_rank12_vs_rank8_long_scaling_benchmark(
            steps=args.long_steps, eval_interval=args.long_eval_interval, device_str=args.device
        )
        _save_checkpoint(out_path, args.steps, args.eval_interval, t0, ablation_results, softmax_free_results, smith_results, long_results)

    print("\n=== Summary Table (EXPERIMENT_PLAN_080) ===")
    all_runs = list(ablation_results) + list(softmax_free_results) + list(long_results)
    if smith_results is not None:
        all_runs.extend(smith_results.get("superconvergence_runs", []))
    for r in all_runs:
        s0, fin = r["step0"], r["final"]
        print(
            f"{r['variant_id']:38s} | R={r['num_cholesky_modes']:2d} | sink={str(r['use_noclick_vacuum_sink']):5s} | "
            f"beta={r['momentum_beta']:.2f} | Step0 CE: {s0['val_ce']:.4f} -> Final Pure Neural Val CE: {fin['val_ce']:.4f} "
            f"(PPL {fin['val_ppl']:.2f}) | Top1: {fin['top1_acc']*100:.2f}% | Top5: {fin['top5_acc']*100:.2f}% | "
            f"Modes: {fin['effective_active_branches']:.2f}/{r['num_cholesky_modes']} | "
            f"p_noclick: {fin['mean_noclick_prob']:.3f}"
        )


if __name__ == "__main__":
    main()
