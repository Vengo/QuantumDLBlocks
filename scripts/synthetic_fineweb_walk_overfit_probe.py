#!/usr/bin/env python3
"""Standalone CLI Diagnostic Probe for Plan 029: FineWeb-Matched 50,257-Vocab Synthetic Walk & Quantum Attention Overfitting.

Usage:
    /home/festivity/agy/QuantumDLBlocks/.venv/bin/python scripts/synthetic_fineweb_walk_overfit_probe.py
    /home/festivity/agy/QuantumDLBlocks/.venv/bin/python scripts/synthetic_fineweb_walk_overfit_probe.py --quick-test
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

from qdlblocks.benchmarks.synthetic_fineweb_walk import run_synthetic_fineweb_overfit_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the FineWeb-Matched 50,257-Vocabulary Synthetic Random-Walk & Small Quantum Attention Overfitting Probe (<= 8.0 GB VRAM)."
    )
    parser.add_argument("--num-seqs", type=int, default=64, help="Number of walk sequences in Task A corpus (default: 64).")
    parser.add_argument("--seq-len", type=int, default=128, help="Walk sequence length T in Task A (default: 128).")
    parser.add_argument("--steps", type=int, default=300, help="Optimization steps for Task A overfitting (default: 300).")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size for Task A training (default: 16).")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed (default: 1337).")
    parser.add_argument("--quick-test", action="store_true", help="Run a fast 120-step verification.")
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / ".cache" / "synthetic_fineweb_walk_overfit_probe.json"),
        help="Output JSON path.",
    )
    return parser.parse_args()


def print_suite_summary(res: dict) -> None:
    st = res["h1_walk_statistics"]
    print("\n" + "=" * 108)
    print("  [H1] FineWeb-Edu vs. Synthetic 50,257-Vocab Random-Walk Statistical Invariant Comparison (64 x 128 = 8,192 Tok)")
    print("=" * 108)
    print(
        f"  * Vocabulary Span       : V_valid={st['vocab_valid']:,} | Unique Visited={st['unique_tokens']:,} "
        f"(>=2,000) | Max Non-EOT ID={st['max_non_eot_token_id']:,} (>=50,000) | EOT(50256) Count={st['eot_token_count']}"
    )
    print(
        f"  * Unigram Entropy (H_1) : Empirical={st['empirical_unigram_entropy_nats']:.4f} nats | "
        f"Theoretical Zipf Prior={st['theoretical_zipf_entropy_nats']:.4f} nats | FineWeb Reference=7.6367 nats"
    )
    print(
        f"  * Zipf Concentration    : Top-100 Mass={st['top100_mass_fineweb_rank']:.4f} (FW-Rank) / "
        f"{st['top100_mass_empirical']:.4f} (Emp) [FW=0.4310] | Top-1000 Mass={st['top1000_mass_fineweb_rank']:.4f} "
        f"(FW-Rank) / {st['top1000_mass_empirical']:.4f} (Emp) [FW=0.6437]"
    )
    print(
        f"  * 1st-Order Markov Bound: H_bigram(X_{{t+1}}|X_t)={st['empirical_bigram_cond_entropy_nats']:.4f} nats (>=1.35) | "
        f"Oracle Markov Bigram Top-1={st['oracle_markov_bigram_top1_acc']:.2f}% (<=48.0%) | H1 PASS={res['h1_pass']}"
    )

    print("\n" + "=" * 108)
    print("  [H2] Task A — Full 50,304-Vocabulary Synthetic FineWeb Random-Walk Overfitting Benchmark (<= 8.0 GB VRAM)")
    print("=" * 108)
    header = (
        f"{'Model Arm':<58} | {'Params':>7} | {'VRAM':>6} | {'Step 0 CE':>9} | "
        f"{'Final CE':>8} | {'Top-1':>7} | {'Top-5':>7} | {'BeamOrtho':>9} | {'Wall':>6}"
    )
    print(header)
    print("-" * len(header))
    for rk, arm in res["task_a_arms"].items():
        print(
            f"{rk:<58} | {arm['params_M']:>6.2f}M | {arm['peak_vram_gb']:>5.3f}G | "
            f"{arm['step0_ce']:>9.4f} | {arm['final_ce']:>8.4f} | {arm['top1_acc']:>6.2f}% | "
            f"{arm['top5_acc']:>6.2f}% | {arm['beam_ortho']:>9.2e} | {arm['wall_s']:>5.2f}s"
        )
    print(f"  * H2 PASS={res['h2_pass']} | Max Peak VRAM={res['max_peak_vram_gb']:.4f} GB (Cap <= 8.0 GB)")

    tb = res["task_b_variable_lag_hub"]
    rad = tb["TaskB_CollinearRadialSubProbe"]
    print("\n" + "=" * 108)
    print("  [H3] Task B — Position-Shifted Variable-Lag (Delta in [8, 32]) Hub & Collinear Radial-Shell Disambiguation")
    print("=" * 108)
    for m_name in ("OracleMarkovBigramFloor", "FastQuantumNoAttnMLP", "FastQuantumBornAttentionSynthLM"):
        info = tb[m_name]
        print(f"  * [Task B Variable-Lag Hub (Delta in [8, 32])] {m_name:<34} -> Held-Out Hub CE={info['hub_ce']:.4f} | Hub Top-1={info['hub_top1_acc']:.2f}%")
    print(
        f"  * [Task B-Radial Collinear Shells (rho in {{0.55, 2.75}} at Delta in [8, 32])] "
        f"StereographicLift=True: Acc={rad['stereographic_lift_true']['eval_acc']:.2f}% (CE={rad['stereographic_lift_true']['eval_ce']:.4f}) vs. "
        f"NaiveRadialErasing=False: Acc={rad['naive_radial_erasing_false']['eval_acc']:.2f}% (CE={rad['naive_radial_erasing_false']['eval_ce']:.4f})"
    )
    print(
        f"  * H3 PASS={res['h3_pass']} | OVERALL PLAN 029 PASS={res['overall_pass']} | Total Wall Time={res['suite_wall_s']:.2f}s"
    )
    print("=" * 108 + "\n")


def main() -> None:
    args = parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.5, device=0)
    steps = 160 if args.quick_test else int(args.steps)
    res = run_synthetic_fineweb_overfit_suite(
        num_seqs=int(args.num_seqs),
        seq_len=int(args.seq_len),
        steps=steps,
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )
    print_suite_summary(res)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Saved Plan 029 synthetic FineWeb walk & quantum attention telemetry -> {out_path}")
    if not res["overall_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
