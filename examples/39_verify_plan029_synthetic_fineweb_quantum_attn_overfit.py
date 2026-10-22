#!/usr/bin/env python3
"""End-to-End Verification Script for Experiment Plan 029 (`<= 8.0 GB` VRAM on Local RTX 5060 Ti).

Verifies:
- `H1`: FineWeb-Edu statistical fidelity of the `V = 50,257` quota-balanced synthetic random walk
  (`H_1 in [7.00, 8.20] nats`, Top-100 mass in `[0.35, 0.52]`, Top-1000 mass in `[0.55, 0.75]`,
  `>= 2,000` unique tokens, `EOT = 50256`, and `H_bigram(X_{t+1} | X_t) >= 1.35 nats`).
- `H2`: Rapid full-vocabulary (`V = 50,304`) overfitting of `FastQuantumBornAttentionSynthLM`
  (`L=2, d=256, H=4`) to `Train CE <= 0.10 nats` and `>= 98.5%` Top-1 accuracy in `<= 300` steps
  (`< 1.0 GB` Peak VRAM), breaking through the 1st-order Markov floor where `FastQuantumNoAttnMLP`
  (`skip_attn_layers=(0, 1)`) and `NaiveQuadraticBornAttnLM` fail.
- `H3`: Position-shifted variable-lag (`Delta in [8, 32]`) hub and collinear radial-shell disambiguation.
Saves `.cache/hypothesis_experiments_029.json`, updates `.cache/benchmarks.db`, and triggers hot reload.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import urllib.request

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.benchmarks.db import DEFAULT_DB_PATH, BenchmarkSQLiteDB, backfill_micro_verifications
from qdlblocks.benchmarks.synthetic_fineweb_walk import run_synthetic_fineweb_overfit_suite

sys.path.insert(0, str(ROOT / "scripts"))
from synthetic_fineweb_walk_overfit_probe import print_suite_summary


def main() -> None:
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.5, device=0)
        dev_name = torch.cuda.get_device_name(0)
        tot_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(
            f"Running Plan 029 Synthetic FineWeb 50,257-Vocab Walk & Quantum Attention Overfit Suite "
            f"on device=cuda ({dev_name}, Total={tot_gb:.2f}GB, Cap<=8.0GB / 50%)"
        )
    else:
        print("Running Plan 029 Suite on device=cpu")

    res = run_synthetic_fineweb_overfit_suite(
        num_seqs=64,
        seq_len=128,
        steps=300,
        batch_size=16,
        seed=1337,
    )
    print_suite_summary(res)

    out_json = ROOT / ".cache" / "hypothesis_experiments_029.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Saved Plan 029 verification telemetry -> {out_json}")

    try:
        db = BenchmarkSQLiteDB(DEFAULT_DB_PATH)
        n_back = backfill_micro_verifications(db, cache_dir=ROOT / ".cache")
        print(f"Backfilled {n_back} micro-verification runs into {DEFAULT_DB_PATH}")
    except Exception as exc:
        print(f"Note: SQLite DB backfill skipped/warned: {exc}")

    try:
        req = urllib.request.Request("http://127.0.0.1:8765/api/reload", method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            print("Live dashboard reload status:", json.loads(resp.read().decode("utf-8")))
    except Exception:
        pass

    q_ce = res["task_a_arms"]["w17_synth_qborn_fast_quantum_attn_stereo_phase_quad_L2_d256"]["final_ce"]
    q_top1 = res["task_a_arms"]["w17_synth_qborn_fast_quantum_attn_stereo_phase_quad_L2_d256"]["top1_acc"]
    no_attn_ce = res["task_a_arms"]["w17_synth_qborn_no_attn_control_L2_d256"]["final_ce"]
    print("=" * 108)
    print(
        f"  PLAN 029 SUMMARY: H1={'PASS' if res['h1_pass'] else 'FAIL'} | "
        f"H2={'PASS' if res['h2_pass'] else 'FAIL'} | "
        f"H3={'PASS' if res['h3_pass'] else 'FAIL'} | "
        f"FastQuantumBornAttentionSynthLM={q_ce:.4f} CE ({q_top1:.2f}% Top-1) vs NoAttn={no_attn_ce:.4f} CE | "
        f"OVERALL={'PASS' if res['overall_pass'] else 'FAIL'}"
    )
    print("=" * 108)
    if not res["overall_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
