#!/usr/bin/env python3
r"""CLI Benchmark for Shallow (`L = 4`) Long-Sequence (`T = 512`) `(50304, 8, 2, 32)` Quantum Wavepacket LM.

Evaluates whether our shallow (`L = 4` layers, `d = 512`, `7 Semantic + 1 Identity` complex wavepackets)
Quantum Wavepacket + SVD-Stiefel Mach-Zehnder Interferometer network can break **under `4.00` Val CE**
on unseen FineWeb Shard 0 (`fineweb_val_000000.bin`, `65,536` validation tokens) when trained on
longer sequences (`T = 512`).
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

from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import (  # noqa: E402
    DEFAULT_SHALLOW_BATCH_SIZE,
    DEFAULT_SHALLOW_NUM_LAYERS,
    DEFAULT_SHALLOW_SEQ_LEN,
    LongSeqFineWebStream,
    ShallowArchMode,
    ShallowLongSeqSpec,
    run_shallow_longseq_experiment,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
DEFAULT_OUT_JSON: Path = _REPO_ROOT / ".cache" / "shallow_longseq_wavepacket_lm_results.json"
DEFAULT_STEPS: int = 1200
DEFAULT_EVAL_EVERY: int = 200


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the shallow long-sequence FineWeb LM benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        type=str,
        default="s1_classical_4l_baseline,s2_quantum_wavepacket_4l_pure,s3_quantum_wavepacket_4l_semantic_ngram,s4_quantum_wavepacket_4l_calibrated_synergy",
        help="Comma-separated list of ShallowArchMode values to run.",
    )
    parser.add_argument("--layers", type=int, default=DEFAULT_SHALLOW_NUM_LAYERS)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SHALLOW_SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_SHALLOW_BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUT_JSON)
    return parser.parse_args()


def main() -> None:
    """Run the shallow long-sequence FineWeb LM benchmark."""
    args = parse_args()
    device = torch.device(args.device)
    print(
        f"[shallow_longseq_wavepacket_lm_bench] Initializing FineWeb stream "
        f"(B={args.batch_size}, T={args.seq_len}, L={args.layers}) on {device}...",
        flush=True,
    )
    stream = LongSeqFineWebStream(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
    )

    mode_list = [ShallowArchMode(m.strip()) for m in args.modes.split(",") if m.strip()]
    results: dict[str, Any] = {}
    if args.json_out.exists():
        try:
            prev = json.loads(args.json_out.read_text(encoding="utf-8"))
            results.update(prev.get("results", {}))
        except Exception:
            pass

    for mode in mode_list:
        spec = ShallowLongSeqSpec(
            arch_mode=mode,
            num_layers=args.layers,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
        )
        res = run_shallow_longseq_experiment(
            spec,
            stream,
            total_steps=args.steps,
            eval_every=args.eval_every,
        )
        results[mode.value] = res
        payload = {
            "script": "shallow_longseq_wavepacket_lm_bench.py",
            "seq_len": int(args.seq_len),
            "batch_size": int(args.batch_size),
            "num_layers": int(args.layers),
            "steps": int(args.steps),
            "results": results,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[shallow_longseq_wavepacket_lm_bench] Saved results to {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
