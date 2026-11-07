r"""First-Class CLI Profiler & Benchmark for Fixed-Once Subsystem Quantum Vocabulary Architectures (`EXPERIMENT_PLAN_085.md` / `Wave 65`).

Compares:
1. `W65_V1_LUDERS_256X256_INSTRUMENT` (`LudersSequentialInstrumentBoundary` — $256 \oplus 197$ Bipartite Quantum Instrument),
2. `W65_V2_MPS_16POW4_BORN_CHAIN` (`RightCanonicalMPSBornBoundary` — 4-Site Base-$16$ Right-Canonical MPS Born Machine), and
3. `W65_V3_SHARED_STIEFEL_128D_FRAME` (`SharedStiefelSubspaceFrameBoundary` — Tied `128D` Sphere + `St(512, 128)` Frame).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from qdlblocks.benchmarks.quantum_subsystem_vocab_lab import run_subsystem_vocab_comparison_suite
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import LongSeqFineWebStream


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wave 65 Fixed-Once Subsystem Quantum Vocabulary Profiler & Benchmark")
    parser.add_argument("--steps", type=int, default=60, help="Training steps per subsystem vocabulary architecture")
    parser.add_argument("--eval-interval", type=int, default=20, help="Validation interval in steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size B")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length T")
    parser.add_argument("--num-layers", type=int, default=4, help="Quantum trunk depth L")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps K_accum")
    parser.add_argument("--only-variant", type=str, default="", help="Optional substring filter for variant ID")
    parser.add_argument("--val-tokens", type=int, default=65_536, help="Validation token slice")
    parser.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cuda', or 'cpu')")
    parser.add_argument("--json-out", type=str, default=".cache/plan085_subsystem_vocab_bench.json", help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(
        f"=== WAVE 65: FIXED-ONCE SUBSYSTEM QUANTUM VOCABULARY BENCHMARK ===\n"
        f"Device: {device} | Steps: {args.steps} (eval={args.eval_interval}) | "
        f"Batch: B={args.batch_size}x{args.grad_accum_steps}, T={args.seq_len}, L={args.num_layers} | ValTokens: {args.val_tokens:,}",
        flush=True,
    )
    t0 = time.perf_counter()
    stream = LongSeqFineWebStream(
        seq_len=args.seq_len, batch_size=args.batch_size, val_tokens=args.val_tokens, device=device
    )
    report = run_subsystem_vocab_comparison_suite(
        stream, steps=args.steps, eval_interval=args.eval_interval,
        num_layers=args.num_layers, grad_accum_steps=args.grad_accum_steps, only_variant=args.only_variant,
    )
    report["total_wall_sec"] = round(time.perf_counter() - t0, 2)

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[Saved Wave 65 Subsystem Vocab Benchmark JSON -> {out_path} ({report['total_wall_sec']:.1f}s)]", flush=True)


if __name__ == "__main__":
    main()
