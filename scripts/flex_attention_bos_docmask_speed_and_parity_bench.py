#!/usr/bin/env python3
"""Benchmark `<BOS>` Document-Masked `FlexAttention` vs Dense Boolean `SDPA` (`Wave 79` / `EXPERIMENT_PLAN_099.md`).

Promoted from `scripts/TOOLING_BACKLOG.md` (`Wave 78` candidate). Verifies `bfloat16` output/gradient
numerical parity and measures forward+backward throughput (`tok/s`, `speedup_x`, peak VRAM) across
sequence lengths $T \\in \\{256, 512, 1024, 2048\\}$ on real FineWeb Shard-0 `<BOS>` document spans.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qdlblocks.benchmarks.wave79_speedrun_and_ablation_runner import (  # noqa: E402
    FlexDocMaskBenchConfig,
    benchmark_flex_attention_bos_docmask_parity_and_speed,
)

DEFAULT_JSON_OUT: Path = REPO_ROOT / ".cache" / "wave79_flex_docmask_bench.json"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments (`<= 8` args)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seq-lens",
        type=str,
        default="256,512,1024,2048",
        help="Comma-separated sequence lengths to benchmark.",
    )
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--batch-tokens", type=int, default=16384)
    parser.add_argument("--bench-iters", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    return parser.parse_args()


def main() -> None:
    """Run `<BOS>` `FlexAttention` vs Dense `SDPA` parity & throughput benchmark."""
    args = parse_args()
    seq_lens = tuple(int(s.strip()) for s in str(args.seq_lens).split(",") if s.strip())
    cfg = FlexDocMaskBenchConfig(
        seq_lens=seq_lens,
        num_heads=int(args.num_heads),
        head_real_dim=int(args.head_dim),
        total_batch_tokens=int(args.batch_tokens),
        bench_iters=int(args.bench_iters),
    )
    device = torch.device(args.device)
    report = benchmark_flex_attention_bos_docmask_parity_and_speed(cfg, device=device)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"=== Wave 79 <BOS> FlexAttention vs Dense SDPA Benchmark ({report['gpu_name']}, "
        f"BLOCK_SIZE={report['block_size']}) ==="
    )
    print(
        "| Seq T | Batch B | Doc-Causal Pair% | Output Cos | Grad Cos | Max |Δ| | "
        "Dense SDPA tok/s | Flex BlockMask tok/s | Speedup |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in report["results"]:
        print(
            f"| {r['seq_len']:4d} | {r['batch_size']:3d} | "
            f"{100.0 * r['active_doc_causal_pair_frac']:6.2f}% | "
            f"{r['output_cosine_sim']:.6f} | {r['grad_q_cosine_sim']:.6f} | "
            f"{r['max_abs_diff']:.5f} | {r['dense_sdpa_tok_per_s']:11,.1f} | "
            f"{r['flex_blockmask_tok_per_s']:12,.1f} | **{r['speedup_x']:.2f}x** |"
        )
    print(f"\nSaved JSON report -> {args.json_out}")


if __name__ == "__main__":
    main()
