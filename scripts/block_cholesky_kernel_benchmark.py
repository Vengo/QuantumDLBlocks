#!/usr/bin/env python3
"""Block-Cholesky Kernel Benchmark: Two-Pass `(x @ L) @ L^T` vs. Single-Pass SPD `x @ (L L^T)` (Plan 038).

Promoted from `scripts/BACKLOG.md` (identified in Plan 037, implemented in Plan 038).
Benchmarks `BlockCholeskyLinear(fast_spd_bmm=False)` vs. `BlockCholeskyLinear(fast_spd_bmm=True)`
across dimensions `[64, 128, 256]`, verifying:
1. Numerical equivalence (`max_abs_diff < 1e-5` in FP32),
2. SPD condition metrics (`min_diag`, `max_chol_sv`, `min_chol_sv`, `spd_cond_number`),
3. Throughput and latency across token-batch shapes `(8192, d)` and vocabulary-table shapes `(65536, d)`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.layers.wavelet_cholesky import BlockCholeskyLinear

DEFAULT_BENCH_DIMS: tuple[int, ...] = (64, 128, 256)


@dataclass(frozen=True)
class BlockCholeskyBenchConfig:
    """Configuration for benchmarking `BlockCholeskyLinear` slow vs. fast SPD kernels."""

    dims: tuple[int, ...] = DEFAULT_BENCH_DIMS
    max_block_size: int = 32
    token_rows: int = 8192
    vocab_rows: int = 65536
    warmup_iters: int = 2
    bench_iters: int = 8
    seed: int = 2038
    device: str = "cpu"
    out_json: str = ""


def _time_forward_ms(
    layer: BlockCholeskyLinear,
    x: torch.Tensor,
    warmup_iters: int,
    bench_iters: int,
) -> tuple[torch.Tensor, float]:
    """Run warmup and timed forward iterations returning `(output, mean_latency_ms)`."""
    with torch.no_grad():
        out = layer(x)
        for _ in range(max(0, warmup_iters - 1)):
            out = layer(x)
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        t0 = time.perf_counter()
        for _ in range(max(1, bench_iters)):
            out = layer(x)
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        elapsed_ms = ((time.perf_counter() - t0) * 1000.0) / float(max(1, bench_iters))
    return out, elapsed_ms


def benchmark_single_dim(
    dim: int,
    cfg: BlockCholeskyBenchConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Benchmark `fast_spd_bmm=False` vs. `True` for a single feature dimension `dim`."""
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed) + int(dim) * 31)
    blk_sz = min(int(cfg.max_block_size), int(dim))
    slow_layer = BlockCholeskyLinear(dim=dim, max_block_size=blk_sz, fast_spd_bmm=False).to(device)
    fast_layer = BlockCholeskyLinear(dim=dim, max_block_size=blk_sz, fast_spd_bmm=True).to(device)

    with torch.no_grad():
        tril_noise = torch.randn(slow_layer.raw_tril.shape, generator=gen).to(device) * 0.14
        diag_noise = torch.randn(slow_layer.raw_diag.shape, generator=gen).to(device) * 0.10
        slow_layer.raw_tril.copy_(tril_noise)
        slow_layer.raw_diag.copy_(diag_noise)
        fast_layer.load_state_dict(slow_layer.state_dict(), strict=True)

    x_tok = torch.randn((int(cfg.token_rows), dim), generator=gen, dtype=torch.float32).to(device)
    x_voc = torch.randn((int(cfg.vocab_rows), dim), generator=gen, dtype=torch.float32).to(device)

    y_tok_slow, tok_slow_ms = _time_forward_ms(slow_layer, x_tok, cfg.warmup_iters, cfg.bench_iters)
    y_tok_fast, tok_fast_ms = _time_forward_ms(fast_layer, x_tok, cfg.warmup_iters, cfg.bench_iters)
    y_voc_slow, voc_slow_ms = _time_forward_ms(slow_layer, x_voc, cfg.warmup_iters, cfg.bench_iters)
    y_voc_fast, voc_fast_ms = _time_forward_ms(fast_layer, x_voc, cfg.warmup_iters, cfg.bench_iters)

    tok_max_diff = float((y_tok_slow - y_tok_fast).abs().max().item())
    voc_max_diff = float((y_voc_slow - y_voc_fast).abs().max().item())
    max_abs_diff = max(tok_max_diff, voc_max_diff)
    cond_metrics = fast_layer.condition_metrics()

    tok_speedup = tok_slow_ms / max(tok_fast_ms, 1e-6)
    voc_speedup = voc_slow_ms / max(voc_fast_ms, 1e-6)
    last_layer_meas_vs_vocab_speedup = voc_fast_ms / max(tok_fast_ms, 1e-6)

    return {
        "dim": int(dim),
        "block_size": blk_sz,
        "num_blocks": int(dim) // blk_sz,
        "token_rows": int(cfg.token_rows),
        "vocab_rows": int(cfg.vocab_rows),
        "max_abs_diff": max_abs_diff,
        "numerical_equivalence_passed": bool(max_abs_diff < 1e-5),
        "tok_slow_ms": round(tok_slow_ms, 4),
        "tok_fast_ms": round(tok_fast_ms, 4),
        "tok_speedup_ratio": round(tok_speedup, 3),
        "vocab_slow_ms": round(voc_slow_ms, 4),
        "vocab_fast_ms": round(voc_fast_ms, 4),
        "vocab_speedup_ratio": round(voc_speedup, 3),
        "last_layer_meas_vs_vocab_speedup_ratio": round(last_layer_meas_vs_vocab_speedup, 3),
        "condition_metrics": cond_metrics,
    }


def run_block_cholesky_benchmark(cfg: BlockCholeskyBenchConfig) -> dict[str, Any]:
    """Run the `BlockCholeskyLinear` benchmark across all configured dimensions and print a Markdown summary."""
    device = torch.device(cfg.device)
    rows: list[dict[str, Any]] = [benchmark_single_dim(int(d), cfg, device) for d in cfg.dims]
    all_equiv = all(bool(r["numerical_equivalence_passed"]) for r in rows)

    print(
        f"\n### BlockCholeskyLinear Kernel Benchmark (`token_rows={cfg.token_rows}`, "
        f"`vocab_rows={cfg.vocab_rows}`, `device={cfg.device}`)"
    )
    print(
        "| Dim `d` | Blocks | Max Abs Diff | Cond `kappa(L L^T)` | "
        "Token Slow (`ms`) | Token Fast (`ms`) | Vocab Slow (`ms`) | Vocab Fast (`ms`) | "
        "Last-Layer `(B*T)` vs Vocab `(N)` Speedup |"
    )
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in rows:
        cond = r["condition_metrics"]["spd_cond_number"]
        print(
            f"| `{r['dim']}` | `{r['num_blocks']}x{r['block_size']}` | `{r['max_abs_diff']:.2e}` | "
            f"`{cond:.3f}` | `{r['tok_slow_ms']:.3f}` | `{r['tok_fast_ms']:.3f}` | "
            f"`{r['vocab_slow_ms']:.3f}` | `{r['vocab_fast_ms']:.3f}` | "
            f"**`{r['last_layer_meas_vs_vocab_speedup_ratio']:.2f}x`** |"
        )

    report = {
        "config": asdict(cfg),
        "all_numerical_equivalence_passed": all_equiv,
        "dimensions": {str(r["dim"]): r for r in rows},
        "rows": rows,
    }
    if cfg.out_json:
        out_p = Path(cfg.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> BlockCholeskyBenchConfig:
    """Parse CLI arguments into a frozen `BlockCholeskyBenchConfig`."""
    p = argparse.ArgumentParser(
        description="Benchmark BlockCholeskyLinear slow two-pass vs. fast single-pass SPD kernels."
    )
    p.add_argument("--dims", type=str, default="64,128,256")
    p.add_argument("--max-block-size", type=int, default=32)
    p.add_argument("--token-rows", type=int, default=8192)
    p.add_argument("--vocab-rows", type=int, default=65536)
    p.add_argument("--warmup-iters", type=int, default=2)
    p.add_argument("--bench-iters", type=int, default=8)
    p.add_argument("--seed", type=int, default=2038)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-json", type=str, default="")
    ns = p.parse_args(argv)
    dims_tuple = tuple(int(x.strip()) for x in ns.dims.split(",") if x.strip())
    return BlockCholeskyBenchConfig(
        dims=dims_tuple,
        max_block_size=ns.max_block_size,
        token_rows=ns.token_rows,
        vocab_rows=ns.vocab_rows,
        warmup_iters=ns.warmup_iters,
        bench_iters=ns.bench_iters,
        seed=ns.seed,
        device=ns.device,
        out_json=ns.out_json,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    rep = run_block_cholesky_benchmark(parse_args(argv))
    return 0 if bool(rep["all_numerical_equivalence_passed"]) else 1


if __name__ == "__main__":
    sys.exit(main())
