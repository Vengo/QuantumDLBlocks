#!/usr/bin/env python3
"""First-Class CLI Profiler & Benchmark Runner for Wave 66 (`EXPERIMENT_PLAN_086.md`): Horizontal-Grassmannian & Gauge-Free Quantum GPT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

from qdlblocks.benchmarks.explicit_density_tomographic_512d_lab import DEFAULT_TOMO_NUM_LAYERS
from qdlblocks.benchmarks.horizontal_grassmann_gauge_free_lab import (
    HorizontalGrassmannGaugeFreeGPT,
    Wave66GaugeVariantMode,
    build_wave66_gauge_free_optimizers,
    build_wave66_variant_spec,
    run_wave66_gauge_free_comparison_suite,
    step_wave66_optimizers,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import LongSeqFineWebStream


def profile_wave66_gauge_spin(
    num_layers: int = DEFAULT_TOMO_NUM_LAYERS,
    batch_size: int = 4,
    seq_len: int = 128,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    """Profile parameter breakdown, MUB unbiasedness error, and vertical gauge-spin fractions across W66_G0..W66_G3 (`<= 8` args)."""
    idx = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    targets = torch.randint(0, 50257, (batch_size, seq_len), device=device)

    report: dict[str, Any] = {}
    for mode in Wave66GaugeVariantMode:
        torch.manual_seed(1337)
        spec = build_wave66_variant_spec(mode, num_layers=num_layers, seed=1337)
        model = HorizontalGrassmannGaugeFreeGPT(spec).to(device)
        model.initialize_model_()
        optimizers = build_wave66_gauge_free_optimizers(model)
        q_trunk_opt = optimizers[0]

        for _ in range(3):
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss, _ = model(idx, targets)
            loss.backward()
            step_wave66_optimizers(model, optimizers, tokens_in_step=int(targets.numel()))

        total_params = sum(p.numel() for p in model.parameters())
        gd = {k: round(float(v), 6) for k, v in q_trunk_opt.last_gauge_diagnostics.items()}
        report[mode.value] = {
            "total_params": total_params,
            "mub_unbiased_err_after_3_steps": round(model.current_mub_unbiasedness_error(), 6),
            "gauge_diagnostics": gd,
        }
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wave 66 Horizontal-Grassmannian & Gauge-Free Quantum GPT Runner")
    parser.add_argument("--steps", type=int, default=60, help="Training steps per Wave 66 variant")
    parser.add_argument("--eval-interval", type=int, default=20, help="Validation interval in steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size B")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length T")
    parser.add_argument("--num-layers", type=int, default=DEFAULT_TOMO_NUM_LAYERS, help="Quantum trunk depth L")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps K_accum")
    parser.add_argument("--only-variant", type=str, default="", help="Optional substring filter for variant ID")
    parser.add_argument("--val-tokens", type=int, default=65_536, help="Validation token slice")
    parser.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cuda', or 'cpu')")
    parser.add_argument("--profile-only", action="store_true", help="Only run 3-step gauge-spin profiler")
    parser.add_argument("--json-out", type=str, default=".cache/plan086_horizontal_grassmann_bench.json", help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    """Parse CLI args and run Wave 66 gauge-spin profiler and/or comparative benchmark suite."""
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    print(
        f"=== WAVE 66: HORIZONTAL-GRASSMANNIAN & GAUGE-FREE QUANTUM GPT BENCHMARK ===\n"
        f"Device: {device} | Steps: {args.steps} (eval={args.eval_interval}) | "
        f"Batch: B={args.batch_size}x{args.grad_accum_steps}, T={args.seq_len}, L={args.num_layers} | ValTokens: {args.val_tokens:,}",
        flush=True,
    )
    t0 = time.perf_counter()
    profile_report = profile_wave66_gauge_spin(
        num_layers=args.num_layers, batch_size=min(args.batch_size, 4), seq_len=min(args.seq_len, 128), device=device
    )
    payload: dict[str, Any] = {"profile_report": profile_report}
    if not args.profile_only:
        stream = LongSeqFineWebStream(
            seq_len=args.seq_len, batch_size=args.batch_size, val_tokens=args.val_tokens, device=device
        )
        bench_report = run_wave66_gauge_free_comparison_suite(
            stream,
            steps=args.steps,
            eval_interval=args.eval_interval,
            num_layers=args.num_layers,
            grad_accum_steps=args.grad_accum_steps,
            only_variant=args.only_variant,
        )
        payload.update(bench_report)
    payload["total_wall_sec"] = round(time.perf_counter() - t0, 2)

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Saved Wave 66 Benchmark JSON -> {out_path} ({payload['total_wall_sec']:.1f}s)]", flush=True)


if __name__ == "__main__":
    main()
