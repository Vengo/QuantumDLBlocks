#!/usr/bin/env python3
r"""CLI Benchmark for Support-Preserving CountSketch & Meaning-vs-Identity Wavepackets (`V = 50,304`).

Evaluates 6 GPU initialization strategies (`I2..I7`) on the full `50,304`-token FineWeb vocabulary across:
1. **Sphere Geometry & Crowding Diagnostics**:
   - `rare_max_cos`: Max pairwise cosine among 512 unseen/rare tokens ($c_v = 0$) — detects rare-token collapse.
   - `freq_max_cos`: Max pairwise cosine among top 512 frequent tokens — measures synonym crowding.
   - `semantic_gram_corr`: Pearson correlation between embedding Gram matrix and raw bigram PPMI Gram matrix.
2. **Task A — Exact Token Identity (`FineWeb Span Reversal`, $V = 50,304$)**:
   Tests whether support-preserving initializers (`I5, I6, I7`) eliminate the `99.07%` rare-token ceiling of raw CountSketch (`I3`) and recover `100.00%` Top-1 accuracy.
3. **Task B — Contextual Meaning + Identity (`FineWeb Causal Next-Token Language Modeling`, $V = 50,304$)**:
   Tests how shared semantic similarity vs. unique identity affects next-token language modeling cross-entropy.
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
    DEFAULT_SPAN_LEN,
    FineWebSpanReversalDataset,
)
from qdlblocks.benchmarks.support_preserving_countsketch_lab import (  # noqa: E402
    DEFAULT_FULL_FINEWEB_VOCAB_SIZE,
    SupportSketchMode,
    SupportSketchSpec,
    run_support_sketch_experiment,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
DEFAULT_OUT_JSON: Path = _REPO_ROOT / ".cache" / "support_preserving_countsketch_v50304_results.json"
DEFAULT_STEPS: int = 300
DEFAULT_EVAL_EVERY: int = 100


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the support-preserving CountSketch benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_FULL_FINEWEB_VOCAB_SIZE)
    parser.add_argument("--span-len", type=int, default=DEFAULT_SPAN_LEN)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUT_JSON)
    return parser.parse_args()


def _format_row(arm_name: str, res: dict[str, Any]) -> str:
    """Format a single-line human-readable summary for a completed experiment arm."""
    g = res["geometry"]
    s0 = res["step0"]
    s100 = next((c for c in res["checkpoints"] if c["step"] == 100), res["final"])
    fin = res["final"]
    extra_stream = (
        f" stream={g['cold_stream_ms']:.0f}ms({g['tokens_streamed_M']:.0f}M,cov={g['active_vocab_pct']:.2f}%)"
        if "cold_stream_ms" in g
        else ""
    )
    return (
        f"[{arm_name} | {res['task_mode']}] init={res['init_ms']:.2f}ms{extra_stream} wall={res['train_wall_s']:.1f}s | "
        f"rare_max_cos={g['rare_max_cos']:.4f} freq_max_cos={g['freq_max_cos']:.4f} sem_corr={g['semantic_gram_corr']:.4f} | "
        f"Step0 CE={s0['val_ce']:.4f} ({s0['top1_acc'] * 100:.2f}%) -> Step100 CE={s100['val_ce']:.4f} ({s100['top1_acc'] * 100:.2f}%) -> "
        f"Final CE={fin['val_ce']:.4f} Top1={fin['top1_acc'] * 100:.2f}% Top5={fin['top5_acc'] * 100:.2f}%"
    )


def main() -> None:
    """Run the initializer comparison across both Reversal and Causal Next-Token LM at V=50,304."""
    args = parse_args()
    device = torch.device(args.device)
    print(f"[support_preserving_countsketch_bench] Loading FineWeb V={args.vocab_size} dataset on {device}...")
    dataset = FineWebSpanReversalDataset(
        vocab_size=args.vocab_size,
        span_len=args.span_len,
        device=device,
    )

    modes: list[tuple[str, SupportSketchMode, str | None]] = [
        ("I2_UNIGRAM_PRIOR_RANDOM_SPHERE", SupportSketchMode.I2_UNIGRAM_PRIOR_RANDOM_SPHERE, "I2_GPU_UNIGRAM_PRIOR_ONLY"),
        ("I3_RAW_COUNTSKETCH_SPHERE", SupportSketchMode.I3_RAW_COUNTSKETCH_SPHERE, "I3_GPU_COUNTSKETCH_BIGRAM_SPHERE"),
        ("I4_LOWRANK_SVD_SPHERE", SupportSketchMode.I4_LOWRANK_SVD_SPHERE, "I4_GPU_LOWRANK_SVD_BIGRAM_SPHERE"),
        ("I5_SUPPORT_GATED_COUNTSKETCH", SupportSketchMode.I5_SUPPORT_GATED_COUNTSKETCH, None),
        ("I6_SPLIT_MEANING_ID_WAVEPACKET", SupportSketchMode.I6_SPLIT_MEANING_ID_WAVEPACKET, None),
        ("I7_WHITENED_SUPPORT_COUNTSKETCH", SupportSketchMode.I7_WHITENED_SUPPORT_COUNTSKETCH, None),
        ("I8_SPLIT_MEANING_ID_ADAPTIVE_PKT_GATE", SupportSketchMode.I8_SPLIT_MEANING_ID_ADAPTIVE_PKT_GATE, None),
        ("I9_COMPLEX_WAVEFUNCTION_4x2x32", SupportSketchMode.I9_COMPLEX_WAVEFUNCTION_4x2x32, None),
        ("I10_COMPLEX_WAVEFUNCTION_4x2x64", SupportSketchMode.I10_COMPLEX_WAVEFUNCTION_4x2x64, None),
        ("I11_COMPLEX_WAVEFUNCTION_8x2x32_ID2", SupportSketchMode.I11_COMPLEX_WAVEFUNCTION_8x2x32_ID2, None),
        ("I12_COMPLEX_WAVEFUNCTION_8x2x32_ID1", SupportSketchMode.I12_COMPLEX_WAVEFUNCTION_8x2x32_ID1, None),
        ("I13_COMPLEX_WAVEFUNCTION_16x2x16_ID4", SupportSketchMode.I13_COMPLEX_WAVEFUNCTION_16x2x16_ID4, None),
        ("I14_COMPLEX_WAVEFUNCTION_16x2x16_ID2", SupportSketchMode.I14_COMPLEX_WAVEFUNCTION_16x2x16_ID2, None),
        ("I15_SHARD1_100M_MULTILAG_SKETCH", SupportSketchMode.I15_SHARD1_100M_MULTILAG_SKETCH, None),
        ("I16_SHARD1_100M_STEP0_CALIBRATED", SupportSketchMode.I16_SHARD1_100M_STEP0_CALIBRATED, None),
        ("I17_SHARDS1_TO_9_900M_STEP0_CALIBRATED", SupportSketchMode.I17_SHARDS1_TO_9_900M_STEP0_CALIBRATED, None),
    ]

    existing_rev: dict[str, Any] = {}
    existing_lm: dict[str, Any] = {}
    if args.json_out.exists() and int(args.vocab_size) == DEFAULT_FULL_FINEWEB_VOCAB_SIZE and int(args.steps) == 300:
        prev_data = json.loads(args.json_out.read_text(encoding="utf-8"))
        existing_rev = prev_data.get("reversal_results", {})
        existing_lm = prev_data.get("causal_lm_results", {})

    cached_rev_path = _REPO_ROOT / ".cache" / "micro_reversal_init_and_readout_v50304_results.json"
    cached_init_ablation: dict[str, Any] = {}
    if cached_rev_path.exists() and int(args.vocab_size) == DEFAULT_FULL_FINEWEB_VOCAB_SIZE and int(args.steps) == 300:
        cached_init_ablation = json.loads(cached_rev_path.read_text(encoding="utf-8")).get("initialization_ablation", {})

    reversal_results: dict[str, Any] = {}
    for arm_name, mode, cache_key in modes:
        spec = SupportSketchSpec(mode=mode, vocab_size=args.vocab_size)
        if arm_name in existing_rev:
            res = existing_rev[arm_name]
        elif cache_key is not None and cache_key in cached_init_ablation:
            c_entry = cached_init_ablation[cache_key]
            zero_res = run_support_sketch_experiment(spec, dataset, task_mode="reversal", total_steps=0, eval_every=100)
            res = {
                "mode": mode.value,
                "task_mode": "reversal",
                "init_ms": c_entry["init_ms"],
                "train_wall_s": c_entry["train_wall_s"],
                "geometry": zero_res["geometry"],
                "step0": c_entry["step0"],
                "final": c_entry["final"],
                "checkpoints": c_entry["checkpoints"],
            }
        else:
            res = run_support_sketch_experiment(
                spec, dataset, task_mode="reversal", total_steps=args.steps, eval_every=args.eval_every
            )
        reversal_results[arm_name] = res
        print(_format_row(arm_name, res), flush=True)

    causal_lm_results: dict[str, Any] = {}
    for arm_name, mode, _ in modes:
        spec = SupportSketchSpec(mode=mode, vocab_size=args.vocab_size)
        if arm_name in existing_lm:
            res = existing_lm[arm_name]
        else:
            res = run_support_sketch_experiment(
                spec, dataset, task_mode="causal_lm", total_steps=args.steps, eval_every=args.eval_every
            )
        causal_lm_results[arm_name] = res
        print(_format_row(arm_name, res), flush=True)

    payload = {
        "script": "support_preserving_countsketch_bench.py",
        "vocab_size": int(args.vocab_size),
        "span_len": int(args.span_len),
        "steps": int(args.steps),
        "reversal_results": reversal_results,
        "causal_lm_results": causal_lm_results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[support_preserving_countsketch_bench] Saved results to {args.json_out}")


if __name__ == "__main__":
    main()
