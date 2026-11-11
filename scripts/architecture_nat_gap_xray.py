#!/usr/bin/env python3
"""Standalone Cross-Architecture Stratified Nat-Gap & Calibration X-Ray CLI (`Wave 76` / `EXPERIMENT_PLAN_096.md`).

Evaluates token-by-token NLL, entropy, Top-1/Top-5, 10-bin ECE, and differential $\\Delta\\text{CE}$
against a pretrained `CompiledLeaderboardClassicalGPT` anchor (`550M @ 3.35 CE` or `300M @ 3.58 CE`)
stratified across:
  1. In-Document Copy (`y_t in x_{<=t}` inside `<BOS>` document) vs. Out-of-Document Parametric Recall,
  2. 10 Empirical Zipf Frequency Deciles (`D0..D9`) + Head / Body / Tail,
  3. Post-`<BOS>` Intra-Document Horizon (`[0..3], [4..31], [32..127], [128+]`), and
  4. Teacher Entropy Regimes (`Sharp < 1.5`, `Medium 1.5..4.0`, `Ambiguous >= 4.0` nats).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qdlblocks.benchmarks.architecture_mechanistic_probes import run_controlled_short_fineweb_sprint
from qdlblocks.benchmarks.architecture_xray_diagnostics import (
    KnownArchitectureKind,
    ShortTrainConfig,
    build_empirical_zipf_decile_map,
    build_known_architecture_adapter,
    build_teacher_reference_profile,
    compute_stratified_nat_gap_xray,
    load_fineweb_token_slice,
)

DEFAULT_EVAL_TOKENS: int = 16384
DEFAULT_SEQ_LEN: int = 256
DEFAULT_BATCH_SIZE: int = 32


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments (`<= 8` args)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        type=str,
        default="classical_teacher_50m,classical_teacher_300m,w74_baton_2l_d64,w45_classical_2l_d512",
        help="Comma-separated KnownArchitectureKind values to X-Ray.",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="classical_teacher_550m",
        help="Reference anchor KnownArchitectureKind (or 'none').",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=100,
        help="Short FineWeb Shard-1 steps to train non-pretrained models before X-Ray.",
    )
    parser.add_argument("--eval-tokens", type=int, default=DEFAULT_EVAL_TOKENS)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / ".cache" / "architecture_nat_gap_xray.json",
    )
    return parser.parse_args()


def print_xray_markdown_table(rows: list[dict[str, Any]], anchor_name: str) -> None:
    """Render concise Markdown summary of Stratified Nat-Gap Attribution (`<= 8` args)."""
    print(f"\n### Stratified Nat-Gap X-Ray (Reference Anchor: `{anchor_name}`)")
    print(
        "| Model | Val CE | ΔCE vs Ref | Copy CE (Δ) | Recall CE (Δ) | Pos 128+ CE (Δ) | Sharp H<1.5 CE (Δ) | Logit Span | ECE |"
    )
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in rows:
        cp = r["copy_vs_recall"]["in_doc_copy"]
        rc = r["copy_vs_recall"]["out_of_doc_recall"]
        lg = r["doc_horizons"]["pos_128_plus_long"]
        sh = r.get("surprisal_regimes", {}).get(
            "sharp_deterministic_lt_1p5", {"ce": 0.0, "delta_ce_vs_ref": 0.0}
        )
        print(
            f"| `{r['key']}` | `{r['val_ce']:.4f}` | `{r['delta_ce_vs_ref']:+.4f}` | "
            f"`{cp['ce']:.3f}` (`{cp['delta_ce_vs_ref']:+.3f}`) | "
            f"`{rc['ce']:.3f}` (`{rc['delta_ce_vs_ref']:+.3f}`) | "
            f"`{lg['ce']:.3f}` (`{lg['delta_ce_vs_ref']:+.3f}`) | "
            f"`{sh['ce']:.3f}` (`{sh['delta_ce_vs_ref']:+.3f}`) | "
            f"`{r['mean_logit_span_max_minus_median']:.2f}` | `{r['ece_10bin']:.4f}` |"
        )


def main() -> None:
    """Execute standalone Stratified Nat-Gap X-Ray CLI (`<= 8` args)."""
    args = parse_args()
    device = torch.device(args.device)
    n_seq = max(1, args.eval_tokens // args.seq_len)
    val_slice = load_fineweb_token_slice(
        "fineweb_val_000000.bin", num_tokens=n_seq * args.seq_len + 1
    )
    x_eval = val_slice[:-1].view(n_seq, args.seq_len)
    y_eval = val_slice[1:].view(n_seq, args.seq_len)
    decile_map = build_empirical_zipf_decile_map(device=torch.device("cpu"))

    ref_profile = None
    anchor_label = "none"
    if args.anchor.lower() != "none":
        anchor_kind = KnownArchitectureKind(args.anchor.strip())
        anchor_adapter = build_known_architecture_adapter(anchor_kind, device=device)
        ref_profile = build_teacher_reference_profile(anchor_adapter, x_eval, y_eval)
        anchor_label = f"{anchor_kind.value} (Val CE={ref_profile.anchor_val_ce:.4f})"
        del anchor_adapter
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pretrained_set = {
        KnownArchitectureKind.TEACHER_50M,
        KnownArchitectureKind.TEACHER_300M,
        KnownArchitectureKind.TEACHER_550M,
    }
    xray_rows: list[dict[str, Any]] = []
    for raw_name in [m.strip() for m in args.models.split(",") if m.strip()]:
        kind = KnownArchitectureKind(raw_name)
        if kind in pretrained_set or args.train_steps <= 0:
            adapter = build_known_architecture_adapter(kind, device=device)
            sprint_summary = None
        else:
            t_cfg = ShortTrainConfig(
                steps=args.train_steps,
                batch_size=DEFAULT_BATCH_SIZE,
                seq_len=args.seq_len,
                eval_interval=max(25, args.train_steps // 2),
                eval_tokens=args.eval_tokens,
                wave_id="W76XRAY",
            )
            adapter, sprint_summary = run_controlled_short_fineweb_sprint(
                kind, t_cfg, x_eval, y_eval, device=device
            )
        row = compute_stratified_nat_gap_xray(
            adapter, x_eval, y_eval, decile_map=decile_map, ref_profile=ref_profile
        )
        if sprint_summary is not None:
            row["sprint_summary"] = sprint_summary
        xray_rows.append(row)
        del adapter
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_xray_markdown_table(xray_rows, anchor_label)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "anchor": anchor_label,
        "eval_tokens": int(x_eval.numel()),
        "models": xray_rows,
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved Stratified Nat-Gap X-Ray JSON -> {args.json_out}")


if __name__ == "__main__":
    main()
