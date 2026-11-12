#!/usr/bin/env python3
"""Copy-Gate Confidence & Rare-Entity Calibration Inspector CLI (`Wave 78` / `EXPERIMENT_PLAN_098.md`).

Promoted from `scripts/TOOLING_BACKLOG.md` (`Wave 77` candidate #1). Inspects per-token copy
confidence $c_t = \\sum_{s < t} w_{\\text{copy}}(t, s)$, scattered target logit gain, worst impostor
false-copy logit gain, and exact net cross-entropy savings (`CE_no_copy - CE_with_copy`) across
Zipf rarity tiers (`D0` function words vs. `D6–D9` rare named entities) and intra-document prior
target occurrence multiplicity $m_t \\in \\{0, 1, 2\\text{–}3, 4+\\}$ on `FineWeb` Shard 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qdlblocks.benchmarks.architecture_xray_diagnostics import (
    build_empirical_zipf_decile_map,
    load_fineweb_token_slice,
)
from qdlblocks.benchmarks.wave77_baton_ladder_models import Wave77RungKind
from qdlblocks.benchmarks.wave78_routing_and_copy_diagnostics import (
    DEFAULT_INSPECTOR_EVAL_TOKENS,
    inspect_copy_gate_confidence_and_entity_calibration,
    load_wave77_checkpoint_or_sprint_adapter,
)

DEFAULT_R4C_100M_CKPT: Path = (
    REPO_ROOT / ".cache" / "checkpoints" / "wave77" / "w77_w77_r4_mup_4l_d256_100m_confirm_wsd.pt"
)
DEFAULT_SEQ_LEN: int = 256


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments (`<= 8` args)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_R4C_100M_CKPT,
        help="Path to saved Wave 77/78 `.pt` checkpoint (falls back to `--rung` sprint if absent).",
    )
    parser.add_argument("--rung", type=str, default=Wave77RungKind.R4_MUP_4L_D256.value)
    parser.add_argument("--sprint-steps", type=int, default=120)
    parser.add_argument("--eval-tokens", type=int, default=DEFAULT_INSPECTOR_EVAL_TOKENS)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / ".cache" / "wave78_copy_gate_entity_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the Copy-Gate Confidence & Rare-Entity Calibration Inspector (`<= 8` args)."""
    args = parse_args()
    t0 = time.perf_counter()
    device = torch.device(args.device)

    n_seq = max(2, int(args.eval_tokens) // int(args.seq_len))
    raw_tok = load_fineweb_token_slice(
        "fineweb_val_000000.bin", num_tokens=n_seq * int(args.seq_len) + 1
    )
    x_val = raw_tok[:-1].view(n_seq, int(args.seq_len))
    y_val = raw_tok[1:].view(n_seq, int(args.seq_len))
    decile_map = build_empirical_zipf_decile_map(device=torch.device("cpu"))

    ckpt_used = args.checkpoint if args.checkpoint.exists() else None
    adapter = load_wave77_checkpoint_or_sprint_adapter(
        checkpoint_path=ckpt_used,
        rung=Wave77RungKind(args.rung),
        sprint_steps=args.sprint_steps,
        x_val=x_val,
        y_val=y_val,
        device=device,
    )
    source_desc = str(ckpt_used.name) if ckpt_used is not None else f"{args.rung} ({args.sprint_steps}s)"
    report = inspect_copy_gate_confidence_and_entity_calibration(
        model=adapter.model,
        x_val=x_val,
        y_val=y_val,
        decile_map=decile_map,
        model_label=f"{adapter.label} [{source_desc}]",
    )
    report["wall_sec"] = round(time.perf_counter() - t0, 2)
    report["checkpoint_used"] = str(ckpt_used) if ckpt_used is not None else None

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n### Copy-Gate Confidence & Rare-Entity Calibration Audit (`Wave 78`)\n", flush=True)
    print(f"- **Model**: `{report['model_label']}` (`{report['eval_tokens']:,}` Shard-0 tokens)", flush=True)
    print(
        "| Bucket | Tokens (`%`) | Copy Conf $c_t$ | Target Gain ($\\Delta z_y$) | Max Impostor ($\\Delta z_{\\ne y}$) | CE w/ Copy | CE No Copy | Net Nat Savings | Top-1 (`%`) |",
        flush=True,
    )
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |", flush=True)

    rows = [
        ("Overall (`All`)", report["overall"]),
        ("Copy-Eligible (`m >= 1`)", report["copyable_all_tiers"]),
        ("`D0` Function Words (`m >= 1`)", report["by_zipf_and_entity_tier"]["D0_function_words_copyable"]),
        ("`D2–D5` Body (`m >= 1`)", report["by_zipf_and_entity_tier"]["D2_D5_body_copyable"]),
        ("`D6–D9` Rare Entities (`m >= 1`)", report["by_zipf_and_entity_tier"]["D6_D9_rare_entities_copyable"]),
        ("Multiplicity `m = 0` (`Recall`)", report["by_prior_doc_multiplicity"]["m_0_out_of_doc_recall"]),
        ("Multiplicity `m = 1` (`1 Mention`)", report["by_prior_doc_multiplicity"]["m_1_single_prior_mention"]),
        ("Multiplicity `m = 2..3`", report["by_prior_doc_multiplicity"]["m_2_3_few_mentions"]),
        ("Multiplicity `m >= 4` (`Repeats`)", report["by_prior_doc_multiplicity"]["m_4plus_frequent_repeat"]),
    ]
    for name, b in rows:
        print(
            f"| {name} | `{int(b['count']):,}` (`{b['frac']*100:4.1f}%`) | "
            f"`{b['mean_copy_conf']:.4f}` | `{b['mean_target_logit_gain']:+.3f}` | "
            f"`{b['mean_max_impostor_logit_gain']:+.3f}` | `{b['ce_with_copy']:.4f}` | "
            f"`{b['ce_without_copy']:.4f}` | **`{b['net_copy_nat_savings']:+.4f}`** | "
            f"`{b['top1_with_copy_pct']:5.1f}%` |",
            flush=True,
        )
    print(f"\nSaved JSON report to {args.json_out} ({report['wall_sec']:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
