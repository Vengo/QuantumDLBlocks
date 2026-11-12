#!/usr/bin/env python3
"""Complementary Schur $p_{\\text{and}}$ & Packet-Purity Layer Profiler CLI (`Wave 78` / `EXPERIMENT_PLAN_098.md`).

Promoted from `scripts/TOOLING_BACKLOG.md` (`Wave 77` candidate #2). Profiles token-by-token
coincidence probability $p_{\\text{and}}(t, \\ell, g)$ and exact $16 \\times 16$ subsystem purities
$\\operatorname{Tr}(\\rho_{\\text{AND}}^2), \\operatorname{Tr}(\\rho_{\\text{comp}}^2)$ against the
`550M` Classical Leaderboard Teacher's surprisal & entropy on `FineWeb` Shard 0, plus
leave-one-packet-out readout ablations (`drop_odd` vs. `drop_even`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qdlblocks.benchmarks.architecture_xray_diagnostics import (
    KnownArchitectureKind,
    build_empirical_zipf_decile_map,
    build_known_architecture_adapter,
    build_teacher_reference_profile,
    load_fineweb_token_slice,
)
from qdlblocks.benchmarks.wave77_baton_ladder_models import Wave77RungKind
from qdlblocks.benchmarks.wave78_routing_and_copy_diagnostics import (
    DEFAULT_INSPECTOR_EVAL_TOKENS,
    load_wave77_checkpoint_or_sprint_adapter,
    profile_complementary_schur_pand_and_purity_layers,
)

DEFAULT_SCHUR_RUNGS: str = ",".join(
    [
        Wave77RungKind.R5_CTRL_SCHUR_BASE_2L_D128.value,
        Wave77RungKind.R5A_CARBON_COMP_SCHUR_2L_D128.value,
        Wave77RungKind.R5B_CARBON_COMP_SCHUR_MTP_2L_D128.value,
    ]
)
DEFAULT_SEQ_LEN: int = 256


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments (`<= 8` args)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rungs", type=str, default=DEFAULT_SCHUR_RUNGS)
    parser.add_argument("--sprint-steps", type=int, default=120)
    parser.add_argument("--eval-tokens", type=int, default=DEFAULT_INSPECTOR_EVAL_TOKENS)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / ".cache" / "wave78_comp_schur_purity_profile.json",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the Complementary Schur $p_{\\text{and}}$ & Packet-Purity Layer Profiler (`<= 8` args)."""
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

    anchor_550m = build_known_architecture_adapter(KnownArchitectureKind.TEACHER_550M, device=device)
    ref_550m = build_teacher_reference_profile(anchor_550m, x_val, y_val)
    del anchor_550m
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rung_names = [s.strip() for s in args.rungs.split(",") if s.strip()]
    profiles: list[dict[str, Any]] = []
    for r_str in rung_names:
        rung = Wave77RungKind(r_str)
        adapter = load_wave77_checkpoint_or_sprint_adapter(
            checkpoint_path=None,
            rung=rung,
            sprint_steps=int(args.sprint_steps),
            x_val=x_val,
            y_val=y_val,
            device=device,
        )
        prof = profile_complementary_schur_pand_and_purity_layers(
            model=adapter.model,
            x_val=x_val,
            y_val=y_val,
            decile_map=decile_map,
            ref_profile=ref_550m,
            model_label=adapter.label,
        )
        prof["rung"] = r_str
        profiles.append(prof)
        del adapter
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "wave_id": "W78",
        "plan_id": "EXPERIMENT_PLAN_098",
        "sprint_steps": int(args.sprint_steps),
        "eval_tokens": int(y_val.numel()),
        "reference_anchor_550m_ce": round(ref_550m.anchor_val_ce, 5),
        "profiles": profiles,
        "wall_sec": round(time.perf_counter() - t0, 2),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n### Complementary Schur $p_{\\text{and}}$, Purity, & Packet Ablation Profile (`Wave 78`)\n", flush=True)
    print(
        "| Rung | Layer | Mean $p_{\\text{and}}$ (`P10..P90`) | Even Pur | Odd Pur | Corr($1-p_{\\text{and}}$, Ent) | Full CE | $\\Delta$CE Drop Odd (`Overall / Tail`) | $\\Delta$CE Drop Even |",
        flush=True,
    )
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |", flush=True)
    for prof in profiles:
        ab = prof["packet_ablation"]
        for ls in prof["per_layer_stats"]:
            print(
                f"| `{prof['rung']}` | `L{ls['layer']}` | "
                f"`{ls['mean_p_and']:.4f}` (`{ls['p10_p_and']:.3f}..{ls['p90_p_and']:.3f}`) | "
                f"`{ls['mean_even_purity']:.4f}` | `{ls['mean_odd_purity']:.4f}` | "
                f"`{ls['corr_disagree_vs_teacher_entropy']:+.4f}` | "
                f"`{ab['full']['overall_ce']:.4f}` | "
                f"**`{ab['delta_ce_drop_odd_overall']:+.4f} / {ab['delta_ce_drop_odd_tail_d6_d9']:+.4f}`** | "
                f"`{ab['delta_ce_drop_even_overall']:+.4f}` |",
                flush=True,
            )
    print(f"\nSaved JSON profile to {args.json_out} ({payload['wall_sec']:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
