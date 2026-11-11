#!/usr/bin/env python3
"""Master `Wave 77` 5-Rung Evidence Ladder + Carbon Routing Ablation + `100M`-Token Confirmation CLI (`EXPERIMENT_PLAN_097.md`).

Executes:
  Phase 1: Loads `550M`-token `CompiledLeaderboardClassicalGPT` teacher reference (`3.4882` CE)
           on `16,384` Shard-0 tokens (plus cached `Wave 76` Rung-0 baseline numbers).
  Phase 2: Controlled `400`-step (`3.28M`-token) FineWeb Shard-1 sprints across:
           - Rung 1 (`w77_r1_calib_copy_sink_2l_d64`)
           - Rung 2 (`w77_r2_pruned_untied_2l_d64`)
           - Rung 3 (`w77_r3_muon_mixer_2l_d64`)
           - Rung 4 (`w77_r4_mup_2l_d128`, `w77_r4_mup_2l_d256`, `w77_r4_mup_4l_d256`)
           - Rung 5 Carbon 3-Way Ablation (`w77_r5_ctrl_post_schur_2l_d128`,
             `w77_r5a_carbon_comp_schur_2l_d128`, `w77_r5b_carbon_comp_schur_mtp_2l_d128`)
           recording Hot/EMA CE, power-law asymptote $L_\\infty$, Stratified Nat-Gap X-Ray, and
           per-layer `mean_p_and` & `packet_purity`.
  Phase 3: `50,257`-vocab Fuzzy Induction ($T \\in \\{64, 128, 256\\}$ with `<BOS>` trap) and
           `MQAR` ($K \\in \\{4, 8\\}$) probes across key rungs.
  Phase 4: Multi-shard `100,000,000`-token (`100M`-token) Warmup-Stable-Decay (`WSD`) confirmation
           spin on the winning `Wave 77` architecture with Official `10,485,760`-token Shard-0 validation!
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
    ShortTrainConfig,
    SyntheticProbeConfig,
    build_empirical_zipf_decile_map,
    build_known_architecture_adapter,
    build_teacher_reference_profile,
    compute_stratified_nat_gap_xray,
    load_fineweb_token_slice,
)
from qdlblocks.benchmarks.wave77_baton_ladder_models import Wave77RungKind
from qdlblocks.benchmarks.wave77_evidence_ladder_runner import (
    DEFAULT_CONFIRM_BATCH_SIZE,
    DEFAULT_CONFIRM_EVAL_INTERVAL,
    DEFAULT_CONFIRM_SEQ_LEN,
    DEFAULT_CONFIRM_TARGET_TOKENS,
    MultiShardConfirmConfig,
    run_wave77_controlled_sprint,
    run_wave77_induction_and_mqar_probes,
    run_wave77_multishard_confirmation_spin,
)

DEFAULT_SPRINT_STEPS: int = 400
DEFAULT_SPRINT_BATCH_SIZE: int = 32
DEFAULT_SPRINT_SEQ_LEN: int = 256
DEFAULT_EVAL_TOKENS: int = 16384


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments (`<= 8` args)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sprint-steps", type=int, default=DEFAULT_SPRINT_STEPS)
    parser.add_argument("--confirm-tokens", type=int, default=DEFAULT_CONFIRM_TARGET_TOKENS)
    parser.add_argument("--confirm-rung", type=str, default="auto")
    parser.add_argument("--skip-ladder", action="store_true")
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--skip-confirm", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / ".cache" / "wave77_evidence_ladder_summary.json",
    )
    return parser.parse_args()


def _save_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist intermediate and final Wave 77 JSON summaries atomically (`<= 8` args)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    """Run Phases 1–4 of the `Wave 77` Evidence Ladder & Confirmation Suite (`<= 8` args)."""
    args = parse_args()
    device = torch.device(args.device)
    suite_t0 = time.perf_counter()

    n_val_seq = max(2, DEFAULT_EVAL_TOKENS // DEFAULT_SPRINT_SEQ_LEN)
    val_tokens = load_fineweb_token_slice(
        "fineweb_val_000000.bin", num_tokens=n_val_seq * DEFAULT_SPRINT_SEQ_LEN + 1
    )
    x_val = val_tokens[:-1].view(n_val_seq, DEFAULT_SPRINT_SEQ_LEN)
    y_val = val_tokens[1:].view(n_val_seq, DEFAULT_SPRINT_SEQ_LEN)
    decile_map = build_empirical_zipf_decile_map(device=torch.device("cpu"))

    print("========================================================================================", flush=True)
    print("WAVE 77: 5-RUNG QUANTUM BATON EVIDENCE LADDER, CARBON ABLATION, & 100M CONFIRMATION", flush=True)
    print(f"Device: {device} | Sprint: {args.sprint_steps} steps | Confirm Budget: {args.confirm_tokens:,} tok", flush=True)
    print("========================================================================================", flush=True)

    print("\n[Phase 1/4] Building 550M Classical Leaderboard Teacher Reference Profile...", flush=True)
    anchor_550m = build_known_architecture_adapter(KnownArchitectureKind.TEACHER_550M, device=device)
    ref_550m = build_teacher_reference_profile(anchor_550m, x_val, y_val)
    del anchor_550m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  Reference Anchor 550M CE on {x_val.numel():,} Shard-0 tokens = {ref_550m.anchor_val_ce:.4f}", flush=True)

    cached_prev: dict[str, Any] = {}
    if args.json_out.exists() and (args.skip_ladder or args.skip_probes):
        cached_prev = json.loads(args.json_out.read_text(encoding="utf-8"))

    payload: dict[str, Any] = {
        "wave_id": "W77",
        "plan_id": "EXPERIMENT_PLAN_097",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "reference_anchor_550m_ce": round(ref_550m.anchor_val_ce, 5),
        "ladder_sprints": list(cached_prev.get("ladder_sprints", [])),
        "synthetic_induction_sweep": list(cached_prev.get("synthetic_induction_sweep", [])),
        "synthetic_mqar_probe": list(cached_prev.get("synthetic_mqar_probe", [])),
        "multishard_confirmation_spin": cached_prev.get("multishard_confirmation_spin"),
    }

    ladder_rungs = [
        Wave77RungKind.R1_CALIB_COPY_SINK_2L_D64,
        Wave77RungKind.R2_PRUNED_UNTIED_2L_D64,
        Wave77RungKind.R3_MUON_MIXER_2L_D64,
        Wave77RungKind.R4_MUP_2L_D128,
        Wave77RungKind.R4_MUP_2L_D256,
        Wave77RungKind.R4_MUP_4L_D256,
        Wave77RungKind.R5_CTRL_SCHUR_BASE_2L_D128,
        Wave77RungKind.R5A_CARBON_COMP_SCHUR_2L_D128,
        Wave77RungKind.R5B_CARBON_COMP_SCHUR_MTP_2L_D128,
    ]
    train_cfg = ShortTrainConfig(
        steps=args.sprint_steps,
        batch_size=DEFAULT_SPRINT_BATCH_SIZE,
        seq_len=DEFAULT_SPRINT_SEQ_LEN,
        eval_interval=max(50, args.sprint_steps // 4),
        eval_tokens=DEFAULT_EVAL_TOKENS,
        wave_id="W77",
        plan_id="EXPERIMENT_PLAN_097",
    )

    if not args.skip_ladder:
        payload["ladder_sprints"] = []
        print(
            f"\n[Phase 2/4] Running {args.sprint_steps}-Step ({args.sprint_steps * 8192 / 1e6:.2f}M Token) "
            f"Controlled Evidence Ladder Across {len(ladder_rungs)} Rungs...",
            flush=True,
        )
        for rung in ladder_rungs:
            adapter, sprint_summary = run_wave77_controlled_sprint(
                rung, train_cfg, x_val, y_val, device=device
            )
            xray = compute_stratified_nat_gap_xray(
                adapter, x_val, y_val, decile_map=decile_map, ref_profile=ref_550m
            )
            sprint_summary["stratified_xray"] = xray
            payload["ladder_sprints"].append(sprint_summary)
            _save_atomic_json(args.json_out, payload)

            asym = sprint_summary["asymptote_fit"]
            rt = sprint_summary["routing_telemetry"]
            extra_rt = ""
            if rt.get("has_schur_gate"):
                extra_rt = (
                    f" | p_and={rt['mean_p_and']:.4f} "
                    f"Pur(E/O)=({rt['mean_even_purity']:.3f}/{rt['mean_odd_purity']:.3f})"
                )
            print(
                f"  [Ladder] {sprint_summary['arch']:36s} | {sprint_summary['tok_per_s']:8,.1f} tok/s | "
                f"Hot={sprint_summary['final_hot_val_ce']:.4f} | SlowEMA={sprint_summary['final_slow_ema_val_ce']:.4f} | "
                f"L_inf={asym['l_inf_asymptote']:.3f} | "
                f"Copy={xray['copy_vs_recall']['in_doc_copy']['ce']:.4f} | "
                f"Recall={xray['copy_vs_recall']['out_of_doc_recall']['ce']:.4f}{extra_rt}",
                flush=True,
            )
            del adapter
            if device.type == "cuda":
                torch.cuda.empty_cache()
    else:
        print(
            f"\n[Phase 2/4] Reusing {len(payload['ladder_sprints'])} cached ladder sprint records from {args.json_out}...",
            flush=True,
        )

    if not args.skip_probes:
        probe_rungs = [
            Wave77RungKind.R1_CALIB_COPY_SINK_2L_D64,
            Wave77RungKind.R3_MUON_MIXER_2L_D64,
            Wave77RungKind.R4_MUP_2L_D256,
            Wave77RungKind.R5B_CARBON_COMP_SCHUR_MTP_2L_D128,
        ]
        probe_cfg = SyntheticProbeConfig(
            induction_steps=180,
            mqar_steps=220,
            overfit_steps=220,
            batch_size=32,
            eval_batches=8,
        )
        print("\n[Phase 3/4] Running 50,257-Vocab Fuzzy Induction & MQAR Probes...", flush=True)
        ind_res, mqar_res = run_wave77_induction_and_mqar_probes(
            probe_rungs, probe_cfg, device=device
        )
        payload["synthetic_induction_sweep"] = ind_res
        payload["synthetic_mqar_probe"] = mqar_res
        _save_atomic_json(args.json_out, payload)

        for r in ind_res:
            print(
                f"  [Induction] {r['arch']:36s} | T={r['seq_len']:3d} | "
                f"CE={r['induction_query_ce']:.4f} | Top-1={r['induction_query_top1_acc']*100:5.1f}% | "
                f"Trap={r['pre_bos_false_trap_rate']*100:5.1f}%",
                flush=True,
            )
        for r in mqar_res:
            print(
                f"  [MQAR]      {r['arch']:36s} | K={r['num_pairs']:2d}  | "
                f"MQAR CE={r['mqar_ce']:.4f} | Top-1={r['mqar_top1_acc']*100:5.1f}%",
                flush=True,
            )

    if not args.skip_confirm and args.confirm_tokens > 0:
        if args.confirm_rung == "auto":
            best_sprint = min(
                payload["ladder_sprints"], key=lambda s: float(s["best_any_val_ce"])
            )
            chosen_rung = Wave77RungKind(best_sprint["arch"])
        else:
            chosen_rung = Wave77RungKind(args.confirm_rung)

        print(
            f"\n[Phase 4/4] Launching {args.confirm_tokens / 1e6:.1f}M-Token Multi-Shard WSD Confirmation "
            f"Spin on Winner `{chosen_rung.value}` + Official 10.49M Shard-0 Eval...",
            flush=True,
        )
        confirm_cfg = MultiShardConfirmConfig(
            target_tokens=args.confirm_tokens,
            batch_size=DEFAULT_CONFIRM_BATCH_SIZE,
            seq_len=DEFAULT_CONFIRM_SEQ_LEN,
            eval_interval=DEFAULT_CONFIRM_EVAL_INTERVAL,
        )
        confirm_summary = run_wave77_multishard_confirmation_spin(
            rung=chosen_rung,
            spin_cfg=confirm_cfg,
            x_xray=x_val,
            y_xray=y_val,
            decile_map=decile_map,
            ref_550m=ref_550m,
            device=device,
        )
        payload["multishard_confirmation_spin"] = confirm_summary
        off = confirm_summary.get("official_shard0_10m_eval", {})
        print(
            f"\n  [Confirmation Complete] {confirm_summary['arch']} | "
            f"{confirm_summary['tokens_M']:.1f}M tokens @ {confirm_summary['tok_per_s']:,.1f} tok/s | "
            f"16k Hot/SlowEMA CE = {confirm_summary['final_hot_16k_ce']:.4f} / {confirm_summary['final_slow_ema_16k_ce']:.4f} | "
            f"Official 10.49M Shard-0 Val CE = {off.get('official_shard0_val_ce', -1.0):.4f} "
            f"(Top-1 = {off.get('official_shard0_top1_pct', 0.0):.2f}%)",
            flush=True,
        )

    payload["total_wall_sec"] = round(time.perf_counter() - suite_t0, 2)
    _save_atomic_json(args.json_out, payload)
    print(f"\nSaved complete Wave 77 report to {args.json_out} ({payload['total_wall_sec']:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
