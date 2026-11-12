#!/usr/bin/env python3
"""Master `Wave 78` `R4c + R5B` Quantum Baton Unification, Probes, & Diagnostics CLI (`EXPERIMENT_PLAN_098.md`).

Executes:
  Phase 1: Loads `550M`-token `CompiledLeaderboardClassicalGPT` teacher reference (`3.4882` CE)
           on `16,384` Shard-0 tokens.
  Phase 2: Controlled `400`-step (`3.28M`-token) FineWeb Shard-1 sprints across `U0..U4`:
           - `w78_u0_r4c_base_4l_d256`: `R4c` Baseline (`4L, d=256`, semantic-only copy)
           - `w78_u1_hybrid_wh_copy_4l_d256`: `+PurityConditionedHybridWalshCopyGate` (`4L, d=256`)
           - `w78_u2_r4c_comp_schur_4l_d256`: `+ZeroSyncComplementarySchurGate` (`4L, d=256`)
           - `w78_u3_unified_r4c_r5b_4l_d256`: Full `R4c + R5B` Unification (`+HorizonPooledCausalMTPMPSSweep`, `4L`)
           - `w78_u4_unified_r4c_r5b_6l_d256`: Deep `6L` (`3+3` U-Net) `R4c + R5B` Unification (`6L, d=256`)
           plus in-memory post-sprint Copy-Gate Entity Calibration & Complementary Schur Layer Profiling!
  Phase 3: `50,257`-vocab Fuzzy Induction ($T \\in \\{64, 128, 256\\}$ with `<BOS>` trap) and
           `MQAR` ($K \\in \\{4, 8\\}$) probes comparing `U0` vs. `U1` vs. `U3`.
  Phase 4: Optional `--confirm-tokens` multi-shard `WSD` confirmation spin + Official `10.49M` Shard-0 eval.
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
from qdlblocks.benchmarks.wave77_evidence_ladder_runner import (
    DEFAULT_CONFIRM_BATCH_SIZE,
    DEFAULT_CONFIRM_EVAL_INTERVAL,
    DEFAULT_CONFIRM_SEQ_LEN,
    MultiShardConfirmConfig,
)
from qdlblocks.benchmarks.wave78_routing_and_copy_diagnostics import (
    inspect_copy_gate_confidence_and_entity_calibration,
    profile_complementary_schur_pand_and_purity_layers,
)
from qdlblocks.benchmarks.wave78_unification_runner import (
    run_wave78_controlled_sprint,
    run_wave78_induction_and_mqar_probes,
    run_wave78_multishard_confirmation_spin,
)
from qdlblocks.benchmarks.wave78_unified_mup_carbon_baton import Wave78VariantKind

DEFAULT_SPRINT_STEPS: int = 400
DEFAULT_SPRINT_BATCH_SIZE: int = 32
DEFAULT_SPRINT_SEQ_LEN: int = 256
DEFAULT_EVAL_TOKENS: int = 16384


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments (`<= 8` args)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sprint-steps", type=int, default=DEFAULT_SPRINT_STEPS)
    parser.add_argument("--confirm-tokens", type=int, default=0)
    parser.add_argument("--confirm-variant", type=str, default="auto")
    parser.add_argument("--skip-sprints", action="store_true")
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / ".cache" / "wave78_unified_r4c_r5b_summary.json",
    )
    return parser.parse_args()


def _save_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist intermediate and final Wave 78 JSON summaries atomically (`<= 8` args)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    """Run Phases 1–4 of the `Wave 78` `R4c + R5B` Unification Suite (`<= 8` args)."""
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
    print("WAVE 78: R4c + R5B QUANTUM BATON UNIFICATION, HYBRID WALSH COPY, & DIAGNOSTIC SUITE", flush=True)
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
    if args.json_out.exists() and (args.skip_sprints or args.skip_probes):
        cached_prev = json.loads(args.json_out.read_text(encoding="utf-8"))

    payload: dict[str, Any] = {
        "wave_id": "W78",
        "plan_id": "EXPERIMENT_PLAN_098",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "reference_anchor_550m_ce": round(ref_550m.anchor_val_ce, 5),
        "unification_sprints": list(cached_prev.get("unification_sprints", [])),
        "post_sprint_copy_audits": list(cached_prev.get("post_sprint_copy_audits", [])),
        "post_sprint_schur_profiles": list(cached_prev.get("post_sprint_schur_profiles", [])),
        "synthetic_induction_sweep": list(cached_prev.get("synthetic_induction_sweep", [])),
        "synthetic_mqar_probe": list(cached_prev.get("synthetic_mqar_probe", [])),
        "multishard_confirmation_spin": cached_prev.get("multishard_confirmation_spin"),
    }

    variants = [
        Wave78VariantKind.U0_R4C_BASE_4L_D256,
        Wave78VariantKind.U1_HYBRID_WH_COPY_4L_D256,
        Wave78VariantKind.U2_R4C_COMP_SCHUR_4L_D256,
        Wave78VariantKind.U3_UNIFIED_R4C_R5B_4L_D256,
        Wave78VariantKind.U4_UNIFIED_R4C_R5B_6L_D256,
    ]
    train_cfg = ShortTrainConfig(
        steps=args.sprint_steps,
        batch_size=DEFAULT_SPRINT_BATCH_SIZE,
        seq_len=DEFAULT_SPRINT_SEQ_LEN,
        eval_interval=max(50, args.sprint_steps // 4),
        eval_tokens=DEFAULT_EVAL_TOKENS,
        wave_id="W78",
        plan_id="EXPERIMENT_PLAN_098",
    )

    if not args.skip_sprints:
        payload["unification_sprints"] = []
        payload["post_sprint_copy_audits"] = []
        payload["post_sprint_schur_profiles"] = []
        print(
            f"\n[Phase 2/4] Running {args.sprint_steps}-Step ({args.sprint_steps * 8192 / 1e6:.2f}M Token) "
            f"Unification Matrix Across {len(variants)} Variants (`U0..U4`)...",
            flush=True,
        )
        for variant in variants:
            adapter, sprint_summary = run_wave78_controlled_sprint(
                variant, train_cfg, x_val, y_val, device=device
            )
            xray = compute_stratified_nat_gap_xray(
                adapter, x_val, y_val, decile_map=decile_map, ref_profile=ref_550m
            )
            sprint_summary["stratified_xray"] = xray
            payload["unification_sprints"].append(sprint_summary)

            if variant in (
                Wave78VariantKind.U0_R4C_BASE_4L_D256,
                Wave78VariantKind.U1_HYBRID_WH_COPY_4L_D256,
                Wave78VariantKind.U3_UNIFIED_R4C_R5B_4L_D256,
            ):
                c_audit = inspect_copy_gate_confidence_and_entity_calibration(
                    adapter.model, x_val, y_val, decile_map, model_label=adapter.label
                )
                c_audit["arch"] = variant.value
                payload["post_sprint_copy_audits"].append(c_audit)

            if variant in (
                Wave78VariantKind.U2_R4C_COMP_SCHUR_4L_D256,
                Wave78VariantKind.U3_UNIFIED_R4C_R5B_4L_D256,
                Wave78VariantKind.U4_UNIFIED_R4C_R5B_6L_D256,
            ):
                s_prof = profile_complementary_schur_pand_and_purity_layers(
                    adapter.model,
                    x_val,
                    y_val,
                    decile_map=decile_map,
                    ref_profile=ref_550m,
                    model_label=adapter.label,
                )
                s_prof["arch"] = variant.value
                payload["post_sprint_schur_profiles"].append(s_prof)

            _save_atomic_json(args.json_out, payload)

            asym = sprint_summary["asymptote_fit"]
            rt = sprint_summary["routing_telemetry"]
            extra_rt = ""
            if rt.get("has_schur_gate"):
                extra_rt = (
                    f" | p_and={rt['mean_p_and']:.4f} "
                    f"Pur(E/O)=({rt['mean_even_purity']:.3f}/{rt['mean_odd_purity']:.3f})"
                )
            if rt.get("mtp_horizon_weights"):
                extra_rt += f" | MTP_w={rt['mtp_horizon_weights']}"
            print(
                f"  [Sprint] {sprint_summary['arch']:32s} | {sprint_summary['tok_per_s']:8,.1f} tok/s | "
                f"Hot={sprint_summary['final_hot_val_ce']:.4f} | SlowEMA={sprint_summary['final_slow_ema_val_ce']:.4f} | "
                f"L_inf={asym['l_inf_asymptote']:.3f} | "
                f"Copy={xray['copy_vs_recall']['in_doc_copy']['ce']:.4f} | "
                f"Tail={xray['zipf_tiers']['tail_d6_d9']['ce']:.4f}{extra_rt}",
                flush=True,
            )
            del adapter
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not args.skip_probes:
        probe_variants = [
            Wave78VariantKind.U0_R4C_BASE_4L_D256,
            Wave78VariantKind.U1_HYBRID_WH_COPY_4L_D256,
            Wave78VariantKind.U3_UNIFIED_R4C_R5B_4L_D256,
        ]
        probe_cfg = SyntheticProbeConfig(
            induction_steps=180,
            mqar_steps=220,
            overfit_steps=220,
            batch_size=32,
            eval_batches=8,
        )
        print("\n[Phase 3/4] Running 50,257-Vocab Fuzzy Induction & MQAR Probes (`U0` vs `U1` vs `U3`)...", flush=True)
        ind_res, mqar_res = run_wave78_induction_and_mqar_probes(
            probe_variants, probe_cfg, device=device
        )
        payload["synthetic_induction_sweep"] = ind_res
        payload["synthetic_mqar_probe"] = mqar_res
        _save_atomic_json(args.json_out, payload)

        for r in ind_res:
            print(
                f"  [Induction] {r['arch']:32s} | T={r['seq_len']:3d} | "
                f"CE={r['induction_query_ce']:.4f} | Top-1={r['induction_query_top1_acc']*100:5.1f}% | "
                f"Trap={r['pre_bos_false_trap_rate']*100:5.1f}%",
                flush=True,
            )
        for r in mqar_res:
            print(
                f"  [MQAR]      {r['arch']:32s} | K={r['num_pairs']:2d}  | "
                f"MQAR CE={r['mqar_ce']:.4f} | Top-1={r['mqar_top1_acc']*100:5.1f}%",
                flush=True,
            )

    if args.confirm_tokens > 0:
        if args.confirm_variant == "auto":
            best_s = min(
                payload["unification_sprints"], key=lambda s: float(s["best_any_val_ce"])
            )
            chosen_var = Wave78VariantKind(best_s["arch"])
        else:
            chosen_var = Wave78VariantKind(args.confirm_variant)

        print(
            f"\n[Phase 4/4] Launching {args.confirm_tokens / 1e6:.1f}M-Token Multi-Shard WSD Confirmation "
            f"Spin on `{chosen_var.value}` + Official 10.49M Shard-0 Eval...",
            flush=True,
        )
        confirm_cfg = MultiShardConfirmConfig(
            target_tokens=args.confirm_tokens,
            batch_size=DEFAULT_CONFIRM_BATCH_SIZE,
            seq_len=DEFAULT_CONFIRM_SEQ_LEN,
            eval_interval=DEFAULT_CONFIRM_EVAL_INTERVAL,
            wave_id="W78",
            plan_id="EXPERIMENT_PLAN_098",
        )
        conf_res = run_wave78_multishard_confirmation_spin(
            chosen_var,
            confirm_cfg,
            x_xray=x_val,
            y_xray=y_val,
            decile_map=decile_map,
            ref_550m=ref_550m,
            device=device,
        )
        payload["multishard_confirmation_spin"] = conf_res

    payload["total_wall_sec"] = round(time.perf_counter() - suite_t0, 2)
    _save_atomic_json(args.json_out, payload)
    print(
        f"\n[Wave 78 Complete] Saved full unification & diagnostic summary to {args.json_out} "
        f"({payload['total_wall_sec']:.1f}s).",
        flush=True,
    )


if __name__ == "__main__":
    main()
