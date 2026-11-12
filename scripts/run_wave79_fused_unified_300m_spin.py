#!/usr/bin/env python3
"""Master `Wave 79` Verification, Induction/MQAR Probe, & `300M`-Token `WSD` Speedrun CLI (`EXPERIMENT_PLAN_099.md`).

Executes:
1. Controlled `400`-step (`3.28M`-token) FineWeb Shard-1 verification sprints across `V0..V3`
   with `StandardizedExperimentLogger`, `DualHorizonShadowEMATracker`, Stratified Nat-Gap X-Ray,
   and post-sprint Copy/Schur & MTP Horizon diagnostics (immediately freeing GPU memory per variant).
2. `50,257`-vocab Unseen-ID Fuzzy Induction ($T \\in \\{64, 128, 256\\}$) and `MQAR` ($K \\in \\{4, 8\\}$) probes.
3. Multi-shard `300,000,000`-token (`300M` token) `WSD` speedrun across `fineweb_train_000001..000003.bin`
   with an automatic `100M`-token iso-horizon branched WSD quench + Official `10,485,760`-token Shard-0
   validation at both `100M` and `300M` tokens.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qdlblocks.benchmarks.architecture_xray_diagnostics import (  # noqa: E402
    KnownArchitectureKind,
    ShortTrainConfig,
    SyntheticProbeConfig,
    build_empirical_zipf_decile_map,
    build_known_architecture_adapter,
    build_teacher_reference_profile,
    compute_stratified_nat_gap_xray,
    load_fineweb_token_slice,
)
from qdlblocks.benchmarks.wave77_evidence_ladder_runner import MultiShardConfirmConfig  # noqa: E402
from qdlblocks.benchmarks.wave79_fused_schur_and_induction_baton import Wave79VariantKind  # noqa: E402
from qdlblocks.benchmarks.wave79_speedrun_and_ablation_runner import (  # noqa: E402
    audit_wave79_copy_and_schur_diagnostics,
    inspect_mtp_horizon_weights_and_rollout_cosines,
    run_wave79_300m_multishard_wsd_speedrun,
    run_wave79_controlled_sprint,
    run_wave79_induction_and_mqar_probes,
)

DEFAULT_SUMMARY_OUT: Path = REPO_ROOT / ".cache" / "wave79_fused_unified_300m_summary.json"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments (`<= 8` args)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sprint-steps", type=int, default=400)
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(v.value for v in Wave79VariantKind),
    )
    parser.add_argument(
        "--confirm-tokens",
        type=int,
        default=300_000_000,
        help="Target tokens for Phase 3 multi-shard WSD speedrun (0 to skip).",
    )
    parser.add_argument(
        "--confirm-variant",
        type=str,
        default="auto",
        help="Variant for Phase 3 ('auto' selects V3 6L if >=28k tok/s & lowest CE, else V2 4L).",
    )
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    return parser.parse_args()


def _select_confirm_variant(
    arg_choice: str, sprint_rows: list[dict[str, Any]]
) -> Wave79VariantKind:
    """Pick the Phase 3 `300M` speedrun variant from `arg_choice` or empirical sprint telemetry."""
    if arg_choice != "auto":
        return Wave79VariantKind(arg_choice)
    by_arch = {r["arch"]: r for r in sprint_rows}
    v3 = by_arch.get(Wave79VariantKind.V3_FUSED_UNIFIED_6L_D256.value)
    v2 = by_arch.get(Wave79VariantKind.V2_FUSED_UNIFIED_4L_D256.value)
    if v3 is not None and float(v3["tok_per_s"]) >= 28000.0:
        if v2 is None or float(v3["final_hot_val_ce"]) <= float(v2["final_hot_val_ce"]) + 0.01:
            return Wave79VariantKind.V3_FUSED_UNIFIED_6L_D256
    return Wave79VariantKind.V2_FUSED_UNIFIED_4L_D256


def _save_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist intermediate and final Wave 79 JSON summaries atomically (`<= 8` args)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    """Execute Phases 1–3 of the `Wave 79` verification & `300M`-token `WSD` speedrun."""
    args = parse_args()
    t0_all = time.perf_counter()
    device = torch.device(args.device)
    variants = [Wave79VariantKind(s.strip()) for s in str(args.variants).split(",") if s.strip()]

    print(f"=== Wave 79 Suite Starting on {device} ===", flush=True)
    decile_map = build_empirical_zipf_decile_map(device=torch.device("cpu"))

    seq_sprint = 256
    val_256 = load_fineweb_token_slice("fineweb_val_000000.bin", num_tokens=64 * seq_sprint + 8)
    x_val256 = val_256[: 64 * seq_sprint].view(64, seq_sprint)
    y_val256 = val_256[1 : 64 * seq_sprint + 1].view(64, seq_sprint)

    t_550_256 = build_known_architecture_adapter(
        KnownArchitectureKind.TEACHER_550M, device=device
    )
    ref_550_256 = build_teacher_reference_profile(t_550_256, x_val256, y_val256)
    del t_550_256
    if device.type == "cuda":
        torch.cuda.empty_cache()

    sprint_rows: list[dict[str, Any]] = []
    ind_rows: list[dict[str, Any]] = []
    mqar_rows: list[dict[str, Any]] = []
    confirm_summary: dict[str, Any] | None = None

    if int(args.sprint_steps) > 0:
        sprint_cfg = ShortTrainConfig(
            steps=int(args.sprint_steps),
            batch_size=32,
            seq_len=seq_sprint,
            eval_interval=max(50, int(args.sprint_steps) // 4),
            eval_tokens=int(x_val256.numel()),
            wave_id="W79",
            plan_id="EXPERIMENT_PLAN_099",
        )
        for var in variants:
            print(f"\n--- [Phase 1 Sprint] {var.value} ({args.sprint_steps} steps) ---", flush=True)
            adapter, s_row = run_wave79_controlled_sprint(
                var, sprint_cfg, x_val256, y_val256, device=device
            )
            s_row["xray_vs_550m"] = compute_stratified_nat_gap_xray(
                adapter, x_val256, y_val256, decile_map=decile_map, ref_profile=ref_550_256
            )
            s_row["copy_and_schur_audit"] = audit_wave79_copy_and_schur_diagnostics(
                adapter, x_val256, y_val256, decile_map
            )
            s_row["mtp_horizon_audit"] = inspect_mtp_horizon_weights_and_rollout_cosines(
                adapter.fused_model, x_val256, y_val256
            )
            sprint_rows.append(s_row)
            _save_atomic_json(
                args.json_out,
                {
                    "wave_id": "W79",
                    "plan_id": "EXPERIMENT_PLAN_099",
                    "device": str(device),
                    "total_suite_wall_sec": round(time.perf_counter() - t0_all, 2),
                    "controlled_sprints": sprint_rows,
                    "induction_probes": ind_rows,
                    "mqar_probes": mqar_rows,
                    "multishard_wsd_speedrun": confirm_summary,
                },
            )
            print(
                f"  -> {var.value}: {s_row['tok_per_s']:,.1f} tok/s | "
                f"Hot CE={s_row['final_hot_val_ce']:.4f} | SlowEMA={s_row['final_slow_ema_val_ce']:.4f} | "
                f"L_inf={s_row['asymptote_fit']['l_inf_asymptote']:.4f} | "
                f"m=0 Copy ΔCE={s_row['copy_and_schur_audit']['multiplicity_buckets']['m0_unseen_recall']['net_copy_nat_savings']:+.4f}",
                flush=True,
            )
            del adapter
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not args.skip_probes:
        probe_vars = [
            v
            for v in (
                Wave79VariantKind.V0_SURPRISAL_WH_INDUCTION_4L_D256,
                Wave79VariantKind.V2_FUSED_UNIFIED_4L_D256,
            )
            if v in variants
        ] or variants[:1]
        print(f"\n--- [Phase 2 Probes] 50,257-Vocab Induction & MQAR on {[v.value for v in probe_vars]} ---", flush=True)
        p_cfg = SyntheticProbeConfig()
        ind_rows, mqar_rows = run_wave79_induction_and_mqar_probes(probe_vars, p_cfg, device=device)
        _save_atomic_json(
            args.json_out,
            {
                "wave_id": "W79",
                "plan_id": "EXPERIMENT_PLAN_099",
                "device": str(device),
                "total_suite_wall_sec": round(time.perf_counter() - t0_all, 2),
                "controlled_sprints": sprint_rows,
                "induction_probes": ind_rows,
                "mqar_probes": mqar_rows,
                "multishard_wsd_speedrun": confirm_summary,
            },
        )
        for ir in ind_rows:
            print(
                f"  [Induction T={ir['seq_len']:3d}] {ir['arch']}: "
                f"Step0={100.0 * ir['step0_zero_shot_top1_acc']:.1f}% -> "
                f"Trained={100.0 * ir['induction_query_top1_acc']:.2f}% (CE={ir['induction_query_ce']:.4f})",
                flush=True,
            )
        for mr in mqar_rows:
            print(
                f"  [MQAR K={mr['num_pairs']}] {mr['arch']}: "
                f"Top1={100.0 * mr['mqar_top1_acc']:.2f}% (CE={mr['mqar_ce']:.4f})",
                flush=True,
            )

    if int(args.confirm_tokens) > 0:
        chosen_var = _select_confirm_variant(str(args.confirm_variant), sprint_rows)
        seq_spin = 512
        val_512 = load_fineweb_token_slice("fineweb_val_000000.bin", num_tokens=32 * seq_spin + 8)
        x_val512 = val_512[: 32 * seq_spin].view(32, seq_spin)
        y_val512 = val_512[1 : 32 * seq_spin + 1].view(32, seq_spin)

        t_550_512 = build_known_architecture_adapter(
            KnownArchitectureKind.TEACHER_550M, device=device
        )
        ref_550_512 = build_teacher_reference_profile(t_550_512, x_val512, y_val512)
        del t_550_512
        if device.type == "cuda":
            torch.cuda.empty_cache()

        spin_cfg = MultiShardConfirmConfig(
            target_tokens=int(args.confirm_tokens),
            batch_size=32,
            seq_len=seq_spin,
            eval_interval=500,
            wave_id="W79",
            plan_id="EXPERIMENT_PLAN_099",
        )
        print(
            f"\n=== [Phase 3 Multi-Shard WSD Speedrun] {chosen_var.value} for "
            f"{args.confirm_tokens:,} tokens ===",
            flush=True,
        )
        confirm_summary = run_wave79_300m_multishard_wsd_speedrun(
            chosen_var, spin_cfg, x_val512, y_val512, decile_map, ref_550_512, device=device
        )
        print(
            f"\n>>> [Wave 79 {args.confirm_tokens // 1_000_000}M COMPLETE] "
            f"Official 10,485,760-Token Shard-0 Val CE = "
            f"{confirm_summary['official_shard0_eval']['official_shard0_val_ce']:.5f} "
            f"(Top-1 = {confirm_summary['official_shard0_eval']['official_shard0_top1_pct']:.2f}%, "
            f"Steady = {confirm_summary['tok_per_s']:,.1f} tok/s) <<<",
            flush=True,
        )

    payload = {
        "wave_id": "W79",
        "plan_id": "EXPERIMENT_PLAN_099",
        "device": str(device),
        "total_suite_wall_sec": round(time.perf_counter() - t0_all, 2),
        "controlled_sprints": sprint_rows,
        "induction_probes": ind_rows,
        "mqar_probes": mqar_rows,
        "multishard_wsd_speedrun": confirm_summary,
    }
    _save_atomic_json(args.json_out, payload)
    print(f"\nSaved complete Wave 79 summary -> {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
