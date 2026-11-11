#!/usr/bin/env python3
"""Master `Wave 76` Cross-Architecture Diagnostic Suite Runner (`EXPERIMENT_PLAN_096.md`).

Executes four coordinated diagnostic phases on known repository architectures (`Waves 45–75`)
plus the `Wave 76` $p=3$ + `<BOS>`-masked Baton diagnostic probe:
  Phase 1: Pretrained `CompiledLeaderboardClassicalGPT` Anchor X-Rays (`50M`, `300M`, `550M` tokens).
  Phase 2: Controlled `400`-step (`3.28M`-token) FineWeb Shard-1 sprints across 7 trainable
           architectures with `StandardizedExperimentLogger`, Dual-Horizon `ShadowEMA`, Power-Law
           Asymptote $L_\\infty$ fitting, Stratified Nat-Gap Attribution vs. the `550M` (`3.35` CE)
           anchor, and Baton Compartment Leave-One-Out + Scale Sweep + Uncollapsed Ridge Audits.
  Phase 3: Synthetic `50,257`-vocab Fuzzy Induction + Pre-`<BOS>` Trap Sweep ($T \\in \\{64, 128, 256\\}$).
  Phase 4: `50,257`-vocab Multi-Query Associative Recall (`MQAR`, $K \\in \\{4, 8\\}$) and Karpathy
           `N=1, 4` Real FineWeb Memorization & Readout Ceiling Probes.
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

from qdlblocks.benchmarks.architecture_mechanistic_probes import (
    run_baton_compartment_ablation_audit,
    run_controlled_short_fineweb_sprint,
    run_memorization_and_readout_ceiling_probe,
    run_synthetic_induction_sweep,
    run_synthetic_mqar_probe,
)
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
from qdlblocks.benchmarks.fused_bf16_four_compartment_baton_lab import (
    FusedBF16FourCompartmentBatonLM,
)

DEFAULT_SPRINT_STEPS: int = 400
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_SEQ_LEN: int = 256
DEFAULT_EVAL_TOKENS: int = 16384


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments (`<= 8` args)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_SPRINT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--eval-tokens", type=int, default=DEFAULT_EVAL_TOKENS)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / ".cache" / "wave76_architecture_diagnostic_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete `Wave 76` Cross-Architecture Diagnostic Suite (`<= 8` args)."""
    args = parse_args()
    device = torch.device(args.device)
    suite_t0 = time.perf_counter()

    n_val_seq = max(2, args.eval_tokens // args.seq_len)
    val_tokens = load_fineweb_token_slice(
        "fineweb_val_000000.bin", num_tokens=n_val_seq * args.seq_len + 1
    )
    x_val = val_tokens[:-1].view(n_val_seq, args.seq_len)
    y_val = val_tokens[1:].view(n_val_seq, args.seq_len)
    decile_map = build_empirical_zipf_decile_map(device=torch.device("cpu"))

    print("================================================================================")
    print("WAVE 76: CROSS-ARCHITECTURE MECHANISTIC X-RAY & SHORT-SPRINT DIAGNOSTIC SUITE")
    print(f"Device: {device} | Eval Slice: {x_val.numel():,} Shard-0 tokens ({n_val_seq}x{args.seq_len})")
    print("================================================================================")

    # ------------------------------------------------------------------
    # Phase 1: Pretrained Classical Leaderboard Reference Anchors
    # ------------------------------------------------------------------
    print("\n[Phase 1/4] Evaluating Pretrained Leaderboard Teacher Anchors (50M, 300M, 550M)...")
    anchor_550m = build_known_architecture_adapter(KnownArchitectureKind.TEACHER_550M, device=device)
    ref_550m = build_teacher_reference_profile(anchor_550m, x_val, y_val)
    teacher_xrays: list[dict[str, Any]] = []

    for t_kind in (
        KnownArchitectureKind.TEACHER_50M,
        KnownArchitectureKind.TEACHER_300M,
        KnownArchitectureKind.TEACHER_550M,
    ):
        t_adap = (
            anchor_550m
            if t_kind == KnownArchitectureKind.TEACHER_550M
            else build_known_architecture_adapter(t_kind, device=device)
        )
        tx = compute_stratified_nat_gap_xray(
            t_adap, x_val, y_val, decile_map=decile_map, ref_profile=ref_550m
        )
        teacher_xrays.append(tx)
        print(
            f"  [Anchor] {tx['label']:38s} | Val CE={tx['val_ce']:.4f} | "
            f"Copy CE={tx['copy_vs_recall']['in_doc_copy']['ce']:.4f} | "
            f"Recall CE={tx['copy_vs_recall']['out_of_doc_recall']['ce']:.4f} | "
            f"Span={tx['mean_logit_span_max_minus_median']:.2f}"
        )
        if t_adap is not anchor_550m:
            del t_adap
    del anchor_550m
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Phase 2: Controlled FineWeb Sprints + Stratified X-Ray + Baton Audits
    # ------------------------------------------------------------------
    sprint_kinds = [
        KnownArchitectureKind.W74_BATON_1L_D64,
        KnownArchitectureKind.W74_BATON_2L_D64,
        KnownArchitectureKind.W76_SHARP_BATON_2L_D64_P3,
        KnownArchitectureKind.W75_EXACT_MIXED_2L_D128,
        KnownArchitectureKind.W74_BATON_2L_D256,
        KnownArchitectureKind.W45_CLASSICAL_2L_D512,
        KnownArchitectureKind.W61_ZERO_PPMI_CHOLESKY_4L_D512,
    ]
    train_cfg = ShortTrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        eval_interval=max(50, args.steps // 4),
        eval_tokens=args.eval_tokens,
        wave_id="W76",
        plan_id="EXPERIMENT_PLAN_096",
    )

    print(
        f"\n[Phase 2/4] Running {args.steps}-Step ({args.steps * args.batch_size * args.seq_len / 1e6:.2f}M Token) "
        f"Controlled FineWeb Sprints Across {len(sprint_kinds)} Architectures..."
    )
    sprint_runs: list[dict[str, Any]] = []
    baton_compartment_audits: dict[str, Any] = {}

    for kind in sprint_kinds:
        adapter, sprint_summary = run_controlled_short_fineweb_sprint(
            kind, train_cfg, x_val, y_val, device=device
        )
        xray = compute_stratified_nat_gap_xray(
            adapter, x_val, y_val, decile_map=decile_map, ref_profile=ref_550m
        )
        sprint_summary["stratified_xray"] = xray

        if isinstance(adapter.model, FusedBF16FourCompartmentBatonLM):
            comp_audit = run_baton_compartment_ablation_audit(
                adapter.model, x_val, y_val, max_batches=4
            )
            baton_compartment_audits[kind.value] = comp_audit
            sprint_summary["baton_compartment_audit"] = comp_audit

        asym = sprint_summary["asymptote_fit"]
        print(
            f"  [Sprint] {sprint_summary['arch']:34s} | Speed={sprint_summary['tok_per_s']:9,.1f} tok/s | "
            f"Hot CE={sprint_summary['final_hot_val_ce']:.4f} | SlowEMA CE={sprint_summary['final_slow_ema_val_ce']:.4f} "
            f"(Δth={sprint_summary['final_thermal_gap_slow']:+.4f}) | L_inf={asym['l_inf_asymptote']:.3f} | "
            f"Copy ΔCE={xray['copy_vs_recall']['in_doc_copy']['delta_ce_vs_ref']:+.3f} | "
            f"Recall ΔCE={xray['copy_vs_recall']['out_of_doc_recall']['delta_ce_vs_ref']:+.3f}"
        )
        if kind.value in baton_compartment_audits:
            ca = baton_compartment_audits[kind.value]
            print(
                f"           -> Baton Audit: Full={ca['full_ce']:.4f} | BestBeta={ca['best_scale_sweep_ce']:.4f} | "
                f"UncollapsedRidge={ca['uncollapsed_ridge_mup_heldout_ce']:.4f} | "
                f"BOS-Leak={ca['attn_telemetry']['cross_bos_leakage_fraction']*100:.2f}% | "
                f"LOO ΔCE={ca['leave_one_out_delta_ce']}"
            )
        sprint_runs.append(sprint_summary)
        del adapter
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Phase 3 & 4: Synthetic Fuzzy Induction, MQAR, & Memorization Ceiling
    # ------------------------------------------------------------------
    probe_cfg = SyntheticProbeConfig(
        induction_steps=180,
        mqar_steps=220,
        overfit_steps=220,
        batch_size=32,
        eval_batches=8,
    )
    mech_kinds = [
        KnownArchitectureKind.W74_BATON_2L_D64,
        KnownArchitectureKind.W76_SHARP_BATON_2L_D64_P3,
        KnownArchitectureKind.W74_BATON_2L_D256,
        KnownArchitectureKind.W45_CLASSICAL_2L_D512,
        KnownArchitectureKind.W61_ZERO_PPMI_CHOLESKY_4L_D512,
    ]

    print("\n[Phase 3/4] Running 50,257-Vocab Fuzzy Induction + Pre-<BOS> Trap Sweep (T=64, 128, 256)...")
    induction_results = run_synthetic_induction_sweep(mech_kinds, probe_cfg, device=device)
    for r in induction_results:
        print(
            f"  [Induction] {r['arch']:34s} | T={r['seq_len']:3d} | Query CE={r['induction_query_ce']:.4f} | "
            f"Top-1={r['induction_query_top1_acc']*100:5.1f}% | Pre-BOS Trap={r['pre_bos_false_trap_rate']*100:5.1f}%"
        )

    print("\n[Phase 4/4] Running 50,257-Vocab MQAR (K=4, 8) & Real FineWeb Memorization Ceiling (N=1, 4)...")
    mqar_results = run_synthetic_mqar_probe(mech_kinds, probe_cfg, device=device)
    for r in mqar_results:
        print(
            f"  [MQAR]      {r['arch']:34s} | K={r['num_pairs']:2d}  | MQAR CE={r['mqar_ce']:.4f} | "
            f"Top-1={r['mqar_top1_acc']*100:5.1f}%"
        )

    memorization_results = run_memorization_and_readout_ceiling_probe(
        mech_kinds, probe_cfg, device=device
    )
    for r in memorization_results:
        print(
            f"  [Overfit]   {r['arch']:34s} | N={r['num_records']}   | Mem CE={r['final_memorization_ce']:.4f} | "
            f"Top-1={r['memorization_top1_acc']*100:5.1f}% | MaxProb={r['mean_max_prob']:.3f} | Span={r['mean_logit_span']:.2f}"
        )

    total_sec = time.perf_counter() - suite_t0
    payload = {
        "wave_id": "W76",
        "plan_id": "EXPERIMENT_PLAN_096",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "eval_tokens": int(x_val.numel()),
        "total_wall_sec": round(total_sec, 2),
        "reference_anchor_550m_ce": round(ref_550m.anchor_val_ce, 5),
        "teacher_anchor_xrays": teacher_xrays,
        "short_fineweb_sprints": sprint_runs,
        "baton_compartment_audits": baton_compartment_audits,
        "synthetic_induction_sweep": induction_results,
        "synthetic_mqar_probe": mqar_results,
        "memorization_ceiling_probe": memorization_results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWave 76 Diagnostic Suite finished in {total_sec:.1f}s -> {args.json_out}")


if __name__ == "__main__":
    main()
