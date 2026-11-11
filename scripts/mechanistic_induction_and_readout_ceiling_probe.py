#!/usr/bin/env python3
"""Standalone Mechanistic Induction, MQAR, Memorization Ceiling, & Baton Compartment Audit CLI (`Wave 76` / `EXPERIMENT_PLAN_096.md`).

Runs fast synthetic & small-sample mechanistic probes across known architectures:
  1. `50,257`-vocab Fuzzy Induction across $T \\in \\{64, 128, 256\\}$ with pre-`<BOS>` false traps,
  2. `50,257`-vocab Multi-Query Associative Recall (`MQAR`, $K \\in \\{4, 8\\}$),
  3. Karpathy `N=1, 4` real FineWeb sequence memorization & readout dynamic-range ceiling, and
  4. Baton compartment leave-one-out, $\\beta$ scale sweep, `<BOS>` leakage, & uncollapsed ridge probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qdlblocks.benchmarks.architecture_mechanistic_probes import (
    run_baton_compartment_ablation_audit,
    run_memorization_and_readout_ceiling_probe,
    run_synthetic_induction_sweep,
    run_synthetic_mqar_probe,
)
from qdlblocks.benchmarks.architecture_xray_diagnostics import (
    KnownArchitectureKind,
    SyntheticProbeConfig,
    build_known_architecture_adapter,
    load_fineweb_token_slice,
)
from qdlblocks.benchmarks.fused_bf16_four_compartment_baton_lab import (
    FusedBF16FourCompartmentBatonLM,
)

DEFAULT_INDUCTION_MODELS: str = (
    "w74_baton_2l_d64,w76_sharp_baton_2l_d64_p3_docmask,w74_baton_2l_d256,w45_classical_2l_d512"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments (`<= 8` args)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=str, default=DEFAULT_INDUCTION_MODELS)
    parser.add_argument("--induction-steps", type=int, default=180)
    parser.add_argument("--mqar-steps", type=int, default=220)
    parser.add_argument("--overfit-steps", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / ".cache" / "mechanistic_induction_and_ceiling_probe.json",
    )
    return parser.parse_args()


def main() -> None:
    """Execute mechanistic induction, MQAR, memorization, and compartment probes (`<= 8` args)."""
    args = parse_args()
    device = torch.device(args.device)
    kinds = [KnownArchitectureKind(m.strip()) for m in args.models.split(",") if m.strip()]
    cfg = SyntheticProbeConfig(
        induction_steps=args.induction_steps,
        mqar_steps=args.mqar_steps,
        overfit_steps=args.overfit_steps,
        batch_size=args.batch_size,
    )

    print("=== [1/4] Running 50,257-Vocab Fuzzy Induction + Pre-<BOS> Trap Sweep ===")
    ind_rows = run_synthetic_induction_sweep(kinds, cfg, device=device)
    for r in ind_rows:
        print(
            f"  {r['arch']:36s} | T={r['seq_len']:3d} | Query CE={r['induction_query_ce']:.4f} | "
            f"Top-1={r['induction_query_top1_acc']*100:5.1f}% | TrapRate={r['pre_bos_false_trap_rate']*100:5.1f}%"
        )

    print("\n=== [2/4] Running 50,257-Vocab Multi-Query Associative Recall (MQAR) ===")
    mqar_rows = run_synthetic_mqar_probe(kinds, cfg, device=device)
    for r in mqar_rows:
        print(
            f"  {r['arch']:36s} | K={r['num_pairs']:2d} | MQAR CE={r['mqar_ce']:.4f} | "
            f"Top-1={r['mqar_top1_acc']*100:5.1f}%"
        )

    print("\n=== [3/4] Running Karpathy N=1, 4 FineWeb Memorization & Readout Ceiling Probe ===")
    mem_rows = run_memorization_and_readout_ceiling_probe(kinds, cfg, device=device)
    for r in mem_rows:
        print(
            f"  {r['arch']:36s} | N={r['num_records']} ({r['total_tokens']:4d} tok) | "
            f"Mem CE={r['final_memorization_ce']:.4f} | Top-1={r['memorization_top1_acc']*100:5.1f}% | "
            f"Span={r['mean_logit_span']:.2f}"
        )

    print("\n=== [4/4] Running Baton Compartment Leave-One-Out & Scale Sweep Audit ===")
    val_raw = load_fineweb_token_slice("fineweb_val_000000.bin", num_tokens=32 * 256 + 1)
    x_eval = val_raw[:-1].view(32, 256)
    y_eval = val_raw[1:].view(32, 256)
    baton_audits = {}
    for kind in kinds:
        adapter = build_known_architecture_adapter(kind, device=device)
        if isinstance(adapter.model, FusedBF16FourCompartmentBatonLM):
            for s_idx in range(1, 41):
                adapter.train_step(x_eval[:16].to(device), y_eval[:16].to(device), s_idx, 40)
            audit = run_baton_compartment_ablation_audit(adapter.model, x_eval[16:], y_eval[16:])
            baton_audits[kind.value] = audit
            print(
                f"  {kind.value:36s} | Heldout CE={audit['full_ce']:.4f} | "
                f"BestBeta CE={audit['best_scale_sweep_ce']:.4f} | "
                f"Ridge CE={audit['uncollapsed_ridge_mup_heldout_ce']:.4f} | "
                f"BOS-Leak={audit['attn_telemetry']['cross_bos_leakage_fraction']*100:5.2f}%"
            )
        del adapter
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "induction_sweep": ind_rows,
        "mqar_probe": mqar_rows,
        "memorization_ceiling": mem_rows,
        "baton_compartment_audits": baton_audits,
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved Mechanistic Probe JSON -> {args.json_out}")


if __name__ == "__main__":
    main()
