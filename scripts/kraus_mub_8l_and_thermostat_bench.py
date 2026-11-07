r"""Steps 1, 3, & 4 CLI Benchmark: Analytical Parseval Temperature ($\beta_0 = 10.0$), Orthonormal Two-Basis $U(16)$ MUB Tomography, Rank-2 CPTP Stiefel Kraus Channels ($\operatorname{St}_{\mathbb{C}}(32, 16)$), `L = 8` Trunk Rebalancing, & Closed-Loop Shadow-EMA Thermostat (`EXPERIMENT_PLAN_084.md` / `Wave 64`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

from qdlblocks.benchmarks.kraus_mub_thermostat_8l_lab import (
    DutyMatchedGroupMultipliers, KrausMUBThermostatGPT, KrausMUBThermostatSpec, KrausMUBVariantMode,
    RiemannianShadowEMAThermostat, build_kraus_mub_optimizers, evaluate_kraus_mub_model,
    initialize_kraus_mub_model_, run_wave64_ablation_and_thermostat_suite, step_kraus_mub_optimizers,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import LongSeqFineWebStream


def run_step4_hot_thermostat_stress_test(
    stream: LongSeqFineWebStream, steps: int = 120, eval_interval: int = 30, hot_lr_scale: float = 2.20
) -> dict[str, Any]:
    """Compare un-thermostatted hot (`2.20x`) training vs. closed-loop `RiemannianShadowEMAThermostat` (`<= 8` args)."""
    device = stream.device
    tokens_per_step = stream.batch_size * stream.seq_len
    dm = DutyMatchedGroupMultipliers(trunk_mult=1.50, vocab_mult=2.50, ngram_mult=1.50, scalar_mult=0.75)
    spec = KrausMUBThermostatSpec(
        variant_mode=KrausMUBVariantMode.W64_A2_KRAUS_RANK2_PLUS_MUB_4L,
        num_layers=4, kraus_rank=2, use_mub_unitary_probes=True,
    )
    arms_out: list[dict[str, Any]] = []

    for arm_name, enable_feedback in [("HOT_2P2X_UNTHERMOSTATTED", False), ("HOT_2P2X_CLOSED_LOOP_THERMOSTAT", True)]:
        torch.manual_seed(spec.seed)
        model = KrausMUBThermostatGPT(spec).to(device=device)
        initialize_kraus_mub_model_(model)
        optimizers = build_kraus_mub_optimizers(model, mults=dm)
        thermostat = RiemannianShadowEMAThermostat(model)

        lr_scale = float(hot_lr_scale)
        checkpoints: list[dict[str, Any]] = []
        model.train()
        for step in range(1, steps + 1):
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            idx_in, targets = stream.sample_train_batch(step)
            active_ids = torch.unique(torch.cat([idx_in.reshape(-1), targets.reshape(-1)]))
            loss = model.forward_loss(idx_in, targets)
            loss.backward()
            del loss

            mom_beta = 0.92 if step > 5 else 0.50
            step_kraus_mub_optimizers(
                model, optimizers, active_ids=active_ids, tokens_in_step=tokens_per_step, lr_mult=lr_scale, mom_beta=mom_beta
            )
            thermostat.update_after_step(model)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            if step % eval_interval == 0 or step == steps:
                chk = thermostat.check_and_thermostat(
                    model, stream, current_lr_scale=lr_scale, step=step, apply_feedback=enable_feedback
                )
                if enable_feedback and chk["triggered"]:
                    lr_scale = float(chk["lr_scale_after"])
                checkpoints.append({
                    "step": step, "hot_val_ce": chk["hot_val_ce"], "ema_val_ce": chk["ema_val_ce"],
                    "active_val_ce": chk["active_val_ce"], "thermal_gap_ce": chk["thermal_gap_ce"],
                    "triggered": chk["triggered"], "lr_scale_before": chk["lr_scale_before"], "lr_scale_after": chk["lr_scale_after"],
                })
                print(
                    f"[STEP4 {arm_name}] Step {step:3d} | Hot CE: {chk['hot_val_ce']:.4f} | EMA CE: {chk['ema_val_ce']:.4f} | "
                    f"Active CE: {chk['active_val_ce']:.4f} | Gap: {chk['thermal_gap_ce']:+.4f} | "
                    f"Trig: {chk['triggered']} | LR: {chk['lr_scale_before']:.3f}x -> {chk['lr_scale_after']:.3f}x",
                    flush=True,
                )

        final_ev = evaluate_kraus_mub_model(model, stream)
        arms_out.append({
            "arm_name": arm_name, "enable_feedback": enable_feedback, "initial_lr_scale": hot_lr_scale,
            "final_lr_scale": round(lr_scale, 4), "final_val_ce": final_ev["val_ce"],
            "num_triggers": len(thermostat.trigger_events), "trigger_events": thermostat.trigger_events,
            "checkpoints": checkpoints,
        })
        del model, optimizers, thermostat
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"hot_stress_arms": arms_out}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wave 64 Steps 1, 3, & 4: Analytical Beta, 2xU(16) MUB, Rank-2 Kraus, L=8, & Shadow-EMA Thermostat")
    parser.add_argument("--steps", type=int, default=300, help="Total training steps per variant")
    parser.add_argument("--eval-interval", type=int, default=60, help="Validation & thermostat check interval")
    parser.add_argument("--quench-steps", type=int, default=60, help="Final cosine quench steps")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Micro-batch size B")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length T")
    parser.add_argument("--val-tokens", type=int, default=524_288, help="Validation token slice")
    parser.add_argument("--only-variant", type=str, default="", help="Optional single variant ID to run (e.g., W64_A3_FULL_CONSOLIDATED_8L_THERMOSTAT)")
    parser.add_argument("--hot-stress-steps", type=int, default=120, help="Steps for Step-4 hot thermostat stress test (0 to skip)")
    parser.add_argument("--json-out", type=str, default=".cache/plan084_kraus_mub_8l_thermostat_bench.json", help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"=== WAVE 64 STEPS 1, 3, & 4: KRAUS + 2xU(16) MUB + L=8 + SHADOW-EMA THERMOSTAT ===\n"
        f"Device: {device} | Steps: {args.steps} (eval={args.eval_interval}, quench={args.quench_steps}) | "
        f"Batch: B={args.batch_size}x{args.grad_accum_steps}, T={args.seq_len} | ValTokens: {args.val_tokens:,}",
        flush=True,
    )
    t0 = time.perf_counter()
    stream = LongSeqFineWebStream(
        seq_len=args.seq_len, batch_size=args.batch_size, val_tokens=args.val_tokens, device=device
    )
    suite_report = run_wave64_ablation_and_thermostat_suite(
        stream, steps=args.steps, eval_interval=args.eval_interval,
        quench_steps=args.quench_steps, grad_accum_steps=args.grad_accum_steps,
        only_variant=(args.only_variant.strip() or None),
    )
    if args.hot_stress_steps > 0:
        stress_report = run_step4_hot_thermostat_stress_test(
            stream, steps=args.hot_stress_steps, eval_interval=max(20, args.hot_stress_steps // 4)
        )
        suite_report.update(stress_report)

    suite_report["total_wall_sec"] = round(time.perf_counter() - t0, 2)
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(suite_report, indent=2), encoding="utf-8")
    print(f"[Saved Wave 64 Benchmark JSON -> {out_path} ({suite_report['total_wall_sec']:.1f}s)]", flush=True)


if __name__ == "__main__":
    main()
