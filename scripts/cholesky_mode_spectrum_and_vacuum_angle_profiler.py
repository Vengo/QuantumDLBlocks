#!/usr/bin/env python3
r"""Per-Mode Cholesky Spectrum, Vacuum Angle $\cos\theta_{\text{vac}}(t, r)$, & Zipf Token-Decile Specialization Profiler (`EXPERIMENT_PLAN_080.md`).

Inspects every individual Cholesky mode $r \in \{0, 1, \dots, R-1\}$ across $R \in \{8, 12\}$ on held-out
`FineWeb` Shard-0 validation tokens (`65,536` tokens), reporting:
1. Per-mode normalized energy share $E_{t,r}$ and prior-weighted mixture responsibility $\pi_{t,r}$,
2. Per-mode Vacuum-Sphere confidence angle $\cos\theta_{\text{vac}}(t, r)$ ($\text{mean}, \text{std}, P_{10}, P_{90}$),
3. Fraction of token positions where excited mode $r \ge 1$ is the dominant excited branch, and
4. Token-frequency decile specialization (`Decile 0` frequent function words .. `Decile 9` rare tail entities).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.benchmarks.noclick_vacuum_and_smith_sgd_lab import (  # noqa: E402
    NoClickVacuumCholeskyGPT,
    NoClickVacuumCholeskySpec,
    NoClickVacuumVariantMode,
    _apply_global_optimizer_step,
    evaluate_noclick_vacuum_cholesky_model,
    profile_per_mode_cholesky_spectrum_and_vacuum_angles,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import LongSeqFineWebStream  # noqa: E402
from qdlblocks.benchmarks.vacuum_subspace_cholesky_lab import DEFAULT_VACUUM_SUBSPACE_DIM  # noqa: E402
from qdlblocks.benchmarks.variable_rank_cholesky_vacuum_lab import (  # noqa: E402
    build_variable_rank_cholesky_optimizers,
    initialize_and_calibrate_variable_rank_cholesky_model_,
)
from qdlblocks.layers.quantum_statmech_channels import (  # noqa: E402
    QuantumMomentumScheduleMode,
    compute_riemannian_momentum_beta,
)


def _train_model_for_profiling(
    model: NoClickVacuumCholeskyGPT,
    stream: LongSeqFineWebStream,
    train_steps: int,
) -> None:
    """Train `model` for `train_steps` on `FineWeb` Shards 1..9 before running the per-mode spectrum profiler."""
    if train_steps <= 0:
        return
    optimizers = build_variable_rank_cholesky_optimizers(model)
    model.train()
    warm_steps = max(1, int(0.08 * train_steps))
    for step in range(1, train_steps + 1):
        lr_mult = (
            0.20 + 0.80 * (float(step - 1) / float(max(1, warm_steps - 1)))
            if step <= warm_steps
            else 0.12 + 0.88 * 0.5 * (1.0 + math.cos(math.pi * float(step - warm_steps) / float(max(1, train_steps - warm_steps))))
        )
        mom_beta = compute_riemannian_momentum_beta(step, train_steps, QuantumMomentumScheduleMode.LOW_STEADY_MOMENTUM_070)
        idx_in, targets = stream.sample_train_batch(step)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss = model.forward_loss(idx_in, targets)
        loss.backward()
        _apply_global_optimizer_step(model, optimizers, idx_in, targets, lr_mult=lr_mult, mom_beta=mom_beta)
    del optimizers


def _print_mode_spectrum_table(rank_r: int, stage: str, val_metrics: dict[str, float], prof: dict[str, Any]) -> None:
    """Print formatted per-mode spectrum and vacuum-angle table to stdout."""
    print(f"\n--- [R={rank_r} | {stage}] Pure Neural Val CE: {val_metrics['val_ce']:.4f} (PPL {val_metrics['val_ppl']:.2f}) | Active Modes: {val_metrics['effective_active_branches']:.2f}/{rank_r} ---")
    print(f"{'Mode':<6} | {'E_{t,r}':<9} | {'pi_{t,r}':<9} | {'DomExc%':<8} | {'cos(vac) Mean (P10..P90)':<25} | {'Tail/Head Post':<14} | {'Role'}")
    print("-" * 118)
    for row in prof["per_mode_spectrum"]:
        m_idx = row["mode_index"]
        cos_str = f"{row['cos_theta_vac_mean']:.3f} ({row['cos_theta_vac_p10']:.3f}..{row['cos_theta_vac_p90']:.3f})"
        dom_str = f"{row['dominant_excited_frac']*100:6.2f}%" if m_idx >= 1 else "  anchor"
        print(
            f"r={m_idx:<4d} | {row['mean_energy_share']:<9.4f} | {row['mean_mixture_pi']:<9.4f} | "
            f"{dom_str:<8} | {cos_str:<25} | {row['tail_over_head_posterior_ratio']:<14.3f} | {row['role']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile per-mode Cholesky energy, vacuum angle, and token-decile specialization.")
    parser.add_argument("--ranks", type=str, default="8,12", help="Comma-separated Cholesky ranks R to profile (default: 8,12)")
    parser.add_argument("--train-steps", type=int, default=300, help="Training steps before post-training profile (default: 300)")
    parser.add_argument("--max-val-batches", type=int, default=8, help="Number of 8,192-token validation batches to profile (default: 8)")
    parser.add_argument("--use-noclick-sink", action="store_true", default=True, help="Enable Quantum No-Click POVM Vacuum Sink")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cuda or cpu)")
    parser.add_argument("--json-out", type=str, default=".cache/cholesky_mode_spectrum_profile.json", help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranks = [int(x.strip()) for x in args.ranks.split(",") if x.strip()]
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    stream = LongSeqFineWebStream(device=device)

    t0 = time.perf_counter()
    rank_profiles: dict[str, Any] = {}

    for rank_r in ranks:
        v_mode = (
            NoClickVacuumVariantMode.N2_RANK12_NOCLICK_VACUUM_POVM
            if rank_r >= 12
            else NoClickVacuumVariantMode.N1_RANK8_NOCLICK_VACUUM_POVM
        )
        spec = NoClickVacuumCholeskySpec(
            variant_mode=v_mode,
            num_cholesky_modes=rank_r,
            vacuum_dim=DEFAULT_VACUUM_SUBSPACE_DIM,
            use_noclick_vacuum_sink=bool(args.use_noclick_sink),
            seed=1337,
        )
        torch.manual_seed(spec.seed)
        model = NoClickVacuumCholeskyGPT(spec).to(device=device)
        initialize_and_calibrate_variable_rank_cholesky_model_(model, stream)
        model.clamp_identity_dispersion_and_scalars_()

        step0_val = evaluate_noclick_vacuum_cholesky_model(model, stream)
        step0_prof = profile_per_mode_cholesky_spectrum_and_vacuum_angles(model, stream, max_val_batches=args.max_val_batches)
        _print_mode_spectrum_table(rank_r, "Step 0 Calibrated", step0_val, step0_prof)

        trained_entry: dict[str, Any] | None = None
        if args.train_steps > 0:
            _train_model_for_profiling(model, stream, train_steps=int(args.train_steps))
            trained_val = evaluate_noclick_vacuum_cholesky_model(model, stream)
            trained_prof = profile_per_mode_cholesky_spectrum_and_vacuum_angles(model, stream, max_val_batches=args.max_val_batches)
            _print_mode_spectrum_table(rank_r, f"Step {args.train_steps} Trained", trained_val, trained_prof)
            trained_entry = {"val_metrics": trained_val, "profile": trained_prof}

        rank_profiles[f"R{rank_r}"] = {
            "num_cholesky_modes": rank_r,
            "use_noclick_vacuum_sink": bool(args.use_noclick_sink),
            "step0": {"val_metrics": step0_val, "profile": step0_prof},
            "trained": trained_entry,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "plan": "EXPERIMENT_PLAN_080",
        "tool": "cholesky_mode_spectrum_and_vacuum_angle_profiler",
        "ranks": ranks,
        "train_steps": args.train_steps,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "profiles": rank_profiles,
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved per-mode Cholesky spectrum & vacuum angle profile to {out_path}")


if __name__ == "__main__":
    main()
