#!/usr/bin/env python3
"""Constant-LR (Zero Cosine Schedule) Stability & Breakdown Ceiling Sweep (`Wave 63` / `EXPERIMENT_PLAN_083.md`).

Tests how high the learning rate can go on the consolidated `ExplicitDensityTomographicGPT`
(`T2_DENSITY_TOMOGRAPHY_PLUS_NOCLICK_AND_VACUUM`, $H = 16$ heads of $\\mathbb{C}^{16 \\times 16}$ outer-product
density matrices + `16 x 32 = 512D` Quantum State Tomography Readout) **without any cosine LR schedule**:

1. **Phase 1 — 100-Step Exponential LR Ramp (`0.25x -> 64.0x`)**:
   - Ramps `lr_mult` exponentially every step (`lr_mult(t) = 0.25 * (64.0 / 0.25)^(t / 100)`) with zero cosine
     decay to identify the instantaneous loss minimum $\\eta_{\\text{min-slope}}$ and the divergence threshold $\\eta_{\\text{crit}}$.
2. **Phase 2 — 300-Step Constant-LR Grid (`lr_mult in {1.0x, 2.0x, 4.0x, 8.0x, 16.0x, 32.0x}`)**:
   - Holds the learning rate strictly constant from Step 1 to Step 300 (no warmup, no cosine cooldown),
     measuring train CE stability (mean, max spike, rolling std), validation CE/PPL/Top-1 every 50 steps,
     gradient RMS by subsystem (`trunk_stiefel`, `vocab_512`, `scalars`), and density rank / vacuum telemetry.
3. **Phase 3 — 300-Step Decoupled Manifold vs. Scalar Constant-LR Probe**:
   - Pushes the Riemannian Stiefel/$U(16)$ + Vocab Manifold optimizers to extreme constant multipliers
     (`8.0x`, `16.0x`, `32.0x`) while keeping scalar/temperature `AdamW` groups at `2.0x` to isolate whether
     high-LR instability originates in the Euclidean scalar gates or the compact Stiefel/sphere manifolds.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from qdlblocks.benchmarks.experiment_logger import ExperimentIntentSpec, StandardizedExperimentLogger
from qdlblocks.benchmarks.explicit_density_tomographic_512d_lab import (
    DEFAULT_SHALLOW_NGRAM_LR,
    DEFAULT_SHALLOW_SCALAR_LR,
    ExplicitDensityTomographicGPT,
    ExplicitDensityTomographicSpec,
    ExplicitDensityVariantMode,
    LongSeqFineWebStream,
    build_explicit_density_tomographic_optimizers,
    evaluate_explicit_density_tomographic_model,
    initialize_explicit_density_tomographic_model_,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7)
# ==============================================================================
DEFAULT_RAMP_MIN_MULT: float = 0.25
DEFAULT_RAMP_MAX_MULT: float = 64.0
DEFAULT_RAMP_STEPS: int = 100
DEFAULT_CONST_LR_STEPS: int = 300
DEFAULT_CONST_EVAL_INTERVAL: int = 50
DEFAULT_STEADY_MOMENTUM_BETA: float = 0.70
DEFAULT_FAST_TEMP_BASE_LR: float = 0.08
DIVERGENCE_CE_THRESHOLD: float = 18.0


@dataclass(frozen=True)
class ConstantLRRunConfig:
    """Configuration for a single constant-LR (zero cosine schedule) run (`<= 8` fields, Rule 2)."""

    run_id: str
    manifold_lr_mult: float
    scalar_lr_mult: float
    mom_beta: float = DEFAULT_STEADY_MOMENTUM_BETA
    steps: int = DEFAULT_CONST_LR_STEPS
    eval_interval: int = DEFAULT_CONST_EVAL_INTERVAL
    seed: int = 1337


def _compute_subsystem_grad_rms(model: ExplicitDensityTomographicGPT) -> dict[str, float]:
    """Compute RMS gradient norms across (`trunk_stiefel`, `vocab_512`, `scalars`) before optimizer step (`<= 8` args)."""
    stiefel_sq, stiefel_n = 0.0, 0
    for p in model.real_stiefel_parameters():
        if p.grad is not None:
            stiefel_sq += float(p.grad.detach().float().square().sum().item())
            stiefel_n += p.numel()
    for p_re, p_im in model.complex_unitary_pairs():
        for p in (p_re, p_im):
            if p.grad is not None:
                stiefel_sq += float(p.grad.detach().float().square().sum().item())
                stiefel_n += p.numel()

    vocab_sq, vocab_n = 0.0, 0
    for p in (model.wte_512, model.w_det_512):
        if p.grad is not None:
            vocab_sq += float(p.grad.detach().float().square().sum().item())
            vocab_n += p.numel()

    scalar_params, fast_temp_params, reg_tables = model.scalar_and_probe_parameters()
    scalar_sq, scalar_n = 0.0, 0
    for p in scalar_params + fast_temp_params + reg_tables:
        if p.grad is not None:
            scalar_sq += float(p.grad.detach().float().square().sum().item())
            scalar_n += p.numel()

    return {
        "grad_rms_stiefel": round(math.sqrt(stiefel_sq / float(max(1, stiefel_n))), 6),
        "grad_rms_vocab": round(math.sqrt(vocab_sq / float(max(1, vocab_n))), 6),
        "grad_rms_scalar": round(math.sqrt(scalar_sq / float(max(1, scalar_n))), 6),
    }


def _step_decoupled_constant_lr_optimizers(
    model: ExplicitDensityTomographicGPT,
    optimizers: tuple[Any, Any, Any, Any],
    idx_in: torch.Tensor,
    targets: torch.Tensor,
    manifold_lr_mult: float,
    scalar_lr_mult: float,
    mom_beta: float,
) -> None:
    """Step 100% `AdamW`-free Riemannian Stiefel+$U(16)$, Vocab+N-Gram Gram-Muon, and Geodesic Scalar+Probe optimizers (`<= 8` args)."""
    q_trunk_opt, wte_opt, head_opt, geo_opt = optimizers
    q_trunk_opt.set_momentum(mom_beta)
    q_trunk_opt.set_lr_scale(manifold_lr_mult)
    q_trunk_opt.step()

    active_ids = torch.unique(torch.cat([idx_in.reshape(-1), targets.reshape(-1)]))
    wte_opt.set_momentum(mom_beta)
    wte_opt.step_active(active_ids, lr_scale=manifold_lr_mult)
    head_opt.set_momentum(mom_beta)
    head_opt.step_full(lr_scale=manifold_lr_mult, active_idx=active_ids)

    geo_opt.set_momentum(0.0 if mom_beta <= 0.0 else 0.80)
    geo_opt.step_active(active_idx=active_ids, lr_scale=scalar_lr_mult)
    model.restrict_scalars_()


def run_exponential_lr_range_finder(
    stream: LongSeqFineWebStream,
    steps: int = DEFAULT_RAMP_STEPS,
    min_mult: float = DEFAULT_RAMP_MIN_MULT,
    max_mult: float = DEFAULT_RAMP_MAX_MULT,
    seed: int = 1337,
) -> dict[str, Any]:
    """Run a 100-step exponential LR ramp (`min_mult -> max_mult`) to find the instability knee (`<= 8` args)."""
    device = stream.device
    spec = ExplicitDensityTomographicSpec(
        variant_mode=ExplicitDensityVariantMode.T2_DENSITY_TOMOGRAPHY_PLUS_NOCLICK_AND_VACUUM,
        enable_density_highway=True, enable_tomographic_readout=True,
        enable_noclick_and_vacuum=True, enable_two_register_bilinear=False, seed=seed,
    )
    torch.manual_seed(seed)
    model = ExplicitDensityTomographicGPT(spec).to(device=device)
    initialize_explicit_density_tomographic_model_(model)
    optimizers = build_explicit_density_tomographic_optimizers(model)
    model.train()

    records: list[dict[str, Any]] = []
    smoothed_loss: float | None = None
    best_smoothed_loss = float("inf")
    best_lr_mult = min_mult
    divergence_lr_mult: float | None = None

    print(f"=== Phase 1: Exponential LR Range Test ({steps} steps, {min_mult:.2f}x -> {max_mult:.2f}x) ===", flush=True)
    for step in range(1, steps + 1):
        frac = float(step - 1) / float(max(1, steps - 1))
        lr_mult = min_mult * ((max_mult / min_mult) ** frac)
        idx_in, targets = stream.sample_train_batch(step)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss = model.forward_loss(idx_in, targets)
        raw_loss = float(loss.detach().item())
        loss.backward()
        del loss

        grad_info = _compute_subsystem_grad_rms(model)
        _step_decoupled_constant_lr_optimizers(
            model, optimizers, idx_in, targets,
            manifold_lr_mult=lr_mult, scalar_lr_mult=lr_mult, mom_beta=DEFAULT_STEADY_MOMENTUM_BETA,
        )
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        smoothed_loss = raw_loss if smoothed_loss is None else (0.15 * raw_loss + 0.85 * smoothed_loss)
        if smoothed_loss < best_smoothed_loss:
            best_smoothed_loss = smoothed_loss
            best_lr_mult = lr_mult
        if divergence_lr_mult is None and step >= 15 and (not math.isfinite(raw_loss) or smoothed_loss > best_smoothed_loss + 0.35):
            divergence_lr_mult = lr_mult

        rec = {
            "step": step,
            "lr_mult": round(lr_mult, 4),
            "stiefel_lr": round(0.032 * lr_mult, 5),
            "wte_lr": round(0.060 * lr_mult, 5),
            "scalar_lr": round(DEFAULT_SHALLOW_SCALAR_LR * lr_mult, 5),
            "train_ce": round(raw_loss, 4),
            "smoothed_ce": round(smoothed_loss, 4),
            "readout_temp": round(float(model.readout_temp.item()), 3),
            **grad_info,
        }
        records.append(rec)
        if step == 1 or step % 10 == 0 or step == steps:
            print(
                f"[LR-Ramp] Step {step:3d}/{steps} | mult={lr_mult:6.2f}x (stiefel={rec['stiefel_lr']:.4f}, "
                f"wte={rec['wte_lr']:.4f}, scalar={rec['scalar_lr']:.4f}) | Train CE: {raw_loss:.4f} "
                f"(smooth {smoothed_loss:.4f}) | temp={rec['readout_temp']:5.2f} | "
                f"grad(stf={rec['grad_rms_stiefel']:.4f}, voc={rec['grad_rms_vocab']:.5f}, scl={rec['grad_rms_scalar']:.4f})",
                flush=True,
            )
        if not math.isfinite(raw_loss) or raw_loss > DIVERGENCE_CE_THRESHOLD:
            break

    del model, optimizers
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "best_smoothed_train_ce": round(best_smoothed_loss, 4),
        "best_lr_mult": round(best_lr_mult, 4),
        "divergence_onset_lr_mult": round(divergence_lr_mult, 4) if divergence_lr_mult is not None else None,
        "records": records,
    }


def run_single_constant_lr_experiment(
    stream: LongSeqFineWebStream, cfg: ConstantLRRunConfig
) -> dict[str, Any]:
    """Run a single 300-step training trajectory with strictly constant LR (zero cosine schedule, `<= 8` args)."""
    device = stream.device
    tokens_per_step = stream.batch_size * stream.seq_len
    spec = ExplicitDensityTomographicSpec(
        variant_mode=ExplicitDensityVariantMode.T2_DENSITY_TOMOGRAPHY_PLUS_NOCLICK_AND_VACUUM,
        enable_density_highway=True, enable_tomographic_readout=True,
        enable_noclick_and_vacuum=True, enable_two_register_bilinear=False, seed=cfg.seed,
    )
    torch.manual_seed(cfg.seed)
    model = ExplicitDensityTomographicGPT(spec).to(device=device)
    init_ms, _ = initialize_explicit_density_tomographic_model_(model)
    optimizers = build_explicit_density_tomographic_optimizers(model)

    run_key = f"plan083_constlr_{cfg.run_id.lower()}_{cfg.steps}steps"
    intent = ExperimentIntentSpec(
        run_key=run_key, wave_id="W63_LR", plan_id="EXPERIMENT_PLAN_083", task_domain="fineweb_speedrun",
        hypothesis_intent=f"Constant LR stability test (no cosine schedule): manifold={cfg.manifold_lr_mult}x, scalar={cfg.scalar_lr_mult}x",
        architecture_summary=f"ExplicitDensityTomographicGPT(T2, const_manifold_lr={cfg.manifold_lr_mult}x, const_scalar_lr={cfg.scalar_lr_mult}x)",
        hardware_budget="1x RTX 5060 Ti 16GB", batch_size=stream.batch_size, seq_len=stream.seq_len,
        total_steps=cfg.steps, target_tokens_M=round((cfg.steps * tokens_per_step) / 1e6, 3),
    )
    exp_logger = StandardizedExperimentLogger(intent)

    step0_ev = evaluate_explicit_density_tomographic_model(model, stream)
    step0_ev["step"], step0_ev["tokens_M"] = 0, 0.0
    exp_logger.log_eval(0, {**step0_ev, "top1_acc": round(100.0 * step0_ev["top1_acc"], 2), "top5_acc": round(100.0 * step0_ev["top5_acc"], 2)})

    curve: list[dict[str, Any]] = [step0_ev]
    train_losses: list[float] = []
    grad_history: list[dict[str, float]] = []
    diverged = False
    diverged_step: int | None = None

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.train()

    for step in range(1, cfg.steps + 1):
        idx_in, targets = stream.sample_train_batch(step)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        loss = model.forward_loss(idx_in, targets)
        loss_val = float(loss.detach().item())
        if not math.isfinite(loss_val) or loss_val > DIVERGENCE_CE_THRESHOLD:
            diverged = True
            diverged_step = step
            print(f"[{cfg.run_id}] DIVERGED at Step {step} (Train CE={loss_val})!", flush=True)
            break

        loss.backward()
        del loss
        should_eval = (step % cfg.eval_interval == 0) or (step == cfg.steps) or (step in (100, 300, 600))
        if should_eval or step == 1:
            grad_history.append(_compute_subsystem_grad_rms(model))

        _step_decoupled_constant_lr_optimizers(
            model, optimizers, idx_in, targets,
            manifold_lr_mult=cfg.manifold_lr_mult, scalar_lr_mult=cfg.scalar_lr_mult, mom_beta=cfg.mom_beta,
        )
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        train_losses.append(loss_val)

        if should_eval:
            ev = evaluate_explicit_density_tomographic_model(model, stream)
            ev["step"], ev["tokens_M"] = step, round((step * tokens_per_step) / 1e6, 3)
            ev["train_loss"] = round(loss_val, 4)
            ev["readout_temp"] = round(float(model.readout_temp.item()), 3)
            ev["tomo_scale"] = round(float(model.tomo_scale.item()), 3)
            ev["filter_scale"] = round(float(model.filter_scale.item()), 3)
            if grad_history:
                ev.update(grad_history[-1])
            curve.append(ev)
            exp_logger.log_eval(step, {**ev, "top1_acc": round(100.0 * ev["top1_acc"], 2), "top5_acc": round(100.0 * ev["top5_acc"], 2)})
            print(
                f"[{cfg.run_id} (man={cfg.manifold_lr_mult:.1f}x, scl={cfg.scalar_lr_mult:.1f}x)] "
                f"Step {step:5d} ({ev['tokens_M']:6.2f}M) | Train CE: {loss_val:.4f} | "
                f"Val CE: {ev['val_ce']:.4f} (PPL {ev['val_ppl']:.2f}) | Top1: {100.0 * ev['top1_acc']:.2f}% | "
                f"temp={ev['readout_temp']:5.2f} | RhoRank={ev['mean_rho_eff_rank']:.2f}/16 | p_nc={ev['mean_noclick_prob']:.3f}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0

    tail_losses = train_losses[-50:] if len(train_losses) >= 50 else train_losses
    mean_tail = sum(tail_losses) / float(max(1, len(tail_losses)))
    var_tail = sum((x - mean_tail) ** 2 for x in tail_losses) / float(max(1, len(tail_losses)))
    std_tail = math.sqrt(var_tail)
    max_spike = max(tail_losses) - min(tail_losses) if tail_losses else 0.0
    best_val_ce = min(pt["val_ce"] for pt in curve[1:]) if len(curve) > 1 else step0_ev["val_ce"]

    del model, optimizers
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "run_id": cfg.run_id,
        "manifold_lr_mult": cfg.manifold_lr_mult,
        "scalar_lr_mult": cfg.scalar_lr_mult,
        "stiefel_lr": round(0.032 * cfg.manifold_lr_mult, 5),
        "wte_lr": round(0.060 * cfg.manifold_lr_mult, 5),
        "readout_lr": round(0.050 * cfg.manifold_lr_mult, 5),
        "scalar_lr": round(DEFAULT_SHALLOW_SCALAR_LR * cfg.scalar_lr_mult, 5),
        "fast_temp_lr": round(DEFAULT_FAST_TEMP_BASE_LR * cfg.scalar_lr_mult, 5),
        "init_ms": round(init_ms, 2),
        "elapsed_sec": round(elapsed_s, 2),
        "diverged": diverged,
        "diverged_step": diverged_step,
        "best_val_ce": round(best_val_ce, 4),
        "final": curve[-1],
        "tail_50step_train_ce_mean": round(mean_tail, 4),
        "tail_50step_train_ce_std": round(std_tail, 4),
        "tail_50step_max_spike_nats": round(max_spike, 4),
        "curve": curve,
    }


def main() -> None:
    """CLI entrypoint for the Constant-LR Stability & Breakdown Ceiling Sweep."""
    parser = argparse.ArgumentParser(description="Constant-LR (No Cosine Schedule) Stability & Breakdown Ceiling Sweep")
    parser.add_argument("--steps", type=int, default=DEFAULT_CONST_LR_STEPS, help="Steps per constant-LR run (default: 300)")
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_CONST_EVAL_INTERVAL, help="Eval interval (default: 50)")
    parser.add_argument("--ramp-steps", type=int, default=DEFAULT_RAMP_STEPS, help="Steps for Phase-1 exponential LR ramp")
    parser.add_argument("--skip-ramp", action="store_true", help="Skip Phase-1 exponential LR range test")
    parser.add_argument("--skip-decoupled", action="store_true", help="Skip Phase-3 decoupled manifold vs scalar probe")
    parser.add_argument("--fine-boundary", action="store_true", help="Run fine-grained boundary sweep (1.25x, 1.5x, 5.0x, 6.0x, 7.0x)")
    parser.add_argument("--mults", type=str, default="", help="Comma-separated uniform LR multipliers (e.g. 1.0,2.0,4.0,6.0,8.0)")
    parser.add_argument("--json-out", type=str, default=".cache/plan083_constant_lr_stability_sweep.json", help="Output JSON path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = LongSeqFineWebStream(seq_len=512, batch_size=16, val_tokens=65_536, device=device)
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.fine_boundary:
        fine_mults = [1.0, 1.25, 1.50, 6.0, 8.0]
        fine_runs: list[dict[str, Any]] = []
        for mult in fine_mults:
            cfg = ConstantLRRunConfig(
                run_id=f"CONST_FINE_{str(mult).replace('.', 'p')}X",
                manifold_lr_mult=mult, scalar_lr_mult=mult,
                steps=args.steps, eval_interval=args.eval_interval,
            )
            fine_runs.append(run_single_constant_lr_experiment(stream, cfg))
        if out_path.exists():
            data = json.loads(out_path.read_text(encoding="utf-8"))
        else:
            data = {"wave_id": "W63_LR", "plan_id": "EXPERIMENT_PLAN_083"}
        data["phase4_fine_boundary_sweep"] = fine_runs
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"\nAppended Phase 4 fine-boundary sweep to {out_path}", flush=True)
        return

    payload: dict[str, Any] = {
        "wave_id": "W63_LR",
        "plan_id": "EXPERIMENT_PLAN_083",
        "steps_per_run": args.steps,
        "tokens_M_per_run": round((args.steps * stream.batch_size * stream.seq_len) / 1e6, 3),
        "cosine_baseline_300step_val_ce": 6.3893,
        "cosine_baseline_300step_top1_pct": 16.31,
        "base_lrs": {
            "stiefel_lr": 0.032, "unitary_lr": 0.025, "wte_lr": 0.060, "readout_lr": 0.050,
            "scalar_lr": DEFAULT_SHALLOW_SCALAR_LR, "fast_temp_lr": DEFAULT_FAST_TEMP_BASE_LR,
            "ngram_lr": DEFAULT_SHALLOW_NGRAM_LR,
        },
    }

    if not args.skip_ramp:
        payload["phase1_exponential_ramp"] = run_exponential_lr_range_finder(stream, steps=args.ramp_steps)

    print("\n=== Phase 2: Constant-LR Grid Without Cosine Schedule (Uniform Multipliers) ===", flush=True)
    if args.mults.strip():
        uniform_mults = [float(x.strip()) for x in args.mults.split(",") if x.strip()]
    else:
        uniform_mults = [1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 16.0, 32.0]
    phase2_runs: list[dict[str, Any]] = []
    for mult in uniform_mults:
        cfg = ConstantLRRunConfig(
            run_id=f"CONST_UNIFORM_{str(mult).replace('.', 'p')}X",
            manifold_lr_mult=mult, scalar_lr_mult=mult,
            steps=args.steps, eval_interval=args.eval_interval,
        )
        res = run_single_constant_lr_experiment(stream, cfg)
        phase2_runs.append(res)
        payload["phase2_uniform_constant_lr_grid"] = phase2_runs
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not args.skip_decoupled:
        print("\n=== Phase 3: Decoupled Manifold High-LR vs. Scalar 2.0x Constant-LR Probe ===", flush=True)
        decoupled_configs = [
            ConstantLRRunConfig("CONST_DECOUPLED_MAN8X_SCL2X", manifold_lr_mult=8.0, scalar_lr_mult=2.0, steps=args.steps, eval_interval=args.eval_interval),
            ConstantLRRunConfig("CONST_DECOUPLED_MAN16X_SCL2X", manifold_lr_mult=16.0, scalar_lr_mult=2.0, steps=args.steps, eval_interval=args.eval_interval),
            ConstantLRRunConfig("CONST_DECOUPLED_MAN32X_SCL2X", manifold_lr_mult=32.0, scalar_lr_mult=2.0, steps=args.steps, eval_interval=args.eval_interval),
        ]
        phase3_runs: list[dict[str, Any]] = []
        for cfg in decoupled_configs:
            res = run_single_constant_lr_experiment(stream, cfg)
            phase3_runs.append(res)
        payload["phase3_decoupled_manifold_probe"] = phase3_runs

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved Constant-LR Stability Sweep results to {out_path}", flush=True)


if __name__ == "__main__":
    main()

