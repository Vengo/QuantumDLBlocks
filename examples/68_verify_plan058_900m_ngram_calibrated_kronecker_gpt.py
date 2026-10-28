"""Verification script for Wave 43 (EXPERIMENT_PLAN_058.md): 900M-Token Step-0 Unigram, Bigram, and Trigram Hilbert-Space Readout Calibration on Official Modded-NanoGPT FineWeb.

Executes Hypotheses H1-H4:
- H1: 5-way Step-0 900M-token N-gram readout calibration audit (`val_ce_step0_65k < 7.35`, `top1_acc >= 9.0%`).
- H2: 150-step controlled ablation (`unigram_1shard_w42` vs. `bigram_900m_9shard` vs. `trigram_900m_9shard`).
- H3: 150-step multi-shard rare-token importance sampling vs. uniform sampling across 9 shards (`900M` tokens).
- H4: 600-second flagship 9-shard training run (`w43_flagship_900m_ngram_calibrated_kronecker_r2_d128c_L6_600s`)
      followed by exact zero-materialization evaluation on the official `10,485,760`-token `modded-nanogpt`
      validation benchmark (`fineweb_val_000000.bin`).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qdlblocks.benchmarks.experiment_logger import (
    ExperimentIntentSpec,
    StandardizedExperimentLogger,
)
from qdlblocks.models.kronecker_multimeaning_16x4_gpt import _load_header_safe_shard_tokens
from qdlblocks.models.ngram_calibrated_kronecker_born_gpt import (
    DEFAULT_MODDED_VAL_TOKENS,
    MultiShardRareTokenSampler,
    Ngram900MCalibrationData,
    NgramCalibratedKroneckerBornGPT,
    NgramCalibratedKroneckerConfig,
    build_ngram_calibrated_kronecker_optimizers,
    build_or_load_900m_ngram_calibration_cache,
    compute_ngram_calibrated_kronecker_penalties,
    evaluate_modded_nanogpt_validation,
)
from scripts.calibrate_900m_ngram_kronecker_readout import run_step0_calibration_audit

# ============================================================================
# Top-of-File Named Constants (Rule 7)
# ============================================================================
WAVE_ID: str = "W43"
PLAN_ID: str = "EXPERIMENT_PLAN_058"
DEFAULT_DATA_DIR: str = "/home/festivity/agy/QuantumDLBlocks/.cache/fineweb_100M"
DEFAULT_CACHE_PATH: str = "/home/festivity/agy/QuantumDLBlocks/.cache/fineweb_900M_ngram_calibration.pt"
DEFAULT_AUDIT_JSON: str = "/home/festivity/agy/QuantumDLBlocks/.cache/calibrated_900m_ngram_readout_audit.json"
DEFAULT_OUT_JSON: str = "/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_058.json"

DEFAULT_NUM_SHARDS: int = 9
DEFAULT_SHARD_LOAD_TOKENS: int = 25_000_000
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_SEQ_LEN: int = 512
DEFAULT_NUM_LAYERS: int = 6
DEFAULT_ABLATION_STEPS: int = 150
DEFAULT_EVAL_TOKENS_FAST: int = 65_536
DEFAULT_FLAGSHIP_WALL_S: float = 600.0
DEFAULT_WSD_WARMUP_FRAC: float = 0.05
DEFAULT_WSD_COOLDOWN_FRAC: float = 0.15
DEFAULT_MIN_LR_RATIO: float = 0.10
DEFAULT_GRAD_CLIP_NORM: float = 1.0
DEFAULT_LOG_INTERVAL: int = 25
DEFAULT_EVAL_INTERVAL: int = 50


@dataclass(frozen=True)
class TrainingRunSpec:
    """Configuration for a standardized Wave 43 training run."""

    run_key: str
    hypothesis_intent: str
    ngram_mode: str
    num_shards: int
    alpha_rare: float
    max_steps: int
    max_wall_s: float
    eval_10m_at_end: bool = False
    num_meanings: int = 2


def _compute_wsd_lr_multiplier(progress: float) -> float:
    """Compute Warmup-Stable-Decay (WSD) learning-rate multiplier for `progress in [0.0, 1.0]`."""
    p = min(max(float(progress), 0.0), 1.0)
    if p < DEFAULT_WSD_WARMUP_FRAC:
        return max(DEFAULT_MIN_LR_RATIO, p / max(1e-6, DEFAULT_WSD_WARMUP_FRAC))
    cooldown_start = 1.0 - DEFAULT_WSD_COOLDOWN_FRAC
    if p <= cooldown_start:
        return 1.0
    cool_frac = (p - cooldown_start) / max(1e-6, DEFAULT_WSD_COOLDOWN_FRAC)
    return 1.0 - (1.0 - DEFAULT_MIN_LR_RATIO) * cool_frac


def _apply_lr_multiplier(
    opt_muon: torch.optim.Optimizer,
    opt_adam: torch.optim.Optimizer,
    lr_mult: float,
) -> None:
    """Scale `lr` from `base_lr` across both `StiefelMuon` and `AdamW` parameter groups."""
    for opt in (opt_muon, opt_adam):
        for group in opt.param_groups:
            base_lr = float(group.get("base_lr", group["lr"]))
            group["lr"] = base_lr * float(lr_mult)


def _load_shard_arrays(data_dir: Path, num_shards: int, max_tokens_per_shard: int) -> list[np.ndarray]:
    """Load `num_shards` training shard arrays into pinned/pageable host RAM for fast window sampling."""
    arrays: list[np.ndarray] = []
    for idx in range(1, num_shards + 1):
        sp = data_dir / f"fineweb_train_{idx:06d}.bin"
        arrays.append(_load_header_safe_shard_tokens(sp, max_tokens=max_tokens_per_shard))
    return arrays


def execute_training_run(
    spec: TrainingRunSpec,
    calib_data: Ngram900MCalibrationData,
    cal_batches: list[torch.Tensor],
    shard_arrays: list[np.ndarray],
    val_shard_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Execute a calibrated Kronecker training run with `StandardizedExperimentLogger` (`<= 8` args)."""
    existing_res_path = Path(".cache/live_logs") / f"{spec.run_key}_results.json"
    if existing_res_path.exists():
        saved = json.loads(existing_res_path.read_text(encoding="utf-8"))
        if saved.get("completed") and "final_val_ce_65k" in saved:
            if not spec.eval_10m_at_end or saved.get("official_10m_validation") is not None:
                return saved

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.manual_seed(1337)
    cfg = NgramCalibratedKroneckerConfig(
        num_meanings=spec.num_meanings,
        num_layers=DEFAULT_NUM_LAYERS,
        block_size=DEFAULT_SEQ_LEN,
        ngram_mode=spec.ngram_mode,
        vocab_chunk_size=1024 if spec.num_meanings > 2 else 4096,
    )
    model = NgramCalibratedKroneckerBornGPT(cfg).to(device)
    calib_stats = model.calibrate_from_900m_ngram_cache(
        cache=calib_data,
        cal_batches=cal_batches,
        ngram_mode=spec.ngram_mode,
        carrier_rho_max=cfg.carrier_rho_max,
        calibrate_povm=True,
    )

    opt_muon, opt_adam = build_ngram_calibrated_kronecker_optimizers(model)
    active_shards = shard_arrays[: max(1, min(spec.num_shards, len(shard_arrays)))]
    sampler = MultiShardRareTokenSampler(
        shard_arrays=active_shards,
        unigram_counts=calib_data.unigram_counts,
        vocab_size=cfg.vocab_size,
        valid_vocab_size=cfg.valid_vocab_size,
        alpha_rare=spec.alpha_rare,
    )

    tokens_per_step = DEFAULT_BATCH_SIZE * DEFAULT_SEQ_LEN
    target_tokens_m = (spec.max_steps * tokens_per_step) / 1e6
    num_params = sum(p.numel() for p in model.parameters())

    intent = ExperimentIntentSpec(
        run_key=spec.run_key,
        wave_id=WAVE_ID,
        plan_id=PLAN_ID,
        task_domain="fineweb_speedrun",
        hypothesis_intent=spec.hypothesis_intent,
        architecture_summary=(
            f"NgramCalibratedKroneckerBornGPT(R={cfg.num_meanings}, M={cfg.num_factors}, d_f={cfg.factor_dim}, "
            f"L={cfg.num_layers}, mode={spec.ngram_mode}, shards={spec.num_shards}, params={num_params / 1e6:.2f}M)"
        ),
        hardware_budget="1x NVIDIA GeForce RTX 5060 Ti (16 GB VRAM), BF16 + Complex Register Triton",
        batch_size=DEFAULT_BATCH_SIZE,
        seq_len=DEFAULT_SEQ_LEN,
        total_steps=spec.max_steps,
        target_tokens_M=round(target_tokens_m, 3),
    )
    logger = StandardizedExperimentLogger(intent)

    # Step-0 Evaluation (65K tokens)
    eval0 = evaluate_modded_nanogpt_validation(
        model=model,
        val_tokens=val_shard_path,
        max_eval_tokens=DEFAULT_EVAL_TOKENS_FAST,
        batch_size=DEFAULT_BATCH_SIZE,
        seq_len=DEFAULT_SEQ_LEN,
    )
    logger.log_eval(
        0,
        {
            "tokens_M": 0.0,
            "compute_tflops": 0.0,
            "tau_progress": 0.0,
            "val_ce": eval0["val_ce"],
            "val_ppl": eval0["val_ppl"],
            "top1_acc": eval0["top1_acc"],
            "sharp_eff_rank": eval0["sharp_eff_rank"],
            "kraus_stiefel_err": calib_stats["unitary_frobenius_err"],
        },
    )

    t_train_start = time.perf_counter()
    step = 0
    best_val_ce = eval0["val_ce"]
    best_top1_acc = eval0["top1_acc"]
    last_ce_loss = eval0["val_ce"]
    rare_tokens_sampled = 0
    total_tokens_sampled = 0

    # Determine top-50% frequency threshold for rare-token tracking
    sorted_counts, _ = torch.sort(calib_data.unigram_counts, descending=True)
    median_count = float(sorted_counts[min(2000, len(sorted_counts) - 1)].item())
    rare_mask_vocab = (calib_data.unigram_counts <= median_count).to(device=device)

    model.train()
    while step < spec.max_steps:
        elapsed_s = time.perf_counter() - t_train_start
        if spec.max_wall_s > 0.0 and elapsed_s >= spec.max_wall_s:
            break

        step += 1
        if spec.max_wall_s > 0.0:
            progress = min(1.0, max(step / float(spec.max_steps), elapsed_s / float(spec.max_wall_s)))
        else:
            progress = min(1.0, step / float(spec.max_steps))

        model.set_training_progress(progress)
        lr_mult = _compute_wsd_lr_multiplier(progress)
        _apply_lr_multiplier(opt_muon, opt_adam, lr_mult)

        x_b, y_b, w_b = sampler.sample_batch(
            batch_size=DEFAULT_BATCH_SIZE,
            seq_len=DEFAULT_SEQ_LEN,
            step_seed=step,
            device=device,
        )
        rare_tokens_sampled += int(rare_mask_vocab[y_b].sum().item())
        total_tokens_sampled += int(y_b.numel())

        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, ce_loss = model(x_b, y_b, token_weights=w_b)
        assert ce_loss is not None
        pen_loss, pen_stats = compute_ngram_calibrated_kronecker_penalties(model)
        total_loss = ce_loss + pen_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_GRAD_CLIP_NORM)
        opt_muon.step()
        opt_adam.step()

        last_ce_loss = float(ce_loss.detach().item())
        tokens_m = (step * tokens_per_step) / 1e6
        compute_tflops = (6.0 * num_params * step * tokens_per_step) / 1e12
        wall_now = max(1e-3, time.perf_counter() - t_train_start)
        tok_per_s = (step * tokens_per_step) / wall_now

        if step % DEFAULT_LOG_INTERVAL == 0 or step == 1:
            logger.log_step(
                step=step,
                train_ce=round(last_ce_loss, 4),
                tok_per_s=round(tok_per_s, 1),
                lr_mult=round(lr_mult, 4),
                elapsed_min=round(wall_now / 60.0, 4),
            )

        if step % DEFAULT_EVAL_INTERVAL == 0:
            ev = evaluate_modded_nanogpt_validation(
                model=model,
                val_tokens=val_shard_path,
                max_eval_tokens=DEFAULT_EVAL_TOKENS_FAST,
                batch_size=DEFAULT_BATCH_SIZE,
                seq_len=DEFAULT_SEQ_LEN,
            )
            best_val_ce = min(best_val_ce, ev["val_ce"])
            best_top1_acc = max(best_top1_acc, ev["top1_acc"])
            logger.log_eval(
                step,
                {
                    "tokens_M": round(tokens_m, 4),
                    "compute_tflops": round(compute_tflops, 4),
                    "tau_progress": round(progress, 4),
                    "val_ce": ev["val_ce"],
                    "val_ppl": ev["val_ppl"],
                    "top1_acc": ev["top1_acc"],
                    "sharp_eff_rank": ev["sharp_eff_rank"],
                    "kraus_stiefel_err": round(pen_stats["unitary_pen"], 6),
                },
            )

    train_wall_s = time.perf_counter() - t_train_start
    final_tokens_m = (step * tokens_per_step) / 1e6
    final_tflops = (6.0 * num_params * step * tokens_per_step) / 1e12
    avg_tok_per_s = (step * tokens_per_step) / max(1e-3, train_wall_s)

    model.zero_grad(set_to_none=True)
    del opt_muon, opt_adam
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_final_65k = evaluate_modded_nanogpt_validation(
        model=model,
        val_tokens=val_shard_path,
        max_eval_tokens=DEFAULT_EVAL_TOKENS_FAST,
        batch_size=DEFAULT_BATCH_SIZE,
        seq_len=DEFAULT_SEQ_LEN,
        compute_top50=True,
    )
    best_val_ce = min(best_val_ce, eval_final_65k["val_ce"])
    best_top1_acc = max(best_top1_acc, eval_final_65k["top1_acc"])
    unitary_err_final = model.compute_all_unitary_frobenius_error()

    logger.log_eval(
        step,
        {
            "tokens_M": round(final_tokens_m, 4),
            "compute_tflops": round(final_tflops, 4),
            "tau_progress": 1.0,
            "val_ce": eval_final_65k["val_ce"],
            "val_ppl": eval_final_65k["val_ppl"],
            "top1_acc": eval_final_65k["top1_acc"],
            "sharp_eff_rank": eval_final_65k["sharp_eff_rank"],
            "kraus_stiefel_err": round(unitary_err_final, 6),
        },
    )

    eval_10m: dict[str, float] | None = None
    if spec.eval_10m_at_end:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t_10m = time.perf_counter()
        eval_10m = evaluate_modded_nanogpt_validation(
            model=model,
            val_tokens=val_shard_path,
            max_eval_tokens=DEFAULT_MODDED_VAL_TOKENS,
            batch_size=32 if spec.num_meanings <= 2 else DEFAULT_BATCH_SIZE,
            seq_len=DEFAULT_SEQ_LEN,
            compute_top50=False,
        )
        eval_10m["eval_wall_s"] = round(time.perf_counter() - t_10m, 2)

    out = {
        "run_key": spec.run_key,
        "ngram_mode": spec.ngram_mode,
        "num_shards": spec.num_shards,
        "alpha_rare": spec.alpha_rare,
        "steps_completed": step,
        "tokens_M": round(final_tokens_m, 4),
        "train_wall_s": round(train_wall_s, 2),
        "avg_tok_per_s": round(avg_tok_per_s, 1),
        "num_params": num_params,
        "step0_val_ce_65k": eval0["val_ce"],
        "step0_val_ppl_65k": eval0["val_ppl"],
        "step0_top1_acc_65k": eval0["top1_acc"],
        "final_val_ce_65k": eval_final_65k["val_ce"],
        "final_val_ppl_65k": eval_final_65k["val_ppl"],
        "final_top1_acc_65k": eval_final_65k["top1_acc"],
        "final_top50_acc_65k": eval_final_65k.get("top50_acc", 0.0),
        "best_val_ce_65k": best_val_ce,
        "best_top1_acc_65k": best_top1_acc,
        "sharp_eff_rank": eval_final_65k["sharp_eff_rank"],
        "mean_factor_rank": eval_final_65k["mean_factor_rank"],
        "cross_branch_overlap": eval_final_65k["cross_branch_overlap"],
        "povm_drift_std": eval_final_65k["povm_drift_std"],
        "unitary_frobenius_err": round(unitary_err_final, 6),
        "rare_token_fraction": round(rare_tokens_sampled / max(1, total_tokens_sampled), 4),
        "official_10m_validation": eval_10m,
    }
    logger.finalize(extra_summary=out)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main() -> None:
    """Execute Wave 43 Hypotheses H1-H4 and save `.cache/hypothesis_experiments_058.json`."""
    parser = argparse.ArgumentParser(description="Verify Wave 43 (EXPERIMENT_PLAN_058) Hypotheses H1-H4")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-path", type=str, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--ablation-steps", type=int, default=DEFAULT_ABLATION_STEPS)
    parser.add_argument("--flagship-wall-s", type=float, default=DEFAULT_FLAGSHIP_WALL_S)
    parser.add_argument("--run-tournament", action="store_true", help="Run extended R=2/R=4 & 1200-step tournament")
    parser.add_argument("--out-json", type=str, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    cache_path = Path(args.cache_path)
    val_shard_path = data_dir / "fineweb_val_000000.bin"
    shard_paths = [data_dir / f"fineweb_train_{idx:06d}.bin" for idx in range(1, DEFAULT_NUM_SHARDS + 1)]

    print("=== [H1] Running Step-0 900M N-Gram Readout Calibration Audit ===")
    h1_summary = run_step0_calibration_audit(
        data_dir=data_dir,
        cache_path=cache_path,
        num_shards=DEFAULT_NUM_SHARDS,
        eval_tokens=DEFAULT_EVAL_TOKENS_FAST,
        seq_len=256,
        batch_size=8,
        force_rebuild=False,
    )
    with open(DEFAULT_AUDIT_JSON, "w", encoding="utf-8") as f:
        json.dump(h1_summary, f, indent=2)

    base_cfg = NgramCalibratedKroneckerConfig(num_layers=DEFAULT_NUM_LAYERS, block_size=DEFAULT_SEQ_LEN)
    calib_data = build_or_load_900m_ngram_calibration_cache(
        shard_paths=shard_paths,
        cache_path=cache_path,
        cfg=base_cfg,
        device=device,
        force_rebuild=False,
    )

    print("=== Loading 9 FineWeb Shard Windows for Multi-Shard Training ===")
    shard_arrays = _load_shard_arrays(
        data_dir=data_dir,
        num_shards=DEFAULT_NUM_SHARDS,
        max_tokens_per_shard=DEFAULT_SHARD_LOAD_TOKENS,
    )
    cal_batches = [
        torch.from_numpy(shard_arrays[0][b * 8 * DEFAULT_SEQ_LEN : (b + 1) * 8 * DEFAULT_SEQ_LEN].astype(np.int64)).view(
            8, DEFAULT_SEQ_LEN
        )
        for b in range(4)
    ]

    print(f"=== [H2] Running {args.ablation_steps}-Step N-Gram Readout Ablation (Unigram vs Bigram vs Trigram) ===")
    h2_specs = [
        TrainingRunSpec(
            run_key="w43_h2_unigram_1shard_w42",
            hypothesis_intent="H2 baseline: 1-shard unigram-only Step-0 calibration + 1-shard sampling",
            ngram_mode="none",
            num_shards=1,
            alpha_rare=0.25,
            max_steps=args.ablation_steps,
            max_wall_s=0.0,
        ),
        TrainingRunSpec(
            run_key="w43_h2_bigram_900m_9shard",
            hypothesis_intent="H2 ablation: 900M unigram+bigram Step-0 calibration + 9-shard sampling",
            ngram_mode="bigram",
            num_shards=9,
            alpha_rare=0.25,
            max_steps=args.ablation_steps,
            max_wall_s=0.0,
        ),
        TrainingRunSpec(
            run_key="w43_h2_trigram_900m_9shard",
            hypothesis_intent="H2 full: 900M unigram+bigram+trigram Step-0 calibration + 9-shard sampling",
            ngram_mode="trigram",
            num_shards=9,
            alpha_rare=0.25,
            max_steps=args.ablation_steps,
            max_wall_s=0.0,
        ),
    ]
    h2_results: dict[str, Any] = {}
    for s in h2_specs:
        res = execute_training_run(s, calib_data, cal_batches, shard_arrays, val_shard_path, device)
        h2_results[s.run_key] = res
        print(
            f"  {s.run_key:32s} | Step0 CE: {res['step0_val_ce_65k']:.4f} -> "
            f"Final CE: {res['final_val_ce_65k']:.4f} | Top-1: {res['final_top1_acc_65k']:.2f}% | "
            f"R_emb: {res['sharp_eff_rank']:.1f} | {res['avg_tok_per_s']:.0f} tok/s"
        )

    print(f"=== [H3] Running {args.ablation_steps}-Step Multi-Shard Sampler Comparison ===")
    h3_uniform_spec = TrainingRunSpec(
        run_key="w43_h3_uniform_9shard",
        hypothesis_intent="H3 baseline: uniform random sampling (alpha_rare=0.0) across 9 FineWeb shards",
        ngram_mode="trigram",
        num_shards=9,
        alpha_rare=0.0,
        max_steps=args.ablation_steps,
        max_wall_s=0.0,
    )
    h3_uniform_res = execute_training_run(
        h3_uniform_spec, calib_data, cal_batches, shard_arrays, val_shard_path, device
    )
    h3_rare_res = h2_results["w43_h2_trigram_900m_9shard"]
    h3_results = {
        "w43_h3_uniform_9shard": h3_uniform_res,
        "w43_h3_rare_weighted_9shard": h3_rare_res,
        "rare_fraction_boost_ratio": round(
            h3_rare_res["rare_token_fraction"] / max(1e-6, h3_uniform_res["rare_token_fraction"]), 3
        ),
    }
    print(
        f"  Uniform Rare Frac: {h3_uniform_res['rare_token_fraction']:.4f} (CE={h3_uniform_res['final_val_ce_65k']:.4f}) vs "
        f"Importance Rare Frac: {h3_rare_res['rare_token_fraction']:.4f} (CE={h3_rare_res['final_val_ce_65k']:.4f})"
    )

    print(f"=== [H4] Running {args.flagship_wall_s:.0f}s Flagship 9-Shard Run + Official 10.49M-Token Evaluation ===")
    est_tok_s = max(4000.0, float(h3_rare_res["avg_tok_per_s"]))
    est_steps_600s = max(200, int((args.flagship_wall_s * est_tok_s) / (DEFAULT_BATCH_SIZE * DEFAULT_SEQ_LEN)))
    flagship_spec = TrainingRunSpec(
        run_key="w43_flagship_900m_ngram_calibrated_kronecker_r2_d128c_L6_600s",
        hypothesis_intent=(
            "H4 Flagship: 600s 9-shard (900M-token) N-gram calibrated Kronecker Born GPT evaluated on "
            "official 10,485,760-token modded-nanogpt FineWeb validation shard"
        ),
        ngram_mode="trigram",
        num_shards=9,
        alpha_rare=0.25,
        max_steps=est_steps_600s,
        max_wall_s=args.flagship_wall_s,
        eval_10m_at_end=True,
    )
    h4_flagship_res = execute_training_run(
        flagship_spec, calib_data, cal_batches, shard_arrays, val_shard_path, device
    )
    print(
        f"  Flagship 65K Eval  | Step0 CE: {h4_flagship_res['step0_val_ce_65k']:.4f} -> "
        f"Final CE: {h4_flagship_res['final_val_ce_65k']:.4f} | Top-1: {h4_flagship_res['final_top1_acc_65k']:.2f}%"
    )
    if h4_flagship_res["official_10m_validation"] is not None:
        ev10 = h4_flagship_res["official_10m_validation"]
        print(
            f"  Official 10.49M Val | Val CE: {ev10['val_ce']:.4f} | PPL: {ev10['val_ppl']:.2f} | "
            f"Top-1: {ev10['top1_acc']:.2f}% | Top-50: {ev10.get('top50_acc', 0.0):.2f}% | "
            f"Wall: {ev10['eval_wall_s']:.1f}s"
        )

    tournament_results: dict[str, Any] = {}
    if args.run_tournament:
        print("=== [Extended Tournament] Running Uniform 600s (R=2 & R=4) + Deep 1,200-Step (9.83M Tok) Models ===")
        tour_specs = [
            TrainingRunSpec(
                run_key="w43_tour_r2_uniform_9shard_600s",
                hypothesis_intent="Extended Tournament: R=2 uniform 9-shard (alpha_rare=0.0) 600s + 10.49M eval",
                ngram_mode="trigram",
                num_shards=9,
                alpha_rare=0.0,
                max_steps=400,
                max_wall_s=600.0,
                eval_10m_at_end=True,
                num_meanings=2,
            ),
            TrainingRunSpec(
                run_key="w43_tour_r4_polysemy_uniform_9shard_600s",
                hypothesis_intent="Extended Tournament: R=4 4-meaning polysemy uniform 9-shard 600s + 10.49M eval",
                ngram_mode="trigram",
                num_shards=9,
                alpha_rare=0.0,
                max_steps=320,
                max_wall_s=600.0,
                eval_10m_at_end=True,
                num_meanings=4,
            ),
            TrainingRunSpec(
                run_key="w43_tour_r2_deep_1200steps_9p83Mtok",
                hypothesis_intent=(
                    "Extended Tournament Deep Flagship: R=2 uniform 9-shard 1,200 steps (9.83M tokens, ~1800s) "
                    "+ official 10.49M modded-nanogpt validation"
                ),
                ngram_mode="trigram",
                num_shards=9,
                alpha_rare=0.0,
                max_steps=1200,
                max_wall_s=1900.0,
                eval_10m_at_end=True,
                num_meanings=2,
            ),
        ]
        for ts in tour_specs:
            t_res = execute_training_run(ts, calib_data, cal_batches, shard_arrays, val_shard_path, device)
            tournament_results[ts.run_key] = t_res
            ev10_t = t_res.get("official_10m_validation") or {}
            print(
                f"  {ts.run_key:42s} | 65K CE: {t_res['final_val_ce_65k']:.4f} (Top1={t_res['final_top1_acc_65k']:.2f}%) | "
                f"10.49M CE: {ev10_t.get('val_ce', 0.0):.4f} (Top1={ev10_t.get('top1_acc', 0.0):.2f}%)",
                flush=True,
            )

    full_report = {
        "wave_id": WAVE_ID,
        "plan_id": PLAN_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "h1_step0_calibration_audit": h1_summary,
        "h2_150step_ngram_ablation": h2_results,
        "h3_multishard_sampler_ablation": h3_results,
        "h4_flagship_600s_and_10m_eval": h4_flagship_res,
        "extended_tournament": tournament_results,
        "overall_pass": bool(
            h1_summary["h1_target_ce_below_7_35_met"]
            and h1_summary["h1_target_gain_above_0_55_met"]
            and h1_summary["h1_target_top1_above_9pct_met"]
            and (h2_results["w43_h2_trigram_900m_9shard"]["final_val_ce_65k"] < h2_results["w43_h2_unigram_1shard_w42"]["final_val_ce_65k"])
        ),
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"Saved complete Wave 43 verification report to: {out_path}")


if __name__ == "__main__":
    main()
