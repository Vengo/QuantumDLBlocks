#!/usr/bin/env python3
"""Automated Hyperparameter Search & Step-0 Readout Calibration Profiler (`Wave 42` / `EXPERIMENT_PLAN_057.md`).

First-class CLI tool (`Section 5.2` of `EXPERIMENT_PLAN_057.md`) that:
1. Audits Step-0 random-init attention output calibration across `uncalibrated_haar`,
   `carrier_angle_only`, `trainable_povm_only`, and `hybrid_procrustes_carrier_povm` with
   zero static `prior_weights` buffer.
2. Runs a multi-configuration hyperparameter search over `CalibratedKroneckerBornGPT` varying
   `(num_meanings, boson_p, kron_blend, coinc_gamma, carrier_rho_max, unigram_temp, muon_lr, embed_lr, povm_lr)`,
   streaming each trial via `StandardizedExperimentLogger` to `.cache/live_logs/`, `.cache/model_cards/`,
   and `.cache/benchmarks.db`.
3. Exports a ranked Pareto leaderboard and winning configuration (`best_trial`) to JSON.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
import urllib.request

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p_str in (str(ROOT / "src"), str(ROOT)):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from qdlblocks.benchmarks.experiment_logger import (  # noqa: E402
    ExperimentIntentSpec,
    StandardizedExperimentLogger,
)
from qdlblocks.models.calibrated_kronecker_born_gpt import (  # noqa: E402
    CalibratedKroneckerBornGPT,
    CalibratedKroneckerConfig,
    DEFAULT_BRANCH_ORTH_WEIGHT,
    DEFAULT_CALIBRATION_BATCHES,
    DEFAULT_CALIBRATION_SEQ_LEN,
    DEFAULT_CAL_HOLO_ALPHA,
    DEFAULT_FACTOR_DIM,
    DEFAULT_GATE_LR,
    DEFAULT_ISO_PENALTY_WEIGHT,
    DEFAULT_NUM_FACTORS,
    DEFAULT_POVM_ANCHOR_WEIGHT,
    DEFAULT_SPHERE_PENALTY_WEIGHT,
    DEFAULT_UNITARY_PENALTY_WEIGHT,
    build_calibrated_kronecker_optimizers,
    compute_calibrated_kronecker_penalties,
)
from qdlblocks.models.kronecker_multimeaning_16x4_gpt import (  # noqa: E402
    _load_header_safe_shard_tokens,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_057.md (Wave 42 Constants Ledger)
# ==============================================================================
DATA_DIR: Path = ROOT / ".cache" / "fineweb_100M"
DEFAULT_SEARCH_JSON: Path = ROOT / ".cache" / "kronecker_born_hyperparam_search.json"
WEBUI_RELOAD_URL: str = "http://127.0.0.1:8765/api/reload"

VOCAB_SIZE_PADDED: int = 50304
VALID_VOCAB_SIZE: int = 50257
SLICE_30M_TOKENS: int = 30_000_000
SLICE_100M_TOKENS: int = 100_000_000
VAL_SLICE_TOKENS: int = 1_048_576

DEFAULT_SEARCH_STEPS: int = 120
DEFAULT_SEARCH_BATCH_SIZE: int = 8
DEFAULT_SEARCH_SEQ_LEN: int = 128
DEFAULT_SEARCH_LAYERS: int = 4
DEFAULT_SEARCH_VAL_BATCHES: int = 6
DEFAULT_SEARCH_VAL_BATCH_SIZE: int = 4
MIN_NATS_IMPROVEMENT_TARGET: float = 0.10
SEARCH_WAVE_ID: str = "W42"
SEARCH_PLAN_ID: str = "EXPERIMENT_PLAN_057"


@dataclass(frozen=True)
class HyperparamTrialSpec:
    """Candidate hyperparameter configuration for `CalibratedKroneckerBornGPT`."""

    trial_id: str
    num_meanings: int = 2
    boson_p: float = 11.0
    kron_blend: float = 0.55
    coinc_gamma: float = 0.45
    carrier_rho_max: float = 0.22
    unigram_temp: float = 0.85
    muon_lr: float = 0.024
    embed_lr: float = 0.015
    povm_lr: float = 0.008
    gate_lr: float = DEFAULT_GATE_LR
    is_baseline: bool = False


@dataclass(frozen=True)
class SearchRunnerConfig:
    """Execution parameters for the hyperparameter search runner (`<= 8` fields)."""

    steps_per_trial: int = DEFAULT_SEARCH_STEPS
    batch_size: int = DEFAULT_SEARCH_BATCH_SIZE
    seq_len: int = DEFAULT_SEARCH_SEQ_LEN
    num_layers: int = DEFAULT_SEARCH_LAYERS
    vocab_size: int = VOCAB_SIZE_PADDED
    valid_vocab_size: int = VALID_VOCAB_SIZE
    val_batches: int = DEFAULT_SEARCH_VAL_BATCHES
    log_to_webui: bool = True


def get_default_candidate_trials() -> tuple[HyperparamTrialSpec, ...]:
    """Return the 7 candidate trials spanning `(R, p, eta_kron, gamma_c, rho_max, tau_uni, lrs)`."""
    return (
        HyperparamTrialSpec(
            trial_id="trial_00_w41_default_baseline",
            num_meanings=2,
            boson_p=11.0,
            kron_blend=0.55,
            coinc_gamma=0.45,
            carrier_rho_max=0.0,
            unigram_temp=0.80,
            muon_lr=0.015,
            embed_lr=0.008,
            povm_lr=0.004,
            is_baseline=True,
        ),
        HyperparamTrialSpec(
            trial_id="trial_01_carrier0p22_mid_lr_tau0p85",
            num_meanings=2,
            boson_p=11.0,
            kron_blend=0.55,
            coinc_gamma=0.50,
            carrier_rho_max=0.22,
            unigram_temp=0.85,
            muon_lr=0.024,
            embed_lr=0.015,
            povm_lr=0.006,
        ),
        HyperparamTrialSpec(
            trial_id="trial_02_sharp_p13p5_coinc0p75_tau0p90",
            num_meanings=2,
            boson_p=13.5,
            kron_blend=0.70,
            coinc_gamma=0.75,
            carrier_rho_max=0.22,
            unigram_temp=0.90,
            muon_lr=0.024,
            embed_lr=0.015,
            povm_lr=0.006,
        ),
        HyperparamTrialSpec(
            trial_id="trial_03_highway_p8p0_fast_lr_tau0p90",
            num_meanings=2,
            boson_p=8.0,
            kron_blend=0.40,
            coinc_gamma=0.35,
            carrier_rho_max=0.22,
            unigram_temp=0.90,
            muon_lr=0.030,
            embed_lr=0.020,
            povm_lr=0.006,
        ),
        HyperparamTrialSpec(
            trial_id="trial_04_soft_p6p5_eta0p35_tau0p92",
            num_meanings=2,
            boson_p=6.5,
            kron_blend=0.35,
            coinc_gamma=0.30,
            carrier_rho_max=0.22,
            unigram_temp=0.92,
            muon_lr=0.030,
            embed_lr=0.022,
            povm_lr=0.006,
        ),
        HyperparamTrialSpec(
            trial_id="trial_05_povm_only_p8p0_fast_lr_tau0p90",
            num_meanings=2,
            boson_p=8.0,
            kron_blend=0.40,
            coinc_gamma=0.40,
            carrier_rho_max=0.0,
            unigram_temp=0.90,
            muon_lr=0.030,
            embed_lr=0.020,
            povm_lr=0.006,
        ),
        HyperparamTrialSpec(
            trial_id="trial_06_rank4_polysemy_p8p0_tau0p92",
            num_meanings=4,
            boson_p=8.0,
            kron_blend=0.40,
            coinc_gamma=0.35,
            carrier_rho_max=0.22,
            unigram_temp=0.92,
            muon_lr=0.028,
            embed_lr=0.020,
            povm_lr=0.006,
        ),
    )


def get_stage2_fine_grid_trials() -> tuple[HyperparamTrialSpec, ...]:
    """Return the 7 Stage-2 fine-grid trials refining the winning `p=7.2..8.0, eta=0.36..0.45, tau=0.90..0.94` basin."""
    return (
        HyperparamTrialSpec(
            trial_id="s2_t00_w41_default_baseline",
            num_meanings=2,
            boson_p=11.0,
            kron_blend=0.55,
            coinc_gamma=0.45,
            carrier_rho_max=0.0,
            unigram_temp=0.80,
            muon_lr=0.015,
            embed_lr=0.008,
            povm_lr=0.004,
            is_baseline=True,
        ),
        HyperparamTrialSpec(
            trial_id="s2_t01_stage1_winner_p8p0_eta0p40",
            num_meanings=2,
            boson_p=8.0,
            kron_blend=0.40,
            coinc_gamma=0.35,
            carrier_rho_max=0.22,
            unigram_temp=0.90,
            muon_lr=0.030,
            embed_lr=0.020,
            povm_lr=0.006,
        ),
        HyperparamTrialSpec(
            trial_id="s2_t02_sweetspot_p7p2_eta0p36_tau0p92",
            num_meanings=2,
            boson_p=7.2,
            kron_blend=0.36,
            coinc_gamma=0.32,
            carrier_rho_max=0.24,
            unigram_temp=0.92,
            muon_lr=0.032,
            embed_lr=0.022,
            povm_lr=0.008,
        ),
        HyperparamTrialSpec(
            trial_id="s2_t03_sweetspot_p7p5_eta0p38_tau0p94",
            num_meanings=2,
            boson_p=7.5,
            kron_blend=0.38,
            coinc_gamma=0.35,
            carrier_rho_max=0.25,
            unigram_temp=0.94,
            muon_lr=0.034,
            embed_lr=0.024,
            povm_lr=0.008,
        ),
        HyperparamTrialSpec(
            trial_id="s2_t04_fast_lr_p7p5_muon0p038_emb0p026",
            num_meanings=2,
            boson_p=7.5,
            kron_blend=0.38,
            coinc_gamma=0.35,
            carrier_rho_max=0.24,
            unigram_temp=0.92,
            muon_lr=0.038,
            embed_lr=0.026,
            povm_lr=0.010,
        ),
        HyperparamTrialSpec(
            trial_id="s2_t05_coinc_boost_p7p5_eta0p45_gamma0p55",
            num_meanings=2,
            boson_p=7.5,
            kron_blend=0.45,
            coinc_gamma=0.55,
            carrier_rho_max=0.24,
            unigram_temp=0.92,
            muon_lr=0.032,
            embed_lr=0.022,
            povm_lr=0.008,
        ),
        HyperparamTrialSpec(
            trial_id="s2_t06_rank4_sweetspot_p7p5_eta0p38_tau0p94",
            num_meanings=4,
            boson_p=7.5,
            kron_blend=0.38,
            coinc_gamma=0.35,
            carrier_rho_max=0.24,
            unigram_temp=0.94,
            muon_lr=0.032,
            embed_lr=0.022,
            povm_lr=0.008,
        ),
    )


def get_stage3_production_trials() -> tuple[HyperparamTrialSpec, ...]:
    """Return the 5 Stage-3 production-scale (`B=16, T=512, L=6`) championship trials."""
    return (
        HyperparamTrialSpec(
            trial_id="s3_t00_w41_default_prod_baseline",
            num_meanings=2,
            boson_p=11.0,
            kron_blend=0.55,
            coinc_gamma=0.45,
            carrier_rho_max=0.0,
            unigram_temp=0.80,
            muon_lr=0.015,
            embed_lr=0.008,
            povm_lr=0.004,
            is_baseline=True,
        ),
        HyperparamTrialSpec(
            trial_id="s3_t01_stage1_winner_p8p0_eta0p40",
            num_meanings=2,
            boson_p=8.0,
            kron_blend=0.40,
            coinc_gamma=0.35,
            carrier_rho_max=0.22,
            unigram_temp=0.90,
            muon_lr=0.030,
            embed_lr=0.020,
            povm_lr=0.006,
        ),
        HyperparamTrialSpec(
            trial_id="s3_t02_stage2_coinc_winner_p7p5_eta0p45_gamma0p55",
            num_meanings=2,
            boson_p=7.5,
            kron_blend=0.45,
            coinc_gamma=0.55,
            carrier_rho_max=0.24,
            unigram_temp=0.92,
            muon_lr=0.032,
            embed_lr=0.022,
            povm_lr=0.008,
        ),
        HyperparamTrialSpec(
            trial_id="s3_t03_stage2_sweetspot_p7p5_eta0p38_tau0p94",
            num_meanings=2,
            boson_p=7.5,
            kron_blend=0.38,
            coinc_gamma=0.35,
            carrier_rho_max=0.25,
            unigram_temp=0.94,
            muon_lr=0.034,
            embed_lr=0.024,
            povm_lr=0.008,
        ),
        HyperparamTrialSpec(
            trial_id="s3_t04_high_coinc_p7p8_eta0p48_gamma0p65",
            num_meanings=2,
            boson_p=7.8,
            kron_blend=0.48,
            coinc_gamma=0.65,
            carrier_rho_max=0.24,
            unigram_temp=0.92,
            muon_lr=0.032,
            embed_lr=0.022,
            povm_lr=0.008,
        ),
    )


def _ping_webui_reload(url: str = WEBUI_RELOAD_URL) -> None:
    """Trigger non-blocking reload on the local `:8765` WebUI daemon."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=1.0):
            pass
    except Exception:
        pass


def apply_progress_lr_schedule(
    opt_muon: torch.optim.Optimizer,
    opt_adam: torch.optim.Optimizer,
    tau: float,
    min_lr_ratio: float = 0.20,
    warmup_frac: float = 0.15,
) -> float:
    """Apply Smith one-cycle warmup + cosine decay envelope for normalized progress `tau in [0, 1]` (`<= 8` args)."""
    progress = max(0.0, min(1.0, float(tau)))
    w_f = max(1e-6, min(0.50, float(warmup_frac)))
    if progress < w_f:
        env = 0.25 + 0.75 * (progress / w_f)
    else:
        t_dec = (progress - w_f) / max(1e-6, 1.0 - w_f)
        env = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * t_dec))
    for g in opt_muon.param_groups:
        g["lr"] = float(g.get("base_lr", 0.022)) * env
    for g in opt_adam.param_groups:
        g["lr"] = float(g.get("base_lr", 0.010)) * env
    return float(opt_muon.param_groups[0]["lr"])


def build_validation_batches(
    val_arr: np.ndarray,
    batch_size: int,
    seq_len: int,
    num_batches: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build deterministic `(x_val, y_val)` validation batches on `device` (`<= 8` args)."""
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    span = batch_size * seq_len
    for b_idx in range(num_batches):
        off = (b_idx * span) % max(1, int(val_arr.shape[0]) - span - 2)
        chunk = np.asarray(val_arr[off : off + span + 1], dtype=np.int64)
        x_t = torch.from_numpy(chunk[:-1].reshape(batch_size, seq_len)).to(device=device, dtype=torch.long)
        y_t = torch.from_numpy(chunk[1:].reshape(batch_size, seq_len)).to(device=device, dtype=torch.long)
        batches.append((x_t, y_t))
    return batches


@torch.no_grad()
def evaluate_calibrated_kronecker_model(
    model: CalibratedKroneckerBornGPT,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    """Evaluate `CalibratedKroneckerBornGPT` on held-out validation batches (`<= 8` args)."""
    was_training = model.training
    model.eval()
    v_valid = int(model.cfg.valid_vocab_size)
    total_nll = 0.0
    total_top1 = 0
    total_top50 = 0
    total_tokens = 0
    dev_type = val_batches[0][0].device.type

    for x_val, y_val in val_batches:
        with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=(dev_type == "cuda")):
            log_probs, _ = model(x_val, targets=None, return_logits=True)
        assert log_probs is not None
        lp_f = log_probs[..., :v_valid].float()
        y_clamped = y_val.clamp(min=0, max=v_valid - 1)
        nll = -lp_f.gather(-1, y_clamped.unsqueeze(-1)).squeeze(-1).sum()
        total_nll += float(nll.item())
        preds = lp_f.argmax(dim=-1)
        total_top1 += int((preds == y_clamped).sum().item())
        top50 = lp_f.topk(k=min(50, v_valid), dim=-1).indices
        total_top50 += int((top50 == y_clamped.unsqueeze(-1)).any(dim=-1).sum().item())
        total_tokens += int(y_val.numel())

    val_ce = total_nll / max(1, total_tokens)
    val_ppl = math.exp(min(val_ce, 20.0))
    top1_pct = round(100.0 * total_top1 / max(1, total_tokens), 2)
    top50_pct = round(100.0 * total_top50 / max(1, total_tokens), 2)
    _, pen_stats = compute_calibrated_kronecker_penalties(model)

    if was_training:
        model.train()
    return {
        "val_ce": round(val_ce, 4),
        "val_ppl": round(val_ppl, 2),
        "val_top1": top1_pct,
        "top1_acc": top1_pct,
        "top50_acc": top50_pct,
        "sharp_eff_rank": round(model.compute_sharp_eff_rank(), 2),
        "mean_factor_rank": round(model.compute_mean_factor_rank(), 2),
        "cross_branch_overlap": float(model.compute_cross_branch_overlap()),
        "povm_drift_std": round(model.compute_povm_drift_std(), 6),
        "embed_norm_median": round(pen_stats["embed_norm_median"], 4),
        "embed_norm_max": round(pen_stats["embed_norm_max"], 4),
        "complex_unitary_err": round(pen_stats["complex_unitary_err"], 4),
        "kraus_stiefel_err": round(pen_stats["complex_unitary_err"], 6),
    }


@torch.no_grad()
def audit_step0_readout_calibration_modes(
    tokens_stream: np.ndarray | torch.Tensor,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    cfg_runner: SearchRunnerConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Audit the 4 Step-0 calibration arms of `H1` (`EXPERIMENT_PLAN_057.md` Section 2.1) (`<= 8` args)."""
    modes = (
        ("uncalibrated_haar", 0.0, False, False),
        ("carrier_angle_only", 0.35, True, False),
        ("trainable_povm_only", 0.0, True, True),
        ("hybrid_procrustes_carrier_povm", 0.25, True, True),
    )
    results: dict[str, Any] = {}
    eval_seqs = torch.cat([xb for xb, _ in val_batches], dim=0)[:4]

    for mode_name, rho_max, do_procrustes, do_povm in modes:
        cfg = CalibratedKroneckerConfig(
            vocab_size=cfg_runner.vocab_size,
            valid_vocab_size=cfg_runner.valid_vocab_size,
            num_meanings=2,
            num_factors=DEFAULT_NUM_FACTORS,
            factor_dim=DEFAULT_FACTOR_DIM,
            num_layers=cfg_runner.num_layers,
            block_size=cfg_runner.seq_len,
            carrier_rho_max=rho_max,
        )
        model = CalibratedKroneckerBornGPT(cfg).to(device=device)
        assert not hasattr(model, "prior_weights")

        if not do_procrustes and not do_povm:
            model.update_unigram_probs_from_tokens(tokens_stream)
            q_re, q_im, _, _ = model._collect_backbone_pre_operator_states(eval_seqs)
            q_mean_re = q_re.mean(dim=0)
            q_mean_im = q_im.mean(dim=0)
            fid = model.compute_carrier_alignment_fidelity(q_mean_re, q_mean_im)
            u_c = torch.complex(model.out_factor_u_re.float(), model.out_factor_u_im.float())
            eye = torch.eye(cfg.factor_dim, device=device, dtype=torch.complex64).unsqueeze(0)
            u_err = float(
                torch.linalg.matrix_norm(torch.bmm(u_c, u_c.conj().transpose(-1, -2)) - eye).max().item()
            )
            kl_step0 = model.compute_empirical_output_kl_to_target(eval_seqs)
            cal_stats = {
                "pre_procrustes_carrier_fidelity": round(fid, 6),
                "carrier_alignment_fidelity": round(fid, 6),
                "unitary_frobenius_err": u_err,
                "step0_kl_to_target_unigram": round(kl_step0, 6),
                "sharp_eff_rank": round(model.compute_sharp_eff_rank(), 2),
                "mean_factor_rank": round(model.compute_mean_factor_rank(), 2),
                "cross_branch_overlap": float(model.compute_cross_branch_overlap()),
            }
        else:
            cal_stats = model.calibrate_readout_to_random_attention_unigram(
                token_stream=tokens_stream,
                num_batches=DEFAULT_CALIBRATION_BATCHES,
                seq_len=min(DEFAULT_CALIBRATION_SEQ_LEN, cfg_runner.seq_len),
                carrier_rho_max=rho_max,
                calibrate_povm=do_povm,
                seed=1337,
            )

        ev0 = evaluate_calibrated_kronecker_model(model, val_batches)
        results[mode_name] = {
            **cal_stats,
            "val_ce_step0": ev0["val_ce"],
            "val_ppl_step0": ev0["val_ppl"],
            "top1_acc_step0": ev0["top1_acc"],
            "top50_acc_step0": ev0["top50_acc"],
            "zero_frozen_prior_buffer": not hasattr(model, "prior_weights"),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    uncal_kl = float(results["uncalibrated_haar"]["step0_kl_to_target_unigram"])
    povm_kl = float(results["trainable_povm_only"]["step0_kl_to_target_unigram"])
    hyb_kl = float(results["hybrid_procrustes_carrier_povm"]["step0_kl_to_target_unigram"])
    kl_reduction_x = uncal_kl / max(hyb_kl, 1e-6)
    return {
        "modes": results,
        "kl_reduction_factor_vs_uncalibrated": round(kl_reduction_x, 1),
        "povm_kl_below_0p005": bool(povm_kl < 0.005),
        "hybrid_kl_below_0p005": bool(hyb_kl < 0.005),
    }


def run_single_hyperparam_trial(
    spec: HyperparamTrialSpec,
    work_gpu: torch.Tensor,
    tokens_stream: np.ndarray | torch.Tensor,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    cfg_runner: SearchRunnerConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Execute a single candidate trial of `CalibratedKroneckerBornGPT` (`<= 8` args)."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    torch.manual_seed(1337)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1337)

    cfg = CalibratedKroneckerConfig(
        vocab_size=cfg_runner.vocab_size,
        valid_vocab_size=cfg_runner.valid_vocab_size,
        num_meanings=spec.num_meanings,
        num_factors=DEFAULT_NUM_FACTORS,
        factor_dim=DEFAULT_FACTOR_DIM,
        num_layers=cfg_runner.num_layers,
        block_size=cfg_runner.seq_len,
        boson_p=spec.boson_p,
        holo_alpha=DEFAULT_CAL_HOLO_ALPHA,
        kron_blend=spec.kron_blend,
        coinc_gamma=spec.coinc_gamma,
        unigram_temp=spec.unigram_temp,
        carrier_rho_max=spec.carrier_rho_max,
        povm_anchor_weight=DEFAULT_POVM_ANCHOR_WEIGHT,
        sphere_penalty_weight=DEFAULT_SPHERE_PENALTY_WEIGHT,
        unitary_penalty_weight=DEFAULT_UNITARY_PENALTY_WEIGHT,
        iso_penalty_weight=DEFAULT_ISO_PENALTY_WEIGHT,
        branch_orth_weight=DEFAULT_BRANCH_ORTH_WEIGHT,
    )
    model = CalibratedKroneckerBornGPT(cfg).to(device=device)
    cal_stats = model.calibrate_readout_to_random_attention_unigram(
        token_stream=tokens_stream,
        num_batches=DEFAULT_CALIBRATION_BATCHES,
        seq_len=min(DEFAULT_CALIBRATION_SEQ_LEN, cfg_runner.seq_len),
        carrier_rho_max=spec.carrier_rho_max,
        calibrate_povm=True,
        seed=1337,
    )
    opt_muon, opt_adam = build_calibrated_kronecker_optimizers(
        model,
        base_muon_lr=spec.muon_lr,
        base_embed_lr=spec.embed_lr,
        base_povm_lr=spec.povm_lr,
        base_gate_lr=spec.gate_lr,
    )

    params_m = round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2)
    tokens_per_step = cfg_runner.batch_size * cfg_runner.seq_len
    total_tokens_m = round((cfg_runner.steps_per_trial * tokens_per_step) / 1e6, 3)
    run_key = f"w42_hsearch_{spec.trial_id}_{cfg_runner.steps_per_trial}steps"

    logger: StandardizedExperimentLogger | None = None
    if cfg_runner.log_to_webui:
        intent = ExperimentIntentSpec(
            run_key=run_key,
            wave_id=SEARCH_WAVE_ID,
            plan_id=SEARCH_PLAN_ID,
            task_domain="micro_verification",
            hypothesis_intent=(
                f"Wave 42 H3 hyperparameter search trial `{spec.trial_id}` "
                f"(R={spec.num_meanings}, p={spec.boson_p}, eta={spec.kron_blend}, "
                f"gamma_c={spec.coinc_gamma}, rho_max={spec.carrier_rho_max}, tau_uni={spec.unigram_temp})."
            ),
            architecture_summary=(
                f"CalibratedKroneckerBornGPT (R={spec.num_meanings}, M=4, d_f=16, L={cfg_runner.num_layers}, "
                f"{params_m:.2f}M params)"
            ),
            hardware_budget="1x RTX 5060 Ti 16GB (BF16 Calibrated 16^4 Kronecker Hyperparam Sweep)",
            batch_size=cfg_runner.batch_size,
            seq_len=cfg_runner.seq_len,
            total_steps=cfg_runner.steps_per_trial,
            target_tokens_M=total_tokens_m,
            params_M=params_m,
            embed_dim=cfg.real_equiv_dim,
            num_layers=cfg.num_layers,
            schedule="smith_one_cycle",
        )
        logger = StandardizedExperimentLogger(intent)

    eval_0 = evaluate_calibrated_kronecker_model(model, val_batches)
    if logger is not None:
        logger.log_eval(
            0,
            {
                **eval_0,
                "step": 0,
                "tokens_M": 0.0,
                "train_ce": eval_0["val_ce"],
                "compute_tflops": 0.0,
                "tau_progress": 0.0,
                "elapsed_min": 0.0,
                "tok_per_s": 0.0,
            },
        )

    n_work = int(work_gpu.numel())
    v_valid = int(cfg_runner.valid_vocab_size)
    nan_inf_events = 0
    last_train_ce = eval_0["val_ce"]
    t0 = time.perf_counter()
    model.train()

    for step in range(1, cfg_runner.steps_per_trial + 1):
        tau = float(step - 1) / float(max(1, cfg_runner.steps_per_trial))
        model.set_training_progress(tau)
        cur_lr = apply_progress_lr_schedule(opt_muon, opt_adam, tau)
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)

        off = ((step - 1) * tokens_per_step) % max(1, n_work - tokens_per_step - 2)
        seq = (work_gpu[off : off + tokens_per_step + 1].to(torch.int64) & 0xFFFF).clamp(0, v_valid - 1)
        x_b = seq[:-1].view(cfg_runner.batch_size, cfg_runner.seq_len)
        y_b = seq[1:].view(cfg_runner.batch_size, cfg_runner.seq_len)

        with torch.amp.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
        ):
            _, nll_loss = model(x_b, targets=y_b, return_logits=False)
        assert nll_loss is not None
        pen_loss, _ = compute_calibrated_kronecker_penalties(model)
        total_loss = nll_loss + pen_loss
        if not bool(torch.isfinite(total_loss).item()):
            nan_inf_events += 1
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_muon.step()
        opt_adam.step()
        last_train_ce = float(nll_loss.detach().item())

        if logger is not None and (step == 1 or step == cfg_runner.steps_per_trial or step % 40 == 0):
            el_s = max(1e-6, time.perf_counter() - t0)
            tok_s = (step * tokens_per_step) / el_s
            logger.log_step(step, last_train_ce, tok_s, cur_lr / max(1e-6, spec.muon_lr), el_s / 60.0)
        if logger is not None and cfg_runner.steps_per_trial >= 200 and step < cfg_runner.steps_per_trial and step % 100 == 0:
            mid_ev = evaluate_calibrated_kronecker_model(model, val_batches)
            el_s = max(1e-6, time.perf_counter() - t0)
            tok_m = round((step * tokens_per_step) / 1e6, 3)
            logger.log_eval(
                step,
                {
                    **mid_ev,
                    "step": step,
                    "tokens_M": tok_m,
                    "train_ce": round(last_train_ce, 4),
                    "compute_tflops": round(6.0 * params_m * tok_m, 4),
                    "tau_progress": round(step / float(cfg_runner.steps_per_trial), 4),
                    "elapsed_min": round(el_s / 60.0, 4),
                    "tok_per_s": round((step * tokens_per_step) / el_s, 1),
                },
            )
            _ping_webui_reload()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = max(1e-6, time.perf_counter() - t0)
    tok_per_s = round((cfg_runner.steps_per_trial * tokens_per_step) / elapsed_s, 1)
    final_ev = evaluate_calibrated_kronecker_model(model, val_batches)
    nats_dropped = round(eval_0["val_ce"] - final_ev["val_ce"], 4)
    nats_per_min = round(nats_dropped / (elapsed_s / 60.0), 4)

    trial_result = {
        "trial_id": spec.trial_id,
        "run_key": run_key,
        "is_baseline": spec.is_baseline,
        "hyperparams": asdict(spec),
        "params_M": params_m,
        "steps": cfg_runner.steps_per_trial,
        "tokens_M": total_tokens_m,
        "step0_val_ce": eval_0["val_ce"],
        "final_val_ce": final_ev["val_ce"],
        "final_val_ppl": final_ev["val_ppl"],
        "final_top1_acc": final_ev["top1_acc"],
        "final_top50_acc": final_ev["top50_acc"],
        "final_train_ce": round(last_train_ce, 4),
        "sharp_eff_rank": final_ev["sharp_eff_rank"],
        "mean_factor_rank": final_ev["mean_factor_rank"],
        "cross_branch_overlap": final_ev["cross_branch_overlap"],
        "povm_drift_std": final_ev["povm_drift_std"],
        "embed_norm_median": final_ev["embed_norm_median"],
        "complex_unitary_err": final_ev["complex_unitary_err"],
        "tok_per_s": tok_per_s,
        "elapsed_sec": round(elapsed_s, 2),
        "nats_improved_from_step0": nats_dropped,
        "nats_improved_per_min": nats_per_min,
        "nan_inf_events": nan_inf_events,
        "calibration_stats": cal_stats,
    }
    if logger is not None:
        logger.log_eval(
            cfg_runner.steps_per_trial,
            {
                **final_ev,
                "step": cfg_runner.steps_per_trial,
                "tokens_M": total_tokens_m,
                "train_ce": round(last_train_ce, 4),
                "compute_tflops": round(6.0 * params_m * total_tokens_m, 4),
                "tau_progress": 1.0,
                "elapsed_min": round(elapsed_s / 60.0, 4),
                "tok_per_s": tok_per_s,
            },
        )
        logger.finalize(extra_summary=trial_result)
        _ping_webui_reload()

    del model, opt_muon, opt_adam
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return trial_result


def run_hyperparam_search_suite(
    trials: tuple[HyperparamTrialSpec, ...] | list[HyperparamTrialSpec],
    work_gpu: torch.Tensor,
    tokens_stream: np.ndarray | torch.Tensor,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    cfg_runner: SearchRunnerConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Run all candidate `trials`, rank by `final_val_ce`, and compute improvement over baseline (`<= 8` args)."""
    trial_results: list[dict[str, Any]] = []
    for idx, spec in enumerate(trials, start=1):
        print(
            f"[H3 Search {idx}/{len(trials)}] Running `{spec.trial_id}` "
            f"(R={spec.num_meanings}, p={spec.boson_p}, eta={spec.kron_blend}, "
            f"gamma_c={spec.coinc_gamma}, rho={spec.carrier_rho_max}, tau={spec.unigram_temp}, "
            f"lr={spec.muon_lr}/{spec.embed_lr}/{spec.povm_lr}) ...",
            flush=True,
        )
        res = run_single_hyperparam_trial(spec, work_gpu, tokens_stream, val_batches, cfg_runner, device)
        print(
            f"  -> {spec.trial_id}: Step0 CE={res['step0_val_ce']:.4f} -> Final Val CE={res['final_val_ce']:.4f} "
            f"(PPL={res['final_val_ppl']:.2f}, Top1={res['final_top1_acc']:.2f}%, R_emb={res['sharp_eff_rank']:.1f}, "
            f"{res['tok_per_s']:,.0f} tok/s, {res['elapsed_sec']:.1f}s)",
            flush=True,
        )
        trial_results.append(res)

    baseline_trials = [r for r in trial_results if r.get("is_baseline")]
    baseline_res = baseline_trials[0] if baseline_trials else trial_results[0]
    baseline_val_ce = float(baseline_res["final_val_ce"])

    ranked = sorted(trial_results, key=lambda r: (float(r["final_val_ce"]), -float(r["final_top1_acc"])))
    for rank_idx, r in enumerate(ranked, start=1):
        r["rank"] = rank_idx
        r["val_ce_delta_vs_baseline"] = round(float(r["final_val_ce"]) - baseline_val_ce, 4)

    best_trial = ranked[0]
    improvement_nats = round(baseline_val_ce - float(best_trial["final_val_ce"]), 4)
    total_nans = sum(int(r["nan_inf_events"]) for r in trial_results)
    passed = bool(
        len(trial_results) >= 2
        and total_nans == 0
        and (
            improvement_nats >= MIN_NATS_IMPROVEMENT_TARGET
            if len(trial_results) >= 6
            else float(best_trial["final_val_ce"]) <= baseline_val_ce
        )
    )
    return {
        "passed": passed,
        "num_trials": len(trial_results),
        "steps_per_trial": cfg_runner.steps_per_trial,
        "total_nan_inf_events": total_nans,
        "baseline_trial_id": baseline_res["trial_id"],
        "baseline_val_ce": baseline_val_ce,
        "best_trial_id": best_trial["trial_id"],
        "best_val_ce": best_trial["final_val_ce"],
        "best_top1_acc": best_trial["final_top1_acc"],
        "best_top50_acc": best_trial["final_top50_acc"],
        "val_ce_improvement_over_baseline_nats": improvement_nats,
        "best_trial": best_trial,
        "leaderboard": ranked,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for `scripts/kronecker_born_hyperparam_search.py`."""
    parser = argparse.ArgumentParser(
        description="Run Step-0 readout calibration audit & automated hyperparameter search for CalibratedKroneckerBornGPT."
    )
    parser.add_argument(
        "--steps-per-trial",
        type=int,
        default=DEFAULT_SEARCH_STEPS,
        help="Training steps per candidate hyperparameter trial (default: 120).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_SEARCH_BATCH_SIZE,
        help="Micro-batch size for screening trials (default: 8).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEARCH_SEQ_LEN,
        help="Sequence length for screening trials (default: 128).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=DEFAULT_SEARCH_LAYERS,
        help="Backbone depth L for screening trials (default: 4).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=("stage1", "stage2", "stage3"),
        default="stage1",
        help="Search grid preset: `stage1` (broad 7-trial), `stage2` (fine 7-trial), or `stage3` (production L=6 5-trial).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_SEARCH_JSON,
        help="Output JSON path for leaderboard and Step-0 calibration telemetry.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute Step-0 calibration audit + automated hyperparameter search and write JSON report."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.95, device=0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    shard1 = DATA_DIR / "train_full_100M.bin"
    if not shard1.exists():
        shard1 = DATA_DIR / "fineweb_train_000001.bin"
    val_path = DATA_DIR / "val_slice.bin"

    tokens_100m = _load_header_safe_shard_tokens(shard1, max_tokens=SLICE_100M_TOKENS)
    val_tokens = _load_header_safe_shard_tokens(val_path, max_tokens=VAL_SLICE_TOKENS)
    work_gpu = torch.from_numpy(tokens_100m[:SLICE_30M_TOKENS].view("int16")).to(device)

    cfg_runner = SearchRunnerConfig(
        steps_per_trial=int(args.steps_per_trial),
        batch_size=int(args.batch_size),
        seq_len=int(args.seq_len),
        num_layers=int(args.num_layers),
    )
    val_batches = build_validation_batches(
        val_tokens,
        batch_size=DEFAULT_SEARCH_VAL_BATCH_SIZE,
        seq_len=cfg_runner.seq_len,
        num_batches=cfg_runner.val_batches,
        device=device,
    )

    print("[Step-0 Calibration Audit] Auditing random-init attention output calibration modes ...", flush=True)
    step0_audit = audit_step0_readout_calibration_modes(tokens_100m, val_batches, cfg_runner, device)
    print(f"[Step-0 Calibration Audit] KL reduction={step0_audit['kl_reduction_factor_vs_uncalibrated']}x", flush=True)

    if str(args.preset) == "stage3":
        candidates = get_stage3_production_trials()
    elif str(args.preset) == "stage2":
        candidates = get_stage2_fine_grid_trials()
    else:
        candidates = get_default_candidate_trials()
    search_report = run_hyperparam_search_suite(
        candidates, work_gpu, tokens_100m, val_batches, cfg_runner, device
    )
    output_payload = {
        "plan_id": SEARCH_PLAN_ID,
        "wave_id": SEARCH_WAVE_ID,
        "preset": str(args.preset),
        "step0_calibration_audit": step0_audit,
        "hyperparam_search": search_report,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(
        f"[W42 Hyperparam Search] Winner: `{search_report['best_trial_id']}` "
        f"(Val CE={search_report['best_val_ce']:.4f}, delta={-search_report['val_ce_improvement_over_baseline_nats']:+.4f} nats vs baseline) "
        f"-> saved to {args.json_out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
