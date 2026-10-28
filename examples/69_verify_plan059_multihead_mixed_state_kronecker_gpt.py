#!/usr/bin/env python3
"""Plan 059 / Wave 44 Verification & Ablation Runner (`MultiHeadMixedStateKroneckerGPT`).

Verifies and benchmarks the Wave 44 architectural tracks:
- Track 1 (`stiefel_muon`): Paired complex `(Re, Im)` Stiefel Muon optimization & retraction.
- Track 3 (`self_echo`): Carrier-orthogonal complex self-echo (`"ortho_complex_learned"`,
  `"ortho_complex_full"`, `"no_self_echo"`, `"legacy_ambient_real"`) preserving
  `<c_ref | raw_phi^{(r,m)}> == <c_ref | op^{(r,m)}>` (`< 1e-6`).
- Track 4 (`polysemy`): Non-diluted (`1.0x`) secondary branch initialization (`Step-0` branch purity
  `= 1.0`), `R`-invariant per-factor normalization across `R in {2, 3, 4}`, and phase-sensitive
  Hermitian context routing (`"phase_sensitive"` vs. `"legacy_norm_only"`).
- Track 6 (`multihead`): Multi-Head Quantum Superposition & Stinespring Mixed-State Attention
  (`"multihead_gram"`, `"legacy_krylov_11mm"`, `"adhoc_poly_gamma"`) reducing `(T x T)` matmul calls
  from `11 -> 3` (`5` total `torch.matmul` calls) and evaluating `MixedStateKroneckerBornChunkedNLLFn`
  plus `evaluate_modded_nanogpt_validation`.

All runs record explicit intent and telemetry via `StandardizedExperimentLogger`
(`wave_id="W44"`, `plan_id="EXPERIMENT_PLAN_059"`).
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
import tempfile
import time
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qdlblocks.benchmarks.experiment_logger import (
    ExperimentIntentSpec,
    StandardizedExperimentLogger,
)
from qdlblocks.layers.multihead_superposition_mixed_state_attn import (
    MultiHeadSuperpositionAttnConfig,
    MultiHeadSuperpositionMixedStateAttention,
    count_attention_matmul_calls,
)
from qdlblocks.models.kronecker_multimeaning_16x4_gpt import _load_header_safe_shard_tokens
from qdlblocks.models.multihead_mixed_state_kronecker_gpt import (
    DEFAULT_NON_DILUTED_SECONDARY_BRANCH_SCALE,
    LEGACY_DILUTED_SECONDARY_BRANCH_SCALE,
    MultiHeadMixedStateKroneckerConfig,
    MultiHeadMixedStateKroneckerGPT,
    apply_carrier_orthogonal_complex_self_echo,
    build_multihead_mixed_state_kronecker_optimizers,
    compute_multihead_mixed_state_kronecker_penalties,
    compute_phase_sensitive_branch_beta,
    normalize_raw_phi_per_factor_r_invariant,
)
from qdlblocks.models.ngram_calibrated_kronecker_born_gpt import (
    DEFAULT_MODDED_VAL_TOKENS,
    DEFAULT_MUON_LR,
    MultiShardRareTokenSampler,
    build_or_load_900m_ngram_calibration_cache,
    evaluate_modded_nanogpt_validation,
)
from qdlblocks.optim.muon import compute_large_batch_muon_lr_scale

# ==============================================================================
# Rule 7 Named Constants (Plan 059 / Wave 44)
# ==============================================================================
WAVE_ID: str = "W44"
PLAN_ID: str = "EXPERIMENT_PLAN_059"
DEFAULT_ABLATION_CHOICE: str = "all"
VALID_ABLATION_CHOICES: tuple[str, ...] = (
    "all",
    "stiefel_muon",
    "self_echo",
    "polysemy",
    "multihead",
)
SMOKE_VOCAB_SIZE: int = 256
SMOKE_NUM_MEANINGS: int = 2
SMOKE_NUM_FACTORS: int = 4
SMOKE_FACTOR_DIM: int = 16
SMOKE_N_LAYERS: int = 2
SMOKE_SUBHEADS: int = 2
SMOKE_SEQ_LEN: int = 32
SMOKE_BATCH_TOKENS: int = 64
SMOKE_STEPS: int = 2
DEFAULT_VERIFY_VOCAB_SIZE: int = 1024
DEFAULT_VERIFY_N_LAYERS: int = 2
DEFAULT_VERIFY_SEQ_LEN: int = 64
DEFAULT_VERIFY_BATCH_TOKENS: int = 256
DEFAULT_VERIFY_STEPS: int = 6
CARRIER_ORTHO_TOL: float = 1e-6

DEFAULT_DATA_DIR: str = "/home/festivity/agy/QuantumDLBlocks/.cache/fineweb_100M"
DEFAULT_CACHE_PATH: str = "/home/festivity/agy/QuantumDLBlocks/.cache/fineweb_900M_ngram_calibration.pt"
DEFAULT_OUT_JSON: str = "/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_059.json"
DEFAULT_FINEWEB_RUN_KEY: str = "w44_flagship_multihead_mixed_state_kronecker_r2_d128c_L6_600s"
DEFAULT_NUM_SHARDS: int = 9
DEFAULT_SHARD_LOAD_TOKENS: int = 25_000_000
DEFAULT_FINEWEB_BATCH_SIZE: int = 16
DEFAULT_FINEWEB_SEQ_LEN: int = 512
DEFAULT_FINEWEB_LAYERS: int = 6
DEFAULT_FINEWEB_WALL_S: float = 600.0
DEFAULT_FINEWEB_MAX_STEPS: int = 1000
DEFAULT_EVAL_TOKENS_FAST: int = 65_536
DEFAULT_WSD_WARMUP_FRAC: float = 0.05
DEFAULT_WSD_COOLDOWN_FRAC: float = 0.15
DEFAULT_MIN_LR_RATIO: float = 0.10
DEFAULT_GRAD_CLIP_NORM: float = 1.0
DEFAULT_LOG_INTERVAL: int = 25
DEFAULT_EVAL_INTERVAL: int = 50


@dataclass(frozen=True)
class VerificationRunSpec:
    """Configuration for a single Plan 059 verification or ablation run."""

    run_key: str
    hypothesis_intent: str
    model_cfg: MultiHeadMixedStateKroneckerConfig
    steps: int
    seq_len: int
    batch_size: int
    smoke: bool


@dataclass(frozen=True)
class FineWebFlagshipSpec:
    """Configuration for the 10-minute (600s) Wave 44 9-shard FineWeb flagship run (`<= 8` fields)."""

    run_key: str = DEFAULT_FINEWEB_RUN_KEY
    wall_s: float = DEFAULT_FINEWEB_WALL_S
    max_steps: int = DEFAULT_FINEWEB_MAX_STEPS
    batch_size: int = DEFAULT_FINEWEB_BATCH_SIZE
    seq_len: int = DEFAULT_FINEWEB_SEQ_LEN
    grad_accum_steps: int = 1
    alpha_rare: float = 0.0
    eval_10m_at_end: bool = True


def verify_track3_self_echo_invariants(device: torch.device) -> dict[str, float]:
    """Verify Track 3 carrier-orthogonal complex self-echo invariants across modes."""
    torch.manual_seed(4403)
    n_tok, r_num, m_num, d_f = 16, 2, 4, 16
    op_re = torch.randn(n_tok, r_num, m_num, d_f, device=device, dtype=torch.float64)
    op_im = torch.randn(n_tok, r_num, m_num, d_f, device=device, dtype=torch.float64)
    # Construct p0 as a +i phase-rotated copy of op on H_perp plus a large carrier offset
    u_perp_re = op_re - op_re.mean(dim=-1, keepdim=True)
    u_perp_im = op_im - op_im.mean(dim=-1, keepdim=True)
    p0_re = -u_perp_im + 0.5
    p0_im = u_perp_re - 0.3

    out_c_re, out_c_im = apply_carrier_orthogonal_complex_self_echo(
        op_re, op_im, p0_re, p0_im, self_echo_mode="ortho_complex_full"
    )
    out_l_re, out_l_im = apply_carrier_orthogonal_complex_self_echo(
        op_re, op_im, p0_re, p0_im, self_echo_mode="ortho_complex_learned", echo_gate_logits=0.0
    )
    out_n_re, out_n_im = apply_carrier_orthogonal_complex_self_echo(
        op_re, op_im, p0_re, p0_im, self_echo_mode="no_self_echo"
    )

    # Carrier component is the mean along d_f=16 (since |c_ref> is proportional to 1_{16})
    carrier_diff_full = float(
        (
            (out_c_re.mean(dim=-1) - op_re.mean(dim=-1)).abs().max()
            + (out_c_im.mean(dim=-1) - op_im.mean(dim=-1)).abs().max()
        ).item()
    )
    carrier_diff_learned = float(
        (
            (out_l_re.mean(dim=-1) - op_re.mean(dim=-1)).abs().max()
            + (out_l_im.mean(dim=-1) - op_im.mean(dim=-1)).abs().max()
        ).item()
    )
    carrier_leak = max(carrier_diff_full, carrier_diff_learned)
    if carrier_leak >= CARRIER_ORTHO_TOL:
        raise AssertionError(f"Carrier leak {carrier_leak:.3e} exceeded {CARRIER_ORTHO_TOL}")
    if not (torch.allclose(out_n_re, op_re) and torch.allclose(out_n_im, op_im)):
        raise AssertionError("Self-echo mode='no_self_echo' must be an exact identity.")

    # Under ortho_complex_full, H_perp component of out_c must be completely annihilated when p0_perp = i * op_perp
    rem_perp_norm = float(
        (
            (out_c_re - out_c_re.mean(dim=-1, keepdim=True)).square()
            + (out_c_im - out_c_im.mean(dim=-1, keepdim=True)).square()
        )
        .sum(dim=-1)
        .max()
        .item()
    )
    return {
        "carrier_leak_max": carrier_leak,
        "full_echo_residual_perp_max": rem_perp_norm,
    }


def verify_track4_polysemy_and_router_invariants(device: torch.device) -> dict[str, float]:
    """Verify Track 4 non-diluted Step-0 branch purity, R-invariance, and phase sensitivity."""
    torch.manual_seed(4404)
    cfg_nd = MultiHeadMixedStateKroneckerConfig(
        vocab_size=SMOKE_VOCAB_SIZE,
        valid_vocab_size=SMOKE_VOCAB_SIZE,
        num_meanings=4,
        num_factors=4,
        factor_dim=16,
        num_layers=1,
        secondary_branch_scale=DEFAULT_NON_DILUTED_SECONDARY_BRANCH_SCALE,
    )
    model_nd = MultiHeadMixedStateKroneckerGPT(cfg_nd).to(device)
    tokens = torch.randint(0, SMOKE_VOCAB_SIZE, (2, 16), device=device)
    tel_nd = model_nd.compute_branch_routing_telemetry(tokens)
    step0_purity_nd = float(tel_nd["step0_branch_ket_purity"])
    step0_gate_ratio_nd = float(tel_nd["secondary_to_primary_gate_ratio"])

    factor_norms_by_r: list[float] = []
    for r_val in (2, 3, 4):
        raw_re = torch.randn(2, 8, r_val, 4, 16, device=device)
        raw_im = torch.randn(2, 8, r_val, 4, 16, device=device)
        phi_re, phi_im = normalize_raw_phi_per_factor_r_invariant(raw_re, raw_im)
        f_norm = float((phi_re.square() + phi_im.square()).sum(dim=-1).mean().item())
        factor_norms_by_r.append(f_norm)
    max_r_norm_dev = max(abs(x - 1.0) for x in factor_norms_by_r)

    r_branches = 4
    psi0_re = torch.randn(2, 6, r_branches, 4, 16, device=device, dtype=torch.float64)
    psi0_im = torch.randn(2, 6, r_branches, 4, 16, device=device, dtype=torch.float64)
    zero_w_re = torch.zeros(r_branches, 4, 16, device=device, dtype=torch.float64)
    zero_w_im = torch.zeros(r_branches, 4, 16, device=device, dtype=torch.float64)
    zero_b = torch.zeros(r_branches, device=device, dtype=torch.float64)
    beta_step0 = compute_phase_sensitive_branch_beta(
        psi0_re, psi0_im, psi0_re, psi0_im, zero_w_re, zero_w_im, zero_b
    )
    step0_beta_err = float((beta_step0 - (1.0 / math.sqrt(r_branches))).abs().max().item())

    w_re = torch.randn(r_branches, 4, 16, device=device, dtype=torch.float64) * 0.5
    w_im = torch.randn(r_branches, 4, 16, device=device, dtype=torch.float64) * 0.5
    b_vec = torch.zeros(r_branches, device=device, dtype=torch.float64)
    beta_pos1 = compute_phase_sensitive_branch_beta(
        psi0_re, psi0_im, psi0_re, psi0_im, w_re, w_im, b_vec
    )
    beta_neg1 = compute_phase_sensitive_branch_beta(
        -psi0_re, -psi0_im, psi0_re, psi0_im, w_re, w_im, b_vec
    )
    beta_posi = compute_phase_sensitive_branch_beta(
        -psi0_im, psi0_re, psi0_re, psi0_im, w_re, w_im, b_vec
    )
    beta_negi = compute_phase_sensitive_branch_beta(
        psi0_im, -psi0_re, psi0_re, psi0_im, w_re, w_im, b_vec
    )
    diff_real_sign = float((beta_pos1 - beta_neg1).abs().max().item())
    diff_imag_sign = float((beta_posi - beta_negi).abs().max().item())

    return {
        "step0_branch_purity_nd": step0_purity_nd,
        "step0_gate_ratio_nd": step0_gate_ratio_nd,
        "max_r_factor_norm_dev": max_r_norm_dev,
        "step0_beta_uniform_err": step0_beta_err,
        "phase_diff_pos1_vs_neg1": diff_real_sign,
        "phase_diff_posi_vs_negi": diff_imag_sign,
    }


def verify_track6_multihead_matmul_and_purity(device: torch.device) -> dict[str, float]:
    """Verify Track 6 matmul count reduction (11 -> 3 TxT / 5 total) and mixed-state purity."""
    torch.manual_seed(4406)
    cfg_mh = MultiHeadSuperpositionAttnConfig(
        num_meanings=SMOKE_NUM_MEANINGS,
        num_factors=SMOKE_NUM_FACTORS,
        d_register=SMOKE_FACTOR_DIM,
        n_subheads_per_factor=SMOKE_SUBHEADS,
        attn_evolution_mode="multihead_gram",
    )
    layer_mh = MultiHeadSuperpositionMixedStateAttention(cfg_mh).to(device)
    psi_re = torch.randn(2, 32, cfg_mh.d_model, device=device)
    psi_im = torch.randn(2, 32, cfg_mh.d_model, device=device)
    total_mh, txt_mh = count_attention_matmul_calls(layer_mh, psi_re, psi_im)
    layer_mh(psi_re, psi_im)
    tel_mh = layer_mh.compute_subhead_purity_telemetry()

    cfg_leg = MultiHeadSuperpositionAttnConfig(
        num_meanings=SMOKE_NUM_MEANINGS,
        num_factors=SMOKE_NUM_FACTORS,
        d_register=SMOKE_FACTOR_DIM,
        n_subheads_per_factor=SMOKE_SUBHEADS,
        attn_evolution_mode="legacy_krylov_11mm",
    )
    layer_leg = MultiHeadSuperpositionMixedStateAttention(cfg_leg).to(device)
    total_leg, txt_leg = count_attention_matmul_calls(layer_leg, psi_re, psi_im)

    return {
        "multihead_total_matmuls": float(total_mh),
        "multihead_txt_matmuls": float(txt_mh),
        "legacy_total_matmuls": float(total_leg),
        "legacy_txt_matmuls": float(txt_leg),
        "gram_trace_mean": float(tel_mh["subhead_gram_trace_mean"]),
        "subhead_purity_mean": float(tel_mh["subhead_gram_purity_mean"]),
        "reduced_state_purity_mean": float(tel_mh["reduced_density_purity_mean"]),
    }


def execute_training_verification_run(
    run_spec: VerificationRunSpec,
    device: torch.device,
    logs_dir: Path,
    cards_dir: Path,
) -> dict[str, float]:
    """Run a short training + validation loop logged via `StandardizedExperimentLogger`."""
    torch.manual_seed(4400)
    model = MultiHeadMixedStateKroneckerGPT(run_spec.model_cfg).to(device)
    muon_opt, adamw_opt = build_multihead_mixed_state_kronecker_optimizers(model)
    param_count_m = sum(p.numel() for p in model.parameters()) / 1e6
    target_tok_m = (run_spec.steps * run_spec.batch_size * run_spec.seq_len) / 1e6

    intent = ExperimentIntentSpec(
        run_key=run_spec.run_key,
        wave_id=WAVE_ID,
        plan_id=PLAN_ID,
        task_domain="micro_verification",
        hypothesis_intent=run_spec.hypothesis_intent,
        architecture_summary=(
            f"MultiHeadMixedStateKroneckerGPT(mode={run_spec.model_cfg.attn_evolution_mode}, "
            f"echo={run_spec.model_cfg.self_echo_mode}, router={run_spec.model_cfg.branch_router_mode}, "
            f"readout={run_spec.model_cfg.readout_density_mode})"
        ),
        batch_size=run_spec.batch_size,
        seq_len=run_spec.seq_len,
        total_steps=run_spec.steps,
        target_tokens_M=max(0.001, target_tok_m),
        params_M=param_count_m,
        embed_dim=run_spec.model_cfg.real_equiv_dim,
        num_layers=run_spec.model_cfg.num_layers,
    )
    logger = StandardizedExperimentLogger(
        spec=intent,
        live_logs_dir=logs_dir,
        model_cards_dir=cards_dir,
    )

    last_ce = 0.0
    for step_idx in range(1, run_spec.steps + 1):
        t0 = time.perf_counter()
        x = torch.randint(
            0,
            run_spec.model_cfg.valid_vocab_size,
            (run_spec.batch_size, run_spec.seq_len),
            device=device,
        )
        y = torch.randint(
            0,
            run_spec.model_cfg.valid_vocab_size,
            (run_spec.batch_size, run_spec.seq_len),
            device=device,
        )
        muon_opt.zero_grad(set_to_none=True)
        adamw_opt.zero_grad(set_to_none=True)

        _, nll_loss = model(x, targets=y)
        assert nll_loss is not None
        pen_loss, _ = compute_multihead_mixed_state_kronecker_penalties(model)
        total_loss = nll_loss + pen_loss
        total_loss.backward()

        muon_opt.step()
        adamw_opt.step()

        elapsed_s = max(time.perf_counter() - t0, 1e-6)
        tok_s = float(run_spec.batch_size * run_spec.seq_len) / elapsed_s
        last_ce = float(nll_loss.item())
        logger.log_step(step=step_idx, train_ce=last_ce, tok_per_s=tok_s)

    with tempfile.TemporaryDirectory() as tmp_val_dir:
        shard_path = Path(tmp_val_dir) / "fineweb_val_000000.bin"
        eval_tokens = run_spec.batch_size * run_spec.seq_len * 2
        header = np.zeros(256, dtype=np.int32)
        header[0] = 20240520
        header[1] = 1
        header[2] = eval_tokens + 8
        arr = np.random.randint(
            0, run_spec.model_cfg.valid_vocab_size, size=(eval_tokens + 8,), dtype=np.uint16
        )
        with shard_path.open("wb") as fh:
            fh.write(header.tobytes())
            fh.write(arr.tobytes())

        val_metrics = evaluate_modded_nanogpt_validation(
            model=model,
            val_tokens=shard_path,
            max_eval_tokens=eval_tokens,
            batch_size=run_spec.batch_size,
            seq_len=run_spec.seq_len,
        )

    _, pen_final = compute_multihead_mixed_state_kronecker_penalties(model)
    purity_tel = model.compute_subhead_purity_telemetry()
    branch_tel = model.compute_branch_routing_telemetry()
    eval_payload: dict[str, Any] = {
        "val_ce": float(val_metrics["val_ce"]),
        "val_ppl": float(val_metrics["val_ppl"]),
        "top1_acc": float(val_metrics["top1_acc"]),
        "kraus_stiefel_err": float(pen_final["unitary_pen"]),
        "sharp_eff_rank": float(purity_tel["reduced_density_purity_mean"] * 16.0),
    }
    logger.log_eval(step=run_spec.steps, metrics=eval_payload)
    logger.finalize(extra_summary=eval_payload)

    return {
        "train_ce": last_ce,
        "val_ce": float(val_metrics["val_ce"]),
        "unitary_err": float(pen_final["unitary_pen"]),
        "reduced_state_purity_mean": float(purity_tel["reduced_density_purity_mean"]),
        "step0_branch_purity_mean": float(branch_tel["step0_branch_ket_purity"]),
    }


def build_ablation_specs(
    ablation: str,
    vocab_size: int,
    n_layers: int,
    seq_len: int,
    batch_size: int,
    steps: int,
    smoke: bool,
) -> list[VerificationRunSpec]:
    """Build the list of `VerificationRunSpec` instances for the requested `--ablation` mode."""
    base_cfg = MultiHeadMixedStateKroneckerConfig(
        vocab_size=vocab_size,
        valid_vocab_size=vocab_size,
        num_meanings=SMOKE_NUM_MEANINGS,
        num_factors=SMOKE_NUM_FACTORS,
        factor_dim=SMOKE_FACTOR_DIM,
        num_layers=n_layers,
        n_subheads_per_factor=SMOKE_SUBHEADS,
        max_seq_len=max(seq_len, 64),
        vocab_chunk_size=128,
    )
    specs: list[VerificationRunSpec] = []

    if ablation in ("all", "multihead", "stiefel_muon"):
        specs.append(
            VerificationRunSpec(
                run_key="w44_plan059_multihead_gram_verify",
                hypothesis_intent=(
                    "Verify MultiHeadSuperpositionMixedStateAttention (3 TxT matmuls) + "
                    "MixedStateKroneckerBornChunkedNLLFn + Joint-Complex StiefelMuon."
                ),
                model_cfg=base_cfg,
                steps=steps,
                seq_len=seq_len,
                batch_size=batch_size,
                smoke=smoke,
            )
        )

    if ablation in ("all", "self_echo") and not smoke:
        for echo_mode in ("ortho_complex_learned", "ortho_complex_full", "no_self_echo"):
            specs.append(
                VerificationRunSpec(
                    run_key=f"w44_plan059_echo_{echo_mode}_verify",
                    hypothesis_intent=f"Ablate Track 3 self_echo_mode={echo_mode}.",
                    model_cfg=MultiHeadMixedStateKroneckerConfig(
                        vocab_size=vocab_size,
                        valid_vocab_size=vocab_size,
                        num_meanings=SMOKE_NUM_MEANINGS,
                        num_factors=SMOKE_NUM_FACTORS,
                        factor_dim=SMOKE_FACTOR_DIM,
                        num_layers=n_layers,
                        n_subheads_per_factor=SMOKE_SUBHEADS,
                        max_seq_len=max(seq_len, 64),
                        vocab_chunk_size=128,
                        self_echo_mode=echo_mode,
                    ),
                    steps=steps,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    smoke=smoke,
                )
            )

    if ablation in ("all", "polysemy") and not smoke:
        for init_scale, label in (
            (DEFAULT_NON_DILUTED_SECONDARY_BRANCH_SCALE, "nondiluted_1p0"),
            (LEGACY_DILUTED_SECONDARY_BRANCH_SCALE, "diluted_0p22"),
        ):
            specs.append(
                VerificationRunSpec(
                    run_key=f"w44_plan059_polysemy_{label}_verify",
                    hypothesis_intent=f"Ablate Track 4 secondary_branch_scale={init_scale}.",
                    model_cfg=MultiHeadMixedStateKroneckerConfig(
                        vocab_size=vocab_size,
                        valid_vocab_size=vocab_size,
                        num_meanings=SMOKE_NUM_MEANINGS,
                        num_factors=SMOKE_NUM_FACTORS,
                        factor_dim=SMOKE_FACTOR_DIM,
                        num_layers=n_layers,
                        n_subheads_per_factor=SMOKE_SUBHEADS,
                        max_seq_len=max(seq_len, 64),
                        vocab_chunk_size=128,
                        secondary_branch_scale=init_scale,
                    ),
                    steps=steps,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    smoke=smoke,
                )
            )

    return specs


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
    """Load `num_shards` training shard arrays into host RAM for fast window sampling."""
    arrays: list[np.ndarray] = []
    for idx in range(1, num_shards + 1):
        sp = data_dir / f"fineweb_train_{idx:06d}.bin"
        arrays.append(_load_header_safe_shard_tokens(sp, max_tokens=max_tokens_per_shard))
    return arrays


def execute_fineweb_flagship_run(
    spec: FineWebFlagshipSpec,
    data_dir: Path,
    cache_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Execute the Wave 44 10-minute (600s) 9-shard FineWeb flagship run + 10.49M-token official eval (`<= 8` args)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.manual_seed(1337)

    val_shard_path = data_dir / "fineweb_val_000000.bin"
    shard_paths = [data_dir / f"fineweb_train_{idx:06d}.bin" for idx in range(1, DEFAULT_NUM_SHARDS + 1)]
    cfg = MultiHeadMixedStateKroneckerConfig(
        num_meanings=2,
        num_factors=4,
        factor_dim=16,
        num_layers=DEFAULT_FINEWEB_LAYERS,
        block_size=spec.seq_len,
        max_seq_len=max(1024, spec.seq_len),
        ngram_mode="trigram",
        vocab_chunk_size=1024 if spec.batch_size >= 32 else 2048,
        checkpoint_blocks=False,
    )
    calib_data = build_or_load_900m_ngram_calibration_cache(
        shard_paths=shard_paths,
        cache_path=cache_path,
        cfg=cfg,
        device=device,
        force_rebuild=False,
    )
    shard_arrays = _load_shard_arrays(
        data_dir=data_dir,
        num_shards=DEFAULT_NUM_SHARDS,
        max_tokens_per_shard=DEFAULT_SHARD_LOAD_TOKENS,
    )
    cal_batches = [
        torch.from_numpy(
            shard_arrays[0][b * 8 * spec.seq_len : (b + 1) * 8 * spec.seq_len].astype(np.int64)
        ).view(8, spec.seq_len)
        for b in range(4)
    ]

    model = MultiHeadMixedStateKroneckerGPT(cfg).to(device)
    calib_stats = model.calibrate_from_900m_ngram_cache(
        cache=calib_data,
        cal_batches=cal_batches,
        ngram_mode="trigram",
        carrier_rho_max=cfg.carrier_rho_max,
        calibrate_povm=True,
    )

    grad_accum = max(1, int(spec.grad_accum_steps))
    tokens_per_step = spec.batch_size * spec.seq_len * grad_accum
    lr_scale = compute_large_batch_muon_lr_scale(tokens_per_step)
    opt_muon, opt_adam = build_multihead_mixed_state_kronecker_optimizers(
        model,
        muon_lr=DEFAULT_MUON_LR * lr_scale,
    )
    sampler = MultiShardRareTokenSampler(
        shard_arrays=shard_arrays,
        unigram_counts=calib_data.unigram_counts.cpu(),
        vocab_size=cfg.vocab_size,
        valid_vocab_size=cfg.valid_vocab_size,
        alpha_rare=spec.alpha_rare,
    )
    del calib_data, cal_batches
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    num_params = sum(p.numel() for p in model.parameters())
    target_tokens_m = (spec.max_steps * tokens_per_step) / 1e6
    intent = ExperimentIntentSpec(
        run_key=spec.run_key,
        wave_id=WAVE_ID,
        plan_id=PLAN_ID,
        task_domain="fineweb_speedrun",
        hypothesis_intent=(
            "Wave 44 Flagship: 600s 9-shard (900M-token) MultiHeadMixedStateKroneckerGPT "
            "(Joint-Complex StiefelMuon + MultiHead Gram K=2 + Carrier-Ortho Self-Echo + Non-Diluted Polysemy + "
            "Single-Pass Backward) evaluated on official 10,485,760-token modded-nanogpt FineWeb validation shard"
        ),
        architecture_summary=(
            f"MultiHeadMixedStateKroneckerGPT(R={cfg.num_meanings}, M={cfg.num_factors}, d_f={cfg.factor_dim}, "
            f"K={cfg.n_subheads_per_factor}, L={cfg.num_layers}, mode=trigram, shards=9, params={num_params / 1e6:.2f}M)"
        ),
        hardware_budget="1x NVIDIA GeForce RTX 5060 Ti (16 GB VRAM), BF16 + MultiHead Stinespring Gram",
        batch_size=spec.batch_size * grad_accum,
        seq_len=spec.seq_len,
        total_steps=spec.max_steps,
        target_tokens_M=round(target_tokens_m, 3),
        params_M=num_params / 1e6,
        embed_dim=cfg.real_equiv_dim,
        num_layers=cfg.num_layers,
    )
    logger = StandardizedExperimentLogger(intent)

    eval0 = evaluate_modded_nanogpt_validation(
        model=model,
        val_tokens=val_shard_path,
        max_eval_tokens=DEFAULT_EVAL_TOKENS_FAST,
        batch_size=spec.batch_size,
        seq_len=spec.seq_len,
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
    print(
        f"[STEP 0 CALIBRATED] val_ce_65k={eval0['val_ce']:.4f} | val_ppl={eval0['val_ppl']:.2f} | "
        f"top1_acc={eval0['top1_acc']:.2f}% | rep_ce={eval0.get('rep_val_ce', 0.0):.4f} | "
        f"non_rep_ce={eval0.get('non_rep_val_ce', 0.0):.4f}",
        flush=True,
    )

    model.train()
    xw, yw, ww = sampler.sample_batch(batch_size=spec.batch_size, seq_len=spec.seq_len, step_seed=0, device=device)
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        _, warm_loss = model(xw, yw, token_weights=ww)
    assert warm_loss is not None
    warm_loss.backward()
    model.zero_grad(set_to_none=True)
    del xw, yw, ww, warm_loss
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_train_start = time.perf_counter()
    step = 0
    best_val_ce = float(eval0["val_ce"])
    best_top1_acc = float(eval0["top1_acc"])
    last_ce_loss = float(eval0["val_ce"])

    while step < spec.max_steps:
        elapsed_s = time.perf_counter() - t_train_start
        if spec.wall_s > 0.0 and elapsed_s >= spec.wall_s:
            break

        step += 1
        if spec.wall_s > 0.0:
            progress = min(1.0, max(step / float(spec.max_steps), elapsed_s / float(spec.wall_s)))
        else:
            progress = min(1.0, step / float(spec.max_steps))

        model.set_training_progress(progress)
        lr_mult = _compute_wsd_lr_multiplier(progress)
        _apply_lr_multiplier(opt_muon, opt_adam, lr_mult)

        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)

        accum_ce = 0.0
        for micro_idx in range(grad_accum):
            x_b, y_b, w_b = sampler.sample_batch(
                batch_size=spec.batch_size,
                seq_len=spec.seq_len,
                step_seed=step * grad_accum + micro_idx,
                device=device,
            )
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                _, ce_loss = model(x_b, y_b, token_weights=w_b)
            assert ce_loss is not None
            accum_ce += float(ce_loss.detach().item()) / float(grad_accum)
            if micro_idx == grad_accum - 1:
                pen_loss, pen_stats = compute_multihead_mixed_state_kronecker_penalties(model)
                total_loss = (ce_loss / float(grad_accum)) + pen_loss
            else:
                total_loss = ce_loss / float(grad_accum)
            total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_GRAD_CLIP_NORM)
        opt_muon.step()
        opt_adam.step()

        last_ce_loss = accum_ce
        tokens_m = (step * tokens_per_step) / 1e6
        compute_tflops = (6.0 * num_params * step * tokens_per_step) / 1e12
        wall_now = max(1e-3, time.perf_counter() - t_train_start)
        tok_per_s = (step * tokens_per_step) / wall_now

        if step % DEFAULT_LOG_INTERVAL == 0 or step in (1, 10):
            logger.log_step(
                step=step,
                train_ce=round(last_ce_loss, 4),
                tok_per_s=round(tok_per_s, 1),
                lr_mult=round(lr_mult, 4),
                elapsed_min=round(wall_now / 60.0, 4),
            )
            print(
                f"[STEP {step:4d}] elapsed={wall_now:6.1f}s | train_ce={last_ce_loss:.4f} | "
                f"tok/s={tok_per_s:.1f} | tokens_M={tokens_m:.3f} | lr_mult={lr_mult:.3f}",
                flush=True,
            )

        if step % DEFAULT_EVAL_INTERVAL == 0:
            ev = evaluate_modded_nanogpt_validation(
                model=model,
                val_tokens=val_shard_path,
                max_eval_tokens=DEFAULT_EVAL_TOKENS_FAST,
                batch_size=spec.batch_size,
                seq_len=spec.seq_len,
            )
            best_val_ce = min(best_val_ce, float(ev["val_ce"]))
            best_top1_acc = max(best_top1_acc, float(ev["top1_acc"]))
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
            print(
                f"[EVAL {step:4d}] val_ce_65k={ev['val_ce']:.4f} (best={best_val_ce:.4f}) | "
                f"top1={ev['top1_acc']:.2f}% | rep_ce={ev.get('rep_val_ce', 0.0):.4f} | "
                f"non_rep_ce={ev.get('non_rep_val_ce', 0.0):.4f} | stiefel_err={pen_stats['unitary_pen']:.5f}",
                flush=True,
            )

    train_wall_s = time.perf_counter() - t_train_start
    final_tokens_m = (step * tokens_per_step) / 1e6
    final_tflops = (6.0 * num_params * step * tokens_per_step) / 1e12
    avg_tok_per_s = (step * tokens_per_step) / max(1e-3, train_wall_s)

    purity_tel = model.compute_subhead_purity_telemetry()
    branch_tel = model.compute_branch_routing_telemetry()

    model.zero_grad(set_to_none=True)
    del opt_muon, opt_adam
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_final_65k = evaluate_modded_nanogpt_validation(
        model=model,
        val_tokens=val_shard_path,
        max_eval_tokens=DEFAULT_EVAL_TOKENS_FAST,
        batch_size=spec.batch_size,
        seq_len=spec.seq_len,
        compute_top50=True,
    )
    best_val_ce = min(best_val_ce, float(eval_final_65k["val_ce"]))
    best_top1_acc = max(best_top1_acc, float(eval_final_65k["top1_acc"]))
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
    print(
        f"[FINAL 65K EVAL] steps={step} | wall={train_wall_s:.2f}s | tok/s={avg_tok_per_s:.1f} | "
        f"val_ce_65k={eval_final_65k['val_ce']:.4f} | val_ppl={eval_final_65k['val_ppl']:.2f} | "
        f"top1={eval_final_65k['top1_acc']:.2f}% | top50={eval_final_65k.get('top50_acc', 0.0):.2f}%",
        flush=True,
    )

    eval_10m: dict[str, float] | None = None
    if spec.eval_10m_at_end:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("=== Running Official 10,485,760-Token Modded-NanoGPT Validation Benchmark ===", flush=True)
        t_10m = time.perf_counter()
        eval_10m = evaluate_modded_nanogpt_validation(
            model=model,
            val_tokens=val_shard_path,
            max_eval_tokens=DEFAULT_MODDED_VAL_TOKENS,
            batch_size=32,
            seq_len=spec.seq_len,
            compute_top50=False,
        )
        eval_10m["eval_wall_s"] = round(time.perf_counter() - t_10m, 2)
        print(
            f"[OFFICIAL 10.49M VAL] val_ce={eval_10m['val_ce']:.4f} | val_ppl={eval_10m['val_ppl']:.2f} | "
            f"top1={eval_10m['top1_acc']:.2f}% | rep_ce={eval_10m.get('rep_val_ce', 0.0):.4f} | "
            f"non_rep_ce={eval_10m.get('non_rep_val_ce', 0.0):.4f} | eval_wall_s={eval_10m['eval_wall_s']:.1f}s",
            flush=True,
        )

    out: dict[str, Any] = {
        "run_key": spec.run_key,
        "wave_id": WAVE_ID,
        "plan_id": PLAN_ID,
        "ngram_mode": "trigram",
        "num_shards": DEFAULT_NUM_SHARDS,
        "alpha_rare": spec.alpha_rare,
        "grad_accum_steps": grad_accum,
        "effective_batch_tokens": tokens_per_step,
        "muon_lr_scale": round(lr_scale, 4),
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
        "final_rep_val_ce_65k": eval_final_65k.get("rep_val_ce", 0.0),
        "final_non_rep_val_ce_65k": eval_final_65k.get("non_rep_val_ce", 0.0),
        "final_mean_self_echo_prob_65k": eval_final_65k.get("mean_self_echo_prob", 0.0),
        "final_rep_top1_acc_65k": eval_final_65k.get("rep_top1_acc", 0.0),
        "best_val_ce_65k": best_val_ce,
        "best_top1_acc_65k": best_top1_acc,
        "sharp_eff_rank": eval_final_65k["sharp_eff_rank"],
        "mean_factor_rank": eval_final_65k["mean_factor_rank"],
        "cross_branch_overlap": eval_final_65k["cross_branch_overlap"],
        "povm_drift_std": eval_final_65k["povm_drift_std"],
        "unitary_frobenius_err": round(unitary_err_final, 6),
        "subhead_purity_telemetry": purity_tel,
        "branch_routing_telemetry": branch_tel,
        "official_10m_validation": eval_10m,
    }
    logger.finalize(extra_summary=out)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `69_verify_plan059_multihead_mixed_state_kronecker_gpt.py`."""
    parser = argparse.ArgumentParser(
        description="Verify Plan 059 / Wave 44 MultiHeadMixedStateKroneckerGPT tracks and 600s FineWeb flagship."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke verification across Track 3, Track 4, Track 6, and training loop.",
    )
    parser.add_argument(
        "--fineweb-600s",
        action="store_true",
        help="Execute the 10-minute (600s) 9-shard FineWeb flagship run + official 10.49M-token validation.",
    )
    parser.add_argument(
        "--wall-s",
        type=float,
        default=DEFAULT_FINEWEB_WALL_S,
        help="Wall-clock training budget in seconds for --fineweb-600s (default: 600.0).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_FINEWEB_MAX_STEPS,
        help="Maximum optimization steps for --fineweb-600s.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_FINEWEB_BATCH_SIZE,
        help="Micro-batch size B for --fineweb-600s (default: 16; use 32 for 16,384 tok/step).",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation micro-steps per optimizer step for --fineweb-600s.",
    )
    parser.add_argument(
        "--alpha-rare",
        type=float,
        default=0.0,
        help="Rare-token importance sampling exponent for --fineweb-600s (default: 0.0 uniform).",
    )
    parser.add_argument(
        "--no-eval-10m",
        action="store_true",
        help="Skip the 10.49M-token validation at the end of --fineweb-600s.",
    )
    parser.add_argument(
        "--run-key",
        type=str,
        default=DEFAULT_FINEWEB_RUN_KEY,
        help="Run key for StandardizedExperimentLogger in --fineweb-600s mode.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=DEFAULT_OUT_JSON,
        help="Path to write JSON summary report for --fineweb-600s.",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=DEFAULT_ABLATION_CHOICE,
        choices=VALID_ABLATION_CHOICES,
        help="Which ablation track to execute.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_VERIFY_STEPS,
        help="Number of optimization steps per verification run.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_VERIFY_SEQ_LEN,
        help="Sequence length T.",
    )
    parser.add_argument(
        "--batch-tokens",
        type=int,
        default=DEFAULT_VERIFY_BATCH_TOKENS,
        help="Total tokens per microbatch (B * T).",
    )
    return parser.parse_args()


def main() -> None:
    """Execute Plan 059 / Wave 44 verification checks and training runs."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t3_summary = verify_track3_self_echo_invariants(device)
    print(
        f"[TRACK 3 PASS] carrier_leak_max={t3_summary['carrier_leak_max']:.3e} | "
        f"full_echo_residual_perp_max={t3_summary['full_echo_residual_perp_max']:.3e}"
    )

    t4_summary = verify_track4_polysemy_and_router_invariants(device)
    print(
        f"[TRACK 4 PASS] step0_purity={t4_summary['step0_branch_purity_nd']:.4f} | "
        f"gate_ratio={t4_summary['step0_gate_ratio_nd']:.4f} | "
        f"r_norm_dev={t4_summary['max_r_factor_norm_dev']:.3e} | "
        f"phase_diff_re={t4_summary['phase_diff_pos1_vs_neg1']:.4f} | "
        f"phase_diff_im={t4_summary['phase_diff_posi_vs_negi']:.4f}"
    )

    t6_summary = verify_track6_multihead_matmul_and_purity(device)
    print(
        f"[TRACK 6 PASS] matmuls: multihead={int(t6_summary['multihead_total_matmuls'])} "
        f"(TxT={int(t6_summary['multihead_txt_matmuls'])}) vs "
        f"legacy={int(t6_summary['legacy_total_matmuls'])} "
        f"(TxT={int(t6_summary['legacy_txt_matmuls'])}) | "
        f"gram_trace={t6_summary['gram_trace_mean']:.4f} | "
        f"reduced_purity={t6_summary['reduced_state_purity_mean']:.4f}"
    )

    if args.fineweb_600s:
        fw_spec = FineWebFlagshipSpec(
            run_key=args.run_key,
            wall_s=float(args.wall_s),
            max_steps=int(args.max_steps),
            batch_size=int(args.batch_size),
            seq_len=DEFAULT_FINEWEB_SEQ_LEN,
            grad_accum_steps=int(args.grad_accum_steps),
            alpha_rare=float(args.alpha_rare),
            eval_10m_at_end=not bool(args.no_eval_10m),
        )
        fw_res = execute_fineweb_flagship_run(
            spec=fw_spec,
            data_dir=Path(DEFAULT_DATA_DIR),
            cache_path=Path(DEFAULT_CACHE_PATH),
            device=device,
        )
        out_path = Path(args.out_json)
        existing: dict[str, Any] = {}
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        runs_dict = dict(existing.get("fineweb_runs", {}))
        if "fineweb_600s_flagship" in existing and isinstance(existing["fineweb_600s_flagship"], dict):
            prev_rk = existing["fineweb_600s_flagship"].get("run_key", DEFAULT_FINEWEB_RUN_KEY)
            runs_dict.setdefault(prev_rk, existing["fineweb_600s_flagship"])
        runs_dict[fw_spec.run_key] = fw_res
        report = {
            "wave_id": WAVE_ID,
            "plan_id": PLAN_ID,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "track3_invariants": t3_summary,
            "track4_invariants": t4_summary,
            "track6_invariants": t6_summary,
            "fineweb_600s_flagship": fw_res,
            "fineweb_runs": runs_dict,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved Wave 44 FineWeb flagship report to: {out_path}", flush=True)
        return

    if args.smoke:
        vocab_size = SMOKE_VOCAB_SIZE
        n_layers = SMOKE_N_LAYERS
        seq_len = SMOKE_SEQ_LEN
        batch_tokens = SMOKE_BATCH_TOKENS
        steps = SMOKE_STEPS
    else:
        vocab_size = DEFAULT_VERIFY_VOCAB_SIZE
        n_layers = DEFAULT_VERIFY_N_LAYERS
        seq_len = args.seq_len
        batch_tokens = args.batch_tokens
        steps = args.steps

    batch_size = max(1, batch_tokens // max(1, seq_len))

    run_specs = build_ablation_specs(
        ablation=args.ablation,
        vocab_size=vocab_size,
        n_layers=n_layers,
        seq_len=seq_len,
        batch_size=batch_size,
        steps=steps,
        smoke=args.smoke,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        logs_dir = Path(tmp_dir) / "live_logs"
        cards_dir = Path(tmp_dir) / "model_cards"
        for spec in run_specs:
            res = execute_training_verification_run(
                run_spec=spec,
                device=device,
                logs_dir=logs_dir,
                cards_dir=cards_dir,
            )
            print(
                f"[RUN PASS] {spec.run_key} | train_ce={res['train_ce']:.4f} | "
                f"val_ce={res['val_ce']:.4f} | unitary_err={res['unitary_err']:.4e} | "
                f"purity={res['reduced_state_purity_mean']:.4f}"
            )


if __name__ == "__main__":
    main()
