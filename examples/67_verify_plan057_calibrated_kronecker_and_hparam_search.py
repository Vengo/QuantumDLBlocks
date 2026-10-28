#!/usr/bin/env python3
"""Wave 42 (`EXPERIMENT_PLAN_057.md`): Calibrated Readout Operators & Tied Kronecker Wavefunctions + HParam Search.

Executes and verifies all four hypotheses (`H1`-`H4`) of `EXPERIMENT_PLAN_057.md`:
1. `H1`: Step-0 Readout Operator Calibration on Random-Init Attention Outputs (`Zero Static External Prior Buffer`)
   across `uncalibrated_haar`, `carrier_angle_only`, `trainable_povm_only`, and `hybrid_procrustes_carrier_povm`
   (`carrier_alignment_fidelity == 1.0 ± 1e-5`, `unitary_frobenius_err < 1e-5`, `D_KL < 0.005` nats / `> 500x`
   reduction, `val_ce_step0 < 8.15` nats, `R_factor >= 28/32`, `R_emb >= 220/256`, and 5-input chunked autograd
   parity `< 1e-5` fwd / `< 1e-4` bwd).
2. `H2`: 150-Step Ablation of `uncalibrated_pure_born`, `frozen_unigram_buffer_w41`, and
   `trainable_calibrated_hybrid_w42` (`>= 0.20` nats over uncalibrated, lower/equal `val_ce` and higher/equal
   `top1_acc` than frozen W41 buffer, active `povm_drift_std > 0`, `R_factor >= 25/32`, `R_emb >= 180/256`).
3. `H3`: Automated Multi-Configuration Hyperparameter Sweep (`scripts/kronecker_born_hyperparam_search.py`)
   across `>= 6` candidate configurations (`>= 0.10` nats improvement over default Wave 41 hyperparameters).
4. `H4`: 5-Minute (`300s`) Head-to-Head Verification of Winning `CalibratedKroneckerBornGPT` (`Wave 42` vs.
   `Wave 41` `val_ce = 7.2066`, `top1_acc = 11.06%`) streamed via `StandardizedExperimentLogger`.
"""

from __future__ import annotations

import argparse
import gc
import inspect
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
from qdlblocks.layers.packed_complex_register_density_attn import complex_l2_normalize  # noqa: E402
from qdlblocks.models.calibrated_kronecker_born_gpt import (  # noqa: E402
    CalibratedKroneckerBornChunkedNLLFn,
    CalibratedKroneckerBornGPT,
    CalibratedKroneckerConfig,
    DEFAULT_BRANCH_ORTH_WEIGHT,
    DEFAULT_CALIBRATION_BATCHES,
    DEFAULT_CALIBRATION_SEQ_LEN,
    DEFAULT_CAL_BOSON_P,
    DEFAULT_CAL_COINC_GAMMA,
    DEFAULT_CAL_HOLO_ALPHA,
    DEFAULT_CAL_KRON_BLEND,
    DEFAULT_FACTOR_DIM,
    DEFAULT_GATE_LR,
    DEFAULT_ISO_PENALTY_WEIGHT,
    DEFAULT_NUM_FACTORS,
    DEFAULT_NUM_MEANINGS,
    DEFAULT_POVM_ANCHOR_WEIGHT,
    DEFAULT_SPHERE_PENALTY_WEIGHT,
    DEFAULT_UNITARY_PENALTY_WEIGHT,
    _algebraic_positive_povm_weights,
    build_calibrated_kronecker_optimizers,
    compute_calibrated_kronecker_penalties,
    eval_calibrated_kronecker_born_probs,
)
from qdlblocks.models.kronecker_multimeaning_16x4_gpt import (  # noqa: E402
    DEFAULT_UNIGRAM_FLOOR_COUNT,
    KroneckerMultiMeaning16x4GPT,
    KroneckerMultiMeaningConfig,
    _algebraic_positive_unit_l2,
    _load_header_safe_shard_tokens,
    build_kronecker_multimeaning_optimizers,
    compute_kronecker_manifold_and_isotropy_penalties,
)
from qdlblocks.models.tied_holo_born_complex_gpt import (  # noqa: E402
    DEFAULT_RARE_ALPHA,
    RareTokenImportanceWindowSampler,
)
from scripts.kronecker_born_hyperparam_search import (  # noqa: E402
    DEFAULT_SEARCH_JSON,
    HyperparamTrialSpec,
    SearchRunnerConfig,
    apply_progress_lr_schedule,
    audit_step0_readout_calibration_modes,
    build_validation_batches,
    evaluate_calibrated_kronecker_model,
    get_default_candidate_trials,
    run_hyperparam_search_suite,
)

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95, device=0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_057.md (Wave 42 Constants Ledger)
# ==============================================================================
DATA_DIR: Path = ROOT / ".cache" / "fineweb_100M"
DEFAULT_OUTPUT_JSON: Path = ROOT / ".cache" / "hypothesis_experiments_057.json"
WEBUI_RELOAD_URL: str = "http://127.0.0.1:8765/api/reload"

VOCAB_SIZE_PADDED: int = 50304
VALID_VOCAB_SIZE: int = 50257
SLICE_2M_TOKENS: int = 2_000_000
SLICE_30M_TOKENS: int = 30_000_000
SLICE_100M_TOKENS: int = 100_000_000
VAL_SLICE_TOKENS: int = 1_048_576

# H1 thresholds & parameters
H1_AUTOGRAD_N_TOK: int = 32
H1_AUTOGRAD_V_EFF: int = 256
H1_AUTOGRAD_CHUNK_SIZE: int = 64
H1_MAX_FIDELITY_ERR: float = 1e-5
H1_MAX_UNITARY_ERR: float = 1e-5
H1_MAX_CALIBRATED_KL: float = 0.005
H1_MIN_KL_REDUCTION_X: float = 500.0
H1_MAX_STEP0_VAL_CE: float = 8.15
H1_MIN_FACTOR_RANK: float = 28.0
H1_MIN_EMB_RANK: float = 220.0
H1_MAX_PROB_SUM_ERR: float = 1e-5
H1_MAX_FWD_DIFF: float = 1e-5
H1_MAX_BWD_DIFF: float = 1e-4

# H2 thresholds & parameters
H2_ABLATION_STEPS: int = 150
H2_BATCH_SIZE: int = 8
H2_SEQ_LEN: int = 128
H2_NUM_LAYERS: int = 4
H2_VAL_BATCHES: int = 6
H2_MIN_NATS_VS_UNCAL: float = 0.20
H2_MIN_FACTOR_RANK: float = 25.0
H2_MIN_EMB_RANK: float = 180.0

# H3 thresholds & parameters
H3_STEPS_PER_TRIAL: int = 120
H3_MIN_TRIALS: int = 6
H3_MIN_NATS_IMPROVEMENT: float = 0.10

# H4 live RTX 5060 Ti run constants & Wave 41 reference targets
H4_WAVE_ID: str = "W42"
H4_PLAN_ID: str = "EXPERIMENT_PLAN_057"
H4_BATCH_SIZE: int = 16
H4_SEQ_LEN: int = 512
H4_NUM_LAYERS: int = 6
H4_VAL_BATCH_SIZE: int = 4
H4_VAL_BATCHES: int = 6
H4_MAX_VRAM_MIB: float = 14500.0
H4_NORM_MEDIAN_MIN: float = 0.95
H4_NORM_MEDIAN_MAX: float = 1.15
WAVE41_REF_VAL_CE: float = 7.2066
WAVE41_REF_TOP1_ACC: float = 11.06
WAVE41_REF_TOP50_ACC: float = 42.38


def _ping_webui_reload(url: str = WEBUI_RELOAD_URL) -> None:
    """Trigger non-blocking reload on the local `:8765` WebUI daemon."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=1.0):
            pass
    except Exception:
        pass


@torch.no_grad()
def _evaluate_w41_baseline_model(
    model: KroneckerMultiMeaning16x4GPT,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    """Evaluate frozen-buffer `KroneckerMultiMeaning16x4GPT` (`Wave 41`) on `val_batches` (`<= 8` args)."""
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
    if was_training:
        model.train()
    return {
        "val_ce": round(val_ce, 4),
        "val_ppl": round(val_ppl, 2),
        "top1_acc": top1_pct,
        "top50_acc": top50_pct,
        "sharp_eff_rank": round(model.compute_sharp_eff_rank(), 2),
        "mean_factor_rank": round(model.compute_mean_factor_rank(), 2),
        "cross_branch_overlap": float(model.compute_cross_branch_overlap()),
        "povm_drift_std": 0.0,
    }


def run_h1_step0_calibration_and_autograd_probe(
    tokens_100m: np.ndarray,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    """Execute `H1`: Step-0 readout calibration across 4 modes and 5-input chunked autograd parity (`<= 8` args)."""
    cfg_runner = SearchRunnerConfig(
        steps_per_trial=0,
        batch_size=H2_BATCH_SIZE,
        seq_len=H2_SEQ_LEN,
        num_layers=H2_NUM_LAYERS,
        vocab_size=VOCAB_SIZE_PADDED,
        valid_vocab_size=VALID_VOCAB_SIZE,
        val_batches=H2_VAL_BATCHES,
        log_to_webui=False,
    )
    audit = audit_step0_readout_calibration_modes(tokens_100m, val_batches, cfg_runner, device)
    modes = audit["modes"]

    # Verify zero exp / softmax / logsumexp in readout source
    for obj in (
        eval_calibrated_kronecker_born_probs,
        CalibratedKroneckerBornChunkedNLLFn,
        _algebraic_positive_povm_weights,
    ):
        src = inspect.getsource(obj)
        for forbidden in ("torch.exp(", ".exp(", "softmax(", "logsumexp("):
            if forbidden in src:
                raise AssertionError(f"Forbidden `{forbidden}` found in {obj}")
    zero_exp_softmax = True

    # 5-input chunked autograd parity check
    gen = torch.Generator(device=device)
    gen.manual_seed(20261026)
    phi_re, phi_im = complex_l2_normalize(
        torch.randn((H1_AUTOGRAD_N_TOK, 2, 4, 16), generator=gen, device=device, dtype=torch.float32),
        torch.randn((H1_AUTOGRAD_N_TOK, 2, 4, 16), generator=gen, device=device, dtype=torch.float32),
        dim=-1,
    )
    psi_re, psi_im = complex_l2_normalize(
        torch.randn((H1_AUTOGRAD_V_EFF, 2, 4, 16), generator=gen, device=device, dtype=torch.float32),
        torch.randn((H1_AUTOGRAD_V_EFF, 2, 4, 16), generator=gen, device=device, dtype=torch.float32),
        dim=-1,
    )
    phi_0 = torch.cat([phi_re, phi_im], dim=-1)
    psi_0 = torch.cat([psi_re, psi_im], dim=-1)
    beta_0 = _algebraic_positive_unit_l2(
        torch.randn((H1_AUTOGRAD_N_TOK, 2), generator=gen, device=device, dtype=torch.float32)
    )
    alpha_0 = _algebraic_positive_unit_l2(
        torch.randn((H1_AUTOGRAD_V_EFF, 2), generator=gen, device=device, dtype=torch.float32)
    )
    povm_0 = _algebraic_positive_povm_weights(
        torch.randn((H1_AUTOGRAD_V_EFF,), generator=gen, device=device, dtype=torch.float32)
    )
    targets = torch.randint(0, H1_AUTOGRAD_V_EFF, (H1_AUTOGRAD_N_TOK,), generator=gen, device=device)
    tw = torch.linspace(0.6, 1.4, H1_AUTOGRAD_N_TOK, device=device, dtype=torch.float32)
    hp = torch.tensor(
        [DEFAULT_CAL_BOSON_P, DEFAULT_CAL_HOLO_ALPHA, DEFAULT_CAL_KRON_BLEND, DEFAULT_CAL_COINC_GAMMA],
        device=device,
        dtype=torch.float32,
    )

    phi_r = phi_0.detach().clone().requires_grad_(True)
    psi_r = psi_0.detach().clone().requires_grad_(True)
    beta_r = beta_0.detach().clone().requires_grad_(True)
    alpha_r = alpha_0.detach().clone().requires_grad_(True)
    povm_r = povm_0.detach().clone().requires_grad_(True)

    probs = eval_calibrated_kronecker_born_probs(phi_r, psi_r, beta_r, alpha_r, povm_r, hp)
    max_prob_sum_err = float((probs.sum(dim=-1) - 1.0).abs().max().item())
    tw_norm = tw / tw.sum()
    loss_ref = -(tw_norm * torch.log(probs.gather(1, targets.unsqueeze(1)).squeeze(1))).sum()
    loss_ref.backward()

    phi_c = phi_0.detach().clone().requires_grad_(True)
    psi_c = psi_0.detach().clone().requires_grad_(True)
    beta_c = beta_0.detach().clone().requires_grad_(True)
    alpha_c = alpha_0.detach().clone().requires_grad_(True)
    povm_c = povm_0.detach().clone().requires_grad_(True)

    saved_chunk = CalibratedKroneckerBornChunkedNLLFn.vocab_chunk_size
    CalibratedKroneckerBornChunkedNLLFn.vocab_chunk_size = H1_AUTOGRAD_CHUNK_SIZE
    try:
        loss_chunk = CalibratedKroneckerBornChunkedNLLFn.apply(
            phi_c, psi_c, beta_c, alpha_c, povm_c, targets, hp, tw
        )
        loss_chunk.backward()
    finally:
        CalibratedKroneckerBornChunkedNLLFn.vocab_chunk_size = saved_chunk

    fwd_diff = float((loss_chunk.detach() - loss_ref.detach()).abs().item())
    bwd_diffs = {
        "phi_factors": float((phi_c.grad - phi_ref_g).abs().max().item())
        for phi_ref_g in (phi_r.grad,)
        if phi_c.grad is not None and phi_ref_g is not None
    }
    bwd_diffs["psi_factors"] = float((psi_c.grad - psi_r.grad).abs().max().item())  # type: ignore[union-attr]
    bwd_diffs["branch_beta"] = float((beta_c.grad - beta_r.grad).abs().max().item())  # type: ignore[union-attr]
    bwd_diffs["branch_alpha"] = float((alpha_c.grad - alpha_r.grad).abs().max().item())  # type: ignore[union-attr]
    bwd_diffs["povm_weights"] = float((povm_c.grad - povm_r.grad).abs().max().item())  # type: ignore[union-attr]
    max_bwd_diff = max(bwd_diffs.values())

    povm_m = modes["trainable_povm_only"]
    hyb_m = modes["hybrid_procrustes_carrier_povm"]
    passed = bool(
        abs(float(hyb_m["carrier_alignment_fidelity"]) - 1.0) <= H1_MAX_FIDELITY_ERR
        and float(hyb_m["unitary_frobenius_err"]) < H1_MAX_UNITARY_ERR
        and float(povm_m["step0_kl_to_target_unigram"]) < H1_MAX_CALIBRATED_KL
        and float(hyb_m["step0_kl_to_target_unigram"]) < H1_MAX_CALIBRATED_KL
        and float(audit["kl_reduction_factor_vs_uncalibrated"]) >= H1_MIN_KL_REDUCTION_X
        and float(povm_m["val_ce_step0"]) < H1_MAX_STEP0_VAL_CE
        and float(hyb_m["val_ce_step0"]) < H1_MAX_STEP0_VAL_CE
        and float(hyb_m["mean_factor_rank"]) >= H1_MIN_FACTOR_RANK
        and float(hyb_m["sharp_eff_rank"]) >= H1_MIN_EMB_RANK
        and zero_exp_softmax
        and max_prob_sum_err < H1_MAX_PROB_SUM_ERR
        and fwd_diff < H1_MAX_FWD_DIFF
        and max_bwd_diff < H1_MAX_BWD_DIFF
    )
    return {
        "passed": passed,
        "calibration_modes": modes,
        "kl_reduction_factor_vs_uncalibrated": audit["kl_reduction_factor_vs_uncalibrated"],
        "zero_exp_softmax_in_readout": zero_exp_softmax,
        "max_prob_sum_err": max_prob_sum_err,
        "autograd_fwd_diff": fwd_diff,
        "autograd_bwd_diffs": bwd_diffs,
        "max_bwd_diff": max_bwd_diff,
    }


def run_h2_trainable_povm_vs_frozen_buffer_ablation(
    work_gpu: torch.Tensor,
    tokens_100m: np.ndarray,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    best_spec: HyperparamTrialSpec,
    device: torch.device,
) -> dict[str, Any]:
    """Execute `H2`: 150-step ablation comparing `uncalibrated_pure_born`, `frozen_unigram_buffer_w41`, and `trainable_calibrated_hybrid_w42` (`<= 8` args)."""
    tokens_per_step = H2_BATCH_SIZE * H2_SEQ_LEN
    n_work = int(work_gpu.numel())
    variants: dict[str, Any] = {}

    # Arm 1: uncalibrated_pure_born (uniform frozen weights w_v = 1)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    torch.manual_seed(1337)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1337)
    cfg_uncal = CalibratedKroneckerConfig(
        vocab_size=VOCAB_SIZE_PADDED,
        valid_vocab_size=VALID_VOCAB_SIZE,
        num_meanings=DEFAULT_NUM_MEANINGS,
        num_factors=DEFAULT_NUM_FACTORS,
        factor_dim=DEFAULT_FACTOR_DIM,
        num_layers=H2_NUM_LAYERS,
        block_size=H2_SEQ_LEN,
        unigram_temp=0.0,
        carrier_rho_max=0.0,
    )
    m_uncal = CalibratedKroneckerBornGPT(cfg_uncal).to(device)
    m_uncal.povm_scale_logits.requires_grad_(False)
    muon_u, adam_u = build_calibrated_kronecker_optimizers(m_uncal, base_muon_lr=0.015, base_embed_lr=0.008)
    ev0_u = evaluate_calibrated_kronecker_model(m_uncal, val_batches)
    t0 = time.perf_counter()
    m_uncal.train()
    for step in range(1, H2_ABLATION_STEPS + 1):
        tau = float(step - 1) / float(H2_ABLATION_STEPS)
        apply_progress_lr_schedule(muon_u, adam_u, tau)
        muon_u.zero_grad(set_to_none=True)
        adam_u.zero_grad(set_to_none=True)
        off = ((step - 1) * tokens_per_step) % max(1, n_work - tokens_per_step - 2)
        seq = (work_gpu[off : off + tokens_per_step + 1].to(torch.int64) & 0xFFFF).clamp(0, VALID_VOCAB_SIZE - 1)
        x_b = seq[:-1].view(H2_BATCH_SIZE, H2_SEQ_LEN)
        y_b = seq[1:].view(H2_BATCH_SIZE, H2_SEQ_LEN)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, nll_u = m_uncal(x_b, targets=y_b, return_logits=False)
        assert nll_u is not None
        pen_u, _ = compute_calibrated_kronecker_penalties(m_uncal)
        (nll_u + pen_u).backward()
        torch.nn.utils.clip_grad_norm_(m_uncal.parameters(), 1.0)
        muon_u.step()
        adam_u.step()
    el_u = time.perf_counter() - t0
    evf_u = evaluate_calibrated_kronecker_model(m_uncal, val_batches)
    variants["uncalibrated_pure_born"] = {
        "step0_val_ce": ev0_u["val_ce"],
        "final_val_ce": evf_u["val_ce"],
        "final_val_ppl": evf_u["val_ppl"],
        "final_top1_acc": evf_u["top1_acc"],
        "final_top50_acc": evf_u["top50_acc"],
        "sharp_eff_rank": evf_u["sharp_eff_rank"],
        "mean_factor_rank": evf_u["mean_factor_rank"],
        "povm_drift_std": evf_u["povm_drift_std"],
        "elapsed_sec": round(el_u, 2),
    }
    del m_uncal, muon_u, adam_u

    # Arm 2: frozen_unigram_buffer_w41 (Wave 41 KroneckerMultiMeaning16x4GPT with static prior_weights)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    torch.manual_seed(1337)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1337)
    cfg_w41 = KroneckerMultiMeaningConfig(
        vocab_size=VOCAB_SIZE_PADDED,
        valid_vocab_size=VALID_VOCAB_SIZE,
        num_meanings=DEFAULT_NUM_MEANINGS,
        num_factors=DEFAULT_NUM_FACTORS,
        factor_dim=DEFAULT_FACTOR_DIM,
        num_layers=H2_NUM_LAYERS,
        block_size=H2_SEQ_LEN,
    )
    m_w41 = KroneckerMultiMeaning16x4GPT(cfg_w41).to(device)
    m_w41.initialize_unigram_prior_from_tokens(tokens_100m, floor_count=DEFAULT_UNIGRAM_FLOOR_COUNT)
    muon_41, adam_41 = build_kronecker_multimeaning_optimizers(m_w41)
    ev0_41 = _evaluate_w41_baseline_model(m_w41, val_batches)
    t0 = time.perf_counter()
    m_w41.train()
    for step in range(1, H2_ABLATION_STEPS + 1):
        tau = float(step - 1) / float(H2_ABLATION_STEPS)
        apply_progress_lr_schedule(muon_41, adam_41, tau)
        muon_41.zero_grad(set_to_none=True)
        adam_41.zero_grad(set_to_none=True)
        off = ((step - 1) * tokens_per_step) % max(1, n_work - tokens_per_step - 2)
        seq = (work_gpu[off : off + tokens_per_step + 1].to(torch.int64) & 0xFFFF).clamp(0, VALID_VOCAB_SIZE - 1)
        x_b = seq[:-1].view(H2_BATCH_SIZE, H2_SEQ_LEN)
        y_b = seq[1:].view(H2_BATCH_SIZE, H2_SEQ_LEN)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, nll_41 = m_w41(x_b, targets=y_b, return_logits=False)
        assert nll_41 is not None
        pen_41, _ = compute_kronecker_manifold_and_isotropy_penalties(m_w41)
        (nll_41 + pen_41).backward()
        torch.nn.utils.clip_grad_norm_(m_w41.parameters(), 1.0)
        muon_41.step()
        adam_41.step()
    el_41 = time.perf_counter() - t0
    evf_41 = _evaluate_w41_baseline_model(m_w41, val_batches)
    variants["frozen_unigram_buffer_w41"] = {
        "step0_val_ce": ev0_41["val_ce"],
        "final_val_ce": evf_41["val_ce"],
        "final_val_ppl": evf_41["val_ppl"],
        "final_top1_acc": evf_41["top1_acc"],
        "final_top50_acc": evf_41["top50_acc"],
        "sharp_eff_rank": evf_41["sharp_eff_rank"],
        "mean_factor_rank": evf_41["mean_factor_rank"],
        "povm_drift_std": 0.0,
        "elapsed_sec": round(el_41, 2),
    }
    del m_w41, muon_41, adam_41

    # Arm 3: trainable_calibrated_hybrid_w42 (Wave 42 CalibratedKroneckerBornGPT)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    torch.manual_seed(1337)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1337)
    cfg_w42 = CalibratedKroneckerConfig(
        vocab_size=VOCAB_SIZE_PADDED,
        valid_vocab_size=VALID_VOCAB_SIZE,
        num_meanings=DEFAULT_NUM_MEANINGS,
        num_factors=DEFAULT_NUM_FACTORS,
        factor_dim=DEFAULT_FACTOR_DIM,
        num_layers=H2_NUM_LAYERS,
        block_size=H2_SEQ_LEN,
        boson_p=best_spec.boson_p,
        kron_blend=best_spec.kron_blend,
        coinc_gamma=best_spec.coinc_gamma,
        unigram_temp=best_spec.unigram_temp,
        carrier_rho_max=best_spec.carrier_rho_max,
        povm_anchor_weight=DEFAULT_POVM_ANCHOR_WEIGHT,
    )
    m_w42 = CalibratedKroneckerBornGPT(cfg_w42).to(device)
    m_w42.calibrate_readout_to_random_attention_unigram(
        token_stream=tokens_100m,
        num_batches=DEFAULT_CALIBRATION_BATCHES,
        seq_len=H2_SEQ_LEN,
        carrier_rho_max=best_spec.carrier_rho_max,
        calibrate_povm=True,
        seed=1337,
    )
    muon_42, adam_42 = build_calibrated_kronecker_optimizers(
        m_w42,
        base_muon_lr=best_spec.muon_lr,
        base_embed_lr=best_spec.embed_lr,
        base_povm_lr=best_spec.povm_lr,
        base_gate_lr=best_spec.gate_lr,
    )
    ev0_42 = evaluate_calibrated_kronecker_model(m_w42, val_batches)
    t0 = time.perf_counter()
    m_w42.train()
    for step in range(1, H2_ABLATION_STEPS + 1):
        tau = float(step - 1) / float(H2_ABLATION_STEPS)
        apply_progress_lr_schedule(muon_42, adam_42, tau, min_lr_ratio=0.05)
        muon_42.zero_grad(set_to_none=True)
        adam_42.zero_grad(set_to_none=True)
        off = ((step - 1) * tokens_per_step) % max(1, n_work - tokens_per_step - 2)
        seq = (work_gpu[off : off + tokens_per_step + 1].to(torch.int64) & 0xFFFF).clamp(0, VALID_VOCAB_SIZE - 1)
        x_b = seq[:-1].view(H2_BATCH_SIZE, H2_SEQ_LEN)
        y_b = seq[1:].view(H2_BATCH_SIZE, H2_SEQ_LEN)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, nll_42 = m_w42(x_b, targets=y_b, return_logits=False)
        assert nll_42 is not None
        pen_42, _ = compute_calibrated_kronecker_penalties(m_w42)
        (nll_42 + pen_42).backward()
        torch.nn.utils.clip_grad_norm_(m_w42.parameters(), 1.0)
        muon_42.step()
        adam_42.step()
    el_42 = time.perf_counter() - t0
    evf_42 = evaluate_calibrated_kronecker_model(m_w42, val_batches)
    variants["trainable_calibrated_hybrid_w42"] = {
        "step0_val_ce": ev0_42["val_ce"],
        "final_val_ce": evf_42["val_ce"],
        "final_val_ppl": evf_42["val_ppl"],
        "final_top1_acc": evf_42["top1_acc"],
        "final_top50_acc": evf_42["top50_acc"],
        "sharp_eff_rank": evf_42["sharp_eff_rank"],
        "mean_factor_rank": evf_42["mean_factor_rank"],
        "povm_drift_std": evf_42["povm_drift_std"],
        "elapsed_sec": round(el_42, 2),
    }
    del m_w42, muon_42, adam_42
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    res_u = variants["uncalibrated_pure_born"]
    res_41 = variants["frozen_unigram_buffer_w41"]
    res_42 = variants["trainable_calibrated_hybrid_w42"]
    gain_vs_uncal = round(float(res_u["final_val_ce"]) - float(res_42["final_val_ce"]), 4)
    gain_vs_w41 = round(float(res_41["final_val_ce"]) - float(res_42["final_val_ce"]), 4)

    passed = bool(
        gain_vs_uncal >= H2_MIN_NATS_VS_UNCAL
        and float(res_42["final_val_ce"]) <= float(res_41["final_val_ce"])
        and float(res_42["final_top1_acc"]) >= float(res_41["final_top1_acc"])
        and float(res_42["povm_drift_std"]) > 0.0
        and float(res_42["mean_factor_rank"]) >= H2_MIN_FACTOR_RANK
        and float(res_42["sharp_eff_rank"]) >= H2_MIN_EMB_RANK
    )
    return {
        "passed": passed,
        "val_ce_gain_vs_uncalibrated_nats": gain_vs_uncal,
        "val_ce_gain_vs_frozen_w41_nats": gain_vs_w41,
        "variants": variants,
    }


def _count_backbone_fp32_leaks(
    model: CalibratedKroneckerBornGPT,
    x_sample: torch.Tensor,
) -> int:
    """Audit `model.blocks` under `bfloat16` autocast to verify `float32_activation_leak_count == 0`."""
    leak_count = 0
    handles = []

    def _hook(_mod: Any, _inp: Any, out: Any) -> None:
        nonlocal leak_count
        tensors = out if isinstance(out, (tuple, list)) else (out,)
        for t in tensors:
            if isinstance(t, torch.Tensor) and t.ndim >= 2 and t.dtype == torch.float32:
                leak_count += 1

    for blk in model.blocks:
        handles.append(blk.register_forward_hook(_hook))
    try:
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = model(x_sample, targets=None, return_logits=True)
    finally:
        for h in handles:
            h.remove()
    return leak_count


def run_h4_live_rtx5060ti_training_verification(
    work_gpu: torch.Tensor,
    tokens_100m: np.ndarray,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    best_spec: HyperparamTrialSpec,
    wall_clock_sec: float,
    device: torch.device,
) -> dict[str, Any]:
    """Execute `H4`: 5-minute (`wall_clock_sec`) live RTX 5060 Ti training of winning `CalibratedKroneckerBornGPT` (`<= 8` args)."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(1337)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1337)

    r_use = 2  # Match R=2 parameter/register budget of w41_kronecker_16x4_R2_M4_df16_L6_300s head-to-head
    cfg = CalibratedKroneckerConfig(
        vocab_size=VOCAB_SIZE_PADDED,
        valid_vocab_size=VALID_VOCAB_SIZE,
        num_meanings=r_use,
        num_factors=DEFAULT_NUM_FACTORS,
        factor_dim=DEFAULT_FACTOR_DIM,
        num_layers=H4_NUM_LAYERS,
        block_size=H4_SEQ_LEN,
        boson_p=best_spec.boson_p,
        holo_alpha=DEFAULT_CAL_HOLO_ALPHA,
        kron_blend=best_spec.kron_blend,
        coinc_gamma=best_spec.coinc_gamma,
        unigram_temp=best_spec.unigram_temp,
        carrier_rho_max=best_spec.carrier_rho_max,
        povm_anchor_weight=DEFAULT_POVM_ANCHOR_WEIGHT,
        sphere_penalty_weight=DEFAULT_SPHERE_PENALTY_WEIGHT,
        unitary_penalty_weight=DEFAULT_UNITARY_PENALTY_WEIGHT,
        iso_penalty_weight=DEFAULT_ISO_PENALTY_WEIGHT,
        branch_orth_weight=DEFAULT_BRANCH_ORTH_WEIGHT,
    )
    model = CalibratedKroneckerBornGPT(cfg).to(device=device)
    cal_stats = model.calibrate_readout_to_random_attention_unigram(
        token_stream=tokens_100m,
        num_batches=DEFAULT_CALIBRATION_BATCHES,
        seq_len=DEFAULT_CALIBRATION_SEQ_LEN,
        carrier_rho_max=best_spec.carrier_rho_max,
        calibrate_povm=True,
        seed=1337,
    )
    opt_muon, opt_adam = build_calibrated_kronecker_optimizers(
        model,
        base_muon_lr=best_spec.muon_lr,
        base_embed_lr=best_spec.embed_lr,
        base_povm_lr=best_spec.povm_lr,
        base_gate_lr=best_spec.gate_lr,
    )

    prior_tokens_t = torch.from_numpy(tokens_100m[:SLICE_2M_TOKENS].astype(np.int64))
    sampler = RareTokenImportanceWindowSampler(
        reference_tokens=prior_tokens_t,
        valid_vocab_size=VALID_VOCAB_SIZE,
        alpha_rare=DEFAULT_RARE_ALPHA,
        laplace_eps=1.0,
    )

    fp32_leaks = 0
    if device.type == "cuda":
        fp32_leaks = _count_backbone_fp32_leaks(model, val_batches[0][0][:2, :64])

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = round(trainable_params / 1e6, 2)
    tokens_per_step = H4_BATCH_SIZE * H4_SEQ_LEN
    sec_tag = int(round(wall_clock_sec))
    est_steps = max(120, int(round((wall_clock_sec / 300.0) * 1450.0)))
    est_tokens_m = round((est_steps * tokens_per_step) / 1e6, 2)
    run_key = (
        f"w42_calibrated_kronecker_16x4_R{cfg.num_meanings}_M{cfg.num_factors}_"
        f"df{cfg.factor_dim}_L{cfg.num_layers}_{sec_tag}s"
    )

    intent = ExperimentIntentSpec(
        run_key=run_key,
        wave_id=H4_WAVE_ID,
        plan_id=H4_PLAN_ID,
        task_domain="fineweb_edu_local",
        hypothesis_intent=(
            f"Wave 42 {sec_tag}s verification of CalibratedKroneckerBornGPT (zero frozen prior_weights buffer, "
            f"Step-0 Procrustes carrier alignment + carrier tilt rho_max={cfg.carrier_rho_max} + trainable "
            f"POVM operator amplitudes, winning H3 config `{best_spec.trial_id}`: p={cfg.boson_p}, "
            f"eta={cfg.kron_blend}, gamma_c={cfg.coinc_gamma}, tau_uni={cfg.unigram_temp})."
        ),
        architecture_summary=(
            f"CalibratedKroneckerBornGPT (R={cfg.num_meanings}, M={cfg.num_factors}, d_f={cfg.factor_dim}, "
            f"basis=65536, L={cfg.num_layers}, {params_m:.2f}M params, zero frozen prior buffer)"
        ),
        hardware_budget="1x RTX 5060 Ti 16GB (BF16 Tensor Cores + Single-bmm Calibrated 16^4 Kronecker Born)",
        batch_size=H4_BATCH_SIZE,
        seq_len=H4_SEQ_LEN,
        total_steps=est_steps,
        target_tokens_M=est_tokens_m,
        params_M=params_m,
        embed_dim=cfg.real_equiv_dim,
        num_layers=cfg.num_layers,
        schedule="smith_one_cycle_wallclock",
    )
    logger = StandardizedExperimentLogger(intent)
    _ping_webui_reload()

    eval_0 = evaluate_calibrated_kronecker_model(model, val_batches)
    eval_0.update(
        {
            "step": 0,
            "tokens_M": 0.0,
            "train_ce": eval_0["val_ce"],
            "compute_tflops": 0.0,
            "tau_progress": 0.0,
            "elapsed_min": 0.0,
            "tok_per_s": 0.0,
            "peak_vram_mib": 0.0,
        }
    )
    logger.log_eval(0, eval_0)
    print(
        f"[{run_key}] Step 0 | Params={params_m:.2f}M | Val CE={eval_0['val_ce']:.4f} "
        f"| Top1={eval_0['top1_acc']:.2f}% | R_factor={eval_0['mean_factor_rank']:.2f}/32 "
        f"| R_emb={eval_0['sharp_eff_rank']:.1f}/256 | Fid={cal_stats['carrier_alignment_fidelity']:.6f}",
        flush=True,
    )

    # Warm up torch.compile chunk kernels for (B=16, T=512) before starting wall-clock timer
    x_w, y_w, tw_w = sampler.sample_batch(work_gpu, H4_BATCH_SIZE, H4_SEQ_LEN, step_seed=29999)
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        _, w_loss = model(x_w, targets=y_w, return_logits=False, token_weights=tw_w)
    assert w_loss is not None
    w_loss.backward()
    opt_muon.zero_grad(set_to_none=True)
    opt_adam.zero_grad(set_to_none=True)
    del x_w, y_w, tw_w, w_loss

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    train_wall_s = 0.0
    step = 0
    next_eval_frac = 0.25
    last_train_ce = eval_0["val_ce"]
    peak_train_vram_mib = 0.0
    min_factor_rank = eval_0["mean_factor_rank"]
    min_emb_rank = eval_0["sharp_eff_rank"]
    nan_inf_events = 0
    history: list[dict[str, Any]] = [dict(eval_0)]

    while train_wall_s < wall_clock_sec:
        step += 1
        t_step_0 = time.perf_counter()
        model.train()
        tau = min(1.0, train_wall_s / max(1e-6, wall_clock_sec))
        model.set_training_progress(tau)
        cur_lr = apply_progress_lr_schedule(opt_muon, opt_adam, tau, min_lr_ratio=0.20)
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)

        x_b, y_b, tok_w = sampler.sample_batch(
            work_gpu, H4_BATCH_SIZE, H4_SEQ_LEN, step_seed=30000 + step
        )
        with torch.amp.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
        ):
            _, nll_loss = model(x_b, targets=y_b, return_logits=False, token_weights=tok_w)
        assert nll_loss is not None
        reg_pen, pen_stats = compute_calibrated_kronecker_penalties(model)
        total_loss = nll_loss + reg_pen
        if not bool(torch.isfinite(total_loss).item()):
            nan_inf_events += 1
        total_loss.backward()
        last_train_ce = float(nll_loss.detach().item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_muon.step()
        opt_adam.step()

        if step % 10 == 0 and device.type == "cuda":
            torch.cuda.synchronize(device)
            cur_mib = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
            peak_train_vram_mib = max(peak_train_vram_mib, cur_mib)

        train_wall_s += time.perf_counter() - t_step_0
        tok_seen = step * tokens_per_step
        tok_s = tok_seen / max(1e-6, train_wall_s)
        tok_m = tok_seen / 1e6
        frac_done = min(1.0, train_wall_s / max(1e-6, wall_clock_sec))

        if step == 1 or step % 20 == 0 or frac_done >= next_eval_frac or train_wall_s >= wall_clock_sec:
            logger.log_step(step, last_train_ce, tok_s, cur_lr / max(1e-6, best_spec.muon_lr), train_wall_s / 60.0)

        if frac_done >= next_eval_frac or train_wall_s >= wall_clock_sec:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                cur_mib = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
                peak_train_vram_mib = max(peak_train_vram_mib, cur_mib)
            ev = evaluate_calibrated_kronecker_model(model, val_batches)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            min_factor_rank = min(min_factor_rank, ev["mean_factor_rank"])
            min_emb_rank = min(min_emb_rank, ev["sharp_eff_rank"])
            ev.update(
                {
                    "step": step,
                    "tokens_M": round(tok_m, 2),
                    "train_ce": round(last_train_ce, 4),
                    "compute_tflops": round(6.0 * params_m * tok_m, 4),
                    "tau_progress": round(frac_done, 4),
                    "elapsed_min": round(train_wall_s / 60.0, 4),
                    "tok_per_s": round(tok_s, 1),
                    "peak_vram_mib": round(peak_train_vram_mib, 1),
                    "sphere_pen": round(pen_stats["sphere_pen"], 6),
                    "unitary_pen": round(pen_stats["unitary_pen"], 6),
                    "iso_pen": round(pen_stats["iso_pen"], 6),
                    "branch_orth_pen": round(pen_stats["branch_orth_pen"], 6),
                    "povm_anchor_pen": round(pen_stats["povm_anchor_pen"], 6),
                }
            )
            history.append(dict(ev))
            logger.log_eval(step, ev)
            print(
                f"[{run_key}] t={train_wall_s:5.1f}s/{wall_clock_sec:.0f}s | Step {step:4d} ({tok_m:5.2f}M tok) "
                f"| Train NLL={last_train_ce:.4f} | Val NLL={ev['val_ce']:.4f} | PPL={ev['val_ppl']:.2f} "
                f"| Top1={ev['top1_acc']:.2f}% | R_fac={ev['mean_factor_rank']:.1f}/32 "
                f"| R_emb={ev['sharp_eff_rank']:.1f}/256 | POVM_std={ev['povm_drift_std']:.4f} "
                f"| {tok_s:,.0f} tok/s",
                flush=True,
            )
            _ping_webui_reload()
            while next_eval_frac <= frac_done:
                next_eval_frac += 0.25

    final_ev = history[-1]
    best_val_ce = min(float(h["val_ce"]) for h in history)
    passed = bool(
        nan_inf_events == 0
        and fp32_leaks == 0
        and peak_train_vram_mib < H4_MAX_VRAM_MIB
        and float(final_ev["val_ce"]) < WAVE41_REF_VAL_CE
        and float(final_ev["top1_acc"]) > WAVE41_REF_TOP1_ACC
        and float(final_ev["sharp_eff_rank"]) >= H2_MIN_EMB_RANK
        and H4_NORM_MEDIAN_MIN <= float(final_ev["embed_norm_median"]) <= H4_NORM_MEDIAN_MAX
    )
    actual_tokens_m = round((step * tokens_per_step) / 1e6, 2)
    summary = {
        "passed": passed,
        "run_key": run_key,
        "winning_trial_id": best_spec.trial_id,
        "num_meanings": cfg.num_meanings,
        "num_factors": cfg.num_factors,
        "factor_dim": cfg.factor_dim,
        "kronecker_hilbert_dim": int(cfg.factor_dim ** cfg.num_factors),
        "d_complex_total": cfg.d_complex_total,
        "real_equiv_dim": cfg.real_equiv_dim,
        "num_layers": cfg.num_layers,
        "batch_size": H4_BATCH_SIZE,
        "seq_len": H4_SEQ_LEN,
        "params_M": params_m,
        "steps": step,
        "tokens_M": actual_tokens_m,
        "target_tokens_M": actual_tokens_m,
        "step0_val_ce": eval_0["val_ce"],
        "best_val_ce": best_val_ce,
        "final_train_ce": round(last_train_ce, 4),
        "final_val_ce": final_ev["val_ce"],
        "final_val_ppl": final_ev["val_ppl"],
        "final_top1_acc": final_ev["top1_acc"],
        "final_top50_acc": final_ev["top50_acc"],
        "wave41_ref_val_ce": WAVE41_REF_VAL_CE,
        "wave41_ref_top1_acc": WAVE41_REF_TOP1_ACC,
        "wave41_ref_top50_acc": WAVE41_REF_TOP50_ACC,
        "val_ce_gain_vs_wave41_nats": round(WAVE41_REF_VAL_CE - float(final_ev["val_ce"]), 4),
        "top1_gain_vs_wave41_pct": round(float(final_ev["top1_acc"]) - WAVE41_REF_TOP1_ACC, 2),
        "step0_sharp_eff_rank": eval_0["sharp_eff_rank"],
        "min_sharp_eff_rank_observed": round(min_emb_rank, 2),
        "final_sharp_eff_rank": final_ev["sharp_eff_rank"],
        "min_mean_factor_rank_observed": round(min_factor_rank, 2),
        "final_mean_factor_rank": final_ev["mean_factor_rank"],
        "final_cross_branch_overlap": final_ev["cross_branch_overlap"],
        "final_povm_drift_std": final_ev["povm_drift_std"],
        "final_embed_norm_median": final_ev["embed_norm_median"],
        "final_complex_unitary_err": final_ev["complex_unitary_err"],
        "float32_activation_leak_count": fp32_leaks,
        "nan_inf_events": nan_inf_events,
        "tok_s": round((step * tokens_per_step) / max(1e-6, train_wall_s), 1),
        "peak_vram_mib": round(peak_train_vram_mib, 1),
        "wall_clock_s": round(train_wall_s, 2),
        "calibration_stats": cal_stats,
        "history": history,
    }
    logger.finalize(extra_summary=summary)
    _ping_webui_reload()
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `examples/67_verify_plan057_calibrated_kronecker_and_hparam_search.py`."""
    parser = argparse.ArgumentParser(
        description="Verify EXPERIMENT_PLAN_057.md (Wave 42: Calibrated Kronecker Born GPT & HParam Search)."
    )
    parser.add_argument(
        "--wall-clock-sec",
        type=float,
        default=300.0,
        help="Wall-clock budget in seconds for H4 live training verification (default: 300.0).",
    )
    parser.add_argument(
        "--search-json",
        type=Path,
        default=DEFAULT_SEARCH_JSON,
        help="Path to precomputed or output hyperparameter search JSON (default: .cache/kronecker_born_hyperparam_search.json).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON path for H1-H4 verification telemetry.",
    )
    return parser.parse_args()


def main() -> None:
    """Run H1-H4 verification suite for `EXPERIMENT_PLAN_057.md` (`Wave 42`)."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PLAN_057 / WAVE_42] Starting H1-H4 verification on device={device} ...", flush=True)

    shard1 = DATA_DIR / "train_full_100M.bin"
    if not shard1.exists():
        shard1 = DATA_DIR / "fineweb_train_000001.bin"
    val_path = DATA_DIR / "val_slice.bin"

    tokens_100m = _load_header_safe_shard_tokens(shard1, max_tokens=SLICE_100M_TOKENS)
    val_tokens = _load_header_safe_shard_tokens(val_path, max_tokens=VAL_SLICE_TOKENS)
    work_gpu = torch.from_numpy(tokens_100m[:SLICE_30M_TOKENS].view("int16")).to(device)
    val_batches_short = build_validation_batches(
        val_tokens, batch_size=H4_VAL_BATCH_SIZE, seq_len=H2_SEQ_LEN, num_batches=H2_VAL_BATCHES, device=device
    )
    val_batches_h4 = build_validation_batches(
        val_tokens, batch_size=H4_VAL_BATCH_SIZE, seq_len=H4_SEQ_LEN, num_batches=H4_VAL_BATCHES, device=device
    )

    print("[H1] Running Step-0 readout operator calibration & 5-input chunked autograd parity probe ...", flush=True)
    h1_res = run_h1_step0_calibration_and_autograd_probe(tokens_100m, val_batches_short, device)
    print(f"[H1] Report: passed={h1_res['passed']} | KL reduction={h1_res['kl_reduction_factor_vs_uncalibrated']}x", flush=True)

    if args.search_json.exists():
        print(f"[H3] Loading hyperparameter search report from {args.search_json} ...", flush=True)
        raw_search = json.loads(args.search_json.read_text(encoding="utf-8"))
        h3_res = raw_search.get("hyperparam_search", raw_search)
    else:
        print("[H3] Running automated multi-configuration hyperparameter search ...", flush=True)
        cfg_runner = SearchRunnerConfig(
            steps_per_trial=H3_STEPS_PER_TRIAL,
            batch_size=H2_BATCH_SIZE,
            seq_len=H2_SEQ_LEN,
            num_layers=H2_NUM_LAYERS,
        )
        h3_res = run_hyperparam_search_suite(
            get_default_candidate_trials(), work_gpu, tokens_100m, val_batches_short, cfg_runner, device
        )
        args.search_json.parent.mkdir(parents=True, exist_ok=True)
        args.search_json.write_text(json.dumps({"hyperparam_search": h3_res}, indent=2), encoding="utf-8")

    best_hp_dict = h3_res["best_trial"]["hyperparams"]
    best_spec = HyperparamTrialSpec(**best_hp_dict)
    print(
        f"[H3] Best trial: `{best_spec.trial_id}` "
        f"(Val CE={h3_res['best_val_ce']:.4f}, improvement={h3_res['val_ce_improvement_over_baseline_nats']:+.4f} nats)",
        flush=True,
    )

    print("[H2] Running 150-step trainable calibrated POVM vs frozen buffer vs uncalibrated ablation ...", flush=True)
    h2_res = run_h2_trainable_povm_vs_frozen_buffer_ablation(
        work_gpu, tokens_100m, val_batches_short, best_spec, device
    )
    print(f"[H2] Report: {json.dumps(h2_res, indent=2)}", flush=True)

    print(f"[H4] Running {args.wall_clock_sec:.0f}s live RTX 5060 Ti training verification ...", flush=True)
    h4_res = run_h4_live_rtx5060ti_training_verification(
        work_gpu, tokens_100m, val_batches_h4, best_spec, float(args.wall_clock_sec), device
    )

    summary = {
        "plan_id": H4_PLAN_ID,
        "wave_id": H4_WAVE_ID,
        "all_passed": bool(
            h1_res["passed"] and h2_res["passed"] and h3_res["passed"] and h4_res["passed"]
        ),
        "H1_step0_readout_calibration_and_autograd": h1_res,
        "H2_trainable_povm_vs_frozen_buffer_ablation": h2_res,
        "H3_hyperparam_search": h3_res,
        "H4_live_rtx5060ti_training": h4_res,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[PLAN_057 / WAVE_42] Wrote results to {args.json_out} (all_passed={summary['all_passed']}).",
        flush=True,
    )


if __name__ == "__main__":
    main()
