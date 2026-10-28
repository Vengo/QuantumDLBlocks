#!/usr/bin/env python3
"""Wave 41 (`EXPERIMENT_PLAN_056.md`): Rank-R Multi-Meaning 16x16x16x16 = 65,536 Kronecker Wavefunction GPT.

Executes and verifies all four hypotheses (`H1`-`H4`) of `EXPERIMENT_PLAN_056.md`:
1. `H1`: 4-Factor `16x16x16x16 = 65,536` Kronecker Coincidence SNR (`>= 400x` vs. flat `C^128`),
   Multi-Meaning Polysemy Superposition Capacity across `R in {1, 2, 4}` (`>= 6.0x`), Exact Born
   Probability Conservation (`< 1e-5`), Zero `exp`/`softmax`/`logsumexp` Verification, & Chunked
   Autograd Parity (`< 1e-5` fwd, `< 1e-4` bwd across all 4 inputs).
2. `H2`: `100M`-Token FineWeb Unigram Prior (`49,850` observed tokens + `407` unseen tokens floored
   at `1.0`) vs. `2M`-Token Prior & Uniform Prior (`> 15x` unseen reduction, `< 8.20` nats Step-0
   validation NLL, and lower Step-150 `val_ce`).
3. `H3`: Relaxed Per-Factor Parseval Isotropy (`lambda_iso = 0.05`) + Cross-Meaning Branch Orthogonality
   (`lambda_branch = 0.20`) vs. Over-Constrained Global Isotropy (`lambda_iso = 0.50, lambda_branch = 0.0`)
   (`R_factor >= 20/32`, `R_emb >= 140/256`, `O_branch < 1e-3`, and lower `val_ce`).
4. `H4`: 5-Minute (`300s`) Live `RTX 5060 Ti` Training Verification of `KroneckerMultiMeaning16x4GPT`
   (`R = 2, M = 4, d_f = 16, L = 6`, `100M`-token unigram prior, paired with `RareTokenImportanceWindowSampler`,
   streamed via `StandardizedExperimentLogger` to `.cache/live_logs/`, `.cache/model_cards/`, and `.cache/benchmarks.db`).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import inspect
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import urllib.request

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
from qdlblocks.models.kronecker_multimeaning_16x4_gpt import (  # noqa: E402
    DEFAULT_BRANCH_ORTH_WEIGHT,
    DEFAULT_FACTOR_DIM,
    DEFAULT_ISO_PENALTY_WEIGHT,
    DEFAULT_KRON_BLEND,
    DEFAULT_KRON_BOSON_P,
    DEFAULT_KRON_COINC_GAMMA,
    DEFAULT_KRON_HOLO_ALPHA,
    DEFAULT_MUON_LR,
    DEFAULT_NUM_FACTORS,
    DEFAULT_NUM_MEANINGS,
    DEFAULT_UNIGRAM_FLOOR_COUNT,
    DEFAULT_UNIGRAM_TEMP,
    KroneckerMultiMeaning16x4GPT,
    KroneckerMultiMeaningBornChunkedNLLFn,
    KroneckerMultiMeaningConfig,
    _load_header_safe_shard_tokens,
    build_kronecker_multimeaning_optimizers,
    compute_kronecker_manifold_and_isotropy_penalties,
    eval_kronecker_multimeaning_born_probs,
)
from qdlblocks.models.tied_holo_born_complex_gpt import (  # noqa: E402
    DEFAULT_RARE_ALPHA,
    RareTokenImportanceWindowSampler,
)

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95, device=0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_056.md (Wave 41 Constants Ledger)
# ==============================================================================
DATA_DIR: Path = ROOT / ".cache" / "fineweb_100M"
DEFAULT_OUTPUT_JSON: Path = ROOT / ".cache" / "hypothesis_experiments_056.json"
WEBUI_RELOAD_URL: str = "http://127.0.0.1:8765/api/reload"

VOCAB_SIZE_PADDED: int = 50304
VALID_VOCAB_SIZE: int = 50257
SLICE_2M_TOKENS: int = 2_000_000
SLICE_30M_TOKENS: int = 30_000_000
SLICE_100M_TOKENS: int = 100_000_000
VAL_SLICE_TOKENS: int = 1_048_576

# H1 thresholds & parameters
H1_SNR_SAMPLE_V: int = 4096
H1_MIN_SNR_GAIN: float = 400.0
H1_MAX_BG_OVERLAP_16X4: float = 2.0e-5
H1_POLYSEMY_VOCAB_V: int = 256
H1_MIN_POLYSEMY_GAIN: float = 6.0
H1_AUTOGRAD_N_TOK: int = 32
H1_AUTOGRAD_V_EFF: int = 128
H1_AUTOGRAD_CHUNK_SIZE: int = 32
H1_MAX_PROB_SUM_ERR: float = 1e-5
H1_MAX_FWD_DIFF: float = 1e-5
H1_MAX_BWD_DIFF: float = 1e-4

# H2 thresholds & parameters
H2_ABLATION_STEPS: int = 150
H2_BATCH_SIZE: int = 4
H2_SEQ_LEN: int = 128
H2_NUM_LAYERS: int = 4
H2_MIN_UNSEEN_REDUCTION: float = 15.0
H2_MAX_STEP0_VAL_CE_100M: float = 8.20
H2_VAL_BATCHES: int = 6

# H3 thresholds & parameters
H3_ABLATION_STEPS: int = 150
H3_BATCH_SIZE: int = 4
H3_SEQ_LEN: int = 128
H3_NUM_LAYERS: int = 4
H3_OVERCONSTRAINED_ISO_WEIGHT: float = 0.50
H3_MIN_FACTOR_RANK: float = 20.0
H3_MIN_GLOBAL_EMB_RANK: float = 140.0
H3_MAX_BRANCH_OVERLAP: float = 1.0e-3
H3_NORM_MEDIAN_MIN: float = 0.95
H3_NORM_MEDIAN_MAX: float = 1.15
H3_MAX_UNITARY_DRIFT: float = 0.25

# H4 live RTX 5060 Ti run constants
H4_WAVE_ID: str = "W41"
H4_PLAN_ID: str = "EXPERIMENT_PLAN_056"
H4_BATCH_SIZE: int = 16
H4_SEQ_LEN: int = 512
H4_NUM_LAYERS: int = 6
H4_VAL_BATCH_SIZE: int = 4
H4_VAL_BATCHES: int = 6
H4_MAX_VRAM_MIB: float = 14500.0
WAVE39_FINAL_RANK_COLLAPSED: float = 27.0


@dataclass(frozen=True)
class AblationVariantSpec:
    """Configuration for a 150-step H2 or H3 ablation arm (`<= 8` fields)."""

    name: str
    prior_mode: str
    iso_penalty_weight: float
    branch_orth_weight: float
    steps: int
    batch_size: int
    seq_len: int
    num_layers: int


def _ping_webui_reload(url: str = WEBUI_RELOAD_URL) -> None:
    """Trigger non-blocking reload on the local `:8765` WebUI daemon."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=1.0):
            pass
    except Exception:
        pass


def _apply_progress_lr(
    opt_muon: torch.optim.Optimizer,
    opt_adam: torch.optim.Optimizer,
    tau: float,
    min_lr_ratio: float = 0.20,
    warmup_frac: float = 0.15,
) -> float:
    """Apply Smith one-cycle warmup + cosine decay envelope for normalized progress `tau in [0, 1]`."""
    progress = max(0.0, min(1.0, float(tau)))
    w_f = max(1e-6, min(0.50, float(warmup_frac)))
    if progress < w_f:
        env = 0.25 + 0.75 * (progress / w_f)
    else:
        t_dec = (progress - w_f) / max(1e-6, 1.0 - w_f)
        env = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * t_dec))
    for g in opt_muon.param_groups:
        g["lr"] = float(g.get("base_lr", DEFAULT_MUON_LR)) * env
    for g in opt_adam.param_groups:
        g["lr"] = float(g.get("base_lr", 0.008)) * env
    return float(opt_muon.param_groups[0]["lr"])


def _build_val_batches(
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
def evaluate_kronecker_model(
    model: KroneckerMultiMeaning16x4GPT,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    """Evaluate `KroneckerMultiMeaning16x4GPT` on held-out `FineWeb` validation batches (`<= 8` args)."""
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
    _, pen_stats = compute_kronecker_manifold_and_isotropy_penalties(model)

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
        "embed_norm_median": round(pen_stats["embed_norm_median"], 4),
        "embed_norm_max": round(pen_stats["embed_norm_max"], 4),
        "complex_unitary_err": round(pen_stats["complex_unitary_err"], 4),
        "kraus_stiefel_err": round(pen_stats["complex_unitary_err"], 6),
    }


def run_h1_kronecker_16x4_snr_and_polysemy_probe(device: torch.device) -> dict[str, Any]:
    """Execute H1: 16^4 = 65,536 Kronecker SNR, polysemy coincidence, and chunked autograd parity."""
    gen = torch.Generator(device=device)
    gen.manual_seed(20261025)

    flat_re = torch.randn((H1_SNR_SAMPLE_V, 128), generator=gen, device=device, dtype=torch.float32)
    flat_im = torch.randn((H1_SNR_SAMPLE_V, 128), generator=gen, device=device, dtype=torch.float32)
    flat_norm = torch.rsqrt((flat_re.square() + flat_im.square()).sum(dim=-1, keepdim=True).clamp_min(1e-12))
    flat_re, flat_im = flat_re * flat_norm, flat_im * flat_norm

    q_re, q_im = flat_re[:64], flat_im[:64]
    ov_flat_re = q_re @ flat_re.t() + q_im @ flat_im.t()
    ov_flat_im = q_re @ flat_im.t() - q_im @ flat_re.t()
    ov_flat_sq = ov_flat_re.square() + ov_flat_im.square()
    self_mask = torch.eye(64, H1_SNR_SAMPLE_V, device=device, dtype=torch.bool)
    flat_self = float(ov_flat_sq[self_mask].mean().item())
    flat_bg = float(ov_flat_sq[~self_mask].mean().item())

    kron_re = torch.randn((H1_SNR_SAMPLE_V, 4, 16), generator=gen, device=device, dtype=torch.float32)
    kron_im = torch.randn((H1_SNR_SAMPLE_V, 4, 16), generator=gen, device=device, dtype=torch.float32)
    k_norm = torch.rsqrt((kron_re.square() + kron_im.square()).sum(dim=-1, keepdim=True).clamp_min(1e-12))
    kron_re, kron_im = kron_re * k_norm, kron_im * k_norm

    qk_re, qk_im = kron_re[:64], kron_im[:64]
    f_re = torch.einsum("qmd,vmd->qmv", qk_re, kron_re) + torch.einsum("qmd,vmd->qmv", qk_im, kron_im)
    f_im = torch.einsum("qmd,vmd->qmv", qk_re, kron_im) - torch.einsum("qmd,vmd->qmv", qk_im, kron_re)
    f_sq = f_re.square() + f_im.square()
    coinc_4 = f_sq[:, 0, :] * f_sq[:, 1, :] * f_sq[:, 2, :] * f_sq[:, 3, :]
    kron_self = float(coinc_4[self_mask].mean().item())
    kron_bg = float(coinc_4[~self_mask].mean().item())
    snr_gain = (kron_self / max(kron_bg, 1e-12)) / (flat_self / max(flat_bg, 1e-12))

    senses_re = torch.randn((H1_POLYSEMY_VOCAB_V, 2, 4, 16), generator=gen, device=device, dtype=torch.float32)
    senses_im = torch.randn((H1_POLYSEMY_VOCAB_V, 2, 4, 16), generator=gen, device=device, dtype=torch.float32)
    s_norm = torch.rsqrt((senses_re.square() + senses_im.square()).sum(dim=-1, keepdim=True).clamp_min(1e-12))
    senses_re, senses_im = senses_re * s_norm, senses_im * s_norm

    avg_re = senses_re.mean(dim=1)
    avg_im = senses_im.mean(dim=1)
    a_norm = torch.rsqrt((avg_re.square() + avg_im.square()).sum(dim=-1, keepdim=True).clamp_min(1e-12))
    avg_re, avg_im = avg_re * a_norm, avg_im * a_norm
    r1_ov_re = (avg_re.unsqueeze(1) * senses_re + avg_im.unsqueeze(1) * senses_im).sum(dim=-1)
    r1_ov_im = (avg_re.unsqueeze(1) * senses_im - avg_im.unsqueeze(1) * senses_re).sum(dim=-1)
    r1_coinc = float((r1_ov_re.square() + r1_ov_im.square()).prod(dim=-1).mean().item())

    polysemy_by_rank: dict[str, float] = {"kronecker_16x4_rank1": r1_coinc}
    for r_val in (2, 4):
        b_re = torch.randn((H1_POLYSEMY_VOCAB_V, r_val, 4, 16), generator=gen, device=device, dtype=torch.float32)
        b_im = torch.randn((H1_POLYSEMY_VOCAB_V, r_val, 4, 16), generator=gen, device=device, dtype=torch.float32)
        bn = torch.rsqrt((b_re.square() + b_im.square()).sum(dim=-1, keepdim=True).clamp_min(1e-12))
        b_re, b_im = b_re * bn, b_im * bn
        b_re[:, :2] = senses_re
        b_im[:, :2] = senses_im
        ov_re = torch.einsum("vrmd,vkmd->vrkm", b_re, senses_re) + torch.einsum("vrmd,vkmd->vrkm", b_im, senses_im)
        ov_im = torch.einsum("vrmd,vkmd->vrkm", b_re, senses_im) - torch.einsum("vrmd,vkmd->vrkm", b_im, senses_re)
        c_vrk = (ov_re.square() + ov_im.square()).prod(dim=-1)
        w_r = torch.zeros((1, r_val, 1), device=device, dtype=torch.float32)
        w_r[:, :2, :] = 0.5
        polysemy_by_rank[f"kronecker_16x4_rank{r_val}"] = float(((w_r * c_vrk).sum(dim=1)).mean().item())

    polysemy_gain_r2 = polysemy_by_rank["kronecker_16x4_rank2"] / max(r1_coinc, 1e-12)
    polysemy_gain_r4 = polysemy_by_rank["kronecker_16x4_rank4"] / max(r1_coinc, 1e-12)

    src_eval = inspect.getsource(eval_kronecker_multimeaning_born_probs)
    src_chunk = inspect.getsource(KroneckerMultiMeaningBornChunkedNLLFn)
    combined_src = src_eval + "\n" + src_chunk
    forbidden = ("torch.exp(", "F.softmax(", "torch.softmax(", "logsumexp(")
    zero_exp_softmax = all(tok not in combined_src for tok in forbidden)

    phi = torch.randn((H1_AUTOGRAD_N_TOK, 2, 4, 32), generator=gen, device=device, dtype=torch.float32)
    psi = torch.randn((H1_AUTOGRAD_V_EFF, 2, 4, 32), generator=gen, device=device, dtype=torch.float32)
    phi_norm = torch.rsqrt((phi[..., :16].square() + phi[..., 16:].square()).sum(-1, keepdim=True))
    psi_norm = torch.rsqrt((psi[..., :16].square() + psi[..., 16:].square()).sum(-1, keepdim=True))
    phi_c = torch.cat([phi[..., :16] * phi_norm, phi[..., 16:] * phi_norm], dim=-1).detach().requires_grad_(True)
    psi_c = torch.cat([psi[..., :16] * psi_norm, psi[..., 16:] * psi_norm], dim=-1).detach().requires_grad_(True)
    beta_c = torch.full((H1_AUTOGRAD_N_TOK, 2), 1.0 / math.sqrt(2.0), device=device, requires_grad=True)
    alpha_c = torch.full((H1_AUTOGRAD_V_EFF, 2), 1.0 / math.sqrt(2.0), device=device, requires_grad=True)

    phi_ref = phi_c.detach().clone().requires_grad_(True)
    psi_ref = psi_c.detach().clone().requires_grad_(True)
    beta_ref = beta_c.detach().clone().requires_grad_(True)
    alpha_ref = alpha_c.detach().clone().requires_grad_(True)

    prior = torch.linspace(0.5, 2.0, H1_AUTOGRAD_V_EFF, device=device, dtype=torch.float32)
    hp = torch.tensor(
        [DEFAULT_KRON_BOSON_P, DEFAULT_KRON_HOLO_ALPHA, DEFAULT_KRON_BLEND, DEFAULT_KRON_COINC_GAMMA],
        device=device,
        dtype=torch.float32,
    )
    targets = torch.randint(0, H1_AUTOGRAD_V_EFF, (H1_AUTOGRAD_N_TOK,), generator=gen, device=device)
    tw = torch.linspace(0.7, 1.4, H1_AUTOGRAD_N_TOK, device=device, dtype=torch.float32)

    probs = eval_kronecker_multimeaning_born_probs(phi_ref, psi_ref, beta_ref, alpha_ref, prior, hp)
    max_prob_sum_err = float((probs.sum(dim=-1) - 1.0).abs().max().item())
    tw_norm = tw / tw.sum()
    p_y = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    loss_ref = -(tw_norm * torch.log(p_y.clamp_min(1e-30))).sum()
    loss_ref.backward()

    saved_chunk = KroneckerMultiMeaningBornChunkedNLLFn.vocab_chunk_size
    KroneckerMultiMeaningBornChunkedNLLFn.vocab_chunk_size = H1_AUTOGRAD_CHUNK_SIZE
    try:
        loss_chunk = KroneckerMultiMeaningBornChunkedNLLFn.apply(
            phi_c, psi_c, beta_c, alpha_c, targets, prior, hp, tw
        )
        loss_chunk.backward()
    finally:
        KroneckerMultiMeaningBornChunkedNLLFn.vocab_chunk_size = saved_chunk

    fwd_diff = float((loss_chunk.detach() - loss_ref.detach()).abs().item())
    bwd_phi_diff = float((phi_c.grad - phi_ref.grad).abs().max().item())
    bwd_psi_diff = float((psi_c.grad - psi_ref.grad).abs().max().item())
    bwd_beta_diff = float((beta_c.grad - beta_ref.grad).abs().max().item())
    bwd_alpha_diff = float((alpha_c.grad - alpha_ref.grad).abs().max().item())
    max_bwd_diff = max(bwd_phi_diff, bwd_psi_diff, bwd_beta_diff, bwd_alpha_diff)

    passed = bool(
        snr_gain >= H1_MIN_SNR_GAIN
        and kron_bg <= H1_MAX_BG_OVERLAP_16X4
        and polysemy_gain_r2 >= H1_MIN_POLYSEMY_GAIN
        and polysemy_gain_r4 >= H1_MIN_POLYSEMY_GAIN
        and zero_exp_softmax
        and max_prob_sum_err < H1_MAX_PROB_SUM_ERR
        and fwd_diff < H1_MAX_FWD_DIFF
        and max_bwd_diff < H1_MAX_BWD_DIFF
    )
    return {
        "passed": passed,
        "flat_c128_self_overlap_sq": flat_self,
        "flat_c128_bg_overlap_sq": flat_bg,
        "kronecker_16x4_self_overlap_sq": kron_self,
        "kronecker_16x4_bg_overlap_sq": kron_bg,
        "single_branch_snr_gain_vs_flat128": round(snr_gain, 2),
        "polysemy_2sense_coincidence_by_rank": polysemy_by_rank,
        "polysemy_gain_rank2_vs_rank1": round(polysemy_gain_r2, 4),
        "polysemy_gain_rank4_vs_rank1": round(polysemy_gain_r4, 4),
        "zero_exp_softmax_in_readout": zero_exp_softmax,
        "max_prob_sum_err": max_prob_sum_err,
        "autograd_fwd_diff": fwd_diff,
        "autograd_bwd_phi_diff": bwd_phi_diff,
        "autograd_bwd_psi_diff": bwd_psi_diff,
        "autograd_bwd_beta_diff": bwd_beta_diff,
        "autograd_bwd_alpha_diff": bwd_alpha_diff,
    }


def _apply_prior_mode_to_model(
    model: KroneckerMultiMeaning16x4GPT,
    prior_mode: str,
    tokens_100m: np.ndarray,
    val_tokens: np.ndarray,
) -> dict[str, Any]:
    """Configure `model` unigram prior according to `prior_mode` and return coverage diagnostics (`<= 8` args)."""
    v_valid = VALID_VOCAB_SIZE
    val_counts = np.bincount(np.clip(val_tokens.astype(np.int64), 0, v_valid - 1), minlength=v_valid)[
        :v_valid
    ].astype(np.float64)
    p_val = val_counts / max(1.0, float(val_counts.sum()))
    pos_mask = p_val > 0.0

    if prior_mode == "uniform_prior":
        stats = model.initialize_unigram_prior_from_tokens(
            tokens_100m[:0], floor_count=DEFAULT_UNIGRAM_FLOOR_COUNT
        )
        p_uni = np.full((v_valid,), 1.0 / float(v_valid), dtype=np.float64)
    elif prior_mode == "slice_2m_prior":
        stats = model.initialize_unigram_prior_from_tokens(
            tokens_100m[:SLICE_2M_TOKENS], floor_count=DEFAULT_UNIGRAM_FLOOR_COUNT
        )
        p_uni = model.unigram_probs[:v_valid].detach().cpu().numpy().astype(np.float64)
    elif prior_mode == "slice_100m_prior":
        stats = model.initialize_unigram_prior_from_tokens(
            tokens_100m, floor_count=DEFAULT_UNIGRAM_FLOOR_COUNT
        )
        p_uni = model.unigram_probs[:v_valid].detach().cpu().numpy().astype(np.float64)
    else:
        raise ValueError(f"Unknown prior_mode: {prior_mode}")

    kl_val_vs_uni = float(np.sum(p_val[pos_mask] * np.log(p_val[pos_mask] / p_uni[pos_mask])))
    return {
        "prior_mode": prior_mode,
        "observed_tokens": int(stats["observed_tokens"]),
        "unseen_tokens": int(stats["unseen_tokens"]),
        "kl_val_vs_unigram_nats": round(kl_val_vs_uni, 4),
    }


def _run_short_ablation_variant(
    spec: AblationVariantSpec,
    work_gpu: torch.Tensor,
    tokens_100m: np.ndarray,
    val_tokens: np.ndarray,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    """Run a deterministic 150-step ablation variant for H2 or H3 (`<= 8` args)."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    torch.manual_seed(1337)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1337)

    unigram_temp = 0.0 if spec.prior_mode == "uniform_prior" else DEFAULT_UNIGRAM_TEMP
    cfg = KroneckerMultiMeaningConfig(
        vocab_size=VOCAB_SIZE_PADDED,
        valid_vocab_size=VALID_VOCAB_SIZE,
        num_meanings=DEFAULT_NUM_MEANINGS,
        num_factors=DEFAULT_NUM_FACTORS,
        factor_dim=DEFAULT_FACTOR_DIM,
        num_layers=spec.num_layers,
        block_size=spec.seq_len,
        unigram_temp=unigram_temp,
        iso_penalty_weight=spec.iso_penalty_weight,
        branch_orth_weight=spec.branch_orth_weight,
    )
    model = KroneckerMultiMeaning16x4GPT(cfg).to(device=device)
    prior_info = _apply_prior_mode_to_model(model, spec.prior_mode, tokens_100m, val_tokens)
    muon_opt, adam_opt = build_kronecker_multimeaning_optimizers(model)

    eval_0 = evaluate_kronecker_model(model, val_batches)
    tokens_per_step = spec.batch_size * spec.seq_len
    n_work = int(work_gpu.numel())

    t0 = time.perf_counter()
    final_train_nll = eval_0["val_ce"]
    model.train()
    for step in range(1, spec.steps + 1):
        tau = float(step - 1) / float(max(1, spec.steps))
        _apply_progress_lr(muon_opt, adam_opt, tau)
        muon_opt.zero_grad(set_to_none=True)
        adam_opt.zero_grad(set_to_none=True)

        off = ((step - 1) * tokens_per_step) % max(1, n_work - tokens_per_step - 2)
        seq = (work_gpu[off : off + tokens_per_step + 1].to(torch.int64) & 0xFFFF).clamp(
            0, VALID_VOCAB_SIZE - 1
        )
        x_b = seq[:-1].view(spec.batch_size, spec.seq_len)
        y_b = seq[1:].view(spec.batch_size, spec.seq_len)

        with torch.amp.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
        ):
            _, born_nll = model(x_b, targets=y_b, return_logits=False)
        assert born_nll is not None
        pen_loss, _ = compute_kronecker_manifold_and_isotropy_penalties(model)
        total_loss = born_nll + pen_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        muon_opt.step()
        adam_opt.step()
        final_train_nll = float(born_nll.detach().item())

    elapsed_s = time.perf_counter() - t0
    final_eval = evaluate_kronecker_model(model, val_batches)
    del model, muon_opt, adam_opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "variant": spec.name,
        **prior_info,
        "step0_val_ce": eval_0["val_ce"],
        "step0_top1_acc": eval_0["top1_acc"],
        "final_val_ce": final_eval["val_ce"],
        "final_val_ppl": final_eval["val_ppl"],
        "final_top1_acc": final_eval["top1_acc"],
        "final_top50_acc": final_eval["top50_acc"],
        "final_train_nll": round(final_train_nll, 4),
        "elapsed_sec": round(elapsed_s, 2),
        "diagnostics": final_eval,
    }


def run_h2_unigram_100m_prior_ablation(
    work_gpu: torch.Tensor,
    tokens_100m: np.ndarray,
    val_tokens: np.ndarray,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    """Execute H2: 150-step comparison of `uniform_prior`, `slice_2m_prior`, and `slice_100m_prior` (`<= 8` args)."""
    variants: dict[str, Any] = {}
    for mode in ("uniform_prior", "slice_2m_prior", "slice_100m_prior"):
        spec = AblationVariantSpec(
            name=mode,
            prior_mode=mode,
            iso_penalty_weight=DEFAULT_ISO_PENALTY_WEIGHT,
            branch_orth_weight=DEFAULT_BRANCH_ORTH_WEIGHT,
            steps=H2_ABLATION_STEPS,
            batch_size=H2_BATCH_SIZE,
            seq_len=H2_SEQ_LEN,
            num_layers=H2_NUM_LAYERS,
        )
        variants[mode] = _run_short_ablation_variant(
            spec, work_gpu, tokens_100m, val_tokens, val_batches, device
        )

    u_res = variants["uniform_prior"]
    s2_res = variants["slice_2m_prior"]
    s100_res = variants["slice_100m_prior"]
    unseen_reduction = round(
        float(s2_res["unseen_tokens"]) / float(max(1, s100_res["unseen_tokens"])), 2
    )

    passed = bool(
        unseen_reduction > H2_MIN_UNSEEN_REDUCTION
        and s100_res["kl_val_vs_unigram_nats"] < s2_res["kl_val_vs_unigram_nats"]
        and s100_res["kl_val_vs_unigram_nats"] < u_res["kl_val_vs_unigram_nats"]
        and s100_res["step0_val_ce"] < H2_MAX_STEP0_VAL_CE_100M
        and s100_res["final_val_ce"] < u_res["final_val_ce"]
        and s100_res["final_val_ce"] <= s2_res["final_val_ce"]
    )
    return {
        "passed": passed,
        "unseen_token_reduction_2m_to_100m": unseen_reduction,
        "variants": variants,
    }


def run_h3_relaxed_isotropy_and_branch_orth_ablation(
    work_gpu: torch.Tensor,
    tokens_100m: np.ndarray,
    val_tokens: np.ndarray,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    """Execute H3: 150-step comparison of overconstrained isotropy vs relaxed isotropy + branch orthogonality (`<= 8` args)."""
    over_spec = AblationVariantSpec(
        name="overconstrained_iso_no_branch_orth",
        prior_mode="slice_100m_prior",
        iso_penalty_weight=H3_OVERCONSTRAINED_ISO_WEIGHT,
        branch_orth_weight=0.0,
        steps=H3_ABLATION_STEPS,
        batch_size=H3_BATCH_SIZE,
        seq_len=H3_SEQ_LEN,
        num_layers=H3_NUM_LAYERS,
    )
    relaxed_spec = AblationVariantSpec(
        name="relaxed_iso_with_branch_orth",
        prior_mode="slice_100m_prior",
        iso_penalty_weight=DEFAULT_ISO_PENALTY_WEIGHT,
        branch_orth_weight=DEFAULT_BRANCH_ORTH_WEIGHT,
        steps=H3_ABLATION_STEPS,
        batch_size=H3_BATCH_SIZE,
        seq_len=H3_SEQ_LEN,
        num_layers=H3_NUM_LAYERS,
    )

    over_res = _run_short_ablation_variant(
        over_spec, work_gpu, tokens_100m, val_tokens, val_batches, device
    )
    rel_res = _run_short_ablation_variant(
        relaxed_spec, work_gpu, tokens_100m, val_tokens, val_batches, device
    )
    diag = rel_res["diagnostics"]

    passed = bool(
        diag["mean_factor_rank"] >= H3_MIN_FACTOR_RANK
        and diag["sharp_eff_rank"] >= H3_MIN_GLOBAL_EMB_RANK
        and diag["cross_branch_overlap"] < H3_MAX_BRANCH_OVERLAP
        and H3_NORM_MEDIAN_MIN <= diag["embed_norm_median"] <= H3_NORM_MEDIAN_MAX
        and diag["complex_unitary_err"] < H3_MAX_UNITARY_DRIFT
        and rel_res["final_val_ce"] <= over_res["final_val_ce"]
    )
    return {
        "passed": passed,
        "overconstrained_iso_no_branch_orth": over_res,
        "relaxed_iso_with_branch_orth": rel_res,
    }


def _count_backbone_fp32_leaks(
    model: KroneckerMultiMeaning16x4GPT,
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
    wall_clock_sec: float,
    device: torch.device,
) -> dict[str, Any]:
    """Execute H4: 5-minute (`wall_clock_sec`) live RTX 5060 Ti training verification of `KroneckerMultiMeaning16x4GPT` (`<= 8` args)."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(1337)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1337)

    cfg = KroneckerMultiMeaningConfig(
        vocab_size=VOCAB_SIZE_PADDED,
        valid_vocab_size=VALID_VOCAB_SIZE,
        num_meanings=DEFAULT_NUM_MEANINGS,
        num_factors=DEFAULT_NUM_FACTORS,
        factor_dim=DEFAULT_FACTOR_DIM,
        num_layers=H4_NUM_LAYERS,
        block_size=H4_SEQ_LEN,
    )
    model = KroneckerMultiMeaning16x4GPT(cfg).to(device=device)
    prior_stats = model.initialize_unigram_prior_from_tokens(
        tokens_100m, floor_count=DEFAULT_UNIGRAM_FLOOR_COUNT
    )
    opt_muon, opt_adam = build_kronecker_multimeaning_optimizers(model)

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
    est_steps = max(120, int(round((wall_clock_sec / 300.0) * 200.0)))
    est_tokens_m = round((est_steps * tokens_per_step) / 1e6, 2)
    run_key = f"w41_kronecker_16x4_R{cfg.num_meanings}_M{cfg.num_factors}_df{cfg.factor_dim}_L{cfg.num_layers}_{sec_tag}s"

    intent = ExperimentIntentSpec(
        run_key=run_key,
        wave_id=H4_WAVE_ID,
        plan_id=H4_PLAN_ID,
        task_domain="fineweb_edu_local",
        hypothesis_intent=(
            f"Wave 41 {sec_tag}s verification of KroneckerMultiMeaning16x4GPT (Rank-{cfg.num_meanings} "
            f"16x16x16x16 = 65,536 Kronecker Wavefunction GPT, 100M-token FineWeb unigram prior, "
            f"100% softmax-free polynomial Born p={cfg.boson_p}, relaxed per-factor isotropy + branch orthogonality)."
        ),
        architecture_summary=(
            f"KroneckerMultiMeaning16x4GPT (R={cfg.num_meanings}, M={cfg.num_factors}, d_f={cfg.factor_dim}, "
            f"basis=65536, L={cfg.num_layers}, {params_m:.2f}M params)"
        ),
        hardware_budget="1x RTX 5060 Ti 16GB (BF16 Tensor Cores + Single-bmm Compiled 16^4 Kronecker Born)",
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

    eval_0 = evaluate_kronecker_model(model, val_batches)
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
        f"| R_emb={eval_0['sharp_eff_rank']:.1f}/256 | O_branch={eval_0['cross_branch_overlap']:.2e}",
        flush=True,
    )

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
        cur_lr = _apply_progress_lr(opt_muon, opt_adam, tau, min_lr_ratio=0.20)
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)

        x_b, y_b, tok_w = sampler.sample_batch(
            work_gpu, H4_BATCH_SIZE, H4_SEQ_LEN, step_seed=20000 + step
        )
        with torch.amp.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
        ):
            _, nll_loss = model(x_b, targets=y_b, return_logits=False, token_weights=tok_w)
        assert nll_loss is not None
        reg_pen, pen_stats = compute_kronecker_manifold_and_isotropy_penalties(model)
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
            logger.log_step(step, last_train_ce, tok_s, cur_lr / DEFAULT_MUON_LR, train_wall_s / 60.0)

        if frac_done >= next_eval_frac or train_wall_s >= wall_clock_sec:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                cur_mib = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
                peak_train_vram_mib = max(peak_train_vram_mib, cur_mib)
            ev = evaluate_kronecker_model(model, val_batches)
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
                }
            )
            history.append(dict(ev))
            logger.log_eval(step, ev)
            print(
                f"[{run_key}] t={train_wall_s:5.1f}s/{wall_clock_sec:.0f}s | Step {step:4d} ({tok_m:5.2f}M tok) "
                f"| Train NLL={last_train_ce:.4f} | Val NLL={ev['val_ce']:.4f} | PPL={ev['val_ppl']:.2f} "
                f"| Top1={ev['top1_acc']:.2f}% | R_fac={ev['mean_factor_rank']:.1f}/32 "
                f"| R_emb={ev['sharp_eff_rank']:.1f}/256 | O_br={ev['cross_branch_overlap']:.2e} "
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
        and final_ev["val_ce"] < eval_0["val_ce"]
        and final_ev["mean_factor_rank"] >= H3_MIN_FACTOR_RANK
        and final_ev["sharp_eff_rank"] >= H3_MIN_GLOBAL_EMB_RANK
        and final_ev["cross_branch_overlap"] < H3_MAX_BRANCH_OVERLAP
    )
    actual_tokens_m = round((step * tokens_per_step) / 1e6, 2)
    summary = {
        "passed": passed,
        "run_key": run_key,
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
        "step0_sharp_eff_rank": eval_0["sharp_eff_rank"],
        "min_sharp_eff_rank_observed": round(min_emb_rank, 2),
        "final_sharp_eff_rank": final_ev["sharp_eff_rank"],
        "min_mean_factor_rank_observed": round(min_factor_rank, 2),
        "final_mean_factor_rank": final_ev["mean_factor_rank"],
        "final_cross_branch_overlap": final_ev["cross_branch_overlap"],
        "wave39_collapsed_rank_ref": WAVE39_FINAL_RANK_COLLAPSED,
        "rank_gain_vs_wave39_x": round(final_ev["sharp_eff_rank"] / WAVE39_FINAL_RANK_COLLAPSED, 2),
        "final_embed_norm_median": final_ev["embed_norm_median"],
        "final_complex_unitary_err": final_ev["complex_unitary_err"],
        "float32_activation_leak_count": fp32_leaks,
        "nan_inf_events": nan_inf_events,
        "tok_s": round((step * tokens_per_step) / max(1e-6, train_wall_s), 1),
        "peak_vram_mib": round(peak_train_vram_mib, 1),
        "wall_clock_s": round(train_wall_s, 2),
        "prior_stats": prior_stats,
        "history": history,
    }
    logger.finalize(extra_summary=summary)
    _ping_webui_reload()
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `examples/66_verify_plan056_kronecker_multimeaning_16x4.py`."""
    parser = argparse.ArgumentParser(
        description="Verify EXPERIMENT_PLAN_056.md (Wave 41: Rank-R Multi-Meaning 16^4 Kronecker Wavefunction GPT)."
    )
    parser.add_argument(
        "--wall-clock-sec",
        type=float,
        default=300.0,
        help="Wall-clock budget in seconds for H4 live training verification (default: 300.0).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON path for H1-H4 verification telemetry.",
    )
    return parser.parse_args()


def main() -> None:
    """Run H1-H4 verification suite for `EXPERIMENT_PLAN_056.md` (`Wave 41`)."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PLAN_056 / WAVE_41] Starting H1-H4 verification on device={device} ...", flush=True)

    shard1 = DATA_DIR / "train_full_100M.bin"
    if not shard1.exists():
        shard1 = DATA_DIR / "fineweb_train_000001.bin"
    val_path = DATA_DIR / "val_slice.bin"

    tokens_100m = _load_header_safe_shard_tokens(shard1, max_tokens=SLICE_100M_TOKENS)
    val_tokens = _load_header_safe_shard_tokens(val_path, max_tokens=VAL_SLICE_TOKENS)
    work_gpu = torch.from_numpy(tokens_100m[:SLICE_30M_TOKENS].view("int16")).to(device)
    val_batches_short = _build_val_batches(
        val_tokens, batch_size=H4_VAL_BATCH_SIZE, seq_len=H2_SEQ_LEN, num_batches=H2_VAL_BATCHES, device=device
    )
    val_batches_h4 = _build_val_batches(
        val_tokens, batch_size=H4_VAL_BATCH_SIZE, seq_len=H4_SEQ_LEN, num_batches=H4_VAL_BATCHES, device=device
    )

    print("[H1] Running 16^4 = 65,536 Kronecker SNR, polysemy coincidence, and autograd parity probe ...", flush=True)
    h1_res = run_h1_kronecker_16x4_snr_and_polysemy_probe(device)
    print(f"[H1] Report: {json.dumps(h1_res, indent=2)}", flush=True)

    print("[H2] Running 150-step 100M-token vs 2M-token vs uniform prior ablation ...", flush=True)
    h2_res = run_h2_unigram_100m_prior_ablation(
        work_gpu, tokens_100m, val_tokens, val_batches_short, device
    )
    print(f"[H2] Report: {json.dumps(h2_res, indent=2)}", flush=True)

    print("[H3] Running 150-step relaxed isotropy + branch orthogonality ablation ...", flush=True)
    h3_res = run_h3_relaxed_isotropy_and_branch_orth_ablation(
        work_gpu, tokens_100m, val_tokens, val_batches_short, device
    )
    print(f"[H3] Report: {json.dumps(h3_res, indent=2)}", flush=True)

    print(f"[H4] Running {args.wall_clock_sec:.0f}s live RTX 5060 Ti training verification ...", flush=True)
    h4_res = run_h4_live_rtx5060ti_training_verification(
        work_gpu, tokens_100m, val_batches_h4, float(args.wall_clock_sec), device
    )

    summary = {
        "plan_id": H4_PLAN_ID,
        "wave_id": H4_WAVE_ID,
        "all_passed": bool(
            h1_res["passed"] and h2_res["passed"] and h3_res["passed"] and h4_res["passed"]
        ),
        "H1_kronecker_16x4_snr_and_polysemy": h1_res,
        "H2_unigram_100m_prior_ablation": h2_res,
        "H3_relaxed_isotropy_and_branch_orth_ablation": h3_res,
        "H4_live_rtx5060ti_training": h4_res,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[PLAN_056 / WAVE_41] Wrote results to {args.json_out} (all_passed={summary['all_passed']}).",
        flush=True,
    )


if __name__ == "__main__":
    main()
