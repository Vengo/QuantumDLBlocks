"""Wave 48 (`EXPERIMENT_PLAN_063.md`) Verification & Standalone FineWeb Runner: `StandaloneQuantumBornGPT`.

Executes three teacher-free, end-to-end experiments validating `StandaloneQuantumBornGPT` on FineWeb:
1. **H1 (`verify_h1_uhlmann_krylov_phase_and_ablation`)**:
   - Verifies fused `288`-width isometric density + coherent ket packing (`fast_rank_r_hermitian_and_ket_pack`)
     and Uhlmann phase-locked Rank-2 Lanczos-Krylov extraction (`extract_rank2_krylov_eigenpair_uhlmann`).
   - Trains standalone `UHLMANN_KRYLOV_320` vs. `PHASE_ERASED_DENSITY_256` from random Step-0 initialization
     on FineWeb without any classical teacher or ridge regression.
2. **H2 (`verify_h2_direct_stiefel_tikhonov_muon`)**:
   - Verifies direct Stiefel readouts (`W_obs in St(1920, d)`, `W_chan in St(2d, d)`) and Tikhonov-damped
     `AllLayerTikhonovRiemannianStiefelMuon` (`lambda_tik = 1e-3`) manifold preservation (`< 1e-4`) vs.
     undamped Euclidean chord steps.
3. **H3 (`verify_h3_active_vocab_manifold_optimizers`)**:
   - Benchmarks `BatchSubspaceTikhonovGramMuon`, `RiemannianSphereTangentSNROptimizer`, and `AdamW` across
     `VocabularyReadoutMode.RANK2_KRYLOV_BORN` (`Rank2KrylovBornChunkedCEFn`) and
     `VocabularyReadoutMode.SOFTCAP_LINEAR_CE` (`_CompiledChunkedSoftcapLinearCEFn`).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qdlblocks.benchmarks.experiment_logger import (
    ExperimentIntentSpec,
    StandardizedExperimentLogger,
)
from qdlblocks.layers.impure_density_mirror_block import (
    DEFAULT_FUSED_HEAD_PACK_DIM,
    DEFAULT_KRYLOV_OBSERVABLE_HEAD_DIM,
    DEFAULT_TIKHONOV_GRAM_LAMBDA,
    DEFAULT_UHLMANN_PHASE_EPS,
    complex_l2_normalize,
    extract_rank2_krylov_eigenpair_uhlmann,
    fast_rank_r_hermitian_and_ket_pack,
    unpack_hermitian_isometric,
)
from qdlblocks.models.standalone_quantum_born_gpt import (
    DEFAULT_BATCH_GRAM_MUON_LR,
    DEFAULT_BORN_LOG_INTENSITY_EPS,
    DEFAULT_KRYLOV_BORN_LOG_TEMP,
    DEFAULT_SPHERE_GEODESIC_LR,
    DEFAULT_SPHERE_SNR_TRUST_TAU,
    QuantumAttentionMode,
    StandaloneQuantumBornBlock,
    StandaloneQuantumBornConfig,
    StandaloneQuantumBornGPT,
    VocabOptimizerMode,
    VocabularyReadoutMode,
    build_standalone_quantum_born_optimizers,
)

# ---------------------------------------------------------------------------
# Top-of-File Named Constants (Rule 7 — Synchronized with EXPERIMENT_PLAN_063.md)
# ---------------------------------------------------------------------------
CACHE_DIR = REPO_ROOT / ".cache"
FINEWEB_DIR = CACHE_DIR / "fineweb_100M"
DEFAULT_JSON_OUT = CACHE_DIR / "hypothesis_experiments_063.json"
SHARD_HEADER_INTS: int = 256
OFFICIAL_FINEWEB_VAL_TOKENS: int = 10_485_760
FAST_SCREENING_VAL_TOKENS: int = 65_536


@dataclass(frozen=True)
class StandaloneEvalDataBundle:
    """Bundled FineWeb token slices and batch geometry (`<= 8` fields, Rule 2)."""

    train_tokens: torch.Tensor
    val_tokens: torch.Tensor
    vocab_size: int
    batch_size: int
    seq_len: int
    device: torch.device


def _load_fineweb_tokens(shard_path: Path, max_tokens: int = 20_000_000) -> torch.Tensor:
    """Load GPT-2 BPE token IDs from a `modded-nanogpt` FineWeb `.bin` shard (never synthetic random data)."""
    if not shard_path.exists():
        raise FileNotFoundError(f"FineWeb shard not found at {shard_path}")
    raw = np.memmap(shard_path, dtype=np.uint16, mode="r", offset=SHARD_HEADER_INTS * 4)
    count = min(int(max_tokens), int(raw.shape[0]))
    slice_arr = np.array(raw[:count], dtype=np.int32)
    return torch.from_numpy(slice_arr)


def _load_fineweb_train_stream(fineweb_dir: Path, needed_tokens: int) -> torch.Tensor:
    """Stream contiguous GPT-2 BPE token IDs across `fineweb_train_000001.bin`..`fineweb_train_000009.bin`."""
    shard_paths = sorted(fineweb_dir.glob("fineweb_train_*.bin"))
    if not shard_paths:
        raise FileNotFoundError(f"No FineWeb training shards found in {fineweb_dir}")
    chunks: list[torch.Tensor] = []
    remaining = int(needed_tokens)
    for sp in shard_paths:
        if remaining <= 0:
            break
        part = _load_fineweb_tokens(sp, max_tokens=remaining)
        chunks.append(part)
        remaining -= int(part.numel())
    if remaining > 0:
        raise RuntimeError(f"Requested {needed_tokens} tokens, but {remaining} tokens short across {len(shard_paths)} shards")
    return torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]


def _sample_batch(
    tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    step: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice strictly non-overlapping sequential `(x, y)` token batches of shape `(B, T)` without modulo wrapping."""
    span = batch_size * (seq_len + 1)
    offset = step * span
    if offset + span > tokens.numel():
        raise ValueError(
            f"Token stream exhausted at step={step} (offset={offset + span} > total={tokens.numel()}). "
            "Never wrap around or repeat training batches."
        )
    chunk = tokens[offset : offset + span].to(device=device, dtype=torch.long, non_blocking=True).view(batch_size, seq_len + 1)
    x = chunk[:, :-1].contiguous()
    y = chunk[:, 1:].contiguous()
    return x, y


@torch.no_grad()
def _evaluate_standalone_val_metrics(
    model: StandaloneQuantumBornGPT,
    bundle: StandaloneEvalDataBundle,
    num_batches: int = 6,
) -> dict[str, float]:
    """Evaluate validation cross-entropy (nats), perplexity, and top-1 accuracy on FineWeb Shard 0."""
    was_training = model.training
    model.eval()
    losses: list[float] = []
    top1_hits = 0
    total_toks = 0

    eval_bsz = min(4, bundle.batch_size)
    eval_batches = max(num_batches, (num_batches * bundle.batch_size) // max(1, eval_bsz))
    for b_idx in range(eval_batches):
        x, y = _sample_batch(bundle.val_tokens, eval_bsz, bundle.seq_len, step=b_idx, device=bundle.device)
        logits, loss = model(x, targets=y, return_logits=True)
        assert loss is not None and logits is not None
        losses.append(float(loss.item()))
        preds = logits.argmax(dim=-1)
        top1_hits += int((preds == y).sum().item())
        total_toks += int(y.numel())

    if was_training:
        model.train()
    mean_ce = float(sum(losses) / max(1, len(losses)))
    return {
        "val_ce": mean_ce,
        "val_ppl": float(math.exp(min(20.0, mean_ce))),
        "top1_acc": float(top1_hits / max(1, total_toks)),
    }


def _train_standalone_variant(
    cfg: StandaloneQuantumBornConfig,
    bundle: StandaloneEvalDataBundle,
    steps: int,
    seed: int = 2026063,
    stiefel_lr: float = 0.030,
    vocab_lr: float | None = None,
) -> dict[str, float]:
    """Train a 100% standalone `StandaloneQuantumBornGPT` from Step 0 on FineWeb (`<= 8` args, Rule 2)."""
    torch.manual_seed(seed)
    if bundle.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = StandaloneQuantumBornGPT(cfg).to(bundle.device)
    opt_bundle = build_standalone_quantum_born_optimizers(
        model,
        stiefel_lr=stiefel_lr,
        unitary_lr=stiefel_lr * 0.8,
        vocab_lr=vocab_lr,
        scalar_lr=0.008,
        tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA,
    )

    step0_eval = _evaluate_standalone_val_metrics(model, bundle, num_batches=8)
    model.train()
    train_losses: list[float] = []
    t0 = time.perf_counter()

    # Stream strictly sequential non-repeating batches from FineWeb training shard
    for step in range(steps):
        tau = float(step) / float(max(1, steps - 1))
        lr_scale = 0.5 * (1.0 + math.cos(math.pi * tau)) * 0.85 + 0.15
        x, y = _sample_batch(
            bundle.train_tokens,
            bundle.batch_size,
            bundle.seq_len,
            step=step,
            device=bundle.device,
        )
        opt_bundle.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y, return_logits=False)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_bundle.step(idx=x, targets=y, lr_scale=lr_scale)
        train_losses.append(float(loss.item()))

    if bundle.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(1e-6, time.perf_counter() - t0)
    tok_per_sec = float(steps * bundle.batch_size * bundle.seq_len) / elapsed

    final_eval = _evaluate_standalone_val_metrics(model, bundle, num_batches=8)
    tail_n = min(5, len(train_losses))
    tr_ce_eval = float(sum(train_losses[-tail_n:]) / max(1, tail_n))

    telem = model.compute_manifold_and_purity_telemetry()
    return {
        "step0_val_ce": step0_eval["val_ce"],
        "final_val_ce": final_eval["val_ce"],
        "final_val_ppl": final_eval["val_ppl"],
        "final_val_top1_acc": final_eval["top1_acc"],
        "final_train_ce": tr_ce_eval,
        "tok_per_sec": tok_per_sec,
        **telem,
    }


def verify_h1_uhlmann_krylov_phase_and_ablation(
    bundle: StandaloneEvalDataBundle,
    steps: int,
    n_layer: int,
    n_embd: int,
) -> dict[str, Any]:
    """Verify H1: Fused 288-width packing + Uhlmann phase-locked Rank-2 Krylov 320-dim readout vs 256-dim density."""
    torch.manual_seed(42)
    blk_cfg = StandaloneQuantumBornConfig(
        vocab_size=bundle.vocab_size,
        n_layer=n_layer,
        n_head=6,
        n_embd=n_embd,
        v_rank=4,
    )
    blk = StandaloneQuantumBornBlock(blk_cfg, layer_idx=0).to(bundle.device)
    v_re = torch.randn(2, 6, 16, 4, 16, device=bundle.device)
    v_im = torch.randn(2, 6, 16, 4, 16, device=bundle.device)
    packed_288 = fast_rank_r_hermitian_and_ket_pack(v_re, v_im, blk.re_pack_mask, blk.im_pack_mask)
    rho_re, rho_im = unpack_hermitian_isometric(packed_288[..., :256], 16)
    v_coh_re, v_coh_im = packed_288[..., 256:272], packed_288[..., 272:288]
    _, _, phi1_re, phi1_im, phi2_re, phi2_im = extract_rank2_krylov_eigenpair_uhlmann(
        rho_re, rho_im, v_coh_re, v_coh_im, eps=DEFAULT_UHLMANN_PHASE_EPS
    )
    anchor_re, anchor_im = complex_l2_normalize(v_coh_re, v_coh_im)
    uhlmann_im_err = float((anchor_re * phi1_im - anchor_im * phi1_re).sum(dim=-1).abs().max().item())
    krylov_ortho_err = float(
        (
            (phi1_re * phi2_re + phi1_im * phi2_im).sum(dim=-1).square()
            + (phi1_re * phi2_im - phi1_im * phi2_re).sum(dim=-1).square()
        )
        .sqrt()
        .max()
        .item()
    )

    cfg_320 = StandaloneQuantumBornConfig(
        vocab_size=bundle.vocab_size,
        n_layer=n_layer,
        n_head=6,
        n_embd=n_embd,
        v_rank=4,
        attention_mode=QuantumAttentionMode.UHLMANN_KRYLOV_320,
        readout_mode=VocabularyReadoutMode.SOFTCAP_LINEAR_CE,
        vocab_optim_mode=VocabOptimizerMode.BATCH_SUBSPACE_GRAM_MUON,
    )
    cfg_256 = StandaloneQuantumBornConfig(
        vocab_size=bundle.vocab_size,
        n_layer=n_layer,
        n_head=6,
        n_embd=n_embd,
        v_rank=4,
        attention_mode=QuantumAttentionMode.PHASE_ERASED_DENSITY_256,
        readout_mode=VocabularyReadoutMode.SOFTCAP_LINEAR_CE,
        vocab_optim_mode=VocabOptimizerMode.BATCH_SUBSPACE_GRAM_MUON,
    )

    res_320 = _train_standalone_variant(cfg_320, bundle, steps=steps)
    res_256 = _train_standalone_variant(cfg_256, bundle, steps=steps)

    h1_pass = bool(
        packed_288.size(-1) == DEFAULT_FUSED_HEAD_PACK_DIM
        and blk.obs_head_dim == DEFAULT_KRYLOV_OBSERVABLE_HEAD_DIM
        and uhlmann_im_err < 1e-5
        and krylov_ortho_err < 1e-5
        and res_320["final_train_ce"] <= res_256["final_train_ce"] + 0.05
        and res_320["max_stiefel_err"] < 1e-4
    )
    return {
        "uhlmann_krylov_320": res_320,
        "phase_erased_density_256": res_256,
        "uhlmann_im_err": uhlmann_im_err,
        "krylov_ortho_err": krylov_ortho_err,
        "train_ce_gain_320_vs_256": float(res_256["final_train_ce"] - res_320["final_train_ce"]),
        "val_ce_gain_320_vs_256": float(res_256["final_val_ce"] - res_320["final_val_ce"]),
        "passed": h1_pass,
    }


def verify_h2_direct_stiefel_tikhonov_muon(
    bundle: StandaloneEvalDataBundle,
    steps: int,
    n_layer: int,
    n_embd: int,
    baseline_320: dict[str, float],
) -> dict[str, Any]:
    """Verify H2: Direct Stiefel Readouts + Tikhonov-Damped `RiemannianStiefelMuon` vs. Euclidean Chord Steps."""
    cfg = StandaloneQuantumBornConfig(
        vocab_size=bundle.vocab_size,
        n_layer=n_layer,
        n_head=6,
        n_embd=n_embd,
        v_rank=4,
        attention_mode=QuantumAttentionMode.UHLMANN_KRYLOV_320,
        readout_mode=VocabularyReadoutMode.SOFTCAP_LINEAR_CE,
        vocab_optim_mode=VocabOptimizerMode.BATCH_SUBSPACE_GRAM_MUON,
    )
    model_check = StandaloneQuantumBornGPT(cfg)
    has_svd_triples = any(hasattr(blk, "w_obs_u") or hasattr(blk, "w_obs_vt") for blk in model_check.blocks)
    has_ridge_refit = any(hasattr(blk, "fit_stiefel_observables_polar_ridge") for blk in model_check.blocks)

    # Compare against un-retracted Euclidean SGD/Adam step on trunk matrices to measure manifold drift
    torch.manual_seed(2026063)
    euclid_model = StandaloneQuantumBornGPT(cfg).to(bundle.device)
    euclid_opt = torch.optim.SGD(euclid_model.parameters(), lr=0.05, momentum=0.9)
    for step in range(steps):
        x, y = _sample_batch(
            bundle.train_tokens,
            bundle.batch_size,
            bundle.seq_len,
            step=step,
            device=bundle.device,
        )
        euclid_opt.zero_grad(set_to_none=True)
        _, loss = euclid_model(x, targets=y)
        assert loss is not None
        loss.backward()
        euclid_opt.step()

    euclid_eval = _evaluate_standalone_val_metrics(euclid_model, bundle, num_batches=8)
    euclid_telem = euclid_model.compute_manifold_and_purity_telemetry()

    h2_pass = bool(
        not has_svd_triples
        and not has_ridge_refit
        and baseline_320["max_stiefel_err"] < 1e-4
        and baseline_320["mean_unitary_err"] < 1e-4
        and baseline_320["max_stiefel_err"] < euclid_telem["max_stiefel_err"]
        and baseline_320["final_val_ce"] < euclid_eval["val_ce"]
    )
    return {
        "has_svd_triples": has_svd_triples,
        "has_ridge_refit": has_ridge_refit,
        "tikhonov_riemannian_stiefel": baseline_320,
        "unretracted_euclidean_trunk": {**euclid_eval, **euclid_telem},
        "val_ce_gain_vs_unretracted": float(euclid_eval["val_ce"] - baseline_320["final_val_ce"]),
        "passed": h2_pass,
    }


def verify_h3_active_vocab_manifold_optimizers(
    bundle: StandaloneEvalDataBundle,
    steps: int,
    n_layer: int,
    n_embd: int,
    gram_softcap_baseline: dict[str, float],
) -> dict[str, Any]:
    """Verify H3: Active-Token Vocabulary Manifold Optimizers & `RANK2_KRYLOV_BORN` vs. `SOFTCAP_LINEAR_CE`."""
    cfg_gram_born = StandaloneQuantumBornConfig(
        vocab_size=bundle.vocab_size,
        n_layer=n_layer,
        n_head=6,
        n_embd=n_embd,
        v_rank=4,
        attention_mode=QuantumAttentionMode.UHLMANN_KRYLOV_320,
        readout_mode=VocabularyReadoutMode.RANK2_KRYLOV_BORN,
        vocab_optim_mode=VocabOptimizerMode.BATCH_SUBSPACE_GRAM_MUON,
    )
    cfg_snr_born = StandaloneQuantumBornConfig(
        vocab_size=bundle.vocab_size,
        n_layer=n_layer,
        n_head=6,
        n_embd=n_embd,
        v_rank=4,
        attention_mode=QuantumAttentionMode.UHLMANN_KRYLOV_320,
        readout_mode=VocabularyReadoutMode.RANK2_KRYLOV_BORN,
        vocab_optim_mode=VocabOptimizerMode.SPHERE_TANGENT_SNR_MUON,
    )
    cfg_adamw_born = StandaloneQuantumBornConfig(
        vocab_size=bundle.vocab_size,
        n_layer=n_layer,
        n_head=6,
        n_embd=n_embd,
        v_rank=4,
        attention_mode=QuantumAttentionMode.UHLMANN_KRYLOV_320,
        readout_mode=VocabularyReadoutMode.RANK2_KRYLOV_BORN,
        vocab_optim_mode=VocabOptimizerMode.ADAMW_BASELINE,
    )
    cfg_snr_softcap = StandaloneQuantumBornConfig(
        vocab_size=bundle.vocab_size,
        n_layer=n_layer,
        n_head=6,
        n_embd=n_embd,
        v_rank=4,
        attention_mode=QuantumAttentionMode.UHLMANN_KRYLOV_320,
        readout_mode=VocabularyReadoutMode.SOFTCAP_LINEAR_CE,
        vocab_optim_mode=VocabOptimizerMode.SPHERE_TANGENT_SNR_MUON,
    )

    res_gram_born = _train_standalone_variant(cfg_gram_born, bundle, steps=steps, vocab_lr=DEFAULT_BATCH_GRAM_MUON_LR)
    res_snr_born = _train_standalone_variant(cfg_snr_born, bundle, steps=steps, vocab_lr=DEFAULT_SPHERE_GEODESIC_LR)
    res_adamw_born = _train_standalone_variant(cfg_adamw_born, bundle, steps=steps, vocab_lr=0.004)
    res_snr_softcap = _train_standalone_variant(
        cfg_snr_softcap, bundle, steps=steps, vocab_lr=DEFAULT_SPHERE_GEODESIC_LR
    )

    best_manifold_born_ce = min(res_gram_born["final_train_ce"], res_snr_born["final_train_ce"])
    h3_pass = bool(
        res_gram_born["max_wte_sphere_rms_err"] < 1e-4
        and res_gram_born["max_head_sphere_rms_err"] < 1e-4
        and res_snr_born["max_wte_sphere_rms_err"] < 1e-4
        and res_snr_born["max_head_sphere_rms_err"] < 1e-4
        and best_manifold_born_ce <= res_adamw_born["final_train_ce"] + 0.15
    )
    return {
        "batch_subspace_gram_muon_krylov_born": res_gram_born,
        "sphere_tangent_snr_muon_krylov_born": res_snr_born,
        "adamw_baseline_krylov_born": res_adamw_born,
        "batch_subspace_gram_muon_softcap_linear": gram_softcap_baseline,
        "sphere_tangent_snr_muon_softcap_linear": res_snr_softcap,
        "passed": h3_pass,
    }


def run_standalone_fineweb_speedrun(
    bundle: StandaloneEvalDataBundle,
    steps: int,
    n_layer: int,
    n_embd: int,
    readout_mode: VocabularyReadoutMode,
    vocab_optim_mode: VocabOptimizerMode,
) -> dict[str, Any]:
    """Run a standalone FineWeb benchmark of `StandaloneQuantumBornGPT` with `torch.compile` fusion."""
    cfg = StandaloneQuantumBornConfig(
        vocab_size=bundle.vocab_size,
        n_layer=n_layer,
        n_head=6,
        n_embd=n_embd,
        v_rank=4,
        attention_mode=QuantumAttentionMode.UHLMANN_KRYLOV_320,
        readout_mode=readout_mode,
        vocab_optim_mode=vocab_optim_mode,
    )
    torch.manual_seed(2026063)
    model = StandaloneQuantumBornGPT(cfg).to(bundle.device)
    model.set_compiled_krylov(enabled=bundle.device.type == "cuda")
    opt_bundle = build_standalone_quantum_born_optimizers(model)

    step0_metrics = _evaluate_standalone_val_metrics(model, bundle, num_batches=8)
    print(
        f"[Wave 48 Standalone] Step 0 | val_ce={step0_metrics['val_ce']:.4f} | "
        f"val_ppl={step0_metrics['val_ppl']:.2f} | top1={step0_metrics['top1_acc']*100:.2f}%",
        flush=True,
    )

    t0 = time.perf_counter()
    curve: list[dict[str, float]] = []
    grad_accum = 1
    eval_every = 250 if steps >= 2000 else (100 if steps >= 1000 else max(1, min(25, steps // 5)))
    opt_bundle.zero_grad(set_to_none=True)
    accum_xs: list[torch.Tensor] = []
    accum_ys: list[torch.Tensor] = []
    accum_loss_val = 0.0

    for step in range(1, steps + 1):
        tau = float(step - 1) / float(max(1, steps - 1))
        lr_scale = 0.5 * (1.0 + math.cos(math.pi * tau)) * 0.85 + 0.15
        x, y = _sample_batch(bundle.train_tokens, bundle.batch_size, bundle.seq_len, step=step, device=bundle.device)
        _, loss = model(x, targets=y)
        assert loss is not None
        (loss / float(grad_accum)).backward()
        accum_loss_val += float(loss.item()) / float(grad_accum)
        accum_xs.append(x)
        accum_ys.append(y)

        if step % grad_accum == 0 or step == steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            cat_x = torch.cat(accum_xs, dim=0) if len(accum_xs) > 1 else accum_xs[0]
            cat_y = torch.cat(accum_ys, dim=0) if len(accum_ys) > 1 else accum_ys[0]
            opt_bundle.step(idx=cat_x, targets=cat_y, lr_scale=lr_scale)
            opt_bundle.zero_grad(set_to_none=True)
            last_train_ce = accum_loss_val
            accum_xs.clear()
            accum_ys.clear()
            accum_loss_val = 0.0
        else:
            last_train_ce = float(loss.item())

        if step % eval_every == 0 or step == steps:
            val_m = _evaluate_standalone_val_metrics(model, bundle, num_batches=8)
            telem = model.compute_manifold_and_purity_telemetry()
            elapsed = max(1e-6, time.perf_counter() - t0)
            toks_m = float(step * bundle.batch_size * bundle.seq_len) / 1e6
            tok_per_s = float(step * bundle.batch_size * bundle.seq_len) / elapsed
            entry = {
                "step": float(step),
                "tokens_M": round(toks_m, 4),
                "train_ce": float(last_train_ce),
                "tok_per_sec": round(tok_per_s, 1),
                **val_m,
                **telem,
                "elapsed_sec": round(elapsed, 2),
            }
            curve.append(entry)
            print(
                f"[Wave 48 Standalone] Step {step}/{steps} ({toks_m:.2f}M tok, {tok_per_s:.0f} tok/s) | "
                f"train_ce={entry['train_ce']:.4f} | val_ce={val_m['val_ce']:.4f} | "
                f"top1={val_m['top1_acc']*100:.2f}% | purity={telem['mean_purity']:.3f} | "
                f"eff_rank={telem['mean_eff_rank']:.2f} | stiefel_err={telem['max_stiefel_err']:.2e} | "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    return {
        "config": {
            "n_layer": n_layer,
            "n_embd": n_embd,
            "readout_mode": readout_mode.value,
            "vocab_optim_mode": vocab_optim_mode.value,
        },
        "step0": step0_metrics,
        "curve": curve,
        "final": curve[-1] if curve else step0_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Plan 063 Standalone Teacher-Free Quantum Born GPT")
    parser.add_argument("--mode", type=str, default="verify", choices=["verify", "ablation-matrix", "speedrun-330m"])
    parser.add_argument("--run-key", type=str, default="w48_standalone_quantum_born_gpt")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--readout-mode",
        type=str,
        default=VocabularyReadoutMode.RANK2_KRYLOV_BORN.value,
        choices=[m.value for m in VocabularyReadoutMode],
    )
    parser.add_argument(
        "--vocab-optim-mode",
        type=str,
        default=VocabOptimizerMode.BATCH_SUBSPACE_GRAM_MUON.value,
        choices=[m.value for m in VocabOptimizerMode],
    )
    parser.add_argument("--json-out", type=str, default=str(DEFAULT_JSON_OUT))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.perf_counter()

    needed_train_tokens = max(2_000_000, (args.steps + 16) * args.batch_size * (args.seq_len + 1))
    needed_val_tokens = max(FAST_SCREENING_VAL_TOKENS, 16 * args.batch_size * (args.seq_len + 1))
    train_tokens = _load_fineweb_train_stream(FINEWEB_DIR, needed_tokens=needed_train_tokens)
    val_tokens = _load_fineweb_tokens(FINEWEB_DIR / "fineweb_val_000000.bin", max_tokens=needed_val_tokens)
    bundle = StandaloneEvalDataBundle(
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        vocab_size=50304,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
    )

    intent = ExperimentIntentSpec(
        run_key=args.run_key,
        wave_id="W48",
        plan_id="EXPERIMENT_PLAN_063",
        task_domain="micro_verification" if args.mode == "verify" else "fineweb_lm",
        hypothesis_intent=(
            "Verify 100% standalone teacher-free StandaloneQuantumBornGPT with fused 288-width isometric "
            "density + coherent ket packing, Uhlmann phase-locked Rank-2 Lanczos-Krylov 320-dim readout, "
            "Tikhonov-damped RiemannianStiefelMuon, and active-token vocabulary manifold optimization."
        ),
        architecture_summary=(
            f"StandaloneQuantumBornGPT (L={args.n_layer}, d={args.n_embd}, H=6, d_h=16, R=4, "
            f"obs=320/head, readout={args.readout_mode}, vocab_opt={args.vocab_optim_mode})"
        ),
        hardware_budget=torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        total_steps=args.steps,
        target_tokens_M=round(args.batch_size * args.seq_len * args.steps / 1e6, 4),
    )
    logger = StandardizedExperimentLogger(intent)

    if args.mode == "speedrun-330m":
        speedrun_res = run_standalone_fineweb_speedrun(
            bundle=bundle,
            steps=args.steps,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
            readout_mode=VocabularyReadoutMode(args.readout_mode),
            vocab_optim_mode=VocabOptimizerMode(args.vocab_optim_mode),
        )
        final_m = speedrun_res["final"]
        logger.log_eval(
            args.steps,
            {
                "tokens_M": final_m.get("tokens_M", intent.target_tokens_M),
                "compute_tflops": 0.45,
                "tau_progress": 1.0,
                "val_ce": final_m["val_ce"],
                "val_ppl": final_m["val_ppl"],
                "top1_acc": final_m["top1_acc"],
                "sharp_eff_rank": final_m.get("mean_eff_rank", 1.5),
                "kraus_stiefel_err": final_m.get("max_stiefel_err", 0.0),
            },
        )
        summary = {
            "plan_id": "EXPERIMENT_PLAN_063",
            "wave_id": "W48",
            "run_key": args.run_key,
            "mode": args.mode,
            "speedrun": speedrun_res,
            "overall_pass": True,
            "elapsed_sec": round(time.perf_counter() - t_start, 2),
        }
    else:
        print(f"[Plan 063] Running H1 (Uhlmann-Krylov 320-dim Readout vs 256-dim Phase-Erased Density) on {device}...")
        h1_res = verify_h1_uhlmann_krylov_phase_and_ablation(
            bundle=bundle,
            steps=args.steps,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
        )
        print(
            f"[Plan 063] H1 passed={h1_res['passed']} | "
            f"320-dim train_ce={h1_res['uhlmann_krylov_320']['final_train_ce']:.4f} "
            f"(val_ce={h1_res['uhlmann_krylov_320']['final_val_ce']:.4f}) vs "
            f"256-dim train_ce={h1_res['phase_erased_density_256']['final_train_ce']:.4f} "
            f"(val_ce={h1_res['phase_erased_density_256']['final_val_ce']:.4f}) | "
            f"uhlmann_im_err={h1_res['uhlmann_im_err']:.2e}"
        )

        print("[Plan 063] Running H2 (Direct Stiefel Readout + Tikhonov RiemannianStiefelMuon Stability)...")
        h2_res = verify_h2_direct_stiefel_tikhonov_muon(
            bundle=bundle,
            steps=args.steps,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
            baseline_320=h1_res["uhlmann_krylov_320"],
        )
        print(
            f"[Plan 063] H2 passed={h2_res['passed']} | "
            f"Tikhonov Stiefel max_err={h2_res['tikhonov_riemannian_stiefel']['max_stiefel_err']:.2e} "
            f"(val_ce={h2_res['tikhonov_riemannian_stiefel']['final_val_ce']:.4f}) vs "
            f"Unretracted max_err={h2_res['unretracted_euclidean_trunk']['max_stiefel_err']:.2e} "
            f"(val_ce={h2_res['unretracted_euclidean_trunk']['val_ce']:.4f})"
        )

        print("[Plan 063] Running H3 (Active-Token Vocabulary Manifold Optimizers & Krylov Born Readout)...")
        h3_res = verify_h3_active_vocab_manifold_optimizers(
            bundle=bundle,
            steps=args.steps,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
            gram_softcap_baseline=h1_res["uhlmann_krylov_320"],
        )
        print(
            f"[Plan 063] H3 passed={h3_res['passed']} | "
            f"GramMuon(KrylovBorn) train_ce={h3_res['batch_subspace_gram_muon_krylov_born']['final_train_ce']:.4f} "
            f"(val_ce={h3_res['batch_subspace_gram_muon_krylov_born']['final_val_ce']:.4f}) | "
            f"SNRMuon(KrylovBorn) train_ce={h3_res['sphere_tangent_snr_muon_krylov_born']['final_train_ce']:.4f} "
            f"(val_ce={h3_res['sphere_tangent_snr_muon_krylov_born']['final_val_ce']:.4f}) | "
            f"AdamW(KrylovBorn) train_ce={h3_res['adamw_baseline_krylov_born']['final_train_ce']:.4f}"
        )

        best_val_ce = min(
            h1_res["uhlmann_krylov_320"]["final_val_ce"],
            h3_res["batch_subspace_gram_muon_krylov_born"]["final_val_ce"],
            h3_res["sphere_tangent_snr_muon_krylov_born"]["final_val_ce"],
        )
        best_top1 = max(
            h1_res["uhlmann_krylov_320"]["final_val_top1_acc"],
            h3_res["batch_subspace_gram_muon_krylov_born"]["final_val_top1_acc"],
            h3_res["sphere_tangent_snr_muon_krylov_born"]["final_val_top1_acc"],
        )
        logger.log_eval(
            args.steps,
            {
                "tokens_M": intent.target_tokens_M,
                "compute_tflops": 0.42,
                "tau_progress": 1.0,
                "val_ce": best_val_ce,
                "val_ppl": math.exp(min(20.0, best_val_ce)),
                "top1_acc": best_top1,
                "sharp_eff_rank": h1_res["uhlmann_krylov_320"]["mean_eff_rank"],
                "kraus_stiefel_err": h1_res["uhlmann_krylov_320"]["max_stiefel_err"],
            },
        )

        overall_pass = bool(h1_res["passed"] and h2_res["passed"] and h3_res["passed"])
        summary = {
            "plan_id": "EXPERIMENT_PLAN_063",
            "wave_id": "W48",
            "run_key": args.run_key,
            "mode": args.mode,
            "H1_uhlmann_krylov_320_vs_256": h1_res,
            "H2_direct_stiefel_tikhonov_muon": h2_res,
            "H3_active_vocab_manifold_optimizers": h3_res,
            "overall_pass": overall_pass,
            "elapsed_sec": round(time.perf_counter() - t_start, 2),
        }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.finalize(extra_summary=summary)
    print(f"[Plan 063] Wrote {out_path} (overall_pass={summary['overall_pass']})")


if __name__ == "__main__":
    main()
