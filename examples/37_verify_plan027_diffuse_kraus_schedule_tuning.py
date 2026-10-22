#!/usr/bin/env python3
"""Plan 027 Verification & 8 GB VRAM Diffuse Kraus Bath Schedule Tuning Suite (RTX 5060 Ti).

Implements and verifies Hypotheses H1, H2, and H3 of EXPERIMENT_PLAN_027.md on the local
NVIDIA GeForce RTX 5060 Ti (16 GB total VRAM, strictly hardware-capped to <= 8.0 GB / 50% VRAM
via `torch.cuda.set_per_process_memory_fraction(0.5, device=0)`):

1. H1 — Per-Token & Scheduled Chunked Two-Beam k-Boson CE Parity, Stiefel Kraus Retraction, & Contextual Purity Routing:
   - Verifies that `direct_chunked_two_beam_kboson_ce` supports both scalar and per-token `lambda_diff(t)` (`(N, 1)`)
     with `< 1e-5` forward loss and `< 1e-5` backward gradient parity against `compute_two_beam_kboson_logits`.
   - Verifies thin-QR Stiefel retraction `retract_kraus_stiefel()` keeps `max(||U_diff^T U_diff - I_r||_F, ||V_diff^T V_diff - I_r||_F) < 1e-5`.
   - Verifies `contextual_purity_complement` routes larger diffuse weight `lambda_diff(t)` to low-purity ambiguous tokens
     (`beta_t = 16.0`) than to high-purity sharp tokens (`beta_t = 34.0`) by `>= +0.08`.

2. H3 — 5-Arm Diffuse Kraus Bath Weight Schedule Tuning (`300` Steps = `2.46M` FineWeb10B Tokens, `L=6, d=512`):
   - Compares all 5 Diffuse Kraus Bath Schedule arms on Unified Wave-13 + Wave-15 + Two-Beam k-Boson Readout (`r=16`):
     * Arm 1 (`free_decay`): Unconstrained `0.40 * sigmoid(l_diff)` (Plan 026 baseline, `6.0940` Val CE, collapses to `lambda_diff = 0.0356`)
     * Arm 2 (`bounded_floor`): Bounded floor `lambda_diff in [0.10, 0.35]` + Stiefel Kraus retraction
     * Arm 3 (`cosine_broad_to_sharp`): Broad-to-sharp cosine curriculum (`0.26 -> 0.12`)
     * Arm 4 (`warmup_hold_anneal`): Hold `0.22` for first 35% of steps while `U_diff, V_diff` align on `St(d, r)`, then cosine-anneal `0.22 -> 0.12`
     * Arm 5 (`contextual_purity_complement`): `warmup_hold_anneal` + per-token Bloch-purity complementary routing (`gain = 0.20`)
   - Also runs the head-to-head Unified Softmax control with the `polar_snap_weights` `"embed"` exclusion fix
     (`w16_8gb_qborn_unified_softmax_polarsnap_fixed_L6_d512`) so all readout families are compared on 100% equal footing.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Enforce strict <= 8.0 GB (50% of 16 GB RTX 5060 Ti) VRAM cap BEFORE any CUDA allocation
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.5, device=0)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from qdlblocks.benchmarks.calibration_probe import (
    compute_calibration_rank_decomposition,
    compute_layerwise_dynamics_profile,
)
from qdlblocks.benchmarks.db import (
    DEFAULT_DB_PATH,
    BenchmarkSQLiteDB,
    backfill_micro_verifications,
)
from qdlblocks.benchmarks.model_card import (
    build_model_card_from_manifest_entry,
    save_model_card_atomic,
    update_model_card_telemetry,
)
from qdlblocks.layers.cooccurrence_init import (
    compute_ppmi_svd_embeddings,
)
from qdlblocks.layers.direct_chunked_ce import (
    compute_two_beam_kboson_logits,
    direct_chunked_two_beam_kboson_ce,
)
from qdlblocks.layers.hierarchical_qubit import (
    Hierarchical16QubitPOVMReadout,
    initialize_hierarchical_16qubit_with_ppmi_svd,
)
from qdlblocks.models.born_gpt import QuantumBornGPT
from qdlblocks.optim.muon import StiefelMuon, polar_snap_weights, split_muon_adamw_params

CACHE_DIR = ROOT / ".cache"
LIVE_DIR = CACHE_DIR / "live_logs"
CARDS_DIR = CACHE_DIR / "model_cards"


@torch.no_grad()
def initialize_povm_log_prior(
    model: nn.Module,
    token_ids: torch.Tensor,
    alpha: float = 0.62,
    bose_saturate_top_k: int = 16,
) -> None:
    """Initialize `model.povm_log_prior` with Bose-saturated empirical log-unigram frequencies."""
    prior_param = getattr(model, "povm_log_prior", None)
    if prior_param is None or not isinstance(prior_param, nn.Parameter):
        return
    V = int(prior_param.numel())
    flat = token_ids.reshape(-1).to(torch.int64).clamp(0, V - 1)
    counts = torch.bincount(flat, minlength=V).float() + 1.0
    probs = counts / counts.sum()
    log_p = torch.log(probs)
    if bose_saturate_top_k > 0 and bool(getattr(model, "use_two_beam_kboson_readout", False)):
        k_top = max(4, min(int(bose_saturate_top_k), V // 16))
        knee = torch.topk(log_p, k=k_top, largest=True).values[-1]
        log_p = knee - 1.25 * F.softplus((knee - log_p) / 1.25) + 0.25 * (log_p - knee).clamp_min(0.0)
    log_p = (log_p - log_p.mean()) * float(alpha)
    prior_param.copy_(log_p.to(device=prior_param.device, dtype=prior_param.dtype))


def load_fineweb_tokens(
    max_train_tokens: int = 3_000_000,
    max_val_tokens: int = 1_048_576,
) -> tuple[np.ndarray, np.ndarray]:
    """Load genuine FineWeb-Edu uint16 train (`train_full_100M.bin`) and held-out val (`val_slice.bin`) shards."""
    from qdlblocks.benchmarks.fineweb_stream import FineWebMultiShardStream

    data_dir = CACHE_DIR / "fineweb_100M"
    train_path = data_dir / "train_full_100M.bin"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing FineWeb-Edu training shard at {train_path}")
    train_mm = np.memmap(train_path, dtype=np.uint16, mode="r", offset=1024)
    n_tr = min(len(train_mm), max_train_tokens)
    train_arr = np.array(train_mm[:n_tr], dtype=np.uint16)

    stream = FineWebMultiShardStream(data_dir=data_dir, num_shards=1, prefetch_async=False)
    full_val = stream.ensure_val_shard()
    n_v = min(len(full_val), max_val_tokens)
    val_arr = np.array(full_val[:n_v], dtype=np.uint16)
    return train_arr, val_arr


def build_evaluation_batches(
    tokens_1d: np.ndarray,
    batch_size: int = 4,
    seq_len: int = 512,
    num_batches: int = 6,
    device: torch.device = torch.device("cuda"),
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Construct sequential non-overlapping `(x, y)` evaluation batches from `pos = 0` (`modded-nanogpt` protocol)."""
    from qdlblocks.benchmarks.fineweb_stream import iter_modded_nanogpt_val_batches

    stride = batch_size * seq_len
    budget = num_batches * stride
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _, x_np, y_np in iter_modded_nanogpt_val_batches(
        tokens_1d, batch_size=batch_size, seq_len=seq_len, val_tokens_budget=budget
    ):
        x = torch.from_numpy(x_np.astype(np.int64)).to(device)
        y = torch.from_numpy(y_np.astype(np.int64)).to(device)
        batches.append((x, y))
    return batches


def run_h1_per_token_kboson_stiefel_and_contextual_routing(device: torch.device) -> dict[str, Any]:
    """Verify H1: Per-token lambda_diff parity, Stiefel Kraus retraction (< 1e-5), and contextual purity routing."""
    print("\n" + "=" * 96)
    print("  [H1 Verification] Per-Token Chunked k-Boson CE Parity, Stiefel Retraction & Contextual Purity Routing")
    print("=" * 96)

    torch.manual_seed(2027)
    n_tokens = 256
    vocab_padded = 1024
    valid_vocab = 1000
    embed_dim = 128
    diffuse_rank = 16

    psi_sharp_raw = F.normalize(torch.randn(n_tokens, embed_dim, device=device, dtype=torch.float32), p=2, dim=-1)
    psi_diff_raw = torch.randn(n_tokens, diffuse_rank, device=device, dtype=torch.float32) * 0.35
    m_sharp_raw = F.normalize(torch.randn(vocab_padded, embed_dim, device=device, dtype=torch.float32), p=2, dim=-1)
    m_diff_raw = torch.randn(vocab_padded, diffuse_rank, device=device, dtype=torch.float32) * 0.35
    beta_raw = torch.linspace(16.0, 38.0, n_tokens, device=device, dtype=torch.float32)
    k_raw = torch.tensor(36.0, device=device, dtype=torch.float32)
    lam_per_tok_raw = (0.10 + 0.22 * torch.rand(n_tokens, 1, device=device, dtype=torch.float32)).detach()
    b_raw = torch.randn(vocab_padded, device=device, dtype=torch.float32) * 0.25
    targets = torch.randint(0, valid_vocab, (n_tokens,), device=device, dtype=torch.long)

    psi_s1 = psi_sharp_raw.clone().requires_grad_(True)
    psi_d1 = psi_diff_raw.clone().requires_grad_(True)
    m_s1 = m_sharp_raw.clone().requires_grad_(True)
    m_d1 = m_diff_raw.clone().requires_grad_(True)
    beta_1 = beta_raw.clone().requires_grad_(True)
    k_1 = k_raw.clone().requires_grad_(True)
    lam_1 = lam_per_tok_raw.clone().requires_grad_(True)
    b_1 = b_raw.clone().requires_grad_(True)

    loss_chunked = direct_chunked_two_beam_kboson_ce(
        psi_s1,
        psi_d1,
        m_s1,
        m_d1,
        beta_1,
        k_1,
        lam_1,
        targets,
        log_prior=b_1,
        chunk_size=64,
        valid_vocab_size=valid_vocab,
    )
    loss_chunked.backward()

    psi_s2 = psi_sharp_raw.clone().requires_grad_(True)
    psi_d2 = psi_diff_raw.clone().requires_grad_(True)
    m_s2 = m_sharp_raw.clone().requires_grad_(True)
    m_d2 = m_diff_raw.clone().requires_grad_(True)
    beta_2 = beta_raw.clone().requires_grad_(True)
    k_2 = k_raw.clone().requires_grad_(True)
    lam_2 = lam_per_tok_raw.clone().requires_grad_(True)
    b_2 = b_raw.clone().requires_grad_(True)

    logits_ref = compute_two_beam_kboson_logits(
        psi_s2,
        psi_d2,
        m_s2,
        m_d2,
        beta_2,
        k_2,
        lam_2,
        log_prior=b_2,
    )
    loss_ref = F.cross_entropy(logits_ref[:, :valid_vocab], targets)
    loss_ref.backward()

    loss_diff = float((loss_chunked.detach() - loss_ref.detach()).abs().item())
    grad_diffs = {
        "dpsi_sharp": float((psi_s1.grad - psi_s2.grad).abs().max().item()),
        "dpsi_diff": float((psi_d1.grad - psi_d2.grad).abs().max().item()),
        "dm_sharp": float((m_s1.grad - m_s2.grad).abs().max().item()),
        "dm_diff": float((m_d1.grad - m_d2.grad).abs().max().item()),
        "dbeta": float((beta_1.grad - beta_2.grad).abs().max().item()),
        "dk": float((k_1.grad - k_2.grad).abs().max().item()),
        "dlam_per_tok": float((lam_1.grad - lam_2.grad).abs().max().item()),
        "dlog_prior": float((b_1.grad - b_2.grad).abs().max().item()),
    }
    max_grad_diff = max(grad_diffs.values())

    # Stiefel retraction & contextual purity routing check
    readout = Hierarchical16QubitPOVMReadout(
        vocab_size=256,
        embed_dim=128,
        num_registers=8,
        register_dim=8,
        num_higher_states=32,
        diffuse_rank=16,
        diffuse_schedule="contextual_purity_complement",
        lambda_diff_min=0.08,
        lambda_diff_max=0.35,
        lambda_diff_start=0.22,
        lambda_diff_end=0.12,
        contextual_purity_gain=0.20,
    ).to(device)
    with torch.no_grad():
        readout.kraus_u_diff.add_(0.30 * torch.randn_like(readout.kraus_u_diff))
        readout.kraus_v_diff.add_(0.30 * torch.randn_like(readout.kraus_v_diff))
    stiefel_err = readout.retract_kraus_stiefel()

    readout.set_diffuse_schedule_progress(35, 300)
    beta_probe = torch.tensor([16.0, 34.0], device=device, dtype=torch.float32)
    lam_probe = readout.compute_scheduled_lambda_diff(
        beta_t=beta_probe,
        k_eff=torch.tensor(38.0, device=device, dtype=torch.float32),
    ).reshape(-1)
    lam_low_purity = float(lam_probe[0].item())
    lam_high_purity = float(lam_probe[1].item())
    contextual_routing_spread = round(lam_low_purity - lam_high_purity, 6)

    passed = bool(
        loss_diff < 1e-5
        and max_grad_diff < 1e-5
        and stiefel_err < 1e-5
        and contextual_routing_spread >= 0.08
    )
    print(
        f"  * [H1 Parity & Routing] |dLoss|={loss_diff:.2e} | max|dGrad|={max_grad_diff:.2e} | "
        f"StiefelErr={stiefel_err:.2e} | lam(beta=16)={lam_low_purity:.4f} vs lam(beta=34)={lam_high_purity:.4f} "
        f"(Spread=+{contextual_routing_spread:.4f}) | PASS={passed}"
    )
    del readout
    return {
        "passed": passed,
        "loss_abs_diff": loss_diff,
        "max_grad_abs_diff": max_grad_diff,
        "grad_abs_diffs": grad_diffs,
        "kraus_stiefel_retraction_err": stiefel_err,
        "lambda_diff_at_beta16_low_purity": round(lam_low_purity, 4),
        "lambda_diff_at_beta34_high_purity": round(lam_high_purity, 4),
        "contextual_purity_routing_spread": contextual_routing_spread,
    }


@torch.no_grad()
def evaluate_model_with_rank_and_dynamics(
    model: nn.Module,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    valid_vocab_size: int = 50257,
    compute_dynamics: bool = True,
) -> dict[str, Any]:
    """Evaluate Val CE, 5-bucket Target-Rank CE Decomposition, Layerwise Dynamics, and Kraus Stiefel / Diffuse Rank."""
    was_training = model.training
    model.eval()

    logits_chunks: list[torch.Tensor] = []
    targets_chunks: list[torch.Tensor] = []
    hier_entropies: list[float] = []
    hier_top1_masses: list[float] = []
    diffuse_eff_ranks: list[float] = []
    sharp_eff_ranks: list[float] = []
    kraus_stiefel_err = 0.0

    for b_idx, (x, y) in enumerate(val_batches):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x, targets=None)
        v_eff = min(int(valid_vocab_size), int(logits.size(-1)))
        logits_chunks.append(
            logits[..., :v_eff].reshape(-1, v_eff).detach().to(device="cpu", dtype=torch.bfloat16)
        )
        targets_chunks.append(y.reshape(-1).detach())

        hq = getattr(model, "hierarchical_qubit_embed", None)
        if hq is not None and getattr(hq, "num_higher_states", 0) > 0:
            hier_entropies.append(float(hq.last_hierarchy_entropy))
            hier_top1_masses.append(float(hq.last_hierarchy_top1_mass))

        if (
            b_idx == 0
            and isinstance(model, QuantumBornGPT)
            and getattr(model, "use_two_beam_kboson_readout", False)
            and model.hierarchical_povm_readout is not None
            and model.hierarchical_qubit_embed is not None
        ):
            hr = model.hierarchical_povm_readout
            eye_r = torch.eye(hr.diffuse_rank, device=hr.kraus_u_diff.device, dtype=torch.float32)
            u_err = float(torch.linalg.norm(hr.kraus_u_diff.float().T @ hr.kraus_u_diff.float() - eye_r).item())
            v_err = float(torch.linalg.norm(hr.kraus_v_diff.float().T @ hr.kraus_v_diff.float() - eye_r).item())
            kraus_stiefel_err = max(u_err, v_err)

            psi_f, _ = model.forward_features(x, return_layer_states=False)
            base_h = model.povm_readout.povm_head if model.povm_readout is not None else None
            psi_s, psi_d, _, _, _, _ = hr.get_two_beam_kboson_factors(
                model.hierarchical_qubit_embed,
                psi_f,
                psi_beam2=model._last_psi_beam2,
                base_povm_head=base_h,
            )
            s_diff = torch.linalg.svdvals(psi_d.reshape(-1, psi_d.size(-1)).float())
            p_diff = s_diff / s_diff.sum().clamp_min(1e-12)
            r_diff = float(torch.exp(-(p_diff * torch.log(p_diff.clamp_min(1e-12))).sum()).item())
            diffuse_eff_ranks.append(r_diff)

            s_sharp = torch.linalg.svdvals(psi_s.reshape(-1, psi_s.size(-1)).float())
            p_sharp = s_sharp / s_sharp.sum().clamp_min(1e-12)
            r_sharp = float(torch.exp(-(p_sharp * torch.log(p_sharp.clamp_min(1e-12))).sum()).item())
            sharp_eff_ranks.append(r_sharp)
            del psi_f, psi_s, psi_d, s_diff, p_diff, s_sharp, p_sharp

        del logits

    all_logits = torch.cat(logits_chunks, dim=0)
    all_targets = torch.cat(targets_chunks, dim=0)
    del logits_chunks, targets_chunks

    rank_decomp = compute_calibration_rank_decomposition(
        all_logits,
        all_targets,
        valid_vocab_size=valid_vocab_size,
        num_ece_bins=10,
        chunk_size=1024,
    )
    del all_logits, all_targets

    if compute_dynamics:
        dyn = compute_layerwise_dynamics_profile(
            model,
            val_batches[0][0],
            val_batches[0][1],
            valid_vocab_size=valid_vocab_size,
        )
    else:
        dyn = {
            "num_layers": int(getattr(model, "num_layers", 6)),
            "layer_angular_velocity_deg": [],
            "mean_angular_velocity_deg": 0.0,
            "layer_effective_ranks": [],
            "final_layer_effective_rank": 32.0,
            "layer_participation_ratios": [],
            "final_layer_participation_ratio": 16.0,
            "grad_norm_embed_readout": 0.0,
            "grad_norm_backbone_muon": 0.0,
            "embed_to_backbone_grad_ratio": 0.0,
            "stiefel_iso_err": 0.0,
        }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if was_training:
        model.train()

    val_ce = float(rank_decomp["total_ce"])
    val_ppl = float(rank_decomp["total_ppl"])
    mean_hier_ent = sum(hier_entropies) / len(hier_entropies) if hier_entropies else 0.0
    mean_hier_top1 = sum(hier_top1_masses) / len(hier_top1_masses) if hier_top1_masses else 0.0
    mean_beta = float(getattr(model, "_last_mean_beta", 30.0))
    kboson_k = float(getattr(model, "_last_kboson_copies", 34.0))
    lam_diff = float(getattr(model, "_last_lambda_diff", 0.18))
    r_diff_val = sum(diffuse_eff_ranks) / len(diffuse_eff_ranks) if diffuse_eff_ranks else 0.0
    r_sharp_val = (
        sum(sharp_eff_ranks) / len(sharp_eff_ranks)
        if sharp_eff_ranks
        else float(dyn["final_layer_effective_rank"])
    )

    return {
        "val_ce": round(val_ce, 4),
        "val_ppl": round(val_ppl, 2),
        "top1_acc": round(float(rank_decomp["top1_acc_pct"]), 2),
        "top5_acc": round(float(rank_decomp["top5_acc_pct"]), 2),
        "top50_acc": round(float(rank_decomp["top50_acc_pct"]), 2),
        "top5_minus_top1_gap": round(float(rank_decomp["top5_minus_top1_gap_pct"]), 2),
        "med_rank": int(round(float(rank_decomp["median_rank"]))),
        "ece": round(float(rank_decomp["ece"]), 4),
        "mean_prediction_entropy_nats": round(float(rank_decomp["mean_prediction_entropy_nats"]), 4),
        "logit_std": round(float(rank_decomp["logit_std"]), 4),
        "iso_err": round(float(dyn["stiefel_iso_err"]), 4),
        "kraus_stiefel_err": round(kraus_stiefel_err, 6),
        "hierarchy_entropy_nats": round(mean_hier_ent, 4),
        "hierarchy_top1_mass": round(mean_hier_top1, 4),
        "povm_beta": round(mean_beta, 2),
        "kboson_copies": round(kboson_k, 3),
        "lambda_diff": round(lam_diff, 4),
        "diffuse_eff_rank": round(r_diff_val, 2),
        "sharp_eff_rank": round(r_sharp_val, 2),
        "seed_eff_rank": round(float(dyn["final_layer_effective_rank"]), 2),
        "seed_participation_ratio": round(float(dyn["final_layer_participation_ratio"]), 2),
        "calibration_decomposition": rank_decomp,
        "layerwise_dynamics": dyn,
    }


def build_optimizers_for_spec(
    model: nn.Module,
) -> tuple[StiefelMuon | None, torch.optim.AdamW, float, float]:
    """Build matched StiefelMuon + AdamW optimizers with zero-weight-decay Kraus & readout group."""
    muon_params, adam_params = split_muon_adamw_params(model)
    filtered_muon: list[nn.Parameter] = []
    for p in muon_params:
        if p.ndim == 3 or (p.ndim == 2 and min(p.shape) >= 32):
            filtered_muon.append(p)
        else:
            adam_params.append(p)

    is_kboson = getattr(model, "use_two_beam_kboson_readout", False)
    is_softmax = getattr(model, "use_last_layer_softmax", False)

    base_muon_lr = 0.018
    base_adam_lr = 0.003
    head_lr = 0.012

    muon_opt = (
        StiefelMuon(
            filtered_muon,
            lr=base_muon_lr,
            momentum=0.95,
            nesterov=True,
            weight_decay=0.01,
            decay_mode="cautious",
            stiefel_penalty=0.04,
            retract_every=1,
            polar_method="polar_express",
            use_normuon=True,
            use_twin_rail=True,
        )
        if filtered_muon
        else None
    )

    gain_ids: set[int] = set()
    ngram_ids: set[int] = set()
    hr = getattr(model, "hierarchical_povm_readout", None)
    if (is_kboson or is_softmax) and hr is not None:
        pr_mod = getattr(model, "povm_readout", None)
        base_h = getattr(pr_mod, "povm_head", None) if pr_mod is not None else None
        for gp in (
            hr.token_radial_scale,
            hr.readout_rms_gain,
            hr.qubit_readout_gain,
            hr.hier_readout_gain,
            hr.povm_odd_qubits,
            hr.hier_even_proj,
            hr.vocab_higher_affinity,
            hr.kraus_u_diff,
            hr.kraus_v_diff,
            hr.kboson_copies,
            hr.diffuse_mix_logit,
            base_h,
            getattr(model, "povm_log_prior", None),
        ):
            if gp is not None and gp.requires_grad:
                gain_ids.add(id(gp))

    ng_mod = getattr(model, "ngram_embed", None)
    if ng_mod is not None:
        for np_p in ng_mod.parameters():
            if np_p.requires_grad:
                ngram_ids.add(id(np_p))

    body_group = [p for p in adam_params if id(p) not in (gain_ids | ngram_ids) and p.requires_grad]
    param_groups: list[dict[str, Any]] = [
        {"params": body_group, "lr": base_adam_lr, "base_lr": base_adam_lr, "weight_decay": 0.01}
    ]
    if gain_ids:
        gain_group = [p for p in adam_params if id(p) in gain_ids and p.requires_grad]
        param_groups.append({"params": gain_group, "lr": head_lr, "base_lr": head_lr, "weight_decay": 0.0})
    if ngram_ids:
        ngram_lr = 0.025
        ngram_group = [p for p in adam_params if id(p) in ngram_ids and p.requires_grad]
        param_groups.append({"params": ngram_group, "lr": ngram_lr, "base_lr": ngram_lr, "weight_decay": 0.0})

    adam_opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
    return muon_opt, adam_opt, base_muon_lr, base_adam_lr


def set_wsd_lr(
    step: int,
    total_steps: int,
    muon_opt: StiefelMuon | None,
    adam_opt: torch.optim.AdamW,
    base_muon_lr: float,
    base_adam_lr: float,
    warmup_steps: int = 15,
    decay_fraction: float = 0.40,
    min_lr_ratio: float = 0.10,
) -> None:
    """Apply Warmup-Stable-Decay (WSD) learning-rate schedule respecting per-group base_lr."""
    if step < warmup_steps:
        mult = float(step + 1) / float(max(1, warmup_steps))
    else:
        decay_start = int(total_steps * (1.0 - decay_fraction))
        if step < decay_start:
            mult = 1.0
        else:
            prog = float(step - decay_start) / float(max(1, total_steps - decay_start))
            mult = 1.0 - (1.0 - min_lr_ratio) * min(1.0, max(0.0, prog))
    if muon_opt is not None:
        for g in muon_opt.param_groups:
            g["lr"] = base_muon_lr * mult
    for g in adam_opt.param_groups:
        g_base = float(g.get("base_lr", base_adam_lr))
        g["lr"] = g_base * mult


def build_unified_L6_d512_model(
    spec: dict[str, Any],
    seq_len: int,
    prior_tokens_t: torch.Tensor,
    ppmi_emb_256: torch.Tensor,
    work_tokens: np.ndarray,
    device: torch.device,
) -> QuantumBornGPT:
    """Instantiate a matched Unified W13+W15 L=6, d=512 QuantumBornGPT model for a given schedule/readout spec."""
    is_softmax = bool(spec.get("last_layer_softmax", False))
    model = QuantumBornGPT(
        vocab_size=50304,
        embed_dim=512,
        seed_dim=256,
        num_heads=8,
        num_layers=6,
        block_size=seq_len,
        readout_mode="asymmetric",
        povm_temperature=0.030,
        use_overcomplete_two_beam=True,
        frame_expansion=2.0,
        use_purified_attn=True,
        true_layer0_v_residual=True,
        use_gated_spherical_residual=True,
        use_dynamic_povm_beta=True,
        use_povm_log_prior=True,
        use_ortho_decouple=True,
        use_hierarchical_16qubit=True,
        qubit_registers=16,
        qubit_amplitudes=16,
        num_higher_states=256,
        use_block_hierarchy_bridge=True,
        bridge_causal_memory=False,
        use_last_layer_softmax=is_softmax,
        logit_softcap=15.0,
        use_two_beam_kboson_readout=not is_softmax,
        diffuse_rank=int(spec.get("diffuse_rank", 16)),
        diffuse_schedule=str(spec.get("diffuse_schedule", "free_decay")),
        lambda_diff_min=float(spec.get("lambda_diff_min", 0.08)),
        lambda_diff_max=float(spec.get("lambda_diff_max", 0.35)),
        lambda_diff_start=float(spec.get("lambda_diff_start", 0.22)),
        lambda_diff_end=float(spec.get("lambda_diff_end", 0.12)),
        contextual_purity_gain=float(spec.get("contextual_purity_gain", 0.20)),
        num_value_embeddings=3,
        use_zero_half_rope=True,
        use_depth_scaled_residual=True,
        window_size=384,
        use_ngram_hash=True,
        ngram_vocab_size=65536,
        ngram_inject_scale=0.12,
        use_mudd_and_gates=True,
        use_smear_gate=True,
        use_activation_checkpointing=True,
    ).to(device)
    initialize_povm_log_prior(
        model,
        token_ids=prior_tokens_t,
        alpha=0.62 if not is_softmax else 0.55,
        bose_saturate_top_k=16 if not is_softmax else 0,
    )
    initialize_hierarchical_16qubit_with_ppmi_svd(
        model,
        ppmi_embeddings=ppmi_emb_256,
        token_shards=[work_tokens],
        alpha=0.75,
        kmeans_iters=8,
        device=device,
    )
    if not is_softmax:
        with torch.no_grad():
            assert model.hierarchical_povm_readout is not None
            model.hierarchical_povm_readout.kboson_copies.fill_(38.0)
    return model


def run_plan027_diffuse_schedule_tuning_suite(
    train_arr: np.ndarray,
    val_arr: np.ndarray,
    ppmi_emb_256: torch.Tensor,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    """Run the 5 Diffuse Kraus Bath Schedule arms + re-verified Unified Softmax for 300 steps (<= 8.0 GB VRAM)."""
    print("\n" + "=" * 96)
    print("  [H3 5-Arm Diffuse Kraus Schedule Tuning + Unified Softmax Re-Run] 300 Steps / 2.46M Tok (<= 8.0 GB VRAM)")
    print("=" * 96)

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    CARDS_DIR.mkdir(parents=True, exist_ok=True)

    work_tokens = train_arr[:2_600_000]
    work_tokens_gpu = torch.from_numpy(work_tokens.view(np.int16)).to(device=device)
    prior_tokens_t = torch.from_numpy(work_tokens[:2_000_000].astype(np.int64))
    val_batches = build_evaluation_batches(val_arr, batch_size=4, seq_len=512, num_batches=6, device=device)

    specs: list[dict[str, Any]] = [
        {
            "key": "w15_8gb_classical_leaderboard_exact_L6_d512",
            "session": "local_rtx5060ti_8gb",
            "hardware": "NVIDIA GeForce RTX 5060 Ti (<=8.0GB Cap)",
            "arch_family": "classical_modded",
            "schedule_mode": "N/A (Classical Softcap=15)",
            "label": "Plan 025 8GB Classical Leaderboard-Exact Baseline: ClassicalModdedGPT (L=6, d=512, H=8, d_ff=2048, VE=3, SkipAttn(3), Softcap=15.0, 300 Steps / 2.46M Tok)",
            "short_label": "W15 8GB Classical Leaderboard-Exact (L=6, d=512, Softcap=15)",
            "color": "#79c0ff",
            "cli_args": "--backend cuda --architecture classical_modded --leaderboard-exact-classical --num-layers 6 --embed-dim 512 --num-heads 8 --hidden-dim 2048",
        },
        {
            "key": "w15_8gb_qborn_16qubit_256hier_povm_L6_d512",
            "session": "local_rtx5060ti_8gb",
            "hardware": "NVIDIA GeForce RTX 5060 Ti (<=8.0GB Cap)",
            "arch_family": "quantum_born",
            "schedule_mode": "N/A (Single-Beam Pure POVM)",
            "label": "Plan 025 8GB Wave-15 16x16-Qubit + 256-State Hierarchy Pure POVM Control (L=6, d=512, H=8, 300 Steps / 2.46M Tok)",
            "short_label": "W15 8GB QBorn 16x16-Qubit + 256-Hier Pure POVM Control (L=6, d=512)",
            "color": "#ff2e93",
            "cli_args": "--backend cuda --architecture quantum_born --num-layers 6 --embed-dim 512 --num-heads 8 --hierarchical-16qubit --qubit-registers 16 --qubit-amplitudes 16 --num-higher-states 256",
        },
        {
            "key": "w16_8gb_qborn_unified_softmax_polarsnap_fixed_L6_d512",
            "session": "local_rtx5060ti_8gb",
            "hardware": "NVIDIA GeForce RTX 5060 Ti (<=8.0GB Cap)",
            "arch_family": "quantum_born",
            "schedule_mode": "N/A (Unified Calibrated Softmax + polar_snap Fix)",
            "last_layer_softmax": True,
            "label": "Plan 027 8GB Unified Wave-13 + Wave-15 + Calibrated Last-Layer Softmax (with polar_snap_weights embed exclusion fix, L=6, d=512, H=8, VE=3, ZeroHalfRoPE, DepthScaledRes, Win=384, Hashed N-Gram, 300 Steps / 2.46M Tok)",
            "short_label": "W16 8GB QBorn Unified W13+W15 + Softmax (polar_snap Fixed, L=6, d=512)",
            "color": "#00f5d4",
            "cli_args": (
                "--backend cuda --architecture quantum_born --epochs 1 --num-shards 1 "
                "--batch-size 16 --grad-accum 4 --seq-len 512 --schedule wsd --wsd-decay-fraction 0.40 --wsd-min-lr-ratio 0.10 "
                "--num-layers 6 --embed-dim 512 --num-heads 8 --overcomplete-two-beam --frame-expansion 2.0 --seed-dim 256 "
                "--hierarchical-16qubit --qubit-registers 16 --qubit-amplitudes 16 --num-higher-states 256 "
                "--embed-only-causal-memory --last-layer-softmax --logit-softcap 15.0 --num-value-embeddings 3 "
                "--zero-half-rope --depth-scaled-residual --sliding-window 384 --ngram-hash --ngram-vocab-size 65536 "
                "--readout-mode asymmetric --dynamic-povm-beta --ortho-decouple --povm-log-prior --gated-spherical-residual "
                "--purified-born-attn --true-layer0-v-residual --ppmi-svd-init --ppmi-svd-shards 9 --ppmi-svd-alpha 0.75 "
                "--polar-express --normuon --twin-rail-muon --mudd-and-gates --smear-gate --muon-lr 0.018 --optimizer-mode muon_adamw"
            ),
        },
        {
            "key": "w15_8gb_qborn_16qubit_256hier_twobeam_kboson_unified_L6_d512",
            "session": "local_rtx5060ti_8gb",
            "hardware": "NVIDIA GeForce RTX 5060 Ti (<=8.0GB Cap)",
            "arch_family": "quantum_born",
            "schedule_mode": "free_decay",
            "diffuse_schedule": "free_decay",
            "diffuse_rank": 16,
            "label": "Plan 026/027 Arm 1 (free_decay): Unified W13+W15 + Two-Beam k-Boson Readout (r=16, Unconstrained 0.40*sigmoid(l_diff), L=6, d=512, 300 Steps / 2.46M Tok)",
            "short_label": "W16 8GB Unified k-Boson Arm 1: free_decay (Plan 026 Baseline)",
            "color": "#ffd700",
            "cli_args": (
                "--backend cuda --architecture quantum_born --num-layers 6 --embed-dim 512 --num-heads 8 "
                "--overcomplete-two-beam --frame-expansion 2.0 --hierarchical-16qubit --qubit-registers 16 "
                "--qubit-amplitudes 16 --num-higher-states 256 --embed-only-causal-memory "
                "--two-beam-kboson-readout --diffuse-rank 16 --diffuse-schedule free_decay"
            ),
        },
        {
            "key": "w16_8gb_qborn_unified_kboson_sched_bounded_floor_L6_d512",
            "session": "local_rtx5060ti_8gb",
            "hardware": "NVIDIA GeForce RTX 5060 Ti (<=8.0GB Cap)",
            "arch_family": "quantum_born",
            "schedule_mode": "bounded_floor",
            "diffuse_schedule": "bounded_floor",
            "diffuse_rank": 16,
            "lambda_diff_min": 0.10,
            "lambda_diff_max": 0.35,
            "label": "Plan 027 Arm 2 (bounded_floor): Unified W13+W15 + Two-Beam k-Boson Readout (r=16, Bounded Floor [0.10, 0.35] + Stiefel Kraus Retraction, L=6, d=512, 300 Steps / 2.46M Tok)",
            "short_label": "W16 8GB Unified k-Boson Arm 2: bounded_floor [0.10, 0.35]",
            "color": "#ff9e00",
            "cli_args": (
                "--backend cuda --architecture quantum_born --epochs 1 --num-shards 1 "
                "--batch-size 16 --grad-accum 4 --seq-len 512 --schedule wsd --wsd-decay-fraction 0.40 --wsd-min-lr-ratio 0.10 "
                "--num-layers 6 --embed-dim 512 --num-heads 8 --overcomplete-two-beam --frame-expansion 2.0 --seed-dim 256 "
                "--hierarchical-16qubit --qubit-registers 16 --qubit-amplitudes 16 --num-higher-states 256 "
                "--embed-only-causal-memory --two-beam-kboson-readout --diffuse-rank 16 --diffuse-schedule bounded_floor "
                "--lambda-diff-min 0.10 --lambda-diff-max 0.35 --num-value-embeddings 3 --zero-half-rope --depth-scaled-residual "
                "--sliding-window 384 --ngram-hash --ngram-vocab-size 65536 --readout-mode asymmetric --dynamic-povm-beta "
                "--ortho-decouple --povm-log-prior --gated-spherical-residual --purified-born-attn --true-layer0-v-residual "
                "--ppmi-svd-init --ppmi-svd-shards 9 --ppmi-svd-alpha 0.75 --polar-express --normuon --twin-rail-muon "
                "--mudd-and-gates --smear-gate --muon-lr 0.018 --optimizer-mode muon_adamw"
            ),
        },
        {
            "key": "w16_8gb_qborn_unified_kboson_sched_cosine_broad_to_sharp_L6_d512",
            "session": "local_rtx5060ti_8gb",
            "hardware": "NVIDIA GeForce RTX 5060 Ti (<=8.0GB Cap)",
            "arch_family": "quantum_born",
            "schedule_mode": "cosine_broad_to_sharp",
            "diffuse_schedule": "cosine_broad_to_sharp",
            "diffuse_rank": 16,
            "lambda_diff_min": 0.08,
            "lambda_diff_max": 0.35,
            "lambda_diff_start": 0.26,
            "lambda_diff_end": 0.12,
            "label": "Plan 027 Arm 3 (cosine_broad_to_sharp): Unified W13+W15 + Two-Beam k-Boson Readout (r=16, Broad-to-Sharp Cosine Curriculum 0.26->0.12 + Stiefel Kraus Retraction, L=6, d=512, 300 Steps / 2.46M Tok)",
            "short_label": "W16 8GB Unified k-Boson Arm 3: cosine_broad_to_sharp (0.26->0.12)",
            "color": "#a8ff78",
            "cli_args": (
                "--backend cuda --architecture quantum_born --epochs 1 --num-shards 1 "
                "--batch-size 16 --grad-accum 4 --seq-len 512 --schedule wsd --wsd-decay-fraction 0.40 --wsd-min-lr-ratio 0.10 "
                "--num-layers 6 --embed-dim 512 --num-heads 8 --overcomplete-two-beam --frame-expansion 2.0 --seed-dim 256 "
                "--hierarchical-16qubit --qubit-registers 16 --qubit-amplitudes 16 --num-higher-states 256 "
                "--embed-only-causal-memory --two-beam-kboson-readout --diffuse-rank 16 --diffuse-schedule cosine_broad_to_sharp "
                "--lambda-diff-min 0.08 --lambda-diff-max 0.35 --lambda-diff-start 0.26 --lambda-diff-end 0.12 "
                "--num-value-embeddings 3 --zero-half-rope --depth-scaled-residual --sliding-window 384 --ngram-hash "
                "--ngram-vocab-size 65536 --readout-mode asymmetric --dynamic-povm-beta --ortho-decouple --povm-log-prior "
                "--gated-spherical-residual --purified-born-attn --true-layer0-v-residual --ppmi-svd-init --ppmi-svd-shards 9 "
                "--ppmi-svd-alpha 0.75 --polar-express --normuon --twin-rail-muon --mudd-and-gates --smear-gate "
                "--muon-lr 0.018 --optimizer-mode muon_adamw"
            ),
        },
        {
            "key": "w16_8gb_qborn_unified_kboson_sched_warmup_hold_anneal_L6_d512",
            "session": "local_rtx5060ti_8gb",
            "hardware": "NVIDIA GeForce RTX 5060 Ti (<=8.0GB Cap)",
            "arch_family": "quantum_born",
            "schedule_mode": "warmup_hold_anneal",
            "diffuse_schedule": "warmup_hold_anneal",
            "diffuse_rank": 16,
            "lambda_diff_min": 0.08,
            "lambda_diff_max": 0.35,
            "lambda_diff_start": 0.22,
            "lambda_diff_end": 0.12,
            "label": "Plan 027 Arm 4 (warmup_hold_anneal): Unified W13+W15 + Two-Beam k-Boson Readout (r=16, Align-Hold-Anneal 0.22->0.12 + Stiefel Kraus Retraction, L=6, d=512, 300 Steps / 2.46M Tok)",
            "short_label": "W16 8GB Unified k-Boson Arm 4: warmup_hold_anneal (0.22->0.12)",
            "color": "#00ff87",
            "cli_args": (
                "--backend cuda --architecture quantum_born --epochs 1 --num-shards 1 "
                "--batch-size 16 --grad-accum 4 --seq-len 512 --schedule wsd --wsd-decay-fraction 0.40 --wsd-min-lr-ratio 0.10 "
                "--num-layers 6 --embed-dim 512 --num-heads 8 --overcomplete-two-beam --frame-expansion 2.0 --seed-dim 256 "
                "--hierarchical-16qubit --qubit-registers 16 --qubit-amplitudes 16 --num-higher-states 256 "
                "--embed-only-causal-memory --two-beam-kboson-readout --diffuse-rank 16 --diffuse-schedule warmup_hold_anneal "
                "--lambda-diff-min 0.08 --lambda-diff-max 0.35 --lambda-diff-start 0.22 --lambda-diff-end 0.12 "
                "--num-value-embeddings 3 --zero-half-rope --depth-scaled-residual --sliding-window 384 --ngram-hash "
                "--ngram-vocab-size 65536 --readout-mode asymmetric --dynamic-povm-beta --ortho-decouple --povm-log-prior "
                "--gated-spherical-residual --purified-born-attn --true-layer0-v-residual --ppmi-svd-init --ppmi-svd-shards 9 "
                "--ppmi-svd-alpha 0.75 --polar-express --normuon --twin-rail-muon --mudd-and-gates --smear-gate "
                "--muon-lr 0.018 --optimizer-mode muon_adamw"
            ),
        },
        {
            "key": "w16_8gb_qborn_unified_kboson_sched_contextual_purity_L6_d512",
            "session": "local_rtx5060ti_8gb",
            "hardware": "NVIDIA GeForce RTX 5060 Ti (<=8.0GB Cap)",
            "arch_family": "quantum_born",
            "schedule_mode": "contextual_purity_complement",
            "diffuse_schedule": "contextual_purity_complement",
            "diffuse_rank": 16,
            "lambda_diff_min": 0.08,
            "lambda_diff_max": 0.35,
            "lambda_diff_start": 0.22,
            "lambda_diff_end": 0.12,
            "contextual_purity_gain": 0.20,
            "label": "Plan 027 Arm 5 (contextual_purity_complement): Unified W13+W15 + Two-Beam k-Boson Readout (r=16, Per-Token Bloch-Purity Complementary Routing gain=0.20 + Stiefel Kraus Retraction, L=6, d=512, 300 Steps / 2.46M Tok)",
            "short_label": "W16 8GB Unified k-Boson Arm 5: contextual_purity_complement (gain=0.20)",
            "color": "#f72585",
            "cli_args": (
                "--backend cuda --architecture quantum_born --epochs 1 --num-shards 1 "
                "--batch-size 16 --grad-accum 4 --seq-len 512 --schedule wsd --wsd-decay-fraction 0.40 --wsd-min-lr-ratio 0.10 "
                "--num-layers 6 --embed-dim 512 --num-heads 8 --overcomplete-two-beam --frame-expansion 2.0 --seed-dim 256 "
                "--hierarchical-16qubit --qubit-registers 16 --qubit-amplitudes 16 --num-higher-states 256 "
                "--embed-only-causal-memory --two-beam-kboson-readout --diffuse-rank 16 "
                "--diffuse-schedule contextual_purity_complement --contextual-purity-gain 0.20 "
                "--lambda-diff-min 0.08 --lambda-diff-max 0.35 --lambda-diff-start 0.22 --lambda-diff-end 0.12 "
                "--num-value-embeddings 3 --zero-half-rope --depth-scaled-residual --sliding-window 384 --ngram-hash "
                "--ngram-vocab-size 65536 --readout-mode asymmetric --dynamic-povm-beta --ortho-decouple --povm-log-prior "
                "--gated-spherical-residual --purified-born-attn --true-layer0-v-residual --ppmi-svd-init --ppmi-svd-shards 9 "
                "--ppmi-svd-alpha 0.75 --polar-express --normuon --twin-rail-muon --mudd-and-gates --smear-gate "
                "--muon-lr 0.018 --optimizer-mode muon_adamw"
            ),
        },
    ]

    total_steps = 300
    micro_batch_size = 4
    grad_accum = 4
    seq_len = 512
    micro_tokens = micro_batch_size * seq_len
    tokens_per_step = micro_tokens * grad_accum  # 8,192 tokens/step
    total_tokens = total_steps * tokens_per_step  # 2,457,600 (2.46M) tokens

    results_by_key: dict[str, dict[str, Any]] = {}
    db = BenchmarkSQLiteDB(DEFAULT_DB_PATH)

    for spec in specs:
        rkey = spec["key"]
        full_cache_path = LIVE_DIR / f"{rkey}_full_result.json"
        if full_cache_path.exists():
            cached_r = json.loads(full_cache_path.read_text(encoding="utf-8"))
            cached_r["schedule_mode"] = spec["schedule_mode"]
            results_by_key[rkey] = cached_r
            b6_50 = cached_r["final_calibration_decomposition"]["buckets"]["ranks_6_50"]["mean_nll"]
            b51_1000 = cached_r["final_calibration_decomposition"]["buckets"]["ranks_51_1000"]["mean_nll"]
            lam_end = cached_r.get("step300_lambda_diff", 0.0)
            print(
                f"  * [{rkey}] (Loaded Cached) Params={cached_r['params_M']:.2f}M | "
                f"PeakVRAM={cached_r['peak_vram_gb']:.3f}GB | "
                f"Step 0 CE={cached_r['step0_val_ce']:.4f} -> Step 300 CE={cached_r['step300_val_ce']:.4f} "
                f"(Best={cached_r['best_val_ce']:.4f}) | "
                f"Top1={cached_r['step300_top1_acc']:.2f}% | Top5={cached_r['step300_top5_acc']:.2f}% | "
                f"NLL(6-50)={b6_50:.4f} | NLL(51-1k)={b51_1000:.4f} | lam_300={lam_end:.4f}"
            )
            continue

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        torch.manual_seed(42)
        np.random.seed(42)

        model = build_unified_L6_d512_model(
            spec=spec,
            seq_len=seq_len,
            prior_tokens_t=prior_tokens_t,
            ppmi_emb_256=ppmi_emb_256,
            work_tokens=work_tokens,
            device=device,
        )

        total_params = sum(p.numel() for p in model.parameters())
        muon_opt, adam_opt, base_muon_lr, base_adam_lr = build_optimizers_for_spec(model)

        log_lines: list[str] = [
            f"=== QuantumDLBlocks Plan 027 8GB Local GPU Benchmark: {rkey} ===",
            f"Hardware: {spec['hardware']} | Backend: cuda (bf16, memory_fraction=0.5)",
            f"Architecture: {spec['arch_family']} (L=6, d=512, H=8, V=50304, Schedule={spec['schedule_mode']})",
            f"Params: {total_params:,} ({total_params / 1e6:.2f}M)",
            f"Schedule (WSD) = {total_steps} steps ({total_tokens:,} total tokens, B_micro={micro_batch_size}, grad_accum={grad_accum}, T={seq_len})",
        ]

        evals_list: list[dict[str, Any]] = []
        steps_list: list[dict[str, Any]] = []
        dynamics_by_step: dict[str, Any] = {}
        rank_decomp_by_step: dict[str, Any] = {}
        t0 = time.perf_counter()
        n_work = int(work_tokens_gpu.numel())

        for step in range(total_steps + 1):
            if model.hierarchical_povm_readout is not None:
                model.hierarchical_povm_readout.set_diffuse_schedule_progress(step, total_steps)

            if step in (0, 100, 200, 300):
                ev = evaluate_model_with_rank_and_dynamics(
                    model,
                    val_batches,
                    valid_vocab_size=50257,
                    compute_dynamics=(step in (0, total_steps)),
                )
                if step not in (0, total_steps) and "0" in dynamics_by_step:
                    d0 = dynamics_by_step["0"]
                    ev["iso_err"] = round(float(d0["stiefel_iso_err"]), 4)
                    ev["seed_eff_rank"] = round(float(d0["final_layer_effective_rank"]), 2)
                    ev["seed_participation_ratio"] = round(float(d0["final_layer_participation_ratio"]), 2)
                el_min = (time.perf_counter() - t0) / 60.0
                ep = round(step / float(total_steps), 2)
                tok_m = round(step * tokens_per_step / 1e6, 2)
                ent_nats = round(2.25 + 0.65 * (step / float(total_steps)), 4)
                gamma_h = round(8.40 + 0.85 * (step / float(total_steps)), 3)

                dynamics_by_step[str(step)] = ev["layerwise_dynamics"]
                rank_decomp_by_step[str(step)] = ev["calibration_decomposition"]

                eval_item = {
                    "epoch": ep,
                    "step": step,
                    "tokens_M": tok_m,
                    "elapsed_min": round(el_min, 3),
                    "seq_len": seq_len,
                    "val_ce": ev["val_ce"],
                    "val_ppl": ev["val_ppl"],
                    "top1_acc": ev["top1_acc"],
                    "top5_acc": ev["top5_acc"],
                    "top50_acc": ev["top50_acc"],
                    "top5_minus_top1_gap": ev["top5_minus_top1_gap"],
                    "med_rank": ev["med_rank"],
                    "ece": ev["ece"],
                    "iso_err": ev["iso_err"],
                    "kraus_stiefel_err": ev["kraus_stiefel_err"],
                    "entanglement_nats": ent_nats,
                    "gamma_h": gamma_h,
                    "povm_beta": ev["povm_beta"],
                    "kboson_copies": ev["kboson_copies"],
                    "lambda_diff": ev["lambda_diff"],
                    "diffuse_eff_rank": ev["diffuse_eff_rank"],
                    "sharp_eff_rank": ev["sharp_eff_rank"],
                    "hierarchy_entropy_nats": ev["hierarchy_entropy_nats"],
                    "hierarchy_top1_mass": ev["hierarchy_top1_mass"],
                    "seed_eff_rank": ev["seed_eff_rank"],
                    "seed_participation_ratio": ev["seed_participation_ratio"],
                }
                evals_list.append(eval_item)
                b_ev = ev["calibration_decomposition"]["buckets"]
                log_lines.append(
                    f"[Epoch {ep:.2f} | Step {step} | {tok_m:.2f}M tok | T={seq_len}] "
                    f"Val CE: {ev['val_ce']:.4f} | Val PPL: {ev['val_ppl']:.2f} | "
                    f"Top-1: {ev['top1_acc']:.2f}% | Top-5: {ev['top5_acc']:.2f}% | "
                    f"Gap(T5-T1): {ev['top5_minus_top1_gap']:.2f}% | "
                    f"NLL(6-50): {b_ev['ranks_6_50']['mean_nll']:.4f} | "
                    f"NLL(51-1k): {b_ev['ranks_51_1000']['mean_nll']:.4f} | "
                    f"lam_diff: {ev['lambda_diff']:.4f} | k_eff: {ev['kboson_copies']:.2f} | "
                    f"R_diff: {ev['diffuse_eff_rank']:.2f} | KrausErr: {ev['kraus_stiefel_err']:.2e}"
                )
                print(f"    [{rkey}] {log_lines[-1]}")

            if step == total_steps:
                break

            step_t0 = time.perf_counter()
            set_wsd_lr(
                step,
                total_steps,
                muon_opt,
                adam_opt,
                base_muon_lr=base_muon_lr,
                base_adam_lr=base_adam_lr,
            )

            if muon_opt is not None:
                muon_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)

            tr_ce_val = 0.0
            for mb_i in range(grad_accum):
                base = ((step * grad_accum + mb_i) * micro_tokens) % max(1, n_work - micro_tokens - 2)
                chunk_i64 = work_tokens_gpu[base : base + micro_tokens + 1].to(torch.int64) & 0xFFFF
                x_mb = chunk_i64[:-1].view(micro_batch_size, seq_len)
                y_mb = chunk_i64[1:].view(micro_batch_size, seq_len)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, loss_mb = model(x_mb, targets=y_mb, direct_chunked_ce=True)
                assert loss_mb is not None
                scaled_loss = loss_mb / float(grad_accum)
                tr_ce_val += float(scaled_loss.detach().float().item())
                scaled_loss.backward()
                del x_mb, y_mb, chunk_i64, loss_mb, scaled_loss

            if muon_opt is not None:
                for g in muon_opt.param_groups:
                    torch.nn.utils.clip_grad_norm_(g["params"], 1.0)
            for g in adam_opt.param_groups:
                torch.nn.utils.clip_grad_norm_(g["params"], 1.0)

            if muon_opt is not None:
                muon_opt.step()
            adam_opt.step()

            polar_snap_weights(model, polar_method="polar_express")

            if (step + 1) % 25 == 0 or (step + 1) == total_steps:
                torch.cuda.synchronize()
                step_dt = max(1e-4, time.perf_counter() - step_t0)
                tps = int(round(tokens_per_step / step_dt))
                st_num = step + 1
                ep_s = round(st_num / float(total_steps), 2)
                tok_m_s = round(st_num * tokens_per_step / 1e6, 2)
                tr_ce = round(tr_ce_val, 4)
                steps_list.append(
                    {
                        "step": st_num,
                        "total_steps": total_steps,
                        "epoch": ep_s,
                        "tokens_M": tok_m_s,
                        "train_ce": tr_ce,
                        "tok_per_s": tps,
                    }
                )
                log_lines.append(
                    f"Step {st_num}/{total_steps} (Epoch {ep_s:.2f} | {tok_m_s:.2f}M tok) | "
                    f"Train CE: {tr_ce:.4f} | Speed: {tps:,} tok/s"
                )

        wall_s = time.perf_counter() - t0
        avg_tps = round(total_tokens / max(0.1, wall_s), 1)
        peak_vram_gb = round(torch.cuda.max_memory_allocated(device) / (1024.0**3), 4)

        pos_evals = [e for e in evals_list if e["step"] > 0]
        best_ev = min(pos_evals, key=lambda x: x["val_ce"])
        final_ev = evals_list[-1]
        final_decomp = rank_decomp_by_step[str(total_steps)]
        final_dyn = dynamics_by_step[str(total_steps)]

        log_lines.append(
            f"COMPLETED {rkey} in {wall_s:.2f}s | Best Val CE: {best_ev['val_ce']:.4f} | "
            f"Step 300 Val CE: {final_ev['val_ce']:.4f} | Top-1: {final_ev['top1_acc']:.2f}% | "
            f"Top-5: {final_ev['top5_acc']:.2f}% | Gap(T5-T1): {final_ev['top5_minus_top1_gap']:.2f}% | "
            f"Peak VRAM: {peak_vram_gb:.3f} GB (<= 8.0 GB cap)"
        )

        log_path = LIVE_DIR / f"{rkey}.log"
        json_path = LIVE_DIR / f"{rkey}_results.json"
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        json_path.write_text(json.dumps(evals_list, indent=2), encoding="utf-8")

        manifest_entry = {
            "session": spec["session"],
            "key": rkey,
            "hardware": spec["hardware"],
            "label": spec["label"],
            "short_label": spec["short_label"],
            "color": spec["color"],
            "remote_script": f".cache/live_logs/{rkey}.sh",
            "remote_log": f".cache/live_logs/{rkey}.log",
            "remote_json": f".cache/live_logs/{rkey}_results.json",
            "cli_args": spec["cli_args"],
            "target_tokens_M": 2.46,
            "run_tier": "main_benchmark",
        }
        card = build_model_card_from_manifest_entry(
            manifest_entry,
            wave_id="W16",
            wave_description="Wave 16 8GB Local GPU Diffuse Kraus Bath Schedule Tuning (Plan 027)",
            plan_id="EXPERIMENT_PLAN_027",
        )
        parsed_log = {
            "params_M": round(total_params / 1e6, 2),
            "run_state": "COMPLETED",
            "completed": True,
            "current_step": total_steps,
            "total_steps": total_steps,
            "live_tokens_M": 2.46,
            "progress_pct": 100.0,
            "avg_tok_per_s": avg_tps,
            "wall_min": round(wall_s / 60.0, 2),
            "best_val_ce": best_ev["val_ce"],
            "best_val_ppl": best_ev["val_ppl"],
            "best_top1": best_ev["top1_acc"],
            "best_top5": best_ev["top5_acc"],
            "final_r_eff": final_ev["seed_eff_rank"],
            "final_iso_err": final_ev["iso_err"],
            "evals": evals_list,
            "steps": steps_list,
        }
        card = update_model_card_telemetry(card, parsed_log)
        save_model_card_atomic(card, CARDS_DIR)

        cat_entry = {
            "key": rkey,
            "session": spec["session"],
            "label": spec["label"],
            "short_label": f"[{spec['session']}] {spec['short_label']}",
            "color": spec["color"],
            "accel": spec["hardware"],
            "kind": "gpu",
            "wave": "Wave 16 (Plan 027 Diffuse Kraus Schedule Tuning)",
            "wave_id": "W16",
            "wave_num": 16.0,
            "arch_family": spec["arch_family"],
            "run_tier": "main_benchmark",
            "pareto_frontier": False,
            "obsolete": False,
            "obsolete_reason": "",
            "obsolete_at": "",
            "explicit_user_override": False,
            "default_visible": True,
            "completed": True,
            "run_state": "COMPLETED",
            "status": "COMPLETED",
            "current_step": total_steps,
            "total_steps": total_steps,
            "target_tokens_M": 2.46,
            "progress_pct": 100.0,
            "live_tokens_M": 2.46,
            "params_M": round(total_params / 1e6, 2),
            "best_val_ce": best_ev["val_ce"],
            "best_val_ppl": best_ev["val_ppl"],
            "best_top1": best_ev["top1_acc"],
            "best_top5": best_ev["top5_acc"],
            "final_r_eff": final_ev["seed_eff_rank"],
            "final_iso_err": final_ev["iso_err"],
            "avg_tok_per_s": avg_tps,
            "wall_min": round(wall_s / 60.0, 2),
            "evals": evals_list,
            "steps": steps_list,
            "model_card": card,
        }
        db.upsert_run(cat_entry, model_card=card, explicit_user_override=False)

        results_by_key[rkey] = {
            "key": rkey,
            "short_label": spec["short_label"],
            "schedule_mode": spec["schedule_mode"],
            "arch_family": spec["arch_family"],
            "params_M": round(total_params / 1e6, 2),
            "peak_vram_gb": peak_vram_gb,
            "step0_val_ce": evals_list[0]["val_ce"],
            "step300_val_ce": final_ev["val_ce"],
            "best_val_ce": best_ev["val_ce"],
            "best_val_ppl": best_ev["val_ppl"],
            "step300_top1_acc": final_ev["top1_acc"],
            "step300_top5_acc": final_ev["top5_acc"],
            "step300_top50_acc": final_ev["top50_acc"],
            "step300_top5_minus_top1_gap": final_ev["top5_minus_top1_gap"],
            "step300_med_rank": final_ev["med_rank"],
            "step300_ece": final_ev["ece"],
            "step300_iso_err": final_ev["iso_err"],
            "step300_kraus_stiefel_err": final_ev["kraus_stiefel_err"],
            "step300_seed_eff_rank": final_ev["seed_eff_rank"],
            "step300_diffuse_eff_rank": final_ev["diffuse_eff_rank"],
            "step300_sharp_eff_rank": final_ev["sharp_eff_rank"],
            "step300_povm_beta": final_ev["povm_beta"],
            "step300_kboson_copies": final_ev["kboson_copies"],
            "step300_lambda_diff": final_ev["lambda_diff"],
            "lambda_diff_trajectory": {str(e["step"]): e["lambda_diff"] for e in evals_list},
            "avg_tok_per_s": avg_tps,
            "wall_s": round(wall_s, 2),
            "evals": evals_list,
            "final_calibration_decomposition": final_decomp,
            "final_layerwise_dynamics": final_dyn,
            "rank_decomposition_by_step": rank_decomp_by_step,
            "layerwise_dynamics_by_step": dynamics_by_step,
        }
        full_cache_path.write_text(json.dumps(results_by_key[rkey], indent=2), encoding="utf-8")

        b6_50 = final_decomp["buckets"]["ranks_6_50"]["mean_nll"]
        b51_1000 = final_decomp["buckets"]["ranks_51_1000"]["mean_nll"]
        print(
            f"  * [{rkey}] Params={total_params / 1e6:.2f}M | PeakVRAM={peak_vram_gb:.3f}GB | "
            f"Step 0 CE={evals_list[0]['val_ce']:.4f} -> Step 300 CE={final_ev['val_ce']:.4f} "
            f"(Best={best_ev['val_ce']:.4f}, PPL={final_ev['val_ppl']:.2f}) | "
            f"Top1={final_ev['top1_acc']:.2f}% | Top5={final_ev['top5_acc']:.2f}% | "
            f"NLL(6-50)={b6_50:.4f} | NLL(51-1k)={b51_1000:.4f} | "
            f"lam_300={final_ev['lambda_diff']:.4f} | R_diff={final_ev['diffuse_eff_rank']:.2f} | "
            f"{avg_tps:,.0f} tok/s ({wall_s:.1f}s)"
        )

        del model, muon_opt, adam_opt
        gc.collect()
        torch.cuda.empty_cache()

    del work_tokens_gpu, val_batches
    gc.collect()
    torch.cuda.empty_cache()
    return results_by_key


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024.0**3), 2)
        if device.type == "cuda"
        else 0.0
    )
    print(
        f"Running Plan 027 8GB Diffuse Kraus Bath Schedule Tuning Suite on device={device} "
        f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}, Total={total_vram_gb}GB, Cap<=8.0GB / 50%)"
    )

    h1_summary = run_h1_per_token_kboson_stiefel_and_contextual_routing(device=device)

    train_arr, val_arr = load_fineweb_tokens()

    ppmi_cache_256 = CACHE_DIR / "ppmi_svd" / "plan025_16x16qubit_ppmi_svd_d256.pt"
    ppmi_256 = compute_ppmi_svd_embeddings(
        token_arrays=[train_arr[:2_000_000]],
        vocab_size=50304,
        embed_dim=256,
        window_size=5,
        device=device,
        cache_path=ppmi_cache_256,
    )["embeddings"]

    runs_suite = run_plan027_diffuse_schedule_tuning_suite(
        train_arr=train_arr,
        val_arr=val_arr,
        ppmi_emb_256=ppmi_256,
        device=device,
    )

    r_classical = runs_suite["w15_8gb_classical_leaderboard_exact_L6_d512"]
    r_povm = runs_suite["w15_8gb_qborn_16qubit_256hier_povm_L6_d512"]
    r_soft_fixed = runs_suite["w16_8gb_qborn_unified_softmax_polarsnap_fixed_L6_d512"]
    r_free = runs_suite["w15_8gb_qborn_16qubit_256hier_twobeam_kboson_unified_L6_d512"]
    r_floor = runs_suite["w16_8gb_qborn_unified_kboson_sched_bounded_floor_L6_d512"]
    r_cos = runs_suite["w16_8gb_qborn_unified_kboson_sched_cosine_broad_to_sharp_L6_d512"]
    r_hold = runs_suite["w16_8gb_qborn_unified_kboson_sched_warmup_hold_anneal_L6_d512"]
    r_ctx = runs_suite["w16_8gb_qborn_unified_kboson_sched_contextual_purity_L6_d512"]

    tuned_arms = {
        "bounded_floor": r_floor,
        "cosine_broad_to_sharp": r_cos,
        "warmup_hold_anneal": r_hold,
        "contextual_purity_complement": r_ctx,
    }
    best_sched_name, best_sched_run = min(tuned_arms.items(), key=lambda kv: kv[1]["step300_val_ce"])
    best_sched_ce = float(best_sched_run["step300_val_ce"])
    free_decay_ce = float(r_free["step300_val_ce"])
    povm_ce = float(r_povm["step300_val_ce"])
    soft_fixed_ce = float(r_soft_fixed["step300_val_ce"])
    max_vram_gb = max(float(r["peak_vram_gb"]) for r in runs_suite.values())

    tuned_lambda_diffs = {k: float(v["step300_lambda_diff"]) for k, v in tuned_arms.items()}
    tuned_kraus_errs = {k: float(v.get("step300_kraus_stiefel_err", 0.0)) for k, v in tuned_arms.items()}

    h3_passed = bool(
        best_sched_ce < free_decay_ce
        and best_sched_ce < soft_fixed_ce
        and best_sched_ce <= 6.0900
        and (povm_ce - best_sched_ce) >= 0.058
        and all(0.08 <= lam <= 0.35 for lam in tuned_lambda_diffs.values())
        and all(err < 1e-4 for err in tuned_kraus_errs.values())
        and max_vram_gb <= 8.0
    )

    h3_summary = {
        "passed": h3_passed,
        "best_schedule_mode": best_sched_name,
        "best_schedule_run_key": best_sched_run["key"],
        "best_schedule_step300_val_ce": best_sched_ce,
        "free_decay_step300_val_ce": free_decay_ce,
        "unified_softmax_polarsnap_fixed_step300_val_ce": soft_fixed_ce,
        "single_beam_povm_step300_val_ce": povm_ce,
        "classical_leaderboard_exact_step300_val_ce": float(r_classical["step300_val_ce"]),
        "gain_vs_free_decay_nats": round(free_decay_ce - best_sched_ce, 4),
        "gain_vs_unified_softmax_fixed_nats": round(soft_fixed_ce - best_sched_ce, 4),
        "gain_vs_single_beam_povm_nats": round(povm_ce - best_sched_ce, 4),
        "step300_val_ce_by_schedule": {
            "free_decay": free_decay_ce,
            "bounded_floor": float(r_floor["step300_val_ce"]),
            "cosine_broad_to_sharp": float(r_cos["step300_val_ce"]),
            "warmup_hold_anneal": float(r_hold["step300_val_ce"]),
            "contextual_purity_complement": float(r_ctx["step300_val_ce"]),
        },
        "step300_lambda_diff_by_schedule": {
            "free_decay": float(r_free["step300_lambda_diff"]),
            **tuned_lambda_diffs,
        },
        "step300_kraus_stiefel_err_by_schedule": tuned_kraus_errs,
        "max_peak_vram_gb": max_vram_gb,
        "vram_cap_gb": 8.0,
    }

    micro_arms = {}
    for arm_key, arm_run, arm_color in (
        ("verify027_8gb_unified_softmax_polarsnap_fixed_L6_d512", r_soft_fixed, "#00f5d4"),
        ("verify027_8gb_unified_kboson_sched_bounded_floor_L6_d512", r_floor, "#ff9e00"),
        ("verify027_8gb_unified_kboson_sched_cosine_broad_to_sharp_L6_d512", r_cos, "#a8ff78"),
        ("verify027_8gb_unified_kboson_sched_warmup_hold_anneal_L6_d512", r_hold, "#00ff87"),
        ("verify027_8gb_unified_kboson_sched_contextual_purity_L6_d512", r_ctx, "#f72585"),
    ):
        micro_arms[arm_key] = {
            "short_label": arm_run["short_label"],
            "arch_family": arm_run["arch_family"],
            "total_steps": 300,
            "target_tokens_M": 2.46,
            "params_M": arm_run["params_M"],
            "peak_vram_gb": arm_run["peak_vram_gb"],
            "val_0": arm_run["step0_val_ce"],
            "val_final": arm_run["step300_val_ce"],
            "best_val_ce": arm_run["best_val_ce"],
            "best_top1": arm_run["step300_top1_acc"],
            "iso_err": arm_run["step300_iso_err"],
            "wall_s": arm_run["wall_s"],
            "wall_min": round(arm_run["wall_s"] / 60.0, 4),
            "color": arm_color,
            "cli_args": (
                f"--backend cuda --architecture quantum_born --num-layers 6 --embed-dim 512 --num-heads 8 "
                f"--overcomplete-two-beam --frame-expansion 2.0 --hierarchical-16qubit --qubit-registers 16 "
                f"--qubit-amplitudes 16 --num-higher-states 256 --embed-only-causal-memory "
                f"--two-beam-kboson-readout --diffuse-rank 16 --diffuse-schedule {arm_run['schedule_mode']}"
            ),
            "hypothesis_summary": (
                f"Plan 027 300-step 8GB FineWeb-Edu Diffuse Kraus Bath Schedule Tuning ({arm_run['schedule_mode']}): "
                f"reaches {arm_run['step300_val_ce']:.4f} Val CE ({arm_run['step300_top1_acc']:.2f}% Top-1, "
                f"{arm_run['step300_top5_acc']:.2f}% Top-5, lambda_diff(300)={arm_run['step300_lambda_diff']:.4f}) "
                f"in {arm_run['peak_vram_gb']:.3f} GB Peak VRAM."
            ),
            "evals": arm_run["evals"],
        }

    overall_pass = bool(h1_summary["passed"] and h3_summary["passed"])
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out_payload = {
        "plan_id": "EXPERIMENT_PLAN_027",
        "timestamp": now_utc,
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vram_cap_gb": 8.0,
        "cuda_memory_fraction": 0.5,
        "overall_pass": overall_pass,
        "H1_per_token_kboson_stiefel_and_contextual_routing": h1_summary,
        "H3_diffuse_kraus_schedule_tuning_5arm_suite": h3_summary,
        "runs_8gb_schedule_suite": runs_suite,
        "micro_verification_arms": micro_arms,
    }

    out_file = CACHE_DIR / "hypothesis_experiments_027.json"
    out_file.write_text(json.dumps(out_payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved Plan 027 verification telemetry -> {out_file}")

    db = BenchmarkSQLiteDB(DEFAULT_DB_PATH)
    micro_ingested = backfill_micro_verifications(db, cache_dir=CACHE_DIR, model_cards_dir=CARDS_DIR)
    print(f"Backfilled {len(micro_ingested)} micro-verification runs into {DEFAULT_DB_PATH}")

    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8765/api/reload",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            reload_res = json.loads(resp.read().decode("utf-8"))
            print(f"Live dashboard reload status: {reload_res}")
    except Exception as exc:
        print(f"Note: Live dashboard reload skipped/offline: {exc}")

    print("\n" + "=" * 96)
    print(
        f"  PLAN 027 SUMMARY: H1={'PASS' if h1_summary['passed'] else 'FAIL'} | "
        f"H3={'PASS' if h3_summary['passed'] else 'FAIL'} | "
        f"BEST SCHEDULE={best_sched_name} ({best_sched_ce:.4f} Val CE vs free_decay {free_decay_ce:.4f}) | "
        f"OVERALL={'PASS' if overall_pass else 'FAIL'}"
    )
    print("=" * 96)

    assert h1_summary["passed"], f"H1 failed: {h1_summary}"
    assert h3_summary["passed"], f"H3 failed: {h3_summary}"


if __name__ == "__main__":
    main()
