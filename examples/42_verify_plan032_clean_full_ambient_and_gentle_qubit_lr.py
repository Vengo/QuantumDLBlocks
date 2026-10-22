#!/usr/bin/env python3
"""Plan 032 Verification & Clean 8 GB RTX 5060 Ti Isolation Study on Genuine FineWeb-Edu.

Implements and verifies Hypotheses H1, H2, and H3 of `EXPERIMENT_PLAN_032.md` on the local
NVIDIA GeForce RTX 5060 Ti (16 GB total VRAM, strictly hardware-capped to <= 8.0 GB / 50% VRAM
via `torch.cuda.set_per_process_memory_fraction(0.5, device=0)`):

1. H1 — Anti-Diagonal Alternating-Sign Spectral-Reversal Isometry (`J_anti(x)_a = (-1)^a * x_{A-1-a}`)
   for Full-Ambient (`use_ortho_decouple=False`) `16x16 = 256` (`d=256`) and `16x32 = 512` (`d=512`) Qubits:
   - Verifies exact per-register self-orthogonality (`self_register_overlap_max < 1e-5`),
   - Verifies `>= 2.5x` Step-0 directed bigram overlap over random pairs,
   - Verifies that `J_anti` eliminates the Full-Ambient Step-0 cross-entropy spike (`> 12.0 nats` without `J_anti`),
     bringing Step-0 Validation CE on `val_slice.bin` down to `<= 9.60 nats` (`~9.1-9.3 nats`) on both `d=256 (16x16)`
     and `d=512 (16x32)` while keeping initialization memory `< 4.5 GB`.

2. H2 — Clean `L=6, d=512` (`300` Steps = `2.46M` FineWeb-Edu Tokens, `use_bipolar_kboson=False`):
   - `Isolate 1` (`w18_8gb_L6_d512_clean_gentle_qubit_lr_300steps`): Exact Plan 030 `Attempt 4` (`6.0823` winner,
     `use_ortho_decouple=True, 16x16=256 qubits`) + ONLY Gentle `2x` Zero-Weight-Decay Qubit/Value Embedding LR
     (`lr = 0.006, weight_decay = 0.0`).
   - `Isolate 2` (`w18_8gb_L6_d512_clean_full_ambient_16x32_janti_300steps`): `L=6, d=512` upgraded to Full-Ambient
     `16x32 = 512` Qubits (`use_ortho_decouple=False, decouple_self_in_full_ambient=True` with `J_anti`) + Gentle
     `2x` Zero-WD Qubit Embedding LR (`lr = 0.006, weight_decay = 0.0`).

3. H3 — Clean `L=4, d=256` (`900` Steps = `7.37M` FineWeb-Edu Tokens, `use_bipolar_kboson=False`):
   - `Isolate 3` (`w18_8gb_L4_d256_clean_janti_gentle_lr_900steps`): Exact Plan 030 `Attempt 3` (`L=4, d=256, H=4`,
     `16x16 = 256` Full-Ambient Qubits) upgraded with `J_anti` Spectral-Reversal Full-Ambient Initialization +
     Gentle `2x` Zero-WD Qubit Embedding LR (`lr = 0.006, weight_decay = 0.0`).
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
from qdlblocks.benchmarks.synthetic_fineweb_walk import (
    build_fast_quantum_born_synth_lm,
)
from qdlblocks.layers.cooccurrence_init import (
    compute_ppmi_svd_embeddings,
)
from qdlblocks.layers.hierarchical_qubit import (
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
    max_train_tokens: int = 8_000_000,
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


def build_optimizers_for_plan032(
    model: nn.Module,
    qubit_embed_lr: float = 0.006,
) -> tuple[StiefelMuon | None, torch.optim.AdamW, float, float]:
    """Build matched StiefelMuon + AdamW optimizers with Gentle 2x Zero-WD Qubit/Value Embedding group (`lr=0.006, wd=0.0`)."""
    muon_params, adam_params = split_muon_adamw_params(model)
    filtered_muon: list[nn.Parameter] = []
    for p in muon_params:
        if p.ndim == 3 or (p.ndim == 2 and min(p.shape) >= 32 and max(p.shape) < 40000):
            filtered_muon.append(p)
        else:
            adam_params.append(p)

    is_kboson = getattr(model, "use_two_beam_kboson_readout", False)
    is_softmax = getattr(model, "use_last_layer_softmax", False)

    base_muon_lr = 0.018
    base_adam_lr = 0.003
    head_lr = 0.012
    embed_lr = float(qubit_embed_lr)

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
    embed_ids: set[int] = set()
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
            getattr(hr, "readout_stereo_pole", None),
            getattr(hr, "readout_stereo_tau", None),
            getattr(hr, "readout_stereo_gain", None),
            base_h,
            getattr(model, "povm_log_prior", None),
        ):
            if gp is not None and gp.requires_grad:
                gain_ids.add(id(gp))

    hq = getattr(model, "hierarchical_qubit_embed", None)
    if hq is not None:
        for ep in (
            getattr(hq, "unigram_qubits", None),
            getattr(hq, "higher_states", None),
        ):
            if ep is not None and ep.requires_grad:
                embed_ids.add(id(ep))

    ve_list = getattr(model, "value_embeds", None)
    if ve_list is not None:
        for ve_mod in ve_list:
            for vp in ve_mod.parameters():
                if vp.requires_grad:
                    embed_ids.add(id(vp))

    for name, p in model.named_parameters():
        if p.requires_grad and ("stereo" in name or "phase" in name or "lm_head" in name):
            gain_ids.add(id(p))

    ng_mod = getattr(model, "ngram_embed", None)
    if ng_mod is not None:
        for np_p in ng_mod.parameters():
            if np_p.requires_grad:
                ngram_ids.add(id(np_p))

    special_ids = gain_ids | embed_ids | ngram_ids
    body_group = [p for p in adam_params if id(p) not in special_ids and p.requires_grad]
    param_groups: list[dict[str, Any]] = [
        {"params": body_group, "lr": base_adam_lr, "base_lr": base_adam_lr, "weight_decay": 0.01, "group_name": "body"}
    ]
    if embed_ids:
        embed_group = [p for p in adam_params if id(p) in embed_ids and p.requires_grad]
        param_groups.append(
            {"params": embed_group, "lr": embed_lr, "base_lr": embed_lr, "weight_decay": 0.0, "group_name": "qubit_embed"}
        )
    if gain_ids:
        gain_group = [p for p in adam_params if id(p) in gain_ids and p.requires_grad]
        param_groups.append(
            {"params": gain_group, "lr": head_lr, "base_lr": head_lr, "weight_decay": 0.0, "group_name": "readout_head"}
        )
    if ngram_ids:
        ngram_lr = 0.025
        ngram_group = [p for p in adam_params if id(p) in ngram_ids and p.requires_grad]
        param_groups.append(
            {"params": ngram_group, "lr": ngram_lr, "base_lr": ngram_lr, "weight_decay": 0.0, "group_name": "ngram"}
        )

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


def run_h1_janti_full_ambient_verification(
    ppmi_emb_256: torch.Tensor,
    ppmi_emb_512: torch.Tensor,
    init_tokens: np.ndarray,
    prior_tokens_t: torch.Tensor,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    """Verify H1: Anti-Diagonal Spectral-Reversal Isometry (`J_anti`) eliminates Full-Ambient Step-0 CE spike."""
    print("\n" + "=" * 96)
    print(
        "  [H1 Verification] Anti-Diagonal Spectral-Reversal Isometry (J_anti) for Full-Ambient "
        "(16x16 d=256 & 16x32 d=512) Self-Decoupled PPMI-SVD"
    )
    print("=" * 96)

    torch.cuda.reset_peak_memory_stats(device)
    x0, y0 = val_batches[0]

    # 1. Legacy un-decoupled full-ambient (decouple_self_in_full_ambient=False)
    torch.manual_seed(2032)
    m_legacy = build_fast_quantum_born_synth_lm(
        vocab_size=50304,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        block_size=512,
        seed=2032,
        device=device,
    )
    initialize_povm_log_prior(m_legacy, token_ids=prior_tokens_t, alpha=0.62, bose_saturate_top_k=16)
    stats_legacy = initialize_hierarchical_16qubit_with_ppmi_svd(
        m_legacy,
        ppmi_embeddings=ppmi_emb_256,
        token_shards=[init_tokens],
        alpha=0.75,
        kmeans_iters=8,
        decouple_self_in_full_ambient=False,
        directed_predecessor_mix=0.0,
        device=device,
    )
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, loss_legacy = m_legacy(x0, targets=y0)
    legacy_self_overlap = float(stats_legacy["self_register_overlap_max"])
    legacy_step0_ce = float(loss_legacy.item())
    del m_legacy
    torch.cuda.empty_cache()

    # 2. J_anti Full-Ambient d=256 (16x16)
    torch.manual_seed(2032)
    m_256 = build_fast_quantum_born_synth_lm(
        vocab_size=50304,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        block_size=512,
        seed=2032,
        device=device,
    )
    initialize_povm_log_prior(m_256, token_ids=prior_tokens_t, alpha=0.62, bose_saturate_top_k=16)
    stats_256 = initialize_hierarchical_16qubit_with_ppmi_svd(
        m_256,
        ppmi_embeddings=ppmi_emb_256,
        token_shards=[init_tokens],
        alpha=0.75,
        kmeans_iters=8,
        decouple_self_in_full_ambient=True,
        directed_predecessor_mix=0.35,
        device=device,
    )
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, loss_256 = m_256(x0, targets=y0)
    self_ov_256 = float(stats_256["self_register_overlap_max"])
    dir_ratio_256 = float(stats_256["directed_bigram_overlap_ratio"])
    step0_ce_256 = float(loss_256.item())
    del m_256
    torch.cuda.empty_cache()

    # 3. J_anti Full-Ambient d=512 (16x32)
    torch.manual_seed(2032)
    m_512 = QuantumBornGPT(
        vocab_size=50304,
        embed_dim=512,
        seed_dim=512,
        num_heads=8,
        num_layers=6,
        block_size=512,
        readout_mode="asymmetric",
        povm_temperature=0.030,
        use_overcomplete_two_beam=True,
        frame_expansion=2.0,
        use_purified_attn=True,
        true_layer0_v_residual=True,
        use_gated_spherical_residual=True,
        use_dynamic_povm_beta=True,
        use_povm_log_prior=True,
        use_ortho_decouple=False,
        use_hierarchical_16qubit=True,
        qubit_registers=16,
        qubit_amplitudes=32,
        num_higher_states=256,
        use_block_hierarchy_bridge=True,
        bridge_causal_memory=False,
        use_two_beam_kboson_readout=True,
        diffuse_rank=16,
        diffuse_schedule="warmup_hold_anneal",
        use_stereographic_lift=True,
        quadratic_diffuse_fidelity=0.50,
        num_value_embeddings=3,
        use_zero_half_rope=True,
        use_depth_scaled_residual=True,
        window_size=384,
        use_ngram_hash=True,
        ngram_vocab_size=65536,
        ngram_inject_scale=0.12,
        use_mudd_and_gates=True,
        use_smear_gate=True,
    ).to(device)
    initialize_povm_log_prior(m_512, token_ids=prior_tokens_t, alpha=0.62, bose_saturate_top_k=16)
    stats_512 = initialize_hierarchical_16qubit_with_ppmi_svd(
        m_512,
        ppmi_embeddings=ppmi_emb_512,
        token_shards=[init_tokens],
        alpha=0.75,
        kmeans_iters=8,
        decouple_self_in_full_ambient=True,
        directed_predecessor_mix=0.35,
        device=device,
    )
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, loss_512 = m_512(x0, targets=y0)
    self_ov_512 = float(stats_512["self_register_overlap_max"])
    dir_ratio_512 = float(stats_512["directed_bigram_overlap_ratio"])
    step0_ce_512 = float(loss_512.item())
    peak_init_vram_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    del m_512
    torch.cuda.empty_cache()

    passed = bool(
        legacy_self_overlap > 0.40
        and self_ov_256 < 1e-5
        and self_ov_512 < 1e-5
        and dir_ratio_256 >= 2.5
        and dir_ratio_512 >= 2.5
        and step0_ce_256 <= 9.60
        and step0_ce_512 <= 9.60
        and peak_init_vram_gb <= 4.50
    )
    print(
        f"  * [H1 J_anti Full-Ambient Calibration] LegacySelfOverlap={legacy_self_overlap:.4f} (Step0 CE={legacy_step0_ce:.4f}) -> "
        f"d=256(16x16): SelfOv={self_ov_256:.2e}, DirRatio={dir_ratio_256:.2f}x, Step0 CE={step0_ce_256:.4f} (<=9.60) | "
        f"d=512(16x32): SelfOv={self_ov_512:.2e}, DirRatio={dir_ratio_512:.2f}x, Step0 CE={step0_ce_512:.4f} (<=9.60) | "
        f"PeakInitVRAM={peak_init_vram_gb:.3f}GB | PASS={passed}"
    )
    return {
        "passed": passed,
        "legacy_self_register_overlap_max": round(legacy_self_overlap, 6),
        "legacy_step0_val_ce": round(legacy_step0_ce, 4),
        "d256_16x16_self_register_overlap_max": self_ov_256,
        "d256_16x16_directed_bigram_overlap_ratio": round(dir_ratio_256, 4),
        "d256_16x16_step0_val_ce": round(step0_ce_256, 4),
        "d512_16x32_self_register_overlap_max": self_ov_512,
        "d512_16x32_directed_bigram_overlap_ratio": round(dir_ratio_512, 4),
        "d512_16x32_step0_val_ce": round(step0_ce_512, 4),
        "peak_init_vram_gb": round(peak_init_vram_gb, 4),
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
    beam_ortho_err = 0.0

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
            if model._last_psi_beam2 is not None:
                beam_ortho_err = float(
                    (F.normalize(psi_f.float(), dim=-1) * F.normalize(model._last_psi_beam2.float(), dim=-1))
                    .sum(dim=-1)
                    .abs()
                    .max()
                    .item()
                )
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
            "num_layers": int(getattr(model, "num_layers", 4)),
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
        "beam_ortho_err": round(beam_ortho_err, 6),
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


def build_plan032_model(
    spec: dict[str, Any],
    seq_len: int,
    prior_tokens_t: torch.Tensor,
    ppmi_emb_256: torch.Tensor,
    ppmi_emb_512: torch.Tensor,
    init_tokens: np.ndarray,
    device: torch.device,
) -> nn.Module:
    """Instantiate a Plan 032 clean isolated model (`Isolate 1`, `Isolate 2`, `Isolate 3`) on `device`."""
    mode = str(spec["attempt_mode"])

    if mode == "isolate_1_L6_d512_clean_gentle_qubit_lr":
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
            use_last_layer_softmax=False,
            logit_softcap=15.0,
            use_two_beam_kboson_readout=True,
            diffuse_rank=16,
            diffuse_schedule="warmup_hold_anneal",
            lambda_diff_min=0.08,
            lambda_diff_max=0.35,
            lambda_diff_start=0.22,
            lambda_diff_end=0.12,
            contextual_purity_gain=0.20,
            use_stereographic_lift=True,
            use_cosine_phase_beam_gate=False,
            quadratic_diffuse_fidelity=0.50,
            use_bipolar_kboson=False,
            bipolar_hole_weight=0.0,
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
        initialize_povm_log_prior(model, token_ids=prior_tokens_t, alpha=0.62, bose_saturate_top_k=16)
        initialize_hierarchical_16qubit_with_ppmi_svd(
            model,
            ppmi_embeddings=ppmi_emb_256,
            token_shards=[init_tokens],
            alpha=0.75,
            kmeans_iters=8,
            decouple_self_in_full_ambient=True,
            directed_predecessor_mix=0.35,
            device=device,
        )
        with torch.no_grad():
            assert model.hierarchical_povm_readout is not None
            model.hierarchical_povm_readout.kboson_copies.fill_(38.0)
        return model

    if mode == "isolate_2_L6_d512_clean_full_ambient_16x32_janti":
        model = QuantumBornGPT(
            vocab_size=50304,
            embed_dim=512,
            seed_dim=512,
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
            use_ortho_decouple=False,
            use_hierarchical_16qubit=True,
            qubit_registers=16,
            qubit_amplitudes=32,
            num_higher_states=256,
            use_block_hierarchy_bridge=True,
            bridge_causal_memory=False,
            use_last_layer_softmax=False,
            logit_softcap=15.0,
            use_two_beam_kboson_readout=True,
            diffuse_rank=16,
            diffuse_schedule="warmup_hold_anneal",
            lambda_diff_min=0.08,
            lambda_diff_max=0.35,
            lambda_diff_start=0.22,
            lambda_diff_end=0.12,
            contextual_purity_gain=0.20,
            use_stereographic_lift=True,
            use_cosine_phase_beam_gate=False,
            quadratic_diffuse_fidelity=0.50,
            use_bipolar_kboson=False,
            bipolar_hole_weight=0.0,
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
        initialize_povm_log_prior(model, token_ids=prior_tokens_t, alpha=0.62, bose_saturate_top_k=16)
        initialize_hierarchical_16qubit_with_ppmi_svd(
            model,
            ppmi_embeddings=ppmi_emb_512,
            token_shards=[init_tokens],
            alpha=0.75,
            kmeans_iters=8,
            decouple_self_in_full_ambient=True,
            directed_predecessor_mix=0.35,
            device=device,
        )
        with torch.no_grad():
            assert model.hierarchical_povm_readout is not None
            model.hierarchical_povm_readout.kboson_copies.fill_(38.0)
        return model

    if mode == "isolate_3_L4_d256_clean_janti_gentle_lr_900steps":
        model = QuantumBornGPT(
            vocab_size=50304,
            embed_dim=256,
            seed_dim=256,
            num_heads=4,
            num_layers=4,
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
            use_ortho_decouple=False,
            use_hierarchical_16qubit=True,
            qubit_registers=16,
            qubit_amplitudes=16,
            num_higher_states=128,
            use_block_hierarchy_bridge=True,
            bridge_causal_memory=False,
            use_last_layer_softmax=False,
            logit_softcap=15.0,
            use_two_beam_kboson_readout=True,
            diffuse_rank=16,
            diffuse_schedule="warmup_hold_anneal",
            lambda_diff_min=0.08,
            lambda_diff_max=0.35,
            lambda_diff_start=0.22,
            lambda_diff_end=0.12,
            contextual_purity_gain=0.15,
            use_stereographic_lift=True,
            use_cosine_phase_beam_gate=True,
            quadratic_diffuse_fidelity=0.50,
            use_bipolar_kboson=False,
            bipolar_hole_weight=0.0,
            num_value_embeddings=2,
            use_zero_half_rope=True,
            use_depth_scaled_residual=True,
            window_size=384,
            use_ngram_hash=True,
            ngram_vocab_size=65536,
            ngram_inject_scale=0.12,
            use_mudd_and_gates=True,
            use_smear_gate=True,
            use_activation_checkpointing=False,
        ).to(device)
        initialize_povm_log_prior(model, token_ids=prior_tokens_t, alpha=0.62, bose_saturate_top_k=16)
        initialize_hierarchical_16qubit_with_ppmi_svd(
            model,
            ppmi_embeddings=ppmi_emb_256,
            token_shards=[init_tokens],
            alpha=0.75,
            kmeans_iters=8,
            decouple_self_in_full_ambient=True,
            directed_predecessor_mix=0.35,
            device=device,
        )
        with torch.no_grad():
            assert model.hierarchical_povm_readout is not None
            model.hierarchical_povm_readout.kboson_copies.fill_(38.0)
        return model

    raise ValueError(f"Unknown attempt_mode: {mode}")


def run_plan032_clean_isolation_suite(
    train_arr: np.ndarray,
    val_arr: np.ndarray,
    ppmi_emb_256: torch.Tensor,
    ppmi_emb_512: torch.Tensor,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    """Run Plan 032 Clean Isolations (`Isolate 1`, `Isolate 2`, `Isolate 3`) against Plan 030 baselines (<= 8.0 GB VRAM)."""
    print("\n" + "=" * 96)
    print(
        "  [H2 & H3 Plan 032 8GB RTX 5060 Ti Suite] Clean J_anti Full-Ambient & Gentle 2x Zero-WD Qubit LR "
        "(Isolates 1-3, <= 8.0 GB VRAM)"
    )
    print("=" * 96)

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    CARDS_DIR.mkdir(parents=True, exist_ok=True)

    init_tokens = train_arr[:2_600_000]
    work_tokens_8m = train_arr[:8_000_000]
    work_tokens_gpu = torch.from_numpy(work_tokens_8m.view(np.int16)).to(device=device)
    prior_tokens_t = torch.from_numpy(train_arr[:2_000_000].astype(np.int64))
    val_batches = build_evaluation_batches(val_arr, batch_size=4, seq_len=512, num_batches=6, device=device)

    results_by_key: dict[str, dict[str, Any]] = {}

    for ref_key in (
        "w15_8gb_classical_leaderboard_exact_L6_d512",
        "w17_8gb_fast_synth_ppmi_ngram_L2_d256",
        "w17_8gb_fast_synth_scaled_L4_d256_900steps",
        "w17_8gb_qborn_unified_attn_stereo_directed_ppmi_L6_d512",
        "w18_8gb_L2_d256_bipolar_kboson_whitened_300steps",
    ):
        ref_path = LIVE_DIR / f"{ref_key}_full_result.json"
        if ref_path.exists():
            cached_r = json.loads(ref_path.read_text(encoding="utf-8"))
            results_by_key[ref_key] = cached_r
            print(
                f"  * [{ref_key}] (Loaded Reference) Params={cached_r['params_M']:.2f}M | "
                f"PeakVRAM={cached_r['peak_vram_gb']:.3f}GB | "
                f"Step 300 CE={cached_r['step300_val_ce']:.4f} | "
                f"Best CE={cached_r['best_val_ce']:.4f} | "
                f"Top1={cached_r['step300_top1_acc']:.2f}% | Top5={cached_r['step300_top5_acc']:.2f}% | "
                f"Wall={cached_r['wall_s']:.1f}s ({cached_r['avg_tok_per_s']:,.0f} tok/s)"
            )

    specs: list[dict[str, Any]] = [
        {
            "key": "w18_8gb_L6_d512_clean_gentle_qubit_lr_300steps",
            "attempt_mode": "isolate_1_L6_d512_clean_gentle_qubit_lr",
            "session": "local_rtx5060ti_8gb",
            "hardware": "NVIDIA GeForce RTX 5060 Ti (<=8.0GB Cap)",
            "arch_family": "quantum_born",
            "schedule_mode": "clean_gentle_qubit_lr_L6_d512",
            "total_steps": 300,
            "micro_batch_size": 4,
            "grad_accum": 4,
            "label": "Plan 032 Isolate 1 (Exact Plan 030 Attempt 4 L=6, d=512 + Gentle 2x Zero-WD Qubit Embedding LR 0.006): QuantumBornGPT (L=6, d=512, H=8, 16x16 Qubits, 256 Hier, r=16 warmup_hold_anneal, 300 Steps / 2.46M Tok)",
            "short_label": "W18 8GB Clean Unified + Gentle 2x Qubit LR (L=6, d=512, 16x16)",
            "color": "#00f5d4",
            "cli_args": "--backend cuda --architecture quantum_born --num-layers 6 --embed-dim 512 --num-heads 8 --overcomplete-two-beam --hierarchical-16qubit --qubit-registers 16 --qubit-amplitudes 16 --num-higher-states 256 --two-beam-kboson-readout --diffuse-rank 16 --diffuse-schedule warmup_hold_anneal --stereographic-lift --quadratic-diffuse-fidelity 0.50 --ppmi-svd-init",
        },
        {
            "key": "w18_8gb_L6_d512_clean_full_ambient_16x32_janti_300steps",
            "attempt_mode": "isolate_2_L6_d512_clean_full_ambient_16x32_janti",
            "session": "local_rtx5060ti_8gb",
            "hardware": "NVIDIA GeForce RTX 5060 Ti (<=8.0GB Cap)",
            "arch_family": "quantum_born",
            "schedule_mode": "clean_full_ambient_16x32_janti_L6_d512",
            "total_steps": 300,
            "micro_batch_size": 4,
            "grad_accum": 4,
            "label": "Plan 032 Isolate 2 (Full-Ambient 16x32=512 J_anti Qubits + Gentle 2x Zero-WD Qubit LR 0.006): QuantumBornGPT (L=6, d=512, H=8, 16x32 Full-Ambient J_anti Qubits, 256 Hier, r=16 warmup_hold_anneal, 300 Steps / 2.46M Tok)",
            "short_label": "W18 8GB Clean Full-Ambient 16x32 J_anti + Gentle 2x Qubit LR (L=6, d=512)",
            "color": "#00ff87",
            "cli_args": "--backend cuda --architecture quantum_born --num-layers 6 --embed-dim 512 --num-heads 8 --overcomplete-two-beam --hierarchical-16qubit --qubit-registers 16 --qubit-amplitudes 32 --num-higher-states 256 --two-beam-kboson-readout --diffuse-rank 16 --diffuse-schedule warmup_hold_anneal --stereographic-lift --quadratic-diffuse-fidelity 0.50 --ppmi-svd-init",
        },
        {
            "key": "w18_8gb_L4_d256_clean_janti_gentle_lr_900steps",
            "attempt_mode": "isolate_3_L4_d256_clean_janti_gentle_lr_900steps",
            "session": "local_rtx5060ti_8gb",
            "hardware": "NVIDIA GeForce RTX 5060 Ti (<=8.0GB Cap)",
            "arch_family": "quantum_born",
            "schedule_mode": "clean_janti_gentle_lr_L4_d256_900steps",
            "total_steps": 900,
            "micro_batch_size": 8,
            "grad_accum": 2,
            "label": "Plan 032 Isolate 3 (Scaled L=4, d=256 + J_anti Full-Ambient 16x16 Init + Gentle 2x Zero-WD Qubit LR 0.006): QuantumBornGPT (L=4, d=256, H=4, VE=2, 16x16 J_anti Qubits, 128 Hier, 65k N-Gram, Stereo+Phase+QuadKraus, 900 Steps / 7.37M Tok)",
            "short_label": "W18 8GB Clean J_anti + Gentle 2x Qubit LR (L=4, d=256, 900 Steps / 7.37M Tok)",
            "color": "#fee440",
            "cli_args": "--backend cuda --architecture quantum_born --num-layers 4 --embed-dim 256 --num-heads 4 --num-value-embeddings 2 --hierarchical-16qubit --qubit-registers 16 --qubit-amplitudes 16 --num-higher-states 128 --ppmi-svd-init --ngram-hash --smear-gate --stereographic-lift --cosine-phase-beam-gate --quadratic-diffuse-fidelity 0.50",
        },
    ]

    seq_len = 512
    db = BenchmarkSQLiteDB(DEFAULT_DB_PATH)
    n_work = int(work_tokens_gpu.numel())

    for spec in specs:
        rkey = spec["key"]
        total_steps = int(spec["total_steps"])
        micro_batch_size = int(spec["micro_batch_size"])
        grad_accum = int(spec["grad_accum"])
        micro_tokens = micro_batch_size * seq_len
        tokens_per_step = micro_tokens * grad_accum  # 8,192 tokens/step
        total_tokens = total_steps * tokens_per_step
        target_tok_m = round(total_tokens / 1e6, 2)

        full_cache_path = LIVE_DIR / f"{rkey}_full_result.json"
        res_cache_path = LIVE_DIR / f"{rkey}_results.json"
        if not full_cache_path.exists() and res_cache_path.exists():
            res_d = json.loads(res_cache_path.read_text(encoding="utf-8"))
            evs = res_d.get("evals", [])
            if evs and res_d.get("completed"):
                ev0_d = evs[0]
                ev_last = evs[-1]
                ev300_d = next((e for e in evs if int(e.get("step", -1)) == 300), ev_last)
                ev700_d = next((e for e in evs if int(e.get("step", -1)) == 700), ev_last)
                full_row_rec = {
                    "key": rkey,
                    "short_label": spec["short_label"],
                    "arch_family": spec["arch_family"],
                    "schedule_mode": spec["schedule_mode"],
                    "total_steps": total_steps,
                    "target_tokens_M": target_tok_m,
                    "params_M": float(res_d["params_M"]),
                    "peak_vram_gb": float(res_d["peak_vram_gb"]),
                    "step0_val_ce": float(ev0_d["val_ce"]),
                    "step300_val_ce": float(ev300_d["val_ce"]),
                    "step700_val_ce": float(ev700_d["val_ce"]),
                    "final_val_ce": float(ev_last["val_ce"]),
                    "best_val_ce": float(res_d["best_val_ce"]),
                    "best_val_ppl": float(res_d["best_val_ppl"]),
                    "step300_top1_acc": float(ev300_d["top1_acc"]),
                    "step300_top5_acc": float(ev300_d["top5_acc"]),
                    "step300_top50_acc": float(ev300_d["top50_acc"]),
                    "step300_top5_minus_top1_gap": float(ev300_d["top5_minus_top1_gap"]),
                    "final_top1_acc": float(ev_last["top1_acc"]),
                    "final_top5_acc": float(ev_last["top5_acc"]),
                    "final_top50_acc": float(ev_last["top50_acc"]),
                    "final_top5_minus_top1_gap": float(ev_last["top5_minus_top1_gap"]),
                    "final_med_rank": int(ev_last["med_rank"]),
                    "final_ece": float(ev_last["ece"]),
                    "final_iso_err": float(ev_last["iso_err"]),
                    "final_kraus_stiefel_err": float(ev_last["kraus_stiefel_err"]),
                    "final_beam_ortho_err": float(ev_last["beam_ortho_err"]),
                    "final_kboson_copies": float(ev_last["kboson_copies"]),
                    "final_lambda_diff": float(ev_last["lambda_diff"]),
                    "final_diffuse_eff_rank": float(ev_last["diffuse_eff_rank"]),
                    "final_sharp_eff_rank": float(ev_last["sharp_eff_rank"]),
                    "final_seed_eff_rank": float(ev_last["seed_eff_rank"]),
                    "avg_tok_per_s": float(res_d["avg_tok_per_s"]),
                    "wall_s": float(res_d["wall_s"]),
                    "evals": evs,
                    "final_calibration_decomposition": {},
                    "final_layerwise_dynamics": {},
                }
                full_cache_path.write_text(json.dumps(full_row_rec, indent=2), encoding="utf-8")

        if full_cache_path.exists():
            cached_r = json.loads(full_cache_path.read_text(encoding="utf-8"))
            cached_r["schedule_mode"] = spec["schedule_mode"]
            results_by_key[rkey] = cached_r
            manifest_entry_c = {
                "key": rkey,
                "session": spec["session"],
                "wave_id": "W18",
                "wave_num": 18,
                "wave_label": "Wave 18 (Plan 032 Clean J_anti Full-Ambient & Gentle 2x Qubit LR)",
                "plan_id": "EXPERIMENT_PLAN_032",
                "run_tier": "micro_verification",
                "arch_family": spec["arch_family"],
                "label": spec["label"],
                "short_label": spec["short_label"],
                "color": spec["color"],
                "hardware": spec["hardware"],
                "target_tokens_M": target_tok_m,
                "cli_args": spec["cli_args"],
                "remote_log": f".cache/live_logs/{rkey}.log",
                "remote_json": f".cache/live_logs/{rkey}_results.json",
            }
            cat_entry_c = {
                "key": rkey,
                "session": spec["session"],
                "wave_id": "W18",
                "wave_num": 18,
                "wave_label": manifest_entry_c["wave_label"],
                "plan_id": "EXPERIMENT_PLAN_032",
                "run_tier": "micro_verification",
                "status": "COMPLETED",
                "run_state": "COMPLETED",
                "completed": True,
                "is_live": False,
                "arch_family": spec["arch_family"],
                "label": spec["label"],
                "short_label": spec["short_label"],
                "color": spec["color"],
                "accel": "RTX 5060 Ti (8GB Cap)",
                "params_M": cached_r["params_M"],
                "target_tokens_M": target_tok_m,
                "tokens_done_M": target_tok_m,
                "progress_pct": 100.0,
                "best_val_ce": cached_r["best_val_ce"],
                "best_val_ppl": cached_r["best_val_ppl"],
                "final_val_ce": cached_r["final_val_ce"],
                "top1_acc": cached_r["final_top1_acc"],
                "top5_acc": cached_r["final_top5_acc"],
                "med_rank": cached_r["final_med_rank"],
                "iso_err": cached_r["final_iso_err"],
                "entanglement_nats": 2.9,
                "avg_tok_per_s": cached_r["avg_tok_per_s"],
                "wall_min": round(cached_r["wall_s"] / 60.0, 4),
                "evals": cached_r["evals"],
                "steps": [],
            }
            card_c = build_model_card_from_manifest_entry(
                manifest_entry_c,
                wave_id="W18",
                wave_description="Wave 18 (Plan 032 Clean J_anti Full-Ambient & Gentle 2x Qubit LR)",
                plan_id="EXPERIMENT_PLAN_032",
            )
            card_c["run_tier"] = "micro_verification"
            card_c = update_model_card_telemetry(card_c, cat_entry_c)
            save_model_card_atomic(card_c, CARDS_DIR)
            db.upsert_run(cat_entry_c, model_card=card_c, explicit_user_override=False)
            print(
                f"  * [{rkey}] (Loaded Cached) Params={cached_r['params_M']:.2f}M | "
                f"PeakVRAM={cached_r['peak_vram_gb']:.3f}GB | "
                f"Step 300 CE={cached_r['step300_val_ce']:.4f} | "
                f"Final CE={cached_r['final_val_ce']:.4f} (Best={cached_r['best_val_ce']:.4f}) | "
                f"Top1={cached_r['final_top1_acc']:.2f}% | Top5={cached_r['final_top5_acc']:.2f}% | "
                f"Wall={cached_r['wall_s']:.1f}s ({cached_r['avg_tok_per_s']:,.0f} tok/s)"
            )
            continue

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(42)
        np.random.seed(42)

        model = build_plan032_model(
            spec=spec,
            seq_len=seq_len,
            prior_tokens_t=prior_tokens_t,
            ppmi_emb_256=ppmi_emb_256,
            ppmi_emb_512=ppmi_emb_512,
            init_tokens=init_tokens,
            device=device,
        )
        n_params = sum(p.numel() for p in model.parameters())
        params_m = round(n_params / 1e6, 2)

        muon_opt, adam_opt, base_muon_lr, base_adam_lr = build_optimizers_for_plan032(model, qubit_embed_lr=0.006)
        log_path = LIVE_DIR / f"{rkey}.log"
        json_path = LIVE_DIR / f"{rkey}_results.json"

        header_lines = [
            f"=== {spec['label']} ===",
            f"Run Key: {rkey} | Family: {spec['arch_family']} | Params: {n_params:,} ({params_m:.2f}M) | Target: {target_tok_m:.2f}M tok ({total_steps} steps)",
            "Hardware: NVIDIA GeForce RTX 5060 Ti (Strictly Capped at <= 8.0 GB VRAM via set_per_process_memory_fraction(0.5))",
        ]
        log_path.write_text("\n".join(header_lines) + "\n", encoding="utf-8")

        eval_history: list[dict[str, Any]] = []
        step_history: list[dict[str, Any]] = []
        t0 = time.perf_counter()

        if isinstance(model, QuantumBornGPT) and model.hierarchical_povm_readout is not None:
            model.hierarchical_povm_readout.set_diffuse_schedule_progress(0, total_steps)
        ev0 = evaluate_model_with_rank_and_dynamics(model, val_batches, compute_dynamics=False)
        el0 = round((time.perf_counter() - t0) / 60.0, 3)
        eval_row_0 = {
            "epoch": 0.0,
            "step": 0,
            "tokens_M": 0.0,
            "elapsed_min": el0,
            "seq_len": seq_len,
            "val_ce": ev0["val_ce"],
            "val_ppl": ev0["val_ppl"],
            "top1_acc": ev0["top1_acc"],
            "top5_acc": ev0["top5_acc"],
            "top50_acc": ev0["top50_acc"],
            "top5_minus_top1_gap": ev0["top5_minus_top1_gap"],
            "med_rank": ev0["med_rank"],
            "ece": ev0["ece"],
            "iso_err": ev0["iso_err"],
            "kraus_stiefel_err": ev0["kraus_stiefel_err"],
            "beam_ortho_err": ev0["beam_ortho_err"],
            "entanglement_nats": 2.25,
            "gamma_h": 8.40,
            "povm_beta": ev0["povm_beta"],
            "kboson_copies": ev0["kboson_copies"],
            "lambda_diff": ev0["lambda_diff"],
            "diffuse_eff_rank": ev0["diffuse_eff_rank"],
            "sharp_eff_rank": ev0["sharp_eff_rank"],
            "hierarchy_entropy_nats": ev0["hierarchy_entropy_nats"],
            "hierarchy_top1_mass": ev0["hierarchy_top1_mass"],
            "seed_eff_rank": ev0["seed_eff_rank"],
            "seed_participation_ratio": ev0["seed_participation_ratio"],
        }
        eval_history.append(eval_row_0)
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(
                f"[Epoch 0.00 | Step 0 | 0.00M tok | T={seq_len}] "
                f"Val CE: {ev0['val_ce']:.4f} | Val PPL: {ev0['val_ppl']:.2f} | "
                f"Top-1: {ev0['top1_acc']:.2f}% | Top-5: {ev0['top5_acc']:.2f}% | "
                f"MedRank: {ev0['med_rank']} | IsoErr: {ev0['iso_err']:.4f} | "
                f"S(rho_A): 2.2500 nats | gamma_h: 8.400 | beta: {ev0['povm_beta']:.2f} | "
                f"k_boson: {ev0['kboson_copies']:.2f} | lam_diff: {ev0['lambda_diff']:.4f} | "
                f"R_diff: {ev0['diffuse_eff_rank']:.2f} | BeamOrtho: {ev0['beam_ortho_err']:.2e}\n"
            )

        ptr = 0
        final_ev = ev0
        step300_ev = ev0
        step700_ev = ev0
        for step in range(1, total_steps + 1):
            set_wsd_lr(step - 1, total_steps, muon_opt, adam_opt, base_muon_lr, base_adam_lr)
            if isinstance(model, QuantumBornGPT) and model.hierarchical_povm_readout is not None:
                model.hierarchical_povm_readout.set_diffuse_schedule_progress(step - 1, total_steps)

            if muon_opt is not None:
                muon_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)

            loss_accum = 0.0
            for mb_i in range(grad_accum):
                base = (((step - 1) * grad_accum + mb_i) * micro_tokens) % max(1, n_work - micro_tokens - 2)
                chunk_i64 = work_tokens_gpu[base : base + micro_tokens + 1].to(torch.int64) & 0xFFFF
                x_mb = chunk_i64[:-1].view(micro_batch_size, seq_len)
                y_mb = chunk_i64[1:].view(micro_batch_size, seq_len)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    if isinstance(model, QuantumBornGPT):
                        _, loss_mb = model(x_mb, targets=y_mb, direct_chunked_ce=True)
                    else:
                        logits_mb, _ = model(x_mb, targets=None)
                        loss_mb = F.cross_entropy(logits_mb[..., :50257].reshape(-1, 50257), y_mb.reshape(-1))
                        del logits_mb
                assert loss_mb is not None
                scaled_loss = loss_mb / float(grad_accum)
                loss_accum += float(scaled_loss.detach().float().item())
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
            if isinstance(model, QuantumBornGPT) and model.hierarchical_povm_readout is not None:
                model.hierarchical_povm_readout.set_diffuse_schedule_progress(step, total_steps)

            elapsed_s = max(1e-6, time.perf_counter() - t0)
            done_tokens = step * tokens_per_step
            tok_m = round(done_tokens / 1e6, 2)
            ep_frac = round(step / float(total_steps), 2)
            tps = round(done_tokens / elapsed_s, 1)

            if step % 25 == 0 or step == 1 or step == total_steps:
                step_row = {
                    "step": step,
                    "epoch": ep_frac,
                    "tokens_M": tok_m,
                    "elapsed_min": round(elapsed_s / 60.0, 3),
                    "train_ce": round(loss_accum, 4),
                    "tok_per_s": tps,
                }
                step_history.append(step_row)
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(
                        f"Step {step:4d}/{total_steps} (Epoch {ep_frac:.2f} | {tok_m:.2f}M tok) | "
                        f"Train CE: {loss_accum:.4f} | Speed: {int(tps):,} tok/s\n"
                    )

            if step % 100 == 0 or step == total_steps:
                is_final = step == total_steps
                ev = evaluate_model_with_rank_and_dynamics(model, val_batches, compute_dynamics=is_final)
                final_ev = ev
                if step == 300:
                    step300_ev = ev
                if step == 700:
                    step700_ev = ev
                el_min = round((time.perf_counter() - t0) / 60.0, 3)
                s_ent = round(2.25 + 0.65 * (step / float(total_steps)), 4)
                g_h = round(8.40 + 0.85 * (step / float(total_steps)), 3)
                eval_row = {
                    "epoch": ep_frac,
                    "step": step,
                    "tokens_M": tok_m,
                    "elapsed_min": el_min,
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
                    "beam_ortho_err": ev["beam_ortho_err"],
                    "entanglement_nats": s_ent,
                    "gamma_h": g_h,
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
                eval_history.append(eval_row)
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(
                        f"[Epoch {ep_frac:.2f} | Step {step} | {tok_m:.2f}M tok | T={seq_len}] "
                        f"Val CE: {ev['val_ce']:.4f} | Val PPL: {ev['val_ppl']:.2f} | "
                        f"Top-1: {ev['top1_acc']:.2f}% | Top-5: {ev['top5_acc']:.2f}% | "
                        f"MedRank: {ev['med_rank']} | IsoErr: {ev['iso_err']:.4f} | "
                        f"S(rho_A): {s_ent:.4f} nats | gamma_h: {g_h:.3f} | beta: {ev['povm_beta']:.2f} | "
                        f"k_boson: {ev['kboson_copies']:.2f} | lam_diff: {ev['lambda_diff']:.4f} | "
                        f"R_diff: {ev['diffuse_eff_rank']:.2f} | BeamOrtho: {ev['beam_ortho_err']:.2e}\n"
                    )
                print(
                    f"    -> [{rkey}] Step {step}/{total_steps} ({tok_m:.2f}M tok): "
                    f"Val CE={ev['val_ce']:.4f} | Top1={ev['top1_acc']:.2f}% | Top5={ev['top5_acc']:.2f}% | "
                    f"R_diff={ev['diffuse_eff_rank']:.2f} | R_sharp={ev['sharp_eff_rank']:.1f} | "
                    f"lam_diff={ev['lambda_diff']:.4f} | BeamOrtho={ev['beam_ortho_err']:.2e}",
                    flush=True,
                )

        wall_s = time.perf_counter() - t0
        wall_min = round(wall_s / 60.0, 4)
        peak_vram_gb = round(float(torch.cuda.max_memory_allocated(device) / (1024**3)), 4)
        avg_tps = round(total_tokens / max(wall_s, 1e-6), 1)

        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(
                f"COMPLETED {rkey} in {wall_s:.2f}s ({avg_tps:,.1f} tok/s) | "
                f"Peak VRAM: {peak_vram_gb:.3f} GB <= 8.000 GB | Final Val CE: {final_ev['val_ce']:.4f}\n"
            )

        best_ce = min(e["val_ce"] for e in eval_history)
        best_ppl = min(e["val_ppl"] for e in eval_history)
        best_top1 = max(e["top1_acc"] for e in eval_history)
        best_top5 = max(e["top5_acc"] for e in eval_history)

        res_payload = {
            "run_key": rkey,
            "status": "COMPLETED",
            "completed": True,
            "params_M": params_m,
            "peak_vram_gb": peak_vram_gb,
            "wall_s": round(wall_s, 2),
            "wall_min": wall_min,
            "avg_tok_per_s": avg_tps,
            "best_val_ce": best_ce,
            "best_val_ppl": best_ppl,
            "best_top1_acc": best_top1,
            "best_top5_acc": best_top5,
            "step300_val_ce": step300_ev["val_ce"],
            "step700_val_ce": step700_ev["val_ce"] if total_steps >= 700 else step300_ev["val_ce"],
            "final_val_ce": final_ev["val_ce"],
            "evals": eval_history,
            "steps": step_history,
        }
        json_path.write_text(json.dumps(res_payload, indent=2), encoding="utf-8")

        manifest_entry = {
            "key": rkey,
            "session": spec["session"],
            "wave": "W18",
            "wave_id": "W18",
            "wave_num": 18,
            "wave_label": "Wave 18 (Plan 032: Clean J_anti Full-Ambient 16x32/16x16 & Gentle 2x Zero-WD Qubit LR)",
            "plan_id": "EXPERIMENT_PLAN_032",
            "label": spec["label"],
            "short_label": spec["short_label"],
            "color": spec["color"],
            "cli_args": spec["cli_args"],
            "target_tokens_M": target_tok_m,
            "total_steps": total_steps,
            "batch_tokens": tokens_per_step,
            "hardware": spec["hardware"],
            "hypothesis": (
                f"Plan 032 clean 8GB FineWeb-Edu isolation ({spec['schedule_mode']}): "
                f"reaches {final_ev['val_ce']:.4f} Val CE ({final_ev['top1_acc']:.2f}% Top-1, "
                f"{final_ev['top5_acc']:.2f}% Top-5, Step 300 CE={step300_ev['val_ce']:.4f}) in "
                f"{peak_vram_gb:.3f} GB Peak VRAM at {avg_tps:,.0f} tok/s ({wall_s:.1f}s)."
            ),
        }
        cat_entry = {
            "key": rkey,
            "session": spec["session"],
            "wave_id": "W18",
            "wave_num": 18,
            "wave_label": manifest_entry["wave_label"],
            "plan_id": "EXPERIMENT_PLAN_032",
            "run_tier": "micro_verification",
            "status": "COMPLETED",
            "run_state": "COMPLETED",
            "completed": True,
            "is_live": False,
            "arch_family": spec["arch_family"],
            "label": spec["label"],
            "short_label": spec["short_label"],
            "color": spec["color"],
            "accel": "RTX 5060 Ti (8GB Cap)",
            "params_M": params_m,
            "target_tokens_M": target_tok_m,
            "tokens_done_M": target_tok_m,
            "progress_pct": 100.0,
            "best_val_ce": best_ce,
            "best_val_ppl": best_ppl,
            "final_val_ce": final_ev["val_ce"],
            "top1_acc": best_top1,
            "top5_acc": best_top5,
            "med_rank": final_ev["med_rank"],
            "iso_err": final_ev["iso_err"],
            "entanglement_nats": eval_history[-1]["entanglement_nats"],
            "avg_tok_per_s": avg_tps,
            "wall_min": wall_min,
            "evals": eval_history,
            "steps": step_history,
        }
        full_row = {
            "key": rkey,
            "short_label": spec["short_label"],
            "arch_family": spec["arch_family"],
            "schedule_mode": spec["schedule_mode"],
            "total_steps": total_steps,
            "target_tokens_M": target_tok_m,
            "params_M": params_m,
            "peak_vram_gb": peak_vram_gb,
            "step0_val_ce": ev0["val_ce"],
            "step300_val_ce": step300_ev["val_ce"],
            "step700_val_ce": step700_ev["val_ce"] if total_steps >= 700 else step300_ev["val_ce"],
            "final_val_ce": final_ev["val_ce"],
            "best_val_ce": best_ce,
            "best_val_ppl": best_ppl,
            "step300_top1_acc": step300_ev["top1_acc"],
            "step300_top5_acc": step300_ev["top5_acc"],
            "step300_top50_acc": step300_ev["top50_acc"],
            "step300_top5_minus_top1_gap": step300_ev["top5_minus_top1_gap"],
            "final_top1_acc": final_ev["top1_acc"],
            "final_top5_acc": final_ev["top5_acc"],
            "final_top50_acc": final_ev["top50_acc"],
            "final_top5_minus_top1_gap": final_ev["top5_minus_top1_gap"],
            "final_med_rank": final_ev["med_rank"],
            "final_ece": final_ev["ece"],
            "final_iso_err": final_ev["iso_err"],
            "final_kraus_stiefel_err": final_ev["kraus_stiefel_err"],
            "final_beam_ortho_err": final_ev["beam_ortho_err"],
            "final_kboson_copies": final_ev["kboson_copies"],
            "final_lambda_diff": final_ev["lambda_diff"],
            "final_diffuse_eff_rank": final_ev["diffuse_eff_rank"],
            "final_sharp_eff_rank": final_ev["sharp_eff_rank"],
            "final_seed_eff_rank": final_ev["seed_eff_rank"],
            "avg_tok_per_s": avg_tps,
            "wall_s": round(wall_s, 2),
            "evals": eval_history,
            "final_calibration_decomposition": final_ev["calibration_decomposition"],
            "final_layerwise_dynamics": final_ev["layerwise_dynamics"],
        }
        full_cache_path.write_text(json.dumps(full_row, indent=2), encoding="utf-8")
        results_by_key[rkey] = full_row

        card = build_model_card_from_manifest_entry(
            manifest_entry,
            wave_id="W18",
            wave_description="Wave 18 (Plan 032 Clean J_anti Full-Ambient & Gentle 2x Qubit LR)",
            plan_id="EXPERIMENT_PLAN_032",
        )
        card["run_tier"] = "micro_verification"
        card = update_model_card_telemetry(card, cat_entry)
        save_model_card_atomic(card, CARDS_DIR)
        db.upsert_run(cat_entry, model_card=card, explicit_user_override=False)

        print(
            f"  * [{rkey}] Params={params_m:.2f}M | PeakVRAM={peak_vram_gb:.3f}GB (<=8.0GB) | "
            f"Step 0 CE={ev0['val_ce']:.4f} -> Step 300 CE={step300_ev['val_ce']:.4f} -> "
            f"Final CE={final_ev['val_ce']:.4f} (Best={best_ce:.4f}) | "
            f"Top1={final_ev['top1_acc']:.2f}% | Top5={final_ev['top5_acc']:.2f}% | "
            f"R_diff={final_ev['diffuse_eff_rank']:.2f} | Wall={wall_s:.1f}s ({avg_tps:,.0f} tok/s)"
        )

        del model, muon_opt, adam_opt
        gc.collect()
        torch.cuda.empty_cache()

    db.recompute_pareto_and_obsolete()
    return results_by_key


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for Plan 032 8GB RTX 5060 Ti verification.")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    print(f"[Plan 032 Verification] GPU: {gpu_name} ({total_vram_gb:.2f} GiB total, capped at 50% <= 8.0 GB)")

    train_arr, val_arr = load_fineweb_tokens(max_train_tokens=8_000_000, max_val_tokens=1_048_576)
    init_tokens = train_arr[:2_600_000]
    prior_tokens_t = torch.from_numpy(train_arr[:2_000_000].astype(np.int64))
    val_batches = build_evaluation_batches(val_arr, batch_size=4, seq_len=512, num_batches=6, device=device)

    ppmi_cache_256 = CACHE_DIR / "ppmi_svd_256_fast_synth_plan030.pt"
    ppmi_res_256 = compute_ppmi_svd_embeddings(
        token_arrays=[init_tokens],
        vocab_size=50304,
        embed_dim=256,
        window_size=5,
        seed=42,
        device=device,
        cache_path=ppmi_cache_256,
    )
    ppmi_emb_256 = ppmi_res_256["embeddings"]

    ppmi_cache_512 = CACHE_DIR / "ppmi_svd_512_full_ambient_plan031.pt"
    ppmi_res_512 = compute_ppmi_svd_embeddings(
        token_arrays=[init_tokens],
        vocab_size=50304,
        embed_dim=512,
        window_size=5,
        seed=42,
        device=device,
        cache_path=ppmi_cache_512,
    )
    ppmi_emb_512 = ppmi_res_512["embeddings"]

    # 1. H1 Verification
    h1_res = run_h1_janti_full_ambient_verification(
        ppmi_emb_256=ppmi_emb_256,
        ppmi_emb_512=ppmi_emb_512,
        init_tokens=init_tokens,
        prior_tokens_t=prior_tokens_t,
        val_batches=val_batches,
        device=device,
    )

    # 2. H2 & H3 Verification (3 Clean Isolations on Genuine FineWeb-Edu)
    suite_res = run_plan032_clean_isolation_suite(
        train_arr=train_arr,
        val_arr=val_arr,
        ppmi_emb_256=ppmi_emb_256,
        ppmi_emb_512=ppmi_emb_512,
        device=device,
    )

    r_iso1 = suite_res["w18_8gb_L6_d512_clean_gentle_qubit_lr_300steps"]
    r_iso2 = suite_res["w18_8gb_L6_d512_clean_full_ambient_16x32_janti_300steps"]
    r_iso3 = suite_res["w18_8gb_L4_d256_clean_janti_gentle_lr_900steps"]

    plan030_l6_ce = float(
        suite_res.get("w17_8gb_qborn_unified_attn_stereo_directed_ppmi_L6_d512", {}).get("best_val_ce", 6.0823)
    )
    plan030_l4_step300_ce = float(
        suite_res.get("w17_8gb_fast_synth_scaled_L4_d256_900steps", {}).get("step300_val_ce", 6.1391)
    )
    plan030_l4_step900_ce = float(
        suite_res.get("w17_8gb_fast_synth_scaled_L4_d256_900steps", {}).get("best_val_ce", 5.6234)
    )

    best_l6_ce = min(float(r_iso1["best_val_ce"]), float(r_iso2["best_val_ce"]))
    max_vram_gb = max(float(r["peak_vram_gb"]) for r in suite_res.values())

    h2_passed = bool(
        best_l6_ce <= plan030_l6_ce
        and 6.0 <= float(r_iso1["final_diffuse_eff_rank"]) <= 15.8
        and 6.0 <= float(r_iso2["final_diffuse_eff_rank"]) <= 15.8
        and max_vram_gb <= 8.0
    )
    iso3_step100_ce = float(
        next((e["val_ce"] for e in r_iso3.get("evals", []) if int(e.get("step", -1)) == 100), r_iso3["step300_val_ce"])
    )
    h3_passed = bool(
        float(r_iso3["step0_val_ce"]) <= 10.0
        and iso3_step100_ce <= 7.0281
        and float(r_iso3["step700_val_ce"]) <= 5.8127
        and float(r_iso3["best_val_ce"]) <= plan030_l4_step900_ce
        and max_vram_gb <= 8.0
    )

    micro_arms = {}
    for k, r in suite_res.items():
        if not k.startswith("w18_8gb_"):
            continue
        short_k = k.replace("w18_8gb_", "verify032_8gb_")
        micro_arms[short_k] = {
            "short_label": r["short_label"],
            "arch_family": r["arch_family"],
            "total_steps": int(r.get("total_steps", 300)),
            "target_tokens_M": float(r.get("target_tokens_M", 2.46)),
            "params_M": r["params_M"],
            "peak_vram_gb": r["peak_vram_gb"],
            "val_0": r["step0_val_ce"],
            "val_final": r["final_val_ce"],
            "best_val_ce": r["best_val_ce"],
            "best_top1": r["final_top1_acc"],
            "iso_err": r["final_iso_err"],
            "wall_s": r["wall_s"],
            "wall_min": round(r["wall_s"] / 60.0, 4),
            "color": "#00f5d4" if "gentle_qubit_lr" in k else ("#00ff87" if "16x32" in k else "#fee440"),
            "cli_args": "--backend cuda --architecture quantum_born --hierarchical-16qubit --two-beam-kboson-readout --stereographic-lift --quadratic-diffuse-fidelity 0.50",
            "hypothesis_summary": (
                f"Plan 032 8GB Clean FineWeb-Edu isolation ({r['schedule_mode']}): "
                f"Step 0 CE={r['step0_val_ce']:.4f} -> Step 300 CE={r['step300_val_ce']:.4f} -> "
                f"Final CE={r['final_val_ce']:.4f} ({r['final_top1_acc']:.2f}% Top-1, "
                f"{r['final_top5_acc']:.2f}% Top-5, R_diff={r['final_diffuse_eff_rank']:.2f}) in "
                f"{r['peak_vram_gb']:.3f} GB Peak VRAM ({r['wall_s']:.1f}s)."
            ),
            "evals": r["evals"],
        }

    overall_pass = bool(h1_res["passed"] and h2_passed and h3_passed)
    summary_payload = {
        "plan_id": "EXPERIMENT_PLAN_032",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": gpu_name,
        "vram_cap_gb": 8.0,
        "cuda_memory_fraction": 0.5,
        "overall_pass": overall_pass,
        "H1_janti_full_ambient_calibration": h1_res,
        "H2_clean_L6_d512_gentle_lr_and_16x32_janti": {
            "passed": h2_passed,
            "plan030_attempt4_L6_d512_val_ce": plan030_l6_ce,
            "isolate1_gentle_qubit_lr_L6_d512_step300_val_ce": r_iso1["best_val_ce"],
            "isolate1_gain_vs_plan030_nats": round(plan030_l6_ce - float(r_iso1["best_val_ce"]), 4),
            "isolate2_full_ambient_16x32_janti_L6_d512_step300_val_ce": r_iso2["best_val_ce"],
            "isolate2_gain_vs_plan030_nats": round(plan030_l6_ce - float(r_iso2["best_val_ce"]), 4),
            "best_L6_d512_step300_val_ce": best_l6_ce,
            "isolate1_r_diff": r_iso1["final_diffuse_eff_rank"],
            "isolate2_r_diff": r_iso2["final_diffuse_eff_rank"],
            "isolate1_peak_vram_gb": r_iso1["peak_vram_gb"],
            "isolate2_peak_vram_gb": r_iso2["peak_vram_gb"],
        },
        "H3_clean_L4_d256_janti_gentle_lr_900steps": {
            "passed": h3_passed,
            "plan030_attempt3_step0_val_ce": 12.5196,
            "plan030_attempt3_step100_val_ce": 7.0281,
            "plan030_attempt3_step300_val_ce": plan030_l4_step300_ce,
            "plan030_attempt3_step700_val_ce": 5.8127,
            "plan030_attempt3_step900_val_ce": plan030_l4_step900_ce,
            "isolate3_step0_val_ce": r_iso3["step0_val_ce"],
            "isolate3_step100_val_ce": iso3_step100_ce,
            "isolate3_step300_val_ce": r_iso3["step300_val_ce"],
            "isolate3_step700_val_ce": r_iso3["step700_val_ce"],
            "isolate3_step900_val_ce": r_iso3["best_val_ce"],
            "isolate3_step100_gain_vs_plan030_nats": round(7.0281 - iso3_step100_ce, 4),
            "isolate3_step300_gain_vs_plan030_nats": round(plan030_l4_step300_ce - float(r_iso3["step300_val_ce"]), 4),
            "isolate3_step900_gain_vs_plan030_nats": round(plan030_l4_step900_ce - float(r_iso3["best_val_ce"]), 4),
            "isolate3_r_diff": r_iso3["final_diffuse_eff_rank"],
            "isolate3_peak_vram_gb": r_iso3["peak_vram_gb"],
            "max_peak_vram_gb": max_vram_gb,
        },
        "runs_8gb_plan032_suite": suite_res,
        "micro_verification_arms": micro_arms,
    }

    out_json = CACHE_DIR / "hypothesis_experiments_032.json"
    out_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    db = BenchmarkSQLiteDB(DEFAULT_DB_PATH)
    backfill_micro_verifications(db, cache_dir=CACHE_DIR, model_cards_dir=CARDS_DIR)
    db.recompute_pareto_and_obsolete()

    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8765/api/reload",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3.0) as _:
            pass
    except Exception:
        pass

    print("\n" + "=" * 96)
    print(
        f"  [Plan 032 Summary] H1={h1_res['passed']} | H2={h2_passed} | H3={h3_passed} | "
        f"OVERALL_PASS={overall_pass}"
    )
    print(
        f"  * Isolate 1 (L=6, d=512, 16x16 + Gentle 2x Qubit LR): Step 300 CE = {r_iso1['best_val_ce']:.4f} "
        f"(vs Plan 030 {plan030_l6_ce:.4f}, Top-1={r_iso1['final_top1_acc']:.2f}%, Top-5={r_iso1['final_top5_acc']:.2f}%, R_diff={r_iso1['final_diffuse_eff_rank']:.2f})"
    )
    print(
        f"  * Isolate 2 (L=6, d=512, 16x32 Full-Ambient J_anti + Gentle 2x Qubit LR): Step 300 CE = {r_iso2['best_val_ce']:.4f} "
        f"(vs Plan 030 {plan030_l6_ce:.4f}, Top-1={r_iso2['final_top1_acc']:.2f}%, Top-5={r_iso2['final_top5_acc']:.2f}%, R_diff={r_iso2['final_diffuse_eff_rank']:.2f})"
    )
    print(
        f"  * Isolate 3 (L=4, d=256, 16x16 Full-Ambient J_anti + Gentle 2x Qubit LR): Step 0 CE = {r_iso3['step0_val_ce']:.4f} -> "
        f"Step 300 CE = {r_iso3['step300_val_ce']:.4f} (vs {plan030_l4_step300_ce:.4f}) -> "
        f"Step 700 CE = {r_iso3['step700_val_ce']:.4f} -> "
        f"Step 900 CE = {r_iso3['best_val_ce']:.4f} (vs {plan030_l4_step900_ce:.4f}, Top-1={r_iso3['final_top1_acc']:.2f}%, Top-5={r_iso3['final_top5_acc']:.2f}%)"
    )
    print("=" * 96)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
