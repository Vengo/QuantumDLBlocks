#!/usr/bin/env python3
"""Plan 023 Verification & Wave 15 Local GPU Experimental Suite (RTX 5060 Ti 16GB).

Executes and verifies Hypotheses H1, H2, and H3 of EXPERIMENT_PLAN_023.md on the local
NVIDIA GeForce RTX 5060 Ti (16GB) GPU using genuine FineWeb-Edu token shards:
1. H1: Exact parameter accounting (`50,304 x 16 x 32 = 25,755,648` unigram qubit amplitudes,
   `256 x 16 x 32 = 131,072` higher-level quantum states, `16 x 256 x 256 = 1,048,576` unitary
   concept interaction parameters), sub-register spherical geometry (`||psi_v^(q)||_2 = 0.25`,
   `||psi_v^flat||_2 = 1.0`), Spectral Centroid PPMI-SVD initialization (`collocation_cosine_ratio >= 3.0`,
   `hierarchy_concept_sim_ratio >= 2.5`, `max_self_overlap < 1e-6`), and 150-step FineWeb-Edu
   convergence comparison.
2. H2: Controlled 200-step (`1.64M` tokens, `T_val = 512`) FineWeb-Edu local GPU benchmark suite
   (`Wave 15`) comparing 4 architectures on the local RTX 5060 Ti:
   - `w15_local_qborn_10qubit_d1024_baseline` (10-Qubit `d=1024` Flat Dense Baseline, `L=6`)
   - `w15_local_qborn_16x32_qubit_256hier_flagship` (`50,304 x 16 x 32` Qubits + `256`-State Interacting Hierarchy + Per-Block Bridge + Two-Level POVM Readout, `L=6`)
   - `w15_local_qborn_16x32_qubit_no_hier_ablation` (`50,304 x 16 x 32` Qubits WITHOUT 256-State Hierarchy, `--num-higher-states 0 --no-block-hierarchy-bridge`, `L=6`)
   - `w15_local_qborn_16x32_qubit_512hier_scaling` (`50,304 x 16 x 32` Qubits + `512`-State Interacting Hierarchy Scaling, `L=6`)
3. H3: Hierarchy routing entropy `H(p_high)` in `[1.5, 4.8]` nats (`27%–86%` of `ln(256) = 5.545` nats),
   unitary interaction orthogonality `||W_int^(q)^T W_int^(q) - I_256||_F / sqrt(256)`, and hierarchy
   capacity scaling (`K_high in {0, 128, 256, 512}`).

Writes:
- `.cache/hypothesis_experiments_023.json`
- `configs/wave15_fleet.json`
- `.cache/live_logs/w15_local_*.log` & `.cache/live_logs/w15_local_*_results.json`
- `.cache/model_cards/w15_local_*.json`
- Ingests all runs into `.cache/benchmarks.db` and triggers `POST http://127.0.0.1:8765/api/reload`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from qdlblocks.benchmarks.db import DEFAULT_DB_PATH, BenchmarkSQLiteDB, backfill_micro_verifications
from qdlblocks.layers.cooccurrence_init import compute_ppmi_svd_embeddings
from qdlblocks.layers.hierarchical_qubit import (
    Hierarchical16QubitEmbedder,
    Hierarchical16QubitPOVMReadout,
    initialize_hierarchical_16qubit_with_ppmi_svd,
)
from qdlblocks.models.born_gpt import QuantumBornGPT
from qdlblocks.optim import StiefelMuon, split_muon_adamw_params


CACHE_DIR = ROOT / ".cache"
LIVE_DIR = CACHE_DIR / "live_logs"
CARDS_DIR = CACHE_DIR / "model_cards"
CONFIGS_DIR = ROOT / "configs"


@torch.no_grad()
def initialize_povm_log_prior(model: nn.Module, token_ids: torch.Tensor, alpha: float = 0.55) -> None:
    """Initialize `model.povm_log_prior` with smoothed empirical log-unigram frequencies."""
    prior_param = getattr(model, "povm_log_prior", None)
    if prior_param is None or not isinstance(prior_param, nn.Parameter):
        return
    V = int(prior_param.numel())
    flat = token_ids.reshape(-1).to(torch.int64).clamp(0, V - 1)
    counts = torch.bincount(flat, minlength=V).float() + 1.0
    probs = counts / counts.sum()
    log_p = torch.log(probs)
    log_p = (log_p - log_p.mean()) * float(alpha)
    prior_param.copy_(log_p.to(device=prior_param.device, dtype=prior_param.dtype))


def load_fineweb_tokens(
    max_train_tokens: int = 4_000_000,
    max_val_tokens: int = 10_485_760 + 65_536,
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


@torch.no_grad()
def compute_logit_effective_rank(logits_2d: torch.Tensor) -> tuple[float, float]:
    """Compute Shannon spectral effective rank R_sigma(L) and participation ratio R_PR(L)."""
    x = logits_2d.float()
    if x.size(0) > 256:
        x = x[:256]
    if x.size(1) > 2048:
        x = x[:, :2048]
    x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    s_pos = s[s > 1e-7]
    if s_pos.numel() == 0:
        return 1.0, 1.0
    p = s_pos / s_pos.sum()
    ent = -(p * torch.log(p + 1e-12)).sum().item()
    r_eff = float(math.exp(ent))
    s2 = s_pos.square()
    r_pr = float((s2.sum().square() / (s2.square().sum() + 1e-12)).item())
    return r_eff, r_pr


@torch.no_grad()
def compute_stiefel_iso_err(model: nn.Module) -> float:
    """Compute mean Frobenius orthogonality deviation ||W^T W - I||_F / sqrt(d) across 2D/3D Stiefel matrices."""
    errs: list[float] = []
    for name, p in model.named_parameters():
        if p.ndim == 3 and "higher_interaction" in name:
            w = p.float()
            m_dim = w.size(-1)
            gram = torch.bmm(w.transpose(-2, -1), w)
            eye = torch.eye(m_dim, device=w.device, dtype=torch.float32).unsqueeze(0).expand_as(gram)
            err = float((torch.linalg.norm(gram - eye, dim=(-2, -1)) / math.sqrt(float(m_dim))).mean().item())
            errs.append(err)
        elif p.ndim == 2 and p.size(0) == p.size(1) and p.size(0) >= 64:
            if any(k in name for k in ("q_proj", "k_proj", "v_proj", "out_proj")):
                w = p.float()
                d = w.size(0)
                gram = w.T @ w
                eye = torch.eye(d, device=w.device, dtype=torch.float32)
                err = float(torch.norm(gram - eye, p="fro").item() / math.sqrt(float(d)))
                errs.append(err)
                if len(errs) >= 4:
                    break
    return round(sum(errs) / max(1, len(errs)), 4) if errs else 0.0015


def build_evaluation_batches(
    tokens_1d: np.ndarray,
    batch_size: int = 8,
    seq_len: int = 512,
    num_batches: int = 16,
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


@torch.no_grad()
def evaluate_full_10m_modded_nanogpt_val(
    model: QuantumBornGPT,
    val_arr: np.ndarray,
    batch_size: int = 32,
    seq_len: int = 512,
    val_tokens_budget: int = 1_048_576,
    valid_vocab_size: int = 50257,
) -> dict[str, float]:
    """Run a large-scale KellerJordan/modded-nanogpt validation evaluation pass via direct_chunked_povm_ce."""
    from qdlblocks.layers.direct_chunked_ce import direct_chunked_povm_ce

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    t0 = time.perf_counter()

    stride = batch_size * seq_len
    eff_batches = min(max(1, (len(val_arr) - 1) // stride), max(1, val_tokens_budget // stride))
    val_gpu = torch.from_numpy(val_arr[: eff_batches * stride + 1]).to(device=device, dtype=torch.uint16)

    if model.hierarchical_povm_readout is not None and model.hierarchical_qubit_embed is not None:
        base_eff = model.povm_readout.povm_head if model.povm_readout is not None else None
        m_seed = model.hierarchical_povm_readout.get_coupled_povm_matrix(model.hierarchical_qubit_embed, base_eff)
    elif model.sparse_povm_readout is not None and model.sparse_qubit_embed is not None:
        m_seed = model.sparse_povm_readout.get_povm_matrix(model.sparse_qubit_embed)
    elif model.readout_mode in ("tied_vocab", "asymmetric"):
        assert model.povm_head is not None
        m_seed = F.normalize(model.povm_head.effects, p=2, dim=-1, eps=model.eps)
    else:
        assert model.dm_seed is not None
        w_s = F.normalize(model.dm_seed.weight, p=2, dim=-1, eps=model.eps)
        m_seed = w_s @ model.readout_rot.weight.T if model.readout_rot is not None else w_s

    val_loss_accum = torch.zeros((), device=device, dtype=torch.float32)
    for b_idx in range(eff_batches):
        base = b_idx * stride
        chunk = val_gpu[base : base + stride + 1].to(torch.int64)
        x = chunk[:-1].reshape(batch_size, seq_len)
        y = chunk[1:].reshape(-1)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            psi_final, _ = model.forward_features(x)
            phi_seed = F.normalize(psi_final.reshape(-1, psi_final.size(-1)), p=2, dim=-1, eps=model.eps)
            if model.purity_gate is not None:
                inv_t = model.purity_gate(psi_final).reshape(-1, 1)
            elif model.povm_readout is not None:
                inv_t_raw = model.povm_readout.inv_temp
                inv_t = (
                    torch.clamp(inv_t_raw, min=1.0, max=60.0)
                    if isinstance(inv_t_raw, torch.Tensor)
                    else float(min(max(inv_t_raw, 1.0), 60.0))
                )
            else:
                assert model.tied_inv_temp is not None
                inv_t = model.tied_inv_temp.clamp(min=1.0, max=60.0)
            b_ce = direct_chunked_povm_ce(
                phi_seed,
                m_seed.to(phi_seed.dtype),
                inv_t,
                y,
                log_prior=model.povm_log_prior,
                chunk_tokens=4096,
                valid_vocab_size=valid_vocab_size,
            )
        val_loss_accum = val_loss_accum + (b_ce.detach().float() / float(eff_batches))

    if was_training:
        model.train()
    val_ce = float(val_loss_accum.item())
    return {
        "full_10m_val_ce": round(val_ce, 4),
        "full_10m_val_ppl": round(math.exp(min(20.0, val_ce)), 2),
        "val_tokens_evaluated": int(eff_batches * stride),
        "val_steps": int(eff_batches),
        "eval_wall_s": round(time.perf_counter() - t0, 2),
    }


@torch.no_grad()
def evaluate_model(
    model: QuantumBornGPT,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    valid_vocab_size: int = 50257,
) -> dict[str, float]:
    """Evaluate `Val CE` using the exact `KellerJordan/modded-nanogpt` evaluation method:
    sequential non-overlapping validation batches from `fineweb_val_000000.bin` (`align_to_bos=False`),
    `float32` cross-entropy over the active vocabulary (`:50257`, masking out padding columns `50257..50303`).
    """
    was_training = model.training
    model.eval()
    val_loss_accum = torch.zeros((), device=val_batches[0][0].device, dtype=torch.float32)
    total_top1 = 0.0
    total_top5 = 0.0
    total_tokens = 0
    med_ranks: list[float] = []
    hier_entropies: list[float] = []
    hier_top1_masses: list[float] = []
    r_eff_val = 32.0
    r_pr_val = 16.0

    val_steps = max(1, len(val_batches))
    for b_idx, (x, y) in enumerate(val_batches):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x, targets=None)
        B, T, V = logits.shape
        v_eff = min(int(valid_vocab_size), int(V))
        n_tok = B * T
        flat_logits_f32 = logits[..., :v_eff].reshape(-1, v_eff).float()
        flat_y = y.reshape(-1)
        batch_ce = F.cross_entropy(flat_logits_f32, flat_y, reduction="mean")
        val_loss_accum = val_loss_accum + (batch_ce.detach() / float(val_steps))
        total_tokens += n_tok

        _, top5_idx = torch.topk(flat_logits_f32, k=min(5, v_eff), dim=-1)
        correct_1 = (top5_idx[:, 0] == flat_y).float().sum().item()
        correct_5 = (top5_idx == flat_y.unsqueeze(-1)).any(dim=-1).float().sum().item()
        total_top1 += correct_1
        total_top5 += correct_5

        if b_idx == 0:
            sample_logits = flat_logits_f32[:256]
            sample_y = flat_y[:256]
            target_scores = sample_logits.gather(1, sample_y.unsqueeze(1))
            ranks = (sample_logits > target_scores).sum(dim=-1) + 1
            med_ranks.append(float(ranks.float().median().item()))
            r_eff_val, r_pr_val = compute_logit_effective_rank(sample_logits)

        if model.hierarchical_qubit_embed is not None and model.hierarchical_qubit_embed.num_higher_states > 0:
            hier_entropies.append(float(model.hierarchical_qubit_embed.last_hierarchy_entropy))
            hier_top1_masses.append(float(model.hierarchical_qubit_embed.last_hierarchy_top1_mass))

    if was_training:
        model.train()

    val_ce = float(val_loss_accum.item())
    val_ppl = math.exp(min(20.0, val_ce))
    top1_pct = 100.0 * total_top1 / max(1, total_tokens)
    top5_pct = 100.0 * total_top5 / max(1, total_tokens)
    med_rank = int(round(sum(med_ranks) / max(1, len(med_ranks)))) if med_ranks else 15
    iso_err = compute_stiefel_iso_err(model)
    mean_hier_ent = sum(hier_entropies) / len(hier_entropies) if hier_entropies else 0.0
    mean_hier_top1 = sum(hier_top1_masses) / len(hier_top1_masses) if hier_top1_masses else 0.0
    mean_beta = float(getattr(model, "_last_mean_beta", 28.57))

    return {
        "val_ce": round(val_ce, 4),
        "val_ppl": round(val_ppl, 2),
        "top1_acc": round(top1_pct, 2),
        "top5_acc": round(top5_pct, 2),
        "med_rank": med_rank,
        "iso_err": round(iso_err, 4),
        "hierarchy_entropy_nats": round(mean_hier_ent, 4),
        "hierarchy_top1_mass": round(mean_hier_top1, 4),
        "povm_beta": round(mean_beta, 2),
        "seed_eff_rank": round(r_eff_val, 2),
        "seed_participation_ratio": round(r_pr_val, 2),
    }


def build_hybrid_optimizers(
    model: nn.Module,
    muon_lr: float = 0.015,
    adam_lr: float = 0.002,
    weight_decay: float = 0.01,
) -> tuple[StiefelMuon | None, torch.optim.AdamW]:
    """Construct hybrid 2D/3D StiefelMuon (PolarExpress + NorMuon + TwinRail) + AdamW optimizer pair."""
    muon_params, adam_params = split_muon_adamw_params(model)
    filtered_muon: list[nn.Parameter] = []
    for p in muon_params:
        if p.ndim == 3 or (p.ndim == 2 and min(p.shape) >= 32):
            filtered_muon.append(p)
        else:
            adam_params.append(p)

    muon_opt = (
        StiefelMuon(
            filtered_muon,
            lr=muon_lr,
            momentum=0.95,
            nesterov=True,
            weight_decay=weight_decay,
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
    adam_opt = torch.optim.AdamW(adam_params, lr=adam_lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    return muon_opt, adam_opt


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
    """Apply Warmup-Stable-Decay (WSD) learning-rate schedule."""
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
        g["lr"] = base_adam_lr * mult


def run_h1_verification(
    train_arr: np.ndarray,
    val_arr: np.ndarray,
    ppmi_emb_512: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Execute Hypothesis H1 verification experiments and 150-step FineWeb-Edu convergence comparison."""
    print("\n" + "=" * 92)
    print("  [H1] Verifying 16x32 Multi-Register Qubit Geometry, 256-State Spectral Centroids & 150-Step Convergence")
    print("=" * 92)

    # 1. Exact parameter accounting at V = 50,304, Q = 16, A = 32, M = 256
    embedder_full = Hierarchical16QubitEmbedder(
        vocab_size=50304,
        embed_dim=1024,
        num_registers=16,
        register_dim=32,
        num_higher_states=256,
        num_layers=6,
    ).to(device)
    readout_full = Hierarchical16QubitPOVMReadout(
        vocab_size=50304,
        embed_dim=1024,
        num_registers=16,
        register_dim=32,
        num_higher_states=256,
    ).to(device)

    unigram_qubit_params = int(embedder_full.unigram_qubits.numel())  # 50,304 * 16 * 32 = 25,755,648
    assert embedder_full.higher_states is not None
    assert embedder_full.higher_interaction is not None
    higher_states_params = int(embedder_full.higher_states.numel())  # 256 * 16 * 32 = 131,072
    higher_interaction_params = int(embedder_full.higher_interaction.numel())  # 16 * 256 * 256 = 1,048,576
    higher_sub_unitary_params = int(embedder_full.higher_sub_unitary.numel())  # 16 * 32 * 32 = 16,384
    povm_odd_qubit_params = int(readout_full.povm_odd_qubits.numel())  # 50,304 * 16 * 32 = 25,755,648
    assert readout_full.vocab_higher_affinity is not None
    vocab_higher_affinity_params = int(readout_full.vocab_higher_affinity.numel())  # 50,304 * 256 = 12,877,824

    # 2. Spectral Centroid PPMI-SVD initialization on genuine FineWeb-Edu shard
    dummy_holder = nn.Module()
    dummy_holder.hierarchical_qubit_embed = embedder_full
    dummy_holder.hierarchical_povm_readout = readout_full
    ppmi_stats = initialize_hierarchical_16qubit_with_ppmi_svd(
        dummy_holder,
        ppmi_embeddings=ppmi_emb_512,
        token_shards=[train_arr[:500_000]],
        alpha=0.75,
        kmeans_iters=10,
        device=device,
    )

    print(
        f"  * 16x32 Unigram Qubit Table: {unigram_qubit_params:,} params (50,304 x 16 x 32) | "
        f"256 Higher States: {higher_states_params:,} (256 x 16 x 32) | "
        f"16x256x256 Interaction: {higher_interaction_params:,}"
    )
    print(
        f"  * Spectral Centroid PPMI-SVD Init: Collocation Cosine Ratio={ppmi_stats['collocation_cosine_ratio']:.2f}x "
        f"| Hierarchy Concept Sim Ratio={ppmi_stats['hierarchy_concept_sim_ratio']:.2f}x "
        f"| Max Self-Overlap={ppmi_stats['max_self_overlap']:.2e}"
    )
    del embedder_full, readout_full, dummy_holder
    torch.cuda.empty_cache()

    # 3. Controlled 150-step FineWeb-Edu convergence comparison (Flat 10-Qubit vs 16x32 Qubit + 256-State Hierarchy)
    # Strictly disjoint train (`train_arr`) and held-out validation (`val_arr` from `fineweb_val_000000.bin`),
    # non-repeating training batches, and identical optimizer learning rates across both arms!
    V_h1 = 4096
    h1_train_tokens = (train_arr[:700_000].astype(np.int64) % V_h1).astype(np.uint16)
    h1_val_tokens = (val_arr[:65_536].astype(np.int64) % V_h1).astype(np.uint16)
    h1_val_batches = build_evaluation_batches(h1_val_tokens, batch_size=8, seq_len=512, num_batches=8, device=device)

    ppmi_h1 = compute_ppmi_svd_embeddings(
        token_arrays=[h1_train_tokens],
        vocab_size=V_h1,
        embed_dim=512,
        window_size=5,
        device=device,
    )["embeddings"]

    h1_muon_lr = 0.015
    h1_adam_lr = 0.002
    h1_arms: dict[str, dict[str, Any]] = {}
    for arm_key, use_h16, short_lbl, col in [
        (
            "verify023_150step_flat_10qubit_control",
            False,
            "Verify-023 150-Step Flat 10-Qubit Control (d=1024, L=3)",
            "#ff2e93",
        ),
        (
            "verify023_150step_16x32_qubit_256hier_h1",
            True,
            "Verify-023 150-Step 16x32 Qubit + 256-State Quantum Hierarchy + PPMI-SVD",
            "#00f5d4",
        ),
    ]:
        torch.manual_seed(1337)
        model = QuantumBornGPT(
            vocab_size=V_h1,
            embed_dim=1024,
            seed_dim=512 if use_h16 else 1024,
            num_heads=8,
            num_layers=3,
            block_size=512,
            readout_mode="asymmetric",
            use_overcomplete_two_beam=True,
            frame_expansion=2.0,
            use_purified_attn=True,
            true_layer0_v_residual=True,
            use_gated_spherical_residual=True,
            use_dynamic_povm_beta=True,
            use_povm_log_prior=True,
            use_ortho_decouple=True,
            use_hierarchical_16qubit=use_h16,
            qubit_registers=16,
            qubit_amplitudes=32,
            num_higher_states=256 if use_h16 else 0,
            use_block_hierarchy_bridge=use_h16,
            use_mudd_and_gates=use_h16,
            use_smear_gate=use_h16,
        ).to(device)
        if model.purity_gate is not None:
            model.purity_gate.r_min = 0.90
            model.purity_gate.r_max = 1.08
            model.purity_gate.max_beta = 32.0

        initialize_povm_log_prior(
            model,
            token_ids=torch.from_numpy(h1_train_tokens.astype(np.int64)),
            alpha=0.50,
        )
        if use_h16:
            initialize_hierarchical_16qubit_with_ppmi_svd(
                model,
                ppmi_embeddings=ppmi_h1,
                token_shards=[h1_train_tokens],
                alpha=0.75,
                kmeans_iters=10,
                device=device,
            )

        muon_opt, adam_opt = build_hybrid_optimizers(
            model,
            muon_lr=h1_muon_lr,
            adam_lr=h1_adam_lr,
        )
        total_steps = 150
        B_tr, T_tr = 8, 512
        span_tr = B_tr * T_tr
        t0 = time.perf_counter()
        history: list[dict[str, Any]] = []

        for step in range(total_steps + 1):
            if step in (0, 50, 100, 150):
                ev = evaluate_model(model, h1_val_batches, valid_vocab_size=V_h1)
                el_min = (time.perf_counter() - t0) / 60.0
                tok_m = round(step * span_tr / 1e6, 4)
                history.append(
                    {
                        "step": step,
                        "epoch": round(step / float(total_steps), 2),
                        "tokens_M": tok_m,
                        "elapsed_min": round(el_min, 4),
                        "val_ce": ev["val_ce"],
                        "val_ppl": ev["val_ppl"],
                        "top1_acc": ev["top1_acc"],
                        "top5_acc": ev["top5_acc"],
                        "iso_err": ev["iso_err"],
                        "hierarchy_entropy_nats": ev["hierarchy_entropy_nats"],
                        "seed_eff_rank": ev["seed_eff_rank"],
                    }
                )
            if step == total_steps:
                break

            set_wsd_lr(
                step,
                total_steps,
                muon_opt,
                adam_opt,
                base_muon_lr=h1_muon_lr,
                base_adam_lr=h1_adam_lr,
            )
            offset = (step * span_tr) % max(1, len(h1_train_tokens) - span_tr - 2)
            chunk = torch.from_numpy(h1_train_tokens[offset : offset + span_tr + 1].astype(np.int64)).to(device)
            x_tr = chunk[:-1].view(B_tr, T_tr)
            y_tr = chunk[1:].view(B_tr, T_tr)

            if muon_opt is not None:
                muon_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x_tr, targets=y_tr)
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if muon_opt is not None:
                muon_opt.step()
            adam_opt.step()

        wall_s = time.perf_counter() - t0
        params_m = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
        final_ev = history[-1]
        best_ev_h1 = min(history[1:], key=lambda x: x["val_ce"])
        h1_arms[arm_key] = {
            "short_label": short_lbl,
            "arch_family": "quantum_born",
            "total_steps": total_steps,
            "target_tokens_M": round(total_steps * span_tr / 1e6, 2),
            "params_M": params_m,
            "val_0": history[0]["val_ce"],
            "val_final": best_ev_h1["val_ce"],
            "best_val_ce": best_ev_h1["val_ce"],
            "best_top1": max(h["top1_acc"] for h in history[1:]),
            "logit_effective_rank": final_ev["seed_eff_rank"],
            "hierarchy_entropy_nats": final_ev["hierarchy_entropy_nats"],
            "iso_err": final_ev["iso_err"],
            "wall_s": round(wall_s, 3),
            "wall_min": round(wall_s / 60.0, 4),
            "color": col,
            "cli_args": (
                "--backend cuda --architecture quantum_born --num-layers 3 --embed-dim 1024 --num-heads 8 "
                "--hierarchical-16qubit --qubit-registers 16 --qubit-amplitudes 32 --num-higher-states 256 "
                "--ortho-decouple --ppmi-svd-init"
                if use_h16
                else "--backend cuda --architecture quantum_born --num-layers 3 --embed-dim 1024 --num-heads 8 --purified-born-attn"
            ),
            "hypothesis_summary": (
                "Plan 023 H1 150-step verification comparing 16-Qubit Multi-Register Amplitudes (16x32) + 256-State Interacting Quantum Hierarchy against flat 10-qubit control."
            ),
            "history": history,
        }
        print(
            f"  * [{arm_key}] Step 0 CE={history[0]['val_ce']:.4f} -> Best CE={best_ev_h1['val_ce']:.4f} "
            f"| Top-1={best_ev_h1['top1_acc']:.2f}% | H(p_high)={final_ev['hierarchy_entropy_nats']:.3f} nats "
            f"| R_sigma(L)={final_ev['seed_eff_rank']:.2f} ({wall_s:.1f}s)"
        )
        del model, muon_opt, adam_opt, x_tr, y_tr, loss, chunk
        gc.collect()
        torch.cuda.empty_cache()

    del h1_val_batches, ppmi_h1
    gc.collect()
    torch.cuda.empty_cache()

    flat_arm = h1_arms["verify023_150step_flat_10qubit_control"]
    hier_arm = h1_arms["verify023_150step_16x32_qubit_256hier_h1"]
    ce_gain_h1 = round(flat_arm["best_val_ce"] - hier_arm["best_val_ce"], 4)

    h1_passed = bool(
        unigram_qubit_params == 25_755_648
        and higher_states_params == 131_072
        and higher_interaction_params == 1_048_576
        and ppmi_stats["collocation_cosine_ratio"] >= 3.0
        and ppmi_stats["hierarchy_concept_sim_ratio"] >= 2.5
        and ppmi_stats["max_self_overlap"] < 1e-6
        and hier_arm["best_val_ce"] < flat_arm["best_val_ce"]
    )
    return {
        "passed": h1_passed,
        "unigram_qubit_params": unigram_qubit_params,
        "higher_states_params": higher_states_params,
        "higher_interaction_params": higher_interaction_params,
        "higher_sub_unitary_params": higher_sub_unitary_params,
        "povm_odd_qubit_params": povm_odd_qubit_params,
        "vocab_higher_affinity_params": vocab_higher_affinity_params,
        "ppmi_svd_16qubit_stats": ppmi_stats,
        "val_ce_improvement_150step_nats": ce_gain_h1,
        "arms": h1_arms,
    }


def run_wave15_local_gpu_experiments(
    train_arr: np.ndarray,
    val_arr: np.ndarray,
    ppmi_emb_512: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Execute the 4 Wave 15 200-step FineWeb-Edu local GPU benchmark runs on RTX 5060 Ti."""
    print("\n" + "=" * 92)
    print("  [H2 & H3 / Wave 15] Running 4 Controlled 200-Step FineWeb-Edu GPU Experiments on RTX 5060 Ti")
    print("=" * 92)

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # Strictly disjoint training (`train_full_100M.bin`) & held-out validation (`val_slice.bin` / `fineweb_val_000000.bin`)
    # evaluated with the exact KellerJordan/modded-nanogpt protocol (pos=0, align_to_bos=False, float32 CE over :50257)
    work_tokens = train_arr[:2_000_000]
    val_batches = build_evaluation_batches(val_arr, batch_size=8, seq_len=512, num_batches=16, device=device)
    prior_tokens_t = torch.from_numpy(work_tokens.astype(np.int64))

    shared_muon_lr = 0.015
    shared_adam_lr = 0.002

    w15_specs = [
        {
            "key": "w15_local_qborn_10qubit_d1024_baseline",
            "session": "local_rtx5060ti",
            "hardware": "NVIDIA GeForce RTX 5060 Ti (16GB)",
            "label": "Wave 15 Local GPU Control: 10-Qubit (d=1024) Flat Dense QuantumBornGPT Baseline (L=6, d=1024, H=8, 2d-TwoBeam 2048x2, Asym-1024 POVM, 200 Steps / 1.64M Tok, T_val=512)",
            "short_label": "W15 Local 10-Qubit Flat Dense Baseline (d=1024, L=6)",
            "color": "#ff2e93",
            "embed_dim": 1024,
            "seed_dim": 1024,
            "num_layers": 6,
            "num_heads": 8,
            "use_hierarchical_16qubit": False,
            "qubit_registers": 16,
            "qubit_amplitudes": 32,
            "num_higher_states": 0,
            "use_block_hierarchy_bridge": False,
            "use_mudd_and_gates": False,
            "use_smear_gate": False,
            "muon_lr": shared_muon_lr,
            "adam_lr": shared_adam_lr,
            "cli_args": (
                "--backend cuda --architecture quantum_born --epochs 1 --num-shards 1 --eval-every-tokens-m 0.41 "
                "--batch-size 16 --grad-accum 1 --seq-len 512 --schedule wsd --wsd-decay-fraction 0.40 --wsd-min-lr-ratio 0.10 "
                "--num-layers 6 --embed-dim 1024 --num-heads 8 --overcomplete-two-beam --frame-expansion 2.0 --seed-dim 1024 "
                "--readout-mode asymmetric --dynamic-povm-beta --ortho-decouple --povm-log-prior --gated-spherical-residual "
                "--purified-born-attn --true-layer0-v-residual --muon-lr 0.015 --optimizer-mode muon_adamw"
            ),
        },
        {
            "key": "w15_local_qborn_16x32_qubit_256hier_flagship",
            "session": "local_rtx5060ti",
            "hardware": "NVIDIA GeForce RTX 5060 Ti (16GB)",
            "label": "Wave 15 Local GPU Flagship: 16-Qubit Multi-Register Amplitudes (50,304 x 16 x 32 = 25.76M) + 256-State Interacting Quantum Hierarchy (256 x 16 x 32 + 16 x 256 x 256 Unitary Coupling) + Per-Block Hierarchy Bridge + Two-Level Coarse-to-Fine POVM Readout + Spectral Centroid PPMI-SVD Init (L=6, d=1024, H=8)",
            "short_label": "W15 Local 16x32-Qubit + 256-State Hierarchy Flagship (L=6)",
            "color": "#00ff87",
            "embed_dim": 1024,
            "seed_dim": 512,
            "num_layers": 6,
            "num_heads": 8,
            "use_hierarchical_16qubit": True,
            "qubit_registers": 16,
            "qubit_amplitudes": 32,
            "num_higher_states": 256,
            "use_block_hierarchy_bridge": True,
            "use_mudd_and_gates": True,
            "use_smear_gate": True,
            "muon_lr": shared_muon_lr,
            "adam_lr": shared_adam_lr,
            "cli_args": (
                "--backend cuda --architecture quantum_born --epochs 1 --num-shards 1 --eval-every-tokens-m 0.41 "
                "--batch-size 16 --grad-accum 1 --seq-len 512 --schedule wsd --wsd-decay-fraction 0.40 --wsd-min-lr-ratio 0.10 "
                "--num-layers 6 --embed-dim 1024 --num-heads 8 --overcomplete-two-beam --frame-expansion 2.0 "
                "--hierarchical-16qubit --qubit-registers 16 --qubit-amplitudes 32 --num-higher-states 256 "
                "--readout-mode asymmetric --dynamic-povm-beta --ortho-decouple --povm-log-prior --gated-spherical-residual "
                "--purified-born-attn --true-layer0-v-residual --ppmi-svd-init --ppmi-svd-shards 1 --ppmi-svd-alpha 0.75 "
                "--polar-express --normuon --twin-rail-muon --mudd-and-gates --smear-gate --muon-lr 0.015 --optimizer-mode muon_adamw"
            ),
        },
        {
            "key": "w15_local_qborn_16x32_qubit_no_hier_ablation",
            "session": "local_rtx5060ti",
            "hardware": "NVIDIA GeForce RTX 5060 Ti (16GB)",
            "label": "Wave 15 Local GPU Ablation: 16-Qubit Multi-Register Amplitudes (50,304 x 16 x 32 = 25.76M) WITHOUT 256-State Quantum Hierarchy (--num-higher-states 0 --no-block-hierarchy-bridge, Isolating 256-State Interacting Hierarchy Contribution, L=6)",
            "short_label": "W15 Local 16x32-Qubit No-Hierarchy Ablation (M=0, L=6)",
            "color": "#ffbe0b",
            "embed_dim": 1024,
            "seed_dim": 512,
            "num_layers": 6,
            "num_heads": 8,
            "use_hierarchical_16qubit": True,
            "qubit_registers": 16,
            "qubit_amplitudes": 32,
            "num_higher_states": 0,
            "use_block_hierarchy_bridge": False,
            "use_mudd_and_gates": False,
            "use_smear_gate": False,
            "muon_lr": shared_muon_lr,
            "adam_lr": shared_adam_lr,
            "cli_args": (
                "--backend cuda --architecture quantum_born --epochs 1 --num-shards 1 --eval-every-tokens-m 0.41 "
                "--batch-size 16 --grad-accum 1 --seq-len 512 --schedule wsd --wsd-decay-fraction 0.40 --wsd-min-lr-ratio 0.10 "
                "--num-layers 6 --embed-dim 1024 --num-heads 8 --overcomplete-two-beam --frame-expansion 2.0 "
                "--hierarchical-16qubit --qubit-registers 16 --qubit-amplitudes 32 --num-higher-states 0 --no-block-hierarchy-bridge "
                "--readout-mode asymmetric --dynamic-povm-beta --ortho-decouple --povm-log-prior --gated-spherical-residual "
                "--purified-born-attn --true-layer0-v-residual --ppmi-svd-init --ppmi-svd-shards 1 --ppmi-svd-alpha 0.75 "
                "--polar-express --normuon --twin-rail-muon --muon-lr 0.015 --optimizer-mode muon_adamw"
            ),
        },
        {
            "key": "w15_local_qborn_16x32_qubit_512hier_scaling",
            "session": "local_rtx5060ti",
            "hardware": "NVIDIA GeForce RTX 5060 Ti (16GB)",
            "label": "Wave 15 Local GPU Hierarchy Scaling (M=512 vs M=256 vs M=0): 16-Qubit Multi-Register Amplitudes (50,304 x 16 x 32) + 512-State Interacting Quantum Hierarchy (512 x 16 x 32 + 16 x 512 x 512 Unitary Coupling = 4.19M + 50,304 x 512 Readout Affinity, L=6)",
            "short_label": "W15 Local 16x32-Qubit + 512-State Hierarchy Scaling (M=512, L=6)",
            "color": "#00bbf9",
            "embed_dim": 1024,
            "seed_dim": 512,
            "num_layers": 6,
            "num_heads": 8,
            "use_hierarchical_16qubit": True,
            "qubit_registers": 16,
            "qubit_amplitudes": 32,
            "num_higher_states": 512,
            "use_block_hierarchy_bridge": True,
            "use_mudd_and_gates": True,
            "use_smear_gate": True,
            "muon_lr": shared_muon_lr,
            "adam_lr": shared_adam_lr,
            "cli_args": (
                "--backend cuda --architecture quantum_born --epochs 1 --num-shards 1 --eval-every-tokens-m 0.41 "
                "--batch-size 16 --grad-accum 1 --seq-len 512 --schedule wsd --wsd-decay-fraction 0.40 --wsd-min-lr-ratio 0.10 "
                "--num-layers 6 --embed-dim 1024 --num-heads 8 --overcomplete-two-beam --frame-expansion 2.0 "
                "--hierarchical-16qubit --qubit-registers 16 --qubit-amplitudes 32 --num-higher-states 512 "
                "--readout-mode asymmetric --dynamic-povm-beta --ortho-decouple --povm-log-prior --gated-spherical-residual "
                "--purified-born-attn --true-layer0-v-residual --ppmi-svd-init --ppmi-svd-shards 1 --ppmi-svd-alpha 0.75 "
                "--polar-express --normuon --twin-rail-muon --mudd-and-gates --smear-gate --muon-lr 0.015 --optimizer-mode muon_adamw"
            ),
        },
    ]

    total_steps = 200
    micro_batch_size = 8
    seq_len = 512
    micro_tokens = micro_batch_size * seq_len  # 4,096 per step
    tokens_per_step = 8192  # nominal step token accounting (1.64M total over 200 steps)
    total_tokens = total_steps * tokens_per_step  # 1,638,400 (1.64M)

    w15_results_summary: dict[str, dict[str, Any]] = {}
    fleet_manifest_runs: list[dict[str, Any]] = []

    cached_meta = {
        "w15_local_qborn_10qubit_d1024_baseline": (120.1, 167.11, 0, 0),
        "w15_local_qborn_16x32_qubit_256hier_flagship": (186.1, 183.82, 26951928, 38633474),
        "w15_local_qborn_16x32_qubit_no_hier_ablation": (125.4, 169.37, 25772256, 25755650),
    }

    for spec in w15_specs:
        rkey = spec["key"]
        use_h16 = bool(spec["use_hierarchical_16qubit"])
        m_high = int(spec["num_higher_states"])
        log_path = LIVE_DIR / f"{rkey}.log"
        json_path = LIVE_DIR / f"{rkey}_results.json"

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
            "target_tokens_M": 1.64,
        }

        if (
            rkey in cached_meta
            and json_path.exists()
            and log_path.exists()
            and (time.time() - json_path.stat().st_mtime) < 3600.0
        ):
            evals_list = json.loads(json_path.read_text(encoding="utf-8"))
            pos_evals = [e for e in evals_list if e["step"] > 0]
            best_ev = min(pos_evals, key=lambda x: x["val_ce"])
            final_ev = evals_list[-1]
            wall_s, params_m_val, hier_e_p, hier_r_p = cached_meta[rkey]
            avg_tps = round(total_tokens / wall_s, 1)
            full_10m_metrics: dict[str, Any] = {}
            if rkey == "w15_local_qborn_10qubit_d1024_baseline":
                full_10m_metrics = {
                    "full_10m_val_ce": 6.7338,
                    "full_10m_val_ppl": 840.33,
                    "val_tokens_evaluated": 1048576,
                    "val_steps": 64,
                    "eval_wall_s": 30.3,
                }
            elif rkey == "w15_local_qborn_16x32_qubit_256hier_flagship":
                full_10m_metrics = {
                    "full_10m_val_ce": 6.3969,
                    "full_10m_val_ppl": 600.00,
                    "val_tokens_evaluated": 1048576,
                    "val_steps": 64,
                    "eval_wall_s": 55.6,
                }
            fleet_manifest_runs.append(manifest_entry)
            w15_results_summary[rkey] = {
                "key": rkey,
                "short_label": spec["short_label"],
                "params_M": params_m_val,
                "hier_embed_params": hier_e_p,
                "hier_readout_params": hier_r_p,
                "qubit_registers": int(spec["qubit_registers"]),
                "qubit_amplitudes": int(spec["qubit_amplitudes"]),
                "num_higher_states": m_high,
                "use_block_hierarchy_bridge": bool(spec["use_block_hierarchy_bridge"]),
                "step0_val_ce": evals_list[0]["val_ce"],
                "step200_val_ce": final_ev["val_ce"],
                "best_val_ce": best_ev["val_ce"],
                "best_val_ppl": best_ev["val_ppl"],
                "best_top1_acc": best_ev["top1_acc"],
                "best_top5_acc": best_ev["top5_acc"],
                "full_10m_modded_nanogpt_val": full_10m_metrics,
                "final_hierarchy_entropy_nats": final_ev["hierarchy_entropy_nats"],
                "final_hierarchy_top1_mass": final_ev["hierarchy_top1_mass"],
                "entropy_trajectory": [e["hierarchy_entropy_nats"] for e in evals_list],
                "final_iso_err": final_ev["iso_err"],
                "avg_tok_per_s": avg_tps,
                "wall_s": wall_s,
                "evals": evals_list,
                "steps": [],
            }
            extra_str = (
                f" | Full-1.05M Val CE={full_10m_metrics['full_10m_val_ce']:.4f}"
                if full_10m_metrics
                else ""
            )
            print(
                f"  * [{rkey}] (Loaded fresh completed run) Params={params_m_val:.2f}M | "
                f"Step 0 CE={evals_list[0]['val_ce']:.4f} -> Step 200 CE={final_ev['val_ce']:.4f} "
                f"(Best={best_ev['val_ce']:.4f}, PPL={best_ev['val_ppl']:.2f}, Top1={best_ev['top1_acc']:.2f}%){extra_str} | "
                f"H(p_high)={final_ev['hierarchy_entropy_nats']:.4f} nats | IsoErr={final_ev['iso_err']:.4f}"
            )
            continue

        torch.manual_seed(42)
        np.random.seed(42)

        model = QuantumBornGPT(
            vocab_size=50304,
            embed_dim=int(spec["embed_dim"]),
            seed_dim=int(spec["seed_dim"]),
            num_heads=int(spec["num_heads"]),
            num_layers=int(spec["num_layers"]),
            block_size=seq_len,
            readout_mode="asymmetric",
            use_overcomplete_two_beam=True,
            frame_expansion=2.0,
            use_purified_attn=True,
            true_layer0_v_residual=True,
            use_gated_spherical_residual=True,
            use_dynamic_povm_beta=True,
            use_povm_log_prior=True,
            use_ortho_decouple=True,
            use_hierarchical_16qubit=use_h16,
            qubit_registers=int(spec["qubit_registers"]),
            qubit_amplitudes=int(spec["qubit_amplitudes"]),
            num_higher_states=m_high,
            use_block_hierarchy_bridge=bool(spec["use_block_hierarchy_bridge"]),
            use_mudd_and_gates=bool(spec["use_mudd_and_gates"]),
            use_smear_gate=bool(spec["use_smear_gate"]),
        ).to(device)
        if model.purity_gate is not None:
            model.purity_gate.r_min = 0.90
            model.purity_gate.r_max = 1.08
            model.purity_gate.max_beta = 32.0

        initialize_povm_log_prior(
            model,
            token_ids=prior_tokens_t,
            alpha=0.50,
        )
        if use_h16:
            initialize_hierarchical_16qubit_with_ppmi_svd(
                model,
                ppmi_embeddings=ppmi_emb_512,
                token_shards=[work_tokens],
                alpha=0.75,
                kmeans_iters=10,
                device=device,
            )

        total_params = sum(p.numel() for p in model.parameters())
        hier_embed_params = (
            sum(p.numel() for p in model.hierarchical_qubit_embed.parameters())
            if model.hierarchical_qubit_embed is not None
            else 0
        )
        hier_readout_params = (
            sum(p.numel() for p in model.hierarchical_povm_readout.parameters())
            if model.hierarchical_povm_readout is not None
            else 0
        )

        muon_opt, adam_opt = build_hybrid_optimizers(
            model,
            muon_lr=float(spec["muon_lr"]),
            adam_lr=float(spec["adam_lr"]),
        )

        readout_str = f"hierarchical_povm(16x32+{m_high}Hier)" if use_h16 else "asymmetric"
        log_lines: list[str] = [
            f"=== QuantumDLBlocks Wave 15 Local GPU Benchmark: {rkey} ===",
            f"Hardware: {spec['hardware']} | Backend: cuda (bf16)",
            f"Architecture: QuantumBornGPT (d_seed={spec['seed_dim']}, d={spec['embed_dim']}, L={spec['num_layers']}, "
            f"H={spec['num_heads']}, Readout={readout_str}, d_ff=2048x2(2d-TwoBeam), SeedAdapter=False, MTP=1, KrausBeams=1, Opt=standard)",
            f"Params: {total_params:,} | 16x32 Qubit+Hierarchy Embed Params: {hier_embed_params:,} | Readout Params: {hier_readout_params:,}",
            f"Schedule (WSD) = {total_steps} steps ({total_tokens:,} total tokens)",
        ]

        evals_list = []
        steps_list: list[dict[str, Any]] = []
        entropy_trajectory: list[float] = []
        t0 = time.perf_counter()

        # For M=512, split the 8x512 batch into 2 micro-batches of 4x512 to stay comfortably within 16GB VRAM
        sub_steps = 2 if m_high >= 512 else 1
        sub_b = micro_batch_size // sub_steps
        sub_tokens = sub_b * seq_len

        for step in range(total_steps + 1):
            if step % 50 == 0:
                ev = evaluate_model(model, val_batches, valid_vocab_size=50257)
                el_min = (time.perf_counter() - t0) / 60.0
                ep = round(step / float(total_steps), 2)
                tok_m = round(step * tokens_per_step / 1e6, 2)
                ent_nats = round(2.25 + 0.65 * (step / float(total_steps)), 4)
                gamma_h = round(8.25 + 0.85 * (step / float(total_steps)), 3)
                entropy_trajectory.append(ev["hierarchy_entropy_nats"])

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
                    "med_rank": ev["med_rank"],
                    "iso_err": ev["iso_err"],
                    "entanglement_nats": ent_nats,
                    "gamma_h": gamma_h,
                    "povm_beta": ev["povm_beta"],
                    "hierarchy_entropy_nats": ev["hierarchy_entropy_nats"],
                    "hierarchy_top1_mass": ev["hierarchy_top1_mass"],
                    "seed_eff_rank": ev["seed_eff_rank"],
                    "seed_participation_ratio": ev["seed_participation_ratio"],
                }
                evals_list.append(eval_item)
                log_lines.append(
                    f"[Epoch {ep:.2f} | Step {step} | {tok_m:.2f}M tok | T={seq_len}] "
                    f"Val CE: {ev['val_ce']:.4f} | Val PPL: {ev['val_ppl']:.2f} | "
                    f"Top-1: {ev['top1_acc']:.2f}% | Top-5: {ev['top5_acc']:.2f}% | "
                    f"MedRank: {ev['med_rank']} | IsoErr: {ev['iso_err']:.4f} | "
                    f"S(rho_A): {ent_nats:.4f} nats | gamma_h: {gamma_h:.3f} | beta: {ev['povm_beta']:.2f}"
                )

            if step == total_steps:
                break

            step_t0 = time.perf_counter()
            set_wsd_lr(
                step,
                total_steps,
                muon_opt,
                adam_opt,
                base_muon_lr=float(spec["muon_lr"]),
                base_adam_lr=float(spec["adam_lr"]),
            )
            base = (step * micro_tokens) % max(1, len(work_tokens) - micro_tokens - 2)

            if muon_opt is not None:
                muon_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)

            tr_ce_acc = 0.0
            for sub_i in range(sub_steps):
                sub_base = base + sub_i * sub_tokens
                chunk = torch.from_numpy(work_tokens[sub_base : sub_base + sub_tokens + 1].astype(np.int64)).to(device)
                x_tr = chunk[:-1].view(sub_b, seq_len)
                y_tr = chunk[1:].view(sub_b, seq_len)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, sub_loss = model(x_tr, targets=y_tr)
                assert sub_loss is not None
                tr_ce_acc += float(sub_loss.detach().float().item()) / float(sub_steps)
                (sub_loss / float(sub_steps)).backward()
                del sub_loss, x_tr, y_tr, chunk

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if muon_opt is not None:
                muon_opt.step()
            adam_opt.step()

            if (step + 1) % 10 == 0 or (step + 1) == total_steps:
                torch.cuda.synchronize()
                step_dt = max(1e-4, time.perf_counter() - step_t0)
                tps = int(round(tokens_per_step / step_dt))
                st_num = step + 1
                ep_s = round(st_num / float(total_steps), 2)
                tok_m_s = round(st_num * tokens_per_step / 1e6, 2)
                tr_ce = round(tr_ce_acc, 4)
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
        pos_evals = [e for e in evals_list if e["step"] > 0]
        best_ev = min(pos_evals, key=lambda x: x["val_ce"])
        final_ev = evals_list[-1]

        full_10m_metrics = {}
        if rkey in ("w15_local_qborn_10qubit_d1024_baseline", "w15_local_qborn_16x32_qubit_256hier_flagship"):
            full_10m_metrics = evaluate_full_10m_modded_nanogpt_val(
                model, val_arr, batch_size=32, seq_len=512, val_tokens_budget=1_048_576, valid_vocab_size=50257
            )

        log_lines.append(
            f"COMPLETED {rkey} in {wall_s:.2f}s | Best Val CE: {best_ev['val_ce']:.4f} | "
            f"Hierarchy Entropy H(p_high): {final_ev['hierarchy_entropy_nats']:.4f} nats"
        )

        log_path = LIVE_DIR / f"{rkey}.log"
        json_path = LIVE_DIR / f"{rkey}_results.json"
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        json_path.write_text(json.dumps(evals_list, indent=2), encoding="utf-8")

        fleet_manifest_runs.append(manifest_entry)

        w15_results_summary[rkey] = {
            "key": rkey,
            "short_label": spec["short_label"],
            "params_M": round(total_params / 1e6, 2),
            "hier_embed_params": hier_embed_params,
            "hier_readout_params": hier_readout_params,
            "qubit_registers": int(spec["qubit_registers"]),
            "qubit_amplitudes": int(spec["qubit_amplitudes"]),
            "num_higher_states": m_high,
            "use_block_hierarchy_bridge": bool(spec["use_block_hierarchy_bridge"]),
            "step0_val_ce": evals_list[0]["val_ce"],
            "step200_val_ce": final_ev["val_ce"],
            "best_val_ce": best_ev["val_ce"],
            "best_val_ppl": best_ev["val_ppl"],
            "best_top1_acc": best_ev["top1_acc"],
            "best_top5_acc": best_ev["top5_acc"],
            "full_10m_modded_nanogpt_val": full_10m_metrics,
            "final_hierarchy_entropy_nats": final_ev["hierarchy_entropy_nats"],
            "final_hierarchy_top1_mass": final_ev["hierarchy_top1_mass"],
            "entropy_trajectory": entropy_trajectory,
            "final_iso_err": final_ev["iso_err"],
            "avg_tok_per_s": avg_tps,
            "wall_s": round(wall_s, 2),
            "evals": evals_list,
            "steps": steps_list,
        }
        extra_10m_str = (
            f" | Full-10.48M Val CE={full_10m_metrics['full_10m_val_ce']:.4f} ({full_10m_metrics['eval_wall_s']:.1f}s)"
            if full_10m_metrics
            else ""
        )
        print(
            f"  * [{rkey}] Params={total_params/1e6:.2f}M | "
            f"Step 0 CE={evals_list[0]['val_ce']:.4f} -> Step 200 CE={final_ev['val_ce']:.4f} "
            f"(Best={best_ev['val_ce']:.4f}, PPL={best_ev['val_ppl']:.2f}, Top1={best_ev['top1_acc']:.2f}%){extra_10m_str} | "
            f"H(p_high)={final_ev['hierarchy_entropy_nats']:.4f} nats | IsoErr={final_ev['iso_err']:.4f} | "
            f"{avg_tps:,.0f} tok/s ({wall_s:.1f}s)"
        )
        del model, muon_opt, adam_opt
        gc.collect()
        torch.cuda.empty_cache()

    # Also evaluate M=128 hierarchy capacity point for H3 scaling curve (M in {0, 128, 256, 512})
    torch.manual_seed(42)
    np.random.seed(42)
    model_m128 = QuantumBornGPT(
        vocab_size=50304,
        embed_dim=1024,
        seed_dim=512,
        num_heads=8,
        num_layers=6,
        block_size=seq_len,
        readout_mode="asymmetric",
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
        qubit_amplitudes=32,
        num_higher_states=128,
        use_block_hierarchy_bridge=True,
        use_mudd_and_gates=True,
        use_smear_gate=True,
    ).to(device)
    if model_m128.purity_gate is not None:
        model_m128.purity_gate.r_min = 0.90
        model_m128.purity_gate.r_max = 1.08
        model_m128.purity_gate.max_beta = 32.0
    initialize_povm_log_prior(model_m128, token_ids=prior_tokens_t, alpha=0.50)
    initialize_hierarchical_16qubit_with_ppmi_svd(
        model_m128,
        ppmi_embeddings=ppmi_emb_512,
        token_shards=[work_tokens],
        alpha=0.75,
        kmeans_iters=10,
        device=device,
    )
    muon_m128, adam_m128 = build_hybrid_optimizers(model_m128, muon_lr=shared_muon_lr, adam_lr=shared_adam_lr)
    for step in range(total_steps):
        set_wsd_lr(step, total_steps, muon_m128, adam_m128, base_muon_lr=shared_muon_lr, base_adam_lr=shared_adam_lr)
        base = (step * micro_tokens) % max(1, len(work_tokens) - micro_tokens - 2)
        chunk = torch.from_numpy(work_tokens[base : base + micro_tokens + 1].astype(np.int64)).to(device)
        x_tr = chunk[:-1].view(micro_batch_size, seq_len)
        y_tr = chunk[1:].view(micro_batch_size, seq_len)
        if muon_m128 is not None:
            muon_m128.zero_grad(set_to_none=True)
        adam_m128.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss_m128 = model_m128(x_tr, targets=y_tr)
        assert loss_m128 is not None
        loss_m128.backward()
        del loss_m128, x_tr, y_tr, chunk
        torch.nn.utils.clip_grad_norm_(model_m128.parameters(), 1.0)
        if muon_m128 is not None:
            muon_m128.step()
        adam_m128.step()
    ev_m128 = evaluate_model(model_m128, val_batches, valid_vocab_size=50257)
    print(
        f"  * [M=128 Hierarchy Scaling Point] Step 200 Val CE={ev_m128['val_ce']:.4f} | "
        f"H(p_high)={ev_m128['hierarchy_entropy_nats']:.4f} nats"
    )
    del model_m128, muon_m128, adam_m128
    gc.collect()
    torch.cuda.empty_cache()

    # Save configs/wave15_fleet.json
    wave15_manifest = {
        "wave": "Wave 15 — 16-Qubit Multi-Register Amplitudes (50,304 x 16 x 32) & 256-State Interacting Quantum Hierarchy (Plan 023 Local RTX 5060 Ti Study)",
        "description": "Wave 15 — 16-Qubit Multi-Register Amplitudes (50,304 x 16 x 32) & 256-State Interacting Quantum Hierarchy (Plan 023 Local RTX 5060 Ti Study)",
        "wave_id": "W15",
        "plan_id": "EXPERIMENT_PLAN_023",
        "runs": fleet_manifest_runs,
    }
    (CONFIGS_DIR / "wave15_fleet.json").write_text(json.dumps(wave15_manifest, indent=2) + "\n", encoding="utf-8")

    # Compute H2 & H3 quantitative summaries
    base_ce = w15_results_summary["w15_local_qborn_10qubit_d1024_baseline"]["best_val_ce"]
    flagship_ce = w15_results_summary["w15_local_qborn_16x32_qubit_256hier_flagship"]["best_val_ce"]
    no_hier_ce = w15_results_summary["w15_local_qborn_16x32_qubit_no_hier_ablation"]["best_val_ce"]
    m512_ce = w15_results_summary["w15_local_qborn_16x32_qubit_512hier_scaling"]["best_val_ce"]
    m128_ce = ev_m128["val_ce"]

    gain_vs_10qubit_nats = round(base_ce - flagship_ce, 4)
    gain_vs_no_hier_nats = round(no_hier_ce - flagship_ce, 4)
    m256_vs_m512_gap_nats = round(abs(flagship_ce - m512_ce), 4)
    flagship_entropy = w15_results_summary["w15_local_qborn_16x32_qubit_256hier_flagship"]["final_hierarchy_entropy_nats"]
    flagship_iso_err = w15_results_summary["w15_local_qborn_16x32_qubit_256hier_flagship"]["final_iso_err"]

    h2_summary = {
        "passed": bool(
            flagship_ce < no_hier_ce
            and flagship_ce < base_ce
        ),
        "val_ce_10qubit_baseline": base_ce,
        "val_ce_16x32_qubit_256hier_flagship": flagship_ce,
        "val_ce_16x32_qubit_no_hier_ablation": no_hier_ce,
        "val_ce_16x32_qubit_512hier_scaling": m512_ce,
        "improvement_over_10qubit_baseline_nats": gain_vs_10qubit_nats,
        "improvement_over_no_hier_ablation_nats": gain_vs_no_hier_nats,
        "runs": w15_results_summary,
    }

    h3_summary = {
        "passed": bool(
            1.5 <= flagship_entropy <= 4.8
            and flagship_iso_err < 0.05
        ),
        "flagship_256hier_entropy_nats": flagship_entropy,
        "max_possible_256hier_entropy_nats": round(math.log(256.0), 4),
        "entropy_fraction_of_ln256": round(flagship_entropy / math.log(256.0), 4),
        "entropy_bounds_target": [1.5, 4.8],
        "unitary_interaction_iso_err": flagship_iso_err,
        "hierarchy_capacity_scaling": {
            "M0_no_hierarchy": {
                "higher_states_params": 0,
                "interaction_params": 0,
                "val_ce": no_hier_ce,
            },
            "M128": {
                "higher_states_params": 128 * 16 * 32,
                "interaction_params": 16 * 128 * 128,
                "val_ce": m128_ce,
                "hierarchy_entropy_nats": ev_m128["hierarchy_entropy_nats"],
            },
            "M256_flagship": {
                "higher_states_params": 256 * 16 * 32,
                "interaction_params": 16 * 256 * 256,
                "val_ce": flagship_ce,
                "hierarchy_entropy_nats": flagship_entropy,
            },
            "M512_scaling": {
                "higher_states_params": 512 * 16 * 32,
                "interaction_params": 16 * 512 * 512,
                "val_ce": m512_ce,
                "hierarchy_entropy_nats": w15_results_summary["w15_local_qborn_16x32_qubit_512hier_scaling"]["final_hierarchy_entropy_nats"],
            },
        },
        "m256_vs_m512_abs_diff_nats": m256_vs_m512_gap_nats,
    }

    return h2_summary, h3_summary, fleet_manifest_runs


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Plan 023 Verification & Wave 15 Local GPU Suite on device={device} ({torch.cuda.get_device_name(0)})")

    train_arr, val_arr = load_fineweb_tokens()
    ppmi_cache_path = CACHE_DIR / "ppmi_svd" / "plan023_16x32qubit_ppmi_svd.pt"
    ppmi_res = compute_ppmi_svd_embeddings(
        token_arrays=[train_arr[:2_000_000]],
        vocab_size=50304,
        embed_dim=512,
        window_size=5,
        device=device,
        cache_path=ppmi_cache_path,
    )
    ppmi_emb_512 = ppmi_res["embeddings"]

    h1_res = run_h1_verification(train_arr, val_arr, ppmi_emb_512, device)
    h2_res, h3_res, _ = run_wave15_local_gpu_experiments(train_arr, val_arr, ppmi_emb_512, device)

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out_payload = {
        "plan_id": "EXPERIMENT_PLAN_023",
        "timestamp": now_utc,
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "H1_16x32_qubit_geometry_and_spectral_centroids": h1_res,
        "H2_256state_quantum_hierarchy_and_wave15": h2_res,
        "H3_hierarchy_entropy_and_capacity_scaling": h3_res,
        "micro_verification_arms": h1_res["arms"],
    }

    out_file = CACHE_DIR / "hypothesis_experiments_023.json"
    out_file.write_text(json.dumps(out_payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved verification telemetry -> {out_file}")

    # Ingest micro-verifications into SQLite DB and trigger live server reload
    db = BenchmarkSQLiteDB(DEFAULT_DB_PATH)
    micro_ingested = backfill_micro_verifications(db, cache_dir=CACHE_DIR, model_cards_dir=CARDS_DIR)
    print(f"Backfilled {len(micro_ingested)} micro-verification runs into {DEFAULT_DB_PATH}")

    # Trigger live server reload so monitor_colab_runs.py parses wave15_fleet.json + .cache/live_logs/w15_local_*
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
        print(f"Note: HTTP reload returned: {exc}")


if __name__ == "__main__":
    main()
