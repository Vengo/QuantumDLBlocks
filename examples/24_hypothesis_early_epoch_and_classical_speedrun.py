#!/usr/bin/env python3
"""Controlled Hypothesis Verification Script for Experiment Plan 014 (`examples/24`).

Tests Hypotheses H1, H2, and H3 on genuine FineWeb tokens (`fineweb_val_000000.bin`):
- H1: Eliminating Step-0 Self-Repetition & Accelerating Early-Epoch Quantum Convergence
  (`baseline_quantum` vs. `earlyfix_quantum` over 200 steps at B=16, T=256).
- H2: Non-Zero Gradient Flow & Monotonic Sharpness of `CoherenceThresholdedBornAttention`.
- H3: Closing the Classical Modded-NanoGPT Single-GPU Optimization Gap
  (`wave6_classical_baseline` vs. `speedrun_classical_upgraded` over 200 steps at B=16, T=256).

Writes results to `.cache/hypothesis_experiments_014.json`.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from qdlblocks.benchmarks.fineweb_stream import (
    FineWebMultiShardStream,
    generate_contextual_zipf_shard,
)
from qdlblocks.layers.attention import SubspaceProjectorBornAttention
from qdlblocks.layers.early_epoch_quantum import (
    compute_born_attention_snr_metrics,
    compute_self_repetition_metrics,
    purified_born_attention_weights,
)
from qdlblocks.models.born_gpt import QuantumBornGPT
from qdlblocks.models.classical_modded_gpt import ClassicalModdedGPT
from qdlblocks.optim.muon import StiefelMuon, split_muon_adamw_params
from qdlblocks.optim.schedules import cosine_lr_multiplier


def load_fineweb_tokens(
    num_train_tokens: int = 1_000_000,
    num_val_tokens: int = 131_072,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Load genuine FineWeb tokens from `.cache/fineweb10B/fineweb_val_000000.bin`."""
    data_dir = REPO_ROOT / ".cache" / "fineweb10B"
    try:
        stream = FineWebMultiShardStream(
            data_dir=data_dir,
            num_shards=1,
            prefetch_async=False,
        )
        val_np = stream.ensure_val_shard()
        train_slice = torch.from_numpy(np.array(val_np[:num_train_tokens], dtype=np.int64))
        val_slice = torch.from_numpy(
            np.array(val_np[num_train_tokens : num_train_tokens + num_val_tokens], dtype=np.int64)
        )
        return train_slice, val_slice, "genuine_fineweb_val_000000.bin"
    except Exception:
        total = num_train_tokens + num_val_tokens
        synth_np = generate_contextual_zipf_shard(num_tokens=total, vocab_size=50304, seed=42)
        train_slice = torch.from_numpy(np.array(synth_np[:num_train_tokens], dtype=np.int64))
        val_slice = torch.from_numpy(np.array(synth_np[num_train_tokens:total], dtype=np.int64))
        return train_slice, val_slice, "synthetic_contextual_zipf_fallback"


@torch.no_grad()
def compute_model_iso_err(model: nn.Module, include_zero_init: bool = False) -> float:
    """Compute mean Stiefel isometry error across 2D backbone matrices."""
    errs: list[float] = []
    for name, p in model.named_parameters():
        if p.ndim == 2 and not any(
            k in name
            for k in (
                "wte",
                "povm_head",
                "lm_head",
                "purity_gate",
                "embed",
                "unet_scales",
                "smear",
                "beam_",
            )
        ):
            if not include_zero_init and any(z in name for z in ("out_proj", "fc2")):
                continue
            w = p.detach().float()
            r, c = w.shape
            if r >= c:
                gram = w.t() @ w
                eye = torch.eye(c, device=w.device, dtype=torch.float32)
                err = torch.norm(gram - eye, p="fro") / math.sqrt(c)
            else:
                gram = w @ w.t()
                eye = torch.eye(r, device=w.device, dtype=torch.float32)
                err = torch.norm(gram - eye, p="fro") / math.sqrt(r)
            errs.append(float(err.item()))
    return float(np.mean(errs)) if errs else 0.0


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    val_tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate validation CE, top-1 accuracy, and self-repetition metrics on FineWeb validation slice."""
    model.eval()
    stride = batch_size * seq_len
    n_batches = min(8, (len(val_tokens) - 1) // stride)
    losses: list[float] = []
    top1_hits = 0
    total_toks = 0
    rep_metrics_b0: dict[str, float] = {}

    for i in range(n_batches):
        chunk = val_tokens[i * stride : (i + 1) * stride + 1].to(device)
        x = chunk[:-1].reshape(batch_size, seq_len)
        y = chunk[1:].reshape(batch_size, seq_len)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits, loss = model(x, targets=y)
            assert loss is not None
            if i == 0:
                rep_metrics_b0 = compute_self_repetition_metrics(logits, x)
        losses.append(float(loss.item()))
        top1_hits += int((logits.argmax(dim=-1) == y).sum().item())
        total_toks += int(y.numel())

    model.train()
    return {
        "val_ce": round(float(np.mean(losses)), 4),
        "val_ppl": round(float(math.exp(min(20.0, float(np.mean(losses))))), 2),
        "top1_acc": round(100.0 * top1_hits / max(1, total_toks), 2),
        "mean_self_repeat_prob": round(float(rep_metrics_b0.get("mean_self_repeat_prob", 0.0)), 6),
        "top1_self_repeat_rate": round(float(rep_metrics_b0.get("top1_self_repeat_rate", 0.0)), 6),
        "mean_entropy_nats": round(float(rep_metrics_b0.get("mean_entropy_nats", 0.0)), 4),
    }


def run_h1_quantum_early_epoch_experiment(
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    device: torch.device,
    steps: int = 200,
    batch_size: int = 16,
    seq_len: int = 256,
) -> dict[str, Any]:
    """Run Experiment H1: baseline_quantum vs. earlyfix_quantum on FineWeb tokens."""
    print("\n" + "=" * 88)
    print("EXPERIMENT H1: Early-Epoch Quantum Convergence & Self-Repetition Elimination")
    print("=" * 88)

    # First evaluate attention SNR metrics at (B=2, H=4, T=512, d_h=64)
    torch.manual_seed(42)
    q_snr = F.normalize(torch.randn(2, 4, 512, 64, device=device), p=2, dim=-1)
    k_snr = F.normalize(torch.randn(2, 4, 512, 64, device=device), p=2, dim=-1)
    snr_metrics = compute_born_attention_snr_metrics(
        q_snr,
        k_snr,
        gamma_h=8.0,
        tau_h=0.0625,
        eps_base=0.05,
        target_overlap=0.85,
    )
    print(
        f"  [H1 SNR @ T=512, d_h=64] std_target_weight={snr_metrics['std_target_weight']:.4f} | "
        f"purified_target_weight={snr_metrics['purified_target_weight']:.4f} | "
        f"snr_gain_ratio={snr_metrics['snr_gain_ratio']:.2f}x"
    )

    arms = {
        "baseline_quantum": dict(
            use_ortho_decouple=False,
            use_povm_log_prior=False,
            use_gated_spherical_residual=False,
            use_purified_attn=False,
            use_dynamic_povm_beta=False,
        ),
        "earlyfix_quantum": dict(
            use_ortho_decouple=True,
            use_povm_log_prior=True,
            use_gated_spherical_residual=True,
            use_purified_attn=True,
            use_dynamic_povm_beta=True,
        ),
    }

    arm_results: dict[str, Any] = {}
    stride = batch_size * seq_len
    max_batches = (len(train_tokens) - 1) // stride

    for arm_name, flags in arms.items():
        torch.manual_seed(42)
        model = QuantumBornGPT(
            vocab_size=50304,
            embed_dim=384,
            seed_dim=192,
            num_heads=6,
            num_layers=6,
            hidden_dim=768,
            block_size=seq_len,
            readout_mode="tied_stiefel",
            learnable_temperature=True,
            value_residual_mixing=True,
            input_highway_mixing=True,
            use_bigram_smear=True,
            use_unet_interferometry=True,
            use_flash_born=False,
            **flags,
        ).to(device)

        adamw_pats = (
            "wte",
            "povm_head",
            "lm_head",
            "purity_gate",
            "highway",
            "v_mix",
            "embed",
            "unet_scales",
            "smear",
            "inv_temp",
            "kraus_readout",
            "seed_gate",
            "povm_log_prior",
            "coherence_tau",
            "res_gate",
        )
        muon_params, adamw_params = split_muon_adamw_params(model, adamw_name_patterns=adamw_pats)
        base_muon_lr = 0.025
        base_adamw_lr = 0.004

        opt_muon = StiefelMuon(
            muon_params,
            lr=base_muon_lr,
            momentum=0.95,
            nesterov=True,
            weight_decay=0.01,
            decay_mode="cautious",
            stiefel_penalty=0.005,
            retract_every=4,
        )
        adamw_2d = [p for p in adamw_params if p.ndim == 2]
        adamw_non2d = [p for p in adamw_params if p.ndim != 2]
        adamw_groups = []
        if adamw_2d:
            adamw_groups.append({"params": adamw_2d, "lr": base_adamw_lr, "base_lr": base_adamw_lr, "weight_decay": 0.01})
        if adamw_non2d:
            adamw_groups.append(
                {"params": adamw_non2d, "lr": base_adamw_lr, "base_lr": base_adamw_lr, "weight_decay": 0.0}
            )
        opt_adamw = torch.optim.AdamW(adamw_groups, lr=base_adamw_lr, betas=(0.9, 0.95))

        history: list[dict[str, float]] = []
        val0 = evaluate_model(model, val_tokens, batch_size, seq_len, device)
        val0["step"] = 0
        val0["iso_err"] = round(compute_model_iso_err(model, include_zero_init=True), 4)
        history.append(val0)

        t0 = time.perf_counter()
        for step in range(1, steps + 1):
            lr_mult = cosine_lr_multiplier(step, steps, warmup_steps=20, min_lr_ratio=0.15)
            for g in opt_muon.param_groups:
                g["lr"] = base_muon_lr * lr_mult
            for g in opt_adamw.param_groups:
                g["lr"] = float(g["base_lr"]) * lr_mult

            b_idx = (step - 1) % max_batches
            chunk = train_tokens[b_idx * stride : (b_idx + 1) * stride + 1].to(device)
            x = chunk[:-1].reshape(batch_size, seq_len)
            y = chunk[1:].reshape(batch_size, seq_len)

            opt_muon.zero_grad(set_to_none=True)
            opt_adamw.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                _, loss = model(x, targets=y)
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_muon.step()
            opt_adamw.step()

            if step in (50, 100, 150, steps):
                vk = evaluate_model(model, val_tokens, batch_size, seq_len, device)
                vk["step"] = step
                vk["iso_err"] = round(compute_model_iso_err(model, include_zero_init=True), 4)
                history.append(vk)

        elapsed = round(time.perf_counter() - t0, 2)
        ck0 = history[0]
        ck_end = history[-1]
        print(
            f"  [{arm_name:<18}] Step 0 CE={ck0['val_ce']:.4f} (SelfRep={ck0['mean_self_repeat_prob']*100:.2f}%) -> "
            f"Step {steps} CE={ck_end['val_ce']:.4f} | Top1={ck_end['top1_acc']:.2f}% | "
            f"IsoErr={ck_end['iso_err']:.4f} ({elapsed:.1f}s)"
        )
        arm_results[arm_name] = {
            "history": history,
            "step0_val_ce": ck0["val_ce"],
            "step0_self_repeat_prob": ck0["mean_self_repeat_prob"],
            "step200_val_ce": ck_end["val_ce"],
            "step200_top1_acc": ck_end["top1_acc"],
            "step200_iso_err": ck_end["iso_err"],
            "elapsed_sec": elapsed,
        }

    base_rep0 = arm_results["baseline_quantum"]["step0_self_repeat_prob"]
    fix_rep0 = arm_results["earlyfix_quantum"]["step0_self_repeat_prob"]
    val_ce_gain = round(
        arm_results["baseline_quantum"]["step200_val_ce"] - arm_results["earlyfix_quantum"]["step200_val_ce"],
        4,
    )
    h1_pass = bool(
        base_rep0 > 0.20
        and fix_rep0 < 0.005
        and snr_metrics["std_target_weight"] < 0.18
        and snr_metrics["purified_target_weight"] >= 0.65
        and snr_metrics["snr_gain_ratio"] >= 3.5
        and val_ce_gain >= 0.20
    )
    print(f"  [H1 Summary] val_ce_gain={val_ce_gain:+.4f} nats | H1 PASS = {h1_pass}")
    return {
        "snr_metrics": snr_metrics,
        "arms": arm_results,
        "val_ce_gain_nats": val_ce_gain,
        "passed": h1_pass,
    }


def run_h2_purified_attention_gradient_and_entropy(device: torch.device) -> dict[str, Any]:
    """Run Experiment H2: Gradient Flow & Monotonic Attention Entropy Reduction."""
    print("\n" + "=" * 88)
    print("EXPERIMENT H2: Purified Born Attention Gradient Flow & Entropy Reduction")
    print("=" * 88)

    torch.manual_seed(42)
    b, t, d, h = 4, 128, 256, 4
    x = F.normalize(torch.randn(b, t, d, device=device), p=2, dim=-1)
    target = torch.randn(b, t, d, device=device)

    torch.manual_seed(7)
    attn_std = SubspaceProjectorBornAttention(
        embed_dim=d,
        num_heads=h,
        block_size=t,
        use_flash_born=False,
        use_purified_attn=False,
    ).to(device)
    out_std = attn_std(x)
    loss_std = (out_std * target).sum(dim=-1).mean()
    loss_std.backward()
    assert attn_std.qk_gain_param is not None and attn_std.qk_gain_param.grad is not None
    std_gamma_grad = float(attn_std.qk_gain_param.grad.norm().item())

    torch.manual_seed(7)
    attn_pur = SubspaceProjectorBornAttention(
        embed_dim=d,
        num_heads=h,
        block_size=t,
        use_flash_born=False,
        use_purified_attn=True,
    ).to(device)
    out_pur = attn_pur(x)
    loss_pur = (out_pur * target).sum(dim=-1).mean()
    loss_pur.backward()
    assert attn_pur.qk_gain_param is not None and attn_pur.qk_gain_param.grad is not None
    assert attn_pur.coherence_tau_raw is not None and attn_pur.coherence_tau_raw.grad is not None
    pur_gamma_grad = float(attn_pur.qk_gain_param.grad.norm().item())
    pur_tau_grad = float(attn_pur.coherence_tau_raw.grad.norm().item())

    # Evaluate monotonic attention Shannon entropy across gamma in [2.0, 4.0, 8.0, 12.0]
    d_h = d // h
    q = torch.zeros(b, h, t, d_h, device=device)
    k = torch.zeros(b, h, t, d_h, device=device)
    k[:, :, 0, 0] = 1.0
    k[:, :, 1:, 1] = 0.05
    k[:, :, 1:, 3:] = F.normalize(torch.randn(b, h, t - 1, d_h - 3, device=device), p=2, dim=-1) * (
        (1.0 - 0.05**2) ** 0.5
    )
    q[:, :, :, 0] = 0.15
    q[:, :, :, 1] = 0.80
    q[:, :, :, 2] = (1.0 - 0.15**2 - 0.80**2) ** 0.5

    entropies: dict[str, float] = {}
    ent_list: list[float] = []
    for g_val in (2.0, 4.0, 8.0, 12.0):
        w = purified_born_attention_weights(q, k, gamma=g_val, tau=0.0625, eps_base=0.05, causal=True)
        w_half = w[:, :, t // 2 :, :].clamp_min(1e-12)
        ent = float(-(w_half * w_half.log()).sum(dim=-1).mean().item())
        entropies[f"gamma_{g_val:.1f}"] = round(ent, 4)
        ent_list.append(ent)

    monotonic = all(ent_list[i] > ent_list[i + 1] for i in range(len(ent_list) - 1))
    entropy_drop = round(ent_list[0] - ent_list[-1], 4)
    h2_pass = bool(
        std_gamma_grad < 1e-7
        and pur_gamma_grad > 1e-4
        and pur_tau_grad > 1e-4
        and monotonic
        and entropy_drop >= 0.80
    )
    print(
        f"  [H2 Grads] std_gamma_grad={std_gamma_grad:.2e} | pur_gamma_grad={pur_gamma_grad:.2e} | "
        f"pur_tau_grad={pur_tau_grad:.2e}"
    )
    print(f"  [H2 Entropy] {entropies} | drop={entropy_drop:.4f} nats | H2 PASS = {h2_pass}")
    return {
        "std_gamma_grad_norm": std_gamma_grad,
        "purified_gamma_grad_norm": pur_gamma_grad,
        "purified_tau_grad_norm": pur_tau_grad,
        "entropies_by_gamma": entropies,
        "monotonic_entropy_decrease": monotonic,
        "entropy_drop_2_to_12_nats": entropy_drop,
        "passed": h2_pass,
    }


def run_h3_classical_speedrun_experiment(
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    device: torch.device,
    steps: int = 200,
    batch_size: int = 16,
    seq_len: int = 256,
) -> dict[str, Any]:
    """Run Experiment H3: wave6_classical_baseline vs. speedrun_classical_upgraded on FineWeb tokens."""
    print("\n" + "=" * 88)
    print("EXPERIMENT H3: Classical Modded-NanoGPT Zero-Init + Decoupled lr_wte + Nesterov Muon")
    print("=" * 88)

    stride = batch_size * seq_len
    max_batches = (len(train_tokens) - 1) // stride
    arm_results: dict[str, Any] = {}

    for arm_name in ("wave6_classical_baseline", "speedrun_classical_upgraded"):
        upgraded = arm_name == "speedrun_classical_upgraded"
        torch.manual_seed(42)
        model = ClassicalModdedGPT(
            vocab_size=50304,
            embed_dim=384,
            num_heads=6,
            num_layers=6,
            hidden_dim=1536,
            block_size=seq_len,
            tie_embeddings=True,
            zero_init_residual_and_head=upgraded,
        ).to(device)

        adamw_pats = (
            "wte",
            "povm_head",
            "lm_head",
            "purity_gate",
            "highway",
            "v_mix",
            "embed",
            "unet_scales",
            "smear",
            "inv_temp",
            "kraus_readout",
            "seed_gate",
        )
        muon_params, adamw_params = split_muon_adamw_params(model, adamw_name_patterns=adamw_pats)
        base_muon_lr = 0.025
        base_adamw_lr = 0.004

        opt_muon = StiefelMuon(
            muon_params,
            lr=base_muon_lr,
            momentum=0.95,
            nesterov=upgraded,
            aspect_scale=upgraded,
            weight_decay=0.01,
            decay_mode="cautious",
            stiefel_penalty=0.005 if upgraded else 0.0,
            retract_every=4 if upgraded else 1_000_000,
        )

        if upgraded:
            wte_params: list[nn.Parameter] = []
            head_params: list[nn.Parameter] = []
            scalar_params: list[nn.Parameter] = []
            muon_ids = {id(p) for p in muon_params}
            for n, p in model.named_parameters():
                if not p.requires_grad or id(p) in muon_ids:
                    continue
                if "wte" in n:
                    wte_params.append(p)
                elif "lm_head" in n:
                    head_params.append(p)
                else:
                    scalar_params.append(p)
            opt_adamw = torch.optim.AdamW(
                [
                    {"params": wte_params, "lr": 0.20, "base_lr": 0.20, "weight_decay": 0.0},
                    {"params": head_params, "lr": 0.008, "base_lr": 0.008, "weight_decay": 0.0},
                    {"params": scalar_params, "lr": 0.02, "base_lr": 0.02, "weight_decay": 0.0},
                ],
                betas=(0.9, 0.95),
            )
        else:
            opt_adamw = torch.optim.AdamW(
                [{"params": adamw_params, "lr": base_adamw_lr, "base_lr": base_adamw_lr, "weight_decay": 0.01}],
                betas=(0.9, 0.95),
            )

        history: list[dict[str, float]] = []
        val0 = evaluate_model(model, val_tokens, batch_size, seq_len, device)
        val0["step"] = 0
        val0["iso_err"] = round(compute_model_iso_err(model, include_zero_init=False), 4)
        history.append(val0)

        t0 = time.perf_counter()
        for step in range(1, steps + 1):
            lr_mult = cosine_lr_multiplier(step, steps, warmup_steps=20, min_lr_ratio=0.15)
            for g in opt_muon.param_groups:
                g["lr"] = base_muon_lr * lr_mult
            for g in opt_adamw.param_groups:
                g["lr"] = float(g["base_lr"]) * lr_mult

            b_idx = (step - 1) % max_batches
            chunk = train_tokens[b_idx * stride : (b_idx + 1) * stride + 1].to(device)
            x = chunk[:-1].reshape(batch_size, seq_len)
            y = chunk[1:].reshape(batch_size, seq_len)

            opt_muon.zero_grad(set_to_none=True)
            opt_adamw.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                _, loss = model(x, targets=y)
            assert loss is not None
            loss.backward()
            if upgraded:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_muon.step()
            opt_adamw.step()

            if step in (50, 100, 150, steps):
                vk = evaluate_model(model, val_tokens, batch_size, seq_len, device)
                vk["step"] = step
                vk["iso_err"] = round(compute_model_iso_err(model, include_zero_init=False), 4)
                history.append(vk)

        elapsed = round(time.perf_counter() - t0, 2)
        ck0 = history[0]
        ck_end = history[-1]
        print(
            f"  [{arm_name:<28}] Step 0 CE={ck0['val_ce']:.4f} -> "
            f"Step {steps} CE={ck_end['val_ce']:.4f} | Top1={ck_end['top1_acc']:.2f}% | "
            f"IsoErr={ck_end['iso_err']:.4f} ({elapsed:.1f}s)"
        )
        arm_results[arm_name] = {
            "history": history,
            "step0_val_ce": ck0["val_ce"],
            "step200_val_ce": ck_end["val_ce"],
            "step200_top1_acc": ck_end["top1_acc"],
            "step200_iso_err": ck_end["iso_err"],
            "elapsed_sec": elapsed,
        }

    base_ce = arm_results["wave6_classical_baseline"]["step200_val_ce"]
    up_ce = arm_results["speedrun_classical_upgraded"]["step200_val_ce"]
    ce_gain = round(base_ce - up_ce, 4)

    base_iso = arm_results["wave6_classical_baseline"]["step200_iso_err"]
    up_iso = arm_results["speedrun_classical_upgraded"]["step200_iso_err"]
    iso_reduction_pct = round(100.0 * (base_iso - up_iso) / max(base_iso, 1e-8), 2)

    h3_pass = bool(ce_gain >= 0.12 and iso_reduction_pct >= 40.0)
    print(
        f"  [H3 Summary] val_ce_gain={ce_gain:+.4f} nats | "
        f"iso_err_reduction={iso_reduction_pct:.2f}% ({base_iso:.4f} -> {up_iso:.4f}) | H3 PASS = {h3_pass}"
    )
    return {
        "arms": arm_results,
        "val_ce_gain_nats": ce_gain,
        "iso_err_reduction_pct": iso_reduction_pct,
        "passed": h3_pass,
    }


def main() -> int:
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tokens, val_tokens, dataset_source = load_fineweb_tokens(
        num_train_tokens=1_000_000,
        num_val_tokens=131_072,
    )
    print(f"Loaded dataset ({dataset_source}) on device={device}")

    h1 = run_h1_quantum_early_epoch_experiment(train_tokens, val_tokens, device=device)
    h2 = run_h2_purified_attention_gradient_and_entropy(device=device)
    h3 = run_h3_classical_speedrun_experiment(train_tokens, val_tokens, device=device)

    all_passed = bool(h1["passed"] and h2["passed"] and h3["passed"])
    out_payload = {
        "experiment_plan": "EXPERIMENT_PLAN_014",
        "device": str(device),
        "dataset_source": dataset_source,
        "H1_early_epoch_quantum": h1,
        "H2_purified_born_attention": h2,
        "H3_classical_speedrun": h3,
        "all_passed": all_passed,
    }
    out_path = REPO_ROOT / ".cache" / "hypothesis_experiments_014.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print("\n" + "=" * 88)
    print(f"Saved results to {out_path} | ALL HYPOTHESES PASSED = {all_passed}")
    print("=" * 88)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
