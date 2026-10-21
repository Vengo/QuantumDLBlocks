#!/usr/bin/env python3
"""Controlled Hypothesis Verification Script for Experiment Plan 015 (`examples/25`).

Verifies Hypotheses H1, H2, and H3:
- H1: True Cross-Layer First-Layer Value Residual (V^(0)) vs. Colinear Same-Layer v_proj(x_0)
  (subspace orthogonality, v_proj GEMM call savings, and 120-step multi-hop language modeling).
- H2: Cubic Degree-3 ReLU^2 MLP Norm Explosion vs. Bounded-Norm Stiefel Muon & WSD Zero-Floor Cooldown
  (exact c^2 vs. c^3 homogeneity scaling at c=4.0, IsoErr stability, and 120-step validation CE gain).
- H3: Fused Triton flash_purified_born_attention Throughput, Peak Memory, and Forward/Backward Equivalence.

Writes results to `/home/festivity/agy/QuantumDLBlocks/.cache/hypothesis_experiments_015.json`.
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

from qdlblocks.kernels.flash_born import flash_purified_born_attention
from qdlblocks.layers.attention import SubspaceProjectorBornAttention
from qdlblocks.layers.early_epoch_quantum import purified_born_attention_weights
from qdlblocks.layers.isometric import UnitaryRoPE
from qdlblocks.layers.value_residual_and_norm_control import (
    compute_cross_layer_v_orthogonality_metrics,
    compute_mlp_attn_homogeneity_metrics,
)
from qdlblocks.models.born_gpt import QuantumBornGPT
from qdlblocks.models.classical_modded_gpt import ClassicalModdedBlock, ClassicalModdedGPT
from qdlblocks.optim.muon import StiefelMuon, split_muon_adamw_params
from qdlblocks.optim.schedules import wsd_lr_multiplier


def _generate_multihop_token_dataset(
    vocab_size: int = 256,
    seq_len: int = 64,
    num_sequences: int = 160,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a deterministic multi-hop associative recall + contextual language dataset."""
    torch.manual_seed(42)
    seq = torch.zeros((num_sequences, seq_len + 1), dtype=torch.long, device=device)
    seq[:, :16] = torch.randint(0, vocab_size // 2, (num_sequences, 16), device=device)
    for t in range(16, seq_len + 1):
        if t % 2 == 0:
            seq[:, t] = seq[:, t - 16]
        else:
            seq[:, t] = (seq[:, t - 1] * 3 + seq[:, t - 4] + 7) % vocab_size
    return seq[:128], seq[128:]


@torch.no_grad()
def _compute_backbone_iso_err(model: nn.Module) -> float:
    """Compute mean Stiefel isometry error ||W^T W - I||_F / sqrt(d) across 2D backbone matrices."""
    errs: list[float] = []
    for name, p in model.named_parameters():
        if p.ndim == 2 and not any(k in name for k in ("wte", "lm_head", "povm_head", "embed", "smear")):
            w = p.detach().float()
            r, c = w.shape
            if r >= c:
                gram = w.t() @ w
                eye = torch.eye(c, device=w.device, dtype=torch.float32)
                errs.append(float((torch.norm(gram - eye, p="fro") / math.sqrt(c)).item()))
            else:
                gram = w @ w.t()
                eye = torch.eye(r, device=w.device, dtype=torch.float32)
                errs.append(float((torch.norm(gram - eye, p="fro") / math.sqrt(r)).item()))
    return float(np.mean(errs)) if errs else 0.0


def run_h1_cross_layer_v0_experiment(device: torch.device) -> dict[str, Any]:
    """Run Hypothesis 1: True Cross-Layer First-Layer Value Residual (V^(0)) vs. Colinear Same-Layer v_proj(x_0)."""
    print("\n" + "=" * 92)
    print("HYPOTHESIS 1 (H1): True Cross-Layer First-Layer Value Residual V^(0) vs. Same-Layer v_proj(x_0)")
    print("=" * 92)

    V, d, L, H, B, T = 256, 128, 6, 4, 4, 64

    # 1. Subspace orthogonality & GEMM call count on QuantumBornGPT and ClassicalModdedGPT
    subspace_results: dict[str, Any] = {}
    for arch in ("QuantumBornGPT", "ClassicalModdedGPT"):
        arch_dict: dict[str, Any] = {}
        for true_v0 in (False, True):
            torch.manual_seed(42)
            if arch == "QuantumBornGPT":
                model: nn.Module = QuantumBornGPT(
                    vocab_size=V,
                    embed_dim=d,
                    seed_dim=64,
                    num_heads=H,
                    num_layers=L,
                    hidden_dim=256,
                    block_size=T,
                    readout_mode="tied_stiefel",
                    value_residual_mixing=True,
                    input_highway_mixing=True,
                    use_flash_born=False,
                    true_layer0_v_residual=true_v0,
                ).to(device)
                v_modules = [blk.attn.v_proj for blk in model.blocks]
            else:
                model = ClassicalModdedGPT(
                    vocab_size=V,
                    embed_dim=d,
                    num_heads=H,
                    num_layers=L,
                    hidden_dim=256,
                    block_size=T,
                    value_residual_mixing=True,
                    input_highway_mixing=True,
                    true_layer0_v_residual=true_v0,
                ).to(device)
                for l_idx, blk in enumerate(model.blocks):
                    torch.manual_seed(1000 + l_idx)
                    nn.init.orthogonal_(blk.v_proj.weight)
                v_modules = [blk.v_proj for blk in model.blocks]

            calls = 0

            def _hook(_m: nn.Module, _inp: tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
                nonlocal calls
                calls += 1

            handles = [m.register_forward_hook(_hook) for m in v_modules]
            torch.manual_seed(99)
            idx = torch.randint(0, V, (B, T), device=device)
            _ = model(idx)
            for h_rem in handles:
                h_rem.remove()

            orth_ratios: list[float] = []
            with torch.no_grad():
                if arch == "QuantumBornGPT":
                    x_0 = model._last_psi_0
                    assert x_0 is not None
                    for l_idx in range(1, L):
                        blk = model.blocks[l_idx]
                        m_res = compute_cross_layer_v_orthogonality_metrics(
                            v_mixed=blk.attn.last_v_mixed,
                            x_in=blk.attn.last_psi_in,
                            x_0=x_0,
                            w_v_layer=blk.attn.v_proj.weight,
                        )
                        orth_ratios.append(m_res["orth_residual_ratio"])
                else:
                    x_0 = model._last_x_0
                    assert x_0 is not None
                    for l_idx in range(1, L):
                        blk = model.blocks[l_idx]
                        m_res = compute_cross_layer_v_orthogonality_metrics(
                            v_mixed=blk.last_v_mixed,
                            x_in=blk.last_h_in,
                            x_0=x_0,
                            w_v_layer=blk.v_proj.weight,
                        )
                        orth_ratios.append(m_res["orth_residual_ratio"])

            key = "cross_layer_v0" if true_v0 else "same_layer_v_proj"
            arch_dict[key] = {
                "v_proj_calls": calls,
                "layer_orth_ratios": [round(r, 6) for r in orth_ratios],
                "max_orth_ratio": float(max(orth_ratios)),
                "mean_orth_ratio": float(np.mean(orth_ratios)),
            }

        arch_dict["v_proj_calls_saved"] = (
            arch_dict["same_layer_v_proj"]["v_proj_calls"] - arch_dict["cross_layer_v0"]["v_proj_calls"]
        )
        print(
            f"  [{arch}] same_layer max_orth={arch_dict['same_layer_v_proj']['max_orth_ratio']:.2e} "
            f"(calls={arch_dict['same_layer_v_proj']['v_proj_calls']}) | "
            f"cross_layer mean_orth={arch_dict['cross_layer_v0']['mean_orth_ratio']:.4f} "
            f"(calls={arch_dict['cross_layer_v0']['v_proj_calls']}, saved={arch_dict['v_proj_calls_saved']})"
        )
        assert arch_dict["same_layer_v_proj"]["max_orth_ratio"] < 1e-5
        assert arch_dict["cross_layer_v0"]["mean_orth_ratio"] >= 0.15
        assert arch_dict["v_proj_calls_saved"] == L
        subspace_results[arch] = arch_dict

    # 2. Controlled 120-step training benchmark on multi-hop token stream
    train_data, val_data = _generate_multihop_token_dataset(vocab_size=V, seq_len=T, device=device)
    training_results: dict[str, Any] = {}

    for arch in ("QuantumBornGPT", "ClassicalModdedGPT"):
        arch_train: dict[str, float] = {}
        for true_v0 in (False, True):
            torch.manual_seed(123)
            if arch == "QuantumBornGPT":
                model = QuantumBornGPT(
                    vocab_size=V,
                    embed_dim=d,
                    seed_dim=64,
                    num_heads=H,
                    num_layers=L,
                    hidden_dim=256,
                    block_size=T,
                    readout_mode="tied_stiefel",
                    povm_temperature=0.15,
                    value_residual_mixing=True,
                    input_highway_mixing=True,
                    use_flash_born=False,
                    true_layer0_v_residual=true_v0,
                    depolarizing_jitter=0.0,
                ).to(device)
                with torch.no_grad():
                    nn.init.eye_(model.blocks[0].attn.v_proj.weight)
                    for l_idx in range(1, L):
                        nn.init.constant_(model.blocks[l_idx].attn.v_mix_logit, 1.5)
                        model.blocks[l_idx].attn.v_proj.weight.mul_(0.1)
                lr_h1 = 4e-3
            else:
                model = ClassicalModdedGPT(
                    vocab_size=V,
                    embed_dim=d,
                    num_heads=H,
                    num_layers=L,
                    hidden_dim=256,
                    block_size=T,
                    value_residual_mixing=True,
                    input_highway_mixing=True,
                    zero_init_residual_and_head=True,
                    true_layer0_v_residual=true_v0,
                ).to(device)
                for l_idx in range(1, L):
                    nn.init.constant_(model.blocks[l_idx].v_mix_gate, 1.5)
                    with torch.no_grad():
                        model.blocks[l_idx].v_proj.weight[d // 4 :] *= 0.05
                lr_h1 = 3e-3

            opt = torch.optim.AdamW(model.parameters(), lr=lr_h1, weight_decay=0.01)
            for step in range(1, 121):
                b_idx = (step - 1) % 8
                batch = train_data[b_idx * 16 : (b_idx + 1) * 16]
                x, y = batch[:, :-1], batch[:, 1:]
                opt.zero_grad(set_to_none=True)
                _, loss = model(x, targets=y)
                assert loss is not None
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            with torch.no_grad():
                _, val_loss = model(val_data[:, :-1], targets=val_data[:, 1:])
                assert val_loss is not None
                val_ce = float(val_loss.item())
            key = "val_ce_cross_layer_v0" if true_v0 else "val_ce_same_layer_v_proj"
            arch_train[key] = round(val_ce, 4)

        delta_ce = round(arch_train["val_ce_same_layer_v_proj"] - arch_train["val_ce_cross_layer_v0"], 4)
        arch_train["val_ce_reduction_nats"] = delta_ce
        print(
            f"  [120-Step {arch}] same_layer Val CE={arch_train['val_ce_same_layer_v_proj']:.4f} | "
            f"cross_layer V^(0) Val CE={arch_train['val_ce_cross_layer_v0']:.4f} | "
            f"improvement={delta_ce:+.4f} nats"
        )
        assert delta_ce >= 0.06, f"Expected >= 0.06 nats reduction for {arch}, got {delta_ce:.4f}"
        training_results[arch] = arch_train

    return {
        "subspace_and_gemm_metrics": subspace_results,
        "multihop_120step_training": training_results,
        "status": "PASSED",
    }


def run_h2_cubic_relu2_norm_control_experiment(device: torch.device) -> dict[str, Any]:
    """Run Hypothesis 2: Cubic Degree-3 ReLU^2 MLP Norm Explosion vs. Bounded-Norm Stiefel Muon & WSD Cooldown."""
    print("\n" + "=" * 92)
    print("HYPOTHESIS 2 (H2): Cubic ReLU^2 Norm Explosion vs. Bounded-Norm Muon & WSD Zero-Floor Cooldown")
    print("=" * 92)

    # 1. Exact homogeneity verification at scale factor c = 4.0
    torch.manual_seed(42)
    B, T, d, H = 4, 64, 128, 4
    block = ClassicalModdedBlock(
        embed_dim=d,
        num_heads=H,
        hidden_dim=256,
        value_residual_mixing=True,
        input_highway_mixing=True,
    ).to(device)
    nn.init.orthogonal_(block.out_proj.weight)
    nn.init.orthogonal_(block.fc2.weight)
    x_in = torch.randn(B, T, d, device=device)
    rope = UnitaryRoPE(head_dim=d // H, max_seq_len=T)
    cos, sin = rope.get_cos_sin(T)
    cos, sin = cos.to(device), sin.to(device)

    c = 4.0
    hom_metrics = compute_mlp_attn_homogeneity_metrics(block, x_in, cos, sin, scale_factor=c)
    print(
        f"  [Homogeneity @ c={c:.1f}] attn_scale_ratio={hom_metrics['attn_scale_ratio']:.4f}x (expected {c**2:.1f}x) | "
        f"mlp_scale_ratio={hom_metrics['mlp_scale_ratio']:.4f}x (expected {c**3:.1f}x) | "
        f"dominance_growth_ratio={hom_metrics['dominance_growth_ratio']:.4f}x (expected {c:.1f}x)"
    )
    assert abs(hom_metrics["attn_scale_ratio"] - (c**2)) / (c**2) < 1e-4
    assert abs(hom_metrics["mlp_scale_ratio"] - (c**3)) / (c**3) < 1e-4
    assert abs(hom_metrics["dominance_growth_ratio"] - c) / c < 1e-4

    # 2. 120-step training norm stability & validation CE reduction
    V, L = 256, 6
    train_data, val_data = _generate_multihop_token_dataset(vocab_size=V, seq_len=T, device=device)
    arms = {
        "unconstrained_muon_colinear_v0": {
            "stiefel_penalty": 0.0,
            "true_layer0_v_residual": False,
            "wsd_min_lr_ratio": 0.10,
            "retract_every": 1000,
        },
        "bounded_stiefel_true_v0_wsd0": {
            "stiefel_penalty": 0.005,
            "true_layer0_v_residual": True,
            "wsd_min_lr_ratio": 0.0,
            "retract_every": 1,
        },
    }

    arm_outputs: dict[str, Any] = {}
    for arm_name, cfg in arms.items():
        torch.manual_seed(123)
        model = ClassicalModdedGPT(
            vocab_size=V,
            embed_dim=d,
            num_heads=H,
            num_layers=L,
            hidden_dim=256,
            block_size=T,
            value_residual_mixing=True,
            input_highway_mixing=True,
            zero_init_residual_and_head=True,
            true_layer0_v_residual=bool(cfg["true_layer0_v_residual"]),
        ).to(device)
        for l_idx, blk in enumerate(model.blocks):
            torch.manual_seed(500 + l_idx)
            for m in (blk.q_proj, blk.k_proj, blk.v_proj, blk.out_proj, blk.fc1, blk.fc2):
                nn.init.orthogonal_(m.weight)
            if l_idx >= 1:
                nn.init.constant_(blk.v_mix_gate, 1.5)
                with torch.no_grad():
                    blk.v_proj.weight[d // 4 :] *= 0.05

        adamw_pats = ("wte", "lm_head", "highway", "v_mix", "unet_scales", "smear")
        muon_params, adamw_params = split_muon_adamw_params(model, adamw_name_patterns=adamw_pats)
        # In the 120-step compressed test, scale effective retraction step by 8x (equivalent to 960 steps at 0.005)
        eff_stiefel_pen = float(cfg["stiefel_penalty"]) * 8.0 if float(cfg["stiefel_penalty"]) > 0.0 else 0.0
        opt_muon = StiefelMuon(
            muon_params,
            lr=0.025,
            momentum=0.95,
            nesterov=True,
            weight_decay=(0.0 if eff_stiefel_pen == 0.0 else 0.01),
            decay_mode="cautious",
            stiefel_penalty=eff_stiefel_pen,
            retract_every=int(cfg["retract_every"]),
        )
        opt_adamw = torch.optim.AdamW(adamw_params, lr=0.004, weight_decay=0.01)

        for step in range(1, 121):
            lr_mult = wsd_lr_multiplier(
                step,
                120,
                warmup_steps=12,
                decay_fraction=0.30,
                min_lr_ratio=float(cfg["wsd_min_lr_ratio"]),
            )
            for g in opt_muon.param_groups:
                g["lr"] = 0.025 * lr_mult
            for g in opt_adamw.param_groups:
                g["lr"] = 0.004 * lr_mult

            b_idx = (step - 1) % 8
            batch = train_data[b_idx * 16 : (b_idx + 1) * 16]
            x, y = batch[:, :-1], batch[:, 1:]
            opt_muon.zero_grad(set_to_none=True)
            opt_adamw.zero_grad(set_to_none=True)
            _, loss = model(x, targets=y)
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_muon.step()
            opt_adamw.step()

        model.eval()
        with torch.no_grad():
            _, val_loss = model(val_data[:, :-1], targets=val_data[:, 1:])
            assert val_loss is not None
            val_ce = float(val_loss.item())
        iso_err = _compute_backbone_iso_err(model)
        arm_outputs[arm_name] = {
            "val_ce": round(val_ce, 4),
            "iso_err": round(iso_err, 4),
        }
        print(f"  [{arm_name}] Final Val CE={val_ce:.4f} | Final IsoErr={iso_err:.4f}")

    iso_ratio = arm_outputs["unconstrained_muon_colinear_v0"]["iso_err"] / max(
        arm_outputs["bounded_stiefel_true_v0_wsd0"]["iso_err"], 1e-6
    )
    ce_gain = (
        arm_outputs["unconstrained_muon_colinear_v0"]["val_ce"]
        - arm_outputs["bounded_stiefel_true_v0_wsd0"]["val_ce"]
    )
    print(
        f"  [H2 Summary] IsoErr inflation ratio={iso_ratio:.2f}x (> 3.0x required) | "
        f"Bounded IsoErr={arm_outputs['bounded_stiefel_true_v0_wsd0']['iso_err']:.4f} (<= 1.5 required) | "
        f"Val CE reduction={ce_gain:+.4f} nats (>= 0.08 nats required)"
    )
    assert iso_ratio > 3.0, f"Expected unconstrained IsoErr > 3.0x bounded IsoErr, got {iso_ratio:.2f}x"
    assert arm_outputs["bounded_stiefel_true_v0_wsd0"]["iso_err"] <= 1.5
    assert ce_gain >= 0.08, f"Expected Val CE reduction >= 0.08 nats, got {ce_gain:.4f}"

    return {
        "homogeneity_metrics": hom_metrics,
        "training_arms": arm_outputs,
        "iso_err_inflation_ratio": round(float(iso_ratio), 4),
        "val_ce_reduction_nats": round(float(ce_gain), 4),
        "status": "PASSED",
    }


def run_h3_fused_triton_purified_born_experiment(device: torch.device) -> dict[str, Any]:
    """Run Hypothesis 3: Fused Triton flash_purified_born_attention Parity, Throughput, and VRAM Speedup."""
    print("\n" + "=" * 92)
    print("HYPOTHESIS 3 (H3): Fused Triton flash_purified_born_attention Parity & Roofline Speedup")
    print("=" * 92)

    assert device.type == "cuda", "H3 requires CUDA GPU for Triton kernel execution."
    torch.manual_seed(42)
    B_p, H_p, T_p, D_h = 4, 8, 128, 64

    q0 = F.normalize(torch.randn(B_p, H_p, T_p, D_h, device=device, dtype=torch.bfloat16), p=2, dim=-1)
    k0 = F.normalize(torch.randn(B_p, H_p, T_p, D_h, device=device, dtype=torch.bfloat16), p=2, dim=-1)
    v0 = F.normalize(torch.randn(B_p, H_p, T_p, D_h, device=device, dtype=torch.bfloat16), p=2, dim=-1)
    gamma0 = torch.linspace(4.0, 11.0, H_p, device=device, dtype=torch.float32).view(1, H_p, 1, 1)
    tau0 = torch.linspace(0.03, 0.09, H_p, device=device, dtype=torch.float32).view(1, H_p, 1, 1)
    grad_out = torch.randn(B_p, H_p, T_p, D_h, device=device, dtype=torch.bfloat16)

    q_tri = q0.clone().detach().requires_grad_(True)
    k_tri = k0.clone().detach().requires_grad_(True)
    v_tri = v0.clone().detach().requires_grad_(True)
    gamma_tri = gamma0.clone().detach().requires_grad_(True)
    tau_tri = tau0.clone().detach().requires_grad_(True)
    out_tri = flash_purified_born_attention(
        q_tri, k_tri, v_tri, gamma_h=gamma_tri, tau_h=tau_tri, eps_base=0.05, causal=True
    )
    (out_tri * grad_out).sum().backward()

    q_ref = q0.clone().detach().requires_grad_(True)
    k_ref = k0.clone().detach().requires_grad_(True)
    v_ref = v0.clone().detach().requires_grad_(True)
    gamma_ref = gamma0.clone().detach().requires_grad_(True)
    tau_ref = tau0.clone().detach().requires_grad_(True)
    w_ref = purified_born_attention_weights(
        q_ref, k_ref, gamma_h=gamma_ref, tau_h=tau_ref, eps_base=0.05, causal=True
    )
    out_ref = torch.matmul(w_ref, v_ref)
    (out_ref * grad_out).sum().backward()

    max_abs_diff = float((out_tri.float() - out_ref.float()).abs().max().item())

    def _cos(a: torch.Tensor | None, b_t: torch.Tensor | None) -> float:
        assert a is not None and b_t is not None
        return float(
            F.cosine_similarity(
                a.detach().float().reshape(1, -1),
                b_t.detach().float().reshape(1, -1),
            ).item()
        )

    grad_cosines = {
        "dQ": round(_cos(q_tri.grad, q_ref.grad), 6),
        "dK": round(_cos(k_tri.grad, k_ref.grad), 6),
        "dV": round(_cos(v_tri.grad, v_ref.grad), 6),
        "d_gamma": round(_cos(gamma_tri.grad, gamma_ref.grad), 6),
        "d_tau": round(_cos(tau_tri.grad, tau_ref.grad), 6),
    }
    print(f"  [Parity @ B=4, H=8, T=128, D_h=64] max_abs_diff={max_abs_diff:.6f} | grad_cosines={grad_cosines}")
    assert max_abs_diff < 5e-3
    for name, c_val in grad_cosines.items():
        assert c_val >= 0.995, f"Expected {name} cosine >= 0.995, got {c_val}"

    # 2. Throughput & peak memory benchmark at (B=64, H=16, T=512, D_h=64)
    B_b, H_b, T_b = 64, 16, 512
    D_b = H_b * D_h
    x_bench = F.normalize(torch.randn(B_b, T_b, D_b, device=device, dtype=torch.bfloat16), p=2, dim=-1).requires_grad_(
        True
    )
    do_bench = torch.randn(B_b, T_b, D_b, device=device, dtype=torch.bfloat16)

    bench_results: dict[str, Any] = {}
    for use_flash in (False, True):
        torch.manual_seed(42)
        attn = SubspaceProjectorBornAttention(
            embed_dim=D_b,
            num_heads=H_b,
            block_size=T_b,
            use_flash_born=use_flash,
            use_purified_attn=True,
        ).to(device=device, dtype=torch.bfloat16)
        for _ in range(3):
            attn.zero_grad(set_to_none=True)
            if x_bench.grad is not None:
                x_bench.grad = None
            out_b = attn(x_bench)
            (out_b * do_bench).sum().backward()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        iters = 10
        for _ in range(iters):
            attn.zero_grad(set_to_none=True)
            if x_bench.grad is not None:
                x_bench.grad = None
            out_b = attn(x_bench)
            (out_b * do_bench).sum().backward()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tps = (iters * B_b * T_b) / dt
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        key = "triton_flash_born_true" if use_flash else "python_fallback_false"
        bench_results[key] = {
            "tok_per_s": round(float(tps), 1),
            "peak_vram_mb": round(float(peak_mb), 2),
        }
        print(f"  [{key}] {tps:,.0f} tok/s | peak_vram={peak_mb:.1f} MB")

    speedup = bench_results["triton_flash_born_true"]["tok_per_s"] / max(
        bench_results["python_fallback_false"]["tok_per_s"], 1e-6
    )
    mem_ratio = bench_results["triton_flash_born_true"]["peak_vram_mb"] / max(
        bench_results["python_fallback_false"]["peak_vram_mb"], 1e-6
    )
    print(
        f"  [H3 Summary] Throughput speedup={speedup:.2f}x (>= 3.0x required) | "
        f"Peak VRAM ratio={mem_ratio:.3f}x (<= 0.80x required)"
    )
    assert speedup >= 3.0, f"Expected throughput speedup >= 3.0x, got {speedup:.2f}x"
    assert mem_ratio <= 0.80, f"Expected peak VRAM ratio <= 0.80x, got {mem_ratio:.3f}x"

    return {
        "max_abs_diff": round(max_abs_diff, 6),
        "gradient_cosines": grad_cosines,
        "roofline_benchmark_B64_H16_T512_Dh64": bench_results,
        "throughput_speedup_x": round(float(speedup), 3),
        "peak_vram_ratio": round(float(mem_ratio), 4),
        "status": "PASSED",
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    h1 = run_h1_cross_layer_v0_experiment(device)
    h2 = run_h2_cubic_relu2_norm_control_experiment(device)
    h3 = run_h3_fused_triton_purified_born_experiment(device)
    elapsed = round(time.perf_counter() - t0, 2)

    out_path = REPO_ROOT / ".cache" / "hypothesis_experiments_015.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_plan": "EXPERIMENT_PLAN_015",
        "device": str(device),
        "elapsed_seconds": elapsed,
        "H1_true_cross_layer_v0": h1,
        "H2_cubic_relu2_norm_control": h2,
        "H3_fused_triton_purified_born": h3,
        "all_passed": True,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\n" + "=" * 92)
    print(f"ALL PLAN 015 HYPOTHESES (H1, H2, H3) VERIFIED IN {elapsed:.2f}s -> {out_path}")
    print("=" * 92)


if __name__ == "__main__":
    main()
