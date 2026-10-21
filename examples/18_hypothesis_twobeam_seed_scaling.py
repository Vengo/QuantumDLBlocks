"""Example 18: Controlled Verification of Observable Hypotheses on Two-Beam Bilinear Interference
vs. Single-Beam Overcomplete Frames (H1) and Seed-Space Capacity Scaling in `QuantumBornGPT` (H2).

Implements Section 3.3 of EXPERIMENT_PLAN_008.md and writes empirical results to
`.cache/hypothesis_experiments_008.json`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.functional import (
    effective_quantum_rank,
    empirical_state_density,
    l2_normalize,
    purity_participation_ratio,
)
from qdlblocks.layers.overcomplete_frame import (
    OvercompleteFrameInterference,
    optimize_overcomplete_tight_frame,
)
from qdlblocks.layers.seed_scaling_diagnostics import (
    compute_cross_beam_coherence,
    compute_cross_subspace_regression_mse,
    compute_effective_seed_utilization_ratio,
    compute_interference_feature_rank,
    single_beam_signed_born_features,
    two_beam_bilinear_born_features,
)
from qdlblocks.models.born_gpt import QuantumBornGPT


def run_experiment_h1_cross_subspace_degree3_fitting(device: torch.device) -> dict[str, Any]:
    """Experiment 1 (H1): Degree-3 Cross-Subspace Feature Rank & Expressivity of Two-Beam vs. Single-Beam."""
    print("\n" + "=" * 96)
    print("EXPERIMENT 1 (H1): Degree-3 Cross-Subspace Expressivity — Two-Beam (1.5d x 2) vs. Single-Beam (2.0d)")
    print("=" * 96)

    torch.manual_seed(42)
    d = 64
    m1 = 2 * d  # 128
    m2 = int(round(1.5 * d))  # 96
    k_out = 32
    n_train = 1024
    n_test = 512
    steps = 250
    lr = 1e-2
    scale = float(d) ** 1.5

    psi_train = l2_normalize(torch.randn(n_train, d, device=device, dtype=torch.float32), dim=-1)
    psi_test = l2_normalize(torch.randn(n_test, d, device=device, dtype=torch.float32), dim=-1)

    # Construct orthogonal U_A via QR decomposition (seed=777) and cyclic-shifted U_B = roll(U_A, 1, 0)
    g_a = torch.Generator(device="cpu").manual_seed(777)
    u_a_cpu, _ = torch.linalg.qr(torch.randn(d, d, generator=g_a, dtype=torch.float32))
    u_a = u_a_cpu.T.contiguous().to(device)
    u_b = torch.roll(u_a, shifts=1, dims=0)

    # Verify <u_{A,i}, u_{B,i}> = 0 for all i
    orth_coh = compute_cross_beam_coherence(u_a, u_b)
    assert orth_coh["max_paired_beam_coherence"] < 1e-5, (
        f"Expected orthogonal paired target bases, got {orth_coh['max_paired_beam_coherence']}"
    )

    g_p = torch.Generator(device="cpu").manual_seed(888)
    p_proj_cpu = torch.empty(k_out, d, dtype=torch.float32)
    nn.init.orthogonal_(p_proj_cpu, generator=g_p)
    p_proj = p_proj_cpu.to(device)

    def _compute_target(psi: torch.Tensor) -> torch.Tensor:
        z_a = F.linear(psi, u_a)
        z_b = F.linear(psi, u_b)
        return scale * F.linear((z_a * z_a) * z_b, p_proj)

    y_train = _compute_target(psi_train)
    y_test = _compute_target(psi_test)

    # Initialize tight frames via optimize_overcomplete_tight_frame
    w_up_raw = optimize_overcomplete_tight_frame(m1, d, steps=150, lr=0.05, seed=42, device=device)
    w1_raw = optimize_overcomplete_tight_frame(m2, d, steps=150, lr=0.05, seed=101, device=device)
    w2_raw = optimize_overcomplete_tight_frame(m2, d, steps=150, lr=0.05, seed=202, device=device)

    w_up_init = l2_normalize(torch.cat([u_a, u_b], dim=0) + 0.05 * w_up_raw, dim=-1)
    w1_init = l2_normalize(torch.cat([u_a, w1_raw[d:]], dim=0) + 0.05 * w1_raw, dim=-1)
    w2_init = l2_normalize(torch.cat([u_b, w2_raw[d:]], dim=0) + 0.05 * w2_raw, dim=-1)

    with torch.no_grad():
        f1_tr_0 = scale * single_beam_signed_born_features(psi_train, w_up_init)
        f1_te_0 = scale * single_beam_signed_born_features(psi_test, w_up_init)
        rank_1beam_init = compute_interference_feature_rank(f1_tr_0)
        ridge_1beam = compute_cross_subspace_regression_mse(
            f1_tr_0, y_train, f1_te_0, y_test, ridge_lambda=1e-4
        )

        f2_tr_0 = scale * two_beam_bilinear_born_features(psi_train, w1_init, w2_init)
        f2_te_0 = scale * two_beam_bilinear_born_features(psi_test, w1_init, w2_init)
        rank_2beam_init = compute_interference_feature_rank(f2_tr_0)
        coh_2beam_init = compute_cross_beam_coherence(w1_init, w2_init)
        ridge_2beam = compute_cross_subspace_regression_mse(
            f2_tr_0, y_train, f2_te_0, y_test, ridge_lambda=1e-4
        )

    ridge_ratio = ridge_1beam["test_mse"] / max(ridge_2beam["test_mse"], 1e-12)
    print(
        f"  [Ridge Regression (lambda=1e-4)] "
        f"SingleBeam (M1={m1}) Test MSE: {ridge_1beam['test_mse']:.6f} (Train: {ridge_1beam['train_mse']:.6f}) | "
        f"TwoBeam (M2={m2}) Test MSE: {ridge_2beam['test_mse']:.6f} (Train: {ridge_2beam['train_mse']:.6f}) | "
        f"Ratio: {ridge_ratio:.2f}x"
    )

    assert ridge_2beam["test_mse"] < ridge_1beam["test_mse"], (
        f"H1 Failed: expected TwoBeam ridge test MSE < SingleBeam ridge test MSE, "
        f"got {ridge_2beam['test_mse']:.6f} vs {ridge_1beam['test_mse']:.6f}"
    )

    # End-to-End 250-Step AdamW Training (lr = 1e-2)
    # Note: row-normalization inside single_beam_signed_born_features / two_beam_bilinear_born_features
    # makes pre-normalized parameter scale invariant on the forward pass while conditioning angular step size.
    w_up = nn.Parameter(20.0 * w_up_init.clone())
    w_down_1 = nn.Parameter(torch.zeros(k_out, m1, device=device, dtype=torch.float32))
    opt1 = torch.optim.AdamW([w_up, w_down_1], lr=lr, weight_decay=1e-2)

    for _ in range(steps):
        opt1.zero_grad(set_to_none=True)
        pred_1 = F.linear(scale * single_beam_signed_born_features(psi_train, w_up), w_down_1)
        loss_1 = F.mse_loss(pred_1, y_train)
        loss_1.backward()
        opt1.step()

    with torch.no_grad():
        f1_tr_final = scale * single_beam_signed_born_features(psi_train, w_up)
        f1_te_final = scale * single_beam_signed_born_features(psi_test, w_up)
        e2e_1beam_train_mse = float(F.mse_loss(F.linear(f1_tr_final, w_down_1), y_train).item())
        e2e_1beam_test_mse = float(F.mse_loss(F.linear(f1_te_final, w_down_1), y_test).item())
        rank_1beam_final = compute_interference_feature_rank(f1_te_final)

    w1 = nn.Parameter(20.0 * w1_init.clone())
    w2 = nn.Parameter(20.0 * w2_init.clone())
    w_down_2 = nn.Parameter(torch.zeros(k_out, m2, device=device, dtype=torch.float32))
    opt2 = torch.optim.AdamW([w1, w2, w_down_2], lr=lr, weight_decay=1e-2)

    for _ in range(steps):
        opt2.zero_grad(set_to_none=True)
        pred_2 = F.linear(scale * two_beam_bilinear_born_features(psi_train, w1, w2), w_down_2)
        loss_2 = F.mse_loss(pred_2, y_train)
        loss_2.backward()
        opt2.step()

    with torch.no_grad():
        f2_tr_final = scale * two_beam_bilinear_born_features(psi_train, w1, w2)
        f2_te_final = scale * two_beam_bilinear_born_features(psi_test, w1, w2)
        e2e_2beam_train_mse = float(F.mse_loss(F.linear(f2_tr_final, w_down_2), y_train).item())
        e2e_2beam_test_mse = float(F.mse_loss(F.linear(f2_te_final, w_down_2), y_test).item())
        rank_2beam_final = compute_interference_feature_rank(f2_te_final)
        coh_2beam_final = compute_cross_beam_coherence(w1, w2)

    e2e_ratio = e2e_1beam_test_mse / max(e2e_2beam_test_mse, 1e-12)
    print(
        f"  [End-to-End 250-Step AdamW]      "
        f"SingleBeam (M1={m1}) Test MSE: {e2e_1beam_test_mse:.6f} (Train: {e2e_1beam_train_mse:.6f}) | "
        f"TwoBeam (M2={m2}) Test MSE: {e2e_2beam_test_mse:.6f} (Train: {e2e_2beam_train_mse:.6f}) | "
        f"Ratio: {e2e_ratio:.2f}x (>= 3.00x)"
    )
    print(
        f"  [Feature Rank & Coherence]       "
        f"1Beam R_eff: {rank_1beam_final['feature_eff_rank']:.2f} | "
        f"2Beam R_eff: {rank_2beam_final['feature_eff_rank']:.2f} | "
        f"2Beam RMS Paired Coherence: {coh_2beam_final['rms_paired_beam_coherence']:.5f}"
    )

    assert e2e_2beam_test_mse <= e2e_1beam_test_mse / 3.0, (
        f"H1 Failed: expected TwoBeam E2E test MSE <= SingleBeam / 3.0, "
        f"got {e2e_2beam_test_mse:.6f} vs {e2e_1beam_test_mse:.6f} (ratio {e2e_ratio:.2f}x)"
    )

    return {
        "embed_dim": d,
        "single_beam_channels": m1,
        "two_beam_channels": m2,
        "target_dim": k_out,
        "ridge_regression": {
            "single_beam": {k: round(v, 6) for k, v in ridge_1beam.items()},
            "two_beam": {k: round(v, 6) for k, v in ridge_2beam.items()},
            "test_mse_reduction_ratio": round(ridge_ratio, 4),
        },
        "end_to_end_250_steps": {
            "single_beam_train_mse": round(e2e_1beam_train_mse, 6),
            "single_beam_test_mse": round(e2e_1beam_test_mse, 6),
            "two_beam_train_mse": round(e2e_2beam_train_mse, 6),
            "two_beam_test_mse": round(e2e_2beam_test_mse, 6),
            "test_mse_reduction_ratio": round(e2e_ratio, 4),
        },
        "feature_rank_init": {
            "single_beam": {k: round(v, 4) for k, v in rank_1beam_init.items()},
            "two_beam": {k: round(v, 4) for k, v in rank_2beam_init.items()},
        },
        "feature_rank_final": {
            "single_beam": {k: round(v, 4) for k, v in rank_1beam_final.items()},
            "two_beam": {k: round(v, 4) for k, v in rank_2beam_final.items()},
        },
        "two_beam_coherence_init": {k: round(v, 6) for k, v in coh_2beam_init.items()},
        "two_beam_coherence_final": {k: round(v, 6) for k, v in coh_2beam_final.items()},
        "passed": True,
    }


@torch.no_grad()
def _evaluate_gpt_arm(
    model: QuantumBornGPT,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    logits, loss, layer_states = model(x_val, y_val, return_layer_states=True)
    assert loss is not None
    val_ce = float(loss.item())
    psi_last = layer_states[-1]
    assert model.readout_rot is not None
    psi_final = l2_normalize(model.readout_rot(psi_last), dim=-1, eps=model.eps)
    phi_seed = model.hilbert_lift.collapse_to_seed(psi_final)
    if model.seed_adapter is not None:
        phi_seed = model.seed_adapter(psi_final, phi_seed)

    rho_seed = empirical_state_density(phi_seed.float())
    seed_eff_rank = float(effective_quantum_rank(rho_seed, is_density=True).item())
    seed_pr = float(purity_participation_ratio(rho_seed, is_density=True).item())
    util = compute_effective_seed_utilization_ratio(phi_seed.float())

    preds = logits.argmax(dim=-1)
    top1_acc = 100.0 * float((preds == y_val).float().mean().item())
    return {
        "val_ce": round(val_ce, 4),
        "val_ppl": round(math.exp(min(val_ce, 20.0)), 2),
        "top1_acc": round(top1_acc, 2),
        "seed_eff_rank": round(seed_eff_rank, 4),
        "seed_participation_ratio": round(seed_pr, 4),
        "seed_utilization_ratio": round(util["seed_utilization_ratio"], 4),
    }


def run_experiment_h2_twobeam_and_seed_scaling(device: torch.device) -> dict[str, Any]:
    """Experiment 2 (H2): 4-Arm Two-Beam vs. Single-Beam x Linear vs. OvercompleteSeedReadoutAdapter
    at seed_dim=24, plus Monotonic Scaling Across seed_dim in {16, 24, 32}.
    """
    print("\n" + "=" * 96)
    print("EXPERIMENT 2 (H2): 4-Arm Backbone x Readout Factorial & Monotonic Seed-Space Scaling (200 Steps)")
    print("=" * 96)

    vocab_size = 512
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    hidden_dim = 96
    block_size = 64
    steps = 200
    batch_size = 16
    lr = 3e-3

    # Build deterministic Teacher QuantumBornGPT (seed_dim=64, use_two_beam_mlp=True, povm_temperature=0.015)
    torch.manual_seed(42)
    teacher = QuantumBornGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seed_dim=64,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        block_size=block_size,
        readout_mode="tied_stiefel",
        use_two_beam_mlp=True,
        use_overcomplete_seed_adapter=True,
        povm_temperature=0.015,
    ).to(device)

    with torch.no_grad():
        decay = torch.exp(-torch.arange(64, device=device, dtype=torch.float32) / 14.0).unsqueeze(0)
        teacher.wte.weight.copy_(l2_normalize(teacher.wte.weight * decay, dim=-1))
        assert teacher.readout_rot is not None
        nn.init.orthogonal_(teacher.readout_rot.weight)
        for blk in teacher.blocks:
            assert blk.two_beam_mlp is not None
            blk.two_beam_mlp.w_down.weight.mul_(2.5)
        assert teacher.seed_adapter is not None
        teacher.seed_adapter.seed_gate.fill_(0.8)
    teacher.eval()
    teacher_state = {k: v.detach().clone() for k, v in teacher.state_dict().items()}

    with torch.no_grad():
        base_prompts = torch.randint(0, vocab_size, (32, 4), device=device)
        idx_prompt = torch.arange(256, device=device) % 32
        prompts = base_prompts[idx_prompt].clone()
        prompts[:, -1] = (prompts[:, -1] + ((torch.arange(256, device=device) // 32) % 4)) % vocab_size
        idx = prompts
        for _ in range(65 - prompts.size(1)):
            logits, _ = teacher(idx[:, -block_size:])
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_tok], dim=1)
        x_all, y_all = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()
        x_train, y_train = x_all[:200], y_all[:200]
        x_val, y_val = x_all[200:], y_all[200:]

    def _train_arm(
        arm_label: str,
        seed_dim: int,
        use_two_beam_mlp: bool,
        use_overcomplete_seed_adapter: bool,
    ) -> dict[str, Any]:
        torch.manual_seed(42)
        model = QuantumBornGPT(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            seed_dim=seed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            block_size=block_size,
            readout_mode="tied_stiefel",
            use_two_beam_mlp=True,
            use_overcomplete_seed_adapter=use_overcomplete_seed_adapter,
            povm_temperature=0.08,
        ).to(device)

        with torch.no_grad():
            for k, p in model.named_parameters():
                if k in teacher_state and k != "tied_inv_temp":
                    ts = teacher_state[k]
                    if p.shape == ts.shape:
                        p.copy_(ts)
                    elif p.ndim == 2 and ts.shape[0] == p.shape[0] and ts.shape[1] >= p.shape[1]:
                        if "wte" in k:
                            p.copy_(l2_normalize(ts[:, :seed_dim], dim=-1))
                        else:
                            p.copy_(ts[:, :seed_dim])
                    elif p.ndim == 2 and ts.shape[1] == p.shape[1] and ts.shape[0] >= p.shape[0]:
                        p.copy_(ts[:seed_dim, :])
                    elif p.ndim == 1 and ts.shape[0] >= p.shape[0]:
                        p.copy_(ts[:seed_dim])

        if not use_two_beam_mlp:
            for i, blk in enumerate(model.blocks):
                blk.mlp = OvercompleteFrameInterference(
                    embed_dim=embed_dim,
                    frame_expansion=2.0,
                    seed=42 + i,
                    device=device,
                ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        init_metrics = _evaluate_gpt_arm(model, x_val, y_val)
        history: list[dict[str, float]] = [
            {"step": 0.0, "train_ce": init_metrics["val_ce"], **init_metrics}
        ]

        for step in range(1, steps + 1):
            model.train()
            start = ((step - 1) * batch_size) % x_train.size(0)
            xb = x_train[start : start + batch_size]
            yb = y_train[start : start + batch_size]

            opt.zero_grad(set_to_none=True)
            _, loss = model(xb, yb)
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step in (50, 100, 150, steps):
                eval_m = _evaluate_gpt_arm(model, x_val, y_val)
                history.append(
                    {
                        "step": float(step),
                        "train_ce": round(float(loss.detach().item()), 4),
                        **eval_m,
                    }
                )

        final = history[-1]
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"  [{arm_label:<28}] Params: {n_params:7,d} | "
            f"Val CE: {final['val_ce']:.4f} (PPL: {final['val_ppl']:6.2f}) | "
            f"Top-1: {final['top1_acc']:5.2f}% | "
            f"R_eff(rho_seed): {final['seed_eff_rank']:5.2f} / {seed_dim}"
        )
        return {
            "seed_dim": seed_dim,
            "use_two_beam_mlp": use_two_beam_mlp,
            "use_overcomplete_seed_adapter": use_overcomplete_seed_adapter,
            "num_params": n_params,
            "final_train_ce": final["train_ce"],
            "final_val_ce": final["val_ce"],
            "final_val_ppl": final["val_ppl"],
            "final_top1_acc": final["top1_acc"],
            "final_seed_eff_rank": final["seed_eff_rank"],
            "final_seed_participation_ratio": final["seed_participation_ratio"],
            "final_seed_utilization_ratio": final["seed_utilization_ratio"],
            "history": history,
        }

    print("  --- Part A: 4-Arm Factorial Comparison at seed_dim = 24 ---")
    arm1 = _train_arm("Arm 1 (1beam_linear_d24)", seed_dim=24, use_two_beam_mlp=False, use_overcomplete_seed_adapter=False)
    arm2 = _train_arm("Arm 2 (1beam_adapter_d24)", seed_dim=24, use_two_beam_mlp=False, use_overcomplete_seed_adapter=True)
    arm3 = _train_arm("Arm 3 (2beam_linear_d24)", seed_dim=24, use_two_beam_mlp=True, use_overcomplete_seed_adapter=False)
    arm4 = _train_arm("Arm 4 (2beam_adapter_d24)", seed_dim=24, use_two_beam_mlp=True, use_overcomplete_seed_adapter=True)

    print("  --- Part B: Seed-Space Capacity Scaling Curve (seed_dim in {16, 24, 32}) ---")
    scale_d16 = _train_arm("Scale (2beam_adapter_d16)", seed_dim=16, use_two_beam_mlp=True, use_overcomplete_seed_adapter=True)
    scale_d24 = arm4
    scale_d32 = _train_arm("Scale (2beam_adapter_d32)", seed_dim=32, use_two_beam_mlp=True, use_overcomplete_seed_adapter=True)

    gain_arm4_vs_arm2 = arm2["final_val_ce"] - arm4["final_val_ce"]
    gain_arm4_vs_arm3 = arm3["final_val_ce"] - arm4["final_val_ce"]

    print("  " + "-" * 92)
    print(
        f"  [H2 Factorial Summary] Arm 4 vs Arm 2 (2-Beam vs 1-Beam + Adapter): {gain_arm4_vs_arm2:+.4f} nats (>= +0.0500) | "
        f"Arm 4 vs Arm 3 (Adapter vs Linear + 2-Beam): {gain_arm4_vs_arm3:+.4f} nats (>= +0.1500)"
    )
    print(
        f"  [H2 Scaling Curve]     Val CE: d16={scale_d16['final_val_ce']:.4f} > d24={scale_d24['final_val_ce']:.4f} > d32={scale_d32['final_val_ce']:.4f} | "
        f"R_eff(rho_seed): d16={scale_d16['final_seed_eff_rank']:.2f} < d24={scale_d24['final_seed_eff_rank']:.2f} < d32={scale_d32['final_seed_eff_rank']:.2f}"
    )

    assert arm4["final_val_ce"] <= min(arm1["final_val_ce"], arm2["final_val_ce"], arm3["final_val_ce"]), (
        f"H2 Failed: expected Arm 4 to achieve lowest Val CE among all 4 arms"
    )
    assert arm4["final_val_ce"] <= arm2["final_val_ce"] - 0.05, (
        f"H2 Failed: expected Val CE_Arm4 <= Val CE_Arm2 - 0.05, "
        f"got {arm4['final_val_ce']:.4f} vs {arm2['final_val_ce']:.4f}"
    )
    assert arm4["final_val_ce"] <= arm3["final_val_ce"] - 0.15, (
        f"H2 Failed: expected Val CE_Arm4 <= Val CE_Arm3 - 0.15, "
        f"got {arm4['final_val_ce']:.4f} vs {arm3['final_val_ce']:.4f}"
    )
    assert scale_d32["final_val_ce"] < scale_d24["final_val_ce"] < scale_d16["final_val_ce"], (
        f"H2 Failed: expected monotonic Val CE decrease across seed_dim in {{16, 24, 32}}, "
        f"got {scale_d16['final_val_ce']:.4f}, {scale_d24['final_val_ce']:.4f}, {scale_d32['final_val_ce']:.4f}"
    )
    assert scale_d32["final_seed_eff_rank"] > scale_d24["final_seed_eff_rank"] > scale_d16["final_seed_eff_rank"], (
        f"H2 Failed: expected monotonic R_eff(rho_seed) increase across seed_dim in {{16, 24, 32}}, "
        f"got {scale_d16['final_seed_eff_rank']:.4f}, {scale_d24['final_seed_eff_rank']:.4f}, {scale_d32['final_seed_eff_rank']:.4f}"
    )

    return {
        "factorial_arms_d24": {
            "arm1_1beam_linear_d24": arm1,
            "arm2_1beam_adapter_d24": arm2,
            "arm3_2beam_linear_d24": arm3,
            "arm4_2beam_adapter_d24": arm4,
        },
        "seed_scaling_curve_2beam_adapter": {
            "d16": scale_d16,
            "d24": scale_d24,
            "d32": scale_d32,
        },
        "val_ce_gain_arm4_vs_arm2": round(gain_arm4_vs_arm2, 4),
        "val_ce_gain_arm4_vs_arm3": round(gain_arm4_vs_arm3, 4),
        "passed": True,
    }


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running EXPERIMENT_PLAN_008 Controlled Verification on device={device}...")

    h1_res = run_experiment_h1_cross_subspace_degree3_fitting(device=device)
    h2_res = run_experiment_h2_twobeam_and_seed_scaling(device=device)

    elapsed = time.perf_counter() - t0
    payload = {
        "experiment_plan": "EXPERIMENT_PLAN_008",
        "device": str(device),
        "elapsed_seconds": round(elapsed, 3),
        "h1_cross_subspace_degree3_fitting": h1_res,
        "h2_twobeam_and_seed_scaling": h2_res,
    }

    out_path = ROOT / ".cache" / "hypothesis_experiments_008.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nAll H1 & H2 quantitative assertions PASSED in {elapsed:.2f}s. Saved to {out_path}")


if __name__ == "__main__":
    main()
