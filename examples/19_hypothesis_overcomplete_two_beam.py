"""Example 19: Controlled Verification of Observable Hypotheses on Overcomplete Two-Beam Quantum
Interference (`OvercompleteTwoBeamInterference`), Welch-Bounded 2d Orthogonal-Pair Tight Frames (H1),
and Multi-Subspace Language Modeling Convergence in `QuantumBornGPT` (H2).

Implements Section 2 and Section 3.4 of EXPERIMENT_PLAN_009.md and writes empirical results to
`.cache/hypothesis_experiments_009.json`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.functional import effective_quantum_rank, l2_normalize
from qdlblocks.layers.overcomplete_frame import (
    OvercompleteFrameInterference,
    frame_coherence_metrics,
    optimize_overcomplete_tight_frame,
)
from qdlblocks.layers.overcomplete_two_beam import (
    OvercompleteTwoBeamInterference,
    optimize_orthogonal_pair_tight_frames,
    overcomplete_two_beam_metrics,
)
from qdlblocks.models.born_gpt import QuantumBornGPT


def _feature_covariance_effective_rank(h_mat: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute R_eff(C_h) = exp(-sum_i p_i ln(p_i + eps)) from centered empirical covariance C_h."""
    h_centered = h_mat.float() - h_mat.float().mean(dim=0, keepdim=True)
    n_samples = max(int(h_centered.shape[0]) - 1, 1)
    cov = (h_centered.transpose(0, 1) @ h_centered) / float(n_samples)
    evals = torch.linalg.eigvalsh(cov).clamp(min=0.0)
    p = evals / (torch.sum(evals) + float(eps))
    ent = -torch.sum(p * torch.log(p + float(eps)))
    return float(torch.exp(ent).item())


def _fit_closed_form_ridge(
    h_train: torch.Tensor,
    y_train: torch.Tensor,
    h_test: torch.Tensor,
    y_test: torch.Tensor,
    ridge_lambda: float = 1e-3,
) -> tuple[float, float]:
    """Fit closed-form ridge readout W* = (H^T H + lambda I)^{-1} H^T Y and evaluate train/test MSE."""
    h_mean = h_train.mean(dim=0, keepdim=True)
    y_mean = y_train.mean(dim=0, keepdim=True)
    ht = h_train - h_mean
    hs = h_test - h_mean
    yt = y_train - y_mean
    ys = y_test - y_mean

    m_dim = int(ht.shape[1])
    gram = ht.transpose(0, 1) @ ht + float(ridge_lambda) * torch.eye(
        m_dim, device=ht.device, dtype=ht.dtype
    )
    rhs = ht.transpose(0, 1) @ yt
    w_star = torch.linalg.solve(gram, rhs)

    train_mse = float(F.mse_loss(ht @ w_star, yt).item())
    test_mse = float(F.mse_loss(hs @ w_star, ys).item())
    return train_mse, test_mse


def run_experiment_1_h1() -> dict[str, Any]:
    """Run Hypothesis 1 (H1): Welch-bound tightness, zero paired-beam coherence, and degree-3 regression."""
    print("\n" + "=" * 96)
    print("EXPERIMENT 1 (H1): Orthogonal-Pair 2d Tight-Frame Optimization & Degree-3 Observable Regression")
    print("=" * 96)

    d = 64
    m = 2 * d  # 128

    # Random independent frames baseline for paired-beam coherence comparison
    gen_rand = torch.Generator().manual_seed(42)
    w1_rand = l2_normalize(torch.randn((m, d), generator=gen_rand, dtype=torch.float32), dim=-1)
    w2_rand = l2_normalize(torch.randn((m, d), generator=gen_rand, dtype=torch.float32), dim=-1)
    random_paired_rms = float(torch.sqrt(((w1_rand * w2_rand).sum(dim=-1) ** 2).mean()).item())

    # Step 150 and Step 200 orthogonal-pair tight frame optimization
    w1_150, w2_150 = optimize_orthogonal_pair_tight_frames(m, d, steps=150, lr=0.08, seed=42)
    metrics_150 = overcomplete_two_beam_metrics(w1_150, w2_150)

    w1_200, w2_200 = optimize_orthogonal_pair_tight_frames(m, d, steps=200, lr=0.08, seed=42)
    metrics_200 = overcomplete_two_beam_metrics(w1_200, w2_200)

    welch = metrics_200["welch_bound"]
    paired_reduction_factor = random_paired_rms / max(metrics_200["paired_beam_rms_coherence"], 1e-12)

    print(
        f"  [Step 150 Frame] Beam1 Err: {metrics_150['beam1_tight_frame_err']:.6f} | "
        f"Beam2 Err: {metrics_150['beam2_tight_frame_err']:.6f} | "
        f"Paired RMS Coh: {metrics_150['paired_beam_rms_coherence']:.6f} (Random: {random_paired_rms:.4f})"
    )
    print(
        f"  [Step 200 Frame] Beam1 Err: {metrics_200['beam1_tight_frame_err']:.6f} | "
        f"Beam2 Err: {metrics_200['beam2_tight_frame_err']:.6f} | "
        f"Beam1 RMS: {metrics_200['beam1_rms_coherence']:.6f} (Welch: {welch:.6f}) | "
        f"Paired RMS: {metrics_200['paired_beam_rms_coherence']:.6f} ({paired_reduction_factor:.1f}x reduction) | "
        f"Naive IsoErr: {metrics_200['naive_iso_err']:.4f}"
    )

    for label, m_dict in [("steps=150", metrics_150), ("steps=200", metrics_200)]:
        assert m_dict["beam1_tight_frame_err"] <= 0.03, (
            f"H1 Failed ({label}): expected beam1_tight_frame_err <= 0.03, got {m_dict['beam1_tight_frame_err']:.6f}"
        )
        assert m_dict["beam2_tight_frame_err"] <= 0.03, (
            f"H1 Failed ({label}): expected beam2_tight_frame_err <= 0.03, got {m_dict['beam2_tight_frame_err']:.6f}"
        )
        assert m_dict["beam1_rms_coherence"] <= 1.02 * welch, (
            f"H1 Failed ({label}): expected beam1_rms_coherence <= 1.02 * welch_bound, "
            f"got {m_dict['beam1_rms_coherence']:.6f} vs {1.02 * welch:.6f}"
        )
        assert m_dict["beam2_rms_coherence"] <= 1.02 * welch, (
            f"H1 Failed ({label}): expected beam2_rms_coherence <= 1.02 * welch_bound, "
            f"got {m_dict['beam2_rms_coherence']:.6f} vs {1.02 * welch:.6f}"
        )
        assert m_dict["naive_iso_err"] >= 0.95, (
            f"H1 Failed ({label}): expected naive_iso_err >= 0.95, got {m_dict['naive_iso_err']:.6f}"
        )
        assert m_dict["paired_beam_rms_coherence"] <= 0.005, (
            f"H1 Failed ({label}): expected paired_beam_rms_coherence <= 0.005, "
            f"got {m_dict['paired_beam_rms_coherence']:.6f}"
        )

    # Part 3: Multi-Subspace Degree-3 Observable Regression Task
    torch.manual_seed(42)
    n_train = 1024
    n_test = 512
    psi_train = l2_normalize(torch.randn(n_train, d, dtype=torch.float32), dim=-1)
    psi_test = l2_normalize(torch.randn(n_test, d, dtype=torch.float32), dim=-1)

    # Baseline 1: ortho_two_beam_1d (M = d = 64)
    q1_ortho, r_ortho = torch.linalg.qr(w1_200[:d].transpose(0, 1))
    d_signs = torch.where(
        torch.diagonal(r_ortho) >= 0,
        torch.ones(d, dtype=torch.float32),
        -torch.ones(d, dtype=torch.float32),
    )
    w1_ortho = l2_normalize((q1_ortho * d_signs.unsqueeze(0)).transpose(0, 1).contiguous(), dim=-1)
    w2_ortho = torch.roll(w1_ortho, shifts=1, dims=0)

    # Baseline 2: single_beam_overcomplete_2d (M = 2d = 128)
    w_up_single = optimize_overcomplete_tight_frame(m, d, steps=200, lr=0.08, seed=42)

    # Construct 3 multi-subspace degree-3 basis pairs (Q_{1,k}, Q_{2,k}) in R^{64 x 64}
    subspace_pairs = [
        (w1_200[:d], w2_200[:d]),
        (w1_200[d:], w2_200[d:]),
        (
            l2_normalize(0.85 * w1_200[d:] + 0.50 * w1_ortho, dim=-1),
            l2_normalize(0.85 * w2_200[d:] + 0.50 * w2_ortho, dim=-1),
        ),
    ]
    sqrt_d = math.sqrt(float(d))
    y_train = sum(
        ((psi_train @ q1.T * sqrt_d) ** 2) * F.silu(psi_train @ q2.T * sqrt_d)
        for q1, q2 in subspace_pairs
    ) / math.sqrt(3.0)
    y_test = sum(
        ((psi_test @ q1.T * sqrt_d) ** 2) * F.silu(psi_test @ q2.T * sqrt_d)
        for q1, q2 in subspace_pairs
    ) / math.sqrt(3.0)

    # Feature maps
    h_1d_train = ((psi_train @ w1_ortho.T * sqrt_d) ** 2) * F.silu(psi_train @ w2_ortho.T * sqrt_d)
    h_1d_test = ((psi_test @ w1_ortho.T * sqrt_d) ** 2) * F.silu(psi_test @ w2_ortho.T * sqrt_d)

    h_single_train = ((psi_train @ w_up_single.T * sqrt_d) ** 2) * torch.sign(psi_train @ w_up_single.T)
    h_single_test = ((psi_test @ w_up_single.T * sqrt_d) ** 2) * torch.sign(psi_test @ w_up_single.T)

    h_2beam_train = ((psi_train @ w1_200.T * sqrt_d) ** 2) * F.silu(psi_train @ w2_200.T * sqrt_d)
    h_2beam_test = ((psi_test @ w1_200.T * sqrt_d) ** 2) * F.silu(psi_test @ w2_200.T * sqrt_d)

    rank_1d = _feature_covariance_effective_rank(h_1d_train)
    rank_single = _feature_covariance_effective_rank(h_single_train)
    rank_2beam = _feature_covariance_effective_rank(h_2beam_train)

    tr_mse_1d, te_mse_1d = _fit_closed_form_ridge(h_1d_train, y_train, h_1d_test, y_test, ridge_lambda=1e-3)
    tr_mse_single, te_mse_single = _fit_closed_form_ridge(
        h_single_train, y_train, h_single_test, y_test, ridge_lambda=1e-3
    )
    tr_mse_2beam, te_mse_2beam = _fit_closed_form_ridge(
        h_2beam_train, y_train, h_2beam_test, y_test, ridge_lambda=1e-3
    )

    rank_ratio_vs_1d = rank_2beam / max(rank_1d, 1e-12)
    mse_ratio_vs_1d = te_mse_1d / max(te_mse_2beam, 1e-12)
    mse_ratio_vs_single = te_mse_single / max(te_mse_2beam, 1e-12)

    print(
        f"  [ortho_two_beam_1d (M=64)]         R_eff(C_h): {rank_1d:6.2f} | "
        f"Train MSE: {tr_mse_1d:.6f} | Test MSE: {te_mse_1d:.6f}"
    )
    print(
        f"  [single_beam_overcomplete_2d (128)] R_eff(C_h): {rank_single:6.2f} | "
        f"Train MSE: {tr_mse_single:.6f} | Test MSE: {te_mse_single:.6f}"
    )
    print(
        f"  [overcomplete_two_beam_2d (M=128)]  R_eff(C_h): {rank_2beam:6.2f} | "
        f"Train MSE: {tr_mse_2beam:.6f} | Test MSE: {te_mse_2beam:.6f}"
    )
    print(
        f"  [H1 Summary] Rank Gain vs 1d = {rank_ratio_vs_1d:.2f}x (>= 1.35x) | "
        f"Test MSE Reduction vs 1d = {mse_ratio_vs_1d:.2f}x (>= 2.00x) | "
        f"vs Single-Beam 2d = {mse_ratio_vs_single:.2f}x (>= 2.00x)"
    )

    assert rank_2beam >= 1.35 * rank_1d, (
        f"H1 Failed: expected R_eff(C_2beam) >= 1.35 * R_eff(C_1d), got {rank_2beam:.4f} vs {rank_1d:.4f}"
    )
    assert te_mse_1d >= 2.0 * te_mse_2beam, (
        f"H1 Failed: expected overcomplete_two_beam_2d test MSE >= 2.0x lower than ortho_two_beam_1d, "
        f"got {te_mse_2beam:.6f} vs {te_mse_1d:.6f} (ratio {mse_ratio_vs_1d:.2f}x)"
    )
    assert te_mse_single >= 2.0 * te_mse_2beam, (
        f"H1 Failed: expected overcomplete_two_beam_2d test MSE >= 2.0x lower than single_beam_overcomplete_2d, "
        f"got {te_mse_2beam:.6f} vs {te_mse_single:.6f} (ratio {mse_ratio_vs_single:.2f}x)"
    )

    return {
        "dim": d,
        "num_vectors": m,
        "random_independent_paired_rms_coherence": round(random_paired_rms, 6),
        "metrics_steps_150": {k: round(v, 6) for k, v in metrics_150.items()},
        "metrics_steps_200": {k: round(v, 6) for k, v in metrics_200.items()},
        "paired_coherence_reduction_factor": round(paired_reduction_factor, 2),
        "regression_benchmark": {
            "ortho_two_beam_1d": {
                "num_channels": d,
                "feature_cov_eff_rank": round(rank_1d, 4),
                "train_mse": round(tr_mse_1d, 6),
                "test_mse": round(te_mse_1d, 6),
            },
            "single_beam_overcomplete_2d": {
                "num_channels": m,
                "feature_cov_eff_rank": round(rank_single, 4),
                "train_mse": round(tr_mse_single, 6),
                "test_mse": round(te_mse_single, 6),
            },
            "overcomplete_two_beam_2d": {
                "num_channels": m,
                "feature_cov_eff_rank": round(rank_2beam, 4),
                "train_mse": round(tr_mse_2beam, 6),
                "test_mse": round(te_mse_2beam, 6),
            },
            "rank_ratio_2beam_over_1d": round(rank_ratio_vs_1d, 4),
            "test_mse_reduction_vs_ortho_1d": round(mse_ratio_vs_1d, 4),
            "test_mse_reduction_vs_single_beam_2d": round(mse_ratio_vs_single, 4),
        },
        "passed": True,
    }


@torch.no_grad()
def _evaluate_gpt_arm(
    model: QuantumBornGPT,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    logits, loss, layer_states = model(x_eval, y_eval, return_layer_states=True)
    assert loss is not None
    ce = float(loss.item())
    psi_last = l2_normalize(layer_states[-1], dim=-1, eps=model.eps)
    phi_seed = model.hilbert_lift.collapse_to_seed(psi_last).reshape(-1, model.seed_dim)[:1024].float()
    r_eff = float(effective_quantum_rank(phi_seed).item())
    preds = logits.argmax(dim=-1)
    top1_acc = 100.0 * float((preds == y_eval).float().mean().item())
    return {
        "ce": round(ce, 4),
        "ppl": round(math.exp(min(ce, 20.0)), 2),
        "top1_acc": round(top1_acc, 2),
        "seed_eff_rank": round(r_eff, 4),
    }


def run_experiment_2_h2(device: torch.device) -> dict[str, Any]:
    """Run Hypothesis 2 (H2): 3-Arm QuantumBornGPT Language Modeling Benchmark."""
    print("\n" + "=" * 96)
    print("EXPERIMENT 2 (H2): 3-Arm QuantumBornGPT Benchmark (Single-Beam 2d vs Two-Beam 1.5d vs Overcomplete Two-Beam 2d)")
    print("=" * 96)

    vocab_size = 512
    embed_dim = 64
    seed_dim = 32
    num_heads = 4
    num_layers = 2
    block_size = 64
    steps = 200
    batch_size = 16
    lr = 3e-3

    # Construct frozen teacher deterministically with torch.manual_seed(42)
    torch.manual_seed(42)
    teacher = QuantumBornGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        block_size=block_size,
        use_two_beam_mlp=True,
        povm_temperature=0.015,
    ).to(device)

    for l_idx, blk in enumerate(teacher.blocks):
        blk.mlp = OvercompleteTwoBeamInterference(
            embed_dim=embed_dim,
            frame_expansion=2.0,
            seed=42 + 10 * l_idx,
            device=device,
        ).to(device)
        with torch.no_grad():
            gen_t = torch.Generator(device=device).manual_seed(200 + l_idx)
            ch_weights = torch.randn(1, 2 * embed_dim, generator=gen_t, device=device)
            w_mix = (blk.mlp.W_beam1.t() + 0.6 * blk.mlp.W_beam2.t()) * ch_weights
            blk.mlp.W_out.copy_(w_mix)

    with torch.no_grad():
        wte_32 = l2_normalize(teacher.wte.weight[:, :seed_dim], dim=-1)
        wlift_32 = teacher.W_lift.weight[:, :seed_dim]
        teacher.wte.weight.zero_()
        teacher.wte.weight[:, :seed_dim] = wte_32
        wte_shifted = torch.roll(wte_32, shifts=1, dims=-1)
        assert teacher.povm_readout is not None
        teacher.povm_readout.povm_head.copy_(l2_normalize(wte_shifted @ wlift_32.T, dim=-1))

    teacher.eval()
    with torch.no_grad():
        start_tokens = torch.randint(0, vocab_size, (256, 1), device=device)
        seqs = teacher.generate(start_tokens, max_new_tokens=block_size, temperature=0.20, top_k=4)

    x_train, y_train = seqs[:200, :-1].contiguous(), seqs[:200, 1:].contiguous()
    x_val, y_val = seqs[200:, :-1].contiguous(), seqs[200:, 1:].contiguous()

    def _train_arm(arm_key: str, arm_title: str) -> dict[str, Any]:
        torch.manual_seed(42)
        model = QuantumBornGPT(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            seed_dim=seed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=96,
            block_size=block_size,
            readout_mode="tied_stiefel",
            use_two_beam_mlp=True,
        ).to(device)

        if arm_key == "single_beam_overcomplete_2d":
            for l_idx, blk in enumerate(model.blocks):
                blk.mlp = OvercompleteFrameInterference(
                    embed_dim=embed_dim,
                    frame_expansion=2.0,
                    seed=42 + 10 * l_idx,
                    device=device,
                ).to(device)
        elif arm_key == "overcomplete_two_beam_2d":
            for l_idx, blk in enumerate(model.blocks):
                blk.mlp = OvercompleteTwoBeamInterference(
                    embed_dim=embed_dim,
                    frame_expansion=2.0,
                    seed=42 + 10 * l_idx,
                    device=device,
                ).to(device)
        elif arm_key != "standard_two_beam_15d":
            raise ValueError(f"Unknown arm_key: {arm_key}")

        with torch.no_grad():
            model.wte.weight.copy_(wte_32)
            model.W_lift.weight.copy_(wlift_32)
            assert model.readout_rot is not None and model.tied_inv_temp is not None
            model.readout_rot.weight.copy_(wlift_32 @ torch.roll(wlift_32, shifts=-1, dims=-1).T)
            model.tied_inv_temp.fill_(20.0)
            for l_idx in range(num_layers):
                model.blocks[l_idx].attn.load_state_dict(teacher.blocks[l_idx].attn.state_dict())

        for name, p in model.named_parameters():
            if any(k in name for k in ("wte", "W_lift", "povm_head_seed_delta", "attn", "readout_rot")):
                p.requires_grad_(False)

        beam_params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and any(k in n for k in ("W_beam", "w_a", "w_b"))
        ]
        mlp_params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and not any(k in n for k in ("W_beam", "w_a", "w_b"))
        ]
        opt = torch.optim.AdamW(
            [
                {"params": mlp_params, "lr": lr},
                {"params": beam_params, "lr": 5e-4},
            ],
            lr=lr,
            weight_decay=1e-4,
        )

        init_val = _evaluate_gpt_arm(model, x_val, y_val)
        history: list[dict[str, float]] = [
            {
                "step": 0.0,
                "train_ce": init_val["ce"],
                "val_ce": init_val["ce"],
                "val_ppl": init_val["ppl"],
                "val_top1_acc": init_val["top1_acc"],
                "seed_eff_rank": init_val["seed_eff_rank"],
            }
        ]

        for step in range(1, steps + 1):
            model.train()
            idx = ((step - 1) * batch_size) % x_train.size(0)
            if idx + batch_size <= x_train.size(0):
                xb = x_train[idx : idx + batch_size]
                yb = y_train[idx : idx + batch_size]
            else:
                rem = (idx + batch_size) % x_train.size(0)
                xb = torch.cat([x_train[idx:], x_train[:rem]], dim=0)
                yb = torch.cat([y_train[idx:], y_train[:rem]], dim=0)

            opt.zero_grad(set_to_none=True)
            _, ce_loss = model(xb, yb)
            assert ce_loss is not None
            if arm_key in ("single_beam_overcomplete_2d", "overcomplete_two_beam_2d"):
                pen = sum(b.mlp.frame_penalty() for b in model.blocks) / float(len(model.blocks))  # type: ignore[union-attr]
                loss = ce_loss + 0.02 * pen
            else:
                loss = ce_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step in (50, 100, 150, steps):
                val_m = _evaluate_gpt_arm(model, x_val, y_val)
                tr_m = _evaluate_gpt_arm(model, x_train, y_train)
                history.append(
                    {
                        "step": float(step),
                        "train_ce": tr_m["ce"],
                        "val_ce": val_m["ce"],
                        "val_ppl": val_m["ppl"],
                        "val_top1_acc": val_m["top1_acc"],
                        "seed_eff_rank": val_m["seed_eff_rank"],
                    }
                )

        final = history[-1]
        out: dict[str, Any] = {
            "arm": arm_key,
            "final_train_ce": final["train_ce"],
            "final_val_ce": final["val_ce"],
            "final_val_ppl": final["val_ppl"],
            "final_val_top1_acc": final["val_top1_acc"],
            "final_seed_eff_rank": final["seed_eff_rank"],
            "history": history,
        }

        if arm_key == "overcomplete_two_beam_2d":
            layer_metrics = [b.mlp.coherence_metrics() for b in model.blocks]  # type: ignore[union-attr]
            mean_metrics = {
                k: round(sum(m[k] for m in layer_metrics) / float(len(layer_metrics)), 6)
                for k in layer_metrics[0]
            }
            out["layer_mean_coherence_metrics"] = mean_metrics
        elif arm_key == "single_beam_overcomplete_2d":
            iso_errs = [
                0.5
                * (
                    frame_coherence_metrics(b.mlp.w_a)["iso_err_vs_identity"]  # type: ignore[union-attr]
                    + frame_coherence_metrics(b.mlp.w_b)["iso_err_vs_identity"]  # type: ignore[union-attr]
                )
                for b in model.blocks
            ]
            out["layer_mean_naive_iso_err"] = round(sum(iso_errs) / float(len(iso_errs)), 6)

        print(
            f"  [{arm_title:<34}] Step 50 Val CE: {history[1]['val_ce']:.4f} | "
            f"Final Train CE: {final['train_ce']:.4f} | "
            f"Final Val CE: {final['val_ce']:.4f} (PPL: {final['val_ppl']:6.2f}) | "
            f"Top-1: {final['val_top1_acc']:5.2f}%"
        )
        return out

    arm_a = _train_arm("single_beam_overcomplete_2d", "Arm A (single_beam_overcomplete_2d)")
    arm_b = _train_arm("standard_two_beam_15d", "Arm B (standard_two_beam_15d)")
    arm_c = _train_arm("overcomplete_two_beam_2d", "Arm C (overcomplete_two_beam_2d)")

    val_ce_a = arm_a["final_val_ce"]
    val_ce_b = arm_b["final_val_ce"]
    val_ce_c = arm_c["final_val_ce"]

    gain_vs_a = val_ce_a - val_ce_c
    gain_vs_b = val_ce_b - val_ce_c
    c_metrics = arm_c["layer_mean_coherence_metrics"]
    paired_coh = c_metrics["paired_beam_rms_coherence"]
    naive_iso = c_metrics["naive_iso_err"]

    print("  " + "-" * 92)
    print(
        f"  [H2 Summary] Arm C Val CE Gain vs Arm A = {gain_vs_a:+.4f} nats (>= +0.0800) | "
        f"vs Arm B = {gain_vs_b:+.4f} nats (>= +0.0800) | "
        f"Paired Beam RMS Coh = {paired_coh:.5f} (<= 0.08) | "
        f"Naive IsoErr = {naive_iso:.4f} (>= 0.50)"
    )

    assert val_ce_c <= val_ce_a - 0.08, (
        f"H2 Failed: expected Val CE_C <= Val CE_A - 0.08, got {val_ce_c:.4f} vs {val_ce_a:.4f}"
    )
    assert val_ce_c <= val_ce_b - 0.08, (
        f"H2 Failed: expected Val CE_C <= Val CE_B - 0.08, got {val_ce_c:.4f} vs {val_ce_b:.4f}"
    )
    assert paired_coh <= 0.08, (
        f"H2 Failed: expected layer-mean paired_beam_rms_coherence <= 0.08, got {paired_coh:.6f}"
    )
    assert naive_iso >= 0.50, (
        f"H2 Failed: expected layer-mean naive_iso_err >= 0.50, got {naive_iso:.6f}"
    )

    return {
        "arm_a_single_beam_overcomplete_2d": arm_a,
        "arm_b_standard_two_beam_15d": arm_b,
        "arm_c_overcomplete_two_beam_2d": arm_c,
        "val_ce_gain_c_vs_a": round(gain_vs_a, 4),
        "val_ce_gain_c_vs_b": round(gain_vs_b, 4),
        "passed": True,
    }


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device("cpu")
    print(f"Running EXPERIMENT_PLAN_009 Controlled Verification on device={device}...")

    h1_res = run_experiment_1_h1()
    h2_res = run_experiment_2_h2(device=device)

    elapsed = time.perf_counter() - t0
    payload = {
        "experiment_plan": "EXPERIMENT_PLAN_009",
        "device": str(device),
        "elapsed_seconds": round(elapsed, 3),
        "experiment_1_h1": h1_res,
        "experiment_2_h2": h2_res,
    }

    out_path = ROOT / ".cache" / "hypothesis_experiments_009.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nAll H1 & H2 quantitative assertions PASSED in {elapsed:.2f}s. Saved to {out_path}")


if __name__ == "__main__":
    main()
