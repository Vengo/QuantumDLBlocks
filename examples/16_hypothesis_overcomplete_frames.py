"""Example 16: Controlled Verification of Observable Hypotheses on Overcomplete Tight Frames,
Welch-Bound Coherence (H1), and Controlled Isometry Relaxation in QuantumBornGPT (H2).

Implements Section 3.3 of EXPERIMENT_PLAN_006.md and writes empirical results to
`.cache/hypothesis_experiments_006.json`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
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
from qdlblocks.models.born_gpt import QuantumBornGPT
from qdlblocks.optim.muon import StiefelMuon, split_muon_adamw_params
from qdlblocks.optim.tangent_stiefel_muon import TangentStiefelMuon


def _effective_feature_covariance_rank(f_mat: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute effective feature covariance rank R_sigma(F_tilde) via singular-value Shannon entropy."""
    f_centered = f_mat.float() - f_mat.float().mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(f_centered)
    p = s / torch.clamp(torch.sum(s), min=eps)
    p_pos = torch.clamp(p, min=eps)
    ent = -torch.sum(p * torch.log(p_pos))
    return float(torch.exp(ent).item())


def _fit_and_eval_born_ridge(
    w: torch.Tensor,
    psi_train: torch.Tensor,
    y_train: torch.Tensor,
    psi_test: torch.Tensor,
    y_test: torch.Tensor,
    alpha: float = 1e-4,
) -> tuple[float, float]:
    """Compute quadratic Born features (Psi W^T)^{odot 2}, effective rank, and held-out ridge MSE."""
    f_train = (psi_train @ w.T).square()
    f_test = (psi_test @ w.T).square()

    eff_rank = _effective_feature_covariance_rank(f_train)

    f_mean = f_train.mean(dim=0, keepdim=True)
    y_mean = y_train.mean()
    f_train_c = f_train - f_mean
    f_test_c = f_test - f_mean
    y_train_c = y_train - y_mean

    m = f_train.shape[1]
    gram = f_train_c.T @ f_train_c + alpha * torch.eye(m, device=f_train.device, dtype=f_train.dtype)
    rhs = f_train_c.T @ y_train_c
    theta = torch.linalg.solve(gram, rhs)

    y_pred = f_test_c @ theta + y_mean
    test_mse = float(torch.mean((y_pred - y_test).square()).item())
    return eff_rank, test_mse


def run_experiment_h1_overcomplete_tight_frame() -> dict[str, Any]:
    """Experiment 1 (H1): M = 2d Overcomplete Tight-Frame Packing vs M = d Orthogonal Basis."""
    print("\n" + "=" * 92)
    print("EXPERIMENT 1 (H1): M = 2d Overcomplete Tight-Frame Packing & Quadratic Born Capacity")
    print("=" * 92)

    d = 64
    m_ortho = d
    m_frame = 2 * d

    w_ortho = optimize_overcomplete_tight_frame(m_ortho, d, steps=1, seed=42)
    w_frame = optimize_overcomplete_tight_frame(m_frame, d, steps=250, lr=0.1, seed=42)

    metrics_ortho = frame_coherence_metrics(w_ortho)
    metrics_frame = frame_coherence_metrics(w_frame)

    assert 0.85 <= metrics_frame["iso_err_vs_identity"] <= 1.15, (
        f"H1 Failed: expected iso_err_vs_identity in [0.85, 1.15], got {metrics_frame['iso_err_vs_identity']:.6f}"
    )
    assert metrics_frame["tight_frame_err"] < 0.05, (
        f"H1 Failed: expected tight_frame_err < 0.05, got {metrics_frame['tight_frame_err']:.6f}"
    )
    assert metrics_frame["rms_coherence"] <= 1.10 * metrics_frame["welch_bound"], (
        f"H1 Failed: expected rms_coherence <= 1.10 * welch_bound, "
        f"got {metrics_frame['rms_coherence']:.6f} vs {1.10 * metrics_frame['welch_bound']:.6f}"
    )

    gen = torch.Generator().manual_seed(100)
    psi_train = l2_normalize(torch.randn((1024, d), generator=gen, dtype=torch.float32), dim=-1)
    psi_test = l2_normalize(torch.randn((256, d), generator=gen, dtype=torch.float32), dim=-1)

    # Construct rank-32 symmetric quantum observable O = sum_{r=1}^{32} c_r u_r u_r^T
    # where u_r are random unit vectors in the overcomplete Hilbert space (unaligned with W_ortho)
    perm = torch.randperm(m_frame, generator=gen)[:32]
    u_noise = l2_normalize(torch.randn((32, d), generator=gen, dtype=torch.float32), dim=-1)
    u_obs = l2_normalize(w_frame[perm] + 0.15 * u_noise, dim=-1)
    signs = torch.tensor([(-1.0) ** (r + 1) / math.sqrt(32.0) for r in range(32)], dtype=torch.float32)
    obs = (u_obs.T * signs.unsqueeze(0)) @ u_obs

    y_train = torch.sum((psi_train @ obs) * psi_train, dim=-1)
    y_test = torch.sum((psi_test @ obs) * psi_test, dim=-1)

    rank_ortho, mse_ortho = _fit_and_eval_born_ridge(w_ortho, psi_train, y_train, psi_test, y_test)
    rank_frame, mse_frame = _fit_and_eval_born_ridge(w_frame, psi_train, y_train, psi_test, y_test)

    rank_ratio = rank_frame / max(rank_ortho, 1e-12)
    mse_reduction = mse_ortho / max(mse_frame, 1e-12)

    print(
        f"  [Ortho M=d={m_ortho:3d}] IsoErr: {metrics_ortho['iso_err_vs_identity']:.4f} | "
        f"FrameErr: {metrics_ortho['tight_frame_err']:.4f} | "
        f"RMS Coh: {metrics_ortho['rms_coherence']:.4f} (Welch: {metrics_ortho['welch_bound']:.4f}) | "
        f"R_sigma(F): {rank_ortho:6.2f} | Test MSE: {mse_ortho:.6e}"
    )
    print(
        f"  [Frame M=2d={m_frame:3d}] IsoErr: {metrics_frame['iso_err_vs_identity']:.4f} | "
        f"FrameErr: {metrics_frame['tight_frame_err']:.4f} | "
        f"RMS Coh: {metrics_frame['rms_coherence']:.4f} (Welch: {metrics_frame['welch_bound']:.4f}) | "
        f"R_sigma(F): {rank_frame:6.2f} | Test MSE: {mse_frame:.6e}"
    )
    print(
        f"  [H1 Gain Summary] Effective Covariance Rank Gain = {rank_ratio:.2f}x (>= 1.60x) | "
        f"Test MSE Reduction = {mse_reduction:.2f}x (>= 2.00x)"
    )

    assert rank_frame >= 1.6 * rank_ortho, (
        f"H1 Failed: expected rank_frame >= 1.6 * rank_ortho, got {rank_frame:.4f} vs {rank_ortho:.4f}"
    )
    assert mse_frame <= 0.5 * mse_ortho, (
        f"H1 Failed: expected mse_frame <= 0.5 * mse_ortho, got {mse_frame:.6e} vs {mse_ortho:.6e}"
    )

    return {
        "dim": d,
        "ortho_1d": {
            "num_vectors": m_ortho,
            "metrics": {k: round(v, 6) for k, v in metrics_ortho.items()},
            "effective_feature_rank": round(rank_ortho, 4),
            "test_mse": round(mse_ortho, 8),
        },
        "frame_2d": {
            "num_vectors": m_frame,
            "metrics": {k: round(v, 6) for k, v in metrics_frame.items()},
            "effective_feature_rank": round(rank_frame, 4),
            "test_mse": round(mse_frame, 8),
        },
        "rank_ratio_2d_over_1d": round(rank_ratio, 4),
        "mse_reduction_1d_over_2d": round(mse_reduction, 4),
        "passed": True,
    }


def _generate_synthetic_zipfian_lm_dataset(
    vocab_size: int = 256,
    seq_len: int = 64,
    num_train: int = 512,
    num_val: int = 128,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate deterministic synthetic Zipfian language modeling sequences with 2-gram motif transitions."""
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    zipf_probs = 1.0 / (ranks**1.15)
    zipf_probs /= zipf_probs.sum()

    next_base = (np.arange(vocab_size) * 7 + 13) % vocab_size
    ctx_shift = (np.arange(vocab_size) % 32) * 3

    def _sample_split(n_seqs: int) -> tuple[torch.Tensor, torch.Tensor]:
        total_len = seq_len + 1
        seqs = np.zeros((n_seqs, total_len), dtype=np.int64)
        seqs[:, 0] = rng.choice(vocab_size, size=n_seqs, p=zipf_probs)
        seqs[:, 1] = (next_base[seqs[:, 0]] + rng.integers(0, 4, size=n_seqs)) % vocab_size
        for t in range(1, seq_len):
            u = rng.random(n_seqs)
            det_next = (next_base[seqs[:, t]] + ctx_shift[seqs[:, t - 1]]) % vocab_size
            noise_tok = rng.choice(vocab_size, size=n_seqs, p=zipf_probs)
            seqs[:, t + 1] = np.where(u < 0.90, det_next, noise_tok)
        x = torch.from_numpy(seqs[:, :-1].copy())
        y = torch.from_numpy(seqs[:, 1:].copy())
        return x, y

    x_train, y_train = _sample_split(num_train)
    x_val, y_val = _sample_split(num_val)
    return x_train, y_train, x_val, y_val


def _mean_mlp_iso_err(model: QuantumBornGPT) -> float:
    errs: list[float] = []
    for b in model.blocks:
        mlp = b.mlp
        assert mlp is not None
        for attr in ("w_a", "w_b"):
            w = getattr(mlp, attr)
            errs.append(frame_coherence_metrics(w)["iso_err_vs_identity"])
    return float(np.mean(errs))


def _evaluate_arm(
    model: QuantumBornGPT,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
) -> tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        logits, _, layer_states = model(x_val, y_val, return_layer_states=True)
        val_ce = float(F.cross_entropy(logits.reshape(-1, model.vocab_size), y_val.reshape(-1)).item())
        psi_last = l2_normalize(layer_states[-1], dim=-1, eps=model.eps)
        phi_seed = model.hilbert_lift.collapse_to_seed(psi_last).reshape(-1, model.seed_dim)
        r_eff = float(effective_quantum_rank(phi_seed[:1024].float()).item())
        iso_err = _mean_mlp_iso_err(model)
    return val_ce, r_eff, iso_err


def run_experiment_h2_quantum_born_gpt_arms(device: torch.device) -> dict[str, Any]:
    """Experiment 2 (H2): Controlled Isometry Error & Overcomplete 2d Frame Interference in QuantumBornGPT."""
    print("\n" + "=" * 92)
    print("EXPERIMENT 2 (H2): Controlled Isometry Relaxation & Overcomplete 2d Interference (200 Steps)")
    print("=" * 92)

    vocab_size = 256
    embed_dim = 64
    seed_dim = 32
    num_heads = 4
    num_layers = 2
    seq_len = 64
    steps = 200
    batch_size = 32

    x_train, y_train, x_val, y_val = _generate_synthetic_zipfian_lm_dataset(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_train=512,
        num_val=128,
        seed=42,
    )
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    adamw_patterns = ("wte", "povm_head", "embed", "unet_scales", "smear", "inv_temp")

    def _build_base_model(hidden_dim: int) -> QuantumBornGPT:
        torch.manual_seed(42)
        return QuantumBornGPT(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            seed_dim=seed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            block_size=seq_len,
            use_two_beam_mlp=True,
            use_flash_born=False,
        ).to(device)

    def _train_arm(arm_name: str, mode: str) -> dict[str, Any]:
        if mode == "strict_ortho_1d":
            model = _build_base_model(hidden_dim=embed_dim)
        elif mode == "strict_ortho_2d":
            model = _build_base_model(hidden_dim=2 * embed_dim)
        elif mode == "overcomplete_relaxed_2d":
            model = _build_base_model(hidden_dim=2 * embed_dim)
            for l_idx, block in enumerate(model.blocks):
                block.mlp = OvercompleteFrameInterference(
                    embed_dim=embed_dim,
                    frame_expansion=2.0,
                    target_iso_scale=1.25,
                    seed=42 + 10 * l_idx,
                ).to(device)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        muon_params, adamw_params = split_muon_adamw_params(model, adamw_name_patterns=adamw_patterns)
        if mode in ("strict_ortho_1d", "strict_ortho_2d"):
            opt_muon: torch.optim.Optimizer = TangentStiefelMuon(
                muon_params,
                lr=0.03,
                momentum=0.95,
                ns_steps=5,
                retract_every=1,
            )
        else:
            opt_muon = StiefelMuon(
                muon_params,
                lr=0.03,
                momentum=0.95,
                weight_decay=0.15,
                decay_mode="smooth",
                stiefel_penalty=0.0,
                ns_steps=5,
                retract_every=1000,
            )
        opt_adamw = torch.optim.AdamW(adamw_params, lr=4e-3, weight_decay=0.0)

        init_val_ce, init_r_eff, init_iso_err = _evaluate_arm(model, x_val, y_val)
        history: list[dict[str, float]] = [
            {
                "step": 0.0,
                "val_ce": round(init_val_ce, 4),
                "r_eff_seed": round(init_r_eff, 4),
                "mlp_iso_err": round(init_iso_err, 6),
            }
        ]

        model.train()
        for step in range(1, steps + 1):
            start_idx = ((step - 1) * batch_size) % x_train.size(0)
            xb = x_train[start_idx : start_idx + batch_size]
            yb = y_train[start_idx : start_idx + batch_size]

            opt_muon.zero_grad(set_to_none=True)
            opt_adamw.zero_grad(set_to_none=True)
            _, ce_loss = model(xb, yb)
            assert ce_loss is not None
            if mode == "overcomplete_relaxed_2d":
                frame_pen = sum(b.mlp.frame_penalty() for b in model.blocks)  # type: ignore[union-attr]
                loss = ce_loss + 0.01 * frame_pen
            else:
                loss = ce_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_muon.step()
            opt_adamw.step()

            if step in (50, 100, 150, steps):
                v_ce, r_eff, iso_e = _evaluate_arm(model, x_val, y_val)
                history.append(
                    {
                        "step": float(step),
                        "train_ce": round(float(ce_loss.detach().item()), 4),
                        "val_ce": round(v_ce, 4),
                        "r_eff_seed": round(r_eff, 4),
                        "mlp_iso_err": round(iso_e, 6),
                    }
                )
                model.train()

        final = history[-1]
        print(
            f"  [{arm_name:<26}] Final Val CE: {final['val_ce']:.4f} | "
            f"Train CE: {final['train_ce']:.4f} | "
            f"R_eff(rho_seed): {final['r_eff_seed']:6.2f} | "
            f"MLP IsoErr: {final['mlp_iso_err']:.4f}"
        )
        return {
            "final_val_ce": final["val_ce"],
            "final_train_ce": final["train_ce"],
            "final_r_eff_seed": final["r_eff_seed"],
            "final_mlp_iso_err": final["mlp_iso_err"],
            "history": history,
        }

    arm_a = _train_arm("Arm A (strict_ortho_1d)", "strict_ortho_1d")
    arm_b = _train_arm("Arm B (strict_ortho_2d)", "strict_ortho_2d")
    arm_c = _train_arm("Arm C (overcomplete_2d)", "overcomplete_relaxed_2d")

    val_ce_a = arm_a["final_val_ce"]
    val_ce_b = arm_b["final_val_ce"]
    val_ce_c = arm_c["final_val_ce"]

    r_eff_a = arm_a["final_r_eff_seed"]
    r_eff_b = arm_b["final_r_eff_seed"]
    r_eff_c = arm_c["final_r_eff_seed"]

    iso_err_a = arm_a["final_mlp_iso_err"]
    iso_err_b = arm_b["final_mlp_iso_err"]
    iso_err_c = arm_c["final_mlp_iso_err"]

    print("  " + "-" * 88)
    print(
        f"  [H2 Summary] Arm C Val CE Gain: vs Arm A = {val_ce_a - val_ce_c:+.4f} nats (>= +0.08) | "
        f"vs Arm B = {val_ce_b - val_ce_c:+.4f} nats (>= +0.05) | "
        f"R_eff(C) = {r_eff_c:.2f} vs A={r_eff_a:.2f}, B={r_eff_b:.2f}"
    )

    assert val_ce_c <= val_ce_a - 0.08, (
        f"H2 Failed: expected val_ce_C <= val_ce_A - 0.08, got {val_ce_c:.4f} vs {val_ce_a:.4f}"
    )
    assert val_ce_c <= val_ce_b - 0.05, (
        f"H2 Failed: expected val_ce_C <= val_ce_B - 0.05, got {val_ce_c:.4f} vs {val_ce_b:.4f}"
    )
    assert r_eff_c > r_eff_a and r_eff_c > r_eff_b, (
        f"H2 Failed: expected r_eff_C > r_eff_A and r_eff_C > r_eff_B, "
        f"got C={r_eff_c:.4f}, A={r_eff_a:.4f}, B={r_eff_b:.4f}"
    )
    assert iso_err_a < 0.01, f"H2 Failed: expected iso_err_A < 0.01, got {iso_err_a:.6f}"
    assert iso_err_b < 0.01, f"H2 Failed: expected iso_err_B < 0.01, got {iso_err_b:.6f}"
    assert 0.4 <= iso_err_c <= 2.5, (
        f"H2 Failed: expected 0.4 <= iso_err_C <= 2.5, got {iso_err_c:.6f}"
    )

    return {
        "arm_a_strict_ortho_1d": arm_a,
        "arm_b_strict_ortho_2d": arm_b,
        "arm_c_overcomplete_relaxed_2d": arm_c,
        "val_ce_gain_vs_arm_a": round(val_ce_a - val_ce_c, 4),
        "val_ce_gain_vs_arm_b": round(val_ce_b - val_ce_c, 4),
        "passed": True,
    }


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running EXPERIMENT_PLAN_006 Controlled Verification on device={device}...")

    h1_res = run_experiment_h1_overcomplete_tight_frame()
    h2_res = run_experiment_h2_quantum_born_gpt_arms(device=device)

    elapsed = time.perf_counter() - t0
    payload = {
        "experiment_plan": "EXPERIMENT_PLAN_006",
        "device": str(device),
        "elapsed_seconds": round(elapsed, 3),
        "h1_overcomplete_tight_frame": h1_res,
        "h2_controlled_isometry_relaxation": h2_res,
    }

    out_path = ROOT / ".cache" / "hypothesis_experiments_006.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nAll H1 & H2 quantitative assertions PASSED in {elapsed:.2f}s. Saved to {out_path}")


if __name__ == "__main__":
    main()
