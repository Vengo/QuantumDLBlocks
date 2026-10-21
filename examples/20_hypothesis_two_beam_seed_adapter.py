"""Example 20: Controlled Verification of Observable Hypotheses on Phase-Sensitive Two-Beam
Bilinear Seed Readout Adapters (`TwoBeamSeedReadoutAdapter`), Nullspace Antipodal Symmetry
Breaking & Odd/Degree-3 Nullspace Fitting (H1), and Fixed-Width Seed Readout Capacity in
`QuantumBornGPT` (H2).

Implements Section 3.4 of EXPERIMENT_PLAN_010.md and writes empirical results to
`.cache/hypothesis_experiments_010.json`.
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
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.functional import effective_quantum_rank, l2_normalize, purity_participation_ratio
from qdlblocks.layers.seed_adapter import (
    OvercompleteSeedReadoutAdapter,
    compute_input_sensitivity_effective_rank,
)
from qdlblocks.layers.two_beam_seed_adapter import (
    TwoBeamSeedReadoutAdapter,
    compute_nullspace_antipodal_separation,
)
from qdlblocks.models.born_gpt import QuantumBornGPT
from qdlblocks.optim.muon import StiefelMuon, split_muon_adamw_params


class LinearSeedCollapse(nn.Module):
    """Purely linear seed collapse phi_lin(psi) = Normalize(W_lin psi) in S^{D_seed-1}."""

    def __init__(self, embed_dim: int, seed_dim: int, eps: float = 1e-12) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.seed_dim = int(seed_dim)
        self.eps = float(eps)
        self.w_lin = nn.Parameter(torch.empty(self.seed_dim, self.embed_dim, dtype=torch.float32))
        nn.init.orthogonal_(self.w_lin)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        return l2_normalize(F.linear(psi, self.w_lin.to(psi.dtype)), dim=-1, eps=self.eps)


def run_experiment_h1_nullspace_antipodal_and_degree3_fitting(device: torch.device) -> dict[str, Any]:
    """Experiment 1 (H1): Nullspace Antipodal Symmetry Breaking & Phase-Sensitive Nullspace Fitting."""
    print("\n" + "=" * 96)
    print("EXPERIMENT 1 (H1): Nullspace Antipodal Separation & Phase-Sensitive Nullspace Target Fitting")
    print("=" * 96)

    torch.manual_seed(42)
    d = 128
    d_seed = 32
    m_frame = 2 * d
    n_train = 512
    n_test = 256

    # Construct full orthonormal basis Q in R^{D x D}:
    # First 32 rows form W_lin; remaining 96 rows span ker(W_lin)
    q_full, _ = torch.linalg.qr(torch.randn(d, d, device=device, dtype=torch.float32))
    q_full = q_full.T.contiguous()
    w_lin_fixed = q_full[:d_seed].contiguous()  # (32, 128)
    b_null = q_full[d_seed : d_seed + 6].contiguous()  # (6, 128) active nullspace basis in ker(W_lin)

    c1 = l2_normalize(torch.randn(64, 6, device=device, dtype=torch.float32), dim=-1)
    c2 = l2_normalize(torch.randn(64, 6, device=device, dtype=torch.float32), dim=-1)
    c2 = l2_normalize(c2 - (c2 * c1).sum(dim=-1, keepdim=True) * c1, dim=-1)
    v1_null = l2_normalize(c1 @ b_null, dim=-1).contiguous()  # (64, 128) in ker(W_lin)
    v2_null = l2_normalize(c2 @ b_null, dim=-1).contiguous()  # (64, 128) in ker(W_lin)

    null_err_1 = float(torch.linalg.norm(v1_null @ w_lin_fixed.T).item())
    null_err_2 = float(torch.linalg.norm(v2_null @ w_lin_fixed.T).item())
    assert max(null_err_1, null_err_2) < 1e-5, f"Expected V1, V2 in ker(W_lin), got {null_err_1}, {null_err_2}"

    single_beam_init = OvercompleteSeedReadoutAdapter(
        embed_dim=d,
        seed_dim=d_seed,
        frame_expansion=2,
        init_gate=0.5,
    ).to(device)
    two_beam_init = TwoBeamSeedReadoutAdapter(
        embed_dim=d,
        seed_dim=d_seed,
        frame_expansion=2,
        init_gate=0.5,
        init_beam_eta=0.6,
    ).to(device)

    sep_single = compute_nullspace_antipodal_separation(
        single_beam_init, w_lin_fixed, num_samples=256, seed=42
    )
    sep_two_beam = compute_nullspace_antipodal_separation(
        two_beam_init, w_lin_fixed, num_samples=256, seed=42
    )

    psi_iso = l2_normalize(torch.randn(256, d, device=device, dtype=torch.float32), dim=-1)

    def _lin_fn(x: torch.Tensor) -> torch.Tensor:
        return l2_normalize(F.linear(x, w_lin_fixed), dim=-1)

    def _single_fn(x: torch.Tensor) -> torch.Tensor:
        return single_beam_init(x, _lin_fn(x))

    def _two_beam_fn(x: torch.Tensor) -> torch.Tensor:
        return two_beam_init(x, _lin_fn(x))

    rank_lin = compute_input_sensitivity_effective_rank(_lin_fn, psi_iso)
    rank_single = compute_input_sensitivity_effective_rank(_single_fn, psi_iso)
    rank_two_beam = compute_input_sensitivity_effective_rank(_two_beam_fn, psi_iso)

    print(
        f"  [Nullspace Antipodal Sep]      Single-Beam (Plan 007): {sep_single:.6e} (<= 1e-6) | "
        f"Two-Beam (Plan 010): {sep_two_beam:.4f} (>= 0.5000)"
    )
    print(
        f"  [Pullback Effective Rank]      Linear: {rank_lin['pullback_eff_rank']:6.2f} | "
        f"Single-Beam: {rank_single['pullback_eff_rank']:6.2f} | "
        f"Two-Beam: {rank_two_beam['pullback_eff_rank']:6.2f}"
    )

    assert sep_single <= 1e-6, (
        f"H1 Failed: expected Delta_antipodal(phi_over) <= 1e-6, got {sep_single:.6e}"
    )
    assert sep_two_beam >= 0.50, (
        f"H1 Failed: expected Delta_antipodal(phi_2beam) >= 0.50, got {sep_two_beam:.4f}"
    )

    # Part 2: Synthetic Phase-Sensitive Nullspace Target y(psi)
    p_null = torch.empty(d_seed, 64, device=device, dtype=torch.float32)
    nn.init.orthogonal_(p_null)
    scale_b = (float(d) ** 1.5) / math.sqrt(float(d_seed))

    def _sample_phase_nullspace_states(n: int) -> torch.Tensor:
        # Include antipodal pairs in ker(W_lin) combined with linear subspace components
        n_half = n // 2
        c_lin = 0.35 * (torch.randn(n_half, d_seed, device=device, dtype=torch.float32) @ w_lin_fixed)
        c_null = 1.6 * (torch.randn(n_half, 6, device=device, dtype=torch.float32) @ b_null)
        psi_pos = l2_normalize(c_lin + c_null, dim=-1)
        psi_neg = l2_normalize(c_lin - c_null, dim=-1)
        return torch.cat([psi_pos, psi_neg], dim=0)

    psi_train = _sample_phase_nullspace_states(n_train)
    psi_test = _sample_phase_nullspace_states(n_test)

    def _compute_target(psi: torch.Tensor) -> torch.Tensor:
        phi_base = F.linear(psi, w_lin_fixed)
        z1 = F.linear(psi, v1_null)
        z2 = F.linear(psi, v2_null)
        b_null_feat = (z1 * z1) * F.silu(z2) * scale_b
        delta_null = 0.9 * F.linear(b_null_feat, p_null)
        return l2_normalize(phi_base + delta_null, dim=-1)

    y_train = _compute_target(psi_train)
    y_test = _compute_target(psi_test)

    phi_lin_train = _lin_fn(psi_train)
    phi_lin_test = _lin_fn(psi_test)

    # 1. Fixed LinearSeedCollapse baseline (phi_lin = Normalize(W_lin psi))
    with torch.no_grad():
        lin_train_mse = float(F.mse_loss(phi_lin_train, y_train).item())
        lin_test_mse = float(F.mse_loss(phi_lin_test, y_test).item())

    # 2. Trainable Single-Beam OvercompleteSeedReadoutAdapter (Plan 007)
    torch.manual_seed(101)
    over_adapter = OvercompleteSeedReadoutAdapter(
        embed_dim=d,
        seed_dim=d_seed,
        frame_expansion=2,
        init_gate=0.5,
    ).to(device)
    opt_over = torch.optim.AdamW(over_adapter.parameters(), lr=3e-3, weight_decay=1e-4)

    for _ in range(250):
        opt_over.zero_grad(set_to_none=True)
        pred_over = over_adapter(psi_train, phi_lin_train)
        loss_over = F.mse_loss(pred_over, y_train)
        loss_over.backward()
        opt_over.step()

    with torch.no_grad():
        over_train_mse = float(F.mse_loss(over_adapter(psi_train, phi_lin_train), y_train).item())
        over_test_mse = float(F.mse_loss(over_adapter(psi_test, phi_lin_test), y_test).item())

    # 3. Trainable Two-Beam TwoBeamSeedReadoutAdapter (Plan 010)
    torch.manual_seed(101)
    two_beam_adapter = TwoBeamSeedReadoutAdapter(
        embed_dim=d,
        seed_dim=d_seed,
        frame_expansion=2,
        init_gate=0.5,
        init_beam_eta=0.6,
    ).to(device)
    opt_two_beam = torch.optim.AdamW(two_beam_adapter.parameters(), lr=3e-3, weight_decay=1e-4)

    for _ in range(250):
        opt_two_beam.zero_grad(set_to_none=True)
        pred_two_beam = two_beam_adapter(psi_train, phi_lin_train)
        loss_two_beam = F.mse_loss(pred_two_beam, y_train)
        loss_two_beam.backward()
        opt_two_beam.step()

    with torch.no_grad():
        two_beam_train_mse = float(F.mse_loss(two_beam_adapter(psi_train, phi_lin_train), y_train).item())
        two_beam_test_mse = float(F.mse_loss(two_beam_adapter(psi_test, phi_lin_test), y_test).item())

    ratio_vs_single_beam = over_test_mse / max(two_beam_test_mse, 1e-12)
    ratio_vs_linear = lin_test_mse / max(two_beam_test_mse, 1e-12)

    print(
        f"  [Phase-Sensitive Target Fit]   Linear Test MSE:      {lin_test_mse:.6e} (Train: {lin_train_mse:.6e})\n"
        f"                                 Single-Beam Test MSE: {over_test_mse:.6e} (Train: {over_train_mse:.6e})\n"
        f"                                 Two-Beam Test MSE:    {two_beam_test_mse:.6e} (Train: {two_beam_train_mse:.6e})\n"
        f"  [Held-Out MSE Reductions]      Two-Beam vs Single-Beam: {ratio_vs_single_beam:6.2f}x (>= 3.00x) | "
        f"Two-Beam vs Linear: {ratio_vs_linear:6.2f}x (>= 20.00x)"
    )

    assert ratio_vs_single_beam >= 3.0, (
        f"H1 Failed: expected TwoBeamSeedReadoutAdapter test MSE at least 3.0x lower than "
        f"OvercompleteSeedReadoutAdapter, got {ratio_vs_single_beam:.2f}x "
        f"({two_beam_test_mse:.6e} vs {over_test_mse:.6e})"
    )
    assert ratio_vs_linear >= 20.0, (
        f"H1 Failed: expected TwoBeamSeedReadoutAdapter test MSE at least 20.0x lower than "
        f"LinearSeedCollapse, got {ratio_vs_linear:.2f}x ({two_beam_test_mse:.6e} vs {lin_test_mse:.6e})"
    )

    return {
        "embed_dim": d,
        "seed_dim": d_seed,
        "frame_dim": m_frame,
        "nullspace_antipodal_separation": {
            "single_beam": round(sep_single, 8),
            "two_beam": round(sep_two_beam, 6),
        },
        "pullback_rank": {
            "linear": {k: round(v, 4) for k, v in rank_lin.items()},
            "single_beam": {k: round(v, 4) for k, v in rank_single.items()},
            "two_beam": {k: round(v, 4) for k, v in rank_two_beam.items()},
        },
        "phase_sensitive_nullspace_fit": {
            "linear_train_mse": round(lin_train_mse, 8),
            "linear_test_mse": round(lin_test_mse, 8),
            "single_beam_train_mse": round(over_train_mse, 8),
            "single_beam_test_mse": round(over_test_mse, 8),
            "two_beam_train_mse": round(two_beam_train_mse, 8),
            "two_beam_test_mse": round(two_beam_test_mse, 8),
            "mse_reduction_vs_single_beam": round(ratio_vs_single_beam, 4),
            "mse_reduction_vs_linear": round(ratio_vs_linear, 4),
        },
        "passed": True,
    }


def _generate_signed_contextual_multisubspace_lm_dataset(
    vocab_size: int = 512,
    seq_len: int = 64,
    num_train: int = 256,
    num_val: int = 64,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a synthetic contextual multi-subspace sequence dataset over vocab_size=512.

    Each token transition x_{t+1} depends on the current token x_t and the signed
    displacement (b_t - b_{t-1}) mod 31 between the 32-way latent subspace coordinates
    b_t = x_t // 16 and b_{t-1} = x_{t-1} // 16, requiring phase/sign-sensitive
    cross-subspace readout capacity when seed_dim=16.
    """
    next_tok_table = np.zeros((vocab_size, 31), dtype=np.int64)
    for v in range(vocab_size):
        for c in range(31):
            next_tok_table[v, c] = (v * 7 + c * 29 + 13) % vocab_size

    def _make_split(n_seqs: int, split_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        r = np.random.default_rng(split_seed)
        seqs = np.zeros((n_seqs, seq_len + 1), dtype=np.int64)
        seqs[:, 0] = r.integers(0, vocab_size, size=n_seqs)
        seqs[:, 1] = (seqs[:, 0] * 7 + 13) % vocab_size
        for t in range(1, seq_len):
            b_prev = seqs[:, t - 1] // 16
            b_curr = seqs[:, t] // 16
            signed_disp_bin = (b_curr - b_prev) % 31
            seqs[:, t + 1] = next_tok_table[seqs[:, t], signed_disp_bin]
        return (
            torch.from_numpy(seqs[:, :-1].copy()),
            torch.from_numpy(seqs[:, 1:].copy()),
        )

    x_train, y_train = _make_split(num_train, seed)
    x_val, y_val = _make_split(num_val, seed + 100)
    return x_train, y_train, x_val, y_val


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
    phi_flat = phi_seed.reshape(-1, model.seed_dim)[:1024].float()
    seed_eff_rank = float(effective_quantum_rank(phi_flat).item())
    seed_pr = float(purity_participation_ratio(phi_flat).item())
    preds = logits.argmax(dim=-1)
    top1_acc = 100.0 * float((preds == y_val).float().mean().item())
    return {
        "val_ce": round(val_ce, 4),
        "val_ppl": round(math.exp(min(val_ce, 20.0)), 2),
        "top1_acc": round(top1_acc, 2),
        "seed_eff_rank": round(seed_eff_rank, 4),
        "seed_participation_ratio": round(seed_pr, 4),
    }


def run_experiment_h2_quantum_born_gpt_seed_arms(device: torch.device) -> dict[str, Any]:
    """Experiment 2 (H2): 3-Arm QuantumBornGPT Seed Adapter Benchmark at Fixed D_seed=16."""
    print("\n" + "=" * 96)
    print("EXPERIMENT 2 (H2): 3-Arm QuantumBornGPT Seed Adapter Benchmark at D_seed=16 (200 Steps)")
    print("=" * 96)

    vocab_size = 512
    embed_dim = 64
    seed_dim = 16
    num_heads = 4
    num_layers = 2
    block_size = 64
    steps = 200
    batch_size = 16
    lr = 3e-3

    x_train, y_train, x_val, y_val = _generate_signed_contextual_multisubspace_lm_dataset(
        vocab_size=vocab_size,
        seq_len=block_size,
        num_train=256,
        num_val=64,
        seed=42,
    )
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    # Build reference d16 model so all 3 arms share identical initial backbone and seed embedding
    torch.manual_seed(42)
    ref_d16 = QuantumBornGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seed_dim=seed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=128,
        block_size=block_size,
        readout_mode="tied_stiefel",
        learnable_temperature=True,
        use_overcomplete_seed_adapter=False,
        use_two_beam_seed_adapter=False,
    ).to(device)
    ref_state = {k: v.detach().clone() for k, v in ref_d16.state_dict().items()}

    def _train_arm(
        arm_label: str,
        use_overcomplete_seed_adapter: bool,
        use_two_beam_seed_adapter: bool,
    ) -> dict[str, Any]:
        torch.manual_seed(42)
        model = QuantumBornGPT(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            seed_dim=seed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=128,
            block_size=block_size,
            readout_mode="tied_stiefel",
            learnable_temperature=True,
            use_overcomplete_seed_adapter=use_overcomplete_seed_adapter,
            use_two_beam_seed_adapter=use_two_beam_seed_adapter,
        ).to(device)

        with torch.no_grad():
            for k, p in model.named_parameters():
                if k in ref_state and p.shape == ref_state[k].shape:
                    p.copy_(ref_state[k])

        muon_params, adam_params = split_muon_adamw_params(
            model,
            adamw_name_patterns=("wte", "povm_head", "embed", "inv_temp", "seed_adapter"),
        )
        opt_muon = StiefelMuon(
            muon_params,
            lr=0.02,
            momentum=0.95,
            stiefel_penalty=0.005,
            ns_steps=5,
            retract_every=4,
        )
        if model.seed_adapter is not None:
            ad_ids = {id(p) for p in model.seed_adapter.parameters()}
            base_adam = [p for p in adam_params if id(p) not in ad_ids]
            opt_adam = torch.optim.AdamW(
                [
                    {"params": base_adam, "lr": lr, "weight_decay": 1e-4},
                    {"params": list(model.seed_adapter.parameters()), "lr": 3.0 * lr, "weight_decay": 1e-4},
                ]
            )
        else:
            opt_adam = torch.optim.AdamW(adam_params, lr=lr, weight_decay=1e-4)

        init_metrics = _evaluate_gpt_arm(model, x_val, y_val)
        history: list[dict[str, float]] = [{"step": 0.0, "train_ce": init_metrics["val_ce"], **init_metrics}]

        for step in range(1, steps + 1):
            model.train()
            start = ((step - 1) * batch_size) % x_train.size(0)
            xb = x_train[start : start + batch_size]
            yb = y_train[start : start + batch_size]

            opt_muon.zero_grad(set_to_none=True)
            opt_adam.zero_grad(set_to_none=True)
            _, loss = model(xb, yb)
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_muon.step()
            opt_adam.step()

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
            f"  [{arm_label:<32}] Params: {n_params:7,d} | "
            f"Val CE: {final['val_ce']:.4f} (PPL: {final['val_ppl']:6.2f}) | "
            f"Top-1: {final['top1_acc']:5.2f}% | "
            f"R_eff(rho_seed): {final['seed_eff_rank']:5.2f}"
        )
        out: dict[str, Any] = {
            "seed_dim": seed_dim,
            "use_overcomplete_seed_adapter": use_overcomplete_seed_adapter,
            "use_two_beam_seed_adapter": use_two_beam_seed_adapter,
            "num_params": n_params,
            "final_train_ce": final["train_ce"],
            "final_val_ce": final["val_ce"],
            "final_val_ppl": final["val_ppl"],
            "final_top1_acc": final["top1_acc"],
            "final_seed_eff_rank": final["seed_eff_rank"],
            "final_seed_participation_ratio": final["seed_participation_ratio"],
            "history": history,
        }
        if model.seed_adapter is not None:
            out["adapter_frame_metrics"] = {
                k: round(v, 5) for k, v in model.seed_adapter.frame_metrics().items()
            }
        return out

    arm_a = _train_arm(
        "Arm A (linear_seed_d16)",
        use_overcomplete_seed_adapter=False,
        use_two_beam_seed_adapter=False,
    )
    arm_b = _train_arm(
        "Arm B (single_beam_seed_d16)",
        use_overcomplete_seed_adapter=True,
        use_two_beam_seed_adapter=False,
    )
    arm_c = _train_arm(
        "Arm C (two_beam_seed_d16)",
        use_overcomplete_seed_adapter=False,
        use_two_beam_seed_adapter=True,
    )

    val_ce_a = arm_a["final_val_ce"]
    val_ce_b = arm_b["final_val_ce"]
    val_ce_c = arm_c["final_val_ce"]

    r_eff_a = arm_a["final_seed_eff_rank"]
    r_eff_b = arm_b["final_seed_eff_rank"]
    r_eff_c = arm_c["final_seed_eff_rank"]

    ce_gain_c_vs_b = val_ce_b - val_ce_c
    ce_gain_c_vs_a = val_ce_a - val_ce_c

    print("  " + "-" * 92)
    print(
        f"  [H2 Summary] Arm C vs Arm B Val CE Gain = {ce_gain_c_vs_b:+.4f} nats (>= +0.0600) | "
        f"Arm C vs Arm A Val CE Gain = {ce_gain_c_vs_a:+.4f} nats (>= +0.4000) | "
        f"R_eff(rho_seed): C={r_eff_c:.2f} >= B={r_eff_b:.2f} > A={r_eff_a:.2f}"
    )

    assert val_ce_c <= val_ce_b - 0.06, (
        f"H2 Failed: expected Val CE_C <= Val CE_B - 0.06, got {val_ce_c:.4f} vs {val_ce_b:.4f} "
        f"(diff {ce_gain_c_vs_b:+.4f})"
    )
    assert val_ce_c <= val_ce_a - 0.40, (
        f"H2 Failed: expected Val CE_C <= Val CE_A - 0.40, got {val_ce_c:.4f} vs {val_ce_a:.4f} "
        f"(diff {ce_gain_c_vs_a:+.4f})"
    )
    assert r_eff_c >= r_eff_b and r_eff_b > r_eff_a, (
        f"H2 Failed: expected R_eff(C) >= R_eff(B) > R_eff(A), "
        f"got C={r_eff_c:.4f}, B={r_eff_b:.4f}, A={r_eff_a:.4f}"
    )

    return {
        "arm_a_linear_seed_d16": arm_a,
        "arm_b_single_beam_seed_adapter_d16": arm_b,
        "arm_c_two_beam_seed_adapter_d16": arm_c,
        "val_ce_improvement_c_vs_b": round(ce_gain_c_vs_b, 4),
        "val_ce_improvement_c_vs_a": round(ce_gain_c_vs_a, 4),
        "top1_acc_improvement_c_vs_b": round(arm_c["final_top1_acc"] - arm_b["final_top1_acc"], 2),
        "top1_acc_improvement_c_vs_a": round(arm_c["final_top1_acc"] - arm_a["final_top1_acc"], 2),
        "passed": True,
    }


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running EXPERIMENT_PLAN_010 Controlled Verification on device={device}...")

    h1_res = run_experiment_h1_nullspace_antipodal_and_degree3_fitting(device=device)
    h2_res = run_experiment_h2_quantum_born_gpt_seed_arms(device=device)

    elapsed = time.perf_counter() - t0
    payload = {
        "experiment_plan": "EXPERIMENT_PLAN_010",
        "device": str(device),
        "elapsed_seconds": round(elapsed, 3),
        "h1_nullspace_antipodal_and_degree3_fitting": h1_res,
        "h2_quantum_born_gpt_seed_arms": h2_res,
    }

    out_path = ROOT / ".cache" / "hypothesis_experiments_010.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nAll H1 & H2 quantitative assertions PASSED in {elapsed:.2f}s. Saved to {out_path}")


if __name__ == "__main__":
    main()
