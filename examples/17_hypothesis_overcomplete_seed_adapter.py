"""Example 17: Controlled Verification of Observable Hypotheses on Overcomplete Born Seed Readout
Adapters (`OvercompleteSeedReadoutAdapter`), Input-Sensitivity Pullback Rank (H1), and
Fixed-Width Seed Capacity Recovery in `QuantumBornGPT` (H2).

Implements Section 3.4 of EXPERIMENT_PLAN_007.md and writes empirical results to
`.cache/hypothesis_experiments_007.json`.
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


def run_experiment_h1_pullback_rank_and_nullspace_fitting(device: torch.device) -> dict[str, Any]:
    """Experiment 1 (H1): Pullback Input-Sensitivity Rank & Nullspace Target Fitting."""
    print("\n" + "=" * 92)
    print("EXPERIMENT 1 (H1): Pullback Input-Sensitivity Rank & Nullspace Target Fitting")
    print("=" * 92)

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
    b_null = q_full[d_seed : d_seed + 8].contiguous()  # (8, 128) active nullspace basis in ker(W_lin)
    c_mix = l2_normalize(torch.randn(64, 8, device=device, dtype=torch.float32), dim=-1)
    v_null = l2_normalize(c_mix @ b_null, dim=-1).contiguous()  # (64, 128) in ker(W_lin)

    null_orthogonality_err = float(torch.linalg.norm(v_null @ w_lin_fixed.T).item())
    assert null_orthogonality_err < 1e-5, f"Expected V_null in ker(W_lin), got {null_orthogonality_err}"

    psi_iso = l2_normalize(torch.randn(n_train, d, device=device, dtype=torch.float32), dim=-1)

    adapter_init = OvercompleteSeedReadoutAdapter(
        embed_dim=d,
        seed_dim=d_seed,
        frame_expansion=2,
        init_gate=0.5,
    ).to(device)

    def _lin_fn(x: torch.Tensor) -> torch.Tensor:
        return l2_normalize(F.linear(x, w_lin_fixed), dim=-1)

    def _over_fn(x: torch.Tensor) -> torch.Tensor:
        phi_l = _lin_fn(x)
        return adapter_init(x, phi_l)

    rank_lin_metrics = compute_input_sensitivity_effective_rank(_lin_fn, psi_iso)
    rank_over_metrics = compute_input_sensitivity_effective_rank(_over_fn, psi_iso)

    r_eff_lin = rank_lin_metrics["pullback_eff_rank"]
    r_eff_over = rank_over_metrics["pullback_eff_rank"]
    rank_expansion_ratio = r_eff_over / max(r_eff_lin, 1e-12)

    print(
        f"  [LinearSeedCollapse]           R_eff(G): {r_eff_lin:6.2f} (ceiling {d_seed}.0) | "
        f"PR(G): {rank_lin_metrics['pullback_participation_ratio']:6.2f} | "
        f"Tr(G): {rank_lin_metrics['trace_G']:7.2f}"
    )
    print(
        f"  [OvercompleteSeedReadout]      R_eff(G): {r_eff_over:6.2f} (M={m_frame} frame) | "
        f"PR(G): {rank_over_metrics['pullback_participation_ratio']:6.2f} | "
        f"Tr(G): {rank_over_metrics['trace_G']:7.2f}"
    )
    print(f"  [Input-Sensitivity Expansion]  R_eff(G_over) / R_eff(G_lin) = {rank_expansion_ratio:.2f}x (>= 2.20x)")

    assert r_eff_lin <= float(d_seed) + 1e-4, (
        f"H1 Failed: expected R_eff(G(phi_lin)) <= {d_seed}.0, got {r_eff_lin:.6f}"
    )
    assert r_eff_over >= 2.2 * r_eff_lin and r_eff_over >= 70.0, (
        f"H1 Failed: expected R_eff(G(phi_over)) >= 2.2 * R_eff(G(phi_lin)) and >= 70.0, "
        f"got {r_eff_over:.4f} vs {r_eff_lin:.4f}"
    )

    # Part 2: Multi-subspace target observable y(psi) depending on row(W_lin) and quadratic Born terms in ker(W_lin)
    p_null = torch.empty(d_seed, 64, device=device, dtype=torch.float32)
    nn.init.orthogonal_(p_null)
    scale_q = float(d) / math.sqrt(float(d_seed))

    def _sample_multisubspace_states(n: int) -> torch.Tensor:
        c_lin = torch.randn(n, d_seed, device=device, dtype=torch.float32) @ w_lin_fixed
        c_null = 1.5 * (torch.randn(n, 8, device=device, dtype=torch.float32) @ b_null)
        c_bg = 0.1 * torch.randn(n, d, device=device, dtype=torch.float32)
        return l2_normalize(c_lin + c_null + c_bg, dim=-1)

    psi_train = _sample_multisubspace_states(n_train)
    psi_test = _sample_multisubspace_states(n_test)

    def _compute_target(psi: torch.Tensor) -> torch.Tensor:
        phi_base = F.linear(psi, w_lin_fixed)
        z_null = F.linear(psi, v_null)
        q_null = (z_null.square() - (1.0 / float(d))) * scale_q
        delta_null = 0.8 * F.linear(q_null, p_null)
        return l2_normalize(phi_base + delta_null, dim=-1)

    y_train = _compute_target(psi_train)
    y_test = _compute_target(psi_test)

    # Trainable Single-Matrix LinearSeedCollapse (even when W_hat is trained end-to-end)
    torch.manual_seed(101)
    lin_model = LinearSeedCollapse(embed_dim=d, seed_dim=d_seed).to(device)
    with torch.no_grad():
        lin_model.w_lin.copy_(w_lin_fixed)
    opt_lin = torch.optim.AdamW(lin_model.parameters(), lr=3e-3)

    for _ in range(250):
        opt_lin.zero_grad(set_to_none=True)
        pred_lin = lin_model(psi_train)
        loss_lin = F.mse_loss(pred_lin, y_train)
        loss_lin.backward()
        opt_lin.step()

    with torch.no_grad():
        lin_train_mse = float(F.mse_loss(lin_model(psi_train), y_train).item())
        lin_test_mse = float(F.mse_loss(lin_model(psi_test), y_test).item())

    # Trainable OvercompleteSeedReadoutAdapter
    torch.manual_seed(101)
    over_adapter = OvercompleteSeedReadoutAdapter(
        embed_dim=d,
        seed_dim=d_seed,
        frame_expansion=2,
        init_gate=0.5,
    ).to(device)
    opt_over = torch.optim.AdamW(over_adapter.parameters(), lr=3e-3, weight_decay=1e-3)

    phi_lin_train = _lin_fn(psi_train)
    phi_lin_test = _lin_fn(psi_test)

    for _ in range(250):
        opt_over.zero_grad(set_to_none=True)
        pred_over = over_adapter(psi_train, phi_lin_train)
        loss_over = F.mse_loss(pred_over, y_train)
        loss_over.backward()
        opt_over.step()

    with torch.no_grad():
        over_train_mse = float(F.mse_loss(over_adapter(psi_train, phi_lin_train), y_train).item())
        over_test_mse = float(F.mse_loss(over_adapter(psi_test, phi_lin_test), y_test).item())

    mse_reduction_ratio = lin_test_mse / max(over_test_mse, 1e-12)
    print(
        f"  [Nullspace Target Fit (250st)] Linear Test MSE: {lin_test_mse:.6e} (Train: {lin_train_mse:.6e}) | "
        f"Overcomplete Test MSE: {over_test_mse:.6e} (Train: {over_train_mse:.6e}) | "
        f"MSE Reduction: {mse_reduction_ratio:.2f}x (>= 3.00x)"
    )

    assert mse_reduction_ratio >= 3.0, (
        f"H1 Failed: expected OvercompleteSeedReadoutAdapter test MSE at least 3.0x lower than LinearSeedCollapse, "
        f"got ratio {mse_reduction_ratio:.2f}x ({over_test_mse:.6e} vs {lin_test_mse:.6e})"
    )

    return {
        "embed_dim": d,
        "seed_dim": d_seed,
        "frame_dim": m_frame,
        "linear_pullback": {k: round(v, 4) for k, v in rank_lin_metrics.items()},
        "overcomplete_pullback": {k: round(v, 4) for k, v in rank_over_metrics.items()},
        "pullback_eff_rank_ratio": round(rank_expansion_ratio, 4),
        "nullspace_target_fit": {
            "linear_train_mse": round(lin_train_mse, 8),
            "linear_test_mse": round(lin_test_mse, 8),
            "overcomplete_train_mse": round(over_train_mse, 8),
            "overcomplete_test_mse": round(over_test_mse, 8),
            "test_mse_reduction_ratio": round(mse_reduction_ratio, 4),
        },
        "passed": True,
    }


def _generate_contextual_multisubspace_lm_dataset(
    vocab_size: int = 512,
    seq_len: int = 64,
    num_train: int = 256,
    num_val: int = 64,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a synthetic contextual multi-subspace sequence dataset over vocab_size=512.

    Each token transition x_{t+1} depends on the current token x_t and a 16-way parity/XOR
    interaction between the latent subspaces of x_t and x_{t-1}, requiring non-linear
    cross-subspace readout capacity when seed_dim=16.
    """
    next_tok_table = np.zeros((vocab_size, 16), dtype=np.int64)
    for v in range(vocab_size):
        for c in range(16):
            next_tok_table[v, c] = (v * 7 + c * 29 + 13) % vocab_size

    def _make_split(n_seqs: int, split_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        r = np.random.default_rng(split_seed)
        seqs = np.zeros((n_seqs, seq_len + 1), dtype=np.int64)
        seqs[:, 0] = r.integers(0, vocab_size, size=n_seqs)
        seqs[:, 1] = (seqs[:, 0] * 7 + 13) % vocab_size
        for t in range(1, seq_len):
            ctx_bin = ((seqs[:, t - 1] // 32) ^ (seqs[:, t] // 32)) % 16
            seqs[:, t + 1] = next_tok_table[seqs[:, t], ctx_bin]
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
    """Experiment 2 (H2): 3-Arm QuantumBornGPT Seed Capacity Benchmark."""
    print("\n" + "=" * 92)
    print("EXPERIMENT 2 (H2): 3-Arm QuantumBornGPT Seed Readout Capacity Benchmark (200 Steps)")
    print("=" * 92)

    vocab_size = 512
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    block_size = 64
    steps = 200
    batch_size = 16
    lr = 3e-3

    x_train, y_train, x_val, y_val = _generate_contextual_multisubspace_lm_dataset(
        vocab_size=vocab_size,
        seq_len=block_size,
        num_train=256,
        num_val=64,
        seed=42,
    )
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    # Build reference d32 model so all 3 arms share identical initial backbone and top-16 seed subspace
    torch.manual_seed(42)
    ref_d32 = QuantumBornGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        seed_dim=32,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=128,
        block_size=block_size,
        readout_mode="tied_stiefel",
        learnable_temperature=True,
        use_overcomplete_seed_adapter=False,
    ).to(device)
    ref_state = {k: v.detach().clone() for k, v in ref_d32.state_dict().items()}

    def _train_arm(
        arm_label: str,
        seed_dim: int,
        use_overcomplete_seed_adapter: bool,
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
        ).to(device)

        with torch.no_grad():
            for k, p in model.named_parameters():
                if k in ref_state:
                    if p.shape == ref_state[k].shape:
                        p.copy_(ref_state[k])
                    elif k.endswith("wte.weight") or k == "povm_head_seed_delta":
                        if "wte" in k:
                            p.copy_(l2_normalize(ref_state[k][:, :seed_dim], dim=-1))
                        else:
                            p.copy_(ref_state[k][:, :seed_dim])
                    elif k.endswith("W_lift.weight"):
                        p.copy_(ref_state[k][:, :seed_dim])

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
            f"  [{arm_label:<28}] Params: {n_params:7,d} | "
            f"Val CE: {final['val_ce']:.4f} (PPL: {final['val_ppl']:6.2f}) | "
            f"Top-1: {final['top1_acc']:5.2f}% | "
            f"R_eff(rho_seed): {final['seed_eff_rank']:5.2f}"
        )
        out: dict[str, Any] = {
            "seed_dim": seed_dim,
            "use_overcomplete_seed_adapter": use_overcomplete_seed_adapter,
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

    arm_a = _train_arm("Arm A (linear_seed_d16)", seed_dim=16, use_overcomplete_seed_adapter=False)
    arm_b = _train_arm("Arm B (overcomplete_seed_d16)", seed_dim=16, use_overcomplete_seed_adapter=True)
    arm_c = _train_arm("Arm C (linear_seed_d32)", seed_dim=32, use_overcomplete_seed_adapter=False)

    val_ce_a = arm_a["final_val_ce"]
    val_ce_b = arm_b["final_val_ce"]
    val_ce_c = arm_c["final_val_ce"]

    r_eff_a = arm_a["final_seed_eff_rank"]
    r_eff_b = arm_b["final_seed_eff_rank"]

    ce_gain_b_vs_a = val_ce_a - val_ce_b
    ce_gain_c_vs_a = max(val_ce_a - val_ce_c, 1e-6)
    fraction_of_d32_gain_recovered = ce_gain_b_vs_a / ce_gain_c_vs_a
    r_eff_ratio_b_vs_a = r_eff_b / max(r_eff_a, 1e-12)

    print("  " + "-" * 88)
    print(
        f"  [H2 Summary] Arm B vs Arm A Val CE Improvement = {ce_gain_b_vs_a:+.4f} nats (>= +0.1000) | "
        f"Recovered {100.0 * fraction_of_d32_gain_recovered:.1f}% of d_seed=32 gain | "
        f"R_eff(rho_seed) Ratio B/A = {r_eff_ratio_b_vs_a:.2f}x (>= 1.15x)"
    )

    assert val_ce_b <= val_ce_a - 0.10, (
        f"H2 Failed: expected Val CE_B <= Val CE_A - 0.10, got {val_ce_b:.4f} vs {val_ce_a:.4f}"
    )
    assert r_eff_b >= 1.15 * r_eff_a, (
        f"H2 Failed: expected R_eff(rho_seed, B) >= 1.15 * R_eff(rho_seed, A), "
        f"got {r_eff_b:.4f} vs {r_eff_a:.4f} (ratio {r_eff_ratio_b_vs_a:.3f}x)"
    )

    return {
        "arm_a_linear_seed_d16": arm_a,
        "arm_b_overcomplete_seed_d16": arm_b,
        "arm_c_linear_seed_d32": arm_c,
        "val_ce_improvement_b_vs_a": round(ce_gain_b_vs_a, 4),
        "val_ce_improvement_c_vs_a": round(val_ce_a - val_ce_c, 4),
        "fraction_of_d32_gain_recovered": round(fraction_of_d32_gain_recovered, 4),
        "seed_eff_rank_ratio_b_vs_a": round(r_eff_ratio_b_vs_a, 4),
        "passed": True,
    }


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running EXPERIMENT_PLAN_007 Controlled Verification on device={device}...")

    h1_res = run_experiment_h1_pullback_rank_and_nullspace_fitting(device=device)
    h2_res = run_experiment_h2_quantum_born_gpt_seed_arms(device=device)

    elapsed = time.perf_counter() - t0
    payload = {
        "experiment_plan": "EXPERIMENT_PLAN_007",
        "device": str(device),
        "elapsed_seconds": round(elapsed, 3),
        "h1_pullback_rank_and_nullspace_fitting": h1_res,
        "h2_quantum_born_gpt_seed_arms": h2_res,
    }

    out_path = ROOT / ".cache" / "hypothesis_experiments_007.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nAll H1 & H2 quantitative assertions PASSED in {elapsed:.2f}s. Saved to {out_path}")


if __name__ == "__main__":
    main()
