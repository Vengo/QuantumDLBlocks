"""Controlled Hypothesis Verification for EXPERIMENT_PLAN_004: Multi-Beam Kraus Readout.

Verifies:
- Hypothesis H1: Algebraic & Effective Singular Rank Expansion across K in {1, 2, 4} beams.
- Hypothesis H2: Neural Learning Capacity on High-Rank Contextual Hilbert-State Readout Tasks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qdlblocks.functional import l2_normalize
from qdlblocks.layers.kraus_readout import (
    MultiBeamKrausReadout,
    compute_logit_matrix_effective_rank,
)


def run_h1_rank_scaling() -> dict[str, float | bool]:
    """Run Experiment 1 (H1): Algebraic & Effective Singular Rank Ceiling Across K in {1, 2, 4}."""
    torch.manual_seed(42)
    psi = l2_normalize(torch.randn(128, 64), dim=-1)
    base_wte = l2_normalize(torch.randn(128, 16), dim=-1)

    r_sigma: dict[int, float] = {}
    for k in (1, 2, 4):
        torch.manual_seed(42 + k)
        readout = MultiBeamKrausReadout(
            vocab_size=128,
            embed_dim=64,
            seed_dim=16,
            num_beams=k,
            beam_adapter_rank=16,
            adapter_init_scale=1.0,
        )
        with torch.no_grad():
            logits = readout(psi, base_wte)
            r_sigma[k] = float(compute_logit_matrix_effective_rank(logits).item())

    r1 = r_sigma[1]
    r2 = r_sigma[2]
    r4 = r_sigma[4]

    assert r1 <= 16.0 + 1e-4, f"Expected R_sigma(K=1) <= 16.0 + 1e-4, got {r1:.6f}"
    assert r4 >= 32.0, f"Expected R_sigma(K=4) >= 32.0, got {r4:.6f}"
    assert r4 > 2.0 * r1, f"Expected R_sigma(K=4) > 2.0 * R_sigma(K=1), got {r4:.6f} vs {r1:.6f}"
    assert r4 > r2 > r1, (
        f"Expected R_sigma(K=4) > R_sigma(K=2) > R_sigma(K=1), got {r4:.6f}, {r2:.6f}, {r1:.6f}"
    )

    return {
        "r_sigma_k1": r1,
        "r_sigma_k2": r2,
        "r_sigma_k4": r4,
        "passed": True,
    }


def _build_rank48_target_dictionary(
    base_wte: torch.Tensor,
    vocab_size: int,
    embed_dim: int,
    seed_dim: int,
) -> torch.Tensor:
    """Construct a ground-truth rank-48 target measurement matrix W_true in R^{V x d}."""
    q0, _ = torch.linalg.qr(torch.randn(vocab_size, seed_dim))
    q1, _ = torch.linalg.qr(torch.randn(vocab_size, seed_dim))
    q2, _ = torch.linalg.qr(torch.randn(vocab_size, seed_dim))
    b0 = l2_normalize(0.75 * base_wte + 0.25 * (q0 * math.sqrt(vocab_size)), dim=-1)
    b1 = l2_normalize(q1 * math.sqrt(vocab_size), dim=-1)
    b2 = l2_normalize(q2 * math.sqrt(vocab_size), dim=-1)
    zero_block = torch.zeros(vocab_size, embed_dim - 3 * seed_dim)
    w_raw = torch.cat([b0, b1, b2, zero_block], dim=-1)
    return l2_normalize(w_raw, dim=-1)


def _train_readout_arm(
    readout: MultiBeamKrausReadout,
    psi_train: torch.Tensor,
    y_train: torch.Tensor,
    psi_val: torch.Tensor,
    y_val: torch.Tensor,
    base_wte: torch.Tensor,
    steps: int = 120,
    lr: float = 0.02,
    weight_decay: float = 1e-4,
) -> dict[str, float]:
    """Train a MultiBeamKrausReadout arm and evaluate validation CE and effective rank."""
    optimizer = torch.optim.AdamW(readout.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        optimizer.zero_grad()
        logits = readout(psi_train, base_wte)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_logits = readout(psi_val, base_wte)
        val_ce = float(F.cross_entropy(val_logits, y_val).item())
        val_eff_rank = float(compute_logit_matrix_effective_rank(val_logits).item())
        train_ce = float(F.cross_entropy(readout(psi_train, base_wte), y_train).item())

    return {
        "train_ce": train_ce,
        "val_ce": val_ce,
        "logit_eff_rank": val_eff_rank,
    }


def run_h2_learning_capacity() -> dict[str, object]:
    """Run Experiment 2 (H2): High-Rank Hilbert State Sequence Readout Capacity."""
    torch.manual_seed(100)
    d = 64
    d_seed = 16
    v = 256
    n_train = 512
    n_val = 256

    base_wte = l2_normalize(torch.randn(v, d_seed), dim=-1)
    w_true = _build_rank48_target_dictionary(
        base_wte=base_wte,
        vocab_size=v,
        embed_dim=d,
        seed_dim=d_seed,
    )

    idx_train = torch.arange(n_train) % v
    idx_val = torch.arange(n_val) % v
    psi_train = l2_normalize(
        w_true[idx_train] + 0.50 * l2_normalize(torch.randn(n_train, d), dim=-1),
        dim=-1,
    )
    psi_val = l2_normalize(
        w_true[idx_val] + 0.50 * l2_normalize(torch.randn(n_val, d), dim=-1),
        dim=-1,
    )
    y_train = torch.argmax(psi_train @ w_true.T, dim=-1)
    y_val = torch.argmax(psi_val @ w_true.T, dim=-1)

    torch.manual_seed(201)
    arm_a = MultiBeamKrausReadout(
        vocab_size=v,
        embed_dim=d,
        seed_dim=d_seed,
        num_beams=1,
    )
    res_a = _train_readout_arm(
        arm_a,
        psi_train=psi_train,
        y_train=y_train,
        psi_val=psi_val,
        y_val=y_val,
        base_wte=base_wte,
        steps=120,
        lr=0.02,
        weight_decay=1e-4,
    )

    torch.manual_seed(202)
    arm_b = MultiBeamKrausReadout(
        vocab_size=v,
        embed_dim=d,
        seed_dim=d_seed,
        num_beams=4,
        beam_adapter_rank=12,
        adapter_init_scale=1.0,
    )
    res_b = _train_readout_arm(
        arm_b,
        psi_train=psi_train,
        y_train=y_train,
        psi_val=psi_val,
        y_val=y_val,
        base_wte=base_wte,
        steps=120,
        lr=0.02,
        weight_decay=1e-4,
    )

    assert res_b["logit_eff_rank"] >= 1.35 * res_a["logit_eff_rank"], (
        f"Expected Arm B logit_eff_rank >= 1.35 * Arm A, got {res_b['logit_eff_rank']:.4f} "
        f"vs {res_a['logit_eff_rank']:.4f}"
    )
    assert res_b["val_ce"] <= res_a["val_ce"] - 0.15, (
        f"Expected Arm B val_ce <= Arm A val_ce - 0.15, got {res_b['val_ce']:.4f} "
        f"vs {res_a['val_ce']:.4f}"
    )

    return {
        "arm_a_k1": res_a,
        "arm_b_k4": res_b,
        "passed": True,
    }


def main() -> None:
    h1 = run_h1_rank_scaling()
    h2 = run_h2_learning_capacity()

    results = {
        "h1_rank_scaling": h1,
        "h2_learning_capacity": h2,
    }

    cache_path = REPO_ROOT / ".cache" / "hypothesis_experiments_004.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    arm_a = h2["arm_a_k1"]
    arm_b = h2["arm_b_k4"]
    assert isinstance(arm_a, dict) and isinstance(arm_b, dict)

    print("=" * 76)
    print("EXPERIMENT_PLAN_004: Multi-Beam Kraus Subspace POVM Readout Verification")
    print("=" * 76)
    print(
        f"[H1] Effective Singular Rank R_sigma(L): "
        f"K=1 -> {h1['r_sigma_k1']:.4f} (ceiling 16.0) | "
        f"K=2 -> {h1['r_sigma_k2']:.4f} | "
        f"K=4 -> {h1['r_sigma_k4']:.4f}"
    )
    print(
        f"[H2] Arm A (Single-Beam K=1): val_ce={arm_a['val_ce']:.4f}, "
        f"train_ce={arm_a['train_ce']:.4f}, logit_eff_rank={arm_a['logit_eff_rank']:.4f}"
    )
    print(
        f"[H2] Arm B (Multi-Beam  K=4): val_ce={arm_b['val_ce']:.4f}, "
        f"train_ce={arm_b['train_ce']:.4f}, logit_eff_rank={arm_b['logit_eff_rank']:.4f}"
    )
    print(f"Saved results to: {cache_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()
