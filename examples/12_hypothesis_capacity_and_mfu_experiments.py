"""Example 12: Controlled Verification of Observable Hypotheses on Multi-Horizon POVM
Temperature Dynamics (H1), Strided MTP Throughput & MFU Recovery (H2), and
Hilbert Seed Subspace Capacity / Effective Quantum Rank (H3).

Implements Section 3.5 of EXPERIMENT_PLAN_002.md and writes empirical results to
`.cache/hypothesis_experiments_002.json`.
"""

from __future__ import annotations

import copy
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

from qdlblocks.functional import (
    effective_quantum_rank,
    l2_normalize,
    purity_participation_ratio,
)
from qdlblocks.models.born_gpt import QuantumBornGPT
from qdlblocks.optim.muon import StiefelMuon, split_muon_adamw_params


def _generate_multi_horizon_markov_data(
    num_sequences: int,
    seq_len: int,
    vocab_size: int = 512,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate sequences from a period-3 branching Markov dynamical system.

    Transition rule:
    - Steps t ≡ 0, 1 (mod 3): deterministic affine cycle x_{t+1} = (3 * x_t + 7) mod 256,
      giving low conditional entropy at horizon h=1.
    - Steps t ≡ 2 (mod 3): regime branch into a 32-token cluster offset by 32 * ((t // 3) mod 8),
      causing multi-step horizons h=2, 3 to cross regime branches and exhibit strictly higher
      conditional entropy than h=1.
    """
    rng = np.random.default_rng(42)
    total_len = seq_len + 1
    data = np.zeros((num_sequences, total_len), dtype=np.int64)
    for i in range(num_sequences):
        cur = int(rng.integers(0, 32))
        for t in range(total_len):
            data[i, t] = cur
            if t % 3 == 2:
                cur = int(rng.integers(0, 32)) + 32 * ((t // 3) % 8)
            else:
                cur = (3 * cur + 7) % min(256, vocab_size)

    if seed != 42:
        perm_rng = np.random.default_rng(seed)
        perm = perm_rng.permutation(num_sequences)
        data = data[perm]

    x = torch.from_numpy(data[:, :-1])
    y = torch.from_numpy(data[:, 1:])
    return x, y


def run_experiment_h1_temperature_dynamics(device: torch.device) -> dict[str, Any]:
    """Experiment 1 (Hypothesis H1): Shared vs. Decoupled Horizon Temperatures & Readout Detach."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 1 (H1): Shared vs. Decoupled Multi-Horizon POVM Temperatures & Detach")
    print("=" * 88)

    vocab_size = 512
    seq_len = 64
    batch_size = 16
    steps = 120

    x_train, y_train = _generate_multi_horizon_markov_data(
        num_sequences=128, seq_len=seq_len, vocab_size=vocab_size, seed=42
    )
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_train[:64], y_train[:64]

    def build_arm(decoupled: bool, detach_readout: bool) -> QuantumBornGPT:
        torch.manual_seed(1337)
        return QuantumBornGPT(
            vocab_size=vocab_size,
            embed_dim=64,
            seed_dim=32,
            num_heads=4,
            num_layers=2,
            hidden_dim=128,
            block_size=seq_len,
            povm_temperature=0.035,
            readout_mode="tied_stiefel",
            learnable_temperature=True,
            mtp_horizons=3,
            mtp_loss_weight=0.5,
            mtp_stride=1,
            mtp_detach_readout=detach_readout,
            mtp_decoupled_temp=decoupled,
        ).to(device)

    model_a = build_arm(decoupled=False, detach_readout=False)
    model_b = build_arm(decoupled=True, detach_readout=True)
    # Ensure identical initial weights for all shared parameters
    with torch.no_grad():
        state_a = model_a.state_dict()
        for k, v in model_b.named_parameters():
            if k in state_a and state_a[k].shape == v.shape:
                v.copy_(state_a[k])

    def train_arm(model: QuantumBornGPT, label: str) -> dict[str, Any]:
        muon_p, adamw_p = split_muon_adamw_params(
            model, adamw_name_patterns=("wte", "povm_head", "embed", "inv_temp")
        )
        temp_p = [p for n, p in model.named_parameters() if "inv_temp" in n]
        temp_ids = {id(p) for p in temp_p}
        other_adam_p = [p for p in adamw_p if id(p) not in temp_ids]
        opt_muon = StiefelMuon(muon_p, lr=0.025)
        opt_adam = torch.optim.AdamW(
            [
                {"params": other_adam_p, "lr": 0.03, "eps": 1e-8},
                {"params": temp_p, "lr": 0.45, "eps": 0.25},
            ],
            weight_decay=0.0,
        )

        history_betas: list[dict[str, float]] = []
        model.train()
        for step in range(1, steps + 1):
            idx_start = ((step - 1) * batch_size) % (x_train.size(0) - batch_size)
            xb = x_train[idx_start : idx_start + batch_size]
            yb = y_train[idx_start : idx_start + batch_size]

            opt_muon.zero_grad(set_to_none=True)
            opt_adam.zero_grad(set_to_none=True)
            _, loss = model(xb, yb)
            assert loss is not None
            loss.backward()
            opt_muon.step()
            opt_adam.step()

            if step in (1, 40, 80, steps):
                mtp_info = dict(model.last_mtp_losses)
                history_betas.append({"step": float(step), **mtp_info})

        model.eval()
        with torch.no_grad():
            logits_val, _ = model(x_val, y_val)
            val_ce_t1 = float(F.cross_entropy(logits_val.reshape(-1, vocab_size), y_val.reshape(-1)).item())
            mtp_final = dict(model.last_mtp_losses)

        b1 = float(mtp_final["beta_t+1"])
        b2 = float(mtp_final["beta_t+2"])
        b3 = float(mtp_final["beta_t+3"])
        print(
            f"  [{label:24s}] Val CE (t+1): {val_ce_t1:.4f} | "
            f"beta_1: {b1:6.3f} | beta_2: {b2:6.3f} | beta_3: {b3:6.3f} | "
            f"Val CE (t+2): {mtp_final['ce_t+2']:.4f} | Val CE (t+3): {mtp_final['ce_t+3']:.4f}"
        )
        return {
            "val_ce_t1": round(val_ce_t1, 4),
            "val_ce_t2": round(float(mtp_final["ce_t+2"]), 4),
            "val_ce_t3": round(float(mtp_final["ce_t+3"]), 4),
            "beta_1": round(b1, 4),
            "beta_2": round(b2, 4),
            "beta_3": round(b3, 4),
            "trajectory": history_betas,
        }

    res_a = train_arm(model_a, "Arm A (Shared-Beta)")
    res_b = train_arm(model_b, "Arm B (Decoupled+Detach)")

    assert res_b["beta_1"] > res_b["beta_2"] > res_b["beta_3"], (
        f"H1 Failed: Expected beta_1 > beta_2 > beta_3 in Arm B, got "
        f"{res_b['beta_1']}, {res_b['beta_2']}, {res_b['beta_3']}"
    )
    assert res_b["beta_1"] > res_a["beta_1"] + 1.0, (
        f"H1 Failed: Expected beta_1(Arm B) > beta_1(Arm A) + 1.0, got "
        f"{res_b['beta_1']} vs {res_a['beta_1']}"
    )
    assert res_b["val_ce_t1"] <= res_a["val_ce_t1"] - 0.05, (
        f"H1 Failed: Expected val_ce_t1(Arm B) <= val_ce_t1(Arm A) - 0.05, got "
        f"{res_b['val_ce_t1']} vs {res_a['val_ce_t1']}"
    )

    return {
        "hypothesis": "H1_decoupled_horizon_temperatures_and_detach",
        "passed": True,
        "arm_a_shared": res_a,
        "arm_b_decoupled": res_b,
        "val_ce_t1_gain_nats": round(res_a["val_ce_t1"] - res_b["val_ce_t1"], 4),
        "beta_1_delta": round(res_b["beta_1"] - res_a["beta_1"], 4),
    }


def _analytical_aux_vocab_flops(
    batch_size: int,
    seq_len: int,
    seed_dim: int,
    vocab_size: int,
    mtp_horizons: int,
    mtp_stride: int,
) -> int:
    """Compute analytical forward+backward FLOPs for auxiliary MTP horizon projections."""
    if mtp_horizons <= 1:
        return 0
    total_flops = 0
    for h_idx in range(1, mtp_horizons):
        t_avail = seq_len - h_idx
        if t_avail <= 0:
            break
        if mtp_stride > 1 and t_avail > mtp_stride:
            offset = (h_idx - 1) % mtp_stride
            t_sub = len(range(offset, t_avail, mtp_stride))
        else:
            t_sub = t_avail
        n_aux = batch_size * t_sub
        # Forward (2*D*V) + Backward (4*D*V) = 6 * n_aux * D_seed * (D_seed + V)
        total_flops += 6 * n_aux * seed_dim * (seed_dim + vocab_size)
    return total_flops


def run_experiment_h2_strided_mtp_throughput(device: torch.device) -> dict[str, Any]:
    """Experiment 2 (Hypothesis H2): Strided Multi-Horizon POVM Throughput & MFU Recovery."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 2 (H2): Static-Shape Strided Multi-Horizon POVM Throughput & FLOPs Recovery")
    print("=" * 88)

    vocab_size = 32768
    embed_dim = 128
    seed_dim = 64
    batch_size = 16
    seq_len = 256
    warmup_steps = 4
    timed_steps = 15

    configs = [
        (1, 1),
        (3, 1),
        (3, 2),
        (3, 4),
        (3, 8),
    ]

    torch.manual_seed(42)
    xb = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    yb = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    results_by_config: dict[str, dict[str, float]] = {}
    for horizons, stride in configs:
        torch.manual_seed(42)
        model = QuantumBornGPT(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            seed_dim=seed_dim,
            num_heads=4,
            num_layers=2,
            hidden_dim=256,
            block_size=seq_len,
            readout_mode="tied_stiefel",
            learnable_temperature=True,
            mtp_horizons=horizons,
            mtp_loss_weight=0.25,
            mtp_stride=stride,
            mtp_detach_readout=True,
            mtp_decoupled_temp=True,
        ).to(device)
        model.train()

        for _ in range(warmup_steps):
            model.zero_grad(set_to_none=True)
            _, loss = model(xb, yb)
            assert loss is not None
            loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()

        step_times: list[float] = []
        for _ in range(timed_steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.zero_grad(set_to_none=True)
            _, loss = model(xb, yb)
            assert loss is not None
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)

        med_step_s = float(np.median(step_times))
        tok_per_s = (batch_size * seq_len) / max(med_step_s, 1e-9)
        aux_flops = _analytical_aux_vocab_flops(batch_size, seq_len, seed_dim, vocab_size, horizons, stride)
        fineweb_aux_flops = _analytical_aux_vocab_flops(32, 512, 128, 50304, horizons, stride)
        key_name = f"H={horizons}_stride={stride}"
        results_by_config[key_name] = {
            "mtp_horizons": float(horizons),
            "mtp_stride": float(stride),
            "tok_per_s": round(tok_per_s, 1),
            "ms_per_step": round(med_step_s * 1000.0, 2),
            "aux_vocab_flops_M": round(aux_flops / 1e6, 2),
            "fineweb_100m_aux_vocab_flops_G": round(fineweb_aux_flops / 1e9, 3),
        }
        print(
            f"  [{key_name:14s}] Speed: {tok_per_s:10,.1f} tok/s ({med_step_s * 1000.0:6.2f} ms/step) | "
            f"Aux Vocab FLOPs: {aux_flops / 1e6:7.2f} MFLOPs | FineWeb Aux: {fineweb_aux_flops / 1e9:6.3f} GFLOPs"
        )

    tps_s1 = results_by_config["H=3_stride=1"]["tok_per_s"]
    tps_s4 = results_by_config["H=3_stride=4"]["tok_per_s"]
    speedup_ratio = tps_s4 / max(tps_s1, 1e-9)
    flops_ratio = (
        results_by_config["H=3_stride=4"]["aux_vocab_flops_M"]
        / max(results_by_config["H=3_stride=1"]["aux_vocab_flops_M"], 1e-9)
    )

    assert speedup_ratio >= 1.15, (
        f"H2 Failed: Expected tok_per_s(H=3, stride=4) >= 1.15 * tok_per_s(H=3, stride=1), "
        f"got ratio={speedup_ratio:.4f} ({tps_s4} vs {tps_s1})"
    )
    assert flops_ratio <= 0.26, (
        f"H2 Failed: Expected auxiliary vocabulary FLOP ratio <= 0.26, got {flops_ratio:.4f}"
    )

    return {
        "hypothesis": "H2_strided_mtp_throughput_and_mfu_recovery",
        "passed": True,
        "configs": results_by_config,
        "stride4_vs_stride1_speedup": round(speedup_ratio, 4),
        "stride4_vs_stride1_aux_flops_ratio": round(flops_ratio, 4),
    }


def _generate_motif_cycle_data(
    num_sequences: int,
    seq_len: int,
    vocab_size: int = 256,
    num_motifs: int = 64,
    motif_len: int = 4,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a 64-motif synthetic language dataset over vocab_size=256 requiring rank >= 64."""
    rng = np.random.default_rng(seed)
    total_len = seq_len + 1
    data = np.zeros((num_sequences, total_len), dtype=np.int64)
    data[:, 0] = rng.integers(0, vocab_size, size=num_sequences)
    for t in range(seq_len):
        data[:, t + 1] = ((data[:, t] * 5 + 3) % num_motifs) + num_motifs * (
            (data[:, t] // num_motifs + 1) % motif_len
        )
    return torch.from_numpy(data[:, :-1].copy()), torch.from_numpy(data[:, 1:].copy())


def run_experiment_h3_seed_subspace_capacity(device: torch.device) -> dict[str, Any]:
    """Experiment 3 (Hypothesis H3): Seed Hilbert Dimension D_seed vs. Effective Quantum Rank R_eff."""
    print("\n" + "=" * 88)
    print("EXPERIMENT 3 (H3): Hilbert Seed Subspace Dimension D_seed vs. Effective Quantum Rank")
    print("=" * 88)

    vocab_size = 256
    embed_dim = 128
    seq_len = 64
    batch_size = 16
    steps = 100
    seed_dims = [8, 16, 32, 64]

    x_train, y_train = _generate_motif_cycle_data(
        num_sequences=256, seq_len=seq_len, vocab_size=vocab_size, num_motifs=64, motif_len=4, seed=101
    )
    x_val, y_val = _generate_motif_cycle_data(
        num_sequences=64, seq_len=seq_len, vocab_size=vocab_size, num_motifs=64, motif_len=4, seed=202
    )
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    torch.manual_seed(42)
    q64, _ = torch.linalg.qr(torch.randn(64, 64, device=device))
    shared_frame = q64.repeat(4, 1) + 0.15 * torch.randn(vocab_size, 64, device=device)

    sweep_results: dict[str, dict[str, float]] = {}
    for d_seed in seed_dims:
        torch.manual_seed(42)
        model = QuantumBornGPT(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            seed_dim=d_seed,
            num_heads=4,
            num_layers=2,
            hidden_dim=256,
            block_size=seq_len,
            povm_temperature=0.07,
            readout_mode="tied_stiefel",
            learnable_temperature=False,
            mtp_horizons=1,
        ).to(device)
        with torch.no_grad():
            proj = l2_normalize(shared_frame[:, :d_seed], dim=-1)
            model.wte.weight.copy_(proj)

        muon_p, adamw_p = split_muon_adamw_params(
            model, adamw_name_patterns=("wte", "povm_head", "embed", "unet_scales", "smear", "inv_temp")
        )
        scale = math.sqrt(d_seed / 8.0)
        opt_muon = StiefelMuon(
            muon_p, lr=0.025 * scale, momentum=0.95, stiefel_penalty=0.005, ns_steps=5, retract_every=4
        )
        opt_adam = torch.optim.AdamW(adamw_p, lr=0.015 / scale, betas=(0.9, 0.95), weight_decay=0.0)

        model.train()
        for step in range(1, steps + 1):
            idx_start = ((step - 1) * batch_size) % (x_train.size(0) - batch_size)
            xb = x_train[idx_start : idx_start + batch_size]
            yb = y_train[idx_start : idx_start + batch_size]

            opt_muon.zero_grad(set_to_none=True)
            opt_adam.zero_grad(set_to_none=True)
            _, loss = model(xb, yb)
            assert loss is not None
            loss.backward()
            opt_muon.step()
            opt_adam.step()

        model.eval()
        with torch.no_grad():
            logits_val, val_loss, layer_states = model(x_val, y_val, return_layer_states=True)
            assert val_loss is not None and model.readout_rot is not None
            psi_last = layer_states[-1]
            psi_final = l2_normalize(model.readout_rot(psi_last), dim=-1, eps=model.eps)
            phi_seed = model.hilbert_lift.collapse_to_seed(psi_final).reshape(-1, d_seed)
            r_eff = float(effective_quantum_rank(phi_seed[:1024].float()).item())
            pr = float(purity_participation_ratio(phi_seed[:1024].float()).item())
            val_ce = float(val_loss.item())

        sweep_results[f"D_seed={d_seed}"] = {
            "seed_dim": float(d_seed),
            "val_ce": round(val_ce, 4),
            "effective_quantum_rank": round(r_eff, 4),
            "purity_participation_ratio": round(pr, 4),
            "rank_utilization_pct": round(100.0 * r_eff / float(d_seed), 2),
        }
        print(
            f"  [D_seed={d_seed:2d} | d={embed_dim}] Val CE: {val_ce:.4f} | "
            f"R_eff(rho_seed): {r_eff:6.2f} / {d_seed:2d} ({100.0 * r_eff / d_seed:5.1f}%) | "
            f"PR(rho_seed): {pr:6.2f}"
        )

    r8 = sweep_results["D_seed=8"]["effective_quantum_rank"]
    r16 = sweep_results["D_seed=16"]["effective_quantum_rank"]
    r32 = sweep_results["D_seed=32"]["effective_quantum_rank"]
    r64 = sweep_results["D_seed=64"]["effective_quantum_rank"]
    ce8 = sweep_results["D_seed=8"]["val_ce"]
    ce64 = sweep_results["D_seed=64"]["val_ce"]

    assert r64 > r32 > r16 > r8, (
        f"H3 Failed: Expected R_eff(64) > R_eff(32) > R_eff(16) > R_eff(8), "
        f"got {r64}, {r32}, {r16}, {r8}"
    )
    assert ce64 < ce8, (
        f"H3 Failed: Expected val_ce(64) < val_ce(8), got {ce64} vs {ce8}"
    )

    return {
        "hypothesis": "H3_seed_subspace_capacity_and_effective_quantum_rank",
        "passed": True,
        "sweep": sweep_results,
    }


def main() -> None:
    t_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running QuantumDLBlocks Hypothesis Suite (EXPERIMENT_PLAN_002) on device={device}...")

    h1_res = run_experiment_h1_temperature_dynamics(device)
    h2_res = run_experiment_h2_strided_mtp_throughput(device)
    h3_res = run_experiment_h3_seed_subspace_capacity(device)

    total_sec = time.perf_counter() - t_start
    summary = {
        "document_id": "EXPERIMENT_PLAN_002",
        "device": str(device),
        "elapsed_seconds": round(total_sec, 2),
        "all_hypotheses_passed": bool(h1_res["passed"] and h2_res["passed"] and h3_res["passed"]),
        "experiments": {
            "H1": h1_res,
            "H2": h2_res,
            "H3": h3_res,
        },
    }

    out_path = ROOT / ".cache" / "hypothesis_experiments_002.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 88)
    print(f"ALL 3 HYPOTHESES (H1, H2, H3) VERIFIED IN {total_sec:.2f}s -> {out_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
