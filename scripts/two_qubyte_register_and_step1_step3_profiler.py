#!/usr/bin/env python3
"""Two 8-Qubit Registers ("Qubytes"), Walsh-Hadamard Pauli-$Z$ Spectrum, & Step-1/Step-3 Readout Throughput Profiler (`EXPERIMENT_PLAN_082.md` / `Wave 62`)."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import (
    DEFAULT_SHALLOW_BATCH_SIZE,
    DEFAULT_SHALLOW_SEQ_LEN,
    RAW_GPT2_VOCAB_SIZE,
)
from qdlblocks.benchmarks.two_qubyte_register_cholesky_lab import (
    QUBYTE_HILBERT_DIM,
    QUBYTE_NUM_QUBITS,
    TwoQubyteRegisterCholeskyGPT,
    TwoQubyteRegisterCholeskySpec,
    TwoQubyteVariantMode,
    build_8qubit_walsh_hadamard_and_spins,
    initialize_two_qubyte_register_model_,
)
from qdlblocks.benchmarks.zero_ppmi_byte_pair_cholesky_lab import (
    NUM_HIGH_BYTE_BUCKETS,
    NUM_LOW_BYTE_BUCKETS,
    ZeroPPMIBytePairCholeskyGPT,
    ZeroPPMIBytePairCholeskySpec,
    ZeroPPMIBytePairVariantMode,
    decompose_token_uint16_byte_pair,
    initialize_zero_ppmi_byte_pair_model_,
)


def audit_8qubit_walsh_hadamard_and_16qubit_shard0(
    val_shard_path: Path, max_tokens: int = 10_485_760
) -> dict[str, Any]:
    r"""Audit $H^{\otimes 8}$ unitarity, 16-qubit marginal entropies $H(q_{15}\dots q_0)$, and Pauli-$Z^{\otimes S}$ Hamming-weight spectrum."""
    walsh_h8, hamming_w, spins = build_8qubit_walsh_hadamard_and_spins()
    eye_256 = torch.eye(QUBYTE_HILBERT_DIM, dtype=torch.float32)
    orth_err = float((walsh_h8.t() @ walsh_h8 - eye_256).abs().max().item())

    mm = np.memmap(val_shard_path, dtype=np.uint16, mode="r", offset=256 * 4)
    n_tok = min(int(max_tokens), len(mm))
    tok_t = torch.from_numpy(np.asarray(mm[:n_tok], dtype=np.int64))

    counts_v = torch.bincount(tok_t.clamp(0, RAW_GPT2_VOCAB_SIZE - 1), minlength=QUBYTE_HILBERT_DIM * QUBYTE_HILBERT_DIM).float()
    p_v = counts_v / counts_v.sum().clamp_min(1.0)
    p_grid_256 = p_v.view(QUBYTE_HILBERT_DIM, QUBYTE_HILBERT_DIM)

    p_hi_256 = p_grid_256.sum(dim=1)
    p_lo_256 = p_grid_256.sum(dim=0)

    # 16-qubit bit marginal probabilities P(q_k = 1) and binary Shannon entropies H(q_k) in bits and nats
    qubit_entropies_bits: list[float] = []
    qubit_p1: list[float] = []
    for k in range(2 * QUBYTE_NUM_QUBITS):
        if k < QUBYTE_NUM_QUBITS:
            mask_k = ((torch.arange(QUBYTE_HILBERT_DIM) >> k) & 1).float()
            pk = float((p_lo_256 * mask_k).sum().item())
        else:
            b_idx = k - QUBYTE_NUM_QUBITS
            mask_k = ((torch.arange(QUBYTE_HILBERT_DIM) >> b_idx) & 1).float()
            pk = float((p_hi_256 * mask_k).sum().item())
        pk_c = min(max(pk, 1e-9), 1.0 - 1e-9)
        h_bits = -(pk_c * math.log2(pk_c) + (1.0 - pk_c) * math.log2(1.0 - pk_c))
        qubit_p1.append(round(pk, 4))
        qubit_entropies_bits.append(round(h_bits, 4))

    # Pauli-Z^{\otimes S} Walsh-Hadamard spectral energy capture across Hamming weights |S| = 0..8
    log_p_hi = torch.log(p_hi_256 + 1e-7)
    log_p_hi_centered = log_p_hi - log_p_hi.mean()
    pauli_coeffs_hi = torch.matmul(walsh_h8.t(), log_p_hi_centered)
    energy_hi = pauli_coeffs_hi.square()
    total_energy_hi = float(energy_hi.sum().clamp_min(1e-12).item())

    log_p_lo = torch.log(p_lo_256 + 1e-7)
    log_p_lo_centered = log_p_lo - log_p_lo.mean()
    pauli_coeffs_lo = torch.matmul(walsh_h8.t(), log_p_lo_centered)
    energy_lo = pauli_coeffs_lo.square()
    total_energy_lo = float(energy_lo.sum().clamp_min(1e-12).item())

    pauli_order_capture: dict[str, dict[str, float]] = {}
    cum_hi, cum_lo = 0.0, 0.0
    for order_k in range(QUBYTE_NUM_QUBITS + 1):
        mask_o = hamming_w == float(order_k)
        frac_hi = float(energy_hi[mask_o].sum().item()) / total_energy_hi
        frac_lo = float(energy_lo[mask_o].sum().item()) / total_energy_lo
        cum_hi += frac_hi
        cum_lo += frac_lo
        pauli_order_capture[f"weight_{order_k}"] = {
            "num_pauli_strings": int(mask_o.sum().item()),
            "hi_energy_pct": round(100.0 * frac_hi, 2),
            "hi_cumulative_pct": round(100.0 * cum_hi, 2),
            "lo_energy_pct": round(100.0 * frac_lo, 2),
            "lo_cumulative_pct": round(100.0 * cum_lo, 2),
        }

    return {
        "val_tokens_audited": n_tok,
        "walsh_h8_orthogonality_max_err": orth_err,
        "spins_shape": list(spins.shape),
        "hi_qubyte_sum_entropy_bits": round(sum(qubit_entropies_bits[8:16]), 4),
        "lo_qubyte_sum_entropy_bits": round(sum(qubit_entropies_bits[0:8]), 4),
        "total_16qubit_marginal_entropy_bits": round(sum(qubit_entropies_bits), 4),
        "qubit_p1_lsb_to_msb": qubit_p1,
        "qubit_entropy_bits_lsb_to_msb": qubit_entropies_bits,
        "pauli_hamming_weight_energy_capture": pauli_order_capture,
        "low_order_k_le_3_hi_capture_pct": pauli_order_capture["weight_3"]["hi_cumulative_pct"],
        "low_order_k_le_3_lo_capture_pct": pauli_order_capture["weight_3"]["lo_cumulative_pct"],
    }


def audit_step1_algebraic_equivalence_and_throughput(
    device: torch.device, batch_size: int = DEFAULT_SHALLOW_BATCH_SIZE, seq_len: int = DEFAULT_SHALLOW_SEQ_LEN
) -> dict[str, Any]:
    r"""Verify exact Step-1 algebraic equivalence (`453` vs `50,257` dot products/mode) and benchmark training step throughput."""
    torch.manual_seed(1337)
    n_modes, d_model = 8, 512
    h_modes = torch.randn((batch_size, seq_len, n_modes, d_model), device=device, dtype=torch.float32)
    u_hi = torch.randn((NUM_HIGH_BYTE_BUCKETS, d_model), device=device, dtype=torch.float32)
    u_lo = torch.randn((NUM_LOW_BYTE_BUCKETS, d_model), device=device, dtype=torch.float32)

    all_v = torch.arange(RAW_GPT2_VOCAB_SIZE, device=device, dtype=torch.long)
    b_hi_v, b_lo_v = decompose_token_uint16_byte_pair(all_v)
    w_factored_50257 = 0.30 * u_hi[b_hi_v] + 0.25 * u_lo[b_lo_v]

    # Direct 50,257-way multi-mode matmul vs Step-1 (197 + 256 = 453)-way projection
    h_sample = h_modes[:2, :32]  # (2, 32, 8, 512)
    z_direct = torch.matmul(h_sample, w_factored_50257.t())
    z_hi = 0.30 * torch.matmul(h_sample, u_hi.t())
    z_lo = 0.25 * torch.matmul(h_sample, u_lo.t())
    z_step1_grid = z_hi.unsqueeze(-1) + z_lo.unsqueeze(-2)
    z_step1_50257 = z_step1_grid.reshape(2, 32, n_modes, NUM_HIGH_BYTE_BUCKETS * NUM_LOW_BYTE_BUCKETS)[
        :, :, :, :RAW_GPT2_VOCAB_SIZE
    ]
    step1_max_err = float((z_direct - z_step1_50257).abs().max().item())
    flop_reduction = float(RAW_GPT2_VOCAB_SIZE) / float(NUM_HIGH_BYTE_BUCKETS + NUM_LOW_BYTE_BUCKETS)

    # Benchmark full forward + backward step wall-clock latency across Z2 vs Q0 (Step 3) vs Q1 (Step 1+3) vs Q3 (R=8) vs Q4 (R=12)
    idx_in = torch.randint(0, RAW_GPT2_VOCAB_SIZE, (batch_size, seq_len), device=device, dtype=torch.long)
    targets = torch.randint(0, RAW_GPT2_VOCAB_SIZE, (batch_size, seq_len), device=device, dtype=torch.long)

    bench_rows: dict[str, dict[str, float]] = {}

    # 1. Z2 baseline (R=8, 1+R full-vocab matmuls)
    z2_model = ZeroPPMIBytePairCholeskyGPT(
        ZeroPPMIBytePairCholeskySpec(
            variant_mode=ZeroPPMIBytePairVariantMode.Z2_ZERO_PPMI_FACTORED_BYTE_PAIR_SCHUR,
            num_cholesky_modes=8,
            enable_online_ema=True,
            enable_byte_pair_schur=True,
        )
    ).to(device=device)
    initialize_zero_ppmi_byte_pair_model_(z2_model)
    for _ in range(2):
        z2_model.zero_grad(set_to_none=True)
        z2_model.forward_loss(idx_in, targets).backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_iters = 5
    for _ in range(n_iters):
        z2_model.zero_grad(set_to_none=True)
        z2_model.forward_loss(idx_in, targets).backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    z2_ms = ((time.perf_counter() - t0) * 1000.0) / float(n_iters)
    bench_rows["Z2_R8_MULTIMODE_50304_GEMM"] = {
        "step_ms": round(z2_ms, 2),
        "tok_per_sec": round((batch_size * seq_len * 1000.0) / z2_ms, 1),
        "speedup_vs_z2": 1.0,
    }
    del z2_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    q_configs = [
        (
            "Q0_STEP3_ONLY_R8",
            TwoQubyteRegisterCholeskySpec(
                variant_mode=TwoQubyteVariantMode.Q0_STEP3_ONLY_LATENT_512D_CONTRACTION,
                num_cholesky_modes=8,
                enable_step1_multimode_qubyte=False,
                enable_walsh_pauli_qubytes=False,
                enable_schmidt_entanglement=False,
                enable_hierarchical_bhi_carrier=False,
            ),
        ),
        (
            "Q1_STEP1_PLUS_STEP3_R8",
            TwoQubyteRegisterCholeskySpec(
                variant_mode=TwoQubyteVariantMode.Q1_STEP1_PLUS_STEP3_FACTORED_MULTIMODE,
                num_cholesky_modes=8,
                enable_step1_multimode_qubyte=True,
                enable_walsh_pauli_qubytes=False,
                enable_schmidt_entanglement=False,
                enable_hierarchical_bhi_carrier=False,
            ),
        ),
        (
            "Q3_TWO_QUBYTE_SCHMIDT_R8",
            TwoQubyteRegisterCholeskySpec(
                variant_mode=TwoQubyteVariantMode.Q3_TWO_8QUBIT_SCHMIDT_ENTANGLED_CHOLESKY_R8,
                num_cholesky_modes=8,
                enable_step1_multimode_qubyte=True,
                enable_walsh_pauli_qubytes=True,
                enable_schmidt_entanglement=True,
                enable_hierarchical_bhi_carrier=True,
            ),
        ),
        (
            "Q4_TWO_QUBYTE_SCHMIDT_R12",
            TwoQubyteRegisterCholeskySpec(
                variant_mode=TwoQubyteVariantMode.Q4_TWO_8QUBIT_SCHMIDT_ENTANGLED_CHOLESKY_R12,
                num_cholesky_modes=12,
                enable_step1_multimode_qubyte=True,
                enable_walsh_pauli_qubytes=True,
                enable_schmidt_entanglement=True,
                enable_hierarchical_bhi_carrier=True,
            ),
        ),
    ]

    for label, q_spec in q_configs:
        q_model = TwoQubyteRegisterCholeskyGPT(q_spec).to(device=device)
        initialize_two_qubyte_register_model_(q_model)
        for _ in range(2):
            q_model.zero_grad(set_to_none=True)
            q_model.forward_loss(idx_in, targets).backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            q_model.zero_grad(set_to_none=True)
            q_model.forward_loss(idx_in, targets).backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        q_ms = ((time.perf_counter() - t0) * 1000.0) / float(n_iters)
        bench_rows[label] = {
            "step_ms": round(q_ms, 2),
            "tok_per_sec": round((batch_size * seq_len * 1000.0) / q_ms, 1),
            "speedup_vs_z2": round(z2_ms / max(1e-6, q_ms), 2),
        }
        del q_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "step1_algebraic_equivalence_max_err": step1_max_err,
        "multimode_dot_products_full_vocab": RAW_GPT2_VOCAB_SIZE,
        "multimode_dot_products_step1_qubyte": NUM_HIGH_BYTE_BUCKETS + NUM_LOW_BYTE_BUCKETS,
        "multimode_flop_reduction_factor": round(flop_reduction, 2),
        "step_throughput_comparison": bench_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Two 8-Qubit Registers & Step-1/Step-3 Readout Profiler (Plan 082).")
    parser.add_argument(
        "--val-shard",
        type=str,
        default="/home/festivity/agy/QuantumDLBlocks/.cache/fineweb_100M/fineweb_val_000000.bin",
    )
    parser.add_argument("--json-out", type=str, default=".cache/two_qubyte_step1_step3_profile.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qubyte_audit = audit_8qubit_walsh_hadamard_and_16qubit_shard0(Path(args.val_shard))
    step13_audit = audit_step1_algebraic_equivalence_and_throughput(device=device)

    payload = {
        "plan_id": "EXPERIMENT_PLAN_082",
        "wave_id": "W62",
        "two_8qubit_register_shard0_audit": qubyte_audit,
        "step1_and_step3_throughput_audit": step13_audit,
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
