#!/usr/bin/env python3
"""Verification script for `EXPERIMENT_PLAN_082.md` (`Wave 62`): Two 8-Qubit Registers ("Qubytes"), Step 1 Multi-Mode Qubyte Projection, Step 3 Latent 512D Contraction, & Rank-R Schmidt Entanglement."""

from __future__ import annotations

import json
import torch

from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import RAW_GPT2_VOCAB_SIZE
from qdlblocks.benchmarks.two_qubyte_register_cholesky_lab import (
    QUBYTE_HILBERT_DIM,
    QUBYTE_NUM_QUBITS,
    TwoQubyteRegisterCholeskyGPT,
    TwoQubyteRegisterCholeskySpec,
    TwoQubyteVariantMode,
    build_8qubit_walsh_hadamard_and_spins,
    contract_step3_latent_512d_modes,
    initialize_two_qubyte_register_model_,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Verify exact 8-qubit Walsh-Hadamard orthogonality and Pauli Hamming weights
    walsh_h8, hamming_w, spins = build_8qubit_walsh_hadamard_and_spins(device=device)
    eye_256 = torch.eye(QUBYTE_HILBERT_DIM, device=device, dtype=torch.float32)
    orth_err = float((walsh_h8.t() @ walsh_h8 - eye_256).abs().max().item())
    assert orth_err < 1e-6, f"Walsh-Hadamard orthogonality error too high: {orth_err}"
    assert spins.shape == (QUBYTE_HILBERT_DIM, QUBYTE_NUM_QUBITS)
    assert int((hamming_w == 1).sum().item()) == 8
    assert int((hamming_w == 2).sum().item()) == 28

    # 2. Verify Step-3 latent 512D contraction shape and finite values
    h_base = torch.randn((2, 16, 464), device=device, dtype=torch.float32)
    h_sem_r = torch.randn((2, 16, 8, 448), device=device, dtype=torch.float32)
    h_id_r = torch.randn((2, 16, 8, 64), device=device, dtype=torch.float32)
    ctrls = torch.randn((2, 16, 8, 3), device=device, dtype=torch.float32)
    h_step3 = contract_step3_latent_512d_modes(h_base, h_sem_r, h_id_r, ctrls)
    assert h_step3.shape == (2, 16, 528)
    assert bool(torch.isfinite(h_step3).all().item())

    # 3. Verify Q0 (Step 3 only) and Q3 (Two-Qubyte + Step 1 + Step 3 + Schmidt Entanglement) forward & backward
    idx_in = torch.randint(0, RAW_GPT2_VOCAB_SIZE, (2, 32), device=device, dtype=torch.long)
    targets = torch.randint(0, RAW_GPT2_VOCAB_SIZE, (2, 32), device=device, dtype=torch.long)

    for mode, step1, walsh, schmidt in [
        (TwoQubyteVariantMode.Q0_STEP3_ONLY_LATENT_512D_CONTRACTION, False, False, False),
        (TwoQubyteVariantMode.Q3_TWO_8QUBIT_SCHMIDT_ENTANGLED_CHOLESKY_R8, True, True, True),
    ]:
        spec = TwoQubyteRegisterCholeskySpec(
            variant_mode=mode,
            num_cholesky_modes=8,
            enable_step1_multimode_qubyte=step1,
            enable_walsh_pauli_qubytes=walsh,
            enable_schmidt_entanglement=schmidt,
        )
        model = TwoQubyteRegisterCholeskyGPT(spec).to(device=device)
        init_ms, meta = initialize_two_qubyte_register_model_(model)
        assert meta["init_tokens_scanned"] == 0
        loss = model.forward_loss(idx_in, targets)
        assert bool(torch.isfinite(loss).item())
        loss.backward()
        assert model.w_det_512.grad is not None and bool(torch.isfinite(model.w_det_512.grad).all().item())
        if schmidt:
            assert model.qubyte_synth.v_det_pauli_hi.grad is not None
            assert float(model.qubyte_synth.v_det_pauli_hi.grad.norm().item()) > 0.0
        del model

    print(
        json.dumps(
            {
                "plan_id": "EXPERIMENT_PLAN_082",
                "walsh_h8_orthogonality_max_err": orth_err,
                "verification_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
