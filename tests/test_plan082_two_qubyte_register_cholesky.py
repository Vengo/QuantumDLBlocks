"""Unit tests for `EXPERIMENT_PLAN_082.md` (`Wave 62`): Two 8-Qubit Registers ("Qubytes"), Step 1 Multi-Mode Qubyte Projection, Step 3 Latent 512D Contraction, & Rank-R Schmidt Entanglement."""

from __future__ import annotations

import torch

from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import DEFAULT_SOFTCAP_LOGIT, RAW_GPT2_VOCAB_SIZE
from qdlblocks.benchmarks.two_qubyte_register_cholesky_lab import (
    QUBYTE_HILBERT_DIM,
    QUBYTE_NUM_QUBITS,
    TwoQubyteRegisterCholeskyGPT,
    TwoQubyteRegisterCholeskySpec,
    TwoQubyteVariantMode,
    _SingleGemmStep3QubyteCEFn,
    build_8qubit_walsh_hadamard_and_spins,
    contract_step3_latent_512d_modes,
    initialize_two_qubyte_register_model_,
)


def test_8qubit_walsh_hadamard_orthogonality_and_hamming_weights() -> None:
    """Verify $H^{\\otimes 8}$ is an exact $256 \\times 256$ orthogonal matrix and Hamming weights match $\\binom{8}{k}$."""
    walsh_h8, hamming_w, spins = build_8qubit_walsh_hadamard_and_spins()
    eye_256 = torch.eye(QUBYTE_HILBERT_DIM, dtype=torch.float32)
    orth_err = float((walsh_h8.t() @ walsh_h8 - eye_256).abs().max().item())
    assert orth_err < 1e-6
    assert spins.shape == (QUBYTE_HILBERT_DIM, QUBYTE_NUM_QUBITS)
    assert set(spins.unique().tolist()) == {-1.0, 1.0}
    assert int((hamming_w == 0).sum().item()) == 1
    assert int((hamming_w == 1).sum().item()) == 8
    assert int((hamming_w == 2).sum().item()) == 28


def test_single_gemm_step3_qubyte_ce_autograd() -> None:
    """Verify `_SingleGemmStep3QubyteCEFn` matches exact reference Cross-Entropy in forward and backward."""
    torch.manual_seed(42)
    bsz, seq_len, k_rank, dim_aug, v_sz = 2, 8, 6, 528, RAW_GPT2_VOCAB_SIZE + 47
    h1 = torch.randn((bsz, seq_len, dim_aug), dtype=torch.float32, requires_grad=True)
    qh1 = torch.randn((bsz, seq_len, k_rank, 197), dtype=torch.float32, requires_grad=True)
    ql1 = torch.randn((bsz, seq_len, k_rank, 256), dtype=torch.float32, requires_grad=True)
    w1 = torch.randn((v_sz, dim_aug), dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, RAW_GPT2_VOCAB_SIZE, (bsz, seq_len), dtype=torch.long)

    loss_fn = _SingleGemmStep3QubyteCEFn.apply(h1, qh1, ql1, w1, targets)
    loss_fn.backward()

    h2 = h1.detach().clone().requires_grad_(True)
    qh2 = qh1.detach().clone().requires_grad_(True)
    ql2 = ql1.detach().clone().requires_grad_(True)
    w2 = w1.detach().clone().requires_grad_(True)
    z_ref = torch.matmul(h2.view(-1, dim_aug), w2.t())
    q_grid = torch.bmm(qh2.view(-1, k_rank, 197).transpose(1, 2), ql2.view(-1, k_rank, 256))
    z_ref[:, :RAW_GPT2_VOCAB_SIZE] = z_ref[:, :RAW_GPT2_VOCAB_SIZE] + q_grid.view(-1, 197 * 256)[:, :RAW_GPT2_VOCAB_SIZE]
    cap = float(DEFAULT_SOFTCAP_LOGIT)
    z_cap = cap * torch.tanh(z_ref / cap)
    loss_ref = torch.nn.functional.cross_entropy(z_cap, targets.view(-1))
    loss_ref.backward()

    assert torch.allclose(loss_fn.detach(), loss_ref.detach(), atol=1e-5)
    assert h1.grad is not None and h2.grad is not None
    assert qh1.grad is not None and qh2.grad is not None
    assert ql1.grad is not None and ql2.grad is not None
    assert w1.grad is not None and w2.grad is not None
    assert torch.allclose(h1.grad, h2.grad, atol=1e-4)
    assert torch.allclose(qh1.grad, qh2.grad, atol=1e-4)
    assert torch.allclose(ql1.grad, ql2.grad, atol=1e-4)
    assert torch.allclose(w1.grad, w2.grad, atol=1e-4)


def test_two_qubyte_register_cholesky_gpt_variants() -> None:
    """Verify `TwoQubyteRegisterCholeskyGPT` forward, backward, and zero-scan initialization for `Q0` and `Q3`."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_in = torch.randint(0, RAW_GPT2_VOCAB_SIZE, (2, 16), device=device, dtype=torch.long)
    targets = torch.randint(0, RAW_GPT2_VOCAB_SIZE, (2, 16), device=device, dtype=torch.long)

    for mode, step1, walsh, schmidt in [
        (TwoQubyteVariantMode.Q0_STEP3_ONLY_LATENT_512D_CONTRACTION, False, False, False),
        (TwoQubyteVariantMode.Q3_TWO_8QUBIT_SCHMIDT_ENTANGLED_CHOLESKY_R8, True, True, True),
    ]:
        spec = TwoQubyteRegisterCholeskySpec(
            variant_mode=mode,
            num_cholesky_modes=4,
            enable_step1_multimode_qubyte=step1,
            enable_walsh_pauli_qubytes=walsh,
            enable_schmidt_entanglement=schmidt,
        )
        model = TwoQubyteRegisterCholeskyGPT(spec).to(device=device)
        _, meta = initialize_two_qubyte_register_model_(model)
        assert meta["init_tokens_scanned"] == 0
        loss = model.forward_loss(idx_in, targets)
        assert bool(torch.isfinite(loss).item())
        loss.backward()
        assert model.w_det_512.grad is not None
        if schmidt:
            assert model.qubyte_synth.ising_coupling_j.grad is not None
            assert float(model.qubyte_synth.ising_coupling_j.grad.norm().item()) > 0.0
        del model
        if False:
            _ = contract_step3_latent_512d_modes
