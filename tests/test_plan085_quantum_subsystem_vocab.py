r"""Unit tests for `EXPERIMENT_PLAN_085.md` (`Wave 65`): Fixed-Once Subsystem Quantum Vocabulary Architectures (`256 ⊕ 197` Lüders Instrument, `16^4` Right-Canonical MPS, & Shared `St(512, 128)` Frame)."""

from __future__ import annotations

import torch

from qdlblocks.benchmarks.quantum_subsystem_vocab_lab import (
    LUDERS_COARSE_OUTCOMES, LUDERS_FINE_MAX_OUTCOMES, LUDERS_FULL_ROW_CUTOFF, MPS_SITE3_FULL_PREFIX_CUTOFF,
    MPS_SITE3_MAX_OUTCOMES, QuantumSubsystemVocabGPT, QuantumSubsystemVocabSpec, VocabSubsystemMode,
    build_luders_hierarchical_zipf_priors, build_mps_4qudit_hierarchical_zipf_priors,
    build_subsystem_vocab_optimizers, step_subsystem_vocab_optimizers,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import RAW_GPT2_VOCAB_SIZE
from qdlblocks.benchmarks.zero_ppmi_byte_pair_cholesky_lab import build_zero_data_bpe_rank_zipf_prior


def test_luders_and_mps_hierarchical_zipf_priors_exact_50257_bijection() -> None:
    """Verify `256x197` Lüders and `16x16x16x13` MPS factorizations cover all `50,257` tokens with `< 1e-5` log-prior error."""
    log_p_flat, _, _ = build_zero_data_bpe_rank_zipf_prior(50304, device="cpu")
    log_p_valid = log_p_flat[:RAW_GPT2_VOCAB_SIZE] - torch.logsumexp(log_p_flat[:RAW_GPT2_VOCAB_SIZE], dim=0)
    v_ids = torch.arange(RAW_GPT2_VOCAB_SIZE, dtype=torch.long)

    b_coarse, b_fine_cond, valid_2d = build_luders_hierarchical_zipf_priors(RAW_GPT2_VOCAB_SIZE, device="cpu")
    assert int(valid_2d.sum().item()) == RAW_GPT2_VOCAB_SIZE
    assert LUDERS_FULL_ROW_CUTOFF * LUDERS_FINE_MAX_OUTCOMES + (LUDERS_COARSE_OUTCOMES - LUDERS_FULL_ROW_CUTOFF) * (LUDERS_FINE_MAX_OUTCOMES - 1) == RAW_GPT2_VOCAB_SIZE
    recon_luders = b_coarse[v_ids & 0xFF] + b_fine_cond[v_ids & 0xFF, v_ids >> 8]
    assert float((recon_luders - log_p_valid).abs().max().item()) < 1e-5

    b0, b1, b2, b3, valid_s3 = build_mps_4qudit_hierarchical_zipf_priors(RAW_GPT2_VOCAB_SIZE, device="cpu")
    assert int(valid_s3.sum().item()) == RAW_GPT2_VOCAB_SIZE
    assert MPS_SITE3_FULL_PREFIX_CUTOFF * MPS_SITE3_MAX_OUTCOMES + (4096 - MPS_SITE3_FULL_PREFIX_CUTOFF) * (MPS_SITE3_MAX_OUTCOMES - 1) == RAW_GPT2_VOCAB_SIZE
    s0, s1, s2, s3 = v_ids & 0xF, (v_ids >> 4) & 0xF, (v_ids >> 8) & 0xF, v_ids >> 12
    recon_mps = b0[s0] + b1[s0, s1] + b2[v_ids & 0xFF, s2] + b3[v_ids & 0xFFF, s3]
    assert float((recon_mps - log_p_valid).abs().max().item()) < 1e-5


def test_luders_sequential_instrument_partition_function_and_compression() -> None:
    """Verify `LUDERS_256X256_INSTRUMENT` achieves `> 35x` vocab compression, `Z == 1.0`, and exact NLL consistency."""
    spec = QuantumSubsystemVocabSpec(vocab_mode=VocabSubsystemMode.LUDERS_256X256_INSTRUMENT, num_layers=2, kraus_rank=2)
    model = QuantumSubsystemVocabGPT(spec)
    model.initialize_model_()
    budget = model.parameter_budget_summary()
    assert budget["compression_vs_flat_51p5m_x"] > 35.0

    idx = torch.randint(0, RAW_GPT2_VOCAB_SIZE, (2, 4), dtype=torch.long)
    targets = torch.randint(0, RAW_GPT2_VOCAB_SIZE, (2, 4), dtype=torch.long)
    part = model.verify_exact_50257_partition_function(idx)
    assert part["max_partition_err"] < 1e-5

    loss_fast, _ = model.forward_loss_and_metrics(idx, targets)
    u_voc, u_vac, rho_re, rho_im = model.encode_trunk(idx)
    full_lp = model.boundary.full_50257_log_probs(u_voc, u_vac, model.vac_rest_mass, rho_re, rho_im, model.current_analytical_beta())
    loss_exact = -full_lp.gather(-1, targets.reshape(-1, 1)).mean()
    assert float((loss_fast - loss_exact).abs().item()) < 1e-5


def test_right_canonical_mps_born_chain_stiefel_gauge_and_partition_function() -> None:
    """Verify `MPS_16POW4_BORN_CHAIN` maintains `A^[k] in St(2048, 128)`, `Z == 1.0`, and `> 30x` vocab compression."""
    spec = QuantumSubsystemVocabSpec(vocab_mode=VocabSubsystemMode.MPS_16POW4_BORN_CHAIN, num_layers=2, kraus_rank=2)
    model = QuantumSubsystemVocabGPT(spec)
    model.initialize_model_()
    budget = model.parameter_budget_summary()
    assert budget["compression_vs_flat_51p5m_x"] > 30.0

    idx = torch.randint(0, RAW_GPT2_VOCAB_SIZE, (2, 4), dtype=torch.long)
    targets = torch.randint(0, RAW_GPT2_VOCAB_SIZE, (2, 4), dtype=torch.long)
    part = model.verify_exact_50257_partition_function(idx)
    assert part["max_partition_err"] < 1e-5

    opts = build_subsystem_vocab_optimizers(model)
    for opt in opts:
        opt.zero_grad(set_to_none=True)
    loss, _ = model.forward_loss_and_metrics(idx, targets)
    loss.backward()
    step_subsystem_vocab_optimizers(model, opts, tokens_in_step=8, lr_mult=1.0, mom_beta=0.50)
    telem = model.get_manifold_telemetry()
    assert telem["kraus_stiefel_err"] < 1e-4
    assert telem["boundary_stiefel_err"] < 1e-4


def test_shared_stiefel_subspace_frame_128d_step() -> None:
    """Verify `SHARED_STIEFEL_128D_FRAME` (`St(512, 128)` + tied `128D` sphere) preserves Stiefel & sphere invariants."""
    spec = QuantumSubsystemVocabSpec(vocab_mode=VocabSubsystemMode.SHARED_STIEFEL_128D_FRAME, num_layers=2, kraus_rank=2)
    model = QuantumSubsystemVocabGPT(spec)
    model.initialize_model_()
    budget = model.parameter_budget_summary()
    assert budget["compression_vs_flat_51p5m_x"] > 7.5

    idx = torch.randint(0, RAW_GPT2_VOCAB_SIZE, (2, 4), dtype=torch.long)
    targets = torch.randint(0, RAW_GPT2_VOCAB_SIZE, (2, 4), dtype=torch.long)
    opts = build_subsystem_vocab_optimizers(model)
    for opt in opts:
        opt.zero_grad(set_to_none=True)
    loss, _ = model.forward_loss_and_metrics(idx, targets)
    loss.backward()
    step_subsystem_vocab_optimizers(model, opts, tokens_in_step=8, lr_mult=1.0, mom_beta=0.50)
    telem = model.get_manifold_telemetry()
    assert telem["boundary_stiefel_err"] < 1e-4
