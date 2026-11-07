"""Unit tests for `EXPERIMENT_PLAN_081.md` (`Wave 61`): Zero-PPMI Byte-Pair & Chain-Rule Cholesky GPT."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import RAW_GPT2_VOCAB_SIZE
from qdlblocks.benchmarks.zero_ppmi_byte_pair_cholesky_lab import (
    LAST_BUCKET_VALID_LOW_BYTES,
    LAST_HIGH_BYTE_BUCKET,
    NUM_HIGH_BYTE_BUCKETS,
    NUM_LOW_BYTE_BUCKETS,
    ExactChainRuleTwoByteReadout,
    OnlineCausalZipfTracker,
    ZeroPPMIBytePairCholeskyGPT,
    ZeroPPMIBytePairCholeskySpec,
    ZeroPPMIBytePairVariantMode,
    audit_step1_vocabulary_detector_coverage,
    build_zero_data_bpe_rank_zipf_prior,
    decompose_token_uint16_byte_pair,
    initialize_zero_ppmi_byte_pair_model_,
    reconstruct_token_from_uint16_byte_pair,
)


def test_uint16_radix_byte_pair_bijection() -> None:
    """Verify 1-to-1 bijection between `0..50256` and `(b_hi, b_lo)`."""
    all_v = torch.arange(RAW_GPT2_VOCAB_SIZE, dtype=torch.long)
    b_hi, b_lo = decompose_token_uint16_byte_pair(all_v)
    recon = reconstruct_token_from_uint16_byte_pair(b_hi, b_lo)
    assert torch.equal(all_v, recon)
    assert int(b_hi.min().item()) == 0
    assert int(b_hi.max().item()) == LAST_HIGH_BYTE_BUCKET == NUM_HIGH_BYTE_BUCKETS - 1
    assert int(b_lo.min().item()) == 0
    assert int(b_lo.max().item()) == NUM_LOW_BYTE_BUCKETS - 1
    assert int((b_hi == LAST_HIGH_BYTE_BUCKET).sum().item()) == LAST_BUCKET_VALID_LOW_BYTES


def test_zero_data_zipf_prior_and_online_ema_tracker() -> None:
    """Verify Step-0 analytical BPE-rank Zipf prior normalizes and online EMA updates only during training."""
    log_p_tok, log_p_hi, radial = build_zero_data_bpe_rank_zipf_prior(50304)
    assert abs(float(torch.exp(log_p_tok).sum().item()) - 1.0) < 1e-4
    assert abs(float(torch.exp(log_p_hi).sum().item()) - 1.0) < 1e-4
    assert radial.shape == (50304,)
    assert float(log_p_tok[300].item()) > float(log_p_tok[45000].item())

    tracker = OnlineCausalZipfTracker(vocab_size=50304, enable_online_ema=True)
    p0_before, _ = tracker.current_log_priors()
    batch = torch.full((2, 16), 300, dtype=torch.long)
    tracker.update_from_training_batch_(batch)
    p0_after, _ = tracker.current_log_priors()
    assert float(tracker.batches_seen.item()) == 1.0
    assert float(p0_after[300].item()) > float(p0_before[300].item())


def test_exact_chain_rule_two_byte_normalization_and_backward() -> None:
    """Verify `ExactChainRuleTwoByteReadout` sums to 1.0 over `50,257` tokens and matches training NLL."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = ExactChainRuleTwoByteReadout(sem_dim=448, id_dim=64).to(device)
    h_s0 = F.rms_norm(torch.randn(6, 448, device=device), (448,)).requires_grad_(True)
    h_s1 = F.rms_norm(torch.randn(6, 448, device=device), (448,)).requires_grad_(True)
    h_id = F.rms_norm(torch.randn(6, 4, 64, device=device), (64,)).requires_grad_(True)
    ctrl = torch.zeros(6, 4, 3, device=device)
    ctrl[:, :, 1] = 1.0
    _, log_p_hi_zero, _ = build_zero_data_bpe_rank_zipf_prior(50304, device=device)
    targets = torch.tensor([15, 257, 1024, 12345, 50255, 50256], device=device, dtype=torch.long)

    train_loss = head.forward_train_loss(h_s0, h_s1, h_id, ctrl, log_p_hi_zero, targets)
    train_loss.backward()
    assert h_s0.grad is not None and torch.isfinite(h_s0.grad).all()
    assert head.w_hi_sem.grad is not None and torch.isfinite(head.w_hi_sem.grad).all()

    log_p_50257 = head.evaluate_chunk_50257_log_probs(h_s0.detach(), h_s1.detach(), h_id.detach(), ctrl, log_p_hi_zero)
    assert log_p_50257.shape == (6, RAW_GPT2_VOCAB_SIZE)
    prob_sums = torch.exp(log_p_50257).sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-4)
    eval_nll = F.nll_loss(log_p_50257, targets)
    assert abs(float(train_loss.item()) - float(eval_nll.item())) < 1e-4


def test_step1_detector_coverage_and_zero_ppmi_variants() -> None:
    """Verify `100%` Step-1 detector update coverage in `Z2` and zero-data initialization across `Z0..Z4`."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx = torch.randint(0, RAW_GPT2_VOCAB_SIZE, (2, 16), device=device)
    tgt = torch.randint(0, RAW_GPT2_VOCAB_SIZE, (2, 16), device=device)
    cov = audit_step1_vocabulary_detector_coverage(idx, tgt)
    assert cov["z2_byte_pair_schur_updated_vocab_pct"] == 100.0
    assert cov["z0_unfactored_updated_vocab_pct"] < 5.0

    for mode, use_chain, use_car in [
        (ZeroPPMIBytePairVariantMode.Z2_ZERO_PPMI_FACTORED_BYTE_PAIR_SCHUR, False, False),
        (ZeroPPMIBytePairVariantMode.Z3_ZERO_PPMI_EXACT_CHAIN_RULE_TWOBYTE, True, False),
        (ZeroPPMIBytePairVariantMode.Z4_ZERO_PPMI_HIERARCHICAL_BYTE_PAIR_R12, False, True),
    ]:
        spec = ZeroPPMIBytePairCholeskySpec(
            variant_mode=mode,
            num_cholesky_modes=4,
            enable_online_ema=True,
            enable_byte_pair_schur=True,
            use_exact_chain_rule_readout=use_chain,
            enable_hierarchical_bhi_carrier=use_car,
        )
        model = ZeroPPMIBytePairCholeskyGPT(spec).to(device)
        _, meta = initialize_zero_ppmi_byte_pair_model_(model)
        assert model.init_tokens_scanned == 0
        assert meta["init_tokens_scanned"] == 0
        loss = model.forward_loss(idx, tgt)
        loss.backward()
        assert torch.isfinite(loss).item()
