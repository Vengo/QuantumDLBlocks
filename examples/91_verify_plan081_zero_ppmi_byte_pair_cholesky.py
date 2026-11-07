#!/usr/bin/env python3
"""Verification script for `EXPERIMENT_PLAN_081.md` (`Wave 61`): Zero-PPMI Byte-Pair & Chain-Rule Cholesky GPT."""

from __future__ import annotations

import json
import torch
import torch.nn.functional as F

from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import RAW_GPT2_VOCAB_SIZE
from qdlblocks.benchmarks.zero_ppmi_byte_pair_cholesky_lab import (
    LAST_BUCKET_VALID_LOW_BYTES,
    LAST_HIGH_BYTE_BUCKET,
    NUM_HIGH_BYTE_BUCKETS,
    NUM_LOW_BYTE_BUCKETS,
    ExactChainRuleTwoByteReadout,
    ZeroPPMIBytePairCholeskyGPT,
    ZeroPPMIBytePairCholeskySpec,
    ZeroPPMIBytePairVariantMode,
    audit_step1_vocabulary_detector_coverage,
    build_zero_data_bpe_rank_zipf_prior,
    decompose_token_uint16_byte_pair,
    initialize_zero_ppmi_byte_pair_model_,
    reconstruct_token_from_uint16_byte_pair,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_v = torch.arange(RAW_GPT2_VOCAB_SIZE, device=device, dtype=torch.long)
    b_hi, b_lo = decompose_token_uint16_byte_pair(all_v)
    recon = reconstruct_token_from_uint16_byte_pair(b_hi, b_lo)
    assert torch.equal(all_v, recon), "uint16 byte-pair decomposition must be 1-to-1 bijective!"
    assert int(b_hi.max().item()) == LAST_HIGH_BYTE_BUCKET
    assert int((b_hi == LAST_HIGH_BYTE_BUCKET).sum().item()) == LAST_BUCKET_VALID_LOW_BYTES

    chain = ExactChainRuleTwoByteReadout(sem_dim=448, id_dim=64).to(device)
    h_s0 = F.rms_norm(torch.randn(3, 448, device=device), (448,))
    h_s1 = F.rms_norm(torch.randn(3, 448, device=device), (448,))
    h_id = F.rms_norm(torch.randn(3, 8, 64, device=device), (64,))
    ctrl = torch.zeros(3, 8, 3, device=device)
    ctrl[:, :, 1] = 1.0
    _, log_p_hi_zero, _ = build_zero_data_bpe_rank_zipf_prior(50304, device=device)
    log_p_50257 = chain.evaluate_chunk_50257_log_probs(h_s0, h_s1, h_id, ctrl, log_p_hi_zero)
    prob_sum_err = float((torch.exp(log_p_50257).sum(dim=-1) - 1.0).abs().max().item())
    assert prob_sum_err < 1e-4, f"Chain-rule probabilities must sum to 1.0, got err={prob_sum_err}"

    idx = torch.randint(0, RAW_GPT2_VOCAB_SIZE, (2, 16), device=device)
    tgt = torch.randint(0, RAW_GPT2_VOCAB_SIZE, (2, 16), device=device)
    cov = audit_step1_vocabulary_detector_coverage(idx, tgt)
    assert cov["z2_byte_pair_schur_updated_vocab_pct"] == 100.0

    for mode, use_chain, use_car, r_modes in [
        (ZeroPPMIBytePairVariantMode.Z2_ZERO_PPMI_FACTORED_BYTE_PAIR_SCHUR, False, False, 8),
        (ZeroPPMIBytePairVariantMode.Z3_ZERO_PPMI_EXACT_CHAIN_RULE_TWOBYTE, True, False, 8),
        (ZeroPPMIBytePairVariantMode.Z4_ZERO_PPMI_HIERARCHICAL_BYTE_PAIR_R12, False, True, 12),
    ]:
        spec = ZeroPPMIBytePairCholeskySpec(
            variant_mode=mode,
            num_cholesky_modes=r_modes,
            enable_online_ema=True,
            enable_byte_pair_schur=True,
            use_exact_chain_rule_readout=use_chain,
            enable_hierarchical_bhi_carrier=use_car,
        )
        model = ZeroPPMIBytePairCholeskyGPT(spec).to(device)
        init_ms, meta = initialize_zero_ppmi_byte_pair_model_(model)
        assert meta["init_tokens_scanned"] == 0
        loss = model.forward_loss(idx, tgt)
        loss.backward()
        assert torch.isfinite(loss).item()

    print(
        json.dumps(
            {
                "plan": "EXPERIMENT_PLAN_081",
                "bijection_ok": True,
                "num_high_byte_buckets": NUM_HIGH_BYTE_BUCKETS,
                "num_low_byte_buckets": NUM_LOW_BYTE_BUCKETS,
                "chain_rule_prob_sum_err": prob_sum_err,
                "step1_coverage": cov,
                "init_ms": round(init_ms, 2),
                "init_tokens_scanned": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
