#!/usr/bin/env python3
"""Diagnostic CLI Profiler for Deterministic `uint16` Radix Byte-Pair `(b_hi, b_lo)` & Zero-PPMI Cold-Start (`EXPERIMENT_PLAN_081.md` / `Wave 61`).

Audits:
1. Bijective `uint16` radix decomposition $v = 256 b_{\\text{hi}}(v) + b_{\\text{lo}}(v)$ across all `50,257` GPT-2 tokens.
2. Unigram entropy decomposition $H(v) = H(b_{\\text{hi}}) + H(b_{\\text{lo}} \\mid b_{\\text{hi}})$ on FineWeb Shard 0.
3. Zero-data analytical BPE-rank Zipf prior CE (`0` tokens scanned) and online EMA trajectory over training batches `1..50`.
4. Step-1 vocabulary detector update coverage (`%` of `50,257` tokens updated on Step 1 in `Z0` vs `Z2`).
5. Exact two-byte chain-rule probability normalization $\\sum_{v=0}^{50,256} P_{\\text{chain}}(v \\mid h_t) = 1.0$.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import RAW_GPT2_VOCAB_SIZE, LongSeqFineWebStream
from qdlblocks.benchmarks.zero_ppmi_byte_pair_cholesky_lab import (
    LAST_BUCKET_VALID_LOW_BYTES,
    LAST_HIGH_BYTE_BUCKET,
    NUM_HIGH_BYTE_BUCKETS,
    NUM_LOW_BYTE_BUCKETS,
    ExactChainRuleTwoByteReadout,
    OnlineCausalZipfTracker,
    audit_step1_vocabulary_detector_coverage,
    build_zero_data_bpe_rank_zipf_prior,
    decompose_token_uint16_byte_pair,
    reconstruct_token_from_uint16_byte_pair,
)


@torch.no_grad()
def profile_shard0_byte_pair_entropy_and_online_ema(stream: LongSeqFineWebStream) -> dict[str, Any]:
    """Compute exact Shard-0 `(b_hi, b_lo)` entropy split and online EMA unigram CE trajectory."""
    val_tokens = torch.cat([y.reshape(-1) for _, y in stream.val_batches], dim=0)
    b_hi_val, b_lo_val = decompose_token_uint16_byte_pair(val_tokens)

    tok_counts = torch.bincount(val_tokens, minlength=RAW_GPT2_VOCAB_SIZE).double()
    p_tok = tok_counts / tok_counts.sum().clamp_min(1.0)
    nz_tok = p_tok > 0
    h_tok_nats = float(-(p_tok[nz_tok] * torch.log(p_tok[nz_tok])).sum().item())

    hi_counts = torch.bincount(b_hi_val, minlength=NUM_HIGH_BYTE_BUCKETS).double()
    p_hi = hi_counts / hi_counts.sum().clamp_min(1.0)
    nz_hi = p_hi > 0
    h_hi_nats = float(-(p_hi[nz_hi] * torch.log(p_hi[nz_hi])).sum().item())
    h_lo_given_hi_nats = h_tok_nats - h_hi_nats

    share_bhi_0 = float(p_hi[0].item())
    share_bhi_1_to_3 = float(p_hi[1:4].sum().item())
    share_bhi_0_to_3 = share_bhi_0 + share_bhi_1_to_3

    vocab_size = 50304
    uniform_ce = float(torch.log(torch.tensor(float(vocab_size))).item())
    log_p_zero, log_p_hi_zero, _ = build_zero_data_bpe_rank_zipf_prior(vocab_size, device=val_tokens.device)
    step0_zipf_ce = float(F.nll_loss(log_p_zero.unsqueeze(0).expand(val_tokens.numel(), -1), val_tokens).item())
    step0_hi_ce = float(F.nll_loss(log_p_hi_zero.unsqueeze(0).expand(b_hi_val.numel(), -1), b_hi_val).item())

    tracker = OnlineCausalZipfTracker(vocab_size=vocab_size, enable_online_ema=True).to(stream.device)
    ema_trajectory: list[dict[str, float]] = [
        {"batches_seen": 0.0, "tokens_seen": 0.0, "val_unigram_ce": round(step0_zipf_ce, 4), "val_bhi_ce": round(step0_hi_ce, 4)}
    ]
    checkpoints = {1, 5, 10, 25, 50}
    tokens_per_batch = stream.batch_size * stream.seq_len
    for b_idx in range(1, 51):
        idx_in, _ = stream.sample_train_batch(b_idx)
        tracker.update_from_training_batch_(idx_in)
        if b_idx in checkpoints:
            log_p_cur, log_phi_cur = tracker.current_log_priors()
            ce_cur = float(F.nll_loss(log_p_cur.unsqueeze(0).expand(val_tokens.numel(), -1), val_tokens).item())
            ce_hi_cur = float(F.nll_loss(log_phi_cur.unsqueeze(0).expand(b_hi_val.numel(), -1), b_hi_val).item())
            ema_trajectory.append({
                "batches_seen": float(b_idx),
                "tokens_seen": float(b_idx * tokens_per_batch),
                "val_unigram_ce": round(ce_cur, 4),
                "val_bhi_ce": round(ce_hi_cur, 4),
            })

    return {
        "shard0_evaluated_tokens": int(val_tokens.numel()),
        "shard0_unigram_entropy_nats": round(h_tok_nats, 4),
        "shard0_b_hi_entropy_nats": round(h_hi_nats, 4),
        "shard0_b_lo_given_b_hi_entropy_nats": round(h_lo_given_hi_nats, 4),
        "b_hi_entropy_fraction": round(h_hi_nats / max(1e-8, h_tok_nats), 4),
        "b_hi_0_ascii_token_share": round(share_bhi_0, 4),
        "b_hi_1_to_3_top_bpe_token_share": round(share_bhi_1_to_3, 4),
        "b_hi_0_to_3_cumulative_share": round(share_bhi_0_to_3, 4),
        "uniform_50304_ce_nats": round(uniform_ce, 4),
        "step0_zero_data_zipf_ce_nats": round(step0_zipf_ce, 4),
        "step0_ce_improvement_over_uniform_nats": round(uniform_ce - step0_zipf_ce, 4),
        "online_ema_trajectory": ema_trajectory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile uint16 byte-pair bijection and Zero-PPMI cold-start.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--json-out", type=str, default=".cache/byte_pair_zero_ppmi_profile.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_v = torch.arange(RAW_GPT2_VOCAB_SIZE, device=device, dtype=torch.long)
    b_hi, b_lo = decompose_token_uint16_byte_pair(all_v)
    recon_v = reconstruct_token_from_uint16_byte_pair(b_hi, b_lo)
    bijection_exact = bool(torch.equal(all_v, recon_v))
    max_b_hi = int(b_hi.max().item())
    bucket_196_count = int((b_hi == LAST_HIGH_BYTE_BUCKET).sum().item())

    chain_head = ExactChainRuleTwoByteReadout(sem_dim=448, id_dim=64).to(device)
    h_s0 = F.rms_norm(torch.randn(4, 448, device=device), (448,))
    h_s1 = F.rms_norm(torch.randn(4, 448, device=device), (448,))
    h_id = F.rms_norm(torch.randn(4, 8, 64, device=device), (64,))
    ctrl = torch.zeros(4, 8, 3, device=device)
    ctrl[:, :, 1] = 1.0
    _, log_p_hi_zero, _ = build_zero_data_bpe_rank_zipf_prior(50304, device=device)
    log_p_50257 = chain_head.evaluate_chunk_50257_log_probs(h_s0, h_s1, h_id, ctrl, log_p_hi_zero)
    prob_sums = torch.exp(log_p_50257).sum(dim=-1)
    max_prob_sum_err = float((prob_sums - 1.0).abs().max().item())

    stream = LongSeqFineWebStream(device=device, batch_size=args.batch_size, seq_len=args.seq_len)
    entropy_and_ema = profile_shard0_byte_pair_entropy_and_online_ema(stream)
    idx_1, tgt_1 = stream.sample_train_batch(1)
    coverage = audit_step1_vocabulary_detector_coverage(idx_1, tgt_1)

    report = {
        "bijection_audit": {
            "raw_gpt2_vocab_size": RAW_GPT2_VOCAB_SIZE,
            "num_high_byte_buckets": NUM_HIGH_BYTE_BUCKETS,
            "num_low_byte_buckets": NUM_LOW_BYTE_BUCKETS,
            "max_b_hi": max_b_hi,
            "last_bucket_196_valid_low_bytes": bucket_196_count,
            "expected_last_bucket_valid_low_bytes": LAST_BUCKET_VALID_LOW_BYTES,
            "bijection_1to1_exact": bijection_exact,
            "chain_rule_50257_prob_sum_max_abs_err": max_prob_sum_err,
        },
        "entropy_and_zero_data_zipf_audit": entropy_and_ema,
        "step1_detector_coverage_audit": coverage,
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
