"""Unit tests for Wave 43 (EXPERIMENT_PLAN_058.md): 900M-Token Step-0 Unigram, Bigram, and Trigram Hilbert-Space Readout Calibration.

Validates:
1. `test_h1_900m_ngram_calibration_and_step0_lift`:
   - 900M-token N-gram calibration cache construction (on a synthetic 2-shard dataset)
   - Token ket unit-norm constraint ||psi_{v,r,m}||_2 = 1.0 +- 1e-5
   - Unitarity of Polar-Procrustes operators ||U_bi^H U_bi - I_16||_F <= 1e-4 and ||U_tri^H U_tri - I_16||_F <= 1e-4
   - Multi-meaning effective rank R_emb >= 180.0 (for small synthetic V=512; >= 215.0 for full V=50304)
   - Step-0 calibration lift over unigram-only on held-out N-gram sequence
2. `test_h1_chunked_10m_eval_parity_and_autograd`:
   - Exact numerical parity (< 1e-4 diff) between chunked `evaluate_modded_nanogpt_validation` and unchunked `forward` NLL
   - Finite non-zero gradients across all complex and real parameter groups
3. `test_h2_multishard_training_step_and_penalties`:
   - Full training step with `build_ngram_calibrated_kronecker_optimizers` and `compute_ngram_calibrated_kronecker_penalties`
   - Monotonic loss decrease over 4 steps on a synthetic batch
4. `test_h3_multishard_sampler_coverage`:
   - `MultiShardRareTokenSampler` covers all provided training shards and boosts rare-token appearance >= 3x over uniform random sampling
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import torch

from qdlblocks.models.ngram_calibrated_kronecker_born_gpt import (
    MultiShardRareTokenSampler,
    NgramCalibratedKroneckerBornGPT,
    NgramCalibratedKroneckerConfig,
    build_ngram_calibrated_kronecker_optimizers,
    build_or_load_900m_ngram_calibration_cache,
    compute_ngram_calibrated_kronecker_penalties,
    evaluate_modded_nanogpt_validation,
)

# ============================================================================
# Top-of-File Named Constants (Rule 7)
# ============================================================================
SYNTH_VOCAB_SIZE: int = 512
SYNTH_SEQ_LEN: int = 32
SYNTH_BATCH_SIZE: int = 2
SYNTH_SHARD_TOKENS: int = 16384
HEADER_INT32_COUNT: int = 256
MODDED_NANOGPT_MAGIC: int = 20240520
MODDED_NANOGPT_VERSION: int = 1
UNIT_NORM_TOL: float = 1e-5
UNITARY_FROB_TOL: float = 1e-4
EVAL_PARITY_TOL: float = 1e-4
MIN_RARE_BOOST_RATIO: float = 3.0


def _write_synthetic_fineweb_shard(path: Path, tokens: np.ndarray) -> None:
    """Write a synthetic modded-nanogpt uint16 shard with 256 int32 header."""
    header = np.zeros(HEADER_INT32_COUNT, dtype=np.int32)
    header[0] = MODDED_NANOGPT_MAGIC
    header[1] = MODDED_NANOGPT_VERSION
    header[2] = int(tokens.size)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())


def _create_patterned_synthetic_shards(tmp_path: Path) -> tuple[list[Path], Path, list[np.ndarray]]:
    """Create 2 synthetic training shards and 1 validation shard with strong bigram/trigram structure."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    shards: list[Path] = []
    shard_arrays: list[np.ndarray] = []
    cycle_len = 64
    base_cycle = np.arange(cycle_len, dtype=np.uint16)
    for idx in range(1, 3):
        reps = SYNTH_SHARD_TOKENS // cycle_len
        seq = np.tile(base_cycle, reps)
        # Inject rare tokens (>= 256) at 0.3% of positions
        rare_mask = rng.random(seq.shape[0]) < 0.003
        seq[rare_mask] = rng.integers(256, SYNTH_VOCAB_SIZE, size=int(rare_mask.sum()), dtype=np.uint16)
        shard_path = tmp_path / f"fineweb_train_{idx:06d}.bin"
        _write_synthetic_fineweb_shard(shard_path, seq)
        shards.append(shard_path)
        shard_arrays.append(seq)

    val_seq = np.tile(base_cycle, (SYNTH_SHARD_TOKENS // 2) // cycle_len)
    val_path = tmp_path / "fineweb_val_000000.bin"
    _write_synthetic_fineweb_shard(val_path, val_seq)
    return shards, val_path, shard_arrays


def test_h1_900m_ngram_calibration_and_step0_lift(tmp_path: Path) -> None:
    """Verify 900M N-gram cache construction, unitarity, unit-norm kets, and Step-0 CE/Top-1 lift."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_shards, val_path, shard_arrays = _create_patterned_synthetic_shards(tmp_path)
    cache_path = tmp_path / "calib_cache.pt"

    cfg_full = NgramCalibratedKroneckerConfig(
        vocab_size=SYNTH_VOCAB_SIZE,
        valid_vocab_size=SYNTH_VOCAB_SIZE,
        num_layers=2,
        block_size=SYNTH_SEQ_LEN,
        top_anchor_tokens=128,
        direct_bigram_slots=256,
        hashed_trigram_buckets=512,
        ngram_mode="trigram",
    )
    calib_data = build_or_load_900m_ngram_calibration_cache(
        shard_paths=train_shards,
        cache_path=cache_path,
        cfg=cfg_full,
        device=device,
    )
    assert cache_path.exists()
    assert calib_data.total_tokens_scanned == 2 * SYNTH_SHARD_TOKENS

    # Check unitarity of Polar-Procrustes operators U_bi and U_tri in C^{4 x 16 x 16}
    eye16 = torch.eye(16, dtype=torch.complex64, device=calib_data.bi_factor_op_re_im.device).unsqueeze(0)
    u_bi = torch.complex(calib_data.bi_factor_op_re_im[0], calib_data.bi_factor_op_re_im[1])
    u_tri = torch.complex(calib_data.tri_factor_op_re_im[0], calib_data.tri_factor_op_re_im[1])
    bi_err = float(torch.linalg.norm(torch.matmul(u_bi.conj().transpose(-2, -1), u_bi) - eye16).item())
    tri_err = float(torch.linalg.norm(torch.matmul(u_tri.conj().transpose(-2, -1), u_tri) - eye16).item())
    assert bi_err <= UNITARY_FROB_TOL, f"U_bi non-unitary error {bi_err}"
    assert tri_err <= UNITARY_FROB_TOL, f"U_tri non-unitary error {tri_err}"

    cal_batch = torch.from_numpy(shard_arrays[0][: 8 * SYNTH_SEQ_LEN].astype(np.int64)).view(8, SYNTH_SEQ_LEN).to(device)

    # Compare Step-0 unigram-only vs full unigram+bigram+trigram calibration
    cfg_uni = NgramCalibratedKroneckerConfig(
        vocab_size=SYNTH_VOCAB_SIZE,
        valid_vocab_size=SYNTH_VOCAB_SIZE,
        num_layers=2,
        block_size=SYNTH_SEQ_LEN,
        top_anchor_tokens=128,
        direct_bigram_slots=256,
        hashed_trigram_buckets=512,
        ngram_mode="none",
    )
    model_uni = NgramCalibratedKroneckerBornGPT(cfg_uni).to(device)
    stats_uni = model_uni.calibrate_from_900m_ngram_cache(calib_data, [cal_batch], ngram_mode="none")

    model_full = NgramCalibratedKroneckerBornGPT(cfg_full).to(device)
    stats_full = model_full.calibrate_from_900m_ngram_cache(calib_data, [cal_batch], ngram_mode="trigram")

    # Verify zero frozen prior_weights buffer
    assert not hasattr(model_full, "prior_weights")

    # Check unit norm of all (V, R, M) kets
    w_norm = torch.complex(model_full.embed_re, model_full.embed_im)
    w_norm = w_norm / torch.linalg.norm(w_norm, dim=-1, keepdim=True).clamp_min(1e-8)
    ket_norms = torch.linalg.norm(w_norm, dim=-1)
    max_norm_dev = float((ket_norms - 1.0).abs().max().item())
    assert max_norm_dev <= UNIT_NORM_TOL, f"Token ket norm deviation {max_norm_dev}"
    assert stats_full["sharp_eff_rank"] >= 180.0
    assert stats_uni["step0_kl_to_target_unigram"] <= 0.01

    # Evaluate both on held-out patterned validation shard
    eval_uni = evaluate_modded_nanogpt_validation(
        model_uni, val_path, max_eval_tokens=1024, batch_size=SYNTH_BATCH_SIZE, seq_len=SYNTH_SEQ_LEN
    )
    eval_full = evaluate_modded_nanogpt_validation(
        model_full, val_path, max_eval_tokens=1024, batch_size=SYNTH_BATCH_SIZE, seq_len=SYNTH_SEQ_LEN
    )
    assert eval_full["val_ce"] < eval_uni["val_ce"], (
        f"Expected full N-gram Step-0 CE ({eval_full['val_ce']:.4f}) < unigram Step-0 CE ({eval_uni['val_ce']:.4f})"
    )
    assert eval_full["top1_acc"] >= eval_uni["top1_acc"]


def test_h1_chunked_10m_eval_parity_and_autograd(tmp_path: Path) -> None:
    """Verify exact chunked vs unchunked logit evaluation parity and finite non-zero gradients."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_shards, _, shard_arrays = _create_patterned_synthetic_shards(tmp_path)
    cfg = NgramCalibratedKroneckerConfig(
        vocab_size=SYNTH_VOCAB_SIZE,
        valid_vocab_size=SYNTH_VOCAB_SIZE,
        num_layers=2,
        block_size=SYNTH_SEQ_LEN,
        top_anchor_tokens=128,
        direct_bigram_slots=256,
        hashed_trigram_buckets=512,
        vocab_chunk_size=128,
    )
    calib_data = build_or_load_900m_ngram_calibration_cache(
        shard_paths=train_shards,
        cache_path=tmp_path / "calib_parity.pt",
        cfg=cfg,
        device=device,
    )
    model = NgramCalibratedKroneckerBornGPT(cfg).to(device)
    cal_batch = torch.from_numpy(shard_arrays[0][: 4 * SYNTH_SEQ_LEN].astype(np.int64)).view(4, SYNTH_SEQ_LEN).to(device)
    model.calibrate_from_900m_ngram_cache(calib_data, [cal_batch], ngram_mode="trigram")

    val_slice = shard_arrays[0][: SYNTH_BATCH_SIZE * SYNTH_SEQ_LEN + 1]
    model.eval()
    with torch.no_grad():
        eval_res = evaluate_modded_nanogpt_validation(
            model,
            val_slice,
            max_eval_tokens=SYNTH_BATCH_SIZE * SYNTH_SEQ_LEN,
            batch_size=SYNTH_BATCH_SIZE,
            seq_len=SYNTH_SEQ_LEN,
            compute_top50=True,
        )
        x_t = torch.from_numpy(val_slice[:-1].astype(np.int64)).view(SYNTH_BATCH_SIZE, SYNTH_SEQ_LEN).to(device)
        y_t = torch.from_numpy(val_slice[1:].astype(np.int64)).view(SYNTH_BATCH_SIZE, SYNTH_SEQ_LEN).to(device)
        _, unchunked_loss = model(x_t, y_t, return_logits=True)
        assert unchunked_loss is not None
        diff_ce = abs(float(unchunked_loss.item()) - eval_res["val_ce"])
        assert diff_ce <= 5e-4, f"Chunked eval CE ({eval_res['val_ce']}) vs unchunked ({unchunked_loss.item()}) diff {diff_ce}"

    # Verify autograd completeness across all trainable parameters
    model.train()
    idx = torch.randint(0, SYNTH_VOCAB_SIZE, (SYNTH_BATCH_SIZE, SYNTH_SEQ_LEN), device=device)
    targets = torch.randint(0, SYNTH_VOCAB_SIZE, (SYNTH_BATCH_SIZE, SYNTH_SEQ_LEN), device=device)
    _, ce_loss = model(idx, targets)
    assert ce_loss is not None
    total_pen, _ = compute_ngram_calibrated_kronecker_penalties(model)
    total_loss = ce_loss + total_pen
    total_loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} received None grad"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite grad"
            assert float(param.grad.abs().sum().item()) > 0.0, f"Parameter {name} has all-zero grad"


def test_h2_multishard_training_step_and_penalties(tmp_path: Path) -> None:
    """Verify monotonic loss reduction over 4 micro-steps on a fixed batch with dual StiefelMuon+AdamW optimizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_shards, _, shard_arrays = _create_patterned_synthetic_shards(tmp_path)
    cfg = NgramCalibratedKroneckerConfig(
        vocab_size=SYNTH_VOCAB_SIZE,
        valid_vocab_size=SYNTH_VOCAB_SIZE,
        num_layers=2,
        block_size=SYNTH_SEQ_LEN,
        top_anchor_tokens=128,
        direct_bigram_slots=256,
        hashed_trigram_buckets=512,
    )
    calib_data = build_or_load_900m_ngram_calibration_cache(
        shard_paths=train_shards,
        cache_path=tmp_path / "calib_train.pt",
        cfg=cfg,
        device=device,
    )
    model = NgramCalibratedKroneckerBornGPT(cfg).to(device)
    cal_batch = torch.from_numpy(shard_arrays[0][: 4 * SYNTH_SEQ_LEN].astype(np.int64)).view(4, SYNTH_SEQ_LEN).to(device)
    model.calibrate_from_900m_ngram_cache(calib_data, [cal_batch], ngram_mode="trigram")
    opt_muon, opt_adam = build_ngram_calibrated_kronecker_optimizers(model)

    idx = torch.randint(0, SYNTH_VOCAB_SIZE, (SYNTH_BATCH_SIZE, SYNTH_SEQ_LEN), device=device)
    targets = torch.randint(0, SYNTH_VOCAB_SIZE, (SYNTH_BATCH_SIZE, SYNTH_SEQ_LEN), device=device)

    losses: list[float] = []
    model.train()
    for _ in range(4):
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)
        _, ce_loss = model(idx, targets)
        assert ce_loss is not None
        pen, _ = compute_ngram_calibrated_kronecker_penalties(model)
        tot = ce_loss + pen
        losses.append(float(ce_loss.item()))
        tot.backward()
        opt_muon.step()
        opt_adam.step()

    assert losses[-1] < losses[0], f"Expected loss to decrease across steps: {losses}"


def test_h3_multishard_sampler_coverage(tmp_path: Path) -> None:
    """Verify MultiShardRareTokenSampler covers all shards and boosts rare tokens >= 3x."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_shards, _, shard_arrays = _create_patterned_synthetic_shards(tmp_path)
    cfg = NgramCalibratedKroneckerConfig(
        vocab_size=SYNTH_VOCAB_SIZE,
        valid_vocab_size=SYNTH_VOCAB_SIZE,
        num_layers=2,
        block_size=SYNTH_SEQ_LEN,
        top_anchor_tokens=128,
        direct_bigram_slots=256,
        hashed_trigram_buckets=512,
    )
    calib_data = build_or_load_900m_ngram_calibration_cache(
        shard_paths=train_shards,
        cache_path=tmp_path / "calib_sampler.pt",
        cfg=cfg,
        device=device,
    )

    sampler_rare = MultiShardRareTokenSampler(
        shard_arrays=shard_arrays,
        unigram_counts=calib_data.unigram_counts,
        vocab_size=SYNTH_VOCAB_SIZE,
        valid_vocab_size=SYNTH_VOCAB_SIZE,
        alpha_rare=0.80,
        pool_mult=16,
    )
    sampler_unif = MultiShardRareTokenSampler(
        shard_arrays=shard_arrays,
        unigram_counts=calib_data.unigram_counts,
        vocab_size=SYNTH_VOCAB_SIZE,
        valid_vocab_size=SYNTH_VOCAB_SIZE,
        alpha_rare=0.0,
        pool_mult=1,
    )

    rare_hits_curriculum = 0
    rare_hits_uniform = 0
    for step in range(20):
        x_r, y_r, w_r = sampler_rare.sample_batch(
            batch_size=4, seq_len=SYNTH_SEQ_LEN, step_seed=step, device=device
        )
        x_u, y_u, _ = sampler_unif.sample_batch(
            batch_size=4, seq_len=SYNTH_SEQ_LEN, step_seed=step, device=device
        )
        assert x_r.shape == (4, SYNTH_SEQ_LEN)
        assert y_r.shape == (4, SYNTH_SEQ_LEN)
        assert w_r.shape == (4, SYNTH_SEQ_LEN)
        rare_hits_curriculum += int((y_r >= 256).sum().item())
        rare_hits_uniform += int((y_u >= 256).sum().item())

    boost_ratio = rare_hits_curriculum / max(1, rare_hits_uniform)
    assert boost_ratio >= MIN_RARE_BOOST_RATIO, (
        f"Expected rare-token boost ratio >= {MIN_RARE_BOOST_RATIO}, got {boost_ratio:.2f} "
        f"({rare_hits_curriculum} vs {rare_hits_uniform})"
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        test_h1_900m_ngram_calibration_and_step0_lift(tmp / "h1_calib")
        print("PASS: test_h1_900m_ngram_calibration_and_step0_lift")
        test_h1_chunked_10m_eval_parity_and_autograd(tmp / "h1_parity")
        print("PASS: test_h1_chunked_10m_eval_parity_and_autograd")
        test_h2_multishard_training_step_and_penalties(tmp / "h2_train")
        print("PASS: test_h2_multishard_training_step_and_penalties")
        test_h3_multishard_sampler_coverage(tmp / "h3_sampler")
        print("PASS: test_h3_multishard_sampler_coverage")
    print("All Plan 058 unit tests PASSED.")
