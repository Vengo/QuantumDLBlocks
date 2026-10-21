"""Unit tests for EXPERIMENT_PLAN_011: FineWeb Multi-Shard Streaming & Memorization Diagnostics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import time

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from qdlblocks import (
    FINEWEB_SHARD_HEADER_BYTES,
    FineWebMultiShardStream,
    QuantumBornGPT,
    StiefelMuon,
    compute_empirical_bigram_coverage,
    compute_memorization_gap_metrics,
    generate_contextual_zipf_shard,
    split_muon_adamw_params,
)
from qdlblocks_jax import (
    compute_memorization_gap_metrics_jax,
    extract_shard_batch_jax,
)


def test_contextual_zipf_shards_shared_distribution_distinct_realizations() -> None:
    """Verify generate_contextual_zipf_shard produces distinct realizations and >= 2.5x bigram coverage across 9 shards."""
    with pytest.raises(ValueError, match="num_tokens must be >= 1"):
        generate_contextual_zipf_shard(0, vocab_size=512)
    with pytest.raises(ValueError, match="vocab_size must be >= 2"):
        generate_contextual_zipf_shard(64, vocab_size=1)

    shards = [
        generate_contextual_zipf_shard(16385, vocab_size=512, shard_idx=i, base_seed=42)
        for i in range(1, 10)
    ]
    for i in range(len(shards)):
        assert shards[i].dtype == np.uint16
        assert shards[i].shape == (16385,)
        for j in range(i + 1, len(shards)):
            assert not np.array_equal(shards[i], shards[j]), f"Shards {i+1} and {j+1} should be distinct"

    # Reproducibility check
    s1_again = generate_contextual_zipf_shard(16385, vocab_size=512, shard_idx=1, base_seed=42)
    assert np.array_equal(shards[0], s1_again)

    cov_1x9 = compute_empirical_bigram_coverage([shards[0]] * 9, max_vocab=512)
    cov_9x1 = compute_empirical_bigram_coverage(shards, max_vocab=512)

    ratio = cov_9x1["unique_bigrams"] / max(1.0, cov_1x9["unique_bigrams"])
    assert ratio >= 2.5, f"Expected 9-shard bigram coverage >= 2.5x single shard, got {ratio:.2f}x"


def test_fineweb_multishard_stream_fallback_and_async_prefetch() -> None:
    """Verify train_full_100M.bin fallback for shard 1 and < 25ms async prefetch across 9 shards."""
    with pytest.raises(ValueError, match="num_shards must be in"):
        FineWebMultiShardStream(data_dir="/tmp", num_shards=0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        legacy_train = tmp_path / "train_full_100M.bin"
        val_file = tmp_path / "val_slice.bin"

        header = np.zeros(FINEWEB_SHARD_HEADER_BYTES // 4, dtype=np.int32)
        mock_train_tokens = generate_contextual_zipf_shard(4097, vocab_size=512, shard_idx=1, base_seed=42)
        mock_val_tokens = generate_contextual_zipf_shard(2049, vocab_size=512, shard_idx=0, base_seed=42)

        with open(legacy_train, "wb") as f:
            f.write(header.tobytes())
            f.write(mock_train_tokens.tobytes())
        with open(val_file, "wb") as f:
            f.write(header.tobytes())
            f.write(mock_val_tokens.tobytes())

        disk_stream = FineWebMultiShardStream(data_dir=tmp_path, num_shards=2, prefetch_async=False)
        assert disk_stream.shard_path(1) == legacy_train
        assert disk_stream.shard_path(2) == tmp_path / "fineweb_train_000002.bin"

        loaded_val = disk_stream.ensure_val_shard()
        assert np.array_equal(loaded_val, mock_val_tokens)
        loaded_s1, _ = disk_stream.get_shard(1)
        assert np.array_equal(np.asarray(loaded_s1), mock_train_tokens)
        disk_stream.close()

        # Verify async prefetching on 9 synthetic shards
        async_stream = FineWebMultiShardStream(
            data_dir=tmp_path,
            num_shards=9,
            prefetch_async=True,
            synthetic_tokens_per_shard=8193,
            vocab_size=512,
            seed=42,
        )
        async_stream.prefetch_shard(1)
        time.sleep(0.015)

        wait_times: list[float] = []
        for pass_idx, s_idx, arr, wait_ms in async_stream.iter_shards(num_passes=1):
            assert pass_idx == 0
            expected = generate_contextual_zipf_shard(8193, vocab_size=512, shard_idx=s_idx, base_seed=42)
            assert np.array_equal(arr, expected)
            wait_times.append(wait_ms)
            time.sleep(0.008)
        async_stream.close()

        assert len(wait_times) == 9
        assert max(wait_times) < 25.0, f"Expected wait_ms < 25.0, got {max(wait_times):.3f} ms"


def test_memorization_gap_metrics_torch_and_jax_parity() -> None:
    """Verify compute_memorization_gap_metrics and compute_memorization_gap_metrics_jax agree to < 1e-6."""
    train_ce = [5.40, 4.65, 4.25, 3.92]
    val_ce = [4.95, 4.58, 4.31, 4.18]
    pre_ce = [4.65, 4.25]
    post_ce = [4.47, 4.08]

    pt_m = compute_memorization_gap_metrics(
        train_ce_at_evals=train_ce,
        val_ce_at_evals=val_ce,
        boundary_pre_train_ce=pre_ce,
        boundary_post_train_ce=post_ce,
    )
    jax_m = compute_memorization_gap_metrics_jax(
        train_ce_arr=jnp.asarray(train_ce, dtype=jnp.float32),
        val_ce_arr=jnp.asarray(val_ce, dtype=jnp.float32),
        boundary_pre_arr=jnp.asarray(pre_ce, dtype=jnp.float32),
        boundary_post_arr=jnp.asarray(post_ce, dtype=jnp.float32),
    )

    for k in ("final_train_ce", "final_val_ce", "final_memorization_gap", "max_memorization_gap", "mean_boundary_drop"):
        diff = abs(float(pt_m[k]) - float(jax_m[k]))
        assert diff < 1e-6, f"Mismatch on {k}: {pt_m[k]} vs {float(jax_m[k])}"

    shard = generate_contextual_zipf_shard(1025, vocab_size=512, shard_idx=1, base_seed=42)
    xj, yj = extract_shard_batch_jax(shard, chunk_idx=0, batch_size=4, seq_len=32)
    assert xj.shape == (4, 32)
    assert yj.shape == (4, 32)
    assert np.array_equal(np.asarray(xj[:, 1:]), np.asarray(yj[:, :-1]))


def test_h1_multishard_streaming_eliminates_memorization_drag() -> None:
    """Verify 9x1 multi-shard streaming reduces memorization gap >= 3x and lowers Val CE vs 1x9 repetition."""
    vocab_size = 256
    n_tokens = 4097  # 8 batches of (B=8, T=64 = 512 tok/step) + 1
    B, T = 8, 64
    tok_per_step = B * T
    steps_per_shard = 8

    val_shard = generate_contextual_zipf_shard(n_tokens, vocab_size=vocab_size, shard_idx=0, base_seed=42)
    shards_9 = [
        generate_contextual_zipf_shard(n_tokens, vocab_size=vocab_size, shard_idx=i, base_seed=42)
        for i in range(1, 10)
    ]

    def _to_xy(sh: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for i in range(steps_per_shard):
            buf = sh[i * tok_per_step : (i + 1) * tok_per_step + 1].astype(np.int64)
            xs.append(buf[:-1].reshape(B, T))
            ys.append(buf[1:].reshape(B, T))
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))

    xv, yv = _to_xy(val_shard)
    batches_9x1 = [_to_xy(s) for s in shards_9]
    batches_1x9 = [batches_9x1[0]] * 9

    torch.manual_seed(42)
    ref_model = QuantumBornGPT(
        vocab_size=vocab_size,
        embed_dim=48,
        seed_dim=16,
        num_heads=4,
        num_layers=2,
        hidden_dim=64,
        block_size=T,
        readout_mode="tied_stiefel",
        use_two_beam_mlp=True,
        use_overcomplete_seed_adapter=True,
    )
    init_state = {k: v.detach().clone() for k, v in ref_model.state_dict().items()}

    def _run_regime(regime_batches: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, float]:
        torch.manual_seed(42)
        m = QuantumBornGPT(
            vocab_size=vocab_size,
            embed_dim=48,
            seed_dim=16,
            num_heads=4,
            num_layers=2,
            hidden_dim=64,
            block_size=T,
            readout_mode="tied_stiefel",
            use_two_beam_mlp=True,
            use_overcomplete_seed_adapter=True,
        )
        with torch.no_grad():
            for k, v in m.state_dict().items():
                v.copy_(init_state[k])
        muon_p, adam_p = split_muon_adamw_params(
            m, adamw_name_patterns=("wte", "povm_head", "embed", "inv_temp", "seed_adapter")
        )
        opt_m = StiefelMuon(muon_p, lr=0.025, momentum=0.95, ns_steps=5, retract_every=4)
        opt_a = torch.optim.AdamW(adam_p, lr=0.006, weight_decay=1e-4)

        for x_sh, y_sh in regime_batches:
            m.train()
            for s in range(steps_per_shard):
                opt_m.zero_grad(set_to_none=True)
                opt_a.zero_grad(set_to_none=True)
                _, loss = m(x_sh[s], y_sh[s])
                assert loss is not None
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt_m.step()
                opt_a.step()

        m.eval()
        with torch.no_grad():
            last_x, last_y = regime_batches[-1]
            tr_losses = [float(m(last_x[i], last_y[i])[1].item()) for i in range(steps_per_shard)]  # type: ignore[index]
            va_losses = [float(m(xv[i], yv[i])[1].item()) for i in range(steps_per_shard)]  # type: ignore[index]
        return compute_memorization_gap_metrics([float(np.mean(tr_losses))], [float(np.mean(va_losses))])

    res_1x9 = _run_regime(batches_1x9)
    res_9x1 = _run_regime(batches_9x1)

    assert res_9x1["final_memorization_gap"] <= res_1x9["final_memorization_gap"] / 3.0
    assert res_9x1["final_val_ce"] < res_1x9["final_val_ce"]


def test_examples_11_cli_accepts_num_shards_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify examples/11_fineweb_100m_multiepoch_gpu_tpu.py parses --num-shards 9 --epochs 1 --eval-every-tokens-m 50.0."""
    script_path = Path(__file__).resolve().parents[1] / "examples" / "11_fineweb_100m_multiepoch_gpu_tpu.py"
    spec = importlib.util.spec_from_file_location("ex11_multishard", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--num-shards",
            "9",
            "--epochs",
            "1",
            "--eval-every-tokens-m",
            "50.0",
            "--seed-dim",
            "384",
            "--overcomplete-seed-adapter",
        ],
    )
    args = mod.parse_args()
    assert args.num_shards == 9
    assert args.epochs == 1
    assert np.isclose(args.eval_every_tokens_m, 50.0)
    assert args.seed_dim == 384
    assert args.overcomplete_seed_adapter is True
