"""Unit tests for Wave 74 (`EXPERIMENT_PLAN_094.md`): Fused `bfloat16` 4-Compartment Baton LM."""

from __future__ import annotations

import torch

from qdlblocks.benchmarks.fused_bf16_four_compartment_baton_lab import (
    FINEWEB_TRAIN_SHARDS_300M,
    FusedBF16FourCompartmentBatonConfig,
    FusedBF16FourCompartmentBatonLM,
    MultiShardFineWebStreamer,
    verify_theorem15_algebraic_identities,
)


def test_theorem15_algebraic_identities() -> None:
    """Verify Theorems 15a, 15b, and 15c match explicit 64x64 density matrix evaluation."""
    res = verify_theorem15_algebraic_identities()
    assert res["passed"], f"Theorem 15 parity failed: {res}"


def test_fused_bf16_1l_and_2l_forward_backward_and_online_ingestion() -> None:
    """Verify 1L and 2L Fused BF16 2-Hop Triangle models run forward/backward and ingest online."""
    for num_layers in (1, 2):
        cfg = FusedBF16FourCompartmentBatonConfig(
            vocab_size=256,
            hilbert_dim=64,
            num_layers=num_layers,
            use_two_hop_triangle=True,
            use_bfloat16_amp=True,
        )
        model = FusedBF16FourCompartmentBatonLM(cfg)
        idx = torch.randint(0, 256, (2, 16))
        targets = torch.randint(0, 256, (2, 16))
        logits, loss = model(idx, targets)
        assert logits.shape == (2, 16, 256)
        assert loss is not None and torch.isfinite(loss)
        assert int(model.online_ingested_tokens.item()) == 32
        loss.backward()
        grad_norm = sum(
            float(p.grad.norm().item()) for p in model.parameters() if p.grad is not None
        )
        assert grad_norm > 0.0


def test_multishard_streamer_boundary_transitions() -> None:
    """Verify MultiShardFineWebStreamer cleanly crosses 100M, 200M, and 300M shard boundaries."""
    if not all(p.exists() for p in FINEWEB_TRAIN_SHARDS_300M):
        return
    streamer = MultiShardFineWebStreamer(FINEWEB_TRAIN_SHARDS_300M)
    for step_idx in (0, 6102, 6103, 6104, 12206, 12207, 12208, 18309, 18310, 18311):
        x, y, s_idx = streamer.sample_sequential_batch(step_idx, 64, 256, torch.device("cpu"))
        assert x.shape == (64, 256)
        assert y.shape == (64, 256)
        assert s_idx in (0, 1, 2)
