"""End-to-end model forward/backward and generation tests."""

from __future__ import annotations

import torch

from qdlblocks import (
    DeepPureQuantumNetwork,
    HybridQuantumClassifier,
    QuantumBornGPT,
    is_valid_density_matrix,
)


def test_deep_pure_quantum_network() -> None:
    torch.manual_seed(30)
    model = DeepPureQuantumNetwork(
        in_features=16,
        hilbert_dim=8,
        num_classes=4,
        num_heads=2,
        num_kraus=3,
        num_kraus_layers=2,
        purification_placement="post_fusion",
        purify_after_kraus=True,
        kraus_residual=True,
    )
    x = torch.randn(5, 16)
    logits, rho, trajectory = model(x, return_trajectory=True)
    assert logits.shape == (5, 4)
    assert rho.shape == (5, 8, 8)
    assert len(trajectory) == 5  # 1 attention + 2 * (Kraus + Purify)
    for state in trajectory:
        assert is_valid_density_matrix(state)[0]

    loss = logits.sum()
    loss.backward()


def test_hybrid_quantum_classifier() -> None:
    torch.manual_seed(31)
    model = HybridQuantumClassifier(
        in_channels=1,
        token_dim=16,
        hilbert_dim=8,
        num_heads=2,
        num_classes=6,
        grid_size=(2, 2),
        pooling_mode="incoherent",
        purification_placement="post_fusion",
        num_kraus=2,
    )
    img = torch.randn(4, 1, 8, 8)
    logits, rho = model(img)
    assert logits.shape == (4, 6)
    assert rho.shape == (4, 8, 8)
    assert is_valid_density_matrix(rho)[0]
    logits.sum().backward()


def test_quantum_born_gpt_modes_and_generation() -> None:
    torch.manual_seed(32)
    for readout_mode in ("asymmetric", "symmetric_dual"):
        model = QuantumBornGPT(
            vocab_size=50,
            embed_dim=32,
            seed_dim=16,
            num_heads=4,
            num_layers=2,
            hidden_dim=64,
            block_size=16,
            readout_mode=readout_mode,  # type: ignore[arg-type]
            learnable_temperature=True,
            value_residual_mixing=True,
            input_highway_mixing=True,
        )
        idx = torch.randint(0, 50, (3, 10))
        targets = torch.randint(0, 50, (3, 10))
        logits, loss = model(idx, targets)
        assert logits.shape == (3, 10, 50)
        assert loss is not None
        loss.backward()

        generated = model.generate(idx[:, :4], max_new_tokens=5, top_k=10)
        assert generated.shape == (3, 9)


def test_quantum_born_gpt_mtp_and_dynamic_rope_extension() -> None:
    torch.manual_seed(33)
    model = QuantumBornGPT(
        vocab_size=48,
        embed_dim=32,
        seed_dim=16,
        num_heads=4,
        num_layers=2,
        hidden_dim=64,
        block_size=8,  # Start with small block_size=8
        readout_mode="tied_stiefel",
        learnable_temperature=True,
        mtp_horizons=3,
        mtp_loss_weight=0.25,
    )
    assert len(model.mtp_rotations) == 2

    # Pass a sequence of length 16 (> initial block_size=8) to test dynamic UnitaryRoPE table extension
    idx = torch.randint(0, 48, (2, 16))
    targets = torch.randint(0, 48, (2, 16))
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 16, 48)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()

    for rot in model.mtp_rotations:
        assert rot.grad is not None
        assert torch.any(rot.grad != 0.0)

    # Verify forward_mtp_logits returns [t+1, t+2, t+3] horizon tensors
    horizon_logits = model.forward_mtp_logits(idx)
    assert len(horizon_logits) == 3
    for h_log in horizon_logits:
        assert h_log.shape == (2, 16, 48)

