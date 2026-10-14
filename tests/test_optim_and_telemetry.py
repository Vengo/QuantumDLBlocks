"""Unit tests for StiefelMuon optimizer and Quantum-Geometric Telemetry Probe."""

from __future__ import annotations

import torch
import torch.nn as nn

from qdlblocks import (
    QuantumBornGPT,
    StiefelMuon,
    inspect_quantum_model,
    polar_snap_weights,
    split_muon_adamw_params,
    stiefel_isometry_error,
)


def test_stiefel_muon_riemannian_retraction_reduces_isometry_error() -> None:
    torch.manual_seed(20)
    # Create a badly scaled non-orthogonal matrix (singular values ~ 3.0)
    W = nn.Parameter(torch.randn(32, 32) * 0.6)
    err_initial = stiefel_isometry_error(W)
    assert err_initial > 1.0

    opt = StiefelMuon(
        [W],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0,
        decay_mode="none",
        stiefel_penalty=0.08,
    )

    for _ in range(60):
        opt.zero_grad()
        # Small dummy gradient
        W.grad = torch.randn_like(W) * 0.01
        opt.step()

    err_final = stiefel_isometry_error(W)
    assert err_final < err_initial * 0.05
    assert err_final < 0.15


def test_stiefel_muon_cautious_and_smooth_modes() -> None:
    torch.manual_seed(21)
    for mode in ("smooth", "cautious"):
        W = nn.Parameter(torch.randn(24, 16) * 0.2)
        opt = StiefelMuon(
            [W],
            lr=0.01,
            weight_decay=0.05,
            decay_mode=mode,  # type: ignore[arg-type]
            aspect_scale=True,
        )
        opt.zero_grad()
        W.grad = torch.randn_like(W)
        w_before = W.detach().clone()
        opt.step()
        assert not torch.allclose(W, w_before)


def test_inspect_quantum_model_and_param_splitting() -> None:
    torch.manual_seed(22)
    model = QuantumBornGPT(
        vocab_size=64,
        embed_dim=32,
        seed_dim=16,
        num_heads=4,
        num_layers=2,
        hidden_dim=64,
        block_size=16,
    )

    muon_params, adamw_params = split_muon_adamw_params(model)
    assert len(muon_params) > 0
    assert len(adamw_params) > 0
    assert all(p.ndim == 2 for p in muon_params)

    x = torch.randint(0, 64, (2, 12))
    y = torch.randint(0, 64, (2, 12))

    report = inspect_quantum_model(model, x, y, print_report=False)
    assert "mean_isometry_error" in report
    assert "layer_effective_ranks" in report
    assert "Layer_00" in report["layer_effective_ranks"]
    assert "logit_stats" in report
    assert report["logit_stats"]["eval_cross_entropy"] > 0.0

    # Test polar_snap_weights
    polar_snap_weights(model)
