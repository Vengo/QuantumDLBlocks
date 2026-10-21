"""Unit tests for EXPERIMENT_PLAN_007: OvercompleteSeedReadoutAdapter and Input-Sensitivity Rank."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from qdlblocks import (
    OvercompleteSeedReadoutAdapter,
    QuantumBornGPT,
    compute_input_sensitivity_effective_rank,
    l2_normalize,
)
from qdlblocks_jax import overcomplete_seed_adapter_forward_jax


def test_overcomplete_seed_adapter_unit_norm_and_gradients() -> None:
    """Verify unit-norm seed outputs, frame_metrics, and finite non-zero gradients on all parameters."""
    with pytest.raises(ValueError, match="embed_dim must be >= 1"):
        OvercompleteSeedReadoutAdapter(embed_dim=0, seed_dim=16)
    with pytest.raises(ValueError, match="seed_dim must be >= 1"):
        OvercompleteSeedReadoutAdapter(embed_dim=64, seed_dim=0)
    with pytest.raises(ValueError, match="frame_expansion must be >= 1"):
        OvercompleteSeedReadoutAdapter(embed_dim=64, seed_dim=16, frame_expansion=0)

    torch.manual_seed(42)
    adapter = OvercompleteSeedReadoutAdapter(embed_dim=64, seed_dim=16)
    assert adapter.U_frame.shape == (128, 64)
    assert adapter.W_down.shape == (16, 128)
    assert adapter.seed_gate.shape == (16,)

    metrics = adapter.frame_metrics()
    assert "tight_frame_err" in metrics
    assert "rms_coherence" in metrics
    assert "welch_bound" in metrics
    assert "mean_abs_seed_gate" in metrics
    assert np.isclose(metrics["mean_abs_seed_gate"], 0.5, atol=1e-6)

    psi_final = l2_normalize(torch.randn(4, 16, 64), dim=-1).detach().requires_grad_(True)
    phi_linear = l2_normalize(torch.randn(4, 16, 16), dim=-1).detach().requires_grad_(True)

    out = adapter(psi_final, phi_linear)
    assert out.shape == (4, 16, 16)
    assert torch.allclose(out.norm(dim=-1), torch.ones_like(out[..., 0]), atol=1e-5)

    target = l2_normalize(torch.randn(4, 16, 16), dim=-1)
    loss = F.mse_loss(out, target)
    loss.backward()

    for name, tensor in [
        ("psi_final", psi_final),
        ("phi_linear", phi_linear),
        ("U_frame", adapter.U_frame),
        ("W_down", adapter.W_down),
        ("seed_gate", adapter.seed_gate),
    ]:
        assert tensor.grad is not None, f"Missing gradient for {name}"
        assert torch.all(torch.isfinite(tensor.grad)), f"Non-finite gradient for {name}"
        assert torch.any(tensor.grad != 0.0), f"Zero gradient for {name}"


def test_input_sensitivity_effective_rank_expansion() -> None:
    """Verify pullback input-sensitivity effective rank is <= 16.01 for linear collapse and >= 36.0 with adapter."""
    torch.manual_seed(42)
    embed_dim = 64
    seed_dim = 16
    w_lin = torch.empty(seed_dim, embed_dim)
    nn.init.orthogonal_(w_lin)

    adapter = OvercompleteSeedReadoutAdapter(embed_dim=embed_dim, seed_dim=seed_dim)
    psi = l2_normalize(torch.randn(256, embed_dim), dim=-1)

    def linear_collapse(x: torch.Tensor) -> torch.Tensor:
        return l2_normalize(F.linear(x, w_lin), dim=-1)

    def overcomplete_collapse(x: torch.Tensor) -> torch.Tensor:
        phi_lin = linear_collapse(x)
        return adapter(x, phi_lin)

    lin_metrics = compute_input_sensitivity_effective_rank(linear_collapse, psi)
    over_metrics = compute_input_sensitivity_effective_rank(overcomplete_collapse, psi)

    assert lin_metrics["pullback_eff_rank"] <= 16.01, (
        f"Expected linear pullback_eff_rank <= 16.01, got {lin_metrics['pullback_eff_rank']:.4f}"
    )
    assert over_metrics["pullback_eff_rank"] >= 36.0, (
        f"Expected overcomplete pullback_eff_rank >= 36.0, got {over_metrics['pullback_eff_rank']:.4f}"
    )
    assert over_metrics["pullback_participation_ratio"] > lin_metrics["pullback_participation_ratio"]
    assert over_metrics["trace_G"] > 0.0


def test_quantum_born_gpt_with_overcomplete_seed_adapter() -> None:
    """Verify QuantumBornGPT with use_overcomplete_seed_adapter=True runs forward, MTP loss, and forward_mtp_logits."""
    torch.manual_seed(42)
    model = QuantumBornGPT(
        vocab_size=128,
        embed_dim=32,
        seed_dim=8,
        num_heads=4,
        num_layers=2,
        block_size=16,
        readout_mode="tied_stiefel",
        use_overcomplete_seed_adapter=True,
        mtp_horizons=2,
    )
    assert model.seed_adapter is not None

    idx = torch.randint(0, 128, (3, 12))
    targets = torch.randint(0, 128, (3, 12))

    logits, loss = model(idx, targets)
    assert logits.shape == (3, 12, 128)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()

    for name, p in model.seed_adapter.named_parameters():
        assert p.grad is not None, f"Missing gradient on seed_adapter.{name}"
        assert torch.all(torch.isfinite(p.grad)), f"Non-finite gradient on seed_adapter.{name}"
        assert torch.any(p.grad != 0.0), f"Zero gradient on seed_adapter.{name}"

    with torch.no_grad():
        mtp_logits = model.forward_mtp_logits(idx)
    assert len(mtp_logits) == 2
    assert mtp_logits[0].shape == (3, 12, 128)
    assert mtp_logits[1].shape == (3, 12, 128)
    assert torch.allclose(mtp_logits[0], logits, atol=1e-5)


def test_pytorch_jax_seed_adapter_equivalence() -> None:
    """Verify numerical parity (< 1e-5 max_abs_diff) between PyTorch and JAX seed adapter forward passes."""
    torch.manual_seed(123)
    adapter = OvercompleteSeedReadoutAdapter(embed_dim=48, seed_dim=12, frame_expansion=2)
    psi_pt = l2_normalize(torch.randn(3, 8, 48), dim=-1)
    phi_lin_pt = l2_normalize(torch.randn(3, 8, 12), dim=-1)

    with torch.no_grad():
        out_pt = adapter(psi_pt, phi_lin_pt).cpu().numpy()

    out_jax = np.asarray(
        overcomplete_seed_adapter_forward_jax(
            psi_final=jnp.asarray(psi_pt.numpy()),
            phi_linear=jnp.asarray(phi_lin_pt.numpy()),
            U_frame=jnp.asarray(adapter.U_frame.detach().numpy()),
            W_down=jnp.asarray(adapter.W_down.detach().numpy()),
            seed_gate=jnp.asarray(adapter.seed_gate.detach().numpy()),
            eps=adapter.eps,
        )
    )
    max_abs_diff = float(np.max(np.abs(out_pt - out_jax)))
    assert max_abs_diff < 1e-5, f"Expected max_abs_diff < 1e-5, got {max_abs_diff:.6e}"
