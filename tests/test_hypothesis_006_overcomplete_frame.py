"""Unit tests for EXPERIMENT_PLAN_006: Overcomplete Tight Frames and Welch-Bound Coherence."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from qdlblocks import (
    OvercompleteFrameInterference,
    frame_coherence_metrics,
    l2_normalize,
    optimize_overcomplete_tight_frame,
    tight_frame_penalty,
    welch_bound_coherence,
)
from qdlblocks_jax import (
    frame_coherence_metrics_jax,
    tight_frame_penalty_jax,
    welch_bound_coherence_jax,
)


def _effective_feature_covariance_rank(f_mat: torch.Tensor, eps: float = 1e-12) -> float:
    f_centered = f_mat.float() - f_mat.float().mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(f_centered)
    p = s / torch.clamp(torch.sum(s), min=eps)
    p_pos = torch.clamp(p, min=eps)
    ent = -torch.sum(p * torch.log(p_pos))
    return float(torch.exp(ent).item())


def test_welch_bound_and_tight_frame_optimization() -> None:
    """Verify Welch bound edge cases, no_grad() safety, tight_frame_err, IsoErr, and RMS coherence."""
    with pytest.raises(ValueError, match="dim must be >= 1"):
        welch_bound_coherence(8, 0)
    with pytest.raises(ValueError, match="dim must be >= 1"):
        welch_bound_coherence_jax(8, 0)
    with pytest.raises(ValueError, match="Require num_vectors >= dim >= 1"):
        optimize_overcomplete_tight_frame(16, 32)

    assert welch_bound_coherence(32, 32) == 0.0
    assert welch_bound_coherence(16, 32) == 0.0
    assert welch_bound_coherence_jax(32, 32) == 0.0

    # Verify M == d orthogonal case
    with torch.no_grad():
        w_ortho = optimize_overcomplete_tight_frame(32, 32, steps=1, seed=7)
    m_ortho = frame_coherence_metrics(w_ortho)
    assert m_ortho["redundancy_ratio"] == 1.0
    assert m_ortho["iso_err_vs_identity"] < 1e-4
    assert m_ortho["tight_frame_err"] < 1e-4
    assert m_ortho["rms_coherence"] < 1e-5

    # Verify M = 2d overcomplete case inside torch.no_grad()
    with torch.no_grad():
        w_frame = optimize_overcomplete_tight_frame(64, 32, steps=200, lr=0.1, seed=42)
    assert w_frame.shape == (64, 32)
    row_norms = torch.linalg.norm(w_frame, dim=-1)
    assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-5)

    metrics = frame_coherence_metrics(w_frame)
    wb = welch_bound_coherence(64, 32)
    assert np.isclose(metrics["welch_bound"], wb, atol=1e-7)
    assert metrics["tight_frame_err"] < 0.05
    assert 0.85 <= metrics["iso_err_vs_identity"] <= 1.15
    assert metrics["rms_coherence"] <= 1.05 * wb


def test_overcomplete_frame_interference_forward_and_penalty() -> None:
    """Verify distinct w_a/w_b init, unit-norm outputs, clean gradients, and near-zero initial penalty."""
    layer = OvercompleteFrameInterference(
        embed_dim=32,
        frame_expansion=2.0,
        target_iso_scale=1.25,
        seed=42,
    )
    assert layer.w_a.shape == (64, 32)
    assert layer.w_b.shape == (64, 32)
    assert layer.w_down.shape == (32, 64)
    assert not torch.allclose(layer.w_a, layer.w_b)

    init_penalty = float(layer.frame_penalty().item())
    assert init_penalty < 0.05

    torch.manual_seed(11)
    x = l2_normalize(torch.randn(3, 8, 32), dim=-1).requires_grad_(True)
    y = layer(x)
    assert y.shape == x.shape
    y_norms = torch.linalg.norm(y, dim=-1)
    assert torch.allclose(y_norms, torch.ones_like(y_norms), atol=1e-5)

    loss = y.square().mean() + 0.01 * layer.frame_penalty()
    loss.backward()
    for name, p in [("w_a", layer.w_a), ("w_b", layer.w_b), ("w_down", layer.w_down)]:
        assert p.grad is not None, f"Missing gradient for {name}"
        assert torch.all(torch.isfinite(p.grad)), f"Non-finite gradient for {name}"
        assert torch.any(p.grad != 0.0), f"Zero gradient for {name}"


def test_jax_pytorch_frame_metrics_parity() -> None:
    """Verify numerical parity between PyTorch and JAX frame metrics and penalties, including M <= 1."""
    torch.manual_seed(99)
    for shape in [(48, 24), (24, 48), (16, 16), (1, 1)]:
        w_pt = torch.randn(*shape, dtype=torch.float32) * 1.15
        w_jax = jnp.asarray(w_pt.numpy())

        pen_pt = float(tight_frame_penalty(w_pt, target_iso_scale=1.25).item())
        pen_jax = float(tight_frame_penalty_jax(w_jax, target_iso_scale=1.25))
        assert np.isclose(pen_pt, pen_jax, atol=1e-5, rtol=1e-5)

        m_pt = frame_coherence_metrics(w_pt)
        m_jax = frame_coherence_metrics_jax(w_jax)
        assert set(m_pt.keys()) == set(m_jax.keys())
        for k in m_pt:
            assert np.isclose(m_pt[k], m_jax[k], atol=1e-5, rtol=1e-5), (
                f"Mismatch on key {k} for shape {shape}: pt={m_pt[k]} vs jax={m_jax[k]}"
            )
        if shape == (1, 1):
            assert m_pt["rms_coherence"] == 0.0
            assert m_pt["max_coherence"] == 0.0
            assert m_jax["rms_coherence"] == 0.0
            assert m_jax["max_coherence"] == 0.0


def test_overcomplete_quadratic_born_capacity() -> None:
    """Verify M = 2d tight frame achieves >= 1.6x higher quadratic Born feature rank than M = d basis."""
    d = 64
    w_ortho = optimize_overcomplete_tight_frame(d, d, steps=1, seed=42)
    w_frame = optimize_overcomplete_tight_frame(2 * d, d, steps=250, lr=0.1, seed=42)

    gen = torch.Generator().manual_seed(100)
    psi = l2_normalize(torch.randn((1024, d), generator=gen), dim=-1)

    f_ortho = (psi @ w_ortho.T).square()
    f_frame = (psi @ w_frame.T).square()

    rank_ortho = _effective_feature_covariance_rank(f_ortho)
    rank_frame = _effective_feature_covariance_rank(f_frame)
    assert rank_frame >= 1.6 * rank_ortho, (
        f"Expected rank_frame >= 1.6 * rank_ortho, got {rank_frame:.4f} vs {rank_ortho:.4f}"
    )
