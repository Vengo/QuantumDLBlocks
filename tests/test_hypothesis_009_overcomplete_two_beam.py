"""Unit tests for EXPERIMENT_PLAN_009: Overcomplete Two-Beam Quantum Interference."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import torch

from qdlblocks.functional import l2_normalize
from qdlblocks.layers.overcomplete_two_beam import (
    OvercompleteTwoBeamInterference,
    optimize_orthogonal_pair_tight_frames,
    overcomplete_two_beam_metrics,
    two_beam_frame_penalty,
)
from qdlblocks_jax.overcomplete_two_beam import (
    overcomplete_two_beam_metrics_jax,
    two_beam_frame_penalty_jax,
)


def test_orthogonal_pair_tight_frame_optimization() -> None:
    """Verify optimize_orthogonal_pair_tight_frames(128, 64, steps=150) achieves tight frames and zero paired coherence."""
    w1, w2 = optimize_orthogonal_pair_tight_frames(128, 64, steps=150, lr=0.08, seed=42)
    assert w1.shape == (128, 64)
    assert w2.shape == (128, 64)

    norms1 = torch.linalg.norm(w1, dim=-1)
    norms2 = torch.linalg.norm(w2, dim=-1)
    assert torch.allclose(norms1, torch.ones_like(norms1), atol=1e-5)
    assert torch.allclose(norms2, torch.ones_like(norms2), atol=1e-5)

    metrics = overcomplete_two_beam_metrics(w1, w2)
    assert metrics["beam1_tight_frame_err"] <= 0.03, (
        f"Expected beam1_tight_frame_err <= 0.03, got {metrics['beam1_tight_frame_err']:.6f}"
    )
    assert metrics["beam2_tight_frame_err"] <= 0.03, (
        f"Expected beam2_tight_frame_err <= 0.03, got {metrics['beam2_tight_frame_err']:.6f}"
    )
    assert metrics["paired_beam_rms_coherence"] <= 0.005, (
        f"Expected paired_beam_rms_coherence <= 0.005, got {metrics['paired_beam_rms_coherence']:.6f}"
    )


def test_overcomplete_two_beam_forward_unit_norm_and_gradients() -> None:
    """Verify unit-norm outputs for 2D/3D inputs and finite non-zero gradients for W_beam1, W_beam2, W_out."""
    mlp = OvercompleteTwoBeamInterference(embed_dim=32, frame_expansion=2.0, init_steps=80, seed=42)
    assert mlp.W_beam1.shape == (64, 32)
    assert mlp.W_beam2.shape == (64, 32)
    assert mlp.W_out.shape == (32, 64)

    torch.manual_seed(17)
    psi_2d = l2_normalize(torch.randn(4, 32), dim=-1).requires_grad_(True)
    out_2d = mlp(psi_2d)
    assert out_2d.shape == (4, 32)
    assert torch.allclose(torch.linalg.norm(out_2d, dim=-1), torch.ones(4), atol=1e-5)

    psi_3d = l2_normalize(torch.randn(2, 8, 32), dim=-1).requires_grad_(True)
    out_3d = mlp(psi_3d)
    assert out_3d.shape == (2, 8, 32)
    assert torch.allclose(torch.linalg.norm(out_3d, dim=-1), torch.ones(2, 8), atol=1e-5)

    target = l2_normalize(torch.randn(2, 8, 32), dim=-1)
    loss = ((out_3d - target) ** 2).mean() + 0.02 * mlp.frame_penalty()
    loss.backward()

    for name, param in [("W_beam1", mlp.W_beam1), ("W_beam2", mlp.W_beam2), ("W_out", mlp.W_out)]:
        assert param.grad is not None, f"Missing gradient for {name}"
        assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradient for {name}"
        assert torch.any(param.grad != 0.0), f"Zero gradient for {name}"


def test_overcomplete_two_beam_breaks_antipodal_symmetry() -> None:
    """Verify mlp(-psi) is neither mlp(psi) nor -mlp(psi), confirming mixed even/odd harmonic expressivity."""
    mlp = OvercompleteTwoBeamInterference(embed_dim=32, frame_expansion=2.0, init_steps=80, seed=42)
    torch.manual_seed(23)
    psi = l2_normalize(torch.randn(8, 32), dim=-1)
    y_pos = mlp(psi)
    y_neg = mlp(-psi)

    diff_even = float(torch.norm(y_pos - y_neg).item())
    diff_odd = float(torch.norm(y_pos + y_neg).item())
    assert diff_even > 0.1, f"Expected ||mlp(psi) - mlp(-psi)|| > 0.1, got {diff_even:.6f}"
    assert diff_odd > 0.1, f"Expected ||mlp(psi) + mlp(-psi)|| > 0.1, got {diff_odd:.6f}"


def test_pytorch_jax_two_beam_penalty_equivalence() -> None:
    """Verify two_beam_frame_penalty and all 7 keys of coherence_metrics agree within 1e-5 between PyTorch and JAX."""
    mlp = OvercompleteTwoBeamInterference(embed_dim=32, frame_expansion=2.0, init_steps=60, seed=42)
    with torch.no_grad():
        torch.manual_seed(99)
        mlp.W_beam1.add_(0.08 * torch.randn_like(mlp.W_beam1))
        mlp.W_beam2.add_(0.08 * torch.randn_like(mlp.W_beam2))

    w1_pt = mlp.W_beam1.detach()
    w2_pt = mlp.W_beam2.detach()
    w1_jax = jnp.asarray(w1_pt.cpu().numpy())
    w2_jax = jnp.asarray(w2_pt.cpu().numpy())

    pen_pt = float(two_beam_frame_penalty(w1_pt, w2_pt, eps=mlp.eps).item())
    pen_jax = float(two_beam_frame_penalty_jax(w1_jax, w2_jax, eps=mlp.eps))
    assert np.isclose(pen_pt, pen_jax, atol=1e-5, rtol=1e-5), (
        f"Penalty mismatch: PyTorch={pen_pt:.8f} vs JAX={pen_jax:.8f}"
    )

    m_pt = mlp.coherence_metrics()
    m_jax = overcomplete_two_beam_metrics_jax(w1_jax, w2_jax, eps=mlp.eps)

    expected_keys = {
        "beam1_tight_frame_err",
        "beam2_tight_frame_err",
        "beam1_rms_coherence",
        "beam2_rms_coherence",
        "welch_bound",
        "paired_beam_rms_coherence",
        "naive_iso_err",
    }
    assert set(m_pt.keys()) == expected_keys
    assert set(m_jax.keys()) == expected_keys

    for key in expected_keys:
        val_pt = float(m_pt[key])
        val_jax = float(m_jax[key])
        assert np.isclose(val_pt, val_jax, atol=1e-5, rtol=1e-5), (
            f"Metric mismatch on {key!r}: PyTorch={val_pt:.8f} vs JAX={val_jax:.8f}"
        )


def test_quantum_born_gpt_overcomplete_two_beam_integration() -> None:
    """Verify QuantumBornGPT with use_overcomplete_two_beam=True runs forward/backward cleanly."""
    from qdlblocks.models.born_gpt import QuantumBornGPT

    model = QuantumBornGPT(
        vocab_size=64,
        embed_dim=32,
        seed_dim=16,
        num_heads=4,
        num_layers=2,
        block_size=16,
        readout_mode="tied_stiefel",
        use_overcomplete_two_beam=True,
    )
    for blk in model.blocks:
        assert isinstance(blk.two_beam_mlp, OvercompleteTwoBeamInterference)

    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    logits, loss, layer_states = model(idx, targets=targets, return_layer_states=True)
    assert logits.shape == (2, 16, 64)
    assert loss is not None and torch.isfinite(loss)
    for st in layer_states:
        assert torch.allclose(torch.linalg.norm(st, dim=-1), torch.ones(2, 16), atol=1e-5)

    total_loss = loss + 0.01 * sum(b.two_beam_mlp.frame_penalty() for b in model.blocks)  # type: ignore[union-attr]
    total_loss.backward()
    for blk in model.blocks:
        assert blk.two_beam_mlp.W_beam1.grad is not None  # type: ignore[union-attr]
        assert blk.two_beam_mlp.W_beam2.grad is not None  # type: ignore[union-attr]
        assert blk.two_beam_mlp.W_out.grad is not None  # type: ignore[union-attr]

