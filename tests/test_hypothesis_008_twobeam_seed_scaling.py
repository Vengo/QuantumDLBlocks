"""Unit tests for Experiment Plan 008: Two-Beam Bilinear Interference vs. Single-Beam Overcomplete Frames
and Seed-Space Capacity Scaling Diagnostics (PyTorch & JAX parity).
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qdlblocks.functional import l2_normalize
from qdlblocks.layers.overcomplete_frame import optimize_overcomplete_tight_frame
from qdlblocks.layers.seed_scaling_diagnostics import (
    compute_coupled_seed_pullback_rank,
    compute_cross_beam_coherence,
    compute_cross_subspace_regression_mse,
    compute_effective_seed_utilization_ratio,
    compute_interference_feature_rank,
    compute_paired_beam_orthogonality_stats,
    single_beam_signed_born_features,
    two_beam_bilinear_born_features,
)
from qdlblocks_jax.seed_scaling_diagnostics import (
    compute_coupled_seed_pullback_rank_jax,
    compute_cross_beam_coherence_jax,
    compute_effective_seed_utilization_ratio_jax,
    compute_interference_feature_rank_jax,
    compute_paired_beam_orthogonality_stats_jax,
    single_beam_signed_born_features_jax,
    two_beam_bilinear_born_features_jax,
)


def test_single_vs_twobeam_feature_shapes_and_gradients() -> None:
    """Verify shapes (B, T, M) and finite, non-zero gradients for single-beam and two-beam features."""
    torch.manual_seed(42)
    b, t, d = 4, 8, 32
    m1 = 64
    m2 = 48

    psi1 = l2_normalize(torch.randn(b, t, d, dtype=torch.float32), dim=-1).detach().requires_grad_(True)
    w_up = torch.randn(m1, d, dtype=torch.float32, requires_grad=True)

    f1 = single_beam_signed_born_features(psi1, w_up)
    assert f1.shape == (b, t, m1)
    assert torch.isfinite(f1).all()

    loss1 = (f1 * torch.linspace(-1.0, 1.0, m1)).sum()
    loss1.backward()
    assert psi1.grad is not None and torch.isfinite(psi1.grad).all() and psi1.grad.abs().sum().item() > 0.0
    assert w_up.grad is not None and torch.isfinite(w_up.grad).all() and w_up.grad.abs().sum().item() > 0.0

    psi2 = l2_normalize(torch.randn(b, t, d, dtype=torch.float32), dim=-1).detach().requires_grad_(True)
    w_beam1 = torch.randn(m2, d, dtype=torch.float32, requires_grad=True)
    w_beam2 = torch.randn(m2, d, dtype=torch.float32, requires_grad=True)

    f2 = two_beam_bilinear_born_features(psi2, w_beam1, w_beam2)
    assert f2.shape == (b, t, m2)
    assert torch.isfinite(f2).all()

    loss2 = (f2 * torch.linspace(-1.0, 1.0, m2)).sum()
    loss2.backward()
    assert psi2.grad is not None and torch.isfinite(psi2.grad).all() and psi2.grad.abs().sum().item() > 0.0
    assert w_beam1.grad is not None and torch.isfinite(w_beam1.grad).all() and w_beam1.grad.abs().sum().item() > 0.0
    assert w_beam2.grad is not None and torch.isfinite(w_beam2.grad).all() and w_beam2.grad.abs().sum().item() > 0.0


def test_interference_feature_rank_and_cross_beam_coherence() -> None:
    """Verify feature rank on 2D/3D/N=1 inputs and paired-beam orthogonality stats."""
    torch.manual_seed(123)
    b, t, m = 8, 16, 24
    feats_3d = torch.randn(b, t, m, dtype=torch.float32)
    feats_2d = feats_3d.reshape(b * t, m)

    r3 = compute_interference_feature_rank(feats_3d)
    r2 = compute_interference_feature_rank(feats_2d)

    for k in ("feature_eff_rank", "feature_participation_ratio", "feature_trace"):
        assert abs(r3[k] - r2[k]) < 1e-9
    assert 1.0 < r2["feature_eff_rank"] <= float(m) + 1e-5
    assert 1.0 < r2["feature_participation_ratio"] <= float(m) + 1e-5
    assert r2["feature_trace"] > 0.0

    # N=1 boundary case
    r1 = compute_interference_feature_rank(feats_2d[:1])
    assert r1 == {
        "feature_eff_rank": 1.0,
        "feature_participation_ratio": 1.0,
        "feature_trace": 0.0,
    }

    # Orthogonal paired beams: cyclic shift of an orthonormal basis in R^{32 x 32}
    q, _ = torch.linalg.qr(torch.randn(32, 32, dtype=torch.float32))
    w_b1 = q.T.contiguous()
    w_b2 = torch.roll(w_b1, shifts=1, dims=0)

    coh = compute_cross_beam_coherence(w_b1, w_b2)
    coh_alias = compute_paired_beam_orthogonality_stats(w_b1, w_b2)
    assert coh["rms_paired_beam_coherence"] < 1e-5
    assert coh["max_paired_beam_coherence"] < 1e-5
    assert coh == coh_alias


def test_twobeam_degree3_cross_subspace_advantage() -> None:
    """Verify on synthetic degree-3 cross-subspace target (d=32, M1=64, M2=48) that two-beam
    achieves >= 2.5x lower held-out test MSE than single-beam after 150 steps of AdamW.
    """
    torch.manual_seed(42)
    d = 32
    m1 = 64
    m2 = 48
    k_out = 16
    n_train = 512
    n_test = 256
    scale = float(d) ** 1.5

    psi_train = l2_normalize(torch.randn(n_train, d, dtype=torch.float32), dim=-1)
    psi_test = l2_normalize(torch.randn(n_test, d, dtype=torch.float32), dim=-1)

    g_a = torch.Generator().manual_seed(777)
    u_a, _ = torch.linalg.qr(torch.randn(d, d, generator=g_a, dtype=torch.float32))
    u_a = u_a.T.contiguous()
    u_b = torch.roll(u_a, shifts=1, dims=0)

    g_p = torch.Generator().manual_seed(888)
    p_proj, _ = torch.linalg.qr(torch.randn(d, k_out, generator=g_p, dtype=torch.float32))
    p_proj = p_proj.T.contiguous()  # (16, 32)

    def _target(psi: torch.Tensor) -> torch.Tensor:
        za = F.linear(psi, u_a)
        zb = F.linear(psi, u_b)
        return scale * F.linear((za * za) * zb, p_proj)

    y_train = _target(psi_train)
    y_test = _target(psi_test)

    w_up_raw = optimize_overcomplete_tight_frame(m1, d, steps=100, lr=0.05, seed=42)
    w1_raw = optimize_overcomplete_tight_frame(m2, d, steps=100, lr=0.05, seed=101)
    w2_raw = optimize_overcomplete_tight_frame(m2, d, steps=100, lr=0.05, seed=202)

    w_up_init = l2_normalize(torch.cat([u_a, u_b], dim=0) + 0.05 * w_up_raw, dim=-1)
    w1_init = l2_normalize(torch.cat([u_a, w1_raw[d:]], dim=0) + 0.05 * w1_raw, dim=-1)
    w2_init = l2_normalize(torch.cat([u_b, w2_raw[d:]], dim=0) + 0.05 * w2_raw, dim=-1)

    # Map A: SingleBeam (M1 = 64)
    w_up = nn.Parameter(20.0 * w_up_init.clone())
    w_down_1 = nn.Parameter(torch.zeros(k_out, m1, dtype=torch.float32))

    with torch.no_grad():
        f1_tr_init = scale * single_beam_signed_born_features(psi_train, w_up)
        f1_te_init = scale * single_beam_signed_born_features(psi_test, w_up)
        ridge_1 = compute_cross_subspace_regression_mse(f1_tr_init, y_train, f1_te_init, y_test, ridge_lambda=1e-4)

    opt1 = torch.optim.AdamW([w_up, w_down_1], lr=1e-2, weight_decay=1e-2)
    for _ in range(150):
        opt1.zero_grad(set_to_none=True)
        pred1 = F.linear(scale * single_beam_signed_born_features(psi_train, w_up), w_down_1)
        loss1 = F.mse_loss(pred1, y_train)
        loss1.backward()
        opt1.step()

    with torch.no_grad():
        pred1_test = F.linear(scale * single_beam_signed_born_features(psi_test, w_up), w_down_1)
        mse_1beam = float(F.mse_loss(pred1_test, y_test).item())

    # Map B: TwoBeam (M2 = 48)
    w1 = nn.Parameter(20.0 * w1_init.clone())
    w2 = nn.Parameter(20.0 * w2_init.clone())
    w_down_2 = nn.Parameter(torch.zeros(k_out, m2, dtype=torch.float32))

    with torch.no_grad():
        f2_tr_init = scale * two_beam_bilinear_born_features(psi_train, w1, w2)
        f2_te_init = scale * two_beam_bilinear_born_features(psi_test, w1, w2)
        ridge_2 = compute_cross_subspace_regression_mse(f2_tr_init, y_train, f2_te_init, y_test, ridge_lambda=1e-4)

    opt2 = torch.optim.AdamW([w1, w2, w_down_2], lr=1e-2, weight_decay=1e-2)
    for _ in range(150):
        opt2.zero_grad(set_to_none=True)
        pred2 = F.linear(scale * two_beam_bilinear_born_features(psi_train, w1, w2), w_down_2)
        loss2 = F.mse_loss(pred2, y_train)
        loss2.backward()
        opt2.step()

    with torch.no_grad():
        pred2_test = F.linear(scale * two_beam_bilinear_born_features(psi_test, w1, w2), w_down_2)
        mse_2beam = float(F.mse_loss(pred2_test, y_test).item())

    assert ridge_2["test_mse"] < ridge_1["test_mse"]
    assert mse_2beam <= mse_1beam / 2.5, (
        f"Expected TwoBeam test MSE <= SingleBeam / 2.5, got {mse_2beam:.6f} vs {mse_1beam:.6f}"
    )


def test_pytorch_jax_diagnostics_equivalence() -> None:
    """Verify PyTorch and JAX diagnostics match within max_abs_diff < 1e-5."""
    torch.manual_seed(99)
    n, d, m = 32, 16, 24
    psi_pt = l2_normalize(torch.randn(n, d, dtype=torch.float32), dim=-1)
    w_up_pt = torch.randn(m, d, dtype=torch.float32)
    w1_pt = torch.randn(m, d, dtype=torch.float32)
    w2_pt = torch.randn(m, d, dtype=torch.float32)

    psi_jax = jnp.asarray(psi_pt.numpy())
    w_up_jax = jnp.asarray(w_up_pt.numpy())
    w1_jax = jnp.asarray(w1_pt.numpy())
    w2_jax = jnp.asarray(w2_pt.numpy())

    f1_pt = single_beam_signed_born_features(psi_pt, w_up_pt)
    f1_jax = single_beam_signed_born_features_jax(psi_jax, w_up_jax)
    assert float(np.max(np.abs(f1_pt.numpy() - np.asarray(f1_jax)))) < 1e-5

    f2_pt = two_beam_bilinear_born_features(psi_pt, w1_pt, w2_pt)
    f2_jax = two_beam_bilinear_born_features_jax(psi_jax, w1_jax, w2_jax)
    assert float(np.max(np.abs(f2_pt.numpy() - np.asarray(f2_jax)))) < 1e-5

    rank_pt = compute_interference_feature_rank(f2_pt)
    rank_jax = compute_interference_feature_rank_jax(f2_jax)
    for k, v_pt in rank_pt.items():
        assert abs(v_pt - float(rank_jax[k])) < 1e-4

    coh_pt = compute_cross_beam_coherence(w1_pt, w2_pt)
    coh_jax = compute_cross_beam_coherence_jax(w1_jax, w2_jax)
    coh_stats_jax = compute_paired_beam_orthogonality_stats_jax(w1_jax, w2_jax)
    for k, v_pt in coh_pt.items():
        assert abs(v_pt - float(coh_jax[k])) < 1e-5
        assert abs(v_pt - float(coh_stats_jax[k])) < 1e-5

    util_pt = compute_effective_seed_utilization_ratio(psi_pt)
    util_jax = compute_effective_seed_utilization_ratio_jax(psi_jax)
    for k, v_pt in util_pt.items():
        assert abs(v_pt - float(util_jax[k])) < 1e-4

    pb_pt = compute_coupled_seed_pullback_rank(
        lambda x: single_beam_signed_born_features(x, w_up_pt),
        psi_pt,
    )
    pb_jax = compute_coupled_seed_pullback_rank_jax(
        lambda x: single_beam_signed_born_features_jax(x, w_up_jax),
        psi_jax,
    )
    for k, v_pt in pb_pt.items():
        assert abs(v_pt - float(pb_jax[k])) < 1e-4
