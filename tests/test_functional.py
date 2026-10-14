"""Unit tests for stateless functional operations and quantum/geometric metrics."""

from __future__ import annotations

import torch

from qdlblocks import (
    apply_unitary_rope,
    convex_density_mixture,
    effective_rank,
    is_valid_density_matrix,
    kraus_transform,
    l2_normalize,
    povm_measure,
    power_purify,
    purity,
    riemannian_center,
    state_to_density,
    stiefel_isometry_error,
    von_neumann_entropy,
    zeropower_via_newtonschulz5,
)


def test_state_to_density_invariants() -> None:
    torch.manual_seed(0)
    psi = torch.complex(torch.randn(8, 12), torch.randn(8, 12))
    rho = state_to_density(psi)
    assert rho.shape == (8, 12, 12)

    valid, diag = is_valid_density_matrix(rho, atol=1e-5)
    assert valid, f"Invalid density matrix: {diag}"

    # Pure states must have purity == 1.0 and von Neumann entropy == 0.0
    p = purity(rho)
    s = von_neumann_entropy(rho)
    assert torch.allclose(p, torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(s, torch.tensor(0.0), atol=1e-4)


def test_power_purify_on_mixed_states() -> None:
    torch.manual_seed(1)
    # Create 4 pure states per batch item and mix them equally -> rank-4 mixed state
    psi = torch.complex(torch.randn(6, 4, 10), torch.randn(6, 4, 10))
    rho_pure = state_to_density(psi)  # (6, 4, 10, 10)
    logits = torch.zeros(4)
    rho_mixed = convex_density_mixture(rho_pure, logits, head_dim=1)  # (6, 10, 10)

    valid_m, _ = is_valid_density_matrix(rho_mixed, atol=1e-5)
    assert valid_m

    p_before = purity(rho_mixed, reduction="none")
    s_before = von_neumann_entropy(rho_mixed, reduction="none")

    rho_purified = power_purify(rho_mixed, power=2)
    valid_p, _ = is_valid_density_matrix(rho_purified, atol=1e-5)
    assert valid_p

    p_after = purity(rho_purified, reduction="none")
    s_after = von_neumann_entropy(rho_purified, reduction="none")

    # Power purification on non-uniform or perturbed mixed states increases purity and lowers entropy
    assert torch.all(p_after >= p_before - 1e-5)
    assert torch.all(s_after <= s_before + 1e-5)


def test_kraus_transform_square_and_rectangular() -> None:
    torch.manual_seed(2)
    psi = torch.complex(torch.randn(4, 8), torch.randn(4, 8))
    rho_in = state_to_density(psi)  # (4, 8, 8)

    # Rectangular Kraus operators: 8 -> 14 with K=3 operators
    kraus_ops = torch.complex(torch.randn(3, 14, 8) * 0.2, torch.randn(3, 14, 8) * 0.2)
    rho_out = kraus_transform(rho_in, kraus_ops, normalize=True)

    assert rho_out.shape == (4, 14, 14)
    valid, diag = is_valid_density_matrix(rho_out, atol=1e-5)
    assert valid, f"Kraus output failed density axioms: {diag}"


def test_povm_measure_matches_explicit_trace() -> None:
    torch.manual_seed(3)
    psi = torch.complex(torch.randn(5, 8), torch.randn(5, 8))
    rho = state_to_density(psi)
    l_factors = torch.complex(torch.randn(7, 8, 8) * 0.1, torch.randn(7, 8, 8) * 0.1)

    logits, m_ops = povm_measure(rho, l_factors, return_operators=True)
    assert logits.shape == (5, 7)

    # Compare against explicit O(B * C * N^3) matmul + diagonal trace
    rho_m = torch.matmul(rho.unsqueeze(1), m_ops.unsqueeze(0))
    expected_logits = torch.diagonal(rho_m, dim1=-2, dim2=-1).sum(dim=-1).real
    assert torch.allclose(logits, expected_logits, atol=1e-5)


def test_spherical_rope_and_newton_schulz() -> None:
    torch.manual_seed(4)
    x = torch.randn(4, 16, 32)
    x_norm = l2_normalize(x, dim=-1)
    norms = torch.norm(x_norm, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    x_centered = riemannian_center(x, dim=-1)
    norms_c = torch.norm(x_centered, p=2, dim=-1)
    assert torch.allclose(norms_c, torch.ones_like(norms_c), atol=1e-5)

    # Unitary RoPE preserves L2 norm
    angles = torch.randn(1, 16, 16)
    cos, sin = torch.cos(angles), torch.sin(angles)
    x_rot = apply_unitary_rope(x_norm, cos, sin)
    norms_rot = torch.norm(x_rot, p=2, dim=-1)
    assert torch.allclose(norms_rot, torch.ones_like(norms_rot), atol=1e-5)

    # Quintic Newton-Schulz reduces Stiefel isometry error (Muon polynomial maps SVs to ~[0.7, 1.2])
    G = torch.randn(64, 32)
    err_before = stiefel_isometry_error(G)
    U = zeropower_via_newtonschulz5(G, steps=5)
    err_after = stiefel_isometry_error(U)
    assert err_after < err_before
    assert err_after < 0.45

    rank = effective_rank(x_norm)
    assert 1.0 <= rank <= 32.0
