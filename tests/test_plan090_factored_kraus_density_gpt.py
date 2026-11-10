"""Unit tests for Wave 70 (`EXPERIMENT_PLAN_090.md`): Factored Cross-Head Kraus Superoperator & LUT Packing."""

from __future__ import annotations

import torch

from scripts.factored_kraus_density_inspector import (
    verify_lut_pack_unpack_equivalence,
    verify_theorem9_algebraic_equivalence_and_speedup,
    verify_theorem10_cptp_and_representational_power,
    verify_theorem11_riemannian_momentum_buffer,
    verify_wave70_model_forward_backward_step,
)


def test_lut_pack_unpack_equivalence() -> None:
    """Verify scatter-free LUT Hermitian pack/unpack matches reference to machine precision."""
    res = verify_lut_pack_unpack_equivalence(torch.device("cpu"))
    assert res["passed"], f"LUT pack/unpack audit failed: {res}"


def test_theorem9_algebraic_equivalence() -> None:
    """Verify Trick A (O(H M) associative) and Trick B (attention-commuted) match O(H^2 M) reference."""
    res = verify_theorem9_algebraic_equivalence_and_speedup(torch.device("cpu"))
    assert res["passed"], f"Theorem 9 algebraic equivalence audit failed: {res}"


def test_theorem10_cptp_and_representational_power() -> None:
    """Verify CPTP closure, strict expressivity separation over scalar mixing, and Rank 16 amplification."""
    res = verify_theorem10_cptp_and_representational_power(torch.device("cpu"))
    assert res["passed"], f"Theorem 10 representational power audit failed: {res}"


def test_wave70_model_forward_backward_step() -> None:
    """Verify full FactoredKrausPureDensityGPT forward, backward, and quotient optimizer step."""
    res = verify_wave70_model_forward_backward_step(torch.device("cpu"))
    assert res["passed"], f"Wave 70 model step audit failed: {res}"


def test_theorem11_riemannian_momentum_buffer() -> None:
    """Verify Theorem 11 (Riemannian Quotient Momentum Buffer at beta=0.95) invariants."""
    res = verify_theorem11_riemannian_momentum_buffer(torch.device("cpu"))
    assert res["passed"], f"Theorem 11 momentum buffer audit failed: {res}"

