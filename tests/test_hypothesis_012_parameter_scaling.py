"""Unit tests for EXPERIMENT_PLAN_012: Multi-Axis Parameter Scaling (Depth, Overcomplete 3d/4d Frames, Width & Readout Reinvestment)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from qdlblocks import (
    OvercompleteTwoBeamInterference,
    QuantumBornGPT,
    compute_depth_fidelity_and_geodesic_profile,
    compute_depth_residual_scale,
    compute_parameter_scaling_breakdown,
    default_overcomplete_target_iso_scale,
    effective_quantum_rank,
    l2_normalize,
)
from qdlblocks_jax import (
    compute_depth_fidelity_and_geodesic_profile_jax,
    compute_depth_residual_scale_jax,
    default_overcomplete_target_iso_scale_jax,
)


def test_parameter_scaling_breakdown_and_residual_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify parameter accounting across readout modes, residual scaling, and CLI flags."""
    assert default_overcomplete_target_iso_scale(1.0) == pytest.approx(1.0)
    assert default_overcomplete_target_iso_scale(2.0) == pytest.approx(1.25)
    assert default_overcomplete_target_iso_scale(3.0) == pytest.approx(1.50)
    assert default_overcomplete_target_iso_scale(4.0) == pytest.approx(1.75)

    assert compute_depth_residual_scale(6, base_layers=6, enabled=True) == pytest.approx(1.0)
    assert compute_depth_residual_scale(12, base_layers=6, enabled=False) == pytest.approx(1.0)
    assert compute_depth_residual_scale(12, base_layers=6, enabled=True) == pytest.approx(math.sqrt(0.5))
    assert compute_depth_residual_scale(9, base_layers=6, enabled=True) == pytest.approx(math.sqrt(6.0 / 9.0))

    V, d, d_seed, L = 256, 64, 24, 6
    m_stiefel = QuantumBornGPT(
        vocab_size=V,
        embed_dim=d,
        seed_dim=d_seed,
        num_heads=4,
        num_layers=L,
        hidden_dim=96,
        readout_mode="tied_stiefel",
        use_two_beam_mlp=True,
        use_unet_interferometry=True,
    )
    m_vocab = QuantumBornGPT(
        vocab_size=V,
        embed_dim=d,
        seed_dim=d_seed,
        num_heads=4,
        num_layers=L,
        hidden_dim=96,
        readout_mode="tied_vocab",
        use_two_beam_mlp=True,
        use_unet_interferometry=True,
    )
    m_asym = QuantumBornGPT(
        vocab_size=V,
        embed_dim=d,
        seed_dim=d_seed,
        num_heads=4,
        num_layers=L,
        hidden_dim=96,
        readout_mode="asymmetric",
        use_two_beam_mlp=True,
        use_unet_interferometry=True,
    )

    b_stiefel = compute_parameter_scaling_breakdown(m_stiefel)
    b_vocab = compute_parameter_scaling_breakdown(m_vocab)
    b_asym = compute_parameter_scaling_breakdown(m_asym)

    for b in (b_stiefel, b_vocab, b_asym):
        assert b["total_params"] == b["backbone_params"] + b["vocab_lift_params"] + b["readout_params"]
        assert b["backbone_params"] == b["attention_params"] + b["mlp_params"] + b["unet_params"]
        assert b["backbone_fraction"] == pytest.approx(b["backbone_params"] / b["total_params"])

    # tied_vocab saves exactly V * d_seed parameters vs tied_stiefel
    assert b_stiefel["total_params"] - b_vocab["total_params"] == V * d_seed
    assert b_stiefel["readout_params"] - b_vocab["readout_params"] == V * d_seed
    assert b_vocab["backbone_fraction"] > b_stiefel["backbone_fraction"]

    # Verify CLI flags in examples/11_fineweb_100m_multiepoch_gpu_tpu.py
    script_path = Path(__file__).resolve().parents[1] / "examples" / "11_fineweb_100m_multiepoch_gpu_tpu.py"
    spec = importlib.util.spec_from_file_location("ex11_scaling", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--num-layers",
            "10",
            "--embed-dim",
            "1280",
            "--num-heads",
            "20",
            "--hidden-dim",
            "1920",
            "--frame-expansion",
            "3.0",
            "--readout-mode",
            "tied_vocab",
            "--depth-scaled-residual",
        ],
    )
    args = mod.parse_args()
    assert args.num_layers == 10
    assert args.embed_dim == 1280
    assert args.num_heads == 20
    assert args.hidden_dim == 1920
    assert args.frame_expansion == pytest.approx(3.0)
    assert args.readout_mode == "tied_vocab"
    assert args.depth_scaled_residual is True


def test_depth_scaled_residual_preserves_fidelity_and_reduces_grad_cv() -> None:
    """Verify H1a: depth_scaled_residual=True at L=12 preserves initial fidelity >= 0.18 and lowers grad CV >= 15%."""
    V, d, d_seed, L = 256, 128, 64, 12
    torch.manual_seed(42)
    m_unscaled = QuantumBornGPT(
        vocab_size=V,
        embed_dim=d,
        seed_dim=d_seed,
        num_heads=8,
        num_layers=L,
        hidden_dim=256,
        block_size=32,
        readout_mode="tied_stiefel",
        value_residual_mixing=True,
        use_unet_interferometry=True,
        depth_scaled_residual=False,
    )
    torch.manual_seed(42)
    m_scaled = QuantumBornGPT(
        vocab_size=V,
        embed_dim=d,
        seed_dim=d_seed,
        num_heads=8,
        num_layers=L,
        hidden_dim=256,
        block_size=32,
        readout_mode="tied_stiefel",
        value_residual_mixing=True,
        use_unet_interferometry=True,
        depth_scaled_residual=True,
    )
    idx = torch.randint(0, V, (8, 32), dtype=torch.long)
    targets = torch.randint(0, V, (8, 32), dtype=torch.long)

    prof_unscaled = compute_depth_fidelity_and_geodesic_profile(m_unscaled, idx, targets=targets)
    prof_scaled = compute_depth_fidelity_and_geodesic_profile(m_scaled, idx, targets=targets)

    fid_unscaled = float(prof_unscaled["end_to_end_fidelity"])
    fid_scaled = float(prof_scaled["end_to_end_fidelity"])
    step_scaled = float(prof_scaled["mean_geodesic_step_rad"])
    cv_unscaled = float(prof_unscaled["grad_norm_cv"])
    cv_scaled = float(prof_scaled["grad_norm_cv"])
    cv_reduction = (cv_unscaled - cv_scaled) / max(cv_unscaled, 1e-12)

    assert fid_unscaled < 0.12, f"Expected unscaled L=12 fidelity < 0.12, got {fid_unscaled:.4f}"
    assert fid_scaled >= 0.18, f"Expected depth-scaled L=12 fidelity >= 0.18, got {fid_scaled:.4f}"
    assert step_scaled <= 0.65, f"Expected mean geodesic step <= 0.65 rad, got {step_scaled:.4f}"
    assert cv_reduction >= 0.15, (
        f"Expected >= 15% gradient norm CV reduction, got {100.0 * cv_reduction:.2f}% "
        f"(unscaled={cv_unscaled:.4f}, scaled={cv_scaled:.4f})"
    )


def test_overcomplete_two_beam_3d_and_4d_rank_monotonicity() -> None:
    """Verify H2a: OvercompleteTwoBeamInterference feature rank grows strictly monotonically for rho in {2, 3, 4}."""
    d = 64
    n_samples = 1024
    gen = torch.Generator().manual_seed(42)
    x = l2_normalize(torch.randn((n_samples, d), generator=gen, dtype=torch.float32), dim=-1)

    ranks: dict[float, float] = {}
    for rho in (2.0, 3.0, 4.0):
        layer = OvercompleteTwoBeamInterference(
            embed_dim=d,
            frame_expansion=rho,
            target_iso_scale=default_overcomplete_target_iso_scale(rho),
            init_steps=150,
            seed=42,
        )
        with torch.no_grad():
            u1 = l2_normalize(layer.W_beam1.float(), dim=-1)
            u2 = l2_normalize(layer.W_beam2.float(), dim=-1)
            z_2beam = F.linear(x, u1) * F.linear(x, u2)
            ranks[rho] = float(effective_quantum_rank(l2_normalize(z_2beam, dim=-1)).item())

    assert ranks[4.0] > ranks[3.0] > ranks[2.0], (
        f"Expected monotonic rank growth R(4.0) > R(3.0) > R(2.0), got {ranks}"
    )
    assert ranks[4.0] >= 1.15 * ranks[2.0], (
        f"Expected R(4.0) >= 1.15 * R(2.0), got {ranks[4.0]:.2f} vs {ranks[2.0]:.2f}"
    )


def test_tied_vocab_and_asymmetric_readout_forward_backward() -> None:
    """Verify tied_vocab and asymmetric readout forward/backward, MTP, and un-bottlenecked rank."""
    V, d, d_seed = 256, 64, 16
    torch.manual_seed(42)
    idx = torch.randint(0, V, (8, 32), dtype=torch.long)
    targets = torch.randint(0, V, (8, 32), dtype=torch.long)

    m_vocab = QuantumBornGPT(
        vocab_size=V,
        embed_dim=d,
        seed_dim=d_seed,
        num_heads=4,
        num_layers=4,
        hidden_dim=96,
        block_size=32,
        readout_mode="tied_vocab",
        use_overcomplete_seed_adapter=True,
        mtp_horizons=3,
    )
    assert m_vocab.povm_head_seed_delta is None
    logits_v, loss_v = m_vocab(idx, targets=targets)
    assert logits_v.shape == (8, 32, V)
    assert loss_v is not None and torch.isfinite(loss_v)
    loss_v.backward()
    assert m_vocab.wte.weight.grad is not None
    assert m_vocab.readout_rot is not None and m_vocab.readout_rot.weight.grad is not None
    mtp_outs = m_vocab.forward_mtp_logits(idx)
    assert len(mtp_outs) == 3

    m_asym = QuantumBornGPT(
        vocab_size=V,
        embed_dim=d,
        seed_dim=d_seed,
        num_heads=4,
        num_layers=4,
        hidden_dim=96,
        block_size=32,
        readout_mode="asymmetric",
        mtp_horizons=2,
    )
    logits_a, loss_a, l_states = m_asym(idx, targets=targets, return_layer_states=True)
    assert logits_a.shape == (8, 32, V)
    assert loss_a is not None and torch.isfinite(loss_a)
    loss_a.backward()
    assert m_asym.povm_readout is not None and m_asym.povm_readout.povm_head.grad is not None

    with torch.no_grad():
        psi_flat = l_states[-1].reshape(-1, d).float()
        r_eff = float(effective_quantum_rank(psi_flat).item())
    assert r_eff > float(d_seed), f"Expected asymmetric pre-POVM rank > d_seed={d_seed}, got {r_eff:.2f}"


def test_jax_parameter_scaling_parity() -> None:
    """Verify numerical parity (<= 1e-5) between PyTorch and JAX parameter scaling diagnostics."""
    for rho in (1.0, 2.0, 3.0, 4.0):
        assert abs(default_overcomplete_target_iso_scale(rho) - default_overcomplete_target_iso_scale_jax(rho)) < 1e-12
    for L in (4, 6, 9, 10, 12):
        assert abs(compute_depth_residual_scale(L) - compute_depth_residual_scale_jax(L)) < 1e-12

    torch.manual_seed(42)
    model = QuantumBornGPT(
        vocab_size=128,
        embed_dim=64,
        seed_dim=24,
        num_heads=4,
        num_layers=8,
        hidden_dim=96,
        block_size=16,
        readout_mode="tied_vocab",
        depth_scaled_residual=True,
    )
    idx = torch.randint(0, 128, (4, 16), dtype=torch.long)
    with torch.no_grad():
        _, _, layer_states = model(idx, return_layer_states=True)
    pt_prof = compute_depth_fidelity_and_geodesic_profile(model, idx, targets=None, eps=1e-6)

    assert model._last_psi_0 is not None
    psi_0_jax = jnp.asarray(model._last_psi_0.detach().cpu().numpy(), dtype=jnp.float32)
    l_states_jax = [jnp.asarray(st.detach().cpu().numpy(), dtype=jnp.float32) for st in layer_states]
    jax_prof = compute_depth_fidelity_and_geodesic_profile_jax(psi_0_jax, l_states_jax, eps=1e-6)

    assert abs(float(pt_prof["end_to_end_fidelity"]) - float(jax_prof["end_to_end_fidelity"])) <= 1e-5
    assert abs(float(pt_prof["cumulative_geodesic_length_rad"]) - float(jax_prof["cumulative_geodesic_length_rad"])) <= 1e-5
    assert abs(float(pt_prof["mean_geodesic_step_rad"]) - float(jax_prof["mean_geodesic_step_rad"])) <= 1e-5
    for a, b in zip(pt_prof["layer_geodesic_steps_rad"], jax_prof["layer_geodesic_steps_rad"]):  # type: ignore[arg-type]
        assert abs(float(a) - float(b)) <= 1e-5
