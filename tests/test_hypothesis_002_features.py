"""Unit tests for EXPERIMENT_PLAN_002 features:
- Empirical state density, Effective Quantum Rank, and Purity Participation Ratio (PyTorch & JAX)
- Decoupled per-horizon POVM temperatures and auxiliary readout stop-gradient
- Static-shape strided multi-token POVM prediction in PyTorch and Haliax/JAX
"""

from __future__ import annotations

import math

from qdlblocks import (
    QuantumBornGPT,
    effective_quantum_rank,
    empirical_state_density,
    l2_normalize,
    purity_participation_ratio,
)
from qdlblocks_jax import (
    HaxQuantumBornGPT,
    HaxQuantumBornLMConfig,
    effective_quantum_rank as jax_effective_quantum_rank,
    empirical_state_density as jax_empirical_state_density,
    purity_participation_ratio as jax_purity_participation_ratio,
)

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch


def test_empirical_density_and_effective_rank_bounds_torch_and_jax() -> None:
    """Verify exact bounds and static is_density dispatch in PyTorch and JAX (with @jax.jit)."""
    d = 16

    # 1. Rank-1 pure state batch (all N=16 rows equal the same unit ket)
    v0 = torch.randn(1, d, dtype=torch.float32)
    v0 = l2_normalize(v0, dim=-1)
    psi_rank1_t = v0.repeat(d, 1)
    psi_rank1_j = jnp.asarray(psi_rank1_t.numpy())

    r_rank1_t = float(effective_quantum_rank(psi_rank1_t).item())
    pr_rank1_t = float(purity_participation_ratio(psi_rank1_t).item())
    r_rank1_j = float(jax.jit(jax_effective_quantum_rank)(psi_rank1_j))
    pr_rank1_j = float(jax.jit(jax_purity_participation_ratio)(psi_rank1_j))

    assert math.isclose(r_rank1_t, 1.0, abs_tol=1e-4)
    assert math.isclose(pr_rank1_t, 1.0, abs_tol=1e-4)
    assert math.isclose(r_rank1_j, 1.0, abs_tol=1e-4)
    assert math.isclose(pr_rank1_j, 1.0, abs_tol=1e-4)

    # 2. Orthonormal basis batch (psi = I_16, N == D == 16) with default is_density=False
    eye_t = torch.eye(d, dtype=torch.float32)
    eye_j = jnp.eye(d, dtype=jnp.float32)

    rho_eye_t = empirical_state_density(eye_t)
    rho_eye_j = jax.jit(jax_empirical_state_density)(eye_j)
    expected_rho = np.eye(d, dtype=np.float32) / float(d)
    np.testing.assert_allclose(rho_eye_t.real.numpy(), expected_rho, atol=1e-5)
    np.testing.assert_allclose(np.real(np.asarray(rho_eye_j)), expected_rho, atol=1e-5)

    assert math.isclose(float(effective_quantum_rank(eye_t).item()), float(d), abs_tol=1e-4)
    assert math.isclose(float(purity_participation_ratio(eye_t).item()), float(d), abs_tol=1e-4)
    assert math.isclose(float(jax.jit(jax_effective_quantum_rank)(eye_j)), float(d), abs_tol=1e-4)
    assert math.isclose(float(jax.jit(jax_purity_participation_ratio)(eye_j)), float(d), abs_tol=1e-4)

    # 3. Explicit is_density=True branch on unnormalized positive-definite matrix 2.5 * I_16
    unnorm_rho_t = 2.5 * eye_t
    unnorm_rho_j = 2.5 * eye_j
    assert math.isclose(
        float(effective_quantum_rank(unnorm_rho_t, is_density=True).item()),
        float(d),
        abs_tol=1e-4,
    )
    assert math.isclose(
        float(purity_participation_ratio(unnorm_rho_t, is_density=True).item()),
        float(d),
        abs_tol=1e-4,
    )
    assert math.isclose(
        float(jax.jit(lambda x: jax_effective_quantum_rank(x, is_density=True))(unnorm_rho_j)),
        float(d),
        abs_tol=1e-4,
    )
    assert math.isclose(
        float(jax.jit(lambda x: jax_purity_participation_ratio(x, is_density=True))(unnorm_rho_j)),
        float(d),
        abs_tol=1e-4,
    )

    # 4. General Renyi-2 vs. von Neumann inequality on random 3D state batch (8, 32, 16)
    torch.manual_seed(99)
    rand_psi_t = torch.randn(8, 32, d, dtype=torch.float32)
    rand_psi_j = jnp.asarray(rand_psi_t.numpy())

    r_rand_t = float(effective_quantum_rank(rand_psi_t).item())
    pr_rand_t = float(purity_participation_ratio(rand_psi_t).item())
    r_rand_j = float(jax_effective_quantum_rank(rand_psi_j))
    pr_rand_j = float(jax_purity_participation_ratio(rand_psi_j))

    assert 1.0 <= pr_rand_t <= r_rand_t + 1e-5 <= float(d) + 1e-4
    assert 1.0 <= pr_rand_j <= r_rand_j + 1e-5 <= float(d) + 1e-4
    assert math.isclose(r_rand_t, r_rand_j, rel_tol=1e-4, abs_tol=1e-4)
    assert math.isclose(pr_rand_t, pr_rand_j, rel_tol=1e-4, abs_tol=1e-4)


@pytest.mark.parametrize("readout_mode", ["tied_stiefel", "symmetric_dual", "asymmetric"])
def test_decoupled_mtp_temperatures_and_detach_torch(readout_mode: str) -> None:
    """Verify horizon-specific temperatures, detach behavior, and telemetry across all readout modes."""
    tau_0 = 0.035
    model_single = QuantumBornGPT(
        vocab_size=64,
        embed_dim=32,
        seed_dim=16,
        num_heads=4,
        num_layers=2,
        hidden_dim=64,
        block_size=16,
        povm_temperature=tau_0,
        readout_mode=readout_mode,  # type: ignore[arg-type]
        learnable_temperature=True,
        mtp_horizons=1,
    )
    assert model_single.mtp_inv_temps is None

    model_shared = QuantumBornGPT(
        vocab_size=64,
        embed_dim=32,
        seed_dim=16,
        num_heads=4,
        num_layers=2,
        hidden_dim=64,
        block_size=16,
        povm_temperature=tau_0,
        readout_mode=readout_mode,  # type: ignore[arg-type]
        learnable_temperature=True,
        mtp_horizons=3,
        mtp_decoupled_temp=False,
        mtp_detach_readout=False,
    )
    assert model_shared.mtp_inv_temps is None

    model = QuantumBornGPT(
        vocab_size=64,
        embed_dim=32,
        seed_dim=16,
        num_heads=4,
        num_layers=2,
        hidden_dim=64,
        block_size=16,
        povm_temperature=tau_0,
        readout_mode=readout_mode,  # type: ignore[arg-type]
        learnable_temperature=True,
        mtp_horizons=3,
        mtp_loss_weight=0.5,
        mtp_stride=1,
        mtp_detach_readout=True,
        mtp_decoupled_temp=True,
    )
    assert model.mtp_inv_temps is not None
    assert model.mtp_inv_temps.shape == (2,)
    expected_b2 = 1.0 / (tau_0 * math.sqrt(2.0))
    expected_b3 = 1.0 / (tau_0 * math.sqrt(3.0))
    assert math.isclose(float(model.mtp_inv_temps[0].item()), expected_b2, rel_tol=1e-5)
    assert math.isclose(float(model.mtp_inv_temps[1].item()), expected_b3, rel_tol=1e-5)

    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 64)
    assert loss is not None and torch.isfinite(loss)

    for key in ("ce_t+1", "ce_t+2", "ce_t+3", "beta_t+1", "beta_t+2", "beta_t+3"):
        assert key in model.last_mtp_losses

    loss.backward()
    assert model.mtp_inv_temps.grad is not None
    assert torch.all(torch.isfinite(model.mtp_inv_temps.grad))
    assert torch.any(model.mtp_inv_temps.grad != 0.0)
    assert model.mtp_rotations is not None
    assert model.mtp_rotations[0].grad is not None
    assert torch.any(model.mtp_rotations[0].grad != 0.0)

    # Verify stop-gradient on readout dictionary & primary temperature:
    # Changing mtp_loss_weight from 0.0 to 2.0 when mtp_detach_readout=True and mtp_decoupled_temp=True
    # must leave d(loss)/d(povm_head_seed_delta) and d(loss)/d(tied_inv_temp) identical!
    if readout_mode == "tied_stiefel":
        torch.manual_seed(123)
        m_zero = QuantumBornGPT(
            vocab_size=64,
            embed_dim=32,
            seed_dim=16,
            num_heads=4,
            num_layers=2,
            hidden_dim=64,
            block_size=16,
            readout_mode="tied_stiefel",
            learnable_temperature=True,
            mtp_horizons=3,
            mtp_loss_weight=0.0,
            mtp_detach_readout=True,
            mtp_decoupled_temp=True,
        )
        torch.manual_seed(123)
        m_heavy = QuantumBornGPT(
            vocab_size=64,
            embed_dim=32,
            seed_dim=16,
            num_heads=4,
            num_layers=2,
            hidden_dim=64,
            block_size=16,
            readout_mode="tied_stiefel",
            learnable_temperature=True,
            mtp_horizons=3,
            mtp_loss_weight=2.0,
            mtp_detach_readout=True,
            mtp_decoupled_temp=True,
        )
        _, l0 = m_zero(x, y)
        _, l2 = m_heavy(x, y)
        assert l0 is not None and l2 is not None
        l0.backward()
        l2.backward()
        assert m_zero.povm_head_seed_delta is not None and m_heavy.povm_head_seed_delta is not None
        assert m_zero.tied_inv_temp is not None and m_heavy.tied_inv_temp is not None
        torch.testing.assert_close(m_zero.povm_head_seed_delta.grad, m_heavy.povm_head_seed_delta.grad)
        torch.testing.assert_close(m_zero.tied_inv_temp.grad, m_heavy.tied_inv_temp.grad)


def test_strided_mtp_shapes_and_jax_parity() -> None:
    """Verify strided MTP in both PyTorch QuantumBornGPT and JAX HaxQuantumBornGPT / HaxQuantumBornLMConfig."""
    # 1. PyTorch strided MTP
    model_pt = QuantumBornGPT(
        vocab_size=64,
        embed_dim=32,
        seed_dim=16,
        num_heads=4,
        num_layers=2,
        hidden_dim=64,
        block_size=16,
        readout_mode="tied_stiefel",
        mtp_horizons=3,
        mtp_stride=4,
        mtp_detach_readout=True,
        mtp_decoupled_temp=True,
    )
    x_pt = torch.randint(0, 64, (2, 16))
    y_pt = torch.randint(0, 64, (2, 16))
    logits_pt, loss_pt = model_pt(x_pt, y_pt)
    assert logits_pt.shape == (2, 16, 64)
    assert loss_pt is not None and torch.isfinite(loss_pt)
    loss_pt.backward()
    assert model_pt.mtp_inv_temps is not None and model_pt.mtp_inv_temps.grad is not None
    assert torch.all(torch.isfinite(model_pt.mtp_inv_temps.grad))

    mtp_logits_pt = model_pt.forward_mtp_logits(x_pt)
    assert len(mtp_logits_pt) == 3
    for lg in mtp_logits_pt:
        assert lg.shape == (2, 16, 64)

    # 2. Haliax / JAX strided MTP via HaxQuantumBornLMConfig
    cfg = HaxQuantumBornLMConfig(
        vocab_size=64,
        seq_len=16,
        seed_dim=16,
        embed_dim=32,
        num_heads=4,
        mlp_dim=64,
        num_layers=2,
        use_flash_born=False,
        mtp_horizons=3,
        mtp_loss_weight=0.5,
        mtp_stride=4,
        mtp_detach_readout=True,
        mtp_decoupled_temp=True,
    )
    key = jax.random.PRNGKey(0)
    model_jax = cfg.build(key=key)
    assert isinstance(model_jax, HaxQuantumBornGPT)
    assert model_jax.mtp_stride == 4
    assert model_jax.mtp_detach_readout is True
    assert model_jax.mtp_decoupled_temp is True
    assert model_jax.mtp_inv_temps.shape == (2,)

    Batch = hax.Axis("batch", 2)
    Seq = hax.Axis("seq", 16)
    tok_arr = jax.random.randint(key, (2, 16), 0, 64)
    tgt_arr = jax.random.randint(jax.random.PRNGKey(1), (2, 16), 0, 64)
    tok_named = hax.NamedArray(tok_arr, (Batch, Seq))
    tgt_named = hax.NamedArray(tgt_arr, (Batch, Seq))

    def loss_fn(m: HaxQuantumBornGPT) -> jax.Array:
        _, l_val = m(tok_named, tgt_named)
        assert l_val is not None
        return l_val

    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model_jax)
    assert jnp.isfinite(loss_val)
    assert jnp.all(jnp.isfinite(grads.mtp_inv_temps))
    assert jnp.any(grads.mtp_inv_temps != 0.0)

    mtp_logits_jax = model_jax.forward_mtp_logits(tok_named)
    assert len(mtp_logits_jax) == 3
    for lg_h in mtp_logits_jax:
        assert lg_h.axes == (Batch, Seq, model_jax.Vocab)
