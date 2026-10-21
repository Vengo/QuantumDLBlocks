"""Unit tests for EXPERIMENT_PLAN_004: Multi-Beam Kraus Subspace POVM Readout."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from qdlblocks.functional import l2_normalize
from qdlblocks.layers.kraus_readout import (
    MultiBeamKrausReadout,
    compute_logit_matrix_effective_rank,
)
from qdlblocks_jax.kraus_readout import (
    compute_logit_matrix_effective_rank_jax,
    init_multibeam_kraus_params,
    multibeam_kraus_forward_jax,
)


def test_logit_matrix_effective_rank_exact_bounds_torch_and_jax() -> None:
    """Verify exact effective singular rank bounds for rank-1 and rank-k matrices in Torch & JAX."""
    torch.manual_seed(7)
    n, v = 64, 48
    u = torch.randn(n, 1)
    u = u - u.mean(dim=0, keepdim=True)
    vec_v = torch.randn(1, v)
    c = torch.randn(1, v)
    logits_rank1 = u @ vec_v + c

    r_rank1_torch = compute_logit_matrix_effective_rank(logits_rank1)
    jit_eff_rank = jax.jit(compute_logit_matrix_effective_rank_jax)
    r_rank1_jax = jit_eff_rank(jnp.asarray(logits_rank1.numpy()))

    assert torch.isclose(r_rank1_torch, torch.tensor(1.0), atol=1e-4)
    assert np.isclose(float(r_rank1_jax), 1.0, atol=1e-4)

    # Construct a column-centered matrix with k = 12 equal nonzero singular values
    k = 12
    q_left, _ = torch.linalg.qr(torch.randn(n, k + 1))
    # Project out constant vector 1_N so columns of U_k are strictly zero-mean
    ones = torch.ones(n, 1) / (n**0.5)
    u_perp = q_left - ones @ (ones.T @ q_left)
    u_k, _ = torch.linalg.qr(u_perp[:, :k])
    v_k, _ = torch.linalg.qr(torch.randn(v, k))
    logits_rank_k = (u_k @ v_k.T) * 3.5 + torch.randn(1, v)

    r_k_torch = compute_logit_matrix_effective_rank(logits_rank_k)
    r_k_jax = jit_eff_rank(jnp.asarray(logits_rank_k.numpy()))

    assert torch.isclose(r_k_torch, torch.tensor(float(k)), atol=1e-4)
    assert np.isclose(float(r_k_jax), float(k), atol=1e-4)


def test_logit_matrix_effective_rank_input_validation_torch_and_jax() -> None:
    """Verify ValueError on invalid logit tensor shapes for Torch and JAX."""
    with pytest.raises(ValueError, match="ndim >= 2"):
        compute_logit_matrix_effective_rank(torch.randn(16))
    with pytest.raises(ValueError, match="N >= 2 and V >= 2"):
        compute_logit_matrix_effective_rank(torch.randn(1, 16))
    with pytest.raises(ValueError, match="N >= 2 and V >= 2"):
        compute_logit_matrix_effective_rank(torch.randn(16, 1))

    with pytest.raises(ValueError, match="ndim >= 2"):
        compute_logit_matrix_effective_rank_jax(jnp.ones((16,)))
    with pytest.raises(ValueError, match="N >= 2 and V >= 2"):
        compute_logit_matrix_effective_rank_jax(jnp.ones((1, 16)))
    with pytest.raises(ValueError, match="N >= 2 and V >= 2"):
        compute_logit_matrix_effective_rank_jax(jnp.ones((16, 1)))


def test_multibeam_kraus_readout_validation_and_shapes() -> None:
    """Verify constructor validation in both Torch and JAX and 3D (B, T, V) output shapes."""
    with pytest.raises(ValueError, match="num_beams \\* seed_dim"):
        MultiBeamKrausReadout(vocab_size=64, embed_dim=64, seed_dim=16, num_beams=5)
    with pytest.raises(ValueError, match="vocab_size must be >= 2"):
        MultiBeamKrausReadout(vocab_size=1, embed_dim=64, seed_dim=16, num_beams=2)
    with pytest.raises(ValueError, match="adapter_init_scale must be > 0.0"):
        MultiBeamKrausReadout(
            vocab_size=64, embed_dim=64, seed_dim=16, num_beams=2, adapter_init_scale=0.0
        )

    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError, match="num_beams \\* seed_dim"):
        init_multibeam_kraus_params(key, vocab_size=64, embed_dim=64, seed_dim=16, num_beams=5)

    psi = l2_normalize(torch.randn(4, 8, 64), dim=-1)
    base_wte = l2_normalize(torch.randn(100, 16), dim=-1)
    for k in (1, 4):
        layer = MultiBeamKrausReadout(vocab_size=100, embed_dim=64, seed_dim=16, num_beams=k)
        out = layer(psi, base_wte)
        assert out.shape == (4, 8, 100)


def test_multibeam_kraus_readout_gradients_and_temperature_modes() -> None:
    """Verify finite gradients on all learnable parameters and fixed temperature buffer mode."""
    torch.manual_seed(19)
    psi = l2_normalize(torch.randn(3, 5, 64), dim=-1)
    base_wte = l2_normalize(torch.randn(48, 16), dim=-1)

    layer = MultiBeamKrausReadout(
        vocab_size=48,
        embed_dim=64,
        seed_dim=16,
        num_beams=4,
        beam_adapter_rank=8,
        learnable_temperature=True,
    )
    logits = layer(psi, base_wte)
    loss = logits.square().mean()
    loss.backward()

    for name in ("readout_rot.weight", "beam_vocab_a", "beam_seed_b", "beam_weights", "inv_temp"):
        param = dict(layer.named_parameters())[name]
        assert param.grad is not None, f"Missing gradient for {name}"
        assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradient for {name}"
        assert torch.any(param.grad != 0.0), f"Zero gradient for {name}"

    fixed_layer = MultiBeamKrausReadout(
        vocab_size=48,
        embed_dim=64,
        seed_dim=16,
        num_beams=1,
        learnable_temperature=False,
    )
    assert "inv_temp" not in dict(fixed_layer.named_parameters())
    assert "inv_temp" in dict(fixed_layer.named_buffers())
    assert fixed_layer.beam_vocab_a is None


def test_multibeam_kraus_readout_rank_expansion_and_torch_jax_parity() -> None:
    """Verify validation, shapes, gradients, rank expansion, and Torch-JAX numerical parity."""
    with pytest.raises(ValueError, match="num_beams \\* seed_dim"):
        MultiBeamKrausReadout(vocab_size=128, embed_dim=64, seed_dim=16, num_beams=5)

    torch.manual_seed(42)
    b, t, d, d_seed, v = 4, 32, 64, 16, 128
    psi = l2_normalize(torch.randn(b, t, d), dim=-1)
    base_wte = l2_normalize(torch.randn(v, d_seed), dim=-1)
    psi_jax = jnp.asarray(psi.numpy())
    wte_jax = jnp.asarray(base_wte.numpy())
    jit_forward = jax.jit(multibeam_kraus_forward_jax)

    eff_ranks: dict[int, float] = {}
    for k in (1, 4):
        torch.manual_seed(100 + k)
        layer = MultiBeamKrausReadout(
            vocab_size=v,
            embed_dim=d,
            seed_dim=d_seed,
            num_beams=k,
            beam_adapter_rank=16,
            adapter_init_scale=1.0,
        )
        out_torch = layer(psi, base_wte)
        assert out_torch.shape == (b, t, v)

        loss = out_torch.square().mean()
        loss.backward()
        assert layer.readout_rot.weight.grad is not None
        assert torch.all(torch.isfinite(layer.readout_rot.weight.grad))
        assert layer.inv_temp.grad is not None
        if k > 1:
            assert layer.beam_vocab_a is not None and layer.beam_vocab_a.grad is not None
            assert layer.beam_seed_b is not None and layer.beam_seed_b.grad is not None
            assert layer.beam_weights is not None and layer.beam_weights.grad is not None
            assert torch.all(torch.isfinite(layer.beam_vocab_a.grad))
            assert torch.all(torch.isfinite(layer.beam_seed_b.grad))
            assert torch.all(torch.isfinite(layer.beam_weights.grad))

        jax_params: dict[str, jax.Array] = {
            "readout_rot": jnp.asarray(layer.readout_rot.weight.detach().numpy()),
            "inv_temp": jnp.asarray(layer.inv_temp.detach().numpy()),
        }
        if k > 1:
            assert (
                layer.beam_vocab_a is not None
                and layer.beam_seed_b is not None
                and layer.beam_weights is not None
            )
            jax_params["beam_vocab_a"] = jnp.asarray(layer.beam_vocab_a.detach().numpy())
            jax_params["beam_seed_b"] = jnp.asarray(layer.beam_seed_b.detach().numpy())
            jax_params["beam_weights"] = jnp.asarray(layer.beam_weights.detach().numpy())

        out_jax = jit_forward(jax_params, psi_jax, wte_jax)
        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            np.asarray(out_jax),
            atol=1e-5,
            rtol=1e-5,
        )
        eff_ranks[k] = float(compute_logit_matrix_effective_rank(out_torch.detach()).item())

    assert eff_ranks[1] <= 16.0 + 1e-4
    assert eff_ranks[4] >= 32.0
    assert eff_ranks[4] > 2.0 * eff_ranks[1]


def test_hypothesis_004_script_execution() -> None:
    """Execute examples/14_hypothesis_multibeam_kraus_readout.py and verify JSON output."""
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "examples" / "14_hypothesis_multibeam_kraus_readout.py"
    cache_path = repo_root / ".cache" / "hypothesis_experiments_004.json"

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "EXPERIMENT_PLAN_004" in proc.stdout
    assert cache_path.exists()

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["h1_rank_scaling"]["passed"] is True
    assert payload["h2_learning_capacity"]["passed"] is True
    assert payload["h1_rank_scaling"]["r_sigma_k1"] <= 16.0 + 1e-4
    assert payload["h1_rank_scaling"]["r_sigma_k4"] >= 32.0
    assert (
        payload["h2_learning_capacity"]["arm_b_k4"]["val_ce"]
        <= payload["h2_learning_capacity"]["arm_a_k1"]["val_ce"] - 0.15
    )
