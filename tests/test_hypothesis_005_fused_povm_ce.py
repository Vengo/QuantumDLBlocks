"""Unit and hypothesis verification tests for EXPERIMENT_PLAN_005:
Analytic-Gradient L2-Tiled POVM Cross-Entropy.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from qdlblocks.kernels.fused_povm_ce import (
    estimate_povm_head_roofline,
    fused_tiled_povm_cross_entropy,
    measure_povm_ce_roofline,
)
from qdlblocks_jax.fused_povm_ce import (
    estimate_povm_head_roofline_jax,
    fused_tiled_povm_cross_entropy_jax,
)


def test_fused_tiled_povm_ce_exact_gradients_torch() -> None:
    """Verify exact forward loss and backward gradients in PyTorch for float32 and bfloat16."""
    torch.manual_seed(7)
    N = 500
    D = 32
    V = 256
    z_init = F.normalize(torch.randn(N, D, dtype=torch.float32), p=2.0, dim=-1)
    m_init = F.normalize(torch.randn(V, D, dtype=torch.float32), p=2.0, dim=-1)
    beta_init = torch.tensor(24.0, dtype=torch.float32)
    targets = torch.randint(0, V, (N,), dtype=torch.long)

    # Dense FP32 reference
    z_ref = z_init.clone().requires_grad_(True)
    m_ref = m_init.clone().requires_grad_(True)
    b_ref = beta_init.clone().requires_grad_(True)
    loss_ref = F.cross_entropy(F.linear(z_ref, m_ref) * b_ref, targets)
    loss_ref.backward()
    assert z_ref.grad is not None and m_ref.grad is not None and b_ref.grad is not None

    for tile_size in (64, 128, 200):
        # 1. float32 inputs (including non-divisible N=500 % tile_size != 0)
        z_t = z_init.clone().requires_grad_(True)
        m_t = m_init.clone().requires_grad_(True)
        b_t = beta_init.clone().requires_grad_(True)
        loss_t = fused_tiled_povm_cross_entropy(
            z_t, m_t, b_t, targets, tile_size=tile_size
        )
        loss_t.backward()
        assert z_t.grad is not None and m_t.grad is not None and b_t.grad is not None

        torch.testing.assert_close(loss_t, loss_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(z_t.grad, z_ref.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(m_t.grad, m_ref.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(b_t.grad, b_ref.grad, atol=1e-4, rtol=1e-4)

        # 2. bfloat16 inputs vs dense bfloat16-cast reference
        z_bf_ref = z_init.to(torch.bfloat16).detach().requires_grad_(True)
        m_bf_ref = m_init.to(torch.bfloat16).detach().requires_grad_(True)
        b_bf_ref = beta_init.to(torch.bfloat16).detach().requires_grad_(True)
        loss_bf_ref = F.cross_entropy(
            F.linear(z_bf_ref.float(), m_bf_ref.float()) * b_bf_ref.float(),
            targets,
        )
        loss_bf_ref.backward()

        z_bf = z_init.to(torch.bfloat16).detach().requires_grad_(True)
        m_bf = m_init.to(torch.bfloat16).detach().requires_grad_(True)
        b_bf = beta_init.to(torch.bfloat16).detach().requires_grad_(True)
        loss_bf = fused_tiled_povm_cross_entropy(
            z_bf, m_bf, b_bf, targets, tile_size=tile_size
        )
        loss_bf.backward()
        assert z_bf.grad is not None and m_bf.grad is not None and b_bf.grad is not None
        assert (
            z_bf_ref.grad is not None
            and m_bf_ref.grad is not None
            and b_bf_ref.grad is not None
        )

        torch.testing.assert_close(loss_bf, loss_bf_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(z_bf.grad, z_bf_ref.grad, atol=5e-3, rtol=1e-2)
        torch.testing.assert_close(m_bf.grad, m_bf_ref.grad, atol=5e-3, rtol=1e-2)
        torch.testing.assert_close(b_bf.grad, b_bf_ref.grad, atol=5e-3, rtol=1e-2)


def test_fused_tiled_povm_ce_exact_gradients_jax() -> None:
    """Verify @jax.jit value_and_grad of fused_tiled_povm_cross_entropy_jax against dense JAX."""
    rng = np.random.default_rng(19)
    N = 256
    D = 32
    V = 128
    z_np = rng.standard_normal((N, D)).astype(np.float32)
    z_np /= np.linalg.norm(z_np, axis=-1, keepdims=True)
    m_np = rng.standard_normal((V, D)).astype(np.float32)
    m_np /= np.linalg.norm(m_np, axis=-1, keepdims=True)
    y_np = rng.integers(0, V, size=(N,), dtype=np.int32)

    z_j = jnp.asarray(z_np)
    m_j = jnp.asarray(m_np)
    b_j = jnp.asarray(25.0, dtype=jnp.float32)
    y_j = jnp.asarray(y_np)

    def _dense_ce(z_in: jax.Array, m_in: jax.Array, b_in: jax.Array) -> jax.Array:
        logits = (
            jnp.matmul(z_in, m_in.T, precision=jax.lax.Precision.HIGHEST)
            * b_in.reshape(())
        )
        lse = jax.nn.logsumexp(logits, axis=-1)
        tgt = jnp.take_along_axis(logits, y_j[:, None], axis=-1).squeeze(-1)
        return jnp.mean(lse - tgt)

    loss_ref, (dz_ref, dm_ref, db_ref) = jax.jit(
        jax.value_and_grad(_dense_ce, argnums=(0, 1, 2))
    )(z_j, m_j, b_j)

    loss_tiled, (dz_tiled, dm_tiled, db_tiled) = jax.jit(
        jax.value_and_grad(
            lambda z_in, m_in, b_in: fused_tiled_povm_cross_entropy_jax(
                z_in, m_in, b_in, y_j, tile_size=64
            ),
            argnums=(0, 1, 2),
        )
    )(z_j, m_j, b_j)

    np.testing.assert_allclose(np.asarray(loss_tiled), np.asarray(loss_ref), atol=1e-5)
    np.testing.assert_allclose(np.asarray(dz_tiled), np.asarray(dz_ref), atol=1e-4)
    np.testing.assert_allclose(np.asarray(dm_tiled), np.asarray(dm_ref), atol=1e-4)
    np.testing.assert_allclose(np.asarray(db_tiled), np.asarray(db_ref), atol=1e-4)


def test_estimate_povm_head_roofline_l2_transition() -> None:
    """Verify L2 cache transition and >= 2.5x arithmetic intensity jump on FineWeb 100M config."""
    N = 16384
    D_seed = 128
    V = 50304
    l2_bytes = 96 * 1024 * 1024

    r_512 = estimate_povm_head_roofline(
        N, D_seed, V, tile_size=512, dtype_bytes=2, l2_cache_bytes=l2_bytes
    )
    r_16384 = measure_povm_ce_roofline(
        N, D_seed, V, tile_size=16384, dtype_bytes=2, l2_cache_bytes=l2_bytes
    )
    r_jax_512 = estimate_povm_head_roofline_jax(
        N, D_seed, V, tile_size=512, dtype_bytes=2, l2_cache_bytes=l2_bytes
    )

    assert r_512 == r_jax_512
    assert r_512["fits_in_l2"] is True
    assert r_512["peak_logit_buffer_mb"] == 49.125
    assert r_16384["fits_in_l2"] is False
    assert r_16384["peak_logit_buffer_mb"] == 1572.0
    assert float(r_512["arithmetic_intensity"]) >= 2.5 * float(
        r_16384["arithmetic_intensity"]
    )


def test_fused_tiled_povm_ce_validation_and_edge_cases() -> None:
    """Verify input validation raises ValueError on invalid shapes or parameters."""
    z = torch.randn(16, 8)
    m = torch.randn(32, 8)
    b = torch.tensor(10.0)
    y = torch.randint(0, 32, (16,))

    with pytest.raises(ValueError, match="tile_size"):
        fused_tiled_povm_cross_entropy(z, m, b, y, tile_size=0)
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        fused_tiled_povm_cross_entropy(z, torch.randn(32, 4), b, y, tile_size=8)
    with pytest.raises(ValueError, match="targets must be 1D"):
        fused_tiled_povm_cross_entropy(z, m, b, torch.randint(0, 32, (8,)), tile_size=8)

    with pytest.raises(ValueError, match="num_tokens"):
        estimate_povm_head_roofline(0, 8, 32, 8)

    z_j = jnp.ones((16, 8), dtype=jnp.float32)
    m_j = jnp.ones((32, 8), dtype=jnp.float32)
    b_j = jnp.asarray(10.0, dtype=jnp.float32)
    y_j = jnp.zeros((16,), dtype=jnp.int32)
    with pytest.raises(ValueError, match="divisible by tile_size"):
        fused_tiled_povm_cross_entropy_jax(z_j, m_j, b_j, y_j, tile_size=5)


def test_hypothesis_005_script_execution() -> None:
    """Execute examples/15_hypothesis_fused_povm_roofline.py and verify artifact output."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "examples" / "15_hypothesis_fused_povm_roofline.py"
    artifact_path = repo_root / ".cache" / "hypothesis_experiments_005.json"

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "EXPERIMENT_PLAN_005" in proc.stdout
    assert artifact_path.is_file()

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["h1_gradient_and_memory"]["passed"] is True
    assert payload["h2_roofline_scaling"]["passed"] is True
    assert payload["h1_gradient_and_memory"]["peak_buffer_mb_tiled"] == 4.0
    assert payload["h1_gradient_and_memory"]["peak_buffer_mb_dense"] == 32.0
    assert payload["h2_roofline_scaling"]["ai_ratio"] >= 2.5
