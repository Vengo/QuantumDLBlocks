"""Unit tests for EXPERIMENT_PLAN_003: Tangent-Space Stiefel Muon Projection & Quadratic Retraction."""

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
import torch.nn as nn

from qdlblocks.metrics import stiefel_isometry_error
from qdlblocks.optim.tangent_stiefel_muon import (
    TangentStiefelMuon,
    project_stiefel_tangent,
    stiefel_quadratic_retract,
)
from qdlblocks_jax.functional import stiefel_isometry_error as stiefel_isometry_error_jax
from qdlblocks_jax.tangent_stiefel import (
    newton_schulz_5_jax,
    project_stiefel_tangent_jax,
    stiefel_quadratic_retract_jax,
    tangent_stiefel_muon_step_jax,
)


@pytest.mark.parametrize("shape", [(64, 32), (32, 32), (32, 64)])
def test_project_stiefel_tangent_skew_symmetry_torch_and_jax(shape: tuple[int, int]) -> None:
    """Verify skew-symmetry of Riemannian Stiefel tangent projection in PyTorch and @jax.jit JAX."""
    r, c = shape
    torch.manual_seed(42)
    w = torch.empty(r, c, dtype=torch.float32)
    nn.init.orthogonal_(w)
    u = torch.randn(r, c, dtype=torch.float32) + 0.5 * w
    u_norm = float(torch.norm(u, p="fro").item())

    delta_pt = project_stiefel_tangent(w, u)
    if r >= c:
        sym_pt = w.t() @ delta_pt + delta_pt.t() @ w
    else:
        sym_pt = w @ delta_pt.t() + delta_pt @ w.t()
    rel_err_pt = float(torch.norm(sym_pt, p="fro").item()) / u_norm
    assert rel_err_pt < 1e-5

    jit_proj = jax.jit(project_stiefel_tangent_jax)
    w_jax = jnp.asarray(w.numpy())
    u_jax = jnp.asarray(u.numpy())
    delta_jax = jit_proj(w_jax, u_jax)
    if r >= c:
        sym_jax = w_jax.T @ delta_jax + delta_jax.T @ w_jax
    else:
        sym_jax = w_jax @ delta_jax.T + delta_jax @ w_jax.T
    rel_err_jax = float(jnp.linalg.norm(sym_jax)) / u_norm
    assert rel_err_jax < 1e-5
    np.testing.assert_allclose(delta_pt.numpy(), np.asarray(delta_jax), atol=1e-5)


@pytest.mark.parametrize("shape", [(64, 32), (32, 64)])
def test_stiefel_quadratic_retract_cubic_convergence_torch_and_jax(shape: tuple[int, int]) -> None:
    """Verify cubic convergence of quadratic Stiefel retraction in PyTorch and @jax.jit JAX."""
    r, c = shape
    torch.manual_seed(123)
    w0 = torch.empty(r, c, dtype=torch.float32)
    nn.init.orthogonal_(w0)
    noise = torch.randn(r, c, dtype=torch.float32)
    noise = noise / torch.norm(noise, p="fro")
    w_pert = 1.07 * w0 + 0.25 * noise

    err_0 = stiefel_isometry_error(w_pert)
    assert 0.10 <= err_0 <= 0.25

    w1 = stiefel_quadratic_retract(w_pert, alpha=0.5)
    err_1 = stiefel_isometry_error(w1)
    assert err_1 < err_0 / 5.0

    w_cur = w1
    for _ in range(2):
        w_cur = stiefel_quadratic_retract(w_cur, alpha=0.5)
    err_3 = stiefel_isometry_error(w_cur)
    assert err_3 < 1e-5

    jit_retract = jax.jit(stiefel_quadratic_retract_jax)
    w_jax = jnp.asarray(w_pert.numpy())
    w1_jax = jit_retract(w_jax)
    err_1_jax = float(stiefel_isometry_error_jax(w1_jax))
    assert err_1_jax < err_0 / 5.0

    w_cur_jax = w1_jax
    for _ in range(2):
        w_cur_jax = jit_retract(w_cur_jax)
    err_3_jax = float(stiefel_isometry_error_jax(w_cur_jax))
    assert err_3_jax < 1e-5
    np.testing.assert_allclose(w_cur.numpy(), np.asarray(w_cur_jax), atol=1e-5)


def test_tangent_stiefel_muon_optimizer_step_and_validation() -> None:
    """Verify constructor validation, 1D/2D parameter updates, and PyTorch-JAX single-step parity."""
    p_dummy = nn.Parameter(torch.eye(8))
    with pytest.raises(ValueError, match="learning rate"):
        TangentStiefelMuon([p_dummy], lr=-0.01)
    with pytest.raises(ValueError, match="momentum"):
        TangentStiefelMuon([p_dummy], momentum=1.0)
    with pytest.raises(ValueError, match="ns_steps"):
        TangentStiefelMuon([p_dummy], ns_steps=0)
    with pytest.raises(ValueError, match="retract_alpha"):
        TangentStiefelMuon([p_dummy], retract_alpha=-0.1)
    with pytest.raises(ValueError, match="retract_every"):
        TangentStiefelMuon([p_dummy], retract_every=0)

    with pytest.raises(ValueError, match="2D"):
        project_stiefel_tangent(torch.ones(4), torch.ones(4))
    with pytest.raises(ValueError, match="shape"):
        project_stiefel_tangent(torch.ones(4, 4), torch.ones(4, 2))
    with pytest.raises(ValueError, match="2D"):
        stiefel_quadratic_retract(torch.ones(4))
    with pytest.raises(ValueError, match="alpha"):
        stiefel_quadratic_retract(torch.ones(4, 4), alpha=-0.1)

    torch.manual_seed(7)
    w_init = torch.empty(32, 16, dtype=torch.float32)
    nn.init.orthogonal_(w_init)
    g_2d = torch.randn(32, 16, dtype=torch.float32)

    p_2d = nn.Parameter(w_init.clone())
    p_1d = nn.Parameter(torch.ones(16, dtype=torch.float32))
    p_2d.grad = g_2d.clone()
    p_1d.grad = torch.full((16,), 0.5, dtype=torch.float32)

    opt = TangentStiefelMuon(
        [p_2d, p_1d],
        lr=0.03,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        retract_alpha=0.5,
        retract_every=1,
    )
    opt.step()

    assert stiefel_isometry_error(p_2d.data) < 0.005
    assert not torch.allclose(p_1d.data, torch.ones(16))

    # Verify exact single-step parity with @jax.jit tangent_stiefel_muon_step_jax
    w_jax = jnp.asarray(w_init.numpy())
    g_jax = jnp.asarray(g_2d.numpy())
    m_jax = jnp.zeros_like(w_jax)
    step_fn = jax.jit(
        lambda w, g, m: tangent_stiefel_muon_step_jax(
            w,
            g,
            m,
            lr=0.03,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            retract_alpha=0.5,
        )
    )
    w_next_jax, m_next_jax = step_fn(w_jax, g_jax, m_jax)
    np.testing.assert_allclose(p_2d.data.numpy(), np.asarray(w_next_jax), atol=1e-5)
    np.testing.assert_allclose(
        opt.state[p_2d]["momentum_buffer"].numpy(), np.asarray(m_next_jax), atol=1e-5
    )

    # Also verify newton_schulz_5_jax helper
    u_jax = newton_schulz_5_jax(g_jax, steps=5)
    assert u_jax.shape == (32, 16)


def test_hypothesis_003_script_execution() -> None:
    """Execute examples/13_hypothesis_tangent_stiefel_wsd.py and verify JSON report passes."""
    root = Path(__file__).resolve().parents[1]
    script_path = root / "examples" / "13_hypothesis_tangent_stiefel_wsd.py"
    json_path = root / ".cache" / "hypothesis_experiments_003.json"

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "PASSED" in proc.stdout
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["h1_tangent_projection"]["passed"] is True
    assert data["h2_wsd_plateau_training"]["passed"] is True
    assert data["h2_wsd_plateau_training"]["arm_b_tangent_stiefel_muon"]["final_iso_err"] <= 0.05
