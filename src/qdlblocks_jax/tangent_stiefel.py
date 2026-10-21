"""Pure JAX Tangent-Space Stiefel Muon Projection and Quadratic Retraction Transformations."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from .functional import zeropower_via_newtonschulz5

__all__ = [
    "project_stiefel_tangent_jax",
    "stiefel_quadratic_retract_jax",
    "newton_schulz_5_jax",
    "tangent_stiefel_muon_step_jax",
]


def newton_schulz_5_jax(
    G: jax.Array,
    steps: int = 5,
    eps: float = 1e-7,
) -> jax.Array:
    """Compute the orthogonal polar factor via 5-step quintic Newton-Schulz iteration in JAX."""
    return zeropower_via_newtonschulz5(G, steps=steps, eps=eps)


def project_stiefel_tangent_jax(w: jax.Array, u: jax.Array) -> jax.Array:
    """Project an ambient 2D direction `u` onto the Riemannian Stiefel tangent space T_w St(r, c) in JAX."""
    if w.ndim != 2:
        raise ValueError(f"project_stiefel_tangent_jax expects a 2D array w, got {w.ndim}D")
    if u.shape != w.shape:
        raise ValueError(
            f"project_stiefel_tangent_jax expects u.shape == w.shape, got {u.shape} vs {w.shape}"
        )
    wf = w.astype(jnp.float32)
    uf = u.astype(jnp.float32)
    r, c = wf.shape
    if r >= c:
        wt_u = jnp.matmul(wf.T, uf, precision=jax.lax.Precision.HIGHEST)
        sym = 0.5 * (wt_u + wt_u.T)
        out = uf - jnp.matmul(wf, sym, precision=jax.lax.Precision.HIGHEST)
    else:
        u_wt = jnp.matmul(uf, wf.T, precision=jax.lax.Precision.HIGHEST)
        sym = 0.5 * (u_wt + u_wt.T)
        out = uf - jnp.matmul(sym, wf, precision=jax.lax.Precision.HIGHEST)
    return out.astype(u.dtype)


def stiefel_quadratic_retract_jax(w: jax.Array, alpha: float = 0.5) -> jax.Array:
    """Apply a first-order quadratic Stiefel retraction step to a 2D array `w` in JAX."""
    if w.ndim != 2:
        raise ValueError(f"stiefel_quadratic_retract_jax expects a 2D array w, got {w.ndim}D")
    if isinstance(alpha, (int, float)) and alpha < 0.0:
        raise ValueError(f"alpha must be >= 0.0, got {alpha}")
    wf = w.astype(jnp.float32)
    r, c = wf.shape
    if r >= c:
        gram = jnp.matmul(wf.T, wf, precision=jax.lax.Precision.HIGHEST)
        eye = jnp.eye(c, dtype=jnp.float32)
        out = wf - alpha * jnp.matmul(wf, gram - eye, precision=jax.lax.Precision.HIGHEST)
    else:
        gram = jnp.matmul(wf, wf.T, precision=jax.lax.Precision.HIGHEST)
        eye = jnp.eye(r, dtype=jnp.float32)
        out = wf - alpha * jnp.matmul(gram - eye, wf, precision=jax.lax.Precision.HIGHEST)
    return out.astype(w.dtype)


def tangent_stiefel_muon_step_jax(
    w: jax.Array,
    g: jax.Array,
    m: jax.Array,
    *,
    lr: float | jax.Array = 0.02,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    retract_alpha: float = 0.5,
    tangent_project_before_ns: bool = True,
    tangent_project_after_ns: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Execute a single pure-JAX Tangent-Space Stiefel Muon parameter and momentum update."""
    if w.ndim != 2 or g.shape != w.shape or m.shape != w.shape:
        raise ValueError(
            f"tangent_stiefel_muon_step_jax expects 2D arrays of matching shape, got {w.shape}, {g.shape}, {m.shape}"
        )
    new_m = momentum * m + g
    g_dir = (g + momentum * new_m) if nesterov else new_m
    if tangent_project_before_ns:
        g_dir = project_stiefel_tangent_jax(w, g_dir)
    aspect_scale = math.sqrt(max(1.0, w.shape[0] / w.shape[1]))
    u = zeropower_via_newtonschulz5(g_dir, steps=ns_steps) * aspect_scale
    if tangent_project_after_ns:
        u = project_stiefel_tangent_jax(w, u)
    w_tent = w - lr * u.astype(w.dtype)
    if isinstance(retract_alpha, (int, float)):
        if retract_alpha > 0.0:
            w_next = stiefel_quadratic_retract_jax(w_tent, alpha=retract_alpha)
        else:
            w_next = w_tent
    else:
        w_next = jnp.where(
            retract_alpha > 0.0,
            stiefel_quadratic_retract_jax(w_tent, alpha=retract_alpha),
            w_tent,
        )
    return w_next, new_m
