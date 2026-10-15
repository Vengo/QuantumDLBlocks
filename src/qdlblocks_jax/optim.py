"""Optax Stiefel-Manifold Muon optimizer with Continuous Riemannian Stiefel Penalty."""

from __future__ import annotations

import math
from typing import Any, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .functional import zeropower_via_newtonschulz5

__all__ = [
    "StiefelMuonState",
    "scale_by_stiefel_muon",
    "build_quantum_optax_optimizer",
]


class StiefelMuonState(NamedTuple):
    """State for Optax Stiefel-Muon transformation."""

    momentum: optax.Updates


def _apply_stiefel_muon_2d(
    g_2d: jax.Array,
    m_2d: jax.Array,
    w_2d: jax.Array,
    lr: float,
    momentum: float,
    nesterov: bool,
    ns_steps: int,
    weight_decay: float,
    decay_mode: Literal["smooth", "cautious"],
    stiefel_penalty: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute 2D Stiefel-Muon update (delta_w, new_m)."""
    new_m = momentum * m_2d + (1.0 - momentum) * g_2d
    u_raw = ((1.0 - momentum) * g_2d + momentum * new_m) if nesterov else new_m
    u_ortho = zeropower_via_newtonschulz5(u_raw, steps=ns_steps)
    m_rows, n_cols = w_2d.shape
    scale = math.sqrt(max(1.0, float(m_rows) / float(n_cols)))
    u_scaled = u_ortho * scale

    w_f = w_2d.astype(jnp.float32)
    u_f = u_scaled.astype(jnp.float32)

    if weight_decay > 0.0:
        if decay_mode == "cautious":
            mask = (u_f * w_f >= 0.0).astype(jnp.float32)
            wd_term = weight_decay * mask * w_f
        else:
            wd_term = weight_decay * w_f
    else:
        wd_term = jnp.zeros_like(w_f)

    if stiefel_penalty > 0.0:
        if m_rows >= n_cols:
            gram_err = (
                jnp.matmul(w_f.T, w_f, precision=jax.lax.Precision.HIGHEST)
                - jnp.eye(n_cols, dtype=jnp.float32)
            )
            r_raw = jnp.matmul(w_f, gram_err, precision=jax.lax.Precision.HIGHEST)
        else:
            gram_err = (
                jnp.matmul(w_f, w_f.T, precision=jax.lax.Precision.HIGHEST)
                - jnp.eye(m_rows, dtype=jnp.float32)
            )
            r_raw = jnp.matmul(gram_err, w_f, precision=jax.lax.Precision.HIGHEST)
        r_norm = jnp.linalg.norm(r_raw)
        w_norm = jnp.linalg.norm(w_f)
        clip_coef = jnp.minimum(1.0, w_norm / (r_norm + 1e-7))
        r_step = stiefel_penalty * (r_raw * clip_coef)
    else:
        r_step = jnp.zeros_like(w_f)

    delta_w = -lr * (u_f + wd_term + r_step)
    return delta_w.astype(w_2d.dtype), new_m.astype(m_2d.dtype)


def _apply_stiefel_muon_tensor(
    g: jax.Array,
    m: jax.Array,
    w: jax.Array,
    lr: float,
    momentum: float,
    nesterov: bool,
    ns_steps: int,
    weight_decay: float,
    decay_mode: Literal["smooth", "cautious"],
    stiefel_penalty: float,
) -> tuple[jax.Array, jax.Array]:
    """Dispatch 2D or 3D Stiefel-Muon update using the unified Stiefel shape rule."""
    if w.ndim == 2:
        return _apply_stiefel_muon_2d(
            g, m, w, lr, momentum, nesterov, ns_steps, weight_decay, decay_mode, stiefel_penalty
        )
    if w.ndim == 3:
        orig_shape = w.shape
        if orig_shape[1] * orig_shape[2] == orig_shape[0]:
            dw, nm = _apply_stiefel_muon_2d(
                g.reshape(orig_shape[0], -1),
                m.reshape(orig_shape[0], -1),
                w.reshape(orig_shape[0], -1),
                lr,
                momentum,
                nesterov,
                ns_steps,
                weight_decay,
                decay_mode,
                stiefel_penalty,
            )
            return dw.reshape(orig_shape), nm.reshape(orig_shape)
        if orig_shape[0] * orig_shape[1] == orig_shape[2]:
            dw, nm = _apply_stiefel_muon_2d(
                g.reshape(-1, orig_shape[2]),
                m.reshape(-1, orig_shape[2]),
                w.reshape(-1, orig_shape[2]),
                lr,
                momentum,
                nesterov,
                ns_steps,
                weight_decay,
                decay_mode,
                stiefel_penalty,
            )
            return dw.reshape(orig_shape), nm.reshape(orig_shape)

        vmapped_fn = jax.vmap(
            lambda gi, mi, wi: _apply_stiefel_muon_2d(
                gi,
                mi,
                wi,
                lr,
                momentum,
                nesterov,
                ns_steps,
                weight_decay,
                decay_mode,
                stiefel_penalty,
            )
        )
        return vmapped_fn(g, m, w)

    # Fallback for any non-2D/3D leaf
    new_m = momentum * m + (1.0 - momentum) * g
    return (-lr * new_m).astype(w.dtype), new_m.astype(m.dtype)


def scale_by_stiefel_muon(
    learning_rate: float = 0.02,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    weight_decay: float = 0.0,
    decay_mode: Literal["smooth", "cautious"] = "cautious",
    stiefel_penalty: float = 0.05,
) -> optax.GradientTransformation:
    """Optax GradientTransformation implementing Stiefel-Manifold Muon with Continuous Penalty."""

    def init_fn(params: optax.Params) -> StiefelMuonState:
        mom = jax.tree_util.tree_map(
            lambda p: jnp.zeros_like(p) if eqx.is_inexact_array(p) else None,
            params,
        )
        return StiefelMuonState(momentum=mom)

    def update_fn(
        updates: optax.Updates,
        state: StiefelMuonState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, StiefelMuonState]:
        if params is None:
            raise ValueError("scale_by_stiefel_muon requires params to be passed to update().")

        results = jax.tree_util.tree_map(
            lambda g, m, w: (
                _apply_stiefel_muon_tensor(
                    g,
                    m,
                    w,
                    learning_rate,
                    momentum,
                    nesterov,
                    ns_steps,
                    weight_decay,
                    decay_mode,
                    stiefel_penalty,
                )
                if (g is not None and m is not None and eqx.is_inexact_array(w))
                else (g, m)
            ),
            updates,
            state.momentum,
            params,
            is_leaf=lambda x: x is None or eqx.is_array(x),
        )
        new_updates = jax.tree_util.tree_map(
            lambda x, r: r[0] if (x is not None and eqx.is_inexact_array(x)) else x,
            params,
            results,
            is_leaf=lambda x: x is None or eqx.is_array(x),
        )
        new_momentum = jax.tree_util.tree_map(
            lambda x, r: r[1] if (x is not None and eqx.is_inexact_array(x)) else None,
            params,
            results,
            is_leaf=lambda x: x is None or eqx.is_array(x),
        )
        return new_updates, StiefelMuonState(momentum=new_momentum)

    return optax.GradientTransformation(init_fn, update_fn)


def build_quantum_optax_optimizer(
    model: Any,
    muon_lr: float = 0.02,
    adamw_lr: float = 3e-4,
    momentum: float = 0.95,
    weight_decay_muon: float = 0.0,
    weight_decay_adamw: float = 0.01,
    decay_mode: Literal["smooth", "cautious"] = "cautious",
    stiefel_penalty: float = 0.05,
) -> optax.GradientTransformation:
    """Partition an Equinox/Haliax quantum model into Stiefel-Muon (2D/3D Stiefel weights) and AdamW."""
    del model  # Labels are computed dynamically on the parameter pytree via _map_labels

    excluded_tokens = ("seed_embed", "token_embed", "povm_head", "m_vocab", "pos_embed")

    def _map_labels(params_pytree: Any) -> Any:
        def label_leaf(path: tuple[Any, ...], leaf: Any) -> str:
            if not eqx.is_inexact_array(leaf):
                return "adamw"
            path_str = ".".join(str(k).lower() for k in path)
            if (
                leaf.ndim in (2, 3)
                and not any(tok in path_str for tok in excluded_tokens)
                and max(leaf.shape) < 20000
            ):
                return "muon"
            return "adamw"

        return jax.tree_util.tree_map_with_path(label_leaf, params_pytree)

    return optax.multi_transform(
        {
            "muon": scale_by_stiefel_muon(
                learning_rate=muon_lr,
                momentum=momentum,
                nesterov=True,
                ns_steps=5,
                weight_decay=weight_decay_muon,
                decay_mode=decay_mode,
                stiefel_penalty=stiefel_penalty,
            ),
            "adamw": optax.adamw(
                learning_rate=adamw_lr,
                b1=0.9,
                b2=0.95,
                eps=1e-8,
                weight_decay=weight_decay_adamw,
            ),
        },
        _map_labels,
    )
