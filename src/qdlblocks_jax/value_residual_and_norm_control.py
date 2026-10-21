"""JAX Counterparts for Cross-Layer First-Layer Value Residual ($V^{(0)}$) & Cubic Norm Control (Plan 015).

Provides pure-JAX implementations with exact numerical parity against PyTorch
(`qdlblocks.layers.value_residual_and_norm_control`):
1. `compute_cross_layer_v_orthogonality_metrics_jax`
2. `compute_mlp_attn_homogeneity_metrics_jax`
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .functional import apply_unitary_rope

__all__ = [
    "compute_cross_layer_v_orthogonality_metrics_jax",
    "compute_mlp_attn_homogeneity_metrics_jax",
]


def compute_cross_layer_v_orthogonality_metrics_jax(
    v_mixed: jax.Array,
    x_in: jax.Array,
    x_0: jax.Array,
    w_v_layer: jax.Array,
    eps: float = 1e-8,
) -> dict[str, jax.Array]:
    """Compute the orthogonal residual ratio of `v_mixed` relative to Layer $l$'s same-layer span in JAX.

    Args:
        v_mixed: Mixed value array of shape `(B, H, T, D_h)`, `(B, T, D)`, or `(B * T, D)`.
        x_in: Layer-$l$ input hidden states `(B, T, D)` or `(B * T, D)`.
        x_0: Initial Layer-0 token states `(B, T, D)` or `(B * T, D)`.
        w_v_layer: Layer-$l$ value projection matrix `(D, D)`.
        eps: Numerical floor for safe normalization.

    Returns:
        Dictionary of scalar JAX arrays: `orth_residual_ratio`, `parallel_energy_ratio`, `total_v_rms`.
    """
    w_f = jnp.asarray(w_v_layer, dtype=jnp.float32)
    d_model = int(w_f.shape[0])
    x_in_f = jnp.asarray(x_in, dtype=jnp.float32)
    x_in_2d = x_in_f.reshape(-1, d_model)
    x_0_2d = jnp.asarray(x_0, dtype=jnp.float32).reshape(-1, d_model)
    n_tokens = int(x_in_2d.shape[0])

    v_f = jnp.asarray(v_mixed, dtype=jnp.float32)
    if v_f.ndim == 4:
        if x_in_f.ndim == 3 and v_f.shape[1] == x_in_f.shape[1] and v_f.shape[2] != x_in_f.shape[1]:
            v_2d = v_f.reshape(n_tokens, d_model)
        else:
            v_2d = jnp.swapaxes(v_f, 1, 2).reshape(n_tokens, d_model)
    else:
        v_2d = v_f.reshape(n_tokens, d_model)

    u1 = jnp.matmul(x_in_2d, w_f.T)
    u2 = jnp.matmul(x_0_2d, w_f.T)

    u1_norm = jnp.linalg.norm(u1, axis=-1, keepdims=True)
    e1 = u1 / (u1_norm + eps)

    proj_u2_on_e1 = jnp.sum(u2 * e1, axis=-1, keepdims=True) * e1
    u2_orth = u2 - proj_u2_on_e1
    u2_orth_norm = jnp.linalg.norm(u2_orth, axis=-1, keepdims=True)
    e2 = jnp.where(u2_orth_norm > eps, u2_orth / (u2_orth_norm + eps), jnp.zeros_like(u2_orth))

    v_par = jnp.sum(v_2d * e1, axis=-1, keepdims=True) * e1 + jnp.sum(v_2d * e2, axis=-1, keepdims=True) * e2
    v_orth = v_2d - v_par

    v_fro = jnp.linalg.norm(v_2d)
    v_orth_fro = jnp.linalg.norm(v_orth)
    v_par_fro = jnp.linalg.norm(v_par)

    orth_residual_ratio = v_orth_fro / (v_fro + eps)
    parallel_energy_ratio = (v_par_fro * v_par_fro) / (v_fro * v_fro + eps)
    total_v_rms = jnp.sqrt(jnp.mean(v_2d * v_2d) + eps)

    return {
        "orth_residual_ratio": orth_residual_ratio,
        "parallel_energy_ratio": parallel_energy_ratio,
        "total_v_rms": total_v_rms,
    }


def _rms_norm_jax(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(ms + eps)


def compute_mlp_attn_homogeneity_metrics_jax(
    block_or_params: Any,
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    scale_factor: float = 4.0,
    num_heads: int | None = None,
    eps: float = 1e-6,
) -> dict[str, jax.Array]:
    """Evaluate attention ($c^2$) vs. $\\text{ReLU}^2$ MLP ($c^3$) homogeneity under weight scaling $c$ in JAX.

    Args:
        block_or_params: Either a parameter dictionary with keys `("w_q", "w_k", "w_v", "w_o", "w_1", "w_2")`
            (or `("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")`), or a `ClassicalModdedBlock` module.
        x: Input hidden state array of shape `(B, T, D)`.
        cos: Unitary RoPE cosine table `(1, 1, T, D_h // 2)`.
        sin: Unitary RoPE sine table `(1, 1, T, D_h // 2)`.
        scale_factor: Scalar multiplier $c > 0$ applied to all 2D block weights (default: `4.0`).
        num_heads: Optional number of attention heads (inferred from `block_or_params` or `cos` if omitted).
        eps: RMSNorm numerical stability floor.

    Returns:
        Dictionary of scalar JAX arrays: `attn_scale_ratio`, `mlp_scale_ratio`,
        `dominance_growth_ratio`, `rho_mlp_attn_base`, and `rho_mlp_attn_scaled`.
    """
    x_f = jnp.asarray(x, dtype=jnp.float32)
    cos_f = jnp.asarray(cos, dtype=jnp.float32)
    sin_f = jnp.asarray(sin, dtype=jnp.float32)
    b_sz, t_sz, d_model = x_f.shape

    if isinstance(block_or_params, dict):
        w_q = jnp.asarray(block_or_params.get("w_q", block_or_params.get("q_proj")), dtype=jnp.float32)
        w_k = jnp.asarray(block_or_params.get("w_k", block_or_params.get("k_proj")), dtype=jnp.float32)
        w_v = jnp.asarray(block_or_params.get("w_v", block_or_params.get("v_proj")), dtype=jnp.float32)
        w_o = jnp.asarray(block_or_params.get("w_o", block_or_params.get("out_proj")), dtype=jnp.float32)
        w_1 = jnp.asarray(block_or_params.get("w_1", block_or_params.get("fc1")), dtype=jnp.float32)
        w_2 = jnp.asarray(block_or_params.get("w_2", block_or_params.get("fc2")), dtype=jnp.float32)
        if num_heads is None:
            num_heads = int(block_or_params.get("num_heads", d_model // (2 * int(cos_f.shape[-1]))))
        eps = float(block_or_params.get("eps", eps))
    else:
        w_q = jnp.asarray(block_or_params.q_proj.weight.detach().cpu().numpy(), dtype=jnp.float32)
        w_k = jnp.asarray(block_or_params.k_proj.weight.detach().cpu().numpy(), dtype=jnp.float32)
        w_v = jnp.asarray(block_or_params.v_proj.weight.detach().cpu().numpy(), dtype=jnp.float32)
        w_o = jnp.asarray(block_or_params.out_proj.weight.detach().cpu().numpy(), dtype=jnp.float32)
        w_1 = jnp.asarray(block_or_params.fc1.weight.detach().cpu().numpy(), dtype=jnp.float32)
        w_2 = jnp.asarray(block_or_params.fc2.weight.detach().cpu().numpy(), dtype=jnp.float32)
        if num_heads is None:
            num_heads = int(getattr(block_or_params, "num_heads", d_model // (2 * int(cos_f.shape[-1]))))
        eps = float(getattr(block_or_params, "eps", eps))

    assert num_heads is not None
    head_dim = d_model // num_heads
    h = _rms_norm_jax(x_f, eps=eps)
    causal_mask = jnp.tril(jnp.ones((t_sz, t_sz), dtype=jnp.bool_))[None, None, :, :]

    def _eval_at_scale(scale: float) -> tuple[jax.Array, jax.Array, jax.Array]:
        q = jnp.matmul(h, (scale * w_q).T).reshape(b_sz, t_sz, num_heads, head_dim).swapaxes(1, 2)
        k = jnp.matmul(h, (scale * w_k).T).reshape(b_sz, t_sz, num_heads, head_dim).swapaxes(1, 2)
        v = jnp.matmul(h, (scale * w_v).T).reshape(b_sz, t_sz, num_heads, head_dim).swapaxes(1, 2)

        q = apply_unitary_rope(q, cos_f, sin_f)
        k = apply_unitary_rope(k, cos_f, sin_f)
        q = _rms_norm_jax(q, eps=1e-12)
        k = _rms_norm_jax(k, eps=1e-12)

        scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / jnp.sqrt(float(head_dim))
        scores = jnp.where(causal_mask, scores, -1e30)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        sdpa = jnp.matmul(attn_weights, v)
        sdpa_flat = jnp.swapaxes(sdpa, 1, 2).reshape(b_sz, t_sz, d_model)
        attn_out = jnp.matmul(sdpa_flat, (scale * w_o).T)

        r_mlp = jax.nn.relu(jnp.matmul(h, (scale * w_1).T))
        mlp_out = jnp.matmul(jnp.square(r_mlp), (scale * w_2).T)

        attn_norm = jnp.linalg.norm(attn_out)
        mlp_norm = jnp.linalg.norm(mlp_out)
        rho = mlp_norm / jnp.maximum(attn_norm, 1e-12)
        return attn_norm, mlp_norm, rho

    attn_1, mlp_1, rho_1 = _eval_at_scale(1.0)
    attn_c, mlp_c, rho_c = _eval_at_scale(float(scale_factor))

    return {
        "attn_scale_ratio": attn_c / jnp.maximum(attn_1, 1e-12),
        "mlp_scale_ratio": mlp_c / jnp.maximum(mlp_1, 1e-12),
        "dominance_growth_ratio": rho_c / jnp.maximum(rho_1, 1e-12),
        "rho_mlp_attn_base": rho_1,
        "rho_mlp_attn_scaled": rho_c,
    }
