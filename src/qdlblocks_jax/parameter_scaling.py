"""JAX Parameter Scaling & Depth Geodesic/Fidelity Diagnostics (Plan 012)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from .functional import l2_normalize

__all__ = [
    "default_overcomplete_target_iso_scale_jax",
    "compute_depth_residual_scale_jax",
    "compute_depth_fidelity_and_geodesic_profile_jax",
]


def default_overcomplete_target_iso_scale_jax(frame_expansion: float) -> float:
    """Compute target Gram scale `1.0 + 0.25 * (rho - 1.0)` for frame expansion `rho`."""
    rho = float(frame_expansion)
    if rho <= 1.0:
        return 1.0
    return float(1.0 + 0.25 * (rho - 1.0))


def compute_depth_residual_scale_jax(
    num_layers: int,
    base_layers: int = 6,
    enabled: bool = True,
) -> float:
    """Compute depth-scaled spherical residual weight `sqrt(base_layers / num_layers)`."""
    if (not enabled) or int(num_layers) <= int(base_layers):
        return 1.0
    return float(math.sqrt(float(base_layers) / float(num_layers)))


def compute_depth_fidelity_and_geodesic_profile_jax(
    psi_0: jax.Array,
    layer_states: list[jax.Array],
    eps: float = 1e-6,
) -> dict[str, float | list[float]]:
    """Measure end-to-end state fidelity and per-layer geodesic step angles on `S^{d-1}` in JAX."""
    if not layer_states:
        raise ValueError("layer_states must be a non-empty list of layer state arrays")

    # Keep unit-sphere normalization at eps=1e-12 in float32 for exact parity with
    # PyTorch rsqrt(norm_sq + 1e-12), while using `eps` (1e-6) only for arccos clamping.
    p0 = l2_normalize(psi_0.astype(jnp.float32), axis=-1, eps=1e-12)
    p_final = l2_normalize(layer_states[-1].astype(jnp.float32), axis=-1, eps=1e-12)
    ov_end = jnp.sum(p0 * p_final, axis=-1)
    end_to_end_fidelity = float(jnp.mean(ov_end * ov_end))

    layer_geodesic_steps_rad: list[float] = []
    prev_s = p0
    for st in layer_states:
        curr_s = l2_normalize(st.astype(jnp.float32), axis=-1, eps=1e-12)
        abs_ov = jnp.clip(jnp.abs(jnp.sum(prev_s * curr_s, axis=-1)), 0.0, 1.0 - eps)
        theta_l = float(jnp.mean(jnp.arccos(abs_ov)))
        layer_geodesic_steps_rad.append(theta_l)
        prev_s = curr_s

    cumulative_geodesic = float(sum(layer_geodesic_steps_rad))
    mean_geodesic = float(cumulative_geodesic / max(1, len(layer_geodesic_steps_rad)))
    return {
        "end_to_end_fidelity": end_to_end_fidelity,
        "layer_geodesic_steps_rad": layer_geodesic_steps_rad,
        "cumulative_geodesic_length_rad": cumulative_geodesic,
        "mean_geodesic_step_rad": mean_geodesic,
    }
