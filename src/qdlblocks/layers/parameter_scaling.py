"""Multi-Axis Parameter Scaling Utilities for QuantumBornGPT (Plan 012).

Provides:
- `default_overcomplete_target_iso_scale`: Target isometry scale for overcomplete
  two-beam tight frames with arbitrary frame expansion `rho >= 1.0`.
- `compute_depth_residual_scale`: Depth-scaled spherical residual factor
  `alpha_res(L) = sqrt(base_layers / L)` stabilizing geodesic steps across deep circuits.
- `compute_parameter_scaling_breakdown`: Fine-grained parameter accounting across
  dense backbone (attention, MLP, U-Net skips), sparse vocabulary lift, and POVM readout.
- `compute_depth_fidelity_and_geodesic_profile`: End-to-end state fidelity, per-layer
  geodesic step lengths on `S^{d-1}`, and across-layer gradient norm conditioning (CV).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import l2_normalize

if TYPE_CHECKING:
    from ..models.born_gpt import QuantumBornGPT

__all__ = [
    "default_overcomplete_target_iso_scale",
    "compute_depth_residual_scale",
    "compute_parameter_scaling_breakdown",
    "compute_depth_fidelity_and_geodesic_profile",
]


def default_overcomplete_target_iso_scale(frame_expansion: float) -> float:
    """Compute target Gram scale `1.0 + 0.25 * (rho - 1.0)` for frame expansion `rho`."""
    rho = float(frame_expansion)
    if rho <= 1.0:
        return 1.0
    return float(1.0 + 0.25 * (rho - 1.0))


def compute_depth_residual_scale(
    num_layers: int,
    base_layers: int = 6,
    enabled: bool = True,
) -> float:
    """Compute depth-scaled spherical residual weight `sqrt(base_layers / num_layers)`.

    At `num_layers <= base_layers` (or when `enabled=False`), returns `1.0` for 100%
    backward compatibility with 6-layer Wave 1-4 architectures.
    """
    if (not enabled) or int(num_layers) <= int(base_layers):
        return 1.0
    return float(math.sqrt(float(base_layers) / float(num_layers)))


def compute_parameter_scaling_breakdown(model: nn.Module) -> dict[str, int | float]:
    """Decompose `model.named_parameters()` into dense backbone vs. sparse vocab/readout shares."""
    total_params = 0
    backbone_params = 0
    attention_params = 0
    mlp_params = 0
    unet_params = 0
    vocab_lift_params = 0
    readout_params = 0

    for name, param in model.named_parameters():
        n_el = int(param.numel())
        total_params += n_el

        if name.startswith("blocks."):
            backbone_params += n_el
            if ".attn." in name:
                attention_params += n_el
            elif any(k in name for k in (".two_beam_mlp.", ".moe.", ".ff1.", ".ff2.")):
                mlp_params += n_el
        elif name.startswith(("unet_gates", "unet_scales")):
            backbone_params += n_el
            unet_params += n_el
        elif name.startswith(("hilbert_lift.", "wte.", "W_lift", "smear_embed_gate")):
            vocab_lift_params += n_el
        elif name.startswith(
            (
                "povm_readout",
                "povm_head",
                "readout_rot",
                "povm_head_seed_delta",
                "seed_adapter",
                "tied_inv_temp",
                "mtp_rotations",
                "mtp_inv_temps",
                "kraus_readout",
            )
        ):
            readout_params += n_el

    backbone_fraction = float(backbone_params) / float(max(1, total_params))
    return {
        "total_params": total_params,
        "backbone_params": backbone_params,
        "attention_params": attention_params,
        "mlp_params": mlp_params,
        "unet_params": unet_params,
        "vocab_lift_params": vocab_lift_params,
        "readout_params": readout_params,
        "backbone_fraction": backbone_fraction,
    }


def compute_depth_fidelity_and_geodesic_profile(
    model: QuantumBornGPT,
    idx: torch.Tensor,
    targets: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> dict[str, float | list[float]]:
    """Measure end-to-end state fidelity, per-layer geodesic step angles, and gradient norm CV."""
    if targets is not None:
        model.zero_grad(set_to_none=True)
        _, loss, layer_states = model(idx, targets=targets, return_layer_states=True)
        assert loss is not None
        loss.backward()
    else:
        with torch.no_grad():
            _, _, layer_states = model(idx, targets=None, return_layer_states=True)

    with torch.no_grad():
        psi_0 = getattr(model, "_last_psi_0", None)
        if psi_0 is None:
            psi_0 = model.hilbert_lift(idx)
            if model.use_bigram_smear and idx.size(1) > 1 and model.smear_embed_gate is not None:
                psi_prev = F.pad(psi_0[:, :-1, :], (0, 0, 1, 0))
                eta = (0.06 * torch.tanh(model.smear_embed_gate)).to(psi_0.dtype)
                psi_0 = l2_normalize(psi_0 + eta * psi_prev, dim=-1, eps=model.eps)

        # Note: Use eps=1e-12 for float32 unit-sphere normalization so that PyTorch's
        # additive rsqrt(norm_sq + eps) matches JAX's maximum(norm_sq, eps^2) to machine
        # precision (~1e-7), while reserving `eps` (default 1e-6) strictly for clamping
        # the arccos argument away from the derivative singularity at |<psi_a | psi_b>| -> 1.
        p0 = l2_normalize(psi_0.detach().float(), dim=-1, eps=1e-12)
        p_final = l2_normalize(layer_states[-1].detach().float(), dim=-1, eps=1e-12)
        ov_end = torch.sum(p0 * p_final, dim=-1)
        end_to_end_fidelity = float(torch.mean(ov_end * ov_end).item())

        layer_geodesic_steps_rad: list[float] = []
        prev_s = p0
        for st in layer_states:
            curr_s = l2_normalize(st.detach().float(), dim=-1, eps=1e-12)
            abs_ov = torch.clamp(torch.abs(torch.sum(prev_s * curr_s, dim=-1)), min=0.0, max=1.0 - eps)
            theta_l = float(torch.mean(torch.acos(abs_ov)).item())
            layer_geodesic_steps_rad.append(theta_l)
            prev_s = curr_s

        cumulative_geodesic = float(sum(layer_geodesic_steps_rad))
        mean_geodesic = float(cumulative_geodesic / max(1, len(layer_geodesic_steps_rad)))

        layer_grad_norms: list[float] = []
        grad_norm_cv = 0.0
        if targets is not None:
            for block in model.blocks:
                sq_sum = 0.0
                for p in block.parameters():
                    if p.grad is not None:
                        sq_sum += float(p.grad.detach().float().pow(2).sum().item())
                layer_grad_norms.append(math.sqrt(sq_sum))
            if layer_grad_norms:
                arr = np.asarray(layer_grad_norms, dtype=np.float64)
                mean_gn = float(np.mean(arr))
                std_gn = float(np.std(arr))
                grad_norm_cv = float(std_gn / max(mean_gn, 1e-12))

    return {
        "end_to_end_fidelity": end_to_end_fidelity,
        "layer_geodesic_steps_rad": layer_geodesic_steps_rad,
        "cumulative_geodesic_length_rad": cumulative_geodesic,
        "mean_geodesic_step_rad": mean_geodesic,
        "grad_norm_cv": grad_norm_cv,
        "layer_grad_norms": layer_grad_norms,
    }
