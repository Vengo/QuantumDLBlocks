"""Stiefel-Manifold Muon Optimizers with Smooth Decay, Cautious Weight Decay, and Riemannian Retraction."""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Literal

import torch
import torch.nn as nn

from ..functional import zeropower_via_newtonschulz5


DecayMode = Literal["smooth", "cautious", "none"]


class StiefelMuon(torch.optim.Optimizer):
    """Stiefel-Manifold Muon Optimizer for 2D Isometric & Unitary Weight Matrices.

    Computes orthogonalized momentum updates on the Stiefel manifold via quintic
    Newton-Schulz polar iterations (`zeropower_via_newtonschulz5`), and supports
    three manifold regularization mechanisms to prevent singular value / isometry drift
    without the catastrophic loss spikes of hard periodic polar snapping:

    1. **Smooth Weight Decay (`decay_mode="smooth"`)**:
       Applies continuous multiplicative weight decay `W <- (1 - lr * wd) * W` before
       stepping along the polar direction.
    2. **Cautious Weight Decay (`decay_mode="cautious"`)**:
       Applies weight decay only to coordinates where the parameter sign aligns with
       the polar update direction `(W * U > 0)`.
    3. **Continuous Riemannian Stiefel Penalty (`stiefel_penalty > 0`)**:
       Applies a soft gradient step of the Stiefel manifold potential `1/4 ||W^T W - I||_F^2`:
           `W <- W - stiefel_penalty * W (W^T W - I)`  (if rows >= cols)
           `W <- W - stiefel_penalty * (W W^T - I) W`  (if rows < cols)
       This continuously drives all singular values toward 1.0 (keeping Mean Isometry Error < 0.10)
       smoothly at every step, avoiding representational shock.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        decay_mode: DecayMode = "smooth",
        stiefel_penalty: float = 0.0,
        ns_steps: int = 5,
        aspect_scale: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Invalid momentum: {momentum}")
        if decay_mode not in ("smooth", "cautious", "none"):
            raise ValueError(f"Invalid decay_mode: {decay_mode!r}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            decay_mode=decay_mode,
            stiefel_penalty=stiefel_penalty,
            ns_steps=ns_steps,
            aspect_scale=aspect_scale,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            momentum: float = group["momentum"]
            wd: float = group["weight_decay"]
            decay_mode: DecayMode = group["decay_mode"]
            stiefel_penalty: float = group["stiefel_penalty"]
            ns_steps: int = group["ns_steps"]
            aspect_scale: bool = group["aspect_scale"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(
                        f"StiefelMuon only supports 2D weight matrices, got shape {tuple(p.shape)}"
                    )

                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf: torch.Tensor = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                u = zeropower_via_newtonschulz5(buf, steps=ns_steps)

                if aspect_scale:
                    scale = math.sqrt(max(1.0, p.shape[0] / p.shape[1]))
                    update = u * scale
                else:
                    update = u

                # 1. Weight decay (smooth or cautious)
                if wd > 0.0:
                    if decay_mode == "smooth":
                        p.mul_(1.0 - lr * wd)
                    elif decay_mode == "cautious":
                        cwd_mask = (p * update) > 0.0
                        p.add_(p * cwd_mask, alpha=-lr * wd)

                # 2. Continuous Riemannian Stiefel retraction penalty:
                # Pulls singular values smoothly toward 1.0 along grad(1/4 ||W^T W - I||_F^2),
                # normalized by (1 + ||W^T W - I||_F) for unconditional numerical stability.
                if stiefel_penalty > 0.0:
                    W_f = p.float()
                    if W_f.size(0) >= W_f.size(1):
                        gram = W_f.t() @ W_f
                        ident = torch.eye(gram.size(0), device=p.device, dtype=W_f.dtype)
                        diff = gram - ident
                        denom = 1.0 + (torch.norm(diff, p="fro") / math.sqrt(diff.size(0)))
                        stiefel_grad = (W_f @ (diff / denom)).type_as(p)
                    else:
                        gram = W_f @ W_f.t()
                        ident = torch.eye(gram.size(0), device=p.device, dtype=W_f.dtype)
                        diff = gram - ident
                        denom = 1.0 + (torch.norm(diff, p="fro") / math.sqrt(diff.size(0)))
                        stiefel_grad = ((diff / denom) @ W_f).type_as(p)
                    p.add_(stiefel_grad, alpha=-stiefel_penalty)

                # 3. Apply Newton-Schulz polar gradient step
                p.add_(update, alpha=-lr)

        return loss


@torch.no_grad()
def polar_snap_weights(
    model: nn.Module,
    steps: int = 5,
    exclude_names: tuple[str, ...] = ("wte", "povm_head"),
) -> None:
    """Hard polar retraction snapping 2D weight matrices directly onto the Stiefel manifold.

    Warning:
        Executing hard periodic polar snaps mid-training abruptly resets all singular
        values to 1.0 simultaneously, which can cause severe validation loss spikes
        (representational shock). Prefer continuous regularization via
        `StiefelMuon(..., stiefel_penalty=0.005)` during training.
    """
    for name, p in model.named_parameters():
        if p.ndim == 2 and not any(excl in name for excl in exclude_names):
            p.data.copy_(zeropower_via_newtonschulz5(p.data, steps=steps))


def split_muon_adamw_params(
    model: nn.Module,
    adamw_name_patterns: tuple[str, ...] = ("wte", "povm_head", "embed"),
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition model parameters into 2D Stiefel isometries (for Muon) and other params (for AdamW).

    Args:
        model: PyTorch model to partition.
        adamw_name_patterns: Substrings identifying 2D lookup tables or readout frames
            that should be optimized by AdamW rather than Muon.

    Returns:
        Tuple `(muon_params, adamw_params)`.
    """
    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and not any(pat in name for pat in adamw_name_patterns):
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params
