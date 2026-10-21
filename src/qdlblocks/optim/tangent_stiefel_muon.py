"""Tangent-Space Stiefel Muon Optimizer with Riemannian Tangent Projection and Quadratic Retraction."""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

import torch

from ..functional import zeropower_via_newtonschulz5

__all__ = [
    "project_stiefel_tangent",
    "stiefel_quadratic_retract",
    "TangentStiefelMuon",
]


def project_stiefel_tangent(w: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Project an ambient 2D direction `u` onto the Riemannian Stiefel tangent space T_w St(r, c).

    For w in R^{r x c}:
      - If r >= c: Pi_{T_w}(u) = u - 1/2 w (w^T u + u^T w)
      - If r < c:  Pi_{T_w}(u) = u - 1/2 (u w^T + w u^T) w

    Args:
        w: 2D Stiefel matrix tensor of shape `(r, c)`.
        u: 2D ambient direction tensor of shape `(r, c)`.

    Returns:
        Tangent vector tensor in `T_w St(r, c)` with the same shape and dtype as `u`.
    """
    if w.ndim != 2:
        raise ValueError(f"project_stiefel_tangent expects a 2D tensor w, got {w.ndim}D")
    if u.shape != w.shape:
        raise ValueError(
            f"project_stiefel_tangent expects u.shape == w.shape, got {tuple(u.shape)} vs {tuple(w.shape)}"
        )
    wf = w.float()
    uf = u.float()
    r, c = wf.shape
    if r >= c:
        wt_u = torch.matmul(wf.transpose(-2, -1), uf)
        sym = 0.5 * (wt_u + wt_u.transpose(-2, -1))
        out = uf - torch.matmul(wf, sym)
    else:
        u_wt = torch.matmul(uf, wf.transpose(-2, -1))
        sym = 0.5 * (u_wt + u_wt.transpose(-2, -1))
        out = uf - torch.matmul(sym, wf)
    return out.to(u.dtype)


def stiefel_quadratic_retract(w: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Apply a first-order quadratic Stiefel retraction step to cancel O(eta^2) normal curvature drift.

    For w in R^{r x c}:
      - If r >= c: R(w) = w - alpha * w (w^T w - I_c)
      - If r < c:  R(w) = w - alpha * (w w^T - I_r) w

    Args:
        w: 2D matrix tensor of shape `(r, c)`.
        alpha: Retraction step coefficient (default: 0.5 for exact quadratic cancellation).

    Returns:
        Retracted 2D tensor with the same shape and dtype as `w`.
    """
    if w.ndim != 2:
        raise ValueError(f"stiefel_quadratic_retract expects a 2D tensor w, got {w.ndim}D")
    if alpha < 0.0:
        raise ValueError(f"alpha must be >= 0.0, got {alpha}")
    wf = w.float()
    r, c = wf.shape
    if r >= c:
        gram = torch.matmul(wf.transpose(-2, -1), wf)
        eye = torch.eye(c, device=w.device, dtype=torch.float32)
        out = wf - alpha * torch.matmul(wf, gram - eye)
    else:
        gram = torch.matmul(wf, wf.transpose(-2, -1))
        eye = torch.eye(r, device=w.device, dtype=torch.float32)
        out = wf - alpha * torch.matmul(gram - eye, wf)
    return out.to(w.dtype)


class TangentStiefelMuon(torch.optim.Optimizer):
    """Tangent-Space Stiefel Muon Optimizer with Double Tangent Projection and Quadratic Retraction.

    Eliminates first-order O(eta) off-manifold drift during constant high-LR WSD plateau training
    by projecting momentum directions onto the Riemannian tangent space T_W St(r, c) both before
    and after the 5-step Newton-Schulz orthogonal polar iteration, followed by a quadratic
    Stiefel retraction step to cancel second-order O(eta^2) curvature error.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        retract_alpha: float = 0.5,
        retract_every: int = 1,
        tangent_project_before_ns: bool = True,
        tangent_project_after_ns: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Invalid momentum: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
        if retract_alpha < 0.0:
            raise ValueError(f"Invalid retract_alpha: {retract_alpha}")
        if retract_every < 1:
            raise ValueError(f"Invalid retract_every: {retract_every}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            retract_alpha=retract_alpha,
            retract_every=retract_every,
            tangent_project_before_ns=tangent_project_before_ns,
            tangent_project_after_ns=tangent_project_after_ns,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.data)
                    state["step"] = 0
                state["step"] += 1
                buf: torch.Tensor = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                g_dir = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf

                if p.ndim == 2:
                    if group["tangent_project_before_ns"]:
                        g_dir = project_stiefel_tangent(p.data, g_dir)
                    u = zeropower_via_newtonschulz5(g_dir, steps=group["ns_steps"])
                    aspect_scale = math.sqrt(max(1.0, p.size(0) / p.size(1)))
                    u = u * aspect_scale
                    if group["tangent_project_after_ns"]:
                        u = project_stiefel_tangent(p.data, u)
                    p.data.add_(u.to(p.dtype), alpha=-group["lr"])
                    if group["retract_alpha"] > 0.0 and (state["step"] % group["retract_every"] == 0):
                        p.data.copy_(stiefel_quadratic_retract(p.data, alpha=group["retract_alpha"]))
                else:
                    p.data.add_(g_dir.to(p.dtype), alpha=-group["lr"])

        return loss
