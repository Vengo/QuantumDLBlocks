"""
Input-Driven Channel Gating, Gated Exclusive Self-Attention (XSA), Multi-Way Dynamic
Dense (MUDD) Residual Skips, and Token-1 Forward Smear Gates (Plan 021 Category A).

Implements Section 4.3.1 of EXPERIMENT_PLAN_021.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import l2_normalize


class InputChannelGate(nn.Module):
    """Per-head input-driven gate `2.0 * sigmoid(x[..., :c] @ W^T + b)` reading leading `gate_channels`."""

    def __init__(
        self,
        num_heads: int = 16,
        gate_channels: int = 6,
        head_first: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.gate_channels = max(1, int(gate_channels))
        self.head_first = bool(head_first)
        self.weight = nn.Parameter(torch.zeros(self.num_heads, self.gate_channels))
        self.bias = nn.Parameter(torch.zeros(self.num_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gate tensor of shape `(B, H, T, 1)` (if `head_first=True`) or `(B, T, H, 1)`."""
        c = min(self.gate_channels, x.size(-1))
        x_slice = x[..., :c].to(self.weight.dtype)
        w_slice = self.weight[:, :c]
        g = 2.0 * torch.sigmoid(F.linear(x_slice, w_slice, self.bias))  # (B, T, H)
        g = g.to(x.dtype)
        if self.head_first:
            return g.transpose(1, 2).unsqueeze(-1)
        return g.unsqueeze(-1)


def apply_gated_xsa(
    y: torch.Tensor,
    v: torch.Tensor,
    xsa_alpha: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Subtract gated self-value projection from attention head outputs `y` `(B, H, T, D_h)`."""
    if xsa_alpha.ndim == 1:
        gate = torch.tanh(xsa_alpha).view(1, -1, 1, 1).to(y.dtype)
    else:
        gate = torch.tanh(xsa_alpha).to(y.dtype)
    y_f = y.float()
    v_f = v.float()
    v_norm_sq = v_f.square().sum(dim=-1, keepdim=True) + eps
    dot = (y_f * v_f).sum(dim=-1, keepdim=True)
    proj = (dot / v_norm_sq) * v_f
    return (y_f - gate.float() * proj).to(y.dtype)


class MUDDDynamicSkip(nn.Module):
    """Multi-way Dynamic Dense Connection (`MUDD`, arXiv:2502.12170)."""

    def __init__(
        self,
        embed_dim: int,
        num_sources: int = 1,
        hidden_dim: int = 64,
        scale: float = 0.1,
        spherical: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_sources = max(1, int(num_sources))
        self.hidden_dim = max(1, int(hidden_dim))
        self.scale = float(scale)
        self.spherical = bool(spherical)

        self.w1 = nn.Linear(self.embed_dim, self.hidden_dim, bias=False)
        nn.init.kaiming_normal_(self.w1.weight, nonlinearity="relu")
        self.w2 = nn.Linear(self.hidden_dim, 1 + self.num_sources, bias=True)
        nn.init.zeros_(self.w2.weight)
        with torch.no_grad():
            self.w2.bias.zero_()
            self.w2.bias[0] = 1.0 / self.scale

    def forward(
        self,
        x: torch.Tensor,
        sources: list[torch.Tensor] | tuple[torch.Tensor, ...],
        spherical: bool | None = None,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        use_sph = self.spherical if spherical is None else bool(spherical)
        if not sources:
            return l2_normalize(x, dim=-1, eps=eps) if use_sph else x
        k = min(len(sources), self.num_sources)
        x_norm = F.rms_norm(x.float(), (self.embed_dim,), eps=max(eps, 1e-6)).to(self.w1.weight.dtype)
        h = F.gelu(self.w1(x_norm))
        base_mu = (self.w2.bias * self.scale) if self.w2.bias is not None else 0.0
        dyn_mu = torch.tanh(F.linear(h, self.w2.weight)) * self.scale
        mu = (base_mu + dyn_mu).to(x.dtype)  # (B, T, 1 + num_sources)
        y = mu[..., 0:1] * x
        for i in range(k):
            y = y + mu[..., i + 1 : i + 2] * sources[i].to(x.dtype)
        if use_sph:
            return l2_normalize(y, dim=-1, eps=eps)
        return y


class ForwardSmearGate(nn.Module):
    """Shifts position `t-1` state into position `t` via a learned low-rank gate at layer 0."""

    def __init__(
        self,
        embed_dim: int,
        gate_channels: int = 12,
        gate_dim: int | None = None,
        spherical: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        eff_channels = int(gate_dim) if gate_dim is not None else int(gate_channels)
        self.gate_channels = min(eff_channels, self.embed_dim)
        self.spherical = bool(spherical)
        self.gate_proj = nn.Linear(self.gate_channels, 1, bias=False)
        nn.init.zeros_(self.gate_proj.weight)
        self.smear_lambda = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        spherical: bool | None = None,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        use_sph = self.spherical if spherical is None else bool(spherical)
        if x.size(1) <= 1:
            return l2_normalize(x, dim=-1, eps=eps) if use_sph else x
        c = min(self.gate_channels, x.size(-1))
        x_slice = x[:, 1:, :c].to(self.gate_proj.weight.dtype)
        g_t = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.gate_proj(x_slice)).to(x.dtype)
        x_tail = x[:, 1:, :] + g_t * x[:, :-1, :]
        x_tilde = torch.cat([x[:, :1, :], x_tail], dim=1)
        if use_sph:
            return l2_normalize(x_tilde, dim=-1, eps=eps)
        return x_tilde


__all__ = [
    "InputChannelGate",
    "apply_gated_xsa",
    "MUDDDynamicSkip",
    "ForwardSmearGate",
]
