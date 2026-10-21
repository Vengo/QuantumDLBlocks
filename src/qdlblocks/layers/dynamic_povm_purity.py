"""Contextual Bloch-Ball Purity Readout Gate & Logit-Norm Variation Diagnostics (Plan 013).

Resolves the Constant-Logit-Norm Heteroscedastic Sharpness Bottleneck on S^{D-1} by
allowing each token state |psi_{b,t}> to reside inside the Bloch ball with a context-dependent
purity radius r_{b,t} in [r_min, r_max], producing per-token inverse temperature
beta_{b,t} = clamp(beta_0 * r(psi_{b,t}), min=1.0, max=beta_max).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import l2_normalize


@torch.no_grad()
def compute_logit_norm_variation_metrics(
    logits: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Compute per-token logit L2-norm statistics, coefficient of variation, and output entropy.

    Under standard global-beta spherical readout (ell_{b,t} = beta * M |phi_{b,t}>) where
    M forms an approximately tight frame on S^{D-1}, ||ell_{b,t}||_2 is nearly constant across
    all token positions (CV(||ell||_2) -> 0). Contextual Bloch-purity gating restores
    input-dependent logit energy modulation across deterministic vs. branching contexts.

    Args:
        logits: Logit tensor of shape `(..., V)`.
        eps: Numerical stability floor.

    Returns:
        Dictionary with `logit_norm_mean`, `logit_norm_std`, `logit_norm_cv`,
        `mean_entropy_nats`, and `mean_max_prob`.
    """
    if logits.ndim < 1 or logits.shape[-1] < 1 or logits.numel() == 0:
        raise ValueError(f"Expected non-empty logits tensor (..., V), got shape {tuple(logits.shape)}.")
    v_dim = logits.shape[-1]
    flat = logits.detach().reshape(-1, v_dim).float()
    norms = torch.linalg.vector_norm(flat, ord=2, dim=-1)
    n_mean = norms.mean()
    n_std = norms.std(unbiased=False)
    n_cv = n_std / (n_mean + eps)

    probs = F.softmax(flat, dim=-1)
    entropies = -torch.sum(probs * torch.log(probs + eps), dim=-1)
    max_probs = probs.max(dim=-1).values

    return {
        "logit_norm_mean": float(n_mean.item()),
        "logit_norm_std": float(n_std.item()),
        "logit_norm_cv": float(n_cv.item()),
        "mean_entropy_nats": float(entropies.mean().item()),
        "mean_max_prob": float(max_probs.mean().item()),
    }


class ContextualBlochPurityGate(nn.Module):
    """Contextual Bloch-Ball Purity Gate predicting per-token POVM inverse temperature beta_{b,t}.

    Parameterizes a purity radius r(psi_{b,t}) in [r_min, r_max] from the final unit-normalized
    Hilbert ket psi_{b,t} in S^{d-1}:
        u(psi) = fc2(SiLU(fc1(N(psi))))
        r(psi) = clamp(softplus(u(psi)) / softplus(init_bias), min=r_min, max=r_max)
        beta(psi) = clamp(base_inv_temp * r(psi), min=1.0, max=max_beta)

    Because `fc2.weight` is zero-initialized and `fc2.bias` is initialized to `init_bias`,
    at step 0 `u(psi) == init_bias` and `r(psi) == 1.0` identically for all tokens,
    preserving exact initial forward-pass equivalence with scalar-beta `QuantumBornGPT`.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 64,
        r_min: float = 0.25,
        r_max: float = 4.0,
        init_bias: float = 1.0,
        max_beta: float | None = None,
        eps: float = 1e-8,
        *,
        use_two_beam_kboson: bool = False,
        use_two_beam_kboson_readout: bool = False,
    ) -> None:
        super().__init__()
        self.use_two_beam_kboson = bool(use_two_beam_kboson or use_two_beam_kboson_readout)
        if max_beta is None:
            max_beta = 45.0 if self.use_two_beam_kboson else 160.0
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {embed_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if not (0.0 < r_min < r_max):
            raise ValueError(f"Expected 0 < r_min < r_max, got r_min={r_min}, r_max={r_max}")
        if max_beta < 1.0:
            raise ValueError(f"max_beta must be >= 1.0, got {max_beta}")

        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.init_bias = float(init_bias)
        self.max_beta = float(max_beta)
        self.eps = float(eps)
        self._s0 = float(F.softplus(torch.tensor(float(init_bias), dtype=torch.float32)).item())

        self.fc1 = nn.Linear(self.embed_dim, self.hidden_dim, bias=True)
        self.fc2 = nn.Linear(self.hidden_dim, 1, bias=True)

        nn.init.orthogonal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, self.init_bias)

    def purity_radius(self, psi: torch.Tensor) -> torch.Tensor:
        """Compute per-token Bloch purity radius `r(psi)` of shape `(..., 1)` in `float32`."""
        psi_norm = l2_normalize(psi, dim=-1, eps=self.eps)
        h = F.silu(self.fc1(psi_norm.to(self.fc1.weight.dtype)))
        u = self.fc2(h).float()
        return torch.clamp(F.softplus(u) / self._s0, min=self.r_min, max=self.r_max)

    def forward(
        self,
        psi: torch.Tensor,
        base_inv_temp: torch.Tensor | float = 28.57,
    ) -> torch.Tensor:
        """Compute per-token inverse temperature `beta_t` of shape `(..., 1)` in `float32`."""
        r = self.purity_radius(psi)
        base_t = (
            base_inv_temp.float()
            if isinstance(base_inv_temp, torch.Tensor)
            else torch.tensor(float(base_inv_temp), device=psi.device, dtype=torch.float32)
        )
        return torch.clamp(base_t * r, min=1.0, max=self.max_beta)

    def scale_logits(
        self,
        psi: torch.Tensor,
        cos_logits: torch.Tensor,
        base_inv_temp: torch.Tensor | float = 28.57,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale directional cosine similarity logits by per-token `beta_t` and return `(logits, beta_t)`."""
        beta_t = self.forward(psi, base_inv_temp)
        return cos_logits * beta_t.to(cos_logits.dtype), beta_t
