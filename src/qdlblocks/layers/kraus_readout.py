"""Multi-Beam Kraus Subspace POVM Readout Layer and Logit Matrix Effective Rank."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import l2_normalize

__all__ = [
    "compute_logit_matrix_effective_rank",
    "MultiBeamKrausReadout",
]


def compute_logit_matrix_effective_rank(
    logits: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the Logit Matrix Effective Singular Rank R_sigma(L) in [1, min(N, V)].

    Centers the logit matrix across the context dimension (N) and computes the
    exponential of the Shannon entropy of its normalized squared singular values.

    Args:
        logits: Logit tensor of shape `(..., V)` with `logits.ndim >= 2`,
            where `N = logits.numel() // V >= 2` and `V >= 2`.
        eps: Small constant for numerical stability.

    Returns:
        Scalar tensor containing the effective singular rank R_sigma(L).
    """
    if logits.ndim < 2:
        raise ValueError(
            f"compute_logit_matrix_effective_rank requires logits.ndim >= 2, got {logits.ndim}."
        )
    v = logits.shape[-1]
    n = logits.numel() // max(v, 1)
    if n < 2 or v < 2:
        raise ValueError(
            f"compute_logit_matrix_effective_rank requires N >= 2 and V >= 2, got N={n}, V={v}."
        )
    x = logits.reshape(n, v).float()
    x_c = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x_c)
    s_sq = s**2
    p = s_sq / (torch.sum(s_sq) + eps)
    ent = -torch.sum(p * torch.log(torch.clamp(p, min=eps)))
    return torch.exp(ent)


class MultiBeamKrausReadout(nn.Module):
    """Multi-Beam Kraus Subspace POVM Readout to overcome the rank-D_seed vocabulary bottleneck.

    Projects a d-dimensional Hilbert state |psi> into K orthogonal D_seed-dimensional sub-beams
    via a semi-orthogonal Stiefel readout operator R_out in St(K * D_seed, d). Beam 0 measures
    against the base normalized vocabulary dictionary M^(0), while beams k = 1..K-1 measure
    against low-rank Kraus-perturbed dictionaries M^(k) = L2Norm(M^(0) + A_k B_k^T).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 1024,
        seed_dim: int = 128,
        num_beams: int = 4,
        beam_adapter_rank: int = 16,
        adapter_init_scale: float = 1.0,
        povm_temperature: float = 0.035,
        learnable_temperature: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {vocab_size}.")
        if num_beams < 1:
            raise ValueError(f"num_beams must be >= 1, got {num_beams}.")
        if seed_dim < 1:
            raise ValueError(f"seed_dim must be >= 1, got {seed_dim}.")
        if embed_dim < seed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be >= seed_dim ({seed_dim}).")
        if num_beams * seed_dim > embed_dim:
            raise ValueError(
                f"num_beams * seed_dim ({num_beams * seed_dim}) must be <= embed_dim ({embed_dim})."
            )
        if beam_adapter_rank < 1:
            raise ValueError(f"beam_adapter_rank must be >= 1, got {beam_adapter_rank}.")
        if adapter_init_scale <= 0.0:
            raise ValueError(f"adapter_init_scale must be > 0.0, got {adapter_init_scale}.")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seed_dim = seed_dim
        self.num_beams = num_beams
        self.beam_adapter_rank = beam_adapter_rank
        self.adapter_init_scale = adapter_init_scale
        self.eps = eps

        self.readout_rot = nn.Linear(embed_dim, num_beams * seed_dim, bias=False)
        nn.init.orthogonal_(self.readout_rot.weight)

        if num_beams > 1:
            k_aux = num_beams - 1
            gain_a = 2.0 * adapter_init_scale * math.sqrt(float(vocab_size) / float(beam_adapter_rank))
            vocab_a = torch.empty(k_aux, vocab_size, beam_adapter_rank, dtype=torch.float32)
            if vocab_size >= k_aux * beam_adapter_rank:
                a_stacked = torch.empty(vocab_size, k_aux * beam_adapter_rank, dtype=torch.float32)
                nn.init.orthogonal_(a_stacked, gain=gain_a)
                vocab_a.copy_(
                    a_stacked.view(vocab_size, k_aux, beam_adapter_rank).permute(1, 0, 2).contiguous()
                )
            else:
                for i in range(k_aux):
                    nn.init.orthogonal_(vocab_a[i], gain=gain_a)

            seed_b = torch.empty(k_aux, seed_dim, beam_adapter_rank, dtype=torch.float32)
            for i in range(k_aux):
                nn.init.orthogonal_(seed_b[i], gain=1.0)

            self.beam_vocab_a = nn.Parameter(vocab_a)
            self.beam_seed_b = nn.Parameter(seed_b)
            self.beam_weights = nn.Parameter(
                torch.ones(num_beams, dtype=torch.float32) / math.sqrt(float(num_beams))
            )
        else:
            self.register_parameter("beam_vocab_a", None)
            self.register_parameter("beam_seed_b", None)
            self.register_parameter("beam_weights", None)

        init_beta = 1.0 / max(povm_temperature, 1e-4)
        if learnable_temperature:
            self.inv_temp = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))
        else:
            self.register_buffer("inv_temp", torch.tensor(float(init_beta), dtype=torch.float32))

    def forward(self, psi: torch.Tensor, base_wte: torch.Tensor) -> torch.Tensor:
        """Compute multi-beam Kraus POVM readout logits.

        Args:
            psi: Contextual Hilbert state tensor of shape `(..., embed_dim)`.
            base_wte: Base seed vocabulary dictionary of shape `(vocab_size, seed_dim)`.

        Returns:
            Logit tensor of shape `(..., vocab_size)`.
        """
        beams = self.readout_rot(psi).reshape(*psi.shape[:-1], self.num_beams, self.seed_dim)
        beams = l2_normalize(beams, dim=-1, eps=self.eps)
        m_0 = l2_normalize(base_wte, dim=-1, eps=self.eps).to(psi.dtype)
        beta = torch.clamp(self.inv_temp, min=1.0, max=60.0).to(psi.dtype)
        if self.num_beams == 1:
            return F.linear(beams.squeeze(-2), m_0) * beta
        assert (
            self.beam_vocab_a is not None
            and self.beam_seed_b is not None
            and self.beam_weights is not None
        )
        w = self.beam_weights.to(psi.dtype)
        w = w / torch.clamp(torch.linalg.norm(w), min=self.eps)
        logits = w[0] * F.linear(beams[..., 0, :], m_0)
        for k in range(1, self.num_beams):
            a_k = self.beam_vocab_a[k - 1].to(psi.dtype)
            b_k = self.beam_seed_b[k - 1].to(psi.dtype)
            delta_m = torch.matmul(a_k, b_k.transpose(-2, -1))
            m_k = l2_normalize(m_0 + delta_m, dim=-1, eps=self.eps)
            logits = logits + w[k] * F.linear(beams[..., k, :], m_k)
        return logits * beta
