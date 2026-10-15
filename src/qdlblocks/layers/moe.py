"""Quantum-Mechanical Sparsely Activated Mixture of Experts (MoE) Layers.

Provides two zero-Softmax, physically grounded MoE architectures:
1. `QuantumBornMoE` (Pillar 2 - Pure States on S^{d-1}):
   Routes state vectors |psi> via a Projective Quantum Instrument along Stiefel
   measurement axes |R_e> in S^{d-1}:
       a_e = max(0, beta * <R_e | psi>),   p_e = a_e^2 / (sum_j a_j^2 + eps)
   Positive-cone projection naturally zeroes ~50% of experts before Top-k selection,
   and optimizing R in St(d, E) via StiefelMuon enforces mutually orthogonal expert
   routing axes (<R_e | R_{e'}> = delta_{ee'}), preventing router collapse by
   Parseval's completeness identity without artificial Softmax or heuristic losses.

2. `SparseKrausMoE` (Pillar 1 - Complex Density Matrices rho in C^{N x N}):
   Measures Born expectation p_e = Re(<r_e | rho | r_e>) >= 0 against unit complex
   probe kets |r_e> in C^N and selectively executes only the Top-k Kraus operators
   A_e rho A_e^dagger / Tr(...).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from qdlblocks.functional import l2_normalize


class QuantumBornMoE(nn.Module):
    """Sparsely Activated Quantum-Geometric Born Mixture of Experts on S^{d-1}.

    Zero Softmax, Zero Sigmoid:
    - Router: Projective measurement along normalized Stiefel axes |R_e> in S^{d-1}
      using positive-cone Born probabilities p_e = max(0, beta * <R_e | psi>)^2 / Z.
    - Experts: Isometric Up-Projection -> Projective ReLU^2 -> Isometric Down-Projection.
    - Recombination: Either Born probability weighting (`"born_prob"`) or coherent
      amplitude superposition (`"coherent_amplitude"`, weighted by sqrt(p_e)) on S^{d-1}.
    """

    def __init__(
        self,
        embed_dim: int,
        expert_hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        combination_mode: Literal["born_prob", "coherent_amplitude"] = "coherent_amplitude",
        learnable_temperature: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) must be in [1, num_experts ({num_experts})].")
        self.embed_dim = embed_dim
        self.expert_hidden_dim = expert_hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.combination_mode = combination_mode
        self.eps = eps

        # Router measurement frame R in St(embed_dim, num_experts), stored as (E, d).
        # Because R is a 2D weight matrix, StiefelMuon automatically keeps its rows
        # mutually orthogonal (R @ R^T = I_E), preventing expert collapse geometrically!
        self.router_proj = nn.Linear(embed_dim, num_experts, bias=False)
        nn.init.orthogonal_(self.router_proj.weight)

        if learnable_temperature:
            self.router_log_scale = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("router_log_scale", torch.zeros(1))

        # Expert banks: each expert is a two-layer isometric ReLU^2 channel
        self.experts_up = nn.ModuleList(
            [nn.Linear(embed_dim, expert_hidden_dim, bias=False) for _ in range(num_experts)]
        )
        self.experts_down = nn.ModuleList(
            [nn.Linear(expert_hidden_dim, embed_dim, bias=False) for _ in range(num_experts)]
        )
        for up, down in zip(self.experts_up, self.experts_down):
            nn.init.orthogonal_(up.weight)
            nn.init.orthogonal_(down.weight)

        self.last_routing_stats: dict[str, float] = {}

    def compute_born_routing(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute positive-cone Born routing probabilities and Top-k expert selection.

        Args:
            x: Input state tensor `(..., d)` on `S^{d-1}`.

        Returns:
            Tuple `(top_k_weights, top_k_indices, full_born_probs)`.
        """
        r_norm = l2_normalize(self.router_proj.weight, dim=-1, eps=self.eps)
        scale = torch.clamp(torch.exp(self.router_log_scale), min=0.1, max=10.0) * math.sqrt(
            self.embed_dim
        )
        overlaps = F.linear(x, r_norm) * scale
        amplitudes = F.relu(overlaps)
        born_energy = amplitudes.square()
        full_born_probs = born_energy / (born_energy.sum(dim=-1, keepdim=True) + self.eps)

        top_k_vals, top_k_indices = torch.topk(full_born_probs, k=self.top_k, dim=-1)
        top_k_weights = top_k_vals / (top_k_vals.sum(dim=-1, keepdim=True) + self.eps)

        with torch.no_grad():
            zero_frac = float((born_energy == 0).float().mean().item())
            p_safe = full_born_probs.clamp(min=1e-12)
            entropy = float((-(full_born_probs * torch.log(p_safe)).sum(dim=-1).mean()).item())
            # Expert load balance across batch/sequence
            flat_idx = top_k_indices.reshape(-1)
            counts = torch.bincount(flat_idx, minlength=self.num_experts).float()
            load_frac = counts / max(float(flat_idx.numel()), 1.0)
            max_load = float(load_frac.max().item())
            self.last_routing_stats = {
                "positive_cone_zero_fraction": zero_frac,
                "routing_entropy_nats": entropy,
                "max_expert_load_fraction": max_load,
            }

        return top_k_weights, top_k_indices, full_born_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.embed_dim)  # (N_tokens, d)
        top_k_weights, top_k_indices, _ = self.compute_born_routing(x_flat)

        if self.combination_mode == "coherent_amplitude":
            coeffs = torch.sqrt(top_k_weights + self.eps)
        else:
            coeffs = top_k_weights

        out_flat = torch.zeros_like(x_flat)
        scale_up = math.sqrt(2.0 / self.embed_dim)

        # Sparse dispatch: only evaluate tokens routed to each expert e
        for e in range(self.num_experts):
            mask = top_k_indices == e  # (N_tokens, top_k)
            if not mask.any():
                continue
            token_idx, slot_idx = torch.where(mask)
            x_e = x_flat[token_idx]
            h_e = F.relu(self.experts_up[e](x_e) * scale_up).square()
            y_e = l2_normalize(self.experts_down[e](h_e), dim=-1, eps=self.eps)
            w_e = coeffs[token_idx, slot_idx].unsqueeze(-1).to(y_e.dtype)
            out_flat.index_add_(0, token_idx, w_e * y_e)

        out_norm = l2_normalize(out_flat, dim=-1, eps=self.eps)
        return out_norm.reshape(orig_shape)


class SparseKrausMoE(nn.Module):
    """Selective Open-System Kraus Mixture of Experts for Complex Density Matrices.

    Given a batch of Hermitian positive semi-definite density matrices `rho` in `C^{N x N}`,
    measures Born expectation values `p_e = Re(<r_e | rho | r_e>) >= 0` against unit complex
    probe states `|r_e> in C^N`, selects the Top-k Kraus branches per sample, and evolves
    `rho` through only those k active Kraus operators:
        rho_out = (sum_{e in TopK} p_e A_e rho A_e^dagger) / Tr(...)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int | None = None,
        num_experts: int = 8,
        top_k: int = 2,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) must be in [1, num_experts ({num_experts})].")
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.eps = eps

        # Unit complex router probe states |r_e> in C^{in_dim}
        probe_scale = 1.0 / math.sqrt(2.0 * in_dim)
        self.probe_real = nn.Parameter(torch.randn(num_experts, in_dim) * probe_scale)
        self.probe_imag = nn.Parameter(torch.randn(num_experts, in_dim) * probe_scale)

        # Expert Kraus operators A_e in C^{out_dim x in_dim}
        kraus_scale = 1.0 / math.sqrt(2.0 * in_dim)
        self.kraus_real = nn.Parameter(
            torch.randn(num_experts, self.out_dim, in_dim) * kraus_scale
        )
        self.kraus_imag = nn.Parameter(
            torch.randn(num_experts, self.out_dim, in_dim) * kraus_scale
        )

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        # rho: (B, N, N) complex
        bsz = rho.size(0)
        probes = torch.complex(self.probe_real, self.probe_imag)
        probes = probes / (torch.linalg.vector_norm(probes, dim=-1, keepdim=True) + self.eps)

        # Born expectation p_e = Re(<r_e | rho | r_e>) >= 0, shape (B, E)
        born_probs = torch.einsum("ei,bij,ej->be", probes.conj(), rho, probes).real
        born_probs = F.relu(born_probs)
        top_vals, top_idx = torch.topk(born_probs, k=self.top_k, dim=-1)
        top_weights = top_vals / (top_vals.sum(dim=-1, keepdim=True) + self.eps)

        kraus_all = torch.complex(self.kraus_real, self.kraus_imag)  # (E, out_dim, in_dim)
        # Gather only the Top-k Kraus operators per batch item: (B, k, out_dim, in_dim)
        idx_expanded = top_idx.unsqueeze(-1).unsqueeze(-1).expand(
            bsz, self.top_k, self.out_dim, self.in_dim
        )
        active_kraus = torch.gather(
            kraus_all.unsqueeze(0).expand(bsz, -1, -1, -1), dim=1, index=idx_expanded
        )

        # Apply active Kraus operators: sum_{m=1}^k w_{b,m} A_{b,m} rho_b A_{b,m}^dagger
        evolved = torch.einsum(
            "bm,bmij,bjk,bmlk->bil",
            top_weights.to(rho.dtype),
            active_kraus,
            rho,
            active_kraus.conj(),
        )
        evolved = 0.5 * (evolved + evolved.conj().transpose(-1, -2))
        trace = torch.diagonal(evolved, dim1=-2, dim2=-1).real.sum(dim=-1)
        return evolved / (trace.view(bsz, 1, 1) + self.eps)
