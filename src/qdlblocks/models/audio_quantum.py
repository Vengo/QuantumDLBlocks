"""Complex Acoustic Phase & Superposition Classifier (ComplexSuperpositionAudioClassifier).

Ingests complex-valued Short-Time Fourier Transform (STFT) spectrograms X(omega, t) in C^F
directly without discarding acoustic phase, evolving states via multi-head complex wave
interference, Kraus / SparseKrausMoE open-system channels, PowerPurification, and POVM readout.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..functional import state_to_density
from ..layers.activations import PowerPurification
from ..layers.channels import KrausChannel
from ..layers.complex_linear import ComplexLinear
from ..layers.measurement import POVMMeasurement
from ..layers.moe import SparseKrausMoE


class ComplexSuperpositionAudioClassifier(nn.Module):
    """Native Complex-Phase Acoustic Classifier operating on C^{hilbert_dim x hilbert_dim}."""

    def __init__(
        self,
        n_freq_bins: int = 65,
        hilbert_dim: int = 24,
        num_classes: int = 10,
        num_heads: int = 4,
        num_kraus: int = 4,
        use_sparse_kraus_moe: bool = False,
        num_experts: int = 8,
        top_k_experts: int = 2,
        purification_power: int = 2,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if hilbert_dim % num_heads != 0:
            raise ValueError(
                f"hilbert_dim ({hilbert_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.n_freq_bins = n_freq_bins
        self.hilbert_dim = hilbert_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.head_dim = hilbert_dim // num_heads
        self.eps = eps

        self.q_proj = ComplexLinear(n_freq_bins, hilbert_dim, bias=False)
        self.k_proj = ComplexLinear(n_freq_bins, hilbert_dim, bias=False)
        self.v_proj = ComplexLinear(n_freq_bins, hilbert_dim, bias=False)

        if use_sparse_kraus_moe:
            self.channel: nn.Module = SparseKrausMoE(
                in_dim=hilbert_dim,
                out_dim=hilbert_dim,
                num_experts=num_experts,
                top_k=top_k_experts,
                eps=eps,
            )
        else:
            self.channel = KrausChannel(
                in_hilbert_dim=hilbert_dim,
                out_hilbert_dim=hilbert_dim,
                num_kraus=num_kraus,
                residual=True,
                eps=eps,
            )

        self.purify = PowerPurification(power=purification_power, eps=eps)
        self.povm = POVMMeasurement(hilbert_dim=hilbert_dim, num_classes=num_classes)
        self.logit_scale = nn.Parameter(torch.tensor(20.0))

    def forward(
        self, x: torch.Tensor, return_density: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not x.is_complex():
            x = x.to(torch.complex64)
        bsz, t_frames, _ = x.shape

        # Frame-wise unit norm in C^{n_freq_bins}
        x_norm = x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + self.eps)

        # Project each frame across frequencies into C^{hilbert_dim}.
        # Relative cross-frequency phases interfere coherently inside q_proj/k_proj/v_proj,
        # while random per-frame global phase exp(i * phi_{0,b,t}) factors out as a scalar phase.
        q_raw = self.q_proj(x_norm)
        k_raw = self.k_proj(x_norm)
        v_raw = self.v_proj(x_norm)

        q = q_raw.view(bsz, t_frames, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_raw.view(bsz, t_frames, self.num_heads, self.head_dim).transpose(1, 2)

        # Born-rule temporal similarity |<q_t | k_{t'}>|^2 is invariant to per-frame phase phi_{0,b,t}
        overlap = torch.matmul(q, k.conj().transpose(-1, -2)) / math.sqrt(self.head_dim)
        born_weights = overlap.real.square() + overlap.imag.square()
        born_weights = born_weights / (born_weights.sum(dim=-1, keepdim=True) + self.eps)
        frame_importance = born_weights.mean(dim=(1, 2))  # (B, T)
        frame_importance = frame_importance / (frame_importance.sum(dim=-1, keepdim=True) + self.eps)

        # Form per-frame density matrices |v_t><v_t| in C^{hilbert_dim x hilbert_dim}.
        # Because |e^{i phi_0} v_t><e^{i phi_0} v_t| = |v_t><v_t|, per-frame phase jitter cancels
        # identically while cross-frequency harmonic phase interference is preserved!
        rho_frames = state_to_density(v_raw, eps=self.eps)  # (B, T, H, H)
        rho_attn = torch.einsum("bt,btij->bij", frame_importance.to(rho_frames.dtype), rho_frames)
        rho_mean = rho_frames.mean(dim=1)
        rho = 0.5 * (rho_attn + rho_mean)

        rho_out = self.purify(self.channel(rho))
        logits = self.povm(rho_out) * self.logit_scale
        if return_density:
            return logits, rho_out
        return logits
