"""Quantum Attention Layers: Density-Matrix Superposition and Subspace Projector Born Attention."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import (
    apply_unitary_rope,
    convex_density_mixture,
    ensure_complex,
    l2_normalize,
    power_purify,
    state_to_density,
)
from ..kernels import flash_born_attention


PurificationPlacement = Literal["per_head", "post_fusion", "both", "none"]
PoolingMode = Literal["coherent", "incoherent", "none"]


class PureQuantumAttention(nn.Module):
    """Pure Quantum Attention Layer (Single-Vector / Global State Interference).

    Implements the quantum attention mechanism from Script 1:
    1. Projects input features x in R^D into H complex Hilbert heads: Q, K, V in C^N.
    2. Computes complex inner product amplitude c = <k | q> = sum(Q * K.conj()) in C
       (preserving complex phase interference, with NO Softmax).
    3. Forms the interfered state ket |psi> = c * |v> in C^N.
    4. Constructs per-head density matrices rho_h = |psi><psi| / ||psi||^2.
    5. Applies Power State Purification (rho -> rho^p / Tr(rho^p)) per-head, post-fusion, or both.
    6. Mixes heads via learnable convex Softmax weights: rho_global = sum_h w_h rho_h.
    """

    def __init__(
        self,
        in_features: int,
        hilbert_dim: int = 16,
        num_heads: int = 4,
        purification_power: int = 2,
        purification_placement: PurificationPlacement = "per_head",
        init_std: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.N = hilbert_dim
        self.H = num_heads
        self.purification_power = purification_power
        self.purification_placement = purification_placement
        self.eps = eps

        self.W_q_r = nn.Parameter(torch.randn(num_heads, in_features, hilbert_dim) * init_std)
        self.W_q_i = nn.Parameter(torch.randn(num_heads, in_features, hilbert_dim) * init_std)
        self.W_k_r = nn.Parameter(torch.randn(num_heads, in_features, hilbert_dim) * init_std)
        self.W_k_i = nn.Parameter(torch.randn(num_heads, in_features, hilbert_dim) * init_std)
        self.W_v_r = nn.Parameter(torch.randn(num_heads, in_features, hilbert_dim) * init_std)
        self.W_v_i = nn.Parameter(torch.randn(num_heads, in_features, hilbert_dim) * init_std)

        self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"PureQuantumAttention expects 2D input (B, D), got {tuple(x.shape)}")

        W_q = torch.complex(self.W_q_r, self.W_q_i)
        W_k = torch.complex(self.W_k_r, self.W_k_i)
        W_v = torch.complex(self.W_v_r, self.W_v_i)

        x_c = ensure_complex(x)

        Q = torch.einsum("bd,hdn->bhn", x_c, W_q)
        K = torch.einsum("bd,hdn->bhn", x_c, W_k)
        V = torch.einsum("bd,hdn->bhn", x_c, W_v)

        c = torch.sum(Q * K.conj(), dim=-1, keepdim=True)
        psi = c * V
        rho_h = state_to_density(psi, eps=self.eps)

        if self.purification_placement in ("per_head", "both"):
            rho_h = power_purify(rho_h, power=self.purification_power)

        rho_global = convex_density_mixture(rho_h, self.head_weights, head_dim=1)

        if self.purification_placement in ("post_fusion", "both"):
            rho_global = power_purify(rho_global, power=self.purification_power)

        return rho_global

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, hilbert_dim={self.N}, num_heads={self.H}, "
            f"purification_power={self.purification_power}, "
            f"purification_placement={self.purification_placement!r}"
        )


class ComplexSuperpositionAttention(nn.Module):
    """Multi-Token Complex Superposition Attention Head (No Softmax).

    Implements and generalizes `PureComplexSuperpositionHead` from Script 2:
    1. Projects input tokens `(B, L, D)` (or `(B, D)`) into complex Q, K, V in C^N across H heads.
    2. Computes token-to-token complex inner product amplitudes:
       c_ij = <k_j | q_i> / sqrt(N) in C (preserving complex phase without Softmax).
    3. Forms quantum superposition states over values:
       |psi_i> = sum_j c_ij |v_j> in C^N.
    4. Pools tokens into density matrices according to `pooling_mode`:
       - `'coherent'`: averages state kets |psi_global> = mean_i(|psi_i>) first, then forms
         pure density matrix rho_h = |psi_global><psi_global| / ||psi_global||^2.
       - `'incoherent'`: forms pure density matrix rho_{h,i} for each token i, then averages
         rho_h = mean_i(rho_{h,i}), yielding a mixed density matrix per head.
       - `'none'`: keeps sequence dimension L, returning `(B, L, N, N)`.
    5. Applies Power State Purification (`per_head`, `post_fusion`, `both`, or `none`).
    6. Combines heads via learnable convex Softmax mixture.
    """

    def __init__(
        self,
        in_features: int,
        hilbert_dim: int = 16,
        num_heads: int = 4,
        pooling_mode: PoolingMode = "coherent",
        purification_power: int = 2,
        purification_placement: PurificationPlacement = "per_head",
        init_std: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.N = hilbert_dim
        self.H = num_heads
        self.pooling_mode = pooling_mode
        self.purification_power = purification_power
        self.purification_placement = purification_placement
        self.eps = eps

        self.W_q_r = nn.Parameter(torch.randn(num_heads, in_features, hilbert_dim) * init_std)
        self.W_q_i = nn.Parameter(torch.randn(num_heads, in_features, hilbert_dim) * init_std)
        self.W_k_r = nn.Parameter(torch.randn(num_heads, in_features, hilbert_dim) * init_std)
        self.W_k_i = nn.Parameter(torch.randn(num_heads, in_features, hilbert_dim) * init_std)
        self.W_v_r = nn.Parameter(torch.randn(num_heads, in_features, hilbert_dim) * init_std)
        self.W_v_i = nn.Parameter(torch.randn(num_heads, in_features, hilbert_dim) * init_std)

        self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(
                f"ComplexSuperpositionAttention expects 2D or 3D input, got {tuple(x.shape)}"
            )

        W_q = torch.complex(self.W_q_r, self.W_q_i)
        W_k = torch.complex(self.W_k_r, self.W_k_i)
        W_v = torch.complex(self.W_v_r, self.W_v_i)

        x_c = ensure_complex(x)

        Q = torch.einsum("bld,hdn->bhln", x_c, W_q)
        K = torch.einsum("bld,hdn->bhln", x_c, W_k)
        V = torch.einsum("bld,hdn->bhln", x_c, W_v)

        K_dag = K.conj().transpose(-2, -1)
        c = torch.matmul(Q, K_dag) / (self.N**0.5)
        psi = torch.matmul(c, V)

        if self.pooling_mode == "coherent":
            psi_global = torch.mean(psi, dim=2)
            rho_h = state_to_density(psi_global, eps=self.eps)
        elif self.pooling_mode == "incoherent":
            rho_tokens = state_to_density(psi, eps=self.eps)
            rho_h = torch.mean(rho_tokens, dim=2)
        elif self.pooling_mode == "none":
            rho_h = state_to_density(psi, eps=self.eps)
        else:
            raise ValueError(f"Unknown pooling_mode: {self.pooling_mode!r}")

        if self.purification_placement in ("per_head", "both"):
            rho_h = power_purify(rho_h, power=self.purification_power)

        rho_out = convex_density_mixture(rho_h, self.head_weights, head_dim=1)

        if self.purification_placement in ("post_fusion", "both"):
            rho_out = power_purify(rho_out, power=self.purification_power)

        return rho_out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, hilbert_dim={self.N}, num_heads={self.H}, "
            f"pooling_mode={self.pooling_mode!r}, purification_power={self.purification_power}, "
            f"purification_placement={self.purification_placement!r}"
        )


class SubspaceProjectorBornAttention(nn.Module):
    """Subspace Projector Born Attention on the Unit Hypersphere S^{d-1} (Zero Softmax).

    Implements the Quantum-Geometric Born Attention mechanism:
    1. Orthogonal bias-free Stiefel projections `q_proj`, `k_proj`, `v_proj`, `out_proj`.
    2. Optional Unitary Rotary Position Embeddings (RoPE) on Q and K, followed by L2 normalization.
    3. Overlap computation:
       - Standard real mode (`complex_phase=False`):
         Positive-cone subspace projection amplitude `a_ij = max(0, scale * <K_j | Q_i>)`,
         squared via Born's rule: `P(j|i) = a_ij^2 / (sum_{m <= i} a_im^2 + eps)`.
       - Complex phase-invariant mode (`complex_phase=True`, splitting R^{d_h} -> C^{d_h/2}):
         Hermitian inner product `<K_j | Q_i> = (K_re^T Q_re + K_im^T Q_im) + i(K_re^T Q_im - K_im^T Q_re)`
         with modulus-squared Born probability `(|S_re|^2 + |S_im|^2)` or positive-real cone.
    4. Supports optional:
       - `learnable_temperature`: Per-head learnable sharpness scaling.
       - `value_residual_mixing`: Coherent reference-beam interference mixing initial state `x_0` into `V`.
    """

    causal_mask: torch.Tensor

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        block_size: int = 1024,
        is_causal: bool = True,
        learnable_temperature: bool = False,
        value_residual_mixing: bool = False,
        use_qk_wave_filter: bool = False,
        complex_phase: bool = False,
        use_flash_born: bool = False,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if complex_phase and self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even when complex_phase=True")

        self.block_size = block_size
        self.is_causal = is_causal
        self.learnable_temperature = learnable_temperature
        self.value_residual_mixing = value_residual_mixing
        self.use_qk_wave_filter = use_qk_wave_filter
        self.complex_phase = complex_phase
        self.use_flash_born = use_flash_born
        self.eps = eps
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        nn.init.orthogonal_(self.q_proj.weight)
        nn.init.orthogonal_(self.k_proj.weight)
        nn.init.orthogonal_(self.v_proj.weight)
        nn.init.orthogonal_(self.out_proj.weight)

        if learnable_temperature:
            self.head_scale = nn.Parameter(torch.full((1, num_heads, 1, 1), self.scale))
        else:
            self.register_parameter("head_scale", None)

        if value_residual_mixing:
            # Sigmoid gate initialized near 0.25 reference beam mix
            self.v_mix_logit = nn.Parameter(torch.tensor(-1.1))
        else:
            self.register_parameter("v_mix_logit", None)

        if use_qk_wave_filter:
            self.q_prev_mix = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
            self.k_prev_mix = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        else:
            self.register_parameter("q_prev_mix", None)
            self.register_parameter("k_prev_mix", None)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            persistent=False,
        )

        # Optional telemetry capture
        self.record_attention: bool = False
        self.last_born_weights: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        psi_0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Subspace Projector Born Attention.

        Args:
            x: Input state sequence of shape `(B, T, embed_dim)`.
            cos: Optional RoPE cosine table `(1, 1, T, head_dim // 2)`.
            sin: Optional RoPE sine table `(1, 1, T, head_dim // 2)`.
            psi_0: Optional initial lifted token state `(B, T, embed_dim)` for coherent
                reference-beam value mixing when `value_residual_mixing=True`.

        Returns:
            Output state sequence of shape `(B, T, embed_dim)`.
        """
        B, T, D = x.shape
        psi = l2_normalize(x, dim=-1, eps=self.eps)

        Q = self.q_proj(psi).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(psi).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(psi).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if self.value_residual_mixing and psi_0 is not None and self.v_mix_logit is not None:
            V_0 = self.v_proj(l2_normalize(psi_0, dim=-1, eps=self.eps))
            V_0 = V_0.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            alpha = torch.sigmoid(self.v_mix_logit).to(V.dtype)
            V = (1.0 - alpha) * V + alpha * V_0

        if (
            self.use_qk_wave_filter
            and T > 1
            and self.q_prev_mix is not None
            and self.k_prev_mix is not None
        ):
            q_prev = F.pad(Q[:, :, :-1, :], (0, 0, 1, 0))
            k_prev = F.pad(K[:, :, :-1, :], (0, 0, 1, 0))
            qm = (0.06 * torch.tanh(self.q_prev_mix)).to(Q.dtype)
            km = (0.06 * torch.tanh(self.k_prev_mix)).to(K.dtype)
            Q = Q + qm * q_prev
            K = K + km * k_prev

        if cos is not None and sin is not None:
            Q = apply_unitary_rope(Q, cos, sin)
            K = apply_unitary_rope(K, cos, sin)

        Q = l2_normalize(Q, dim=-1, eps=self.eps)
        K = l2_normalize(K, dim=-1, eps=self.eps)

        eff_scale = (
            torch.clamp(self.head_scale, min=1.0, max=100.0)
            if self.head_scale is not None
            else self.scale
        )

        # Fast SRAM-tiled Triton Flash-Born Attention path (zero T x T matrix materialization)
        if self.use_flash_born and not self.complex_phase and not self.record_attention:
            out_heads = flash_born_attention(
                Q, K, V, scale=eff_scale, is_causal=self.is_causal, eps=self.eps
            )
            out_concat = out_heads.transpose(1, 2).contiguous().view(B, T, D)
            return self.out_proj(out_concat)

        if self.complex_phase:
            # Split R^{head_dim} -> C^{head_dim // 2}: (Q_re, Q_im) and (K_re, K_im)
            half = self.head_dim // 2
            Q_re, Q_im = Q[..., :half], Q[..., half:]
            K_re, K_im = K[..., :half], K[..., half:]

            # Real and Imaginary parts of <K_j | Q_i>
            s_re = torch.matmul(Q_re, K_re.transpose(-1, -2)) + torch.matmul(
                Q_im, K_im.transpose(-1, -2)
            )
            s_im = torch.matmul(Q_im, K_re.transpose(-1, -2)) - torch.matmul(
                Q_re, K_im.transpose(-1, -2)
            )
            # Positive-cone real alignment + imaginary phase interference modulus squared
            amp_re = F.relu(s_re * eff_scale)
            amp_im = F.relu(s_im * eff_scale)
            prob = amp_re.square() + amp_im.square()
        else:
            scaled_overlap = torch.matmul(Q, K.transpose(-1, -2)) * eff_scale
            projected_amp = F.relu(scaled_overlap)
            prob = projected_amp.square()

        if self.is_causal:
            if T <= self.block_size:
                mask = self.causal_mask[:, :, :T, :T]
            else:
                mask = torch.tril(torch.ones(T, T, device=x.device, dtype=prob.dtype)).view(
                    1, 1, T, T
                )
            prob = prob * mask

        born_weights = prob / (torch.sum(prob, dim=-1, keepdim=True) + self.eps)

        if self.record_attention:
            self.last_born_weights = born_weights.detach()

        out_heads = torch.matmul(born_weights, V)
        out_concat = out_heads.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out_concat)
