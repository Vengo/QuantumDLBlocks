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
from ..kernels import flash_born_attention, flash_purified_born_attention
from .early_epoch_quantum import purified_born_attention_weights
from .isometric import YaRNRotarySchedule, apply_partial_key_offset
from .mudd_and_gates import InputChannelGate, apply_gated_xsa



PurificationPlacement = Literal["per_head", "post_fusion", "both", "none"]
PoolingMode = Literal["coherent", "incoherent", "none"]


class _MemoryEfficientPurifiedBornAttentionFn(torch.autograd.Function):
    """Chunked activation-recomputed autograd function for Purified Born Attention.

    Avoids materializing and storing full `(B, H, T, T)` intermediates across layers in HBM
    when `B * H * T * T` is large (e.g. `B=128, H=16, T=512`), while preserving exact gradients
    with respect to `Q`, `K`, `V`, `gamma_h`, and `tau_h`.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gamma_h: torch.Tensor,
        tau_h: torch.Tensor,
        eps_base: float,
        is_causal: bool,
        eps: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(q, k, v, gamma_h, tau_h)
        ctx.eps_base = float(eps_base)
        ctx.is_causal = bool(is_causal)
        ctx.eps = float(eps)

        b_sz = q.size(0)
        chunk_b = min(b_sz, 8)
        out_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for b0 in range(0, b_sz, chunk_b):
                b1 = min(b_sz, b0 + chunk_b)
                w_c = purified_born_attention_weights(
                    q[b0:b1],
                    k[b0:b1],
                    gamma=gamma_h,
                    tau=tau_h,
                    eps_base=ctx.eps_base,
                    causal=ctx.is_causal,
                    eps=ctx.eps,
                )
                out_chunks.append(torch.matmul(w_c.to(v.dtype), v[b0:b1]))
        return torch.cat(out_chunks, dim=0)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        q, k, v, gamma_h, tau_h = ctx.saved_tensors
        b_sz = q.size(0)
        chunk_b = min(b_sz, 8)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dgamma = torch.zeros_like(gamma_h, dtype=torch.float32)
        dtau = torch.zeros_like(tau_h, dtype=torch.float32)

        with torch.enable_grad():
            for b0 in range(0, b_sz, chunk_b):
                b1 = min(b_sz, b0 + chunk_b)
                q_c = q[b0:b1].detach().requires_grad_(True)
                k_c = k[b0:b1].detach().requires_grad_(True)
                v_c = v[b0:b1].detach().requires_grad_(True)
                g_c = gamma_h.detach().float().requires_grad_(True)
                t_c = tau_h.detach().float().requires_grad_(True)
                w_c = purified_born_attention_weights(
                    q_c,
                    k_c,
                    gamma=g_c,
                    tau=t_c,
                    eps_base=ctx.eps_base,
                    causal=ctx.is_causal,
                    eps=ctx.eps,
                )
                out_c = torch.matmul(w_c.to(v_c.dtype), v_c)
                gq, gk, gv, gg, gt = torch.autograd.grad(
                    out_c,
                    (q_c, k_c, v_c, g_c, t_c),
                    grad_outputs=grad_out[b0:b1],
                )
                dq[b0:b1].copy_(gq)
                dk[b0:b1].copy_(gk)
                dv[b0:b1].copy_(gv)
                dgamma.add_(gg)
                dtau.add_(gt)

        return dq, dk, dv, dgamma.to(gamma_h.dtype), dtau.to(tau_h.dtype), None, None, None


@torch._dynamo.disable
def _memory_efficient_purified_born_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gamma_h: torch.Tensor,
    tau_h: torch.Tensor,
    eps_base: float,
    is_causal: bool,
    eps: float,
) -> torch.Tensor:
    return _MemoryEfficientPurifiedBornAttentionFn.apply(  # type: ignore[return-value]
        q, k, v, gamma_h, tau_h, eps_base, is_causal, eps
    )


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


def compute_document_ids(idx: torch.Tensor, eot_token_id: int = 50256) -> torch.Tensor:
    """Compute per-token document segment IDs `(B, T)` (`int32`) from token indices `idx`.

    An `<|endoftext|>` token (`eot_token_id`) at position `t` belongs to the document
    it terminates, and position `t + 1` increments to the next document ID.
    """
    is_eot = (idx == int(eot_token_id)).to(torch.int32)
    return torch.cumsum(is_eot, dim=-1, dtype=torch.int32) - is_eot


class SubspaceProjectorBornAttention(nn.Module):
    """Subspace Projector Born Attention on the Unit Hypersphere S^{d-1} (Zero Softmax).

    Implements the Quantum-Geometric Born Attention mechanism:
    1. Orthogonal bias-free Stiefel projections `q_proj`, `k_proj`, `v_proj`, `out_proj`
       (or fused `qkv_proj` when `use_fused_qkv=True`).
    2. Optional Unitary Rotary Position Embeddings (RoPE) on Q and K, followed by L2 normalization.
    3. Overlap computation with optional sliding-window (`window_size > 0`) and document-boundary
       (`doc_ids`) masking.
    4. Supports optional:
       - `learnable_temperature`: Per-head learnable sharpness scaling.
       - `value_residual_mixing`: Coherent reference-beam interference mixing initial state `x_0` / `v_0` into `V`.
       - `has_value_embedding`: Learnable gate `ve_mix_logit` mixing token value embeddings `ve` into `V`.
    """

    causal_mask: torch.Tensor

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        block_size: int = 1024,
        is_causal: bool = True,
        learnable_temperature: bool = True,
        value_residual_mixing: bool = False,
        has_value_embedding: bool = False,
        use_fused_qkv: bool = False,
        use_qk_wave_filter: bool = False,
        complex_phase: bool = False,
        use_flash_born: bool = False,
        use_purified_attn: bool = False,
        coherence_tau_init: float = 0.0625,
        coherence_tau_max: float = 0.35,
        purified_eps_base: float = 0.05,
        use_mudd_and_gates: bool = False,
        use_partial_key_offset: bool = False,
        use_paired_heads: bool = False,
        qk_head_dim: int | None = None,
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
        self.qk_head_dim = int(qk_head_dim) if qk_head_dim is not None else self.head_dim
        if complex_phase and self.qk_head_dim % 2 != 0:
            raise ValueError(f"qk_head_dim ({self.qk_head_dim}) must be even when complex_phase=True")
        if use_paired_heads and num_heads % 2 != 0:
            raise ValueError(f"num_heads ({num_heads}) must be even when use_paired_heads=True")

        self.block_size = block_size
        self.is_causal = is_causal
        self.causal = is_causal
        self.learnable_temperature = learnable_temperature
        self.value_residual_mixing = value_residual_mixing
        self.has_value_embedding = bool(has_value_embedding)
        self.use_fused_qkv = bool(use_fused_qkv) and (self.qk_head_dim == self.head_dim)
        self.use_qk_wave_filter = use_qk_wave_filter
        self.complex_phase = complex_phase
        self.use_flash_born = use_flash_born
        self.use_purified_attn = bool(use_purified_attn)
        self.coherence_tau_init = float(coherence_tau_init)
        self.coherence_tau_max = float(coherence_tau_max)
        self.purified_eps_base = float(purified_eps_base)
        self.use_mudd_and_gates = bool(use_mudd_and_gates)
        self.use_partial_key_offset = bool(use_partial_key_offset)
        self.use_paired_heads = bool(use_paired_heads)
        self.eps = eps
        self.scale = math.sqrt(self.qk_head_dim)

        qk_total_dim = self.num_heads * self.qk_head_dim
        if self.use_fused_qkv:
            self.qkv_proj: nn.Linear | None = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
            with torch.no_grad():
                for chunk_idx in range(3):
                    nn.init.orthogonal_(
                        self.qkv_proj.weight[chunk_idx * embed_dim : (chunk_idx + 1) * embed_dim]
                    )
            self.q_proj: nn.Linear | None = None
            self.k_proj: nn.Linear | None = None
            self.v_proj: nn.Linear | None = None
        else:
            self.qkv_proj = None
            self.q_proj = nn.Linear(embed_dim, qk_total_dim, bias=False)
            self.k_proj = nn.Linear(embed_dim, qk_total_dim, bias=False)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            nn.init.orthogonal_(self.q_proj.weight)
            nn.init.orthogonal_(self.k_proj.weight)
            nn.init.orthogonal_(self.v_proj.weight)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.orthogonal_(self.out_proj.weight)

        if self.use_mudd_and_gates:
            self.ve_input_gate: InputChannelGate | None = InputChannelGate(
                num_heads=num_heads, gate_channels=6, head_first=True
            )
            self.attn_output_gate: InputChannelGate | None = InputChannelGate(
                num_heads=num_heads, gate_channels=6, head_first=True
            )
            self.xsa_alpha: nn.Parameter | None = nn.Parameter(torch.zeros(num_heads))
        else:
            self.ve_input_gate = None
            self.attn_output_gate = None
            self.register_parameter("xsa_alpha", None)

        if learnable_temperature or self.use_purified_attn:
            self.head_scale = nn.Parameter(torch.full((1, num_heads, 1, 1), self.scale))
        else:
            self.register_parameter("head_scale", None)

        if self.use_purified_attn:
            tau_0 = min(max(self.coherence_tau_init, 1e-4), self.coherence_tau_max - 1e-4)
            raw_init = math.log(tau_0 / (self.coherence_tau_max - tau_0))
            self.coherence_tau_raw = nn.Parameter(torch.full((1, num_heads, 1, 1), raw_init))
        else:
            self.register_parameter("coherence_tau_raw", None)

        if value_residual_mixing:
            # Sigmoid gate initialized near 0.25 reference beam mix
            self.v_mix_logit = nn.Parameter(torch.tensor(-1.1))
        else:
            self.register_parameter("v_mix_logit", None)

        if self.has_value_embedding:
            self.ve_mix_logit = nn.Parameter(torch.tensor(-2.2))
        else:
            self.register_parameter("ve_mix_logit", None)

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
        self.last_v_mixed: torch.Tensor | None = None
        self.last_psi_in: torch.Tensor | None = None

    @property
    def qk_gain_param(self) -> nn.Parameter | None:
        return self.head_scale

    @qk_gain_param.setter
    def qk_gain_param(self, value: nn.Parameter | None) -> None:
        self.head_scale = value

    def _finalize_heads(
        self,
        out_heads: torch.Tensor,
        V: torch.Tensor,
        psi_in: torch.Tensor,
        B: int,
        T: int,
        D: int,
    ) -> torch.Tensor:
        if self.use_mudd_and_gates:
            if self.xsa_alpha is not None:
                out_heads = apply_gated_xsa(out_heads, V, self.xsa_alpha, eps=self.eps)
            if self.attn_output_gate is not None:
                out_heads = self.attn_output_gate(psi_in) * out_heads
        out_concat = out_heads.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out_concat)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        psi_0: torch.Tensor | None = None,
        v_0: torch.Tensor | None = None,
        ve: torch.Tensor | None = None,
        window_size: int = 0,
        doc_ids: torch.Tensor | None = None,
        return_v: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute Subspace Projector Born Attention."""
        B, T, D = x.shape
        psi = l2_normalize(x, dim=-1, eps=self.eps)
        psi_raw = psi

        if self.use_fused_qkv and self.qkv_proj is not None:
            q_lin, k_lin, v_lin = self.qkv_proj(psi).chunk(3, dim=-1)
            Q = q_lin.reshape(B, T, self.num_heads, self.qk_head_dim).transpose(1, 2).contiguous()
            K = k_lin.reshape(B, T, self.num_heads, self.qk_head_dim).transpose(1, 2).contiguous()
            V = v_lin.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        else:
            assert self.q_proj is not None and self.k_proj is not None and self.v_proj is not None
            Q = self.q_proj(psi).reshape(B, T, self.num_heads, self.qk_head_dim).transpose(1, 2).contiguous()
            K = self.k_proj(psi).reshape(B, T, self.num_heads, self.qk_head_dim).transpose(1, 2).contiguous()
            V = self.v_proj(psi).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        if return_v:
            V_raw = V.detach().clone().contiguous() if not V.requires_grad else (V + 0.0).contiguous()
        else:
            V_raw = V

        if self.value_residual_mixing and self.v_mix_logit is not None:
            if v_0 is not None:
                alpha = torch.sigmoid(self.v_mix_logit).to(V.dtype)
                V = (1.0 - alpha) * V + alpha * v_0.to(V.dtype)
            elif psi_0 is not None and not return_v:
                psi_0_n = l2_normalize(psi_0, dim=-1, eps=self.eps)
                if self.use_fused_qkv and self.qkv_proj is not None:
                    _, _, v0_lin = self.qkv_proj(psi_0_n).chunk(3, dim=-1)
                else:
                    assert self.v_proj is not None
                    v0_lin = self.v_proj(psi_0_n)
                V_0 = v0_lin.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
                alpha = torch.sigmoid(self.v_mix_logit).to(V.dtype)
                V = (1.0 - alpha) * V + alpha * V_0

        if ve is not None:
            ve_n = l2_normalize(ve, dim=-1, eps=self.eps).to(V.dtype)
            ve_heads = ve_n.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
            if self.ve_input_gate is not None:
                ve_heads = self.ve_input_gate(psi) * ve_heads
            alpha_ve = (
                torch.sigmoid(self.ve_mix_logit).to(V.dtype)
                if self.ve_mix_logit is not None
                else torch.tensor(0.10, device=V.device, dtype=V.dtype)
            )
            V = (1.0 - alpha_ve) * V + alpha_ve * ve_heads


        if not torch.compiler.is_compiling():
            self.last_v_mixed = V.detach()
            self.last_psi_in = psi.detach()

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
            half_qk = self.qk_head_dim // 2
            if cos.size(-1) >= half_qk:
                c_use = cos[..., :half_qk]
                s_use = sin[..., :half_qk]
            else:
                pad_d = half_qk - cos.size(-1)
                c_use = F.pad(cos, (0, pad_d), value=1.0)
                s_use = F.pad(sin, (0, pad_d), value=0.0)
            Q = apply_unitary_rope(Q, c_use, s_use)
            K = apply_unitary_rope(K, c_use, s_use)

        if self.use_partial_key_offset:
            K = apply_partial_key_offset(K, preserve_norm=False, eps=self.eps)

        Q = l2_normalize(Q, dim=-1, eps=self.eps)
        K = l2_normalize(K, dim=-1, eps=self.eps)

        if self.use_paired_heads:
            # Interleave adjacent heads (2h, 2h+1) at each token t into (B, H//2, 2*T, D)
            h_pairs = self.num_heads // 2
            Q_p = Q.transpose(1, 2).reshape(B, T * 2, h_pairs, self.qk_head_dim).transpose(1, 2).contiguous()
            K_p = K.transpose(1, 2).reshape(B, T * 2, h_pairs, self.qk_head_dim).transpose(1, 2).contiguous()
            V_p = V.transpose(1, 2).reshape(B, T * 2, h_pairs, self.head_dim).transpose(1, 2).contiguous()
            if self.head_scale is not None:
                scale_p = self.head_scale.view(1, h_pairs, 2, 1, 1).mean(dim=2)
            else:
                scale_p = torch.tensor(self.scale, device=Q.device, dtype=Q.dtype)
            s_ov = torch.matmul(Q_p, K_p.transpose(-1, -2)) * torch.clamp(scale_p, min=0.5, max=100.0)
            if self.use_purified_attn:
                if self.coherence_tau_raw is not None:
                    tau_raw_p = self.coherence_tau_raw.view(1, h_pairs, 2, 1, 1).mean(dim=2)
                    tau_p = self.coherence_tau_max * torch.sigmoid(tau_raw_p)
                else:
                    tau_p = torch.tensor(self.coherence_tau_init, device=Q.device, dtype=Q.dtype)
                amp_p = F.relu(s_ov - tau_p)
                prob_p = amp_p.square() * (amp_p.square() + self.purified_eps_base)
            else:
                prob_p = F.relu(s_ov).square()
            idx_2t = torch.arange(T * 2, device=x.device) // 2
            diff_p = idx_2t[:, None] - idx_2t[None, :]
            mask_bool_p = (diff_p >= 0) if self.is_causal else torch.ones((T * 2, T * 2), device=x.device, dtype=torch.bool)
            if window_size > 0:
                mask_bool_p = mask_bool_p & (diff_p < int(window_size))
            mask_4d_p = mask_bool_p.view(1, 1, T * 2, T * 2)
            if doc_ids is not None:
                doc_2t = doc_ids.repeat_interleave(2, dim=-1)
                mask_4d_p = mask_4d_p & (doc_2t[:, None, :, None] == doc_2t[:, None, None, :])
            prob_p = prob_p * mask_4d_p.to(prob_p.dtype)
            w_p = prob_p / (torch.sum(prob_p, dim=-1, keepdim=True) + self.eps)
            out_p = torch.matmul(w_p.to(V_p.dtype), V_p)
            out_heads = out_p.transpose(1, 2).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
            out = self._finalize_heads(out_heads, V, psi_raw, B, T, D)
            return (out, V_raw) if return_v else out

        if self.use_purified_attn:
            gamma_h = (
                torch.clamp(self.head_scale, min=0.5, max=20.0)
                if self.head_scale is not None
                else torch.tensor(self.scale, device=Q.device, dtype=Q.dtype)
            )
            tau_h = (
                self.coherence_tau_max * torch.sigmoid(self.coherence_tau_raw)
                if self.coherence_tau_raw is not None
                else torch.tensor(self.coherence_tau_init, device=Q.device, dtype=Q.dtype)
            )
            if (
                self.use_flash_born
                and not self.record_attention
                and Q.is_cuda
                and self.qk_head_dim == self.head_dim
            ):
                out_heads = flash_purified_born_attention(
                    Q,
                    K,
                    V,
                    scale=gamma_h,
                    tau=tau_h,
                    eps_base=self.purified_eps_base,
                    is_causal=self.is_causal,
                    eps=self.eps,
                    window_size=window_size,
                    doc_ids=doc_ids,
                )
            elif (
                not self.record_attention
                and window_size <= 0
                and doc_ids is None
                and (B * self.num_heads * T * T >= 16_777_216)
            ):
                out_heads = _memory_efficient_purified_born_attn(
                    Q,
                    K,
                    V,
                    gamma_h,
                    tau_h,
                    self.purified_eps_base,
                    self.is_causal,
                    self.eps,
                )
            else:
                born_weights = purified_born_attention_weights(
                    Q,
                    K,
                    gamma=gamma_h,
                    tau=tau_h,
                    eps_base=self.purified_eps_base,
                    causal=self.is_causal,
                    eps=self.eps,
                    window_size=window_size,
                    doc_ids=doc_ids,
                )
                if self.record_attention:
                    self.last_born_weights = born_weights.detach()
                out_heads = torch.matmul(born_weights.to(V.dtype), V)
            out = self._finalize_heads(out_heads, V, psi_raw, B, T, D)
            return (out, V_raw) if return_v else out

        eff_scale = (
            torch.clamp(self.head_scale, min=1.0, max=100.0)
            if self.head_scale is not None
            else self.scale
        )

        # Fast SRAM-tiled Triton Flash-Born Attention path (zero T x T matrix materialization)
        if (
            self.use_flash_born
            and not self.complex_phase
            and not self.record_attention
            and self.qk_head_dim == self.head_dim
        ):
            out_heads = flash_born_attention(
                Q,
                K,
                V,
                scale=eff_scale,
                is_causal=self.is_causal,
                eps=self.eps,
                window_size=window_size,
                doc_ids=doc_ids,
            )
            out = self._finalize_heads(out_heads, V, psi_raw, B, T, D)
            return (out, V_raw) if return_v else out

        if self.complex_phase:
            half = self.qk_head_dim // 2
            Q_re, Q_im = Q[..., :half], Q[..., half:]
            K_re, K_im = K[..., :half], K[..., half:]

            s_re = torch.matmul(Q_re, K_re.transpose(-1, -2)) + torch.matmul(
                Q_im, K_im.transpose(-1, -2)
            )
            s_im = torch.matmul(Q_im, K_re.transpose(-1, -2)) - torch.matmul(
                Q_re, K_im.transpose(-1, -2)
            )
            amp_re = F.relu(s_re * eff_scale)
            amp_im = F.relu(s_im * eff_scale)
            prob = amp_re.square() + amp_im.square()
        else:
            scaled_overlap = torch.matmul(Q, K.transpose(-1, -2)) * eff_scale
            projected_amp = F.relu(scaled_overlap)
            prob = projected_amp.square()

        if self.is_causal or window_size > 0 or doc_ids is not None:
            if window_size <= 0 and doc_ids is None and self.is_causal:
                if T <= self.block_size:
                    mask = self.causal_mask[:, :, :T, :T]
                else:
                    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=prob.dtype)).view(
                        1, 1, T, T
                    )
                prob = prob * mask
            else:
                idx_t = torch.arange(T, device=x.device)
                diff = idx_t[:, None] - idx_t[None, :]
                mask_bool = (diff >= 0) if self.is_causal else torch.ones((T, T), device=x.device, dtype=torch.bool)
                if window_size > 0:
                    mask_bool = mask_bool & (diff < int(window_size))
                mask_4d = mask_bool.view(1, 1, T, T)
                if doc_ids is not None:
                    mask_4d = mask_4d & (doc_ids[:, None, :, None] == doc_ids[:, None, None, :])
                prob = prob * mask_4d.to(prob.dtype)

        born_weights = prob / (torch.sum(prob, dim=-1, keepdim=True) + self.eps)

        if self.record_attention:
            self.last_born_weights = born_weights.detach()

        out_heads = torch.matmul(born_weights, V)
        out = self._finalize_heads(out_heads, V, psi_raw, B, T, D)
        return (out, V_raw) if return_v else out


    @property
    def w_qkv(self) -> nn.Linear | None:
        return self.qkv_proj

    @staticmethod
    def _born_attention_weights(
        q: torch.Tensor,
        k: torch.Tensor,
        scale: float | torch.Tensor = 10.0,
        is_causal: bool = True,
        eps: float = 1e-8,
        window_size: int = 0,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute normalized positive-cone Born attention weights `(B, H, T, T)` with optional window and doc masking."""
        t = q.size(-2)
        eff_scale = scale.view(1, -1, 1, 1) if isinstance(scale, torch.Tensor) and scale.ndim == 1 else scale
        prob = F.relu(torch.matmul(q, k.transpose(-1, -2)) * eff_scale).square()
        if is_causal or window_size > 0 or doc_ids is not None:
            idx_t = torch.arange(t, device=q.device)
            diff = idx_t[:, None] - idx_t[None, :]
            mask_bool = (diff >= 0) if is_causal else torch.ones((t, t), device=q.device, dtype=torch.bool)
            if window_size > 0:
                mask_bool = mask_bool & (diff < int(window_size))
            mask_4d = mask_bool.view(1, 1, t, t)
            if doc_ids is not None:
                mask_4d = mask_4d & (doc_ids[:, None, :, None] == doc_ids[:, None, None, :])
            prob = prob * mask_4d.to(prob.dtype)
        return prob / (torch.sum(prob, dim=-1, keepdim=True) + eps)



