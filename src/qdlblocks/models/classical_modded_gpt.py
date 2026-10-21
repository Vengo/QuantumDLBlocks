"""Apples-to-Apples Single-GPU Classical Modded-NanoGPT Baseline Architecture (Plans 013–016).

Implements the modern classical Transformer architecture (`KellerJordan/modded-nanogpt`
single-GPU primitives) matched to `QuantumBornGPT` in width, depth, sequence length,
data stream, and Muon + AdamW optimization:
1. Parameterless `F.rms_norm` before attention and MLP, and on token embeddings.
2. Unitary Rotary Position Embeddings (`UnitaryRoPE`, with optional half-truncated zero-frequency
   subspace `zero_half_rope=True`) + QK-RMSNorm Causal Softmax Attention (`F.scaled_dot_product_attention`,
   with optional sliding-window and `<|endoftext|>` document-boundary masking).
3. ReLU^2 MLP (`F.relu(x @ W1).square() @ W2`).
4. Learnable input highway mixing, cross-layer Layer-0 value residual mixing (`v_0`),
   3-bank token value embeddings (`ve`), interior attention layer skipping (`skip_attn_layers`),
   and matrix or scalar U-Net encoder-to-decoder skip connections (`unet_skip_mode`).
5. Untied (or optionally tied) Euclidean linear readout `lm_head` with `tanh` logit
   softcapping at `logit_softcap = 30.0`.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import apply_unitary_rope
from ..layers.attention import compute_document_ids
from ..layers.isometric import UnitaryRoPE, apply_partial_key_offset
from ..layers.mudd_and_gates import (
    ForwardSmearGate,
    InputChannelGate,
    MUDDDynamicSkip,
    apply_gated_xsa,
)
from ..layers.ngram_hash import HashedNgramEmbedding
from ..layers.parameter_scaling import compute_depth_residual_scale

UNetSkipMode = Literal["matrix", "scalar"]


class LeaderboardRotary(nn.Module):
    """Exact KellerJordan/modded-nanogpt half-truncated Rotary embedding (base=1024, chunk(2, dim=-1))."""

    def __init__(self, head_dim: int, max_seq_len: int = 65536, base: float = 1024.0) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.max_seq_len = int(max_seq_len)
        self.base = float(base)
        quarter = max(1, self.head_dim // 4)
        angular_freq = (1.0 / self.base) ** torch.linspace(0.0, 1.0, steps=quarter, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(quarter)])
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        theta = torch.outer(t, angular_freq)
        self.register_buffer("cos_cached", theta.cos(), persistent=False)
        self.register_buffer("sin_cached", theta.sin(), persistent=False)

    def get_cos_sin(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_leaderboard_rotary(
    x_bhtd: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply half-chunk rotary embedding to `(B, H, T, D)` tensor."""
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    x1, x2 = x_bhtd.float().chunk(2, dim=-1)
    y1 = x1 * c + x2 * s
    y2 = x1 * (-s) + x2 * c
    return torch.cat((y1, y2), dim=-1).type_as(x_bhtd)


class ClassicalModdedBlock(nn.Module):
    """Single-GPU Modded-NanoGPT Transformer Block (RMSNorm + RoPE + QK-Norm SDPA + ReLU^2 MLP)."""

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        hidden_dim: int = 2048,
        value_residual_mixing: bool = True,
        has_value_embedding: bool = False,
        use_fused_qkv: bool = False,
        skip_attention: bool = False,
        input_highway_mixing: bool = True,
        residual_scale: float = 1.0,
        zero_init_residual: bool = False,
        leaderboard_exact: bool = False,
        attn_scale: float | None = None,
        use_mudd_and_gates: bool = False,
        use_partial_key_offset: bool = False,
        use_paired_heads: bool = False,
        qk_head_dim: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.qk_head_dim = int(qk_head_dim) if qk_head_dim is not None else self.head_dim
        self.hidden_dim = int(hidden_dim)
        self.leaderboard_exact = bool(leaderboard_exact)
        self.value_residual_mixing = bool(value_residual_mixing)
        self.has_value_embedding = bool(has_value_embedding)
        self.use_fused_qkv = True if self.leaderboard_exact else bool(use_fused_qkv)
        self.skip_attention = bool(skip_attention)
        self.input_highway_mixing = bool(input_highway_mixing)
        self.residual_scale = 1.0 if self.leaderboard_exact else float(residual_scale)
        self.zero_init_residual = True if self.leaderboard_exact else bool(zero_init_residual)
        self.attn_scale: float | None = (
            (0.12 if attn_scale is None else float(attn_scale))
            if self.leaderboard_exact
            else (float(attn_scale) if attn_scale is not None else None)
        )
        self.use_mudd_and_gates = bool(use_mudd_and_gates)
        self.use_partial_key_offset = bool(use_partial_key_offset)
        self.use_paired_heads = bool(use_paired_heads)
        self.eps = float(eps)

        if self.skip_attention:
            self.qkv_proj: nn.Linear | None = None
            self.q_proj: nn.Linear | None = None
            self.k_proj: nn.Linear | None = None
            self.v_proj: nn.Linear | None = None
            self.out_proj: nn.Linear | None = None
            self.ve_input_gate: InputChannelGate | None = None
            self.attn_output_gate: InputChannelGate | None = None
            self.register_parameter("xsa_alpha", None)
        else:
            if self.use_fused_qkv:
                self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
                if not self.leaderboard_exact:
                    with torch.no_grad():
                        for chunk_idx in range(3):
                            nn.init.orthogonal_(
                                self.qkv_proj.weight[chunk_idx * self.embed_dim : (chunk_idx + 1) * self.embed_dim]
                            )
                self.q_proj = None
                self.k_proj = None
                self.v_proj = None
                self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            else:
                self.qkv_proj = None
                self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
                self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
                self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
                nn.init.orthogonal_(self.q_proj.weight)
                nn.init.orthogonal_(self.k_proj.weight)
                nn.init.orthogonal_(self.v_proj.weight)
                self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

            if self.use_mudd_and_gates:
                self.ve_input_gate = InputChannelGate(
                    num_heads=self.num_heads, gate_channels=6, head_first=True
                )
                self.attn_output_gate = InputChannelGate(
                    num_heads=self.num_heads, gate_channels=6, head_first=True
                )
                self.xsa_alpha: nn.Parameter | None = nn.Parameter(torch.zeros(self.num_heads))
            else:
                self.ve_input_gate = None
                self.attn_output_gate = None
                self.register_parameter("xsa_alpha", None)

        self.fc1 = nn.Linear(self.embed_dim, self.hidden_dim, bias=False)
        self.fc2 = nn.Linear(self.hidden_dim, self.embed_dim, bias=False)

        if self.leaderboard_exact:
            std = 0.5 * (self.embed_dim ** -0.5)
            bound = (3.0 ** 0.5) * std
            with torch.no_grad():
                if self.qkv_proj is not None:
                    self.qkv_proj.weight.uniform_(-bound, bound)
                if self.out_proj is not None:
                    self.out_proj.weight.zero_()
                self.fc1.weight.uniform_(-bound, bound)
                self.fc2.weight.zero_()
            self.register_parameter("highway_gate", None)
            self.register_parameter("v_mix_gate", None)
            self.register_parameter("ve_mix_logit", None)
            self.block_lambdas: nn.Parameter | None = nn.Parameter(
                torch.tensor([1.0, 0.0], dtype=torch.float32)
            )
            if not self.skip_attention:
                self.sa_lambdas: nn.Parameter | None = nn.Parameter(
                    torch.tensor([0.5, 0.5], dtype=torch.float32)
                )
            else:
                self.register_parameter("sa_lambdas", None)
        else:
            nn.init.orthogonal_(self.fc1.weight)
            if self.zero_init_residual:
                if self.out_proj is not None:
                    nn.init.zeros_(self.out_proj.weight)
                nn.init.zeros_(self.fc2.weight)
            else:
                if self.out_proj is not None:
                    nn.init.orthogonal_(self.out_proj.weight, gain=0.5)
                nn.init.orthogonal_(self.fc2.weight, gain=0.5)

            if self.input_highway_mixing:
                self.highway_gate = nn.Parameter(torch.zeros(self.embed_dim))
            else:
                self.register_parameter("highway_gate", None)

            if self.value_residual_mixing and not self.skip_attention:
                self.v_mix_gate = nn.Parameter(torch.zeros(self.num_heads))
            else:
                self.register_parameter("v_mix_gate", None)

            if self.has_value_embedding and not self.skip_attention:
                self.ve_mix_logit = nn.Parameter(torch.full((self.num_heads,), -1.1))
            else:
                self.register_parameter("ve_mix_logit", None)

            self.register_parameter("block_lambdas", None)
            self.register_parameter("sa_lambdas", None)

        self.last_v_mixed: torch.Tensor | None = None
        self.last_h_in: torch.Tensor | None = None

    @property
    def attn(self) -> ClassicalModdedBlock | None:
        return None if self.skip_attention else self

    @property
    def w_qkv(self) -> nn.Linear | None:
        return self.qkv_proj

    def _project_qkv(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = h.shape
        if self.use_fused_qkv and self.qkv_proj is not None:
            q_lin, k_lin, v_lin = self.qkv_proj(h).chunk(3, dim=-1)
        else:
            assert self.q_proj is not None and self.k_proj is not None and self.v_proj is not None
            q_lin = self.q_proj(h)
            k_lin = self.k_proj(h)
            v_lin = self.v_proj(h)
        q = q_lin.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k_lin.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v_lin.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        x_0: torch.Tensor | None = None,
        v_0: torch.Tensor | None = None,
        ve: torch.Tensor | None = None,
        window_size: int = 0,
        doc_ids: torch.Tensor | None = None,
        return_v0: bool = False,
        attn_mask_4d: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of `ClassicalModdedBlock` on `(B, T, embed_dim)`."""
        if self.leaderboard_exact:
            if self.block_lambdas is not None and x_0 is not None:
                x = self.block_lambdas[0].to(x.dtype) * x + self.block_lambdas[1].to(x.dtype) * x_0
            B, T, D = x.shape
            v_raw: torch.Tensor | None = None
            if not self.skip_attention:
                h = F.rms_norm(x, (D,), eps=self.eps).type_as(x)
                q, k, v = self._project_qkv(h)
                if return_v0:
                    v_raw = v.detach().clone().contiguous() if not v.requires_grad else (v + 0.0).contiguous()
                if self.sa_lambdas is not None:
                    if ve is not None:
                        ve_heads = ve.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous().to(v.dtype)
                        if self.ve_input_gate is not None:
                            ve_heads = self.ve_input_gate(h) * ve_heads
                        v = self.sa_lambdas[0].to(v.dtype) * v + self.sa_lambdas[1].to(v.dtype) * ve_heads
                    else:
                        v = self.sa_lambdas[0].to(v.dtype) * v
                if self.use_partial_key_offset:
                    k = apply_partial_key_offset(k, preserve_norm=False)
                q = F.rms_norm(q, (self.head_dim,), eps=self.eps).type_as(q)
                k = F.rms_norm(k, (self.head_dim,), eps=self.eps).type_as(k)
                q = apply_leaderboard_rotary(q, cos, sin)
                k = apply_leaderboard_rotary(k, cos, sin)
                if not torch.compiler.is_compiling():
                    self.last_v_mixed = v.detach()
                    self.last_h_in = h.detach()
                if attn_mask_4d is not None:
                    attn_out = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_mask_4d, is_causal=False, scale=self.attn_scale
                    )
                elif window_size <= 0 and doc_ids is None:
                    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.attn_scale)
                else:
                    idx_t = torch.arange(T, device=x.device)
                    diff = idx_t[:, None] - idx_t[None, :]
                    mask_bool = diff >= 0
                    if window_size > 0:
                        mask_bool = mask_bool & (diff < int(window_size))
                    mask_4d = mask_bool.view(1, 1, T, T)
                    if doc_ids is not None:
                        mask_4d = mask_4d & (doc_ids[:, None, :, None] == doc_ids[:, None, None, :])
                    attn_out = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=mask_4d, is_causal=False, scale=self.attn_scale
                    )
                if self.use_mudd_and_gates:
                    if self.xsa_alpha is not None:
                        attn_out = apply_gated_xsa(attn_out, v, self.xsa_alpha, eps=self.eps)
                    if self.attn_output_gate is not None:
                        attn_out = self.attn_output_gate(h) * attn_out
                attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
                assert self.out_proj is not None
                x = x + self.out_proj(attn_out)
            h_mlp = F.rms_norm(x, (D,), eps=self.eps).type_as(x)
            r_mlp = F.relu(self.fc1(h_mlp))
            mlp_out = self.fc2(r_mlp * r_mlp)
            x = x + mlp_out
            if return_v0:
                assert v_raw is not None
                return x, v_raw
            return x

        if self.highway_gate is not None and x_0 is not None:
            x = x + (0.15 * torch.tanh(self.highway_gate)).to(x.dtype) * x_0

        B, T, D = x.shape
        v_raw = None

        if not self.skip_attention:
            h = F.rms_norm(x, (D,), eps=self.eps).type_as(x)
            q, k, v = self._project_qkv(h)
            if return_v0:
                v_raw = v.detach().clone().contiguous() if not v.requires_grad else (v + 0.0).contiguous()

            q = apply_unitary_rope(q, cos, sin)
            k = apply_unitary_rope(k, cos, sin)
            if self.use_partial_key_offset:
                k = apply_partial_key_offset(k, preserve_norm=False)
            q = F.rms_norm(q, (self.head_dim,), eps=self.eps).type_as(q)
            k = F.rms_norm(k, (self.head_dim,), eps=self.eps).type_as(k)

            if self.v_mix_gate is not None:
                if v_0 is not None:
                    alpha_v = (0.5 * torch.sigmoid(self.v_mix_gate)).view(1, self.num_heads, 1, 1).to(v.dtype)
                    v = (1.0 - alpha_v) * v + alpha_v * v_0.to(v.dtype)
                elif x_0 is not None and not return_v0:
                    h_x0 = F.rms_norm(x_0, (D,), eps=self.eps).type_as(x_0)
                    _, _, v_0_same = self._project_qkv(h_x0)
                    alpha_v = (0.5 * torch.sigmoid(self.v_mix_gate)).view(1, self.num_heads, 1, 1).to(v.dtype)
                    v = (1.0 - alpha_v) * v + alpha_v * v_0_same

            if ve is not None:
                ve_norm = F.rms_norm(ve, (D,), eps=self.eps).to(v.dtype)
                ve_heads = ve_norm.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
                if self.ve_input_gate is not None:
                    ve_heads = self.ve_input_gate(h) * ve_heads
                alpha_ve = (
                    (0.5 * torch.sigmoid(self.ve_mix_logit)).view(1, self.num_heads, 1, 1).to(v.dtype)
                    if self.ve_mix_logit is not None
                    else torch.tensor(0.10, device=v.device, dtype=v.dtype)
                )
                v = (1.0 - alpha_ve) * v + alpha_ve * ve_heads

            if not torch.compiler.is_compiling():
                self.last_v_mixed = v.detach()
                self.last_h_in = h.detach()

            if attn_mask_4d is not None:
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask_4d, is_causal=False, scale=self.attn_scale
                )
            elif window_size <= 0 and doc_ids is None:
                attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.attn_scale)
            else:
                idx_t = torch.arange(T, device=x.device)
                diff = idx_t[:, None] - idx_t[None, :]
                mask_bool = diff >= 0
                if window_size > 0:
                    mask_bool = mask_bool & (diff < int(window_size))
                mask_4d = mask_bool.view(1, 1, T, T)
                if doc_ids is not None:
                    mask_4d = mask_4d & (doc_ids[:, None, :, None] == doc_ids[:, None, None, :])
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask_4d, is_causal=False, scale=self.attn_scale
                )

            if self.use_mudd_and_gates:
                if self.xsa_alpha is not None:
                    attn_out = apply_gated_xsa(attn_out, v, self.xsa_alpha, eps=self.eps)
                if self.attn_output_gate is not None:
                    attn_out = self.attn_output_gate(h) * attn_out

            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
            x = x + self.residual_scale * self.out_proj(attn_out)
        elif return_v0:
            h = F.rms_norm(x, (D,), eps=self.eps).type_as(x)
            _, _, v = self._project_qkv(h)
            v_raw = v.detach().clone().contiguous() if not v.requires_grad else (v + 0.0).contiguous()

        h_mlp = F.rms_norm(x, (D,), eps=self.eps).type_as(x)
        r_mlp = F.relu(self.fc1(h_mlp))
        mlp_out = self.fc2(r_mlp * r_mlp)
        x = x + self.residual_scale * mlp_out
        if return_v0:
            assert v_raw is not None
            return x, v_raw
        return x



class ClassicalModdedGPT(nn.Module):
    """Apples-to-Apples Classical Modded-NanoGPT Language Model."""

    def __init__(
        self,
        vocab_size: int = 50304,
        embed_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 6,
        hidden_dim: int = 2048,
        block_size: int = 1024,
        max_seq_len: int | None = None,
        use_bigram_smear: bool = True,
        use_unet_interferometry: bool = True,
        unet_skip_mode: UNetSkipMode = "matrix",
        use_fused_qkv: bool = False,
        num_value_embeddings: int = 0,
        skip_attn_layers: tuple[int, ...] = (),
        zero_half_rope: bool = False,
        window_size: int = 0,
        doc_mask_attn: bool = False,
        eot_token_id: int = 50256,
        value_residual_mixing: bool = True,
        input_highway_mixing: bool = True,
        depth_scaled_residual: bool = False,
        logit_softcap: float = 30.0,
        tie_embeddings: bool = False,
        zero_init_residual_and_head: bool = False,
        true_layer0_v_residual: bool = False,
        leaderboard_exact: bool = False,
        attn_scale: float | None = None,
        use_ngram_hash: bool = False,
        ngram_vocab_size: int = 262144,
        ngram_inject_scale: float = 0.05,
        use_mudd_and_gates: bool = False,
        use_smear_gate: bool = False,
        use_partial_key_offset: bool = False,
        use_paired_heads: bool = False,
        qk_head_dim: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if max_seq_len is not None:
            block_size = int(max_seq_len)
        if unet_skip_mode not in ("matrix", "scalar"):
            raise ValueError(f"Invalid unet_skip_mode: {unet_skip_mode!r}")
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.block_size = int(block_size)
        self.leaderboard_exact = bool(leaderboard_exact)
        self.use_ngram_hash = bool(use_ngram_hash)
        self.ngram_vocab_size = int(ngram_vocab_size)
        self.ngram_inject_scale = float(ngram_inject_scale)
        self.use_mudd_and_gates = bool(use_mudd_and_gates)
        self.use_smear_gate = bool(use_smear_gate)
        self.use_partial_key_offset = bool(use_partial_key_offset)
        self.use_paired_heads = bool(use_paired_heads)
        self.qk_head_dim = qk_head_dim

        if self.leaderboard_exact:
            self.use_bigram_smear = False
            self.use_unet_interferometry = bool(use_unet_interferometry)
            self.unet_skip_mode: UNetSkipMode = "scalar"
            self.use_fused_qkv = True
            self.num_value_embeddings = (
                min(3, self.num_layers // 2)
                if int(num_value_embeddings) == 0
                else min(max(0, int(num_value_embeddings)), max(1, self.num_layers // 2))
            )
            self.skip_attn_layers: tuple[int, ...] = (
                (7,)
                if (self.num_layers >= 8 and len(skip_attn_layers) == 0)
                else tuple(int(i) for i in skip_attn_layers)
            )
            self.zero_half_rope = True
            self.depth_scaled_residual = False
            self.logit_softcap = 15.0
            self.zero_init_residual_and_head = True
            self.true_layer0_v_residual = False
            self.tie_embeddings = False
            self.attn_scale: float | None = 0.12 if attn_scale is None else float(attn_scale)
        else:
            self.use_bigram_smear = bool(use_bigram_smear)
            self.use_unet_interferometry = bool(use_unet_interferometry)
            self.unet_skip_mode = unet_skip_mode
            self.use_fused_qkv = bool(use_fused_qkv)
            self.num_value_embeddings = max(0, int(num_value_embeddings))
            self.skip_attn_layers = tuple(int(i) for i in skip_attn_layers)
            self.zero_half_rope = bool(zero_half_rope)
            self.depth_scaled_residual = bool(depth_scaled_residual)
            self.logit_softcap = float(logit_softcap)
            self.zero_init_residual_and_head = bool(zero_init_residual_and_head)
            self.true_layer0_v_residual = bool(true_layer0_v_residual)
            self.tie_embeddings = False if self.zero_init_residual_and_head else bool(tie_embeddings)
            self.attn_scale = float(attn_scale) if attn_scale is not None else None

        init_win = int(window_size)
        self._use_sliding_window = bool(init_win > 0)
        self.register_buffer("_window_size_buf", torch.tensor(init_win, dtype=torch.int32), persistent=False)
        self.doc_mask_attn = bool(doc_mask_attn)
        self.eot_token_id = int(eot_token_id)
        self.value_residual_mixing = bool(value_residual_mixing)
        self.input_highway_mixing = bool(input_highway_mixing)
        self.eps = float(eps)
        self._last_x_0: torch.Tensor | None = None
        self._last_v_0: torch.Tensor | None = None

        self.residual_scale = compute_depth_residual_scale(
            self.num_layers,
            base_layers=6,
            enabled=self.depth_scaled_residual,
        )

        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)
        if not self.leaderboard_exact:
            nn.init.normal_(self.wte.weight, mean=0.0, std=1.0 / math.sqrt(self.embed_dim))

        if self.use_ngram_hash:
            self.ngram_embed: HashedNgramEmbedding | None = HashedNgramEmbedding(
                ngram_vocab_size=self.ngram_vocab_size,
                embed_dim=self.embed_dim,
                num_layers=self.num_layers,
                inject_scale_init=self.ngram_inject_scale,
                zero_init=True,
            )
        else:
            self.ngram_embed = None

        if self.use_smear_gate:
            self.smear_gate: ForwardSmearGate | None = ForwardSmearGate(embed_dim=self.embed_dim)
        else:
            self.smear_gate = None

        if self.num_value_embeddings > 0:
            self.value_embeds: nn.ModuleList | None = nn.ModuleList(
                [nn.Embedding(self.vocab_size, self.embed_dim) for _ in range(self.num_value_embeddings)]
            )
            if not self.leaderboard_exact:
                with torch.no_grad():
                    for ve_emb in self.value_embeds:
                        ve_emb.weight.copy_(self.wte.weight)
        else:
            self.value_embeds = None

        if self.use_bigram_smear:
            self.smear_embed_gate = nn.Parameter(torch.zeros(self.embed_dim))
        else:
            self.register_parameter("smear_embed_gate", None)

        if self.leaderboard_exact:
            self.rope: UnitaryRoPE | LeaderboardRotary = LeaderboardRotary(
                head_dim=self.embed_dim // self.num_heads,
                max_seq_len=max(65536, self.block_size),
                base=1024.0,
            )
        else:
            self.rope = UnitaryRoPE(
                head_dim=self.embed_dim // self.num_heads,
                max_seq_len=self.block_size,
                zero_half_freq=self.zero_half_rope,
            )

        self.blocks = nn.ModuleList(
            [
                ClassicalModdedBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    hidden_dim=self.hidden_dim,
                    value_residual_mixing=self.value_residual_mixing,
                    has_value_embedding=(self._ve_bank_index(layer_idx) is not None),
                    use_fused_qkv=self.use_fused_qkv,
                    skip_attention=(layer_idx in self.skip_attn_layers),
                    input_highway_mixing=self.input_highway_mixing,
                    residual_scale=self.residual_scale,
                    zero_init_residual=self.zero_init_residual_and_head,
                    leaderboard_exact=self.leaderboard_exact,
                    attn_scale=self.attn_scale,
                    use_mudd_and_gates=self.use_mudd_and_gates,
                    use_partial_key_offset=self.use_partial_key_offset,
                    use_paired_heads=self.use_paired_heads,
                    qk_head_dim=self.qk_head_dim,
                    eps=self.eps,
                )
                for layer_idx in range(self.num_layers)
            ]
        )

        if self.use_mudd_and_gates:
            self.mudd_skips: nn.ModuleList | None = nn.ModuleList(
                [
                    MUDDDynamicSkip(embed_dim=self.embed_dim, num_sources=(1 if i == 0 else 2))
                    for i in range(self.num_layers)
                ]
            )
        else:
            self.mudd_skips = None


        if self.use_unet_interferometry and self.num_layers >= 2:
            n_half = self.num_layers // 2
            if self.leaderboard_exact:
                self.unet_gates: nn.ParameterList | None = None
                self.unet_scales: nn.ParameterList | None = None
                self.register_parameter("unet_lambdas", None)
                self.skip_weights: nn.Parameter | None = nn.Parameter(
                    torch.ones(n_half, dtype=torch.float32)
                )
            elif self.unet_skip_mode == "scalar":
                self.unet_gates = None
                self.unet_scales = None
                self.unet_lambdas: nn.Parameter | None = nn.Parameter(
                    torch.tensor([[1.0, 0.1]] * n_half, dtype=torch.float32)
                )
                self.register_parameter("skip_weights", None)
            else:
                self.unet_gates = nn.ParameterList(
                    [nn.Parameter(torch.eye(self.embed_dim)) for _ in range(n_half)]
                )
                self.unet_scales = nn.ParameterList(
                    [nn.Parameter(torch.zeros(self.embed_dim)) for _ in range(n_half)]
                )
                self.register_parameter("unet_lambdas", None)
                self.register_parameter("skip_weights", None)
        else:
            self.unet_gates = None
            self.unet_scales = None
            self.register_parameter("unet_lambdas", None)
            self.register_parameter("skip_weights", None)

        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)
        if self.zero_init_residual_and_head:
            nn.init.zeros_(self.lm_head.weight)
        else:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=1.0 / math.sqrt(self.embed_dim))
        if self.tie_embeddings:
            self.lm_head.weight = self.wte.weight

    @property
    def window_size(self) -> int:
        return int(self._window_size_buf.item())

    @window_size.setter
    def window_size(self, value: int) -> None:
        self.set_window_size(value)

    def set_window_size(self, window_size: int) -> None:
        """Update the sliding-window attention span in-place on `_window_size_buf` without triggering Dynamo recompilation."""
        w = int(window_size)
        self._use_sliding_window = bool(w > 0)
        self._window_size_buf.fill_(w)

    def _ve_bank_index(self, layer_idx: int) -> int | None:
        """Map layer index `l` to value embedding bank index in `[0, num_value_embeddings)`."""
        k = min(self.num_value_embeddings, self.num_layers // 2)
        if k <= 0:
            return None
        if layer_idx < k:
            return layer_idx
        if layer_idx >= self.num_layers - k:
            return layer_idx - (self.num_layers - k)
        return None

    def _get_value_embed(
        self,
        layer_idx: int,
        idx: torch.Tensor,
        target_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        bank_idx = self._ve_bank_index(layer_idx)
        if bank_idx is not None and self.value_embeds is not None:
            ve_out = self.value_embeds[bank_idx](idx)
            if target_dtype is not None and ve_out.dtype != target_dtype:
                return ve_out.to(target_dtype)
            return ve_out
        return None

    def forward_features(
        self,
        idx: torch.Tensor,
        return_layer_states: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode token indices `(B, T)` into final RMSNormed hidden states `(B, T, embed_dim)`."""
        B, T = idx.shape
        wte_x = self.wte(idx)
        if wte_x.is_cuda and torch.is_autocast_enabled("cuda"):
            wte_x = wte_x.to(torch.get_autocast_dtype("cuda"))
        if self.ngram_embed is not None:
            h_ngram = self.ngram_embed(idx)
            s0 = self.ngram_embed.layer_scales[0].to(wte_x.dtype)
            wte_x = wte_x + s0 * h_ngram.to(wte_x.dtype)
        x_0 = F.rms_norm(wte_x, (self.embed_dim,), eps=self.eps).type_as(wte_x)
        if self.smear_gate is not None:
            x_0 = self.smear_gate(x_0, spherical=False, eps=self.eps)
            x_0 = F.rms_norm(x_0, (self.embed_dim,), eps=self.eps).type_as(x_0)
        elif self.use_bigram_smear and T > 1 and self.smear_embed_gate is not None:
            x_prev = F.pad(x_0[:, :-1, :], (0, 0, 1, 0))
            eta = (0.06 * torch.tanh(self.smear_embed_gate)).to(x_0.dtype)
            x_0 = F.rms_norm(x_0 + eta * x_prev, (self.embed_dim,), eps=self.eps).type_as(x_0)

        if not torch.compiler.is_compiling():
            self._last_x_0 = x_0.detach()
        x = x_0
        prev_x = x_0
        encoder_states: list[torch.Tensor] = []
        layer_states: list[torch.Tensor] = []
        n_half = self.num_layers // 2
        cos, sin = self.rope.get_cos_sin(T)
        doc_ids = compute_document_ids(idx, eot_token_id=self.eot_token_id) if self.doc_mask_attn else None

        use_block_ckpt = bool(
            self.training
            and torch.is_grad_enabled()
            and x.is_cuda
            and (B * T >= 65536)
            and (torch.cuda.get_device_properties(x.device).total_memory < 24 * (1024**3))
        )

        if self.leaderboard_exact:
            layer_masks_4d: list[torch.Tensor | None] = [None] * self.num_layers
            if self._use_sliding_window or self.doc_mask_attn:
                idx_t = torch.arange(T, device=x.device)
                diff = idx_t[:, None] - idx_t[None, :]
                base_mask = (diff >= 0).view(1, 1, T, T)
                if self.doc_mask_attn and doc_ids is not None:
                    base_mask = base_mask & (doc_ids[:, None, :, None] == doc_ids[:, None, None, :])
                if self._use_sliding_window:
                    w_long = self._window_size_buf
                    w_short = torch.clamp_min(w_long // 2, 128)
                    long_mask_4d = base_mask & (diff < w_long).view(1, 1, T, T)
                    short_mask_4d = base_mask & (diff < w_short).view(1, 1, T, T)
                else:
                    long_mask_4d = base_mask
                    short_mask_4d = base_mask
                enc_masks = [
                    long_mask_4d if (i in (0, 4) or (n_half < 5 and i == 0)) else short_mask_4d
                    for i in range(n_half)
                ]
                dec_masks = list(reversed(enc_masks))
                layer_masks_4d = enc_masks + dec_masks
                while len(layer_masks_4d) < self.num_layers:
                    layer_masks_4d.append(long_mask_4d)

            for layer_idx, block in enumerate(self.blocks):
                x_before_block = x
                if self.use_unet_interferometry and n_half > 0 and layer_idx >= n_half:
                    dec_i = layer_idx - n_half
                    if encoder_states and self.skip_weights is not None and dec_i < self.skip_weights.numel():
                        x = x + self.skip_weights[dec_i].to(x.dtype) * encoder_states.pop()
                ve_l = self._get_value_embed(layer_idx, idx, target_dtype=x.dtype)
                mask_4d = layer_masks_4d[layer_idx]
                if use_block_ckpt:
                    from torch.utils.checkpoint import checkpoint

                    x = checkpoint(
                        block,
                        x,
                        cos,
                        sin,
                        x_0,
                        None,
                        ve_l,
                        0,
                        None,
                        False,
                        mask_4d,
                        use_reentrant=False,
                    )
                else:
                    x = block(
                        x,
                        cos=cos,
                        sin=sin,
                        x_0=x_0,
                        ve=ve_l,
                        attn_mask_4d=mask_4d,
                    )
                if self.mudd_skips is not None:
                    mudd_sources = [x_0] if layer_idx == 0 else [x_0, prev_x]
                    x = self.mudd_skips[layer_idx](x, mudd_sources, spherical=False, eps=self.eps)
                prev_x = x_before_block
                if self.use_unet_interferometry and n_half > 0 and layer_idx < n_half:
                    encoder_states.append(x)
                if return_layer_states:
                    layer_states.append(x)

            x_final = F.rms_norm(x, (self.embed_dim,), eps=self.eps).type_as(x)
            return x_final, layer_states

        win_sz = self.window_size
        v_0: torch.Tensor | None = None
        need_v0 = bool(self.true_layer0_v_residual or self.value_embeds is not None)
        for layer_idx, block in enumerate(self.blocks):
            x_before_block = x
            ve_l = self._get_value_embed(layer_idx, idx, target_dtype=x.dtype)
            if need_v0:
                if layer_idx == 0 and block.attn is not None:
                    if use_block_ckpt:
                        from torch.utils.checkpoint import checkpoint

                        x, v_0 = checkpoint(
                            block,
                            x,
                            cos,
                            sin,
                            x_0,
                            None,
                            ve_l,
                            win_sz,
                            doc_ids,
                            True,
                            use_reentrant=False,
                        )
                    else:
                        x, v_0 = block(
                            x,
                            cos=cos,
                            sin=sin,
                            x_0=x_0,
                            v_0=None,
                            ve=ve_l,
                            window_size=win_sz,
                            doc_ids=doc_ids,
                            return_v0=True,
                        )
                    if not torch.compiler.is_compiling():
                        self._last_v_0 = v_0.detach()
                else:
                    if use_block_ckpt:
                        from torch.utils.checkpoint import checkpoint

                        x = checkpoint(
                            block,
                            x,
                            cos,
                            sin,
                            x_0,
                            v_0,
                            ve_l,
                            win_sz,
                            doc_ids,
                            False,
                            use_reentrant=False,
                        )
                    else:
                        x = block(
                            x,
                            cos=cos,
                            sin=sin,
                            x_0=x_0,
                            v_0=v_0,
                            ve=ve_l,
                            window_size=win_sz,
                            doc_ids=doc_ids,
                            return_v0=False,
                        )
            else:
                if use_block_ckpt:
                    from torch.utils.checkpoint import checkpoint

                    x = checkpoint(
                        block,
                        x,
                        cos,
                        sin,
                        x_0,
                        None,
                        ve_l,
                        win_sz,
                        doc_ids,
                        False,
                        use_reentrant=False,
                    )
                else:
                    x = block(
                        x,
                        cos=cos,
                        sin=sin,
                        x_0=x_0,
                        ve=ve_l,
                        window_size=win_sz,
                        doc_ids=doc_ids,
                    )
            if self.mudd_skips is not None:
                mudd_sources = [x_0] if layer_idx == 0 else [x_0, prev_x]
                x = self.mudd_skips[layer_idx](x, mudd_sources, spherical=False, eps=self.eps)
            prev_x = x_before_block
            if self.use_unet_interferometry and n_half > 0:
                if layer_idx < n_half:
                    encoder_states.append(x)
                elif layer_idx >= self.num_layers - n_half:
                    enc_idx = self.num_layers - 1 - layer_idx
                    skip_x = encoder_states[enc_idx]
                    if self.unet_skip_mode == "scalar" and self.unet_lambdas is not None:
                        lam_res = self.unet_lambdas[enc_idx, 0].to(x.dtype)
                        lam_skip = self.unet_lambdas[enc_idx, 1].to(x.dtype)
                        x = lam_res * x + lam_skip * skip_x
                    elif self.unet_gates is not None and self.unet_scales is not None:
                        u_scale = (0.35 * torch.tanh(self.unet_scales[enc_idx])).to(x.dtype)
                        skip_rot = F.linear(skip_x, self.unet_gates[enc_idx].to(x.dtype))
                        x = x + u_scale * F.rms_norm(skip_rot, (self.embed_dim,), eps=self.eps).type_as(x)
            if return_layer_states:
                layer_states.append(x)


        x_final = F.rms_norm(x, (self.embed_dim,), eps=self.eps).type_as(x)
        return x_final, layer_states

    def compute_logits(self, x_final: torch.Tensor) -> torch.Tensor:
        """Project hidden features `(..., embed_dim)` to softcapped vocabulary logits `(..., vocab_size)`."""
        raw = F.linear(x_final, self.lm_head.weight.to(x_final.dtype))
        if self.logit_softcap > 0.0:
            return self.logit_softcap * torch.tanh(raw / self.logit_softcap)
        return raw

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_layer_states: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        """Forward pass from token indices `(B, T)` to `(logits, loss[, layer_states])`."""
        x_final, layer_states = self.forward_features(idx, return_layer_states=return_layer_states)
        logits = self.compute_logits(x_final)
        loss: torch.Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        if return_layer_states:
            return logits, loss, layer_states
        return logits, loss
