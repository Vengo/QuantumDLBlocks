"""Matched Classical Baselines (Softmax + LayerNorm + GELU) for Multi-Domain Benchmarks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import apply_unitary_rope, l2_normalize
from ..layers.isometric import UnitaryRoPE


class _ClassicalTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        is_causal: bool = False,
        use_rope: bool = False,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_causal = is_causal
        self.use_rope = use_rope

        self.ln1 = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.act = nn.GELU()
        self.ff2 = nn.Linear(hidden_dim, embed_dim, bias=False)

        self.rope = UnitaryRoPE(head_dim=self.head_dim, max_seq_len=max_seq_len) if use_rope else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        h = self.ln1(x)
        q = self.q_proj(h).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            cos, sin = self.rope.get_cos_sin(seq_len)
            q = apply_unitary_rope(q, cos, sin)
            k = apply_unitary_rope(k, cos, sin)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, self.embed_dim)
        x = x + self.o_proj(attn_out)
        x = x + self.ff2(self.act(self.ff1(self.ln2(x))))
        return x


class ClassicalViTBaseline(nn.Module):
    """Matched Classical Vision Transformer (LayerNorm + Softmax + GELU)."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        seed_dim: int = 64,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_expansion: float = 2.5,
    ) -> None:
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        hidden_dim = int(embed_dim * mlp_expansion)

        self.patch_proj = nn.Conv2d(
            in_chans, seed_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.w_lift = nn.Linear(seed_dim, embed_dim, bias=False)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                _ClassicalTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    is_causal=False,
                    use_rope=False,
                    max_seq_len=max(64, num_patches),
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_proj(x).flatten(2).transpose(1, 2)
        h = self.w_lift(patches) + self.pos_embed[:, : patches.size(1), :]
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h).mean(dim=1)
        return self.head(h)


class MagnitudeMelAudioBaseline(nn.Module):
    """Classical Audio Baseline that discards complex STFT phase via |X(omega, t)|."""

    def __init__(
        self,
        n_freq_bins: int = 65,
        embed_dim: int = 24,
        num_classes: int = 10,
        num_heads: int = 4,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(n_freq_bins, embed_dim)
        self.blocks = nn.ModuleList(
            [
                _ClassicalTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_dim=2 * embed_dim,
                    is_causal=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = l2_normalize(torch.abs(x), dim=-1, eps=1e-8)
        h = self.in_proj(mag)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h).mean(dim=1)
        return self.head(h)


class ClassicalGPTBaseline(nn.Module):
    """Matched Classical GPT Baseline (LayerNorm + Softmax + GELU + RoPE)."""

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 128,
        seed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 4,
        hidden_dim: int = 320,
        block_size: int = 256,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, seed_dim)
        self.w_lift = nn.Linear(seed_dim, embed_dim, bias=False)
        self.blocks = nn.ModuleList(
            [
                _ClassicalTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    is_causal=True,
                    use_rope=True,
                    max_seq_len=block_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.w_lift(self.wte(idx))
        for blk in self.blocks:
            h = blk(h)
        logits = self.lm_head(self.ln_f(h))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss
