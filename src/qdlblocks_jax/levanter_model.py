"""Levanter-compatible configuration and model wrapper for HaxQuantumBornGPT."""

from __future__ import annotations

from dataclasses import dataclass

import haliax as hax
import jax

from .haliax_modules import HaxQuantumBornGPT

try:
    import levanter  # type: ignore[import-not-found]

    HAS_LEVANTER = True
except ImportError:
    HAS_LEVANTER = False

__all__ = [
    "HAS_LEVANTER",
    "HaxQuantumBornLMConfig",
    "HaxQuantumBornLM",
]

HaxQuantumBornLM = HaxQuantumBornGPT


@dataclass(frozen=True)
class HaxQuantumBornLMConfig:
    """Configuration for building a Haliax/Levanter HaxQuantumBornGPT model."""

    vocab_size: int = 32000
    seq_len: int = 256
    seed_dim: int = 64
    embed_dim: int = 256
    num_heads: int = 4
    mlp_dim: int = 1024
    num_layers: int = 4
    num_experts: int | None = None
    top_k_experts: int = 2
    povm_temperature: float = 0.035
    use_flash_born: bool = True

    def build(
        self,
        Vocab: hax.Axis | None = None,
        *,
        key: jax.Array,
    ) -> HaxQuantumBornGPT:
        """Instantiate a HaxQuantumBornGPT from this configuration."""
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})."
            )
        vocab_ax = Vocab if Vocab is not None else hax.Axis("vocab", self.vocab_size)
        seq_ax = hax.Axis("seq", self.seq_len)
        seed_ax = hax.Axis("seed", self.seed_dim)
        hilbert_ax = hax.Axis("hilbert", self.embed_dim)
        heads_ax = hax.Axis("heads", self.num_heads)
        head_dim_ax = hax.Axis("head_dim", self.embed_dim // self.num_heads)
        mlp_ax = hax.Axis("mlp", self.mlp_dim)
        experts_ax = (
            hax.Axis("experts", self.num_experts)
            if (self.num_experts is not None and self.num_experts > 1)
            else None
        )
        return HaxQuantumBornGPT(
            Vocab=vocab_ax,
            Seq=seq_ax,
            Seed=seed_ax,
            Hilbert=hilbert_ax,
            Heads=heads_ax,
            HeadDim=head_dim_ax,
            MlpDim=mlp_ax,
            num_layers=self.num_layers,
            Experts=experts_ax,
            top_k_experts=self.top_k_experts,
            povm_temperature=self.povm_temperature,
            use_flash_born=self.use_flash_born,
            key=key,
        )
