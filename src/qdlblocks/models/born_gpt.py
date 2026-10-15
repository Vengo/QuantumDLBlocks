"""Quantum-Geometric Born GPT Sequence Transducer (Asymmetric & Symmetric Factorized Architectures)."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import l2_normalize
from ..layers.born_block import QuantumBornBlock
from ..layers.isometric import IsometricHilbertLift, UnitaryRoPE
from ..layers.measurement import ProjectivePOVMHead


ReadoutMode = Literal["asymmetric", "symmetric_dual"]


class QuantumBornGPT(nn.Module):
    """Quantum-Geometric Born GPT Architecture on S^{d-1}.

    Supports:
    - **10-Qubit Pure State Space (`embed_dim=1024`, `num_heads=16`, `num_layers=7`, `hidden_dim=2560`)**:
      ~124.14M parameters at `vocab_size=50304`, `seed_dim=128`.
    - **Asymmetric Full-Rank POVM Readout (`readout_mode="asymmetric"`)**:
      Compact seed input embedding (`seed_dim=128`) lifted via Stiefel operator `W_lift` into
      `S^{embed_dim - 1}`, paired with an un-bottlenecked full-rank POVM measurement frame
      in `R^{vocab_size x embed_dim}`.
    - **Self-Adjoint Dual POVM Readout (`readout_mode="symmetric_dual"`)**:
      Projects final contextual state back through `W_lift` into seed space and measures
      against the normalized seed embedding frame.
    - **Research Extensions**:
      - `learnable_temperature`: Learnable per-head sharpness and POVM inverse temperature.
      - `value_residual_mixing`: Coherent reference-beam mixing of initial lifted state `|psi_0>` into `V`.
      - `input_highway_mixing`: Coherent input-state skip connections into intermediate blocks.
      - `complex_phase_attention`: Splitting `R^{d_h} -> C^{d_h/2}` for phase-invariant Born overlaps.
      - `riemannian_centering`: Zero-mean centering prior to spherical projection.
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        embed_dim: int = 1024,
        seed_dim: int = 128,
        num_heads: int = 16,
        num_layers: int = 7,
        hidden_dim: int | None = 2560,
        mlp_expansion: float = 2.5,
        num_experts: int = 1,
        top_k_experts: int = 2,
        block_size: int = 1024,
        povm_temperature: float = 0.035,
        readout_mode: ReadoutMode = "asymmetric",
        strict_spherical_residual: bool = True,
        riemannian_centering: bool = False,
        learnable_temperature: bool = False,
        value_residual_mixing: bool = False,
        input_highway_mixing: bool = False,
        complex_phase_attention: bool = False,
        use_flash_born: bool = False,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seed_dim = seed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.block_size = block_size
        self.readout_mode = readout_mode
        self.eps = eps
        self.head_dim = embed_dim // num_heads

        # 1. Compact Seed Embedding + Isometric Stiefel Lift [seed_dim -> embed_dim]
        self.hilbert_lift = IsometricHilbertLift(
            vocab_size=vocab_size,
            seed_dim=seed_dim,
            embed_dim=embed_dim,
            eps=eps,
        )

        # Expose wte and W_lift aliases at top level for compatibility with notebook scripts
        self.wte = self.hilbert_lift.wte
        self.W_lift = self.hilbert_lift.W_lift

        # 2. Unitary RoPE tables
        self.rope = UnitaryRoPE(head_dim=self.head_dim, max_seq_len=block_size)

        # 3. Stack of QuantumBornBlocks
        self.blocks = nn.ModuleList(
            [
                QuantumBornBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    block_size=block_size,
                    hidden_dim=hidden_dim,
                    mlp_expansion=mlp_expansion,
                    num_experts=num_experts,
                    top_k_experts=top_k_experts,
                    is_causal=True,
                    strict_spherical_residual=strict_spherical_residual,
                    riemannian_centering=riemannian_centering,
                    learnable_temperature=learnable_temperature,
                    value_residual_mixing=value_residual_mixing,
                    input_highway_mixing=(input_highway_mixing and layer_idx > 0),
                    complex_phase_attention=complex_phase_attention,
                    use_flash_born=use_flash_born,
                    eps=eps,
                )
                for layer_idx in range(num_layers)
            ]
        )

        # 4. Output POVM Measurement Readout
        if readout_mode == "asymmetric":
            self.povm_readout: ProjectivePOVMHead | None = ProjectivePOVMHead(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                povm_temperature=povm_temperature,
                learnable_temperature=learnable_temperature,
                eps=eps,
            )
            self.povm_head = self.povm_readout.povm_head
            self._dual_inv_temp = 1.0 / povm_temperature
        elif readout_mode == "symmetric_dual":
            self.povm_readout = None
            self.register_parameter("povm_head", None)
            self._dual_inv_temp = 1.0 / povm_temperature
        else:
            raise ValueError(f"Unknown readout_mode: {readout_mode!r}")

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_layer_states: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        """Forward pass from token indices `(B, T)` to `(logits, loss)`."""
        B, T = idx.shape

        # 1. State preparation in seed space & isometric lift into full Hilbert space
        psi_0 = self.hilbert_lift(idx)  # (B, T, embed_dim)
        psi = psi_0
        layer_states: list[torch.Tensor] = []

        cos, sin = self.rope.get_cos_sin(T)
        for block in self.blocks:
            psi = block(psi, cos=cos, sin=sin, psi_0=psi_0)
            if return_layer_states:
                layer_states.append(psi)

        # 2. POVM Measurement Collapse
        if self.readout_mode == "asymmetric" and self.povm_readout is not None:
            logits = self.povm_readout(psi)
        else:
            psi_final = l2_normalize(psi, dim=-1, eps=self.eps)
            phi_seed = self.hilbert_lift.collapse_to_seed(psi_final)
            M_seed = l2_normalize(self.wte.weight, dim=1, eps=self.eps)
            amplitudes = torch.matmul(phi_seed, M_seed.t())
            logits = amplitudes * self._dual_inv_temp

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        if return_layer_states:
            return logits, loss, layer_states
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample new tokens from the QuantumBornGPT model."""
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        if was_training:
            self.train()
        return idx
