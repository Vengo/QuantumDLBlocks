"""Quantum Born Transformer Block with Subspace Projector Born Attention and ReLU^2 Channels."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..functional import l2_normalize, riemannian_center
from .activations import ReLUSquared, TwoBeamQuantumInterference
from .attention import SubspaceProjectorBornAttention
from .moe import QuantumBornMoE
from .overcomplete_two_beam import OvercompleteTwoBeamInterference
from .parameter_scaling import default_overcomplete_target_iso_scale



class QuantumBornBlock(nn.Module):
    """Quantum-Geometric Born Block operating on the unit hypersphere S^{d-1}.

    Combines:
    1. `SubspaceProjectorBornAttention` (positive-cone squared overlaps, zero Softmax),
       optionally skipped when `skip_attention=True`.
    2. Either:
       - Sparsely activated `QuantumBornMoE` projective instrument router (`num_experts > 1`),
       - Overcomplete two-beam tight-frame interferometer `OvercompleteTwoBeamInterference` (`use_overcomplete_two_beam=True`),
       - Two-beam bilinear Stiefel wave interferometer `TwoBeamQuantumInterference` (`use_two_beam_mlp=True`), or
       - Dense orthogonal Stiefel `ReLUSquared` channel (`num_experts=1, use_two_beam_mlp=False`).
    3. Strict hyperspherical residual blending `x <- l2_normalize(x + residual_scale * sublayer(x))`
       (with optional zero-mean Riemannian centering and coherent input-state highway mixing).
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        block_size: int = 1024,
        hidden_dim: int | None = None,
        mlp_expansion: float = 2.5,
        num_experts: int = 1,
        top_k_experts: int = 2,
        is_causal: bool = True,
        strict_spherical_residual: bool = True,
        riemannian_centering: bool = False,
        learnable_temperature: bool = False,
        value_residual_mixing: bool = False,
        has_value_embedding: bool = False,
        use_fused_qkv: bool = False,
        skip_attention: bool = False,
        input_highway_mixing: bool = False,
        use_qk_wave_filter: bool = False,
        use_two_beam_mlp: bool = False,
        use_overcomplete_two_beam: bool = False,
        frame_expansion: float = 2.0,
        residual_scale: float = 1.0,
        complex_phase_attention: bool = False,
        use_flash_born: bool = False,
        use_gated_spherical_residual: bool = False,
        residual_gate_init: float = 0.25,
        use_purified_attn: bool = False,
        use_mudd_and_gates: bool = False,
        use_partial_key_offset: bool = False,
        use_paired_heads: bool = False,
        qk_head_dim: int | None = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim if hidden_dim is not None else int(embed_dim * mlp_expansion)
        self.num_experts = num_experts
        self.top_k_experts = min(top_k_experts, num_experts)
        self.strict_spherical_residual = strict_spherical_residual
        self.riemannian_centering = riemannian_centering
        self.has_value_embedding = bool(has_value_embedding)
        self.use_fused_qkv = bool(use_fused_qkv)
        self.skip_attention = bool(skip_attention)
        self.input_highway_mixing = input_highway_mixing
        self.use_two_beam_mlp = use_two_beam_mlp
        self.use_overcomplete_two_beam = bool(use_overcomplete_two_beam)
        self.frame_expansion = float(frame_expansion)
        self.residual_scale = float(residual_scale)
        self.use_gated_spherical_residual = bool(use_gated_spherical_residual)
        self.residual_gate_init = float(residual_gate_init)
        self.use_purified_attn = bool(use_purified_attn)
        self.use_mudd_and_gates = bool(use_mudd_and_gates)
        self.use_partial_key_offset = bool(use_partial_key_offset)
        self.use_paired_heads = bool(use_paired_heads)
        self.qk_head_dim = qk_head_dim
        self.eps = eps

        if self.skip_attention:
            self.attn: SubspaceProjectorBornAttention | None = None
        else:
            self.attn = SubspaceProjectorBornAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                block_size=block_size,
                is_causal=is_causal,
                learnable_temperature=learnable_temperature or self.use_purified_attn,
                value_residual_mixing=value_residual_mixing,
                has_value_embedding=self.has_value_embedding,
                use_fused_qkv=self.use_fused_qkv,
                use_qk_wave_filter=use_qk_wave_filter,
                complex_phase=complex_phase_attention,
                use_flash_born=use_flash_born,
                use_purified_attn=self.use_purified_attn,
                use_mudd_and_gates=self.use_mudd_and_gates,
                use_partial_key_offset=self.use_partial_key_offset,
                use_paired_heads=self.use_paired_heads,
                qk_head_dim=self.qk_head_dim,
                eps=eps,
            )

        if self.use_gated_spherical_residual:
            if not self.skip_attention:
                self.attn_res_gate: nn.Parameter | None = nn.Parameter(torch.tensor(self.residual_gate_init))
            else:
                self.register_parameter("attn_res_gate", None)
            self.mlp_res_gate: nn.Parameter | None = nn.Parameter(torch.tensor(self.residual_gate_init))
        else:
            self.register_parameter("attn_res_gate", None)
            self.register_parameter("mlp_res_gate", None)

        if num_experts > 1:
            self.moe: QuantumBornMoE | None = QuantumBornMoE(
                embed_dim=embed_dim,
                expert_hidden_dim=self.hidden_dim,
                num_experts=num_experts,
                top_k=self.top_k_experts,
                learnable_temperature=learnable_temperature,
            )
            self.two_beam_mlp: nn.Module | None = None
            self.ff1 = None
            self.act = None
            self.ff2 = None
        elif self.use_overcomplete_two_beam:
            self.moe = None
            block_seed = int(torch.randint(0, 1_000_000, (1,)).item())
            self.two_beam_mlp = OvercompleteTwoBeamInterference(
                embed_dim=embed_dim,
                frame_expansion=self.frame_expansion,
                target_iso_scale=default_overcomplete_target_iso_scale(self.frame_expansion),
                init_steps=60,
                seed=block_seed,
                eps=eps,
            )
            self.ff1 = None
            self.act = None
            self.ff2 = None
        elif use_two_beam_mlp:
            self.moe = None
            self.two_beam_mlp = TwoBeamQuantumInterference(
                embed_dim=embed_dim,
                hidden_dim=self.hidden_dim,
                normalize_output=True,
                eps=eps,
            )
            self.ff1 = None
            self.act = None
            self.ff2 = None
        else:
            self.moe = None
            self.two_beam_mlp = None
            self.ff1 = nn.Linear(embed_dim, self.hidden_dim, bias=False)
            self.act = ReLUSquared()
            self.ff2 = nn.Linear(self.hidden_dim, embed_dim, bias=False)
            nn.init.orthogonal_(self.ff1.weight)
            nn.init.orthogonal_(self.ff2.weight)

        if input_highway_mixing:
            # Sigmoid gate initialized near ~0.12 reference-beam highway blend
            self.highway_logit = nn.Parameter(torch.tensor(-2.0))
        else:
            self.register_parameter("highway_logit", None)
        self.last_beam2: torch.Tensor | None = None

    @property
    def mlp(self) -> nn.Module | None:
        return self.two_beam_mlp

    def __setattr__(self, name: str, value: object) -> None:
        if name == "mlp" and (isinstance(value, nn.Module) or value is None):
            super().__setattr__("two_beam_mlp", value)
            return
        super().__setattr__(name, value)

    def _spherical_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.riemannian_centering:
            return riemannian_center(x, dim=-1, eps=self.eps)
        return l2_normalize(x, dim=-1, eps=self.eps)

    def _channel_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.moe is not None:
            return self.moe(x)
        if self.two_beam_mlp is not None:
            return self.two_beam_mlp(x)
        assert self.ff1 is not None and self.act is not None and self.ff2 is not None
        return self.ff2(self.act(self.ff1(x)))

    def _update_last_beam2(self, x_attn: torch.Tensor, h: torch.Tensor) -> None:
        raw_beam2 = (
            getattr(self.two_beam_mlp, "last_beam2", None)
            if self.two_beam_mlp is not None
            else (getattr(self.act, "last_beam2", None) if self.act is not None else None)
        )
        if raw_beam2 is not None:
            self.last_beam2 = self._spherical_norm(x_attn + 0.5 * raw_beam2.to(x_attn.dtype))
        else:
            self.last_beam2 = self._spherical_norm(x_attn - 0.25 * self._spherical_norm(h))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        psi_0: torch.Tensor | None = None,
        v_0: torch.Tensor | None = None,
        return_v0: bool = False,
        ve: torch.Tensor | None = None,
        window_size: int = 0,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass on S^{embed_dim - 1}."""
        if self.input_highway_mixing and psi_0 is not None and self.highway_logit is not None:
            gamma = torch.sigmoid(self.highway_logit).to(x.dtype)
            x = self._spherical_norm((1.0 - gamma) * x + gamma * psi_0)

        v0_out: torch.Tensor | None = None
        if self.strict_spherical_residual:
            if self.attn is not None:
                if return_v0:
                    a, v0_out = self.attn(
                        x,
                        cos=cos,
                        sin=sin,
                        psi_0=psi_0,
                        v_0=v_0,
                        ve=ve,
                        window_size=window_size,
                        doc_ids=doc_ids,
                        return_v=True,
                    )  # type: ignore[misc]
                else:
                    a = self.attn(
                        x,
                        cos=cos,
                        sin=sin,
                        psi_0=psi_0,
                        v_0=v_0,
                        ve=ve,
                        window_size=window_size,
                        doc_ids=doc_ids,
                        return_v=False,
                    )  # type: ignore[assignment]
                if self.use_gated_spherical_residual and self.attn_res_gate is not None:
                    a_norm = self._spherical_norm(a)
                    x = self._spherical_norm(x + self.attn_res_gate.to(x.dtype) * a_norm)
                else:
                    x = self._spherical_norm(x + self.residual_scale * a)

            x_attn = x
            h = self._channel_forward(x)
            self._update_last_beam2(x_attn, h)
            if self.use_gated_spherical_residual and self.mlp_res_gate is not None:
                h_norm = self._spherical_norm(h)
                x = self._spherical_norm(x + self.mlp_res_gate.to(x.dtype) * h_norm)
            else:
                x = self._spherical_norm(x + self.residual_scale * h)
        else:
            x_norm = self._spherical_norm(x)
            if self.attn is not None:
                if return_v0:
                    a, v0_out = self.attn(
                        x_norm,
                        cos=cos,
                        sin=sin,
                        psi_0=psi_0,
                        v_0=v_0,
                        ve=ve,
                        window_size=window_size,
                        doc_ids=doc_ids,
                        return_v=True,
                    )  # type: ignore[misc]
                else:
                    a = self.attn(
                        x_norm,
                        cos=cos,
                        sin=sin,
                        psi_0=psi_0,
                        v_0=v_0,
                        ve=ve,
                        window_size=window_size,
                        doc_ids=doc_ids,
                        return_v=False,
                    )  # type: ignore[assignment]
                x = x + self.residual_scale * a
            x_attn = self._spherical_norm(x)
            h = self._channel_forward(x_attn)
            self._update_last_beam2(x_attn, h)
            x = x + self.residual_scale * h

        if return_v0 and v0_out is not None:
            return x, v0_out
        return x
