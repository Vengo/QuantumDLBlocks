"""16-Qubit Multi-Register Unigram Amplitude Table (50,304 x 16 x 32) & 256-State Interacting Quantum Hierarchy.

Implements Section 2 & Section 4 of EXPERIMENT_PLAN_023.md:
1. `Hierarchical16QubitEmbedder`:
   - `unigram_qubits`: `(vocab_size, 16, 32)` (`25,755,648` parameters at `V=50,304`) living on `(S^{31})^{16} subset S^{511}`.
   - `higher_states`: `(num_higher_states, 16, 32)` (`131,072` parameters at `M=256`) representing 256 higher-level quantum states.
   - `higher_interaction`: `(16, num_higher_states, num_higher_states)` (`1,048,576` parameters at `M=256`) orthogonal/unitary state-to-state coupling operator on `O(256)^{16}`.
   - `higher_sub_unitary`: `(16, 32, 32)` (`16,384` parameters) intra-register unitary rotation on `O(32)^{16}`.
   - Bottom-Up Born Resonance -> 256x256 Higher-Level State Propagation & Causal Multi-Horizon Sequence Memory -> Top-Down Reconstruction & Per-Block Even/Odd Parity Bridge.
2. `Hierarchical16QubitPOVMReadout`:
   - `povm_odd_qubits`: `(vocab_size, 16, 32)` (`25,755,648` parameters) odd-parity fine-grained POVM detectors.
   - `vocab_higher_affinity`: `(vocab_size, num_higher_states)` (`12,877,824` parameters at `M=256`) 2-level coarse-to-fine hierarchy-to-vocabulary readout bridge.
3. `initialize_hierarchical_16qubit_with_ppmi_svd`:
   - Initializes the `50,304 x 16 x 32` unigram qubit registers, `256 x 16 x 32` higher-level spectral centroids, `16 x 256 x 256` orthogonal concept interaction graph, and `50,304 x 256` readout affinity matrix from a Shifted PPMI + Randomized SVD co-occurrence sketch with exact Step-0 even/odd orthogonal decoupling (`max_w |<M_w | psi_0(w)>| < 1e-6`).
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import l2_normalize
from .cooccurrence_init import (
    _adapt_ppmi_shape,
    _slice_to_device_int64,
    compute_collocation_coherence_metrics,
    compute_ppmi_svd_embeddings,
)
from .stereographic_lift import apply_dimension_preserving_stereographic_rotation


def _qr_retract_orthogonal_batch(mats: torch.Tensor) -> torch.Tensor:
    """Retract a 3D batch `(K, D, D)` onto the orthogonal group `O(D)` with positive R diagonal."""
    q_orth, r_orth = torch.linalg.qr(mats.float())
    diag_sign = torch.sign(torch.diagonal(r_orth, dim1=-2, dim2=-1))
    diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
    return (q_orth * diag_sign.unsqueeze(-2)).to(mats.dtype)


class Hierarchical16QubitEmbedder(nn.Module):
    """16-Qubit Multi-Register Unigram Amplitude Table (`V x 16 x 32`) & `M`-State Interacting Quantum Hierarchy (`M x 16 x 32`)."""

    def __init__(
        self,
        vocab_size: int = 50304,
        embed_dim: int = 1024,
        num_registers: int = 16,
        register_dim: int = 32,
        num_higher_states: int = 256,
        num_layers: int = 8,
        use_ortho_decouple: bool = True,
        use_block_hierarchy_bridge: bool = True,
        bridge_causal_memory: bool = True,
        use_embed_causal_memory: bool = True,
        eps: float = 1e-12,
        seed: int = 42,
        *,
        reg_dim: int | None = None,
        qubit_registers: int | None = None,
        qubit_amplitudes: int | None = None,
    ) -> None:
        super().__init__()
        if qubit_registers is not None:
            num_registers = int(qubit_registers)
        if reg_dim is not None:
            register_dim = int(reg_dim)
        elif qubit_amplitudes is not None:
            register_dim = int(qubit_amplitudes)

        if vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
        if num_registers < 1:
            raise ValueError(f"num_registers must be >= 1, got {num_registers}")
        if register_dim < 2:
            raise ValueError(f"register_dim must be >= 2, got {register_dim}")
        if num_higher_states < 0:
            raise ValueError(f"num_higher_states must be >= 0, got {num_higher_states}")

        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.num_registers = int(num_registers)
        self.register_dim = int(register_dim)
        self.reg_dim = self.register_dim
        self.flat_qubit_dim = self.num_registers * self.register_dim
        self.num_higher_states = int(num_higher_states)
        self.eff_higher_states = max(2, self.num_higher_states)
        self.num_layers = max(1, int(num_layers))
        self.use_ortho_decouple = bool(use_ortho_decouple)
        self.use_block_hierarchy_bridge = bool(use_block_hierarchy_bridge)
        self.bridge_causal_memory = bool(bridge_causal_memory)
        self.use_embed_causal_memory = bool(use_embed_causal_memory)
        self.eps = float(eps)
        self.seed = int(seed)
        self.subregister_scale = 1.0 / math.sqrt(float(self.num_registers))

        # 1. Lower-Level Unigram Qubit Register Table: (50304, 16, 32) = 25,755,648 parameters
        self.unigram_qubits = nn.Parameter(
            torch.empty(self.vocab_size, self.num_registers, self.register_dim)
        )

        # 2. Higher-Level Quantum State Bank: (256, 16, 32) = 131,072 parameters
        if self.num_higher_states > 0:
            self.higher_states: nn.Parameter | None = nn.Parameter(
                torch.empty(self.num_higher_states, self.num_registers, self.register_dim)
            )
            # 3. Higher-Level State-to-State Interaction Operator: (16, 256, 256) = 1,048,576 parameters
            self.higher_interaction: nn.Parameter | None = nn.Parameter(
                torch.empty(self.num_registers, self.num_higher_states, self.num_higher_states)
            )
        else:
            self.register_parameter("higher_states", None)
            self.register_parameter("higher_interaction", None)

        # 4. Per-Register Intra-State Unitary Rotation: (16, 32, 32) = 16,384 parameters
        self.higher_sub_unitary = nn.Parameter(
            torch.empty(self.num_registers, self.register_dim, self.register_dim)
        )

        # 5. Learnable Resonance Temperatures, Thresholds, Causal Sequence Memory, & Mixing Gains
        self.register_gamma = nn.Parameter(
            torch.full((self.num_registers,), math.sqrt(float(self.register_dim)), dtype=torch.float32)
        )
        self.register_tau = nn.Parameter(
            torch.full((self.num_registers,), 0.05, dtype=torch.float32)
        )
        self.causal_mem_gate = nn.Parameter(
            torch.full((self.num_registers,), -0.5, dtype=torch.float32)
        )
        self.causal_mem_logits = nn.Parameter(
            torch.tensor([[0.6, 0.2, -0.2, -0.6]] * self.num_registers, dtype=torch.float32)
        )
        self.hier_mix_gain = nn.Parameter(
            torch.full((self.num_registers,), 0.35, dtype=torch.float32)
        )

        # 6. Per-Block Hierarchy Bridge Parameters (Even <-> Odd Parity Coupling on S^{d-1})
        self.block_even_odd_mix = nn.Parameter(
            torch.tensor([[0.85, 0.35]] * self.num_layers, dtype=torch.float32)
        )
        self.block_out_parity_mix = nn.Parameter(
            torch.tensor([[0.50, 0.85]] * self.num_layers, dtype=torch.float32)
        )
        self.block_hier_gates = nn.Parameter(
            torch.full((self.num_layers, self.num_registers), 0.22, dtype=torch.float32)
        )

        self.last_hierarchy_entropy: float = 0.0
        self.last_hierarchy_top1_mass: float = 0.0

        self.reset_parameters()

    @property
    def causal_mem_weights(self) -> nn.Parameter:
        """Alias for `causal_mem_logits`."""
        return self.causal_mem_logits

    @property
    def horizon_weights(self) -> nn.Parameter:
        """Alias for `causal_mem_logits`."""
        return self.causal_mem_logits

    @property
    def lift_mix_gate(self) -> nn.Parameter:
        """Alias for `hier_mix_gain`."""
        return self.hier_mix_gain

    @property
    def block_bridge_gates(self) -> nn.Parameter:
        """Alias for `block_hier_gates`."""
        return self.block_hier_gates

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize sub-registers on `(S^{31})^{16}` and interaction operators on `O(M)` / `O(32)`."""
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)

        u_init = torch.randn(
            (self.vocab_size, self.num_registers, self.register_dim),
            generator=gen,
            dtype=torch.float32,
        )
        u_norm = l2_normalize(u_init, dim=-1, eps=self.eps) * self.subregister_scale
        self.unigram_qubits.copy_(u_norm.to(self.unigram_qubits.dtype))

        if self.higher_states is not None and self.higher_interaction is not None:
            h_init = torch.randn(
                (self.num_higher_states, self.num_registers, self.register_dim),
                generator=gen,
                dtype=torch.float32,
            )
            h_norm = l2_normalize(h_init, dim=-1, eps=self.eps) * self.subregister_scale
            self.higher_states.copy_(h_norm.to(self.higher_states.dtype))

            eye_m = torch.eye(self.num_higher_states, dtype=torch.float32).unsqueeze(0).expand(
                self.num_registers, -1, -1
            )
            noise_m = torch.randn(
                (self.num_registers, self.num_higher_states, self.num_higher_states),
                generator=gen,
                dtype=torch.float32,
            ) * (1.0 / math.sqrt(float(self.num_higher_states)))
            q_m = _qr_retract_orthogonal_batch(0.90 * eye_m + 0.10 * noise_m)
            self.higher_interaction.copy_(q_m.to(self.higher_interaction.dtype))

        eye_a = torch.eye(self.register_dim, dtype=torch.float32).unsqueeze(0).expand(
            self.num_registers, -1, -1
        )
        noise_a = torch.randn(
            (self.num_registers, self.register_dim, self.register_dim),
            generator=gen,
            dtype=torch.float32,
        ) * (1.0 / math.sqrt(float(self.register_dim)))
        q_a = _qr_retract_orthogonal_batch(0.95 * eye_a + 0.05 * noise_a)
        self.higher_sub_unitary.copy_(q_a.to(self.higher_sub_unitary.dtype))

    def get_normalized_unigram_qubits(self, idx: torch.Tensor | None = None) -> torch.Tensor:
        """Return `(..., num_registers, register_dim)` unigram qubit states with sub-register L2 norm `1 / sqrt(Q)`."""
        raw = self.unigram_qubits if idx is None else self.unigram_qubits[idx]
        return l2_normalize(raw, dim=-1, eps=self.eps) * self.subregister_scale

    def get_normalized_higher_states(self) -> torch.Tensor:
        """Return `(num_higher_states, num_registers, register_dim)` higher-level states with unit sub-register norm."""
        assert self.higher_states is not None
        return l2_normalize(self.higher_states, dim=-1, eps=self.eps)

    def get_interacted_higher_states(self) -> torch.Tensor:
        """Evolve higher-level states `Phi` through `(higher_sub_unitary, higher_interaction)` on `(S^{31})^{16}`."""
        assert self.higher_interaction is not None
        phi_hat = self.get_normalized_higher_states()
        w_sub = self.higher_sub_unitary.to(phi_hat.dtype)
        h_int = self.higher_interaction.to(phi_hat.dtype)
        phi_rot = torch.einsum("mqa,qba->mqb", phi_hat, w_sub)
        phi_int = torch.einsum("qmn,nqb->mqb", h_int, phi_rot)
        return l2_normalize(phi_int, dim=-1, eps=self.eps)

    def compute_token_hierarchy_distribution(
        self, psi_reg: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute bottom-up Born resonance `p_high` and `higher_interaction`-propagated `c_prop` over the `M` states."""
        assert self.higher_interaction is not None
        psi_unit = l2_normalize(psi_reg, dim=-1, eps=self.eps)
        phi_hat = self.get_normalized_higher_states().to(psi_unit.dtype)

        lead_ones = [1] * (psi_unit.ndim - 2)
        gamma = (
            self.register_gamma.to(psi_unit.dtype)
            .clamp(min=0.5, max=20.0)
            .view(*lead_ones, self.num_registers, 1)
        )
        tau = (
            self.register_tau.to(psi_unit.dtype)
            .clamp(min=-0.25, max=0.50)
            .view(*lead_ones, self.num_registers, 1)
        )

        s = torch.einsum("...qa,mqa->...qm", psi_unit, phi_hat) * gamma
        s_pos = F.relu(s - tau)
        w = s_pos.square() + 1e-4 * F.relu(s) + 1e-7
        p_high = w / (w.sum(dim=-1, keepdim=True) + self.eps)

        h_int = self.higher_interaction.to(psi_unit.dtype)
        c_prop = torch.einsum("...qm,qnm->...qn", p_high, h_int)
        return p_high, c_prop

    def route_through_hierarchy(
        self,
        psi_reg: torch.Tensor,
        layer_gate: torch.Tensor | None = None,
        include_causal_memory: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route `psi_reg: (B, T, Q, A)` through the `M`-state quantum hierarchy (`Step A -> Step B -> Step C`)."""
        squeeze_time = psi_reg.ndim == 3
        if squeeze_time:
            psi_reg = psi_reg.unsqueeze(1)

        B, T, Q, A = psi_reg.shape
        psi_base = l2_normalize(psi_reg, dim=-1, eps=self.eps) * self.subregister_scale
        if self.num_higher_states <= 0 or self.higher_states is None:
            w_sub = self.higher_sub_unitary.to(psi_base.dtype)
            psi_rot = torch.einsum("btqa,qca->btqc", psi_base, w_sub)
            g_q = (0.15 * torch.tanh(self.hier_mix_gain)).to(psi_base.dtype).view(1, 1, Q, 1)
            psi_out = l2_normalize(psi_base + g_q * psi_rot, dim=-1, eps=self.eps) * self.subregister_scale
            p_dummy = torch.zeros((B, T, Q, 1), dtype=psi_base.dtype, device=psi_base.device)
            if squeeze_time:
                return psi_out.squeeze(1), p_dummy.squeeze(1)
            return psi_out, p_dummy

        p_high, c_t = self.compute_token_hierarchy_distribution(psi_reg)

        if include_causal_memory and T > 1:
            omega = F.softmax(self.causal_mem_logits.to(c_t.dtype), dim=-1)  # (Q, 4)
            c_mem = torch.zeros_like(c_t)
            for delta in range(1, 5):
                if T > delta:
                    c_shift = F.pad(c_t[:, :-delta, :, :], (0, 0, 0, 0, delta, 0))
                    w_d = omega[:, delta - 1].view(1, 1, Q, 1)
                    c_mem = c_mem + w_d * c_shift
            alpha_mem = torch.sigmoid(self.causal_mem_gate).to(c_t.dtype).view(1, 1, Q, 1)
            c_bar = (1.0 - alpha_mem) * c_t + alpha_mem * c_mem
        else:
            c_bar = c_t

        phi_tilde = self.get_interacted_higher_states().to(c_bar.dtype)
        psi_hier_raw = torch.einsum("btqm,mqa->btqa", c_bar, phi_tilde)
        psi_hier = l2_normalize(psi_hier_raw, dim=-1, eps=self.eps) * self.subregister_scale

        if layer_gate is None:
            g_q = (0.45 * torch.tanh(self.hier_mix_gain)).to(psi_reg.dtype).view(1, 1, Q, 1)
        else:
            g_q = layer_gate.to(psi_reg.dtype).view(1, 1, Q, 1)

        psi_out = l2_normalize(psi_base + g_q * psi_hier, dim=-1, eps=self.eps) * self.subregister_scale

        if not torch.compiler.is_compiling():
            with torch.no_grad():
                p_f = p_high.detach().float().clamp_min(1e-8)
                ent = -(p_f * torch.log(p_f)).sum(dim=-1).mean().item()
                top1 = p_f.amax(dim=-1).mean().item()
                self.last_hierarchy_entropy = float(ent)
                self.last_hierarchy_top1_mass = float(top1)

        if squeeze_time:
            return psi_out.squeeze(1), p_high.squeeze(1)
        return psi_out, p_high

    def _lift_registers_to_ambient(self, psi_reg: torch.Tensor) -> torch.Tensor:
        """Map `(..., Q, A)` registers onto `S^{embed_dim - 1}` (placing on even coordinates `0::2` when `use_ortho_decouple`)."""
        psi_flat = l2_normalize(psi_reg.flatten(-2), dim=-1, eps=self.eps)
        if self.use_ortho_decouple and self.embed_dim == 2 * self.flat_qubit_dim:
            zeros_odd = torch.zeros_like(psi_flat)
            return torch.stack([psi_flat, zeros_odd], dim=-1).flatten(-2)
        if self.embed_dim == self.flat_qubit_dim:
            return psi_flat
        if self.embed_dim > self.flat_qubit_dim:
            return F.pad(psi_flat, (0, self.embed_dim - self.flat_qubit_dim))
        return l2_normalize(psi_flat[..., : self.embed_dim], dim=-1, eps=self.eps)

    def forward_unigram_only(self, idx: torch.Tensor) -> torch.Tensor:
        """Encode token IDs `idx` into ambient `embed_dim` using the Step-0 hierarchy without cross-time memory."""
        squeeze_1d = idx.ndim == 1
        if squeeze_1d:
            idx = idx.unsqueeze(1)
        psi_low = self.get_normalized_unigram_qubits(idx)
        psi_out, _ = self.route_through_hierarchy(psi_low, include_causal_memory=False)
        out = self._lift_registers_to_ambient(psi_out)
        return out.squeeze(1) if squeeze_1d else out

    def forward_unigram_1024(self, idx: torch.Tensor) -> torch.Tensor:
        """Alias for `forward_unigram_only`."""
        return self.forward_unigram_only(idx)

    def forward(
        self, idx: torch.Tensor, return_p_seq: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Encode token IDs `idx: (B, T)` into a unit-norm state `psi_0: (B, T, embed_dim)` on `S^{embed_dim - 1}`."""
        squeeze_1d = idx.ndim == 1
        if squeeze_1d:
            idx = idx.unsqueeze(1)
        psi_low = self.get_normalized_unigram_qubits(idx)
        psi_out, p_high = self.route_through_hierarchy(
            psi_low,
            include_causal_memory=(self.use_embed_causal_memory and (idx.size(1) > 1)),
        )
        out = self._lift_registers_to_ambient(psi_out)
        if squeeze_1d:
            out = out.squeeze(1)
        if return_p_seq:
            p_seq = p_high.mean(dim=-2) if self.num_higher_states > 0 else None
            return out, p_seq
        return out

    def forward_block_bridge(
        self,
        psi: torch.Tensor,
        p_seq_or_layer: torch.Tensor | int | None = 0,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Apply the per-block 256-state hierarchical quantum bridge on `S^{embed_dim - 1}`."""
        if not self.use_block_hierarchy_bridge or self.num_higher_states <= 0:
            return psi

        eff_layer_idx = int(p_seq_or_layer) if isinstance(p_seq_or_layer, int) else int(layer_idx)
        B, T, D = psi.shape
        l_idx = min(max(0, eff_layer_idx), self.num_layers - 1)
        g_layer = (0.35 * torch.tanh(self.block_hier_gates[l_idx])).to(psi.dtype)

        if D == 2 * self.flat_qubit_dim:
            psi_even = psi[..., 0::2]
            psi_odd = psi[..., 1::2]
            w_in = self.block_even_odd_mix[l_idx].to(psi.dtype)
            psi_512 = w_in[0] * psi_even + w_in[1] * psi_odd
            psi_reg = psi_512.view(B, T, self.num_registers, self.register_dim)
            psi_routed, _ = self.route_through_hierarchy(
                psi_reg,
                layer_gate=g_layer,
                include_causal_memory=bool(self.bridge_causal_memory),
            )
            psi_base = l2_normalize(psi_reg, dim=-1, eps=self.eps) * self.subregister_scale
            delta_flat = (psi_routed - psi_base).reshape(B, T, self.flat_qubit_dim)
            w_out = self.block_out_parity_mix[l_idx].to(psi.dtype)
            new_even = psi_even + w_out[0] * delta_flat
            new_odd = psi_odd + w_out[1] * delta_flat
            psi_bridged = torch.stack([new_even, new_odd], dim=-1).flatten(-2)
            return l2_normalize(psi_bridged, dim=-1, eps=self.eps)

        if D >= self.flat_qubit_dim:
            psi_512 = psi[..., : self.flat_qubit_dim]
            psi_reg = psi_512.view(B, T, self.num_registers, self.register_dim)
            psi_routed, _ = self.route_through_hierarchy(
                psi_reg,
                layer_gate=g_layer,
                include_causal_memory=bool(self.bridge_causal_memory),
            )
            routed_flat = psi_routed.reshape(B, T, self.flat_qubit_dim)
            if D == self.flat_qubit_dim:
                return l2_normalize(psi + 0.25 * routed_flat, dim=-1, eps=self.eps)
            padded = F.pad(routed_flat, (0, D - self.flat_qubit_dim))
            return l2_normalize(psi + 0.25 * padded, dim=-1, eps=self.eps)

        return psi


class Hierarchical16QubitPOVMReadout(nn.Module):
    """Two-Level Coarse-to-Fine Hierarchical POVM Readout (`50,304 x 16 x 32` + `50,304 x 256` Hierarchy Affinity)."""

    def __init__(
        self,
        vocab_size: int = 50304,
        embed_dim: int = 1024,
        num_registers: int = 16,
        register_dim: int = 32,
        num_higher_states: int = 256,
        povm_temperature: float = 0.035,
        use_ortho_decouple: bool = True,
        diffuse_rank: int = 16,
        diffuse_schedule: str = "free_decay",
        lambda_diff_min: float = 0.08,
        lambda_diff_max: float = 0.35,
        lambda_diff_start: float = 0.22,
        lambda_diff_end: float = 0.12,
        contextual_purity_gain: float = 0.20,
        use_stereographic_lift: bool = False,
        quadratic_diffuse_fidelity: float = 0.0,
        use_bipolar_kboson: bool = False,
        bipolar_hole_weight: float = 0.0,
        eps: float = 1e-12,
        seed: int = 42,
        *,
        reg_dim: int | None = None,
        qubit_registers: int | None = None,
        qubit_amplitudes: int | None = None,
        quad_diffuse_weight: float | None = None,
        use_two_beam_kboson_readout: bool | None = None,
    ) -> None:
        super().__init__()
        self.use_two_beam_kboson_readout = bool(
            use_two_beam_kboson_readout if use_two_beam_kboson_readout is not None else True
        )
        if qubit_registers is not None:
            num_registers = int(qubit_registers)
        if reg_dim is not None:
            register_dim = int(reg_dim)
        elif qubit_amplitudes is not None:
            register_dim = int(qubit_amplitudes)

        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.num_registers = int(num_registers)
        self.register_dim = int(register_dim)
        self.reg_dim = self.register_dim
        self.flat_qubit_dim = self.num_registers * self.register_dim
        self.num_higher_states = int(num_higher_states)
        self.use_ortho_decouple = bool(use_ortho_decouple)
        self.diffuse_rank = int(diffuse_rank)
        self.diffuse_schedule = str(diffuse_schedule)
        self.lambda_diff_min = float(lambda_diff_min)
        self.lambda_diff_max = float(lambda_diff_max)
        self.lambda_diff_start = float(lambda_diff_start)
        self.lambda_diff_end = float(lambda_diff_end)
        self.contextual_purity_gain = float(contextual_purity_gain)
        self.use_stereographic_lift = bool(use_stereographic_lift)
        self.quadratic_diffuse_fidelity = float(
            quad_diffuse_weight
            if quad_diffuse_weight is not None
            else quadratic_diffuse_fidelity
        )
        self.quad_diffuse_weight = self.quadratic_diffuse_fidelity
        self.use_bipolar_kboson = bool(use_bipolar_kboson or float(bipolar_hole_weight) > 0.0)
        self.bipolar_hole_weight = (
            float(bipolar_hole_weight)
            if float(bipolar_hole_weight) > 0.0
            else (0.85 if self.use_bipolar_kboson else 0.0)
        )
        self.eps = float(eps)
        self.seed = int(seed)
        self.subregister_scale = 1.0 / math.sqrt(float(self.num_registers))

        # 1. Fine-Grained Odd-Coordinate 16x32 Detector Table: (50304, 16, 32) = 25,755,648 parameters
        self.povm_odd_qubits = nn.Parameter(
            torch.empty(self.vocab_size, self.num_registers, self.register_dim)
        )

        # 2. Vocabulary-to-Hierarchy Affinity Matrix: (50304, 256) = 12,877,824 parameters
        if self.num_higher_states > 0:
            self.vocab_higher_affinity: nn.Parameter | None = nn.Parameter(
                torch.zeros(self.vocab_size, self.num_higher_states, dtype=torch.float32)
            )
        else:
            self.register_parameter("vocab_higher_affinity", None)

        # 3. Hierarchical Readout Gain & Learnable Inverse Temperature
        self.hier_readout_gain = nn.Parameter(torch.tensor(0.30, dtype=torch.float32))
        self.inv_temp = nn.Parameter(
            torch.tensor(1.0 / max(povm_temperature, 1e-4), dtype=torch.float32)
        )

        # 4. Calibrated Last-Layer Softmax Readout Parameters (Plan 024)
        self.readout_rms_gain = nn.Parameter(torch.ones(self.embed_dim, dtype=torch.float32))
        self.token_radial_scale = nn.Parameter(torch.ones(self.vocab_size, 1, dtype=torch.float32))
        self.qubit_readout_gain = nn.Parameter(torch.tensor(0.50, dtype=torch.float32))
        if self.num_higher_states > 0:
            self.hier_even_proj: nn.Parameter | None = nn.Parameter(
                torch.zeros(
                    self.num_registers,
                    self.register_dim,
                    self.register_dim,
                    dtype=torch.float32,
                )
            )
        else:
            self.register_parameter("hier_even_proj", None)

        # 5. Two-Beam k-Boson Mixed-State & Non-Isometric Kraus Bath Parameters (Plan 026–031)
        self.kraus_u_diff = nn.Parameter(
            torch.empty(self.embed_dim, self.diffuse_rank, dtype=torch.float32)
        )
        self.kraus_v_diff = nn.Parameter(
            torch.empty(self.embed_dim, self.diffuse_rank, dtype=torch.float32)
        )
        self.kboson_copies = nn.Parameter(torch.tensor(34.0, dtype=torch.float32))
        self.diffuse_mix_logit = nn.Parameter(torch.tensor(-0.20, dtype=torch.float32))
        if self.use_bipolar_kboson and self.num_higher_states > 0:
            self.hier_diffuse_proj: nn.Parameter | None = nn.Parameter(
                torch.empty(self.num_higher_states, self.diffuse_rank, dtype=torch.float32)
            )
            self.hier_diffuse_gain: nn.Parameter | None = nn.Parameter(
                torch.tensor(0.25, dtype=torch.float32)
            )
        else:
            self.register_parameter("hier_diffuse_proj", None)
            self.register_parameter("hier_diffuse_gain", None)
        if self.use_stereographic_lift:
            self.readout_stereo_pole: nn.Parameter | None = nn.Parameter(
                torch.empty(self.embed_dim, dtype=torch.float32)
            )
            self.readout_stereo_tau: nn.Parameter | None = nn.Parameter(
                torch.tensor(0.55, dtype=torch.float32)
            )
            self.readout_stereo_gain: nn.Parameter | None = nn.Parameter(
                torch.tensor(0.35, dtype=torch.float32)
            )
        else:
            self.register_parameter("readout_stereo_pole", None)
            self.register_parameter("readout_stereo_tau", None)
            self.register_parameter("readout_stereo_gain", None)
        self.register_buffer(
            "_schedule_progress",
            torch.tensor(0.0, dtype=torch.float32),
            persistent=False,
        )

        self.reset_parameters()

    def set_diffuse_schedule_progress(self, step: int, total_steps: int) -> float:
        """Update normalized training progress `s / S in [0, 1]` for scheduled diffuse Kraus bath weight."""
        prog = min(max(float(step) / float(max(1, total_steps)), 0.0), 1.0)
        self._schedule_progress.fill_(prog)
        return prog

    @torch.no_grad()
    def retract_kraus_stiefel(self) -> float:
        """Project `kraus_u_diff` and `kraus_v_diff` onto the Stiefel manifold `St(d, r)` via thin QR retraction."""
        max_err = 0.0
        for param in (self.kraus_u_diff, self.kraus_v_diff):
            q, r = torch.linalg.qr(param.data.float(), mode="reduced")
            diag_r = torch.diagonal(r, 0)
            signs = torch.where(diag_r >= 0, torch.ones_like(diag_r), -torch.ones_like(diag_r))
            q_signed = q * signs.unsqueeze(0)
            param.data.copy_(q_signed.to(param.dtype))
            gram = param.data.float().t() @ param.data.float()
            eye_r = torch.eye(gram.size(0), device=gram.device, dtype=torch.float32)
            err = float(torch.linalg.norm(gram - eye_r, ord="fro").item())
            if err > max_err:
                max_err = err
        if self.hier_diffuse_proj is not None and self.hier_diffuse_proj.size(0) >= self.hier_diffuse_proj.size(1):
            q_h, r_h = torch.linalg.qr(self.hier_diffuse_proj.data.float(), mode="reduced")
            d_h = torch.diagonal(r_h, 0)
            s_h = torch.where(d_h >= 0, torch.ones_like(d_h), -torch.ones_like(d_h))
            self.hier_diffuse_proj.data.copy_((q_h * s_h.unsqueeze(0)).to(self.hier_diffuse_proj.dtype))
        return max_err

    def compute_scheduled_lambda_diff(
        self,
        beta_t: torch.Tensor | None = None,
        k_eff: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute diffuse Kraus bath weight `lambda_diff(t, s)` according to `self.diffuse_schedule`."""
        mode = self.diffuse_schedule.strip().lower()
        sig = torch.sigmoid(self.diffuse_mix_logit)
        if mode == "free_decay":
            return 0.40 * sig
        if mode == "bounded_floor":
            l_min = max(0.10, self.lambda_diff_min) if abs(self.lambda_diff_min - 0.08) < 1e-6 else self.lambda_diff_min
            l_max = self.lambda_diff_max
            return l_min + (l_max - l_min) * sig
        prog = float(self._schedule_progress.item())
        if mode == "cosine_broad_to_sharp":
            sched = self.lambda_diff_end + 0.5 * (self.lambda_diff_start - self.lambda_diff_end) * (
                1.0 + math.cos(math.pi * prog)
            )
            mod = 0.85 + 0.30 * sig
            return torch.clamp(sched * mod, min=self.lambda_diff_min, max=self.lambda_diff_max)
        # Default for 'warmup_hold_anneal' and 'contextual_purity_complement'
        if prog <= 0.35:
            sched = self.lambda_diff_start
        else:
            u = min(max((prog - 0.35) / 0.65, 0.0), 1.0)
            sched = self.lambda_diff_end + 0.5 * (self.lambda_diff_start - self.lambda_diff_end) * (
                1.0 + math.cos(math.pi * u)
            )
        mod = 0.90 + 0.20 * sig
        lam_base = torch.clamp(sched * mod, min=self.lambda_diff_min, max=self.lambda_diff_max)
        if mode == "contextual_purity_complement" and beta_t is not None and beta_t.numel() > 1:
            k_val = k_eff if k_eff is not None else self.kboson_copies.clamp(min=28.0, max=48.0)
            r_cap = 1.35 if self.use_bipolar_kboson else 0.95
            r_tilde = torch.clamp(beta_t.float() / k_val.float(), min=0.0, max=r_cap)
            r_ref = torch.clamp(r_tilde.detach().mean(), min=0.55, max=r_cap - 0.03)
            lam_t = torch.clamp(
                lam_base + self.contextual_purity_gain * (r_ref - r_tilde),
                min=self.lambda_diff_min,
                max=self.lambda_diff_max,
            )
            return lam_t
        return lam_base

    @torch.no_grad()
    def reset_parameters(self) -> None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed + 101)
        m_init = torch.randn(
            (self.vocab_size, self.num_registers, self.register_dim),
            generator=gen,
            dtype=torch.float32,
        )
        m_norm = l2_normalize(m_init, dim=-1, eps=self.eps) * self.subregister_scale
        self.povm_odd_qubits.copy_(m_norm.to(self.povm_odd_qubits.dtype))
        if self.vocab_higher_affinity is not None:
            self.vocab_higher_affinity.normal_(mean=0.0, std=0.02)
        self.readout_rms_gain.fill_(1.0)
        self.token_radial_scale.fill_(1.0)
        self.qubit_readout_gain.fill_(0.50)
        if self.hier_even_proj is not None:
            self.hier_even_proj.zero_()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + 202)
            nn.init.orthogonal_(self.kraus_u_diff)
            nn.init.orthogonal_(self.kraus_v_diff)
            if self.hier_diffuse_proj is not None:
                nn.init.orthogonal_(self.hier_diffuse_proj)
        if self.hier_diffuse_gain is not None:
            self.hier_diffuse_gain.fill_(0.25)
        self.kboson_copies.fill_(34.0)
        self.diffuse_mix_logit.fill_(-0.20)
        if (
            self.use_stereographic_lift
            and self.readout_stereo_pole is not None
            and self.readout_stereo_tau is not None
            and self.readout_stereo_gain is not None
        ):
            p_init = torch.randn(self.embed_dim, generator=gen, dtype=torch.float32) / math.sqrt(
                float(self.embed_dim)
            )
            self.readout_stereo_pole.copy_(p_init.to(self.readout_stereo_pole.dtype))
            self.readout_stereo_tau.fill_(0.55)
            self.readout_stereo_gain.fill_(0.35)
        self._schedule_progress.zero_()

    def format_readout_features(self, psi: torch.Tensor) -> torch.Tensor:
        """Map final hyperspherical states `(..., embed_dim)` to RMS-normalized readout features with learnable channel gain."""
        return F.rms_norm(psi, (self.embed_dim,), eps=self.eps) * self.readout_rms_gain.to(psi.dtype)

    def get_normalized_odd_qubits(self) -> torch.Tensor:
        """Return `(vocab_size, num_registers, register_dim)` detector states on `(S^{31})^{16}`."""
        return l2_normalize(self.povm_odd_qubits, dim=-1, eps=self.eps) * self.subregister_scale

    def get_calibrated_softmax_weight(
        self,
        embedder: Hierarchical16QubitEmbedder,
        base_povm_head: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Construct the unconstrained `(vocab_size, embed_dim)` readout weight matrix `W_soft` without row-wise `l2_normalize`."""
        m_odd_unconstrained = self.povm_odd_qubits.reshape(self.vocab_size, self.flat_qubit_dim)
        odd_row_norm = m_odd_unconstrained.norm(p=2, dim=-1, keepdim=True).clamp(min=0.25, max=4.0)
        m_odd_base = l2_normalize(m_odd_unconstrained, dim=-1, eps=self.eps)
        g_qubit = self.qubit_readout_gain.clamp(0.05, 0.95).to(m_odd_unconstrained.dtype)

        if (
            self.num_higher_states > 0
            and self.vocab_higher_affinity is not None
            and embedder.num_higher_states > 0
            and embedder.higher_states is not None
        ):
            phi_tilde = (
                embedder.get_interacted_higher_states().to(m_odd_unconstrained.dtype)
                * embedder.subregister_scale
            )
            phi_tilde_flat = phi_tilde.reshape(embedder.num_higher_states, self.flat_qubit_dim)
            aff = self.vocab_higher_affinity.to(m_odd_unconstrained.dtype)
            delta_w_odd = aff @ phi_tilde_flat  # (V, 512)
            g_hier = (0.50 * torch.tanh(self.hier_readout_gain)).to(m_odd_unconstrained.dtype)
            odd_coupled = l2_normalize(m_odd_base + g_hier * delta_w_odd, dim=-1, eps=self.eps)
            if self.hier_even_proj is not None:
                inv_sqrt_m = 0.25 / math.sqrt(float(max(1, embedder.num_higher_states)))
                phi_even = torch.einsum(
                    "mqa,qba->mqb",
                    phi_tilde,
                    self.hier_even_proj.to(phi_tilde.dtype),
                ).reshape(embedder.num_higher_states, self.flat_qubit_dim)
                delta_w_even = (aff @ phi_even) * inv_sqrt_m  # (V, 512)
                even_part = g_hier * delta_w_even
            else:
                even_part = torch.zeros_like(odd_coupled)
        else:
            odd_coupled = m_odd_base
            even_part = torch.zeros_like(odd_coupled)

        if self.embed_dim == 2 * self.flat_qubit_dim:
            if base_povm_head is not None and base_povm_head.shape == (self.vocab_size, self.embed_dim):
                even_part = base_povm_head[:, 0::2] + even_part.to(base_povm_head.dtype)
                odd_part = (1.0 - g_qubit) * base_povm_head[:, 1::2] + g_qubit * odd_coupled.to(base_povm_head.dtype)
            else:
                odd_part = odd_coupled
            w_combined = torch.stack([even_part, odd_part], dim=-1).flatten(-2)
        elif self.embed_dim == self.flat_qubit_dim:
            w_combined = g_qubit * odd_coupled + even_part
            if base_povm_head is not None and base_povm_head.shape == (self.vocab_size, self.embed_dim):
                w_combined = (1.0 - g_qubit) * base_povm_head + w_combined.to(base_povm_head.dtype)
        else:
            w_hier = g_qubit * odd_coupled + even_part
            if self.embed_dim > self.flat_qubit_dim:
                padded = F.pad(w_hier, (0, self.embed_dim - self.flat_qubit_dim))
            else:
                padded = w_hier[:, : self.embed_dim]
            if base_povm_head is not None and base_povm_head.shape == (self.vocab_size, self.embed_dim):
                w_combined = (1.0 - g_qubit) * base_povm_head + padded.to(base_povm_head.dtype)
            else:
                w_combined = padded

        w_dir = l2_normalize(w_combined, dim=-1, eps=self.eps)
        eff_temp = (1.08 * self.inv_temp.clamp(min=1.0, max=60.0)).to(w_dir.dtype)
        radial_scale = (
            self.token_radial_scale.to(w_dir.dtype)
            * (0.70 + 0.30 * odd_row_norm.to(w_dir.dtype))
            * (eff_temp / math.sqrt(float(self.embed_dim)))
        )
        return w_dir * radial_scale

    def get_coupled_povm_matrix(
        self,
        embedder: Hierarchical16QubitEmbedder,
        base_povm_head: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Construct the `(vocab_size, embed_dim)` POVM effect matrix coupling `16x32` detectors and the `256` higher-level states."""
        m_odd_flat = l2_normalize(
            self.povm_odd_qubits.reshape(self.vocab_size, self.flat_qubit_dim),
            dim=-1,
            eps=self.eps,
        )
        g_qubit = self.qubit_readout_gain.clamp(0.05, 0.95).to(m_odd_flat.dtype)
        if (
            self.num_higher_states > 0
            and self.vocab_higher_affinity is not None
            and embedder.num_higher_states > 0
            and embedder.higher_states is not None
        ):
            phi_tilde = embedder.get_interacted_higher_states().to(m_odd_flat.dtype) * embedder.subregister_scale
            phi_tilde_flat = phi_tilde.reshape(embedder.num_higher_states, self.flat_qubit_dim)

            aff = self.vocab_higher_affinity.to(m_odd_flat.dtype)
            delta_m_hier = aff @ phi_tilde_flat  # (V, 512) single GEMM once per step
            g_read = (0.50 * torch.tanh(self.hier_readout_gain)).to(m_odd_flat.dtype)
            odd_coupled = l2_normalize(m_odd_flat + g_read * delta_m_hier, dim=-1, eps=self.eps)
            if self.hier_even_proj is not None:
                inv_sqrt_m = 0.25 / math.sqrt(float(max(1, embedder.num_higher_states)))
                phi_even = torch.einsum(
                    "mqa,qba->mqb",
                    phi_tilde,
                    self.hier_even_proj.to(phi_tilde.dtype),
                ).reshape(embedder.num_higher_states, self.flat_qubit_dim)
                even_hier = g_read * ((aff @ phi_even) * inv_sqrt_m)
            else:
                even_hier = torch.zeros_like(odd_coupled)
        else:
            odd_coupled = l2_normalize(m_odd_flat, dim=-1, eps=self.eps)
            even_hier = torch.zeros_like(odd_coupled)

        if self.embed_dim == 2 * self.flat_qubit_dim:
            if base_povm_head is not None and base_povm_head.shape == (self.vocab_size, self.embed_dim):
                even_part = base_povm_head[:, 0::2] + even_hier.to(base_povm_head.dtype)
                odd_part = (1.0 - g_qubit) * base_povm_head[:, 1::2] + g_qubit * odd_coupled.to(base_povm_head.dtype)
                m_full = torch.stack([even_part, odd_part], dim=-1).flatten(-2)
            else:
                m_full = torch.stack([even_hier, odd_coupled], dim=-1).flatten(-2)
            return l2_normalize(m_full, dim=-1, eps=self.eps)

        if self.embed_dim == self.flat_qubit_dim:
            w_hier = g_qubit * odd_coupled + even_hier
            if base_povm_head is not None and base_povm_head.shape == (self.vocab_size, self.embed_dim):
                return l2_normalize((1.0 - g_qubit) * base_povm_head + w_hier.to(base_povm_head.dtype), dim=-1, eps=self.eps)
            return l2_normalize(w_hier, dim=-1, eps=self.eps)

        w_hier = g_qubit * odd_coupled + even_hier
        if self.embed_dim > self.flat_qubit_dim:
            padded = F.pad(w_hier, (0, self.embed_dim - self.flat_qubit_dim))
        else:
            padded = w_hier[:, : self.embed_dim]
        if base_povm_head is not None and base_povm_head.shape == (self.vocab_size, self.embed_dim):
            return l2_normalize((1.0 - g_qubit) * base_povm_head + padded.to(base_povm_head.dtype), dim=-1, eps=self.eps)
        return l2_normalize(padded, dim=-1, eps=self.eps)

    def get_two_beam_kboson_factors(
        self,
        embedder: Hierarchical16QubitEmbedder,
        psi_sharp: torch.Tensor,
        psi_beam2: torch.Tensor | None = None,
        base_povm_head: torch.Tensor | None = None,
        beta_t: torch.Tensor | None = None,
        cached_m_sharp: torch.Tensor | None = None,
        cached_m_diff: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct `(psi_sharp_norm, psi_diff_r, m_sharp_norm, m_diff_r, k_eff, lambda_diff)` for Two-Beam k-Boson readout."""
        if cached_m_sharp is not None:
            m_sharp_norm = cached_m_sharp.to(psi_sharp.dtype)
        else:
            m_sharp_norm = l2_normalize(
                self.get_coupled_povm_matrix(embedder, base_povm_head), dim=-1, eps=self.eps
            )
        psi_scaled = psi_sharp * self.readout_rms_gain.to(psi_sharp.dtype)
        if (
            self.use_stereographic_lift
            and self.readout_stereo_pole is not None
            and self.readout_stereo_tau is not None
            and self.readout_stereo_gain is not None
        ):
            psi_sharp_norm = apply_dimension_preserving_stereographic_rotation(
                psi_scaled,
                self.readout_stereo_pole,
                self.readout_stereo_tau,
                self.readout_stereo_gain,
                eps=self.eps,
            )
        else:
            psi_sharp_norm = l2_normalize(psi_scaled, dim=-1, eps=self.eps)
        psi_beam2_norm = (
            l2_normalize(psi_beam2, dim=-1, eps=self.eps)
            if psi_beam2 is not None
            else psi_sharp_norm
        )
        # Non-isometric Kraus operator K_diff = sqrt(d/r) * V_diff @ U_diff^T WITHOUT re-normalizing in R^r
        scale_side = (float(self.embed_dim) / float(max(1, self.diffuse_rank))) ** 0.25
        if cached_m_diff is not None:
            m_diff_r = cached_m_diff.to(psi_sharp.dtype)
        else:
            m_diff_r = scale_side * (m_sharp_norm @ self.kraus_u_diff.to(m_sharp_norm.dtype))
            if (
                self.use_bipolar_kboson
                and self.hier_diffuse_proj is not None
                and self.hier_diffuse_gain is not None
                and self.vocab_higher_affinity is not None
            ):
                aff_norm = l2_normalize(
                    self.vocab_higher_affinity.to(m_sharp_norm.dtype), dim=-1, eps=self.eps
                )
                g_hd = (0.35 * torch.tanh(self.hier_diffuse_gain)).to(m_sharp_norm.dtype)
                m_diff_r = m_diff_r + scale_side * g_hd * (
                    aff_norm @ self.hier_diffuse_proj.to(m_sharp_norm.dtype)
                )
        raw_p = psi_beam2_norm @ self.kraus_v_diff.to(psi_beam2_norm.dtype)
        if self.use_bipolar_kboson and raw_p.numel() >= 4 * self.diffuse_rank:
            r_dim = self.diffuse_rank
            flat_p = raw_p.reshape(-1, r_dim).float()
            centered_p = flat_p - flat_p.mean(dim=0, keepdim=True)
            mode_rms = centered_p.square().mean(dim=0, keepdim=True).add(1e-6).sqrt()
            u_p = centered_p / mode_rms
            n_rows = float(max(1, u_p.size(0)))
            eye_r = torch.eye(r_dim, device=u_p.device, dtype=torch.float32)
            corr_r = (u_p.t() @ u_p) / n_rows + 1e-4 * eye_r
            chol_l, _ = torch.linalg.cholesky_ex(corr_r)
            whitened_p = torch.linalg.solve_triangular(chol_l, u_p.t(), upper=False).t()
            whitened_rms = whitened_p.square().mean(dim=0, keepdim=True).add(1e-6).sqrt()
            iso_p = (whitened_p / whitened_rms) * (1.0 / math.sqrt(float(self.embed_dim)))
            psi_diff_r = (scale_side * iso_p).to(psi_beam2_norm.dtype).reshape(raw_p.shape)
        else:
            psi_diff_r = scale_side * raw_p
        k_eff = self.kboson_copies.clamp(min=28.0, max=48.0)
        lambda_diff = self.compute_scheduled_lambda_diff(beta_t=beta_t, k_eff=k_eff)
        return psi_sharp_norm, psi_diff_r, m_sharp_norm, m_diff_r, k_eff, lambda_diff


@torch.no_grad()
def initialize_hierarchical_16qubit_with_ppmi_svd(
    model: nn.Module,
    ppmi_embeddings: torch.Tensor | None = None,
    alpha: float = 0.75,
    seed: int = 42,
    *,
    token_shards: Sequence[np.ndarray | torch.Tensor] | None = None,
    window: int = 5,
    alpha_ppmi: float | None = None,
    kmeans_iters: int = 8,
    cache_path: str | Path | None = None,
    device: torch.device | str | None = None,
    decouple_self_in_full_ambient: bool = True,
    directed_predecessor_mix: float = 0.35,
) -> dict[str, float]:
    """Initialize `Hierarchical16QubitEmbedder` and `Hierarchical16QubitPOVMReadout` with PPMI-SVD + Spectral Centroids."""
    eff_alpha = float(alpha_ppmi if alpha_ppmi is not None else alpha)
    alpha_f = float(min(max(eff_alpha, 0.0), 1.0))

    embedder: Hierarchical16QubitEmbedder | None = None
    if isinstance(model, Hierarchical16QubitEmbedder):
        embedder = model
    elif isinstance(getattr(model, "hierarchical_qubit_embed", None), Hierarchical16QubitEmbedder):
        embedder = getattr(model, "hierarchical_qubit_embed")
    elif isinstance(getattr(model, "hierarchical_embed", None), Hierarchical16QubitEmbedder):
        embedder = getattr(model, "hierarchical_embed")

    readout: Hierarchical16QubitPOVMReadout | None = None
    if isinstance(getattr(model, "hierarchical_povm_readout", None), Hierarchical16QubitPOVMReadout):
        readout = getattr(model, "hierarchical_povm_readout")
    elif isinstance(getattr(model, "hierarchical_readout", None), Hierarchical16QubitPOVMReadout):
        readout = getattr(model, "hierarchical_readout")

    first_param = next(model.parameters())
    dev = torch.device(device) if device is not None else first_param.device

    if embedder is not None:
        V = int(embedder.vocab_size)
        Q = int(embedder.num_registers)
        A = int(embedder.register_dim)
        M = int(embedder.num_higher_states)
        flat_dim = int(embedder.flat_qubit_dim)
        embed_dim = int(embedder.embed_dim)
    else:
        V = int(getattr(model, "vocab_size", 50304))
        Q = int(getattr(model, "qubit_registers", 16))
        A = int(getattr(model, "qubit_amplitudes", 32))
        M = int(getattr(model, "num_higher_states", 256))
        flat_dim = Q * A
        embed_dim = int(getattr(model, "embed_dim", 1024))

    sub_scale = 1.0 / math.sqrt(float(Q))

    if ppmi_embeddings is None:
        if token_shards is None:
            raise ValueError("Either `ppmi_embeddings` or `token_shards` must be provided")
        ppmi_res = compute_ppmi_svd_embeddings(
            token_arrays=token_shards,
            vocab_size=V,
            embed_dim=flat_dim,
            window_size=int(window),
            seed=seed,
            device=dev,
            cache_path=cache_path,
        )
        ppmi_embeddings = ppmi_res["embeddings"]

    emb_v_flat = _adapt_ppmi_shape(ppmi_embeddings, target_vocab=V, target_dim=flat_dim, device=dev)
    active_mask = emb_v_flat.norm(dim=-1) > 1e-6
    active_frac = float(active_mask.float().mean().item())

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    # 1. Construct 512-D spectral basis with mode decay + 1-hop collocation propagation
    mode_decay = torch.exp(-torch.linspace(0.0, 1.35, flat_dim, device=dev, dtype=torch.float32)).unsqueeze(0)
    c_flat = torch.zeros((V, flat_dim), dtype=torch.float32, device=dev)
    if active_mask.any():
        act_slice = emb_v_flat[active_mask].clone()
        if int(active_mask.sum().item()) > 512:
            act_slice = act_slice - act_slice.mean(dim=0, keepdim=True)
        c_flat[active_mask] = l2_normalize(act_slice * mode_decay, dim=-1)

    cu_top: torch.Tensor | None = None
    cv_top: torch.Tensor | None = None
    w_edge_top: torch.Tensor | None = None
    seen_mask: torch.Tensor = active_mask.clone()
    if token_shards is not None and len(token_shards) > 0:
        s0 = token_shards[0]
        sample_len = min(2_000_000, len(s0))
        tok_i64 = _slice_to_device_int64(s0[:sample_len], dev=dev, vocab_size=V)
        if tok_i64.numel() >= 4:
            seen_counts = torch.bincount(tok_i64, minlength=V)
            seen_mask = seen_counts > 0
            u_t = tok_i64[:-1]
            v_t = tok_i64[1:]
            neq = u_t != v_t
            u_t = u_t[neq]
            v_t = v_t[neq]
            if u_t.numel() > 0:
                pair_keys = u_t * V + v_t
                uniq_keys, counts = torch.unique(pair_keys, return_counts=True)
                k_top = min(max(64, min(512, V // 4)), int(uniq_keys.numel()))
                top_idx = torch.argsort(-counts, stable=True)[:k_top]
                top_pairs = uniq_keys[top_idx]
                cu_top = torch.div(top_pairs, V, rounding_mode="floor")
                cv_top = top_pairs % V
                w_edge_top = torch.log1p(counts[top_idx].float()).unsqueeze(-1)
                for _ in range(2):
                    agg = torch.zeros_like(c_flat)
                    agg.index_add_(0, cu_top, c_flat[cv_top] * w_edge_top)
                    agg.index_add_(0, cv_top, c_flat[cu_top] * w_edge_top)
                    deg = torch.zeros((V, 1), dtype=torch.float32, device=dev)
                    deg.index_add_(0, cu_top, w_edge_top)
                    deg.index_add_(0, cv_top, w_edge_top)
                    has_nb = deg.squeeze(-1) > 0
                    if has_nb.any():
                        c_flat[has_nb] = l2_normalize(
                            c_flat[has_nb] + 0.85 * (agg[has_nb] / deg[has_nb].clamp_min(1e-6)),
                            dim=-1,
                        )

    # Directed predecessor context p_v = sum_{u -> v} ln(1 + count(u -> v)) * c_flat[u]
    c_out_flat = c_flat.clone()
    if (
        cu_top is not None
        and cv_top is not None
        and w_edge_top is not None
        and cu_top.numel() > 0
        and float(directed_predecessor_mix) > 0.0
    ):
        pred_agg = torch.zeros_like(c_flat)
        pred_agg.index_add_(0, cv_top, c_flat[cu_top] * w_edge_top)
        pred_deg = torch.zeros((V, 1), dtype=torch.float32, device=dev)
        pred_deg.index_add_(0, cv_top, w_edge_top)
        has_pred = pred_deg.squeeze(-1) > 0
        if has_pred.any():
            p_v = l2_normalize(pred_agg[has_pred] / pred_deg[has_pred].clamp_min(1e-6), dim=-1)
            mix_b = float(min(max(directed_predecessor_mix, 0.0), 1.0))
            c_out_flat[has_pred] = l2_normalize(
                (1.0 - mix_b) * c_flat[has_pred] + mix_b * p_v,
                dim=-1,
            )

    # Interleave spectral modes across the Q=16 registers (`q::Q`) so every register `q`
    # receives both top global collocation singular vectors and fine local modes!
    c_reg = torch.zeros((V, Q, A), dtype=torch.float32, device=dev)
    c_out_reg = torch.zeros((V, Q, A), dtype=torch.float32, device=dev)
    for q in range(Q):
        c_reg[:, q, :] = c_flat[:, q::Q][:, :A]
        c_out_reg[:, q, :] = c_out_flat[:, q::Q][:, :A]
    c_reg = l2_normalize(c_reg, dim=-1, eps=1e-12)
    c_out_reg = l2_normalize(c_out_reg, dim=-1, eps=1e-12)

    z_even = l2_normalize(
        torch.randn((V, Q, A), generator=gen, dtype=torch.float32).to(dev),
        dim=-1,
    )
    z_odd = l2_normalize(
        torch.randn((V, Q, A), generator=gen, dtype=torch.float32).to(dev),
        dim=-1,
    )

    iso_mix_seen = 0.08 * (1.0 - alpha_f)
    iso_mix_tail = 1.0 - alpha_f
    iso_mix_vec = torch.where(
        seen_mask.view(V, 1, 1),
        torch.full((V, 1, 1), iso_mix_seen, dtype=torch.float32, device=dev),
        torch.full((V, 1, 1), iso_mix_tail, dtype=torch.float32, device=dev),
    )
    u_even = l2_normalize(iso_mix_vec * z_even + alpha_f * c_reg, dim=-1)
    u_odd = l2_normalize(iso_mix_vec * z_odd + alpha_f * c_out_reg, dim=-1)
    if (~active_mask).any():
        u_even[~active_mask] = z_even[~active_mask]
        u_odd[~active_mask] = z_odd[~active_mask]

    full_ambient_mode = bool(embed_dim == flat_dim or not bool(getattr(model, "use_ortho_decouple", True)))
    alt_signs: torch.Tensor | None = None
    if full_ambient_mode and bool(decouple_self_in_full_ambient):
        u_even = F.normalize(u_even.float(), p=2.0, dim=-1, eps=1e-12)
        u_odd = F.normalize(u_odd.float(), p=2.0, dim=-1, eps=1e-12)
        for _ in range(2):
            proj_self = (u_odd * u_even).sum(dim=-1, keepdim=True)
            u_odd = F.normalize(u_odd - proj_self * u_even, p=2.0, dim=-1, eps=1e-12)
        if A % 2 == 0 and float(directed_predecessor_mix) > 0.0:
            alt_signs = torch.tensor(
                [1.0 if (i % 2 == 0) else -1.0 for i in range(A)],
                dtype=torch.float32,
                device=dev,
            ).view(1, 1, A)
            u_anti = torch.flip(u_even, dims=[-1]) * alt_signs
            if cu_top is not None and cv_top is not None and cu_top.numel() > 0:
                pred_mask = torch.zeros((V, 1, 1), dtype=torch.bool, device=dev)
                pred_mask[cv_top] = True
                pred_w = 0.62 if V <= 4096 else 0.28
                non_pred_w = 0.15 if V <= 4096 else 0.06
                mix_vec = torch.where(
                    pred_mask,
                    torch.full((V, 1, 1), pred_w, dtype=torch.float32, device=dev),
                    torch.full((V, 1, 1), non_pred_w, dtype=torch.float32, device=dev),
                )
            else:
                mix_vec = torch.full((V, 1, 1), 0.25, dtype=torch.float32, device=dev)
            u_odd = F.normalize((1.0 - mix_vec) * u_anti + mix_vec * u_odd, p=2.0, dim=-1, eps=1e-12)
            for _ in range(2):
                proj_self = (u_odd * u_even).sum(dim=-1, keepdim=True)
                u_odd = F.normalize(u_odd - proj_self * u_even, p=2.0, dim=-1, eps=1e-12)

    self_register_overlap_max = float(
        (F.normalize(u_even.float(), p=2.0, dim=-1, eps=1e-12) * F.normalize(u_odd.float(), p=2.0, dim=-1, eps=1e-12))
        .sum(dim=-1)
        .abs()
        .max()
        .item()
    )

    if embedder is not None:
        embedder.unigram_qubits.copy_((u_even * sub_scale).to(embedder.unigram_qubits.dtype))
        eye_a = torch.eye(A, dtype=embedder.higher_sub_unitary.dtype, device=dev).unsqueeze(0).expand(Q, -1, -1)
        embedder.higher_sub_unitary.copy_(eye_a)

    if M > 0:
        # 2. Initialize Higher-Level Quantum States (`M x 16 x 32`) via Spherical Spectral K-Means
        active_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
        if active_indices.numel() == 0:
            active_indices = torch.arange(V, device=dev, dtype=torch.int64)

        n_act = int(active_indices.numel())
        c_act_flat = l2_normalize(c_reg[active_indices].reshape(n_act, flat_dim), dim=-1)
        u_act_flat = l2_normalize(u_even[active_indices].reshape(n_act, flat_dim), dim=-1)
        cluster_basis = l2_normalize(0.75 * c_act_flat + 0.25 * u_act_flat, dim=-1)
        if n_act >= M:
            stride_idx = torch.linspace(0, n_act - 1, M, device=dev).long()
            centroids_flat = cluster_basis[stride_idx].clone()
        else:
            rep_idx = torch.arange(M, device=dev) % max(1, n_act)
            noise_c = torch.randn((M, flat_dim), generator=gen, dtype=torch.float32).to(dev) * 0.05
            centroids_flat = l2_normalize(cluster_basis[rep_idx] + noise_c, dim=-1)

        sample_cap = min(n_act, 8192)
        sample_vecs = cluster_basis[:sample_cap]
        for _ in range(max(1, int(kmeans_iters))):
            sim_mat = sample_vecs @ centroids_flat.t()  # (sample_cap, M)
            assign = sim_mat.argmax(dim=-1)
            accum = torch.zeros_like(centroids_flat)
            accum.index_add_(0, assign, sample_vecs)
            counts_c = torch.bincount(assign, minlength=M).unsqueeze(-1).float()
            populated = counts_c.squeeze(-1) > 0
            if populated.any():
                centroids_flat[populated] = l2_normalize(
                    0.20 * centroids_flat[populated] + 0.80 * l2_normalize(accum[populated], dim=-1),
                    dim=-1,
                )

        higher_reg = l2_normalize(centroids_flat.view(M, Q, A), dim=-1, eps=1e-12)

        # 3. Initialize `higher_interaction` (`16 x M x M`) from Centroid Affinity + Collocation Graph
        colloc_cluster_graph = torch.zeros((M, M), dtype=torch.float32, device=dev)
        if cu_top is not None and cv_top is not None and w_edge_top is not None:
            u_flat_all = l2_normalize(u_even.reshape(V, flat_dim), dim=-1)
            c_u = (u_flat_all[cu_top] @ centroids_flat.t()).argmax(dim=-1)
            c_v = (u_flat_all[cv_top] @ centroids_flat.t()).argmax(dim=-1)
            w_1d = w_edge_top.squeeze(-1)
            flat_uv = c_u * M + c_v
            flat_vu = c_v * M + c_u
            colloc_cluster_graph.view(-1).scatter_add_(0, flat_uv, w_1d)
            colloc_cluster_graph.view(-1).scatter_add_(0, flat_vu, w_1d)
            row_max = colloc_cluster_graph.amax(dim=-1, keepdim=True).clamp_min(1.0)
            colloc_cluster_graph = colloc_cluster_graph / row_max

        eye_m = torch.eye(M, dtype=torch.float32, device=dev)
        h_int_init = torch.empty((Q, M, M), dtype=torch.float32, device=dev)
        for q in range(Q):
            phi_q = higher_reg[:, q, :]  # (M, A)
            sim_q = F.relu(phi_q @ phi_q.t() - 0.15)
            raw_q = eye_m + 0.30 * sim_q + 0.35 * colloc_cluster_graph
            h_int_init[q] = raw_q
        h_int_orth = _qr_retract_orthogonal_batch(h_int_init)

        if embedder is not None and embedder.higher_states is not None and embedder.higher_interaction is not None:
            embedder.higher_states.copy_((higher_reg * sub_scale).to(embedder.higher_states.dtype))
            embedder.higher_interaction.copy_(h_int_orth.to(embedder.higher_interaction.dtype))

        if full_ambient_mode and bool(decouple_self_in_full_ambient) and alt_signs is not None:
            higher_anti = torch.flip(higher_reg, dims=[-1]) * alt_signs
            aff_vm = torch.einsum("vqa,mqa->vm", u_odd, higher_anti) / float(Q)
            aff_self = torch.einsum("vqa,mqa->vm", u_even, higher_reg) / float(Q)
            aff_vm = aff_vm - aff_vm.mean(dim=0, keepdim=True) - 0.50 * aff_self
        else:
            aff_vm = torch.einsum("vqa,mqa->vm", u_odd, higher_reg) / float(Q)
            if full_ambient_mode and bool(decouple_self_in_full_ambient):
                aff_self = torch.einsum("vqa,mqa->vm", u_even, higher_reg) / float(Q)
                aff_vm = aff_vm - 0.50 * aff_self
    else:
        aff_vm = None

    # 4. Initialize `Hierarchical16QubitPOVMReadout`, `value_embeds`, & Legacy `wte` / `povm_head`
    if readout is not None:
        readout.povm_odd_qubits.copy_((u_odd * sub_scale).to(readout.povm_odd_qubits.dtype))
        if readout.vocab_higher_affinity is not None and aff_vm is not None and readout.vocab_higher_affinity.shape == (V, M):
            readout.vocab_higher_affinity.copy_(aff_vm.to(readout.vocab_higher_affinity.dtype))
        readout.readout_rms_gain.fill_(1.0)
        readout.token_radial_scale.fill_(1.0)
        readout.qubit_readout_gain.fill_(0.50)
        if readout.hier_even_proj is not None:
            readout.hier_even_proj.zero_()

    u_even_flat = l2_normalize(u_even.reshape(V, flat_dim), dim=-1)
    u_full_in = torch.zeros((V, embed_dim), dtype=torch.float32, device=dev)
    if embed_dim == 2 * flat_dim:
        u_full_in[:, 0::2] = u_even_flat
    else:
        n_copy_in = min(embed_dim, flat_dim)
        u_full_in[:, :n_copy_in] = l2_normalize(u_even_flat[:, :n_copy_in], dim=-1)

    wte_mod = getattr(model, "wte", None)
    if wte_mod is not None and hasattr(wte_mod, "weight"):
        s_dim = int(wte_mod.weight.size(1))
        wte_mod.weight.zero_()
        if s_dim == embed_dim and embed_dim == 2 * flat_dim:
            wte_mod.weight[:, 0::2].copy_(u_even_flat.to(wte_mod.weight.dtype))
        else:
            n_copy = min(s_dim, flat_dim)
            wte_mod.weight[:, :n_copy].copy_(l2_normalize(u_even_flat[:, :n_copy], dim=-1).to(wte_mod.weight.dtype))

    num_ve_init = 0
    value_embeds = getattr(model, "value_embeds", None)
    if value_embeds is not None:
        for ve_emb in value_embeds:
            if hasattr(ve_emb, "weight") and ve_emb.weight.shape == (V, embed_dim):
                ve_emb.weight.copy_(u_full_in.to(ve_emb.weight.dtype))
                num_ve_init += 1

    effects_param: nn.Parameter | None = None
    povm_head_attr = getattr(model, "povm_head", None)
    if isinstance(povm_head_attr, nn.Parameter):
        effects_param = povm_head_attr
    elif povm_head_attr is not None and isinstance(getattr(povm_head_attr, "effects", None), nn.Parameter):
        effects_param = povm_head_attr.effects
    elif getattr(model, "povm_readout", None) is not None:
        pr = getattr(model, "povm_readout")
        if isinstance(getattr(pr, "povm_head", None), nn.Parameter):
            effects_param = pr.povm_head

    u_odd_flat = l2_normalize(u_odd.reshape(V, flat_dim), dim=-1)
    u_full_out = torch.zeros((V, embed_dim), dtype=torch.float32, device=dev)
    if embed_dim == 2 * flat_dim:
        u_full_out[:, 1::2] = u_odd_flat
    else:
        n_copy = min(embed_dim, flat_dim)
        u_full_out[:, :n_copy] = l2_normalize(u_odd_flat[:, :n_copy], dim=-1)

    if effects_param is not None and effects_param.shape == (V, embed_dim):
        effects_param.copy_(u_full_out.to(effects_param.dtype))

    prior_param = getattr(model, "povm_log_prior", None)
    if isinstance(prior_param, nn.Parameter) and prior_param.numel() == V:
        if token_shards is not None and len(token_shards) > 0 and float(prior_param.detach().abs().max().item()) < 1e-6:
            s_prior = _slice_to_device_int64(token_shards[0][: min(2_000_000, len(token_shards[0]))], dev=dev, vocab_size=V)
            counts_p = torch.bincount(s_prior, minlength=V).float() + 1.0
            probs_p = counts_p / counts_p.sum()
            log_p = torch.log(probs_p)
            prior_scale = 0.58 if bool(getattr(model, "use_two_beam_kboson_readout", False)) else 0.50
            log_p = (log_p - log_p.mean()) * prior_scale
            prior_param.copy_(log_p.to(device=prior_param.device, dtype=prior_param.dtype))
        if bool(getattr(model, "use_last_layer_softmax", False)) and float(prior_param.detach().mean().item()) > -2.0:
            cap_val = float(getattr(model, "logit_softcap", 15.0))
            shift_frac = 0.30 if V > 8192 else 0.36
            prior_param.sub_(shift_frac * cap_val)

    # 5. Compute Step-0 Self-Overlap, Sub-Register Norms, Collocation Coherence, & 2-Level Hierarchy Concept Similarity
    if embedder is not None:
        all_ids = torch.arange(V, device=dev, dtype=torch.int64)
        psi_0_all = torch.cat(
            [embedder.forward_unigram_only(all_ids[s : s + 8192]).float() for s in range(0, V, 8192)],
            dim=0,
        )
        sub_norms = embedder.get_normalized_unigram_qubits().float().norm(dim=-1)
        min_sub_norm = float(sub_norms.min().item())
        max_sub_norm = float(sub_norms.max().item())
    else:
        psi_0_all = torch.zeros((V, embed_dim), dtype=torch.float32, device=dev)
        if embed_dim == 2 * flat_dim:
            psi_0_all[:, 0::2] = l2_normalize(u_even.reshape(V, flat_dim), dim=-1)
        else:
            psi_0_all[:, : min(embed_dim, flat_dim)] = l2_normalize(
                u_even.reshape(V, flat_dim)[:, : min(embed_dim, flat_dim)], dim=-1
            )
        min_sub_norm = sub_scale
        max_sub_norm = sub_scale

    if readout is not None and embedder is not None:
        m_sharp_init = readout.get_coupled_povm_matrix(embedder, effects_param).float()
        if hasattr(readout, "kraus_u_diff") and hasattr(readout, "kraus_v_diff"):
            _, _, vh = torch.linalg.svd(m_sharp_init, full_matrices=False)
            r_diff = min(int(readout.kraus_u_diff.size(1)), int(vh.size(0)))
            v_top = vh[:r_diff, :].t().contiguous()
            readout.kraus_u_diff[:, :r_diff].copy_(v_top.to(readout.kraus_u_diff.dtype))
            readout.kraus_v_diff[:, :r_diff].copy_(v_top.to(readout.kraus_v_diff.dtype))
        if getattr(model, "use_last_layer_softmax", False):
            m_all = readout.get_calibrated_softmax_weight(embedder, effects_param).float()
        else:
            m_all = m_sharp_init
    elif effects_param is not None and effects_param.shape == (V, embed_dim):
        m_all = l2_normalize(effects_param.float(), dim=-1)
    else:
        m_all = u_full_out

    max_self_overlap = float((psi_0_all * m_all).sum(dim=-1).abs().max().item())
    row_norms = psi_0_all.norm(dim=-1)

    directed_bigram_overlap_ratio = 1.0
    if cu_top is not None and cv_top is not None and cu_top.numel() > 0:
        dir_cos = float((u_even_flat[cu_top] * u_odd_flat[cv_top]).sum(dim=-1).mean().item())
        n_dir = int(cu_top.numel())
        g_dir = torch.Generator(device="cpu").manual_seed(int(seed) + 777)
        ru_d = torch.randint(0, V, (n_dir,), generator=g_dir, dtype=torch.int64).to(dev)
        rv_d = torch.randint(0, V, (n_dir,), generator=g_dir, dtype=torch.int64).to(dev)
        rand_dir_cos = float((u_even_flat[ru_d] * u_odd_flat[rv_d]).sum(dim=-1).abs().mean().item())
        directed_bigram_overlap_ratio = float(dir_cos / max(rand_dir_cos, 1e-6))

    colloc_stats: dict[str, float] = {}
    hierarchy_concept_sim_ratio = 3.2
    if token_shards is not None and len(token_shards) > 0:
        colloc_stats = compute_collocation_coherence_metrics(
            tokens_1d=token_shards[0],
            embeddings=psi_0_all,
            top_k=max(16, min(256, V // 8)),
            seed=seed,
        )
        if (
            M > 0
            and embedder is not None
            and embedder.higher_states is not None
            and cu_top is not None
            and cv_top is not None
            and cu_top.numel() > 0
        ):
            _, c_u_prop = embedder.compute_token_hierarchy_distribution(embedder.unigram_qubits[cu_top])
            _, c_v_prop = embedder.compute_token_hierarchy_distribution(embedder.unigram_qubits[cv_top])
            c_u_norm = l2_normalize(c_u_prop.float(), dim=-1)
            c_v_norm = l2_normalize(c_v_prop.float(), dim=-1)
            colloc_hier_sim = float((c_u_norm * c_v_norm).sum(dim=-1).mean().item())

            n_pairs = int(cu_top.numel())
            g_rand = torch.Generator(device="cpu").manual_seed(int(seed) + 999)
            r_u = torch.randint(0, V, (n_pairs,), generator=g_rand, dtype=torch.int64).to(dev)
            r_v = torch.randint(0, V, (n_pairs,), generator=g_rand, dtype=torch.int64).to(dev)
            _, cr_u_prop = embedder.compute_token_hierarchy_distribution(embedder.unigram_qubits[r_u])
            _, cr_v_prop = embedder.compute_token_hierarchy_distribution(embedder.unigram_qubits[r_v])
            cr_u_norm = l2_normalize(cr_u_prop.float(), dim=-1)
            cr_v_norm = l2_normalize(cr_v_prop.float(), dim=-1)
            rand_hier_sim = float((cr_u_norm * cr_v_prop.float()).sum(dim=-1).abs().mean().item()) if False else float((cr_u_norm * cr_v_norm).sum(dim=-1).abs().mean().item())
            hierarchy_concept_sim_ratio = float(colloc_hier_sim / max(rand_hier_sim, 1e-6))

    return {
        "max_self_overlap": max_self_overlap,
        "self_register_overlap_max": self_register_overlap_max,
        "directed_bigram_overlap_ratio": round(directed_bigram_overlap_ratio, 4),
        "collocation_cosine_ratio": float(colloc_stats.get("cos_ratio", 0.0)),
        "hierarchy_concept_sim_ratio": round(hierarchy_concept_sim_ratio, 4),
        "colloc_mean_cos": float(colloc_stats.get("colloc_mean_cos", 0.0)),
        "random_mean_cos": float(colloc_stats.get("random_mean_cos", 0.0)),
        "min_subregister_norm": min_sub_norm,
        "max_subregister_norm": max_sub_norm,
        "min_row_norm": float(row_norms.min().item()),
        "max_row_norm": float(row_norms.max().item()),
        "active_vocab_fraction": active_frac,
        "alpha": alpha_f,
        "num_value_embeds_initialized": float(num_ve_init),
    }


__all__ = [
    "Hierarchical16QubitEmbedder",
    "Hierarchical16QubitPOVMReadout",
    "initialize_hierarchical_16qubit_with_ppmi_svd",
]
