"""Quantum-Geometric Born GPT Sequence Transducer (Asymmetric & Symmetric Factorized Architectures)."""

from __future__ import annotations

from typing import Literal
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import l2_normalize
from ..layers.attention import compute_document_ids
from ..layers.born_block import QuantumBornBlock
from ..layers.dynamic_povm_purity import ContextualBlochPurityGate
from ..layers.early_epoch_quantum import apply_orthogonal_highway_decoupling
from ..layers.isometric import IsometricHilbertLift, UnitaryRoPE
from ..layers.measurement import ProjectivePOVMHead
from ..layers.mudd_and_gates import ForwardSmearGate, MUDDDynamicSkip
from ..layers.ngram_hash import HashedNgramEmbedding
from ..layers.parameter_scaling import compute_depth_residual_scale
from ..layers.seed_adapter import OvercompleteSeedReadoutAdapter
from ..layers.sparse_qubit_fock import (
    Sparse64QubitDensityLayer,
    Sparse64QubitNgramEmbedder,
    Sparse64QubitPOVMReadout,
)
from ..layers.hierarchical_qubit import (
    Hierarchical16QubitEmbedder,
    Hierarchical16QubitPOVMReadout,
)
from ..layers.direct_chunked_ce import (
    compute_two_beam_kboson_logits,
    direct_chunked_softcap_linear_ce,
    direct_chunked_two_beam_kboson_ce,
)
from ..layers.two_beam_seed_adapter import TwoBeamSeedReadoutAdapter


ReadoutMode = Literal["asymmetric", "symmetric_dual", "tied_stiefel", "tied_vocab"]
UNetSkipMode = Literal["matrix", "scalar"]


class QuantumBornGPT(nn.Module):
    """Quantum-Geometric Born GPT Architecture on S^{d-1}."""

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
        use_bigram_smear: bool = False,
        use_qk_wave_filter: bool = False,
        use_unet_interferometry: bool = False,
        unet_skip_mode: UNetSkipMode = "matrix",
        use_fused_qkv: bool = False,
        num_value_embeddings: int = 0,
        skip_attn_layers: tuple[int, ...] = (),
        zero_half_rope: bool = False,
        window_size: int = 0,
        doc_mask_attn: bool = False,
        eot_token_id: int = 50256,
        use_two_beam_mlp: bool = False,
        use_overcomplete_two_beam: bool = False,
        frame_expansion: float = 2.0,
        depth_scaled_residual: bool = False,
        depolarizing_jitter: float = 0.0,
        complex_phase_attention: bool = False,
        use_flash_born: bool = False,
        use_overcomplete_seed_adapter: bool = False,
        use_two_beam_seed_adapter: bool = False,
        use_dynamic_povm_beta: bool = False,
        use_ortho_decouple: bool = False,
        use_povm_log_prior: bool = False,
        use_gated_spherical_residual: bool = False,
        use_purified_attn: bool = False,
        true_layer0_v_residual: bool = False,
        use_ngram_hash: bool = False,
        ngram_vocab_size: int = 262144,
        ngram_inject_scale: float = 0.05,
        use_mudd_and_gates: bool = False,
        use_smear_gate: bool = False,
        use_partial_key_offset: bool = False,
        use_paired_heads: bool = False,
        qk_head_dim: int | None = None,
        use_sparse_64qubit: bool = False,
        num_qubits: int = 64,
        sparse_ngram_k: int = 16,
        use_sparse_density_layer: bool = False,
        density_matrix_rank: int = 4,
        use_fock_coupler: bool = True,
        use_hierarchical_16qubit: bool = False,
        qubit_registers: int = 16,
        qubit_amplitudes: int = 32,
        num_higher_states: int = 256,
        use_block_hierarchy_bridge: bool = True,
        bridge_causal_memory: bool = True,
        use_embed_causal_memory: bool = True,
        use_last_layer_softmax: bool = False,
        use_two_beam_kboson: bool = False,
        use_two_beam_kboson_readout: bool | None = None,
        diffuse_rank: int = 16,
        diffuse_bath_rank: int | None = None,
        diffuse_schedule: str = "free_decay",
        lambda_diff_min: float = 0.08,
        lambda_diff_max: float = 0.35,
        lambda_diff_start: float = 0.22,
        lambda_diff_end: float = 0.12,
        contextual_purity_gain: float = 0.20,
        use_stereographic_lift: bool = False,
        use_cosine_phase_beam_gate: bool = False,
        quadratic_diffuse_fidelity: float = 0.0,
        use_bipolar_kboson: bool = False,
        bipolar_hole_weight: float = 0.0,
        logit_softcap: float = 15.0,
        use_activation_checkpointing: bool = False,
        mtp_horizons: int = 1,
        mtp_loss_weight: float = 0.25,
        mtp_stride: int = 1,
        mtp_detach_readout: bool = True,
        mtp_decoupled_temp: bool = True,
        eps: float = 1e-12,
        *,
        use_zero_half_rope: bool | None = None,
        use_depth_scaled_residual: bool | None = None,
        quad_diffuse_weight: float | None = None,
    ) -> None:
        super().__init__()
        if use_zero_half_rope is not None:
            zero_half_rope = bool(use_zero_half_rope)
        if use_depth_scaled_residual is not None:
            depth_scaled_residual = bool(use_depth_scaled_residual)
        if mtp_horizons < 1:
            raise ValueError(f"mtp_horizons must be >= 1, got {mtp_horizons}")
        if mtp_stride < 1:
            raise ValueError(f"mtp_stride must be >= 1, got {mtp_stride}")
        if unet_skip_mode not in ("matrix", "scalar"):
            raise ValueError(f"Invalid unet_skip_mode: {unet_skip_mode!r}")
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seed_dim = seed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.block_size = block_size
        self.readout_mode = readout_mode
        self.use_bigram_smear = use_bigram_smear
        self.use_unet_interferometry = use_unet_interferometry
        self.unet_skip_mode: UNetSkipMode = unet_skip_mode
        self.use_fused_qkv = bool(use_fused_qkv)
        self.num_value_embeddings = max(0, int(num_value_embeddings))
        self.skip_attn_layers: tuple[int, ...] = tuple(int(i) for i in skip_attn_layers)
        self.zero_half_rope = bool(zero_half_rope)
        self.window_size = int(window_size)
        self.doc_mask_attn = bool(doc_mask_attn)
        self.eot_token_id = int(eot_token_id)
        self.use_overcomplete_two_beam = bool(use_overcomplete_two_beam)
        self.frame_expansion = float(frame_expansion)
        self.depth_scaled_residual = bool(depth_scaled_residual)
        self.residual_scale = compute_depth_residual_scale(
            num_layers, base_layers=6, enabled=self.depth_scaled_residual
        )
        self.depolarizing_jitter = depolarizing_jitter
        self.use_overcomplete_seed_adapter = bool(use_overcomplete_seed_adapter)
        self.use_two_beam_seed_adapter = bool(use_two_beam_seed_adapter)
        self.use_dynamic_povm_beta = bool(use_dynamic_povm_beta)
        self.use_ortho_decouple = bool(use_ortho_decouple)
        self.use_povm_log_prior = bool(use_povm_log_prior)
        self.use_gated_spherical_residual = bool(use_gated_spherical_residual)
        self.use_purified_attn = bool(use_purified_attn)
        self.true_layer0_v_residual = bool(true_layer0_v_residual)
        self.use_ngram_hash = bool(use_ngram_hash)
        self.ngram_vocab_size = int(ngram_vocab_size)
        self.ngram_inject_scale = float(ngram_inject_scale)
        self.use_mudd_and_gates = bool(use_mudd_and_gates)
        self.use_smear_gate = bool(use_smear_gate)
        self.use_partial_key_offset = bool(use_partial_key_offset)
        self.use_paired_heads = bool(use_paired_heads)
        self.qk_head_dim = qk_head_dim
        self.use_sparse_64qubit = bool(use_sparse_64qubit)
        self.num_qubits = int(num_qubits)
        self.sparse_ngram_k = int(sparse_ngram_k)
        self.use_sparse_density_layer = bool(use_sparse_density_layer)
        self.density_matrix_rank = max(1, int(density_matrix_rank))
        self.use_fock_coupler = bool(use_fock_coupler)
        self.use_hierarchical_16qubit = bool(use_hierarchical_16qubit)
        self.qubit_registers = int(qubit_registers)
        self.qubit_amplitudes = int(qubit_amplitudes)
        self.num_higher_states = int(num_higher_states)
        self.use_block_hierarchy_bridge = bool(use_block_hierarchy_bridge)
        self.bridge_causal_memory = bool(bridge_causal_memory)
        self.use_embed_causal_memory = bool(use_embed_causal_memory)
        self.use_last_layer_softmax = bool(use_last_layer_softmax)
        self.use_two_beam_kboson_readout = bool(
            use_two_beam_kboson_readout if use_two_beam_kboson_readout is not None else use_two_beam_kboson
        )
        self.use_two_beam_kboson = self.use_two_beam_kboson_readout
        self.diffuse_rank = int(diffuse_bath_rank if diffuse_bath_rank is not None else diffuse_rank)
        self.diffuse_bath_rank = self.diffuse_rank
        self.diffuse_schedule = str(diffuse_schedule)
        self.lambda_diff_min = float(lambda_diff_min)
        self.lambda_diff_max = float(lambda_diff_max)
        self.lambda_diff_start = float(lambda_diff_start)
        self.lambda_diff_end = float(lambda_diff_end)
        self.contextual_purity_gain = float(contextual_purity_gain)
        self.use_stereographic_lift = bool(use_stereographic_lift)
        self.use_cosine_phase_beam_gate = bool(use_cosine_phase_beam_gate)
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
        self.logit_softcap = float(logit_softcap)
        self.use_activation_checkpointing = bool(use_activation_checkpointing)
        if self.use_hierarchical_16qubit and embed_dim == 2 * self.qubit_registers * self.qubit_amplitudes:
            self.use_ortho_decouple = True
        self._last_mean_beta: float = float(1.0 / max(povm_temperature, 1e-4))
        self._last_kboson_copies: float = 34.0
        self._last_lambda_diff: float = 0.18
        self.mtp_horizons = mtp_horizons
        self.mtp_loss_weight = mtp_loss_weight
        self.mtp_stride = int(mtp_stride)
        self.mtp_detach_readout = bool(mtp_detach_readout)
        self.mtp_decoupled_temp = bool(mtp_decoupled_temp)
        self.eps = eps
        self.head_dim = embed_dim // num_heads
        self.last_mtp_losses: dict[str, float] = {}
        self._last_psi_0: torch.Tensor | None = None
        self._last_v_0: torch.Tensor | None = None
        self._last_psi_beam2: torch.Tensor | None = None
        self._last_density_branches: torch.Tensor | None = None
        self._last_density_weights: torch.Tensor | None = None

        # 1. Compact Seed Embedding + Isometric Stiefel Lift [seed_dim -> embed_dim]
        if self.use_sparse_64qubit:
            eff_seed_dim = min(seed_dim, self.num_qubits)
        elif self.use_hierarchical_16qubit:
            eff_seed_dim = min(seed_dim, 64)
        else:
            eff_seed_dim = seed_dim
        self.hilbert_lift = IsometricHilbertLift(
            vocab_size=vocab_size,
            seed_dim=eff_seed_dim,
            embed_dim=embed_dim,
            eps=eps,
        )

        # Expose wte, dm_seed, and W_lift aliases at top level
        self.wte = self.hilbert_lift.wte
        self.dm_seed = self.wte
        self.W_lift = self.hilbert_lift.W_lift

        if self.use_hierarchical_16qubit:
            self.hierarchical_qubit_embed: Hierarchical16QubitEmbedder | None = Hierarchical16QubitEmbedder(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_registers=self.qubit_registers,
                register_dim=self.qubit_amplitudes,
                num_higher_states=self.num_higher_states,
                num_layers=num_layers,
                use_ortho_decouple=self.use_ortho_decouple,
                use_block_hierarchy_bridge=self.use_block_hierarchy_bridge,
                bridge_causal_memory=self.bridge_causal_memory,
                use_embed_causal_memory=self.use_embed_causal_memory,
                eps=eps,
            )
            self.hierarchical_povm_readout: Hierarchical16QubitPOVMReadout | None = Hierarchical16QubitPOVMReadout(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_registers=self.qubit_registers,
                register_dim=self.qubit_amplitudes,
                num_higher_states=self.num_higher_states,
                povm_temperature=povm_temperature,
                use_ortho_decouple=self.use_ortho_decouple,
                diffuse_rank=self.diffuse_rank,
                diffuse_schedule=self.diffuse_schedule,
                lambda_diff_min=self.lambda_diff_min,
                lambda_diff_max=self.lambda_diff_max,
                lambda_diff_start=self.lambda_diff_start,
                lambda_diff_end=self.lambda_diff_end,
                contextual_purity_gain=self.contextual_purity_gain,
                use_stereographic_lift=self.use_stereographic_lift,
                quadratic_diffuse_fidelity=self.quadratic_diffuse_fidelity,
                use_bipolar_kboson=self.use_bipolar_kboson,
                bipolar_hole_weight=self.bipolar_hole_weight,
                eps=eps,
            )
        else:
            self.hierarchical_qubit_embed = None
            self.hierarchical_povm_readout = None
        self.hierarchical_embed = self.hierarchical_qubit_embed
        self.hierarchical_readout = self.hierarchical_povm_readout

        if self.use_sparse_64qubit:
            self.sparse_qubit_embed: Sparse64QubitNgramEmbedder | None = Sparse64QubitNgramEmbedder(
                vocab_size=vocab_size,
                fock_dim=embed_dim,
                num_qubits=self.num_qubits,
                sparse_k=self.sparse_ngram_k,
                ngram_vocab_size=self.ngram_vocab_size,
                num_layers=num_layers,
                eps=eps,
            )
        else:
            self.sparse_qubit_embed = None

        if self.use_sparse_64qubit or self.use_sparse_density_layer:
            self.sparse_povm_readout: Sparse64QubitPOVMReadout | None = Sparse64QubitPOVMReadout(
                vocab_size=vocab_size,
                fock_dim=embed_dim,
                num_qubits=self.num_qubits,
                povm_temperature=povm_temperature,
                eps=eps,
            )
        else:
            self.sparse_povm_readout = None
        self.sparse_density_layer: Sparse64QubitDensityLayer | None = None

        if self.use_ngram_hash and not self.use_sparse_64qubit:
            self.ngram_embed: HashedNgramEmbedding | None = HashedNgramEmbedding(
                ngram_vocab_size=self.ngram_vocab_size,
                embed_dim=embed_dim,
                num_layers=num_layers,
                inject_scale_init=self.ngram_inject_scale,
                zero_init=True,
            )
        else:
            self.ngram_embed = None

        if self.use_smear_gate:
            self.smear_gate: ForwardSmearGate | None = ForwardSmearGate(embed_dim=embed_dim)
        else:
            self.smear_gate = None

        if self.num_value_embeddings > 0:
            self.value_embeds: nn.ModuleList | None = nn.ModuleList(
                [nn.Embedding(vocab_size, embed_dim) for _ in range(self.num_value_embeddings)]
            )
            with torch.no_grad():
                lifted_init = F.linear(self.wte.weight, self.W_lift.weight)
                for ve_emb in self.value_embeds:
                    ve_emb.weight.copy_(lifted_init)
        else:
            self.value_embeds = None

        if use_bigram_smear:
            self.smear_embed_gate = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.register_parameter("smear_embed_gate", None)

        # 2. Unitary RoPE tables
        self.rope = UnitaryRoPE(
            head_dim=self.head_dim,
            max_seq_len=block_size,
            zero_half_freq=self.zero_half_rope,
        )

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
                    learnable_temperature=learnable_temperature or self.use_purified_attn,
                    value_residual_mixing=value_residual_mixing,
                    has_value_embedding=(self._ve_bank_index(layer_idx) is not None),
                    use_fused_qkv=self.use_fused_qkv,
                    skip_attention=(layer_idx in self.skip_attn_layers),
                    input_highway_mixing=(input_highway_mixing and layer_idx > 0),
                    use_qk_wave_filter=use_qk_wave_filter,
                    use_two_beam_mlp=use_two_beam_mlp and not self.use_sparse_density_layer,
                    use_overcomplete_two_beam=use_overcomplete_two_beam and not self.use_sparse_density_layer,
                    frame_expansion=self.frame_expansion,
                    residual_scale=self.residual_scale,
                    complex_phase_attention=complex_phase_attention,
                    use_flash_born=use_flash_born,
                    use_gated_spherical_residual=self.use_gated_spherical_residual,
                    use_purified_attn=self.use_purified_attn,
                    use_mudd_and_gates=self.use_mudd_and_gates,
                    use_partial_key_offset=self.use_partial_key_offset,
                    use_paired_heads=self.use_paired_heads,
                    qk_head_dim=self.qk_head_dim,
                    use_stereographic_lift=self.use_stereographic_lift,
                    use_cosine_phase_beam_gate=self.use_cosine_phase_beam_gate,
                    eps=eps,
                )
                for layer_idx in range(num_layers)
            ]
        )

        if self.use_sparse_density_layer:
            for blk in self.blocks:
                d_layer = Sparse64QubitDensityLayer(
                    fock_dim=embed_dim,
                    num_qubits=self.num_qubits,
                    density_rank=self.density_matrix_rank,
                    use_fock_coupler=self.use_fock_coupler,
                    eps=eps,
                )
                blk.ff1 = None
                blk.act = None
                blk.ff2 = None
                blk.two_beam_mlp = d_layer
                blk.sparse_density_layer = d_layer
            if len(self.blocks) > 0:
                self.sparse_density_layer = self.blocks[-1].sparse_density_layer  # type: ignore[assignment]
                if self.sparse_density_layer is not None:
                    self.sparse_density_layer._capture_branches = True

        if self.use_mudd_and_gates:
            self.mudd_skips: nn.ModuleList | None = nn.ModuleList(
                [
                    MUDDDynamicSkip(
                        embed_dim=embed_dim,
                        num_sources=(1 if i == 0 else 2),
                        spherical=True,
                    )
                    for i in range(num_layers)
                ]
            )
        else:
            self.mudd_skips = None

        if use_unet_interferometry and num_layers >= 2:
            n_half = num_layers // 2
            if self.unet_skip_mode == "scalar":
                self.unet_gates: nn.ParameterList | None = None
                self.register_parameter("unet_scales", None)
                self.unet_lambdas: nn.Parameter | None = nn.Parameter(
                    torch.tensor([[1.0, 0.1]] * n_half, dtype=torch.float32)
                )
            else:
                self.unet_gates = nn.ParameterList(
                    [nn.Parameter(torch.eye(embed_dim)) for _ in range(n_half)]
                )
                self.unet_scales = nn.Parameter(torch.full((n_half,), 0.18))
                self.register_parameter("unet_lambdas", None)
        else:
            self.unet_gates = None
            self.register_parameter("unet_scales", None)
            self.register_parameter("unet_lambdas", None)

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
            self.povm_head.effects = self.povm_head  # type: ignore[attr-defined]
            self.readout_rot = None
            self.register_parameter("povm_head_seed_delta", None)
            self.register_parameter("tied_inv_temp", None)
            self._dual_inv_temp = 1.0 / povm_temperature

        elif readout_mode == "symmetric_dual":
            self.povm_readout = None
            self.register_parameter("povm_head", None)
            self.readout_rot = None
            self.register_parameter("povm_head_seed_delta", None)
            self.register_parameter("tied_inv_temp", None)
            self._dual_inv_temp = 1.0 / povm_temperature
        elif readout_mode == "tied_stiefel":
            self.povm_readout = None
            self.register_parameter("povm_head", None)
            self.readout_rot = nn.Linear(embed_dim, embed_dim, bias=False)
            nn.init.eye_(self.readout_rot.weight)
            self.povm_head_seed_delta = nn.Parameter(torch.zeros(vocab_size, seed_dim))
            self.tied_inv_temp = nn.Parameter(torch.tensor(1.0 / max(povm_temperature, 1e-4)))
            self._dual_inv_temp = 1.0 / povm_temperature
        elif readout_mode == "tied_vocab":
            self.povm_readout = None
            self.register_parameter("povm_head", None)
            self.readout_rot = nn.Linear(embed_dim, embed_dim, bias=False)
            nn.init.eye_(self.readout_rot.weight)
            self.register_parameter("povm_head_seed_delta", None)
            self.tied_inv_temp = nn.Parameter(torch.tensor(1.0 / max(povm_temperature, 1e-4)))
            self._dual_inv_temp = 1.0 / povm_temperature
        else:
            raise ValueError(f"Unknown readout_mode: {readout_mode!r}")

        if self.use_povm_log_prior:
            self.povm_log_prior: nn.Parameter | None = nn.Parameter(torch.zeros(vocab_size))
        else:
            self.register_parameter("povm_log_prior", None)

        if self.use_two_beam_seed_adapter:
            self.seed_adapter: TwoBeamSeedReadoutAdapter | OvercompleteSeedReadoutAdapter | None = (
                TwoBeamSeedReadoutAdapter(
                    embed_dim=embed_dim,
                    seed_dim=seed_dim,
                    frame_expansion=2,
                    eps=eps,
                )
            )
        elif self.use_overcomplete_seed_adapter:
            self.seed_adapter = OvercompleteSeedReadoutAdapter(
                embed_dim=embed_dim,
                seed_dim=seed_dim,
                frame_expansion=2,
                eps=eps,
            )
        else:
            self.seed_adapter = None

        if self.use_dynamic_povm_beta:
            self.purity_gate: ContextualBlochPurityGate | None = ContextualBlochPurityGate(
                embed_dim=embed_dim,
                use_two_beam_kboson=self.use_two_beam_kboson_readout,
            )
        else:
            self.purity_gate = None

        # 5. Optional Multi-Token Quantum POVM Prediction (MTP) Stiefel Rotation Heads U_{Delta t}
        if mtp_horizons > 1:
            mtp_dim = embed_dim if readout_mode == "asymmetric" else seed_dim
            mtp_list = []
            for _ in range(mtp_horizons - 1):
                u_mat = torch.empty(mtp_dim, mtp_dim)
                nn.init.orthogonal_(u_mat)
                mtp_list.append(nn.Parameter(u_mat))
            self.mtp_rotations: nn.ParameterList | None = nn.ParameterList(mtp_list)
        else:
            self.mtp_rotations = None

        if mtp_horizons > 1 and mtp_decoupled_temp:
            base_t = max(povm_temperature, 1e-4)
            init_betas = [1.0 / (base_t * math.sqrt(float(h))) for h in range(2, mtp_horizons + 1)]
            self.mtp_inv_temps: nn.Parameter | None = nn.Parameter(torch.tensor(init_betas, dtype=torch.float32))
        else:
            self.register_parameter("mtp_inv_temps", None)

        if self.use_ortho_decouple:
            apply_orthogonal_highway_decoupling(self)
        if self.use_hierarchical_16qubit and self.use_ortho_decouple and self.povm_readout is not None:
            with torch.no_grad():
                self.povm_readout.povm_head.data[:, 0::2].zero_()
                self.povm_readout.povm_head.data[:, 1::2] = l2_normalize(
                    self.povm_readout.povm_head.data[:, 1::2], dim=-1, eps=self.eps
                )

    def set_window_size(self, window_size: int) -> None:
        self.window_size = int(window_size)

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

    def _get_value_embed(self, layer_idx: int, idx: torch.Tensor) -> torch.Tensor | None:
        bank_idx = self._ve_bank_index(layer_idx)
        if bank_idx is not None and self.value_embeds is not None:
            return self.value_embeds[bank_idx](idx)
        return None

    def forward_features(
        self,
        idx: torch.Tensor,
        return_layer_states: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode token indices `(B, T)` into final hyperspherical states `(B, T, embed_dim)`."""
        B, T = idx.shape

        h_ngram_sparse: torch.Tensor | None = None
        if self.hierarchical_qubit_embed is not None:
            psi_0 = self.hierarchical_qubit_embed(idx)
            if self.ngram_embed is not None:
                h_ngram = self.ngram_embed(idx)
                s0 = self.ngram_embed.layer_scales[0].to(psi_0.dtype)
                psi_0 = l2_normalize(psi_0 + s0 * h_ngram.to(psi_0.dtype), dim=-1, eps=self.eps)
        elif self.sparse_qubit_embed is not None:
            psi_0 = self.sparse_qubit_embed(idx)
            h_ngram_sparse = self.sparse_qubit_embed.forward_ngram_delta(idx)
        else:
            psi_0 = self.hilbert_lift(idx)  # (B, T, embed_dim)
            if self.ngram_embed is not None:
                h_ngram = self.ngram_embed(idx)
                s0 = self.ngram_embed.layer_scales[0].to(psi_0.dtype)
                psi_0 = l2_normalize(psi_0 + s0 * h_ngram.to(psi_0.dtype), dim=-1, eps=self.eps)

        if self.training and self.depolarizing_jitter > 0.0:
            noise = torch.randn_like(psi_0) * (self.depolarizing_jitter / math.sqrt(self.embed_dim))
            psi_0 = l2_normalize(psi_0 + noise, dim=-1, eps=self.eps)

        if self.smear_gate is not None:
            psi_0 = self.smear_gate(psi_0, spherical=True, eps=self.eps)
        elif self.use_bigram_smear and T > 1 and self.smear_embed_gate is not None:
            psi_prev = F.pad(psi_0[:, :-1, :], (0, 0, 1, 0))
            eta = (0.06 * torch.tanh(self.smear_embed_gate)).to(psi_0.dtype)
            psi_0 = l2_normalize(psi_0 + eta * psi_prev, dim=-1, eps=self.eps)

        if not torch.compiler.is_compiling():
            self._last_psi_0 = psi_0.detach()
        psi = psi_0
        prev_psi = psi_0
        layer_states: list[torch.Tensor] = []
        encoder_states: list[torch.Tensor] = []
        n_half = self.num_layers // 2
        mid_layer_idx = max(0, n_half - 1)

        cos, sin = self.rope.get_cos_sin(T)
        doc_ids = compute_document_ids(idx, eot_token_id=self.eot_token_id) if self.doc_mask_attn else None
        v_0: torch.Tensor | None = None
        last_beam2_final: torch.Tensor | None = None
        need_v0 = bool(self.true_layer0_v_residual or self.value_embeds is not None)
        use_block_ckpt = bool(self.use_activation_checkpointing and self.training and torch.is_grad_enabled())
        for layer_idx, block in enumerate(self.blocks):
            if (
                layer_idx > 0
                and self.sparse_qubit_embed is not None
                and h_ngram_sparse is not None
                and layer_idx < self.sparse_qubit_embed.layer_scales.numel()
            ):
                s_l = (0.15 * torch.tanh(self.sparse_qubit_embed.layer_scales[layer_idx])).to(psi.dtype)
                psi = l2_normalize(psi + s_l * h_ngram_sparse.to(psi.dtype), dim=-1, eps=self.eps)
            psi_before_block = psi
            ve_l = self._get_value_embed(layer_idx, idx)
            is_last_b2 = (layer_idx == self.num_layers - 1) and self.use_two_beam_kboson_readout
            if use_block_ckpt:
                from torch.utils.checkpoint import checkpoint

                def _run_ckpt_block(
                    psi_in: torch.Tensor,
                    prev_in: torch.Tensor,
                    v0_in: torch.Tensor | None,
                    ve_in: torch.Tensor | None,
                    _blk: nn.Module = block,
                    _l_idx: int = layer_idx,
                    _ret_b2: bool = is_last_b2,
                ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
                    v0_out = v0_in
                    beam2_b: torch.Tensor | None = None
                    if need_v0:
                        if _l_idx == 0 and getattr(_blk, "attn", None) is not None:
                            blk_out = _blk(
                                psi_in,
                                cos=cos,
                                sin=sin,
                                psi_0=psi_0,
                                v_0=None,
                                return_v0=True,
                                ve=ve_in,
                                window_size=self.window_size,
                                doc_ids=doc_ids,
                                return_beam2=_ret_b2,
                            )
                            if _ret_b2:
                                psi_b, v0_out, beam2_b = blk_out
                            else:
                                psi_b, v0_out = blk_out
                        else:
                            blk_out = _blk(
                                psi_in,
                                cos=cos,
                                sin=sin,
                                psi_0=psi_0,
                                v_0=v0_in,
                                return_v0=False,
                                ve=ve_in,
                                window_size=self.window_size,
                                doc_ids=doc_ids,
                                return_beam2=_ret_b2,
                            )
                            if _ret_b2:
                                psi_b, beam2_b = blk_out
                            else:
                                psi_b = blk_out
                    else:
                        blk_out = _blk(
                            psi_in,
                            cos=cos,
                            sin=sin,
                            psi_0=psi_0,
                            ve=ve_in,
                            window_size=self.window_size,
                            doc_ids=doc_ids,
                            return_beam2=_ret_b2,
                        )
                        if _ret_b2:
                            psi_b, beam2_b = blk_out
                        else:
                            psi_b = blk_out
                    if beam2_b is None and _l_idx == self.num_layers - 1:
                        beam2_b = getattr(_blk, "last_beam2", None)
                    if self.hierarchical_qubit_embed is not None and self.use_block_hierarchy_bridge:
                        psi_b = self.hierarchical_qubit_embed.forward_block_bridge(psi_b, layer_idx=_l_idx)
                    if self.mudd_skips is not None:
                        m_src = [psi_0] if _l_idx == 0 else [psi_0, prev_in]
                        psi_b = self.mudd_skips[_l_idx](psi_b, m_src, spherical=True, eps=self.eps)
                    return psi_b, v0_out, beam2_b

                psi, v_0, b2_out = checkpoint(
                    _run_ckpt_block,
                    psi,
                    prev_psi,
                    v_0,
                    ve_l,
                    use_reentrant=False,
                )
                if layer_idx == self.num_layers - 1:
                    last_beam2_final = b2_out
                if layer_idx == 0 and v_0 is not None and not torch.compiler.is_compiling():
                    self._last_v_0 = v_0.detach()
            else:
                if need_v0:
                    if layer_idx == 0 and block.attn is not None:
                        blk_out = block(
                            psi,
                            cos=cos,
                            sin=sin,
                            psi_0=psi_0,
                            v_0=None,
                            return_v0=True,
                            ve=ve_l,
                            window_size=self.window_size,
                            doc_ids=doc_ids,
                            return_beam2=is_last_b2,
                        )
                        if is_last_b2:
                            psi, v_0, last_beam2_final = blk_out
                        else:
                            psi, v_0 = blk_out
                        if not torch.compiler.is_compiling():
                            self._last_v_0 = v_0.detach()
                    else:
                        blk_out = block(
                            psi,
                            cos=cos,
                            sin=sin,
                            psi_0=psi_0,
                            v_0=v_0,
                            return_v0=False,
                            ve=ve_l,
                            window_size=self.window_size,
                            doc_ids=doc_ids,
                            return_beam2=is_last_b2,
                        )
                        if is_last_b2:
                            psi, last_beam2_final = blk_out
                        else:
                            psi = blk_out
                else:
                    blk_out = block(
                        psi,
                        cos=cos,
                        sin=sin,
                        psi_0=psi_0,
                        ve=ve_l,
                        window_size=self.window_size,
                        doc_ids=doc_ids,
                        return_beam2=is_last_b2,
                    )
                    if is_last_b2:
                        psi, last_beam2_final = blk_out
                    else:
                        psi = blk_out
                if self.hierarchical_qubit_embed is not None and self.use_block_hierarchy_bridge:
                    psi = self.hierarchical_qubit_embed.forward_block_bridge(psi, layer_idx=layer_idx)
                if self.mudd_skips is not None:
                    mudd_sources = [psi_0] if layer_idx == 0 else [psi_0, prev_psi]
                    psi = self.mudd_skips[layer_idx](psi, mudd_sources, spherical=True, eps=self.eps)
            prev_psi = psi_before_block

            if self.use_unet_interferometry and n_half > 0:
                if layer_idx < n_half:
                    encoder_states.append(psi)
                elif layer_idx >= self.num_layers - n_half:
                    enc_idx = self.num_layers - 1 - layer_idx
                    skip_psi = encoder_states[enc_idx]
                    if self.unet_skip_mode == "scalar" and self.unet_lambdas is not None:
                        lam_res = self.unet_lambdas[enc_idx, 0].to(psi.dtype)
                        lam_skip = self.unet_lambdas[enc_idx, 1].to(psi.dtype)
                        psi = l2_normalize(
                            lam_res * psi + lam_skip * skip_psi,
                            dim=-1,
                            eps=self.eps,
                        )
                    elif self.unet_gates is not None and self.unet_scales is not None:
                        u_scale = (0.35 * torch.tanh(self.unet_scales[enc_idx])).to(psi.dtype)
                        skip_rot = F.linear(skip_psi, self.unet_gates[enc_idx].to(psi.dtype))
                        psi = l2_normalize(
                            psi + u_scale * l2_normalize(skip_rot, dim=-1, eps=self.eps),
                            dim=-1,
                            eps=self.eps,
                        )
            if return_layer_states:
                layer_states.append(psi)

        raw_beam2 = (
            last_beam2_final
            if last_beam2_final is not None
            else (getattr(self.blocks[-1], "last_beam2", None) if len(self.blocks) > 0 else None)
        )
        if self.use_cosine_phase_beam_gate and raw_beam2 is not None:
            proj_final = (raw_beam2.float() * psi.float()).sum(dim=-1, keepdim=True)
            raw_beam2 = l2_normalize(
                raw_beam2.float() - proj_final * psi.float(),
                dim=-1,
                eps=self.eps,
            ).to(psi.dtype)
        self._last_psi_beam2 = raw_beam2

        if self.sparse_density_layer is not None:
            last_blk_density = getattr(self.blocks[-1], "sparse_density_layer", None) if len(self.blocks) > 0 else None
            if (
                last_blk_density is self.sparse_density_layer
                and self.sparse_density_layer._last_branches is not None
                and self.sparse_density_layer._last_p_weights is not None
            ):
                self._last_density_branches = self.sparse_density_layer._last_branches
                self._last_density_weights = self.sparse_density_layer._last_p_weights
            else:
                psi_out, branches, p_weights = self.sparse_density_layer(psi, return_branches=True)  # type: ignore[misc]
                psi = psi_out
                self._last_density_branches = branches
                self._last_density_weights = p_weights
        else:
            self._last_density_branches = None
            self._last_density_weights = None

        return psi, layer_states

    def compute_logits(self, psi: torch.Tensor) -> torch.Tensor:
        """Compute vocabulary logits `(..., vocab_size)` from final state `psi`."""
        if (
            self.use_two_beam_kboson_readout
            and self.hierarchical_povm_readout is not None
            and self.hierarchical_qubit_embed is not None
        ):
            inv_t_raw = (
                self.povm_readout.inv_temp
                if self.povm_readout is not None
                else self.hierarchical_povm_readout.inv_temp
            )
            inv_t_primary = (
                torch.clamp(inv_t_raw, min=1.0, max=45.0)
                if isinstance(inv_t_raw, torch.Tensor)
                else float(min(max(inv_t_raw, 1.0), 45.0))
            )
            if self.purity_gate is not None:
                beta_t = self.purity_gate(psi, inv_t_primary)
            else:
                beta_t = (
                    inv_t_primary
                    if isinstance(inv_t_primary, torch.Tensor)
                    else torch.tensor(float(inv_t_primary), device=psi.device, dtype=torch.float32)
                )
            base_head = self.povm_readout.povm_head if self.povm_readout is not None else None
            psi_s, psi_d, m_s, m_d, k_eff, lam_diff = (
                self.hierarchical_povm_readout.get_two_beam_kboson_factors(
                    self.hierarchical_qubit_embed,
                    psi,
                    psi_beam2=self._last_psi_beam2,
                    base_povm_head=base_head,
                    beta_t=beta_t,
                )
            )
            if not torch.compiler.is_compiling():
                self._last_mean_beta = float(beta_t.detach().mean().item())
                self._last_kboson_copies = float(k_eff.detach().item())
                self._last_lambda_diff = float(lam_diff.detach().float().mean().item())
            return compute_two_beam_kboson_logits(
                psi_s,
                psi_d,
                m_s,
                m_d,
                beta_t,
                k_eff,
                lam_diff,
                log_prior=self.povm_log_prior,
                quad_diffuse_weight=self.quadratic_diffuse_fidelity,
                bipolar_hole_weight=self.bipolar_hole_weight if self.use_bipolar_kboson else 0.0,
            )

        if self.use_last_layer_softmax:
            if self.hierarchical_povm_readout is not None and self.hierarchical_qubit_embed is not None:
                phi_seed = self.hierarchical_povm_readout.format_readout_features(psi)
                base_head = self.povm_readout.povm_head if self.povm_readout is not None else None
                M_seed = self.hierarchical_povm_readout.get_calibrated_softmax_weight(
                    self.hierarchical_qubit_embed,
                    base_head,
                ).to(phi_seed.dtype)
            elif self.povm_readout is not None:
                phi_seed = F.rms_norm(psi, (self.embed_dim,), eps=self.eps)
                M_seed = (self.povm_readout.povm_head * (1.0 / math.sqrt(float(self.embed_dim)))).to(phi_seed.dtype)
            else:
                phi_seed = F.rms_norm(psi, (self.embed_dim,), eps=self.eps)
                M_seed = (
                    F.linear(self.wte.weight, self.W_lift.weight) * (1.0 / math.sqrt(float(self.embed_dim)))
                ).to(phi_seed.dtype)
            raw = F.linear(phi_seed, M_seed)
            if self.povm_log_prior is not None:
                raw = raw + self.povm_log_prior.to(raw.dtype)
            cap = float(self.logit_softcap)
            logits = cap * torch.tanh(raw / cap) if cap > 0.0 else raw
            if not torch.compiler.is_compiling():
                self._last_mean_beta = float(logits.detach().float().std(dim=-1).mean().item())
            return logits

        if self.readout_mode == "asymmetric" and self.povm_readout is not None:
            phi_seed = l2_normalize(psi, dim=-1, eps=self.eps)
            if self.hierarchical_povm_readout is not None and self.hierarchical_qubit_embed is not None:
                M_seed = self.hierarchical_povm_readout.get_coupled_povm_matrix(
                    self.hierarchical_qubit_embed,
                    self.povm_readout.povm_head,
                ).to(phi_seed.dtype)
            else:
                M_seed = l2_normalize(self.povm_readout.povm_head, dim=1, eps=self.eps).to(phi_seed.dtype)
            inv_t_raw = self.povm_readout.inv_temp
            inv_t_primary = (
                torch.clamp(inv_t_raw, min=1.0, max=60.0)
                if isinstance(inv_t_raw, torch.Tensor)
                else float(min(max(inv_t_raw, 1.0), 60.0))
            )
            cos_sim = F.linear(phi_seed, M_seed)
            if self.purity_gate is not None:
                logits, beta_t = self.purity_gate.scale_logits(psi, cos_sim, inv_t_primary)
                self._last_mean_beta = float(beta_t.detach().mean().item())
            else:
                logits = cos_sim * (
                    inv_t_primary.to(phi_seed.dtype)
                    if isinstance(inv_t_primary, torch.Tensor)
                    else float(inv_t_primary)
                )
        elif (
            self.readout_mode in ("tied_stiefel", "tied_vocab")
            and self.readout_rot is not None
            and self.tied_inv_temp is not None
        ):
            psi_final = l2_normalize(self.readout_rot(psi), dim=-1, eps=self.eps)
            phi_seed = self.hilbert_lift.collapse_to_seed(psi_final)
            if self.seed_adapter is not None:
                phi_seed = self.seed_adapter(psi_final, phi_seed)
            if self.povm_head_seed_delta is not None:
                M_seed = l2_normalize(self.wte.weight + self.povm_head_seed_delta, dim=1, eps=self.eps).to(
                    phi_seed.dtype
                )
            else:
                M_seed = l2_normalize(self.wte.weight, dim=1, eps=self.eps).to(phi_seed.dtype)
            inv_t_primary = torch.clamp(self.tied_inv_temp, min=1.0, max=60.0)
            cos_sim = F.linear(phi_seed, M_seed)
            logits = cos_sim * inv_t_primary.to(phi_seed.dtype)
        else:
            psi_final = l2_normalize(psi, dim=-1, eps=self.eps)
            phi_seed = self.hilbert_lift.collapse_to_seed(psi_final)
            if self.seed_adapter is not None:
                phi_seed = self.seed_adapter(psi_final, phi_seed)
            M_seed = l2_normalize(self.wte.weight, dim=1, eps=self.eps).to(phi_seed.dtype)
            logits = F.linear(phi_seed, M_seed) * float(self._dual_inv_temp)

        if self.povm_log_prior is not None:
            logits = logits + self.povm_log_prior.to(logits.dtype)
        return logits

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_layer_states: bool = False,
        direct_chunked_ce: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        """Forward pass from token indices `(B, T)` to `(logits, loss)`."""
        B, T = idx.shape
        psi, layer_states = self.forward_features(idx, return_layer_states=return_layer_states)

        # 2. POVM, Two-Beam k-Boson, or Calibrated Last-Layer Softmax Measurement Collapse
        phi_seed: torch.Tensor | None = None
        M_seed: torch.Tensor | None = None
        inv_t_primary: torch.Tensor | float = self._dual_inv_temp

        phi_seed_for_chunked: torch.Tensor | None = None
        kboson_chunked_loss: torch.Tensor | None = None

        if (
            self.use_two_beam_kboson_readout
            and self.hierarchical_povm_readout is not None
            and self.hierarchical_qubit_embed is not None
        ):
            inv_t_raw = (
                self.povm_readout.inv_temp
                if self.povm_readout is not None
                else self.hierarchical_povm_readout.inv_temp
            )
            inv_t_primary = (
                torch.clamp(inv_t_raw, min=1.0, max=45.0)
                if isinstance(inv_t_raw, torch.Tensor)
                else float(min(max(inv_t_raw, 1.0), 45.0))
            )
            if self.purity_gate is not None:
                beta_t = self.purity_gate(psi, inv_t_primary)
            else:
                beta_t = (
                    inv_t_primary
                    if isinstance(inv_t_primary, torch.Tensor)
                    else torch.tensor(float(inv_t_primary), device=psi.device, dtype=torch.float32)
                )
            base_head = self.povm_readout.povm_head if self.povm_readout is not None else None
            psi_s, psi_d, m_s, m_d, k_eff, lam_diff = (
                self.hierarchical_povm_readout.get_two_beam_kboson_factors(
                    self.hierarchical_qubit_embed,
                    psi,
                    psi_beam2=self._last_psi_beam2,
                    base_povm_head=base_head,
                    beta_t=beta_t,
                )
            )
            phi_seed = psi_s
            M_seed = m_s
            if not torch.compiler.is_compiling():
                self._last_mean_beta = float(beta_t.detach().mean().item())
                self._last_kboson_copies = float(k_eff.detach().item())
                self._last_lambda_diff = float(lam_diff.detach().float().mean().item())
            if targets is not None and direct_chunked_ce:
                logits = torch.empty((B, 0, self.vocab_size), device=idx.device, dtype=psi.dtype)
                kboson_chunked_loss = direct_chunked_two_beam_kboson_ce(
                    psi_s,
                    psi_d,
                    m_s,
                    m_d,
                    beta_t,
                    k_eff,
                    lam_diff,
                    targets,
                    log_prior=self.povm_log_prior,
                    chunk_tokens=max(1, min(1024, B * T)),
                    quad_diffuse_weight=self.quadratic_diffuse_fidelity,
                    bipolar_hole_weight=self.bipolar_hole_weight if self.use_bipolar_kboson else 0.0,
                )
            else:
                logits = compute_two_beam_kboson_logits(
                    psi_s,
                    psi_d,
                    m_s,
                    m_d,
                    beta_t,
                    k_eff,
                    lam_diff,
                    log_prior=self.povm_log_prior,
                    quad_diffuse_weight=self.quadratic_diffuse_fidelity,
                    bipolar_hole_weight=self.bipolar_hole_weight if self.use_bipolar_kboson else 0.0,
                )
            self._last_density_branches = None
            self._last_density_weights = None
            if self.sparse_density_layer is not None:
                self.sparse_density_layer._last_branches = None
                self.sparse_density_layer._last_p_weights = None
        elif self.use_last_layer_softmax:
            if self.hierarchical_povm_readout is not None and self.hierarchical_qubit_embed is not None:
                phi_seed = self.hierarchical_povm_readout.format_readout_features(psi)
                base_head = self.povm_readout.povm_head if self.povm_readout is not None else None
                M_seed = self.hierarchical_povm_readout.get_calibrated_softmax_weight(
                    self.hierarchical_qubit_embed,
                    base_head,
                ).to(phi_seed.dtype)
            elif self.povm_readout is not None:
                phi_seed = F.rms_norm(psi, (self.embed_dim,), eps=self.eps)
                M_seed = (self.povm_readout.povm_head * (1.0 / math.sqrt(float(self.embed_dim)))).to(phi_seed.dtype)
            else:
                phi_seed = F.rms_norm(psi, (self.embed_dim,), eps=self.eps)
                M_seed = (
                    F.linear(self.wte.weight, self.W_lift.weight) * (1.0 / math.sqrt(float(self.embed_dim)))
                ).to(phi_seed.dtype)
            cap = float(self.logit_softcap)
            inv_t_primary = cap
            phi_seed_for_chunked = phi_seed
            if targets is not None and direct_chunked_ce:
                logits = torch.empty((B, 0, self.vocab_size), device=idx.device, dtype=psi.dtype)
                self._last_mean_beta = cap
            else:
                raw = F.linear(phi_seed, M_seed)
                if self.povm_log_prior is not None:
                    raw = raw + self.povm_log_prior.to(raw.dtype)
                logits = cap * torch.tanh(raw / cap) if cap > 0.0 else raw
                if not torch.compiler.is_compiling():
                    self._last_mean_beta = float(logits.detach().float().std(dim=-1).mean().item())
            self._last_density_branches = None
            self._last_density_weights = None
            if self.sparse_density_layer is not None:
                self.sparse_density_layer._last_branches = None
                self.sparse_density_layer._last_p_weights = None
        elif self.readout_mode == "asymmetric" and self.povm_readout is not None:
            phi_seed = l2_normalize(psi, dim=-1, eps=self.eps)
            if self.hierarchical_povm_readout is not None and self.hierarchical_qubit_embed is not None:
                M_seed = self.hierarchical_povm_readout.get_coupled_povm_matrix(
                    self.hierarchical_qubit_embed,
                    self.povm_readout.povm_head,
                ).to(phi_seed.dtype)
            else:
                M_seed = l2_normalize(self.povm_readout.povm_head, dim=1, eps=self.eps).to(phi_seed.dtype)
            inv_t_raw = self.povm_readout.inv_temp
            inv_t_primary = (
                torch.clamp(inv_t_raw, min=1.0, max=60.0)
                if isinstance(inv_t_raw, torch.Tensor)
                else float(min(max(inv_t_raw, 1.0), 60.0))
            )
            if (
                self.sparse_povm_readout is not None
                and self._last_density_branches is not None
                and self._last_density_weights is not None
            ):
                phi_seed, beta_rho = self.sparse_povm_readout.compute_effective_state_and_beta(
                    psi,
                    branches=self._last_density_branches,
                    p_weights=self._last_density_weights,
                    base_inv_temp=inv_t_primary,
                )
                cos_sim = F.linear(phi_seed, M_seed)
                if (
                    self.use_sparse_64qubit
                    and self.sparse_qubit_embed is not None
                    and phi_seed.size(-1) >= self.num_qubits
                ):
                    q_out = self.sparse_povm_readout.get_qubit_povm_matrix(self.sparse_qubit_embed).to(phi_seed.dtype)
                    cos_q = F.linear(l2_normalize(phi_seed[..., : self.num_qubits], dim=-1, eps=self.eps), q_out)
                    cos_sim = 0.70 * cos_sim + 0.30 * cos_q
                if self.purity_gate is not None:
                    logits, beta_t = self.purity_gate.scale_logits(psi, cos_sim, beta_rho)
                    self._last_mean_beta = float(beta_t.detach().mean().item())
                else:
                    logits = cos_sim * beta_rho.to(phi_seed.dtype)
                    self._last_mean_beta = float(beta_rho.detach().mean().item())
            elif targets is not None and direct_chunked_ce and not self.use_sparse_64qubit:
                if self.purity_gate is not None:
                    beta_t = self.purity_gate(psi, inv_t_primary)
                    self._last_mean_beta = float(beta_t.detach().mean().item())
                    phi_seed_for_chunked = phi_seed * beta_t.to(phi_seed.dtype)
                else:
                    scale_v = (
                        inv_t_primary.to(phi_seed.dtype)
                        if isinstance(inv_t_primary, torch.Tensor)
                        else float(inv_t_primary)
                    )
                    self._last_mean_beta = (
                        float(inv_t_primary.detach().item())
                        if isinstance(inv_t_primary, torch.Tensor)
                        else float(inv_t_primary)
                    )
                    phi_seed_for_chunked = phi_seed * scale_v
                logits = torch.empty((B, 0, self.vocab_size), device=idx.device, dtype=psi.dtype)
            else:
                cos_sim = F.linear(phi_seed, M_seed)
                if (
                    self.use_sparse_64qubit
                    and self.sparse_povm_readout is not None
                    and self.sparse_qubit_embed is not None
                    and phi_seed.size(-1) >= self.num_qubits
                ):
                    q_out = self.sparse_povm_readout.get_qubit_povm_matrix(self.sparse_qubit_embed).to(phi_seed.dtype)
                    cos_q = F.linear(l2_normalize(phi_seed[..., : self.num_qubits], dim=-1, eps=self.eps), q_out)
                    cos_sim = 0.70 * cos_sim + 0.30 * cos_q
                if self.purity_gate is not None:
                    logits, beta_t = self.purity_gate.scale_logits(psi, cos_sim, inv_t_primary)
                    self._last_mean_beta = float(beta_t.detach().mean().item())
                else:
                    logits = cos_sim * (
                        inv_t_primary.to(phi_seed.dtype)
                        if isinstance(inv_t_primary, torch.Tensor)
                        else float(inv_t_primary)
                    )
                    self._last_mean_beta = (
                        float(inv_t_primary.detach().item())
                        if isinstance(inv_t_primary, torch.Tensor)
                        else float(inv_t_primary)
                    )
            self._last_density_branches = None
            self._last_density_weights = None
            if self.sparse_density_layer is not None:
                self.sparse_density_layer._last_branches = None
                self.sparse_density_layer._last_p_weights = None
        elif (
            self.readout_mode in ("tied_stiefel", "tied_vocab")
            and self.readout_rot is not None
            and self.tied_inv_temp is not None
        ):
            psi_final = l2_normalize(self.readout_rot(psi), dim=-1, eps=self.eps)
            phi_seed = self.hilbert_lift.collapse_to_seed(psi_final)
            if self.seed_adapter is not None:
                phi_seed = self.seed_adapter(psi_final, phi_seed)
            if self.povm_head_seed_delta is not None:
                M_seed = l2_normalize(self.wte.weight + self.povm_head_seed_delta, dim=1, eps=self.eps).to(
                    phi_seed.dtype
                )
            else:
                M_seed = l2_normalize(self.wte.weight, dim=1, eps=self.eps).to(phi_seed.dtype)
            inv_t_primary = torch.clamp(self.tied_inv_temp, min=1.0, max=60.0)
            cos_sim = F.linear(phi_seed, M_seed)
            if self.purity_gate is not None:
                logits, beta_t = self.purity_gate.scale_logits(psi, cos_sim, inv_t_primary)
                self._last_mean_beta = float(beta_t.detach().mean().item())
            else:
                logits = cos_sim * inv_t_primary.to(phi_seed.dtype)
                self._last_mean_beta = float(inv_t_primary.detach().item())
        else:
            psi_final = l2_normalize(psi, dim=-1, eps=self.eps)
            phi_seed = self.hilbert_lift.collapse_to_seed(psi_final)
            if self.seed_adapter is not None:
                phi_seed = self.seed_adapter(psi_final, phi_seed)
            M_seed = l2_normalize(self.wte.weight, dim=1, eps=self.eps).to(phi_seed.dtype)
            inv_t_primary = float(self._dual_inv_temp)
            cos_sim = F.linear(phi_seed, M_seed)
            if self.purity_gate is not None:
                logits, beta_t = self.purity_gate.scale_logits(psi, cos_sim, inv_t_primary)
                self._last_mean_beta = float(beta_t.detach().mean().item())
            else:
                logits = cos_sim * inv_t_primary
                self._last_mean_beta = float(inv_t_primary)

        if (
            self.povm_log_prior is not None
            and not self.use_last_layer_softmax
            and not self.use_two_beam_kboson_readout
            and logits.numel() > 0
        ):
            logits = logits + self.povm_log_prior.to(logits.dtype)

        loss = None
        if targets is not None:
            if kboson_chunked_loss is not None:
                loss_t1 = kboson_chunked_loss
            elif direct_chunked_ce and phi_seed_for_chunked is not None and M_seed is not None:
                flat_phi = phi_seed_for_chunked.reshape(-1, phi_seed_for_chunked.size(-1))
                flat_tgt = targets.reshape(-1)
                eff_softcap = float(self.logit_softcap) if self.use_last_layer_softmax else 0.0
                loss_t1 = direct_chunked_softcap_linear_ce(
                    flat_phi,
                    M_seed,
                    flat_tgt,
                    softcap=eff_softcap,
                    chunk_tokens=max(1, min(1024, flat_phi.size(0))),
                    bias=self.povm_log_prior,
                )
            else:
                loss_t1 = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            loss = loss_t1
            if self.mtp_horizons > 1 and self.mtp_rotations is not None:
                beta_1_val = (
                    float(inv_t_primary.detach().item())
                    if isinstance(inv_t_primary, torch.Tensor)
                    else float(inv_t_primary)
                )
                mtp_dict: dict[str, float] = {
                    "ce_t+1": float(loss_t1.detach().item()),
                    "beta_t+1": beta_1_val,
                }
                for h_idx, u_rot in enumerate(self.mtp_rotations, start=1):
                    T_avail = T - h_idx
                    if T_avail <= 0:
                        break
                    if self.mtp_stride > 1 and T_avail > self.mtp_stride:
                        offset = (h_idx - 1) % self.mtp_stride
                        state_slice = slice(offset, T_avail, self.mtp_stride)
                        target_slice = slice(h_idx + offset, T, self.mtp_stride)
                    else:
                        state_slice = slice(0, T_avail)
                        target_slice = slice(h_idx, T)
                    tgt_h = targets[:, target_slice].reshape(-1)

                    if self.mtp_decoupled_temp and self.mtp_inv_temps is not None:
                        beta_h = torch.clamp(self.mtp_inv_temps[h_idx - 1], min=1.0, max=60.0)
                    else:
                        beta_h = (
                            inv_t_primary
                            if isinstance(inv_t_primary, torch.Tensor)
                            else torch.tensor(float(inv_t_primary), device=idx.device, dtype=torch.float32)
                        )

                    if self.readout_mode == "asymmetric" and self.povm_readout is not None:
                        psi_h = l2_normalize(
                            F.linear(psi[:, state_slice, :], u_rot.to(psi.dtype)),
                            dim=-1,
                            eps=self.eps,
                        )
                        M_full = l2_normalize(self.povm_readout.povm_head, dim=1, eps=self.eps).to(psi_h.dtype)
                        M_aux = M_full.detach() if self.mtp_detach_readout else M_full
                        logits_h = F.linear(psi_h, M_aux) * beta_h.to(psi_h.dtype)
                    else:
                        assert phi_seed is not None and M_seed is not None
                        z_h = l2_normalize(
                            F.linear(phi_seed[:, state_slice, :], u_rot.to(phi_seed.dtype)),
                            dim=-1,
                            eps=self.eps,
                        )
                        M_aux = M_seed.detach() if self.mtp_detach_readout else M_seed
                        logits_h = F.linear(z_h, M_aux) * beta_h.to(z_h.dtype)

                    if self.povm_log_prior is not None:
                        logits_h = logits_h + self.povm_log_prior.to(logits_h.dtype)

                    loss_h = F.cross_entropy(logits_h.reshape(-1, self.vocab_size), tgt_h)
                    mtp_dict[f"ce_t+{h_idx + 1}"] = float(loss_h.detach().item())
                    mtp_dict[f"beta_t+{h_idx + 1}"] = float(beta_h.detach().item())
                    loss = loss + (self.mtp_loss_weight / float(h_idx)) * loss_h
                self.last_mtp_losses = mtp_dict
        if return_layer_states:
            return logits, loss, layer_states
        return logits, loss

    def forward_mtp_logits(self, idx: torch.Tensor) -> list[torch.Tensor]:
        """Compute POVM measurement logits for all configured horizons `[t+1, t+2, ..., t+mtp_horizons]`."""
        logits_t1, _, layer_states = self(idx, return_layer_states=True)
        out_horizons = [logits_t1]
        if self.mtp_horizons <= 1 or self.mtp_rotations is None or not layer_states:
            return out_horizons

        psi = layer_states[-1]
        if self.readout_mode == "asymmetric" and self.povm_readout is not None:
            M_full = l2_normalize(self.povm_readout.povm_head, dim=1, eps=self.eps).to(psi.dtype)
            inv_t_primary = self.povm_readout.inv_temp
            for h_idx, u_rot in enumerate(self.mtp_rotations, start=1):
                if self.mtp_decoupled_temp and self.mtp_inv_temps is not None:
                    beta_h = torch.clamp(self.mtp_inv_temps[h_idx - 1], min=1.0, max=60.0)
                else:
                    beta_h = (
                        inv_t_primary
                        if isinstance(inv_t_primary, torch.Tensor)
                        else torch.tensor(float(inv_t_primary), device=psi.device, dtype=torch.float32)
                    )
                psi_h = l2_normalize(F.linear(psi, u_rot.to(psi.dtype)), dim=-1, eps=self.eps)
                out_horizons.append(F.linear(psi_h, M_full) * beta_h.to(psi_h.dtype))
        elif (
            self.readout_mode in ("tied_stiefel", "tied_vocab")
            and self.readout_rot is not None
            and self.tied_inv_temp is not None
        ):
            psi_final = l2_normalize(self.readout_rot(psi), dim=-1, eps=self.eps)
            phi_seed = self.hilbert_lift.collapse_to_seed(psi_final)
            if self.seed_adapter is not None:
                phi_seed = self.seed_adapter(psi_final, phi_seed)
            if self.povm_head_seed_delta is not None:
                M_seed = l2_normalize(self.wte.weight + self.povm_head_seed_delta, dim=1, eps=self.eps).to(
                    phi_seed.dtype
                )
            else:
                M_seed = l2_normalize(self.wte.weight, dim=1, eps=self.eps).to(phi_seed.dtype)
            inv_t_primary = torch.clamp(self.tied_inv_temp, min=1.0, max=60.0)
            for h_idx, u_rot in enumerate(self.mtp_rotations, start=1):
                if self.mtp_decoupled_temp and self.mtp_inv_temps is not None:
                    beta_h = torch.clamp(self.mtp_inv_temps[h_idx - 1], min=1.0, max=60.0)
                else:
                    beta_h = inv_t_primary
                z_h = l2_normalize(F.linear(phi_seed, u_rot.to(phi_seed.dtype)), dim=-1, eps=self.eps)
                out_horizons.append(F.linear(z_h, M_seed) * beta_h.to(z_h.dtype))
        else:
            psi_final = l2_normalize(psi, dim=-1, eps=self.eps)
            phi_seed = self.hilbert_lift.collapse_to_seed(psi_final)
            if self.seed_adapter is not None:
                phi_seed = self.seed_adapter(psi_final, phi_seed)
            M_seed = l2_normalize(self.wte.weight, dim=1, eps=self.eps).to(phi_seed.dtype)
            for h_idx, u_rot in enumerate(self.mtp_rotations, start=1):
                if self.mtp_decoupled_temp and self.mtp_inv_temps is not None:
                    beta_h = torch.clamp(self.mtp_inv_temps[h_idx - 1], min=1.0, max=60.0)
                else:
                    beta_h = torch.tensor(float(self._dual_inv_temp), device=phi_seed.device, dtype=torch.float32)
                z_h = l2_normalize(F.linear(phi_seed, u_rot.to(phi_seed.dtype)), dim=-1, eps=self.eps)
                out_horizons.append(F.linear(z_h, M_seed) * beta_h.to(z_h.dtype))
        return out_horizons

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
