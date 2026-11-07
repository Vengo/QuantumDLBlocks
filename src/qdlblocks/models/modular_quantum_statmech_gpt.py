"""Wave 49 (`EXPERIMENT_PLAN_064.md`): Modular Quantum Statistical-Mechanics Language Model (`ModularQuantumStatMechGPT`).

Rebuilds `StandaloneQuantumBornGPT` (`Wave 48 v3`) by composing the factored library modules:
- [`src/qdlblocks/layers/quantum_statmech_channels.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/quantum_statmech_channels.py):
  `PurificationMode` (`QUADRATIC_MCWEENY`, `CUBIC_FERMI_DIRAC`, `LEARNABLE_TEMP_MCWEENY`),
  `QuantumChannelMode` (`UNITARY_U16`, `CPTP_KRAUS_RANK_K`),
  `EntropyGateMode` (`NONE`, `DETACHED_TSALLIS_PURITY_GATE`),
  and `compiled_statmech_purify_evolve_uhlmann_krylov_step`.
- [`src/qdlblocks/layers/isometric_born_trace_readout.py`](file:///home/nixos/QuantumDLBlocks/src/qdlblocks/layers/isometric_born_trace_readout.py):
  `VocabularyReadoutMode` (`SOFTCAP_LINEAR_CE`, `RANK2_KRYLOV_BORN`, `MEAN_CENTERED_ISOMETRIC_TRACE`),
  `MeanCenteredIsometricBornTraceCEFn`, `Rank2KrylovBornChunkedCEFn`, and `evaluate_readout_divergence_batch`.

Also eliminates per-layer `.item()` GPU-to-CPU synchronization barriers during forward passes and supports
configurable token embedding regimes (`EmbeddingTrainMode`: `TRAINABLE_SPHERE_MANIFOLD`,
`FROZEN_RANDOM_SPHERE`, and `FROZEN_PPMI_SVD_SPHERE`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.impure_density_mirror_block import (
    DEFAULT_KRYLOV_OBSERVABLE_HEAD_DIM, DEFAULT_NS5_STEPS, DEFAULT_RIEMANNIAN_MOMENTUM_BETA,
    DEFAULT_ROPE_BASE_THETA, DEFAULT_TIKHONOV_GRAM_LAMBDA, SQRT2_ISOMETRIC_OFFDIAG_SCALE,
    TRACE_NORM_EPS, complex_l2_normalize, fast_rank_r_hermitian_and_ket_pack,
    initialize_stiefel_parameter, riemannian_complex_unitary_muon_step, riemannian_stiefel_muon_step,
)
from ..layers.isometric import apply_partial_key_offset
from ..layers.isometric_born_trace_readout import (
    DEFAULT_CE_CHUNK_TOKENS, DEFAULT_ISOMETRIC_TRACE_INIT_BETA, DEFAULT_KRYLOV_BORN_LOG_TEMP,
    DEFAULT_LOGIT_SOFTCAP, MeanCenteredIsometricBornTraceCEFn, Rank2KrylovBornChunkedCEFn,
    VocabularyReadoutMode, eval_mean_centered_isometric_trace_logits,
    eval_rank2_krylov_born_logits, pack_multihead_wavepacket_density_isometric,
)
from ..layers.quantum_statmech_channels import (
    DEFAULT_KRAUS_CHANNEL_RANK, EntropyGateMode, PurificationMode, QuantumChannelMode,
    StiefelMomentumMode, causal_born_p3_weights, compiled_statmech_purify_evolve_uhlmann_krylov_step,
    compute_real_stiefel_vector_diagnostics, euclidean_frobenius_normed_muon_step,
    initialize_complex_kraus_stiefel_pair, project_complex_stiefel_tangent,
    project_real_stiefel_tangent, riemannian_complex_kraus_stiefel_muon_step,
    riemannian_complex_tangent_transport_muon_step, riemannian_stiefel_tangent_transport_muon_step,
)
from ..layers.wavepacket_statmech_block import (
    NativeWavepacketStatMechBlock,
    apply_causal_wavepacket_dispersion,
    apply_factored_wavepacket_stiefel,
    initialize_wavepacket_ppmi_sphere_embeddings_,
)
from ..optim.vocab_manifold_muon import (
    DEFAULT_BATCH_GRAM_MUON_LR, DEFAULT_SPHERE_GEODESIC_LR, DEFAULT_SPHERE_SNR_TRUST_TAU,
    DEFAULT_VOCAB_RADIAL_LR_RATIO, BatchSubspaceTikhonovGramMuon,
    RiemannianSphereTangentSNROptimizer, _project_to_sphere_tangent, project_rows_to_rms_sphere_,
)
from .compiled_leaderboard_classical_gpt import _CompiledChunkedSoftcapLinearCEFn
from .standalone_quantum_born_gpt import VocabOptimizerMode

# ---------------------------------------------------------------------------
# Top-of-File Named Constants (Rule 7 — Synchronized with EXPERIMENT_PLAN_064.md)
# ---------------------------------------------------------------------------
DEFAULT_STATMECH_VOCAB_SIZE: int = 50304
DEFAULT_STATMECH_EMBED_DIM: int = 768
DEFAULT_STATMECH_NUM_LAYERS: int = 12
DEFAULT_STATMECH_NUM_HEADS: int = 6
DEFAULT_STATMECH_HEAD_COMPLEX_DIM: int = 16
DEFAULT_STATMECH_VALUE_BRANCHES: int = 4
DEFAULT_WAVEPACKET_CHANNELS: int = 48
DEFAULT_COMPLEX_WAVEPACKET_MODES: int = 24

DEFAULT_STIEFEL_GEODESIC_LR: float = 0.025
DEFAULT_UNITARY_GEODESIC_LR: float = 0.020
DEFAULT_SCALAR_ADAM_LR: float = 0.005
DEFAULT_ADAMW_VOCAB_LR: float = 0.004
DEFAULT_V0_HIGHWAY_INIT_LAMBDA: float = 0.15
DEFAULT_RESIDUAL_GAIN_COEFF: float = 0.25


class EmbeddingTrainMode(str, Enum):
    """Strategy enum selecting token embedding initialization and trainability regime (Rule 3)."""

    TRAINABLE_SPHERE_MANIFOLD = "trainable_sphere_manifold"
    FROZEN_RANDOM_SPHERE = "frozen_random_sphere"
    FROZEN_PPMI_SVD_SPHERE = "frozen_ppmi_svd_sphere"
    WAVEPACKET_SPHERE_48X16 = "wavepacket_sphere_48x16"
    CAUSAL_WAVEPACKET_48X16 = "causal_wavepacket_48x16"
    NATIVE_WAVEPACKET_48X16 = "native_wavepacket_48x16"
    NATIVE_CAUSAL_WAVEPACKET_48X16 = "native_causal_wavepacket_48x16"


@dataclass(frozen=True)
class ModularQuantumStatMechConfig:
    """Configuration for `ModularQuantumStatMechGPT` (`<= 8` fields, Rule 2)."""

    vocab_size: int = DEFAULT_STATMECH_VOCAB_SIZE
    n_layer: int = DEFAULT_STATMECH_NUM_LAYERS
    n_head: int = DEFAULT_STATMECH_NUM_HEADS
    n_embd: int = DEFAULT_STATMECH_EMBED_DIM
    purification_mode: PurificationMode = PurificationMode.QUADRATIC_MCWEENY
    channel_mode: QuantumChannelMode = QuantumChannelMode.UNITARY_U16
    entropy_gate_mode: EntropyGateMode = EntropyGateMode.NONE
    readout_mode: VocabularyReadoutMode = VocabularyReadoutMode.SOFTCAP_LINEAR_CE


def _mach_zehnder_interference_impl(beam_a: torch.Tensor, beam_b: torch.Tensor) -> torch.Tensor:
    """Compute two-beam Mach-Zehnder constructive minus destructive interference intensity."""
    return F.relu(beam_a + beam_b).square() - 0.25 * F.relu(beam_a - beam_b).square()


_COMPILED_MZ_HELPER: Any = None
_COMPILED_PACK_HELPER: Any = None


class ModularQuantumStatMechBlock(nn.Module):
    """Modular Quantum Statistical-Mechanics Transformer Layer composing factored channels and purifiers."""

    def __init__(
        self,
        cfg: ModularQuantumStatMechConfig,
        layer_idx: int = 0,
        kraus_rank: int = DEFAULT_KRAUS_CHANNEL_RANK,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.embed_dim = cfg.n_embd
        self.num_heads = cfg.n_head
        self.d_h = DEFAULT_STATMECH_HEAD_COMPLEX_DIM
        self.v_rank = DEFAULT_STATMECH_VALUE_BRANCHES
        self.kraus_rank = max(1, int(kraus_rank))
        self.obs_head_dim = DEFAULT_KRYLOV_OBSERVABLE_HEAD_DIM
        self.total_obs_dim = self.num_heads * self.obs_head_dim
        self.qk_complex_dim = self.v_rank * self.d_h
        self.ket_real_dim = self.num_heads * (2 * self.qk_complex_dim)
        self.v_real_dim = self.num_heads * self.v_rank * (2 * self.d_h)
        self.channel_hidden_dim = self.embed_dim * 2
        self.is_long_window_layer = (self.layer_idx % 4 == 3) or (self.layer_idx == cfg.n_layer - 1)
        self.use_compiled_krylov: bool = False

        self.block_lambdas = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float32))
        v0_init = DEFAULT_V0_HIGHWAY_INIT_LAMBDA if self.layer_idx > 0 else 0.0
        self.v_lambdas = nn.Parameter(torch.tensor([1.0, v0_init], dtype=torch.float32))

        self.w_q = nn.Parameter(torch.empty(self.embed_dim, self.ket_real_dim))
        self.w_k = nn.Parameter(torch.empty(self.embed_dim, self.ket_real_dim))
        self.w_v = nn.Parameter(torch.empty(self.embed_dim, self.v_real_dim))
        initialize_stiefel_parameter(self.w_q)
        initialize_stiefel_parameter(self.w_k)
        initialize_stiefel_parameter(self.w_v)

        if cfg.channel_mode == QuantumChannelMode.CPTP_KRAUS_RANK_K:
            kr_re, kr_im = initialize_complex_kraus_stiefel_pair(self.num_heads, self.d_h, kraus_rank=self.kraus_rank)
            self.head_unitary_re = nn.Parameter(kr_re)
            self.head_unitary_im = nn.Parameter(kr_im)
        else:
            phase_angles = torch.linspace(-0.25, 0.25, self.d_h, dtype=torch.float32)
            diag_re = torch.diag(torch.cos(phase_angles)).unsqueeze(0).expand(self.num_heads, -1, -1).clone()
            diag_im = torch.diag(torch.sin(phase_angles)).unsqueeze(0).expand(self.num_heads, -1, -1).clone()
            self.head_unitary_re = nn.Parameter(diag_re)
            self.head_unitary_im = nn.Parameter(diag_im)

        init_gain = DEFAULT_RESIDUAL_GAIN_COEFF / math.sqrt(float(max(1, cfg.n_layer)))
        self.w_obs = nn.Parameter(torch.empty(self.total_obs_dim, self.embed_dim))
        initialize_stiefel_parameter(self.w_obs)
        self.obs_scale = nn.Parameter(torch.tensor(init_gain, dtype=torch.float32))

        self.w_beam_a = nn.Parameter(torch.empty(self.embed_dim, self.channel_hidden_dim))
        self.w_beam_b = nn.Parameter(torch.empty(self.embed_dim, self.channel_hidden_dim))
        self.w_chan = nn.Parameter(torch.empty(self.channel_hidden_dim, self.embed_dim))
        initialize_stiefel_parameter(self.w_beam_a)
        initialize_stiefel_parameter(self.w_beam_b)
        initialize_stiefel_parameter(self.w_chan)
        self.chan_scale = nn.Parameter(torch.tensor(init_gain, dtype=torch.float32))

        init_purify = torch.tensor([-2.0, 2.0, 0.0], dtype=torch.float32).unsqueeze(0).expand(self.num_heads, -1).clone()
        self.purify_logits = nn.Parameter(
            init_purify, requires_grad=(cfg.purification_mode == PurificationMode.LEARNABLE_TEMP_MCWEENY)
        )
        self.entropy_gate_params = nn.Parameter(
            torch.zeros(self.num_heads, 3, dtype=torch.float32),
            requires_grad=(
                cfg.entropy_gate_mode
                in (EntropyGateMode.DETACHED_TSALLIS_PURITY_GATE, EntropyGateMode.ISOMETRIC_NORMALIZED_PLUS_TSALLIS)
            ),
        )

        inv_freq = 1.0 / (
            DEFAULT_ROPE_BASE_THETA
            ** (torch.arange(0, self.qk_complex_dim, dtype=torch.float32) / float(self.qk_complex_dim))
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

        re_mask = torch.diag(torch.ones(self.d_h)) + torch.triu(
            torch.full((self.d_h, self.d_h), SQRT2_ISOMETRIC_OFFDIAG_SCALE), diagonal=1
        )
        im_mask = torch.tril(torch.full((self.d_h, self.d_h), -SQRT2_ISOMETRIC_OFFDIAG_SCALE), diagonal=-1)
        self.register_buffer("re_pack_mask", re_mask, persistent=False)
        self.register_buffer("im_pack_mask", im_mask, persistent=False)
        self._last_telemetry_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Return all 7 direct 2D real Stiefel matrices in this layer."""
        return [self.w_q, self.w_k, self.w_v, self.w_obs, self.w_beam_a, self.w_beam_b, self.w_chan]

    def complex_unitary_pair(self) -> tuple[nn.Parameter, nn.Parameter]:
        """Return the 3D complex head operator pair `(head_unitary_re, head_unitary_im)`."""
        return self.head_unitary_re, self.head_unitary_im

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return 1D scalar residual gains, highway weights, and optional purification/gate parameters."""
        params = [self.obs_scale, self.chan_scale, self.block_lambdas, self.v_lambdas]
        if self.purify_logits.requires_grad:
            params.append(self.purify_logits)
        if self.entropy_gate_params.requires_grad:
            params.append(self.entropy_gate_params)
        return params

    def get_last_telemetry_dict(self) -> dict[str, float]:
        """Convert detached GPU telemetry tensors to Python floats only when requested at logging steps."""
        if self._last_telemetry_tensors is None:
            return {"mean_purity": 1.0, "mean_eff_rank": 1.0, "mean_lambda1": 0.5, "mean_lambda2": 0.5}
        p_t, er_t, l1_t, l2_t = self._last_telemetry_tensors
        return {
            "mean_purity": float(p_t.item()),
            "mean_eff_rank": float(er_t.item()),
            "mean_lambda1": float(l1_t.item()),
            "mean_lambda2": float(l2_t.item()),
        }

    def _apply_complex_rope(self, z_re: torch.Tensor, z_im: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply complex rotary phase $e^{i t \\omega_k}$ to `(z_re, z_im)` `(B, H, T, d_qk)`."""
        seq_len = z_re.size(-2)
        pos = torch.arange(seq_len, device=z_re.device, dtype=torch.float32)
        angles = torch.outer(pos, self.rope_inv_freq).unsqueeze(0).unsqueeze(0)
        cos_a = torch.cos(angles).to(z_re.dtype)
        sin_a = torch.sin(angles).to(z_re.dtype)
        return z_re * cos_a - z_im * sin_a, z_re * sin_a + z_im * cos_a

    def compute_quantum_attention_features(
        self,
        x_base: torch.Tensor,
        v_0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute `(obs_concat, v_0_next, krylov_weights_bt2, packed_rho_trace1_bth256)`."""
        bsz, seq_len, _ = x_base.shape
        work_dtype = x_base.dtype
        h_norm = F.rms_norm(x_base, (self.embed_dim,), eps=TRACE_NORM_EPS)

        q_raw = torch.matmul(h_norm, self.w_q.to(work_dtype)).view(
            bsz, seq_len, self.num_heads, 2, self.qk_complex_dim
        )
        k_raw = torch.matmul(h_norm, self.w_k.to(work_dtype)).view(
            bsz, seq_len, self.num_heads, 2, self.qk_complex_dim
        )
        v_proj = torch.matmul(h_norm, self.w_v.to(work_dtype))

        if v_0 is None:
            v_0_next = v_proj
            v_mixed = v_proj
        else:
            v_0_next = v_0
            if v_0.shape == v_proj.shape:
                v_mixed = self.v_lambdas[0].to(work_dtype) * v_proj + self.v_lambdas[1].to(work_dtype) * v_0
            else:
                v_mixed = v_proj

        q_re, q_im = q_raw[..., 0, :].transpose(1, 2), q_raw[..., 1, :].transpose(1, 2)
        k_re, k_im = k_raw[..., 0, :].transpose(1, 2), k_raw[..., 1, :].transpose(1, 2)
        q_re, q_im = complex_l2_normalize(q_re, q_im)
        k_re, k_im = complex_l2_normalize(k_re, k_im)
        q_re, q_im = self._apply_complex_rope(q_re, q_im)
        k_re, k_im = self._apply_complex_rope(k_re, k_im)

        if self.is_long_window_layer and seq_len > 1:
            k_re = apply_partial_key_offset(k_re.transpose(1, 2)).transpose(1, 2)
            k_im = apply_partial_key_offset(k_im.transpose(1, 2)).transpose(1, 2)

        q_cat = torch.cat([q_re, q_im], dim=-1)
        k_cat = torch.cat([k_re, k_im], dim=-1)
        k_skew = torch.cat([-k_im, k_re], dim=-1)
        ip_re = torch.matmul(q_cat, k_cat.transpose(-2, -1))
        ip_im = torch.matmul(q_cat, k_skew.transpose(-2, -1))
        attn_weights = causal_born_p3_weights(ip_re, ip_im, use_compile=self.use_compiled_krylov)

        v_raw = v_mixed.view(bsz, seq_len, self.num_heads, self.v_rank, 2, self.d_h)
        v_re = v_raw[..., 0, :].permute(0, 2, 1, 3, 4)
        v_im = v_raw[..., 1, :].permute(0, 2, 1, 3, 4)
        v_branch_dim = self.v_rank * 2 * self.d_h
        v_branch_flat = torch.cat([v_re, v_im], dim=-1).reshape(bsz, self.num_heads, seq_len, v_branch_dim)

        global _COMPILED_PACK_HELPER
        if self.use_compiled_krylov and v_re.is_cuda:
            if _COMPILED_PACK_HELPER is None:
                _COMPILED_PACK_HELPER = torch.compile(fast_rank_r_hermitian_and_ket_pack, dynamic=False)
            packed_tok_288 = _COMPILED_PACK_HELPER(v_re, v_im, self.re_pack_mask, self.im_pack_mask, TRACE_NORM_EPS)
        else:
            packed_tok_288 = fast_rank_r_hermitian_and_ket_pack(
                v_re, v_im, self.re_pack_mask, self.im_pack_mask, eps=TRACE_NORM_EPS
            )
        packed_ctx_all = torch.matmul(attn_weights, torch.cat([packed_tok_288, v_branch_flat], dim=-1))

        obs_320, _, packed_rho_trace1, lam1, lam2, purity, eff_rank = compiled_statmech_purify_evolve_uhlmann_krylov_step(
            packed_ctx_all,
            self.head_unitary_re,
            self.head_unitary_im,
            self.re_pack_mask,
            self.im_pack_mask,
            purify_logits_h3=self.purify_logits,
            entropy_gate_params=self.entropy_gate_params,
            modes=(
                self.cfg.purification_mode.value,
                self.cfg.channel_mode.value,
                self.cfg.entropy_gate_mode.value,
                self.use_compiled_krylov,
            ),
        )

        with torch.no_grad():
            self._last_telemetry_tensors = (
                purity.detach().mean(),
                eff_rank.detach().mean(),
                lam1.detach().mean(),
                lam2.detach().mean(),
            )

        lam1_bar, lam2_bar = lam1.mean(dim=1), lam2.mean(dim=1)
        lam_sum = (lam1_bar + lam2_bar).clamp_min(1e-6)
        krylov_weights_bt2 = torch.stack([lam1_bar / lam_sum, lam2_bar / lam_sum], dim=-1)

        obs_concat = obs_320.to(work_dtype).transpose(1, 2).reshape(bsz, seq_len, self.total_obs_dim)
        rho_trace1_bt = packed_rho_trace1.to(work_dtype).transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.d_h * self.d_h)
        return obs_concat, v_0_next, krylov_weights_bt2, rho_trace1_bt

    def compute_channel_features(self, x_mid: torch.Tensor) -> torch.Tensor:
        """Compute Stiefel two-beam Mach-Zehnder quantum interference features `(B, T, channel_hidden_dim)`."""
        work_dtype = x_mid.dtype
        m_norm = F.rms_norm(x_mid, (self.embed_dim,), eps=TRACE_NORM_EPS)
        beam_a = torch.matmul(m_norm, self.w_beam_a.to(work_dtype))
        beam_b = torch.matmul(m_norm, self.w_beam_b.to(work_dtype))
        global _COMPILED_MZ_HELPER
        if self.use_compiled_krylov and beam_a.is_cuda:
            if _COMPILED_MZ_HELPER is None:
                _COMPILED_MZ_HELPER = torch.compile(_mach_zehnder_interference_impl, dynamic=False)
            return _COMPILED_MZ_HELPER(beam_a, beam_b)
        return _mach_zehnder_interference_impl(beam_a, beam_b)

    def forward(
        self,
        x: torch.Tensor,
        x_0: torch.Tensor | None = None,
        v_0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning `(x_out, v_0_next, krylov_weights_bt2, rho_trace1_bt)`."""
        work_dtype = x.dtype
        if x_0 is not None:
            x_cur = self.block_lambdas[0].to(work_dtype) * x + self.block_lambdas[1].to(work_dtype) * x_0.to(work_dtype)
        else:
            x_cur = x

        obs_concat, v_0_next, krylov_weights_bt2, rho_trace1_bt = self.compute_quantum_attention_features(x_cur, v_0=v_0)
        attn_delta = torch.matmul(obs_concat, self.w_obs.to(work_dtype)) * self.obs_scale.to(work_dtype)
        x_mid = x_cur + attn_delta

        interference = self.compute_channel_features(x_mid)
        chan_delta = torch.matmul(interference, self.w_chan.to(work_dtype)) * self.chan_scale.to(work_dtype)
        x_out = x_mid + chan_delta
        return x_out, v_0_next, krylov_weights_bt2, rho_trace1_bt


class ModularRiemannianStiefelKrausMuon(torch.optim.Optimizer):
    """Riemannian Stiefel & Unitary/Kraus optimizer supporting both $U(d_h)$ and tall Kraus $\\mathrm{St}_{\\mathbb{C}}(K d_h, d_h)$."""

    def __init__(
        self,
        real_stiefel_params: list[nn.Parameter],
        complex_operator_pairs: list[tuple[nn.Parameter, nn.Parameter]],
        stiefel_lr: float = DEFAULT_STIEFEL_GEODESIC_LR,
        unitary_lr: float = DEFAULT_UNITARY_GEODESIC_LR,
        beta: float = DEFAULT_RIEMANNIAN_MOMENTUM_BETA,
        ns_steps: int = DEFAULT_NS5_STEPS,
        tikhonov_lambda: float = DEFAULT_TIKHONOV_GRAM_LAMBDA,
        momentum_mode: StiefelMomentumMode = StiefelMomentumMode.RIEMANNIAN_TANGENT_TRANSPORTED,
    ) -> None:
        self.complex_operator_pairs = list(complex_operator_pairs)
        self.momentum_mode = momentum_mode
        self.unnormalized_param_ids: set[int] = set()
        self.collect_diagnostics: bool = False
        self._last_step_diagnostics: dict[str, float] = {}
        all_params = list(real_stiefel_params)
        for u_re, u_im in self.complex_operator_pairs:
            all_params.extend([u_re, u_im])
        defaults = {
            "lr": float(stiefel_lr),
            "unitary_lr": float(unitary_lr),
            "beta": float(beta),
            "ns_steps": int(ns_steps),
            "tikhonov_lambda": float(tikhonov_lambda),
        }
        super().__init__(all_params, defaults)
        self._complex_ids = {id(u_re) for u_re, _ in self.complex_operator_pairs} | {
            id(u_im) for _, u_im in self.complex_operator_pairs
        }

    def set_lr_scale(self, lr_mult: float) -> None:
        """Scale geodesic step size by `lr_mult`."""
        for group in self.param_groups:
            group["lr_mult"] = float(lr_mult)

    def set_momentum(self, beta: float) -> None:
        """Update Riemannian tangent momentum coefficient `beta` across all parameter groups."""
        for group in self.param_groups:
            group["beta"] = float(beta)

    def get_last_step_diagnostics(self) -> dict[str, float]:
        """Return the most recent parameter-gradient / momentum / retraction vector diagnostics."""
        return dict(self._last_step_diagnostics)

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr_mult = float(group.get("lr_mult", 1.0))
        stiefel_lr = float(group["lr"]) * lr_mult
        unitary_lr = float(group["unitary_lr"]) * lr_mult
        beta = float(group["beta"])
        ns_steps = int(group["ns_steps"])
        tik_lam = float(group["tikhonov_lambda"])

        shape_buckets: dict[tuple[tuple[int, ...], bool], list[nn.Parameter]] = {}
        for p in group["params"]:
            if id(p) in self._complex_ids or p.grad is None:
                continue
            shape_buckets.setdefault((tuple(p.shape), id(p) in self.unnormalized_param_ids), []).append(p)

        diag_accum: dict[str, float] = {}
        diag_weight_sum: int = 0
        for (_, is_unnorm), plist in shape_buckets.items():
            w_stack = torch.stack([p.data for p in plist], dim=0)
            g_stack = torch.stack([p.grad for p in plist], dim=0)
            m_list: list[torch.Tensor] = [
                self.state[p].setdefault("momentum_buffer", torch.zeros_like(p, dtype=torch.float32)) for p in plist
            ]
            m_stack = torch.stack(m_list, dim=0)
            orig_stack_shape = w_stack.shape
            if w_stack.ndim == 4:
                flat_b = orig_stack_shape[0] * orig_stack_shape[1]
                w_stack = w_stack.reshape(flat_b, orig_stack_shape[2], orig_stack_shape[3])
                g_stack = g_stack.reshape(flat_b, orig_stack_shape[2], orig_stack_shape[3])
                m_stack = m_stack.reshape(flat_b, orig_stack_shape[2], orig_stack_shape[3])
            if is_unnorm:
                euclidean_frobenius_normed_muon_step(
                    w_stack, g_stack, m_stack, lr=stiefel_lr, beta=beta, ns_steps=ns_steps, tikhonov_lambda=tik_lam
                )
            elif self.collect_diagnostics or self.momentum_mode != StiefelMomentumMode.EUCLIDEAN_AMBIENT:
                w_prev = w_stack.clone() if self.collect_diagnostics else w_stack
                m_pre = m_stack.clone() if self.collect_diagnostics else m_stack
                u_tan, m_updated = riemannian_stiefel_tangent_transport_muon_step(
                    w_stack, g_stack, m_stack, lr=stiefel_lr, beta=beta, ns_steps=ns_steps, tikhonov_lambda=tik_lam, momentum_mode=self.momentum_mode
                )
                if self.collect_diagnostics:
                    b_diag = compute_real_stiefel_vector_diagnostics(
                        w_prev, g_stack, m_pre, m_updated, m_stack, u_tan, w_stack
                    )
                    diag_weight_sum += len(plist)
                    for k, v in b_diag.items():
                        diag_accum[k] = diag_accum.get(k, 0.0) + float(v) * len(plist)
            else:
                riemannian_stiefel_muon_step(
                    w_stack, g_stack, m_stack, lr=stiefel_lr, beta=beta, ns_steps=ns_steps, tikhonov_lambda=tik_lam
                )
            w_stack = w_stack.reshape(orig_stack_shape)
            m_stack = m_stack.reshape(orig_stack_shape)
            torch._foreach_copy_([p.data for p in plist], list(w_stack.unbind(0)))
            torch._foreach_copy_(m_list, list(m_stack.unbind(0)))

        if self.collect_diagnostics and diag_weight_sum > 0:
            self._last_step_diagnostics = {k: float(v / diag_weight_sum) for k, v in diag_accum.items()}

        complex_buckets: dict[tuple[int, int], list[tuple[nn.Parameter, nn.Parameter]]] = {}
        for u_re, u_im in self.complex_operator_pairs:
            if u_re.grad is not None and u_im.grad is not None:
                complex_buckets.setdefault((int(u_re.size(-2)), int(u_re.size(-1))), []).append((u_re, u_im))

        for _, active_pairs in complex_buckets.items():
            u_re_list = [u_re.data for u_re, _ in active_pairs]
            u_im_list = [u_im.data for _, u_im in active_pairs]
            g_re_list = [u_re.grad for u_re, _ in active_pairs if u_re.grad is not None]
            g_im_list = [u_im.grad for _, u_im in active_pairs if u_im.grad is not None]
            m_re_list: list[torch.Tensor] = []
            m_im_list: list[torch.Tensor] = []
            for u_re, u_im in active_pairs:
                st_re = self.state[u_re]
                st_im = self.state[u_im]
                if "momentum_buffer" not in st_re:
                    st_re["momentum_buffer"] = torch.zeros_like(u_re, dtype=torch.float32)
                if "momentum_buffer" not in st_im:
                    st_im["momentum_buffer"] = torch.zeros_like(u_im, dtype=torch.float32)
                m_re_list.append(st_re["momentum_buffer"])
                m_im_list.append(st_im["momentum_buffer"])

            ur_cat = torch.cat(u_re_list, dim=0)
            ui_cat = torch.cat(u_im_list, dim=0)
            gr_cat = torch.cat(g_re_list, dim=0)
            gi_cat = torch.cat(g_im_list, dim=0)
            mr_cat = torch.cat(m_re_list, dim=0)
            mi_cat = torch.cat(m_im_list, dim=0)
            if self.collect_diagnostics:
                ur_f, ui_f = ur_cat.float(), ui_cat.float()
                gr_f, gi_f = gr_cat.float(), gi_cat.float()
                mr_f, mi_f = mr_cat.float(), mi_cat.float()
                gt_re, gt_im = project_complex_stiefel_tangent(gr_f, gi_f, ur_f, ui_f)
                gn_norm = ((gr_f - gt_re).square() + (gi_f - gt_im).square()).sum(dim=(-2, -1)).sqrt()
                g_tot = (gr_f.square() + gi_f.square()).sum(dim=(-2, -1)).sqrt().clamp_min(1e-8)
                mt_re, mt_im = project_complex_stiefel_tangent(mr_f, mi_f, ur_f, ui_f)
                mn_norm = ((mr_f - mt_re).square() + (mi_f - mt_im).square()).sum(dim=(-2, -1)).sqrt()
                m_tot = (mr_f.square() + mi_f.square()).sum(dim=(-2, -1)).sqrt().clamp_min(1e-8)
                self._last_step_diagnostics["unitary_grad_normal_frac"] = float((gn_norm / g_tot).mean().item())
                self._last_step_diagnostics["unitary_pre_mom_normal_frac"] = float((mn_norm / m_tot).mean().item())
            if self.momentum_mode != StiefelMomentumMode.EUCLIDEAN_AMBIENT:
                riemannian_complex_tangent_transport_muon_step(
                    ur_cat, ui_cat, gr_cat, gi_cat, mr_cat, mi_cat, lr=(unitary_lr, beta), momentum_mode=self.momentum_mode
                )
            elif ur_cat.size(-2) == ur_cat.size(-1):
                riemannian_complex_unitary_muon_step(
                    ur_cat, ui_cat, gr_cat, gi_cat, mr_cat, mi_cat, lr=unitary_lr, tikhonov_lambda=tik_lam
                )
            else:
                riemannian_complex_kraus_stiefel_muon_step(
                    ur_cat, ui_cat, gr_cat, gi_cat, mr_cat, mi_cat, lr=unitary_lr, tikhonov_lambda=tik_lam
                )
            if self.collect_diagnostics:
                ur_n, ui_n = ur_cat.float(), ui_cat.float()
                mr_n, mi_n = mr_cat.float(), mi_cat.float()
                mtn_re, mtn_im = project_complex_stiefel_tangent(mr_n, mi_n, ur_n, ui_n)
                mn_post = ((mr_n - mtn_re).square() + (mi_n - mtn_im).square()).sum(dim=(-2, -1)).sqrt()
                m_post_tot = (mr_n.square() + mi_n.square()).sum(dim=(-2, -1)).sqrt().clamp_min(1e-8)
                self._last_step_diagnostics["unitary_post_mom_normal_frac"] = float((mn_post / m_post_tot).mean().item())
            splits = [u.size(0) for u in u_re_list]
            torch._foreach_copy_(u_re_list, list(ur_cat.split(splits, dim=0)))
            torch._foreach_copy_(u_im_list, list(ui_cat.split(splits, dim=0)))
            torch._foreach_copy_(m_re_list, list(mr_cat.split(splits, dim=0)))
            torch._foreach_copy_(m_im_list, list(mi_cat.split(splits, dim=0)))

        self.collect_diagnostics = False
        return loss


class ModularQuantumStatMechGPT(nn.Module):
    """Modular Teacher-Free $L$-Layer Quantum Statistical-Mechanics Language Model (`Wave 49` / `EXPERIMENT_PLAN_064.md`)."""

    def __init__(
        self,
        cfg: ModularQuantumStatMechConfig | None = None,
        kraus_rank: int = DEFAULT_KRAUS_CHANNEL_RANK,
    ) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else ModularQuantumStatMechConfig()
        self.vocab_size = self.cfg.vocab_size
        self.embed_dim = self.cfg.n_embd
        self.num_layers = self.cfg.n_layer
        self.num_heads = self.cfg.n_head
        self.d_h = DEFAULT_STATMECH_HEAD_COMPLEX_DIM
        self.v_rank = DEFAULT_STATMECH_VALUE_BRANCHES
        self.wp_channels = max(1, self.embed_dim // self.d_h)
        self.wp_complex_modes = max(1, self.wp_channels // 2)
        self.embedding_train_mode = EmbeddingTrainMode.TRAINABLE_SPHERE_MANIFOLD
        self.vocab_optim_mode = VocabOptimizerMode.BATCH_SUBSPACE_GRAM_MUON

        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)
        with torch.no_grad():
            nn.init.normal_(self.wte.weight, mean=0.0, std=1.0 / math.sqrt(float(self.embed_dim)))
            project_rows_to_rms_sphere_(self.wte.weight)

        self.w_in_stiefel = nn.Parameter(torch.eye(self.embed_dim, dtype=torch.float32))
        self.w_in_pkt: nn.Parameter | None = None
        self.w_in_mode: nn.Parameter | None = None
        self.kraus_rank = int(kraus_rank)
        self.in_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.blocks = nn.ModuleList(
            [ModularQuantumStatMechBlock(self.cfg, layer_idx=l_idx, kraus_rank=kraus_rank) for l_idx in range(self.num_layers)]
        )

        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)
        with torch.no_grad():
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=1.0 / math.sqrt(float(self.embed_dim)))
            project_rows_to_rms_sphere_(self.lm_head.weight)

        self.logit_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(float(self.embed_dim)), dtype=torch.float32))
        self.born_tau = nn.Parameter(torch.tensor(DEFAULT_KRYLOV_BORN_LOG_TEMP, dtype=torch.float32))
        self.trace_beta = nn.Parameter(
            torch.tensor(DEFAULT_ISOMETRIC_TRACE_INIT_BETA, dtype=torch.float32),
            requires_grad=(self.cfg.readout_mode == VocabularyReadoutMode.MEAN_CENTERED_ISOMETRIC_TRACE),
        )
        n_groups = max(1, self.wp_complex_modes // 4)
        init_wp_re = torch.tensor([0.15, 0.10, 0.08, 0.05], dtype=torch.float32).repeat(n_groups)[: self.wp_complex_modes]
        init_wp_im = torch.tensor([0.00, 0.10, -0.08, 0.05], dtype=torch.float32).repeat(n_groups)[: self.wp_complex_modes]
        self.wp_causal_re = nn.Parameter(init_wp_re)
        self.wp_causal_im = nn.Parameter(init_wp_im)
        self._unnorm_stiefel_ids: set[int] = set()

    @property
    def is_native_wavepacket_mode(self) -> bool:
        """Return True when the model maintains a native `(B, T, 48, 16)` state across all $L$ layers."""
        return self.embedding_train_mode in (
            EmbeddingTrainMode.NATIVE_WAVEPACKET_48X16,
            EmbeddingTrainMode.NATIVE_CAUSAL_WAVEPACKET_48X16,
        )

    def configure_embedding_mode(
        self,
        mode: EmbeddingTrainMode,
        ppmi_embeddings: torch.Tensor | None = None,
    ) -> None:
        """Configure `wte` trainability, `(50304, 48, 16)` wavepacket geometry, or native `(B, T, 48, 16)` layers."""
        self.embedding_train_mode = mode
        dev = self.wte.weight.device
        with torch.no_grad():
            if mode == EmbeddingTrainMode.FROZEN_PPMI_SVD_SPHERE and ppmi_embeddings is not None:
                self.wte.weight.copy_(ppmi_embeddings.to(device=dev, dtype=self.wte.weight.dtype))
                project_rows_to_rms_sphere_(self.wte.weight)
            elif mode in (
                EmbeddingTrainMode.WAVEPACKET_SPHERE_48X16,
                EmbeddingTrainMode.CAUSAL_WAVEPACKET_48X16,
                EmbeddingTrainMode.NATIVE_WAVEPACKET_48X16,
                EmbeddingTrainMode.NATIVE_CAUSAL_WAVEPACKET_48X16,
            ):
                wp_w = self.wte.weight.detach().reshape(self.vocab_size, self.wp_channels, self.d_h).clone()
                used_ppmi = initialize_wavepacket_ppmi_sphere_embeddings_(wp_w, ppmi_embeddings=ppmi_embeddings)
                self.wte.weight.set_(wp_w)
                if used_ppmi:
                    self.lm_head.weight.copy_(
                        wp_w.reshape(self.vocab_size, self.embed_dim).to(dtype=self.lm_head.weight.dtype)
                    )
                if self.is_native_wavepacket_mode and self.w_in_pkt is None:
                    eye_pkt = torch.eye(self.d_h, device=dev, dtype=torch.float32).unsqueeze(0).expand(self.wp_channels, -1, -1).clone()
                    eye_mode = torch.eye(self.wp_channels, device=dev, dtype=torch.float32).unsqueeze(0).expand(self.d_h, -1, -1).clone()
                    self.w_in_pkt = nn.Parameter(eye_pkt)
                    self.w_in_mode = nn.Parameter(eye_mode)
                    self.w_in_stiefel.requires_grad_(False)
                    self.blocks = nn.ModuleList(
                        [
                            NativeWavepacketStatMechBlock(self.cfg, layer_idx=l_idx, kraus_rank=self.kraus_rank).to(dev)
                            for l_idx in range(self.num_layers)
                        ]
                    )
            self.wte.weight.requires_grad_(
                mode
                not in (
                    EmbeddingTrainMode.FROZEN_RANDOM_SPHERE,
                    EmbeddingTrainMode.FROZEN_PPMI_SVD_SPHERE,
                )
            )

    def set_compiled_krylov(self, enabled: bool = True) -> None:
        """Enable or disable `torch.compile` fusion across all blocks."""
        for blk in self.blocks:
            blk.use_compiled_krylov = bool(enabled)

    def all_real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Collect all real Stiefel parameters across input injection and all $L$ blocks."""
        if self.is_native_wavepacket_mode and self.w_in_pkt is not None and self.w_in_mode is not None:
            params: list[nn.Parameter] = [self.w_in_pkt, self.w_in_mode]
        else:
            params = [self.w_in_stiefel]
        for blk in self.blocks:
            params.extend(blk.real_stiefel_parameters())
        return params

    def all_complex_unitary_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        """Collect all 3D complex head Unitary/Kraus pairs `(head_unitary_re, head_unitary_im)` across all $L$ blocks."""
        return [blk.complex_unitary_pair() for blk in self.blocks]

    def all_scalar_parameters(self) -> list[nn.Parameter]:
        """Collect all 1D scalar gain, residual highway, temperature, and gate parameters."""
        params: list[nn.Parameter] = [
            self.in_scale,
            self.logit_scale,
            self.born_tau,
            self.trace_beta,
            self.wp_causal_re,
            self.wp_causal_im,
        ]
        for blk in self.blocks:
            params.extend(blk.scalar_parameters())
        return params

    @torch.no_grad()
    def restrict_scalar_parameters_(self) -> None:
        """Project 1D scalar gains, residual highway weights, and inverse temperatures onto bounded physical intervals."""
        self.in_scale.clamp_(0.50, 2.00)
        min_scale = 0.75 / math.sqrt(float(self.embed_dim))
        self.logit_scale.clamp_(min_scale, 8.0 * min_scale)
        self.born_tau.clamp_(-2.00, 3.00)
        self.trace_beta.clamp_(0.25, 16.00)
        self.wp_causal_re.clamp_(-1.50, 1.50)
        self.wp_causal_im.clamp_(-1.50, 1.50)
        for blk in self.blocks:
            blk.obs_scale.clamp_(0.02, 0.50)
            blk.chan_scale.clamp_(0.02, 0.50)
            blk.block_lambdas[0].clamp_(0.50, 1.50)
            blk.block_lambdas[1].clamp_(-0.50, 0.50)
            blk.v_lambdas[0].clamp_(0.50, 1.50)
            blk.v_lambdas[1].clamp_(-0.50, 0.50)
            blk.purify_logits.clamp_(-6.0, 6.0)
            blk.entropy_gate_params.clamp_(-8.0, 8.0)

    @torch.no_grad()
    def clip_riemannian_tangent_grad_norm_(self, max_norm: float = 1.0) -> float:
        """Clip gradients strictly by their intrinsic Riemannian tangent norm across Stiefel, Unitary, and Sphere manifolds."""
        sq_norms: list[torch.Tensor] = []
        for w in self.all_real_stiefel_parameters():
            if w.grad is None:
                continue
            wf, gf = w.data.float(), w.grad.float()
            if wf.size(-2) < wf.size(-1):
                wf, gf = wf.transpose(-2, -1), gf.transpose(-2, -1)
            wf_unit = wf / wf.norm(dim=-2, keepdim=True).clamp_min(1e-8)
            g_tan = project_real_stiefel_tangent(gf, wf_unit)
            sq_norms.append(g_tan.square().sum())

        for u_re, u_im in self.all_complex_unitary_pairs():
            if u_re.grad is None or u_im.grad is None:
                continue
            gt_re, gt_im = project_complex_stiefel_tangent(
                u_re.grad.float(), u_im.grad.float(), u_re.data.float(), u_im.data.float()
            )
            sq_norms.append(gt_re.square().sum() + gt_im.square().sum())

        for sp_w in (self.wte.weight, self.lm_head.weight):
            if sp_w.grad is None:
                continue
            g_tan_sp = _project_to_sphere_tangent(sp_w.grad.float(), sp_w.data.float())
            sq_norms.append(g_tan_sp.square().sum())

        for sc_p in self.all_scalar_parameters():
            if sc_p.grad is not None:
                sq_norms.append(sc_p.grad.float().square().sum())

        if not sq_norms:
            return 0.0
        total_norm = torch.stack(sq_norms).sum().sqrt()
        norm_val = float(total_norm.item())
        if math.isfinite(norm_val) and norm_val > float(max_norm):
            clip_coef = float(max_norm) / (norm_val + 1e-6)
            for p in self.parameters():
                if p.grad is not None:
                    p.grad.mul_(clip_coef)
        return norm_val

    @torch.no_grad()
    def compute_manifold_and_purity_telemetry(self) -> dict[str, float]:
        """Measure Stiefel error, Unitary/Kraus Stiefel error, sphere RMS error, purity, and Krylov Rayleigh quotients."""
        stiefel_errs: list[float] = []
        col_scale_stds: list[float] = []
        for w in self.all_real_stiefel_parameters():
            wf = w.float()
            if wf.size(-2) < wf.size(-1):
                wf = wf.transpose(-2, -1)
            col_norms = wf.norm(dim=-2, keepdim=True).clamp_min(1e-8)
            wf_unit = wf / col_norms
            d_cols = wf.size(-1)
            eye = torch.eye(d_cols, device=wf.device, dtype=wf.dtype)
            gram_diff = torch.matmul(wf_unit.transpose(-2, -1), wf_unit) - eye
            ortho_err = (
                0.0
                if id(w) in self._unnorm_stiefel_ids
                else float(torch.linalg.matrix_norm(gram_diff, ord="fro", dim=(-2, -1)).mean().item()) / math.sqrt(float(d_cols))
            )
            rms_err = float((col_norms.square().mean(dim=(-2, -1)).sqrt() - 1.0).abs().mean().item())
            stiefel_errs.append(ortho_err + rms_err)
            col_scale_stds.append(float(col_norms.std().item()))

        unitary_errs: list[float] = []
        for u_re, u_im in self.all_complex_unitary_pairs():
            ur = u_re.float()
            ui = u_im.float()
            d_h = ur.size(-1)
            eye_h = torch.eye(d_h, device=ur.device, dtype=ur.dtype).unsqueeze(0)
            gram_re = torch.matmul(ur.transpose(-2, -1), ur) + torch.matmul(ui.transpose(-2, -1), ui)
            gram_im = torch.matmul(ur.transpose(-2, -1), ui) - torch.matmul(ui.transpose(-2, -1), ur)
            u_err = float(
                ((gram_re - eye_h).square().sum(dim=(-2, -1)) + gram_im.square().sum(dim=(-2, -1))).sqrt().mean().item()
            ) / math.sqrt(float(d_h))
            unitary_errs.append(u_err)

        target_radius = math.sqrt(float(self.embed_dim))
        wte_target_radius = math.sqrt(float(self.wte.weight.size(-1)))
        wte_rms_err = float((self.wte.weight.float().norm(dim=-1) / wte_target_radius - 1.0).abs().max().item())
        head_row_scales = self.lm_head.weight.float().norm(dim=-1) / target_radius
        head_rms_err = float((head_row_scales.square().mean().sqrt() - 1.0).abs().item())
        head_scale_std = float(head_row_scales.std().item())

        blk_tel = [blk.get_last_telemetry_dict() for blk in self.blocks]
        purities = [t["mean_purity"] for t in blk_tel]
        eff_ranks = [t["mean_eff_rank"] for t in blk_tel]
        lam1s = [t["mean_lambda1"] for t in blk_tel]
        lam2s = [t["mean_lambda2"] for t in blk_tel]
        return {
            "mean_stiefel_err": float(sum(stiefel_errs) / max(1, len(stiefel_errs))),
            "max_stiefel_err": float(max(stiefel_errs) if stiefel_errs else 0.0),
            "mean_stiefel_col_scale_std": float(sum(col_scale_stds) / max(1, len(col_scale_stds))),
            "mean_unitary_err": float(sum(unitary_errs) / max(1, len(unitary_errs))),
            "max_unitary_err": float(max(unitary_errs) if unitary_errs else 0.0),
            "max_wte_sphere_rms_err": wte_rms_err,
            "max_head_sphere_rms_err": head_rms_err,
            "head_row_scale_std": head_scale_std,
            "mean_purity": float(sum(purities) / max(1, len(purities))),
            "mean_eff_rank": float(sum(eff_ranks) / max(1, len(eff_ranks))),
            "mean_lambda1": float(sum(lam1s) / max(1, len(lam1s))),
            "mean_lambda2": float(sum(lam2s) / max(1, len(lam2s))),
        }

    def forward_hidden(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute `(h_final, krylov_weights_bt2, last_rho_trace1_bt)`."""
        bsz, seq_len = idx.shape
        work_dtype = torch.bfloat16 if (idx.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
        krylov_weights_bt2 = torch.full((bsz, seq_len, 2), 0.5, device=idx.device, dtype=torch.float32)
        rho_trace1_bt = torch.zeros(
            (bsz, seq_len, self.num_heads * self.d_h * self.d_h),
            device=idx.device,
            dtype=work_dtype,
        )
        v_0: torch.Tensor | None = None

        if self.is_native_wavepacket_mode and self.w_in_pkt is not None and self.w_in_mode is not None:
            raw_wp = (
                F.embedding(idx, self.wte.weight.view(self.vocab_size, self.embed_dim))
                .view(bsz, seq_len, self.wp_channels, self.d_h)
                .to(work_dtype)
            )
            if self.embedding_train_mode == EmbeddingTrainMode.NATIVE_CAUSAL_WAVEPACKET_48X16 and seq_len > 1:
                raw_wp = apply_causal_wavepacket_dispersion(raw_wp, self.wp_causal_re, self.wp_causal_im)
            tok_wp = F.rms_norm(raw_wp, (self.d_h,), eps=TRACE_NORM_EPS)
            x_wp = (
                apply_factored_wavepacket_stiefel(tok_wp, self.w_in_pkt.to(work_dtype), self.w_in_mode.to(work_dtype))
                * self.in_scale.to(work_dtype)
            )
            x_0_wp = x_wp
            for blk in self.blocks:
                x_wp, v_0, krylov_weights_bt2 = blk(x_wp, x_0=x_0_wp, v_0=v_0)
            h_final = F.rms_norm(x_wp, (self.d_h,), eps=TRACE_NORM_EPS).reshape(bsz, seq_len, self.embed_dim)
            return h_final, krylov_weights_bt2, rho_trace1_bt

        if self.wte.weight.ndim == 3:
            raw_wp = (
                F.embedding(idx, self.wte.weight.view(self.vocab_size, self.embed_dim))
                .view(bsz, seq_len, self.wp_channels, self.d_h)
                .to(work_dtype)
            )
            if self.embedding_train_mode == EmbeddingTrainMode.CAUSAL_WAVEPACKET_48X16 and seq_len > 1:
                raw_wp = apply_causal_wavepacket_dispersion(raw_wp, self.wp_causal_re, self.wp_causal_im)
            tok_emb = F.rms_norm(raw_wp, (self.d_h,), eps=TRACE_NORM_EPS).reshape(bsz, seq_len, self.embed_dim)
        else:
            tok_emb = F.rms_norm(self.wte(idx).to(work_dtype), (self.embed_dim,), eps=TRACE_NORM_EPS)
        x = torch.matmul(tok_emb, self.w_in_stiefel.to(work_dtype)) * self.in_scale.to(work_dtype)
        x_0 = x
        for blk in self.blocks:
            x, v_0, krylov_weights_bt2, rho_trace1_bt = blk(x, x_0=x_0, v_0=v_0)

        h_final = F.rms_norm(x, (self.embed_dim,), eps=TRACE_NORM_EPS)
        return h_final, krylov_weights_bt2, rho_trace1_bt

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Forward pass supporting `SOFTCAP_LINEAR_CE`, `RANK2_KRYLOV_BORN`, and `MEAN_CENTERED_ISOMETRIC_TRACE`."""
        h_final, krylov_weights_bt2, rho_trace1_bt = self.forward_hidden(idx)
        loss: torch.Tensor | None = None
        logits: torch.Tensor | None = None

        if self.cfg.readout_mode == VocabularyReadoutMode.RANK2_KRYLOV_BORN:
            if targets is not None:
                loss = Rank2KrylovBornChunkedCEFn.apply(
                    h_final.reshape(-1, self.embed_dim), krylov_weights_bt2.reshape(-1, 2),
                    self.lm_head.weight, targets.reshape(-1), self.born_tau, DEFAULT_LOGIT_SOFTCAP, DEFAULT_CE_CHUNK_TOKENS,
                )
            if return_logits or targets is None:
                logits = eval_rank2_krylov_born_logits(
                    h_final, krylov_weights_bt2, self.lm_head.weight, self.born_tau, softcap=DEFAULT_LOGIT_SOFTCAP
                )
        elif self.cfg.readout_mode == VocabularyReadoutMode.MEAN_CENTERED_ISOMETRIC_TRACE:
            min_scale = 0.75 / math.sqrt(float(self.embed_dim))
            eff_scale = self.logit_scale.clamp(min=min_scale, max=8.0 * min_scale).to(self.lm_head.weight.dtype)
            w_lm_scaled = self.lm_head.weight * eff_scale
            last_blk = self.blocks[-1]
            p_h_final = pack_multihead_wavepacket_density_isometric(
                h_final, self.num_heads, self.v_rank, self.d_h, last_blk.re_pack_mask, last_blk.im_pack_mask
            )
            p_rho_eff = 0.5 * rho_trace1_bt + 0.5 * p_h_final
            p_e_vocab = pack_multihead_wavepacket_density_isometric(
                self.lm_head.weight.to(h_final.dtype), self.num_heads, self.v_rank, self.d_h, last_blk.re_pack_mask, last_blk.im_pack_mask
            )
            if targets is not None:
                loss = MeanCenteredIsometricBornTraceCEFn.apply(
                    h_final.reshape(-1, self.embed_dim), p_rho_eff.reshape(-1, self.num_heads * self.d_h * self.d_h),
                    w_lm_scaled, p_e_vocab, targets.reshape(-1), self.trace_beta, self.num_heads,
                )
            if return_logits or targets is None:
                logits = eval_mean_centered_isometric_trace_logits(
                    h_final, p_rho_eff, w_lm_scaled, p_e_vocab, self.trace_beta, num_heads=self.num_heads
                )
        else:
            min_scale = 0.75 / math.sqrt(float(self.embed_dim))
            eff_scale = self.logit_scale.clamp(min=min_scale, max=8.0 * min_scale).to(self.lm_head.weight.dtype)
            w_lm_scaled = self.lm_head.weight * eff_scale
            if targets is not None:
                loss = _CompiledChunkedSoftcapLinearCEFn.apply(
                    h_final.reshape(-1, self.embed_dim), w_lm_scaled, targets.reshape(-1), DEFAULT_LOGIT_SOFTCAP, DEFAULT_CE_CHUNK_TOKENS
                )
            if return_logits or targets is None:
                raw_logits = F.linear(h_final, w_lm_scaled.to(h_final.dtype)).float()
                logits = DEFAULT_LOGIT_SOFTCAP * torch.tanh(raw_logits / DEFAULT_LOGIT_SOFTCAP)

        return logits, loss


def build_modular_quantum_statmech_optimizers(
    model: ModularQuantumStatMechGPT,
    stiefel_lr: float = DEFAULT_STIEFEL_GEODESIC_LR,
    unitary_lr: float = DEFAULT_UNITARY_GEODESIC_LR,
    vocab_lr: float | None = None,
    scalar_lr: float = DEFAULT_SCALAR_ADAM_LR,
    tikhonov_lambda: float = DEFAULT_TIKHONOV_GRAM_LAMBDA,
    momentum_mode: StiefelMomentumMode = StiefelMomentumMode.RIEMANNIAN_TANGENT_TRANSPORTED,
) -> Any:
    """Build `StandaloneRiemannianOptimizerBundle` for `ModularQuantumStatMechGPT` (`<= 8` args, Rule 2)."""
    from .standalone_quantum_born_gpt import StandaloneRiemannianOptimizerBundle

    trunk_opt = ModularRiemannianStiefelKrausMuon(
        real_stiefel_params=model.all_real_stiefel_parameters(),
        complex_operator_pairs=model.all_complex_unitary_pairs(),
        stiefel_lr=stiefel_lr,
        unitary_lr=unitary_lr,
        tikhonov_lambda=tikhonov_lambda,
        momentum_mode=momentum_mode,
    )
    if momentum_mode == StiefelMomentumMode.UNNORMALIZED_FINAL_BLOCK_AND_HEAD and len(model.blocks) > 0:
        last_blk = model.blocks[-1]
        unnorm_ids = {id(last_blk.w_obs), id(last_blk.w_chan)}
        trunk_opt.unnormalized_param_ids = unnorm_ids
        model._unnorm_stiefel_ids = unnorm_ids

    scalar_groups: list[dict[str, Any]] = [
        {"params": [p for p in model.all_scalar_parameters() if p.requires_grad], "lr": float(scalar_lr), "weight_decay": 0.0}
    ]
    wte_opt: BatchSubspaceTikhonovGramMuon | RiemannianSphereTangentSNROptimizer | None = None
    head_opt: BatchSubspaceTikhonovGramMuon | RiemannianSphereTangentSNROptimizer | None = None

    mode = model.vocab_optim_mode
    wte_trainable = bool(model.wte.weight.requires_grad)
    head_radial_ratio = (
        DEFAULT_VOCAB_RADIAL_LR_RATIO
        if momentum_mode in (StiefelMomentumMode.UNNORMALIZED_FINAL_READOUT_HEAD, StiefelMomentumMode.UNNORMALIZED_FINAL_BLOCK_AND_HEAD)
        else 0.0
    )
    if mode == VocabOptimizerMode.BATCH_SUBSPACE_GRAM_MUON:
        eff_vocab_lr = float(vocab_lr) if vocab_lr is not None else DEFAULT_BATCH_GRAM_MUON_LR
        if wte_trainable:
            wte_opt = BatchSubspaceTikhonovGramMuon([model.wte.weight], lr=eff_vocab_lr, tikhonov_lambda=tikhonov_lambda)
        head_opt = BatchSubspaceTikhonovGramMuon(
            [model.lm_head.weight], lr=eff_vocab_lr, tikhonov_lambda=tikhonov_lambda, radial_lr_ratio=head_radial_ratio
        )
    elif mode == VocabOptimizerMode.SPHERE_TANGENT_SNR_MUON:
        eff_vocab_lr = float(vocab_lr) if vocab_lr is not None else DEFAULT_SPHERE_GEODESIC_LR
        if wte_trainable:
            wte_opt = RiemannianSphereTangentSNROptimizer([model.wte.weight], lr=eff_vocab_lr, snr_tau=DEFAULT_SPHERE_SNR_TRUST_TAU)
        head_opt = RiemannianSphereTangentSNROptimizer([model.lm_head.weight], lr=eff_vocab_lr, snr_tau=DEFAULT_SPHERE_SNR_TRUST_TAU)
    else:
        eff_vocab_lr = float(vocab_lr) if vocab_lr is not None else DEFAULT_ADAMW_VOCAB_LR
        d_scale = math.sqrt(float(model.embed_dim))
        if wte_trainable:
            scalar_groups.append({"params": [model.wte.weight], "lr": eff_vocab_lr * d_scale, "weight_decay": 0.0})
        scalar_groups.append({"params": [model.lm_head.weight], "lr": eff_vocab_lr * d_scale, "weight_decay": 0.0})

    scalar_opt = torch.optim.AdamW(scalar_groups, betas=(0.9, 0.95), eps=1e-8)
    bundle = StandaloneRiemannianOptimizerBundle(
        trunk_opt=trunk_opt, scalar_opt=scalar_opt, vocab_optim_mode=mode, wte_opt=wte_opt, head_opt=head_opt  # type: ignore[arg-type]
    )
    bundle.scalar_post_step_hook = model.restrict_scalar_parameters_
    return bundle


__all__ = [
    "DEFAULT_STATMECH_EMBED_DIM", "DEFAULT_STATMECH_HEAD_COMPLEX_DIM", "DEFAULT_STATMECH_NUM_HEADS",
    "DEFAULT_STATMECH_NUM_LAYERS", "DEFAULT_STATMECH_VALUE_BRANCHES", "DEFAULT_STATMECH_VOCAB_SIZE",
    "EmbeddingTrainMode", "ModularQuantumStatMechBlock", "ModularQuantumStatMechConfig",
    "ModularQuantumStatMechGPT", "ModularRiemannianStiefelKrausMuon", "build_modular_quantum_statmech_optimizers",
]
