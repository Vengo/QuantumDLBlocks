"""Level-Separated Wavelet Tensors & Multi-Rate 1D Wavefunction Convolution GPT (`Waves 32–33` / `Plans 047–048`).

When to Use This Building Block:
- Use `LevelSeparatedWaveletConvGPT`, `PreAttnMultiRateWaveletConvBlock`, and
  `synthesize_level_separated_wavelet_state` when you want:
  1. The 4 dyadic wavelet levels (`coarse_256`, `mid_2048`, `sub_8192`, `fine_65536`) represented as
     explicitly separate tensors in code (`WaveletScaleState`), each normalized on its own unit
     sphere `(1 / sqrt(4)) S^{d/4 - 1}` so coarse semantic clustering in `wav_256` can never
     cannibalize the L2 norm of fine lexical leaves in `wav_65536`.
  2. Every backbone layer (`PreAttnMultiRateWaveletConvBlock`) to evolve tokens through a causal 1D
     strided/dilated wavefunction convolution bank (`MultiRateWavefunctionConvBank`, temporal
     dilations `(4, 2, 1, 1)`) *before* feeding into Quantum Born Attention
     (`SubspaceProjectorBornAttention`), completely replacing the post-attention channel U-Net.
  3. Progress-ramped multi-subspace coincidence warmup (`coincidence_gamma(tau) = 0.22 * tau`,
     `subspace_var_penalty(tau) = 0.14 * tau`) so the 65,536 fine leaves accumulate sufficient
     token updates before full 4-way coincidence gating kicks in.

Why It Was Built & Mathematical Provenance (`EXPERIMENT_PLAN_047.md` & `EXPERIMENT_PLAN_048.md` / `Waves 32–33`):
- Directly addresses the root causes identified across `Waves 30–32`:
  1. Separating the 4 wavelet levels into distinct tensors in `WaveletScaleState` locks each scale
     at exact `25.00%` energy (`13.01 / 64` coarse semantic rank vs. `33.79 / 64` fine leaf rank).
  2. Moving `MultiRateWavefunctionConvBank` inside every backbone block prior to Born attention
     gives queries and keys an expanding multi-rate temporal receptive field across layers while
     eliminating post-attention channel Sweldens U-Net overhead.
  3. Progress-ramped coincidence warmup eliminates the early-step `+0.14` nat unconverged-leaf
     penalty at step 150 while delivering the full multi-subspace coincidence margin boost over
     `1,800`-step (`14.75M`-token) and `3,600`-step (`29.49M`-token) horizons.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..functional import l2_normalize
from ..layers.attention import SubspaceProjectorBornAttention, compute_document_ids
from ..layers.multirate_wavefunction_conv import (
    DEFAULT_LEVEL_TEMPORAL_STRIDES,
    MultiRateWavefunctionConvBank,
    NUM_WAVELET_PYRAMID_LEVELS,
    WaveletScaleState,
)
from ..layers.overcomplete_two_beam import OvercompleteTwoBeamInterference
from ..layers.subspace_coincidence_born import (
    SubspaceCoincidenceSpec,
    blend_subspace_l2_normalize,
    eval_subspace_coincidence_full_logits,
    subspace_coincidence_dyadic_chunked_ce,
)
from ..layers.wavelet_cholesky_branch import (
    BlockCholeskyLinear,
    WaveletCholeskyConfig,
    WaveletCholeskyMultiscaleBranch,
    apply_soft_powerlaw_spectral_floor,
)
from ..optim.muon import StiefelMuon
from .parseval_mup_ngram_speedrun_gpt import update_parseval_mup_ngram_learning_rates
from .parseval_mup_subspace_coincidence_gpt import (
    DEFAULT_SUBSPACE_HEAD_LR,
    DEFAULT_SUBSPACE_MUON_LR,
    DEFAULT_SUBSPACE_WAV_LR,
    MAX_INLINE_LOGIT_TOKENS,
    ParsevalMuPSubspaceCoincidenceConfig,
    ParsevalMuPSubspaceCoincidenceGPT,
    build_subspace_coincidence_optimizers,
    initialize_subspace_coincidence_with_ppmi_svd,
)
from .wavelet_cholesky_2pow16_gpt import Radical2Pow16Config, Tied2Pow16WaveletCholeskyPyramid

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Documented in EXPERIMENT_PLAN_047.md, EXPERIMENT_PLAN_048.md, & EXPERIMENT_PLAN_049.md
# ==============================================================================
LEVEL_ORTHO_RESIDUAL_WEIGHT: float = 0.08
LEVEL_ORTHO_RESIDUAL_MIN_WEIGHT: float = 0.01
INIT_PRE_ATTN_CONV_GATE: float = 0.04
DEFAULT_WARMED_BASE_SUBSPACE_BLEND_MAX: float = 0.12
PRE_ATTN_BLOCK_INTERFEROMETER_MIX: float = 0.55
PRE_ATTN_BLOCK_CONV_INIT_SCALE: float = 0.06
PRE_ATTN_BLOCK_ATTN_INIT_SCALE: float = 0.35
INIT_COARSE_CAUSAL_GATE: float = 0.32
DEFAULT_FLASH_BORN_HEAD_DIM: int = 64
SYMPLECTIC_PAIR_DIM: int = 2
MIN_AUTO_HEAD_DIM_WIDTH: int = 128
CKPT_BLOCK_MIN_EMBED_DIM: int = 1024
CKPT_BLOCK_MIN_BATCH_SIZE: int = 12
UP_FRAME_LANDING_SCALE: float = 0.5


def _fold_to_level_dim(x: torch.Tensor, level_dim: int) -> torch.Tensor:
    """Fold a tensor `x` of shape `(..., d)` into `(..., level_dim)` while preserving symplectic `(even, odd)` pairs.

    Updated in `Wave 34` (`EXPERIMENT_PLAN_049.md` Section 3.1.1): reshapes `x` into
    `(..., level_dim // 2, d // level_dim, 2)` and averages across `dim=-2` so even coordinates
    only average with even coordinates and odd coordinates only average with odd coordinates,
    preserving symplectic orthogonality `<psi_u, J psi_u> = 0` and preventing step-0 rank collapse.
    """
    d_in = x.size(-1)
    if d_in == level_dim:
        return x
    if d_in > level_dim and (d_in % level_dim == 0):
        ratio = d_in // level_dim
        lead_shape = x.shape[:-1]
        if level_dim % SYMPLECTIC_PAIR_DIM == 0:
            x_pairs = x.reshape(*lead_shape, level_dim // SYMPLECTIC_PAIR_DIM, ratio, SYMPLECTIC_PAIR_DIM)
            return x_pairs.mean(dim=-2).reshape(*lead_shape, level_dim)
        return x.reshape(*lead_shape, level_dim, ratio).mean(dim=-1)
    return x[..., :level_dim]


def synthesize_level_separated_wavelet_state(
    pyramid: Tied2Pow16WaveletCholeskyPyramid,
    leaf_idx: torch.Tensor,
) -> WaveletScaleState:
    """Synthesize the 4 wavelet levels as explicitly separate tensors in `WaveletScaleState`."""
    flat_p = leaf_idx.reshape(-1)
    d = int(pyramid.cfg.embed_dim)
    level_dim = d // NUM_WAVELET_PYRAMID_LEVELS

    c3_tab = _fold_to_level_dim(pyramid.chol_256(pyramid.wav_256), level_dim)
    c3_lvl = c3_tab.index_select(0, flat_p >> 8).view(*leaf_idx.shape, level_dim)
    c2_tab = _fold_to_level_dim(pyramid.chol_2048(pyramid.wav_2048), level_dim)
    c2_lvl = c2_tab.index_select(0, flat_p >> 5).view(*leaf_idx.shape, level_dim)
    up8_lvl = _fold_to_level_dim(pyramid.up_8192, level_dim)
    c1_tab = F.linear(pyramid.chol_8192(pyramid.wav_8192), up8_lvl.t())
    c1_lvl = c1_tab.index_select(0, flat_p >> 3).view(*leaf_idx.shape, level_dim)
    if pyramid.num_fine_banks == 1:
        up64_lvl = _fold_to_level_dim(pyramid.up_65536, level_dim)
        w0_sel = pyramid.chol_65536(pyramid.wav_65536.index_select(0, flat_p))
        c0_lvl = F.linear(w0_sel, up64_lvl.t()).view(*leaf_idx.shape, level_dim)
    else:
        c0_lvl = _fold_to_level_dim(pyramid._lift_fine_wavelet(flat_p, leaf_idx.shape), level_dim)

    raw_state = WaveletScaleState(
        coarse_256=c3_lvl,
        mid_2048=c2_lvl,
        sub_8192=c1_lvl,
        fine_65536=c0_lvl,
    )
    return raw_state.normalize_levels()


class PreAttnMultiRateWaveletConvBlock(nn.Module):
    """Per-layer Pre-Attention 1D Wavefunction Conv + Purified Born Attention + Wavelet-Kronecker & Interferometer (`Wave 34`)."""

    def __init__(
        self,
        pyr_cfg: Radical2Pow16Config,
        layer_idx: int,
        temporal_strides: tuple[int, int, int, int] = DEFAULT_LEVEL_TEMPORAL_STRIDES,
    ) -> None:
        super().__init__()
        d = int(pyr_cfg.embed_dim)
        self.half_d = d // 2
        self.window_size = int(pyr_cfg.window_size)
        self.conv_bank = MultiRateWavefunctionConvBank(
            embed_dim=d,
            temporal_strides=temporal_strides,
            fast_spd_bmm=True,
        )
        self.attn = SubspaceProjectorBornAttention(
            embed_dim=d,
            num_heads=pyr_cfg.num_heads,
            block_size=pyr_cfg.block_size,
            is_causal=True,
            learnable_temperature=True,
            value_residual_mixing=True,
            use_fused_qkv=True,
            use_flash_born=True,
            use_purified_attn=True,
            use_mudd_and_gates=True,
        )
        self.wavelet = WaveletCholeskyMultiscaleBranch(
            WaveletCholeskyConfig(embed_dim=d, num_levels=3, max_block_size=32, fast_spd_bmm=True)
        )
        self.coarse_causal_chol = BlockCholeskyLinear(d >> 3, max_block_size=32, fast_spd_bmm=True)
        self.coarse_dilated_chol = BlockCholeskyLinear(d >> 3, max_block_size=32, fast_spd_bmm=True)
        self.mid_causal_chol = BlockCholeskyLinear(d >> 2, max_block_size=32, fast_spd_bmm=True)
        self.subband_shift_mix = nn.Parameter(torch.tensor([0.38, 0.22, 0.28], dtype=torch.float32))
        self.kron_beamsplitter_theta = nn.Parameter(torch.tensor(math.pi / 7.0, dtype=torch.float32))
        self.kron_modulation_chol = BlockCholeskyLinear(self.half_d, max_block_size=32, fast_spd_bmm=True)
        self.kron_reverse_chol = BlockCholeskyLinear(self.half_d, max_block_size=32, fast_spd_bmm=True)
        self.interf = OvercompleteTwoBeamInterference(
            embed_dim=d,
            frame_expansion=pyr_cfg.interferometer_expansion,
            use_cosine_phase_beam_gate=True,
            use_mach_zehnder_beamsplitter=True,
            use_wavelet_cholesky_branch=True,
        )
        if getattr(self.interf, "wavelet_cholesky_branch", None) is not None:
            for m in self.interf.wavelet_cholesky_branch.modules():
                if isinstance(m, BlockCholeskyLinear):
                    m.fast_spd_bmm = True
        depth_denom = math.sqrt(float(max(1, layer_idx + 1)))
        conv_init = PRE_ATTN_BLOCK_CONV_INIT_SCALE / depth_denom
        attn_init = PRE_ATTN_BLOCK_ATTN_INIT_SCALE / depth_denom
        self.pre_attn_conv_gate = nn.Parameter(torch.full((d,), conv_init))
        self.attn_gate = nn.Parameter(torch.full((d,), attn_init))
        self.fused_gate = nn.Parameter(torch.full((d,), attn_init))

    @property
    def pre_attn_conv(self) -> MultiRateWavefunctionConvBank:
        """Backward-compatible alias for `self.conv_bank`."""
        return self.conv_bank

    @property
    def interferometer(self) -> OvercompleteTwoBeamInterference:
        """Backward-compatible alias for `self.interf`."""
        return self.interf

    def _wavelet_kronecker_pathway(self, psi_a: torch.Tensor) -> torch.Tensor:
        """Multi-scale causal subband Cholesky + coarse-fine Kronecker entanglement pathway."""
        coarse, details = self.wavelet.downsample_amplitudes(psi_a)
        coarse_t, details_t = self.wavelet.apply_cholesky_scale_operators(coarse, details)
        mix_w = torch.sigmoid(self.subband_shift_mix).to(coarse_t.dtype)
        seq_t = coarse_t.size(1)
        if seq_t > 1:
            coarse_t = coarse_t + mix_w[0] * self.coarse_causal_chol(F.pad(coarse_t[:, :-1, :], (0, 0, 1, 0)))
            details_t[1] = details_t[1] + mix_w[2] * self.mid_causal_chol(F.pad(details_t[1][:, :-1, :], (0, 0, 1, 0)))
        if seq_t > 2:
            coarse_t = coarse_t + mix_w[1] * self.coarse_dilated_chol(F.pad(coarse_t[:, :-2, :], (0, 0, 2, 0)))
        d0_c, d0_f = details_t[0].chunk(2, dim=-1)
        th_k = self.kron_beamsplitter_theta.to(d0_c.dtype)
        c_th, s_th = torch.cos(th_k), torch.sin(th_k)
        details_t[0] = torch.cat([c_th * d0_c + s_th * d0_f, -s_th * d0_c + c_th * d0_f], dim=-1)
        wav_recon = self.wavelet.upsample_amplitudes(coarse_t, details_t)
        rec_c, rec_f = wav_recon[..., : self.half_d], wav_recon[..., self.half_d :]
        ent_c = rec_c * (1.0 + 0.28 * torch.tanh(self.kron_modulation_chol(rec_f)))
        ent_f = rec_f * (1.0 + 0.28 * torch.tanh(self.kron_reverse_chol(rec_c)))
        return l2_normalize(torch.cat([ent_c, ent_f], dim=-1), dim=-1).to(psi_a.dtype)

    def forward(
        self,
        psi: torch.Tensor,
        v_0: torch.Tensor | None = None,
        return_v: bool = False,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run pre-attention 1D conv -> Purified Born attention with V^(0) -> post-attention Wavelet-Kronecker + Interferometer."""
        scale_in = WaveletScaleState.from_interleaved(psi).normalize_levels()
        psi_conv = self.conv_bank.forward_levels(scale_in).to_interleaved_product_of_spheres(normalize=False).to(psi.dtype)
        g_conv = torch.tanh(self.pre_attn_conv_gate).to(dtype=psi.dtype)
        psi_pre = l2_normalize(psi + g_conv * psi_conv, dim=-1)

        attn_res = self.attn(
            psi_pre,
            cos=cos,
            sin=sin,
            v_0=v_0,
            window_size=self.window_size,
            doc_ids=doc_ids,
            return_v=True,
        )
        psi_a, v_curr = attn_res[0], attn_res[1]
        g_attn = torch.tanh(self.attn_gate).to(dtype=psi.dtype)
        psi_mid = l2_normalize(psi_pre + g_attn * psi_a, dim=-1)

        wav_recon = self._wavelet_kronecker_pathway(psi_mid)
        inter_out = self.interf(psi_mid).to(dtype=psi.dtype)
        w_int = PRE_ATTN_BLOCK_INTERFEROMETER_MIX
        psi_inter = l2_normalize(w_int * inter_out + (1.0 - w_int) * wav_recon, dim=-1)
        g_f = torch.tanh(self.fused_gate).to(dtype=psi.dtype)
        psi_out = l2_normalize(psi_mid + g_f * psi_inter, dim=-1)
        if return_v:
            return psi_out, wav_recon, v_curr
        return psi_out, wav_recon


@dataclass
class LevelSeparatedWaveletConvConfig(ParsevalMuPSubspaceCoincidenceConfig):
    """Configuration for Level-Separated Wavelet + Per-Layer Pre-Attn 1D Conv GPT (`Waves 32–34`)."""

    temporal_strides: tuple[int, int, int, int] = DEFAULT_LEVEL_TEMPORAL_STRIDES
    level_ortho_residual_weight: float = LEVEL_ORTHO_RESIDUAL_WEIGHT
    level_ortho_residual_min_weight: float = LEVEL_ORTHO_RESIDUAL_MIN_WEIGHT
    use_coincidence_warmup: bool = True
    subspace_balance_blend_max: float = DEFAULT_WARMED_BASE_SUBSPACE_BLEND_MAX
    use_per_layer_pre_attn_conv: bool = True
    checkpoint_detector_synthesis: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.embed_dim >= MIN_AUTO_HEAD_DIM_WIDTH and (self.embed_dim % DEFAULT_FLASH_BORN_HEAD_DIM == 0):
            if self.num_heads <= 0 or (self.embed_dim // self.num_heads) != DEFAULT_FLASH_BORN_HEAD_DIM:
                self.num_heads = max(2, self.embed_dim // DEFAULT_FLASH_BORN_HEAD_DIM)
        elif self.num_heads <= 0:
            self.num_heads = max(2, self.embed_dim // 16)

    @property
    def head_dim(self) -> int:
        """Return per-head attention dimension `embed_dim // num_heads`."""
        return int(self.embed_dim) // int(self.num_heads)


class LevelSeparatedWaveletConvGPT(ParsevalMuPSubspaceCoincidenceGPT):
    """Flag-Free Level-Separated Wavelet + Per-Layer Pre-Attention 1D Wavefunction Conv GPT (`Waves 32–34`)."""

    cfg: LevelSeparatedWaveletConvConfig

    def __init__(self, cfg: LevelSeparatedWaveletConvConfig) -> None:
        super().__init__(cfg)
        self.conv_bank = MultiRateWavefunctionConvBank(
            embed_dim=cfg.embed_dim,
            temporal_strides=cfg.temporal_strides,
            fast_spd_bmm=True,
        )
        self.pre_attn_conv_gate = torch.nn.Parameter(
            torch.full((cfg.embed_dim,), INIT_PRE_ATTN_CONV_GATE)
        )
        if bool(cfg.use_per_layer_pre_attn_conv):
            self.blocks = nn.ModuleList(
                [
                    PreAttnMultiRateWaveletConvBlock(self.pyramid.cfg, i, cfg.temporal_strides)
                    for i in range(cfg.num_layers)
                ]
            )
        with torch.no_grad():
            self.psi0_readout_gate.fill_(0.02)
            self.ngram_proj.alpha_ngram.fill_(0.04)
            self.ngram_proj.input_alpha_ngram.fill_(0.03)
            rms = self.pyramid.init_wav_rms
            for idx_s, name in enumerate(("wav_256", "wav_2048", "wav_8192", "wav_65536")):
                param = getattr(self.pyramid, name)
                if float(param.abs().max().item()) == 0.0:
                    torch.nn.init.normal_(param, mean=0.0, std=float(rms[idx_s].item()))

    @property
    def head_dim(self) -> int:
        """Return per-head attention dimension `embed_dim // num_heads`."""
        return int(self.cfg.embed_dim) // int(self.cfg.num_heads)

    def current_level_ortho_weight(self) -> float:
        """Return progress-ramped level-separated detector residual weight `w_sep(tau)` (`Wave 33`)."""
        w_max = float(self.cfg.level_ortho_residual_weight)
        if not bool(self.cfg.use_coincidence_warmup):
            return w_max
        w_min = min(w_max, float(self.cfg.level_ortho_residual_min_weight))
        return w_min + (w_max - w_min) * self.current_coincidence_ramp()

    def build_coincidence_spec(self) -> SubspaceCoincidenceSpec:
        """Build `SubspaceCoincidenceSpec` preserving `coincidence_gamma` under `readout_chol` absorption."""
        s = self._beta_absorbed_scale
        return SubspaceCoincidenceSpec(
            num_subspaces=self.num_subspaces,
            coincidence_gamma=self.current_coincidence_gamma(),
            subspace_var_penalty=self.current_subspace_var_penalty() / s,
            coarse_coincidence_weight=self.current_coarse_coincidence_weight(),
            partition_mode=str(self.cfg.partition_mode),
            full_vocab_coincidence_eval=bool(self.cfg.full_vocab_coincidence_eval),
            subspace_balance_blend=self.current_subspace_balance_blend(),
            use_fine_m4_chunked_coincidence=True,
            pre_normalized_inputs=True,
        )

    def embed_level_separated_tokens(self, idx: torch.Tensor) -> WaveletScaleState:
        """Return the 4 separated wavelet level tensors for token indices `idx`."""
        p = self.pyramid.perm_v_to_leaf.index_select(0, idx.reshape(-1)).view_as(idx)
        return synthesize_level_separated_wavelet_state(self.pyramid, p)

    def synthesize_level_separated_detectors(self) -> WaveletScaleState:
        """Synthesize the 4 separated sharp detector levels, each orthogonal to its input level."""
        pyr = self.pyramid
        p = pyr.perm_v_to_leaf[: self.cfg.vocab_size]
        self_state = synthesize_level_separated_wavelet_state(pyr, p)
        level_dim = self.cfg.embed_dim // NUM_WAVELET_PYRAMID_LEVELS

        base_256 = _fold_to_level_dim(pyr.chol_256(pyr.wav_256), level_dim)
        trans_delta = torch.tanh(pyr.transition_gain) * (pyr.coarse_transition @ base_256).index_select(0, p >> 8)
        pred_delta = (
            torch.tanh(pyr.pred_mix_gain)
            * _fold_to_level_dim(pyr.chol_pred_2048(pyr.pred_wav_2048), level_dim).index_select(0, p >> 5)
            if hasattr(pyr, "pred_wav_2048")
            else None
        )

        det_levels: list[torch.Tensor] = []
        for idx_m, psi_m in enumerate(self_state.as_tuple()):
            swapped = pyr.apply_symplectic_swap(psi_m)
            if idx_m == 0:
                swapped = swapped + trans_delta
            elif idx_m == 1 and pred_delta is not None:
                swapped = swapped + pred_delta
            ov = (swapped * psi_m).sum(dim=-1, keepdim=True) / (psi_m.square().sum(dim=-1, keepdim=True) + 1e-8)
            orth_m = swapped - ov * psi_m
            det_levels.append(orth_m)

        return WaveletScaleState(
            coarse_256=det_levels[0],
            mid_2048=det_levels[1],
            sub_8192=det_levels[2],
            fine_65536=det_levels[3],
        ).normalize_levels()

    def _raw_combined_detectors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Synthesize pre-Cholesky Product-of-Spheres balanced sharp & diffuse detectors `(m_sharp_bal, m_diff, w_k)`."""
        pyr = self.pyramid
        w_raw = F.softmax(pyr.kraus_raw_weights.float(), dim=0)
        w_k = apply_soft_powerlaw_spectral_floor(w_raw, min_eff_rank=self.cfg.kraus_spectral_floor)
        _, m_sharp_raw = pyr._synthesize_raw_detectors()
        b_sub = self.current_subspace_balance_blend()
        pmode = str(self.cfg.partition_mode)
        m_strided = blend_subspace_l2_normalize(
            m_sharp_raw, num_subspaces=self.num_subspaces, blend=b_sub, partition_mode=pmode
        )
        m_sep = self.synthesize_level_separated_detectors().to_interleaved_product_of_spheres(normalize=False)
        w_sep = self.current_level_ortho_weight()
        m_comb = l2_normalize((1.0 - w_sep) * m_strided + w_sep * m_sep.to(m_strided.dtype), dim=-1)
        m_sharp_bal = blend_subspace_l2_normalize(
            m_comb,
            num_subspaces=self.num_subspaces,
            blend=b_sub,
            partition_mode=pmode,
        )
        op_mean = torch.einsum("r,rij->ij", torch.sqrt(w_k + 1e-8), pyr.kraus_generators.float())
        if m_sharp_bal.is_cuda and torch.is_autocast_enabled():
            m_diff = l2_normalize(F.linear(m_sharp_bal.bfloat16(), op_mean.bfloat16()), dim=-1).to(m_sharp_bal.dtype)
        else:
            m_diff = l2_normalize(F.linear(m_sharp_bal.float(), op_mean), dim=-1).to(m_sharp_bal.dtype)
        return m_sharp_bal, m_diff, w_k

    def synthesize_detectors(self, use_checkpoint: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Synthesize sharp & diffuse vocabulary detectors (optionally gradient-checkpointed)."""
        if use_checkpoint and torch.is_grad_enabled() and not torch.compiler.is_compiling():
            res = checkpoint(self._raw_combined_detectors, use_reentrant=False)
            assert isinstance(res, tuple)
            return res
        return self._raw_combined_detectors()

    def _synthesize_combined_detectors(
        self, use_checkpoint: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Synthesize detectors with gradient checkpointing during training and apply `readout_chol` post-norm."""
        do_ckpt = (
            bool(
                self.cfg.checkpoint_detector_synthesis
                and self.training
                and torch.is_grad_enabled()
                and self.pyramid.wav_65536.requires_grad
            )
            if use_checkpoint is None
            else bool(use_checkpoint)
        )
        m_sharp_unit, m_diff, w_k = self.synthesize_detectors(use_checkpoint=do_ckpt)
        use_chol = bool(self._beta_absorbed_scale != 1.0 or not m_sharp_unit.is_cuda)
        m_sharp = self.pyramid.readout_chol(m_sharp_unit) if use_chol else m_sharp_unit
        return m_sharp, m_diff, w_k

    @torch.no_grad()
    def inspect_level_energy_and_rank(self) -> dict[str, float]:
        """Return per-level energy fractions (`0.25` each) and per-level participation ratios `R_sharp^{(m)}`."""
        det_state = self.synthesize_level_separated_detectors()
        energies = det_state.level_energy_fractions()
        v_eff = min(self.cfg.valid_vocab_size, det_state.coarse_256.size(0))
        out: dict[str, float] = {}
        for name, tensor_m in zip(("coarse_256", "mid_2048", "sub_8192", "fine_65536"), det_state.as_tuple()):
            out[f"energy_{name}"] = energies[name]
            m_cent = tensor_m[:v_eff].detach().float()
            m_cent = m_cent - m_cent.mean(dim=0, keepdim=True)
            gram = m_cent.t() @ m_cent
            tr = torch.diagonal(gram).sum()
            out[f"r_sharp_{name}"] = float(((tr * tr) / (gram * gram).sum().clamp_min(1e-12)).item())
        return out

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_logits: bool | None = None,
        precomputed_detectors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Execute level-separated wavelet synthesis -> per-layer 1D conv + Born attention -> readout."""
        m_sub = self.num_subspaces
        pmode = str(self.cfg.partition_mode)
        b_sub = self.current_subspace_balance_blend()
        psi_in_base = self.pyramid.embed_tokens(idx)
        sep_state = self.embed_level_separated_tokens(idx)
        conv_state = self.conv_bank.forward_levels(sep_state)
        psi_conv = conv_state.to_interleaved_product_of_spheres(normalize=False).to(psi_in_base.dtype)

        bigram_psi = self.pyramid.embed_dyadic_bigram_phase(idx).to(psi_in_base.dtype)
        in_ngram, readout_ngram = self.ngram_proj(idx, target_dtype=psi_in_base.dtype)
        c_gate = torch.tanh(self.pre_attn_conv_gate).to(psi_in_base.dtype)
        psi_in_0 = l2_normalize(psi_in_base + c_gate * psi_conv, dim=-1)

        psi_in = l2_normalize(
            psi_in_0 + torch.tanh(self.bigram_embed_gate).to(psi_in_0.dtype) * bigram_psi + in_ngram,
            dim=-1,
        )
        psi = self.smear_gate(psi_in)
        doc_ids = compute_document_ids(idx)
        cos, sin = [t.to(psi.device) for t in self.rope.get_cos_sin(idx.size(1))]
        v0_src = l2_normalize(psi_in + torch.tanh(self.v0_gate).to(psi_in.dtype) * psi_in_0, dim=-1)
        v_0 = self.pyramid.synthesize_value_anchor(v0_src, bigram_psi)
        psi_diff = psi
        n_half = self.cfg.num_layers // 2
        enc_skips: list[torch.Tensor] = []
        ckpt_blk = bool(
            self.cfg.embed_dim >= CKPT_BLOCK_MIN_EMBED_DIM
            and idx.size(0) >= CKPT_BLOCK_MIN_BATCH_SIZE
            and self.training
            and torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
        )
        for l_idx, block in enumerate(self.blocks):
            if l_idx == n_half:
                psi = l2_normalize(
                    psi + torch.tanh(self.bigram_mid_gate).to(psi.dtype) * bigram_psi,
                    dim=-1,
                )
            if isinstance(block, PreAttnMultiRateWaveletConvBlock):
                if ckpt_blk:
                    psi, blk_diff, v_0 = checkpoint(
                        block, psi, v_0=v_0, return_v=True, cos=cos, sin=sin, doc_ids=doc_ids, use_reentrant=False
                    )
                else:
                    psi, blk_diff, v_0 = block(psi, v_0=v_0, return_v=True, cos=cos, sin=sin, doc_ids=doc_ids)
            else:
                psi, blk_diff = block(psi, cos=cos, sin=sin, v_0=v_0, doc_ids=doc_ids)
            g_d = torch.sigmoid(self.diff_accum_gate).to(psi_diff.dtype)
            psi_diff = l2_normalize((1.0 - g_d) * psi_diff + g_d * blk_diff, dim=-1)
            if l_idx < n_half:
                enc_skips.append(psi)
            elif enc_skips:
                skip_g = torch.tanh(self.unet_skip_gates[min(l_idx - n_half, self.unet_skip_gates.size(0) - 1)])
                psi = l2_normalize(psi + skip_g.to(psi.dtype) * enc_skips.pop(), dim=-1)

        psi0_z = torch.stack((psi_in_0[..., 0::2], -psi_in_0[..., 1::2]), dim=-1).reshape_as(psi_in_0)
        psi_pre = blend_subspace_l2_normalize(
            psi + torch.tanh(self.psi0_readout_gate).to(psi.dtype) * psi0_z + readout_ngram,
            num_subspaces=m_sub,
            blend=b_sub,
            partition_mode=pmode,
        )
        m_sharp, m_diff, w_k = (
            precomputed_detectors if precomputed_detectors is not None else self._synthesize_combined_detectors()
        )
        use_chol = bool(self._beta_absorbed_scale != 1.0 or not idx.is_cuda)
        psi_s_out = self.pyramid.readout_chol(psi_pre) if use_chol else psi_pre
        self._last_diffuse_eff_rank = float(1.0 / ((w_k * w_k).sum().item() + 1e-12))
        self._last_mean_beta = self.current_beta()
        self._last_lambda_diff = self.current_lambda_diff()

        r_spec = self.build_readout_spec()
        c_spec = self.build_coincidence_spec()
        want_logits = (targets is None) if return_logits is None else bool(return_logits)
        prior = self.povm_log_prior
        loss = (
            subspace_coincidence_dyadic_chunked_ce(
                psi_s_out, psi_diff, m_sharp, m_diff, targets, r_spec, c_spec, prior
            )
            if targets is not None
            else None
        )
        logits = (
            eval_subspace_coincidence_full_logits(
                psi_s_out, psi_diff, m_sharp, m_diff, r_spec, c_spec, prior
            )
            if want_logits
            else (loss if (targets is not None and return_logits is None) else None)
        )
        if isinstance(logits, torch.Tensor) and logits.ndim >= 2 and getattr(self.ngram_proj, "_kn_state", None) is not None:
            logits = self.ngram_proj.apply_surface_logit_lift(idx, logits, self._training_progress, self.cfg.embed_dim)
        return logits, loss

    @torch.no_grad()
    def retract_kraus_generators(self, smooth_landing_rate: float = 0.08) -> float:
        """Retract `kraus_generators`, `u_ngram`, and apply Riemannian Frame-Potential Landing to wavelet pyramid."""
        err = super().retract_kraus_generators(smooth_landing_rate=smooth_landing_rate)
        self.pyramid._stabilize_pyramid_frame(rate=0.04)
        level_dim = self.cfg.embed_dim // NUM_WAVELET_PYRAMID_LEVELS
        for up_p in (self.pyramid.up_8192, self.pyramid.up_65536):
            u_f = up_p.data.float()
            r_dim, c_dim = u_f.shape
            gram = u_f @ u_f.t() - torch.eye(r_dim, device=u_f.device, dtype=torch.float32)
            denom = 1.0 + (torch.linalg.norm(gram) / math.sqrt(float(r_dim)))
            u_f = u_f - (UP_FRAME_LANDING_SCALE * smooth_landing_rate) * ((gram / denom) @ u_f)
            if c_dim > level_dim and (c_dim % level_dim == 0) and (level_dim % SYMPLECTIC_PAIR_DIM == 0):
                ratio = c_dim // level_dim
                u_pairs = u_f.reshape(r_dim, level_dim // SYMPLECTIC_PAIR_DIM, ratio, SYMPLECTIC_PAIR_DIM)
                u_fold = u_pairs.mean(dim=-2).reshape(r_dim, level_dim)
                f_sq = (u_fold * u_fold).sum().clamp_min(1e-12)
                g_col = (float(level_dim) / f_sq) * (u_fold.t() @ u_fold) - torch.eye(level_dim, device=u_f.device)
                d_col = 1.0 + (torch.linalg.norm(g_col) / math.sqrt(float(level_dim)))
                delta = -(UP_FRAME_LANDING_SCALE * smooth_landing_rate) * (u_fold @ (g_col / d_col))
                u_pairs = u_pairs + delta.reshape(r_dim, level_dim // SYMPLECTIC_PAIR_DIM, 1, SYMPLECTIC_PAIR_DIM)
                u_f = u_pairs.reshape(r_dim, c_dim)
            up_p.data.copy_(u_f.to(up_p.dtype))
        return err



@torch.no_grad()
def initialize_level_separated_wavelet_conv_with_ppmi_svd(
    model: LevelSeparatedWaveletConvGPT,
    ppmi_embeddings: torch.Tensor,
    init_tokens: torch.Tensor,
    prior_tokens: torch.Tensor,
) -> dict[str, float]:
    """Initialize `LevelSeparatedWaveletConvGPT` and record per-level energy and `R_sharp`."""
    stats = initialize_subspace_coincidence_with_ppmi_svd(model, ppmi_embeddings, init_tokens, prior_tokens)
    pyr = model.pyramid
    for idx_s, w_param in enumerate((pyr.wav_256, pyr.wav_2048, pyr.wav_8192, pyr.wav_65536)):
        w_f = w_param.data.float()
        w_f = w_f - w_f.mean(dim=0, keepdim=True)
        pyr.init_wav_rms[idx_s] = torch.sqrt((w_f * w_f).sum() / float(w_f.size(0)) + 1e-12)
    pyr._stabilize_pyramid_frame(rate=0.18)
    m_sharp, _, _ = model.synthesize_detectors(use_checkpoint=False)
    stats["sharp_eff_rank"] = model.compute_sharp_eff_rank(m_sharp)
    stats.update(model.inspect_level_energy_and_rank())
    return stats


def build_level_separated_wavelet_conv_optimizers(
    model: LevelSeparatedWaveletConvGPT,
    base_muon_lr: float = DEFAULT_SUBSPACE_MUON_LR,
    base_wav_lr: float = DEFAULT_SUBSPACE_WAV_LR,
    base_head_lr: float = DEFAULT_SUBSPACE_HEAD_LR,
) -> tuple[StiefelMuon, torch.optim.AdamW]:
    """Build width-invariant `muP` (`sqrt(256 / d)`) `StiefelMuon` + `AdamW` optimizers."""
    return build_subspace_coincidence_optimizers(
        model, base_muon_lr=base_muon_lr, base_wav_lr=base_wav_lr, base_head_lr=base_head_lr
    )


__all__ = [
    "DEFAULT_FLASH_BORN_HEAD_DIM",
    "DEFAULT_WARMED_BASE_SUBSPACE_BLEND_MAX",
    "INIT_COARSE_CAUSAL_GATE",
    "INIT_PRE_ATTN_CONV_GATE",
    "LEVEL_ORTHO_RESIDUAL_MIN_WEIGHT",
    "LEVEL_ORTHO_RESIDUAL_WEIGHT",
    "PRE_ATTN_BLOCK_ATTN_INIT_SCALE",
    "PRE_ATTN_BLOCK_CONV_INIT_SCALE",
    "PRE_ATTN_BLOCK_INTERFEROMETER_MIX",
    "LevelSeparatedWaveletConvConfig",
    "LevelSeparatedWaveletConvGPT",
    "PreAttnMultiRateWaveletConvBlock",
    "build_level_separated_wavelet_conv_optimizers",
    "initialize_level_separated_wavelet_conv_with_ppmi_svd",
    "synthesize_level_separated_wavelet_state",
    "update_parseval_mup_ngram_learning_rates",
]

