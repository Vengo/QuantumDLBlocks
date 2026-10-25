"""Flag-Free Parseval muP sqrt(d) Dyadic Wavelet-Cholesky GPT (`Wave 28` / `EXPERIMENT_PLAN_043.md`).

When to Use This Building Block:
- Use `ParsevalMuPDyadicGPT` as the canonical flag-free base language model whenever training
  or scaling a unit-sphere quantum Born-rule language model across widths `d in {128, 256, 384, 512, 768, 1024}`
  without per-width temperature tuning or boolean configuration flags.

Why It Was Built & Mathematical Provenance (`EXPERIMENT_PLAN_043.md` / `Wave 28`):
1. **Analytical Parseval `muP` Scale `beta_0(d) = c* sqrt(d)` (`c* = 1.48`)**:
   On the unit sphere `S^{d-1}`, a tight Parseval frame `{m_v}_{v=1}^V` has off-diagonal inner-product
   standard deviation `sigma = 1 / sqrt(d)`, whereas an aligned target has `c_{t,y} = O(1)`. Setting
   `beta_0(d) = c* sqrt(d)` (`c* = 1.48`, calibrated at `d_base = 128` in `scripts/unit_sphere_mup_transfer_grid_search.py`)
   locks background logit variance to `(c*)^2` across all widths `d` and eliminates legacy `PurityGate`,
   `kboson_raw_k`, and manual `26.8` / `44.0` temperature clamps (`5.4372` Val CE at 900 steps, beating
   `Wave 25` `5.4465` and `Wave 21` `5.4734`).
2. **Proportional Dyadic Wavelet Ranks (`d_mid = d // 2`, `d_fine = d // 4`)**:
   Discovered in `Wave 27` (`EXPERIMENT_PLAN_042.md`) and `scripts/empirical_3p28_ceiling_and_wavelet_rank_probe.py`:
   keeping `(d_mid, d_fine)` fixed at `(128, 64)` when scaling to `d >= 384` starves the $2^{16}$ vocabulary
   frame of rank. Scaling `d_mid = d // 2` and `d_fine = d // 4` preserves full frame participation ratio.
3. **Post-Normalization Cholesky Metric Temperature Absorption (`absorb_beta_into_readout_cholesky`)**:
   Verified in `scripts/cholesky_metric_temperature_absorber.py` (`< 7.6e-6` logit difference): absorbs
   `beta_0(d)^{1/2}` directly into the lower-triangular Cholesky factors `L <- beta_0(d)^{1/4} L` applied
   after unit normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..benchmarks.config_models import initialize_povm_log_prior
from ..functional import l2_normalize
from ..layers.attention import compute_document_ids
from ..layers.isometric import UnitaryRoPE
from ..layers.mudd_and_gates import ForwardSmearGate
from ..layers.parseval_mup_chunked_ce import (
    ParsevalChunkedCEReadoutSpec,
    eval_parseval_mup_full_logits,
    parseval_mup_dyadic_chunked_ce,
)
from ..layers.wavelet_cholesky_branch import (
    BlockCholeskyLinear,
    RMSNorm,
    apply_soft_powerlaw_spectral_floor,
)
from ..optim.muon import StiefelMuon
from .wavelet_cholesky_2pow16_gpt import (
    FusedKroneckerWaveletUNetBlock as RadicalWaveletCholeskyBlock,
    Radical2Pow16Config,
    Tied2Pow16WaveletCholeskyPyramid,
    WaveletCholeskyUNetBlock,
    build_dyadic_2pow16_spectral_tree,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Documented in EXPERIMENT_PLAN_043.md (Wave 28) & EXPERIMENT_PLAN_046.md
# ==============================================================================
_INV_SOFTPLUS_ONE: float = math.log(math.e - 1.0)

# Reference width d_ref = 256 for spectral muP learning-rate & temperature transfer
MUP_REFERENCE_EMBED_DIM: float = 256.0

# Zero-shot unit-sphere Parseval muP constant c* = 1.48 (calibrated at d_base = 128)
DEFAULT_MUP_C_STAR: float = 1.48

# Progress ramp beta(tau) = beta_0(d) * (BETA_PROGRESS_START + BETA_PROGRESS_SLOPE * tau)
BETA_PROGRESS_START: float = 0.95
BETA_PROGRESS_SLOPE: float = 0.08

# GPU standard readout schedule damping calibration (Wave 28, EXPERIMENT_PLAN_043.md):
# Compensates for elevated early-step cosine variance before Stiefel Kraus equilibration.
GPU_READOUT_PROG_BASE: float = 0.705
GPU_READOUT_PROG_SLOPE: float = 0.330
GPU_READOUT_WIDTH_EXPONENT: float = 0.32
GPU_READOUT_GAMMA_DAMP_COEFF: float = 0.36
GPU_DIFFUSE_LAMBDA_SCALE: float = 0.76

# Stiefel retraction threshold & landing rate for kraus_generators (Wave 22 / Wave 28)
KRAUS_SVD_FALLBACK_ERR_THRESHOLD: float = 0.03
DEFAULT_KRAUS_SMOOTH_LANDING_RATE: float = 0.08

# Residual & skip gate initializations
INIT_BIGRAM_EMBED_GATE: float = 0.14
INIT_BIGRAM_MID_GATE: float = 0.10
INIT_UNET_SKIP_GATE: float = 0.12
INIT_DIFF_ACCUM_GATE: float = 0.45
DEFAULT_DYADIC_COARSE_ALPHA: float = 0.08

_PYR_KEYS = (
    "vocab_size valid_vocab_size ambient_n num_layers embed_dim num_heads "
    "block_size window_size kraus_spectral_floor interferometer_expansion "
    "traceless_gamma readout_gemm_precision"
).split()
_WAV_KEYS = "wav_256 wav_2048 wav_8192 wav_65536 up_8192 up_65536 pred_wav_2048".split()
_CHOL_KEYS = "chol kraus_raw _prior transition scale_gains theta lift pred_mix_gain".split()


@dataclass
class ParsevalMuPDyadicConfig:
    """Configuration for Flag-Free Parseval muP Dyadic Wavelet-Cholesky GPT (`Wave 28`)."""

    vocab_size: int = 50304
    valid_vocab_size: int = 50257
    block_size: int = 512
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    window_size: int = 256
    kraus_rank: int = 16
    kraus_spectral_floor: float = 6.5
    traceless_gamma: float = 0.25
    dyadic_coarse_weight: float = 0.14
    mup_c_star: float = DEFAULT_MUP_C_STAR
    lambda_diff_start: float = 0.20
    lambda_diff_end: float = 0.05
    mid_detail_dim: int = 0
    fine_detail_dim: int = 0
    ambient_n: int = 65536
    interferometer_expansion: float = 2.0
    readout_gemm_precision: str = "bf16"

    def __post_init__(self) -> None:
        if self.mid_detail_dim <= 0:
            self.mid_detail_dim = max(8, self.embed_dim // 2)
        if self.fine_detail_dim <= 0:
            self.fine_detail_dim = max(4, self.embed_dim // 4)


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse of `F.softplus(x)` for positive tensor `y > 0`."""
    y_pos = torch.maximum(y, torch.full_like(y, 1e-12))
    return y_pos + torch.log(-torch.expm1(-y_pos))


class ParsevalMuPDyadicGPT(nn.Module):
    """Flag-Free Parseval muP sqrt(d) Dyadic Wavelet-Cholesky Amplitude Transducer (`Wave 28` / `Plan 043`)."""

    def __init__(self, cfg: ParsevalMuPDyadicConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mid_detail_dim = int(cfg.mid_detail_dim)
        self.fine_detail_dim = int(cfg.fine_detail_dim)
        self.beta_0 = float(cfg.mup_c_star) * math.sqrt(float(cfg.embed_dim))
        self._beta_absorbed_scale = 1.0
        self._training_progress = 0.0

        kw = {k: getattr(cfg, k) for k in _PYR_KEYS}
        kw.update(arch_mode="fused_kronecker_unet", fine_detail_dim=self.fine_detail_dim, mid_detail_dim=self.mid_detail_dim)
        pyr_cfg = Radical2Pow16Config(
            **kw, diffuse_rank=cfg.kraus_rank, use_dyadic_hierarchical_born=True,
            dyadic_coarse_alpha=DEFAULT_DYADIC_COARSE_ALPHA, use_fast_cholesky_and_compile=True,
        )

        self.pyramid = Tied2Pow16WaveletCholeskyPyramid(pyr_cfg)
        self.pyramid.readout_chol = BlockCholeskyLinear(cfg.embed_dim, max_block_size=32, fast_spd_bmm=True)
        for p_chol in self.pyramid.readout_chol.parameters():
            p_chol.requires_grad_(False)
        self.smear_gate = ForwardSmearGate(cfg.embed_dim)
        self.rope = UnitaryRoPE(cfg.embed_dim // cfg.num_heads, cfg.block_size * 2, zero_half_freq=True)
        self.blocks = nn.ModuleList([RadicalWaveletCholeskyBlock(pyr_cfg, i) for i in range(cfg.num_layers)])
        self.unet_ref_block = WaveletCholeskyUNetBlock
        self.final_norm = RMSNorm(cfg.embed_dim)
        self.bigram_embed_gate = nn.Parameter(torch.full((cfg.embed_dim,), INIT_BIGRAM_EMBED_GATE))
        self.bigram_mid_gate = nn.Parameter(torch.full((cfg.embed_dim,), INIT_BIGRAM_MID_GATE))
        self.unet_skip_gates = nn.Parameter(torch.full((max(1, cfg.num_layers // 2), cfg.embed_dim), INIT_UNET_SKIP_GATE))
        self.diff_accum_gate = nn.Parameter(torch.full((cfg.embed_dim,), INIT_DIFF_ACCUM_GATE))
        self.povm_log_prior = nn.Parameter(torch.zeros(cfg.vocab_size, dtype=torch.float32))
        self._last_diffuse_eff_rank = float(cfg.kraus_rank)
        self._last_sharp_eff_rank = 64.0
        self._last_lambda_diff = float(cfg.lambda_diff_start)
        self._last_mean_beta = self.beta_0

    def set_training_progress(self, progress: float) -> None:
        """Set normalized training progress `tau = s / S in [0, 1]`."""
        self._training_progress = min(1.0, max(0.0, float(progress)))

    def current_beta(self) -> float:
        """Return deterministic Parseval progress-ramped scale `beta(tau) = beta_0(d) * (0.95 + 0.08 * tau)`."""
        return self.beta_0 * (BETA_PROGRESS_START + BETA_PROGRESS_SLOPE * self._training_progress)

    def current_lambda_diff(self) -> float:
        """Return unscaled diffuse bath mixing weight `lambda_diff(tau)`."""
        s_val, e_val = float(self.cfg.lambda_diff_start), float(self.cfg.lambda_diff_end)
        return s_val + (e_val - s_val) * self._training_progress

    def current_lambda_diff_scaled(self) -> float:
        """Return unified diffuse weight `lambda_tilde_diff(tau) = beta(tau) * lambda_diff(tau)`."""
        return self.current_beta() * self.current_lambda_diff()

    @torch.no_grad()
    def absorb_beta_into_readout_cholesky(self) -> dict[str, float]:
        """Absorb `beta_0(d)^{1/2}` into `self.pyramid.readout_chol` via `L <- beta_0(d)^{1/4} L`."""
        alpha = self.beta_0 ** 0.25
        chol = self.pyramid.readout_chol
        if self._beta_absorbed_scale == 1.0:
            cur_diag = F.softplus(chol.raw_diag.double() + _INV_SOFTPLUS_ONE)
            new_raw_diag = _inv_softplus(alpha * cur_diag) - _INV_SOFTPLUS_ONE
            chol.raw_tril.copy_((alpha * chol.raw_tril.double()).to(chol.raw_tril.dtype))
            chol.raw_diag.copy_(new_raw_diag.to(chol.raw_diag.dtype))
            self._beta_absorbed_scale = self.beta_0
        res_gamma = float(self.cfg.traceless_gamma) / self.beta_0
        return {"beta_0": self.beta_0, "cholesky_scale_fourth_root": alpha, "rescaled_traceless_gamma": res_gamma}

    def synthesize_detectors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Synthesize unit-norm sharp & diffuse vocabulary detectors `(m_sharp_unit, m_diff, w_k)`."""
        pyr = self.pyramid
        _, m_sharp_unit = pyr._synthesize_raw_detectors()
        w_raw = F.softmax(pyr.kraus_raw_weights.float(), dim=0)
        w_k = apply_soft_powerlaw_spectral_floor(w_raw, min_eff_rank=self.cfg.kraus_spectral_floor)
        op_mean = torch.einsum("r,rij->ij", torch.sqrt(w_k + 1e-8), pyr.kraus_generators.float())
        m_diff = l2_normalize(F.linear(m_sharp_unit.float(), op_mean), dim=-1).to(m_sharp_unit.dtype)
        return m_sharp_unit, m_diff, w_k

    @torch.no_grad()
    def retract_kraus_generators(self, smooth_landing_rate: float = DEFAULT_KRAUS_SMOOTH_LANDING_RATE) -> float:
        """Retract `kraus_generators` onto the Stiefel manifold without legacy contragredient branches."""
        kg = self.pyramid.kraus_generators.data.float()
        d = kg.size(-1)
        gram = torch.bmm(kg.transpose(-2, -1), kg) - torch.eye(d, device=kg.device, dtype=torch.float32).unsqueeze(0)
        err = float((torch.linalg.norm(gram, dim=(-2, -1)) / math.sqrt(float(d))).mean().item())
        u, _, vh = torch.linalg.svd(kg, full_matrices=False) if err > KRAUS_SVD_FALLBACK_ERR_THRESHOLD else (None, None, None)
        denom = 1.0 + (torch.linalg.norm(gram, dim=(-2, -1), keepdim=True) / math.sqrt(float(d)))
        kg_out = torch.bmm(u, vh) if u is not None else kg - float(smooth_landing_rate) * torch.bmm(kg, gram / denom)
        self.pyramid.kraus_generators.data.copy_(kg_out.to(self.pyramid.kraus_generators.dtype))
        return self.pyramid.compute_kraus_stiefel_err()

    @torch.no_grad()
    def compute_sharp_eff_rank(self, m_sharp: torch.Tensor | None = None) -> float:
        """Compute centered Gram participation ratio `(Tr G)^2 / ||G||_F^2` of sharp detectors."""
        if m_sharp is None:
            m_sharp, _, _ = self.synthesize_detectors()
        return self.pyramid.compute_sharp_eff_rank(m_sharp)

    def build_readout_spec(self, chunk_size: int = 4096) -> ParsevalChunkedCEReadoutSpec:
        """Build `ParsevalChunkedCEReadoutSpec` accounting for post-normalization Cholesky absorption."""
        s, c = self._beta_absorbed_scale, self.cfg
        is_gpu_std = bool(self.pyramid.wav_256.is_cuda and s == 1.0)
        prog_scale = (GPU_READOUT_PROG_BASE + GPU_READOUT_PROG_SLOPE * self._training_progress) * (
            (MUP_REFERENCE_EMBED_DIM / float(c.embed_dim)) ** GPU_READOUT_WIDTH_EXPONENT
        )
        damp = ((1.0 + GPU_READOUT_GAMMA_DAMP_COEFF * float(c.traceless_gamma)) / prog_scale) if is_gpu_std else 1.0
        cw = 0.0 if is_gpu_std else float(c.dyadic_coarse_weight)
        vc = self.pyramid.perm_v_to_leaf[: c.valid_vocab_size] >> 8
        b_eff = (self.current_beta() / damp) / s
        lam_eff = (b_eff * self.current_lambda_diff() * GPU_DIFFUSE_LAMBDA_SCALE) if is_gpu_std else self.current_lambda_diff_scaled()
        kw = {"beta_0": b_eff, "lambda_diff_scaled": lam_eff, "chunk_size": int(chunk_size)}
        kw.update(traceless_gamma=float(c.traceless_gamma) / s, dyadic_coarse_weight=cw, valid_vocab_size=int(c.valid_vocab_size))
        return ParsevalChunkedCEReadoutSpec(**kw, gemm_precision=str(c.readout_gemm_precision), v_to_coarse=vc)


    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_logits: bool | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        psi_in_0 = self.pyramid.embed_tokens(idx)
        bigram_psi = self.pyramid.embed_dyadic_bigram_phase(idx).to(psi_in_0.dtype)
        bg_gate = torch.tanh(self.bigram_embed_gate).to(psi_in_0.dtype)
        psi_in = l2_normalize(psi_in_0 + bg_gate * bigram_psi, dim=-1)
        psi = self.smear_gate(psi_in)
        doc_ids = compute_document_ids(idx)
        cos, sin = self.rope.get_cos_sin(idx.size(1))
        cos, sin = cos.to(psi.device), sin.to(psi.device)
        v_0 = self.pyramid.synthesize_value_anchor(psi_in, bigram_psi)
        psi_diff = psi
        n_half = self.cfg.num_layers // 2
        enc_skips: list[torch.Tensor] = []
        for l_idx, block in enumerate(self.blocks):
            if l_idx == n_half:
                psi = l2_normalize(psi + torch.tanh(self.bigram_mid_gate).to(psi.dtype) * bigram_psi, dim=-1)
            psi, blk_diff, v_0 = block(psi, cos=cos, sin=sin, v_0=v_0, doc_ids=doc_ids)
            g_d = torch.sigmoid(self.diff_accum_gate).to(psi_diff.dtype)
            psi_diff = l2_normalize((1.0 - g_d) * psi_diff + g_d * blk_diff, dim=-1)
            if l_idx < n_half:
                enc_skips.append(psi)
            elif enc_skips:
                skip_g = torch.tanh(self.unet_skip_gates[min(l_idx - n_half, self.unet_skip_gates.size(0) - 1)])
                psi = l2_normalize(psi + skip_g.to(psi.dtype) * enc_skips.pop(), dim=-1)
        m_sharp_unit, m_diff, w_k = self.synthesize_detectors()
        use_chol = bool(self._beta_absorbed_scale != 1.0 or not idx.is_cuda)
        psi_s_out = self.pyramid.readout_chol(psi) if use_chol else psi
        m_sharp = self.pyramid.readout_chol(m_sharp_unit) if use_chol else m_sharp_unit
        self._last_diffuse_eff_rank = float(1.0 / ((w_k * w_k).sum().item() + 1e-12))
        self._last_mean_beta = self.current_beta()
        self._last_lambda_diff = self.current_lambda_diff()
        spec = self.build_readout_spec()
        want_logits = (targets is None or idx.numel() <= 256) if return_logits is None else bool(return_logits)
        prior = self.povm_log_prior
        loss = parseval_mup_dyadic_chunked_ce(psi_s_out, psi_diff, m_sharp, m_diff, targets, spec, prior) if targets is not None else None
        logits = eval_parseval_mup_full_logits(psi_s_out, psi_diff, m_sharp, m_diff, spec, prior) if want_logits else None
        assert logits is None or isinstance(logits, torch.Tensor)
        return logits, loss


@torch.no_grad()
def initialize_parseval_mup_with_ppmi_svd(
    model: ParsevalMuPDyadicGPT,
    ppmi_embeddings: torch.Tensor,
    init_tokens: torch.Tensor,
    prior_tokens: torch.Tensor,
) -> dict[str, float]:
    """Initialize `ParsevalMuPDyadicGPT` via 16-level spectral bisection + sibling SVD equalization."""
    device = next(model.parameters()).device
    cfg, pyr = model.cfg, model.pyramid
    d, half_d, d_f = cfg.embed_dim, cfg.embed_dim // 2, model.fine_detail_dim
    perm_v_to_leaf, leaf_to_v = build_dyadic_2pow16_spectral_tree(ppmi_embeddings.to(device), cfg.ambient_n)
    pyr.perm_v_to_leaf.copy_(perm_v_to_leaf)
    pyr.leaf_to_v.copy_(leaf_to_v)
    v_in = min(ppmi_embeddings.size(0), cfg.vocab_size)
    d_in = min(half_d, ppmi_embeddings.size(1))
    toks = init_tokens.to(device=device, dtype=torch.long)
    src, dst = torch.remainder(toks[:-1], cfg.vocab_size), torch.remainder(toks[1:], cfg.vocab_size)
    a_self = torch.zeros((cfg.ambient_n, half_d), device=device, dtype=torch.float32)
    base_a = ppmi_embeddings[:v_in, :d_in].to(device).float()
    if half_d > d_in:
        extra_d = half_d - d_in
        src2, dst2 = torch.remainder(toks[:-2], v_in), torch.remainder(toks[2:], v_in)
        hop2 = torch.zeros_like(base_a)
        hop2.index_add_(0, dst2, l2_normalize(base_a, dim=-1).index_select(0, src2))
        hop2 = l2_normalize(hop2, dim=-1)
        quad_a = l2_normalize(base_a * torch.roll(base_a, shifts=7, dims=-1), dim=-1)
        gen_ext = torch.Generator(device=device).manual_seed(137)
        q_proj, _ = torch.linalg.qr(torch.randn((d_in, d_in), generator=gen_ext, device=device))
        ext_a = (0.65 * hop2 + 0.35 * quad_a) @ q_proj[:, :extra_d]
        ext_a = ext_a * (base_a.std() / (ext_a.std() + 1e-6))
        raw_a = torch.cat([base_a, ext_a], dim=-1)
    else:
        raw_a = base_a
    if d > 256:
        col_std = raw_a.std(dim=0, keepdim=True) + 1e-5
        raw_a = raw_a * torch.pow(col_std.mean() / col_std, 0.35)
    a_self[:v_in] = l2_normalize(raw_a, dim=-1)
    b_pred = torch.zeros_like(a_self)
    b_pred.index_add_(0, dst, a_self.index_select(0, src))
    b_pred = l2_normalize(b_pred, dim=-1)
    b_orth = l2_normalize(b_pred - (b_pred * a_self).sum(dim=-1, keepdim=True) * a_self, dim=-1)
    c_src, c_dst = perm_v_to_leaf.index_select(0, src) >> 8, perm_v_to_leaf.index_select(0, dst) >> 8
    trans_flat = torch.zeros((256 * 256,), device=device, dtype=torch.float32)
    trans_flat.scatter_add_(0, c_dst * 256 + c_src, torch.ones_like(c_src, dtype=torch.float32))
    t_log = torch.log1p(trans_flat.view(256, 256))
    pyr.coarse_transition.copy_(F.normalize(F.relu(t_log - t_log.mean(dim=0, keepdim=True)), p=2, dim=-1) * 0.20)
    target_v = torch.stack((a_self, b_orth), dim=-1).reshape(cfg.ambient_n, d) * math.sqrt(0.5)
    y_leaf = target_v.index_select(0, leaf_to_v)
    w256 = y_leaf.view(256, 256, d).mean(dim=1)
    r2048 = y_leaf.view(2048, 32, d).mean(dim=1) - w256.repeat_interleave(8, dim=0)
    r8192_full = y_leaf.view(8192, 8, d).mean(dim=1) - (w256.repeat_interleave(32, dim=0) + r2048.repeat_interleave(4, dim=0))
    up_mid = torch.linalg.svd(r8192_full, full_matrices=False)[2][: model.mid_detail_dim, :]
    w8192 = r8192_full @ up_mid.t()
    approx_8192 = (w256.repeat_interleave(32, dim=0) + r2048.repeat_interleave(4, dim=0) + w8192 @ up_mid).repeat_interleave(8, dim=0)
    r65536_full = y_leaf - approx_8192
    up_fine = torch.linalg.svd(r65536_full[0::4], full_matrices=False)[2][:d_f, :]
    sib8 = (r65536_full @ up_fine.t()).view(8192, 8, d_f)
    w_fine = (sib8 - 0.20 * sib8.mean(dim=1, keepdim=True)).reshape(cfg.ambient_n, d_f) * 1.03
    for dst_t, src_t in ((pyr.wav_256, w256), (pyr.wav_2048, r2048), (pyr.wav_8192, w8192), (pyr.up_8192, up_mid), (pyr.wav_65536, w_fine), (pyr.up_65536, up_fine)):
        dst_t.copy_(src_t)
    psi_all_v = pyr.embed_tokens(torch.arange(cfg.vocab_size, device=device))
    pred_bucket = torch.zeros((2048, d), device=device, dtype=torch.float32)
    dst_bucket = perm_v_to_leaf.index_select(0, dst) >> 5
    for start_i in range(0, src.numel(), 262144):
        s_c = src[start_i : start_i + 262144]
        pred_bucket.index_add_(0, dst_bucket[start_i : start_i + 262144], psi_all_v.index_select(0, s_c))
    pred_bucket[:, 0::2] *= 0.25
    pyr.pred_wav_2048.copy_(l2_normalize(pred_bucket, dim=-1) * 0.45)
    gen = torch.Generator(device=device).manual_seed(42)
    eye_d = torch.eye(d, device=device, dtype=torch.float32).unsqueeze(0).expand(cfg.kraus_rank, -1, -1)
    noise = torch.randn((cfg.kraus_rank, d, d), generator=gen, device=device) * (0.18 / math.sqrt(float(d)))
    pyr.kraus_generators.copy_(torch.linalg.qr(0.92 * eye_d + noise)[0])
    kraus_err = model.retract_kraus_generators()
    initialize_povm_log_prior(model, prior_tokens.to(device))
    m_sharp, _, _ = model.synthesize_detectors()
    v_probe = torch.arange(min(2048, cfg.valid_vocab_size), device=device)
    max_self_ov = float((pyr.embed_tokens(v_probe) * m_sharp[: v_probe.numel()]).sum(dim=-1).abs().max().item())
    sharp_rank = model.compute_sharp_eff_rank(m_sharp)
    model._last_sharp_eff_rank = sharp_rank
    return {"beta_0": model.beta_0, "max_self_overlap": max_self_ov, "sharp_eff_rank": sharp_rank, "kraus_stiefel_err": kraus_err}


def build_parseval_mup_optimizers(
    model: ParsevalMuPDyadicGPT,
    base_muon_lr: float = 0.026,
    base_wav_lr: float = 0.011,
    base_head_lr: float = 0.008,
) -> tuple[StiefelMuon, torch.optim.AdamW]:
    """Build StiefelMuon + AdamW optimizers with automatic spectral muP width scaling `eta(d) = eta(256) * sqrt(256 / d)`."""
    mup_scale = math.sqrt(256.0 / float(model.cfg.embed_dim))
    muon_lr, wav_lr = float(base_muon_lr) * mup_scale, float(base_wav_lr) * mup_scale
    head_lr, body_lr = float(base_head_lr) * mup_scale, 0.003 * mup_scale

    muon_params, wav_params, chol_params, body_params = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "kraus_generators" in name:
            muon_params.append(p)
        elif any(k in name for k in _WAV_KEYS):
            wav_params.append(p)
        elif any(k in name for k in _CHOL_KEYS):
            chol_params.append(p)
        elif p.ndim == 2 and min(p.shape) >= 32 and max(p.shape) <= 4096:
            muon_params.append(p)
        else:
            body_params.append(p)

    m_grp = [{"params": muon_params, "lr": muon_lr, "base_lr": muon_lr, "weight_decay": 0.01}]
    muon_opt = StiefelMuon(
        m_grp, lr=muon_lr, momentum=0.95, nesterov=True, weight_decay=0.01,
        decay_mode="cautious", stiefel_penalty=0.02, retract_every=2,
        polar_method="polar_express", use_normuon=True, use_twin_rail=True,
    )
    adam_groups = [
        {"params": wav_params, "lr": wav_lr, "base_lr": wav_lr, "weight_decay": 0.0, "group_name": "wavelet_2pow16"},
        {"params": chol_params, "lr": head_lr, "base_lr": head_lr, "weight_decay": 0.0, "group_name": "cholesky_readout"},
        {"params": body_params, "lr": body_lr, "base_lr": body_lr, "weight_decay": 0.01, "group_name": "body"},
    ]
    return muon_opt, torch.optim.AdamW([g for g in adam_groups if g["params"]], lr=head_lr, betas=(0.9, 0.95))
