"""Product-of-Spheres Multi-Subspace Coincidence Parseval muP Speedrun GPT (`Wave 31` / `EXPERIMENT_PLAN_046.md`).

When to Use This Building Block:
- Use `ParsevalMuPSubspaceCoincidenceGPT` when scaling `ParsevalMuPNGramSpeedrunGPT` (`Waves 29–30`)
  to prevent sharp detector rank collapse (`R_sharp`) and suppress single-subspace runner-up impostors
  via Product-of-Spheres `M_{d, M} = ((1 / sqrt(M)) S^{d/M - 1})^M` normalization (`M = 4` or `8`).

Why It Was Built & Mathematical Provenance (`EXPERIMENT_PLAN_046.md` / `Wave 31`, Lever 3):
1. Replaces single-sphere `S^{d-1}` detector and pre-readout normalization with Product-of-Spheres
   `M_{d, M} = ((1 / sqrt(M)) S^{d/M - 1})^M` normalization (`subspace_l2_normalize`), preventing
   additive branch norm cannibalization (`readout_ngram` / `wav_256` dominating fine wavelet
   leaves) and locking sharp detector effective rank `R_sharp` high throughout training.
2. Evaluates the M-subspace coincidence Born observable (`SubspaceCoincidenceSpec`) to suppress
   single-subspace positive runner-up impostors (`Z_pos_runnerup`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..functional import l2_normalize
from ..layers.attention import compute_document_ids
from ..layers.subspace_coincidence_born import (
    DEFAULT_COARSE_COINCIDENCE_WEIGHT,
    DEFAULT_COINCIDENCE_GAMMA,
    DEFAULT_NUM_SUBSPACES,
    DEFAULT_PARTITION_MODE,
    DEFAULT_SUBSPACE_BALANCE_BLEND,
    DEFAULT_SUBSPACE_VAR_PENALTY,
    SubspaceCoincidenceSpec,
    blend_subspace_l2_normalize,
    eval_subspace_coincidence_full_logits,
    subspace_coincidence_dyadic_chunked_ce,
)
from ..layers.wavelet_cholesky_branch import apply_soft_powerlaw_spectral_floor
from ..optim.muon import StiefelMuon
from .parseval_mup_dyadic_gpt import DEFAULT_KRAUS_SMOOTH_LANDING_RATE
from .parseval_mup_ngram_speedrun_gpt import (
    ParsevalMuPNGramSpeedrunConfig,
    ParsevalMuPNGramSpeedrunGPT,
    build_parseval_mup_ngram_optimizers,
    initialize_parseval_mup_ngram_with_ppmi_svd,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Documented in EXPERIMENT_PLAN_046.md (Wave 31) & EXPERIMENT_PLAN_048.md (Wave 33)
# ==============================================================================
DEFAULT_SUBSPACE_MUON_LR: float = 0.026
DEFAULT_SUBSPACE_WAV_LR: float = 0.0085
DEFAULT_SUBSPACE_HEAD_LR: float = 0.0050
MAX_INLINE_LOGIT_TOKENS: int = 256
DEFAULT_COINCIDENCE_WARMUP_START_TAU: float = 0.0
DEFAULT_COINCIDENCE_WARMUP_END_TAU: float = 1.0
DEFAULT_COARSE_WARMUP_FLOOR: float = 0.20


@dataclass
class ParsevalMuPSubspaceCoincidenceConfig(ParsevalMuPNGramSpeedrunConfig):
    """Configuration for Product-of-Spheres Multi-Subspace Coincidence GPT (`Wave 31`–`Wave 33`)."""

    num_subspaces: int = DEFAULT_NUM_SUBSPACES
    coincidence_gamma: float = DEFAULT_COINCIDENCE_GAMMA
    subspace_var_penalty: float = DEFAULT_SUBSPACE_VAR_PENALTY
    coarse_coincidence_weight: float = DEFAULT_COARSE_COINCIDENCE_WEIGHT
    partition_mode: str = DEFAULT_PARTITION_MODE
    full_vocab_coincidence_eval: bool = False
    use_coincidence_warmup: bool = False
    coincidence_warmup_start_tau: float = DEFAULT_COINCIDENCE_WARMUP_START_TAU
    coincidence_warmup_end_tau: float = DEFAULT_COINCIDENCE_WARMUP_END_TAU
    subspace_balance_blend_max: float = DEFAULT_SUBSPACE_BALANCE_BLEND


class ParsevalMuPSubspaceCoincidenceGPT(ParsevalMuPNGramSpeedrunGPT):
    """Product-of-Spheres `(S^{d/M - 1})^M` Multi-Subspace Coincidence Parseval muP GPT (`Wave 31`–`Wave 33`)."""

    cfg: ParsevalMuPSubspaceCoincidenceConfig

    def __init__(self, cfg: ParsevalMuPSubspaceCoincidenceConfig) -> None:
        super().__init__(cfg)
        self.num_subspaces = max(1, int(cfg.num_subspaces))

    def current_coincidence_ramp(self) -> float:
        """Return normalized coincidence warmup factor `r(tau) in [0, 1]` (`Wave 33` / `Plan 048`)."""
        if not bool(self.cfg.use_coincidence_warmup):
            return 1.0
        t0 = float(self.cfg.coincidence_warmup_start_tau)
        t1 = max(t0 + 1e-6, float(self.cfg.coincidence_warmup_end_tau))
        return min(1.0, max(0.0, (float(self._training_progress) - t0) / (t1 - t0)))

    def current_coincidence_gamma(self) -> float:
        """Return progress-ramped `coincidence_gamma(tau) = coincidence_gamma * r(tau)` (`Wave 33`)."""
        return float(self.cfg.coincidence_gamma) * self.current_coincidence_ramp()

    def current_subspace_var_penalty(self) -> float:
        """Return progress-ramped `subspace_var_penalty(tau) = subspace_var_penalty * r(tau)` (`Wave 33`)."""
        return float(self.cfg.subspace_var_penalty) * self.current_coincidence_ramp()

    def current_coarse_coincidence_weight(self) -> float:
        """Return progress-ramped 256-root coarse coincidence weight `w_coarse(tau)` (`Wave 33`)."""
        w_max = float(self.cfg.coarse_coincidence_weight)
        if not bool(self.cfg.use_coincidence_warmup):
            return w_max
        r = self.current_coincidence_ramp()
        return w_max * (DEFAULT_COARSE_WARMUP_FLOOR + (1.0 - DEFAULT_COARSE_WARMUP_FLOOR) * r)

    def current_subspace_balance_blend(self) -> float:
        """Return progress-ramped base-readout subspace balance blend `rho(tau) in [0, 1]` (`Wave 33`)."""
        b_max = float(self.cfg.subspace_balance_blend_max)
        if not bool(self.cfg.use_coincidence_warmup):
            return b_max
        return b_max * self.current_coincidence_ramp()

    def build_coincidence_spec(self) -> SubspaceCoincidenceSpec:
        """Build `SubspaceCoincidenceSpec` from model configuration and progress ramp `tau`."""
        s = self._beta_absorbed_scale
        return SubspaceCoincidenceSpec(
            num_subspaces=self.num_subspaces,
            coincidence_gamma=self.current_coincidence_gamma() / s,
            subspace_var_penalty=self.current_subspace_var_penalty() / s,
            coarse_coincidence_weight=self.current_coarse_coincidence_weight(),
            partition_mode=str(self.cfg.partition_mode),
            full_vocab_coincidence_eval=bool(self.cfg.full_vocab_coincidence_eval),
            subspace_balance_blend=self.current_subspace_balance_blend(),
        )

    def synthesize_detectors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Synthesize Product-of-Spheres balanced sharp & diffuse vocabulary detectors."""
        pyr = self.pyramid
        w_raw = F.softmax(pyr.kraus_raw_weights.float(), dim=0)
        w_k = apply_soft_powerlaw_spectral_floor(w_raw, min_eff_rank=self.cfg.kraus_spectral_floor)
        _, m_sharp_raw = pyr._synthesize_raw_detectors()
        m_sharp_bal = blend_subspace_l2_normalize(
            m_sharp_raw,
            num_subspaces=self.num_subspaces,
            blend=self.current_subspace_balance_blend(),
            partition_mode=str(self.cfg.partition_mode),
        )
        op_mean = torch.einsum("r,rij->ij", torch.sqrt(w_k + 1e-8), pyr.kraus_generators.float())
        if m_sharp_bal.is_cuda and torch.is_autocast_enabled():
            m_diff = l2_normalize(F.linear(m_sharp_bal.bfloat16(), op_mean.bfloat16()), dim=-1).to(m_sharp_bal.dtype)
        else:
            m_diff = l2_normalize(F.linear(m_sharp_bal.float(), op_mean), dim=-1).to(m_sharp_bal.dtype)
        return m_sharp_bal, m_diff, w_k

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_logits: bool | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Execute causal forward pass and Product-of-Spheres M-subspace coincidence Born readout."""
        m_sub = self.num_subspaces
        pmode = str(self.cfg.partition_mode)
        b_sub = self.current_subspace_balance_blend()
        psi_in_0 = self.pyramid.embed_tokens(idx)
        bigram_psi = self.pyramid.embed_dyadic_bigram_phase(idx).to(psi_in_0.dtype)
        in_ngram, readout_ngram = self.ngram_proj(idx, target_dtype=psi_in_0.dtype)

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
        for l_idx, block in enumerate(self.blocks):
            if l_idx == n_half:
                psi = l2_normalize(
                    psi + torch.tanh(self.bigram_mid_gate).to(psi.dtype) * bigram_psi,
                    dim=-1,
                )
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
        m_sharp_unit, m_diff, w_k = self.synthesize_detectors()
        use_chol = bool(self._beta_absorbed_scale != 1.0 or not idx.is_cuda)
        psi_s_out = self.pyramid.readout_chol(psi_pre) if use_chol else psi_pre
        m_sharp = self.pyramid.readout_chol(m_sharp_unit) if use_chol else m_sharp_unit
        self._last_diffuse_eff_rank = float(1.0 / ((w_k * w_k).sum().item() + 1e-12))
        self._last_mean_beta = self.current_beta()
        self._last_lambda_diff = self.current_lambda_diff()

        r_spec = self.build_readout_spec()
        c_spec = self.build_coincidence_spec()
        want_logits = (targets is None or idx.numel() <= 256) if return_logits is None else bool(return_logits)
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
            else None
        )
        return logits, loss

    @torch.no_grad()
    def retract_kraus_generators(self, smooth_landing_rate: float = 0.08) -> float:
        """Retract `kraus_generators` and `ngram_proj.u_ngram` onto the Stiefel manifold."""
        return super().retract_kraus_generators(smooth_landing_rate=smooth_landing_rate)


@torch.no_grad()
def initialize_subspace_coincidence_with_ppmi_svd(
    model: ParsevalMuPSubspaceCoincidenceGPT,
    ppmi_embeddings: torch.Tensor,
    init_tokens: torch.Tensor,
    prior_tokens: torch.Tensor,
) -> dict[str, float]:
    """Initialize `ParsevalMuPSubspaceCoincidenceGPT` and record Product-of-Spheres `R_sharp`."""
    stats = initialize_parseval_mup_ngram_with_ppmi_svd(model, ppmi_embeddings, init_tokens, prior_tokens)
    m_sharp, _, _ = model.synthesize_detectors()
    stats["sharp_eff_rank"] = model.compute_sharp_eff_rank(m_sharp)
    return stats


def build_subspace_coincidence_optimizers(
    model: ParsevalMuPSubspaceCoincidenceGPT,
    base_muon_lr: float = 0.026,
    base_wav_lr: float = 0.0085,
    base_head_lr: float = 0.005,
) -> tuple[StiefelMuon, torch.optim.AdamW]:
    """Build width-invariant `muP` (`sqrt(256 / d)`) `StiefelMuon` + `AdamW` optimizers."""
    return build_parseval_mup_ngram_optimizers(
        model, base_muon_lr=base_muon_lr, base_wav_lr=base_wav_lr, base_head_lr=base_head_lr
    )


__all__ = [
    "DEFAULT_COARSE_WARMUP_FLOOR",
    "DEFAULT_COINCIDENCE_WARMUP_END_TAU",
    "DEFAULT_COINCIDENCE_WARMUP_START_TAU",
    "ParsevalMuPSubspaceCoincidenceConfig",
    "ParsevalMuPSubspaceCoincidenceGPT",
    "build_subspace_coincidence_optimizers",
    "initialize_subspace_coincidence_with_ppmi_svd",
]
