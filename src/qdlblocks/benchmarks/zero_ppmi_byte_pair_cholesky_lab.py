"""Strict Speedrun-Compliant Zero-PPMI Byte-Pair (`b_hi, b_lo`) & Chain-Rule Quantum Cholesky Architecture (`EXPERIMENT_PLAN_081.md` / `Wave 61`).

Eliminates offline `900M`-token PPMI and n-gram pre-scans (`0` training tokens touched before Step 1, `< 50 ms` Step-0 init)
while preserving the exact external `modded-nanogpt` signature `idx: (B, T) -> CE loss over 50,257 GPT-2 tokens` via:
1. **Deterministic `uint16` Radix Byte-Pair Bijection**:
   $$b_{\\text{hi}}(v) = v \\gg 8 \\in \\{0, \\dots, 196\\}, \\qquad b_{\\text{lo}}(v) = v \\ \\& \\ 255 \\in \\{0, \\dots, 255\\}, \\qquad v = 256\\,b_{\\text{hi}}(v) + b_{\\text{lo}}(v)$$
2. **Zero-Data Analytical BPE-Rank Zipf Prior & Online Causal Training-Stream EMA Tracker (`OnlineCausalZipfTracker`)**:
   Initializes `unigram_bias` purely as a closed-form function of token ID $v$ at Step 0 (`8.9050` Val CE vs `10.8258` uniform with `0` tokens scanned) and updates running `(b_hi, b_lo)` + token counts strictly online during `model.training` forward passes.
3. **Factored `(b_hi, b_lo)` + Schur Coincidence Wavepacket & Detector Synthesizer (`BytePairSchurWavepacketSynthesizer`)**:
   Shares high-byte (`197 x 512`), low-byte (`256 x 512`), and Schur coincidence $\\hat{D}_{\\text{hi}}[b_{\\text{hi}}] \\odot \\hat{D}_{\\text{lo}}[b_{\\text{lo}}]$ basis vectors across all `50,257` tokens so `100.0%` of vocabulary detectors receive gradient updates on Step 1.
4. **Exact Chain-Rule Two-Byte Autoregressive Readout (`ExactChainRuleTwoByteReadout`, `197 + 256 = 453` Outcomes)**:
   Factorizes $\\ln P(v \\mid h_t) = \\ln P(b_{\\text{hi}}(v) \\mid h_t) + \\ln P(b_{\\text{lo}}(v) \\mid h_t, b_{\\text{hi}}(v))$ with exact invalid-tail masking on bucket `196` ($b_{\\text{lo}} \\in \\{0..80\\}$), satisfying $\\sum_{v=0}^{50,256} P(v \\mid h_t) \\equiv 1.0$.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from qdlblocks.benchmarks.experiment_logger import ExperimentIntentSpec, StandardizedExperimentLogger
from qdlblocks.benchmarks.noclick_vacuum_and_smith_sgd_lab import (
    NoClickVacuumCholeskyGPT,
    NoClickVacuumCholeskySpec,
    NoClickVacuumVariantMode,
    _apply_global_optimizer_step,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import (
    DEFAULT_CARRIER_BIAS_DIM,
    DEFAULT_SHALLOW_NGRAM_LR,
    DEFAULT_SHALLOW_SCALAR_LR,
    DEFAULT_SOFTCAP_LOGIT,
    RAW_GPT2_VOCAB_SIZE,
    LongSeqFineWebStream,
)
from qdlblocks.benchmarks.vacuum_subspace_cholesky_lab import (
    DEFAULT_VACUUM_SUBSPACE_DIM as DEFAULT_VACUUM_DIM,
    _VacuumFourMode512DWavepacketCEFn,
    vacuum_sphere_rms_norm,
)
from qdlblocks.benchmarks.variable_rank_cholesky_vacuum_lab import (
    build_layer0_variable_rank_cholesky_factor,
    build_variable_rank_cholesky_optimizers,
    evaluate_variable_rank_cholesky_model,
)
from qdlblocks.layers.quantum_statmech_channels import QuantumMomentumScheduleMode, compute_riemannian_momentum_beta
from qdlblocks.models.modular_quantum_statmech_gpt import ModularRiemannianStiefelKrausMuon
from qdlblocks.optim.vocab_manifold_muon import BatchSubspaceTikhonovGramMuon

# -----------------------------------------------------------------------------
# Top-of-File Named Constants (Synchronized with EXPERIMENT_PLAN_081.md Rule 7)
# -----------------------------------------------------------------------------
NUM_HIGH_BYTE_BUCKETS: int = 197
NUM_LOW_BYTE_BUCKETS: int = 256
LAST_HIGH_BYTE_BUCKET: int = 196
LAST_BUCKET_VALID_LOW_BYTES: int = 81
DEFAULT_ZIPF_ALPHA: float = 1.08
DEFAULT_ZIPF_BPE_OFFSET: float = 200.0
DEFAULT_ZIPF_ASCII_RANK: float = 180.0
DEFAULT_ZIPF_HIGH_BYTE_RANK: float = 8000.0
DEFAULT_ZIPF_EOT_RANK: float = 600.0
DEFAULT_ONLINE_EMA_PSEUDOCOUNT: float = 400.0
DEFAULT_ONLINE_EMA_DECAY: float = 0.992
DEFAULT_BYTE_IN_HI_SCALE: float = 0.35
DEFAULT_BYTE_IN_LO_SCALE: float = 0.25
DEFAULT_BYTE_IN_SCHUR_SCALE: float = 0.20
DEFAULT_BYTE_DET_HI_SCALE: float = 0.30
DEFAULT_BYTE_DET_LO_SCALE: float = 0.25
DEFAULT_BYTE_DET_SCHUR_SCALE: float = 0.20
DEFAULT_HIER_BHI_PRIOR_SCALE: float = 0.22
DEFAULT_BYTE_NGRAM_BUCKETS: int = 4096


class ZeroPPMIBytePairVariantMode(str, Enum):
    """Ablation and scaling modes for `EXPERIMENT_PLAN_081.md` (`Wave 61`)."""

    Z0_ZERO_PPMI_COLD_START_BASELINE = "z0_zero_ppmi_cold_start_baseline"
    Z1_ZERO_PPMI_ONLINE_EMA_ZIPF = "z1_zero_ppmi_online_ema_zipf"
    Z2_ZERO_PPMI_FACTORED_BYTE_PAIR_SCHUR = "z2_zero_ppmi_factored_byte_pair_schur"
    Z3_ZERO_PPMI_EXACT_CHAIN_RULE_TWOBYTE = "z3_zero_ppmi_exact_chain_rule_twobyte"
    Z4_ZERO_PPMI_HIERARCHICAL_BYTE_PAIR_R12 = "z4_zero_ppmi_hierarchical_byte_pair_r12"
    Z5_ZERO_PPMI_BYTE_PAIR_SCALING_1200 = "z5_zero_ppmi_byte_pair_scaling_1200"


@dataclass(frozen=True)
class ZeroPPMIBytePairCholeskySpec:
    """Configuration for Strict Speedrun-Compliant Zero-PPMI Byte-Pair Cholesky GPT (`<= 8` fields)."""

    variant_mode: ZeroPPMIBytePairVariantMode = ZeroPPMIBytePairVariantMode.Z2_ZERO_PPMI_FACTORED_BYTE_PAIR_SCHUR
    num_cholesky_modes: int = 8
    vacuum_dim: int = DEFAULT_VACUUM_DIM
    enable_online_ema: bool = True
    enable_byte_pair_schur: bool = True
    use_exact_chain_rule_readout: bool = False
    enable_hierarchical_bhi_carrier: bool = False
    seed: int = 1337


def decompose_token_uint16_byte_pair(idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Deterministically map GPT-2 token IDs `idx` $\in \{0, \dots, 50256\}$ to `(b_hi, b_lo)` via bit shifts."""
    idx_long = idx.to(torch.long)
    b_hi = (idx_long >> 8).clamp(0, NUM_HIGH_BYTE_BUCKETS - 1)
    b_lo = idx_long & 0xFF
    return b_hi, b_lo


def reconstruct_token_from_uint16_byte_pair(b_hi: torch.Tensor, b_lo: torch.Tensor) -> torch.Tensor:
    r"""Invert `decompose_token_uint16_byte_pair`: $v = (b_{\text{hi}} \ll 8) \mid b_{\text{lo}}$."""
    return (b_hi.to(torch.long) << 8) | (b_lo.to(torch.long) & 0xFF)


def build_zero_data_bpe_rank_zipf_prior(
    vocab_size: int, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Construct Step-0 analytical BPE-rank Zipf log-prior `b_zipf` `(V,)`, `b_hi_zipf` `(197,)`, and `radial` `(V,)` with zero data scan."""
    v_idx = torch.arange(vocab_size, device=device, dtype=torch.float32)
    raw_logits = torch.full((vocab_size,), -18.0, device=device, dtype=torch.float32)

    n_valid = min(RAW_GPT2_VOCAB_SIZE, vocab_size)
    ascii_end = min(95, n_valid)
    byte_end = min(256, n_valid)
    raw_logits[:ascii_end] = -DEFAULT_ZIPF_ALPHA * math.log(DEFAULT_ZIPF_ASCII_RANK)
    if byte_end > ascii_end:
        raw_logits[ascii_end:byte_end] = -DEFAULT_ZIPF_ALPHA * math.log(DEFAULT_ZIPF_HIGH_BYTE_RANK)
    if n_valid > 256:
        bpe_end = min(RAW_GPT2_VOCAB_SIZE - 1, n_valid)
        ranks = (v_idx[256:bpe_end] - DEFAULT_ZIPF_BPE_OFFSET).clamp_min(32.0)
        raw_logits[256:bpe_end] = -DEFAULT_ZIPF_ALPHA * torch.log(ranks)
    if n_valid >= RAW_GPT2_VOCAB_SIZE:
        raw_logits[RAW_GPT2_VOCAB_SIZE - 1] = -DEFAULT_ZIPF_ALPHA * math.log(DEFAULT_ZIPF_EOT_RANK)

    log_p_tok = F.log_softmax(raw_logits, dim=0)
    p_tok = torch.exp(log_p_tok)
    b_hi_all, _ = decompose_token_uint16_byte_pair(torch.arange(vocab_size, device=device))
    p_hi = torch.zeros((NUM_HIGH_BYTE_BUCKETS,), device=device, dtype=torch.float32)
    p_hi.scatter_add_(0, b_hi_all[:n_valid], p_tok[:n_valid])
    log_p_hi = torch.log(p_hi.clamp_min(1e-8))

    valid_slice = log_p_tok[:n_valid]
    mean_lp = valid_slice.mean()
    std_lp = valid_slice.std().clamp_min(1e-5)
    z_lp = (log_p_tok - mean_lp) / std_lp
    radial = (1.0 + 0.14 * z_lp).clamp(0.65, 1.55)
    if vocab_size > n_valid:
        radial[n_valid:] = 0.25
    return log_p_tok, log_p_hi, radial


class OnlineCausalZipfTracker(nn.Module):
    r"""Online training-stream EMA unigram & `(b_hi, b_lo)` tracker using zero offline pre-scan tokens."""

    def __init__(self, vocab_size: int, enable_online_ema: bool = True) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.enable_online_ema = bool(enable_online_ema)
        log_p_tok, log_p_hi, _ = build_zero_data_bpe_rank_zipf_prior(vocab_size)
        p_zipf = torch.exp(log_p_tok)
        p_hi_zipf = torch.exp(log_p_hi)

        b_hi_vocab, _ = decompose_token_uint16_byte_pair(torch.arange(vocab_size))
        self.register_buffer("b_hi_vocab", b_hi_vocab)
        self.register_buffer("log_p_zipf", log_p_tok)
        self.register_buffer("p_zipf", p_zipf)
        self.register_buffer("log_p_hi_zipf", log_p_hi)
        self.register_buffer("p_hi_zipf", p_hi_zipf)
        self.register_buffer("ema_tok_counts", DEFAULT_ONLINE_EMA_PSEUDOCOUNT * p_zipf.clone())
        self.register_buffer("ema_bhi_counts", (DEFAULT_ONLINE_EMA_PSEUDOCOUNT * 0.25) * p_hi_zipf.clone())
        self.register_buffer("batches_seen", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update_from_training_batch_(self, idx_in: torch.Tensor) -> None:
        """Update running token and `b_hi` counts strictly from the current training batch."""
        if not self.enable_online_ema:
            return
        flat = idx_in.reshape(-1).to(torch.long)
        tok_bc = torch.bincount(flat.clamp(0, self.vocab_size - 1), minlength=self.vocab_size).to(torch.float32)
        b_hi = (flat >> 8).clamp(0, NUM_HIGH_BYTE_BUCKETS - 1)
        bhi_bc = torch.bincount(b_hi, minlength=NUM_HIGH_BYTE_BUCKETS).to(torch.float32)

        decay = float(DEFAULT_ONLINE_EMA_DECAY)
        self.ema_tok_counts.mul_(decay).add_(tok_bc * (1.0 - decay))
        self.ema_bhi_counts.mul_(decay).add_(bhi_bc * (1.0 - decay))
        self.batches_seen.add_(1)

    def current_log_priors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(log_p_tok (V,), log_p_hi (197,))` combining online `(b_hi, token)` counts with analytical Zipf."""
        if not self.enable_online_ema or int(self.batches_seen.item()) == 0:
            return self.log_p_zipf, self.log_p_hi_zipf

        n_valid = min(RAW_GPT2_VOCAB_SIZE, self.vocab_size)
        p_hi_cur = self.ema_bhi_counts / self.ema_bhi_counts.sum().clamp_min(1e-6)
        log_p_hi = torch.log(p_hi_cur.clamp_min(1e-8))

        tok_c = self.ema_tok_counts[:n_valid]
        p_tok_direct = tok_c / tok_c.sum().clamp_min(1e-6)

        bhi_idx = self.b_hi_vocab[:n_valid]
        bhi_sum_tok = torch.zeros((NUM_HIGH_BYTE_BUCKETS,), device=tok_c.device, dtype=torch.float32)
        bhi_sum_tok.scatter_add_(0, bhi_idx, tok_c)
        p_lo_given_hi = tok_c / bhi_sum_tok[bhi_idx].clamp_min(1e-6)
        p_tok_hier = p_hi_cur[bhi_idx] * p_lo_given_hi

        p_blend = 0.55 * p_tok_direct + 0.45 * p_tok_hier
        log_valid = torch.log(p_blend.clamp_min(1e-9))
        if self.vocab_size > n_valid:
            pad_tail = torch.full((self.vocab_size - n_valid,), -18.0, device=tok_c.device, dtype=torch.float32)
            log_full = torch.cat([log_valid, pad_tail], dim=0)
        else:
            log_full = log_valid
        return F.log_softmax(log_full, dim=0), log_p_hi


class ZeroPPMIBytePairNGramProjector(nn.Module):
    r"""Zero-initialized causal `(b_hi, b_lo)` & token N-gram projector requiring zero offline shard calibration."""

    def __init__(
        self, num_buckets: int = DEFAULT_BYTE_NGRAM_BUCKETS, sem_dim: int = 448, id_dim: int = 64, d_vac: int = DEFAULT_VACUUM_DIM
    ) -> None:
        super().__init__()
        self.num_buckets = int(num_buckets)
        self.sem_dim = int(sem_dim)
        self.id_dim = int(id_dim)
        self.d_vac = int(d_vac)

        self.bhi_unigram_sem = nn.Embedding(NUM_HIGH_BYTE_BUCKETS, self.sem_dim)
        self.bhi_unigram_id = nn.Embedding(NUM_HIGH_BYTE_BUCKETS, self.id_dim)
        self.blo_unigram_sem = nn.Embedding(NUM_LOW_BYTE_BUCKETS, self.sem_dim)
        self.bhi_bigram_sem = nn.Embedding(self.num_buckets, self.sem_dim)
        self.blo_bigram_sem = nn.Embedding(self.num_buckets, self.sem_dim)
        self.tok_bigram_sem = nn.Embedding(self.num_buckets, self.sem_dim)
        self.tok_bigram_id = nn.Embedding(self.num_buckets, self.id_dim)

        for emb in (
            self.bhi_unigram_sem,
            self.bhi_unigram_id,
            self.blo_unigram_sem,
            self.bhi_bigram_sem,
            self.blo_bigram_sem,
            self.tok_bigram_sem,
            self.tok_bigram_id,
        ):
            nn.init.zeros_(emb.weight)

        self.in_sem_scale = nn.Parameter(torch.tensor(0.28, dtype=torch.float32))
        self.out_sem_scale = nn.Parameter(torch.tensor(0.55, dtype=torch.float32))
        self.out_id_scale = nn.Parameter(torch.tensor(0.38, dtype=torch.float32))

    def table_parameters(self) -> list[nn.Parameter]:
        """Return embedding table parameters for high-LR online N-gram updates."""
        return [
            self.bhi_unigram_sem.weight,
            self.bhi_unigram_id.weight,
            self.blo_unigram_sem.weight,
            self.bhi_bigram_sem.weight,
            self.blo_bigram_sem.weight,
            self.tok_bigram_sem.weight,
            self.tok_bigram_id.weight,
        ]

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return scalar gate parameters."""
        return [self.in_sem_scale, self.out_sem_scale, self.out_id_scale]

    def _compute_bucket_indices(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute causal `(b_hi, b_lo, b_hi bigram, b_lo bigram, token bigram/trigram)` indices `(B, T)`."""
        bsz, seq_len = idx.shape
        b_hi, b_lo = decompose_token_uint16_byte_pair(idx)
        if seq_len > 1:
            b_hi_prev = torch.cat([b_hi[:, :1], b_hi[:, :-1]], dim=1)
            b_lo_prev = torch.cat([b_lo[:, :1], b_lo[:, :-1]], dim=1)
            idx_prev = torch.cat([idx[:, :1], idx[:, :-1]], dim=1).to(torch.long)
        else:
            b_hi_prev, b_lo_prev, idx_prev = b_hi, b_lo, idx.to(torch.long)
        if seq_len > 2:
            idx_prev2 = torch.cat([idx[:, :2], idx[:, :-2]], dim=1).to(torch.long)
        else:
            idx_prev2 = idx_prev

        nb = self.num_buckets
        h_bhi = (b_hi * 257 + b_hi_prev * 31 + b_lo_prev) % nb
        h_blo = (b_lo * 263 + b_lo_prev * 131 + b_hi) % nb
        h_tok = (idx.to(torch.long) * 1000003 + idx_prev * 65537 + idx_prev2 * 257) % nb
        return b_hi, b_lo, h_bhi, h_blo, h_tok

    def forward(
        self, idx: torch.Tensor, work_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(in_full_delta (B, T, 512), readout_sem_448 (B, T, 448), readout_id_64 (B, T, 64))`."""
        bsz, seq_len = idx.shape
        b_hi, b_lo, h_bhi, h_blo, h_tok = self._compute_bucket_indices(idx)
        sem_raw = (
            self.bhi_unigram_sem(b_hi).float()
            + 0.65 * self.blo_unigram_sem(b_lo).float()
            + self.bhi_bigram_sem(h_bhi).float()
            + 0.65 * self.blo_bigram_sem(h_blo).float()
            + self.tok_bigram_sem(h_tok).float()
        )
        id_raw = self.bhi_unigram_id(b_hi).float() + self.tok_bigram_id(h_tok).float()

        sem_448 = (sem_raw * torch.rsqrt(sem_raw.square().mean(dim=-1, keepdim=True) + 0.0625)).to(work_dtype)
        id_64 = (id_raw * torch.rsqrt(id_raw.square().mean(dim=-1, keepdim=True) + 0.25)).to(work_dtype)

        in_sem = torch.tanh(self.in_sem_scale).to(work_dtype) * sem_448
        zero_id = torch.zeros((bsz, seq_len, self.id_dim), device=idx.device, dtype=work_dtype)
        in_full_delta = torch.cat([in_sem, zero_id], dim=-1)

        out_sem = (1.5 * torch.tanh(self.out_sem_scale)).to(work_dtype) * sem_448
        out_id = (1.5 * torch.tanh(self.out_id_scale)).to(work_dtype) * id_64
        return in_full_delta, out_sem, out_id


class BytePairSchurWavepacketSynthesizer(nn.Module):
    r"""Deterministic `(b_hi, b_lo)` + Schur Coincidence synthesizer for input wavepackets and readout detectors."""

    def __init__(
        self,
        vocab_size: int,
        pkt_count: int = 8,
        pkt_dim: int = 64,
        sem_dim: int = 448,
        id_dim: int = 64,
        carrier_dim: int = DEFAULT_CARRIER_BIAS_DIM,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.pkt_count = int(pkt_count)
        self.pkt_dim = int(pkt_dim)
        self.embed_dim = self.pkt_count * self.pkt_dim
        self.sem_dim = int(sem_dim)
        self.id_dim = int(id_dim)
        self.carrier_dim = int(carrier_dim)

        b_hi_v, b_lo_v = decompose_token_uint16_byte_pair(torch.arange(self.vocab_size))
        self.register_buffer("vocab_b_hi", b_hi_v)
        self.register_buffer("vocab_b_lo", b_lo_v)

        self.u_in_hi = nn.Parameter(torch.empty((NUM_HIGH_BYTE_BUCKETS, self.pkt_count, self.pkt_dim), dtype=torch.float32))
        self.u_in_lo = nn.Parameter(torch.empty((NUM_LOW_BYTE_BUCKETS, self.pkt_count, self.pkt_dim), dtype=torch.float32))
        self.d_out_hi = nn.Parameter(torch.empty((NUM_HIGH_BYTE_BUCKETS, self.embed_dim), dtype=torch.float32))
        self.d_out_lo = nn.Parameter(torch.empty((NUM_LOW_BYTE_BUCKETS, self.embed_dim), dtype=torch.float32))

        self.in_hi_scale = nn.Parameter(torch.tensor(DEFAULT_BYTE_IN_HI_SCALE, dtype=torch.float32))
        self.in_lo_scale = nn.Parameter(torch.tensor(DEFAULT_BYTE_IN_LO_SCALE, dtype=torch.float32))
        self.in_schur_scale = nn.Parameter(torch.tensor(DEFAULT_BYTE_IN_SCHUR_SCALE, dtype=torch.float32))

        self.det_hi_scale = nn.Parameter(torch.tensor(DEFAULT_BYTE_DET_HI_SCALE, dtype=torch.float32))
        self.det_lo_scale = nn.Parameter(torch.tensor(DEFAULT_BYTE_DET_LO_SCALE, dtype=torch.float32))
        self.det_schur_scale = nn.Parameter(torch.tensor(DEFAULT_BYTE_DET_SCHUR_SCALE, dtype=torch.float32))

        self.ctx_bhi_proj = nn.Linear(self.embed_dim, self.carrier_dim, bias=False)
        self.vocab_bhi_carrier = nn.Parameter(torch.empty((NUM_HIGH_BYTE_BUCKETS, self.carrier_dim), dtype=torch.float32))
        self.vocab_blo_carrier = nn.Parameter(torch.empty((NUM_LOW_BYTE_BUCKETS, self.carrier_dim), dtype=torch.float32))
        self.hier_bhi_scale = nn.Parameter(torch.tensor(DEFAULT_HIER_BHI_PRIOR_SCALE, dtype=torch.float32))

        self.reset_parameters_()

    def table_parameters(self) -> list[nn.Parameter]:
        """Return `(b_hi, b_lo)` basis tables for fast online AdamW optimization."""
        return [
            self.u_in_hi,
            self.u_in_lo,
            self.d_out_hi,
            self.d_out_lo,
            self.ctx_bhi_proj.weight,
            self.vocab_bhi_carrier,
            self.vocab_blo_carrier,
        ]

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return scalar gate parameters."""
        return [
            self.in_hi_scale,
            self.in_lo_scale,
            self.in_schur_scale,
            self.det_hi_scale,
            self.det_lo_scale,
            self.det_schur_scale,
            self.hier_bhi_scale,
        ]

    @torch.no_grad()
    def reset_parameters_(self, seed: int = 1337) -> None:
        """Initialize `(b_hi, b_lo)` wavepacket and detector tables on unit spheres with BPE merge-rank harmonics."""
        gen = torch.Generator(device=self.u_in_hi.device)
        gen.manual_seed(int(seed) + 7919)

        raw_hi = torch.randn(self.u_in_hi.shape, generator=gen, device=self.u_in_hi.device, dtype=torch.float32)
        raw_lo = torch.randn(self.u_in_lo.shape, generator=gen, device=self.u_in_lo.device, dtype=torch.float32)

        t_hi = torch.linspace(0.0, math.pi, NUM_HIGH_BYTE_BUCKETS, device=self.u_in_hi.device, dtype=torch.float32)
        for k_harm in range(4):
            freq = float(k_harm + 1)
            raw_hi[:, k_harm, 0] += 1.5 * torch.cos(freq * t_hi)
            raw_hi[:, k_harm, 1] += 1.5 * torch.sin(freq * t_hi)

        norm_hi = F.normalize(raw_hi, dim=-1, eps=1e-6)
        norm_lo = F.normalize(raw_lo, dim=-1, eps=1e-6)
        self.u_in_hi.copy_(norm_hi)
        self.u_in_lo.copy_(norm_lo)

        flat_hi = norm_hi.reshape(NUM_HIGH_BYTE_BUCKETS, self.embed_dim)
        flat_lo = norm_lo.reshape(NUM_LOW_BYTE_BUCKETS, self.embed_dim)
        self.d_out_hi.copy_(F.rms_norm(flat_hi, (self.embed_dim,), eps=1e-6))
        self.d_out_lo.copy_(F.rms_norm(flat_lo, (self.embed_dim,), eps=1e-6))

        nn.init.normal_(self.ctx_bhi_proj.weight, std=0.02)
        c_hi = torch.randn(self.vocab_bhi_carrier.shape, generator=gen, device=self.u_in_hi.device, dtype=torch.float32)
        c_lo = torch.randn(self.vocab_blo_carrier.shape, generator=gen, device=self.u_in_hi.device, dtype=torch.float32)
        self.vocab_bhi_carrier.copy_(F.rms_norm(c_hi, (self.carrier_dim,), eps=1e-6))
        self.vocab_blo_carrier.copy_(F.rms_norm(c_lo, (self.carrier_dim,), eps=1e-6) * 0.35)

    def lift_input_wavepackets(self, idx: torch.Tensor, base_wp: torch.Tensor) -> torch.Tensor:
        r"""Combine fine token wavepackets `base_wp` `(B, T, 8, 64)` with `(b_hi, b_lo)` and Schur coincidence."""
        work_dtype = base_wp.dtype
        b_hi, b_lo = decompose_token_uint16_byte_pair(idx)
        u_hi = F.embedding(b_hi, self.u_in_hi.view(NUM_HIGH_BYTE_BUCKETS, self.embed_dim)).view_as(base_wp).to(work_dtype)
        u_lo = F.embedding(b_lo, self.u_in_lo.view(NUM_LOW_BYTE_BUCKETS, self.embed_dim)).view_as(base_wp).to(work_dtype)

        u_hi_rms = F.rms_norm(u_hi, (self.pkt_dim,), eps=1e-6)
        u_lo_rms = F.rms_norm(u_lo, (self.pkt_dim,), eps=1e-6)
        u_schur = u_hi_rms * u_lo_rms

        g_hi = self.in_hi_scale.to(work_dtype)
        g_lo = self.in_lo_scale.to(work_dtype)
        g_sc = self.in_schur_scale.to(work_dtype)
        lifted = base_wp + (g_hi * u_hi_rms + g_lo * u_lo_rms + g_sc * u_schur) * (1.0 / math.sqrt(float(self.pkt_dim)))
        return F.normalize(lifted.float(), dim=-1, eps=1e-6).to(work_dtype)

    def synthesize_detector_table(self, w_det_base: torch.Tensor, work_dtype: torch.dtype) -> torch.Tensor:
        r"""Synthesize `(50304, 512)` readout detectors combining `w_det_base` with shared `(D_hi, D_lo, D_hi * D_lo)`."""
        d_hi_rms = F.rms_norm(self.d_out_hi.to(work_dtype), (self.embed_dim,), eps=1e-6)
        d_lo_rms = F.rms_norm(self.d_out_lo.to(work_dtype), (self.embed_dim,), eps=1e-6)

        hi_exp = d_hi_rms[self.vocab_b_hi]
        lo_exp = d_lo_rms[self.vocab_b_lo]
        schur_exp = hi_exp * lo_exp

        g_hi = self.det_hi_scale.to(work_dtype)
        g_lo = self.det_lo_scale.to(work_dtype)
        g_sc = self.det_schur_scale.to(work_dtype)
        w_comb = w_det_base.to(work_dtype) + g_hi * hi_exp + g_lo * lo_exp + g_sc * schur_exp

        w_sem = F.rms_norm(w_comb[:, : self.sem_dim], (self.sem_dim,), eps=1e-6)
        w_id = F.rms_norm(w_comb[:, self.sem_dim :], (self.id_dim,), eps=1e-6)
        return torch.cat([w_sem, w_id], dim=-1)

    def build_contextual_byte_carrier(
        self, h_pre: torch.Tensor, carrier_c0: torch.Tensor, work_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build `16D` byte-pair carrier strictly orthogonal to `carrier_c0` (`15D` active orthogonal complement)."""
        c0_unit = F.normalize(carrier_c0.float(), dim=-1, eps=1e-6)
        raw_h = F.linear(h_pre.to(work_dtype), self.ctx_bhi_proj.weight.to(work_dtype)).float()
        h_perp = raw_h - (raw_h * c0_unit.view(1, 1, -1)).sum(dim=-1, keepdim=True) * c0_unit.view(1, 1, -1)
        h_car_16 = (self.hier_bhi_scale.float() * F.normalize(h_perp, dim=-1, eps=1e-6)).to(work_dtype)

        raw_w = self.vocab_bhi_carrier[self.vocab_b_hi].float() + 0.35 * self.vocab_blo_carrier[self.vocab_b_lo].float()
        w_perp = raw_w - (raw_w * c0_unit.view(1, -1)).sum(dim=-1, keepdim=True) * c0_unit.view(1, -1)
        w_car_16 = F.normalize(w_perp, dim=-1, eps=1e-6).to(work_dtype)
        return h_car_16, w_car_16


class ExactChainRuleTwoByteReadout(nn.Module):
    r"""Exact Chain-Rule Two-Byte Autoregressive Readout (`197` $b_{\text{hi}}$ + `256` $b_{\text{lo}} \mid b_{\text{hi}}$ outcomes).

    Computes exact `50,257`-token probabilities via:
    $$\ln P_{\text{chain}}(v \mid h_t) = \ln P(b_{\text{hi}}(v) \mid h_t) + \ln P(b_{\text{lo}}(v) \mid h_t, b_{\text{hi}}(v))$$
    with invalid-tail masking on bucket `b_hi = 196` ($b_{\text{lo}} \in \{81, \dots, 255\} \mapsto -\infty$) so that
    $\sum_{v=0}^{50,256} P_{\text{chain}}(v \mid h_t) \equiv 1.000000$ without any `50,304`-way matmul during training.
    """

    def __init__(self, sem_dim: int = 448, id_dim: int = 64) -> None:
        super().__init__()
        self.sem_dim = int(sem_dim)
        self.id_dim = int(id_dim)
        self.embed_dim = self.sem_dim + self.id_dim

        self.w_hi_sem = nn.Parameter(torch.empty((NUM_HIGH_BYTE_BUCKETS, self.sem_dim), dtype=torch.float32))
        self.w_hi_id = nn.Parameter(torch.empty((NUM_HIGH_BYTE_BUCKETS, self.id_dim), dtype=torch.float32))
        self.c_hi_sem = nn.Parameter(torch.zeros((NUM_HIGH_BYTE_BUCKETS, self.sem_dim), dtype=torch.float32))
        self.c_hi_id = nn.Parameter(torch.zeros((NUM_HIGH_BYTE_BUCKETS, self.id_dim), dtype=torch.float32))

        self.w_lo_sem = nn.Parameter(torch.empty((NUM_LOW_BYTE_BUCKETS, self.sem_dim), dtype=torch.float32))
        self.w_lo_id = nn.Parameter(torch.empty((NUM_LOW_BYTE_BUCKETS, self.id_dim), dtype=torch.float32))
        self.pair_bias = nn.Parameter(torch.zeros((NUM_HIGH_BYTE_BUCKETS, NUM_LOW_BYTE_BUCKETS), dtype=torch.float32))
        self.hi_radial_scale = nn.Parameter(torch.tensor(2.20, dtype=torch.float32))
        self.lo_radial_scale = nn.Parameter(torch.tensor(2.20, dtype=torch.float32))

        valid_mask = torch.ones((NUM_HIGH_BYTE_BUCKETS, NUM_LOW_BYTE_BUCKETS), dtype=torch.bool)
        valid_mask[LAST_HIGH_BYTE_BUCKET, LAST_BUCKET_VALID_LOW_BYTES:] = False
        self.register_buffer("valid_pair_mask", valid_mask)
        self.reset_parameters_()

    def table_parameters(self) -> list[nn.Parameter]:
        """Return high-byte and low-byte detector and conditioning tables."""
        return [self.w_hi_sem, self.w_hi_id, self.c_hi_sem, self.c_hi_id, self.w_lo_sem, self.w_lo_id, self.pair_bias]

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return radial scale scalars."""
        return [self.hi_radial_scale, self.lo_radial_scale]

    @torch.no_grad()
    def reset_parameters_(self, seed: int = 1337, d_out_hi: torch.Tensor | None = None, d_out_lo: torch.Tensor | None = None) -> None:
        """Initialize `197` high-byte and `256` low-byte detectors aligned with `(d_out_hi, d_out_lo)` and Zipf pair bias."""
        gen = torch.Generator(device=self.w_hi_sem.device)
        gen.manual_seed(int(seed) + 4242)
        if d_out_hi is not None and d_out_lo is not None:
            self.w_hi_sem.copy_(F.rms_norm(d_out_hi[:, : self.sem_dim].float(), (self.sem_dim,), eps=1e-6))
            self.w_hi_id.copy_(F.rms_norm(d_out_hi[:, self.sem_dim :].float(), (self.id_dim,), eps=1e-6))
            self.w_lo_sem.copy_(F.rms_norm(d_out_lo[:, : self.sem_dim].float(), (self.sem_dim,), eps=1e-6))
            self.w_lo_id.copy_(F.rms_norm(d_out_lo[:, self.sem_dim :].float(), (self.id_dim,), eps=1e-6))
        else:
            self.w_hi_sem.copy_(
                F.rms_norm(torch.randn(self.w_hi_sem.shape, generator=gen, device=self.w_hi_sem.device), (self.sem_dim,), eps=1e-6)
            )
            self.w_hi_id.copy_(
                F.rms_norm(torch.randn(self.w_hi_id.shape, generator=gen, device=self.w_hi_id.device), (self.id_dim,), eps=1e-6)
            )
            self.w_lo_sem.copy_(
                F.rms_norm(torch.randn(self.w_lo_sem.shape, generator=gen, device=self.w_lo_sem.device), (self.sem_dim,), eps=1e-6)
            )
            self.w_lo_id.copy_(
                F.rms_norm(torch.randn(self.w_lo_id.shape, generator=gen, device=self.w_lo_id.device), (self.id_dim,), eps=1e-6)
            )
        nn.init.normal_(self.c_hi_sem, std=0.15)
        nn.init.normal_(self.c_hi_id, std=0.15)
        self.hi_radial_scale.fill_(2.20)
        self.lo_radial_scale.fill_(2.20)

        log_p_tok, log_p_hi, _ = build_zero_data_bpe_rank_zipf_prior(
            NUM_HIGH_BYTE_BUCKETS * NUM_LOW_BYTE_BUCKETS, device=self.pair_bias.device
        )
        pair_grid = log_p_tok.view(NUM_HIGH_BYTE_BUCKETS, NUM_LOW_BYTE_BUCKETS) - log_p_hi.unsqueeze(-1)
        self.pair_bias.copy_(pair_grid)

    def _compute_log_p_hi(
        self,
        h_sem_0: torch.Tensor,
        h_sem_1: torch.Tensor,
        h_id_branches: torch.Tensor,
        branch_controls: torch.Tensor,
        log_p_hi_prior: torch.Tensor,
    ) -> torch.Tensor:
        """Compute `197`-way high-byte log-probabilities `(N, 197)` across the $R$-mode Cholesky mixture (`<= 8` args)."""
        n_pos, r_m, _ = h_id_branches.shape
        work_dtype = h_sem_0.dtype
        cap, inv_cap = float(DEFAULT_SOFTCAP_LOGIT), 1.0 / float(DEFAULT_SOFTCAP_LOGIT)

        hi_r = self.hi_radial_scale.to(work_dtype)
        w_hi_s = F.rms_norm(self.w_hi_sem.to(work_dtype), (self.sem_dim,), eps=1e-6) * hi_r
        w_hi_i = F.rms_norm(self.w_hi_id.to(work_dtype), (self.id_dim,), eps=1e-6) * hi_r

        z_sem0 = torch.matmul(h_sem_0, w_hi_s.t()).float() + log_p_hi_prior.float().unsqueeze(0)
        z_sem1 = torch.matmul(h_sem_1, w_hi_s.t()).float()
        z_id = torch.matmul(h_id_branches.reshape(n_pos * r_m, self.id_dim), w_hi_i.t()).float().view(n_pos, r_m, NUM_HIGH_BYTE_BUCKETS)

        sem1_g = branch_controls[:, :, 0].float().unsqueeze(-1)
        id_sc = branch_controls[:, :, 1].float().unsqueeze(-1)
        log_pi = F.log_softmax(branch_controls[:, :, 2].float(), dim=-1).unsqueeze(-1)

        raw_r = z_sem0.unsqueeze(1) + sem1_g * z_sem1.unsqueeze(1) + id_sc * z_id
        capped_r = cap * torch.tanh(raw_r * inv_cap)
        log_p_r = F.log_softmax(capped_r, dim=-1)
        return torch.logsumexp(log_pi + log_p_r, dim=1)

    def forward_train_loss(
        self,
        h_sem_0: torch.Tensor,
        h_sem_1: torch.Tensor,
        h_id_branches: torch.Tensor,
        branch_controls: torch.Tensor,
        log_p_hi_prior: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute exact two-byte chain-rule NLL using only `197` + `256` projections per token (`<= 8` args)."""
        y_flat = targets.reshape(-1)
        valid_mask = (y_flat >= 0) & (y_flat < RAW_GPT2_VOCAB_SIZE)
        y_safe = y_flat.clamp(0, RAW_GPT2_VOCAB_SIZE - 1)
        b_hi_y, b_lo_y = decompose_token_uint16_byte_pair(y_safe)

        n_pos, r_m, _ = h_id_branches.shape
        work_dtype = h_sem_0.dtype
        cap, inv_cap = float(DEFAULT_SOFTCAP_LOGIT), 1.0 / float(DEFAULT_SOFTCAP_LOGIT)

        log_p_hi = self._compute_log_p_hi(h_sem_0, h_sem_1, h_id_branches, branch_controls, log_p_hi_prior)
        nll_hi = -log_p_hi.gather(dim=-1, index=b_hi_y.unsqueeze(-1)).squeeze(-1)

        c_sem = (1.0 + 0.5 * torch.tanh(self.c_hi_sem[b_hi_y])).to(work_dtype)
        c_id = (1.0 + 0.5 * torch.tanh(self.c_hi_id[b_hi_y])).to(work_dtype).unsqueeze(1)

        lo_r = self.lo_radial_scale.to(work_dtype)
        w_lo_s = F.rms_norm(self.w_lo_sem.to(work_dtype), (self.sem_dim,), eps=1e-6) * lo_r
        w_lo_i = F.rms_norm(self.w_lo_id.to(work_dtype), (self.id_dim,), eps=1e-6) * lo_r

        z_lo_sem0 = torch.matmul(h_sem_0 * c_sem, w_lo_s.t()).float() + self.pair_bias[b_hi_y].float()
        z_lo_sem1 = torch.matmul(h_sem_1 * c_sem, w_lo_s.t()).float()
        h_id_cond = (h_id_branches * c_id).reshape(n_pos * r_m, self.id_dim)
        z_lo_id = torch.matmul(h_id_cond, w_lo_i.t()).float().view(n_pos, r_m, NUM_LOW_BYTE_BUCKETS)

        sem1_g = branch_controls[:, :, 0].float().unsqueeze(-1)
        id_sc = branch_controls[:, :, 1].float().unsqueeze(-1)
        log_pi = F.log_softmax(branch_controls[:, :, 2].float(), dim=-1).unsqueeze(-1)

        raw_lo_r = z_lo_sem0.unsqueeze(1) + sem1_g * z_lo_sem1.unsqueeze(1) + id_sc * z_lo_id
        capped_lo_r = cap * torch.tanh(raw_lo_r * inv_cap)
        row_mask = self.valid_pair_mask[b_hi_y].unsqueeze(1)
        masked_lo_r = capped_lo_r.masked_fill(~row_mask, -1e9)
        log_p_lo_r = F.log_softmax(masked_lo_r, dim=-1)
        log_p_lo = torch.logsumexp(log_pi + log_p_lo_r, dim=1)
        nll_lo = -log_p_lo.gather(dim=-1, index=b_lo_y.unsqueeze(-1)).squeeze(-1)

        total_nll = nll_hi + nll_lo
        if bool(valid_mask.all()):
            return total_nll.mean()
        return total_nll[valid_mask].mean()

    @torch.no_grad()
    def evaluate_chunk_50257_log_probs(
        self,
        h_sem_0: torch.Tensor,
        h_sem_1: torch.Tensor,
        h_id_branches: torch.Tensor,
        branch_controls: torch.Tensor,
        log_p_hi_prior: torch.Tensor,
    ) -> torch.Tensor:
        """Materialize exact `(c_len, 50257)` log-probabilities $\\ln P(b_{\\text{hi}}) + \\ln P(b_{\\text{lo}} \\mid b_{\\text{hi}})$ (`<= 8` args)."""
        c_len, r_m, _ = h_id_branches.shape
        work_dtype = h_sem_0.dtype
        cap, inv_cap = float(DEFAULT_SOFTCAP_LOGIT), 1.0 / float(DEFAULT_SOFTCAP_LOGIT)

        log_p_hi = self._compute_log_p_hi(h_sem_0, h_sem_1, h_id_branches, branch_controls, log_p_hi_prior)

        lo_r = self.lo_radial_scale.to(work_dtype)
        w_lo_s = F.rms_norm(self.w_lo_sem.to(work_dtype), (self.sem_dim,), eps=1e-6) * lo_r
        w_lo_i = F.rms_norm(self.w_lo_id.to(work_dtype), (self.id_dim,), eps=1e-6) * lo_r
        c_sem_all = (1.0 + 0.5 * torch.tanh(self.c_hi_sem)).to(work_dtype)
        c_id_all = (1.0 + 0.5 * torch.tanh(self.c_hi_id)).to(work_dtype)

        sem1_g = branch_controls[:, :, 0].float().unsqueeze(-1)
        id_sc = branch_controls[:, :, 1].float().unsqueeze(-1)
        log_pi = F.log_softmax(branch_controls[:, :, 2].float(), dim=-1).unsqueeze(-1)

        out_50257 = torch.empty((c_len, RAW_GPT2_VOCAB_SIZE), device=h_sem_0.device, dtype=torch.float32)
        b_step = 49
        for b_start in range(0, NUM_HIGH_BYTE_BUCKETS, b_step):
            b_end = min(b_start + b_step, NUM_HIGH_BYTE_BUCKETS)
            nb = b_end - b_start
            cs_b = c_sem_all[b_start:b_end]
            ci_b = c_id_all[b_start:b_end]

            h_s0_b = (h_sem_0.unsqueeze(1) * cs_b.unsqueeze(0)).reshape(c_len * nb, self.sem_dim)
            h_s1_b = (h_sem_1.unsqueeze(1) * cs_b.unsqueeze(0)).reshape(c_len * nb, self.sem_dim)
            z_s0 = torch.matmul(h_s0_b, w_lo_s.t()).float().view(c_len, nb, NUM_LOW_BYTE_BUCKETS) + self.pair_bias[b_start:b_end].float().unsqueeze(0)
            z_s1 = torch.matmul(h_s1_b, w_lo_s.t()).float().view(c_len, nb, NUM_LOW_BYTE_BUCKETS)

            h_id_b = (h_id_branches.unsqueeze(1) * ci_b.view(1, nb, 1, self.id_dim)).reshape(c_len * nb * r_m, self.id_dim)
            z_id = torch.matmul(h_id_b, w_lo_i.t()).float().view(c_len, nb, r_m, NUM_LOW_BYTE_BUCKETS)

            raw_r = z_s0.unsqueeze(2) + sem1_g.unsqueeze(1) * z_s1.unsqueeze(2) + id_sc.unsqueeze(1) * z_id
            capped_r = cap * torch.tanh(raw_r * inv_cap)
            mask_b = self.valid_pair_mask[b_start:b_end].view(1, nb, 1, NUM_LOW_BYTE_BUCKETS)
            capped_r = capped_r.masked_fill(~mask_b, -1e9)
            log_p_lo_r = F.log_softmax(capped_r, dim=-1)
            log_p_lo = torch.logsumexp(log_pi.unsqueeze(1) + log_p_lo_r, dim=2)
            log_joint = log_p_hi[:, b_start:b_end].unsqueeze(-1) + log_p_lo

            v_start = b_start * NUM_LOW_BYTE_BUCKETS
            v_end = min(b_end * NUM_LOW_BYTE_BUCKETS, RAW_GPT2_VOCAB_SIZE)
            flat_joint = log_joint.reshape(c_len, nb * NUM_LOW_BYTE_BUCKETS)
            out_50257[:, v_start:v_end] = flat_joint[:, : (v_end - v_start)]

        return out_50257


class ZeroPPMIBytePairCholeskyGPT(NoClickVacuumCholeskyGPT):
    r"""Strict Speedrun-Compliant Zero-PPMI Byte-Pair (`b_hi, b_lo`) & Chain-Rule Quantum Cholesky GPT (`Wave 61`)."""

    def __init__(self, zp_spec: ZeroPPMIBytePairCholeskySpec) -> None:
        nc_spec = NoClickVacuumCholeskySpec(
            variant_mode=NoClickVacuumVariantMode.N2_RANK12_NOCLICK_VACUUM_POVM
            if zp_spec.num_cholesky_modes >= 12
            else NoClickVacuumVariantMode.N1_RANK8_NOCLICK_VACUUM_POVM,
            num_cholesky_modes=zp_spec.num_cholesky_modes,
            vacuum_dim=zp_spec.vacuum_dim,
            use_noclick_vacuum_sink=True,
            seed=zp_spec.seed,
        )
        super().__init__(nc_spec)
        self.zp_spec = zp_spec
        self.zp_variant_mode = ZeroPPMIBytePairVariantMode(zp_spec.variant_mode)

        # Disable parent VacuumAwareNGramProjector so build_variable_rank_cholesky_optimizers uses clean byte_ngram groups
        self.sem_ngram = None  # type: ignore[assignment]
        self.online_zipf = OnlineCausalZipfTracker(self.vocab_size, enable_online_ema=zp_spec.enable_online_ema)
        self.byte_ngram = ZeroPPMIBytePairNGramProjector(
            num_buckets=DEFAULT_BYTE_NGRAM_BUCKETS, sem_dim=self.sem_real_dim, id_dim=self.id_real_dim, d_vac=self.d_vac
        )
        self.byte_synth = BytePairSchurWavepacketSynthesizer(
            vocab_size=self.vocab_size,
            pkt_count=self.pkt_count,
            pkt_dim=self.pkt_dim,
            sem_dim=self.sem_real_dim,
            id_dim=self.id_real_dim,
            carrier_dim=DEFAULT_CARRIER_BIAS_DIM,
        )
        self.chain_readout: ExactChainRuleTwoByteReadout | None = (
            ExactChainRuleTwoByteReadout(sem_dim=self.sem_real_dim, id_dim=self.id_real_dim)
            if zp_spec.use_exact_chain_rule_readout
            else None
        )
        self.init_tokens_scanned: int = 0

    def _encode_multiscale_trunk(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]:
        r"""Encode `idx` `(B, T)` with optional online EMA update and `(b_hi, b_lo)` + Schur wavepacket synthesis."""
        if self.training and self.zp_spec.enable_online_ema:
            self.online_zipf.update_from_training_batch_(idx)
            log_p_tok, _ = self.online_zipf.current_log_priors()
            with torch.no_grad():
                n_seen = int(self.online_zipf.batches_seen.item())
                ema_pull = 0.25 / (1.0 + 0.05 * float(n_seen))
                self.unigram_bias.lerp_(log_p_tok, ema_pull)

        bsz, seq_len = idx.shape
        work_dtype = torch.bfloat16 if (idx.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
        tok_wp = F.embedding(idx, self.wte_3d.view(self.vocab_size, self.embed_dim)).view(
            bsz, seq_len, self.pkt_count, self.pkt_dim
        )
        raw_wp = tok_wp.to(work_dtype)
        if self.zp_spec.enable_byte_pair_schur:
            raw_wp = self.byte_synth.lift_input_wavepackets(idx, raw_wp)

        if seq_len > 1:
            raw_wp = self._apply_dispersion(raw_wp)

        raw_flat = raw_wp.reshape(bsz, seq_len, self.embed_dim)
        if seq_len > 1:
            s_gate = self.smear_lambda.to(work_dtype) * torch.sigmoid(
                F.linear(raw_flat[:, 1:, : self.smear_in_ch], self.smear_gate.weight.to(work_dtype))
            )
            sem_curr = raw_flat[:, 1:, : self.sem_real_dim] + s_gate * raw_flat[:, :-1, : self.sem_real_dim]
            sem_full = torch.cat([raw_flat[:, :1, : self.sem_real_dim], sem_curr], dim=1)
            raw_flat = torch.cat([sem_full, raw_flat[:, :, self.sem_real_dim :]], dim=-1)

        readout_sem_448: torch.Tensor | None = None
        readout_id_64: torch.Tensor | None = None
        if self.zp_spec.enable_online_ema or self.zp_spec.enable_byte_pair_schur:
            in_sem_delta, readout_sem_448, readout_id_64 = self.byte_ngram(idx, work_dtype)
            raw_flat = raw_flat + in_sem_delta

        x_0 = F.rms_norm(raw_flat, (self.embed_dim,), eps=1e-6)
        chol_state: tuple[torch.Tensor, torch.Tensor] | None = None
        if self.r_spec.use_layer0_input_cholesky:
            blk0 = self.quantum_blocks[0]
            v0_wp = blk0.v_proj(x_0).view(bsz, seq_len, self.num_heads, 2, self.pkt_dim)  # type: ignore[union-attr]
            if self.training and v0_wp.requires_grad:
                chol_state = torch.utils.checkpoint.checkpoint(
                    build_layer0_variable_rank_cholesky_factor,
                    v0_wp, self.wp_causal_re, self.wp_causal_im, self.layer0_chol_row_scales, self.num_branches,
                    use_reentrant=False,
                )
            else:
                chol_state = build_layer0_variable_rank_cholesky_factor(
                    v0_wp, self.wp_causal_re, self.wp_causal_im, self.layer0_chol_row_scales, self.num_branches
                )

        x = x_0
        cos_4d, sin_4d = self.rope.get_cos_sin(seq_len, dtype=work_dtype)
        for q_blk in self.quantum_blocks:
            x = q_blk(x, x_0, cos_4d, sin_4d, in_cholesky=chol_state)  # type: ignore[call-arg]
            chol_state = q_blk.last_cholesky_pre_u  # type: ignore[assignment]
            q_blk.last_cholesky_pre_u = None  # type: ignore[assignment]

        h_pre = F.rms_norm(x, (self.embed_dim,), eps=1e-6)
        return h_pre, readout_sem_448, readout_id_64, chol_state

    def build_multibranch_readout_tensors(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build readout tensors incorporating online Zipf prior, `(b_hi, b_lo)` Schur detectors, and contextual `b_hi` carrier."""
        h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug = super().build_multibranch_readout_tensors(idx)
        work_dtype = h_base_464.dtype

        if self.zp_spec.enable_byte_pair_schur:
            w_det_synth = self.byte_synth.synthesize_detector_table(self.w_det_512, work_dtype)
            r_scale = self.radial_scale.to(work_dtype).unsqueeze(1)
            w_scaled = w_det_synth * r_scale
            w_aug = torch.cat([w_scaled, w_aug[:, self.embed_dim :]], dim=-1)

        if self.zp_spec.enable_hierarchical_bhi_carrier:
            h_sem_only = h_base_464[:, :, : self.sem_real_dim]
            h_id_0 = h_id_branches[:, :, 0, :]
            h_pre_approx = torch.cat([h_sem_only, h_id_0], dim=-1)
            h_car_16, w_car_16 = self.byte_synth.build_contextual_byte_carrier(h_pre_approx, self.carrier_c0, work_dtype)
            h_base_464 = torch.cat(
                [h_base_464[:, :, : self.sem_real_dim], h_base_464[:, :, self.sem_real_dim :] + h_car_16], dim=-1
            ).contiguous()
            w_aug = torch.cat([w_aug[:, : self.embed_dim], w_aug[:, self.embed_dim :] + w_car_16], dim=-1).contiguous()

        return h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug

    def forward_loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Zero-PPMI Byte-Pair Cholesky cross-entropy loss (either `50,304`-way or Exact Chain-Rule `197 x 256`)."""
        h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug = self.build_multibranch_readout_tensors(idx)
        if self.zp_spec.use_exact_chain_rule_readout and self.chain_readout is not None:
            bsz, seq_len, _ = h_base_464.shape
            n_pos = bsz * seq_len
            h_sem_0 = h_base_464[:, :, : self.sem_real_dim].reshape(n_pos, self.sem_real_dim)
            h_sem_1 = h_sem_branches.reshape(n_pos, self.sem_real_dim)
            h_id_flat = h_id_branches.reshape(n_pos, self.num_branches, self.id_real_dim)
            ctrl_flat = branch_controls.reshape(n_pos, self.num_branches, 3)
            _, log_p_hi_prior = self.online_zipf.current_log_priors()
            return self.chain_readout.forward_train_loss(h_sem_0, h_sem_1, h_id_flat, ctrl_flat, log_p_hi_prior, targets)

        return _VacuumFourMode512DWavepacketCEFn.apply(
            h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug, targets
        )


def build_zero_ppmi_byte_pair_optimizers(
    model: ZeroPPMIBytePairCholeskyGPT,
) -> tuple[
    ModularRiemannianStiefelKrausMuon,
    BatchSubspaceTikhonovGramMuon,
    BatchSubspaceTikhonovGramMuon,
    torch.optim.AdamW,
]:
    """Build Riemannian Stiefel, Vocab Gram-Muon, and AdamW optimizers with byte-pair table groups (`<= 8` args)."""
    q_trunk_opt, wte_opt, head_opt, adam_opt = build_variable_rank_cholesky_optimizers(model)
    table_params: list[nn.Parameter] = list(model.byte_ngram.table_parameters())
    table_params.extend(model.byte_synth.table_parameters())
    scalar_ext_params: list[nn.Parameter] = list(model.byte_ngram.scalar_parameters())
    scalar_ext_params.extend(model.byte_synth.scalar_parameters())
    if model.chain_readout is not None:
        table_params.extend(model.chain_readout.table_parameters())
        scalar_ext_params.extend(model.chain_readout.scalar_parameters())

    adam_opt.add_param_group({"params": table_params, "lr": DEFAULT_SHALLOW_NGRAM_LR, "weight_decay": 0.0})
    adam_opt.add_param_group({"params": scalar_ext_params, "lr": DEFAULT_SHALLOW_SCALAR_LR, "weight_decay": 0.0})
    return q_trunk_opt, wte_opt, head_opt, adam_opt


@torch.no_grad()
def initialize_zero_ppmi_byte_pair_model_(model: ZeroPPMIBytePairCholeskyGPT) -> tuple[float, dict[str, Any]]:
    r"""Initialize `ZeroPPMIBytePairCholeskyGPT` at Step 0 with **zero training tokens scanned** (`< 50 ms`)."""
    device = model.wte_3d.device
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    gen = torch.Generator(device=device)
    gen.manual_seed(int(model.zp_spec.seed))

    model.byte_synth.reset_parameters_(seed=model.zp_spec.seed)
    if model.chain_readout is not None:
        model.chain_readout.reset_parameters_(
            seed=model.zp_spec.seed,
            d_out_hi=model.byte_synth.d_out_hi,
            d_out_lo=model.byte_synth.d_out_lo,
        )

    raw_wte = torch.randn(model.wte_3d.shape, generator=gen, device=device, dtype=torch.float32)
    b_hi_v = model.byte_synth.vocab_b_hi
    b_lo_v = model.byte_synth.vocab_b_lo
    hi_part = model.byte_synth.u_in_hi[b_hi_v]
    lo_part = model.byte_synth.u_in_lo[b_lo_v]
    schur_part = F.rms_norm(hi_part, (model.pkt_dim,), eps=1e-6) * F.rms_norm(lo_part, (model.pkt_dim,), eps=1e-6)

    blended_wte = 0.65 * F.normalize(raw_wte, dim=-1, eps=1e-6) + 0.20 * hi_part + 0.10 * lo_part + 0.05 * schur_part
    norm_wte = F.normalize(blended_wte, dim=-1, eps=1e-6)
    model.wte_3d.copy_(norm_wte)

    log_p_tok, log_p_hi, radial = build_zero_data_bpe_rank_zipf_prior(model.vocab_size, device=device)
    model.unigram_bias.copy_(log_p_tok)
    model.radial_scale.copy_(radial)
    model.prior_gate.fill_(1.0)

    init_disp = torch.tensor([0.22, 0.22, 0.14, 0.08, 0.22, 0.22, 0.14, 0.00], device=device, dtype=torch.float32)
    model.wp_causal_re.copy_(init_disp)
    model.wp_causal_im.zero_()

    for q_blk in model.quantum_blocks:
        q_blk.obs_scale.fill_(0.45)  # type: ignore[union-attr]
        q_blk.mz_mlp.mz_scale.fill_(0.45)  # type: ignore[union-attr]

    wte_flat = model.wte_3d.detach().float().view(model.vocab_size, model.embed_dim)
    p0 = torch.exp(log_p_tok).unsqueeze(1)
    sem_raw, id_raw = wte_flat[:, : model.sem_real_dim], wte_flat[:, model.sem_real_dim :]
    centered_sem = F.normalize(sem_raw - (p0 * sem_raw).sum(dim=0, keepdim=True), dim=-1, eps=1e-6)
    centered_id = F.normalize(id_raw - (p0 * id_raw).sum(dim=0, keepdim=True), dim=-1, eps=1e-6)

    target_sem_rms = centered_sem * math.sqrt(float(model.sem_real_dim))
    target_id_rms = centered_id * math.sqrt(float(model.id_real_dim))
    model.w_det_512.copy_(torch.cat([target_sem_rms, target_id_rms], dim=-1))

    eye_d = torch.eye(model.embed_dim, device=device, dtype=torch.float32)
    model.exit_bridge.u_stiefel.copy_(eye_d)
    model.exit_bridge.v_stiefel.copy_(eye_d)
    model.exit_bridge.sigma.fill_(1.0)

    model.temp_sem_scale.fill_(0.25)
    model.temp_id_scale.fill_(0.25)
    model.trunk_id_gate.fill_(0.50)
    model.init_tokens_scanned = 0
    model.clamp_identity_dispersion_and_scalars_()

    if device.type == "cuda":
        torch.cuda.synchronize()
    init_ms = (time.perf_counter() - t0) * 1000.0
    return init_ms, {
        "init_tokens_scanned": 0,
        "init_ms": round(init_ms, 3),
        "num_high_byte_buckets": NUM_HIGH_BYTE_BUCKETS,
        "num_low_byte_buckets": NUM_LOW_BYTE_BUCKETS,
        "step0_analytical_hi_entropy_nats": round(float(-(torch.exp(log_p_hi) * log_p_hi).sum().item()), 4),
    }


@torch.no_grad()
def evaluate_zero_ppmi_byte_pair_model(
    model: ZeroPPMIBytePairCholeskyGPT, stream: LongSeqFineWebStream
) -> dict[str, float]:
    """Evaluate `ZeroPPMIBytePairCholeskyGPT` on unseen Shard-0 validation batches (`<= 8` args)."""
    if not (model.zp_spec.use_exact_chain_rule_readout and model.chain_readout is not None):
        ev = evaluate_variable_rank_cholesky_model(model, stream)
        ev.update(model.get_noclick_telemetry())
        ev["online_batches_seen"] = float(model.online_zipf.batches_seen.item())
        return ev

    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_top1 = 0
    total_top5 = 0
    total_tokens = 0
    gate_ent_sum = 0.0
    n_batches = 0
    anc_cos_sum, exc_cos_sum, core_cos_sum, high_cos_sum = 0.0, 0.0, 0.0, 0.0

    _, log_p_hi_prior = model.online_zipf.current_log_priors()
    for idx_in, targets in stream.val_batches:
        bsz, seq_len = idx_in.shape
        h_base_464, h_sem_branches, h_id_branches, branch_controls, _ = model.build_multibranch_readout_tensors(idx_in)
        gate_logits = branch_controls[:, :, :, 2].float()
        pi = F.softmax(gate_logits, dim=-1)
        log_pi = F.log_softmax(gate_logits, dim=-1)
        gate_ent_sum += float(-(pi * log_pi).sum(dim=-1).mean().item())
        anc_cos_sum += float(getattr(model, "last_anchor_id_cos_vac", 1.0))
        exc_cos_sum += float(getattr(model, "last_excited_id_cos_vac", 1.0))
        core_cos_sum += float(getattr(model, "last_core_excited_cos_vac", 1.0))
        high_cos_sum += float(getattr(model, "last_high_excited_cos_vac", 1.0))
        n_batches += 1

        t_step = 16
        for t_start in range(0, seq_len, t_step):
            t_end = min(t_start + t_step, seq_len)
            c_len = bsz * (t_end - t_start)
            h_s0_c = h_base_464[:, t_start:t_end, : model.sem_real_dim].reshape(c_len, model.sem_real_dim)
            h_s1_c = h_sem_branches[:, t_start:t_end].reshape(c_len, model.sem_real_dim)
            h_id_c = h_id_branches[:, t_start:t_end].reshape(c_len, model.num_branches, model.id_real_dim)
            ctrl_c = branch_controls[:, t_start:t_end].reshape(c_len, model.num_branches, 3)
            y_c = targets[:, t_start:t_end].reshape(c_len)

            log_p_50257 = model.chain_readout.evaluate_chunk_50257_log_probs(
                h_s0_c, h_s1_c, h_id_c, ctrl_c, log_p_hi_prior
            )
            nll_c = -log_p_50257.gather(dim=-1, index=y_c.unsqueeze(-1)).squeeze(-1)
            total_nll += float(nll_c.sum().item())

            top5_idx = torch.topk(log_p_50257, k=5, dim=-1).indices
            total_top1 += int((top5_idx[:, 0] == y_c).sum().item())
            total_top5 += int((top5_idx == y_c.unsqueeze(-1)).any(dim=-1).sum().item())
            total_tokens += c_len
            del log_p_50257
        del h_base_464, h_sem_branches, h_id_branches, branch_controls

    if stream.device.type == "cuda":
        torch.cuda.empty_cache()
    if was_training:
        model.train()

    denom = float(max(1, total_tokens))
    nb = float(max(1, n_batches))
    val_ce = total_nll / denom
    mean_ent = gate_ent_sum / nb
    ev = {
        "val_ce": round(val_ce, 4),
        "neural_val_ce": round(val_ce, 4),
        "val_ppl": round(math.exp(min(20.0, val_ce)), 2),
        "top1_acc": round(total_top1 / denom, 4),
        "top5_acc": round(total_top5 / denom, 4),
        "branch_gate_entropy_nats": round(mean_ent, 4),
        "effective_active_branches": round(math.exp(mean_ent), 2),
        "mean_anchor_id_cos_theta_vac": round(anc_cos_sum / nb, 4),
        "mean_excited_id_cos_theta_vac": round(exc_cos_sum / nb, 4),
        "mean_core_excited_cos_theta_vac": round(core_cos_sum / nb, 4),
        "mean_high_excited_cos_theta_vac": round(high_cos_sum / nb, 4),
        "online_batches_seen": float(model.online_zipf.batches_seen.item()),
    }
    ev.update(model.get_noclick_telemetry())
    return ev


def audit_step1_vocabulary_detector_coverage(
    idx_in: torch.Tensor, targets: torch.Tensor, seed: int = 1337
) -> dict[str, float]:
    """Compare Step-1 vocabulary wavepacket & detector update coverage (`%` of `50,257` tokens changed) between `Z0` and `Z2`."""
    device = idx_in.device
    all_v = torch.arange(RAW_GPT2_VOCAB_SIZE, device=device, dtype=torch.long).unsqueeze(0)
    results: dict[str, float] = {}
    for label, enable_bp in [("z0_unfactored", False), ("z2_byte_pair_schur", True)]:
        spec = ZeroPPMIBytePairCholeskySpec(
            variant_mode=ZeroPPMIBytePairVariantMode.Z2_ZERO_PPMI_FACTORED_BYTE_PAIR_SCHUR
            if enable_bp else ZeroPPMIBytePairVariantMode.Z0_ZERO_PPMI_COLD_START_BASELINE,
            num_cholesky_modes=4, enable_online_ema=enable_bp, enable_byte_pair_schur=enable_bp, seed=seed,
        )
        model = ZeroPPMIBytePairCholeskyGPT(spec).to(device=device)
        initialize_zero_ppmi_byte_pair_model_(model)
        optimizers = build_zero_ppmi_byte_pair_optimizers(model)

        with torch.no_grad():
            base_b = model.wte_3d[:RAW_GPT2_VOCAB_SIZE].unsqueeze(0).float()
            wp_before = (model.byte_synth.lift_input_wavepackets(all_v, base_b) if enable_bp else base_b).reshape(RAW_GPT2_VOCAB_SIZE, -1).clone()

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss = model.forward_loss(idx_in, targets)
        loss.backward()
        _apply_global_optimizer_step(model, optimizers, idx_in, targets, lr_mult=1.0, mom_beta=0.70)

        with torch.no_grad():
            base_a = model.wte_3d[:RAW_GPT2_VOCAB_SIZE].unsqueeze(0).float()
            wp_after = (model.byte_synth.lift_input_wavepackets(all_v, base_a) if enable_bp else base_a).reshape(RAW_GPT2_VOCAB_SIZE, -1)
            row_delta = (wp_after - wp_before).norm(dim=-1)
            results[f"{label}_updated_vocab_pct"] = round(100.0 * float((row_delta > 1e-6).float().mean().item()), 2)
        del model, optimizers
    return results


def _compute_zero_ppmi_step_lr_and_momentum(step: int, total_steps: int) -> tuple[float, float]:
    """Compute cosine LR multiplier and Riemannian momentum beta for `step` in `1..total_steps`."""
    warm_steps = max(1, int(0.08 * total_steps))
    lr_mult = (
        0.20 + 0.80 * (float(step - 1) / float(max(1, warm_steps - 1)))
        if step <= warm_steps
        else 0.12 + 0.88 * 0.5 * (1.0 + math.cos(math.pi * float(step - warm_steps) / float(max(1, total_steps - warm_steps))))
    )
    mom_beta = compute_riemannian_momentum_beta(step, total_steps, QuantumMomentumScheduleMode.LOW_STEADY_MOMENTUM_070)
    return lr_mult, mom_beta


def _train_and_eval_zero_ppmi_variants(
    stream: LongSeqFineWebStream,
    variants: list[tuple[str, ZeroPPMIBytePairCholeskySpec, str]],
    steps: int,
    eval_interval: int,
    wave_id: str = "W61",
) -> list[dict[str, Any]]:
    """Train and evaluate Zero-PPMI Byte-Pair Cholesky variants (`<= 8` args)."""
    device = stream.device
    results: list[dict[str, Any]] = []
    tokens_per_step = stream.batch_size * stream.seq_len

    for variant_id, zp_spec, desc in variants:
        torch.manual_seed(zp_spec.seed)
        model = ZeroPPMIBytePairCholeskyGPT(zp_spec).to(device=device)
        init_ms, init_meta = initialize_zero_ppmi_byte_pair_model_(model)

        run_key = f"plan081_{variant_id.lower()}_{steps}steps"
        intent = ExperimentIntentSpec(
            run_key=run_key,
            wave_id=wave_id,
            plan_id="EXPERIMENT_PLAN_081",
            task_domain="fineweb_speedrun",
            hypothesis_intent=desc,
            architecture_summary=(
                f"ZeroPPMIBytePairCholeskyGPT(L=4, d=512, R={zp_spec.num_cholesky_modes}, "
                f"bp={zp_spec.enable_byte_pair_schur}, chain={zp_spec.use_exact_chain_rule_readout}, init_tok=0)"
            ),
            hardware_budget="1x RTX 5060 Ti 16GB",
            batch_size=stream.batch_size,
            seq_len=stream.seq_len,
            total_steps=steps,
            target_tokens_M=round((steps * tokens_per_step) / 1e6, 3),
        )
        exp_logger = StandardizedExperimentLogger(intent)
        optimizers = build_zero_ppmi_byte_pair_optimizers(model)

        step0_eval = evaluate_zero_ppmi_byte_pair_model(model, stream)
        step0_eval["step"], step0_eval["tokens_M"] = 0, 0.0
        exp_logger.log_eval(
            0, {**step0_eval, "top1_acc": round(100.0 * step0_eval["top1_acc"], 2), "top5_acc": round(100.0 * step0_eval["top5_acc"], 2)}
        )
        print(
            f"[{variant_id} (R={zp_spec.num_cholesky_modes}, bp={zp_spec.enable_byte_pair_schur}, "
            f"chain={zp_spec.use_exact_chain_rule_readout}, init={init_ms:.1f}ms, scan=0)] "
            f"Step    0 | Zero-PPMI Val CE: {step0_eval['val_ce']:.4f} (PPL {step0_eval['val_ppl']:.2f}) | "
            f"Top1: {100.0 * step0_eval['top1_acc']:.2f}% | Modes: {step0_eval['effective_active_branches']:.2f}/{zp_spec.num_cholesky_modes}",
            flush=True,
        )

        curve: list[dict[str, Any]] = [step0_eval]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        model.train()

        for step in range(1, steps + 1):
            idx_in, targets = stream.sample_train_batch(step)
            lr_mult, mom_beta = _compute_zero_ppmi_step_lr_and_momentum(step, steps)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            loss = model.forward_loss(idx_in, targets)
            train_loss_val = float(loss.detach().item())
            loss.backward()
            del loss
            _apply_global_optimizer_step(model, optimizers, idx_in, targets, lr_mult=lr_mult, mom_beta=mom_beta)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            if step % eval_interval == 0 or step == steps:
                ev = evaluate_zero_ppmi_byte_pair_model(model, stream)
                ev["step"] = step
                ev["tokens_M"] = round((step * tokens_per_step) / 1e6, 3)
                ev["train_loss"] = round(train_loss_val, 4)
                ev["lr_mult"] = round(lr_mult, 4)
                curve.append(ev)
                exp_logger.log_eval(
                    step, {**ev, "top1_acc": round(100.0 * ev["top1_acc"], 2), "top5_acc": round(100.0 * ev["top5_acc"], 2)}
                )
                print(
                    f"[{variant_id} (R={zp_spec.num_cholesky_modes}, bp={zp_spec.enable_byte_pair_schur}, chain={zp_spec.use_exact_chain_rule_readout})] "
                    f"Step {step:4d} ({ev['tokens_M']:5.2f}M tok, lr={lr_mult:.2f}x) | "
                    f"Train CE: {ev['train_loss']:.4f} | Zero-PPMI Val CE: {ev['val_ce']:.4f} (PPL {ev['val_ppl']:.2f}) | "
                    f"Top1: {100.0 * ev['top1_acc']:.2f}% | Top5: {100.0 * ev['top5_acc']:.2f}% | "
                    f"Modes: {ev['effective_active_branches']:.2f}/{zp_spec.num_cholesky_modes} | "
                    f"p_noclick: {ev['mean_noclick_prob']:.3f}",
                    flush=True,
                )

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - t_start
        results.append({
            "variant_id": variant_id, "variant_mode": zp_spec.variant_mode.value, "description": desc,
            "num_cholesky_modes": zp_spec.num_cholesky_modes, "enable_online_ema": zp_spec.enable_online_ema,
            "enable_byte_pair_schur": zp_spec.enable_byte_pair_schur, "use_exact_chain_rule_readout": zp_spec.use_exact_chain_rule_readout,
            "enable_hierarchical_bhi_carrier": zp_spec.enable_hierarchical_bhi_carrier,
            "init_tokens_scanned": init_meta["init_tokens_scanned"], "init_ms": round(init_ms, 2),
            "elapsed_sec": round(elapsed_s, 2), "step0": step0_eval, "final": curve[-1], "curve": curve,
        })
        del model, optimizers
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def run_zero_ppmi_byte_pair_ablation_benchmark(
    stream: LongSeqFineWebStream, steps: int = 300, eval_interval: int = 100
) -> list[dict[str, Any]]:
    """Run the 300-step (`2.46M`-token) Zero-PPMI Byte-Pair Cholesky ablation suite (`Z0..Z4`, `<= 8` args)."""
    variants = [
        (
            "Z0_ZERO_PPMI_COLD_START_BASELINE",
            ZeroPPMIBytePairCholeskySpec(
                variant_mode=ZeroPPMIBytePairVariantMode.Z0_ZERO_PPMI_COLD_START_BASELINE,
                num_cholesky_modes=8, enable_online_ema=False, enable_byte_pair_schur=False, use_exact_chain_rule_readout=False,
            ),
            "R=8 Zero-PPMI Cold-Start Baseline (Static Analytical Zipf Prior, Unfactored 50,304 Table, 0 Tokens Scanned)",
        ),
        (
            "Z1_ZERO_PPMI_ONLINE_EMA_ZIPF",
            ZeroPPMIBytePairCholeskySpec(
                variant_mode=ZeroPPMIBytePairVariantMode.Z1_ZERO_PPMI_ONLINE_EMA_ZIPF,
                num_cholesky_modes=8, enable_online_ema=True, enable_byte_pair_schur=False, use_exact_chain_rule_readout=False,
            ),
            "R=8 Zero-PPMI + Online Training-Stream EMA (b_hi, token) Prior + Zero-Init Byte-Pair Causal N-Gram",
        ),
        (
            "Z2_ZERO_PPMI_FACTORED_BYTE_PAIR_SCHUR",
            ZeroPPMIBytePairCholeskySpec(
                variant_mode=ZeroPPMIBytePairVariantMode.Z2_ZERO_PPMI_FACTORED_BYTE_PAIR_SCHUR,
                num_cholesky_modes=8, enable_online_ema=True, enable_byte_pair_schur=True, use_exact_chain_rule_readout=False,
            ),
            "R=8 Zero-PPMI + Deterministic (b_hi, b_lo) + Schur Coincidence Wavepackets & Detectors (100% Step-1 Coverage)",
        ),
        (
            "Z3_ZERO_PPMI_EXACT_CHAIN_RULE_TWOBYTE",
            ZeroPPMIBytePairCholeskySpec(
                variant_mode=ZeroPPMIBytePairVariantMode.Z3_ZERO_PPMI_EXACT_CHAIN_RULE_TWOBYTE,
                num_cholesky_modes=8, enable_online_ema=True, enable_byte_pair_schur=True, use_exact_chain_rule_readout=True,
            ),
            "R=8 Zero-PPMI + Exact 197x256 Chain-Rule Two-Byte Autoregressive Readout (453 Outcomes, Zero 50,304 Matmul)",
        ),
        (
            "Z4_ZERO_PPMI_HIERARCHICAL_BYTE_PAIR_R12",
            ZeroPPMIBytePairCholeskySpec(
                variant_mode=ZeroPPMIBytePairVariantMode.Z4_ZERO_PPMI_HIERARCHICAL_BYTE_PAIR_R12,
                num_cholesky_modes=12, enable_online_ema=True, enable_byte_pair_schur=True,
                use_exact_chain_rule_readout=False, enable_hierarchical_bhi_carrier=True,
            ),
            "R=12 Zero-PPMI + Factored (b_hi, b_lo) Schur + 15D Contextual Byte-Pair Carrier + Trunk No-Click POVM",
        ),
    ]
    return _train_and_eval_zero_ppmi_variants(stream, variants, steps=steps, eval_interval=eval_interval)


def run_zero_ppmi_byte_pair_scaling_benchmark(
    stream: LongSeqFineWebStream, steps: int = 1200, eval_interval: int = 150
) -> list[dict[str, Any]]:
    """Run the 1,200-step (`9.83M`-token) Strict Speedrun-Compliant Zero-PPMI Byte-Pair Scaling Run (`Z5`, `<= 8` args)."""
    variants = [
        (
            "Z5_ZERO_PPMI_BYTE_PAIR_SCALING_1200",
            ZeroPPMIBytePairCholeskySpec(
                variant_mode=ZeroPPMIBytePairVariantMode.Z5_ZERO_PPMI_BYTE_PAIR_SCALING_1200,
                num_cholesky_modes=12, enable_online_ema=True, enable_byte_pair_schur=True,
                use_exact_chain_rule_readout=False, enable_hierarchical_bhi_carrier=True,
            ),
            "1,200-Step (9.83M-Token) Strict Speedrun-Compliant Zero-PPMI Byte-Pair + R=12 No-Click Vacuum Cholesky Scaling",
        ),
    ]
    return _train_and_eval_zero_ppmi_variants(stream, variants, steps=steps, eval_interval=eval_interval)
