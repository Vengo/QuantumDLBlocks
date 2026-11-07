r"""Small-Scale Local Laboratory for Initialization & Probability Readout Strategies on FineWeb Span Reversal.

Provides a fast (`~1.5–2.5s` per 300-step run on a single local GPU) controlled benchmark comparing
**Small Classical** (`MicroClassicalBackbone`) vs. **Small Quantum Wavepacket** (`MicroQuantumWavepacketBackbone`)
models on **FineWeb Span Reversal** (`[t_1, ..., t_L, <REV>, t_L, ..., t_1]`):

1. **Millisecond GPU Initialization Strategies (`MicroInitMode`)**:
   - `RANDOM_SPHERE`: Random unit-RMS sphere embeddings $\sqrt{d_h}\,\mathbb{S}^{d_h-1}$ and zero readout bias.
   - `GPU_UNIGRAM_PRIOR`: Sub-millisecond GPU `torch.bincount` unigram log-prior $b_v = \ln p(v) - \overline{\ln p}$
     and Zipfian radial norm initialization $r_v^{(0)}$.
   - `GPU_COUNTSKETCH_BIGRAM_SPHERE`: SVD-free GPU 2-hash Rademacher CountSketch (`~2–4 ms` on GPU) of forward +
     reverse bigram PPMI statistics with orthogonal packet balancing $Q_M \in O(M_{\text{pkt}})$ onto $(\sqrt{d_h}\,\mathbb{S}^{d_h-1})^{M_{\text{pkt}}}$.
   - `GPU_SVD_BIGRAM_SPHERE`: GPU anchor bigram PPMI + `torch.svd_lowrank` (`~25–50 ms` on GPU) + orthogonal packet balancing.

2. **Probability Readout Strategies (`MicroReadoutMode`)**:
   - `R1_LINEAR_SOFTCAP`: Standard unconstrained Euclidean `lm_head` with `15 * tanh(z / 15)` softcapping.
   - `R2_LEVEL_A_RADIAL_SPHERE`: Untied Wavepacket Product-Sphere detector table $w_v = r_v u_v$ with unit-RMS sphere
     directions $u_v \in (\sqrt{d_h}\,\mathbb{S}^{d_h-1})^{M_{\text{pkt}}}$, learnable per-token radial norms $r_v$, and Parseval $\mu\text{P}$ scale $\beta_0 = c^*\sqrt{d}$.
   - `R3_TIED_WAVEPACKET_EXIT_BRIDGE`: Parameter-efficient tied readout ($u_v = \text{wte}[v]$) coupled through a sequence-side
     Biorthogonal SVD-Stiefel exit operator $W_{\text{exit}}(h_t) = ((h_t V_{\text{exit}})\odot \vec{\sigma}_{\text{exit}})U_{\text{exit}}^\top$.
   - `R4_HERMITIAN_QUADRATIC_BORN`: Tied wavepacket exit state with phase-invariant complex Hermitian quadratic Born
     boost $\widehat{\mathcal{O}}_{\text{born}}(t,v) = \hat{c}_{\text{re}}(t,v) + \gamma_{\text{born}}\big(\max(\hat{c}_{\text{re}}(t,v), 0)^2 + \hat{c}_{\text{im}}(t,v)^2 - \tau_t\big)$.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.fused_quantum_wavepacket_attn import (
    associative_mcweeny_unitary_ket_modulation,
    fused_quantum_wavepacket_rho_ket_attention,
)
from ..layers.impure_density_mirror_block import (
    DEFAULT_TIKHONOV_GRAM_LAMBDA,
    TRACE_NORM_EPS,
)
from ..layers.quantum_statmech_channels import (
    QuantumMomentumScheduleMode,
    StiefelMomentumMode,
    compute_riemannian_momentum_beta,
)
from ..layers.quantum_svd_stiefel_graft import (
    BiorthogonalSVDStiefelProjection,
    QuantumStiefelProjMode,
    apply_fubini_study_self_ortho,
)
from ..layers.wavepacket_statmech_block import apply_causal_wavepacket_dispersion
from ..models.compiled_leaderboard_classical_gpt import (
    NativeLeaderboardRotary,
    apply_native_leaderboard_rotary,
)
from ..models.modular_quantum_statmech_gpt import ModularRiemannianStiefelKrausMuon
from ..models.quantum_full_backbone_graft_gpt import SVDStiefelMachZehnderInterferometer
from ..optim.muon import StiefelMuon, StiefelMuonConfig
from ..optim.vocab_manifold_muon import (
    BatchSubspaceTikhonovGramMuon,
    project_rows_to_rms_sphere_,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
DEFAULT_FINEWEB_DIR: Path = _REPO_ROOT / ".cache" / "fineweb_100M"
DEFAULT_TRAIN_SHARD_PATH: Path = DEFAULT_FINEWEB_DIR / "fineweb_train_000001.bin"
DEFAULT_VAL_SHARD_PATH: Path = DEFAULT_FINEWEB_DIR / "fineweb_val_000000.bin"

FINEWEB_HEADER_INT32: int = 256
RAW_GPT2_VOCAB_SIZE: int = 50257
DEFAULT_MICRO_VOCAB_SIZE: int = 4096
DEFAULT_SPAN_LEN: int = 31
DEFAULT_MICRO_BATCH_SIZE: int = 64
DEFAULT_MICRO_EMBED_DIM: int = 256
DEFAULT_MICRO_NUM_HEADS: int = 4
DEFAULT_MICRO_NUM_LAYERS: int = 2
DEFAULT_MICRO_MLP_HIDDEN: int = 768
DEFAULT_MICRO_PKT_COUNT: int = 8
DEFAULT_MICRO_PKT_DIM: int = 32

DEFAULT_PARSEVAL_MUP_C_STAR: float = 1.48
DEFAULT_READOUT_SOFTCAP: float = 15.0
DEFAULT_RADIAL_MIN: float = 0.40
DEFAULT_RADIAL_MAX: float = 2.50
DEFAULT_BORN_GAMMA_INIT: float = 0.85
DEFAULT_BORN_GAMMA_MAX: float = 3.00
DEFAULT_UNIGRAM_PRIOR_SCALE_INIT: float = 1.00
DEFAULT_UNIGRAM_RADIAL_EXPONENT: float = 0.12

DEFAULT_COUNTSKETCH_TOKENS: int = 2_000_000
DEFAULT_SVD_ANCHOR_COLS: int = 512
DEFAULT_HASH_PRIME_1: int = 2654435761
DEFAULT_HASH_PRIME_2: int = 2246822519
DEFAULT_ORTHO_BALANCE_SEED: int = 20261102

DEFAULT_MICRO_WTE_LR: float = 0.045
DEFAULT_MICRO_STIEFEL_LR: float = 0.028
DEFAULT_MICRO_UNITARY_LR: float = 0.028
DEFAULT_MICRO_SCALAR_LR: float = 0.015
DEFAULT_MICRO_CLASSICAL_ADAM_LR: float = 0.006
DEFAULT_MICRO_CLASSICAL_MUON_LR: float = 0.028


class MicroBackboneMode(str, Enum):
    """Strategy enum selecting Classical vs. Quantum Wavepacket backbone (Rule 3)."""

    CLASSICAL = "classical"
    QUANTUM_WAVEPACKET = "quantum_wavepacket"


class MicroInitMode(str, Enum):
    """Strategy enum selecting GPU initialization strategy (Rule 3)."""

    RANDOM_SPHERE = "random_sphere"
    GPU_UNIGRAM_PRIOR = "gpu_unigram_prior"
    GPU_COUNTSKETCH_BIGRAM_SPHERE = "gpu_countsketch_bigram_sphere"
    GPU_SVD_BIGRAM_SPHERE = "gpu_svd_bigram_sphere"


class MicroReadoutMode(str, Enum):
    """Strategy enum selecting vocabulary probability readout architecture (Rule 3)."""

    R1_LINEAR_SOFTCAP = "r1_linear_softcap"
    R2_LEVEL_A_RADIAL_SPHERE = "r2_level_a_radial_sphere"
    R3_TIED_WAVEPACKET_EXIT_BRIDGE = "r3_tied_wavepacket_exit_bridge"
    R4_HERMITIAN_QUADRATIC_BORN = "r4_hermitian_quadratic_born"


@dataclass(frozen=True)
class MicroReversalConfig:
    """Configuration for the FineWeb Span Reversal initialization & readout laboratory (Rule 2)."""

    backbone_mode: MicroBackboneMode = MicroBackboneMode.QUANTUM_WAVEPACKET
    init_mode: MicroInitMode = MicroInitMode.GPU_COUNTSKETCH_BIGRAM_SPHERE
    readout_mode: MicroReadoutMode = MicroReadoutMode.R3_TIED_WAVEPACKET_EXIT_BRIDGE
    vocab_size: int = DEFAULT_MICRO_VOCAB_SIZE
    span_len: int = DEFAULT_SPAN_LEN
    embed_dim: int = DEFAULT_MICRO_EMBED_DIM
    num_heads: int = DEFAULT_MICRO_NUM_HEADS
    num_layers: int = DEFAULT_MICRO_NUM_LAYERS
    mlp_hidden: int = DEFAULT_MICRO_MLP_HIDDEN
    pkt_count: int = DEFAULT_MICRO_PKT_COUNT
    pkt_dim: int = DEFAULT_MICRO_PKT_DIM
    seed: int = 1337


def load_fineweb_token_slice(shard_path: Path, max_tokens: int) -> torch.Tensor:
    """Load `max_tokens` uint16 tokens from a FineWeb `.bin` shard as a 1D `int64` CPU tensor."""
    if not shard_path.exists():
        gen = torch.Generator(device="cpu").manual_seed(42)
        ranks = torch.arange(1, RAW_GPT2_VOCAB_SIZE + 1, dtype=torch.float32)
        probs = 1.0 / ranks.pow(1.05)
        probs = probs / probs.sum()
        return torch.multinomial(probs, num_samples=max_tokens, replacement=True, generator=gen).long()
    raw = np.memmap(shard_path, dtype=np.uint16, mode="r", offset=FINEWEB_HEADER_INT32 * 4)
    n_read = min(int(max_tokens), int(raw.shape[0]))
    arr = np.array(raw[:n_read], dtype=np.int64)
    return torch.from_numpy(arr)


class FineWebSpanReversalDataset:
    r"""Constructs FineWeb span reversal batches `[t_1, ..., t_L, <REV>, t_L, ..., t_1]` on GPU.

    Maps the top `vocab_size - 2` most frequent FineWeb tokens to `0 .. vocab_size - 3`, rare tokens to
    `<UNK> = vocab_size - 2`, and the reversal delimiter `<REV> = vocab_size - 1`.
    """

    def __init__(
        self,
        vocab_size: int = DEFAULT_MICRO_VOCAB_SIZE,
        span_len: int = DEFAULT_SPAN_LEN,
        train_tokens_limit: int = DEFAULT_COUNTSKETCH_TOKENS,
        val_tokens_limit: int = 262_144,
        device: str | torch.device = "cuda",
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.span_len = int(span_len)
        self.unk_token_id = self.vocab_size - 2
        self.rev_token_id = self.vocab_size - 1
        self.seq_len = 2 * self.span_len
        self.device = torch.device(device)

        raw_train = load_fineweb_token_slice(DEFAULT_TRAIN_SHARD_PATH, train_tokens_limit).to(self.device)
        raw_val = load_fineweb_token_slice(DEFAULT_VAL_SHARD_PATH, val_tokens_limit).to(self.device)

        if self.vocab_size < RAW_GPT2_VOCAB_SIZE:
            counts = torch.bincount(raw_train.clamp(0, RAW_GPT2_VOCAB_SIZE - 1), minlength=RAW_GPT2_VOCAB_SIZE)
            top_ids = torch.topk(counts, k=self.vocab_size - 2, largest=True, sorted=True).indices
            remap_lut = torch.full((RAW_GPT2_VOCAB_SIZE + 1,), self.unk_token_id, dtype=torch.long, device=self.device)
            remap_lut[top_ids] = torch.arange(self.vocab_size - 2, dtype=torch.long, device=self.device)
            self.train_tokens = remap_lut[raw_train.clamp(0, RAW_GPT2_VOCAB_SIZE)]
            self.val_tokens = remap_lut[raw_val.clamp(0, RAW_GPT2_VOCAB_SIZE)]
        else:
            self.train_tokens = raw_train.clamp(0, self.vocab_size - 2)
            self.val_tokens = raw_val.clamp(0, self.vocab_size - 2)

        self.val_batches = self._build_fixed_eval_batches(num_batches=8, batch_size=DEFAULT_MICRO_BATCH_SIZE)

    def _build_reversal_batch_from_starts(self, source_tokens: torch.Tensor, starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice contiguous spans of length `L` and build `(idx_in, targets_rev)` of shapes `(B, 2L)` and `(B, L)` with zero target leakage."""
        offsets = torch.arange(self.span_len, device=self.device).unsqueeze(0)
        spans = source_tokens[starts.unsqueeze(1) + offsets]
        rev_spans = torch.flip(spans, dims=[1])
        rev_queries = torch.full((starts.size(0), self.span_len), self.rev_token_id, dtype=torch.long, device=self.device)
        idx_in = torch.cat([spans, rev_queries], dim=1).contiguous()
        return idx_in, rev_spans.contiguous()

    def _build_fixed_eval_batches(
        self, num_batches: int = 8, batch_size: int = DEFAULT_MICRO_BATCH_SIZE
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Build deterministic validation batches from `val_tokens`."""
        batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        max_start = self.val_tokens.numel() - self.span_len - 1
        stride = max(1, max_start // max(1, num_batches * batch_size))
        for b_idx in range(num_batches):
            base = b_idx * batch_size * stride
            starts = (base + torch.arange(batch_size, device=self.device) * stride) % max_start
            batches.append(self._build_reversal_batch_from_starts(self.val_tokens, starts))
        return batches

    def sample_train_batch(self, step: int, batch_size: int = DEFAULT_MICRO_BATCH_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a deterministic training batch of FineWeb span reversals for `step`."""
        max_start = self.train_tokens.numel() - self.span_len - 1
        offsets = (step * batch_size * self.span_len + torch.arange(batch_size, device=self.device) * 997) % max_start
        return self._build_reversal_batch_from_starts(self.train_tokens, offsets)


# ==============================================================================
# Millisecond GPU Unigram, CountSketch Bigram, and Low-Rank SVD Initializers
# ==============================================================================
@torch.no_grad()
def compute_gpu_unigram_log_prior_and_radial(
    train_tokens: torch.Tensor,
    vocab_size: int,
    rev_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute centered log-unigram bias `b_v` and unit-RMS Zipfian radial scale `r_v` on GPU in `< 1 ms`."""
    counts = torch.bincount(train_tokens, minlength=vocab_size).float()
    counts[rev_token_id] = max(1.0, float(train_tokens.numel()) / float(max(1, DEFAULT_SPAN_LEN)))
    probs = (counts + 1.0) / (counts.sum() + float(vocab_size))
    log_p = torch.log(probs)
    log_prior_centered = log_p - log_p.mean()

    shifted = (log_p - log_p.min() + 1.0).pow(DEFAULT_UNIGRAM_RADIAL_EXPONENT)
    radial = shifted / shifted.square().mean().sqrt().clamp_min(TRACE_NORM_EPS)
    radial = radial.clamp(DEFAULT_RADIAL_MIN, DEFAULT_RADIAL_MAX)
    return log_prior_centered, radial


@torch.no_grad()
def compute_gpu_countsketch_bigram_sphere_embeddings(
    train_tokens: torch.Tensor,
    vocab_size: int,
    pkt_count: int,
    pkt_dim: int,
) -> torch.Tensor:
    r"""Compute `(V, pkt_count, pkt_dim)` unit-RMS sphere embeddings via GPU 2-hash Rademacher CountSketch (`~2–4 ms`)."""
    d_emb = pkt_count * pkt_dim
    device = train_tokens.device
    u_tok = train_tokens[:-1]
    v_tok = train_tokens[1:]
    n_pairs = float(u_tok.numel())

    v_ids = torch.arange(vocab_size, device=device, dtype=torch.int64)
    col_hash_fwd = ((v_ids * DEFAULT_HASH_PRIME_1 + 17) % (d_emb // 2)).long()
    col_hash_rev = (d_emb // 2) + ((v_ids * DEFAULT_HASH_PRIME_2 + 31) % (d_emb // 2)).long()
    sign_fwd = torch.where(((v_ids * DEFAULT_HASH_PRIME_2) & 1) == 0, 1.0, -1.0)
    sign_rev = torch.where(((v_ids * DEFAULT_HASH_PRIME_1) & 1) == 0, 1.0, -1.0)

    sketch_pos = torch.zeros((vocab_size, d_emb), device=device, dtype=torch.float32)
    flat_fwd_idx = u_tok * d_emb + col_hash_fwd[v_tok]
    flat_rev_idx = v_tok * d_emb + col_hash_rev[u_tok]
    ones_pairs = torch.ones_like(u_tok, dtype=torch.float32)
    sketch_pos.view(-1).scatter_add_(0, flat_fwd_idx, ones_pairs)
    sketch_pos.view(-1).scatter_add_(0, flat_rev_idx, ones_pairs)

    row_counts = sketch_pos.sum(dim=1, keepdim=True).clamp_min(1.0)
    col_counts = sketch_pos.sum(dim=0, keepdim=True).clamp_min(1.0)
    expected = (row_counts * col_counts) / (2.0 * n_pairs + 1.0)
    ppmi_sketch = F.relu(torch.log((sketch_pos + 0.5) / (expected + 0.5)))

    col_signs = torch.ones(d_emb, device=device, dtype=torch.float32)
    col_signs[: d_emb // 2] = sign_fwd[: d_emb // 2] if vocab_size >= d_emb // 2 else 1.0
    col_signs[d_emb // 2 :] = sign_rev[: d_emb // 2] if vocab_size >= d_emb // 2 else -1.0
    signed_sketch = ppmi_sketch * col_signs.unsqueeze(0)

    gen = torch.Generator(device=device).manual_seed(DEFAULT_ORTHO_BALANCE_SEED + pkt_count)
    noise = 1e-3 * torch.randn((vocab_size, d_emb), generator=gen, device=device, dtype=torch.float32)
    raw_feat = signed_sketch + noise
    raw_feat = raw_feat - raw_feat.mean(dim=0, keepdim=True)

    q_m, r_m = torch.linalg.qr(torch.randn((pkt_count, pkt_count), generator=gen, device=device, dtype=torch.float32))
    diag_sign = torch.sign(torch.diagonal(r_m))
    q_m = q_m * torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign).unsqueeze(0)

    balanced_3d = torch.matmul(raw_feat.view(vocab_size, pkt_dim, pkt_count), q_m).transpose(1, 2).contiguous()
    project_rows_to_rms_sphere_(balanced_3d)
    return balanced_3d


@torch.no_grad()
def compute_gpu_svd_bigram_sphere_embeddings(
    train_tokens: torch.Tensor,
    vocab_size: int,
    pkt_count: int,
    pkt_dim: int,
    anchor_cols: int = DEFAULT_SVD_ANCHOR_COLS,
) -> torch.Tensor:
    r"""Compute `(V, pkt_count, pkt_dim)` sphere embeddings via GPU Anchor Bigram PPMI + `torch.svd_lowrank`."""
    d_emb = pkt_count * pkt_dim
    device = train_tokens.device
    n_anchors = min(int(anchor_cols), vocab_size)
    u_tok = train_tokens[:-1]
    v_tok = train_tokens[1:]

    counts = torch.bincount(train_tokens, minlength=vocab_size)
    top_anchors = torch.topk(counts, k=n_anchors, largest=True, sorted=True).indices
    anchor_lut = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
    anchor_lut[top_anchors] = torch.arange(n_anchors, dtype=torch.long, device=device)

    cooc = torch.zeros((vocab_size, n_anchors), device=device, dtype=torch.float32)
    v_anc = anchor_lut[v_tok]
    mask_fwd = v_anc >= 0
    u_f, v_f = u_tok[mask_fwd], v_anc[mask_fwd]
    cooc.view(-1).scatter_add_(0, u_f * n_anchors + v_f, torch.ones_like(u_f, dtype=torch.float32))

    u_anc = anchor_lut[u_tok]
    mask_rev = u_anc >= 0
    v_r, u_r = v_tok[mask_rev], u_anc[mask_rev]
    cooc.view(-1).scatter_add_(0, v_r * n_anchors + u_r, torch.ones_like(v_r, dtype=torch.float32))

    row_sum = cooc.sum(dim=1, keepdim=True).clamp_min(1.0)
    col_sum = cooc.sum(dim=0, keepdim=True).clamp_min(1.0)
    tot = float(cooc.sum().item()) + 1.0
    ppmi = F.relu(torch.log((cooc * tot + 1.0) / (row_sum * col_sum + 1.0)))

    u_svd, s_svd, _ = torch.svd_lowrank(ppmi, q=min(d_emb, n_anchors), niter=2)
    spectral = u_svd[:, :d_emb] * s_svd[:d_emb].sqrt().unsqueeze(0)
    if spectral.size(1) < d_emb:
        pad = torch.randn((vocab_size, d_emb - spectral.size(1)), device=device, dtype=torch.float32) * 0.01
        spectral = torch.cat([spectral, pad], dim=1)

    gen = torch.Generator(device=device).manual_seed(DEFAULT_ORTHO_BALANCE_SEED + pkt_count)
    noise = 1e-3 * torch.randn((vocab_size, d_emb), generator=gen, device=device, dtype=torch.float32)
    spectral = spectral + noise
    q_m, r_m = torch.linalg.qr(torch.randn((pkt_count, pkt_count), generator=gen, device=device, dtype=torch.float32))
    diag_sign = torch.sign(torch.diagonal(r_m))
    q_m = q_m * torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign).unsqueeze(0)

    balanced_3d = torch.matmul(spectral.view(vocab_size, pkt_dim, pkt_count), q_m).transpose(1, 2).contiguous()
    project_rows_to_rms_sphere_(balanced_3d)
    return balanced_3d


@torch.no_grad()
def benchmark_gpu_init_timings_ms(dataset: FineWebSpanReversalDataset, repeats: int = 5) -> dict[str, float]:
    """Measure exact GPU latency (`ms`) for Unigram `bincount`, CountSketch Bigram Sphere, and Low-Rank SVD Bigram Sphere."""
    tokens = dataset.train_tokens
    v_sz = dataset.vocab_size
    rev_id = dataset.rev_token_id

    def _time_fn(fn: Any) -> float:
        fn()
        if tokens.is_cuda:
            torch.cuda.synchronize()
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(repeats):
                fn()
            end_ev.record()
            torch.cuda.synchronize()
            return float(start_ev.elapsed_time(end_ev)) / float(repeats)
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        return (time.perf_counter() - t0) * 1000.0 / float(repeats)

    unigram_ms = _time_fn(lambda: compute_gpu_unigram_log_prior_and_radial(tokens, v_sz, rev_id))
    countsketch_ms = _time_fn(
        lambda: compute_gpu_countsketch_bigram_sphere_embeddings(
            tokens, v_sz, DEFAULT_MICRO_PKT_COUNT, DEFAULT_MICRO_PKT_DIM
        )
    )
    svd_ms = _time_fn(
        lambda: compute_gpu_svd_bigram_sphere_embeddings(
            tokens, v_sz, DEFAULT_MICRO_PKT_COUNT, DEFAULT_MICRO_PKT_DIM, DEFAULT_SVD_ANCHOR_COLS
        )
    )
    return {
        "num_tokens_M": round(float(tokens.numel()) / 1e6, 3),
        "vocab_size": float(v_sz),
        "gpu_unigram_prior_ms": round(unigram_ms, 3),
        "gpu_countsketch_bigram_sphere_ms": round(countsketch_ms, 3),
        "gpu_svd_lowrank_bigram_sphere_ms": round(svd_ms, 3),
        "countsketch_speedup_vs_svd": round(svd_ms / max(1e-4, countsketch_ms), 2),
    }


# ==============================================================================
# Probability Readout Strategies (`R1..R4`)
# ==============================================================================
class MicroVocabularyReadout(nn.Module):
    r"""Unified strategy module implementing `R1..R4` probability readout architectures (Rule 3)."""

    def __init__(self, cfg: MicroReversalConfig, tied_wte_3d: nn.Parameter) -> None:
        super().__init__()
        self.cfg = cfg
        self.readout_mode = MicroReadoutMode(cfg.readout_mode)
        self.vocab_size = int(cfg.vocab_size)
        self.embed_dim = int(cfg.embed_dim)
        self.pkt_count = int(cfg.pkt_count)
        self.pkt_dim = int(cfg.pkt_dim)
        self.base_beta = DEFAULT_PARSEVAL_MUP_C_STAR * math.sqrt(float(self.embed_dim))

        self.tied_wte_ref = tied_wte_3d
        self.lm_head: nn.Linear | None = None
        self.sphere_detectors: nn.Parameter | None = None
        self.exit_bridge: BiorthogonalSVDStiefelProjection | None = None

        if self.readout_mode == MicroReadoutMode.R1_LINEAR_SOFTCAP:
            self.lm_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)
            nn.init.zeros_(self.lm_head.weight)
        elif self.readout_mode == MicroReadoutMode.R2_LEVEL_A_RADIAL_SPHERE:
            det = torch.randn((self.vocab_size, self.pkt_count, self.pkt_dim), dtype=torch.float32)
            project_rows_to_rms_sphere_(det)
            self.sphere_detectors = nn.Parameter(det)
        else:
            self.exit_bridge = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)

        self.radial_scale = nn.Parameter(torch.ones(self.vocab_size, dtype=torch.float32))
        self.unigram_bias = nn.Parameter(torch.zeros(self.vocab_size, dtype=torch.float32))
        self.prior_gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.temp_scale = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        self.born_gamma = nn.Parameter(torch.tensor(DEFAULT_BORN_GAMMA_INIT, dtype=torch.float32))
        self.quad_phase = nn.Parameter(torch.zeros(self.pkt_count // 2, self.pkt_dim, dtype=torch.float32))
        self.pkt_gate_logits = nn.Parameter(torch.zeros(self.pkt_count, dtype=torch.float32))
        self.pkt_gate_router = nn.Linear(self.embed_dim, self.pkt_count, bias=False)
        nn.init.zeros_(self.pkt_gate_router.weight)
        self.complex_sphere_mode: bool = False

    @torch.no_grad()
    def initialize_readout_priors_(
        self,
        log_prior: torch.Tensor | None,
        radial_init: torch.Tensor | None,
        init_detectors_from_wte: bool = True,
    ) -> None:
        """Initialize readout log-prior bias `unigram_bias`, radial scale `radial_scale`, and ortho-rotated sphere detectors."""
        if self.sphere_detectors is not None and init_detectors_from_wte:
            gen = torch.Generator(device=self.tied_wte_ref.device).manual_seed(DEFAULT_ORTHO_BALANCE_SEED + 77)
            q_pkt, r_pkt = torch.linalg.qr(
                torch.randn((self.pkt_dim, self.pkt_dim), generator=gen, device=self.tied_wte_ref.device, dtype=torch.float32)
            )
            d_sign = torch.sign(torch.diagonal(r_pkt))
            q_pkt = q_pkt * torch.where(d_sign == 0, torch.ones_like(d_sign), d_sign).unsqueeze(0)
            rot_det = torch.matmul(self.tied_wte_ref.detach().float(), q_pkt)
            project_rows_to_rms_sphere_(rot_det)
            self.sphere_detectors.copy_(rot_det.to(self.sphere_detectors.dtype))
        if log_prior is not None:
            self.unigram_bias.copy_(log_prior.to(self.unigram_bias.device, dtype=torch.float32))
            self.prior_gate.fill_(DEFAULT_UNIGRAM_PRIOR_SCALE_INIT)
        else:
            self.unigram_bias.zero_()
            self.prior_gate.fill_(0.0)
        if radial_init is not None and self.readout_mode != MicroReadoutMode.R1_LINEAR_SOFTCAP:
            self.radial_scale.copy_(radial_init.to(self.radial_scale.device, dtype=torch.float32))
        else:
            self.radial_scale.fill_(1.0)
        self.pkt_gate_logits.zero_()
        nn.init.zeros_(self.pkt_gate_router.weight)
        self.complex_sphere_mode = False

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Return Stiefel parameters owned by `exit_bridge` (`R3` and `R4`)."""
        if self.exit_bridge is not None:
            return self.exit_bridge.real_stiefel_parameters()
        return []

    def sphere_parameters(self) -> list[nn.Parameter]:
        """Return 3D unit-RMS sphere detector parameters (`R2`)."""
        if self.sphere_detectors is not None:
            return [self.sphere_detectors]
        return []

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return scalar/vector parameters (`radial_scale`, `unigram_bias`, `prior_gate`, `temp_scale`, `born_gamma`, `quad_phase`)."""
        params: list[nn.Parameter] = [
            self.radial_scale,
            self.unigram_bias,
            self.prior_gate,
            self.temp_scale,
            self.born_gamma,
            self.quad_phase,
            self.pkt_gate_logits,
            self.pkt_gate_router.weight,
        ]
        if self.exit_bridge is not None:
            params.extend(self.exit_bridge.scalar_parameters())
        return params

    @torch.no_grad()
    def restrict_scalars_(self) -> None:
        """Clamp readout temperatures, radial norms, and Born intensity scales to physical intervals."""
        self.radial_scale.clamp_(DEFAULT_RADIAL_MIN, DEFAULT_RADIAL_MAX)
        self.prior_gate.clamp_(0.0, 2.50)
        self.temp_scale.clamp_(0.25, 3.00)
        self.born_gamma.clamp_(0.0, DEFAULT_BORN_GAMMA_MAX)
        self.pkt_gate_logits.clamp_(-2.50, 2.50)
        if self.exit_bridge is not None:
            self.exit_bridge.clamp_sigma_()

    def forward(self, h_rev: torch.Tensor) -> torch.Tensor:
        """Compute softcapped vocabulary logits `(B, L_span, V)` on the reversal suffix hidden states."""
        bsz, span_len, _ = h_rev.shape
        work_dtype = h_rev.dtype
        bias_term = (self.prior_gate.float() * self.unigram_bias.float()).view(1, 1, self.vocab_size)

        if self.readout_mode == MicroReadoutMode.R1_LINEAR_SOFTCAP:
            assert self.lm_head is not None
            raw_logits = F.linear(h_rev, self.lm_head.weight.to(work_dtype)).float() + bias_term
            return DEFAULT_READOUT_SOFTCAP * torch.tanh(raw_logits / DEFAULT_READOUT_SOFTCAP)

        if self.readout_mode == MicroReadoutMode.R2_LEVEL_A_RADIAL_SPHERE:
            assert self.sphere_detectors is not None
            det_3d = self.sphere_detectors
            q_pkt = F.rms_norm(h_rev.view(bsz, span_len, self.pkt_count, self.pkt_dim), (self.pkt_dim,), eps=1e-6)
            q_flat = q_pkt.reshape(bsz, span_len, self.embed_dim).to(work_dtype)
            det_flat = det_3d.view(self.vocab_size, self.embed_dim).to(work_dtype)
            c_re = F.linear(q_flat, det_flat).float() / float(self.embed_dim)
            beta_eff = self.base_beta * self.temp_scale.float()
            r_v = self.radial_scale.float().view(1, 1, self.vocab_size)
            raw_logits = beta_eff * r_v * c_re + bias_term
            return DEFAULT_READOUT_SOFTCAP * torch.tanh(raw_logits / DEFAULT_READOUT_SOFTCAP)

        assert self.exit_bridge is not None
        h_exit = self.exit_bridge(h_rev)
        gate_dyn = F.linear(h_rev, self.pkt_gate_router.weight.to(work_dtype)).float()
        raw_g = F.softplus(self.pkt_gate_logits.float().view(1, 1, self.pkt_count) + gate_dyn)
        if self.complex_sphere_mode:
            n_complex = self.pkt_count // 2
            q_comp_norm = F.rms_norm(
                h_exit.view(bsz, span_len, n_complex, 2, self.pkt_dim), (2, self.pkt_dim), eps=1e-6
            )
            raw_g_c = raw_g.view(bsz, span_len, n_complex, 2).mean(dim=-1, keepdim=True)
            g_c = (raw_g_c / raw_g_c.square().mean(dim=-2, keepdim=True).sqrt().clamp_min(1e-6)).to(work_dtype)
            q_pkt = (q_comp_norm * g_c.unsqueeze(-1)).view(bsz, span_len, self.pkt_count, self.pkt_dim)
        else:
            q_pkt = F.rms_norm(h_exit.view(bsz, span_len, self.pkt_count, self.pkt_dim), (self.pkt_dim,), eps=1e-6)
            g_pkt = (raw_g / raw_g.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)).to(work_dtype).unsqueeze(-1)
            q_pkt = q_pkt * g_pkt
        q_flat = q_pkt.reshape(bsz, span_len, self.embed_dim).to(work_dtype)
        det_flat = self.tied_wte_ref.view(self.vocab_size, self.embed_dim).to(work_dtype)
        c_re = F.linear(q_flat, det_flat).float() / float(self.embed_dim)

        beta_eff = self.base_beta * self.temp_scale.float()
        r_v = self.radial_scale.float().view(1, 1, self.vocab_size)

        if self.readout_mode == MicroReadoutMode.R3_TIED_WAVEPACKET_EXIT_BRIDGE:
            raw_logits = beta_eff * r_v * c_re + bias_term
            return DEFAULT_READOUT_SOFTCAP * torch.tanh(raw_logits / DEFAULT_READOUT_SOFTCAP)

        # R4_HERMITIAN_QUADRATIC_BORN: Complex Hermitian coherent + traceless quadratic Born intensity
        n_complex = self.pkt_count // 2
        q_comp = q_pkt.view(bsz, span_len, n_complex, 2, self.pkt_dim)
        q_re, q_im = q_comp[..., 0, :], q_comp[..., 1, :]
        phi = self.quad_phase.to(work_dtype).view(1, 1, n_complex, self.pkt_dim)
        c_p, s_p = torch.cos(phi), torch.sin(phi)
        q_rot_re = c_p * q_re - s_p * q_im
        q_rot_im = s_p * q_re + c_p * q_im
        # 90-degree quadrature state (-Im, Re) to extract Im(<E_v | Psi_t>) via standard real GEMM
        q_quad = torch.stack([-q_rot_im, q_rot_re], dim=3).reshape(bsz, span_len, self.embed_dim).to(work_dtype)
        c_im = F.linear(q_quad, det_flat).float() / float(self.embed_dim)

        born_int = F.relu(c_re).square() + c_im.square()
        born_centered = born_int - born_int.mean(dim=-1, keepdim=True)
        gamma_b = self.born_gamma.float()
        obs_total = c_re + gamma_b * born_centered
        raw_logits = beta_eff * r_v * obs_total + bias_term
        return DEFAULT_READOUT_SOFTCAP * torch.tanh(raw_logits / DEFAULT_READOUT_SOFTCAP)


# ==============================================================================
# Small Classical vs. Small Quantum Wavepacket Reversal Models (`L = 2`)
# ==============================================================================
class _MicroClassicalBlock(nn.Module):
    """Single classical Transformer block with RoPE, causal SDPA, XSA, and ReLU^2 MLP."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_hidden: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.fc1 = nn.Linear(self.embed_dim, int(mlp_hidden), bias=False)
        self.fc2 = nn.Linear(int(mlp_hidden), self.embed_dim, bias=False)
        self.xsa_alpha = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x: torch.Tensor, cos_4d: torch.Tensor, sin_4d: torch.Tensor) -> torch.Tensor:
        """Run causal attention + ReLU^2 MLP."""
        bsz, seq_len, d_model = x.shape
        work_dtype = x.dtype
        h = F.rms_norm(x, (d_model,), eps=1e-6)
        qkv = F.linear(h, self.qkv_proj.weight.to(work_dtype))
        q, k, v = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim).unbind(dim=2)
        q = apply_native_leaderboard_rotary(F.rms_norm(q, (self.head_dim,), eps=1e-6), cos_4d, sin_4d)
        k = apply_native_leaderboard_rotary(F.rms_norm(k, (self.head_dim,), eps=1e-6), cos_4d, sin_4d)
        q_b, k_b, v_b = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=True)
        v_unit = F.normalize(v_b, dim=-1, eps=1e-5)
        alpha_h = torch.tanh(self.xsa_alpha).view(1, self.num_heads, 1, 1).to(attn_out.dtype)
        attn_out = attn_out - alpha_h * ((attn_out * v_unit).sum(dim=-1, keepdim=True) * v_unit)
        attn_flat = attn_out.transpose(1, 2).reshape(bsz, seq_len, d_model)
        x = x + F.linear(attn_flat, self.out_proj.weight.to(work_dtype))
        h_mlp = F.rms_norm(x, (d_model,), eps=1e-6)
        mlp_hid = F.relu(F.linear(h_mlp, self.fc1.weight.to(work_dtype))).square()
        return x + F.linear(mlp_hid, self.fc2.weight.to(work_dtype))


class _MicroQuantumWavepacketBlock(nn.Module):
    r"""Single Quantum Wavepacket StatMech block (`(B, T, 8, 32)`) + SVD-Stiefel Mach-Zehnder Interferometer MLP."""

    def __init__(self, embed_dim: int, num_heads: int, pkt_count: int, pkt_dim: int, mlp_hidden: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.pkt_count = int(pkt_count)
        self.d_h = int(pkt_dim)
        self.head_dim = self.embed_dim // self.num_heads
        self.v_rank = max(1, self.pkt_count // (2 * self.num_heads))
        self.sdpa_scale = 0.22

        self.q_proj = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)
        self.k_proj = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)
        self.v_proj = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)
        self.obs_proj = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)

        eye_h = torch.eye(self.d_h, dtype=torch.float32).unsqueeze(0).expand(self.num_heads, self.d_h, self.d_h).clone()
        self.u_evol_re = nn.Parameter(eye_h)
        self.u_evol_im = nn.Parameter(torch.zeros(self.num_heads, self.d_h, self.d_h, dtype=torch.float32))
        self.purify_alpha = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.rho_coupling = nn.Parameter(torch.full((self.num_heads,), 0.25, dtype=torch.float32))
        self.self_ortho_alpha = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.berry_qk_phase = nn.Parameter(torch.zeros(self.num_heads, self.v_rank, self.d_h, dtype=torch.float32))
        self.born_qk_beta = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.obs_scale = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))

        self.mz_mlp = SVDStiefelMachZehnderInterferometer(
            self.embed_dim, int(mlp_hidden), QuantumStiefelProjMode.BIORTHOGONAL_SVD_STIEFEL_R256
        )
        self.last_mean_purity: float = 1.0 / float(self.d_h)
        self.last_mean_eff_rank: float = float(self.d_h)

    def real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Return all Stiefel manifold matrices in attention and Mach-Zehnder MLP."""
        params: list[nn.Parameter] = []
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.obs_proj):
            params.extend(proj.real_stiefel_parameters())
        params.extend(self.mz_mlp.real_stiefel_parameters())
        return params

    def complex_unitary_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        """Return complex `(u_evol_re, u_evol_im)` unitary pair."""
        return [(self.u_evol_re, self.u_evol_im)]

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Return singular values and physical coupling scalars."""
        params: list[nn.Parameter] = []
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.obs_proj):
            params.extend(proj.scalar_parameters())
        params.extend(
            [
                self.purify_alpha,
                self.rho_coupling,
                self.self_ortho_alpha,
                self.berry_qk_phase,
                self.born_qk_beta,
                self.obs_scale,
            ]
        )
        params.extend(self.mz_mlp.scalar_parameters())
        return params

    @torch.no_grad()
    def restrict_scalars_(self) -> None:
        """Clamp singular values and quantum coupling scalars to bounded physical intervals."""
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.obs_proj):
            proj.clamp_sigma_()
        self.obs_scale.clamp_(0.05, 3.00)
        self.rho_coupling.clamp_(-1.50, 1.50)
        self.self_ortho_alpha.clamp_(-2.00, 2.00)
        self.berry_qk_phase.clamp_(-math.pi, math.pi)
        self.born_qk_beta.clamp_(-2.00, 2.00)
        self.mz_mlp.restrict_scalars_()

    def forward(self, x: torch.Tensor, cos_4d: torch.Tensor, sin_4d: torch.Tensor) -> torch.Tensor:
        """Run Quantum Wavepacket Outer-Product Attention + SVD-Stiefel Mach-Zehnder Interferometer MLP."""
        bsz, seq_len, d_model = x.shape
        h_in = F.rms_norm(x, (d_model,), eps=1e-6)
        q_raw = self.q_proj(h_in).view(bsz, seq_len, self.num_heads, self.head_dim)
        k_raw = self.k_proj(h_in).view(bsz, seq_len, self.num_heads, self.head_dim)
        v_wp = self.v_proj(h_in).view(bsz, seq_len, self.pkt_count, self.d_h)

        q_rot = apply_native_leaderboard_rotary(F.rms_norm(q_raw, (self.head_dim,), eps=1e-6), cos_4d, sin_4d)
        k_rot = apply_native_leaderboard_rotary(F.rms_norm(k_raw, (self.head_dim,), eps=1e-6), cos_4d, sin_4d)

        v_c = v_wp.view(bsz, seq_len, self.num_heads, self.v_rank, 2, self.d_h).permute(0, 2, 1, 3, 4, 5)
        vn_re, vn_im = v_c[..., 0, :], v_c[..., 1, :]
        nsq = (
            (vn_re.float().square().sum(dim=(-2, -1), keepdim=True) + vn_im.float().square().sum(dim=(-2, -1), keepdim=True))
            .clamp_min(TRACE_NORM_EPS)
            .rsqrt()
            .to(v_wp.dtype)
        )
        r_re, r_im = vn_re * nsq, vn_im * nsq
        tok_rho_re = torch.matmul(r_re.transpose(-2, -1), r_re) + torch.matmul(r_im.transpose(-2, -1), r_im)
        tok_rho_im = torch.matmul(r_im.transpose(-2, -1), r_re) - torch.matmul(r_re.transpose(-2, -1), r_im)

        v_mixed = v_wp.view(bsz, seq_len, self.num_heads, self.head_dim)
        rho0_re, rho0_im, v_bar_re, v_bar_im = fused_quantum_wavepacket_rho_ket_attention(
            q_rot,
            k_rot,
            tok_rho_re,
            tok_rho_im,
            v_mixed,
            attn_mask=None,
            phase_and_born=(self.berry_qk_phase, self.born_qk_beta, self.sdpa_scale, True),
        )
        rho_scale = torch.tanh(self.rho_coupling).view(1, self.num_heads, 1, 1, 1)
        y_heads, pur_t = associative_mcweeny_unitary_ket_modulation(
            rho0_re,
            rho0_im,
            v_bar_re,
            v_bar_im,
            v_mixed,
            (self.u_evol_re, self.u_evol_im),
            (rho_scale, self.obs_scale, True),
        )
        self.last_mean_purity = float(pur_t.mean().item())
        self.last_mean_eff_rank = float((1.0 / pur_t.clamp_min(1e-4)).mean().item())

        y_ortho = apply_fubini_study_self_ortho(y_heads, v_mixed, self.self_ortho_alpha)
        y_flat = F.rms_norm(y_ortho.reshape(bsz, seq_len, self.pkt_count, self.d_h), (self.d_h,), eps=1e-6).reshape(
            bsz, seq_len, d_model
        )
        x_mid = x + self.obs_proj(y_flat)
        return self.mz_mlp(x_mid)


class MicroReversalGPT(nn.Module):
    """Unified small-scale reversal model pairing `MicroBackboneMode` (`CLASSICAL` vs `QUANTUM_WAVEPACKET`) with `MicroReadoutMode`."""

    def __init__(self, cfg: MicroReversalConfig, seq_len: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.seq_len = int(seq_len)
        self.vocab_size = int(cfg.vocab_size)
        self.embed_dim = int(cfg.embed_dim)
        self.pkt_count = int(cfg.pkt_count)
        self.pkt_dim = int(cfg.pkt_dim)
        self.backbone_mode = MicroBackboneMode(cfg.backbone_mode)

        torch.manual_seed(int(cfg.seed))
        wte_init = torch.randn((self.vocab_size, self.pkt_count, self.pkt_dim), dtype=torch.float32)
        project_rows_to_rms_sphere_(wte_init)
        self.wte_3d = nn.Parameter(wte_init)

        wpe_init = torch.randn((self.seq_len, self.pkt_count, self.pkt_dim), dtype=torch.float32)
        project_rows_to_rms_sphere_(wpe_init)
        self.wpe_3d = nn.Parameter(wpe_init)
        self.pos_scale = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        self.wp_causal_re = nn.Parameter(torch.zeros(self.pkt_count // 2, dtype=torch.float32))
        self.wp_causal_im = nn.Parameter(torch.zeros(self.pkt_count // 2, dtype=torch.float32))

        head_dim = self.embed_dim // int(cfg.num_heads)
        self.rope = NativeLeaderboardRotary(head_dim=head_dim, max_seq_len=max(128, self.seq_len))

        self.classical_blocks = nn.ModuleList()
        self.quantum_blocks = nn.ModuleList()
        if self.backbone_mode == MicroBackboneMode.CLASSICAL:
            for _ in range(int(cfg.num_layers)):
                self.classical_blocks.append(
                    _MicroClassicalBlock(self.embed_dim, int(cfg.num_heads), int(cfg.mlp_hidden))
                )
        else:
            for _ in range(int(cfg.num_layers)):
                self.quantum_blocks.append(
                    _MicroQuantumWavepacketBlock(
                        self.embed_dim, int(cfg.num_heads), self.pkt_count, self.pkt_dim, int(cfg.mlp_hidden)
                    )
                )

        self.readout = MicroVocabularyReadout(cfg, self.wte_3d)

    @torch.no_grad()
    def apply_initialization_(self, dataset: FineWebSpanReversalDataset) -> float:
        """Initialize `wte_3d` and `readout` according to `cfg.init_mode` and return elapsed GPU time (`ms`)."""
        tokens = dataset.train_tokens
        mode = MicroInitMode(self.cfg.init_mode)
        if tokens.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if mode == MicroInitMode.RANDOM_SPHERE:
            gen = torch.Generator(device=self.wte_3d.device).manual_seed(int(self.cfg.seed) + 11)
            rnd = torch.randn(self.wte_3d.shape, generator=gen, device=self.wte_3d.device, dtype=torch.float32)
            project_rows_to_rms_sphere_(rnd)
            self.wte_3d.copy_(rnd)
            self.readout.initialize_readout_priors_(None, None, init_detectors_from_wte=True)
        elif mode == MicroInitMode.GPU_UNIGRAM_PRIOR:
            log_prior, radial = compute_gpu_unigram_log_prior_and_radial(tokens, self.vocab_size, dataset.rev_token_id)
            self.readout.initialize_readout_priors_(log_prior, radial, init_detectors_from_wte=True)
        elif mode == MicroInitMode.GPU_COUNTSKETCH_BIGRAM_SPHERE:
            log_prior, radial = compute_gpu_unigram_log_prior_and_radial(tokens, self.vocab_size, dataset.rev_token_id)
            emb_3d = compute_gpu_countsketch_bigram_sphere_embeddings(
                tokens, self.vocab_size, self.pkt_count, self.pkt_dim
            )
            self.wte_3d.copy_(emb_3d.to(self.wte_3d.device, dtype=self.wte_3d.dtype))
            self.readout.initialize_readout_priors_(log_prior, radial, init_detectors_from_wte=True)
        elif mode == MicroInitMode.GPU_SVD_BIGRAM_SPHERE:
            log_prior, radial = compute_gpu_unigram_log_prior_and_radial(tokens, self.vocab_size, dataset.rev_token_id)
            emb_3d = compute_gpu_svd_bigram_sphere_embeddings(
                tokens, self.vocab_size, self.pkt_count, self.pkt_dim, DEFAULT_SVD_ANCHOR_COLS
            )
            self.wte_3d.copy_(emb_3d.to(self.wte_3d.device, dtype=self.wte_3d.dtype))
            self.readout.initialize_readout_priors_(log_prior, radial, init_detectors_from_wte=True)

        if tokens.is_cuda:
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    def forward(self, idx_in: torch.Tensor, targets_rev: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode `idx_in` `(B, 2L+1)` and evaluate reversal readout logits `(B, L, V)` on the suffix after `<REV>`."""
        bsz, seq_len = idx_in.shape
        work_dtype = torch.bfloat16 if (idx_in.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32

        tok_wp = F.embedding(idx_in, self.wte_3d.view(self.vocab_size, self.embed_dim)).view(
            bsz, seq_len, self.pkt_count, self.pkt_dim
        )
        pos_wp = self.wpe_3d[:seq_len].unsqueeze(0) * self.pos_scale
        raw_wp = (tok_wp + pos_wp).to(work_dtype)
        if seq_len > 1:
            raw_wp = apply_causal_wavepacket_dispersion(raw_wp, self.wp_causal_re, self.wp_causal_im)

        x = F.rms_norm(raw_wp.reshape(bsz, seq_len, self.embed_dim), (self.embed_dim,), eps=1e-6)
        cos_4d, sin_4d = self.rope.get_cos_sin(seq_len, dtype=work_dtype)

        if self.backbone_mode == MicroBackboneMode.CLASSICAL:
            for c_blk in self.classical_blocks:
                x = c_blk(x, cos_4d, sin_4d)
        else:
            for q_blk in self.quantum_blocks:
                x = q_blk(x, cos_4d, sin_4d)

        x_final = F.rms_norm(x, (self.embed_dim,), eps=1e-6)
        span_len = self.cfg.span_len
        h_rev = x_final[:, span_len : 2 * span_len, :].contiguous()
        logits_rev = self.readout(h_rev)

        loss: torch.Tensor | None = None
        if targets_rev is not None:
            loss = F.cross_entropy(logits_rev.reshape(-1, self.vocab_size), targets_rev.reshape(-1).long())
        return logits_rev, loss

    @torch.no_grad()
    def restrict_all_scalars_(self) -> None:
        """Clamp all backbone and readout scalars to physical bounds."""
        self.pos_scale.clamp_(0.05, 1.50)
        self.wp_causal_re.clamp_(-1.50, 1.50)
        self.wp_causal_im.clamp_(-1.50, 1.50)
        for q_blk in self.quantum_blocks:
            q_blk.restrict_scalars_()  # type: ignore[operator]
        self.readout.restrict_scalars_()


@torch.no_grad()
def evaluate_micro_reversal_model(
    model: MicroReversalGPT,
    dataset: FineWebSpanReversalDataset,
) -> dict[str, float]:
    """Evaluate `model` on all validation batches, returning CE, Top-1/Top-5, near/far reversal accuracy, and rank-bucketed NLL."""
    was_training = model.training
    model.eval()
    nll_list: list[torch.Tensor] = []
    ranks_list: list[torch.Tensor] = []
    std_list: list[torch.Tensor] = []
    correct_list: list[torch.Tensor] = []

    for idx_in, targets_rev in dataset.val_batches:
        logits_rev, _ = model(idx_in, None)
        z = logits_rev.reshape(-1, logits_rev.size(-1)).float()
        y = targets_rev.reshape(-1).long()
        lse = torch.logsumexp(z, dim=-1)
        target_z = z.gather(1, y.unsqueeze(1)).squeeze(1)
        nll_list.append(lse - target_z)
        ranks_list.append((z > target_z.unsqueeze(1)).sum(dim=-1) + 1)
        std_list.append(z.std(dim=-1))
        preds = logits_rev.argmax(dim=-1)
        correct_list.append((preds == targets_rev).float())

    if was_training:
        model.train()

    token_nll = torch.cat(nll_list, dim=0)
    ranks = torch.cat(ranks_list, dim=0)
    logit_stds = torch.cat(std_list, dim=0)
    correct = torch.cat(correct_list, dim=0)

    mask_r1 = ranks == 1
    mask_r2_5 = (ranks >= 2) & (ranks <= 5)
    mask_r6_50 = (ranks >= 6) & (ranks <= 50)
    mask_r51_plus = ranks > 50

    top1 = float(mask_r1.float().mean().item())
    top5 = float((ranks <= 5).float().mean().item())
    val_ce = float(token_nll.mean().item())
    near_k = min(8, correct.size(1))
    return {
        "val_ce": val_ce,
        "val_ppl": float(math.exp(min(20.0, val_ce))),
        "top1_acc": top1,
        "top5_acc": top5,
        "gap_top5_top1": float(top5 - top1),
        "nll_rank1": float(token_nll[mask_r1].mean().item()) if bool(mask_r1.any()) else 0.0,
        "nll_rank2_5": float(token_nll[mask_r2_5].mean().item()) if bool(mask_r2_5.any()) else 0.0,
        "nll_rank6_50": float(token_nll[mask_r6_50].mean().item()) if bool(mask_r6_50.any()) else 0.0,
        "nll_rank51_plus": float(token_nll[mask_r51_plus].mean().item()) if bool(mask_r51_plus.any()) else 0.0,
        "logit_std": float(logit_stds.mean().item()),
        "near_top1_acc": float(correct[:, :near_k].mean().item()),
        "far_top1_acc": float(correct[:, -near_k:].mean().item()),
        "radial_scale_std": float(model.readout.radial_scale.detach().float().std().item()),
    }


def run_micro_reversal_experiment(
    cfg: MicroReversalConfig,
    dataset: FineWebSpanReversalDataset,
    total_steps: int = 300,
    eval_every: int = 100,
) -> dict[str, Any]:
    """Train and evaluate one `(backbone_mode, init_mode, readout_mode)` configuration on FineWeb Span Reversal."""
    device = dataset.device
    model = MicroReversalGPT(cfg, seq_len=dataset.seq_len).to(device)
    init_ms = model.apply_initialization_(dataset)

    stiefel_params: list[nn.Parameter] = list(model.readout.real_stiefel_parameters())
    unitary_pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
    for q_blk in model.quantum_blocks:
        stiefel_params.extend(q_blk.real_stiefel_parameters())  # type: ignore[operator]
        unitary_pairs.extend(q_blk.complex_unitary_pairs())  # type: ignore[operator]

    q_trunk_opt = (
        ModularRiemannianStiefelKrausMuon(
            real_stiefel_params=stiefel_params,
            complex_operator_pairs=unitary_pairs,
            stiefel_lr=DEFAULT_MICRO_STIEFEL_LR,
            unitary_lr=DEFAULT_MICRO_UNITARY_LR,
            tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA,
            momentum_mode=StiefelMomentumMode.RIEMANNIAN_TANGENT_SNR_GATED,
        )
        if (stiefel_params or unitary_pairs)
        else None
    )

    sphere_params = [model.wte_3d] + model.readout.sphere_parameters()
    wte_opt = BatchSubspaceTikhonovGramMuon(
        sphere_params,
        lr=DEFAULT_MICRO_WTE_LR,
        tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA,
    )

    scalar_and_pos_params: list[nn.Parameter] = [
        model.wpe_3d,
        model.pos_scale,
        model.wp_causal_re,
        model.wp_causal_im,
    ]
    scalar_and_pos_params.extend(model.readout.scalar_parameters())
    for q_blk in model.quantum_blocks:
        scalar_and_pos_params.extend(q_blk.scalar_parameters())  # type: ignore[operator]
    for c_blk in model.classical_blocks:
        scalar_and_pos_params.append(c_blk.xsa_alpha)  # type: ignore[arg-type]

    adam_groups: list[dict[str, Any]] = [
        {"params": scalar_and_pos_params, "lr": DEFAULT_MICRO_SCALAR_LR, "weight_decay": 0.0}
    ]
    if model.readout.lm_head is not None:
        adam_groups.append(
            {"params": [model.readout.lm_head.weight], "lr": DEFAULT_MICRO_CLASSICAL_ADAM_LR, "weight_decay": 0.0}
        )
    scalar_opt = torch.optim.AdamW(adam_groups, betas=(0.9, 0.95), eps=1e-8)

    c_muon_opt: Any = None
    if model.classical_blocks:
        c_mat_params: list[nn.Parameter] = []
        for c_blk in model.classical_blocks:
            c_mat_params.extend([c_blk.qkv_proj.weight, c_blk.out_proj.weight, c_blk.fc1.weight, c_blk.fc2.weight])  # type: ignore[list-item]
        c_muon_opt = StiefelMuon(
            c_mat_params,
            lr=DEFAULT_MICRO_CLASSICAL_MUON_LR,
            momentum=0.90,
            weight_decay=0.0,
            decay_mode="none",
            aspect_scale=True,
            config=StiefelMuonConfig(
                nesterov=True,
                ns_steps=5,
                polar_method="polar_express",
                stiefel_penalty=0.0,
                use_normuon=True,
                use_twin_rail=True,
            ),
            batched_ns=True,
        )

    checkpoints: list[dict[str, Any]] = []
    step0_eval = evaluate_micro_reversal_model(model, dataset)
    step0_eval["step"] = 0
    checkpoints.append(step0_eval)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_train0 = time.perf_counter()

    model.train()
    for step in range(1, int(total_steps) + 1):
        idx_in, targets_rev = dataset.sample_train_batch(step)
        warm_steps = max(1, int(0.10 * total_steps))
        if step <= warm_steps:
            lr_mult = 0.15 + 0.85 * (float(step - 1) / float(max(1, warm_steps - 1)))
        else:
            prog = float(step - warm_steps) / float(max(1, total_steps - warm_steps))
            lr_mult = 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * prog))
        mom_beta = compute_riemannian_momentum_beta(
            step, total_steps, QuantumMomentumScheduleMode.LOW_STEADY_MOMENTUM_070
        )

        if q_trunk_opt is not None:
            q_trunk_opt.zero_grad(set_to_none=True)
        wte_opt.zero_grad(set_to_none=True)
        scalar_opt.zero_grad(set_to_none=True)
        if c_muon_opt is not None:
            c_muon_opt.zero_grad(set_to_none=True)

        _, loss = model(idx_in, targets_rev)
        assert loss is not None
        loss.backward()

        if q_trunk_opt is not None:
            q_trunk_opt.set_momentum(mom_beta)
            q_trunk_opt.set_lr_scale(lr_mult)
            q_trunk_opt.step()
        active_ids = torch.unique(torch.cat([idx_in.reshape(-1), targets_rev.reshape(-1)]))
        wte_opt.set_momentum(mom_beta)
        wte_opt.step_active(active_ids, lr_scale=lr_mult)

        scalar_opt.param_groups[0]["lr"] = DEFAULT_MICRO_SCALAR_LR * lr_mult
        if len(scalar_opt.param_groups) > 1:
            scalar_opt.param_groups[1]["lr"] = DEFAULT_MICRO_CLASSICAL_ADAM_LR * lr_mult
        scalar_opt.step()
        if c_muon_opt is not None:
            for grp in c_muon_opt.param_groups:
                grp["lr"] = DEFAULT_MICRO_CLASSICAL_MUON_LR * lr_mult
            c_muon_opt.step()

        project_rows_to_rms_sphere_(model.wpe_3d)
        model.restrict_all_scalars_()

        if step % int(eval_every) == 0 or step == int(total_steps):
            ev = evaluate_micro_reversal_model(model, dataset)
            ev["step"] = int(step)
            ev["train_loss"] = float(loss.detach().item())
            checkpoints.append(ev)

    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - t_train0
    n_total_params = sum(p.numel() for p in model.parameters())
    n_readout_params = sum(p.numel() for p in model.readout.parameters())

    return {
        "backbone_mode": cfg.backbone_mode.value,
        "init_mode": cfg.init_mode.value,
        "readout_mode": cfg.readout_mode.value,
        "total_params": int(n_total_params),
        "readout_params": int(n_readout_params),
        "init_ms": round(init_ms, 3),
        "train_wall_s": round(wall_s, 3),
        "steps_per_s": round(float(total_steps) / max(1e-4, wall_s), 1),
        "step0": checkpoints[0],
        "final": checkpoints[-1],
        "checkpoints": checkpoints,
    }


__all__ = [
    "DEFAULT_MICRO_EMBED_DIM",
    "DEFAULT_MICRO_NUM_HEADS",
    "DEFAULT_MICRO_NUM_LAYERS",
    "DEFAULT_MICRO_PKT_COUNT",
    "DEFAULT_MICRO_PKT_DIM",
    "DEFAULT_MICRO_VOCAB_SIZE",
    "DEFAULT_SPAN_LEN",
    "FineWebSpanReversalDataset",
    "MicroBackboneMode",
    "MicroInitMode",
    "MicroReadoutMode",
    "MicroReversalConfig",
    "MicroReversalGPT",
    "MicroVocabularyReadout",
    "benchmark_gpu_init_timings_ms",
    "compute_gpu_countsketch_bigram_sphere_embeddings",
    "compute_gpu_svd_bigram_sphere_embeddings",
    "compute_gpu_unigram_log_prior_and_radial",
    "evaluate_micro_reversal_model",
    "run_micro_reversal_experiment",
]
