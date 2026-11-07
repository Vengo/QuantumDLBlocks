r"""Shallow Long-Sequence `(50304, 8, 2, 32)` Quantum Wavepacket Language Modeling Laboratory.

Investigates how our **`(50304, 8, 2, 32)` (`7 Semantic + 1 Identity` Complex Wavepacket)** architecture
performs on **FineWeb Causal Language Modeling** with **longer sequences (`T = 512`)** in a **shallow network
(`L = 4` layers, `d = 512`, `H = 8` heads of $\mathbb{C}^{32}$)**, testing whether a shallow Quantum Wavepacket
network can break **under `4.00` Validation Cross-Entropy (`Val CE < 4.00`, Perplexity `< 54.6`)** on unseen
FineWeb Shard 0 (`fineweb_val_000000.bin`).

Key Mathematical & Architectural Components:
1. **`(50304, 8, 2, 32)` Split Semantic-Identity Complex Wavepacket Embedding (`q_wte`)**:
   - **Packets `0..6` (`7` Semantic Complex Wavepackets, `448D`)**: Initialized from the `900M`-token (Shards `1..9`)
     multi-lag ($k \in \{1, 2, 4, 8\}$) bilateral PPMI CountSketch (`compute_multishard_multilag_complex_wavefunction`),
     with asymmetric multi-lag causal dispersion (`lags 1..4`) and context-gated causal smearing.
   - **Packet `7` (`1` Identity Complex Wavepacket, `64D`)**: Initialized via Hermitian Gram-Schmidt orthogonalization
     ($\langle s_{v,7} \mid z_{v,7}^{\perp} \rangle_{\mathbb{C}} = 0$) and kept strictly at **zero causal dispersion**
     (`wp_causal_re[7] = 0, wp_causal_im[7] = 0`) so exact token identity is never smeared across time.
2. **Shallow (`L = 4`) Quantum Wavepacket Outer-Product Attention + Mach-Zehnder Interferometer Trunk**:
   - **8 Heads Matched `1:1` to the 8 Complex Wavepackets (`7 Semantic Heads + 1 Identity/Induction Head`)**:
     Each head forms a $32 \times 32$ complex Hermitian density matrix $\rho_{t,h} \in \mathbb{C}^{32 \times 32}$ via
     zero-mask Hermitian sum packing (`fused_quantum_wavepacket_rho_ket_attention`), purifies and evolves it via
     right-associative McWeeny unitary ket modulation (`associative_mcweeny_unitary_ket_modulation`), and applies
     Fubini-Study projective self-state orthogonalization (`apply_fubini_study_self_ortho`).
   - **Single-Hop Induction via Partial-Key Offset & Anchor/Value Wavepacket Highway**:
     Shifts the second half of each head's key by 1 token ($K_{t, d_{\text{half}}:} \leftarrow K_{t-1, d_{\text{half}}:}$)
     and routes the initial `(8, 2, 32)` wavepacket anchor $X^{(0)}$ into every layer's residual and $V$-stream
     (`block_lambdas`, `sa_lambdas`, `ve_gate`), enabling shallow 4-layer in-context copying and induction across `T = 512` tokens.
   - **Two-Beam Biorthogonal SVD-Stiefel Mach-Zehnder Interferometer MLP (`SVDStiefelMachZehnderInterferometer`)**.
3. **Semantic-Subspace Causal Horizon & Surface N-Gram Wavepacket Lift (`ShallowHorizonMode`)**:
   - Compares pure multi-lag causal dispersion (`MULTILAG_DISPERSION_ONLY`) against lifting local causal bigram/trigram
     context exclusively into the **`448D` Semantic Wavepacket Subspace (`Packets 0..6`)** via a Stiefel frame
     $U_{\text{sem}} \in \mathrm{St}(448, d_{\text{ngram}})$ (`SEMANTIC_SUBSPACE_NGRAM_LIFT`), plus Step-0 empirical-Bayes
     conditional semantic transition calibration from training Shards `1..9`.
4. **Fused Chunked Softcap Readout (`_CompiledChunkedSoftcapLinearCEFn`) with `528D` Carrier-Augmented Unigram Bias**:
   - Packs the `512D` Procrustes-calibrated wavepacket inner product and `16D` constant carrier unigram log-prior into a
     single `528D` (`33 x 16`) dot product executed inside the fused Triton softcap cross-entropy kernel (`bfloat16`),
     eliminating `(B, T, 50304)` `float32` logit materialization while starting Step 0 at calibrated `~6.1–7.2` Val CE.
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
    associative_low_rank_cholesky_gram_modulation,
    fused_low_rank_cholesky_ket_attention,
)
from ..layers.impure_density_mirror_block import DEFAULT_TIKHONOV_GRAM_LAMBDA, TRACE_NORM_EPS
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
    _CompiledChunkedSoftcapLinearCEFn,
    apply_native_leaderboard_rotary,
)
from ..models.modular_quantum_statmech_gpt import ModularRiemannianStiefelKrausMuon
from ..models.parseval_mup_ngram_speedrun_gpt import _apply_causal_induction, _apply_csr_kn, _build_csr_kn
from ..models.quantum_full_backbone_graft_gpt import SVDStiefelMachZehnderInterferometer
from ..optim.muon import StiefelMuon, StiefelMuonConfig
from ..optim.vocab_manifold_muon import BatchSubspaceTikhonovGramMuon, project_rows_to_rms_sphere_
from .micro_reversal_readout_lab import (
    DEFAULT_FINEWEB_DIR,
    DEFAULT_VAL_SHARD_PATH,
    FINEWEB_HEADER_INT32,
    RAW_GPT2_VOCAB_SIZE,
)
from .support_preserving_countsketch_lab import (
    DEFAULT_FULL_FINEWEB_VOCAB_SIZE,
    SupportSketchMode,
    SupportSketchSpec,
    compute_multishard_multilag_complex_wavefunction,
    project_complex_wavepackets_to_rms_sphere_,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
DEFAULT_SHALLOW_VOCAB_SIZE: int = DEFAULT_FULL_FINEWEB_VOCAB_SIZE
DEFAULT_SHALLOW_EMBED_DIM: int = 512
DEFAULT_SHALLOW_NUM_HEADS: int = 8
DEFAULT_SHALLOW_NUM_LAYERS: int = 4
DEFAULT_SHALLOW_MLP_HIDDEN: int = 1536
DEFAULT_SHALLOW_PKT_COUNT: int = 16
DEFAULT_SHALLOW_PKT_DIM: int = 32
DEFAULT_SHALLOW_SEM_COMPLEX_PKTS: int = 7
DEFAULT_SHALLOW_SEQ_LEN: int = 512
DEFAULT_SHALLOW_BATCH_SIZE: int = 16
DEFAULT_SHALLOW_VAL_TOKENS: int = 65_536

DEFAULT_CARRIER_BIAS_DIM: int = 16
DEFAULT_CARRIER_SCALE: float = 0.25  # 1 / sqrt(16) so ||c_0||_2^2 == 1.0
DEFAULT_SOFTCAP_LOGIT: float = 15.0
DEFAULT_CE_CHUNK_TOKENS: int = 8192

DEFAULT_NGRAM_BUCKETS: int = 131072
DEFAULT_NGRAM_SEM_DIM: int = 128
DEFAULT_NGRAM_ID_DIM: int = 64
DEFAULT_NGRAM_DIM: int = DEFAULT_NGRAM_SEM_DIM + DEFAULT_NGRAM_ID_DIM
DEFAULT_BIGRAM_HASH_PRIME: int = 31337
DEFAULT_TRIGRAM_HASH_PRIME: int = 1000003
DEFAULT_SKIPGRAM_HASH_PRIME: int = 16777619

DEFAULT_NGRAM_DIRICHLET_PRIOR: float = 18.0
DEFAULT_NGRAM_DAMPED_RMS_TAU: float = 0.35

DEFAULT_SHALLOW_STIEFEL_LR: float = 0.032
DEFAULT_SHALLOW_UNITARY_LR: float = 0.032
DEFAULT_SHALLOW_WTE_LR: float = 0.060
DEFAULT_SHALLOW_READOUT_LR: float = 0.050
DEFAULT_SHALLOW_NGRAM_LR: float = 0.015
DEFAULT_SHALLOW_SCALAR_LR: float = 0.015
DEFAULT_SHALLOW_MUON_RETRACT_EVERY: int = 2
DEFAULT_STEP0_PROCRUSTES_BATCHES: int = 20
DEFAULT_STEP0_PROCRUSTES_LAMBDA: float = 0.06


class ShallowArchMode(str, Enum):
    """Strategy enum selecting the shallow long-sequence architecture variant (Rule 3)."""

    S1_CLASSICAL_4L_BASELINE = "s1_classical_4l_baseline"
    S2_QUANTUM_WAVEPACKET_4L_PURE = "s2_quantum_wavepacket_4l_pure"
    S3_QUANTUM_WAVEPACKET_4L_SEMANTIC_NGRAM = "s3_quantum_wavepacket_4l_semantic_ngram"
    S4_QUANTUM_WAVEPACKET_4L_CALIBRATED_SYNERGY = "s4_quantum_wavepacket_4l_calibrated_synergy"


@dataclass(frozen=True)
class ShallowLongSeqSpec:
    """Configuration for Shallow Long-Sequence `(50304, 8, 2, 32)` FineWeb LM experiments (Rule 2)."""

    arch_mode: ShallowArchMode = ShallowArchMode.S4_QUANTUM_WAVEPACKET_4L_CALIBRATED_SYNERGY
    vocab_size: int = DEFAULT_SHALLOW_VOCAB_SIZE
    embed_dim: int = DEFAULT_SHALLOW_EMBED_DIM
    num_heads: int = DEFAULT_SHALLOW_NUM_HEADS
    num_layers: int = DEFAULT_SHALLOW_NUM_LAYERS
    mlp_hidden: int = DEFAULT_SHALLOW_MLP_HIDDEN
    seq_len: int = DEFAULT_SHALLOW_SEQ_LEN
    batch_size: int = DEFAULT_SHALLOW_BATCH_SIZE
    num_init_shards: int = 9
    seed: int = 1337


class LongSeqFineWebStream:
    """Streams contiguous `(B, T)` training batches from FineWeb Shards `1..9` and validation batches from Shard `0`."""

    def __init__(
        self,
        seq_len: int = DEFAULT_SHALLOW_SEQ_LEN,
        batch_size: int = DEFAULT_SHALLOW_BATCH_SIZE,
        val_tokens: int = DEFAULT_SHALLOW_VAL_TOKENS,
        device: torch.device | str = "cuda",
    ) -> None:
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)

        # Load validation Shard 0 (strictly disjoint from training Shards 1..9)
        val_raw = np.memmap(DEFAULT_VAL_SHARD_PATH, dtype=np.uint16, mode="r", offset=FINEWEB_HEADER_INT32 * 4)
        n_val = max((self.batch_size * self.seq_len) + 2, int(val_tokens) + self.batch_size)
        val_np = np.asarray(val_raw[:n_val], dtype=np.int64)
        self.val_tokens = torch.from_numpy(val_np).to(device=self.device, dtype=torch.long)

        # Memory-map training Shards 1..9 on CPU and stream slices to GPU
        self.train_memmaps: list[np.memmap] = []
        for s_idx in range(1, 10):
            s_path = DEFAULT_FINEWEB_DIR / f"fineweb_train_{s_idx:06d}.bin"
            if s_path.exists():
                self.train_memmaps.append(
                    np.memmap(s_path, dtype=np.uint16, mode="r", offset=FINEWEB_HEADER_INT32 * 4)
                )
        if not self.train_memmaps:
            raise FileNotFoundError(f"No FineWeb training shards found in {DEFAULT_FINEWEB_DIR}")

        # Keep first 25M tokens of Shard 1 resident on GPU for zero-overhead Step-0 calibration & fast streaming
        res_len = min(25_000_000, len(self.train_memmaps[0]))
        self.shard1_gpu = torch.from_numpy(np.asarray(self.train_memmaps[0][:res_len], dtype=np.int64)).to(
            device=self.device, dtype=torch.long
        )

        # Pre-slice deterministic validation batches covering `val_tokens`
        self.val_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        n_val_batches = max(1, int(val_tokens) // (self.batch_size * self.seq_len))
        for b_idx in range(n_val_batches):
            base = b_idx * self.batch_size * self.seq_len
            starts = base + torch.arange(self.batch_size, device=self.device) * self.seq_len
            offsets = torch.arange(self.seq_len + 1, device=self.device).unsqueeze(0)
            chunk = self.val_tokens[starts.unsqueeze(1) + offsets]
            self.val_batches.append((chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()))

    def sample_train_batch(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a contiguous `(B, T)` training batch across Shards `1..9`."""
        n_shards = len(self.train_memmaps)
        shard_idx = (step - 1) % n_shards
        if shard_idx == 0 and step <= 1200:
            # Fast GPU-resident path on Shard 1
            max_start = self.shard1_gpu.numel() - self.seq_len - 2
            base = ((step - 1) * self.batch_size * self.seq_len) % max_start
            starts = (base + torch.arange(self.batch_size, device=self.device) * self.seq_len) % max_start
            offsets = torch.arange(self.seq_len + 1, device=self.device).unsqueeze(0)
            chunk = self.shard1_gpu[starts.unsqueeze(1) + offsets]
            return chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()

        mm = self.train_memmaps[shard_idx]
        max_start = len(mm) - (self.batch_size * (self.seq_len + 1)) - 2
        base = (((step - 1) // n_shards) * self.batch_size * self.seq_len) % max_start
        span = self.batch_size * (self.seq_len + 1)
        arr = np.asarray(mm[base : base + span], dtype=np.int64).reshape(self.batch_size, self.seq_len + 1)
        chunk = torch.from_numpy(arr).to(device=self.device, dtype=torch.long, non_blocking=True)
        return chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()


def _scale_table_preserving_shrinkage(
    mean_mat: torch.Tensor,
    cnt_vec: torch.Tensor,
    target_dim: int,
) -> torch.Tensor:
    """Scale `mean_mat` by global active-row RMS so Bayesian shrinkage on low-count buckets is preserved."""
    active = cnt_vec > 0
    if not bool(active.any().item()):
        return mean_mat
    rms = mean_mat[active].square().mean().sqrt().clamp_min(1e-6)
    return (mean_mat / rms) * (0.65 * math.sqrt(float(target_dim)))


def _damped_rms_norm(x: torch.Tensor, tau: float = DEFAULT_NGRAM_DAMPED_RMS_TAU) -> torch.Tensor:
    """Confidence-preserving Tikhonov-damped RMS normalization so zero/unseen buckets stay near zero."""
    ms = x.float().square().mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(ms + float(tau * tau))).to(x.dtype)


class SemanticSubspaceNGramProjector(nn.Module):
    r"""Differentiable causal bigram/skipgram/trigram wavepacket lift for `(50304, 8, 2, 32)`.

    - **Input Injection (`in_sem_delta_512`)**: Strictly restricted to the `448D` Semantic Subspace (`Packets 0..6`),
      leaving Identity Packet `7` (`dims 448..512`) identically zero so token identity entering Layer 1 is never blurred.
    - **Readout Fiber-Bundle Prediction (`readout_sem_448`, `readout_id_64`)**: Predicts both the next token's
      `448D` Semantic base wavepacket (via $U_{\text{sem}} \in \mathrm{St}(448, 128)$) and its `64D` Identity fiber (`Packet 7`).
    """

    def __init__(
        self,
        embed_dim: int = DEFAULT_SHALLOW_EMBED_DIM,
        sem_dim: int = 2 * DEFAULT_SHALLOW_SEM_COMPLEX_PKTS * DEFAULT_SHALLOW_PKT_DIM,
        ngram_buckets: int = DEFAULT_NGRAM_BUCKETS,
        sem_ngram_dim: int = DEFAULT_NGRAM_SEM_DIM,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.sem_dim = int(sem_dim)
        self.id_dim = self.embed_dim - self.sem_dim
        self.ngram_buckets = int(ngram_buckets)
        self.sem_ngram_dim = int(sem_ngram_dim)
        self.ngram_dim = self.sem_ngram_dim + self.id_dim

        self.unigram_next_table = nn.Parameter(torch.zeros(DEFAULT_SHALLOW_VOCAB_SIZE, self.ngram_dim))
        self.bigram_table = nn.Parameter(torch.zeros(self.ngram_buckets, self.ngram_dim))
        self.trigram_table = nn.Parameter(torch.zeros(self.ngram_buckets, self.ngram_dim))
        self.skipgram_table = nn.Parameter(torch.zeros(self.ngram_buckets, self.ngram_dim))

        # Orthonormal Stiefel frame U_sem in St(sem_dim, sem_ngram_dim): maps 128D -> 448D Semantic Subspace
        q_init, _ = torch.linalg.qr(torch.randn(self.sem_dim, self.sem_ngram_dim, dtype=torch.float32))
        self.u_sem = nn.Parameter(q_init.contiguous())

        self.in_scale = nn.Parameter(torch.full((self.sem_dim,), 0.22, dtype=torch.float32))
        self.readout_sem_scale = nn.Parameter(torch.full((self.sem_dim,), 0.85, dtype=torch.float32))
        self.readout_id_scale = nn.Parameter(torch.full((self.id_dim,), 0.85, dtype=torch.float32))
        self.order_logits = nn.Parameter(torch.tensor([0.35, 0.95, 0.65, 0.25], dtype=torch.float32))

    def compute_causal_buckets(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute strictly causal `(x_{t-1}, x_t)` bigram, `(x_{t-2}, x_{t-1}, x_t)` trigram, and `(x_{t-2}, x_t)` skip-bigram buckets."""
        x = idx.to(torch.int64)
        x_prev1 = torch.cat([x[:, :1], x[:, :-1]], dim=1) if x.size(1) > 1 else x
        x_prev2 = torch.cat([x[:, :2], x[:, :-2]], dim=1) if x.size(1) > 2 else x_prev1
        bi = torch.remainder(DEFAULT_BIGRAM_HASH_PRIME * x_prev1 + x, self.ngram_buckets)
        tri = torch.remainder(
            DEFAULT_TRIGRAM_HASH_PRIME * x_prev2 + DEFAULT_BIGRAM_HASH_PRIME * x_prev1 + x,
            self.ngram_buckets,
        )
        skip = torch.remainder(DEFAULT_SKIPGRAM_HASH_PRIME * x_prev2 + x, self.ngram_buckets)
        return bi, tri, skip

    @torch.no_grad()
    def calibrate_from_semantic_wavepackets_(
        self,
        train_tokens: torch.Tensor,
        target_table_512: torch.Tensor,
    ) -> None:
        r"""Calibrate `u_sem` and causal `(128D Semantic + 64D Identity)` tables from Shard-1 tokens in `< 90 ms` on GPU."""
        device = train_tokens.device
        sem_target = target_table_512[:, : self.sem_dim].float()
        id_target = target_table_512[:, self.sem_dim :].float()

        # 1. Extract top `sem_ngram_dim` principal directions of `sem_target` (V x 448) via fast 448x448 Gram eigh
        gram_448 = torch.matmul(sem_target.t(), sem_target)
        _, eigvecs = torch.linalg.eigh(gram_448)
        u_top = eigvecs[:, -self.sem_ngram_dim :].flip(-1).contiguous()  # (448, 128)
        self.u_sem.copy_(u_top)

        # Build 192D target code: [128D projected semantic packets, 64D exact identity packet]
        sem_low = torch.matmul(sem_target, u_top)  # (V, 128)
        target_code = torch.cat([sem_low, id_target], dim=-1).contiguous()  # (V, 192)

        uni_cnt = torch.zeros(DEFAULT_SHALLOW_VOCAB_SIZE, device=device, dtype=torch.float32)
        uni_sum = torch.zeros((DEFAULT_SHALLOW_VOCAB_SIZE, self.ngram_dim), device=device, dtype=torch.float32)
        bi_cnt = torch.zeros(self.ngram_buckets, device=device, dtype=torch.float32)
        bi_sum = torch.zeros((self.ngram_buckets, self.ngram_dim), device=device, dtype=torch.float32)
        tri_cnt = torch.zeros(self.ngram_buckets, device=device, dtype=torch.float32)
        tri_sum = torch.zeros((self.ngram_buckets, self.ngram_dim), device=device, dtype=torch.float32)
        skip_cnt = torch.zeros(self.ngram_buckets, device=device, dtype=torch.float32)
        skip_sum = torch.zeros((self.ngram_buckets, self.ngram_dim), device=device, dtype=torch.float32)

        # Stream in 2.5M-token GPU chunks so peak temporary memory is < 1.8 GB
        n_total = min(24_000_000, train_tokens.numel() - 4)
        chunk_sz = 2_500_000
        for start in range(0, n_total, chunk_sz):
            c_len = min(chunk_sz, n_total - start)
            tok = train_tokens[start : start + c_len + 3]
            x_t = tok[2 : 2 + c_len]
            x_p1 = tok[1 : 1 + c_len]
            x_p2 = tok[0:c_len]
            y_next = tok[3 : 3 + c_len]

            y_vec = target_code[y_next]
            ones = torch.ones(c_len, device=device, dtype=torch.float32)

            uni_cnt.scatter_add_(0, x_t, ones)
            uni_sum.index_add_(0, x_t, y_vec)

            bi_idx = torch.remainder(DEFAULT_BIGRAM_HASH_PRIME * x_p1 + x_t, self.ngram_buckets)
            bi_cnt.scatter_add_(0, bi_idx, ones)
            bi_sum.index_add_(0, bi_idx, y_vec)

            tri_idx = torch.remainder(
                DEFAULT_TRIGRAM_HASH_PRIME * x_p2 + DEFAULT_BIGRAM_HASH_PRIME * x_p1 + x_t,
                self.ngram_buckets,
            )
            tri_cnt.scatter_add_(0, tri_idx, ones)
            tri_sum.index_add_(0, tri_idx, y_vec)

            skip_idx = torch.remainder(DEFAULT_SKIPGRAM_HASH_PRIME * x_p2 + x_t, self.ngram_buckets)
            skip_cnt.scatter_add_(0, skip_idx, ones)
            skip_sum.index_add_(0, skip_idx, y_vec)

        uni_mean = uni_sum / (uni_cnt.unsqueeze(1) + 8.0)
        bi_mean = bi_sum / (bi_cnt.unsqueeze(1) + DEFAULT_NGRAM_DIRICHLET_PRIOR)
        tri_mean = tri_sum / (tri_cnt.unsqueeze(1) + 1.5 * DEFAULT_NGRAM_DIRICHLET_PRIOR)
        skip_mean = skip_sum / (skip_cnt.unsqueeze(1) + 1.5 * DEFAULT_NGRAM_DIRICHLET_PRIOR)

        self.unigram_next_table.copy_(_scale_table_preserving_shrinkage(uni_mean, uni_cnt, self.ngram_dim))
        self.bigram_table.copy_(_scale_table_preserving_shrinkage(bi_mean, bi_cnt, self.ngram_dim))
        self.trigram_table.copy_(_scale_table_preserving_shrinkage(tri_mean, tri_cnt, self.ngram_dim))
        self.skipgram_table.copy_(_scale_table_preserving_shrinkage(skip_mean, skip_cnt, self.ngram_dim))
        del uni_sum, bi_sum, tri_sum, skip_sum, uni_mean, bi_mean, tri_mean, skip_mean, target_code
        if device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def retract_stiefel_basis_(self) -> None:
        """Retract `u_sem` onto the tall Stiefel manifold `St(448, sem_ngram_dim)`."""
        u = self.u_sem.data.float()
        eye = torch.eye(self.sem_ngram_dim, device=u.device, dtype=torch.float32)
        gram = u.t() @ u - eye
        err = float(torch.linalg.norm(gram).item() / math.sqrt(float(self.sem_ngram_dim)))
        if err > 0.02:
            u_svd, _, vh_svd = torch.linalg.svd(u, full_matrices=False)
            u_new = u_svd @ vh_svd
        else:
            u_new = u - 0.5 * (u @ gram)
        self.u_sem.data.copy_(u_new.to(self.u_sem.dtype))

    def forward(
        self,
        idx: torch.Tensor,
        work_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(in_sem_delta_512, readout_sem_448, readout_id_64)` preserving clean input identity."""
        bsz, seq_len = idx.shape
        bi_idx, tri_idx, skip_idx = self.compute_causal_buckets(idx)
        w_ord = F.softmax(self.order_logits.float(), dim=0).to(work_dtype)
        low_comb = (
            w_ord[0] * F.embedding(idx, self.unigram_next_table.to(work_dtype))
            + w_ord[1] * F.embedding(bi_idx, self.bigram_table.to(work_dtype))
            + w_ord[2] * F.embedding(tri_idx, self.trigram_table.to(work_dtype))
            + w_ord[3] * F.embedding(skip_idx, self.skipgram_table.to(work_dtype))
        )
        sem_code = low_comb[:, :, : self.sem_ngram_dim]
        id_code = low_comb[:, :, self.sem_ngram_dim :]

        sem_lift_scale = math.sqrt(float(self.sem_dim) / float(self.sem_ngram_dim))
        sem_448 = _damped_rms_norm(F.linear(sem_code, self.u_sem.to(work_dtype)) * sem_lift_scale)
        id_64 = _damped_rms_norm(id_code)

        in_sem = sem_448 * torch.tanh(self.in_scale).to(work_dtype)
        id_zeros = torch.zeros((bsz, seq_len, self.id_dim), device=idx.device, dtype=work_dtype)
        in_sem_delta_512 = torch.cat([in_sem, id_zeros], dim=-1)

        out_sem_448 = sem_448 * (1.5 * torch.tanh(self.readout_sem_scale)).to(work_dtype)
        out_id_64 = id_64 * (1.5 * torch.tanh(self.readout_id_scale)).to(work_dtype)
        return in_sem_delta_512, out_sem_448, out_id_64


class _ShallowClassicalBlock(nn.Module):
    """Single classical Transformer block with Partial-Key Offset, Anchor/Value Highway, XSA, and ReLU^2 MLP."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_hidden: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.gate_in_ch = min(16, self.embed_dim)

        self.block_lambdas = nn.Parameter(torch.tensor([1.0, 0.15], dtype=torch.float32))
        self.sa_lambdas = nn.Parameter(torch.tensor([1.0, 0.35], dtype=torch.float32))
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.ve_gate = nn.Linear(self.gate_in_ch, self.num_heads, bias=False)
        self.attn_gate = nn.Linear(self.gate_in_ch, self.num_heads, bias=False)
        self.xsa_alpha = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.fc1 = nn.Linear(self.embed_dim, int(mlp_hidden), bias=False)
        self.fc2 = nn.Linear(int(mlp_hidden), self.embed_dim, bias=False)

        nn.init.zeros_(self.ve_gate.weight)
        nn.init.zeros_(self.attn_gate.weight)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.fc2.weight)

    def forward(
        self,
        x: torch.Tensor,
        x_anchor: torch.Tensor,
        cos_4d: torch.Tensor,
        sin_4d: torch.Tensor,
    ) -> torch.Tensor:
        """Execute classical causal attention + ReLU^2 MLP with anchor/value highways."""
        bsz, seq_len, d_model = x.shape
        work_dtype = x.dtype
        l0, l1 = self.block_lambdas[0].to(work_dtype), self.block_lambdas[1].to(work_dtype)
        x_in = l0 * x + l1 * x_anchor
        h = F.rms_norm(x_in, (d_model,), eps=1e-6)

        qkv = F.linear(h, self.qkv_proj.weight.to(work_dtype))
        q, k, v = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim).unbind(dim=2)

        # Value highway from initial wavepacket anchor
        g_ve = (2.0 * torch.sigmoid(F.linear(h[..., : self.gate_in_ch], self.ve_gate.weight.to(work_dtype)))).unsqueeze(-1)
        v_anc = x_anchor.view(bsz, seq_len, self.num_heads, self.head_dim) * g_ve
        s0, s1 = self.sa_lambdas[0].to(work_dtype), self.sa_lambdas[1].to(work_dtype)
        v = s0 * v + s1 * v_anc

        q = apply_native_leaderboard_rotary(F.rms_norm(q, (self.head_dim,), eps=1e-6), cos_4d, sin_4d)
        k = apply_native_leaderboard_rotary(F.rms_norm(k, (self.head_dim,), eps=1e-6), cos_4d, sin_4d)
        if seq_len > 1:
            d_half = self.head_dim // 2
            k_shifted = torch.cat([k[:, :1, :, d_half:], k[:, :-1, :, d_half:]], dim=1)
            k = torch.cat([k[..., :d_half], k_shifted], dim=-1)

        q_b, k_b, v_b = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=True, scale=0.14)
        v_unit = F.normalize(v_b, dim=-1, eps=1e-5)
        alpha_h = torch.tanh(self.xsa_alpha).view(1, self.num_heads, 1, 1).to(attn_out.dtype)
        attn_out = attn_out - alpha_h * ((attn_out * v_unit).sum(dim=-1, keepdim=True) * v_unit)

        attn_heads = attn_out.transpose(1, 2)
        g_at = (2.0 * torch.sigmoid(F.linear(h[..., : self.gate_in_ch], self.attn_gate.weight.to(work_dtype)))).unsqueeze(-1)
        attn_flat = (attn_heads * g_at).reshape(bsz, seq_len, d_model)
        x_mid = x_in + F.linear(attn_flat, self.out_proj.weight.to(work_dtype))

        h_mlp = F.rms_norm(x_mid, (d_model,), eps=1e-6)
        mlp_hid = F.relu(F.linear(h_mlp, self.fc1.weight.to(work_dtype))).square()
        return x_mid + F.linear(mlp_hid, self.fc2.weight.to(work_dtype))


class _ShallowQuantumWavepacketBlock(nn.Module):
    r"""Shallow Quantum Wavepacket Low-Rank Cholesky Outer-Product Attention + Mach-Zehnder Interferometer Block."""

    def __init__(
        self,
        embed_dim: int = DEFAULT_SHALLOW_EMBED_DIM,
        num_heads: int = DEFAULT_SHALLOW_NUM_HEADS,
        pkt_count: int = DEFAULT_SHALLOW_PKT_COUNT,
        pkt_dim: int = DEFAULT_SHALLOW_PKT_DIM,
        mlp_hidden: int = DEFAULT_SHALLOW_MLP_HIDDEN,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.pkt_count = int(pkt_count)
        self.d_h = int(pkt_dim)
        self.head_dim = self.embed_dim // self.num_heads
        self.v_rank = max(1, self.pkt_count // (2 * self.num_heads))
        self.gate_in_ch = min(16, self.embed_dim)
        self.sdpa_scale = 0.16

        self.block_lambdas = nn.Parameter(torch.tensor([1.0, 0.15], dtype=torch.float32))
        self.sa_lambdas = nn.Parameter(torch.tensor([1.0, 0.35], dtype=torch.float32))

        self.q_proj = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)
        self.k_proj = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)
        self.v_proj = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)
        self.obs_proj = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)

        eye_h = torch.eye(self.d_h, dtype=torch.float32).unsqueeze(0).expand(self.num_heads, self.d_h, self.d_h).clone()
        self.u_evol_re = nn.Parameter(eye_h)
        self.u_evol_im = nn.Parameter(torch.zeros(self.num_heads, self.d_h, self.d_h, dtype=torch.float32))
        self.rho_coupling = nn.Parameter(torch.full((self.num_heads,), 0.25, dtype=torch.float32))
        self.self_ortho_alpha = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.berry_qk_phase = nn.Parameter(torch.zeros(self.num_heads, self.v_rank, self.d_h, dtype=torch.float32))
        self.born_qk_beta = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.obs_scale = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))

        self.ve_gate = nn.Linear(self.gate_in_ch, self.num_heads, bias=False)
        self.q_attn_gate = nn.Linear(self.gate_in_ch, self.num_heads, bias=False)
        nn.init.zeros_(self.ve_gate.weight)
        nn.init.zeros_(self.q_attn_gate.weight)

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
        """Return singular values, highway scalars, gates, and physical coupling parameters."""
        params: list[nn.Parameter] = [self.block_lambdas, self.sa_lambdas]
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.obs_proj):
            params.extend(proj.scalar_parameters())
        params.extend(
            [
                self.rho_coupling, self.self_ortho_alpha, self.berry_qk_phase,
                self.born_qk_beta, self.obs_scale, self.ve_gate.weight, self.q_attn_gate.weight,
            ]
        )
        params.extend(self.mz_mlp.scalar_parameters())
        return params

    @torch.no_grad()
    def restrict_scalars_(self) -> None:
        """Clamp singular values and quantum coupling scalars to bounded physical intervals."""
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.obs_proj):
            proj.clamp_sigma_()
        self.block_lambdas.clamp_(-2.50, 2.50)
        self.sa_lambdas.clamp_(-2.50, 2.50)
        self.obs_scale.clamp_(0.02, 3.50)
        self.rho_coupling.clamp_(-1.50, 1.50)
        self.self_ortho_alpha.clamp_(-2.50, 2.50)
        self.berry_qk_phase.clamp_(-math.pi, math.pi)
        self.born_qk_beta.clamp_(-2.00, 2.00)
        self.mz_mlp.restrict_scalars_()

    def forward(
        self,
        x: torch.Tensor,
        x_anchor: torch.Tensor,
        cos_4d: torch.Tensor,
        sin_4d: torch.Tensor,
    ) -> torch.Tensor:
        """Execute Fused Low-Rank Cholesky Quantum Wavepacket Attention + Mach-Zehnder Interferometer MLP."""
        bsz, seq_len, d_model = x.shape
        work_dtype = x.dtype
        l0, l1 = self.block_lambdas[0].to(work_dtype), self.block_lambdas[1].to(work_dtype)
        x_in = l0 * x + l1 * x_anchor
        h_in = F.rms_norm(x_in, (d_model,), eps=1e-6)

        q_raw = self.q_proj(h_in).view(bsz, seq_len, self.num_heads, self.head_dim)
        k_raw = self.k_proj(h_in).view(bsz, seq_len, self.num_heads, self.head_dim)
        v_wp = self.v_proj(h_in).view(bsz, seq_len, self.pkt_count, self.d_h)

        # Value wavepacket highway from initial (8, 2, 32) wavepacket anchor
        g_ve = (
            2.0 * torch.sigmoid(F.linear(h_in[..., : self.gate_in_ch], self.ve_gate.weight.to(work_dtype)))
        ).unsqueeze(-1)
        v_anc_h = x_anchor.view(bsz, seq_len, self.num_heads, self.head_dim) * g_ve
        s0, s1 = self.sa_lambdas[0].to(work_dtype), self.sa_lambdas[1].to(work_dtype)
        v_mixed_h = s0 * v_wp.view(bsz, seq_len, self.num_heads, self.head_dim) + s1 * v_anc_h

        q_rot = apply_native_leaderboard_rotary(F.rms_norm(q_raw, (self.head_dim,), eps=1e-6), cos_4d, sin_4d)
        k_rot = apply_native_leaderboard_rotary(F.rms_norm(k_raw, (self.head_dim,), eps=1e-6), cos_4d, sin_4d)
        if seq_len > 1:
            d_half = self.head_dim // 2
            k_shifted = torch.cat([k_rot[:, :1, :, d_half:], k_rot[:, :-1, :, d_half:]], dim=1)
            k_rot = torch.cat([k_rot[..., :d_half], k_shifted], dim=-1)

        v_c = v_mixed_h.view(bsz, seq_len, self.num_heads, self.v_rank, 2, self.d_h).permute(0, 2, 1, 3, 4, 5)
        vn_re, vn_im = v_c[..., 0, :], v_c[..., 1, :]
        nsq = (
            (vn_re.float().square().sum(dim=(-2, -1), keepdim=True) + vn_im.float().square().sum(dim=(-2, -1), keepdim=True))
            .clamp_min(TRACE_NORM_EPS)
            .rsqrt()
            .to(work_dtype)
        )
        r_re, r_im = vn_re * nsq, vn_im * nsq

        l_re, l_im, v_bar_re, v_bar_im = fused_low_rank_cholesky_ket_attention(
            q_rot, k_rot, (r_re, r_im), v_mixed_h, attn_mask=None,
            phase_and_born=(self.berry_qk_phase, self.born_qk_beta, self.sdpa_scale, True), k_chol=4,
        )
        rho_scale = torch.tanh(self.rho_coupling).view(1, self.num_heads, 1, 1, 1)
        y_heads, pur_t = associative_low_rank_cholesky_gram_modulation(
            l_re, l_im, v_bar_re, v_bar_im, v_mixed_h,
            (self.u_evol_re, self.u_evol_im), (rho_scale, self.obs_scale, True),
        )
        self.last_mean_purity = float(pur_t.mean().item())
        self.last_mean_eff_rank = float((1.0 / pur_t.clamp_min(1e-4)).mean().item())

        v_head_rms = v_mixed_h.float().pow(2).mean(dim=-1, keepdim=True).clamp_min(TRACE_NORM_EPS).sqrt().to(work_dtype)
        y_scaled = y_heads.view(bsz, seq_len, self.num_heads, self.head_dim) * v_head_rms
        y_ortho = apply_fubini_study_self_ortho(y_scaled, v_mixed_h, self.self_ortho_alpha)
        g_at = (
            2.0 * torch.sigmoid(F.linear(h_in[..., : self.gate_in_ch], self.q_attn_gate.weight.to(work_dtype)))
        ).unsqueeze(-1)
        y_flat = (y_ortho * g_at).reshape(bsz, seq_len, d_model)
        x_mid = x_in + self.obs_proj(y_flat)
        return self.mz_mlp(x_mid)


class ShallowLongSeqWavepacketGPT(nn.Module):
    r"""Shallow (`L = 4`) Long-Sequence (`T = 512`) `(50304, 8, 2, 32)` Quantum Wavepacket Language Model."""

    def __init__(self, spec: ShallowLongSeqSpec) -> None:
        super().__init__()
        self.spec = spec
        self.arch_mode = ShallowArchMode(spec.arch_mode)
        self.vocab_size = int(spec.vocab_size)
        self.embed_dim = int(spec.embed_dim)
        self.num_heads = int(spec.num_heads)
        self.num_layers = int(spec.num_layers)
        self.mlp_hidden = int(spec.mlp_hidden)
        self.pkt_count = DEFAULT_SHALLOW_PKT_COUNT
        self.pkt_dim = DEFAULT_SHALLOW_PKT_DIM
        self.sem_complex_pkts = DEFAULT_SHALLOW_SEM_COMPLEX_PKTS
        self.sem_real_dim = 2 * self.sem_complex_pkts * self.pkt_dim  # 448
        self.id_real_dim = self.embed_dim - self.sem_real_dim  # 64

        torch.manual_seed(int(spec.seed))
        wte_init = torch.randn((self.vocab_size, self.pkt_count, self.pkt_dim), dtype=torch.float32)
        project_complex_wavepackets_to_rms_sphere_(wte_init.view(self.vocab_size, self.pkt_count // 2, 2, self.pkt_dim))
        self.wte_3d = nn.Parameter(wte_init)
        setattr(self.wte_3d, "_complex_wavepacket_pair", True)

        # Asymmetric causal wavepacket dispersion (8 complex modes: 0..6 Semantic, 7 Identity == 0)
        self.wp_causal_re = nn.Parameter(torch.zeros(self.pkt_count // 2, dtype=torch.float32))
        self.wp_causal_im = nn.Parameter(torch.zeros(self.pkt_count // 2, dtype=torch.float32))

        # Context-dependent semantic smear gate (operates only on the 448 semantic dims; identity packet 7 is untouched)
        self.smear_in_ch = 16
        self.smear_gate = nn.Linear(self.smear_in_ch, 1, bias=False)
        nn.init.zeros_(self.smear_gate.weight)
        self.smear_lambda = nn.Parameter(torch.tensor(0.18, dtype=torch.float32))

        self.sem_ngram: SemanticSubspaceNGramProjector | None = None
        if self.arch_mode in (
            ShallowArchMode.S3_QUANTUM_WAVEPACKET_4L_SEMANTIC_NGRAM,
            ShallowArchMode.S4_QUANTUM_WAVEPACKET_4L_CALIBRATED_SYNERGY,
        ):
            self.sem_ngram = SemanticSubspaceNGramProjector(embed_dim=self.embed_dim, sem_dim=self.sem_real_dim)

        head_dim = self.embed_dim // self.num_heads
        self.rope = NativeLeaderboardRotary(head_dim=head_dim, max_seq_len=max(256, int(spec.seq_len)))

        self.classical_blocks = nn.ModuleList()
        self.quantum_blocks = nn.ModuleList()
        if self.arch_mode == ShallowArchMode.S1_CLASSICAL_4L_BASELINE:
            for _ in range(self.num_layers):
                self.classical_blocks.append(_ShallowClassicalBlock(self.embed_dim, self.num_heads, self.mlp_hidden))
        else:
            for _ in range(self.num_layers):
                self.quantum_blocks.append(
                    _ShallowQuantumWavepacketBlock(
                        self.embed_dim, self.num_heads, self.pkt_count, self.pkt_dim, self.mlp_hidden
                    )
                )

        self.exit_bridge = BiorthogonalSVDStiefelProjection(self.embed_dim, self.embed_dim, self.embed_dim)
        self.w_det_512 = nn.Parameter(torch.zeros((self.vocab_size, self.embed_dim), dtype=torch.float32))
        self.radial_scale = nn.Parameter(torch.ones(self.vocab_size, dtype=torch.float32))
        self.unigram_bias = nn.Parameter(torch.zeros(self.vocab_size, dtype=torch.float32))
        self.prior_gate = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.trunk_sem_gate = nn.Parameter(torch.tensor(0.55, dtype=torch.float32))
        self.trunk_id_gate = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))
        self.temp_sem_scale = nn.Parameter(torch.tensor(8.0, dtype=torch.float32))
        self.temp_id_scale = nn.Parameter(torch.tensor(4.0, dtype=torch.float32))

        self.register_buffer(
            "carrier_c0",
            torch.full((DEFAULT_CARRIER_BIAS_DIM,), DEFAULT_CARRIER_SCALE, dtype=torch.float32),
            persistent=False,
        )

    @torch.no_grad()
    def clamp_identity_dispersion_and_scalars_(self) -> None:
        """Keep Identity Packet `7` strictly zero-dispersion and clamp physical scalars."""
        self.wp_causal_re[self.sem_complex_pkts :].zero_()
        self.wp_causal_im[self.sem_complex_pkts :].zero_()
        self.wp_causal_re.clamp_(-1.50, 1.50)
        self.wp_causal_im.clamp_(-1.50, 1.50)
        self.smear_lambda.clamp_(-1.00, 1.00)
        self.radial_scale.clamp_(0.25, 4.00)
        self.prior_gate.clamp_(0.0, 2.50)
        self.trunk_sem_gate.clamp_(0.05, 3.00)
        self.trunk_id_gate.clamp_(0.05, 3.00)
        self.temp_sem_scale.clamp_(0.50, 28.00)
        self.temp_id_scale.clamp_(0.10, 24.00)
        self.exit_bridge.clamp_sigma_()
        for q_blk in self.quantum_blocks:
            q_blk.restrict_scalars_()  # type: ignore[operator]
        if self.sem_ngram is not None:
            self.sem_ngram.retract_stiefel_basis_()

    def _encode_trunk_and_ngram(
        self,
        idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Encode `idx` `(B, T)` into pre-bridge trunk state `h_pre` and optional `(readout_sem_448, readout_id_64)`."""
        bsz, seq_len = idx.shape
        work_dtype = torch.bfloat16 if (idx.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32

        tok_wp = F.embedding(idx, self.wte_3d.view(self.vocab_size, self.embed_dim)).view(
            bsz, seq_len, self.pkt_count, self.pkt_dim
        )
        raw_wp = tok_wp.to(work_dtype)
        if seq_len > 1:
            raw_wp = apply_causal_wavepacket_dispersion(raw_wp, self.wp_causal_re, self.wp_causal_im)

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
        if self.sem_ngram is not None:
            in_sem_delta, readout_sem_448, readout_id_64 = self.sem_ngram(idx, work_dtype)
            raw_flat = raw_flat + in_sem_delta

        x_0 = F.rms_norm(raw_flat, (self.embed_dim,), eps=1e-6)
        x = x_0
        cos_4d, sin_4d = self.rope.get_cos_sin(seq_len, dtype=work_dtype)

        if self.arch_mode == ShallowArchMode.S1_CLASSICAL_4L_BASELINE:
            for c_blk in self.classical_blocks:
                x = c_blk(x, x_0, cos_4d, sin_4d)
        else:
            for q_blk in self.quantum_blocks:
                x = q_blk(x, x_0, cos_4d, sin_4d)

        h_pre = F.rms_norm(x, (self.embed_dim,), eps=1e-6)
        return h_pre, readout_sem_448, readout_id_64

    def encode_pre_bridge_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        """Encode `idx` `(B, T)` into pre-bridge hidden state `(B, T, 512)`."""
        h_pre, _, _ = self._encode_trunk_and_ngram(idx)
        return h_pre

    def _build_augmented_readout_pair(
        self,
        h_pre: torch.Tensor,
        readout_sem_448: torch.Tensor | None = None,
        readout_id_64: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build `(h_aug_528, w_aug_528)` with balanced unit-L2 Semantic (`448D`) and Identity (`64D`) scaling."""
        bsz, seq_len, _ = h_pre.shape
        work_dtype = h_pre.dtype
        h_bridge = self.exit_bridge(h_pre)

        h_sem_tr = F.rms_norm(h_bridge[:, :, : self.sem_real_dim], (self.sem_real_dim,), eps=1e-6)
        h_id_tr = F.rms_norm(h_bridge[:, :, self.sem_real_dim :], (self.id_real_dim,), eps=1e-6)
        if readout_sem_448 is not None and readout_id_64 is not None:
            g_sem, g_id = self.trunk_sem_gate.to(work_dtype), self.trunk_id_gate.to(work_dtype)
            h_sem = F.rms_norm(g_sem * h_sem_tr + readout_sem_448, (self.sem_real_dim,), eps=1e-6)
            h_id = F.rms_norm(g_id * h_id_tr + readout_id_64, (self.id_real_dim,), eps=1e-6)
        else:
            h_sem, h_id = h_sem_tr, h_id_tr

        # Normalize to unit-RMS (||h_sem||_2 = sqrt(448), ||w_sem||_2 = sqrt(448)) -> divide by d_sub for exact cosine
        h_sem_scaled = h_sem * (self.temp_sem_scale / float(self.sem_real_dim)).to(work_dtype)
        h_id_scaled = h_id * (self.temp_id_scale / float(self.id_real_dim)).to(work_dtype)

        c0_batch = self.carrier_c0.to(work_dtype).view(1, 1, DEFAULT_CARRIER_BIAS_DIM).expand(
            bsz, seq_len, DEFAULT_CARRIER_BIAS_DIM
        )
        h_aug = torch.cat([h_sem_scaled, h_id_scaled, c0_batch], dim=-1).contiguous()

        w_dir = (self.w_det_512 * self.radial_scale.unsqueeze(1)).to(work_dtype)
        b_col = (self.prior_gate * self.unigram_bias).to(work_dtype).unsqueeze(1) * self.carrier_c0.to(work_dtype).unsqueeze(0)
        w_aug = torch.cat([w_dir, b_col], dim=-1).contiguous()
        return h_aug, w_aug

    def forward_loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute fused chunked softcapped cross-entropy loss on `(idx, targets)` `(B, T)`."""
        h_pre, readout_sem_448, readout_id_64 = self._encode_trunk_and_ngram(idx)
        h_aug, w_aug = self._build_augmented_readout_pair(h_pre, readout_sem_448, readout_id_64)
        return _CompiledChunkedSoftcapLinearCEFn.apply(
            h_aug, w_aug, targets, DEFAULT_SOFTCAP_LOGIT, DEFAULT_CE_CHUNK_TOKENS
        )


@torch.no_grad()
def _get_or_build_val_kn_log_priors(stream: LongSeqFineWebStream, vocab_size: int) -> list[torch.Tensor]:
    """Build Shard-1 CSR Kneser-Ney + `T = 512` causal induction log-priors on CPU (`0 MB` persistent GPU VRAM)."""
    cached = getattr(stream, "_val_log_kn_cpu", None)
    if cached is not None:
        return cached  # type: ignore[return-value]
    n_tk = min(int(stream.shard1_gpu.numel()), 18_000_000)
    tk = stream.shard1_gpu[:n_tk].clamp(0, vocab_size - 1)
    u_s, dst = tk[:-1], tk[1:].to(torch.int32)
    csr1 = tuple(t.cpu() for t in _build_csr_kn(u_s, dst, vocab_size, 0.55, 0.80, 0.90))
    p_cont = (torch.bincount(csr1[4].to(tk.device, dtype=torch.int64), minlength=vocab_size).float() + 0.5).cpu()
    p_cont = p_cont / p_cont.sum()
    p0_f = (torch.bincount(dst.to(torch.int64), minlength=vocab_size).float() + 1.0).cpu()
    p0_f = p0_f / p0_f.sum()
    idf_w = ((-torch.log(p0_f.clamp_min(1e-8))).clamp(1.0, 14.0) ** 1.35).to(tk.device)
    p0_gpu = p0_f.to(tk.device)
    xp1 = torch.cat([u_s[:1], u_s[:-1]])
    csr_gap = tuple(t.cpu() for t in _build_csr_kn(xp1, dst, vocab_size, 0.65, 0.85, 0.92))
    csr2 = tuple(t.cpu() for t in _build_csr_kn(xp1 * 100003 + u_s, dst, vocab_size, 0.65, 0.85, 0.92))
    xp2 = torch.cat([xp1[:1], xp1[:-1]])
    csr3 = tuple(t.cpu() for t in _build_csr_kn(xp2 * 10000019 + xp1 * 100003 + u_s, dst, vocab_size, 0.72, 0.88, 0.95))
    del tk, u_s, dst, xp1, xp2
    if stream.device.type == "cuda":
        torch.cuda.empty_cache()

    val_log_kn_cpu: list[torch.Tensor] = []
    p0_base_cpu = 0.75 * p_cont + 0.25 * p0_f
    for idx_in, _ in stream.val_batches:
        bsz, seq_len = idx_in.shape
        batch_slices: list[torch.Tensor] = []
        for b0 in range(0, bsz, 4):
            b1 = min(b0 + 4, bsz)
            x_cpu = idx_in[b0:b1].to(device="cpu", dtype=torch.int64).clamp(0, vocab_size - 1)
            xp1_c = torch.cat([x_cpu[:, :1], x_cpu[:, :-1]], dim=1)
            xp2_c = torch.cat([x_cpu[:, :2], x_cpu[:, :-2]], dim=1)
            p_0 = p0_base_cpu.view(1, 1, vocab_size).expand(b1 - b0, seq_len, vocab_size)
            p_1 = 0.88 * _apply_csr_kn(x_cpu, csr1, p_0) + 0.12 * _apply_csr_kn(xp1_c, csr_gap, p_0)
            p_2 = _apply_csr_kn(xp1_c * 100003 + x_cpu, csr2, p_1)
            p_3_gpu = _apply_csr_kn(xp2_c * 10000019 + xp1_c * 100003 + x_cpu, csr3, p_2).to(stream.device)
            x_gpu = idx_in[b0:b1].clamp(0, vocab_size - 1)
            p_ind = _apply_causal_induction(x_gpu, p_3_gpu, p0_gpu, idf_w, gain=1.25)
            batch_slices.append(torch.log(p_ind.clamp_min(1e-12)).to(dtype=torch.bfloat16, device="cpu"))
            del p_3_gpu, p_ind
        val_log_kn_cpu.append(torch.cat(batch_slices, dim=0))
    del csr1, csr_gap, csr2, csr3, idf_w, p0_gpu
    if stream.device.type == "cuda":
        torch.cuda.empty_cache()
    setattr(stream, "_val_log_kn_cpu", val_log_kn_cpu)
    return val_log_kn_cpu


@torch.no_grad()
def initialize_and_calibrate_shallow_model_(
    model: ShallowLongSeqWavepacketGPT,
    stream: LongSeqFineWebStream,
) -> tuple[float, dict[str, float]]:
    r"""Initialize `(50304, 8, 2, 32)` wavepackets from Shards `1..9` and calibrate Step-0 Procrustes readout on Shard `1`."""
    device = stream.device
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    sketch_spec = SupportSketchSpec(
        mode=SupportSketchMode.I17_SHARDS1_TO_9_900M_STEP0_CALIBRATED,
        vocab_size=model.vocab_size,
        pkt_count=model.pkt_count,
        pkt_dim=model.pkt_dim,
        seed=model.spec.seed,
    )
    psi_4d, log_prior, radial, meta = compute_multishard_multilag_complex_wavefunction(
        sketch_spec,
        num_shards=model.spec.num_init_shards,
        device=device,
        rev_token_id=RAW_GPT2_VOCAB_SIZE,
    )
    model.wte_3d.copy_(psi_4d.view(model.vocab_size, model.pkt_count, model.pkt_dim))
    model.unigram_bias.copy_(log_prior)
    model.radial_scale.copy_(radial)
    model.prior_gate.fill_(1.0)

    # Initialize multi-lag causal dispersion on Semantic Packets 0..6; keep Identity Packet 7 strictly 0.0
    init_disp = torch.tensor([0.22, 0.22, 0.14, 0.08, 0.22, 0.22, 0.14, 0.00], device=device, dtype=torch.float32)
    model.wp_causal_re.copy_(init_disp)
    model.wp_causal_im.zero_()

    # Healthy initial residual authority (0.45) on quantum blocks; Procrustes SVD calibrates exit_bridge around it
    for q_blk in model.quantum_blocks:
        q_blk.obs_scale.fill_(0.45)  # type: ignore[union-attr]
        q_blk.mz_mlp.mz_scale.fill_(0.45)  # type: ignore[union-attr]

    # Build unigram-centered target detector wavepackets with unit-RMS Semantic (448D) and Identity (64D) spheres
    wte_flat = model.wte_3d.detach().float().view(model.vocab_size, model.embed_dim)
    p0 = F.softmax(model.unigram_bias.detach().float(), dim=0).unsqueeze(1)
    sem_raw = wte_flat[:, : model.sem_real_dim]
    id_raw = wte_flat[:, model.sem_real_dim :]
    active_sem_mask = (sem_raw.norm(dim=-1, keepdim=True) > 1e-3).float()
    mean_sem = (p0 * sem_raw).sum(dim=0, keepdim=True)
    centered_sem = F.normalize(sem_raw - active_sem_mask * mean_sem, dim=-1, eps=1e-6)
    centered_id = F.normalize(id_raw - (p0 * id_raw).sum(dim=0, keepdim=True), dim=-1, eps=1e-6)

    target_sem_rms = centered_sem * math.sqrt(float(model.sem_real_dim))
    target_id_rms = centered_id * math.sqrt(float(model.id_real_dim))
    model.w_det_512.copy_(torch.cat([target_sem_rms, target_id_rms], dim=-1))
    target_table = model.w_det_512.detach().float() * model.radial_scale.detach().float().unsqueeze(1)

    if model.sem_ngram is not None:
        model.sem_ngram.calibrate_from_semantic_wavepackets_(stream.shard1_gpu, target_table)
    if model.arch_mode == ShallowArchMode.S4_QUANTUM_WAVEPACKET_4L_CALIBRATED_SYNERGY:
        _get_or_build_val_kn_log_priors(stream, model.vocab_size)

    # Solve Step-0 Ridge-Procrustes SVD calibration of `exit_bridge` on Shard 1
    d_emb = model.embed_dim
    cov_hh = torch.zeros((d_emb, d_emb), device=device, dtype=torch.float32)
    cov_hy = torch.zeros((d_emb, d_emb), device=device, dtype=torch.float32)
    cal_batches: list[tuple[torch.Tensor, torch.Tensor]] = []

    was_training = model.training
    model.eval()
    n_cal = DEFAULT_STEP0_PROCRUSTES_BATCHES
    max_start = stream.shard1_gpu.numel() - stream.seq_len - 2
    stride = max(1, max_start // (n_cal * stream.batch_size))

    for b_idx in range(n_cal):
        base = b_idx * stream.batch_size * stride
        starts = (base + torch.arange(stream.batch_size, device=device) * stride) % max_start
        offsets = torch.arange(stream.seq_len + 1, device=device).unsqueeze(0)
        chunk = stream.shard1_gpu[starts.unsqueeze(1) + offsets]
        idx_in = chunk[:, :-1].contiguous()
        targets = chunk[:, 1:].contiguous()
        if b_idx < 2:
            cal_batches.append((idx_in, targets))

        h_flat = model.encode_pre_bridge_hidden(idx_in).reshape(-1, d_emb).float()
        y_flat = target_table[targets.reshape(-1)]
        cov_hh.add_(torch.matmul(h_flat.t(), h_flat))
        cov_hy.add_(torch.matmul(h_flat.t(), y_flat))

    n_samples = float(n_cal * stream.batch_size * stream.seq_len)
    cov_hh.div_(n_samples)
    cov_hy.div_(n_samples)

    reg_eye = DEFAULT_STEP0_PROCRUSTES_LAMBDA * torch.eye(d_emb, device=device, dtype=torch.float32)
    w_star = torch.linalg.solve(cov_hh + reg_eye, cov_hy)
    u_svd, s_svd, vh_svd = torch.linalg.svd(w_star.t(), full_matrices=False)
    s_rms = s_svd.square().mean().sqrt().clamp_min(1e-6)
    s_norm = (s_svd / s_rms).clamp(0.05, 5.0)

    model.exit_bridge.u_stiefel.copy_(u_svd.contiguous())
    model.exit_bridge.v_stiefel.copy_(vh_svd.t().contiguous())
    model.exit_bridge.sigma.copy_(s_norm.contiguous())
    del cov_hh, cov_hy, w_star, u_svd, s_svd, vh_svd, target_table

    # Line-search optimal Step-0 (temp_sem_scale, temp_id_scale) on Shard-1 calibration batches
    best_sem_t, best_id_t, best_loss = 4.0, 0.8, float("inf")
    for cand_sem in (3.5, 5.5, 8.0, 11.0):
        for cand_id in (0.6, 1.5, 3.0, 5.0):
            model.temp_sem_scale.fill_(cand_sem)
            model.temp_id_scale.fill_(cand_id)
            tot_l = sum(float(model.forward_loss(x_b, y_b).item()) for x_b, y_b in cal_batches)
            if tot_l < best_loss:
                best_loss, best_sem_t, best_id_t = tot_l, cand_sem, cand_id
    model.temp_sem_scale.fill_(best_sem_t)
    model.temp_id_scale.fill_(best_id_t)
    del cal_batches
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if was_training:
        model.train()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms, meta


@torch.no_grad()
def evaluate_shallow_longseq_model(
    model: ShallowLongSeqWavepacketGPT,
    stream: LongSeqFineWebStream,
    progress: float = 0.0,
) -> dict[str, float]:
    """Evaluate `model` across unseen Shard-0 validation batches (`65,536` tokens), returning `val_ce`, `val_ppl`, `top1_acc`, `top5_acc`."""
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_neural_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_tokens = 0
    use_synergy = model.arch_mode == ShallowArchMode.S4_QUANTUM_WAVEPACKET_4L_CALIBRATED_SYNERGY
    val_kn_list = _get_or_build_val_kn_log_priors(stream, model.vocab_size) if use_synergy else None

    for b_idx, (idx_in, targets) in enumerate(stream.val_batches):
        h_pre, readout_sem_448, readout_id_64 = model._encode_trunk_and_ngram(idx_in)
        h_aug, w_aug = model._build_augmented_readout_pair(h_pre, readout_sem_448, readout_id_64)

        loss_neural = _CompiledChunkedSoftcapLinearCEFn.apply(
            h_aug, w_aug, targets, DEFAULT_SOFTCAP_LOGIT, DEFAULT_CE_CHUNK_TOKENS
        )
        n_tok = int(targets.numel())
        total_neural_loss += float(loss_neural.item()) * n_tok

        if not use_synergy or val_kn_list is None:
            total_loss += float(loss_neural.item()) * n_tok
            h_flat = h_aug.reshape(-1, h_aug.size(-1))
            y_flat = targets.reshape(-1)
            w_t = w_aug.t()
            for start in range(0, n_tok, 2048):
                end = min(start + 2048, n_tok)
                logits_c = torch.matmul(h_flat[start:end], w_t)
                yc = y_flat[start:end]
                top5_idx = torch.topk(logits_c, k=5, dim=-1, largest=True, sorted=True).indices
                total_top1 += int((top5_idx[:, 0] == yc).sum().item())
                total_top5 += int((top5_idx == yc.unsqueeze(1)).any(dim=-1).sum().item())
            total_tokens += n_tok
            continue

        # S4 Calibrated Synergy: Jensen-convex mixture of neural wavepacket + Kneser-Ney/induction distributions
        bsz = idx_in.size(0)
        v_sz = model.vocab_size
        w_aug_t = w_aug.t()
        log_alpha = math.log(0.55)
        log_one_minus_alpha = math.log(0.45)
        kn_batch_cpu = val_kn_list[b_idx]
        for b0 in range(0, bsz, 4):
            b1 = min(b0 + 4, bsz)
            y_sl = targets[b0:b1]
            log_kn = F.log_softmax(kn_batch_cpu[b0:b1].to(device=h_aug.device, dtype=torch.float32), dim=-1)
            raw_z = torch.matmul(h_aug[b0:b1], w_aug_t).float()
            z_neural = DEFAULT_SOFTCAP_LOGIT * torch.tanh(raw_z / DEFAULT_SOFTCAP_LOGIT)
            log_p_neural = F.log_softmax(z_neural, dim=-1)
            logits_syn = torch.logaddexp(log_p_neural + log_one_minus_alpha, log_kn + log_alpha)
            total_loss += float(F.nll_loss(logits_syn.reshape(-1, v_sz), y_sl.reshape(-1), reduction="sum").item())
            top5_idx = torch.topk(logits_syn.reshape(-1, v_sz), k=5, dim=-1, largest=True, sorted=True).indices
            yc = y_sl.reshape(-1)
            total_top1 += int((top5_idx[:, 0] == yc).sum().item())
            total_top5 += int((top5_idx == yc.unsqueeze(1)).any(dim=-1).sum().item())
        total_tokens += n_tok

    if was_training:
        model.train()

    val_ce = total_loss / float(max(1, total_tokens))
    neural_ce = total_neural_loss / float(max(1, total_tokens))
    return {
        "val_ce": round(val_ce, 4),
        "neural_val_ce": round(neural_ce, 4),
        "val_ppl": round(float(math.exp(min(20.0, val_ce))), 2),
        "top1_acc": round(float(total_top1) / float(max(1, total_tokens)), 4),
        "top5_acc": round(float(total_top5) / float(max(1, total_tokens)), 4),
    }


def run_shallow_longseq_experiment(
    spec: ShallowLongSeqSpec,
    stream: LongSeqFineWebStream,
    total_steps: int = 1200,
    eval_every: int = 200,
) -> dict[str, Any]:
    """Train and evaluate a shallow (`L = 4`) long-sequence (`T = 512`) FineWeb language model."""
    device = stream.device
    model = ShallowLongSeqWavepacketGPT(spec).to(device)
    init_ms, meta = initialize_and_calibrate_shallow_model_(model, stream)

    stiefel_params: list[nn.Parameter] = list(model.exit_bridge.real_stiefel_parameters())
    unitary_pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
    for q_blk in model.quantum_blocks:
        stiefel_params.extend(q_blk.real_stiefel_parameters())  # type: ignore[operator]
        unitary_pairs.extend(q_blk.complex_unitary_pairs())  # type: ignore[operator]

    q_trunk_opt = (
        ModularRiemannianStiefelKrausMuon(
            real_stiefel_params=stiefel_params,
            complex_operator_pairs=unitary_pairs,
            stiefel_lr=DEFAULT_SHALLOW_STIEFEL_LR,
            unitary_lr=DEFAULT_SHALLOW_UNITARY_LR,
            tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA,
            momentum_mode=StiefelMomentumMode.RIEMANNIAN_TANGENT_SNR_GATED,
        )
        if (stiefel_params or unitary_pairs)
        else None
    )

    c_muon_opt: Any = None
    if model.classical_blocks:
        c_mat_params: list[nn.Parameter] = []
        for c_blk in model.classical_blocks:
            c_mat_params.extend([c_blk.qkv_proj.weight, c_blk.out_proj.weight, c_blk.fc1.weight, c_blk.fc2.weight])  # type: ignore[list-item]
        c_cfg = StiefelMuonConfig(nesterov=True, ns_steps=5, polar_method="polar_express", stiefel_penalty=0.0, use_normuon=True, use_twin_rail=True)
        c_muon_opt = StiefelMuon(c_mat_params, lr=DEFAULT_SHALLOW_STIEFEL_LR, momentum=0.90, weight_decay=0.0, decay_mode="none", aspect_scale=True, config=c_cfg, batched_ns=True)

    wte_opt = BatchSubspaceTikhonovGramMuon([model.wte_3d], lr=DEFAULT_SHALLOW_WTE_LR, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA)
    head_opt = BatchSubspaceTikhonovGramMuon([model.w_det_512], lr=DEFAULT_SHALLOW_READOUT_LR, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA)

    fast_temp_params: list[nn.Parameter] = [model.temp_sem_scale, model.temp_id_scale, model.trunk_sem_gate, model.trunk_id_gate]
    scalar_params: list[nn.Parameter] = [
        model.wp_causal_re, model.wp_causal_im, model.smear_gate.weight, model.smear_lambda,
        model.radial_scale, model.unigram_bias, model.prior_gate,
    ]
    scalar_params.extend(model.exit_bridge.scalar_parameters())
    for q_blk in model.quantum_blocks:
        scalar_params.extend(q_blk.scalar_parameters())  # type: ignore[operator]
    for c_blk in model.classical_blocks:
        scalar_params.extend([c_blk.block_lambdas, c_blk.sa_lambdas, c_blk.ve_gate.weight, c_blk.attn_gate.weight, c_blk.xsa_alpha])  # type: ignore[list-item]

    adam_groups: list[dict[str, Any]] = [
        {"params": scalar_params, "lr": DEFAULT_SHALLOW_SCALAR_LR, "weight_decay": 0.0},
        {"params": fast_temp_params, "lr": 0.06, "weight_decay": 0.0},
    ]
    if model.sem_ngram is not None:
        ngram_tables = [
            model.sem_ngram.unigram_next_table, model.sem_ngram.bigram_table,
            model.sem_ngram.trigram_table, model.sem_ngram.skipgram_table, model.sem_ngram.u_sem,
        ]
        ngram_scalars = [
            model.sem_ngram.in_scale, model.sem_ngram.readout_sem_scale,
            model.sem_ngram.readout_id_scale, model.sem_ngram.order_logits,
        ]
        adam_groups.append({"params": ngram_tables, "lr": DEFAULT_SHALLOW_NGRAM_LR, "weight_decay": 0.0})
        adam_groups.append({"params": ngram_scalars, "lr": DEFAULT_SHALLOW_SCALAR_LR, "weight_decay": 0.0})

    adam_opt = torch.optim.AdamW(adam_groups, betas=(0.80, 0.95), eps=1e-8, fused=(device.type == "cuda"))

    checkpoints: list[dict[str, Any]] = []
    s0 = evaluate_shallow_longseq_model(model, stream, progress=0.0)
    s0["step"] = 0
    s0["tokens_M"] = 0.0
    checkpoints.append(s0)
    print(
        f"[{spec.arch_mode.value}] Step 0 (0.00M tok) | Val CE={s0['val_ce']:.4f} (Neural={s0['neural_val_ce']:.4f}) "
        f"PPL={s0['val_ppl']:.2f} Top1={s0['top1_acc'] * 100:.2f}% Top5={s0['top5_acc'] * 100:.2f}% "
        f"(T_sem={model.temp_sem_scale.item():.1f}, T_id={model.temp_id_scale.item():.1f}, init={init_ms:.0f}ms)",
        flush=True,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_train0 = time.perf_counter()
    tokens_per_step = int(spec.batch_size) * int(spec.seq_len)

    model.train()
    for step in range(1, int(total_steps) + 1):
        warm_steps = max(1, int(0.08 * total_steps))
        if step <= warm_steps:
            lr_mult = 0.20 + 0.80 * (float(step - 1) / float(max(1, warm_steps - 1)))
        else:
            prog = float(step - warm_steps) / float(max(1, total_steps - warm_steps))
            lr_mult = 0.12 + 0.88 * 0.5 * (1.0 + math.cos(math.pi * prog))
        mom_beta = compute_riemannian_momentum_beta(
            step, total_steps, QuantumMomentumScheduleMode.LOW_STEADY_MOMENTUM_070
        )

        if q_trunk_opt is not None:
            q_trunk_opt.zero_grad(set_to_none=True)
        if c_muon_opt is not None:
            c_muon_opt.zero_grad(set_to_none=True)
        wte_opt.zero_grad(set_to_none=True)
        head_opt.zero_grad(set_to_none=True)
        adam_opt.zero_grad(set_to_none=True)

        idx_in, targets = stream.sample_train_batch(step)
        loss = model.forward_loss(idx_in, targets)
        loss.backward()

        if q_trunk_opt is not None:
            q_trunk_opt.set_momentum(mom_beta)
            q_trunk_opt.set_lr_scale(lr_mult)
            q_trunk_opt.step()
        if c_muon_opt is not None:
            for g in c_muon_opt.param_groups:
                g["lr"] = DEFAULT_SHALLOW_STIEFEL_LR * lr_mult
            c_muon_opt.step()

        active_ids = torch.unique(torch.cat([idx_in.reshape(-1), targets.reshape(-1)]))
        wte_opt.set_momentum(mom_beta)
        wte_opt.step_active(active_ids, lr_scale=lr_mult)
        head_opt.set_momentum(mom_beta)
        head_opt.step_full(lr_scale=lr_mult, active_idx=active_ids)

        adam_opt.param_groups[0]["lr"] = DEFAULT_SHALLOW_SCALAR_LR * lr_mult
        adam_opt.param_groups[1]["lr"] = 0.06 * lr_mult
        if len(adam_opt.param_groups) > 2:
            adam_opt.param_groups[2]["lr"] = DEFAULT_SHALLOW_NGRAM_LR * lr_mult
            adam_opt.param_groups[3]["lr"] = DEFAULT_SHALLOW_SCALAR_LR * lr_mult
        adam_opt.step()

        model.clamp_identity_dispersion_and_scalars_()

        if step % int(eval_every) == 0 or step == int(total_steps):
            ev = evaluate_shallow_longseq_model(model, stream, progress=float(step) / float(max(1, total_steps)))
            ev["step"] = int(step)
            ev["tokens_M"] = round(float(step * tokens_per_step) / 1e6, 3)
            ev["train_loss"] = round(float(loss.detach().item()), 4)
            ev["elapsed_s"] = round(time.perf_counter() - t_train0, 1)
            checkpoints.append(ev)
            print(
                f"[{spec.arch_mode.value}] Step {step:4d} ({ev['tokens_M']:5.2f}M tok, {ev['elapsed_s']:5.1f}s) | "
                f"Train CE={ev['train_loss']:.4f} | Val CE={ev['val_ce']:.4f} (Neural={ev['neural_val_ce']:.4f}) "
                f"PPL={ev['val_ppl']:6.2f} Top1={ev['top1_acc'] * 100:.2f}% Top5={ev['top5_acc'] * 100:.2f}%",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - t_train0
    del model, q_trunk_opt, c_muon_opt, wte_opt, head_opt, adam_opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "arch_mode": spec.arch_mode.value,
        "num_layers": int(spec.num_layers),
        "embed_dim": int(spec.embed_dim),
        "seq_len": int(spec.seq_len),
        "batch_size": int(spec.batch_size),
        "total_steps": int(total_steps),
        "total_tokens_M": round(float(total_steps * tokens_per_step) / 1e6, 3),
        "init_ms": round(init_ms, 2),
        "train_wall_s": round(wall_s, 2),
        "meta": meta,
        "step0": checkpoints[0],
        "final": checkpoints[-1],
        "checkpoints": checkpoints,
    }
