"""900M-Token Step-0 Unigram, Bigram, and Trigram Calibrated Kronecker Born GPT (`Wave 43` / `EXPERIMENT_PLAN_058.md`).

When to Use This Building Block:
- Use `NgramCalibratedKroneckerBornGPT`, `NgramCalibratedKroneckerConfig`, `Ngram900MCalibrationData`,
  `build_or_load_900m_ngram_calibration_cache`, `evaluate_modded_nanogpt_validation`,
  `MultiShardRareTokenSampler`, `compute_ngram_calibrated_kronecker_penalties`, and
  `build_ngram_calibrated_kronecker_optimizers` when training pure-complex Rank-`R` `16^4 = 65,536`
  Kronecker wavefunction language models on multi-shard `FineWeb` (`900,000,000` training tokens)
  with **zero frozen `prior_weights` buffer** and **zero `exp`/`softmax`/`logsumexp` operations**.

Why It Was Built & Mathematical Provenance (`EXPERIMENT_PLAN_058.md` / `Wave 43`):
1. **Isometric Helmert Subspace Embedding (`embed_real30_to_carrier_ortho_c16` & `project_c16_to_carrier_ortho_real30`)**:
   Maps 30D real spectral coordinates `w in R^{30}` per Kronecker factor isometrically into the 15D complex
   carrier-orthogonal subspace `H_perp = {|u> in C^16 : <c_ref | u> = 0}` where
   `|c_ref> = (1 + i)/(4 * sqrt(2)) 1_16 in S_C^15`. Guarantees `<c_ref | E_perp(w)> == 0` (`< 1e-7`)
   and `||E_perp(w)||_{C^16} == ||w||_{R^{30}}` identically so conditional N-gram amplitudes never perturb
   the unigram carrier alignment.
2. **2-Pass `900M`-Token GPU Spectral & Conditional Expectation Cache (`build_or_load_900m_ngram_calibration_cache`)**:
   - Pass 1 streams all `900,000,000` FineWeb tokens (`fineweb_train_000001.bin`..`fineweb_train_000009.bin`)
     on GPU to compute exact `900M`-token unigram counts and top-`1,024` anchor bigram and lag-1 skip-bigram
     root-lift matrices `S_bi, S_skip1`, extracting the top `D_svd = 120 = 4 * 30` spectral vocabulary factor
     coordinates `U_vocab in R^{50257 x 4 x 30}` via `torch.svd_lowrank` and Polar ZCA whitening.
   - Pass 2 accumulates `P_uni`-whitened conditional expectation kets in `H_perp` across all `900M` tokens
     for `K_bi = 8,192` direct bigram slots (plus slot 0 rare fallback) and `H_tri = 16,384` polynomial
     XOR-hashed trigram buckets, and fits strictly unitary `U(16)` causal bigram and lag-1 skip-bigram factor
     operators (`bi_factor_op_re_im`, `tri_factor_op_re_im`) via Complex Unitary Procrustes SVD (`err < 1e-5`).
3. **Pass 3 Step-0 Random-Attention Procrustes + Algebraic POVM Polish (`calibrate_from_900m_ngram_cache`)**:
   Aligns `out_factor_u` onto `|c_ref>` (`carrier_alignment_fidelity = 1.000000`), injects the `900M`-token
   causal Bigram + Trigram Hilbert-space kets in `H_perp`, and solves for `povm_scale_logits` in closed form
   so Step-0 (`0` gradient steps!) validation CE drops by `0.674` nats (`7.9413 -> 7.2669`) and Top-1 accuracy
   jumps from `3.28%` to `9.41%` before training begins.
4. **Zero-Materialization `10,485,760`-Token Official `modded-nanogpt` Evaluator (`evaluate_modded_nanogpt_validation`)**:
   Evaluates exact polynomial Born NLL, Perplexity, Top-1, and optional Top-50 accuracy across the full
   `10.49M`-token `fineweb_val_000000.bin` slice in vocabulary chunks without allocating `(N, 50304)` logits.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..layers.packed_complex_register_density_attn import ComplexLinear, complex_l2_normalize
from ..optim.muon import StiefelMuon
from .calibrated_kronecker_born_gpt import (
    CalibratedKroneckerBornChunkedNLLFn, CalibratedKroneckerBornGPT, CalibratedKroneckerConfig,
    DEFAULT_CALIBRATION_MAX_EVAL_TOKENS, DEFAULT_CARRIER_STD_DIVISOR, DEFAULT_POVM_INV_FLOOR,
    DEFAULT_POVM_MIN_EFF_WEIGHT, _build_canonical_carrier_ket, _build_unitary_procrustes_carrier_batch,
    build_calibrated_kronecker_optimizers, eval_calibrated_kronecker_born_probs,
)
from .kronecker_multimeaning_16x4_gpt import (
    DEFAULT_BRANCH_EPS, DEFAULT_KRON_EPS, DEFAULT_MUON_REFERENCE_BATCH_TOKENS, DEFAULT_PSI0_GATE_INIT,
    DEFAULT_REF_PHASE_COS_SIN, DEFAULT_SELF_ECHO_LAMBDA_INIT, DEFAULT_SELF_ECHO_MODE,
    Kronecker16x4UnitaryMeaningMixer, KroneckerComplexRegisterBlock, PARTITION_CLAMP_MIN,
    PROB_LOG_CLAMP_MIN, _batch_complex_unitary_terms, _compiled_kronecker_chunk_intensity,
    _compiled_mixed_state_eval_intensity, _compute_kronecker_chunk_intensity,
    _compute_mixed_state_eval_intensity, _init_square_complex_linear_haar, _sample_haar_complex_unitary_batch,
    _self_echo_lambda_to_logit, apply_carrier_orthogonal_complex_self_echo,
    compute_kronecker_manifold_and_isotropy_penalties,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_058.md & Tracks 1, 2, 3, 5
# ==============================================================================
DEFAULT_NUM_FACTORS: int = 4
DEFAULT_FACTOR_DIM: int = 16
DEFAULT_NUM_MEANINGS: int = 2
DEFAULT_NGRAM_BOSON_P: float = 8.0
DEFAULT_NGRAM_HOLO_ALPHA: float = 0.85
DEFAULT_NGRAM_KRON_BLEND: float = 0.40
DEFAULT_NGRAM_COINC_GAMMA: float = 0.35
DEFAULT_NGRAM_UNIGRAM_TEMP: float = 0.90
DEFAULT_CARRIER_RHO_MAX: float = 0.22
DEFAULT_TOP_ANCHOR_TOKENS: int = 1024
DEFAULT_DIRECT_BIGRAM_SLOTS: int = 8192
DEFAULT_HASHED_TRIGRAM_BUCKETS: int = 16384
DEFAULT_SPECTRAL_SVD_RANK: int = 120
DEFAULT_SPECTRAL_WAVE_BLEND: float = 0.88
DEFAULT_BIGRAM_DIRICHLET_BETA: float = 120.0
DEFAULT_TRIGRAM_DIRICHLET_BETA: float = 160.0
DEFAULT_NGRAM_SKIP_SCALE: float = 0.42
DEFAULT_SMEAR_GATE_INIT: float = -1.50
DEFAULT_VOCAB_CHUNK_SIZE: int = 4096
DEFAULT_EVAL_VOCAB_CHUNK_SIZE: int = 8192
MIN_EVAL_VOCAB_CHUNK_SIZE: int = 256
TRIGRAM_HASH_PRIME_1: int = 1000003
TRIGRAM_HASH_PRIME_2: int = 16777619

DEFAULT_ANCHOR_FREQ_EXP: float = 0.35
DEFAULT_SVD_SIGMA_EXP: float = 0.35
DEFAULT_SKIP1_STACK_WEIGHT: float = 0.50
DEFAULT_WHITENING_RIDGE: float = 1.5e-3
DEFAULT_PROCRUSTES_RIDGE: float = 1e-4
DEFAULT_SHARD_SCAN_CHUNK_TOKENS: int = 4_000_000
DEFAULT_NGRAM_GATE_BI_INIT: float = 1.45
DEFAULT_NGRAM_GATE_TRI_INIT: float = 0.90
DEFAULT_NGRAM_GATE_OP_INIT: float = 0.24
DEFAULT_NGRAM_SECONDARY_BRANCH_SCALE: float = 0.22

DEFAULT_POVM_ANCHOR_WEIGHT: float = 0.02
DEFAULT_CTX_ANCHOR_WEIGHT: float = 0.015
DEFAULT_SPHERE_PENALTY_WEIGHT: float = 2.0
DEFAULT_UNITARY_PENALTY_WEIGHT: float = 0.08
DEFAULT_ISO_PENALTY_WEIGHT: float = 0.05
DEFAULT_BRANCH_ORTH_WEIGHT: float = 0.20

DEFAULT_MUON_LR: float = 0.030
DEFAULT_EMBED_LR: float = 0.020
DEFAULT_POVM_LR: float = 0.006
DEFAULT_GATE_LR: float = 0.004
DEFAULT_MODDED_VAL_TOKENS: int = 10_485_760
DEFAULT_RARE_SAMPLER_ALPHA: float = 0.25
DEFAULT_RARE_SAMPLER_POOL_MULT: int = 4


@dataclass(frozen=True)
class NgramCalibratedKroneckerConfig(CalibratedKroneckerConfig):
    """Frozen configuration for `NgramCalibratedKroneckerBornGPT` (`Wave 43` / `EXPERIMENT_PLAN_058.md`)."""

    boson_p: float = DEFAULT_NGRAM_BOSON_P
    holo_alpha: float = DEFAULT_NGRAM_HOLO_ALPHA
    kron_blend: float = DEFAULT_NGRAM_KRON_BLEND
    coinc_gamma: float = DEFAULT_NGRAM_COINC_GAMMA
    unigram_temp: float = DEFAULT_NGRAM_UNIGRAM_TEMP
    top_anchor_tokens: int = DEFAULT_TOP_ANCHOR_TOKENS
    direct_bigram_slots: int = DEFAULT_DIRECT_BIGRAM_SLOTS
    hashed_trigram_buckets: int = DEFAULT_HASHED_TRIGRAM_BUCKETS
    spectral_wave_blend: float = DEFAULT_SPECTRAL_WAVE_BLEND
    bigram_dirichlet_beta: float = DEFAULT_BIGRAM_DIRICHLET_BETA
    trigram_dirichlet_beta: float = DEFAULT_TRIGRAM_DIRICHLET_BETA
    ngram_skip_scale: float = DEFAULT_NGRAM_SKIP_SCALE
    ngram_mode: str = "trigram"
    ctx_anchor_weight: float = DEFAULT_CTX_ANCHOR_WEIGHT

    @property
    def ortho_real_dim_per_factor(self) -> int:
        """Real degrees of freedom in the carrier-orthogonal subspace `H_perp` per factor (`2 * (d_f - 1) = 30`)."""
        return 2 * (int(self.factor_dim) - 1)


@dataclass(frozen=True)
class Ngram900MCalibrationData:
    """Cached `900M`-token Unigram, Spectral Basis, Bigram, Skip-Bigram, and Trigram kets (`<= 8` fields)."""

    unigram_counts: torch.Tensor
    spectral_vocab_ortho: torch.Tensor
    vocab_to_bi_slot: torch.Tensor
    bi_slot_ortho_kets: torch.Tensor
    tri_bucket_ortho_kets: torch.Tensor
    bi_factor_op_re_im: torch.Tensor
    tri_factor_op_re_im: torch.Tensor
    total_tokens_scanned: int


_HELMERT_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _get_helmert_zero_sum_basis(factor_dim: int, device: torch.device) -> torch.Tensor:
    """Return deterministic orthonormal `(d_f - 1, d_f)` Helmert basis `H` satisfying `H 1 = 0` and `H H^T = I`."""
    key = (int(factor_dim), str(device))
    cached = _HELMERT_CACHE.get(key)
    if cached is not None:
        return cached
    d_m1 = int(factor_dim) - 1
    h_mat = torch.zeros((d_m1, int(factor_dim)), device=device, dtype=torch.float32)
    for k in range(d_m1):
        denom = math.sqrt(float((k + 1) * (k + 2)))
        h_mat[k, : k + 1] = 1.0 / denom
        h_mat[k, k + 1] = -float(k + 1) / denom
    _HELMERT_CACHE[key] = h_mat
    return h_mat


def embed_real30_to_carrier_ortho_c16(
    w_real30: torch.Tensor, factor_dim: int = DEFAULT_FACTOR_DIM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Isometrically embed `w_real30` of shape `(..., 2*(d_f-1))` into `H_perp subset C^{d_f}` (`<= 8` args).

    Maps `w = [w_re; w_im] in R^{30}` to `E_perp(w) = ((1 + i)/sqrt(2)) * (H_15^T w_re + i H_15^T w_im)`,
    guaranteeing `<c_ref | E_perp(w)>_{C^16} == 0` (`< 1e-7`) and `||E_perp(w)||_{C^16}^2 == ||w||_{R^30}^2`.
    """
    d_m1 = int(factor_dim) - 1
    w_f = w_real30.float()
    h_mat = _get_helmert_zero_sum_basis(factor_dim, w_f.device)
    a_vec = torch.matmul(w_f[..., :d_m1], h_mat)
    b_vec = torch.matmul(w_f[..., d_m1 : 2 * d_m1], h_mat)
    s0 = DEFAULT_REF_PHASE_COS_SIN
    return s0 * (a_vec - b_vec), s0 * (a_vec + b_vec)


def project_c16_to_carrier_ortho_real30(u_re: torch.Tensor, u_im: torch.Tensor) -> torch.Tensor:
    """Adjoint isometric projection `P_perp: C^{d_f} -> R^{2*(d_f-1)}` annihilating `|c_ref>` (`<= 8` args)."""
    ur, ui = u_re.float(), u_im.float()
    h_t = _get_helmert_zero_sum_basis(int(u_re.size(-1)), ur.device).t()
    s0 = DEFAULT_REF_PHASE_COS_SIN
    return torch.cat([torch.matmul(s0 * (ur + ui), h_t), torch.matmul(s0 * (ui - ur), h_t)], dim=-1)


def _project_c16_ket_orthogonal_to_carrier(u_re: torch.Tensor, u_im: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract the `C`-linear component along `1_{d_f}` so `<c_ref | u_perp> == 0` identically (`<= 8` args)."""
    return u_re - u_re.mean(dim=-1, keepdim=True), u_im - u_im.mean(dim=-1, keepdim=True)


def _apply_factor_unitary_bank(
    x_re: torch.Tensor, x_im: torch.Tensor, u_re: torch.Tensor, u_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply batched complex unitary `u_re + i u_im` `(K, d_f, d_f)` to `(..., K, d_f)` kets (`<= 8` args)."""
    y_re = torch.einsum("...kd,kdo->...ko", x_re, u_re) - torch.einsum("...kd,kdo->...ko", x_im, u_im)
    y_im = torch.einsum("...kd,kdo->...ko", x_re, u_im) + torch.einsum("...kd,kdo->...ko", x_im, u_re)
    return y_re, y_im


def compute_trigram_xor_hash_buckets(
    prev_tokens: torch.Tensor, curr_tokens: torch.Tensor, num_buckets: int = DEFAULT_HASHED_TRIGRAM_BUCKETS,
) -> torch.Tensor:
    """Compute polynomial XOR-hash bucket `((x_{t-1} * P1) ^ (x_t * P2)) % H_tri` (`<= 8` args)."""
    mixed = torch.bitwise_xor(
        prev_tokens.to(torch.int64) * TRIGRAM_HASH_PRIME_1, curr_tokens.to(torch.int64) * TRIGRAM_HASH_PRIME_2
    )
    return torch.remainder(mixed.abs(), int(num_buckets))


def _deterministic_orthogonal_matrix(dim: int, seed: int, device: torch.device) -> torch.Tensor:
    """Construct a deterministic real orthogonal matrix `Q in O(dim)` via QR decomposition (`<= 8` args)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    mat = torch.randn((int(dim), int(dim)), generator=gen, dtype=torch.float32).to(device=device)
    q_mat, r_mat = torch.linalg.qr(mat)
    diag_r = torch.diagonal(r_mat)
    signs = torch.where(diag_r >= 0.0, torch.ones_like(diag_r), -torch.ones_like(diag_r))
    return (q_mat * signs.unsqueeze(0)).contiguous()


def _fit_complex_unitary_procrustes_operator(
    src_ortho_30: torch.Tensor, dst_ortho_30: torch.Tensor, weights: torch.Tensor, factor_dim: int = DEFAULT_FACTOR_DIM,
) -> torch.Tensor:
    """Fit `4` strictly unitary `U(16)` matrices mapping `E_perp(src)` to phase-aligned `E_perp(dst)` (`<= 8` args)."""
    m_num = int(src_ortho_30.size(1))
    dev = src_ortho_30.device
    w_sqrt = torch.sqrt(weights.float().clamp_min(PARTITION_CLAMP_MIN)).view(-1, 1)

    src_re, src_im = embed_real30_to_carrier_ortho_c16(src_ortho_30, factor_dim=factor_dim)
    dst_raw_re, dst_raw_im = embed_real30_to_carrier_ortho_c16(dst_ortho_30, factor_dim=factor_dim)
    s0 = DEFAULT_REF_PHASE_COS_SIN
    dst_re, dst_im = s0 * (dst_raw_re + dst_raw_im), s0 * (dst_raw_im - dst_raw_re)

    op_re = torch.empty((m_num, factor_dim, factor_dim), device=dev, dtype=torch.float32)
    op_im = torch.empty((m_num, factor_dim, factor_dim), device=dev, dtype=torch.float32)
    eye_c = torch.eye(factor_dim, device=dev, dtype=torch.complex64)

    for m in range(m_num):
        x_c = torch.complex(src_re[:, m] * w_sqrt, src_im[:, m] * w_sqrt)
        y_c = torch.complex(dst_re[:, m] * w_sqrt, dst_im[:, m] * w_sqrt)
        u_svd, _, vh_svd = torch.linalg.svd(x_c.conj().t() @ y_c + DEFAULT_PROCRUSTES_RIDGE * eye_c, full_matrices=False)
        u_polar = u_svd @ vh_svd
        op_re[m], op_im[m] = u_polar.real, u_polar.imag
    return torch.stack([op_re, op_im], dim=0).contiguous()


def _open_token_source(item: Path | str | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Open `.bin` shard as header-safe `uint16` `np.memmap` or return in-memory array/tensor."""
    if isinstance(item, (str, Path)):
        p = Path(item)
        fsize = p.stat().st_size
        offset = 1024 if fsize >= 200_000_000 else 0
        if offset == 0 and fsize >= 1024:
            hdr = np.fromfile(p, dtype=np.int32, count=256)
            if hdr.shape[0] == 256 and (int(hdr[0]) == 20240520 or bool(np.all(hdr[4:] == 0))):
                offset = 1024
        return np.memmap(p, dtype=np.uint16, mode="r", offset=offset)
    return item


def _load_source_slice_to_gpu(
    source: Path | str | np.ndarray | torch.Tensor, start_idx: int, end_idx: int,
    valid_vocab_size: int, device: torch.device,
) -> torch.Tensor:
    """Load token slice `[start_idx:end_idx]` from disk shard or array directly onto `device` as `int64`."""
    arr = _open_token_source(source)
    v_max = int(valid_vocab_size) - 1
    if isinstance(arr, np.ndarray):
        sub = np.array(arr.reshape(-1)[start_idx:end_idx], copy=True)
        if sub.dtype == np.uint16:
            return (torch.from_numpy(sub.view(np.int16)).to(device=device).to(torch.int64) & 0xFFFF).clamp_(0, v_max)
        return torch.from_numpy(sub.astype(np.int64, copy=False)).to(device=device).clamp_(0, v_max)
    return arr.reshape(-1)[start_idx:end_idx].to(device=device, dtype=torch.int64).clamp(0, v_max)


def _iter_token_chunks_from_sources(
    shard_paths: list[Path | np.ndarray | torch.Tensor],
) -> list[tuple[np.ndarray | torch.Tensor, int]]:
    """Build index of `(opened_source, num_tokens)` for fast streaming across `.bin` shards or arrays."""
    sources: list[tuple[np.ndarray | torch.Tensor, int]] = []
    for item in shard_paths:
        if isinstance(item, (str, Path)) and not Path(item).exists():
            continue
        opened = _open_token_source(item)
        sources.append((opened, int(opened.size) if isinstance(opened, np.ndarray) else int(opened.numel())))
    return sources


def _run_pass1_unigram_and_spectral_svd(
    sources: list[tuple[np.ndarray | torch.Tensor, int]], cfg: NgramCalibratedKroneckerConfig, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Execute Pass 1: accumulate `900M` unigram + anchor bigram/skip-bigram counts and SVD basis (`<= 8` args)."""
    v_eff = int(cfg.valid_vocab_size)
    k_anchor, k_bi = min(int(cfg.top_anchor_tokens), v_eff), min(int(cfg.direct_bigram_slots), v_eff)
    m_num, d_ortho = int(cfg.num_factors), int(cfg.ortho_real_dim_per_factor)
    q_svd = m_num * d_ortho

    unigram_counts = torch.zeros((v_eff,), device=device, dtype=torch.float64)
    total_tokens = 0
    for src, n_tok in sources:
        for s0 in range(0, n_tok, DEFAULT_SHARD_SCAN_CHUNK_TOKENS):
            chunk = _load_source_slice_to_gpu(src, s0, min(n_tok, s0 + DEFAULT_SHARD_SCAN_CHUNK_TOKENS), v_eff, device)
            unigram_counts.add_(torch.bincount(chunk, minlength=v_eff).to(torch.float64))
            total_tokens += int(chunk.numel())

    sorted_vocab = torch.argsort(unigram_counts, descending=True)
    top_anchors, top_bi_tokens = sorted_vocab[:k_anchor], sorted_vocab[:k_bi]
    vocab_to_anchor = torch.full((v_eff,), -1, device=device, dtype=torch.int64)
    vocab_to_anchor[top_anchors] = torch.arange(k_anchor, device=device, dtype=torch.int64)
    vocab_to_bi_slot = torch.zeros((v_eff,), device=device, dtype=torch.int64)
    vocab_to_bi_slot[top_bi_tokens] = torch.arange(1, k_bi + 1, device=device, dtype=torch.int64)

    c_bi = torch.zeros((k_anchor * v_eff,), device=device, dtype=torch.float32)
    c_skip1 = torch.zeros((k_anchor * v_eff,), device=device, dtype=torch.float32)
    ones_buf: torch.Tensor | None = None

    for src, n_tok in sources:
        for s0 in range(0, n_tok, DEFAULT_SHARD_SCAN_CHUNK_TOKENS):
            chunk = _load_source_slice_to_gpu(src, s0, min(n_tok, s0 + DEFAULT_SHARD_SCAN_CHUNK_TOKENS), v_eff, device)
            if chunk.numel() >= 2:
                a_bi = vocab_to_anchor[chunk[:-1]]
                mask_bi = a_bi >= 0
                idx_bi = a_bi[mask_bi] * v_eff + chunk[1:][mask_bi]
                if ones_buf is None or ones_buf.numel() < idx_bi.numel():
                    ones_buf = torch.ones((max(idx_bi.numel(), 1024),), device=device, dtype=torch.float32)
                c_bi.scatter_add_(0, idx_bi, ones_buf[: idx_bi.numel()])
            if chunk.numel() >= 3:
                a_sk = vocab_to_anchor[chunk[:-2]]
                mask_sk = a_sk >= 0
                idx_sk = a_sk[mask_sk] * v_eff + chunk[2:][mask_sk]
                if ones_buf is None or ones_buf.numel() < idx_sk.numel():
                    ones_buf = torch.ones((max(idx_sk.numel(), 1024),), device=device, dtype=torch.float32)
                c_skip1.scatter_add_(0, idx_sk, ones_buf[: idx_sk.numel()])

    p_uni = ((unigram_counts + 1.0) / float(unigram_counts.sum().item() + float(v_eff))).to(torch.float32)
    c_bi_2d, c_sk_2d = c_bi.view(k_anchor, v_eff), c_skip1.view(k_anchor, v_eff)
    beta_bi, inv_2p = float(cfg.bigram_dirichlet_beta), 1.0 / (2.0 * max(1.0, float(cfg.boson_p)))

    p_bi_cond = (c_bi_2d + beta_bi * p_uni.unsqueeze(0)) / (c_bi_2d.sum(dim=1, keepdim=True) + beta_bi)
    p_sk_cond = (c_sk_2d + beta_bi * p_uni.unsqueeze(0)) / (c_sk_2d.sum(dim=1, keepdim=True) + beta_bi)
    anchor_w = torch.pow(p_uni[top_anchors], DEFAULT_ANCHOR_FREQ_EXP).unsqueeze(1)
    s_bi = anchor_w * (torch.pow((p_bi_cond / p_uni.unsqueeze(0)).clamp_min(1e-8), inv_2p) - 1.0)
    s_sk = anchor_w * (torch.pow((p_sk_cond / p_uni.unsqueeze(0)).clamp_min(1e-8), inv_2p) - 1.0)

    s_joint = torch.cat([s_bi.t(), DEFAULT_SKIP1_STACK_WEIGHT * s_sk.t()], dim=1).contiguous()
    s_joint.sub_(s_joint.mean(dim=0, keepdim=True))
    q_eff = min(q_svd, int(s_joint.size(0)), int(s_joint.size(1)))
    torch.manual_seed(1337)
    u_svd, s_val, _ = torch.svd_lowrank(s_joint, q=q_eff, niter=4)
    s_rel = s_val.clamp_min(1e-8) / s_val.max().clamp_min(1e-8)
    u_weighted = u_svd * torch.pow(s_rel.clamp_min(0.20), DEFAULT_SVD_SIGMA_EXP).unsqueeze(0)
    gen_pad = torch.Generator(device=device)
    gen_pad.manual_seed(1337)
    if q_eff < q_svd:
        u_weighted = torch.cat([u_weighted, torch.randn((v_eff, q_svd - q_eff), generator=gen_pad, device=device)], dim=1)

    u_weighted.sub_(u_weighted.mean(dim=0, keepdim=True))
    u_row_unit = u_weighted * torch.rsqrt(u_weighted.square().sum(dim=-1, keepdim=True).clamp_min(1e-8))
    cooc_counts = c_bi_2d.sum(dim=0) + c_sk_2d.sum(dim=0)
    conf_w = torch.sqrt(cooc_counts / (cooc_counts + 16.0)).unsqueeze(1)
    iso_fallback = torch.randn((v_eff, q_svd), generator=gen_pad, device=device, dtype=torch.float32)
    iso_fallback = iso_fallback * torch.rsqrt(iso_fallback.square().sum(dim=-1, keepdim=True).clamp_min(1e-8))
    u_blended = conf_w * u_row_unit + torch.sqrt((1.0 - conf_w.square()).clamp_min(0.0)) * iso_fallback
    u_blended.sub_(u_blended.mean(dim=0, keepdim=True))

    cov_120 = (u_blended.t() @ u_blended) / float(max(1, v_eff))
    evals, evecs = torch.linalg.eigh(cov_120 + 1e-5 * torch.eye(q_svd, device=device, dtype=torch.float32))
    u_polar = u_blended @ (evecs @ torch.diag(torch.rsqrt(evals.clamp_min(1e-5))) @ evecs.t())

    u_mixed = (u_polar @ _deterministic_orthogonal_matrix(q_svd, seed=1338, device=device)).reshape(v_eff, m_num, d_ortho)
    u_mixed.sub_(u_mixed.mean(dim=0, keepdim=True))
    u_norm = u_mixed * torch.rsqrt(u_mixed.square().sum(dim=-1, keepdim=True).clamp_min(1e-8))
    return unigram_counts, u_norm.contiguous(), vocab_to_bi_slot, top_bi_tokens, total_tokens


@torch.no_grad()
def build_or_load_900m_ngram_calibration_cache(
    shard_paths: list[Path | np.ndarray | torch.Tensor], cache_path: Path | str | None,
    cfg: NgramCalibratedKroneckerConfig, device: torch.device, force_rebuild: bool = False,
) -> Ngram900MCalibrationData:
    """Build or load the 2-pass `900M`-token Hilbert-space N-gram calibration cache (`<= 8` args)."""
    c_path = Path(cache_path) if cache_path is not None else None
    v_eff, k_bi = int(cfg.valid_vocab_size), min(int(cfg.direct_bigram_slots), int(cfg.valid_vocab_size))
    h_tri, m_num, d_ortho = int(cfg.hashed_trigram_buckets), int(cfg.num_factors), int(cfg.ortho_real_dim_per_factor)
    d_flat = m_num * d_ortho

    if c_path is not None and c_path.exists() and not force_rebuild:
        p = torch.load(c_path, map_location=device, weights_only=False)
        if int(p["unigram_counts"].numel()) == v_eff and int(p["bi_slot_ortho_kets"].size(0)) == k_bi + 1:
            return Ngram900MCalibrationData(
                p["unigram_counts"].to(device, torch.float64), p["spectral_vocab_ortho"].to(device, torch.float32),
                p["vocab_to_bi_slot"].to(device, torch.int64), p["bi_slot_ortho_kets"].to(device, torch.float32),
                p["tri_bucket_ortho_kets"].to(device, torch.float32), p["bi_factor_op_re_im"].to(device, torch.float32),
                p["tri_factor_op_re_im"].to(device, torch.float32), int(p["total_tokens_scanned"]),
            )

    sources = _iter_token_chunks_from_sources(shard_paths)
    unigram_counts, spectral_vocab_ortho, vocab_to_bi_slot, top_bi_tokens, total_tokens = (
        _run_pass1_unigram_and_spectral_svd(sources, cfg, device)
    )

    p_uni = ((unigram_counts + 1.0) / float(unigram_counts.sum().item() + float(v_eff))).to(torch.float32)
    u_flat = spectral_vocab_ortho.reshape(v_eff, d_flat)
    u_tilde = (u_flat - (p_uni.unsqueeze(1) * u_flat).sum(dim=0, keepdim=True)).contiguous()
    sigma_u = (u_tilde * p_uni.unsqueeze(1)).t() @ u_tilde
    w_whiten = torch.linalg.inv(sigma_u + DEFAULT_WHITENING_RIDGE * torch.eye(d_flat, device=device, dtype=torch.float32))
    u_whitened = (u_tilde @ w_whiten).contiguous()

    bi_sum = torch.zeros((k_bi + 1, d_flat), device=device, dtype=torch.float32)
    bi_cnt = torch.zeros((k_bi + 1,), device=device, dtype=torch.float32)
    sk_sum = torch.zeros((k_bi + 1, d_flat), device=device, dtype=torch.float32)
    sk_cnt = torch.zeros((k_bi + 1,), device=device, dtype=torch.float32)
    tri_sum = torch.zeros((h_tri, d_flat), device=device, dtype=torch.float32)
    tri_cnt = torch.zeros((h_tri,), device=device, dtype=torch.float32)

    for src, n_tok in sources:
        for s0 in range(0, n_tok, DEFAULT_SHARD_SCAN_CHUNK_TOKENS):
            chunk = _load_source_slice_to_gpu(src, s0, min(n_tok, s0 + DEFAULT_SHARD_SCAN_CHUNK_TOKENS), v_eff, device)
            if chunk.numel() >= 2:
                slots_bi = vocab_to_bi_slot[chunk[:-1]]
                u_y = u_whitened.index_select(0, chunk[1:])
                bi_cnt.add_(torch.bincount(slots_bi, minlength=k_bi + 1).to(torch.float32))
                bi_sum.index_add_(0, slots_bi, u_y)
                if chunk.numel() >= 3:
                    x_prev, x_curr = chunk[:-2], chunk[1:-1]
                    u_y_tri = u_y[1:]
                    slots_sk = vocab_to_bi_slot[x_prev]
                    sk_cnt.add_(torch.bincount(slots_sk, minlength=k_bi + 1).to(torch.float32))
                    sk_sum.index_add_(0, slots_sk, u_y_tri)
                    buckets = compute_trigram_xor_hash_buckets(x_prev, x_curr, num_buckets=h_tri)
                    tri_cnt.add_(torch.bincount(buckets, minlength=h_tri).to(torch.float32))
                    tri_sum.index_add_(0, buckets, u_y_tri)
                del u_y

    beta_bi, beta_tri = float(cfg.bigram_dirichlet_beta), float(cfg.trigram_dirichlet_beta)
    bi_Slot4d = (bi_sum / (bi_cnt.unsqueeze(1) + beta_bi)).reshape(k_bi + 1, m_num, d_ortho)
    sk_Slot4d = (sk_sum / (sk_cnt.unsqueeze(1) + beta_bi)).reshape(k_bi + 1, m_num, d_ortho)
    tri_Bucket4d = (tri_sum / (tri_cnt.unsqueeze(1) + beta_tri)).reshape(h_tri, m_num, d_ortho)

    bi_rms = torch.sqrt(bi_Slot4d[1:].square().sum(dim=-1).mean().clamp_min(1e-8))
    sk_rms = torch.sqrt(sk_Slot4d[1:].square().sum(dim=-1).mean().clamp_min(1e-8))
    tri_rms = torch.sqrt(tri_Bucket4d.square().sum(dim=-1).mean().clamp_min(1e-8))

    bi_slot_ortho = (bi_Slot4d / bi_rms).clamp(-3.5, 3.5).contiguous()
    sk_slot_ortho = (sk_Slot4d / sk_rms).clamp(-3.5, 3.5).contiguous()
    tri_bucket_ortho = (tri_Bucket4d / tri_rms).clamp(-3.5, 3.5).contiguous()

    src_top, w_top = spectral_vocab_ortho.index_select(0, top_bi_tokens), p_uni.index_select(0, top_bi_tokens)
    bi_op = _fit_complex_unitary_procrustes_operator(src_top, bi_slot_ortho[1 : k_bi + 1], w_top, cfg.factor_dim)
    tri_op = _fit_complex_unitary_procrustes_operator(src_top, sk_slot_ortho[1 : k_bi + 1], w_top, cfg.factor_dim)

    cache_obj = Ngram900MCalibrationData(
        unigram_counts.detach(), spectral_vocab_ortho.detach(), vocab_to_bi_slot.detach(),
        bi_slot_ortho.detach(), tri_bucket_ortho.detach(), bi_op.detach(), tri_op.detach(), int(total_tokens),
    )
    if c_path is not None:
        c_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: getattr(cache_obj, k).cpu() if isinstance(getattr(cache_obj, k), torch.Tensor) else getattr(cache_obj, k) for k in cache_obj.__dataclass_fields__}, c_path)
    return cache_obj


class NgramCalibratedKroneckerBornGPT(CalibratedKroneckerBornGPT):
    """Flag-Free Rank-`R` `16^4 = 65,536` Kronecker Wavefunction GPT with `900M`-Token N-Gram Readout (`Wave 43`)."""

    cfg: NgramCalibratedKroneckerConfig

    def __init__(self, cfg: NgramCalibratedKroneckerConfig) -> None:
        nn.Module.__init__(self)
        self.cfg = cfg
        kron_cfg = cfg.to_kronecker_config()
        v_sz, r_num, m_num, d_f = cfg.vocab_size, cfg.num_meanings, cfg.num_factors, cfg.factor_dim
        k_tot, k_bi, h_tri = r_num * m_num, min(int(cfg.direct_bigram_slots), int(cfg.valid_vocab_size)), int(cfg.hashed_trigram_buckets)

        self.embed_re = nn.Parameter(torch.empty(v_sz, r_num, m_num, d_f))
        self.embed_im = nn.Parameter(torch.empty(v_sz, r_num, m_num, d_f))
        self.branch_logits = nn.Parameter(torch.zeros(v_sz, r_num))
        self.povm_scale_logits = nn.Parameter(torch.zeros(v_sz, dtype=torch.float32))
        self.bi_ctx_wave_re = nn.Parameter(torch.zeros(k_bi + 1, m_num, d_f))
        self.bi_ctx_wave_im = nn.Parameter(torch.zeros(k_bi + 1, m_num, d_f))
        self.tri_ctx_wave_re = nn.Parameter(torch.zeros(h_tri, m_num, d_f))
        self.tri_ctx_wave_im = nn.Parameter(torch.zeros(h_tri, m_num, d_f))

        for uname in ("out_factor_u", "bi_factor_u", "tri_factor_u", "smear_u"):
            setattr(self, f"{uname}_re", nn.Parameter(torch.empty(k_tot, d_f, d_f)))
            setattr(self, f"{uname}_im", nn.Parameter(torch.empty(k_tot, d_f, d_f)))

        self.ngram_branch_gates = nn.Parameter(torch.zeros(r_num, 3, dtype=torch.float32))
        self.smear_gate_logit = nn.Parameter(torch.tensor(DEFAULT_SMEAR_GATE_INIT, dtype=torch.float32))
        self.context_branch_bias = nn.Parameter(torch.zeros(r_num, dtype=torch.float32))
        self.psi0_readout_gate = nn.Parameter(torch.tensor(DEFAULT_PSI0_GATE_INIT, dtype=torch.float32))
        if cfg.self_echo_mode == "ortho_complex_learned":
            init_u = _self_echo_lambda_to_logit(cfg.self_echo_lambda_init)
            self.echo_gate_logits = nn.Parameter(torch.full((r_num, m_num), init_u, dtype=torch.float32))
        else:
            self.register_parameter("echo_gate_logits", None)

        self.register_buffer("initial_povm_logits", torch.zeros(v_sz, dtype=torch.float32))
        self.register_buffer("initial_bi_ctx_re", torch.zeros(k_bi + 1, m_num, d_f))
        self.register_buffer("initial_bi_ctx_im", torch.zeros(k_bi + 1, m_num, d_f))
        self.register_buffer("branch_ortho_rot", torch.eye(d_f).view(1, 1, d_f, d_f).repeat(r_num, m_num, 1, 1))
        self.register_buffer("unigram_probs", torch.full((v_sz,), 1.0 / float(cfg.valid_vocab_size)))
        self.register_buffer("vocab_to_bi_slot", torch.zeros(v_sz, dtype=torch.int64))

        self.blocks = nn.ModuleList([KroneckerComplexRegisterBlock(kron_cfg, i) for i in range(cfg.num_layers)])
        self.kron_mixers = nn.ModuleList([Kronecker16x4UnitaryMeaningMixer(kron_cfg) for _ in range(cfg.num_layers)])
        self._active_ngram_mode, self._training_progress = str(cfg.ngram_mode), 0.0
        self.initialize_isotropic_kronecker(seed=1337)

    @torch.no_grad()
    def initialize_isotropic_kronecker(self, seed: int = 1337) -> dict[str, float]:
        """Initialize `(V, R, 4, 16)` factor kets isotropically on `C^16` and all unitary banks in `U(16)`."""
        v_eff, r_num = int(self.cfg.valid_vocab_size), int(self.cfg.num_meanings)
        m_num, d_f = int(self.cfg.num_factors), int(self.cfg.factor_dim)
        k_tot, dev = r_num * m_num, self.embed_re.device
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed))

        raw_re = torch.randn((self.cfg.vocab_size, r_num, m_num, d_f), generator=gen, device=dev, dtype=torch.float32)
        raw_im = torch.randn((self.cfg.vocab_size, r_num, m_num, d_f), generator=gen, device=dev, dtype=torch.float32)
        inv_norm = torch.rsqrt(raw_re.square().sum(dim=-1, keepdim=True) + raw_im.square().sum(dim=-1, keepdim=True) + DEFAULT_KRON_EPS)
        self.embed_re.data.copy_((raw_re * inv_norm).to(dtype=self.embed_re.dtype))
        self.embed_im.data.copy_((raw_im * inv_norm).to(dtype=self.embed_im.dtype))

        for t_zero in (
            self.branch_logits.data, self.povm_scale_logits.data, self.initial_povm_logits,
            self.bi_ctx_wave_re.data, self.bi_ctx_wave_im.data, self.tri_ctx_wave_re.data,
            self.tri_ctx_wave_im.data, self.initial_bi_ctx_re, self.initial_bi_ctx_im, self.context_branch_bias.data,
        ):
            t_zero.zero_()
        self.psi0_readout_gate.data.fill_(DEFAULT_PSI0_GATE_INIT)
        self.smear_gate_logit.data.fill_(DEFAULT_SMEAR_GATE_INIT)
        if self.echo_gate_logits is not None:
            self.echo_gate_logits.data.fill_(_self_echo_lambda_to_logit(self.cfg.self_echo_lambda_init))

        h_mat, c_proj = _get_helmert_zero_sum_basis(d_f, dev), torch.full((d_f, d_f), 1.0 / float(d_f), device=dev)
        for r in range(r_num):
            for m in range(m_num):
                if r == 0:
                    self.branch_ortho_rot[r, m].copy_(torch.eye(d_f, device=dev, dtype=torch.float32))
                else:
                    o_15 = _deterministic_orthogonal_matrix(d_f - 1, seed=2000 + 17 * r + m, device=dev)
                    self.branch_ortho_rot[r, m].copy_(h_mat.t() @ o_15 @ h_mat + c_proj)

        g_init = torch.tensor([DEFAULT_NGRAM_GATE_BI_INIT, DEFAULT_NGRAM_GATE_TRI_INIT, DEFAULT_NGRAM_GATE_OP_INIT], device=dev)
        self.ngram_branch_gates.data.copy_(g_init.unsqueeze(0).repeat(r_num, 1))
        if r_num > 1:
            self.ngram_branch_gates.data[1:] *= DEFAULT_NGRAM_SECONDARY_BRANCH_SCALE

        for mod in self.modules():
            if isinstance(mod, ComplexLinear) and mod.in_features == mod.out_features:
                _init_square_complex_linear_haar(mod, gen=gen)
        for mixer in self.kron_mixers:
            mixer.reset_parameters(gen=gen)
        for u_re_p, u_im_p in (
            (self.out_factor_u_re, self.out_factor_u_im), (self.bi_factor_u_re, self.bi_factor_u_im),
            (self.tri_factor_u_re, self.tri_factor_u_im), (self.smear_u_re, self.smear_u_im),
        ):
            ur, ui = _sample_haar_complex_unitary_batch(k_tot, d_f, dev, gen=gen)
            u_re_p.data.copy_(ur.to(dtype=u_re_p.dtype))
            u_im_p.data.copy_(ui.to(dtype=u_im_p.dtype))

        return {
            "sharp_eff_rank": self.compute_sharp_eff_rank(), "mean_factor_rank": self.compute_mean_factor_rank(),
            "cross_branch_overlap": self.compute_cross_branch_overlap(), "valid_vocab_size": float(v_eff),
        }

    @torch.no_grad()
    def compute_all_unitary_frobenius_error(self) -> float:
        """Compute mean `||U_k U_k^H - I_16||_F` across `U_out`, `U_bi`, `U_tri`, and `U_smear`."""
        d_f = int(self.cfg.factor_dim)
        eye = torch.eye(d_f, device=self.out_factor_u_re.device, dtype=torch.complex64).unsqueeze(0)
        errs: list[torch.Tensor] = []
        for ur, ui in (
            (self.out_factor_u_re, self.out_factor_u_im), (self.bi_factor_u_re, self.bi_factor_u_im),
            (self.tri_factor_u_re, self.tri_factor_u_im), (self.smear_u_re, self.smear_u_im),
        ):
            u_c = torch.complex(ur.detach().float(), ui.detach().float())
            errs.append(torch.linalg.matrix_norm(torch.bmm(u_c, u_c.conj().transpose(-1, -2)) - eye))
        return float(torch.cat(errs, dim=0).mean().item())

    def _pack_smeared_token_wavefunctions(
        self, idx_clamped: torch.Tensor, branch_alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Look up token factor kets, apply causal 1-token look-back unitary smear, and pack (`<= 8` args)."""
        b_sz, seq_t = idx_clamped.shape
        r_num, m_num, d_f = int(self.cfg.num_meanings), int(self.cfg.num_factors), int(self.cfg.factor_dim)
        k_tot, d_c = r_num * m_num, int(self.cfg.d_complex_total)
        flat_idx = idx_clamped.reshape(-1)

        raw_re = self.embed_re.index_select(0, flat_idx).view(b_sz, seq_t, r_num, m_num, d_f)
        raw_im = self.embed_im.index_select(0, flat_idx).view(b_sz, seq_t, r_num, m_num, d_f)
        f_re, f_im = complex_l2_normalize(raw_re, raw_im, dim=-1)

        if self._active_ngram_mode in ("bigram", "trigram") and seq_t > 1:
            prev_re, prev_im = torch.zeros_like(f_re), torch.zeros_like(f_im)
            prev_re[:, 1:], prev_im[:, 1:] = f_re[:, :-1], f_im[:, :-1]
            sm_k_re, sm_k_im = _apply_factor_unitary_bank(
                prev_re.view(b_sz, seq_t, k_tot, d_f), prev_im.view(b_sz, seq_t, k_tot, d_f),
                self.smear_u_re.to(dtype=f_re.dtype), self.smear_u_im.to(dtype=f_im.dtype),
            )
            u_sm = self.smear_gate_logit.float()
            g_sm = (0.5 * (1.0 + u_sm * torch.rsqrt(u_sm.square() + 1.0))).to(dtype=f_re.dtype)
            f_sm_re, f_sm_im = complex_l2_normalize(
                f_re + g_sm * sm_k_re.view(b_sz, seq_t, r_num, m_num, d_f),
                f_im + g_sm * sm_k_im.view(b_sz, seq_t, r_num, m_num, d_f), dim=-1,
            )
        else:
            f_sm_re, f_sm_im = f_re, f_im

        alpha_tok = branch_alpha.index_select(0, flat_idx).view(b_sz, seq_t, r_num, 1, 1)
        scale_tok = (alpha_tok * (1.0 / math.sqrt(float(m_num)))).to(dtype=f_re.dtype)
        return (
            (f_sm_re * scale_tok).reshape(b_sz, seq_t, d_c), (f_sm_im * scale_tok).reshape(b_sz, seq_t, d_c),
            (f_re * scale_tok).reshape(b_sz, seq_t, d_c), (f_im * scale_tok).reshape(b_sz, seq_t, d_c),
        )

    def _compute_ngram_carrier_ortho_delta(
        self, idx_clamped: torch.Tensor, out_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute causal Bigram + Trigram readout ket `|Delta_phi(x_{t-1}, x_t)> in H_perp` (`<= 8` args)."""
        b_sz, seq_t = idx_clamped.shape
        n_tok = b_sz * seq_t
        r_num, m_num, d_f = int(self.cfg.num_meanings), int(self.cfg.num_factors), int(self.cfg.factor_dim)
        k_tot, mode = r_num * m_num, self._active_ngram_mode

        if mode == "none" or float(self.cfg.ngram_skip_scale) == 0.0:
            z = torch.zeros((n_tok, r_num, m_num, d_f), device=idx_clamped.device, dtype=out_dtype)
            return z, z

        flat_curr = idx_clamped.reshape(-1)
        bi_slots = self.vocab_to_bi_slot.index_select(0, flat_curr)
        rot_rm = self.branch_ortho_rot.float()
        xi_bi_re = torch.einsum("nmd,rmdo->nrmo", self.bi_ctx_wave_re.index_select(0, bi_slots).float(), rot_rm)
        xi_bi_im = torch.einsum("nmd,rmdo->nrmo", self.bi_ctx_wave_im.index_select(0, bi_slots).float(), rot_rm)

        cur_emb_re, cur_emb_im = _project_c16_ket_orthogonal_to_carrier(
            self.embed_re.index_select(0, flat_curr).float(), self.embed_im.index_select(0, flat_curr).float()
        )
        op_bi_re, op_bi_im = _apply_factor_unitary_bank(
            cur_emb_re.reshape(n_tok, k_tot, d_f), cur_emb_im.reshape(n_tok, k_tot, d_f),
            self.bi_factor_u_re.float(), self.bi_factor_u_im.float(),
        )

        gates = self.ngram_branch_gates.float()
        g_bi = gates[:, 0].view(1, r_num, 1, 1)
        g_tri = gates[:, 1].view(1, r_num, 1, 1) if mode == "trigram" else 0.0
        g_op = gates[:, 2].view(1, r_num, 1, 1)

        delta_re = g_bi * xi_bi_re + g_op * op_bi_re.reshape(n_tok, r_num, m_num, d_f)
        delta_im = g_bi * xi_bi_im + g_op * op_bi_im.reshape(n_tok, r_num, m_num, d_f)

        if mode == "trigram" and seq_t > 1:
            prev_idx = torch.cat([idx_clamped[:, :1], idx_clamped[:, :-1]], dim=1)
            buckets = compute_trigram_xor_hash_buckets(prev_idx, idx_clamped, int(self.cfg.hashed_trigram_buckets))
            pos_mask = torch.ones((b_sz, seq_t, 1, 1, 1), device=idx_clamped.device, dtype=torch.float32)
            pos_mask[:, 0] = 0.0
            pos_mask_flat = pos_mask.reshape(n_tok, 1, 1, 1)

            b_flat = buckets.reshape(-1)
            xi_tri_re = torch.einsum("nmd,rmdo->nrmo", self.tri_ctx_wave_re.index_select(0, b_flat).float(), rot_rm)
            xi_tri_im = torch.einsum("nmd,rmdo->nrmo", self.tri_ctx_wave_im.index_select(0, b_flat).float(), rot_rm)

            cur_bt_re = cur_emb_re.view(b_sz, seq_t, r_num, m_num, d_f)
            cur_bt_im = cur_emb_im.view(b_sz, seq_t, r_num, m_num, d_f)
            prv_emb_re = torch.cat([cur_bt_re[:, :1], cur_bt_re[:, :-1]], dim=1)
            prv_emb_im = torch.cat([cur_bt_im[:, :1], cur_bt_im[:, :-1]], dim=1)
            op_tri_re, op_tri_im = _apply_factor_unitary_bank(
                prv_emb_re.reshape(n_tok, k_tot, d_f), prv_emb_im.reshape(n_tok, k_tot, d_f),
                self.tri_factor_u_re.float(), self.tri_factor_u_im.float(),
            )
            w_sk = DEFAULT_SKIP1_STACK_WEIGHT * g_op
            delta_re = delta_re + pos_mask_flat * (g_tri * xi_tri_re + w_sk * op_tri_re.reshape(n_tok, r_num, m_num, d_f))
            delta_im = delta_im + pos_mask_flat * (g_tri * xi_tri_im + w_sk * op_tri_im.reshape(n_tok, r_num, m_num, d_f))

        perp_re, perp_im = _project_c16_ket_orthogonal_to_carrier(delta_re, delta_im)
        scale_s = float(self.cfg.ngram_skip_scale)
        return (scale_s * perp_re).to(dtype=out_dtype), (scale_s * perp_im).to(dtype=out_dtype)

    def build_calibrated_readout_factors(
        self, psi_re: torch.Tensor, psi_im: torch.Tensor, psi0_re: torch.Tensor,
        psi0_im: torch.Tensor, idx_clamped: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply `out_factor_u`, carrier-orthogonal self-echo gating, and `H_perp` N-gram readout injection (`<= 8` args)."""
        b_sz, seq_t, _ = psi_re.shape
        n_tok = b_sz * seq_t
        r_num, m_num, d_f = int(self.cfg.num_meanings), int(self.cfg.num_factors), int(self.cfg.factor_dim)
        k_tot = r_num * m_num

        g0 = torch.tanh(self.psi0_readout_gate).to(dtype=psi_re.dtype)
        p0_re, p0_im = psi0_re.to(dtype=psi_re.dtype), psi0_im.to(dtype=psi_im.dtype)
        mixed_re, mixed_im = complex_l2_normalize(psi_re + g0 * p0_re, psi_im - g0 * p0_im, dim=-1)

        op_k_re, op_k_im = _apply_factor_unitary_bank(
            mixed_re.reshape(n_tok, k_tot, d_f), mixed_im.reshape(n_tok, k_tot, d_f),
            self.out_factor_u_re.to(dtype=mixed_re.dtype), self.out_factor_u_im.to(dtype=mixed_re.dtype),
        )
        op_re, op_im = op_k_re.reshape(n_tok, r_num, m_num, d_f), op_k_im.reshape(n_tok, r_num, m_num, d_f)
        p0_4d_re, p0_4d_im = p0_re.reshape(n_tok, r_num, m_num, d_f), p0_im.reshape(n_tok, r_num, m_num, d_f)
        raw_phi_re, raw_phi_im, _ = apply_carrier_orthogonal_complex_self_echo(
            op_re, op_im, p0_4d_re, p0_4d_im, mode=self.cfg.self_echo_mode, echo_gate_logits=self.echo_gate_logits,
        )

        ng_re, ng_im = self._compute_ngram_carrier_ortho_delta(idx_clamped, out_dtype=raw_phi_re.dtype)
        comb_re, comb_im = raw_phi_re + ng_re, raw_phi_im + ng_im
        branch_beta = self.compute_context_branch_beta(comb_re, comb_im)
        phi_re, phi_im = complex_l2_normalize(comb_re, comb_im, dim=-1)
        return torch.cat([phi_re, phi_im], dim=-1), branch_beta

    def _run_smeared_backbone(
        self, idx_clamped: torch.Tensor, branch_alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run smeared Layer-0 + `blocks` + `kron_mixers` and return `(psi_re, psi_im, psi0_re, psi0_im)`."""
        b_sz, seq_t = idx_clamped.shape
        h_num, dh = int(self.cfg.n_registers), int(self.cfg.d_register)
        psi_re, psi_im, psi0_re, psi0_im = self._pack_smeared_token_wavefunctions(idx_clamped, branch_alpha)
        v0_re: torch.Tensor | None = None
        v0_im: torch.Tensor | None = None
        use_ckpt = bool(self.cfg.checkpoint_blocks and self.training and torch.is_grad_enabled() and not torch.compiler.is_compiling())
        for l_idx, (block, mixer) in enumerate(zip(self.blocks, self.kron_mixers)):
            if use_ckpt:
                psi_re, psi_im, vr, vi = checkpoint(block, psi_re, psi_im, v0_re, v0_im, use_reentrant=False)
            else:
                psi_re, psi_im, vr, vi = block(psi_re, psi_im, v0_re=v0_re, v0_im=v0_im)
            if l_idx == 0:
                v0_re, v0_im = vr.view(b_sz, h_num, seq_t, dh), vi.view(b_sz, h_num, seq_t, dh)
            psi_re, psi_im = mixer(psi_re, psi_im)
        return psi_re, psi_im, psi0_re, psi0_im

    @torch.no_grad()
    def _collect_backbone_pre_operator_states(
        self, cal_seqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run `cal_seqs` `(B_cal, T_cal)` through smeared Layer-0 + `blocks` + `kron_mixers` (`<= 8` args)."""
        b_sz, seq_t = cal_seqs.shape
        k_tot, d_f = int(self.cfg.num_meanings * self.cfg.num_factors), int(self.cfg.factor_dim)
        idx_clamped = cal_seqs.clamp(min=0, max=int(self.cfg.valid_vocab_size) - 1)
        dev_type = cal_seqs.device.type
        with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=(dev_type == "cuda")):
            psi_re, psi_im, psi0_re, psi0_im = self._run_smeared_backbone(idx_clamped, self.compute_branch_alpha())
        g0 = torch.tanh(self.psi0_readout_gate).to(dtype=psi_re.dtype)
        mixed_re, mixed_im = complex_l2_normalize(
            psi_re + g0 * psi0_re.to(dtype=psi_re.dtype), psi_im - g0 * psi0_im.to(dtype=psi_im.dtype), dim=-1
        )
        return (
            mixed_re.reshape(b_sz * seq_t, k_tot, d_f).float(), mixed_im.reshape(b_sz * seq_t, k_tot, d_f).float(),
            psi_re.float(), psi_im.float(), psi0_re.float(), psi0_im.float(),
        )

    @torch.no_grad()
    def _initialize_spectral_wavefunctions_and_tables_from_cache(
        self, cache: Ngram900MCalibrationData, rho_eff: float, seed: int = 1337,
    ) -> None:
        """Embed `900M`-token spectral coordinates, context kets, and `U(16)` operators into model (`<= 8` args)."""
        v_eff, r_num = int(self.cfg.valid_vocab_size), int(self.cfg.num_meanings)
        m_num, d_f = int(self.cfg.num_factors), int(self.cfg.factor_dim)
        d_ortho = int(self.cfg.ortho_real_dim_per_factor)
        k_bi, dev = min(int(self.cfg.direct_bigram_slots), v_eff), self.embed_re.device

        counts_f = torch.clamp_min(cache.unigram_counts[:v_eff].to(device=dev, dtype=torch.float64), 1.0)
        self.unigram_probs.zero_()
        self.unigram_probs[:v_eff].copy_((counts_f / counts_f.sum().clamp_min(DEFAULT_KRON_EPS)).to(torch.float32))
        self.vocab_to_bi_slot.zero_()
        self.vocab_to_bi_slot[:v_eff].copy_(cache.vocab_to_bi_slot[:v_eff].to(device=dev))

        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed) + 101)
        total_ortho_dim = r_num * m_num * d_ortho
        rand_raw = torch.randn((v_eff, total_ortho_dim), generator=gen, device=dev, dtype=torch.float32)
        rand_raw.sub_(rand_raw.mean(dim=0, keepdim=True))
        if v_eff >= total_ortho_dim:
            q_rand, _ = torch.linalg.qr(rand_raw, mode="reduced")
            rand_ortho = (q_rand * math.sqrt(float(v_eff))).reshape(v_eff, r_num, m_num, d_ortho)
        else:
            rand_ortho = rand_raw.reshape(v_eff, r_num, m_num, d_ortho)
        rand_ortho = rand_ortho * torch.rsqrt(rand_ortho.square().sum(dim=-1, keepdim=True).clamp_min(1e-8))

        if self._active_ngram_mode in ("bigram", "trigram"):
            alpha_spec = float(self.cfg.spectral_wave_blend)
            base_re, base_im = embed_real30_to_carrier_ortho_c16(cache.spectral_vocab_ortho[:v_eff].to(dev), factor_dim=d_f)
            for r in range(r_num):
                alpha_r = alpha_spec if r == 0 else (alpha_spec * DEFAULT_NGRAM_SECONDARY_BRANCH_SCALE)
                noise_scale = math.sqrt(max(0.0, 1.0 - alpha_r * alpha_r))
                for m in range(m_num):
                    rot_16 = self.branch_ortho_rot[r, m].float()
                    nr_re, nr_im = embed_real30_to_carrier_ortho_c16(rand_ortho[:, r, m], factor_dim=d_f)
                    u_re, u_im = complex_l2_normalize(
                        alpha_r * (base_re[:, m, :] @ rot_16) + noise_scale * nr_re,
                        alpha_r * (base_im[:, m, :] @ rot_16) + noise_scale * nr_im, dim=-1,
                    )
                    self.embed_re.data[:v_eff, r, m].copy_(u_re.to(dtype=self.embed_re.dtype))
                    self.embed_im.data[:v_eff, r, m].copy_(u_im.to(dtype=self.embed_im.dtype))
        else:
            for r in range(r_num):
                for m in range(m_num):
                    u_re, u_im = embed_real30_to_carrier_ortho_c16(rand_ortho[:, r, m], factor_dim=d_f)
                    self.embed_re.data[:v_eff, r, m].copy_(u_re.to(dtype=self.embed_re.dtype))
                    self.embed_im.data[:v_eff, r, m].copy_(u_im.to(dtype=self.embed_im.dtype))

        if rho_eff > 0.0:
            log_p = torch.log(self.get_target_tempered_unigram().clamp_min(PARTITION_CLAMP_MIN))
            z_raw = (log_p - log_p.mean()) / max(float(log_p.std().item()), DEFAULT_CARRIER_STD_DIVISOR)
            rho_4d = (float(rho_eff) * (z_raw * torch.rsqrt(1.0 + z_raw.square()))).view(v_eff, 1, 1, 1)
            orth_scale = torch.sqrt((1.0 - rho_4d.square()).clamp_min(DEFAULT_KRON_EPS))
            c_re, c_im = _build_canonical_carrier_ket(d_f, dev)
            u_perp_re, u_perp_im = complex_l2_normalize(
                self.embed_re.data[:v_eff].float(), self.embed_im.data[:v_eff].float(), dim=-1
            )
            tilted_re, tilted_im = complex_l2_normalize(
                rho_4d * c_re.view(1, 1, 1, d_f) + orth_scale * u_perp_re,
                rho_4d * c_im.view(1, 1, 1, d_f) + orth_scale * u_perp_im, dim=-1,
            )
            self.embed_re.data[:v_eff].copy_(tilted_re.to(dtype=self.embed_re.dtype))
            self.embed_im.data[:v_eff].copy_(tilted_im.to(dtype=self.embed_im.dtype))

        s0 = DEFAULT_REF_PHASE_COS_SIN
        bi_r, bi_i = embed_real30_to_carrier_ortho_c16(cache.bi_slot_ortho_kets[: k_bi + 1].to(device=dev), factor_dim=d_f)
        bi_rot_re, bi_rot_im = s0 * (bi_r + bi_i), s0 * (bi_i - bi_r)
        self.bi_ctx_wave_re.data.copy_(bi_rot_re.to(dtype=self.bi_ctx_wave_re.dtype))
        self.bi_ctx_wave_im.data.copy_(bi_rot_im.to(dtype=self.bi_ctx_wave_im.dtype))
        self.initial_bi_ctx_re.copy_(bi_rot_re)
        self.initial_bi_ctx_im.copy_(bi_rot_im)

        tri_r, tri_i = embed_real30_to_carrier_ortho_c16(cache.tri_bucket_ortho_kets.to(device=dev), factor_dim=d_f)
        self.tri_ctx_wave_re.data.copy_((s0 * (tri_r + tri_i)).to(dtype=self.tri_ctx_wave_re.dtype))
        self.tri_ctx_wave_im.data.copy_((s0 * (tri_i - tri_r)).to(dtype=self.tri_ctx_wave_im.dtype))

        bi_op = cache.bi_factor_op_re_im.to(device=dev, dtype=torch.float32)
        tri_op = cache.tri_factor_op_re_im.to(device=dev, dtype=torch.float32)
        for r in range(r_num):
            for m in range(m_num):
                k_idx = r * m_num + m
                rot_m = self.branch_ortho_rot[r, m].float()
                rot_t = rot_m.t()
                b_re, b_im = rot_t @ bi_op[0, m] @ rot_m, rot_t @ bi_op[1, m] @ rot_m
                t_re, t_im = rot_t @ tri_op[0, m] @ rot_m, rot_t @ tri_op[1, m] @ rot_m
                self.bi_factor_u_re.data[k_idx].copy_(b_re.to(dtype=self.bi_factor_u_re.dtype))
                self.bi_factor_u_im.data[k_idx].copy_(b_im.to(dtype=self.bi_factor_u_im.dtype))
                self.tri_factor_u_re.data[k_idx].copy_(t_re.to(dtype=self.tri_factor_u_re.dtype))
                self.tri_factor_u_im.data[k_idx].copy_(t_im.to(dtype=self.tri_factor_u_im.dtype))
                self.smear_u_re.data[k_idx].copy_(b_re.to(dtype=self.smear_u_re.dtype))
                self.smear_u_im.data[k_idx].copy_(b_im.to(dtype=self.smear_u_im.dtype))

    @torch.no_grad()
    def calibrate_from_900m_ngram_cache(
        self, cache: Ngram900MCalibrationData, cal_batches: list[torch.Tensor],
        ngram_mode: str = "trigram", carrier_rho_max: float | None = None, calibrate_povm: bool = True,
    ) -> dict[str, float]:
        """Calibrate `embed`, `out_factor_u`, `bi/tri` tables & `U(16)` ops, and `povm_scale_logits` (`<= 8` args)."""
        v_eff, d_f, dev = int(self.cfg.valid_vocab_size), int(self.cfg.factor_dim), self.embed_re.device
        self._active_ngram_mode = str(ngram_mode)
        rho_eff = float(self.cfg.carrier_rho_max if carrier_rho_max is None else carrier_rho_max)
        self._initialize_spectral_wavefunctions_and_tables_from_cache(cache, rho_eff=rho_eff, seed=1337)

        if len(cal_batches) > 0:
            cal_seqs = torch.cat([b.to(device=dev, dtype=torch.int64) for b in cal_batches], dim=0)
        else:
            gen_s = torch.Generator(device=dev)
            gen_s.manual_seed(1354)
            p_s = self.unigram_probs[:v_eff].float().clamp_min(PARTITION_CLAMP_MIN)
            cal_seqs = torch.multinomial(p_s / p_s.sum(), num_samples=16 * 128, replacement=True, generator=gen_s).view(16, 128)

        q_re, q_im, psi_re, psi_im, p0_re, p0_im = self._collect_backbone_pre_operator_states(cal_seqs)
        q_mean_re, q_mean_im = complex_l2_normalize(q_re.mean(dim=0), q_im.mean(dim=0), dim=-1)
        pre_fidelity = self.compute_carrier_alignment_fidelity(q_mean_re, q_mean_im)

        gen_pro = torch.Generator(device=dev)
        gen_pro.manual_seed(1399)
        c_re, c_im = _build_canonical_carrier_ket(d_f, dev)
        u_pro_re, u_pro_im = _build_unitary_procrustes_carrier_batch(q_mean_re, q_mean_im, c_re, c_im, gen=gen_pro)
        self.out_factor_u_re.data.copy_(u_pro_re.to(dtype=self.out_factor_u_re.dtype))
        self.out_factor_u_im.data.copy_(u_pro_im.to(dtype=self.out_factor_u_im.dtype))
        post_fidelity = self.compute_carrier_alignment_fidelity(q_mean_re, q_mean_im)
        unitary_frob_err = self.compute_all_unitary_frobenius_error()

        idx_clamped = cal_seqs.clamp(min=0, max=v_eff - 1)
        phi_factors, branch_beta = self.build_calibrated_readout_factors(psi_re, psi_im, p0_re, p0_im, idx_clamped)
        n_eval = min(int(phi_factors.size(0)), DEFAULT_CALIBRATION_MAX_EVAL_TOKENS)
        e_mat = self._compute_unscaled_born_energies(phi_factors[:n_eval], branch_beta[:n_eval])
        bar_e = self._smooth_carrier_energy_expectation(e_mat.mean(dim=0))
        p_target = self.get_target_tempered_unigram().clamp_min(PARTITION_CLAMP_MIN)

        p_bar_unif = (bar_e / bar_e.sum().clamp_min(PARTITION_CLAMP_MIN)).clamp_min(PARTITION_CLAMP_MIN)
        kl_unif = float(max(0.0, float((p_target * (torch.log(p_target) - torch.log(p_bar_unif))).sum().item())))

        if calibrate_povm:
            a_star = (p_target / bar_e) / (p_target / bar_e).mean().clamp_min(PARTITION_CLAMP_MIN)
            scale_c = max(1.0, (DEFAULT_POVM_MIN_EFF_WEIGHT + DEFAULT_BRANCH_EPS) / max(float(a_star.min().item()), PARTITION_CLAMP_MIN))
            a_eff = (a_star * scale_c - DEFAULT_BRANCH_EPS).clamp_min(DEFAULT_POVM_INV_FLOOR)
            theta_0 = a_eff - (1.0 / (4.0 * a_eff))
            self.povm_scale_logits.data.zero_()
            self.povm_scale_logits.data[:v_eff].copy_(theta_0)
            self.initial_povm_logits.zero_()
            self.initial_povm_logits[:v_eff].copy_(theta_0)
        else:
            self.povm_scale_logits.data.zero_()
            self.initial_povm_logits.zero_()

        p_bar_fin = (self.compute_povm_weights().float() * bar_e).clamp_min(PARTITION_CLAMP_MIN)
        p_bar_fin = p_bar_fin / p_bar_fin.sum().clamp_min(PARTITION_CLAMP_MIN)
        kl_final = float(max(0.0, float((p_target * (torch.log(p_target) - torch.log(p_bar_fin))).sum().item())))

        return {
            "ngram_mode": 0.0 if ngram_mode == "none" else (1.0 if ngram_mode == "bigram" else 2.0),
            "pre_procrustes_carrier_fidelity": round(pre_fidelity, 6), "carrier_alignment_fidelity": round(post_fidelity, 6),
            "unitary_frobenius_err": unitary_frob_err, "uncalibrated_kl_to_target_unigram": round(kl_unif, 6),
            "step0_kl_to_target_unigram": round(kl_final, 6), "carrier_rho_max": round(rho_eff, 4),
            "sharp_eff_rank": self.compute_sharp_eff_rank(), "mean_factor_rank": self.compute_mean_factor_rank(),
            "cross_branch_overlap": self.compute_cross_branch_overlap(), "d_complex_total": float(self.cfg.d_complex_total),
        }

    def extract_context_readout_states(self, idx: torch.Tensor, branch_alpha: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward backbone + N-gram readout injection without normalizing `(V, R, 4, 16)` vocabulary factors (`<= 8` args)."""
        idx_clamped, alpha_eff = idx.clamp(min=0, max=int(self.cfg.valid_vocab_size) - 1), (self.compute_branch_alpha() if branch_alpha is None else branch_alpha)
        psi_re, psi_im, psi0_re, psi0_im = self._run_smeared_backbone(idx_clamped, alpha_eff)
        phi_factors, branch_beta = self.build_calibrated_readout_factors(psi_re, psi_im, psi0_re, psi0_im, idx_clamped)
        return phi_factors, branch_beta, alpha_eff

    def extract_readout_states(self, idx: torch.Tensor, branch_alpha: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward backbone + N-gram readout injection and return `(phi_factors, psi_factors, branch_beta, branch_alpha)`."""
        phi_factors, branch_beta, alpha_eff = self.extract_context_readout_states(idx, branch_alpha=branch_alpha)
        return phi_factors, self.get_tied_vocab_factors(), branch_beta, alpha_eff

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None,
        return_logits: bool | None = None, token_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Run `L` complex register + `16x4` Kronecker mixer layers and N-gram Calibrated Born readout (`<= 8` args)."""
        b_sz, seq_t = idx.shape
        r_num, m_num, d_2 = int(self.cfg.num_meanings), int(self.cfg.num_factors), 2 * int(self.cfg.factor_dim)
        v_eff = int(self.cfg.valid_vocab_size)
        phi_factors, psi_factors, branch_beta, branch_alpha = self.extract_readout_states(idx)
        povm_w = self.compute_povm_weights()
        hyperparams = torch.tensor(
            [self.cfg.boson_p, self.cfg.holo_alpha, self.cfg.kron_blend, self.cfg.coinc_gamma],
            device=idx.device, dtype=torch.float32,
        )
        want_logits = (targets is None) if return_logits is None else bool(return_logits)

        if targets is not None and not want_logits:
            valid_targets = targets.clamp(min=0, max=v_eff - 1).reshape(-1)
            CalibratedKroneckerBornChunkedNLLFn.vocab_chunk_size = int(self.cfg.vocab_chunk_size)
            loss = CalibratedKroneckerBornChunkedNLLFn.apply(
                phi_factors, psi_factors, branch_beta, branch_alpha, povm_w, valid_targets, hyperparams, token_weights
            )
            return (loss if return_logits is None else None), loss

        probs = eval_calibrated_kronecker_born_probs(
            phi_factors.reshape(b_sz, seq_t, r_num, m_num, d_2),
            psi_factors, branch_beta.reshape(b_sz, seq_t, r_num), branch_alpha, povm_w, hyperparams,
        )
        log_probs = torch.log(probs.clamp_min(PROB_LOG_CLAMP_MIN))
        loss = None
        if targets is not None:
            token_nll = -log_probs.gather(-1, targets.clamp(0, v_eff - 1).unsqueeze(-1)).squeeze(-1).reshape(-1)
            tw_f = token_weights.reshape(-1).float() if token_weights is not None else None
            loss = token_nll.mean() if tw_f is None else ((tw_f / tw_f.sum().clamp_min(PROB_LOG_CLAMP_MIN)) * token_nll).sum()
        return (log_probs if want_logits else (loss if (targets is not None and return_logits is None) else None)), loss


@dataclass(frozen=True)
class _EvalChunkedVocabContext:
    """Pre-packed vocabulary evaluation tables hoisted outside the validation batch loop (`<= 8` fields)."""

    w_all: torch.Tensor
    alpha_f: torch.Tensor
    povm_f: torch.Tensor
    coeffs: tuple[float, float, float, float]
    eval_chunk_size: int = DEFAULT_EVAL_VOCAB_CHUNK_SIZE
    compute_top50: bool = False
    mixed_density: bool = False


def _eval_chunked_batch_metrics(
    phi_factors: torch.Tensor, branch_beta: torch.Tensor, targets_flat: torch.Tensor,
    prev_tokens_flat: torch.Tensor, ctx: _EvalChunkedVocabContext,
) -> tuple[float, int, int, float, int, float, int, float, int, int]:
    """Zero-materialization chunked Born NLL, Top-1/Top-50, and self-echo repetition accumulator (`<= 8` args)."""
    n_tok, r_num, m_num, d_2 = phi_factors.shape
    d_f, v_eff, k_tot = d_2 // 2, ctx.alpha_f.size(0), r_num * m_num
    boson_p, holo_alpha, kron_blend, coinc_gamma = ctx.coeffs
    dev, gemm_dtype = phi_factors.device, ctx.w_all.dtype
    phi_w, beta_f = phi_factors.to(dtype=gemm_dtype).contiguous(), branch_beta.float()

    phi_re, phi_im = phi_w[..., :d_f], phi_w[..., d_f:]
    phi_skew_w = torch.cat([-phi_im, phi_re], dim=-1)
    u_batched = torch.cat([phi_w.reshape(n_tok, k_tot, d_2).permute(1, 0, 2), phi_skew_w.reshape(n_tok, k_tot, d_2).permute(1, 0, 2)], dim=1).contiguous()

    z_t = torch.zeros((n_tok,), device=dev, dtype=torch.float32)
    target_p_tilde = torch.zeros((n_tok,), device=dev, dtype=torch.float32)
    prev_p_tilde = torch.zeros((n_tok,), device=dev, dtype=torch.float32)
    best_val, best_idx = torch.full((n_tok,), -1.0, device=dev, dtype=torch.float32), torch.zeros((n_tok,), device=dev, dtype=torch.int64)
    k50 = min(50, v_eff)
    top50_vals = torch.full((n_tok, k50), -1.0, device=dev, dtype=torch.float32) if ctx.compute_top50 else None
    top50_idx = torch.zeros((n_tok, k50), device=dev, dtype=torch.int64) if ctx.compute_top50 else None

    use_comp = bool(phi_factors.is_cuda and gemm_dtype == torch.bfloat16)
    int_fn = (
        (_compiled_mixed_state_eval_intensity if use_comp else _compute_mixed_state_eval_intensity)
        if ctx.mixed_density
        else (_compiled_kronecker_chunk_intensity if use_comp else _compute_kronecker_chunk_intensity)
    )
    eval_chunk = max(MIN_EVAL_VOCAB_CHUNK_SIZE, int(ctx.eval_chunk_size))

    for v0 in range(0, v_eff, eval_chunk):
        v1 = min(v_eff, v0 + eval_chunk)
        vc = v1 - v0
        z_cat = torch.bmm(u_batched, ctx.w_all[:, :, v0:v1])
        x_ov, y_ov = z_cat[:, :n_tok, :].reshape(r_num, m_num, n_tok, vc), z_cat[:, n_tok:, :].reshape(r_num, m_num, n_tok, vc)
        i_tv, _, _ = int_fn(x_ov, y_ov, beta_f, ctx.alpha_f[v0:v1], holo_alpha, kron_blend, coinc_gamma)
        p_tilde_c = ctx.povm_f[v0:v1].unsqueeze(0) * torch.pow(i_tv, boson_p)
        z_t.add_(p_tilde_c.sum(dim=-1))

        in_c = (targets_flat >= v0) & (targets_flat < v1)
        target_p_tilde.add_(p_tilde_c.gather(1, (targets_flat - v0).clamp(0, vc - 1).unsqueeze(1)).squeeze(1) * in_c.to(torch.float32))
        prev_in_c = (prev_tokens_flat >= v0) & (prev_tokens_flat < v1)
        prev_p_tilde.add_(p_tilde_c.gather(1, (prev_tokens_flat - v0).clamp(0, vc - 1).unsqueeze(1)).squeeze(1) * prev_in_c.to(torch.float32))

        c_max, c_arg = p_tilde_c.max(dim=-1)
        better = c_max > best_val
        best_val, best_idx = torch.where(better, c_max, best_val), torch.where(better, c_arg + v0, best_idx)
        if ctx.compute_top50 and top50_vals is not None and top50_idx is not None:
            c_vals, c_idx = torch.topk(p_tilde_c, k=min(k50, vc), dim=-1)
            m_vals, m_idx = torch.cat([top50_vals, c_vals], dim=-1), torch.cat([top50_idx, c_idx + v0], dim=-1)
            top50_vals, sel = torch.topk(m_vals, k=k50, dim=-1)
            top50_idx = m_idx.gather(-1, sel)
        del z_cat, x_ov, y_ov, i_tv, p_tilde_c

    z_clamped = z_t.clamp_min(PARTITION_CLAMP_MIN)
    nll_vec = torch.log(z_clamped) - torch.log(target_p_tilde.clamp_min(PARTITION_CLAMP_MIN))
    self_echo_prob = prev_p_tilde / z_clamped
    is_rep, non_rep = (targets_flat == prev_tokens_flat), (targets_flat != prev_tokens_flat)
    top1_match, pred_rep = (best_idx == targets_flat), (best_idx == prev_tokens_flat)
    top50_hits = int((top50_idx == targets_flat.unsqueeze(1)).any(dim=-1).sum().item()) if top50_idx is not None else 0
    return (
        float(nll_vec.sum().item()), int(top1_match.sum().item()), top50_hits,
        float(nll_vec[is_rep].sum().item()) if bool(is_rep.any()) else 0.0, int(is_rep.sum().item()),
        float(nll_vec[non_rep].sum().item()) if bool(non_rep.any()) else 0.0, int(non_rep.sum().item()),
        float(self_echo_prob.sum().item()), int(pred_rep.sum().item()), int((top1_match & is_rep).sum().item()),
    )


@torch.no_grad()
def evaluate_modded_nanogpt_validation(
    model: NgramCalibratedKroneckerBornGPT, val_tokens: Path | str | np.ndarray | torch.Tensor,
    max_eval_tokens: int = DEFAULT_MODDED_VAL_TOKENS, batch_size: int = 16,
    seq_len: int = 512, compute_top50: bool = False,
    eval_vocab_chunk_size: int = DEFAULT_EVAL_VOCAB_CHUNK_SIZE,
) -> dict[str, float]:
    """Zero-materialization chunked evaluator returning `val_ce`, `val_ppl`, `top1_acc`, and repetition telemetry (`<= 8` args)."""
    was_training = model.training
    model.eval()
    v_eff, dev = int(model.cfg.valid_vocab_size), model.embed_re.device
    r_num, m_num, d_2 = int(model.cfg.num_meanings), int(model.cfg.num_factors), 2 * int(model.cfg.factor_dim)
    k_tot = r_num * m_num

    opened_val = _open_token_source(val_tokens)
    n_avail = (int(opened_val.size) if isinstance(opened_val, np.ndarray) else int(opened_val.numel())) - 1
    t_fixed, b_max = min(max(1, int(seq_len)), max(1, n_avail)), max(1, int(batch_size))
    tokens_per_batch = b_max * t_fixed
    n_target = min(int(max_eval_tokens), (n_avail // t_fixed) * t_fixed)
    if n_target < t_fixed:
        n_target = min(n_avail, int(max_eval_tokens))

    gemm_dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    w_all = model.get_tied_vocab_factors().to(dtype=gemm_dtype).reshape(v_eff, k_tot, d_2).permute(1, 2, 0).contiguous()
    alpha_all = model.compute_branch_alpha().float().contiguous()
    povm_w = model.compute_povm_weights()[:v_eff].float().contiguous()
    coeffs = (float(model.cfg.boson_p), float(model.cfg.holo_alpha), float(model.cfg.kron_blend), float(model.cfg.coinc_gamma))
    is_mixed = str(getattr(model.cfg, "readout_type", "pure_ket")) == "mixed_density"
    ctx = _EvalChunkedVocabContext(w_all, alpha_all, povm_w, coeffs, int(eval_vocab_chunk_size), bool(compute_top50), is_mixed)

    total_nll, total_top1, total_top50, total_tok = 0.0, 0, 0, 0
    rep_nll, rep_cnt, non_rep_nll, non_rep_cnt, echo_prob_sum, pred_rep_cnt, rep_t1_cnt = 0.0, 0, 0.0, 0, 0.0, 0, 0

    for offset in range(0, n_target, tokens_per_batch):
        take = min(tokens_per_batch, n_target - offset)
        b_cur, t_cur = (max(1, take // t_fixed), t_fixed) if take >= t_fixed else (1, take)
        span = b_cur * t_cur
        if span < 1:
            break
        seq = _load_source_slice_to_gpu(opened_val, offset, offset + span + 1, v_eff, dev)
        x_b, y_b = seq[:-1].view(b_cur, t_cur), seq[1:].view(b_cur, t_cur)
        with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
            if hasattr(model, "extract_context_readout_states"):
                phi_factors, branch_beta, _ = model.extract_context_readout_states(x_b, branch_alpha=alpha_all)
            else:
                phi_factors, _, branch_beta, _ = model.extract_readout_states(x_b)
        nll_s, t1_h, t50_h, r_nll, r_c, nr_nll, nr_c, e_sum, pr_h, rt1_h = _eval_chunked_batch_metrics(
            phi_factors, branch_beta, y_b.reshape(-1), x_b.reshape(-1), ctx,
        )
        total_nll, total_top1, total_top50, total_tok = total_nll + nll_s, total_top1 + t1_h, total_top50 + t50_h, total_tok + span
        rep_nll, rep_cnt, non_rep_nll, non_rep_cnt = rep_nll + r_nll, rep_cnt + r_c, non_rep_nll + nr_nll, non_rep_cnt + nr_c
        echo_prob_sum, pred_rep_cnt, rep_t1_cnt = echo_prob_sum + e_sum, pred_rep_cnt + pr_h, rep_t1_cnt + rt1_h

    if was_training:
        model.train()
    val_ce = total_nll / max(1, total_tok)
    rep_ce = (rep_nll / rep_cnt) if rep_cnt > 0 else val_ce
    non_rep_ce = (non_rep_nll / non_rep_cnt) if non_rep_cnt > 0 else val_ce
    res = {
        "val_ce": round(val_ce, 4), "val_ppl": round(math.exp(min(val_ce, 20.0)), 2),
        "top1_acc": round(100.0 * total_top1 / max(1, total_tok), 2), "eval_tokens": float(total_tok),
        "sharp_eff_rank": round(model.compute_sharp_eff_rank(), 2), "mean_factor_rank": round(model.compute_mean_factor_rank(), 2),
        "cross_branch_overlap": round(model.compute_cross_branch_overlap(), 6), "povm_drift_std": round(model.compute_povm_drift_std(), 4),
        "rep_val_ce": round(rep_ce, 4), "non_rep_val_ce": round(non_rep_ce, 4),
        "mean_self_echo_prob": round(echo_prob_sum / max(1, total_tok), 6),
        "empirical_rep_rate": round(100.0 * pred_rep_cnt / max(1, total_tok), 2),
        "rep_top1_acc": round(100.0 * rep_t1_cnt / max(1, rep_cnt), 2) if rep_cnt > 0 else 0.0,
    }
    if compute_top50:
        res["top50_acc"] = round(100.0 * total_top50 / max(1, total_tok), 2)
    return res


def compute_ngram_calibrated_kronecker_penalties(
    model: NgramCalibratedKroneckerBornGPT,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute Sphere, Unitary (`out/bi/tri/smear`), Isotropy, Branch-Orth, POVM-Anchor, & Context-Anchor penalties."""
    v_eff = int(model.cfg.valid_vocab_size)
    base_penalty, stats = compute_kronecker_manifold_and_isotropy_penalties(model)  # type: ignore[arg-type]

    u_bi = _batch_complex_unitary_terms(model.bi_factor_u_re, model.bi_factor_u_im)
    u_tri = _batch_complex_unitary_terms(model.tri_factor_u_re, model.tri_factor_u_im)
    u_sm = _batch_complex_unitary_terms(model.smear_u_re, model.smear_u_im)
    extra_u = torch.cat([u_bi, u_tri, u_sm], dim=0)

    povm_diff = model.povm_scale_logits[:v_eff].float() - model.initial_povm_logits[:v_eff].float()
    povm_anchor_pen = povm_diff.square().mean()

    ctx_diff_re = model.bi_ctx_wave_re.float() - model.initial_bi_ctx_re.float()
    ctx_diff_im = model.bi_ctx_wave_im.float() - model.initial_bi_ctx_im.float()
    tri_energy = (model.tri_ctx_wave_re.float().square() + model.tri_ctx_wave_im.float().square()).mean()
    ctx_anchor_pen = (ctx_diff_re.square() + ctx_diff_im.square()).mean() + 0.25 * tri_energy

    w_u, w_p, w_c = float(model.cfg.unitary_penalty_weight), float(model.cfg.povm_anchor_weight), float(model.cfg.ctx_anchor_weight)
    total_penalty = base_penalty + w_u * extra_u.sum() + w_p * povm_anchor_pen + w_c * ctx_anchor_pen
    stats["unitary_pen"] = float(0.5 * (stats["unitary_pen"] + float(extra_u.mean().detach().item())))
    stats["povm_anchor_pen"] = float(povm_anchor_pen.detach().item())
    stats["ctx_anchor_pen"] = float(ctx_anchor_pen.detach().item())
    stats["total_pen"] = float(total_penalty.detach().item())
    stats["povm_drift_std"] = float(povm_diff.detach().std().item())
    return total_penalty, stats


def build_ngram_calibrated_kronecker_optimizers(
    model: NgramCalibratedKroneckerBornGPT, muon_lr: float = DEFAULT_MUON_LR,
    embed_lr: float = DEFAULT_EMBED_LR, povm_lr: float = DEFAULT_POVM_LR,
    gate_lr: float = DEFAULT_GATE_LR, weight_decay: float = 0.0,
    use_joint_complex_muon: bool = True, batch_tokens: int = DEFAULT_MUON_REFERENCE_BATCH_TOKENS,
) -> tuple[StiefelMuon, torch.optim.AdamW]:
    """Route all 2D/3D complex unitary matrices (`d >= 2`, `out/bi/tri/smear`, backbone) to `StiefelMuon` & 3 groups to `AdamW` (`<= 8` args)."""
    return build_calibrated_kronecker_optimizers(
        model, base_muon_lr=muon_lr, base_embed_lr=embed_lr, base_povm_lr=povm_lr, base_gate_lr=gate_lr,
        weight_decay=weight_decay, use_joint_complex_muon=use_joint_complex_muon, batch_tokens=batch_tokens,
    )


class MultiShardRareTokenSampler:
    """Streams importance-sampled windows across multiple 100M-token FineWeb shards (`<= 8` args per method)."""

    def __init__(
        self, shard_arrays: list[np.ndarray], unigram_counts: torch.Tensor, vocab_size: int = 50304,
        valid_vocab_size: int = 50257, alpha_rare: float = DEFAULT_RARE_SAMPLER_ALPHA,
        pool_mult: int = DEFAULT_RARE_SAMPLER_POOL_MULT,
    ) -> None:
        if len(shard_arrays) == 0:
            raise ValueError("shard_arrays must contain at least one token shard array.")
        self.shard_arrays = shard_arrays
        self.vocab_size, self.valid_vocab_size = int(vocab_size), int(valid_vocab_size)
        self.alpha_rare, self.pool_mult = float(alpha_rare), max(1, int(pool_mult))

        counts_f = torch.clamp_min(unigram_counts[: self.valid_vocab_size].detach().cpu().to(torch.float64), 1.0)
        p_uni = (counts_f / counts_f.sum()).to(torch.float32)
        inv_pow = torch.pow(p_uni.clamp_min(1e-10), -self.alpha_rare) if self.alpha_rare > 0.0 else torch.ones_like(p_uni)
        self._token_rarity_cpu = (inv_pow / inv_pow.mean().clamp_min(1e-8)).contiguous()
        self._token_rarity_dev: dict[str, torch.Tensor] = {}

    def _get_rarity_table(self, device: torch.device) -> torch.Tensor:
        """Return cached token rarity score table `p_uni(v)^{-alpha_rare}` on `device`."""
        key = str(device)
        if key not in self._token_rarity_dev:
            self._token_rarity_dev[key] = self._token_rarity_cpu.to(device=device)
        return self._token_rarity_dev[key]

    def sample_batch(
        self, batch_size: int, seq_len: int, step_seed: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select shard `(step_seed % num_shards)` and draw importance-sampled batch with unbiased weights."""
        shard_np = self.shard_arrays[int(step_seed) % len(self.shard_arrays)]
        win_len = int(seq_len) + 1
        max_start = max(1, int(shard_np.size) - win_len)
        m_pool = max(int(batch_size), self.pool_mult * int(batch_size))
        rng = np.random.default_rng(int(step_seed) * 1009 + 7919)
        starts = rng.integers(0, max_start, size=m_pool)

        raw_win = shard_np.reshape(-1)[starts[:, None] + np.arange(win_len, dtype=np.int64)[None, :]]
        if raw_win.dtype == np.uint16:
            t_i16 = torch.from_numpy(np.ascontiguousarray(raw_win).view(np.int16)).to(device=device)
            win_gpu = (t_i16.to(torch.int64) & 0xFFFF).clamp_(0, self.valid_vocab_size - 1)
        else:
            win_gpu = torch.from_numpy(np.ascontiguousarray(raw_win, dtype=np.int64)).to(device=device).clamp_(
                0, self.valid_vocab_size - 1
            )

        if self.alpha_rare <= 0.0 or m_pool == int(batch_size):
            chosen = win_gpu[: int(batch_size)]
            w_b = torch.ones((int(batch_size), int(seq_len)), device=device, dtype=torch.float32)
            return chosen[:, :-1].contiguous(), chosen[:, 1:].contiguous(), w_b

        scores = self._get_rarity_table(device).index_select(0, win_gpu[:, 1:].reshape(-1)).view(m_pool, int(seq_len)).mean(dim=-1)
        q_probs = scores / scores.sum().clamp_min(1e-12)

        gen = torch.Generator(device=device)
        gen.manual_seed(int(step_seed) * 2027 + 13)
        sel_idx = torch.multinomial(q_probs, num_samples=int(batch_size), replacement=True, generator=gen)

        chosen = win_gpu.index_select(0, sel_idx)
        inv_w = 1.0 / (float(m_pool) * q_probs.index_select(0, sel_idx).clamp_min(1e-12))
        w_b = (inv_w / inv_w.mean().clamp_min(1e-12)).unsqueeze(1).expand(int(batch_size), int(seq_len)).contiguous()
        return chosen[:, :-1].contiguous(), chosen[:, 1:].contiguous(), w_b

