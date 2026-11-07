r"""Orthogonal Vacuum Subspace ($\mathcal{H}_{\text{vac}}$) on the Readout Sphere, Layer-0 Cholesky Initialization, and 4-Mode $512\text{D}$ Cholesky Language Modeling Laboratory (`EXPERIMENT_PLAN_078.md`).

Mathematical & Architectural Motivation
---------------------------------------
1. **Why Unit-Sphere (`RMSNorm`) Readouts Need an Orthogonal Vacuum Subspace ($\mathcal{H}_{\text{vac}}$)**:
   - On a standard unit sphere $\mathbb{S}^{d-1}$ ($d = 64$ for the Identity Fiber or $d = 448$ for the Semantic Base),
     every normalized vector $h = \operatorname{RMSNorm}(u)$ is forced to have fixed norm $\|h\|_2 = \sqrt{d}$ inside
     the vocabulary subspace $\mathcal{H}_{\text{vocab}}$. It is geometrically forbidden from saying *"I don't know."*
   - In `64D` identity space with $V = 50,304$ tokens, pointing in a random unit direction produces spurious
     extreme-value cosine overlaps up to $\sqrt{2\ln(50,304)/64} \approx 0.58$.
   - Adding an extra $(V+1)$-th "null token" class to `softmax` hurts Cross-Entropy because any probability mass placed
     on $V+1$ directly penalizes $-\ln P(y_{\text{true}})$.
   - By contrast, embedding the readout sphere in $\mathbb{S}^{(d + d_{\text{vac}}) - 1} = \mathcal{H}_{\text{vocab}} \oplus \mathcal{H}_{\text{vac}}$
     ($d_{\text{vac}} = 4$), where every vocabulary detector $w_v$ has **identically zero support** on $\mathcal{H}_{\text{vac}}$
     ($w_{v, \text{vac}} \equiv 0$), preserves exact unit-sphere norm while giving each readout branch and Cholesky mode
     a smooth geometric **abstention / confidence angle** $\cos\theta_{\text{vac}}(t, r) \in [0, 1]$:
     $$h_{\text{vocab}} \;=\; \frac{u_{\text{vocab}}}{\sqrt{\operatorname{MS}(u_{\text{vocab}}) + \operatorname{MS}(u_{\text{vac}}) + \varepsilon}}
       \;=\; \cos\theta_{\text{vac}} \cdot \operatorname{RMSNorm}(u_{\text{vocab}}).$$

2. **Three Architectural Places Where $\mathcal{H}_{\text{vac}}$ Directly Improves Language Modeling**:
   - **High-Entropy Identity Abstention & Pure-Neural Step-0 Calibration**:
     In `VacuumAwareNGramProjector`, the empirical next-token average $\mu_{\text{id}}(b) \in \mathbb{R}^{64}$ has
     mean-square $\operatorname{MS}(\mu_{\text{id}}(b)) \approx \sum_v P(v \mid b)^2$ (the Rényi-2 purity of the
     continuation distribution). Preserving this norm and normalizing on $\mathcal{H}_{\text{id}} \oplus \mathcal{H}_{\text{vac}}$
     rotates ambiguous contexts into $\mathcal{H}_{\text{vac}}$ ($\cos\theta_{\text{vac}} \to 0$) while firing at full
     strength ($\cos\theta_{\text{vac}} \to 1$) on confident contexts—unlocking a much sharper Step-0 `temp_id_scale`
     without any external Kneser-Ney lookup table.
   - **Dark Parking State for Unused Cholesky Eigenmodes ($r = 1, 2, 3$)**:
     Passing all 4 Cholesky modes through the Procrustes-calibrated `exit_bridge` and normalizing each mode on
     $\mathcal{H}_{\text{vocab}} \oplus \mathcal{H}_{\text{vac}}$ allows low-energy or noisy excited modes ($r \in \{1, 2, 3\}$)
     to park harmlessly in $\mathcal{H}_{\text{vac}}$ ($\cos\theta_{\text{vac}, r} \to 0$) instead of injecting false-positive
     `64D` cosine peaks.
   - **Layer-0 Multi-Horizon Cholesky Initialization ($L_{t,h}^{(0)} \in \mathbb{C}^{4 \times 32}$) & True 4-Mode $512\text{D}$ Readout**:
     Constructs a unit-trace rank-4 Cholesky state directly at Layer 0 from $[|\psi_{t,h}\rangle,\, |\tilde{\psi}_{t-1,h}\rangle,\, |\tilde{\psi}_{t-2,h}\rangle,\, \sqrt{d_c}(|\psi_{t,h}\rangle \odot |\tilde{\psi}_{t-1,h}\rangle)]$
     and evaluates all 4 final-layer Cholesky modes as independent $512\text{D}$ ($448\text{D semantic} + 64\text{D identity}$)
     hypotheses on $\mathcal{H}_{\text{vocab}} \oplus \mathcal{H}_{\text{vac}}$.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.impure_density_mirror_block import DEFAULT_TIKHONOV_GRAM_LAMBDA, TRACE_NORM_EPS
from ..layers.quantum_statmech_channels import QuantumMomentumScheduleMode, StiefelMomentumMode, compute_riemannian_momentum_beta
from ..models.modular_quantum_statmech_gpt import ModularRiemannianStiefelKrausMuon
from ..optim.vocab_manifold_muon import BatchSubspaceTikhonovGramMuon
from .born_schur_density_lab import DEFAULT_FINEWEB_EVAL_INTERVAL, DEFAULT_FINEWEB_STEPS, DEFAULT_SCHUR_EPS
from .experiment_logger import ExperimentIntentSpec, StandardizedExperimentLogger
from .multibranch_wavepacket_readout_lab import DEFAULT_MULTIBRANCH_TOKEN_CHUNK
from .shallow_longseq_wavepacket_lm_lab import (
    DEFAULT_BIGRAM_HASH_PRIME, DEFAULT_CARRIER_BIAS_DIM, DEFAULT_NGRAM_DIRICHLET_PRIOR, DEFAULT_NGRAM_SEM_DIM,
    DEFAULT_SHALLOW_EMBED_DIM, DEFAULT_SHALLOW_NGRAM_LR, DEFAULT_SHALLOW_PKT_DIM, DEFAULT_SHALLOW_READOUT_LR,
    DEFAULT_SHALLOW_SCALAR_LR, DEFAULT_SHALLOW_SEM_COMPLEX_PKTS, DEFAULT_SHALLOW_STIEFEL_LR, DEFAULT_SHALLOW_UNITARY_LR,
    DEFAULT_SHALLOW_VOCAB_SIZE, DEFAULT_SHALLOW_WTE_LR, DEFAULT_SKIPGRAM_HASH_PRIME, DEFAULT_SOFTCAP_LOGIT,
    DEFAULT_STEP0_PROCRUSTES_BATCHES, DEFAULT_STEP0_PROCRUSTES_LAMBDA, DEFAULT_TRIGRAM_HASH_PRIME,
    RAW_GPT2_VOCAB_SIZE, LongSeqFineWebStream,
)
from .support_preserving_countsketch_lab import SupportSketchMode, SupportSketchSpec, compute_multishard_multilag_complex_wavefunction
from .wavefunction_outer_density_lab import (
    DEFAULT_CHOLESKY_ANCHOR_LOGIT_BIAS, DEFAULT_CHOLESKY_MODE_COHERENCE_MIX, DEFAULT_INIT_SCHUR_GATE_SCALE,
    DEFAULT_LONG_FINEWEB_EVAL_INTERVAL, DEFAULT_LONG_FINEWEB_STEPS, NativeCholeskyDensitySpec,
    NativeCholeskyDensityWavepacketGPT, _NativeCholeskySchurWavepacketBlock,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
DEFAULT_VACUUM_SUBSPACE_DIM: int = 4
DEFAULT_INIT_ID_VACUUM_MASS: float = 0.14
DEFAULT_INIT_SEM_VACUUM_MASS: float = 0.04
DEFAULT_INIT_EXCITED_CHOL_VAC_MASS: float = 0.30
DEFAULT_INIT_ATTN_VACUUM_MASS: float = 0.12
DEFAULT_NGRAM_PURITY_GATE_GAMMA: float = 0.18
DEFAULT_NGRAM_ID_VAC_VARIANCE: float = 3.24
DEFAULT_VACUUM_NGRAM_BUCKETS: int = 65536
DEFAULT_LAYER0_CHOL_LAG1_SCALE: float = -0.80
DEFAULT_LAYER0_CHOL_LAG2_SCALE: float = -1.20
DEFAULT_LAYER0_CHOL_SCHUR_SCALE: float = -1.10
CHOLESKY_UNIT_RMS_SCALE: float = 16.0


class VacuumCholeskyVariantMode(str, Enum):
    """Strategy enum for Orthogonal Vacuum Subspace & 4-Mode Cholesky experiments (`V0..V4`, Rule 3)."""

    V0_D8_UNIT_SPHERE_BASELINE = "v0_d8_unit_sphere_baseline"
    V1_VACUUM_SUBSPACE_STEP0_PURE_NEURAL = "v1_vacuum_subspace_step0_pure_neural"
    V2_VACUUM_PLUS_LAYER0_INPUT_CHOLESKY = "v2_vacuum_plus_layer0_input_cholesky"
    V3_VACUUM_4MODE_INDEPENDENT_512D_CHOLESKY = "v3_vacuum_4mode_independent_512d_cholesky"
    V4_PURE_NEURAL_VACUUM_CHOLESKY_SCALING_1200 = "v4_pure_neural_vacuum_cholesky_scaling_1200"


@dataclass(frozen=True)
class VacuumCholeskyExperimentSpec:
    """Configuration for `VacuumSubspaceCholeskyGPT` variants (`<= 8` fields, Rule 2)."""

    variant_mode: VacuumCholeskyVariantMode = VacuumCholeskyVariantMode.V1_VACUUM_SUBSPACE_STEP0_PURE_NEURAL
    vacuum_dim: int = DEFAULT_VACUUM_SUBSPACE_DIM
    use_layer0_input_cholesky: bool = False
    use_independent_4mode_512d: bool = False
    init_id_vac_mass: float = DEFAULT_INIT_ID_VACUUM_MASS
    init_sem_vac_mass: float = DEFAULT_INIT_SEM_VACUUM_MASS
    init_excited_chol_vac_mass: float = DEFAULT_INIT_EXCITED_CHOL_VAC_MASS
    seed: int = 1337


def vacuum_sphere_rms_norm(
    u_vocab: torch.Tensor,
    u_vac: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Normalize $(u_{\text{vocab}}, u_{\text{vac}})$ onto $\mathbb{S}^{(d + d_{\text{vac}}) - 1} = \mathcal{H}_{\text{vocab}} \oplus \mathcal{H}_{\text{vac}}$ (`<= 8` args).

    Because all $V$ vocabulary detectors have zero support on $\mathcal{H}_{\text{vac}}$, the vocabulary projection is:
    $$h_{\text{vocab}} = \frac{u_{\text{vocab}}}{\sqrt{\operatorname{MS}(u_{\text{vocab}}) + \operatorname{MS}(u_{\text{vac}}) + \varepsilon}}
      = \cos\theta_{\text{vac}} \cdot \operatorname{RMSNorm}(u_{\text{vocab}})$$
    where the lifted state $[h_{\text{vocab}}, \sqrt{d / d_{\text{vac}}}\, h_{\text{vac}}]$ has exact constant squared
    norm $d$ on $\mathbb{S}^{(d + d_{\text{vac}}) - 1}$, and $\cos\theta_{\text{vac}} \in [0, 1]$ is the geometric
    confidence / non-abstention angle.
    """
    ms_vocab = u_vocab.float().square().mean(dim=-1, keepdim=True)
    if u_vac.numel() > 0 and u_vac.size(-1) > 0:
        ms_vac = u_vac.float().square().mean(dim=-1, keepdim=True)
    else:
        ms_vac = torch.zeros_like(ms_vocab)
    inv_rms = torch.rsqrt(ms_vocab + ms_vac + float(eps))
    h_vocab = (u_vocab.float() * inv_rms).to(u_vocab.dtype)
    h_vac = (u_vac.float() * inv_rms).to(u_vocab.dtype)
    cos_theta = (torch.sqrt(ms_vocab.clamp_min(0.0)) * inv_rms).squeeze(-1)
    return h_vocab, h_vac, cos_theta


def build_layer0_multihorizon_cholesky_factor(
    raw_wp_4d: torch.Tensor,
    wp_causal_re: torch.Tensor,
    wp_causal_im: torch.Tensor,
    row_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Construct unit-Frobenius Layer-0 rank-4 Cholesky factor $L_{t,h}^{(0)} \in \mathbb{C}^{4 \times 32}$ (`<= 8` args).

    Given Layer-0 wavepackets $|\psi_{t,h}\rangle \in \mathbb{C}^{32}$ (`raw_wp_4d` of shape `(B, T, 8, 2, 32)`):
      - Row $r = 0$: Current wavepacket $|\psi_{t,h}\rangle$
      - Row $r = 1$: Lag-1 phase-transported wavepacket $\gamma_1 |\tilde{\psi}_{t-1,h}\rangle$ (zeroed on ID packet $h=7$)
      - Row $r = 2$: Lag-2 phase-transported wavepacket $\gamma_2 |\tilde{\psi}_{t-2,h}\rangle$ (zeroed on ID packet $h=7$)
      - Row $r = 3$: Bigram Schur coincidence $\gamma_3 \sqrt{d_c}(|\psi_{t,h}\rangle \odot_{\mathbb{C}} |\tilde{\psi}_{t-1,h}\rangle)$ (zeroed on ID packet $h=7$)
    Returns `(l0_re, l0_im)` each of shape `(B, 8, T, 4, 32)` satisfying $\|L_{t,h}^{(0)}\|_F = 1$.
    """
    bsz, seq_len, n_heads, _, d_h = raw_wp_4d.shape
    work_dtype = raw_wp_4d.dtype
    re_0 = raw_wp_4d[:, :, :, 0, :].float()
    im_0 = raw_wp_4d[:, :, :, 1, :].float()

    n0 = torch.rsqrt(re_0.square().sum(dim=-1, keepdim=True) + im_0.square().sum(dim=-1, keepdim=True) + DEFAULT_SCHUR_EPS)
    u0_re, u0_im = re_0 * n0, im_0 * n0

    sem_mask = torch.ones((1, 1, n_heads, 1), device=raw_wp_4d.device, dtype=torch.float32)
    sem_mask[:, :, -1, :] = 0.0

    cr = torch.cos(wp_causal_re.float()).view(1, 1, n_heads, 1)
    si = torch.sin(wp_causal_im.float()).view(1, 1, n_heads, 1)
    rot_re = (u0_re * cr - u0_im * si) * sem_mask
    rot_im = (u0_re * si + u0_im * cr) * sem_mask

    if seq_len > 1:
        z1 = torch.zeros_like(rot_re[:, :1])
        u1_re, u1_im = torch.cat([z1, rot_re[:, :-1]], dim=1), torch.cat([z1, rot_im[:, :-1]], dim=1)
    else:
        u1_re, u1_im = torch.zeros_like(u0_re), torch.zeros_like(u0_im)

    if seq_len > 2:
        z2 = torch.zeros_like(rot_re[:, :2])
        u2_re, u2_im = torch.cat([z2, rot_re[:, :-2]], dim=1), torch.cat([z2, rot_im[:, :-2]], dim=1)
    else:
        u2_re, u2_im = torch.zeros_like(u0_re), torch.zeros_like(u0_im)

    sqrt_d = math.sqrt(float(d_h))
    sch_re = sqrt_d * (u0_re * u1_re - u0_im * u1_im) * sem_mask
    sch_im = sqrt_d * (u0_re * u1_im + u0_im * u1_re) * sem_mask

    g = (0.45 * torch.sigmoid(row_scales.float())).view(1, 1, 1, 3, 1)
    r_exc_re = torch.stack([u1_re, u2_re, sch_re], dim=-2) * g
    r_exc_im = torch.stack([u1_im, u2_im, sch_im], dim=-2) * g

    all_re = torch.cat([u0_re.unsqueeze(-2), r_exc_re], dim=-2).permute(0, 2, 1, 3, 4).contiguous()
    all_im = torch.cat([u0_im.unsqueeze(-2), r_exc_im], dim=-2).permute(0, 2, 1, 3, 4).contiguous()

    frob = torch.rsqrt(
        all_re.square().sum(dim=(-2, -1), keepdim=True) + all_im.square().sum(dim=(-2, -1), keepdim=True) + TRACE_NORM_EPS
    )
    return (all_re * frob).to(work_dtype), (all_im * frob).to(work_dtype)


class VacuumAwareNGramProjector(nn.Module):
    r"""Purity-Preserving Causal Multi-Order Wavepacket Projector for $\mathcal{H}_{\text{vocab}} \oplus \mathcal{H}_{\text{vac}}$.

    Preserves the fast `65,536`-bucket shrinkage scale (`0.65 * sqrt(192)`) and Tikhonov-damped RMS normalization
    ($\tau_{\text{vac}} = 0.25$) of `SemanticSubspaceNGramProjector` while calibrating across all 9 training shards
    (`Shards 1..9`) and dynamically backing off unseen higher-order buckets via Hilbert-space signal purity.
    """

    def __init__(
        self,
        embed_dim: int = DEFAULT_SHALLOW_EMBED_DIM,
        sem_dim: int = 2 * DEFAULT_SHALLOW_SEM_COMPLEX_PKTS * DEFAULT_SHALLOW_PKT_DIM,
        ngram_buckets: int = DEFAULT_VACUUM_NGRAM_BUCKETS,
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

        q_init, _ = torch.linalg.qr(torch.randn(self.sem_dim, self.sem_ngram_dim, dtype=torch.float32))
        self.u_sem = nn.Parameter(q_init.contiguous())

        self.in_scale = nn.Parameter(torch.full((self.sem_dim,), 0.22, dtype=torch.float32))
        self.readout_sem_scale = nn.Parameter(torch.full((self.sem_dim,), 0.85, dtype=torch.float32))
        self.readout_id_scale = nn.Parameter(torch.full((self.id_dim,), 0.85, dtype=torch.float32))
        self.order_logits = nn.Parameter(torch.tensor([0.35, 0.95, 0.65, 0.25], dtype=torch.float32))
        self.purity_gate_gamma = nn.Parameter(torch.tensor(DEFAULT_NGRAM_PURITY_GATE_GAMMA, dtype=torch.float32))

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

    @staticmethod
    def _scale_table_preserving_shrinkage(
        mean_mat: torch.Tensor,
        cnt_vec: torch.Tensor,
        target_dim: int,
    ) -> torch.Tensor:
        """Scale `mean_mat` by global active-row RMS (`0.65 * sqrt(target_dim)`) so Bayesian shrinkage & AdamW step scale match `D8`."""
        active = cnt_vec > 0
        if not bool(active.any().item()):
            return mean_mat
        rms = mean_mat[active].square().mean().sqrt().clamp_min(1e-6)
        return (mean_mat / rms) * (0.65 * math.sqrt(float(target_dim)))

    @torch.no_grad()
    def calibrate_from_training_shards_(
        self,
        stream: LongSeqFineWebStream,
        target_table_512: torch.Tensor,
    ) -> None:
        """Calibrate `u_sem` and purity-preserving causal tables strictly from training Shards `1..9` (zero Shard `0` access)."""
        device = stream.device
        sem_target = target_table_512[:, : self.sem_dim].float()
        id_target = target_table_512[:, self.sem_dim :].float()

        gram_448 = torch.matmul(sem_target.t(), sem_target)
        _, eigvecs = torch.linalg.eigh(gram_448)
        u_top = eigvecs[:, -self.sem_ngram_dim :].flip(-1).contiguous()
        self.u_sem.copy_(u_top)

        sem_low = torch.matmul(sem_target, u_top)
        target_code = torch.cat([sem_low, id_target], dim=-1).contiguous()

        uni_cnt = torch.zeros(DEFAULT_SHALLOW_VOCAB_SIZE, device=device, dtype=torch.float32)
        uni_sum = torch.zeros((DEFAULT_SHALLOW_VOCAB_SIZE, self.ngram_dim), device=device, dtype=torch.float32)
        bi_cnt = torch.zeros(self.ngram_buckets, device=device, dtype=torch.float32)
        bi_sum = torch.zeros((self.ngram_buckets, self.ngram_dim), device=device, dtype=torch.float32)
        tri_cnt = torch.zeros(self.ngram_buckets, device=device, dtype=torch.float32)
        tri_sum = torch.zeros((self.ngram_buckets, self.ngram_dim), device=device, dtype=torch.float32)
        skip_cnt = torch.zeros(self.ngram_buckets, device=device, dtype=torch.float32)
        skip_sum = torch.zeros((self.ngram_buckets, self.ngram_dim), device=device, dtype=torch.float32)

        # Prioritize Shard 1 (24M tokens) + 3M tokens from Shards 2..9 (48M tokens total, < 220ms on GPU)
        chunk_sz = 2_500_000
        for s_idx, mm in enumerate(stream.train_memmaps):
            tokens_this_shard = 24_000_000 if s_idx == 0 else 3_000_000
            n_avail = min(tokens_this_shard, len(mm) - 8)
            if s_idx == 0 and stream.shard1_gpu.numel() >= n_avail + 4:
                shard_tok = stream.shard1_gpu[: n_avail + 4]
            else:
                arr = np.asarray(mm[: n_avail + 4], dtype=np.int64)
                shard_tok = torch.from_numpy(arr).to(device=device, dtype=torch.long)

            for start in range(0, n_avail, chunk_sz):
                c_len = min(chunk_sz, n_avail - start)
                tok = shard_tok[start : start + c_len + 3]
                x_p2, x_p1, x_t, y_next = tok[0:c_len], tok[1 : 1 + c_len], tok[2 : 2 + c_len], tok[3 : 3 + c_len]
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

        self.unigram_next_table.copy_(self._scale_table_preserving_shrinkage(uni_mean, uni_cnt, self.ngram_dim))
        self.bigram_table.copy_(self._scale_table_preserving_shrinkage(bi_mean, bi_cnt, self.ngram_dim))
        self.trigram_table.copy_(self._scale_table_preserving_shrinkage(tri_mean, tri_cnt, self.ngram_dim))
        self.skipgram_table.copy_(self._scale_table_preserving_shrinkage(skip_mean, skip_cnt, self.ngram_dim))
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
        """Return `(in_sem_delta_512, out_sem_448, out_id_64)` with purity-aware order backoff and Tikhonov vacuum damping."""
        bsz, seq_len = idx.shape
        bi_idx, tri_idx, skip_idx = self.compute_causal_buckets(idx)

        v_uni = F.embedding(idx, self.unigram_next_table.to(work_dtype))
        v_bi = F.embedding(bi_idx, self.bigram_table.to(work_dtype))
        v_tri = F.embedding(tri_idx, self.trigram_table.to(work_dtype))
        v_skip = F.embedding(skip_idx, self.skipgram_table.to(work_dtype))

        v_stack = torch.stack([v_uni, v_bi, v_tri, v_skip], dim=-2)
        order_ms = v_stack.float().square().mean(dim=-1)
        gamma = self.purity_gate_gamma.float().clamp(0.0, 0.35)
        dyn_logits = self.order_logits.float().view(1, 1, 4) + gamma * torch.tanh(0.25 * torch.log(order_ms + 1e-3))
        w_dyn = F.softmax(dyn_logits, dim=-1).to(work_dtype).unsqueeze(-1)
        low_comb = (v_stack * w_dyn).sum(dim=-2)

        sem_code = low_comb[:, :, : self.sem_ngram_dim]
        id_code = low_comb[:, :, self.sem_ngram_dim :]

        sem_lift_scale = math.sqrt(float(self.sem_dim) / float(self.sem_ngram_dim))
        sem_raw = F.linear(sem_code, self.u_sem.to(work_dtype)) * sem_lift_scale
        sem_448 = (sem_raw.float() * torch.rsqrt(sem_raw.float().square().mean(dim=-1, keepdim=True) + 0.0625)).to(work_dtype)
        id_64 = (
            id_code.float() * torch.rsqrt(id_code.float().square().mean(dim=-1, keepdim=True) + DEFAULT_NGRAM_ID_VAC_VARIANCE)
        ).to(work_dtype)

        in_sem = sem_448 * torch.tanh(self.in_scale).to(work_dtype)
        id_zeros = torch.zeros((bsz, seq_len, self.id_dim), device=idx.device, dtype=work_dtype)
        in_sem_delta_512 = torch.cat([in_sem, id_zeros], dim=-1)

        out_sem_448 = sem_448 * (1.5 * torch.tanh(self.readout_sem_scale)).to(work_dtype)
        out_id_64 = id_64 * (1.5 * torch.tanh(self.readout_id_scale)).to(work_dtype)
        return in_sem_delta_512, out_sem_448, out_id_64


class _VacuumFourMode512DWavepacketCEFn(torch.autograd.Function):
    r"""Rule-9 Single-Pass Chunked Autograd Function for 4-Mode Independent $512\text{D}$ (`448D` Semantic + `64D` Identity) Cholesky Readout.

    Supports both:
      - Shared secondary semantic vector `h_sem_branches` of shape `(B, T, 448)` (`V0, V1, V2`), and
      - Per-mode independent semantic Cholesky branches `h_sem_branches` of shape `(B, T, R, 448)` (`V3, V4`),
    where both `h_sem_branches` and `h_id_branches` (`(B, T, R, 64)`) have already been projected from the
    Vacuum-Augmented Sphere $\mathcal{H}_{\text{vocab}} \oplus \mathcal{H}_{\text{vac}}$ (`bwd_flops / fwd_flops <= 2.3 <= 3.5`).
    """

    @staticmethod
    def forward(
        ctx: Any, h_base_464: torch.Tensor, h_sem_branches: torch.Tensor, h_id_branches: torch.Tensor,
        branch_controls: torch.Tensor, w_aug_528: torch.Tensor, targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute single-pass chunked forward cross-entropy for 4-mode independent or shared semantic + identity branches."""
        bsz, seq_len, _ = h_base_464.shape
        n_total, num_branches, v_sz = bsz * seq_len, h_id_branches.size(-2), w_aug_528.size(0)
        sem_dim, id_dim = h_sem_branches.size(-1), h_id_branches.size(-1)
        car_dim, indep_sem = h_base_464.size(-1) - sem_dim, h_sem_branches.ndim == 4

        hb_flat = h_base_464.reshape(n_total, sem_dim + car_dim)
        hs_flat = h_sem_branches.reshape(n_total, num_branches, sem_dim) if indep_sem else h_sem_branches.reshape(n_total, sem_dim)
        hid_flat = h_id_branches.reshape(n_total, num_branches, id_dim)
        ctrl_flat = branch_controls.reshape(n_total, num_branches, 3).float()
        beta_flat, tau_flat, alpha_flat = ctrl_flat[:, :, 0], ctrl_flat[:, :, 1], ctrl_flat[:, :, 2]
        y_flat = targets.reshape(n_total)

        w_sem, w_id = w_aug_528[:, :sem_dim], w_aug_528[:, sem_dim : sem_dim + id_dim]
        w_car = w_aug_528[:, sem_dim + id_dim : sem_dim + id_dim + car_dim]
        w_base_t = torch.cat([w_sem, w_car], dim=-1).t().contiguous()
        w_sem_t, w_id_t = w_sem.t().contiguous(), w_id.t().contiguous()

        cap, inv_cap = float(DEFAULT_SOFTCAP_LOGIT), 1.0 / float(DEFAULT_SOFTCAP_LOGIT)
        raw_chunk = (DEFAULT_MULTIBRANCH_TOKEN_CHUNK * 2) // max(1, num_branches)
        chunk_sz = max(128, (raw_chunk // 64) * 64)
        total_nll = torch.zeros((), device=h_base_464.device, dtype=torch.float32)
        n_valid = int((y_flat >= 0).sum().item())
        all_valid = bool(n_valid == n_total)
        has_sem1 = bool(beta_flat.abs().max().item() > 1e-8)

        for start in range(0, n_total, chunk_sz):
            end = min(start + chunk_sz, n_total)
            c_len, yc = end - start, y_flat[start:end]
            if not all_valid and not bool((yc >= 0).any().item()):
                continue

            z_base = torch.matmul(hb_flat[start:end], w_base_t).float()
            raw_r = torch.matmul(hid_flat[start:end].reshape(c_len * num_branches, id_dim), w_id_t).float().view(c_len, num_branches, v_sz)
            raw_r.mul_(tau_flat[start:end].unsqueeze(-1)).add_(z_base.unsqueeze(1))
            del z_base
            if has_sem1:
                if indep_sem:
                    z_sem1 = torch.matmul(hs_flat[start:end].reshape(c_len * num_branches, sem_dim), w_sem_t).float().view(c_len, num_branches, v_sz)
                    raw_r.addcmul_(beta_flat[start:end].unsqueeze(-1), z_sem1)
                else:
                    z_sem1 = torch.matmul(hs_flat[start:end], w_sem_t).float()
                    raw_r.addcmul_(beta_flat[start:end].unsqueeze(-1), z_sem1.unsqueeze(1))
                del z_sem1
            raw_r.mul_(inv_cap).tanh_().mul_(cap)

            log_pi = F.log_softmax(alpha_flat[start:end], dim=-1)
            yc_idx = yc.clamp_min(0).view(c_len, 1, 1).expand(c_len, num_branches, 1)
            target_log_p_r = raw_r.gather(dim=-1, index=yc_idx).squeeze(-1) - torch.logsumexp(raw_r, dim=-1)
            del raw_r
            target_log_mix = torch.logsumexp(log_pi + target_log_p_r, dim=-1)
            total_nll = total_nll - (target_log_mix.sum() if all_valid else target_log_mix[yc >= 0].sum())

        ctx.save_for_backward(h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug_528, targets)
        ctx.n_valid, ctx.all_valid, ctx.has_sem1, ctx.indep_sem = max(1, n_valid), all_valid, has_sem1, indep_sem
        return total_nll / float(max(1, n_valid))

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Single-pass chunked backward computing all input and vocabulary gradients inside one loop (Rule 9)."""
        h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug_528, targets = ctx.saved_tensors
        bsz, seq_len, _ = h_base_464.shape
        n_total, num_branches, v_sz = bsz * seq_len, h_id_branches.size(-2), w_aug_528.size(0)
        sem_dim, id_dim = h_sem_branches.size(-1), h_id_branches.size(-1)
        car_dim, work_dtype = h_base_464.size(-1) - sem_dim, h_base_464.dtype
        has_sem1, indep_sem = bool(getattr(ctx, "has_sem1", True)), bool(getattr(ctx, "indep_sem", False))
        all_valid = bool(getattr(ctx, "all_valid", False))

        hb_flat = h_base_464.reshape(n_total, sem_dim + car_dim)
        hs_flat = h_sem_branches.reshape(n_total, num_branches, sem_dim) if indep_sem else h_sem_branches.reshape(n_total, sem_dim)
        hid_flat = h_id_branches.reshape(n_total, num_branches, id_dim)
        ctrl_flat = branch_controls.reshape(n_total, num_branches, 3).float()
        beta_flat, tau_flat, alpha_flat = ctrl_flat[:, :, 0], ctrl_flat[:, :, 1], ctrl_flat[:, :, 2]
        y_flat = targets.reshape(n_total)

        w_sem, w_id = w_aug_528[:, :sem_dim], w_aug_528[:, sem_dim : sem_dim + id_dim]
        w_car = w_aug_528[:, sem_dim + id_dim : sem_dim + id_dim + car_dim]
        w_base = torch.cat([w_sem, w_car], dim=-1).contiguous()
        w_base_t, w_sem_t, w_id_t = w_base.t().contiguous(), w_sem.t().contiguous(), w_id.t().contiguous()

        grad_hb, grad_hs, grad_hid = torch.zeros_like(hb_flat), torch.zeros_like(hs_flat), torch.zeros_like(hid_flat)
        grad_ctrl = torch.zeros_like(ctrl_flat, dtype=branch_controls.dtype)
        grad_w_base, grad_w_sem1, grad_w_id = torch.zeros_like(w_base), torch.zeros_like(w_sem), torch.zeros_like(w_id)

        cap, inv_cap = float(DEFAULT_SOFTCAP_LOGIT), 1.0 / float(DEFAULT_SOFTCAP_LOGIT)
        scale = float(grad_output.item()) / float(ctx.n_valid)
        raw_chunk = (DEFAULT_MULTIBRANCH_TOKEN_CHUNK * 2) // max(1, num_branches)
        chunk_sz = max(128, (raw_chunk // 64) * 64)

        for start in range(0, n_total, chunk_sz):
            end = min(start + chunk_sz, n_total)
            c_len, yc = end - start, y_flat[start:end]
            valid_c = yc >= 0
            if not all_valid and not bool(valid_c.any().item()):
                continue

            hb_c, hs_c, hid_c = hb_flat[start:end], hs_flat[start:end], hid_flat[start:end]
            b_c, t_c, a_c = beta_flat[start:end], tau_flat[start:end], alpha_flat[start:end]

            z_base = torch.matmul(hb_c, w_base_t).float()
            tanh_u = torch.matmul(hid_c.reshape(c_len * num_branches, id_dim), w_id_t).float().view(c_len, num_branches, v_sz)
            tanh_u.mul_(t_c.unsqueeze(-1)).add_(z_base.unsqueeze(1))
            del z_base
            if has_sem1:
                if indep_sem:
                    z_sem1 = torch.matmul(hs_c.reshape(c_len * num_branches, sem_dim), w_sem_t).float().view(c_len, num_branches, v_sz)
                    tanh_u.addcmul_(b_c.unsqueeze(-1), z_sem1)
                else:
                    z_sem1 = torch.matmul(hs_c, w_sem_t).float()
                    tanh_u.addcmul_(b_c.unsqueeze(-1), z_sem1.unsqueeze(1))
                del z_sem1
            tanh_u.mul_(inv_cap).tanh_()
            log_p_r = F.log_softmax(tanh_u * cap, dim=-1)

            log_pi = F.log_softmax(a_c, dim=-1)
            pi_c = log_pi.exp()
            yc_idx = yc.clamp_min(0).view(c_len, 1, 1).expand(c_len, num_branches, 1)
            target_log_p_r = log_p_r.gather(dim=-1, index=yc_idx).squeeze(-1)

            gamma_c = F.softmax(log_pi + target_log_p_r, dim=-1)
            valid_f = torch.ones((c_len, 1), device=yc.device, dtype=torch.float32) if all_valid else valid_c.float().unsqueeze(-1)
            grad_ctrl[start:end, :, 2] = (scale * valid_f * (pi_c - gamma_c)).to(branch_controls.dtype)

            p_r = log_p_r.exp_()
            p_r.scatter_add_(dim=-1, index=yc_idx, src=torch.full((c_len, num_branches, 1), -1.0, device=p_r.device, dtype=p_r.dtype))
            tanh_u.square_().neg_().add_(1.0)
            p_r.mul_(scale * (valid_f * gamma_c).unsqueeze(-1)).mul_(tanh_u)
            del tanh_u
            g_u = p_r

            g_base = g_u.sum(dim=1).to(work_dtype)
            grad_hb[start:end] = torch.matmul(g_base, w_base)
            grad_w_base.add_(torch.matmul(g_base.t(), hb_c))

            g_u_w = g_u.to(work_dtype)
            del g_u

            if has_sem1:
                if indep_sem:
                    grad_hs_unscaled = torch.matmul(g_u_w.reshape(c_len * num_branches, v_sz), w_sem).view(c_len, num_branches, sem_dim)
                    grad_hs[start:end] = grad_hs_unscaled * b_c.unsqueeze(-1).to(work_dtype)
                    grad_ctrl[start:end, :, 0] = (grad_hs_unscaled.float() * hs_c.float()).sum(dim=-1).to(branch_controls.dtype)
                    g_sem_r = (g_u_w * b_c.unsqueeze(-1).to(work_dtype)).reshape(c_len * num_branches, v_sz)
                    grad_w_sem1.add_(torch.matmul(g_sem_r.t(), hs_c.reshape(c_len * num_branches, sem_dim)))
                else:
                    grad_hs_unscaled = torch.matmul(g_u_w.reshape(c_len * num_branches, v_sz), w_sem).view(c_len, num_branches, sem_dim)
                    grad_ctrl[start:end, :, 0] = (grad_hs_unscaled.float() * hs_c.float().unsqueeze(1)).sum(dim=-1).to(branch_controls.dtype)
                    grad_hs[start:end] = (grad_hs_unscaled * b_c.unsqueeze(-1).to(work_dtype)).sum(dim=1)
                    g_sem1 = (g_u_w * b_c.unsqueeze(-1).to(work_dtype)).sum(dim=1)
                    grad_w_sem1.add_(torch.matmul(g_sem1.t(), hs_c))

            grad_hid_unscaled = torch.matmul(g_u_w.reshape(c_len * num_branches, v_sz), w_id).view(c_len, num_branches, id_dim)
            grad_hid[start:end] = grad_hid_unscaled * t_c.unsqueeze(-1).to(work_dtype)
            grad_ctrl[start:end, :, 1] = (grad_hid_unscaled.float() * hid_c.float()).sum(dim=-1).to(branch_controls.dtype)
            g_id_flat = (g_u_w * t_c.unsqueeze(-1).to(work_dtype)).reshape(c_len * num_branches, v_sz)
            grad_w_id.add_(torch.matmul(g_id_flat.t(), hid_c.reshape(c_len * num_branches, id_dim)))
            del g_u_w, grad_hid_unscaled

        grad_w_aug = torch.cat([grad_w_base[:, :sem_dim] + grad_w_sem1, grad_w_id, grad_w_base[:, sem_dim:]], dim=-1)
        return (
            grad_hb.view_as(h_base_464), grad_hs.view_as(h_sem_branches), grad_hid.view_as(h_id_branches),
            grad_ctrl.view_as(branch_controls), grad_w_aug, None,
        )


class VacuumSubspaceCholeskyGPT(NativeCholeskyDensityWavepacketGPT):
    r"""Uncollapsed 4-Mode Cholesky Density-Matrix GPT with Orthogonal Vacuum Subspace $\mathcal{H}_{\text{vac}}$ (`V0..V4`).

    - When `variant_mode == V0_D8_UNIT_SPHERE_BASELINE`, delegates directly to `D8_UNCOLLAPSED_CHOLESKY_SEM_ID_READOUT`
      (`d_vac = 0`, standard `F.rms_norm(..., (64,))` on $\mathbb{S}^{63}$).
    - In `V1..V4`, embeds both the `64D` Identity Fiber and `448D` Semantic Base (plus all 4 native wavepacket Cholesky
      modes scaled by `CHOLESKY_UNIT_RMS_SCALE = 16.0` so $\operatorname{MS} = 4 E_{t,r}$) into
      $\mathcal{H}_{\text{vocab}} \oplus \mathcal{H}_{\text{vac}}$ ($d_{\text{vac}} = 4$), giving low-energy excited
      Cholesky modes ($r = 1, 2, 3$) a dark parking state and uncertain tokens a smooth abstention angle $\cos\theta_{\text{vac}}$.
    """

    def __init__(self, vac_spec: VacuumCholeskyExperimentSpec) -> None:
        nat_spec = NativeCholeskyDensitySpec(
            use_outer_product_input=False, uncollapsed_cholesky_highway=True, use_semantic_cholesky_readout=True,
            enable_kn_synergy_eval=False, init_schur_gate_scale=DEFAULT_INIT_SCHUR_GATE_SCALE,
            anchor_logit_bias=DEFAULT_CHOLESKY_ANCHOR_LOGIT_BIAS, mode_coherence_mix=DEFAULT_CHOLESKY_MODE_COHERENCE_MIX,
            seed=vac_spec.seed,
        )
        super().__init__(nat_spec)
        self.vac_spec = vac_spec
        self.variant_mode = VacuumCholeskyVariantMode(vac_spec.variant_mode)
        self.d_vac = int(vac_spec.vacuum_dim) if self.variant_mode != VacuumCholeskyVariantMode.V0_D8_UNIT_SPHERE_BASELINE else 0

        if self.d_vac > 0:
            self.sem_ngram = VacuumAwareNGramProjector(embed_dim=self.embed_dim, sem_dim=self.sem_real_dim)  # type: ignore[assignment]

        vac_d = max(1, self.d_vac)
        self.id_vac_mass = nn.Parameter(torch.full((vac_d,), float(vac_spec.init_id_vac_mass), dtype=torch.float32))
        self.sem_vac_mass = nn.Parameter(torch.full((vac_d,), float(vac_spec.init_sem_vac_mass), dtype=torch.float32))
        m_id, m_exc = float(vac_spec.init_id_vac_mass), float(vac_spec.init_excited_chol_vac_mass)
        self.chol_id_vac_mass = nn.Parameter(
            torch.tensor([[m_id * 0.5] * vac_d, [m_exc] * vac_d, [m_exc * 1.15] * vac_d, [m_exc * 1.30] * vac_d], dtype=torch.float32)
        )
        self.chol_sem_vac_mass = nn.Parameter(torch.full((4, vac_d), float(vac_spec.init_sem_vac_mass) * 1.5, dtype=torch.float32))

        self.id_vac_proj = nn.Linear(16, vac_d, bias=False)
        self.chol_vac_proj = nn.Linear(16, 4 * vac_d, bias=False)
        nn.init.zeros_(self.id_vac_proj.weight)
        nn.init.zeros_(self.chol_vac_proj.weight)

        self.layer0_chol_row_scales = nn.Parameter(
            torch.tensor([DEFAULT_LAYER0_CHOL_LAG1_SCALE, DEFAULT_LAYER0_CHOL_LAG2_SCALE, DEFAULT_LAYER0_CHOL_SCHUR_SCALE], dtype=torch.float32)
        )

        self.last_anchor_id_cos_vac: float = 1.0
        self.last_excited_id_cos_vac: float = 1.0
        self.last_sem_cos_vac: float = 1.0

    def vacuum_scalar_parameters(self) -> list[nn.Parameter]:
        """Return scalar and vacuum-subspace parameters for `AdamW` optimization."""
        params = list(self.native_density_scalar_parameters())
        if self.d_vac > 0:
            params.extend([
                self.id_vac_mass, self.sem_vac_mass, self.chol_id_vac_mass, self.chol_sem_vac_mass,
                self.id_vac_proj.weight, self.chol_vac_proj.weight,
            ])
        if self.vac_spec.use_layer0_input_cholesky:
            params.append(self.layer0_chol_row_scales)
        return params

    @torch.no_grad()
    def clamp_identity_dispersion_and_scalars_(self) -> None:
        """Keep Identity Packet 7 zero-dispersion and clamp vacuum rest-mass coordinates to physical bounds."""
        super().clamp_identity_dispersion_and_scalars_()
        if self.d_vac > 0:
            self.id_vac_mass.clamp_(0.02, 0.32)
            self.sem_vac_mass.clamp_(0.01, 0.18)
            self.chol_id_vac_mass.clamp_(0.02, 0.65)
            self.chol_sem_vac_mass.clamp_(0.02, 0.45)
        self.layer0_chol_row_scales.clamp_(-3.5, 2.0)

    def _encode_multiscale_trunk(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        r"""Encode `idx` `(B, T)` with optional basis-aligned Layer-0 4-mode Cholesky initialization $L_{t,h}^{(0)} \in \mathbb{C}^{4 \times 32}$."""
        if not self.vac_spec.use_layer0_input_cholesky:
            return super()._encode_multiscale_trunk(idx)

        bsz, seq_len = idx.shape
        work_dtype = torch.bfloat16 if (idx.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
        tok_wp = F.embedding(idx, self.wte_3d.view(self.vocab_size, self.embed_dim)).view(bsz, seq_len, self.pkt_count, self.pkt_dim)
        raw_wp = tok_wp.to(work_dtype)
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
        if self.sem_ngram is not None:
            in_sem_delta, readout_sem_448, readout_id_64 = self.sem_ngram(idx, work_dtype)
            raw_flat = raw_flat + in_sem_delta

        x_0 = F.rms_norm(raw_flat, (self.embed_dim,), eps=1e-6)

        # Build Layer-0 rank-4 Cholesky factor L^{(0)}_{t,h} in C^{4 x 32} aligned with block 0's value wavepacket basis!
        blk0: _NativeCholeskySchurWavepacketBlock = self.quantum_blocks[0]  # type: ignore[assignment]
        v0_wp = blk0.v_proj(x_0).view(bsz, seq_len, self.num_heads, 2, self.pkt_dim)
        chol_state: tuple[torch.Tensor, torch.Tensor] | None = build_layer0_multihorizon_cholesky_factor(
            v0_wp, self.wp_causal_re, self.wp_causal_im, self.layer0_chol_row_scales
        )

        x = x_0
        cos_4d, sin_4d = self.rope.get_cos_sin(seq_len, dtype=work_dtype)
        mid_idx = max(0, (len(self.quantum_blocks) // 2) - 1)
        h_mid = x_0
        for b_idx, q_blk in enumerate(self.quantum_blocks):
            x = q_blk(x, x_0, cos_4d, sin_4d, in_cholesky=chol_state)  # type: ignore[call-arg]
            chol_state = q_blk.last_cholesky_pre_u  # type: ignore[assignment]
            if b_idx == mid_idx:
                h_mid = F.rms_norm(x, (self.embed_dim,), eps=1e-6)

        h_pre = F.rms_norm(x, (self.embed_dim,), eps=1e-6)
        return h_pre, h_mid, readout_sem_448, readout_id_64

    def encode_pre_bridge_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        """Encode `idx` `(B, T)` using the uncollapsed Cholesky trunk so Step-0 Procrustes calibration matches runtime."""
        h_pre, _, _, _ = self._encode_multiscale_trunk(idx)
        return h_pre

    def build_multibranch_readout_tensors(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Build 4 Cholesky readout modes on the Vacuum-Augmented Sphere $\mathcal{H}_{\text{vocab}} \oplus \mathcal{H}_{\text{vac}}$."""
        if self.variant_mode == VacuumCholeskyVariantMode.V0_D8_UNIT_SPHERE_BASELINE:
            return super().build_multibranch_readout_tensors(idx)

        h_pre, _h_mid, readout_sem_448, readout_id_64 = self._encode_multiscale_trunk(idx)
        bsz, seq_len, _ = h_pre.shape
        work_dtype = h_pre.dtype
        h_bridge = self.exit_bridge(h_pre)

        ctx_16 = h_pre[:, :, :16]
        # Bounded contextual vacuum coordinates: lets uncertain tokens point into H_vac without globally collapsing cos(theta_vac)
        u_id_vac_0 = self.id_vac_mass.to(work_dtype).view(1, 1, self.d_vac) * (
            2.0 * torch.sigmoid(F.linear(ctx_16, self.id_vac_proj.weight.to(work_dtype)))
        )
        u_sem_vac_0 = self.sem_vac_mass.to(work_dtype).view(1, 1, self.d_vac).expand(bsz, seq_len, self.d_vac)

        # Normalize trunk bridge slices once on the sphere, then apply H_vocab (+) H_vac normalization to the combined readout
        h_sem_tr = F.rms_norm(h_bridge[:, :, : self.sem_real_dim], (self.sem_real_dim,), eps=1e-6)
        h_id_tr = F.rms_norm(h_bridge[:, :, self.sem_real_dim :], (self.id_real_dim,), eps=1e-6)

        if readout_sem_448 is not None and readout_id_64 is not None:
            g_sem, g_id = self.trunk_sem_gate.to(work_dtype), self.trunk_id_gate.to(work_dtype)
            raw_sem_0, raw_id_0 = g_sem * h_sem_tr + readout_sem_448, g_id * h_id_tr + readout_id_64
        else:
            raw_sem_0, raw_id_0 = h_sem_tr, h_id_tr

        h_sem_0, _, sem_cos_0 = vacuum_sphere_rms_norm(raw_sem_0, u_sem_vac_0)
        h_id_0, h_id_0_vac, id_cos_0 = vacuum_sphere_rms_norm(raw_id_0, u_id_vac_0)
        self.last_anchor_id_cos_vac = float(id_cos_0.detach().mean().item())
        self.last_sem_cos_vac = float(sem_cos_0.detach().mean().item())

        sem_scale = (self.temp_sem_scale / float(self.sem_real_dim)).to(work_dtype)
        id_scale_scalar = (self.temp_id_scale / float(self.id_real_dim)).to(work_dtype)

        h_sem_0_scaled = h_sem_0 * sem_scale
        c0_batch = self.carrier_c0.to(work_dtype).view(1, 1, DEFAULT_CARRIER_BIAS_DIM).expand(bsz, seq_len, DEFAULT_CARRIER_BIAS_DIM)
        h_base_464 = torch.cat([h_sem_0_scaled, c0_batch], dim=-1).contiguous()

        w_dir = (self.w_det_512 * self.radial_scale.unsqueeze(1)).to(work_dtype)
        b_col = (self.prior_gate * self.unigram_bias).to(work_dtype).unsqueeze(1) * self.carrier_c0.to(work_dtype).unsqueeze(0)
        w_aug = torch.cat([w_dir, b_col], dim=-1).contiguous()

        last_blk: _NativeCholeskySchurWavepacketBlock = self.quantum_blocks[-1]  # type: ignore[assignment]
        assert last_blk.last_cholesky_all_re is not None and last_blk.last_cholesky_all_im is not None
        l_all_re, l_all_im = last_blk.last_cholesky_all_re, last_blk.last_cholesky_all_im  # (B, 8, T, 4, 32)

        l7_re, l7_im = l_all_re[:, -1], l_all_im[:, -1]
        id_energy = (l7_re.float().square().sum(dim=-1) + l7_im.float().square().sum(dim=-1)).clamp_min(1e-6)
        l_sem_re, l_sem_im = l_all_re[:, : self.sem_complex_pkts], l_all_im[:, : self.sem_complex_pkts]
        sem_energy = (l_sem_re.float().square().sum(dim=-1) + l_sem_im.float().square().sum(dim=-1)).mean(dim=1).clamp_min(1e-6)
        w_sem = torch.sigmoid(self.sem_energy_mix_logit.float())
        mode_energy = (1.0 - w_sem) * id_energy + w_sem * sem_energy
        mode_log_weights = torch.log(mode_energy) + self.chol_mode_log_prior.view(1, 1, 4)

        # Keep Cholesky modes in native wavepacket basis and scale by CHOLESKY_UNIT_RMS_SCALE = 16.0 so MS = 4 * E_{t,r}!
        l7_64_scaled = torch.stack([l7_re, l7_im], dim=-2).reshape(bsz, seq_len, 4, self.id_real_dim) * CHOLESKY_UNIT_RMS_SCALE
        u_chol_vac = self.chol_id_vac_mass.to(work_dtype).view(1, 1, 4, self.d_vac) * (
            2.0 * torch.sigmoid(F.linear(ctx_16, self.chol_vac_proj.weight.to(work_dtype)).view(bsz, seq_len, 4, self.d_vac))
        )
        l7_vocab, l7_vac, _ = vacuum_sphere_rms_norm(l7_64_scaled, u_chol_vac)

        # Coherence blend in the full (64 + d_vac) sphere so low-energy excited modes park in H_vac!
        coh = torch.sigmoid(self.chol_mode_coherence).to(work_dtype)
        mix_exc_vocab = coh * h_id_0.unsqueeze(-2) + (1.0 - coh) * l7_vocab[:, :, 1:, :]
        mix_exc_vac = coh * h_id_0_vac.unsqueeze(-2) + (1.0 - coh) * l7_vac[:, :, 1:, :]
        excited_modes, _, exc_cos = vacuum_sphere_rms_norm(mix_exc_vocab, mix_exc_vac)
        self.last_excited_id_cos_vac = float(exc_cos.detach().mean().item())

        cholesky_id_branches = torch.cat([h_id_0.unsqueeze(-2), excited_modes], dim=-2)
        h_id_branches = (cholesky_id_branches * id_scale_scalar).contiguous()

        b_exc = (0.35 * torch.tanh(self.sem_chol_branch_gates.float())).view(1, 1, 3).expand(bsz, seq_len, 3)
        sem1_gates = torch.cat([torch.zeros((bsz, seq_len, 1), device=idx.device, dtype=torch.float32), b_exc], dim=-1)

        sem_exc_raw = (
            torch.stack([l_sem_re[:, :, :, 1:, :], l_sem_im[:, :, :, 1:, :]], dim=-2)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(bsz, seq_len, 3, self.sem_real_dim)
            * CHOLESKY_UNIT_RMS_SCALE
        )
        w_m = F.softmax(self.sem_chol_mode_weights.float(), dim=0).to(work_dtype).view(1, 1, 3, 1)
        sem_super = (sem_exc_raw * w_m).sum(dim=2)
        if self.vac_spec.use_independent_4mode_512d:
            # V3 / V4: True 4-Mode Independent 512D (448D Semantic + 64D Identity) Cholesky Readout with H_vac
            s0_f = h_sem_0.float().unsqueeze(-2)
            s0_norm_sq = s0_f.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
            exc_sem_f = (coh * sem_super.unsqueeze(-2) + (1.0 - coh) * sem_exc_raw).float()
            proj_c = (exc_sem_f * s0_f).sum(dim=-1, keepdim=True) / s0_norm_sq
            ortho_sem_exc = (exc_sem_f - proj_c * s0_f).to(work_dtype)
            u_sem_exc_vac = self.chol_sem_vac_mass[1:].to(work_dtype).view(1, 1, 3, self.d_vac).expand(bsz, seq_len, 3, self.d_vac)
            norm_sem_exc, _, _ = vacuum_sphere_rms_norm(ortho_sem_exc, u_sem_exc_vac)
            zero_sem_0 = torch.zeros((bsz, seq_len, 1, self.sem_real_dim), device=idx.device, dtype=work_dtype)
            h_sem_branches = (torch.cat([zero_sem_0, norm_sem_exc], dim=-2) * sem_scale).contiguous()
        else:
            # V1 / V2: Shared orthogonal semantic Cholesky direction with H_vac
            u_f, s0_f_2d = sem_super.float(), h_sem_0.float()
            proj_c = (u_f * s0_f_2d).sum(dim=-1, keepdim=True) / s0_f_2d.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
            ortho_sem_1 = (u_f - proj_c * s0_f_2d).to(work_dtype)
            u_sem1_vac = self.chol_sem_vac_mass[1].to(work_dtype).view(1, 1, self.d_vac).expand(bsz, seq_len, self.d_vac)
            h_sem_1, _, _ = vacuum_sphere_rms_norm(ortho_sem_1, u_sem1_vac)
            h_sem_branches = (h_sem_1 * sem_scale).contiguous()

        id_ones = torch.ones_like(mode_log_weights)
        branch_controls = torch.stack([sem1_gates, id_ones, mode_log_weights], dim=-1).to(work_dtype)
        return h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug

    def forward_loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Vacuum-Subspace 4-Mode Cholesky cross-entropy loss."""
        h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug = self.build_multibranch_readout_tensors(idx)
        return _VacuumFourMode512DWavepacketCEFn.apply(h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug, targets)


@torch.no_grad()
def initialize_and_calibrate_vacuum_cholesky_model_(
    model: VacuumSubspaceCholeskyGPT, stream: LongSeqFineWebStream
) -> tuple[float, dict[str, float]]:
    r"""Initialize `(50304, 8, 2, 32)` wavepackets from Shards `1..9` and calibrate Pure-Neural Step-0 Vacuum Readout (`<= 8` args)."""
    device = stream.device
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    sketch_spec = SupportSketchSpec(
        mode=SupportSketchMode.I17_SHARDS1_TO_9_900M_STEP0_CALIBRATED,
        vocab_size=model.vocab_size, pkt_count=model.pkt_count, pkt_dim=model.pkt_dim, seed=model.spec.seed,
    )
    psi_4d, log_prior, radial, meta = compute_multishard_multilag_complex_wavefunction(
        sketch_spec, num_shards=model.spec.num_init_shards, device=device, rev_token_id=RAW_GPT2_VOCAB_SIZE
    )
    model.wte_3d.copy_(psi_4d.view(model.vocab_size, model.pkt_count, model.pkt_dim))
    model.unigram_bias.copy_(log_prior)
    model.radial_scale.copy_(radial)
    model.prior_gate.fill_(1.0)

    init_disp = torch.tensor([0.22, 0.22, 0.14, 0.08, 0.22, 0.22, 0.14, 0.00], device=device, dtype=torch.float32)
    model.wp_causal_re.copy_(init_disp)
    model.wp_causal_im.zero_()

    for q_blk in model.quantum_blocks:
        q_blk.obs_scale.fill_(0.45)  # type: ignore[union-attr]
        q_blk.mz_mlp.mz_scale.fill_(0.45)  # type: ignore[union-attr]

    wte_flat = model.wte_3d.detach().float().view(model.vocab_size, model.embed_dim)
    p0 = F.softmax(model.unigram_bias.detach().float(), dim=0).unsqueeze(1)
    sem_raw, id_raw = wte_flat[:, : model.sem_real_dim], wte_flat[:, model.sem_real_dim :]
    active_sem_mask = (sem_raw.norm(dim=-1, keepdim=True) > 1e-3).float()
    mean_sem = (p0 * sem_raw).sum(dim=0, keepdim=True)
    centered_sem = F.normalize(sem_raw - active_sem_mask * mean_sem, dim=-1, eps=1e-6)
    centered_id = F.normalize(id_raw - (p0 * id_raw).sum(dim=0, keepdim=True), dim=-1, eps=1e-6)

    target_sem_rms = centered_sem * math.sqrt(float(model.sem_real_dim))
    target_id_rms = centered_id * math.sqrt(float(model.id_real_dim))
    model.w_det_512.copy_(torch.cat([target_sem_rms, target_id_rms], dim=-1))
    target_table = model.w_det_512.detach().float() * model.radial_scale.detach().float().unsqueeze(1)

    if model.sem_ngram is not None:
        if isinstance(model.sem_ngram, VacuumAwareNGramProjector):
            model.sem_ngram.calibrate_from_training_shards_(stream, target_table)
        else:
            model.sem_ngram.calibrate_from_semantic_wavepackets_(stream.shard1_gpu, target_table)

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
        idx_in, targets = chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()
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

    if model.d_vac > 0:
        best_sem_t, best_id_t, best_vac_m, best_loss = 5.5, 0.8, 0.14, float("inf")
        for cand_vac in (0.08, 0.14, 0.22):
            model.id_vac_mass.fill_(cand_vac)
            for cand_sem in (3.5, 4.8, 5.5, 7.0):
                for cand_id in (0.6, 1.0, 1.5, 2.2):
                    model.temp_sem_scale.fill_(cand_sem)
                    model.temp_id_scale.fill_(cand_id)
                    tot_l = sum(float(model.forward_loss(x_b, y_b).item()) for x_b, y_b in cal_batches)
                    if tot_l < best_loss:
                        best_loss, best_sem_t, best_id_t, best_vac_m = tot_l, cand_sem, cand_id, cand_vac
        model.id_vac_mass.fill_(best_vac_m)
        model.temp_sem_scale.fill_(best_sem_t)
        model.temp_id_scale.fill_(best_id_t)
    else:
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
def evaluate_vacuum_cholesky_model(
    model: VacuumSubspaceCholeskyGPT, stream: LongSeqFineWebStream
) -> dict[str, float]:
    """Evaluate `VacuumSubspaceCholeskyGPT` across unseen Shard-0 validation batches (`65,536` tokens, 100% Pure Neural)."""
    was_training = model.training
    model.eval()

    total_neural_loss, branch_entropy_sum, anchor_id_cos_sum, excited_id_cos_sum = 0.0, 0.0, 0.0, 0.0
    total_top1, total_top5, total_tokens, n_batches = 0, 0, 0, 0
    cap, inv_cap = float(DEFAULT_SOFTCAP_LOGIT), 1.0 / float(DEFAULT_SOFTCAP_LOGIT)

    for idx_in, targets in stream.val_batches:
        bsz, seq_len = idx_in.shape
        n_tok = int(targets.numel())
        h_base_464, h_sem_branches, h_id_branches, branch_controls, w_aug_528 = model.build_multibranch_readout_tensors(idx_in)

        num_r, v_sz = h_id_branches.size(-2), w_aug_528.size(0)
        sem_dim, id_dim = h_sem_branches.size(-1), h_id_branches.size(-1)
        car_dim = h_base_464.size(-1) - sem_dim
        indep_sem = h_sem_branches.ndim == 4

        w_sem, w_id = w_aug_528[:, :sem_dim], w_aug_528[:, sem_dim : sem_dim + id_dim]
        w_car = w_aug_528[:, sem_dim + id_dim : sem_dim + id_dim + car_dim]
        w_base_t = torch.cat([w_sem, w_car], dim=-1).t().contiguous()
        w_sem_t, w_id_t = w_sem.t().contiguous(), w_id.t().contiguous()

        sem1_gates, id_scales, gate_logits = branch_controls[:, :, :, 0], branch_controls[:, :, :, 1], branch_controls[:, :, :, 2]

        pi_b = F.softmax(gate_logits.float(), dim=-1)
        ent_b = -(pi_b * torch.log(pi_b.clamp_min(1e-8))).sum(dim=-1).mean()
        branch_entropy_sum += float(ent_b.item())
        anchor_id_cos_sum += float(model.last_anchor_id_cos_vac)
        excited_id_cos_sum += float(model.last_excited_id_cos_vac)
        n_batches += 1

        for b0 in range(0, bsz, 1):
            b1, c_len = b0 + 1, seq_len
            yc = targets[b0:b1].reshape(-1)

            hb_c = h_base_464[b0:b1].reshape(c_len, sem_dim + car_dim)
            hid_c = h_id_branches[b0:b1].reshape(c_len * num_r, id_dim)
            b_c = sem1_gates[b0:b1].reshape(c_len, num_r, 1).float()
            t_c = id_scales[b0:b1].reshape(c_len, num_r, 1).float()
            log_pi = F.log_softmax(gate_logits[b0:b1].reshape(c_len, num_r).float(), dim=-1).unsqueeze(-1)

            z_base = torch.matmul(hb_c, w_base_t).float()
            z_id = torch.matmul(hid_c, w_id_t).float().view(c_len, num_r, v_sz)
            if indep_sem:
                hs_c = h_sem_branches[b0:b1].reshape(c_len * num_r, sem_dim)
                z_sem1 = torch.matmul(hs_c, w_sem_t).float().view(c_len, num_r, v_sz)
                raw_r = z_base.unsqueeze(1) + b_c * z_sem1 + t_c * z_id
            else:
                hs1_c = h_sem_branches[b0:b1].reshape(c_len, sem_dim)
                z_sem1 = torch.matmul(hs1_c, w_sem_t).float()
                raw_r = z_base.unsqueeze(1) + b_c * z_sem1.unsqueeze(1) + t_c * z_id

            z_r = cap * torch.tanh(raw_r * inv_cap)
            log_p_r = F.log_softmax(z_r, dim=-1)
            log_p_neural = torch.logsumexp(log_pi + log_p_r, dim=1)
            del raw_r, z_r, log_p_r, z_id, z_sem1

            total_neural_loss += float(F.nll_loss(log_p_neural, yc, reduction="sum").item())
            top5_idx = torch.topk(log_p_neural, k=5, dim=-1, largest=True, sorted=True).indices
            total_top1 += int((top5_idx[:, 0] == yc).sum().item())
            total_top5 += int((top5_idx == yc.unsqueeze(1)).any(dim=-1).sum().item())

        total_tokens += n_tok

    if stream.device.type == "cuda":
        torch.cuda.empty_cache()
    if was_training:
        model.train()

    neural_ce = total_neural_loss / float(max(1, total_tokens))
    mean_ent = branch_entropy_sum / float(max(1, n_batches))
    return {
        "val_ce": round(neural_ce, 4), "neural_val_ce": round(neural_ce, 4),
        "val_ppl": round(float(math.exp(min(20.0, neural_ce))), 2),
        "top1_acc": round(float(total_top1) / float(max(1, total_tokens)), 4),
        "top5_acc": round(float(total_top5) / float(max(1, total_tokens)), 4),
        "branch_gate_entropy_nats": round(mean_ent, 4),
        "effective_active_branches": round(float(math.exp(mean_ent)), 2),
        "mean_anchor_id_cos_theta_vac": round(anchor_id_cos_sum / float(max(1, n_batches)), 4),
        "mean_excited_id_cos_theta_vac": round(excited_id_cos_sum / float(max(1, n_batches)), 4),
    }


def _train_and_eval_vacuum_cholesky_variants(
    variants: list[tuple[str, VacuumCholeskyExperimentSpec, str]],
    stream: LongSeqFineWebStream,
    steps: int,
    eval_interval: int,
) -> list[dict[str, Any]]:
    """Shared Pure-Neural training and Shard-0 evaluation loop for `VacuumSubspaceCholeskyGPT` (`<= 8` args)."""
    dev = stream.device
    results: list[dict[str, Any]] = []
    for var_id, vac_spec, desc in variants:
        torch.manual_seed(int(vac_spec.seed))
        model = VacuumSubspaceCholeskyGPT(vac_spec).to(dev)
        init_ms, _ = initialize_and_calibrate_vacuum_cholesky_model_(model, stream)
        model.clamp_identity_dispersion_and_scalars_()

        intent = ExperimentIntentSpec(
            run_key=var_id.lower(), wave_id="W57", plan_id="EXPERIMENT_PLAN_078", task_domain="fineweb_speedrun",
            hypothesis_intent=desc,
            architecture_summary=f"VacuumSubspaceCholeskyGPT ({vac_spec.variant_mode.value}, d_vac={model.d_vac}, L=4, d=512, T=512)",
            hardware_budget="1x RTX 5060 Ti (16GB)", batch_size=stream.batch_size, seq_len=stream.seq_len,
            total_steps=steps, target_tokens_M=round(float(steps * stream.batch_size * stream.seq_len) / 1.0e6, 3),
        )
        exp_logger = StandardizedExperimentLogger(intent)

        stiefel_params: list[nn.Parameter] = list(model.exit_bridge.real_stiefel_parameters())
        unitary_pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
        for q_blk in model.quantum_blocks:
            stiefel_params.extend(q_blk.real_stiefel_parameters())  # type: ignore[operator]
            unitary_pairs.extend(q_blk.complex_unitary_pairs())  # type: ignore[operator]

        fast_temp_params = [model.temp_sem_scale, model.temp_id_scale, model.trunk_sem_gate, model.trunk_id_gate]
        scalar_params = [
            model.wp_causal_re, model.wp_causal_im, model.smear_gate.weight, model.smear_lambda,
            model.radial_scale, model.unigram_bias, model.prior_gate,
        ]
        scalar_params.extend(model.exit_bridge.scalar_parameters())
        scalar_params.extend(model.vacuum_scalar_parameters())
        for q_blk in model.quantum_blocks:
            scalar_params.extend(q_blk.scalar_parameters())  # type: ignore[operator]

        q_trunk_opt = ModularRiemannianStiefelKrausMuon(
            real_stiefel_params=stiefel_params, complex_operator_pairs=unitary_pairs,
            stiefel_lr=DEFAULT_SHALLOW_STIEFEL_LR, unitary_lr=DEFAULT_SHALLOW_UNITARY_LR,
            tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA, momentum_mode=StiefelMomentumMode.RIEMANNIAN_TANGENT_SNR_GATED,
        )
        wte_opt = BatchSubspaceTikhonovGramMuon([model.wte_3d], lr=DEFAULT_SHALLOW_WTE_LR, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA)
        head_opt = BatchSubspaceTikhonovGramMuon([model.w_det_512], lr=DEFAULT_SHALLOW_READOUT_LR, tikhonov_lambda=DEFAULT_TIKHONOV_GRAM_LAMBDA)

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
            if isinstance(model.sem_ngram, VacuumAwareNGramProjector):
                ngram_scalars.append(model.sem_ngram.purity_gate_gamma)
            adam_groups.append({"params": ngram_tables, "lr": DEFAULT_SHALLOW_NGRAM_LR, "weight_decay": 0.0})
            adam_groups.append({"params": ngram_scalars, "lr": DEFAULT_SHALLOW_SCALAR_LR, "weight_decay": 0.0})
        adam_opt = torch.optim.AdamW(adam_groups, betas=(0.80, 0.95), eps=1e-8, fused=(dev.type == "cuda"))

        s0 = evaluate_vacuum_cholesky_model(model, stream)
        s0["step"] = 0
        s0["tokens_M"] = 0.0
        curve = [s0]
        exp_logger.log_eval(0, {**s0, "top1_acc": round(100.0 * s0["top1_acc"], 2), "top5_acc": round(100.0 * s0["top5_acc"], 2)})
        print(
            f"[{var_id}] Step    0 | Pure Neural Val CE: {s0['val_ce']:.4f} (PPL {s0['val_ppl']:.2f}) | "
            f"Top1: {100.0 * s0['top1_acc']:.2f}% | Top5: {100.0 * s0['top5_acc']:.2f}% | "
            f"Modes: {s0['effective_active_branches']:.2f} | cos(theta_vac): {s0['mean_anchor_id_cos_theta_vac']:.3f}/{s0['mean_excited_id_cos_theta_vac']:.3f}",
            flush=True,
        )

        t0 = time.perf_counter()
        model.train()
        for step in range(1, steps + 1):
            warm_steps = max(1, int(0.08 * steps))
            lr_mult = (
                0.20 + 0.80 * (float(step - 1) / float(max(1, warm_steps - 1)))
                if step <= warm_steps
                else 0.12 + 0.88 * 0.5 * (1.0 + math.cos(math.pi * float(step - warm_steps) / float(max(1, steps - warm_steps))))
            )
            idx_in, targets = stream.sample_train_batch(step)
            q_trunk_opt.zero_grad(set_to_none=True)
            wte_opt.zero_grad(set_to_none=True)
            head_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)

            loss = model.forward_loss(idx_in, targets)
            loss.backward()

            mom_beta = compute_riemannian_momentum_beta(step, steps, QuantumMomentumScheduleMode.LOW_STEADY_MOMENTUM_070)
            q_trunk_opt.set_momentum(mom_beta)
            q_trunk_opt.set_lr_scale(lr_mult)
            q_trunk_opt.step()

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

            if step % eval_interval == 0 or step == steps:
                ev = evaluate_vacuum_cholesky_model(model, stream)
                ev["step"] = step
                ev["tokens_M"] = round(float(step * stream.batch_size * stream.seq_len) / 1.0e6, 3)
                ev["train_loss"] = round(float(loss.detach().item()), 4)
                ev["lr_mult"] = round(float(lr_mult), 4)
                curve.append(ev)
                exp_logger.log_eval(step, {**ev, "top1_acc": round(100.0 * ev["top1_acc"], 2), "top5_acc": round(100.0 * ev["top5_acc"], 2)})
                print(
                    f"[{var_id}] Step {step:4d} ({ev['tokens_M']:5.2f}M tok) | Train CE: {ev['train_loss']:.4f} | "
                    f"Pure Neural Val CE: {ev['val_ce']:.4f} (PPL {ev['val_ppl']:.2f}) | "
                    f"Top1: {100.0 * ev['top1_acc']:.2f}% | Top5: {100.0 * ev['top5_acc']:.2f}% | "
                    f"Modes: {ev['effective_active_branches']:.2f} | cos(theta_vac): {ev['mean_anchor_id_cos_theta_vac']:.3f}/{ev['mean_excited_id_cos_theta_vac']:.3f}",
                    flush=True,
                )

        elapsed = time.perf_counter() - t0
        results.append(
            {
                "variant_id": var_id, "variant_mode": vac_spec.variant_mode.value, "description": desc,
                "vacuum_dim": model.d_vac, "use_layer0_input_cholesky": vac_spec.use_layer0_input_cholesky,
                "use_independent_4mode_512d": vac_spec.use_independent_4mode_512d, "init_ms": round(init_ms, 2),
                "elapsed_sec": round(elapsed, 2), "step0": curve[0], "final": curve[-1], "curve": curve,
            }
        )
        del model, q_trunk_opt, wte_opt, head_opt, adam_opt
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    return results


def run_vacuum_subspace_cholesky_ablation_benchmark(
    steps: int = DEFAULT_FINEWEB_STEPS,
    eval_interval: int = DEFAULT_FINEWEB_EVAL_INTERVAL,
    device: str = "cuda",
    include_v0_baseline: bool = True,
) -> list[dict[str, Any]]:
    """Run 300-step (`2.46M`-token) FineWeb ablation of `V0..V3` (100% Pure Neural Evaluation)."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    stream = LongSeqFineWebStream(device=dev)
    variants: list[tuple[str, VacuumCholeskyExperimentSpec, str]] = []
    if include_v0_baseline:
        variants.append(
            (
                "V0_D8_UNIT_SPHERE_BASELINE",
                VacuumCholeskyExperimentSpec(
                    variant_mode=VacuumCholeskyVariantMode.V0_D8_UNIT_SPHERE_BASELINE,
                    vacuum_dim=0, use_layer0_input_cholesky=False, use_independent_4mode_512d=False, seed=1337,
                ),
                "Reference D8 Uncollapsed 4-Mode Cholesky without Vacuum Subspace (forced unit norm on S^63)",
            )
        )
    variants.extend(
        [
            (
                "V1_VACUUM_SUBSPACE_STEP0_PURE_NEURAL",
                VacuumCholeskyExperimentSpec(
                    variant_mode=VacuumCholeskyVariantMode.V1_VACUUM_SUBSPACE_STEP0_PURE_NEURAL,
                    vacuum_dim=DEFAULT_VACUUM_SUBSPACE_DIM, use_layer0_input_cholesky=False, use_independent_4mode_512d=False, seed=1337,
                ),
                "Orthogonal Vacuum Subspace H_vac (d_vac=4) on Readout & Bridged Cholesky Modes + Purity-Preserving Step-0 Calibration",
            ),
            (
                "V2_VACUUM_PLUS_LAYER0_INPUT_CHOLESKY",
                VacuumCholeskyExperimentSpec(
                    variant_mode=VacuumCholeskyVariantMode.V2_VACUUM_PLUS_LAYER0_INPUT_CHOLESKY,
                    vacuum_dim=DEFAULT_VACUUM_SUBSPACE_DIM, use_layer0_input_cholesky=True, use_independent_4mode_512d=False, seed=1337,
                ),
                "Vacuum Subspace H_vac + Layer-0 Multi-Horizon Rank-4 Cholesky Initialization L^{(0)}_{t,h} in C^{4x32}",
            ),
            (
                "V3_VACUUM_4MODE_INDEPENDENT_512D_CHOLESKY",
                VacuumCholeskyExperimentSpec(
                    variant_mode=VacuumCholeskyVariantMode.V3_VACUUM_4MODE_INDEPENDENT_512D_CHOLESKY,
                    vacuum_dim=DEFAULT_VACUUM_SUBSPACE_DIM, use_layer0_input_cholesky=True, use_independent_4mode_512d=True, seed=1337,
                ),
                "Vacuum Subspace H_vac + Layer-0 Cholesky + True 4-Mode Independent 512D (448D Sem + 64D ID) Cholesky Readout",
            ),
        ]
    )
    return _train_and_eval_vacuum_cholesky_variants(variants, stream, steps, eval_interval)


def run_vacuum_subspace_cholesky_long_scaling_benchmark(
    steps: int = DEFAULT_LONG_FINEWEB_STEPS,
    eval_interval: int = DEFAULT_LONG_FINEWEB_EVAL_INTERVAL,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """Run `1,200`-step (`9.83M`-token) Pure-Neural FineWeb scaling run (`V4_PURE_NEURAL_VACUUM_CHOLESKY_SCALING_1200`)."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    stream = LongSeqFineWebStream(device=dev)
    variants = [
        (
            "V4_PURE_NEURAL_VACUUM_CHOLESKY_SCALING_1200",
            VacuumCholeskyExperimentSpec(
                variant_mode=VacuumCholeskyVariantMode.V4_PURE_NEURAL_VACUUM_CHOLESKY_SCALING_1200,
                vacuum_dim=DEFAULT_VACUUM_SUBSPACE_DIM, use_layer0_input_cholesky=True, use_independent_4mode_512d=True, seed=1337,
            ),
            "1,200-step (9.83M-token) 100% Pure-Neural Scaling Run with Vacuum Subspace H_vac + Layer-0 Cholesky + 4-Mode 512D Readout",
        )
    ]
    return _train_and_eval_vacuum_cholesky_variants(variants, stream, steps, eval_interval)
