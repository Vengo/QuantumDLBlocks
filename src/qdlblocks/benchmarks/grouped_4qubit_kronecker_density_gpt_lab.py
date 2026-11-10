r"""Wave 72 (`EXPERIMENT_PLAN_092.md`): Grouped 4-Qubit Kronecker Density GPT (`(C^2)^{x4} = C^16`, `16^4 = 65,536D`).

Implements Theorems 13a, 13b, & 13c from `lean/QuantumDLBlocksLean/GroupedFourQubitKroneckerDensity.lean`:
1. Grouped 4-Qubit Segre Preparation (`Theorem 13a`): Prepares each vocabulary token $v \in \{0 \dots 50256\}$
   from 16 single-qubit spinors $|\psi_{v,g,q}\rangle \in \mathbb{C}^2$ ($g \in \{1..4\},\; q \in \{1..4\}$)
   initialized around the token's unique 16-bit orthogonal binary codeword in $(\mathbb{C}^2)^{\otimes 16} = \mathbb{C}^{65,536}$,
   and forms the exact 4-qubit Kronecker product within each $\mathbb{C}^{16}$ head group:
   $$|\Phi_{t,g}^{(0)}\rangle = |\psi_{t,g,1}\rangle \otimes |\psi_{t,g,2}\rangle \otimes |\psi_{t,g,3}\rangle \otimes |\psi_{t,g,4}\rangle \in (\mathbb{C}^2)^{\otimes 4} \cong \mathbb{C}^{16}.$$
2. Intra-Head 4-Qubit Entanglement Diagnostics (`Theorem 13a`): Computes the exact 2-qubit partial trace
   $\rho_A = \operatorname{Tr}_{\text{qubits } 3,4}(\rho) \in \mathcal{D}(\mathbb{C}^4)$ and its purity
   $\operatorname{Tr}(\rho_A^2) \in [\tfrac{1}{4}, 1]$, verifying that Layer-0 Segre states have purity $1.0$
   while trunk $SU(16)$ and Kraus channels generate genuine multi-qubit entanglement ($\operatorname{Tr}(\rho_A^2) < 1$).
3. Multi-Branch $16^4 = 65,536\text{D}$ Kronecker Coincidence Readout (`Theorems 13b & 13c`): Partitions the
   $H = 16$ heads into $B = 4$ branches $\times$ $G = 4$ Kronecker factors to represent the entangled
   $65,536\text{D}$ mixture $\boldsymbol{\rho}_t^{\text{joint}} = \sum_{b=1}^4 w_{t,b} \bigotimes_{g=1}^4 \rho_{t,(b,g)}^{(L)}$
   without mean-field superposition collapse and without materializing $65,536\text{D}$ tensors.
"""

from __future__ import annotations

import enum
import gc
import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_ckpt

from qdlblocks.benchmarks.anchored_bipartite_density_gpt_lab import (
    AnchoredBipartitePureDensityGPT,
    Wave71ExperimentSpec,
    Wave71VariantMode,
    _build_deterministic_haar_unitary_stack,
    _logit_from_prob,
)
from qdlblocks.benchmarks.complementary_512d_lindblad_8l_lab import _collect_active_vocab_indices_w68
from qdlblocks.benchmarks.experiment_logger import (
    ExperimentIntentSpec,
    StandardizedExperimentLogger,
)
from qdlblocks.benchmarks.explicit_density_tomographic_512d_lab import _complex_hermitian_mv
from qdlblocks.benchmarks.factored_kraus_density_gpt_lab import compute_restored_riemannian_momentum_schedule
from qdlblocks.benchmarks.horizontal_grassmann_gauge_free_lab import compute_gauge_locked_mub_probe_expectations
from qdlblocks.benchmarks.pure_density_matrix_gpt_lab import (
    DEFAULT_PURE_CHOLESKY_RANK,
    DEFAULT_PURE_RHO_DIM,
    DEFAULT_PURE_RHO_HEADS,
    DEFAULT_TOK_NGRAM_BUCKETS,
    PURE_RHO_TRACE_EPS,
    apply_complex_unitary_to_cholesky,
    build_wave69_optimizers,
    evaluate_wave69_model,
    normalize_cholesky_factor_frobenius,
    normalize_packed_density_trace,
    pack_cholesky_factor_outer_isometric,
    unpack_hermitian_isometric,
)
from qdlblocks.benchmarks.selective_exact_quotient_lab import (
    DEFAULT_NARROW_SUBSPACE_DIM,
    DEFAULT_THERMOSTAT_EMA_DECAY,
    ReadOnlyQuenchRiemannianShadowEMA,
)
from qdlblocks.benchmarks.shallow_longseq_wavepacket_lm_lab import LongSeqFineWebStream
from qdlblocks.layers.impure_density_mirror_block import complex_l2_normalize

# ==============================================================================
# Top-of-File Named Constants (Rule 7 — Synchronized with EXPERIMENT_PLAN_092.md)
# ==============================================================================
DEFAULT_KRONECKER_GROUPS: int = 4
DEFAULT_QUBITS_PER_GROUP: int = 4
DEFAULT_KRONECKER_BRANCHES: int = 4
DEFAULT_SEGRE_QUBIT_BLEND_INIT: float = 0.65
DEFAULT_KRON_COINCIDENCE_SCALE_INIT: float = 0.45
QUBIT_BINARY_ORTHO_PERTURB_STD: float = 0.08


class Wave72VariantMode(str, enum.Enum):
    """Wave 72 (`EXPERIMENT_PLAN_092.md`) ablation modes."""

    W72_Q0_ANCHORED_BIPARTITE_REF_4L = "W72_Q0_ANCHORED_BIPARTITE_REF_4L"
    W72_Q1_GROUPED_4QUBIT_SEGRE_PREP_4L = "W72_Q1_GROUPED_4QUBIT_SEGRE_PREP_4L"
    W72_Q2_GROUPED_4QUBIT_MULTIBRANCH_KRON_4L = "W72_Q2_GROUPED_4QUBIT_MULTIBRANCH_KRON_4L"
    W72_Q3_GROUPED_4QUBIT_MULTIBRANCH_KRON_8L = "W72_Q3_GROUPED_4QUBIT_MULTIBRANCH_KRON_8L"


@dataclass(frozen=True)
class Wave72ExperimentSpec:
    """Configuration for Wave 72 `GroupedFourQubitKroneckerDensityGPT` (`<= 8` fields)."""

    variant_mode: Wave72VariantMode = Wave72VariantMode.W72_Q2_GROUPED_4QUBIT_MULTIBRANCH_KRON_4L
    num_layers: int = 4
    kronecker_groups: int = DEFAULT_KRONECKER_GROUPS
    qubits_per_group: int = DEFAULT_QUBITS_PER_GROUP
    enable_grouped_4qubit_segre_prep: bool = True
    enable_multibranch_kron_readout: bool = True
    seed: int = 1337


# ==============================================================================
# 1. Grouped 4-Qubit Segre Product & Intra-Head Entanglement Helpers (Theorem 13a)
# ==============================================================================
def _complex_mul(a_re: torch.Tensor, a_im: torch.Tensor, b_re: torch.Tensor, b_im: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Elementwise complex multiplication $(a_{\\text{re}} + i a_{\\text{im}})(b_{\\text{re}} + i b_{\\text{im}})$ (`<= 8` args)."""
    return a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re


def compute_grouped_4qubit_segre_product(
    qubits_re: torch.Tensor,
    qubits_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Form the exact 4-qubit Segre Kronecker product in $(\mathbb{C}^2)^{\otimes 4} \cong \mathbb{C}^{16}$ (`Lean 4 Theorem 13a`, `<= 8` args).

    Args:
        qubits_re: Real part of normalized 4-qubit spinors `(..., 4, 2)` where axis `-2` is qubit $q \in \{0,1,2,3\}$
            and axis `-1` is spinor basis $|0\rangle, |1\rangle \in \mathbb{C}^2$.
        qubits_im: Imaginary part of normalized 4-qubit spinors `(..., 4, 2)`.

    Returns:
        `(phi_re, phi_im)` of shape `(..., 16)` with exact unit $\ell_2$ norm $\|\Phi\|_2^2 = \prod_{q=1}^4 \|\psi_q\|_2^2 = 1$.
    """
    q_norm = torch.sqrt((qubits_re.square() + qubits_im.square()).sum(dim=-1, keepdim=True).clamp_min(PURE_RHO_TRACE_EPS))
    qr, qi = qubits_re / q_norm, qubits_im / q_norm

    # Pairwise Kronecker products: (q0 x q1) in C^4 and (q2 x q3) in C^4, then (q01 x q23) in C^16
    q01_re, q01_im = _complex_mul(qr[..., 0, :, None], qi[..., 0, :, None], qr[..., 1, None, :], qi[..., 1, None, :])
    q23_re, q23_im = _complex_mul(qr[..., 2, :, None], qi[..., 2, :, None], qr[..., 3, None, :], qi[..., 3, None, :])

    q01_flat_re = q01_re.reshape(*qr.shape[:-2], 4)
    q01_flat_im = q01_im.reshape(*qi.shape[:-2], 4)
    q23_flat_re = q23_re.reshape(*qr.shape[:-2], 4)
    q23_flat_im = q23_im.reshape(*qi.shape[:-2], 4)

    phi_re, phi_im = _complex_mul(q01_flat_re[..., :, None], q01_flat_im[..., :, None], q23_flat_re[..., None, :], q23_flat_im[..., None, :])
    out_shape = (*qr.shape[:-2], 16)
    return phi_re.reshape(out_shape), phi_im.reshape(out_shape)


def build_deterministic_16qubit_binary_codebook(
    vocab_size: int,
    num_groups: int = DEFAULT_KRONECKER_GROUPS,
    qubits_per_group: int = DEFAULT_QUBITS_PER_GROUP,
    perturb_std: float = QUBIT_BINARY_ORTHO_PERTURB_STD,
    seed: int = 1337,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Build 16-qubit codeword spinors `(V, G=4, Q=4, 2)` in $(\mathbb{C}^2)^{16}$ (`Theorem 13a`, `<= 8` args).

    When `perturb_std == 0.0`, every token $v \in \{0 \dots V-1\}$ is mapped to its unique 16-bit binary
    eigenstate rotated by a shared per-qubit Haar $SU(2)$ basis, guaranteeing exact mutual orthogonality:
    $$\forall u \neq v,\qquad \langle \Psi(u) | \Psi(v) \rangle = \prod_{g=1}^4 \langle \Phi_{u,g} | \Phi_{v,g} \rangle = 0.$$
    """
    dev = device or torch.device("cpu")
    total_qubits = int(num_groups * qubits_per_group)
    v_idx = torch.arange(vocab_size, device=dev, dtype=torch.long).unsqueeze(1)
    bit_pos = torch.arange(total_qubits, device=dev, dtype=torch.long).unsqueeze(0)
    bits = ((v_idx >> bit_pos) & 1).view(vocab_size, num_groups, qubits_per_group)

    # Computational basis spinor |b> in C^2: (1 - b, b)
    base_re = torch.stack([1.0 - bits.float(), bits.float()], dim=-1)
    base_im = torch.zeros_like(base_re)

    gen = torch.Generator(device=dev).manual_seed(int(seed) + 7200)
    if perturb_std > 0.0:
        base_re = base_re + float(perturb_std) * torch.randn(base_re.shape, generator=gen, device=dev)
        base_im = base_im + float(perturb_std) * torch.randn(base_im.shape, generator=gen, device=dev)

    # Apply a shared per-qubit Haar SU(2) rotation so computational axes are isotropically distributed
    u2_re, u2_im = _build_deterministic_haar_unitary_stack(total_qubits, 2, int(seed) + 7272)
    u2_re = u2_re.to(dev).view(1, num_groups, qubits_per_group, 2, 2)
    u2_im = u2_im.to(dev).view(1, num_groups, qubits_per_group, 2, 2)

    rot_re = torch.einsum("vgqij,vgqj->vgqi", u2_re, base_re) - torch.einsum("vgqij,vgqj->vgqi", u2_im, base_im)
    rot_im = torch.einsum("vgqij,vgqj->vgqi", u2_re, base_im) + torch.einsum("vgqij,vgqj->vgqi", u2_im, base_re)
    q_norm = torch.sqrt((rot_re.square() + rot_im.square()).sum(dim=-1, keepdim=True).clamp_min(PURE_RHO_TRACE_EPS))
    return rot_re / q_norm, rot_im / q_norm


def compute_four_qubit_partial_trace_purity(
    l_re: torch.Tensor,
    l_im: torch.Tensor,
) -> torch.Tensor:
    r"""Compute 2-qubit bipartition purity $\operatorname{Tr}(\rho_A^2)$ of $\rho = L L^\dagger \in \mathcal{D}((\mathbb{C}^2)^{\otimes 4})$ (`Theorem 13a`, `<= 8` args).

    Bipartitions $\mathbb{C}^{16} \cong \mathbb{C}^4_A \otimes \mathbb{C}^4_B$ (Qubits $\{1,2\}$ vs. Qubits $\{3,4\}$),
    computes $\rho_A = \operatorname{Tr}_B(L L^\dagger) \in \mathcal{D}(\mathbb{C}^4)$, and returns $\operatorname{Tr}(\rho_A^2) \in [\tfrac{1}{4}, 1]$.
    For any separable 4-qubit Segre product state $|\Phi\rangle = \bigotimes_{q=1}^4 |\psi_q\rangle$, $\operatorname{Tr}(\rho_A^2) \equiv 1.0$.
    """
    r_rank = l_re.size(-1)
    la_re = l_re.float().reshape(*l_re.shape[:-2], 4, 4, r_rank)
    la_im = l_im.float().reshape(*l_im.shape[:-2], 4, 4, r_rank)
    # Reduced density matrix rho_A[i, j] = sum_{b=0..3, r=0..R-1} L[i, b, r] * conj(L[j, b, r])
    rho_a_re = torch.einsum("...ibr,...jbr->...ij", la_re, la_re) + torch.einsum("...ibr,...jbr->...ij", la_im, la_im)
    rho_a_im = torch.einsum("...ibr,...jbr->...ij", la_im, la_re) - torch.einsum("...ibr,...jbr->...ij", la_re, la_im)
    return (rho_a_re.square() + rho_a_im.square()).sum(dim=(-2, -1))


# ==============================================================================
# 2. Multi-Branch 16^4 = 65,536D Kronecker Coincidence Readout (Theorems 13b & 13c)
# ==============================================================================
def evaluate_factorized_65536d_kronecker_born_overlap(
    l_branch_group_pair: tuple[torch.Tensor, torch.Tensor],
    phi_vocab_group_pair: tuple[torch.Tensor, torch.Tensor],
    branch_weights: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate exact $65,536\text{D}$ Born overlap $\operatorname{Tr}(\boldsymbol{\rho}^{\text{joint}} E_v)$ without materializing $\mathbb{C}^{65,536}$ (`Theorem 13c`, `<= 8` args).

    Computes:
    $$\operatorname{Tr}\!\left(\boldsymbol{\rho}^{\text{joint}} \bigotimes_{g=1}^4 |\Phi_{v,g}\rangle\langle\Phi_{v,g}|\right)
      = \sum_{b=1}^{B} w_b \prod_{g=1}^{4} \left\| L_{(b,g)}^\dagger |\Phi_{v,g}\rangle \right\|_2^2$$
    where $L_{(b,g)} \in \mathbb{C}^{16 \times R}$ (`(..., B, G, 16, R)`) and $|\Phi_{v,g}\rangle \in \mathbb{C}^{16}$ (`(V, G, 16)`).
    """
    l_re, l_im = l_branch_group_pair[0].float(), l_branch_group_pair[1].float()
    phi_re, phi_im = phi_vocab_group_pair[0].float(), phi_vocab_group_pair[1].float()

    # Overlap <L_{b,g,r} | Phi_{v,g}> = sum_i (L_re * Phi_re + L_im * Phi_im) + i (L_re * Phi_im - L_im * Phi_re)
    ov_re = torch.einsum("...bgir,vgi->...vbgr", l_re, phi_re) + torch.einsum("...bgir,vgi->...vbgr", l_im, phi_im)
    ov_im = torch.einsum("...bgir,vgi->...vbgr", l_re, phi_im) - torch.einsum("...bgir,vgi->...vbgr", l_im, phi_re)
    factor_trace = (ov_re.square() + ov_im.square()).sum(dim=-1)  # (..., V, B, G)
    branch_prod = factor_trace.prod(dim=-1)  # (..., V, B)
    w_b = branch_weights.float()
    return (branch_prod * w_b).sum(dim=-1)


def compute_multibranch_kronecker_65536d_coincidence_features(
    rho_packed: torch.Tensor,
    w_kron_obs: torch.Tensor,
    branch_logits: torch.Tensor,
    u_branch_proj: torch.Tensor,
) -> torch.Tensor:
    r"""Project $B=4$ branches $\times$ $G=4$ Kronecker factors in $\mathcal{D}(\mathbb{C}^{65,536})$ into $\mathbb{R}^{512}$ (`Theorems 13b & 13c`, `<= 8` args).

    1. Reshapes the $H = 16$ packed density matrices `(B_sz, H=16, T, 256)` into `(B_sz, B=4, G=4, T, 256)`.
    2. Evaluates positive POVM/observable expectations $p_{t,b,g,k} = \operatorname{Softplus}(\langle \tilde{\rho}_{t,(b,g)}, W_{b,g,k} \rangle) + \varepsilon$
       across $K = 32$ modes per $(b, g)$.
    3. Forms the **4-factor Kronecker geometric coincidence** across $g \in \{1, 2, 3, 4\}$ within each branch $b$:
       $$C_{t,b,k} = \left(\prod_{g=1}^{4} p_{t,b,g,k}\right)^{1/4} = \exp\!\left(\frac{1}{4}\sum_{g=1}^{4} \ln p_{t,b,g,k}\right),$$
       and mixes across the $B=4$ entangled branches weighted by $w_b = \operatorname{softmax}(\beta_{\text{branch}})_b$.
    """
    bsz, num_h, seq_len, packed_dim = rho_packed.shape
    work_dtype = rho_packed.dtype
    b_branches = DEFAULT_KRONECKER_BRANCHES
    g_groups = num_h // b_branches

    rho_bg = rho_packed.view(bsz, b_branches, g_groups, seq_len, packed_dim)
    obs_bg = w_kron_obs.to(work_dtype).view(1, b_branches, g_groups, packed_dim, -1)
    raw_proj = torch.matmul(rho_bg, obs_bg)  # (B_sz, B=4, G=4, T, K=32)

    # Positive factor intensities and geometric-mean 4-factor coincidence across G=4 Kronecker factors
    pos_int = F.softplus(raw_proj.float()).clamp_min(1e-5)
    coinc_branch = torch.exp(0.25 * torch.log(pos_int).sum(dim=2)).to(work_dtype)  # (B_sz, B=4, T, K=32)

    w_b = F.softmax(branch_logits.float(), dim=0).to(work_dtype).view(1, b_branches, 1, 1)
    # Project weighted branch coincidence features (B=4 * K=32 = 128) via St(512, 128) -> 512D
    coinc_weighted = (coinc_branch * w_b).permute(0, 2, 1, 3).reshape(bsz, seq_len, -1)  # (B_sz, T, 128)
    proj_512 = torch.matmul(coinc_weighted, u_branch_proj.to(work_dtype).t())  # (B_sz, T, 512)
    return F.rms_norm(proj_512, (proj_512.size(-1),), eps=1e-6)


# ==============================================================================
# 3. Wave 72 Model: GroupedFourQubitKroneckerDensityGPT
# ==============================================================================
class GroupedFourQubitKroneckerDensityGPT(AnchoredBipartitePureDensityGPT):
    r"""Wave 72 (`EXPERIMENT_PLAN_092.md`): Grouped 4-Qubit ($\mathbb{C}^{16} = (\mathbb{C}^2)^{\otimes 4}$) Kronecker Density GPT."""

    def __init__(self, spec: Wave72ExperimentSpec | None = None) -> None:
        self.w72_spec = spec or Wave72ExperimentSpec()
        super().__init__(
            Wave71ExperimentSpec(
                variant_mode=Wave71VariantMode.W71_A2_ANCHORED_BIPARTITE_DECOUPLED_4L,
                num_layers=self.w72_spec.num_layers,
                enable_sensory_anchor=True,
                enable_bipartite_register=True,
                enable_decoupled_io_frames=True,
                seed=self.w72_spec.seed,
            )
        )
        self.kron_groups = int(self.w72_spec.kronecker_groups)
        self.qubits_per_group = int(self.w72_spec.qubits_per_group)
        self.kron_branches = self.num_heads // self.kron_groups

        # Branch-specific SU(16) frames mapping the 4-group Segre product states into H = B * G = 16 heads
        ub_re, ub_im = _build_deterministic_haar_unitary_stack(self.num_heads, self.d_h, int(self.w72_spec.seed) + 7210)
        self.segre_head_u_re = nn.Parameter(ub_re)
        self.segre_head_u_im = nn.Parameter(ub_im)
        self.segre_qubit_blend_logit = nn.Parameter(
            torch.full((self.num_heads,), _logit_from_prob(DEFAULT_SEGRE_QUBIT_BLEND_INIT), dtype=torch.float32)
        )

        # Multi-branch 16^4 = 65,536D Kronecker coincidence readout parameters on St(512, 128)
        self.kron_branch_logits = nn.Parameter(torch.zeros(self.kron_branches, dtype=torch.float32))
        self.kron_coincidence_scale = nn.Parameter(torch.tensor(DEFAULT_KRON_COINCIDENCE_SCALE_INIT, dtype=torch.float32))
        self.u_kron_branch_proj = nn.Parameter(torch.empty(self.embed_dim, self.kron_branches * 32, dtype=torch.float32))
        self.initialize_quantum_state(seed=self.w72_spec.seed)

    @torch.no_grad()
    def initialize_quantum_state(self, seed: int = 1337) -> float:
        """Initialize boundary frames, Segre head unitaries, and Kronecker branch projection (`<= 8` args)."""
        elapsed_ms = super().initialize_quantum_state(seed=seed)
        if getattr(self, "u_kron_branch_proj", None) is not None:
            dev = self.u_kron_branch_proj.device
            gen = torch.Generator(device=dev).manual_seed(int(seed) + 7299)
            q_full, _ = torch.linalg.qr(torch.randn(self.embed_dim, self.embed_dim, generator=gen, device=dev))
            self.u_kron_branch_proj.copy_(q_full[:, : self.kron_branches * 32])
            self.kron_branch_logits.zero_()
            self.kron_coincidence_scale.fill_(DEFAULT_KRON_COINCIDENCE_SCALE_INIT)
        return elapsed_ms

    def extra_real_stiefel_parameters(self) -> list[nn.Parameter]:
        """Include `u_kron_branch_proj` in real Stiefel manifold optimization (`<= 8` args)."""
        params = super().extra_real_stiefel_parameters()
        if self.w72_spec.enable_multibranch_kron_readout:
            params.append(self.u_kron_branch_proj)
        return params

    def mub_su_parameter_pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        r"""Include `segre_head_u` in $\mathfrak{su}(16)$ unitary optimization (`<= 8` args)."""
        pairs = super().mub_su_parameter_pairs()
        if self.w72_spec.enable_grouped_4qubit_segre_prep:
            pairs.append((self.segre_head_u_re, self.segre_head_u_im))
        return pairs

    def scalar_parameters(self) -> list[nn.Parameter]:
        """Include Segre blend and Kronecker branch scalars (`<= 8` args)."""
        return super().scalar_parameters() + [
            self.segre_qubit_blend_logit,
            self.kron_branch_logits,
            self.kron_coincidence_scale,
        ]

    @torch.no_grad()
    def restrict_scalars_(self) -> None:
        """Clamp Wave 72 scalar gates in-place (`<= 8` args)."""
        super().restrict_scalars_()
        self.segre_qubit_blend_logit.clamp_(-3.5, 3.5)
        self.kron_branch_logits.clamp_(-3.0, 3.0)
        self.kron_coincidence_scale.clamp_(0.05, 2.5)

    def prepare_grouped_4qubit_segre_kets(
        self,
        idx: torch.Tensor,
        raw_wave_512: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Prepare Layer-0 kets via Grouped 4-Qubit Segre Kronecker products in $(\mathbb{C}^2)^{\otimes 4} = \mathbb{C}^{16}$ (`<= 8` args).

        Extracts 16 single-qubit spinors per token (combining the token's continuous `raw_wave_512` coordinates
        with its deterministic 16-bit binary orthogonal qubit codeword), forms the 4-qubit Kronecker product
        $|\Phi_{t,g}^{(0)}\rangle = \bigotimes_{q=1}^4 |\psi_{t,g,q}\rangle \in \mathbb{C}^{16}$ for each of the $G = 4$ groups,
        and maps the 4 groups across $H = B \times G = 16$ heads via $U_{\text{segre}, h} \in SU(16)$.
        """
        bsz, seq_len = idx.shape
        work_dtype = raw_wave_512.dtype
        # Extract 16 continuous qubit spinors from the first 64 real coordinates of raw_wave_512: (B, T, G=4, Q=4, 2_re_im, 2_spinor)
        q_cont = raw_wave_512[..., :64].view(bsz, seq_len, self.kron_groups, self.qubits_per_group, 2, 2)
        q_cont_re, q_cont_im = q_cont[..., 0, :], q_cont[..., 1, :]

        # Deterministic 16-bit binary qubit codeword for exact 65,536D distinctness: bit (4g + q) of token idx
        bit_pos = torch.arange(16, device=idx.device, dtype=torch.long).view(1, 1, self.kron_groups, self.qubits_per_group)
        bits = ((idx.to(torch.long).unsqueeze(-1).unsqueeze(-1) >> bit_pos) & 1).to(work_dtype)
        q_bin_re = torch.stack([1.0 - bits, bits], dim=-1)
        q_re = q_cont_re + 0.50 * q_bin_re
        q_im = q_cont_im

        # Compute exact 4-qubit Segre Kronecker product in C^16 for each of the G=4 groups: (B, T, G=4, 16)
        segre_g_re, segre_g_im = compute_grouped_4qubit_segre_product(q_re, q_im)

        # Broadcast across B=4 branches -> H = B * G = 16 heads: (B, H=16, T, 16, 1)
        segre_h_re = segre_g_re.permute(0, 2, 1, 3).unsqueeze(1).expand(bsz, self.kron_branches, self.kron_groups, seq_len, self.d_h)
        segre_h_im = segre_g_im.permute(0, 2, 1, 3).unsqueeze(1).expand(bsz, self.kron_branches, self.kron_groups, seq_len, self.d_h)
        segre_h_re = segre_h_re.reshape(bsz, self.num_heads, seq_len, self.d_h, 1)
        segre_h_im = segre_h_im.reshape(bsz, self.num_heads, seq_len, self.d_h, 1)

        rot_re, rot_im = apply_complex_unitary_to_cholesky(segre_h_re, segre_h_im, self.segre_head_u_re, self.segre_head_u_im)
        return complex_l2_normalize(rot_re.squeeze(-1), rot_im.squeeze(-1))

    def prepare_layer0_density_state(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Prepare Layer-0 multi-head density state $(L_{t,h}^{(0)}, \tilde{\rho}_{t,h}^{(0)})$ with Grouped 4-Qubit Segre kets (`<= 8` args)."""
        if not self.w72_spec.enable_grouped_4qubit_segre_prep:
            return super().prepare_layer0_density_state(idx)

        bsz, seq_len = idx.shape
        work_dtype = torch.bfloat16 if (idx.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
        idx_l = idx.to(torch.long)
        idx_p1 = torch.cat([idx_l[:, :1], idx_l[:, :-1]], dim=1) if seq_len > 1 else idx_l
        idx_p2 = torch.cat([idx_l[:, :2], idx_l[:, :-2]], dim=1) if seq_len > 2 else idx_p1
        h_tok = (idx_l * 1000003 + idx_p1 * 65537 + idx_p2 * 257) & (DEFAULT_TOK_NGRAM_BUCKETS - 1)

        in_mup = math.sqrt(float(DEFAULT_NARROW_SUBSPACE_DIM) / float(self.subspace_dim))
        raw_wave = F.rms_norm((self.boundary.encode_tokens(idx) * in_mup).to(work_dtype), (self.embed_dim,), eps=1e-6)
        psi_0 = raw_wave.view(bsz, seq_len, self.num_heads, 2, self.d_h).permute(0, 2, 1, 3, 4)
        psi_flat_re, psi_flat_im = complex_l2_normalize(psi_0[..., 0, :], psi_0[..., 1, :])

        # Blend Grouped 4-Qubit Segre Kronecker state into Layer-0 token ket
        seg_re, seg_im = self.prepare_grouped_4qubit_segre_kets(idx, raw_wave)
        gam_s = torch.sigmoid(self.segre_qubit_blend_logit.float()).to(work_dtype).view(1, self.num_heads, 1, 1)
        psi_raw_re, psi_raw_im = complex_l2_normalize((1.0 - gam_s) * psi_flat_re + gam_s * seg_re, (1.0 - gam_s) * psi_flat_im + gam_s * seg_im)

        ng_re, ng_im, ema_rho_packed = self.ngram_density_bank.lookup_and_update(h_tok, psi_raw_re, psi_raw_im)
        psi0_re, psi0_im = complex_l2_normalize(psi_raw_re + 0.30 * ng_re, psi_raw_im + 0.30 * ng_im)

        prev_re = torch.cat([psi0_re[:, :, :1, :], psi0_re[:, :, :-1, :]], dim=2) if seq_len > 1 else psi0_re
        prev_im = torch.cat([psi0_im[:, :, :1, :], psi0_im[:, :, :-1, :]], dim=2) if seq_len > 1 else psi0_im
        cos_p = torch.cos(self.lag_phase.float()).to(work_dtype).view(1, self.num_heads, 1, 1)
        sin_p = torch.sin(self.lag_phase.float()).to(work_dtype).view(1, self.num_heads, 1, 1)
        lag_re, lag_im = prev_re * cos_p - prev_im * sin_p, prev_re * sin_p + prev_im * cos_p
        coinc_re, coinc_im = complex_l2_normalize(psi0_re * lag_re - psi0_im * lag_im, psi0_re * lag_im + psi0_im * lag_re)

        pi_w = torch.sqrt(F.softmax(self.layer0_branch_logits.float(), dim=0).clamp_min(PURE_RHO_TRACE_EPS)).to(work_dtype)
        # Mode 0 is pure Grouped 4-Qubit Segre ket, Mode 1 is sensory-blended ket, Modes 2 & 3 are contextual lag/n-gram kets
        l0_re = torch.stack([pi_w[0] * seg_re, pi_w[1] * psi0_re, pi_w[2] * coinc_re, pi_w[3] * ng_re], dim=-1)
        l0_im = torch.stack([pi_w[0] * seg_im, pi_w[1] * psi0_im, pi_w[2] * coinc_im, pi_w[3] * ng_im], dim=-1)
        l0_re, l0_im = normalize_cholesky_factor_frobenius(l0_re, l0_im)

        rho0_coh = pack_cholesky_factor_outer_isometric(l0_re, l0_im)
        g_ema = (0.35 * torch.sigmoid(self.ngram_density_bank.ema_blend_logit.float())).to(work_dtype).view(1, self.num_heads, 1, 1)
        rho0_packed = normalize_packed_density_trace((1.0 - g_ema) * rho0_coh + g_ema * ema_rho_packed, self.d_h)
        return l0_re, l0_im, rho0_packed

    def encode_pure_density_trunk(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Evolve $(L_{t,h}^{(\ell)}, \tilde{\rho}_{t,h}^{(\ell)})$ and apply Multi-Branch $16^4 = 65,536\text{D}$ Kronecker Coincidence Readout (`<= 8` args)."""
        u_vocab_512, l_re, rho_packed = super().encode_pure_density_trunk(idx)
        if not self.w72_spec.enable_multibranch_kron_readout:
            return u_vocab_512, l_re, rho_packed

        q_kron_512 = compute_multibranch_kronecker_65536d_coincidence_features(
            rho_packed, self.w_herm_obs, self.kron_branch_logits, self.u_kron_branch_proj
        )
        u_vocab_512 = u_vocab_512 + self.kron_coincidence_scale.to(u_vocab_512.dtype) * q_kron_512
        return u_vocab_512, l_re, rho_packed


def build_wave72_variant_spec(mode: Wave72VariantMode, seed: int = 1337) -> Wave72ExperimentSpec:
    """Resolve `Wave72ExperimentSpec` from `Wave72VariantMode` (`<= 8` args)."""
    if mode == Wave72VariantMode.W72_Q0_ANCHORED_BIPARTITE_REF_4L:
        return Wave72ExperimentSpec(
            variant_mode=mode, num_layers=4,
            enable_grouped_4qubit_segre_prep=False, enable_multibranch_kron_readout=False, seed=seed,
        )
    if mode == Wave72VariantMode.W72_Q1_GROUPED_4QUBIT_SEGRE_PREP_4L:
        return Wave72ExperimentSpec(
            variant_mode=mode, num_layers=4,
            enable_grouped_4qubit_segre_prep=True, enable_multibranch_kron_readout=False, seed=seed,
        )
    if mode == Wave72VariantMode.W72_Q2_GROUPED_4QUBIT_MULTIBRANCH_KRON_4L:
        return Wave72ExperimentSpec(
            variant_mode=mode, num_layers=4,
            enable_grouped_4qubit_segre_prep=True, enable_multibranch_kron_readout=True, seed=seed,
        )
    return Wave72ExperimentSpec(
        variant_mode=mode, num_layers=8,
        enable_grouped_4qubit_segre_prep=True, enable_multibranch_kron_readout=True, seed=seed,
    )


def run_wave72_variant(
    spec: Wave72ExperimentSpec,
    stream: LongSeqFineWebStream,
    steps: int = 200,
    batch_size: int = 8,
    seq_len: int = 512,
    eval_interval: int = 25,
) -> dict[str, Any]:
    """Train and evaluate one `Wave72VariantMode` run with `StandardizedExperimentLogger` (`<= 8` args)."""
    device = stream.device
    torch.manual_seed(spec.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(spec.seed)
        gc.collect()
        torch.cuda.empty_cache()

    model = GroupedFourQubitKroneckerDensityGPT(spec).to(device)
    init_ms = model.initialize_quantum_state(seed=spec.seed)
    total_params = sum(p.numel() for p in model.parameters())
    trunk_opt, vocab_opt, scalar_opt = build_wave69_optimizers(model)
    retr_cad = 8 if steps >= 1000 else 5
    thermostat = ReadOnlyQuenchRiemannianShadowEMA(
        model, ema_decay=DEFAULT_THERMOSTAT_EMA_DECAY, eval_fn=evaluate_wave69_model, retract_cadence=retr_cad
    )

    tokens_per_step = batch_size * seq_len
    run_key = f"plan092_{spec.variant_mode.value.lower()}_{steps}steps"
    exp_logger = StandardizedExperimentLogger(
        ExperimentIntentSpec(
            run_key=run_key,
            wave_id="W72",
            plan_id="EXPERIMENT_PLAN_092",
            task_domain="fineweb_speedrun",
            hypothesis_intent=(
                f"Wave 72 {spec.variant_mode.value} (L={spec.num_layers}, "
                f"segre_4qubit={spec.enable_grouped_4qubit_segre_prep}, "
                f"kron_readout={spec.enable_multibranch_kron_readout})."
            ),
            architecture_summary=f"GroupedFourQubitKroneckerDensityGPT(L={spec.num_layers}, params={total_params:,})",
            hardware_budget=str(device),
            batch_size=batch_size,
            seq_len=seq_len,
            total_steps=steps,
            target_tokens_M=round((steps * tokens_per_step) / 1e6, 3),
        )
    )

    step0_ev = evaluate_wave69_model(model, stream)
    step0_ev["step"], step0_ev["tokens_M"] = 0, 0.0
    exp_logger.log_eval(0, {**step0_ev, "top1_acc": round(100.0 * step0_ev["top1_acc"], 2), "top5_acc": round(100.0 * step0_ev["top5_acc"], 2)})
    print(f"[{spec.variant_mode.value} (init={init_ms:.1f}ms, L={spec.num_layers}, params={total_params:,})] Step    0 | Val CE: {step0_ev['val_ce']:.4f}", flush=True)

    curve: list[dict[str, Any]] = [step0_ev]
    t0 = time.perf_counter()
    model.train()

    for step in range(1, steps + 1):
        for opt in (trunk_opt, vocab_opt, scalar_opt):
            opt.zero_grad(set_to_none=True)
        idx_in, targets = stream.sample_train_batch(step)
        loss = model(idx_in, targets=targets, return_metrics=False)
        loss.backward()

        lr_mult, mom_beta = compute_restored_riemannian_momentum_schedule(step, steps)
        is_eval_step = (step % eval_interval == 0) or (step == steps) or (step == 1)
        trunk_opt.record_diagnostics = is_eval_step
        trunk_opt.set_momentum(mom_beta)
        trunk_opt.set_lr_scale(lr_mult)
        trunk_opt.step()

        vocab_opt.set_momentum(mom_beta)
        vocab_opt.step_active(_collect_active_vocab_indices_w68(idx_in, targets), lr_scale=lr_mult)
        scalar_opt.set_momentum(0.0 if mom_beta <= 0.0 else min(0.90, mom_beta))
        scalar_opt.step_active(active_idx=None, lr_scale=lr_mult)
        model.restrict_scalars_()
        model.advance_thermostat_tokens_(tokens_per_step)
        thermostat.update_after_step(model)

        if step % 10 == 0 or step == 1:
            el_s = max(1e-3, time.perf_counter() - t0)
            exp_logger.log_step(step, train_ce=float(loss.detach().item()), tok_per_s=(float(step * tokens_per_step) / el_s), lr_mult=lr_mult)

        if step % eval_interval == 0 or step == steps:
            th_out = thermostat.check_and_thermostat(model, stream, current_lr_scale=1.0, step=step, apply_feedback=True)
            gd = {k: round(float(v), 6) for k, v in trunk_opt.last_gauge_diagnostics.items()}
            ev = dict(th_out["active_eval"])
            ev["step"], ev["tokens_M"], ev["train_ce"] = step, round((step * tokens_per_step) / 1e6, 3), round(float(loss.detach().item()), 4)
            ev["hot_val_ce"], ev["ema_val_ce"], ev["thermal_gap_ce"] = th_out["hot_val_ce"], th_out["ema_val_ce"], th_out["thermal_gap_ce"]
            ev["thermostat_triggered"], ev["thermostat_lr_scale"], ev["mom_beta"] = False, round(lr_mult, 4), round(mom_beta, 4)
            ev.update(gd)
            curve.append(ev)
            exp_logger.log_eval(step, {**ev, "top1_acc": round(100.0 * ev["top1_acc"], 2), "top5_acc": round(100.0 * ev["top5_acc"], 2)})
            print(
                f"[{spec.variant_mode.value}] Step {step:5d} ({ev['tokens_M']:6.2f}M tok, lr={lr_mult:.3f}x, mom={mom_beta:.3f}) | "
                f"Hot CE: {th_out['hot_val_ce']:.4f} | EMA CE: {th_out['ema_val_ce']:.4f} "
                f"(gap={th_out['thermal_gap_ce']:+.4f}) | Top1: {100.0 * ev['top1_acc']:.2f}% | "
                f"MUBErr: {ev['mub_unbiased_err']:.2e} | SU16Phase: {gd.get('su16_step_phase_frac', 0.0):.2e} | "
                f"KrausGauge: {gd.get('kraus_step_gauge_frac', 0.0):.2e}",
                flush=True,
            )

    wall_s = max(1e-6, time.perf_counter() - t0)
    exp_logger.finalize()
    summary = {
        "variant": spec.variant_mode.value, "num_layers": spec.num_layers,
        "total_params": total_params, "total_params_M": round(total_params / 1e6, 3),
        "init_ms": round(init_ms, 2), "elapsed_sec": round(wall_s, 2),
        "ms_per_step": round((wall_s * 1000.0) / max(1, steps), 2),
        "tok_per_s": round((steps * tokens_per_step) / wall_s, 1),
        "step0": step0_ev, "final": curve[-1], "curve": curve,
    }
    del model, thermostat, vocab_opt, trunk_opt, scalar_opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary
