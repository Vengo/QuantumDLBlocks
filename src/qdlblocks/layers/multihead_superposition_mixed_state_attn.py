"""Multi-Head Quantum Superposition & Stinespring Mixed-State Attention (`Wave 44` / `EXPERIMENT_PLAN_059.md`).

When to Use This Building Block:
- Use `MultiHeadSuperpositionMixedStateAttention`, `MultiHeadSuperpositionAttnConfig`,
  `causal_born_p2_complex_attn`, `compute_normalized_cross_head_gram`,
  `compute_cross_head_hermitian_gram_step`, `apply_stinespring_subhead_to_branch_coupling`,
  and `compute_subhead_purity_telemetry` when scaling pure-complex Rank-`R` `16^4 = 65,536`
  Kronecker wavefunction models to sequence lengths `T in {512, 1024}` with:
  1. Single-pass `3`-matmul causal `p=2` Born attention (`c_re`, `c_im`, `w_born @ v_cat`)
     across `H_total = M * K` sub-heads of register dimension `d_f = 16` (`M = 4, K in {2, 4}`),
     eliminating the `8` extra intra-head `(T x T)` Krylov power-iteration matmuls per layer
     (`11 -> 3` `(T x T)` GEMMs, `11 -> 5` total `torch.matmul` calls).
  2. Exact trace-1 `K x K` cross-head Hermitian Gram density matrix
     `G_hat_{t,m} = (1 / K) Phi_{t,m} Phi_{t,m}^H in C^{K x K}` (`Tr(G_hat_{t,m}) = 1`,
     `G_hat_{t,m} >= 0`) and `U(1)`-gauge-invariant Gram transport
     `tilde_Phi_{t,m} = Phi_{t,m} + eta_{G,m} (G_hat_{t,m} Phi_{t,m})` in `O(T M K^2 d_f)` time
     with zero `T^2` overhead, zero ad-hoc `1.5*rho - 0.5*rho^2` polynomial, and zero ad-hoc
     `gamma` purity gate in the default `"multihead_gram"` / `"stinespring_gram"` mode.
  3. Semi-unitary Stinespring coupling `W_m in C^{K x R}` (`W_m^H W_m = I_R`) mapping the `K`
     contextual sub-heads per Kronecker factor `m` into `R` coherent polysemy branch kets
     `|psi_{t,r,m}^attn> in C^{16}` with observable constructive/destructive branch interference
     energy `nu_{t,r}` and exact Schmidt cross-head purity `Tr(G_hat_{t,m}^2) in [1/K, 1]`.

Why It Was Built & Mathematical Provenance (`EXPERIMENT_PLAN_059.md` / `Wave 44` Track 6):
- Resolves Issue 6 (`11` intra-head `(T x T)` Krylov GEMMs per layer bottlenecking `T = 512 -> 1024`
  scaling and collapsing intra-head contextual modes) and replaces single-stream ad-hoc polynomial
  and `gamma` purity gates with first-principles Stinespring/Kraus quantum mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from ..kernels.flash_complex_krylov_triton import (
    TRITON_COMPLEX_MAX_DIM,
    TRITON_COMPLEX_MIN_DIM,
    factored_complex_krylov_step,
    fused_causal_pboson_born_triton,
    fused_complex_norm_rope_triton,
)
from ..kernels.fused_kronecker_mixed_state_kernels import (
    HAS_FUSED_KRONECKER_TRITON,
    TRITON_FLASH_MIN_SEQ_LEN,
    flash_causal_pboson_complex_attn_triton,
    fused_cross_head_gram_stinespring_step,
)
from .packed_complex_register_density_attn import (
    ComplexLinear,
    ComplexRoPE,
    complex_l2_normalize,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# Synchronized with EXPERIMENT_PLAN_059.md & Track 6 Specification
# ==============================================================================
DEFAULT_NUM_MEANINGS: int = 2
DEFAULT_NUM_KRONECKER_FACTORS: int = 4
DEFAULT_SUBHEADS_PER_FACTOR: int = 2
DEFAULT_FACTOR_HEAD_DIM: int = 16
DEFAULT_MAX_SEQ_LEN: int = 1024
DEFAULT_P_BOSON_POWER: int = 2
DEFAULT_GRAM_COUPLING_ETA_INIT: float = 0.15
DEFAULT_ATTN_EVOLUTION_MODE: str = "multihead_gram"
DEFAULT_ATTN_VARIANT: str = "stinespring_gram"
VALID_ATTN_EVOLUTION_MODES: tuple[str, ...] = (
    "multihead_gram",
    "stinespring_gram",
    "legacy_krylov_11mm",
    "legacy_krylov",
    "adhoc_poly_gamma",
)
VALID_ATTN_VARIANTS: tuple[str, ...] = VALID_ATTN_EVOLUTION_MODES
ADHOC_POLY_COEFF_LIN: float = 1.5
ADHOC_POLY_COEFF_QUAD: float = 0.5
DEFAULT_ADHOC_PURITY_GAMMA_INIT: float = 0.25
SUBHEAD_KET_NORM_EPS: float = 1e-6
INIT_V0_SUBHEAD_MIX_LOGIT: float = -1.1
ALGEBRAIC_GATE_HALF: float = 0.5
ALGEBRAIC_GATE_ONE: float = 1.0
STINESPRING_SEED_BASE: int = 3000
STINESPRING_SEED_LAYER_STRIDE: int = 31


@dataclass(frozen=True)
class MultiHeadSuperpositionAttnConfig:
    """Frozen configuration for `MultiHeadSuperpositionMixedStateAttention` (`Wave 44` Track 6)."""

    num_meanings: int = DEFAULT_NUM_MEANINGS
    num_factors: int = DEFAULT_NUM_KRONECKER_FACTORS
    d_register: int = DEFAULT_FACTOR_HEAD_DIM
    n_subheads_per_factor: int = DEFAULT_SUBHEADS_PER_FACTOR
    attn_evolution_mode: str = DEFAULT_ATTN_EVOLUTION_MODE
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    boson_p: int = DEFAULT_P_BOSON_POWER
    gram_eta_init: float = DEFAULT_GRAM_COUPLING_ETA_INIT
    value_residual_mixing: bool = True
    use_fused_kernels: bool = True
    subheads_per_factor: int | None = None
    factor_dim: int | None = None
    attn_variant: str | None = None

    def __post_init__(self) -> None:
        eff_k = int(
            self.subheads_per_factor
            if self.subheads_per_factor is not None
            else self.n_subheads_per_factor
        )
        eff_df = int(self.factor_dim if self.factor_dim is not None else self.d_register)
        eff_mode = str(
            self.attn_variant if self.attn_variant is not None else self.attn_evolution_mode
        )
        if eff_mode not in VALID_ATTN_EVOLUTION_MODES:
            raise ValueError(
                f"Unknown attn_evolution_mode {eff_mode!r}; expected one of {VALID_ATTN_EVOLUTION_MODES}"
            )
        if eff_k < 1 or eff_df < 2 or int(self.num_meanings) < 1 or int(self.num_factors) < 1:
            raise ValueError(
                f"Invalid MultiHeadSuperpositionAttnConfig dimensions: R={self.num_meanings}, "
                f"M={self.num_factors}, K={eff_k}, d_register={eff_df}"
            )
        object.__setattr__(self, "n_subheads_per_factor", eff_k)
        object.__setattr__(self, "subheads_per_factor", eff_k)
        object.__setattr__(self, "d_register", eff_df)
        object.__setattr__(self, "factor_dim", eff_df)
        object.__setattr__(self, "attn_evolution_mode", eff_mode)
        object.__setattr__(self, "attn_variant", eff_mode)

    @property
    def d_model(self) -> int:
        """Total complex residual dimension `R * M * d_register = 64 * R`."""
        return int(self.num_meanings) * int(self.num_factors) * int(self.d_register)

    @property
    def n_subheads_total(self) -> int:
        """Total complex attention sub-heads `M * K`."""
        return int(self.num_factors) * int(self.n_subheads_per_factor)

    @property
    def d_attn(self) -> int:
        """Total complex attention projection width `M * K * d_register = 64 * K`."""
        return self.n_subheads_total * int(self.d_register)


MultiHeadSuperpositionAttentionConfig = MultiHeadSuperpositionAttnConfig


def _algebraic_root_sigmoid(u: torch.Tensor) -> torch.Tensor:
    """Evaluate 100% `exp`/`sigmoid`-free algebraic gate `0.5 * (1 + u / sqrt(1 + u^2)) in (0, 1)`."""
    uf = u.float()
    return ALGEBRAIC_GATE_HALF * (
        ALGEBRAIC_GATE_ONE + uf * torch.rsqrt(ALGEBRAIC_GATE_ONE + uf * uf)
    )


def causal_born_p2_complex_attn(
    q_re: torch.Tensor,
    q_im: torch.Tensor,
    k_re: torch.Tensor,
    k_im: torch.Tensor,
    v_re: torch.Tensor,
    v_im: torch.Tensor,
    p_boson: int = DEFAULT_P_BOSON_POWER,
    eps: float = SUBHEAD_KET_NORM_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute single-pass `3`-matmul causal `p`-boson Born attention on `(B, H, T, d_h)` (`<= 8` args).

    Replaces the `11` `(T x T)` intra-head Krylov power-iteration matmuls of
    `factored_complex_krylov_step` with exactly `3` `(T x T)` matmuls (`c_re`, `c_im`, `vb_cat`),
    returning unit-normalized contextual sub-head kets `(phi_re, phi_im)` on `CS^{d_h - 1}`.
    """
    dh = v_re.size(-1)
    q_cat = torch.cat([q_re, q_im], dim=-1)
    k_cat = torch.cat([k_re, k_im], dim=-1)
    k_skew = torch.cat([-k_im, k_re], dim=-1)
    v_cat = torch.cat([v_re, v_im], dim=-1)

    # Matmul #1 and Matmul #2: Hermitian inner products c_{t,s} = <q_t | k_s>_C
    c_re = torch.matmul(q_cat, k_cat.transpose(-2, -1))
    c_im = torch.matmul(q_cat, k_skew.transpose(-2, -1))

    # Causal p-boson Born weights w_{t,s} = |c_{t,s}|^{2p} I(s <= t) / Z_t
    w_born = fused_causal_pboson_born_triton(
        c_re, c_im, p_boson=int(p_boson), eps=float(eps)
    ).to(dtype=v_cat.dtype)

    # Matmul #3: First-moment value ket aggregation
    vb_cat = torch.matmul(w_born, v_cat)
    vb_re = vb_cat[..., :dh]
    vb_im = vb_cat[..., dh:]
    return complex_l2_normalize(vb_re, vb_im, dim=-1, eps=float(eps))


def compute_normalized_cross_head_gram(
    phi_re: torch.Tensor,
    phi_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute trace-1 `K x K` cross-head Hermitian Gram density `G_hat = (1/K) Phi Phi^H` (`<= 8` args).

    Uses `2` packed matmuls (`phi_cat @ phi_cat^T` and `phi_cat @ phi_skew^T`) so that
    `(G_{t,m})_{k,j} = sum_{d=0}^{d_f-1} phi_{t,m,k,d} * conj(phi_{t,m,j,d})`.
    By the Schmidt decomposition, `G_hat_{t,m} in C^{K x K}` and the `(d_f x d_f)` outer density
    matrix `rho_{t,m} = (1/K) sum_k |phi_{t,m,k}><phi_{t,m,k}|` share the exact same non-zero
    eigenvalues and Bloch purity `Tr(G_hat_{t,m}^2) = ||G_hat_{t,m}||_F^2 in [1/K, 1]`.
    """
    k_sub = phi_re.size(-2)
    inv_k = 1.0 / float(max(1, k_sub))
    phi_cat = torch.cat([phi_re, phi_im], dim=-1)
    phi_skew = torch.cat([-phi_im, phi_re], dim=-1)
    g_hat_re = torch.matmul(phi_cat, phi_cat.transpose(-2, -1)) * inv_k
    g_hat_im = torch.matmul(phi_cat, phi_skew.transpose(-2, -1)) * inv_k
    return g_hat_re, g_hat_im


def compute_cross_head_hermitian_gram_step(
    phi_re: torch.Tensor,
    phi_im: torch.Tensor,
    gram_eta: torch.Tensor,
    attn_evolution_mode: str = DEFAULT_ATTN_EVOLUTION_MODE,
    purity_gamma: torch.Tensor | None = None,
    attn_variant: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply `U(1)`-covariant cross-head Hermitian Gram coupling across `K` sub-heads (`<= 8` args).

    Args:
        phi_re: Real part of unit-normalized sub-head kets of shape `(B, T, M, K, d_f)`.
        phi_im: Imaginary part of unit-normalized sub-head kets of shape `(B, T, M, K, d_f)`.
        gram_eta: Learnable per-factor Gram coupling vector of shape `(M,)`.
        attn_evolution_mode: Evolution mode in `VALID_ATTN_EVOLUTION_MODES`.
        purity_gamma: Optional `(M,)` gate logits used only when `mode == "adhoc_poly_gamma"`.
        attn_variant: Optional alias overriding `attn_evolution_mode`.

    Returns:
        Tuple `(tilde_re, tilde_im, purity_tm)` of shapes `((B, T, M, K, d_f), (B, T, M, K, d_f), (B, T, M))`.
    """
    mode = str(attn_variant if attn_variant is not None else attn_evolution_mode)
    if mode not in VALID_ATTN_EVOLUTION_MODES:
        raise ValueError(
            f"Unknown attn_evolution_mode {mode!r}; expected one of {VALID_ATTN_EVOLUTION_MODES}"
        )

    g_hat_re, g_hat_im = compute_normalized_cross_head_gram(phi_re, phi_im)

    if mode == "adhoc_poly_gamma":
        gh_r = g_hat_re.float()
        gh_i = g_hat_im.float()
        g2_re = torch.einsum("...kj,...jl->...kl", gh_r, gh_r) - torch.einsum(
            "...kj,...jl->...kl", gh_i, gh_i
        )
        g2_im = torch.einsum("...kj,...jl->...kl", gh_r, gh_i) + torch.einsum(
            "...kj,...jl->...kl", gh_i, gh_r
        )
        poly_re = ADHOC_POLY_COEFF_LIN * gh_r - ADHOC_POLY_COEFF_QUAD * g2_re
        poly_im = ADHOC_POLY_COEFF_LIN * gh_i - ADHOC_POLY_COEFF_QUAD * g2_im
        if purity_gamma is None:
            gam = torch.full(
                (phi_re.size(-3),),
                DEFAULT_ADHOC_PURITY_GAMMA_INIT,
                device=phi_re.device,
                dtype=torch.float32,
            )
        else:
            gam = _algebraic_root_sigmoid(purity_gamma)
        gam_view = gam.view(*([1] * (phi_re.ndim - 3)), -1, 1, 1)
        g_eff_re = ((1.0 - gam_view) * gh_r + gam_view * poly_re).to(dtype=phi_re.dtype)
        g_eff_im = ((1.0 - gam_view) * gh_i + gam_view * poly_im).to(dtype=phi_im.dtype)
    else:
        g_eff_re, g_eff_im = g_hat_re, g_hat_im

    mix_re = torch.einsum("...kj,...jd->...kd", g_eff_re, phi_re) - torch.einsum(
        "...kj,...jd->...kd", g_eff_im, phi_im
    )
    mix_im = torch.einsum("...kj,...jd->...kd", g_eff_re, phi_im) + torch.einsum(
        "...kj,...jd->...kd", g_eff_im, phi_re
    )
    eta = gram_eta.to(dtype=phi_re.dtype).view(*([1] * (phi_re.ndim - 3)), -1, 1, 1)
    tilde_re = phi_re + eta * mix_re
    tilde_im = phi_im + eta * mix_im

    purity_tm = (g_hat_re.float().square() + g_hat_im.float().square()).sum(dim=(-2, -1))
    return tilde_re, tilde_im, purity_tm


def apply_stinespring_subhead_to_branch_coupling(
    tilde_re: torch.Tensor,
    tilde_im: torch.Tensor,
    stinespring_w_re: torch.Tensor,
    stinespring_w_im: torch.Tensor,
    eps: float = SUBHEAD_KET_NORM_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project `K` Gram-coupled sub-heads per factor `m` into `R` coherent polysemy branches (`<= 8` args).

    Evaluates `|psi_{t,r,m}^attn> = sum_{k=0}^{K-1} W_{m,k,r} |tilde_phi_{t,m,k}> in C^{d_f}`,
    records the pre-normalization constructive/destructive interference energy
    `nu_{t,r} = mean_m(||psi_{t,r,m}^attn||_2^2)` of shape `(B, T, R)`, and returns unit-normalized
    packed branch states `(psi_norm_re, psi_norm_im, nu_tr)` of shapes
    `((B, T, R * M * d_f), (B, T, R * M * d_f), (B, T, R))`.
    """
    w_re = stinespring_w_re.to(dtype=tilde_re.dtype)
    w_im = stinespring_w_im.to(dtype=tilde_im.dtype)
    psi_re = torch.einsum("btmkd,mkr->btrmd", tilde_re, w_re) - torch.einsum(
        "btmkd,mkr->btrmd", tilde_im, w_im
    )
    psi_im = torch.einsum("btmkd,mkr->btrmd", tilde_re, w_im) + torch.einsum(
        "btmkd,mkr->btrmd", tilde_im, w_re
    )

    factor_sq_norm = (psi_re.float().square() + psi_im.float().square()).sum(dim=-1)
    nu_tr = factor_sq_norm.mean(dim=-1)

    inv_norm = torch.rsqrt(factor_sq_norm.unsqueeze(-1) + float(eps)).to(dtype=psi_re.dtype)
    norm_re = (psi_re * inv_norm).flatten(start_dim=2)
    norm_im = (psi_im * inv_norm).flatten(start_dim=2)
    return norm_re, norm_im, nu_tr


@torch.no_grad()
def compute_subhead_purity_telemetry(
    phi_re: torch.Tensor,
    phi_im: torch.Tensor,
    branch_nu: torch.Tensor | None = None,
    eps: float = SUBHEAD_KET_NORM_EPS,
) -> dict[str, float]:
    """Compute exact quantum Gram trace, Bloch purity, Rényi-2 entropy, and effective rank (`<= 8` args).

    Verifies that the normalized cross-head Gram operator `G_hat_{t,m} = (1/K) Phi_{t,m} Phi_{t,m}^H`
    and the reduced `(d_f x d_f)` density matrix `rho_{t,m} = (1/K) sum_k |phi_{t,m,k}><phi_{t,m,k}|`
    have unit trace (`Tr(G_hat) == 1`) and purity `P in [1 / min(K, d_f), 1]`.
    """
    pr_n, pi_n = complex_l2_normalize(phi_re.float(), phi_im.float(), dim=-1, eps=float(eps))
    g_hat_re, g_hat_im = compute_normalized_cross_head_gram(pr_n, pi_n)

    trace_vals = torch.diagonal(g_hat_re, dim1=-2, dim2=-1).sum(dim=-1)
    trace_mean = float(trace_vals.mean().item())
    trace_err = float((trace_vals - 1.0).abs().max().item())

    purity_vals = (g_hat_re.square() + g_hat_im.square()).sum(dim=(-2, -1)).clamp(min=float(eps), max=1.0)
    purity_mean = float(purity_vals.mean().item())
    purity_min = float(purity_vals.min().item())
    purity_max = float(purity_vals.max().item())

    renyi2_vals = -torch.log(purity_vals)
    renyi2_mean = float(renyi2_vals.mean().item())
    eff_rank_mean = float((1.0 / purity_vals).mean().item())
    nu_mean = float(branch_nu.float().mean().item()) if branch_nu is not None else 1.0

    return {
        "subhead_gram_purity_mean": purity_mean,
        "head_purity_mean": purity_mean,
        "purity_mean": purity_mean,
        "reduced_density_purity_mean": purity_mean,
        "purity_min": purity_min,
        "purity_max": purity_max,
        "subhead_gram_trace_mean": trace_mean,
        "gram_trace_err": trace_err,
        "subhead_renyi2_entropy": renyi2_mean,
        "renyi2_entropy_nats": renyi2_mean,
        "subhead_effective_rank": eff_rank_mean,
        "mixed_rank_mean": eff_rank_mean,
        "branch_interference_nu_mean": nu_mean,
    }


class MultiHeadSuperpositionMixedStateAttention(nn.Module):
    """Multi-Head Quantum Superposition & Stinespring Mixed-State Attention (`Wave 44` Track 6).

    Supports all three `attn_evolution_mode` / `attn_variant` arms:
    - `"multihead_gram"` / `"stinespring_gram"` (Default Flagship): Single-pass `3`-matmul causal `p=2`
      Born attention + trace-1 `K x K` cross-head Hermitian Gram coupling + Stinespring semi-unitary
      subhead-to-branch projection (`zero` intra-head Krylov matmuls, `zero` ad-hoc `gamma` gate).
    - `"adhoc_poly_gamma"`: Single-pass `3`-matmul causal Born attention + degree-2 Gram polynomial
      `1.5*G_hat - 0.5*G_hat^2` blended with scalar `purity_gamma`.
    - `"legacy_krylov_11mm"` / `"legacy_krylov"`: Legacy `11`-matmul intra-head 2-step associative
      Krylov rollout (`factored_complex_krylov_step(use_vkrylov="v_krylov_norm")`).
    """

    def __init__(
        self,
        cfg: MultiHeadSuperpositionAttnConfig | None = None,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else MultiHeadSuperpositionAttnConfig()
        self.layer_idx = int(layer_idx)
        self.num_meanings = int(self.cfg.num_meanings)
        self.num_factors = int(self.cfg.num_factors)
        self.dh = int(self.cfg.d_register)
        self.subheads_per_factor = int(self.cfg.n_subheads_per_factor)
        self.H = int(self.cfg.n_subheads_total)
        self.d_complex = int(self.cfg.d_model)
        self.d_attn = int(self.cfg.d_attn)
        self.p_boson = int(self.cfg.boson_p)
        self.attn_evolution_mode = str(self.cfg.attn_evolution_mode)
        self.value_residual_mixing = bool(self.cfg.value_residual_mixing and (self.layer_idx > 0))
        self.use_fused_kernels: bool = bool(self.cfg.use_fused_kernels)

        self.rope = ComplexRoPE(d_register=self.dh)
        self.w_q = ComplexLinear(self.d_complex, self.d_attn)
        self.w_k = ComplexLinear(self.d_complex, self.d_attn)
        self.w_v = ComplexLinear(self.d_complex, self.d_attn)
        self.w_out = ComplexLinear(self.d_complex, self.d_complex)

        self.gram_eta = nn.Parameter(
            torch.full(
                (self.num_factors,),
                float(self.cfg.gram_eta_init),
                dtype=torch.float32,
            )
        )
        self.stinespring_w_re = nn.Parameter(
            torch.empty(self.num_factors, self.subheads_per_factor, self.num_meanings)
        )
        self.stinespring_w_im = nn.Parameter(
            torch.empty(self.num_factors, self.subheads_per_factor, self.num_meanings)
        )

        if self.attn_evolution_mode == "adhoc_poly_gamma":
            self.purity_gamma: nn.Parameter | None = nn.Parameter(
                torch.full(
                    (self.num_factors,),
                    DEFAULT_ADHOC_PURITY_GAMMA_INIT,
                    dtype=torch.float32,
                )
            )
        else:
            self.register_parameter("purity_gamma", None)

        if self.value_residual_mixing:
            self.v_mix_logit: nn.Parameter | None = nn.Parameter(
                torch.tensor(INIT_V0_SUBHEAD_MIX_LOGIT, dtype=torch.float32)
            )
        else:
            self.register_parameter("v_mix_logit", None)

        self.last_purity_mean: float = 1.0 / float(max(1, self.subheads_per_factor))
        self.last_mixed_rank_mean: float = float(self.subheads_per_factor)
        self.last_renyi2_entropy_mean: float = math.log(float(max(1, self.subheads_per_factor)))
        self.last_gram_trace_err: float = 0.0
        self.last_branch_nu_mean: float = 1.0
        self._last_purity_tm: torch.Tensor | None = None
        self._last_nu_tr: torch.Tensor | None = None
        self._telemetry_dirty: bool = False
        self._last_telemetry_dict: dict[str, float] = {
            "subhead_gram_purity_mean": self.last_purity_mean,
            "head_purity_mean": self.last_purity_mean,
            "purity_mean": self.last_purity_mean,
            "reduced_density_purity_mean": self.last_purity_mean,
            "purity_min": self.last_purity_mean,
            "purity_max": self.last_purity_mean,
            "subhead_gram_trace_mean": 1.0,
            "gram_trace_err": 0.0,
            "subhead_renyi2_entropy": self.last_renyi2_entropy_mean,
            "renyi2_entropy_nats": self.last_renyi2_entropy_mean,
            "subhead_effective_rank": self.last_mixed_rank_mean,
            "mixed_rank_mean": self.last_mixed_rank_mean,
            "branch_interference_nu_mean": 1.0,
        }

        self.reset_stinespring_parameters()

    @property
    def W_q(self) -> ComplexLinear:
        """Alias for `self.w_q` matching `PackedComplexRegisterDensityAttention`."""
        return self.w_q

    @property
    def W_k(self) -> ComplexLinear:
        """Alias for `self.w_k` matching `PackedComplexRegisterDensityAttention`."""
        return self.w_k

    @property
    def W_v(self) -> ComplexLinear:
        """Alias for `self.w_v` matching `PackedComplexRegisterDensityAttention`."""
        return self.w_v

    @property
    def W_out(self) -> ComplexLinear:
        """Alias for `self.w_out` matching `PackedComplexRegisterDensityAttention`."""
        return self.w_out

    @torch.no_grad()
    def reset_stinespring_parameters(self, seed_offset: int = 0) -> None:
        """Initialize `stinespring_w_re/im` `(M, K, R)` as complex semi-unitary matrices via QR (`<= 8` args)."""
        dev = self.stinespring_w_re.device
        k_sub = self.subheads_per_factor
        r_mean = self.num_meanings
        rows = max(k_sub, r_mean)
        cols = min(k_sub, r_mean)

        for m in range(self.num_factors):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(
                STINESPRING_SEED_BASE
                + STINESPRING_SEED_LAYER_STRIDE * self.layer_idx
                + 7 * int(seed_offset)
                + m
            )
            z = torch.complex(
                torch.randn((rows, cols), generator=gen, dtype=torch.float32),
                torch.randn((rows, cols), generator=gen, dtype=torch.float32),
            ).to(device=dev)
            q_mat, r_mat = torch.linalg.qr(z)
            diag_r = torch.diagonal(r_mat, 0)
            phase = diag_r / diag_r.abs().clamp_min(SUBHEAD_KET_NORM_EPS)
            q_mat = q_mat * phase.unsqueeze(0)
            if k_sub < r_mean:
                q_mat = q_mat.t().contiguous()
            self.stinespring_w_re.data[m].copy_(q_mat.real.to(dtype=self.stinespring_w_re.dtype))
            self.stinespring_w_im.data[m].copy_(q_mat.imag.to(dtype=self.stinespring_w_im.dtype))

    @torch.no_grad()
    def _sync_cached_telemetry(self, eps: float = SUBHEAD_KET_NORM_EPS) -> None:
        """Lazily synchronize cached GPU purity & interference tensors to CPU floats (`<= 8` args)."""
        if not self._telemetry_dirty or self._last_purity_tm is None or self._last_nu_tr is None:
            return
        p_clamped = self._last_purity_tm.float().clamp(min=float(eps), max=1.0)
        p_mean = float(p_clamped.mean().item())
        r_mean = float((1.0 / p_clamped).mean().item())
        s2_mean = float((-torch.log(p_clamped)).mean().item())
        nu_mean = float(self._last_nu_tr.float().mean().item())
        self.last_purity_mean = p_mean
        self.last_mixed_rank_mean = r_mean
        self.last_renyi2_entropy_mean = s2_mean
        self.last_gram_trace_err = 0.0
        self.last_branch_nu_mean = nu_mean
        self._last_telemetry_dict = {
            "subhead_gram_purity_mean": p_mean,
            "head_purity_mean": p_mean,
            "purity_mean": p_mean,
            "reduced_density_purity_mean": p_mean,
            "purity_min": float(p_clamped.min().item()),
            "purity_max": float(p_clamped.max().item()),
            "subhead_gram_trace_mean": 1.0,
            "gram_trace_err": 0.0,
            "subhead_renyi2_entropy": s2_mean,
            "renyi2_entropy_nats": s2_mean,
            "subhead_effective_rank": r_mean,
            "mixed_rank_mean": r_mean,
            "branch_interference_nu_mean": nu_mean,
        }
        self._telemetry_dirty = False

    @torch.no_grad()
    def compute_subhead_purity_telemetry(
        self,
        phi_re: torch.Tensor | None = None,
        phi_im: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Return cross-head Gram purity, trace error, Rényi-2 entropy, and effective rank (`<= 8` args)."""
        if phi_re is not None and phi_im is not None:
            return compute_subhead_purity_telemetry(phi_re, phi_im, eps=SUBHEAD_KET_NORM_EPS)
        self._sync_cached_telemetry()
        return dict(self._last_telemetry_dict)

    def _normalize_and_apply_rope(
        self,
        q_re: torch.Tensor,
        q_im: torch.Tensor,
        k_re: torch.Tensor,
        k_im: torch.Tensor,
        v_re: torch.Tensor,
        v_im: torch.Tensor,
        eps: float = SUBHEAD_KET_NORM_EPS,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unit-normalize `Q, K, V` on `CS^{d_h - 1}` and apply `ComplexRoPE` strictly to `Q, K` (`<= 8` args)."""
        seq_t = q_re.size(2)
        if TRITON_COMPLEX_MIN_DIM <= self.dh <= TRITON_COMPLEX_MAX_DIM and q_re.is_cuda:
            cos_2d, sin_2d = self.rope.get_cos_sin(seq_t, q_re.device, torch.float32)
            q_rot_re, q_rot_im = fused_complex_norm_rope_triton(
                q_re, q_im, cos_2d, sin_2d, eps=float(eps)
            )
            k_rot_re, k_rot_im = fused_complex_norm_rope_triton(
                k_re, k_im, cos_2d, sin_2d, eps=float(eps)
            )
            v_norm_re, v_norm_im = fused_complex_norm_rope_triton(
                v_re, v_im, None, None, eps=float(eps)
            )
            return q_rot_re, q_rot_im, k_rot_re, k_rot_im, v_norm_re, v_norm_im

        q_n_re, q_n_im = complex_l2_normalize(q_re, q_im, dim=-1, eps=float(eps))
        k_n_re, k_n_im = complex_l2_normalize(k_re, k_im, dim=-1, eps=float(eps))
        v_norm_re, v_norm_im = complex_l2_normalize(v_re, v_im, dim=-1, eps=float(eps))
        q_rot_re, q_rot_im = self.rope(q_n_re, q_n_im)
        k_rot_re, k_rot_im = self.rope(k_n_re, k_n_im)
        return q_rot_re, q_rot_im, k_rot_re, k_rot_im, v_norm_re, v_norm_im

    def forward_complex(
        self,
        psi_re: torch.Tensor,
        psi_im: torch.Tensor,
        v0_re: torch.Tensor | None = None,
        v0_im: torch.Tensor | None = None,
        eps: float = SUBHEAD_KET_NORM_EPS,
        return_telemetry: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
    ):
        """Run multi-head Stinespring superposition attention (`<= 8` args)."""
        b_sz, seq_t, _ = psi_re.shape
        m_fac, k_sub, dh = self.num_factors, self.subheads_per_factor, self.dh
        h_tot = self.H

        if v0_re is not None and v0_re.ndim == 2:
            v0_re, v0_im = None, None

        q_re, q_im = self.w_q(psi_re, psi_im)
        k_re, k_im = self.w_k(psi_re, psi_im)
        v_re, v_im = self.w_v(psi_re, psi_im)

        q_re = q_re.view(b_sz, seq_t, h_tot, dh).transpose(1, 2).contiguous()
        q_im = q_im.view(b_sz, seq_t, h_tot, dh).transpose(1, 2).contiguous()
        k_re = k_re.view(b_sz, seq_t, h_tot, dh).transpose(1, 2).contiguous()
        k_im = k_im.view(b_sz, seq_t, h_tot, dh).transpose(1, 2).contiguous()
        v_re = v_re.view(b_sz, seq_t, h_tot, dh).transpose(1, 2).contiguous()
        v_im = v_im.view(b_sz, seq_t, h_tot, dh).transpose(1, 2).contiguous()

        v_raw_re, v_raw_im = v_re, v_im
        if (
            self.value_residual_mixing
            and self.v_mix_logit is not None
            and v0_re is not None
            and v0_im is not None
        ):
            alpha = _algebraic_root_sigmoid(self.v_mix_logit).to(dtype=v_re.dtype)
            v0_r = v0_re.view(b_sz, h_tot, seq_t, dh).to(dtype=v_re.dtype)
            v0_i = v0_im.view(b_sz, h_tot, seq_t, dh).to(dtype=v_im.dtype)
            v_re = (1.0 - alpha) * v_re + alpha * v0_r
            v_im = (1.0 - alpha) * v_im + alpha * v0_i

        q_rot_re, q_rot_im, k_rot_re, k_rot_im, v_n_re, v_n_im = self._normalize_and_apply_rope(
            q_re, q_im, k_re, k_im, v_re, v_im, eps=float(eps)
        )

        use_fused = bool(
            self.use_fused_kernels and psi_re.is_cuda and not torch.compiler.is_compiling()
        )
        if self.attn_evolution_mode in ("legacy_krylov_11mm", "legacy_krylov"):
            phi_h_re, phi_h_im, _, _ = factored_complex_krylov_step(
                q_rot_re,
                q_rot_im,
                k_rot_re,
                k_rot_im,
                v_n_re,
                v_n_im,
                use_vkrylov="v_krylov_norm",
                eps=float(eps),
            )
        elif (
            use_fused
            and HAS_FUSED_KRONECKER_TRITON
            and self.p_boson == 2
            and seq_t >= TRITON_FLASH_MIN_SEQ_LEN
        ):
            phi_h_re, phi_h_im = flash_causal_pboson_complex_attn_triton(
                q_rot_re,
                q_rot_im,
                k_rot_re,
                k_rot_im,
                v_n_re,
                v_n_im,
                eps=float(eps),
            )
        else:
            phi_h_re, phi_h_im = causal_born_p2_complex_attn(
                q_rot_re,
                q_rot_im,
                k_rot_re,
                k_rot_im,
                v_n_re,
                v_n_im,
                p_boson=self.p_boson,
                eps=float(eps),
            )

        phi_re = phi_h_re.transpose(1, 2).contiguous().view(b_sz, seq_t, m_fac, k_sub, dh)
        phi_im = phi_h_im.transpose(1, 2).contiguous().view(b_sz, seq_t, m_fac, k_sub, dh)

        if use_fused and self.attn_evolution_mode != "adhoc_poly_gamma":
            branch_re, branch_im, purity_tm, nu_tr = fused_cross_head_gram_stinespring_step(
                phi_re,
                phi_im,
                self.gram_eta,
                self.stinespring_w_re,
                self.stinespring_w_im,
                float(eps),
            )
        else:
            tilde_re, tilde_im, purity_tm = compute_cross_head_hermitian_gram_step(
                phi_re,
                phi_im,
                self.gram_eta,
                attn_evolution_mode=self.attn_evolution_mode,
                purity_gamma=self.purity_gamma,
            )
            branch_re, branch_im, nu_tr = apply_stinespring_subhead_to_branch_coupling(
                tilde_re,
                tilde_im,
                self.stinespring_w_re,
                self.stinespring_w_im,
                eps=float(eps),
            )

        if not torch.compiler.is_compiling():
            self._last_purity_tm = purity_tm.detach()
            self._last_nu_tr = nu_tr.detach()
            self._telemetry_dirty = True

        out_re, out_im = self.w_out(branch_re, branch_im)
        if return_telemetry:
            return out_re, out_im, v_raw_re, v_raw_im, {"head_purity": purity_tm, "branch_nu": nu_tr}
        return out_re, out_im, v_raw_re, v_raw_im

    def forward(
        self,
        psi_re: torch.Tensor,
        psi_im: torch.Tensor,
        v0_re: torch.Tensor | None = None,
        v0_im: torch.Tensor | None = None,
        eps: float = SUBHEAD_KET_NORM_EPS,
        return_telemetry: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
    ):
        """Delegate directly to `forward_complex` (`<= 8` args)."""
        return self.forward_complex(
            psi_re,
            psi_im,
            v0_re=v0_re,
            v0_im=v0_im,
            eps=eps,
            return_telemetry=return_telemetry,
        )


MultiHeadStinespringSuperpositionAttention = MultiHeadSuperpositionMixedStateAttention


def count_attention_matmul_calls(
    attn_layer: MultiHeadSuperpositionMixedStateAttention,
    psi_re: torch.Tensor,
    psi_im: torch.Tensor,
) -> tuple[int, int]:
    """Count total `torch.matmul` calls and `(T x T)` `torch.matmul` calls in one forward pass (`<= 8` args)."""
    from unittest.mock import patch

    orig_matmul = torch.matmul
    total_calls = 0
    txt_calls = 0
    seq_len = psi_re.size(1)

    def _counting_matmul(a: torch.Tensor, b: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        nonlocal total_calls, txt_calls
        total_calls += 1
        if a.ndim >= 2 and b.ndim >= 2:
            if (a.size(-2) == seq_len and a.size(-1) == seq_len) or (
                a.size(-2) == seq_len and b.size(-1) == seq_len
            ):
                txt_calls += 1
        return orig_matmul(a, b, *args, **kwargs)

    prev_fused = getattr(attn_layer, "use_fused_kernels", False)
    attn_layer.use_fused_kernels = False
    try:
        with torch.no_grad(), patch("torch.matmul", side_effect=_counting_matmul):
            attn_layer(psi_re, psi_im)
    finally:
        attn_layer.use_fused_kernels = prev_fused

    return total_calls, txt_calls


__all__ = [
    "ADHOC_POLY_COEFF_LIN",
    "ADHOC_POLY_COEFF_QUAD",
    "DEFAULT_ADHOC_PURITY_GAMMA_INIT",
    "DEFAULT_ATTN_EVOLUTION_MODE",
    "DEFAULT_ATTN_VARIANT",
    "DEFAULT_FACTOR_HEAD_DIM",
    "DEFAULT_GRAM_COUPLING_ETA_INIT",
    "DEFAULT_MAX_SEQ_LEN",
    "DEFAULT_NUM_KRONECKER_FACTORS",
    "DEFAULT_NUM_MEANINGS",
    "DEFAULT_P_BOSON_POWER",
    "DEFAULT_SUBHEADS_PER_FACTOR",
    "SUBHEAD_KET_NORM_EPS",
    "VALID_ATTN_EVOLUTION_MODES",
    "VALID_ATTN_VARIANTS",
    "MultiHeadStinespringSuperpositionAttention",
    "MultiHeadSuperpositionAttentionConfig",
    "MultiHeadSuperpositionAttnConfig",
    "MultiHeadSuperpositionMixedStateAttention",
    "apply_stinespring_subhead_to_branch_coupling",
    "causal_born_p2_complex_attn",
    "compute_cross_head_hermitian_gram_step",
    "compute_normalized_cross_head_gram",
    "compute_subhead_purity_telemetry",
    "count_attention_matmul_calls",
]
